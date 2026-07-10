import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

def split(X, y, shuffle, test_size):
    idx = np.arange(X.shape[0]).reshape(-1,1)
    X_idx = np.hstack([idx, X])

    X_train, X_test, y_train, y_test = train_test_split(X_idx, y, shuffle=shuffle, test_size=test_size)

    train_idx = X_train[:,0]
    test_idx = X_test[:,0]

    return X_train[:,1:], X_test[:,1:], y_train, y_test, train_idx, test_idx

def conservative_predict(logits, trade_min_prob=0.6, gap=0.15):
    probs = torch.softmax(logits, dim=-1)
    p0, p1, p2 = probs[:,0], probs[:,1], probs[:,2]
    preds = torch.ones(len(probs), dtype=torch.long)  # default FLAT
    # only set to 2 when p2 > trade_min_prob and p2 > p1 + gap
    preds[(p2 > trade_min_prob) & (p2 > p1 + gap)] = 2
    preds[(p0 > trade_min_prob) & (p0 > p1 + gap)] = 0
    return preds

def generate_trade_signals(long_quality: torch.Tensor, short_quality: torch.Tensor, threshold: float = 0.5) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate discrete trade signals from dual-head regression outputs.
    
    Returns:
        signals: Tensor of predicted classes (0 = SELL, 1 = FLAT, 2 = BUY)
        signal_strength: Tensor of the maximum quality score (max(long, short))
    """
    signal_strength = torch.max(long_quality, short_quality)
    signals = torch.ones_like(long_quality, dtype=torch.long)  # Default to FLAT (1)

    # If long is stronger and above threshold, BUY (2)
    buy_mask = (long_quality > threshold) & (long_quality > short_quality)
    # If short is stronger and above threshold, SELL (0)
    sell_mask = (short_quality > threshold) & (short_quality > long_quality)

    signals[buy_mask] = 2
    signals[sell_mask] = 0

    return signals, signal_strength


class TwoHeadMLP(nn.Module):
    """
    Small shared-trunk MLP with two heads, mirroring the LGBM two-head design
    (classifier + resolved-only magnitude regressor) for one side (long or short):
      - classifier_head: 3-class logits over {SL=0, TP=1, timeout=2}
      - regressor_head: scalar magnitude prediction (log1p(MFE/MAE)-style quality),
        only meaningful/trained on resolved (TP or SL) rows.
    """

    def __init__(self, n_features, hidden_sizes=(128, 64), n_classes=3):
        super().__init__()
        layers = []
        in_size = n_features
        for h in hidden_sizes:
            layers += [nn.Linear(in_size, h), nn.ReLU()]
            in_size = h
        self.trunk = nn.Sequential(*layers)
        self.classifier_head = nn.Linear(in_size, n_classes)
        self.regressor_head = nn.Linear(in_size, 1)

    def forward(self, x):
        z = self.trunk(x)
        class_logits = self.classifier_head(z)
        reg_pred = self.regressor_head(z).squeeze(-1)
        return class_logits, reg_pred


def train_two_head_mlp(
    X_train,
    y_class_train,
    y_reg_train,
    resolved_mask_train,
    hidden_sizes=(128, 64),
    epochs=60,
    lr=1e-3,
    batch_size=4096,
    device=None,
    verbose=True,
):
    """
    Train a TwoHeadMLP with a combined loss: CrossEntropyLoss on the 3-class
    win/loss/timeout target (all rows) plus MSELoss on the magnitude target,
    computed only over resolved rows within each batch (skipped entirely for a
    batch with zero resolved rows).

    y_class_train: int array, 0=SL, 1=TP, 2=timeout.
    y_reg_train: float array, magnitude target (only meaningful where resolved_mask_train
        is True; values elsewhere are ignored by the masked loss).
    resolved_mask_train: bool array, True where the row is a resolved (TP/SL) trade.

    Returns the trained model in eval mode.
    """
    device = device or torch.device("cpu")
    model = TwoHeadMLP(X_train.shape[1], hidden_sizes=hidden_sizes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()

    X_t = torch.from_numpy(np.asarray(X_train, dtype=np.float32))
    y_class_t = torch.from_numpy(np.asarray(y_class_train, dtype=np.int64))
    y_reg_t = torch.from_numpy(np.asarray(y_reg_train, dtype=np.float32))
    resolved_t = torch.from_numpy(np.asarray(resolved_mask_train, dtype=bool))

    n = X_t.shape[0]
    model.train()
    for epoch in range(int(epochs)):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb = X_t[idx].to(device)
            yb_class = y_class_t[idx].to(device)
            yb_reg = y_reg_t[idx].to(device)
            yb_resolved = resolved_t[idx].to(device)

            optimizer.zero_grad()
            class_logits, reg_pred = model(xb)
            loss = ce_loss(class_logits, yb_class)
            if yb_resolved.any():
                loss = loss + mse_loss(reg_pred[yb_resolved], yb_reg[yb_resolved])
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            n_batches += 1

        if verbose and (epoch == 0 or (epoch + 1) % max(1, epochs // 5) == 0 or epoch == epochs - 1):
            print(f"  MLP epoch {epoch + 1}/{epochs}: avg loss={epoch_loss / max(n_batches, 1):.4f}")

    model.eval()
    return model


def predict_two_head_mlp(model, X, device=None):
    """
    Run inference with a trained TwoHeadMLP.

    Returns (class_proba, reg_pred): class_proba is an (n, 3) array of softmax
    probabilities over {SL, TP, timeout}; reg_pred is an (n,) array of magnitude
    predictions.
    """
    device = device or torch.device("cpu")
    model.eval()
    with torch.no_grad():
        X_t = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(device)
        class_logits, reg_pred = model(X_t)
        proba = torch.softmax(class_logits, dim=-1).cpu().numpy()
        reg_pred = reg_pred.cpu().numpy()
    return proba, reg_pred