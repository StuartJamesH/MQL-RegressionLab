import numpy as np
import torch
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