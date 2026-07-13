"""
rl_finetune.py — RL Policy Fine-Tuning (Phase 3 of three-phase curriculum).

Implements ``RLPolicyFinetuning`` — REINFORCE-with-baseline trainer that
tunes only the top transformer layers + heads, using signed trade-quality
rewards with entropy bonus and auxiliary supervised loss.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class RLPolicyFinetuning:
    """Phase 3: REINFORCE with baseline for trade-policy optimisation.

    Freezes bottom transformer layers, fine-tunes only top layers + heads.
    Action space: {HOLD=0, BUY=1, SELL=2}.
    Reward: signed trade-quality.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Any,
        device: Optional[torch.device] = None,
        num_actions: int = 3,
        value_head_dim: int = 64,
        entropy_coef: float = 0.01,
        aux_loss_coef: float = 0.1,
        discount_gamma: float = 0.99,
        lr: float = 1e-5,
        grad_clip_norm: float = 1.0,
        freeze_bottom_layers: int = 2,
    ) -> None:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model = model.to(device)
        self.config = config

        self.num_actions = num_actions
        self.entropy_coef = entropy_coef
        self.aux_loss_coef = aux_loss_coef
        self.discount_gamma = discount_gamma
        self.grad_clip_norm = grad_clip_norm
        self.freeze_bottom_layers = freeze_bottom_layers

        d_model = getattr(config, "d_model", getattr(config, "hidden_dim", 128))

        # Value head: small MLP
        self.value_head = nn.Sequential(
            nn.Linear(d_model, value_head_dim),
            nn.GELU(),
            nn.Linear(value_head_dim, 1),
        ).to(device)

        self.optimizer = torch.optim.AdamW(
            self._trainable_parameters(), lr=lr, weight_decay=1e-4,
        )

        self.global_step: int = 0
        self._episode_rewards: List[float] = []
        self._policy_losses: List[float] = []
        self._value_losses: List[float] = []
        self._frozen_params: set = set()

    # ------------------------------------------------------------------
    # Freeze / unfreeze
    # ------------------------------------------------------------------

    def freeze_encoder(self) -> None:
        """Freeze bottom transformer layers."""
        transformer_layers = self._find_transformer_layers()
        if transformer_layers is None or len(transformer_layers) == 0:
            logger.warning("Could not locate transformer layers to freeze.")
            return

        total_layers = len(transformer_layers)
        freeze_count = min(self.freeze_bottom_layers, total_layers)

        for i, layer in enumerate(transformer_layers):
            if i < freeze_count:
                for param in layer.parameters():
                    param.requires_grad = False
                    self._frozen_params.add(id(param))
            else:
                for param in layer.parameters():
                    param.requires_grad = True

        logger.info(
            "Froze %d/%d bottom transformer layers; top %d are trainable.",
            freeze_count, total_layers, total_layers - freeze_count,
        )

    def unfreeze_all(self) -> None:
        self._frozen_params.clear()
        for param in self.model.parameters():
            param.requires_grad = True
        for param in self.value_head.parameters():
            param.requires_grad = True
        logger.info("All parameters unfrozen.")

    def _find_transformer_layers(self) -> Optional[nn.ModuleList]:
        """Walk model tree to locate transformer encoder layers."""
        for attr_name in ("layers", "encoder", "transformer", "blocks"):
            candidate = getattr(self.model, attr_name, None)
            if candidate is None:
                continue
            if isinstance(candidate, nn.ModuleList):
                return candidate
            sublayers = getattr(candidate, "layers", None)
            if isinstance(sublayers, nn.ModuleList):
                return sublayers

        for _name, module in self.model.named_modules():
            if isinstance(module, nn.ModuleList) and len(module) >= 2:
                first_child = module[0]
                if any(isinstance(c, (nn.MultiheadAttention, nn.TransformerEncoderLayer))
                       for c in first_child.modules()):
                    return module
        return None

    def _trainable_parameters(self):
        for param in self.model.parameters():
            if param.requires_grad:
                yield param
        for param in self.value_head.parameters():
            yield param

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train_trajectories(
        self,
        trajectories: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Train on collected trajectories using REINFORCE.

        Each trajectory dict:
            states:      Tensor (T, seq_len, n_features)
            actions:     Tensor (T,)
            rewards:     Tensor (T,)
            log_probs:   Tensor (T,)
            aux_targets: Optional dict of supervised targets
        """
        self.model.train()
        self.value_head.train()

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_reward = 0.0
        n_steps = 0

        for traj in trajectories:
            states = traj["states"].to(self.device)
            actions = traj["actions"].to(self.device)
            rewards = traj["rewards"].to(self.device)
            log_probs = traj["log_probs"].to(self.device)
            aux_targets = traj.get("aux_targets", None)

            T = len(actions)

            # ---- Compute returns ----
            returns = self._compute_returns(rewards)

            # ---- Encode for value estimation ----
            encoded = self._encode(states)

            # ---- Value estimate and advantage ----
            values = self.value_head(encoded).squeeze(-1)
            advantages = returns - values.detach()

            # ---- Policy loss ----
            policy_loss = -(log_probs * advantages).mean()

            # ---- Value loss ----
            value_loss = F.mse_loss(values, returns)

            # ---- Entropy bonus ----
            with torch.no_grad():
                output = self.model(states)
                action_logits = self._get_action_logits(output)
                probs = F.softmax(action_logits, dim=-1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).mean()
                total_entropy += entropy.item()

            # ---- Auxiliary supervised loss ----
            aux_loss = torch.tensor(0.0, device=self.device)
            if aux_targets is not None and self.aux_loss_coef > 0:
                from Learn.v2.training.losses import composite_loss
                output = self.model(states)
                aux_total, _ = composite_loss(output, aux_targets)
                aux_loss = aux_total

            # ---- Combined loss ----
            total_loss = (
                policy_loss
                + value_loss
                - self.entropy_coef * entropy
                + self.aux_loss_coef * aux_loss
            )

            # ---- Backward ----
            self.optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(
                self._trainable_parameters(), self.grad_clip_norm,
            )
            self.optimizer.step()

            self.global_step += 1
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_reward += rewards.sum().item()
            n_steps += T

        n_trajs = max(len(trajectories), 1)
        avg_reward = total_reward / max(n_steps, 1)
        avg_policy_loss = total_policy_loss / n_trajs
        avg_value_loss = total_value_loss / n_trajs
        avg_entropy = total_entropy / n_trajs

        self._episode_rewards.append(avg_reward)
        self._policy_losses.append(avg_policy_loss)
        self._value_losses.append(avg_value_loss)

        logger.info(
            "RL step=%d: reward=%.4f, policy_loss=%.6f, value_loss=%.6f, entropy=%.4f",
            self.global_step, avg_reward, avg_policy_loss, avg_value_loss, avg_entropy,
        )

        return {
            "avg_reward": avg_reward,
            "policy_loss": avg_policy_loss,
            "value_loss": avg_value_loss,
            "entropy": avg_entropy,
        }

    # ------------------------------------------------------------------
    # Roll-out helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample_action(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[int, float, torch.Tensor]:
        """Sample an action from the policy.

        Returns: (action_idx, log_prob, action_logits)
        """
        self.model.eval()
        state = state.to(self.device)
        output = self.model(state)
        logits = self._get_action_logits(output)
        probs = F.softmax(logits, dim=-1)

        if deterministic:
            action = int(probs.argmax(dim=-1).item())
        else:
            dist = torch.distributions.Categorical(probs)
            action = int(dist.sample().item())

        log_prob = float(torch.log(probs[0, action] + 1e-10).item())
        return action, log_prob, logits.cpu()

    @torch.no_grad()
    def estimate_value(self, state: torch.Tensor) -> float:
        """Return value estimate V(s) for a single state."""
        self.value_head.eval()
        state = state.to(self.device)
        encoded = self._encode(state)
        value = self.value_head(encoded)
        return float(value.squeeze().item())

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "value_head_state_dict": self.value_head.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "episode_rewards": self._episode_rewards,
            "policy_losses": self._policy_losses,
            "value_losses": self._value_losses,
        }
        torch.save(checkpoint, path)
        logger.info("RL checkpoint saved to '%s'", path)

    def load_checkpoint(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.value_head.load_state_dict(checkpoint["value_head_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint.get("global_step", 0)
        self._episode_rewards = checkpoint.get("episode_rewards", [])
        self._policy_losses = checkpoint.get("policy_losses", [])
        self._value_losses = checkpoint.get("value_losses", [])
        logger.info("RL checkpoint loaded from '%s' (step=%d)", path, self.global_step)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_returns(self, rewards: torch.Tensor) -> torch.Tensor:
        """Discounted cumulative returns with normalisation."""
        T = len(rewards)
        returns = torch.zeros(T, device=rewards.device, dtype=rewards.dtype)
        running = torch.tensor(0.0, device=rewards.device, dtype=rewards.dtype)
        for t in range(T - 1, -1, -1):
            running = rewards[t] + self.discount_gamma * running
            returns[t] = running
        if returns.std() > 1e-8:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Get a pooled state representation for the value head."""
        if hasattr(self.model, "encode"):
            encoded = self.model.encode(x)
        else:
            output = self.model(x)
            if hasattr(output, "hidden_states") and output.hidden_states is not None:
                encoded = output.hidden_states
            elif hasattr(output, "distribution") and hasattr(output.distribution, "mu"):
                encoded = output.distribution.mu
            else:
                raise RuntimeError(
                    "Cannot extract pooled encoder state. Model must expose "
                    "an encode method or hidden_states attribute."
                )

        if encoded.dim() == 3:
            encoded = encoded.mean(dim=1)
        elif encoded.dim() == 4:
            encoded = encoded.mean(dim=(1, 2))
        return encoded

    def _get_action_logits(self, output: Any) -> torch.Tensor:
        """Extract action logits from model output.

        Priority:
        1. output.action_logits
        2. output.direction → expand to 3-class
        3. output.distribution.mu → heuristic
        """
        if hasattr(output, "action_logits") and output.action_logits is not None:
            return output.action_logits

        if hasattr(output, "direction") and output.direction is not None:
            dir_logits = output.direction
            dir_score = dir_logits.mean(dim=-1).mean(dim=-1, keepdim=True)
            sell_logits = -dir_score * 0.5
            buy_logits = dir_score * 0.5
            hold_logits = torch.zeros_like(dir_score)
            return torch.cat([sell_logits, hold_logits, buy_logits], dim=-1)

        if hasattr(output, "distribution") and output.distribution is not None:
            mu = output.distribution.mu
            dir_score = mu.mean(dim=-1, keepdim=True)
            sell_logits = -dir_score * 0.5
            buy_logits = dir_score * 0.5
            hold_logits = torch.zeros_like(dir_score)
            return torch.cat([sell_logits, hold_logits, buy_logits], dim=-1)

        B = getattr(
            getattr(output, "distribution", None), "mu", torch.zeros(1, 1)
        ).shape[0]
        return torch.zeros(B, self.num_actions, device=self.device)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def episode_rewards(self) -> List[float]:
        return self._episode_rewards

    @property
    def policy_losses(self) -> List[float]:
        return self._policy_losses
