from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
import json, os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


def maybe_writer(tb_dir: str | None) -> SummaryWriter | None:
    return SummaryWriter(tb_dir) if tb_dir else None


@dataclass
class PPOBatch:
    obs: torch.Tensor
    h: int
    actions: torch.Tensor
    action_mask: Optional[torch.Tensor]
    old_logprobs: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    old_values: torch.Tensor
    anchor_action_logits: Optional[torch.Tensor] = None
    raw_states: Optional[List] = (
        None  # Store raw states for re-featurization when featurizer is trainable
    )


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    T = rewards.shape[0]
    adv = torch.zeros(T, dtype=torch.float32)
    lastgaelam = 0.0
    for t in reversed(range(T)):
        nextnonterminal = 1.0 - (dones[t] if t < T - 1 else 0.0)
        nextvalue = values[t + 1] if t < T - 1 else 0.0
        delta = rewards[t] + gamma * nextvalue * nextnonterminal - values[t]
        lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
        adv[t] = lastgaelam
    returns = adv + values
    return adv, returns


def _safe_log_softmax_lines(line_logits: torch.Tensor) -> torch.Tensor:
    """
    line_logits: (B, A, h). Some rows may be all -inf (globally illegal action).
    Returns log-probs with no NaNs: rows that are all -inf are set to zeros.
    """
    # Regular log_softmax first
    logp = torch.log_softmax(line_logits, dim=-1)
    # Detect rows where all entries were -inf
    all_neginf = torch.isneginf(line_logits).all(dim=-1, keepdim=True)  # (B, A, 1)
    # Replace those rows with zeros to avoid NaNs propagating
    logp = torch.where(all_neginf.expand_as(logp), torch.zeros_like(logp), logp)
    return logp


def action_logprob_from_logits(action_logits, line_logits, actions):
    B, A = action_logits.shape
    logp_a = torch.log_softmax(action_logits, dim=-1)  # (B, A)
    logp_lines = _safe_log_softmax_lines(line_logits)  # (B, A, h)
    a_idx = actions[:, 0].long()
    line_idx = actions[:, 1].long()
    line_idx_clamped = torch.clamp(line_idx, min=0)
    # If line_idx is -1, we ignore the line term by multiplying with 0
    lp = (
        logp_a[torch.arange(B), a_idx]
        + logp_lines[torch.arange(B), a_idx, line_idx_clamped] * (line_idx >= 0).float()
    )
    return lp


class BehaviorCloningTrainer:
    def __init__(
        self, policy, featurizer=None, lr=1e-3, device="cpu", tb_dir: str | None = None
    ):
        self.policy = policy.to(device)
        self.featurizer = featurizer
        params = list(policy.parameters())
        if featurizer is not None:
            params += featurizer.trainable_params
        self.opt = optim.Adam(params, lr=lr)
        self.device = device
        self.writer = maybe_writer(tb_dir)
        self.global_step = 0

    def step(
        self,
        obs,
        h,
        action_targets,
        line_targets,
        action_mask=None,
        value_targets=None,
        goal_targets=None,
    ):
        self.policy.train()
        # Only detach obs if featurizer is not trainable
        if self.featurizer is not None:
            if self.featurizer.trainable:
                obs = obs.to(self.device)  # Keep gradients for trainable featurizer
        else:
            obs = obs.to(self.device).detach()  # Detach for non-trainable featurizer
        action_targets = action_targets.to(self.device).long()
        line_targets = line_targets.to(self.device).long()
        if action_mask is not None:
            action_mask = action_mask.to(self.device)
        if value_targets is not None:
            value_targets = value_targets.to(self.device).float()
        if goal_targets is not None:
            goal_targets = goal_targets.to(self.device).float()
        if self.featurizer:
            self.featurizer.train(self.featurizer.trainable)

        # Compute Loss, acc, etc.
        out = self.policy(obs, h, action_mask)
        a_loss = F.cross_entropy(out.action_logits, action_targets)
        B = line_targets.shape[0]
        valid = line_targets >= 0
        if valid.any():
            chosen = out.line_logits[torch.arange(B), action_targets, :]
            l_loss = F.cross_entropy(chosen[valid], line_targets[valid])
        else:
            l_loss = torch.tensor(0.0, device=self.device)
        v_loss = torch.tensor(0.0, device=self.device)
        if value_targets is not None:
            v_loss = F.mse_loss(out.value.squeeze(-1), value_targets)
        g_loss = torch.tensor(0.0, device=self.device)
        if goal_targets is not None:
            g_loss = (
                F.mse_loss(out.goal_logits, goal_targets)
                if goal_targets.ndim > 1
                else F.cross_entropy(out.goal_logits, goal_targets.long())
            )
        loss = a_loss + l_loss + 0.5 * v_loss + 0.2 * g_loss
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.opt.step()

        # Log and return
        if self.writer:
            self.writer.add_scalar("bc/loss", float(loss.item()), self.global_step)
            self.writer.add_scalar(
                "bc/action_loss", float(a_loss.item()), self.global_step
            )
            self.writer.add_scalar(
                "bc/line_loss", float(l_loss.item()), self.global_step
            )
            self.global_step += 1
        return {
            "loss": float(loss.item()),
            "action_loss": float(a_loss.item()),
            "line_loss": float(
                l_loss.item() if isinstance(l_loss, torch.Tensor) else l_loss
            ),
        }


class PPOTrainer:
    def __init__(
        self, policy, cfg, featurizer=None, device="cpu", tb_dir: str | None = None
    ):
        self.policy = policy.to(device)
        self.featurizer = featurizer
        params = list(policy.parameters())
        if featurizer is not None:
            params += featurizer.trainable_params
        self.opt = optim.Adam(params, lr=cfg.lr)
        self.cfg = cfg
        self.device = device
        self.writer = maybe_writer(tb_dir)
        self.global_step = 0

    def ppo_update(self, batch: PPOBatch):
        actions = batch.actions.to(self.device)
        old_lp = batch.old_logprobs.to(self.device).detach()
        rets = batch.returns.to(self.device).detach()
        adv = batch.advantages.to(self.device).detach()
        old_v = batch.old_values.to(self.device).detach()
        mask = (
            batch.action_mask.to(self.device) if batch.action_mask is not None else None
        )
        anchor = (
            batch.anchor_action_logits.to(self.device).detach()
            if batch.anchor_action_logits is not None
            else None
        )

        # PPO loop
        self.policy.train()
        if self.featurizer:
            self.featurizer.train(self.featurizer.trainable)
        for _ in range(self.cfg.epochs):
            # Handle observations: re-featurize if featurizer is trainable, otherwise use pre-computed
            if (
                self.featurizer is not None
                and self.featurizer.trainable
                and batch.raw_states is not None
            ):
                # Re-featurize raw states with gradients enabled (fresh graph each epoch)
                obs_list = []
                for state in batch.raw_states:
                    x, _, _ = self.featurizer.featurize(state, agent="A")
                    obs_list.append(x.squeeze(0))
                obs = torch.stack(obs_list, dim=0).to(self.device)
            else:
                # Use pre-computed observations (detached for non-trainable featurizer)
                obs = batch.obs.to(self.device).detach()
            out = self.policy(obs, batch.h, mask)
            new_lp = action_logprob_from_logits(
                out.action_logits, out.line_logits, actions
            )
            ratio = torch.exp(new_lp - old_lp)

            # PPO clipped objective
            pg1 = -adv * ratio
            pg2 = -adv * torch.clamp(
                ratio, 1.0 - self.cfg.clip_coef, 1.0 + self.cfg.clip_coef
            )
            policy_loss = torch.max(pg1, pg2).mean()

            # Entropy from probabilities (avoid 0 * -inf)
            pa = F.softmax(out.action_logits, dim=-1)
            ent = -(pa * torch.log(pa + 1e-8)).sum(dim=-1).mean()

            # Value loss (clipped)
            v = out.value.squeeze(-1)
            v_unclip = (v - rets) ** 2
            v_clip = old_v + torch.clamp(
                v - old_v, -self.cfg.clip_coef, self.cfg.clip_coef
            )
            v_clip = (v_clip - rets) ** 2
            v_loss = 0.5 * torch.max(v_unclip, v_clip).mean()

            # piKL anchor (optional)
            pikl_loss = torch.tensor(0.0, device=self.device)
            if anchor is not None and self.cfg.pikl_beta > 0.0:
                pi_log = F.log_softmax(out.action_logits, dim=-1)
                anc = F.softmax(anchor, dim=-1)
                pikl = (anc * (torch.log(anc + 1e-8) - pi_log)).sum(dim=-1).mean()
                pikl_loss = self.cfg.pikl_beta * pikl

            loss = (
                policy_loss
                - self.cfg.ent_coef * ent
                + self.cfg.vf_coef * v_loss
                + pikl_loss
            )
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.max_grad_norm)
            self.opt.step()

        # Log and return
        with torch.no_grad():
            approx_kl = (old_lp - new_lp).mean().item()
            clip_frac = (
                (torch.abs(ratio - 1.0) > self.cfg.clip_coef).float().mean().item()
            )
        if self.writer:
            self.writer.add_scalar("ppo/loss", float(loss.item()), self.global_step)
            self.writer.add_scalar(
                "ppo/policy_loss", float(policy_loss.item()), self.global_step
            )
            self.writer.add_scalar(
                "ppo/value_loss", float(v_loss.item()), self.global_step
            )
            self.writer.add_scalar("ppo/entropy", float(ent.item()), self.global_step)
            self.writer.add_scalar("ppo/approx_kl", approx_kl, self.global_step)
            self.writer.add_scalar("ppo/clip_frac", clip_frac, self.global_step)
            self.global_step += 1
        return {
            "loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(v_loss.item()),
            "entropy": float(ent.item()),
            "approx_kl": approx_kl,
            "clip_frac": clip_frac,
        }


# ---- Logging & Checkpointing ----#
class JsonlLogger:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, "a", encoding="utf-8")

    def log(self, obj: Dict[str, Any]):
        self.f.write(json.dumps(obj) + "\n")
        self.f.flush()

    def close(self):
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def __del__(self):
        try:
            if not self.f.closed:
                self.f.close()
        except Exception:
            pass


class Checkpointer:
    def __init__(self, out_dir: str, pv_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.pv_dir = pv_dir

    def save(self, assistant, human, epoch: int):
        self.save_assistant(assistant, epoch)
        self.save_human(human, epoch)

    def save_assistant(self, assistant, epoch: int):
        torch.save(
            assistant.state_dict(),
            os.path.join(self.out_dir, f"assistant_policy_e{epoch}.pt"),
        )

    def save_human(self, human, epoch: int):
        torch.save(
            human.state_dict(), os.path.join(self.out_dir, f"human_policy_e{epoch}.pt")
        )

    def persist_final_models(
        self, assistant, human, featurizer, model_cfg, featurizer_cfg
    ):
        # Save assistant model with config
        assistant_checkpoint = {
            "model_state_dict": assistant.state_dict(),
            "config": model_cfg.__dict__,
        }
        torch.save(
            assistant_checkpoint, os.path.join(self.pv_dir, f"asm_assistant_model.pt")
        )

        # Save human model with config
        human_checkpoint = {
            "model_state_dict": human.state_dict(),
            "config": model_cfg.__dict__,
        }
        torch.save(human_checkpoint, os.path.join(self.pv_dir, f"asm_human_model.pt"))

        # Save featurizer with config
        featurizer_checkpoint = {
            "featurizer_state_dict": featurizer.state_dict(),
            "config": featurizer_cfg.__dict__,
        }
        torch.save(
            featurizer_checkpoint, os.path.join(self.pv_dir, f"asm_featurizer.pt")
        )
