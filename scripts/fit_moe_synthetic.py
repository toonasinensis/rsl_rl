#!/usr/bin/env python3
# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# SPDX-License-Identifier: BSD-3-Clause

"""Fit BackboneMoE on randomly generated (obs, target) pairs.

Synthetic targets are produced by a fixed mixture of teacher MLPs with a
region-based router, so the MoE backbone has a learnable regression problem.
Use this script to sanity-check that forward, backward, and aux losses work.

Example:
    python scripts/fit_moe_synthetic.py --epochs 500 --num-samples 4096
"""

from __future__ import annotations

import argparse
import random

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.models import BackboneMoE
from rsl_rl.modules import MLP


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_teacher(
    obs_dim: int,
    output_dim: int,
    num_experts: int,
    hidden_dims: list[int],
    device: torch.device,
) -> tuple[list[MLP], torch.Tensor]:
    """Fixed teacher experts and region boundaries along obs[:, 0]."""
    experts = [
        MLP(obs_dim, output_dim, hidden_dims, activation="elu").to(device)
        for _ in range(num_experts)
    ]
    for expert in experts:
        expert.eval()
        for p in expert.parameters():
            p.requires_grad = False

    # num_experts - 1 boundaries split the first observation dimension
    bounds = torch.linspace(-1.5, 1.5, num_experts - 1, device=device) if num_experts > 1 else torch.tensor([], device=device)
    return experts, bounds


@torch.no_grad()
def teacher_router_index(x: torch.Tensor, bounds: torch.Tensor, num_experts: int) -> torch.Tensor:
    """Assign each sample to expert id in [0, num_experts) using x[:, 0]."""
    if num_experts == 1:
        return torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
    idx = torch.bucketize(x[:, 0].contiguous(), bounds)
    return idx.clamp(max=num_experts - 1)


@torch.no_grad()
def generate_dataset(
    num_samples: int,
    obs_dim: int,
    output_dim: int,
    num_experts: int,
    hidden_dims: list[int],
    noise_std: float,
    device: torch.device,
    seed: int,
) -> tuple[TensorDict, torch.Tensor, list[MLP], torch.Tensor]:
    """Return (obs TensorDict, targets, teacher experts, teacher bounds)."""
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    x = torch.randn(num_samples, obs_dim, device=device, generator=gen)
    teachers, bounds = build_teacher(obs_dim, output_dim, num_experts, hidden_dims, device)

    expert_ids = teacher_router_index(x, bounds, num_experts)
    targets = torch.empty(num_samples, output_dim, device=device)
    for i in range(num_experts):
        mask = expert_ids == i
        if mask.any():
            targets[mask] = teachers[i](x[mask])

    if noise_std > 0.0:
        noise = torch.randn(targets.shape, device=device, generator=gen)
        targets = targets + noise_std * noise

    obs = TensorDict({"policy": x}, batch_size=[num_samples], device=device)
    return obs, targets, teachers, bounds


def make_moe_backbone(
    obs: TensorDict,
    output_dim: int,
    num_experts: int,
    top_k: int,
    hidden_dims: list[int],
    load_balance_loss_weight: float,
    router_z_loss_weight: float,
    router_only: bool,
) -> BackboneMoE:
    obs_groups = {"actor": ["policy"]}
    return BackboneMoE(
        obs,
        obs_groups,
        "actor",
        output_dim,
        hidden_dims=hidden_dims,
        expert_hidden_dims=hidden_dims,
        router_hidden_dims=hidden_dims,
        activation="elu",
        num_experts=num_experts,
        top_k=top_k,
        router_temperature=1.0,
        load_balance_loss_weight=load_balance_loss_weight,
        router_z_loss_weight=router_z_loss_weight,
        router_only=router_only,
    )


def compute_loss(
    model: BackboneMoE,
    obs: TensorDict,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """MSE on actions plus MoE aux losses (same pattern as MoEPPO)."""
    out = model(obs, train_mode=True)
    pred = out["actions"]
    loss = nn.functional.mse_loss(pred, targets)

    metrics: dict[str, float] = {"mse": loss.item()}
    aux = out.get("aux_losses") or {}
    for name, aux_loss in aux.items():
        loss = loss + aux_loss
        metrics[name] = aux_loss.item()
    metrics["total"] = loss.item()
    return loss, metrics


def train(
    model: BackboneMoE,
    obs: TensorDict,
    targets: torch.Tensor,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> list[dict[str, float]]:
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    num_samples = obs.batch_size[0]
    history: list[dict[str, float]] = []

    for epoch in range(epochs):
        perm = torch.randperm(num_samples, device=device)
        epoch_metrics: dict[str, float] = {}

        for start in range(0, num_samples, batch_size):
            idx = perm[start : start + batch_size]
            batch_obs = obs[idx]
            batch_targets = targets[idx]

            optimizer.zero_grad()
            loss, metrics = compute_loss(model, batch_obs, batch_targets)
            loss.backward()
            optimizer.step()

            for k, v in metrics.items():
                epoch_metrics[k] = epoch_metrics.get(k, 0.0) + v

        n_batches = (num_samples + batch_size - 1) // batch_size
        epoch_metrics = {k: v / n_batches for k, v in epoch_metrics.items()}
        history.append(epoch_metrics)

        if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == 0:
            print(
                f"epoch {epoch + 1:4d}/{epochs} | "
                f"mse={epoch_metrics['mse']:.6f} | total={epoch_metrics['total']:.6f}"
            )

    return history


@torch.no_grad()
def evaluate(model: BackboneMoE, obs: TensorDict, targets: torch.Tensor) -> dict[str, float]:
    model.eval()
    out = model(obs, train_mode=False)
    mse = nn.functional.mse_loss(out["actions"], targets).item()
    gates = model(obs, train_mode=True)["moe_gates"]
    usage = (gates > 1e-6).float().mean(dim=0)
    return {
        "mse": mse,
        "gate_entropy": -(gates * gates.clamp_min(1e-8).log()).sum(dim=-1).mean().item(),
        "expert_usage": usage.cpu().tolist(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-samples", type=int, default=4096)
    parser.add_argument("--obs-dim", type=int, default=16)
    parser.add_argument("--output-dim", type=int, default=6)
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[64, 64])
    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--load-balance-loss-weight", type=float, default=0.01)
    parser.add_argument("--router-z-loss-weight", type=float, default=0.001)
    parser.add_argument("--router-only", action="store_true")
    parser.add_argument("--mse-threshold", type=float, default=0.05, help="Success if eval MSE is below this.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    obs, targets, _, _ = generate_dataset(
        num_samples=args.num_samples,
        obs_dim=args.obs_dim,
        output_dim=args.output_dim,
        num_experts=args.num_experts,
        hidden_dims=args.hidden_dims,
        noise_std=args.noise_std,
        device=device,
        seed=args.seed,
    )

    # Hold out 20% for evaluation
    n = obs.batch_size[0]
    n_train = int(0.8 * n)
    perm = torch.randperm(n, device=device)
    train_idx, eval_idx = perm[:n_train], perm[n_train:]

    train_obs, train_targets = obs[train_idx], targets[train_idx]
    eval_obs, eval_targets = obs[eval_idx], targets[eval_idx]

    model = make_moe_backbone(
        train_obs,
        args.output_dim,
        args.num_experts,
        args.top_k,
        args.hidden_dims,
        args.load_balance_loss_weight,
        args.router_z_loss_weight,
        args.router_only,
    ).to(device)

    print(f"Device: {device}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Train samples: {n_train}, eval samples: {n - n_train}")

    before = evaluate(model, eval_obs, eval_targets)
    print(f"Eval before training: mse={before['mse']:.6f}")

    train(
        model,
        train_obs,
        train_targets,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
    )

    after = evaluate(model, eval_obs, eval_targets)
    print(f"Eval after training:  mse={after['mse']:.6f}")
    print(f"Expert usage (eval):  {[f'{u:.2f}' for u in after['expert_usage']]}")
    print(f"Gate entropy:         {after['gate_entropy']:.4f}")

    if after["mse"] < args.mse_threshold:
        print(f"OK: eval MSE {after['mse']:.6f} < threshold {args.mse_threshold}")
    else:
        print(f"WARN: eval MSE {after['mse']:.6f} >= threshold {args.mse_threshold} (try more epochs/samples)")


if __name__ == "__main__":
    main()
