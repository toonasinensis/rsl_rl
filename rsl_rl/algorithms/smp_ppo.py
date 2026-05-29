# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
import torch.nn as nn
from collections.abc import Callable
from itertools import chain
from tensordict import TensorDict

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.env import VecEnv
from rsl_rl.models import BaseModel, StochasticWrapper
from rsl_rl.modules import MLP, EmpiricalNormalization
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import (
    clone_state_dict_tensors,
    construct_actor_with_shell,
    resolve_callable,
    resolve_obs_groups,
    resolve_optimizer,
)


class SMPDiffusionModel(nn.Module):
    """Denoising diffusion prior used by SMP rewards."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str = "smp",
        hidden_dims: tuple[int, ...] | list[int] = (256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        num_diffusion_steps: int = 16,
        beta_start: float = 1.0e-4,
        beta_end: float = 2.0e-2,
    ) -> None:
        """Initialize a compact diffusion denoiser over SMP observations."""
        super().__init__()
        self.obs_groups = obs_groups[obs_set]
        self.input_dim = sum(obs[group].shape[-1] for group in self.obs_groups)
        self.num_diffusion_steps = num_diffusion_steps

        self.obs_normalizers = (
            nn.ModuleDict({group: EmpiricalNormalization(obs[group].shape[-1]) for group in self.obs_groups})
            if obs_normalization
            else None
        )
        self.mlp = MLP(self.input_dim + 1, self.input_dim, hidden_dims, activation)

        betas = torch.linspace(beta_start, beta_end, num_diffusion_steps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - alpha_bars))

    def extract_input(self, obs: TensorDict | torch.Tensor) -> torch.Tensor:
        """Return the concatenated SMP observation tensor."""
        if isinstance(obs, torch.Tensor):
            return obs

        inputs = []
        for group in self.obs_groups:
            value = obs[group]
            if self.obs_normalizers is not None:
                value = self.obs_normalizers[group](value)
            inputs.append(value)
        return torch.cat(inputs, dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        """Update running statistics for SMP observations."""
        if self.obs_normalizers is not None:
            for group in self.obs_groups:
                self.obs_normalizers[group].update(obs[group])

    def sample_noisy(
        self,
        clean_samples: torch.Tensor,
        timesteps: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply one forward diffusion step to clean samples."""
        batch_size = clean_samples.shape[0]
        if timesteps is None:
            timesteps = torch.randint(self.num_diffusion_steps, (batch_size,), device=clean_samples.device)
        if noise is None:
            noise = torch.randn_like(clean_samples)

        sqrt_alpha = self.sqrt_alpha_bars[timesteps].unsqueeze(-1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alpha_bars[timesteps].unsqueeze(-1)
        noisy_samples = sqrt_alpha * clean_samples + sqrt_one_minus_alpha * noise
        return noisy_samples, noise, timesteps

    def forward(self, noisy_samples: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Predict diffusion noise for noisy SMP samples."""
        time_feature = timesteps.to(dtype=noisy_samples.dtype).unsqueeze(-1) / float(self.num_diffusion_steps - 1)
        return self.mlp(torch.cat([noisy_samples, time_feature], dim=-1))

    def compute_loss(self, clean_samples: torch.Tensor) -> dict[str, torch.Tensor]:
        """Train the denoiser to predict sampled Gaussian noise."""
        noisy_samples, noise, timesteps = self.sample_noisy(clean_samples)
        predicted_noise = self(noisy_samples, timesteps)
        denoising_error = (predicted_noise - noise).pow(2).mean(dim=-1)
        return {
            "smp_loss": denoising_error.mean(),
            "denoising_error": denoising_error.detach().mean(),
        }

    def denoising_error(self, clean_samples: torch.Tensor, timestep: int | None = None) -> torch.Tensor:
        """Return per-sample denoising error used as a style distance."""
        if timestep is None:
            timesteps = torch.full(
                (clean_samples.shape[0],),
                self.num_diffusion_steps // 2,
                dtype=torch.long,
                device=clean_samples.device,
            )
        else:
            timesteps = torch.full((clean_samples.shape[0],), timestep, dtype=torch.long, device=clean_samples.device)

        noisy_samples, noise, timesteps = self.sample_noisy(clean_samples, timesteps=timesteps)
        predicted_noise = self(noisy_samples, timesteps)
        return (predicted_noise - noise).pow(2).mean(dim=-1)


class SMPPPO(PPO):
    """PPO with a diffusion-model style prior."""

    smp_model: SMPDiffusionModel
    """Denoising diffusion model trained on expert SMP observations."""

    def __init__(
        self,
        actor: StochasticWrapper,
        critic: BaseModel,
        storage: RolloutStorage,
        smp_model: SMPDiffusionModel,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        learning_rate: float = 0.001,
        max_grad_norm: float = 1.0,
        optimizer: str = "adam",
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float = 0.01,
        normalize_advantage_per_mini_batch: bool = False,
        device: str = "cpu",
        smp_cfg: dict | None = None,
        rnd_cfg: dict | None = None,
        symmetry_cfg: dict | None = None,
        multi_gpu_cfg: dict | None = None,
    ) -> None:
        """Initialize PPO components and the SMP diffusion model."""
        super().__init__(
            actor,
            critic,
            storage,
            num_learning_epochs=num_learning_epochs,
            num_mini_batches=num_mini_batches,
            clip_param=clip_param,
            gamma=gamma,
            lam=lam,
            value_loss_coef=value_loss_coef,
            entropy_coef=entropy_coef,
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            optimizer=optimizer,
            use_clipped_value_loss=use_clipped_value_loss,
            schedule=schedule,
            desired_kl=desired_kl,
            normalize_advantage_per_mini_batch=normalize_advantage_per_mini_batch,
            device=device,
            rnd_cfg=rnd_cfg,
            symmetry_cfg=symmetry_cfg,
            multi_gpu_cfg=multi_gpu_cfg,
        )

        smp_cfg = {} if smp_cfg is None else dict(smp_cfg)
        self.smp_model = smp_model.to(self.device)
        self.smp_reward_scale = float(smp_cfg.get("reward_scale", 1.0))
        self.smp_reward_temperature = float(smp_cfg.get("reward_temperature", 1.0))
        self.smp_loss_coef = float(smp_cfg.get("loss_coef", 1.0))
        self.smp_reward_timestep = smp_cfg.get("reward_timestep")
        self.smp_expert_data: TensorDict | None = None
        self.smp_expert_sampler: Callable[[int], TensorDict | dict[str, torch.Tensor] | torch.Tensor] | None = None
        self.smp_rewards = torch.zeros(storage.num_envs, device=self.device)

        if smp_cfg.get("expert_data") is not None:
            self.set_smp_expert_data(smp_cfg["expert_data"])

        self.optimizer = resolve_optimizer(optimizer)(
            chain(self.actor.parameters(), self.critic.parameters(), self.smp_model.parameters()),
            lr=learning_rate,
        )  # type: ignore

    def set_smp_expert_data(self, expert_data: TensorDict | dict[str, torch.Tensor] | torch.Tensor) -> None:
        """Set a fixed pool of expert SMP observations."""
        self.smp_expert_data = self._as_smp_tensordict(expert_data)

    def set_smp_expert_sampler(
        self, sampler: Callable[[int], TensorDict | dict[str, torch.Tensor] | torch.Tensor]
    ) -> None:
        """Set a sampler that returns expert SMP observations for a requested batch size."""
        self.smp_expert_sampler = sampler

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        """Add SMP diffusion reward before storing the transition."""
        self.smp_model.update_normalization(obs)
        self.smp_rewards = self.compute_smp_rewards(obs)
        super().process_env_step(obs, rewards + self.smp_rewards, dones, extras)

    def compute_smp_rewards(self, obs: TensorDict) -> torch.Tensor:
        """Compute expert-likeness rewards from denoising error."""
        with torch.no_grad():
            inputs = self.smp_model.extract_input(obs)
            inputs = inputs.reshape(-1, inputs.shape[-1])
            error = self.smp_model.denoising_error(inputs, timestep=self.smp_reward_timestep)
            rewards = torch.exp(-error / self.smp_reward_temperature)
        return self.smp_reward_scale * rewards.reshape(obs.batch_size)

    def _compute_loss(self, mb_forward_results: dict, mb_rollout_data: dict) -> dict:
        """Compute PPO and SMP diffusion losses."""
        loss_results = super()._compute_loss(mb_forward_results, mb_rollout_data)
        smp_loss_dict = self._compute_smp_loss(mb_rollout_data["batch"].observations)
        loss_results["loss"] = loss_results["loss"] + self.smp_loss_coef * smp_loss_dict["smp_loss"]
        loss_results["smp_loss_dict"] = smp_loss_dict
        return loss_results

    def _compute_smp_loss(self, policy_obs: TensorDict) -> dict[str, torch.Tensor]:
        """Compute diffusion score-matching loss on expert SMP observations."""
        policy_inputs = self.smp_model.extract_input(policy_obs).detach()
        policy_inputs = policy_inputs.reshape(-1, policy_inputs.shape[-1])
        expert_obs = self._sample_smp_expert_observations(policy_inputs.shape[0])
        self.smp_model.update_normalization(expert_obs)
        expert_inputs = self.smp_model.extract_input(expert_obs).detach()
        expert_inputs = expert_inputs.reshape(-1, expert_inputs.shape[-1])
        loss_dict = self.smp_model.compute_loss(expert_inputs)

        with torch.no_grad():
            policy_error = self.smp_model.denoising_error(policy_inputs, timestep=self.smp_reward_timestep).mean()
        loss_dict["policy_denoising_error"] = policy_error
        return loss_dict

    def _as_smp_tensordict(self, data: TensorDict | dict[str, torch.Tensor] | torch.Tensor) -> TensorDict:
        """Move expert SMP data to the training device and flatten leading batch dims."""
        if isinstance(data, torch.Tensor):
            group = self.smp_model.obs_groups[0]
            data = TensorDict({group: data.to(self.device)}, batch_size=[data.shape[0]], device=self.device)
        elif isinstance(data, TensorDict):
            data = data.to(self.device)
        else:
            first = next(iter(data.values()))
            data = TensorDict(
                {key: value.to(self.device) for key, value in data.items()},
                batch_size=[first.shape[0]],
                device=self.device,
            )

        if len(data.batch_size) > 1:
            data = data.flatten(0, len(data.batch_size) - 1)
        return data

    def _sample_smp_expert_observations(self, batch_size: int) -> TensorDict:
        """Sample expert SMP observations from the sampler or fixed expert pool."""
        if self.smp_expert_sampler is not None:
            return self._as_smp_tensordict(self.smp_expert_sampler(batch_size))

        assert self.smp_expert_data is not None
        indices = torch.randint(self.smp_expert_data.batch_size[0], (batch_size,), device=self.device)
        return self.smp_expert_data[indices]

    def train_mode(self) -> None:
        """Set train mode for learnable models."""
        super().train_mode()
        self.smp_model.train()

    def eval_mode(self) -> None:
        """Set evaluation mode for learnable models."""
        super().eval_mode()
        self.smp_model.eval()

    def save(self) -> dict:
        """Return a dict of all models for saving."""
        saved_dict = super().save()
        saved_dict["smp_model_state_dict"] = clone_state_dict_tensors(self.smp_model.state_dict())
        return saved_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Load specified models from a saved dict."""
        load_smp = load_cfg is None or load_cfg.get("smp", True)
        load_iteration = super().load(loaded_dict, load_cfg, strict)
        if load_smp:
            self.smp_model.load_state_dict(
                clone_state_dict_tensors(loaded_dict["smp_model_state_dict"]),
                strict=strict,
            )
        return load_iteration

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> SMPPPO:
        """Construct the SMP PPO algorithm."""
        alg_class: type[SMPPPO] = resolve_callable(cfg["algorithm"].pop("class_name"))  # type: ignore
        critic_class: type[BaseModel] = resolve_callable(cfg["critic"].pop("class_name"))  # type: ignore

        default_sets = ["actor", "critic", "smp"]
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)

        smp_cfg = dict(cfg["algorithm"].pop("smp_cfg", {}))
        model_cfg = dict(smp_cfg.pop("model", {}))
        model_class: type[SMPDiffusionModel] = resolve_callable(
            model_cfg.pop("class_name", "SMPDiffusionModel")
        )  # type: ignore

        cfg["algorithm"].pop("symmetry_cfg", None)
        cfg["algorithm"].pop("aux_modules", None)

        actor = construct_actor_with_shell(obs, cfg["obs_groups"], cfg["actor"], env.num_actions).to(device)
        print(f"Actor Model: {actor}")
        if cfg["algorithm"].pop("share_cnn_encoders", None):
            cfg["critic"]["cnns"] = actor.backbone.cnns  # type: ignore
        critic: BaseModel = critic_class(obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]).to(device)
        print(f"Critic Model: {critic}")

        smp_model = model_class(obs, cfg["obs_groups"], "smp", **model_cfg).to(device)
        print(f"SMP Diffusion Model: {smp_model}")

        storage = RolloutStorage("rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)

        alg: SMPPPO = alg_class(
            actor,
            critic,
            storage,
            smp_model,
            device=device,
            smp_cfg=smp_cfg,
            **cfg["algorithm"],
            multi_gpu_cfg=cfg["multi_gpu"],
        )
        return alg

    def broadcast_parameters(self) -> None:
        """Broadcast model parameters to all GPUs."""
        model_params = [self.actor.state_dict(), self.critic.state_dict(), self.smp_model.state_dict()]
        torch.distributed.broadcast_object_list(model_params, src=0)
        self.actor.load_state_dict(model_params[0])
        self.critic.load_state_dict(model_params[1])
        self.smp_model.load_state_dict(model_params[2])

    def reduce_parameters(self) -> None:
        """Collect gradients from all GPUs and average them."""
        all_params = list(chain(self.actor.parameters(), self.critic.parameters(), self.smp_model.parameters()))
        grads = [param.grad.view(-1) for param in all_params if param.grad is not None]
        if len(grads) == 0:
            return
        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                offset += numel
