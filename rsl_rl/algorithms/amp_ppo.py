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
from rsl_rl.algorithms.plugins import AMPPlugin, ExternalAMPProvider
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


class AMPDiscriminator(nn.Module):
    """Small MLP discriminator used by adversarial motion prior training."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str = "amp",
        hidden_dims: tuple[int, ...] | list[int] = (256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
    ) -> None:
        """Initialize the discriminator network and optional observation normalizers."""
        super().__init__()
        self.obs_groups = obs_groups[obs_set]
        self.single_input_dim = sum(obs[group].shape[-1] for group in self.obs_groups)
        input_dim = 2 * self.single_input_dim

        self.obs_normalization = obs_normalization
        self.obs_normalizers = (
            nn.ModuleDict({group: EmpiricalNormalization(obs[group].shape[-1]) for group in self.obs_groups})
            if obs_normalization
            else None
        )
        self.mlp = MLP(input_dim, 1, hidden_dims, activation)

    def _extract_features(self, obs: TensorDict | torch.Tensor) -> torch.Tensor:
        """Return a flattened AMP feature tensor for one observation frame."""
        if isinstance(obs, torch.Tensor):
            return obs
        inputs = []
        for group in self.obs_groups:
            value = obs[group]
            if self.obs_normalizers is not None:
                value = self.obs_normalizers[group](value)
            inputs.append(value)
        return torch.cat(inputs, dim=-1)

    def extract_input(
        self,
        obs: TensorDict | torch.Tensor,
        next_obs: TensorDict | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the default transition input ``[state, next_state]`` for AMP discrimination."""
        if isinstance(obs, torch.Tensor) and next_obs is None and obs.shape[-1] == 2 * self.single_input_dim:
            return obs
        current = self._extract_features(obs)
        if next_obs is None:
            next_inputs = current
        else:
            next_inputs = self._extract_features(next_obs)
        inputs = [current, next_inputs]
        return torch.cat(inputs, dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        """Update running statistics for AMP observations."""
        if self.obs_normalizers is not None:
            for group in self.obs_groups:
                self.obs_normalizers[group].update(obs[group])

    def forward(
        self,
        obs: TensorDict | torch.Tensor,
        next_obs: TensorDict | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return discriminator logits. Positive logits indicate expert-like transitions."""
        return self.mlp(self.extract_input(obs, next_obs))


class AMPPPO(PPO):
    """Compatibility AMP PPO wrapper that reuses the shared AMP plugin core."""

    amp_discriminator: AMPDiscriminator
    """Discriminator that separates policy AMP observations from expert AMP observations."""

    def __init__(
        self,
        actor: StochasticWrapper,
        critic: BaseModel,
        storage: RolloutStorage,
        amp_discriminator: AMPDiscriminator,
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
        amp_cfg: dict | None = None,
        rnd_cfg: dict | None = None,
        symmetry_cfg: dict | None = None,
        plugins=None,
        multi_gpu_cfg: dict | None = None,
    ) -> None:
        """Initialize PPO components and the AMP discriminator."""
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
            plugins=plugins,
            multi_gpu_cfg=multi_gpu_cfg,
        )

        amp_cfg = {} if amp_cfg is None else dict(amp_cfg)
        self.amp_discriminator = amp_discriminator.to(self.device)
        self.amp_reward_scale = float(amp_cfg.get("reward_scale", 1.0))
        self.amp_loss_coef = float(amp_cfg.get("loss_coef", 1.0))
        self.amp_gradient_penalty_coef = float(amp_cfg.get("gradient_penalty_coef", 10.0))
        self.amp_rewards = torch.zeros(storage.num_envs, device=self.device)
        self._amp_plugin_core = AMPPlugin(
            reward_scale=self.amp_reward_scale,
            loss_coef=self.amp_loss_coef,
            gradient_penalty_coef=self.amp_gradient_penalty_coef,
            provider=ExternalAMPProvider(),
            discriminator=self.amp_discriminator,
        )
        self._amp_plugin_core._embedded_loss_handled_externally = True
        self._amp_plugin_core.to(self.device)
        self._amp_plugin_core.provider.setup(env=None, device=self.device, obs=None)
        self._amp_plugin_core_metrics_enabled = False

        if amp_cfg.get("expert_data") is not None:
            self.set_amp_expert_data(amp_cfg["expert_data"])

        self.optimizer = resolve_optimizer(optimizer)(
            chain(self.actor.parameters(), self.critic.parameters(), self.amp_discriminator.parameters()),
            lr=learning_rate,
        )  # type: ignore

    def set_amp_expert_data(self, expert_data: TensorDict | dict[str, torch.Tensor] | torch.Tensor) -> None:
        """Set a fixed pool of expert AMP observations."""
        self._amp_plugin_core.provider.set_expert_data(expert_data)

    def set_amp_expert_sampler(
        self, sampler: Callable[[int], TensorDict | dict[str, torch.Tensor] | torch.Tensor]
    ) -> None:
        """Set a sampler that returns expert AMP observations for a requested batch size."""
        self._amp_plugin_core.provider.set_expert_sampler(sampler)

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        """Add AMP style reward before storing the transition."""
        self.amp_discriminator.update_normalization(obs)
        current_obs = self.transition.observations if self.transition.observations is not None else obs
        self.amp_rewards = self.compute_amp_rewards(current_obs, obs)
        super().process_env_step(obs, rewards + self.amp_rewards, dones, extras)

    def compute_amp_rewards(self, obs: TensorDict, next_obs: TensorDict | None = None) -> torch.Tensor:
        """Compute the discriminator reward through the shared AMP plugin core."""
        current_obs = obs
        next_observations = next_obs
        if next_observations is None:
            transition_obs = self.transition.observations
            if transition_obs is not None and transition_obs is not obs:
                current_obs = transition_obs
                next_observations = obs
            else:
                next_observations = obs
        return self._amp_plugin_core.compute_reward_from_observations(current_obs, next_observations)

    def _compute_loss(self, mb_forward_results: dict, mb_rollout_data: dict) -> dict:
        """Compute PPO and AMP discriminator losses."""
        loss_results = super()._compute_loss(mb_forward_results, mb_rollout_data)
        batch = mb_rollout_data["batch"]
        amp_loss_dict = self._compute_amp_loss(batch.observations, batch.next_observations)
        loss_results["loss"] = loss_results["loss"] + self.amp_loss_coef * amp_loss_dict["amp_loss"]
        loss_results["amp_loss_dict"] = amp_loss_dict
        return loss_results

    def _compute_amp_loss(
        self,
        policy_obs: TensorDict,
        next_policy_obs: TensorDict | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute binary discriminator loss on policy and expert AMP transitions."""
        amp_loss_dict = self._amp_plugin_core.compute_loss_from_policy_obs(
            (policy_obs, next_policy_obs) if next_policy_obs is not None else policy_obs,
            use_policy_replay=False,
            detach_metrics=False,
        )
        if self._amp_plugin_core_metrics_enabled and amp_loss_dict:
            self._amp_plugin_core.record_loss_metrics(amp_loss_dict)
        return amp_loss_dict

    def update(self) -> dict[str, float]:
        """Run PPO update while reusing the shared AMPPlugin lifecycle for AMP internals."""
        self._amp_plugin_core_metrics_enabled = True
        self.plugins.append(self._amp_plugin_core)
        try:
            return super().update()
        finally:
            plugin = self.plugins.pop()
            if plugin is not self._amp_plugin_core:
                raise RuntimeError("AMPPPO internal AMP plugin lifecycle stack became inconsistent.")
            self._amp_plugin_core_metrics_enabled = False

    def _as_amp_tensordict(self, data: TensorDict | dict[str, torch.Tensor] | torch.Tensor) -> TensorDict:
        """Move expert AMP data to the training device and flatten leading batch dims."""
        if isinstance(data, torch.Tensor):
            group = self.amp_discriminator.obs_groups[0]
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

    @property
    def amp_expert_data(self) -> TensorDict | None:
        """Compatibility view of expert AMP data backed by the internal provider."""
        provider = getattr(getattr(self, "_amp_plugin_core", None), "provider", None)
        expert_data = getattr(provider, "expert_data", None)
        if expert_data is None:
            return None
        return self._as_amp_tensordict(expert_data)

    @amp_expert_data.setter
    def amp_expert_data(self, expert_data: TensorDict | dict[str, torch.Tensor] | torch.Tensor | None) -> None:
        """Route legacy expert-data assignments through the internal provider."""
        if expert_data is None:
            provider = getattr(getattr(self, "_amp_plugin_core", None), "provider", None)
            if provider is not None and hasattr(provider, "expert_data"):
                provider.expert_data = None
            return
        self.set_amp_expert_data(expert_data)

    @property
    def amp_expert_sampler(
        self,
    ) -> Callable[[int], TensorDict | dict[str, torch.Tensor] | torch.Tensor] | None:
        """Compatibility view of the expert AMP sampler backed by the internal provider."""
        provider = getattr(getattr(self, "_amp_plugin_core", None), "provider", None)
        return getattr(provider, "expert_sampler", None)

    @amp_expert_sampler.setter
    def amp_expert_sampler(
        self,
        sampler: Callable[[int], TensorDict | dict[str, torch.Tensor] | torch.Tensor] | None,
    ) -> None:
        """Route legacy sampler assignments through the internal provider."""
        if sampler is None:
            provider = getattr(getattr(self, "_amp_plugin_core", None), "provider", None)
            if provider is not None and hasattr(provider, "expert_sampler"):
                provider.expert_sampler = None
            return
        self.set_amp_expert_sampler(sampler)

    def train_mode(self) -> None:
        """Set train mode for learnable models."""
        super().train_mode()
        self._amp_plugin_core.on_train_mode(self)

    def eval_mode(self) -> None:
        """Set evaluation mode for learnable models."""
        super().eval_mode()
        self._amp_plugin_core.on_eval_mode(self)

    def save(self) -> dict:
        """Return a dict of all models for saving."""
        saved_dict = super().save()
        self._amp_plugin_core.on_save(self, saved_dict)
        saved_dict["amp_discriminator_state_dict"] = clone_state_dict_tensors(self.amp_discriminator.state_dict())
        return saved_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Load specified models from a saved dict."""
        load_amp = load_cfg is None or load_cfg.get("amp", True)
        load_iteration = super().load(loaded_dict, load_cfg, strict)
        self._amp_plugin_core.on_load(self, loaded_dict)
        if load_amp:
            self.amp_discriminator.load_state_dict(
                clone_state_dict_tensors(loaded_dict["amp_discriminator_state_dict"]),
                strict=strict,
            )
        return load_iteration

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> AMPPPO:
        """Construct the AMP PPO algorithm."""
        alg_class: type[AMPPPO] = resolve_callable(cfg["algorithm"].pop("class_name"))  # type: ignore
        critic_class: type[BaseModel] = resolve_callable(cfg["critic"].pop("class_name"))  # type: ignore

        default_sets = ["actor", "critic", "amp"]
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)

        amp_cfg = dict(cfg["algorithm"].pop("amp_cfg", {}))
        discriminator_cfg = dict(amp_cfg.pop("discriminator", {}))
        discriminator_class: type[AMPDiscriminator] = resolve_callable(
            discriminator_cfg.pop("class_name", "AMPDiscriminator")
        )  # type: ignore

        cfg["algorithm"].pop("symmetry_cfg", None)
        plugin_cfgs = PPO._resolve_plugin_cfgs(cfg["algorithm"], include_amp_cfg=False)
        if any(PPO._is_amp_plugin_cfg(plugin_cfg) for plugin_cfg in plugin_cfgs):
            raise ValueError("AMPPPO already owns an internal AMP core and cannot be combined with an explicit AMPPlugin.")
        plugins = PPO._build_plugins(plugin_cfgs, obs)

        actor = construct_actor_with_shell(obs, cfg["obs_groups"], cfg["actor"], env.num_actions).to(device)
        print(f"Actor Model: {actor}")
        if cfg["algorithm"].pop("share_cnn_encoders", None):
            cfg["critic"]["cnns"] = actor.backbone.cnns  # type: ignore
        critic: BaseModel = critic_class(obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]).to(device)
        print(f"Critic Model: {critic}")

        amp_discriminator = discriminator_class(obs, cfg["obs_groups"], "amp", **discriminator_cfg).to(device)
        print(f"AMP Discriminator: {amp_discriminator}")

        storage = RolloutStorage("rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)

        alg: AMPPPO = alg_class(
            actor,
            critic,
            storage,
            amp_discriminator,
            device=device,
            amp_cfg=amp_cfg,
            plugins=plugins,
            **cfg["algorithm"],
            multi_gpu_cfg=cfg.get("multi_gpu"),
        )
        PPO._initialize_plugins(alg, env)
        return alg

    def broadcast_parameters(self) -> None:
        """Broadcast model parameters to all GPUs."""
        model_params = [self.actor.state_dict(), self.critic.state_dict(), self.amp_discriminator.state_dict()]
        torch.distributed.broadcast_object_list(model_params, src=0)
        self.actor.load_state_dict(model_params[0])
        self.critic.load_state_dict(model_params[1])
        self.amp_discriminator.load_state_dict(model_params[2])

    def reduce_parameters(self) -> None:
        """Collect gradients from all GPUs and average them."""
        all_params = list(chain(self.actor.parameters(), self.critic.parameters(), self.amp_discriminator.parameters()))
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
