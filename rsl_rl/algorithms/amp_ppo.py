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
from torch.nn.functional import binary_cross_entropy_with_logits

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.env import VecEnv
from rsl_rl.models import BaseModel, StochasticWrapper
from rsl_rl.modules import MLP, EmpiricalNormalization, HiddenState
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
        input_dim = sum(obs[group].shape[-1] for group in self.obs_groups)

        self.obs_normalization = obs_normalization
        self.obs_normalizers = (
            nn.ModuleDict({group: EmpiricalNormalization(obs[group].shape[-1]) for group in self.obs_groups})
            if obs_normalization
            else None
        )
        self.mlp = MLP(input_dim, 1, hidden_dims, activation)

    def extract_input(self, obs: TensorDict | torch.Tensor) -> torch.Tensor:
        """Return the concatenated AMP observation tensor."""
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
        """Update running statistics for AMP observations."""
        if self.obs_normalizers is not None:
            for group in self.obs_groups:
                self.obs_normalizers[group].update(obs[group])

    def forward(self, obs: TensorDict | torch.Tensor) -> torch.Tensor:
        """Return discriminator logits. Positive logits indicate expert-like samples."""
        return self.mlp(self.extract_input(obs))


class AMPPPO(PPO):
    """PPO with an adversarial motion prior discriminator."""

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
            multi_gpu_cfg=multi_gpu_cfg,
        )

        amp_cfg = {} if amp_cfg is None else dict(amp_cfg)
        self.amp_discriminator = amp_discriminator.to(self.device)
        self.amp_reward_scale = float(amp_cfg.get("reward_scale", 1.0))
        self.amp_loss_coef = float(amp_cfg.get("loss_coef", 1.0))
        self.amp_gradient_penalty_coef = float(amp_cfg.get("gradient_penalty_coef", 10.0))
        self.amp_expert_data: TensorDict | None = None
        self.amp_expert_sampler: Callable[[int], TensorDict | dict[str, torch.Tensor] | torch.Tensor] | None = None
        self.amp_rewards = torch.zeros(storage.num_envs, device=self.device)

        if amp_cfg.get("expert_data") is not None:
            self.set_amp_expert_data(amp_cfg["expert_data"])

        self.optimizer = resolve_optimizer(optimizer)(
            chain(self.actor.parameters(), self.critic.parameters(), self.amp_discriminator.parameters()),
            lr=learning_rate,
        )  # type: ignore

    def set_amp_expert_data(self, expert_data: TensorDict | dict[str, torch.Tensor] | torch.Tensor) -> None:
        """Set a fixed pool of expert AMP observations."""
        self.amp_expert_data = self._as_amp_tensordict(expert_data)

    def set_amp_expert_sampler(
        self, sampler: Callable[[int], TensorDict | dict[str, torch.Tensor] | torch.Tensor]
    ) -> None:
        """Set a sampler that returns expert AMP observations for a requested batch size."""
        self.amp_expert_sampler = sampler

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Sample actions and store transition data."""
        self.transition.hidden_states = (self.actor.get_hidden_state(), self.critic.get_hidden_state())
        self.transition.actions = self.actor(obs, stochastic_output=True)["actions"].detach()
        self.transition.values = self._critic_value(obs).detach()
        self.transition.actions_log_prob = self.actor.get_output_log_prob(  # type: ignore
            self.transition.actions
        ).detach()
        self.transition.distribution_params = tuple(p.detach() for p in self.actor.output_distribution_params)
        self.transition.observations = obs
        return self.transition.actions  # type: ignore

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        """Add AMP style reward before storing the transition."""
        self.amp_discriminator.update_normalization(obs)
        self.amp_rewards = self.compute_amp_rewards(obs)
        super().process_env_step(obs, rewards + self.amp_rewards, dones, extras)

    def compute_amp_rewards(self, obs: TensorDict) -> torch.Tensor:
        """Compute the discriminator reward for AMP observations."""
        with torch.no_grad():
            logits = self.amp_discriminator(obs)
            expert_prob = torch.sigmoid(logits)
            rewards = -torch.log(torch.clamp(1.0 - expert_prob, min=1.0e-4))
        return self.amp_reward_scale * rewards.squeeze(-1)

    def compute_returns(self, obs: TensorDict) -> None:
        """Compute return and advantage targets from stored transitions."""
        st = self.storage
        last_values = self._critic_value(obs).detach()
        advantage = 0
        for step in reversed(range(st.num_transitions_per_env)):
            next_values = last_values if step == st.num_transitions_per_env - 1 else st.values[step + 1]
            next_is_not_terminal = 1.0 - st.dones[step].float()  # type: ignore
            delta = st.rewards[step] + next_is_not_terminal * self.gamma * next_values - st.values[step]
            advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage
            st.returns[step] = advantage + st.values[step]
        st.advantages = st.returns - st.values
        if not self.normalize_advantage_per_mini_batch:
            st.advantages = (st.advantages - st.advantages.mean()) / (st.advantages.std() + 1e-8)

    def _forward_model(
        self, batch: TensorDict | RolloutStorage.Batch, original_batch_size: int
    ) -> dict[str, torch.Tensor | tuple[torch.Tensor, ...] | dict]:
        """Run actor/critic forward pass for one mini-batch."""
        forward_dict = self.actor(
            batch.observations,
            masks=batch.masks,
            hidden_state=batch.hidden_states[0],
            stochastic_output=True,
            train_mode=True,
        )
        actions_log_prob = self.actor.get_output_log_prob(batch.actions)  # type: ignore
        values = self._critic_value(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
        distribution_params = tuple(p[:original_batch_size] for p in self.actor.output_distribution_params)
        entropy = self.actor.output_entropy[:original_batch_size]  # type: ignore

        extra = forward_dict.get("extra", {}) if isinstance(forward_dict, dict) else {}
        aux_losses: dict = extra.get("aux_losses") or {}
        if not aux_losses and extra.get("aux_loss") is not None:
            aux_losses = {"aux_loss": extra["aux_loss"]}

        return {
            "actions_log_prob": actions_log_prob,
            "values": values,
            "distribution_params": distribution_params,
            "entropy": entropy,
            "aux_losses": aux_losses,
        }

    def _compute_loss(self, mb_forward_results: dict, mb_rollout_data: dict) -> dict:
        """Compute PPO and AMP discriminator losses."""
        loss_results = super()._compute_loss(mb_forward_results, mb_rollout_data)
        amp_loss_dict = self._compute_amp_loss(mb_rollout_data["batch"].observations)
        loss_results["loss"] = loss_results["loss"] + self.amp_loss_coef * amp_loss_dict["amp_loss"]
        loss_results["amp_loss_dict"] = amp_loss_dict
        return loss_results

    def _compute_amp_loss(self, policy_obs: TensorDict) -> dict[str, torch.Tensor]:
        """Compute binary discriminator loss on policy and expert AMP observations."""
        policy_inputs = self.amp_discriminator.extract_input(policy_obs).detach()
        policy_inputs = policy_inputs.reshape(-1, policy_inputs.shape[-1])
        expert_obs = self._sample_amp_expert_observations(policy_inputs.shape[0])
        expert_inputs = self.amp_discriminator.extract_input(expert_obs).detach()
        expert_inputs = expert_inputs.reshape(-1, expert_inputs.shape[-1])

        policy_logits = self.amp_discriminator(policy_inputs)
        expert_logits = self.amp_discriminator(expert_inputs)
        policy_loss = binary_cross_entropy_with_logits(policy_logits, torch.zeros_like(policy_logits))
        expert_loss = binary_cross_entropy_with_logits(expert_logits, torch.ones_like(expert_logits))
        gradient_penalty = policy_loss.new_tensor(0.0)

        if self.amp_gradient_penalty_coef > 0.0:
            expert_inputs.requires_grad_(True)
            expert_logits_for_gp = self.amp_discriminator(expert_inputs)
            gradients = torch.autograd.grad(
                expert_logits_for_gp.sum(),
                expert_inputs,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            gradient_penalty = gradients.pow(2).sum(dim=-1).mean()

        amp_loss = 0.5 * (policy_loss + expert_loss) + self.amp_gradient_penalty_coef * gradient_penalty
        policy_acc = (policy_logits.detach() < 0.0).float().mean()
        expert_acc = (expert_logits.detach() > 0.0).float().mean()

        return {
            "amp_loss": amp_loss,
            "policy_loss": policy_loss,
            "expert_loss": expert_loss,
            "gradient_penalty": gradient_penalty,
            "policy_accuracy": policy_acc,
            "expert_accuracy": expert_acc,
        }

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

    def _sample_amp_expert_observations(self, batch_size: int) -> TensorDict:
        """Sample expert AMP observations from the sampler or fixed expert pool."""
        if self.amp_expert_sampler is not None:
            return self._as_amp_tensordict(self.amp_expert_sampler(batch_size))

        assert self.amp_expert_data is not None
        indices = torch.randint(self.amp_expert_data.batch_size[0], (batch_size,), device=self.device)
        return self.amp_expert_data[indices]

    @staticmethod
    def _extract_actions(model_output: torch.Tensor | dict) -> torch.Tensor:
        """Extract the primary tensor from tensor- or dict-returning models."""
        if isinstance(model_output, dict):
            return model_output["actions"]
        return model_output

    def _critic_value(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
    ) -> torch.Tensor:
        """Normalize dict-returning critic outputs to a value tensor."""
        return self._extract_actions(self.critic(obs, masks=masks, hidden_state=hidden_state))

    def train_mode(self) -> None:
        """Set train mode for learnable models."""
        super().train_mode()
        self.amp_discriminator.train()

    def eval_mode(self) -> None:
        """Set evaluation mode for learnable models."""
        super().eval_mode()
        self.amp_discriminator.eval()

    def save(self) -> dict:
        """Return a dict of all models for saving."""
        saved_dict = super().save()
        saved_dict["amp_discriminator_state_dict"] = clone_state_dict_tensors(self.amp_discriminator.state_dict())
        return saved_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Load specified models from a saved dict."""
        load_amp = load_cfg is None or load_cfg.get("amp", True)
        load_iteration = super().load(loaded_dict, load_cfg, strict)
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
        cfg["algorithm"].pop("aux_modules", None)

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
            **cfg["algorithm"],
            multi_gpu_cfg=cfg["multi_gpu"],
        )
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
