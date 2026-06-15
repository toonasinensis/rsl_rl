from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.env import VecEnv
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import construct_actor_with_shell, resolve_callable, resolve_obs_groups, resolve_optimizer


class ConfigurableDistillation:
    """Online DAgger-style distillation with config-built student, teacher, and losses."""

    teacher_loaded: bool = False

    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module,
        storage: RolloutStorage,
        num_learning_epochs: int = 1,
        gradient_length: int = 15,
        learning_rate: float = 1e-3,
        max_grad_norm: float | None = None,
        optimizer: str = "adam",
        loss_terms: list[dict[str, Any]] | None = None,
        device: str = "cpu",
        student_train_mode_in_rollout: bool = True,
        teacher_stochastic: bool = False,
        multi_gpu_cfg: dict | None = None,
        **kwargs: dict,
    ) -> None:
        del kwargs
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        self.student = student.to(self.device)
        self.teacher = teacher.to(self.device)
        self._freeze_teacher()

        self.optimizer = resolve_optimizer(optimizer)(self.student.parameters(), lr=learning_rate)  # type: ignore
        self.storage = storage
        self.transition = RolloutStorage.Transition()
        self.last_hidden_states = (None, None)
        self.plugins: list = []

        self.num_learning_epochs = int(num_learning_epochs)
        self.gradient_length = int(gradient_length)
        self.learning_rate = float(learning_rate)
        self.max_grad_norm = max_grad_norm
        self.loss_terms = loss_terms or [
            {
                "name": "action",
                "kind": "supervised",
                "loss": "mse",
                "pred_key": "actions",
                "target_key": "privileged_actions",
                "weight": 1.0,
            }
        ]
        self.student_train_mode_in_rollout = bool(student_train_mode_in_rollout)
        self.teacher_stochastic = bool(teacher_stochastic)
        self.num_updates = 0

    def _freeze_teacher(self) -> None:
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad_(False)

    @staticmethod
    def _construct_model(
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        cfg: dict,
    ) -> nn.Module:
        model_cfg = copy.deepcopy(cfg)
        class_name = model_cfg.get("class_name")
        model_cls = resolve_callable(class_name)
        if isinstance(model_cls, type) and model_cls.__name__ == "ActorModel":
            return construct_actor_with_shell(obs, obs_groups, model_cfg, output_dim, obs_set=obs_set)

        model_cfg.pop("class_name")
        return model_cls(obs, obs_groups, obs_set, output_dim, **model_cfg)

    @staticmethod
    def _normalize_output(output: torch.Tensor | dict[str, Any]) -> dict[str, Any]:
        if isinstance(output, torch.Tensor):
            return {"actions": output}
        if not isinstance(output, dict):
            raise TypeError(f"Model output must be a Tensor or dict, got {type(output)!r}.")

        normalized = dict(output)
        extra = normalized.pop("extra", None)
        if isinstance(extra, dict):
            normalized.update(extra)
        if "actions" not in normalized:
            raise KeyError(f"Model output dict must contain 'actions'. Available keys: {list(normalized.keys())}")
        return normalized

    @staticmethod
    def _forward_model(
        model: nn.Module,
        obs: TensorDict,
        *,
        train_mode: bool,
        stochastic_output: bool = False,
    ) -> dict[str, Any]:
        try:
            output = model(obs, stochastic_output=stochastic_output, train_mode=train_mode)
        except TypeError:
            output = model(obs, train_mode=train_mode)
        return ConfigurableDistillation._normalize_output(output)

    @staticmethod
    def _lookup(source: Any, key: str) -> Any:
        current = source
        for part in key.split("."):
            if isinstance(current, dict):
                current = current[part]
            else:
                current = current[part] if hasattr(current, "__getitem__") and part in current else getattr(current, part)
        return current

    @staticmethod
    def _loss_fn(name: str):
        loss_fns = {
            "mse": nn.functional.mse_loss,
            "huber": nn.functional.huber_loss,
            "l1": nn.functional.l1_loss,
        }
        if name not in loss_fns:
            raise ValueError(f"Unknown supervised loss {name!r}. Supported losses: {list(loss_fns.keys())}")
        return loss_fns[name]

    def _scheduled_weight(self, term: dict[str, Any]) -> float:
        weight = float(term.get("weight", 1.0))
        schedule = term.get("schedule")
        if not schedule:
            return weight

        schedule_type = schedule.get("type")
        if schedule_type == "linear_warmup":
            steps = max(int(schedule.get("steps", 1)), 1)
            return weight * min(float(self.num_updates) / float(steps), 1.0)
        raise ValueError(f"Unknown loss schedule type: {schedule_type!r}")

    def _compute_loss_terms(self, student_output: dict[str, Any], batch: RolloutStorage.Batch) -> tuple[torch.Tensor, dict]:
        total_loss = None
        metrics: dict[str, torch.Tensor] = {}

        for term in self.loss_terms:
            name = term["name"]
            kind = term.get("kind", "supervised")
            weight = self._scheduled_weight(term)

            if kind == "supervised":
                pred = self._lookup(student_output, term["pred_key"])
                target = self._lookup(batch, term["target_key"])
                raw = self._loss_fn(term.get("loss", "mse"))(pred, target)
            elif kind == "output_mean":
                raw = self._lookup(student_output, term["pred_key"]).mean()
            elif kind == "aux_losses":
                aux_losses = student_output.get("aux_losses", {})
                aux_key = term.get("key")
                raw = aux_losses[aux_key] if aux_key else sum(aux_losses.values())
            else:
                raise ValueError(f"Unknown loss term kind: {kind!r}")

            weighted = raw * weight
            metrics[name] = raw.detach()
            metrics[f"{name}_weighted"] = weighted.detach()
            metrics[f"{name}_weight"] = torch.as_tensor(weight, device=weighted.device)
            total_loss = weighted if total_loss is None else total_loss + weighted

        if total_loss is None:
            raise ValueError("At least one loss term must be configured.")
        metrics["total"] = total_loss.detach()
        return total_loss, metrics

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Run student and teacher on the current online state and store transition data."""
        student_output = self._forward_model(
            self.student,
            obs,
            train_mode=self.student_train_mode_in_rollout,
            stochastic_output=False,
        )
        teacher_output = self._forward_model(
            self.teacher,
            obs,
            train_mode=False,
            stochastic_output=self.teacher_stochastic,
        )

        actions = student_output["actions"].detach()
        self.transition["observations"] = obs
        self.transition["actions"] = actions
        self.transition["privileged_actions"] = teacher_output["actions"].detach()
        return actions

    def process_env_step(
        self,
        obs: TensorDict,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict[str, torch.Tensor],
    ) -> None:
        del extras
        if hasattr(self.student, "update_normalization"):
            self.student.update_normalization(obs)
        self.transition["rewards"] = rewards
        self.transition["dones"] = dones
        self.storage.add_transition(self.transition)
        self.transition.clear()
        if hasattr(self.student, "reset"):
            self.student.reset(dones)
        if hasattr(self.teacher, "reset"):
            self.teacher.reset(dones)

    def compute_returns(self, obs: TensorDict) -> None:
        del obs

    def _optimizer_step(self, loss: torch.Tensor) -> None:
        self.optimizer.zero_grad()
        loss.backward()
        if self.is_multi_gpu:
            self.reduce_parameters()
        if self.max_grad_norm:
            nn.utils.clip_grad_norm_(self.student.parameters(), self.max_grad_norm)
        self.optimizer.step()
        if hasattr(self.student, "detach_hidden_state"):
            self.student.detach_hidden_state()

    def update(self) -> dict[str, float]:
        self._freeze_teacher()
        metric_sums: dict[str, float] = {}
        batch_count = 0
        pending_loss = None
        pending_count = 0

        for _ in range(self.num_learning_epochs):
            if hasattr(self.student, "reset"):
                self.student.reset(hidden_state=self.last_hidden_states[0])
            if hasattr(self.teacher, "reset"):
                self.teacher.reset(hidden_state=self.last_hidden_states[1])
            if hasattr(self.student, "detach_hidden_state"):
                self.student.detach_hidden_state()

            for batch in self.storage.generator():
                student_output = self._forward_model(self.student, batch["observations"], train_mode=True)
                loss, metrics = self._compute_loss_terms(student_output, batch)

                pending_loss = loss if pending_loss is None else pending_loss + loss
                pending_count += 1
                batch_count += 1
                for key, value in metrics.items():
                    metric_sums[key] = metric_sums.get(key, 0.0) + float(value.mean().item())

                if pending_count % self.gradient_length == 0:
                    self._optimizer_step(pending_loss)
                    pending_loss = None
                    pending_count = 0

                dones = batch["dones"].view(-1)
                if hasattr(self.student, "reset"):
                    self.student.reset(dones)
                if hasattr(self.teacher, "reset"):
                    self.teacher.reset(dones)
                if hasattr(self.student, "detach_hidden_state"):
                    self.student.detach_hidden_state(dones)

        if pending_loss is not None:
            self._optimizer_step(pending_loss)

        self.num_updates += 1
        self.storage.clear()
        if hasattr(self.student, "get_hidden_state") and hasattr(self.teacher, "get_hidden_state"):
            self.last_hidden_states = (self.student.get_hidden_state(), self.teacher.get_hidden_state())
        if hasattr(self.student, "detach_hidden_state"):
            self.student.detach_hidden_state()

        n = max(batch_count, 1)
        return {key: value / n for key, value in metric_sums.items()}

    def train_mode(self) -> None:
        self.student.train()
        self._freeze_teacher()

    def eval_mode(self) -> None:
        self.student.eval()
        self.teacher.eval()

    def save(self) -> dict:
        return {
            "student_state_dict": self.student.state_dict(),
            "teacher_state_dict": self.teacher.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "num_updates": self.num_updates,
        }

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        if load_cfg is None and "actor_state_dict" in loaded_dict:
            load_cfg = {"teacher": True, "iteration": False}
        elif load_cfg is None:
            load_cfg = {"student": True, "teacher": True, "optimizer": True, "iteration": True}

        if load_cfg.get("student"):
            self.student.load_state_dict(loaded_dict["student_state_dict"], strict=strict)
        if load_cfg.get("teacher"):
            self.teacher.load_state_dict(
                loaded_dict.get("teacher_state_dict") or loaded_dict["actor_state_dict"], strict=strict
            )
            self.teacher_loaded = True
            self._freeze_teacher()
        if load_cfg.get("optimizer") and "optimizer_state_dict" in loaded_dict:
            self.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        self.num_updates = int(loaded_dict.get("num_updates", self.num_updates))
        return load_cfg.get("iteration", False)

    def get_policy(self) -> nn.Module:
        return self.student

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> "ConfigurableDistillation":
        cfg = copy.deepcopy(cfg)
        alg_class = resolve_callable(cfg["algorithm"].pop("class_name"))
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], ["student", "teacher"])

        student = ConfigurableDistillation._construct_model(
            obs, cfg["obs_groups"], "student", env.num_actions, cfg["student"]
        )
        print(f"Student Model: {student}")
        teacher = ConfigurableDistillation._construct_model(
            obs, cfg["obs_groups"], "teacher", env.num_actions, cfg["teacher"]
        )
        print(f"Teacher Model: {teacher}")

        storage = RolloutStorage("distillation", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)
        return alg_class(student, teacher, storage, device=device, **cfg["algorithm"], multi_gpu_cfg=cfg["multi_gpu"])

    def broadcast_parameters(self) -> None:
        model_params = [self.student.state_dict(), self.teacher.state_dict()]
        torch.distributed.broadcast_object_list(model_params, src=0)
        self.student.load_state_dict(model_params[0])
        self.teacher.load_state_dict(model_params[1])

    def reduce_parameters(self) -> None:
        params = list(self.student.parameters())
        grads = [param.grad.view(-1) for param in params if param.grad is not None]
        if not grads:
            return
        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        offset = 0
        for param in params:
            if param.grad is not None:
                numel = param.numel()
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                offset += numel
