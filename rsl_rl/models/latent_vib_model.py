from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.models.model_base import Model_Base
from rsl_rl.modules import HiddenState, MLP
from rsl_rl.utils import unpad_trajectories


class LatentVIBModel(Model_Base):
    """Encoder-prior-decoder student with a conditional variational bottleneck."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        state_groups: list[str] | tuple[str, ...] | None = None,
        target_groups: list[str] | tuple[str, ...] | None = None,
        latent_dim: int = 64,
        posterior_hidden_dims: list[int] | tuple[int, ...] = (1024, 512, 256),
        prior_hidden_dims: list[int] | tuple[int, ...] = (512, 256),
        decoder_hidden_dims: list[int] | tuple[int, ...] = (1024, 512, 256),
        activation: str = "swish",
        min_log_std: float = -5.0,
        max_log_std: float = 2.0,
        sample_latent_train: bool = True,
        sample_latent_eval: bool = False,
        obs_normalization: bool = True,
    ) -> None:
        super().__init__(obs, obs_groups, obs_set, output_dim, obs_normalization=obs_normalization)

        self.state_groups = list(state_groups or ["prop"])
        self.target_groups = list(target_groups or ["rbt_cmd_mf"])
        self.latent_dim = int(latent_dim)
        self.min_log_std = float(min_log_std)
        self.max_log_std = float(max_log_std)
        self.sample_latent_train = bool(sample_latent_train)
        self.sample_latent_eval = bool(sample_latent_eval)

        self._validate_groups(self.state_groups, "state_groups")
        self._validate_groups(self.target_groups, "target_groups")

        state_dim = sum(self.obs_dim[group] for group in self.state_groups)
        target_dim = sum(self.obs_dim[group] for group in self.target_groups)

        self.posterior = MLP(
            input_dim=state_dim + target_dim,
            output_dim=2 * self.latent_dim,
            hidden_dims=posterior_hidden_dims,
            activation=activation,
        )
        self.prior = MLP(
            input_dim=state_dim,
            output_dim=2 * self.latent_dim,
            hidden_dims=prior_hidden_dims,
            activation=activation,
        )
        self.decoder = MLP(
            input_dim=state_dim + self.latent_dim,
            output_dim=output_dim,
            hidden_dims=decoder_hidden_dims,
            activation=activation,
        )

    def _validate_groups(self, groups: list[str], field_name: str) -> None:
        missing = [group for group in groups if group not in self.obs_dim]
        if missing:
            raise ValueError(
                f"LatentVIBModel {field_name} contains groups not available in obs_set: "
                f"{missing}. Available groups: {list(self.obs_dim.keys())}"
            )

    @staticmethod
    def _cat_groups(obs: TensorDict, groups: list[str]) -> torch.Tensor:
        if not groups:
            batch_size = obs.batch_size[0]
            return torch.zeros(batch_size, 0, device=obs.device)
        return torch.cat([obs[group] for group in groups], dim=-1)

    def _split_gaussian_params(self, params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, log_std = torch.chunk(params, chunks=2, dim=-1)
        log_std = torch.clamp(log_std, min=self.min_log_std, max=self.max_log_std)
        return mu, log_std

    def _prepare_obs(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        train_mode: bool = False,
        groups: list[str] | None = None,
    ) -> TensorDict:
        del hidden_state, train_mode
        obs = unpad_trajectories(obs, masks) if masks is not None else obs
        if not self.obs_normalization:
            return obs

        obs_normed = obs.clone()
        for group in groups or self.obs_groups:
            obs_normed[group] = self.obs_normalizers[group](obs[group])
        return obs_normed

    def _prior_from_prepared_obs(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        state = self._cat_groups(obs, self.state_groups)
        return self._split_gaussian_params(self.prior(state))

    def _decode_from_prepared_obs(self, obs: TensorDict, latent: torch.Tensor) -> torch.Tensor:
        state = self._cat_groups(obs, self.state_groups)
        return self.decoder(torch.cat([state, latent], dim=-1))

    def encode_prior(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        train_mode: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Encode the learnable conditional prior ``p(z | state)`` for downstream controllers."""
        obs = self._prepare_obs(obs, masks, hidden_state, train_mode, groups=self.state_groups)
        prior_mu, prior_log_std = self._prior_from_prepared_obs(obs)
        return {
            "prior_mu": prior_mu,
            "prior_log_std": prior_log_std,
        }

    def decode(
        self,
        obs: TensorDict,
        latent: torch.Tensor,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        train_mode: bool = False,
    ) -> torch.Tensor:
        """Decode a latent code into the full action space conditioned on state."""
        obs = self._prepare_obs(obs, masks, hidden_state, train_mode, groups=self.state_groups)
        return self._decode_from_prepared_obs(obs, latent)

    def decode_prior_residual(
        self,
        obs: TensorDict,
        z_residual: torch.Tensor,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        train_mode: bool = False,
    ) -> torch.Tensor:
        """Decode a residual latent action around the prior mean."""
        obs = self._prepare_obs(obs, masks, hidden_state, train_mode, groups=self.state_groups)
        prior_mu, _ = self._prior_from_prepared_obs(obs)
        return self._decode_from_prepared_obs(obs, prior_mu + z_residual)

    @staticmethod
    def _kl_diag_gaussians(
        q_mu: torch.Tensor,
        q_log_std: torch.Tensor,
        p_mu: torch.Tensor,
        p_log_std: torch.Tensor,
    ) -> torch.Tensor:
        q_var = torch.exp(2.0 * q_log_std)
        p_var = torch.exp(2.0 * p_log_std)
        kl = p_log_std - q_log_std + (q_var + (q_mu - p_mu).pow(2)) / (2.0 * p_var) - 0.5
        return kl.sum(dim=-1)

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        train_mode: bool = False,
    ) -> dict[str, torch.Tensor]:
        obs = self._prepare_obs(obs, masks, hidden_state, train_mode)

        state = self._cat_groups(obs, self.state_groups)
        target = self._cat_groups(obs, self.target_groups)
        q_mu, q_log_std = self._split_gaussian_params(self.posterior(torch.cat([state, target], dim=-1)))
        p_mu, p_log_std = self._split_gaussian_params(self.prior(state))

        sample_latent = self.sample_latent_train if train_mode else self.sample_latent_eval
        if sample_latent:
            latent = q_mu + torch.randn_like(q_mu) * torch.exp(q_log_std)
        else:
            latent = q_mu

        actions = self.decoder(torch.cat([state, latent], dim=-1))
        kl = self._kl_diag_gaussians(q_mu, q_log_std, p_mu, p_log_std)

        return {
            "actions": actions,
            "kl": kl,
            "latent": latent,
            "posterior_mu": q_mu,
            "posterior_log_std": q_log_std,
            "prior_mu": p_mu,
            "prior_log_std": p_log_std,
        }
