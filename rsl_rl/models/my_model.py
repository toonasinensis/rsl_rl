from .model_base import Model_Base

import torch
import torch.nn as nn
from tensordict import TensorDict
from rsl_rl.modules import  HiddenState
from rsl_rl.modules import MLP
from rsl_rl.utils import resolve_callable
import torch.nn.functional as F
from vector_quantize_pytorch import FSQ


class MyModel(Model_Base):
    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        **backbone_cfg
    ) -> None:
        super().__init__(obs, obs_groups, obs_set, output_dim, **backbone_cfg)

        # [FSQ] 从 fsq config 推导 latent_dim = num_fsq_levels × max_num_tokens；无 fsq 时沿用 backbone_cfg["latent_dim"]
        fsq_cfg = backbone_cfg.get("fsq", None)
        if fsq_cfg is not None:
            self.num_fsq_levels = fsq_cfg["num_fsq_levels"]
            self.max_num_tokens = fsq_cfg["max_num_tokens"]
            fsq_level_list = fsq_cfg["fsq_level_list"]
            self.fsq = FSQ(levels=[fsq_level_list] * self.num_fsq_levels)
            latent_dim = self.num_fsq_levels * self.max_num_tokens
        else:
            self.fsq = None
            latent_dim = backbone_cfg["latent_dim"]

        # encoder
        self.encoders = nn.ModuleDict()
        self.encoder_cfg = backbone_cfg.get("encoder", {})
        self.main_encoder_name = backbone_cfg.get("main_encoder", None)

        for k, v in self.encoder_cfg.items():
            encoder_groups = v.get("encoder_groups", [])
            encoder_hidden_dims = v.get("hidden_dims", [])
            activation = v.get("activation", "relu")
            mlp = MLP(
                input_dim=sum(self.obs_dim[g] for g in encoder_groups),
                hidden_dims=encoder_hidden_dims,
                output_dim=latent_dim,
                activation=activation,
            )
            self.encoders[k] = mlp

        # decoder
        self.decoders = nn.ModuleDict()
        self.decoder_cfg = backbone_cfg.get("decoder", {})

        for k, v in self.decoder_cfg.items():
            decoder_groups = v.get("decoder_groups", [])
            decoder_hidden_dims = v.get("hidden_dims", [])
            activation = v.get("activation", "relu")
            output = v.get("outputs", [])

            if "actions" in output:
                output_dim = self.output_dim
            else:
                output_dim = sum(self.obs_dim[g] for g in output)

            input_dim = latent_dim + sum(self.obs_dim[g] for g in decoder_groups)
            mlp = MLP(
                input_dim=input_dim,
                hidden_dims=decoder_hidden_dims,
                output_dim=output_dim,
                activation=activation,
            )
            self.decoders[k] = mlp

    def forward(
            self,
            obs: TensorDict,
            masks: torch.Tensor | None = None,
            hidden_state: HiddenState = None,
            train_mode: bool = False,
        ) -> dict[str, torch.Tensor]:
            obs = super().forward(obs, masks, hidden_state, train_mode)

            latents = {}
            for k, encoder in self.encoders.items():
                encoder_groups = self.encoder_cfg[k].get("encoder_groups", [])
                encoder_input = torch.cat([obs[g] for g in encoder_groups], dim=-1)
                latents[k] = encoder(encoder_input)

            if self.main_encoder_name is not None:
                main_latent = latents[self.main_encoder_name]

            # [FSQ] (batch, latent_dim) → (batch, max_num_tokens, num_fsq_levels) → 量化 → (batch, latent_dim)
            if self.fsq is not None:
                B = main_latent.shape[0]
                main_latent = main_latent.view(B, self.max_num_tokens, self.num_fsq_levels)
                main_latent, _ = self.fsq(main_latent)
                main_latent = main_latent.view(B, -1)

            backbone_output = {}
            actions = self.decoders["action_decoder"](
                torch.cat([main_latent] + [obs[g] for g in self.decoder_cfg["action_decoder"].get("decoder_groups", [])], dim=-1)
            )
            backbone_output["actions"] = actions

            if train_mode:
                recon = {}
                losses = {}
                for k, decoder in self.decoders.items():
                    if k == "action_decoder":
                        continue
                    cfg = self.decoder_cfg[k]
                    decoder_groups = cfg.get("decoder_groups", [])
                    decoder_input = torch.cat([main_latent] + [obs[g] for g in decoder_groups], dim=-1)
                    recon[k] = decoder(decoder_input)
                    target = torch.cat([obs[g] for g in cfg.get("outputs", [])], dim=-1)
                    losses[f"{k}_recon"] = F.mse_loss(recon[k], target) * cfg["recon_weight"]
                    reecode_latent = self.encoders[self.main_encoder_name](recon[k].detach())
                    losses[f"{k}_reecode"] = F.mse_loss(
                        reecode_latent, main_latent.detach()
                    ) * cfg["reecode_loss_weight"]
                backbone_output["aux_losses"] = losses

            return backbone_output
