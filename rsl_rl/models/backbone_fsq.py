from .backbone_base import BaseModel

import torch
import torch.nn as nn
from tensordict import TensorDict
from rsl_rl.modules import  HiddenState
from rsl_rl.modules import MLP
import torch.nn.functional as F
from vector_quantize_pytorch import FSQ


class BackboneFSQ(BaseModel):
    """ FSQ model based on ModelBase """
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
        self.loss_cfg = backbone_cfg.get("loss", None)
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

        # each encoder owns an equal slice of latent_dim; mask shape: (n_encoders, latent_dim)
        n_enc = len(self.encoder_cfg)
        if n_enc > 1:
            chunk = latent_dim // n_enc
            masks = torch.zeros(n_enc, latent_dim)
            for i in range(n_enc):
                masks[i, i * chunk : (i + 1) * chunk] = 1.0
            self.register_buffer("encoder_masks", masks)   # (n, D), reusable in losses
        else:
            self.encoder_masks = None

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
            obs: TensorDict | dict[str, torch.Tensor],
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
                if self.encoder_masks is None:
                    main_latent = latents[self.main_encoder_name]
                else:
                    # latent_stack: (n, B, D); masks: (n, 1, D) → sum over encoders → (B, D)
                    latent_stack = torch.stack(list(latents.values()), dim=0)
                    main_latent = (latent_stack * self.encoder_masks.unsqueeze(1)).sum(0)

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

                    if self.loss_cfg is not None:
                        if self.loss_cfg.get("token", 0.0) > 0.0:
                            losses[f"{k}_token"] = F.mse_loss(latents["encoder_g1"], latents["encoder_smpl"]) * self.loss_cfg["token"]
                        
                        if self.loss_cfg.get("recon", 0.0) > 0.0:
                            losses[f"{k}_recon"] = F.mse_loss(recon["g1_kin_decoder"], target) * self.loss_cfg["recon"]
                        
                        if self.loss_cfg.get("reencode", 0.0) > 0.0:
                            recon_kin_smpl = self.decoders["g1_kin_decoder"](latents["encoder_smpl"])
                            latents_matched = self.encoders["encoder_g1"](recon_kin_smpl)
                            losses[f"{k}_re_encode"] = F.mse_loss(
                                latents_matched, latents["encoder_g1"].detach()
                            ) * self.loss_cfg["re_encode"]

                backbone_output["aux_losses"] = losses

            return backbone_output
