from typing import List, Optional, Union
import einops
import torch
from pytest import Cache
from transformers import (
    AutoConfig,
    GemmaForCausalLM,
    PaliGemmaForConditionalGeneration,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.models.auto import CONFIG_MAPPING

from .gemma_utils import apply_rope, eager_attention_forward

from torch import Tensor, nn

from .gemma_utils import (
    create_sinusoidal_pos_embedding,
    make_att_2d_masks,
    resize_with_pad,
    sample_beta,
)

import torch.nn.functional as F


class PaliGemmaWithExpertConfig(PretrainedConfig):
    model_type = "PaliGemmaWithExpertModel"
    sub_configs = {"paligemma_config": AutoConfig, "gemma_expert_config": AutoConfig}

    def __init__(
        self,
        paligemma_config: dict | None = None,
        gemma_expert_config: dict | None = None,
        freeze_vision_encoder: bool = True,
        train_expert_only: bool = True,
        attention_implementation: str = "eager",
        **kwargs,
    ):
        self.freeze_vision_encoder = freeze_vision_encoder
        self.train_expert_only = train_expert_only
        self.attention_implementation = attention_implementation

        if paligemma_config is None:
            # Default config from Pi0
            self.paligemma_config = CONFIG_MAPPING["paligemma"](
                transformers_version="4.48.1",
                _vocab_size=257152,
                bos_token_id=2,
                eos_token_id=1,
                hidden_size=2048,
                image_token_index=257152,
                model_type="paligemma",
                pad_token_id=0,
                projection_dim=2048,
                text_config={
                    "hidden_activation": "gelu_pytorch_tanh",
                    "hidden_size": 2048,
                    "intermediate_size": 16384,
                    "model_type": "gemma",
                    "num_attention_heads": 8,
                    "num_hidden_layers": 18,
                    "num_image_tokens": 256,
                    "num_key_value_heads": 1,
                    "torch_dtype": "float32",
                    "vocab_size": 257152,
                },
                vision_config={
                    "hidden_size": 1152,
                    "intermediate_size": 4304,
                    "model_type": "siglip_vision_model",
                    "num_attention_heads": 16,
                    "num_hidden_layers": 27,
                    "num_image_tokens": 256,
                    "patch_size": 14,
                    "projection_dim": 2048,
                    "projector_hidden_act": "gelu_fast",
                    "torch_dtype": "float32",
                    "vision_use_head": False,
                },
            )
        elif isinstance(self.paligemma_config, dict):
            # Override Pi0 default config for PaliGemma
            if "model_type" not in gemma_expert_config:
                paligemma_config["model_type"] = "paligemma"

            cfg_cls = CONFIG_MAPPING[paligemma_config["model_type"]]
            self.paligemma_config = cfg_cls(**paligemma_config)

        if gemma_expert_config is None:
            # Default config from Pi0
            self.gemma_expert_config = CONFIG_MAPPING["gemma"](
                attention_bias=False,
                attention_dropout=0.0,
                bos_token_id=2,
                eos_token_id=1,
                head_dim=256,
                hidden_act="gelu_pytorch_tanh",
                hidden_activation="gelu_pytorch_tanh",
                hidden_size=1024,
                initializer_range=0.02,
                intermediate_size=4096,
                max_position_embeddings=8192,
                model_type="gemma",
                num_attention_heads=8,
                num_hidden_layers=18,
                num_key_value_heads=1,
                pad_token_id=0,
                rms_norm_eps=1e-06,
                rope_theta=10000.0,
                torch_dtype="float32",
                transformers_version="4.48.1",
                use_cache=True,
                vocab_size=257152,
            )
        elif isinstance(self.gemma_expert_config, dict):
            # Override Pi0 default config for Gemma Expert
            if "model_type" not in gemma_expert_config:
                gemma_expert_config["model_type"] = "gemma"

            cfg_cls = CONFIG_MAPPING[paligemma_config["model_type"]]
            self.gemma_expert_config = cfg_cls(**gemma_expert_config)

        super().__init__(**kwargs)

    def __post_init__(self):
        super().__post_init__()
        if self.train_expert_only and not self.freeze_vision_encoder:
            raise ValueError(
                "You set `freeze_vision_encoder=False` and `train_expert_only=True` which are not compatible."
            )

        if self.attention_implementation not in ["eager", "fa2", "flex"]:
            raise ValueError(
                f"Wrong value provided for `attention_implementation` ({self.attention_implementation}). Expected 'eager', 'fa2' or 'flex'."
            )




class GemmaExpertModel(PreTrainedModel,nn.Module):
    # config_class = PaliGemmaWithExpertConfig
    # config_class = PaliGemmaConfig
    def __init__(self, config):
        
        self.config_common = config

        paligemma_with_export_config = PaliGemmaWithExpertConfig(
            freeze_vision_encoder=config.freeze_vision_encoder,
            train_expert_only=config.train_expert_only,
            attention_implementation=config.attention_implementation,
            )

        super().__init__(config=paligemma_with_export_config)

        self.gemma_expert = GemmaForCausalLM(config=paligemma_with_export_config.gemma_expert_config)
        # Remove unused embed_tokens
        self.gemma_expert.model.embed_tokens = None

        self.state_proj = nn.Linear(self.config_common.max_state_dim, self.config_common.proj_width)
        self.action_in_proj = nn.Linear(
            self.config_common.max_action_dim, self.config_common.proj_width
        )
        self.action_out_proj = nn.Linear(
            self.config_common.proj_width, self.config_common.max_action_dim
        )

        self.action_time_mlp_in = nn.Linear(
            self.config_common.proj_width * 2, self.config_common.proj_width
        )
        self.action_time_mlp_out = nn.Linear(
            self.config_common.proj_width, self.config_common.proj_width
        )


        self.attention_interface = self.get_attention_interface()



        # self.to_bfloat16_like_physical_intelligence()
        self.set_requires_grad()

    def set_requires_grad(self):
        """sets the requires_grad attribute of the model parameters based on the configuration.
        If `freeze_vision_encoder` is True, the vision tower parameters are frozen.
        If `train_expert_only` is True, the entire PaliGemma model is frozen.
        """
        # if self.config.freeze_vision_encoder:
        #     self.paligemma.vision_tower.eval()
        #     for params in self.paligemma.vision_tower.parameters():
        #         params.requires_grad = False

        # if self.config.train_expert_only:
        #     self.paligemma.eval()
        #     for params in self.paligemma.parameters():
        #         params.requires_grad = False
        pass

    def train(self, mode: bool = True):
        super().train(mode)
        # if self.config.freeze_vision_encoder:
        #     self.paligemma.vision_tower.eval()
        # if self.config.train_expert_only:
        #     self.paligemma.eval()


    def to_bfloat16_like_physical_intelligence(self):
        """casts the model to bfloat16.

        Modules not casted to bfloat16:
        - paligemma.language_model.model.embed_tokens.weight
        - paligemma.language_model.model.norm.weight
        - gemma_expert.model.norm.weight
        - gemma_expert.lm_head.weight
        """
        self.paligemma = self.paligemma.to(dtype=torch.bfloat16)

        params_to_change_dtype = [
            "language_model.model.layers",
            "gemma_expert.model.layers",
            "vision_tower",
            "multi_modal",
        ]
        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_change_dtype):
                param.data = param.data.to(dtype=torch.bfloat16)

    def embed_image(self, image: torch.Tensor):
        return self.paligemma.get_image_features(image)

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.paligemma.language_model.model.embed_tokens(tokens)

    def handle_kv_cache(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        past_key_values: Optional[Union[List[torch.FloatTensor], Cache]] = None,
        use_cache: Optional[bool] = None,
        fill_kv_cache: Optional[bool] = None,
    ):
        if use_cache:
            if past_key_values is None:
                past_key_values = {}

            if fill_kv_cache:
                past_key_values[layer_idx] = {
                    "key_states": key_states,
                    "value_states": value_states,
                }
            else:
                key_states = torch.cat(
                    [past_key_values[layer_idx][0].permute(0, 2, 1, 3), key_states], dim=1
                )
                value_states = torch.cat(
                    [past_key_values[layer_idx][0].permute(0, 2, 1, 3), value_states],
                    dim=1,
                )
        return key_states, value_states, past_key_values

    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[List[torch.FloatTensor], Cache]] = None,
        inputs_embeds: List[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        fill_kv_cache: Optional[bool] = None,
    ):
        """
        Args:
            attention_mask (Optional[torch.Tensor], optional):
                Attention mask with shape (b, seq_len, seq_len). Defaults to None.
            position_ids (Optional[torch.LongTensor], optional):
                Position indices for applying RoPE. Defaults to None.
            past_key_values (Optional[Union[List[torch.FloatTensor], Cache]], optional):
                Optional kv cache. Defaults to None.
            inputs_embeds (List[torch.FloatTensor], optional):
                Input embeddings. Defaults to None.
            use_cache (Optional[bool], optional):
                Whether to use kv cache. Defaults to None.
            fill_kv_cache (Optional[bool], optional):
                Whether to return kv tensors in this forward pass as cache. Defaults to None.

        Returns:
            outputs_embeds (torch.Tensor): Output embeddings.
            past_key_values (Optional[Union[List[torch.FloatTensor], Cache]]):
                Optional kv cache.
        """
        models = [self.gemma_expert.model]

        # RMSNorm
        num_layers = self.gemma_expert.config.num_hidden_layers
        for layer_idx in range(num_layers):
            query_states = []
            key_states = []
            value_states = []
            for i, hidden_states in enumerate(inputs_embeds):
                if hidden_states is None:
                    continue

                layer = models[i].layers[layer_idx]
                hidden_states = layer.input_layernorm(hidden_states)
                hidden_shape = (*hidden_states.shape[:-1], -1, layer.self_attn.head_dim)

                query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape)
                key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape)
                value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape)

                query_states.append(query_state)
                key_states.append(key_state)
                value_states.append(value_state)

            # B,L,H,D with L sequence length, H number of heads, D head dim
            # concatenate on the number of embeddings/tokens
            query_states = torch.cat(query_states, dim=1)
            key_states = torch.cat(key_states, dim=1)
            value_states = torch.cat(value_states, dim=1)

            query_states = apply_rope(query_states, position_ids)
            key_states = apply_rope(key_states, position_ids)

            key_states, value_states, past_key_values = self.handle_kv_cache(
                key_states,
                value_states,
                layer_idx,
                past_key_values=past_key_values,
                use_cache=use_cache,
                fill_kv_cache=fill_kv_cache,
            )

            att_output = self.attention_interface(
                query_states, key_states, value_states, attention_mask
            )

            # first part of att_output is prefix (up to sequence length, [:, 0:prefix_seq_len])
            outputs_embeds = []
            start = 0
            for i, hidden_states in enumerate(inputs_embeds):
                layer = models[i].layers[layer_idx]

                if hidden_states is not None:
                    end = start + hidden_states.shape[1]

                    if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                        att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
                    out_emb = layer.self_attn.o_proj(att_output[:, start:end])

                    # first residual
                    out_emb += hidden_states
                    after_first_residual = out_emb.clone()

                    out_emb = layer.post_attention_layernorm(out_emb)
                    out_emb = layer.mlp(out_emb)

                    # second residual
                    out_emb += after_first_residual
                    outputs_embeds.append(out_emb)

                    start = end
                else:
                    outputs_embeds.append(None)

            inputs_embeds = outputs_embeds

        # final norm
        outputs_embeds = []
        for i, hidden_states in enumerate(inputs_embeds):
            if hidden_states is not None:
                out_emb = models[i].norm(hidden_states)
                outputs_embeds.append(out_emb)
            else:
                outputs_embeds.append(None)

        return outputs_embeds, past_key_values

    def get_attention_interface(self):
        if self.config.attention_implementation == "fa2":
            raise NotImplementedError("FA2 is not implemented (yet)")
        elif self.config.attention_implementation == "flex":
            # attention_interface = flex_attention_forward
            raise NotImplementedError("Flex attention is not implemented (yet)")
        elif self.config.attention_implementation == "eager":
            attention_interface = eager_attention_forward
        elif self.config.attention_implementation == "xformer":
            # attention_interface = xformer_attention_forward
            raise NotImplementedError("Xformer attention is not implemented (yet)")
        else:
            raise ValueError(
                f"Invalid attention implementation: {self.config.attention_implementation}. "
                "Expected one of ['fa2', 'flex', 'eager', 'xformer']."
            )
        return attention_interface

    
    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing.

        Args:
            state (torch.Tensor):         float32 (*b, s) robot state
            noisy_actions (torch.Tensor): float32 (*b, n, m) noisy actions
            timestep (torch.Tensor):      float32 (*b,) timestep in [0, 1] range
        """
        bsize = state.shape[0]
        device = state.device
        dtype = state.dtype
    
        # embed state
        state_emb = self.state_proj(state)

        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.config_common.proj_width,
            min_period=4e-3,
            max_period=4.0,
            device=device,
        )
        time_emb = time_emb.type(dtype=dtype)

        # Fuse timestep + action information using an MLP
        action_emb = self.action_in_proj(noisy_actions)
        time_emb = einops.repeat(time_emb, "b d -> b n d", n=action_emb.shape[1])
        action_time_emb = torch.cat([action_emb, time_emb], dim=-1)

        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)  # swish == silu
        action_time_emb = self.action_time_mlp_out(action_time_emb)
        action_time_dim = action_time_emb.shape[1]

        # Add to input tokens
        embs = torch.cat([state_emb[:, None], action_time_emb], dim=1)
        pad_masks = torch.ones(
            (bsize, action_time_dim + 1), device=device, dtype=torch.bool
        )

        # Set attention masks for suffix tokens so that prefix tokens cannot attend to suffix tokens.
        # And state token cannot attend action tokens.
        # Action tokens use a bidirectional attention.
        att_masks = torch.zeros(
            (bsize, action_time_dim + 1), device=device, dtype=torch.bool
        )
        att_masks[:, :2] = True

        return embs, pad_masks, att_masks

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)


    def loss_compute(
        self,
        # images,
        # img_masks,
        # lang_tokens,
        # lang_masks,
        past_key_values,
        prefix_pad_masks,
        prefix_att_masks,
        state,
        actions,
        noise=None,
        time=None,
    ) -> Tensor:
        bsize = state.shape[0]
        dtype = state.dtype
        device = state.device
        action_real_dim = actions.shape[2]
        # padding actions to max_action_dim
                # padding actions to max_action_dim (right-side)
        pad_right = max(0, self.config_common.max_action_dim - actions.shape[2])
        if pad_right > 0:
            actions = torch.nn.functional.pad(actions, (0, pad_right), mode="constant", value=0)
            state = torch.nn.functional.pad(state, (0, pad_right), mode="constant", value=0)
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        if noise is None:
            actions_shape = (
                bsize,
                self.config_common.n_action_steps,
                self.config_common.max_action_dim,
            )
            noise = torch.randn(actions_shape, device=device, dtype=dtype)

        if time is None:
            time = self.sample_time(bsize, device).to(dtype)

        # noise[:,-1,:] = actions[:,-1,:]

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions

        x_t[:,-1,:] = actions[:,-1,:]

        noise[:,-1,:] = actions[:,-1,:]
        
        u_t = noise - actions

        # prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
        #     images, img_masks, lang_tokens, lang_masks
        # )
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(
            state, x_t, time
        )

        pad_masks = suffix_pad_masks
        att_masks = suffix_att_masks

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        att_2d_masks = torch.cat([torch.ones((prefix_pad_masks.shape[0], att_2d_masks.shape[2],prefix_pad_masks.shape[1])).to(att_2d_masks.device).bool(), att_2d_masks], dim=2)
        (suffix_out), _ = self.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[suffix_embs],
            use_cache=True,
            fill_kv_cache=None,
        )
        suffix_out = suffix_out[0]
        suffix_out = suffix_out[:, -self.config_common.n_action_steps :]
        v_t = self.action_out_proj(suffix_out)
        losses = F.mse_loss(u_t, v_t, reduction="none")
        losses = losses[:,:,:action_real_dim].mean()
        return losses

    def sample_actions(
        self, past_key_values, prefix_pad_masks, state, noise=None,action_guide=None
    ) -> Tensor:
    # images, img_masks, lang_tokens, lang_masks,
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = state.shape[0]
        device = state.device
        dtype = state.dtype

        if noise is None:
            actions_shape = (
                bsize,
                self.config_common.n_action_steps,
                self.config_common.max_action_dim,
            )
            noise = torch.randn(actions_shape, device=device, dtype=dtype)

        # prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
        #     images, img_masks, lang_tokens, lang_masks
        # )
        # prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        # prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # # Compute image and language key value cache
        # _, past_key_values = self.paligemma_with_expert.forward(
        #     attention_mask=prefix_att_2d_masks,
        #     position_ids=prefix_position_ids,
        #     past_key_values=None,
        #     inputs_embeds=[prefix_embs, None],
        #     use_cache=self.config.use_cache,
        #     fill_kv_cache=True,
        # )

        dt = torch.tensor(-1.0 / self.config.num_steps, dtype=dtype, device=device)
        x_t = noise


        time = torch.tensor(1.0, dtype=dtype, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            if action_guide is not None:
                x_t[:, -1, :] = action_guide
            v_t = self.predict_velocity(
                state, prefix_pad_masks, past_key_values, x_t, expanded_time
            )

            # Euler step
            x_t += dt * v_t
            time += dt

        return x_t

    def predict_velocity(self, state, prefix_pad_masks, past_key_values, x_t, timestep):
        """predict velocity at time t using the suffix model."""
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(
            state, x_t, timestep
        )

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(
            batch_size, suffix_len, prefix_len
        )

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        outputs_embeds, _ = self.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[suffix_embs],
            use_cache=self.config.use_cache,
            fill_kv_cache=False,
        )
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config_common.n_action_steps :]
        v_t = self.action_out_proj(suffix_out)
        return v_t
