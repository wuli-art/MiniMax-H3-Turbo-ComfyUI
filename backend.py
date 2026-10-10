"""MiniMax H3 Turbo transformer and Diffusers pipeline loader."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from torch import nn

import diffusers
from diffusers import ComponentsManager, ModularPipeline
from diffusers.models.modeling_utils import get_parameter_dtype
from diffusers.models.transformers.transformer_minimax_h3 import (
    MiniMaxH3Transformer3DModel,
    MiniMaxH3TransformerOutput,
)

from .four_step import (
    apply_four_step_adapter,
    ensure_processor_compat,
    reset_four_step_adapter,
    resolve_four_step_checkpoint,
    validate_four_step_call,
)


OFFICIAL_REPO = "MiniMaxAI/MiniMax-H3"
TURBO_REPO = "Wuli-art/MiniMax-H3-Turbo"
DIFFUSERS_VERSION = "0.40.0"
BASE_COMPONENTS = (
    "text_encoder", "tokenizer", "processor", "vae", "audio_vae",
    "scheduler", "audio_scheduler",
)


class _BlockTimeEmbedding(nn.Module):
    def __init__(self, rows: int, modalities: int, hidden_size: int) -> None:
        super().__init__()
        self.register_buffer(
            "weight", torch.empty(rows, modalities, 6, hidden_size, dtype=torch.bfloat16)
        )


class _OutputTimeEmbedding(nn.Module):
    def __init__(self, rows: int, hidden_size: int) -> None:
        super().__init__()
        self.register_buffer("weight", torch.empty(rows, 2, hidden_size, dtype=torch.bfloat16))


class FixedMiniMaxH3Transformer3DModel(MiniMaxH3Transformer3DModel):
    """Official H3 transformer with precomputed video and audio conditioning."""

    _supports_gradient_checkpointing = False
    _keep_in_fp32_modules = ["proj_in", "audio_proj_in", "proj_out", "audio_proj_out", "rope"]

    def __init__(
        self,
        num_attention_heads: int = 56,
        attention_head_dim: int = 128,
        hidden_size: int = 5376,
        num_layers: int = 50,
        num_refiner_layers: int = 2,
        ffn_dim: int = 14336,
        in_channels: int = 24,
        audio_in_channels: int = 32,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        text_dim: int = 5120,
        freq_dim: int = 256,
        time_embed_hidden_dim: int = 5376,
        time_embed_dim: int = 2688,
        rope_freq_dim: int = 16,
        rope_theta: float = 10000.0,
        norm_eps: float = 1e-5,
        qk_norm_eps: float = 1e-5,
        final_norm_eps: float = 1e-5,
        fixed_num_steps: int = 8,
    ) -> None:
        super().__init__(
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_refiner_layers=num_refiner_layers,
            ffn_dim=ffn_dim,
            in_channels=in_channels,
            audio_in_channels=audio_in_channels,
            patch_size=patch_size,
            text_dim=text_dim,
            freq_dim=freq_dim,
            time_embed_hidden_dim=time_embed_hidden_dim,
            time_embed_dim=time_embed_dim,
            rope_freq_dim=rope_freq_dim,
            rope_theta=rope_theta,
            norm_eps=norm_eps,
            qk_norm_eps=qk_norm_eps,
            final_norm_eps=final_norm_eps,
        )
        fixed_num_steps = int(fixed_num_steps)
        self.register_to_config(fixed_num_steps=fixed_num_steps)
        hidden_size = int(self.config.hidden_size)
        self.video_time_embedding = nn.ModuleList(
            [_BlockTimeEmbedding(fixed_num_steps + 2, 2, hidden_size) for _ in self.transformer_blocks]
        )
        self.audio_time_embedding = nn.ModuleList(
            [_BlockTimeEmbedding(fixed_num_steps, 1, hidden_size) for _ in self.transformer_blocks]
        )
        self.video_output_time_embedding = _OutputTimeEmbedding(fixed_num_steps + 2, hidden_size)
        self.audio_output_time_embedding = _OutputTimeEmbedding(fixed_num_steps, hidden_size)
        self.register_buffer("video_timesteps", torch.empty(fixed_num_steps, dtype=torch.float32))
        self.register_buffer("audio_timesteps", torch.empty(fixed_num_steps, dtype=torch.float32))
        self.register_buffer("visual_condition_timestep", torch.tensor(0.999, dtype=torch.float32))

        self.time_proj = None
        self.time_embedder = None
        for block in self.transformer_blocks:
            block.adaln_proj = None
        self.norm_out.linear = None

    @staticmethod
    def _step(value: torch.Tensor, schedule: torch.Tensor) -> int:
        error = (schedule - value.float()).abs()
        index = int(error.argmin())
        if float(error[index]) > 2e-6:
            raise ValueError(
                f"unsupported timestep {float(value):.9f}; "
                f"use the checkpoint's fixed {schedule.numel()}-step schedule"
            )
        return index

    def _row_plan(self, timestep, timestep_indices, token_tags):
        row_t = timestep.index_select(0, timestep_indices).float()
        is_video = token_tags == 0
        is_text = token_tags == 1
        is_audio = token_tags == 2
        is_condition = is_video & torch.isclose(
            row_t, self.visual_condition_timestep, rtol=0.0, atol=2e-6
        )
        probe = row_t[is_text]
        if probe.numel() == 0:
            probe = row_t[is_video & ~is_condition]
        if probe.numel() == 0:
            raise ValueError("cannot resolve the current video step")
        step = self._step(probe[0], self.video_timesteps)
        if is_audio.any() and self._step(row_t[is_audio][0], self.audio_timesteps) != step:
            raise ValueError("video and audio schedules are on different steps")
        return step, is_video, is_text, is_audio, is_condition

    def forward(
        self,
        hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        timestep_indices: torch.Tensor,
        token_tags: torch.Tensor,
        position_ids: torch.Tensor,
        video_indices: torch.Tensor,
        audio_indices: torch.Tensor,
        text_indices: torch.Tensor,
        attention_kwargs: dict | None = None,
        return_dict: bool = True,
    ):
        del attention_kwargs
        rotary_emb = self.rope(position_ids)
        video = self.proj_in(hidden_states.to(get_parameter_dtype(self.proj_in)))
        audio = self.audio_proj_in(audio_hidden_states.to(get_parameter_dtype(self.audio_proj_in)))
        text = self.context_embedder(encoder_hidden_states.to(get_parameter_dtype(self.context_embedder)))
        text = self.token_refiner(text)
        packed = text.new_zeros((text.shape[0], position_ids.shape[0], text.shape[-1]))
        packed = packed.index_copy(1, text_indices, text)
        packed = packed.index_copy(1, video_indices, video.to(text.dtype))
        packed = packed.index_copy(1, audio_indices, audio.to(text.dtype))

        step, is_video, is_text, is_audio, is_condition = self._row_plan(
            timestep, timestep_indices, token_tags
        )
        condition_row = self.config.fixed_num_steps + int(step == self.config.fixed_num_steps - 1)
        for layer, block in enumerate(self.transformer_blocks):
            table = packed.new_empty((token_tags.numel(), 6, packed.shape[-1]))
            table[is_video & ~is_condition] = self.video_time_embedding[layer].weight[step, 0]
            table[is_condition] = self.video_time_embedding[layer].weight[condition_row, 0]
            table[is_text] = self.video_time_embedding[layer].weight[step, 1]
            table[is_audio] = self.audio_time_embedding[layer].weight[step, 0]
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = table.unbind(dim=1)

            residual = packed
            value = block.norm1(packed)
            value = value * (1.0 + scale_msa[None]) + shift_msa[None]
            packed = residual + gate_msa[None] * block.attn(value, rotary_emb)
            residual = packed
            value = block.norm2(packed)
            value = value * (1.0 + scale_mlp[None]) + shift_mlp[None]
            packed = residual + gate_mlp[None] * block.ff(value)

        output_table = packed.new_empty((token_tags.numel(), 2, packed.shape[-1]))
        output_table[is_video & ~is_condition] = self.video_output_time_embedding.weight[step]
        output_table[is_condition] = self.video_output_time_embedding.weight[condition_row]
        output_table[is_text] = self.video_output_time_embedding.weight[step]
        output_table[is_audio] = self.audio_output_time_embedding.weight[step]
        shift, scale = output_table.unbind(dim=1)
        packed = self.norm_out.norm(packed) * (1.0 + scale[None]) + shift[None]
        packed = packed.to(get_parameter_dtype(self.proj_out))
        video = self.proj_out(packed).index_select(1, video_indices)
        audio = self.audio_proj_out(packed).index_select(1, audio_indices)
        if not return_dict:
            return video, audio
        return MiniMaxH3TransformerOutput(sample=video, audio_sample=audio)


def load_pipeline(
    model_path: str | None = None,
    components_path: str = OFFICIAL_REPO,
    steps: int = 8,
) -> ModularPipeline:
    if diffusers.__version__ != DIFFUSERS_VERSION:
        raise RuntimeError(
            f"MiniMax H3 Turbo requires diffusers=={DIFFUSERS_VERSION}; "
            f"found {diffusers.__version__}"
        )
    if steps not in (4, 6, 8):
        raise ValueError("steps must be 4, 6, or 8")

    component_names = list(BASE_COMPONENTS)
    if steps == 4:
        checkpoint = resolve_four_step_checkpoint(model_path, TURBO_REPO)
        component_names.append("transformer")
    else:
        if model_path is None:
            variant = f"turbo-{steps}step"
            model_root = Path(snapshot_download(
                repo_id=TURBO_REPO,
                allow_patterns=[f"{variant}/transformer/*"],
            )) / variant
        else:
            model_root = Path(model_path).expanduser()
        transformer_path = model_root / "transformer"
        if not transformer_path.is_dir():
            raise FileNotFoundError(f"Transformer directory not found: {transformer_path}")
        config = json.loads((transformer_path / "config.json").read_text())
        if config.get("fixed_num_steps") != steps:
            raise ValueError(
                f"Selected {steps}-step mode, but {transformer_path} is "
                f"a {config.get('fixed_num_steps')}-step checkpoint"
            )

    base_root = Path(components_path).expanduser()
    if not base_root.is_dir():
        base_root = Path(snapshot_download(
            repo_id=components_path,
            allow_patterns=["model_index.json", "modular_model_index.json"]
            + [f"{name}/*" for name in component_names],
        ))

    manager = ComponentsManager()
    manager.enable_auto_cpu_offload(device="cuda", memory_reserve_margin="12GB")
    pipe = ModularPipeline.from_pretrained(base_root, components_manager=manager)
    pipe.load_components(
        names=component_names,
        dtype=torch.bfloat16,
        pretrained_model_name_or_path=str(base_root),
    )
    ensure_processor_compat(pipe.processor)

    if steps == 4:
        config = apply_four_step_adapter(pipe.transformer, checkpoint)
        pipe.scheduler.set_shift(float(config["video_flow_shift"]))
        pipe.audio_scheduler.set_shift(float(config["audio_flow_shift"]))
        return pipe

    transformer = FixedMiniMaxH3Transformer3DModel.from_pretrained(
        transformer_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    index = json.loads(
        (transformer_path / "diffusion_pytorch_model.safetensors.index.json").read_text()
    )["weight_map"]
    for name in ("video_timesteps", "audio_timesteps", "visual_condition_timestep"):
        with safe_open(transformer_path / index[name], framework="pt", device="cpu") as handle:
            setattr(transformer, name, handle.get_tensor(name))
    pipe.update_components(transformer=transformer)
    return pipe
