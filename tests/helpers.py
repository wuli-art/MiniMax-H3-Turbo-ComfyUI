"""Small, deterministic fixtures that never download model weights."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys

import torch
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Transformer3DModel
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from safetensors.torch import save_file


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_plugin():
    name = "h3_turbo_test_plugin"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, REPO_ROOT / "__init__.py", submodule_search_locations=[str(REPO_ROOT)]
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def make_tiny_transformer():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        return MiniMaxH3Transformer3DModel(
            num_attention_heads=2,
            attention_head_dim=8,
            hidden_size=16,
            num_layers=1,
            num_refiner_layers=1,
            ffn_dim=24,
            in_channels=2,
            audio_in_channels=3,
            patch_size=(1, 1, 1),
            text_dim=8,
            freq_dim=8,
            time_embed_hidden_dim=16,
            time_embed_dim=8,
            rope_freq_dim=1,
        ).eval()


def tiny_transformer_inputs():
    generator = torch.Generator().manual_seed(456)
    return {
        "hidden_states": torch.randn(1, 2, 2, generator=generator),
        "audio_hidden_states": torch.randn(1, 2, 3, generator=generator),
        "encoder_hidden_states": torch.randn(1, 2, 8, generator=generator),
        "timestep": torch.tensor([0.0, 0.0]),
        "timestep_indices": torch.tensor([0, 0, 0, 0, 1, 1]),
        "token_tags": torch.tensor([1, 1, 0, 0, 2, 2]),
        "position_ids": torch.tensor([[0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 0, 1], [0, 0, 0], [1, 0, 0]]),
        "video_indices": torch.tensor([2, 3]),
        "audio_indices": torch.tensor([4, 5]),
        "text_indices": torch.tensor([0, 1]),
    }


def write_four_step_checkpoint(
    path,
    transformer,
    *,
    config_overrides=None,
    omit_keys=(),
    metadata_overrides=None,
    tensor_overrides=None,
):
    """Serialize a real PEFT adapter and nontrivial per-step output heads."""
    reference = copy.deepcopy(transformer)
    adapter_config = {
        "r": 2,
        "lora_alpha": 2,
        "lora_dropout": 0.0,
        "target_modules": ["transformer_blocks.0.attn.to_q"],
    }
    reference.add_adapter(LoraConfig(**adapter_config), adapter_name="generator")
    with torch.no_grad():
        for name, parameter in reference.named_parameters():
            if "lora_" in name:
                parameter.copy_(torch.linspace(0.01, 0.1, parameter.numel()).reshape(parameter.shape))
    tensors = {
        f"lora.generator.{key}": tensor.detach().clone().contiguous()
        for key, tensor in get_peft_model_state_dict(reference, adapter_name="generator").items()
    }
    for modality, attribute in (("video", "proj_out"), ("audio", "audio_proj_out")):
        projection = getattr(transformer, attribute)
        for step in range(4):
            tensors[f"pdd.{modality}.{step}.weight"] = (
                projection.weight.detach().clone() * (step + 1) / 8
            ).contiguous()
            tensors[f"pdd.{modality}.{step}.bias"] = (
                projection.bias.detach().clone() + (step + 1) / 10
            ).contiguous()
    tensors.update(tensor_overrides or {})
    for key in omit_keys:
        tensors.pop(key)
    config = {
        "format": "wuli_minimax_h3_turbo_v1",
        "nfe": 4,
        "grid_size": 256,
        "step_adapter_names": ["generator"] * 4,
        "video_flow_shift": 6.0,
        "audio_flow_shift": 3.0,
    }
    config.update(config_overrides or {})
    metadata = {
        "pdd_config": json.dumps(config),
        "lora_adapter_metadata": json.dumps(adapter_config),
    }
    for key, value in (metadata_overrides or {}).items():
        if value is None:
            metadata.pop(key, None)
        else:
            metadata[key] = value
    save_file(tensors, str(path), metadata=metadata)
    return tensors, config
