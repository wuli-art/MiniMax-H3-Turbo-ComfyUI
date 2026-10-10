"""Compact four-step adapter for the official MiniMax H3 transformer.

The Turbo V1 checkpoint contains a shared LoRA and four output heads for each
modality. Each head predicts an integrated update, so divide it by its shifted
sigma span before using it with the Diffusers flow-matching schedulers.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import MethodType

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from torch import nn


FOUR_STEP_FILENAME = "Wuli-MiniMax-H3-Turbo-4steps-V1.0-bf16.safetensors"
FOUR_STEP_FORMAT = "wuli_minimax_h3_turbo_v1"


def resolve_four_step_checkpoint(model_path: str | Path | None, repo_id: str) -> Path:
    if model_path is None:
        return Path(hf_hub_download(repo_id, FOUR_STEP_FILENAME, subfolder="turbo-4step"))
    local = Path(model_path).expanduser()
    for candidate in (local, local / FOUR_STEP_FILENAME, local / "turbo-4step" / FOUR_STEP_FILENAME):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not find {FOUR_STEP_FILENAME} at {local}, in {local}, "
        f"or in {local / 'turbo-4step'}"
    )


def _shifted_sigma(sigma: torch.Tensor, shift: float) -> torch.Tensor:
    return shift * sigma / (1 + (shift - 1) * sigma)


class _FusedProjection(nn.Module):
    def __init__(self, weights: list[torch.Tensor], biases: list[torch.Tensor]) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.stack(weights), requires_grad=False)
        self.biases = nn.Parameter(torch.stack(biases), requires_grad=False)
        self.step = 0

    @property
    def weight(self) -> torch.Tensor:
        return self.weights[self.step]

    @property
    def bias(self) -> torch.Tensor:
        return self.biases[self.step]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden_states, self.weight, self.bias)


class _FourStepController:
    def __init__(self, transformer: nn.Module, adapter_names: list[str]) -> None:
        self.transformer = transformer
        self.adapter_names = adapter_names
        self.reset()

    def _select(self, index: int) -> None:
        self.transformer.proj_out.step = index
        self.transformer.audio_proj_out.step = index
        self.transformer.set_adapter(self.adapter_names[index])
        # PEFT's set_adapter can mark the selected adapter trainable again.
        self.transformer.requires_grad_(False)

    def reset(self) -> None:
        self.call_count = 0
        self.completed_calls = 0
        self._select(0)

    def before_forward(self, _module, _args) -> None:
        index = self.call_count
        self.call_count += 1
        if index >= len(self.adapter_names):
            raise RuntimeError(
                "The four-step adapter requires exactly four transformer calls; "
                "reset it before starting another generation"
            )
        if self.completed_calls != index:
            raise RuntimeError("Reset the four-step adapter after an interrupted transformer call")
        self._select(index)

    def after_forward(self, _module, _args, output):
        self.completed_calls += 1
        return output


def _install_four_step_lora(
    transformer: nn.Module, state: dict[str, torch.Tensor], saved: dict
) -> None:
    # Keep the existing six/eight-step loader usable without importing PEFT.
    from peft import LoraConfig
    from peft.utils import set_peft_model_state_dict

    name = "generator"
    transformer.add_adapter(
        LoraConfig(
            r=int(saved["r"]),
            lora_alpha=float(saved.get("lora_alpha") or saved["r"]),
            lora_dropout=float(saved.get("lora_dropout") or 0),
            target_modules=list(saved["target_modules"]),
        ),
        adapter_name=name,
    )
    incompatible = set_peft_model_state_dict(transformer, state, adapter_name=name)
    missing = [
        key for key in (getattr(incompatible, "missing_keys", None) or [])
        if "lora_" in key and f".{name}." in key
    ]
    unexpected = getattr(incompatible, "unexpected_keys", None) or []
    if missing or unexpected:
        raise ValueError(f"Incomplete four-step LoRA: missing={missing[:4]}, unexpected={unexpected[:4]}")


def apply_four_step_adapter(transformer: nn.Module, checkpoint: str | Path) -> dict:
    if hasattr(transformer, "_four_step_controller"):
        raise ValueError("The four-step adapter is already installed on this transformer")
    with safe_open(checkpoint, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        if "pdd_config" not in metadata or "lora_adapter_metadata" not in metadata:
            raise ValueError("The four-step checkpoint is missing required metadata")
        config = json.loads(metadata["pdd_config"])
        adapter_config = json.loads(metadata["lora_adapter_metadata"])
        keys = list(handle.keys())
        lora_state = {
            key.removeprefix("lora.generator."): handle.get_tensor(key)
            for key in keys if key.startswith("lora.generator.")
        }
        heads = {
            key.removeprefix("pdd."): handle.get_tensor(key)
            for key in keys if key.startswith("pdd.")
        }

    if config.get("format") != FOUR_STEP_FORMAT or int(config.get("nfe", 0)) != 4:
        raise ValueError("Expected a Wuli MiniMax-H3 Turbo V1 four-step checkpoint")
    grid_size = int(config["grid_size"])
    if grid_size != 256:
        raise ValueError(f"The four-step V1 checkpoint requires grid_size=256, got {grid_size}")
    adapter_names = list(config["step_adapter_names"])
    if adapter_names != ["generator"] * 4:
        raise ValueError("The four-step V1 checkpoint requires the shared generator adapter")
    if not lora_state:
        raise ValueError("The four-step checkpoint is missing the generator LoRA")

    block_size = grid_size // 4
    base_sigma = torch.linspace(1, 0, grid_size + 1, dtype=torch.float32)
    projections = {}
    for modality, attribute, shift_key in (
        ("video", "proj_out", "video_flow_shift"),
        ("audio", "audio_proj_out", "audio_flow_shift"),
    ):
        shift = float(config[shift_key])
        if not math.isfinite(shift) or shift <= 0:
            raise ValueError(f"{shift_key} must be finite and positive")
        sigma = _shifted_sigma(base_sigma, shift)
        spans = [sigma[i * block_size] - sigma[(i + 1) * block_size] for i in range(4)]
        source = getattr(transformer, attribute)
        weights, biases = [], []
        for index, span in enumerate(spans):
            weight_key, bias_key = f"{modality}.{index}.weight", f"{modality}.{index}.bias"
            if weight_key not in heads or bias_key not in heads:
                raise ValueError(f"The four-step checkpoint is missing {modality} head {index}")
            weight, bias = heads[weight_key], heads[bias_key]
            if weight.shape != source.weight.shape or source.bias is None or bias.shape != source.bias.shape:
                raise ValueError(f"{modality} fused head {index} shape does not match MiniMax-H3")
            weights.append(weight / span)
            biases.append(bias / span)
        projections[attribute] = _FusedProjection(weights, biases)

    _install_four_step_lora(transformer, lora_state, adapter_config)
    for attribute, projection in projections.items():
        setattr(transformer, attribute, projection)

    controller = _FourStepController(transformer, adapter_names)
    transformer.register_forward_pre_hook(controller.before_forward)
    transformer.register_forward_hook(controller.after_forward)
    transformer._four_step_controller = controller
    transformer.requires_grad_(False).eval()
    return config


def reset_four_step_adapter(transformer: nn.Module) -> None:
    controller = getattr(transformer, "_four_step_controller", None)
    if controller is None:
        raise RuntimeError("The four-step adapter is not installed")
    controller.reset()


def validate_four_step_call(transformer: nn.Module) -> None:
    controller = getattr(transformer, "_four_step_controller", None)
    if controller is None or controller.call_count != 4 or controller.completed_calls != 4:
        calls = controller.call_count if controller is not None else 0
        completed = controller.completed_calls if controller is not None else 0
        raise RuntimeError(
            "The pipeline did not execute exactly four transformer calls "
            f"(started={calls}, completed={completed})"
        )


def ensure_processor_compat(processor) -> None:
    if hasattr(processor, "create_mm_token_type_ids"):
        return

    def create_mm_token_type_ids(instance, batches):
        return [[
            1 if token == instance.image_token_id else
            2 if token == instance.video_token_id else 0
            for token in row
        ] for row in batches]

    processor.create_mm_token_type_ids = MethodType(create_mm_token_type_ids, processor)
