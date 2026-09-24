"""ComfyUI nodes for MiniMax H3 Turbo video and audio generation."""

from __future__ import annotations

import gc
import threading
from pathlib import Path

import torch
from PIL import Image

from .backend import OFFICIAL_REPO, load_pipeline


_pipeline = None
_pipeline_key: tuple[str, str, int] | None = None
_pipeline_lock = threading.Lock()


def _to_pil(image: torch.Tensor | None) -> Image.Image | None:
    if image is None:
        return None
    if image.ndim != 4 or image.shape[0] != 1 or image.shape[-1] != 3:
        raise ValueError("First/last frame must be a single RGB ComfyUI IMAGE")
    value = image[0].detach().float().cpu().clamp(0, 1)
    value = (value * 255.0).round().to(torch.uint8).numpy()
    return Image.fromarray(value, mode="RGB")


def _video_tensor(video: object) -> torch.Tensor:
    value = torch.as_tensor(video).detach().cpu()
    if value.ndim != 4:
        raise ValueError(f"Expected video with four dimensions, got {tuple(value.shape)}")
    if value.shape[-1] in (1, 3, 4):
        result = value[..., :3]
    elif value.shape[1] in (1, 3, 4):
        result = value[:, :3].permute(0, 2, 3, 1)
    else:
        raise ValueError(f"Cannot identify video channel dimension: {tuple(value.shape)}")
    result = result.float()
    if float(result.max()) > 1.0:
        result = result / 255.0
    return result.clamp(0, 1)


def _audio_value(audio: object, sample_rate: int) -> dict[str, object]:
    value = torch.as_tensor(audio).detach().float().cpu()
    if value.ndim == 1:
        value = value[None, None]
    elif value.ndim == 2:
        value = value[None]
    elif value.ndim != 3:
        raise ValueError(f"Expected mono/stereo audio, got {tuple(value.shape)}")
    return {"waveform": value, "sample_rate": int(sample_rate)}


class MiniMaxH3TurboGenerate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "steps": ([6, 8], {"default": 8}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0x7FFFFFFF}),
                "num_frames": ("INT", {"default": 124, "min": 1, "max": 1024}),
                "height": ("INT", {"default": 768, "min": 32, "max": 2048, "step": 32}),
                "width": ("INT", {"default": 1344, "min": 32, "max": 2048, "step": 32}),
                "model_root": ("STRING", {"default": "", "multiline": False}),
                "components_root": (
                    "STRING", {"default": OFFICIAL_REPO, "multiline": False}
                ),
            },
            "optional": {
                "first_frame": ("IMAGE",),
                "last_frame": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING")
    RETURN_NAMES = ("frames", "audio", "info")
    FUNCTION = "generate"
    CATEGORY = "MiniMax H3 Turbo"

    def generate(
        self,
        steps: int,
        prompt: str,
        seed: int,
        num_frames: int,
        height: int,
        width: int,
        model_root: str,
        components_root: str,
        first_frame: torch.Tensor | None = None,
        last_frame: torch.Tensor | None = None,
    ):
        steps = int(steps)
        if steps not in (6, 8):
            raise ValueError("steps must be 6 or 8")
        if not torch.cuda.is_available():
            raise RuntimeError("MiniMax H3 Turbo requires a CUDA GPU")
        model_path = model_root.strip() or None
        if model_path is not None and not Path(model_path).expanduser().is_dir():
            raise FileNotFoundError(f"Model directory not found: {model_path}")
        components_path = components_root.strip() or OFFICIAL_REPO
        key = (model_path or "", components_path, steps)

        request = {
            "prompt": prompt,
            "num_inference_steps": steps + 1,
            "num_frames": int(num_frames),
            "height": int(height),
            "width": int(width),
            "generator": torch.Generator(device="cuda").manual_seed(int(seed)),
            "output_type": "np",
        }
        image = _to_pil(first_frame)
        last = _to_pil(last_frame)
        if image is not None:
            request["image"] = image
        if last is not None:
            request["last_image"] = last

        global _pipeline, _pipeline_key
        with _pipeline_lock:
            if _pipeline_key != key:
                _pipeline = None
                _pipeline_key = None
                gc.collect()
                _pipeline = load_pipeline(model_path, components_path, steps)
                _pipeline_key = key
            with torch.inference_mode():
                result = _pipeline(**request)

        frames = _video_tensor(result.videos[0])
        audio = _audio_value(result.audio[0], result.sampling_rate)
        info = f"MiniMax H3 Turbo {steps}-step; seed={seed}; frames={frames.shape[0]}"
        return (frames, audio, info)


class MiniMaxH3TurboSaveMP4:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "audio": ("AUDIO",),
                "filename_prefix": ("STRING", {"default": "h3_turbo"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("path",)
    FUNCTION = "save"
    CATEGORY = "MiniMax H3 Turbo"
    OUTPUT_NODE = True

    @staticmethod
    def _output_path(filename_prefix: str) -> Path:
        import folder_paths

        output_dir = Path(folder_paths.get_output_directory())
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_prefix = Path(filename_prefix or "h3_turbo").name or "h3_turbo"
        counter = 0
        while True:
            candidate = output_dir / f"{safe_prefix}_{counter:05d}.mp4"
            if not candidate.exists():
                return candidate
            counter += 1

    def save(self, frames: torch.Tensor, audio: dict[str, object], filename_prefix: str):
        from diffusers.utils import encode_video

        video = frames.detach().float().cpu().clamp(0, 1)
        if video.ndim != 4 or video.shape[-1] != 3:
            raise ValueError(f"Expected frames shaped [T,H,W,3], got {tuple(video.shape)}")
        waveform = torch.as_tensor(audio["waveform"]).detach().float().cpu()
        if waveform.ndim == 3 and waveform.shape[0] == 1:
            waveform = waveform[0]
        if waveform.ndim != 2 or waveform.shape[0] != 2:
            raise ValueError(f"Expected stereo audio shaped [2,T], got {tuple(waveform.shape)}")

        path = self._output_path(filename_prefix)
        encode_video(
            video.numpy(), fps=24, output_path=str(path),
            audio=waveform, audio_sample_rate=int(audio["sample_rate"]),
        )
        return (str(path),)


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3TurboGenerate": MiniMaxH3TurboGenerate,
    "MiniMaxH3TurboSaveMP4": MiniMaxH3TurboSaveMP4,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3TurboGenerate": "MiniMax H3 Turbo Generate",
    "MiniMaxH3TurboSaveMP4": "MiniMax H3 Turbo Save MP4",
}
