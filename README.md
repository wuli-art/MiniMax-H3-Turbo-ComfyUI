# MiniMax H3 Turbo for ComfyUI

ComfyUI nodes for text-to-video and first/last-frame-conditioned video with
audio. The 4-step, 6-step, and 8-step checkpoints are loaded with Diffusers 0.40.0.
The generated video and audio can be saved together as one 24 fps MP4.

## Install

Clone the repository into `ComfyUI/custom_nodes/`, then install dependencies
with the same Python interpreter that runs ComfyUI:

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/wuli-art/MiniMax-H3-Turbo-ComfyUI.git
cd MiniMax-H3-Turbo-ComfyUI
python -m pip install -r requirements.txt
```

Restart ComfyUI. A CUDA GPU is required. The model and its official components
are large; the first run needs enough disk space, system memory, and time to
download and load them.

When updating an existing installation, install `requirements.txt` again before
restarting ComfyUI. It includes PEFT for the 4-step adapter and Transformers with
Qwen3-VL support for the text encoder.

## Model sources

Leave `model_root` empty to download the selected checkpoint from
[`Wuli-art/MiniMax-H3-Turbo`](https://huggingface.co/Wuli-art/MiniMax-H3-Turbo):

- `steps = 4` loads `turbo-4step/Wuli-MiniMax-H3-Turbo-4steps-V1.0-bf16.safetensors`.
- `steps = 6` loads `turbo-6step/transformer/`.
- `steps = 8` loads `turbo-8step/transformer/`.

`components_root` defaults to
[`MiniMaxAI/MiniMax-H3`](https://huggingface.co/MiniMaxAI/MiniMax-H3) for the
tokenizer, text encoder, processor, VAEs, and schedulers. In 4-step mode it also
supplies the official FL2VA `transformer/`. Downloads use the normal Hugging Face
cache. No model files are included in this node pack.

The compact 4-step checkpoint contains a backbone LoRA and four sets of fused
video/audio output projections. The node applies them to the official MiniMax H3
FL2VA transformer and selects the matching projections at each step. This file
requires the official transformer; loading it through a generic LoRA loader does
not apply the complete checkpoint. The compact download does not remove the
base model's memory requirements.

For offline use, set `components_root` to a local copy of the official MiniMax H3
components, including `transformer/` when using 4 steps. Set `model_root` according
to the selected checkpoint:

- **4 steps:** the `.safetensors` file itself, the `turbo-4step/` directory
  containing that file, or the Turbo repository root containing `turbo-4step/`.
- **6 or 8 steps:** the corresponding `turbo-6step/` or `turbo-8step/` directory
  containing `transformer/`.

The selected step count must match the checkpoint. Changing to 4 steps from an
existing local 6-step or 8-step workflow requires updating `model_root`, or
clearing it to use automatic downloads.

## Workflow

Use **MiniMax H3 Turbo Generate** and connect its `frames` and `audio` outputs
to **MiniMax H3 Turbo Save MP4**. Leave both frame inputs disconnected for
text-to-video. Connect a ComfyUI `Load Image` node to `first_frame` for
first-frame-conditioned generation; `last_frame` is optional. The save node
writes a single H264/AAC MP4 to ComfyUI's output directory at the model's
fixed 24 fps.

The `example_workflows/` directory contains:

- [`t2va_4step.json`](example_workflows/t2va_4step.json): 4-step text-to-video
  with audio; leave `model_root` empty to download the required models.
- [`fl2va_4step.json`](example_workflows/fl2va_4step.json): 4-step generation
  conditioned on a first frame, with an optional last frame.
- [`t2va_8step.json`](example_workflows/t2va_8step.json) and
  [`fl2va_8step.json`](example_workflows/fl2va_8step.json): the original 8-step
  examples. Change the `steps` widget to 6 for the 6-step model.

In either frame-conditioned example, upload/select your own first image in
`Load Image` before queuing. To condition on a last frame as well, connect a second
`Load Image` node to `last_frame`. New generate nodes still default to 8 steps.

The implementation imports Diffusers as a dependency; it does not modify
ComfyUI or Diffusers source code. The 4-step adapter loader and the 6/8-step
fixed-step transformer implementation are bundled with this node pack. MP4
export calls Diffusers' `encode_video` directly.

## Tests

With the dependencies installed, run `python -m unittest discover -s tests -v`.
The tests use a small randomly initialized H3 transformer and local synthetic
adapters; they do not download model weights or require a running ComfyUI server.
