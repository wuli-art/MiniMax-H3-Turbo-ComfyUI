# MiniMax H3 Turbo for ComfyUI

ComfyUI nodes for text-to-video and first/last-frame-conditioned video with
audio. The 6-step and 8-step checkpoints are loaded with Diffusers 0.40.0.
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

## Model sources

Leave `model_root` empty to download the selected checkpoint from
[`Wuli-art/MiniMax-H3-Turbo`](https://huggingface.co/Wuli-art/MiniMax-H3-Turbo):

- `steps = 6` loads `turbo-6step/transformer/`.
- `steps = 8` loads `turbo-8step/transformer/`.

`components_root` defaults to
[`MiniMaxAI/MiniMax-H3`](https://huggingface.co/MiniMaxAI/MiniMax-H3) for the
tokenizer, text encoder, processor, VAEs, and schedulers. Both downloads use
the normal Hugging Face cache. No model files are included in this node pack.

For offline use, set `model_root` to a local `turbo-6step` or `turbo-8step`
directory containing `transformer/`, and set `components_root` to a local
copy of the official MiniMax H3 components. The selected step count must match
the checkpoint.

## Workflow

Use **MiniMax H3 Turbo Generate** and connect its `frames` and `audio` outputs
to **MiniMax H3 Turbo Save MP4**. Leave both frame inputs disconnected for
text-to-video. Connect a ComfyUI `Load Image` node to `first_frame` for
first-frame-conditioned generation; `last_frame` is optional. The save node
writes a single H264/AAC MP4 to ComfyUI's output directory at the model's
fixed 24 fps.

The `example_workflows/` directory contains an 8-step text-to-video workflow
and a frame-conditioned workflow. Change the `steps` widget to 6 for the
6-step model. In the frame-conditioned example, upload/select your own first
image in `Load Image` before queuing.

The implementation imports Diffusers as a dependency; it does not modify
ComfyUI or Diffusers source code. The fixed-step transformer implementation
is bundled with this node pack, and MP4 export calls Diffusers' `encode_video`
directly.
