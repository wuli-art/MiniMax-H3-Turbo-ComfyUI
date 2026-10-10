"""Loading and ComfyUI request regressions without downloading model weights."""

import importlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from diffusers import MiniMaxH3Scheduler
from safetensors.torch import save_file

from helpers import load_plugin


plugin = load_plugin()
backend = importlib.import_module(f"{plugin.__name__}.backend")
nodes = importlib.import_module(f"{plugin.__name__}.nodes")
four_step = importlib.import_module(f"{plugin.__name__}.four_step")


class LoadingTests(unittest.TestCase):
    def test_four_step_checkpoint_paths_and_download(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            variant = root / "turbo-4step"
            variant.mkdir()
            checkpoint = variant / four_step.FOUR_STEP_FILENAME
            checkpoint.touch()
            for local in (root, variant, checkpoint):
                with self.subTest(local=local):
                    self.assertEqual(
                        four_step.resolve_four_step_checkpoint(local, backend.TURBO_REPO), checkpoint
                    )
            with patch.object(four_step, "hf_hub_download", return_value=str(checkpoint)) as download:
                self.assertEqual(
                    four_step.resolve_four_step_checkpoint(None, backend.TURBO_REPO), checkpoint
                )
                download.assert_called_once_with(
                    backend.TURBO_REPO, four_step.FOUR_STEP_FILENAME, subfolder="turbo-4step"
                )
            with self.assertRaises(FileNotFoundError):
                four_step.resolve_four_step_checkpoint(root / "missing", backend.TURBO_REPO)

    def test_four_step_loads_official_transformer_and_sets_both_schedules(self):
        pipe = SimpleNamespace(
            transformer=object(),
            processor=SimpleNamespace(image_token_id=101, video_token_id=102),
            scheduler=MiniMaxH3Scheduler(),
            audio_scheduler=MiniMaxH3Scheduler(),
            load_components=Mock(),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / four_step.FOUR_STEP_FILENAME
            checkpoint.touch()
            with (
                patch.object(backend, "ComponentsManager"),
                patch.object(backend.ModularPipeline, "from_pretrained", return_value=pipe),
                patch.object(backend, "snapshot_download", return_value=str(root)) as download,
                patch.object(backend, "apply_four_step_adapter", return_value={
                    "video_flow_shift": 6.0, "audio_flow_shift": 3.0,
                }) as apply,
            ):
                self.assertIs(backend.load_pipeline(str(checkpoint), steps=4), pipe)
                apply.assert_called_once_with(pipe.transformer, checkpoint)
                self.assertIn("transformer/*", download.call_args.kwargs["allow_patterns"])
                self.assertEqual(download.call_args.kwargs["repo_id"], backend.OFFICIAL_REPO)
                self.assertEqual(
                    pipe.load_components.call_args.kwargs["names"],
                    [*backend.BASE_COMPONENTS, "transformer"],
                )
            self.assertEqual(pipe.scheduler.shift, 6.0)
            self.assertEqual(pipe.audio_scheduler.shift, 3.0)
            for scheduler in (pipe.scheduler, pipe.audio_scheduler):
                scheduler.set_timesteps(5)
                self.assertEqual(len(scheduler.timesteps), 4)
            self.assertEqual(
                pipe.processor.create_mm_token_type_ids([[7, 101, 102, 8]]), [[0, 1, 2, 0]]
            )

    def test_fixed_step_loading_preserves_checkpoint_and_schedule(self):
        for steps in (6, 8):
            with self.subTest(steps=steps), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                transformer_dir = root / "transformer"
                transformer_dir.mkdir()
                (transformer_dir / "config.json").write_text(json.dumps({"fixed_num_steps": steps}))
                tensors = {
                    "video_timesteps": torch.linspace(0, 1, steps),
                    "audio_timesteps": torch.linspace(0, 1, steps),
                    "visual_condition_timestep": torch.tensor(0.999),
                }
                save_file(tensors, transformer_dir / "weights.safetensors")
                (transformer_dir / "diffusion_pytorch_model.safetensors.index.json").write_text(
                    json.dumps({"weight_map": {name: "weights.safetensors" for name in tensors}})
                )
                pipe, transformer = Mock(), torch.nn.Module()
                pipe.processor = SimpleNamespace(image_token_id=101, video_token_id=102)
                with (
                    patch.object(backend, "ComponentsManager"),
                    patch.object(backend.ModularPipeline, "from_pretrained", return_value=pipe),
                    patch.object(backend.FixedMiniMaxH3Transformer3DModel, "from_pretrained",
                                 return_value=transformer),
                    patch.object(backend, "snapshot_download") as download,
                    patch.object(backend, "apply_four_step_adapter") as apply,
                ):
                    self.assertIs(backend.load_pipeline(str(root), str(root), steps), pipe)
                    download.assert_not_called()
                    apply.assert_not_called()
                    self.assertEqual(pipe.load_components.call_args.kwargs["names"], list(backend.BASE_COMPONENTS))
                    pipe.update_components.assert_called_once_with(transformer=transformer)
                    self.assertEqual(
                        pipe.processor.create_mm_token_type_ids([[7, 101, 102, 8]]), [[0, 1, 2, 0]]
                    )
                    for name, value in tensors.items():
                        torch.testing.assert_close(getattr(transformer, name), value)
                    with self.assertRaisesRegex(ValueError, "checkpoint"):
                        backend.load_pipeline(str(root), str(root), 8 if steps == 6 else 6)


class NodeTests(unittest.TestCase):
    def setUp(self):
        nodes._pipeline = None
        nodes._pipeline_key = None
        self.result = SimpleNamespace(
            videos=[torch.zeros(4, 32, 32, 3)],
            audio=[torch.zeros(2, 160)], sampling_rate=32000,
        )
        self.request = dict(
            steps=4, prompt="A paper airplane", seed=42, num_frames=4,
            height=32, width=32, model_root="", components_root="",
        )
        self.node = nodes.MiniMaxH3TurboGenerate()
        # These tests exercise routing and pipeline caching on CPU.
        self.cuda = patch.object(nodes.torch.cuda, "is_available", return_value=True)
        self.generator = patch.object(nodes.torch, "Generator")
        self.cuda.start()
        self.generator.start()
        self.addCleanup(self.cuda.stop)
        self.addCleanup(self.generator.stop)

    def tearDown(self):
        nodes._pipeline = None
        nodes._pipeline_key = None

    def test_four_step_file_and_frame_inputs_are_forwarded(self):
        pipe = Mock(return_value=self.result)
        events = []
        pipe.side_effect = lambda **kwargs: (events.append("generate") or self.result)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "adapter.safetensors"
            checkpoint.touch()
            self.request.update(model_root=str(checkpoint), first_frame=torch.zeros(1, 32, 32, 3),
                                last_frame=torch.ones(1, 32, 32, 3))
            with (
                patch.object(nodes, "load_pipeline", return_value=pipe) as load,
                patch.object(nodes, "reset_four_step_adapter", side_effect=lambda _: events.append("reset")),
                patch.object(nodes, "validate_four_step_call", side_effect=lambda _: events.append("validate")),
            ):
                frames, audio, info = self.node.generate(**self.request)
                load.assert_called_once_with(str(checkpoint), backend.OFFICIAL_REPO, 4)
                self.assertEqual(events, ["reset", "generate", "validate"])
                self.assertEqual(pipe.call_args.kwargs["num_inference_steps"], 5)
                self.assertEqual(pipe.call_args.kwargs["image"].getpixel((0, 0)), (0, 0, 0))
                self.assertEqual(pipe.call_args.kwargs["last_image"].getpixel((0, 0)), (255, 255, 255))
                self.assertEqual(tuple(frames.shape), (4, 32, 32, 3))
                self.assertEqual(tuple(audio["waveform"].shape), (1, 2, 160))
                self.assertIn("4-step", info)

    def test_cached_pipeline_is_reset_before_retry_and_each_generation(self):
        pipe = Mock(side_effect=[RuntimeError("interrupted"), self.result, self.result])
        with (
            patch.object(nodes, "load_pipeline", return_value=pipe) as load,
            patch.object(nodes, "reset_four_step_adapter") as reset,
            patch.object(nodes, "validate_four_step_call") as validate,
        ):
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                self.node.generate(**self.request)
            self.node.generate(**self.request)
            self.node.generate(**self.request)
            self.assertEqual(load.call_count, 1)
            self.assertEqual(reset.call_count, 3)
            self.assertEqual(validate.call_count, 2)

    def test_switching_steps_reloads_and_fixed_modes_do_not_use_adapter(self):
        pipe = Mock(return_value=self.result)
        with (
            patch.object(nodes, "load_pipeline", return_value=pipe) as load,
            patch.object(nodes, "reset_four_step_adapter") as reset,
            patch.object(nodes, "validate_four_step_call") as validate,
        ):
            for steps in (4, 6, 8, 4):
                self.request["steps"] = steps
                self.node.generate(**self.request)
                self.assertEqual(pipe.call_args.kwargs["num_inference_steps"], steps + 1)
            self.assertEqual(load.call_count, 4)
            self.assertEqual(reset.call_count, 2)
            self.assertEqual(validate.call_count, 2)


if __name__ == "__main__":
    unittest.main()
