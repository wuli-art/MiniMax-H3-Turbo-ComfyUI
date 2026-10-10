"""Exercise the compact adapter with real Diffusers, PEFT, and safetensors."""

from __future__ import annotations

import importlib
from pathlib import Path
import tempfile
import unittest

import torch
import torch.nn.functional as F
from diffusers.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler
from peft.utils import get_peft_model_state_dict

from helpers import load_plugin, make_tiny_transformer, tiny_transformer_inputs, write_four_step_checkpoint


four_step = importlib.import_module(f"{load_plugin().__name__}.four_step")


class FourStepAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.checkpoint = Path(self.temporary.name) / four_step.FOUR_STEP_FILENAME
        self.transformer = make_tiny_transformer()
        self.tensors, self.config = write_four_step_checkpoint(self.checkpoint, self.transformer)
        self.inputs = tiny_transformer_inputs()

    def install(self):
        return four_step.apply_four_step_adapter(self.transformer, self.checkpoint)

    def run_four_calls(self):
        outputs = []
        with torch.no_grad():
            for _ in range(4):
                output = self.transformer(**self.inputs)
                outputs.append((output.sample.clone(), output.audio_sample.clone()))
        four_step.validate_four_step_call(self.transformer)
        return outputs

    def test_real_lora_is_loaded_and_changes_the_backbone(self):
        self.assertEqual(self.install(), self.config)
        loaded = get_peft_model_state_dict(self.transformer, adapter_name="generator")
        self.assertTrue(loaded)
        for key, value in loaded.items():
            torch.testing.assert_close(value, self.tensors[f"lora.generator.{key}"])
        projection = self.transformer.transformer_blocks[0].attn.to_q
        sample = torch.linspace(-1.0, 1.0, 16).reshape(1, 1, 16)
        with torch.no_grad():
            self.assertGreater((projection(sample) - projection.base_layer(sample)).abs().max().item(), 0.001)
        self.assertTrue(all(not parameter.requires_grad for parameter in self.transformer.parameters()))

    def test_all_heads_match_real_scheduler_updates_during_four_forwards(self):
        self.install()
        schedulers = {}
        for modality in ("video", "audio"):
            shift = self.config[f"{modality}_flow_shift"]
            scheduler = MiniMaxH3Scheduler(shift=shift)
            scheduler.set_timesteps(5)
            dense_scheduler = MiniMaxH3Scheduler(shift=shift)
            dense_scheduler.set_timesteps(257)
            torch.testing.assert_close(scheduler.sigmas, dense_scheduler.sigmas[::64])
            self.assertEqual(len(scheduler.timesteps), 4)
            schedulers[modality] = scheduler

        captured = {}
        def record(modality):
            def hook(_module, args):
                captured[modality] = args[0].detach().clone()
            return hook
        hooks = [
            self.transformer.proj_out.register_forward_pre_hook(record("video")),
            self.transformer.audio_proj_out.register_forward_pre_hook(record("audio")),
        ]
        for hook in hooks:
            self.addCleanup(hook.remove)
        self.transformer.to("cpu")
        self.assertIn("proj_out.weights", dict(self.transformer.named_parameters()))
        self.assertIn("audio_proj_out.biases", dict(self.transformer.named_parameters()))

        with torch.no_grad():
            for step in range(4):
                self.inputs["timestep"] = torch.stack([
                    schedulers["video"].timesteps[step], schedulers["audio"].timesteps[step]
                ])
                output = self.transformer(**self.inputs)
                self.assertEqual(output.sample.shape, (1, 2, 2))
                self.assertEqual(output.audio_sample.shape, (1, 2, 3))
                self.assertEqual(self.transformer.proj_out.step, step)
                self.assertEqual(self.transformer.audio_proj_out.step, step)
                self.assertTrue(all(not p.requires_grad for p in self.transformer.parameters()))
                for modality, velocity, input_name in (
                    ("video", output.sample, "hidden_states"),
                    ("audio", output.audio_sample, "audio_hidden_states"),
                ):
                    scheduler = schedulers[modality]
                    span = scheduler.sigmas[step] - scheduler.sigmas[step + 1]
                    integrated = F.linear(
                        captured[modality],
                        self.tensors[f"pdd.{modality}.{step}.weight"],
                        self.tensors[f"pdd.{modality}.{step}.bias"],
                    ).index_select(1, self.inputs[f"{modality}_indices"])
                    torch.testing.assert_close(velocity * span, integrated, rtol=2e-5, atol=2e-6)
                    initial = self.inputs[input_name]
                    updated = scheduler.step(velocity, scheduler.timesteps[step], initial).prev_sample
                    torch.testing.assert_close(updated, initial + integrated, rtol=2e-5, atol=2e-6)
                    self.inputs[input_name] = updated
        four_step.validate_four_step_call(self.transformer)

    def test_zero_or_incomplete_generations_fail_validation(self):
        self.install()
        with torch.no_grad():
            for calls in range(4):
                with self.subTest(calls=calls):
                    four_step.reset_four_step_adapter(self.transformer)
                    for _ in range(calls):
                        self.transformer(**self.inputs)
                    with self.assertRaisesRegex(RuntimeError, "exactly four"):
                        four_step.validate_four_step_call(self.transformer)

    def test_fifth_forward_is_rejected_and_reset_reproduces_outputs(self):
        self.install()
        expected = self.run_four_calls()
        with self.assertRaisesRegex(RuntimeError, "exactly four"):
            self.transformer(**self.inputs)
        with self.assertRaisesRegex(RuntimeError, "exactly four"):
            four_step.validate_four_step_call(self.transformer)
        four_step.reset_four_step_adapter(self.transformer)
        actual = self.run_four_calls()
        for expected_pair, actual_pair in zip(expected, actual):
            for expected_value, actual_value in zip(expected_pair, actual_pair):
                torch.testing.assert_close(actual_value, expected_value)

    def test_reset_recovers_from_an_interrupted_real_forward(self):
        self.install()
        expected = self.run_four_calls()
        for successful_calls in (0, 1, 2, 3):
            with self.subTest(successful_calls=successful_calls):
                four_step.reset_four_step_adapter(self.transformer)
                with torch.no_grad():
                    for _ in range(successful_calls):
                        self.transformer(**self.inputs)
                invalid = dict(self.inputs, position_ids=torch.zeros(6, 2))
                with self.assertRaisesRegex(ValueError, "position_ids"):
                    self.transformer(**invalid)
                with self.assertRaisesRegex(RuntimeError, "exactly four"):
                    four_step.validate_four_step_call(self.transformer)
                with self.assertRaises(RuntimeError):
                    self.transformer(**self.inputs)
                four_step.reset_four_step_adapter(self.transformer)
                actual = self.run_four_calls()
                for expected_pair, actual_pair in zip(expected, actual):
                    for expected_value, actual_value in zip(expected_pair, actual_pair):
                        torch.testing.assert_close(actual_value, expected_value)

    def test_invalid_metadata_is_rejected_before_adapter_installation(self):
        cases = (
            {"metadata_overrides": {"pdd_config": None}},
            {"metadata_overrides": {"lora_adapter_metadata": None}},
            {"metadata_overrides": {"pdd_config": "invalid-json"}},
            {"config_overrides": {"format": "other-format"}},
            {"config_overrides": {"nfe": 8}},
            {"config_overrides": {"grid_size": 128}},
            {"config_overrides": {"step_adapter_names": ["generator"] * 3}},
            {"config_overrides": {"video_flow_shift": 0}},
            {"config_overrides": {"audio_flow_shift": float("nan")}},
        )
        for options in cases:
            with self.subTest(options=options):
                transformer = make_tiny_transformer()
                write_four_step_checkpoint(self.checkpoint, transformer, **options)
                with self.assertRaises(ValueError):
                    four_step.apply_four_step_adapter(transformer, self.checkpoint)
                self.assertFalse(hasattr(transformer, "_four_step_controller"))
                self.assertIsInstance(transformer.proj_out, torch.nn.Linear)
                self.assertFalse(getattr(transformer, "peft_config", {}))

    def test_missing_or_malformed_heads_are_rejected(self):
        for modality in ("video", "audio"):
            for step in range(4):
                with self.subTest(modality=modality, step=step):
                    key = f"pdd.{modality}.{step}.weight"
                    for options in ({"omit_keys": [key]}, {"tensor_overrides": {key: torch.zeros(1, 1)}}):
                        transformer = make_tiny_transformer()
                        write_four_step_checkpoint(self.checkpoint, transformer, **options)
                        with self.assertRaisesRegex(ValueError, "head"):
                            four_step.apply_four_step_adapter(transformer, self.checkpoint)
                        self.assertFalse(getattr(transformer, "peft_config", {}))

    def test_incomplete_lora_is_rejected(self):
        lora_keys = [key for key in self.tensors if key.startswith("lora.generator.")]
        for missing in ([lora_keys[0]], lora_keys):
            with self.subTest(missing=missing):
                transformer = make_tiny_transformer()
                write_four_step_checkpoint(self.checkpoint, transformer, omit_keys=missing)
                with self.assertRaisesRegex(ValueError, "LoRA"):
                    four_step.apply_four_step_adapter(transformer, self.checkpoint)
                self.assertFalse(hasattr(transformer, "_four_step_controller"))

    def test_duplicate_install_and_uninstalled_reset_are_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "not installed"):
            four_step.reset_four_step_adapter(self.transformer)
        with self.assertRaisesRegex(RuntimeError, "exactly four"):
            four_step.validate_four_step_call(self.transformer)
        self.install()
        with self.assertRaisesRegex(ValueError, "already installed"):
            self.install()


if __name__ == "__main__":
    unittest.main()
