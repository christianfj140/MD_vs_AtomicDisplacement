"""Tests for pipeline_ui's runtime backend-provenance manifest.

A static source-code scanner (audit_epc_backends.py) can only prove this
pipeline is *capable* of recording requested/effective backend info -- it
cannot prove a real run produced verifiable evidence. These tests call
write_backend_runtime_provenance() directly (with detect_hardware()'s real,
un-mocked probe of *this* machine) so the manifest asserted on here is
genuinely produced by running code, not fabricated after the fact.
"""

from __future__ import annotations

import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "Comparison" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

pipeline_ui = importlib.import_module("pipeline_ui")


class BackendRuntimeProvenanceManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.config_path = Path(self.tmp.name) / "pipeline_config.yaml"
        self.config_path.write_text("paths: {}\n", encoding="utf-8")
        # Real hardware probe of this machine -- nvidia-smi + torch.cuda.is_available(),
        # not a synthetic/mocked value. Whatever this machine actually has, honestly.
        self.hardware = pipeline_ui.detect_hardware()

    def test_manifest_written_to_a_real_findable_path_next_to_config(self) -> None:
        out_path = pipeline_ui.write_backend_runtime_provenance(
            self.config_path,
            requested_backend="auto",
            effective_backend="gpu" if self.hardware.get("cuda_available") else "cpu",
            device="cuda:0" if self.hardware.get("cuda_available") else "cpu",
            hardware=self.hardware,
            performance={"torch_mixed_precision": "bf16-mixed", "torch_float32_matmul_precision": "high"},
            fallback_reason=None if self.hardware.get("cuda_available") else "cuda_unavailable: detect_hardware() reported cuda_available=False",
        )
        self.assertEqual(out_path, self.config_path.parent / "backend_runtime_provenance.json")
        self.assertTrue(out_path.is_file())

    def test_happy_path_has_all_required_fields_with_real_values(self) -> None:
        effective = "gpu" if self.hardware.get("cuda_available") else "cpu"
        out_path = pipeline_ui.write_backend_runtime_provenance(
            self.config_path,
            requested_backend="auto",
            effective_backend=effective,
            device="cuda:0" if effective == "gpu" else "cpu",
            hardware=self.hardware,
            performance={"torch_mixed_precision": "bf16-mixed", "torch_float32_matmul_precision": "high"},
            fallback_reason=None if effective == "gpu" else "cuda_unavailable: detect_hardware() reported cuda_available=False",
        )
        manifest = json.loads(out_path.read_text(encoding="utf-8"))

        self.assertEqual(manifest["schema"], "backend_runtime_provenance_v1")
        self.assertTrue(manifest["generated_at"])
        self.assertEqual(manifest["requested_backend"], "auto")
        self.assertEqual(manifest["effective_backend"], effective)
        self.assertIn(manifest["device"], {"cuda:0", "cpu"})

        # precision: real values from the performance dict, not None/placeholder.
        self.assertEqual(manifest["precision"]["torch_mixed_precision"], "bf16-mixed")
        self.assertEqual(manifest["precision"]["torch_float32_matmul_precision"], "high")

        # preflight: real detect_hardware() evidence for this machine, not fabricated.
        preflight = manifest["preflight"]
        self.assertEqual(preflight["cuda_available"], bool(self.hardware.get("cuda_available")))
        self.assertIn("device_name", preflight)
        self.assertIn("notes", preflight)

        if effective == "gpu":
            self.assertIsNone(manifest["backend_fallback_reason"])
            self.assertIsNotNone(preflight["device_name"])
        else:
            self.assertIsNotNone(manifest["backend_fallback_reason"])

    def test_cpu_only_style_request_records_fallback_reason_as_deliberate_not_real(self) -> None:
        # Mirrors performance_preset_catalog()'s cpu_only preset: requested_backend
        # is explicitly "cpu", and backend_fallback_reason still gets recorded even
        # though this is a deliberate choice, not hardware forcing a real fallback.
        out_path = pipeline_ui.write_backend_runtime_provenance(
            self.config_path,
            requested_backend="cpu",
            effective_backend="cpu",
            device="cpu",
            hardware=self.hardware,
            performance={},
            fallback_reason="cpu requested: training.trainer.accelerator is 'cpu' (explicit or default), not a GPU fallback",
        )
        manifest = json.loads(out_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["requested_backend"], "cpu")
        self.assertEqual(manifest["effective_backend"], "cpu")
        self.assertIsNotNone(manifest["backend_fallback_reason"])
        self.assertIn("not a GPU fallback", manifest["backend_fallback_reason"])

    def test_missing_precision_tracking_is_recorded_as_explicit_gap_not_omitted(self) -> None:
        out_path = pipeline_ui.write_backend_runtime_provenance(
            self.config_path,
            requested_backend="cpu",
            effective_backend="cpu",
            device="cpu",
            hardware=self.hardware,
            performance={},  # no torch_mixed_precision / torch_float32_matmul_precision tracked
            fallback_reason=None,
        )
        manifest = json.loads(out_path.read_text(encoding="utf-8"))
        precision = manifest["precision"]
        self.assertIsNone(precision["value"])
        self.assertEqual(precision["reason"], "not_tracked_for_this_run")

    def test_manifest_path_helper_matches_write_output_location(self) -> None:
        expected = pipeline_ui.backend_runtime_provenance_path(self.config_path)
        out_path = pipeline_ui.write_backend_runtime_provenance(
            self.config_path,
            requested_backend="cpu",
            effective_backend="cpu",
            device="cpu",
            hardware=self.hardware,
            performance={},
            fallback_reason=None,
        )
        self.assertEqual(expected, out_path)


if __name__ == "__main__":
    unittest.main()
