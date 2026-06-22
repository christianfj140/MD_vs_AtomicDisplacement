from __future__ import annotations

import configparser
import json
import os
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "Comparison" / "scripts"
SHARED_DIR = REPO_ROOT / "shared"
for directory in (SCRIPTS_DIR, SHARED_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from build_hamiltonian_derivative_stencils import build_derivative_stencils  # noqa: E402
from hamiltonian_derivative_stencil import discover_derivative_stencils  # noqa: E402
from run_hamiltonian_derivative_predictions import (  # noqa: E402
    DerivativePredictionStageError,
    run_derivative_predictions,
)
from deeph_prediction_adapter import DeepHPredictionAdapterResult  # noqa: E402
from predict_model_on_dataset import normalize_pattern_for_workdir  # noqa: E402


def synthetic_base_fdf() -> str:
    return "\n".join(
        [
            "SystemName synthetic base",
            "SystemLabel shared_label",
            "NumberOfSpecies 1",
            "NumberOfAtoms 2",
            "%block ChemicalSpeciesLabel",
            " 1 6 C",
            "%endblock ChemicalSpeciesLabel",
            "LatticeConstant 1.0 Ang",
            "%block LatticeVectors",
            " 8.0 0.0 0.0",
            " 0.0 8.0 0.0",
            " 0.0 0.0 8.0",
            "%endblock LatticeVectors",
            "AtomicCoordinatesFormat Ang",
            "%block AtomicCoordinatesAndAtomicSpecies",
            " 0.0 0.0 0.0 1",
            " 1.0 0.0 0.0 1",
            "%endblock AtomicCoordinatesAndAtomicSpecies",
            "",
        ]
    )


class DerivativePredictionStageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.dataset_root = self.root / "source_dataset"
        self.sample_dir = self.dataset_root / "splits" / "test" / "base_0"
        self.sample_dir.mkdir(parents=True)
        (self.sample_dir / "RUN.fdf").write_text(synthetic_base_fdf(), encoding="utf-8")
        (self.sample_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "sample_id": "base_0",
                    "material_label": "synthetic",
                    "material_compatibility_hash": "material-hash",
                    "orbital_ordering_hash": "orbital-hash",
                    "basis_hash": "basis-hash",
                    "pseudopotential_hash": "pseudo-hash",
                }
            ),
            encoding="utf-8",
        )
        (self.dataset_root / "frozen_split_manifest.json").write_text(
            json.dumps({"rows": [{"sample_id": "base_0", "split": "test", "sample_dir": str(self.sample_dir)}]}),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def build_stencil(self) -> tuple[Path, dict]:
        stencil_root = self.root / "stencils"
        manifest = build_derivative_stencils(
            source_dataset_root=self.dataset_root,
            output_stencil_root=stencil_root,
            split="test",
            method="central",
            delta_ang_values=[0.1],
            atom_indices_zero_based=[1],
            axes=["y"],
        )
        return stencil_root, manifest

    def checkpoint(self) -> Path:
        checkpoint = self.root / "training" / "lightning_logs" / "version_0" / "checkpoints" / "best.ckpt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"checkpoint")
        return checkpoint

    def write_existing_predictions(self, root: Path, samples: list[str]) -> None:
        for sample_id in samples:
            sample_dir = root / sample_id
            sample_dir.mkdir(parents=True)
            (sample_dir / "ML_prediction.HSX").write_bytes(f"prediction {sample_id}".encode("utf-8"))

    def test_staging_graph2mat_predictions_without_checkpoint_succeeds(self) -> None:
        stencil_root, _manifest = self.build_stencil()
        existing_root = self.root / "existing"
        self.write_existing_predictions(existing_root, [path.name for path in (stencil_root / "structures").iterdir()])

        result = run_derivative_predictions(
            stencil_root=stencil_root,
            model="graph2mat",
            existing_prediction_root=existing_root,
        )

        self.assertEqual(result["samples_failed"], 0)
        self.assertEqual(result["checkpoint"], "")
        self.assertTrue(all(row["status"] == "staged" for row in result["rows"]))

    def test_staging_deeph_predictions_without_model_dir_succeeds(self) -> None:
        stencil_root, _manifest = self.build_stencil()
        existing_root = self.root / "existing_deeph"
        self.write_existing_predictions(existing_root, [path.name for path in (stencil_root / "structures").iterdir()])

        result = run_derivative_predictions(
            stencil_root=stencil_root,
            model="deeph",
            existing_prediction_root=existing_root,
        )

        self.assertEqual(result["samples_failed"], 0)
        self.assertEqual(result["model_dir"], "")
        self.assertTrue(all(row["status"] == "staged" for row in result["rows"]))

    def test_graph2mat_prediction_stage_rejects_missing_checkpoint_when_running(self) -> None:
        stencil_root, _manifest = self.build_stencil()

        with self.assertRaisesRegex(DerivativePredictionStageError, "Graph2Mat prediction requires"):
            run_derivative_predictions(
                stencil_root=stencil_root,
                model="graph2mat",
                checkpoint=self.root / "missing.ckpt",
            )

    def test_graph2mat_prediction_stage_requires_basis_files_when_running(self) -> None:
        stencil_root, _manifest = self.build_stencil()

        with self.assertRaisesRegex(DerivativePredictionStageError, "--basis-files"):
            run_derivative_predictions(
                stencil_root=stencil_root,
                model="graph2mat",
                checkpoint=self.checkpoint(),
            )

    def test_repo_relative_basis_glob_is_reanchored_for_checkpoint_workdir(self) -> None:
        checkpoint = self.checkpoint()
        run_cwd = checkpoint.parents[3]
        repo_relative = "Comparison/datasets/example/material_basis/*.ion.xml"

        normalized = normalize_pattern_for_workdir(
            repo_relative,
            source_cwd=REPO_ROOT,
            target_cwd=run_cwd,
        )

        self.assertNotEqual(normalized, repo_relative)
        self.assertEqual(
            normalized,
            os.path.relpath((REPO_ROOT / repo_relative).resolve(strict=False), run_cwd).replace("\\", "/"),
        )

    def test_deeph_prediction_stage_rejects_missing_model_dir_when_running(self) -> None:
        stencil_root, _manifest = self.build_stencil()

        with self.assertRaisesRegex(DerivativePredictionStageError, "DeepH prediction requires"):
            run_derivative_predictions(
                stencil_root=stencil_root,
                model="deeph",
                deeph_command=f"{sys.executable} -c \"print('unused')\"",
            )

    def test_skip_if_exists_avoids_prediction_command(self) -> None:
        stencil_root, manifest = self.build_stencil()
        output_root = stencil_root / "predicted_hamiltonians"
        sample_ids = [record["sample_id"] for record in manifest["samples"]]
        self.write_existing_predictions(output_root, sample_ids)

        result = run_derivative_predictions(
            stencil_root=stencil_root,
            model="graph2mat",
            checkpoint=self.checkpoint(),
            output_root=output_root,
            skip_if_exists=True,
        )

        self.assertEqual(result["samples_failed"], 0)
        self.assertTrue(all(row["status"] == "skipped_existing" for row in result["rows"]))

    def test_staged_graph2mat_predictions_are_discoverable(self) -> None:
        stencil_root, manifest = self.build_stencil()
        existing_root = self.root / "existing_predictions"
        sample_ids = [record["sample_id"] for record in manifest["samples"]]
        self.write_existing_predictions(existing_root, sample_ids)

        result = run_derivative_predictions(
            stencil_root=stencil_root,
            model="graph2mat",
            checkpoint=self.checkpoint(),
            existing_prediction_root=existing_root,
        )

        self.assertEqual(result["samples_failed"], 0)
        self.assertTrue(all(row["status"] == "staged" for row in result["rows"]))
        discoveries = discover_derivative_stencils(
            stencil_root,
            method="graph2mat",
            split="test",
            finite_difference_method="central",
            require_central=True,
        )
        self.assertEqual(len(discoveries), 1)
        self.assertIsNotNone(discoveries[0].stencil)
        self.assertIsNotNone(discoveries[0].stencil.ml_plus)
        self.assertIsNotNone(discoveries[0].stencil.ml_minus)
        self.assertEqual(discoveries[0].stencil.ml_plus.matrix_path.name, "ML_prediction.HSX")

    def test_max_samples_limits_staged_prediction_samples(self) -> None:
        stencil_root, _manifest = self.build_stencil()
        existing_root = self.root / "existing_limited"
        self.write_existing_predictions(existing_root, [path.name for path in (stencil_root / "structures").iterdir()])

        result = run_derivative_predictions(
            stencil_root=stencil_root,
            model="graph2mat",
            existing_prediction_root=existing_root,
            max_samples=1,
        )

        self.assertEqual(result["samples_total"], 1)
        self.assertEqual(result["max_samples"], 1)
        self.assertIsNone(result["max_jobs"])

    def test_max_jobs_alias_limits_samples_and_is_recorded(self) -> None:
        stencil_root, _manifest = self.build_stencil()
        existing_root = self.root / "existing_alias"
        self.write_existing_predictions(existing_root, [path.name for path in (stencil_root / "structures").iterdir()])

        result = run_derivative_predictions(
            stencil_root=stencil_root,
            model="graph2mat",
            existing_prediction_root=existing_root,
            max_jobs=1,
            max_jobs_alias_used=True,
        )

        self.assertEqual(result["samples_total"], 1)
        self.assertEqual(result["max_samples"], 1)
        self.assertEqual(result["max_jobs"], 1)
        self.assertTrue(result["max_jobs_alias_used"])

    def test_deeph_command_template_runs_as_argv_by_default(self) -> None:
        stencil_root, _manifest = self.build_stencil()
        model_dir = self.root / "deeph_model"
        model_dir.mkdir()
        output_root = self.root / "deeph_predictions"
        captured = {}

        def fake_run(command, **kwargs):
            captured["command"] = command
            captured["shell"] = kwargs.get("shell")
            for structure in sorted((stencil_root / "structures").iterdir()):
                prediction_dir = output_root / structure.name
                prediction_dir.mkdir(parents=True, exist_ok=True)
                (prediction_dir / "ML_prediction.HSX").write_bytes(b"deeph")
            return mock.Mock(returncode=0, stdout="", stderr="")

        with mock.patch("run_hamiltonian_derivative_predictions.subprocess.run", side_effect=fake_run):
            result = run_derivative_predictions(
                stencil_root=stencil_root,
                model="deeph",
                output_root=output_root,
                model_dir=model_dir,
                deeph_command=f"{sys.executable} -c \"print('deeph')\"",
            )

        self.assertIsInstance(captured["command"], list)
        self.assertFalse(captured["shell"])
        self.assertEqual(result["samples_failed"], 0)
        self.assertFalse(result["deeph_shell"])

    def test_deeph_model_dir_auto_backend_writes_prediction_manifests(self) -> None:
        stencil_root, manifest = self.build_stencil()
        sample_ids = [record["sample_id"] for record in manifest["samples"]]
        siesta_root = stencil_root / "siesta_hamiltonians"
        for sample_id in sample_ids:
            sample_dir = siesta_root / sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)
            for suffix in (".HSX", ".STRUCT_OUT", ".XV", ".ORB_INDX"):
                (sample_dir / f"{sample_id}{suffix}").write_text("stub\n", encoding="utf-8")

        model_dir = self.root / "deeph_model"
        model_dir.mkdir()
        (model_dir / "config.ini").write_text("[basic]\ndisable_cuda = True\ndevice = cpu\n[graph]\nradius = -1.0\n", encoding="utf-8")
        output_root = self.root / "deeph_predictions"
        cli_dir = self.root / "deeph_bin"
        cli_dir.mkdir()
        inference_cli = cli_dir / "deeph-inference"
        preprocess_cli = cli_dir / "deeph-preprocess"
        inference_cli.write_text("#!/bin/sh\n", encoding="utf-8")
        preprocess_cli.write_text("#!/bin/sh\n", encoding="utf-8")

        def fake_raw_mirror(*, references, raw_dir):
            rows = []
            for index, sample_id in enumerate(sample_ids):
                raw_sample = raw_dir / f"{index:06d}_{sample_id}"
                raw_sample.mkdir(parents=True, exist_ok=True)
                rows.append({"sample_id": sample_id, "raw_dir": str(raw_sample)})
            return {"rows": rows}

        def fake_run(command, **kwargs):
            command_list = list(command)
            if Path(command_list[0]).name == "deeph-preprocess":
                config = configparser.ConfigParser()
                config.read(command_list[-1])
                raw_dir = Path(config.get("basic", "raw_dir"))
                processed_dir = Path(config.get("basic", "processed_dir"))
                for raw_sample in sorted(raw_dir.iterdir()):
                    sample_dir = processed_dir / raw_sample.name
                    sample_dir.mkdir(parents=True, exist_ok=True)
                    for name in ("R_list.dat", "orbital_types.dat", "element.dat", "site_positions.dat", "lat.dat", "rlat.dat", "rc.h5", "hamiltonians.h5", "overlaps.h5"):
                        (sample_dir / name).write_text("stub\n", encoding="utf-8")
                return mock.Mock(returncode=0, stdout="", stderr="")
            if Path(command_list[0]).name == "deeph-inference":
                config = configparser.ConfigParser()
                config.read(command_list[-1])
                work_dir = Path(config.get("basic", "work_dir"))
                work_dir.mkdir(parents=True, exist_ok=True)
                (work_dir / "hamiltonians_pred.h5").write_bytes(b"h5")
                return mock.Mock(returncode=0, stdout="", stderr="")
            raise AssertionError(command_list)

        def fake_adapter(*, work_dir, processed_sample_dir, sample_id, prediction_filename="hamiltonians_pred.h5"):
            return DeepHPredictionAdapterResult(
                sample_id=sample_id,
                status="ok",
                metrics_ready=True,
                diagnostic_only=True,
                diagnostic_reason="diagnostic",
                prediction_path=str(Path(work_dir) / prediction_filename),
                processed_sample_dir=str(processed_sample_dir),
                reference_hamiltonian_path=str(Path(processed_sample_dir) / "hamiltonians.h5"),
                reference_overlap_path=str(Path(processed_sample_dir) / "overlaps.h5"),
                orbital_types_path=str(Path(processed_sample_dir) / "orbital_types.dat"),
                n_orbitals=8,
                block_count=1,
                prediction_key_count=1,
                reference_key_count=1,
            )

        def fake_reconstruct(*, output_path, **_kwargs):
            output_path.write_bytes(b"npz-like")
            return {"kind": "deeph_h5_reconstructed_siesta_sparse_layout_v1", "nnz": 1, "shape_rows": 8, "shape_cols": 8}

        with mock.patch("run_hamiltonian_derivative_predictions.build_deeph_derivative_raw_mirror", side_effect=fake_raw_mirror), mock.patch(
            "run_hamiltonian_derivative_predictions.subprocess.run", side_effect=fake_run
        ), mock.patch(
            "run_hamiltonian_derivative_predictions.adapt_deeph_prediction_sample", side_effect=fake_adapter
        ), mock.patch(
            "run_hamiltonian_derivative_predictions.reconstruct_deeph_sparse_layout_prediction", side_effect=fake_reconstruct
        ):
            result = run_derivative_predictions(
                stencil_root=stencil_root,
                model="deeph",
                output_root=output_root,
                model_dir=model_dir,
                deeph_command=str(inference_cli),
            )

        self.assertEqual(result["samples_failed"], 0)
        self.assertEqual(result["samples_ok"], len(sample_ids))
        self.assertTrue((output_root / "derivative_deeph_prediction_manifest.json").exists())
        self.assertTrue((output_root / "deeph_sparse_layout_note.json").exists())
        self.assertTrue((output_root.parent / "deeph" / "inference" / "adapter_manifest.json").exists())
        for sample_id in sample_ids:
            self.assertTrue((output_root / sample_id / "ML_prediction.HSX").exists(), sample_id)


if __name__ == "__main__":
    unittest.main()
