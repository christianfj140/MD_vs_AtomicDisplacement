from __future__ import annotations

import json
import sys
import tempfile
import unittest
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

    def test_graph2mat_prediction_stage_rejects_missing_checkpoint(self) -> None:
        stencil_root, _manifest = self.build_stencil()

        with self.assertRaisesRegex(DerivativePredictionStageError, "Graph2Mat prediction requires"):
            run_derivative_predictions(
                stencil_root=stencil_root,
                model="graph2mat",
                checkpoint=self.root / "missing.ckpt",
                existing_prediction_root=self.root / "existing",
            )

    def test_deeph_prediction_stage_rejects_missing_model_dir(self) -> None:
        stencil_root, _manifest = self.build_stencil()

        with self.assertRaisesRegex(DerivativePredictionStageError, "DeepH prediction requires"):
            run_derivative_predictions(
                stencil_root=stencil_root,
                model="deeph",
                model_dir=self.root / "missing_model",
                existing_prediction_root=self.root / "existing",
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


if __name__ == "__main__":
    unittest.main()
