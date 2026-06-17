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
from fdf_materialization import extract_fdf_structure  # noqa: E402
from hamiltonian_derivative_stencil import discover_derivative_stencils  # noqa: E402


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


class HamiltonianDerivativeStencilBuilderTests(unittest.TestCase):
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
                    "system_label": "shared_label",
                    "material_compatibility_hash": "material-hash",
                    "orbital_ordering_hash": "orbital-hash",
                    "neighbor_list_hash": "neighbor-hash",
                    "sparsity_pattern_hash": "sparsity-hash",
                    "basis_hash": "basis-hash",
                    "pseudopotential_hash": "pseudo-hash",
                }
            ),
            encoding="utf-8",
        )
        (self.dataset_root / "frozen_split_manifest.json").write_text(
            json.dumps(
                {
                    "rows": [
                        {
                            "sample_id": "base_0",
                            "split": "test",
                            "sample_dir": str(self.sample_dir),
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_central_builder_writes_plus_minus_structures_and_discoverable_metadata(self) -> None:
        output_root = self.root / "stencils"

        manifest = build_derivative_stencils(
            source_dataset_root=self.dataset_root,
            output_stencil_root=output_root,
            split="test",
            method="central",
            delta_ang_values=[0.1],
            atom_indices_zero_based=[1],
            axes=["y"],
        )

        self.assertEqual(manifest["sample_count"], 2)
        self.assertEqual(manifest["stencil_count"], 1)
        self.assertFalse(manifest["siesta_run"])
        self.assertFalse(manifest["ml_predictions_run"])
        self.assertFalse(manifest["derivative_metrics_run"])

        records_by_sign = {record["sign_label"]: record for record in manifest["samples"]}
        plus = records_by_sign["+"]
        minus = records_by_sign["-"]
        self.assertEqual(plus["atom_index_zero_based"], 1)
        self.assertEqual(plus["axis"], "y")
        self.assertEqual(plus["delta_ang"], 0.1)

        plus_structure = extract_fdf_structure(Path(plus["run_fdf"]))
        minus_structure = extract_fdf_structure(Path(minus["run_fdf"]))
        self.assertEqual(plus_structure.positions_ang[0], (0.0, 0.0, 0.0))
        self.assertEqual(minus_structure.positions_ang[0], (0.0, 0.0, 0.0))
        self.assertEqual(plus_structure.positions_ang[1], (1.0, 0.1, 0.0))
        self.assertEqual(minus_structure.positions_ang[1], (1.0, -0.1, 0.0))

        plus_metadata = json.loads(Path(plus["metadata_path"]).read_text(encoding="utf-8"))
        minus_metadata = json.loads(Path(minus["metadata_path"]).read_text(encoding="utf-8"))
        self.assertEqual(plus_metadata["base_sample_id"], "base_0")
        self.assertEqual(plus_metadata["sign_label"], "+")
        self.assertEqual(minus_metadata["sign_label"], "-")
        self.assertEqual(plus_metadata["axis_index"], 1)
        self.assertEqual(plus_metadata["finite_difference_method"], "central")
        self.assertEqual(plus_metadata["split"], "test")
        self.assertEqual(plus_metadata["hamiltonian_units"], "eV")
        self.assertEqual(plus_metadata["displacement_units"], "Ang")
        self.assertEqual(plus_metadata["derivative_units"], "eV/Ang")
        self.assertEqual(plus_metadata["split_group_id"], minus_metadata["split_group_id"])

        discoveries = discover_derivative_stencils(
            output_root,
            method="graph2mat",
            split="test",
            finite_difference_method="central",
            require_central=True,
        )

        self.assertEqual(len(discoveries), 1)
        self.assertEqual(discoveries[0].method, "central")
        self.assertEqual(discoveries[0].status, "incomplete")
        self.assertIsNotNone(discoveries[0].stencil)
        self.assertEqual(discoveries[0].stencil.metadata.plus_sample_id, plus["sample_id"])
        self.assertEqual(discoveries[0].stencil.metadata.minus_sample_id, minus["sample_id"])


if __name__ == "__main__":
    unittest.main()
