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

from build_hamiltonian_derivative_stencils import DerivativeStencilBuildError, build_derivative_stencils  # noqa: E402
from fdf_materialization import extract_fdf_structure  # noqa: E402
from hamiltonian_derivative_stencil import discover_derivative_stencils, validate_derivative_geometry  # noqa: E402


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

    def write_frozen_test_samples(self, count: int) -> None:
        rows = []
        for index in range(count):
            sample_id = f"base_{index}"
            sample_dir = self.dataset_root / "splits" / "test" / sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)
            (sample_dir / "RUN.fdf").write_text(synthetic_base_fdf(), encoding="utf-8")
            (sample_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "sample_id": sample_id,
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
            rows.append({"sample_id": sample_id, "split": "test", "sample_dir": str(sample_dir)})
        (self.dataset_root / "frozen_split_manifest.json").write_text(
            json.dumps({"rows": rows}),
            encoding="utf-8",
        )

    def test_central_builder_writes_base_plus_minus_structures_and_discoverable_metadata(self) -> None:
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

        self.assertEqual(manifest["sample_count"], 3)
        self.assertEqual(manifest["stencil_count"], 1)
        self.assertTrue(manifest["include_base"])
        self.assertFalse(manifest["siesta_run"])
        self.assertFalse(manifest["ml_predictions_run"])
        self.assertFalse(manifest["derivative_metrics_run"])

        records_by_sign = {record["sign_label"]: record for record in manifest["samples"]}
        base = records_by_sign["0"]
        plus = records_by_sign["+"]
        minus = records_by_sign["-"]
        self.assertEqual(base["sample_id"], "base_0_base")
        self.assertEqual(base["delta_ang"], 0.0)
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
        self.assertEqual(discoveries[0].stencil.base_structure_path.parent.name, base["sample_id"])
        self.assertEqual(validate_derivative_geometry(discoveries[0]), [])

    def test_stencil_run_fdfs_are_single_point_even_when_base_has_md(self) -> None:
        # Regression: MD settings inherited from the dataset fdf made SIESTA evolve
        # the geometry before writing TSHS, invalidating the FD reference.
        (self.sample_dir / "RUN.fdf").write_text(
            synthetic_base_fdf() + "MD.TypeOfRun Verlet\nMD.Steps 20\nLua.Script md_store.lua\n",
            encoding="utf-8",
        )
        manifest = build_derivative_stencils(
            source_dataset_root=self.dataset_root,
            output_stencil_root=self.root / "stencils_md",
            split="test",
            method="central",
            delta_ang_values=[0.1],
            atom_indices_zero_based=[1],
            axes=["y"],
        )

        self.assertEqual(manifest["sample_count"], 3)
        for record in manifest["samples"]:
            text = Path(record["run_fdf"]).read_text(encoding="utf-8")
            self.assertNotIn("Verlet", text)
            self.assertNotIn("Lua.Script", text)
            self.assertIn("MD.NumCGsteps", text)

    def test_adaptive_min_fraction_selects_expected_base_counts(self) -> None:
        cases = [(10, 10), (20, 20), (30, 20), (80, 20), (110, 22)]

        for n_available, expected in cases:
            with self.subTest(n_available=n_available):
                self.write_frozen_test_samples(n_available)
                manifest = build_derivative_stencils(
                    source_dataset_root=self.dataset_root,
                    output_stencil_root=self.root / f"adaptive_{n_available}",
                    split="test",
                    method="central",
                    delta_ang_values=[0.1],
                    atom_indices_zero_based=[0],
                    axes=["x"],
                    base_selection_policy="adaptive_min_fraction",
                    min_base_snapshots=20,
                    base_fraction=0.2,
                )

                self.assertEqual(manifest["available_base_snapshot_count"], n_available)
                self.assertEqual(manifest["selected_base_snapshot_count"], expected)
                self.assertEqual(manifest["base_snapshots"], [f"base_{index}" for index in range(expected)])
                self.assertEqual(manifest["selected_base_snapshot_ids"], manifest["base_snapshots"])
                self.assertEqual(manifest["stencil_count"], expected)
                self.assertEqual(manifest["selection_mode"], "deterministic_ordered")

    def test_derivative_cost_fields_report_expected_structure_count_for_real_graphene_case(self) -> None:
        self.write_frozen_test_samples(110)

        manifest = build_derivative_stencils(
            source_dataset_root=self.dataset_root,
            output_stencil_root=self.root / "adaptive_cost",
            split="test",
            method="central",
            delta_ang_values=[0.005, 0.01],
            atom_indices_zero_based=[0, 1],
            axes=["x", "y", "z"],
            base_selection_policy="adaptive_min_fraction",
            min_base_snapshots=20,
            base_fraction=0.2,
            include_base=True,
        )

        self.assertEqual(manifest["selected_base_snapshot_count"], 22)
        self.assertEqual(manifest["stencils_per_base_snapshot"], 12)
        self.assertEqual(manifest["expected_structures_per_base_snapshot"], 25)
        self.assertEqual(manifest["expected_total_structure_samples"], 550)
        self.assertEqual(manifest["sample_count"], 550)

    def test_adaptive_min_fraction_rejects_invalid_selection_parameters(self) -> None:
        cases = [
            ({"base_fraction": 0}, "--base-fraction"),
            ({"base_fraction": 1.1}, "--base-fraction"),
            ({"min_base_snapshots": 0}, "--min-base-snapshots"),
            ({"max_base_snapshots": 1}, "--max-base-snapshots cannot be combined"),
            ({"base_selection_policy": "unknown"}, "--base-selection-policy"),
        ]

        for index, (overrides, error) in enumerate(cases):
            with self.subTest(error=error):
                kwargs = {
                    "source_dataset_root": self.dataset_root,
                    "output_stencil_root": self.root / f"invalid_adaptive_{index}",
                    "split": "test",
                    "method": "central",
                    "delta_ang_values": [0.1],
                    "atom_indices_zero_based": [0],
                    "axes": ["x"],
                    "base_selection_policy": "adaptive_min_fraction",
                    "min_base_snapshots": 20,
                    "base_fraction": 0.2,
                    **overrides,
                }
                with self.assertRaisesRegex(DerivativeStencilBuildError, error):
                    build_derivative_stencils(**kwargs)


if __name__ == "__main__":
    unittest.main()
