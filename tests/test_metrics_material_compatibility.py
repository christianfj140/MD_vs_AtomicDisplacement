from __future__ import annotations

import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "Comparison" / "scripts"


def load_metrics_module(name: str = "evaluate_hamiltonian_metrics_material_compat"):
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / "evaluate_hamiltonian_metrics.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def synthetic_fdf(path: Path, *, structure_type: str | None = None, kpoints: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "SystemLabel sic",
                "%block ChemicalSpeciesLabel",
                "1 14 Si",
                "2 6 C",
                "%endblock ChemicalSpeciesLabel",
                "%block AtomicCoordinatesAndAtomicSpecies",
                "0.0 0.0 0.0 1",
                "1.8 0.0 0.0 2",
                "%endblock AtomicCoordinatesAndAtomicSpecies",
                *(
                    [
                        "%block kgrid_Monkhorst_Pack",
                        "2 0 0 0.0",
                        "0 2 0 0.0",
                        "0 0 2 0.0",
                        "%endblock kgrid_Monkhorst_Pack",
                    ]
                    if kpoints
                    else []
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )
    if structure_type:
        (path.parent / "metadata.json").write_text(
            json.dumps({"material": {"structure_type": structure_type}}) + "\n",
            encoding="utf-8",
        )


def synthetic_h_fdf(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "SystemLabel h2",
                "%block ChemicalSpeciesLabel",
                "1 1 H",
                "%endblock ChemicalSpeciesLabel",
                "%block AtomicCoordinatesAndAtomicSpecies",
                "0.0 0.0 0.0 1",
                "1.0 0.0 0.0 1",
                "%endblock AtomicCoordinatesAndAtomicSpecies",
                "",
            ]
        ),
        encoding="utf-8",
    )


class MetricsMaterialCompatibilityTests(unittest.TestCase):
    def setUp(self) -> None:
        try:
            import numpy as np  # noqa: F401
            from scipy import sparse  # noqa: F401
        except ModuleNotFoundError as exc:
            self.skipTest(f"scientific Python dependency unavailable: {exc.name}")
        self.module = load_metrics_module()

    def matrix_data(self, values, *, overlap=None, orthogonal=True, component_count=1, spin_kind=None):
        import numpy as np
        from scipy import sparse

        matrix = sparse.csr_matrix(values)
        return self.module.MatrixData(
            path=Path("synthetic.HSX"),
            hamiltonian=matrix,
            overlap=overlap,
            own_eigenvalues=np.asarray([], dtype=float),
            fermi_level=0.0,
            fermi_level_source="siesta_file",
            orthogonal=orthogonal,
            has_overlap=overlap is not None,
            overlap_error=None if overlap is not None else "missing overlap",
            component_count=component_count,
            spin_kind=spin_kind,
        )

    def test_non_h2o_structural_metrics_pass_with_matching_basis(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fdf = Path(tmp) / "sample" / "RUN.fdf"
            synthetic_fdf(fdf)
            reference = self.matrix_data([[1.0, 0.2], [0.2, 2.0]])
            predicted = self.matrix_data([[1.1, 0.0], [0.4, 1.7]])

            structural = self.module.structural_sparse_metrics(
                "sample",
                reference,
                predicted,
                fdf,
                {"Si": 1, "C": 1},
            )

        self.assertTrue(structural["available"])
        self.assertEqual(structural["warnings"], [])
        self.assertTrue(structural["distance_bin_rows"])
        self.assertIn("Si-C", {row["species_pair"] for row in structural["species_pair_rows"]})

    def test_orbital_pair_metrics_group_same_species_local_indices(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fdf = Path(tmp) / "sample" / "RUN.fdf"
            synthetic_h_fdf(fdf)
            reference = self.matrix_data(
                [
                    [1.0, 0.0, 2.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [3.0, 0.0, 4.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            )
            predicted = self.matrix_data(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [5.0, 0.0, 4.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            )

            structural = self.module.structural_sparse_metrics(
                "sample",
                reference,
                predicted,
                fdf,
                {"H": 2},
            )

        rows = structural["orbital_pair_rows"]
        required_columns = {
            "sample",
            "row_species",
            "col_species",
            "species_pair",
            "row_orbital_index",
            "col_orbital_index",
            "row_orbital_label",
            "col_orbital_label",
            "n_entries",
            "mae_union_eV",
            "mae_union_meV",
            "mse_union_eV2",
            "rmse_union_eV",
            "r2_union",
            "max_abs_error_union_eV",
            "mean_abs_ref_eV",
            "mean_signed_error_eV",
            "metric_target_space",
            "basis_source",
        }
        self.assertTrue(rows)
        self.assertTrue(required_columns.issubset(rows[0]))
        target = next(
            row
            for row in rows
            if row["row_species"] == "H"
            and row["col_species"] == "H"
            and row["row_orbital_index"] == 0
            and row["col_orbital_index"] == 0
        )
        self.assertEqual(target["n_entries"], 4)
        self.assertEqual(target["row_orbital_label"], "orbital_0")
        self.assertEqual(target["col_orbital_label"], "orbital_0")
        self.assertAlmostEqual(target["mae_union_eV"], 1.0)
        self.assertAlmostEqual(target["mae_union_meV"], 1000.0)
        self.assertAlmostEqual(target["mse_union_eV2"], 2.0)
        self.assertAlmostEqual(target["rmse_union_eV"], math.sqrt(2.0))
        self.assertAlmostEqual(target["r2_union"], -0.6)
        self.assertAlmostEqual(target["max_abs_error_union_eV"], 2.0)
        self.assertAlmostEqual(target["mean_abs_ref_eV"], 2.5)
        self.assertAlmostEqual(target["mean_signed_error_eV"], 0.0)
        self.assertEqual(target["metric_target_space"], self.module.ORBITAL_PAIR_METRIC_TARGET_SPACE)
        self.assertEqual(target["basis_source"], self.module.ORBITAL_PAIR_BASIS_SOURCE)
        self.assertTrue(structural["block_rows"])
        self.assertTrue(structural["species_pair_rows"])
        self.assertTrue(structural["distance_bin_rows"])
        self.assertEqual(
            set(structural["species_pair_rows"][0]),
            {
                "sample",
                "species_pair",
                "n_entries",
                "mae_union_eV",
                "rmse_union_eV",
                "max_abs_error_union_eV",
                "mean_distance_ang",
            },
        )

    def test_orbital_pair_metrics_keep_same_local_index_species_separate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fdf = Path(tmp) / "sample" / "RUN.fdf"
            synthetic_fdf(fdf)
            reference = self.matrix_data(
                [
                    [1.0, 0.5, 0.7],
                    [0.0, 2.0, 0.0],
                    [0.2, 0.0, 3.0],
                ]
            )
            predicted = self.matrix_data(
                [
                    [1.1, 0.4, 0.0],
                    [0.0, 2.2, 0.0],
                    [0.5, 0.0, 2.8],
                ]
            )

            structural = self.module.structural_sparse_metrics(
                "sample",
                reference,
                predicted,
                fdf,
                {"Si": 2, "C": 1},
            )

        keys = {
            (
                row["row_species"],
                row["col_species"],
                row["row_orbital_index"],
                row["col_orbital_index"],
            )
            for row in structural["orbital_pair_rows"]
        }
        self.assertIn(("Si", "Si", 0, 0), keys)
        self.assertIn(("Si", "Si", 1, 1), keys)
        self.assertIn(("Si", "C", 0, 0), keys)
        self.assertIn(("C", "Si", 0, 0), keys)
        self.assertIn(("C", "C", 0, 0), keys)

    def test_orbital_pair_summary_aggregates_by_species_and_local_pair(self) -> None:
        rows = [
            {
                "sample": "a",
                "row_species": "H",
                "col_species": "H",
                "species_pair": "H-H",
                "row_orbital_index": 0,
                "col_orbital_index": 0,
                "row_orbital_label": "orbital_0",
                "col_orbital_label": "orbital_0",
                "n_entries": 4,
                "mae_union_eV": 1.0,
                "mae_union_meV": 1000.0,
                "mse_union_eV2": 2.0,
                "rmse_union_eV": 2.0,
                "r2_union": 0.5,
                "max_abs_error_union_eV": 3.0,
                "mean_abs_ref_eV": 4.0,
                "mean_signed_error_eV": -1.0,
                "metric_target_space": self.module.ORBITAL_PAIR_METRIC_TARGET_SPACE,
                "basis_source": self.module.ORBITAL_PAIR_BASIS_SOURCE,
            },
            {
                "sample": "b",
                "row_species": "H",
                "col_species": "H",
                "species_pair": "H-H",
                "row_orbital_index": 0,
                "col_orbital_index": 0,
                "row_orbital_label": "orbital_0",
                "col_orbital_label": "orbital_0",
                "n_entries": 2,
                "mae_union_eV": 3.0,
                "mae_union_meV": 3000.0,
                "mse_union_eV2": 6.0,
                "rmse_union_eV": 4.0,
                "r2_union": 1.0,
                "max_abs_error_union_eV": 5.0,
                "mean_abs_ref_eV": 6.0,
                "mean_signed_error_eV": 1.0,
                "metric_target_space": self.module.ORBITAL_PAIR_METRIC_TARGET_SPACE,
                "basis_source": self.module.ORBITAL_PAIR_BASIS_SOURCE,
            },
        ]

        summary = self.module.orbital_pair_summary_rows(rows)

        self.assertEqual(len(summary), 1)
        row = summary[0]
        self.assertEqual(row["n_samples"], 2)
        self.assertEqual(row["n_entries"], 6)
        self.assertAlmostEqual(row["mae_union_eV_mean"], 2.0)
        self.assertAlmostEqual(row["mae_union_eV_std"], 1.0)
        self.assertAlmostEqual(row["mae_union_eV_min"], 1.0)
        self.assertAlmostEqual(row["mae_union_eV_max"], 3.0)
        self.assertAlmostEqual(row["r2_union_mean"], 0.75)

    def test_missing_basis_species_fails_structural_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fdf = Path(tmp) / "sample" / "RUN.fdf"
            synthetic_fdf(fdf)
            reference = self.matrix_data([[1.0, 0.0], [0.0, 2.0]])
            predicted = self.matrix_data([[1.0, 0.0], [0.0, 2.0]])

            with self.assertRaisesRegex(RuntimeError, "Missing .ion.xml basis for species: C"):
                self.module.structural_sparse_metrics(
                    "sample",
                    reference,
                    predicted,
                    fdf,
                    {"Si": 1},
                )

    def test_periodic_distance_bins_are_marked_unsupported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fdf = Path(tmp) / "sample" / "RUN.fdf"
            synthetic_fdf(fdf, structure_type="crystal")
            reference = self.matrix_data([[1.0, 0.2], [0.2, 2.0]])
            predicted = self.matrix_data([[1.1, 0.0], [0.4, 1.7]])

            structural = self.module.structural_sparse_metrics(
                "sample",
                reference,
                predicted,
                fdf,
                {"Si": 1, "C": 1},
            )

        self.assertTrue(structural["available"])
        self.assertEqual(structural["distance_bin_rows"], [])
        self.assertIn("Periodic distance-bin", structural["distance_unavailable_reason"])
        self.assertIn("unsupported_periodic_distance_bins", {warning["kind"] for warning in structural["warnings"]})

    def test_matrix_compatibility_rejects_complex_spin_and_missing_overlap(self) -> None:
        import numpy as np
        from scipy import sparse

        complex_reference = self.matrix_data([[1.0 + 1.0j, 0.0], [0.0, 2.0]])
        predicted = self.matrix_data([[1.0, 0.0], [0.0, 2.0]])
        complex_errors = self.module.matrix_compatibility_errors("complex", complex_reference, predicted)
        self.assertIn("unsupported_complex_hamiltonian", {error["kind"] for error in complex_errors})

        spin_reference = self.matrix_data([[1.0, 0.0], [0.0, 2.0]], spin_kind="Spin{polarized}")
        spin_errors = self.module.matrix_compatibility_errors("spin", spin_reference, predicted)
        self.assertIn("unsupported_spin_kind", {error["kind"] for error in spin_errors})

        nonorthogonal = self.matrix_data([[1.0, 0.0], [0.0, 2.0]], orthogonal=False, overlap=None)
        overlap_errors = self.module.matrix_compatibility_errors("overlap", nonorthogonal, predicted)
        self.assertIn("missing_required_overlap", {error["kind"] for error in overlap_errors})

        valid_nonorthogonal = self.matrix_data(
            [[1.0, 0.0], [0.0, 2.0]],
            orthogonal=False,
            overlap=sparse.eye(2, format="csr"),
            spin_kind="Spin{unpolarized}",
        )
        valid_predicted = self.matrix_data(
            [[1.1, 0.0], [0.0, 2.1]],
            orthogonal=False,
            overlap=sparse.eye(2, format="csr"),
            spin_kind="Spin{unpolarized}",
        )
        self.assertEqual(
            self.module.matrix_compatibility_errors("valid", valid_nonorthogonal, valid_predicted),
            [],
        )

    def test_kpoint_sampled_structure_is_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fdf = Path(tmp) / "sample" / "RUN.fdf"
            synthetic_fdf(fdf, kpoints=True)

            issues = self.module.unsupported_kpoint_issues("sample", fdf)

        self.assertTrue(issues)
        self.assertEqual(issues[0]["kind"], "unsupported_kpoint_sampling")
        self.assertEqual(issues[0]["severity"], "fatal")


if __name__ == "__main__":
    unittest.main()
