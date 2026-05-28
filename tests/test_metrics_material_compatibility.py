from __future__ import annotations

import importlib.util
import json
import math
import sys
import tempfile
import unittest
from unittest import mock
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

    def test_write_csv_extends_fieldnames_for_provenance_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "metrics.csv"
            self.module.write_csv(
                output,
                ["sample", "metric"],
                [
                    {
                        "sample": "001",
                        "metric": 1.0,
                        "prediction_self_contained_hsx_safe": False,
                        "metrics_schema_version": "h_only_sref_v2",
                    }
                ],
            )
            header = output.read_text(encoding="utf-8").splitlines()[0].split(",")

        self.assertEqual(header[:2], ["sample", "metric"])
        self.assertIn("metrics_schema_version", header)
        self.assertIn("prediction_self_contained_hsx_safe", header)

    def matrix_data(
        self,
        values,
        *,
        overlap=None,
        orthogonal=True,
        component_count=1,
        spin_kind=None,
        components=None,
    ):
        import numpy as np
        from scipy import sparse

        matrix = sparse.csr_matrix(values)
        component_matrices = (
            tuple(sparse.csr_matrix(component) for component in components)
            if components is not None
            else ()
        )
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
            components=component_matrices,
        )

    def write_minimal_kpoint_result(self, root: Path, *, mesh: tuple[int, int, int] = (2, 1, 1)) -> None:
        structure_dir = root / "structures" / "001"
        prediction_dir = root / "predicted_hamiltonians" / "001"
        reference_dir = root / "siesta_hamiltonians" / "001"
        structure_dir.mkdir(parents=True)
        prediction_dir.mkdir(parents=True)
        reference_dir.mkdir(parents=True)
        structure_dir.joinpath("RUN.fdf").write_text(
            "\n".join(
                [
                    "%block kgrid_Monkhorst_Pack",
                    f"{mesh[0]} 0 0 0.0",
                    f"0 {mesh[1]} 0 0.0",
                    f"0 0 {mesh[2]} 0.0",
                    "%endblock kgrid_Monkhorst_Pack",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        prediction_dir.joinpath("ML_prediction.HSX").write_bytes(b"prediction")
        reference_dir.joinpath("siesta.TSHS").write_bytes(b"reference")

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

    def test_matrix_semantics_records_h_only_and_reference_overlap_policy(self) -> None:
        from scipy import sparse

        reference = self.matrix_data(
            [[1.0, 0.0], [0.0, 2.0]],
            orthogonal=False,
            overlap=sparse.eye(2, format="csr"),
            spin_kind="Spin{unpolarized}",
        )
        predicted = self.matrix_data(
            [[1.1, 0.0], [0.0, 2.1]],
            orthogonal=False,
            overlap=sparse.eye(2, format="csr"),
            spin_kind="Spin{unpolarized}",
        )

        semantics = self.module.matrix_semantics_fields(
            reference,
            predicted,
            target_component_policy="h_only",
        )

        self.assertEqual(semantics["metrics_schema_version"], self.module.METRICS_SCHEMA_VERSION)
        self.assertEqual(
            semantics["metrics_provenance_generation"],
            self.module.METRICS_PROVENANCE_GENERATION,
        )
        self.assertEqual(semantics["target_component_policy"], "h_only")
        self.assertEqual(semantics["reference_component_count"], 1)
        self.assertEqual(semantics["prediction_component_count"], 1)
        self.assertEqual(semantics["reference_spin_kind"], "Spin{unpolarized}")
        self.assertEqual(semantics["prediction_spin_kind"], "Spin{unpolarized}")
        self.assertEqual(semantics["overlap_source"], "siesta_reference")
        self.assertFalse(semantics["prediction_own_overlap_used"])
        self.assertFalse(semantics["graph2mat_auxiliary_component_ignored"])
        self.assertTrue(semantics["prediction_self_contained_hsx_safe"])
        self.assertEqual(semantics["prediction_self_contained_hsx_unsafe_reason"], "")
        self.assertAlmostEqual(semantics["prediction_overlap_relative_frobenius_vs_reference"], 0.0)

    def test_extract_refuses_to_overwrite_existing_metric_outputs_without_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result_dir = Path(tmp) / "result"
            metrics_dir = result_dir / "metrics"
            metrics_dir.mkdir(parents=True)
            (metrics_dir / "manifest.json").write_text("{}", encoding="utf-8")
            stale_file = result_dir / "eigenvalues" / "siesta" / "stale.csv"
            stale_file.parent.mkdir(parents=True)
            stale_file.write_text("old\n", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "Refusing to overwrite existing Hamiltonian metric outputs"):
                self.module.extract(result_dir)

            manifest = self.module.extract(result_dir, overwrite=True)

            self.assertEqual(manifest["metrics_schema_version"], self.module.METRICS_SCHEMA_VERSION)
            self.assertEqual(manifest["metrics_provenance_status"], "post_h_only_sref")
            self.assertFalse(stale_file.exists())

    def test_prediction_overlap_validation_tolerance_controls_standalone_safety(self) -> None:
        from scipy import sparse

        reference = self.matrix_data(
            [[1.0, 0.0], [0.0, 2.0]],
            orthogonal=False,
            overlap=sparse.eye(2, format="csr"),
            spin_kind="Spin{unpolarized}",
        )
        within_tolerance = self.matrix_data(
            [[1.1, 0.0], [0.0, 2.1]],
            orthogonal=False,
            overlap=(1.0 + self.module.OVERLAP_RELATIVE_FROBENIUS_WARNING_THRESHOLD / 2.0)
            * sparse.eye(2, format="csr"),
            spin_kind="Spin{unpolarized}",
        )
        outside_tolerance = self.matrix_data(
            [[1.1, 0.0], [0.0, 2.1]],
            orthogonal=False,
            overlap=(1.0 + 2.0 * self.module.OVERLAP_RELATIVE_FROBENIUS_WARNING_THRESHOLD)
            * sparse.eye(2, format="csr"),
            spin_kind="Spin{unpolarized}",
        )

        safe = self.module.matrix_semantics_fields(
            reference,
            within_tolerance,
            target_component_policy="h_only",
        )
        unsafe = self.module.matrix_semantics_fields(
            reference,
            outside_tolerance,
            target_component_policy="h_only",
        )

        self.assertTrue(safe["prediction_self_contained_hsx_safe"])
        self.assertEqual(safe["prediction_self_contained_hsx_unsafe_reason"], "")
        self.assertFalse(unsafe["prediction_self_contained_hsx_safe"])
        self.assertEqual(unsafe["prediction_self_contained_hsx_unsafe_reason"], "prediction_overlap_mismatch")

    def test_graph2mat_auxiliary_prediction_is_severe_and_not_self_contained(self) -> None:
        from scipy import sparse

        reference = self.matrix_data(
            [[1.0, 0.0], [0.0, 2.0]],
            orthogonal=False,
            overlap=sparse.eye(2, format="csr"),
            component_count=1,
            spin_kind="Spin{unpolarized}",
            components=[[[1.0, 0.0], [0.0, 2.0]]],
        )
        predicted = self.matrix_data(
            [[1.1, 0.0], [0.0, 2.1]],
            orthogonal=False,
            overlap=sparse.eye(2, format="csr"),
            component_count=2,
            spin_kind="Spin{polarized}",
            components=[
                [[1.1, 0.0], [0.0, 2.1]],
                [[9.0, 0.0], [0.0, 9.0]],
            ],
        )

        errors = self.module.matrix_compatibility_errors(
            "graph2mat_auxiliary",
            reference,
            predicted,
            target_component_policy="h_only",
        )
        self.assertNotIn("target_component_policy_mismatch", {error["kind"] for error in errors})
        self.assertNotIn("unsupported_matrix_components", {error["kind"] for error in errors})
        self.assertNotIn("spin_state_mismatch", {error["kind"] for error in errors})

        sparse_row = self.module.sparse_metrics("graph2mat_auxiliary", reference, predicted)
        self.assertAlmostEqual(sparse_row["mae_union_eV"], 0.1)
        self.assertAlmostEqual(sparse_row["h_matrix_mae_eV"], sparse_row["mae_union_eV"])
        self.assertAlmostEqual(sparse_row["h_matrix_rmse_eV"], sparse_row["rmse_union_eV"])
        self.assertTrue(sparse_row["h_matrix_metric_independent_of_training_loss"])
        self.assertEqual(sparse_row["h_matrix_component_index"], 0)

        warnings = self.module.matrix_compatibility_warnings(
            "graph2mat_auxiliary",
            reference,
            predicted,
        )
        auxiliary_warning = next(
            warning for warning in warnings if warning["kind"] == "graph2mat_auxiliary_component_ignored"
        )
        self.assertEqual(auxiliary_warning["severity"], "severe")

        semantics = self.module.matrix_semantics_fields(
            reference,
            predicted,
            target_component_policy="h_only",
        )
        self.assertTrue(semantics["graph2mat_auxiliary_component_ignored"])
        self.assertFalse(semantics["prediction_self_contained_hsx_safe"])
        self.assertEqual(
            semantics["prediction_self_contained_hsx_unsafe_reason"],
            "graph2mat_auxiliary_component_ignored",
        )

        component_rows = self.module.component_channel_metrics(
            "graph2mat_auxiliary",
            reference,
            predicted,
            semantics,
        )
        self.assertEqual(len(component_rows), 2)
        self.assertTrue(component_rows[0]["component_metric_available"])
        self.assertEqual(component_rows[0]["component_role"], "hamiltonian")
        self.assertEqual(component_rows[0]["component_target_label"], "H")
        self.assertEqual(component_rows[0]["component_units"], "eV")
        self.assertTrue(component_rows[0]["component_is_official_hamiltonian_target"])
        self.assertTrue(component_rows[0]["component_in_official_h_only_loss"])
        self.assertTrue(component_rows[0]["component_in_official_sparse_h_metric"])
        self.assertFalse(component_rows[1]["component_metric_available"])
        self.assertEqual(component_rows[1]["component_role"], "auxiliary")
        self.assertEqual(component_rows[1]["component_target_label"], "auxiliary_non_target")
        self.assertEqual(component_rows[1]["component_units"], "auxiliary_or_dimensionless")
        self.assertFalse(component_rows[1]["component_is_official_hamiltonian_target"])
        self.assertFalse(component_rows[1]["component_in_official_h_only_loss"])
        self.assertFalse(component_rows[1]["component_in_official_sparse_h_metric"])
        self.assertEqual(component_rows[1]["component_unavailable_reason"], "missing_reference_component")

    def test_prediction_overlap_mismatch_warns_and_marks_hsx_unsafe(self) -> None:
        from scipy import sparse

        reference = self.matrix_data(
            [[1.0, 0.0], [0.0, 2.0]],
            orthogonal=False,
            overlap=sparse.eye(2, format="csr"),
            spin_kind="Spin{unpolarized}",
        )
        predicted = self.matrix_data(
            [[1.1, 0.0], [0.0, 2.1]],
            orthogonal=False,
            overlap=2.0 * sparse.eye(2, format="csr"),
            spin_kind="Spin{unpolarized}",
        )

        semantics = self.module.matrix_semantics_fields(
            reference,
            predicted,
            target_component_policy="h_only",
        )
        self.assertEqual(semantics["overlap_source"], "siesta_reference")
        self.assertFalse(semantics["prediction_own_overlap_used"])
        self.assertGreater(semantics["prediction_overlap_relative_frobenius_vs_reference"], 0.0)
        self.assertFalse(semantics["prediction_self_contained_hsx_safe"])
        self.assertEqual(
            semantics["prediction_self_contained_hsx_unsafe_reason"],
            "prediction_overlap_mismatch",
        )

        warnings = self.module.matrix_compatibility_warnings("overlap_mismatch", reference, predicted)
        mismatch_warning = next(warning for warning in warnings if warning["kind"] == "prediction_overlap_mismatch")
        self.assertEqual(mismatch_warning["severity"], "severe")
        self.assertFalse(mismatch_warning["prediction_own_overlap_used"])

    def test_prediction_artifact_safety_summary_counts_safe_and_unsafe_samples(self) -> None:
        rows = [
            {
                "sample": "safe",
                "prediction_self_contained_hsx_safe": True,
                "prediction_self_contained_hsx_unsafe_reason": "",
                "prediction_overlap_relative_frobenius_vs_reference": 0.0,
                "graph2mat_auxiliary_component_ignored": False,
            },
            {
                "sample": "auxiliary",
                "prediction_self_contained_hsx_safe": False,
                "prediction_self_contained_hsx_unsafe_reason": "graph2mat_auxiliary_component_ignored",
                "prediction_overlap_relative_frobenius_vs_reference": 0.0,
                "graph2mat_auxiliary_component_ignored": True,
            },
            {
                "sample": "mismatch",
                "prediction_self_contained_hsx_safe": False,
                "prediction_self_contained_hsx_unsafe_reason": "prediction_overlap_mismatch",
                "prediction_overlap_relative_frobenius_vs_reference": 0.5,
                "graph2mat_auxiliary_component_ignored": False,
            },
        ]

        summary = self.module.prediction_artifact_safety_summary(rows)

        self.assertFalse(summary["prediction_artifacts_standalone_safe"])
        self.assertFalse(summary["prediction_own_overlap_used_for_spectra"])
        self.assertEqual(summary["samples_with_prediction_semantics"], 3)
        self.assertEqual(summary["prediction_self_contained_hsx_safe_samples"], 1)
        self.assertEqual(summary["prediction_self_contained_hsx_unsafe_samples"], 2)
        self.assertEqual(summary["graph2mat_auxiliary_component_ignored_samples"], 1)
        self.assertEqual(summary["prediction_overlap_mismatch_samples"], 1)
        self.assertEqual(
            summary["prediction_self_contained_hsx_unsafe_reasons"],
            {"graph2mat_auxiliary_component_ignored": 1, "prediction_overlap_mismatch": 1},
        )
        self.assertIn("ML_prediction.HSX", summary["standalone_hsx_caveat"])

    def test_h_only_policy_rejects_unexpected_multicomponent_prediction(self) -> None:
        reference = self.matrix_data([[1.0, 0.0], [0.0, 2.0]], component_count=1)
        predicted = self.matrix_data(
            [[1.1, 0.0], [0.0, 2.1]],
            component_count=2,
            spin_kind="Spin{unpolarized}",
        )

        errors = self.module.matrix_compatibility_errors(
            "unexpected_multicomponent",
            reference,
            predicted,
            target_component_policy="h_only",
        )

        self.assertIn("target_component_policy_mismatch", {error["kind"] for error in errors})
        self.assertIn("unsupported_matrix_components", {error["kind"] for error in errors})

    def test_complex_hermitian_standard_eigenproblem(self) -> None:
        import numpy as np

        hamiltonian = np.asarray([[1.0, 1.0j], [-1.0j, 2.0]], dtype=complex)

        values = self.module.complex_generalized_eigenvalues(hamiltonian)

        np.testing.assert_allclose(values, np.linalg.eigvalsh(hamiltonian))

    def test_complex_hermitian_generalized_eigenproblem(self) -> None:
        import numpy as np
        import scipy.linalg

        hamiltonian = np.asarray([[1.0, 0.2 + 0.3j], [0.2 - 0.3j, 2.0]], dtype=complex)
        overlap = np.asarray([[1.4, 0.1j], [-0.1j, 1.2]], dtype=complex)

        values = self.module.complex_generalized_eigenvalues(hamiltonian, overlap)
        expected = scipy.linalg.eigh(hamiltonian, overlap, eigvals_only=True, check_finite=False)

        np.testing.assert_allclose(values, expected)

    def test_non_hermitian_complex_input_is_measured_and_symmetrized(self) -> None:
        import numpy as np

        matrix = np.asarray([[1.0, 2.0 + 1.0j], [0.0, 1.0]], dtype=complex)

        defect = self.module.complex_hermiticity_defect(matrix)
        symmetrized = self.module.symmetrized_hermitian_dense(matrix)
        values = self.module.complex_generalized_eigenvalues(matrix)

        self.assertGreater(defect, 0.0)
        np.testing.assert_allclose(symmetrized, symmetrized.conj().T)
        self.assertEqual(values.shape, (2,))

    def test_complex_matrix_error_metrics_use_absolute_complex_magnitude(self) -> None:
        import numpy as np

        reference = np.zeros((1, 1), dtype=complex)
        predicted = np.asarray([[3.0 + 4.0j]], dtype=complex)

        metrics = self.module.complex_matrix_error_metrics(reference, predicted)

        self.assertEqual(metrics["n_entries"], 1)
        self.assertAlmostEqual(metrics["mae_eV"], 5.0)
        self.assertAlmostEqual(metrics["rmse_eV"], 5.0)
        self.assertAlmostEqual(metrics["mse_eV2"], 25.0)
        self.assertTrue(math.isnan(metrics["relative_frobenius"]))

    def test_missing_kpoint_overlap_for_nonorthogonal_reference_raises(self) -> None:
        class MissingOverlapHamiltonian:
            orthogonal = False

            def Hk(self, k, format="array"):
                return [[1.0, 0.0], [0.0, 2.0]]

        with self.assertRaisesRegex(RuntimeError, "requires S\\(k\\)"):
            self.module.kpoint_overlap_matrix(MissingOverlapHamiltonian(), (0.0, 0.0, 0.0))

    def test_kpoint_helpers_use_reference_overlap_for_prediction_spectrum(self) -> None:
        import numpy as np

        class FakeHamiltonian:
            orthogonal = False

            def __init__(self, scale: float) -> None:
                self.scale = scale
                self.requested_kpoints = []

            def Hk(self, k, format="array"):
                self.requested_kpoints.append(tuple(k))
                return np.asarray([[self.scale + k[0], 0.1j], [-0.1j, self.scale + 1.0]], dtype=complex)

            def Sk(self, k, format="array"):
                self.requested_kpoints.append(tuple(k))
                return np.asarray([[1.5, 0.05j], [-0.05j, 1.2]], dtype=complex)

        reference = FakeHamiltonian(scale=1.0)
        predicted = FakeHamiltonian(scale=2.0)

        values = self.module.kpoint_eigenvalues_with_reference_overlap(predicted, reference, (0.25, 0.0, 0.0))
        expected = self.module.complex_generalized_eigenvalues(
            predicted.Hk((0.25, 0.0, 0.0), format="array"),
            reference.Sk((0.25, 0.0, 0.0), format="array"),
        )

        np.testing.assert_allclose(values, expected)
        self.assertIn((0.25, 0.0, 0.0), reference.requested_kpoints)
        self.assertIn((0.25, 0.0, 0.0), predicted.requested_kpoints)

    def test_gamma_complex_eigenvalues_match_existing_sparse_path(self) -> None:
        import numpy as np
        from scipy import sparse

        hamiltonian = sparse.csr_matrix([[1.0, 0.2], [0.2, 2.0]])

        old_values = self.module.generalized_eigenvalues(hamiltonian, None)
        new_values = self.module.complex_generalized_eigenvalues(hamiltonian, None)

        np.testing.assert_allclose(new_values, old_values)

    def test_extract_non_gamma_without_flag_remains_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result_dir = Path(tmp) / "result"
            self.write_minimal_kpoint_result(result_dir, mesh=(2, 1, 1))

            manifest = self.module.extract(result_dir, workers=1, enable_kpoint_metrics=False)

        self.assertEqual(manifest["samples_compared"], 0)
        self.assertEqual(manifest["kpoint_samples_compared"], 0)
        self.assertTrue(manifest["fatal_errors"])
        self.assertEqual(manifest["fatal_errors"][0]["kind"], "unsupported_kpoint_sampling")
        self.assertFalse(manifest["kpoint_metrics_enabled"])

    def test_reference_selection_still_rejects_ml_prediction_as_ground_truth(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sample_dir = Path(tmp) / "sample"
            sample_dir.mkdir()
            sample_dir.joinpath("ML_prediction.HSX").write_bytes(b"prediction")

            selection = self.module.choose_reference_matrix(sample_dir)

        self.assertFalse(selection.ok)
        self.assertEqual(selection.reason, "missing_reference_matrix")

    def test_extract_non_gamma_with_flag_writes_kpoint_outputs(self) -> None:
        import numpy as np
        from scipy import sparse

        class FakeHamiltonian:
            orthogonal = False
            spin = "Spin{unpolarized}"

            def __init__(self, scale: float) -> None:
                self.scale = scale

            def dim(self) -> int:
                return 2

            def tocsr(self, dim: int = 0, isc=None, **kwargs):
                return sparse.csr_matrix([[self.scale, 0.1], [0.1, self.scale + 1.0]])

            def Hk(self, k, format="array"):
                return np.asarray(
                    [[self.scale + float(k[0]), 0.1j], [-0.1j, self.scale + 1.0]],
                    dtype=complex,
                )

            def Sk(self, k, format="array"):
                return np.asarray([[1.4, 0.02j], [-0.02j, 1.1]], dtype=complex)

            def eigh(self, k=(0, 0, 0), **kwargs):
                return np.asarray([self.scale, self.scale + 1.0], dtype=float)

        class FakeSile:
            def __init__(self, path: str) -> None:
                self.path = path

            def read_hamiltonian(self):
                return FakeHamiltonian(2.0 if "ML_prediction" in self.path else 1.0)

            def read_overlap(self):
                return sparse.eye(2, format="csr")

            def read_fermi_level(self):
                return 1.5

        with tempfile.TemporaryDirectory() as tmp:
            result_dir = Path(tmp) / "result"
            self.write_minimal_kpoint_result(result_dir, mesh=(2, 1, 1))
            with mock.patch.object(self.module.sisl, "get_sile", side_effect=lambda path: FakeSile(path)):
                manifest = self.module.extract(result_dir, workers=1, enable_kpoint_metrics=True)

            matrix_csv = result_dir / "metrics" / "kpoint_matrix_metrics.csv"
            spectral_csv = result_dir / "metrics" / "kpoint_spectral_metrics.csv"
            dos_csv = result_dir / "metrics" / "kpoint_dos_metrics.csv"
            kpoints_csv = result_dir / "eigenvalues" / "kpoints.csv"
            siesta_eigs = sorted((result_dir / "eigenvalues" / "siesta").glob("*_k*.csv"))
            predicted_eigs = sorted((result_dir / "eigenvalues" / "predicted").glob("*_k*.csv"))
            band_errors = sorted((result_dir / "eigenvalues" / "kpoint_band_errors").glob("*_k*.csv"))
            self.assertEqual(manifest["fatal_errors"], [])
            self.assertEqual(manifest["samples_compared"], 1)
            self.assertEqual(manifest["kpoint_samples_compared"], 1)
            self.assertTrue(manifest["kpoint_metrics_enabled"])
            self.assertEqual(manifest["kpoint_mesh"], [2, 1, 1])
            self.assertEqual(manifest["kpoint_count"], 2)
            self.assertTrue(manifest["uses_reference_overlap_k"])
            self.assertIn("prediction_artifact_semantics", manifest)
            self.assertTrue(manifest["prediction_artifacts_standalone_safe"])
            self.assertEqual(manifest["prediction_self_contained_hsx_safe_samples"], 1)
            self.assertEqual(manifest["prediction_self_contained_hsx_unsafe_samples"], 0)
            metrics_manifest = json.loads((result_dir / "metrics" / "manifest.json").read_text(encoding="utf-8"))
            self.assertIn("prediction_artifact_semantics", metrics_manifest)
            self.assertTrue(metrics_manifest["prediction_artifact_semantics"]["prediction_artifacts_standalone_safe"])
            for path in [matrix_csv, spectral_csv, dos_csv, kpoints_csv]:
                self.assertTrue(path.exists(), path)
                header = path.read_text(encoding="utf-8").splitlines()[0]
                self.assertIn("sample", header)
            self.assertIn("uses_reference_overlap_k", spectral_csv.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(len(siesta_eigs), 2)
            self.assertEqual(len(predicted_eigs), 2)
            self.assertEqual(len(band_errors), 2)

    def test_kpoint_sampled_structure_is_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fdf = Path(tmp) / "sample" / "RUN.fdf"
            synthetic_fdf(fdf, kpoints=True)

            issues = self.module.unsupported_kpoint_issues("sample", fdf)

        self.assertTrue(issues)
        self.assertEqual(issues[0]["kind"], "unsupported_kpoint_sampling")
        self.assertEqual(issues[0]["severity"], "fatal")

    def test_parse_gamma_monkhorst_pack_block(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fdf = Path(tmp) / "sample" / "RUN.fdf"
            fdf.parent.mkdir(parents=True)
            fdf.write_text(
                "\n".join(
                    [
                        "%block kgrid_Monkhorst_Pack",
                        "1 0 0 0.0",
                        "0 1 0 0.0",
                        "0 0 1 0.0",
                        "%endblock kgrid_Monkhorst_Pack",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            grid = self.module.parse_monkhorst_pack_kgrid(fdf)

        self.assertIsNotNone(grid)
        self.assertTrue(grid.ok)
        self.assertEqual(grid.mesh, (1, 1, 1))
        self.assertEqual(grid.shifts, (0.0, 0.0, 0.0))
        self.assertTrue(grid.is_gamma_only)
        self.assertEqual(grid.fractional_kpoints, ((0.0, 0.0, 0.0),))
        self.assertEqual(grid.weights, (1.0,))

    def test_parse_graphene_monkhorst_pack_block(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fdf = Path(tmp) / "sample" / "RUN.fdf"
            fdf.parent.mkdir(parents=True)
            fdf.write_text(
                "\n".join(
                    [
                        "%block kgrid_Monkhorst_Pack",
                        "6 0 0 0.0",
                        "0 6 0 0.0",
                        "0 0 1 0.0",
                        "%endblock kgrid_Monkhorst_Pack",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            grid = self.module.parse_monkhorst_pack_kgrid(fdf)
            issues = self.module.unsupported_kpoint_issues("sample", fdf)

        self.assertIsNotNone(grid)
        self.assertTrue(grid.ok)
        self.assertEqual(grid.mesh, (6, 6, 1))
        self.assertEqual(grid.shifts, (0.0, 0.0, 0.0))
        self.assertFalse(grid.is_gamma_only)
        self.assertEqual(len(grid.fractional_kpoints), 36)
        self.assertEqual(len(grid.weights), 36)
        self.assertTrue(all(math.isclose(weight, 1.0 / 36.0) for weight in grid.weights))
        self.assertTrue(math.isclose(sum(grid.weights), 1.0))
        self.assertTrue(all(math.isclose(kpoint[2], 0.0) for kpoint in grid.fractional_kpoints))
        self.assertTrue(issues)
        self.assertEqual(issues[0]["kind"], "unsupported_kpoint_sampling")

    def test_parse_inline_gamma_monkhorst_pack(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fdf = Path(tmp) / "sample" / "RUN.fdf"
            fdf.parent.mkdir(parents=True)
            fdf.write_text("kgrid_Monkhorst_Pack 1 1 1 0.0 0.0 0.0\n", encoding="utf-8")

            grid = self.module.parse_monkhorst_pack_kgrid(fdf)
            issues = self.module.unsupported_kpoint_issues("sample", fdf)

        self.assertIsNotNone(grid)
        self.assertTrue(grid.ok)
        self.assertEqual(grid.mesh, (1, 1, 1))
        self.assertTrue(grid.is_gamma_only)
        self.assertEqual(grid.fractional_kpoints, ((0.0, 0.0, 0.0),))
        self.assertEqual(issues, [])

    def test_parse_malformed_monkhorst_pack_block(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fdf = Path(tmp) / "sample" / "RUN.fdf"
            fdf.parent.mkdir(parents=True)
            fdf.write_text(
                "\n".join(
                    [
                        "%block kgrid_Monkhorst_Pack",
                        "2 0 0 0.0",
                        "0 nope 0 0.0",
                        "0 0 2 0.0",
                        "%endblock kgrid_Monkhorst_Pack",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            grid = self.module.parse_monkhorst_pack_kgrid(fdf)
            issues = self.module.unsupported_kpoint_issues("sample", fdf)

        self.assertIsNotNone(grid)
        self.assertFalse(grid.ok)
        self.assertEqual(grid.error, "malformed_monkhorst_pack_row")
        self.assertTrue(issues)
        self.assertEqual(issues[0]["kind"], "unsupported_kpoint_sampling")

    def test_parse_monkhorst_pack_comments_and_case_insensitive_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fdf = Path(tmp) / "sample" / "RUN.fdf"
            fdf.parent.mkdir(parents=True)
            fdf.write_text(
                "\n".join(
                    [
                        "%BLOCK Kgrid.MonkhorstPack # comment",
                        "2 0 0 0.0 # x mesh",
                        "0 2 0 0.0",
                        "0 0 1 0.0",
                        "%ENDBLOCK Kgrid.MonkhorstPack",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            grid = self.module.parse_monkhorst_pack_kgrid(fdf)

        self.assertIsNotNone(grid)
        self.assertTrue(grid.ok)
        self.assertEqual(grid.mesh, (2, 2, 1))
        self.assertEqual(grid.source_directive, "kgrid.monkhorstpack")
        self.assertEqual(len(grid.fractional_kpoints), 4)
        self.assertTrue(math.isclose(sum(grid.weights), 1.0))

    def test_gamma_monkhorst_pack_structure_is_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fdf = Path(tmp) / "sample" / "RUN.fdf"
            fdf.parent.mkdir(parents=True)
            fdf.write_text(
                "\n".join(
                    [
                        "%block kgrid_Monkhorst_Pack",
                        " 1  0  0  0.0",
                        " 0  1  0  0.0",
                        " 0  0  1  0.0",
                        "%endblock kgrid_Monkhorst_Pack",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            issues = self.module.unsupported_kpoint_issues("sample", fdf)

        self.assertEqual(issues, [])


if __name__ == "__main__":
    unittest.main()
