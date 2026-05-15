from __future__ import annotations

import importlib.util
import json
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

