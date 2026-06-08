import csv
import importlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "Comparison" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

bands = importlib.import_module("compare_graphene_bands_siesta_g2m_deeph")


def write_band_csv(path: Path, *, offset: float = 0.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for k_index in range(4):
        rows.append(
            {
                "k_index": k_index,
                "k_distance": float(k_index),
                "band_index": 0,
                "energy_eV": -1.0 + 0.1 * k_index + offset,
                "fermi_level_eV": 0.0,
            }
        )
        rows.append(
            {
                "k_index": k_index,
                "k_distance": float(k_index),
                "band_index": 1,
                "energy_eV": 1.0 + 0.1 * k_index + offset,
                "fermi_level_eV": 0.0,
            }
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["k_index", "k_distance", "band_index", "energy_eV", "fermi_level_eV"])
        writer.writeheader()
        writer.writerows(rows)


class GrapheneBandComparisonTests(unittest.TestCase):
    def test_graphene_gkm_path_labels(self) -> None:
        _name, nodes = bands.load_kpath(type("Args", (), {"kpath": "graphene_gkm", "kpath_json": None})())

        self.assertEqual([bands.label_for_display(node.label) for node in nodes], ["Γ", "K", "M", "Γ"])

    def test_graphene_gkm_interpolation_no_duplicate_internal_nodes(self) -> None:
        _name, nodes = bands.load_kpath(type("Args", (), {"kpath": "graphene_gkm", "kpath_json": None})())
        records = bands.interpolate_kpath(nodes, points_per_segment=2)

        labels = [record.k_label for record in records if record.k_label]
        self.assertEqual(labels, ["Γ", "K", "M", "Γ"])
        self.assertEqual(labels.count("K"), 1)
        self.assertEqual(labels.count("M"), 1)
        self.assertEqual(len(records), 7)

    def test_bandlines_block_generation(self) -> None:
        _name, nodes = bands.load_kpath(type("Args", (), {"kpath": "graphene_gkm", "kpath_json": None})())
        block = bands.bandlines_block(nodes, points_per_segment=80)

        self.assertIn("BandLinesScale ReciprocalLatticeVectors", block)
        self.assertIn("%block BandLines", block)
        self.assertIn("0.3333333333    0.3333333333    0.0000000000    K", block)
        self.assertIn("0.5000000000    0.0000000000    0.0000000000    M", block)
        self.assertIn("%endblock BandLines", block)

    def test_parse_fdf_bandlines_and_temperature(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fdf = Path(tmp) / "RUN.fdf"
            fdf.write_text(
                "\n".join(
                    [
                        "MD.InitialTemperature 450 K",
                        "BandLinesScale ReciprocalLatticeVectors",
                        "%block BandLines",
                        "1 0.0 0.0 0.0 \\Gamma",
                        "50 0.33333 0.666667 0.0 K",
                        "50 0.5 0.5 0.0 M",
                        "50 0.0 0.0 0.0 \\Gamma",
                        "%endblock BandLines",
                    ]
                ),
                encoding="utf-8",
            )

            _name, nodes = bands.parse_fdf_bandlines(fdf)
            temperature = bands.parse_md_initial_temperature(fdf)

            self.assertEqual([bands.label_for_display(node.label) for node in nodes], ["Γ", "K", "M", "Γ"])
            self.assertEqual(nodes[1].k, (0.33333, 0.666667, 0.0))
            self.assertEqual(temperature["value"], 450.0)
            self.assertEqual(temperature["unit"], "K")

    def test_reject_ml_prediction_as_reference(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "ML_prediction.HSX"):
            bands.resolve_reference_path(Path("/tmp/ML_prediction.HSX"))

    def test_energy_alignment_fermi(self) -> None:
        self.assertAlmostEqual(bands.align_energy(4.25, 1.5, "fermi", None), 2.75)

    def test_band_error_metrics(self) -> None:
        rows = [
            {"error_eV": 1.0},
            {"error_eV": -2.0},
            {"error_eV": 2.0},
        ]

        metrics = bands.metric_summary(rows)

        self.assertAlmostEqual(metrics["band_mae_eV"], 5.0 / 3.0)
        self.assertAlmostEqual(metrics["band_rmse_eV"], 3.0**0.5)
        self.assertAlmostEqual(metrics["max_abs_error_eV"], 2.0)

    def test_dirac_diagnostic_flags_shift_and_gap(self) -> None:
        method = bands.MethodBands(
            method="SIESTA",
            sample_id="s0",
            bands=[[-2.0, -1.0, 1.4, 1.6]],
            fermi_level_eV=0.0,
            energy_zero_policy="fermi",
            raw_source="synthetic",
            hermiticity_defects=[],
            overlap_used=True,
            diagonalization_errors=[],
        )

        diagnostic = bands.dirac_diagnostic_for_method(
            method,
            k_index=0,
            occupied_bands=2,
            fermi_level_eV=0.0,
            gap_warning_meV=10.0,
            fermi_warning_meV=50.0,
        )

        self.assertAlmostEqual(diagnostic["gap_eV"], 2.4)
        self.assertAlmostEqual(diagnostic["dirac_minus_fermi_eV"], 0.2)
        self.assertTrue(diagnostic["warnings"])

    def test_prediction_dirac_diagnostic_uses_already_aligned_energy(self) -> None:
        method = bands.MethodBands(
            method="Graph2Mat",
            sample_id="s0",
            bands=[[-0.25, 0.35]],
            fermi_level_eV=-5.7,
            energy_zero_policy="fermi",
            raw_source="synthetic",
            hermiticity_defects=[],
            overlap_used=True,
            diagonalization_errors=[],
        )

        diagnostic = bands.dirac_diagnostic_for_method(
            method,
            k_index=0,
            occupied_bands=1,
            fermi_level_eV=-5.7,
            gap_warning_meV=1000.0,
            fermi_warning_meV=1000.0,
        )

        self.assertAlmostEqual(diagnostic["dirac_minus_fermi_eV"], 0.05)
        self.assertEqual(diagnostic["dirac_fermi_convention"], "prediction_already_fermi_aligned")

    def test_prediction_band_errors_do_not_subtract_reference_fermi_twice(self) -> None:
        siesta = bands.MethodBands(
            method="SIESTA",
            sample_id="s0",
            bands=[[-6.0, -5.4]],
            fermi_level_eV=-5.7,
            energy_zero_policy="fermi",
            raw_source="synthetic",
            hermiticity_defects=[],
            overlap_used=True,
            diagonalization_errors=[],
        )
        g2m = bands.MethodBands(
            method="Graph2Mat",
            sample_id="s0",
            bands=[[-0.25, 0.35]],
            fermi_level_eV=-5.7,
            energy_zero_policy="fermi",
            raw_source="synthetic",
            hermiticity_defects=[],
            overlap_used=True,
            diagonalization_errors=[],
        )
        kpoints = [bands.KPointRecord(0, 0.0, 0.0, 0.0, 0.0, "K", "Γ-K")]

        rows = bands.error_rows(g2m, siesta, kpoints, "fermi", -5.7)

        self.assertAlmostEqual(rows[0]["siesta_energy_eV"], -0.3)
        self.assertAlmostEqual(rows[0]["predicted_energy_eV"], -0.25)
        self.assertAlmostEqual(rows[0]["error_eV"], 0.05)

    def test_missing_overlap_fail_closed(self) -> None:
        import numpy as np

        class FakeHamiltonian:
            orthogonal = False

            def Hk(self, k, format="array"):
                return np.eye(2)

        _name, nodes = bands.load_kpath(type("Args", (), {"kpath": "graphene_gkm", "kpath_json": None})())
        records = bands.interpolate_kpath(nodes, points_per_segment=1)

        with self.assertRaisesRegex(RuntimeError, "requires S\\(k\\)"):
            bands.matrix_bands_from_sisl(
                method="Graph2Mat",
                sample_id="s0",
                hamiltonian_obj=FakeHamiltonian(),
                reference_obj=FakeHamiltonian(),
                kpoints=records,
                fermi_level=0.0,
                fail_closed=True,
            )

    def test_missing_overlap_no_fail_closed_marks_diagnostic(self) -> None:
        import numpy as np

        class FakeHamiltonian:
            orthogonal = False

            def Hk(self, k, format="array"):
                return np.eye(2)

        _name, nodes = bands.load_kpath(type("Args", (), {"kpath": "graphene_gkm", "kpath_json": None})())
        records = bands.interpolate_kpath(nodes, points_per_segment=1)

        result = bands.matrix_bands_from_sisl(
            method="Graph2Mat",
            sample_id="s0",
            hamiltonian_obj=FakeHamiltonian(),
            reference_obj=FakeHamiltonian(),
            kpoints=records,
            fermi_level=0.0,
            fail_closed=False,
        )

        self.assertEqual(result.scientific_status, "diagnostic_only")
        self.assertFalse(result.overlap_used)
        self.assertTrue(result.diagonalization_errors)

    def test_manifest_contains_required_fields_and_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_band_csv(root / "siesta.csv", offset=0.0)
            write_band_csv(root / "g2m.csv", offset=0.1)
            write_band_csv(root / "deeph.csv", offset=-0.1)
            output = root / "out"

            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS_DIR / "compare_graphene_bands_siesta_g2m_deeph.py"),
                    "--sample-id",
                    "s0",
                    "--siesta-band-data",
                    str(root / "siesta.csv"),
                    "--graph2mat-band-data",
                    str(root / "g2m.csv"),
                    "--deeph-band-data",
                    str(root / "deeph.csv"),
                    "--output-dir",
                    str(output),
                    "--points-per-segment",
                    "2",
                    "--fermi-level",
                    "0.0",
                    "--skip-plot",
                ],
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["overlap_policy"], "siesta_reference_overlap_for_all_methods")
            self.assertEqual(manifest["energy_zero_policy"], "fermi")
            self.assertIn("kpath", manifest)
            self.assertIn("input_hashes", manifest)
            self.assertEqual(manifest["status"], "completed")
            self.assertTrue((output / "bands_siesta.csv").exists())
            self.assertTrue((output / "band_errors_graph2mat.csv").exists())
            self.assertTrue((output / "band_summary.json").exists())

    def test_plot_outputs_are_created(self) -> None:
        try:
            import matplotlib  # noqa: F401
        except ModuleNotFoundError as exc:
            self.skipTest(f"matplotlib unavailable: {exc.name}")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_band_csv(root / "siesta.csv", offset=0.0)
            write_band_csv(root / "g2m.csv", offset=0.1)
            write_band_csv(root / "deeph.csv", offset=-0.1)
            output = root / "out"

            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS_DIR / "compare_graphene_bands_siesta_g2m_deeph.py"),
                    "--sample-id",
                    "s0",
                    "--siesta-band-data",
                    str(root / "siesta.csv"),
                    "--graph2mat-band-data",
                    str(root / "g2m.csv"),
                    "--deeph-band-data",
                    str(root / "deeph.csv"),
                    "--output-dir",
                    str(output),
                    "--points-per-segment",
                    "2",
                    "--fermi-level",
                    "0.0",
                    "--max-bands",
                    "2",
                ],
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue((output / "band_comparison.png").exists())
            self.assertTrue((output / "band_comparison.pdf").exists())


if __name__ == "__main__":
    unittest.main()
