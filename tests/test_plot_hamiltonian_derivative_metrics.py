from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "Comparison" / "scripts" / "plot_hamiltonian_derivative_metrics.py"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or ["sample"])
        writer.writeheader()
        writer.writerows(rows)


class PlotHamiltonianDerivativeMetricsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def derivative_root(self, name: str) -> Path:
        return self.root / name / "derivative_metrics"

    def write_derivative_fixture(
        self,
        name: str,
        *,
        source_model: str,
        rows: list[dict[str, object]] | None = None,
        hermiticity_rows: list[dict[str, object]] | None = None,
        scientific_status: str = "diagnostic_only",
        stencils_failed: int = 0,
        fatal_errors: list[dict[str, object]] | None = None,
    ) -> Path:
        derivative_root = self.derivative_root(name)
        write_json(
            derivative_root / "manifest.json",
            {
                "schema_version": "hamiltonian_derivative_metrics_v1",
                "scientific_status": scientific_status,
                "force_constants_used": False,
                "stencils_total": len(rows or []),
                "stencils_ok": len(rows or []),
                "stencils_failed": stencils_failed,
                "fatal_errors": fatal_errors or [],
            },
        )
        write_csv(derivative_root / "derivative_matrix_metrics.csv", rows or [])
        write_csv(derivative_root / "derivative_hermiticity.csv", hermiticity_rows or [])
        write_csv(derivative_root / "stencil_status.csv", [])
        return derivative_root

    def run_script(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPT), *args],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    def test_plot_script_handles_empty_metrics_gracefully(self) -> None:
        derivative_root = self.write_derivative_fixture("empty", source_model="graph2mat")
        output_dir = self.root / "plots_empty"

        completed = self.run_script("--derivative-root", str(derivative_root), "--output-dir", str(output_dir))

        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads((output_dir / "derivative_plot_payload.json").read_text(encoding="utf-8"))
        self.assertFalse(payload["available"])
        self.assertTrue(payload["diagnostic_only"])
        self.assertTrue(payload["scientific_warnings"])

    def test_plot_script_writes_manifest(self) -> None:
        derivative_root = self.write_derivative_fixture(
            "single",
            source_model="graph2mat",
            rows=[
                {
                    "sample": "s0",
                    "source_model": "graph2mat",
                    "atom_index_zero_based": 0,
                    "axis": "x",
                    "delta_ang": 0.01,
                    "finite_difference_method": "central",
                    "dh_mae_union_eV_per_Ang": 0.2,
                    "dh_rmse_union_eV_per_Ang": 0.3,
                    "dh_relative_frobenius_ref": 0.4,
                    "dh_false_zero_rate": 0.1,
                    "dh_false_nonzero_rate": 0.05,
                }
            ],
            hermiticity_rows=[
                {
                    "sample": "s0",
                    "finite_difference_method": "central",
                    "dH_ref_hermiticity_defect": 0.0,
                    "dH_pred_hermiticity_defect": 0.02,
                    "dH_hermiticity_error_delta": 0.02,
                }
            ],
        )
        output_dir = self.root / "plots_manifest"

        completed = self.run_script("--derivative-root", str(derivative_root), "--output-dir", str(output_dir))

        self.assertEqual(completed.returncode, 0, completed.stderr)
        manifest = json.loads((output_dir / "derivative_plot_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema"], "hamiltonian_derivative_plot_manifest_v1")
        self.assertEqual(manifest["title"], "Hamiltonian derivative diagnostics")
        self.assertTrue((output_dir / "derivative_plot_payload.json").exists())

    def test_plot_metadata_contains_scientific_warnings(self) -> None:
        derivative_root = self.write_derivative_fixture(
            "warnings",
            source_model="graph2mat",
            rows=[
                {
                    "sample": "s0",
                    "source_model": "graph2mat",
                    "atom_index_zero_based": 0,
                    "axis": "x",
                    "delta_ang": 0.01,
                    "finite_difference_method": "forward",
                    "dh_mae_union_eV_per_Ang": 0.2,
                    "dh_rmse_union_eV_per_Ang": 0.3,
                    "dh_relative_frobenius_ref": 0.4,
                    "dh_false_zero_rate": 0.1,
                    "dh_false_nonzero_rate": 0.05,
                }
            ],
            scientific_status="diagnostic_only",
            stencils_failed=1,
            fatal_errors=[{"kind": "incomplete_derivative_stencil"}],
        )
        output_dir = self.root / "plots_warnings"

        completed = self.run_script("--derivative-root", str(derivative_root), "--output-dir", str(output_dir))

        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads((output_dir / "derivative_plot_payload.json").read_text(encoding="utf-8"))
        codes = {warning["code"] for warning in payload["scientific_warnings"]}
        self.assertIn("scientific_status_diagnostic", codes)
        self.assertIn("failed_stencils_present", codes)
        self.assertIn("fatal_errors_present", codes)

    def test_no_paper_level_wording_appears(self) -> None:
        graph2mat_root = self.write_derivative_fixture(
            "graph2mat",
            source_model="graph2mat",
            rows=[
                {
                    "sample": "shared",
                    "source_model": "graph2mat",
                    "atom_index_zero_based": 0,
                    "axis": "x",
                    "delta_ang": 0.01,
                    "finite_difference_method": "central",
                    "dh_mae_union_eV_per_Ang": 0.2,
                    "dh_rmse_union_eV_per_Ang": 0.3,
                    "dh_relative_frobenius_ref": 0.4,
                    "dh_false_zero_rate": 0.1,
                    "dh_false_nonzero_rate": 0.05,
                }
            ],
            hermiticity_rows=[
                {
                    "sample": "shared",
                    "finite_difference_method": "central",
                    "dH_ref_hermiticity_defect": 0.0,
                    "dH_pred_hermiticity_defect": 0.02,
                    "dH_hermiticity_error_delta": 0.02,
                }
            ],
        )
        deeph_root = self.write_derivative_fixture(
            "deeph",
            source_model="deeph",
            rows=[
                {
                    "sample": "shared",
                    "source_model": "deeph",
                    "atom_index_zero_based": 0,
                    "axis": "x",
                    "delta_ang": 0.01,
                    "finite_difference_method": "central",
                    "dh_mae_union_eV_per_Ang": 0.15,
                    "dh_rmse_union_eV_per_Ang": 0.25,
                    "dh_relative_frobenius_ref": 0.35,
                    "dh_false_zero_rate": 0.05,
                    "dh_false_nonzero_rate": 0.02,
                }
            ],
            hermiticity_rows=[
                {
                    "sample": "shared",
                    "finite_difference_method": "central",
                    "dH_ref_hermiticity_defect": 0.0,
                    "dH_pred_hermiticity_defect": 0.01,
                    "dH_hermiticity_error_delta": 0.01,
                }
            ],
        )
        output_dir = self.root / "plots_labels"

        completed = self.run_script(
            "--graph2mat-root",
            str(graph2mat_root),
            "--deeph-root",
            str(deeph_root),
            "--output-dir",
            str(output_dir),
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload_text = (output_dir / "derivative_plot_payload.json").read_text(encoding="utf-8").lower()
        self.assertNotIn("paper-level", payload_text)
        payload = json.loads((output_dir / "derivative_plot_payload.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["title"], "Hamiltonian derivative diagnostics")
        self.assertEqual(payload["reference_label"], "Reference: finite differences of SIESTA Hamiltonians")
        self.assertEqual(payload["force_constants_label"], "SIESTA force constants are not treated as dH/dR")
        paired_plot = next(plot for plot in payload["plots"] if plot["id"] == "graph2mat_vs_deeph_paired_comparison")
        self.assertEqual(len(paired_plot["rows"]), 1)


if __name__ == "__main__":
    unittest.main()
