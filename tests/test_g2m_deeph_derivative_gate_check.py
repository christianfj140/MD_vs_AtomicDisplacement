from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "Comparison" / "scripts"
SCRIPT = SCRIPTS_DIR / "g2m_deeph_derivative_gate_check.py"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from g2m_deeph_derivative_gate_check import build_derivative_gate_report  # noqa: E402


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class DerivativeGateCheckTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.derivative_root = self.root / "graph2mat_eval" / "derivative_metrics"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def write_fixture(
        self,
        *,
        manifest_overrides: dict | None = None,
        metric_rows: list[dict[str, object]] | None = None,
        stencil_rows: list[dict[str, object]] | None = None,
        hermiticity_rows: list[dict[str, object]] | None = None,
    ) -> Path:
        manifest = {
            "schema_version": "hamiltonian_derivative_metrics_v1",
            "scientific_status": "presentation_ready",
            "finite_difference_method": "central",
            "force_constants_used": False,
            "reference_definition": "siesta_hamiltonian_finite_difference",
            "derivative_units": "eV/Ang",
            "split": "test",
            "diagnostic_only_requested": False,
            "stencils_total": 1,
            "stencils_ok": 1,
            "stencils_failed": 0,
            "warnings": [],
            "fatal_errors": [],
        }
        if manifest_overrides:
            manifest.update(manifest_overrides)
        write_json(self.derivative_root / "manifest.json", manifest)
        write_csv(
            self.derivative_root / "derivative_matrix_metrics.csv",
            [
                "sample",
                "atom_index_zero_based",
                "axis",
                "axis_index",
                "delta_ang",
                "finite_difference_method",
                "source_model",
                "reference_source",
                "derivative_units",
                "comparison_status",
                "dh_mae_union_eV_per_Ang",
                "dh_rmse_union_eV_per_Ang",
                "dh_relative_frobenius_ref",
                "dh_false_zero_rate",
                "dh_false_nonzero_rate",
                "dh_support_changed",
                "dh_hermiticity_ref",
                "dh_hermiticity_pred",
                "dh_hermiticity_error_delta",
            ],
            metric_rows
            or [
                {
                    "sample": "sample_0",
                    "atom_index_zero_based": 0,
                    "axis": "x",
                    "axis_index": 0,
                    "delta_ang": 0.01,
                    "finite_difference_method": "central",
                    "source_model": "graph2mat",
                    "reference_source": "siesta",
                    "derivative_units": "eV/Ang",
                    "comparison_status": "presentation_ready",
                    "dh_mae_union_eV_per_Ang": 0.1,
                    "dh_rmse_union_eV_per_Ang": 0.2,
                    "dh_relative_frobenius_ref": 0.05,
                    "dh_false_zero_rate": 0.0,
                    "dh_false_nonzero_rate": 0.0,
                    "dh_support_changed": False,
                    "dh_hermiticity_ref": 0.0,
                    "dh_hermiticity_pred": 0.0,
                    "dh_hermiticity_error_delta": 0.0,
                }
            ],
        )
        write_csv(
            self.derivative_root / "stencil_status.csv",
            [
                "sample",
                "status",
                "finite_difference_method",
                "base_sample_id",
                "plus_sample_id",
                "minus_sample_id",
                "atom_index_zero_based",
                "axis",
                "axis_index",
                "delta_ang",
                "issue_codes",
                "issue_messages",
            ],
            stencil_rows
            or [
                {
                    "sample": "sample_0",
                    "status": "ok",
                    "finite_difference_method": "central",
                    "base_sample_id": "base",
                    "plus_sample_id": "plus",
                    "minus_sample_id": "minus",
                    "atom_index_zero_based": 0,
                    "axis": "x",
                    "axis_index": 0,
                    "delta_ang": 0.01,
                    "issue_codes": "",
                    "issue_messages": "",
                }
            ],
        )
        write_csv(
            self.derivative_root / "derivative_hermiticity.csv",
            [
                "sample",
                "finite_difference_method",
                "source_model",
                "reference_source",
                "dH_ref_hermiticity_defect",
                "dH_pred_hermiticity_defect",
                "dH_hermiticity_error_delta",
                "finite_values",
            ],
            hermiticity_rows
            or [
                {
                    "sample": "sample_0",
                    "finite_difference_method": "central",
                    "source_model": "graph2mat",
                    "reference_source": "siesta",
                    "dH_ref_hermiticity_defect": 0.0,
                    "dH_pred_hermiticity_defect": 0.0,
                    "dH_hermiticity_error_delta": 0.0,
                    "finite_values": True,
                }
            ],
        )
        return self.derivative_root

    def build_report(self) -> dict:
        return build_derivative_gate_report(derivative_roots=[self.derivative_root])

    def test_force_constants_blocker(self) -> None:
        self.write_fixture(manifest_overrides={"force_constants_used": True})

        report = self.build_report()

        self.assertEqual(report["scientific_status"], "blocked")
        blocker_ids = {row["id"] for row in report["blockers"]}
        self.assertIn("force_constants_used", blocker_ids)

    def test_missing_central_stencil_blocker(self) -> None:
        self.write_fixture(
            manifest_overrides={"finite_difference_method": "forward"},
            metric_rows=[
                {
                    "sample": "sample_0",
                    "atom_index_zero_based": 0,
                    "axis": "x",
                    "axis_index": 0,
                    "delta_ang": 0.01,
                    "finite_difference_method": "forward",
                    "source_model": "graph2mat",
                    "reference_source": "siesta",
                    "derivative_units": "eV/Ang",
                    "comparison_status": "presentation_ready",
                    "dh_mae_union_eV_per_Ang": 0.1,
                    "dh_rmse_union_eV_per_Ang": 0.2,
                    "dh_relative_frobenius_ref": 0.05,
                    "dh_false_zero_rate": 0.0,
                    "dh_false_nonzero_rate": 0.0,
                    "dh_support_changed": False,
                    "dh_hermiticity_ref": 0.0,
                    "dh_hermiticity_pred": 0.0,
                    "dh_hermiticity_error_delta": 0.0,
                }
            ],
            stencil_rows=[
                {
                    "sample": "sample_0",
                    "status": "ok",
                    "finite_difference_method": "forward",
                    "base_sample_id": "base",
                    "plus_sample_id": "plus",
                    "minus_sample_id": "",
                    "atom_index_zero_based": 0,
                    "axis": "x",
                    "axis_index": 0,
                    "delta_ang": 0.01,
                    "issue_codes": "",
                    "issue_messages": "",
                }
            ],
        )

        report = self.build_report()

        self.assertEqual(report["scientific_status"], "blocked")
        blocker_ids = {row["id"] for row in report["blockers"]}
        self.assertIn("missing_central_stencil", blocker_ids)

    def test_diagnostic_only_allowed(self) -> None:
        self.write_fixture(
            manifest_overrides={"scientific_status": "diagnostic_only", "diagnostic_only_requested": True},
            metric_rows=[
                {
                    "sample": "sample_0",
                    "atom_index_zero_based": 0,
                    "axis": "x",
                    "axis_index": 0,
                    "delta_ang": 0.01,
                    "finite_difference_method": "central",
                    "source_model": "graph2mat",
                    "reference_source": "siesta",
                    "derivative_units": "eV/Ang",
                    "comparison_status": "diagnostic_only",
                    "dh_mae_union_eV_per_Ang": 0.1,
                    "dh_rmse_union_eV_per_Ang": 0.2,
                    "dh_relative_frobenius_ref": 0.05,
                    "dh_false_zero_rate": 0.0,
                    "dh_false_nonzero_rate": 0.0,
                    "dh_support_changed": False,
                    "dh_hermiticity_ref": 0.0,
                    "dh_hermiticity_pred": 0.0,
                    "dh_hermiticity_error_delta": 0.0,
                }
            ],
        )

        report = self.build_report()

        self.assertEqual(report["scientific_status"], "internal_diagnostic")
        self.assertIn("diagnostic-only", " ".join(report["allowed_claims"]).lower())

    def test_paper_level_blocked_without_delta_sweep(self) -> None:
        self.write_fixture(
            manifest_overrides={
                "basis_gauge_verified": True,
                "orbital_ordering_verified": True,
                "independent_dataset_metadata": True,
            }
        )

        report = self.build_report()

        self.assertEqual(report["scientific_status"], "technical_presentation")
        blocker_ids = {row["id"] for row in report["blockers"]}
        self.assertIn("paper_level_delta_sweep_missing", blocker_ids)

    def test_paper_level_blocked_without_ordering_or_gauge_evidence(self) -> None:
        self.write_fixture(
            manifest_overrides={
                "delta_sensitivity_study_passed": True,
                "independent_dataset_metadata": True,
            },
        )

        report = self.build_report()

        self.assertEqual(report["scientific_status"], "technical_presentation")
        blocker_ids = {row["id"] for row in report["blockers"]}
        self.assertIn("paper_level_ordering_or_gauge_evidence_missing", blocker_ids)

    def test_clean_synthetic_technical_presentation_case(self) -> None:
        self.write_fixture(
            manifest_overrides={
                "basis_gauge_verified": True,
                "orbital_ordering_verified": True,
                "independent_dataset_metadata": True,
            },
            metric_rows=[
                {
                    "sample": "sample_a",
                    "atom_index_zero_based": 0,
                    "axis": "x",
                    "axis_index": 0,
                    "delta_ang": 0.01,
                    "finite_difference_method": "central",
                    "source_model": "graph2mat",
                    "reference_source": "siesta",
                    "derivative_units": "eV/Ang",
                    "comparison_status": "presentation_ready",
                    "dh_mae_union_eV_per_Ang": 0.1,
                    "dh_rmse_union_eV_per_Ang": 0.2,
                    "dh_relative_frobenius_ref": 0.05,
                    "dh_false_zero_rate": 0.0,
                    "dh_false_nonzero_rate": 0.0,
                    "dh_support_changed": False,
                    "dh_hermiticity_ref": 0.0,
                    "dh_hermiticity_pred": 0.0,
                    "dh_hermiticity_error_delta": 0.0,
                },
                {
                    "sample": "sample_b",
                    "atom_index_zero_based": 0,
                    "axis": "x",
                    "axis_index": 0,
                    "delta_ang": 0.02,
                    "finite_difference_method": "central",
                    "source_model": "graph2mat",
                    "reference_source": "siesta",
                    "derivative_units": "eV/Ang",
                    "comparison_status": "presentation_ready",
                    "dh_mae_union_eV_per_Ang": 0.1,
                    "dh_rmse_union_eV_per_Ang": 0.2,
                    "dh_relative_frobenius_ref": 0.05,
                    "dh_false_zero_rate": 0.0,
                    "dh_false_nonzero_rate": 0.0,
                    "dh_support_changed": False,
                    "dh_hermiticity_ref": 0.0,
                    "dh_hermiticity_pred": 0.0,
                    "dh_hermiticity_error_delta": 0.0,
                },
            ],
        )

        report = self.build_report()

        self.assertEqual(report["scientific_status"], "technical_presentation")
        self.assertNotIn("paper_level_delta_sweep_missing", {row["id"] for row in report["blockers"]})
        self.assertIn("No paper-level or winner claim is allowed", " ".join(report["allowed_claims"]))

    def test_cli_writes_gate_report(self) -> None:
        self.write_fixture(manifest_overrides={"scientific_status": "diagnostic_only", "diagnostic_only_requested": True})
        output = self.root / "derivative_gate_report.json"

        completed = subprocess.run(
            [sys.executable, str(SCRIPT), "--derivative-root", str(self.derivative_root), "--output", str(output)],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, msg=completed.stderr + completed.stdout)
        payload = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(payload["schema_version"], "graph2mat_deeph_derivative_gate_report_v1")
        self.assertEqual(payload["scientific_status"], "internal_diagnostic")


if __name__ == "__main__":
    unittest.main()
