import csv
import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "Comparison" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

pipeline_ui = importlib.import_module("pipeline_ui")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
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


class G2MDeepHDerivativeUIBackendTests(unittest.TestCase):
    def write_derivative_tree(self, run_root: Path, name: str, *, source_model: str, fatal_error: bool = False) -> None:
        root = run_root / "common_metrics" / f"{name}_eval" / "derivative_metrics"
        write_json(
            root / "manifest.json",
            {
                "scientific_status": "diagnostic_only",
                "finite_difference_method": "central",
                "derivative_units": "eV/Ang",
                "stencils_ok": 1,
                "stencils_failed": 1 if fatal_error else 0,
                "warnings": [],
                "fatal_errors": [{"kind": "missing_required_metadata", "message": "orbital ordering missing"}] if fatal_error else [],
            },
        )
        write_csv(
            root / "derivative_matrix_metrics.csv",
            [
                {
                    "sample": "shared",
                    "source_model": source_model,
                    "atom_index_zero_based": 0,
                    "axis": "x",
                    "delta_ang": 0.01,
                    "finite_difference_method": "central",
                    "derivative_units": "eV/Ang",
                    "dh_mae_union_eV_per_Ang": 0.2 if source_model == "graph2mat" else 0.1,
                    "dh_rmse_union_eV_per_Ang": 0.3 if source_model == "graph2mat" else 0.2,
                    "dh_relative_frobenius_ref": 0.4 if source_model == "graph2mat" else 0.3,
                    "dh_false_zero_rate": 0.1,
                    "dh_false_nonzero_rate": 0.05,
                }
            ],
        )
        write_csv(
            root / "derivative_hermiticity.csv",
            [
                {
                    "sample": "shared",
                    "finite_difference_method": "central",
                    "dH_ref_hermiticity_defect": 0.0,
                    "dH_pred_hermiticity_defect": 0.02 if source_model == "graph2mat" else 0.01,
                    "dH_hermiticity_error_delta": 0.02 if source_model == "graph2mat" else 0.01,
                }
            ],
        )
        write_csv(
            root / "stencil_status.csv",
            [
                {
                    "sample": "shared",
                    "status": "failed" if fatal_error else "ok",
                    "issue_codes": "missing_required_metadata" if fatal_error else "",
                    "issue_messages": "orbital ordering missing" if fatal_error else "",
                }
            ],
        )

    def test_missing_metrics_returns_not_computed(self) -> None:
        with patch.object(pipeline_ui, "resolve_g2m_deeph_run_root", return_value=None):
            payload = pipeline_ui.g2m_deeph_derivative_metrics_payload("missing-run")
        self.assertFalse(payload["available"])
        self.assertTrue(payload["not_computed"])
        self.assertEqual(payload["status"], "not_computed")

    def test_not_computed_payload_can_still_include_gate_report_for_selected_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run_demo"
            write_json(
                run_root / "common_metrics" / "summary" / "common_summary.json",
                {
                    "status": "diagnostic_only",
                    "recommendation": {"status": "diagnostic_only", "reason": "Common metrics are diagnostic only."},
                },
            )
            write_json(
                run_root / "summary" / "ranking" / "ranking_summary.json",
                {
                    "recommendation": {
                        "status": "no_robust_winner",
                        "scientific_status": "diagnostic_only",
                        "reason": "Review gates and warnings.",
                    }
                },
            )
            with patch.object(pipeline_ui, "resolve_g2m_deeph_run_root", return_value=run_root):
                payload = pipeline_ui.g2m_deeph_derivative_metrics_payload("run_demo")
        self.assertFalse(payload["available"])
        self.assertTrue(payload["not_computed"])
        self.assertEqual(payload["gate_report"]["derivative_winner_claim"], "none")

    def test_derivative_payload_includes_status_warnings_and_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run_demo"
            self.write_derivative_tree(run_root, "graph2mat", source_model="graph2mat", fatal_error=True)
            self.write_derivative_tree(run_root, "deeph", source_model="deeph")
            write_json(
                run_root / "common_metrics" / "summary" / "common_summary.json",
                {
                    "status": "diagnostic_only",
                    "recommendation": {
                        "status": "diagnostic_only",
                        "primary_metric": "h_mae_eV_mean",
                        "reason": "Comparability warnings prevent a stronger claim.",
                    },
                },
            )
            write_json(
                run_root / "summary" / "ranking" / "ranking_summary.json",
                {
                    "recommendation": {
                        "status": "no_robust_winner",
                        "scientific_status": "diagnostic_only",
                        "winner": None,
                        "primary_metric": "h_mae_eV_mean",
                        "reason": "Review gates and warnings.",
                        "gates_failed": ["adapter_equivalence"],
                        "gates_passed": ["split_audit"],
                    }
                },
            )
            with patch.object(pipeline_ui, "resolve_g2m_deeph_run_root", return_value=run_root):
                payload = pipeline_ui.g2m_deeph_derivative_metrics_payload("run_demo")
        self.assertTrue(payload["available"])
        self.assertEqual(payload["run_id"], "run_demo")
        self.assertEqual(payload["title"], "Hamiltonian derivative diagnostics")
        self.assertTrue(payload["status_rows"])
        self.assertTrue(payload["comparison_rows"])
        self.assertTrue(payload["artifact_rows"])
        self.assertTrue(payload["issue_rows"])
        self.assertTrue(payload["prominent_issue_rows"])
        self.assertIsNone(payload["winner"])
        self.assertEqual(payload["reference_label"], "Reference: finite differences of SIESTA Hamiltonians")
        self.assertEqual(payload["force_constants_label"], "SIESTA force constants are not treated as dH/dR")
        self.assertEqual(payload["gate_report"]["derivative_winner_claim"], "none")
        self.assertEqual(payload["gate_report"]["ranking_status"], "no_robust_winner")
        self.assertTrue(payload["gate_report"]["gate_rows"])


if __name__ == "__main__":
    unittest.main()
