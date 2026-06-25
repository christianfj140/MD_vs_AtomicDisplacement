import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "Comparison" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_incremental_derivative_metrics as incremental  # noqa: E402


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row}) or ["sample"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_workflow(run_root: Path, size: int, *, g2m: bool = True, deeph: bool = True) -> Path:
    wf = run_root / "derivative_workflows" / f"graphene_w90_scale_iid{size}"
    write_json(wf / "derivative_stencil_manifest.json", {"sample_count": size})
    write_json(wf / "siesta_hamiltonians" / "derivative_siesta_reference_manifest.json", {"samples_ok": size, "samples_total": size, "samples_failed": 0})
    if g2m:
        write_json(
            wf / "graph2mat_derivative_result" / "predicted_hamiltonians" / "derivative_graph2mat_prediction_manifest.json",
            {"samples_ok": size, "samples_total": size, "samples_failed": 0, "stencil_root": str(wf / "graph2mat_derivative_result")},
        )
    if deeph:
        write_json(
            wf / "deeph_derivative_result" / "predicted_hamiltonians" / "derivative_deeph_prediction_manifest.json",
            {"samples_ok": size, "samples_total": size, "samples_failed": 0, "stencil_root": str(wf / "deeph_derivative_result")},
        )
    return wf


def fake_evaluate(result_dir: Path, *, output_dir: Path, source_model: str, **_: object) -> dict:
    write_json(output_dir / "manifest.json", {"scientific_status": "diagnostic_only", "stencils_ok": 1, "stencils_failed": 0})
    write_csv(output_dir / "derivative_matrix_metrics.csv", [{"sample": "s", "source_model": source_model, "dh_mae_union_eV_per_Ang": 0.1}])
    return {"stencils_ok": 1}


def fake_gate(**_: object) -> dict:
    return {"scientific_status": "internal_diagnostic"}


def fake_gate_write(path: Path, payload: dict) -> None:
    write_json(path, payload)


def fake_comparison(*, output_dir: Path, **_: object) -> dict:
    write_json(output_dir / "derivative_model_comparison_summary.json", {"paired_count": 1})
    write_csv(output_dir / "derivative_model_paired_comparison.csv", [{"sample": "s"}])
    return {"paired_count": 1}


def fake_plots(*, output_dir: Path, **_: object) -> dict:
    write_json(output_dir / "derivative_plot_payload.json", {"available": True, "plots": []})
    write_json(output_dir / "derivative_plot_manifest.json", {"available": True})
    return {"payload": {"available": True}}


class IncrementalDerivativeMetricsTests(unittest.TestCase):
    def patch_postprocess(self):
        return mock.patch.multiple(
            incremental,
            evaluate_derivative_metrics=fake_evaluate,
            build_derivative_gate_report=fake_gate,
            write_gate_json=fake_gate_write,
            build_derivative_model_comparison_summary=fake_comparison,
            write_derivative_plot_outputs=fake_plots,
            active_process_lines=lambda _wf: [],
        )

    def test_ready_size_runs_metrics_and_pending_size_does_not_block(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, self.patch_postprocess():
            run_root = Path(tmp) / "run"
            make_workflow(run_root, 20)
            make_workflow(run_root, 40, g2m=False)

            summary = incremental.run_incremental(run_root, sizes=[20, 40], skip_active=True, overwrite_missing_only=True)

            statuses = {row["dataset_size"]: row["status"] for row in summary["rows"]}
            self.assertEqual(statuses[20], "metrics_completed")
            self.assertEqual(statuses[40], "pending_graph2mat_prediction")
            self.assertTrue((run_root / "derivative_workflows" / "graphene_w90_scale_iid20" / "derivative_metrics" / "graph2mat" / "manifest.json").exists())
            self.assertTrue((run_root / "summary" / "incremental_derivative_metrics_summary.json").exists())

    def test_missing_graph2mat_prediction_is_pending(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, self.patch_postprocess():
            run_root = Path(tmp) / "run"
            make_workflow(run_root, 20, g2m=False)

            row = incremental.run_incremental_derivative_metrics_for_workflow(
                run_root / "derivative_workflows" / "graphene_w90_scale_iid20",
                size=20,
            )

            self.assertEqual(row["status"], "pending_graph2mat_prediction")
            self.assertFalse((run_root / "derivative_workflows" / "graphene_w90_scale_iid20" / "derivative_metrics").exists())

    def test_skip_active_leaves_workflow_untouched(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, self.patch_postprocess():
            run_root = Path(tmp) / "run"
            wf = make_workflow(run_root, 500)
            with mock.patch.object(incremental, "active_process_lines", return_value=["pid cmd graphene_w90_scale_iid500"]):
                row = incremental.run_incremental_derivative_metrics_for_workflow(wf, size=500, skip_active=True)

            self.assertEqual(row["status"], "skipped_active")
            self.assertFalse((wf / "derivative_metrics").exists())


if __name__ == "__main__":
    unittest.main()
