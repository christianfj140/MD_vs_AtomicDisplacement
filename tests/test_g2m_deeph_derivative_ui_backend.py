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
from plot_hamiltonian_derivative_metrics import write_derivative_plot_outputs  # noqa: E402


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
    def write_derivative_tree(
        self,
        run_root: Path,
        name: str,
        *,
        source_model: str,
        fatal_error: bool = False,
        manifest_overrides: dict | None = None,
        row_overrides: dict | None = None,
    ) -> None:
        root = run_root / "common_metrics" / f"{name}_eval" / "derivative_metrics"
        manifest = {
            "scientific_status": "diagnostic_only",
            "finite_difference_method": "central",
            "derivative_units": "eV/Ang",
            "stencils_ok": 1,
            "stencils_failed": 1 if fatal_error else 0,
            "warnings": [],
            "fatal_errors": [{"kind": "missing_required_metadata", "message": "orbital ordering missing"}] if fatal_error else [],
        }
        manifest.update(manifest_overrides or {})
        write_json(
            root / "manifest.json",
            manifest,
        )
        metric_row = {
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
        metric_row.update(row_overrides or {})
        write_csv(
            root / "derivative_matrix_metrics.csv",
            [metric_row],
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

    def write_standalone_derivative_smoke(self, run_root: Path) -> None:
        root = run_root / "derivative_metrics" / "graph2mat"
        write_json(
            root / "manifest.json",
            {
                "scientific_status": "diagnostic_only",
                "finite_difference_method": "central",
                "derivative_units": "eV/Ang",
                "stencils_ok": 2,
                "stencils_failed": 0,
                "warnings": [],
                "fatal_errors": [],
            },
        )
        write_csv(
            root / "derivative_matrix_metrics.csv",
            [
                {
                    "sample": "smoke",
                    "source_model": "graph2mat",
                    "atom_index_zero_based": 0,
                    "axis": "x",
                    "delta_ang": 0.01,
                    "finite_difference_method": "central",
                    "derivative_units": "eV/Ang",
                    "dh_mae_union_eV_per_Ang": 0.2,
                    "dh_rmse_union_eV_per_Ang": 0.3,
                    "dh_relative_frobenius_ref": 0.4,
                    "dh_false_zero_rate": 0.1,
                    "dh_false_nonzero_rate": 0.05,
                }
            ],
        )
        write_csv(
            root / "derivative_hermiticity.csv",
            [
                {
                    "sample": "smoke",
                    "finite_difference_method": "central",
                    "dH_ref_hermiticity_defect": 0.0,
                    "dH_pred_hermiticity_defect": 0.02,
                    "dH_hermiticity_error_delta": 0.02,
                }
            ],
        )
        write_csv(
            root / "stencil_status.csv",
            [
                {
                    "sample": "smoke",
                    "status": "ok",
                    "issue_codes": "",
                    "issue_messages": "",
                }
            ],
        )
        write_json(root / "derivative_delta_stability.json", {"status": "available"})
        write_json(root / "derivative_summary.json", {"scientific_status": "diagnostic_only"})
        write_json(root / "derivative_geometry_validation.json", {"errors": 0})
        write_csv(root / "derivative_support_sweep.csv", [{"threshold": "1e-12", "f1": "0.9"}])
        write_json(
            root / "summary" / "derivative_gate_report.json",
            {
                "scientific_status": "blocked",
                "derivative_winner_claim": "none",
                "message": "Standalone derivative smoke gate report.",
                "blockers": [
                    {
                        "id": "orbital_ordering_metadata_missing_or_inconsistent",
                        "severity": "blocker",
                        "status": "fail",
                        "message": "Orbital ordering evidence is missing.",
                    }
                ],
                "warnings": [],
            },
        )
        write_json(
            root / "summary" / "derivative_plot_payload.json",
            {
                "available": True,
                "plots": [
                    {
                        "id": "dh_mae_by_model",
                        "kind": "grouped_bar",
                        "title": "dH MAE by model",
                        "metrics": [{"key": "dh_mae_union_eV_per_Ang", "label": "dH MAE", "unit": "eV/Ang"}],
                        "rows": [{"method": "Graph2Mat", "dh_mae_union_eV_per_Ang": 0.2}],
                    }
                ],
            },
        )
        write_json(
            root / "summary" / "derivative_plot_manifest.json",
            {
                "available": True,
                "plot_payload": str(root / "summary" / "derivative_plot_payload.json"),
            },
        )
        write_json(run_root / "derivative_artifact_validation.json", {"ok": True})
        write_json(run_root / "derivative_stencil_manifest.json", {"schema": "smoke"})

    def write_standalone_derivative_method(self, run_root: Path, method: str, *, mae: float, rmse: float, frob: float) -> None:
        root = run_root / "derivative_metrics" / method
        write_json(
            root / "manifest.json",
            {
                "scientific_status": "diagnostic_only",
                "finite_difference_method": "central",
                "derivative_units": "eV/Ang",
                "stencils_ok": 2,
                "stencils_failed": 0,
                "warnings": [],
                "fatal_errors": [],
            },
        )
        write_csv(
            root / "derivative_matrix_metrics.csv",
            [
                {
                    "sample": "smoke",
                    "source_model": method,
                    "atom_index_zero_based": 0,
                    "axis": "x",
                    "delta_ang": 0.01,
                    "finite_difference_method": "central",
                    "derivative_units": "eV/Ang",
                    "dh_mae_union_eV_per_Ang": mae,
                    "dh_rmse_union_eV_per_Ang": rmse,
                    "dh_relative_frobenius_ref": frob,
                    "dh_false_zero_rate": 0.1,
                    "dh_false_nonzero_rate": 0.05,
                }
            ],
        )
        write_csv(
            root / "derivative_hermiticity.csv",
            [
                {
                    "sample": "smoke",
                    "finite_difference_method": "central",
                    "dH_ref_hermiticity_defect": 0.0,
                    "dH_pred_hermiticity_defect": 0.02,
                    "dH_hermiticity_error_delta": 0.02,
                }
            ],
        )
        write_csv(
            root / "stencil_status.csv",
            [
                {
                    "sample": "smoke",
                    "status": "ok",
                    "issue_codes": "",
                    "issue_messages": "",
                }
            ],
        )
        write_json(root / "derivative_delta_stability.json", {"status": "available"})
        write_json(root / "derivative_summary.json", {"scientific_status": "diagnostic_only"})
        write_json(root / "derivative_geometry_validation.json", {"errors": 0})
        write_csv(root / "derivative_support_sweep.csv", [{"threshold": "1e-12", "f1": "0.9"}])

    def test_missing_metrics_returns_not_computed(self) -> None:
        with patch.object(pipeline_ui, "resolve_g2m_deeph_run_root", return_value=None):
            payload = pipeline_ui.g2m_deeph_derivative_metrics_payload("missing-run")
        self.assertFalse(payload["available"])
        self.assertTrue(payload["not_computed"])
        self.assertEqual(payload["status"], "not_computed")
        self.assertIn("optional post-processing", payload["message"])
        self.assertIn("H-vs-H metrics", payload["message"])

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
        self.assertIn("optional post-processing", payload["message"])

    def test_derivative_backend_exposes_partial_workflow_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run_demo"
            workflow = run_root / "derivative_workflows" / "graphene_w90_scale_iid20"
            self.write_standalone_derivative_method(workflow, "graph2mat", mae=0.2, rmse=0.3, frob=0.4)
            self.write_standalone_derivative_method(workflow, "deeph", mae=0.1, rmse=0.2, frob=0.3)
            write_json(
                run_root / "summary" / "derivative_plots" / "derivative_plot_payload.json",
                {"available": True, "plots": [{"id": "dh_mae_vs_dataset_size", "rows": [{"x_dataset_size": 20}]}]},
            )
            write_json(workflow / "derivative_incremental_status.json", {"status": "metrics_completed"})

            with patch.object(pipeline_ui, "resolve_g2m_deeph_run_root", return_value=run_root):
                payload = pipeline_ui.g2m_deeph_derivative_metrics_payload("run_demo")

        self.assertTrue(payload["available"])
        self.assertFalse(payload["not_computed"])
        self.assertTrue(payload["workflow_status_rows"])
        self.assertEqual(payload["workflow_status_rows"][0]["status"], "completed_visible")
        self.assertTrue(payload["plot_payload"]["available"])

    def test_derivative_payload_includes_status_warnings_and_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run_demo"
            self.write_derivative_tree(run_root, "graph2mat", source_model="graph2mat", fatal_error=True)
            self.write_derivative_tree(run_root, "deeph", source_model="deeph")
            write_json(
                run_root / "common_metrics" / "summary" / "derivative_gate_report.json",
                {
                    "scientific_status": "internal_diagnostic",
                    "derivative_winner_claim": "none",
                    "message": "Persisted derivative gate report.",
                    "blockers": [],
                    "warnings": [],
                },
            )
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
        self.assertEqual(payload["gate_report"]["scientific_status"], "internal_diagnostic")
        self.assertTrue(any(row["kind"] == "gate_report" for row in payload["artifact_rows"]))
        self.assertIn("technical internal diagnostic", payload["message"])
        self.assertFalse((run_root / "common_metrics" / "summary" / "derivative_plots" / "derivative_plot_payload.json").exists())
        self.assertTrue(payload["comparison_rows"])

    def test_derivative_backend_loads_old_and_dataset_size_plot_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            old_run_root = Path(tmp) / "old_payload_run"
            self.write_derivative_tree(old_run_root, "graph2mat", source_model="graph2mat")
            write_json(
                old_run_root / "common_metrics" / "summary" / "derivative_plot_payload.json",
                {
                    "available": True,
                    "plots": [
                        {
                            "id": "dh_mae_by_model",
                            "kind": "grouped_bar",
                            "rows": [{"method": "Graph2Mat", "dh_mae_union_eV_per_Ang": 0.2}],
                            "metrics": [{"key": "dh_mae_union_eV_per_Ang", "label": "dH MAE", "unit": "eV/Ang"}],
                        }
                    ],
                },
            )
            new_run_root = Path(tmp) / "new_payload_run"
            self.write_derivative_tree(new_run_root, "graph2mat", source_model="graph2mat")
            write_json(
                new_run_root / "common_metrics" / "summary" / "derivative_plot_payload.json",
                {
                    "available": True,
                    "primary_plot_ids": ["dh_mae_vs_dataset_size", "dh_rmse_vs_dataset_size"],
                    "dataset_size_plot_ids": ["dh_mae_vs_dataset_size", "dh_rmse_vs_dataset_size"],
                    "diagnostic_plot_ids": ["dh_mae_by_model"],
                    "plots": [
                        {
                            "id": "dh_mae_vs_dataset_size",
                            "kind": "scatter",
                            "dataset_size_plot": True,
                            "x_key": "x_dataset_size",
                            "x_title": "N_train snapshots",
                            "y_key": "dh_mae_union_eV_per_Ang",
                            "series_key": "model_label",
                            "rows": [
                                {
                                    "model": "graph2mat",
                                    "model_label": "Graph2Mat",
                                    "x_dataset_size": 12,
                                    "x_dataset_size_kind": "N_train",
                                    "dataset_size_source": "frozen_split_manifest",
                                    "dh_mae_union_eV_per_Ang": 0.2,
                                },
                                {
                                    "model": "graph2mat",
                                    "model_label": "Graph2Mat",
                                    "x_dataset_size": 50,
                                    "x_dataset_size_kind": "N_train",
                                    "dataset_size_source": "frozen_split_manifest",
                                    "dh_mae_union_eV_per_Ang": 0.1,
                                },
                            ],
                            "diagnostic_only": True,
                        },
                        {
                            "id": "dh_rmse_vs_dataset_size",
                            "kind": "scatter",
                            "dataset_size_plot": True,
                            "x_key": "x_dataset_size",
                            "rows": [],
                        },
                        {
                            "id": "dh_mae_by_model",
                            "kind": "grouped_bar",
                            "rows": [{"method": "Graph2Mat", "dh_mae_union_eV_per_Ang": 0.15}],
                        }
                    ],
                    "scientific_warnings": [],
                },
            )

            with patch.object(pipeline_ui, "resolve_g2m_deeph_run_root", return_value=old_run_root):
                old_payload = pipeline_ui.g2m_deeph_derivative_metrics_payload("old_payload_run")
            with patch.object(pipeline_ui, "resolve_g2m_deeph_run_root", return_value=new_run_root):
                new_payload = pipeline_ui.g2m_deeph_derivative_metrics_payload("new_payload_run")

        self.assertTrue(old_payload["plot_payload"]["available"])
        self.assertEqual(old_payload["plot_payload"]["plots"][0]["id"], "dh_mae_by_model")
        self.assertTrue(new_payload["plot_payload"]["available"])
        dataset_plot = new_payload["plot_payload"]["plots"][0]
        self.assertEqual(dataset_plot["id"], "dh_mae_vs_dataset_size")
        self.assertEqual(dataset_plot["x_key"], "x_dataset_size")
        self.assertEqual({row["x_dataset_size"] for row in dataset_plot["rows"]}, {12, 50})
        self.assertEqual(new_payload["plot_payload"]["primary_plot_ids"], ["dh_mae_vs_dataset_size", "dh_rmse_vs_dataset_size"])
        self.assertEqual(new_payload["plot_payload"]["dataset_size_plot_ids"], ["dh_mae_vs_dataset_size", "dh_rmse_vs_dataset_size"])
        self.assertEqual(new_payload["plot_payload"]["diagnostic_plot_ids"], ["dh_mae_by_model"])
        self.assertEqual([plot["id"] for plot in new_payload["plot_payload"]["plots"][:2]], ["dh_mae_vs_dataset_size", "dh_rmse_vs_dataset_size"])
        self.assertTrue(new_payload["plot_payload"]["plots"][0]["dataset_size_plot"])

    def test_autograd_derivative_dataset_size_payload_reaches_mixing_ui_plot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run_autograd"
            dataset_root = Path(tmp) / "dataset_iid20"
            dataset_root_40 = Path(tmp) / "dataset_iid40"
            write_json(
                dataset_root / "frozen_split_manifest.json",
                {"split_counts": {"train": 20, "validation": 2, "test": 2}},
            )
            write_json(
                dataset_root_40 / "frozen_split_manifest.json",
                {"split_counts": {"train": 40, "validation": 4, "test": 4}},
            )
            direct_path = run_root / "common_metrics" / "graph2mat_eval" / "predicted_derivative_hamiltonians" / "base" / "dH_pred_atom0_axis0.npz"
            manifest_overrides = {
                "dataset_root": str(dataset_root),
                "dataset_id": "iid20",
                "mode": "add",
                "ratio": 0.25,
                "predicted_derivative_method": "autograd_graph2mat_vectorized",
                "graph2mat_prediction_method": "autograd_vectorized",
                "predicted_delta_ang": None,
            }
            self.write_derivative_tree(
                run_root,
                "graph2mat",
                source_model="graph2mat",
                manifest_overrides=manifest_overrides,
                row_overrides={
                    "reference_derivative_method": "finite_difference_siesta",
                    "predicted_derivative_method": "autograd_graph2mat_vectorized",
                    "graph2mat_prediction_method": "autograd_vectorized",
                    "predicted_delta_ang": "",
                    "direct_prediction_path": str(direct_path),
                },
            )
            derivative_root_40 = run_root / "derivative_workflows" / "iid40" / "derivative_metrics" / "graph2mat"
            write_json(
                derivative_root_40 / "manifest.json",
                {
                    **manifest_overrides,
                    "dataset_root": str(dataset_root_40),
                    "dataset_id": "iid40",
                },
            )
            write_csv(
                derivative_root_40 / "derivative_matrix_metrics.csv",
                [
                    {
                        "sample": "shared",
                        "source_model": "graph2mat",
                        "atom_index_zero_based": 0,
                        "axis": "x",
                        "delta_ang": 0.01,
                        "finite_difference_method": "central",
                        "derivative_units": "eV/Ang",
                        "dh_mae_union_eV_per_Ang": 0.12,
                        "dh_rmse_union_eV_per_Ang": 0.2,
                        "dh_relative_frobenius_ref": 0.3,
                        "reference_derivative_method": "finite_difference_siesta",
                        "predicted_derivative_method": "autograd_graph2mat_vectorized",
                        "graph2mat_prediction_method": "autograd_vectorized",
                        "predicted_delta_ang": "",
                        "direct_prediction_path": str(direct_path),
                    }
                ],
            )
            write_csv(derivative_root_40 / "derivative_hermiticity.csv", [])
            write_csv(derivative_root_40 / "stencil_status.csv", [])
            plot_result = write_derivative_plot_outputs(
                derivative_roots=[
                    run_root / "common_metrics" / "graph2mat_eval" / "derivative_metrics",
                    derivative_root_40,
                ],
                output_dir=run_root / "common_metrics" / "summary" / "derivative_plots",
            )

            with patch.object(pipeline_ui, "resolve_g2m_deeph_run_root", return_value=run_root):
                payload = pipeline_ui.g2m_deeph_derivative_metrics_payload("run_autograd")

            self.assertTrue(Path(plot_result["payload_path"]).is_file())
            plot = next(plot for plot in payload["plot_payload"]["plots"] if plot["id"] == "dh_mae_vs_dataset_size")
            self.assertTrue(plot["dataset_size_plot"])
            self.assertEqual(plot["series_key"], "series_label")
            self.assertEqual({row["x_dataset_size"] for row in plot["rows"]}, {20, 40})
            for row in plot["rows"]:
                self.assertEqual(row["series_label"], "mode=add · ratio=0.25 · Graph2Mat")
                self.assertEqual(row["predicted_derivative_methods"], ["autograd_graph2mat_vectorized"])
                self.assertEqual(row["graph2mat_prediction_methods"], ["autograd_vectorized"])
                self.assertEqual(row["reference_derivative_methods"], ["finite_difference_siesta"])
                self.assertEqual(row["direct_prediction_count"], 1)
        index_html = (REPO_ROOT / "Comparison" / "ui" / "index.html").read_text(encoding="utf-8")
        mixing_html = index_html.split('<section id="view-mixing-datasets" class="view">', 1)[1].split(
            '<section id="view-terminal" class="view">',
            1,
        )[0]
        self.assertIn('id="g2m-deeph-derivative-mae-dataset-chart"', mixing_html)

    def test_derivative_backend_combines_multiple_run_plot_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            roots = []
            for name, value in (("run_a", 0.2), ("run_b", 0.1)):
                run_root = Path(tmp) / name
                roots.append(run_root)
                self.write_derivative_tree(run_root, "graph2mat", source_model="graph2mat")
                write_json(
                    run_root / "common_metrics" / "summary" / "derivative_plot_payload.json",
                    {
                        "available": True,
                        "diagnostic_plot_ids": ["dh_mae_by_model"],
                        "plots": [
                            {
                                "id": "dh_mae_by_model",
                                "kind": "grouped_bar",
                                "rows": [{"method": "Graph2Mat", "dh_mae_union_eV_per_Ang": value}],
                            }
                        ],
                    },
                )

            with patch.object(pipeline_ui, "resolve_g2m_deeph_run_root", side_effect=roots):
                payload = pipeline_ui.g2m_deeph_derivative_metrics_multi_payload(["run_a", "run_b"])

        plot = payload["plot_payload"]["plots"][0]
        self.assertEqual(plot["id"], "dh_mae_by_model")
        self.assertEqual(len(plot["rows"]), 2)
        self.assertEqual({row["run_id"] for row in plot["rows"]}, {"run_a", "run_b"})
        self.assertEqual({row["combined_series"] for row in plot["rows"]}, {"Graph2Mat"})
        self.assertEqual(payload["run_ids"], ["run_a", "run_b"])

    def test_derivative_backend_combines_dataset_size_plot_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            roots = []
            dataset_size_plot_ids = [
                "dh_mae_vs_dataset_size",
                "robust_relative_frobenius_vs_dataset_size",
                "dh_mae_vs_dataset_size_by_delta",
                "derivative_hermiticity_vs_dataset_size",
                "onsite_offsite_derivative_error_vs_dataset_size",
            ]
            for name, size, value in (("run_a", 20, 0.2), ("run_b", 100, 0.1)):
                run_root = Path(tmp) / name
                roots.append(run_root)
                self.write_derivative_tree(run_root, "graph2mat", source_model="graph2mat")
                write_json(
                    run_root / "common_metrics" / "summary" / "derivative_plot_payload.json",
                    {
                        "available": True,
                        "primary_plot_ids": dataset_size_plot_ids,
                        "dataset_size_plot_ids": dataset_size_plot_ids,
                        "diagnostic_plot_ids": ["dh_mae_by_model"],
                        "plots": [
                            {
                                "id": plot_id,
                                "kind": "scatter",
                                "dataset_size_plot": True,
                                "x_key": "x_dataset_size",
                                "x_title": "N_train snapshots",
                                "y_key": "dh_mae_union_eV_per_Ang",
                                "series_key": "series_label" if plot_id.endswith("_by_delta") else "model_label",
                                "rows": [
                                    {
                                        "model": "graph2mat",
                                        "model_label": "Graph2Mat",
                                        "series_label": "Graph2Mat delta=0.01",
                                        "x_dataset_size": size,
                                        "x_dataset_size_kind": "N_train",
                                        "dataset_size_source": "synthetic_fixture",
                                        "dh_mae_union_eV_per_Ang": value,
                                        "dh_relative_frobenius_union_robust": value + 0.1,
                                        "dH_pred_hermiticity_defect": value + 0.2,
                                        "dh_onsite_relative_frobenius_robust": value + 0.3,
                                    }
                                ],
                            }
                            for plot_id in dataset_size_plot_ids
                        ] + [
                            {
                                "id": "dh_mae_by_model",
                                "kind": "grouped_bar",
                                "rows": [{"method": "Graph2Mat", "dh_mae_union_eV_per_Ang": value}],
                            },
                        ],
                    },
                )

            with patch.object(pipeline_ui, "resolve_g2m_deeph_run_root", side_effect=roots):
                payload = pipeline_ui.g2m_deeph_derivative_metrics_multi_payload(["run_a", "run_b"])

        plot_payload = payload["plot_payload"]
        self.assertEqual(plot_payload["primary_plot_ids"], dataset_size_plot_ids)
        self.assertEqual(plot_payload["dataset_size_plot_ids"], dataset_size_plot_ids)
        self.assertEqual(plot_payload["diagnostic_plot_ids"], ["dh_mae_by_model"])
        plots = {plot["id"]: plot for plot in plot_payload["plots"]}
        for plot_id in dataset_size_plot_ids:
            plot = plots[plot_id]
            self.assertTrue(plot["dataset_size_plot"])
            self.assertEqual(plot["x_key"], "x_dataset_size")
            self.assertIn(plot["series_key"], {"model_label", "series_label"})
            self.assertEqual({row["x_dataset_size"] for row in plot["rows"]}, {20, 100})
            self.assertEqual({row["model_label"] for row in plot["rows"]}, {"Graph2Mat"})
            self.assertEqual({row["run_id"] for row in plot["rows"]}, {"run_a", "run_b"})
            self.assertEqual(
                {row["combined_series"] for row in plot["rows"]},
                {"Graph2Mat delta=0.01"} if plot["series_key"] == "series_label" else {"Graph2Mat"},
            )
            rows_by_run = {row["run_id"]: row for row in plot["rows"]}
            self.assertEqual(rows_by_run["run_a"]["x_dataset_size"], 20)
            self.assertEqual(rows_by_run["run_b"]["x_dataset_size"], 100)
            self.assertEqual(rows_by_run["run_a"]["dataset_size_source"], "synthetic_fixture")

    def test_blocked_gate_report_renders_blockers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run_blocked"
            self.write_derivative_tree(run_root, "graph2mat", source_model="graph2mat")
            write_json(
                run_root / "common_metrics" / "summary" / "derivative_gate_report.json",
                {
                    "scientific_status": "blocked",
                    "blockers": [
                        {
                            "gate_id": "derivative_geometry_validation_failed",
                            "severity": "blocker",
                            "status": "fail",
                            "message": "Wrong atom was displaced.",
                        }
                    ],
                    "warnings": [],
                },
            )
            with patch.object(pipeline_ui, "resolve_g2m_deeph_run_root", return_value=run_root):
                payload = pipeline_ui.g2m_deeph_derivative_metrics_payload("run_blocked")

        self.assertEqual(payload["status"], "blocked")
        self.assertEqual(payload["gate_report"]["scientific_status"], "blocked")
        self.assertEqual(payload["gate_report"]["derivative_winner_claim"], "none")
        self.assertTrue(payload["gate_report"]["gate_rows"])
        self.assertIn("Wrong atom", payload["gate_report"]["gate_rows"][0]["message"])

    def test_existing_model_comparison_artifacts_are_linked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run_comparison"
            self.write_derivative_tree(run_root, "graph2mat", source_model="graph2mat")
            self.write_derivative_tree(run_root, "deeph", source_model="deeph")
            comparison_root = run_root / "common_metrics" / "summary" / "derivative_model_comparison"
            write_json(
                comparison_root / "derivative_model_comparison_summary.json",
                {"claim_status": "diagnostic_only", "winner": None},
            )
            write_csv(
                comparison_root / "derivative_model_paired_comparison.csv",
                [
                    {
                        "base_sample_id": "base",
                        "atom_index_zero_based": "0",
                        "axis": "x",
                        "delta_ang": "0.01",
                        "finite_difference_method": "central",
                        "delta_graph2mat_minus_deeph_dh_mae_union_eV_per_Ang": "0.1",
                    }
                ],
            )
            with patch.object(pipeline_ui, "resolve_g2m_deeph_run_root", return_value=run_root):
                payload = pipeline_ui.g2m_deeph_derivative_metrics_payload("run_comparison")

        kinds = {row["kind"]: row for row in payload["artifact_rows"]}
        self.assertTrue(kinds["model_comparison_summary"]["exists"])
        self.assertTrue(kinds["model_paired_comparison"]["exists"])
        self.assertEqual(payload["paired_comparison_rows"][0]["base_sample_id"], "base")

    def test_technical_presentation_gate_maps_to_presentation_ready_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run_presentation"
            self.write_derivative_tree(run_root, "graph2mat", source_model="graph2mat")
            write_json(
                run_root / "common_metrics" / "summary" / "derivative_gate_report.json",
                {
                    "scientific_status": "technical_presentation",
                    "derivative_winner_claim": "none",
                    "blockers": [],
                    "warnings": [],
                },
            )
            with patch.object(pipeline_ui, "resolve_g2m_deeph_run_root", return_value=run_root):
                payload = pipeline_ui.g2m_deeph_derivative_metrics_payload("run_presentation")

        self.assertEqual(payload["status"], "presentation_ready")
        self.assertIsNone(payload["winner"])

    def test_standalone_derivative_smoke_roots_are_discoverable_and_renderable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            results_root = Path(tmp) / "results"
            run_root = results_root / "derivative_smoke" / "graph2mat_derivative_result"
            self.write_standalone_derivative_smoke(run_root)
            with patch.object(pipeline_ui, "RESULTS_ROOT", results_root), \
                 patch.object(pipeline_ui.G2M_DEEPH_RUNNER, "plot_runs", return_value={"runs": [], "default_selected_run_ids": []}), \
                 patch.object(pipeline_ui.G2M_DEEPH_RUNNER, "status", return_value={"run_root": ""}):
                selector_payload = pipeline_ui._g2m_deeph_plot_runs_payload()
                resolved = pipeline_ui.resolve_g2m_deeph_run_root("graph2mat_derivative_result")
                payload = pipeline_ui.g2m_deeph_derivative_metrics_payload("graph2mat_derivative_result")

        self.assertEqual(selector_payload["runs"][0]["run_id"], "graph2mat_derivative_result")
        self.assertEqual(resolved, run_root)
        self.assertTrue(payload["available"])
        self.assertEqual(payload["status"], "blocked")
        self.assertTrue(payload["status_rows"])
        self.assertTrue(payload["comparison_rows"])
        self.assertTrue(payload["artifact_rows"])
        self.assertEqual(payload["gate_report"]["gate_rows"][0]["gate"], "orbital_ordering_metadata_missing_or_inconsistent")
        self.assertTrue(payload["plot_payload"]["available"])
        self.assertTrue(payload["plot_payload"]["plots"])
        self.assertTrue(any(row["kind"] == "artifact_validation" and row["exists"] for row in payload["artifact_rows"]))

    def test_nested_derivative_postprocess_root_is_discoverable_and_renderable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            results_root = Path(tmp) / "results"
            run_root = results_root / "e2e_smoke" / "e2e_smoke" / "derivative_postprocess"
            self.write_standalone_derivative_smoke(run_root)
            plot_payload_path = run_root / "derivative_metrics" / "graph2mat" / "summary" / "derivative_plot_payload.json"
            plot_payload = json.loads(plot_payload_path.read_text(encoding="utf-8"))
            plot_payload_path.unlink()
            write_json(run_root / "derivative_metrics" / "summary" / "derivative_plots" / "derivative_plot_payload.json", plot_payload)
            with patch.object(pipeline_ui, "RESULTS_ROOT", results_root), \
                 patch.object(pipeline_ui.G2M_DEEPH_RUNNER, "plot_runs", return_value={"runs": [], "default_selected_run_ids": []}), \
                 patch.object(pipeline_ui.G2M_DEEPH_RUNNER, "status", return_value={"run_root": ""}):
                selector_payload = pipeline_ui._g2m_deeph_plot_runs_payload()
                resolved = pipeline_ui.resolve_g2m_deeph_run_root("derivative_postprocess")
                payload = pipeline_ui.g2m_deeph_derivative_metrics_payload("derivative_postprocess")

        self.assertEqual(selector_payload["runs"][0]["run_id"], "derivative_postprocess")
        self.assertEqual(resolved, run_root)
        self.assertTrue(payload["available"])
        self.assertTrue(payload["plot_payload"]["available"])
        self.assertTrue(any(row["kind"] == "artifact_validation" and row["exists"] for row in payload["artifact_rows"]))

    def test_training_sweep_derivative_workflow_root_is_renderable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            results_root = Path(tmp) / "results"
            run_root = results_root / "e2e_smoke_12snap_20ep" / "e2e_smoke_12snap_20ep"
            derivative_root = run_root / "sweep" / "derivative_workflows" / "graphene_w90_scale_iid12"
            self.write_standalone_derivative_method(derivative_root, "graph2mat", mae=0.2, rmse=0.3, frob=0.4)
            self.write_standalone_derivative_method(derivative_root, "deeph", mae=0.1, rmse=0.2, frob=0.3)
            write_json(
                derivative_root / "derivative_metrics" / "summary" / "derivative_gate_report.json",
                {
                    "scientific_status": "blocked",
                    "derivative_winner_claim": "none",
                    "blockers": [
                        {
                            "id": "support_pattern_discontinuity",
                            "severity": "blocker",
                            "status": "fail",
                            "message": "Support pattern changes across the stencil.",
                        }
                    ],
                    "warnings": [],
                },
            )
            write_json(
                derivative_root / "derivative_metrics" / "summary" / "derivative_plots" / "derivative_plot_payload.json",
                {
                    "available": True,
                    "plots": [{"id": "paired_graph2mat_vs_deeph", "rows": [{"model": "Graph2Mat"}]}],
                },
            )
            write_json(
                derivative_root / "derivative_metrics" / "summary" / "derivative_plots" / "derivative_plot_manifest.json",
                {"available": True},
            )
            write_csv(
                derivative_root
                / "derivative_metrics"
                / "summary"
                / "derivative_model_comparison"
                / "derivative_model_paired_comparison.csv",
                [
                    {
                        "base_sample_id": "base",
                        "atom_index_zero_based": "0",
                        "axis": "x",
                        "delta_ang": "0.01",
                        "finite_difference_method": "central",
                        "delta_graph2mat_minus_deeph_dh_mae_union_eV_per_Ang": "0.1",
                    }
                ],
            )
            write_json(
                derivative_root
                / "derivative_metrics"
                / "summary"
                / "derivative_model_comparison"
                / "derivative_model_comparison_summary.json",
                {"paired_count": 1},
            )
            write_json(derivative_root / "derivative_artifact_validation.json", {"status": "ok"})
            write_json(derivative_root / "derivative_stencil_manifest.json", {"schema": "smoke"})
            write_json(derivative_root / "derivative_workflow_manifest.json", {"status": "completed"})
            write_json(
                run_root / "sweep" / "training_sweep_manifest.json",
                {
                    "derivative_workflows": [
                        {
                            "run_id": "graphene_w90_scale_iid12",
                            "derivative_workflow_status": "completed",
                            "derivative_workflow_manifest_path": str(derivative_root / "derivative_workflow_manifest.json"),
                        }
                    ]
                },
            )
            with patch.object(pipeline_ui, "RESULTS_ROOT", results_root), \
                 patch.object(
                     pipeline_ui.G2M_DEEPH_RUNNER,
                     "plot_runs",
                     return_value={
                         "runs": [
                             {
                                 "run_id": "e2e_smoke_12snap_20ep",
                                 "run_root": str(run_root),
                                 "status": "completed",
                             }
                         ],
                         "default_selected_run_ids": [],
                     },
                 ), \
                 patch.object(pipeline_ui.G2M_DEEPH_RUNNER, "status", return_value={"run_root": ""}):
                selector_payload = pipeline_ui._g2m_deeph_plot_runs_payload()
                resolved = pipeline_ui.resolve_g2m_deeph_run_root("e2e_smoke_12snap_20ep")
                payload = pipeline_ui.g2m_deeph_derivative_metrics_payload("e2e_smoke_12snap_20ep")

        self.assertEqual(selector_payload["runs"][0]["run_id"], "e2e_smoke_12snap_20ep")
        self.assertEqual(resolved, run_root)
        self.assertTrue(payload["available"])
        self.assertEqual(payload["status"], "blocked")
        self.assertEqual(len(payload["status_rows"]), 2)
        self.assertEqual(len(payload["paired_comparison_rows"]), 1)
        self.assertTrue(payload["plot_payload"]["available"])
        self.assertTrue(any(row["kind"] == "artifact_validation" and row["exists"] for row in payload["artifact_rows"]))

    def test_sweep_run_entry_uses_manifest_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "results" / "manifest_only" / "manifest_only"
            write_json(
                run_root / "sweep" / "training_sweep_manifest.json",
                {
                    "status": "running",
                    "planned_runs": [
                        {"model": "graph2mat", "dataset_id": "joint_a", "config_id": "g1"},
                        {"model": "deeph", "dataset_id": "joint_a", "config_id": "d1"},
                        {"model": "graph2mat", "dataset_id": "joint_b", "config_id": "g2"},
                    ],
                    "runs": [
                        {"status": "completed", "model": "graph2mat"},
                        {"status": "failed", "model": "deeph"},
                        {"status": "completed", "model": "graph2mat"},
                    ],
                },
            )

            payload = pipeline_ui._g2m_deeph_sweep_run_entry(run_root)

        self.assertEqual(payload["status"], "running")
        self.assertEqual(payload["planned_runs"], 3)
        self.assertEqual(payload["completed_runs"], 2)
        self.assertEqual(payload["failed_runs"], 1)

    def test_sweep_run_entry_uses_runner_status_counts_without_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "results" / "runner_only" / "runner_only"
            write_json(
                run_root / "runner_status.json",
                {
                    "status": {
                        "running": True,
                        "stage": "training_sweep",
                        "training_sweep": {
                            "enabled": True,
                            "completed": 8,
                            "failed": 1,
                            "total": 16,
                        },
                    }
                },
            )

            payload = pipeline_ui._g2m_deeph_sweep_run_entry(run_root)

        self.assertEqual(payload["planned_runs"], 16)
        self.assertEqual(payload["completed_runs"], 8)
        self.assertEqual(payload["failed_runs"], 1)

    def test_plot_runs_payload_reports_sweep_progress_for_fallback_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            results_root = Path(tmp) / "results"
            run_root = results_root / "graphene_5x2_snapshot_scaling_12_1300_600epochs_1train" / "graphene_5x2_snapshot_scaling_12_1300_600epochs_1train_20260622_173854"
            write_json(
                run_root / "sweep" / "training_sweep_manifest.json",
                {
                    "status": "running",
                    "planned_runs": [
                        {"model": "graph2mat", "dataset_id": f"joint_{index}", "config_id": f"g{index}"}
                        for index in range(16)
                    ],
                    "completed_runs": [{"config_id": f"done_{index}"} for index in range(8)],
                    "failed_runs": [{"config_id": "failed_0"}],
                },
            )
            write_json(
                run_root / "runner_status.json",
                {
                    "status": {
                        "running": True,
                        "stage": "training_sweep",
                        "run_id": run_root.name,
                        "run_root": str(run_root),
                        "training_sweep": {
                            "enabled": True,
                            "completed": 8,
                            "failed": 1,
                            "total": 16,
                            "active_model": "graph2mat_parallel",
                        },
                    }
                },
            )
            with patch.object(pipeline_ui, "RESULTS_ROOT", results_root),                  patch.object(pipeline_ui.G2M_DEEPH_RUNNER, "plot_runs", return_value={"runs": [], "default_selected_run_ids": []}),                  patch.object(pipeline_ui.G2M_DEEPH_RUNNER, "status", return_value={"run_root": ""}):
                payload = pipeline_ui._g2m_deeph_plot_runs_payload()

        self.assertEqual(len(payload["runs"]), 1)
        entry = payload["runs"][0]
        self.assertEqual(entry["run_id"], run_root.name)
        self.assertEqual(entry["status"], "running")
        self.assertEqual(entry["planned_runs"], 16)
        self.assertEqual(entry["completed_runs"], 8)
        self.assertEqual(entry["failed_runs"], 1)
        self.assertTrue(entry["has_training_sweep"])

    def test_hamiltonian_metrics_root_resolves_nested_inner_while_derivatives_stay_outer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            outer_root = Path(tmp) / "outer"
            inner_root = outer_root / outer_root.name
            (inner_root / "benchmark_manifest.yaml").parent.mkdir(parents=True, exist_ok=True)
            (inner_root / "benchmark_manifest.yaml").write_text("schema: 1\n", encoding="utf-8")
            write_json(
                inner_root
                / "sweep"
                / "graph2mat"
                / "graphene_w90_scale_iid12"
                / "G2M"
                / "metrics"
                / "graph2mat"
                / "eval_input"
                / "metrics"
                / "manifest.json",
                {"status": "ok"},
            )
            write_json(
                inner_root
                / "sweep"
                / "deeph"
                / "graphene_w90_scale_iid12"
                / "DH"
                / "metrics"
                / "deeph"
                / "eval"
                / "metrics"
                / "manifest.json",
                {"status": "ok"},
            )

            workflow_root = outer_root / "derivative_workflows" / "graphene_w90_scale_iid12"
            self.write_standalone_derivative_method(workflow_root, "graph2mat", mae=0.2, rmse=0.3, frob=0.4)
            self.write_standalone_derivative_method(workflow_root, "deeph", mae=0.1, rmse=0.2, frob=0.3)

            # Non-nested layout: run_root itself already has sweep/, so it stays as-is.
            non_nested_root = Path(tmp) / "non_nested"
            (non_nested_root / "sweep").mkdir(parents=True, exist_ok=True)

            self.assertEqual(pipeline_ui._g2m_deeph_hamiltonian_metrics_root(outer_root), inner_root)
            self.assertEqual(pipeline_ui._g2m_deeph_hamiltonian_metrics_root(inner_root), inner_root)
            self.assertEqual(
                pipeline_ui._g2m_deeph_hamiltonian_metrics_root(non_nested_root),
                non_nested_root,
            )

            with patch.object(pipeline_ui, "resolve_g2m_deeph_run_root", return_value=outer_root):
                payload = pipeline_ui.g2m_deeph_derivative_metrics_payload("outer")

        self.assertTrue(payload["available"])
        self.assertEqual(Path(payload["hamiltonian_metrics_root"]), inner_root)
        self.assertEqual(payload["run_root"], str(outer_root))
        self.assertTrue(payload["comparison_rows"])
        comparison_by_method = {row["method"]: row for row in payload["comparison_rows"]}
        self.assertAlmostEqual(comparison_by_method["Graph2Mat"]["dh_mae_union_eV_per_Ang"], 0.2)
        self.assertAlmostEqual(comparison_by_method["DeepH"]["dh_mae_union_eV_per_Ang"], 0.1)

    def test_standalone_derivative_smoke_paired_comparison_rows_are_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            results_root = Path(tmp) / "results"
            run_root = results_root / "derivative_smoke" / "deeph_derivative_result"
            self.write_standalone_derivative_method(run_root, "graph2mat", mae=0.2, rmse=0.3, frob=0.4)
            self.write_standalone_derivative_method(run_root, "deeph", mae=0.1, rmse=0.2, frob=0.3)
            write_json(
                run_root / "derivative_metrics" / "summary" / "derivative_gate_report.json",
                {
                    "scientific_status": "blocked",
                    "derivative_winner_claim": "none",
                    "blockers": [],
                    "warnings": [],
                },
            )
            write_json(
                run_root / "derivative_metrics" / "summary" / "derivative_plot_payload.json",
                {
                    "available": True,
                    "plots": [],
                },
            )
            write_csv(
                run_root / "derivative_metrics" / "summary" / "derivative_model_comparison" / "derivative_model_paired_comparison.csv",
                [
                    {
                        "base_sample_id": "base",
                        "atom_index_zero_based": "0",
                        "axis": "x",
                        "delta_ang": "0.01",
                        "finite_difference_method": "central",
                        "delta_graph2mat_minus_deeph_dh_mae_union_eV_per_Ang": "0.1",
                    }
                ],
            )
            write_json(run_root / "derivative_artifact_validation.json", {"ok": True})
            write_json(run_root / "derivative_stencil_manifest.json", {"schema": "smoke"})
            with patch.object(pipeline_ui, "RESULTS_ROOT", results_root), \
                 patch.object(pipeline_ui.G2M_DEEPH_RUNNER, "plot_runs", return_value={"runs": [], "default_selected_run_ids": []}), \
                 patch.object(pipeline_ui.G2M_DEEPH_RUNNER, "status", return_value={"run_root": ""}):
                payload = pipeline_ui.g2m_deeph_derivative_metrics_payload("deeph_derivative_result")

        self.assertTrue(payload["available"])
        self.assertEqual(len(payload["paired_comparison_rows"]), 1)
        self.assertEqual(payload["paired_comparison_rows"][0]["base_sample_id"], "base")


if __name__ == "__main__":
    unittest.main()
