import csv
import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "Comparison" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from g2m_deeph_metrics import (  # noqa: E402
    aggregate_common_metrics,
    build_common_plot_payload,
    validate_no_forbidden_references,
)
from deeph_prediction_adapter import EQUIVALENCE_PROVEN_RAW_GLOBAL  # noqa: E402


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row}) or ["status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_method_metrics(
    root: Path,
    *,
    sample_ids: list[str],
    h_mae: float,
    uses_s_ref: bool = True,
    kpoint_enabled: bool = True,
    diagnostic_only: bool = False,
) -> None:
    matrix_rows = []
    spectral_rows = []
    dos_rows = []
    sparse_rows = []
    for index, sample_id in enumerate(sample_ids):
        mae = h_mae + index * 0.001
        diagnostic = "true" if diagnostic_only else "false"
        matrix_rows.append(
            {
                "sample": sample_id,
                "row_type": "weighted_sample",
                "h_mae_eV": mae,
                "h_rmse_eV": mae * 2,
                "h_mse_eV2": mae * mae,
                "relative_frobenius": mae / 10,
                "hermiticity_pred": mae / 100,
                "deeph_diagnostic_only": diagnostic,
            }
        )
        spectral_rows.append(
            {
                "sample": sample_id,
                "global_rmse_eV": mae * 3,
                "low_energy_rmse_eV": mae * 4,
                "fermi_window_rmse_eV": mae * 5,
                "frontier_window_rmse_eV": mae * 6,
                "deeph_diagnostic_only": diagnostic,
            }
        )
        dos_rows.append(
            {
                "sample": sample_id,
                "dos_mae_500_fermi_window": mae * 7,
                "dos_wasserstein_eV": mae * 8,
                "deeph_diagnostic_only": diagnostic,
            }
        )
        sparse_rows.append(
            {
                "sample": sample_id,
                "mae_union_eV": mae,
                "rmse_union_eV": mae * 2,
                "mse_union_eV2": mae * mae,
                "r2_union": 0.9,
                "support_precision": 1.0,
                "support_recall": 0.95,
                "support_f1": 0.974,
            }
        )
    write_csv(root / "kpoint_matrix_metrics.csv", matrix_rows)
    write_csv(root / "kpoint_spectral_metrics.csv", spectral_rows)
    write_csv(root / "kpoint_dos_metrics.csv", dos_rows)
    write_csv(root / "sparse_metrics.csv", sparse_rows)
    write_json(
        root / "manifest.json",
        {
            "samples_compared": len(sample_ids),
            "samples_failed": 0,
            "fatal_errors": [],
            "warnings": [],
            "kpoint_metrics_enabled": kpoint_enabled,
            "uses_reference_overlap_k": uses_s_ref,
        },
    )


def write_deeph_adapter_manifest(metrics_root: Path, *, proven: bool = True) -> None:
    status = EQUIVALENCE_PROVEN_RAW_GLOBAL if proven else "invalid_orbital_order_unknown"
    write_json(
        metrics_root.parent / "adapter_manifest.json",
        {
            "adapter_equivalence_statuses": [status],
            "diagnostic_only_count": 0 if proven else 1,
            "raw_global_equivalence_proven_count": 1 if proven else 0,
            "robust_matrix_metrics_allowed": proven,
            "samples": [
                {
                    "sample_id": "s0",
                    "adapter_equivalence_status": status,
                    "diagnostic_only": not proven,
                }
            ],
        },
    )


def write_derivative_metrics(root: Path, *, status: str = "diagnostic_only") -> None:
    derivative_root = root / "derivative_metrics"
    write_json(
        derivative_root / "manifest.json",
        {
            "schema_version": "hamiltonian_derivative_metrics_v1",
            "scientific_status": status,
            "force_constants_used": False,
            "paper_level": False,
            "stencils_total": 1,
            "stencils_ok": 1,
            "stencils_failed": 0,
            "warnings": [],
            "fatal_errors": [],
        },
    )
    write_csv(
        derivative_root / "derivative_matrix_metrics.csv",
        [
            {
                "sample": "dH_s0",
                "dh_mae_union_eV_per_Ang": 0.2,
                "dh_rmse_union_eV_per_Ang": 0.3,
                "dh_support_f1": 0.9,
                "dh_relative_frobenius_ref": 0.4,
                "comparison_status": "diagnostic_only",
            }
        ],
    )
    write_csv(
        derivative_root / "derivative_hermiticity.csv",
        [
            {
                "sample": "dH_s0",
                "dH_ref_hermiticity_defect": 0.0,
                "dH_pred_hermiticity_defect": 0.01,
            }
        ],
    )


def write_failed_derivative_metrics(root: Path) -> None:
    derivative_root = root / "derivative_metrics"
    write_json(
        derivative_root / "manifest.json",
        {
            "schema_version": "hamiltonian_derivative_metrics_v1",
            "scientific_status": "diagnostic_only",
            "force_constants_used": False,
            "paper_level": False,
            "stencils_total": 1,
            "stencils_ok": 0,
            "stencils_failed": 1,
            "warnings": [],
            "fatal_errors": [{"kind": "incomplete_derivative_stencil"}],
        },
    )


def write_valid_dataset_manifests(root: Path, sample_ids: list[str]) -> tuple[Path, Path]:
    dataset_manifest = root / "benchmark_dataset_manifest.json"
    split_manifest = root / "frozen_split_manifest.json"
    write_json(
        dataset_manifest,
        {
            "benchmark_ready": True,
            "generation_mode": "clean_one_pass",
            "warnings": [],
        },
    )
    write_json(
        split_manifest,
        {
            "valid": True,
            "split_hash": "split-hash",
            "rows": [
                {
                    "sample_id": sample_id,
                    "split": "test",
                    "sample_dir": str(root / "samples" / sample_id),
                    "hamiltonian_path": str(root / "samples" / sample_id / "graphene.HSX"),
                }
                for sample_id in sample_ids
            ],
        },
    )
    return dataset_manifest, split_manifest


class Graph2MatDeepHCommonMetricsTests(unittest.TestCase):
    def test_deeph_low_energy_metric_policy_matches_common_kpoint_policy(self) -> None:
        try:
            import numpy as np
            deeph_metrics = importlib.import_module("evaluate_deeph_kpoint_metrics")
        except ModuleNotFoundError as exc:
            self.skipTest(f"DeepH metric dependency unavailable: {exc.name}")

        first = deeph_metrics.low_energy_metrics_from_eigenvalues(
            np.asarray([0.0, 1.0, 3.0], dtype=float),
            np.asarray([0.1, 0.8, 4.0], dtype=float),
            n_states=2,
            alignment="none",
        )
        second = deeph_metrics.low_energy_metrics_from_eigenvalues(
            np.asarray([0.0, 2.0, 5.0], dtype=float),
            np.asarray([0.0, 2.3, 6.0], dtype=float),
            n_states=2,
            alignment="none",
        )

        self.assertEqual(first["low_energy_n_states"], 2)
        self.assertAlmostEqual(first["low_energy_rmse_eV"], ((0.1**2 + 0.2**2) / 2.0) ** 0.5)
        weighted = deeph_metrics.weighted_metric_rmse(
            [
                {"k_weight": 0.25, "low_energy_rmse_eV": first["low_energy_rmse_eV"]},
                {"k_weight": 0.75, "low_energy_rmse_eV": second["low_energy_rmse_eV"]},
            ],
            "low_energy_rmse_eV",
        )
        self.assertTrue(np.isfinite(weighted))

    def test_ml_prediction_cannot_be_selected_as_reference(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "ML_prediction.HSX"):
            validate_no_forbidden_references(
                {
                    "rows": [
                        {
                            "sample_id": "s0",
                            "split": "test",
                            "hamiltonian_path": "/tmp/s0/ML_prediction.HSX",
                        }
                    ]
                }
            )

    def test_mismatched_sample_ids_fail_comparability(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_manifest, split_manifest = write_valid_dataset_manifests(root, ["s0"])
            write_method_metrics(root / "g2m", sample_ids=["s0"], h_mae=0.2)
            write_method_metrics(root / "deeph", sample_ids=["s1"], h_mae=0.1)

            manifest = aggregate_common_metrics(
                graph2mat_metrics_root=root / "g2m",
                deeph_metrics_root=root / "deeph",
                output_dir=root / "summary",
                frozen_split_manifest_path=split_manifest,
                dataset_manifest_path=dataset_manifest,
            )

            self.assertEqual(manifest["status"], "invalid_incompatible_splits")
            self.assertIsNone(manifest["recommendation"]["winner"])

    def test_missing_reference_overlap_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_manifest, split_manifest = write_valid_dataset_manifests(root, ["s0"])
            write_method_metrics(root / "g2m", sample_ids=["s0"], h_mae=0.2, uses_s_ref=False)
            write_method_metrics(root / "deeph", sample_ids=["s0"], h_mae=0.1)

            manifest = aggregate_common_metrics(
                graph2mat_metrics_root=root / "g2m",
                deeph_metrics_root=root / "deeph",
                output_dir=root / "summary",
                frozen_split_manifest_path=split_manifest,
                dataset_manifest_path=dataset_manifest,
            )

            self.assertEqual(manifest["status"], "diagnostic_only")
            self.assertTrue(any(warning["kind"] == "graph2mat_missing_s_ref" for warning in manifest["warnings"]))
            self.assertIsNone(manifest["recommendation"]["winner"])

    def test_unsupported_kgrid_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_manifest, split_manifest = write_valid_dataset_manifests(root, ["s0"])
            write_method_metrics(root / "g2m", sample_ids=["s0"], h_mae=0.2, kpoint_enabled=False)
            write_method_metrics(root / "deeph", sample_ids=["s0"], h_mae=0.1)

            manifest = aggregate_common_metrics(
                graph2mat_metrics_root=root / "g2m",
                deeph_metrics_root=root / "deeph",
                output_dir=root / "summary",
                frozen_split_manifest_path=split_manifest,
                dataset_manifest_path=dataset_manifest,
            )

            self.assertEqual(manifest["status"], "diagnostic_only")
            self.assertTrue(any(warning["kind"] == "graph2mat_unsupported_kgrid" for warning in manifest["warnings"]))
            self.assertIsNone(manifest["recommendation"]["winner"])

    def test_diagnostic_only_comparison_does_not_declare_winner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_manifest, split_manifest = write_valid_dataset_manifests(root, ["s0"])
            write_method_metrics(root / "g2m", sample_ids=["s0"], h_mae=0.2)
            write_method_metrics(root / "deeph", sample_ids=["s0"], h_mae=0.1, diagnostic_only=True)
            write_deeph_adapter_manifest(root / "deeph", proven=False)

            manifest = aggregate_common_metrics(
                graph2mat_metrics_root=root / "g2m",
                deeph_metrics_root=root / "deeph",
                output_dir=root / "summary",
                frozen_split_manifest_path=split_manifest,
                dataset_manifest_path=dataset_manifest,
            )

            self.assertEqual(manifest["status"], "diagnostic_only")
            self.assertIsNone(manifest["recommendation"]["winner"])

    def test_valid_minimal_fixture_produces_summary_and_winner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_manifest, split_manifest = write_valid_dataset_manifests(root, ["s0", "s1"])
            write_method_metrics(root / "g2m", sample_ids=["s0", "s1"], h_mae=0.2)
            write_method_metrics(root / "deeph", sample_ids=["s0", "s1"], h_mae=0.1)
            write_deeph_adapter_manifest(root / "deeph", proven=True)

            manifest = aggregate_common_metrics(
                graph2mat_metrics_root=root / "g2m",
                deeph_metrics_root=root / "deeph",
                output_dir=root / "summary",
                frozen_split_manifest_path=split_manifest,
                dataset_manifest_path=dataset_manifest,
            )

            self.assertEqual(manifest["status"], "valid_joint_one_pass_dataset")
            self.assertEqual(manifest["recommendation"]["winner"], "deeph")
            self.assertTrue(manifest["recommendation"]["robust_recommendation"])
            self.assertTrue((root / "summary" / "common_method_metrics.csv").exists())
            self.assertTrue((root / "summary" / "common_summary.json").exists())

    def test_aggregate_common_metrics_includes_derivative_diagnostics_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_manifest, split_manifest = write_valid_dataset_manifests(root, ["s0"])
            write_method_metrics(root / "g2m", sample_ids=["s0"], h_mae=0.2)
            write_method_metrics(root / "deeph", sample_ids=["s0"], h_mae=0.1)
            write_deeph_adapter_manifest(root / "deeph", proven=True)
            write_derivative_metrics(root / "g2m_derivative")
            write_derivative_metrics(root / "deeph_derivative")

            manifest = aggregate_common_metrics(
                graph2mat_metrics_root=root / "g2m",
                deeph_metrics_root=root / "deeph",
                output_dir=root / "summary",
                frozen_split_manifest_path=split_manifest,
                dataset_manifest_path=dataset_manifest,
                graph2mat_derivative_root=root / "g2m_derivative",
                deeph_derivative_root=root / "deeph_derivative",
            )

            self.assertTrue(manifest["derivative_metrics"]["available"])
            self.assertEqual({row["method"] for row in manifest["derivative_summary_rows"]}, {"graph2mat", "deeph"})
            self.assertFalse(manifest["derivative_metrics"]["winner_metric"])
            self.assertTrue((root / "summary" / "common_derivative_method_metrics.csv").exists())
            self.assertTrue(manifest["recommendation"]["diagnostic_notes"])
            self.assertEqual(manifest["recommendation"]["winner"], "deeph")

    def test_failed_derivative_diagnostics_do_not_hide_h_metric_recommendation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_manifest, split_manifest = write_valid_dataset_manifests(root, ["s0"])
            write_method_metrics(root / "g2m", sample_ids=["s0"], h_mae=0.2)
            write_method_metrics(root / "deeph", sample_ids=["s0"], h_mae=0.1)
            write_deeph_adapter_manifest(root / "deeph", proven=True)
            write_failed_derivative_metrics(root / "g2m_derivative")

            manifest = aggregate_common_metrics(
                graph2mat_metrics_root=root / "g2m",
                deeph_metrics_root=root / "deeph",
                output_dir=root / "summary",
                frozen_split_manifest_path=split_manifest,
                dataset_manifest_path=dataset_manifest,
                graph2mat_derivative_root=root / "g2m_derivative",
            )

            self.assertEqual(manifest["recommendation"]["winner"], "deeph")
            self.assertTrue(manifest["derivative_metrics"]["available"])
            derivative_row = manifest["derivative_summary_rows"][0]
            self.assertEqual(derivative_row["derivative_stencils_failed"], 1)
            self.assertEqual(derivative_row["derivative_scientific_status"], "diagnostic_only")

    def test_plot_payload_includes_derivative_diagnostic_plot(self) -> None:
        payload = build_common_plot_payload(
            {
                "status": "valid_reused_joint_dataset",
                "summary_rows": [
                    {"method": "graph2mat", "h_mae_eV_mean": 0.2},
                    {"method": "deeph", "h_mae_eV_mean": 0.1},
                ],
                "derivative_summary_rows": [
                    {
                        "method": "graph2mat",
                        "derivative_metrics_available": True,
                        "dh_mae_union_eV_per_Ang_mean": 0.25,
                    },
                    {
                        "method": "deeph",
                        "derivative_metrics_available": True,
                        "dh_mae_union_eV_per_Ang_mean": 0.35,
                    },
                ],
                "warnings": [],
                "recommendation": {"winner": "deeph", "robust_recommendation": True},
            }
        )

        derivative_plot = next(plot for plot in payload["plots"] if plot["id"] == "derivative_mae")
        self.assertTrue(derivative_plot["diagnostic_only"])
        self.assertEqual({row["method"] for row in derivative_plot["rows"]}, {"graph2mat", "deeph"})

    def test_deeph_adapter_without_proven_equivalence_blocks_winner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_manifest, split_manifest = write_valid_dataset_manifests(root, ["s0"])
            write_method_metrics(root / "g2m", sample_ids=["s0"], h_mae=0.2)
            write_method_metrics(root / "deeph", sample_ids=["s0"], h_mae=0.1)

            manifest = aggregate_common_metrics(
                graph2mat_metrics_root=root / "g2m",
                deeph_metrics_root=root / "deeph",
                output_dir=root / "summary",
                frozen_split_manifest_path=split_manifest,
                dataset_manifest_path=dataset_manifest,
            )

            self.assertEqual(manifest["status"], "diagnostic_only")
            self.assertIsNone(manifest["recommendation"]["winner"])
            self.assertTrue(any(warning["kind"] == "deeph_adapter_equivalence_not_proven" for warning in manifest["warnings"]))
            deeph_row = next(row for row in manifest["summary_rows"] if row["method"] == "deeph")
            self.assertFalse(deeph_row["raw_global_equivalence_proven"])

    def test_plot_payload_handles_missing_metrics(self) -> None:
        payload = build_common_plot_payload(
            {
                "status": "valid_reused_joint_dataset",
                "summary_rows": [
                    {"method": "graph2mat", "h_mae_eV_mean": 0.2},
                    {"method": "deeph"},
                ],
                "warnings": [],
                "recommendation": {
                    "winner": None,
                    "robust_recommendation": False,
                    "reason": "missing primary metric",
                },
            }
        )

        self.assertTrue(payload["available"])
        matrix_plot = next(plot for plot in payload["plots"] if plot["id"] == "h_mae")
        self.assertTrue(matrix_plot["missing_metrics"])

    def test_plot_payload_diagnostic_only_disables_winner(self) -> None:
        payload = build_common_plot_payload(
            {
                "status": "diagnostic_only",
                "summary_rows": [
                    {"method": "graph2mat", "h_mae_eV_mean": 0.2},
                    {"method": "deeph", "h_mae_eV_mean": 0.1},
                ],
                "warnings": [{"severity": "severe", "kind": "deeph_diagnostic_only"}],
                "recommendation": {
                    "winner": "deeph",
                    "robust_recommendation": True,
                    "primary_metric": "h_mae_eV_mean",
                },
            }
        )

        self.assertTrue(payload["diagnostic_only"])
        self.assertIsNone(payload["recommendation"]["winner"])
        self.assertFalse(payload["recommendation"]["robust_recommendation"])

    def test_plot_payload_invalid_status_disables_winner(self) -> None:
        payload = build_common_plot_payload(
            {
                "status": "invalid_incompatible_splits",
                "summary_rows": [
                    {"method": "graph2mat", "h_mae_eV_mean": 0.2},
                    {"method": "deeph", "h_mae_eV_mean": 0.1},
                ],
                "warnings": [{"severity": "severe", "kind": "mismatched_sample_ids"}],
                "recommendation": {
                    "winner": "deeph",
                    "robust_recommendation": True,
                    "primary_metric": "h_mae_eV_mean",
                },
            }
        )

        self.assertTrue(payload["invalid"])
        self.assertIsNone(payload["recommendation"]["winner"])
        self.assertFalse(payload["recommendation"]["robust_recommendation"])

    def test_plot_payload_includes_timing_rows(self) -> None:
        timing_rows = [
            {
                "phase": "graph2mat_train",
                "label": "Graph2Mat train",
                "elapsed_seconds": 12.5,
                "source": "test",
                "status": "available",
            }
        ]
        payload = build_common_plot_payload(
            {
                "status": "valid_reused_joint_dataset",
                "summary_rows": [],
                "warnings": [],
                "recommendation": {"winner": None, "robust_recommendation": False},
            },
            timing_rows=timing_rows,
        )

        self.assertEqual(payload["timing_rows"], timing_rows)

    def test_plot_payload_includes_timing_scaling_plot_without_metrics(self) -> None:
        timing_scaling_rows = [
            {
                "dataset_id": "md_6",
                "dataset_size": 6,
                "phase": "graph2mat_train",
                "label": "Graph2Mat train",
                "model": "graph2mat",
                "elapsed_seconds": 12.0,
                "seconds_per_snapshot": 2.0,
            }
        ]
        payload = build_common_plot_payload(None, timing_scaling_rows=timing_scaling_rows)

        self.assertTrue(payload["available"])
        self.assertEqual(payload["timing_scaling_rows"], timing_scaling_rows)
        scaling_plot = next(plot for plot in payload["plots"] if plot["id"] == "timing_scaling")
        self.assertEqual(scaling_plot["kind"], "timing_scaling")
        self.assertEqual(scaling_plot["rows"], timing_scaling_rows)

    def test_plot_payload_includes_metric_scaling_plots_without_current_metrics(self) -> None:
        metric_scaling_rows = [
            {
                "run_id": "run_a",
                "dataset_size": 20,
                "method": "graph2mat",
                "metric_key": "h_mae_eV_mean",
                "metric_value": 0.2,
            },
            {
                "run_id": "run_b",
                "dataset_size": 40,
                "method": "graph2mat",
                "metric_key": "low_energy_rmse_eV_mean",
                "metric_value": 1.2,
            },
        ]
        payload = build_common_plot_payload(None, metric_scaling_rows=metric_scaling_rows)

        self.assertTrue(payload["available"])
        self.assertEqual(payload["metric_scaling_rows"], metric_scaling_rows)
        plot_ids = {plot["id"] for plot in payload["plots"]}
        self.assertIn("metric_scaling_h_mae", plot_ids)
        self.assertIn("metric_scaling_spectral_low_energy", plot_ids)
        self.assertTrue(
            all(len(plot.get("metrics") or []) == 1 for plot in payload["plots"] if plot["kind"] == "metric_scaling")
        )

    def test_plot_payload_splits_metric_groups_by_scale(self) -> None:
        payload = build_common_plot_payload(
            {
                "status": "valid_reused_joint_dataset",
                "summary_rows": [
                    {
                        "method": "graph2mat",
                        "h_mae_eV_mean": 0.1,
                        "h_rmse_eV_mean": 0.2,
                        "h_mse_eV2_mean": 0.04,
                        "global_rmse_eV_mean": 0.3,
                        "low_energy_rmse_eV_mean": 0.4,
                        "fermi_window_rmse_eV_mean": 0.5,
                        "frontier_window_rmse_eV_mean": 0.6,
                        "dos_mae_500_fermi_window_mean": 0.7,
                        "dos_wasserstein_eV_mean": 0.8,
                    }
                ],
                "warnings": [],
                "recommendation": {"winner": None, "robust_recommendation": False},
            }
        )

        plot_ids = {plot["id"] for plot in payload["plots"]}
        for plot_id in (
            "h_mae",
            "h_rmse",
            "h_mse",
            "spectral_global",
            "spectral_low_energy",
            "spectral_fermi",
            "spectral_frontier",
            "dos_mae",
            "dos_wasserstein",
        ):
            self.assertIn(plot_id, plot_ids)
        self.assertTrue(all(len(plot.get("metrics") or []) == 1 for plot in payload["plots"] if plot["kind"] == "grouped_bar"))

    def test_plot_payload_includes_artifact_missing_counts(self) -> None:
        payload = build_common_plot_payload(
            {
                "status": "invalid_missing_artifacts",
                "summary_rows": [],
                "warnings": [],
                "recommendation": {"winner": None, "robust_recommendation": False},
            },
            artifact_summary={"missing_required_counts": {"hsx": 3}},
        )

        self.assertEqual(payload["artifact_summary"]["missing_required_counts"]["hsx"], 3)


if __name__ == "__main__":
    unittest.main()
