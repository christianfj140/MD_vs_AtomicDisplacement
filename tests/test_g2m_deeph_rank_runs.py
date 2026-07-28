import json
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "Comparison" / "scripts"
for directory in (SCRIPTS_DIR, REPO_ROOT / "shared"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from g2m_deeph_rank_runs import (  # noqa: E402
    build_recommendation,
    choose_primary_metric,
    load_metric_rows,
    pairwise_comparisons,
    pareto_frontier,
    rank_graph2mat_deeph_runs,
    rank_metric_groups,
    row_from_training_record,
    seed_robustness_analysis,
)
from deeph_prediction_adapter import (  # noqa: E402
    EQUIVALENCE_PROVEN_RAW_GLOBAL,
    EQUIVALENCE_STATUS_UNPROVEN,
)
from reference_provenance import build_positive_reference_provenance  # noqa: E402


def valid_metric_row(model: str, *, value: float = 0.1, seed: int = 1, **overrides) -> dict:
    row = {
        "model": model,
        "dataset_id": "d",
        "config_id": f"{model}_cfg",
        "seed": seed,
        "run_status": "completed",
        "method_status": "ok",
        "comparability_status": "valid",
        "artifact_contract_status": "valid",
        "required_provenance_present": True,
        "provenance_status": "valid",
        "warning_status": "ok",
        "metric_fail_policy": "fail_closed",
        "low_energy_rmse_eV_mean": value,
    }
    if model == "deeph":
        row.update(
            {
                "adapter_equivalence_status": EQUIVALENCE_PROVEN_RAW_GLOBAL,
                "raw_global_equivalence_proven": True,
                "split_audit_status": "valid",
            }
        )
    row.update(overrides)
    return row


def best_row(model: str, *, mean: float, seeds: int = 5, **overrides) -> dict:
    row = {
        "scope": "global",
        "dataset_id": "all",
        "model": model,
        "config_id": f"{model}_cfg",
        "metric": "low_energy_rmse_eV",
        "mean": mean,
        "valid_seed_count": seeds,
        "seed_stability_status": "robust_candidate" if seeds >= 5 else "exploratory_only",
    }
    row.update(overrides)
    return row


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_dataset(
    root: Path,
    split_hash: str = "split-a",
    compatibility: str = "compat-a",
    material_profile: str = "production",
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    sample = root / "sample"
    sample.mkdir()
    (sample / "RUN.fdf").write_text(
        "SystemLabel graphene\nNumberOfAtoms 1\nNumberOfSpecies 1\n"
        "%block ChemicalSpeciesLabel\n1 6 C\n%endblock ChemicalSpeciesLabel\n"
        "%block LatticeVectors\n8 0 0\n0 8 0\n0 0 8\n%endblock LatticeVectors\n"
        "%block AtomicCoordinatesAndAtomicSpecies\n0 0 0 1\n"
        "%endblock AtomicCoordinatesAndAtomicSpecies\n",
        encoding="utf-8",
    )
    write_json(sample / "metadata.json", {"system_label": "graphene"})
    for suffix in (".TSHS", ".TSDE", ".HSX", ".STRUCT_OUT", ".XV", ".ORB_INDX"):
        (sample / f"graphene{suffix}").write_text(f"{suffix}\n", encoding="utf-8")
    (sample / "RUN.out").write_text(
        "iscf     Eharris\nSCF cycle converged\nJob completed\n",
        encoding="utf-8",
    )
    write_json(
        sample / "siesta_reference_provenance.json",
        build_positive_reference_provenance(
            sample,
            sample / "graphene.TSHS",
            frozen_sample_id="sample",
            split="test",
            frozen_split_hash=split_hash,
            basis_hashes={"C.ion.xml": "basis-hash"},
            pseudopotential_hashes={"C": "pseudo-hash"},
            siesta_version="SIESTA 5.4.2-test",
            siesta_command="siesta < RUN.fdf",
        ),
    )
    write_json(
        root / "artifact_validation.json",
        {"valid": True, "snapshots": [{"snapshot_dir": str(sample), "valid": True}]},
    )
    write_json(root / "frozen_split_manifest.json", {"valid": True, "split_hash": split_hash, "rows": []})
    write_json(
        root / "benchmark_dataset_manifest.json",
        {
            "benchmark_ready": True,
            "benchmark_dataset_id": compatibility,
            "material_label": "graphene",
            "material_profile": material_profile,
            "frozen_split_manifest": {"split_hash": split_hash},
        },
    )


def write_metric_root(root: Path, *, h_mae: float, low_energy: float, r2: float = 0.5) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "kpoint_matrix_metrics.csv").write_text(
        "row_type,sample,h_mae_eV,h_rmse_eV,h_mse_eV2,relative_frobenius,hermiticity_pred\n"
        f"weighted_sample,s1,{h_mae},{h_mae * 2},{h_mae * h_mae},0.2,0.0\n",
        encoding="utf-8",
    )
    (root / "sparse_metrics.csv").write_text(
        "sample,mae_union_eV,rmse_union_eV,mse_union_eV2,r2_union,support_f1\n"
        f"s1,{h_mae},{h_mae * 2},{h_mae * h_mae},{r2},0.7\n",
        encoding="utf-8",
    )
    (root / "kpoint_spectral_metrics.csv").write_text(
        "sample,global_rmse_eV,low_energy_rmse_eV,fermi_window_rmse_eV,frontier_window_rmse_eV\n"
        f"s1,{low_energy * 2},{low_energy},{low_energy * 1.1},{low_energy * 1.2}\n",
        encoding="utf-8",
    )
    (root / "kpoint_dos_metrics.csv").write_text(
        "sample,dos_mae_500_fermi_window\n"
        f"s1,{low_energy * 0.1}\n",
        encoding="utf-8",
    )
    write_json(root / "manifest.json", {"uses_reference_overlap_k": True, "kpoint_metrics_enabled": True, "warnings": []})


def write_deeph_adapter_manifest(metrics_root: Path, *, proven: bool = True) -> None:
    status = EQUIVALENCE_PROVEN_RAW_GLOBAL if proven else "diagnostic_local_frame_only"
    write_json(
        metrics_root.parent / "adapter_manifest.json",
        {
            "adapter_equivalence_statuses": [status],
            "diagnostic_only_count": 0 if proven else 1,
            "raw_global_equivalence_proven_count": 1 if proven else 0,
            "robust_matrix_metrics_allowed": proven,
            "samples": [
                {
                    "sample_id": "s1",
                    "adapter_equivalence_status": status,
                    "diagnostic_only": not proven,
                }
            ],
        },
    )


def write_run(
    base: Path,
    dataset: Path,
    *,
    model: str,
    config_id: str,
    seed: int,
    h_mae: float,
    low_energy: float,
    seconds: float = 10.0,
    metric_fail_policy: str = "fail_closed",
) -> dict:
    run_root = base / "sweep" / model / dataset.name / f"{config_id}_{seed}"
    metrics_root = (
        run_root / "metrics" / "graph2mat" / "eval_input" / "metrics"
        if model == "graph2mat"
        else run_root / "metrics" / "deeph" / "eval" / "metrics"
    )
    write_metric_root(metrics_root, h_mae=h_mae, low_energy=low_energy)
    deeph_manifest_path = ""
    if model == "deeph":
        write_deeph_adapter_manifest(metrics_root, proven=True)
        deeph_manifest_path = str(run_root / "deeph" / "deeph_manifest.json")
        write_json(Path(deeph_manifest_path), {"split_audit_status": "valid", "split_audit": {"status": "valid"}})
    return {
        "model": model,
        "dataset_id": dataset.name,
        "dataset_root": str(dataset),
        "config_id": config_id,
        "config_hash": config_id,
        "common": {"seed": seed},
        "status": "completed",
        "run_root": str(run_root),
        "train_run": {"elapsed_seconds": seconds},
        "predict_run": {"elapsed_seconds": seconds / 10.0},
        "metrics_run": {"elapsed_seconds": seconds / 20.0},
        "metric_fail_policy": metric_fail_policy,
        "deeph_manifest_path": deeph_manifest_path,
    }


class Graph2MatDeepHRankingTests(unittest.TestCase):
    def test_loader_reads_valid_training_sweep_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            write_dataset(dataset)
            record = write_run(root, dataset, model="graph2mat", config_id="g2m_a", seed=1, h_mae=0.2, low_energy=0.3)
            manifest = root / "sweep" / "training_sweep_manifest.json"
            write_json(manifest, {"runs": [record]})

            rows = load_metric_rows(training_sweep_manifest_path=manifest)

            self.assertEqual(rows[0]["model"], "graph2mat")
            self.assertEqual(rows[0]["config_id"], "g2m_a")
            self.assertAlmostEqual(float(rows[0]["low_energy_rmse_eV_mean"]), 0.3)

    def test_loader_rejects_stale_manifest_when_live_siesta_output_is_truncated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            write_dataset(dataset)
            (dataset / "sample" / "RUN.out").write_text("startup only\n", encoding="utf-8")
            record = write_run(
                root,
                dataset,
                model="graph2mat",
                config_id="g2m_a",
                seed=1,
                h_mae=0.2,
                low_energy=0.3,
            )

            row = row_from_training_record(record)

            self.assertEqual(row["artifact_contract_status"], "invalid")

    def test_missing_config_id_fails_clearly(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "config_id"):
            row_from_training_record({"model": "graph2mat", "status": "completed"})

    def test_primary_metric_ignores_non_finite_values(self) -> None:
        rows = [
            {"model": "graph2mat", "low_energy_rmse_eV_mean": "nan", "h_mae_eV_mean": 0.2},
            {"model": "deeph", "low_energy_rmse_eV_mean": 0.1, "h_mae_eV_mean": 0.3},
        ]

        self.assertEqual(choose_primary_metric(rows), "h_mae_eV")

    def test_ranking_lower_and_higher_directions(self) -> None:
        rows = [
            {"model": "graph2mat", "dataset_id": "d", "config_id": "a", "seed": 1, "run_status": "completed", "method_status": "ok", "comparability_status": "valid", "low_energy_rmse_eV_mean": 0.2, "r2_mean": 0.4},
            {"model": "graph2mat", "dataset_id": "d", "config_id": "b", "seed": 1, "run_status": "completed", "method_status": "ok", "comparability_status": "valid", "low_energy_rmse_eV_mean": 0.1, "r2_mean": 0.9},
        ]

        lower = rank_metric_groups(rows, "low_energy_rmse_eV")
        higher = rank_metric_groups(rows, "r2")

        self.assertEqual([row for row in lower if row["rank"] == 1][0]["config_id"], "b")
        self.assertEqual([row for row in higher if row["rank"] == 1][0]["config_id"], "b")

    def test_pairwise_blocks_incompatible_split_hash(self) -> None:
        best_rows = [
            {"scope": "dataset", "dataset_id": "d", "model": "graph2mat", "config_id": "g", "metric": "low_energy_rmse_eV", "mean": 0.2, "frozen_split_hash": "a", "dataset_compatibility_hash": "c", "robust_eligible": True},
            {
                "scope": "dataset",
                "dataset_id": "d",
                "model": "deeph",
                "config_id": "d",
                "metric": "low_energy_rmse_eV",
                "mean": 0.1,
                "frozen_split_hash": "b",
                "dataset_compatibility_hash": "c",
                "robust_eligible": True,
                "adapter_equivalence_status": EQUIVALENCE_PROVEN_RAW_GLOBAL,
            },
        ]

        pairs = pairwise_comparisons(best_rows)

        self.assertEqual(pairs[0]["status"], "invalid_incompatible_splits")
        self.assertIsNone(pairs[0]["winner"])

    def test_single_seed_recommendation_is_exploratory_not_robust(self) -> None:
        best_rows = [
            best_row("graph2mat", mean=0.2, seeds=1),
            best_row("deeph", mean=0.1, seeds=1),
        ]
        pairs = [{"metric": "low_energy_rmse_eV", "status": "comparable", "winner": "deeph"}]
        rows = [valid_metric_row("graph2mat", value=0.2), valid_metric_row("deeph", value=0.1)]

        rec = build_recommendation(rows=rows, best_rows=best_rows, pairs=pairs, primary_metric="low_energy_rmse_eV")

        self.assertEqual(rec["status"], "exploratory_deeph_win")
        self.assertEqual(rec["scientific_status"], "exploratory_only")

    def test_severe_warning_blocks_robust_winner(self) -> None:
        best_rows = [
            best_row("graph2mat", mean=0.2),
            best_row("deeph", mean=0.1),
        ]
        rows = [
            valid_metric_row("graph2mat", value=0.2, severe_warnings=["severe overlap"], warning_status="severe"),
            valid_metric_row("deeph", value=0.1),
        ]

        rec = build_recommendation(rows=rows, best_rows=best_rows, pairs=[{"metric": "low_energy_rmse_eV", "status": "comparable"}], primary_metric="low_energy_rmse_eV")

        self.assertIsNone(rec["winner"])
        self.assertIn("severe_warnings", rec["gates_failed"])

    def test_metric_fail_policy_diagnostic_blocks_robust_winner(self) -> None:
        best_rows = [
            best_row("graph2mat", mean=0.2),
            best_row("deeph", mean=0.1),
        ]
        rows = [
            valid_metric_row("graph2mat", value=0.2),
            valid_metric_row("deeph", value=0.1, metric_fail_policy="diagnostic_only", fail_open_metric_outputs=True),
        ]

        rec = build_recommendation(rows=rows, best_rows=best_rows, pairs=[{"metric": "low_energy_rmse_eV", "status": "comparable"}], primary_metric="low_energy_rmse_eV")

        self.assertIsNone(rec["winner"])
        self.assertEqual(rec["status"], "diagnostic_only")
        self.assertIn("metric_fail_policy_diagnostic_only", rec["gates_failed"])

    def test_missing_provenance_gets_explicit_status(self) -> None:
        best_rows = [best_row("graph2mat", mean=0.1), best_row("deeph", mean=0.2)]
        rows = [
            valid_metric_row("graph2mat", value=0.1, required_provenance_present=False, provenance_status="invalid"),
            valid_metric_row("deeph", value=0.2),
        ]

        rec = build_recommendation(rows=rows, best_rows=best_rows, pairs=[{"metric": "low_energy_rmse_eV", "status": "comparable"}], primary_metric="low_energy_rmse_eV")

        self.assertEqual(rec["status"], "invalid_missing_provenance")
        self.assertIn("invalid_missing_provenance", rec["gates_failed"])

    def test_deeph_split_audit_missing_gets_explicit_status(self) -> None:
        best_rows = [best_row("graph2mat", mean=0.2), best_row("deeph", mean=0.1)]
        rows = [
            valid_metric_row("graph2mat", value=0.2),
            valid_metric_row("deeph", value=0.1, split_audit_status="missing"),
        ]

        rec = build_recommendation(rows=rows, best_rows=best_rows, pairs=[{"metric": "low_energy_rmse_eV", "status": "comparable"}], primary_metric="low_energy_rmse_eV")

        self.assertEqual(rec["status"], "invalid_unverified_deeph_split")
        self.assertIn("invalid_unverified_deeph_split", rec["gates_failed"])

    def test_incomplete_grid_gets_explicit_status(self) -> None:
        rows = [valid_metric_row("graph2mat", value=0.1)]

        rec = build_recommendation(rows=rows, best_rows=[best_row("graph2mat", mean=0.1)], pairs=[], primary_metric="low_energy_rmse_eV")

        self.assertEqual(rec["status"], "invalid_incomplete_grid")
        self.assertIn("missing_model", rec["gates_failed"])

    def test_stable_robust_graph2mat_win(self) -> None:
        best_rows = [best_row("graph2mat", mean=0.1), best_row("deeph", mean=0.2)]
        rows = [valid_metric_row("graph2mat", value=0.1), valid_metric_row("deeph", value=0.2)]

        rec = build_recommendation(rows=rows, best_rows=best_rows, pairs=[{"metric": "low_energy_rmse_eV", "status": "comparable"}], primary_metric="low_energy_rmse_eV")

        self.assertEqual(rec["status"], "robust_graph2mat_win")
        self.assertEqual(rec["winner"], "graph2mat")
        self.assertEqual(rec["adapter_equivalence_status"], EQUIVALENCE_PROVEN_RAW_GLOBAL)
        self.assertEqual(rec["split_audit_status"], "valid")

    def test_stable_robust_deeph_win(self) -> None:
        best_rows = [best_row("graph2mat", mean=0.2), best_row("deeph", mean=0.1)]
        rows = [valid_metric_row("graph2mat", value=0.2), valid_metric_row("deeph", value=0.1)]

        rec = build_recommendation(rows=rows, best_rows=best_rows, pairs=[{"metric": "low_energy_rmse_eV", "status": "comparable"}], primary_metric="low_energy_rmse_eV")

        self.assertEqual(rec["status"], "robust_deeph_win")
        self.assertEqual(rec["winner"], "deeph")

    def test_paired_seed_analysis_requires_stable_leave_one_out(self) -> None:
        rows = []
        for seed, graph_value, deeph_value in (
            (0, 0.10, 0.20),
            (1, 0.11, 0.21),
            (2, 0.09, 0.19),
            (3, 0.10, 0.22),
            (4, 0.12, 0.20),
        ):
            rows.extend(
                [
                    valid_metric_row("graph2mat", value=graph_value, seed=seed),
                    valid_metric_row("deeph", value=deeph_value, seed=seed),
                ]
            )
        analysis = seed_robustness_analysis(
            rows,
            [best_row("graph2mat", mean=0.104), best_row("deeph", mean=0.204)],
            "low_energy_rmse_eV",
        )

        self.assertEqual(analysis["status"], "robust")
        self.assertEqual(analysis["winner"], "graph2mat")
        self.assertTrue(analysis["leave_one_seed_out_stable"])

    def test_paired_seed_noise_returns_no_robust_winner(self) -> None:
        rows = []
        for seed, graph_value, deeph_value in (
            (0, 0.10, 0.11),
            (1, 0.12, 0.11),
            (2, 0.10, 0.11),
            (3, 0.12, 0.11),
            (4, 0.10, 0.11),
        ):
            rows.extend(
                [
                    valid_metric_row("graph2mat", value=graph_value, seed=seed),
                    valid_metric_row("deeph", value=deeph_value, seed=seed),
                ]
            )
        analysis = seed_robustness_analysis(
            rows,
            [best_row("graph2mat", mean=0.108), best_row("deeph", mean=0.11)],
            "low_energy_rmse_eV",
        )

        self.assertEqual(analysis["status"], "no_robust_winner")
        self.assertIsNone(analysis["winner"])

    def test_training_record_fail_open_metrics_are_diagnostic_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            write_dataset(dataset)
            record = write_run(
                root,
                dataset,
                model="deeph",
                config_id="deeph_diag",
                seed=1,
                h_mae=0.1,
                low_energy=0.1,
                metric_fail_policy="diagnostic_only",
            )

            row = row_from_training_record(record)

            self.assertEqual(row["comparability_status"], "diagnostic_only")
            self.assertTrue(row["diagnostic_only"])
            self.assertTrue(row["fail_open_metric_outputs"])

    def test_deeph_adapter_diagnostic_blocks_robust_winner(self) -> None:
        best_rows = [
            best_row("graph2mat", mean=0.2),
            best_row("deeph", mean=0.1),
        ]
        rows = [
            valid_metric_row("graph2mat", value=0.2),
            valid_metric_row("deeph", value=0.1, adapter_equivalence_status="diagnostic_local_frame_only", raw_global_equivalence_proven=False),
        ]

        rec = build_recommendation(rows=rows, best_rows=best_rows, pairs=[{"metric": "low_energy_rmse_eV", "status": "comparable"}], primary_metric="low_energy_rmse_eV")

        self.assertIsNone(rec["winner"])
        self.assertEqual(rec["status"], "diagnostic_only")
        self.assertIn("deeph_adapter_equivalence_not_proven", rec["gates_failed"])

    def test_unproven_equivalence_status_blocks_even_with_proven_adapter_label(self) -> None:
        best_rows = [
            best_row("graph2mat", mean=0.2),
            best_row("deeph", mean=0.1),
        ]
        rows = [
            valid_metric_row("graph2mat", value=0.2),
            valid_metric_row(
                "deeph",
                value=0.1,
                adapter_equivalence_status=EQUIVALENCE_PROVEN_RAW_GLOBAL,
                equivalence_status=EQUIVALENCE_STATUS_UNPROVEN,
            ),
        ]

        rec = build_recommendation(
            rows=rows,
            best_rows=best_rows,
            pairs=[{"metric": "low_energy_rmse_eV", "status": "comparable"}],
            primary_metric="low_energy_rmse_eV",
        )

        self.assertIsNone(rec["winner"])
        self.assertEqual(rec["status"], "diagnostic_only")
        self.assertIn("deeph_adapter_equivalence_not_proven", rec["gates_failed"])

    def test_deeph_missing_adapter_status_is_not_robust_eligible(self) -> None:
        rows = [
            {"model": "deeph", "dataset_id": "d", "config_id": "local", "seed": 1, "run_status": "completed", "method_status": "ok", "comparability_status": "valid", "low_energy_rmse_eV_mean": 0.1, "adapter_equivalence_status": "diagnostic_local_frame_only"},
        ]

        ranked = rank_metric_groups(rows, "low_energy_rmse_eV")

        self.assertFalse(ranked[0]["robust_eligible"])
        self.assertEqual(ranked[0]["adapter_equivalence_status"], "diagnostic_local_frame_only")

    def test_pareto_excludes_dominated_runs(self) -> None:
        rows = [
            valid_metric_row("graph2mat", value=0.4, config_id="slow_bad", total_time_seconds=20),
            valid_metric_row("graph2mat", value=0.2, config_id="fast_good", total_time_seconds=10),
        ]

        frontier = pareto_frontier(rows, "low_energy_rmse_eV")

        self.assertEqual([row["config_id"] for row in frontier], ["fast_good"])

    def test_ranker_writes_outputs_and_exploratory_winner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            write_dataset(dataset)
            runs = [
                write_run(root, dataset, model="graph2mat", config_id="g2m_a", seed=1, h_mae=0.2, low_energy=0.2),
                write_run(root, dataset, model="deeph", config_id="deeph_a", seed=1, h_mae=0.1, low_energy=0.1),
            ]
            write_json(root / "sweep" / "training_sweep_manifest.json", {"runs": runs})

            manifest = rank_graph2mat_deeph_runs(run_root=root)

            self.assertEqual(manifest["recommendation"]["status"], "exploratory_deeph_win")
            self.assertTrue((root / "summary" / "ranking" / "recommendation.json").exists())
            self.assertTrue((root / "summary" / "ranking" / "pareto_accuracy_cost.csv").exists())

    def test_artificially_best_smoke_material_is_not_ranked_as_winner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            write_dataset(dataset, material_profile="smoke")
            runs = [
                write_run(
                    root,
                    dataset,
                    model="graph2mat",
                    config_id="smoke_other",
                    seed=1,
                    h_mae=1.0,
                    low_energy=1.0,
                ),
                write_run(
                    root,
                    dataset,
                    model="deeph",
                    config_id="smoke_best",
                    seed=1,
                    h_mae=1e-12,
                    low_energy=1e-12,
                )
            ]
            write_json(root / "sweep" / "training_sweep_manifest.json", {"runs": runs})

            manifest = rank_graph2mat_deeph_runs(run_root=root)

            self.assertEqual(
                manifest["recommendation"]["status"],
                "invalid_incompatible_artifacts",
            )


if __name__ == "__main__":
    unittest.main()
