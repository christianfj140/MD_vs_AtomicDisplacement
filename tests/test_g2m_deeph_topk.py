import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "Comparison" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from g2m_deeph_topk import (  # noqa: E402
    ROBUST_RERUN_PLAN_SCHEMA,
    SELECTED_CONFIGS_SCHEMA,
    generate_robust_rerun_plan,
    select_top_configs,
    validation_metric_value,
    write_selection_artifacts,
)


def _record(
    model: str,
    config_id: str,
    value: float,
    *,
    dataset_id: str = "dataset_a",
    status: str = "completed",
    split: str = "validation",
) -> dict:
    return {
        "status": status,
        "model": model,
        "dataset_id": dataset_id,
        "dataset_root": f"/tmp/{dataset_id}",
        "config_id": config_id,
        "config_hash": f"hash-{config_id}",
        "run_root": f"/tmp/runs/{config_id}",
        "metric_split": split,
        "val_loss": value,
        "common": {"epochs": 10, "learning_rate": 0.001, "batch_size": 2, "seed": 42},
        "overrides": {"hidden_irreps_channels": 16} if model == "graph2mat" else {"atom_fea_len": 64},
    }


class Graph2MatDeepHTopKTests(unittest.TestCase):
    def test_top_k_by_validation_metric_min(self) -> None:
        selected = select_top_configs(
            [
                _record("graph2mat", "g2m_a", 0.2),
                _record("graph2mat", "g2m_b", 0.1),
                _record("deeph", "dh_a", 0.3),
                _record("deeph", "dh_b", 0.25),
            ],
            metric="val_loss",
            mode="min",
            k_per_model=1,
        )

        self.assertEqual(selected["schema"], SELECTED_CONFIGS_SCHEMA)
        self.assertEqual(
            {(row["model"], row["config_id"]) for row in selected["selected_configs"]},
            {("graph2mat", "g2m_b"), ("deeph", "dh_b")},
        )

    def test_top_k_by_validation_metric_max(self) -> None:
        selected = select_top_configs(
            [_record("graph2mat", "g2m_a", 0.2), _record("graph2mat", "g2m_b", 0.7)],
            metric="val_loss",
            mode="max",
            k_per_model=1,
        )

        self.assertEqual(selected["selected_configs"][0]["config_id"], "g2m_b")

    def test_ties_are_deterministic(self) -> None:
        selected = select_top_configs(
            [_record("graph2mat", "g2m_b", 0.2), _record("graph2mat", "g2m_a", 0.2)],
            metric="val_loss",
            mode="min",
            k_per_model=1,
        )

        self.assertEqual(selected["selected_configs"][0]["config_id"], "g2m_a")

    def test_failed_and_incomplete_configs_are_excluded(self) -> None:
        selected = select_top_configs(
            [
                _record("graph2mat", "g2m_failed", 0.01, status="failed"),
                _record("graph2mat", "g2m_pending", 0.02, status="planned"),
                _record("graph2mat", "g2m_done", 0.4),
            ],
            metric="val_loss",
            mode="min",
            k_per_model=1,
        )

        self.assertEqual(selected["selected_configs"][0]["config_id"], "g2m_done")

    def test_test_metric_is_ignored_when_validation_exists(self) -> None:
        record = _record("graph2mat", "g2m_a", 0.3)
        record["test_metrics"] = {"val_loss": 0.001}
        record["metrics"] = [
            {"metric_split": "test", "metric": "val_loss", "value": 0.001},
            {"metric_split": "validation", "metric": "val_loss", "value": 0.3},
        ]

        self.assertEqual(validation_metric_value(record, "val_loss"), 0.3)

    def test_final_seed_expansion_preserves_hyperparameters_except_seed(self) -> None:
        selected = select_top_configs(
            [_record("graph2mat", "g2m_a", 0.2), _record("deeph", "dh_a", 0.3)],
            metric="val_loss",
            mode="min",
            k_per_model=1,
        )

        plan = generate_robust_rerun_plan(selected, final_seeds=[0, 1, 2])

        self.assertEqual(plan["schema"], ROBUST_RERUN_PLAN_SCHEMA)
        self.assertEqual(plan["planned_run_count"], 6)
        graph2mat_row = next(row for row in plan["planned_runs"] if row["model"] == "graph2mat")
        deeph_row = next(row for row in plan["planned_runs"] if row["model"] == "deeph")
        self.assertEqual(graph2mat_row["overrides"]["hidden_irreps_channels"], 16)
        self.assertEqual(graph2mat_row["overrides"]["seed_everything"], graph2mat_row["seed"])
        self.assertEqual(deeph_row["overrides"]["atom_fea_len"], 64)
        self.assertEqual(deeph_row["overrides"]["seed"], deeph_row["seed"])

    def test_insufficient_completed_configs_fail_clearly(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "Insufficient completed configs"):
            select_top_configs(
                [_record("graph2mat", "g2m_a", 0.2)],
                metric="val_loss",
                mode="min",
                k_per_model=2,
            )

    def test_selection_artifacts_are_serialized(self) -> None:
        selected = select_top_configs(
            [_record("graph2mat", "g2m_a", 0.2)],
            metric="val_loss",
            mode="min",
            k_per_model=1,
        )
        plan = generate_robust_rerun_plan(selected, final_seeds=[0, 1, 2])

        with tempfile.TemporaryDirectory() as tmp:
            paths = write_selection_artifacts(Path(tmp), selected, plan)

            self.assertTrue(Path(paths["selected_configs_json"]).exists())
            self.assertTrue(Path(paths["robust_rerun_plan_json"]).exists())
            self.assertTrue(Path(paths["selected_configs_csv"]).exists())
            loaded = json.loads(Path(paths["robust_rerun_plan_json"]).read_text(encoding="utf-8"))
            self.assertEqual(loaded["planned_run_count"], 3)


if __name__ == "__main__":
    unittest.main()
