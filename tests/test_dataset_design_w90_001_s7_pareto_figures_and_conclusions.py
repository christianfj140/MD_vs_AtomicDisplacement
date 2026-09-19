from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "Comparison" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import dataset_design_w90_001_s7_pareto_figures_and_conclusions as s7  # noqa: E402


def test_build_pareto_table_marks_only_non_dominated_points_as_frontier():
    df_ok = pd.DataFrame(
        [
            {"family": "a", "dim": "3D", "k": 1, "N_train": 16, "seed": 0, "H_MAE": 0.5},
            {"family": "b", "dim": "3D", "k": 1, "N_train": 16, "seed": 0, "H_MAE": 0.4},  # frontier @16
            {"family": "a", "dim": "3D", "k": 1, "N_train": 32, "seed": 0, "H_MAE": 0.45},  # dominated by cost=16/0.4
            {"family": "b", "dim": "3D", "k": 1, "N_train": 64, "seed": 0, "H_MAE": 0.3},  # frontier @64
        ]
    )
    table = s7.build_pareto_table(df_ok)
    v = s7.s4.VAL_COUNT
    frontier = table[~table.pareto_dominated]
    assert set(zip(frontier["siesta_cost"], frontier["family"])) == {(16 + v, "b"), (64 + v, "b")}
    dominated = table[table.pareto_dominated]
    assert set(zip(dominated["siesta_cost"], dominated["family"])) == {(16 + v, "a"), (32 + v, "a")}


def test_build_pruning_log_flags_zero_ok_family_as_data_unavailable():
    df_all = pd.DataFrame(
        [
            {"family": "good", "dim": "3D", "k": 1, "N_train": 16, "seed": 0, "status": "ok", "H_MAE": 0.4},
            {"family": "bad", "dim": "3D", "k": 1, "N_train": 16, "seed": 0, "status": "insufficient_samples", "H_MAE": None},
        ]
    )
    df_ok = df_all[df_all.status == "ok"]
    pareto_table = s7.build_pareto_table(df_ok)
    log = s7.build_pruning_log(df_all, pareto_table)
    scopes = {row["scope"]: row for row in log}
    assert "family=bad" in scopes
    assert scopes["family=bad"]["decision"] == "excluded_pre_pareto"
    assert "family=good" not in scopes  # has ok data, not a data-availability exclusion


def test_build_pruning_log_flags_saturation_when_both_increments_are_flat():
    df_all = pd.DataFrame(
        [
            {"family": "f", "dim": "3D", "k": 1, "N_train": 16, "seed": s, "status": "ok", "H_MAE": 0.40}
            for s in (0, 1)
        ]
        + [
            {"family": "f", "dim": "3D", "k": 1, "N_train": 32, "seed": s, "status": "ok", "H_MAE": 0.395}
            for s in (0, 1)
        ]
        + [
            {"family": "f", "dim": "3D", "k": 1, "N_train": 64, "seed": s, "status": "ok", "H_MAE": 0.39}
            for s in (0, 1)
        ]
    )
    df_ok = df_all[df_all.status == "ok"]
    pareto_table = s7.build_pareto_table(df_ok)
    log = s7.build_pruning_log(df_all, pareto_table)
    saturated = [row for row in log if row["rule"] == "no_improvement_two_N_increments"]
    assert any(row["scope"] == "family=f, dim=3D, k=1" for row in saturated)


if __name__ == "__main__":
    test_build_pareto_table_marks_only_non_dominated_points_as_frontier()
    test_build_pruning_log_flags_zero_ok_family_as_data_unavailable()
    test_build_pruning_log_flags_saturation_when_both_increments_are_flat()
    print("ok")
