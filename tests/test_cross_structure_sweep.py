"""Cross-structure sweep: planner product, MAE aggregator, live-record runner."""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "Comparison" / "scripts"
SHARED_DIR = REPO_ROOT / "shared"
for path in (SCRIPTS_DIR, SHARED_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import ml_vs_siesta as mvs  # noqa: E402
from ml_vs_siesta import cross_structure_sweep as sweep  # noqa: E402
from test_cross_structure_evaluation import _make_dataset  # noqa: E402


@pytest.fixture
def source_pair(tmp_path: Path):
    """Two compatible sources (10 and 20 train+val snapshots-ish) + one target."""
    small = _make_dataset(
        tmp_path / "src_small", n_atoms=2, label="w90_small",
        counts={"train": 2, "validation": 1, "test": 1},
    )
    big = _make_dataset(
        tmp_path / "src_big", n_atoms=2, label="w90_big",
        counts={"train": 4, "validation": 2, "test": 1},
    )
    target = _make_dataset(
        tmp_path / "target", n_atoms=50, label="cell5x5",
        kgrid=2, lattice_scale=5.0,
    )
    return small, big, target


def test_plan_is_full_product(source_pair) -> None:
    small, big, target = source_pair
    plan = mvs.plan_cross_structure_sweep([small, big], [target])
    assert plan["n_permutations"] == 2
    assert plan["n_compatible"] == 2
    by_snapshots = {p["source_n_snapshots"]: p for p in plan["permutations"]}
    # train+validation counts are the plot x axis.
    assert set(by_snapshots) == {3, 6}
    # Sibling datasets sharing a provenance label must get distinct ids.
    assert by_snapshots[3]["source_id"] != by_snapshots[6]["source_id"]
    assert by_snapshots[3]["target_id"] == by_snapshots[6]["target_id"]
    assert by_snapshots[3]["payload_id"].endswith("__to__" + by_snapshots[3]["target_id"])
    assert by_snapshots[3]["payload_id"] != by_snapshots[6]["payload_id"]


def test_incompatible_pair_does_not_abort(tmp_path: Path) -> None:
    good = _make_dataset(tmp_path / "good", n_atoms=2, label="good")
    # Different basis hash -> validate_datasets_compatible fails for this target.
    bad_target = _make_dataset(
        tmp_path / "bad", n_atoms=50, label="bad", basis="other-basis",
        kgrid=2, lattice_scale=5.0,
    )
    ok_target = _make_dataset(
        tmp_path / "ok", n_atoms=50, label="ok", kgrid=2, lattice_scale=5.0,
    )
    plan = mvs.plan_cross_structure_sweep([good], [bad_target, ok_target])
    assert plan["n_permutations"] == 2
    status = {p["target_id"]: p["status"] for p in plan["permutations"]}
    assert status["bad"] == "incompatible"
    assert status["ok"] == "compatible"
    incompatible = next(p for p in plan["permutations"] if p["target_id"] == "bad")
    assert incompatible["reason"]
    assert plan["warnings"]


def test_aggregate_groups_by_target_model() -> None:
    records = [
        {"source_id": "s10", "target_id": "t50", "source_n_snapshots": 10,
         "model": "graph2mat", "seed": 0, "h_mae_eV": 0.4, "relative_frobenius": 0.1},
        {"source_id": "s10", "target_id": "t50", "source_n_snapshots": 10,
         "model": "graph2mat", "seed": 1, "h_mae_eV": 0.6, "relative_frobenius": 0.3},
        {"source_id": "s50", "target_id": "t50", "source_n_snapshots": 50,
         "model": "graph2mat", "seed": 0, "h_mae_eV": 0.2},
        {"source_id": "s10", "target_id": "t50", "source_n_snapshots": 10,
         "model": "deeph", "seed": 0, "h_mae_eV": 0.9},
    ]
    agg = mvs.aggregate_cross_structure_mae(records)
    assert agg["n_curves"] == 2  # (t50, graph2mat) and (t50, deeph)
    g2m = next(c for c in agg["curves"] if c["model"] == "graph2mat")
    assert g2m["target_id"] == "t50"
    assert [p["x"] for p in g2m["points"]] == [10, 50]
    first = g2m["points"][0]
    assert abs(first["mae"] - 0.5) < 1e-9
    assert abs(first["relative_frobenius"] - 0.2) < 1e-9
    assert first["relative_frobenius_std"] is not None
    assert first["n_seeds"] == 2
    assert {p["id"] for p in agg["payloads"]} == {"s10__to__t50", "s50__to__t50"}


def test_aggregate_falls_back_to_atoms_for_x() -> None:
    records = [
        {"source_id": "s", "target_id": "t", "source_n_atoms": 2,
         "model": "graph2mat", "seed": 0, "h_mae_eV": 0.3},
    ]
    agg = mvs.aggregate_cross_structure_mae(records)
    assert agg["curves"][0]["points"][0]["x"] == 2


def test_aggregate_keeps_source_structures_in_distinct_curves() -> None:
    records = [
        {"source_id": "w90_iid20", "source_system_label": "graphene_w90",
         "target_id": "vacancy", "source_n_snapshots": 20,
         "model": "graph2mat", "seed": 0, "h_mae_eV": 0.4},
        {"source_id": "x5_iid20", "source_system_label": "graphene_5x5",
         "target_id": "vacancy", "source_n_snapshots": 20,
         "model": "graph2mat", "seed": 0, "h_mae_eV": 0.2},
    ]
    agg = mvs.aggregate_cross_structure_mae(records)
    assert agg["n_curves"] == 2
    assert {curve["source_system_label"] for curve in agg["curves"]} == {
        "graphene_w90", "graphene_5x5",
    }


@pytest.mark.parametrize("action", ["materialize", "train"])
def test_non_prediction_actions_do_not_require_existing_artifacts(
    source_pair, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, action: str,
) -> None:
    small, _big, target = source_pair
    seen: list[dict] = []

    def fake_run(payload, launch_fn=None):
        seen.append(payload)
        if payload["action"] == "materialize":
            return {"materialized": {"reused": False}}
        return {
            "materialized": {"reused": False},
            "runner_result": {
                "ok": True,
                "metrics": {
                    "graph2mat": {"h_mae_eV": 0.1},
                    "deeph": {"h_mae_eV": 0.2},
                },
            },
        }

    monkeypatch.setattr(sweep, "run_cross_structure_payload", fake_run)
    summary = sweep.run_cross_structure_sweep(
        [small], [target], tmp_path / f"out_{action}", action=action, dry_run=False,
    )
    assert len(seen) == 1
    assert "existing_model_artifacts" not in seen[0]["runner_payload"]
    expected_status = "materialized" if action == "materialize" else "trained"
    assert summary["permutations"][0]["status"] == expected_status


def test_predict_metrics_requires_source_artifacts(source_pair, tmp_path: Path) -> None:
    small, _big, target = source_pair
    with pytest.raises(ValueError, match="requires existing_artifacts"):
        sweep.run_cross_structure_sweep(
            [small], [target], tmp_path / "missing_artifacts",
            action="predict_metrics", dry_run=False,
        )


def test_predict_metrics_fails_when_no_target_is_compatible(
    source_pair, tmp_path: Path,
) -> None:
    small, _big, _target = source_pair
    missing = tmp_path / "missing_vacancy"
    with pytest.raises(ValueError, match="predict_metrics has no compatible pairs.*missing_vacancy"):
        sweep.run_cross_structure_sweep(
            [small], [missing], tmp_path / "no_target",
            action="predict_metrics", dry_run=False,
        )


def test_predict_metrics_passes_exact_source_artifacts(
    source_pair, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    small, _big, target = source_pair
    source_id = mvs.plan_cross_structure_sweep([small], [target])["permutations"][0]["source_id"]
    artifacts = {
        "graph2mat": {"checkpoint": "/models/w90.ckpt", "run_dir": "/models/w90"},
        "deeph": {"checkpoint": "/models/w90.pt", "run_dir": "/models/w90_deeph"},
    }
    seen: list[dict] = []

    def fake_run(payload, launch_fn=None):
        seen.append(payload)
        return {
            "materialized": {"reused": True},
            "runner_result": {
                "ok": True,
                "metrics": {
                    "graph2mat": {"h_mae_eV": 0.1},
                    "deeph": {"h_mae_eV": 0.2},
                },
            },
        }

    monkeypatch.setattr(sweep, "run_cross_structure_payload", fake_run)
    summary = sweep.run_cross_structure_sweep(
        [small], [target], tmp_path / "predict",
        action="predict_metrics", dry_run=False,
        existing_artifacts={source_id: artifacts},
    )
    runner_payload = seen[0]["runner_payload"]
    assert runner_payload["predict_metrics_only"] is True
    assert runner_payload["existing_model_artifacts"] == artifacts
    assert summary["n_evaluated"] == 1
    assert summary["n_trained"] == 0


def test_cross_structure_pairs_run_in_parallel(source_pair, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    small, big, target = source_pair
    lock = threading.Lock()
    active = 0
    peak = 0

    def fake_run(payload, launch_fn=None):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        return {
            "materialized": {"reused": True},
            "runner_result": {
                "ok": True,
                "metrics": {"graph2mat": {"h_mae_eV": 0.1}, "deeph": {"h_mae_eV": 0.2}},
            },
        }

    monkeypatch.setattr(sweep, "run_cross_structure_payload", fake_run)
    summary = sweep.run_cross_structure_sweep(
        [small, big], [target], tmp_path / "parallel", action="train", dry_run=False,
        performance={"max_parallel_prediction_jobs": 2},
    )

    assert peak == 2
    assert summary["max_parallel_jobs"] == 2
    assert summary["n_trained"] == 2


def test_cross_structure_runs_deeph_batch_before_graph2mat(source_pair, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    small, big, target = source_pair
    calls: list[str] = []

    def fake_run(payload, launch_fn=None):
        model = payload["runner_payload"]["models"][0]
        calls.append(model)
        return {
            "materialized": {"reused": True},
            "runner_result": {"ok": True, "metrics": {model: {"h_mae_eV": 0.1}}},
        }

    monkeypatch.setattr(sweep, "run_cross_structure_payload", fake_run)
    summary = sweep.run_cross_structure_sweep(
        [small, big], [target], tmp_path / "ordered", action="train", dry_run=False,
        performance={
            "cross_model_schedule": "deeph_then_graph2mat",
            "max_parallel_deeph_training_jobs": 2,
            "max_parallel_graph2mat_training_jobs": 2,
        },
    )

    assert calls == ["deeph", "deeph", "graph2mat", "graph2mat"]
    assert summary["model_schedule"] == "deeph_then_graph2mat"
    assert summary["n_trained"] == 2


def test_runner_metrics_merge_live_records() -> None:
    import pipeline_ui as ui

    runner = ui.CrossStructureSweepRunner()
    # Seed the live status directly (no training) to exercise metrics() merge.
    records = [
        {"source_id": "s10", "target_id": "t50", "payload_id": "s10__to__t50",
         "source_n_snapshots": 10, "model": "graph2mat", "seed": 0, "h_mae_eV": 0.4},
    ]
    perms = [
        {"source_id": "s10", "target_id": "t50", "payload_id": "s10__to__t50",
         "source_n_snapshots": 10, "status": "trained"},
        {"source_id": "s99", "target_id": "t50", "payload_id": "s99__to__t50",
         "source_n_snapshots": 99, "status": "incompatible", "reason": "boom"},
    ]
    runner._status = {
        "state": "running",
        "action": "predict_metrics",
        "live_records": records,
        "payloads": ui._cross_testing_payloads_from_permutations(perms),
    }
    metrics = runner.metrics()
    ids = {p["id"] for p in metrics["payloads"]}
    # The trained pair (from records) and the incompatible pair (from perms) both
    # survive; the incompatible one carries its reason.
    assert "s10__to__t50" in ids
    assert "s99__to__t50" in ids
    incompatible = next(p for p in metrics["payloads"] if p["id"] == "s99__to__t50")
    assert incompatible["status"] == "incompatible"
    assert incompatible["reason"] == "boom"
    evaluated = next(p for p in metrics["payloads"] if p["id"] == "s10__to__t50")
    assert evaluated["status"] == "evaluated"
    assert metrics["n_curves"] == 1


def test_runner_metrics_ignores_empty_summary_and_loads_real_result(tmp_path: Path) -> None:
    import json
    import pipeline_ui as ui

    (tmp_path / "cross_structure_sweep_summary.json").write_text(
        json.dumps({"action": "predict_metrics", "records": []}), encoding="utf-8"
    )
    records = [
        {"source_id": "w90", "target_id": "x5", "source_n_snapshots": 20,
         "model": "graph2mat", "seed": 0, "h_mae_eV": 0.1},
    ]
    (tmp_path / "predict_metrics_result.json").write_text(
        json.dumps({"action": "predict_metrics", "records": records}), encoding="utf-8"
    )
    runner = ui.CrossStructureSweepRunner(tmp_path)
    metrics = runner.metrics()
    assert metrics["n_curves"] == 1
    assert metrics["curves"][0]["points"][0]["mae"] == 0.1


def test_runner_payload_seed_overrides_payload_hyperparam_seed() -> None:
    """The sweep seed must reach model-init overrides, beating pinned hyperparams.
    The 10-seed campaign of 2026-07-13 shipped every DeepH replicate with the
    default seed 42 because nothing forced the seed into the runner payload."""
    from ml_vs_siesta.cross_structure_sweep import _runner_payload

    payload = _runner_payload(
        ("graph2mat", "deeph"),
        None,
        None,
        hyperparams={"graph2mat": {"seed_everything": 1}, "deeph": {"seed": 1}},
        seed=7,
    )
    assert payload["graph2mat_overrides"]["seed_everything"] == 7
    assert payload["deeph"]["seed"] == 7
    assert payload["random_seed"] == 7
