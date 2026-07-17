"""Cross-structure sweep: planner product, MAE aggregator, live-record runner."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "Comparison" / "scripts"
SHARED_DIR = REPO_ROOT / "shared"
for path in (SCRIPTS_DIR, SHARED_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import ml_vs_siesta as mvs  # noqa: E402
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
         "model": "graph2mat", "seed": 0, "h_mae_eV": 0.4},
        {"source_id": "s10", "target_id": "t50", "source_n_snapshots": 10,
         "model": "graph2mat", "seed": 1, "h_mae_eV": 0.6},
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
    assert first["n_seeds"] == 2
    assert {p["id"] for p in agg["payloads"]} == {"s10__to__t50", "s50__to__t50"}


def test_aggregate_falls_back_to_atoms_for_x() -> None:
    records = [
        {"source_id": "s", "target_id": "t", "source_n_atoms": 2,
         "model": "graph2mat", "seed": 0, "h_mae_eV": 0.3},
    ]
    agg = mvs.aggregate_cross_structure_mae(records)
    assert agg["curves"][0]["points"][0]["x"] == 2


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
    assert metrics["n_curves"] == 1


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
