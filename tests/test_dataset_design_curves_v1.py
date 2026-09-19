"""Pre-registered decision rules of dataset_design_curves_v1 (pure logic, no training)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Comparison/scripts"))

import dataset_design_curves_v1 as c  # noqa: E402


def _recipe(rid, family, dim, r, curve):
    return {"recipe_id": rid, "family": family, "dim": dim, "k": 1, "R": r, "E": curve,
            "cost": {n: n * 0.01 for n in curve}}


def test_n_star_gains_and_n12():
    curve = {4: 100.0, 8: 90.0, 16: 60.0, 32: 58.0, 64: 57.5}
    assert c.n_star(curve) == 16
    assert abs(c.gains(curve)["8->16"] - 1 / 3) < 1e-12
    assert c.needs_n12(curve)  # big 8->16 drop, 8 unsaturated, 16 saturated
    assert not c.needs_n12({4: 100.0, 8: 80.0, 16: 64.0, 32: 51.0, 64: 41.0})


def test_pareto_front():
    points = [{"cost": 1, "E": 10}, {"cost": 2, "E": 5}, {"cost": 2, "E": 6}, {"cost": 3, "E": 5}]
    assert c.pareto_front(points) == [{"cost": 1, "E": 10}, {"cost": 2, "E": 5}]


def test_control_uses_only_a_and_b():
    a = {"recipe_id": "a", "dim": "2D_in", "R": 0.05}
    b = {"recipe_id": "b", "dim": "3D", "R": 0.03}
    available = [{"recipe_id": f"{d}{r}", "dim": d, "R": r} for d in c.DIMS for r in (0.03, 0.05, 0.08, 0.12)]
    control = c.choose_control(a, b, available)
    assert control["dim"] == "1D_z"  # farthest axis set from {x,y} and {x,y,z}
    assert control["R"] == 0.12  # farthest in log R from 0.03 and 0.05


def test_select_recipes_b_changes_dim():
    ranking = [{"recipe_id": f"{f}{d}{r}", "family": f, "dim": d, "R": r, "k": 1, "dev_H_MAE_meV": e}
               for f in c.CURVE_FAMILIES for e, (d, r) in enumerate(
                   [("2D_in", 0.05), ("2D_in", 0.03), ("3D", 0.05), ("1D_in", 0.08), ("1D_z", 0.03)])]
    selected = c.select_recipes(ranking)
    assert [s["role"] for s in selected] == ["A", "B", "C"] * 2
    assert selected[0]["dim"] == "2D_in" and selected[1]["dim"] == "3D"


def test_finalists_precision_efficient_alternative():
    curves = {
        "s1": _recipe("s1", "sobol_sparse", "2D_in", 0.05, {4: 90, 8: 70, 16: 51, 32: 50, 64: 50}),
        "s2": _recipe("s2", "sobol_sparse", "3D", 0.05, {4: 95, 8: 60, 16: 52, 32: 52, 64: 52}),
        "r1": _recipe("r1", "random_cartesian", "1D_in", 0.08, {4: 99, 8: 80, 16: 70, 32: 60, 64: 55}),
    }
    finalists = c.choose_finalists(curves, None)
    assert [(f["role"], f["recipe_id"]) for f in finalists] == [
        ("precision", "s1"), ("efficient", "s2"), ("alternative", "r1")]
    # s1@16 (51) is the cheapest point within 5% of E_s1(64)=50 -> same recipe, so s1 keeps N*=16
    # and "efficient" becomes the cheapest other-recipe point within 5%: s2@16 (52 <= 52.5).
    assert finalists[0]["N_star_candidate"] == 16
    assert finalists[1]["N_star_candidate"] == 16
    assert finalists[2]["N_star_candidate"] == 64  # r1: E(32)=60 > 1.05*55


def test_efficient_falls_back_to_next_pareto_point():
    curves = {
        "r1": _recipe("r1", "random_cartesian", "1D_in", 0.08, {4: 130, 8: 97, 16: 115, 32: 79, 64: 61}),
        "s1": _recipe("s1", "sobol_sparse", "2D_in", 0.08, {4: 94, 8: 82, 16: 85, 32: 132, 64: 80}),
        "r2": _recipe("r2", "random_cartesian", "3D", 0.08, {4: 95, 8: 76, 16: 94, 32: 92, 64: 70}),
    }
    efficient = c.choose_finalists(curves, None)[1]
    # nothing within 5% of 61 -> the frontier neighbour below r1@64, not the cheapest point s1@4
    assert (efficient["recipe_id"], efficient["N_star_candidate"]) == ("r2", 8)


def test_confirmation_rules():
    ok = c.confirm_n_star([51.0, 52.0, 50.5], [50.0, 50.5, 49.8], True)
    assert ok["confirmed"]
    bad = c.confirm_n_star([60.0, 52.0, 51.0], [50.0, 50.5, 49.8], True)
    assert not bad["confirmed"] and bad["rule1_mean_and_median"] is False


def test_final_test_generator_hits_exact_amplitudes():
    entries = c.final_test_configs("w90")
    assert len(entries) == 48
    for _sid, config, meta in entries:
        norms = [np.linalg.norm(v) for v in config.displacements_ang.values()]
        assert abs(max(norms) - meta["A"]) < 1e-12
        if meta["dim"] == "1D_z":
            assert all(abs(v[0]) < 1e-15 and abs(v[1]) < 1e-15 for v in config.displacements_ang.values())
