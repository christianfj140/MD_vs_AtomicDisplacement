"""Pre-registered decision rules of dataset_design_curves_v1 (pure logic, no training)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Comparison/scripts"))

import dataset_design_curves_v1 as c  # noqa: E402
import run_6x6_label_budget as budget  # noqa: E402


def _recipe(rid, family, dim, r, curve):
    return {"recipe_id": rid, "family": family, "dim": dim, "k": 1, "R": r, "E": curve,
            "cost": {n: n * 0.01 for n in curve}}


def test_no_early_stop_preserves_paired_experiment(monkeypatch):
    import copy

    baseline = [{"stage": "curve_elemmse", "job_id": f"old_{seed}",
                 "training_seed": seed, "N": 64, "train": ["same_data"],
                 "config_patch": c.elemmse_patch("w90")} for seed in range(5)]
    monkeypatch.setattr(c, "curve_elemmse_jobs", lambda system: copy.deepcopy(baseline))
    assert c.no_early_stop_jobs("6x6") == []
    jobs = c.no_early_stop_jobs("w90")
    assert len(jobs) == 5
    for old, new in zip(baseline, jobs):
        assert new.pop("stage") == "no_early_stop"
        assert new.pop("job_id") == old["job_id"] + "__no_early_stop"
        callbacks = new["config_patch"]["trainer"].pop("callbacks")
        assert [cb["class_path"] for cb in callbacks] == ["ModelCheckpoint"]
        assert callbacks[0]["init_args"]["monitor"] == "val_loss"
        assert callbacks[0]["init_args"]["save_top_k"] == 1
        assert new == {k: v for k, v in old.items() if k not in ("stage", "job_id")}


def test_no_early_stop_4k_only_extends_budget_and_schedule(monkeypatch):
    import copy

    baseline = [{"stage": "no_early_stop", "job_id": f"old_{seed}",
                 "training_seed": seed, "N": 64, "train": ["same_data"],
                 "config_patch": c.elemmse_patch("w90")} for seed in range(5)]
    monkeypatch.setattr(c, "no_early_stop_jobs",
                        lambda system: copy.deepcopy(baseline) if system == "w90" else [])
    assert c.no_early_stop_4k_jobs("6x6") == []
    jobs = c.no_early_stop_4k_jobs("w90")
    assert len(jobs) == 5
    for old, new in zip(baseline, jobs):
        assert new.pop("stage") == "no_early_stop_4k"
        assert new.pop("job_id") == old["job_id"] + "__4k"
        assert new["config_patch"]["trainer"]["max_epochs"] == 4000
        assert new["config_patch"]["lr_scheduler"] == c._cosine(4000)
        new["config_patch"]["trainer"]["max_epochs"] = 2000
        new["config_patch"]["lr_scheduler"] = c._cosine(2000)
        assert new == {k: v for k, v in old.items() if k not in ("stage", "job_id")}


def test_coverage_uses_same_budget_and_common_validation_for_md(monkeypatch):
    rows = [{"sample_id": str(i)} for i in range(64)]
    def recipe(name):
        return {"recipe_id": name, "family": "random_cartesian", "dim": "3D", "k": 2,
                "R": 0.08, "pool_hash": name}
    monkeypatch.setattr(c, "_finalist", lambda *args: (recipe("3d08"), rows))
    monkeypatch.setattr(c, "read_json", lambda path:
                        recipe("3d12") | {"samples": rows} if "coverage_pool" in str(path)
                        else {"R_train_md": 0.025, "pool_hash": "md", "train_pool": rows})
    patch = {"trainer": {"max_epochs": 4000}, "lr_scheduler": c._cosine(4000)}
    monkeypatch.setattr(c, "no_early_stop_4k_jobs", lambda system:
                        [{"training_seed": seed, "config_patch": patch} for seed in range(5)])
    assert c.coverage_4k_jobs("6x6") == []
    jobs = c.coverage_4k_jobs("w90")
    assert len(jobs) == len({j["job_id"] for j in jobs}) == 15
    for name in ("3d08", "3d12", "md_w90"):
        arm = [j for j in jobs if j["recipe_id"] == name]
        assert [j["training_seed"] for j in arm] == list(range(5))
        assert all(j["val"] == "dev" and j["N"] == 64 and j["train"] == rows
                   and j["config_patch"] == patch for j in arm)


def test_coverage_curve_equalizes_updates_and_validations(monkeypatch):
    recipe = {"recipe_id": "3d12", "family": "random_cartesian", "dim": "3D", "k": 2,
              "R": 0.12, "pool_hash": "pool", "samples": [{"sample_id": str(i)} for i in range(64)]}
    monkeypatch.setattr(c, "read_json", lambda path: recipe)
    assert c.coverage_curve_jobs("6x6") == []
    jobs = c.coverage_curve_jobs("w90")
    assert len(jobs) == len({j["job_id"] for j in jobs}) == 20
    for n in (4, 8, 16, 32):
        arm = [j for j in jobs if j["N"] == n]
        assert [j["training_seed"] for j in arm] == list(range(5))
        for job in arm:
            assert len(job["train"]) == n
            assert job["config_patch"]["trainer"]["max_epochs"] == 8000
            assert job["config_patch"]["trainer"]["check_val_every_n_epoch"] == 2
            assert [cb["class_path"] for cb in job["config_patch"]["trainer"]["callbacks"]] == ["ModelCheckpoint"]
            assert job["config_patch"]["lr_scheduler"] == c._cosine(8000)


def test_6x6_followup_has_five_seeds_equal_updates_and_no_early_stop(monkeypatch):
    rows = [{"sample_id": str(i)} for i in range(64)]

    def recipe(name, dim, radius):
        return {"recipe_id": name, "family": "sobol_sparse", "dim": dim, "k": 72,
                "R": radius, "pool_hash": name}

    one_d, three_d = recipe("1d12", "1D_in", 0.12), recipe("3d08", "3D", 0.08)
    expanded = recipe("3d12", "3D", 0.12) | {"samples": rows}
    monkeypatch.setattr(c, "_finalist", lambda system, role:
                        (one_d, rows) if role == "precision" else (three_d, rows))
    monkeypatch.setattr(c, "read_json", lambda path: expanded)
    jobs = c.coverage_6x6_jobs("6x6")
    assert len(jobs) == len({job["job_id"] for job in jobs}) == 15
    assert c.coverage_6x6_jobs("w90") == []
    for job in jobs:
        patch = job["config_patch"]
        assert job["N"] == 64 and patch["trainer"]["max_epochs"] == 2000
        assert patch["trainer"]["check_val_every_n_epoch"] == 1
        assert [cb["class_path"] for cb in patch["trainer"]["callbacks"]] == ["ModelCheckpoint"]
        assert patch["lr_scheduler"] == c._cosine(2000)

    winner = {"recipe": three_d, "samples": rows}
    monkeypatch.setattr(c, "read_json", lambda path: winner)
    curve = c.coverage_curve_6x6_jobs("6x6")
    assert len(curve) == len({job["job_id"] for job in curve}) == 20
    for n in (4, 8, 16, 32):
        arm = [job for job in curve if job["N"] == n]
        assert [job["training_seed"] for job in arm] == list(range(5))
        assert arm[0]["config_patch"]["trainer"]["max_epochs"] == 8000 // max(1, -(-n // 16))
        assert arm[0]["config_patch"]["trainer"]["check_val_every_n_epoch"] == 4 // max(1, -(-n // 16))


def test_label_budget_validation_sets_are_balanced_nested_and_disjoint():
    rows = [{"sample_id": f"{family}-{dim}-{stratum}", "hash": f"v-{family}-{dim}-{stratum}",
             "family": family, "dim": dim, "stratum": stratum}
            for family in ("latin", "random", "sobol") for dim in c.DIMS for stratum in range(4)]
    sets = budget.validation_sets(rows)
    ids = {n: {row["sample_id"] for row in selected} for n, selected in sets.items()}
    assert len(ids[8]) == 8 and len(ids[24]) == 24 and len(ids[48]) == 48
    assert ids[8] < ids[24] < ids[48]
    for n, per_level in ((8, 2), (24, 6), (48, 12)):
        assert {dim: sum(row["dim"] == dim for row in sets[n]) for dim in c.DIMS} == dict.fromkeys(c.DIMS, per_level)
        assert {level: sum(row["stratum"] == level for row in sets[n]) for level in range(4)} == dict.fromkeys(range(4), per_level)
    assert sorted(sum(row["family"] == family for row in sets[8]) for family in ("latin", "random", "sobol")) == [2, 3, 3]
    assert budget.disjoint_hashes(rows, [{"hash": "test"}], [{"hash": "train"}])
    assert not budget.disjoint_hashes(rows, [{"hash": rows[0]["hash"]}], [{"hash": "train"}])


def test_label_budget_test_draws_are_paired_nested_balanced_and_weighted():
    rows = [{"sample_id": f"{dim}-{index:02d}", "dim": dim} for dim in c.DIMS for index in range(12)]
    orders = budget.dimension_orders(rows, repeats=20, seed=7)
    for dim in c.DIMS:
        for repeat in range(20):
            assert set(orders[dim][repeat, :1]) < set(orders[dim][repeat, :2]) < set(orders[dim][repeat, :4])
    values = {row["sample_id"]: 10.0 * c.DIMS.index(row["dim"]) for row in rows}
    for n_test in budget.TEST_SIZES:
        means = budget.subsample_means(values, orders, n_test)
        assert means.shape == (20,)
        assert np.allclose(means, 15.0)  # equal weight for all four dimensional strata
        assert np.array_equal(means, budget.subsample_means(values, orders, n_test))


def test_label_budget_minimum_uses_total_labels_then_conservative_error():
    rows = [
        {"pass": False, "N_total": 10, "H_MAE_conservative_meV": 1.0, "N_train": 4, "N_val": 4, "N_test": 2},
        {"pass": True, "N_total": 20, "H_MAE_conservative_meV": 1.2, "N_train": 8, "N_val": 8, "N_test": 4},
        {"pass": True, "N_total": 20, "H_MAE_conservative_meV": 1.1, "N_train": 4, "N_val": 8, "N_test": 8},
    ]
    assert budget.choose_minimum(rows) is rows[2]
    assert budget.choose_minimum([rows[0]]) is None


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
