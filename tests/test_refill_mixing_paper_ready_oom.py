from __future__ import annotations

import json
from pathlib import Path

import Comparison.scripts.refill_mixing_paper_ready_oom as refill

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_plan_has_exactly_the_failed_permutations_no_cartesian_blowup() -> None:
    datasets_list, manual_runs = refill._build_plan()

    g2m = [r for r in manual_runs if r["model"] == "graph2mat"]
    deeph = [r for r in manual_runs if r["model"] == "deeph"]
    assert len(g2m) == 20
    assert len(deeph) == 4
    # No re-training of successes: exactly the requested names, no 7x2x2 product.
    assert {r["dataset_id"] for r in g2m} == set(refill.G2M_PERMUTATIONS)
    assert {r["dataset_id"] for r in deeph} == set(refill.DEEPH_PERMUTATIONS)

    # One dataset entry per unique permutation; DeepH-only ones are included.
    assert len(datasets_list) == 21
    ids = [d["dataset_id"] for d in datasets_list]
    assert len(ids) == len(set(ids)), "duplicate dataset_id"
    assert "size200_add_r1p000" in ids  # DeepH-only perm not in the G2M list


def test_every_run_resolves_to_its_own_dataset_root() -> None:
    datasets_list, manual_runs = refill._build_plan()
    root_by_id = {d["dataset_id"]: d["dataset_root"] for d in datasets_list}
    for run in manual_runs:
        root = root_by_id[run["dataset_id"]]
        assert root.endswith(run["dataset_id"])
        assert str(refill.SWEEP_ROOT) in root


def test_hyperparams_match_original_paper_ready_payload_verbatim() -> None:
    original = json.loads(
        (REPO_ROOT / "Comparison" / "config"
         / "ml_vs_siesta_mixing_sweep_20_500_paper_ready_train_payload.json").read_text()
    )
    hp = original["hyperparams"]
    # Every key the original pins must survive unchanged in our overrides.
    for key, val in hp["graph2mat"].items():
        assert refill.G2M_OVERRIDES[key] == val, f"g2m override drift: {key}"
    for key, val in hp["deeph"].items():
        assert refill.DEEPH_OVERRIDES[key] == val, f"deeph override drift: {key}"
    assert refill.G2M_OVERRIDES["max_epochs"] == original["epochs"] == 750
    assert refill.DEEPH_OVERRIDES["epochs"] == original["epochs"] == 750


def test_training_parallelism_lowered_to_fit_gpu() -> None:
    # The whole point of the refill: fewer replicas so they fit in GPU RAM
    # (dropped further to G2M=1/DeepH=2 after the first attempt).
    assert refill.PERFORMANCE["max_parallel_graph2mat_training_jobs"] == 1
    assert refill.PERFORMANCE["max_parallel_deeph_training_jobs"] == 2


def test_output_root_is_separate_from_original_sweep() -> None:
    # Must never overwrite the 28 G2M + 44 DeepH successes.
    assert refill.OUTPUT_ROOT != refill.SWEEP_ROOT
    assert refill.OUTPUT_ROOT.name.endswith("_refill")


def test_payload_is_json_serializable_and_reuses_validated() -> None:
    datasets_list, manual_runs = refill._build_plan()
    payload = refill._build_payload(datasets_list, manual_runs)
    json.dumps(payload)  # raises if not serializable
    assert payload["dataset_mode"] == "reuse_validated"
    assert payload["strict_dataset_validation"] is False
    assert payload["_mixing_datasets_list"] == datasets_list
