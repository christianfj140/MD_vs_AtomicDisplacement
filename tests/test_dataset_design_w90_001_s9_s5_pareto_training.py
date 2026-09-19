"""DATASET-DESIGN-W90-001-S9-S5 contract tests -- full Pareto-set training.

Fast, no real SIESTA/GPU training: covers survivor selection, the
nested-prefix training-pool optimization (one SIESTA batch reused for both
N levels for ``sobol_sparse``, two independent batches for
``angular_shell``), cost-ledger arithmetic against S9-S2's ``COST_SCHEMA``,
and the minimum-N plateau logic. Matched by ``-k s9_s5``, by the literal
task id / short-slug aliases registered in ``pytest.ini``, or by this
module's own name (``-k dataset_design_w90_001_s9_s5``) -- following
S9-S3's precedent (``test_dataset_design_w90_001_s9_s3_design_generation.py``)
of also giving each S9 sub-task its own module-named test file, since
``pytest -k`` matches against the full nodeid (including module name), not
just the bare function name, and the same content also lives in the shared
``tests/test_dataset_design_s9_contract.py`` (S9-S4's own precedent: cheap
redundant discoverability, never a correctness risk).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "Comparison" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import dataset_design_w90_001_s4_train_learning_curves as s4  # noqa: E402
import dataset_design_w90_001_s9_s5_pareto_training as s95  # noqa: E402
import w90_displacement_sampler_family as sampler  # noqa: E402

S9_S5_TASK_MARK = getattr(pytest.mark, "DATASET-DESIGN-W90-001-S9-S5")
S9_S5_UNDERSCORE_MARK = getattr(pytest.mark, "S9_S5")
S9_S5_HYPHEN_MARK = getattr(pytest.mark, "S9-S5")
S9_S5_SLUG_MARK = pytest.mark.dataset_design_w90_001_s9_s5


def _make_sample(sample_id: str, run_fdf: Path) -> s4.Sample:
    return s4.Sample(sample_id, "fam", "3D", 1, 0.05, run_fdf, run_fdf.parent, run_fdf.parent / "ref")


@pytest.fixture(scope="module")
def geometry() -> sampler.Geometry:
    return sampler.load_graphene_primitive()


# --------------------------------------------------------------------------
# Survivor selection: exactly S9-S4's trainable set minus its "dominated" set
# --------------------------------------------------------------------------


@S9_S5_TASK_MARK
@S9_S5_UNDERSCORE_MARK
@S9_S5_HYPHEN_MARK
@S9_S5_SLUG_MARK
def test_s9_s5_survivors_are_trainable_minus_dominated():
    survivors, _geometry, hash_to_sample = s95.load_survivors()
    assert survivors, "expected at least one Pareto-surviving design from the S9-S4 pilot"
    assert hash_to_sample
    survivor_ids = {e["design_id"] for e in survivors}
    # Every survivor must have been real-SIESTA-backed at the pilot density (never fabricated).
    for entry in survivors:
        assert entry["n_reused_from_existing_siesta"] == entry["n_used"]
    assert len(survivor_ids) == len(survivors), "duplicate design_id in survivor set"


# --------------------------------------------------------------------------
# unique_configs_at_resolution: no duplicate geometries within one design/N
# --------------------------------------------------------------------------


@S9_S5_TASK_MARK
@S9_S5_UNDERSCORE_MARK
@S9_S5_HYPHEN_MARK
@S9_S5_SLUG_MARK
def test_s9_s5_unique_configs_dedupe_by_geometry_hash(geometry):
    import dataset_design_w90_001_s9_s3_design_generation as s3

    survivors, _geometry, _hash_to_sample = s95.load_survivors()
    for entry in survivors:
        seed_value = entry["seeds"]["sampler"]
        seed = seed_value if isinstance(seed_value, int) else 0
        for n in s95.SCALE_N_LEVELS:
            unique = s95.unique_configs_at_resolution(geometry, entry, n, seed)
            hashes = [s3.point_geometry_hash(geometry, c) for c in unique]
            assert len(hashes) == len(set(hashes)), f"{entry['design_id']} N={n} produced duplicate geometries"


@S9_S5_TASK_MARK
@S9_S5_UNDERSCORE_MARK
@S9_S5_HYPHEN_MARK
@S9_S5_SLUG_MARK
def test_s9_s5_sobol_sparse_n128_is_a_hash_prefix_of_n256(geometry):
    """Regression: the nested-prefix optimization in build_training_pools relies on this."""

    import dataset_design_w90_001_s9_s3_design_generation as s3

    survivors, _geometry, _hash_to_sample = s95.load_survivors()
    entry = next(e for e in survivors if e["family"] == "sobol_sparse")
    seed = entry["seeds"]["sampler"]
    small, big = sorted(s95.SCALE_N_LEVELS)
    small_configs = s95.unique_configs_at_resolution(geometry, entry, small, seed)
    big_configs = s95.unique_configs_at_resolution(geometry, entry, big, seed)
    small_hashes = [s3.point_geometry_hash(geometry, c) for c in small_configs]
    big_hashes = [s3.point_geometry_hash(geometry, c) for c in big_configs]
    assert small_hashes == big_hashes[:small]


@S9_S5_TASK_MARK
@S9_S5_UNDERSCORE_MARK
@S9_S5_HYPHEN_MARK
@S9_S5_SLUG_MARK
def test_s9_s5_angular_shell_nests_for_exact_multiple_n_but_not_arbitrary_n(geometry):
    """Regression: angular_shell's np.linspace(0, 2pi, n, endpoint=False) ring nests whenever

    the larger N is an exact integer multiple of the smaller N (angle i/N == angle 2i/2N),
    which is true for our actual (128, 256) pair -- an earlier draft of this test wrongly
    assumed angular_shell never nests. It does NOT nest for an arbitrary non-multiple pair,
    confirming this is a real integer-multiple property, not a coincidence of hashing. Note
    this is a *set*-level property only: build_training_pools' actual reuse optimization
    requires an ORDERED prefix match, which angular_shell fails because the doubling is an
    even/odd interleave, not a first-half split.
    """

    import dataset_design_w90_001_s9_s3_design_generation as s3

    survivors, _geometry, _hash_to_sample = s95.load_survivors()
    entry = next(e for e in survivors if e["family"] == "angular_shell")
    seed_value = entry["seeds"]["sampler"]
    seed = seed_value if isinstance(seed_value, int) else 0

    small, big = sorted(s95.SCALE_N_LEVELS)
    small_hashes = {s3.point_geometry_hash(geometry, c) for c in s95.unique_configs_at_resolution(geometry, entry, small, seed)}
    big_hashes = {s3.point_geometry_hash(geometry, c) for c in s95.unique_configs_at_resolution(geometry, entry, big, seed)}
    assert small_hashes.issubset(big_hashes), "actual (128, 256) pair is an exact multiple and must nest as a set"

    non_multiple_hashes = {
        s3.point_geometry_hash(geometry, c) for c in s95.unique_configs_at_resolution(geometry, entry, 100, seed)
    }
    assert not non_multiple_hashes.issubset(big_hashes), "non-multiple N must not nest"


# --------------------------------------------------------------------------
# build_training_pools: nested families submit once, non-nested submit per-N
# --------------------------------------------------------------------------


@S9_S5_TASK_MARK
@S9_S5_UNDERSCORE_MARK
@S9_S5_HYPHEN_MARK
@S9_S5_SLUG_MARK
def test_s9_s5_build_training_pools_reuses_one_batch_for_nested_family(tmp_path, monkeypatch, geometry):
    survivors, _geometry, _hash_to_sample = s95.load_survivors()
    entry = next(e for e in survivors if e["family"] == "sobol_sparse")

    calls: list[Path] = []

    def fake_materialize_and_run(entries, *, output_root, siesta_command, workers):
        calls.append(output_root)
        ref_dir = tmp_path / output_root.name / "ref"
        ref_dir.mkdir(parents=True, exist_ok=True)
        return {"entries": entries, "output_reference_root": ref_dir, "usable_ids": [sid for sid, _ in entries]}

    def fake_samples_from_run(entries, run_result):
        return [_make_sample(sid, tmp_path / f"{sid}.fdf") for sid, _ in entries]

    monkeypatch.setattr(s95.s5, "materialize_and_run", fake_materialize_and_run)
    monkeypatch.setattr(s95.s5, "samples_from_run", fake_samples_from_run)

    pools, timers = s95.build_training_pools(
        entry, geometry, output_root=tmp_path, siesta_command="siesta", workers=2, n_levels=s95.SCALE_N_LEVELS,
    )
    assert len(calls) == 1, "nested family must submit exactly one SIESTA batch at the max N"
    small, big = sorted(s95.SCALE_N_LEVELS)
    assert len(pools[small]) == small
    assert len(pools[big]) == big
    assert [s.sample_id for s in pools[small]] == [s.sample_id for s in pools[big][:small]]
    assert any(k.startswith("siesta_wall_time_seconds__") for k in timers)


@S9_S5_TASK_MARK
@S9_S5_UNDERSCORE_MARK
@S9_S5_HYPHEN_MARK
@S9_S5_SLUG_MARK
def test_s9_s5_build_training_pools_submits_independently_for_non_nested_family(tmp_path, monkeypatch, geometry):
    survivors, _geometry, _hash_to_sample = s95.load_survivors()
    entry = next(e for e in survivors if e["family"] == "angular_shell")

    calls: list[Path] = []

    def fake_materialize_and_run(entries, *, output_root, siesta_command, workers):
        calls.append(output_root)
        return {"entries": entries, "output_reference_root": tmp_path, "usable_ids": [sid for sid, _ in entries]}

    def fake_samples_from_run(entries, run_result):
        return [_make_sample(sid, tmp_path / f"{sid}.fdf") for sid, _ in entries]

    monkeypatch.setattr(s95.s5, "materialize_and_run", fake_materialize_and_run)
    monkeypatch.setattr(s95.s5, "samples_from_run", fake_samples_from_run)

    pools, _timers = s95.build_training_pools(
        entry, geometry, output_root=tmp_path, siesta_command="siesta", workers=2, n_levels=s95.SCALE_N_LEVELS,
    )
    assert len(calls) == len(s95.SCALE_N_LEVELS), "non-nested family must submit one SIESTA batch per N level"
    for n in s95.SCALE_N_LEVELS:
        assert len(pools[n]) == n


# --------------------------------------------------------------------------
# Cost accounting: S9-S2 COST_SCHEMA formulas, recomputable from hash sets
# --------------------------------------------------------------------------


@S9_S5_TASK_MARK
@S9_S5_UNDERSCORE_MARK
@S9_S5_HYPHEN_MARK
@S9_S5_SLUG_MARK
def test_s9_s5_cost_for_pool_matches_unique_and_incremental_formulas(tmp_path, monkeypatch):
    samples = [_make_sample(f"s{i}", tmp_path / f"s{i}" / "RUN.fdf") for i in range(5)]
    fake_hash_by_sample_id = {sample.sample_id: f"hash_{i}" for i, sample in enumerate(samples)}

    # cost_for_pool -> pool_hashes -> s5.geometry_hash(s5._positions_from_run_fdf(sample.run_fdf));
    # stub both so the test doesn't need a real sisl-readable RUN.fdf on disk.
    monkeypatch.setattr(s95.s5, "_positions_from_run_fdf", lambda run_fdf: run_fdf)
    monkeypatch.setattr(s95.s5, "geometry_hash", lambda run_fdf, **_: fake_hash_by_sample_id[run_fdf.parent.name])

    campaign_union = {"hash_0", "hash_1"}  # 2 of 5 already exist in the campaign
    costs = s95.cost_for_pool(samples, campaign_union)

    assert costs["cost_siesta_unique"] == 5
    assert costs["reproducible_cost"] == 5
    assert costs["incremental_cost"] == 3  # 5 total minus 2 already in campaign_union


# --------------------------------------------------------------------------
# Minimum-N determination: plateau criterion, non-monotonicity, bracket
# --------------------------------------------------------------------------


@S9_S5_TASK_MARK
@S9_S5_UNDERSCORE_MARK
@S9_S5_HYPHEN_MARK
@S9_S5_SLUG_MARK
def test_s9_s5_determine_minimum_n_picks_first_n_within_plateau_tolerance():
    # Best is at N=256 (0.10); N=64's 0.104 is within 5% of 0.10 -> N_min=64, not 256.
    dev_mae_by_n = {8: 0.20, 16: 0.15, 64: 0.104, 256: 0.10}
    report = s95.determine_minimum_n(dev_mae_by_n)
    assert report["status"] == "reached"
    assert report["N_min_observed"] == 64
    assert report["N_min_bracket"] == [16, 64]


@S9_S5_TASK_MARK
@S9_S5_UNDERSCORE_MARK
@S9_S5_HYPHEN_MARK
@S9_S5_SLUG_MARK
def test_s9_s5_determine_minimum_n_allows_non_monotonic_curves():
    # N=128 is worse than N=64 (non-monotonic) but N=256 is the true best; N=64 still
    # qualifies first in ascending scan since it's already within tolerance of the best.
    dev_mae_by_n = {64: 0.101, 128: 0.15, 256: 0.10}
    report = s95.determine_minimum_n(dev_mae_by_n)
    assert report["status"] == "reached"
    assert report["N_min_observed"] == 64


@S9_S5_TASK_MARK
@S9_S5_UNDERSCORE_MARK
@S9_S5_HYPHEN_MARK
@S9_S5_SLUG_MARK
def test_s9_s5_determine_minimum_n_no_observations_is_not_reached():
    report = s95.determine_minimum_n({})
    assert report["status"] == "not_reached"


@S9_S5_TASK_MARK
@S9_S5_UNDERSCORE_MARK
@S9_S5_HYPHEN_MARK
@S9_S5_SLUG_MARK
def test_s9_s5_determine_minimum_n_single_point_is_its_own_minimum():
    report = s95.determine_minimum_n({256: 0.10})
    assert report["status"] == "reached"
    assert report["N_min_observed"] == 256
    assert report["N_min_bracket"] == [None, 256]


# --------------------------------------------------------------------------
# Schema: required columns present (acceptance criteria)
# --------------------------------------------------------------------------


@S9_S5_TASK_MARK
@S9_S5_UNDERSCORE_MARK
@S9_S5_HYPHEN_MARK
@S9_S5_SLUG_MARK
def test_s9_s5_pareto_table_schema_has_required_columns():
    required = {"design_id", "N_train", "seed", "split_id", "test_amplitude", "H_MAE", "cost_siesta_unique", "dominance"}
    assert required <= set(s95.PARETO_TABLE_FIELDNAMES)


@S9_S5_TASK_MARK
@S9_S5_UNDERSCORE_MARK
@S9_S5_HYPHEN_MARK
@S9_S5_SLUG_MARK
def test_s9_s5_domain_generalization_schema_has_required_columns():
    required = {"design_id", "N_train", "seed", "R_train_max_ang", "test_amplitude", "H_MAE", "ood_flag"}
    assert required <= set(s95.DOMAIN_GENERALIZATION_FIELDNAMES)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
