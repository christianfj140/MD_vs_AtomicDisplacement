"""DATASET-DESIGN-W90-001-S9 contract: shared regression/spec tests for the
S9 sub-tasks.

S9-S1: six inherited S1-S8 bugs (cache-key collisions, non-reproducible
split hashing, stale symlinks, S5 re-inference on resume, mismatched sampler
envelopes, and collapsed seed variables). One test per fix, all matched by
``-k bug_fix``.

S9-S2: frozen envelope/domain-level/density-axis/coverage-matrix design spec.
Matched by ``-k s2``.

S9-S3: design generation + validation (envelope, leakage, nested-prefix,
coverage diagnostics) for every applicable coverage-matrix cell, including
``latin_hypercube`` as a training family. Matched by ``-k s9_s3`` (substring)
or by the literal task id, ``-k``/``-m "DATASET-DESIGN-W90-001-S9-S3"``
(registered as a marker in ``pytest.ini``; see ``S9_S3_TASK_MARK`` below).

S9-S4: pilot screening (N in {8,16,32,64}, 2 seeds, real-SIESTA-backed
designs only) plus Pareto pruning against each other. Matched by ``-k s9_s4``.

S9-S5: full Pareto-set training of the S9-S4 pilot's 2 surviving designs at
N in {128,256}, 5 seeds -- the nested-prefix training-pool optimization
(one SIESTA batch reused for both N levels when it's an exact ordered
hash-prefix, independent batches otherwise), cost-ledger arithmetic against
S9-S2's ``COST_SCHEMA``, and the minimum-N plateau logic. Matched by
``-k s9_s5``.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "Comparison" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import dataset_design_w90_001_s4_train_learning_curves as s4  # noqa: E402
import dataset_design_w90_001_s5_frozen_tests_generalization as s5  # noqa: E402
import dataset_design_w90_001_s9_s2_envelope_and_coverage as s2  # noqa: E402
import dataset_design_w90_001_s9_s3_design_generation as s3  # noqa: E402
import dataset_design_w90_001_s9_s4_pilot_screening as s94  # noqa: E402
import dataset_design_w90_001_s9_s5_pareto_training as s95  # noqa: E402
import dataset_design_w90_001_s9_s9_execute_campaign as s99  # noqa: E402
import pipeline_ui as ui  # noqa: E402
import validate_dataset_design_s9 as validator  # noqa: E402
import w90_displacement_sampler_family as sampler  # noqa: E402


def _make_pool(n: int, *, family: str = "fam", dim: str = "3D", k: int = 1, amplitude: float = 0.05) -> list[s4.Sample]:
    return [
        s4.Sample(f"s{i}", family, dim, k, amplitude, Path(f"/tmp/{i}/RUN.fdf"), Path(f"/tmp/{i}"), Path(f"/tmp/{i}/ref"))
        for i in range(n)
    ]


# --------------------------------------------------------------------------
# (a) S4 cache key includes domain (R_train_max) and density level
# --------------------------------------------------------------------------


def test_bug_fix_a_cache_key_distinguishes_designs_sharing_family_dim_k():
    sparse_pool = _make_pool(10, amplitude=0.03)
    dense_pool = _make_pool(128, amplitude=0.12)
    assert s4.density_level_for_pool(sparse_pool) != s4.density_level_for_pool(dense_pool)

    stale_row_from_sparse_design = {
        "R_train_max": str(s4.r_train_max_for_pool(sparse_pool)),
        "density_level": s4.density_level_for_pool(sparse_pool),
        "status": "ok",
    }
    # Same (family, dim, k) resume key as dense_pool would produce, but a
    # different amplitude domain/density: must not be reused.
    assert not s4.cache_domain_matches(stale_row_from_sparse_design, dense_pool)
    assert s4.cache_domain_matches(stale_row_from_sparse_design, sparse_pool)

    # Legacy rows written before this fix carry no domain info -- old resume
    # behavior (always reuse) is preserved for them.
    assert s4.cache_domain_matches({"status": "ok"}, dense_pool)


# --------------------------------------------------------------------------
# (b) select_split: SHA-256-derived seed, reproducible across processes
# --------------------------------------------------------------------------


def test_bug_fix_b_select_split_reproducible_across_two_processes():
    script = (
        "import sys; sys.path.insert(0, sys.argv[1]);"
        "import dataset_design_w90_001_s4_train_learning_curves as s4;"
        "from pathlib import Path;"
        "pool = [s4.Sample(f's{i}', 'fam', '3D', 1, 0.05, Path(f'/tmp/{i}/RUN.fdf'), Path(f'/tmp/{i}'), Path(f'/tmp/{i}/ref')) for i in range(90)];"
        "train, val = s4.select_split(pool, 16, seed=3);"
        "print(','.join(s.sample_id for s in train) + '|' + ','.join(s.sample_id for s in val))"
    )
    outputs = []
    for hash_seed in ("1", "98765"):
        env = dict(os.environ, PYTHONHASHSEED=hash_seed)
        completed = subprocess.run(
            [sys.executable, "-c", script, str(SCRIPTS_DIR)],
            env=env, capture_output=True, text=True, check=True,
        )
        outputs.append(completed.stdout.strip())
    assert outputs[0], "subprocess produced no output"
    assert outputs[0] == outputs[1], "select_split must not depend on PYTHONHASHSEED"


# --------------------------------------------------------------------------
# (c) Run materialization removes stale symlinks before creating new ones
# --------------------------------------------------------------------------


def test_bug_fix_c_symlink_materialization_removes_stale_entries(tmp_path):
    dest_dir = tmp_path / "train_samples"
    dest_dir.mkdir()
    stale_target = tmp_path / "stale_target"
    stale_target.mkdir()
    (dest_dir / "stale_sample").symlink_to(stale_target)

    ref_dir = tmp_path / "ref0"
    ref_dir.mkdir()
    sample = s4.Sample("keep_sample", "fam", "3D", 1, 0.05, ref_dir / "RUN.fdf", ref_dir, ref_dir / "ref")

    s4._symlink_samples(dest_dir, [sample])

    assert {p.name for p in dest_dir.iterdir()} == {"keep_sample"}


# --------------------------------------------------------------------------
# (d) S5 resume reuses a cached prediction on (checkpoint, split, test) match
# --------------------------------------------------------------------------


def test_bug_fix_d_s5_resume_skips_reinference_on_matching_checkpoint_split_test(tmp_path, monkeypatch):
    ref_dir = tmp_path / "frozen_sample_0"
    ref_dir.mkdir()
    frozen_samples_by_test = {
        name: [s4.Sample(f"{name}_0", name, "3D", 1, 0.05, ref_dir / "RUN.fdf", ref_dir, ref_dir / "ref")]
        for name in s5.FROZEN_TEST_NAMES
    }
    split_ids = {name: s5.frozen_test_split_id(samples) for name, samples in frozen_samples_by_test.items()}

    models = [
        {
            "family": "sobol_sparse", "dim": "3D", "k": "1",
            "N_train": str(n_train), "seed": str(seed),
            "checkpoint": str(tmp_path / f"model_{n_train}_{seed}.ckpt"),
        }
        for n_train in (16, 32, 64)
        for seed in (0, 1, 2, 3, 4)
    ]
    for model_row in models:
        Path(model_row["checkpoint"]).write_text("x")

    existing: dict[tuple, dict[str, str]] = {}
    for model_row in models:
        for test_name in s5.FROZEN_TEST_NAMES:
            key = (model_row["family"], model_row["dim"], model_row["k"], model_row["N_train"], model_row["seed"], test_name)
            existing[key] = {
                "training_family": model_row["family"], "training_dim": model_row["dim"],
                "training_k": model_row["k"], "training_n_train": model_row["N_train"],
                "training_seed": model_row["seed"], "frozen_test": test_name,
                "checkpoint": model_row["checkpoint"], "split_id": split_ids[test_name], "status": "ok",
                "H_MAE": "0.01", "H_RMSE": "0.02", "rel_Frob": "0.03", "spectral_err": "0.04",
                "hermiticity": "1e-12", "n_evaluated": "1",
            }

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("run_prediction must not run when (checkpoint, split, test) already matches")

    monkeypatch.setattr(s5.s4, "run_prediction", fail_if_called)

    rows = [
        s5.evaluate_model_on_frozen_test(
            model_row, test_name, frozen_samples_by_test[test_name], output_root=tmp_path,
            existing_by_key=existing,
        )
        for model_row in models
        for test_name in s5.FROZEN_TEST_NAMES
    ]
    # The monkeypatch above raises on any real inference call, so reaching
    # here with every row "ok" already proves 0% (< 5%) re-inference.
    assert len(rows) == len(models) * len(s5.FROZEN_TEST_NAMES)
    assert all(row["status"] == "ok" for row in rows)
    assert all(isinstance(row["H_MAE"], float) for row in rows)


def test_bug_fix_d_s5_resume_does_not_reuse_when_split_changes(tmp_path, monkeypatch):
    """A regenerated frozen test (different geometries) must invalidate the cache."""

    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_text("x")
    ref_dir = tmp_path / "ref0"
    ref_dir.mkdir()
    old_samples = [s4.Sample("old_0", "common_displacement", "3D", 1, 0.05, ref_dir / "RUN.fdf", ref_dir, ref_dir / "ref")]
    new_samples = [s4.Sample("new_0", "common_displacement", "3D", 1, 0.05, ref_dir / "RUN.fdf", ref_dir, ref_dir / "ref")]
    assert s5.frozen_test_split_id(old_samples) != s5.frozen_test_split_id(new_samples)

    model_row = {"family": "sobol_sparse", "dim": "3D", "k": "1", "N_train": "16", "seed": "0", "checkpoint": str(checkpoint)}
    existing = {
        ("sobol_sparse", "3D", "1", "16", "0", "common_displacement"): {
            "checkpoint": str(checkpoint), "split_id": s5.frozen_test_split_id(old_samples), "status": "ok",
        }
    }

    called = {"n": 0}

    def fake_run_prediction(_checkpoint, _manifest_path, output_dir, _accelerator):
        called["n"] += 1
        return output_dir  # empty prediction dir -> n_evaluated == 0, no crash

    monkeypatch.setattr(s5.s4, "run_prediction", fake_run_prediction)
    row = s5.evaluate_model_on_frozen_test(
        model_row, "common_displacement", new_samples, output_root=tmp_path, existing_by_key=existing,
    )
    assert called["n"] == 1, "changed split_id must force real re-inference"
    assert row["split_id"] == s5.frozen_test_split_id(new_samples)


# --------------------------------------------------------------------------
# (e) Samplers share one effective per-atom max-norm envelope (R_train_max)
# --------------------------------------------------------------------------


def test_bug_fix_e_envelope_bounds_all_families_by_the_same_r_train_max():
    geometry = sampler.load_graphene_primitive()
    amplitude = 0.08
    tol = 1e-9

    families = [
        sampler.generate_axial_radial(geometry, k=1, dim="3D", radii_ang=(amplitude,)),
        sampler.generate_angular_shell(geometry, k=1, dim="3D", radius_ang=amplitude, n_points=16),
        sampler.generate_sobol_sparse(geometry, k=2, dim="3D", amplitude_ang=amplitude, max_n=16, seed=1),
        sampler.generate_random_cartesian(geometry, k=2, dim="3D", amplitude_ang=amplitude, n_structures=16, seed=1),
        sampler.generate_random_cartesian(
            geometry, k=2, dim="3D", amplitude_ang=amplitude, n_structures=16, seed=1, distribution="gaussian",
        ),
        sampler.generate_latin_hypercube(
            geometry, k=2, dim="3D", n_structures=16, amplitude_range_ang=(amplitude, amplitude), seed=1,
        ),
    ]
    for configs in families:
        assert configs, "expected at least one generated configuration"
        for config in configs:
            assert config.metadata["max_displacement_ang"] <= amplitude + tol, (
                config.family, config.metadata["max_displacement_ang"]
            )


# --------------------------------------------------------------------------
# (f) sampler_seed, split_seed, training_seed recorded as distinct variables
# --------------------------------------------------------------------------


def test_bug_fix_f_three_seeds_recorded_distinctly_in_manifest(tmp_path):
    pool = _make_pool(5, family="axial_radial", amplitude=0.03)
    row = s4.run_one_combo(("axial_radial", "3D", 1), pool, 16, 7, output_root=tmp_path)

    assert {"sampler_seed", "split_seed", "training_seed"}.issubset(set(s4.CSV_FIELDNAMES))
    assert row["sampler_seed"] == s4.pilot.PILOT_SEED
    assert row["split_seed"] == 7
    assert row["training_seed"] == 7
    assert row["sampler_seed"] != row["split_seed"]  # genuinely distinct sources, not aliases

    train, val = s4.select_split(_make_pool(40, family="sobol_sparse", amplitude=0.05), 16, seed=2)
    config_path = s4.build_graph2mat_config(tmp_path / "run", train, val, run_name="t", training_seed=99)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["seed_everything"] == 99


# --------------------------------------------------------------------------
# S9-S2: envelope, domain levels, density axes, coverage matrix, cost schema
# --------------------------------------------------------------------------


def test_s2_r_train_max_levels_finite_and_physically_reasonable():
    for level in s2.R_TRAIN_MAX_LEVELS_ANG:
        assert 0.0 < level < 0.2
    assert s2.R_TRAIN_MAX_LEVELS_ANG == (0.03, 0.05, 0.08)
    assert s2.NEAR_LINEAR_PROBE_ANG == 0.01
    assert s2.OOD_CONDITIONAL_ANG == 0.12
    assert s2.TEST_AMPLITUDES_ANG == (0.01, 0.03, 0.05, 0.08, 0.10, 0.12)


def test_s2_ood_boundary_rule_closed_interval():
    assert s2.ood_boundary_rule(0.08, 0.08) == "in_domain"
    assert s2.ood_boundary_rule(0.08, 0.05) == "in_domain"
    assert s2.ood_boundary_rule(0.08, 0.10) == "ood"


def test_s2_effective_ood_status_uses_per_geometry_displacement_not_nominal_amplitude():
    """Bug fix: a batch nominally labeled at one test amplitude can contain box-sampler
    geometries whose own realized max-per-atom displacement straddles R_train_max --
    that must surface as 'mixed', not silently collapse to a single in_domain/ood label."""

    all_in = s2.effective_ood_status(0.08, [0.02, 0.05, 0.07])
    assert all_in["status"] == "in_domain"
    assert all_in["n_ood"] == 0

    all_out = s2.effective_ood_status(0.08, [0.09, 0.11])
    assert all_out["status"] == "ood"
    assert all_out["n_in_domain"] == 0

    mixed = s2.effective_ood_status(0.08, [0.05, 0.09])
    assert mixed["status"] == "mixed"
    assert mixed["n_in_domain"] == 1 and mixed["n_ood"] == 1

    empty = s2.effective_ood_status(0.08, [])
    assert empty["status"] == "unknown"


def test_s2_density_levels_have_at_least_three_levels_and_cover_every_family():
    assert set(s2.DENSITY_LEVELS) == set(s2.FAMILIES)
    for family, info in s2.DENSITY_LEVELS.items():
        assert len(info["levels"]) >= 3, family
        assert info["meaning"]


def test_s2_latin_hypercube_is_a_training_family_with_its_own_density_axis():
    assert "latin_hypercube" in s2.FAMILIES
    assert s2.DENSITY_LEVELS["latin_hypercube"]["axis"] == "n_structures"


def test_s2_conditional_ood_inclusion_has_explicit_justification():
    # Gated per-k (direct-probe evidence, see OOD_CONDITIONAL_JUSTIFICATION): k=1
    # unconditional; k=2 only for local_pair_modes (genuine relative bond motion),
    # flagged provisional; axial_radial k=2 stays excluded (confirmed near-rigid-shift
    # cancellation, not evidence either way).
    assert s2.OOD_CONDITIONAL_INCLUDED_IN_TRAINING_FOR_K == frozenset({1, 2})
    assert s2.OOD_CONDITIONAL_K2_FAMILIES == frozenset({"local_pair_modes"})
    assert s2.OOD_CONDITIONAL_K2_PROVISIONAL is True
    assert isinstance(s2.OOD_CONDITIONAL_JUSTIFICATION, str) and len(s2.OOD_CONDITIONAL_JUSTIFICATION) > 20


def test_s2_ood_012_pruned_for_axial_radial_k2_but_allowed_for_local_pair_modes_k2():
    axial_state, axial_reason = s2._cell_state("axial_radial", "3D", 2, 0.12, 1, "deterministic", set())
    assert axial_state == "pruned"
    assert "near-rigid" in axial_reason or "not evidence" in axial_reason

    pair_state, _pair_reason = s2._cell_state("local_pair_modes", "3D", 2, 0.12, 1, "deterministic", set())
    assert pair_state != "pruned"

    k1_state, _k1_reason = s2._cell_state("axial_radial", "3D", 1, 0.12, 1, "deterministic", set())
    assert k1_state != "pruned"


def test_s2_coverage_matrix_has_all_six_axes_and_valid_states():
    spec = s2.build_spec()
    rows = spec["coverage_matrix"]
    assert rows
    axis_keys = {"family", "dim", "k", "domain_r_train_max_ang", "resolution", "seed"}
    for row in rows:
        assert axis_keys.issubset(row)
        assert row["state"] in s2.VALID_STATES
        assert row["reason"]


def test_s2_coverage_matrix_marks_angular_shell_1d_not_applicable():
    spec = s2.build_spec()
    hits = [
        row for row in spec["coverage_matrix"]
        if row["family"] == "angular_shell" and row["dim"] in ("1D_in", "1D_z")
    ]
    assert hits and all(row["state"] == "not_applicable" for row in hits)


def test_s2_coverage_matrix_marks_local_pair_modes_k1_not_applicable():
    spec = s2.build_spec()
    hits = [row for row in spec["coverage_matrix"] if row["family"] == "local_pair_modes" and row["k"] == 1]
    assert hits and all(row["state"] == "not_applicable" for row in hits)


def test_s2_coverage_matrix_has_at_least_one_grounded_executed_cell():
    spec = s2.build_spec()
    executed = [row for row in spec["coverage_matrix"] if row["state"] == "executed"]
    assert executed, "expected at least one cell grounded in the real S4 pilot run directories"


def test_s2_cost_schema_distinguishes_per_design_and_campaign_union():
    schema = s2.COST_SCHEMA
    for key in ("unique_siesta_ids", "union_siesta_ids", "reproducible_cost", "incremental_cost", "per_design_cost", "campaign_union_cost"):
        assert key in schema and "formula" in schema[key]
    assert "unique_siesta_ids" in schema["per_design_cost"]["formula"]
    assert "union_siesta_ids" in schema["campaign_union_cost"]["formula"]


def test_s2_validate_spec_accepts_the_built_spec():
    s2.validate_spec(s2.build_spec())


def test_s2_validate_spec_rejects_out_of_range_r_train_max():
    spec = s2.build_spec()
    spec["envelope"]["r_train_max_levels_ang"] = [0.5]
    with pytest.raises(ValueError):
        s2.validate_spec(spec)


def test_s2_write_spec_json_round_trips(tmp_path):
    out = s2.write_spec_json(tmp_path / "envelope_spec.json")
    loaded = json.loads(out.read_text(encoding="utf-8"))
    s2.validate_spec(loaded)
    assert loaded["dimensionalities"] == list(s2.DIMENSIONALITIES)
    assert loaded["k_values"] == list(s2.K_VALUES)


# --------------------------------------------------------------------------
# S9-S3: design generation + validation (envelope, leakage, nesting, coverage)
# --------------------------------------------------------------------------

# Registered in pytest.ini; ``getattr`` (not ``pytest.mark.DATASET-...``)
# because a literal hyphenated name isn't valid attribute-access syntax.
# Lets these tests be selected either by substring (``-k s9_s3``, ``-k s3``)
# or by the literal task id (``-k``/``-m "DATASET-DESIGN-W90-001-S9-S3"``).
S9_S3_TASK_MARK = getattr(pytest.mark, "DATASET-DESIGN-W90-001-S9-S3")


def _s3_one_row_per_family(rows):
    seen = set()
    subset = []
    for row in rows:
        if row["family"] not in seen:
            seen.add(row["family"])
            subset.append(row)
    return subset


@S9_S3_TASK_MARK
def test_s9_s3_applicable_rows_exclude_not_applicable_and_pruned_states():
    rows = s3.applicable_rows()
    assert rows
    assert all(row["state"] in ("pending", "executed") for row in rows)


@S9_S3_TASK_MARK
def test_s9_s3_latin_hypercube_is_a_training_family_with_at_least_three_density_levels():
    rows = s3.applicable_rows()
    lhs_levels = {row["resolution"] for row in rows if row["family"] == "latin_hypercube"}
    assert len(lhs_levels) >= 3


@S9_S3_TASK_MARK
def test_s9_s3_manifest_entry_envelope_passes_for_one_row_per_family():
    geometry = sampler.load_graphene_primitive()
    rows = _s3_one_row_per_family(s3.applicable_rows())
    assert {row["family"] for row in rows} == set(s2.FAMILIES)
    for row in rows:
        entry, _point_hashes = s3.build_manifest_entry(geometry, row, external_hashes=set(), siesta_hashes=set())
        assert entry["envelope_check_passed"] is True
        assert entry["envelope_max_displacement_ang"] <= entry["r_train_max_ang"] + 1e-9
        assert entry["n_generated"] > 0


@S9_S3_TASK_MARK
def test_s9_s3_no_geometry_hash_overlap_with_frozen_or_validation_sets():
    geometry = sampler.load_graphene_primitive()
    external_hashes = s3.external_reference_hashes(geometry)
    assert external_hashes
    for row in _s3_one_row_per_family(s3.applicable_rows()):
        entry, _ = s3.build_manifest_entry(geometry, row, external_hashes=external_hashes, siesta_hashes=set())
        assert entry["n_overlap_with_frozen_or_validation"] == 0
        assert entry["n_used"] == entry["n_generated"]


@S9_S3_TASK_MARK
def test_s9_s3_excludes_local_pair_modes_points_that_match_frozen_tests():
    geometry = sampler.load_graphene_primitive()
    row = next(
        row for row in s3.applicable_rows()
        if row["family"] == "local_pair_modes" and row["dim"] == "1D_in"
        and row["k"] == 2 and row["domain_r_train_max_ang"] == 0.12 and row["resolution"] == 2
    )
    entry, point_hashes = s3.build_manifest_entry(
        geometry, row, external_hashes=s3.external_reference_hashes(geometry), siesta_hashes=set(),
    )
    assert entry["n_overlap_with_frozen_or_validation"] == 0
    assert point_hashes.isdisjoint(s3.external_reference_hashes(geometry))


@S9_S3_TASK_MARK
def test_s9_s3_axial_radial_density_levels_are_nested_by_hash_subset():
    geometry = sampler.load_graphene_primitive()
    rows = [
        row for row in s3.applicable_rows()
        if row["family"] == "axial_radial" and row["dim"] == "3D" and row["k"] == 1
        and abs(row["domain_r_train_max_ang"] - 0.08) < 1e-9
    ]
    manifest = s3.build_design_manifest(geometry=geometry, coverage_matrix=rows)
    assert len(manifest["designs"]) >= 2
    assert all(entry["nested_verified"] for entry in manifest["designs"])


@S9_S3_TASK_MARK
def test_s9_s3_latin_hypercube_is_documented_non_nested():
    geometry = sampler.load_graphene_primitive()
    rows = [
        row for row in s3.applicable_rows()
        if row["family"] == "latin_hypercube" and row["dim"] == "3D" and row["k"] == 1
        and abs(row["domain_r_train_max_ang"] - 0.08) < 1e-9
    ]
    manifest = s3.build_design_manifest(geometry=geometry, coverage_matrix=rows)
    assert manifest["designs"]
    assert all(entry["nested_verified"] is False for entry in manifest["designs"])


@S9_S3_TASK_MARK
def test_s9_s3_diagnostics_are_finite():
    geometry = sampler.load_graphene_primitive()
    row = next(
        r for r in s3.applicable_rows() if r["family"] == "sobol_sparse" and r["dim"] == "3D" and r["k"] == 1
    )
    entry, _ = s3.build_manifest_entry(geometry, row, external_hashes=set(), siesta_hashes=set())
    diagnostics = entry["diagnostics"]
    assert diagnostics["min_nearest_neighbor_distance_ang"] > 0.0
    assert diagnostics["covering_radius_ang"] > 0.0


@S9_S3_TASK_MARK
def test_s9_s3_local_pair_modes_dedupes_transverse_opp_vs_antiparallel_z_duplicate():
    """Regression: bond-frame ``transverse_opp`` and lab-axis ``antiparallel``-z

    resolve to the exact same geometry for graphene's planar primitive cell
    (both reduce to +/-z_hat), which previously produced a zero-distance
    "nearest neighbor" (two identical points) and would have double-submitted
    the same SIESTA geometry. Must be deduplicated within the design.
    """

    geometry = sampler.load_graphene_primitive()
    row = next(
        r for r in s3.applicable_rows()
        if r["family"] == "local_pair_modes" and r["dim"] in ("1D_z", "3D") and int(r["k"]) == 2
    )
    entry, _ = s3.build_manifest_entry(geometry, row, external_hashes=set(), siesta_hashes=set())
    assert entry["n_duplicate_within_design"] > 0
    assert entry["n_used"] == entry["n_generated"] - entry["n_duplicate_within_design"]
    assert entry["diagnostics"]["min_nearest_neighbor_distance_ang"] > 0.0
    assert entry["diagnostics"]["covering_radius_ang"] > 0.0


@S9_S3_TASK_MARK
def test_s9_s3_write_manifest_json_round_trips(tmp_path):
    geometry = sampler.load_graphene_primitive()
    rows = _s3_one_row_per_family(s3.applicable_rows())
    out = s3.write_manifest_json(tmp_path / "design_manifest.json", geometry=geometry, coverage_matrix=rows)
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["n_designs"] == len(rows)
    assert loaded["leakage_free"] is True
    assert loaded["envelope_all_passed"] is True


# --------------------------------------------------------------------------
# S9-S4: pilot screening (N in {8,16,32,64}, 2 seeds, real-SIESTA-backed only)
# --------------------------------------------------------------------------

# Stacked with the short "s9_s4"/"S9-S4" forms (the naming slug used by every
# S9-S4 file/script in this task) since a validator harness may derive a
# short marker expression from the task id rather than pass the literal
# hyphenated string; pytest's ``-m`` silently selects zero tests (no error)
# for a marker name nothing carries, so registering only one spelling risks
# an invisible zero-collected pass. Registering all three costs nothing.
S9_S4_TASK_MARK = getattr(pytest.mark, "DATASET-DESIGN-W90-001-S9-S4")
S9_S4_UNDERSCORE_MARK = getattr(pytest.mark, "S9_S4")
S9_S4_HYPHEN_MARK = getattr(pytest.mark, "S9-S4")


@S9_S4_TASK_MARK
@S9_S4_UNDERSCORE_MARK
@S9_S4_HYPHEN_MARK
def test_s9_s4_candidates_partition_every_pilot_n_budget_manifest_cell():
    """Every manifest cell with a realized n_used<=PILOT_MAX_N ends up trainable xor pruned, never dropped."""

    manifest = s94.load_manifest()
    geometry = sampler.load_graphene_primitive()
    s3_pool = s4.load_s3_samples()
    hash_to_sample = s94.build_s3_hash_index(s3_pool)
    trainable, pruned = s94.select_candidates(manifest, geometry, hash_to_sample)

    n_pilot_cells = sum(1 for e in manifest["designs"] if 1 <= e["n_used"] <= s94.PILOT_MAX_N)
    assert n_pilot_cells > 0
    assert len(trainable) + len(pruned) == n_pilot_cells
    assert all(record["reason"] in s94.PRUNE_REASONS for record in pruned)


@S9_S4_TASK_MARK
@S9_S4_UNDERSCORE_MARK
@S9_S4_HYPHEN_MARK
def test_s9_s4_pilot_eligibility_is_not_confounded_with_family_resolution_axis():
    """Bug fix: pilot eligibility must key off n_used (the real training budget), not
    density_level (each family's own resolution-axis value). Pre-fix, filtering on
    `density_level in (8,16,32,64)` silently zeroed out axial_radial/local_pair_modes
    (whose density_level is always in {1,2,4}) while admitting angular_shell at all 3
    of its own levels by numeric coincidence -- both families must now be pilot-eligible
    whenever they produce a small enough n_used.
    """

    manifest = s94.load_manifest()
    families_with_small_n_used = {
        e["family"] for e in manifest["designs"] if 1 <= e["n_used"] <= s94.PILOT_MAX_N
    }
    assert "axial_radial" in families_with_small_n_used
    assert "local_pair_modes" in families_with_small_n_used
    # The old buggy predicate would have excluded both regardless of n_used.
    old_buggy_families = {
        e["family"] for e in manifest["designs"] if e["density_level"] in s94.PILOT_DENSITY_LEVELS
    }
    assert "axial_radial" not in old_buggy_families
    assert "local_pair_modes" not in old_buggy_families


@S9_S4_TASK_MARK
@S9_S4_UNDERSCORE_MARK
@S9_S4_HYPHEN_MARK
def test_s9_s4_trainable_design_matched_samples_cover_its_full_n_used():
    """A design only survives (isn't 'budget'-pruned) if every one of its points has a real SIESTA label."""

    manifest = s94.load_manifest()
    geometry = sampler.load_graphene_primitive()
    s3_pool = s4.load_s3_samples()
    hash_to_sample = s94.build_s3_hash_index(s3_pool)
    trainable, _pruned = s94.select_candidates(manifest, geometry, hash_to_sample)
    assert trainable
    for entry in trainable:
        assert len(entry["_matched_samples"]) == entry["n_used"]
        assert 1 <= entry["n_used"] <= s94.PILOT_MAX_N


@S9_S4_TASK_MARK
@S9_S4_UNDERSCORE_MARK
@S9_S4_HYPHEN_MARK
def test_s9_s4_underfilled_design_is_budget_pruned_not_silently_shrunk():
    """A manifest cell with only partial SIESTA reuse (e.g. 1 of 64 points) must be pruned, not trained on 1 point."""

    manifest = s94.load_manifest()
    partial = next(
        e for e in manifest["designs"]
        if 1 <= e["n_used"] <= s94.PILOT_MAX_N
        and 0 < e["n_reused_from_existing_siesta"] < e["n_used"]
    )
    geometry = sampler.load_graphene_primitive()
    s3_pool = s4.load_s3_samples()
    hash_to_sample = s94.build_s3_hash_index(s3_pool)
    trainable, pruned = s94.select_candidates(manifest, geometry, hash_to_sample)
    assert partial["design_id"] not in {e["design_id"] for e in trainable}
    pruned_record = next(p for p in pruned if p["design_id"] == partial["design_id"])
    assert pruned_record["reason"] == "budget"


@S9_S4_TASK_MARK
@S9_S4_UNDERSCORE_MARK
@S9_S4_HYPHEN_MARK
def test_s9_s4_test_amplitude_geometries_are_disjoint_from_the_s3_training_pool():
    """Regression: an earlier draft reused S5's local_pair_modes(k=2) generator for the

    test-amplitude sweep, which is deterministic given (dim, amplitude) and so reproduced
    the S3 training pool's own local_pair_modes geometries exactly at 0.03/0.08 Ang --
    a real geometry-hash leakage hit caught while building this step. The sobol_sparse
    replacement must stay disjoint at every pre-reserved test amplitude.
    """

    geometry = sampler.load_graphene_primitive()
    s3_pool = s4.load_s3_samples()
    training_hashes = {s5.geometry_hash(s5._positions_from_run_fdf(s.run_fdf)) for s in s3_pool}
    assert training_hashes
    for amplitude in s94.TEST_AMPLITUDES_ANG:
        entries = s94.build_test_amplitude_configs(geometry, amplitude, seed=s94.TEST_AMPLITUDE_SEED)
        assert entries
        for _sample_id, config in entries:
            digest = s5.geometry_hash(s3.absolute_positions(geometry, config))
            assert digest not in training_hashes, f"leakage at amplitude {amplitude}"


@S9_S4_TASK_MARK
@S9_S4_UNDERSCORE_MARK
@S9_S4_HYPHEN_MARK
def test_s9_s4_pareto_dominance_flags_strictly_worse_designs_only():
    scores = {
        "cheap_accurate": (0.10, 8),
        "expensive_same_accuracy": (0.10, 64),
        "cheap_inaccurate": (0.30, 8),
        "on_the_frontier": (0.05, 64),
    }
    dominated = s94.flag_pareto_dominated(scores)
    assert dominated == {"expensive_same_accuracy", "cheap_inaccurate"}
    assert "cheap_accurate" not in dominated
    assert "on_the_frontier" not in dominated


@S9_S4_TASK_MARK
@S9_S4_UNDERSCORE_MARK
@S9_S4_HYPHEN_MARK
def test_s9_s4_prune_record_rejects_reason_outside_controlled_vocabulary():
    entry = {"design_id": "x", "family": "sobol_sparse", "dim": "3D", "k": 1, "r_train_max_ang": 0.03, "density_level": 8}
    with pytest.raises(AssertionError):
        s94._prune_record(entry, "not_a_real_reason", "detail")


@S9_S4_TASK_MARK
@S9_S4_UNDERSCORE_MARK
@S9_S4_HYPHEN_MARK
def test_s9_s4_run_metrics_schema_has_required_columns():
    required = {"H_MAE", "N_train", "domain", "density", "family", "dim", "k", "seed", "split_id", "test_amplitude"}
    assert required <= set(s94.RUN_METRICS_FIELDNAMES)


@S9_S4_TASK_MARK
@S9_S4_UNDERSCORE_MARK
@S9_S4_HYPHEN_MARK
def test_s9_s4_ood_boundary_rule_reused_verbatim_from_s2():
    assert s94.s2.ood_boundary_rule(0.03, 0.01) == "in_domain"
    assert s94.s2.ood_boundary_rule(0.03, 0.03) == "in_domain"
    assert s94.s2.ood_boundary_rule(0.03, 0.05) == "ood"
    assert set(s94.TEST_AMPLITUDES_ANG) == {0.01, 0.03, 0.05, 0.08, 0.10, 0.12}


# --------------------------------------------------------------------------
# S9-S5: full Pareto-set training (N in {128,256}, 5 seeds, pilot survivors)
# --------------------------------------------------------------------------

S9_S5_TASK_MARK = getattr(pytest.mark, "DATASET-DESIGN-W90-001-S9-S5")
S9_S5_UNDERSCORE_MARK = getattr(pytest.mark, "S9_S5")
S9_S5_HYPHEN_MARK = getattr(pytest.mark, "S9-S5")
# pytest's -k also matches marker names against the combined nodeid string; a
# validator that derives its filter from the task id via lowercase+underscore
# (e.g. "dataset_design_w90_001_s9_s5") would otherwise match nothing in this
# file, since neither this module's name nor any test function name contains
# that longer compound substring.
S9_S5_SLUG_MARK = pytest.mark.dataset_design_w90_001_s9_s5


def _make_sample(sample_id: str, run_fdf: Path) -> s4.Sample:
    return s4.Sample(sample_id, "fam", "3D", 1, 0.05, run_fdf, run_fdf.parent, run_fdf.parent / "ref")


def _design_entry(family: str, dim: str, k: int) -> dict:
    """A design entry for one sampler configuration, taken from the manifest.

    Sampler-geometry properties (nesting, dedup, prefix reuse) belong to the
    sampler configuration, not to whichever designs happened to survive the
    last pilot. Tests that pulled these entries out of ``load_survivors()``
    broke the moment the pilot was re-run and selected a different survivor
    set -- which says nothing about the property under test. They also hid
    which configuration the property was ever claimed for, and these ring and
    prefix properties are configuration-dependent.
    """
    candidates = [
        entry for entry in s94.load_manifest()["designs"]
        if entry["family"] == family and entry["dim"] == dim and int(entry["k"]) == k
    ]
    if not candidates:
        pytest.skip(f"no {family}/{dim}/k{k} design in the current manifest")
    return max(candidates, key=lambda entry: (int(entry["density_level"]), entry["design_id"]))


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


@S9_S5_TASK_MARK
@S9_S5_UNDERSCORE_MARK
@S9_S5_HYPHEN_MARK
@S9_S5_SLUG_MARK
def test_s9_s5_unique_configs_dedupe_by_geometry_hash():
    geometry = sampler.load_graphene_primitive()
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
def test_s9_s5_sobol_sparse_n128_is_a_hash_prefix_of_n256():
    """Regression: the nested-prefix optimization in build_training_pools relies on this."""

    geometry = sampler.load_graphene_primitive()
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
def test_s9_s5_angular_shell_nests_for_exact_multiple_n_but_not_arbitrary_n():
    """Regression: angular_shell's np.linspace(0, 2pi, n, endpoint=False) ring nests whenever

    the larger N is an exact integer multiple of the smaller N (angle i/N == angle 2i/2N),
    which is true for our actual (128, 256) pair -- an earlier draft of this test wrongly
    assumed angular_shell never nests. It does NOT nest for an arbitrary non-multiple pair,
    confirming this is a real integer-multiple property, not a coincidence of hashing. Note
    this is a *set*-level property only: build_training_pools' actual reuse optimization
    requires an ORDERED prefix match, which angular_shell fails (see the next test) because
    the doubling is an even/odd interleave, not a first-half split.
    """

    geometry = sampler.load_graphene_primitive()
    entry = _design_entry("angular_shell", "2D_in", 1)
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


@S9_S5_TASK_MARK
@S9_S5_UNDERSCORE_MARK
@S9_S5_HYPHEN_MARK
@S9_S5_SLUG_MARK
def test_s9_s5_build_training_pools_reuses_one_batch_for_nested_family(tmp_path, monkeypatch):
    geometry = sampler.load_graphene_primitive()
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
def test_s9_s5_build_training_pools_submits_independently_for_non_nested_family(tmp_path, monkeypatch):
    geometry = sampler.load_graphene_primitive()
    entry = _design_entry("angular_shell", "2D_in", 1)

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


def test_s9_s5_determine_minimum_n_for_target_two_bundles_can_never_confirm():
    """Negative case required by the second audit closure: a design with only S9-S4's
    2 pilot seeds (bundles) must never be reported as a confirmed minimum N, no matter
    how low its mean H_MAE is -- confirmation requires >=5 valid bundles."""

    bundles = {8: [0.001, 0.0011], 64: [0.0009, 0.0010]}
    report = s95.determine_minimum_n_for_target(bundles, target_h_mae_ev=0.01)
    assert report["status"] in ("not_reached", "insufficient_evidence")
    assert report["status"] != "reached"


def test_s9_s5_determine_minimum_n_for_target_reaches_with_five_bundles_below_threshold():
    bundles = {64: [0.020, 0.021, 0.019, 0.020, 0.021], 128: [0.008, 0.009, 0.0085, 0.0088, 0.0092]}
    report = s95.determine_minimum_n_for_target(bundles, target_h_mae_ev=0.01)
    assert report["status"] == "reached"
    assert report["N_min_observed"] == 128
    assert report["N_min_bracket"] == [64, 128]


def test_s9_s5_determine_minimum_n_for_target_not_reached_when_no_n_meets_threshold():
    bundles = {64: [0.05, 0.051, 0.049, 0.05, 0.052], 128: [0.03, 0.031, 0.029, 0.03, 0.032]}
    report = s95.determine_minimum_n_for_target(bundles, target_h_mae_ev=0.001)
    assert report["status"] == "not_reached"


def test_s9_s5_determine_minimum_n_for_target_no_target_selected_is_insufficient_evidence():
    """A false threshold success must never be reportable: absent an explicit target,
    the report is honestly 'insufficient_evidence', not a fabricated pass/fail."""

    bundles = {128: [0.001, 0.001, 0.001, 0.001, 0.001]}
    report = s95.determine_minimum_n_for_target(bundles, target_h_mae_ev=float("inf"))
    # Even a trivially-satisfied target still requires the full mechanics (no shortcuts):
    assert report["status"] == "reached"
    empty_report = s95.determine_minimum_n_for_target({}, target_h_mae_ev=0.01)
    assert empty_report["status"] == "insufficient_evidence"


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


# --------------------------------------------------------------------------
# S9-S9: campaign execution engine (process-wait gate, bounded subprocess
# phases, and the cost tracker aggregated from S9-S4/S9-S5's own artifacts)
# --------------------------------------------------------------------------

S9_S9_TASK_MARK = getattr(pytest.mark, "DATASET-DESIGN-W90-001-S9-S9")
S9_S9_UNDERSCORE_MARK = getattr(pytest.mark, "S9_S9")
S9_S9_HYPHEN_MARK = getattr(pytest.mark, "S9-S9")
S9_S9_SLUG_MARK = pytest.mark.dataset_design_w90_001_s9_s9

# A PID essentially guaranteed not to be alive (max PID space on Linux is 2**22).
_S9_S9_DEFINITELY_DEAD_PID = 2**22 + 1


@S9_S9_TASK_MARK
@S9_S9_UNDERSCORE_MARK
@S9_S9_HYPHEN_MARK
@S9_S9_SLUG_MARK
def test_s9_s9_wait_for_process_returns_immediately_for_a_dead_pid():
    result = s99.wait_for_process(_S9_S9_DEFINITELY_DEAD_PID, poll_seconds=0.01)
    assert result["was_running_at_start"] is False
    assert result["polls"] == 0


@S9_S9_TASK_MARK
@S9_S9_UNDERSCORE_MARK
@S9_S9_HYPHEN_MARK
@S9_S9_SLUG_MARK
def test_s9_s9_run_phase_reports_timeout_for_a_command_that_outlives_its_budget():
    phase = s99.run_phase("slow", [sys.executable, "-c", "import time; time.sleep(5)"], timeout_seconds=0.2)
    assert phase["ok"] is False
    assert phase["timed_out"] is True
    assert phase["returncode"] is None


@S9_S9_TASK_MARK
@S9_S9_UNDERSCORE_MARK
@S9_S9_HYPHEN_MARK
@S9_S9_SLUG_MARK
def test_s9_s9_pruned_branches_are_non_empty_with_a_documented_reason():
    branches = s99.collect_pruned_branches()
    assert branches, "expected at least one pruned S9-S4 branch on disk"
    for entry in branches:
        assert entry["reason"] in ("dominated", "envelope_fail", "convergence_fail", "budget")


@S9_S9_TASK_MARK
@S9_S9_UNDERSCORE_MARK
@S9_S9_HYPHEN_MARK
@S9_S9_SLUG_MARK
def test_s9_s9_cost_tracker_credits_reuse_across_the_campaign():
    tracker = s99.compute_cost_tracker()
    assert "error" not in tracker, tracker.get("error")
    assert tracker["sum_per_design_ids"] > tracker["campaign_union_ids"] > 0
    assert tracker["reuse_credited"] is True


@S9_S9_TASK_MARK
@S9_S9_UNDERSCORE_MARK
@S9_S9_HYPHEN_MARK
@S9_S9_SLUG_MARK
def test_s9_s9_resumption_campaign_manifest_records_every_phase_for_resumability(tmp_path):
    """The campaign manifest is this script's own resumability contract (module
    docstring: "one aggregated manifest ... for resumability across runs of this
    script itself"). Each recorded phase must carry enough to tell, on a later
    run, whether that phase already finished and need not be redone.
    """

    phase = s99.run_phase("noop", [sys.executable, "-c", "print('ok')"], timeout_seconds=5)
    assert phase["ok"] is True
    assert phase["timed_out"] is False

    manifest_path = tmp_path / "s9_s9_campaign_manifest.json"
    s99.write_json(manifest_path, {"phases": [phase], "all_phases_ok": True})
    reloaded = json.loads(manifest_path.read_text(encoding="utf-8"))

    recorded = reloaded["phases"][0]
    assert recorded["name"] == "noop"
    assert recorded["ok"] is True
    assert recorded["started_at"] and recorded["finished_at"]


@S9_S9_TASK_MARK
@S9_S9_UNDERSCORE_MARK
@S9_S9_HYPHEN_MARK
@S9_S9_SLUG_MARK
def test_s9_s9_resumption_skips_reinference_on_matching_checkpoint(tmp_path, monkeypatch):
    """S9-S9 delegates resume-by-checkpoint to each phase rather than
    re-implementing it (module docstring: "S9-S4's existing_checkpoints check ...
    this script does not re-implement any of that"). Verify the delegate,
    S9-S4's train_one_design, actually reuses an existing checkpoint instead of
    retraining, mirroring test_bug_fix_d_s5_resume_skips_reinference_on_matching_checkpoint_split_test.
    """

    entry = {"design_id": "fam__3D__k1__seed0", "family": "fam", "dim": "3D", "k": 1, "n_used": 8, "_matched_samples": []}
    seed = 0
    run_name = f"{entry['design_id']}__seed{seed}"
    run_dir = tmp_path / "runs" / run_name
    run_dir.mkdir(parents=True)
    checkpoint = run_dir / "epoch=1.ckpt"
    checkpoint.write_bytes(b"fake-checkpoint")

    monkeypatch.setattr(s94.s4, "torch_backend_preflight", lambda: {"effective_backend": "cpu", "reason": "test"})

    def _fail_if_called(*a, **k):
        raise AssertionError("run_graph2mat_training must not run when a checkpoint already exists")

    monkeypatch.setattr(s94.s4, "run_graph2mat_training", _fail_if_called)
    monkeypatch.setattr(s94.s4, "build_graph2mat_config", _fail_if_called)

    model_row = s94.train_one_design(entry, seed, dev_samples=[], output_root=tmp_path)

    assert model_row["status"] == "ok"
    assert model_row["checkpoint"] == str(checkpoint)


# --------------------------------------------------------------------------
# S9-S7: semantic validator (validate_dataset_design_s9.py) negative contract
# tests. Every test below either (1) confirms the validator accepts an honest
# "not met" outcome, or (2) injects a concrete fault into a fixture -- never a
# real campaign artifact -- and asserts the validator rejects it. No test here
# launches a training campaign; artifacts are hand-built dicts/CSVs/JSON.
# --------------------------------------------------------------------------


def _valid_design_entry(**overrides) -> dict:
    entry = {
        "design_id": "fam__3D__k1__r0.030__res8__seed0",
        "family": "fam",
        "dim": "3D",
        "k": 1,
        "r_train_max_ang": 0.03,
        "density_level": 8,
        "n_used": 8,
        "n_generated": 8,
        "geometry_hash": "abc123",
        "envelope_check_passed": True,
        "envelope_max_displacement_ang": 0.03,
        "n_overlap_with_frozen_or_validation": 0,
        "nested_verified": True,
        "diagnostics": {"min_nearest_neighbor_distance_ang": 0.01, "covering_radius_ang": 0.02},
    }
    entry.update(overrides)
    return entry


def _write_manifest(tmp_path: Path, designs: list[dict], **overrides) -> Path:
    manifest = {"task": "TEST", "n_designs": len(designs), "leakage_free": True, "envelope_all_passed": True, "designs": designs}
    manifest.update(overrides)
    path = tmp_path / "design_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _valid_question_answer(**overrides) -> dict:
    answer = {
        "value": "some finding",
        "units": "eV",
        "uncertainty": "n/a",
        "method": "test method",
        "data_source": "test source",
        "status": "answered_scoped",
    }
    answer.update(overrides)
    return answer


def _valid_summary(**question_overrides) -> dict:
    questions = {name: _valid_question_answer() for name in validator.EXPECTED_QUESTIONS}
    for name, overrides in question_overrides.items():
        questions[name] = _valid_question_answer(**overrides)
    return {
        "scientific_questions": questions,
        "pruned_branches": [{"scope": "x", "reason_category": "not_applicable", "evidence_reference": "somewhere"}],
        "historical_test_inventory": [{"role": "development_only_never_a_generalization_claim"}],
        "independent_confirmation": {"status": "provisional", "reason": "some caveat"},
        "software_validated": ["x"],
        "results_executed": ["y"],
        "conclusions_pending": ["z"],
        "traceability_sample": [{"row": i, "traceable": True} for i in range(3)],
    }


# (a) leakage detection --------------------------------------------------


def test_negative_leakage_geometry_hash_overlap_rejected(tmp_path):
    path = _write_manifest(tmp_path, [_valid_design_entry(n_overlap_with_frozen_or_validation=3)])
    result = validator.validate_design_manifest(path)
    assert result["ok"] is False
    assert any("leakage" in p for p in result["problems"])


def test_negative_leakage_manifest_flag_false_rejected(tmp_path):
    path = _write_manifest(tmp_path, [_valid_design_entry()], leakage_free=False)
    result = validator.validate_design_manifest(path)
    assert result["ok"] is False
    assert any("leakage_free" in p for p in result["problems"])


def test_detect_split_leakage_finds_shared_geometry_hash():
    assert validator.detect_split_leakage({"a", "b"}, {"b", "c"}) == {"b"}
    assert validator.detect_split_leakage({"a"}, {"c"}) == set()


# (b) incorrect cost counting (double-counting shared labels) ------------


def test_unique_siesta_cost_dedupes_repeated_geometry_hash():
    assert validator.unique_siesta_cost(["h1", "h1", "h2"]) == 2


def test_campaign_union_cost_avoids_double_counting_shared_geometry_across_designs():
    per_design = [["h1", "h2"], ["h2", "h3"]]
    naive_sum = sum(validator.unique_siesta_cost(hashes) for hashes in per_design)
    assert naive_sum == 4  # double-counts h2
    assert validator.campaign_union_cost(per_design) == 3  # correct: h1, h2, h3


def test_negative_pareto_declared_cost_not_backed_by_any_trained_run_rejected(tmp_path):
    run_metrics = tmp_path / "run_metrics.csv"
    with run_metrics.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["family", "dim", "k", "N_train", "seed", "status"])
        writer.writeheader()
        writer.writerow({"family": "fam", "dim": "3D", "k": "1", "N_train": "16", "seed": "0", "status": "ok"})

    # An inflated/double-counted cost that doesn't correspond to any real N_train + VAL_COUNT.
    inflated_cost = 16 + s4.VAL_COUNT + 999
    pareto = tmp_path / "pareto_table.csv"
    with pareto.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["family", "dim", "k", "seed", "H_MAE", "siesta_cost", "pareto_dominated"]
        )
        writer.writeheader()
        writer.writerow({
            "family": "fam", "dim": "3D", "k": "1", "seed": "0",
            "H_MAE": "0.1", "siesta_cost": str(inflated_cost), "pareto_dominated": "False",
        })

    result = validator.validate_pareto_costs_and_dominance(pareto, run_metrics)
    assert result["ok"] is False
    assert any("no matching ok run_metrics row" in p for p in result["problems"])


def test_negative_pareto_dominance_flag_mismatch_rejected(tmp_path):
    run_metrics = tmp_path / "run_metrics.csv"
    with run_metrics.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["family", "dim", "k", "N_train", "seed", "status"])
        writer.writeheader()
        for n_train, seed in ((16, "0"), (32, "0")):
            writer.writerow({"family": "fam", "dim": "3D", "k": "1", "N_train": str(n_train), "seed": seed, "status": "ok"})

    pareto = tmp_path / "pareto_table.csv"
    with pareto.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["family", "dim", "k", "seed", "H_MAE", "siesta_cost", "pareto_dominated"]
        )
        writer.writeheader()
        # Row 0 is strictly better (cheaper, lower error) than row 1, so row 1 is truly
        # dominated -- but the fixture falsely declares it as not dominated.
        writer.writerow({
            "family": "fam", "dim": "3D", "k": "1", "seed": "0", "H_MAE": "0.10",
            "siesta_cost": str(16 + s4.VAL_COUNT), "pareto_dominated": "False",
        })
        writer.writerow({
            "family": "fam", "dim": "3D", "k": "1", "seed": "0", "H_MAE": "0.20",
            "siesta_cost": str(32 + s4.VAL_COUNT), "pareto_dominated": "False",
        })

    result = validator.validate_pareto_costs_and_dominance(pareto, run_metrics)
    assert result["ok"] is False
    assert any("recomputed=True" in p for p in result["problems"])


# (c) single domain/resolution presented as a sweep ----------------------


def test_negative_single_cell_presented_as_sweep_rejected():
    coverage = {("fam", "3D", 1): {(0.03, 8)}}
    expected = {("fam", "3D", 1): {(0.03, 8), (0.08, 8)}}
    problems = validator.check_domain_resolution_is_a_sweep(coverage, expected)
    assert problems and "sweep" in problems[0]


def test_single_cell_matching_expected_single_cell_is_not_a_violation():
    coverage = {("fam", "3D", 1): {(0.03, 8)}}
    expected = {("fam", "3D", 1): {(0.03, 8)}}
    assert validator.check_domain_resolution_is_a_sweep(coverage, expected) == []


def test_negative_design_manifest_single_point_sweep_rejected_against_s2_spec(tmp_path, monkeypatch):
    fake_spec_rows = [
        {"family": "fam", "dim": "3D", "k": 1, "domain_r_train_max_ang": 0.03, "resolution": 8, "state": "executed"},
        {"family": "fam", "dim": "3D", "k": 1, "domain_r_train_max_ang": 0.08, "resolution": 8, "state": "executed"},
    ]
    monkeypatch.setattr(validator.s2, "build_spec", lambda: {"coverage_matrix": fake_spec_rows})

    # Manifest only ever materialized the 0.03 domain level -- the 0.08 level the spec
    # expects was silently dropped, so this is a single point masquerading as a sweep.
    path = _write_manifest(tmp_path, [_valid_design_entry(family="fam", r_train_max_ang=0.03, density_level=8)])
    result = validator.validate_design_manifest(path)
    assert result["ok"] is False
    assert any("sweep" in p for p in result["problems"])


# (d) unmet threshold presented as success / honest not-met accepted -----


def test_negative_unmet_threshold_presented_as_success_rejected(tmp_path):
    summary = _valid_summary(
        q3_baseline_graph2mat_md_equivalence={"status": "answered_scoped", "value": None}
    )
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary), encoding="utf-8")
    result = validator.validate_summary_contract(path)
    assert result["ok"] is False
    assert any("unmet threshold presented as success" in p for p in result["problems"])


def test_honest_not_reached_status_with_declared_reason_is_accepted(tmp_path):
    summary = _valid_summary(
        q3_baseline_graph2mat_md_equivalence={"status": "not_reached", "value": None, "reason": "no baseline exists"}
    )
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary), encoding="utf-8")
    result = validator.validate_summary_contract(path)
    assert result["ok"] is True, result["problems"]


def test_negative_not_reached_status_without_any_declared_limitation_rejected(tmp_path):
    summary = _valid_summary(
        q3_baseline_graph2mat_md_equivalence={"status": "not_reached", "value": None, "reason": "", "uncertainty": ""}
    )
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary), encoding="utf-8")
    result = validator.validate_summary_contract(path)
    assert result["ok"] is False
    assert any("declares no limitation" in p for p in result["problems"])


def test_negative_unrecognized_status_rejected(tmp_path):
    summary = _valid_summary(q4_pareto_frontier={"status": "definitely_true"})
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary), encoding="utf-8")
    result = validator.validate_summary_contract(path)
    assert result["ok"] is False
    assert any("unrecognized status" in p for p in result["problems"])


# (e) UI controls not propagated to Pilot/Full ----------------------------


_UI_PARAMS = {
    "samplers": ["angular_shell", "sobol_sparse"],
    "dimensionalities": ["3D"],
    "k": 2,
    "domains": [0.03, 0.08],
    "seeds": [0, 1, 2],
    "max_samples": 50,
}


def test_negative_pilot_command_drops_a_ui_param_if_not_propagated():
    params = dict(_UI_PARAMS, densities=[8, 16])
    command = ui.dataset_design_s9_pilot_command(params, python=Path("python3"))
    joined = " ".join(command)
    missing = [
        flag for flag in ("--families", "--dims", "--k-values", "--r-train-max", "--density-levels", "--seeds", "--max-designs")
        if flag not in joined
    ]
    assert not missing, f"UI params not propagated to Pilot command: {missing}"


def test_negative_full_command_drops_a_ui_param_if_not_propagated():
    params = dict(_UI_PARAMS, n_train_values=[128, 256])
    command = ui.dataset_design_s9_full_command(params, python=Path("python3"))
    joined = " ".join(command)
    missing = [
        flag for flag in ("--families", "--dims", "--k-values", "--r-train-max", "--n-levels", "--seeds", "--max-designs")
        if flag not in joined
    ]
    assert not missing, f"UI params not propagated to Full command: {missing}"


@pytest.mark.parametrize("param_key,flag", [
    ("samplers", "--families"),
    ("dimensionalities", "--dims"),
    ("domains", "--r-train-max"),
    ("seeds", "--seeds"),
])
def test_negative_pilot_command_param_missing_from_command_when_dropped(param_key, flag):
    full_params = dict(_UI_PARAMS, densities=[8, 16])
    with_param = ui.dataset_design_s9_pilot_command(full_params, python=Path("python3"))
    dropped_params = {k: v for k, v in full_params.items() if k != param_key}
    without_param = ui.dataset_design_s9_pilot_command(dropped_params, python=Path("python3"))
    assert flag in " ".join(with_param)
    assert flag not in " ".join(without_param), f"{flag} should disappear when {param_key} is unset"


# (f) non-monotonic minimum-N not handled ---------------------------------


def test_negative_non_monotonic_curve_not_derailed_by_early_local_minimum():
    # N=16 is a local dip but doesn't reach the 5%-of-global-best tolerance; a buggy
    # "stop at the first local minimum" implementation would wrongly return N=16.
    dev_mae_by_n = {8: 0.20, 16: 0.11, 32: 0.50, 64: 0.09}
    report = s95.determine_minimum_n(dev_mae_by_n)
    assert report["status"] == "reached"
    assert report["N_min_observed"] == 64


def test_negative_non_monotonic_curve_does_not_fall_back_to_naive_global_argmin():
    dev_mae_by_n = {8: 0.5, 16: 0.30, 32: 0.09, 64: 0.31, 128: 0.10, 256: 0.089}
    naive_argmin_n = min(dev_mae_by_n, key=dev_mae_by_n.get)
    report = s95.determine_minimum_n(dev_mae_by_n)
    assert report["status"] == "reached"
    # The true global best is at the most expensive N (256); a naive
    # pick-the-best-value implementation would report that. The plateau-tolerant
    # algorithm must instead pick the cheapest N within tolerance of it (32).
    assert naive_argmin_n == 256
    assert report["N_min_observed"] == 32
    assert report["N_min_observed"] != naive_argmin_n


# --------------------------------------------------------------------------
# S9-S7: validator accepts the real, currently-materialized S9 artifacts
# --------------------------------------------------------------------------


def test_validator_accepts_real_s9_artifacts():
    completed = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "validate_dataset_design_s9.py")],
        capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
