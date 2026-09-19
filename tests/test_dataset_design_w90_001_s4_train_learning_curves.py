from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "Comparison" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import dataset_design_w90_001_s4_train_learning_curves as s4  # noqa: E402

S3_ROOT = REPO_ROOT / "Comparison/results/dataset_design_w90_001_s3"


def _skip_if_no_s3_data() -> bool:
    return not (S3_ROOT / "structures").exists()


def test_load_s3_samples_excludes_reference_and_resolves_matrices():
    if _skip_if_no_s3_data():
        return
    samples = s4.load_s3_samples(S3_ROOT)
    assert samples
    assert all(sample.family != "reference" for sample in samples)
    assert all(sample.reference_matrix.exists() for sample in samples)
    assert all(sample.run_fdf.exists() for sample in samples)


def test_group_samples_pools_across_amplitudes():
    if _skip_if_no_s3_data():
        return
    samples = s4.load_s3_samples(S3_ROOT)
    groups = s4.group_samples(samples)
    key = ("random_cartesian", "3D", 1)
    assert key in groups
    amplitudes = {sample.amplitude_ang for sample in groups[key]}
    assert amplitudes == {0.03, 0.08}
    assert len(groups[key]) == 128


def test_val_count_for_uses_fraction_with_floor():
    assert s4.val_count_for(16) == 4
    assert s4.val_count_for(64) == 16


def test_select_split_is_deterministic_and_disjoint():
    pool = [
        s4.Sample(f"s{i}", "fam", "3D", 1, 0.03, Path(f"/tmp/{i}/RUN.fdf"), Path(f"/tmp/{i}"), Path(f"/tmp/{i}/ref"))
        for i in range(90)
    ]
    first = s4.select_split(pool, 16, seed=0)
    second = s4.select_split(pool, 16, seed=0)
    assert first is not None and second is not None
    train1, val1 = first
    train2, val2 = second
    assert [s.sample_id for s in train1] == [s.sample_id for s in train2]
    assert [s.sample_id for s in val1] == [s.sample_id for s in val2]
    assert len(train1) == 16
    assert len(val1) == s4.VAL_COUNT
    assert set(s.sample_id for s in train1).isdisjoint(s.sample_id for s in val1)


def test_select_split_is_nested_across_n_train():
    """train(16) subset train(32) subset train(64), and val is identical at every N (S4 nesting fix)."""

    pool = [
        s4.Sample(f"s{i}", "fam", "3D", 1, 0.03, Path(f"/tmp/{i}/RUN.fdf"), Path(f"/tmp/{i}"), Path(f"/tmp/{i}/ref"))
        for i in range(90)
    ]
    splits = {n: s4.select_split(pool, n, seed=0) for n in (16, 32, 64)}
    assert all(split is not None for split in splits.values())
    train_ids = {n: [s.sample_id for s in splits[n][0]] for n in (16, 32, 64)}
    val_ids = {n: [s.sample_id for s in splits[n][1]] for n in (16, 32, 64)}
    assert train_ids[16] == train_ids[32][:16]
    assert train_ids[32] == train_ids[64][:32]
    assert val_ids[16] == val_ids[32] == val_ids[64]


def test_select_split_returns_none_when_pool_too_small():
    pool = [
        s4.Sample(f"s{i}", "fam", "3D", 1, 0.03, Path(f"/tmp/{i}/RUN.fdf"), Path(f"/tmp/{i}"), Path(f"/tmp/{i}/ref"))
        for i in range(10)
    ]
    assert s4.select_split(pool, 16, seed=0) is None


def test_run_one_combo_records_insufficient_samples_without_training(tmp_path):
    pool = [
        s4.Sample(f"s{i}", "axial_radial", "3D", 2, 0.03, Path(f"/tmp/{i}/RUN.fdf"), Path(f"/tmp/{i}"), Path(f"/tmp/{i}/ref"))
        for i in range(5)
    ]
    row = s4.run_one_combo(("axial_radial", "3D", 2), pool, 16, 0, output_root=tmp_path)
    assert row["status"] == "insufficient_samples"
    assert row["available_n"] == 5
    assert row["deeph_included"] is False
    assert row["deeph_exclusion_reason"] == s4.DEEPH_NOT_EVALUATED_REASON
    assert not (tmp_path / "runs").exists()


def test_csv_round_trip_uses_required_columns(tmp_path):
    rows = [
        {
            "family": "sobol_sparse", "dim": "3D", "k": 1, "N_train": 16, "seed": 0,
            "H_MAE": 0.01, "H_RMSE": 0.02, "rel_Frob": 0.03, "spectral_err": 0.04,
            "hermiticity": 1e-12, "backend": "gpu", "status": "ok", "available_n": 128,
            "val_n": 4, "checkpoint": "x.ckpt", "deeph_included": False,
            "deeph_exclusion_reason": "not_evaluated",
        }
    ]
    csv_path = tmp_path / "learning_curves.csv"
    s4.write_csv(csv_path, rows)
    existing = s4.read_existing_rows(csv_path)
    assert ("sobol_sparse", "3D", "1", "16", "0") in existing
    required = {"family", "dim", "k", "N_train", "seed", "H_MAE", "H_RMSE", "rel_Frob", "spectral_err", "hermiticity", "backend"}
    assert required.issubset(s4.CSV_FIELDNAMES)


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
