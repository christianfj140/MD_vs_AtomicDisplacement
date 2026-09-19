from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "Comparison" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import dataset_design_w90_001_s4_train_learning_curves as s4  # noqa: E402
import dataset_design_w90_001_s5_frozen_tests_generalization as s5  # noqa: E402
import w90_displacement_sampler_family as sampler  # noqa: E402


def test_common_displacement_covers_k_and_all_dimensionalities():
    geometry = sampler.load_graphene_primitive()
    entries = s5.build_common_displacement_configs(geometry, n_per_group=2, seed=1)
    sample_ids = [sample_id for sample_id, _ in entries]
    assert len(sample_ids) == len(set(sample_ids))
    ks = {config.metadata["k"] for _sample_id, config in entries}
    assert ks == {1, 2}
    dims = {config.metadata["dimensionality"] for _sample_id, config in entries}
    assert dims == set(sampler.DIMENSIONALITIES)
    families = {config.family for _sample_id, config in entries}
    assert families == {"latin_hypercube"}
    amplitudes = {round(config.metadata["amplitude_ang"], 6) for _sample_id, config in entries}
    lo, hi = sampler.COMMON_DISPLACEMENT_AMPLITUDE_RANGE_ANG
    assert all(lo <= amp <= hi for amp in amplitudes)
    assert len(amplitudes) > 1  # amplitudes mix, not fixed per S2 training generators


def test_ood_displacement_uses_ood_amplitude_and_novel_family_amplitude_combo():
    geometry = sampler.load_graphene_primitive()
    entries = s5.build_ood_displacement_configs(geometry, n_per_group=2, seed=1)
    amplitudes = {config.metadata["amplitude_ang"] for _sample_id, config in entries}
    assert amplitudes == {s5.OOD_AMPLITUDE_ANG}
    k2_entries = [config for _sample_id, config in entries if config.metadata["k"] == 2]
    assert k2_entries
    # local_pair_modes at amplitude 0.12 is a (family, amplitude) combination
    # S3's build_pilot_configurations never generates (only 0.03/0.08 there).
    assert all(config.family == "local_pair_modes" for config in k2_entries)


def test_ood_amplitude_is_outside_pilot_training_grid():
    assert s5.OOD_AMPLITUDE_ANG not in (0.03, 0.08)


def test_estimate_statistical_inefficiency_detects_strong_short_range_correlation():
    indices = list(range(0, 40))
    values = [((-1) ** i) * 1.0 + 0.0001 * i for i in indices]  # alternating -> decorrelates by lag 2
    result = s5.estimate_statistical_inefficiency(indices, values)
    assert result["g"] <= 3
    assert result["n_points"] == 40


def test_estimate_statistical_inefficiency_handles_sparse_irregular_indices():
    indices = [18, 19, 27, 28, 29, 36, 37, 38, 39, 54, 55, 56, 57, 58, 59]
    values = [float(i % 5) for i in indices]
    result = s5.estimate_statistical_inefficiency(indices, values)
    assert result["g"] >= 1
    assert isinstance(result["rho_by_lag"], dict)


def test_select_decorrelated_frames_respects_gap():
    indices = [18, 19, 27, 28, 29, 36, 37, 38, 39, 54, 55, 56, 57, 58, 59]
    selected = s5.select_decorrelated_frames(indices, g=5)
    assert selected[0] == 18
    for a, b in zip(selected, selected[1:]):
        assert b - a >= 5
    assert selected == sorted(selected)


def test_geometry_hash_is_stable_and_position_sensitive():
    import numpy as np

    positions = np.array([[0.0, 0.0, 0.0], [1.467, 0.0, 0.0]])
    assert s5.geometry_hash(positions) == s5.geometry_hash(positions.copy())
    shifted = positions + np.array([0.01, 0.0, 0.0])
    assert s5.geometry_hash(positions) != s5.geometry_hash(shifted)


def test_check_no_leakage_flags_identical_geometry(tmp_path, monkeypatch):
    shared_positions = [(0.0, 0.0, 0.0), (1.467, 0.0, 0.0)]

    training_dir = tmp_path / "train_sample"
    training_dir.mkdir()
    (training_dir / "RUN.fdf").write_text("dummy", encoding="utf-8")
    training_sample = s4.Sample(
        "train_0", "axial_radial", "3D", 1, 0.03, training_dir / "RUN.fdf", training_dir, training_dir / "ref"
    )

    leaked_dir = tmp_path / "frozen_sample"
    leaked_dir.mkdir()
    (leaked_dir / "RUN.fdf").write_text("dummy", encoding="utf-8")
    leaked_sample = s4.Sample(
        "frozen_0", "common_displacement", "3D", 1, 0.05, leaked_dir / "RUN.fdf", leaked_dir, leaked_dir / "ref"
    )

    monkeypatch.setattr(s5, "_positions_from_run_fdf", lambda run_fdf: shared_positions)

    report = s5.check_no_leakage([training_sample], {"common_displacement": [leaked_sample]})
    assert report["leakage_free"] is False
    assert "common_displacement" in report["overlaps"]


def test_build_generalization_matrix_aggregates_by_family_and_test():
    rows = [
        {
            "training_family": "sobol_sparse", "frozen_test": "ood_displacement", "status": "ok",
            "H_MAE": 0.1, "H_RMSE": 0.2, "rel_Frob": 0.05, "spectral_err": 1.0, "hermiticity": 1e-12,
        },
        {
            "training_family": "sobol_sparse", "frozen_test": "ood_displacement", "status": "ok",
            "H_MAE": 0.3, "H_RMSE": 0.4, "rel_Frob": 0.15, "spectral_err": 2.0, "hermiticity": 2e-12,
        },
        {
            "training_family": "sobol_sparse", "frozen_test": "ood_displacement", "status": "error: x",
            "H_MAE": None, "H_RMSE": None, "rel_Frob": None, "spectral_err": None, "hermiticity": None,
        },
    ]
    matrix = s5.build_generalization_matrix(rows)
    assert len(matrix) == 1
    entry = matrix[0]
    assert entry["training_family"] == "sobol_sparse"
    assert entry["frozen_test"] == "ood_displacement"
    assert entry["n_models"] == 2
    assert entry["H_MAE_mean"] == 0.2


def test_generalization_matrix_csv_round_trip(tmp_path):
    rows = [
        {
            "training_family": "sobol_sparse", "frozen_test": "md_frozen", "n_models": 3,
            "H_MAE_mean": 0.1, "H_RMSE_mean": 0.2, "rel_Frob_mean": 0.05,
            "spectral_err_mean": 1.0, "hermiticity_mean": 1e-12,
        }
    ]
    csv_path = tmp_path / "generalization_matrix.csv"
    s5.write_csv(csv_path, rows, s5.GENERALIZATION_MATRIX_FIELDNAMES)
    assert csv_path.exists()
    text = csv_path.read_text(encoding="utf-8")
    assert "training_family" in text.splitlines()[0]


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
