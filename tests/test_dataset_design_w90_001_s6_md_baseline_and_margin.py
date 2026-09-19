from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "Comparison" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import dataset_design_w90_001_s4_train_learning_curves as s4  # noqa: E402
import dataset_design_w90_001_s6_md_baseline_and_margin as s6  # noqa: E402

MD_RESULTS_ROOT = REPO_ROOT / "Comparison/results/results_md"


def _skip_if_no_md_data() -> bool:
    return not MD_RESULTS_ROOT.exists()


def test_verify_md_split_integrity_finds_no_partition_overlap():
    if _skip_if_no_md_data():
        return
    report = s6.verify_md_split_integrity(MD_RESULTS_ROOT)
    assert report["split_verified"] is True
    assert report["no_partition_overlap"] is True
    for entry in report["per_dataset"]:
        assert entry["status"] == "ok"


def test_verify_md_split_integrity_reports_train_block_hamiltonians_missing():
    # Documents the real, current repo state: only test-partition frames
    # persisted a SIESTA Hamiltonian (workspace with train-block data was
    # cleaned up) -- this is the root cause behind (b)'s insufficient result.
    if _skip_if_no_md_data():
        return
    report = s6.verify_md_split_integrity(MD_RESULTS_ROOT)
    assert report["train_block_hamiltonians_available"] is False


def test_check_md_frozen_decorrelation_selection_respects_measured_gap(tmp_path):
    if _skip_if_no_md_data():
        return
    samples, report, verified = s6.check_md_frozen_decorrelation(tmp_path)
    assert verified is True
    selected = report["selected_frame_indices"]
    g = report["g_used"]
    for a, b in zip(selected, selected[1:]):
        assert b - a >= g
    assert len(samples) == len(selected)


def test_md_training_pool_excludes_frozen_indices():
    if _skip_if_no_md_data():
        return
    frozen_report = {"selected_frame_indices": [18, 27, 36, 45, 49, 54, 58]}
    pool = s6.md_training_pool_indices(MD_RESULTS_ROOT, frozen_report)
    assert set(pool).isdisjoint(frozen_report["selected_frame_indices"])
    # Real repo state: 20 persisted frames total, 7 already claimed by
    # md_frozen -> at most 13 remain, short of even N_train=16.
    assert len(pool) <= 13


def test_run_md_baseline_combo_records_insufficient_without_training(tmp_path):
    pool = [
        s4.Sample(f"md_{i}", "md_train_pool", "md", 0, float("nan"), Path(f"/tmp/{i}/RUN.fdf"), Path(f"/tmp/{i}"), Path(f"/tmp/{i}/ref"))
        for i in range(13)
    ]
    rows = s6.run_md_baseline_combo(
        pool, 16, 0, output_root=tmp_path, get_frozen_samples_by_test=lambda: {}
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "insufficient_md_training_data"
    assert row["available_n"] == 13
    assert not (tmp_path / "runs").exists()


def test_fix_non_inferiority_margin_not_justified_when_md_baseline_missing():
    pilot_signal = {"rel_Frob_min": 0.158, "rel_Frob_median": 0.186, "n_models": 120}
    report = s6.fix_non_inferiority_margin(pilot_signal, md_training_available=False)
    assert report["delta_ni_fixed"] is False
    assert report["delta_ni_relative_h_mae"] is None
    assert report["reasons_not_justified"]


def test_fix_non_inferiority_margin_justified_when_evidence_supports_candidate():
    pilot_signal = {"rel_Frob_min": 0.02, "rel_Frob_median": 0.03, "n_models": 120}
    report = s6.fix_non_inferiority_margin(pilot_signal, md_training_available=True)
    assert report["delta_ni_fixed"] is True
    assert report["delta_ni_relative_h_mae"] == s6.CANDIDATE_MARGIN_RELATIVE_H_MAE
    assert report["reasons_not_justified"] == []


def test_summarize_pilot_signal_reads_real_s4_csv_when_present():
    if not s6.DEFAULT_S4_ROOT.joinpath("learning_curves.csv").exists():
        return
    signal = s6.summarize_pilot_signal()
    assert signal["n_models"] > 0
    assert signal["rel_Frob_min"] is not None


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
