"""Tests for DATASET-DESIGN-W90-001-S9-S8 (final acceptance integration).

Registered in pytest.ini as marker ``DATASET-DESIGN-W90-001-S9-S8``.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "Comparison" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import dataset_design_w90_001_s9_s8_final_acceptance as s8  # noqa: E402

S9_S8_MARK = getattr(pytest.mark, "DATASET-DESIGN-W90-001-S9-S8")

REQUIRED_FILES = ("design_manifest.json", "run_metrics.csv", "domain_generalization.csv", "pareto_table.csv", "summary.json")


@pytest.fixture(scope="module")
def output_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("s9_s8")
    return s8.write_acceptance_package(root)


@S9_S8_MARK
def test_all_five_result_files_present_and_non_empty(output_root):
    for name in REQUIRED_FILES:
        path = output_root / name
        assert path.exists(), name
        assert path.stat().st_size > 0, name


@S9_S8_MARK
def test_summary_answers_all_five_scientific_questions_with_required_fields(output_root):
    summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
    questions = summary["scientific_questions"]
    assert set(questions) == {
        "q1_minimum_training_set_size",
        "q2_distribution_density",
        "q3_baseline_graph2mat_md_equivalence",
        "q4_pareto_frontier",
        "q5_required_displacement_positions_amplitudes",
    }
    for name, answer in questions.items():
        for field in ("value", "units", "uncertainty", "method", "data_source", "status"):
            assert field in answer, (name, field)


@S9_S8_MARK
def test_pruned_branches_non_empty_with_reason_and_evidence(output_root):
    summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
    branches = summary["pruned_branches"]
    assert branches
    for entry in branches:
        assert entry["reason_category"]
        assert entry["evidence_reference"]


@S9_S8_MARK
def test_historical_test_inventory_documents_development_set_usage(output_root):
    summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
    inventory = summary["historical_test_inventory"]
    assert any("development" in entry["role"] for entry in inventory)
    assert any(entry["decisions_influenced"] for entry in inventory)


@S9_S8_MARK
def test_independent_confirmation_is_explicitly_provisional_with_hard_evidence(output_root):
    """Provisional, and for the reason that is actually true right now.

    It used to be "these numbers predate the bug fixes", which stopped being
    the reason once the package started reading the S9 lineage. The binding
    limitation now is executed coverage, and the evidence must name which
    cells are missing rather than carry a caveat nobody can check.
    """
    summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
    confirmation = summary["independent_confirmation"]
    assert confirmation["status"] in ("confirmed", "provisional")
    assert confirmation["status"] == "provisional"
    evidence = confirmation["evidence"]
    assert evidence["conclusion"] == "incomplete_executed_coverage"
    assert evidence["gaps"], "a provisional status must say which cells are missing"
    assert set(evidence["pilot_executed_domains_ang"]) < set(evidence["designed_domains_ang"])
    assert evidence["n_designs_without_siesta_labels"] > 0


@S9_S8_MARK
def test_package_reports_the_bugfixed_s9_lineage_not_the_historical_pilot(output_root):
    """Regression guard for the wiring defect this package was built around.

    The acceptance package used to copy the pre-S9-S1 historical CSVs verbatim,
    so its headline numbers described runs the bug fixes never touched while
    every file-presence check still passed. Assert the published table carries
    the S9 schema and both executed S9 stages, which the historical file cannot.
    """
    rows = list(csv.DictReader((output_root / "run_metrics.csv").open(encoding="utf-8")))
    assert rows, "run_metrics.csv must not be empty"
    assert {"stage", "design_id", "R_train_max_ang"} <= set(rows[0])
    stages = {row["stage"] for row in rows}
    assert stages == {"pilot", "full"}, f"both executed S9 stages must be reported, got {stages}"

    historical = REPO_ROOT / "Comparison/results/dataset_design_w90_001_s4/learning_curves.csv"
    if historical.exists():
        assert (output_root / "run_metrics.csv").read_bytes() != historical.read_bytes(), (
            "run_metrics.csv is a byte-identical copy of the pre-bugfix historical pilot"
        )


@S9_S8_MARK
def test_epistemic_status_buckets_are_distinct_and_non_empty(output_root):
    summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
    for bucket in ("software_validated", "results_executed", "conclusions_pending"):
        assert summary[bucket]


@S9_S8_MARK
def test_three_ui_rows_trace_to_checkpoint_split_geometry_hash_and_siesta_reference(output_root):
    summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
    sample = summary["traceability_sample"]
    assert len(sample) == 3
    for entry in sample:
        assert entry["checkpoint_exists"] is True
        assert entry["split_reproducible"] is True
        assert entry["geometry_hash"]
        assert entry["siesta_references_exist"] is True
        assert entry["traceable"] is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
