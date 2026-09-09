"""E-F_001-S50: preregister_matbg_relaxed_epc.py builds a frozen, honest protocol."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPO_ROOT / "Comparison/scripts"
for _path in (SCRIPT_DIR, REPO_ROOT / "shared"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import preregister_matbg_relaxed_epc as mod  # noqa: E402


@pytest.fixture(scope="module")
def protocol() -> dict:
    return mod.build_protocol()


def test_both_branches_are_present(protocol):
    assert set(protocol["branches"]) == {"gamma", "qneq0"}


def test_gamma_branch_is_q_zero_and_qneq0_branch_is_not(protocol):
    gamma = protocol["branches"]["gamma"]["experiment"]
    qneq0 = protocol["branches"]["qneq0"]["experiment"]
    assert gamma["q_label_primary"] == "Gamma"
    assert gamma["k_plus_q_equals_k"] is True
    assert qneq0["q_label_primary"] != "Gamma"
    assert qneq0["k_plus_q_equals_k"] is False
    assert any(abs(c) > 1e-9 for c in qneq0["q_fractional_primary"])


def test_go10_is_required_in_both_branches(protocol):
    for branch in protocol["branches"].values():
        required = {row["name"] for row in branch["preconditions"] if row["required_before_execution"]}
        assert "GO-10 MATBG relaxation provider" in required


def test_go8b_is_required_only_in_qneq0_branch(protocol):
    gamma_required = {
        row["name"] for row in protocol["branches"]["gamma"]["preconditions"] if row["required_before_execution"]
    }
    qneq0_required = {
        row["name"] for row in protocol["branches"]["qneq0"]["preconditions"] if row["required_before_execution"]
    }
    assert "GO-8b MATBG q!=0 response" not in gamma_required
    assert "GO-8b MATBG q!=0 response" in qneq0_required


def test_s48_and_s49_bounds_are_required_but_never_license_go10(protocol):
    for branch in protocol["branches"].values():
        required = {row["name"] for row in branch["preconditions"] if row["required_before_execution"]}
        assert "S48 local-stacking transferability bound" in required
        assert "S49 relaxed phonon-provider bound" in required
        proxy_item = next(item for item in branch["physics_review"]["items"] if item["item"] == "proxy_bounds")
        assert "explicitly license neither GO-10" in proxy_item["basis"]


def test_splits_are_disjoint_in_both_branches():
    mod.prereg.assert_splits_disjoint(mod.SPLITS_GAMMA)
    mod.prereg.assert_splits_disjoint(mod.SPLITS_QNEQ0)


def test_no_rigid_numeric_tolerance_is_embedded_in_the_protocol(protocol):
    for branch in protocol["branches"].values():
        assert "no_rigid_numeric_reuse" in {c["id"] for c in branch["checks"]}
        assert "not inherited from the rigid one" in branch["cost_model"]["reason"]


def test_publication_label_never_exceeds_relaxed_tbg_model(protocol):
    for branch in protocol["branches"].values():
        top_claim = branch["claim_ladder"][0]["claim"]
        assert "relaxed_tbg_model" in top_claim
        assert "quantitative_relaxed_matbg_prediction" in branch["experiment"]["forbidden_publication_labels"]


def test_ready_to_execute_is_false_today(protocol):
    assert protocol["ready_to_execute"] is False
    for branch in protocol["branches"].values():
        assert branch["ready_to_execute"] is False
        assert branch["blocking"]
        assert any("GO-10" in reason for reason in branch["blocking"])


def test_freeze_hash_is_idempotent_modulo_generated_at(protocol):
    second = mod.build_protocol()
    assert second["frozen_content_sha256"] == protocol["frozen_content_sha256"]


def test_require_frozen_protocol_refuses_to_run_while_blocked(tmp_path):
    output_dir = tmp_path / "matbg_relaxed_prereg"
    output_dir.mkdir()
    protocol = mod.build_protocol()
    (output_dir / mod.PROTOCOL_NAME).write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(mod.PreregistrationError):
        mod.require_frozen_protocol(output_dir / mod.PROTOCOL_NAME, branch="gamma")
    with pytest.raises(mod.PreregistrationError):
        mod.require_frozen_protocol(output_dir / mod.PROTOCOL_NAME, branch="qneq0")


def test_require_frozen_protocol_detects_tampering(tmp_path):
    output_dir = tmp_path / "matbg_relaxed_prereg_tamper"
    output_dir.mkdir()
    protocol = mod.build_protocol()
    protocol["branches"]["gamma"]["ready_to_execute"] = True  # forge approval without re-freezing
    (output_dir / mod.PROTOCOL_NAME).write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(mod.PreregistrationError):
        mod.require_frozen_protocol(output_dir / mod.PROTOCOL_NAME, branch="gamma")
