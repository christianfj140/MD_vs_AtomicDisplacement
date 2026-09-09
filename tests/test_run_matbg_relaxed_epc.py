"""E-F_001-S51: the relaxed-MATBG EPC production stage.

S50 (preregister_matbg_relaxed_epc.py) unconditionally blocks both branches
today (no relaxed geometry: S47 is NO-GO-10), so this module's real job is to
refuse correctly and record that refusal as a granular, idempotent artifact --
not to compute anything. These tests exercise that refusal path against the
real, live S50 protocol (cheap: no GPU, no checkpoint, no SIESTA) plus the
idempotency/signature bookkeeping in isolation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPO_ROOT / "Comparison/scripts"
for _path in (SCRIPT_DIR, REPO_ROOT / "shared"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import preregister_matbg_relaxed_epc as prereg_mod  # noqa: E402
import run_matbg_relaxed_epc as mod  # noqa: E402


@pytest.fixture()
def protocol() -> dict:
    return prereg_mod.build_protocol()


def test_produce_branch_rejects_an_unknown_branch(protocol):
    with pytest.raises(mod.RelaxedEpcProductionError):
        mod.produce_branch("delta", protocol=protocol)


@pytest.mark.parametrize("branch", mod.BRANCHES)
def test_produce_branch_blocks_today_and_never_fabricates(tmp_path, monkeypatch, protocol, branch):
    monkeypatch.setattr(mod, "EPC_STATUS_PATH", tmp_path / "artifact_status.json")
    result = mod.produce_branch(branch, protocol=protocol)

    assert result["state"] == "blocked"
    assert result["result_status"] == mod.RESULT_STATUS == "candidate_relaxed_tbg"
    assert result["raw_responses"]["status"] == "not_attempted"
    assert result["pao_covariant_response"]["status"] == "not_attempted"
    assert result["g"]["status"] == "not_attempted"
    for backend_field in (
        "electronic_derivative_backend",
        "basis_response_backend",
        "phonon_backend",
        "epc_backend_class",
    ):
        assert result[backend_field] == "not_attempted"
    assert any("no_relaxed_geometry" in reason for reason in result["blocking"])
    assert result["preregistration_sha256"] == protocol["frozen_content_sha256"]


def test_produce_branch_is_idempotent(tmp_path, protocol, monkeypatch):
    monkeypatch.setattr(mod, "EPC_STATUS_PATH", tmp_path / "artifact_status.json")
    first = mod.produce_branch("gamma", protocol=protocol)
    assert "reused" not in first or first["reused"] is False

    second = mod.produce_branch("gamma", protocol=protocol)
    assert second.get("reused") is True
    assert second["signature_sha256"] == first["signature_sha256"]


def test_produce_branch_recomputes_when_the_protocol_hash_changes(tmp_path, protocol, monkeypatch):
    monkeypatch.setattr(mod, "EPC_STATUS_PATH", tmp_path / "artifact_status.json")
    first = mod.produce_branch("gamma", protocol=protocol)

    mutated = {**protocol, "frozen_content_sha256": "DIFFERENT_HASH_FOR_TEST"}
    mutated["branches"] = protocol["branches"]
    second = mod.produce_branch("gamma", protocol=mutated)
    assert second.get("reused") is not True
    assert second["signature_sha256"] != first["signature_sha256"]


def _patch_load_protocol(monkeypatch, protocol: dict) -> None:
    # patched on the shared preregister_matbg_relaxed_epc module object, so it
    # also covers require_frozen_protocol's own internal load_protocol(path) call.
    monkeypatch.setattr(mod.epc_prereg, "load_protocol", lambda *_args, **_kwargs: protocol)


def test_run_relaxed_epc_writes_a_summary_citing_the_real_s50_protocol(tmp_path, monkeypatch):
    protocol = prereg_mod.build_protocol()
    _patch_load_protocol(monkeypatch, protocol)
    monkeypatch.setattr(mod, "EPC_STATUS_PATH", tmp_path / "artifact_status.json")
    monkeypatch.setattr(mod, "EPC_SUMMARY_PATH", tmp_path / "relaxed_epc_summary.json")

    summary = mod.run_relaxed_epc()

    assert set(summary["branches"]) == set(mod.BRANCHES)
    assert summary["overall_claim"] == "no_execution_authorised"
    assert summary["result_status"] == "candidate_relaxed_tbg"
    for label in ("quantitative_relaxed_matbg_prediction", "g_mn_nu", "phonon_eigenmode"):
        assert label in summary["forbidden_publication_labels"]
    assert summary["preregistration_sha256"] == protocol["frozen_content_sha256"]
    assert (tmp_path / "relaxed_epc_summary.json").is_file()


def test_run_relaxed_epc_supports_a_single_branch(tmp_path, monkeypatch):
    protocol = prereg_mod.build_protocol()
    _patch_load_protocol(monkeypatch, protocol)
    monkeypatch.setattr(mod, "EPC_STATUS_PATH", tmp_path / "artifact_status.json")
    monkeypatch.setattr(mod, "EPC_SUMMARY_PATH", tmp_path / "relaxed_epc_summary.json")

    summary = mod.run_relaxed_epc(branches=("qneq0",))
    assert set(summary["branches"]) == {"qneq0"}
