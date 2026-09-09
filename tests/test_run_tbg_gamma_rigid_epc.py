"""E-F_001-S42: the rigid-Gamma raw-derivative stage added to
run_tbg_pure_graph2mat_campaign.py.

No GPU, no checkpoint, no SIESTA: the heavy producers
(epc_c14.load_model_and_batch / directional_derivative_field /
frozen_derivative_field, epc_g2m_deriv.resolve_jvp_backend /
batch_topology_hash) are monkeypatched with small deterministic stand-ins so
what is actually exercised is this stage's own logic -- the S41 gate, the
per-artifact status bookkeeping/idempotency, and the "blocked" bookkeeping for
PAO-covariant response/g -- not the model or the sparse solver.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPO_ROOT / "Comparison/scripts"
for _path in (SCRIPT_DIR, REPO_ROOT / "shared"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import fd_perturbation_space as fdp  # noqa: E402
import run_tbg_pure_graph2mat_campaign as camp  # noqa: E402


class _FakeBackend:
    requested = "cuda"
    effective = "cpu"
    preflight = "failed"
    reason = "cuda_unavailable: fake backend for this test"

    def to_metadata(self) -> dict:
        return {
            "requested_backend": self.requested,
            "effective_backend": self.effective,
            "backend_preflight": self.preflight,
            "backend_fallback_reason": self.reason,
        }


def _tiny_directions() -> dict[str, fdp.Direction]:
    translation = fdp.uniform_translation(3, "x")
    other = fdp.Direction(
        name="probe_direction",
        kind="collective",
        vectors=np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]) / 1.0,
    )
    return {"translation_x": translation, "probe_direction": other}


def _fake_protocol(directions: dict[str, fdp.Direction]) -> dict:
    return {
        "frozen_content_sha256": "FAKESHA",
        "blocking": [
            "GO-8a MATBG PhononProvider: NO-GO-8a",
            "C14C basis response (global formalism gate): NO_GO",
            "k_selection: no live electronic spectrum measured yet",
            "moire_hexagonal_convention_checked: not yet verified",
        ],
        "claim_ladder": [
            {
                "outcome": "derivative-level checks pass (jvp_equals_coordinate_combination, "
                "jvp_equals_frozen, generalized_hellmann_feynman_diagonal) but GO-8a is still "
                "NO-GO-8a",
                "claim": "graph2mat_derivative_validated_no_full_ks_coupling",
            },
        ],
        "phonon": {
            "candidate_sectors": {
                name: {"direction_hash": direction.direction_hash} for name, direction in directions.items()
            }
        },
        "perturbation": {"delta_sweep_ang": [0.01, 0.02]},
        "references": {
            "formalism_id": "fake_formalism_v1",
            "basis_response": {"representation": "S_L_S_R"},
        },
    }


def test_epc_authorize_rejects_an_unexpected_blocking_reason(monkeypatch):
    protocol = _fake_protocol(_tiny_directions())
    monkeypatch.setattr(camp.epc_prereg.prereg, "freeze_hash", lambda payload: protocol["frozen_content_sha256"])
    protocol["blocking"].append("some_new_gate_nobody_pre-authorised: FAIL")
    with pytest.raises(camp.EpcGammaRigidError):
        camp.epc_authorize(protocol)


def test_epc_authorize_rejects_a_tampered_protocol(monkeypatch):
    protocol = _fake_protocol(_tiny_directions())
    monkeypatch.setattr(camp.epc_prereg.prereg, "freeze_hash", lambda payload: "DIFFERENT_HASH")
    with pytest.raises(camp.EpcGammaRigidError):
        camp.epc_authorize(protocol)


def test_epc_authorize_grants_the_derivative_only_rung(monkeypatch):
    protocol = _fake_protocol(_tiny_directions())
    monkeypatch.setattr(camp.epc_prereg.prereg, "freeze_hash", lambda payload: protocol["frozen_content_sha256"])
    gate = camp.epc_authorize(protocol)
    assert gate["result_status"] == camp.EPC_RESULT_STATUS == "candidate_rigid_tbg"
    assert gate["authorised_claim"] == "graph2mat_derivative_validated_no_full_ks_coupling"


def test_write_sparse_field_npz_round_trips(tmp_path):
    field = {(0, 0, (0, 0, 0)): 1.5, (0, 1, (1, 0, 0)): -2.25}
    path = tmp_path / "field.npz"
    camp._write_sparse_field_npz(path, d_h=field)
    with np.load(path) as data:
        assert list(data["d_h__values"]) == [1.5, -2.25]
        assert list(data["d_h__rows"]) == [0, 0]


def test_raw_derivative_stage_is_idempotent_and_records_status(tmp_path, monkeypatch):
    directions = _tiny_directions()
    direction = directions["translation_x"]

    monkeypatch.setattr(camp, "EPC_ARRAYS_DIR", tmp_path / "arrays")
    monkeypatch.setattr(camp, "EPC_STATUS_PATH", tmp_path / "artifact_status.json")
    calls = {"jvp": 0, "frozen": 0}

    def fake_jvp(model, batch, direction, *, processor, change_of_basis, backend):
        calls["jvp"] += 1
        return {(0, 0, (0, 0, 0)): 1.0}, {"backend": "fake"}

    def fake_frozen(model, batch, direction, delta, *, processor, change_of_basis):
        calls["frozen"] += 1
        # A tiny delta-independent field: the plateau is exactly flat, so
        # tau_num == 0 and jvp_equals_frozen requires an exact match.
        return {(0, 0, (0, 0, 0)): 1.0}, ["topo-hash"]

    monkeypatch.setattr(camp.epc_c14, "directional_derivative_field", fake_jvp)
    monkeypatch.setattr(camp.epc_c14, "frozen_derivative_field", fake_frozen)

    status: dict = {"artifacts": {}}
    kwargs = dict(
        name="translation_x",
        direction=direction,
        model=None,
        batch=None,
        processor=None,
        change_of_basis=np.eye(3),
        deltas=[0.01, 0.02],
        checkpoint=Path(__file__),  # any real file, so file_sha256 resolves
        topology_hash="topo-hash",
        backend=_FakeBackend(),
        status=status,
    )

    first = camp.epc_raw_derivative_for_direction(**kwargs)
    assert first["state"] == "completed"
    assert first["checks"]["jvp_equals_frozen_within_backend_margin"] is True
    assert calls == {"jvp": 1, "frozen": 2}

    second = camp.epc_raw_derivative_for_direction(**kwargs)
    assert second.get("reused") is True
    # Reused: the fakes were not called again.
    assert calls == {"jvp": 1, "frozen": 2}

    # A different checkpoint (different file content) changes the signature and
    # forces recompute.
    kwargs["checkpoint"] = Path(camp.__file__)
    third = camp.epc_raw_derivative_for_direction(**kwargs)
    assert third["state"] == "completed"
    assert calls == {"jvp": 2, "frozen": 4}


def test_raw_derivative_stage_records_failure_without_crashing(tmp_path, monkeypatch):
    direction = _tiny_directions()["translation_x"]
    monkeypatch.setattr(camp, "EPC_ARRAYS_DIR", tmp_path / "arrays")
    monkeypatch.setattr(camp, "EPC_STATUS_PATH", tmp_path / "artifact_status.json")

    def boom(*args, **kwargs):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(camp.epc_c14, "directional_derivative_field", boom)

    status: dict = {"artifacts": {}}
    result = camp.epc_raw_derivative_for_direction(
        name="translation_x",
        direction=direction,
        model=None,
        batch=None,
        processor=None,
        change_of_basis=np.eye(3),
        deltas=[0.01],
        checkpoint=Path(__file__),
        topology_hash="topo-hash",
        backend=_FakeBackend(),
        status=status,
    )
    assert result["state"] == "failed"
    assert "synthetic failure" in result["error"]
    assert status["artifacts"]["raw_derivative__translation_x"]["state"] == "failed"
