"""E-F_001-S41: preregister_matbg_gamma_rigid_epc.py builds a frozen, honest protocol."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPO_ROOT / "Comparison/scripts"
for _path in (SCRIPT_DIR, REPO_ROOT / "shared"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import preregister_matbg_gamma_rigid_epc as mod  # noqa: E402

pytestmark = pytest.mark.skipif(
    not mod.DEFAULT_POSITIONS_XYZ.is_file() or not mod.DEFAULT_NEUTRALITY_ESTIMATE.is_file(),
    reason="requires the real tbg_pure_graph2mat overlap/neutrality artifacts on disk",
)


@pytest.fixture(scope="module")
def protocol() -> dict:
    return mod.build_protocol()


def test_geometry_matches_the_declared_target(protocol):
    geometry = protocol["experiment"]["geometry"]
    assert geometry["atom_count"] == mod.TARGET_ATOM_COUNT
    assert geometry["atom_count_matches_target"] is True
    assert geometry["geometry_signature"]


def test_splits_are_disjoint_in_direction_and_k():
    mod.prereg.assert_splits_disjoint(mod.SPLITS)


def test_candidate_directions_cover_every_split_name(protocol):
    named = set(protocol["phonon"]["candidate_sectors"])
    expected = set(mod.RESULT_DIRECTION_NAMES) | set(mod.VALIDATION_DIRECTION_NAMES) | set(mod.CALIBRATION_DIRECTION_NAMES)
    assert named == expected


def test_candidate_directions_are_unit_frobenius(protocol):
    import numpy as np

    for payload in protocol["phonon"]["candidate_sectors"].values():
        vectors = np.asarray(payload["vectors"], dtype=np.float64)
        assert vectors.shape == (mod.TARGET_ATOM_COUNT, 3)
        assert abs(float(np.linalg.norm(vectors)) - 1.0) < 1e-9


def test_no_siesta_reference_path_is_claimed(protocol):
    paths = {row["path"] for row in protocol["references"]["paths"]}
    assert paths == {"graph2mat_jvp", "graph2mat_frozen"}
    assert "66984" in protocol["references"]["siesta_reference_excluded_reason"]


def test_tau_model_is_not_defined(protocol):
    assert "undefined" in protocol["tolerances"]["tau_model"]["definition"]


def test_ready_to_execute_is_false_today(protocol):
    # NO-GO-8a and C14C NO_GO are both live, real states as of this ticket.
    assert protocol["ready_to_execute"] is False
    assert protocol["blocking"]
    assert any("GO-8a" in reason for reason in protocol["blocking"])


def test_claim_ladder_never_exceeds_rigid_tbg_model_label(protocol):
    top_claim = protocol["claim_ladder"][0]["claim"]
    assert "rigid_tbg_model" in top_claim
    forbidden = protocol["experiment"]["forbidden_publication_labels"]
    assert "quantitative_relaxed_matbg_prediction" in forbidden


def test_candidate_sectors_are_tagged_synthetic_not_physical(protocol):
    assert protocol["phonon"]["artifact_kind_of_candidates"] == "synthetic_test_displacement"
    assert protocol["phonon"]["artifact_kind_forbidden_until_go8a"] == "physical_phonon"


def test_freeze_hash_is_idempotent_modulo_generated_at(protocol):
    second = mod.build_protocol()
    assert second["frozen_content_sha256"] == protocol["frozen_content_sha256"]


def test_require_frozen_protocol_refuses_to_run_while_blocked(tmp_path):
    import json

    output_dir = tmp_path / "matbg_prereg"
    output_dir.mkdir()
    protocol = mod.build_protocol()
    (output_dir / mod.PROTOCOL_NAME).write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(mod.PreregistrationError):
        mod.require_frozen_protocol(output_dir / mod.PROTOCOL_NAME)


def test_require_frozen_protocol_detects_tampering(tmp_path):
    import json

    output_dir = tmp_path / "matbg_prereg_tamper"
    output_dir.mkdir()
    protocol = mod.build_protocol()
    protocol["ready_to_execute"] = True  # forge approval without re-freezing the hash
    (output_dir / mod.PROTOCOL_NAME).write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(mod.PreregistrationError):
        mod.require_frozen_protocol(output_dir / mod.PROTOCOL_NAME)
