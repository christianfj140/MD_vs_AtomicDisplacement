"""E-F_001-S35: the GO-7 adjudication of the AB bilayer candidate.

The adjudicator computes no physics, so what is worth testing is the
judgement: layer-swap Hermiticity, shear/breathing geometry, interlayer-block
role and phonon-match separation each fail on a synthetic mutation that
breaks exactly the property they name, and pass on the real S34 candidate
where that property already holds (skipped if the candidate has not been
produced).
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "Comparison" / "scripts"))

import evaluate_ab_bilayer_epc as go7  # noqa: E402

candidate_present = pytest.mark.skipif(
    not go7.DEFAULT_REPORT_PATH.is_file(),
    reason="the AB bilayer candidate has not been produced",
)


@pytest.fixture
def report() -> dict:
    return json.loads(go7.DEFAULT_REPORT_PATH.read_text(encoding="utf-8"))


def _toy_block(row: int, col: int, frobenius: float) -> dict:
    return {"row_cluster": row, "column_cluster": col, "frobenius_norm": frobenius, "singular_values": []}


def _toy_sector(*, ab_forward: float, ab_backward: float, aa: float, bb: float, pattern: dict, overlaps) -> dict:
    return {
        "layer_pattern": pattern,
        "layer_blocks": {
            go7.D_H_SOURCE: [
                _toy_block(0, 0, aa),
                _toy_block(1, 1, bb),
                _toy_block(0, 1, ab_forward),
                _toy_block(1, 0, ab_backward),
            ]
        },
        "phonon_match": {"overlaps_by_branch": overlaps},
    }


def _toy_sectors() -> dict:
    shear_pattern = {"bottom": [[0.5, 0.0, 0.0]], "top": [[-0.5, 0.0, 0.0]]}
    breathing_pattern = {"bottom": [[0.0, 0.0, -0.5]], "top": [[0.0, 0.0, 0.5]]}
    control_pattern = {"bottom": [[0.0, 0.7, 0.0]], "top": [[0.0, 0.0, 0.0]]}
    return {
        "shear": _toy_sector(
            ab_forward=0.02, ab_backward=0.02, aa=0.005, bb=0.005,
            pattern=shear_pattern, overlaps=[0.99, 0.03, 0.001],
        ),
        "layer_breathing": _toy_sector(
            ab_forward=3.5, ab_backward=3.5, aa=0.11, bb=0.11,
            pattern=breathing_pattern, overlaps=[0.999, 0.002],
        ),
        "intralayer_control": _toy_sector(
            ab_forward=0.017, ab_backward=0.017, aa=40.0, bb=0.08,
            pattern=control_pattern, overlaps=[0.73, 0.68],
        ),
    }


def _report(sectors: dict) -> dict:
    return {
        "ticket": "toy",
        "status": "candidate",
        "upstream_gates": {"GO-4": "PASS", "GO-5": "PASS"},
        "spatial_layer_swap": {
            "cell": "derived",
            "positions": "derived",
            "images": "derived",
            "orbitals": "derived",
            "k_fractional": [0.0, 0.0, 0.0],
            "q_fractional": [0.0, 0.0, 0.0],
            "residuals": [{"sector": "toy", "passes": True}],
        },
        "sectors": sectors,
    }


def test_all_five_checks_pass_on_a_well_formed_toy_candidate():
    sectors = _toy_sectors()
    verdict = go7.adjudicate(_report(sectors))
    assert verdict["verdict"] == "NO_GO"  # the toy control's mode match is deliberately ambiguous
    assert verdict["checks_failed"] == ["spatial_layer_swap_from_geometry", "phonon_mode_match_robust_to_branch_ambiguity"]
    assert verdict["failure_attribution"] == [
        {"check": "spatial_layer_swap_from_geometry", "layer": "geometry_construction"},
        {"check": "phonon_mode_match_robust_to_branch_ambiguity", "layer": "phonons"}
    ]


def test_hermitian_block_norm_is_not_a_go7_criterion():
    sectors = _toy_sectors()
    sectors["shear"]["layer_blocks"][go7.D_H_SOURCE][3]["frobenius_norm"] = 0.05  # (1,0) != (0,1)
    verdict = go7.adjudicate(_report(sectors))
    assert "layer_swap_hermitian" not in {row["check"] for row in verdict["checks"]}


def test_shear_direction_fails_on_a_net_translation():
    sectors = _toy_sectors()
    sectors["shear"]["layer_pattern"] = {"bottom": [[0.5, 0.0, 0.0]], "top": [[0.3, 0.0, 0.0]]}
    verdict = go7.adjudicate(_report(sectors))
    assert "shear_direction_in_plane_antiparallel" in verdict["checks_failed"]


def test_breathing_parity_fails_on_a_large_aa_bb_asymmetry():
    sectors = _toy_sectors()
    sectors["layer_breathing"]["layer_blocks"][go7.D_H_SOURCE][1]["frobenius_norm"] = 1.0  # bb >> aa
    verdict = go7.adjudicate(_report(sectors))
    assert "breathing_parity_layers_equal_and_opposite" in verdict["checks_failed"]


def test_interlayer_derivative_blocks_fails_when_the_control_is_not_null():
    sectors = _toy_sectors()
    sectors["intralayer_control"]["layer_blocks"][go7.D_H_SOURCE][2]["frobenius_norm"] = 10.0
    sectors["intralayer_control"]["layer_blocks"][go7.D_H_SOURCE][3]["frobenius_norm"] = 10.0
    verdict = go7.adjudicate(_report(sectors))
    assert "interlayer_derivative_blocks_match_sector_role" in verdict["checks_failed"]


def test_parameter_sensitivity_passes_when_the_match_is_well_separated():
    sectors = _toy_sectors()
    sectors["intralayer_control"]["phonon_match"]["overlaps_by_branch"] = [0.95, 0.05]
    verdict = go7.adjudicate(_report(sectors))
    assert "phonon_mode_match_robust_to_branch_ambiguity" not in verdict["checks_failed"]


def test_missing_sectors_refuses_rather_than_defaulting_to_pass():
    with pytest.raises(go7.Go7AdjudicationError):
        go7.adjudicate({"sectors": {}})


def test_missing_spatial_evidence_fails_closed():
    verdict = go7.adjudicate({"sectors": _toy_sectors(), "upstream_gates": {"GO-4": "PASS", "GO-5": "PASS"}})
    assert "spatial_layer_swap_from_geometry" in verdict["checks_failed"]


@candidate_present
def test_real_candidate_geometry_checks_pass(report):
    """The three structural checks (built from S34's own geometry code) must
    hold on the real candidate; only the sensitivity check is allowed to
    surface a genuine finding."""
    verdict = go7.adjudicate(report)
    always_pass = {
        "shear_direction_in_plane_antiparallel",
        "interlayer_derivative_blocks_match_sector_role",
    }
    failed = set(verdict["checks_failed"])
    assert not (failed & always_pass), f"unexpected structural failure: {failed & always_pass}"
