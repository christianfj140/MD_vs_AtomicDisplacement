"""E-F_001-S49: the relaxed-MATBG ``PhononProvider`` bound.

S47 (NO-GO-10, no relaxed geometry) and S39/S38 (NO-GO-8a, no MATBG-scale
PhononProvider) are both already-certified refusals this module imports and
reuses rather than recomputing; this suite checks the bound built on top of
them: the displacement-sensitivity and local-stacking phonon suites measured
against real archived FC data, and that no relaxed physical_phonon is ever
fabricated while either upstream gate stays NO-GO.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT / "Comparison" / "scripts", REPO_ROOT / "shared"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import certify_relaxed_matbg_phonon_provider as bound_mod  # noqa: E402
import phonon_provider as pp  # noqa: E402
from artifact_signature import PHYSICAL_PHONON  # noqa: E402


# --------------------------------------------------------------------------- #
# Current repository state: NO-GO-10 and NO-GO-8a, nothing fabricated.
# --------------------------------------------------------------------------- #


def test_no_relaxed_geometry_and_no_provider_produce_no_relaxed_phonon():
    bound = bound_mod.certify_relaxed_matbg_phonon_provider_bound()
    assert bound["go10_status"] == "NO-GO-10"
    assert bound["go8a_status"] == "NO-GO-8a"
    assert bound["relaxed_geometry_available"] is False
    assert bound["relaxed_physical_phonon_produced"] is False
    assert bound["relaxed_physical_phonon"] is None
    assert bound["licenses_go8a_for_relaxed_geometry"] is False
    assert bound["quantitative_relaxed_matbg_phonon_claim_licensed"] is False


def test_require_relaxed_matbg_physical_phonon_refuses_current_state():
    bound = bound_mod.certify_relaxed_matbg_phonon_provider_bound()
    with pytest.raises(bound_mod.RelaxedMatbgPhononBoundError, match="go10=.*go8a="):
        bound_mod.require_relaxed_matbg_physical_phonon(bound)


def test_require_relaxed_matbg_physical_phonon_refuses_a_malformed_object():
    with pytest.raises(bound_mod.RelaxedMatbgPhononBoundError):
        bound_mod.require_relaxed_matbg_physical_phonon({"relaxed_physical_phonon_produced": True})


# --------------------------------------------------------------------------- #
# displacement_sensitivity_suite: real archived graphene Gamma FC at 3 deltas.
# --------------------------------------------------------------------------- #


def test_displacement_sensitivity_suite_passes_at_every_archived_delta():
    if not all(path.is_dir() for path in bound_mod.GRAPHENE_DELTA_RUNS.values()):
        pytest.skip("not every archived graphene delta FC run is present")
    bound = bound_mod.certify_relaxed_matbg_phonon_provider_bound()
    suite = bound["displacement_sensitivity_suite"]
    assert suite["status"] == "PASS"
    assert suite["deltas_available"] == sorted(bound_mod.GRAPHENE_DELTA_RUNS)
    by_name = {entry["check"]: entry for entry in suite["checks"]}
    for label in bound_mod.GRAPHENE_DELTA_RUNS:
        assert by_name[f"graphene_gamma_{label}"]["passed"] is True


def test_displacement_sensitivity_spread_diagnostics_are_measured_not_gated():
    if not all(path.is_dir() for path in bound_mod.GRAPHENE_DELTA_RUNS.values()):
        pytest.skip("not every archived graphene delta FC run is present")
    bound = bound_mod.certify_relaxed_matbg_phonon_provider_bound()
    spread = bound["displacement_sensitivity_suite"]["spread_diagnostics"]
    assert spread["applicable"] is True
    assert spread["zo_spread_cm1"] >= 0.0
    assert spread["e2g_spread_cm1"] >= 0.0
    # The frequency spread across delta=0.005..0.02 Ang is far smaller than
    # the branch separation these thresholds already certify (S39: ZO/E2g
    # windows are >100 cm^-1 wide) -- measured, not asserted against an
    # invented tolerance.
    assert spread["zo_spread_cm1"] < 5.0
    assert spread["e2g_spread_cm1"] < 5.0
    angle = spread["e2g_eigenvector_subspace_angle"]
    assert angle is not None
    assert angle["maximum_principal_angle_rad"] >= 0.0
    assert angle["maximum_principal_angle_rad"] < 1e-3


def test_displacement_sensitivity_reports_missing_deltas_as_not_applicable(monkeypatch, tmp_path):
    monkeypatch.setattr(
        bound_mod, "GRAPHENE_DELTA_RUNS",
        {"d0p005": tmp_path / "missing_a", "d0p01": tmp_path / "missing_b", "d0p02": tmp_path / "missing_c"},
    )
    suite = bound_mod._certify_displacement_sensitivity()
    assert suite["status"] == "NOT_VERIFIED"
    assert suite["deltas_available"] == []
    assert suite["spread_diagnostics"]["applicable"] is False
    for entry in suite["checks"]:
        assert entry["applicable"] is False
        assert entry["passed"] is True


# --------------------------------------------------------------------------- #
# local_stacking_phonon_suite: only AB has an archived FC run today.
# --------------------------------------------------------------------------- #


def test_local_stacking_suite_measures_ab_and_marks_aa_ba_not_applicable():
    ab_run_root = bound_mod.SIESTA_REFERENCE_ROOT / "bilayer_graphene_AB" / "runs"
    if not any(ab_run_root.glob("fc_dhs_*")):
        pytest.skip("no archived AB bilayer FC run")
    bound = bound_mod.certify_relaxed_matbg_phonon_provider_bound()
    suite = bound["local_stacking_phonon_suite"]
    by_name = {entry["check"]: entry for entry in suite["checks"]}

    ab = by_name["bilayer_graphene_AB_phonon"]
    assert ab["applicable"] is True
    assert ab["passed"] is True
    assert ab["geometry_signature"]
    assert ab["backend"] == pp.SIESTA_FC_BACKEND

    for label in ("bilayer_graphene_AA_phonon", "bilayer_graphene_BA_phonon"):
        entry = by_name[label]
        if entry["applicable"]:
            # only true if someone has since archived an FC run for it
            continue
        assert entry["passed"] is True  # not-applicable never fails the gate
        assert "not marked compatible" in entry["detail"]


def test_local_stacking_material_with_no_archived_fc_is_not_applicable(tmp_path, monkeypatch):
    monkeypatch.setattr(bound_mod, "SIESTA_REFERENCE_ROOT", tmp_path)
    entry = bound_mod._local_stacking_material({"label": "bilayer_graphene_AA", "fdf": "unused"})
    assert entry["applicable"] is False
    assert entry["passed"] is True
    assert "not marked compatible" in entry["detail"]


# --------------------------------------------------------------------------- #
# Reopen path: the GO-8a contract is re-run verbatim against a real provider.
# --------------------------------------------------------------------------- #


def test_go8a_contract_is_reused_verbatim_when_a_provider_is_supplied():
    """Not a rewrite: passing a provider forwards straight into S39's own certifier."""
    if not bound_mod.go8a_cert.GRAPHENE_FC_RUN.is_dir():
        pytest.skip(f"no archived FC run at {bound_mod.go8a_cert.GRAPHENE_FC_RUN}")
    stand_in_fdf = bound_mod.go8a_cert.GRAPHENE_FC_RUN / "RUN.fdf"
    provider = pp.SiestaFcPhononProvider.from_run_dir(bound_mod.go8a_cert.GRAPHENE_FC_RUN, apply_asr=True)

    direct = bound_mod.go8a_cert.certify_matbg_phonon_provider_go8a(provider, target_fdf=stand_in_fdf)
    via_bound = bound_mod.certify_relaxed_matbg_phonon_provider_bound(provider, relaxed_target_fdf=stand_in_fdf)

    assert via_bound["go8a_status"] == direct["go8a_status"]
    assert via_bound["go8a_certification"]["matbg_scale_suite"]["status"] == direct["matbg_scale_suite"]["status"]
    # go10 (relaxed geometry existence) is a separate, still-NO-GO gate: a
    # provider on its own cannot flip relaxed_physical_phonon_produced.
    assert via_bound["go10_status"] == "NO-GO-10"
    assert via_bound["relaxed_physical_phonon_produced"] is False
    assert via_bound["relaxed_physical_phonon"] is None


def test_relaxed_physical_phonon_requires_both_go10_and_go8a(monkeypatch):
    """A provider clearing GO-8a alone must not flip relaxed_physical_phonon_produced."""
    if not bound_mod.go8a_cert.GRAPHENE_FC_RUN.is_dir():
        pytest.skip(f"no archived FC run at {bound_mod.go8a_cert.GRAPHENE_FC_RUN}")
    stand_in_fdf = bound_mod.go8a_cert.GRAPHENE_FC_RUN / "RUN.fdf"
    provider = pp.SiestaFcPhononProvider.from_run_dir(bound_mod.go8a_cert.GRAPHENE_FC_RUN, apply_asr=True)
    monkeypatch.setattr(bound_mod.go8a_cert, "TARGET_ATOM_COUNT", 2)

    bound = bound_mod.certify_relaxed_matbg_phonon_provider_bound(provider, relaxed_target_fdf=stand_in_fdf)

    assert bound["go8a_status"] in ("GO-8a", "NO-GO-8a")
    assert bound["go10_status"] == "NO-GO-10"
    # relaxed_geometry_available stays False regardless of go8a: S47 alone gates it.
    assert bound["relaxed_geometry_available"] is False
    assert bound["relaxed_physical_phonon_produced"] is False
    with pytest.raises(bound_mod.RelaxedMatbgPhononBoundError):
        bound_mod.require_relaxed_matbg_physical_phonon(bound)


# --------------------------------------------------------------------------- #
# Structural sanity of the report itself.
# --------------------------------------------------------------------------- #


def test_bound_report_is_json_safe_and_carries_provenance():
    import json

    bound = bound_mod.certify_relaxed_matbg_phonon_provider_bound()
    payload = json.dumps(bound_mod._json_safe(bound))
    reloaded = json.loads(payload)
    assert reloaded["schema"] == bound_mod.SCHEMA
    assert reloaded["ticket"] == "E-F_001-S49"
    assert reloaded["status_label"] == "relaxed_matbg_phonon_provider_bound"
    assert isinstance(reloaded["references"], list) and reloaded["references"]
