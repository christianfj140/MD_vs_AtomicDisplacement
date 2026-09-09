"""The gauge conversion inside the *real* comparator, not inside the helper module.

``epc_gauge_alignment`` was already tested in isolation. What was untested -- and
what ``quantify_checkpoint_derivative_error`` said about itself in its own
``decision_blockers`` -- is whether the comparator that actually forms the
residual puts both sides into one electrostatic gauge, and whether it does so
*before* the finite-difference stencil rather than as an afterwards shift.

These tests are about the comparison, never about the model. None of them
asserts that the checkpoint is good, and the last one asserts that a passing
gauge gate still leaves ``fine_tuning_candidate`` BLOCKED.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "Comparison" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import epc_gauge_alignment as gauge  # noqa: E402
import quantify_checkpoint_derivative_error as qc  # noqa: E402
from epc_claim_policy import claim_policy  # noqa: E402

GAUGE_CONTRACT = (
    REPO_ROOT / "Comparison" / "results" / "epc" / "pao_flow_audit"
    / "graph2mat_target_gauge_contract.json"
)
CERTIFIED_D_GAUGE_EV_PER_ANG = 3.4869062475910892e-3


# --------------------------------------------------------------------------- #
# 1. Both product-rule terms non-zero: a-posteriori-only is a different number
# --------------------------------------------------------------------------- #


class ProductRuleTests(unittest.TestCase):
    """A stencil where c_vac - E_F AND S both move, so neither term vanishes."""

    STEP = 0.005

    def _samples(self):
        samples = []
        for offset in (-2, -1, 0, 1, 2):
            displacement = offset * self.STEP
            samples.append(
                {
                    "offset": offset,
                    "k_cvac": {("a",): 1.0 + 2.0 * displacement, ("b",): -0.5 * displacement},
                    # S varies with the geometry -> dS != 0
                    "overlap": {("a",): 1.0 + 0.25 * displacement, ("b",): 0.125},
                    # c_vac - E_F varies with the geometry -> d(c_vac - E_F) != 0
                    "c_vac_ev": 4.0 + CERTIFIED_D_GAUGE_EV_PER_ANG * displacement,
                    "fermi_ev": -1.0,
                }
            )
        return samples

    def test_both_terms_are_actually_non_zero(self):
        """Guard on the fixture itself: a test of 'both terms' needs both terms."""
        result = gauge.aligned_derivative(self._samples(), step_ang=self.STEP)
        self.assertNotAlmostEqual(result["d_cvac_minus_Ef_ev_per_ang"], 0.0, places=6)
        self.assertNotAlmostEqual(result["d_overlap"][("a",)], 0.0, places=6)

    def test_a_posteriori_only_correction_is_not_the_per_geometry_conversion(self):
        """Dropping (c_vac - E_F).dS changes the answer; both terms are first order."""
        samples = self._samples()
        result = gauge.aligned_derivative(samples, step_ang=self.STEP)
        equilibrium = next(entry for entry in samples if entry["offset"] == 0)

        a_posteriori_only = gauge._combine(
            (1.0, result["unconverted_derivative"]),
            (result["d_cvac_minus_Ef_ev_per_ang"], equilibrium["overlap"]),
        )
        per_geometry = result["aligned_derivative"]

        self.assertNotAlmostEqual(per_geometry[("a",)], a_posteriori_only[("a",)], places=6)
        # And the gap is exactly the omitted product-rule term.
        omitted = result["cvac_minus_Ef_at_equilibrium_ev"] * result["d_overlap"][("a",)]
        self.assertAlmostEqual(
            per_geometry[("a",)] - a_posteriori_only[("a",)], omitted, places=10
        )


# --------------------------------------------------------------------------- #
# 2. The certified scalar, and the real artifact it came from
# --------------------------------------------------------------------------- #


class CertifiedTermTests(unittest.TestCase):
    STEP = 0.005

    def test_five_point_stencil_recovers_the_certified_term(self):
        values = {
            offset: 4.0 + CERTIFIED_D_GAUGE_EV_PER_ANG * (offset * self.STEP)
            for offset in (-2, -1, 1, 2)
        }
        recovered = gauge.five_point_scalar_derivative(values, step_ang=self.STEP)
        self.assertAlmostEqual(recovered, CERTIFIED_D_GAUGE_EV_PER_ANG, places=12)

    def test_the_certified_term_matches_the_real_gauge_contract_artifact(self):
        if not GAUGE_CONTRACT.is_file():
            self.skipTest(f"{GAUGE_CONTRACT} absent")
        payload = json.loads(GAUGE_CONTRACT.read_text(encoding="utf-8"))
        derivatives = payload["C1_x_five_point_derivatives_ev_per_ang"]
        self.assertAlmostEqual(
            derivatives["d_cvac_minus_Ef"], CERTIFIED_D_GAUGE_EV_PER_ANG, places=15
        )
        # and it really is d_c_vac - d_Ef, not an independently stored number
        self.assertAlmostEqual(
            derivatives["d_c_vac"] - derivatives["d_Ef"],
            CERTIFIED_D_GAUGE_EV_PER_ANG,
            places=15,
        )
        self.assertEqual(payload["verdict"], "PASS")
        self.assertGreater(
            abs(CERTIFIED_D_GAUGE_EV_PER_ANG),
            0.0,
            "a zero gauge term would make the whole conversion moot",
        )


# --------------------------------------------------------------------------- #
# 3. Per-geometry conversion == direct construction in the target gauge
# --------------------------------------------------------------------------- #


class EquivalenceTests(unittest.TestCase):
    """Converting each geometry then differentiating == building in E_F gauge directly."""

    STEP = 0.005

    def test_converted_pipeline_equals_direct_construction_in_the_fermi_gauge(self):
        rng = np.random.default_rng(20260906)
        keys = [("a",), ("b",), ("c",)]
        samples, direct = [], {}
        for offset in (-2, -1, 0, 1, 2):
            displacement = offset * self.STEP
            overlap = {key: 1.0 + 0.3 * (index + 1) * displacement
                       for index, key in enumerate(keys)}
            c_vac = 4.0 + 0.7 * displacement
            fermi = -1.0 + 0.2 * displacement
            k_cvac = {key: float(rng.normal()) + 1.5 * (index + 1) * displacement
                      for index, key in enumerate(keys)}
            samples.append(
                {"offset": offset, "k_cvac": k_cvac, "overlap": overlap,
                 "c_vac_ev": c_vac, "fermi_ev": fermi}
            )
            # The same physical object built straight in the Fermi gauge.
            direct[offset] = {key: k_cvac[key] + (c_vac - fermi) * overlap[key] for key in keys}

        via_conversion = gauge.aligned_derivative(samples, step_ang=self.STEP)["aligned_derivative"]
        built_directly = gauge.five_point_derivative(direct, step_ang=self.STEP)

        for key in keys:
            self.assertAlmostEqual(via_conversion[key], built_directly[key], places=12, msg=str(key))

    def test_the_two_gauges_round_trip_exactly(self):
        k_cvac = {("a",): 1.25, ("b",): -0.75}
        overlap = {("a",): 0.9, ("b",): 0.1}
        forward = gauge.to_fermi_gauge(k_cvac, overlap, c_vac_ev=4.0, fermi_ev=-1.0)
        back = gauge.to_vacuum_gauge(forward, overlap, c_vac_ev=4.0, fermi_ev=-1.0)
        for key, value in k_cvac.items():
            self.assertAlmostEqual(back[key], value, places=12)


# --------------------------------------------------------------------------- #
# 4. Regression: comparing across gauges must be caught
# --------------------------------------------------------------------------- #


def _gauge_row(name, *, reference_norm, separation, closure, converted, unconverted):
    """One ``gauge_conversion`` record shaped as ``compare_direction`` emits it."""
    return {
        "direction_name": name,
        "gauge_conversion": {
            "d_cvac_minus_Ef_ev_per_ang": CERTIFIED_D_GAUGE_EV_PER_ANG,
            "cvac_minus_Ef_at_equilibrium_ev": 5.82,
            "reference_vacuum_gauge": {"frobenius": reference_norm},
            "per_geometry_minus_a_posteriori": {"frobenius": separation},
            "identity_closure": {"frobenius": closure},
            "omitted_product_rule_term_cvac_minus_Ef_times_dS": {"frobenius": separation},
            "residual_in_vacuum_gauge": {"frobenius": converted},
            "control_unconverted_model_vs_vacuum_reference": {"frobenius": unconverted},
        },
    }


class CrossGaugeRegressionTests(unittest.TestCase):
    """If anyone compares H_G2M against K_ref without converting, this must fail."""

    def test_gate_passes_when_the_conversion_is_present_and_effective(self):
        gate = qc.gauge_conversion_gate(
            [
                _gauge_row("atom0000_x", reference_norm=48.49, separation=11.7,
                           closure=5e-13, converted=1.0, unconverted=12.0),
                _gauge_row("random_seed0", reference_norm=23.75, separation=5.85,
                           closure=5e-13, converted=0.8, unconverted=6.0),
            ]
        )
        self.assertEqual(gate["verdict"], "PASS")

    def test_comparing_H_G2M_directly_against_K_ref_without_conversion_is_caught(self):
        """The regression this gate exists for.

        Skipping the conversion means K^{c_vac} is never formed, so the discrete
        three-term identity does not close: the residual left over is exactly
        the size of the terms that were never applied. The gate must refuse.
        """
        unconverted_closure = 11.7  # the whole (c_vac - E_F).dS term, unapplied
        gate = qc.gauge_conversion_gate(
            [
                _gauge_row("atom0000_x", reference_norm=48.49, separation=11.7,
                           closure=unconverted_closure, converted=25.9, unconverted=21.3),
            ]
        )
        self.assertEqual(gate["verdict"], "NO_GO")
        self.assertFalse(gate["per_direction"][0]["identity_closes"])

    def test_an_a_posteriori_only_correction_is_caught(self):
        """Applying only (c_vac' - E_F').S leaves the (c_vac - E_F).dS term open."""
        gate = qc.gauge_conversion_gate(
            [
                _gauge_row("atom0000_x", reference_norm=48.49, separation=11.7,
                           closure=11.72, converted=1.0, unconverted=12.0),
            ]
        )
        self.assertEqual(gate["verdict"], "NO_GO")
        self.assertFalse(gate["per_direction"][0]["identity_closes"])

    def test_the_gate_does_not_take_its_verdict_from_the_models_error(self):
        """A gate on the comparison may not be decided by how wrong the model is.

        Same conversion evidence, three very different model residuals: the
        verdict must not move. This is what stops an overshooting model -- whose
        error partially cancels the gauge term -- from being read as a gauge
        failure, and stops that accidental cancellation from being rewarded.
        """
        verdicts = {
            qc.gauge_conversion_gate(
                [
                    _gauge_row("atom0000_x", reference_norm=48.49, separation=11.7,
                               closure=5e-13, converted=converted, unconverted=21.3)
                ]
            )["verdict"]
            for converted in (1.0e-3, 12.0, 25.9, 40.0)
        }
        self.assertEqual(verdicts, {"PASS"})

    def test_the_gauge_blocker_is_only_retired_by_a_passing_gate(self):
        blocker = "common electrostatic gauge not yet wired into this comparison artifact"
        rows = [
            {
                "direction_name": "atom0000_x", "direction_kind": "one_hot",
                "reference": {"frobenius": 48.49}, "relative_frobenius": 0.02,
                "residual": {"frobenius": 1.0},
                "error_budget": {"residual_over_explained": 100.0},
            }
        ]
        failing = qc.decide(rows, calibration=("atom0000_x",),
                            gauge_gate={"verdict": "NO_GO", "per_direction": []})
        self.assertIn(blocker, failing["decision_blockers"])
        self.assertFalse(failing["final_comparator_gauge_conversion_verified"])

        passing = qc.decide(rows, calibration=("atom0000_x",),
                            gauge_gate={"verdict": "PASS", "per_direction": []})
        self.assertNotIn(blocker, passing["decision_blockers"])
        self.assertTrue(passing["final_comparator_gauge_conversion_verified"])
        # The other two blockers are untouched by gauge work.
        self.assertIn("finite-PAO basis completeness and Delta_out not bounded",
                      passing["decision_blockers"])
        self.assertIn("reference convergence evidence not attached to this comparison artifact",
                      passing["decision_blockers"])

    def test_no_direction_at_all_cannot_pass_the_gate(self):
        self.assertEqual(qc.gauge_conversion_gate([])["verdict"], "NO_GO")

    def test_a_null_direction_alone_cannot_pass_the_gate(self):
        """translation_x has dS = 0 exactly; it can testify to nothing."""
        gate = qc.gauge_conversion_gate(
            [
                _gauge_row("translation_x", reference_norm=0.0108, separation=1e-12,
                           closure=5e-13, converted=1e-3, unconverted=1e-3),
            ]
        )
        self.assertEqual(gate["verdict"], "NO_GO")


# --------------------------------------------------------------------------- #
# The critical constraint: a passing gauge gate promotes nothing
# --------------------------------------------------------------------------- #


class ClaimPolicyIsolationTests(unittest.TestCase):
    def test_gauge_pass_does_not_unblock_fine_tuning_while_lineage_is_no_go(self):
        gates = {
            "final_comparator_gauge_conversion_verified": True,
            "gauge_derivative_audit": "PASS",
            "graph2mat_target_gauge_contract": "PASS",
            "numerical_PAO_convergence": "PASS",
            "final_case_exclusion": "PASS",
            "gpu_runtime": "PASS",
            "GO-5": "PASS",
            "GO-7": "PASS",
            "checkpoint_lineage": "NO_GO",   # the historical selection, unrepaired
            "delta_out_closure": "NO_GO",
            "previously_inspected": True,
        }
        policy = claim_policy(gates)
        self.assertEqual(policy["fine_tuning_candidate"], "BLOCKED")
        self.assertEqual(policy["final_claim"], "BLOCKED")
        self.assertEqual(policy["full_KS"], "BLOCKED")
        self.assertIn("checkpoint_lineage", policy["blocked_by"])
        self.assertNotIn("final_comparator_gauge_conversion_verified", policy["blocked_by"])
        self.assertIs(policy["blind_holdout"], False)

    def test_lineage_is_the_sole_remaining_blocker_once_the_gauge_is_wired(self):
        """What the gauge work did and did not buy, stated as an assertion."""
        gates = {
            "final_comparator_gauge_conversion_verified": True,
            "gauge_derivative_audit": "PASS",
            "graph2mat_target_gauge_contract": "PASS",
            "numerical_PAO_convergence": "PASS",
            "final_case_exclusion": "PASS",
            "gpu_runtime": "PASS",
            "GO-5": "PASS",
            "GO-7": "PASS",
            "checkpoint_lineage": "NO_GO",
            "previously_inspected": True,
        }
        self.assertEqual(claim_policy(gates)["blocked_by"], ["checkpoint_lineage"])


# --------------------------------------------------------------------------- #
# The real artifacts
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_per_geometry_conversion_on_the_real_graphene_campaign():
    """The conversion, on the certified runs, before the stencil."""
    pytest.importorskip("sisl")
    if not qc.DEFAULT_ENERGY_ZERO.is_file():
        pytest.skip(f"no energy-zero artifact at {qc.DEFAULT_ENERGY_ZERO}")
    if not (qc.DEFAULT_REFERENCE_ROOT / "epc_siesta_reference_manifest.json").is_file():
        pytest.skip("no graphene reference campaign")

    from certify_siesta_dhsdr import load_campaign, read_tshs
    from read_siesta_dhsdr import read_dhsdr

    campaign = load_campaign(qc.DEFAULT_REFERENCE_ROOT)
    levels = qc.load_vacuum_levels(qc.DEFAULT_ENERGY_ZERO, campaign)
    assert levels, "no vacuum levels matched this campaign's runs"

    equilibrium_id = campaign.manifest["certification_set"]["equilibrium_run_id"]
    equilibrium = read_tshs(
        campaign.run_dir(equilibrium_id), equilibrium_id,
        fermi_ev=campaign.fermi_ev(equilibrium_id),
    )
    entry = next(
        e for e in campaign.manifest["perturbation_space"]["directions"]
        if e["direction_name"] == "atom0000_x"
    )
    delta = 0.005
    reference = qc.reference_directional(
        campaign, equilibrium,
        direction_name="atom0000_x",
        direction_kind=str(entry["direction_kind"]),
        vectors=np.asarray(entry["vectors"], dtype=np.float64),
        delta_ang=delta,
        dhsdr=read_dhsdr(campaign.dhsdr_path(delta)),
        vacuum_levels=levels,
        equilibrium_run_id=equilibrium_id,
    )
    record = reference.gauge

    # The discrete three-term identity closes to float64: the conversion really
    # happened per geometry, before the difference.
    assert record["identity_closure"]["frobenius"] < 1e-9, record["identity_closure"]

    # The omitted term is not a rounding detail on real data.
    omitted = record["omitted_product_rule_term_cvac_minus_Ef_times_dS"]["frobenius"]
    assert omitted > qc.GAUGE_TERM_FLOOR_EV_PER_ANG, omitted
    assert record["per_geometry_minus_a_posteriori"]["frobenius"] > qc.GAUGE_TERM_FLOOR_EV_PER_ANG

    # And the per-geometry result reproduces the independently certified
    # vacuum-gauge derivative norm recorded by the energy-zero campaign.
    payload = json.loads(qc.DEFAULT_ENERGY_ZERO.read_text(encoding="utf-8"))
    certified = next(
        row["aligned_fd_frobenius_ev_per_ang"]
        for row in payload["directional_offsets"]
        if row["direction_name"] == "atom0000_x" and abs(row["delta_ang"] - delta) < 1e-12
    )
    assert record["d_vacuum_fd"]["frobenius"] == pytest.approx(certified, rel=1e-9)


if __name__ == "__main__":
    unittest.main()
