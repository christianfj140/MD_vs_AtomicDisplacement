from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "Comparison" / "scripts" / "certify_checkpoint_selection_remediation.py"
SPEC = importlib.util.spec_from_file_location("certify_checkpoint_selection_remediation", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

sys.path.insert(0, str(REPO_ROOT / "Comparison" / "scripts"))
import epc_gauge_alignment as GAUGE  # noqa: E402

GAUGE_CONTRACT = (
    REPO_ROOT / "Comparison" / "results" / "epc" / "pao_flow_audit"
    / "graph2mat_target_gauge_contract.json"
)
REPORT = (
    REPO_ROOT / "Comparison" / "results" / "epc" / "checkpoint_selection_remediation"
    / "checkpoint_selection_remediation.json"
)
CERTIFIED_D_GAUGE_EV_PER_ANG = 3.4869062475910892e-3


class IdentityRuleTests(unittest.TestCase):
    """The clean manifest must never inherit the basis-descriptor false collision."""

    def test_identity_artifacts_are_shared_with_the_lineage_gate(self) -> None:
        # One rule, imported, not restated -- so the two gates cannot drift apart.
        self.assertEqual(MODULE.IDENTITY_ARTIFACTS, MODULE.LINEAGE.IDENTITY_ARTIFACTS)

    def test_basis_descriptor_artifacts_are_excluded_from_identity(self) -> None:
        self.assertNotIn("orb_indx", MODULE.IDENTITY_ARTIFACTS)
        self.assertNotIn("metadata", MODULE.IDENTITY_ARTIFACTS)
        for artifact in ("run_fdf", "xv", "struct_out", "reference_tshs"):
            self.assertIn(artifact, MODULE.IDENTITY_ARTIFACTS)

    def test_no_sample_id_is_hardcoded_as_a_criterion(self) -> None:
        """Removals must be reproducible from artifact hashes, not from names."""
        source = SCRIPT.read_text(encoding="utf-8")
        for known_collision in (
            "bilayer_graphene_aa_md200__md_160",
            "bilayer_graphene_ab_md200__md_160",
            "bilayer_graphene_ba_md200__md_160",
        ):
            self.assertNotIn(known_collision, source)

    def test_filename_epoch_is_never_the_source_of_membership(self) -> None:
        self.assertNotIn("filename_claims_epoch", MODULE.RUN_IDENTITY_KEYS)
        for key in MODULE.RUN_IDENTITY_KEYS:
            self.assertNotIn("filename", key)


class GaugeAlignmentTests(unittest.TestCase):
    """K^{E_F}(R) = K^{c_vac}(R) + [c_vac(R) - E_F(R)].S(R), applied per geometry."""

    STEP = 0.005

    def _samples(self):
        """A five-point stencil in which BOTH gauge terms are non-zero.

        c_vac - E_F varies with the offset, and so does S, so an a-posteriori
        constant shift cannot reproduce the answer.
        """
        samples = []
        for offset in (-2, -1, 0, 1, 2):
            displacement = offset * self.STEP
            samples.append(
                {
                    "offset": offset,
                    # K^{c_vac} and S both move with the geometry.
                    "k_cvac": {("a",): 1.0 + 2.0 * displacement, ("b",): -0.5 * displacement},
                    "overlap": {("a",): 1.0 + 0.25 * displacement, ("b",): 0.125},
                    # (c_vac - E_F)(R) with the certified slope.
                    "c_vac_ev": 4.0 + CERTIFIED_D_GAUGE_EV_PER_ANG * displacement,
                    "fermi_ev": -1.0,
                }
            )
        return samples

    def test_recovers_the_certified_gauge_derivative(self) -> None:
        result = GAUGE.aligned_derivative(self._samples(), step_ang=self.STEP)
        self.assertAlmostEqual(
            result["d_cvac_minus_Ef_ev_per_ang"], CERTIFIED_D_GAUGE_EV_PER_ANG, places=12
        )

    def test_certified_value_matches_the_gauge_contract_artifact(self) -> None:
        if not GAUGE_CONTRACT.is_file():
            self.skipTest("gauge contract artifact not present")
        payload = json.loads(GAUGE_CONTRACT.read_text(encoding="utf-8"))
        recorded = payload["C1_x_five_point_derivatives_ev_per_ang"]["d_cvac_minus_Ef"]
        self.assertAlmostEqual(recorded, CERTIFIED_D_GAUGE_EV_PER_ANG, places=15)
        self.assertEqual(payload["verdict"], "PASS")

    def test_aligned_derivative_equals_the_full_three_term_identity(self) -> None:
        """dK^{E_F} = dK^{c_vac} + (c_vac'-E_F').S + (c_vac-E_F).dS, no term dropped."""
        samples = self._samples()
        result = GAUGE.aligned_derivative(samples, step_ang=self.STEP)
        equilibrium = next(entry for entry in samples if entry["offset"] == 0)

        expected = GAUGE._combine(
            (1.0, result["unconverted_derivative"]),
            (result["d_cvac_minus_Ef_ev_per_ang"], equilibrium["overlap"]),
            (result["cvac_minus_Ef_at_equilibrium_ev"], result["d_overlap"]),
        )
        for key, value in result["aligned_derivative"].items():
            self.assertAlmostEqual(value, expected[key], places=10, msg=f"key {key}")

    def test_a_posteriori_constant_shift_is_not_equivalent(self) -> None:
        """The dS term is first order; dropping it changes the answer."""
        samples = self._samples()
        result = GAUGE.aligned_derivative(samples, step_ang=self.STEP)
        equilibrium = next(entry for entry in samples if entry["offset"] == 0)

        approximate = GAUGE._combine(
            (1.0, result["unconverted_derivative"]),
            (result["d_cvac_minus_Ef_ev_per_ang"], equilibrium["overlap"]),
        )
        # ("a",) has a geometry-dependent overlap, so the neglected term bites.
        self.assertNotAlmostEqual(
            result["aligned_derivative"][("a",)], approximate[("a",)], places=6
        )

    def test_round_trip_between_the_two_gauges(self) -> None:
        k_cvac = {("a",): 1.25, ("b",): -0.75}
        overlap = {("a",): 0.9, ("b",): 0.1}
        forward = GAUGE.to_fermi_gauge(k_cvac, overlap, c_vac_ev=4.0, fermi_ev=-1.0)
        back = GAUGE.to_vacuum_gauge(forward, overlap, c_vac_ev=4.0, fermi_ev=-1.0)
        for key, value in k_cvac.items():
            self.assertAlmostEqual(back[key], value, places=12)

    def test_stencil_refuses_an_incomplete_sample_set(self) -> None:
        with self.assertRaises(ValueError):
            GAUGE.five_point_derivative({-2: {}, -1: {}, 1: {}}, step_ang=self.STEP)

    def test_module_adjudicates_nothing(self) -> None:
        result = GAUGE.aligned_derivative(self._samples(), step_ang=self.STEP)
        self.assertTrue(result["computes_no_residual"])
        for forbidden in ("residual", "verdict", "decision", "fine_tuning_candidate"):
            self.assertNotIn(forbidden, result)


@unittest.skipUnless(REPORT.is_file(), "remediation report not generated")
class RealReportTests(unittest.TestCase):
    """The emitted verdict, against the real run's artifacts."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.report = json.loads(REPORT.read_text(encoding="utf-8"))

    def test_schema_and_verdict(self) -> None:
        self.assertEqual(self.report["schema"], MODULE.SCHEMA)
        self.assertEqual(self.report["verdict"], "BLOCKED")
        self.assertTrue(self.report["blockers"])

    def test_every_snapshot_is_proven_to_belong_to_the_same_run(self) -> None:
        inventory = self.report["checkpoint_inventory"]
        self.assertTrue(inventory["all_snapshots_same_run"])
        for snapshot in inventory["snapshots"]:
            self.assertTrue(snapshot["sha256"])
            self.assertIsNotNone(snapshot["epoch"])
            self.assertEqual(snapshot["run_identity_mismatches"], [])

    def test_clean_manifest_is_frozen_before_evaluation(self) -> None:
        manifest = self.report["clean_validation_manifest_v1"]
        self.assertTrue(manifest["frozen_before_any_checkpoint_evaluation"])
        self.assertEqual(manifest["validation_rows_before"], 84)
        self.assertEqual(manifest["validation_rows_removed"], 3)
        self.assertEqual(manifest["validation_rows_after"], 81)
        self.assertTrue(manifest["test_is_never_a_selection_input"])
        # Every removal is justified by a geometry-bearing artifact hash.
        for removal in manifest["removals"]:
            self.assertTrue(removal["colliding_identity_keys"])
            for key in removal["colliding_identity_keys"]:
                self.assertIn(key.split(":")[0], MODULE.IDENTITY_ARTIFACTS)

    def test_the_removal_rate_is_not_the_false_84_of_84(self) -> None:
        manifest = self.report["clean_validation_manifest_v1"]
        self.assertNotEqual(
            manifest["validation_rows_removed"], manifest["validation_rows_before"]
        )

    def test_selection_is_not_reproducible_and_says_why_with_numbers(self) -> None:
        repro = self.report["selection_reproducibility"]
        self.assertFalse(repro["reproducible"])
        self.assertGreater(repro["running_best_epochs_without_weights"], 0)
        self.assertLess(
            repro["epochs_with_surviving_weights"], repro["epochs_the_original_rule_ranked"]
        )
        self.assertEqual(repro["monitor"], "val_spectral_frontier_aligned_rmse_eV")

    def test_the_original_checkpoint_is_not_promoted(self) -> None:
        preserved = self.report["preserved_verdicts"]
        self.assertEqual(preserved["checkpoint_lineage_original"], "NO_GO")
        self.assertEqual(preserved["final_case_exclusion"], "PASS")
        self.assertEqual(preserved["fine_tuning_candidate"], "BLOCKED")
        self.assertEqual(preserved["full_KS"], "BLOCKED")
        self.assertEqual(preserved["delta_out_closure"], "NO_GO")


if __name__ == "__main__":
    unittest.main()
