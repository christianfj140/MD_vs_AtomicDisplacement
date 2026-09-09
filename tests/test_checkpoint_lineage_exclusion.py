from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "Comparison" / "scripts" / "certify_checkpoint_lineage_and_exclusion.py"
SPEC = importlib.util.spec_from_file_location("certify_checkpoint_lineage_and_exclusion", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

sys.path.insert(0, str(REPO_ROOT / "Comparison" / "scripts"))
from epc_claim_policy import claim_policy  # noqa: E402

REAL_PREREGISTRATION = REPO_ROOT / "docs" / "epc_preregistration_v2.json"
REAL_MANIFEST = (
    REPO_ROOT / "Comparison" / "datasets" / "tbg_md_plus_registry" / "n558" / "frozen_split_manifest.json"
)


def write_run_fdf(path: Path, coords: list[tuple[float, float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["%block AtomicCoordinatesAndAtomicSpecies"]
    for x, y, z in coords:
        lines.append(f"{x} {y} {z} 1")
    lines.append("%endblock AtomicCoordinatesAndAtomicSpecies")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class RowLinkingTests(unittest.TestCase):
    """A row is linked only when every declared artifact re-hashes correctly."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def _row(self, sample_id: str = "s0", split: str = "train", **overrides) -> dict:
        sample_dir = self.root / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        run_fdf = sample_dir / "RUN.fdf"
        write_run_fdf(run_fdf, [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)])
        paths, shas = {"run_fdf": str(run_fdf)}, {}
        for artifact in MODULE.REQUIRED_ARTIFACTS:
            if artifact == "run_fdf":
                continue
            blob = sample_dir / artifact
            blob.write_text(f"{sample_id}:{artifact}", encoding="utf-8")
            paths[artifact] = str(blob)
        for artifact, path in paths.items():
            shas[artifact] = MODULE.file_sha256(path)
        row = {
            "sample_id": sample_id,
            "split": split,
            "source_run": str(self.root),
            "structure_path": str(run_fdf),
            "artifact_paths": paths,
            "artifact_sha256": shas,
        }
        row.update(overrides)
        return row

    def test_complete_row_links(self) -> None:
        linked = MODULE.link_row(self._row())
        self.assertTrue(linked["linked"])
        self.assertEqual(linked["atom_count"], 2)
        self.assertEqual(len(linked["verified_artifact_sha256"]), len(MODULE.REQUIRED_ARTIFACTS))

    def test_missing_source_run_is_unlinked(self) -> None:
        row = self._row()
        row["source_run"] = ""
        linked = MODULE.link_row(row)
        self.assertFalse(linked["linked"])
        self.assertIn("missing field 'source_run'", linked["provenance_problems"])

    def test_wrong_sha256_is_unlinked(self) -> None:
        row = self._row()
        row["artifact_sha256"]["xv"] = "0" * 64
        linked = MODULE.link_row(row)
        self.assertFalse(linked["linked"])
        self.assertTrue(any("xv: sha256" in p for p in linked["provenance_problems"]))

    def test_absent_file_is_unlinked_not_silently_skipped(self) -> None:
        row = self._row()
        row["artifact_paths"]["struct_out"] = str(self.root / "gone")
        linked = MODULE.link_row(row)
        self.assertFalse(linked["linked"])
        self.assertIn("struct_out: file absent, sha256 unverifiable", linked["provenance_problems"])

    def test_unreadable_structure_is_unlinked(self) -> None:
        row = self._row()
        row["structure_path"] = str(self.root / "nope" / "RUN.fdf")
        linked = MODULE.link_row(row)
        self.assertFalse(linked["linked"])
        self.assertIn("structure_path does not exist", linked["provenance_problems"])


class ExclusionTests(unittest.TestCase):
    """Exact exclusion and family exclusion are separate claims."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def _linked(self, sample_id: str, coords, split: str = "train") -> dict:
        path = self.root / sample_id / "RUN.fdf"
        write_run_fdf(path, coords)
        return {
            "sample_id": sample_id,
            "split": split,
            "structure_path": str(path),
            "atom_count": len(coords),
            "species_multiset": ["1"] * len(coords),
            "verified_artifact_sha256": {"run_fdf": MODULE.file_sha256(str(path))},
            "linked": True,
            "provenance_problems": [],
            "_coords": coords,
        }

    def test_identical_geometry_breaks_exact_exclusion(self) -> None:
        coords = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
        rows = [self._linked("t0", coords)]
        path = self.root / "eval" / "RUN.fdf"
        write_run_fdf(path, coords)
        result = MODULE.audit_exclusion(rows, [{"label": "eval0", "path": str(path)}])
        item = result["per_structure"][0]
        self.assertFalse(item["exact_exclusion"]["holds"])
        self.assertEqual(
            [hit["sample_id"] for hit in item["exact_exclusion"]["geometry_identity_collisions"]], ["t0"]
        )
        # Same atom count and species, so family exclusion also fails: the two
        # claims agree here, which is what makes the next test meaningful.
        self.assertFalse(item["family_exclusion"]["holds"])

    def test_family_exclusion_can_hold_while_nothing_was_compared_pairwise(self) -> None:
        """The failure mode the earlier audit had: 0 pairs checked read as 'independent'."""
        rows = [self._linked("t0", [(0.0, 0.0, 0.0)] * 4)]
        path = self.root / "eval" / "RUN.fdf"
        write_run_fdf(path, [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)])  # 2 atoms vs 4
        result = MODULE.audit_exclusion(rows, [{"label": "eval0", "path": str(path)}])
        item = result["per_structure"][0]
        self.assertTrue(item["family_exclusion"]["holds"])
        self.assertEqual(item["family_exclusion"]["same_family_rows"], 0)
        # Distinct claims, separately reported, never merged into one boolean.
        self.assertIn("exact_exclusion", item)
        self.assertIn("family_exclusion", item)
        self.assertNotEqual(id(item["exact_exclusion"]), id(item["family_exclusion"]))

    def test_unlinked_rows_are_never_counted_as_excluded(self) -> None:
        good = self._linked("t0", [(9.0, 9.0, 9.0), (8.0, 8.0, 8.0)])
        bad = {**self._linked("t1", [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]),
               "linked": False, "provenance_problems": ["structure_path does not exist"]}
        path = self.root / "eval" / "RUN.fdf"
        write_run_fdf(path, [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)])
        result = MODULE.audit_exclusion([good, bad], [{"label": "eval0", "path": str(path)}])
        # t1 is geometrically identical to the evaluation structure, but it is
        # unlinked, so it must not silently count either way.
        self.assertEqual(result["unlinked_rows_excluded_from_the_claim"], 1)
        self.assertEqual(result["per_structure"][0]["rows_not_compared_because_unlinked"], 1)


class SelectionIntegrityTests(unittest.TestCase):
    """A validation row identical to a training row is not held out."""

    def _row(self, sample_id, split, shas) -> dict:
        return {"sample_id": sample_id, "split": split, "verified_artifact_sha256": shas}

    def test_identical_validation_row_is_reported(self) -> None:
        result = MODULE.selection_integrity(
            [
                self._row("t0", "train", {"run_fdf": "a" * 64, "xv": "b" * 64}),
                self._row("v0", "validation", {"run_fdf": "a" * 64, "xv": "c" * 64}),
            ]
        )
        self.assertFalse(result["held_out_for_selection"])
        self.assertEqual(result["validation_rows_identical_to_train_rows"], 1)
        self.assertEqual(result["collisions"][0]["identical_train_sample_ids"], ["t0"])

    def test_disjoint_validation_is_held_out(self) -> None:
        result = MODULE.selection_integrity(
            [
                self._row("t0", "train", {"run_fdf": "a" * 64}),
                self._row("v0", "validation", {"run_fdf": "d" * 64}),
            ]
        )
        self.assertTrue(result["held_out_for_selection"])

    def test_basis_descriptor_artifacts_never_create_a_false_collision(self) -> None:
        """ORB_INDX repeats verbatim across physically different samples.

        Matching identity on it would report every validation row as a
        duplicate of every training row (84/84 instead of the real 3).
        """
        self.assertNotIn("orb_indx", MODULE.IDENTITY_ARTIFACTS)
        self.assertNotIn("metadata", MODULE.IDENTITY_ARTIFACTS)
        result = MODULE.selection_integrity(
            [
                self._row("t0", "train", {"run_fdf": "a" * 64, "orb_indx": "z" * 64}),
                self._row("v0", "validation", {"run_fdf": "d" * 64, "orb_indx": "z" * 64}),
            ]
        )
        self.assertTrue(result["held_out_for_selection"])


class ClaimPolicyGaugeTests(unittest.TestCase):
    """The gauge conversion is a separate, non-optional condition."""

    ALL_PASS = {
        name: "PASS"
        for name in (
            "gauge_derivative_audit", "graph2mat_target_gauge_contract", "numerical_PAO_convergence",
            "checkpoint_lineage", "final_case_exclusion", "gpu_runtime", "GO-5", "GO-7",
        )
    }

    def test_both_gates_passing_is_not_enough_without_the_gauge_conversion(self) -> None:
        policy = claim_policy(dict(self.ALL_PASS))
        self.assertEqual(policy["fine_tuning_candidate"], "BLOCKED")
        self.assertIn("final_comparator_gauge_conversion_verified", policy["blocked_by"])

    def test_gauge_conversion_alone_does_not_unblock_when_a_gate_fails(self) -> None:
        gates = {**self.ALL_PASS, "checkpoint_lineage": "NO_GO",
                 "final_comparator_gauge_conversion_verified": True}
        policy = claim_policy(gates)
        self.assertEqual(policy["fine_tuning_candidate"], "BLOCKED")
        self.assertIn("checkpoint_lineage", policy["blocked_by"])

    def test_unblocks_only_when_every_gate_and_the_gauge_conversion_hold(self) -> None:
        gates = {**self.ALL_PASS, "final_comparator_gauge_conversion_verified": True}
        policy = claim_policy(gates)
        self.assertEqual(policy["fine_tuning_candidate"], "UNADJUDICATED")
        self.assertEqual(policy["blocked_by"], [])

    def test_delta_out_closure_still_blocks_full_ks_independently(self) -> None:
        gates = {**self.ALL_PASS, "final_comparator_gauge_conversion_verified": True,
                 "delta_out_closure": "NO_GO"}
        self.assertEqual(claim_policy(gates)["full_KS"], "BLOCKED")


@unittest.skipUnless(
    REAL_PREREGISTRATION.exists() and REAL_MANIFEST.exists(), "real artifacts not present"
)
class RealArtifactTests(unittest.TestCase):
    """Runs against the real checkpoint, manifest and evaluation structures."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.report = MODULE.build_report(REAL_PREREGISTRATION)

    def test_every_manifest_row_is_linked_individually(self) -> None:
        lineage = self.report["checkpoint_lineage"]
        self.assertEqual(lineage["rows_total"], 738)
        self.assertEqual(lineage["rows_linked_individually"], 738)
        self.assertEqual(lineage["rows_unlinked"], 0)
        self.assertEqual(lineage["rows_by_split"], {"train": 558, "validation": 84, "test": 96})
        self.assertEqual(
            lineage["artifact_sha256_verified_count"], 738 * len(MODULE.REQUIRED_ARTIFACTS)
        )

    def test_checkpoint_hash_matches_the_preregistered_value(self) -> None:
        checkpoint = self.report["checkpoint"]
        self.assertEqual(checkpoint["sha256"], checkpoint["sha256_preregistered"])
        self.assertEqual(
            checkpoint["sha256"],
            "9eb7a80961690d5524fc3241c309b63b931c3c108cff550216c6c851cdb9bdea",
        )

    def test_lineage_is_no_go_because_validation_is_not_held_out(self) -> None:
        self.assertEqual(self.report["gates"]["checkpoint_lineage"], "NO_GO")
        integrity = self.report["checkpoint_lineage"]["selection_split_integrity"]
        self.assertFalse(integrity["held_out_for_selection"])
        self.assertEqual(integrity["validation_rows_identical_to_train_rows"], 3)

    def test_exclusion_reports_exact_and_family_separately(self) -> None:
        exclusion = self.report["final_case_exclusion"]
        self.assertEqual(exclusion["verdict"], "PASS")
        for item in exclusion["per_structure"]:
            self.assertTrue(item["exact_exclusion"]["holds"])
            self.assertTrue(item["family_exclusion"]["holds"])
            self.assertEqual(item["rows_compared_for_exact_exclusion"], 642)

    def test_evaluation_structures_are_never_blind_holdouts(self) -> None:
        for item in self.report["final_case_exclusion"]["per_structure"]:
            self.assertTrue(item["previously_inspected"])
            self.assertFalse(item["blind_holdout"])

    def test_preserved_verdicts_are_not_relabelled(self) -> None:
        self.assertEqual(
            self.report["preserved_verdicts"],
            {
                "graph2mat_target_gauge_contract": "PASS",
                "gauge_derivative_audit": "PASS",
                "numerical_PAO_convergence": "PASS",
                "delta_out_closure": "NO_GO",
                "full_KS": "BLOCKED",
                "full_basis_connection_v1": "NO_GO",
                "onlyS_semantic_preflight_v1": "NO_GO",
                "gate_1_support_structure": "PASS",
            },
        )

    def test_convergence_margin_is_reported_as_close_to_the_limit(self) -> None:
        margin = self.report["numerical_convergence_margin"]
        self.assertEqual(margin["limit_rms_mev_per_ang"], 5.0)
        self.assertLess(margin["egg_box_rms_mev_per_ang"], 5.0)
        self.assertGreater(margin["egg_box_rms_mev_per_ang"], 4.5)
        self.assertGreater(margin["mesh_rms_mev_per_ang"], 4.0)

    def test_gauge_difference_matches_the_certified_derivative(self) -> None:
        source = REPO_ROOT / "Comparison/results/epc/pao_flow_audit/graph2mat_target_gauge_contract.json"
        certified = json.loads(source.read_text(encoding="utf-8"))
        self.assertAlmostEqual(
            self.report["gauge_constraint"]["d_cvac_minus_dEf_ev_per_ang"],
            certified["C1_x_five_point_derivatives_ev_per_ang"]["d_cvac_minus_Ef"],
            places=12,
        )

    def test_real_gates_keep_fine_tuning_candidate_blocked(self) -> None:
        policy = claim_policy(
            {
                "gauge_derivative_audit": "PASS",
                "graph2mat_target_gauge_contract": "PASS",
                "numerical_PAO_convergence": "PASS",
                **self.report["gates"],
            }
        )
        self.assertEqual(policy["fine_tuning_candidate"], "BLOCKED")
        self.assertIn("checkpoint_lineage", policy["blocked_by"])
        self.assertIn("final_comparator_gauge_conversion_verified", policy["blocked_by"])


if __name__ == "__main__":
    unittest.main()
