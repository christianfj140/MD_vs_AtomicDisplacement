"""S46 / E-F_001-S46: the rigid-MATBG Gamma / q != 0 campaigns are published, not recomputed.

The endpoint must read the canonical GO-8a/GO-8b/GO-9 certifications and the frozen S41/S44
protocols and pass their fields through unchanged; the frontend must only format them. A missing
q != 0 execution artifact is reported as absent, never as a zero or a validated result.
"""

import importlib
import json
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "Comparison" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

pipeline_ui = importlib.import_module("pipeline_ui")


def artifacts_present() -> bool:
    return all(path.exists() for path in pipeline_ui.EPC_MATBG_REQUIRED_SOURCES.values())


class EpcMatbgPayloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not artifacts_present():
            raise unittest.SkipTest("GO-8a/GO-8b/GO-9 MATBG artifacts are not in this checkout")
        cls.payload = pipeline_ui.epc_matbg_payload()
        cls.docs = {
            name: json.loads(path.read_text(encoding="utf-8"))
            for name, path in pipeline_ui.EPC_MATBG_REQUIRED_SOURCES.items()
        }

    def test_available_and_gates_come_from_the_certification_artifacts(self) -> None:
        self.assertTrue(self.payload["available"])
        gates = self.payload["gates"]
        self.assertEqual(gates["go8a"]["status"], self.docs["go8a"]["go8a_status"])
        self.assertEqual(gates["go8b"]["status"], self.docs["go8b"]["go8b_status"])
        self.assertEqual(gates["go9_gamma"]["status"], self.docs["go9_gamma"]["go9_status"])
        self.assertEqual(gates["go9_qneq0"]["status"], self.docs["go9_qneq0"]["go9_status"])
        self.assertEqual(
            gates["go9_gamma"]["final_publication_label"],
            self.docs["go9_gamma"]["final_publication_label"],
        )

    def test_branch_protocols_expose_blocking_and_claim_ladder_unchanged(self) -> None:
        branches = self.payload["branches"]
        gamma_protocol = self.docs["protocol_gamma"]
        qneq0_protocol = self.docs["protocol_qneq0"]
        self.assertEqual(branches["gamma"]["protocol"]["blocking"], gamma_protocol["blocking"])
        self.assertEqual(branches["gamma"]["protocol"]["claim_ladder"], gamma_protocol["claim_ladder"])
        self.assertEqual(branches["gamma"]["protocol"]["ready_to_execute"], gamma_protocol["ready_to_execute"])
        self.assertEqual(branches["qneq0"]["protocol"]["blocking"], qneq0_protocol["blocking"])
        self.assertEqual(branches["qneq0"]["protocol"]["claim_ladder"], qneq0_protocol["claim_ladder"])

    def test_phonon_candidate_vectors_are_dropped_but_metadata_survives(self) -> None:
        gamma_phonon = self.payload["branches"]["gamma"]["protocol"]["phonon"]
        source_sectors = self.docs["protocol_gamma"]["phonon"]["candidate_sectors"]
        self.assertEqual(set(gamma_phonon["candidate_sectors"]), set(source_sectors))
        for name, row in gamma_phonon["candidate_sectors"].items():
            self.assertNotIn("vectors", row)
            self.assertEqual(row["direction_hash"], source_sectors[name]["direction_hash"])

    def test_qneq0_branch_absence_is_reported_not_hidden_or_zeroed(self) -> None:
        campaign = self.payload["branches"]["qneq0"]["campaign"]
        self.assertFalse(campaign["executed"])
        self.assertIn("reason", campaign)
        self.assertNotIn("claim", campaign)

    def test_gamma_campaign_defaults_to_not_executed_when_no_run_artifact_exists(self) -> None:
        campaign = self.payload["branches"]["gamma"]["campaign"]
        if not pipeline_ui.EPC_MATBG_OPTIONAL_SOURCES["gamma_campaign_summary"].exists():
            self.assertFalse(campaign["executed"])
            self.assertEqual(campaign["directions"], {})
            self.assertEqual(campaign["restart_state"], [])

    def test_provenance_hashes_every_required_artifact(self) -> None:
        artifacts = self.payload["provenance"]["artifacts"]
        self.assertEqual({a["artifact"] for a in artifacts}, set(pipeline_ui.EPC_MATBG_REQUIRED_SOURCES))
        for artifact in artifacts:
            self.assertEqual(len(artifact["sha256"] or ""), 64)

    def test_missing_artifact_is_never_reported_as_available(self) -> None:
        broken = dict(pipeline_ui.EPC_MATBG_REQUIRED_SOURCES)
        broken["go9_qneq0"] = REPO_ROOT / "does_not_exist_s45_certification.json"
        original = pipeline_ui.EPC_MATBG_REQUIRED_SOURCES
        pipeline_ui.EPC_MATBG_REQUIRED_SOURCES = broken
        try:
            payload = pipeline_ui.epc_matbg_payload()
        finally:
            pipeline_ui.EPC_MATBG_REQUIRED_SOURCES = original
        self.assertFalse(payload["available"])
        self.assertEqual(payload["missing"], ["go9_qneq0"])
        self.assertNotIn("gates", payload)


class EpcMatbgFrontendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.index_html = (REPO_ROOT / "Comparison" / "ui" / "index.html").read_text(encoding="utf-8")
        cls.app_js = (REPO_ROOT / "Comparison" / "ui" / "app.js").read_text(encoding="utf-8")

    def test_tab_and_view_are_wired_into_the_existing_ui(self) -> None:
        self.assertIn('data-view="epc-matbg"', self.index_html)
        self.assertIn('id="view-epc-matbg"', self.index_html)
        self.assertIn('tab.dataset.view === "epc-matbg"', self.app_js)
        self.assertIn('request("/api/epc/matbg")', self.app_js)
        self.assertIn('getElementById("epc-matbg-refresh")', self.app_js)

    def test_every_rendered_container_exists_in_the_markup(self) -> None:
        for element_id in (
            "epc-matbg-verdict-title",
            "epc-matbg-chips",
            "epc-matbg-gates",
            "epc-matbg-gamma-identity",
            "epc-matbg-gamma-campaign",
            "epc-matbg-gamma-blocking",
            "epc-matbg-gamma-ladder",
            "epc-matbg-gamma-directions",
            "epc-matbg-gamma-restart",
            "epc-matbg-qneq0-identity",
            "epc-matbg-qneq0-campaign",
            "epc-matbg-qneq0-blocking",
            "epc-matbg-qneq0-ladder",
            "epc-matbg-artifacts",
        ):
            self.assertIn(f'id="{element_id}"', self.index_html)
            self.assertIn(f'"{element_id}"', self.app_js)

    def test_frontend_does_no_physics(self) -> None:
        start = self.app_js.index("// ===== EPC MATBG rigid Gamma")
        end = self.app_js.index("function setupTabs()", start)
        section = self.app_js[start:end]
        for forbidden in ("Math.sqrt", "Math.exp", "Math.cos", "Math.sin", "fourier", "**"):
            self.assertNotIn(forbidden, section)


if __name__ == "__main__":
    unittest.main()
