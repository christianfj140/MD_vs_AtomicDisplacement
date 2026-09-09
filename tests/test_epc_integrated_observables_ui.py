"""E-F_001-S56: the rigid/relaxed integrated-observables results (S54/S55) are published, not

recomputed. The endpoint must read those two canonical result artifacts and pass their fields
through unchanged; a missing artifact is reported as absent, never as a zero or a computed value.
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
    return all(path.exists() for path in pipeline_ui.EPC_INTEGRATED_OBSERVABLES_SOURCES.values())


class EpcIntegratedObservablesPayloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not artifacts_present():
            raise unittest.SkipTest("S54/S55 integrated-observables artifacts are not in this checkout")
        cls.payload = pipeline_ui.epc_integrated_observables_payload()
        cls.docs = {
            name: json.loads(path.read_text(encoding="utf-8"))
            for name, path in pipeline_ui.EPC_INTEGRATED_OBSERVABLES_SOURCES.items()
        }

    def test_available_and_branch_fields_come_from_the_result_artifacts(self) -> None:
        self.assertTrue(self.payload["available"])
        for name in ("rigid", "relaxed"):
            branch = self.payload["branches"][name]
            doc = self.docs[name]
            self.assertEqual(branch["ticket"], doc["ticket"])
            self.assertEqual(branch["overall_claim"], doc["overall_claim"])
            self.assertEqual(branch["verdict"], doc["interpretation"]["verdict"])
            self.assertEqual(branch["blocking"], doc["production"]["blocking"])
            self.assertEqual(branch["observables"], doc["production"]["observables"])

    def test_no_observable_is_reported_as_computed_while_both_sides_are_blocked(self) -> None:
        for name in ("rigid", "relaxed"):
            for row in self.payload["branches"][name]["observables"].values():
                self.assertEqual(row["status"], "not_attempted")

    def test_rigid_vs_relaxed_delta_is_passed_through_unchanged(self) -> None:
        expected = self.docs["relaxed"]["production"]["rigid_vs_relaxed_delta"]
        self.assertEqual(self.payload["rigid_vs_relaxed_delta"], expected)

    def test_provenance_hashes_every_source_artifact(self) -> None:
        artifacts = self.payload["provenance"]["artifacts"]
        self.assertEqual({a["artifact"] for a in artifacts}, set(pipeline_ui.EPC_INTEGRATED_OBSERVABLES_SOURCES))
        for artifact in artifacts:
            self.assertEqual(len(artifact["sha256"] or ""), 64)

    def test_missing_artifact_is_never_reported_as_available(self) -> None:
        broken = dict(pipeline_ui.EPC_INTEGRATED_OBSERVABLES_SOURCES)
        broken["relaxed"] = REPO_ROOT / "does_not_exist_s55_result.json"
        original = pipeline_ui.EPC_INTEGRATED_OBSERVABLES_SOURCES
        pipeline_ui.EPC_INTEGRATED_OBSERVABLES_SOURCES = broken
        try:
            payload = pipeline_ui.epc_integrated_observables_payload()
        finally:
            pipeline_ui.EPC_INTEGRATED_OBSERVABLES_SOURCES = original
        self.assertFalse(payload["available"])
        self.assertEqual(payload["missing"], ["relaxed"])
        self.assertNotIn("branches", payload)


if __name__ == "__main__":
    unittest.main()
