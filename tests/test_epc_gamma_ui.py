"""C21 / E-F_001-S28: the GO-4 graphene-Gamma adjudication is published, not recomputed.

The endpoint must read the canonical artifacts and pass their numbers through unchanged; the
frontend must only format them. Absence is never a PASS.
"""

import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "Comparison" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

pipeline_ui = importlib.import_module("pipeline_ui")


def artifacts_present() -> bool:
    return all(path.exists() for path in pipeline_ui.EPC_GAMMA_SOURCES.values())


class EpcGammaPayloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not artifacts_present():
            raise unittest.SkipTest("GO-4 graphene-Gamma artifacts are not in this checkout")
        cls.payload = pipeline_ui.epc_graphene_gamma_payload()
        cls.verdict = json.loads(
            pipeline_ui.EPC_GAMMA_SOURCES["verdict"].read_text(encoding="utf-8")
        )
        cls.summary = json.loads(
            pipeline_ui.EPC_GAMMA_SOURCES["summary"].read_text(encoding="utf-8")
        )
        cls.g_blocks = json.loads(
            pipeline_ui.EPC_GAMMA_SOURCES["g_blocks"].read_text(encoding="utf-8")
        )

    def test_verdict_and_levels_come_from_the_artifact(self) -> None:
        self.assertTrue(self.payload["available"])
        self.assertEqual(self.payload["verdict"], self.verdict["verdict"])
        self.assertEqual(self.payload["levels"], self.verdict["levels"])
        self.assertEqual(self.payload["label"], self.verdict["label"])
        self.assertEqual(self.payload["gates"], self.verdict["gates"])
        self.assertEqual(self.payload["frozen_checks"], self.verdict["frozen_checks"])
        self.assertEqual(self.payload["GO4_finite_PAO"], self.verdict["GO4_finite_PAO"])
        self.assertEqual(self.payload["full_KS"], self.verdict["full_KS"])

    def test_formalism_backends_and_taus_are_exposed(self) -> None:
        self.assertEqual(self.payload["formalism_id"], self.verdict["formalism_id"])
        tolerances = self.payload["tolerances"]
        self.assertTrue(tolerances["tau_num"])
        self.assertTrue(tolerances["tau_backend"])
        self.assertEqual(tolerances["tau_model"], self.summary["tolerances"]["tau_model"])
        for block in self.payload["g_blocks"]:
            for field in pipeline_ui.EPC_GAMMA_BACKEND_FIELDS:
                self.assertIn(field, block["backends"])
            self.assertIsNotNone(block["backends"]["epc_backend_class"])

    def test_gauge_invariant_g_metrics_are_passed_through_unchanged(self) -> None:
        reports = {report["label"]: report for report in self.g_blocks["reports"]}
        self.assertTrue(reports)
        for block in self.payload["g_blocks"]:
            report = reports[block["label"]]
            self.assertEqual(block["g"]["gauge_invariant"], report["g"]["gauge_invariant"])
            self.assertEqual(
                block["g_per_unit_displacement"]["gauge_invariant"],
                report["g_per_unit_displacement"]["gauge_invariant"],
            )
            self.assertEqual(block["hellmann_feynman"], report["hellmann_feynman"])
            self.assertEqual(block["mode"]["artifact_kind"], report["mode"]["artifact_kind"])

    def test_tau_rows_only_split_the_compound_key(self) -> None:
        source = self.summary["tolerances"]["tau_num"]
        rows = {row["key"]: row for row in self.payload["tolerances"]["tau_num"]}
        self.assertEqual(set(rows), set(source))
        for key, row in rows.items():
            self.assertEqual(row["tau_num"], source[key]["tau_num"])
            self.assertEqual(f"{row['path']}|{row['direction']}|{row['k']}", key)

    def test_siesta_provenance_distinguishes_source_from_binary_only(self) -> None:
        provenance = self.payload["provenance"]
        classification = provenance["siesta_runtime_classification"]
        self.assertIn(classification, {"VERIFIED_SOURCE", "VERIFIED_BINARY_ONLY"})
        self.assertEqual(
            provenance["siesta_source_verified"], classification == "VERIFIED_SOURCE"
        )
        self.assertTrue(provenance["artifacts"])
        for artifact in provenance["artifacts"]:
            self.assertEqual(len(artifact["sha256"] or ""), 64)

    def test_missing_artifact_is_never_reported_as_available(self) -> None:
        broken = dict(pipeline_ui.EPC_GAMMA_SOURCES)
        broken["verdict"] = REPO_ROOT / "does_not_exist_go4_verdict.json"
        original = pipeline_ui.EPC_GAMMA_SOURCES
        pipeline_ui.EPC_GAMMA_SOURCES = broken
        try:
            payload = pipeline_ui.epc_graphene_gamma_payload()
        finally:
            pipeline_ui.EPC_GAMMA_SOURCES = original
        self.assertFalse(payload["available"])
        self.assertEqual(payload["missing"], ["verdict"])
        self.assertNotIn("verdict", payload.get("levels", {}) or {})


class EpcGammaFrontendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.index_html = (REPO_ROOT / "Comparison" / "ui" / "index.html").read_text(encoding="utf-8")
        cls.app_js = (REPO_ROOT / "Comparison" / "ui" / "app.js").read_text(encoding="utf-8")

    def test_tab_and_view_are_wired_into_the_existing_ui(self) -> None:
        self.assertIn('data-view="epc-gamma"', self.index_html)
        self.assertIn('id="view-epc-gamma"', self.index_html)
        for view in ("experiment", "results", "g2m-deeph", "cross-testing"):
            self.assertIn(f'data-view="{view}"', self.index_html)
        self.assertIn('tab.dataset.view === "epc-gamma"', self.app_js)
        self.assertIn('request("/api/epc/graphene-gamma")', self.app_js)
        self.assertIn('getElementById("epc-gamma-refresh")', self.app_js)

    def test_every_rendered_container_exists_in_the_markup(self) -> None:
        for element_id in (
            "epc-gamma-verdict-title",
            "epc-gamma-chips",
            "epc-gamma-identity",
            "epc-gamma-model-statement",
            "epc-gamma-claim",
            "epc-gamma-gates-chart",
            "epc-gamma-tau-model-chart",
            "epc-gamma-paths-chart",
            "epc-gamma-gates",
            "epc-gamma-frozen-checks",
            "epc-gamma-tau-num",
            "epc-gamma-tau-backend",
            "epc-gamma-tau-model",
            "epc-gamma-g-blocks",
            "epc-gamma-provenance",
            "epc-gamma-contracts",
            "epc-gamma-artifacts",
        ):
            self.assertIn(f'id="{element_id}"', self.index_html)
            self.assertIn(f'"{element_id}"', self.app_js)

    def test_frontend_does_no_physics(self) -> None:
        start = self.app_js.index("// ===== EPC graphene Gamma")
        end = self.app_js.index("function setupTabs()", start)
        section = self.app_js[start:end]
        for forbidden in ("Math.sqrt", "Math.exp", "Math.cos", "Math.sin", "fourier", "**"):
            self.assertNotIn(forbidden, section)

    def test_epc_precision_has_its_own_screen(self) -> None:
        self.assertIn('data-view="epc-scaling"', self.index_html)
        self.assertIn('id="view-epc-scaling"', self.index_html)
        self.assertIn('request("/api/epc/snapshot-scaling")', self.app_js)
        self.assertIn('id="epc-scaling-family"', self.index_html)
        self.assertIn('id="epc-scaling-deeph-plots"', self.index_html)
        self.assertIn('payload.paired_comparison', self.app_js)


class EpcSnapshotScalingPayloadTests(unittest.TestCase):
    def test_endpoint_reads_the_generated_plot_payload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            original = pipeline_ui.RESULTS_ROOT
            pipeline_ui.RESULTS_ROOT = Path(directory)
            path = Path(directory) / "ui_real_metrics_derivatives/epc/snapshot_scaling_20260908/derivative_metrics/summary/derivative_plots/derivative_plot_payload.json"
            path.parent.mkdir(parents=True)
            expected = {"available": True, "plots": [{"id": "dh_mae_vs_dataset_size", "rows": []}]}
            path.write_text(json.dumps(expected), encoding="utf-8")
            try:
                self.assertEqual(pipeline_ui.epc_snapshot_scaling_payload(), expected)
            finally:
                pipeline_ui.RESULTS_ROOT = original

    def test_endpoint_adds_existing_paired_deeph_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            original = pipeline_ui.RESULTS_ROOT
            pipeline_ui.RESULTS_ROOT = Path(directory)
            primary = Path(directory) / "ui_real_metrics_derivatives/epc/snapshot_scaling_20260908/derivative_metrics/summary/derivative_plots/derivative_plot_payload.json"
            paired = Path(directory) / "ui_real_metrics_derivatives/derivative_metrics/summary/derivative_plots/derivative_plot_payload.json"
            primary.parent.mkdir(parents=True)
            paired.parent.mkdir(parents=True)
            primary.write_text(json.dumps({"available": True, "plots": []}), encoding="utf-8")
            paired.write_text(json.dumps({"plots": [
                {"id": "dh_mae_vs_dataset_size", "rows": [{"model": "deeph"}]},
                {"id": "unrelated", "rows": []},
            ]}), encoding="utf-8")
            try:
                result = pipeline_ui.epc_snapshot_scaling_payload()
            finally:
                pipeline_ui.RESULTS_ROOT = original
            self.assertTrue(result["paired_comparison"]["diagnostic_only"])
            self.assertEqual([plot["id"] for plot in result["paired_comparison"]["plots"]], ["dh_mae_vs_dataset_size"])


if __name__ == "__main__":
    unittest.main()
