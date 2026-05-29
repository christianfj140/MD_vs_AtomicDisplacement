from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DOC_PATH = REPO_ROOT / "docs" / "graph2mat_deeph_benchmark.md"


class Graph2MatDeepHDocsTests(unittest.TestCase):
    def test_paper_ready_runbook_documents_required_claim_gates(self) -> None:
        doc = DOC_PATH.read_text(encoding="utf-8")

        for text in (
            "Paper-Ready Reviewer Runbook",
            "Required external artifacts",
            "Dataset freeze/verify",
            "DeepH raw/global equivalence preflight",
            "Test-blind search",
            "Validation-only top-k",
            "Final multi-seed plan",
            "Final statistics and Pareto",
            "Gate checker",
            "Allowed And Forbidden Claims",
            "Diagnostic-Only Examples",
            "Troubleshooting Blocked Gates",
            "DeepH paper numbers",
            "`ML_prediction.HSX` is never a reference",
            "H-MAE alone",
            "raw_global_equivalence_evidence.json",
            "not_a_scientific_run",
        ):
            self.assertIn(text, doc)

    def test_documented_paper_ready_scripts_expose_expected_flags(self) -> None:
        commands = {
            "Comparison/scripts/g2m_deeph_verify_protocol_datasets.py": ("--protocol", "--output", "--strict"),
            "Comparison/scripts/g2m_deeph_final_workflow.py": ("--stage", "--workflow-root", "--verify-datasets"),
            "Comparison/scripts/deeph_raw_global_equivalence_preflight.py": (
                "--frozen-split-manifest",
                "--deeph-processed-dir",
                "--fail-closed",
            ),
            "Comparison/scripts/g2m_deeph_gate_check.py": ("--protocol", "--workflow-root", "--output"),
            "Comparison/scripts/g2m_deeph_release_manifest.py": ("--dataset-root", "--workflow-root", "--strict"),
            "Comparison/scripts/g2m_deeph_final_stats.py": ("--metric", "--expected-seeds", "--min-final-seeds"),
            "Comparison/scripts/g2m_deeph_report.py": ("--metric", "--final-statistics", "--gate-status"),
            "Comparison/scripts/g2m_deeph_smoke.py": ("--paper-workflow-dry-run", "--output-dir"),
        }

        for script, flags in commands.items():
            result = subprocess.run(
                [sys.executable, str(REPO_ROOT / script), "--help"],
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr + result.stdout)
            for flag in flags:
                self.assertIn(flag, result.stdout, script)


if __name__ == "__main__":
    unittest.main()
