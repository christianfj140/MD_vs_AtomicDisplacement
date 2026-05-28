import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DOC_PATH = REPO_ROOT / "docs" / "graph2mat_deeph_benchmark.md"


class Graph2MatDeepHDocumentationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.doc = DOC_PATH.read_text(encoding="utf-8")
        self.readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        self.metrics = (REPO_ROOT / "Comparison" / "METRICS.md").read_text(encoding="utf-8")
        self.readme_flat = " ".join(self.readme.split())

    def test_required_deeph_artifacts_are_documented(self) -> None:
        for artifact in ("HSX", "STRUCT_OUT", "XV", "ORB_INDX", "TSHS", "TSDE"):
            self.assertIn(artifact, self.doc)

    def test_no_silent_siesta_repair_is_documented(self) -> None:
        self.assertIn("normal benchmark workflow must not silently rerun SIESTA", self.doc)
        self.assertIn("no hay reparacion SIESTA silenciosa", self.readme_flat)
        self.assertIn("must not trigger silent SIESTA repair", self.metrics)

    def test_one_pass_artifact_generation_is_documented(self) -> None:
        lower_doc = self.doc.lower()
        self.assertIn("one-pass artifact generation", lower_doc)
        self.assertIn("one-pass dataset", lower_doc)
        self.assertIn("una sola pasada SIESTA", self.readme_flat)

    def test_diagnostic_only_metrics_are_documented(self) -> None:
        self.assertIn("diagnostic_only", self.doc)
        self.assertIn("No robust winner", self.doc)
        self.assertIn("not exact DeepH local-frame", self.doc)
        self.assertIn("diagnostic_only", self.metrics)

    def test_readme_links_to_dedicated_guide(self) -> None:
        self.assertIn("docs/graph2mat_deeph_benchmark.md", self.readme)


if __name__ == "__main__":
    unittest.main()
