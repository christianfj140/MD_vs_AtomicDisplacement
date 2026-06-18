import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DOC_PATH = REPO_ROOT / "docs" / "graph2mat_deeph_benchmark.md"


class Graph2MatDeepHDocumentationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.doc = DOC_PATH.read_text(encoding="utf-8")
        self.readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        self.metrics = (REPO_ROOT / "Comparison" / "METRICS.md").read_text(encoding="utf-8")
        self.workflows = (REPO_ROOT / "docs" / "workflows.md").read_text(encoding="utf-8")
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

    def test_derivative_reference_and_non_reference_are_documented(self) -> None:
        for text in (
            "(H_SIESTA(R + delta) - H_SIESTA(R - delta)) / (2 * delta)",
            "RUN.fdf",
            "metadata.json",
            "ML_prediction.HSX",
            "ORB_INDX",
            "basis/gauge evidence",
            "internal_diagnostic",
            "technical_presentation",
            "paper_level_candidate",
            "blocked",
            "no force-constants comparison is implemented",
        ):
            self.assertIn(text, self.doc)
            self.assertIn(text, self.metrics)
        for text in (
            "derivative of Hamiltonian matrix",
            "Cartesian atomic displacement",
            "SIESTA force constants",
            "dynamical matrices",
            "phonons are not",
        ):
            self.assertIn(text, self.doc)
            self.assertIn(text, self.metrics)

    def test_modular_derivative_workflow_examples_are_documented(self) -> None:
        for text in (
            "workflow_mode",
            "hamiltonian_only",
            "derivative_stencils_only",
            "derivative_metrics_only",
            "h_then_derivative_postprocess",
            "h_then_derivative_full",
            "full_end_to_end",
            "/api/g2m-deeph/run",
            "source_dataset_root",
            "result_dir",
            "delta_ang",
            "[0.005, 0.01, 0.02]",
            "atoms",
            "axes",
            "graph2mat_checkpoint",
            "deeph_model_dir",
            "graph2mat_existing_prediction_root",
            "deeph_existing_prediction_root",
        ):
            self.assertIn(text, self.workflows)
        for text in (
            "A stencil is the set of displaced geometries",
            "Finite differences are the formula applied to Hamiltonians",
            "central stencil with `R+` and `R-`",
            "MD snapshots are base geometries `R0`",
            "not derivative stencils",
            "same split",
            "delta stability",
            "basis/gauge/orbital-order compatibility",
            "reference-noise checks",
            "Graph2Mat checkpoint manifest",
            "DeepH save directory",
            "model-specific derivative result roots",
            "graph2mat_derivative_result",
            "deeph_derivative_result",
            "inside each model-specific derivative result root",
            "force constants",
            "not used as `dH/dR` references",
            "siesta_hamiltonians/<sample>/*.HSX|*.TSHS",
            "predicted_hamiltonians/<sample>/ML_prediction.HSX",
            "derivative_matrix_metrics.csv",
            "derivative_support_sweep.csv",
            "derivative_hermiticity.csv",
            "derivative_model_comparison_summary.json",
            "derivative_model_paired_comparison.csv",
        ):
            self.assertIn(text, self.workflows)

    def test_readme_links_to_dedicated_guide(self) -> None:
        self.assertIn("docs/graph2mat_deeph_benchmark.md", self.readme)


if __name__ == "__main__":
    unittest.main()
