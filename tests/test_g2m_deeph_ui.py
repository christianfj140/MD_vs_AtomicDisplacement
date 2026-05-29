import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class Graph2MatDeepHUITests(unittest.TestCase):
    def setUp(self) -> None:
        self.index_html = (REPO_ROOT / "Comparison" / "ui" / "index.html").read_text(encoding="utf-8")
        self.app_js = (REPO_ROOT / "Comparison" / "ui" / "app.js").read_text(encoding="utf-8")
        self.styles_css = (REPO_ROOT / "Comparison" / "ui" / "styles.css").read_text(encoding="utf-8")

    def test_new_sidebar_tab_exists_without_removing_existing_tabs(self) -> None:
        self.assertIn('data-view="g2m-deeph"', self.index_html)
        self.assertIn("G2M vs DeepH", self.index_html)
        for view in ("experiment", "results", "performance", "api"):
            self.assertIn(f'data-view="{view}"', self.index_html)

    def test_dedicated_view_and_primary_buttons_exist(self) -> None:
        self.assertIn('id="view-g2m-deeph" class="view"', self.index_html)
        self.assertIn('id="g2m-deeph-validate"', self.index_html)
        self.assertIn("Validate dataset artifacts", self.index_html)
        self.assertIn('id="g2m-deeph-run"', self.index_html)
        self.assertIn("Run Graph2Mat vs DeepH benchmark", self.index_html)
        self.assertIn('id="g2m-deeph-stop"', self.index_html)

    def test_repair_mode_is_explicitly_marked_slow_and_expensive(self) -> None:
        self.assertIn('value="repair_expensive"', self.index_html)
        self.assertIn("slow/expensive", self.index_html)
        self.assertIn("rerun SIESTA per missing snapshot", self.index_html)

    def test_required_controls_are_present(self) -> None:
        for control_id in (
            "g2m-deeph-material-preset",
            "g2m-deeph-dataset-mode",
            "g2m-deeph-dataset-root",
            "g2m-deeph-refresh-datasets",
            "g2m-deeph-dataset-picker-list",
            "g2m-deeph-snapshot-count",
            "g2m-deeph-split-train",
            "g2m-deeph-g2m-epochs",
            "g2m-deeph-g2m-lr",
            "g2m-deeph-deeph-epochs",
            "g2m-deeph-deeph-lr",
            "g2m-deeph-artifact-summary",
            "g2m-deeph-metric-summary",
            "g2m-deeph-log",
            "g2m-deeph-log-bottom",
            "g2m-deeph-log-clear",
            "g2m-deeph-plots",
        ):
            self.assertIn(f'id="{control_id}"', self.index_html)
        self.assertIn('value="Comparison/datasets/graphene_w90_joint"', self.index_html)
        self.assertIn("separado de results/ y workspaces/", self.index_html)
        self.assertIn("Available joint datasets for training / reuse", self.index_html)
        self.assertIn("Selecciona un dataset joint ya validado", self.index_html)
        self.assertIn("Full strict pipeline: generate + train/test sweep", self.index_html)

    def test_dataset_sweep_controls_are_in_g2m_deeph_tab(self) -> None:
        for control_id in (
            "g2m-deeph-dataset-sweep-max",
            "g2m-deeph-md-dataset-editor",
            "g2m-deeph-md-add-dataset",
            "g2m-deeph-md-sweep-table",
            "g2m-deeph-dataset-sweep-preview",
            "g2m-deeph-split-mode",
        ):
            self.assertIn(f'id="{control_id}"', self.index_html)
        self.assertIn("MD dataset builder", self.index_html)
        self.assertIn("Es el mismo esquema MD del panel Experiment", self.index_html)
        self.assertIn("snapshots | temperature_K", self.index_html)
        self.assertIn("artefactos joint Graph2Mat+DeepH", self.index_html)

    def test_dataset_sweep_controls_are_not_added_to_experiment_tab(self) -> None:
        experiment_html = self.index_html.split('<section id="view-g2m-deeph" class="view">', 1)[0]
        self.assertNotIn('id="g2m-deeph-md-dataset-editor"', experiment_html)
        self.assertNotIn('id="g2m-deeph-md-sweep-table"', experiment_html)

    def test_training_sweep_controls_are_in_g2m_deeph_tab_only(self) -> None:
        for control_id in (
            "g2m-deeph-training-sweep-enabled",
            "g2m-deeph-training-sweep-max-runs",
            "g2m-deeph-sweep-common-epochs",
            "g2m-deeph-sweep-common-lr",
            "g2m-deeph-sweep-g2m-hidden-channels",
            "g2m-deeph-sweep-g2m-batch-size",
            "g2m-deeph-sweep-deeph-atom-fea-len",
            "g2m-deeph-training-sweep-preview",
        ):
            self.assertIn(f'id="{control_id}"', self.index_html)
        experiment_html = self.index_html.split('<section id="view-g2m-deeph" class="view">', 1)[0]
        self.assertNotIn('id="g2m-deeph-training-sweep-enabled"', experiment_html)

    def test_javascript_wires_new_backend_endpoints(self) -> None:
        for endpoint in (
            "/api/g2m-deeph/validate-dataset",
            "/api/g2m-deeph/run",
            "/api/g2m-deeph/stop",
            "/api/g2m-deeph/status",
            "/api/g2m-deeph/datasets",
            "/api/g2m-deeph/logs",
            "/api/g2m-deeph/results",
            "/api/g2m-deeph/plots",
        ):
            self.assertIn(endpoint, self.app_js)
        self.assertIn("g2mDeephPayload", self.app_js)
        self.assertIn("pollG2MDeepHLogs", self.app_js)
        self.assertIn("formatG2MDeepHValidationError", self.app_js)
        self.assertIn("validateG2MDeepHDataset();", self.app_js)
        self.assertIn("loadG2MDeepHDatasets", self.app_js)
        self.assertIn("g2m-deeph-dataset-checkbox", self.app_js)
        self.assertIn('datasetMode !== "repair_expensive"', self.app_js)
        self.assertIn('"full_strict_pipeline"', self.app_js)
        self.assertIn("const performance = performanceSettings();", self.app_js)
        self.assertIn("compute_accelerator: performance.compute_accelerator", self.app_js)
        self.assertIn("performance,", self.app_js)
        self.assertIn("max_parallel_graph2mat_training_jobs", self.app_js)
        self.assertIn("max_parallel_deeph_training_jobs", self.app_js)
        self.assertIn("G2M_DEEPH_LIVE_METRICS_URL", self.app_js)
        self.assertIn("/api/g2m-deeph/live-plots", self.app_js)
        self.assertIn("maybeLoadG2MDeepHLiveMetrics", self.app_js)
        self.assertIn("mergeG2MDeepHLivePlotPayload", self.app_js)
        self.assertIn("G2M_DEEPH_LIVE_PLOT_REFRESH_MS", self.app_js)
        self.assertIn("maybeRefreshG2MDeepHLivePlots", self.app_js)
        self.assertIn("g2mDeephPlotsInFlight", self.app_js)
        self.assertIn("Live Graph2Mat/DeepH metrics", self.app_js)
        self.assertIn("graph2mat_log_every_n_steps", self.app_js)
        self.assertIn("graph2mat_check_val_every_n_epoch", self.app_js)
        self.assertIn("graph2mat_checkpoint_every_n_epochs", self.app_js)
        self.assertIn("graph2mat_require_cuequivariance", self.app_js)
        self.assertIn("torch_mixed_precision", self.app_js)
        self.assertIn('id="performance-max-parallel-graph2mat-training-jobs"', self.index_html)
        self.assertIn('id="performance-max-parallel-deeph-training-jobs"', self.index_html)
        self.assertIn('id="performance-graph2mat-log-every-n-steps"', self.index_html)
        self.assertIn('id="performance-graph2mat-check-val-every-n-epoch"', self.index_html)
        self.assertIn('id="performance-graph2mat-checkpoint-every-n-epochs"', self.index_html)
        self.assertIn('id="performance-graph2mat-require-cuequivariance"', self.index_html)
        self.assertIn('id="g2m-deeph-training-sweep-status"', self.index_html)
        self.assertIn('id="performance-torch-mixed-precision"', self.index_html)
        self.assertIn('<option value="parallel_trains">Parallel trains</option>', self.index_html)
        self.assertNotIn("generacion de dataset joint desde este boton aun no esta implementada", self.app_js)

    def test_javascript_wires_dataset_sweep_payload_and_preview(self) -> None:
        for token in (
            "g2mDeephDatasetSweepPayload",
            "g2mDeephDatasetSweepRecipes",
            "renderG2MDeepHDatasetSweepPreview",
            "generate_datasets_only",
            "full_strict_pipeline",
            "dataset_sweep",
            "g2m_deeph_md",
            "g2m-deeph-md-sweep-table",
        ):
            self.assertIn(token, self.app_js)

    def test_javascript_wires_training_sweep_payload_and_preview(self) -> None:
        for token in (
            "g2mDeephTrainingSweepPayload",
            "renderG2MDeepHTrainingSweepPreview",
            "g2mDeephTrainingSweepPayload(performance)",
            "performance?.batch_size",
            "training_sweep: trainingSweep",
            "Training sweep: primero genera/valida",
            "No SIESTA generation in training sweep",
        ):
            self.assertIn(token, self.app_js)

    def test_styles_for_phase_progress_exist(self) -> None:
        self.assertIn(".phase-progress", self.styles_css)
        self.assertIn(".phase-chip.active", self.styles_css)
        self.assertIn(".summary-row", self.styles_css)
        self.assertIn(".terminal-navbar", self.styles_css)
        self.assertIn(".g2m-deeph-terminal .log-output", self.styles_css)
        self.assertIn("max-height: 1560px", self.styles_css)
        self.assertIn(".g2m-deeph-scroll-table-wrap", self.styles_css)
        self.assertIn("max-height: min(230px, 26vh)", self.styles_css)
        self.assertIn("position: sticky", self.styles_css)

    def test_final_summary_tables_and_scientific_status_rendering_exist(self) -> None:
        for text in (
            "Artifact completeness",
            "Final recommendation",
            "Hamiltonian MAE",
            "Hamiltonian RMSE",
            "Hamiltonian MSE",
            "Sparse support R2",
            "Relative Frobenius comparison",
            "Predicted Hamiltonian hermiticity",
            "Global spectral RMSE",
            "Low-energy spectral RMSE",
            "Fermi-window spectral RMSE",
            "Frontier-window spectral RMSE",
            "DOS Fermi-window MAE",
            "DOS Wasserstein distance",
            "Phase timing",
            "No robust winner",
            "Ranking recommendation",
            "Best Graph2Mat / DeepH runs",
            "Pairwise Graph2Mat vs DeepH",
            "Accuracy-vs-time Pareto",
            "Scientific gates",
            "Adapter equivalence",
            "DeepH split audit",
        ):
            self.assertIn(text, self.app_js)
        self.assertIn('status === "diagnostic_only"', self.app_js)
        self.assertIn('status === "no_robust_winner"', self.app_js)
        self.assertIn(".comparison-status-banner.diagnostic", self.styles_css)
        self.assertIn(".comparison-status-banner.invalid", self.styles_css)

    def test_plot_payload_renders_grouped_bar_plots(self) -> None:
        self.assertIn("renderG2MDeepHGroupedBarPlot", self.app_js)
        self.assertIn("renderG2MDeepHTimingScalingPlot", self.app_js)
        self.assertIn("renderG2MDeepHMetricScalingPlot", self.app_js)
        self.assertIn("normalizeG2MDeepHMetricPlots", self.app_js)
        self.assertIn("g2mDeephReadableMetricGroups", self.app_js)
        self.assertIn("grouped_bar", self.app_js)
        self.assertIn("timing_scaling", self.app_js)
        self.assertIn("metric_scaling", self.app_js)
        self.assertIn("spectral_fermi", self.app_js)
        self.assertIn("DOS Wasserstein distance", self.app_js)
        self.assertIn("Metrics vs dataset size", self.app_js)
        self.assertIn("Timing vs dataset size", self.app_js)
        self.assertIn("g2m-deeph-plot-", self.app_js)
        self.assertIn("g2mDeephMarkerSymbol", self.app_js)
        self.assertIn("g2m-deeph-scroll-table-wrap", self.app_js)
        self.assertIn("g2mDeephIntegerValue", self.app_js)
        self.assertIn('label: "Snapshots", format: g2mDeephIntegerValue', self.app_js)
        self.assertIn('"triangle-up"', self.app_js)
        self.assertIn('"circle"', self.app_js)

    def test_plotly_science_style_theme_is_applied_without_matplotlib(self) -> None:
        self.assertIn("SCIENCE_PLOT_FONT_FAMILY", self.app_js)
        self.assertIn("sciencePlotLayout", self.app_js)
        self.assertIn("sciencePlotTrace", self.app_js)
        self.assertIn("SCIENCE_PLOT_GRID_COLOR", self.app_js)
        self.assertIn('format: "svg"', self.app_js)
        self.assertIn("#4477aa", self.app_js)
        self.assertNotIn("matplotlib", self.app_js.lower())
        self.assertIn("border-radius: 6px", self.styles_css)
        self.assertIn("background: #ffffff", self.styles_css)


if __name__ == "__main__":
    unittest.main()
