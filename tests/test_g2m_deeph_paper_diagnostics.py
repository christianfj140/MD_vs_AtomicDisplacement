import csv
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

diagnostics = importlib.import_module("g2m_deeph_paper_diagnostics")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class G2MDeepHPaperDiagnosticsTests(unittest.TestCase):
    def test_best_median_worst_from_dos_metric(self) -> None:
        rows = [
            {"sample": "s0", "dos_mae_500_fermi_window": 0.3},
            {"sample": "s1", "dos_mae_500_fermi_window": 0.1},
            {"sample": "s2", "dos_mae_500_fermi_window": 0.2},
        ]

        ranked = diagnostics.best_median_worst(rows, "dos_mae_500_fermi_window")

        self.assertEqual([row["rank_label"] for row in ranked], ["best", "median", "worst"])
        self.assertEqual([row["sample"] for row in ranked], ["s1", "s2", "s0"])

    def test_linear_regression_summary(self) -> None:
        summary = diagnostics.linear_regression_summary([1.0, 2.0, 3.0], [1.1, 1.9, 3.2])

        self.assertEqual(summary["n"], 3)
        self.assertGreater(summary["r2"], 0.95)
        self.assertAlmostEqual(summary["mae"], (0.1 + 0.1 + 0.2) / 3.0)

    def test_loads_winner_rows_and_dos_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "workflow"
            metrics = root / "runs" / "run_a" / "sweep" / "graph2mat" / "dataset" / "G2M-T600-26_seed1" / "metrics"
            write_csv(
                metrics / "kpoint_dos_metrics.csv",
                [
                    {"sample": "s0", "dos_mae_500_fermi_window": 0.2, "dos_wasserstein_eV": 0.4},
                    {"sample": "s1", "dos_mae_500_fermi_window": 0.1, "dos_wasserstein_eV": 0.3},
                ],
            )
            write_csv(
                root / "runs" / "run_a" / "sweep" / "training_sweep_metrics.csv",
                [
                    {
                        "model": "graph2mat",
                        "dataset_id": "graphene_w90_phase1_iid600",
                        "selected_config_id": "G2M-T600-26",
                        "seed": 3001,
                        "final_test_metrics_path": str(metrics / "manifest.json"),
                    },
                    {
                        "model": "deeph",
                        "dataset_id": "graphene_w90_phase1_iid600",
                        "selected_config_id": "DH-T600-04",
                        "seed": 3001,
                        "final_test_metrics_path": "",
                    },
                ],
            )

            rows = diagnostics.load_final_rows(root, "run_a")
            winners = diagnostics.selected_winner_rows(rows, "iid600")
            dos = diagnostics.load_dos_sample_metrics(winners)

            self.assertEqual(len(winners), 1)
            self.assertEqual(len(dos), 2)
            self.assertEqual(dos[0]["model"], "graph2mat")

    def test_band_residual_loader_reads_existing_csv_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            band_root = Path(tmp) / "bands"
            out = band_root / "gkm_fdf_dirac_diagnostic_1" / "iid600_md_1_native_bands_ref"
            write_csv(
                out / "band_errors_graph2mat.csv",
                [
                    {"sample_id": "s", "k_index": 0, "band_index": 0, "error_eV": 0.1, "abs_error_eV": 0.1},
                    {"sample_id": "s", "k_index": 1, "band_index": 0, "error_eV": -0.2, "abs_error_eV": 0.2},
                ],
            )
            write_csv(out / "band_errors_deeph.csv", [{"sample_id": "s", "k_index": 0, "band_index": 0, "error_eV": 0.05}])

            rows = diagnostics.load_band_residual_rows(band_root)

            self.assertEqual(len(rows), 3)
            self.assertEqual({row["model"] for row in rows}, {"graph2mat", "deeph"})

    def test_band_residual_loader_prefers_corrected_band_csv_convention(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            band_root = Path(tmp) / "bands"
            out = band_root / "gkm_fdf_dirac_diagnostic_1" / "iid600_md_1_native_bands_ref"
            write_csv(
                out / "bands_siesta.csv",
                [
                    {"sample_id": "s", "k_index": 0, "band_index": 0, "energy_eV": -6.0, "energy_aligned_eV": -0.3},
                ],
            )
            write_csv(
                out / "bands_graph2mat.csv",
                [
                    {"sample_id": "s", "k_index": 0, "band_index": 0, "energy_eV": -0.25, "energy_aligned_eV": 5.45},
                ],
            )
            write_csv(
                out / "bands_deeph.csv",
                [
                    {"sample_id": "s", "k_index": 0, "band_index": 0, "energy_eV": -0.28, "energy_aligned_eV": 5.42},
                ],
            )
            write_csv(
                out / "band_errors_graph2mat.csv",
                [{"sample_id": "s", "k_index": 0, "band_index": 0, "error_eV": 5.75}],
            )

            rows = diagnostics.load_band_residual_rows(band_root)

            self.assertEqual(len(rows), 2)
            by_model = {row["model"]: row for row in rows}
            self.assertAlmostEqual(by_model["graph2mat"]["error_eV"], 0.05)
            self.assertAlmostEqual(by_model["deeph"]["error_eV"], 0.02)

    def test_matrix_summary_uses_weighted_sample_rows_only(self) -> None:
        rows = [
            {"sample": "s0", "row_type": "weighted_sample", "h_mae_eV": 0.01},
            {"sample": "s0", "row_type": "per_k", "h_mae_eV": 9.0},
            {"sample": "s0", "row_type": "per_k", "h_mae_eV": 10.0},
        ]

        summary_rows = diagnostics.matrix_summary_source_rows(rows)

        self.assertEqual(len(summary_rows), 1)
        self.assertEqual(summary_rows[0]["h_mae_eV"], 0.01)

    def test_pareto_diagonal_guides_draw_x(self) -> None:
        class FakeAxis:
            def __init__(self) -> None:
                self.calls = []
                self.xlim = [0.2, 2.0]
                self.ylim = [0.1, 1.0]

            def set_xlim(self, left=None, right=None) -> None:
                if left is not None:
                    self.xlim[0] = left
                if right is not None:
                    self.xlim[1] = right

            def set_ylim(self, bottom=None, top=None) -> None:
                if bottom is not None:
                    self.ylim[0] = bottom
                if top is not None:
                    self.ylim[1] = top

            def get_xlim(self):
                return tuple(self.xlim)

            def get_ylim(self):
                return tuple(self.ylim)

            def plot(self, xs, ys, **kwargs) -> None:
                self.calls.append((xs, ys, kwargs))

        ax = FakeAxis()
        diagnostics.add_pareto_diagonal_guides(ax)

        self.assertEqual(len(ax.calls), 2)
        self.assertEqual(ax.calls[0][0], [0.0, 2.0])
        self.assertEqual(ax.calls[0][1], [0.0, 1.0])
        self.assertEqual(ax.calls[1][0], [0.0, 2.0])
        self.assertEqual(ax.calls[1][1], [1.0, 0.0])

    def test_build_diagnostics_manifest_with_minimal_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            for dataset_key, dataset_id, g_cfg, d_cfg, run_name in [
                ("iid600", "graphene_w90_phase1_iid600", "G2M-T600-26", "DH-T600-13", "run600"),
                ("iid1000", "graphene_w90_phase1_iid1000", "G2M-T1000-03", "DH-T1000-03", "run1000"),
            ]:
                root = base / dataset_key
                rows = []
                for model, cfg in [("graph2mat", g_cfg), ("deeph", d_cfg)]:
                    metrics_dir = root / "runs" / run_name / "sweep" / model / dataset_id / f"{cfg}_seed3001" / "metrics"
                    write_csv(
                        metrics_dir / "kpoint_dos_metrics.csv",
                        [{"sample": "s0", "dos_mae_500_fermi_window": 0.1, "dos_wasserstein_eV": 0.2}],
                    )
                    write_csv(
                        metrics_dir / "kpoint_matrix_metrics.csv",
                        [{"sample": "s0", "row_type": "weighted_sample", "h_mae_eV": 0.01, "h_rmse_eV": 0.02, "relative_frobenius": 0.03}],
                    )
                    if model == "graph2mat":
                        write_csv(
                            metrics_dir / "orbital_pair_metrics.csv",
                            [
                                {
                                    "sample": "s0",
                                    "row_orbital_label": "2pz",
                                    "col_orbital_label": "2pz",
                                    "mae_union_meV": 5.0,
                                    "rmse_union_eV": 0.01,
                                    "r2_union": 0.9,
                                }
                            ],
                        )
                    rows.append(
                        {
                            "model": model,
                            "dataset_id": dataset_id,
                            "selected_config_id": cfg,
                            "seed": 3001,
                            "final_test_metrics_path": str(metrics_dir / "manifest.json"),
                            "final_test_metrics": json.dumps(
                                {
                                    "low_energy_rmse_eV": 0.1,
                                    "fermi_window_rmse_eV": 0.2,
                                    "dos_mae_500_fermi_window": 0.1,
                                    "dos_wasserstein_eV": 0.2,
                                }
                            ),
                            "telemetry": json.dumps({"gpu_hours_total": 0.5}),
                        }
                    )
                write_csv(root / "runs" / run_name / "sweep" / "training_sweep_metrics.csv", rows)

            band_root = base / "bands"
            for dataset_key in ["iid600", "iid1000"]:
                out = band_root / "gkm_fdf_dirac_diagnostic_1" / f"{dataset_key}_md_1_native_bands_ref"
                write_csv(out / "band_errors_graph2mat.csv", [{"sample_id": "s", "k_index": 0, "band_index": 0, "error_eV": 0.1}])
                write_csv(out / "band_errors_deeph.csv", [{"sample_id": "s", "k_index": 0, "band_index": 0, "error_eV": 0.05}])
                (out / "dirac_diagnostic.json").write_text(
                    json.dumps(
                        {
                            "methods": {
                                "siesta": {"method": "SIESTA", "gap_eV": 0.1, "dirac_minus_fermi_eV": 0.0, "status": "ok"},
                                "graph2mat": {"method": "Graph2Mat", "gap_eV": 0.11, "dirac_minus_fermi_eV": 0.01, "status": "ok"},
                                "deeph": {"method": "DeepH", "gap_eV": 0.09, "dirac_minus_fermi_eV": 0.02, "status": "ok"},
                            }
                        }
                    ),
                    encoding="utf-8",
                )

            output_dir = base / "out"
            args = type(
                "Args",
                (),
                {
                    "iid600_root": base / "iid600",
                    "iid1000_root": base / "iid1000",
                    "run_iid600": "run600",
                    "run_iid1000": "run1000",
                    "band_root": band_root,
                    "output_dir": output_dir,
                    "formats": "png",
                },
            )()

            manifest = diagnostics.build_diagnostics(args)

            self.assertIn(manifest["status"], {"ok", "tables_only"})
            self.assertTrue((output_dir / "paper_diagnostics_manifest.json").exists())
            self.assertTrue((output_dir / "dos_summary.csv").exists())
            self.assertTrue((output_dir / "paper_diagnostics_summary.md").exists())

    def test_script_does_not_import_subprocess_or_build_training_commands(self) -> None:
        source = (SCRIPTS_DIR / "g2m_deeph_paper_diagnostics.py").read_text(encoding="utf-8")

        self.assertNotIn("import subprocess", source)
        self.assertNotIn("os.system", source)
        self.assertNotIn("Popen", source)
        self.assertIn("FORBIDDEN_COMPUTE_COMMANDS", source)


if __name__ == "__main__":
    unittest.main()
