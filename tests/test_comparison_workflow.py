from __future__ import annotations

import csv
import contextlib
import importlib.util
import json
import shutil
import subprocess
import sys
import unittest
import uuid
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_TMP_ROOT = REPO_ROOT / ".test_tmp"


@contextlib.contextmanager
def workspace_tempdir():
    TEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)
    path = TEST_TMP_ROOT / f"case_{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield str(path)
    finally:
        shutil.rmtree(path, ignore_errors=True)


def run_script(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=cwd or REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def minimal_run_fdf(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "SystemLabel water",
                "%block LatticeVectors",
                "10.0 0.0 0.0",
                "0.0 10.0 0.0",
                "0.0 0.0 10.0",
                "%endblock LatticeVectors",
                "%block AtomicCoordinatesAndAtomicSpecies",
                "0.0 0.0 0.0 1",
                "0.0 0.7 0.0 2",
                "0.0 -0.7 0.0 2",
                "%endblock AtomicCoordinatesAndAtomicSpecies",
                "",
            ]
        ),
        encoding="utf-8",
    )


def make_sample(root: Path, name: str, *, hamiltonian: bool = True, converged: bool = True) -> Path:
    sample_dir = root / name
    sample_dir.mkdir(parents=True)
    minimal_run_fdf(sample_dir / "RUN.fdf")
    if hamiltonian:
        (sample_dir / "siesta.TSHS").write_bytes(b"fake")
    run_out = "Job completed\n"
    if converged:
        run_out += "SCF cycle converged\n"
    sample_dir.joinpath("RUN.out").write_text(run_out, encoding="utf-8")
    return sample_dir


class ComparisonWorkflowTests(unittest.TestCase):
    def test_validate_sample_bundle_accepts_complete_sample(self) -> None:
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            make_sample(root / "samples", "001")
            output = root / "validation"
            result = run_script(
                "Comparison/scripts/validate_sample_bundle.py",
                "--samples-dir",
                str(root / "samples"),
                "--method",
                "md",
                "--output-dir",
                str(output),
                "--min-valid",
                "1",
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            summary = json.loads((output / "validation_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["valid_samples"], 1)
            self.assertEqual(summary["invalid_samples"], 0)

    def test_validate_sample_bundle_rejects_missing_hamiltonian_and_failed_scf(self) -> None:
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            make_sample(root / "samples", "missing_h", hamiltonian=False)
            make_sample(root / "samples", "failed_scf", converged=False)
            output = root / "validation"
            result = run_script(
                "Comparison/scripts/validate_sample_bundle.py",
                "--samples-dir",
                str(root / "samples"),
                "--method",
                "md",
                "--output-dir",
                str(output),
                "--min-valid",
                "2",
            )
            self.assertEqual(result.returncode, 2)
            with (output / "invalid_samples.csv").open(encoding="utf-8") as handle:
                invalid = list(csv.DictReader(handle))
            reasons = " ".join(row["invalid_reasons"] for row in invalid)
            self.assertIn("missing_matrix", reasons)
            self.assertIn("scf_not_converged", reasons)

    def test_strict_atomdisp_validation_rejects_missing_run_out_and_failed_scf(self) -> None:
        sys.path.insert(0, str(REPO_ROOT / "AtomDisplacement" / "scripts"))
        spec = importlib.util.spec_from_file_location(
            "atom_displacement_utils",
            REPO_ROOT / "AtomDisplacement" / "scripts" / "atom_displacement_utils.py",
        )
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with workspace_tempdir() as tmp:
            root = Path(tmp)
            missing_out = make_sample(root, "missing_out")
            (missing_out / "RUN.out").unlink()
            failed_scf = make_sample(root, "failed_scf", converged=False)
            valid = make_sample(root, "valid")

            self.assertFalse(module.validate_sample_dir(missing_out)["valid"])
            self.assertIn("missing_output", module.validate_sample_dir(missing_out)["validation_reason"])
            self.assertFalse(module.validate_sample_dir(failed_scf)["valid"])
            self.assertIn("scf_not_converged", module.validate_sample_dir(failed_scf)["validation_reason"])
            self.assertTrue(module.validate_sample_dir(valid)["valid"])

    def test_run_single_points_reruns_stale_matrix_instead_of_skipping(self) -> None:
        sys.path.insert(0, str(REPO_ROOT / "AtomDisplacement" / "scripts"))
        spec = importlib.util.spec_from_file_location(
            "run_single_points",
            REPO_ROOT / "AtomDisplacement" / "scripts" / "run_single_points.py",
        )
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with workspace_tempdir() as tmp:
            root = Path(tmp)
            sample = make_sample(root, "001")
            (sample / "RUN.out").unlink()
            calls: list[Path] = []

            module.generated_sample_dirs = lambda: [sample]
            module.require_command = lambda command: None
            module.DATASET_DIR = root
            module.PIPELINE_PATHS["run_summary_path"] = root / "run_summary.json"

            def fake_run_siesta(sample_dir: Path, output_path: Path) -> int:
                calls.append(sample_dir)
                output_path.write_text("Job completed\nSCF cycle converged\n", encoding="utf-8")
                return 0

            module.run_siesta_in_dir = fake_run_siesta
            argv = sys.argv
            try:
                sys.argv = ["run_single_points.py"]
                rc = module.main()
            finally:
                sys.argv = argv
            self.assertEqual(rc, 0)
            self.assertEqual(calls, [sample])
            summary = json.loads((root / "validation" / "validation_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["valid_samples"], 1)

    def test_build_common_tests_refuses_train_test_overlap(self) -> None:
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            md_sample = make_sample(root / "md", "001")
            atom_sample = make_sample(root / "atom", "001")
            md_test = root / "md_test.csv"
            atom_test = root / "atom_test.csv"
            train = root / "train.csv"
            write_csv(
                md_test,
                [
                    {
                        "sample_id": "md_001",
                        "method": "md",
                        "structure_path": str(md_sample / "RUN.fdf"),
                        "hamiltonian_path": str(md_sample / "siesta.TSHS"),
                        "status": "valid",
                    }
                ],
            )
            write_csv(
                atom_test,
                [
                    {
                        "sample_id": "atomdisp_001",
                        "method": "atom_displacement",
                        "structure_path": str(atom_sample / "RUN.fdf"),
                        "hamiltonian_path": str(atom_sample / "siesta.TSHS"),
                        "status": "valid",
                    }
                ],
            )
            write_csv(
                train,
                [
                    {
                        "sample_id": "md_001",
                        "method": "md",
                        "structure_path": str(md_sample / "RUN.fdf"),
                        "hamiltonian_path": str(md_sample / "siesta.TSHS"),
                        "status": "valid",
                    }
                ],
            )
            result = run_script(
                "Comparison/scripts/build_common_tests.py",
                "--md-test-manifest",
                str(md_test),
                "--atomdisp-test-manifest",
                str(atom_test),
                "--train-manifest",
                str(train),
                "--output-dir",
                str(root / "common"),
            )
            self.assertEqual(result.returncode, 2)
            summary = json.loads((root / "common" / "common_test_summary.json").read_text(encoding="utf-8"))
            self.assertFalse(summary["ok"])

    def test_cross_aggregation_contains_required_cells(self) -> None:
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            cross_root = root / "cross"
            for method in ("md", "atom_displacement"):
                for test_set in ("test_md", "test_atomdisp", "test_mixed"):
                    result_dir = cross_root / f"{method}__on__{test_set}"
                    (result_dir / "metrics").mkdir(parents=True)
                    write_csv(
                        result_dir / "metrics" / "sparse_metrics.csv",
                        [{"sample": "sample_1", "relative_frobenius_union": "1.0"}],
                    )
                    (result_dir / "cross_evaluation_manifest.json").write_text(
                        json.dumps(
                            {
                                "train_method": method,
                                "test_set": test_set,
                                "dataset_size": 3,
                                "seed": 42,
                                "model_checkpoint": "model.ckpt",
                            }
                        ),
                        encoding="utf-8",
                    )
            result = run_script(
                "Comparison/scripts/aggregate_cross_metrics.py",
                "--experiment-id",
                "exp_test",
                "--cross-root",
                str(cross_root),
                "--output-dir",
                str(root / "summary"),
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            with (root / "summary" / "cross_evaluation_metrics.csv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            cells = {(row["train_method"], row["test_set"]) for row in rows}
            self.assertEqual(
                cells,
                {
                    ("md", "test_md"),
                    ("md", "test_atomdisp"),
                    ("md", "test_mixed"),
                    ("atom_displacement", "test_md"),
                    ("atom_displacement", "test_atomdisp"),
                    ("atom_displacement", "test_mixed"),
                },
            )

    def test_winner_analysis_does_not_declare_winner_on_different_test_sets(self) -> None:
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            metrics = root / "cross_evaluation_metrics.csv"
            write_csv(
                metrics,
                [
                    {
                        "experiment_id": "exp_test",
                        "train_method": "md",
                        "test_set": "test_md",
                        "dataset_size": "3",
                        "seed": "42",
                        "sample_id": "md_1",
                        "global_rmse_eV": "1.0",
                    },
                    {
                        "experiment_id": "exp_test",
                        "train_method": "atom_displacement",
                        "test_set": "test_atomdisp",
                        "dataset_size": "3",
                        "seed": "42",
                        "sample_id": "atom_1",
                        "global_rmse_eV": "0.1",
                    },
                ],
            )
            result = run_script(
                "Comparison/scripts/analyze_winners.py",
                "--metrics-csv",
                str(metrics),
                "--output-dir",
                str(root / "summary"),
                "--primary-metric",
                "global_rmse_eV",
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            recommendation = json.loads((root / "summary" / "recommendation.json").read_text(encoding="utf-8"))
            self.assertNotEqual(recommendation["status"], "atom_displacement_conservative_win")

    def test_winner_analysis_preserves_seeds_and_marks_stability(self) -> None:
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            metrics = root / "cross_evaluation_metrics.csv"
            rows = []
            for seed in ("1", "2"):
                rows.extend(
                    [
                        {
                            "experiment_id": "exp_test",
                            "train_method": "md",
                            "test_set": "test_mixed",
                            "md_dataset_size": "10",
                            "atom_dataset_size": "10",
                            "seed": seed,
                            "model_checkpoint": f"md_{seed}.ckpt",
                            "global_rmse_eV": "0.5",
                        },
                        {
                            "experiment_id": "exp_test",
                            "train_method": "atom_displacement",
                            "test_set": "test_mixed",
                            "md_dataset_size": "10",
                            "atom_dataset_size": "10",
                            "seed": seed,
                            "model_checkpoint": f"ad_{seed}.ckpt",
                            "global_rmse_eV": "1.0",
                        },
                    ]
                )
            write_csv(metrics, rows)
            result = run_script(
                "Comparison/scripts/analyze_winners.py",
                "--metrics-csv",
                str(metrics),
                "--output-dir",
                str(root / "summary"),
                "--primary-metric",
                "global_rmse_eV",
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            with (root / "summary" / "winner_summary.csv").open(encoding="utf-8") as handle:
                summary_rows = list(csv.DictReader(handle))
            self.assertEqual(summary_rows[0]["winner"], "md")
            self.assertEqual(summary_rows[0]["n_seeds"], "2")
            self.assertEqual(summary_rows[0]["winner_stability"], "stable")
            with (root / "summary" / "winner_by_dataset_size.csv").open(encoding="utf-8") as handle:
                pairs = list(csv.DictReader(handle))
            self.assertEqual({row["seed"] for row in pairs}, {"1", "2"})
            self.assertNotIn("pooled", {row["seed"] for row in pairs})

    def test_winner_analysis_single_seed_warning_and_checkpoint_isolation(self) -> None:
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            metrics = root / "cross_evaluation_metrics.csv"
            write_csv(
                metrics,
                [
                    {
                        "experiment_id": "exp_test",
                        "train_method": "md",
                        "test_set": "test_mixed",
                        "md_dataset_size": "10",
                        "atom_dataset_size": "10",
                        "seed": "1",
                        "model_checkpoint": "md_a.ckpt",
                        "global_rmse_eV": "0.5",
                    },
                    {
                        "experiment_id": "exp_test",
                        "train_method": "atom_displacement",
                        "test_set": "test_mixed",
                        "md_dataset_size": "10",
                        "atom_dataset_size": "10",
                        "seed": "1",
                        "model_checkpoint": "ad_a.ckpt",
                        "global_rmse_eV": "1.0",
                    },
                    {
                        "experiment_id": "other_exp",
                        "train_method": "md",
                        "test_set": "test_mixed",
                        "md_dataset_size": "10",
                        "atom_dataset_size": "10",
                        "seed": "1",
                        "model_checkpoint": "md_b.ckpt",
                        "global_rmse_eV": "9.0",
                    },
                    {
                        "experiment_id": "other_exp",
                        "train_method": "atom_displacement",
                        "test_set": "test_mixed",
                        "md_dataset_size": "10",
                        "atom_dataset_size": "10",
                        "seed": "1",
                        "model_checkpoint": "ad_b.ckpt",
                        "global_rmse_eV": "8.0",
                    },
                ],
            )
            result = run_script(
                "Comparison/scripts/analyze_winners.py",
                "--metrics-csv",
                str(metrics),
                "--output-dir",
                str(root / "summary"),
                "--primary-metric",
                "global_rmse_eV",
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            recommendation = json.loads((root / "summary" / "recommendation.json").read_text(encoding="utf-8"))
            self.assertTrue(recommendation["single_seed_warning"])
            self.assertNotIn("conservative_win", recommendation["status"])
            with (root / "summary" / "winner_by_dataset_size.csv").open(encoding="utf-8") as handle:
                pairs = list(csv.DictReader(handle))
            self.assertEqual({row["experiment_id"] for row in pairs}, {"exp_test", "other_exp"})
            self.assertEqual(len(pairs), 2)

    def test_geometry_leakage_detects_duplicates_near_duplicates_and_md_neighbors(self) -> None:
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            train_a = make_sample(root / "samples", "train_a")
            test_same = make_sample(root / "samples", "test_same")
            test_near = make_sample(root / "samples", "test_near")
            minimal_run_fdf(test_near / "RUN.fdf")
            text = (test_near / "RUN.fdf").read_text(encoding="utf-8").replace("0.0 0.7 0.0 2", "0.0 0.70001 0.0 2")
            (test_near / "RUN.fdf").write_text(text, encoding="utf-8")
            train_manifest = root / "train.csv"
            test_manifest = root / "test.csv"
            write_csv(
                train_manifest,
                [
                    {
                        "sample_id": "train_a",
                        "method": "md",
                        "frame_index": "10",
                        "structure_path": str(train_a / "RUN.fdf"),
                    }
                ],
            )
            write_csv(
                test_manifest,
                [
                    {
                        "sample_id": "test_same",
                        "method": "md",
                        "frame_index": "11",
                        "structure_path": str(test_same / "RUN.fdf"),
                    },
                    {
                        "sample_id": "test_near",
                        "method": "md",
                        "frame_index": "20",
                        "structure_path": str(test_near / "RUN.fdf"),
                    },
                ],
            )
            result = run_script(
                "Comparison/scripts/check_geometry_leakage.py",
                "--train-manifest",
                str(train_manifest),
                "--test-manifest",
                str(test_manifest),
                "--output-dir",
                str(root / "leakage"),
                "--rmsd-threshold",
                "0.001",
            )
            self.assertEqual(result.returncode, 2)
            summary = json.loads((root / "leakage" / "geometry_leakage_summary.json").read_text(encoding="utf-8"))
            self.assertGreaterEqual(summary["exact_duplicates"], 1)
            self.assertGreaterEqual(summary["near_duplicates"], 1)
            self.assertGreaterEqual(summary["md_neighbor_warnings"], 1)

    def test_geometry_leakage_allows_different_geometries(self) -> None:
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            train = make_sample(root / "samples", "train")
            test = make_sample(root / "samples", "test")
            text = (test / "RUN.fdf").read_text(encoding="utf-8").replace("0.0 0.7 0.0 2", "0.0 2.0 0.0 2")
            (test / "RUN.fdf").write_text(text, encoding="utf-8")
            train_manifest = root / "train.csv"
            test_manifest = root / "test.csv"
            write_csv(train_manifest, [{"sample_id": "train", "method": "md", "structure_path": str(train / "RUN.fdf")}])
            write_csv(test_manifest, [{"sample_id": "test", "method": "md", "structure_path": str(test / "RUN.fdf")}])
            result = run_script(
                "Comparison/scripts/check_geometry_leakage.py",
                "--train-manifest",
                str(train_manifest),
                "--test-manifest",
                str(test_manifest),
                "--output-dir",
                str(root / "leakage"),
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)

    def test_predict_inputs_do_not_copy_reference_hamiltonians(self) -> None:
        spec = importlib.util.spec_from_file_location(
            "predict_model_on_dataset",
            REPO_ROOT / "Comparison" / "scripts" / "predict_model_on_dataset.py",
        )
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with workspace_tempdir() as tmp:
            root = Path(tmp)
            sample = make_sample(root / "samples", "001")
            rows = [{"sample_id": "001", "structure_path": str(sample / "RUN.fdf")}]
            copied = module.copy_sample_inputs(rows, root / "workspace")
            copied_dir = root / "workspace" / "predict_structures" / "001"
            self.assertTrue((copied_dir / "RUN.fdf").exists())
            self.assertFalse(any(copied_dir.glob("*.TSHS")))
            self.assertFalse(any(copied_dir.glob("*.HSX")))
            self.assertFalse(copied[0]["reference_hamiltonian_copied_to_input"])

    def test_siesta_settings_hash_and_mismatch_warning(self) -> None:
        sys.path.insert(0, str(REPO_ROOT / "Comparison" / "scripts"))
        spec = importlib.util.spec_from_file_location(
            "siesta_settings",
            REPO_ROOT / "Comparison" / "scripts" / "siesta_settings.py",
        )
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        shared = {"siesta": {"mesh_cutoff": "200 Ry"}}
        md = {"md": {"mesh_cutoff": "200 Ry", "lattice_constant": 15, "save_hs": True}}
        atom = {
            "structure": {
                "siesta": {"MeshCutoff": "200 Ry", "ForceAuxCell": "F"},
                "lattice_constant": 15,
                "force_constants": {"save_tshs": True},
            }
        }
        ok_report = module.compare_settings(md, atom, shared)
        self.assertTrue(ok_report["ok"])
        atom_bad = {"structure": {"siesta": {"mesh_cutoff": "300 Ry", "lattice_constant": 15}}}
        bad_report = module.compare_settings(md, atom_bad, shared)
        self.assertFalse(bad_report["ok"])
        self.assertTrue(bad_report["warning"])

    def test_compute_budget_pairing_helpers(self) -> None:
        sys.path.insert(0, str(REPO_ROOT / "Comparison" / "scripts"))
        spec = importlib.util.spec_from_file_location(
            "pipeline_ui",
            REPO_ROOT / "Comparison" / "scripts" / "pipeline_ui.py",
        )
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        sys.modules["pipeline_ui"] = module
        spec.loader.exec_module(module)
        md = {"dataset_size": 10, "siesta_hamiltonians": 10}
        atom_same = {"dataset_size": 10, "siesta_hamiltonians": 10}
        atom_other = {"dataset_size": 20, "siesta_hamiltonians": 11}
        self.assertTrue(module.should_compare_budget_pair(md, atom_same, [atom_same, atom_other], "equal_sample_count"))
        self.assertFalse(module.should_compare_budget_pair(md, atom_other, [atom_same, atom_other], "equal_sample_count"))
        self.assertTrue(module.should_compare_budget_pair(md, atom_other, [atom_other], "equal_siesta_budget"))
        self.assertTrue(module.budget_warning(10, 30))

    def test_verify_integrity_accepts_multi_displacement_fc_layout(self) -> None:
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            base = root / "base"
            base.mkdir()
            minimal_run_fdf(base / "RUN.fdf")
            (base / "H.psf").write_text("pseudo", encoding="utf-8")
            md_steps = root / "MD_steps"
            fc_steps = root / "FC_steps"
            for index in range(1, 4):
                make_sample(md_steps, str(index))
            for index in range(1, 9):
                make_sample(fc_steps, str(index))
            fc_raw = root / "dataset"
            for label in ("d0p01", "d0p02"):
                run_dir = fc_raw / "FC_runs" / label
                run_dir.mkdir(parents=True)
                (run_dir / "FORCE_CONSTANTS.FC").write_text("fc", encoding="utf-8")
            manifest = {
                "generation_mode": "siesta_fc_multi_run",
                "runs": [{"label": "d0p01", "selected_count": 4}, {"label": "d0p02", "selected_count": 4}],
            }
            manifest_path = root / "samples_manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            result = run_script(
                "Comparison/scripts/verify_dataset_integrity.py",
                "--base-fdf",
                str(base / "RUN.fdf"),
                "--md-steps-dir",
                str(md_steps),
                "--fc-steps-dir",
                str(fc_steps),
                "--fc-raw-dir",
                str(fc_raw),
                "--samples-manifest",
                str(manifest_path),
                "--siesta-bin",
                "true",
                "--dry-run",
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            report = json.loads(result.stdout)
            self.assertEqual(report["fc_mode"], "siesta_fc_multi_run")
            self.assertEqual(report["actual_normalized_samples"], 8)

    def test_verify_integrity_reports_missing_fc_run_and_stale_matrix(self) -> None:
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            base = root / "base"
            base.mkdir()
            minimal_run_fdf(base / "RUN.fdf")
            (base / "H.psf").write_text("pseudo", encoding="utf-8")
            md_steps = root / "MD_steps"
            fc_steps = root / "FC_steps"
            make_sample(md_steps, "1")
            stale = make_sample(fc_steps, "1")
            (stale / "RUN.out").unlink()
            fc_raw = root / "dataset"
            (fc_raw / "FC_runs" / "d0p01").mkdir(parents=True)
            (fc_raw / "FC_runs" / "d0p01" / "FC").write_text("fc", encoding="utf-8")
            manifest_path = root / "samples_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "generation_mode": "siesta_fc_multi_run",
                        "runs": [{"label": "d0p01", "selected_count": 1}, {"label": "d0p02", "selected_count": 1}],
                    }
                ),
                encoding="utf-8",
            )
            result = run_script(
                "Comparison/scripts/verify_dataset_integrity.py",
                "--base-fdf",
                str(base / "RUN.fdf"),
                "--md-steps-dir",
                str(md_steps),
                "--fc-steps-dir",
                str(fc_steps),
                "--fc-raw-dir",
                str(fc_raw),
                "--samples-manifest",
                str(manifest_path),
                "--siesta-bin",
                "true",
                "--dry-run",
            )
            self.assertEqual(result.returncode, 2)
            report = json.loads(result.stdout)
            self.assertIn("d0p02", report["missing_raw_fc_runs"])
            self.assertIn("1", report["fc"]["missing_output"])

    def test_ui_latest_cross_experiment_does_not_flatten_all_metrics(self) -> None:
        app_js = (REPO_ROOT / "Comparison" / "ui" / "app.js").read_text(encoding="utf-8")
        self.assertNotIn("experiments.flatMap((experiment) => experiment.metrics", app_js)
        self.assertIn("Mostrando solo el experimento cross mas reciente", app_js)

    def test_timing_breakdown_schema_keys(self) -> None:
        required = {
            "md_siesta_generation_seconds",
            "atomdisp_siesta_generation_seconds",
            "dataset_preparation_seconds",
            "normalization_seconds",
            "training_seconds",
            "prediction_seconds",
            "evaluation_seconds",
            "winner_analysis_seconds",
            "total_experiment_seconds",
        }
        schema_text = (REPO_ROOT / "Comparison" / "scripts" / "pipeline_ui.py").read_text(encoding="utf-8")
        for key in required:
            self.assertIn(key, schema_text)

    def test_md_rewrite_makes_effective_geometries_differ_when_xv_differs(self) -> None:
        sys.path.insert(0, str(REPO_ROOT / "MD" / "scripts"))
        spec = importlib.util.spec_from_file_location(
            "generate_md_dataset",
            REPO_ROOT / "MD" / "scripts" / "generate_md_dataset.py",
        )
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with workspace_tempdir() as tmp:
            root = Path(tmp)
            signatures = []
            for index, z in enumerate((0.0, 0.5), start=1):
                sample_dir = root / str(index)
                sample_dir.mkdir()
                minimal_run_fdf(sample_dir / "RUN.fdf")
                sample_dir.joinpath("siesta.XV").write_text(
                    "\n".join(
                        [
                            "18.89726125 0.0 0.0",
                            "0.0 18.89726125 0.0",
                            "0.0 0.0 18.89726125",
                            "3",
                            f"1 8 0.0 0.0 {z} 0.0 0.0 0.0",
                            "2 1 0.0 1.0 0.0 0.0 0.0 0.0",
                            "2 1 0.0 -1.0 0.0 0.0 0.0 0.0",
                            "",
                        ]
                    ),
                    encoding="utf-8",
                )
                module.rewrite_run_fdf_from_xv(sample_dir / "RUN.fdf", sample_dir / "siesta.XV")
                signatures.append(module.effective_fdf_geometry_signature(sample_dir / "RUN.fdf"))
            self.assertNotEqual(signatures[0], signatures[1])


if __name__ == "__main__":
    unittest.main()
