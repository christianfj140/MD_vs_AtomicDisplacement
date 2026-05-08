from __future__ import annotations

import csv
import contextlib
import importlib.util
import json
import os
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
    def load_pipeline_ui_module(self):
        sys.path.insert(0, str(REPO_ROOT / "Comparison" / "scripts"))
        spec = importlib.util.spec_from_file_location(
            "pipeline_ui",
            REPO_ROOT / "Comparison" / "scripts" / "pipeline_ui.py",
        )
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    def test_method_selection_normalization_and_validation(self) -> None:
        module = self.load_pipeline_ui_module()
        self.assertEqual(
            module.normalize_selected_methods(["md", "atom_displacement"]),
            ["md", "siesta_fc_cartesian"],
        )
        self.assertEqual(module.normalize_selected_methods(None), ["md", "siesta_fc_cartesian"])
        with self.assertRaisesRegex(RuntimeError, "Selecciona al menos un metodo"):
            module.normalize_selected_methods([])
        with self.assertRaisesRegex(RuntimeError, "no soportados"):
            module.normalize_selected_methods(["md", "bogus"])
        self.assertEqual(module.normalize_selected_methods(["random_cartesian"]), ["random_cartesian"])

    def test_run_mode_validation(self) -> None:
        module = self.load_pipeline_ui_module()
        self.assertEqual(module.parse_run_mode(None), "full_strict_pipeline")
        self.assertEqual(module.parse_run_mode("dataset_only"), "dataset_only")
        with self.assertRaisesRegex(RuntimeError, "run_mode"):
            module.parse_run_mode("training_only")

    def test_compute_accelerator_validation_and_config_application(self) -> None:
        module = self.load_pipeline_ui_module()
        self.assertEqual(module.parse_compute_accelerator(None), "cpu")
        self.assertEqual(module.parse_compute_accelerator("GPU"), "gpu")
        self.assertEqual(module.parse_compute_accelerator("auto"), "auto")
        with self.assertRaisesRegex(RuntimeError, "compute_accelerator"):
            module.parse_compute_accelerator("tpu")
        config = {"training": {"trainer": {"accelerator": "cpu"}}}
        module.apply_training_accelerator(config, "gpu")
        self.assertEqual(config["training"]["trainer"]["accelerator"], "gpu")

    def test_performance_settings_validation_env_and_config_application(self) -> None:
        module = self.load_pipeline_ui_module()
        settings = module.parse_performance_settings(
            {
                "max_parallel_siesta_jobs": "2",
                "omp_num_threads": "3",
                "mkl_num_threads": None,
                "openblas_num_threads": "",
                "torch_num_threads": "4",
                "compute_accelerator": "auto",
                "batch_size": "16",
                "store_in_memory": "false",
                "torch_float32_matmul_precision": "high",
            }
        )
        self.assertEqual(settings["max_parallel_siesta_jobs"], 2)
        self.assertEqual(settings["compute_accelerator"], "auto")
        self.assertEqual(settings["batch_size"], 16)
        self.assertIs(settings["store_in_memory"], False)
        env = module.performance_env(settings)
        self.assertEqual(env["OMP_NUM_THREADS"], "3")
        self.assertEqual(env["TORCH_NUM_THREADS"], "4")
        self.assertEqual(env["TORCH_FLOAT32_MATMUL_PRECISION"], "high")
        config = {
            "training": {"trainer": {"accelerator": "cpu"}, "data": {}},
            "testing": {"data": {}},
            "prediction": {"data": {}},
            "single_points": {},
        }
        module.apply_performance_to_config(config, settings)
        self.assertEqual(config["training"]["trainer"]["accelerator"], "auto")
        self.assertEqual(config["training"]["data"]["batch_size"], 16)
        self.assertIs(config["training"]["data"]["store_in_memory"], False)
        self.assertIs(config["testing"]["data"]["store_in_memory"], False)
        self.assertEqual(config["single_points"]["workers"], 2)
        with self.assertRaisesRegex(RuntimeError, "performance.max_parallel_siesta_jobs"):
            module.parse_performance_settings({"max_parallel_siesta_jobs": 0})
        with self.assertRaisesRegex(RuntimeError, "store_in_memory"):
            module.parse_performance_settings({"store_in_memory": "maybe"})

    def test_default_venv_command_is_repo_portable(self) -> None:
        module = self.load_pipeline_ui_module()
        self.assertEqual(
            module.DEFAULT_VENV_ACTIVATE_COMMAND,
            "source ${REPO_ROOT}/.venv/bin/activate",
        )
        self.assertEqual(
            module.resolve_venv_activate_from_command("source .venv/bin/activate"),
            "${REPO_ROOT}/.venv/bin/activate",
        )
        self.assertEqual(
            module.resolve_venv_activate_from_command("source ${REPO_ROOT}/.venv/bin/activate"),
            "${REPO_ROOT}/.venv/bin/activate",
        )

    def test_random_cartesian_options_support_multiple_sizes(self) -> None:
        module = self.load_pipeline_ui_module()
        self.assertEqual(module.random_cartesian_sizes_from_options({"n_structures": 3}), [3])
        self.assertEqual(module.random_cartesian_sizes_from_options({"n_structures": [11, 23, 30]}), [11, 23, 30])
        self.assertEqual(module.random_cartesian_sizes_from_options({"n_structures": "11, 23, 30"}), [11, 23, 30])

    def test_dataset_only_records_status_and_skips_cross_evaluation(self) -> None:
        module = self.load_pipeline_ui_module()
        module.STRICT_COMPARISON_MODE = False
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            module.RESULTS_ROOT = root / "results"
            module.WORKSPACES_ROOT = root / "workspaces"
            runner = module.ExperimentRunner()
            calls = {"run_one": [], "cross": 0}

            def fake_run_one(key, size, run_id, **kwargs):
                calls["run_one"].append((key, kwargs.get("run_mode"), kwargs.get("compute_accelerator")))
                return {
                    "pipeline": key,
                    "method_id": "siesta_fc_cartesian" if key == "atom_displacement" else key,
                    "dataset_label": kwargs.get("dataset_label") or f"dataset_{size}",
                    "dataset_size": size,
                    "returncode": 0,
                    "result_dir": str(root / "fake" / key / f"dataset_{size}"),
                    "dataset_sample_ids": [f"{key}-{size}"],
                    "dataset_sample_hash": f"hash-{key}-{size}",
                    "run_mode": kwargs.get("run_mode"),
                    "pipeline_elapsed_seconds": 1.25,
                    "siesta_counts": {"launched": 1, "skipped_or_reused": 0, "failed": 0},
                }

            def fake_cross(*_args, **_kwargs):
                calls["cross"] += 1
                return {"ok": True}

            runner._run_one = fake_run_one  # type: ignore[method-assign]
            runner._run_cross_evaluation = fake_cross  # type: ignore[method-assign]
            runner._started_at = 1.0
            runner._run(
                [3],
                [3],
                "dataset_only_case",
                split_ratios={"train": 1 / 3, "validation": 1 / 3, "test": 1 / 3},
                selected_methods=["md", "siesta_fc_cartesian"],
                run_mode="dataset_only",
                compute_accelerator="gpu",
                performance={"compute_accelerator": "gpu", "max_parallel_siesta_jobs": 2},
            )
            manifest = module.load_config(root / "results" / "dataset_only_case" / "experiment_manifest.yaml")
            self.assertEqual(manifest["run_mode"], "dataset_only")
            self.assertEqual(manifest["scientific_status"], "dataset_only")
            self.assertEqual(manifest["selected_methods"], ["md", "siesta_fc_cartesian"])
            self.assertEqual(manifest["compute_accelerator"], "gpu")
            self.assertEqual(manifest["performance"]["max_parallel_siesta_jobs"], 2)
            self.assertEqual(manifest["timing"]["counters"]["siesta_launched"], 2)
            self.assertTrue(manifest["timing"]["stages"])
            self.assertEqual(calls["cross"], 0)
            self.assertEqual(
                calls["run_one"],
                [("md", "dataset_only", "gpu"), ("atom_displacement", "dataset_only", "gpu")],
            )
            self.assertTrue(manifest["cross_evaluation"]["skipped"])

    def test_single_method_full_pipeline_is_non_comparative(self) -> None:
        module = self.load_pipeline_ui_module()
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            module.RESULTS_ROOT = root / "results"
            module.WORKSPACES_ROOT = root / "workspaces"
            runner = module.ExperimentRunner()
            calls = {"cross": 0}

            def fake_run_one(key, size, run_id, **kwargs):
                return {
                    "pipeline": key,
                    "method_id": key,
                    "dataset_label": f"dataset_{size}",
                    "dataset_size": size,
                    "returncode": 0,
                    "result_dir": str(root / "fake" / key / f"dataset_{size}"),
                    "dataset_sample_ids": [f"{key}-{size}"],
                    "dataset_sample_hash": f"hash-{key}-{size}",
                }

            def fake_cross(*_args, **_kwargs):
                calls["cross"] += 1
                return {"ok": True}

            runner._run_one = fake_run_one  # type: ignore[method-assign]
            runner._run_cross_evaluation = fake_cross  # type: ignore[method-assign]
            runner._started_at = 1.0
            runner._run(
                [3],
                [],
                "md_only_case",
                split_ratios={"train": 1 / 3, "validation": 1 / 3, "test": 1 / 3},
                selected_methods=["md"],
                run_mode="full_strict_pipeline",
            )
            manifest = module.load_config(root / "results" / "md_only_case" / "experiment_manifest.yaml")
            self.assertEqual(manifest["selected_methods"], ["md"])
            self.assertEqual(manifest["scientific_status"], "non_comparative")
            self.assertEqual(calls["cross"], 0)

    def test_random_cartesian_full_pipeline_is_accepted_for_phase_three(self) -> None:
        module = self.load_pipeline_ui_module()
        runner = module.ExperimentRunner()
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            module.RESULTS_ROOT = root / "results"
            module.WORKSPACES_ROOT = root / "workspaces"
            started = {}

            class FakeThread:
                def __init__(self, target, args, daemon):
                    self.target = target
                    self.args = args
                    self.daemon = daemon

                def start(self):
                    started["args"] = self.args

                def is_alive(self):
                    return False

            original_thread = module.threading.Thread
            module.threading.Thread = FakeThread
            try:
                status = runner.start(
                    [],
                    [],
                    selected_methods=["random_cartesian"],
                    run_mode="full_strict_pipeline",
                    compute_accelerator="gpu",
                    random_cartesian_options={"n_structures": 3},
                    split_ratios={"train": 1 / 3, "validation": 1 / 3, "test": 1 / 3},
                )
            finally:
                module.threading.Thread = original_thread
            self.assertEqual(status["run_id"], runner._run_id)
            self.assertEqual(started["args"][-5], "gpu")
            self.assertEqual(started["args"][-4], ["random_cartesian"])
            self.assertEqual(started["args"][-3], "full_strict_pipeline")
            self.assertEqual(started["args"][-2], {"n_structures": 3})
            self.assertEqual(started["args"][-1]["compute_accelerator"], "gpu")

    def test_ui_experiment_payload_includes_methods_and_run_mode(self) -> None:
        index_html = (REPO_ROOT / "Comparison" / "ui" / "index.html").read_text(encoding="utf-8")
        app_js = (REPO_ROOT / "Comparison" / "ui" / "app.js").read_text(encoding="utf-8")
        self.assertIn('value="md"', index_html)
        self.assertIn('value="siesta_fc_cartesian"', index_html)
        self.assertIn('value="random_cartesian"', index_html)
        self.assertIn('value="random_cartesian" checked', index_html)
        self.assertIn('id="random-cartesian-n-structures"', index_html)
        self.assertNotIn("phonon", index_html.lower())
        self.assertIn("selected_methods: methods", app_js)
        self.assertIn('run_mode: document.getElementById("run-mode").value', app_js)
        self.assertIn('value="source ${REPO_ROOT}/.venv/bin/activate"', index_html)
        self.assertIn('DEFAULT_VENV_ACTIVATE_COMMAND = "source ${REPO_ROOT}/.venv/bin/activate"', app_js)
        self.assertNotIn("graph2mat-env", index_html)
        self.assertNotIn("graph2mat-env", app_js)
        self.assertIn('id="compute-accelerator"', index_html)
        self.assertIn('value="cpu"', index_html)
        self.assertIn('value="gpu"', index_html)
        self.assertIn('value="auto"', index_html)
        self.assertIn('id="performance-compute-accelerator"', index_html)
        self.assertIn('id="performance-max-parallel-siesta-jobs"', index_html)
        self.assertIn("performanceSettings()", app_js)
        self.assertIn("performance,", app_js)
        self.assertIn("compute_accelerator: performance.compute_accelerator", app_js)
        self.assertIn("parseRandomCartesianOptions(methods)", app_js)
        self.assertIn("random_cartesian_options: randomCartesianOptions", app_js)
        self.assertIn("n_structures", app_js)
        self.assertIn("syncRandomCartesianSizeFromAtomPlan(sizes)", app_js)
        self.assertIn("validSizes.join", app_js)
        self.assertIn("resultPipelines", app_js)
        self.assertIn("results_random_cartesian", app_js)
        self.assertIn("wasRunning && !status.running && state.plotsEnabled", app_js)
        self.assertIn('<option value="test_random_cartesian" selected>', index_html)
        self.assertIn("crossTrainMethods(experiment)", app_js)
        self.assertIn("crossTestSets(experiment)", app_js)
        self.assertNotIn('const trainMethods = ["md", "atom_displacement"]', app_js)

    def test_cross_prediction_command_uses_training_accelerator(self) -> None:
        script = (REPO_ROOT / "Comparison" / "scripts" / "pipeline_ui.py").read_text(encoding="utf-8")
        self.assertIn('"--accelerator"', script)
        self.assertIn('.get("accelerator", "cpu")', script)

    def test_random_cartesian_archived_results_are_in_plot_summary(self) -> None:
        module = self.load_pipeline_ui_module()
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            module.RESULTS_ROOT = root / "results"
            result_dir = module.RESULTS_ROOT / "results_random_cartesian" / "dataset_5" / "run_random"
            metrics_dir = result_dir / "metrics"
            metrics_dir.mkdir(parents=True)
            write_csv(
                metrics_dir / "sparse_metrics.csv",
                [{"sample": "sample_1", "relative_frobenius_union": "0.25"}],
            )
            write_csv(
                metrics_dir / "spectral_metrics.csv",
                [{"sample": "sample_1", "global_rmse_eV": "0.4"}],
            )
            (result_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "run_id": "random",
                        "result_dir": str(result_dir),
                        "dataset_size": 5,
                        "requested_dataset_size": 5,
                    }
                ),
                encoding="utf-8",
            )
            plots = module.plot_data_summary()
            runs = plots["runs"]
            self.assertEqual(len(runs), 1)
            self.assertEqual(runs[0]["pipeline"], "random_cartesian")
            self.assertEqual(runs[0]["label"], "Random Cartesian")
            self.assertEqual(runs[0]["means"]["sparse"]["relative_frobenius_union"], 0.25)
            archived = module.archived_results_summary()
            self.assertIn("random_cartesian", archived)
            self.assertEqual(len(archived["random_cartesian"]), 1)
            results = module.result_summary()
            self.assertIn("random_cartesian", results)
            self.assertEqual(
                results["random_cartesian"]["prediction_glob"],
                "AtomDisplacement/dataset/RandomCartesian_steps/*/ML_prediction.HSX",
            )

    def test_repo_local_paths_do_not_raise_reproducibility_warning(self) -> None:
        module = self.load_pipeline_ui_module()
        warning = module.absolute_path_warning(
            [
                REPO_ROOT / "MD" / "dataset",
                REPO_ROOT / ".venv" / "bin" / "activate",
            ]
        )
        self.assertEqual(warning, "")
        warning = module.absolute_path_warning([Path.home() / "external_data" / "RUN.fdf"])
        self.assertIn("User-local absolute paths detected", warning)
        old_warning = (
            "User-local absolute paths detected: "
            f"{REPO_ROOT / 'MD' / 'dataset'}, "
            f"{REPO_ROOT / '.venv' / 'bin' / 'activate'}"
        )
        self.assertEqual(module.sanitize_reproducibility_warning(old_warning), "")

    def load_random_cartesian_module(self):
        sys.path.insert(0, str(REPO_ROOT / "AtomDisplacement" / "scripts"))
        spec = importlib.util.spec_from_file_location(
            "generate_random_cartesian_dataset",
            REPO_ROOT / "AtomDisplacement" / "scripts" / "generate_random_cartesian_dataset.py",
        )
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    def random_cartesian_test_config(self, n_structures: int = 2, seed: int = 7) -> dict:
        return {
            "paths": {"run_fdf_name": "RUN.fdf", "run_out_name": "RUN.out"},
            "structure": {
                "single_point": {
                    "system_name_template": "H2O {sample_id}",
                    "title": "Random Cartesian single point",
                },
                "force_constants": {"enabled": False},
                "lattice_constant": {"value": 1.0, "unit": "Ang"},
                "lattice_vectors": [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
                "coordinates_format": "Ang",
                "species": [
                    {"index": 1, "atomic_number": 8, "symbol": "O"},
                    {"index": 2, "atomic_number": 1, "symbol": "H"},
                ],
                "atoms": [
                    {"label": "O", "species_index": 1, "position": [0.0, 0.0, 0.0]},
                    {"label": "H", "species_index": 2, "position": [0.0, 1.0, 0.0]},
                    {"label": "H", "species_index": 2, "position": [0.0, -1.0, 0.0]},
                ],
                "kgrid_monkhorst_pack": [[1, 0, 0, 0.0], [0, 1, 0, 0.0], [0, 0, 1, 0.0]],
                "siesta": {"Save.HS": "T", "XML.Write": "T"},
                "single_point_overrides": {},
                "random_cartesian": {
                    "enabled": True,
                    "n_structures": n_structures,
                    "seed": seed,
                    "distribution": "gaussian",
                    "sigma_ang": 0.03,
                    "uniform_range_ang": 0.05,
                    "move_atoms": "all",
                    "species_filter": [],
                    "min_distance_ang": 0.65,
                    "max_attempts_per_structure": 10,
                    "remove_center_of_mass_translation": True,
                },
            },
        }

    def configure_random_cartesian_module(self, module, root: Path, *, n_structures: int = 2, seed: int = 7) -> None:
        base = root / "base"
        relaxed = root / "relaxed"
        dataset = root / "dataset"
        base.mkdir()
        relaxed.mkdir()
        dataset.mkdir()
        (base / "H.psf").write_text("pseudo H\n", encoding="utf-8")
        (base / "O.psf").write_text("pseudo O\n", encoding="utf-8")
        (relaxed / "H.ion.xml").write_text("<ion />\n", encoding="utf-8")
        (relaxed / "O.ion.xml").write_text("<ion />\n", encoding="utf-8")
        structure = module.Structure(
            lattice_vectors_ang=[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
            species_labels={1: (8, "O"), 2: (1, "H")},
            atom_species=[1, 2, 2],
            positions_ang=[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]],
        )
        module.BASE_DIR = base
        module.RELAXED_DIR = relaxed
        module.DATASET_DIR = dataset
        module.PIPELINE_CONFIG = self.random_cartesian_test_config(n_structures=n_structures, seed=seed)
        module.PIPELINE_PATHS = {"samples_manifest_path": dataset / "samples_manifest.json"}
        module.load_reference_structure = lambda: (structure, str(relaxed / "reference.fdf"))

    def test_random_cartesian_same_seed_reproduces_and_uses_separate_root(self) -> None:
        module = self.load_random_cartesian_module()
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            self.configure_random_cartesian_module(module, root, n_structures=2, seed=11)
            module.generate_dataset(module.PIPELINE_CONFIG)
            dataset_root = root / "dataset" / "RandomCartesian_steps"
            first_hashes = {
                path.name: module.file_sha256(path / "RUN.fdf")
                for path in sorted(dataset_root.glob("sample_*"))
            }
            module.generate_dataset(module.PIPELINE_CONFIG)
            second_hashes = {
                path.name: module.file_sha256(path / "RUN.fdf")
                for path in sorted(dataset_root.glob("sample_*"))
            }
            self.assertEqual(first_hashes, second_hashes)
            self.assertTrue((dataset_root / "dataset_manifest.json").exists())
            self.assertTrue((dataset_root / "artifact_hashes.json").exists())
            self.assertFalse((root / "dataset" / "FC_steps").exists())

    def test_random_cartesian_different_seed_changes_geometries(self) -> None:
        module = self.load_random_cartesian_module()
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            self.configure_random_cartesian_module(module, root, n_structures=1, seed=11)
            module.generate_dataset(module.PIPELINE_CONFIG)
            first = (root / "dataset" / "RandomCartesian_steps" / "sample_000001" / "RUN.fdf").read_text(encoding="utf-8")
            module.PIPELINE_CONFIG["structure"]["random_cartesian"]["seed"] = 12
            module.generate_dataset(module.PIPELINE_CONFIG)
            second = (root / "dataset" / "RandomCartesian_steps" / "sample_000001" / "RUN.fdf").read_text(encoding="utf-8")
            self.assertNotEqual(first, second)

    def test_random_cartesian_gaussian_and_uniform_deterministic_vectors(self) -> None:
        module = self.load_random_cartesian_module()
        structure = module.Structure(
            lattice_vectors_ang=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            species_labels={1: (8, "O")},
            atom_species=[1],
            positions_ang=[[0.0, 0.0, 0.0]],
        )
        gaussian = module.random_cartesian_config(
            {"structure": {"random_cartesian": {"seed": 7, "n_structures": 1, "sigma_ang": 0.1, "move_atoms": [1], "remove_center_of_mass_translation": False}}}
        )
        self.assertEqual(
            module.displacement_field(structure, gaussian, __import__("random").Random(7))[0],
            [-0.025588028844760042, 0.0511431512516514, -0.022609616478310474],
        )
        uniform = module.random_cartesian_config(
            {"structure": {"random_cartesian": {"seed": 7, "n_structures": 1, "distribution": "uniform", "uniform_range_ang": 0.2, "move_atoms": [1], "remove_center_of_mass_translation": False}}}
        )
        self.assertEqual(
            module.displacement_field(structure, uniform, __import__("random").Random(7))[0],
            [-0.07046689406673506, -0.13966033043019924, 0.0603737892159415],
        )

    def test_random_cartesian_min_distance_and_failure(self) -> None:
        module = self.load_random_cartesian_module()
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            self.configure_random_cartesian_module(module, root, n_structures=1, seed=1)
            module.PIPELINE_CONFIG["structure"]["random_cartesian"].update(
                {
                    "sigma_ang": 0.0,
                    "min_distance_ang": 10.0,
                    "max_attempts_per_structure": 2,
                }
            )
            with self.assertRaisesRegex(RuntimeError, "could not generate a valid structure"):
                module.generate_dataset(module.PIPELINE_CONFIG)

    def test_random_cartesian_metadata_and_required_basis(self) -> None:
        module = self.load_random_cartesian_module()
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            self.configure_random_cartesian_module(module, root, n_structures=1, seed=5)
            module.generate_dataset(module.PIPELINE_CONFIG)
            metadata = json.loads(
                (root / "dataset" / "RandomCartesian_steps" / "sample_000001" / "metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["generation_method"], "random_cartesian")
            self.assertEqual(metadata["seed"], 5)
            self.assertEqual(metadata["sample_index"], 0)
            self.assertIn("base_geometry_hash", metadata)
            self.assertIn("displacements_ang", metadata)
            self.assertIn("accepted_attempt", metadata)
            self.assertIn("split_group_id", metadata)

            for pseudo_file in (root / "base").glob("*.psf"):
                pseudo_file.unlink()
            with self.assertRaisesRegex(RuntimeError, "requires pseudopotential"):
                module.generate_dataset(module.PIPELINE_CONFIG)
            (root / "base" / "H.psf").write_text("pseudo H\n", encoding="utf-8")
            (root / "base" / "O.psf").write_text("pseudo O\n", encoding="utf-8")
            for basis_file in (root / "relaxed").glob("*.ion.xml"):
                basis_file.unlink()
            with self.assertRaisesRegex(RuntimeError, "requires basis"):
                module.generate_dataset(module.PIPELINE_CONFIG)

    def test_run_single_points_updates_random_cartesian_matrix_hashes(self) -> None:
        sys.path.insert(0, str(REPO_ROOT / "AtomDisplacement" / "scripts"))
        spec = importlib.util.spec_from_file_location(
            "run_single_points",
            REPO_ROOT / "AtomDisplacement" / "scripts" / "run_single_points.py",
        )
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        with workspace_tempdir() as tmp:
            root = Path(tmp)
            sample = make_sample(root / "RandomCartesian_steps", "sample_000001", hamiltonian=False)
            (sample / "RUN.out").unlink()
            (root / "RandomCartesian_steps" / "dataset_manifest.json").write_text(
                json.dumps(
                    {
                        "method_id": "random_cartesian",
                        "samples": [{"sample_id": "sample_000001", "sample_dir": str(sample)}],
                    }
                ),
                encoding="utf-8",
            )
            (root / "RandomCartesian_steps" / "artifact_hashes.json").write_text("{}", encoding="utf-8")
            module.generated_sample_dirs = lambda: [sample]
            module.require_command = lambda command: None
            module.DATASET_DIR = root
            module.PIPELINE_PATHS["run_summary_path"] = root / "run_summary.json"
            module.PIPELINE_CONFIG.setdefault("single_points", {})["allow_unvalidated_matrices"] = False

            def fake_run_siesta(sample_dir: Path, output_path: Path) -> int:
                output_path.write_text("Job completed\nSCF cycle converged\n", encoding="utf-8")
                (sample_dir / "siesta.HSX").write_bytes(b"matrix")
                return 0

            module.run_siesta_in_dir = fake_run_siesta
            argv = sys.argv
            try:
                sys.argv = ["run_single_points.py"]
                rc = module.main()
            finally:
                sys.argv = argv
            self.assertEqual(rc, 0)
            manifest = json.loads((root / "RandomCartesian_steps" / "dataset_manifest.json").read_text(encoding="utf-8"))
            artifact = json.loads((root / "RandomCartesian_steps" / "artifact_hashes.json").read_text(encoding="utf-8"))
            self.assertIn("sample_000001/siesta.HSX", manifest["matrix_file_hashes"])
            self.assertIn("sample_000001/siesta.HSX", artifact["matrices"])

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
        sys.modules[spec.name] = module
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

    def test_validation_accepts_matching_tshs_and_hsx_outputs(self) -> None:
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            sample = make_sample(root / "samples", "001")
            (sample / "siesta.HSX").write_bytes(b"fake hsx")
            output = root / "validation"
            result = run_script(
                "Comparison/scripts/validate_sample_bundle.py",
                "--samples-dir",
                str(root / "samples"),
                "--method",
                "atom_displacement",
                "--output-dir",
                str(output),
                "--min-valid",
                "1",
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            with (output / "valid_samples.csv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertTrue(rows[0]["hamiltonian_path"].endswith(".TSHS"))

    def test_atomdisp_training_requires_split_manifest_in_strict_mode(self) -> None:
        sys.path.insert(0, str(REPO_ROOT / "AtomDisplacement" / "scripts"))
        spec = importlib.util.spec_from_file_location(
            "run_atdisp_training",
            REPO_ROOT / "AtomDisplacement" / "scripts" / "run_atdisp_training.py",
        )
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        with workspace_tempdir() as tmp:
            root = Path(tmp)
            sample = make_sample(root / "FC_steps", "001")
            module.DATASET_DIR = root
            module.SPLITS_DIR = root / "splits"
            module.generated_sample_dirs = lambda: [sample]
            module.completed_sample_dirs = lambda: [sample]
            module.PIPELINE_CONFIG.setdefault("training", {}).pop("allow_all_completed_debug", None)
            with self.assertRaisesRegex(RuntimeError, "strict training requiere"):
                module.strict_train_sample_dirs()

            module.SPLITS_DIR.mkdir()
            write_csv(
                module.SPLITS_DIR / "train_manifest.csv",
                [
                    {
                        "sample_id": "001",
                        "sample_dir": str(sample),
                        "structure_path": str(sample / "RUN.fdf"),
                        "status": "valid",
                        "valid": "true",
                    }
                ],
            )
            self.assertEqual(module.strict_train_sample_dirs(), [sample])

    def test_run_single_points_reruns_stale_matrix_instead_of_skipping(self) -> None:
        sys.path.insert(0, str(REPO_ROOT / "AtomDisplacement" / "scripts"))
        spec = importlib.util.spec_from_file_location(
            "run_single_points",
            REPO_ROOT / "AtomDisplacement" / "scripts" / "run_single_points.py",
        )
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
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
            module.PIPELINE_CONFIG.setdefault("single_points", {})["allow_unvalidated_matrices"] = False

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

    def test_run_single_points_workers_keep_summary_order(self) -> None:
        sys.path.insert(0, str(REPO_ROOT / "AtomDisplacement" / "scripts"))
        spec = importlib.util.spec_from_file_location(
            "run_single_points",
            REPO_ROOT / "AtomDisplacement" / "scripts" / "run_single_points.py",
        )
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        with workspace_tempdir() as tmp:
            root = Path(tmp)
            samples = [make_sample(root, f"{index:03d}", hamiltonian=False) for index in range(4)]
            for sample in samples:
                (sample / "RUN.out").unlink()
            calls: list[str] = []

            module.generated_sample_dirs = lambda: list(samples)
            module.require_command = lambda command: None
            module.DATASET_DIR = root
            module.PIPELINE_PATHS["run_summary_path"] = root / "run_summary.json"
            module.PIPELINE_CONFIG.setdefault("single_points", {})["allow_unvalidated_matrices"] = False

            def fake_run_siesta(sample_dir: Path, output_path: Path) -> int:
                calls.append(sample_dir.name)
                output_path.write_text("Job completed\nSCF cycle converged\n", encoding="utf-8")
                (sample_dir / "siesta.TSHS").write_bytes(b"matrix")
                return 0

            module.run_siesta_in_dir = fake_run_siesta
            argv = sys.argv
            try:
                sys.argv = ["run_single_points.py", "--workers", "2"]
                rc = module.main()
            finally:
                sys.argv = argv
            self.assertEqual(rc, 0)
            self.assertCountEqual(calls, [sample.name for sample in samples])
            summary = json.loads((root / "run_summary.json").read_text(encoding="utf-8"))
            self.assertEqual([row["id"] for row in summary], [sample.name for sample in samples])

    def test_run_single_points_reuses_local_validated_outputs(self) -> None:
        sys.path.insert(0, str(REPO_ROOT / "AtomDisplacement" / "scripts"))
        spec = importlib.util.spec_from_file_location(
            "run_single_points",
            REPO_ROOT / "AtomDisplacement" / "scripts" / "run_single_points.py",
        )
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        with workspace_tempdir() as tmp:
            root = Path(tmp)
            sample = make_sample(root, "001")

            module.generated_sample_dirs = lambda: [sample]
            module.require_command = lambda command: None
            module.DATASET_DIR = root
            module.PIPELINE_PATHS["run_summary_path"] = root / "run_summary.json"
            module.PIPELINE_CONFIG.setdefault("single_points", {})["allow_unvalidated_matrices"] = False

            def fail_if_called(sample_dir: Path, output_path: Path) -> int:
                raise AssertionError("SIESTA should not run when the local sample is already validated")

            module.run_siesta_in_dir = fail_if_called
            argv = sys.argv
            try:
                sys.argv = ["run_single_points.py"]
                self.assertEqual(module.main(), 0)
            finally:
                sys.argv = argv
            summary = json.loads((root / "run_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary[0]["status"], "skipped_validated")

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

    def test_build_common_tests_writes_frozen_hash_manifests(self) -> None:
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            md_sample = make_sample(root / "md", "001")
            atom_sample = make_sample(root / "atom", "001")
            md_test = root / "md_test.csv"
            atom_test = root / "atom_test.csv"
            write_csv(
                md_test,
                [
                    {
                        "sample_id": "md_001",
                        "method": "md",
                        "structure_path": str(md_sample / "RUN.fdf"),
                        "hamiltonian_path": str(md_sample / "siesta.TSHS"),
                        "run_out_path": str(md_sample / "RUN.out"),
                        "status": "valid",
                    }
                ],
            )
            write_csv(
                atom_test,
                [
                    {
                        "sample_id": "atom_001",
                        "method": "atom_displacement",
                        "structure_path": str(atom_sample / "RUN.fdf"),
                        "hamiltonian_path": str(atom_sample / "siesta.TSHS"),
                        "run_out_path": str(atom_sample / "RUN.out"),
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
                "--output-dir",
                str(root / "common"),
                "--mixed-max-per-method",
                "1",
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            frozen_path = root / "common" / "test_mixed" / "frozen_test_manifest.json"
            frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
            self.assertEqual(frozen["sample_count"], 2)
            self.assertTrue(frozen["frozen_test_hash"])
            self.assertTrue(all(sample["structure_sha256"] for sample in frozen["samples"]))
            self.assertTrue(all(sample["hamiltonian_sha256"] for sample in frozen["samples"]))
            summary = json.loads((root / "common" / "common_test_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["test_sets"]["test_mixed"]["frozen_test_hash"], frozen["frozen_test_hash"])

    def test_build_common_tests_supports_nway_method_manifests(self) -> None:
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            manifests = {}
            for method in ("md", "siesta_fc_cartesian", "random_cartesian"):
                sample = make_sample(root / method, "001")
                manifest = root / f"{method}.csv"
                write_csv(
                    manifest,
                    [
                        {
                            "sample_id": f"{method}_001",
                            "method": method,
                            "structure_path": str(sample / "RUN.fdf"),
                            "hamiltonian_path": str(sample / "siesta.TSHS"),
                            "run_out_path": str(sample / "RUN.out"),
                            "status": "valid",
                        }
                    ],
                )
                manifests[method] = manifest
            result = run_script(
                "Comparison/scripts/build_common_tests.py",
                "--test-manifest",
                f"md={manifests['md']}",
                "--test-manifest",
                f"siesta_fc_cartesian={manifests['siesta_fc_cartesian']}",
                "--test-manifest",
                f"random_cartesian={manifests['random_cartesian']}",
                "--output-dir",
                str(root / "common"),
                "--mixed-seed",
                "99",
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            for test_set in ("test_md", "test_siesta_fc_cartesian", "test_random_cartesian", "test_mixed"):
                self.assertTrue((root / "common" / test_set / "test_manifest.csv").exists())
            mixed = json.loads((root / "common" / "test_mixed" / "frozen_test_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(mixed["selected_methods"], ["md", "random_cartesian", "siesta_fc_cartesian"])
            self.assertEqual(mixed["sample_counts_per_method"]["md"], 1)
            self.assertEqual(mixed["sample_counts_per_method"]["random_cartesian"], 1)
            self.assertEqual(mixed["random_seed"], 99)
            self.assertTrue(mixed["composition_hash"])
            self.assertIn("md", mixed["source_manifest_hashes"])

    def test_build_common_tests_accepts_atomdisp_alias_with_method_manifest(self) -> None:
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            sample = make_sample(root / "siesta_fc_cartesian", "001")
            manifest = root / "siesta_fc_cartesian.csv"
            write_csv(
                manifest,
                [
                    {
                        "sample_id": "siesta_fc_cartesian_001",
                        "method": "siesta_fc_cartesian",
                        "structure_path": str(sample / "RUN.fdf"),
                        "hamiltonian_path": str(sample / "siesta.TSHS"),
                        "run_out_path": str(sample / "RUN.out"),
                        "status": "valid",
                    }
                ],
            )
            result = run_script(
                "Comparison/scripts/build_common_tests.py",
                "--test-manifest",
                f"siesta_fc_cartesian={manifest}",
                "--test-sets",
                "test_siesta_fc_cartesian,test_atomdisp",
                "--output-dir",
                str(root / "common"),
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            self.assertTrue((root / "common" / "test_siesta_fc_cartesian" / "test_manifest.csv").exists())
            self.assertTrue((root / "common" / "test_atomdisp" / "test_manifest.csv").exists())

    def test_evaluate_cross_requires_complete_grid_by_default(self) -> None:
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            for name in ("md_on_md", "fc_on_fc"):
                metrics_dir = root / name / "metrics"
                metrics_dir.mkdir(parents=True)
                (metrics_dir / "manifest.json").write_text(
                    json.dumps({"summary": {"sparse": {"mae_ref_eV": {"mean": 1.0}}}}),
                    encoding="utf-8",
                )
            result = run_script(
                "Comparison/scripts/evaluate_cross.py",
                "--md-on-md",
                str(root / "md_on_md"),
                "--fc-on-fc",
                str(root / "fc_on_fc"),
                "--output",
                str(root / "metrics.csv"),
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("Incomplete cross-evaluation grid", result.stdout)

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
                    write_csv(
                        result_dir / "metrics" / "spectral_metrics.csv",
                        [
                            {
                                "sample": "sample_1",
                                "low_energy_n_states": "3",
                                "low_energy_mae_eV": "0.1",
                                "low_energy_rmse_eV": "0.2",
                                "low_energy_max_abs_error_eV": "0.3",
                            }
                        ],
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

    def test_cross_aggregation_contains_nway_method_fields(self) -> None:
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            cross_root = root / "cross"
            methods = ("md", "siesta_fc_cartesian", "random_cartesian")
            test_sets = ("test_md", "test_siesta_fc_cartesian", "test_random_cartesian", "test_mixed")
            dataset_sizes = {"md": 3, "siesta_fc_cartesian": 4, "random_cartesian": 5}
            for method in methods:
                for test_set in test_sets:
                    result_dir = cross_root / f"{method}__on__{test_set}"
                    (result_dir / "metrics").mkdir(parents=True)
                    write_csv(
                        result_dir / "metrics" / "sparse_metrics.csv",
                        [{"sample": "sample_1", "relative_frobenius_union": "1.0"}],
                    )
                    write_csv(
                        result_dir / "metrics" / "spectral_metrics.csv",
                        [
                            {
                                "sample": "sample_1",
                                "low_energy_n_states": "3",
                                "low_energy_mae_eV": "0.1",
                                "low_energy_rmse_eV": "0.2",
                                "low_energy_max_abs_error_eV": "0.3",
                            }
                        ],
                    )
                    (result_dir / "cross_evaluation_manifest.json").write_text(
                        json.dumps(
                            {
                                "train_method": method,
                                "test_set": test_set,
                                "test_method": test_set.removeprefix("test_"),
                                "dataset_size": dataset_sizes[method],
                                "dataset_size_by_method": dataset_sizes,
                                "md_dataset_size": dataset_sizes["md"],
                                "atom_dataset_size": dataset_sizes["siesta_fc_cartesian"],
                                "seed": 42,
                                "model_checkpoint": f"{method}.ckpt",
                            }
                        ),
                        encoding="utf-8",
                    )
            result = run_script(
                "Comparison/scripts/aggregate_cross_metrics.py",
                "--experiment-id",
                "exp_nway",
                "--cross-root",
                str(cross_root),
                "--output-dir",
                str(root / "summary"),
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            with (root / "summary" / "cross_evaluation_metrics.csv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), len(methods) * len(test_sets))
            cells = {(row["train_method"], row["test_set"]) for row in rows}
            self.assertIn(("random_cartesian", "test_siesta_fc_cartesian"), cells)
            random_row = next(row for row in rows if row["train_method"] == "random_cartesian")
            self.assertEqual(random_row["test_method"], random_row["test_set"].removeprefix("test_"))
            self.assertEqual(json.loads(random_row["dataset_size_by_method"]), dataset_sizes)
            self.assertEqual(float(random_row["low_energy_n_states"]), 3.0)
            self.assertEqual(random_row["low_energy_rmse_eV"], "0.2")

    def test_md_blocked_with_gap_separates_splits_and_excludes_gap_frames(self) -> None:
        sys.path.insert(0, str(REPO_ROOT / "MD" / "scripts"))
        spec = importlib.util.spec_from_file_location(
            "generate_md_dataset",
            REPO_ROOT / "MD" / "scripts" / "generate_md_dataset.py",
        )
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        with workspace_tempdir() as tmp:
            root = Path(tmp)
            frames = []
            for index in range(10):
                frame = root / str(index)
                frame.mkdir()
                frames.append(frame)
            splits, excluded = module._split_blocked_with_gap(
                frames,
                {"train": 3, "validation": 2, "test": 2},
                temporal_gap=1,
                block_order=["train", "validation", "test"],
            )
            self.assertEqual([path.name for path in splits["train"]], ["0", "1", "2"])
            self.assertEqual([path.name for path in splits["validation"]], ["4", "5"])
            self.assertEqual([path.name for path in splits["test"]], ["7", "8"])
            self.assertEqual([path.name for path, _reason in excluded], ["3", "6"])
            used = {path.name for samples in splits.values() for path in samples}
            self.assertTrue(used.isdisjoint({path.name for path, _reason in excluded}))
            with self.assertRaisesRegex(RuntimeError, "necesita mas frames"):
                module._split_blocked_with_gap(
                    frames[:5],
                    {"train": 3, "validation": 2, "test": 2},
                    temporal_gap=1,
                    block_order=["train", "validation", "test"],
                )

    def test_atomdisp_grouped_split_keeps_group_ids_isolated(self) -> None:
        sys.path.insert(0, str(REPO_ROOT / "Comparison" / "scripts"))
        spec = importlib.util.spec_from_file_location(
            "pipeline_ui",
            REPO_ROOT / "Comparison" / "scripts" / "pipeline_ui.py",
        )
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        with workspace_tempdir() as tmp:
            root = Path(tmp)
            samples = []
            for index in range(3):
                sample = make_sample(root / "samples", f"{index}")
                (sample / "metadata.json").write_text(
                    json.dumps(
                        {
                            "raw_displacement_run_id": f"run_{index}",
                            "raw_fc_run_dir": f"fc_{index}",
                            "atom": index,
                            "direction": "x",
                            "sign": "+",
                            "displacement_ang": 0.01,
                        }
                    ),
                    encoding="utf-8",
                )
                samples.append(sample)
            splits = module.split_grouped_exact(samples, {"train": 1, "validation": 1, "test": 1})
            groups_by_split = {
                split: {module.atom_split_group_id(sample) for sample in split_samples}
                for split, split_samples in splits.items()
            }
            self.assertTrue(groups_by_split["train"].isdisjoint(groups_by_split["validation"]))
            self.assertTrue(groups_by_split["train"].isdisjoint(groups_by_split["test"]))
            self.assertTrue(groups_by_split["validation"].isdisjoint(groups_by_split["test"]))

            same_group = samples[:2]
            for sample in same_group:
                (sample / "metadata.json").write_text(
                    json.dumps({"raw_displacement_run_id": "same", "atom": 1, "direction": "x", "sign": "+"}),
                    encoding="utf-8",
                )
            with self.assertRaisesRegex(RuntimeError, "sin partir familias"):
                module.split_grouped_exact(same_group, {"train": 1, "validation": 1, "test": 0})

    def test_checkpoint_manifest_is_preferred_over_latest_checkpoint(self) -> None:
        sys.path.insert(0, str(REPO_ROOT / "Comparison" / "scripts"))
        spec = importlib.util.spec_from_file_location(
            "pipeline_ui",
            REPO_ROOT / "Comparison" / "scripts" / "pipeline_ui.py",
        )
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        with workspace_tempdir() as tmp:
            root = Path(tmp)
            training = root / "training"
            manifest_ckpt = training / "lightning_logs" / "version_0" / "checkpoints" / "manifest.ckpt"
            latest_ckpt = training / "lightning_logs" / "version_9" / "checkpoints" / "latest.ckpt"
            manifest_ckpt.parent.mkdir(parents=True)
            latest_ckpt.parent.mkdir(parents=True)
            manifest_ckpt.write_bytes(b"manifest")
            latest_ckpt.write_bytes(b"latest")
            os.utime(manifest_ckpt, (1, 1))
            os.utime(latest_ckpt, (2, 2))
            (training / "checkpoint_manifest.json").write_text(
                json.dumps(
                    {
                        "checkpoint_path": str(manifest_ckpt),
                        "checkpoint_sha256": module.file_sha256(manifest_ckpt),
                    }
                ),
                encoding="utf-8",
            )
            selected = module.find_latest_checkpoint(training, {})
            self.assertEqual(selected, manifest_ckpt)
            self.assertEqual(module.checkpoint_selection_warning(training, selected), "")

            (training / "checkpoint_manifest.json").unlink()
            selected = module.find_latest_checkpoint(training, {})
            self.assertEqual(selected, latest_ckpt)
            self.assertIn("latest-version fallback", module.checkpoint_selection_warning(training, selected))

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

    def test_winner_analysis_inconclusive_on_validation_warnings(self) -> None:
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            metrics = root / "cross_evaluation_metrics.csv"
            rows = []
            for method, value in (("md", "0.5"), ("atom_displacement", "1.0")):
                for test_set in ("test_md", "test_atomdisp", "test_mixed"):
                    rows.append(
                        {
                            "experiment_id": "exp_test",
                            "train_method": method,
                            "test_set": test_set,
                            "md_dataset_size": "10",
                            "atom_dataset_size": "10",
                            "seed": "1",
                            "model_checkpoint": f"{method}.ckpt",
                            "global_rmse_eV": value,
                            "siesta_settings_warning": "settings mismatch",
                        }
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
            recommendation = json.loads((root / "summary" / "recommendation.json").read_text(encoding="utf-8"))
            self.assertEqual(recommendation["status"], "inconclusive")
            self.assertIn("settings mismatch", recommendation["reason"])

    def test_winner_analysis_scientific_status_requires_three_seeds(self) -> None:
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            metrics = root / "cross_evaluation_metrics.csv"
            rows = []
            for seed in ("1", "2"):
                for method, value in (("md", "0.5"), ("atom_displacement", "1.0")):
                    for test_set in ("test_md", "test_atomdisp", "test_mixed"):
                        rows.append(
                            {
                                "experiment_id": "exp_test",
                                "train_method": method,
                                "test_set": test_set,
                                "md_dataset_size": "10",
                                "atom_dataset_size": "10",
                                "seed": seed,
                                "model_checkpoint": f"{method}_{seed}.ckpt",
                                "global_rmse_eV": value,
                            }
                        )
            write_csv(metrics, rows)
            result = run_script(
                "Comparison/scripts/analyze_winners.py",
                "--metrics-csv",
                str(metrics),
                "--output-dir",
                str(root / "summary_two"),
                "--primary-metric",
                "global_rmse_eV",
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            recommendation = json.loads((root / "summary_two" / "recommendation.json").read_text(encoding="utf-8"))
            self.assertEqual(recommendation["scientific_status"], "exploratory")
            self.assertTrue(recommendation["insufficient_robust_seeds"])

            for method, value in (("md", "0.5"), ("atom_displacement", "1.0")):
                for test_set in ("test_md", "test_atomdisp", "test_mixed"):
                    rows.append(
                        {
                            "experiment_id": "exp_test",
                            "train_method": method,
                            "test_set": test_set,
                            "md_dataset_size": "10",
                            "atom_dataset_size": "10",
                            "seed": "3",
                            "model_checkpoint": f"{method}_3.ckpt",
                            "global_rmse_eV": value,
                        }
                    )
            write_csv(metrics, rows)
            result = run_script(
                "Comparison/scripts/analyze_winners.py",
                "--metrics-csv",
                str(metrics),
                "--output-dir",
                str(root / "summary_three"),
                "--primary-metric",
                "global_rmse_eV",
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            recommendation = json.loads((root / "summary_three" / "recommendation.json").read_text(encoding="utf-8"))
            self.assertEqual(recommendation["scientific_status"], "robust_comparison")

    def test_winner_analysis_treats_siesta_fc_cartesian_as_legacy_fc_alias(self) -> None:
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            metrics = root / "cross_evaluation_metrics.csv"
            rows = []
            for seed in ("1", "2", "3"):
                for method, value in (("md", "0.5"), ("siesta_fc_cartesian", "1.0")):
                    for test_set in ("test_md", "test_siesta_fc_cartesian", "test_mixed"):
                        rows.append(
                            {
                                "experiment_id": "exp_test",
                                "train_method": method,
                                "test_set": test_set,
                                "test_method": test_set.removeprefix("test_"),
                                "dataset_size_by_method": json.dumps({"md": 10, "siesta_fc_cartesian": 10}),
                                "seed": seed,
                                "model_checkpoint": f"{method}_{seed}.ckpt",
                                "global_rmse_eV": value,
                            }
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
            recommendation = json.loads((root / "summary" / "recommendation.json").read_text(encoding="utf-8"))
            self.assertEqual(recommendation["scientific_status"], "robust_comparison")
            self.assertEqual(recommendation["status"], "md_conservative_win")

    def test_winner_analysis_does_not_use_global_seed_count_for_underpowered_winner(self) -> None:
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
                            "test_set": "test_md",
                            "md_dataset_size": "10",
                            "atom_dataset_size": "10",
                            "seed": seed,
                            "model_checkpoint": f"md_{seed}.ckpt",
                            "global_rmse_eV": "0.5",
                        },
                        {
                            "experiment_id": "exp_test",
                            "train_method": "atom_displacement",
                            "test_set": "test_md",
                            "md_dataset_size": "10",
                            "atom_dataset_size": "10",
                            "seed": seed,
                            "model_checkpoint": f"ad_{seed}.ckpt",
                            "global_rmse_eV": "1.0",
                        },
                    ]
                )
            for seed in ("1", "2", "3"):
                for test_set in ("test_atomdisp", "test_mixed"):
                    rows.extend(
                        [
                            {
                                "experiment_id": "exp_test",
                                "train_method": "md",
                                "test_set": test_set,
                                "md_dataset_size": "10",
                                "atom_dataset_size": "10",
                                "seed": seed,
                                "model_checkpoint": f"md_{seed}.ckpt",
                                "global_rmse_eV": "1.0",
                            },
                            {
                                "experiment_id": "exp_test",
                                "train_method": "atom_displacement",
                                "test_set": test_set,
                                "md_dataset_size": "10",
                                "atom_dataset_size": "10",
                                "seed": seed,
                                "model_checkpoint": f"ad_{seed}.ckpt",
                                "global_rmse_eV": "1.0" if test_set == "test_mixed" else "0.5",
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
            recommendation = json.loads((root / "summary" / "recommendation.json").read_text(encoding="utf-8"))
            self.assertEqual(recommendation["n_seeds"], 3)
            self.assertEqual(recommendation["scientific_status"], "exploratory")
            self.assertTrue(recommendation["underpowered_stable_wins"])

    def test_winner_analysis_inconclusive_on_checkpoint_fallback_warning(self) -> None:
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            metrics = root / "cross_evaluation_metrics.csv"
            rows = []
            for seed in ("1", "2", "3"):
                for method, value in (("md", "0.5"), ("atom_displacement", "1.0")):
                    for test_set in ("test_md", "test_atomdisp", "test_mixed"):
                        rows.append(
                            {
                                "experiment_id": "exp_test",
                                "train_method": method,
                                "test_set": test_set,
                                "md_dataset_size": "10",
                                "atom_dataset_size": "10",
                                "seed": seed,
                                "model_checkpoint": f"{method}_{seed}.ckpt",
                                "checkpoint_selection_warning": "Checkpoint selected by latest-version fallback",
                                "global_rmse_eV": value,
                            }
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
            recommendation = json.loads((root / "summary" / "recommendation.json").read_text(encoding="utf-8"))
            self.assertEqual(recommendation["status"], "inconclusive")
            self.assertEqual(recommendation["scientific_status"], "scientifically_inconclusive")
            self.assertIn("latest-version fallback", " ".join(recommendation["severe_warnings"]))

    def test_winner_analysis_inconclusive_on_basis_pseudopotential_warning(self) -> None:
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            metrics = root / "cross_evaluation_metrics.csv"
            rows = []
            for seed in ("1", "2", "3"):
                for method, value in (("md", "0.5"), ("atom_displacement", "1.0")):
                    for test_set in ("test_md", "test_atomdisp", "test_mixed"):
                        rows.append(
                            {
                                "experiment_id": "exp_test",
                                "train_method": method,
                                "test_set": test_set,
                                "md_dataset_size": "10",
                                "atom_dataset_size": "10",
                                "seed": seed,
                                "model_checkpoint": f"{method}_{seed}.ckpt",
                                "basis_pseudopotential_warning": "MD and AtomDisplacement basis .ion.xml content hashes differ.",
                                "global_rmse_eV": value,
                            }
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
            recommendation = json.loads((root / "summary" / "recommendation.json").read_text(encoding="utf-8"))
            self.assertEqual(recommendation["status"], "inconclusive")
            self.assertIn("basis", " ".join(recommendation["severe_warnings"]))

    def test_winner_analysis_inconclusive_when_primary_metric_missing_in_required_cell(self) -> None:
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            metrics = root / "cross_evaluation_metrics.csv"
            rows = []
            for method in ("md", "atom_displacement"):
                for test_set in ("test_md", "test_atomdisp", "test_mixed"):
                    row = {
                        "experiment_id": "exp_test",
                        "train_method": method,
                        "test_set": test_set,
                        "md_dataset_size": "10",
                        "atom_dataset_size": "10",
                        "seed": "1",
                        "model_checkpoint": f"{method}.ckpt",
                    }
                    if not (method == "atom_displacement" and test_set == "test_mixed"):
                        row["global_rmse_eV"] = "0.5" if method == "md" else "1.0"
                    rows.append(row)
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
            recommendation = json.loads((root / "summary" / "recommendation.json").read_text(encoding="utf-8"))
            self.assertEqual(recommendation["status"], "insufficient_primary_metric")
            self.assertIn("atom_displacement on test_mixed", recommendation["missing_primary_metric_cells"])

    def test_winner_analysis_low_energy_primary_missing_is_inconclusive(self) -> None:
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            metrics = root / "cross_evaluation_metrics.csv"
            rows = []
            for method in ("md", "atom_displacement"):
                for test_set in ("test_md", "test_atomdisp", "test_mixed"):
                    row = {
                        "experiment_id": "exp_low",
                        "train_method": method,
                        "test_set": test_set,
                        "md_dataset_size": "10",
                        "atom_dataset_size": "10",
                        "seed": "1",
                        "model_checkpoint": f"{method}.ckpt",
                    }
                    if not (method == "atom_displacement" and test_set == "test_mixed"):
                        row["low_energy_rmse_eV"] = "0.2" if method == "md" else "0.4"
                    rows.append(row)
            write_csv(metrics, rows)
            result = run_script(
                "Comparison/scripts/analyze_winners.py",
                "--metrics-csv",
                str(metrics),
                "--output-dir",
                str(root / "summary"),
                "--primary-metric",
                "low_energy_rmse_eV",
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            recommendation = json.loads((root / "summary" / "recommendation.json").read_text(encoding="utf-8"))
            self.assertEqual(recommendation["scientific_status"], "scientifically_inconclusive")
            self.assertEqual(recommendation["status"], "insufficient_primary_metric")
            self.assertIn("atom_displacement on test_mixed", recommendation["missing_primary_metric_cells"])

    def test_winner_analysis_nway_missing_cell_is_inconclusive(self) -> None:
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            metrics = root / "cross_evaluation_metrics.csv"
            rows = []
            methods = ("md", "siesta_fc_cartesian", "random_cartesian")
            test_sets = ("test_md", "test_siesta_fc_cartesian", "test_random_cartesian", "test_mixed")
            for method in methods:
                for test_set in test_sets:
                    if method == "random_cartesian" and test_set == "test_mixed":
                        continue
                    rows.append(
                        {
                            "experiment_id": "exp_nway",
                            "train_method": method,
                            "test_set": test_set,
                            "test_method": test_set.removeprefix("test_"),
                            "dataset_size_by_method": json.dumps(
                                {"md": 3, "siesta_fc_cartesian": 4, "random_cartesian": 5}
                            ),
                            "seed": "1",
                            "model_checkpoint": f"{method}.ckpt",
                            "global_rmse_eV": "0.5" if method == "md" else "1.0",
                        }
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
            recommendation = json.loads((root / "summary" / "recommendation.json").read_text(encoding="utf-8"))
            self.assertEqual(recommendation["scientific_status"], "scientifically_inconclusive")
            self.assertIn("random_cartesian on test_mixed", recommendation["missing_required_cells"])
            self.assertIn("test_random_cartesian", recommendation["rankings_by_test_set"])

    def test_winner_analysis_nway_complete_reports_consensus_leader(self) -> None:
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            metrics = root / "cross_evaluation_metrics.csv"
            rows = []
            methods = ("md", "siesta_fc_cartesian", "random_cartesian")
            test_sets = ("test_md", "test_siesta_fc_cartesian", "test_random_cartesian", "test_mixed")
            values = {"md": "0.4", "siesta_fc_cartesian": "0.8", "random_cartesian": "1.2"}
            for method in methods:
                for test_set in test_sets:
                    rows.append(
                        {
                            "experiment_id": "exp_nway_complete",
                            "train_method": method,
                            "test_set": test_set,
                            "test_method": test_set.removeprefix("test_"),
                            "dataset_size_by_method": json.dumps(
                                {"md": 3, "siesta_fc_cartesian": 4, "random_cartesian": 5}
                            ),
                            "seed": "1",
                            "model_checkpoint": f"{method}.ckpt",
                            "global_rmse_eV": values[method],
                        }
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
            recommendation = json.loads((root / "summary" / "recommendation.json").read_text(encoding="utf-8"))
            self.assertEqual(recommendation["status"], "nway_consensus_win")
            self.assertEqual(recommendation["scientific_status"], "exploratory")
            self.assertEqual(recommendation["nway_consensus_leader"], "md")
            self.assertEqual(set(recommendation["nway_leaders_by_test_set"].values()), {"md"})
            self.assertNotIn("legacy global-winner", recommendation["reason"])

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

    def test_geometry_leakage_detects_rotated_translated_duplicate(self) -> None:
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            train = make_sample(root / "samples", "train")
            test = make_sample(root / "samples", "test")
            rotated = "\n".join(
                [
                    "SystemLabel water",
                    "%block AtomicCoordinatesAndAtomicSpecies",
                    "1.0 1.0 0.0 1",
                    "0.3 1.0 0.0 2",
                    "1.7 1.0 0.0 2",
                    "%endblock AtomicCoordinatesAndAtomicSpecies",
                    "",
                ]
            )
            (test / "RUN.fdf").write_text(rotated, encoding="utf-8")
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
                "--aligned-rmsd-threshold",
                "0.001",
                "--distance-threshold",
                "0.001",
            )
            self.assertEqual(result.returncode, 2)
            summary = json.loads((root / "leakage" / "geometry_leakage_summary.json").read_text(encoding="utf-8"))
            self.assertGreaterEqual(summary["aligned_near_duplicates"], 1)
            self.assertGreaterEqual(summary["internal_distance_near_duplicates"], 1)

    def test_geometry_leakage_detects_atom_displacement_family_aliases(self) -> None:
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            train = make_sample(root / "samples", "train")
            test = make_sample(root / "samples", "test")
            common = {
                "method": "atom_displacement",
                "raw_displacement_run_id": "fc_001",
                "atom": "2",
                "direction": "1",
                "sign": "-1",
                "displacement_ang": "0.03",
            }
            train_manifest = root / "train.csv"
            test_manifest = root / "test.csv"
            write_csv(
                train_manifest,
                [{**common, "sample_id": "train", "structure_path": str(train / "RUN.fdf")}],
            )
            write_csv(
                test_manifest,
                [{**common, "sample_id": "test", "structure_path": str(test / "RUN.fdf")}],
            )
            result = run_script(
                "Comparison/scripts/check_geometry_leakage.py",
                "--train-manifest",
                str(train_manifest),
                "--test-manifest",
                str(test_manifest),
                "--output-dir",
                str(root / "leakage"),
            )
            self.assertEqual(result.returncode, 2)
            summary = json.loads((root / "leakage" / "geometry_leakage_summary.json").read_text(encoding="utf-8"))
            self.assertGreaterEqual(summary["atom_displacement_family_warnings"], 1)

    def test_geometry_leakage_allows_distinct_random_cartesian_samples_from_same_group(self) -> None:
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            train = make_sample(root / "samples", "train")
            test = make_sample(root / "samples", "test")
            text = (test / "RUN.fdf").read_text(encoding="utf-8").replace("0.0 0.7 0.0 2", "0.0 0.76 0.0 2")
            (test / "RUN.fdf").write_text(text, encoding="utf-8")
            train_metadata = train / "metadata.json"
            test_metadata = test / "metadata.json"
            common = {
                "base_geometry_hash": "base",
                "distribution": "gaussian",
                "sigma_ang": 0.03,
                "seed": 1234,
                "split_group_id": "same-generation-group",
            }
            train_metadata.write_text(json.dumps({**common, "id": "sample_000001", "sample_index": 0}), encoding="utf-8")
            test_metadata.write_text(json.dumps({**common, "id": "sample_000002", "sample_index": 1}), encoding="utf-8")
            train_manifest = root / "train.csv"
            test_manifest = root / "test.csv"
            write_csv(
                train_manifest,
                [
                    {
                        "sample_id": "random_sample_000001",
                        "method": "random_cartesian",
                        "structure_path": str(train / "RUN.fdf"),
                        "metadata_path": str(train_metadata),
                        "split_group_id": "same-generation-group",
                    }
                ],
            )
            write_csv(
                test_manifest,
                [
                    {
                        "sample_id": "random_sample_000002",
                        "method": "random_cartesian",
                        "structure_path": str(test / "RUN.fdf"),
                        "metadata_path": str(test_metadata),
                        "split_group_id": "same-generation-group",
                    }
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
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            summary = json.loads((root / "leakage" / "geometry_leakage_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["random_cartesian_family_warnings"], 0)

    def test_geometry_leakage_detects_same_random_cartesian_sample_index(self) -> None:
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            train = make_sample(root / "samples", "train")
            test = make_sample(root / "samples", "test")
            text = (test / "RUN.fdf").read_text(encoding="utf-8").replace("0.0 0.7 0.0 2", "0.0 0.76 0.0 2")
            (test / "RUN.fdf").write_text(text, encoding="utf-8")
            train_metadata = train / "metadata.json"
            test_metadata = test / "metadata.json"
            common = {
                "base_geometry_hash": "base",
                "distribution": "gaussian",
                "sigma_ang": 0.03,
                "seed": 1234,
                "sample_index": 7,
            }
            train_metadata.write_text(json.dumps({**common, "id": "sample_000008"}), encoding="utf-8")
            test_metadata.write_text(json.dumps({**common, "id": "sample_000008"}), encoding="utf-8")
            train_manifest = root / "train.csv"
            test_manifest = root / "test.csv"
            write_csv(
                train_manifest,
                [{"sample_id": "train", "method": "random_cartesian", "structure_path": str(train / "RUN.fdf"), "metadata_path": str(train_metadata)}],
            )
            write_csv(
                test_manifest,
                [{"sample_id": "test", "method": "random_cartesian", "structure_path": str(test / "RUN.fdf"), "metadata_path": str(test_metadata)}],
            )
            result = run_script(
                "Comparison/scripts/check_geometry_leakage.py",
                "--train-manifest",
                str(train_manifest),
                "--test-manifest",
                str(test_manifest),
                "--output-dir",
                str(root / "leakage"),
            )
            self.assertEqual(result.returncode, 2)
            summary = json.loads((root / "leakage" / "geometry_leakage_summary.json").read_text(encoding="utf-8"))
            self.assertGreaterEqual(summary["random_cartesian_family_warnings"], 1)

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

    def test_cross_predict_parser_accepts_all_method_labels(self) -> None:
        spec = importlib.util.spec_from_file_location(
            "predict_model_on_dataset",
            REPO_ROOT / "Comparison" / "scripts" / "predict_model_on_dataset.py",
        )
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        parser = module.build_parser()
        for method in ("md", "atom_displacement", "siesta_fc_cartesian", "random_cartesian"):
            args = parser.parse_args(
                [
                    "--checkpoint",
                    "model.ckpt",
                    "--train-method",
                    method,
                    "--test-set",
                    "test_md",
                    "--test-manifest",
                    "test.csv",
                    "--output-dir",
                    "out",
                    "--basis-files",
                    "basis/*.ion.xml",
                ]
            )
            self.assertEqual(args.train_method, method)

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
        md = {
            "md": {
                "lattice_constant": 15,
                "lattice_vectors": [[15, 0, 0], [0, 15, 0], [0, 0, 15]],
                "basis_type": "split",
                "basis_size": "DZP",
                "energy_shift": "0.03 eV",
                "mesh_cutoff": "200 Ry",
                "xc_functional": "GGA",
                "xc_authors": "PBE",
                "max_scf_iterations": 200,
                "solution_method": "diagon",
                "dm_mixing_weight": 0.02,
                "dm_number_pulay": 3,
                "dm_tolerance": "1.d-5",
                "dm_require_energy_convergence": "T",
                "dm_energy_tolerance": "1.e-5 eV",
                "spin_polarized": "F",
                "fix_spin": "F",
                "non_collinear_spin": "F",
                "force_aux_cell": False,
                "save_hs_file": True,
                "save_hs": True,
            }
        }
        atom = {
            "structure": {
                "lattice_constant": 15,
                "lattice_vectors": [[15, 0, 0], [0, 15, 0], [0, 0, 15]],
                "force_constants": {"save_tshs": True},
                "siesta": {
                    "ForceAuxCell": "F",
                    "Save.HS": "T",
                    "MeshCutoff": "200 Ry",
                    "PAO.BasisType": "split",
                    "PAO.BasisSize": "DZP",
                    "PAO.EnergyShift": "0.03 eV",
                    "XC.functional": "GGA",
                    "XC.authors": "PBE",
                    "MaxSCFIterations": 200,
                    "SolutionMethod": "diagon",
                    "DM.MixingWeight": 0.02,
                    "DM.NumberPulay": 3,
                    "DM.Tolerance": "1.d-5",
                    "DM.Require.Energy.Convergence": "T",
                    "DM.Energy.Tolerance": "1.e-5 eV",
                    "SpinPolarized": "F",
                    "FixSpin": "F",
                    "NonCollinearSpin": "F",
                },
            }
        }
        ok_report = module.compare_settings(md, atom, shared)
        self.assertTrue(ok_report["ok"])
        atom_bad = json.loads(json.dumps(atom))
        atom_bad["structure"]["siesta"]["MeshCutoff"] = "300 Ry"
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

    def test_model_settings_detects_hyperparameter_mismatch(self) -> None:
        sys.path.insert(0, str(REPO_ROOT / "Comparison" / "scripts"))
        spec = importlib.util.spec_from_file_location(
            "model_settings",
            REPO_ROOT / "Comparison" / "scripts" / "model_settings.py",
        )
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        md = {"training": {"data": {"out_matrix": "hamiltonian", "batch_size": 8}, "model": {"optim_lr": 0.005}, "trainer": {"max_epochs": 10}}}
        atom = {"training": {"data": {"out_matrix": "hamiltonian", "batch_size": 16}, "model": {"optim_lr": 0.005}, "trainer": {"max_epochs": 10}}}
        report = module.compare_model_settings(md, atom)
        self.assertFalse(report["ok"])
        self.assertEqual(report["mismatches"][0]["section"], "data")

    def test_write_graph2mat_configs_uses_split_paths_by_default(self) -> None:
        sys.path.insert(0, str(REPO_ROOT / "Comparison" / "scripts"))
        spec = importlib.util.spec_from_file_location(
            "write_graph2mat_configs",
            REPO_ROOT / "Comparison" / "scripts" / "write_graph2mat_configs.py",
        )
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        block = {
            "data": {
                "out_matrix": "hamiltonian",
                "symmetric_matrix": True,
                "basis_files": "old",
                "train_runs": "old",
                "batch_size": 2,
                "store_in_memory": True,
            },
            "model": {},
            "trainer": {"logger": {"init_args": {"name": "x"}}},
        }
        md_block = json.loads(json.dumps(block))
        fc_block = json.loads(json.dumps(block))
        module.force_shared_hyperparams(md_block, fc_block)
        self.assertEqual(md_block["data"]["train_runs"], "../MD/dataset/splits/train/*/RUN.fdf")
        self.assertEqual(fc_block["data"]["train_runs"], "../AtomDisplacement/dataset/train_samples/*/RUN.fdf")
        module.force_shared_hyperparams(md_block, fc_block, debug_full_dataset_globs=True)
        self.assertEqual(md_block["data"]["train_runs"], "../MD/dataset/MD_steps/*/RUN.fdf")

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

    def test_path_tokens_are_resolved_in_config_loaders(self) -> None:
        spec = importlib.util.spec_from_file_location(
            "md_pipeline_config",
            REPO_ROOT / "MD" / "scripts" / "md_pipeline_config.py",
        )
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        old_venv = os.environ.get("GRAPH2MAT_VENV")
        os.environ["GRAPH2MAT_VENV"] = "/tmp/graph2mat_venv"
        try:
            spec.loader.exec_module(module)
            self.assertEqual(module.expand_path_tokens("${GRAPH2MAT_VENV}/bin/activate"), "/tmp/graph2mat_venv/bin/activate")
            self.assertIn(str(REPO_ROOT), module.expand_path_tokens("${REPO_ROOT}/dataset"))
        finally:
            if old_venv is None:
                os.environ.pop("GRAPH2MAT_VENV", None)
            else:
                os.environ["GRAPH2MAT_VENV"] = old_venv

    def test_structural_metric_outputs_are_part_of_metrics_manifest_schema(self) -> None:
        script = (REPO_ROOT / "Comparison" / "scripts" / "evaluate_hamiltonian_metrics.py").read_text(encoding="utf-8")
        self.assertIn("block_metrics.csv", script)
        self.assertIn("species_pair_metrics.csv", script)
        self.assertIn("distance_bin_metrics.csv", script)
        self.assertIn("structural_metrics_available", script)
        self.assertIn("basis_orbital_counts", script)
        self.assertIn("Basis files are required", script)
        self.assertIn("2 * angular_momentum + 1", script)

    def test_low_energy_metrics_from_synthetic_diagonal_hamiltonians(self) -> None:
        try:
            import numpy as np
            from scipy import sparse
        except ModuleNotFoundError as exc:
            self.skipTest(f"scientific Python dependency unavailable: {exc.name}")

        sys.path.insert(0, str(REPO_ROOT / "Comparison" / "scripts"))
        spec = importlib.util.spec_from_file_location(
            "evaluate_hamiltonian_metrics",
            REPO_ROOT / "Comparison" / "scripts" / "evaluate_hamiltonian_metrics.py",
        )
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        def matrix_data(values, *, overlap=None, orthogonal=True):
            return module.MatrixData(
                path=Path("synthetic.HSX"),
                hamiltonian=sparse.diags(values, format="csr"),
                overlap=overlap,
                own_eigenvalues=np.asarray(values, dtype=float),
                fermi_level=None,
                fermi_level_source="unavailable",
                orthogonal=orthogonal,
                has_overlap=overlap is not None,
                overlap_error=None,
            )

        reference = matrix_data([0.0, 1.0, 3.0, 5.0])
        predicted = matrix_data([0.1, 0.8, 4.0, 6.0])
        metrics = module.low_energy_metrics(reference, predicted, n_states=3)
        self.assertEqual(metrics["low_energy_n_states"], 3)
        self.assertAlmostEqual(metrics["low_energy_mae_eV"], (0.1 + 0.2 + 1.0) / 3.0)
        self.assertAlmostEqual(metrics["low_energy_rmse_eV"], ((0.01 + 0.04 + 1.0) / 3.0) ** 0.5)
        self.assertAlmostEqual(metrics["low_energy_max_abs_error_eV"], 1.0)

        fewer = module.low_energy_metrics(reference, predicted, n_states=10)
        self.assertEqual(fewer["low_energy_n_states"], 4)

    def test_frontier_metrics_use_homo_lumo_when_fermi_window_is_empty(self) -> None:
        try:
            import numpy as np
        except ModuleNotFoundError as exc:
            self.skipTest(f"scientific Python dependency unavailable: {exc.name}")

        sys.path.insert(0, str(REPO_ROOT / "Comparison" / "scripts"))
        spec = importlib.util.spec_from_file_location(
            "evaluate_hamiltonian_metrics_frontier",
            REPO_ROOT / "Comparison" / "scripts" / "evaluate_hamiltonian_metrics.py",
        )
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        _, metrics = module.eigen_error_metrics(
            np.asarray([-10.0, -8.0, 8.0, 10.0], dtype=float),
            np.asarray([-11.0, -6.0, 11.0, 9.0], dtype=float),
            0.0,
            "siesta_file",
        )
        self.assertEqual(metrics["fermi_window_bands"], 0)
        self.assertTrue(np.isnan(metrics["fermi_window_rmse_eV"]))
        self.assertEqual(metrics["frontier_window_bands"], 2)
        self.assertAlmostEqual(metrics["frontier_window_mae_eV"], 2.5)
        self.assertAlmostEqual(metrics["frontier_window_rmse_eV"], ((2.0**2 + 3.0**2) / 2.0) ** 0.5)

    def test_low_energy_metrics_warn_when_required_overlap_is_missing(self) -> None:
        import math
        try:
            import numpy as np
            from scipy import sparse
        except ModuleNotFoundError as exc:
            self.skipTest(f"scientific Python dependency unavailable: {exc.name}")

        sys.path.insert(0, str(REPO_ROOT / "Comparison" / "scripts"))
        spec = importlib.util.spec_from_file_location(
            "evaluate_hamiltonian_metrics_missing_overlap",
            REPO_ROOT / "Comparison" / "scripts" / "evaluate_hamiltonian_metrics.py",
        )
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        reference = module.MatrixData(
            path=Path("reference.HSX"),
            hamiltonian=sparse.diags([0.0, 1.0], format="csr"),
            overlap=None,
            own_eigenvalues=np.asarray([0.0, 1.0], dtype=float),
            fermi_level=None,
            fermi_level_source="unavailable",
            orthogonal=False,
            has_overlap=False,
            overlap_error="missing overlap",
        )
        predicted = module.MatrixData(
            path=Path("predicted.HSX"),
            hamiltonian=sparse.diags([0.1, 1.2], format="csr"),
            overlap=None,
            own_eigenvalues=np.asarray([0.1, 1.2], dtype=float),
            fermi_level=None,
            fermi_level_source="unavailable",
            orthogonal=False,
            has_overlap=False,
            overlap_error="missing overlap",
        )
        metrics = module.low_energy_metrics(reference, predicted, n_states=2)
        self.assertIsNone(metrics["low_energy_n_states"])
        self.assertTrue(math.isnan(metrics["low_energy_rmse_eV"]))
        self.assertIn("reference overlap is required", metrics["low_energy_warning"])

    def test_ui_and_metrics_docs_expose_low_energy_metric(self) -> None:
        index_html = (REPO_ROOT / "Comparison" / "ui" / "index.html").read_text(encoding="utf-8")
        app_js = (REPO_ROOT / "Comparison" / "ui" / "app.js").read_text(encoding="utf-8")
        metrics_doc = (REPO_ROOT / "Comparison" / "METRICS.md").read_text(encoding="utf-8")
        self.assertIn("low_energy_rmse_eV", index_html)
        self.assertIn("low_energy_rmse_eV", app_js)
        self.assertIn("low_energy_mae_eV", metrics_doc)
        self.assertIn("A lower matrix MAE/RMSE does not necessarily imply a better", metrics_doc)

    def test_strict_ui_does_not_allow_missing_atomdisp_matrices(self) -> None:
        script = (REPO_ROOT / "Comparison" / "scripts" / "pipeline_ui.py").read_text(encoding="utf-8")
        self.assertIn('config["structure"]["force_constants"]["allow_missing_matrix"] = False', script)
        self.assertNotIn('config["structure"]["force_constants"]["allow_missing_matrix"] = True', script)

    def test_checkpoint_manifest_written_by_training_scripts(self) -> None:
        for path in (
            REPO_ROOT / "MD" / "scripts" / "run_md_training.py",
            REPO_ROOT / "MD" / "scripts" / "run_md_testing.py",
            REPO_ROOT / "AtomDisplacement" / "scripts" / "run_atdisp_training.py",
            REPO_ROOT / "AtomDisplacement" / "scripts" / "run_atdisp_testing.py",
        ):
            text = path.read_text(encoding="utf-8")
            self.assertIn("write_checkpoint_manifest", text, str(path))
            self.assertIn("selection_metric", text, str(path))

    def test_ui_records_nested_subset_metadata_and_warnings(self) -> None:
        script = (REPO_ROOT / "Comparison" / "scripts" / "pipeline_ui.py").read_text(encoding="utf-8")
        self.assertIn("parent_dataset_size", script)
        self.assertIn("nested_subset_hash", script)
        self.assertIn("nested_subset_warning", script)

    def test_geometry_leakage_uses_kabsch_when_numpy_available(self) -> None:
        script = (REPO_ROOT / "Comparison" / "scripts" / "check_geometry_leakage.py").read_text(encoding="utf-8")
        self.assertIn("np.linalg.svd", script)
        self.assertIn("distance_signature_max_diff", script)

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
