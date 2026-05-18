from __future__ import annotations

import csv
import contextlib
import copy
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
                "Save.HS T",
                "XML.Write T",
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


def make_reusable_md_dataset(root: Path, *, n_samples: int = 3) -> Path:
    dataset_dir = root / "dataset"
    (dataset_dir / "MD_steps" / "basis").mkdir(parents=True)
    (dataset_dir / "MD_steps" / "basis" / "H.ion.xml").write_text("<ion />\n", encoding="utf-8")
    rows: list[dict[str, str]] = []
    rows_by_split: dict[str, list[dict[str, str]]] = {"train": [], "validation": [], "test": []}
    split_names = ["train"] * max(1, n_samples - 2) + ["validation", "test"]
    for index, split_name in enumerate(split_names[:n_samples]):
        raw_dir = dataset_dir / "MD_steps" / str(index)
        raw_dir.mkdir(parents=True)
        minimal_run_fdf(raw_dir / "RUN.fdf")
        (raw_dir / "siesta.TSHS").write_bytes(b"matrix")
        (raw_dir / "metadata.json").write_text("{}", encoding="utf-8")
        sample_dir = dataset_dir / "splits" / split_name / str(index)
        sample_dir.mkdir(parents=True)
        minimal_run_fdf(sample_dir / "RUN.fdf")
        (sample_dir / "siesta.TSHS").write_bytes(b"matrix")
        (sample_dir / "metadata.json").write_text("{}", encoding="utf-8")
        rows_for_split = [
            {
                "sample_id": f"md_{index}",
                "method": "md",
                "structure_path": str(sample_dir / "RUN.fdf"),
                "hamiltonian_path": str(sample_dir / "siesta.TSHS"),
                "run_out_path": str(dataset_dir / "RUN.out"),
                "status": "completed",
                "valid": "true",
                "split": split_name,
                "sample_dir": str(sample_dir),
            }
        ]
        rows_by_split[split_name].extend(rows_for_split)
        rows.extend(rows_for_split)
    for split_name, split_rows in rows_by_split.items():
        if split_rows:
            write_csv(dataset_dir / "splits" / f"{split_name}_manifest.csv", split_rows)
    (dataset_dir / "RUN.out").write_text("Job completed\nSCF cycle converged\n", encoding="utf-8")
    (dataset_dir / "run_summary.json").write_text(
        json.dumps({"samples": [{"status": "skipped_validated", "valid": True} for _row in rows]}),
        encoding="utf-8",
    )
    return dataset_dir


def make_reusable_atom_like_dataset(root: Path, *, source_root_name: str = "FC_steps") -> Path:
    dataset_dir = root / "dataset"
    (dataset_dir / "basis").mkdir(parents=True)
    (dataset_dir / "basis" / "H.ion.xml").write_text("<ion />\n", encoding="utf-8")
    (dataset_dir / source_root_name).mkdir(parents=True)
    for split_name, index in (("train", 0), ("validation", 1), ("test", 2)):
        sample_dir = dataset_dir / f"{split_name}_samples" / f"{index:03d}"
        sample_dir.mkdir(parents=True)
        minimal_run_fdf(sample_dir / "RUN.fdf")
        (sample_dir / "siesta.TSHS").write_bytes(b"matrix")
        (sample_dir / "metadata.json").write_text("{}", encoding="utf-8")
        write_csv(
            dataset_dir / "splits" / f"{split_name}_manifest.csv",
            [
                {
                    "sample_id": f"sample_{index:03d}",
                    "method": "random_cartesian" if source_root_name == "RandomCartesian_steps" else "atom_displacement",
                    "structure_path": str(sample_dir / "RUN.fdf"),
                    "hamiltonian_path": str(sample_dir / "siesta.TSHS"),
                    "run_out_path": str(sample_dir / "RUN.out"),
                    "metadata_path": str(sample_dir / "metadata.json"),
                    "status": "completed",
                    "valid": "true",
                    "split": split_name,
                    "sample_dir": str(sample_dir),
                }
            ],
        )
        (sample_dir / "RUN.out").write_text("Job completed\nSCF cycle converged\n", encoding="utf-8")
    return dataset_dir


def write_archived_retrainable_run(
    results_root: Path,
    *,
    method_id: str,
    dataset_label: str,
    dataset_size: int,
    dataset_dir: Path,
    run_id: str = "source_experiment",
) -> Path:
    result_groups = {
        "md": "results_md",
        "siesta_fc_cartesian": "results_atomdisp",
        "random_cartesian": "results_random_cartesian",
    }
    pipelines = {
        "md": "md",
        "siesta_fc_cartesian": "atom_displacement",
        "random_cartesian": "random_cartesian",
    }
    result_dir = results_root / result_groups[method_id] / dataset_label / f"run_{run_id}"
    result_dir.mkdir(parents=True)
    manifest = {
        "pipeline": pipelines[method_id],
        "method_id": method_id,
        "dataset_label": dataset_label,
        "dataset_size": dataset_size,
        "requested_dataset_size": dataset_size,
        "effective_dataset_size": dataset_size,
        "returncode": 0,
        "run_id": run_id,
        "dataset_dir": str(dataset_dir),
        "result_dir": str(result_dir),
        "run_mode": "full_strict_pipeline",
    }
    (result_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    write_csv(
        result_dir / "metrics" / "spectral_metrics.csv",
        [{"sample": "sample_1", "low_energy_rmse_eV": "0.1"}],
    )
    return result_dir / "manifest.json"


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

    def load_method_registry_module(self):
        return self.load_module_from_path(
            "method_registry_test",
            REPO_ROOT / "Comparison" / "scripts" / "method_registry.py",
        )

    def load_module_from_path(self, name: str, path: Path):
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    def load_metrics_module(self, name: str = "evaluate_hamiltonian_metrics_test"):
        sys.path.insert(0, str(REPO_ROOT / "Comparison" / "scripts"))
        return self.load_module_from_path(
            name,
            REPO_ROOT / "Comparison" / "scripts" / "evaluate_hamiltonian_metrics.py",
        )

    def load_model_settings_module(self, name: str = "model_settings_test"):
        sys.path.insert(0, str(REPO_ROOT / "Comparison" / "scripts"))
        return self.load_module_from_path(
            name,
            REPO_ROOT / "Comparison" / "scripts" / "model_settings.py",
        )

    def load_md_dataset_module(self, name: str = "generate_md_dataset_test"):
        sys.path.insert(0, str(REPO_ROOT / "MD" / "scripts"))
        return self.load_module_from_path(
            name,
            REPO_ROOT / "MD" / "scripts" / "generate_md_dataset.py",
        )

    def test_ui_log_payload_is_bounded_for_polling(self):
        module = self.load_pipeline_ui_module()
        logs = [f"line {index}\n" for index in range(10)]

        payload = module.bounded_log_payload(logs, since=0, limit=3)

        self.assertEqual(payload["offset"], 10)
        self.assertTrue(payload["truncated"])
        self.assertEqual(payload["dropped_lines"], 7)
        self.assertEqual(payload["returned_from"], 7)
        self.assertIn("Log truncado", payload["lines"][0])
        self.assertEqual(payload["lines"][-1], "line 9\n")

        current_payload = module.bounded_log_payload(logs, since=8, limit=3)
        self.assertFalse(current_payload["truncated"])
        self.assertEqual(current_payload["lines"], ["line 8\n", "line 9\n"])

    def write_expected_cross_grid(
        self,
        path: Path,
        methods: tuple[str, ...],
        test_sets: tuple[str, ...],
        *,
        expected_context_cells: list[dict[str, object]] | None = None,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        expected_cells = [
            {"train_method": method, "test_set": test_set, "cell_id": f"{method} on {test_set}"}
            for method in methods
            for test_set in test_sets
        ]
        payload = {
            "experiment_id": "exp_grid",
            "canonical_method_ids": ["md", "siesta_fc_cartesian", "random_cartesian"],
            "selected_methods": list(methods),
            "selected_frozen_test_sets": list(test_sets),
            "expected_cell_count": len(expected_cells),
            "expected_cells": expected_cells,
        }
        if expected_context_cells is not None:
            payload["expected_context_cell_count"] = len(expected_context_cells)
            payload["expected_context_cells"] = expected_context_cells
        path.write_text(
            json.dumps(payload),
            encoding="utf-8",
        )

    def write_completeness_report(
        self,
        path: Path,
        methods: tuple[str, ...],
        test_sets: tuple[str, ...],
        *,
        primary_metric: str = "global_rmse_eV",
        missing_cells: tuple[str, ...] = (),
        missing_primary_metric_cells: tuple[str, ...] = (),
        extra_unexpected_cells: tuple[str, ...] = (),
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        expected_cells = [f"{method} on {test_set}" for method in methods for test_set in test_sets]
        missing = set(missing_cells)
        extras = set(extra_unexpected_cells)
        actual_cells = sorted((set(expected_cells) - missing) | extras)
        complete = not missing_cells and not missing_primary_metric_cells and not extra_unexpected_cells
        path.write_text(
            json.dumps(
                {
                    "experiment_id": "exp_grid",
                    "primary_metric": primary_metric,
                    "expected_cell_count": len(expected_cells),
                    "actual_cell_count": len(actual_cells),
                    "expected_cells": expected_cells,
                    "actual_cells": actual_cells,
                    "missing_cells": list(missing_cells),
                    "extra_unexpected_cells": list(extra_unexpected_cells),
                    "missing_primary_metric_cells": list(missing_primary_metric_cells),
                    "complete": complete,
                    "scientific_status": "valid_grid" if complete else "invalid_incomplete_grid",
                }
            ),
            encoding="utf-8",
        )

    def write_cross_metric_cell(
        self,
        cross_root: Path,
        method: str,
        test_set: str,
        *,
        pair_id: str | None = None,
        result_name: str | None = None,
        primary_metric: str = "low_energy_rmse_eV",
        primary_value: str | None = "0.2",
    ) -> None:
        result_dir = cross_root / (result_name or f"{method}__on__{test_set}")
        (result_dir / "metrics").mkdir(parents=True)
        write_csv(
            result_dir / "metrics" / "sparse_metrics.csv",
            [{"sample": "sample_1", "relative_frobenius_union": "1.0"}],
        )
        spectral_row = {
            "sample": "sample_1",
            "low_energy_n_states": "3",
            "low_energy_mae_eV": "0.1",
            "low_energy_max_abs_error_eV": "0.3",
        }
        if primary_value is not None:
            spectral_row[primary_metric] = primary_value
        write_csv(result_dir / "metrics" / "spectral_metrics.csv", [spectral_row])
        (result_dir / "cross_evaluation_manifest.json").write_text(
            json.dumps(
                {
                    **({"pair_id": pair_id} if pair_id else {}),
                    "train_method": method,
                    "test_set": test_set,
                    "dataset_size": 3,
                    "seed": 42,
                    "model_checkpoint": f"{method}.ckpt",
                }
            ),
            encoding="utf-8",
        )

    def write_final_recommendation_metric_grid(
        self,
        path: Path,
        *,
        methods: tuple[str, ...] = ("md", "siesta_fc_cartesian", "random_cartesian"),
        test_sets: tuple[str, ...] = (
            "test_md",
            "test_siesta_fc_cartesian",
            "test_random_cartesian",
            "test_mixed",
        ),
        seeds: tuple[str, ...] = ("1", "2", "3"),
        primary_metric: str = "low_energy_rmse_eV",
        value_for=None,
        dataset_sizes: dict[str, int] | None = None,
        extra_for=None,
        skip_cell=None,
    ) -> list[dict[str, str]]:
        dataset_sizes = dataset_sizes or {
            "md": 1000,
            "siesta_fc_cartesian": 750,
            "random_cartesian": 3500,
        }
        rows = []
        for seed in seeds:
            for method in methods:
                for test_set in test_sets:
                    if skip_cell and skip_cell(method, test_set, seed):
                        continue
                    row = {
                        "experiment_id": "exp_final_recommendation",
                        "train_method": method,
                        "test_set": test_set,
                        "dataset_size_by_method": json.dumps(dataset_sizes),
                        "dataset_label_by_method": json.dumps(
                            {item: f"{item}_{size}" for item, size in dataset_sizes.items()}
                        ),
                        "recipe_hash_by_method": json.dumps(
                            {item: f"{item}_hash_{size}" for item, size in dataset_sizes.items()}
                        ),
                        "seed": seed,
                        "model_checkpoint": f"{method}_{seed}.ckpt",
                    }
                    if value_for:
                        value = value_for(method, test_set, seed)
                    else:
                        value = "0.5" if method == "md" else "1.0"
                    if value is not None:
                        row[primary_metric] = str(value)
                    if extra_for:
                        row.update(extra_for(method, test_set, seed) or {})
                    rows.append(row)
        write_csv(path, rows)
        return rows

    def siesta_settings_fixture(self) -> tuple[dict, dict, dict]:
        shared = {"MeshCutoff": "200 Ry"}
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
                "save_de": True,
                "xml_write": True,
            }
        }
        atom = {
            "structure": {
                "lattice_constant": 15,
                "lattice_vectors": [[15, 0, 0], [0, 15, 0], [0, 0, 15]],
                "force_constants": {"save_tshs": True, "save_tsde": True},
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
                    "XML.Write": "T",
                },
            }
        }
        return shared, md, atom

    def model_settings_fixture(self) -> tuple[dict, dict, dict]:
        base_training = {
            "torch_float32_matmul_precision": "high",
            "data": {
                "out_matrix": "hamiltonian",
                "symmetric_matrix": True,
                "sub_point_matrix": False,
                "n_matrix_components": 2,
                "basis_files": "../dataset/basis/*.ion.xml",
                "train_runs": "../dataset/train/*/RUN.fdf",
                "batch_size": 8,
                "store_in_memory": True,
            },
            "model": {
                "num_interactions": 1,
                "correlation": 1,
                "max_ell": 2,
                "hidden_irreps": "10x0e + 10x1o + 10x2e",
                "loss": "graph2mat.metrics.block_type_mae",
                "optim_lr": 0.005,
            },
            "trainer": {
                "accelerator": "cpu",
                "logger": {
                    "class_path": "TensorBoardLogger",
                    "init_args": {"name": "method_model", "save_dir": "lightning_logs"},
                },
                "max_epochs": 100,
            },
        }
        md = {"training": copy.deepcopy(base_training)}
        fc = {"training": copy.deepcopy(base_training)}
        rc = {"training": copy.deepcopy(base_training)}
        fc["training"]["data"]["basis_files"] = "../fc/basis/*.ion.xml"
        fc["training"]["data"]["runs_json"] = "../fc/runs.json"
        fc["training"]["trainer"]["logger"]["init_args"]["name"] = "fc_model"
        rc["training"]["data"]["basis_files"] = "../rc/basis/*.ion.xml"
        rc["training"]["data"]["train_runs"] = "../rc/train/*/RUN.fdf"
        rc["training"]["trainer"]["logger"]["init_args"]["name"] = "rc_model"
        return md, fc, rc

    def provenance_run_fixture(
        self,
        root: Path,
        method: str,
        *,
        size: int = 3,
        label: str | None = None,
        checkpoint_hash: str | None = "checkpoint_hash",
        legacy_pipeline_only: bool = False,
    ) -> dict:
        method_id = "siesta_fc_cartesian" if method in {"atom_displacement", "atomdisp"} else method
        pipeline = "atom_displacement" if method_id == "siesta_fc_cartesian" else method_id
        label = label or f"{method_id}_dataset"
        result_dir = root / method_id / label
        split_dir = result_dir / "splits"
        split_dir.mkdir(parents=True)
        for split in ("train", "validation", "test"):
            write_csv(split_dir / f"{split}_manifest.csv", [{"sample_id": f"{split}_1"}])
        checkpoint_path = result_dir / "training" / "best.ckpt"
        checkpoint_path.parent.mkdir(parents=True)
        checkpoint_path.write_bytes(b"checkpoint")
        run = {
            "pipeline": pipeline,
            "dataset_label": label,
            "dataset_size": size,
            "effective_dataset_size": size,
            "result_dir": str(result_dir),
            "model_checkpoint": str(checkpoint_path),
            "model_checkpoint_sha256": checkpoint_hash,
            "checkpoint_manifest": str(result_dir / "training" / "checkpoint_manifest.json"),
            "checkpoint_selection_warning": "",
            "artifact_hashes": {
                "basis": f"{method_id}_basis_hash",
                "pseudopotentials": f"{method_id}_pseudo_hash",
            },
            "recipe_id": f"{method_id}_recipe",
            "recipe_label": f"{method_id} recipe",
            "recipe_set_hash": f"{method_id}_recipe_hash",
            "returncode": 0,
            "seed": 42,
        }
        if not legacy_pipeline_only:
            run["method_id"] = method_id
        return run

    def test_method_selection_normalization_and_validation(self) -> None:
        module = self.load_pipeline_ui_module()
        self.assertEqual(
            module.normalize_selected_methods(["md", "atom_displacement"]),
            ["md", "siesta_fc_cartesian"],
        )
        self.assertEqual(
            module.normalize_selected_methods(["atomdisp", "siesta_fc_cartesian", "random_cartesian"]),
            ["siesta_fc_cartesian", "random_cartesian"],
        )
        self.assertEqual(module.normalize_selected_methods(None), ["md", "siesta_fc_cartesian"])
        with self.assertRaisesRegex(RuntimeError, "Selecciona al menos un metodo"):
            module.normalize_selected_methods([])
        with self.assertRaisesRegex(RuntimeError, "no soportados"):
            module.normalize_selected_methods(["md", "bogus"])
        self.assertEqual(module.normalize_selected_methods(["random_cartesian"]), ["random_cartesian"])

    def test_method_registry_canonical_ids_and_aliases(self) -> None:
        module = self.load_method_registry_module()
        self.assertEqual(
            tuple(module.METHOD_REGISTRY),
            ("md", "siesta_fc_cartesian", "random_cartesian"),
        )
        self.assertTrue(module.METHOD_REGISTRY["md"].is_baseline)
        self.assertFalse(module.METHOD_REGISTRY["random_cartesian"].is_baseline)
        self.assertEqual(module.METHOD_REGISTRY["siesta_fc_cartesian"].results_dir, "results_atomdisp")
        self.assertEqual(module.normalize_method_id("atom_displacement"), "siesta_fc_cartesian")
        self.assertEqual(module.normalize_method_id("atomdisp"), "siesta_fc_cartesian")
        self.assertEqual(module.get_method("atomdisp").method_id, "siesta_fc_cartesian")
        self.assertEqual(module.get_method("random_cartesian").frozen_test_set, "test_random_cartesian")
        self.assertEqual(module.normalize_test_set_id("test_atomdisp"), "test_siesta_fc_cartesian")
        with self.assertRaisesRegex(ValueError, "Unknown scientific method"):
            module.normalize_method_id("bogus")

    def test_run_mode_validation(self) -> None:
        module = self.load_pipeline_ui_module()
        self.assertEqual(module.parse_run_mode(None), "full_strict_pipeline")
        self.assertEqual(module.parse_run_mode("dataset_only"), "dataset_only")
        self.assertEqual(
            module.parse_run_mode("train_test_metrics_plots_only"),
            "train_test_metrics_plots_only",
        )
        with self.assertRaisesRegex(RuntimeError, "run_mode"):
            module.parse_run_mode("training_only")

    def test_training_plan_validation(self) -> None:
        module = self.load_pipeline_ui_module()
        plan = module.parse_training_plan(
            [
                {
                    "label": "epochs400",
                    "reusable_dataset_ids": ["run_a", "run_b"],
                    "training_settings": {"max_epochs": 400},
                },
                {
                    "label": "epochs600",
                    "reusable_dataset_ids": "run_a",
                    "training_settings": {"max_epochs": 600, "optim_lr": 0.001},
                },
            ]
        )
        self.assertEqual([item["label"] for item in plan], ["epochs400", "epochs600"])
        self.assertEqual(plan[0]["training_settings"], {"max_epochs": 400})
        self.assertEqual(plan[1]["reusable_dataset_ids"], ["run_a"])
        full_plan = module.parse_training_plan(
            [
                {
                    "label": "strict400",
                    "dataset_targets": [
                        {"target_id": "md:md_100", "method_id": "md", "recipe_id": "md_100"}
                    ],
                    "training_settings": {"max_epochs": 400},
                }
            ]
        )
        self.assertEqual(full_plan[0]["dataset_targets"][0]["target_id"], "md:md_100")
        with self.assertRaisesRegex(RuntimeError, "training_plan"):
            module.parse_training_plan({"bad": True})
        with self.assertRaisesRegex(RuntimeError, "reusable_dataset_ids o dataset_targets"):
            module.parse_training_plan([{"label": "empty"}])
        with self.assertRaisesRegex(RuntimeError, "dataset_only"):
            module.ExperimentRunner().start(
                [3],
                [],
                selected_methods=["md"],
                run_mode="dataset_only",
                training_plan=[
                    {
                        "label": "epochs400",
                        "dataset_targets": ["md:md_3"],
                        "training_settings": {"max_epochs": 400},
                    }
                ],
            )
        with self.assertRaisesRegex(RuntimeError, "dataset_targets"):
            module.ExperimentRunner().start(
                [3],
                [],
                selected_methods=["md"],
                run_mode="full_strict_pipeline",
                training_plan=[
                    {
                        "label": "epochs400",
                        "reusable_dataset_ids": ["run_a"],
                        "training_settings": {"max_epochs": 400},
                    }
                ],
            )

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

    def test_pipeline_config_path_env_selects_snapshot_config(self) -> None:
        md_module = self.load_module_from_path(
            "md_pipeline_config_snapshot_test",
            REPO_ROOT / "MD" / "scripts" / "md_pipeline_config.py",
        )
        atom_module = self.load_module_from_path(
            "atom_pipeline_config_snapshot_test",
            REPO_ROOT / "AtomDisplacement" / "scripts" / "pipeline_config_utils.py",
        )
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            md_snapshot = root / "md_snapshot.yaml"
            atom_snapshot = root / "atom_snapshot.yaml"
            shutil.copyfile(REPO_ROOT / "MD" / "pipeline_config.yaml", md_snapshot)
            shutil.copyfile(REPO_ROOT / "AtomDisplacement" / "pipeline_config.yaml", atom_snapshot)
            old_env = os.environ.get("PIPELINE_CONFIG_PATH")
            try:
                os.environ["PIPELINE_CONFIG_PATH"] = str(md_snapshot)
                md_config = md_module.load_pipeline_config()
                self.assertEqual(Path(md_config["_config_path"]), md_snapshot)
                self.assertEqual(Path(md_config["_config_dir"]), root)
                os.environ["PIPELINE_CONFIG_PATH"] = str(atom_snapshot)
                atom_config = atom_module.load_pipeline_config()
                self.assertEqual(Path(atom_config["_config_path"]), atom_snapshot)
                self.assertEqual(Path(atom_config["_config_dir"]), root)
            finally:
                if old_env is None:
                    os.environ.pop("PIPELINE_CONFIG_PATH", None)
                else:
                    os.environ["PIPELINE_CONFIG_PATH"] = old_env

    def test_performance_settings_validation_env_and_config_application(self) -> None:
        module = self.load_pipeline_ui_module()
        default_settings = module.parse_performance_settings(None)
        self.assertEqual(default_settings["preset"], "balanced")
        self.assertIn(default_settings["compute_accelerator"], {"cpu", "gpu"})
        self.assertGreaterEqual(default_settings["batch_size"], 32)
        self.assertEqual(default_settings["torch_float32_matmul_precision"], "high")
        self.assertGreaterEqual(default_settings["max_parallel_siesta_jobs"], 1)
        self.assertEqual(default_settings["max_parallel_dataset_jobs"], 1)
        fake_hardware = {
            "cpu_physical_cores": 16,
            "cpu_logical_cores": 32,
            "ram_total_gb": 128,
            "cuda_available": True,
            "gpu_name": "NVIDIA GeForce RTX 5090",
            "gpu_vram_total_gb": 32,
            "torch_available": True,
            "torch_cuda_available": True,
        }
        catalog = module.performance_preset_catalog(fake_hardware)
        self.assertEqual(catalog["default_preset"], "balanced")
        self.assertIn("dynamic_gpu_focused", [item["id"] for item in catalog["dynamic_profiles"]])
        self.assertEqual(
            next(item for item in catalog["presets"] if item["id"] == "balanced")["settings"]["batch_size"],
            128,
        )
        self.assertEqual(
            next(item for item in catalog["presets"] if item["id"] == "gpu_focused")["settings"]["compute_accelerator"],
            "gpu",
        )
        resolved, auto_settings = module.preset_settings_by_id("auto_detect", fake_hardware)
        self.assertEqual(resolved, "balanced")
        self.assertEqual(auto_settings["compute_accelerator"], "gpu")
        cpu_settings = module.parse_performance_settings({"preset": "cpu_only"})
        self.assertEqual(cpu_settings["compute_accelerator"], "cpu")
        settings = module.parse_performance_settings(
            {
                "max_parallel_siesta_jobs": "2",
                "max_parallel_dataset_jobs": "3",
                "max_parallel_prediction_jobs": "4",
                "max_parallel_evaluation_jobs": "5",
                "max_parallel_metric_jobs": "6",
                "omp_num_threads": "3",
                "mkl_num_threads": None,
                "openblas_num_threads": "",
                "numexpr_num_threads": "7",
                "torch_num_threads": "4",
                "compute_accelerator": "auto",
                "batch_size": "16",
                "store_in_memory": "false",
                "reuse_validated_siesta_outputs": "true",
                "enable_experiment_cache": "false",
                "error_policy": "continue_on_error",
                "torch_float32_matmul_precision": "high",
            }
        )
        self.assertEqual(settings["max_parallel_siesta_jobs"], 2)
        self.assertEqual(settings["max_parallel_dataset_jobs"], 3)
        self.assertEqual(settings["max_parallel_prediction_jobs"], 4)
        self.assertEqual(settings["max_parallel_evaluation_jobs"], 5)
        self.assertEqual(settings["max_parallel_metric_jobs"], 6)
        self.assertEqual(settings["compute_accelerator"], "auto")
        self.assertEqual(settings["batch_size"], 16)
        self.assertIs(settings["store_in_memory"], False)
        self.assertIs(settings["reuse_validated_siesta_outputs"], True)
        self.assertIs(settings["enable_experiment_cache"], False)
        self.assertEqual(settings["error_policy"], "continue_on_error")
        env = module.performance_env(settings)
        self.assertEqual(env["OMP_NUM_THREADS"], "3")
        self.assertEqual(env["NUMEXPR_NUM_THREADS"], "7")
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
        self.assertFalse(config["single_points"]["rerun"])
        with self.assertRaisesRegex(RuntimeError, "performance.max_parallel_siesta_jobs"):
            module.parse_performance_settings({"max_parallel_siesta_jobs": 0})
        with self.assertRaisesRegex(RuntimeError, "error_policy"):
            module.parse_performance_settings({"error_policy": "ignore"})
        with self.assertRaisesRegex(RuntimeError, "enable_experiment_cache"):
            module.parse_performance_settings({"enable_experiment_cache": True})
        with self.assertRaisesRegex(RuntimeError, "store_in_memory"):
            module.parse_performance_settings({"store_in_memory": "maybe"})

    def test_training_settings_parse_and_apply_to_pipeline_config(self) -> None:
        module = self.load_pipeline_ui_module()
        settings = module.parse_training_settings(
            {
                "max_epochs": "50",
                "optim_lr": "0.001",
                "batch_size": "32",
                "loader_threads": "8",
                "num_interactions": "2",
                "correlation": "3",
                "max_ell": "4",
                "hidden_irreps": "32x0e + 32x1o + 32x2e + 32x3o + 32x4e",
                "loss": "graph2mat.metrics.block_type_mae",
            }
        )
        config = {"training": {"data": {}, "model": {}, "trainer": {}}}
        module.apply_training_settings_to_config(config, settings)
        self.assertEqual(config["training"]["trainer"]["max_epochs"], 50)
        self.assertEqual(config["training"]["data"]["batch_size"], 32)
        self.assertEqual(config["training"]["data"]["loader_threads"], 8)
        self.assertEqual(config["training"]["model"]["num_interactions"], 2)
        self.assertEqual(config["training"]["model"]["correlation"], 3)
        self.assertEqual(config["training"]["model"]["max_ell"], 4)
        self.assertEqual(config["training"]["model"]["hidden_irreps"], "32x0e + 32x1o + 32x2e + 32x3o + 32x4e")
        self.assertEqual(config["training"]["model"]["optim_lr"], 0.001)
        self.assertEqual(config["training"]["ui_training_settings"], settings)
        with self.assertRaisesRegex(RuntimeError, "training_settings.max_epochs"):
            module.parse_training_settings({"max_epochs": 0})
        with self.assertRaisesRegex(RuntimeError, "training_settings.optim_lr"):
            module.parse_training_settings({"optim_lr": -0.1})
        with self.assertRaisesRegex(RuntimeError, "training_settings.max_ell"):
            module.parse_training_settings({"max_ell": -1})
        with self.assertRaisesRegex(RuntimeError, "training_settings.loader_threads"):
            module.parse_training_settings({"loader_threads": 0})
        with self.assertRaisesRegex(RuntimeError, "misma multiplicidad"):
            module.parse_training_settings(
                {"max_ell": 2, "hidden_irreps": "32x0e + 16x1o + 32x2e"}
            )
        with self.assertRaisesRegex(RuntimeError, "lmax=1"):
            module.parse_training_settings(
                {"max_ell": 2, "hidden_irreps": "32x0e + 32x1o"}
            )
        with self.assertRaisesRegex(RuntimeError, "Faltan|faltan"):
            module.parse_training_settings(
                {"max_ell": 2, "hidden_irreps": "32x0e + 32x2e"}
            )
        with self.assertRaisesRegex(RuntimeError, "misma multiplicidad"):
            module.parse_training_settings({"hidden_irreps": "32x0e + 24x1o"})
        with self.assertRaisesRegex(RuntimeError, "Irreps valido"):
            module.parse_training_settings({"hidden_irreps": "32 0e"})

    def test_md_graph2mat_training_config_excludes_ui_metadata(self) -> None:
        sys.path.insert(0, str(REPO_ROOT / "MD" / "scripts"))
        module = self.load_module_from_path(
            "md_pipeline_config_render_test",
            REPO_ROOT / "MD" / "scripts" / "md_pipeline_config.py",
        )
        rendered = module.render_training_config(
            {
                "training": {
                    "torch_float32_matmul_precision": "high",
                    "data": {
                        "out_matrix": "hamiltonian",
                        "batch_size": 4,
                        "n_matrix_components": 2,
                    },
                    "model": {"optim_lr": 0.001},
                    "trainer": {"max_epochs": 12},
                    "optimizer": {"class_path": "Adam"},
                    "ui_training_settings": {"max_epochs": 12, "batch_size": 4},
                }
            }
        )
        self.assertIn("data:", rendered)
        self.assertIn("model:", rendered)
        self.assertIn("trainer:", rendered)
        self.assertIn("optimizer:", rendered)
        self.assertIn("n_matrix_components: 2", rendered)
        self.assertIn("max_epochs: 12", rendered)
        self.assertNotIn("ui_training_settings", rendered)
        self.assertNotIn("torch_float32_matmul_precision", rendered)

    def test_default_venv_command_uses_repo_local_venv(self) -> None:
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

    def test_graph2mat_venv_token_requires_environment_override(self) -> None:
        module = self.load_pipeline_ui_module()
        old_venv = os.environ.pop("GRAPH2MAT_VENV", None)
        try:
            with self.assertRaisesRegex(RuntimeError, "GRAPH2MAT_VENV"):
                module.resolve_venv_activate_from_command("source ${GRAPH2MAT_VENV}/bin/activate")
            os.environ["GRAPH2MAT_VENV"] = "/tmp/graph2mat_venv"
            self.assertEqual(
                module.resolve_venv_activate_from_command("source ${GRAPH2MAT_VENV}/bin/activate"),
                "${GRAPH2MAT_VENV}/bin/activate",
            )
        finally:
            if old_venv is not None:
                os.environ["GRAPH2MAT_VENV"] = old_venv
            else:
                os.environ.pop("GRAPH2MAT_VENV", None)

    def test_versioned_pipeline_configs_do_not_default_to_local_absolute_venv(self) -> None:
        for config_path in (
            REPO_ROOT / "MD" / "pipeline_config.yaml",
            REPO_ROOT / "AtomDisplacement" / "pipeline_config.yaml",
        ):
            config_text = config_path.read_text(encoding="utf-8")
            self.assertIn("venv_activate: ${REPO_ROOT}/.venv/bin/activate", config_text)
            self.assertNotIn("/home/christian", config_text)

    def test_random_cartesian_options_support_multiple_sizes(self) -> None:
        module = self.load_pipeline_ui_module()
        self.assertEqual(module.random_cartesian_sizes_from_options({"n_structures": 3}), [3])
        self.assertEqual(module.random_cartesian_sizes_from_options({"n_structures": [11, 23, 30]}), [11, 23, 30])
        self.assertEqual(module.random_cartesian_sizes_from_options({"n_structures": "11, 23, 30"}), [11, 23, 30])

    def test_dataset_recipes_parse_specs_and_accept_md_temperature_blocks(self) -> None:
        module = self.load_pipeline_ui_module()
        recipes = {
            "md": [
                {
                    "recipe_id": "md_plain",
                    "label": "MD plain",
                    "blocks": [
                        {"block_id": "md_300", "n_snapshots": 3, "temperature_K": 300, "seed": 1},
                        {"block_id": "md_700", "n_snapshots": 6, "temperature_K": 700, "seed": 2},
                    ],
                }
            ],
            "random_cartesian": [
                {
                    "recipe_id": "rc_sigma",
                    "blocks": [
                        {
                            "block_id": "rc_sigma_0p03",
                            "n_structures": 6,
                            "distribution": "gaussian",
                            "sigma_ang": 0.03,
                            "seed": 7,
                        }
                    ],
                }
            ],
        }
        info = module.dataset_recipes_to_execution_specs(
            recipes,
            selected_methods=["md", "random_cartesian"],
            split_ratios={"train": 1 / 3, "validation": 1 / 3, "test": 1 / 3},
            random_cartesian_defaults={"min_distance_ang": 0.65},
        )
        self.assertIsNotNone(info)
        self.assertEqual(info["md_dataset_specs"][0]["size"], 9)
        self.assertEqual(
            [block["temperature_K"] for block in info["md_dataset_specs"][0]["temperature_blocks"]],
            [300.0, 700.0],
        )
        self.assertEqual(
            [block["n_snapshots"] for block in info["md_dataset_specs"][0]["temperature_blocks"]],
            [3, 6],
        )
        self.assertRegex(info["md_dataset_specs"][0]["label"], r"^MD_dataset1_[A-Za-z0-9]{6}$")
        self.assertEqual(info["random_cartesian_dataset_specs"][0]["size"], 6)
        self.assertEqual(info["random_cartesian_dataset_specs"][0]["options"]["sigma_ang"], 0.03)
        self.assertRegex(info["random_cartesian_dataset_specs"][0]["label"], r"^RC_dataset1_[A-Za-z0-9]{6}$")

    def test_dataset_recipes_preserve_random_cartesian_nested_components(self) -> None:
        module = self.load_pipeline_ui_module()
        recipes = {
            "random_cartesian": [
                {
                    "recipe_id": "rc_components",
                    "blocks": [
                        {
                            "block_id": "rc_bond",
                            "n_structures": 6,
                            "components": {
                                "atom_displacement": {"enabled": False},
                                "bond_displacement": {
                                    "enabled": True,
                                    "distribution": "uniform",
                                    "min_delta_ang": 0.02,
                                    "max_delta_ang": 0.02,
                                },
                            },
                        },
                        {
                            "block_id": "rc_atom",
                            "n_structures": 3,
                            "components": {
                                "atom_displacement": {
                                    "enabled": True,
                                    "distribution": "gaussian",
                                    "sigma_ang": 0.01,
                                },
                                "angle_displacement": {
                                    "enabled": True,
                                    "distribution": "uniform",
                                    "min_delta_deg": -5.0,
                                    "max_delta_deg": -5.0,
                                },
                            },
                        },
                    ],
                }
            ]
        }
        info = module.dataset_recipes_to_execution_specs(
            recipes,
            selected_methods=["random_cartesian"],
            split_ratios={"train": 1 / 3, "validation": 1 / 3, "test": 1 / 3},
            random_cartesian_defaults={"distribution": "gaussian", "sigma_ang": 0.03, "seed": 7},
        )
        rc_spec = info["random_cartesian_dataset_specs"][0]
        block = rc_spec["options"]["blocks"][0]
        self.assertFalse(block["components"]["atom_displacement"]["enabled"])
        self.assertTrue(block["components"]["bond_displacement"]["enabled"])
        self.assertEqual(block["components"]["bond_displacement"]["min_delta_ang"], 0.02)
        second_block = rc_spec["options"]["blocks"][1]
        self.assertTrue(second_block["components"]["atom_displacement"]["enabled"])
        self.assertTrue(second_block["components"]["angle_displacement"]["enabled"])
        self.assertEqual(second_block["components"]["atom_displacement"]["sigma_ang"], 0.01)
        self.assertEqual(second_block["components"]["angle_displacement"]["min_delta_deg"], -5.0)
        self.assertNotIn("components", rc_spec["options"])

    def test_composite_dataset_recipes_create_one_run_per_recipe(self) -> None:
        module = self.load_pipeline_ui_module()
        module.atom_fc_sample_limit = lambda _config: 100
        recipes = {
            "md": [
                {
                    "recipe_id": "md_dataset_1",
                    "blocks": [
                        {"n_snapshots": 3, "temperature_K": 300},
                        {"n_snapshots": 3, "temperature_K": 500},
                    ],
                },
                {
                    "recipe_id": "md_dataset_2",
                    "blocks": [
                        {"n_snapshots": 6, "temperature_K": 300},
                        {"n_snapshots": 3, "temperature_K": 700},
                    ],
                },
            ],
            "siesta_fc_cartesian": [
                {
                    "recipe_id": "fc_dataset_1",
                    "blocks": [
                        {"n_structures": 3, "displacement": "0.02 Ang"},
                        {"n_structures": 3, "displacement": "0.04 Ang"},
                    ],
                }
            ],
            "random_cartesian": [
                {
                    "recipe_id": "rc_dataset_1",
                    "blocks": [
                        {"n_structures": 3, "max_displacement": "0.02 Ang", "seed": 11},
                        {"n_structures": 6, "max_displacement": "0.05 Ang", "seed": 12},
                    ],
                }
            ],
        }
        info = module.dataset_recipes_to_execution_specs(
            recipes,
            selected_methods=["md", "siesta_fc_cartesian", "random_cartesian"],
            split_ratios={"train": 1 / 3, "validation": 1 / 3, "test": 1 / 3},
            random_cartesian_defaults={"distribution": "gaussian", "sigma_ang": 0.03, "seed": 7},
        )
        self.assertEqual(len(info["md_dataset_specs"]), 2)
        self.assertEqual([spec["size"] for spec in info["md_dataset_specs"]], [6, 9])
        self.assertEqual(len(info["atom_dataset_specs"]), 1)
        self.assertEqual(info["atom_dataset_specs"][0]["size"], 6)
        self.assertEqual(len(info["atom_dataset_specs"][0]["recipe_metadata"]["blocks"]), 2)
        self.assertEqual(len(info["random_cartesian_dataset_specs"]), 1)
        rc_spec = info["random_cartesian_dataset_specs"][0]
        self.assertEqual(rc_spec["size"], 9)
        self.assertEqual(len(rc_spec["options"]["blocks"]), 2)
        self.assertEqual(rc_spec["options"]["blocks"][0]["n_structures"], 3)
        self.assertEqual(rc_spec["options"]["blocks"][1]["max_displacement"], "0.05 Ang")
        self.assertEqual(len(rc_spec["recipe_metadata"]["blocks"]), 2)
        labels = (
            [spec["label"] for spec in info["md_dataset_specs"]]
            + [spec["label"] for spec in info["atom_dataset_specs"]]
            + [spec["label"] for spec in info["random_cartesian_dataset_specs"]]
        )
        self.assertEqual(len(labels), len(set(labels)))
        self.assertEqual(len({label.rsplit("_", 1)[-1] for label in labels}), len(labels))
        self.assertRegex(labels[0], r"^MD_dataset1_[A-Za-z0-9]{6}$")
        self.assertRegex(labels[1], r"^MD_dataset2_[A-Za-z0-9]{6}$")
        self.assertRegex(labels[2], r"^FC_dataset1_[A-Za-z0-9]{6}$")
        self.assertRegex(labels[3], r"^RC_dataset1_[A-Za-z0-9]{6}$")

    def test_dataset_recipe_seeds_propagate_per_dataset_for_all_methods(self) -> None:
        module = self.load_pipeline_ui_module()
        module.atom_fc_sample_limit = lambda _config: 100
        recipes = {
            "md": [
                {
                    "recipe_id": "md_seeded",
                    "seed": 101,
                    "blocks": [{"n_snapshots": 3, "temperature_K": 300}],
                }
            ],
            "siesta_fc_cartesian": [
                {
                    "recipe_id": "fc_seeded",
                    "seed": 202,
                    "blocks": [{"n_structures": 3, "displacement": "0.02 Ang"}],
                }
            ],
            "random_cartesian": [
                {
                    "recipe_id": "rc_seeded",
                    "seed": 303,
                    "blocks": [{"n_structures": 3, "sigma_ang": 0.03}],
                }
            ],
        }
        info = module.dataset_recipes_to_execution_specs(
            recipes,
            selected_methods=["md", "siesta_fc_cartesian", "random_cartesian"],
            split_ratios={"train": 1 / 3, "validation": 1 / 3, "test": 1 / 3},
            random_cartesian_defaults={"distribution": "gaussian", "sigma_ang": 0.01, "seed": 999},
        )
        self.assertEqual(info["md_dataset_specs"][0]["recipe_metadata"]["seed"], 101)
        self.assertEqual(info["md_dataset_specs"][0]["temperature_blocks"][0]["seed"], 101)
        self.assertEqual(info["atom_dataset_specs"][0]["recipe_metadata"]["seed"], 202)
        rc_spec = info["random_cartesian_dataset_specs"][0]
        self.assertEqual(rc_spec["recipe_metadata"]["seed"], 303)
        self.assertEqual(rc_spec["options"]["seed"], 303)
        self.assertNotIn("seed", rc_spec["options"]["blocks"][0])

        with workspace_tempdir() as tmp:
            root = Path(tmp)
            module.RESULTS_ROOT = root / "results"
            module.WORKSPACES_ROOT = root / "workspaces"
            runner = module.ExperimentRunner()
            calls: dict[str, list] = {"run_one": [], "random": []}

            def fake_run_one(key, size, run_id, **kwargs):
                calls["run_one"].append((key, kwargs))
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
                    "pipeline_elapsed_seconds": 0.1,
                    "siesta_counts": {"launched": 0, "skipped_or_reused": 0, "failed": 0},
                    "seed": (kwargs.get("recipe_metadata") or {}).get("seed"),
                }

            def fake_run_random(size, run_id, **kwargs):
                calls["random"].append((size, kwargs))
                return {
                    "pipeline": "random_cartesian",
                    "method_id": "random_cartesian",
                    "dataset_label": kwargs.get("dataset_label") or f"dataset_{size}",
                    "dataset_size": size,
                    "returncode": 0,
                    "result_dir": str(root / "fake" / "random_cartesian" / f"dataset_{size}"),
                    "dataset_sample_ids": [f"rc-{size}"],
                    "dataset_sample_hash": f"hash-rc-{size}",
                    "run_mode": kwargs.get("run_mode"),
                    "pipeline_elapsed_seconds": 0.1,
                    "siesta_counts": {"launched": 0, "skipped_or_reused": 0, "failed": 0},
                    "seed": (kwargs.get("recipe_metadata") or {}).get("seed"),
                }

            runner._run_one = fake_run_one  # type: ignore[method-assign]
            runner._run_random_cartesian = fake_run_random  # type: ignore[method-assign]
            runner._run(
                [3],
                [3],
                "seeded_case",
                split_ratios={"train": 1 / 3, "validation": 1 / 3, "test": 1 / 3},
                selected_methods=["md", "siesta_fc_cartesian", "random_cartesian"],
                run_mode="dataset_only",
                random_seed=42,
                random_cartesian_options={"distribution": "gaussian", "sigma_ang": 0.01, "seed": 999},
                dataset_recipes_info=info,
            )
            md_call = next(kwargs for key, kwargs in calls["run_one"] if key == "md")
            fc_call = next(kwargs for key, kwargs in calls["run_one"] if key == "atom_displacement")
            self.assertEqual(md_call["md_temperature_blocks"][0]["seed"], 101)
            self.assertEqual(fc_call["random_seed"], 202)
            self.assertEqual(calls["random"][0][1]["random_cartesian_options"]["seed"], 303)
            manifest = module.load_config(root / "results" / "seeded_case" / "experiment_manifest.yaml")
            self.assertEqual(manifest["seeds"], [101, 202, 303, 42])

    def test_fc_dataset_labels_are_compact_for_many_displacements(self) -> None:
        module = self.load_pipeline_ui_module()
        module.atom_fc_sample_limit = lambda _config: 100
        blocks = [
            {"n_structures": 3, "displacement": f"{0.01 + index * 0.005:.3f} Ang"}
            for index in range(24)
        ]
        recipes = {"siesta_fc_cartesian": [{"recipe_id": "fc_many_distances", "blocks": blocks}]}
        info = module.dataset_recipes_to_execution_specs(
            recipes,
            selected_methods=["siesta_fc_cartesian"],
            split_ratios={"train": 1 / 3, "validation": 1 / 3, "test": 1 / 3},
            random_cartesian_defaults={},
        )
        label = info["atom_dataset_specs"][0]["label"]
        self.assertLessEqual(len(label), module.MAX_DATASET_LABEL_LENGTH)
        self.assertRegex(label, r"^FC_dataset1_[A-Za-z0-9]{6}$")
        self.assertEqual(info["atom_dataset_specs"][0]["size"], 72)

        displacement_options = {
            f"{0.01 + index * 0.005:.3f} Ang": [3]
            for index in range(24)
        }
        legacy_specs = module.build_fc_aligned_dataset_specs(
            displacement_options,
            per_displacement_limit=100,
            split_ratios={"train": 1 / 3, "validation": 1 / 3, "test": 1 / 3},
            max_datasets=10,
        )
        self.assertLessEqual(len(legacy_specs[0]["label"]), module.MAX_DATASET_LABEL_LENGTH)
        self.assertRegex(legacy_specs[0]["label"], r"^FC_dataset1_[A-Za-z0-9]{6}$")

    def test_cross_evaluation_names_are_compact(self) -> None:
        module = self.load_pipeline_ui_module()
        long_label = "fc_" + "_".join(f"disp_{index}_0_0{index}" for index in range(30))
        combo = {
            "md": {
                "dataset_label": "MD_dataset1_ABC123",
                "dataset_short_id": "ABC123",
                "dataset_size": 10,
                "result_dir": "/tmp/md",
            },
            "siesta_fc_cartesian": {
                "dataset_label": long_label,
                "dataset_short_id": "DEF456",
                "dataset_size": 72,
                "result_dir": "/tmp/fc",
            },
            "random_cartesian": {
                "dataset_label": "RC_dataset1_GHI789",
                "dataset_short_id": "GHI789",
                "dataset_size": 10,
                "result_dir": "/tmp/rc",
            },
        }
        pair_id = module.cross_pair_id(
            combo,
            ["md", "siesta_fc_cartesian", "random_cartesian"],
        )
        cross_name = module.cross_result_name(pair_id, "random_cartesian", "test_random_cartesian")
        self.assertRegex(pair_id, r"^cross_[0-9a-f]{12}$")
        self.assertEqual(cross_name, f"{pair_id}__rc__on__test_rc")
        self.assertLessEqual(len(cross_name), 48)
        self.assertNotIn(long_label, cross_name)
        test_set_names = {
            module.cross_result_name(pair_id, "md", test_set)
            for test_set in (
                "test_md",
                "test_siesta_fc_cartesian",
                "test_atomdisp",
                "test_random_cartesian",
                "test_mixed",
            )
        }
        self.assertEqual(len(test_set_names), 4)
        self.assertIn(f"{pair_id}__md__on__test_fc", test_set_names)
        self.assertNotIn(f"{pair_id}__md__on__test_ad", test_set_names)

    def test_common_test_set_aliases_are_deduplicated(self) -> None:
        module = self.load_pipeline_ui_module()
        self.assertEqual(
            module.deduplicate_common_test_sets(
                ["test_md", "test_siesta_fc_cartesian", "test_atomdisp", "test_mixed"]
            ),
            ["test_md", "test_siesta_fc_cartesian", "test_mixed"],
        )
        self.assertEqual(
            module.deduplicate_common_test_sets(["test_md", "test_atomdisp", "test_mixed"]),
            ["test_md", "test_siesta_fc_cartesian", "test_mixed"],
        )

    def test_expected_cross_grid_for_two_methods_with_and_without_mixed(self) -> None:
        module = self.load_pipeline_ui_module()
        grid = module.build_cross_evaluation_expected_grid(
            ["md", "atomdisp"],
            ["test_md", "test_atomdisp"],
            experiment_id="exp_two",
        )
        self.assertEqual(grid["experiment_id"], "exp_two")
        self.assertEqual(grid["canonical_method_ids"], ["md", "siesta_fc_cartesian", "random_cartesian"])
        self.assertEqual(grid["selected_methods"], ["md", "siesta_fc_cartesian"])
        self.assertEqual(grid["selected_frozen_test_sets"], ["test_md", "test_siesta_fc_cartesian"])
        self.assertEqual(grid["expected_cell_count"], 4)
        self.assertEqual(
            {(cell["train_method"], cell["test_set"]) for cell in grid["expected_cells"]},
            {
                ("md", "test_md"),
                ("md", "test_siesta_fc_cartesian"),
                ("siesta_fc_cartesian", "test_md"),
                ("siesta_fc_cartesian", "test_siesta_fc_cartesian"),
            },
        )

        mixed_grid = module.build_cross_evaluation_expected_grid(
            ["md", "siesta_fc_cartesian"],
            ["test_md", "test_siesta_fc_cartesian", "test_mixed"],
        )
        self.assertEqual(mixed_grid["expected_cell_count"], 6)
        self.assertIn(
            {"train_method": "siesta_fc_cartesian", "test_set": "test_mixed", "cell_id": "siesta_fc_cartesian on test_mixed"},
            mixed_grid["expected_cells"],
        )

    def test_expected_cross_grid_for_three_methods_with_and_without_mixed(self) -> None:
        module = self.load_pipeline_ui_module()
        methods = ["md", "siesta_fc_cartesian", "random_cartesian"]
        base_tests = ["test_md", "test_siesta_fc_cartesian", "test_random_cartesian"]
        grid = module.build_cross_evaluation_expected_grid(methods, base_tests)
        self.assertEqual(grid["selected_methods"], methods)
        self.assertEqual(grid["selected_frozen_test_sets"], base_tests)
        self.assertEqual(grid["expected_cell_count"], 9)
        self.assertEqual(
            {(cell["train_method"], cell["test_set"]) for cell in grid["expected_cells"]},
            {
                (method, test_set)
                for method in methods
                for test_set in base_tests
            },
        )

        mixed_grid = module.build_cross_evaluation_expected_grid(methods, [*base_tests, "test_mixed"])
        self.assertEqual(mixed_grid["expected_cell_count"], 12)
        for method in methods:
            self.assertIn(
                {"train_method": method, "test_set": "test_mixed", "cell_id": f"{method} on test_mixed"},
                mixed_grid["expected_cells"],
            )

    def test_cross_evaluation_writes_expected_grid_before_missing_runs(self) -> None:
        module = self.load_pipeline_ui_module()
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            module.RESULTS_ROOT = root / "results"
            runner = module.ExperimentRunner()
            summary = runner._run_cross_evaluation(
                "exp_missing_grid",
                {
                    "selected_methods": ["md", "siesta_fc_cartesian", "random_cartesian"],
                    "test_sets": ["test_md", "test_siesta_fc_cartesian", "test_random_cartesian", "test_mixed"],
                    "runs": [
                        {
                            "pipeline": "md",
                            "method_id": "md",
                            "returncode": 0,
                            "dataset_size": 3,
                            "dataset_label": "md_3",
                        }
                    ],
                },
            )
            self.assertFalse(summary["ok"])
            grid_path = root / "results" / "exp_missing_grid" / "summary" / "cross_evaluation_expected_grid.json"
            self.assertEqual(summary["outputs"]["cross_evaluation_expected_grid"], str(grid_path))
            grid = json.loads(grid_path.read_text(encoding="utf-8"))
            self.assertEqual(grid["expected_cell_count"], 12)
            self.assertIn(
                {"train_method": "random_cartesian", "test_set": "test_random_cartesian", "cell_id": "random_cartesian on test_random_cartesian"},
                grid["expected_cells"],
            )

    def test_cross_evaluation_skips_winner_when_completeness_invalid(self) -> None:
        module = self.load_pipeline_ui_module()
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            module.RESULTS_ROOT = root / "results"
            runner = module.ExperimentRunner()
            calls = {"winner": 0}
            runner._split_manifest_for_result = lambda _result, split: root / f"{split}.csv"  # type: ignore[method-assign]

            def fake_run_local_script(command, label="", env=None):
                if "Agregando metricas cruzadas" in label:
                    output_dir = Path(command[command.index("--output-dir") + 1])
                    output_dir.mkdir(parents=True, exist_ok=True)
                    (output_dir / "cross_evaluation_metrics.csv").write_text(
                        "experiment_id,train_method,test_set,sample_id\n",
                        encoding="utf-8",
                    )
                    (output_dir / "cross_evaluation_completeness.json").write_text(
                        json.dumps(
                            {
                                "complete": False,
                                "scientific_status": "invalid_incomplete_grid",
                                "missing_cells": ["random_cartesian on test_md"],
                                "extra_unexpected_cells": [],
                                "missing_primary_metric_cells": [],
                            }
                        ),
                        encoding="utf-8",
                    )
                if "Analizando winners" in label:
                    calls["winner"] += 1
                return subprocess.CompletedProcess(command, 0, "", "")

            runner._run_local_script = fake_run_local_script  # type: ignore[method-assign]
            summary = runner._run_cross_evaluation(
                "exp_incomplete_guard",
                {
                    "selected_methods": ["md", "random_cartesian"],
                    "test_sets": ["test_md", "test_random_cartesian"],
                    "runs": [
                        {"pipeline": "md", "method_id": "md", "returncode": 0, "dataset_size": 3, "dataset_label": "md_3", "result_dir": str(root / "md")},
                        {
                            "pipeline": "random_cartesian",
                            "method_id": "random_cartesian",
                            "returncode": 0,
                            "dataset_size": 3,
                            "dataset_label": "rc_3",
                            "result_dir": str(root / "rc"),
                        },
                    ],
                },
            )
            self.assertFalse(summary["ok"])
            self.assertEqual(calls["winner"], 0)
            recommendation = json.loads(
                (root / "results" / "exp_incomplete_guard" / "summary" / "recommendation.json").read_text(encoding="utf-8")
            )
            self.assertEqual(recommendation["scientific_status"], "invalid_incomplete_grid")
            self.assertIsNone(recommendation["winner"])
            self.assertIn("random_cartesian on test_md", recommendation["missing_cells"])

    def test_cross_metric_aggregation_preserves_training_plan_metadata(self) -> None:
        sys.path.insert(0, str(REPO_ROOT / "Comparison" / "scripts"))
        aggregate = self.load_module_from_path(
            "aggregate_cross_metrics_training_plan_test",
            REPO_ROOT / "Comparison" / "scripts" / "aggregate_cross_metrics.py",
        )
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            result_dir = root / "cross" / "cross_abc__md__on__test_md"
            result_dir.mkdir(parents=True)
            (result_dir / "cross_evaluation_manifest.json").write_text(
                json.dumps(
                    {
                        "train_method": "md",
                        "test_set": "test_md",
                        "test_method": "md",
                        "dataset_size": 3,
                        "train_dataset_label": "md_3__epochs400",
                        "training_tag": "md_3_train2",
                        "training_index": 2,
                        "training_settings": {"max_epochs": 400},
                        "training_plan_label": "epochs400",
                        "training_plan_settings": {"max_epochs": 400},
                        "training_plan_source_dataset_label": "md_3",
                        "training_plan_settings_warning": "Training plan settings differ across methods for this cross-evaluation pair.",
                        "training_tag_by_method": {"md": "md_3_train2"},
                        "training_plan_label_by_method": {"md": "epochs400"},
                        "training_plan_settings_by_method": {"md": {"max_epochs": 400}},
                    }
                ),
                encoding="utf-8",
            )
            write_csv(
                result_dir / "metrics" / "spectral_metrics.csv",
                [{"sample": "sample_1", "low_energy_rmse_eV": "0.1"}],
            )

            rows = aggregate.aggregate_one(result_dir, "exp_training_plan")

            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["training_tag"], "md_3_train2")
            self.assertEqual(row["train_training_tag"], "md_3_train2")
            self.assertEqual(row["training_plan_label"], "epochs400")
            self.assertEqual(row["train_training_plan_label"], "epochs400")
            self.assertIn("Training plan settings differ", row["training_plan_settings_warning"])
            self.assertEqual(json.loads(row["training_plan_settings"])["max_epochs"], 400)
            self.assertEqual(json.loads(row["training_tag_by_method"])["md"], "md_3_train2")
            self.assertEqual(
                json.loads(row["training_plan_settings_by_method"])["md"]["max_epochs"],
                400,
            )

    def test_prepare_cross_result_dir_resets_existing_nonempty_output(self) -> None:
        module = self.load_pipeline_ui_module()
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            prediction_dir = root / "prediction"
            prediction_sample = prediction_dir / "predicted_hamiltonians" / "sample_1"
            prediction_sample.mkdir(parents=True)
            (prediction_sample / "ML_prediction.HSX").write_bytes(b"prediction")
            structure = root / "RUN.fdf"
            hamiltonian = root / "siesta.TSHS"
            metadata = root / "metadata.json"
            structure.write_text("structure", encoding="utf-8")
            hamiltonian.write_bytes(b"reference")
            metadata.write_text("{}", encoding="utf-8")
            manifest = root / "test_manifest.csv"
            write_csv(
                manifest,
                [
                    {
                        "sample_id": "sample_1",
                        "structure_path": str(structure),
                        "hamiltonian_path": str(hamiltonian),
                        "metadata_path": str(metadata),
                        "status": "valid",
                    }
                ],
            )
            cross_dir = root / "cross" / "cell"
            stale_prediction = cross_dir / "predicted_hamiltonians" / "old_sample"
            stale_prediction.mkdir(parents=True)
            (stale_prediction / "old.HSX").write_bytes(b"old")

            counts = module.ExperimentRunner()._prepare_cross_result_dir(cross_dir, prediction_dir, manifest)

            self.assertEqual(counts, {"references": 1, "structures": 1})
            self.assertTrue((cross_dir / "predicted_hamiltonians" / "sample_1" / "ML_prediction.HSX").exists())
            self.assertFalse(stale_prediction.exists())
            self.assertTrue((cross_dir / "siesta_hamiltonians" / "sample_1" / "siesta.TSHS").exists())
            self.assertTrue((cross_dir / "structures" / "sample_1" / "RUN.fdf").exists())

    def test_md_fc_random_render_shared_base_and_specific_layers(self) -> None:
        sys.path.insert(0, str(REPO_ROOT / "MD" / "scripts"))
        sys.path.insert(0, str(REPO_ROOT / "AtomDisplacement" / "scripts"))
        md_module = self.load_module_from_path(
            "md_pipeline_config_layer_test",
            REPO_ROOT / "MD" / "scripts" / "md_pipeline_config.py",
        )
        atom_module = self.load_module_from_path(
            "atom_pipeline_config_layer_test",
            REPO_ROOT / "AtomDisplacement" / "scripts" / "pipeline_config_utils.py",
        )
        md_config = md_module.load_pipeline_config(REPO_ROOT / "MD" / "pipeline_config.yaml")
        md_config["md"]["temperature_blocks"] = [
            {"block_id": "md_300", "temperature_K": 300, "n_snapshots": 3},
            {"block_id": "md_500", "temperature_K": 500, "n_snapshots": 2},
        ]
        self.assertEqual(md_module.md_total_steps(md_config), 5)
        md_fdf = md_module.render_run_fdf(md_config, block=md_config["md"]["temperature_blocks"][0])
        atom_config = atom_module.load_pipeline_config(REPO_ROOT / "AtomDisplacement" / "pipeline_config.yaml")
        fc_fdf = atom_module.render_single_point_fdf(
            atom_config,
            positions_ang=[atom["position"] for atom in atom_config["structure"]["atoms"]],
            atom_species=[atom["species_index"] for atom in atom_config["structure"]["atoms"]],
            sample_id="fc_sample",
        )
        random_config = copy.deepcopy(atom_config)
        random_config["structure"]["force_constants"]["enabled"] = False
        random_fdf = atom_module.render_single_point_fdf(
            random_config,
            positions_ang=[atom["position"] for atom in random_config["structure"]["atoms"]],
            atom_species=[atom["species_index"] for atom in random_config["structure"]["atoms"]],
            sample_id="rc_sample",
        )
        for text in (md_fdf, fc_fdf, random_fdf):
            self.assertIn("Generated from pipeline_config.yaml using shared RUN.fdf layers", text)
            self.assertIn("PAO.BasisSize", text)
            self.assertIn("MeshCutoff", text)
            self.assertIn("XML.Write", text)
        self.assertIn("MD.TypeOfRun", md_fdf)
        self.assertIn("Verlet", md_fdf)
        self.assertIn("MD.InitialTemperature", md_fdf)
        self.assertIn("300 K", md_fdf)
        self.assertIn("MD.TypeOfRun", fc_fdf)
        self.assertIn("FC", fc_fdf)
        self.assertIn("FC.Displacement", fc_fdf)
        self.assertNotIn("FC.Displacement", md_fdf)
        self.assertNotIn("MD.TypeOfRun", random_fdf)
        self.assertNotIn("FC.Displacement", random_fdf)

    def test_legacy_payload_converts_to_dataset_recipes_without_sync(self) -> None:
        module = self.load_pipeline_ui_module()
        recipes = module.legacy_payload_to_dataset_recipes(
            md_sizes=[5, 9],
            atom_dataset_specs=[
                {
                    "label": "dataset_fc",
                    "size": 7,
                    "displacements": [{"value": "0.02 Ang", "n_structures": 3}],
                }
            ],
            atom_sizes=[7],
            fc_dataset_specs=None,
            random_cartesian_options={"n_structures": [6, 8]},
            selected_methods=["md", "siesta_fc_cartesian", "random_cartesian"],
        )
        self.assertEqual([item["blocks"][0]["n_snapshots"] for item in recipes["md"]], [5, 9])
        self.assertEqual(recipes["siesta_fc_cartesian"][0]["blocks"][0]["n_structures"], 3)
        self.assertEqual([item["blocks"][0]["n_structures"] for item in recipes["random_cartesian"]], [6, 8])

    def test_cleanup_generated_datasets_preserves_code_and_configs(self) -> None:
        cleanup = self.load_module_from_path(
            "cleanup_generated_datasets_test",
            REPO_ROOT / "Comparison" / "scripts" / "cleanup_generated_datasets.py",
        )
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            (root / "MD" / "scripts").mkdir(parents=True)
            (root / "MD" / "scripts" / "keep.py").write_text("print('keep')\n", encoding="utf-8")
            (root / "MD" / "pipeline_config.yaml").write_text("config: keep\n", encoding="utf-8")
            (root / "MD" / "dataset").mkdir(parents=True)
            (root / "MD" / "dataset" / "H.psf").write_text("pseudo\n", encoding="utf-8")
            (root / "MD" / "dataset" / "MD_steps").mkdir()
            (root / "AtomDisplacement" / "dataset" / "FC_steps").mkdir(parents=True)
            (root / "AtomDisplacement" / "pipeline_config.yaml").parent.mkdir(parents=True, exist_ok=True)
            (root / "AtomDisplacement" / "pipeline_config.yaml").write_text("config: keep\n", encoding="utf-8")
            (root / "Comparison" / "workspaces" / "run" / "md" / "dataset_3").mkdir(parents=True)
            (root / "Comparison" / "results" / "results_md" / "dataset_3").mkdir(parents=True)
            (root / "Comparison" / "results" / "results_md" / "MD_dataset1_ABC123").mkdir(parents=True)
            (root / "Comparison" / "results" / "results_atomdisp" / "FC_dataset1_DEF456").mkdir(parents=True)
            (root / "Comparison" / "results" / "results_random_cartesian" / "RC_dataset1_GHI789").mkdir(parents=True)
            (root / "Comparison" / "results" / "20260101_000001" / "summary").mkdir(parents=True)
            manifest = cleanup.cleanup_generated_datasets(root)
            self.assertIn("MD/dataset/MD_steps", manifest["removed"])
            self.assertFalse((root / "MD" / "dataset" / "MD_steps").exists())
            self.assertFalse((root / "AtomDisplacement" / "dataset" / "FC_steps").exists())
            self.assertFalse((root / "Comparison" / "workspaces").exists())
            self.assertFalse((root / "Comparison" / "results" / "results_md" / "dataset_3").exists())
            self.assertFalse((root / "Comparison" / "results" / "results_md" / "MD_dataset1_ABC123").exists())
            self.assertFalse((root / "Comparison" / "results" / "results_atomdisp" / "FC_dataset1_DEF456").exists())
            self.assertFalse((root / "Comparison" / "results" / "results_random_cartesian" / "RC_dataset1_GHI789").exists())
            self.assertFalse((root / "Comparison" / "results" / "20260101_000001").exists())
            self.assertTrue((root / "MD" / "scripts" / "keep.py").exists())
            self.assertTrue((root / "MD" / "pipeline_config.yaml").exists())
            self.assertTrue((root / "AtomDisplacement" / "pipeline_config.yaml").exists())
            self.assertTrue((root / "MD" / "dataset" / "H.psf").exists())
            self.assertTrue((root / "Comparison" / "generated_dataset_cleanup_manifest.json").exists())
            cleanup_log = (root / "Comparison" / "generated_dataset_cleanup_manifest.json").read_text(
                encoding="utf-8"
            )
            self.assertNotIn(str(root), cleanup_log)

    def test_cleanup_selected_generated_datasets_uses_ids_and_preserves_unselected(self) -> None:
        cleanup = self.load_module_from_path(
            "cleanup_generated_datasets_selected_test",
            REPO_ROOT / "Comparison" / "scripts" / "cleanup_generated_datasets.py",
        )
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            keep = root / "Comparison" / "results" / "results_md" / "keep_dataset"
            delete = root / "Comparison" / "results" / "results_md" / "delete_dataset"
            keep.mkdir(parents=True)
            (keep / "marker.txt").write_text("keep\n", encoding="utf-8")
            manifest_dir = delete / "run_1"
            manifest_dir.mkdir(parents=True)
            (manifest_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "method_id": "md",
                        "dataset_label": "delete_dataset_label",
                        "dataset_size": 12,
                        "run_id": "run_1",
                    }
                ),
                encoding="utf-8",
            )

            records = cleanup.generated_dataset_records(root)
            delete_record = next(record for record in records if record["name"] == "delete_dataset")
            self.assertEqual(delete_record["method"], "md")
            self.assertEqual(delete_record["dataset_label"], "delete_dataset_label")
            self.assertEqual(delete_record["dataset_size"], 12)

            manifest = cleanup.cleanup_selected_generated_datasets(root, target_ids=[delete_record["id"]])
            self.assertIn("Comparison/results/results_md/delete_dataset", manifest["removed"])
            self.assertFalse(delete.exists())
            self.assertTrue(keep.exists())
            self.assertTrue((root / "Comparison" / "generated_dataset_cleanup_manifest.json").exists())
            cleanup_log = json.loads(
                (root / "Comparison" / "generated_dataset_cleanup_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                cleanup_log["selected"][0]["path"],
                "Comparison/results/results_md/delete_dataset",
            )
            self.assertNotIn(str(root), json.dumps(cleanup_log))

            with self.assertRaisesRegex(RuntimeError, "no reconocidos"):
                cleanup.cleanup_selected_generated_datasets(root, target_ids=["not-a-real-id"])

    def test_cleanup_generated_manifest_is_ignored_and_not_required(self) -> None:
        cleanup = self.load_module_from_path(
            "cleanup_generated_datasets_no_manifest_test",
            REPO_ROOT / "Comparison" / "scripts" / "cleanup_generated_datasets.py",
        )
        gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("Comparison/generated_dataset_cleanup_manifest.json", gitignore)
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            dataset = root / "Comparison" / "results" / "results_md" / "dataset_a"
            dataset.mkdir(parents=True)
            manifest_path = root / "Comparison" / "generated_dataset_cleanup_manifest.json"
            self.assertFalse(manifest_path.exists())
            records = cleanup.generated_dataset_records(root)
            self.assertEqual([record["relative_path"] for record in records], ["Comparison/results/results_md/dataset_a"])
            self.assertEqual([record["path"] for record in records], ["Comparison/results/results_md/dataset_a"])

    def test_cleanup_refuses_outside_repo_and_repo_root_targets(self) -> None:
        cleanup = self.load_module_from_path(
            "cleanup_generated_datasets_boundary_test",
            REPO_ROOT / "Comparison" / "scripts" / "cleanup_generated_datasets.py",
        )
        with workspace_tempdir() as tmp:
            root = Path(tmp) / "repo"
            root.mkdir()
            outside = Path(tmp) / "outside"
            outside.mkdir()
            with self.assertRaisesRegex(RuntimeError, "fuera del repositorio"):
                cleanup._safe_remove_target(root, outside)
            with self.assertRaisesRegex(RuntimeError, "raiz del repositorio"):
                cleanup._safe_remove_target(root, root)
            self.assertTrue(outside.exists())
            self.assertTrue(root.exists())

    def test_cleanup_dry_run_preserves_targets_and_skips_manifest_write(self) -> None:
        cleanup = self.load_module_from_path(
            "cleanup_generated_datasets_dry_run_test",
            REPO_ROOT / "Comparison" / "scripts" / "cleanup_generated_datasets.py",
        )
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            target = root / "Comparison" / "results" / "results_md" / "dry_run_dataset"
            target.mkdir(parents=True)
            manifest = cleanup.cleanup_generated_datasets(root, dry_run=True)
            self.assertIn("Comparison/results/results_md/dry_run_dataset", manifest["removed"])
            self.assertTrue(target.exists())
            self.assertFalse((root / "Comparison" / "generated_dataset_cleanup_manifest.json").exists())

    def test_cleanup_selected_generated_datasets_rejects_symlink_targets(self) -> None:
        cleanup = self.load_module_from_path(
            "cleanup_generated_datasets_symlink_test",
            REPO_ROOT / "Comparison" / "scripts" / "cleanup_generated_datasets.py",
        )
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            outside = root / "outside_dataset_target"
            outside.mkdir()
            link = root / "MD" / "dataset" / "bad_link"
            link.parent.mkdir(parents=True)
            try:
                link.symlink_to(outside, target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"symlink unavailable in this environment: {exc}")

            records = cleanup.generated_dataset_records(root)
            link_record = next(record for record in records if record["name"] == "bad_link")
            with self.assertRaisesRegex(RuntimeError, "enlace simbolico|fuera"):
                cleanup.cleanup_selected_generated_datasets(root, target_ids=[link_record["id"]])
            self.assertTrue(outside.exists())

    def test_dataset_only_records_status_and_skips_cross_evaluation(self) -> None:
        module = self.load_pipeline_ui_module()
        module.STRICT_COMPARISON_MODE = False
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            module.RESULTS_ROOT = root / "results"
            module.WORKSPACES_ROOT = root / "workspaces"
            runner = module.ExperimentRunner()
            calls = {"run_one": [], "cross": 0, "labels": []}
            md_yaml_before = (REPO_ROOT / "MD" / "pipeline_config.yaml").read_text(encoding="utf-8")
            atom_yaml_before = (REPO_ROOT / "AtomDisplacement" / "pipeline_config.yaml").read_text(encoding="utf-8")

            def fake_run_one(key, size, run_id, **kwargs):
                calls["run_one"].append((key, kwargs.get("run_mode"), kwargs.get("compute_accelerator")))
                calls["labels"].append(kwargs.get("dataset_label"))
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
            self.assertRegex(calls["labels"][0], r"^MD_dataset1_[A-Za-z0-9]{6}$")
            self.assertRegex(calls["labels"][1], r"^FC_dataset1_[A-Za-z0-9]{6}$")
            self.assertNotEqual(calls["labels"][0].rsplit("_", 1)[-1], calls["labels"][1].rsplit("_", 1)[-1])
            self.assertTrue(manifest["cross_evaluation"]["skipped"])
            self.assertEqual((REPO_ROOT / "MD" / "pipeline_config.yaml").read_text(encoding="utf-8"), md_yaml_before)
            self.assertEqual(
                (REPO_ROOT / "AtomDisplacement" / "pipeline_config.yaml").read_text(encoding="utf-8"),
                atom_yaml_before,
            )
            self.assertTrue((root / "results" / "dataset_only_case" / "performance_report.json").exists())

    def test_downstream_only_records_runs_and_cross_evaluation(self) -> None:
        module = self.load_pipeline_ui_module()
        module.STRICT_COMPARISON_MODE = False
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            module.RESULTS_ROOT = root / "results"
            module.WORKSPACES_ROOT = root / "workspaces"
            runner = module.ExperimentRunner()
            calls = {"run_one": [], "cross": 0}

            def fake_run_one(key, size, run_id, **kwargs):
                calls["run_one"].append((key, kwargs.get("run_mode")))
                return {
                    "pipeline": key,
                    "method_id": "siesta_fc_cartesian" if key == "atom_displacement" else key,
                    "dataset_label": kwargs.get("dataset_label") or f"dataset_{size}",
                    "dataset_size": size,
                    "returncode": 0,
                    "result_dir": str(root / "fake" / key / f"dataset_{size}"),
                    "dataset_dir": str(root / "existing" / key / f"dataset_{size}"),
                    "dataset_sample_ids": [f"{key}-{size}"],
                    "dataset_sample_hash": f"hash-{key}-{size}",
                    "run_mode": kwargs.get("run_mode"),
                    "pipeline_elapsed_seconds": 1.25,
                    "siesta_counts": {"launched": 0, "skipped_or_reused": size, "failed": 0, "total": size},
                }

            def fake_cross(*_args, **_kwargs):
                calls["cross"] += 1
                return {"ok": True, "cross_evaluations": [], "outputs": {}}

            runner._run_one = fake_run_one  # type: ignore[method-assign]
            runner._run_cross_evaluation = fake_cross  # type: ignore[method-assign]
            runner._started_at = 1.0
            runner._run(
                [3],
                [3],
                "downstream_only_case",
                split_ratios={"train": 1 / 3, "validation": 1 / 3, "test": 1 / 3},
                selected_methods=["md", "siesta_fc_cartesian"],
                run_mode="train_test_metrics_plots_only",
            )
            manifest = module.load_config(root / "results" / "downstream_only_case" / "experiment_manifest.yaml")
            self.assertEqual(manifest["run_mode"], "train_test_metrics_plots_only")
            self.assertEqual(calls["cross"], 1)
            self.assertEqual(
                calls["run_one"],
                [("md", "train_test_metrics_plots_only"), ("atom_displacement", "train_test_metrics_plots_only")],
            )
            self.assertEqual(manifest["timing"]["counters"]["siesta_launched"], 0)
            self.assertEqual(manifest["timing"]["counters"]["graph2mat_trainings"], 2)

    def test_training_plan_schedules_configs_over_selected_reusable_datasets(self) -> None:
        module = self.load_pipeline_ui_module()
        module.STRICT_COMPARISON_MODE = False
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            module.RESULTS_ROOT = root / "results"
            module.WORKSPACES_ROOT = root / "workspaces"
            for label, size in (("md_100", 3), ("md_200", 4), ("md_300", 5)):
                dataset_dir = make_reusable_md_dataset(root / f"source_{label}", n_samples=size)
                write_archived_retrainable_run(
                    module.RESULTS_ROOT,
                    method_id="md",
                    dataset_label=label,
                    dataset_size=size,
                    dataset_dir=dataset_dir,
                    run_id=f"{label}_source",
                )
            runner = module.ExperimentRunner()
            candidates = runner.reusable_dataset_candidates_payload()["datasets"]
            ids_by_label = {item["dataset_label"]: item["id"] for item in candidates}
            calls: list[tuple[str, int, dict[str, object], str]] = []

            def fake_run_one(key, size, run_id, **kwargs):
                metadata = kwargs.get("recipe_metadata") or {}
                calls.append(
                    (
                        kwargs.get("dataset_label"),
                        size,
                        kwargs.get("training_settings"),
                        metadata.get("training_plan_label"),
                    )
                )
                return {
                    "pipeline": key,
                    "method_id": key,
                    "dataset_label": kwargs.get("dataset_label"),
                    "dataset_size": size,
                    "returncode": 0,
                    "result_dir": str(root / "fake" / str(kwargs.get("dataset_label"))),
                    "dataset_dir": str(root / "existing" / str(kwargs.get("dataset_label"))),
                    "dataset_sample_ids": [f"{key}-{size}"],
                    "dataset_sample_hash": f"hash-{key}-{size}",
                    "run_mode": kwargs.get("run_mode"),
                    "pipeline_elapsed_seconds": 1.0,
                    "siesta_counts": {"launched": 0, "skipped_or_reused": size, "failed": 0, "total": size},
                }

            runner._run_one = fake_run_one  # type: ignore[method-assign]
            runner._started_at = 1.0
            runner._run(
                [],
                [],
                "training_plan_case",
                split_ratios={"train": 1 / 3, "validation": 1 / 3, "test": 1 / 3},
                selected_methods=["md"],
                run_mode="train_test_metrics_plots_only",
                performance={"max_parallel_dataset_jobs": 4},
                training_plan=[
                    {
                        "label": "epochs400",
                        "reusable_dataset_ids": [
                            ids_by_label["md_100"],
                            ids_by_label["md_200"],
                            ids_by_label["md_300"],
                        ],
                        "training_settings": {"max_epochs": 400},
                    },
                    {
                        "label": "epochs600",
                        "reusable_dataset_ids": [
                            ids_by_label["md_100"],
                            ids_by_label["md_200"],
                        ],
                        "training_settings": {"max_epochs": 600},
                    },
                ],
            )

            self.assertEqual(len(calls), 5)
            self.assertEqual([call[3] for call in calls], ["epochs400"] * 3 + ["epochs600"] * 2)
            self.assertEqual([call[2]["max_epochs"] for call in calls], [400, 400, 400, 600, 600])
            self.assertTrue(all("epochs" in str(call[0]) for call in calls))
            manifest = module.load_config(root / "results" / "training_plan_case" / "experiment_manifest.yaml")
            self.assertEqual(manifest["timing"]["counters"]["dataset_jobs"], 5)
            self.assertEqual(manifest["timing"]["counters"]["dataset_workers_used"], 1)
            self.assertEqual(manifest["timing"]["counters"]["graph2mat_trainings"], 5)
            self.assertEqual([item["label"] for item in manifest["training_plan"]], ["epochs400", "epochs600"])

    def test_full_strict_without_training_plan_keeps_single_phase_behavior(self) -> None:
        module = self.load_pipeline_ui_module()
        module.STRICT_COMPARISON_MODE = False
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            module.RESULTS_ROOT = root / "results"
            module.WORKSPACES_ROOT = root / "workspaces"
            recipes = {
                "md": [
                    {
                        "recipe_id": "md_100",
                        "label": "MD 100",
                        "blocks": [{"block_id": "md_100", "n_snapshots": 3}],
                    }
                ]
            }
            recipe_info = module.dataset_recipes_to_execution_specs(
                recipes,
                selected_methods=["md"],
                split_ratios={"train": 1 / 3, "validation": 1 / 3, "test": 1 / 3},
                random_cartesian_defaults={},
            )
            calls: list[dict[str, object]] = []

            def fake_run_one(key, size, run_id, **kwargs):
                calls.append(
                    {
                        "key": key,
                        "size": size,
                        "run_mode": kwargs.get("run_mode"),
                        "source_run": kwargs.get("source_run"),
                        "training_settings": kwargs.get("training_settings"),
                    }
                )
                label = str(kwargs.get("dataset_label"))
                return {
                    "pipeline": key,
                    "method_id": key,
                    "dataset_label": label,
                    "dataset_size": size,
                    "returncode": 0,
                    "result_dir": str(root / "fake_results" / label),
                    "dataset_dir": str(root / "fake_datasets" / label),
                    "dataset_sample_ids": [f"{key}-{size}"],
                    "dataset_sample_hash": f"hash-{key}-{size}",
                    "run_mode": kwargs.get("run_mode"),
                    "pipeline_elapsed_seconds": 1.0,
                    "predicted_hamiltonians": 1,
                    "siesta_counts": {"launched": size, "skipped_or_reused": 0, "failed": 0, "total": size},
                    "timing_breakdown": {"training_prediction_seconds": 1.0},
                }

            runner = module.ExperimentRunner()
            runner._run_one = fake_run_one  # type: ignore[method-assign]
            runner._started_at = 1.0
            runner._run(
                [3],
                [],
                "full_no_plan",
                split_ratios={"train": 1 / 3, "validation": 1 / 3, "test": 1 / 3},
                selected_methods=["md"],
                run_mode="full_strict_pipeline",
                dataset_recipes_info=recipe_info,
            )

            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["run_mode"], "full_strict_pipeline")
            self.assertIsNone(calls[0]["source_run"])
            manifest = module.load_config(root / "results" / "full_no_plan" / "experiment_manifest.yaml")
            self.assertEqual(len(manifest["runs"]), 1)
            self.assertNotIn("generated_dataset_sources", manifest)

    def test_full_strict_training_plan_generates_once_then_trains_granular_targets(self) -> None:
        module = self.load_pipeline_ui_module()
        module.STRICT_COMPARISON_MODE = False
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            module.RESULTS_ROOT = root / "results"
            module.WORKSPACES_ROOT = root / "workspaces"
            recipes = {
                "md": [
                    {
                        "recipe_id": "md_100",
                        "label": "MD 100",
                        "blocks": [{"block_id": "md_100", "n_snapshots": 3}],
                    },
                    {
                        "recipe_id": "md_200",
                        "label": "MD 200",
                        "blocks": [{"block_id": "md_200", "n_snapshots": 4}],
                    },
                    {
                        "recipe_id": "md_300",
                        "label": "MD 300",
                        "blocks": [{"block_id": "md_300", "n_snapshots": 5}],
                    },
                ]
            }
            split_ratios = {"train": 1 / 3, "validation": 1 / 3, "test": 1 / 3}
            recipe_info = module.dataset_recipes_to_execution_specs(
                recipes,
                selected_methods=["md"],
                split_ratios=split_ratios,
                random_cartesian_defaults={},
            )
            calls: list[dict[str, object]] = []

            def fake_run_one(key, size, run_id, **kwargs):
                metadata = kwargs.get("recipe_metadata") or {}
                run_mode = kwargs.get("run_mode")
                source_run = kwargs.get("source_run")
                label = str(kwargs.get("dataset_label"))
                calls.append(
                    {
                        "label": label,
                        "size": size,
                        "run_mode": run_mode,
                        "source_run": source_run,
                        "training_settings": kwargs.get("training_settings"),
                        "training_plan_label": metadata.get("training_plan_label"),
                    }
                )
                is_source = run_mode == module.DATASET_ONLY_RUN_MODE
                return {
                    "pipeline": key,
                    "method_id": key,
                    "dataset_label": label,
                    "dataset_size": size,
                    "returncode": 0,
                    "result_dir": str(root / "fake_results" / label / str(run_mode)),
                    "dataset_dir": str(root / "fake_datasets" / label / str(run_mode)),
                    "dataset_sample_ids": [f"{key}-{label}-{size}"],
                    "dataset_sample_hash": f"hash-{key}-{label}-{size}",
                    "run_mode": run_mode,
                    "pipeline_elapsed_seconds": 1.0,
                    "predicted_hamiltonians": 0 if is_source else 1,
                    "siesta_counts": {
                        "launched": size if is_source else 0,
                        "skipped_or_reused": 0 if is_source else size,
                        "failed": 0,
                        "total": size,
                    },
                    "timing_breakdown": {
                        "md_siesta_generation_seconds": 1.0 if is_source else None,
                        "training_prediction_seconds": None if is_source else 1.0,
                    },
                    "training_plan_index": metadata.get("training_plan_index"),
                    "training_plan_label": metadata.get("training_plan_label"),
                    "training_plan_settings": metadata.get("training_plan_settings"),
                    "training_plan_source_dataset_label": metadata.get("training_plan_source_dataset_label"),
                    "training_tag": None if is_source else f"{label}_train1",
                }

            runner = module.ExperimentRunner()
            runner._run_one = fake_run_one  # type: ignore[method-assign]
            runner._started_at = 1.0
            runner._run(
                [],
                [],
                "full_plan_granular",
                split_ratios=split_ratios,
                selected_methods=["md"],
                run_mode="full_strict_pipeline",
                dataset_recipes_info=recipe_info,
                performance={"max_parallel_dataset_jobs": 4},
                training_plan=[
                    {
                        "label": "epochs400",
                        "dataset_targets": [
                            module.planned_dataset_target_id("md", "md_100"),
                            module.planned_dataset_target_id("md", "md_200"),
                            module.planned_dataset_target_id("md", "md_300"),
                        ],
                        "training_settings": {"max_epochs": 400},
                    },
                    {
                        "label": "epochs600",
                        "dataset_targets": [
                            module.planned_dataset_target_id("md", "md_100"),
                            module.planned_dataset_target_id("md", "md_300"),
                        ],
                        "training_settings": {"max_epochs": 600},
                    },
                ],
            )

            source_calls = [call for call in calls if call["run_mode"] == module.DATASET_ONLY_RUN_MODE]
            train_calls = [call for call in calls if call["run_mode"] == module.FULL_STRICT_RUN_MODE]
            self.assertEqual(len(source_calls), 3)
            self.assertEqual(len(train_calls), 5)
            self.assertTrue(all(call["source_run"] is None for call in source_calls))
            self.assertTrue(all(call["source_run"] is not None for call in train_calls))
            self.assertEqual(
                [call["training_plan_label"] for call in train_calls],
                ["epochs400", "epochs400", "epochs400", "epochs600", "epochs600"],
            )
            self.assertEqual(
                [call["training_settings"]["max_epochs"] for call in train_calls],
                [400, 400, 400, 600, 600],
            )

            manifest = module.load_config(root / "results" / "full_plan_granular" / "experiment_manifest.yaml")
            self.assertEqual(len(manifest["generated_dataset_sources"]), 3)
            self.assertEqual(len(manifest["runs"]), 5)
            self.assertEqual(manifest["timing"]["counters"]["dataset_jobs"], 8)
            self.assertEqual(manifest["timing"]["counters"]["dataset_workers_used"], 1)
            self.assertEqual(manifest["timing"]["counters"]["graph2mat_trainings"], 5)
            manifest_plan_labels = [run["training_plan_label"] for run in manifest["runs"]]
            self.assertEqual(manifest_plan_labels.count("epochs400"), 3)
            self.assertEqual(manifest_plan_labels.count("epochs600"), 2)
            self.assertTrue(all(run["training_tag"] for run in manifest["runs"]))

    def test_full_strict_training_plan_missing_target_fails_clearly(self) -> None:
        module = self.load_pipeline_ui_module()
        module.STRICT_COMPARISON_MODE = False
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            module.RESULTS_ROOT = root / "results"
            module.WORKSPACES_ROOT = root / "workspaces"
            recipes = {
                "md": [
                    {
                        "recipe_id": "md_100",
                        "label": "MD 100",
                        "blocks": [{"block_id": "md_100", "n_snapshots": 3}],
                    }
                ]
            }
            recipe_info = module.dataset_recipes_to_execution_specs(
                recipes,
                selected_methods=["md"],
                split_ratios={"train": 1 / 3, "validation": 1 / 3, "test": 1 / 3},
                random_cartesian_defaults={},
            )
            calls: list[object] = []

            def fake_run_one(*args, **kwargs):
                calls.append((args, kwargs))
                raise AssertionError("generation should not start for missing target")

            runner = module.ExperimentRunner()
            runner._run_one = fake_run_one  # type: ignore[method-assign]
            runner._started_at = 1.0
            runner._run(
                [],
                [],
                "full_plan_missing_target",
                split_ratios={"train": 1 / 3, "validation": 1 / 3, "test": 1 / 3},
                selected_methods=["md"],
                run_mode="full_strict_pipeline",
                dataset_recipes_info=recipe_info,
                training_plan=[
                    {
                        "label": "bad",
                        "dataset_targets": ["md:missing"],
                        "training_settings": {"max_epochs": 400},
                    }
                ],
            )

            self.assertEqual(calls, [])
            manifest = module.load_config(root / "results" / "full_plan_missing_target" / "experiment_manifest.yaml")
            self.assertTrue(any("dataset_targets no encontrados" in warning for warning in manifest["warnings"]))

    def test_full_strict_training_plan_missing_generated_source_fails_clearly(self) -> None:
        module = self.load_pipeline_ui_module()
        module.STRICT_COMPARISON_MODE = False
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            module.RESULTS_ROOT = root / "results"
            module.WORKSPACES_ROOT = root / "workspaces"
            recipes = {
                "md": [
                    {
                        "recipe_id": "md_100",
                        "label": "MD 100",
                        "blocks": [{"block_id": "md_100", "n_snapshots": 3}],
                    },
                    {
                        "recipe_id": "md_200",
                        "label": "MD 200",
                        "blocks": [{"block_id": "md_200", "n_snapshots": 4}],
                    },
                ]
            }
            recipe_info = module.dataset_recipes_to_execution_specs(
                recipes,
                selected_methods=["md"],
                split_ratios={"train": 0.5, "validation": 0.25, "test": 0.25},
                random_cartesian_defaults={},
            )
            calls: list[dict[str, object]] = []

            def fake_run_one(key, size, run_id, **kwargs):
                run_mode = kwargs.get("run_mode")
                calls.append({"size": size, "run_mode": run_mode})
                if run_mode == module.DATASET_ONLY_RUN_MODE and size == 4:
                    raise RuntimeError("source generation failed")
                if run_mode == module.FULL_STRICT_RUN_MODE:
                    raise AssertionError("training must not start when a selected source was not generated")
                label = str(kwargs.get("dataset_label"))
                return {
                    "pipeline": key,
                    "method_id": key,
                    "dataset_label": label,
                    "dataset_size": size,
                    "returncode": 0,
                    "result_dir": str(root / "fake_results" / label),
                    "dataset_dir": str(root / "fake_datasets" / label),
                    "dataset_sample_ids": [f"{key}-{label}-{size}"],
                    "dataset_sample_hash": f"hash-{key}-{label}-{size}",
                    "run_mode": run_mode,
                    "pipeline_elapsed_seconds": 1.0,
                    "predicted_hamiltonians": 0,
                    "siesta_counts": {"launched": size, "skipped_or_reused": 0, "failed": 0, "total": size},
                    "timing_breakdown": {"md_siesta_generation_seconds": 1.0},
                }

            runner = module.ExperimentRunner()
            runner._run_one = fake_run_one  # type: ignore[method-assign]
            runner._started_at = 1.0
            runner._run(
                [],
                [],
                "full_plan_missing_source",
                split_ratios={"train": 0.5, "validation": 0.25, "test": 0.25},
                selected_methods=["md"],
                run_mode="full_strict_pipeline",
                dataset_recipes_info=recipe_info,
                performance={"error_policy": "continue_on_error"},
                training_plan=[
                    {
                        "label": "epochs400",
                        "dataset_targets": [
                            module.planned_dataset_target_id("md", "md_100"),
                            module.planned_dataset_target_id("md", "md_200"),
                        ],
                        "training_settings": {"max_epochs": 400},
                    }
                ],
            )

            self.assertEqual([call["run_mode"] for call in calls], [module.DATASET_ONLY_RUN_MODE, module.DATASET_ONLY_RUN_MODE])
            manifest = module.load_config(root / "results" / "full_plan_missing_source" / "experiment_manifest.yaml")
            self.assertTrue(
                any("no se generaron los datasets fuente" in warning for warning in manifest["warnings"])
            )

    def test_reusable_dataset_candidates_match_plot_visible_archived_results(self) -> None:
        module = self.load_pipeline_ui_module()
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            module.RESULTS_ROOT = root / "results"
            visible_dataset = make_reusable_md_dataset(root / "visible_source")
            hidden_dataset = make_reusable_md_dataset(root / "hidden_source")
            write_archived_retrainable_run(
                module.RESULTS_ROOT,
                method_id="md",
                dataset_label="visible_md",
                dataset_size=3,
                dataset_dir=visible_dataset,
                run_id="visible_run",
            )
            (module.RESULTS_ROOT / "legacy_experiment").mkdir(parents=True)
            module.write_yaml(
                module.RESULTS_ROOT / "legacy_experiment" / "experiment_manifest.yaml",
                {
                    "runs": [
                        {
                            "pipeline": "md",
                            "method_id": "md",
                            "dataset_label": "experiment_manifest_only",
                            "dataset_size": 3,
                            "returncode": 0,
                            "run_id": "legacy_experiment",
                            "dataset_dir": str(hidden_dataset),
                            "result_dir": str(root / "legacy_result"),
                        }
                    ]
                },
            )
            no_metric_result = module.RESULTS_ROOT / "results_md" / "no_metrics" / "run_no_metrics"
            no_metric_result.mkdir(parents=True)
            (no_metric_result / "manifest.json").write_text(
                json.dumps(
                    {
                        "pipeline": "md",
                        "method_id": "md",
                        "dataset_label": "no_metrics",
                        "dataset_size": 3,
                        "returncode": 0,
                        "run_id": "no_metrics",
                        "dataset_dir": str(hidden_dataset),
                        "result_dir": str(no_metric_result),
                    }
                ),
                encoding="utf-8",
            )

            runner = module.ExperimentRunner()
            payload = runner.reusable_dataset_candidates_payload()
            labels = [item["dataset_label"] for item in payload["datasets"]]
            self.assertEqual(labels, ["visible_md"])
            self.assertEqual([run["dataset_label"] for run in module.plot_data_summary()["runs"]], ["visible_md"])

    def test_parallel_callable_tasks_honor_continue_and_fail_fast(self) -> None:
        module = self.load_pipeline_ui_module()
        runner = module.ExperimentRunner()
        tasks = [
            ("ok-1", lambda: "a"),
            ("bad", lambda: (_ for _ in ()).throw(RuntimeError("boom"))),
            ("ok-2", lambda: "b"),
        ]
        results, failures = runner._run_callable_tasks(  # type: ignore[attr-defined]
            tasks,
            workers=2,
            error_policy="continue_on_error",
            stage="unit_stage",
        )
        self.assertEqual(results, ["a", "b"])
        self.assertEqual(len(failures), 1)
        self.assertIn("boom", failures[0])
        with self.assertRaisesRegex(RuntimeError, "boom"):
            runner._run_callable_tasks(  # type: ignore[attr-defined]
                tasks,
                workers=2,
                error_policy="fail_fast",
                stage="unit_stage",
            )

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

    def test_downstream_only_md_reuses_existing_dataset_without_generation(self) -> None:
        module = self.load_pipeline_ui_module()
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            module.RESULTS_ROOT = root / "results"
            module.WORKSPACES_ROOT = root / "workspaces"
            source_dataset = make_reusable_md_dataset(root / "source_md")
            write_archived_retrainable_run(
                module.RESULTS_ROOT,
                method_id="md",
                dataset_label="existing_md",
                dataset_size=3,
                dataset_dir=source_dataset,
            )

            runner = module.ExperimentRunner()
            calls: list[tuple[str, object]] = []

            def forbidden_generation(*_args, **_kwargs):
                raise AssertionError("dataset generation should not be called")

            def fake_validate(key, config, size):
                calls.append(("validate", (key, size, config["paths"]["dataset_dir"])))
                return {}

            def fake_process(spec, **kwargs):
                config = module.load_config(kwargs["config_path"])
                calls.append(("process", tuple(config["pipeline"]["steps"])))
                self.assertEqual(tuple(config["pipeline"]["steps"]), module.MD_DOWNSTREAM_STEPS)
                self.assertIn("workspaces/reuse_run/md/existing_md/dataset", config["paths"]["dataset_dir"])
                self.assertNotEqual(config["paths"]["dataset_dir"], str(source_dataset))
                return 0

            def fake_archive(key, size, run_id, workspace, config, returncode, run_log, prepare_metadata, **kwargs):
                calls.append(("archive", prepare_metadata.get("dataset_reused")))
                return {
                    "pipeline": key,
                    "method_id": key,
                    "dataset_label": kwargs.get("dataset_label"),
                    "dataset_size": size,
                    "returncode": returncode,
                    "run_mode": kwargs.get("run_mode"),
                    "dataset_dir": config["paths"]["dataset_dir"],
                    "result_dir": str(root / "archived"),
                    "dataset_reused": prepare_metadata.get("dataset_reused"),
                }

            runner._prepare_md_config = forbidden_generation  # type: ignore[method-assign]
            runner._validate_split_manifests = fake_validate  # type: ignore[method-assign]
            runner._run_pipeline_process = fake_process  # type: ignore[method-assign]
            runner._archive_outputs = fake_archive  # type: ignore[method-assign]

            result = runner._run_one(
                "md",
                3,
                "reuse_run",
                dataset_label="existing_md",
                run_mode="train_test_metrics_plots_only",
            )

            self.assertTrue(result["dataset_reused"])
            self.assertEqual(calls[0][0], "validate")
            self.assertEqual(calls[1], ("process", module.MD_DOWNSTREAM_STEPS))
            self.assertEqual(calls[2], ("archive", True))
            copied_manifest = (
                module.WORKSPACES_ROOT
                / "reuse_run"
                / "md"
                / "existing_md"
                / "dataset"
                / "splits"
                / "train_manifest.csv"
            )
            self.assertIn(
                str(module.WORKSPACES_ROOT / "reuse_run" / "md" / "existing_md" / "dataset"),
                copied_manifest.read_text(encoding="utf-8"),
            )
            self.assertNotIn(str(source_dataset), copied_manifest.read_text(encoding="utf-8"))

    def test_downstream_only_md_can_rebuild_splits_without_generation(self) -> None:
        module = self.load_pipeline_ui_module()
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            module.RESULTS_ROOT = root / "results"
            module.WORKSPACES_ROOT = root / "workspaces"
            source_dataset = make_reusable_md_dataset(root / "source_md", n_samples=6)
            write_archived_retrainable_run(
                module.RESULTS_ROOT,
                method_id="md",
                dataset_label="existing_md",
                dataset_size=6,
                dataset_dir=source_dataset,
            )

            runner = module.ExperimentRunner()

            def forbidden_generation(*_args, **_kwargs):
                raise AssertionError("dataset generation should not be called")

            def fake_validate(key, config, size):
                dataset_dir = Path(config["paths"]["dataset_dir"])
                counts = {
                    split: len(module.read_csv_rows(dataset_dir / "splits" / f"{split}_manifest.csv"))
                    for split in ("train", "validation", "test")
                }
                self.assertEqual(key, "md")
                self.assertEqual(size, 6)
                self.assertEqual(counts, {"train": 3, "validation": 1, "test": 2})
                self.assertEqual(config["splits"]["strategy"], "spread")
                self.assertTrue((dataset_dir / "splits" / "test" / "1" / "RUN.fdf").exists())
                self.assertTrue((dataset_dir / "splits" / "test" / "4" / "RUN.fdf").exists())
                return {}

            def fake_process(*_args, **_kwargs):
                return 0

            def fake_archive(key, size, run_id, workspace, config, returncode, run_log, prepare_metadata, **kwargs):
                return {
                    "pipeline": key,
                    "method_id": key,
                    "dataset_label": kwargs.get("dataset_label"),
                    "dataset_size": size,
                    "returncode": returncode,
                    "run_mode": kwargs.get("run_mode"),
                    "dataset_dir": config["paths"]["dataset_dir"],
                    "result_dir": str(root / "archived"),
                    "dataset_reused": prepare_metadata.get("dataset_reused"),
                    "reused_split_policy": prepare_metadata.get("reused_split_policy"),
                    "reused_split_counts": prepare_metadata.get("reused_split_counts"),
                }

            runner._prepare_md_config = forbidden_generation  # type: ignore[method-assign]
            runner._validate_split_manifests = fake_validate  # type: ignore[method-assign]
            runner._run_pipeline_process = fake_process  # type: ignore[method-assign]
            runner._archive_outputs = fake_archive  # type: ignore[method-assign]

            result = runner._run_one(
                "md",
                6,
                "reuse_resplit",
                dataset_label="existing_md",
                split_ratios={"train": 0.5, "validation": 1 / 6, "test": 1 / 3},
                split_mode="spread",
                run_mode="train_test_metrics_plots_only",
                reusable_split_policy="rebuild_splits",
            )

            self.assertTrue(result["dataset_reused"])
            self.assertEqual(result["reused_split_policy"], "rebuild_splits")
            self.assertEqual(result["reused_split_counts"], {"train": 3, "validation": 1, "test": 2})

    def test_downstream_only_md_rebuild_includes_excluded_gap_sources(self) -> None:
        module = self.load_pipeline_ui_module()
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            dataset_dir = root / "safe_md" / "dataset"
            (dataset_dir / "MD_steps" / "basis").mkdir(parents=True)
            (dataset_dir / "MD_steps" / "basis" / "H.ion.xml").write_text("<ion />\n", encoding="utf-8")
            (dataset_dir / "RUN.out").write_text("Job completed\nSCF cycle converged\n", encoding="utf-8")
            rows_by_split: dict[str, list[dict[str, str]]] = {
                "train": [],
                "validation": [],
                "test": [],
                "excluded_gap": [],
            }
            split_for_index = {
                0: "train",
                1: "train",
                2: "train",
                3: "train",
                4: "train",
                5: "train",
                6: "excluded_gap",
                7: "validation",
                8: "excluded_gap",
                9: "test",
            }
            for index, split_name in split_for_index.items():
                raw_dir = dataset_dir / "MD_steps" / str(index)
                raw_dir.mkdir(parents=True)
                minimal_run_fdf(raw_dir / "RUN.fdf")
                (raw_dir / "siesta.TSHS").write_bytes(b"matrix")
                sample_dir = raw_dir if split_name == "excluded_gap" else dataset_dir / "splits" / split_name / str(index)
                if split_name != "excluded_gap":
                    sample_dir.mkdir(parents=True)
                    minimal_run_fdf(sample_dir / "RUN.fdf")
                    (sample_dir / "siesta.TSHS").write_bytes(b"matrix")
                rows_by_split[split_name].append(
                    {
                        "sample_id": f"md_{index}",
                        "method": "md",
                        "structure_path": str(sample_dir / "RUN.fdf"),
                        "hamiltonian_path": str(sample_dir / "siesta.TSHS"),
                        "run_out_path": str(dataset_dir / "RUN.out"),
                        "status": "excluded" if split_name == "excluded_gap" else "completed",
                        "valid": "false" if split_name == "excluded_gap" else "true",
                        "split": split_name,
                        "split_strategy": "blocked_with_gap",
                        "temporal_gap": "1",
                        "source_frame_index": str(index),
                        "excluded_gap_reason": (
                            "temporal_gap_between_train_and_validation"
                            if index == 6
                            else "temporal_gap_between_validation_and_test" if index == 8 else ""
                        ),
                        "sample_dir": str(sample_dir),
                    }
                )
            for split_name, rows in rows_by_split.items():
                write_csv(dataset_dir / "splits" / f"{split_name}_manifest.csv", rows)

            runner = module.ExperimentRunner()
            runner._validate_existing_dataset_source("md", dataset_dir, 10)
            rebuilt = runner._rebuild_reusable_dataset_splits(
                "md",
                dataset_dir,
                10,
                {"train": 0.8, "validation": 0.1, "test": 0.1},
                "blocked_with_gap",
            )

            self.assertEqual(rebuilt["reused_split_counts"], {"train": 6, "validation": 1, "test": 1})
            self.assertEqual(rebuilt["reused_temporal_gap"], 1)
            self.assertEqual(rebuilt["reused_excluded_gap_count"], 2)
            with (dataset_dir / "splits" / "excluded_gap_manifest.csv").open(encoding="utf-8") as handle:
                excluded_rows = list(csv.DictReader(handle))
            self.assertEqual([row["source_frame_index"] for row in excluded_rows], ["6", "8"])
            self.assertTrue((dataset_dir / "splits" / "validation" / "7" / "RUN.fdf").exists())
            self.assertTrue((dataset_dir / "splits" / "test" / "9" / "RUN.fdf").exists())

    def test_downstream_only_md_can_select_exact_reusable_dataset(self) -> None:
        module = self.load_pipeline_ui_module()
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            module.RESULTS_ROOT = root / "results"
            module.WORKSPACES_ROOT = root / "workspaces"
            source_a = make_reusable_md_dataset(root / "source_a")
            source_b = make_reusable_md_dataset(root / "source_b")
            (source_b / "selected_source_marker.txt").write_text("selected\n", encoding="utf-8")
            for run_name, dataset_dir in (("run_a", source_a), ("run_b", source_b)):
                write_archived_retrainable_run(
                    module.RESULTS_ROOT,
                    method_id="md",
                    dataset_label="same_label",
                    dataset_size=3,
                    dataset_dir=dataset_dir,
                    run_id=run_name,
                )

            runner = module.ExperimentRunner()
            runs_by_dataset = {
                Path(str(run["dataset_dir"])): run
                for run in runner._archived_dataset_run_candidates()  # type: ignore[attr-defined]
            }
            selected_id = runner._dataset_candidate_id(runs_by_dataset[source_b])  # type: ignore[attr-defined]
            info = runner.dataset_recipes_info_for_reusable_dataset_ids(
                [selected_id],
                selected_methods=["md"],
            )
            self.assertEqual(info["md_dataset_specs"][0]["recipe_metadata"]["reuse_source_id"], selected_id)

            def forbidden_generation(*_args, **_kwargs):
                raise AssertionError("dataset generation should not be called")

            def fake_validate(*_args, **_kwargs):
                return {}

            def fake_process(*_args, **_kwargs):
                return 0

            def fake_archive(key, size, run_id, workspace, config, returncode, run_log, prepare_metadata, **kwargs):
                return {
                    "pipeline": key,
                    "method_id": key,
                    "dataset_label": kwargs.get("dataset_label"),
                    "dataset_size": size,
                    "returncode": returncode,
                    "run_mode": kwargs.get("run_mode"),
                    "dataset_dir": config["paths"]["dataset_dir"],
                    "result_dir": str(root / "archived"),
                    "dataset_reused": prepare_metadata.get("dataset_reused"),
                    "reused_dataset_dir": prepare_metadata.get("reused_dataset_dir"),
                }

            runner._prepare_md_config = forbidden_generation  # type: ignore[method-assign]
            runner._validate_split_manifests = fake_validate  # type: ignore[method-assign]
            runner._run_pipeline_process = fake_process  # type: ignore[method-assign]
            runner._archive_outputs = fake_archive  # type: ignore[method-assign]
            spec = info["md_dataset_specs"][0]
            result = runner._run_one(
                "md",
                int(spec["size"]),
                "explicit_reuse",
                dataset_label=str(spec["label"]),
                recipe_metadata=spec["recipe_metadata"],
                run_mode="train_test_metrics_plots_only",
            )

            self.assertTrue(result["dataset_reused"])
            self.assertEqual(result["reused_dataset_dir"], str(source_b))
            copied_dataset = module.WORKSPACES_ROOT / "explicit_reuse" / "md" / "same_label" / "dataset"
            self.assertTrue((copied_dataset / "selected_source_marker.txt").exists())

    def test_downstream_only_random_cartesian_reuses_existing_dataset_without_generation(self) -> None:
        module = self.load_pipeline_ui_module()
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            module.RESULTS_ROOT = root / "results"
            module.WORKSPACES_ROOT = root / "workspaces"
            source_dataset = make_reusable_atom_like_dataset(
                root / "source_random_cartesian",
                source_root_name="RandomCartesian_steps",
            )
            write_archived_retrainable_run(
                module.RESULTS_ROOT,
                method_id="random_cartesian",
                dataset_label="existing_random",
                dataset_size=3,
                dataset_dir=source_dataset,
            )

            runner = module.ExperimentRunner()
            calls: list[tuple[str, object]] = []

            def forbidden_generation(*_args, **_kwargs):
                raise AssertionError("dataset generation should not be called")

            def fake_validate(key, config, size):
                calls.append(("validate", (key, size, config["paths"]["dataset_dir"])))
                return {}

            def fake_process(spec, **kwargs):
                config = module.load_config(kwargs["config_path"])
                calls.append(("process", tuple(config["pipeline"]["steps"])))
                self.assertEqual(tuple(config["pipeline"]["steps"]), module.ATOM_DOWNSTREAM_STEPS)
                self.assertIn(
                    "workspaces/reuse_random/random_cartesian/existing_random/dataset",
                    config["paths"]["dataset_dir"],
                )
                self.assertEqual(
                    config["paths"]["samples_dir"].split("/")[-1],
                    "train_samples",
                )
                self.assertNotEqual(config["paths"]["dataset_dir"], str(source_dataset))
                return 0

            def fake_archive(key, size, run_id, workspace, config, returncode, run_log, prepare_metadata, **kwargs):
                calls.append(("archive", prepare_metadata.get("dataset_reused")))
                return {
                    "pipeline": key,
                    "method_id": key,
                    "dataset_label": kwargs.get("dataset_label"),
                    "dataset_size": size,
                    "returncode": returncode,
                    "run_mode": kwargs.get("run_mode"),
                    "dataset_dir": config["paths"]["dataset_dir"],
                    "result_dir": str(root / "archived"),
                    "dataset_reused": prepare_metadata.get("dataset_reused"),
                }

            runner._prepare_atom_config = forbidden_generation  # type: ignore[method-assign]
            runner._validate_split_manifests = fake_validate  # type: ignore[method-assign]
            runner._run_pipeline_process = fake_process  # type: ignore[method-assign]
            runner._archive_outputs = fake_archive  # type: ignore[method-assign]

            result = runner._run_random_cartesian(
                3,
                "reuse_random",
                dataset_label="existing_random",
                run_mode="train_test_metrics_plots_only",
            )

            self.assertTrue(result["dataset_reused"])
            self.assertEqual(calls[0][0], "validate")
            self.assertEqual(calls[1], ("process", module.ATOM_DOWNSTREAM_STEPS))
            self.assertEqual(calls[2], ("archive", True))

    def test_random_cartesian_dataset_only_prepares_reusable_splits(self) -> None:
        module = self.load_pipeline_ui_module()
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            module.RESULTS_ROOT = root / "results"
            module.WORKSPACES_ROOT = root / "workspaces"
            runner = module.ExperimentRunner()
            calls: list[tuple[str, object]] = []

            def fake_process(spec, **kwargs):
                config = module.load_config(kwargs["config_path"])
                calls.append(("process", tuple(config["pipeline"]["steps"])))
                return 0

            def fake_prepare_atom_config(config, workspace, size, split_ratios, source_samples_dir=None, method_id="atom_displacement"):
                calls.append(("prepare_splits", (method_id, Path(source_samples_dir).name if source_samples_dir else None)))
                dataset_dir = Path(config["paths"]["dataset_dir"])
                config["paths"]["samples_dir"] = str(dataset_dir / "train_samples")
                config["paths"]["validation_samples_dir"] = str(dataset_dir / "validation_samples")
                config["paths"]["test_samples_dir"] = str(dataset_dir / "test_samples")
                config["paths"]["training_dir"] = str(workspace / "training")
                config["pipeline"]["steps"] = list(module.ATOM_DOWNSTREAM_STEPS)
                return {
                    "requested_size": size,
                    "effective_size": size,
                    "generated_samples": size,
                    "completed_samples": size,
                    "test_needs_siesta": False,
                    "dataset_preparation_seconds": 0.1,
                }

            def fake_validate(key, config, size):
                calls.append(("validate", (key, tuple(config["pipeline"]["steps"]))))
                return {}

            def fake_archive(key, size, run_id, workspace, config, returncode, run_log, prepare_metadata, **kwargs):
                calls.append(("archive", tuple(config["pipeline"]["steps"])))
                return {
                    "pipeline": key,
                    "method_id": key,
                    "dataset_label": kwargs.get("dataset_label"),
                    "dataset_size": size,
                    "returncode": returncode,
                    "run_mode": kwargs.get("run_mode"),
                    "dataset_dir": config["paths"]["dataset_dir"],
                    "result_dir": str(root / "archived"),
                    "generated_samples": prepare_metadata.get("generated_samples"),
                }

            runner._run_pipeline_process = fake_process  # type: ignore[method-assign]
            runner._prepare_atom_config = fake_prepare_atom_config  # type: ignore[method-assign]
            runner._validate_split_manifests = fake_validate  # type: ignore[method-assign]
            runner._archive_outputs = fake_archive  # type: ignore[method-assign]

            result = runner._run_random_cartesian(
                3,
                "rc_dataset_only",
                dataset_label="rc_3",
                run_mode="dataset_only",
                split_ratios={"train": 1 / 3, "validation": 1 / 3, "test": 1 / 3},
            )

            self.assertEqual(result["run_mode"], "dataset_only")
            self.assertEqual(calls[0][0], "process")
            self.assertIn("generate_random_cartesian_dataset", calls[0][1])
            self.assertEqual(calls[1], ("prepare_splits", ("random_cartesian", "RandomCartesian_steps")))
            self.assertEqual(calls[2], ("validate", ("atom_displacement", module.ATOM_DOWNSTREAM_STEPS)))
            self.assertEqual(calls[3], ("archive", ()))

    def test_downstream_only_fails_clearly_when_dataset_is_missing(self) -> None:
        module = self.load_pipeline_ui_module()
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            module.RESULTS_ROOT = root / "results"
            module.WORKSPACES_ROOT = root / "workspaces"
            runner = module.ExperimentRunner()

            def forbidden_generation(*_args, **_kwargs):
                raise AssertionError("dataset generation should not be called")

            runner._prepare_md_config = forbidden_generation  # type: ignore[method-assign]
            with self.assertRaisesRegex(RuntimeError, "No existing dataset found"):
                runner._run_one(
                    "md",
                    3,
                    "missing_reuse",
                    dataset_label="missing_md",
                    run_mode="train_test_metrics_plots_only",
                )

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
            arg_names = [
                "md_sizes",
                "atom_sizes",
                "run_id",
                "fc_dataset_specs",
                "atom_dataset_specs",
                "split_ratios",
                "random_seed",
                "split_mode",
                "test_sets",
                "primary_metric",
                "compute_budget_mode",
                "compute_accelerator",
                "selected_methods",
                "run_mode",
                "random_cartesian_options",
                "performance_settings",
                "training_settings",
                "venv_activate_path",
                "dataset_recipes_info",
                "reusable_split_policy",
                "training_plan",
                "material_config",
            ]
            self.assertEqual(len(started["args"]), len(arg_names))
            args = dict(zip(arg_names, started["args"]))
            self.assertEqual(args["compute_accelerator"], "gpu")
            self.assertEqual(args["selected_methods"], ["random_cartesian"])
            self.assertEqual(args["run_mode"], "full_strict_pipeline")
            self.assertEqual(args["random_cartesian_options"], {"n_structures": 3})
            self.assertEqual(args["performance_settings"]["compute_accelerator"], "gpu")
            self.assertEqual(args["training_settings"], {})
            self.assertIsNone(args["venv_activate_path"])
            self.assertIn("recipes", args["dataset_recipes_info"])
            self.assertEqual(args["reusable_split_policy"], "preserve_archived_splits")
            self.assertEqual(args["training_plan"], [])
            self.assertIsNone(args["material_config"])

    def test_experiment_start_rejects_zero_selected_methods(self) -> None:
        module = self.load_pipeline_ui_module()
        runner = module.ExperimentRunner()
        with self.assertRaisesRegex(RuntimeError, "Selecciona al menos un metodo"):
            runner.start([], [], selected_methods=[], run_mode="dataset_only")

    def test_ui_experiment_payload_includes_methods_and_run_mode(self) -> None:
        index_html = (REPO_ROOT / "Comparison" / "ui" / "index.html").read_text(encoding="utf-8")
        app_js = (REPO_ROOT / "Comparison" / "ui" / "app.js").read_text(encoding="utf-8")
        styles_css = (REPO_ROOT / "Comparison" / "ui" / "styles.css").read_text(encoding="utf-8")
        pipeline_ui = (REPO_ROOT / "Comparison" / "scripts" / "pipeline_ui.py").read_text(encoding="utf-8")
        self.assertIn('value="md"', index_html)
        self.assertIn('value="siesta_fc_cartesian"', index_html)
        self.assertIn('value="random_cartesian"', index_html)
        self.assertIn('class="method-execution-checkbox"', index_html)
        self.assertIn('value="random_cartesian" checked', index_html)
        self.assertIn('class="tab hidden" data-view="run"', index_html)
        self.assertIn('class="tab active" data-view="experiment"', index_html)
        self.assertIn('id="view-run" class="view"', index_html)
        self.assertIn('id="view-experiment" class="view active"', index_html)
        self.assertIn('id="show-plots" type="checkbox" checked', index_html)
        self.assertIn('value="train_test_metrics_plots_only"', index_html)
        self.assertIn('id="reusable-dataset-panel"', index_html)
        self.assertIn('id="reusable-dataset-list"', index_html)
        self.assertIn('id="refresh-reusable-datasets"', index_html)
        self.assertIn('id="reusable-split-policy"', index_html)
        self.assertIn('id="training-plan-panel"', index_html)
        self.assertIn('id="planned-dataset-target-panel"', index_html)
        self.assertIn('id="planned-dataset-target-list"', index_html)
        self.assertIn('id="add-training-plan-entry"', index_html)
        self.assertIn('class="metric-info-bubble"', index_html)
        self.assertIn("Metric guide", index_html)
        self.assertIn("Low-energy RMSE", index_html)
        self.assertIn("DOS W1", index_html)
        self.assertIn("Hamiltonian R2", index_html)
        self.assertIn("DOS MAE 500 Fermi window", index_html)
        self.assertIn("DeepH-comparable metrics", index_html)
        self.assertIn('<option value="mse_ref_eV2">Sparse MSE on ref</option>', index_html)
        self.assertIn('<option value="r2_ref">Sparse R2 on ref</option>', index_html)
        self.assertIn('<option value="dos_mae_500_fermi_window">DOS MAE 500 Fermi window</option>', index_html)
        self.assertIn("METRIC_HELP", app_js)
        self.assertIn("PLOT_HELP_BY_ID", app_js)
        self.assertIn("CROSS_PLOT_HELP_BY_ID", app_js)
        self.assertIn('id="plot-deeph-mev"', index_html)
        self.assertIn('id="plot-deeph-mse"', index_html)
        self.assertIn('id="plot-deeph-r2"', index_html)
        self.assertIn('id="plot-deeph-dos"', index_html)
        self.assertIn('id="plot-orbital-pair"', index_html)
        self.assertIn("installPlotInfoBubble", app_js)
        self.assertIn("plotInfoFor", app_js)
        self.assertIn("plot-info-bubble", app_js)
        self.assertNotIn("plotly-2.35.2.min.js", index_html)
        self.assertNotIn("tex-chtml.js", index_html)
        self.assertIn("ensurePlotlyLoaded", app_js)
        self.assertIn("ensureMathJaxLoaded", app_js)
        self.assertIn("plotly-2.35.2.min.js", app_js)
        self.assertIn("tex-chtml.js", app_js)
        self.assertIn("typesetPlotInfoMath", app_js)
        self.assertIn("plot-info-formula", app_js)
        self.assertIn('["Formula", info.formula]', app_js)
        self.assertIn(r"\\operatorname{RMSE}_S", app_js)
        self.assertIn(r"\\|H^{pred}-H^{ref}\\|_{F,union}", app_js)
        self.assertIn(r"\\frac{2\\,\\mathrm{precision}\\,\\mathrm{recall}}", app_js)
        self.assertIn(r"\\operatorname{mean}_{\\text{finite rows}}", app_js)
        self.assertIn("DOS Wasserstein-1", app_js)
        self.assertIn("dos_mae_500_fermi_window", app_js)
        self.assertIn("mse_ref_eV2", app_js)
        self.assertIn("r2_ref", app_js)
        self.assertIn("plot-deeph-mev", app_js)
        self.assertIn("plot-deeph-mse", app_js)
        self.assertIn("plot-deeph-r2", app_js)
        self.assertIn("plot-deeph-dos", app_js)
        self.assertIn("plot-orbital-pair", app_js)
        self.assertIn("plotsEnabled: true", app_js)
        self.assertIn("orbital_pair_summary", pipeline_ui)
        self.assertIn("metricHigherIsBetter", app_js)
        self.assertIn("Support F1", app_js)
        self.assertIn('"dos_mae_500_fermi_window"', pipeline_ui)
        self.assertIn('"mse_union_eV2"', pipeline_ui)
        self.assertIn('"r2_union"', pipeline_ui)
        self.assertIn(".plot-info-bubble", styles_css)
        self.assertIn(".plot-info-popover", styles_css)
        self.assertIn(".plot-info-formula", styles_css)
        self.assertIn(".plot-section-heading", styles_css)
        self.assertNotIn('id="random-cartesian-n-structures"', index_html)
        self.assertNotIn("phonon", index_html.lower())
        self.assertIn("selected_methods: methods", app_js)
        self.assertIn("run_mode: runMode", app_js)
        self.assertIn("reusable_dataset_ids: reusableDatasetIds", app_js)
        self.assertIn("reusable_split_policy: reusableSplitPolicy()", app_js)
        self.assertIn("training_plan: plan", app_js)
        self.assertIn("trainingPlanPayload()", app_js)
        self.assertIn("dataset_targets", app_js)
        self.assertIn("plannedDatasetTargetsFromRecipes", app_js)
        self.assertIn("defaultRecipeIdForMethod", app_js)
        self.assertIn("fc_recipe_", app_js)
        self.assertIn("rc_recipe_", app_js)
        self.assertIn("selectedPlannedDatasetTargets()", app_js)
        self.assertIn("/api/datasets/reusable", app_js)
        self.assertIn("selectedReusableDatasetIds()", app_js)
        self.assertIn("run?.training_tag", app_js)
        self.assertIn("item.training_tag", app_js)
        self.assertIn("includeTrainingContext", app_js)
        self.assertIn("runTrainingGroupLabel", app_js)
        self.assertIn('value="source ${REPO_ROOT}/.venv/bin/activate"', index_html)
        self.assertIn('DEFAULT_VENV_ACTIVATE_COMMAND = "source ${REPO_ROOT}/.venv/bin/activate"', app_js)
        self.assertNotIn("/home/christian/graph2mat-env", index_html)
        self.assertNotIn("/home/christian/graph2mat-env", app_js)
        self.assertNotIn('id="compute-accelerator"', index_html)
        self.assertIn('value="cpu"', index_html)
        self.assertIn('value="gpu" selected', index_html)
        self.assertIn('value="auto"', index_html)
        self.assertIn('<option value="balanced" selected>Balanced</option>', index_html)
        self.assertIn('<option value="auto_detect">Auto detect</option>', index_html)
        self.assertIn('<option value="max_aggressive">Max aggressive / stress</option>', index_html)
        self.assertIn('id="performance-batch-size" type="number" step="1" min="1" value="128"', index_html)
        self.assertIn('id="performance-torch-num-threads" type="number" step="1" min="1" value="8"', index_html)
        self.assertIn('<option value="high" selected>high</option>', index_html)
        self.assertIn('<option value="true" selected>true</option>', index_html)
        self.assertIn('id="performance-preset-description"', index_html)
        self.assertIn('id="performance-hardware-summary"', index_html)
        self.assertIn('id="performance-preset-warnings"', index_html)
        self.assertIn('id="performance-compute-accelerator"', index_html)
        self.assertIn('id="performance-max-parallel-siesta-jobs"', index_html)
        self.assertIn('id="performance-max-parallel-dataset-jobs"', index_html)
        self.assertIn('id="performance-max-parallel-prediction-jobs"', index_html)
        self.assertIn('id="performance-max-parallel-evaluation-jobs"', index_html)
        self.assertIn('id="performance-max-parallel-metric-jobs"', index_html)
        self.assertIn('id="performance-numexpr-num-threads"', index_html)
        self.assertIn('id="performance-reuse-validated-siesta-outputs"', index_html)
        self.assertIn('id="performance-enable-experiment-cache"', index_html)
        self.assertIn('id="performance-error-policy"', index_html)
        self.assertIn('id="training-max-epochs"', index_html)
        self.assertIn('id="training-optim-lr"', index_html)
        self.assertIn('id="training-batch-size"', index_html)
        self.assertIn('id="training-loader-threads"', index_html)
        self.assertIn('id="training-num-interactions"', index_html)
        self.assertIn('id="training-correlation"', index_html)
        self.assertIn('id="training-max-ell"', index_html)
        self.assertIn('id="training-hidden-irreps"', index_html)
        self.assertIn('id="training-hidden-irreps-alert"', index_html)
        self.assertIn('id="training-loss"', index_html)
        self.assertIn("performanceSettings()", app_js)
        self.assertIn("trainingSettings()", app_js)
        self.assertIn("validateHiddenIrrepsText", app_js)
        self.assertIn("renderHiddenIrrepsValidation", app_js)
        self.assertIn("loadPerformancePresets()", app_js)
        self.assertIn("/api/performance-presets", app_js)
        self.assertIn("applyPerformancePreset", app_js)
        self.assertNotIn("experimentAccelerator", app_js)
        self.assertIn('id="sync-md-sizes"', index_html)
        self.assertIn('id="md-dataset-editor"', index_html)
        self.assertIn('id="md-add-dataset"', index_html)
        self.assertIn('id="md-dataset-table"', index_html)
        self.assertIn('id="fc-dataset-editor"', index_html)
        self.assertIn('id="fc-add-dataset"', index_html)
        self.assertNotIn('id="fc-dataset-preview"', index_html)
        self.assertNotIn("Dataset preview", index_html)
        self.assertIn('id="random-dataset-editor"', index_html)
        self.assertIn('id="random-add-dataset"', index_html)
        self.assertIn('id="random-cartesian-dataset-table"', index_html)
        self.assertIn("Datasets MD: snapshots | temperatura", index_html)
        self.assertIn("Datasets FC Cartesian: snapshots | desplazamiento", index_html)
        self.assertIn('id="dataset-recipes-json"', index_html)
        self.assertIn('id="use-dataset-recipes-json"', index_html)
        self.assertIn('id="export-dataset-recipes"', index_html)
        self.assertNotIn('id="random-cartesian-distribution"', index_html)
        self.assertNotIn('id="random-cartesian-sigma-ang"', index_html)
        self.assertNotIn('id="random-cartesian-min-distance-ang"', index_html)
        self.assertNotIn('id="random-cartesian-enable-atom"', index_html)
        self.assertNotIn('id="random-cartesian-seed"', index_html)
        self.assertNotIn("random-global-grid", index_html)
        self.assertIn("Dataset builder", index_html)
        self.assertIn("setupDatasetEditors()", app_js)
        self.assertIn("parseDatasetRecipes()", app_js)
        self.assertIn("parseMdDatasetTableSpecs()", app_js)
        self.assertIn("parseMdTemperatureBlocks()", app_js)
        self.assertIn("parseFcDatasetTableSpecs()", app_js)
        self.assertNotIn("renderFcPreview", app_js)
        self.assertIn("parseRandomCartesianDatasetTableSpecs()", app_js)
        self.assertIn("parseRandomCartesianDatasetSpecsFromEditor", app_js)
        self.assertIn('row.querySelector(\'[data-field="count"]\')', app_js)
        self.assertNotIn('parseNumberListInput("random-cartesian-n-structures"', app_js)
        self.assertIn('data-field="dataset-seed"', app_js)
        self.assertIn('document.createElement("details")', app_js)
        self.assertIn('className = "dataset-card"', app_js)
        self.assertIn('class="dataset-card-body"', app_js)
        self.assertIn('class="dataset-card-chevron"', app_js)
        self.assertIn("applyDatasetSeeds", app_js)
        self.assertIn("datasetSeedPatch", app_js)
        self.assertIn("syncMdTableFromSizes", app_js)
        self.assertIn("builderDatasetRecipes", app_js)
        self.assertIn("exportCurrentDatasetRecipes", app_js)
        self.assertIn("dataset_recipes: datasetRecipes", app_js)
        self.assertIn('sync_md_sizes: Boolean(document.getElementById("sync-md-sizes")?.checked)', app_js)
        self.assertIn("max_parallel_dataset_jobs", app_js)
        self.assertIn("reuse_validated_siesta_outputs", app_js)
        self.assertIn("error_policy", app_js)
        self.assertIn("performance,", app_js)
        self.assertIn("training_settings: training", app_js)
        self.assertIn("compute_accelerator: performance.compute_accelerator", app_js)
        self.assertIn("parseRandomCartesianOptions(methods)", app_js)
        self.assertIn("random_cartesian_options: randomCartesianOptions", app_js)
        self.assertIn("habilita al menos atom, bond o angle displacement", app_js)
        self.assertIn("atom_displacement", app_js)
        self.assertIn("bond_displacement", app_js)
        self.assertIn("angle_displacement", app_js)
        self.assertIn("validation = {", app_js)
        self.assertIn("randomCartesianSettingsFromCard", app_js)
        self.assertIn("applyRandomCartesianDatasetSettings", app_js)
        self.assertIn("rows[blockIndex]", app_js)
        self.assertIn("component ${blockIndex + 1}", app_js)
        self.assertIn('class="random-block-details"', app_js)
        self.assertIn('data-field="rc-enable-atom"', app_js)
        self.assertIn('data-field="rc-atom-sigma"', app_js)
        self.assertIn('data-field="rc-enable-bond"', app_js)
        self.assertIn('data-field="rc-bond-sigma"', app_js)
        self.assertIn('data-field="rc-bond-min-delta"', app_js)
        self.assertIn('data-field="rc-enable-angle"', app_js)
        self.assertIn('data-field="rc-angle-sigma"', app_js)
        self.assertIn('data-field="rc-angle-min-delta"', app_js)
        self.assertIn('data-field="rc-max-rmsd"', app_js)
        self.assertIn('data-field="rc-max-attempts"', app_js)
        self.assertIn("n_structures", app_js)
        self.assertNotIn("random-cartesian-enable-atom", app_js)
        self.assertNotIn("random-cartesian-seed", app_js)
        self.assertNotIn("parseMoveAtomsInput", app_js)
        self.assertNotIn("syncRandomCartesianSizeFromAtomPlan", app_js)
        self.assertNotIn("validSizes.join", app_js)
        self.assertIn("resultPipelines", app_js)
        self.assertIn("results_random_cartesian", app_js)
        self.assertIn("wasRunning && !status.running && state.plotsEnabled", app_js)
        self.assertIn("setMethodSelected", app_js)
        self.assertIn("Selecciona al menos un metodo", app_js)
        self.assertIn('id="plot-cross-selection"', index_html)
        self.assertIn('id="plot-cross-metric"', index_html)
        self.assertIn('<option value="all" selected>All experiments</option>', index_html)
        self.assertIn('<option value="low_energy_rmse_eV" selected>', index_html)
        self.assertIn('<option value="low_energy_rmse_eV" selected>Low-energy eigenvalue RMSE</option>', index_html)
        self.assertIn('id="clear-datasets"', index_html)
        self.assertGreater(index_html.index('id="clear-datasets"'), index_html.index('id="plot-winner"'))
        self.assertIn('id="delete-selected-datasets"', index_html)
        self.assertIn('id="dataset-cleanup-list"', index_html)
        self.assertIn("clearGeneratedDatasets()", app_js)
        self.assertIn("/api/datasets/clear", app_js)
        self.assertIn("/api/datasets/targets", app_js)
        self.assertIn("selectedCrossExperimentSet(payload)", app_js)
        self.assertIn("selectedCrossMetric", app_js)
        self.assertIn("groupedCrossMetrics", app_js)
        self.assertIn("No metric", app_js)
        self.assertIn("n_finite", app_js)
        self.assertIn("sin bandas dentro de ±2 eV de Fermi", app_js)
        self.assertIn("Frontier fallback", app_js)
        self.assertIn("no se reescribe Fermi-window RMSE", app_js)
        self.assertNotIn("function groupedCrossMeans", app_js)
        self.assertNotIn("function latestCrossExperiment", app_js)
        self.assertNotIn("sync_md_sizes: true", app_js)
        self.assertIn('<option value="test_random_cartesian" selected>', index_html)
        self.assertIn("crossTrainMethods(experiment)", app_js)
        self.assertIn("crossTestSets(experiment)", app_js)
        self.assertNotIn('const trainMethods = ["md", "atom_displacement"]', app_js)

    def test_ui_sidebar_visible_navigation_order(self) -> None:
        index_html = (REPO_ROOT / "Comparison" / "ui" / "index.html").read_text(encoding="utf-8")
        expected = [
            'data-view="experiment"',
            'data-view="results"',
            'data-view="performance"',
            'data-view="api"',
        ]
        positions = [index_html.index(item) for item in expected]
        self.assertEqual(positions, sorted(positions))
        self.assertLess(index_html.index('data-view="run"'), positions[0])
        self.assertIn('class="tab hidden" data-view="run"', index_html)

    def test_ui_plotly_traces_use_scatter_points_with_selectable_fit_lines(self) -> None:
        app_js = (REPO_ROOT / "Comparison" / "ui" / "app.js").read_text(encoding="utf-8")
        self.assertIn('mode: "markers"', app_js)
        self.assertIn('mode: "lines"', app_js)
        self.assertIn("aggregateFitPoints", app_js)
        self.assertIn("grouped.has(x)", app_js)
        self.assertIn("polynomialCoefficients", app_js)
        self.assertIn('kind === "quadratic" ? 2 : 1', app_js)
        self.assertIn('label: "Linear fit"', app_js)
        self.assertIn('label: "Quadratic fit"', app_js)
        self.assertIn('label: "No fit"', app_js)
        self.assertIn("withFitSelector", app_js)
        self.assertIn('visible: kind === "linear"', app_js)
        self.assertIn('hoverinfo: "skip"', app_js)
        self.assertNotIn("aggregateSmoothPoints", app_js)
        self.assertNotIn("smoothLineTrace", app_js)
        self.assertNotIn("smoothing:", app_js)
        self.assertNotIn("shape:", app_js)
        self.assertNotIn('mode: "lines+markers"', app_js)
        self.assertNotIn("mode: 'lines+markers'", app_js)
        self.assertNotIn("mode: 'lines'", app_js)

    def test_ui_and_backend_default_primary_metric_is_low_energy(self) -> None:
        index_html = (REPO_ROOT / "Comparison" / "ui" / "index.html").read_text(encoding="utf-8")
        app_js = (REPO_ROOT / "Comparison" / "ui" / "app.js").read_text(encoding="utf-8")
        script = (REPO_ROOT / "Comparison" / "scripts" / "pipeline_ui.py").read_text(encoding="utf-8")
        self.assertIn('DEFAULT_PRIMARY_METRIC = "low_energy_rmse_eV"', script)
        self.assertIn('const PRIMARY_METRIC_DEFAULT = "low_energy_rmse_eV"', app_js)
        self.assertIn('<option value="low_energy_rmse_eV" selected>Low-energy eigenvalue RMSE</option>', index_html)
        self.assertIn("POST /api/datasets/clear", index_html)

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
            availability = runs[0]["metric_availability"]["spectral"]
            self.assertEqual(availability["fermi_window_rmse_eV"]["n_total"], 1)
            self.assertEqual(availability["fermi_window_rmse_eV"]["n_finite"], 0)
            self.assertEqual(availability["low_energy_rmse_eV"]["n_total"], 1)
            archived = module.archived_results_summary()
            self.assertIn("random_cartesian", archived)
            self.assertEqual(len(archived["random_cartesian"]), 1)
            results = module.result_summary()
            self.assertIn("random_cartesian", results)
            self.assertEqual(
                results["random_cartesian"]["prediction_glob"],
                "AtomDisplacement/dataset/RandomCartesian_steps/*/ML_prediction.HSX",
            )

    def test_orbital_pair_diagnostic_outputs_are_discoverable_without_loading_rows(self) -> None:
        module = self.load_pipeline_ui_module()
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            module.RESULTS_ROOT = root / "results"
            old_dir = module.RESULTS_ROOT / "results_md" / "dataset_3" / "run_old"
            new_dir = module.RESULTS_ROOT / "results_md" / "dataset_4" / "run_new"
            for result_dir, run_id, size in ((old_dir, "old", 3), (new_dir, "new", 4)):
                metrics_dir = result_dir / "metrics"
                metrics_dir.mkdir(parents=True)
                write_csv(
                    metrics_dir / "sparse_metrics.csv",
                    [{"sample": "sample_1", "relative_frobenius_union": "0.25"}],
                )
                write_csv(
                    metrics_dir / "spectral_metrics.csv",
                    [{"sample": "sample_1", "low_energy_rmse_eV": "0.4"}],
                )
                (result_dir / "manifest.json").write_text(
                    json.dumps(
                        {
                            "pipeline": "md",
                            "method_id": "md",
                            "run_id": run_id,
                            "result_dir": str(result_dir),
                            "dataset_size": size,
                            "requested_dataset_size": size,
                            "dataset_label": f"dataset_{size}",
                        }
                    ),
                    encoding="utf-8",
                )
            write_csv(
                new_dir / "metrics" / "orbital_pair_metrics.csv",
                [
                    {
                        "sample": "sample_1",
                        "row_species": "H",
                        "col_species": "H",
                        "species_pair": "H-H",
                        "row_orbital_index": "0",
                        "col_orbital_index": "0",
                        "mae_union_meV": "12.5",
                    }
                ],
            )
            write_csv(
                new_dir / "metrics" / "orbital_pair_summary.csv",
                [
                    {
                        "row_species": "H",
                        "col_species": "H",
                        "species_pair": "H-H",
                        "row_orbital_index": "0",
                        "col_orbital_index": "0",
                        "mae_union_meV_mean": "12.5",
                    }
                ],
            )

            plots = module.plot_data_summary()
            runs = {run["run_id"]: run for run in plots["runs"]}
            self.assertFalse(runs["old"]["diagnostic_outputs"]["orbital_pair_metrics"]["exists"])
            self.assertTrue(runs["new"]["diagnostic_outputs"]["orbital_pair_metrics"]["exists"])
            self.assertEqual(runs["new"]["diagnostic_outputs"]["orbital_pair_metrics"]["rows"], 1)
            self.assertEqual(runs["new"]["diagnostic_outputs"]["orbital_pair_summary"]["rows"], 1)
            self.assertNotIn("orbital_pair", runs["new"]["samples"])
            self.assertEqual(len(runs["new"]["samples"]["orbital_pair_summary"]), 1)
            self.assertAlmostEqual(
                runs["new"]["samples"]["orbital_pair_summary"][0]["mae_union_meV_mean"],
                12.5,
                places=12,
            )
            self.assertEqual(runs["old"]["samples"]["orbital_pair_summary"], [])

            archived = module.archived_results_summary()["md"]
            archived_by_run = {item["run_id"]: item for item in archived}
            self.assertTrue(archived_by_run["new"]["diagnostic_outputs"]["orbital_pair_metrics"]["exists"])
            self.assertFalse(archived_by_run["old"]["diagnostic_outputs"]["orbital_pair_metrics"]["exists"])
            results = module.result_summary()
            results_by_run = {item["run_id"]: item for item in results["archived"]["md"]}
            self.assertTrue(results_by_run["new"]["diagnostic_outputs"]["orbital_pair_summary"]["exists"])

    def test_deeph_like_plot_payload_exposes_metric_columns(self) -> None:
        module = self.load_pipeline_ui_module()
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            module.RESULTS_ROOT = root / "results"
            result_dir = module.RESULTS_ROOT / "results_md" / "dataset_4" / "run_deeph"
            metrics_dir = result_dir / "metrics"
            metrics_dir.mkdir(parents=True)
            write_csv(
                metrics_dir / "sparse_metrics.csv",
                [
                    {
                        "sample": "sample_1",
                        "mae_union_meV": "8.0",
                        "rmse_union_meV": "10.0",
                        "mae_ref_meV": "7.0",
                        "rmse_ref_meV": "9.0",
                        "mse_union_eV2": "0.0001",
                        "mse_ref_eV2": "0.000081",
                        "r2_union": "0.97",
                        "r2_ref": "0.98",
                    }
                ],
            )
            write_csv(
                metrics_dir / "spectral_metrics.csv",
                [{"sample": "sample_1", "low_energy_rmse_eV": "0.2"}],
            )
            write_csv(
                metrics_dir / "dos_metrics.csv",
                [
                    {
                        "sample": "sample_1",
                        "dos_mae_500_fermi_window": "0.04",
                        "dos_window_metric_available": "True",
                        "dos_window_points": "500",
                        "dos_window_min_eV": "-6.0",
                        "dos_window_max_eV": "6.0",
                        "dos_window_sigma_eV": "0.1",
                        "dos_window_unavailable_reason": "",
                    }
                ],
            )
            write_csv(
                metrics_dir / "orbital_pair_summary.csv",
                [
                    {
                        "row_species": "H",
                        "col_species": "O",
                        "species_pair": "H-O",
                        "row_orbital_index": "0",
                        "col_orbital_index": "1",
                        "row_orbital_label": "orbital_0",
                        "col_orbital_label": "orbital_1",
                        "n_samples": "1",
                        "n_entries": "2",
                        "mae_union_meV_mean": "12.5",
                        "rmse_union_eV_mean": "0.014",
                        "r2_union_mean": "0.91",
                    }
                ],
            )
            (result_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "pipeline": "md",
                        "method_id": "md",
                        "run_id": "deeph",
                        "result_dir": str(result_dir),
                        "dataset_size": 4,
                        "requested_dataset_size": 4,
                    }
                ),
                encoding="utf-8",
            )

            plots = module.plot_data_summary()
            run = plots["runs"][0]
            self.assertAlmostEqual(run["means"]["sparse"]["mae_union_meV"], 8.0, places=12)
            self.assertAlmostEqual(run["means"]["sparse"]["mse_union_eV2"], 0.0001, places=12)
            self.assertAlmostEqual(run["means"]["sparse"]["r2_union"], 0.97, places=12)
            self.assertAlmostEqual(run["means"]["dos"]["dos_mae_500_fermi_window"], 0.04, places=12)
            self.assertEqual(run["samples"]["dos"][0]["dos_window_points"], 500.0)
            self.assertEqual(run["samples"]["orbital_pair_summary"][0]["species_pair"], "H-O")
            self.assertAlmostEqual(
                run["samples"]["orbital_pair_summary"][0]["mae_union_meV_mean"],
                12.5,
                places=12,
            )

    def test_deeph_like_plot_payload_preserves_missing_dos_reason(self) -> None:
        module = self.load_pipeline_ui_module()
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            module.RESULTS_ROOT = root / "results"
            result_dir = module.RESULTS_ROOT / "results_md" / "dataset_3" / "run_missing_fermi"
            metrics_dir = result_dir / "metrics"
            metrics_dir.mkdir(parents=True)
            write_csv(
                metrics_dir / "sparse_metrics.csv",
                [{"sample": "sample_1", "mae_ref_eV": "0.001"}],
            )
            write_csv(
                metrics_dir / "dos_metrics.csv",
                [
                    {
                        "sample": "sample_1",
                        "dos_window_metric_available": "False",
                        "dos_window_unavailable_reason": "missing_fermi_level",
                    }
                ],
            )
            (result_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "pipeline": "md",
                        "method_id": "md",
                        "run_id": "missing_fermi",
                        "result_dir": str(result_dir),
                        "dataset_size": 3,
                        "requested_dataset_size": 3,
                    }
                ),
                encoding="utf-8",
            )

            plots = module.plot_data_summary()
            run = plots["runs"][0]
            self.assertNotIn("dos_mae_500_fermi_window", run["means"]["dos"])
            self.assertEqual(run["samples"]["dos"][0]["dos_window_metric_available"], "False")
            self.assertEqual(run["samples"]["dos"][0]["dos_window_unavailable_reason"], "missing_fermi_level")

    def test_retraining_runs_get_incremental_training_tags_for_plots(self) -> None:
        module = self.load_pipeline_ui_module()
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            module.RESULTS_ROOT = root / "results"
            dataset_dir = make_reusable_md_dataset(root / "source_dataset")
            training_dir = root / "training"
            training_dir.mkdir(parents=True)
            config = {
                "paths": {
                    "dataset_dir": str(dataset_dir),
                    "training_dir": str(training_dir),
                },
                "checkpoint": {},
                "performance": {},
                "training": {"ui_training_settings": {"max_epochs": 5}},
            }
            runner = module.ExperimentRunner()

            def fake_evaluate(_key, _config, result_dir):
                write_csv(
                    result_dir / "metrics" / "spectral_metrics.csv",
                    [{"sample": "sample_1", "low_energy_rmse_eV": "0.25"}],
                )
                return {"evaluation_time_seconds": 0.0, "returncode": 0}

            runner._evaluate_hamiltonian_metrics = fake_evaluate  # type: ignore[method-assign]

            first = runner._archive_outputs(
                "md",
                3,
                "source",
                root / "workspace_source",
                config,
                0,
                "",
                {},
                dataset_label="Dataset_1000",
                run_mode="full_strict_pipeline",
            )
            second = runner._archive_outputs(
                "md",
                3,
                "retrain",
                root / "workspace_retrain",
                config,
                0,
                "",
                {
                    "dataset_reused": True,
                    "reused_dataset_dir": first["dataset_dir"],
                    "reused_result_dir": first["result_dir"],
                    "reused_run_id": first["run_id"],
                },
                dataset_label="Dataset_1000",
                run_mode="train_test_metrics_plots_only",
            )

            self.assertEqual(first["training_tag"], "dataset_1000_train1")
            self.assertEqual(first["training_index"], 1)
            self.assertEqual(second["training_tag"], "dataset_1000_train2")
            self.assertEqual(second["training_index"], 2)
            self.assertEqual(second["training_base_dataset_label"], "Dataset_1000")
            plots = module.plot_data_summary()
            tags = {run["run_id"]: run["training_tag"] for run in plots["runs"]}
            self.assertEqual(tags["source"], "dataset_1000_train1")
            self.assertEqual(tags["retrain"], "dataset_1000_train2")

    def test_plot_summary_reports_missing_cross_csv_and_metric_gaps(self) -> None:
        module = self.load_pipeline_ui_module()
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            module.RESULTS_ROOT = root / "results"
            experiment_id = "20260101_120000"
            result_dir = module.RESULTS_ROOT / "results_atomdisp" / "dataset_5" / f"run_{experiment_id}"
            metrics_dir = result_dir / "metrics"
            metrics_dir.mkdir(parents=True)
            write_csv(
                metrics_dir / "spectral_metrics.csv",
                [
                    {
                        "sample": "sample_1",
                        "fermi_window_rmse_eV": "nan",
                        "low_energy_rmse_eV": "0.2",
                        "frontier_window_rmse_eV": "0.3",
                    }
                ],
            )
            (result_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "run_id": experiment_id,
                        "result_dir": str(result_dir),
                        "dataset_size": 5,
                        "requested_dataset_size": 5,
                    }
                ),
                encoding="utf-8",
            )
            experiment_dir = module.RESULTS_ROOT / experiment_id
            experiment_dir.mkdir(parents=True)
            module.write_yaml(
                experiment_dir / "experiment_manifest.yaml",
                {
                    "run_mode": "full_strict_pipeline",
                    "selected_methods": ["md", "siesta_fc_cartesian"],
                    "runs": [],
                    "cross_evaluation": {},
                },
            )

            plots = module.plot_data_summary()
            cross_diagnostic = plots["plot_diagnostics"]["cross"][0]
            self.assertEqual(cross_diagnostic["reason"], "cross_evaluation_metrics_missing")
            self.assertEqual(cross_diagnostic["archived_runs"], 1)
            atom_gap = next(
                item
                for item in plots["plot_diagnostics"]["run_metric_gaps"]
                if item["pipeline"] == "atom_displacement"
            )
            fermi_gap = next(
                item
                for item in atom_gap["metrics"]
                if item["metric"] == "fermi_window_rmse_eV"
            )
            self.assertEqual(fermi_gap["n_total"], 1)
            self.assertEqual(fermi_gap["n_finite"], 0)

    def test_plot_summary_discovers_recipe_named_archived_runs(self) -> None:
        module = self.load_pipeline_ui_module()
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            module.RESULTS_ROOT = root / "results"
            result_dir = (
                module.RESULTS_ROOT
                / "results_md"
                / "md_md_table_1_30_md_d1_t300_1_30_30"
                / "run_20260511_131313"
            )
            metrics_dir = result_dir / "metrics"
            metrics_dir.mkdir(parents=True)
            write_csv(
                metrics_dir / "sparse_metrics.csv",
                [{"sample": "27", "relative_frobenius_union": "0.12"}],
            )
            write_csv(
                metrics_dir / "spectral_metrics.csv",
                [{"sample": "27", "low_energy_rmse_eV": "0.34"}],
            )
            (result_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "run_id": "20260511_131313",
                        "result_dir": str(result_dir),
                        "dataset_size": 30,
                        "requested_dataset_size": 30,
                        "dataset_label": "md_md_table_1_30_md_d1_t300_1_30_30",
                    }
                ),
                encoding="utf-8",
            )

            plots = module.plot_data_summary()
            self.assertEqual(len(plots["runs"]), 1)
            self.assertEqual(plots["runs"][0]["pipeline"], "md")
            self.assertEqual(plots["runs"][0]["dataset_label"], "md_md_table_1_30_md_d1_t300_1_30_30")
            self.assertEqual(plots["runs"][0]["means"]["spectral"]["low_energy_rmse_eV"], 0.34)
            archived = module.archived_results_summary()
            self.assertEqual(len(archived["md"]), 1)

    def test_plot_summary_keeps_all_compatible_cross_experiments(self) -> None:
        module = self.load_pipeline_ui_module()
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            module.RESULTS_ROOT = root / "results"
            for experiment_id, size in (("20260101_000001", 5), ("20260101_000002", 9)):
                summary_dir = module.RESULTS_ROOT / experiment_id / "summary"
                summary_dir.mkdir(parents=True)
                write_csv(
                    summary_dir / "cross_evaluation_metrics.csv",
                    [
                        {
                            "experiment_id": experiment_id,
                            "sample_id": "s1",
                            "train_method": "md",
                            "test_set": "test_mixed",
                            "md_dataset_size": str(size),
                            "atom_dataset_size": str(size),
                            "train_dataset_size": str(size),
                            "frontier_window_rmse_eV": "1.0",
                            "low_energy_rmse_eV": "0.8",
                        }
                    ],
                )
                (summary_dir / "recommendation.json").write_text(
                    json.dumps({"status": "ok", "scientific_status": "exploratory"}),
                    encoding="utf-8",
                )
                module.write_yaml(
                    module.RESULTS_ROOT / experiment_id / "experiment_manifest.yaml",
                    {
                        "metric_version": "test_metric",
                        "molecule_system_name": "water",
                        "siesta_settings_hash": "same",
                        "model_config_hash": "same",
                        "test_sets": ["test_mixed"],
                        "selected_methods": ["md", "siesta_fc_cartesian"],
                    },
                )
            plots = module.plot_data_summary()
            self.assertEqual(len(plots["cross_experiments"]), 2)
            self.assertEqual(len(plots["compatible_experiment_groups"]), 1)
            self.assertEqual(plots["compatible_experiment_groups"][0]["rows"], 2)
            self.assertEqual(
                plots["default_plot_selection"]["experiment_ids"],
                ["20260101_000001", "20260101_000002"],
            )
            self.assertEqual(plots["default_plot_selection"]["mode"], "all")
            availability = plots["cross_experiments"][0]["metric_availability"][0]["metrics"]
            self.assertEqual(availability["low_energy_rmse_eV"]["n_finite"], 1)
            self.assertEqual(availability["fermi_window_rmse_eV"]["n_finite"], 0)

    def test_plot_summary_keeps_different_metric_versions_visible(self) -> None:
        module = self.load_pipeline_ui_module()
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            module.RESULTS_ROOT = root / "results"
            for experiment_id, size, metric_version in (
                ("20260101_000001", 25, "v1"),
                ("20260101_000002", 30, "v2"),
            ):
                summary_dir = module.RESULTS_ROOT / experiment_id / "summary"
                summary_dir.mkdir(parents=True)
                write_csv(
                    summary_dir / "cross_evaluation_metrics.csv",
                    [
                        {
                            "experiment_id": experiment_id,
                            "sample_id": "s1",
                            "train_method": "md",
                            "test_set": "test_mixed",
                            "md_dataset_size": str(size),
                            "atom_dataset_size": str(size),
                            "train_dataset_size": str(size),
                            "low_energy_rmse_eV": "0.5",
                        },
                        {
                            "experiment_id": experiment_id,
                            "sample_id": "s1",
                            "train_method": "atom_displacement",
                            "test_set": "test_mixed",
                            "md_dataset_size": str(size),
                            "atom_dataset_size": str(size),
                            "train_dataset_size": str(size),
                            "low_energy_rmse_eV": "",
                        },
                    ],
                )
                (summary_dir / "recommendation.json").write_text(
                    json.dumps({"status": "ok", "scientific_status": "exploratory"}),
                    encoding="utf-8",
                )
                module.write_yaml(
                    module.RESULTS_ROOT / experiment_id / "experiment_manifest.yaml",
                    {
                        "metric_version": metric_version,
                        "molecule_system_name": "water",
                        "siesta_settings_hash": "same",
                        "model_config_hash": "same",
                        "test_sets": ["test_mixed"],
                        "selected_methods": ["md", "siesta_fc_cartesian"],
                    },
                )

            plots = module.plot_data_summary()
            self.assertEqual(len(plots["cross_experiments"]), 2)
            self.assertEqual(len(plots["compatible_experiment_groups"]), 2)
            self.assertEqual(plots["default_plot_selection"]["mode"], "all")
            self.assertEqual(
                plots["default_plot_selection"]["experiment_ids"],
                ["20260101_000001", "20260101_000002"],
            )
            self.assertTrue(plots["visualization_warnings"])
            experiment_30 = next(
                experiment
                for experiment in plots["cross_experiments"]
                if experiment["experiment_id"] == "20260101_000002"
            )
            atom_availability = next(
                item
                for item in experiment_30["metric_availability"]
                if item["train_method"] == "atom_displacement"
            )
            self.assertEqual(atom_availability["train_dataset_size"], 30)
            self.assertEqual(atom_availability["metrics"]["low_energy_rmse_eV"]["n_total"], 1)
            self.assertEqual(atom_availability["metrics"]["low_energy_rmse_eV"]["n_finite"], 0)

    def test_plot_payload_contains_scientific_warnings_for_incomplete_three_method_experiment(self) -> None:
        module = self.load_pipeline_ui_module()
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            module.RESULTS_ROOT = root / "results"
            experiment_id = "20260101_plot_warnings"
            summary_dir = module.RESULTS_ROOT / experiment_id / "summary"
            summary_dir.mkdir(parents=True)
            write_csv(
                summary_dir / "cross_evaluation_metrics.csv",
                [
                    {
                        "experiment_id": experiment_id,
                        "sample_id": "s1",
                        "seed": "1",
                        "train_method": method,
                        "test_set": "test_md",
                        "train_dataset_size": "190",
                        "low_energy_rmse_eV": "0.5",
                    }
                    for method in ("md", "siesta_fc_cartesian", "random_cartesian")
                ],
            )
            (summary_dir / "recommendation.json").write_text(
                json.dumps(
                    {
                        "status": "invalid_incomplete_grid",
                        "scientific_status": "not_scientifically_valid",
                        "reason": "Incomplete 3-method cross-evaluation grid",
                        "missing_cells": ["random_cartesian on test_random_cartesian"],
                        "missing_required_cells": ["random_cartesian on test_random_cartesian"],
                        "missing_primary_metric_cells": ["siesta_fc_cartesian on test_md"],
                        "single_seed_warning": True,
                        "n_seeds": 1,
                        "valid_seed_count": 1,
                        "seed_stability": {
                            "status": "unstable",
                            "unstable_groups": ["low_energy_rmse_eV/test_md"],
                        },
                        "severe_warnings": [
                            "exact_duplicate_geometry leakage detected",
                            "model config mismatch",
                        ],
                    }
                ),
                encoding="utf-8",
            )
            module.write_yaml(
                module.RESULTS_ROOT / experiment_id / "experiment_manifest.yaml",
                {
                    "metric_version": "test_metric",
                    "molecule_system_name": "water",
                    "selected_methods": ["md", "siesta_fc_cartesian", "random_cartesian"],
                    "test_sets": ["test_md", "test_siesta_fc_cartesian", "test_random_cartesian"],
                    "method_provenance": {"md": {}, "siesta_fc_cartesian": {}},
                },
            )

            plots = module.plot_data_summary()
            self.assertEqual(len(plots["cross_experiments"]), 1)
            experiment = plots["cross_experiments"][0]
            self.assertEqual(experiment["plot_scientific_status"], "invalid_incomplete_grid")
            self.assertEqual(
                set(experiment["methods"]["selected"]),
                {"md", "siesta_fc_cartesian", "random_cartesian"},
            )
            self.assertEqual(
                set(experiment["methods"]["observed"]),
                {"md", "siesta_fc_cartesian", "random_cartesian"},
            )
            self.assertEqual(
                set(experiment["test_sets"]["missing"]),
                {"test_siesta_fc_cartesian", "test_random_cartesian"},
            )
            codes = {warning["code"] for warning in experiment["plot_warnings"]}
            self.assertTrue(
                {
                    "incomplete_grid",
                    "missing_primary_metric",
                    "missing_test_set",
                    "single_seed_or_insufficient_seeds",
                    "unstable_winner",
                    "severe_leakage",
                    "fairness_provenance_mismatch",
                    "missing_random_cartesian_provenance",
                    "missing_compute_timings",
                    "plot_scientific_status",
                }.issubset(codes)
            )
            top_level_codes = {warning["code"] for warning in plots["plot_warnings"]}
            self.assertIn("incomplete_grid", top_level_codes)
            self.assertIn("missing_compute_timings", top_level_codes)

    def test_plot_payload_reports_missing_selected_method(self) -> None:
        module = self.load_pipeline_ui_module()
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            module.RESULTS_ROOT = root / "results"
            experiment_id = "20260101_missing_method"
            summary_dir = module.RESULTS_ROOT / experiment_id / "summary"
            summary_dir.mkdir(parents=True)
            write_csv(
                summary_dir / "cross_evaluation_metrics.csv",
                [
                    {
                        "experiment_id": experiment_id,
                        "sample_id": f"s_{method}",
                        "seed": "1",
                        "train_method": method,
                        "test_set": "test_md",
                        "train_dataset_size": "190",
                        "low_energy_rmse_eV": "0.5",
                    }
                    for method in ("md", "siesta_fc_cartesian")
                ],
            )
            (summary_dir / "recommendation.json").write_text(
                json.dumps({"status": "ok", "scientific_status": "exploratory_only"}),
                encoding="utf-8",
            )
            module.write_yaml(
                module.RESULTS_ROOT / experiment_id / "experiment_manifest.yaml",
                {
                    "metric_version": "test_metric",
                    "molecule_system_name": "water",
                    "selected_methods": ["md", "siesta_fc_cartesian", "random_cartesian"],
                    "test_sets": ["test_md"],
                    "method_provenance": {
                        "md": {},
                        "siesta_fc_cartesian": {},
                        "random_cartesian": {},
                    },
                },
            )

            plots = module.plot_data_summary()
            experiment = plots["cross_experiments"][0]
            codes = {warning["code"] for warning in experiment["plot_warnings"]}
            self.assertIn("missing_method", codes)
            self.assertEqual(experiment["methods"]["missing"], ["random_cartesian"])

    def test_ui_renders_plot_warning_banner(self) -> None:
        index_html = (REPO_ROOT / "Comparison" / "ui" / "index.html").read_text(encoding="utf-8")
        app_js = (REPO_ROOT / "Comparison" / "ui" / "app.js").read_text(encoding="utf-8")
        self.assertIn('id="plot-warnings"', index_html)
        self.assertIn("function renderPlotWarnings", app_js)
        self.assertIn("plot_scientific_status", app_js)
        self.assertIn("plot_warnings", app_js)
        self.assertIn("invalid_incomplete_grid", app_js)

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

    def test_random_cartesian_composite_blocks_preserve_manifest_metadata(self) -> None:
        module = self.load_random_cartesian_module()
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            self.configure_random_cartesian_module(module, root, n_structures=5, seed=13)
            module.PIPELINE_CONFIG["dataset_recipe"] = {
                "recipe_id": "rc_dataset_1",
                "recipe_label": "RC composite",
                "generation_parameters_json": json.dumps({"blocks": ["rc_small", "rc_large"]}),
            }
            module.PIPELINE_CONFIG["structure"]["random_cartesian"]["blocks"] = [
                {
                    "block_id": "rc_small",
                    "label": "2 structures @ 0.02 Ang",
                    "n_structures": 2,
                    "max_displacement": "0.02 Ang",
                    "seed": 21,
                },
                {
                    "block_id": "rc_large",
                    "label": "3 structures @ 0.05 Ang",
                    "n_structures": 3,
                    "distribution": "uniform",
                    "max_displacement": "0.05 Ang",
                    "seed": 22,
                },
            ]
            manifest = module.generate_dataset(module.PIPELINE_CONFIG)
            dataset_root = root / "dataset" / "RandomCartesian_steps"
            self.assertEqual(manifest["requested_structures"], 5)
            self.assertEqual(manifest["generated_structures"], 5)
            self.assertEqual(len(manifest["blocks"]), 2)
            self.assertEqual([sample["block_id"] for sample in manifest["samples"]], ["rc_small", "rc_small", "rc_large", "rc_large", "rc_large"])
            metadata = json.loads((dataset_root / "sample_000003" / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["block_id"], "rc_large")
            self.assertEqual(metadata["sample_index_within_block"], 0)
            self.assertEqual(metadata["block_n_structures"], 3)
            self.assertEqual(metadata["uniform_range_ang"], 0.05)
            run_fdf = (dataset_root / "sample_000003" / "RUN.fdf").read_text(encoding="utf-8")
            self.assertNotIn("MD.TypeOfRun", run_fdf)
            self.assertNotIn("FC.Displacement", run_fdf)

    def test_random_cartesian_split_manifests_keep_families_isolated(self) -> None:
        module = self.load_random_cartesian_module()
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            self.configure_random_cartesian_module(module, root, n_structures=5, seed=13)
            module.PIPELINE_CONFIG["structure"]["random_cartesian"]["blocks"] = [
                {"block_id": "rc_small", "n_structures": 2, "max_displacement": "0.02 Ang", "seed": 21},
                {"block_id": "rc_large", "n_structures": 3, "max_displacement": "0.05 Ang", "seed": 22},
            ]
            manifest = module.generate_dataset(module.PIPELINE_CONFIG)
            dataset_root = root / "dataset" / "RandomCartesian_steps"
            split_rows = {
                split: json.loads((dataset_root / f"split_manifest_{split}.json").read_text(encoding="utf-8"))
                for split in ("train", "validation", "test")
            }
            group_to_split: dict[str, str] = {}
            for split, payload in split_rows.items():
                self.assertEqual(payload["split_strategy"], "grouped_family_round_robin")
                self.assertTrue(payload["group_aware"])
                for sample in payload["samples"]:
                    group_id = sample["split_group_id"]
                    previous = group_to_split.setdefault(group_id, split)
                    self.assertEqual(previous, split)

            self.assertEqual(len(group_to_split), 2)
            self.assertEqual(manifest["split_strategy"], "grouped_family_round_robin")
            self.assertEqual(manifest["split_summary"]["counts"], {"train": 2, "validation": 3, "test": 0})
            self.assertEqual(manifest["split_summary"]["split_group_keys_used"], ["split_group_id"])
            self.assertTrue((dataset_root / "split_manifest_summary.json").exists())
            self.assertTrue((dataset_root / "dataset_manifest.json").exists())
            self.assertTrue((dataset_root / "artifact_hashes.json").exists())

    def test_random_cartesian_grouped_split_is_deterministic_for_distinct_families(self) -> None:
        module = self.load_random_cartesian_module()
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            samples = [
                {"sample_id": "sample_a1", "split_group_id": "family_a"},
                {"sample_id": "sample_a2", "split_group_id": "family_a"},
                {"sample_id": "sample_b1", "split_group_id": "family_b"},
                {"sample_id": "sample_c1", "split_group_id": "family_c"},
            ]
            first_root = root / "first"
            second_root = root / "second"
            first_root.mkdir()
            second_root.mkdir()
            first = module.write_split_manifests(first_root, samples)
            second = module.write_split_manifests(second_root, list(reversed(list(reversed(samples)))))
            self.assertEqual(first, second)
            self.assertEqual(first["counts"], {"train": 2, "validation": 1, "test": 1})
            self.assertEqual(
                [(group["group_id"], group["split"]) for group in first["groups"]],
                [("family_a", "train"), ("family_b", "validation"), ("family_c", "test")],
            )

    def test_random_cartesian_missing_group_metadata_uses_explicit_fallback_warning(self) -> None:
        module = self.load_random_cartesian_module()
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            samples = [
                {
                    "sample_id": "sample_1",
                    "base_geometry_hash": "base",
                    "distribution": "gaussian",
                    "sigma_ang": 0.03,
                    "seed_family": 123,
                    "move_atoms": "all",
                    "species_filter": "[]",
                    "block_id": "block_a",
                },
                {
                    "sample_id": "sample_2",
                    "base_geometry_hash": "base",
                    "distribution": "gaussian",
                    "sigma_ang": 0.03,
                    "seed_family": 123,
                    "move_atoms": "all",
                    "species_filter": "[]",
                    "block_id": "block_a",
                },
                {"sample_id": "sample_3"},
            ]
            split_root = root / "splits"
            split_root.mkdir()
            summary = module.write_split_manifests(split_root, samples)
            self.assertEqual(summary["scientific_status"], "grouped_family_splits_with_fallback")
            self.assertIn("derived_random_cartesian_family", summary["split_group_keys_used"])
            self.assertIn("sample_id_fallback", summary["split_group_keys_used"])
            self.assertTrue(any("missing split_group_id" in warning for warning in summary["warnings"]))
            self.assertTrue(any("weak fallback group" in warning for warning in summary["warnings"]))
            train_payload = json.loads((split_root / "split_manifest_train.json").read_text(encoding="utf-8"))
            derived_ids = {sample["sample_id"] for sample in train_payload["samples"]}
            self.assertEqual(derived_ids, {"sample_1", "sample_2"})

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

    def test_random_cartesian_nested_atom_only_preserves_legacy_geometry(self) -> None:
        module = self.load_random_cartesian_module()
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            legacy_root = root / "legacy"
            nested_root = root / "nested"
            legacy_root.mkdir()
            nested_root.mkdir()
            self.configure_random_cartesian_module(module, legacy_root, n_structures=1, seed=17)
            module.PIPELINE_CONFIG["structure"]["random_cartesian"].update(
                {
                    "sigma_ang": 0.04,
                    "remove_center_of_mass_translation": False,
                }
            )
            module.generate_dataset(module.PIPELINE_CONFIG)
            legacy_run = (legacy_root / "dataset" / "RandomCartesian_steps" / "sample_000001" / "RUN.fdf").read_text(encoding="utf-8")

            self.configure_random_cartesian_module(module, nested_root, n_structures=1, seed=17)
            module.PIPELINE_CONFIG["structure"]["random_cartesian"].update(
                {
                    "components": {
                        "atom_displacement": {
                            "enabled": True,
                            "distribution": "gaussian",
                            "sigma_ang": 0.04,
                            "remove_center_of_mass_translation": False,
                        },
                        "bond_displacement": {"enabled": False},
                        "angle_displacement": {"enabled": False},
                    }
                }
            )
            module.generate_dataset(module.PIPELINE_CONFIG)
            nested_run = (nested_root / "dataset" / "RandomCartesian_steps" / "sample_000001" / "RUN.fdf").read_text(encoding="utf-8")
            metadata = json.loads(
                (nested_root / "dataset" / "RandomCartesian_steps" / "sample_000001" / "metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(legacy_run, nested_run)
            self.assertEqual(metadata["enabled_components"], ["atom_displacement"])

    def test_random_cartesian_bond_only_generation_records_water_metrics(self) -> None:
        module = self.load_random_cartesian_module()
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            self.configure_random_cartesian_module(module, root, n_structures=1, seed=3)
            module.PIPELINE_CONFIG["structure"]["random_cartesian"].update(
                {
                    "components": {
                        "atom_displacement": {"enabled": False},
                        "bond_displacement": {
                            "enabled": True,
                            "distribution": "uniform",
                            "min_delta_ang": 0.1,
                            "max_delta_ang": 0.1,
                            "min_bond_ang": 0.7,
                            "max_bond_ang": 1.3,
                        },
                        "angle_displacement": {"enabled": False},
                    },
                    "validation": {"min_distance_ang": 0.1, "max_attempts_per_structure": 3},
                }
            )
            manifest = module.generate_dataset(module.PIPELINE_CONFIG)
            metadata = json.loads(
                (root / "dataset" / "RandomCartesian_steps" / "sample_000001" / "metadata.json").read_text(encoding="utf-8")
            )
            metrics = metadata["final_geometry_metrics"]
            self.assertEqual(metadata["enabled_components"], ["bond_displacement"])
            self.assertAlmostEqual(metrics["oh_1_ang"], 1.1, places=12)
            self.assertAlmostEqual(metrics["oh_2_ang"], 1.1, places=12)
            self.assertIn(module.SCIENTIFIC_WARNING, manifest["warnings"])

    def test_random_cartesian_angle_only_generation_records_angle_delta(self) -> None:
        module = self.load_random_cartesian_module()
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            self.configure_random_cartesian_module(module, root, n_structures=1, seed=3)
            module.PIPELINE_CONFIG["structure"]["random_cartesian"].update(
                {
                    "components": {
                        "atom_displacement": {"enabled": False},
                        "bond_displacement": {"enabled": False},
                        "angle_displacement": {
                            "enabled": True,
                            "distribution": "uniform",
                            "min_delta_deg": -10.0,
                            "max_delta_deg": -10.0,
                            "min_angle_deg": 150.0,
                            "max_angle_deg": 180.0,
                        },
                    },
                    "validation": {"min_distance_ang": 0.1, "max_attempts_per_structure": 3},
                }
            )
            module.generate_dataset(module.PIPELINE_CONFIG)
            metadata = json.loads(
                (root / "dataset" / "RandomCartesian_steps" / "sample_000001" / "metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["enabled_components"], ["angle_displacement"])
            self.assertAlmostEqual(metadata["final_geometry_metrics"]["hoh_angle_deg"], 170.0, places=10)
            self.assertAlmostEqual(metadata["angle_displacement_deg"]["delta_deg"], -10.0, places=12)

    def test_random_cartesian_all_components_manifest_and_splits(self) -> None:
        module = self.load_random_cartesian_module()
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            self.configure_random_cartesian_module(module, root, n_structures=3, seed=9)
            module.PIPELINE_CONFIG["structure"]["random_cartesian"].update(
                {
                    "components": {
                        "atom_displacement": {
                            "enabled": True,
                            "distribution": "gaussian",
                            "sigma_ang": 0.0,
                        },
                        "bond_displacement": {
                            "enabled": True,
                            "distribution": "uniform",
                            "min_delta_ang": 0.05,
                            "max_delta_ang": 0.05,
                            "min_bond_ang": 0.7,
                            "max_bond_ang": 1.3,
                        },
                        "angle_displacement": {
                            "enabled": True,
                            "distribution": "uniform",
                            "min_delta_deg": -5.0,
                            "max_delta_deg": -5.0,
                            "min_angle_deg": 150.0,
                            "max_angle_deg": 180.0,
                        },
                    },
                    "validation": {"min_distance_ang": 0.1, "max_attempts_per_structure": 3},
                }
            )
            manifest = module.generate_dataset(module.PIPELINE_CONFIG)
            dataset_root = root / "dataset" / "RandomCartesian_steps"
            metadata = json.loads((dataset_root / "sample_000001" / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(
                metadata["enabled_components"],
                ["atom_displacement", "bond_displacement", "angle_displacement"],
            )
            self.assertEqual(manifest["component_config"]["bond_displacement"]["min_delta_ang"], 0.05)
            self.assertEqual(manifest["acceptance_ratio"], 1.0)
            self.assertIn("deterministic_hashes", manifest)
            self.assertIn("config_hash", manifest["deterministic_hashes"])
            self.assertTrue((dataset_root / "split_manifest_train.json").exists())
            self.assertTrue((dataset_root / "split_manifest_validation.json").exists())
            self.assertTrue((dataset_root / "split_manifest_test.json").exists())

    def test_random_cartesian_rejection_counts_capture_explicit_reason(self) -> None:
        module = self.load_random_cartesian_module()
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            self.configure_random_cartesian_module(module, root, n_structures=1, seed=3)
            module.PIPELINE_CONFIG["structure"]["random_cartesian"].update(
                {
                    "components": {
                        "atom_displacement": {"enabled": False},
                        "bond_displacement": {
                            "enabled": True,
                            "distribution": "uniform",
                            "min_delta_ang": 0.0,
                            "max_delta_ang": 0.0,
                            "min_bond_ang": 0.7,
                            "max_bond_ang": 1.3,
                        },
                        "angle_displacement": {"enabled": False},
                    },
                    "validation": {"min_distance_ang": 0.1, "max_attempts_per_structure": 3},
                }
            )
            original_sample = module.sample_bounded_scalar
            calls = {"count": 0}

            def forced_sample(rng, component, *, sigma_key, min_key, max_key):
                calls["count"] += 1
                return -0.4 if calls["count"] == 1 else 0.0

            module.sample_bounded_scalar = forced_sample
            try:
                manifest = module.generate_dataset(module.PIPELINE_CONFIG)
            finally:
                module.sample_bounded_scalar = original_sample
            metadata = json.loads(
                (root / "dataset" / "RandomCartesian_steps" / "sample_000001" / "metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["rejection_counts_by_reason"], {"oh_bond_out_of_range": 1})
            self.assertEqual(metadata["accepted_attempt"], 2)
            self.assertEqual(metadata["last_rejection_reason"], "oh_bond_out_of_range")

    def test_random_cartesian_block_nested_override_and_legacy_block_keys_work(self) -> None:
        module = self.load_random_cartesian_module()
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            self.configure_random_cartesian_module(module, root, n_structures=2, seed=3)
            module.PIPELINE_CONFIG["structure"]["random_cartesian"].update(
                {
                    "blocks": [
                        {
                            "block_id": "bond_block",
                            "n_structures": 1,
                            "components": {
                                "atom_displacement": {"enabled": False},
                                "bond_displacement": {
                                    "enabled": True,
                                    "distribution": "uniform",
                                    "min_delta_ang": 0.02,
                                    "max_delta_ang": 0.02,
                                },
                            },
                            "validation": {"min_distance_ang": 0.1},
                        },
                        {
                            "block_id": "legacy_atom_block",
                            "n_structures": 1,
                            "max_displacement": "0.01 Ang",
                            "remove_center_of_mass_translation": False,
                            "components": {
                                "atom_displacement": {"enabled": True},
                                "bond_displacement": {"enabled": False},
                                "angle_displacement": {"enabled": False},
                            },
                        },
                    ]
                }
            )
            manifest = module.generate_dataset(module.PIPELINE_CONFIG)
            dataset_root = root / "dataset" / "RandomCartesian_steps"
            first = json.loads((dataset_root / "sample_000001" / "metadata.json").read_text(encoding="utf-8"))
            second = json.loads((dataset_root / "sample_000002" / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(first["enabled_components"], ["bond_displacement"])
            self.assertEqual(second["enabled_components"], ["atom_displacement"])
            self.assertEqual(manifest["blocks"][1]["components"]["atom_displacement"]["sigma_ang"], 0.01)

    def test_random_cartesian_cli_parser_accepts_component_flags(self) -> None:
        module = self.load_random_cartesian_module()
        args = module.build_argument_parser().parse_args(
            [
                "--n-structures",
                "3",
                "--disable-atom-displacement",
                "--enable-bond-displacement",
                "--enable-angle-displacement",
                "--bond-uniform-range-ang",
                "0.03",
                "--angle-sigma-deg",
                "2.5",
                "--max-rmsd-from-reference-ang",
                "0.2",
            ]
        )
        self.assertEqual(args.n_structures, 3)
        self.assertFalse(args.atom_displacement_enabled)
        self.assertTrue(args.bond_displacement_enabled)
        self.assertTrue(args.angle_displacement_enabled)
        self.assertEqual(args.bond_uniform_range_ang, 0.03)
        self.assertEqual(args.angle_sigma_deg, 2.5)
        self.assertEqual(args.max_rmsd_from_reference_ang, 0.2)

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

    def test_atomdisp_single_points_default_config_is_strict(self) -> None:
        config = (REPO_ROOT / "AtomDisplacement" / "pipeline_config.yaml").read_text(encoding="utf-8")
        self.assertIn("allow_unvalidated_matrices: false", config)

    def test_strict_atomdisp_validation_rejects_stale_matrix(self) -> None:
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
            sample = make_sample(Path(tmp), "stale_matrix")
            matrix = sample / "siesta.TSHS"
            run_fdf = sample / "RUN.fdf"
            run_out = sample / "RUN.out"
            os.utime(matrix, (1000, 1000))
            os.utime(run_fdf, (2000, 2000))
            os.utime(run_out, (3000, 3000))

            validation = module.validate_sample_dir(sample)
            self.assertFalse(validation["valid"])
            self.assertIn("stale_matrix", validation["validation_reason"])

    def test_atomdisp_unsafe_mode_marks_unvalidated_matrix_reuse(self) -> None:
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
            sample = make_sample(root, "unsafe_missing_output")
            (sample / "RUN.out").unlink()

            strict = module.validate_sample_dir(sample)
            self.assertFalse(strict["valid"])
            self.assertIn("missing_output", strict["validation_reason"])

            unsafe = module.validate_sample_dir(sample, allow_unvalidated_matrices=True)
            self.assertTrue(unsafe["valid"])
            self.assertTrue(unsafe["unsafe_unvalidated_matrices"])
            self.assertIn("missing_output", unsafe["unsafe_validation_reasons"])
            self.assertIn("UNSAFE_UNVALIDATED_MATRIX_REFERENCE", unsafe["severe_warnings"])
            self.assertIn("unsafe_unvalidated_matrices", unsafe["validation_reason"])

            summary = module.write_validation_outputs(root / "validation", [unsafe])
            self.assertEqual(summary["invalid_samples"], 0)
            self.assertEqual(summary["unsafe_unvalidated_samples"], 1)
            self.assertTrue(summary["unsafe_mode"])
            self.assertTrue(any("UNSAFE_UNVALIDATED_MATRIX_REFERENCE" in item for item in summary["severe_warnings"]))
            with (root / "validation" / "valid_samples.csv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["unsafe_unvalidated_matrices"], "True")
            self.assertIn("missing_output", rows[0]["unsafe_validation_reasons"])

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

    def test_strict_reference_selection_rejects_ambiguous_references_and_predictions(self) -> None:
        sys.path.insert(0, str(REPO_ROOT / "Comparison" / "scripts"))
        module = self.load_module_from_path(
            "reference_selection_test",
            REPO_ROOT / "Comparison" / "scripts" / "reference_selection.py",
        )
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            one = root / "one"
            one.mkdir()
            (one / "siesta.TSHS").write_bytes(b"tshs")
            (one / "siesta.HSX").write_bytes(b"hsx")
            selection = module.choose_reference_matrix(one)
            self.assertTrue(selection.ok)
            self.assertEqual(selection.path.name, "siesta.TSHS")

            many_tshs = root / "many_tshs"
            many_tshs.mkdir()
            (many_tshs / "a.TSHS").write_bytes(b"a")
            (many_tshs / "b.TSHS").write_bytes(b"b")
            self.assertFalse(module.choose_reference_matrix(many_tshs).ok)
            self.assertTrue(module.choose_reference_matrix(many_tshs).ambiguous)

            many_hsx = root / "many_hsx"
            many_hsx.mkdir()
            (many_hsx / "a.HSX").write_bytes(b"a")
            (many_hsx / "b.HSX").write_bytes(b"b")
            self.assertFalse(module.choose_reference_matrix(many_hsx).ok)
            self.assertTrue(module.choose_reference_matrix(many_hsx).ambiguous)

            prediction_only = root / "prediction_only"
            prediction_only.mkdir()
            (prediction_only / "ML_prediction.HSX").write_bytes(b"prediction")
            self.assertFalse(module.choose_reference_matrix(prediction_only).ok)

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
                    ("md", "test_siesta_fc_cartesian"),
                    ("md", "test_mixed"),
                    ("siesta_fc_cartesian", "test_md"),
                    ("siesta_fc_cartesian", "test_siesta_fc_cartesian"),
                    ("siesta_fc_cartesian", "test_mixed"),
                },
            )

    def test_cross_aggregation_preserves_p0_metric_columns_and_old_runs(self) -> None:
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            cross_root = root / "cross"
            new_dir = cross_root / "md__on__test_md"
            old_dir = cross_root / "siesta_fc_cartesian__on__test_md"
            for result_dir in (new_dir, old_dir):
                (result_dir / "metrics").mkdir(parents=True)

            write_csv(
                new_dir / "metrics" / "sparse_metrics.csv",
                [
                    {
                        "sample": "sample_1",
                        "mae_ref_eV": "0.001",
                        "mse_ref_eV2": "0.000004",
                        "r2_ref": "0.98",
                        "mae_ref_meV": "1.0",
                        "rmse_ref_meV": "2.0",
                        "matrix_metric_target_space": "raw_global_hamiltonian",
                    }
                ],
            )
            write_csv(
                new_dir / "metrics" / "spectral_metrics.csv",
                [{"sample": "sample_1", "low_energy_rmse_eV": "0.2"}],
            )
            write_csv(
                new_dir / "metrics" / "dos_metrics.csv",
                [
                    {
                        "sample": "sample_1",
                        "dos_wasserstein_eV": "0.3",
                        "dos_mae_500_fermi_window": "0.04",
                        "dos_window_metric_available": "True",
                        "dos_window_unavailable_reason": "",
                    }
                ],
            )
            write_csv(
                new_dir / "metrics" / "orbital_pair_metrics.csv",
                [
                    {
                        "sample": "sample_1",
                        "row_species": "H",
                        "col_species": "H",
                        "species_pair": "H-H",
                        "row_orbital_index": "0",
                        "col_orbital_index": "0",
                        "mae_union_meV": "1.0",
                    }
                ],
            )
            write_csv(
                new_dir / "metrics" / "orbital_pair_summary.csv",
                [
                    {
                        "row_species": "H",
                        "col_species": "H",
                        "species_pair": "H-H",
                        "row_orbital_index": "0",
                        "col_orbital_index": "0",
                        "mae_union_meV_mean": "1.0",
                    }
                ],
            )
            write_csv(
                old_dir / "metrics" / "sparse_metrics.csv",
                [{"sample": "sample_1", "mae_ref_eV": "0.002"}],
            )
            write_csv(
                old_dir / "metrics" / "spectral_metrics.csv",
                [{"sample": "sample_1", "low_energy_rmse_eV": "0.4"}],
            )
            for result_dir, method in ((new_dir, "md"), (old_dir, "siesta_fc_cartesian")):
                (result_dir / "cross_evaluation_manifest.json").write_text(
                    json.dumps(
                        {
                            "train_method": method,
                            "test_set": "test_md",
                            "dataset_size": 3,
                            "seed": 42,
                            "model_checkpoint": f"{method}.ckpt",
                        }
                    ),
                    encoding="utf-8",
                )

            result = run_script(
                "Comparison/scripts/aggregate_cross_metrics.py",
                "--experiment-id",
                "exp_p0",
                "--cross-root",
                str(cross_root),
                "--output-dir",
                str(root / "summary"),
                "--primary-metric",
                "low_energy_rmse_eV",
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            with (root / "summary" / "cross_evaluation_metrics.csv").open(encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
                fieldnames = list(reader.fieldnames or [])
            self.assertIn("mse_ref_eV2", fieldnames)
            self.assertIn("r2_ref", fieldnames)
            self.assertIn("mae_ref_meV", fieldnames)
            self.assertIn("dos_mae_500_fermi_window", fieldnames)
            self.assertNotIn("orbital_pair_metrics", fieldnames)
            self.assertNotIn("mae_union_meV_mean", fieldnames)
            new_row = next(row for row in rows if row["train_method"] == "md")
            old_row = next(row for row in rows if row["train_method"] == "siesta_fc_cartesian")
            self.assertEqual(new_row["mse_ref_eV2"], "4e-06")
            self.assertEqual(new_row["r2_ref"], "0.98")
            self.assertEqual(new_row["dos_mae_500_fermi_window"], "0.04")
            self.assertEqual(old_row["mse_ref_eV2"], "")
            self.assertEqual(old_row["dos_mae_500_fermi_window"], "")

    def test_cross_aggregation_rejects_duplicate_sample_ids_and_propagates_evaluator_errors(self) -> None:
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            cross_root = root / "cross"
            result_dir = cross_root / "md__on__test_md"
            (result_dir / "metrics").mkdir(parents=True)
            write_csv(
                result_dir / "metrics" / "sparse_metrics.csv",
                [
                    {"sample": "sample_1", "relative_frobenius_union": "1.0"},
                    {"sample": "sample_1", "relative_frobenius_union": "2.0"},
                ],
            )
            (result_dir / "cross_evaluation_manifest.json").write_text(
                json.dumps({"train_method": "md", "test_set": "test_md"}),
                encoding="utf-8",
            )
            result = run_script(
                "Comparison/scripts/aggregate_cross_metrics.py",
                "--experiment-id",
                "exp_dup",
                "--cross-root",
                str(cross_root),
                "--output-dir",
                str(root / "summary_dup"),
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("duplicate sample", result.stderr + result.stdout)

            shutil.rmtree(root / "cross")
            result_dir = cross_root / "md__on__test_md"
            (result_dir / "metrics").mkdir(parents=True)
            write_csv(
                result_dir / "metrics" / "sparse_metrics.csv",
                [{"sample": "sample_1", "relative_frobenius_union": "1.0"}],
            )
            write_csv(
                result_dir / "metrics" / "spectral_metrics.csv",
                [{"sample": "sample_1", "fermi_window_rmse_eV": "0.3"}],
            )
            (result_dir / "metrics" / "manifest.json").write_text(
                json.dumps(
                    {
                        "fatal_errors": [{"sample": "sample_1", "kind": "matrix_shape_mismatch", "error": "bad shape"}],
                        "warnings": [{"sample": "sample_1", "kind": "missing_fermi_level", "error": "no fermi"}],
                    }
                ),
                encoding="utf-8",
            )
            (result_dir / "cross_evaluation_manifest.json").write_text(
                json.dumps({"train_method": "md", "test_set": "test_md"}),
                encoding="utf-8",
            )
            result = run_script(
                "Comparison/scripts/aggregate_cross_metrics.py",
                "--experiment-id",
                "exp_warn",
                "--cross-root",
                str(cross_root),
                "--output-dir",
                str(root / "summary_warn"),
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            with (root / "summary_warn" / "cross_evaluation_metrics.csv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["severe_warning_status"], "severe")
            self.assertIn("matrix_shape_mismatch", rows[0]["severe_warnings"])
            self.assertIn("missing_fermi_level", rows[0]["severe_warnings"])

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

    def test_cross_aggregation_completeness_complete_three_method_grid_passes(self) -> None:
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            methods = ("md", "siesta_fc_cartesian", "random_cartesian")
            test_sets = ("test_md", "test_siesta_fc_cartesian", "test_random_cartesian")
            cross_root = root / "cross"
            for method in methods:
                for test_set in test_sets:
                    self.write_cross_metric_cell(cross_root, method, test_set)
            expected_grid = root / "summary" / "cross_evaluation_expected_grid.json"
            self.write_expected_cross_grid(expected_grid, methods, test_sets)
            result = run_script(
                "Comparison/scripts/aggregate_cross_metrics.py",
                "--experiment-id",
                "exp_complete",
                "--cross-root",
                str(cross_root),
                "--output-dir",
                str(root / "summary"),
                "--expected-grid",
                str(expected_grid),
                "--primary-metric",
                "low_energy_rmse_eV",
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            report = json.loads((root / "summary" / "cross_evaluation_completeness.json").read_text(encoding="utf-8"))
            self.assertTrue(report["complete"])
            self.assertEqual(report["scientific_status"], "valid_grid")
            self.assertEqual(report["expected_cell_count"], 9)
            self.assertEqual(report["actual_cell_count"], 9)
            self.assertEqual(report["missing_cells"], [])
            self.assertEqual(report["extra_unexpected_cells"], [])
            self.assertEqual(report["missing_primary_metric_cells"], [])

    def test_cross_aggregation_completeness_missing_random_on_test_md_fails(self) -> None:
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            methods = ("md", "siesta_fc_cartesian", "random_cartesian")
            test_sets = ("test_md", "test_siesta_fc_cartesian", "test_random_cartesian")
            cross_root = root / "cross"
            for method in methods:
                for test_set in test_sets:
                    if method == "random_cartesian" and test_set == "test_md":
                        continue
                    self.write_cross_metric_cell(cross_root, method, test_set)
            expected_grid = root / "summary" / "cross_evaluation_expected_grid.json"
            self.write_expected_cross_grid(expected_grid, methods, test_sets)
            result = run_script(
                "Comparison/scripts/aggregate_cross_metrics.py",
                "--experiment-id",
                "exp_missing_rc_md",
                "--cross-root",
                str(cross_root),
                "--output-dir",
                str(root / "summary"),
                "--expected-grid",
                str(expected_grid),
                "--primary-metric",
                "low_energy_rmse_eV",
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            report = json.loads((root / "summary" / "cross_evaluation_completeness.json").read_text(encoding="utf-8"))
            self.assertFalse(report["complete"])
            self.assertEqual(report["scientific_status"], "invalid_incomplete_grid")
            self.assertIn("random_cartesian on test_md", report["missing_cells"])
            with (root / "summary" / "missing_cross_evaluation_cells.csv").open(encoding="utf-8") as handle:
                missing_rows = list(csv.DictReader(handle))
            self.assertIn(
                {"cell_id": "random_cartesian on test_md", "issue": "missing_cell", "test_set": "test_md", "train_method": "random_cartesian"},
                missing_rows,
            )

    def test_cross_aggregation_completeness_missing_md_on_test_random_fails(self) -> None:
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            methods = ("md", "siesta_fc_cartesian", "random_cartesian")
            test_sets = ("test_md", "test_siesta_fc_cartesian", "test_random_cartesian")
            cross_root = root / "cross"
            for method in methods:
                for test_set in test_sets:
                    if method == "md" and test_set == "test_random_cartesian":
                        continue
                    self.write_cross_metric_cell(cross_root, method, test_set)
            expected_grid = root / "summary" / "cross_evaluation_expected_grid.json"
            self.write_expected_cross_grid(expected_grid, methods, test_sets)
            result = run_script(
                "Comparison/scripts/aggregate_cross_metrics.py",
                "--experiment-id",
                "exp_missing_md_rc",
                "--cross-root",
                str(cross_root),
                "--output-dir",
                str(root / "summary"),
                "--expected-grid",
                str(expected_grid),
                "--primary-metric",
                "low_energy_rmse_eV",
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            report = json.loads((root / "summary" / "cross_evaluation_completeness.json").read_text(encoding="utf-8"))
            self.assertFalse(report["complete"])
            self.assertEqual(report["scientific_status"], "invalid_incomplete_grid")
            self.assertIn("md on test_random_cartesian", report["missing_cells"])

    def test_cross_aggregation_completeness_missing_primary_metric_fails(self) -> None:
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            methods = ("md", "siesta_fc_cartesian", "random_cartesian")
            test_sets = ("test_md", "test_siesta_fc_cartesian", "test_random_cartesian")
            cross_root = root / "cross"
            for method in methods:
                for test_set in test_sets:
                    primary_value = None if (method == "md" and test_set == "test_random_cartesian") else "0.2"
                    self.write_cross_metric_cell(cross_root, method, test_set, primary_value=primary_value)
            expected_grid = root / "summary" / "cross_evaluation_expected_grid.json"
            self.write_expected_cross_grid(expected_grid, methods, test_sets)
            result = run_script(
                "Comparison/scripts/aggregate_cross_metrics.py",
                "--experiment-id",
                "exp_missing_primary",
                "--cross-root",
                str(cross_root),
                "--output-dir",
                str(root / "summary"),
                "--expected-grid",
                str(expected_grid),
                "--primary-metric",
                "low_energy_rmse_eV",
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            report = json.loads((root / "summary" / "cross_evaluation_completeness.json").read_text(encoding="utf-8"))
            self.assertFalse(report["complete"])
            self.assertEqual(report["missing_cells"], [])
            self.assertEqual(report["scientific_status"], "invalid_incomplete_grid")
            self.assertIn("md on test_random_cartesian", report["missing_primary_metric_cells"])

    def test_cross_aggregation_completeness_missing_retraining_context_fails(self) -> None:
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            cross_root = root / "cross"
            self.write_cross_metric_cell(
                cross_root,
                "md",
                "test_md",
                pair_id="cross_train1",
                result_name="cross_train1__md__on__test_md",
            )
            expected_grid = root / "summary" / "cross_evaluation_expected_grid.json"
            self.write_expected_cross_grid(
                expected_grid,
                ("md",),
                ("test_md",),
                expected_context_cells=[
                    {
                        "pair_id": "cross_train1",
                        "train_method": "md",
                        "test_set": "test_md",
                        "cell_id": "cross_train1 :: md on test_md",
                    },
                    {
                        "pair_id": "cross_train2",
                        "train_method": "md",
                        "test_set": "test_md",
                        "cell_id": "cross_train2 :: md on test_md",
                    },
                ],
            )
            result = run_script(
                "Comparison/scripts/aggregate_cross_metrics.py",
                "--experiment-id",
                "exp_missing_context",
                "--cross-root",
                str(cross_root),
                "--output-dir",
                str(root / "summary"),
                "--expected-grid",
                str(expected_grid),
                "--primary-metric",
                "low_energy_rmse_eV",
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            report = json.loads((root / "summary" / "cross_evaluation_completeness.json").read_text(encoding="utf-8"))
            self.assertFalse(report["complete"])
            self.assertEqual(report["missing_cells"], [])
            self.assertIn("cross_train2 :: md on test_md", report["missing_context_cells"])
            with (root / "summary" / "missing_cross_evaluation_cells.csv").open(encoding="utf-8") as handle:
                missing_rows = list(csv.DictReader(handle))
            self.assertIn(
                {
                    "cell_id": "cross_train2 :: md on test_md",
                    "issue": "missing_context_cell",
                    "test_set": "test_md",
                    "train_method": "md",
                },
                missing_rows,
            )

    def test_cross_aggregation_normalizes_legacy_method_ids(self) -> None:
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            result_dir = root / "cross" / "atom_displacement__on__test_atomdisp"
            (result_dir / "metrics").mkdir(parents=True)
            write_csv(
                result_dir / "metrics" / "sparse_metrics.csv",
                [{"sample": "sample_1", "relative_frobenius_union": "1.0"}],
            )
            (result_dir / "cross_evaluation_manifest.json").write_text(
                json.dumps(
                    {
                        "train_method": "atom_displacement",
                        "test_set": "test_atomdisp",
                        "test_method": "atomdisp",
                        "dataset_size": 4,
                        "dataset_size_by_method": {"md": 3, "atom_displacement": 4},
                    }
                ),
                encoding="utf-8",
            )
            result = run_script(
                "Comparison/scripts/aggregate_cross_metrics.py",
                "--experiment-id",
                "exp_alias",
                "--cross-root",
                str(root / "cross"),
                "--output-dir",
                str(root / "summary"),
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            with (root / "summary" / "cross_evaluation_metrics.csv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["train_method"], "siesta_fc_cartesian")
            self.assertEqual(rows[0]["test_set"], "test_siesta_fc_cartesian")
            self.assertEqual(rows[0]["test_method"], "siesta_fc_cartesian")
            self.assertEqual(
                json.loads(rows[0]["dataset_size_by_method"]),
                {"md": 3, "siesta_fc_cartesian": 4},
            )

    def test_md_blocked_with_gap_separates_splits_and_excludes_gap_frames(self) -> None:
        module = self.load_md_dataset_module("generate_md_dataset_gap_unit_test")

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
            assigned = {
                split_name: [int(path.name) for path in samples]
                for split_name, samples in splits.items()
            }
            split_names = ["train", "validation", "test"]
            for left_index, left_name in enumerate(split_names):
                for right_name in split_names[left_index + 1 :]:
                    for left_frame in assigned[left_name]:
                        for right_frame in assigned[right_name]:
                            self.assertGreater(abs(left_frame - right_frame), 1)
            used = {path.name for samples in splits.values() for path in samples}
            self.assertTrue(used.isdisjoint({path.name for path, _reason in excluded}))
            with self.assertRaisesRegex(RuntimeError, "necesita mas frames"):
                module._split_blocked_with_gap(
                    frames[:5],
                    {"train": 3, "validation": 2, "test": 2},
                    temporal_gap=1,
                    block_order=["train", "validation", "test"],
                )

    def test_md_default_split_config_and_ui_are_leakage_safer(self) -> None:
        config_text = (REPO_ROOT / "MD" / "pipeline_config.yaml").read_text(encoding="utf-8")
        index_html = (REPO_ROOT / "Comparison" / "ui" / "index.html").read_text(encoding="utf-8")
        module = self.load_pipeline_ui_module()

        self.assertIn("strategy: blocked_with_gap", config_text)
        self.assertIn("temporal_gap: 1", config_text)
        self.assertNotIn("strategy: spread", config_text)
        self.assertIn('value="blocked_with_gap" selected', index_html)
        self.assertIn("Spread split (exploratory)", index_html)
        self.assertEqual(module.parse_split_mode(None), "blocked_with_gap")
        counts, reserved_gap_frames = module.md_split_counts_for_mode(
            10,
            {"train": 0.8, "validation": 0.1, "test": 0.1},
            split_mode="blocked_with_gap",
            label="MD dataset_10",
        )
        self.assertEqual(counts, {"train": 6, "validation": 1, "test": 1})
        self.assertEqual(reserved_gap_frames, 2)

    def test_md_prepare_dataset_splits_writes_gap_metadata_and_spread_warning(self) -> None:
        module = self.load_md_dataset_module("generate_md_dataset_gap_manifest_test")

        def make_config(root: Path, *, strategy: str) -> dict:
            dataset_dir = root / strategy / "dataset"
            steps_dir = dataset_dir / "MD_steps"
            steps_dir.mkdir(parents=True)
            for index in range(10):
                sample_dir = steps_dir / str(index)
                sample_dir.mkdir()
                (sample_dir / "RUN.fdf").write_text("SystemLabel siesta\n", encoding="utf-8")
                (sample_dir / "siesta.TSHS").write_text("matrix\n", encoding="utf-8")
            (dataset_dir / "RUN.out").write_text("siesta completed\n", encoding="utf-8")
            return {
                "paths": {
                    "dataset_dir": str(dataset_dir),
                    "training_dir": str(root / strategy / "training"),
                    "run_fdf_name": "RUN.fdf",
                    "run_out_name": "RUN.out",
                    "training_config_name": "config.yaml",
                    "venv_activate": str(root / "venv" / "activate"),
                },
                "md": {"steps": 10, "temperature_blocks": []},
                "splits": {
                    "enabled": True,
                    "strategy": strategy,
                    "train": 6,
                    "validation": 1,
                    "test": 1,
                    "temporal_gap": 1,
                    "block_order": "train,validation,test",
                },
            }

        with workspace_tempdir() as tmp:
            root = Path(tmp)
            safe_config = make_config(root, strategy="blocked_with_gap")
            module.prepare_dataset_splits(safe_config)
            split_root = Path(safe_config["paths"]["dataset_dir"]) / "splits"
            summary = json.loads((split_root / "split_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["strategy"], "blocked_with_gap")
            self.assertEqual(summary["temporal_gap"], 1)
            self.assertEqual(summary["counts"], {"train": 6, "validation": 1, "test": 1})
            self.assertEqual(
                [item["frame_index"] for item in summary["excluded_gap_samples"]],
                ["6", "8"],
            )
            with (split_root / "train_manifest.csv").open(encoding="utf-8") as handle:
                train_rows = list(csv.DictReader(handle))
            self.assertEqual({row["temporal_gap"] for row in train_rows}, {"1"})
            with (split_root / "excluded_gap_manifest.csv").open(encoding="utf-8") as handle:
                excluded_rows = list(csv.DictReader(handle))
            self.assertEqual(
                [row["excluded_gap_reason"] for row in excluded_rows],
                [
                    "temporal_gap_between_train_and_validation",
                    "temporal_gap_between_validation_and_test",
                ],
            )

            spread_config = make_config(root, strategy="spread")
            module.prepare_dataset_splits(spread_config)
            spread_summary = json.loads(
                (
                    Path(spread_config["paths"]["dataset_dir"])
                    / "splits"
                    / "split_summary.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(
                spread_summary["scientific_status"],
                "exploratory_temporal_leakage_risk",
            )
            self.assertTrue(
                any("exploratory/debug only" in warning for warning in spread_summary["warnings"])
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

    def test_fc_zero_reference_duplicates_are_excluded_from_splits(self) -> None:
        module = self.load_pipeline_ui_module()
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            samples = []
            for index, run_id in enumerate(("fc_000_0_02_ang", "fc_001_0_03_ang", "fc_002_0_05_ang")):
                sample = make_sample(root / "samples", f"{index:03d}")
                (sample / "metadata.json").write_text(
                    json.dumps(
                        {
                            "generation_mode": "siesta_fc_multi_normalized_step",
                            "raw_displacement_run_id": run_id,
                            "raw_fc_run_dir": f"/tmp/{run_id}",
                            "matrix_label": f"{run_id}.00000",
                            "displacement_ang": 0.0,
                            "displacement_input": "0 Ang",
                            "positions_ang": [[0.0, 0.0, 0.0], [0.0, 0.7, 0.0], [0.0, -0.7, 0.0]],
                        }
                    ),
                    encoding="utf-8",
                )
                samples.append(sample)
            for offset, axis in enumerate((1, 2, 3), start=3):
                sample = make_sample(root / "samples", f"{offset:03d}")
                (sample / "metadata.json").write_text(
                    json.dumps(
                        {
                            "generation_mode": "siesta_fc_multi_normalized_step",
                            "raw_displacement_run_id": "fc_000_0_02_ang",
                            "raw_fc_run_dir": "/tmp/fc_000_0_02_ang",
                            "matrix_label": f"fc_000_0_02_ang.00001-{axis}",
                            "displacement_ang": 0.02,
                            "displacement_input": "0.02 Ang",
                            "atom": 1,
                            "direction": axis,
                            "sign": 1,
                        }
                    ),
                    encoding="utf-8",
                )
                samples.append(sample)

            zero_groups = {module.atom_split_group_id(sample) for sample in samples[:3]}
            self.assertEqual(len(zero_groups), 1)
            self.assertEqual([sample.name for sample in module.atom_visible_split_samples(samples)], ["003", "004", "005"])
            splits = module.split_grouped_exact(samples, {"train": 1, "validation": 1, "test": 1})
            split_names = {
                split: {sample.name for sample in split_samples}
                for split, split_samples in splits.items()
            }
            self.assertEqual(split_names["train"] | split_names["validation"] | split_names["test"], {"003", "004", "005"})
            self.assertTrue({"000", "001", "002"}.isdisjoint(split_names["train"]))
            self.assertTrue({"000", "001", "002"}.isdisjoint(split_names["validation"]))
            self.assertTrue({"000", "001", "002"}.isdisjoint(split_names["test"]))

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

    def test_winner_analysis_does_not_treat_atomdisp_only_win_as_generalization(self) -> None:
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            metrics = root / "cross_evaluation_metrics.csv"
            rows = []
            for seed in ("1", "2", "3"):
                for method, value in (("md", "1.0"), ("atom_displacement", "0.1")):
                    rows.append(
                        {
                            "experiment_id": "exp_atom_only",
                            "train_method": method,
                            "test_set": "test_atomdisp",
                            "md_dataset_size": "3",
                            "atom_dataset_size": "3",
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
            self.assertNotEqual(recommendation["status"], "atom_displacement_conservative_win")
            self.assertIn("missing_required_cells", recommendation)

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
            self.assertEqual(recommendation["status"], "fairness_provenance_mismatch")
            self.assertEqual(recommendation["scientific_status"], "not_scientifically_valid")
            self.assertIn("settings mismatch", " ".join(recommendation["severe_warnings"]))

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
                                "low_energy_rmse_eV": value,
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
                "low_energy_rmse_eV",
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            recommendation = json.loads((root / "summary_two" / "recommendation.json").read_text(encoding="utf-8"))
            self.assertEqual(recommendation["scientific_status"], "exploratory_only")
            self.assertEqual(recommendation["status"], "insufficient_seeds")
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
                            "low_energy_rmse_eV": value,
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
                "low_energy_rmse_eV",
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            recommendation = json.loads((root / "summary_three" / "recommendation.json").read_text(encoding="utf-8"))
            self.assertEqual(recommendation["scientific_status"], "robust_comparison")

    def test_winner_analysis_normalizes_legacy_fc_aliases(self) -> None:
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
                                "test_method": test_set.removeprefix("test_"),
                                "dataset_size_by_method": json.dumps({"md": 10, "siesta_fc_cartesian": 10}),
                                "seed": seed,
                                "model_checkpoint": f"{method}_{seed}.ckpt",
                                "low_energy_rmse_eV": value,
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
                "low_energy_rmse_eV",
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            recommendation = json.loads((root / "summary" / "recommendation.json").read_text(encoding="utf-8"))
            self.assertEqual(recommendation["scientific_status"], "robust_comparison")
            self.assertEqual(recommendation["status"], "md_conservative_win")
            self.assertIn("siesta_fc_cartesian", recommendation["methods_seen"])
            self.assertNotIn("atom_displacement", recommendation["methods_seen"])
            self.assertIn("test_siesta_fc_cartesian", recommendation["test_sets_seen"])
            self.assertNotIn("test_atomdisp", recommendation["test_sets_seen"])
            with (root / "summary" / "winner_by_dataset_size.csv").open(encoding="utf-8") as handle:
                pair_rows = list(csv.DictReader(handle))
            winners = {row["winner"] for row in pair_rows}
            self.assertLessEqual(winners, {"md", "siesta_fc_cartesian", "tie"})
            with (root / "summary" / "nway_method_summary.csv").open(encoding="utf-8") as handle:
                method_rows = list(csv.DictReader(handle))
            self.assertEqual({row["method"] for row in method_rows}, {"md", "siesta_fc_cartesian"})
            self.assertNotIn("atom_displacement", {row["method"] for row in method_rows})

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
            self.assertEqual(recommendation["scientific_status"], "exploratory_only")
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
                                "low_energy_rmse_eV": value,
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
                "low_energy_rmse_eV",
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            recommendation = json.loads((root / "summary" / "recommendation.json").read_text(encoding="utf-8"))
            self.assertEqual(recommendation["status"], "fairness_provenance_mismatch")
            self.assertEqual(recommendation["scientific_status"], "not_scientifically_valid")
            self.assertIn("latest-version fallback", " ".join(recommendation["severe_warnings"]))

    def test_winner_analysis_inconclusive_on_random_cartesian_family_leakage(self) -> None:
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            metrics = root / "cross_evaluation_metrics.csv"
            rows = []
            for seed in ("1", "2", "3"):
                for method, value in (("md", "0.5"), ("random_cartesian", "0.4")):
                    for test_set in ("test_md", "test_random_cartesian"):
                        rows.append(
                            {
                                "experiment_id": "exp_test",
                                "train_method": method,
                                "test_set": test_set,
                                "dataset_size_by_method": json.dumps({"md": 10, "random_cartesian": 10}),
                                "seed": seed,
                                "model_checkpoint": f"{method}_{seed}.ckpt",
                                "leakage_warning": (
                                    "Geometry leakage detected for md_10__rc_10 test_random_cartesian; "
                                    "scientific_status=exploratory_only; "
                                    "random_cartesian_family_warnings=2"
                                ),
                                "low_energy_rmse_eV": value,
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
                "low_energy_rmse_eV",
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            recommendation = json.loads((root / "summary" / "recommendation.json").read_text(encoding="utf-8"))
            self.assertEqual(recommendation["status"], "leakage_exploratory_only")
            self.assertEqual(recommendation["scientific_status"], "not_scientifically_valid")
            self.assertNotEqual(recommendation["scientific_status"], "robust_comparison")
            self.assertIn("random_cartesian_family_warnings", " ".join(recommendation["severe_warnings"]))

    def test_winner_analysis_invalid_leakage_blocks_recommendation(self) -> None:
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            metrics = root / "cross_evaluation_metrics.csv"
            rows = []
            for seed in ("1", "2", "3"):
                for method, value in (("md", "0.4"), ("atom_displacement", "0.9")):
                    for test_set in ("test_md", "test_atomdisp", "test_mixed"):
                        rows.append(
                            {
                                "experiment_id": "exp_exact_leakage",
                                "train_method": method,
                                "test_set": test_set,
                                "md_dataset_size": "10",
                                "atom_dataset_size": "10",
                                "seed": seed,
                                "model_checkpoint": f"{method}_{seed}.ckpt",
                                "leakage_scientific_status": "invalid_leakage",
                                "leakage_severe_warnings": json.dumps(
                                    ["exact_duplicate_geometry_cross_split: 1 pair(s)"]
                                ),
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
            self.assertEqual(recommendation["status"], "invalid_leakage")
            self.assertEqual(recommendation["scientific_status"], "not_scientifically_valid")
            self.assertNotEqual(recommendation["scientific_status"], "robust_comparison")
            self.assertTrue(recommendation["leakage_diagnostics"]["invalid"])

    def test_winner_analysis_clean_leakage_status_allows_robust_recommendation(self) -> None:
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            metrics = root / "cross_evaluation_metrics.csv"
            rows = []
            for seed in ("1", "2", "3"):
                for method, value in (("md", "0.4"), ("atom_displacement", "0.9")):
                    for test_set in ("test_md", "test_atomdisp", "test_mixed"):
                        rows.append(
                            {
                                "experiment_id": "exp_clean_leakage",
                                "train_method": method,
                                "test_set": test_set,
                                "md_dataset_size": "10",
                                "atom_dataset_size": "10",
                                "seed": seed,
                                "model_checkpoint": f"{method}_{seed}.ckpt",
                                "leakage_scientific_status": "valid_independent_splits",
                                "low_energy_rmse_eV": value,
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
                "low_energy_rmse_eV",
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            recommendation = json.loads((root / "summary" / "recommendation.json").read_text(encoding="utf-8"))
            self.assertEqual(recommendation["status"], "md_conservative_win")
            self.assertEqual(recommendation["scientific_status"], "robust_comparison")
            self.assertTrue(recommendation["leakage_diagnostics"]["clean"])

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
            self.assertEqual(recommendation["status"], "fairness_provenance_mismatch")
            self.assertEqual(recommendation["scientific_status"], "not_scientifically_valid")
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
            self.assertIsNone(recommendation["winner"])
            self.assertEqual(recommendation["primary_metric_policy_status"], "missing_required_metric")
            self.assertIn("siesta_fc_cartesian on test_mixed", recommendation["missing_primary_metric_cells"])

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
            self.assertEqual(recommendation["scientific_status"], "not_scientifically_valid")
            self.assertEqual(recommendation["status"], "insufficient_primary_metric")
            self.assertIn("siesta_fc_cartesian on test_mixed", recommendation["missing_primary_metric_cells"])

    def test_winner_analysis_missing_fermi_metric_is_explicitly_insufficient(self) -> None:
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            metrics = root / "cross_evaluation_metrics.csv"
            rows = []
            for seed in ("1", "2", "3"):
                for method, value in (("md", "0.5"), ("atom_displacement", "1.0")):
                    for test_set in ("test_md", "test_atomdisp", "test_mixed"):
                        rows.append(
                            {
                                "experiment_id": "exp_no_fermi",
                                "train_method": method,
                                "test_set": test_set,
                                "md_dataset_size": "10",
                                "atom_dataset_size": "10",
                                "seed": seed,
                                "model_checkpoint": f"{method}_{seed}.ckpt",
                                "fermi_level_source": "unavailable",
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
                "fermi_window_rmse_eV",
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            recommendation = json.loads((root / "summary" / "recommendation.json").read_text(encoding="utf-8"))
            self.assertEqual(recommendation["status"], "insufficient_primary_metric")
            self.assertEqual(recommendation["scientific_status"], "not_scientifically_valid")
            self.assertIsNone(recommendation["winner"])
            self.assertEqual(recommendation["primary_metric_policy_status"], "missing_required_metric")
            self.assertIn("fermi_window_rmse_eV", recommendation["metric_policy"]["primary_metric"])
            self.assertGreater(recommendation["metric_policy"]["missing_required_cell_count"], 0)

    def test_winner_analysis_global_rmse_only_rc_win_is_exploratory(self) -> None:
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            metrics = root / "cross_evaluation_metrics.csv"
            rows = []
            methods = ("md", "random_cartesian")
            test_sets = ("test_md", "test_random_cartesian", "test_mixed")
            for seed in ("1", "2", "3"):
                for method in methods:
                    for test_set in test_sets:
                        rows.append(
                            {
                                "experiment_id": "exp_global_only_rc",
                                "train_method": method,
                                "test_set": test_set,
                                "dataset_size_by_method": json.dumps({"md": 10, "random_cartesian": 10}),
                                "seed": seed,
                                "model_checkpoint": f"{method}_{seed}.ckpt",
                                "global_rmse_eV": "0.4" if method == "random_cartesian" else "1.0",
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
            self.assertEqual(recommendation["primary_metric_policy_status"], "diagnostic_only_metric")
            self.assertFalse(recommendation["robust_primary_metric_allowed"])
            self.assertNotEqual(recommendation["scientific_status"], "robust_comparison")
            self.assertIn(recommendation["scientific_status"], {"exploratory", "exploratory_only"})
            self.assertEqual(recommendation["nway_consensus_leader"], "random_cartesian")

    def test_winner_analysis_p0_matrix_metrics_are_diagnostic_with_r2_direction(self) -> None:
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            metrics = root / "cross_evaluation_metrics.csv"
            rows = []
            for seed in ("1", "2", "3"):
                for method in ("md", "siesta_fc_cartesian"):
                    for test_set in ("test_md", "test_siesta_fc_cartesian", "test_mixed"):
                        rows.append(
                            {
                                "experiment_id": "exp_p0_matrix",
                                "train_method": method,
                                "test_set": test_set,
                                "dataset_size_by_method": json.dumps({"md": 10, "siesta_fc_cartesian": 10}),
                                "seed": seed,
                                "model_checkpoint": f"{method}_{seed}.ckpt",
                                "mse_ref_eV2": "0.4" if method == "md" else "0.1",
                                "r2_ref": "0.2" if method == "md" else "0.9",
                            }
                        )
            write_csv(metrics, rows)

            mse_result = run_script(
                "Comparison/scripts/analyze_winners.py",
                "--metrics-csv",
                str(metrics),
                "--output-dir",
                str(root / "summary_mse"),
                "--primary-metric",
                "mse_ref_eV2",
            )
            self.assertEqual(mse_result.returncode, 0, mse_result.stderr + mse_result.stdout)
            mse_recommendation = json.loads((root / "summary_mse" / "recommendation.json").read_text(encoding="utf-8"))
            self.assertEqual(mse_recommendation["primary_metric_policy_status"], "diagnostic_only_metric")
            self.assertEqual(mse_recommendation["primary_metric_recommendation_role"], "diagnostic_only")
            self.assertEqual(mse_recommendation["metric_policy"]["metric_direction"], "lower_is_better")
            self.assertFalse(mse_recommendation["robust_primary_metric_allowed"])

            r2_result = run_script(
                "Comparison/scripts/analyze_winners.py",
                "--metrics-csv",
                str(metrics),
                "--output-dir",
                str(root / "summary_r2"),
                "--primary-metric",
                "r2_ref",
            )
            self.assertEqual(r2_result.returncode, 0, r2_result.stderr + r2_result.stdout)
            r2_recommendation = json.loads((root / "summary_r2" / "recommendation.json").read_text(encoding="utf-8"))
            self.assertEqual(r2_recommendation["primary_metric_policy_status"], "diagnostic_only_metric")
            self.assertEqual(r2_recommendation["metric_policy"]["metric_direction"], "higher_is_better")
            with (root / "summary_r2" / "winner_by_dataset_size.csv").open(encoding="utf-8") as handle:
                pair_rows = [row for row in csv.DictReader(handle) if row["metric"] == "r2_ref"]
            self.assertTrue(pair_rows)
            self.assertEqual({row["lower_is_better"] for row in pair_rows}, {"False"})
            self.assertEqual({row["winner"] for row in pair_rows}, {"siesta_fc_cartesian"})

    def test_winner_analysis_dos_fermi_window_mae_missing_is_not_substituted(self) -> None:
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            metrics = root / "cross_evaluation_metrics.csv"
            rows = []
            for seed in ("1", "2", "3"):
                for method in ("md", "siesta_fc_cartesian"):
                    for test_set in ("test_md", "test_siesta_fc_cartesian", "test_mixed"):
                        rows.append(
                            {
                                "experiment_id": "exp_p0_dos_missing",
                                "train_method": method,
                                "test_set": test_set,
                                "dataset_size_by_method": json.dumps({"md": 10, "siesta_fc_cartesian": 10}),
                                "seed": seed,
                                "model_checkpoint": f"{method}_{seed}.ckpt",
                                "low_energy_rmse_eV": "0.1" if method == "md" else "0.2",
                                "dos_wasserstein_eV": "0.1" if method == "md" else "0.2",
                                "dos_window_metric_available": "False",
                                "dos_window_unavailable_reason": "missing_fermi_level",
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
                "dos_mae_500_fermi_window",
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            recommendation = json.loads((root / "summary" / "recommendation.json").read_text(encoding="utf-8"))
            self.assertEqual(recommendation["status"], "insufficient_primary_metric")
            self.assertEqual(recommendation["primary_metric_policy_status"], "missing_required_metric")
            self.assertIsNone(recommendation["winner"])
            self.assertIn("dos_mae_500_fermi_window", recommendation["metric_policy"]["primary_metric"])
            self.assertGreater(recommendation["metric_policy"]["missing_required_cell_count"], 0)

            present_metrics = root / "cross_evaluation_metrics_present.csv"
            present_rows = [
                {
                    **row,
                    "dos_mae_500_fermi_window": "0.1" if row["train_method"] == "md" else "0.2",
                    "dos_window_metric_available": "True",
                    "dos_window_unavailable_reason": "",
                }
                for row in rows
            ]
            write_csv(present_metrics, present_rows)
            present_result = run_script(
                "Comparison/scripts/analyze_winners.py",
                "--metrics-csv",
                str(present_metrics),
                "--output-dir",
                str(root / "summary_present"),
                "--primary-metric",
                "dos_mae_500_fermi_window",
            )
            self.assertEqual(present_result.returncode, 0, present_result.stderr + present_result.stdout)
            present_recommendation = json.loads(
                (root / "summary_present" / "recommendation.json").read_text(encoding="utf-8")
            )
            self.assertEqual(present_recommendation["primary_metric_policy_status"], "fermi_window_metric_unjustified")
            self.assertEqual(present_recommendation["primary_metric_recommendation_role"], "fermi_window_conditional")
            self.assertFalse(present_recommendation["robust_primary_metric_allowed"])

    def test_final_recommendation_robust_random_cartesian_winner(self) -> None:
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            metrics = root / "cross_evaluation_metrics.csv"
            output_dir = root / "summary"
            methods = ("md", "siesta_fc_cartesian", "random_cartesian")
            test_sets = ("test_md", "test_siesta_fc_cartesian", "test_random_cartesian", "test_mixed")

            def values(method: str, _test_set: str, _seed: str) -> str:
                return {"random_cartesian": "0.2", "md": "0.5", "siesta_fc_cartesian": "0.8"}[method]

            self.write_final_recommendation_metric_grid(metrics, value_for=values)
            self.write_completeness_report(
                output_dir / "cross_evaluation_completeness.json",
                methods,
                test_sets,
                primary_metric="low_energy_rmse_eV",
            )
            result = run_script(
                "Comparison/scripts/analyze_winners.py",
                "--metrics-csv",
                str(metrics),
                "--output-dir",
                str(output_dir),
                "--primary-metric",
                "low_energy_rmse_eV",
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            recommendation = json.loads((output_dir / "recommendation.json").read_text(encoding="utf-8"))
            self.assertEqual(recommendation["scientific_status"], "robust_comparison")
            self.assertEqual(recommendation["baseline"], "md")
            self.assertEqual(recommendation["challengers"], ["siesta_fc_cartesian", "random_cartesian"])
            self.assertEqual(recommendation["winner"], "random_cartesian")
            self.assertEqual(recommendation["first_dataset_size_surpassing_md"], 3500)
            self.assertIsNone(recommendation["first_compute_budget_surpassing_md"])
            self.assertTrue(recommendation["complete_grid"])

    def test_final_recommendation_robust_fc_winner(self) -> None:
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            metrics = root / "cross_evaluation_metrics.csv"
            output_dir = root / "summary"
            methods = ("md", "siesta_fc_cartesian", "random_cartesian")
            test_sets = ("test_md", "test_siesta_fc_cartesian", "test_random_cartesian", "test_mixed")

            def values(method: str, _test_set: str, _seed: str) -> str:
                return {"siesta_fc_cartesian": "0.2", "md": "0.5", "random_cartesian": "0.8"}[method]

            self.write_final_recommendation_metric_grid(metrics, value_for=values)
            self.write_completeness_report(
                output_dir / "cross_evaluation_completeness.json",
                methods,
                test_sets,
                primary_metric="low_energy_rmse_eV",
            )
            result = run_script(
                "Comparison/scripts/analyze_winners.py",
                "--metrics-csv",
                str(metrics),
                "--output-dir",
                str(output_dir),
                "--primary-metric",
                "low_energy_rmse_eV",
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            recommendation = json.loads((output_dir / "recommendation.json").read_text(encoding="utf-8"))
            self.assertEqual(recommendation["scientific_status"], "robust_comparison")
            self.assertEqual(recommendation["winner"], "siesta_fc_cartesian")
            self.assertEqual(recommendation["first_dataset_size_surpassing_md"], 750)

    def test_final_recommendation_md_remains_winner(self) -> None:
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            metrics = root / "cross_evaluation_metrics.csv"
            output_dir = root / "summary"
            methods = ("md", "siesta_fc_cartesian", "random_cartesian")
            test_sets = ("test_md", "test_siesta_fc_cartesian", "test_random_cartesian", "test_mixed")

            def values(method: str, _test_set: str, _seed: str) -> str:
                return {"md": "0.2", "siesta_fc_cartesian": "0.5", "random_cartesian": "0.8"}[method]

            self.write_final_recommendation_metric_grid(metrics, value_for=values)
            self.write_completeness_report(
                output_dir / "cross_evaluation_completeness.json",
                methods,
                test_sets,
                primary_metric="low_energy_rmse_eV",
            )
            result = run_script(
                "Comparison/scripts/analyze_winners.py",
                "--metrics-csv",
                str(metrics),
                "--output-dir",
                str(output_dir),
                "--primary-metric",
                "low_energy_rmse_eV",
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            recommendation = json.loads((output_dir / "recommendation.json").read_text(encoding="utf-8"))
            self.assertEqual(recommendation["scientific_status"], "robust_comparison")
            self.assertEqual(recommendation["winner"], "md")
            self.assertIsNone(recommendation["first_dataset_size_surpassing_md"])

    def test_final_recommendation_blocks_incomplete_grid(self) -> None:
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            metrics = root / "cross_evaluation_metrics.csv"
            output_dir = root / "summary"
            methods = ("md", "siesta_fc_cartesian", "random_cartesian")
            test_sets = ("test_md", "test_siesta_fc_cartesian", "test_random_cartesian", "test_mixed")
            self.write_final_recommendation_metric_grid(
                metrics,
                skip_cell=lambda method, test_set, _seed: method == "random_cartesian" and test_set == "test_md",
            )
            self.write_completeness_report(
                output_dir / "cross_evaluation_completeness.json",
                methods,
                test_sets,
                primary_metric="low_energy_rmse_eV",
                missing_cells=("random_cartesian on test_md",),
            )
            result = run_script(
                "Comparison/scripts/analyze_winners.py",
                "--metrics-csv",
                str(metrics),
                "--output-dir",
                str(output_dir),
                "--primary-metric",
                "low_energy_rmse_eV",
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            recommendation = json.loads((output_dir / "recommendation.json").read_text(encoding="utf-8"))
            self.assertEqual(recommendation["scientific_status"], "not_scientifically_valid")
            self.assertIsNone(recommendation["winner"])
            self.assertEqual(recommendation["reason"], "Incomplete 3-method cross-evaluation grid")
            self.assertIn("random_cartesian on test_md", recommendation["missing_cells"])

    def test_final_recommendation_blocks_one_seed_only(self) -> None:
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            metrics = root / "cross_evaluation_metrics.csv"
            output_dir = root / "summary"
            methods = ("md", "siesta_fc_cartesian", "random_cartesian")
            test_sets = ("test_md", "test_siesta_fc_cartesian", "test_random_cartesian", "test_mixed")
            self.write_final_recommendation_metric_grid(
                metrics,
                seeds=("1",),
                value_for=lambda method, _test_set, _seed: {"random_cartesian": "0.2", "md": "0.5", "siesta_fc_cartesian": "0.8"}[method],
            )
            self.write_completeness_report(
                output_dir / "cross_evaluation_completeness.json",
                methods,
                test_sets,
                primary_metric="low_energy_rmse_eV",
            )
            result = run_script(
                "Comparison/scripts/analyze_winners.py",
                "--metrics-csv",
                str(metrics),
                "--output-dir",
                str(output_dir),
                "--primary-metric",
                "low_energy_rmse_eV",
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            recommendation = json.loads((output_dir / "recommendation.json").read_text(encoding="utf-8"))
            self.assertEqual(recommendation["scientific_status"], "exploratory_only")
            self.assertEqual(recommendation["status"], "insufficient_seeds")
            self.assertIsNone(recommendation["winner"])

    def test_final_recommendation_blocks_unstable_seeds(self) -> None:
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            metrics = root / "cross_evaluation_metrics.csv"
            output_dir = root / "summary"
            methods = ("md", "siesta_fc_cartesian", "random_cartesian")
            test_sets = ("test_md", "test_siesta_fc_cartesian", "test_random_cartesian", "test_mixed")

            def values(method: str, _test_set: str, seed: str) -> str:
                if method == "siesta_fc_cartesian":
                    return "0.8"
                if seed == "3":
                    return {"md": "0.2", "random_cartesian": "0.6"}[method]
                return {"md": "0.6", "random_cartesian": "0.2"}[method]

            self.write_final_recommendation_metric_grid(metrics, value_for=values)
            self.write_completeness_report(
                output_dir / "cross_evaluation_completeness.json",
                methods,
                test_sets,
                primary_metric="low_energy_rmse_eV",
            )
            result = run_script(
                "Comparison/scripts/analyze_winners.py",
                "--metrics-csv",
                str(metrics),
                "--output-dir",
                str(output_dir),
                "--primary-metric",
                "low_energy_rmse_eV",
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            recommendation = json.loads((output_dir / "recommendation.json").read_text(encoding="utf-8"))
            self.assertEqual(recommendation["scientific_status"], "not_scientifically_valid")
            self.assertEqual(recommendation["status"], "unstable_seed_winner")
            self.assertIsNone(recommendation["winner"])

    def test_final_recommendation_blocks_leakage(self) -> None:
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            metrics = root / "cross_evaluation_metrics.csv"
            output_dir = root / "summary"
            methods = ("md", "siesta_fc_cartesian", "random_cartesian")
            test_sets = ("test_md", "test_siesta_fc_cartesian", "test_random_cartesian", "test_mixed")
            self.write_final_recommendation_metric_grid(
                metrics,
                value_for=lambda method, _test_set, _seed: {"random_cartesian": "0.2", "md": "0.5", "siesta_fc_cartesian": "0.8"}[method],
                extra_for=lambda _method, _test_set, _seed: {
                    "leakage_scientific_status": "invalid_leakage",
                    "leakage_severe_warnings": json.dumps(["exact_duplicate_geometry_cross_split: 1 pair(s)"]),
                },
            )
            self.write_completeness_report(
                output_dir / "cross_evaluation_completeness.json",
                methods,
                test_sets,
                primary_metric="low_energy_rmse_eV",
            )
            result = run_script(
                "Comparison/scripts/analyze_winners.py",
                "--metrics-csv",
                str(metrics),
                "--output-dir",
                str(output_dir),
                "--primary-metric",
                "low_energy_rmse_eV",
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            recommendation = json.loads((output_dir / "recommendation.json").read_text(encoding="utf-8"))
            self.assertEqual(recommendation["scientific_status"], "not_scientifically_valid")
            self.assertEqual(recommendation["status"], "invalid_leakage")
            self.assertIsNone(recommendation["winner"])

    def test_final_recommendation_blocks_missing_primary_metric(self) -> None:
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            metrics = root / "cross_evaluation_metrics.csv"
            output_dir = root / "summary"
            methods = ("md", "siesta_fc_cartesian", "random_cartesian")
            test_sets = ("test_md", "test_siesta_fc_cartesian", "test_random_cartesian", "test_mixed")
            self.write_final_recommendation_metric_grid(
                metrics,
                value_for=lambda method, test_set, _seed: None
                if method == "random_cartesian" and test_set == "test_mixed"
                else {"random_cartesian": "0.2", "md": "0.5", "siesta_fc_cartesian": "0.8"}[method],
            )
            self.write_completeness_report(
                output_dir / "cross_evaluation_completeness.json",
                methods,
                test_sets,
                primary_metric="low_energy_rmse_eV",
                missing_primary_metric_cells=("random_cartesian on test_mixed",),
            )
            result = run_script(
                "Comparison/scripts/analyze_winners.py",
                "--metrics-csv",
                str(metrics),
                "--output-dir",
                str(output_dir),
                "--primary-metric",
                "low_energy_rmse_eV",
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            recommendation = json.loads((output_dir / "recommendation.json").read_text(encoding="utf-8"))
            self.assertEqual(recommendation["scientific_status"], "not_scientifically_valid")
            self.assertEqual(recommendation["status"], "insufficient_primary_metric")
            self.assertIsNone(recommendation["winner"])
            self.assertIn("random_cartesian on test_mixed", recommendation["missing_primary_metric_cells"])

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
            self.assertEqual(recommendation["scientific_status"], "not_scientifically_valid")
            self.assertEqual(recommendation["status"], "invalid_incomplete_grid")
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
            self.assertEqual(recommendation["status"], "insufficient_seeds")
            self.assertEqual(recommendation["scientific_status"], "exploratory_only")
            self.assertEqual(recommendation["legacy_recommendation"]["status"], "nway_consensus_win")
            self.assertEqual(recommendation["nway_consensus_leader"], "md")
            self.assertEqual(set(recommendation["nway_leaders_by_test_set"].values()), {"md"})
            self.assertNotIn("legacy global-winner", recommendation["reason"])
            with (root / "summary" / "nway_method_summary.csv").open(encoding="utf-8") as handle:
                method_rows = list(csv.DictReader(handle))
            method_test_cells = {(row["method"], row["test_set"]) for row in method_rows}
            self.assertIn(("random_cartesian", "test_md"), method_test_cells)
            self.assertIn(("random_cartesian", "test_siesta_fc_cartesian"), method_test_cells)
            self.assertIn(("random_cartesian", "test_random_cartesian"), method_test_cells)
            self.assertNotIn("atom_displacement", {row["method"] for row in method_rows})
            with (root / "summary" / "nway_ranking.csv").open(encoding="utf-8") as handle:
                ranking_rows = list(csv.DictReader(handle))
            self.assertIn("random_cartesian", {row["method"] for row in ranking_rows})
            random_rank = next(
                row
                for row in ranking_rows
                if row["method"] == "random_cartesian" and row["test_set"] == "test_md"
            )
            self.assertEqual(random_rank["rank"], "3")
            self.assertEqual(random_rank["n_methods_ranked"], "3")

    def test_nway_ranking_reports_stable_three_method_ranking_across_seeds(self) -> None:
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            metrics = root / "cross_evaluation_metrics.csv"
            methods = ("md", "siesta_fc_cartesian", "random_cartesian")
            values = {"md": "0.2", "siesta_fc_cartesian": "0.4", "random_cartesian": "0.6"}
            rows = []
            for seed in ("1", "2", "3"):
                for method in methods:
                    rows.append(
                        {
                            "experiment_id": "exp_nway_stable",
                            "train_method": method,
                            "test_set": "test_md",
                            "dataset_size_by_method": json.dumps(
                                {"md": 750, "siesta_fc_cartesian": 750, "random_cartesian": 3500}
                            ),
                            "dataset_label_by_method": json.dumps(
                                {"md": "MD_750", "siesta_fc_cartesian": "FC_750", "random_cartesian": "RC_3500"}
                            ),
                            "recipe_hash_by_method": json.dumps(
                                {"md": "md_hash", "siesta_fc_cartesian": "fc_hash", "random_cartesian": "rc_hash"}
                            ),
                            "seed": seed,
                            "model_checkpoint": f"{method}_{seed}.ckpt",
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
            with (root / "summary" / "nway_ranking.csv").open(encoding="utf-8") as handle:
                ranking_rows = list(csv.DictReader(handle))
            self.assertEqual({row["ranking_stability_status"] for row in ranking_rows}, {"robust_candidate"})
            self.assertEqual({row["ranking_status"] for row in ranking_rows}, {"robust_candidate"})
            self.assertEqual({row["ranking_seed_count"] for row in ranking_rows}, {"3"})
            self.assertEqual({row["ranking_valid_seed_count"] for row in ranking_rows}, {"3"})
            self.assertEqual({row["missing_methods"] for row in ranking_rows}, {""})
            md_rows = [row for row in ranking_rows if row["method"] == "md"]
            self.assertEqual({row["rank"] for row in md_rows}, {"1"})
            self.assertEqual({row["dataset_label_by_method"] for row in md_rows}, {
                json.dumps({"md": "MD_750", "random_cartesian": "RC_3500", "siesta_fc_cartesian": "FC_750"}, sort_keys=True)
            })
            summary = json.loads((root / "summary" / "nway_ranking_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["scientific_status"], "nway_ranking_diagnostic_only")
            self.assertEqual(summary["unstable_group_count"], 0)
            self.assertEqual(summary["missing_method_cell_count"], 0)
            self.assertEqual(summary["robust_candidate_group_count"], 1)

    def test_nway_ranking_marks_unstable_rankings_across_seeds(self) -> None:
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            metrics = root / "cross_evaluation_metrics.csv"
            methods = ("md", "siesta_fc_cartesian", "random_cartesian")
            values_by_seed = {
                "1": {"md": "0.2", "siesta_fc_cartesian": "0.4", "random_cartesian": "0.6"},
                "2": {"md": "0.6", "siesta_fc_cartesian": "0.4", "random_cartesian": "0.2"},
            }
            rows = []
            for seed, values in values_by_seed.items():
                for method in methods:
                    rows.append(
                        {
                            "experiment_id": "exp_nway_unstable",
                            "train_method": method,
                            "test_set": "test_md",
                            "dataset_size_by_method": json.dumps(
                                {"md": 750, "siesta_fc_cartesian": 750, "random_cartesian": 3500}
                            ),
                            "seed": seed,
                            "model_checkpoint": f"{method}_{seed}.ckpt",
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
            with (root / "summary" / "nway_ranking.csv").open(encoding="utf-8") as handle:
                ranking_rows = list(csv.DictReader(handle))
            self.assertEqual({row["ranking_stability_status"] for row in ranking_rows}, {"unstable"})
            self.assertEqual({row["ranking_valid_seed_count"] for row in ranking_rows}, {"2"})
            self.assertEqual({row["ranking_status"] for row in ranking_rows}, {"diagnostic_only_unstable"})
            summary = json.loads((root / "summary" / "nway_ranking_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["exploratory_group_count"], 0)
            self.assertEqual(summary["unstable_group_count"], 1)

    def test_nway_ranking_marks_unstable_rankings_across_three_seeds(self) -> None:
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            metrics = root / "cross_evaluation_metrics.csv"
            methods = ("md", "siesta_fc_cartesian", "random_cartesian")
            values_by_seed = {
                "1": {"md": "0.2", "siesta_fc_cartesian": "0.4", "random_cartesian": "0.6"},
                "2": {"md": "0.6", "siesta_fc_cartesian": "0.4", "random_cartesian": "0.2"},
                "3": {"md": "0.3", "siesta_fc_cartesian": "0.4", "random_cartesian": "0.5"},
            }
            rows = []
            for seed, values in values_by_seed.items():
                for method in methods:
                    rows.append(
                        {
                            "experiment_id": "exp_nway_unstable_three",
                            "train_method": method,
                            "test_set": "test_md",
                            "dataset_size_by_method": json.dumps(
                                {"md": 750, "siesta_fc_cartesian": 750, "random_cartesian": 3500}
                            ),
                            "seed": seed,
                            "model_checkpoint": f"{method}_{seed}.ckpt",
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
            with (root / "summary" / "nway_ranking.csv").open(encoding="utf-8") as handle:
                ranking_rows = list(csv.DictReader(handle))
            self.assertEqual({row["ranking_stability_status"] for row in ranking_rows}, {"unstable"})
            self.assertEqual({row["ranking_status"] for row in ranking_rows}, {"diagnostic_only_unstable"})
            summary = json.loads((root / "summary" / "nway_ranking_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["unstable_group_count"], 1)
            self.assertEqual(summary["unstable_groups"][0]["test_set"], "test_md")

    def test_nway_ranking_handles_ties(self) -> None:
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            metrics = root / "cross_evaluation_metrics.csv"
            values = {"md": "0.3", "siesta_fc_cartesian": "0.3", "random_cartesian": "0.8"}
            rows = []
            for method, value in values.items():
                rows.append(
                    {
                        "experiment_id": "exp_nway_tie",
                        "train_method": method,
                        "test_set": "test_md",
                        "dataset_size_by_method": json.dumps(
                            {"md": 750, "siesta_fc_cartesian": 750, "random_cartesian": 3500}
                        ),
                        "seed": "1",
                        "model_checkpoint": f"{method}.ckpt",
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
            with (root / "summary" / "nway_ranking.csv").open(encoding="utf-8") as handle:
                ranking_rows = list(csv.DictReader(handle))
            tied_rows = [row for row in ranking_rows if row["method"] in {"md", "siesta_fc_cartesian"}]
            self.assertEqual({row["rank"] for row in tied_rows}, {"1"})
            self.assertEqual({row["tie"] for row in tied_rows}, {"True"})
            self.assertEqual({row["leader_tie"] for row in tied_rows}, {"True"})
            self.assertEqual({row["ranking_stability_status"] for row in ranking_rows}, {"exploratory_only"})
            summary = json.loads((root / "summary" / "nway_ranking_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["tie_group_count"], 1)
            self.assertEqual(set(summary["tie_groups"][0]["leader_methods"]), {"md", "siesta_fc_cartesian"})

    def test_nway_ranking_reports_missing_method_in_test_set(self) -> None:
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            metrics = root / "cross_evaluation_metrics.csv"
            size_map = {"md": 750, "siesta_fc_cartesian": 750, "random_cartesian": 3500}
            rows = []
            for method, value in {"md": "0.2", "siesta_fc_cartesian": "0.4", "random_cartesian": "0.6"}.items():
                rows.append(
                    {
                        "experiment_id": "exp_nway_missing",
                        "train_method": method,
                        "test_set": "test_md",
                        "dataset_size_by_method": json.dumps(size_map),
                        "seed": "1",
                        "model_checkpoint": f"{method}.ckpt",
                        "global_rmse_eV": value,
                    }
                )
            for method, value in {"md": "0.2", "siesta_fc_cartesian": "0.4"}.items():
                rows.append(
                    {
                        "experiment_id": "exp_nway_missing",
                        "train_method": method,
                        "test_set": "test_random_cartesian",
                        "dataset_size_by_method": json.dumps(size_map),
                        "seed": "1",
                        "model_checkpoint": f"{method}_random_test.ckpt",
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
            with (root / "summary" / "nway_ranking.csv").open(encoding="utf-8") as handle:
                ranking_rows = list(csv.DictReader(handle))
            missing_rows = [row for row in ranking_rows if row["test_set"] == "test_random_cartesian"]
            self.assertEqual({row["missing_methods"] for row in missing_rows}, {"random_cartesian"})
            self.assertEqual({row["ranking_grid_status"] for row in missing_rows}, {"missing_method"})
            self.assertEqual({row["ranking_status"] for row in missing_rows}, {"diagnostic_only_missing_method"})
            summary = json.loads((root / "summary" / "nway_ranking_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["missing_method_cell_count"], 1)
            self.assertEqual(summary["missing_method_cells"][0]["missing_methods"], ["random_cartesian"])

    def test_pairwise_vs_baseline_reports_random_cartesian_beats_md(self) -> None:
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            metrics = root / "cross_evaluation_metrics.csv"
            rows = []
            values = {"md": "1.0", "random_cartesian": "0.5"}
            for method, value in values.items():
                rows.append(
                    {
                        "experiment_id": "exp_pair_rc",
                        "train_method": method,
                        "test_set": "test_md",
                        "dataset_size_by_method": json.dumps({"md": 750, "random_cartesian": 3500}),
                        "dataset_label_by_method": json.dumps({"md": "MD_750", "random_cartesian": "RC_3500"}),
                        "recipe_hash_by_method": json.dumps({"md": "md_hash", "random_cartesian": "rc_hash"}),
                        "seed": "1",
                        "model_checkpoint": f"{method}.ckpt",
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
            with (root / "summary" / "pairwise_vs_baseline.csv").open(encoding="utf-8") as handle:
                pair_rows = list(csv.DictReader(handle))
            self.assertEqual(len(pair_rows), 1)
            row = pair_rows[0]
            self.assertEqual(row["baseline_method"], "md")
            self.assertEqual(row["challenger_method"], "random_cartesian")
            self.assertEqual(row["winner"], "random_cartesian")
            self.assertEqual(row["baseline_dataset_size"], "750")
            self.assertEqual(row["challenger_dataset_size"], "3500")
            self.assertEqual(row["challenger_dataset_label"], "RC_3500")
            self.assertEqual(row["challenger_recipe_hash"], "rc_hash")
            self.assertGreater(float(row["percent_improvement_challenger_vs_baseline"]), 0.0)
            summary = json.loads((root / "summary" / "pairwise_vs_baseline_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["scientific_status"], "pairwise_diagnostic_only")
            self.assertEqual(summary["wins_by_challenger"]["random_cartesian"], 1)

    def test_pairwise_seed_stability_one_seed_is_exploratory_only(self) -> None:
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            metrics = root / "cross_evaluation_metrics.csv"
            rows = []
            for method, value in {"md": "1.0", "random_cartesian": "0.5"}.items():
                rows.append(
                    {
                        "experiment_id": "exp_pair_seed_one",
                        "train_method": method,
                        "test_set": "test_md",
                        "dataset_size_by_method": json.dumps({"md": 750, "random_cartesian": 3500}),
                        "seed": "1",
                        "model_checkpoint": f"{method}.ckpt",
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
            with (root / "summary" / "pairwise_vs_baseline.csv").open(encoding="utf-8") as handle:
                pair_rows = list(csv.DictReader(handle))
            self.assertEqual({row["seed_stability_status"] for row in pair_rows}, {"exploratory_only"})
            self.assertEqual({row["robust_candidate"] for row in pair_rows}, {"False"})
            summary = json.loads((root / "summary" / "pairwise_vs_baseline_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["robust_candidate_count"], 0)
            self.assertEqual(summary["exploratory_group_count"], 1)

    def test_pairwise_seed_stability_three_seed_rc_win_is_robust_candidate(self) -> None:
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            metrics = root / "cross_evaluation_metrics.csv"
            rows = []
            for seed in ("1", "2", "3"):
                for method, value in {"md": "1.0", "random_cartesian": "0.5"}.items():
                    rows.append(
                        {
                            "experiment_id": "exp_pair_seed_robust",
                            "train_method": method,
                            "test_set": "test_md",
                            "dataset_size_by_method": json.dumps({"md": 750, "random_cartesian": 3500}),
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
            with (root / "summary" / "pairwise_vs_baseline.csv").open(encoding="utf-8") as handle:
                pair_rows = list(csv.DictReader(handle))
            self.assertEqual({row["winner"] for row in pair_rows}, {"random_cartesian"})
            self.assertEqual({row["seed_stability_status"] for row in pair_rows}, {"robust_candidate"})
            self.assertEqual({row["seed_stability_valid_n_seeds"] for row in pair_rows}, {"3"})
            summary = json.loads((root / "summary" / "pairwise_vs_baseline_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["robust_candidate_count"], 1)
            self.assertEqual(summary["robust_candidates"][0]["winner"], "random_cartesian")

    def test_pairwise_seed_stability_three_seed_split_is_unstable(self) -> None:
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            metrics = root / "cross_evaluation_metrics.csv"
            rows = []
            values_by_seed = {
                "1": {"md": "1.0", "random_cartesian": "0.5"},
                "2": {"md": "1.0", "random_cartesian": "0.5"},
                "3": {"md": "0.4", "random_cartesian": "0.8"},
            }
            for seed, values in values_by_seed.items():
                for method, value in values.items():
                    rows.append(
                        {
                            "experiment_id": "exp_pair_seed_unstable",
                            "train_method": method,
                            "test_set": "test_md",
                            "dataset_size_by_method": json.dumps({"md": 750, "random_cartesian": 3500}),
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
            with (root / "summary" / "pairwise_vs_baseline.csv").open(encoding="utf-8") as handle:
                pair_rows = list(csv.DictReader(handle))
            self.assertEqual({row["seed_stability_status"] for row in pair_rows}, {"unstable"})
            self.assertEqual({row["robust_candidate"] for row in pair_rows}, {"False"})
            summary = json.loads((root / "summary" / "pairwise_vs_baseline_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["unstable_group_count"], 1)
            self.assertEqual(summary["robust_candidate_count"], 0)

    def test_pairwise_seed_stability_missing_seed_is_exploratory_only(self) -> None:
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            metrics = root / "cross_evaluation_metrics.csv"
            rows = []
            for method, value in {"md": "1.0", "random_cartesian": "0.5"}.items():
                rows.append(
                    {
                        "experiment_id": "exp_pair_seed_missing",
                        "train_method": method,
                        "test_set": "test_md",
                        "dataset_size_by_method": json.dumps({"md": 750, "random_cartesian": 3500}),
                        "model_checkpoint": f"{method}.ckpt",
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
            with (root / "summary" / "pairwise_vs_baseline.csv").open(encoding="utf-8") as handle:
                pair_rows = list(csv.DictReader(handle))
            self.assertEqual({row["seed"] for row in pair_rows}, {"unknown"})
            self.assertEqual({row["seed_stability_status"] for row in pair_rows}, {"exploratory_only"})
            self.assertEqual({row["seed_stability_valid_n_seeds"] for row in pair_rows}, {"0"})

    def test_pairwise_vs_baseline_reports_fc_beats_md(self) -> None:
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            metrics = root / "cross_evaluation_metrics.csv"
            rows = []
            values = {"md": "1.0", "siesta_fc_cartesian": "0.25"}
            for method, value in values.items():
                rows.append(
                    {
                        "experiment_id": "exp_pair_fc",
                        "train_method": method,
                        "test_set": "test_md",
                        "dataset_size_by_method": json.dumps({"md": 750, "siesta_fc_cartesian": 750}),
                        "seed": "1",
                        "model_checkpoint": f"{method}.ckpt",
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
            with (root / "summary" / "pairwise_vs_baseline.csv").open(encoding="utf-8") as handle:
                pair_rows = list(csv.DictReader(handle))
            self.assertEqual(len(pair_rows), 1)
            row = pair_rows[0]
            self.assertEqual(row["challenger_method"], "siesta_fc_cartesian")
            self.assertEqual(row["winner"], "siesta_fc_cartesian")
            self.assertEqual(row["lower_is_better"], "True")
            self.assertLess(float(row["absolute_difference_challenger_minus_baseline"]), 0.0)

    def test_pairwise_vs_baseline_reports_md_beats_both_challengers(self) -> None:
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            metrics = root / "cross_evaluation_metrics.csv"
            rows = []
            values = {"md": "0.4", "siesta_fc_cartesian": "0.8", "random_cartesian": "1.2"}
            for method, value in values.items():
                rows.append(
                    {
                        "experiment_id": "exp_pair_md",
                        "train_method": method,
                        "test_set": "test_md",
                        "dataset_size_by_method": json.dumps(
                            {"md": 750, "siesta_fc_cartesian": 750, "random_cartesian": 3500}
                        ),
                        "seed": "1",
                        "model_checkpoint": f"{method}.ckpt",
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
            with (root / "summary" / "pairwise_vs_baseline.csv").open(encoding="utf-8") as handle:
                pair_rows = list(csv.DictReader(handle))
            self.assertEqual({row["challenger_method"] for row in pair_rows}, {"siesta_fc_cartesian", "random_cartesian"})
            self.assertEqual({row["winner"] for row in pair_rows}, {"md"})
            summary = json.loads((root / "summary" / "pairwise_vs_baseline_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["wins_by_baseline"]["siesta_fc_cartesian"], 1)
            self.assertEqual(summary["wins_by_baseline"]["random_cartesian"], 1)

    def test_pairwise_vs_baseline_marks_rc_own_distribution_win_as_specific(self) -> None:
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            metrics = root / "cross_evaluation_metrics.csv"
            rows = []
            values = {
                ("md", "test_md"): "0.4",
                ("random_cartesian", "test_md"): "0.9",
                ("md", "test_random_cartesian"): "1.0",
                ("random_cartesian", "test_random_cartesian"): "0.3",
            }
            for (method, test_set), value in values.items():
                rows.append(
                    {
                        "experiment_id": "exp_pair_rc_specific",
                        "train_method": method,
                        "test_set": test_set,
                        "dataset_size_by_method": json.dumps({"md": 750, "random_cartesian": 3500}),
                        "seed": "1",
                        "model_checkpoint": f"{method}.ckpt",
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
            with (root / "summary" / "pairwise_vs_baseline.csv").open(encoding="utf-8") as handle:
                pair_rows = list(csv.DictReader(handle))
            own_distribution_row = next(row for row in pair_rows if row["test_set"] == "test_random_cartesian")
            md_distribution_row = next(row for row in pair_rows if row["test_set"] == "test_md")
            self.assertEqual(own_distribution_row["winner"], "random_cartesian")
            self.assertEqual(own_distribution_row["distribution_status"], "distribution_specific")
            self.assertEqual(own_distribution_row["win_scope_status"], "distribution_specific_only")
            self.assertEqual(md_distribution_row["winner"], "md")
            self.assertEqual(md_distribution_row["win_scope_status"], "distribution_specific_only")
            summary = json.loads((root / "summary" / "pairwise_vs_baseline_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["distribution_specific_only_count"], 1)
            self.assertEqual(summary["distribution_specific_only"][0]["challenger_method"], "random_cartesian")

    def test_dataset_size_thresholds_reports_rc_threshold_at_3500(self) -> None:
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            metrics = root / "cross_evaluation_metrics.csv"
            rows = []
            values_by_size = {
                1500: {"md": "0.5", "random_cartesian": "0.8"},
                3500: {"md": "0.8", "random_cartesian": "0.4"},
            }
            for rc_size, values in values_by_size.items():
                for seed in ("1", "2", "3"):
                    for method, value in values.items():
                        rows.append(
                            {
                                "experiment_id": "exp_threshold_rc",
                                "train_method": method,
                                "test_set": "test_md",
                                "dataset_size_by_method": json.dumps({"md": 750, "random_cartesian": rc_size}),
                                "dataset_label_by_method": json.dumps(
                                    {"md": "MD_750", "random_cartesian": f"RC_{rc_size}"}
                                ),
                                "recipe_hash_by_method": json.dumps(
                                    {"md": "md_hash", "random_cartesian": f"rc_hash_{rc_size}"}
                                ),
                                "seed": seed,
                                "model_checkpoint": f"{method}_{rc_size}_{seed}.ckpt",
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
            threshold_summary = json.loads(
                (root / "summary" / "dataset_size_thresholds_vs_md.json").read_text(encoding="utf-8")
            )
            rc_thresholds = [
                row
                for row in threshold_summary["thresholds"]
                if row["challenger_method"] == "random_cartesian"
            ]
            self.assertEqual(len(rc_thresholds), 1)
            self.assertEqual(rc_thresholds[0]["first_stable_dataset_size"], 3500)
            self.assertEqual(rc_thresholds[0]["corresponding_md_dataset_size"], 750)
            self.assertEqual(rc_thresholds[0]["challenger_recipe_hash"], "rc_hash_3500")
            self.assertEqual(rc_thresholds[0]["stability_status"], "robust_candidate")
            self.assertEqual(rc_thresholds[0]["distribution_scope"], "general")
            with (root / "summary" / "dataset_size_thresholds_vs_md.csv").open(encoding="utf-8") as handle:
                threshold_rows = list(csv.DictReader(handle))
            available = [row for row in threshold_rows if row["threshold_available"] == "True"]
            self.assertEqual(len(available), 1)
            self.assertEqual(available[0]["first_stable_dataset_size"], "3500")

    def test_dataset_size_thresholds_reports_fc_threshold_at_750(self) -> None:
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            metrics = root / "cross_evaluation_metrics.csv"
            rows = []
            for seed in ("1", "2", "3"):
                for method, value in {"md": "1.0", "siesta_fc_cartesian": "0.5"}.items():
                    rows.append(
                        {
                            "experiment_id": "exp_threshold_fc",
                            "train_method": method,
                            "test_set": "test_md",
                            "dataset_size_by_method": json.dumps({"md": 750, "siesta_fc_cartesian": 750}),
                            "dataset_label_by_method": json.dumps(
                                {"md": "MD_750", "siesta_fc_cartesian": "FC_750"}
                            ),
                            "recipe_hash_by_method": json.dumps(
                                {"md": "md_hash", "siesta_fc_cartesian": "fc_hash"}
                            ),
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
            threshold_summary = json.loads(
                (root / "summary" / "dataset_size_thresholds_vs_md.json").read_text(encoding="utf-8")
            )
            fc_thresholds = [
                row
                for row in threshold_summary["thresholds"]
                if row["challenger_method"] == "siesta_fc_cartesian"
            ]
            self.assertEqual(len(fc_thresholds), 1)
            self.assertEqual(fc_thresholds[0]["first_stable_dataset_size"], 750)
            self.assertEqual(fc_thresholds[0]["challenger_dataset_label"], "FC_750")
            self.assertEqual(fc_thresholds[0]["challenger_recipe_hash"], "fc_hash")

    def test_dataset_size_thresholds_rejects_rc_own_distribution_only_win(self) -> None:
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            metrics = root / "cross_evaluation_metrics.csv"
            rows = []
            values = {
                ("md", "test_md"): "0.4",
                ("random_cartesian", "test_md"): "0.9",
                ("md", "test_random_cartesian"): "1.0",
                ("random_cartesian", "test_random_cartesian"): "0.3",
            }
            for seed in ("1", "2", "3"):
                for (method, test_set), value in values.items():
                    rows.append(
                        {
                            "experiment_id": "exp_threshold_rc_specific",
                            "train_method": method,
                            "test_set": test_set,
                            "dataset_size_by_method": json.dumps({"md": 750, "random_cartesian": 3500}),
                            "seed": seed,
                            "model_checkpoint": f"{method}_{test_set}_{seed}.ckpt",
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
            threshold_summary = json.loads(
                (root / "summary" / "dataset_size_thresholds_vs_md.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                [row for row in threshold_summary["thresholds"] if row["challenger_method"] == "random_cartesian"],
                [],
            )
            own_distribution_rejections = [
                row
                for row in threshold_summary["unavailable_thresholds"]
                if row["challenger_method"] == "random_cartesian"
                and row["test_set"] == "test_random_cartesian"
            ]
            self.assertEqual(len(own_distribution_rejections), 1)
            self.assertIn("distribution-specific", own_distribution_rejections[0]["reason"])
            self.assertEqual(own_distribution_rejections[0]["distribution_scope"], "distribution_specific")

    def test_dataset_size_thresholds_rejects_one_seed_win(self) -> None:
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            metrics = root / "cross_evaluation_metrics.csv"
            rows = []
            for method, value in {"md": "1.0", "random_cartesian": "0.5"}.items():
                rows.append(
                    {
                        "experiment_id": "exp_threshold_one_seed",
                        "train_method": method,
                        "test_set": "test_md",
                        "dataset_size_by_method": json.dumps({"md": 750, "random_cartesian": 3500}),
                        "seed": "1",
                        "model_checkpoint": f"{method}.ckpt",
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
            threshold_summary = json.loads(
                (root / "summary" / "dataset_size_thresholds_vs_md.json").read_text(encoding="utf-8")
            )
            self.assertEqual(threshold_summary["thresholds"], [])
            unavailable = threshold_summary["unavailable_thresholds"]
            self.assertEqual(len(unavailable), 1)
            self.assertEqual(unavailable[0]["challenger_method"], "random_cartesian")
            self.assertEqual(unavailable[0]["stability_status"], "exploratory_only")
            self.assertIn("Fewer than 3 valid seeds", unavailable[0]["reason"])

    def test_compute_budget_thresholds_compute_with_reliable_total_timing(self) -> None:
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            metrics = root / "cross_evaluation_metrics.csv"
            output_dir = root / "summary"
            rows = []
            scenarios = {
                1500: {"values": {"md": "0.5", "random_cartesian": "0.8"}, "times": {"md": "100", "random_cartesian": "90"}},
                3500: {"values": {"md": "0.8", "random_cartesian": "0.4"}, "times": {"md": "100", "random_cartesian": "200"}},
            }
            for rc_size, scenario in scenarios.items():
                for seed in ("1", "2", "3"):
                    for method, value in scenario["values"].items():
                        rows.append(
                            {
                                "experiment_id": "exp_compute_threshold_rc",
                                "train_method": method,
                                "test_set": "test_md",
                                "dataset_size_by_method": json.dumps({"md": 750, "random_cartesian": rc_size}),
                                "seed": seed,
                                "model_checkpoint": f"{method}_{rc_size}_{seed}.ckpt",
                                "total_time_seconds": scenario["times"][method],
                                "siesta_time_seconds": "40",
                                "training_time_seconds": "50",
                                "prediction_time_seconds": "5",
                                "evaluation_time_seconds": "5",
                                "global_rmse_eV": value,
                            }
                        )
            write_csv(metrics, rows)
            self.write_completeness_report(output_dir / "cross_evaluation_completeness.json", ("md", "random_cartesian"), ("test_md",))
            result = run_script(
                "Comparison/scripts/analyze_winners.py",
                "--metrics-csv",
                str(metrics),
                "--output-dir",
                str(output_dir),
                "--primary-metric",
                "global_rmse_eV",
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            compute_summary = json.loads((output_dir / "compute_budget_thresholds_vs_md.json").read_text(encoding="utf-8"))
            rc_thresholds = [
                row
                for row in compute_summary["thresholds"]
                if row["challenger_method"] == "random_cartesian"
            ]
            self.assertEqual(len(rc_thresholds), 1)
            self.assertEqual(rc_thresholds[0]["first_stable_compute_budget_seconds"], 200.0)
            self.assertEqual(rc_thresholds[0]["challenger_compute_budget_seconds"], 200.0)
            self.assertEqual(rc_thresholds[0]["baseline_compute_budget_seconds"], 100.0)
            self.assertEqual(rc_thresholds[0]["compute_budget_timing_source"], "total_time_seconds")
            self.assertEqual(rc_thresholds[0]["stability_status"], "robust_candidate")
            with (output_dir / "compute_budget_thresholds_vs_md.csv").open(encoding="utf-8") as handle:
                threshold_rows = list(csv.DictReader(handle))
            available = [row for row in threshold_rows if row["compute_threshold_available"] == "True"]
            self.assertEqual(len(available), 1)
            self.assertEqual(available[0]["first_stable_compute_budget_seconds"], "200.0")

    def test_compute_budget_thresholds_missing_timing_fields_do_not_fake_threshold(self) -> None:
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            metrics = root / "cross_evaluation_metrics.csv"
            output_dir = root / "summary"
            rows = []
            for seed in ("1", "2", "3"):
                for method, value in {"md": "1.0", "siesta_fc_cartesian": "0.4"}.items():
                    rows.append(
                        {
                            "experiment_id": "exp_compute_missing_timing",
                            "train_method": method,
                            "test_set": "test_md",
                            "dataset_size_by_method": json.dumps({"md": 750, "siesta_fc_cartesian": 750}),
                            "seed": seed,
                            "model_checkpoint": f"{method}_{seed}.ckpt",
                            "global_rmse_eV": value,
                        }
                    )
            write_csv(metrics, rows)
            self.write_completeness_report(
                output_dir / "cross_evaluation_completeness.json",
                ("md", "siesta_fc_cartesian"),
                ("test_md",),
            )
            result = run_script(
                "Comparison/scripts/analyze_winners.py",
                "--metrics-csv",
                str(metrics),
                "--output-dir",
                str(output_dir),
                "--primary-metric",
                "global_rmse_eV",
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            compute_summary = json.loads((output_dir / "compute_budget_thresholds_vs_md.json").read_text(encoding="utf-8"))
            self.assertEqual(compute_summary["thresholds"], [])
            self.assertEqual(len(compute_summary["unavailable_thresholds"]), 1)
            unavailable = compute_summary["unavailable_thresholds"][0]
            self.assertEqual(unavailable["challenger_method"], "siesta_fc_cartesian")
            self.assertEqual(unavailable["compute_threshold_unavailable"], "missing reliable timing fields")
            self.assertEqual(unavailable["compute_threshold_timing_status"], "missing_reliable_timing_fields")

    def test_compute_budget_thresholds_missing_rc_timing_preserves_dataset_threshold(self) -> None:
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            metrics = root / "cross_evaluation_metrics.csv"
            output_dir = root / "summary"
            rows = []
            for seed in ("1", "2", "3"):
                for method, value in {"md": "1.0", "random_cartesian": "0.4"}.items():
                    row = {
                        "experiment_id": "exp_compute_rc_missing_timing",
                        "train_method": method,
                        "test_set": "test_md",
                        "dataset_size_by_method": json.dumps({"md": 750, "random_cartesian": 3500}),
                        "seed": seed,
                        "model_checkpoint": f"{method}_{seed}.ckpt",
                        "global_rmse_eV": value,
                    }
                    if method == "md":
                        row["total_time_seconds"] = "100"
                    rows.append(row)
            write_csv(metrics, rows)
            self.write_completeness_report(output_dir / "cross_evaluation_completeness.json", ("md", "random_cartesian"), ("test_md",))
            result = run_script(
                "Comparison/scripts/analyze_winners.py",
                "--metrics-csv",
                str(metrics),
                "--output-dir",
                str(output_dir),
                "--primary-metric",
                "global_rmse_eV",
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            dataset_summary = json.loads((output_dir / "dataset_size_thresholds_vs_md.json").read_text(encoding="utf-8"))
            compute_summary = json.loads((output_dir / "compute_budget_thresholds_vs_md.json").read_text(encoding="utf-8"))
            self.assertEqual(len(dataset_summary["thresholds"]), 1)
            self.assertEqual(dataset_summary["thresholds"][0]["challenger_method"], "random_cartesian")
            self.assertEqual(dataset_summary["thresholds"][0]["first_stable_dataset_size"], 3500)
            self.assertEqual(compute_summary["thresholds"], [])
            self.assertEqual(compute_summary["unavailable_thresholds"][0]["compute_threshold_unavailable"], "missing reliable timing fields")

    def test_winner_analysis_keeps_random_cartesian_sizes_separate_in_context(self) -> None:
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            metrics = root / "cross_evaluation_metrics.csv"
            rows = []
            methods = ("md", "siesta_fc_cartesian", "random_cartesian")
            values = {"md": "0.4", "siesta_fc_cartesian": "0.8", "random_cartesian": "1.2"}
            for rc_size in (1500, 3500):
                size_map = {"md": 750, "siesta_fc_cartesian": 750, "random_cartesian": rc_size}
                label_map = {
                    "md": "MD_750",
                    "siesta_fc_cartesian": "FC_750",
                    "random_cartesian": f"RC_{rc_size}",
                }
                for method in methods:
                    rows.append(
                        {
                            "experiment_id": "exp_rc_sizes",
                            "train_method": method,
                            "test_set": "test_md",
                            "dataset_size_by_method": json.dumps(size_map),
                            "dataset_label_by_method": json.dumps(label_map),
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
            with (root / "summary" / "nway_method_summary.csv").open(encoding="utf-8") as handle:
                method_rows = list(csv.DictReader(handle))
            rc_rows = [row for row in method_rows if row["method"] == "random_cartesian"]
            self.assertEqual({row["dataset_size"] for row in rc_rows}, {"1500", "3500"})
            self.assertEqual(len({row["dataset_context_key"] for row in method_rows}), 2)
            with (root / "summary" / "nway_ranking.csv").open(encoding="utf-8") as handle:
                ranking_rows = list(csv.DictReader(handle))
            self.assertEqual(len({row["dataset_context_key"] for row in ranking_rows}), 2)
            self.assertEqual({row["n_methods_ranked"] for row in ranking_rows}, {"3"})
            with (root / "summary" / "winner_by_dataset_size.csv").open(encoding="utf-8") as handle:
                pair_rows = list(csv.DictReader(handle))
            self.assertEqual(len(pair_rows), 2)
            self.assertEqual(len({row["dataset_context_key"] for row in pair_rows}), 2)

    def test_winner_analysis_keeps_same_size_random_cartesian_recipe_hashes_separate(self) -> None:
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            metrics = root / "cross_evaluation_metrics.csv"
            rows = []
            methods = ("md", "siesta_fc_cartesian", "random_cartesian")
            size_map = {"md": 750, "siesta_fc_cartesian": 750, "random_cartesian": 3500}
            values = {"md": "0.4", "siesta_fc_cartesian": "0.8", "random_cartesian": "1.2"}
            for rc_hash in ("rc_hash_a", "rc_hash_b"):
                hash_map = {
                    "md": "md_hash",
                    "siesta_fc_cartesian": "fc_hash",
                    "random_cartesian": rc_hash,
                }
                for method in methods:
                    rows.append(
                        {
                            "experiment_id": "exp_rc_hashes",
                            "train_method": method,
                            "test_set": "test_md",
                            "dataset_size_by_method": json.dumps(size_map),
                            "recipe_hash_by_method": json.dumps(hash_map),
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
            with (root / "summary" / "nway_method_summary.csv").open(encoding="utf-8") as handle:
                method_rows = list(csv.DictReader(handle))
            rc_rows = [row for row in method_rows if row["method"] == "random_cartesian"]
            self.assertEqual({row["dataset_size"] for row in rc_rows}, {"3500"})
            self.assertEqual({row["recipe_hash"] for row in rc_rows}, {"rc_hash_a", "rc_hash_b"})
            self.assertEqual(len({row["dataset_context_key"] for row in method_rows}), 2)
            with (root / "summary" / "winner_by_dataset_size.csv").open(encoding="utf-8") as handle:
                pair_rows = list(csv.DictReader(handle))
            self.assertEqual(len(pair_rows), 2)
            self.assertEqual(len({row["dataset_context_key"] for row in pair_rows}), 2)

    def test_winner_analysis_valid_completeness_report_allows_nway_analysis(self) -> None:
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            metrics = root / "cross_evaluation_metrics.csv"
            output_dir = root / "summary"
            methods = ("md", "siesta_fc_cartesian", "random_cartesian")
            test_sets = ("test_md", "test_siesta_fc_cartesian", "test_random_cartesian", "test_mixed")
            values = {"md": "0.4", "siesta_fc_cartesian": "0.8", "random_cartesian": "1.2"}
            rows = []
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
            self.write_completeness_report(
                output_dir / "cross_evaluation_completeness.json",
                methods,
                test_sets,
            )
            result = run_script(
                "Comparison/scripts/analyze_winners.py",
                "--metrics-csv",
                str(metrics),
                "--output-dir",
                str(output_dir),
                "--primary-metric",
                "global_rmse_eV",
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            recommendation = json.loads((output_dir / "recommendation.json").read_text(encoding="utf-8"))
            self.assertEqual(recommendation["status"], "insufficient_seeds")
            self.assertEqual(recommendation["scientific_status"], "exploratory_only")
            self.assertEqual(recommendation["legacy_recommendation"]["status"], "nway_consensus_win")
            self.assertEqual(recommendation["nway_consensus_leader"], "md")

    def test_winner_analysis_invalid_completeness_report_blocks_recommendation(self) -> None:
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            metrics = root / "cross_evaluation_metrics.csv"
            output_dir = root / "summary"
            methods = ("md", "siesta_fc_cartesian", "random_cartesian")
            test_sets = ("test_md", "test_siesta_fc_cartesian", "test_random_cartesian")
            values = {"md": "0.4", "siesta_fc_cartesian": "0.8", "random_cartesian": "1.2"}
            rows = []
            for method in methods:
                for test_set in test_sets:
                    rows.append(
                        {
                            "experiment_id": "exp_nway_invalid",
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
            self.write_completeness_report(
                output_dir / "cross_evaluation_completeness.json",
                methods,
                test_sets,
                missing_cells=("random_cartesian on test_md",),
            )
            result = run_script(
                "Comparison/scripts/analyze_winners.py",
                "--metrics-csv",
                str(metrics),
                "--output-dir",
                str(output_dir),
                "--primary-metric",
                "global_rmse_eV",
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            recommendation = json.loads((output_dir / "recommendation.json").read_text(encoding="utf-8"))
            self.assertEqual(recommendation["status"], "invalid_incomplete_grid")
            self.assertEqual(recommendation["scientific_status"], "not_scientifically_valid")
            self.assertIsNone(recommendation["winner"])
            self.assertEqual(recommendation["reason"], "Incomplete 3-method cross-evaluation grid")
            self.assertIn("random_cartesian on test_md", recommendation["missing_cells"])
            self.assertNotIn(recommendation["scientific_status"], {"robust_comparison", "nway_consensus_win"})
            threshold_summary = json.loads(
                (output_dir / "dataset_size_thresholds_vs_md.json").read_text(encoding="utf-8")
            )
            self.assertEqual(threshold_summary["scientific_status"], "invalid_incomplete_grid")
            self.assertEqual(threshold_summary["thresholds"], [])
            compute_threshold_summary = json.loads(
                (output_dir / "compute_budget_thresholds_vs_md.json").read_text(encoding="utf-8")
            )
            self.assertEqual(compute_threshold_summary["scientific_status"], "invalid_incomplete_grid")
            self.assertEqual(compute_threshold_summary["thresholds"], [])

    def test_winner_analysis_missing_primary_metric_completeness_blocks_recommendation(self) -> None:
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            metrics = root / "cross_evaluation_metrics.csv"
            output_dir = root / "summary"
            methods = ("md", "siesta_fc_cartesian", "random_cartesian")
            test_sets = ("test_md", "test_siesta_fc_cartesian", "test_random_cartesian")
            rows = []
            for method in methods:
                for test_set in test_sets:
                    row = {
                        "experiment_id": "exp_missing_metric",
                        "train_method": method,
                        "test_set": test_set,
                        "test_method": test_set.removeprefix("test_"),
                        "seed": "1",
                        "model_checkpoint": f"{method}.ckpt",
                    }
                    if not (method == "md" and test_set == "test_random_cartesian"):
                        row["global_rmse_eV"] = "0.4" if method == "md" else "0.8"
                    rows.append(row)
            write_csv(metrics, rows)
            self.write_completeness_report(
                output_dir / "cross_evaluation_completeness.json",
                methods,
                test_sets,
                missing_primary_metric_cells=("md on test_random_cartesian",),
            )
            result = run_script(
                "Comparison/scripts/analyze_winners.py",
                "--metrics-csv",
                str(metrics),
                "--output-dir",
                str(output_dir),
                "--primary-metric",
                "global_rmse_eV",
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            recommendation = json.loads((output_dir / "recommendation.json").read_text(encoding="utf-8"))
            self.assertEqual(recommendation["scientific_status"], "not_scientifically_valid")
            self.assertEqual(recommendation["status"], "insufficient_primary_metric")
            self.assertIsNone(recommendation["winner"])
            self.assertIn("md on test_random_cartesian", recommendation["missing_cells"])
            self.assertIn("md on test_random_cartesian", recommendation["missing_primary_metric_cells"])

    def test_winner_analysis_two_method_complete_completeness_report_remains_compatible(self) -> None:
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            metrics = root / "cross_evaluation_metrics.csv"
            output_dir = root / "summary"
            methods = ("md", "siesta_fc_cartesian")
            test_sets = ("test_md", "test_siesta_fc_cartesian", "test_mixed")
            rows = []
            for seed in ("1", "2", "3"):
                for method, value in (("md", "0.5"), ("siesta_fc_cartesian", "1.0")):
                    for test_set in test_sets:
                        rows.append(
                            {
                                "experiment_id": "exp_two_method",
                                "train_method": method,
                                "test_set": test_set,
                                "md_dataset_size": "10",
                                "atom_dataset_size": "10",
                                "seed": seed,
                                "model_checkpoint": f"{method}_{seed}.ckpt",
                                "low_energy_rmse_eV": value,
                            }
                        )
            write_csv(metrics, rows)
            self.write_completeness_report(
                output_dir / "cross_evaluation_completeness.json",
                methods,
                test_sets,
                primary_metric="low_energy_rmse_eV",
            )
            result = run_script(
                "Comparison/scripts/analyze_winners.py",
                "--metrics-csv",
                str(metrics),
                "--output-dir",
                str(output_dir),
                "--primary-metric",
                "low_energy_rmse_eV",
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            recommendation = json.loads((output_dir / "recommendation.json").read_text(encoding="utf-8"))
            self.assertEqual(recommendation["status"], "md_conservative_win")
            self.assertEqual(recommendation["scientific_status"], "robust_comparison")
            self.assertNotEqual(recommendation["status"], "invalid_incomplete_grid")
            with (output_dir / "nway_method_summary.csv").open(encoding="utf-8") as handle:
                method_rows = list(csv.DictReader(handle))
            self.assertEqual({row["method"] for row in method_rows}, {"md", "siesta_fc_cartesian"})
            self.assertNotIn("random_cartesian", {row["method"] for row in method_rows})
            with (output_dir / "nway_ranking.csv").open(encoding="utf-8") as handle:
                ranking_rows = list(csv.DictReader(handle))
            self.assertEqual({row["n_methods_ranked"] for row in ranking_rows}, {"2"})

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
            self.assertEqual(summary["scientific_status"], "invalid_leakage")
            self.assertEqual(summary["leakage_status_detail"], "invalid_exact_geometry_leakage")
            self.assertTrue(any("exact_duplicate_geometry" in warning for warning in summary["severe_warnings"]))

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
            self.assertEqual(summary["scientific_status"], "scientifically_inconclusive")
            self.assertEqual(summary["leakage_status_detail"], "potential_geometry_leakage")
            self.assertEqual(summary["severe_warnings"], [])

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

    def test_geometry_leakage_detects_same_random_cartesian_family(self) -> None:
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
            self.assertEqual(result.returncode, 2)
            summary = json.loads((root / "leakage" / "geometry_leakage_summary.json").read_text(encoding="utf-8"))
            self.assertGreaterEqual(summary["random_cartesian_family_warnings"], 1)
            self.assertEqual(summary["scientific_status"], "exploratory_only")
            self.assertEqual(summary["leakage_status_detail"], "random_cartesian_same_family_cross_split")
            self.assertFalse(summary["scientifically_independent"])

    def test_geometry_leakage_allows_different_random_cartesian_families(self) -> None:
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            train = make_sample(root / "samples", "train")
            test = make_sample(root / "samples", "test")
            text = (test / "RUN.fdf").read_text(encoding="utf-8").replace("0.0 0.7 0.0 2", "0.0 1.5 0.0 2")
            (test / "RUN.fdf").write_text(text, encoding="utf-8")
            train_metadata = train / "metadata.json"
            test_metadata = test / "metadata.json"
            common = {
                "base_geometry_hash": "base",
                "distribution": "gaussian",
                "seed_family": 1234,
                "move_atoms": "all",
                "species_filter": [],
                "recipe_id": "rc_recipe",
            }
            train_metadata.write_text(
                json.dumps({**common, "id": "sample_000001", "sigma_ang": 0.03, "block_id": "small"}),
                encoding="utf-8",
            )
            test_metadata.write_text(
                json.dumps({**common, "id": "sample_000002", "sigma_ang": 0.05, "block_id": "large"}),
                encoding="utf-8",
            )
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
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            summary = json.loads((root / "leakage" / "geometry_leakage_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["random_cartesian_family_warnings"], 0)
            self.assertEqual(summary["scientific_status"], "valid_independent_splits")

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

    def test_cross_prediction_matrix_writer_partial_error_fails_even_if_output_exists(self) -> None:
        module = self.load_module_from_path(
            "predict_model_on_dataset_strict_writer",
            REPO_ROOT / "Comparison" / "scripts" / "predict_model_on_dataset.py",
        )

        with workspace_tempdir() as tmp:
            output = Path(tmp) / "ML_prediction.HSX"
            output.write_bytes(b"partial prediction")

            class PartialWriterBase:
                def __init__(self, output_file: Path, splits: list[str]) -> None:
                    self.output_file = Path(output_file)
                    self.splits = splits

                def _get_out_file(self, _example, _trainer):
                    return self.output_file

                def _on_batch_end(self, split, trainer, pl_module, prediction, batch, batch_idx, dataloader_idx):
                    raise ValueError(module.EDGE_LABEL_CONSUMPTION_ERROR)

            class Batch:
                num_graphs = 1

                def get_example(self, _index):
                    return object()

            writer_cls = module.strict_matrix_writer_class(PartialWriterBase)
            writer = writer_cls(output_file=output, splits=["predict"])
            with self.assertRaisesRegex(RuntimeError, "incomplete prediction export"):
                writer._on_batch_end("predict", object(), object(), object(), Batch(), 0, 0)

    def test_cross_prediction_command_fails_after_partial_export_error(self) -> None:
        module = self.load_module_from_path(
            "predict_model_on_dataset_partial_main",
            REPO_ROOT / "Comparison" / "scripts" / "predict_model_on_dataset.py",
        )

        with workspace_tempdir() as tmp:
            root = Path(tmp)
            checkpoint = root / "training" / "lightning_logs" / "version_0" / "checkpoints" / "best.ckpt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"checkpoint")
            sample = make_sample(root / "samples", "001")
            manifest = root / "test_manifest.csv"
            write_csv(manifest, [{"sample_id": "001", "structure_path": str(sample / "RUN.fdf")}])
            output_dir = root / "predictions"

            def fake_run_prediction(args, _predict_glob):
                prediction = args.output_dir / "workspace" / "predict_structures" / "001" / "ML_prediction.HSX"
                prediction.write_bytes(b"partial prediction")
                raise module.incomplete_prediction_error(ValueError(module.EDGE_LABEL_CONSUMPTION_ERROR))

            original_run_prediction = module.run_prediction
            original_argv = sys.argv[:]
            try:
                module.run_prediction = fake_run_prediction
                sys.argv = [
                    "predict_model_on_dataset.py",
                    "--checkpoint",
                    str(checkpoint),
                    "--train-method",
                    "md",
                    "--test-set",
                    "test_md",
                    "--test-manifest",
                    str(manifest),
                    "--output-dir",
                    str(output_dir),
                    "--basis-files",
                    "basis/*.ion.xml",
                ]
                status = module.main()
            finally:
                module.run_prediction = original_run_prediction
                sys.argv = original_argv

            self.assertEqual(status, 2)
            summary = json.loads((output_dir / "prediction_summary.json").read_text(encoding="utf-8"))
            self.assertFalse(summary["ok"])
            self.assertEqual(summary["status"], "failed")
            self.assertEqual(summary["predicted_samples"], 1)
            self.assertFalse(summary["prediction_output_validation"]["validated"])
            self.assertIn("incomplete prediction export", summary["error"])
            rows = module.read_rows(output_dir / "prediction_manifest.csv")
            self.assertEqual(rows[0]["prediction_output_validated"], "False")

    def test_cross_prediction_existing_output_alone_fails_validation(self) -> None:
        module = self.load_module_from_path(
            "predict_model_on_dataset_existing_only",
            REPO_ROOT / "Comparison" / "scripts" / "predict_model_on_dataset.py",
        )

        with workspace_tempdir() as tmp:
            root = Path(tmp)
            output = root / "out" / "predicted_hamiltonians" / "001" / "ML_prediction.HSX"
            output.parent.mkdir(parents=True)
            output.write_bytes(b"prediction")
            rows = [{"sample_id": "001", "status": "predicted", "prediction_path": str(output)}]
            with self.assertRaisesRegex(RuntimeError, "missing prediction for samples: 001"):
                module.validate_prediction_outputs(rows, root / "workspace", root / "out")

    def test_cross_prediction_missing_sample_fails_validation(self) -> None:
        module = self.load_module_from_path(
            "predict_model_on_dataset_missing",
            REPO_ROOT / "Comparison" / "scripts" / "predict_model_on_dataset.py",
        )

        with workspace_tempdir() as tmp:
            root = Path(tmp)
            sample_1 = make_sample(root / "samples", "001")
            sample_2 = make_sample(root / "samples", "002")
            rows = [
                {"sample_id": "001", "structure_path": str(sample_1 / "RUN.fdf")},
                {"sample_id": "002", "structure_path": str(sample_2 / "RUN.fdf")},
            ]
            workspace = root / "workspace"
            copied = module.copy_sample_inputs(rows, workspace)
            (workspace / "predict_structures" / "001" / "ML_prediction.HSX").write_bytes(b"prediction")
            prediction_rows = module.collect_predictions(copied, workspace, root / "out")
            with self.assertRaisesRegex(RuntimeError, "missing prediction for samples: 002"):
                module.validate_prediction_outputs(prediction_rows, workspace, root / "out")

    def test_cross_prediction_mismatched_output_path_fails_validation(self) -> None:
        module = self.load_module_from_path(
            "predict_model_on_dataset_path_mismatch",
            REPO_ROOT / "Comparison" / "scripts" / "predict_model_on_dataset.py",
        )

        with workspace_tempdir() as tmp:
            root = Path(tmp)
            workspace_prediction = root / "workspace" / "predict_structures" / "001" / "ML_prediction.HSX"
            archive_prediction = root / "out" / "predicted_hamiltonians" / "001" / "ML_prediction.HSX"
            wrong_prediction = root / "out" / "predicted_hamiltonians" / "wrong" / "ML_prediction.HSX"
            workspace_prediction.parent.mkdir(parents=True)
            archive_prediction.parent.mkdir(parents=True)
            wrong_prediction.parent.mkdir(parents=True)
            workspace_prediction.write_bytes(b"prediction")
            archive_prediction.write_bytes(b"prediction")
            wrong_prediction.write_bytes(b"prediction")
            rows = [{"sample_id": "001", "status": "predicted", "prediction_path": str(wrong_prediction)}]
            with self.assertRaisesRegex(RuntimeError, "prediction path mismatch for samples: 001"):
                module.validate_prediction_outputs(rows, root / "workspace", root / "out")

    def test_cross_prediction_complete_set_passes_validation(self) -> None:
        module = self.load_module_from_path(
            "predict_model_on_dataset_complete",
            REPO_ROOT / "Comparison" / "scripts" / "predict_model_on_dataset.py",
        )

        with workspace_tempdir() as tmp:
            root = Path(tmp)
            sample_1 = make_sample(root / "samples", "001")
            sample_2 = make_sample(root / "samples", "002")
            rows = [
                {"sample_id": "001", "structure_path": str(sample_1 / "RUN.fdf")},
                {"sample_id": "002", "structure_path": str(sample_2 / "RUN.fdf")},
            ]
            workspace = root / "workspace"
            copied = module.copy_sample_inputs(rows, workspace)
            for sample_id in ("001", "002"):
                (workspace / "predict_structures" / sample_id / "ML_prediction.HSX").write_bytes(
                    f"prediction {sample_id}".encode("utf-8")
                )
            prediction_rows = module.collect_predictions(copied, workspace, root / "out")
            validation = module.validate_prediction_outputs(prediction_rows, workspace, root / "out")
            self.assertTrue(validation["validated"])
            self.assertEqual(validation["validated_samples"], 2)

    def test_siesta_settings_hash_and_mismatch_warning(self) -> None:
        sys.path.insert(0, str(REPO_ROOT / "Comparison" / "scripts"))
        spec = importlib.util.spec_from_file_location(
            "siesta_settings",
            REPO_ROOT / "Comparison" / "scripts" / "siesta_settings.py",
        )
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        shared, md, atom = self.siesta_settings_fixture()
        ok_report = module.compare_settings(md, atom, shared)
        self.assertTrue(ok_report["ok"])
        atom_bad = json.loads(json.dumps(atom))
        atom_bad["structure"]["siesta"]["MeshCutoff"] = "300 Ry"
        bad_report = module.compare_settings(md, atom_bad, shared)
        self.assertFalse(bad_report["ok"])
        self.assertTrue(bad_report["warning"])

    def test_siesta_settings_three_method_equivalent_settings_pass(self) -> None:
        module = self.load_module_from_path(
            "siesta_settings_three_method_ok",
            REPO_ROOT / "Comparison" / "scripts" / "siesta_settings.py",
        )
        shared, md, atom = self.siesta_settings_fixture()
        same_artifacts = {
            method: {"basis_hash": "basis_same", "pseudopotential_hash": "pseudo_same"}
            for method in ("md", "siesta_fc_cartesian", "random_cartesian")
        }
        report = module.compare_method_settings(
            {"md": md, "siesta_fc_cartesian": atom, "random_cartesian": atom},
            shared,
            artifact_hashes_by_method=same_artifacts,
            selected_methods=["md", "siesta_fc_cartesian", "random_cartesian"],
        )
        self.assertTrue(report["ok"])
        self.assertEqual(
            set(report["siesta_settings_hash_by_method"]),
            {"md", "siesta_fc_cartesian", "random_cartesian"},
        )
        self.assertIn("random_cartesian", report["method_siesta_settings"])
        self.assertEqual(report["pairwise_mismatch_report"], [])
        self.assertFalse(report["severe_warning"])

    def test_siesta_settings_random_cartesian_meshcutoff_mismatch_is_severe(self) -> None:
        module = self.load_module_from_path(
            "siesta_settings_rc_mesh_bad",
            REPO_ROOT / "Comparison" / "scripts" / "siesta_settings.py",
        )
        shared, md, atom = self.siesta_settings_fixture()
        random_config = copy.deepcopy(atom)
        random_config["structure"]["random_cartesian"] = {"siesta": {"MeshCutoff": "300 Ry"}}
        report = module.compare_method_settings(
            {"md": md, "siesta_fc_cartesian": atom, "random_cartesian": random_config},
            shared,
            selected_methods=["md", "siesta_fc_cartesian", "random_cartesian"],
        )
        self.assertFalse(report["ok"])
        severe_keys = {mismatch["key"] for mismatch in report["severe_mismatches"]}
        self.assertIn("MeshCutoff", severe_keys)
        self.assertTrue(
            any("random_cartesian" in mismatch["methods"] for mismatch in report["severe_mismatches"])
        )
        self.assertTrue(report["severe_warning"])

    def test_siesta_settings_fc_basis_mismatch_is_severe(self) -> None:
        module = self.load_module_from_path(
            "siesta_settings_fc_basis_bad",
            REPO_ROOT / "Comparison" / "scripts" / "siesta_settings.py",
        )
        shared, md, atom = self.siesta_settings_fixture()
        atom_bad = copy.deepcopy(atom)
        atom_bad["structure"]["siesta"]["PAO.BasisSize"] = "SZ"
        report = module.compare_method_settings(
            {"md": md, "siesta_fc_cartesian": atom_bad, "random_cartesian": atom},
            shared,
            selected_methods=["md", "siesta_fc_cartesian", "random_cartesian"],
        )
        self.assertFalse(report["ok"])
        severe = [
            mismatch
            for mismatch in report["severe_mismatches"]
            if mismatch["key"] == "PAO.BasisSize"
        ]
        self.assertTrue(severe)
        self.assertTrue(any("siesta_fc_cartesian" in mismatch["methods"] for mismatch in severe))

    def test_siesta_settings_hamiltonian_output_flag_mismatches_are_severe(self) -> None:
        module = self.load_module_from_path(
            "siesta_settings_output_flags_bad",
            REPO_ROOT / "Comparison" / "scripts" / "siesta_settings.py",
        )
        shared, md, atom = self.siesta_settings_fixture()
        cases = {
            "Save.HS": lambda config: config["structure"]["siesta"].__setitem__("Save.HS", "F"),
            "TS.HS.Save": lambda config: config["structure"]["force_constants"].__setitem__("save_tshs", False),
            "TS.DE.Save": lambda config: config["structure"]["force_constants"].__setitem__("save_tsde", False),
            "XML.Write": lambda config: config["structure"]["siesta"].__setitem__("XML.Write", "F"),
        }
        for key, mutate in cases.items():
            with self.subTest(key=key):
                atom_bad = copy.deepcopy(atom)
                mutate(atom_bad)
                report = module.compare_method_settings(
                    {"md": md, "siesta_fc_cartesian": atom_bad},
                    shared,
                    selected_methods=["md", "siesta_fc_cartesian"],
                )
                severe = [
                    mismatch
                    for mismatch in report["severe_mismatches"]
                    if mismatch["key"] == key
                ]
                self.assertFalse(report["ok"])
                self.assertTrue(report["severe_warning"])
                self.assertTrue(severe)
                self.assertTrue(all(mismatch["scientifically_relevant"] for mismatch in severe))

    def test_siesta_settings_cosmetic_untracked_setting_is_not_severe(self) -> None:
        module = self.load_module_from_path(
            "siesta_settings_cosmetic_not_severe",
            REPO_ROOT / "Comparison" / "scripts" / "siesta_settings.py",
        )
        shared, md, atom = self.siesta_settings_fixture()
        atom_bad = copy.deepcopy(atom)
        atom_bad["structure"]["siesta"]["LongOutput"] = "F"
        report = module.compare_method_settings(
            {"md": md, "siesta_fc_cartesian": atom_bad},
            shared,
            selected_methods=["md", "siesta_fc_cartesian"],
        )
        self.assertTrue(report["ok"])
        self.assertEqual(report["severe_mismatches"], [])
        self.assertNotIn("LongOutput", {mismatch["key"] for mismatch in report["pairwise_mismatch_report"]})

    def test_siesta_settings_path_only_artifact_differences_are_not_severe(self) -> None:
        module = self.load_module_from_path(
            "siesta_settings_path_only",
            REPO_ROOT / "Comparison" / "scripts" / "siesta_settings.py",
        )
        shared, md, atom = self.siesta_settings_fixture()
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            artifact_hashes = {}
            for method in ("md", "siesta_fc_cartesian", "random_cartesian"):
                method_dir = root / method / "different" / "absolute" / "path"
                method_dir.mkdir(parents=True)
                basis = method_dir / "O.ion.xml"
                pseudo = method_dir / "O.psf"
                basis.write_text("<basis>same</basis>\n", encoding="utf-8")
                pseudo.write_text("pseudo same\n", encoding="utf-8")
                artifact_hashes[method] = module.artifact_hash_payload(
                    basis_files=[basis],
                    pseudopotential_files=[pseudo],
                )
            report = module.compare_method_settings(
                {"md": md, "siesta_fc_cartesian": atom, "random_cartesian": atom},
                shared,
                artifact_hashes_by_method=artifact_hashes,
                selected_methods=["md", "siesta_fc_cartesian", "random_cartesian"],
            )
        self.assertTrue(report["ok"])
        self.assertEqual(report["severe_mismatches"], [])
        self.assertFalse(report["basis_pseudopotential_warning"])

    def test_initial_manifest_records_per_method_siesta_settings_for_three_methods(self) -> None:
        module = self.load_pipeline_ui_module()
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            module.RESULTS_ROOT = root / "results"
            runner = module.ExperimentRunner()
            manifest = runner._initial_experiment_manifest(
                run_id="siesta_manifest_case",
                md_sizes=[3],
                atom_sizes=[3],
                split_ratios={"train": 1 / 3, "validation": 1 / 3, "test": 1 / 3},
                random_seed=42,
                split_mode="shared",
                atom_dataset_specs=None,
                test_sets=["test_md", "test_siesta_fc_cartesian", "test_random_cartesian"],
                primary_metric="global_rmse_eV",
                compute_budget_mode="dataset_size",
                compute_accelerator="cpu",
                selected_methods=["md", "siesta_fc_cartesian", "random_cartesian"],
                run_mode="dataset_only",
                random_cartesian_options={"n_structures": 3},
            )
        self.assertEqual(
            set(manifest["siesta_settings_hash_by_method"]),
            {"md", "siesta_fc_cartesian", "random_cartesian"},
        )
        self.assertIn("random_cartesian", manifest["method_siesta_settings"])
        self.assertIn("random_cartesian", manifest["basis_hash_by_method"])
        self.assertIn("random_cartesian", manifest["pseudopotential_hash_by_method"])
        self.assertIsInstance(manifest["siesta_settings_pairwise_mismatch_report"], list)

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
        module = self.load_model_settings_module("model_settings_binary_mismatch")
        md = {"training": {"data": {"out_matrix": "hamiltonian", "batch_size": 8}, "model": {"optim_lr": 0.005}, "trainer": {"max_epochs": 10}}}
        atom = {"training": {"data": {"out_matrix": "hamiltonian", "batch_size": 16}, "model": {"optim_lr": 0.005}, "trainer": {"max_epochs": 10}}}
        report = module.compare_model_settings(md, atom)
        self.assertFalse(report["ok"])
        self.assertEqual(report["mismatches"][0]["section"], "data")

    def test_model_settings_three_method_equivalent_settings_pass(self) -> None:
        module = self.load_model_settings_module("model_settings_three_method_ok")
        md, fc, rc = self.model_settings_fixture()
        report = module.compare_method_model_settings(
            {"md": md, "siesta_fc_cartesian": fc, "random_cartesian": rc},
            selected_methods=["md", "siesta_fc_cartesian", "random_cartesian"],
        )
        self.assertTrue(report["ok"])
        self.assertEqual(
            set(report["model_config_hash_by_method"]),
            {"md", "siesta_fc_cartesian", "random_cartesian"},
        )
        self.assertIn("random_cartesian", report["method_model_settings"])
        self.assertEqual(report["pairwise_mismatch_report"], [])
        self.assertFalse(report["severe_warning"])

    def test_model_settings_random_cartesian_architecture_mismatch_is_severe(self) -> None:
        module = self.load_model_settings_module("model_settings_rc_arch_bad")
        md, fc, rc = self.model_settings_fixture()
        rc["training"]["model"]["num_interactions"] = 2
        report = module.compare_method_model_settings(
            {"md": md, "siesta_fc_cartesian": fc, "random_cartesian": rc},
            selected_methods=["md", "siesta_fc_cartesian", "random_cartesian"],
        )
        self.assertFalse(report["ok"])
        severe = [
            mismatch
            for mismatch in report["severe_mismatches"]
            if mismatch["section"] == "model" and mismatch["key"] == "num_interactions"
        ]
        self.assertTrue(severe)
        self.assertTrue(any("random_cartesian" in mismatch["methods"] for mismatch in severe))
        self.assertTrue(report["severe_warning"])

    def test_model_settings_fc_loss_mismatch_is_severe(self) -> None:
        module = self.load_model_settings_module("model_settings_fc_loss_bad")
        md, fc, rc = self.model_settings_fixture()
        fc["training"]["model"]["loss"] = "graph2mat.metrics.mse"
        report = module.compare_method_model_settings(
            {"md": md, "siesta_fc_cartesian": fc, "random_cartesian": rc},
            selected_methods=["md", "siesta_fc_cartesian", "random_cartesian"],
        )
        self.assertFalse(report["ok"])
        severe = [
            mismatch
            for mismatch in report["severe_mismatches"]
            if mismatch["section"] == "model" and mismatch["key"] == "loss"
        ]
        self.assertTrue(severe)
        self.assertTrue(any("siesta_fc_cartesian" in mismatch["methods"] for mismatch in severe))

    def test_model_settings_path_only_differences_are_not_severe(self) -> None:
        module = self.load_model_settings_module("model_settings_path_only")
        md, fc, rc = self.model_settings_fixture()
        md["training"]["data"]["dataset_path"] = "/tmp/md/dataset"
        fc["training"]["data"]["dataset_path"] = "/tmp/fc/dataset"
        rc["training"]["data"]["dataset_path"] = "/tmp/rc/dataset"
        md["training"]["trainer"]["default_root_dir"] = "/tmp/md/out"
        fc["training"]["trainer"]["default_root_dir"] = "/tmp/fc/out"
        rc["training"]["trainer"]["default_root_dir"] = "/tmp/rc/out"
        report = module.compare_method_model_settings(
            {"md": md, "siesta_fc_cartesian": fc, "random_cartesian": rc},
            selected_methods=["md", "siesta_fc_cartesian", "random_cartesian"],
        )
        self.assertTrue(report["ok"])
        self.assertEqual(report["severe_mismatches"], [])
        self.assertEqual(report["pairwise_mismatch_report"], [])

    def test_model_settings_hardware_accelerator_difference_is_info(self) -> None:
        module = self.load_model_settings_module("model_settings_accelerator_info")
        md, fc, rc = self.model_settings_fixture()
        rc["training"]["trainer"]["accelerator"] = "gpu"
        report = module.compare_method_model_settings(
            {"md": md, "siesta_fc_cartesian": fc, "random_cartesian": rc},
            selected_methods=["md", "siesta_fc_cartesian", "random_cartesian"],
        )
        self.assertTrue(report["ok"])
        self.assertEqual(report["severe_mismatches"], [])
        self.assertTrue(report["warning_mismatches"])
        self.assertTrue(
            all(mismatch["severity"] == "info" for mismatch in report["warning_mismatches"])
        )

    def test_initial_manifest_records_per_method_model_settings_for_three_methods(self) -> None:
        module = self.load_pipeline_ui_module()
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            module.RESULTS_ROOT = root / "results"
            runner = module.ExperimentRunner()
            manifest = runner._initial_experiment_manifest(
                run_id="model_manifest_case",
                md_sizes=[3],
                atom_sizes=[3],
                split_ratios={"train": 1 / 3, "validation": 1 / 3, "test": 1 / 3},
                random_seed=42,
                split_mode="shared",
                atom_dataset_specs=None,
                test_sets=["test_md", "test_siesta_fc_cartesian", "test_random_cartesian"],
                primary_metric="global_rmse_eV",
                compute_budget_mode="dataset_size",
                compute_accelerator="cpu",
                selected_methods=["md", "siesta_fc_cartesian", "random_cartesian"],
                run_mode="dataset_only",
                random_cartesian_options={"n_structures": 3},
            )
        self.assertEqual(
            set(manifest["model_config_hash_by_method"]),
            {"md", "siesta_fc_cartesian", "random_cartesian"},
        )
        self.assertIn("random_cartesian", manifest["method_model_settings"])
        self.assertIn("random_cartesian", manifest["training_hyperparameters"])
        self.assertIsInstance(manifest["model_config_pairwise_mismatch_report"], list)

    def test_method_provenance_contains_three_selected_methods(self) -> None:
        module = self.load_pipeline_ui_module()
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            module.RESULTS_ROOT = root / "results"
            runs = [
                self.provenance_run_fixture(root, "md", label="md_190", size=190),
                self.provenance_run_fixture(root, "siesta_fc_cartesian", label="fc_190", size=190),
                self.provenance_run_fixture(root, "random_cartesian", label="rc_3500", size=3500),
            ]
            manifest = {
                "experiment_id": "provenance_case",
                "selected_methods": ["md", "siesta_fc_cartesian", "random_cartesian"],
                "runs": runs,
                "siesta_settings_hash_by_method": {
                    "md": "siesta_md",
                    "siesta_fc_cartesian": "siesta_fc",
                    "random_cartesian": "siesta_rc",
                },
                "model_config_hash_by_method": {
                    "md": "model_md",
                    "siesta_fc_cartesian": "model_fc",
                    "random_cartesian": "model_rc",
                },
                "basis_hash_by_method": {
                    "md": "basis_md",
                    "siesta_fc_cartesian": "basis_fc",
                    "random_cartesian": "basis_rc",
                },
                "pseudopotential_hash_by_method": {
                    "md": "pseudo_md",
                    "siesta_fc_cartesian": "pseudo_fc",
                    "random_cartesian": "pseudo_rc",
                },
            }
            module.refresh_method_provenance(manifest)
            frozen_manifest = root / "common_tests" / "test_random_cartesian" / "frozen_test_manifest.json"
            frozen_manifest.parent.mkdir(parents=True)
            frozen_manifest.write_text("{}", encoding="utf-8")
            cross_provenance = module.build_method_provenance(
                manifest,
                selected_methods=["md", "siesta_fc_cartesian", "random_cartesian"],
                runs=runs,
                frozen_test_manifests_by_test_set={
                    "test_random_cartesian": str(frozen_manifest),
                },
            )
            runner = module.ExperimentRunner()
            runner._write_experiment_manifest(manifest)
            written_manifest = module.load_config(root / "results" / "provenance_case" / "experiment_manifest.yaml")
        self.assertEqual(
            set(manifest["method_provenance"]),
            {"md", "siesta_fc_cartesian", "random_cartesian"},
        )
        self.assertEqual(set(written_manifest["method_provenance"]), set(manifest["method_provenance"]))
        self.assertEqual(manifest["method_provenance"]["random_cartesian"]["dataset_size"], 3500)
        self.assertEqual(manifest["method_provenance"]["random_cartesian"]["dataset_label"], "rc_3500")
        self.assertNotIn("atom_displacement", manifest["method_provenance"])
        self.assertIn("test", manifest["method_provenance"]["md"]["split_manifest"])
        self.assertEqual(
            cross_provenance["random_cartesian"]["frozen_test_manifest"],
            str(frozen_manifest),
        )

    def test_method_provenance_missing_checkpoint_hash_warns(self) -> None:
        module = self.load_pipeline_ui_module()
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            manifest = {
                "experiment_id": "missing_checkpoint_hash",
                "selected_methods": ["random_cartesian"],
                "runs": [
                    self.provenance_run_fixture(
                        root,
                        "random_cartesian",
                        label="rc_no_hash",
                        checkpoint_hash=None,
                    )
                ],
                "siesta_settings_hash_by_method": {"random_cartesian": "siesta_rc"},
                "model_config_hash_by_method": {"random_cartesian": "model_rc"},
                "basis_hash_by_method": {"random_cartesian": "basis_rc"},
                "pseudopotential_hash_by_method": {"random_cartesian": "pseudo_rc"},
            }
            module.refresh_method_provenance(manifest)
        warnings = manifest["method_provenance"]["random_cartesian"]["warnings"]
        self.assertTrue(any("Missing checkpoint hash" in warning for warning in warnings))
        self.assertTrue(any("Missing checkpoint hash" in warning for warning in manifest["method_provenance_warnings"]))

    def test_method_provenance_reads_legacy_two_method_runs(self) -> None:
        module = self.load_pipeline_ui_module()
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            manifest = {
                "experiment_id": "legacy_two_method",
                "selected_methods": ["md", "atom_displacement"],
                "runs": [
                    self.provenance_run_fixture(root, "md", label="md_legacy", legacy_pipeline_only=True),
                    self.provenance_run_fixture(
                        root,
                        "atom_displacement",
                        label="atom_legacy",
                        legacy_pipeline_only=True,
                    ),
                ],
                "siesta_settings_hash_by_method": {
                    "md": "siesta_md",
                    "siesta_fc_cartesian": "siesta_fc",
                },
                "model_config_hash_by_method": {
                    "md": "model_md",
                    "siesta_fc_cartesian": "model_fc",
                },
                "basis_hash_by_method": {
                    "md": "basis_md",
                    "siesta_fc_cartesian": "basis_fc",
                },
                "pseudopotential_hash_by_method": {
                    "md": "pseudo_md",
                    "siesta_fc_cartesian": "pseudo_fc",
                },
            }
            module.refresh_method_provenance(manifest)
        self.assertEqual(set(manifest["method_provenance"]), {"md", "siesta_fc_cartesian"})
        self.assertEqual(manifest["method_provenance"]["siesta_fc_cartesian"]["dataset_label"], "atom_legacy")
        self.assertNotIn("random_cartesian", manifest["method_provenance"])

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
        self.assertEqual(md_block["data"]["val_runs"], "../MD/dataset/splits/validation/*/RUN.fdf")
        self.assertEqual(fc_block["data"]["train_runs"], "../AtomDisplacement/dataset/train_samples/*/RUN.fdf")
        self.assertEqual(fc_block["data"]["val_runs"], "../AtomDisplacement/dataset/validation_samples/*/RUN.fdf")
        module.force_shared_hyperparams(md_block, fc_block, debug_full_dataset_globs=True)
        self.assertEqual(md_block["data"]["train_runs"], "../MD/dataset/MD_steps/*/RUN.fdf")
        self.assertIsNone(md_block["data"]["val_runs"])

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
        metrics_doc = (REPO_ROOT / "Comparison" / "METRICS.md").read_text(encoding="utf-8")
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("block_metrics.csv", script)
        self.assertIn("species_pair_metrics.csv", script)
        self.assertIn("distance_bin_metrics.csv", script)
        self.assertIn("orbital_pair_metrics.csv", script)
        self.assertIn("orbital_pair_summary.csv", script)
        self.assertIn("structural_metrics_available", script)
        self.assertIn("orbital_pair_metrics_available", script)
        self.assertIn("ORBITAL_PAIR_METRIC_TARGET_SPACE", script)
        self.assertIn("ORBITAL_PAIR_BASIS_SOURCE", script)
        self.assertIn("basis_orbital_counts", script)
        self.assertIn("Basis files are required", script)
        self.assertIn("2 * angular_momentum + 1", script)
        for text in (metrics_doc, readme):
            self.assertIn("metrics/orbital_pair_metrics.csv", text)
            self.assertIn("metrics/orbital_pair_summary.csv", text)
            self.assertIn("mae_union_meV", text)
        self.assertIn("row_orbital_index", metrics_doc)
        self.assertIn("col_orbital_index", metrics_doc)
        self.assertIn("not DeepH's local-coordinate H' block representation", metrics_doc)

    def test_sparse_deeph_comparable_metric_outputs_are_part_of_schema(self) -> None:
        script = (REPO_ROOT / "Comparison" / "scripts" / "evaluate_hamiltonian_metrics.py").read_text(encoding="utf-8")
        for column in (
            "mse_ref_eV2",
            "mse_pred_eV2",
            "mse_union_eV2",
            "r2_ref",
            "r2_pred",
            "r2_union",
            "mae_ref_meV",
            "rmse_ref_meV",
            "mae_pred_meV",
            "rmse_pred_meV",
            "mae_union_meV",
            "rmse_union_meV",
            "matrix_metric_target_space",
        ):
            self.assertIn(column, script)
        self.assertIn('MATRIX_METRIC_TARGET_SPACE = "raw_global_hamiltonian"', script)

    def test_deeph_comparability_status_documents_future_work(self) -> None:
        script = (REPO_ROOT / "Comparison" / "scripts" / "evaluate_hamiltonian_metrics.py").read_text(encoding="utf-8")
        metrics_doc = (REPO_ROOT / "Comparison" / "METRICS.md").read_text(encoding="utf-8")
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        performance_doc = (REPO_ROOT / "Comparison" / "PERFORMANCE.md").read_text(encoding="utf-8")

        self.assertIn("DEEPH_COMPARABILITY_STATUS", script)
        self.assertIn("deeph_comparability_status", script)
        for key in (
            "high_symmetry_kpath_band_structure",
            "soc_complex_hamiltonians",
            "optical_berry_susceptibility_shift_current",
            "ensemble_uncertainty",
            "deeph_vs_dft_system_size_scaling",
        ):
            self.assertIn(key, script)
        for phrase in (
            "DeepH Comparability Status",
            "Implemented repo-compatible analogues",
            "Future work not implemented",
            "True high-symmetry k-path band structures",
            "SOC/complex Hamiltonians",
            "Optical response, Berry quantities, electric susceptibility and shift current",
            "Ensemble uncertainty",
            "DeepH-vs-DFT scaling by system size",
            "not exact DeepH local-coordinate H' metrics",
        ):
            self.assertIn(phrase, metrics_doc)
        self.assertIn("No reproducen todavia H' local", readme)
        self.assertIn("benchmark DeepH-vs-DFT de escalado", performance_doc)

    def test_dos_fermi_window_outputs_are_part_of_schema_and_policy(self) -> None:
        script = (REPO_ROOT / "Comparison" / "scripts" / "evaluate_hamiltonian_metrics.py").read_text(encoding="utf-8")
        metrics_doc = (REPO_ROOT / "Comparison" / "METRICS.md").read_text(encoding="utf-8")
        for column in (
            "dos_mae_500_fermi_window",
            "dos_window_min_eV",
            "dos_window_max_eV",
            "dos_window_points",
            "dos_window_sigma_eV",
            "dos_window_alignment",
            "dos_window_metric_available",
            "dos_window_unavailable_reason",
        ):
            self.assertIn(column, script)
        self.assertIn("DOS_FERMI_WINDOW_POINTS = 500", script)
        self.assertIn("DOS_FERMI_WINDOW_MIN_EV = -6.0", script)
        self.assertIn("DOS_FERMI_WINDOW_MAX_EV = 6.0", script)
        self.assertIn('DOS_FERMI_WINDOW_ALIGNMENT = "reference_fermi_level"', script)
        self.assertIn("requires_real_siesta_fermi_level", script)
        self.assertIn("dos_sweep_fields", script)
        self.assertIn('write_csv(metrics_root / "dos_metrics.csv", dos_fields, dos_rows)', script)
        self.assertIn('write_csv(metrics_root / "dos_sigma_sweep.csv", dos_sweep_fields, dos_sweep_rows)', script)
        self.assertIn("DeepH's DOS MAE benchmark", metrics_doc)
        self.assertIn("does not estimate Fermi", metrics_doc)

    def test_dos_fermi_window_metrics_fixed_grid_and_missing_fermi(self) -> None:
        import math
        try:
            import numpy as np
            import scipy  # noqa: F401
            import sisl  # noqa: F401
        except ModuleNotFoundError as exc:
            self.skipTest(f"scientific Python dependency unavailable: {exc.name}")

        module = self.load_metrics_module("evaluate_hamiltonian_metrics_dos_window_test")
        reference = np.asarray([-1.0, 0.0, 1.0], dtype=float)
        predicted = np.asarray([-1.0, 0.0, 1.0], dtype=float)
        grid, metrics = module.dos_fermi_window_metrics(reference, predicted, 2.0)

        self.assertEqual(grid.size, 500)
        self.assertAlmostEqual(float(grid[0]), -4.0)
        self.assertAlmostEqual(float(grid[-1]), 8.0)
        self.assertEqual(metrics["dos_window_points"], 500)
        self.assertEqual(metrics["dos_window_min_eV"], -6.0)
        self.assertEqual(metrics["dos_window_max_eV"], 6.0)
        self.assertEqual(metrics["dos_window_sigma_eV"], module.DOS_SIGMA_EV)
        self.assertEqual(metrics["dos_window_alignment"], "reference_fermi_level")
        self.assertTrue(metrics["dos_window_metric_available"])
        self.assertEqual(metrics["dos_window_unavailable_reason"], "")
        self.assertAlmostEqual(metrics["dos_mae_500_fermi_window"], 0.0)

        _different_grid, different_metrics = module.dos_fermi_window_metrics(
            reference,
            np.asarray([-0.5, 0.6, 1.4], dtype=float),
            2.0,
        )
        self.assertTrue(math.isfinite(different_metrics["dos_mae_500_fermi_window"]))
        self.assertGreater(different_metrics["dos_mae_500_fermi_window"], 0.0)

        missing_grid, missing_metrics = module.dos_fermi_window_metrics(reference, predicted, None)
        self.assertEqual(missing_grid.size, 0)
        self.assertTrue(np.isnan(missing_metrics["dos_mae_500_fermi_window"]))
        self.assertFalse(missing_metrics["dos_window_metric_available"])
        self.assertEqual(missing_metrics["dos_window_unavailable_reason"], "missing_fermi_level")

        adaptive_rows, adaptive_metrics = module.dos_for_sample(reference, np.asarray([-0.5, 0.6, 1.4], dtype=float))
        self.assertEqual(len(adaptive_rows), module.DOS_POINTS)
        self.assertEqual(adaptive_metrics["dos_grid_points"], module.DOS_POINTS)
        for key in ("dos_wasserstein_eV", "dos_l1", "dos_l2"):
            self.assertIn(key, adaptive_metrics)
            self.assertTrue(math.isfinite(adaptive_metrics[key]))
        self.assertIn("siesta_dos_normalized", adaptive_rows[0])
        self.assertIn("predicted_dos_normalized", adaptive_rows[0])

    def test_sparse_metrics_exact_perturbation_and_support_failures(self) -> None:
        import math
        try:
            import numpy as np
            from scipy import sparse
        except ModuleNotFoundError as exc:
            self.skipTest(f"scientific Python dependency unavailable: {exc.name}")

        module = self.load_metrics_module("evaluate_hamiltonian_metrics_sparse_test")

        def matrix_data(matrix):
            return module.MatrixData(
                path=Path("synthetic.HSX"),
                hamiltonian=sparse.csr_matrix(matrix),
                overlap=None,
                own_eigenvalues=np.asarray([], dtype=float),
                fermi_level=0.0,
                fermi_level_source="siesta_file",
                orthogonal=True,
                has_overlap=False,
                overlap_error=None,
            )

        reference = matrix_data([[1.0, 2.0], [0.0, 4.0]])
        exact = module.sparse_metrics("exact", reference, matrix_data([[1.0, 2.0], [0.0, 4.0]]))
        self.assertEqual(exact["mae_union_eV"], 0.0)
        self.assertEqual(exact["rmse_union_eV"], 0.0)
        self.assertEqual(exact["mse_union_eV2"], 0.0)
        self.assertEqual(exact["r2_ref"], 1.0)
        self.assertEqual(exact["r2_pred"], 1.0)
        self.assertEqual(exact["r2_union"], 1.0)
        self.assertEqual(exact["support_f1"], 1.0)
        self.assertEqual(exact["matrix_metric_target_space"], "raw_global_hamiltonian")

        perturbed = module.sparse_metrics("perturbed", reference, matrix_data([[1.5, 0.0], [3.0, 4.0]]))
        self.assertEqual(perturbed["false_zeros"], 1)
        self.assertEqual(perturbed["false_nonzeros"], 1)
        self.assertAlmostEqual(perturbed["weighted_false_zeros_eV"], 2.0)
        self.assertAlmostEqual(perturbed["weighted_false_nonzeros_eV"], 3.0)
        self.assertAlmostEqual(perturbed["mae_ref_eV"], (0.5 + 2.0 + 0.0) / 3.0)
        self.assertAlmostEqual(perturbed["rmse_ref_eV"], math.sqrt((0.5**2 + 2.0**2 + 0.0**2) / 3.0))
        self.assertAlmostEqual(perturbed["mae_union_eV"], (0.5 + 2.0 + 3.0) / 4.0)
        self.assertAlmostEqual(perturbed["rmse_union_eV"], math.sqrt((0.5**2 + 2.0**2 + 3.0**2) / 4.0))
        self.assertAlmostEqual(perturbed["mse_ref_eV2"], (0.5**2 + 2.0**2 + 0.0**2) / 3.0)
        self.assertAlmostEqual(perturbed["mse_pred_eV2"], (0.5**2 + 3.0**2 + 0.0**2) / 3.0)
        self.assertAlmostEqual(perturbed["mse_union_eV2"], (0.5**2 + 2.0**2 + 3.0**2 + 0.0**2) / 4.0)
        self.assertAlmostEqual(perturbed["mse_union_eV2"], perturbed["rmse_union_eV"] ** 2)
        self.assertAlmostEqual(perturbed["r2_ref"], 5.0 / 56.0)
        self.assertAlmostEqual(perturbed["r2_pred"], -7.0 / 104.0)
        self.assertAlmostEqual(perturbed["r2_union"], -18.0 / 35.0)
        self.assertAlmostEqual(perturbed["mae_ref_meV"], 1000.0 * perturbed["mae_ref_eV"])
        self.assertAlmostEqual(perturbed["rmse_ref_meV"], 1000.0 * perturbed["rmse_ref_eV"])
        self.assertAlmostEqual(perturbed["mae_pred_meV"], 1000.0 * perturbed["mae_pred_eV"])
        self.assertAlmostEqual(perturbed["rmse_pred_meV"], 1000.0 * perturbed["rmse_pred_eV"])
        self.assertAlmostEqual(perturbed["mae_union_meV"], 1000.0 * perturbed["mae_union_eV"])
        self.assertAlmostEqual(perturbed["rmse_union_meV"], 1000.0 * perturbed["rmse_union_eV"])

        constant_target = module.sparse_metrics(
            "constant",
            matrix_data([[2.0, 2.0], [0.0, 0.0]]),
            matrix_data([[2.0, 1.0], [0.0, 0.0]]),
        )
        self.assertTrue(math.isnan(constant_target["r2_ref"]))

        non_hermitian = module.sparse_metrics("nonhermitian", reference, matrix_data([[1.0, 5.0], [0.0, 4.0]]))
        self.assertGreater(non_hermitian["hermiticity_pred"], 0.0)

    def test_metric_compatibility_blocks_shape_overlap_and_components(self) -> None:
        try:
            import numpy as np
            from scipy import sparse
        except ModuleNotFoundError as exc:
            self.skipTest(f"scientific Python dependency unavailable: {exc.name}")

        module = self.load_metrics_module("evaluate_hamiltonian_metrics_compat_test")

        def matrix_data(values, *, orthogonal=True, overlap=None, component_count=1, spin_kind=None):
            return module.MatrixData(
                path=Path("synthetic.HSX"),
                hamiltonian=sparse.diags(values, format="csr"),
                overlap=overlap,
                own_eigenvalues=np.asarray(values, dtype=float),
                fermi_level=0.0,
                fermi_level_source="siesta_file",
                orthogonal=orthogonal,
                has_overlap=overlap is not None,
                overlap_error="missing overlap" if overlap is None else None,
                component_count=component_count,
                spin_kind=spin_kind,
            )

        shape_errors = module.matrix_compatibility_errors("shape", matrix_data([0.0, 1.0]), matrix_data([0.0, 1.0, 2.0]))
        self.assertIn("matrix_shape_mismatch", {error["kind"] for error in shape_errors})

        overlap_errors = module.matrix_compatibility_errors("overlap", matrix_data([0.0, 1.0], orthogonal=False), matrix_data([0.0, 1.0]))
        self.assertIn("missing_required_overlap", {error["kind"] for error in overlap_errors})

        component_errors = module.matrix_compatibility_errors("components", matrix_data([0.0, 1.0], component_count=2), matrix_data([0.0, 1.0]))
        self.assertIn("unsupported_matrix_components", {error["kind"] for error in component_errors})

        graph2mat_reference = matrix_data(
            [0.0, 1.0],
            orthogonal=False,
            overlap=sparse.eye(2, format="csr"),
            component_count=1,
            spin_kind="Spin{unpolarized}",
        )
        graph2mat_prediction = matrix_data(
            [0.1, 1.2],
            orthogonal=False,
            overlap=sparse.eye(2, format="csr"),
            component_count=2,
            spin_kind="Spin{polarized}",
        )
        auxiliary_errors = module.matrix_compatibility_errors(
            "graph2mat_auxiliary",
            graph2mat_reference,
            graph2mat_prediction,
        )
        self.assertNotIn("unsupported_matrix_components", {error["kind"] for error in auxiliary_errors})
        self.assertNotIn("spin_state_mismatch", {error["kind"] for error in auxiliary_errors})
        auxiliary_warnings = module.matrix_compatibility_warnings(
            "graph2mat_auxiliary",
            graph2mat_reference,
            graph2mat_prediction,
        )
        self.assertIn("graph2mat_auxiliary_component_ignored", {warning["kind"] for warning in auxiliary_warnings})

    def test_missing_fermi_does_not_infer_frontier_or_gap(self) -> None:
        try:
            import numpy as np
        except ModuleNotFoundError as exc:
            self.skipTest(f"scientific Python dependency unavailable: {exc.name}")

        module = self.load_metrics_module("evaluate_hamiltonian_metrics_no_fermi_test")
        _rows, metrics = module.eigen_error_metrics(
            np.asarray([-2.0, -1.0, 1.0, 2.0], dtype=float),
            np.asarray([-2.1, -0.5, 2.0, 3.0], dtype=float),
            None,
            "unavailable",
        )
        self.assertEqual(metrics["occupied_bands"], 0)
        self.assertEqual(metrics["frontier_window_bands"], 0)
        self.assertFalse(metrics["fermi_metric_available"])
        self.assertFalse(metrics["fermi_window_metric_available"])
        self.assertTrue(np.isnan(metrics["frontier_window_rmse_eV"]))
        self.assertTrue(np.isnan(metrics["fermi_window_rmse_eV"]))
        self.assertTrue(np.isnan(metrics["gap_abs_error_eV"]))

    def test_evaluate_sample_status_records_reference_hash(self) -> None:
        try:
            import numpy as np
            from scipy import sparse
        except ModuleNotFoundError as exc:
            self.skipTest(f"scientific Python dependency unavailable: {exc.name}")

        module = self.load_metrics_module("evaluate_hamiltonian_metrics_status_test")
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            pred_dir = root / "predicted" / "sample_1"
            ref_dir = root / "siesta" / "sample_1"
            pred_dir.mkdir(parents=True)
            ref_dir.mkdir(parents=True)
            pred_path = pred_dir / "ML_prediction.HSX"
            ref_path = ref_dir / "siesta.TSHS"
            pred_path.write_bytes(b"prediction")
            ref_path.write_bytes(b"reference")

            def fake_read_matrix(path: Path):
                values = [0.0, 1.0]
                return module.MatrixData(
                    path=path,
                    hamiltonian=sparse.diags(values, format="csr"),
                    overlap=None,
                    own_eigenvalues=np.asarray(values, dtype=float),
                    fermi_level=0.5,
                    fermi_level_source="siesta_file",
                    orthogonal=True,
                    has_overlap=False,
                    overlap_error=None,
                    sha256=module.file_sha256(path),
                )

            original = module.read_matrix
            module.read_matrix = fake_read_matrix
            try:
                rows = module.evaluate_sample(
                    "sample_1",
                    pred_dir,
                    ref_dir,
                    root,
                    {},
                    low_energy_enabled=False,
                    low_energy_n_states=2,
                    low_energy_alignment="none",
                )
            finally:
                module.read_matrix = original
            status = rows["sample_status"][0]
            self.assertEqual(status["status"], "warning")
            self.assertEqual(status["reference_kind"], ".TSHS")
            self.assertEqual(status["reference_sha256"], module.file_sha256(ref_path))
            self.assertEqual(rows["fatal_errors"], [])

    def test_orthogonal_spectral_metrics_are_comparable_without_overlap(self) -> None:
        try:
            import numpy as np
            from scipy import sparse
        except ModuleNotFoundError as exc:
            self.skipTest(f"scientific Python dependency unavailable: {exc.name}")

        module = self.load_metrics_module("evaluate_hamiltonian_metrics_orthogonal_comparable_test")
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            pred_dir = root / "predicted" / "sample_1"
            ref_dir = root / "siesta" / "sample_1"
            pred_dir.mkdir(parents=True)
            ref_dir.mkdir(parents=True)
            (pred_dir / "ML_prediction.HSX").write_bytes(b"prediction")
            (ref_dir / "siesta.TSHS").write_bytes(b"reference")

            def fake_read_matrix(path: Path):
                values = [0.0, 1.0, 2.0]
                return module.MatrixData(
                    path=path,
                    hamiltonian=sparse.diags(values, format="csr"),
                    overlap=None,
                    own_eigenvalues=np.asarray(values, dtype=float),
                    fermi_level=0.5,
                    fermi_level_source="siesta_file",
                    orthogonal=True,
                    has_overlap=False,
                    overlap_error=None,
                )

            original = module.read_matrix
            module.read_matrix = fake_read_matrix
            try:
                rows = module.evaluate_sample(
                    "sample_1",
                    pred_dir,
                    ref_dir,
                    root,
                    {},
                    low_energy_enabled=False,
                    low_energy_n_states=2,
                    low_energy_alignment="none",
                )
            finally:
                module.read_matrix = original

            self.assertTrue(rows["spectral"][0]["spectral_comparable"])
            self.assertFalse(rows["spectral"][0]["reference_has_overlap"])

    def test_md_post_siesta_run_fdf_timestamp_is_warning_only(self) -> None:
        try:
            import numpy  # noqa: F401
        except ModuleNotFoundError as exc:
            self.skipTest(f"scientific Python dependency unavailable: {exc.name}")

        module = self.load_metrics_module("evaluate_hamiltonian_metrics_md_mtime_test")
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            sample_dir = root / "structures" / "1"
            sample_dir.mkdir(parents=True)
            reference_path = root / "siesta.TSHS"
            structure_path = sample_dir / "RUN.fdf"
            reference_path.write_bytes(b"reference")
            minimal_run_fdf(structure_path)
            os.utime(reference_path, (100.0, 100.0))
            os.utime(structure_path, (200.0, 200.0))

            non_md_issue = module.stale_reference_issue(
                "1",
                reference_path,
                structure_path,
                method_id="siesta_fc_cartesian",
            )
            self.assertIsNotNone(non_md_issue)
            self.assertEqual(non_md_issue["kind"], "stale_reference_matrix")
            self.assertEqual(non_md_issue["severity"], "fatal")

            md_issue = module.stale_reference_issue(
                "1",
                reference_path,
                structure_path,
                method_id="md",
            )
            self.assertIsNotNone(md_issue)
            self.assertEqual(md_issue["kind"], "md_post_siesta_run_fdf_mtime")
            self.assertEqual(md_issue["severity"], "warning")

            (sample_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "method": "md",
                        "run_fdf_rewritten_from_xv": True,
                        "run_fdf_geometry_source": "siesta.XV",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            cross_md_issue = module.stale_reference_issue(
                "md_1",
                reference_path,
                structure_path,
                method_id="",
            )
            self.assertIsNotNone(cross_md_issue)
            self.assertEqual(cross_md_issue["kind"], "md_post_siesta_run_fdf_mtime")
            self.assertEqual(cross_md_issue["severity"], "warning")
            self.assertEqual(cross_md_issue["method_id"], "md")

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
        self.assertNotIn("CROSS_METRIC_FALLBACKS", app_js)
        self.assertIn("no se sustituye", app_js)
        self.assertIn('"fermi_window_rmse_eV"', app_js)

    def test_strict_ui_does_not_allow_missing_atomdisp_matrices(self) -> None:
        script = (REPO_ROOT / "Comparison" / "scripts" / "pipeline_ui.py").read_text(encoding="utf-8")
        self.assertIn('config["structure"]["force_constants"]["allow_missing_matrix"] = False', script)
        self.assertNotIn('config["structure"]["force_constants"]["allow_missing_matrix"] = True', script)

    def test_md_training_config_and_checkpoint_manifest_record_validation_split(self) -> None:
        sys.path.insert(0, str(REPO_ROOT / "MD" / "scripts"))
        module = self.load_module_from_path(
            "md_pipeline_config_validation_test",
            REPO_ROOT / "MD" / "scripts" / "md_pipeline_config.py",
        )
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            training_dir = root / "training"
            validation_dir = root / "dataset" / "splits" / "validation" / "0"
            checkpoint = training_dir / "lightning_logs" / "version_0" / "checkpoints" / "best-01.ckpt"
            validation_dir.mkdir(parents=True)
            checkpoint.parent.mkdir(parents=True)
            minimal_run_fdf(validation_dir / "RUN.fdf")
            checkpoint.write_bytes(b"checkpoint")
            config = {
                "paths": {
                    "dataset_dir": str(root / "dataset"),
                    "training_dir": str(training_dir),
                    "run_fdf_name": "RUN.fdf",
                    "run_out_name": "RUN.out",
                    "training_config_name": "config.yaml",
                    "venv_activate": str(root / "venv" / "activate"),
                },
                "training": {
                    "data": {
                        "out_matrix": "hamiltonian",
                        "symmetric_matrix": True,
                        "basis_files": "../dataset/MD_steps/basis/*.ion.xml",
                        "train_runs": "../dataset/splits/train/*/RUN.fdf",
                        "val_runs": "../dataset/splits/validation/*/RUN.fdf",
                    },
                    "model": {},
                    "trainer": {},
                },
            }

            rendered = module.render_training_config(config)
            self.assertIn("val_runs: ../dataset/splits/validation/*/RUN.fdf", rendered)
            metadata = module.require_explicit_validation_split(config)
            self.assertEqual(metadata["validation_source"], "training.data.val_runs")
            self.assertEqual(metadata["validation_sample_count"], 1)
            manifest_path = module.write_checkpoint_manifest(
                config,
                checkpoint.relative_to(training_dir).as_posix(),
                selection_mode="latest_version",
                selection_metric="val_loss",
            )
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertTrue(payload["checkpoint_selection_validation_backed"])
            self.assertEqual(payload["checkpoint_selection_metric"], "val_loss")
            self.assertEqual(payload["validation_source"], "training.data.val_runs")
            self.assertEqual(payload["validation_sample_count"], 1)

    def test_md_training_fails_without_explicit_validation_split(self) -> None:
        sys.path.insert(0, str(REPO_ROOT / "MD" / "scripts"))
        module = self.load_module_from_path(
            "md_pipeline_config_missing_validation_test",
            REPO_ROOT / "MD" / "scripts" / "md_pipeline_config.py",
        )
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            config = {
                "paths": {
                    "dataset_dir": str(root / "dataset"),
                    "training_dir": str(root / "training"),
                    "run_fdf_name": "RUN.fdf",
                    "run_out_name": "RUN.out",
                    "training_config_name": "config.yaml",
                    "venv_activate": str(root / "venv" / "activate"),
                },
                "training": {
                    "data": {
                        "train_runs": "../dataset/splits/train/*/RUN.fdf",
                        "val_runs": "../dataset/splits/validation/*/RUN.fdf",
                    }
                },
            }
            with self.assertRaisesRegex(RuntimeError, "explicit validation split"):
                module.require_explicit_validation_split(config)

    def test_atomdisp_training_config_and_checkpoint_manifest_record_validation_split(self) -> None:
        sys.path.insert(0, str(REPO_ROOT / "AtomDisplacement" / "scripts"))
        module = self.load_module_from_path(
            "atom_pipeline_config_validation_test",
            REPO_ROOT / "AtomDisplacement" / "scripts" / "pipeline_config_utils.py",
        )
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            training_dir = root / "training"
            checkpoint = training_dir / "lightning_logs" / "version_0" / "checkpoints" / "best-01.ckpt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"checkpoint")
            training_dir.joinpath("runs.json").write_text(
                json.dumps(
                    {
                        "train": ["../dataset/train_samples/0/RUN.fdf"],
                        "val": ["../dataset/validation_samples/1/RUN.fdf"],
                    }
                ),
                encoding="utf-8",
            )
            config = {
                "paths": {
                    "base_dir": str(root / "base"),
                    "relaxed_dir": str(root / "relaxed"),
                    "dataset_dir": str(root / "dataset"),
                    "samples_dir": str(root / "dataset" / "train_samples"),
                    "collected_dir": str(root / "dataset" / "collected"),
                    "training_dir": str(training_dir),
                    "run_fdf_name": "RUN.fdf",
                    "run_out_name": "RUN.out",
                    "training_config_name": "config.yaml",
                    "runs_json_name": "runs.json",
                    "samples_manifest_name": "samples.json",
                    "run_summary_name": "run_summary.json",
                    "collected_json_name": "collected.json",
                    "collected_csv_name": "collected.csv",
                    "venv_activate": str(root / "venv" / "activate"),
                },
                "training": {
                    "data": {
                        "out_matrix": "hamiltonian",
                        "symmetric_matrix": True,
                        "basis_files": "../dataset/basis/*.ion.xml",
                        "runs_json": "runs.json",
                    },
                    "model": {},
                    "trainer": {},
                },
            }

            rendered = module.render_training_config(config)
            self.assertIn("runs_json: runs.json", rendered)
            metadata = module.require_explicit_validation_split(config)
            self.assertEqual(metadata["validation_source"], "training.data.runs_json:val")
            self.assertEqual(metadata["validation_sample_count"], 1)
            manifest_path = module.write_checkpoint_manifest(
                config,
                checkpoint.relative_to(training_dir).as_posix(),
                selection_mode="latest_version",
                selection_metric="val_loss",
            )
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertTrue(payload["checkpoint_selection_validation_backed"])
            self.assertEqual(payload["checkpoint_selection_metric"], "val_loss")
            self.assertEqual(payload["validation_source"], "training.data.runs_json:val")

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

    def test_hamiltonian_prediction_uses_graph2mat_nonorthogonal_component_defaults(self) -> None:
        for path in (
            REPO_ROOT / "MD" / "pipeline_config.yaml",
            REPO_ROOT / "AtomDisplacement" / "pipeline_config.yaml",
        ):
            text = path.read_text(encoding="utf-8")
            self.assertIn("n_matrix_components: 2", text, str(path))

        for path in (
            REPO_ROOT / "MD" / "scripts" / "run_md_testing.py",
            REPO_ROOT / "MD" / "scripts" / "run_md_prediction.py",
            REPO_ROOT / "AtomDisplacement" / "scripts" / "run_atdisp_testing.py",
            REPO_ROOT / "AtomDisplacement" / "scripts" / "run_atdisp_prediction.py",
        ):
            text = path.read_text(encoding="utf-8")
            self.assertIn('get("n_matrix_components", 2)', text, str(path))

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
                self.assertIn(
                    module.MD_RUN_FDF_XV_MARKER,
                    sample_dir.joinpath("RUN.fdf").read_text(encoding="utf-8"),
                )
                signatures.append(module.effective_fdf_geometry_signature(sample_dir / "RUN.fdf"))
            self.assertNotEqual(signatures[0], signatures[1])


if __name__ == "__main__":
    unittest.main()
