import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "Comparison" / "scripts"


def load_script_module(name: str, relative_path: str):
    sys.path.insert(0, str(SCRIPTS_DIR))
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / relative_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_repo_script_module(name: str, relative_path: str, *extra_paths: Path):
    for path in (SCRIPTS_DIR, *extra_paths):
        path_text = str(path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def siesta_fixtures() -> tuple[dict, dict, dict, dict]:
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
    fc = {
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
    rc = copy.deepcopy(fc)
    rc["structure"]["random_cartesian"] = {"siesta": {}}
    return shared, md, fc, rc


def model_fixtures() -> tuple[dict, dict, dict]:
    training = {
        "torch_float32_matmul_precision": "high",
        "data": {
            "out_matrix": "hamiltonian",
            "symmetric_matrix": True,
            "sub_point_matrix": False,
            "matrix_component_policy": "h_only",
            "n_matrix_components": 1,
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
    md = {"training": copy.deepcopy(training)}
    fc = {"training": copy.deepcopy(training)}
    rc = {"training": copy.deepcopy(training)}
    return md, fc, rc


def provenance_run(root: Path, method: str, label: str, size: int) -> dict:
    result_dir = root / method / label
    split_dir = result_dir / "splits"
    split_dir.mkdir(parents=True)
    for split in ("train", "validation", "test"):
        (split_dir / f"{split}_manifest.csv").write_text("sample_id\nsample_1\n", encoding="utf-8")
    checkpoint_path = result_dir / "training" / "best.ckpt"
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_bytes(b"checkpoint")
    return {
        "method_id": method,
        "pipeline": "atom_displacement" if method == "siesta_fc_cartesian" else method,
        "dataset_label": label,
        "dataset_size": size,
        "effective_dataset_size": size,
        "result_dir": str(result_dir),
        "model_checkpoint": str(checkpoint_path),
        "model_checkpoint_sha256": f"{method}_checkpoint_hash",
        "checkpoint_manifest": str(result_dir / "training" / "checkpoint_manifest.json"),
        "checkpoint_selection_warning": "",
        "artifact_hashes": {
            "basis": f"{method}_basis_hash",
            "pseudopotentials": f"{method}_pseudo_hash",
        },
        "recipe_id": f"{method}_recipe",
        "recipe_label": f"{method} recipe",
        "recipe_set_hash": f"{method}_recipe_hash",
        "returncode": 0,
        "seed": 42,
    }


class MethodProvenanceFairnessTests(unittest.TestCase):
    def compare_siesta(self, configs: dict[str, dict], *, artifacts: dict[str, dict[str, str]] | None = None):
        module = load_script_module("siesta_settings_method_provenance_tests", "siesta_settings.py")
        shared, _, _, _ = siesta_fixtures()
        return module.compare_method_settings(
            configs,
            shared,
            artifact_hashes_by_method=artifacts,
            selected_methods=["md", "siesta_fc_cartesian", "random_cartesian"],
        )

    def compare_models(self, configs: dict[str, dict]):
        module = load_script_module("model_settings_method_provenance_tests", "model_settings.py")
        return module.compare_method_model_settings(
            configs,
            selected_methods=["md", "siesta_fc_cartesian", "random_cartesian"],
        )

    def test_equivalent_siesta_settings_have_hashes_and_no_severe_warning(self) -> None:
        _, md, fc, rc = siesta_fixtures()
        artifacts = {
            method: {"basis_hash": "basis_same", "pseudopotential_hash": "pseudo_same"}
            for method in ("md", "siesta_fc_cartesian", "random_cartesian")
        }
        report = self.compare_siesta(
            {"md": md, "siesta_fc_cartesian": fc, "random_cartesian": rc},
            artifacts=artifacts,
        )
        self.assertTrue(report["ok"])
        self.assertEqual(report["pairwise_mismatch_report"], [])
        self.assertEqual(report["severe_mismatches"], [])
        self.assertEqual(
            set(report["siesta_settings_hash_by_method"]),
            {"md", "siesta_fc_cartesian", "random_cartesian"},
        )
        self.assertTrue(all(report["siesta_settings_hash_by_method"].values()))
        self.assertEqual(report["basis_hash_by_method"]["random_cartesian"], "basis_same")

    def test_random_cartesian_meshcutoff_mismatch_is_severe(self) -> None:
        _, md, fc, rc = siesta_fixtures()
        rc["structure"]["random_cartesian"]["siesta"]["MeshCutoff"] = "300 Ry"
        report = self.compare_siesta({"md": md, "siesta_fc_cartesian": fc, "random_cartesian": rc})
        self.assertFalse(report["ok"])
        self.assertTrue(report["severe_warning"])
        self.assertTrue(
            any(
                mismatch["key"] == "MeshCutoff" and "random_cartesian" in mismatch["methods"]
                for mismatch in report["severe_mismatches"]
            )
        )

    def test_hamiltonian_output_flag_mismatch_is_method_provenance_severe(self) -> None:
        _, md, fc, rc = siesta_fixtures()
        fc["structure"]["siesta"]["Save.HS"] = "F"
        report = self.compare_siesta({"md": md, "siesta_fc_cartesian": fc, "random_cartesian": rc})
        self.assertFalse(report["ok"])
        severe = [
            mismatch
            for mismatch in report["severe_mismatches"]
            if mismatch["key"] == "Save.HS"
        ]
        self.assertTrue(severe)
        self.assertTrue(all(mismatch["severity"] == "severe" for mismatch in severe))

        module = load_script_module("pipeline_ui_output_flag_provenance_tests", "pipeline_ui.py")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = {
                "experiment_id": "output_flag_mismatch_case",
                "selected_methods": ["md", "siesta_fc_cartesian"],
                "runs": [
                    provenance_run(root, "md", "md_190", 190),
                    provenance_run(root, "siesta_fc_cartesian", "fc_190", 190),
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
                "siesta_settings_severe_mismatches": severe,
            }
            module.refresh_method_provenance(manifest)

        self.assertTrue(
            any("Save.HS" in warning for warning in manifest["method_provenance_severe_warnings"])
        )

    def test_fc_basis_hash_mismatch_is_severe(self) -> None:
        _, md, fc, rc = siesta_fixtures()
        artifacts = {
            "md": {"basis_hash": "basis_same", "pseudopotential_hash": "pseudo_same"},
            "siesta_fc_cartesian": {"basis_hash": "basis_fc_different", "pseudopotential_hash": "pseudo_same"},
            "random_cartesian": {"basis_hash": "basis_same", "pseudopotential_hash": "pseudo_same"},
        }
        report = self.compare_siesta(
            {"md": md, "siesta_fc_cartesian": fc, "random_cartesian": rc},
            artifacts=artifacts,
        )
        severe = [
            mismatch
            for mismatch in report["severe_mismatches"]
            if mismatch["type"] == "basis_pseudopotential" and mismatch["key"] == "basis_hash"
        ]
        self.assertFalse(report["ok"])
        self.assertTrue(severe)
        self.assertTrue(any("siesta_fc_cartesian" in mismatch["methods"] for mismatch in severe))
        self.assertTrue(report["basis_pseudopotential_warning"])

    def test_random_cartesian_graph2mat_architecture_mismatch_is_severe(self) -> None:
        md, fc, rc = model_fixtures()
        rc["training"]["model"]["num_interactions"] = 2
        report = self.compare_models({"md": md, "siesta_fc_cartesian": fc, "random_cartesian": rc})
        severe = [
            mismatch
            for mismatch in report["severe_mismatches"]
            if mismatch["section"] == "model" and mismatch["key"] == "num_interactions"
        ]
        self.assertFalse(report["ok"])
        self.assertTrue(severe)
        self.assertTrue(any("random_cartesian" in mismatch["methods"] for mismatch in severe))

    def test_fc_graph2mat_loss_mismatch_is_severe(self) -> None:
        md, fc, rc = model_fixtures()
        fc["training"]["model"]["loss"] = "graph2mat.metrics.mse"
        report = self.compare_models({"md": md, "siesta_fc_cartesian": fc, "random_cartesian": rc})
        severe = [
            mismatch
            for mismatch in report["severe_mismatches"]
            if mismatch["section"] == "model" and mismatch["key"] == "loss"
        ]
        self.assertFalse(report["ok"])
        self.assertTrue(severe)
        self.assertTrue(any("siesta_fc_cartesian" in mismatch["methods"] for mismatch in severe))

    def test_dataset_path_only_model_differences_have_no_severe_warning(self) -> None:
        md, fc, rc = model_fixtures()
        md["training"]["data"]["dataset_path"] = "/tmp/md/dataset"
        fc["training"]["data"]["dataset_path"] = "/tmp/fc/dataset"
        rc["training"]["data"]["dataset_path"] = "/tmp/rc/dataset"
        md["training"]["trainer"]["default_root_dir"] = "/tmp/md/out"
        fc["training"]["trainer"]["default_root_dir"] = "/tmp/fc/out"
        rc["training"]["trainer"]["default_root_dir"] = "/tmp/rc/out"
        report = self.compare_models({"md": md, "siesta_fc_cartesian": fc, "random_cartesian": rc})
        self.assertTrue(report["ok"])
        self.assertEqual(report["pairwise_mismatch_report"], [])
        self.assertEqual(report["severe_mismatches"], [])
        self.assertEqual(
            set(report["model_config_hash_by_method"]),
            {"md", "siesta_fc_cartesian", "random_cartesian"},
        )
        self.assertEqual(len(set(report["model_config_hash_by_method"].values())), 1)

    def test_final_manifest_method_provenance_includes_three_explicit_methods(self) -> None:
        module = load_script_module("pipeline_ui_method_provenance_tests", "pipeline_ui.py")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = [
                provenance_run(root, "md", "md_190", 190),
                provenance_run(root, "siesta_fc_cartesian", "fc_190", 190),
                provenance_run(root, "random_cartesian", "rc_3500", 3500),
            ]
            manifest = {
                "experiment_id": "method_provenance_case",
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

        provenance = manifest["method_provenance"]
        self.assertEqual(set(provenance), {"md", "siesta_fc_cartesian", "random_cartesian"})
        self.assertNotIn("atom_displacement", provenance)
        self.assertEqual(provenance["random_cartesian"]["dataset_label"], "rc_3500")
        self.assertEqual(provenance["random_cartesian"]["siesta_settings_hash"], "siesta_rc")
        self.assertEqual(provenance["random_cartesian"]["model_settings_hash"], "model_rc")
        self.assertEqual(provenance["random_cartesian"]["basis_hash"], "basis_rc")
        self.assertEqual(provenance["random_cartesian"]["pseudopotential_hash"], "pseudo_rc")
        self.assertFalse(manifest["method_provenance_severe_warnings"])

    def test_method_provenance_warnings_are_aggregated_as_blocking_warnings(self) -> None:
        module = load_script_module("aggregate_cross_method_provenance_warning_tests", "aggregate_cross_metrics.py")
        with tempfile.TemporaryDirectory() as tmp:
            result_dir = Path(tmp) / "md__on__test_md"
            metrics_dir = result_dir / "metrics"
            metrics_dir.mkdir(parents=True)
            (metrics_dir / "sparse_metrics.csv").write_text(
                "sample,relative_frobenius_union\nsample_1,0.2\n",
                encoding="utf-8",
            )
            (metrics_dir / "spectral_metrics.csv").write_text(
                "sample,low_energy_rmse_eV\nsample_1,0.1\n",
                encoding="utf-8",
            )
            (result_dir / "cross_evaluation_manifest.json").write_text(
                json.dumps(
                    {
                        "train_method": "md",
                        "test_set": "test_md",
                        "method_provenance_warnings": ["md: Missing SIESTA settings hash."],
                        "method_provenance_severe_warnings": [
                            "md: Severe SIESTA settings mismatch: Save.HS"
                        ],
                    }
                ),
                encoding="utf-8",
            )

            rows = module.aggregate_one(result_dir, "warning_case")

        self.assertEqual(len(rows), 1)
        self.assertIn("Missing SIESTA settings hash", rows[0]["severe_warnings"])
        self.assertIn("Save.HS", rows[0]["severe_warnings"])
        self.assertIn("Missing SIESTA settings hash", rows[0]["method_provenance_warnings"])
        self.assertIn("Save.HS", rows[0]["method_provenance_severe_warnings"])

    def test_winner_analysis_treats_method_provenance_warning_as_severe(self) -> None:
        module = load_script_module("analyze_winners_method_provenance_warning_tests", "analyze_winners.py")
        rows = [
            {
                "train_method": "md",
                "test_set": "test_md",
                "low_energy_rmse_eV": 0.2,
                "method_provenance_warnings": "md: Missing SIESTA settings hash.",
                "seed": 1,
            },
            {
                "train_method": "siesta_fc_cartesian",
                "test_set": "test_md",
                "low_energy_rmse_eV": 0.3,
                "seed": 1,
            },
        ]
        recommendation = module.build_recommendation(
            rows,
            summary_rows=[],
            pair_rows=[],
            primary_metric="low_energy_rmse_eV",
        )

        self.assertIn("Missing SIESTA settings hash", " | ".join(recommendation["severe_warnings"]))
        self.assertNotEqual(recommendation["scientific_status"], "robust_comparison")
        self.assertIsNone(recommendation["winner"])

    def test_md_blocked_split_has_temporal_gap_between_partitions(self) -> None:
        module = load_repo_script_module(
            "md_blocked_split_fairness_tests",
            "MD/scripts/generate_md_dataset.py",
            REPO_ROOT / "MD" / "scripts",
        )
        items = [Path(str(index)) for index in range(10)]
        split_ranges, excluded = module._split_blocked_with_gap(
            items,
            {"train": 4, "validation": 2, "test": 2},
            temporal_gap=1,
            block_order=["train", "validation", "test"],
        )

        split_by_frame = {
            int(path.name): split
            for split, paths in split_ranges.items()
            for path in paths
        }
        for frame, split in split_by_frame.items():
            for neighbor in (frame - 1, frame + 1):
                if neighbor in split_by_frame:
                    self.assertEqual(split, split_by_frame[neighbor])
        self.assertEqual([int(path.name) for path, _reason in excluded], [4, 7])

    def test_md_spread_split_summary_is_marked_exploratory(self) -> None:
        module = load_repo_script_module(
            "md_spread_split_fairness_tests",
            "MD/scripts/generate_md_dataset.py",
            REPO_ROOT / "MD" / "scripts",
        )
        with tempfile.TemporaryDirectory() as tmp:
            split_root = Path(tmp)
            module.write_split_summary(
                split_root,
                {"train": [Path("0")], "validation": [Path("2")], "test": [Path("4")]},
                [],
                strategy="spread",
                temporal_gap=0,
                warnings=[module.SPREAD_SPLIT_WARNING],
            )
            summary = json.loads((split_root / "split_summary.json").read_text(encoding="utf-8"))

        self.assertEqual(summary["scientific_status"], "exploratory_temporal_leakage_risk")
        self.assertTrue(any("interleaves trajectory frames" in warning for warning in summary["warnings"]))

    def test_random_cartesian_grouped_split_keeps_family_together(self) -> None:
        module = load_repo_script_module(
            "random_cartesian_group_split_fairness_tests",
            "AtomDisplacement/scripts/generate_random_cartesian_dataset.py",
            REPO_ROOT / "AtomDisplacement" / "scripts",
        )
        samples = [
            {"sample_id": "a1", "split_group_id": "family_a"},
            {"sample_id": "a2", "split_group_id": "family_a"},
            {"sample_id": "b1", "split_group_id": "family_b"},
            {"sample_id": "c1", "split_group_id": "family_c"},
        ]
        split_samples, summary = module.grouped_split_assignment(samples)
        module.assert_group_isolation(split_samples)

        family_a_splits = [
            split
            for split, rows in split_samples.items()
            if any(row["sample_id"].startswith("a") for row in rows)
        ]
        self.assertEqual(family_a_splits, ["train"])
        self.assertTrue(summary["group_aware"])
        self.assertEqual(summary["scientific_status"], "valid_grouped_family_splits")

    def test_random_cartesian_group_isolation_rejects_split_family(self) -> None:
        module = load_repo_script_module(
            "random_cartesian_group_isolation_fairness_tests",
            "AtomDisplacement/scripts/generate_random_cartesian_dataset.py",
            REPO_ROOT / "AtomDisplacement" / "scripts",
        )
        with self.assertRaisesRegex(RuntimeError, "split a family"):
            module.assert_group_isolation(
                {
                    "train": [{"sample_id": "a1", "split_group_id": "family_a"}],
                    "validation": [],
                    "test": [{"sample_id": "a2", "split_group_id": "family_a"}],
                }
            )

    def test_random_cartesian_provenance_is_not_hidden_under_atom_displacement(self) -> None:
        module = load_script_module("pipeline_ui_method_provenance_alias_tests", "pipeline_ui.py")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = {
                "experiment_id": "rc_provenance_case",
                "selected_methods": ["random_cartesian"],
                "runs": [provenance_run(root, "random_cartesian", "rc_570", 570)],
                "siesta_settings_hash_by_method": {"random_cartesian": "siesta_rc"},
                "model_config_hash_by_method": {"random_cartesian": "model_rc"},
                "basis_hash_by_method": {"random_cartesian": "basis_rc"},
                "pseudopotential_hash_by_method": {"random_cartesian": "pseudo_rc"},
            }
            module.refresh_method_provenance(manifest)

        self.assertEqual(set(manifest["method_provenance"]), {"random_cartesian"})
        rc_entry = manifest["method_provenance"]["random_cartesian"]
        self.assertEqual(rc_entry["method_id"], "random_cartesian")
        self.assertEqual(rc_entry["dataset_size"], 570)
        self.assertEqual(rc_entry["runs"][0]["pipeline"], "random_cartesian")
        self.assertNotEqual(rc_entry["runs"][0]["pipeline"], "atom_displacement")


if __name__ == "__main__":
    unittest.main()
