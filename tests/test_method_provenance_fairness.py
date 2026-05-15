import copy
import importlib.util
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
