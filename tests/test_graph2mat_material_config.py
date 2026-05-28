from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SHARED_DIR = REPO_ROOT / "shared"
MD_SCRIPTS_DIR = REPO_ROOT / "MD" / "scripts"
for path in (SHARED_DIR, MD_SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from graph2mat_material_config import (  # noqa: E402
    apply_material_graph2mat_config,
    resolve_matrix_component_policy,
    validate_model_matrix_component_policy,
    write_graph2mat_config_provenance,
)


def synthetic_fdf_text() -> str:
    return "\n".join(
        [
            "SystemName synthetic",
            "SystemLabel synthetic",
            "NumberOfSpecies 2",
            "NumberOfAtoms 2",
            "%block ChemicalSpeciesLabel",
            " 1 14 Si",
            " 2 6 C",
            "%endblock ChemicalSpeciesLabel",
            "%block AtomicCoordinatesAndAtomicSpecies",
            " 0.0 0.0 0.0 1",
            " 1.0 0.0 0.0 2",
            "%endblock AtomicCoordinatesAndAtomicSpecies",
            "Save.HS T",
            "XML.Write T",
            "",
        ]
    )


class Graph2MatMaterialConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.write_material()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def write_material(self, *, include_basis: bool = True) -> None:
        material_root = self.root / "materials" / "sic"
        material_root.mkdir(parents=True, exist_ok=True)
        (material_root / "RUN.fdf").write_text(synthetic_fdf_text(), encoding="utf-8")
        pseudo_dir = material_root / "pseudos"
        pseudo_dir.mkdir(exist_ok=True)
        (pseudo_dir / "Si.psf").write_text("si pseudo\n", encoding="utf-8")
        (pseudo_dir / "C.psml").write_text("c pseudo\n", encoding="utf-8")
        basis_dir = material_root / "basis"
        basis_dir.mkdir(exist_ok=True)
        if include_basis:
            (basis_dir / "Si.ion.xml").write_text("<si />\n", encoding="utf-8")
            (basis_dir / "C.ion.xml").write_text("<c />\n", encoding="utf-8")

    def config(self) -> dict:
        return {
            "material": {
                "label": "sic",
                "fdf": "materials/sic/RUN.fdf",
                "pseudopotential_dir": "materials/sic/pseudos",
                "basis_dir": "materials/sic/basis",
                "structure_type": "crystal",
            },
            "training": {
                "data": {
                    "out_matrix": "hamiltonian",
                    "symmetric_matrix": True,
                    "matrix_component_policy": "h_only",
                    "n_matrix_components": 1,
                    "basis_files": "old",
                    "train_runs": "../dataset/splits/train/*/RUN.fdf",
                    "val_runs": "../dataset/splits/validation/*/RUN.fdf",
                },
                "model": {},
                "trainer": {},
            },
            "testing": {
                "data": {
                    "out_matrix": "hamiltonian",
                    "symmetric_matrix": True,
                    "matrix_component_policy": "h_only",
                    "n_matrix_components": 1,
                    "basis_files": "old",
                }
            },
            "prediction": {
                "data": {
                    "out_matrix": "hamiltonian",
                    "symmetric_matrix": True,
                    "matrix_component_policy": "h_only",
                    "n_matrix_components": 1,
                    "basis_files": "old",
                }
            },
        }

    def write_split_files(self, dataset_dir: Path) -> None:
        for split in ("train", "validation", "test"):
            sample_dir = dataset_dir / "splits" / split / "0"
            sample_dir.mkdir(parents=True)
            (sample_dir / "RUN.fdf").write_text(synthetic_fdf_text(), encoding="utf-8")
            manifest = dataset_dir / "splits" / f"{split}_manifest.csv"
            manifest.write_text(
                "sample_id,sample_dir,structure_path,valid,status\n"
                f"0,{sample_dir},{sample_dir / 'RUN.fdf'},True,valid\n",
                encoding="utf-8",
            )

    def test_non_h2o_material_sets_basis_files_and_records_hashes(self) -> None:
        dataset_dir = self.root / "dataset"
        training_dir = self.root / "training"
        training_dir.mkdir()
        self.write_split_files(dataset_dir)
        config = self.config()

        provenance = apply_material_graph2mat_config(
            config,
            base_dir=self.root,
            dataset_dir=dataset_dir,
            training_dir=training_dir,
        )

        self.assertEqual(config["training"]["data"]["basis_files"], "../dataset/material_basis/*.ion.xml")
        self.assertEqual(config["testing"]["data"]["basis_files"], "../dataset/material_basis/*.ion.xml")
        self.assertEqual(config["prediction"]["data"]["basis_files"], "../dataset/material_basis/*.ion.xml")
        self.assertTrue((dataset_dir / "material_basis" / "Si.ion.xml").exists())
        self.assertTrue((dataset_dir / "material_basis" / "C.ion.xml").exists())
        self.assertEqual(provenance["material"]["label"], "sic")
        self.assertEqual(sorted(provenance["graph2mat"]["basis_files_by_species"]), ["C", "Si"])
        self.assertIn("../dataset/splits/train_manifest.csv", provenance["graph2mat"]["split_file_sha256"])
        self.assertEqual(provenance["graph2mat"]["matrix_target"], "hamiltonian")
        self.assertEqual(provenance["graph2mat"]["matrix_component_policy"], "h_only")
        self.assertEqual(provenance["graph2mat"]["n_matrix_components"], 1)

    def test_h_only_policy_with_one_component_passes(self) -> None:
        policy, n_components = resolve_matrix_component_policy(
            {"matrix_component_policy": "h_only", "n_matrix_components": 1},
            context="training.data",
        )

        self.assertEqual(policy, "h_only")
        self.assertEqual(n_components, 1)

    def test_h_only_policy_with_two_components_fails(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "training.data.*h_only.*n_matrix_components=1"):
            resolve_matrix_component_policy(
                {"matrix_component_policy": "h_only", "n_matrix_components": 2},
                context="training.data",
            )

    def test_missing_matrix_component_policy_fails(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "training.data.matrix_component_policy"):
            resolve_matrix_component_policy(
                {"n_matrix_components": 1},
                context="training.data",
            )

    def test_missing_n_matrix_components_fails(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "training.data.n_matrix_components"):
            resolve_matrix_component_policy(
                {"matrix_component_policy": "h_only"},
                context="training.data",
            )

    def test_checkpoint_policy_mismatch_fails_when_metadata_available(self) -> None:
        class Model:
            hparams = {"data": {"matrix_component_policy": "h_and_overlap", "n_matrix_components": 2}}

        with self.assertRaisesRegex(RuntimeError, "checkpoint policy"):
            validate_model_matrix_component_policy(
                Model(),
                matrix_component_policy="h_only",
                n_matrix_components=1,
                context="prediction",
            )

    def test_checkpoint_component_count_mismatch_fails_when_metadata_available(self) -> None:
        class Model:
            hparams = {"data": {"matrix_component_policy": "h_only", "n_matrix_components": 2}}

        with self.assertRaisesRegex(RuntimeError, "checkpoint n_matrix_components=2"):
            validate_model_matrix_component_policy(
                Model(),
                matrix_component_policy="h_only",
                n_matrix_components=1,
                context="testing",
            )

    def test_material_config_generation_requires_training_data(self) -> None:
        config = self.config()
        config.pop("training")

        with self.assertRaisesRegex(RuntimeError, "training.data is required"):
            apply_material_graph2mat_config(
                config,
                base_dir=self.root,
                dataset_dir=self.root / "dataset",
                training_dir=self.root / "training",
            )

    def test_missing_basis_file_fails_clearly(self) -> None:
        (self.root / "materials" / "sic" / "basis" / "C.ion.xml").unlink()

        with self.assertRaisesRegex(RuntimeError, "Missing Graph2Mat basis file for species 'C'"):
            apply_material_graph2mat_config(
                self.config(),
                base_dir=self.root,
                dataset_dir=self.root / "dataset",
                training_dir=self.root / "training",
            )

    def test_config_provenance_sidecar_and_checkpoint_manifest_include_material(self) -> None:
        import md_pipeline_config

        dataset_dir = self.root / "dataset"
        training_dir = self.root / "training"
        training_dir.mkdir()
        self.write_split_files(dataset_dir)
        config = self.config()
        config["paths"] = {
            "dataset_dir": str(dataset_dir),
            "training_dir": str(training_dir),
            "run_fdf_name": "RUN.fdf",
            "run_out_name": "RUN.out",
            "training_config_name": "config.yaml",
            "venv_activate": str(self.root / "venv" / "activate"),
        }
        provenance = apply_material_graph2mat_config(
            config,
            base_dir=self.root,
            dataset_dir=dataset_dir,
            training_dir=training_dir,
        )
        config_path = training_dir / "config.yaml"
        config_path.write_text("data:\n  out_matrix: hamiltonian\n", encoding="utf-8")
        write_graph2mat_config_provenance(
            config_path,
            provenance,
            validation_metadata={"validation_source": "training.data.val_runs"},
        )
        checkpoint = training_dir / "lightning_logs" / "version_0" / "checkpoints" / "best.ckpt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"checkpoint")

        manifest = md_pipeline_config.write_checkpoint_manifest(
            config,
            checkpoint.relative_to(training_dir).as_posix(),
            selection_mode="latest_version",
            selection_metric="val_loss",
        )
        payload = json.loads(manifest.read_text(encoding="utf-8"))

        self.assertEqual(
            payload["graph2mat_config_provenance"]["material"]["label"],
            "sic",
        )
        self.assertEqual(
            payload["graph2mat_config_provenance"]["graph2mat"]["validation"]["validation_source"],
            "training.data.val_runs",
        )

    def test_h2o_preset_remains_supported_from_pipeline_config_dir(self) -> None:
        dataset_dir = self.root / "dataset"
        training_dir = self.root / "training"
        training_dir.mkdir()
        config = {
            "material": {"preset": "h2o"},
            "training": {
                "data": {
                    "out_matrix": "hamiltonian",
                    "matrix_component_policy": "h_only",
                    "n_matrix_components": 1,
                }
            },
        }

        provenance = apply_material_graph2mat_config(
            config,
            base_dir=REPO_ROOT / "MD",
            dataset_dir=dataset_dir,
            training_dir=training_dir,
        )

        self.assertEqual(provenance["material"]["label"], "h2o")
        self.assertTrue((dataset_dir / "material_basis" / "H.ion.xml").exists())
        self.assertTrue((dataset_dir / "material_basis" / "O.ion.xml").exists())


if __name__ == "__main__":
    unittest.main()
