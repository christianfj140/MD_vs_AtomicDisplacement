from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "Comparison" / "scripts" / "diagnose_h2o_one_sample_overfit.py"


def load_module():
    spec = importlib.util.spec_from_file_location("diagnose_h2o_one_sample_overfit_test", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_base_config(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "out_matrix": "hamiltonian",
                    "matrix_component_policy": "h_only",
                    "n_matrix_components": 1,
                    "symmetric_matrix": True,
                    "sub_point_matrix": False,
                    "batch_size": 8,
                    "basis_files": "old/*.ion.xml",
                    "train_runs": "old/train/*/RUN.fdf",
                    "val_runs": "old/validation/*/RUN.fdf",
                },
                "model": {
                    "num_interactions": 1,
                    "correlation": 1,
                    "max_ell": 2,
                    "hidden_irreps": "4x0e + 4x1o",
                    "loss": "graph2mat.core.data.metrics.block_type_huber",
                    "loss_kwargs": {"beta": 0.01},
                    "optim_lr": 0.001,
                },
                "trainer": {
                    "accelerator": "cpu",
                    "logger": {
                        "class_path": "TensorBoardLogger",
                        "init_args": {"name": "old", "save_dir": "old_logs"},
                    },
                    "max_epochs": 12,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def write_evaluation_root(root: Path) -> tuple[Path, Path]:
    structure_dir = root / "structures" / "001"
    reference_dir = root / "siesta_hamiltonians" / "001"
    basis_dir = root / "basis"
    structure_dir.mkdir(parents=True)
    reference_dir.mkdir(parents=True)
    basis_dir.mkdir(parents=True)
    (structure_dir / "RUN.fdf").write_text(
        "\n".join(
            [
                "SystemLabel siesta",
                "%block ChemicalSpeciesLabel",
                "1 1 H",
                "2 8 O",
                "%endblock ChemicalSpeciesLabel",
                "%block AtomicCoordinatesAndAtomicSpecies",
                "0.0 0.0 0.0 2",
                "0.7 0.0 0.0 1",
                "-0.7 0.0 0.0 1",
                "%endblock AtomicCoordinatesAndAtomicSpecies",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (reference_dir / "siesta.TSHS").write_bytes(b"reference")
    h_basis = basis_dir / "H.ion.xml"
    o_basis = basis_dir / "O.ion.xml"
    h_basis.write_text("<basis symbol='H'/>\n", encoding="utf-8")
    o_basis.write_text("<basis symbol='O'/>\n", encoding="utf-8")
    return h_basis, o_basis


class H2OOneSampleOverfitDiagnosticTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_module()

    def test_dry_run_creates_h_only_batch_one_config_and_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evaluation_root = root / "eval"
            workspace = root / "diagnostics" / "one_sample"
            base_config = root / "base_config.yaml"
            write_evaluation_root(evaluation_root)
            write_base_config(base_config)

            result = self.module.main(
                [
                    "--evaluation-root",
                    str(evaluation_root),
                    "--sample",
                    "001",
                    "--workspace",
                    str(workspace),
                    "--base-config",
                    str(base_config),
                    "--max-epochs",
                    "77",
                    "--dry-run",
                ]
            )

            self.assertEqual(result, 0)
            config = yaml.safe_load((workspace / "training" / "config.yaml").read_text(encoding="utf-8"))
            data = config["data"]
            self.assertEqual(data["out_matrix"], "hamiltonian")
            self.assertEqual(data["matrix_component_policy"], "h_only")
            self.assertEqual(data["n_matrix_components"], 1)
            self.assertTrue(data["symmetric_matrix"])
            self.assertEqual(data["batch_size"], 1)
            self.assertEqual(config["trainer"]["max_epochs"], 77)
            self.assertTrue((workspace / "dataset" / "splits" / "train" / "001" / "RUN.fdf").exists())
            self.assertTrue((workspace / "dataset" / "splits" / "validation" / "001" / "RUN.fdf").exists())
            self.assertTrue((workspace / "dataset" / "splits" / "test" / "001" / "RUN.fdf").exists())
            provenance = json.loads((workspace / "provenance.json").read_text(encoding="utf-8"))
            self.assertEqual(provenance["status"], "DRY_RUN")
            self.assertTrue(provenance["dry_run"])
            self.assertEqual(provenance["sample"], "001")
            self.assertIn("command", provenance)
            self.assertIn("structure_sha256", provenance["inputs"])
            self.assertIn("direct_label_diagnostic_template", provenance["commands"])
            self.assertIn("diagnose_h2o_overfit_direct_labels.py", " ".join(provenance["commands"]["direct_label_diagnostic_template"]))
            self.assertEqual(
                provenance["config"]["h_only_target_policy"]["matrix_component_policy"],
                "h_only",
            )

    def test_missing_sample_fails_with_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evaluation_root = root / "eval"
            base_config = root / "base_config.yaml"
            (evaluation_root / "structures").mkdir(parents=True)
            write_base_config(base_config)

            with self.assertRaisesRegex(RuntimeError, "Missing required one-sample inputs"):
                self.module.main(
                    [
                        "--evaluation-root",
                        str(evaluation_root),
                        "--sample",
                        "missing",
                        "--workspace",
                        str(root / "diagnostics"),
                        "--base-config",
                        str(base_config),
                        "--dry-run",
                    ]
                )

    def test_workspace_refuses_production_results_root_without_override(self) -> None:
        production_workspace = REPO_ROOT / "Comparison" / "results" / "results_md" / "bad_diagnostic"
        with self.assertRaisesRegex(RuntimeError, "production output root"):
            self.module.ensure_isolated_workspace(
                production_workspace,
                overwrite=False,
                allow_non_diagnostic_output=False,
            )

    def test_existing_workspace_requires_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "diagnostics"
            workspace.mkdir()
            (workspace / "sentinel.txt").write_text("keep\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "already exists"):
                self.module.ensure_isolated_workspace(
                    workspace,
                    overwrite=False,
                    allow_non_diagnostic_output=False,
                )

    def test_validate_h_only_config_rejects_h_plus_s(self) -> None:
        config = {
            "data": {
                "out_matrix": "hamiltonian",
                "matrix_component_policy": "h_only",
                "n_matrix_components": 2,
                "symmetric_matrix": True,
                "batch_size": 1,
            }
        }
        with self.assertRaisesRegex(RuntimeError, "not H-only safe"):
            self.module.validate_h_only_config(config)


if __name__ == "__main__":
    unittest.main()
