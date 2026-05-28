import configparser
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "Comparison" / "scripts"
SHARED_DIR = REPO_ROOT / "shared"
for path in (SCRIPTS_DIR, SHARED_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from deeph_config import (  # noqa: E402
    build_deeph_raw_mirror,
    deeph_orbital_json_from_raw_mirror,
    deeph_orbital_list_from_siesta_sample,
    ordered_rows_for_deeph_split,
    render_inference_config,
    render_preprocess_config,
    render_train_config,
    validate_deeph_siesta_sample,
)


def write_snapshot(path: Path, *, missing_suffix: str | None = None) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "RUN.fdf").write_text("SystemLabel graphene\n", encoding="utf-8")
    (path / "RUN.out").write_text("Job completed\n", encoding="utf-8")
    (path / "metadata.json").write_text(json.dumps({"system_label": "graphene"}) + "\n", encoding="utf-8")
    for suffix in (".TSHS", ".TSDE", ".HSX", ".XV"):
        if suffix == missing_suffix:
            continue
        (path / f"graphene{suffix}").write_text(f"{suffix}\n", encoding="utf-8")
    if missing_suffix != ".STRUCT_OUT":
        (path / "graphene.STRUCT_OUT").write_text(
            "1 0 0\n0 1 0\n0 0 1\n1\n1 6 0 0 0\n",
            encoding="utf-8",
        )
    if missing_suffix != ".ORB_INDX":
        (path / "graphene.ORB_INDX").write_text(
            "      2     2 = orbitals in unit cell and supercell. See end of file.\n\n"
            "io ia is spec iao n l m z p sym rc isc iuo\n"
            "1 1 1 C 1 2 0 0 1 F s 4.0 0 0 0 1\n"
            "2 1 1 C 2 2 1 -1 1 F py 4.0 0 0 0 2\n",
            encoding="utf-8",
        )


def frozen_split_for(samples: list[tuple[str, str, Path]]) -> dict:
    rows = [
        {
            "sample_id": sample_id,
            "split": split,
            "sample_dir": str(sample_dir),
            "graph2mat_sample_id": sample_id,
            "deeph_sample_id": sample_id,
        }
        for sample_id, split, sample_dir in samples
    ]
    return {
        "valid": True,
        "split_hash": "unit-split",
        "split_counts": {
            "train": sum(1 for _, split, _ in samples if split == "train"),
            "validation": sum(1 for _, split, _ in samples if split == "validation"),
            "test": sum(1 for _, split, _ in samples if split == "test"),
        },
        "rows": rows,
    }


class DeepHConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_preprocess_ini_renders_required_siesta_fields(self) -> None:
        path = self.root / "config" / "preprocess.ini"
        render_preprocess_config(
            path,
            raw_dir=self.root / "raw",
            processed_dir=self.root / "processed",
            multiprocessing=2,
            local_coordinate=True,
            get_s=True,
        )

        config = configparser.ConfigParser()
        config.read(path)

        self.assertEqual(config["basic"]["raw_dir"], str(self.root / "raw"))
        self.assertEqual(config["basic"]["processed_dir"], str(self.root / "processed"))
        self.assertEqual(config["basic"]["interface"], "siesta")
        self.assertEqual(config["basic"]["target"], "hamiltonian")
        self.assertEqual(config["basic"]["get_S"], "True")

    def test_train_ini_renders_required_h5_fields(self) -> None:
        path = self.root / "config" / "train.ini"
        render_train_config(
            path,
            processed_dir=self.root / "processed",
            graph_dir=self.root / "graph",
            save_dir=self.root / "train",
            dataset_name="graphene_unit",
            split_ratios={"train_ratio": "0.5", "val_ratio": "0.25", "test_ratio": "0.25"},
            seed=17,
            epochs=12,
            batch_size=4,
            learning_rate=0.002,
            disable_cuda=True,
            device="cpu",
        )

        config = configparser.ConfigParser()
        config.read(path)

        self.assertEqual(config["basic"]["raw_dir"], str(self.root / "processed"))
        self.assertEqual(config["basic"]["graph_dir"], str(self.root / "graph"))
        self.assertEqual(config["basic"]["save_dir"], str(self.root / "train"))
        self.assertEqual(config["basic"]["interface"], "h5")
        self.assertEqual(config["basic"]["target"], "hamiltonian")
        self.assertEqual(config["train"]["epochs"], "12")
        self.assertEqual(config["hyperparameter"]["batch_size"], "4")

    def test_train_ini_renders_sweep_overrides(self) -> None:
        path = self.root / "config" / "train_sweep.ini"
        render_train_config(
            path,
            processed_dir=self.root / "processed",
            graph_dir=self.root / "graph",
            save_dir=self.root / "train",
            dataset_name="graphene_unit",
            split_ratios={"train_ratio": "0.5", "val_ratio": "0.25", "test_ratio": "0.25"},
            seed=17,
            epochs=12,
            batch_size=4,
            learning_rate=0.002,
            disable_cuda=True,
            device="cpu",
            overrides={
                "optimizer": "adam",
                "weight_decay": 0.01,
                "criterion": "MaskMSELoss",
                "atom_fea_len": 64,
                "edge_fea_len": 128,
                "gauss_stop": 6,
                "num_l": 5,
                "if_edge_update": True,
                "if_lcmp": False,
                "normalization": "LayerNorm",
                "atom_update_net": "CGConv",
                "retain_edge_fea": True,
            },
        )

        config = configparser.ConfigParser()
        config.read(path)

        self.assertEqual(config["hyperparameter"]["optimizer"], "adam")
        self.assertEqual(config["hyperparameter"]["criterion"], "MaskMSELoss")
        self.assertEqual(config["hyperparameter"]["retain_edge_fea"], "True")
        self.assertEqual(config["network"]["atom_fea_len"], "64")
        self.assertEqual(config["network"]["edge_fea_len"], "128")
        self.assertEqual(config["network"]["num_l"], "5")
        self.assertEqual(config["network"]["if_lcmp"], "False")

    def test_train_ini_renders_basis_derived_orbital_mask(self) -> None:
        path = self.root / "config" / "train_orbital.ini"
        orbital = [{"6 6": [0, 0]}, {"6 6": [0, 1]}, {"6 6": [1, 0]}, {"6 6": [1, 1]}]
        render_train_config(
            path,
            processed_dir=self.root / "processed",
            graph_dir=self.root / "graph",
            save_dir=self.root / "train",
            dataset_name="graphene_unit",
            split_ratios={"train_ratio": "0.5", "val_ratio": "0.25", "test_ratio": "0.25"},
            seed=17,
            epochs=12,
            batch_size=4,
            learning_rate=0.002,
            disable_cuda=True,
            device="cpu",
            orbital=orbital,
        )

        config = configparser.ConfigParser()
        config.read(path)

        self.assertEqual(json.loads(config["basic"]["orbital"]), orbital)

    def test_orbital_mask_is_derived_from_siesta_orb_indx(self) -> None:
        sample = self.root / "sample_orbitals"
        sample.mkdir()
        (sample / "graphene.STRUCT_OUT").write_text(
            "1 0 0\n0 1 0\n0 0 1\n2\n1 6 0 0 0\n1 6 0.5 0.5 0\n",
            encoding="utf-8",
        )
        (sample / "graphene.ORB_INDX").write_text(
            "      8     8 = orbitals in unit cell and supercell. See end of file.\n\n"
            "io ia is spec iao n l m z p sym rc isc iuo\n"
            "1 1 1 C 1 2 0 0 1 F s 4.0 0 0 0 1\n"
            "2 1 1 C 2 2 1 -1 1 F py 4.0 0 0 0 2\n"
            "3 1 1 C 3 2 1 0 1 F pz 4.0 0 0 0 3\n"
            "4 1 1 C 4 2 1 1 1 F px 4.0 0 0 0 4\n"
            "5 2 1 C 1 2 0 0 1 F s 4.0 0 0 0 5\n"
            "6 2 1 C 2 2 1 -1 1 F py 4.0 0 0 0 6\n"
            "7 2 1 C 3 2 1 0 1 F pz 4.0 0 0 0 7\n"
            "8 2 1 C 4 2 1 1 1 F px 4.0 0 0 0 8\n",
            encoding="utf-8",
        )

        orbital = deeph_orbital_list_from_siesta_sample(sample)

        self.assertEqual(len(orbital), 16)
        self.assertEqual(orbital[0], {"6 6": [0, 0]})
        self.assertEqual(orbital[-1], {"6 6": [3, 3]})

    def test_orbital_json_from_raw_mirror_uses_raw_sample(self) -> None:
        sample = self.root / "sample_raw_mirror_orbitals"
        sample.mkdir()
        (sample / "graphene.STRUCT_OUT").write_text(
            "1 0 0\n0 1 0\n0 0 1\n1\n1 6 0 0 0\n",
            encoding="utf-8",
        )
        (sample / "graphene.ORB_INDX").write_text(
            "      2     2 = orbitals in unit cell and supercell. See end of file.\n\n"
            "io ia is spec iao n l m z p sym rc isc iuo\n"
            "1 1 1 C 1 2 0 0 1 F s 4.0 0 0 0 1\n"
            "2 1 1 C 2 2 1 -1 1 F py 4.0 0 0 0 2\n",
            encoding="utf-8",
        )

        orbital = json.loads(deeph_orbital_json_from_raw_mirror({"rows": [{"raw_dir": str(sample)}]}))

        self.assertEqual(orbital, [{"6 6": [0, 0]}, {"6 6": [0, 1]}, {"6 6": [1, 0]}, {"6 6": [1, 1]}])

    def test_inference_ini_renders_required_fields(self) -> None:
        path = self.root / "config" / "inference.ini"
        render_inference_config(
            path,
            work_dir=self.root / "inference" / "sample0",
            trained_model_dir=self.root / "train",
            python_interpreter="/usr/bin/python3",
            task=[3, 4],
        )

        config = configparser.ConfigParser()
        config.read(path)

        self.assertEqual(config["basic"]["work_dir"], str(self.root / "inference" / "sample0"))
        self.assertEqual(json.loads(config["basic"]["trained_model_dir"]), [str(self.root / "train")])
        self.assertEqual(json.loads(config["basic"]["task"]), [3, 4])
        self.assertEqual(config["basic"]["interface"], "openmx")
        self.assertEqual(config["interpreter"]["python_interpreter"], "/usr/bin/python3")

    def test_missing_deeph_required_artifact_blocks_preprocess(self) -> None:
        sample = self.root / "sample"
        write_snapshot(sample, missing_suffix=".HSX")

        with self.assertRaisesRegex(RuntimeError, "HSX/STRUCT_OUT/XV/ORB_INDX"):
            validate_deeph_siesta_sample(sample)

    def test_raw_mirror_orders_rows_to_reproduce_frozen_split(self) -> None:
        samples: list[tuple[str, str, Path]] = []
        for index, split in enumerate(("train", "validation", "test")):
            sample_dir = self.root / "source" / str(index)
            write_snapshot(sample_dir)
            samples.append((f"sample{index}", split, sample_dir))
        frozen_split = frozen_split_for(samples)

        mirror = build_deeph_raw_mirror(
            frozen_split,
            raw_dir=self.root / "workspace" / "raw",
            workspace_root=self.root / "workspace",
            seed=123,
        )

        ordered = ordered_rows_for_deeph_split(frozen_split, seed=123)
        self.assertEqual([row["sample_id"] for row in mirror["rows"]], [row["sample_id"] for row in ordered])
        self.assertEqual(mirror["split_ratios"], {"train_ratio": "0.3333333333333333", "val_ratio": "0.3333333333333333", "test_ratio": "0.3333333333333333"})
        for row in mirror["rows"]:
            raw_dir = Path(row["raw_dir"])
            self.assertTrue((raw_dir / "graphene.HSX").exists())
            self.assertFalse((raw_dir / "ML_prediction.HSX").exists())

    def test_raw_mirror_path_must_stay_inside_workspace(self) -> None:
        sample = self.root / "source" / "0"
        write_snapshot(sample)
        frozen_split = frozen_split_for(
            [
                ("train0", "train", sample),
                ("val0", "validation", sample),
                ("test0", "test", sample),
            ]
        )

        with self.assertRaisesRegex(RuntimeError, "inside benchmark workspace"):
            build_deeph_raw_mirror(
                frozen_split,
                raw_dir=self.root / "outside_raw",
                workspace_root=self.root / "workspace",
                seed=1,
            )


if __name__ == "__main__":
    unittest.main()
