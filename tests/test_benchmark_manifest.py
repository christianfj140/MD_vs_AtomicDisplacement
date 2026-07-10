from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SHARED_DIR = REPO_ROOT / "shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

from benchmark_manifest import (  # noqa: E402
    build_benchmark_dataset_manifest,
    build_frozen_split_manifest,
    write_benchmark_manifests,
)
from joint_artifact_contract import CONTRACT_NAME, validate_dataset  # noqa: E402


def write_snapshot(path: Path, *, label: str = "graphene", hsx_text: str = "hsx\n") -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "RUN.fdf").write_text(
        "\n".join(
            [
                f"SystemLabel {label}",
                "SaveHS true",
                "Save.HS T",
                "TS.HS.Save T",
                "TS.DE.Save T",
                "XML.Write T",
                "Write.OrbitalIndex T",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (path / "RUN.out").write_text("Job completed\nSCF cycle converged\n", encoding="utf-8")
    (path / "metadata.json").write_text('{"system_label": "%s"}\n' % label, encoding="utf-8")
    for suffix, content in {
        ".TSHS": "tshs\n",
        ".TSDE": "tsde\n",
        ".HSX": hsx_text,
        ".STRUCT_OUT": "struct\n",
        ".XV": "xv\n",
        ".ORB_INDX": "orb\n",
    }.items():
        (path / f"{label}{suffix}").write_text(content, encoding="utf-8")


def write_split_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["sample_id", "split", "sample_dir", "structure_path", "hamiltonian_path", "metadata_path"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class BenchmarkManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.dataset = self.root / "dataset"
        self.steps = self.dataset / "MD_steps"
        self.split_root = self.dataset / "splits"
        self.dataset.mkdir(parents=True)
        (self.dataset / "RUN.fdf").write_text("SystemLabel graphene\nSave.HS T\n", encoding="utf-8")
        (self.dataset / "RUN.out").write_text("Job completed\n", encoding="utf-8")
        (self.dataset / "material_provenance.json").write_text(
            json.dumps(
                {
                    "label": "graphene",
                    "fdf": "materials/graphene/RUN.fdf",
                    "fdf_sha256": "fdfhash",
                    "basis_file_sha256": {"C.ion.xml": "basis"},
                    "pseudopotential_sha256": {"C": "pseudo"},
                    "siesta_version": "SIESTA 5.4.2-test",
                    "siesta_executable": "siesta",
                    "siesta_command_line": "bash -lc 'siesta < RUN.fdf'",
                    "run_out_path": str(self.dataset / "RUN.out"),
                    "siesta_returncode": 0,
                    "environment": {
                        "python_version": "3.11.0",
                        "platform": "test-platform",
                        "SECRET_TOKEN": "must-not-serialize",
                    },
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def prepare_dataset(self, *, corrupt_validation: bool = False) -> dict:
        samples = []
        for index, split in enumerate(("train", "validation", "test")):
            step = self.steps / str(index)
            write_snapshot(step)
            split_sample = self.split_root / split / str(index)
            write_snapshot(split_sample)
            samples.append((split, index, split_sample))

        artifact_validation = validate_dataset(
            self.steps,
            snapshot_dirs=[self.steps / "0", self.steps / "1", self.steps / "2"],
        ).to_dict()
        if corrupt_validation:
            artifact_validation["valid"] = False
            artifact_validation["warnings"] = ["forced invalid for test"]
        (self.dataset / "artifact_validation.json").write_text(
            json.dumps(artifact_validation, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        for split, index, sample_dir in samples:
            write_split_csv(
                self.split_root / f"{split}_manifest.csv",
                [
                    {
                        "sample_id": f"md_{index}",
                        "split": split,
                        "sample_dir": str(sample_dir),
                        "structure_path": str(sample_dir / "RUN.fdf"),
                        "hamiltonian_path": str(sample_dir / "graphene.TSHS"),
                        "metadata_path": str(sample_dir / "metadata.json"),
                    }
                ],
            )
        return artifact_validation

    def test_manifest_contains_required_fields(self) -> None:
        self.prepare_dataset()

        dataset_manifest, frozen_split = write_benchmark_manifests(
            dataset_root=self.dataset,
            split_root=self.split_root,
        )

        self.assertEqual(dataset_manifest["artifact_contract_version"], CONTRACT_NAME)
        self.assertEqual(dataset_manifest["material_label"], "graphene")
        self.assertEqual(dataset_manifest["system_label"], "graphene")
        self.assertEqual(dataset_manifest["generation_mode"], "clean_one_pass")
        self.assertEqual(dataset_manifest["validation_status"], "valid")
        self.assertTrue(dataset_manifest["siesta_input_sha256"])
        self.assertEqual(dataset_manifest["siesta_version"], "SIESTA 5.4.2-test")
        self.assertEqual(dataset_manifest["siesta_command_line"], "bash -lc 'siesta < RUN.fdf'")
        self.assertEqual(dataset_manifest["siesta_returncode"], 0)
        self.assertEqual(dataset_manifest["environment"]["python_version"], "3.11.0")
        self.assertNotIn("SECRET_TOKEN", dataset_manifest["environment"])
        self.assertEqual(dataset_manifest["basis_hashes"], {"C.ion.xml": "basis"})
        self.assertEqual(dataset_manifest["pseudopotential_hashes"], {"C": "pseudo"})
        self.assertTrue(dataset_manifest["provenance_status"]["valid"])
        self.assertEqual(len(dataset_manifest["samples"]), 3)
        self.assertEqual(frozen_split["split_counts"], {"train": 1, "validation": 1, "test": 1})

    def test_split_hash_is_deterministic(self) -> None:
        self.prepare_dataset()

        first = build_frozen_split_manifest(self.dataset, self.split_root)
        second = build_frozen_split_manifest(self.dataset, self.split_root)

        self.assertEqual(first["split_hash"], second["split_hash"])

    def test_graph2mat_and_deeph_split_sample_ids_are_identical(self) -> None:
        self.prepare_dataset()

        frozen_split = build_frozen_split_manifest(self.dataset, self.split_root)

        for row in frozen_split["rows"]:
            self.assertEqual(row["graph2mat_sample_id"], row["deeph_sample_id"])
            self.assertEqual(row["sample_id"], row["graph2mat_sample_id"])

    def test_changing_one_artifact_changes_relevant_hash(self) -> None:
        self.prepare_dataset()
        first = build_frozen_split_manifest(self.dataset, self.split_root)

        train_hsx = self.split_root / "train" / "0" / "graphene.HSX"
        train_hsx.write_text("changed\n", encoding="utf-8")
        second = build_frozen_split_manifest(self.dataset, self.split_root)

        self.assertNotEqual(
            first["rows"][0]["artifact_sha256"]["reference_hsx"],
            second["rows"][0]["artifact_sha256"]["reference_hsx"],
        )
        self.assertNotEqual(first["split_hash"], second["split_hash"])

    def test_invalid_artifact_contract_prevents_valid_manifest_status(self) -> None:
        artifact_validation = self.prepare_dataset(corrupt_validation=True)
        frozen_split = build_frozen_split_manifest(self.dataset, self.split_root)

        dataset_manifest = build_benchmark_dataset_manifest(
            self.dataset,
            artifact_validation=artifact_validation,
            frozen_split_manifest=frozen_split,
            material_provenance={"label": "graphene"},
        )

        self.assertFalse(dataset_manifest["benchmark_ready"])
        self.assertEqual(dataset_manifest["validation_status"], "invalid")

    def test_missing_basis_provenance_prevents_benchmark_ready_manifest(self) -> None:
        artifact_validation = self.prepare_dataset()
        frozen_split = build_frozen_split_manifest(self.dataset, self.split_root)

        dataset_manifest = build_benchmark_dataset_manifest(
            self.dataset,
            artifact_validation=artifact_validation,
            frozen_split_manifest=frozen_split,
            material_provenance={
                "label": "graphene",
                "pseudopotential_sha256": {"C": "pseudo"},
                "fdf_sha256": "fdfhash",
            },
        )

        self.assertFalse(dataset_manifest["benchmark_ready"])
        self.assertIn("basis_provenance", dataset_manifest["provenance_status"]["missing"])

    def test_missing_pseudopotential_provenance_prevents_benchmark_ready_manifest(self) -> None:
        artifact_validation = self.prepare_dataset()
        frozen_split = build_frozen_split_manifest(self.dataset, self.split_root)

        dataset_manifest = build_benchmark_dataset_manifest(
            self.dataset,
            artifact_validation=artifact_validation,
            frozen_split_manifest=frozen_split,
            material_provenance={
                "label": "graphene",
                "basis_file_sha256": {"C.ion.xml": "basis"},
                "fdf_sha256": "fdfhash",
            },
        )

        self.assertFalse(dataset_manifest["benchmark_ready"])
        self.assertIn("pseudopotential_provenance", dataset_manifest["provenance_status"]["missing"])

    def test_pseudopotential_sha256_by_source_satisfies_provenance(self) -> None:
        # mixed_dataset_materialize.py writes this key (not the flat
        # pseudopotential_sha256) when small/large pseudopotentials differ,
        # e.g. the small (W90) pool carries Ghost-H and the large (5x5) pool
        # doesn't. The gate must accept it as real provenance.
        artifact_validation = self.prepare_dataset()
        frozen_split = build_frozen_split_manifest(self.dataset, self.split_root)

        dataset_manifest = build_benchmark_dataset_manifest(
            self.dataset,
            artifact_validation=artifact_validation,
            frozen_split_manifest=frozen_split,
            material_provenance={
                "label": "graphene",
                "basis_file_sha256": {"C.ion.xml": "basis"},
                "pseudopotential_sha256_by_source": {
                    "small": {"C": "pseudo", "Ghost-H": "pseudo-ghost"},
                    "large": {"C": "pseudo"},
                },
                "fdf_sha256": "fdfhash",
            },
        )

        self.assertNotIn("pseudopotential_provenance", dataset_manifest["provenance_status"]["missing"])

    def test_missing_material_identity_prevents_benchmark_ready_manifest(self) -> None:
        artifact_validation = self.prepare_dataset()
        frozen_split = build_frozen_split_manifest(self.dataset, self.split_root)

        dataset_manifest = build_benchmark_dataset_manifest(
            self.dataset,
            artifact_validation=artifact_validation,
            frozen_split_manifest=frozen_split,
            material_provenance={
                "basis_file_sha256": {"C.ion.xml": "basis"},
                "pseudopotential_sha256": {"C": "pseudo"},
                "fdf_sha256": "fdfhash",
            },
        )

        self.assertFalse(dataset_manifest["benchmark_ready"])
        self.assertIn("material_identity", dataset_manifest["provenance_status"]["missing"])

    def test_strict_paper_ready_manifest_requires_siesta_version(self) -> None:
        artifact_validation = self.prepare_dataset()
        frozen_split = build_frozen_split_manifest(self.dataset, self.split_root)

        dataset_manifest = build_benchmark_dataset_manifest(
            self.dataset,
            artifact_validation=artifact_validation,
            frozen_split_manifest=frozen_split,
            material_provenance={
                "label": "graphene",
                "basis_file_sha256": {"C.ion.xml": "basis"},
                "pseudopotential_sha256": {"C": "pseudo"},
                "fdf_sha256": "fdfhash",
                "siesta_command_line": "bash -lc 'siesta < RUN.fdf'",
                "run_out_path": str(self.dataset / "RUN.out"),
                "environment": {"python_version": "3.11.0", "platform": "test-platform"},
            },
            strict_paper_ready_provenance=True,
        )

        self.assertFalse(dataset_manifest["benchmark_ready"])
        self.assertIn("siesta_version_provenance", dataset_manifest["provenance_status"]["missing"])

    def test_strict_paper_ready_manifest_requires_command_line(self) -> None:
        artifact_validation = self.prepare_dataset()
        frozen_split = build_frozen_split_manifest(self.dataset, self.split_root)

        dataset_manifest = build_benchmark_dataset_manifest(
            self.dataset,
            artifact_validation=artifact_validation,
            frozen_split_manifest=frozen_split,
            material_provenance={
                "label": "graphene",
                "basis_file_sha256": {"C.ion.xml": "basis"},
                "pseudopotential_sha256": {"C": "pseudo"},
                "fdf_sha256": "fdfhash",
                "siesta_version": "SIESTA 5.4.2-test",
                "run_out_path": str(self.dataset / "RUN.out"),
                "environment": {"python_version": "3.11.0", "platform": "test-platform"},
            },
            strict_paper_ready_provenance=True,
        )

        self.assertFalse(dataset_manifest["benchmark_ready"])
        self.assertIn("siesta_command_line_provenance", dataset_manifest["provenance_status"]["missing"])

    def test_strict_paper_ready_manifest_passes_with_execution_provenance(self) -> None:
        artifact_validation = self.prepare_dataset()
        frozen_split = build_frozen_split_manifest(self.dataset, self.split_root)
        material = json.loads((self.dataset / "material_provenance.json").read_text(encoding="utf-8"))

        dataset_manifest = build_benchmark_dataset_manifest(
            self.dataset,
            artifact_validation=artifact_validation,
            frozen_split_manifest=frozen_split,
            material_provenance=material,
            strict_paper_ready_provenance=True,
        )

        self.assertTrue(dataset_manifest["benchmark_ready"])
        self.assertTrue(dataset_manifest["provenance_status"]["siesta_environment_provenance"])
        self.assertTrue(dataset_manifest["provenance_status"]["siesta_execution_log_provenance"])


if __name__ == "__main__":
    unittest.main()
