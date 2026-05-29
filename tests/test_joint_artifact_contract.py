from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SHARED_DIR = REPO_ROOT / "shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

from joint_artifact_contract import (  # noqa: E402
    CONTRACT_NAME,
    G2M_DEEPH_BENCHMARK_PROFILE,
    validate_dataset,
    validate_snapshot,
)


def write_snapshot(
    path: Path,
    *,
    label: str = "graphene",
    include_run_out: bool = True,
    include_tshs: bool = True,
    include_tsde: bool = True,
    include_hsx: bool = True,
    include_struct_out: bool = True,
    include_xv: bool = True,
    include_orb_indx: bool = True,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "RUN.fdf").write_text(f"SystemLabel {label}\n", encoding="utf-8")
    (path / "metadata.json").write_text('{"system_label": "%s"}\n' % label, encoding="utf-8")
    if include_run_out:
        (path / "RUN.out").write_text("Job completed\nSCF cycle converged\n", encoding="utf-8")
    files = {
        ".TSHS": include_tshs,
        ".TSDE": include_tsde,
        ".HSX": include_hsx,
        ".STRUCT_OUT": include_struct_out,
        ".XV": include_xv,
        ".ORB_INDX": include_orb_indx,
    }
    for suffix, enabled in files.items():
        if enabled:
            (path / f"{label}{suffix}").write_text(f"{suffix}\n", encoding="utf-8")


def write_dataset_provenance(
    dataset: Path,
    *,
    include_basis: bool = True,
    include_pseudo: bool = True,
    include_material: bool = True,
    include_fdf: bool = True,
    include_siesta_version: bool = True,
    include_siesta_command: bool = True,
    include_environment: bool = True,
    include_run_log: bool = True,
) -> None:
    payload = {}
    if include_material:
        payload["label"] = "graphene"
    if include_basis:
        payload["basis_file_sha256"] = {"C.ion.xml": "basis-hash"}
    if include_pseudo:
        payload["pseudopotential_sha256"] = {"C": "pseudo-hash"}
    if include_fdf:
        payload["fdf_sha256"] = "fdf-hash"
    if include_siesta_version:
        payload["siesta_version"] = "SIESTA test-version"
    if include_siesta_command:
        payload["siesta_command_line"] = "bash -lc 'siesta < RUN.fdf'"
    if include_environment:
        payload["environment"] = {"python_version": "3.11.0", "platform": "test-platform"}
    if include_run_log:
        (dataset / "RUN.out").write_text("Job completed\n", encoding="utf-8")
        payload["run_out_path"] = str(dataset / "RUN.out")
    (dataset / "material_provenance.json").write_text(
        json.dumps(payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )


class JointArtifactContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_valid_snapshot_with_all_artifacts(self) -> None:
        sample = self.root / "sample_0001"
        write_snapshot(sample)

        result = validate_snapshot(sample)

        self.assertEqual(result.contract_name, CONTRACT_NAME)
        self.assertTrue(result.valid)
        self.assertFalse(result.repair_required)
        self.assertEqual(result.system_label, "graphene")
        self.assertEqual(result.missing_required, [])
        self.assertIn("hsx", result.present_artifacts)

    def test_missing_hsx_is_invalid_and_repair_required(self) -> None:
        sample = self.root / "sample_0001"
        write_snapshot(sample, include_hsx=False)

        result = validate_snapshot(sample)

        self.assertFalse(result.valid)
        self.assertTrue(result.repair_required)
        self.assertIn("hsx", result.missing_required)

    def test_missing_struct_out_is_invalid_and_repair_required(self) -> None:
        sample = self.root / "sample_0001"
        write_snapshot(sample, include_struct_out=False)

        result = validate_snapshot(sample)

        self.assertFalse(result.valid)
        self.assertIn("struct_out", result.missing_required)

    def test_missing_orb_indx_is_invalid_and_repair_required(self) -> None:
        sample = self.root / "sample_0001"
        write_snapshot(sample, include_orb_indx=False)

        result = validate_snapshot(sample)

        self.assertFalse(result.valid)
        self.assertIn("orb_indx", result.missing_required)

    def test_missing_tshs_when_required_is_invalid(self) -> None:
        sample = self.root / "sample_0001"
        write_snapshot(sample, include_tshs=False)

        result = validate_snapshot(sample, require_tshs=True)

        self.assertFalse(result.valid)
        self.assertIn("tshs", result.missing_required)

    def test_missing_tsde_when_required_is_invalid(self) -> None:
        sample = self.root / "sample_0001"
        write_snapshot(sample, include_tsde=False)

        result = validate_snapshot(sample, require_tsde=True)

        self.assertFalse(result.valid)
        self.assertIn("tsde", result.missing_required)

    def test_ambiguous_system_label_fails_clearly(self) -> None:
        sample = self.root / "sample_0001"
        write_snapshot(sample, label="graphene")
        (sample / "other.HSX").write_text("other\n", encoding="utf-8")

        result = validate_snapshot(sample)

        self.assertFalse(result.valid)
        self.assertIsNone(result.system_label)
        self.assertTrue(any("ambiguous SystemLabel" in error for error in result.errors))

    def test_dataset_summary_counts_valid_and_invalid_snapshots(self) -> None:
        valid = self.root / "dataset" / "valid"
        invalid = self.root / "dataset" / "invalid"
        write_snapshot(valid)
        write_snapshot(invalid, include_hsx=False)

        result = validate_dataset(self.root / "dataset")

        self.assertFalse(result.valid)
        self.assertEqual(result.total_snapshots, 2)
        self.assertEqual(result.valid_snapshots, 1)
        self.assertEqual(result.invalid_snapshots, 1)
        self.assertEqual(result.repair_required_snapshots, 1)

    def test_old_graph2mat_only_snapshot_is_not_benchmark_ready(self) -> None:
        sample = self.root / "old_graph2mat_only"
        write_snapshot(sample, include_hsx=False, include_struct_out=False, include_orb_indx=False)

        result = validate_snapshot(sample)

        self.assertFalse(result.valid)
        self.assertTrue(result.repair_required)
        self.assertEqual(
            {"hsx", "struct_out", "orb_indx"},
            set(result.missing_required),
        )

    def test_strict_profile_requires_basis_provenance(self) -> None:
        dataset = self.root / "dataset"
        write_snapshot(dataset / "sample_0001")
        write_dataset_provenance(dataset, include_basis=False)

        result = validate_dataset(dataset, validation_profile=G2M_DEEPH_BENCHMARK_PROFILE)

        self.assertFalse(result.valid)
        self.assertIn("dataset-level basis provenance or basis file hashes are missing", result.errors)

    def test_strict_profile_requires_pseudopotential_provenance(self) -> None:
        dataset = self.root / "dataset"
        write_snapshot(dataset / "sample_0001")
        write_dataset_provenance(dataset, include_pseudo=False)

        result = validate_dataset(dataset, validation_profile=G2M_DEEPH_BENCHMARK_PROFILE)

        self.assertFalse(result.valid)
        self.assertIn("dataset-level pseudopotential provenance or hashes are missing", result.errors)

    def test_strict_profile_requires_material_identity(self) -> None:
        dataset = self.root / "dataset"
        write_snapshot(dataset / "sample_0001")
        write_dataset_provenance(dataset, include_material=False)

        result = validate_dataset(dataset, validation_profile=G2M_DEEPH_BENCHMARK_PROFILE)

        self.assertFalse(result.valid)
        self.assertIn("dataset-level material identity is missing", result.errors)

    def test_strict_profile_with_full_provenance_passes(self) -> None:
        dataset = self.root / "dataset"
        write_snapshot(dataset / "sample_0001")
        write_dataset_provenance(dataset)

        result = validate_dataset(dataset, validation_profile=G2M_DEEPH_BENCHMARK_PROFILE)

        self.assertTrue(result.valid)
        self.assertTrue(result.basis_present)
        self.assertTrue(result.pseudopotential_provenance_present)
        self.assertTrue(result.material_identity_present)
        self.assertTrue(result.siesta_input_provenance_present)
        self.assertTrue(result.siesta_version_provenance_present)
        self.assertTrue(result.siesta_command_line_provenance_present)
        self.assertTrue(result.siesta_environment_provenance_present)
        self.assertTrue(result.siesta_execution_log_present)

    def test_strict_profile_requires_siesta_version_provenance(self) -> None:
        dataset = self.root / "dataset"
        write_snapshot(dataset / "sample_0001")
        write_dataset_provenance(dataset, include_siesta_version=False)

        result = validate_dataset(dataset, validation_profile=G2M_DEEPH_BENCHMARK_PROFILE)

        self.assertFalse(result.valid)
        self.assertIn("dataset-level SIESTA version provenance is missing", result.errors)

    def test_strict_profile_requires_siesta_command_line_provenance(self) -> None:
        dataset = self.root / "dataset"
        write_snapshot(dataset / "sample_0001")
        write_dataset_provenance(dataset, include_siesta_command=False)

        result = validate_dataset(dataset, validation_profile=G2M_DEEPH_BENCHMARK_PROFILE)

        self.assertFalse(result.valid)
        self.assertIn("dataset-level SIESTA command-line provenance is missing", result.errors)

    def test_strict_profile_requires_environment_provenance(self) -> None:
        dataset = self.root / "dataset"
        write_snapshot(dataset / "sample_0001")
        write_dataset_provenance(dataset, include_environment=False)

        result = validate_dataset(dataset, validation_profile=G2M_DEEPH_BENCHMARK_PROFILE)

        self.assertFalse(result.valid)
        self.assertIn("dataset-level execution environment provenance is missing", result.errors)

    def test_non_strict_fixture_can_skip_dataset_provenance_explicitly(self) -> None:
        dataset = self.root / "dataset"
        write_snapshot(dataset / "sample_0001")

        result = validate_dataset(dataset, require_dataset_provenance=False)

        self.assertTrue(result.valid)
        self.assertFalse(result.basis_present)
        self.assertFalse(result.pseudopotential_provenance_present)
        self.assertFalse(result.material_identity_present)


if __name__ == "__main__":
    unittest.main()
