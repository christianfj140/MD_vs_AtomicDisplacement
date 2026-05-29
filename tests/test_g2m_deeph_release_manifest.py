from __future__ import annotations

import hashlib
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "Comparison" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from g2m_deeph_release_manifest import build_release_manifest, main  # noqa: E402


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class ReleaseManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.dataset_root = self.root / "dataset"
        self.run_root = self.root / "run"
        self.workflow_root = self.root / "workflow"
        self.dataset_root.mkdir(parents=True)
        self.run_root.mkdir(parents=True)
        self.workflow_root.mkdir(parents=True)
        self.sample_hsx = self.dataset_root / "MD_steps" / "0" / "graphene.HSX"
        self._write_dataset()
        self._write_run()
        self._write_workflow()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _snapshot_artifacts(self, sample_dir: Path) -> dict[str, str]:
        files = {
            "run_fdf": sample_dir / "RUN.fdf",
            "run_output": sample_dir / "RUN.out",
            "metadata": sample_dir / "metadata.json",
            "reference_hsx": sample_dir / "graphene.HSX",
            "reference_tshs": sample_dir / "graphene.TSHS",
            "reference_tsde": sample_dir / "graphene.TSDE",
            "struct_out": sample_dir / "graphene.STRUCT_OUT",
            "xv": sample_dir / "graphene.XV",
            "orb_indx": sample_dir / "graphene.ORB_INDX",
        }
        contents = {
            "RUN.fdf": "SystemLabel graphene\nSave.HS T\n",
            "RUN.out": "Job completed\n",
            "metadata.json": '{"sample_id": "md_0"}\n',
            "graphene.HSX": "hsx\n",
            "graphene.TSHS": "tshs\n",
            "graphene.TSDE": "tsde\n",
            "graphene.STRUCT_OUT": "struct\n",
            "graphene.XV": "xv\n",
            "graphene.ORB_INDX": "orb\n",
        }
        for name, text in contents.items():
            write_text(sample_dir / name, text)
        return {key: str(path) for key, path in files.items()}

    def _write_dataset(self) -> None:
        write_json(self.dataset_root / "material_provenance.json", {"label": "graphene"})
        write_json(self.dataset_root / "artifact_validation.json", {"valid": True, "snapshots": []})
        rows = []
        for index, split in enumerate(("train", "validation", "test")):
            sample_dir = self.dataset_root / "MD_steps" / str(index)
            artifacts = self._snapshot_artifacts(sample_dir)
            rows.append(
                {
                    "sample_id": f"md_{index}",
                    "split": split,
                    "artifact_paths": artifacts,
                }
            )
        write_json(
            self.dataset_root / "frozen_split_manifest.json",
            {
                "schema": "joint_graph2mat_deeph_frozen_split_manifest_v1",
                "artifact_contract_version": "joint_graph2mat_deeph_v1",
                "split_hash": "split123",
                "split_counts": {"train": 1, "validation": 1, "test": 1},
                "valid": True,
                "rows": rows,
            },
        )
        write_json(
            self.dataset_root / "benchmark_dataset_manifest.json",
            {
                "schema": "joint_graph2mat_deeph_benchmark_manifest_v1",
                "benchmark_dataset_id": "unit_dataset",
                "artifact_contract_version": "joint_graph2mat_deeph_v1",
                "benchmark_ready": True,
                "frozen_split_manifest": {"split_hash": "split123"},
            },
        )

    def _write_run(self) -> None:
        write_json(self.run_root / "sweep" / "training_sweep_manifest.json", {"schema": "training"})
        write_json(self.run_root / "telemetry" / "graph2mat.json", {"gpu_hours_total": 1.25})
        write_json(
            self.run_root / "deeph" / "adapter_manifest.json",
            {"adapter_equivalence_status": "raw_global_equivalence_proven"},
        )
        write_json(
            self.run_root / "deeph" / "raw_global_equivalence_evidence.json",
            {"equivalence_status": "proven"},
        )
        write_text(self.run_root / "graph2mat" / "ML_prediction.HSX", "prediction\n")
        write_text(self.run_root / "deeph" / "hamiltonians_pred.h5", "deeph prediction\n")
        write_text(self.run_root / "deeph" / "hamiltonians.h5", "deeph reference\n")
        write_text(self.run_root / "deeph" / "overlaps.h5", "deeph overlap\n")
        write_text(self.run_root / "deeph" / "orbital_types.dat", "orbital types\n")

    def _write_workflow(self) -> None:
        write_json(self.workflow_root / "protocol" / "validated_protocol.json", {"protocol_id": "unit"})
        write_json(self.workflow_root / "search" / "search_plan.json", {"runs": []})
        write_json(self.workflow_root / "selection" / "selected_configs.json", {"selected": []})
        write_json(self.workflow_root / "selection" / "robust_rerun_plan.json", {"runs": []})
        write_json(self.workflow_root / "final_test" / "final_statistics.json", {"winner_decision": {}})
        write_json(self.workflow_root / "report" / "report_summary.json", {"status": "diagnostic"})
        write_json(self.workflow_root / "evidence" / "evidence_bundle_manifest.json", {"files": []})

    def test_hashes_and_sizes_are_recorded(self) -> None:
        manifest = build_release_manifest(
            dataset_root=self.dataset_root,
            run_root=self.run_root,
            workflow_root=self.workflow_root,
            strict=True,
        )

        self.assertEqual(manifest["status"], "complete")
        rows = {
            (row["role"], row["relative_path"]): row
            for row in manifest["files"]
        }
        artifact_row = rows[("artifact_validation", "artifact_validation.json")]
        text = (self.dataset_root / "artifact_validation.json").read_text(encoding="utf-8")
        self.assertEqual(artifact_row["size_bytes"], len(text.encode("utf-8")))
        self.assertEqual(artifact_row["sha256"], sha256_text(text))

    def test_strict_missing_required_file_is_invalid(self) -> None:
        self.sample_hsx.unlink()

        manifest = build_release_manifest(
            dataset_root=self.dataset_root,
            run_root=self.run_root,
            workflow_root=self.workflow_root,
            strict=True,
        )

        self.assertEqual(manifest["status"], "invalid")
        self.assertTrue(any("siesta_reference_hsx" in item for item in manifest["missing_required"]))

    def test_non_strict_missing_required_file_is_partial(self) -> None:
        self.sample_hsx.unlink()

        manifest = build_release_manifest(
            dataset_root=self.dataset_root,
            run_root=self.run_root,
            workflow_root=self.workflow_root,
            strict=False,
        )

        self.assertEqual(manifest["status"], "partial")
        self.assertTrue(manifest["missing_required"])

    def test_ml_prediction_as_reference_is_flagged(self) -> None:
        frozen = json.loads((self.dataset_root / "frozen_split_manifest.json").read_text(encoding="utf-8"))
        frozen["rows"][0]["artifact_paths"]["reference_hsx"] = str(self.run_root / "graph2mat" / "ML_prediction.HSX")
        write_json(self.dataset_root / "frozen_split_manifest.json", frozen)

        manifest = build_release_manifest(
            dataset_root=self.dataset_root,
            run_root=self.run_root,
            workflow_root=self.workflow_root,
            strict=False,
        )

        self.assertEqual(manifest["status"], "invalid")
        self.assertTrue(manifest["forbidden_reference_findings"])

    def test_paths_are_relative_when_under_allowed_roots(self) -> None:
        manifest = build_release_manifest(
            dataset_root=self.dataset_root,
            run_root=self.run_root,
            workflow_root=self.workflow_root,
            strict=True,
        )

        relative_paths = [row["relative_path"] for row in manifest["files"]]
        self.assertIn("artifact_validation.json", relative_paths)
        self.assertFalse(any(path.startswith("/") for path in relative_paths))

    def test_symlink_outside_allowed_roots_is_not_hashed(self) -> None:
        outside = self.root / "outside_artifacts" / "graphene.HSX"
        write_text(outside, "outside\n")
        self.sample_hsx.unlink()
        self.sample_hsx.symlink_to(outside)

        manifest = build_release_manifest(
            dataset_root=self.dataset_root,
            run_root=self.run_root,
            workflow_root=self.workflow_root,
            strict=True,
        )

        hsx_rows = [row for row in manifest["files"] if row["role"] == "siesta_reference_hsx"]
        self.assertTrue(any(row["symlink_outside_allowed_roots"] for row in hsx_rows))
        self.assertTrue(any(row["sha256"] is None for row in hsx_rows))
        self.assertEqual(manifest["status"], "invalid")

    def test_cli_writes_manifest(self) -> None:
        output = self.root / "release_manifest.json"

        with redirect_stdout(io.StringIO()):
            exit_code = main(
                [
                    "--dataset-root",
                    str(self.dataset_root),
                    "--run-root",
                    str(self.run_root),
                    "--workflow-root",
                    str(self.workflow_root),
                    "--output",
                    str(output),
                    "--strict",
                ]
            )

        self.assertEqual(exit_code, 0)
        manifest = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema"], "graph2mat_deeph_artifact_release_manifest_v1")
        self.assertEqual(manifest["status"], "complete")


if __name__ == "__main__":
    unittest.main()
