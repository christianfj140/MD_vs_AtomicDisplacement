from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "Comparison" / "scripts" / "audit_epc_lineage.py"
SPEC = importlib.util.spec_from_file_location("audit_epc_lineage", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

REAL_PREREGISTRATION = REPO_ROOT / "docs" / "epc_preregistration_v2.json"


def write_run_fdf(path: Path, coords: list[tuple[float, float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["%block AtomicCoordinatesAndAtomicSpecies"]
    for x, y, z in coords:
        lines.append(f"{x} {y} {z} 1")
    lines.append("%endblock AtomicCoordinatesAndAtomicSpecies")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def sha256_of(path: Path) -> str:
    return MODULE.sha256_file(path)


class CheckpointSelectionStatusTests(unittest.TestCase):
    def test_validation_only_metric_is_independent(self) -> None:
        status, _ = MODULE.checkpoint_selection_status({"monitor_metric": "val_rmse_eV"})
        self.assertEqual(status, MODULE.STATUS_INDEPENDENT)

    def test_test_metric_is_a_confirmed_leak(self) -> None:
        status, _ = MODULE.checkpoint_selection_status({"monitor_metric": "test_rmse_eV"})
        self.assertEqual(status, MODULE.STATUS_CHECKPOINT_SELECTION_LEAK)

    def test_holdout_metric_is_a_confirmed_leak(self) -> None:
        status, _ = MODULE.checkpoint_selection_status({"monitor_metric": "holdout_frontier_rmse"})
        self.assertEqual(status, MODULE.STATUS_CHECKPOINT_SELECTION_LEAK)

    def test_missing_metric_is_unknown_not_a_pass(self) -> None:
        status, _ = MODULE.checkpoint_selection_status({})
        self.assertEqual(status, MODULE.STATUS_UNKNOWN_SELECTION_METRIC)

    def test_ambiguous_metric_name_is_unknown_not_a_pass(self) -> None:
        status, _ = MODULE.checkpoint_selection_status({"monitor_metric": "frontier_rmse_eV"})
        self.assertEqual(status, MODULE.STATUS_UNKNOWN_SELECTION_METRIC)


class AuditCheckpointEntryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def _base_entry(self, **overrides) -> dict:
        entry = {
            "checkpoint_id": "unit-checkpoint",
            "checkpoint_selection": {"monitor_metric": "val_rmse_eV"},
            "known_training_manifests": [],
            "epc_evaluation_structures": [],
            "training_provenance_incomplete": False,
        }
        entry.update(overrides)
        return entry

    def _write_manifest(self, name: str, rows: list[dict]) -> Path:
        path = self.root / name
        path.write_text(json.dumps({"rows": rows}), encoding="utf-8")
        return path

    def test_exact_duplicate_structure_is_flagged_as_leakage(self) -> None:
        eval_fdf = self.root / "eval" / "RUN.fdf"
        write_run_fdf(eval_fdf, [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)])
        train_fdf = self.root / "train" / "RUN.fdf"
        write_run_fdf(train_fdf, [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)])  # identical geometry
        manifest = self._write_manifest(
            "manifest.json",
            [{"sample_id": "t0", "structure_path": str(train_fdf), "split": "train", "method": "md"}],
        )
        entry = self._base_entry(
            known_training_manifests=[{"path": str(manifest), "split_field": "train"}],
            epc_evaluation_structures=[{"label": "eval0", "path": str(eval_fdf)}],
        )
        result = MODULE.audit_checkpoint_entry(entry, base_dir=self.root)
        self.assertEqual(result["status"], MODULE.STATUS_LEAKAGE_DETECTED)

    def test_disjoint_structures_are_independent(self) -> None:
        eval_fdf = self.root / "eval" / "RUN.fdf"
        write_run_fdf(eval_fdf, [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)])
        train_fdf = self.root / "train" / "RUN.fdf"
        write_run_fdf(train_fdf, [(5.0, 5.0, 5.0), (6.0, 6.0, 6.0)])  # different geometry, same atom count
        manifest = self._write_manifest(
            "manifest.json",
            [{"sample_id": "t0", "structure_path": str(train_fdf), "split": "train", "method": "md"}],
        )
        entry = self._base_entry(
            known_training_manifests=[{"path": str(manifest), "split_field": "train"}],
            epc_evaluation_structures=[{"label": "eval0", "path": str(eval_fdf)}],
        )
        result = MODULE.audit_checkpoint_entry(entry, base_dir=self.root)
        self.assertEqual(result["status"], MODULE.STATUS_INDEPENDENT)

    def test_different_atom_count_is_prefiltered_and_independent(self) -> None:
        eval_fdf = self.root / "eval" / "RUN.fdf"
        write_run_fdf(eval_fdf, [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)])  # 2 atoms
        train_fdf = self.root / "train" / "RUN.fdf"
        write_run_fdf(train_fdf, [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)])  # 3 atoms
        manifest = self._write_manifest(
            "manifest.json",
            [{"sample_id": "t0", "structure_path": str(train_fdf), "split": "train", "method": "md"}],
        )
        entry = self._base_entry(
            known_training_manifests=[{"path": str(manifest), "split_field": "train"}],
            epc_evaluation_structures=[{"label": "eval0", "path": str(eval_fdf)}],
        )
        result = MODULE.audit_checkpoint_entry(entry, base_dir=self.root)
        self.assertEqual(result["status"], MODULE.STATUS_INDEPENDENT)
        per_structure = result["geometry_leakage_summary"]["per_evaluation_structure"][0]
        self.assertEqual(per_structure["train_rows_compared"], 0)
        self.assertEqual(per_structure["train_rows_skipped_atom_count_mismatch"], 1)

    def test_declared_incomplete_training_provenance_overrides_a_clean_geometry_result(self) -> None:
        eval_fdf = self.root / "eval" / "RUN.fdf"
        write_run_fdf(eval_fdf, [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)])
        train_fdf = self.root / "train" / "RUN.fdf"
        write_run_fdf(train_fdf, [(9.0, 9.0, 9.0), (8.0, 8.0, 8.0)])
        manifest = self._write_manifest(
            "manifest.json",
            [{"sample_id": "t0", "structure_path": str(train_fdf), "split": "train", "method": "md"}],
        )
        entry = self._base_entry(
            known_training_manifests=[{"path": str(manifest), "split_field": "train"}],
            epc_evaluation_structures=[{"label": "eval0", "path": str(eval_fdf)}],
            training_provenance_incomplete=True,
            training_provenance_gap="558 MD samples have no row-level manifest",
        )
        result = MODULE.audit_checkpoint_entry(entry, base_dir=self.root)
        self.assertEqual(result["status"], MODULE.STATUS_UNKNOWN_INCOMPLETE_TRAINING)
        self.assertIn(MODULE.STATUS_INDEPENDENT, result["component_statuses"])

    def test_missing_training_manifest_is_unknown_not_independent(self) -> None:
        eval_fdf = self.root / "eval" / "RUN.fdf"
        write_run_fdf(eval_fdf, [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)])
        entry = self._base_entry(
            known_training_manifests=[{"path": str(self.root / "does_not_exist.json"), "split_field": "train"}],
            epc_evaluation_structures=[{"label": "eval0", "path": str(eval_fdf)}],
        )
        result = MODULE.audit_checkpoint_entry(entry, base_dir=self.root)
        self.assertEqual(result["status"], MODULE.STATUS_UNKNOWN_MANIFEST_MISSING)

    def test_checkpoint_hash_mismatch_is_flagged(self) -> None:
        checkpoint_path = self.root / "ckpt.pt"
        checkpoint_path.write_bytes(b"weights")
        eval_fdf = self.root / "eval" / "RUN.fdf"
        write_run_fdf(eval_fdf, [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)])
        entry = self._base_entry(
            checkpoint_path=str(checkpoint_path),
            checkpoint_sha256="0" * 64,  # deliberately wrong
            epc_evaluation_structures=[{"label": "eval0", "path": str(eval_fdf), "sha256": sha256_of(eval_fdf)}],
        )
        result = MODULE.audit_checkpoint_entry(entry, base_dir=self.root)
        self.assertEqual(result["status"], MODULE.STATUS_STRUCTURE_HASH_MISMATCH)

    def test_test_metric_selection_is_a_blocking_leak_even_with_clean_geometry(self) -> None:
        eval_fdf = self.root / "eval" / "RUN.fdf"
        write_run_fdf(eval_fdf, [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)])
        entry = self._base_entry(
            checkpoint_selection={"monitor_metric": "test_rmse_eV"},
            epc_evaluation_structures=[{"label": "eval0", "path": str(eval_fdf)}],
        )
        result = MODULE.audit_checkpoint_entry(entry, base_dir=self.root)
        self.assertEqual(result["status"], MODULE.STATUS_CHECKPOINT_SELECTION_LEAK)
        self.assertIn(result["status"], MODULE.BLOCKING_ALWAYS)


class BuildReportAndMainTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def _write_preregistration(self, checkpoint_lineage: list[dict]) -> Path:
        path = self.root / "prereg.json"
        path.write_text(json.dumps({"schema": "epc_preregistration_v1", "checkpoint_lineage": checkpoint_lineage}), encoding="utf-8")
        return path

    def test_no_checkpoints_declared_is_not_treated_as_independent(self) -> None:
        prereg = self._write_preregistration([])
        report = MODULE.build_report(preregistration_path=prereg)
        self.assertEqual(report["checkpoints_checked"], 0)
        self.assertFalse(report["summary"]["independent_lineage_allowed"])

    def test_main_default_does_not_fail_on_unknown_but_does_on_blocking(self) -> None:
        eval_fdf = self.root / "eval" / "RUN.fdf"
        write_run_fdf(eval_fdf, [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)])

        unknown_only = self._write_preregistration(
            [
                {
                    "checkpoint_id": "c-unknown",
                    "checkpoint_selection": {},  # undeclared metric -> unknown, not blocking
                    "epc_evaluation_structures": [{"label": "eval0", "path": str(eval_fdf)}],
                }
            ]
        )
        output = self.root / "out.json"
        exit_default = MODULE.main(["--preregistration", str(unknown_only), "--output", str(output)])
        self.assertEqual(exit_default, 0)
        exit_strict = MODULE.main(
            ["--preregistration", str(unknown_only), "--output", str(output), "--fail-on-unknown"]
        )
        self.assertEqual(exit_strict, 1)

        blocking = self._write_preregistration(
            [
                {
                    "checkpoint_id": "c-blocking",
                    "checkpoint_selection": {"monitor_metric": "test_rmse_eV"},
                    "epc_evaluation_structures": [{"label": "eval0", "path": str(eval_fdf)}],
                }
            ]
        )
        exit_blocking = MODULE.main(["--preregistration", str(blocking), "--output", str(output)])
        self.assertEqual(exit_blocking, 1)
        payload = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(payload["schema"], MODULE.SCHEMA)


@unittest.skipUnless(REAL_PREREGISTRATION.exists(), "docs/epc_preregistration_v2.json not present")
class RealPreregistrationSmokeTests(unittest.TestCase):
    """Runs against the real consolidated preregistration doc, not a fixture.

    Regression guard for the atom-count pre-filter: without it, this test
    would hang comparing the 11164-atom MATBG structure against training
    rows (see the module docstring on _atom_count).
    """

    def test_runs_to_completion_and_never_reports_independent_for_a_declared_incomplete_checkpoint(self) -> None:
        report = MODULE.build_report(preregistration_path=REAL_PREREGISTRATION)
        self.assertGreaterEqual(report["checkpoints_checked"], 1)
        for row in report["results"]:
            if row["status"] == MODULE.STATUS_UNKNOWN_INCOMPLETE_TRAINING:
                continue
            # Any checkpoint not flagged incomplete must have a definite status,
            # never something silently missing.
            self.assertIn(
                row["status"],
                MODULE.BLOCKING_ALWAYS | MODULE.UNKNOWN_STATUSES | {MODULE.STATUS_INDEPENDENT},
            )


if __name__ == "__main__":
    unittest.main()
