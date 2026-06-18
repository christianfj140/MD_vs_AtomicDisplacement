from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "Comparison" / "scripts"
SCRIPT = SCRIPTS_DIR / "validate_derivative_workflow_artifacts.py"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from validate_derivative_workflow_artifacts import validate_derivative_workflow_artifacts  # noqa: E402


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_fdf(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("SystemLabel test\n", encoding="utf-8")


def write_sample(
    root: Path,
    sample_id: str,
    *,
    base_sample_id: str,
    sign: int,
    sign_label: str,
    delta_ang: float,
    include_metadata: bool = True,
) -> Path:
    sample_dir = root / "structures" / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    write_fdf(sample_dir / "RUN.fdf")
    if include_metadata:
        write_json(
            sample_dir / "metadata.json",
            {
                "sample_id": sample_id,
                "base_sample_id": base_sample_id,
                "atom_index_zero_based": 0,
                "axis": "x",
                "axis_index": 0,
                "delta_ang": delta_ang,
                "finite_difference_method": "central",
                "split": "test",
                "sign": sign,
                "sign_label": sign_label,
                "is_reference": sign == 0,
            },
        )
    return sample_dir


def write_stencil_manifest(root: Path, *, include_minus: bool = True, include_base: bool = True) -> None:
    samples = []
    stencils = []
    if include_base:
        samples.append(
            {
                "sample_id": "base_0",
                "base_sample_id": "base_0",
                "atom_index_zero_based": 0,
                "axis": "x",
                "delta_ang": 0.0,
                "finite_difference_method": "central",
                "split": "test",
                "sign": 0,
                "sign_label": "0",
            }
        )
    samples.append(
        {
            "sample_id": "plus_0",
            "base_sample_id": "base_0",
            "atom_index_zero_based": 0,
            "axis": "x",
            "delta_ang": 0.1,
            "finite_difference_method": "central",
            "split": "test",
            "sign": 1,
            "sign_label": "+",
        }
    )
    if include_minus:
        samples.append(
            {
                "sample_id": "minus_0",
                "base_sample_id": "base_0",
                "atom_index_zero_based": 0,
                "axis": "x",
                "delta_ang": 0.1,
                "finite_difference_method": "central",
                "split": "test",
                "sign": -1,
                "sign_label": "-",
            }
        )
    stencils.append(
        {
            "base_sample_id": "base_0",
            "plus_sample_id": "plus_0",
            "minus_sample_id": "minus_0" if include_minus else "",
            "atom_index_zero_based": 0,
            "axis": "x",
            "delta_ang": 0.1,
            "finite_difference_method": "central",
            "split_group_id": "dH_base_0_atom0000_x_delta0.1",
        }
    )
    write_json(
        root / "derivative_stencil_manifest.json",
        {
            "finite_difference_method": "central",
            "split": "test",
            "samples": samples,
            "stencils": stencils,
        },
    )


def write_reference_manifest(root: Path) -> None:
    reference_root = root / "siesta_hamiltonians"
    rows = []
    for sample_id in ("base_0", "plus_0", "minus_0"):
        sample_dir = reference_root / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        reference_matrix = sample_dir / "siesta.TSHS"
        reference_matrix.write_text("reference\n", encoding="utf-8")
        rows.append(
            {
                "sample_id": sample_id,
                "status": "ok",
                "structure_dir": str(root / "structures" / sample_id),
                "reference_dir": str(sample_dir),
                "reference_matrix": str(reference_matrix),
            }
        )
    write_json(
        reference_root / "derivative_siesta_reference_manifest.json",
        {
            "samples_total": len(rows),
            "samples_ok": len(rows),
            "samples_failed": 0,
            "rows": rows,
        },
    )


def write_prediction_manifest(root: Path, model: str, *, include_file: bool = True) -> None:
    prediction_root = root / "predicted_hamiltonians"
    rows = []
    for sample_id in ("base_0", "plus_0", "minus_0"):
        sample_dir = prediction_root / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        prediction_path = sample_dir / "ML_prediction.HSX"
        if include_file:
            prediction_path.write_text("prediction\n", encoding="utf-8")
        rows.append(
            {
                "sample_id": sample_id,
                "status": "predicted",
                "model": model,
                "structure_dir": str(root / "structures" / sample_id),
                "prediction_dir": str(sample_dir),
                "prediction_path": str(prediction_path),
                "checkpoint": str(root / "checkpoint.ckpt"),
                "model_dir": str(root / f"{model}_model"),
            }
        )
    write_json(
        prediction_root / f"derivative_{model}_prediction_manifest.json",
        {
            "samples_total": len(rows),
            "samples_ok": len(rows),
            "samples_failed": 0,
            "model": model,
            "rows": rows,
        },
    )


def write_metrics_manifest(root: Path, model: str, *, include_delta_stability: bool = True) -> None:
    metrics_root = root / "derivative_metrics" / model
    metrics_root.mkdir(parents=True, exist_ok=True)
    write_json(
        metrics_root / "manifest.json",
        {
            "scientific_status": "diagnostic_only",
            "finite_difference_method": "central",
            "reference_definition": "siesta_hamiltonian_finite_difference",
            "derivative_units": "eV/Ang",
            "stencils_total": 1,
            "stencils_ok": 1,
            "stencils_failed": 0,
        },
    )
    if include_delta_stability:
        write_json(metrics_root / "derivative_delta_stability.json", {"status": "available", "rows": []})


def build_valid_scope(root: Path, *, model: str = "graph2mat") -> Path:
    write_sample(root, "base_0", base_sample_id="base_0", sign=0, sign_label="0", delta_ang=0.0)
    write_sample(root, "plus_0", base_sample_id="base_0", sign=1, sign_label="+", delta_ang=0.1)
    write_sample(root, "minus_0", base_sample_id="base_0", sign=-1, sign_label="-", delta_ang=0.1)
    write_stencil_manifest(root)
    return root


class DerivativeWorkflowArtifactValidatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "derivative_result"
        self.root.mkdir()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def build_minimal_valid_tree(self) -> Path:
        write_sample(self.root, "base_0", base_sample_id="base_0", sign=0, sign_label="0", delta_ang=0.0)
        write_sample(self.root, "plus_0", base_sample_id="base_0", sign=1, sign_label="+", delta_ang=0.1)
        write_sample(self.root, "minus_0", base_sample_id="base_0", sign=-1, sign_label="-", delta_ang=0.1)
        write_stencil_manifest(self.root)
        return self.root

    def test_valid_minimal_artifact_tree_passes(self) -> None:
        summary = validate_derivative_workflow_artifacts(self.build_minimal_valid_tree())

        self.assertEqual(summary["status"], "ok")
        self.assertEqual(summary["errors"], [])
        self.assertIn("stencil_manifest:", " ".join(summary["checked"]))

    def test_parent_smoke_root_with_graph2mat_child_validates_child_scope(self) -> None:
        parent_root = self.root / "derivative_smoke"
        graph2mat_root = parent_root / "graph2mat_derivative_result"
        build_valid_scope(graph2mat_root)

        summary = validate_derivative_workflow_artifacts(parent_root)

        self.assertEqual(summary["status"], "ok")
        self.assertIn(str(graph2mat_root), summary["checked_roots"])
        self.assertIn(str(graph2mat_root / "derivative_stencil_manifest.json"), summary["checked_paths"])

    def test_parent_smoke_root_missing_child_metadata_fails(self) -> None:
        parent_root = self.root / "derivative_smoke"
        graph2mat_root = parent_root / "graph2mat_derivative_result"
        write_sample(graph2mat_root, "base_0", base_sample_id="base_0", sign=0, sign_label="0", delta_ang=0.0)
        write_sample(graph2mat_root, "plus_0", base_sample_id="base_0", sign=1, sign_label="+", delta_ang=0.1, include_metadata=False)
        write_sample(graph2mat_root, "minus_0", base_sample_id="base_0", sign=-1, sign_label="-", delta_ang=0.1)
        write_stencil_manifest(graph2mat_root)

        summary = validate_derivative_workflow_artifacts(parent_root)

        self.assertEqual(summary["status"], "failed")
        self.assertTrue(any("Missing metadata.json" in error for error in summary["errors"]))

    def test_parent_smoke_root_with_graph2mat_and_deeph_children_validates_both(self) -> None:
        parent_root = self.root / "derivative_smoke"
        graph2mat_root = parent_root / "graph2mat_derivative_result"
        deeph_root = parent_root / "deeph_derivative_result"
        build_valid_scope(graph2mat_root)
        build_valid_scope(deeph_root)

        summary = validate_derivative_workflow_artifacts(parent_root)

        self.assertEqual(summary["status"], "ok")
        self.assertIn(str(graph2mat_root), summary["checked_roots"])
        self.assertIn(str(deeph_root), summary["checked_roots"])

    def test_model_graph2mat_ignores_deeph_only_prediction_checks(self) -> None:
        parent_root = self.root / "derivative_smoke"
        graph2mat_root = parent_root / "graph2mat_derivative_result"
        deeph_root = parent_root / "deeph_derivative_result"
        build_valid_scope(graph2mat_root)
        build_valid_scope(deeph_root)
        write_prediction_manifest(deeph_root, "deeph", include_file=False)

        summary = validate_derivative_workflow_artifacts(parent_root, model="graph2mat")

        self.assertEqual(summary["status"], "ok")
        self.assertIn(str(graph2mat_root), summary["checked_roots"])
        self.assertNotIn(str(deeph_root), summary["checked_roots"])

    def test_missing_metadata_fails(self) -> None:
        write_sample(self.root, "base_0", base_sample_id="base_0", sign=0, sign_label="0", delta_ang=0.0)
        write_sample(self.root, "plus_0", base_sample_id="base_0", sign=1, sign_label="+", delta_ang=0.1, include_metadata=False)
        write_sample(self.root, "minus_0", base_sample_id="base_0", sign=-1, sign_label="-", delta_ang=0.1)
        write_stencil_manifest(self.root)

        summary = validate_derivative_workflow_artifacts(self.root)

        self.assertEqual(summary["status"], "failed")
        self.assertTrue(any("Missing metadata.json" in error for error in summary["errors"]))

    def test_central_family_missing_plus_minus_base_fails(self) -> None:
        write_sample(self.root, "base_0", base_sample_id="base_0", sign=0, sign_label="0", delta_ang=0.0)
        write_sample(self.root, "plus_0", base_sample_id="base_0", sign=1, sign_label="+", delta_ang=0.1)
        write_stencil_manifest(self.root, include_minus=False)

        summary = validate_derivative_workflow_artifacts(self.root)

        self.assertEqual(summary["status"], "failed")
        self.assertTrue(any("missing R0/R+/R- members" in error for error in summary["errors"]))

    def test_missing_prediction_file_fails_only_when_prediction_manifest_exists(self) -> None:
        self.build_minimal_valid_tree()
        write_prediction_manifest(self.root, "graph2mat", include_file=False)

        summary = validate_derivative_workflow_artifacts(self.root, model="graph2mat")

        self.assertEqual(summary["status"], "failed")
        self.assertTrue(any("Missing prediction file" in error for error in summary["errors"]))

    def test_missing_delta_stability_fails_only_when_metrics_manifest_exists(self) -> None:
        self.build_minimal_valid_tree()
        write_metrics_manifest(self.root, "graph2mat", include_delta_stability=False)

        summary = validate_derivative_workflow_artifacts(self.root, model="graph2mat")

        self.assertEqual(summary["status"], "failed")
        self.assertTrue(any("Missing derivative_delta_stability.json" in error for error in summary["errors"]))

    def test_json_output_is_written(self) -> None:
        output_json = self.root / "summary.json"

        completed = subprocess.run(
            [sys.executable, str(SCRIPT), str(self.build_minimal_valid_tree()), "--output-json", str(output_json)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, msg=f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}")
        self.assertTrue(output_json.exists())
        written = json.loads(output_json.read_text(encoding="utf-8"))
        self.assertEqual(written["status"], "ok")
        self.assertEqual(written["root"], str(self.root))
        self.assertIn("checked_roots", written)
        self.assertIn("checked_paths", written)

    def test_cli_json_output_includes_checked_subroot_paths(self) -> None:
        parent_root = self.root / "derivative_smoke"
        graph2mat_root = parent_root / "graph2mat_derivative_result"
        build_valid_scope(graph2mat_root)
        output_json = parent_root / "summary.json"

        completed = subprocess.run(
            [sys.executable, str(SCRIPT), str(parent_root), "--output-json", str(output_json)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, msg=f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}")
        written = json.loads(output_json.read_text(encoding="utf-8"))
        self.assertIn(str(graph2mat_root), written["checked_roots"])
        self.assertIn(str(graph2mat_root / "derivative_stencil_manifest.json"), written["checked_paths"])


if __name__ == "__main__":
    unittest.main()
