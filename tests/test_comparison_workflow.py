from __future__ import annotations

import csv
import contextlib
import importlib.util
import json
import shutil
import subprocess
import sys
import unittest
import uuid
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_TMP_ROOT = REPO_ROOT / ".test_tmp"


@contextlib.contextmanager
def workspace_tempdir():
    TEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)
    path = TEST_TMP_ROOT / f"case_{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield str(path)
    finally:
        shutil.rmtree(path, ignore_errors=True)


def run_script(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=cwd or REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def minimal_run_fdf(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "SystemLabel water",
                "%block LatticeVectors",
                "10.0 0.0 0.0",
                "0.0 10.0 0.0",
                "0.0 0.0 10.0",
                "%endblock LatticeVectors",
                "%block AtomicCoordinatesAndAtomicSpecies",
                "0.0 0.0 0.0 1",
                "0.0 0.7 0.0 2",
                "0.0 -0.7 0.0 2",
                "%endblock AtomicCoordinatesAndAtomicSpecies",
                "",
            ]
        ),
        encoding="utf-8",
    )


def make_sample(root: Path, name: str, *, hamiltonian: bool = True, converged: bool = True) -> Path:
    sample_dir = root / name
    sample_dir.mkdir(parents=True)
    minimal_run_fdf(sample_dir / "RUN.fdf")
    if hamiltonian:
        (sample_dir / "siesta.TSHS").write_bytes(b"fake")
    run_out = "Job completed\n"
    if converged:
        run_out += "SCF cycle converged\n"
    sample_dir.joinpath("RUN.out").write_text(run_out, encoding="utf-8")
    return sample_dir


class ComparisonWorkflowTests(unittest.TestCase):
    def test_validate_sample_bundle_accepts_complete_sample(self) -> None:
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            make_sample(root / "samples", "001")
            output = root / "validation"
            result = run_script(
                "Comparison/scripts/validate_sample_bundle.py",
                "--samples-dir",
                str(root / "samples"),
                "--method",
                "md",
                "--output-dir",
                str(output),
                "--min-valid",
                "1",
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            summary = json.loads((output / "validation_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["valid_samples"], 1)
            self.assertEqual(summary["invalid_samples"], 0)

    def test_validate_sample_bundle_rejects_missing_hamiltonian_and_failed_scf(self) -> None:
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            make_sample(root / "samples", "missing_h", hamiltonian=False)
            make_sample(root / "samples", "failed_scf", converged=False)
            output = root / "validation"
            result = run_script(
                "Comparison/scripts/validate_sample_bundle.py",
                "--samples-dir",
                str(root / "samples"),
                "--method",
                "md",
                "--output-dir",
                str(output),
                "--min-valid",
                "2",
            )
            self.assertEqual(result.returncode, 2)
            with (output / "invalid_samples.csv").open(encoding="utf-8") as handle:
                invalid = list(csv.DictReader(handle))
            reasons = " ".join(row["invalid_reasons"] for row in invalid)
            self.assertIn("missing_hamiltonian", reasons)
            self.assertIn("scf_not_converged", reasons)

    def test_build_common_tests_refuses_train_test_overlap(self) -> None:
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            md_sample = make_sample(root / "md", "001")
            atom_sample = make_sample(root / "atom", "001")
            md_test = root / "md_test.csv"
            atom_test = root / "atom_test.csv"
            train = root / "train.csv"
            write_csv(
                md_test,
                [
                    {
                        "sample_id": "md_001",
                        "method": "md",
                        "structure_path": str(md_sample / "RUN.fdf"),
                        "hamiltonian_path": str(md_sample / "siesta.TSHS"),
                        "status": "valid",
                    }
                ],
            )
            write_csv(
                atom_test,
                [
                    {
                        "sample_id": "atomdisp_001",
                        "method": "atom_displacement",
                        "structure_path": str(atom_sample / "RUN.fdf"),
                        "hamiltonian_path": str(atom_sample / "siesta.TSHS"),
                        "status": "valid",
                    }
                ],
            )
            write_csv(
                train,
                [
                    {
                        "sample_id": "md_001",
                        "method": "md",
                        "structure_path": str(md_sample / "RUN.fdf"),
                        "hamiltonian_path": str(md_sample / "siesta.TSHS"),
                        "status": "valid",
                    }
                ],
            )
            result = run_script(
                "Comparison/scripts/build_common_tests.py",
                "--md-test-manifest",
                str(md_test),
                "--atomdisp-test-manifest",
                str(atom_test),
                "--train-manifest",
                str(train),
                "--output-dir",
                str(root / "common"),
            )
            self.assertEqual(result.returncode, 2)
            summary = json.loads((root / "common" / "common_test_summary.json").read_text(encoding="utf-8"))
            self.assertFalse(summary["ok"])

    def test_cross_aggregation_contains_required_cells(self) -> None:
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            cross_root = root / "cross"
            for method in ("md", "atom_displacement"):
                for test_set in ("test_md", "test_atomdisp", "test_mixed"):
                    result_dir = cross_root / f"{method}__on__{test_set}"
                    (result_dir / "metrics").mkdir(parents=True)
                    write_csv(
                        result_dir / "metrics" / "sparse_metrics.csv",
                        [{"sample": "sample_1", "relative_frobenius_union": "1.0"}],
                    )
                    (result_dir / "cross_evaluation_manifest.json").write_text(
                        json.dumps(
                            {
                                "train_method": method,
                                "test_set": test_set,
                                "dataset_size": 3,
                                "seed": 42,
                                "model_checkpoint": "model.ckpt",
                            }
                        ),
                        encoding="utf-8",
                    )
            result = run_script(
                "Comparison/scripts/aggregate_cross_metrics.py",
                "--experiment-id",
                "exp_test",
                "--cross-root",
                str(cross_root),
                "--output-dir",
                str(root / "summary"),
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            with (root / "summary" / "cross_evaluation_metrics.csv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            cells = {(row["train_method"], row["test_set"]) for row in rows}
            self.assertEqual(
                cells,
                {
                    ("md", "test_md"),
                    ("md", "test_atomdisp"),
                    ("md", "test_mixed"),
                    ("atom_displacement", "test_md"),
                    ("atom_displacement", "test_atomdisp"),
                    ("atom_displacement", "test_mixed"),
                },
            )

    def test_winner_analysis_does_not_declare_winner_on_different_test_sets(self) -> None:
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            metrics = root / "cross_evaluation_metrics.csv"
            write_csv(
                metrics,
                [
                    {
                        "experiment_id": "exp_test",
                        "train_method": "md",
                        "test_set": "test_md",
                        "dataset_size": "3",
                        "seed": "42",
                        "sample_id": "md_1",
                        "global_rmse_eV": "1.0",
                    },
                    {
                        "experiment_id": "exp_test",
                        "train_method": "atom_displacement",
                        "test_set": "test_atomdisp",
                        "dataset_size": "3",
                        "seed": "42",
                        "sample_id": "atom_1",
                        "global_rmse_eV": "0.1",
                    },
                ],
            )
            result = run_script(
                "Comparison/scripts/analyze_winners.py",
                "--metrics-csv",
                str(metrics),
                "--output-dir",
                str(root / "summary"),
                "--primary-metric",
                "global_rmse_eV",
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            recommendation = json.loads((root / "summary" / "recommendation.json").read_text(encoding="utf-8"))
            self.assertNotEqual(recommendation["status"], "atom_displacement_conservative_win")

    def test_md_rewrite_makes_effective_geometries_differ_when_xv_differs(self) -> None:
        sys.path.insert(0, str(REPO_ROOT / "MD" / "scripts"))
        spec = importlib.util.spec_from_file_location(
            "generate_md_dataset",
            REPO_ROOT / "MD" / "scripts" / "generate_md_dataset.py",
        )
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with workspace_tempdir() as tmp:
            root = Path(tmp)
            signatures = []
            for index, z in enumerate((0.0, 0.5), start=1):
                sample_dir = root / str(index)
                sample_dir.mkdir()
                minimal_run_fdf(sample_dir / "RUN.fdf")
                sample_dir.joinpath("siesta.XV").write_text(
                    "\n".join(
                        [
                            "18.89726125 0.0 0.0",
                            "0.0 18.89726125 0.0",
                            "0.0 0.0 18.89726125",
                            "3",
                            f"1 8 0.0 0.0 {z} 0.0 0.0 0.0",
                            "2 1 0.0 1.0 0.0 0.0 0.0 0.0",
                            "2 1 0.0 -1.0 0.0 0.0 0.0 0.0",
                            "",
                        ]
                    ),
                    encoding="utf-8",
                )
                module.rewrite_run_fdf_from_xv(sample_dir / "RUN.fdf", sample_dir / "siesta.XV")
                signatures.append(module.effective_fdf_geometry_signature(sample_dir / "RUN.fdf"))
            self.assertNotEqual(signatures[0], signatures[1])


if __name__ == "__main__":
    unittest.main()
