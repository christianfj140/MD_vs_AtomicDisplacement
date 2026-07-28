from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "Comparison" / "scripts"
SHARED_DIR = REPO_ROOT / "shared"
for directory in (SCRIPTS_DIR, SHARED_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from build_hamiltonian_derivative_stencils import build_derivative_stencils  # noqa: E402
from hamiltonian_derivative_stencil import discover_derivative_stencils  # noqa: E402
from run_hamiltonian_derivative_siesta_references import (  # noqa: E402
    DerivativeSiestaReferenceError,
    build_argument_parser,
    geometry_mismatch_message,
    run_derivative_siesta_references,
)
from reference_provenance import build_positive_reference_provenance  # noqa: E402


def synthetic_base_fdf(*, include_ghost: bool = False) -> str:
    species_count = 2 if include_ghost else 1
    species_block = [
        "%block ChemicalSpeciesLabel",
        " 1 6 C",
    ]
    if include_ghost:
        species_block.append(" 2 -1 Ghost-H")
    species_block.append("%endblock ChemicalSpeciesLabel")
    return "\n".join(
        [
            "SystemName synthetic base",
            "SystemLabel shared_label",
            f"NumberOfSpecies {species_count}",
            "NumberOfAtoms 2",
            *species_block,
            "LatticeConstant 1.0 Ang",
            "%block LatticeVectors",
            " 8.0 0.0 0.0",
            " 0.0 8.0 0.0",
            " 0.0 0.0 8.0",
            "%endblock LatticeVectors",
            "AtomicCoordinatesFormat Ang",
            "%block AtomicCoordinatesAndAtomicSpecies",
            " 0.0 0.0 0.0 1",
            " 1.0 0.0 0.0 1",
            "%endblock AtomicCoordinatesAndAtomicSpecies",
            "",
        ]
    )


class DerivativeSiestaReferenceStageTests(unittest.TestCase):
    def setUp(self) -> None:
        # These tests stage fake (non-sisl-readable) matrices; the geometry
        # guard fails closed on unreadable files, so neutralize it here. The
        # guard itself is unit-tested in ReferenceGeometryGuardTests.
        guard_patcher = mock.patch(
            "run_hamiltonian_derivative_siesta_references.reference_output_geometry_error",
            return_value="",
        )
        guard_patcher.start()
        self.addCleanup(guard_patcher.stop)
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.dataset_root = self.root / "source_dataset"
        self.sample_dir = self.dataset_root / "splits" / "test" / "base_0"
        self.sample_dir.mkdir(parents=True)
        (self.sample_dir / "RUN.fdf").write_text(synthetic_base_fdf(), encoding="utf-8")
        (self.sample_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "sample_id": "base_0",
                    "material_label": "synthetic",
                    "material_compatibility_hash": "material-hash",
                    "orbital_ordering_hash": "orbital-hash",
                    "basis_hash": "basis-hash",
                    "pseudopotential_hash": "pseudo-hash",
                }
            ),
            encoding="utf-8",
        )
        (self.dataset_root / "frozen_split_manifest.json").write_text(
            json.dumps({"rows": [{"sample_id": "base_0", "split": "test", "sample_dir": str(self.sample_dir)}]}),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def build_stencil(self) -> tuple[Path, dict]:
        stencil_root = self.root / "stencils"
        manifest = build_derivative_stencils(
            source_dataset_root=self.dataset_root,
            output_stencil_root=stencil_root,
            split="test",
            method="central",
            delta_ang_values=[0.1],
            atom_indices_zero_based=[1],
            axes=["y"],
            include_base=True,
        )
        return stencil_root, manifest

    def write_pseudo(self, label: str, *, suffix: str = ".psf") -> Path:
        path = self.dataset_root / f"{label}{suffix}"
        path.write_text(f"pseudo for {label}\n", encoding="utf-8")
        return path

    def selected_sample_ids(self, manifest: dict, limit: int) -> list[str]:
        return sorted(record["sample_id"] for record in manifest["samples"])[:limit]

    def write_proven_reference(self, reference_dir: Path, text: bytes = b"existing") -> None:
        reference_dir.mkdir(parents=True, exist_ok=True)
        files = {
            "siesta.TSHS": text,
            "RUN.fdf": (
                b"SystemLabel siesta\nNumberOfAtoms 1\nNumberOfSpecies 1\n"
                b"%block ChemicalSpeciesLabel\n1 6 C\n%endblock ChemicalSpeciesLabel\n"
                b"%block LatticeVectors\n8 0 0\n0 8 0\n0 0 8\n%endblock LatticeVectors\n"
                b"%block AtomicCoordinatesAndAtomicSpecies\n0 0 0 1\n"
                b"%endblock AtomicCoordinatesAndAtomicSpecies\n"
            ),
            "RUN.out": b"iscf     Eharris\nSCF cycle converged\nJob completed\n",
            "siesta.ORB_INDX": b"orb\n",
        }
        for name, content in files.items():
            (reference_dir / name).write_bytes(content)
        (reference_dir / "siesta_reference_provenance.json").write_text(
            json.dumps(
                build_positive_reference_provenance(
                    reference_dir,
                    reference_dir / "siesta.TSHS",
                    frozen_sample_id=reference_dir.name,
                    split="test",
                    frozen_split_hash="fixture-split-hash",
                    basis_hashes={"C.ion.xml": "basis-hash"},
                    pseudopotential_hashes={"C": "pseudo-hash"},
                    siesta_version="SIESTA 5.4.2-test",
                    siesta_command="siesta < RUN.fdf",
                )
            ),
            encoding="utf-8",
        )

    def test_argument_parser_accepts_workers(self) -> None:
        args = build_argument_parser().parse_args(["--stencil-root", str(self.root / "stencils"), "--workers", "2"])

        self.assertEqual(args.workers, 2)

    def test_skip_if_exists_avoids_rerunning_existing_reference(self) -> None:
        stencil_root, manifest = self.build_stencil()
        self.write_pseudo("C")
        sample_ids = self.selected_sample_ids(manifest, 2)
        reference_dir = stencil_root / "siesta_hamiltonians" / sample_ids[0]
        self.write_proven_reference(reference_dir)
        calls: list[str] = []

        def fake_run(command, **kwargs):
            cwd = Path(kwargs["cwd"])
            calls.append(cwd.name)
            (cwd / "siesta.TSHS").write_bytes(b"new")
            return mock.Mock(returncode=0)

        with mock.patch("run_hamiltonian_derivative_siesta_references.subprocess.run", side_effect=fake_run):
            result = run_derivative_siesta_references(
                stencil_root=stencil_root,
                workers=2,
                max_samples=2,
                skip_if_exists=True,
            )

        self.assertEqual(result["samples_ok"], 2)
        self.assertEqual([row["sample_id"] for row in result["rows"]], sample_ids)
        self.assertEqual(result["rows"][0]["status"], "skipped_existing")
        self.assertEqual(result["rows"][1]["status"], "ok")
        self.assertEqual(calls, [sample_ids[1]])
        self.assertEqual((reference_dir / "siesta.TSHS").read_bytes(), b"existing")

    def test_workers_one_preserves_sequential_behavior(self) -> None:
        stencil_root, manifest = self.build_stencil()
        self.write_pseudo("C")
        sample_ids = self.selected_sample_ids(manifest, 2)
        active = 0
        max_active = 0
        lock = threading.Lock()

        def fake_run(command, **kwargs):
            nonlocal active, max_active
            cwd = Path(kwargs["cwd"])
            with lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.05)
            (cwd / "siesta.TSHS").write_bytes(b"reference")
            with lock:
                active -= 1
            return mock.Mock(returncode=0)

        with mock.patch("run_hamiltonian_derivative_siesta_references.subprocess.run", side_effect=fake_run):
            result = run_derivative_siesta_references(
                stencil_root=stencil_root,
                workers=1,
                max_samples=2,
            )

        self.assertEqual(max_active, 1)
        self.assertEqual([row["sample_id"] for row in result["rows"]], sample_ids)
        self.assertEqual(result["workers"], 1)
        self.assertFalse(result["parallel_execution_enabled"])

    def test_workers_two_overlap_jobs_and_preserve_row_order(self) -> None:
        stencil_root, manifest = self.build_stencil()
        self.write_pseudo("C")
        sample_ids = self.selected_sample_ids(manifest, 2)
        active = 0
        max_active = 0
        lock = threading.Lock()

        def fake_run(command, **kwargs):
            nonlocal active, max_active
            cwd = Path(kwargs["cwd"])
            with lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.2 if cwd.name == sample_ids[0] else 0.05)
            (cwd / "siesta.TSHS").write_bytes(f"reference {cwd.name}".encode("utf-8"))
            with lock:
                active -= 1
            return mock.Mock(returncode=0)

        with mock.patch("run_hamiltonian_derivative_siesta_references.subprocess.run", side_effect=fake_run):
            result = run_derivative_siesta_references(
                stencil_root=stencil_root,
                workers=2,
                max_samples=2,
            )

        self.assertEqual(max_active, 2)
        self.assertEqual([row["sample_id"] for row in result["rows"]], sample_ids)
        self.assertEqual(result["workers"], 2)
        self.assertTrue(result["parallel_execution_enabled"])

    def test_missing_structure_reports_actionable_error(self) -> None:
        stencil_root = self.root / "broken_stencil"
        bad = stencil_root / "structures" / "bad_sample"
        bad.mkdir(parents=True)
        (bad / "metadata.json").write_text(json.dumps({"sample_id": "bad_sample"}), encoding="utf-8")

        result = run_derivative_siesta_references(
            stencil_root=stencil_root,
            diagnostic_only=True,
        )

        self.assertEqual(result["samples_failed"], 1)
        self.assertEqual(result["rows"][0]["status"], "error")
        self.assertEqual(result["rows"][0]["error"], "missing_structure_run_fdf")

    def test_failed_siesta_run_is_recorded_and_fails_closed(self) -> None:
        stencil_root, _manifest = self.build_stencil()
        self.write_pseudo("C")

        with self.assertRaisesRegex(DerivativeSiestaReferenceError, "failed for"):
            run_derivative_siesta_references(
                stencil_root=stencil_root,
                siesta_command=f"{sys.executable} -c \"import sys; sys.exit(3)\"",
                max_jobs=1,
            )

        manifest_path = stencil_root / "siesta_hamiltonians" / "derivative_siesta_reference_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["samples_failed"], 1)
        self.assertEqual(manifest["rows"][0]["returncode"], 3)
        self.assertEqual(manifest["rows"][0]["error"], "siesta_returncode_nonzero")

    def test_dataset_root_pseudopotential_is_staged_before_siesta_run(self) -> None:
        stencil_root, manifest = self.build_stencil()
        self.write_pseudo("C")
        sample_ids = [record["sample_id"] for record in manifest["samples"]]
        captured_dirs: list[Path] = []

        def fake_run(command, **kwargs):
            cwd = Path(kwargs["cwd"])
            captured_dirs.append(cwd)
            self.assertTrue((cwd / "C.psf").exists())
            (cwd / "siesta.TSHS").write_bytes(b"reference")
            return mock.Mock(returncode=0)

        with mock.patch("run_hamiltonian_derivative_siesta_references.subprocess.run", side_effect=fake_run):
            result = run_derivative_siesta_references(
                stencil_root=stencil_root,
                source_dataset_root=self.dataset_root,
                max_samples=1,
            )

        self.assertEqual(result["samples_ok"], 1)
        self.assertEqual(len(captured_dirs), 1)
        self.assertTrue((captured_dirs[0] / "C.psf").exists())
        self.assertEqual(result["source_dataset_root"], str(self.dataset_root))
        self.assertIn(captured_dirs[0].name, sample_ids)

    def test_ghost_pseudopotential_is_staged_when_species_is_declared(self) -> None:
        (self.sample_dir / "RUN.fdf").write_text(synthetic_base_fdf(include_ghost=True), encoding="utf-8")
        stencil_root, _manifest = self.build_stencil()
        self.write_pseudo("C")
        self.write_pseudo("Ghost-H")
        staged = {}

        def fake_run(command, **kwargs):
            cwd = Path(kwargs["cwd"])
            staged["files"] = sorted(path.name for path in cwd.iterdir() if path.is_file())
            self.assertTrue((cwd / "C.psf").exists())
            self.assertTrue((cwd / "Ghost-H.psf").exists())
            (cwd / "siesta.TSHS").write_bytes(b"reference")
            return mock.Mock(returncode=0)

        with mock.patch("run_hamiltonian_derivative_siesta_references.subprocess.run", side_effect=fake_run):
            result = run_derivative_siesta_references(
                stencil_root=stencil_root,
                source_dataset_root=self.dataset_root,
                max_samples=1,
            )

        self.assertEqual(result["samples_failed"], 0)
        self.assertIn("Ghost-H.psf", staged["files"])

    def test_missing_required_pseudo_fails_with_actionable_error(self) -> None:
        stencil_root, _manifest = self.build_stencil()

        result = run_derivative_siesta_references(
            stencil_root=stencil_root,
            source_dataset_root=self.dataset_root,
            diagnostic_only=True,
            max_samples=1,
        )

        self.assertEqual(result["samples_failed"], 1)
        error = result["rows"][0]["error"]
        self.assertIn("Missing required pseudopotential", error)
        self.assertIn("'C'", error)
        self.assertIn(str(self.dataset_root), error)
        self.assertIn("C.psf", error)
        self.assertIn("C.vps", error)
        self.assertIn("C.psml", error)

    def test_max_samples_limits_reference_samples(self) -> None:
        stencil_root, _manifest = self.build_stencil()
        self.write_pseudo("C")
        calls: list[str] = []

        def fake_run(command, **kwargs):
            cwd = Path(kwargs["cwd"])
            calls.append(cwd.name)
            (cwd / "siesta.TSHS").write_bytes(b"h")
            return mock.Mock(returncode=0)

        with mock.patch("run_hamiltonian_derivative_siesta_references.subprocess.run", side_effect=fake_run):
            result = run_derivative_siesta_references(
                stencil_root=stencil_root,
                workers=2,
                max_samples=1,
            )

        self.assertEqual(len(calls), 1)
        self.assertEqual(result["samples_total"], 1)
        self.assertEqual(result["samples_ok"], 1)
        self.assertEqual(result["max_samples"], 1)
        self.assertIsNone(result["max_jobs"])

    def test_max_jobs_alias_limits_reference_samples_and_is_recorded(self) -> None:
        stencil_root, _manifest = self.build_stencil()
        self.write_pseudo("C")
        calls: list[str] = []

        def fake_run(command, **kwargs):
            cwd = Path(kwargs["cwd"])
            calls.append(cwd.name)
            (cwd / "siesta.TSHS").write_bytes(b"h")
            return mock.Mock(returncode=0)

        with mock.patch("run_hamiltonian_derivative_siesta_references.subprocess.run", side_effect=fake_run):
            result = run_derivative_siesta_references(
                stencil_root=stencil_root,
                workers=2,
                max_jobs=1,
                max_jobs_alias_used=True,
            )

        self.assertEqual(len(calls), 1)
        self.assertEqual(result["samples_total"], 1)
        self.assertEqual(result["samples_ok"], 1)
        self.assertEqual(result["max_samples"], 1)
        self.assertEqual(result["max_jobs"], 1)
        self.assertTrue(result["max_jobs_alias_used"])

    def test_diagnostic_only_records_failure_while_other_samples_complete(self) -> None:
        stencil_root, manifest = self.build_stencil()
        self.write_pseudo("C")
        sample_ids = self.selected_sample_ids(manifest, 2)

        def fake_run(command, **kwargs):
            cwd = Path(kwargs["cwd"])
            if cwd.name == sample_ids[0]:
                return mock.Mock(returncode=3)
            (cwd / "siesta.TSHS").write_bytes(b"reference")
            return mock.Mock(returncode=0)

        with mock.patch("run_hamiltonian_derivative_siesta_references.subprocess.run", side_effect=fake_run):
            result = run_derivative_siesta_references(
                stencil_root=stencil_root,
                workers=2,
                max_samples=2,
                diagnostic_only=True,
            )

        self.assertEqual([row["sample_id"] for row in result["rows"]], sample_ids)
        self.assertEqual(result["samples_failed"], 1)
        self.assertEqual(result["samples_ok"], 1)
        self.assertEqual(result["rows"][0]["status"], "error")
        self.assertEqual(result["rows"][1]["status"], "ok")

    def test_failures_write_manifest_and_status_before_raise(self) -> None:
        stencil_root, manifest = self.build_stencil()
        self.write_pseudo("C")
        sample_ids = self.selected_sample_ids(manifest, 2)

        def fake_run(command, **kwargs):
            cwd = Path(kwargs["cwd"])
            if cwd.name == sample_ids[0]:
                return mock.Mock(returncode=3)
            (cwd / "siesta.TSHS").write_bytes(b"reference")
            return mock.Mock(returncode=0)

        with mock.patch("run_hamiltonian_derivative_siesta_references.subprocess.run", side_effect=fake_run):
            with self.assertRaisesRegex(DerivativeSiestaReferenceError, "failed for"):
                run_derivative_siesta_references(
                    stencil_root=stencil_root,
                    workers=2,
                    max_samples=2,
                )

        output_root = stencil_root / "siesta_hamiltonians"
        manifest_path = output_root / "derivative_siesta_reference_manifest.json"
        status_path = output_root / "derivative_siesta_reference_status.csv"
        written_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        with status_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))

        self.assertEqual(written_manifest["samples_failed"], 1)
        self.assertEqual([row["sample_id"] for row in written_manifest["rows"]], sample_ids)
        self.assertEqual(len(rows), 2)

    def test_invalid_workers_fail_clearly(self) -> None:
        stencil_root, _manifest = self.build_stencil()

        for workers in (0, -1, 1.5):
            with self.subTest(workers=workers):
                with self.assertRaisesRegex(DerivativeSiestaReferenceError, "workers must be a positive integer"):
                    run_derivative_siesta_references(
                        stencil_root=stencil_root,
                        workers=workers,
                    )

    def test_siesta_command_runs_as_argv_by_default(self) -> None:
        stencil_root, _manifest = self.build_stencil()
        self.write_pseudo("C")
        captured = {}

        def fake_run(command, **kwargs):
            captured["command"] = command
            captured["shell"] = kwargs.get("shell")
            self.assertTrue((Path(kwargs["cwd"]) / "C.psf").exists())
            (Path(kwargs["cwd"]) / "siesta.TSHS").write_bytes(b"reference")
            return mock.Mock(returncode=0)

        with mock.patch("run_hamiltonian_derivative_siesta_references.subprocess.run", side_effect=fake_run):
            result = run_derivative_siesta_references(
                stencil_root=stencil_root,
                siesta_command=f"{sys.executable} -c \"print('siesta')\"",
                max_samples=1,
            )

        self.assertIsInstance(captured["command"], list)
        self.assertFalse(captured["shell"])
        self.assertEqual(result["samples_failed"], 0)
        self.assertFalse(result["siesta_shell"])

    def test_staged_reference_artifacts_are_discoverable(self) -> None:
        stencil_root, manifest = self.build_stencil()
        existing_root = self.root / "existing_references"
        for record in manifest["samples"]:
            ref_dir = existing_root / record["sample_id"]
            self.write_proven_reference(
                ref_dir,
                f"reference {record['sample_id']}".encode("utf-8"),
            )

        result = run_derivative_siesta_references(
            stencil_root=stencil_root,
            existing_reference_root=existing_root,
        )

        self.assertEqual(result["samples_failed"], 0)
        self.assertTrue(all(row["status"] == "staged" for row in result["rows"]))
        discoveries = discover_derivative_stencils(
            stencil_root,
            method="graph2mat",
            split="test",
            finite_difference_method="central",
            require_central=True,
        )
        self.assertEqual(len(discoveries), 1)
        self.assertIsNotNone(discoveries[0].stencil)
        self.assertIsNotNone(discoveries[0].stencil.siesta_plus)
        self.assertIsNotNone(discoveries[0].stencil.siesta_minus)
        self.assertNotEqual(discoveries[0].stencil.siesta_plus.matrix_path.name, "ML_prediction.HSX")


class ReferenceGeometryGuardTests(unittest.TestCase):
    # Regression: MD settings leaked into stencil fdfs made SIESTA evolve the
    # geometry, so the stored TSHS no longer matched the +-delta displacement.
    CELL = [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]

    def test_matching_geometry_passes(self) -> None:
        fdf = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
        matrix = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.00005]]
        self.assertEqual(geometry_mismatch_message(fdf, matrix), "")

    def test_md_drifted_geometry_fails_closed(self) -> None:
        fdf = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.01]]
        matrix = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.045]]
        message = geometry_mismatch_message(fdf, matrix, fdf_cell=self.CELL, matrix_cell=self.CELL)
        self.assertIn("reference_geometry_mismatch", message)
        self.assertIn("single-point", message)

    def test_periodically_wrapped_positions_pass_with_minimum_image(self) -> None:
        # Same physical position, wrapped across the boundary: |raw diff| ~ 10 Ang.
        fdf = [[9.99995, 0.0, 0.0]]
        matrix = [[-0.00005, 0.0, 0.0]]
        self.assertEqual(
            geometry_mismatch_message(fdf, matrix, fdf_cell=self.CELL, matrix_cell=self.CELL), ""
        )

    def test_cell_mismatch_fails_closed(self) -> None:
        other_cell = [[10.5, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]
        message = geometry_mismatch_message(
            [[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]], fdf_cell=self.CELL, matrix_cell=other_cell
        )
        self.assertIn("lattice vectors differ", message)

    def test_atom_count_mismatch_fails_closed(self) -> None:
        message = geometry_mismatch_message([[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        self.assertIn("reference_geometry_mismatch", message)


if __name__ == "__main__":
    unittest.main()
