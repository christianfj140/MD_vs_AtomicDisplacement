"""Dedicated tests for run_deeph_autograd_derivative_predictions.py.

Everything runs with a fake DeepH CLI (subprocess stub) — no DeepH install, no
models, no SIESTA — following the fixture pattern of
tests/test_run_hamiltonian_derivative_predictions.py.
"""

from __future__ import annotations

import configparser
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import h5py
import numpy as np
from scipy import sparse

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "Comparison" / "scripts"
SHARED_DIR = REPO_ROOT / "shared"
for directory in (SCRIPTS_DIR, SHARED_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from build_hamiltonian_derivative_stencils import build_derivative_stencils  # noqa: E402
from deeph_prediction_adapter import DeepHPredictionAdapterResult  # noqa: E402
import run_deeph_autograd_derivative_predictions as rd  # noqa: E402


def synthetic_base_fdf() -> str:
    return "\n".join(
        [
            "SystemName synthetic base",
            "SystemLabel shared_label",
            "NumberOfSpecies 1",
            "NumberOfAtoms 2",
            "%block ChemicalSpeciesLabel",
            " 1 6 C",
            "%endblock ChemicalSpeciesLabel",
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


def _displaced_payload(
    base_id: str,
    atom: int,
    axis: str,
    delta: float = 0.1,
    structure_id: str | None = None,
) -> dict:
    structure_id = structure_id or f"{base_id}_atom{atom:04d}_{axis}_d{delta:g}_plus"
    return {
        "base_sample_id": base_id,
        "atom_index_zero_based": atom,
        "axis": axis,
        "delta_ang": delta,
        "sign": 1,
        "is_reference": False,
        "_sample_dir": Path(f"/fake/{structure_id}"),
        "_structure_sample_id": structure_id,
    }


def _base_payload(base_id: str) -> dict:
    return {
        "base_sample_id": base_id,
        "is_reference": True,
        "sign": 0,
        "split": "test",
        "_sample_dir": Path(f"/fake/{base_id}_base"),
        "_structure_sample_id": f"{base_id}_base",
    }


class CollectDerivativeRequestsTests(unittest.TestCase):
    def test_groups_atom_axis_pairs_by_base_structure(self) -> None:
        payloads = [
            _base_payload("md_1"),
            _displaced_payload("md_1", 0, "x", 0.05),
            _displaced_payload("md_1", 0, "x", 0.1, structure_id="md_1_atom0_x_other"),
            _displaced_payload("md_1", 1, "y"),
        ]
        requests = rd.collect_derivative_requests(
            payloads, base_sample_ids=[], atoms=[], axes=[], max_base_structures=None
        )
        self.assertEqual(len(requests), 1)
        request = requests[0]
        self.assertEqual(request["base_sample_id"], "md_1")
        self.assertEqual(request["base_structure_sample_id"], "md_1_base")
        self.assertEqual(
            request["pairs"], {(0, "x"): {0.05, 0.1}, (1, "y"): {0.1}}
        )

    def test_missing_base_structure_raises(self) -> None:
        payloads = [
            _base_payload("md_1"),
            _displaced_payload("md_1", 0, "x"),
            _displaced_payload("md_2", 0, "x"),  # md_2 has no base sample
        ]
        with self.assertRaises(rd.DeepHAutogradDerivativePredictionError) as ctx:
            rd.collect_derivative_requests(
                payloads, base_sample_ids=[], atoms=[], axes=[], max_base_structures=None
            )
        self.assertIn("md_2", str(ctx.exception))

    def test_filters_atoms_axes_and_base_sample_id(self) -> None:
        payloads = [
            _base_payload("md_1"),
            _base_payload("md_2"),
            _displaced_payload("md_1", 0, "x"),
            _displaced_payload("md_1", 1, "y"),
            _displaced_payload("md_2", 0, "x"),
        ]
        requests = rd.collect_derivative_requests(
            payloads, base_sample_ids=["md_1"], atoms=[0], axes=["x"], max_base_structures=None
        )
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0]["base_sample_id"], "md_1")
        self.assertEqual(list(requests[0]["pairs"]), [(0, "x")])
        # Filters that match nothing fail loudly instead of silently predicting nothing.
        with self.assertRaises(rd.DeepHAutogradDerivativePredictionError):
            rd.collect_derivative_requests(
                payloads, base_sample_ids=[], atoms=[5], axes=[], max_base_structures=None
            )

    def test_max_base_structures_limits_requests(self) -> None:
        payloads = [
            _base_payload("md_1"),
            _base_payload("md_2"),
            _displaced_payload("md_1", 0, "x"),
            _displaced_payload("md_2", 0, "x", structure_id="md_2_atom0_x"),
        ]
        requests = rd.collect_derivative_requests(
            payloads, base_sample_ids=[], atoms=[], axes=[], max_base_structures=1
        )
        self.assertEqual([request["base_sample_id"] for request in requests], ["md_1"])


class SelectGradientBlockTests(unittest.TestCase):
    def test_selects_atom_axis_component(self) -> None:
        block = np.arange(4 * 4 * 2 * 3, dtype=float).reshape(4, 4, 2, 3)
        selected = rd._select_gradient_block(block, 1, 2)
        np.testing.assert_array_equal(selected, block[..., 1, 2])

    def test_rejects_non_4d_block(self) -> None:
        with self.assertRaises(rd.DeepHAutogradDerivativePredictionError):
            rd._select_gradient_block(np.zeros((4, 4)), 0, 0)

    def test_rejects_out_of_range_indices(self) -> None:
        block = np.zeros((4, 4, 2, 3))
        with self.assertRaises(rd.DeepHAutogradDerivativePredictionError):
            rd._select_gradient_block(block, 2, 0)
        with self.assertRaises(rd.DeepHAutogradDerivativePredictionError):
            rd._select_gradient_block(block, 0, 3)


class AutogradFlowTests(unittest.TestCase):
    """End-to-end flow with a fake DeepH CLI and a fake hamiltonians_grad_pred.h5."""

    GRAD_SHAPE = (4, 4, 2, 3)  # rows, cols, atoms, axes
    ATOM = 1
    AXIS = "y"
    AXIS_INDEX = 1

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        dataset_root = self.root / "source_dataset"
        sample_dir = dataset_root / "splits" / "test" / "base_0"
        sample_dir.mkdir(parents=True)
        (sample_dir / "RUN.fdf").write_text(synthetic_base_fdf(), encoding="utf-8")
        (sample_dir / "metadata.json").write_text(
            json.dumps({"sample_id": "base_0", "material_label": "synthetic"}), encoding="utf-8"
        )
        (dataset_root / "frozen_split_manifest.json").write_text(
            json.dumps({"rows": [{"sample_id": "base_0", "split": "test", "sample_dir": str(sample_dir)}]}),
            encoding="utf-8",
        )
        self.stencil_root = self.root / "stencils"
        manifest = build_derivative_stencils(
            source_dataset_root=dataset_root,
            output_stencil_root=self.stencil_root,
            split="test",
            method="central",
            delta_ang_values=[0.1],
            atom_indices_zero_based=[self.ATOM],
            axes=[self.AXIS],
        )
        siesta_root = self.stencil_root / "siesta_hamiltonians"
        for record in manifest["samples"]:
            sample_id = record["sample_id"]
            ref_dir = siesta_root / sample_id
            ref_dir.mkdir(parents=True, exist_ok=True)
            for suffix in (".HSX", ".STRUCT_OUT", ".XV", ".ORB_INDX"):
                (ref_dir / f"{sample_id}{suffix}").write_text("stub\n", encoding="utf-8")

        self.model_dir = self.root / "deeph_model"
        self.model_dir.mkdir()
        (self.model_dir / "config.ini").write_text(
            "[basic]\ndisable_cuda = True\ndevice = cpu\n[graph]\nradius = -1.0\n", encoding="utf-8"
        )
        self.output_root = self.root / "deeph_autograd_predictions"
        deeph_repo = self.root / "DeepH-pack"
        (deeph_repo / "deeph").mkdir(parents=True)
        cli_dir = deeph_repo / ".venv" / "bin"
        cli_dir.mkdir(parents=True)
        self.inference_cli = cli_dir / "deeph-inference"
        self.inference_cli.write_text("#!/bin/sh\n", encoding="utf-8")
        (cli_dir / "deeph-preprocess").write_text("#!/bin/sh\n", encoding="utf-8")
        self.grad_block = np.arange(np.prod(self.GRAD_SHAPE), dtype=np.float64).reshape(self.GRAD_SHAPE)
        self.inference_configs: list[configparser.ConfigParser] = []

    def tearDown(self) -> None:
        self.tmp.cleanup()

    # ------------------------------------------------------------------ #
    # fakes
    # ------------------------------------------------------------------ #
    def fake_raw_mirror(self, *, references, raw_dir):
        rows = []
        for index, sample_id in enumerate(sorted(references)):
            raw_sample = Path(raw_dir) / f"{index:06d}_{sample_id}"
            raw_sample.mkdir(parents=True, exist_ok=True)
            rows.append({"sample_id": sample_id, "raw_dir": str(raw_sample)})
        return {"rows": rows}

    def make_fake_run(self, *, preprocess_rc=0, inference_rc=0, write_grad_h5=True):
        def fake_run(command, **_kwargs):
            command_list = list(command)
            name = Path(command_list[0]).name
            if name == "deeph-preprocess":
                if preprocess_rc == 0:
                    config = configparser.ConfigParser()
                    config.read(command_list[-1])
                    raw_dir = Path(config.get("basic", "raw_dir"))
                    processed_dir = Path(config.get("basic", "processed_dir"))
                    for raw_sample in sorted(raw_dir.iterdir()):
                        processed = processed_dir / raw_sample.name
                        processed.mkdir(parents=True, exist_ok=True)
                        for filename in (
                            "R_list.dat", "orbital_types.dat", "element.dat",
                            "site_positions.dat", "lat.dat", "rlat.dat", "rc.h5",
                            "hamiltonians.h5", "overlaps.h5",
                        ):
                            (processed / filename).write_text("stub\n", encoding="utf-8")
                return mock.Mock(returncode=preprocess_rc, stdout="", stderr="")
            if name == "deeph-inference":
                config = configparser.ConfigParser()
                config.read(command_list[-1])
                self.inference_configs.append(config)
                work_dir = Path(config.get("basic", "work_dir"))
                work_dir.mkdir(parents=True, exist_ok=True)
                if inference_rc == 0:
                    with h5py.File(work_dir / "hamiltonians_pred.h5", "w") as fid:
                        fid["[0, 0, 0, 1, 1]"] = self.grad_block[..., 0, 0]
                    if write_grad_h5:
                        with h5py.File(work_dir / "hamiltonians_grad_pred.h5", "w") as fid:
                            fid["[0, 0, 0, 1, 1]"] = self.grad_block
                return mock.Mock(returncode=inference_rc, stdout="", stderr="")
            raise AssertionError(command_list)

        return fake_run

    def fake_adapter(self, *, work_dir, processed_sample_dir, sample_id,
                     prediction_filename="hamiltonians_pred.h5"):
        return DeepHPredictionAdapterResult(
            sample_id=sample_id,
            status="ok",
            metrics_ready=True,
            diagnostic_only=True,
            diagnostic_reason="diagnostic",
            prediction_path=str(Path(work_dir) / prediction_filename),
            processed_sample_dir=str(processed_sample_dir),
            reference_hamiltonian_path=str(Path(processed_sample_dir) / "hamiltonians.h5"),
            reference_overlap_path=str(Path(processed_sample_dir) / "overlaps.h5"),
            orbital_types_path=str(Path(processed_sample_dir) / "orbital_types.dat"),
            n_orbitals=4,
            block_count=1,
            prediction_key_count=1,
            reference_key_count=1,
        )

    def fake_reconstruct(self, *, prediction_h5, output_path, block_transform=None, **_kwargs):
        # Apply the runner's block_transform to a known 4-d block so the test
        # can verify the exact (atom, axis) component ends up in the .npz.
        with h5py.File(prediction_h5, "r") as fid:
            block = np.asarray(fid["[0, 0, 0, 1, 1]"])
        selected = block_transform(block) if block_transform is not None else block
        matrix = sparse.csr_matrix(np.asarray(selected, dtype=np.float64))
        with Path(output_path).open("wb") as handle:
            sparse.save_npz(handle, matrix)
        return {"kind": "deeph_h5_reconstructed_siesta_sparse_layout_v1",
                "nnz": int(matrix.nnz), "shape_rows": int(matrix.shape[0]),
                "shape_cols": int(matrix.shape[1])}

    FAKE_CAPABILITY = {
        "schema": "deeph_autograd_capability_v1",
        "available": True,
        "implementation": "torch_forward_ad_jvp",
        "output_schema": "hamiltonians_grad_pred_v2",
    }

    def run_flow(self, fake_run, *, expect_error=False):
        args = rd.build_argument_parser().parse_args(
            [
                "--stencil-root", str(self.stencil_root),
                "--model-dir", str(self.model_dir),
                "--output-root", str(self.output_root),
                "--deeph-command", str(self.inference_cli),
            ]
        )
        with mock.patch.object(rd, "build_deeph_derivative_raw_mirror", side_effect=self.fake_raw_mirror), \
                mock.patch("run_hamiltonian_derivative_predictions.subprocess.run", side_effect=fake_run), \
                mock.patch.object(rd, "deeph_autograd_capability_preflight",
                                  return_value=dict(self.FAKE_CAPABILITY)), \
                mock.patch.object(rd, "adapt_deeph_prediction_sample", side_effect=self.fake_adapter), \
                mock.patch.object(rd, "reconstruct_deeph_sparse_layout_prediction",
                                  side_effect=self.fake_reconstruct):
            if expect_error:
                with self.assertRaises(rd.DeepHAutogradDerivativePredictionError) as ctx:
                    rd.run_deeph_autograd_derivative_predictions(args)
                return ctx.exception
            return rd.run_deeph_autograd_derivative_predictions(args)

    def read_status_rows(self) -> list[dict]:
        import csv

        with (self.output_root / rd.STATUS_FILENAME).open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    # ------------------------------------------------------------------ #
    # tests
    # ------------------------------------------------------------------ #
    def test_full_flow_writes_npz_and_metadata(self) -> None:
        manifest = self.run_flow(self.make_fake_run())
        self.assertEqual(manifest["samples_failed"], 0)
        self.assertEqual(manifest["samples_ok"], 1)

        structure_id = manifest["rows"][0]["base_structure_sample_id"]
        basename = rd.direct_derivative_prediction_basename(self.ATOM, self.AXIS_INDEX)
        npz_path = self.output_root / structure_id / f"{basename}.npz"
        json_path = self.output_root / structure_id / f"{basename}.json"
        self.assertTrue(npz_path.is_file(), npz_path)
        self.assertTrue(json_path.is_file(), json_path)

        # The .npz holds exactly the (atom, axis) gradient component.
        matrix = sparse.load_npz(npz_path).toarray()
        np.testing.assert_allclose(matrix, self.grad_block[..., self.ATOM, self.AXIS_INDEX])

        metadata = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertEqual(metadata["schema"], "deeph_autograd_direct_derivative_v1")
        self.assertEqual(metadata["predicted_derivative_method"], rd.PREDICTED_DERIVATIVE_METHOD_AUTOGRAD_DEEPH)
        self.assertEqual(metadata["deeph_prediction_method"], rd.DEEPH_PREDICTION_METHOD_AUTOGRAD)
        self.assertIsNone(metadata["predicted_delta_ang"])
        self.assertEqual(metadata["reference_delta_ang"], 0.1)
        self.assertTrue(metadata["with_grad"])
        self.assertTrue(metadata["topology_fixed"])
        self.assertEqual(metadata["atom_index_zero_based"], self.ATOM)
        self.assertEqual(metadata["axis"], self.AXIS)
        # _deeph_base_equivalence_fields must be merged into the metadata.
        self.assertIn("claim_status", metadata)
        self.assertIn(metadata["claim_status"], {"diagnostic_only", "raw_global_equivalence_proven"})

        # The rendered inference config asked for exactly the needed directions.
        self.assertTrue(self.inference_configs)
        config = self.inference_configs[0]
        self.assertTrue(config.getboolean("basic", "with_grad"))
        self.assertEqual(json.loads(config.get("basic", "grad_atom_indices")), [self.ATOM])
        self.assertEqual(json.loads(config.get("basic", "grad_axis_indices")), [self.AXIS_INDEX])

        rows = self.read_status_rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "predicted")
        self.assertEqual(rows[0]["error"], "")

    def test_preprocess_failure_fails_closed_with_status_csv(self) -> None:
        exception = self.run_flow(self.make_fake_run(preprocess_rc=1), expect_error=True)
        self.assertIn("preprocess", str(exception).lower())
        rows = self.read_status_rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "error")
        self.assertEqual(rows[0]["error"], "deeph_preprocess_failed")
        self.assertTrue((self.output_root / rd.MANIFEST_FILENAME).is_file())

    def test_inference_failure_marks_rows_error(self) -> None:
        self.run_flow(self.make_fake_run(inference_rc=1), expect_error=True)
        rows = self.read_status_rows()
        self.assertEqual(rows[0]["status"], "error")
        self.assertEqual(rows[0]["error"], "prediction_command_failed")

    def test_missing_grad_h5_marks_rows_error(self) -> None:
        self.run_flow(self.make_fake_run(write_grad_h5=False), expect_error=True)
        rows = self.read_status_rows()
        self.assertEqual(rows[0]["status"], "error")
        self.assertEqual(rows[0]["error"], "missing_hamiltonians_grad_pred_h5")

    def test_nan_in_requested_direction_fails_closed(self) -> None:
        # hamiltonians_grad_pred_v2: NaN means "never computed"; a requested
        # direction that comes back NaN must never be marked 'predicted'.
        self.grad_block[..., self.ATOM, self.AXIS_INDEX] = np.nan
        self.run_flow(self.make_fake_run(), expect_error=True)
        rows = self.read_status_rows()
        self.assertEqual(rows[0]["status"], "error")
        self.assertIn("deeph_autograd_non_finite", rows[0]["error"])

    def test_capability_preflight_blocks_before_inference(self) -> None:
        args = rd.build_argument_parser().parse_args(
            [
                "--stencil-root", str(self.stencil_root),
                "--model-dir", str(self.model_dir),
                "--output-root", str(self.output_root),
                "--deeph-command", str(self.inference_cli),
            ]
        )
        with mock.patch.object(rd, "build_deeph_derivative_raw_mirror", side_effect=self.fake_raw_mirror), \
                mock.patch("run_hamiltonian_derivative_predictions.subprocess.run",
                           side_effect=self.make_fake_run()), \
                mock.patch.object(
                    rd, "deeph_autograd_capability_preflight",
                    side_effect=rd.DeepHAutogradDerivativePredictionError("capability_unavailable: stub"),
                ):
            with self.assertRaises(rd.DeepHAutogradDerivativePredictionError) as ctx:
                rd.run_deeph_autograd_derivative_predictions(args)
        self.assertIn("capability_unavailable", str(ctx.exception))
        self.assertEqual(self.inference_configs, [])  # inference never launched


class CapabilityPreflightTests(unittest.TestCase):
    """deeph_autograd_capability_preflight parses/validates the backend manifest."""

    def _record(self, payload, returncode=0, stderr=""):
        return {
            "command": ["python", "-c", "stub"],
            "returncode": returncode,
            "stdout": json.dumps(payload) if isinstance(payload, dict) else str(payload),
            "stderr": stderr,
            "started_at": 0.0,
            "finished_at": 0.0,
        }

    def test_accepts_real_jvp_manifest(self) -> None:
        manifest = {
            "available": True,
            "implementation": "torch_forward_ad_jvp",
            "output_schema": "hamiltonians_grad_pred_v2",
        }
        with mock.patch.object(rd, "run_command", return_value=self._record(manifest)):
            result = rd.deeph_autograd_capability_preflight("python")
        self.assertTrue(result["available"])

    def test_rejects_placeholder_backend(self) -> None:
        manifest = {"available": False, "implementation": None, "errors": ["missing__forward_ad_jvp_blocks"]}
        with mock.patch.object(rd, "run_command", return_value=self._record(manifest)):
            with self.assertRaises(rd.DeepHAutogradDerivativePredictionError) as ctx:
                rd.deeph_autograd_capability_preflight("python")
        self.assertIn("capability_unavailable", str(ctx.exception))

    def test_rejects_backend_without_capability_module(self) -> None:
        record = self._record("ModuleNotFoundError", returncode=1, stderr="No module named capability")
        with mock.patch.object(rd, "run_command", return_value=record):
            with self.assertRaises(rd.DeepHAutogradDerivativePredictionError) as ctx:
                rd.deeph_autograd_capability_preflight("python")
        self.assertIn("capability_unavailable", str(ctx.exception))

    def test_rejects_unknown_output_schema(self) -> None:
        manifest = {
            "available": True,
            "implementation": "torch_forward_ad_jvp",
            "output_schema": "hamiltonians_grad_pred_v1",
        }
        with mock.patch.object(rd, "run_command", return_value=self._record(manifest)):
            with self.assertRaises(rd.DeepHAutogradDerivativePredictionError) as ctx:
                rd.deeph_autograd_capability_preflight("python")
        self.assertIn("unsupported grad output schema", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
