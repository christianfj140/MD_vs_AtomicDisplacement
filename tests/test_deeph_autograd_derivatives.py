from __future__ import annotations

import argparse
import configparser
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import h5py
import numpy as np
from scipy import sparse

try:
    import pytest
except ImportError:  # pragma: no cover - unittest can still run the direct script test.
    pytest = None


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "Comparison" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_deeph_autograd_derivative_predictions as deeph_autograd  # noqa: E402
from deeph_config import default_deeph_paths  # noqa: E402
from hamiltonian_derivative_stencil import PREDICTED_DERIVATIVE_METHOD_AUTOGRAD_DEEPH  # noqa: E402


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


class DeepHAutogradDerivativeScriptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _write_stencil_fixture(self) -> Path:
        stencil_root = self.root / "stencil"
        base = stencil_root / "structures" / "base_0_base"
        plus = stencil_root / "structures" / "base_0_plus"
        base.mkdir(parents=True)
        plus.mkdir(parents=True)
        write_json(
            base / "metadata.json",
            {
                "sample_id": "base_0",
                "base_sample_id": "base_0",
                "is_reference": True,
                "sign": 0,
                "split": "test",
            },
        )
        write_json(
            plus / "metadata.json",
            {
                "sample_id": "base_0_plus",
                "base_sample_id": "base_0",
                "atom_index_zero_based": 0,
                "axis": "z",
                "axis_index": 2,
                "delta_ang": 0.01,
                "sign": 1,
                "split": "test",
            },
        )
        return stencil_root

    def test_script_runs_autograd_flow_and_writes_direct_derivative_metadata(self) -> None:
        stencil_root = self._write_stencil_fixture()
        output_root = self.root / "predicted_derivatives"
        model_dir = self.root / "deeph_model"
        model_dir.mkdir()
        (model_dir / "config.ini").write_text("[graph]\nradius = 5.0\n[basic]\ndisable_cuda = True\ndevice = cpu\n", encoding="utf-8")
        bin_dir = self.root / "bin"
        bin_dir.mkdir()
        inference_cli = bin_dir / "deeph-inference"
        preprocess_cli = bin_dir / "deeph-preprocess"
        inference_cli.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        preprocess_cli.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        inference_cli.chmod(0o755)
        preprocess_cli.chmod(0o755)

        deeph_paths = default_deeph_paths(output_root.parent)
        processed_sample = deeph_paths.processed_dir / "base_0_base"
        config_checks: list[dict[str, object]] = []
        commands: list[str] = []

        def fake_references(_stencil_root: Path, *, structures: list[Path]) -> dict[str, Path]:
            return {structure.name: self.root / "siesta_refs" / structure.name for structure in structures}

        def fake_raw_mirror(*, references: dict[str, Path], raw_dir: Path) -> dict:
            raw_sample = raw_dir / "base_0_base"
            raw_sample.mkdir(parents=True, exist_ok=True)
            return {
                "rows": [
                    {
                        "sample_id": "base_0_base",
                        "raw_dir": str(raw_sample),
                        "source_dir": str(references["base_0_base"]),
                    }
                ]
            }

        def fake_run_command(command, *, cwd, env):
            commands.append(Path(command[0]).name)
            if Path(command[0]).name == "deeph-preprocess":
                processed_sample.mkdir(parents=True, exist_ok=True)
                (processed_sample / "lat.dat").write_text("1 0 0\n0 1 0\n0 0 1\n", encoding="utf-8")
                return {"command": command, "returncode": 0, "stdout": "", "stderr": "", "started_at": 1.0, "finished_at": 2.0}

            config = configparser.ConfigParser()
            config.read(command[command.index("--config") + 1])
            work_dir = Path(config.get("basic", "work_dir"))
            config_checks.append(
                {
                    "with_grad": config.getboolean("basic", "with_grad"),
                    "task": json.loads(config.get("basic", "task")),
                    "grad_atom_indices": json.loads(config.get("basic", "grad_atom_indices")),
                    "grad_axis_indices": json.loads(config.get("basic", "grad_axis_indices")),
                }
            )
            work_dir.mkdir(parents=True, exist_ok=True)
            (work_dir / "hamiltonians_grad_pred.h5").write_bytes(b"grad-h5")
            (work_dir / "hamiltonians_pred.h5").write_bytes(b"pred-h5")
            return {"command": command, "returncode": 0, "stdout": "", "stderr": "", "started_at": 3.0, "finished_at": 4.0}

        def fake_reconstruct(*, output_path: Path, **_kwargs):
            sparse.save_npz(output_path, sparse.csr_matrix([[2.0]]))
            return {"kind": "fake_deeph_sparse_layout", "shape_rows": 1, "shape_cols": 1, "nnz": 1}

        adapter = SimpleNamespace(
            diagnostic_only=False,
            metric_fields=lambda: {
                "deeph_raw_global_equivalence_proven": True,
                "deeph_diagnostic_only": False,
                "deeph_equivalence_status": "proven",
            },
            to_dict=lambda: {"diagnostic_only": False},
        )
        args = argparse.Namespace(
            stencil_root=stencil_root,
            output_root=output_root,
            model_dir=model_dir,
            deeph_command=str(inference_cli),
            python_executable=sys.executable,
            overwrite=True,
            skip_if_exists=True,
            base_sample_id=[],
            atoms=[],
            axes=[],
            max_base_structures=None,
            max_samples=None,
        )

        with mock.patch.object(deeph_autograd, "discover_siesta_reference_samples", side_effect=fake_references), mock.patch.object(
            deeph_autograd, "build_deeph_derivative_raw_mirror", side_effect=fake_raw_mirror
        ), mock.patch.object(deeph_autograd, "run_command", side_effect=fake_run_command), mock.patch.object(
            deeph_autograd, "reconstruct_deeph_sparse_layout_prediction", side_effect=fake_reconstruct
        ), mock.patch.object(
            deeph_autograd, "adapt_deeph_prediction_sample", return_value=adapter
        ):
            manifest = deeph_autograd.run_deeph_autograd_derivative_predictions(args)

        self.assertEqual(commands, ["deeph-preprocess", "deeph-inference"])
        self.assertEqual(config_checks, [{"with_grad": True, "task": [3], "grad_atom_indices": [0], "grad_axis_indices": [2]}])
        self.assertEqual(manifest["samples_failed"], 0)
        self.assertEqual(manifest["predicted_derivative_method"], PREDICTED_DERIVATIVE_METHOD_AUTOGRAD_DEEPH)
        self.assertEqual(manifest["deeph_prediction_method"], "autograd_vectorized")
        metadata_path = output_root / "base_0_base" / "dH_pred_atom0_axis2.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.assertTrue(metadata["with_grad"])
        self.assertIsNone(metadata["predicted_delta_ang"])
        self.assertTrue(metadata["deeph_raw_global_equivalence_proven"])
        self.assertFalse(metadata["deeph_diagnostic_only"])
        self.assertTrue((output_root / "base_0_base" / "dH_pred_atom0_axis2.npz").exists())


def _read_h5_blocks(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as handle:
        return {key: np.asarray(handle[key][()]) for key in handle.keys()}


def _copy_sample_with_shift(source: Path, target: Path, *, atom: int, axis: int, delta: float) -> None:
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)
    positions_path = target / "site_positions.dat"
    positions = np.loadtxt(positions_path)
    shifted = np.array(positions, copy=True, ndmin=2)
    if shifted.shape[0] == 3:
        shifted[axis, atom] += delta
    elif shifted.shape[1] == 3:
        shifted[atom, axis] += delta
    else:
        raise AssertionError(f"Unsupported site_positions.dat shape: {shifted.shape}")
    np.savetxt(positions_path, shifted)
    for stale in ("rc.h5", "hamiltonians_pred.h5", "hamiltonians_grad_pred.h5", "rh_pred.h5"):
        path = target / stale
        if path.exists():
            path.unlink()


def _run_deeph_inference(*, deeph_cli: Path, config_path: Path) -> None:
    completed = subprocess.run(
        [str(deeph_cli), "--config", str(config_path)],
        cwd=str(deeph_cli.resolve(strict=False).parents[2]) if len(deeph_cli.resolve(strict=False).parents) > 2 else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"DeepH inference failed with {completed.returncode}\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )


slow_mark = pytest.mark.slow if pytest is not None else (lambda func: func)


class DeepHAutogradFiniteDifferenceSmokeTests(unittest.TestCase):
    @slow_mark
    def test_deeph_autograd_matches_predict_finite_difference_when_fixture_is_available(self) -> None:
        """DeepH dH/dR from with_grad vs central finite difference of DeepH predict()."""

        from deeph_config import render_inference_config

        model_env = os.environ.get("DEEPH_AUTOGRAD_SMOKE_MODEL_DIR")
        sample_env = os.environ.get("DEEPH_AUTOGRAD_SMOKE_SAMPLE_DIR")
        if not model_env:
            raise unittest.SkipTest("Set DEEPH_AUTOGRAD_SMOKE_MODEL_DIR to a trained DeepH model directory.")
        if not sample_env:
            raise unittest.SkipTest("Set DEEPH_AUTOGRAD_SMOKE_SAMPLE_DIR to a processed DeepH sample/work directory.")
        model_dir = Path(model_env)
        sample_dir = Path(sample_env)
        deeph_cli = Path(
            os.environ.get(
                "DEEPH_AUTOGRAD_SMOKE_DEEPH_COMMAND",
                str(Path("/home/christian/repositorios/DeepH-pack/.venv/bin/deeph-inference")),
            )
        )
        if not model_dir.is_dir():
            raise unittest.SkipTest(f"DeepH model directory is unavailable: {model_dir}")
        if not sample_dir.is_dir():
            raise unittest.SkipTest(f"DeepH sample/work directory is unavailable: {sample_dir}")
        if not deeph_cli.is_file():
            raise unittest.SkipTest(f"DeepH inference CLI is unavailable: {deeph_cli}")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            atom = int(os.environ.get("DEEPH_AUTOGRAD_SMOKE_ATOM", "0"))
            axis = int(os.environ.get("DEEPH_AUTOGRAD_SMOKE_AXIS", "2"))
            deltas = [float(item) for item in os.environ.get("DEEPH_AUTOGRAD_SMOKE_DELTAS", "0.01,0.005").split(",")]
            base = tmp_path / "base"
            shutil.copytree(sample_dir, base)
            autograd_config = tmp_path / "with_grad.ini"
            render_inference_config(
                autograd_config,
                work_dir=base,
                trained_model_dir=model_dir,
                python_interpreter=sys.executable,
                task=[3],
                with_grad=True,
                grad_atom_indices=[atom],
                grad_axis_indices=[axis],
            )
            _run_deeph_inference(deeph_cli=deeph_cli, config_path=autograd_config)
            grad_blocks = _read_h5_blocks(base / "hamiltonians_grad_pred.h5")
            rel_errors: list[float] = []
            compared = 0
            grad_norm = 0.0
            for delta in deltas:
                plus = tmp_path / f"plus_{delta:g}"
                minus = tmp_path / f"minus_{delta:g}"
                _copy_sample_with_shift(sample_dir, plus, atom=atom, axis=axis, delta=delta)
                _copy_sample_with_shift(sample_dir, minus, atom=atom, axis=axis, delta=-delta)
                for work_dir in (plus, minus):
                    config_path = work_dir / "predict_fd.ini"
                    render_inference_config(
                        config_path,
                        work_dir=work_dir,
                        trained_model_dir=model_dir,
                        python_interpreter=sys.executable,
                        task=[2, 3, 4],
                        with_grad=False,
                    )
                    _run_deeph_inference(deeph_cli=deeph_cli, config_path=config_path)
                plus_blocks = _read_h5_blocks(plus / "hamiltonians_pred.h5")
                minus_blocks = _read_h5_blocks(minus / "hamiltonians_pred.h5")
                errors = []
                refs = []
                for key, grad_block in grad_blocks.items():
                    if key not in plus_blocks or key not in minus_blocks:
                        continue
                    analytic = np.asarray(grad_block)[..., atom, axis]
                    fd = (plus_blocks[key] - minus_blocks[key]) / (2.0 * delta)
                    errors.append((analytic - fd).reshape(-1))
                    refs.append(analytic.reshape(-1))
                self.assertTrue(errors, "No common DeepH Hamiltonian blocks found between autograd and finite-difference outputs.")
                error = np.concatenate(errors)
                ref = np.concatenate(refs)
                compared += error.size
                grad_norm = max(grad_norm, float(np.linalg.norm(ref)))
                rel_errors.append(float(np.linalg.norm(error) / (np.linalg.norm(ref) + 1e-30)))

            self.assertGreater(compared, 0)
            self.assertGreater(grad_norm, 0.0)
            self.assertTrue(np.all(np.isfinite(rel_errors)), rel_errors)
            self.assertLess(min(rel_errors), 0.25, rel_errors)
            self.assertLessEqual(rel_errors[-1], rel_errors[0] * 1.2, rel_errors)
