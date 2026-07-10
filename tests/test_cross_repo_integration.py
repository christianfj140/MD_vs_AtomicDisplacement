"""Fase 16.4 (audit): cheap cross-repo integration (MD <-> DeepH-pack, no mocks).

Skips cleanly when the sibling DeepH-pack venv is not available.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "Comparison" / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "shared"))

DEEPH_PYTHON = REPO_ROOT.parent / "DeepH-pack" / ".venv" / "bin" / "python"

needs_deeph = pytest.mark.skipif(
    not DEEPH_PYTHON.exists(), reason="DeepH-pack venv not available"
)


@needs_deeph
def test_real_deeph_backend_passes_capability_preflight():
    import run_deeph_autograd_derivative_predictions as rd

    capability = rd.deeph_autograd_capability_preflight(str(DEEPH_PYTHON))
    assert capability["available"] is True
    assert capability["implementation"] == "torch_forward_ad_jvp"
    assert capability["output_schema"] == "hamiltonians_grad_pred_v2"
    assert capability["jvp_smoke"]["passed"] is True


@needs_deeph
def test_real_deeph_import_matches_inspected_checkout():
    from run_inventory import collect_run_inventory

    inventory = collect_run_inventory(deeph_python=str(DEEPH_PYTHON))
    deeph_import = inventory["imports"]["deeph"]
    assert deeph_import["module_path"], deeph_import
    assert deeph_import["matches_inspected_repo"] is True


@needs_deeph
def test_synthetic_grad_h5_with_v2_schema_is_consumable(tmp_path):
    """DeepH writes a v2 grad payload; the MD runner-side validation accepts the
    computed direction and rejects the NaN-sentinel (uncomputed) one."""
    script = r"""
import json, sys
import numpy as np
from deeph import write_ham_h5

grad = {"[0, 0, 0, 1, 1]": np.full((2, 2, 2, 3), np.nan)}
grad["[0, 0, 0, 1, 1]"][..., 0, 1] = 1.5  # only (atom 0, axis 1) computed
write_ham_h5(grad, path=sys.argv[1] + "/hamiltonians_grad_pred.h5")
json.dump({
    "isspinful": False,
    "hamiltonians_grad_pred_schema": "hamiltonians_grad_pred_v2",
    "grad_computed_atom_indices": [0],
    "grad_computed_axis_indices": [1],
    "grad_uncomputed_sentinel": "nan",
}, open(sys.argv[1] + "/info.json", "w"))
"""
    completed = subprocess.run(
        [str(DEEPH_PYTHON), "-c", script, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stderr

    import h5py

    import run_deeph_autograd_derivative_predictions as rd

    with h5py.File(tmp_path / "hamiltonians_grad_pred.h5", "r") as handle:
        block = np.asarray(handle["[0, 0, 0, 1, 1]"])
    computed = sparse.csr_matrix(rd._select_gradient_block(block, 0, 1))
    rd._validate_requested_direction(computed, work_dir=tmp_path, atom_index=0, axis_index=1)

    uncomputed = sparse.csr_matrix(np.nan_to_num(block[..., 1, 0], nan=np.nan))
    with pytest.raises(rd.DeepHAutogradDerivativePredictionError):
        rd._validate_requested_direction(
            sparse.csr_matrix(block[..., 1, 0]), work_dir=tmp_path, atom_index=1, axis_index=0
        )


@needs_deeph
def test_signature_covers_cross_repo_code_state(tmp_path):
    from artifact_signature import input_signature_sha256
    from run_inventory import collect_run_inventory

    inventory = collect_run_inventory(deeph_python=str(DEEPH_PYTHON))
    commits = {k: v.get("commit") for k, v in inventory["repositories"].items()}
    base = {"model": "deeph", "repository_commits": commits, "atom_index": 0, "axis_index": 0}
    sig = input_signature_sha256(base)
    other = input_signature_sha256({**base, "repository_commits": {**commits, "DeepH-pack": "0" * 40}})
    assert sig != other  # changing the imported code invalidates the cache
