import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Comparison/scripts"))
from audit_gauge_derivative import gauge_residual
from audit_epc_backends import audit_runtime_manifests


def test_moving_gauge_sign_and_constant_overlap():
    vacuum = {j: 2 + .03*j*.005 for j in range(-2, 3)}
    overlap = {j: np.eye(2) for j in vacuum}
    np.testing.assert_allclose(gauge_residual(vacuum, overlap), 0, atol=1e-13)


def test_runtime_rejects_failed_and_incomplete_records(tmp_path):
    p = tmp_path / "runtime.json"
    payload = dict(schema="graph2mat_jvp_runtime_preflight_v1", generated_at="now",
                   requested_backend="cuda", effective_backend="cuda", device="cuda:0",
                   precision="float64", backend_preflight="failed", backend_fallback_reason=None)
    p.write_text(json.dumps(payload))
    assert not audit_runtime_manifests([p])[0]["valid"]
    payload["backend_preflight"] = "passed"
    p.write_text(json.dumps(payload))
    assert audit_runtime_manifests([p])[0]["valid"]
    payload["other_run"] = {"requested_backend": "cuda"}
    p.write_text(json.dumps(payload))
    assert not audit_runtime_manifests([p])[0]["valid"]
