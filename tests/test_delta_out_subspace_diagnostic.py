from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "Comparison/scripts"), str(ROOT / "shared")]


def test_mode_sum_is_invariant_to_a_constant_complement_change():
    from diagnose_delta_out_subspace import complement_selectors, decompose

    rng = np.random.default_rng(7)
    q, _ = np.linalg.qr(rng.normal(size=(5, 5)))
    s = q @ np.diag([1.7, 1.3, 0.9, 0.5, 0.2]) @ q.T
    k, right = rng.normal(size=(5, 5)), rng.normal(size=(5, 5))
    selector = np.eye(5)[:, :2]
    complement, _ = complement_selectors(selector)
    reference, *_ = decompose(s, k, right, selector, complement)
    transform = np.array([[1.1, 0.1, 0.0], [0.0, 0.9, 0.1], [0.0, 0.0, 1.2]])
    z = selector @ selector.T + complement @ transform @ complement.T
    changed, *_ = decompose(z.T @ s @ z, z.T @ k @ z, z.T @ right @ z,
                            selector, complement)
    assert np.allclose(changed, reference, rtol=1e-11, atol=1e-11)
