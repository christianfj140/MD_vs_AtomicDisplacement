from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "Comparison/scripts"), str(ROOT / "shared")]


def test_embedded_modal_identity_on_a_small_positive_metric():
    from certify_delta_out_closure import delta_out
    from diagnose_delta_out_subspace import complement_selectors, decompose

    rng = np.random.default_rng(4)
    q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    s = q @ np.diag([2.0, 1.2, 0.7, 0.3]) @ q.T
    k, right, derivative = (rng.normal(size=(4, 4)) for _ in range(3))
    k, derivative = (k + k.T) / 2, (derivative + derivative.T) / 2
    selector = np.eye(4)[:, :2]
    complement, _ = complement_selectors(selector)
    left = right.T
    s_a = selector.T @ s @ selector
    embedded, *_ = delta_out(derivative, k, s, left, right, selector, s_a,
                             selector.T @ left @ selector, selector.T @ right @ selector)
    modal, *_ = decompose(s, k, right, selector, complement)
    assert np.allclose(embedded, modal, rtol=1e-11, atol=1e-11)
