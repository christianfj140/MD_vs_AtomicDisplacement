from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "Comparison/scripts"))

from run_displaced_projection_sentinel import _render_fdf  # noqa: E402


def test_sentinel_fdf_moves_only_c1_and_freezes_the_contract():
    source = (REPO_ROOT / "Comparison/results/epc/convergence/gauge_delta/runs/equilibrium/RUN.fdf").read_text()
    xyz = np.array([[1.467, 0.0, 0.01], [2.934, 0.0, 0.0]])
    text = _render_fdf(source, "sentinel", xyz, large_basis=True)
    assert "PAO.BasisSize TZP" in text and "PAO.EnergyShift 0.005000 Ry" in text
    assert "MeshCutoff             600.0 Ry" in text
    assert "DM.Tolerance 1.d-6" in text
    assert "1.467000000000  0.000000000000  0.010000000000" in text
    assert "2.934000000000  0.000000000000  0.000000000000" in text
