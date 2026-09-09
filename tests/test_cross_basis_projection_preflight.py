from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "Comparison/scripts"))

from cross_basis_projection_preflight import _replace_explicit_basis  # noqa: E402


def test_central_preflight_replaces_basis_without_creating_a_stencil():
    source = (REPO_ROOT / "Comparison/results/epc/convergence/gauge_delta/runs/equilibrium/RUN.fdf").read_text()
    rendered = _replace_explicit_basis(source, "tzp_es0p005", "TZP", 0.005)
    assert "%block PAO.Basis\n" not in rendered
    assert "PAO.BasisSize TZP" in rendered
    assert "PAO.EnergyShift 0.005000 Ry" in rendered
    assert "MD.NumCGsteps                    0" in rendered
    assert all(token not in rendered for token in ("+0.005", "-0.005", "FC.Save.dHS"))
