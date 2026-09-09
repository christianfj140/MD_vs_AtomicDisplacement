from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "Comparison/scripts"))

from nested_basis_identity_preflight import BASIS_BLOCKS, render_fdf  # noqa: E402


def test_nested_bases_keep_prod_cutoffs_and_only_run_central_geometry():
    source = (REPO_ROOT / "Comparison/results/epc/convergence/gauge_delta/runs/equilibrium/RUN.fdf").read_text()
    text = render_fdf(source, "nested", BASIS_BLOCKS["n_tzp_d10_12"])
    assert "n=2 0 3\n   4.088 0.0 0.0" in text
    assert "n=2 1 3 P 1\n   4.870 0.0 0.0" in text
    assert "n=3 0 1\n   10.000" in text and "n=3 1 1\n   12.000" in text
    assert "DM.Tolerance 1.d-6" in text
    assert all(token not in text for token in ("+0.005", "-0.005", "FC.Save.dHS"))
