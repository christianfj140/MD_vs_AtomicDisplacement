from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "Comparison/scripts"))

from run_nested_basis_sentinel import _production_run  # noqa: E402


def test_nested_sentinel_reuses_matching_prod_stencil():
    assert _production_run("C1_x", 0).name == "central"
    assert _production_run("C1_x", -2).as_posix().endswith("PROD-SZ/C1_x/m2")
    assert _production_run("C1_z", 2).as_posix().endswith("PROD-SZ/C1_z/p2")
