from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "Comparison/scripts"))

from run_nested_basis_ladder import BASIS_ORDER, COMPARE  # noqa: E402


def test_ladder_gates_only_preregistered_final_increments():
    assert BASIS_ORDER == ("n_dzp", "n_tzp", "n_tzp_d10_12", "n_tzp_d12_14")
    assert COMPARE["cardinality"][2] is True
    assert COMPARE["diffuse_importance"][2] is False
    assert COMPARE["diffuse_range"][2] is True
