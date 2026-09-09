from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Comparison/scripts"))
from run_nested_basis_ladder import LIMITS  # noqa: E402
from run_qz_cardinality_sentinel import PARENT, QZ  # noqa: E402


def test_qz_sentinel_is_the_single_preregistered_cardinality_increment():
    assert (PARENT, QZ) == ("n_tzp_d12_14", "n_qzp_d12_14")
    assert LIMITS == {"rms_ev_per_ang": 0.005, "relative": 0.01, "max_ev_per_ang": 0.015}
