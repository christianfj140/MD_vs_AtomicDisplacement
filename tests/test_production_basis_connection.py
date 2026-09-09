from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "Comparison/scripts"))

from certify_production_basis_connection import H_VALUES, LIMIT_EV_PER_ANG  # noqa: E402


def test_connection_preregistration_is_frozen():
    assert H_VALUES == (0.010, 0.005, 0.0025, 0.00125)
    assert LIMIT_EV_PER_ANG == 0.001
