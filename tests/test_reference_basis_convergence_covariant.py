from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "Comparison/scripts"))

from certify_reference_basis_convergence_covariant import LIMITS  # noqa: E402


def test_covariant_gate_keeps_raw_preregistered_thresholds():
    assert LIMITS == {"rms_ev_per_ang": 0.005, "relative": 0.01, "max_ev_per_ang": 0.015}
