from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Comparison/scripts"))
from certify_full_basis_connection import CASES  # noqa: E402


def test_full_connection_uses_only_existing_stencils():
    assert CASES == (("n_tzp_d12_14", "C1_x"), ("n_tzp_d12_14", "C1_z"),
                     ("n_qzp_d12_14", "C1_x"), ("n_qzp_d12_14", "C1_z"),
                     ("n_qzp_d10_12", "C1_x"))
