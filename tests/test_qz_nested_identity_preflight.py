from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "Comparison/scripts"))

from qz_nested_identity_preflight import QZ_BLOCK  # noqa: E402


def test_qz_candidate_changes_only_valence_zeta_count():
    assert "n=2 0 4\n   4.088 0.0 0.0 0.0" in QZ_BLOCK
    assert "n=2 1 4 P 1\n   4.870 0.0 0.0 0.0" in QZ_BLOCK
    assert "n=3 0 1\n   12.000" in QZ_BLOCK and "n=3 1 1\n   14.000" in QZ_BLOCK
