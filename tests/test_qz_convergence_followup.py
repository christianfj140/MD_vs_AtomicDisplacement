from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Comparison/scripts"))
from run_qz_convergence_followup import QZ_D10_BLOCK  # noqa: E402


def test_qz_d10_block_changes_only_diffuse_cutoffs():
    assert "10.000" in QZ_D10_BLOCK and "12.000" in QZ_D10_BLOCK and "14.000" not in QZ_D10_BLOCK
