from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "Comparison/scripts"))

from diagnose_nested_basis_central import _last_total_energy  # noqa: E402


def test_total_energy_parser_uses_last_scf_value():
    assert _last_total_energy("siesta: Etot = -1.0\nsiesta: Etot = -2.5\n") == -2.5
