from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Comparison/scripts"))
from certify_reference_basis_convergence import _verdict  # noqa: E402


def test_reference_gate_fails_closed():
    assert _verdict({"a": {"verdict": "PASS"}, "b": {"verdict": "NO_GO"}}) == "NO_GO"
    assert _verdict({"a": {"verdict": "PASS"}}) == "PASS"
