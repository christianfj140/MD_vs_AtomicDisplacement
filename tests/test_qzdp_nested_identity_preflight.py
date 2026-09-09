from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Comparison/scripts"))

from qzdp_nested_identity_preflight import (  # noqa: E402
    QZDP_BLOCK, SPLIT_NORM_ATTEMPTS, generation_failure_report,
)


def test_qzdp_requests_exactly_two_polarization_zetas():
    assert "n=2 1 4 P 2 S 0.99" in QZDP_BLOCK
    assert "n=2 0 4" in QZDP_BLOCK
    assert "12.000" in QZDP_BLOCK and "14.000" in QZDP_BLOCK


def test_generation_failure_is_fail_closed(tmp_path):
    report = generation_failure_report(tmp_path, tmp_path / "run", "POLgen failed")
    assert report["verdict"] == "NO_GO"
    assert report["split_norm_attempts"] == list(SPLIT_NORM_ATTEMPTS)
    assert "no TSHS" in report["claim_scope"]
