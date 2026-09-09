from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Comparison/scripts"))
from audit_reference_basis_dense_k import _runs  # noqa: E402


def test_dense_audit_reuses_existing_qz_runs():
    assert _runs("n_qzp_d12_14", "C1_x")[0].name == "n_qzp_d12_14"
    assert _runs("n_qzp_d10_12", "C1_x")[1].name == "p1"
