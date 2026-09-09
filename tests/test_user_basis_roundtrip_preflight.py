from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Comparison/scripts"))

from user_basis_roundtrip_preflight import LIMITS, render_fdf  # noqa: E402


def test_user_basis_roundtrip_contract_is_frozen():
    source = """SystemName old
SystemLabel old
%block PAO.Basis
C 1
n=2 0 1
4.0
%endblock PAO.Basis
"""
    rendered = render_fdf(source)
    assert "%block PAO.Basis" not in rendered
    assert "User.Basis T" in rendered
    assert "SystemLabel n_qzp_d12_14_user_basis" in rendered
    assert LIMITS == {"radial_relative": 1e-10, "overlap_relative_spectral": 1e-12,
                      "hamiltonian_rms_ev": 1e-5, "total_energy_abs_ev": 1e-6,
                      "fermi_abs_ev": 1e-6, "density_relative_l2": 1e-6}
