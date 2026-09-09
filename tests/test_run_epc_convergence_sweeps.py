from pathlib import Path
import sys

SCRIPTS = Path(__file__).resolve().parents[1] / "Comparison/scripts"
sys.path.insert(0, str(SCRIPTS))

from run_epc_convergence_sweeps import materialize_variant  # noqa: E402


def test_variant_changes_only_requested_numerics_and_enables_potentials():
    source = (Path(__file__).resolve().parents[1] / "materials/graphene/RUN.fdf").read_text()
    result = materialize_variant(source, mesh=800, scf=1e-5, kgrid=24)
    assert "MeshCutoff             800.0 Ry" in result
    assert "DM.Tolerance           1.0e-05" in result
    assert "  24   0   0  0.0" in result and "   0  24   0  0.0" in result
    assert result.count("SaveTotalPotential") == 1
    assert result.count("SaveElectrostaticPotential") == 1
