from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Comparison/scripts"))

from ghost_d_augmentation_preflight import (  # noqa: E402
    GHOST_CUTOFF_BOHR, GHOST_LABEL, SCHUR_REL_MIN, generation_failure_report,
    render_ghost_fdf,
)


def test_ghost_d_contract_and_colocation_are_frozen():
    source = """SystemName old
SystemLabel old
NumberOfAtoms 2
NumberOfSpecies 2
%block ChemicalSpeciesLabel
1 6 C
2 -1 Ghost-H
%endblock ChemicalSpeciesLabel
%block AtomicCoordinatesAndAtomicSpecies
0 0 0 1
1 0 0 1
%endblock AtomicCoordinatesAndAtomicSpecies
%block PAO.Basis
C 1
n=2 0 1
4.0
%endblock PAO.Basis
"""
    rendered = render_ghost_fdf(source, np.array([[0., 0., 0.], [1., 0., 0.]]))
    assert "NumberOfAtoms 4" in rendered and "NumberOfSpecies 3" in rendered
    assert f"3  -206  {GHOST_LABEL}  C.psf" in rendered
    assert "%block SyntheticAtoms" in rendered and "0.0 0.0 0.0 0.0" in rendered
    assert rendered.count("# co-moving auxiliary d") == 2
    assert f"{GHOST_CUTOFF_BOHR:.3f}" in rendered
    assert SCHUR_REL_MIN == 1e-6


def test_generation_failure_stops_before_probe(tmp_path):
    report = generation_failure_report(tmp_path, tmp_path / "run", "synthetic species not detailed")
    assert report["verdict"] == "NO_GO"
    assert [row["atomic_number"] for row in report["attempts"]] == [-6, -100, -206]
    assert "no TSHS" in report["claim_scope"]
