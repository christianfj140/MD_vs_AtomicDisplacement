from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.sparse


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "Comparison" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import generate_siesta_overlap_only as overlap  # noqa: E402


def test_overlap_only_fdf_disables_scf_and_retains_periodic_blocks() -> None:
    source = REPO_ROOT / "materials/bilayer_graphene_hBN_AA/RUN.fdf"
    text = overlap.overlap_only_fdf(source, kgrid=3)
    assert "MaxSCFIterations                 0" in text
    assert "TS.onlyS                         T" in text
    assert "%block kgrid_Monkhorst_Pack\n  3  0  0  0.0\n  0  3  0  0.0" in text


def test_orbital_parser_handles_siesta_six_digit_column_overflow(tmp_path: Path) -> None:
    path = tmp_path / "large.ORB_INDX"
    path.write_text(
        """ 3 27 = orbitals in unit cell and supercell
    io    ia is spec iao n l m z p sym rc isc iuo
99999 1 1 C 1 2 0 0 1 F s 4.0 0 0 0 99999
100000 1 1 C 2 2 1 -1 1 F py 4.0 0 0 0100000
100001 1 1 C 3 2 1 0 1 F pz 4.0 0 0 0100001
""".replace("99999", "1", 2).replace("100000", "2", 1).replace("100001", "3", 1),
        encoding="utf-8",
    )
    rows = overlap._orbital_rows(path)
    assert len(rows) == 3
    assert [row["l"] for row in rows] == [0, 1, 1]


def test_deeph_orbital_map_preserves_shell_count_and_phase() -> None:
    rows = [
        {"io": 0, "atom": 0, "n": 2, "l": 0, "m": 0, "zeta": 1},
        {"io": 1, "atom": 0, "n": 2, "l": 1, "m": -1, "zeta": 1},
        {"io": 2, "atom": 0, "n": 2, "l": 1, "m": 0, "zeta": 1},
        {"io": 3, "atom": 0, "n": 2, "l": 1, "m": 1, "zeta": 1},
    ]
    mapping, signs, orbital_types = overlap._deeph_orbital_map(rows)
    assert sorted(mapping) == [0, 1, 2, 3]
    assert signs == [1, -1, 1, -1]
    assert orbital_types == [[0, 1]]


def test_sparse_extreme_eigenvalues_find_positive_bounds() -> None:
    matrix = scipy.sparse.diags([0.25, 1.0, 4.0], format="csr")
    minimum, maximum = overlap._sparse_extreme_eigenvalues(matrix)
    assert np.isclose(minimum, 0.25)
    assert np.isclose(maximum, 4.0)
