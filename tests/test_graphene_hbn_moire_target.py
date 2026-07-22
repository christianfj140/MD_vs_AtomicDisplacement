"""Graphene-hBN twisted moire target geometry builder."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "Comparison" / "scripts"
SHARED = REPO_ROOT / "shared"
for path in (SCRIPTS, SHARED):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import build_graphene_hbn_moire_target as moire  # noqa: E402
from fdf_materialization import extract_fdf_structure  # noqa: E402


STACKING_FDF = REPO_ROOT / "materials" / "graphene_hBN_AA" / "RUN.fdf"


def test_commensurate_angle_matches_known_values() -> None:
    # (1,2) -> ~21.79 deg is the standard first commensurate hexagonal twist.
    assert math.isclose(moire.commensurate_angle_degrees(1, 2), 21.7867892, abs_tol=1e-4)
    # (1,3) -> ~32.2 deg.
    assert math.isclose(moire.commensurate_angle_degrees(1, 3), 32.2042469, abs_tol=1e-4)
    with pytest.raises(RuntimeError):
        moire.commensurate_angle_degrees(2, 2)


def test_moire_geometry_atom_count_and_species(tmp_path: Path) -> None:
    text, metadata = moire.moire_geometry(STACKING_FDF, approximant=2, m=1, n=2)
    assert metadata["num_atoms"] == 4 * 2 * 2 == 16
    assert metadata["twisted_sublayer"] == "hBN"
    assert metadata["graphene_hBN_lattice_mismatch_percent"] == 1.8
    out = tmp_path / "RUN.fdf"
    out.write_text(text, encoding="utf-8")
    parsed = extract_fdf_structure(out, structure_type="crystal")
    assert parsed.atom_count == 16
    labels = {sp.label for sp in parsed.species}
    assert labels == {"C", "B", "N"}
    # 3x3 approximant -> 36 atoms, N >> 4.
    _, meta3 = moire.moire_geometry(STACKING_FDF, approximant=3, m=1, n=2)
    assert meta3["num_atoms"] == 36
    assert "MD.TypeOfRun" not in text and "Lua.Script" not in text


def test_kgrid_is_divided_by_approximant_to_match_density(tmp_path: Path) -> None:
    # p x p supercell must divide the two in-plane MP counts by p (keep k_z=1).
    text, _ = moire.moire_geometry(STACKING_FDF, approximant=2, m=1, n=2)
    block = text[text.index("%block kgrid_Monkhorst_Pack"):text.index("%endblock kgrid_Monkhorst_Pack")]
    rows = [line.split() for line in block.splitlines() if line.split() and line.split()[0].lstrip("-").isdigit()]
    # stacking is 20x20x1 -> moire (approximant 2) must be 10x10x1.
    assert int(rows[0][0]) == 10
    assert int(rows[1][1]) == 10
    assert int(rows[2][2]) == 1


def test_dry_run_never_invokes_siesta(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(moire, "run_siesta", lambda *_a, **_k: pytest.fail("SIESTA invoked"))
    plan = moire.dry_run_plan(STACKING_FDF, approximant=2, m=1, n=2, limit=1)
    assert plan["siesta_invoked"] is False
    assert plan["num_atoms"] == 16
    assert math.isclose(plan["twist_angle_deg"], 21.7867892, abs_tol=1e-4)


def test_moire_basis_hashes_match_bilayer_train_presets() -> None:
    # The moire must share basis with the flat stackings for the planner to accept it.
    from material_presets import resolve_material_bundle

    stacking = resolve_material_bundle({"material": {"preset": "graphene_hBN_AA"}}).validated
    for basis in (moire.COMMON_MATERIAL_ROOT / "basis").glob("*.ion.xml"):
        assert basis.name in stacking.basis_file_sha256
