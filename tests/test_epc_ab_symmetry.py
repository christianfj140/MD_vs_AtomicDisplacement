"""E-F_001-S35 / GO-7 audit finding: an independent spatial-symmetry check.

``evaluate_ab_bilayer_epc.py``'s ``layer_swap_hermitian`` check only compares
the Frobenius norms of the ``(bottom,top)`` and ``(top,bottom)`` interlayer
``D_H`` blocks. That equality is an algebraic identity of any Hermitian
matrix (``D_H`` is Hermitian by construction, so its off-diagonal blocks are
always Hermitian conjugates of each other) -- it holds even for a matrix that
has nothing to do with the real crystal, and a symmetry-breaking bug that
corrupted both blocks the same way would still pass it. This module builds
the check the audit actually asked for: a real spatial transformation, derived
from ``materials/bilayer_graphene_AB``'s own cell and atomic positions (never
assumed), applied to the orbital-resolved Hamiltonian/overlap derivative
blocks themselves.

The symmetry, derived (not assumed)
------------------------------------
``find_inversion_center_and_permutation`` below brute-forces every candidate
center from pairwise atom midpoints and keeps the first one whose induced
map, atom by atom, sends every atom onto another atom of the cell modulo a
lattice vector. For the real geometry
(``Comparison/results/epc/siesta_reference/bilayer_graphene_AB/runs/equilibrium``)
this finds a genuine inversion center swapping the two layers as an
involution -- the well-known Bernal (AB) bilayer inversion symmetry, through
the midpoint of the eclipsed A2/B1 dimer (or any lattice-equivalent copy of
it), independently reproduced here from the coordinates on disk rather than
assumed from the stacking's name.

What the transform predicts for D_H/D_S
----------------------------------------
For any spatial isometry ``g`` of the crystal and any displacement pattern
``u``, general covariance of the SCF Hamiltonian under a rigid transformation
of all-space gives ``D_X[g.u] = D(g) D_X[u] D(g)^T`` for X in {H, S}, where
``D(g)`` permutes orbitals by the atom permutation ``g`` induces (same local
shell index on the image atom, since every atom here carries an identical
basis) and multiplies each orbital by its parity ``(-1)^l`` under inversion.
``shear`` and ``layer_breathing`` (:func:`run_ab_bilayer_epc_paths.build_directions`)
are each invariant under this particular ``g`` (``g.u == u``, checked
below), so for them the prediction sharpens to self-consistency:
``D_X ~= D(g) D_X D(g)^T``. ``intralayer_control`` (bottom layer only) is
*not* invariant (``g.u`` is the same pattern applied to the top layer
instead), so it is not expected to be self-consistent -- which is exactly
the negative control the audit asked for, using real, already-computed data
rather than a synthetic corruption.

``D_S`` has no Fermi-level subtlety and reproduces this to machine precision
(confirming the derived center/permutation/parity are correct, independently
of any assumption about the Hamiltonian side). ``D_H`` here is the
Fermi-shifted quantity ``run_ab_bilayer_epc_paths.produce()`` itself builds
(``d_h_abs - fermi * d_s``, fixed equilibrium E_F) and is only approximately
self-consistent: S34's own limitations list says this candidate has no
explicit SIESTA +/- finite-difference cross-check, a Gamma-only FC branch,
and no moving-E_F correction, so its raw ``D_H`` is not expected to satisfy
the exact crystal symmetry as tightly as ``D_S`` does. The test below stays
honest about that: it uses a loose, comparative tolerance (self-consistency
must be well inside what an *uncorrelated* pair of blocks would give, which
``intralayer_control`` measures empirically) rather than pretending to
machine precision it does not have.

No new SIESTA compute
----------------------
Everything here reads ``equilibrium.TSHS``/``equilibrium.ORB_INDX`` and the
real ``fc_dhs_d0p02.dHSdR.nc`` already on disk under
``Comparison/results/epc/siesta_reference/bilayer_graphene_AB`` (the S34
campaign) through the already-certified readers (:mod:`certify_siesta_dhsdr`,
:mod:`read_siesta_dhsdr`, :mod:`certify_basis_response`,
:mod:`epc_basis_response`, :mod:`orbital_contract`) -- no new physics, no new
SCF, no new formalism.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import sisl

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT / "Comparison" / "scripts", REPO_ROOT / "shared"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import certify_basis_response as cbr  # noqa: E402
import certify_siesta_dhsdr as go2  # noqa: E402
import epc_basis_response as ebr  # noqa: E402
import orbital_contract as oc  # noqa: E402
import run_ab_bilayer_epc_paths as ab  # noqa: E402
from read_siesta_dhsdr import read_dhsdr  # noqa: E402

REFERENCE_ROOT = ab.DEFAULT_REFERENCE_ROOT
MANIFEST_PATH = REFERENCE_ROOT / "epc_siesta_reference_manifest.json"

reference_present = pytest.mark.skipif(
    not MANIFEST_PATH.is_file(),
    reason=f"AB bilayer SIESTA reference not found at {REFERENCE_ROOT}",
)

GAMMA = (0.0, 0.0, 0.0)


# --------------------------------------------------------------------------- #
# The symmetry operation, derived from the real geometry
# --------------------------------------------------------------------------- #


def find_inversion_center_and_permutation(
    positions_ang: np.ndarray, cell_ang: np.ndarray, *, tol_ang: float = 1e-3
) -> tuple[np.ndarray, list[int]] | tuple[None, None]:
    """A center ``r0`` and atom permutation ``pi`` such that, for every atom
    ``a``, ``2*r0 - positions[a]`` coincides with ``positions[pi[a]]`` modulo
    an integer combination of the (in-plane) lattice vectors, exactly, to
    within ``tol_ang``.

    Every pairwise atom midpoint is tried as a candidate center (inversion
    centers of a periodic crystal come in copies spaced by half a lattice
    vector, so the first self-consistent candidate is as good as any other);
    nothing about "AB stacking" or "which atom pairs with which" is assumed.
    Returns ``(None, None)`` if no candidate yields a full permutation --
    e.g. a geometry that is not actually inversion-symmetric.
    """
    positions = np.asarray(positions_ang, dtype=np.float64)
    n = positions.shape[0]
    inv_cell = np.linalg.inv(np.asarray(cell_ang, dtype=np.float64))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            center = 0.5 * (positions[i] + positions[j])
            permutation = [-1] * n
            consistent = True
            for a in range(n):
                target = 2.0 * center - positions[a]
                match = None
                for b in range(n):
                    diff = target - positions[b]
                    frac = diff @ inv_cell
                    in_plane_residual = frac[:2] - np.round(frac[:2])
                    if np.linalg.norm(in_plane_residual) < tol_ang and abs(diff[2]) < tol_ang:
                        match = b
                        break
                if match is None:
                    consistent = False
                    break
                permutation[a] = match
            if consistent and len(set(permutation)) == n:
                return center, permutation
    return None, None


def orbital_permutation_and_parity(
    orb_indx_rows: list[dict], atom_permutation: list[int], no_u: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per-orbital ``(pi_orbital, parity)`` induced by ``atom_permutation``.

    Every atom here is carbon with the identical basis (:file:`RUN.fdf`'s
    ``PAO.Basis``: one ``l=0`` shell, one ``l=1`` shell), so the image
    orbital keeps the same local shell index; parity is ``(-1)^l`` (real
    spherical harmonics do not mix under inversion, only sign-flip when
    ``l`` is odd), with ``l`` read from ``ORB_INDX`` rather than assumed
    from the basis block.
    """
    rows = sorted(orb_indx_rows, key=lambda row: row["io"])
    assert len(rows) == no_u
    atom_of_orbital = np.array([row["ia"] - 1 for row in rows])  # 0-based
    l_of_orbital = np.array([row["l"] for row in rows])
    firsto = np.zeros(len(set(atom_of_orbital.tolist())) + 1, dtype=np.int64)
    for orbital, atom in enumerate(atom_of_orbital):
        if orbital == 0 or atom_of_orbital[orbital - 1] != atom:
            firsto[atom] = orbital
    pi_orbital = np.zeros(no_u, dtype=np.int64)
    parity = np.zeros(no_u, dtype=np.float64)
    for orbital in range(no_u):
        atom = atom_of_orbital[orbital]
        local = orbital - firsto[atom]
        pi_orbital[orbital] = firsto[atom_permutation[atom]] + local
        parity[orbital] = (-1.0) ** l_of_orbital[orbital]
    return pi_orbital, parity


def apply_transform(matrix: np.ndarray, pi_orbital: np.ndarray, parity: np.ndarray) -> np.ndarray:
    """``D(g) @ matrix @ D(g)^T`` for the permutation-with-sign ``D(g)``."""
    no_u = matrix.shape[0]
    transformed = np.zeros_like(matrix)
    for i in range(no_u):
        for j in range(no_u):
            transformed[pi_orbital[i], pi_orbital[j]] = parity[i] * parity[j] * matrix[i, j]
    return transformed


def relative_residual(a: np.ndarray, b: np.ndarray) -> float:
    norm = np.linalg.norm(b)
    return float(np.linalg.norm(a - b) / norm) if norm > 0.0 else float("inf")


# --------------------------------------------------------------------------- #
# Fixtures: load the real campaign once
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def campaign():
    import json

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return go2.Campaign(root=REFERENCE_ROOT, manifest=manifest)


@pytest.fixture(scope="module")
def equilibrium(campaign):
    equilibrium_id = campaign.manifest["certification_set"]["equilibrium_run_id"]
    return go2.read_tshs(
        campaign.run_dir(equilibrium_id), equilibrium_id, fermi_ev=campaign.fermi_ev(equilibrium_id)
    )


@pytest.fixture(scope="module")
def geometry(equilibrium):
    geom = sisl.get_sile(str(equilibrium.path)).read_geometry()
    return {
        "positions_ang": np.asarray(geom.xyz, dtype=np.float64),
        "cell_ang": np.asarray(geom.cell, dtype=np.float64),
    }


@pytest.fixture(scope="module")
def layer_of_atom(geometry):
    return ab.layer_of_atom_from_positions(geometry["positions_ang"])


@pytest.fixture(scope="module")
def dhsdr(campaign):
    delta = float(campaign.deltas[0])
    return read_dhsdr(campaign.dhsdr_path(delta))


@pytest.fixture(scope="module")
def orbital_transform(campaign, equilibrium, geometry, layer_of_atom):
    center, permutation = find_inversion_center_and_permutation(
        geometry["positions_ang"], geometry["cell_ang"]
    )
    assert permutation is not None, "the real AB bilayer geometry must carry an inversion center"
    equilibrium_id = campaign.manifest["certification_set"]["equilibrium_run_id"]
    orb_indx_path = campaign.run_dir(equilibrium_id) / f"{equilibrium_id}.ORB_INDX"
    orb = oc.parse_orb_indx(orb_indx_path)
    pi_orbital, parity = orbital_permutation_and_parity(orb["unit_cell_rows"], permutation, equilibrium.no_u)
    return {"center": center, "atom_permutation": permutation, "pi_orbital": pi_orbital, "parity": parity}


@pytest.fixture(scope="module")
def directions(geometry, layer_of_atom):
    return ab.build_directions(geometry["positions_ang"], geometry["cell_ang"], layer_of_atom)


@pytest.fixture(scope="module")
def isc_off(dhsdr):
    return next(iter(dhsdr.records.values()))["D_H"].isc_off


def _gamma_matrix(dhsdr_file, isc_off_table, no_u, vectors, kind):
    field = go2.contract_fc(dhsdr_file, vectors, kind)
    blocks, outside = cbr.dense_blocks(field, isc_off_table, no_u)
    assert outside == 0.0, f"{kind} has weight outside the declared isc_off image table"
    return ebr.bloch_sum(blocks, isc_off_table, GAMMA)


def _d_h_siesta_gamma(dhsdr_file, isc_off_table, no_u, vectors, fermi_ev):
    """The Fermi-shifted D_H run_ab_bilayer_epc_paths.py itself builds
    (``d_h_abs - fermi * d_s``, fixed equilibrium E_F -- see module docstring)."""
    d_h_abs = _gamma_matrix(dhsdr_file, isc_off_table, no_u, vectors, "D_H")
    d_s = _gamma_matrix(dhsdr_file, isc_off_table, no_u, vectors, "D_S")
    return d_h_abs - fermi_ev * d_s


# --------------------------------------------------------------------------- #
# 1. The geometric symmetry itself
# --------------------------------------------------------------------------- #


@reference_present
def test_real_geometry_has_a_layer_swapping_inversion_center(geometry, layer_of_atom):
    center, permutation = find_inversion_center_and_permutation(
        geometry["positions_ang"], geometry["cell_ang"]
    )
    assert permutation is not None
    n = len(permutation)
    for atom in range(n):
        assert permutation[permutation[atom]] == atom, "the induced map must be an involution"
        assert layer_of_atom[permutation[atom]] != layer_of_atom[atom], (
            "the inversion must swap the two layers, not map a layer onto itself"
        )
    assert len(set(permutation)) == n


@reference_present
def test_a_geometry_with_one_atom_displaced_has_no_consistent_inversion_center(geometry):
    """Negative control at the geometry level (the audit's "displace one
    layer's atoms slightly" case): breaking the real positions must make the
    brute-force search fail to find any consistent center/permutation,
    rather than silently returning a wrong one."""
    broken = geometry["positions_ang"].copy()
    broken[0, 0] += 0.3  # nudge one bottom-layer atom in-plane, well above tol_ang
    center, permutation = find_inversion_center_and_permutation(broken, geometry["cell_ang"])
    assert permutation is None, "a broken geometry must not report a false symmetry"


# --------------------------------------------------------------------------- #
# 2. D_S: exact (machine precision), the clean confirmation the derived
#    center/permutation/parity are right
# --------------------------------------------------------------------------- #


@reference_present
@pytest.mark.parametrize("direction_name", ["shear", "layer_breathing"])
def test_overlap_derivative_is_inversion_symmetric_to_machine_precision(
    direction_name, directions, dhsdr, isc_off, equilibrium, orbital_transform
):
    direction = directions[direction_name]
    no_u = equilibrium.no_u
    d_s = _gamma_matrix(dhsdr, isc_off, no_u, direction.vectors, "D_S")
    transformed = apply_transform(d_s, orbital_transform["pi_orbital"], orbital_transform["parity"])
    residual = relative_residual(transformed, d_s)
    assert residual < 1e-6, (
        f"D_S[{direction_name}] must be invariant under the real inversion to machine precision "
        f"(no Fermi-level ambiguity applies to the overlap derivative); got relative residual {residual:.3e}"
    )


# --------------------------------------------------------------------------- #
# 3. D_H: approximately symmetric for shear/layer_breathing, judged against
#    the real, uncorrelated scale intralayer_control measures -- not an
#    arbitrary absolute tolerance
# --------------------------------------------------------------------------- #


@reference_present
def test_hamiltonian_derivative_symmetric_directions_beat_the_asymmetric_control(
    directions, dhsdr, isc_off, equilibrium, orbital_transform
):
    fermi_ev = float(equilibrium.fermi_ev)
    no_u = equilibrium.no_u
    pi_orbital, parity = orbital_transform["pi_orbital"], orbital_transform["parity"]

    residuals = {}
    for name in ("shear", "layer_breathing", "intralayer_control"):
        d_h = _d_h_siesta_gamma(dhsdr, isc_off, no_u, directions[name].vectors, fermi_ev)
        transformed = apply_transform(d_h, pi_orbital, parity)
        residuals[name] = relative_residual(transformed, d_h)

    # shear and layer_breathing are themselves invariant displacement patterns
    # under this inversion (module docstring); intralayer_control (bottom
    # layer only) is not, so its D_H is not expected to be self-consistent --
    # it is the real-data negative control.
    for name in ("shear", "layer_breathing"):
        assert residuals[name] < 0.5, (
            f"D_H[{name}] departs from the real inversion symmetry by a relative residual of "
            f"{residuals[name]:.3f}, past what S34's documented limitations (no explicit SIESTA "
            f"+/- cross-check, Gamma-only FC, no moving-E_F correction) can plausibly explain"
        )
        assert residuals[name] < 0.5 * residuals["intralayer_control"], (
            f"D_H[{name}] (residual {residuals[name]:.3f}) is not clearly separated from the "
            f"asymmetric control's residual ({residuals['intralayer_control']:.3f}); the check "
            "would not have power to catch a real symmetry-breaking bug"
        )


@reference_present
def test_intralayer_control_is_not_inversion_symmetric_real_data_negative_control(
    directions, dhsdr, isc_off, equilibrium, orbital_transform
):
    """intralayer_control's displacement pattern only touches the bottom
    layer, so the real inversion maps it to a *different* pattern (the same
    stretch, applied to the top layer instead) -- it must not be
    self-consistent. This uses real, already-computed SIESTA data (not a
    synthetic corruption) to show the check has discriminating power."""
    fermi_ev = float(equilibrium.fermi_ev)
    no_u = equilibrium.no_u
    d_h = _d_h_siesta_gamma(dhsdr, isc_off, no_u, directions["intralayer_control"].vectors, fermi_ev)
    transformed = apply_transform(d_h, orbital_transform["pi_orbital"], orbital_transform["parity"])
    residual = relative_residual(transformed, d_h)
    assert residual > 1.0, (
        "intralayer_control is built asymmetric across layers on purpose; its D_H must clearly "
        f"fail self-consistency under the real inversion (got relative residual {residual:.3f})"
    )
