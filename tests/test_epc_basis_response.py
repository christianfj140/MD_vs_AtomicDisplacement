"""C14B / E-F_001-S16: the basis-response provider.

Two things are under test. First the physics of the SIESTA split: a synthetic
``S_L``/``S_R`` pair obeying the exact identity ``S_L(R_l)_{mu,nu} =
S_R(-R_l)_{nu,mu}`` is folded into per-atom ``dS/dR`` exactly the way a moving
atom folds it, and the provider has to unfold it again — including the fact
that same-atom blocks cancel out of ``dS/dR`` entirely.

Second the contract: which representation satisfies which ``formalism_id``,
that a ``D_S`` is never split back into halves, that a Gamma response cannot be
relabelled at finite ``q``, and that a Graph2Mat ``D_H`` with a SIESTA overlap
response signs itself ``hybrid``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "Comparison" / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "shared"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from artifact_signature import EPC_BACKEND_CLASS_HYBRID, epc_artifact_node  # noqa: E402
from epc_basis_response import (  # noqa: E402
    GAMMA,
    METHOD_DHSDR_INTER_ATOMIC,
    REPRESENTATION_COVARIANT,
    BasisResponseArtifact,
    BasisResponseError,
    ResponseContext,
    basis_response_contract,
    from_covariant_connection,
    from_dhsdr,
    from_D_S,
    from_S_L_S_R,
    inter_atomic_split,
)
from epc_formalism import (  # noqa: E402
    FORMALISM_ID,
    FORMALISM_ID_IGNORE_OVERLAP,
    FORMALISM_ID_SYMMETRIC,
    REPRESENTATION_D_S,
    REPRESENTATION_S_L_S_R,
    EpcFormalismError,
    epc_matrix_elements,
)
from fd_perturbation_space import collective, one_hot  # noqa: E402

# Two atoms, two orbitals each, three images with +R and -R both present.
ISC = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]], dtype=np.int32)
ORBITAL_ATOM = np.array([1, 1, 2, 2])
NO_U = ORBITAL_ATOM.size
N_S = ISC.shape[0]
MINUS = {0: 0, 1: 2, 2: 1}  # index of -R_l in ISC


def _context(**overrides):
    base = {
        "geometry_signature": "geo-sha",
        "basis_signature": "orbital-contract-sha",
        "electronic_derivative_backend": "graph2mat",
        "direction": collective(np.eye(2, 3) / np.sqrt(2.0), name="collective_x"),
    }
    return ResponseContext(**{**base, **overrides})


def _true_split(seed: int = 20260902):
    """``S_L^alpha``/``S_R^alpha`` per axis, inter-atomic only, exact identity held."""
    rng = np.random.default_rng(seed)
    inter = ORBITAL_ATOM[:, None] != ORBITAL_ATOM[None, :]
    left = rng.normal(size=(3, N_S, NO_U, NO_U)) * inter
    right = np.empty_like(left)
    for axis in range(3):
        for image in range(N_S):
            right[axis, image] = left[axis, MINUS[image]].T
    return left, right


def _ds_records(left, right):
    """``dS/dR_{I,alpha}``: rows on ``I`` carry ``S_L``, columns on ``I`` carry ``S_R``."""
    records = {}
    for atom in (1, 2):
        rows = (ORBITAL_ATOM == atom)[:, None]
        cols = (ORBITAL_ATOM == atom)[None, :]
        for axis in (1, 2, 3):
            records[(atom, axis)] = left[axis - 1] * rows + right[axis - 1] * cols
    return records


# ---------------------------------------------------------------------------
# The split
# ---------------------------------------------------------------------------


def test_split_recovers_S_L_and_S_R_from_atom_resolved_dS():
    left, right = _true_split()
    direction = collective(np.array([[0.3, -0.4, 0.1], [0.2, 0.5, -0.6]]), name="v")
    v = direction.vectors

    split = inter_atomic_split(_ds_records(left, right), v, ORBITAL_ATOM)

    expected_left = sum(
        v[ORBITAL_ATOM - 1, axis][:, None] * left[axis] for axis in range(3)
    )
    expected_right = sum(
        v[ORBITAL_ATOM - 1, axis][None, :] * right[axis] for axis in range(3)
    )
    assert np.allclose(split["S_left"], expected_left, atol=1e-12)
    assert np.allclose(split["S_right"], expected_right, atol=1e-12)
    assert split["intra_atomic_residual"] == 0.0


def test_split_is_not_a_reconstruction_from_the_sum():
    """``S_L`` and ``S_R`` differ; their sum is what ``dS/dR`` alone would give."""
    left, right = _true_split()
    v = np.array([[0.3, -0.4, 0.1], [0.2, 0.5, -0.6]])
    split = inter_atomic_split(_ds_records(left, right), v, ORBITAL_ATOM)
    antisymmetric = 0.5 * (split["S_left"] - split["S_right"])
    assert np.linalg.norm(antisymmetric) > 0.1 * np.linalg.norm(split["S_left"])


def test_S_L_is_the_hermitian_conjugate_of_S_R_at_every_k():
    left, right = _true_split()
    direction = one_hot(2, 0, "y")
    split = inter_atomic_split(_ds_records(left, right), direction.vectors, ORBITAL_ATOM)
    artifact = from_S_L_S_R(
        split["S_left"],
        split["S_right"],
        _context(direction=direction),
        backend="siesta",
        method=METHOD_DHSDR_INTER_ATOMIC,
        intra_atomic_included=False,
        isc_off=ISC,
    )
    for k in (GAMMA, (0.25, 0.0, 0.0), (0.13, 0.41, -0.2)):
        response = artifact.at_k(k)
        assert np.allclose(response.S_left, response.S_right.conj().T, atol=1e-12)


def test_missing_record_for_a_displaced_atom_is_refused():
    left, right = _true_split()
    records = _ds_records(left, right)
    records.pop((2, 1))
    with pytest.raises(BasisResponseError, match="atom 2 along axis 1"):
        inter_atomic_split(records, np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), ORBITAL_ATOM)


# ---------------------------------------------------------------------------
# Which representation satisfies which formalism
# ---------------------------------------------------------------------------


def test_D_S_response_is_refused_by_the_production_formalism():
    with pytest.raises(BasisResponseError, match="not a supported operation"):
        from_D_S(np.eye(NO_U), _context(), backend="siesta_frozen_fd", method="central_fd_of_S")


def test_D_S_response_serves_the_degraded_formalism_and_says_so():
    artifact = from_D_S(
        np.eye(NO_U),
        _context(formalism_id=FORMALISM_ID_SYMMETRIC),
        backend="siesta_frozen_fd",
        method="central_fd_of_S",
    )
    assert artifact.intra_atomic_included is False
    response = artifact.at_k()
    assert response.representation == REPRESENTATION_D_S
    with pytest.raises(EpcFormalismError):  # no S_L/S_R to be had, by construction
        _ = response.antisymmetric


def test_ignore_overlap_formalism_takes_no_basis_response():
    with pytest.raises(BasisResponseError, match="control negative"):
        from_S_L_S_R(
            np.eye(NO_U),
            np.eye(NO_U),
            _context(formalism_id=FORMALISM_ID_IGNORE_OVERLAP),
            backend="siesta",
            method="whatever",
            intra_atomic_included=True,
        )


def test_covariant_connection_is_accepted_and_equals_the_pair():
    rng = np.random.default_rng(7)
    a = rng.normal(size=(NO_U, NO_U))
    s = a @ a.T + NO_U * np.eye(NO_U)
    gamma = rng.normal(size=(NO_U, NO_U))
    artifact = from_covariant_connection(
        gamma, s, _context(), backend="analytic_pao", method="Gamma_from_radial_derivatives",
        intra_atomic_included=True,
    )
    assert artifact.representation == REPRESENTATION_COVARIANT
    response = artifact.at_k()
    assert response.representation == REPRESENTATION_S_L_S_R
    assert np.allclose(response.S_right, s @ gamma)
    assert np.allclose(response.S_left, gamma.conj().T @ s)
    assert response.intra_atomic_included is True


def test_unknown_representation_is_refused():
    with pytest.raises(BasisResponseError, match="representation must be one of"):
        BasisResponseArtifact(
            representation="whatever_the_backend_had",
            blocks={"D_S": np.eye(NO_U)[None]},
            isc_off=[[0, 0, 0]],
            backend="siesta",
            method="m",
            intra_atomic_included=False,
            context=_context(),
        )


# ---------------------------------------------------------------------------
# Provenance: backends, q, signatures
# ---------------------------------------------------------------------------


def test_graph2mat_derivative_with_siesta_overlap_is_hybrid():
    artifact = from_S_L_S_R(
        np.eye(NO_U), np.eye(NO_U), _context(electronic_derivative_backend="graph2mat"),
        backend="siesta", method="m", intra_atomic_included=False,
    )
    provenance = artifact.provenance()
    assert provenance["basis_response_backend"] == "siesta"
    assert provenance["epc_backend_class"] == EPC_BACKEND_CLASS_HYBRID

    pure = from_S_L_S_R(
        np.eye(NO_U), np.eye(NO_U), _context(electronic_derivative_backend="siesta"),
        backend="siesta", method="m", intra_atomic_included=False,
    )
    assert pure.provenance()["epc_backend_class"] == "siesta"


def test_signature_distinguishes_the_three_representations():
    geometry = epc_artifact_node(
        "geometry",
        {"cell": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "species": ["C"], "positions_sha256": "p",
         "coordinate_convention": "fractional"},
    )
    overlap = epc_artifact_node(
        "overlap",
        {"pao_basis": "C_2s2p", "siesta_runtime_ref": "VERIFIED_BINARY_ONLY", "fdf_physics_sha256": "f",
         "orb_indx_sha256": "o", "overlap_export_convention": "c", "units": "dimensionless"},
        {"geometry": geometry},
    )
    rng = np.random.default_rng(3)
    a = rng.normal(size=(NO_U, NO_U))
    s = a @ a.T + NO_U * np.eye(NO_U)

    artifacts = {
        REPRESENTATION_S_L_S_R: from_S_L_S_R(
            np.eye(NO_U), np.eye(NO_U), _context(), backend="b", method="m",
            intra_atomic_included=False,
        ),
        REPRESENTATION_D_S: from_D_S(
            np.eye(NO_U), _context(formalism_id=FORMALISM_ID_SYMMETRIC), backend="b", method="m",
        ),
        REPRESENTATION_COVARIANT: from_covariant_connection(
            np.eye(NO_U), s, _context(), backend="b", method="m", intra_atomic_included=False,
        ),
    }
    signatures = {}
    for representation, artifact in artifacts.items():
        node = artifact.signature_node(geometry=geometry, overlap=overlap)
        assert node["fields"]["representation"] == representation
        signatures[representation] = node["signature_sha256"]
    assert len(set(signatures.values())) == 3


def test_gamma_response_cannot_be_relabelled_at_finite_q():
    with pytest.raises(BasisResponseError, match="not q-aware"):
        _context(q=(1 / 3, 1 / 3, 0.0))
    context = _context(q=(1 / 3, 1 / 3, 0.0), q_aware_source=True)
    assert context.q == (1 / 3, 1 / 3, 0.0)


def test_context_demands_geometry_basis_and_one_perturbation():
    with pytest.raises(BasisResponseError, match="basis_signature"):
        _context(basis_signature=" ")
    with pytest.raises(BasisResponseError, match="exactly one"):
        _context(perturbation_definition="a second definition")
    with pytest.raises(BasisResponseError, match="exactly one"):
        ResponseContext(
            geometry_signature="g", basis_signature="b", electronic_derivative_backend="siesta"
        )


def test_artifact_fields_are_exactly_what_the_dag_declares():
    artifact = from_S_L_S_R(
        np.eye(NO_U), np.eye(NO_U), _context(), backend="siesta", method="m",
        intra_atomic_included=False,
    )
    fields = artifact.artifact_fields()
    assert fields["formalism_id"] == FORMALISM_ID
    assert fields["units"] == "1/Ang"
    assert fields["q"] == [0.0, 0.0, 0.0]
    assert fields["perturbation_definition"]["direction_hash"]
    convention = fields["convention"]
    assert convention["geometry_signature"] == "geo-sha"
    assert convention["basis_signature"] == "orbital-contract-sha"
    assert "exp(+2i*pi*k.R_l)" in convention["bloch"]
    assert basis_response_contract()["forbidden"].startswith("reconstructing")


# ---------------------------------------------------------------------------
# End to end on a real dHSdR.nc
# ---------------------------------------------------------------------------


def test_from_dhsdr_builds_a_usable_response(tmp_path):
    pytest.importorskip("netCDF4")
    from read_siesta_dhsdr import BOHR_TO_ANG, read_dhsdr
    from test_read_siesta_dhsdr import write_dhsdr

    left, right = _true_split()
    records = _ds_records(left, right)  # 1/Ang, already inter-atomic only

    # Same numbers, written the way SIESTA writes them: CSR over the supercell,
    # in 1/Bohr, filtered onto the displaced atom.
    blocks = {}
    for (atom, axis), values in records.items():
        n_col, list_col, data = [], [], []
        for row in range(NO_U):
            columns = [
                (image, col)
                for image in range(N_S)
                for col in range(NO_U)
                if ORBITAL_ATOM[row] == atom or ORBITAL_ATOM[col] == atom
            ]
            n_col.append(len(columns))
            list_col.extend(col + image * NO_U + 1 for image, col in columns)
            data.extend(values[image, row, col] * BOHR_TO_ANG for image, col in columns)
        dense_h = np.zeros((1, len(list_col)))
        blocks[(atom, axis)] = {
            "dH": (n_col, list_col, dense_h),
            "dS": (n_col, list_col, np.array(data)),
        }

    path = write_dhsdr(
        tmp_path / "tiny.dHSdR.nc", blocks=blocks, no_u=NO_U, na_u=2, nsc=(3, 1, 1), isc_off=ISC
    )
    dhsdr = read_dhsdr(path)

    direction = collective(np.array([[0.3, -0.4, 0.1], [0.2, 0.5, -0.6]]), name="v")
    artifact = from_dhsdr(
        dhsdr, _context(direction=direction), atom_of_orbital=ORBITAL_ATOM
    )

    assert artifact.representation == REPRESENTATION_S_L_S_R
    assert artifact.intra_atomic_included is False
    assert "intra_atomic_omitted" in artifact.method
    assert artifact.provenance()["source"]["dhsdr"]["derivative_method"] == "siesta_fc_dHS"
    assert artifact.provenance()["epc_backend_class"] == EPC_BACKEND_CLASS_HYBRID
    assert artifact.diagnostics["intra_atomic_residual_inv_ang"] == 0.0

    expected = inter_atomic_split(records, direction.vectors, ORBITAL_ATOM)
    assert np.allclose(artifact.blocks["S_left"], expected["S_left"], atol=1e-10)
    assert np.allclose(artifact.blocks["S_right"], expected["S_right"], atol=1e-10)

    # And it drops straight into the contraction, which cannot be reached without it.
    response = artifact.at_k((0.25, 0.0, 0.0))
    rng = np.random.default_rng(11)
    c = rng.normal(size=(NO_U, 2)) + 1j * rng.normal(size=(NO_U, 2))
    eps = np.array([-1.0, 0.5])
    block = epc_matrix_elements(
        rng.normal(size=(NO_U, NO_U)), response, C_row=c, C_col=c, eps_row=eps, eps_col=eps
    )
    assert block.formalism_id == FORMALISM_ID
    assert block.intra_atomic_included is False
    assert block.basis_response_backend == "siesta"
