"""C19 / E-F_001-S24: the PAO-covariant EPC contraction.

The fixture is a moving basis with an exactly known answer. Let the basis at
parameter ``t`` be ``phi(t) = (1 + t*Gamma) phi(0)`` and let a physical
perturbation ``V`` act on top of it:

    S(t) = (1 + t G)^dag S (1 + t G)          -> D_S = G^dag S + S G = S_L + S_R
    H(t) = (1 + t G)^dag H (1 + t G) + t V    -> D_H = G^dag H + H G + V

so ``PAO-covariant response`` (F1) is ``V`` exactly, and the generalized Hellmann-Feynman
diagonal of the pencil is ``diag(C^dag V C)``. Every number below is checked
against that closed form or against a central difference of the pencil's own
eigenvalues -- never against another contraction of the same code.

The rest of the file is about what the module refuses: a ``D_H`` without a
basis response, a synthetic displacement, a finite ``q``, a mode outside the
span of the computed directions, mismatched signatures, and a window that is
not S-orthonormal.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "Comparison" / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "shared"))

from artifact_signature import (  # noqa: E402
    EPC_BACKEND_CLASS_HYBRID,
    PAO_PROJECTED_MODE_COUPLING,
    EpcDagError,
    epc_artifact_node,
)
from compute_epc_matrix_elements import (  # noqa: E402
    RESULT_BOUND,
    RESULT_QUANTITATIVE,
    EpcContractionError,
    Eigenspace,
    PaoCovariantResponse,
    RawDerivative,
    contract,
    contraction_contract,
    dense_eigenspace,
    direction_coefficients,
    epc_matrix_element_report,
    finite_difference_eigenvalue_derivatives,
    hellmann_feynman_diagonal,
)
from epc_basis_response import ResponseContext, from_D_S, from_S_L_S_R  # noqa: E402
from epc_formalism import FORMALISM_ID, FORMALISM_ID_SYMMETRIC  # noqa: E402
from epc_subspaces import (  # noqa: E402
    block_metrics,
    cluster_groups,
    cross_cluster_mixing,
    random_gauge_rotation,
)
from fd_perturbation_space import collective  # noqa: E402
from phonon_provider import (  # noqa: E402
    ATOMIC_PHASE_ATOM,
    PhononConventions,
    PhononModeSet,
    SyntheticTestDisplacement,
)

GEOMETRY_SIGNATURE = "geometry-sha256-fixture"
BASIS_SIGNATURE = "orbital-contract-fixture"
ATOM_COUNT = 2
NO_U = 6  # three orbitals per atom
CELL = np.diag([2.46, 2.46, 15.0])
POSITIONS = np.array([[0.0, 0.0, 0.0], [1.23, 0.71, 0.0]])
MASSES = np.array([12.011, 12.011])


# --------------------------------------------------------------------------- #
# Fixture: a moving basis with a closed-form answer
# --------------------------------------------------------------------------- #


def _hermitian(generator: np.random.Generator, size: int) -> np.ndarray:
    matrix = generator.normal(size=(size, size))
    return 0.5 * (matrix + matrix.T)


def toy_system(seed: int = 7, degenerate: bool = True) -> dict:
    """``H``, ``S`` and one moving-basis response with a known ``PAO-covariant response``."""
    generator = np.random.default_rng(seed)
    overlap = np.eye(NO_U) + 0.08 * _hermitian(generator, NO_U)
    eigenvalues, vectors = np.linalg.eigh(overlap)
    overlap = (vectors * np.clip(eigenvalues, 0.4, None)) @ vectors.T

    # A spectrum with one exactly degenerate pair, so the gauge machinery has
    # something to rotate, and wide gaps elsewhere so the FD check has no crossings.
    spectrum = np.array([-6.0, -2.5, -0.4, -0.4 if degenerate else 0.9, 3.0, 7.5])
    basis, _ = np.linalg.qr(generator.normal(size=(NO_U, NO_U)))
    hamiltonian = basis @ np.diag(spectrum) @ basis.T
    # H must be the matrix of the pencil, i.e. in the same (non-orthogonal) basis.
    root = np.linalg.cholesky(overlap)
    hamiltonian = root @ hamiltonian @ root.T

    connection = 0.35 * generator.normal(size=(NO_U, NO_U))  # Gamma, 1/Ang
    potential = 0.9 * _hermitian(generator, NO_U)  # V, eV/Ang
    s_left = connection.T @ overlap
    s_right = overlap @ connection
    d_h = connection.T @ hamiltonian + hamiltonian @ connection + potential
    return {
        "H": hamiltonian,
        "S": overlap,
        "Gamma": connection,
        "V": potential,
        "S_left": s_left,
        "S_right": s_right,
        "D_S": s_left + s_right,
        "D_H": d_h,
    }


def make_direction(vectors=None, name: str = "e2g_bond_longitudinal"):
    if vectors is None:
        vectors = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    return collective(vectors, name=name)


def make_perturbation(
    system: dict,
    *,
    direction=None,
    derivative_backend: str = "siesta",
    response_backend: str = "siesta",
    intra_atomic_included: bool = True,
    geometry_signature: str = GEOMETRY_SIGNATURE,
    with_raw_D_S: bool = True,
    d_h: np.ndarray | None = None,
    d_s_scale: float = 1.0,
) -> PaoCovariantResponse:
    direction = direction or make_direction()
    blocks = {"D_H": system["D_H"] if d_h is None else d_h}
    if with_raw_D_S:
        blocks["D_S"] = d_s_scale * system["D_S"]
    raw = RawDerivative(
        blocks=blocks,
        isc_off=((0, 0, 0),),
        backend=derivative_backend,
        method="siesta_fc_dHS",
        direction=direction,
        geometry_signature=geometry_signature,
        basis_signature=BASIS_SIGNATURE,
        topology_sha256="topology-fixture",
        delta_or_jvp={"delta_ang": 0.01, "method": "central"},
    )
    context = ResponseContext(
        geometry_signature=geometry_signature,
        basis_signature=BASIS_SIGNATURE,
        electronic_derivative_backend=derivative_backend,
        direction=direction,
    )
    response = from_S_L_S_R(
        system["S_left"],
        system["S_right"],
        context,
        backend=response_backend,
        method="analytic_pao_overlap_derivative",
        intra_atomic_included=intra_atomic_included,
    )
    return PaoCovariantResponse(raw=raw, response=response)


def make_mode_set(
    *,
    q=(0.0, 0.0, 0.0),
    eigenvectors=None,
    frequencies=(0.196, 0.196),
    conventions: PhononConventions | None = None,
    geometry_signature: str = GEOMETRY_SIGNATURE,
    provider: str = "siesta_fc",
) -> PhononModeSet:
    """A two-branch degenerate doublet, mass-weighted and unit-norm (E2g-like)."""
    if eigenvectors is None:
        longitudinal = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]) / np.sqrt(2.0)
        transverse = np.array([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]) / np.sqrt(2.0)
        eigenvectors = np.stack([longitudinal, transverse])
    return PhononModeSet(
        q_fractional=q,
        frequencies_ev=np.asarray(frequencies, dtype=float),
        eigenvectors=eigenvectors,
        masses_amu=MASSES,
        cell_ang=CELL,
        positions_ang=POSITIONS,
        conventions=conventions or PhononConventions(),
        provider=provider,
        provider_version="1",
        force_source={"fc": "graphene.FC"},
        primitive_mapping={"repeats": [5, 5, 1]},
        fc_range={"label": "sc5x5x1"},
        asr_policy="corrected",
        asr_residual_ev_ang2=0.0,
        asr_residual_raw_ev_ang2=1.3e-3,
        geometry_signature=geometry_signature,
    )


def doublet_perturbations(system: dict, **kwargs) -> list[PaoCovariantResponse]:
    """One response per doublet member, so the two-mode block can be rotated."""
    generator = np.random.default_rng(11)
    patterns = (
        np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
        np.array([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]),
    )
    perturbations = []
    for index, vectors in enumerate(patterns):
        # A different D_H per direction; otherwise the two branches would be
        # trivially identical and a rotation could not be seen to work.
        d_h = system["D_H"] + 0.4 * index * _hermitian(generator, NO_U)
        perturbations.append(
            make_perturbation(
                system,
                direction=collective(vectors, name=f"e2g_member{index}"),
                d_h=d_h,
                **kwargs,
            )
        )
    return perturbations


def make_eigenspace(system: dict, label: str = "k_generic_1", node: dict | None = None):
    space = dense_eigenspace((0.29, 0.11, 0.0), system["H"], system["S"], label=label)
    if node is None:
        return space
    return Eigenspace(
        label=space.label,
        k=space.k,
        C=space.C,
        eps=space.eps,
        identity_error=space.identity_error,
        identity_tolerance=space.identity_tolerance,
        resolutions_ev=space.resolutions_ev,
        cluster_labels=space.cluster_labels,
        node=node,
        diagnostics=space.diagnostics,
    )


# --------------------------------------------------------------------------- #
# PAO-covariant response and the generalized Hellmann-Feynman diagonal
# --------------------------------------------------------------------------- #


def test_pao_covariant_response_recovers_the_pao_covariant_response():
    """(F1) on the moving-basis fixture is exactly ``V``, not ``D_H``."""
    system = toy_system()
    perturbation = make_perturbation(system)
    delta = perturbation.pao_covariant_response(system["H"], system["S"])
    assert np.allclose(delta, system["V"], atol=1e-10)
    # And it is emphatically not D_H: the basis response is the difference.
    assert np.linalg.norm(delta - system["D_H"]) > 1.0


def test_hellmann_feynman_diagonal_reproduces_the_finite_difference():
    """diag(F2) == d(eps)/dt of the pencil (H + t D_H, S + t D_S)."""
    system = toy_system(degenerate=False)
    perturbation = make_perturbation(system)
    space = make_eigenspace(system)
    block = perturbation.block(space, space)
    reference = finite_difference_eigenvalue_derivatives(
        system["H"], system["S"], system["D_H"], system["D_S"], delta=1e-4
    )
    assert np.allclose(np.real(np.diag(block.values)), reference, rtol=1e-6, atol=1e-7)
    # The same numbers are C_n^dag V C_n, the closed form of the fixture.
    closed_form = np.real(np.einsum("in,ij,jn->n", space.C.conj(), system["V"], space.C))
    assert np.allclose(np.real(np.diag(block.values)), closed_form, atol=1e-10)


def test_hellmann_feynman_check_passes_and_has_teeth():
    system = toy_system(degenerate=False)
    space = make_eigenspace(system)

    report = hellmann_feynman_diagonal(make_perturbation(system), space)
    assert report["d_s_source"] == "raw_derivative"
    assert report["passes"] and report["residual"] <= report["tolerance"]

    # A raw D_S in the wrong units is exactly what this identity is for.
    broken = hellmann_feynman_diagonal(make_perturbation(system, d_s_scale=2.0), space)
    assert not broken["passes"]

    fallback = hellmann_feynman_diagonal(make_perturbation(system, with_raw_D_S=False), space)
    assert fallback["d_s_source"] == "basis_response_sum" and fallback["passes"]


def test_ignoring_the_overlap_response_changes_the_answer():
    """The control negative is a different number, so the response is not decoration."""
    system = toy_system()
    space = make_eigenspace(system)
    production = make_perturbation(system).block(space, space).values
    raw_only = space.C.conj().T @ system["D_H"] @ space.C
    assert np.linalg.norm(production - raw_only) > 0.1 * np.linalg.norm(production)


# --------------------------------------------------------------------------- #
# What cannot be built
# --------------------------------------------------------------------------- #


def test_d_h_alone_is_not_a_pao_covariant_response():
    system = toy_system()
    raw = make_perturbation(system).raw
    with pytest.raises(EpcContractionError, match="BasisResponseArtifact"):
        PaoCovariantResponse(raw=raw, response=None)


def test_sum_only_response_cannot_feed_the_production_formalism():
    """A ``D_S`` response is a different representation, not a cheaper one."""
    system = toy_system()
    direction = make_direction()
    context = ResponseContext(
        geometry_signature=GEOMETRY_SIGNATURE,
        basis_signature=BASIS_SIGNATURE,
        electronic_derivative_backend="siesta",
        direction=direction,
    )
    with pytest.raises(Exception, match="does not determine it"):
        from_D_S(system["D_S"], context, backend="siesta", method="dhsdr")

    degraded = from_D_S(
        system["D_S"],
        ResponseContext(
            geometry_signature=GEOMETRY_SIGNATURE,
            basis_signature=BASIS_SIGNATURE,
            electronic_derivative_backend="siesta",
            direction=direction,
            formalism_id=FORMALISM_ID_SYMMETRIC,
        ),
        backend="siesta",
        method="dhsdr",
    )
    perturbation = PaoCovariantResponse(raw=make_perturbation(system).raw, response=degraded)
    assert perturbation.is_degraded
    space = make_eigenspace(system)
    assert perturbation.block(space, space).formalism_id == FORMALISM_ID_SYMMETRIC


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"geometry_signature": "another-geometry"}, "not the same calculation"),
        ({"derivative_backend": "graph2mat"}, "not the same calculation"),
    ],
)
def test_incompatible_signatures_are_refused(kwargs, match):
    system = toy_system()
    raw = make_perturbation(system).raw
    other = make_perturbation(system, **kwargs)
    with pytest.raises(EpcContractionError, match=match):
        PaoCovariantResponse(raw=raw, response=other.response)


def test_a_different_direction_is_refused():
    system = toy_system()
    raw = make_perturbation(system).raw
    other = make_perturbation(system, direction=make_direction(name="translation_x"))
    with pytest.raises(EpcContractionError, match="perturbation_definition"):
        PaoCovariantResponse(raw=raw, response=other.response)


def test_synthetic_displacement_cannot_produce_a_g():
    system = toy_system()
    space = make_eigenspace(system)
    synthetic = SyntheticTestDisplacement(
        label="graphene_optical_like",
        displacement_ang=np.array([[0.01, 0.0, 0.0], [-0.01, 0.0, 0.0]]),
    )
    with pytest.raises(Exception, match="physical_phonon"):
        epc_matrix_element_report([make_perturbation(system)], synthetic, 0, space)


def test_finite_q_is_refused_until_go5():
    system = toy_system()
    space = make_eigenspace(system)
    modes = make_mode_set(q=(1.0 / 3.0, 0.0, 0.0))
    with pytest.raises(EpcContractionError, match="GO-5|D01"):
        epc_matrix_element_report([make_perturbation(system)], modes, 0, space)


def test_non_canonical_phonon_conventions_are_refused():
    system = toy_system()
    space = make_eigenspace(system)
    modes = make_mode_set(conventions=PhononConventions(atomic_phase=ATOMIC_PHASE_ATOM))
    with pytest.raises(EpcContractionError, match="to_canonical_conventions"):
        epc_matrix_element_report([make_perturbation(system)], modes, 0, space)


def test_unsigned_or_foreign_phonon_geometry_is_refused():
    system = toy_system()
    space = make_eigenspace(system)
    with pytest.raises(EpcContractionError, match="no geometry_signature"):
        epc_matrix_element_report(
            [make_perturbation(system)], make_mode_set(geometry_signature=""), 0, space
        )
    with pytest.raises(EpcContractionError, match="different geometries"):
        epc_matrix_element_report(
            [make_perturbation(system)], make_mode_set(geometry_signature="other"), 0, space
        )


def test_mode_outside_the_span_of_the_computed_directions_is_refused():
    system = toy_system()
    # Only the longitudinal response exists; the transverse branch is not in its span.
    with pytest.raises(EpcContractionError, match="not in the span"):
        direction_coefficients(
            [make_perturbation(system)], np.array([[0.0, 0.01, 0.0], [0.0, -0.01, 0.0]])
        )


def test_a_window_that_is_not_s_orthonormal_is_refused():
    system = toy_system()
    space = make_eigenspace(system)
    with pytest.raises(EpcContractionError, match="not S-orthonormal"):
        Eigenspace(
            label="broken",
            k=space.k,
            C=space.C,
            eps=space.eps,
            identity_error=1e-3,
            identity_tolerance=space.identity_tolerance,
            resolutions_ev=space.resolutions_ev,
            cluster_labels=space.cluster_labels,
        )


def test_raw_derivative_cannot_smuggle_the_basis_response():
    system = toy_system()
    with pytest.raises(EpcContractionError, match="belong to the basis-response artifact"):
        RawDerivative(
            blocks={"D_H": system["D_H"], "S_left": system["S_left"]},
            isc_off=((0, 0, 0),),
            backend="siesta",
            method="siesta_fc_dHS",
            direction=make_direction(),
            geometry_signature=GEOMETRY_SIGNATURE,
            basis_signature=BASIS_SIGNATURE,
            topology_sha256="topology-fixture",
            delta_or_jvp={"delta_ang": 0.01},
        )


# --------------------------------------------------------------------------- #
# Contraction, linearity and gauge
# --------------------------------------------------------------------------- #


def test_contraction_is_linear_in_the_displacement():
    system = toy_system()
    space = make_eigenspace(system)
    perturbations = doublet_perturbations(system)
    first = contract(perturbations, [1.0, 0.0], space, space).values
    second = contract(perturbations, [0.0, 1.0], space, space).values
    mixed = contract(perturbations, [0.3, -1.7], space, space).values
    assert np.allclose(mixed, 0.3 * first - 1.7 * second, atol=1e-12)


def test_mode_expansion_reproduces_the_displacement_pattern():
    system = toy_system()
    perturbations = doublet_perturbations(system)
    modes = make_mode_set()
    pattern = modes.mode(0).displacement_pattern()
    coefficients, diagnostics = direction_coefficients(perturbations, pattern)
    rebuilt = sum(
        weight * p.direction.vectors for weight, p in zip(coefficients, perturbations)
    )
    assert np.allclose(rebuilt, pattern, atol=1e-12)
    assert diagnostics["span_relative_residual"] < 1e-12


def test_electronic_gauge_rotation_leaves_the_cluster_block_invariants():
    """``C -> C U`` inside a cluster leaves every cluster block's singular values;
    mixing across a resolved gap does not. The whole-block singular values are
    invariant under *any* unitary, so only the cluster-resolved metrics can tell
    a gauge from a change of physics."""
    system = toy_system(degenerate=True)
    space = make_eigenspace(system)
    labels = space.cluster_labels
    groups = cluster_groups(labels)
    assert any(len(group) > 1 for group in groups), "the fixture must have a degenerate cluster"

    block = make_perturbation(system).block(space, space).values
    reference = block_metrics(block, labels, labels)

    gauge = random_gauge_rotation(groups, space.state_count, np.random.default_rng(3))
    rotated = block_metrics(gauge.conj().T @ block @ gauge, labels, labels)
    for expected, measured in zip(reference, rotated):
        assert np.allclose(expected["singular_values"], measured["singular_values"], atol=1e-10)
        assert np.isclose(expected["frobenius_norm"], measured["frobenius_norm"], atol=1e-10)

    mixing = cross_cluster_mixing(groups, space.state_count)
    mixed = block_metrics(mixing.conj().T @ block @ mixing, labels, labels)
    assert any(
        not np.allclose(expected["singular_values"], measured["singular_values"], atol=1e-6)
        for expected, measured in zip(reference, mixed)
    ), "mixing across a resolved gap must move the metrics; the check has no teeth otherwise"


def test_phonon_doublet_rotation_leaves_the_two_mode_block_invariant():
    """The gauge-fixed doublet member is a label; the two-mode block is the claim."""
    system = toy_system()
    space = make_eigenspace(system)
    perturbations = doublet_perturbations(system)

    def two_mode_singular_values(modes: PhononModeSet) -> np.ndarray:
        blocks = []
        for branch in range(modes.mode_count):
            coefficients, _ = direction_coefficients(
                perturbations, modes.mode(branch).displacement_pattern()
            )
            blocks.append(contract(perturbations, coefficients, space, space).values.reshape(-1))
        return np.linalg.svd(np.stack(blocks), compute_uv=False)

    modes = make_mode_set()
    angle = 0.7
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    rotated = make_mode_set(
        eigenvectors=np.einsum("ab,bij->aij", rotation, modes.eigenvectors)
    )
    assert np.allclose(
        two_mode_singular_values(modes), two_mode_singular_values(rotated), rtol=1e-10
    )


# --------------------------------------------------------------------------- #
# The artifact: backends, claims and signatures
# --------------------------------------------------------------------------- #


def _nodes(perturbation: PaoCovariantResponse, modes: PhononModeSet | None = None) -> dict:
    geometry = epc_artifact_node(
        "geometry",
        {
            "cell": CELL.tolist(),
            "species": ["C", "C"],
            "positions_sha256": "positions-fixture",
            "coordinate_convention": "Ang",
        },
    )
    hamiltonian = epc_artifact_node(
        "electronic_hamiltonian",
        {
            "source_backend": perturbation.electronic_derivative_backend,
            "model_or_binary_sha256": "model-fixture",
            "code_version": "1",
            "basis": BASIS_SIGNATURE,
            "neighbor_cutoff_policy": "fixed",
            "topology_sha256": "topology-fixture",
            "dtype": "float64",
            "mapping_version": "1",
        },
        {"geometry": geometry},
    )
    overlap = epc_artifact_node(
        "overlap",
        {
            "pao_basis": "C.ion.xml",
            "siesta_runtime_ref": "VERIFIED_BINARY_ONLY",
            "fdf_physics_sha256": "fdf-fixture",
            "orb_indx_sha256": "orb-fixture",
            "overlap_export_convention": "sisl",
            "units": "dimensionless",
        },
        {"geometry": geometry},
    )
    raw_node = perturbation.raw.signature_node(hamiltonian=hamiltonian, overlap=overlap)
    basis_node = perturbation.response.signature_node(geometry=geometry, overlap=overlap)
    eigenspace = epc_artifact_node(
        "electronic_eigenspace",
        {
            "k": [0.29, 0.11, 0.0],
            "solver_backend": "dense_loewdin_numpy",
            "solver_version": "1",
            "window": [0, NO_U],
            "shift": 0.0,
            "tolerances": {"residual": 1e-12},
            "state_count": NO_U,
            "dtype": "complex128",
            "occupations": {
                "electron_count": 8,
                "spin_degeneracy": 2,
                "mu_convention": "exported_hamiltonian_fermi_zero",
            },
        },
        {"hamiltonian": hamiltonian, "overlap": overlap},
    )
    phonon = epc_artifact_node(
        "physical_phonon",
        (modes or make_mode_set()).physical_phonon_fields(),
        {"geometry": geometry},
    )
    return {
        "perturbation": perturbation.signature_node(
            raw_derivative=raw_node, basis_response=basis_node
        ),
        "phonon": phonon,
        "eigenspace": eigenspace,
    }


def test_report_records_the_backends_and_the_gauge_invariant_metrics():
    system = toy_system()
    perturbations = doublet_perturbations(system)
    modes = make_mode_set(provider="siesta")
    nodes = _nodes(perturbations[0], modes)
    space = make_eigenspace(system, node=nodes["eigenspace"])
    report = epc_matrix_element_report(
        perturbations,
        modes,
        0,
        space,
        perturbation_nodes=[_nodes(p, modes)["perturbation"] for p in perturbations],
        phonon_node=nodes["phonon"],
    )
    assert report["formalism_id"] == FORMALISM_ID
    assert report["electronic_derivative_backend"] == "siesta"
    assert report["basis_response_backend"] == "siesta"
    assert report["phonon_backend"] == "siesta"
    assert report["epc_backend_class"] == "siesta"
    assert report["result_class"] == RESULT_QUANTITATIVE
    assert report["g"]["units"] == "eV"
    assert report["g_per_unit_displacement"]["units"] == "eV/Ang"
    assert report["g"]["gauge_invariant"]["singular_values"]
    assert report["g"]["cluster_blocks"], "cluster-resolved metrics must be emitted"
    assert all(entry["passes"] for entry in report["hellmann_feynman"])
    # eV/Ang * Ang = eV, through the mode's own displacement norm.
    scale = report["mode"]["displacement_frobenius_Ang"]
    assert np.allclose(
        np.asarray(report["g"]["values_real"]),
        scale * np.asarray(report["g_per_unit_displacement"]["values_real"]),
        atol=1e-12,
    )
    assert report["signatures"][0]["kind"] == PAO_PROJECTED_MODE_COUPLING


def test_a_hybrid_result_is_labelled_hybrid():
    """Three backend names, one label, and the caller never supplies it."""
    system = toy_system()
    perturbation = make_perturbation(system, derivative_backend="graph2mat")
    space = make_eigenspace(system)
    report = epc_matrix_element_report([perturbation], make_mode_set(), 0, space)
    assert report["electronic_derivative_backend"] == "graph2mat"
    assert report["basis_response_backend"] == "siesta"
    assert report["phonon_backend"] == "siesta_fc"
    assert report["epc_backend_class"] == EPC_BACKEND_CLASS_HYBRID

    # Same rule for a SIESTA D_H under SIESTA-FC phonons: the DAG compares the
    # declared names, so two different names are two backends.
    all_siesta = epc_matrix_element_report(
        [make_perturbation(system)], make_mode_set(), 0, space
    )
    assert all_siesta["epc_backend_class"] == EPC_BACKEND_CLASS_HYBRID


def test_an_unresolved_intra_atomic_term_downgrades_the_claim_to_a_bound():
    system = toy_system()
    perturbation = make_perturbation(system, intra_atomic_included=False)
    space = make_eigenspace(system)
    report = epc_matrix_element_report([perturbation], make_mode_set(), 0, space)
    assert report["result_class"] == RESULT_BOUND
    assert report["intra_atomic_included"] is False


def test_presentation_cannot_enter_the_signature():
    """Re-plotting must not invalidate a contraction (roadmap V/XIII)."""
    system = toy_system()
    perturbation = make_perturbation(system)
    nodes = _nodes(perturbation)
    space = make_eigenspace(system, node=nodes["eigenspace"])
    common = dict(
        perturbation_nodes=[nodes["perturbation"]], phonon_node=nodes["phonon"]
    )
    first = epc_matrix_element_report(
        [perturbation], make_mode_set(), 0, space, label="for the paper figure", **common
    )
    second = epc_matrix_element_report(
        [perturbation], make_mode_set(), 0, space, label="rerun with a new colour map", **common
    )
    signature = first["signatures"][0]["signature_sha256"]
    assert signature == second["signatures"][0]["signature_sha256"]

    fields = first["signatures"][0]["fields"]
    assert not {"broadening", "smearing", "mesh", "plot"} & set(fields)
    with pytest.raises(EpcDagError):
        epc_artifact_node(
            PAO_PROJECTED_MODE_COUPLING,
            {**fields, "broadening": 0.05},
            {
                "perturbation": nodes["perturbation"],
                "phonon": nodes["phonon"],
                "eigenspace_k": nodes["eigenspace"],
                "eigenspace_k_plus_q": nodes["eigenspace"],
            },
        )


def test_contract_declares_its_refusals():
    contract_data = contraction_contract()
    assert contract_data["formalism_id"] == FORMALISM_ID
    assert set(contract_data["backends_recorded"]) == {
        "electronic_derivative_backend",
        "basis_response_backend",
        "phonon_backend",
        "epc_backend_class",
    }
    assert "BasisResponseArtifact" in contract_data["refusals"]["d_h_as_pao_covariant_response"]
