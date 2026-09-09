"""C20 / E-F_001-S25: the three-path runner.

What is worth testing here is not the physics — that is measured by the run
itself and written into its report — but the machinery that decides *whether the
run is allowed and whether its numbers mean what they say*: the gate that reads
the frozen protocol, the eigenspace format the contraction and GO-6 consume, and
the two comparisons whose validity depends on a degeneracy (the Hellmann-Feynman
identity and the numerical floor). Everything here runs on toy matrices or on
the frozen protocol; nothing calls SIESTA, torch or the checkpoint.
"""

from __future__ import annotations

import copy
import ast
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "Comparison" / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "shared"))

import compute_epc_matrix_elements as epc  # noqa: E402
import epc_basis_response as ebr  # noqa: E402
import fd_perturbation_space as fdp  # noqa: E402
import preregister_graphene_gamma_epc as prereg  # noqa: E402
import run_graphene_gamma_epc_paths as c20  # noqa: E402
from artifact_signature import epc_artifact_node  # noqa: E402
from run_deeph_sparse_spectrum import load_persisted_eigenspace  # noqa: E402

PROTOCOL_PATH = prereg.DEFAULT_OUTPUT_DIR / prereg.PROTOCOL_NAME
protocol_present = pytest.mark.skipif(
    not PROTOCOL_PATH.is_file(), reason="the graphene Gamma protocol has not been frozen"
)


@pytest.fixture
def protocol() -> dict:
    return prereg.load_protocol(PROTOCOL_PATH)


# -- the gate ---------------------------------------------------------------


@protocol_present
def test_authorize_grants_finite_pao_only_when_protocol_is_ready(protocol):
    gate = c20.authorize(protocol)
    assert gate["result_class"] == epc.RESULT_QUANTITATIVE
    assert gate["status"] == c20.RESULT_STATUS_CANDIDATE
    assert gate["full_KS"] == "BLOCKED: Delta_out"


def test_c19_c20_path_contains_no_siesta_generator_calls():
    tree = ast.parse(inspect.getsource(c20))
    forbidden = {"run_campaign", "execute_run", "siesta_e2g_campaign"}
    calls = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, (ast.Attribute, ast.Name))
    }
    assert calls.isdisjoint(forbidden)


@protocol_present
def test_authorize_refuses_an_edited_protocol(protocol):
    tampered = copy.deepcopy(protocol)
    tampered["electronic"]["min_window_gap_ev"] = 0.5
    with pytest.raises(c20.ThreePathError, match="edited since it was frozen"):
        c20.authorize(tampered)


@protocol_present
def test_authorize_refuses_an_unanticipated_blocker(protocol):
    tampered = copy.deepcopy(protocol)
    tampered["blocking"] = [*protocol["blocking"], "GO-2 siesta derivative: FAIL"]
    tampered["frozen_content_sha256"] = prereg.freeze_hash(tampered)
    with pytest.raises(c20.ThreePathError, match="claim ladder did not pre-authorise"):
        c20.authorize(tampered)


@protocol_present
def test_authorize_refuses_an_unapproved_physics_review(protocol):
    tampered = copy.deepcopy(protocol)
    tampered["physics_review"]["status"] = "PENDING"
    tampered["frozen_content_sha256"] = prereg.freeze_hash(tampered)
    with pytest.raises(c20.ThreePathError, match="physics review"):
        c20.authorize(tampered)


@protocol_present
def test_directions_are_rebuilt_from_the_frozen_vectors(protocol):
    directions = c20.protocol_directions(protocol)
    frozen = {row["direction_name"]: row for row in protocol["perturbation"]["directions"]}
    assert set(directions) == set(frozen)
    for name, direction in directions.items():
        assert direction.direction_hash == frozen[name]["direction_hash"]
        assert direction.to_metadata() == {
            key: value for key, value in frozen[name].items() if key != "vectors"
        }


@protocol_present
def test_splits_are_evaluated_in_the_pre_registered_order_and_stay_disjoint(protocol):
    splits = c20.split_pairs(protocol)
    assert list(splits) == ["calibration", "validation", "result"]
    pairs = [pair for split in splits.values() for pair in split]
    assert len(pairs) == len(set(pairs))
    directions = {name for split in splits.values() for name, _ in split}
    assert len(directions) == sum(
        len(set(name for name, _ in split)) for split in splits.values()
    )


# -- the persisted eigenspaces ----------------------------------------------


def _pencil(size: int = 6, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed)
    a = generator.normal(size=(size, size)) + 1j * generator.normal(size=(size, size))
    b = generator.normal(size=(size, size)) + 1j * generator.normal(size=(size, size))
    return (a + a.conj().T) / 2, np.eye(size) + 0.05 * (b + b.conj().T) / 2


def test_solve_window_is_S_orthonormal_and_keeps_the_eigenvalues():
    H, S = _pencil()
    solved = c20._solve_window(H, S, [1, 2, 3])
    C = solved["coefficients"]
    identity = C.conj().T @ S @ C
    assert np.abs(identity - np.eye(3)).max() < 1e-12
    assert solved["maximum_energy_shift_vs_solver_eV"] < 1e-12
    residual = H @ C - (S @ C) * solved["energies_eV"][None, :]
    assert np.abs(residual).max() < 1e-10


def test_persisted_eigenspaces_round_trip_into_the_contraction(tmp_path):
    H, S = _pencil(size=6)
    blocks_h = H[None, :, :]
    blocks_s = S[None, :, :]
    isc_off = np.zeros((1, 3), dtype=np.int64)
    geometry = epc_artifact_node(
        "geometry",
        {"cell": [[1, 0, 0]], "species": ["C"], "positions_sha256": "x", "coordinate_convention": "cartesian_ang"},
    )
    parents = {
        "hamiltonian": epc_artifact_node(
            "electronic_hamiltonian",
            {
                "source_backend": "test",
                "model_or_binary_sha256": "h",
                "code_version": "0",
                "basis": {},
                "neighbor_cutoff_policy": "none",
                "topology_sha256": "t",
                "dtype": "complex128",
                "mapping_version": "v1",
            },
            {"geometry": geometry},
        ),
        "overlap": epc_artifact_node(
            "overlap",
            {
                "pao_basis": "b",
                "siesta_runtime_ref": "r",
                "fdf_physics_sha256": "f",
                "orb_indx_sha256": "o",
                "overlap_export_convention": "v1",
                "units": "dimensionless",
            },
            {"geometry": geometry},
        ),
    }
    manifest = c20.persist_eigenspaces(
        tmp_path,
        h_blocks=blocks_h,
        s_blocks=blocks_s,
        isc_off=isc_off,
        k_points=[{"label": "gamma", "k_fractional": [0.0, 0.0, 0.0], "window_indices": [2, 3]}],
        parents=parents,
        occupations={"electron_count": 4.0, "spin_degeneracy": 2, "mu_convention": "test"},
        solver_version={"solver": "dense_loewdin_numpy"},
        shift_ev=0.0,
    )
    assert manifest["kpoint_count"] == 1
    payload = load_persisted_eigenspace(tmp_path, 0)
    window = epc.Eigenspace.from_persisted(payload, label="gamma")
    assert window.state_count == 2
    # The contraction's own S-orthonormality budget accepts it, which is the
    # only thing that makes the format usable downstream.
    assert window.identity_error <= window.identity_tolerance
    assert window.node is not None


# -- the two comparisons a degeneracy invalidates ----------------------------


def _perturbation(D_H: np.ndarray, S_L: np.ndarray, S_R: np.ndarray, direction) -> epc.PaoCovariantResponse:
    isc_off = np.zeros((1, 3), dtype=np.int64)
    context = ebr.ResponseContext(
        geometry_signature="g",
        basis_signature="b",
        electronic_derivative_backend="test",
        direction=direction,
    )
    response = ebr.from_S_L_S_R(
        S_L[None, :, :],
        S_R[None, :, :],
        context,
        backend="test",
        method="analytic",
        intra_atomic_included=True,
        isc_off=isc_off,
    )
    raw = epc.RawDerivative(
        blocks={"D_H": D_H[None, :, :], "D_S": (S_L + S_R)[None, :, :]},
        isc_off=isc_off,
        backend="test",
        method="jvp",
        direction=direction,
        geometry_signature="g",
        basis_signature="b",
        topology_sha256="t",
        delta_or_jvp={"delta_ang": 0.01},
    )
    return epc.PaoCovariantResponse(raw, response)


def _hermitian(size: int, seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    a = generator.normal(size=(size, size)) + 1j * generator.normal(size=(size, size))
    return (a + a.conj().T) / 2


def test_hellmann_feynman_matches_the_eigenvalue_sweep_when_no_level_is_mixed():
    size = 5
    S = np.eye(size) + 0.02 * _hermitian(size, 1)
    H = np.diag(np.linspace(-8.0, 8.0, size)).astype(complex)
    D_H = _hermitian(size, 2)
    S_L = 0.01 * _hermitian(size, 3)
    direction = fdp.one_hot(2, 0, "x")
    report = c20.hellmann_feynman_against_eigenvalue_fd(
        {
            delta: _perturbation(D_H, S_L, S_L.conj().T, direction)
            for delta in (0.0005, 0.001, 0.002)
        },
        c20.epc.dense_eigenspace(
            (0.0, 0.0, 0.0), H, S, label="w", window=slice(1, 4), separation_factor=1.0
        ),
        H_k=H,
        S_k=S,
        window=(1, 4),
        all_energies_ev=np.linalg.eigvalsh(np.linalg.inv(np.linalg.cholesky(S)) @ H @ np.linalg.inv(np.linalg.cholesky(S)).conj().T),
        tau_num=1e-6,
    )
    # Every level is 4 eV from its neighbour and the perturbation moves them by
    # milli-eV, so each state is its own group and the identity is exact.
    assert [row["states"] for row in report["groups"]] == [[0], [1], [2]]
    assert report["passes"]
    assert not report["groups_inapplicable"]


def test_hellmann_feynman_groups_a_degeneracy_and_refuses_a_cut_one():
    size = 6
    S = np.eye(size)
    # Two exact degeneracies: one inside the window, one straddling its edge.
    energies = np.array([-9.0, -1.0, -1.0, 3.0, 3.0, 9.0])
    H = np.diag(energies).astype(complex)
    D_H = _hermitian(size, 7)
    S_L = np.zeros((size, size), dtype=complex)
    direction = fdp.one_hot(2, 0, "x")
    window = (0, 4)
    report = c20.hellmann_feynman_against_eigenvalue_fd(
        {
            delta: _perturbation(D_H, S_L, S_L.conj().T, direction)
            for delta in (0.0005, 0.001, 0.002)
        },
        c20.epc.dense_eigenspace(
            (0.0, 0.0, 0.0), H, S, label="w", window=slice(*window), separation_factor=1.0
        ),
        H_k=H,
        S_k=S,
        window=window,
        all_energies_ev=energies,
        tau_num=1e-6,
    )
    groups = {tuple(row["states"]): row for row in report["groups"]}
    assert (1, 2) in groups and groups[(1, 2)]["applicable"]
    # The state at the window edge shares its level with one outside it: the
    # block cannot contain the mixing, so the comparison is refused rather than
    # reported as a disagreement.
    assert report["groups_inapplicable"] == [[3]]
    assert report["passes"]


def test_tau_num_is_the_worst_plateau_pair_not_the_average():
    blocks = {
        0.005: np.zeros((2, 2)),
        0.01: np.full((2, 2), 0.1),
        0.02: np.full((2, 2), 0.5),
    }
    swept = c20.tau_num_from_sweep(blocks, [0.005, 0.01, 0.02])
    assert swept["tau_num"] == pytest.approx(np.linalg.norm(np.full((2, 2), 0.5)))
    # A plateau that names fewer amplitudes must not be widened by the others.
    restricted = c20.tau_num_from_sweep(blocks, [0.005, 0.01])
    assert restricted["plateau_deltas_ang"] == [0.005, 0.01]
    assert restricted["tau_num"] == pytest.approx(np.linalg.norm(np.full((2, 2), 0.1)))


def test_fc_derivative_is_moved_to_the_preregistered_fermi_gauge():
    equilibrium = type("Eq", (), {"fermi_ev": 3.0, "overlap": {(0, 0, (0, 0, 0)): 2.0, (0, 1, (0, 0, 0)): 1.0}})()
    absolute = {(0, 0, (0, 0, 0)): 5.0}
    d_s = {(0, 0, (0, 0, 0)): 0.25}
    converted = c20.fc_to_fermi_zero(absolute, d_s, equilibrium, 0.5)
    assert converted[(0, 0, (0, 0, 0))] == pytest.approx(5.0 - 3.0 * 0.25 - 0.5 * 2.0)
    # The term is not confined to the diagonal of the sparse field: every
    # overlap element carries it, and dropping the rest would bias g.
    assert converted[(0, 1, (0, 0, 0))] == pytest.approx(-0.5)


@protocol_present
def test_result_root_reports_cite_the_frozen_protocol():
    """Anything C20 writes under the result root must be citable as this run."""
    root = c20.DEFAULT_RESULT_ROOT
    reports = sorted(root.glob("*.json")) if root.is_dir() else []
    if not reports:
        pytest.skip("the campaign has not been produced in this checkout")
    stored = prereg.load_protocol(PROTOCOL_PATH)["frozen_content_sha256"]
    for report in reports:
        payload = json.loads(report.read_text(encoding="utf-8"))
        if payload.get("schema") not in {c20.SCHEMA, c20.PREFLIGHT_SCHEMA}:
            continue
        assert payload.get("preregistration_sha256") == stored, report
        if payload.get("schema") == c20.SCHEMA:
            assert payload.get("status") == c20.RESULT_STATUS_CANDIDATE, report
