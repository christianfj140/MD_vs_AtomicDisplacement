"""C15 / E-F_001-S18: persisted S-normalised eigenspaces.

Two halves. The Julia half is the physics: DeepH hands back Ritz vectors that
are an arbitrary, unnormalised basis inside a degenerate cluster, and
``record_eigenspace!`` has to turn that into ``C†SC = I`` without moving the
eigenvalues. The Python half is the contract: the identity budget comes from the
residual and the window conditioning, a window is never cut through a cluster,
and the signature moves with H/S/k/solver/shift/window/states/tolerances while
staying blind to q and to phonon modes (which is what makes eigenpairs reusable).
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import eigh

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "Comparison" / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "shared"))

import run_deeph_sparse_spectrum as solver  # noqa: E402
from run_deeph_sparse_spectrum import (  # noqa: E402
    EIGENSPACE_SCHEMA,
    degenerate_subspace_labels,
    eigenspace_cache_status,
    eigenspace_identity_tolerance,
    eigenspace_manifest,
    eigenspace_set_signature,
    eigenspace_signature_node,
    load_persisted_eigenspace,
    persisted_eigenspace_data,
    solver_input_nodes,
)

SOLVER_ENVIRONMENT = REPO_ROOT / "Comparison/.tools/deeph_sparse_solver"


# -- fixtures ---------------------------------------------------------------


def _pencil(seed: int, size: int = 8, spectrum: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """A non-orthogonal pencil; with ``spectrum``, one of prescribed eigenvalues.

    ``H = S^{1/2} Q diag(e) Q† S^{1/2}`` has exactly the eigenvalues ``e`` in the
    ``S`` metric, which is how an *exact* degeneracy gets into a fixture — a
    random ``H`` never has one.
    """
    generator = np.random.default_rng(seed)
    a = generator.normal(size=(size, size)) + 1j * generator.normal(size=(size, size))
    h = (a + a.conj().T) / 2
    b = generator.normal(size=(size, size)) + 1j * generator.normal(size=(size, size))
    overlap = np.eye(size) + 0.05 * (b + b.conj().T) / 2
    if spectrum is not None:
        values, vectors = np.linalg.eigh(overlap)
        root = vectors @ np.diag(np.sqrt(values)) @ vectors.conj().T
        unitary, _ = np.linalg.qr(
            generator.normal(size=(size, size)) + 1j * generator.normal(size=(size, size))
        )
        h = root @ unitary @ np.diag(np.asarray(spectrum, dtype=float)) @ unitary.conj().T @ root
        h = (h + h.conj().T) / 2
    return h, overlap


def write_eigenspace(
    output_dir: Path,
    k_index: int,
    *,
    seed: int = 0,
    states: int = 4,
    size: int = 8,
    spectrum: np.ndarray | None = None,
) -> np.ndarray:
    """Write what ``deeph_eigenspace_persist.jl`` writes, for one k point."""
    h, overlap = _pencil(seed, size=size, spectrum=spectrum)
    energies, vectors = eigh(h, overlap)
    coefficients = np.ascontiguousarray(vectors[:, :states], dtype=np.complex128)
    energies = energies[:states]
    deviation = coefficients.conj().T @ overlap @ coefficients - np.eye(states)
    residuals = [
        float(
            np.linalg.norm(h @ coefficients[:, n] - energies[n] * overlap @ coefficients[:, n])
            / (
                np.linalg.norm(h @ coefficients[:, n])
                + abs(energies[n]) * np.linalg.norm(overlap @ coefficients[:, n])
            )
        )
        for n in range(states)
    ]
    tag = f"{k_index:03d}"
    (output_dir / f"eigenspace_vectors_{tag}.bin").write_bytes(coefficients.tobytes(order="F"))
    (output_dir / f"eigenspace_overlap_{tag}.bin").write_bytes(deviation.tobytes(order="F"))
    (output_dir / f"eigenspace_{tag}.json").write_text(
        json.dumps({
            "schema": EIGENSPACE_SCHEMA,
            "k_index": k_index,
            "k_fractional": [0.0, 0.5 * k_index, 0.0],
            "shift_eV": 0.0,
            "window_half_width_eV": None,
            "degeneracy_tolerance_eV": 1e-6,
            "solver_band_indices": list(range(states)),
            "solver_energies_eV": energies.tolist(),
            "energies_eV": energies.tolist(),
            "maximum_energy_shift_vs_solver_eV": 0.0,
            "generalized_relative_residual": residuals,
            "window_metric_minimum_eigenvalue": 0.9,
            "window_metric_maximum_eigenvalue": 1.1,
            "window_metric_condition_number": 1.1 / 0.9,
            "norbits": coefficients.shape[0],
            "state_count": states,
            "dtype": "complex128",
            "storage_order": "fortran",
            "vectors_file": f"eigenspace_vectors_{tag}.bin",
            "overlap_minus_identity_file": f"eigenspace_overlap_{tag}.bin",
            "method": "loewdin_S_metric_then_rayleigh_ritz_in_window",
            "eigenvectors_persisted": True,
        }),
        encoding="utf-8",
    )
    return coefficients


def write_solver_input(input_dir: Path) -> Path:
    input_dir.mkdir(parents=True, exist_ok=True)
    (input_dir / "lat.dat").write_text("2.46 0.0 0.0\n0.0 2.46 0.0\n0.0 0.0 20.0\n", encoding="utf-8")
    (input_dir / "rlat.dat").write_text("2.55 0.0 0.0\n0.0 2.55 0.0\n0.0 0.0 0.31\n", encoding="utf-8")
    (input_dir / "site_positions.dat").write_text("0.0 1.23\n0.0 0.71\n0.0 0.0\n", encoding="utf-8")
    (input_dir / "orbital_types.dat").write_text("0 1\n0 1\n", encoding="utf-8")
    (input_dir / "element.dat").write_text("6\n6\n", encoding="utf-8")
    (input_dir / "info.json").write_text(
        json.dumps({"isorthogonal": False, "isspinful": False, "norbits": 8, "nsites": 2}),
        encoding="utf-8",
    )
    return input_dir


NEUTRALITY = {
    "method": "uniform_kmesh_generalized_inertia_zero_temperature",
    "energy_eV": 0.0,
    "chemical_potential_available": True,
    "neutral_electrons": 8,
    "spin_degeneracy": 2,
}


# -- tolerance and subspace labels ------------------------------------------


def test_identity_tolerance_is_driven_by_residual_and_window_conditioning() -> None:
    roundoff = eigenspace_identity_tolerance(0.0, 1.0, 8)
    assert 0 < roundoff < 1e-14
    # an ill-conditioned window metric loosens the budget ...
    assert eigenspace_identity_tolerance(0.0, 1e6, 8) == pytest.approx(roundoff * 1e6)
    # ... and a loosely converged eigenpair dominates it outright
    assert eigenspace_identity_tolerance(1e-7, 10.0, 8) == pytest.approx(1e-7)


def test_degenerate_subspace_labels_follow_gaps_not_band_indices() -> None:
    energies = np.array([-1.0, 0.1, 0.1 + 1e-12, 0.1 + 2e-12, 0.9])
    assert degenerate_subspace_labels(energies, 1e-6) == [0, 1, 1, 1, 2]
    assert degenerate_subspace_labels(energies, 1e-15) == [0, 1, 2, 3, 4]
    with pytest.raises(RuntimeError, match="ascending"):
        degenerate_subspace_labels(np.array([1.0, 0.0]), 1e-6)


# -- reading and validating what the solver wrote ---------------------------


def test_persisted_window_round_trips_and_is_s_orthonormal(tmp_path: Path) -> None:
    expected = write_eigenspace(tmp_path, 0, seed=1)
    payload = load_persisted_eigenspace(tmp_path, 0)
    assert np.allclose(payload["coefficients"], expected)
    assert payload["coefficients"].flags["F_CONTIGUOUS"] or payload["coefficients"].shape[1] == 1

    write_eigenspace(tmp_path, 1, seed=2)
    rows, diagnostics = persisted_eigenspace_data(tmp_path, 2, degeneracy_ev=1e-6)
    assert diagnostics["status"] == "valid"
    assert diagnostics["eigenvectors_persisted"] is True
    assert diagnostics["maximum_identity_absolute_error"] < diagnostics["maximum_identity_tolerance"]
    assert [row["k_index"] for row in rows] == [0, 1]
    assert all(row["s_orthonormal"] for row in rows)
    assert rows[0]["subspace_labels"] == [0, 1, 2, 3]
    assert rows[0]["vectors_sha256"]


def test_a_window_that_is_not_s_orthonormal_is_rejected(tmp_path: Path) -> None:
    write_eigenspace(tmp_path, 0, seed=3)
    deviation = np.fromfile(tmp_path / "eigenspace_overlap_000.bin", dtype=np.complex128)
    deviation[1] += 1e-3
    (tmp_path / "eigenspace_overlap_000.bin").write_bytes(deviation.tobytes())
    with pytest.raises(RuntimeError, match="not S-orthonormal"):
        persisted_eigenspace_data(tmp_path, 1, degeneracy_ev=1e-6)


def test_a_truncated_coefficient_file_is_not_silently_reshaped(tmp_path: Path) -> None:
    write_eigenspace(tmp_path, 0, seed=4)
    path = tmp_path / "eigenspace_vectors_000.bin"
    path.write_bytes(path.read_bytes()[:-16])
    with pytest.raises(RuntimeError, match="expected"):
        load_persisted_eigenspace(tmp_path, 0)


# -- signature: what invalidates an eigenspace, and what must not -----------


def _node(tmp_path: Path, **overrides):
    parents = solver_input_nodes(
        write_solver_input(tmp_path / "input"),
        hamiltonian_backend="graph2mat",
        siesta_runtime_ref="VERIFIED_BINARY_ONLY",
    )
    fields = {
        "k": [0.0, 0.0, 0.0],
        "solver_backend": "cpu_mkl_pardiso",
        "solver_version": {"julia_version": "1.10.11"},
        "window": {"half_width_eV": 0.5},
        "shift": 0.0,
        "tolerances": {"solver_max_iter": 1000},
        "state_count": 4,
        "occupations": {"electron_count": 8.0, "spin_degeneracy": 2, "mu_convention": "zero"},
    }
    fields.update(overrides)
    return eigenspace_signature_node(parents, **fields)


def test_signature_moves_with_k_window_shift_states_solver_and_tolerances(tmp_path: Path) -> None:
    reference = _node(tmp_path)["signature_sha256"]
    assert _node(tmp_path)["signature_sha256"] == reference
    for change in (
        {"k": [1 / 3, 2 / 3, 0.0]},
        {"window": {"half_width_eV": 0.25}},
        {"shift": 0.1},
        {"state_count": 6},
        {"solver_backend": "gpu_cudss"},
        {"solver_version": {"julia_version": "1.11.0"}},
        {"tolerances": {"solver_max_iter": 2000}},
        {"occupations": {"electron_count": 4.0, "spin_degeneracy": 2, "mu_convention": "zero"}},
    ):
        assert _node(tmp_path, **change)["signature_sha256"] != reference, change


def test_signature_moves_with_the_hamiltonian_and_the_overlap(tmp_path: Path) -> None:
    input_dir = write_solver_input(tmp_path / "input")
    (input_dir / "hamiltonians_pred.h5").write_bytes(b"H-version-1")
    (input_dir / "overlaps.h5").write_bytes(b"S-version-1")
    reference = _node(tmp_path)["signature_sha256"]
    (input_dir / "hamiltonians_pred.h5").write_bytes(b"H-version-2")
    predicted_again = _node(tmp_path)["signature_sha256"]
    assert predicted_again != reference
    (input_dir / "overlaps.h5").write_bytes(b"S-version-2")
    assert _node(tmp_path)["signature_sha256"] != predicted_again


def test_eigenspaces_are_reusable_across_modes_and_q(tmp_path: Path) -> None:
    """The kind declares no q and no mode: nothing phonon-shaped can move it."""
    node = _node(tmp_path)
    assert "q" not in node["fields"] and "mode" not in node["dependencies"]
    assert set(node["dependencies"]) == {"hamiltonian", "overlap"}
    assert node["fields"]["occupations"].keys() == {
        "electron_count",
        "spin_degeneracy",
        "mu_convention",
    }


# -- the manifest the campaign consumes -------------------------------------


def _manifest(tmp_path: Path, **overrides) -> dict:
    output_dir = tmp_path / "output"
    output_dir.mkdir(exist_ok=True)
    write_eigenspace(output_dir, 0, seed=5)
    arguments = {
        "kpoint_count": 1,
        "shift": 0.0,
        "solver_backend": "cpu_mkl_pardiso",
        "backend_requested": "cpu_mkl_pardiso",
        "gpu_preflight": None,
        "environment": {"julia_version": "1.10.11", "julia_sha256": "abc"},
        "solver_script": REPO_ROOT / "Comparison/scripts/deeph_sparse_calc_eigenspace.jl",
        "window_ev": 0.5,
        "degeneracy_ev": 1e-6,
        "min_metric": 1e-10,
        "state_budget": 8,
        "max_iter": 1000,
        "neutrality_reference": NEUTRALITY,
        "hamiltonian_backend": "graph2mat",
        "siesta_runtime_ref": "VERIFIED_BINARY_ONLY",
    }
    arguments.update(overrides)
    return eigenspace_manifest(write_solver_input(tmp_path / "input"), output_dir, **arguments)


def test_manifest_declares_persistence_occupations_and_a_reusable_signature(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    assert manifest["status"] == "valid"
    assert manifest["eigenvectors_persisted"] is True
    contract = manifest["occupation_contract"]
    assert contract["electron_count"] == 8.0
    assert contract["spin_degeneracy"] == 2
    assert contract["mu_ev"] == 0.0 and contract["mu_policy"] == "fixed_N"
    assert contract["mu_convention"] and contract["occupation_function"]
    assert manifest["window"]["half_width_eV"] == 0.5
    assert manifest["eigenspaces"][0]["signature"]["kind"] == "electronic_eigenspace"
    assert manifest["input_signature_sha256"] == eigenspace_set_signature(
        [row["signature"] for row in manifest["eigenspaces"]]
    )
    assert json.loads(
        (tmp_path / "output" / "eigenspaces_manifest.json").read_text(encoding="utf-8")
    )["input_signature_sha256"] == manifest["input_signature_sha256"]


def test_manifest_without_an_electron_count_refuses_to_declare_occupations(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="electron count"):
        _manifest(tmp_path, neutrality_reference={"energy_eV": 0.0, "limitation": "no element.dat"})


def test_cache_is_valid_only_for_the_same_signature_and_the_same_bytes(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    path = tmp_path / "output" / "eigenspaces_manifest.json"
    signature = manifest["input_signature_sha256"]
    assert eigenspace_cache_status(signature, path) == "valid"
    assert eigenspace_cache_status("other", path) == "signature_mismatch"
    assert eigenspace_cache_status(signature, tmp_path / "absent.json") == "missing_metadata"
    vectors = tmp_path / "output" / "eigenspace_vectors_000.bin"
    vectors.write_bytes(vectors.read_bytes()[:-16])
    assert eigenspace_cache_status(signature, path) == "unreadable"


# -- the solver-side numerics, in Julia -------------------------------------


JULIA_CHECK = textwrap.dedent(
    """
    using LinearAlgebra, JSON, Random, SparseArrays
    include(joinpath(ARGS[1], "Comparison/scripts/deeph_eigenspace_persist.jl"))

    source = read(joinpath(ARGS[1], "..", "DeepH-pack/deeph/inference/sparse_calc.jl"), String)
    include(joinpath(ARGS[1], "Comparison/scripts/deeph_mulliken_weights.jl"))
    # every order the shipped wrappers can produce (mulliken always runs first)
    for patched in (
        patch_sparse_calc_for_eigenspace(source),
        patch_sparse_calc_for_mulliken(source),
        patch_sparse_calc_for_eigenspace(patch_sparse_calc_for_mulliken(source)),
    )
        count("physical_egvec = egvec_sub", patched) == 2 || error("expected two Ritz captures")
        # one ill-projection capture per branch, each at its own indentation: a
        # 20-space needle without the leading newline also matches the 28-space
        # band line, which would leave the DOS branch stale
        count("\\n" * " "^28 * "physical_egvec = egvec\\n", patched) == 1 || error("band capture")
        count("\\n" * " "^20 * "physical_egvec = egvec\\n", patched) == 1 || error("DOS capture")
    end
    count("record_eigenspace!(parsed_args", patch_sparse_calc_for_eigenspace(source)) == 2 ||
        error("expected two record hooks")

    n = 6
    ENV["DEEPH_EIGENSPACE_WINDOW_EV"] = "0.6"
    ENV["DEEPH_EIGENSPACE_DEGENERACY_EV"] = "1e-6"
    generator = MersenneTwister(7)
    b = randn(generator, ComplexF64, n, n)
    S = Matrix{ComplexF64}(I, n, n) + 0.05*(b + b')/2
    S = (S + S')/2
    F = eigen(Hermitian(S))
    Shalf = F.vectors * Diagonal(sqrt.(F.values)) * F.vectors'
    Sinvhalf = F.vectors * Diagonal(1 ./ sqrt.(F.values)) * F.vectors'
    Q = Matrix(qr(randn(generator, ComplexF64, n, n)).Q)
    energies = [-3.0, -0.2, 0.1, 0.1, 0.5, 4.0]
    H = Shalf * Q * Diagonal(energies) * Q' * Shalf
    H = (H + H')/2

    # ARPACK hands over an arbitrary, unnormalised basis of the degenerate pair
    C = Sinvhalf * Q
    V = copy(C)
    V[:, 3] = C[:, 3] + C[:, 4]
    V[:, 4] = C[:, 3] - 0.5*C[:, 4]
    V .*= 3.7

    dir = mktempdir()
    record_eigenspace!(dir, 1, [0.0, 0.0, 0.0], energies, V, sparse(H), sparse(S), 0.0)
    payload = JSON.parsefile(joinpath(dir, "eigenspace_000.json"))
    states = payload["state_count"]
    Cout = reshape(reinterpret(ComplexF64, read(joinpath(dir, payload["vectors_file"]))), n, states)
    stored = reshape(
        reinterpret(ComplexF64, read(joinpath(dir, payload["overlap_minus_identity_file"]))),
        states, states,
    )
    JSON.print(stdout, Dict(
        "state_count" => states,
        "band_indices" => payload["solver_band_indices"],
        "identity" => maximum(abs.(Cout' * S * Cout - I)),
        "stored_identity_matches" => maximum(abs.(stored - (Cout' * S * Cout - I))),
        "residual" => maximum(payload["generalized_relative_residual"]),
        "energy_shift" => payload["maximum_energy_shift_vs_solver_eV"],
        "window_edge_keeps_cluster" => eigenspace_window(energies, 0.0, 0.35, 1e-6),
        "cluster_split_would_have_been" => [i for i in 1:6 if abs(energies[i]) <= 0.35],
    ), 2)
    """
).strip()


@pytest.mark.skipif(
    not (SOLVER_ENVIRONMENT / "julia-1.10.11/bin/julia").is_file(),
    reason="the bootstrapped DeepH sparse-solver Julia environment is not present",
)
def test_julia_makes_a_degenerate_ritz_basis_s_orthonormal(tmp_path: Path) -> None:
    script = tmp_path / "check.jl"
    script.write_text(JULIA_CHECK, encoding="utf-8")
    result = subprocess.run(
        [
            str(SOLVER_ENVIRONMENT / "julia-1.10.11/bin/julia"),
            "--startup-file=no",
            f"--project={SOLVER_ENVIRONMENT / 'project'}",
            str(script),
            str(REPO_ROOT),
        ],
        env={**dict(__import__("os").environ), "JULIA_DEPOT_PATH": str(SOLVER_ENVIRONMENT / "depot")},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    observed = json.loads(result.stdout)
    assert observed["state_count"] == 4
    assert observed["band_indices"] == [1, 2, 3, 4]
    assert observed["identity"] < 1e-12
    assert observed["stored_identity_matches"] < 1e-14
    assert observed["residual"] < 1e-10
    assert observed["energy_shift"] < 1e-10
    # a window edge at 0.35 eV would keep only band 2 on its own; the cluster at
    # 0.1 eV must come along whole
    assert observed["cluster_split_would_have_been"] == [2, 3, 4]
    assert sorted(observed["window_edge_keeps_cluster"]) == [2, 3, 4]


def test_the_shipped_wrappers_stay_in_step_with_the_patch() -> None:
    for wrapper in ("deeph_sparse_calc_eigenspace.jl", "deeph_sparse_calc_gpu.jl", "deeph_sparse_calc_projected.jl"):
        text = (REPO_ROOT / "Comparison/scripts" / wrapper).read_text(encoding="utf-8")
        assert "deeph_eigenspace_persist.jl" in text
    assert solver.EIGENSPACE_MANIFEST_NAME == "eigenspaces_manifest.json"
