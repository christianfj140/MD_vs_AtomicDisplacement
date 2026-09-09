"""Fase 5 (audit): derivative cache signatures — stale results must not be reused."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import sparse

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "shared"))

import pytest

from artifact_signature import (  # noqa: E402
    CACHE_LEGACY_UNVERIFIED,
    CACHE_MISSING_METADATA,
    CACHE_NON_FINITE,
    CACHE_SIGNATURE_MISMATCH,
    CACHE_UNREADABLE,
    CACHE_VALID,
    EPC_ARTIFACT_KINDS,
    EPC_ARTIFACT_SIGNATURE_SCHEMA,
    PHYSICAL_PHONON,
    SYNTHETIC_TEST_DISPLACEMENT,
    EpcDagError,
    cached_result_status,
    epc_artifact_node,
    epc_reuse_status,
    input_signature_sha256,
)

BASE_PAYLOAD = {
    "model": "graph2mat",
    "checkpoint_sha256": "abc",
    "repository_commits": {"MD": "sha1"},
    "structure_fdf_sha256": "fdf",
    "dtype": "float64",
    "atom_index": 0,
    "axis_index": 1,
}


def _write_cache(tmp_path: Path, signature: str | None, data=None, shape=None):
    matrix = sparse.csr_matrix(
        np.asarray(data if data is not None else [[1.0, 0.0], [0.0, 2.0]])
    )
    npz = tmp_path / "d.npz"
    with npz.open("wb") as handle:
        sparse.save_npz(handle, matrix)
    metadata = {"matrix_shape": shape or list(matrix.shape)}
    if signature is not None:
        metadata["input_signature_sha256"] = signature
    meta = tmp_path / "d.json"
    meta.write_text(json.dumps(metadata), encoding="utf-8")
    return npz, meta


def test_signature_is_deterministic_and_sensitive():
    sig = input_signature_sha256(BASE_PAYLOAD)
    assert sig == input_signature_sha256(dict(BASE_PAYLOAD))
    for key, value in [
        ("checkpoint_sha256", "OTHER"),        # different checkpoint
        ("structure_fdf_sha256", "OTHER"),      # different coordinates
        ("dtype", "float32"),                   # different dtype
        ("repository_commits", {"MD": "sha2"}),  # different code
        ("atom_index", 1),
    ]:
        assert input_signature_sha256({**BASE_PAYLOAD, key: value}) != sig, key


def test_matching_signature_is_valid(tmp_path):
    sig = input_signature_sha256(BASE_PAYLOAD)
    npz, meta = _write_cache(tmp_path, sig)
    assert cached_result_status(npz, meta, sig) == CACHE_VALID


def test_mismatched_signature_rejected(tmp_path):
    npz, meta = _write_cache(tmp_path, "stale-signature")
    sig = input_signature_sha256(BASE_PAYLOAD)
    assert cached_result_status(npz, meta, sig) == CACHE_SIGNATURE_MISMATCH


def test_legacy_without_signature_rejected(tmp_path):
    npz, meta = _write_cache(tmp_path, None)
    sig = input_signature_sha256(BASE_PAYLOAD)
    assert cached_result_status(npz, meta, sig) == CACHE_LEGACY_UNVERIFIED


def test_missing_metadata_rejected(tmp_path):
    npz, meta = _write_cache(tmp_path, "x")
    meta.unlink()
    assert cached_result_status(npz, meta, "x") == CACHE_MISSING_METADATA


def test_non_finite_payload_rejected(tmp_path):
    sig = input_signature_sha256(BASE_PAYLOAD)
    npz, meta = _write_cache(tmp_path, sig, data=[[np.nan, 0.0], [0.0, 1.0]])
    assert cached_result_status(npz, meta, sig) == CACHE_NON_FINITE


def test_shape_mismatch_rejected(tmp_path):
    sig = input_signature_sha256(BASE_PAYLOAD)
    npz, meta = _write_cache(tmp_path, sig, shape=[4, 4])
    assert cached_result_status(npz, meta, sig) == CACHE_SIGNATURE_MISMATCH


def test_corrupt_npz_rejected(tmp_path):
    sig = input_signature_sha256(BASE_PAYLOAD)
    npz, meta = _write_cache(tmp_path, sig)
    npz.write_bytes(b"not-an-npz")
    assert cached_result_status(npz, meta, sig) == CACHE_UNREADABLE


# ---------------------------------------------------------------------------
# EPC artifact DAG (E-F_001-S3)
# ---------------------------------------------------------------------------

FIELDS = {
    "geometry": {
        "cell": [[2.46, 0, 0], [1.23, 2.13, 0], [0, 0, 20.0]],
        "species": ["C", "C"],
        "positions_sha256": "pos-equilibrium",
        "coordinate_convention": "fractional",
    },
    "electronic_hamiltonian": {
        "source_backend": "siesta",
        "model_or_binary_sha256": "siesta-binary",
        "code_version": "5.4",
        "basis": "C_2s2p_DZP",
        "neighbor_cutoff_policy": "orbital_support",
        "topology_sha256": "topo-a",
        "dtype": "float64",
        "mapping_version": "orb_indx_v1",
    },
    "overlap": {
        "pao_basis": "C.ion.xml:abc",
        "siesta_runtime_ref": "VERIFIED_BINARY_ONLY:sha-binary",
        "fdf_physics_sha256": "fdf-abc",
        "orb_indx_sha256": "orb-abc",
        "overlap_export_convention": "onlyS_TSHS_v1",
        "units": "dimensionless",
    },
    "electronic_eigenspace": {
        "k": [0.0, 0.0, 0.0],
        "solver_backend": "scipy_sparse",
        "solver_version": "1.14",
        "window": [-2.0, 2.0],
        "shift": 0.0,
        "tolerances": {"residual": 1e-10},
        "state_count": 8,
        "dtype": "complex128",
        "occupations": {
            "electron_count": 8.0,
            "spin_degeneracy": 2,
            "mu_convention": "exported_hamiltonian_fermi_zero",
        },
    },
    "electronic_occupations": {
        "electron_count": 8.0,
        "spin_degeneracy": 2,
        "mu_convention": "exported_hamiltonian_fermi_zero",
        "occupation_function": "fermi_dirac",
        "smearing_width_ev": 0.025,
        "mu_policy": "fixed_N",
        "mu_ev": 0.0,
        "mu_source": "uniform_kmesh_generalized_inertia_zero_temperature",
    },
    "eigenvalue_directional_derivative": {
        "k": [0.0, 0.0, 0.0],
        "band_indices": [3, 4],
        "mu_convention": "exported_hamiltonian_fermi_zero",
        "mu_subtracted": False,
        "units": "eV/Ang",
    },
    "deformation_potential_relative_to_mu": {
        "k": [0.0, 0.0, 0.0],
        "band_indices": [3, 4],
        "units": "eV/Ang",
    },
    "fermi_surface_response": {
        "response_quantity": "d_mu",
        "kpoint_weights_sha256": "weights-uniform-12x12",
        "units": "eV/Ang",
    },
    PHYSICAL_PHONON: {
        "provider": "siesta_vibra",
        "provider_version": "5.4",
        "force_source": "FC",
        "primitive_mapping": "2atom_primitive_v1",
        "fc_range": [5, 5, 1],
        "q": [0.0, 0.0, 0.0],
        "frequency_ev": 0.196,
        "eigenvector_sha256": "e2g-1",
        "masses": {"C": 12.011},
        "normalization": "mass_weighted_unit_norm",
        "asr_policy": "corrected",
        "fourier_convention": "atomic_phase_tau",
        "units": "eV",
    },
    SYNTHETIC_TEST_DISPLACEMENT: {
        "label": "graphene_optical_like",
        "displacement_sha256": "synthetic-1",
        "normalization": "unit_norm",
        "units": "Ang",
    },
    "raw_derivative": {
        "derivative_backend": "siesta_fc",
        "derivative_method": "siesta_fc_dHS",
        "perturbation_definition": "collective_direction_v",
        "q": [0.0, 0.0, 0.0],
        "delta_or_jvp": {"delta_ang": 0.01},
        "topology_sha256": "topo-a",
        "convention": "repo_canonical_v1",
        "dtype": "float64",
        "units": "eV/Ang",
    },
    "raw_derivative_kernel_table": {
        "derivative_backend": "graph2mat",
        "derivative_method": "graph2mat_jvp_per_shell",
        "direction_hash": "direction-v1",
        "topology_sha256": "topo-a",
        "delta_or_jvp": {"jvp": True},
        "shells": 4,
        "dtype": "float64",
        "units": "eV/Ang",
    },
    "basis_response": {
        "formalism_id": "moving_pao_v1",
        "representation": "S_L_S_R",
        "backend": "siesta",
        "method": "analytic_pao_overlap_derivative",
        "perturbation_definition": "collective_direction_v",
        "q": [0.0, 0.0, 0.0],
        "convention": "repo_canonical_v1",
        "dtype": "float64",
        "units": "1/Ang",
    },
    "pao_covariant_response": {
        "formalism_id": "moving_pao_v1",
        "basis_response_convention": "repo_canonical_v1",
        "electronic_derivative_backend": "siesta",
        "basis_response_backend": "siesta",
        "units": "eV/Ang",
    },
    "pao_projected_mode_coupling": {
        "k": [0.0, 0.0, 0.0],
        "k_plus_q": [0.0, 0.0, 0.0],
        "fourier_convention": "repo_canonical_v1",
        "phonon_normalization": "zero_point_amplitude",
        "phonon_backend": "siesta",
        "epc_backend_class": "siesta",
        "units": "eV",
    },
    "integrated_observable": {
        "observable": "a2F",
        "weights": {"spin": 2},
        "mesh": {"k": [12, 12, 1], "q": [6, 6, 1]},
        "occupations": "fermi_dirac",
        "smearing": 0.025,
        "broadening": 0.005,
        "integration_convention": "2d_per_area",
    },
}


def _fields(kind, **overrides):
    return {**FIELDS[kind], **overrides}


def build_dag(
    geometry_positions="pos-equilibrium",
    broadening=0.005,
    mode_kind=None,
    smearing=0.025,
    mu_policy="fixed_N",
):
    """Full geometry -> ... -> a2F chain; ``mode_kind`` attaches a mode to the derivative."""
    nodes = {}
    nodes["geometry"] = epc_artifact_node(
        "geometry", _fields("geometry", positions_sha256=geometry_positions)
    )
    geometry = {"geometry": nodes["geometry"]}
    for kind in ("electronic_hamiltonian", "overlap", PHYSICAL_PHONON, SYNTHETIC_TEST_DISPLACEMENT):
        nodes[kind] = epc_artifact_node(kind, _fields(kind), geometry)
    nodes["electronic_eigenspace"] = epc_artifact_node(
        "electronic_eigenspace",
        _fields("electronic_eigenspace"),
        {"hamiltonian": nodes["electronic_hamiltonian"], "overlap": nodes["overlap"]},
    )
    nodes["raw_derivative_kernel_table"] = epc_artifact_node(
        "raw_derivative_kernel_table",
        _fields("raw_derivative_kernel_table"),
        {"hamiltonian": nodes["electronic_hamiltonian"]},
    )
    nodes["raw_derivative"] = epc_artifact_node(
        "raw_derivative",
        _fields("raw_derivative"),
        {
            "hamiltonian": nodes["electronic_hamiltonian"],
            "overlap": nodes["overlap"],
            "mode": nodes[mode_kind] if mode_kind else None,
            "kernel_table": nodes["raw_derivative_kernel_table"],
        },
    )
    nodes["basis_response"] = epc_artifact_node(
        "basis_response",
        _fields("basis_response"),
        {"geometry": nodes["geometry"], "overlap": nodes["overlap"]},
    )
    if mode_kind == SYNTHETIC_TEST_DISPLACEMENT:
        return nodes  # benchmark branch: it stops here, by contract
    nodes["pao_covariant_response"] = epc_artifact_node(
        "pao_covariant_response",
        _fields("pao_covariant_response"),
        {"raw_derivative": nodes["raw_derivative"], "basis_response": nodes["basis_response"]},
    )
    nodes["pao_projected_mode_coupling"] = epc_artifact_node(
        "pao_projected_mode_coupling",
        _fields("pao_projected_mode_coupling"),
        {
            "perturbation": nodes["pao_covariant_response"],
            "phonon": nodes[PHYSICAL_PHONON],
            "eigenspace_k": nodes["electronic_eigenspace"],
            "eigenspace_k_plus_q": nodes["electronic_eigenspace"],
        },
    )
    nodes["integrated_observable"] = epc_artifact_node(
        "integrated_observable",
        _fields("integrated_observable", broadening=broadening),
        {"g_set": [nodes["pao_projected_mode_coupling"]]},
    )
    nodes["electronic_occupations"] = epc_artifact_node(
        "electronic_occupations",
        _fields("electronic_occupations", smearing_width_ev=smearing, mu_policy=mu_policy),
        {"eigenspaces": [nodes["electronic_eigenspace"]]},
    )
    nodes["eigenvalue_directional_derivative"] = epc_artifact_node(
        "eigenvalue_directional_derivative",
        _fields("eigenvalue_directional_derivative"),
        {
            "perturbation": nodes["pao_covariant_response"],
            "eigenspace": nodes["electronic_eigenspace"],
        },
    )
    nodes["deformation_potential_relative_to_mu"] = epc_artifact_node(
        "deformation_potential_relative_to_mu",
        _fields("deformation_potential_relative_to_mu"),
        {
            "eigenvalue_derivative": nodes["eigenvalue_directional_derivative"],
            "occupations": nodes["electronic_occupations"],
        },
    )
    nodes["fermi_surface_response"] = epc_artifact_node(
        "fermi_surface_response",
        _fields("fermi_surface_response"),
        {
            "eigenvalue_derivatives": [nodes["eigenvalue_directional_derivative"]],
            "occupations": nodes["electronic_occupations"],
        },
    )
    return nodes


def test_every_declared_kind_is_covered_and_signed():
    nodes = build_dag()
    assert set(nodes) == set(EPC_ARTIFACT_KINDS)
    for kind, node in nodes.items():
        assert node["schema"] == EPC_ARTIFACT_SIGNATURE_SCHEMA
        assert node["kind"] == kind
        assert len(node["signature_sha256"]) == 64
    # distinct kinds with identical-looking provenance must never collide
    assert len({node["signature_sha256"] for node in nodes.values()}) == len(nodes)


def test_signatures_are_deterministic():
    assert {k: v["signature_sha256"] for k, v in build_dag().items()} == {
        k: v["signature_sha256"] for k, v in build_dag().items()
    }


def test_moving_an_atom_invalidates_every_scientific_descendant():
    before = build_dag()
    after = build_dag(geometry_positions="pos-displaced")
    for kind in EPC_ARTIFACT_KINDS:
        assert before[kind]["signature_sha256"] != after[kind]["signature_sha256"], kind


def test_refiguring_a_broadening_invalidates_only_the_integrated_observable():
    before = build_dag()
    after = build_dag(broadening=0.010)
    assert (
        before["integrated_observable"]["signature_sha256"]
        != after["integrated_observable"]["signature_sha256"]
    )
    for kind in set(EPC_ARTIFACT_KINDS) - {"integrated_observable"}:
        assert before[kind]["signature_sha256"] == after[kind]["signature_sha256"], kind


def test_figure_or_ui_fields_cannot_enter_g_or_a_derivative():
    for kind in ("pao_projected_mode_coupling", "raw_derivative"):
        with pytest.raises(EpcDagError, match="unexpected"):
            build_dag()  # sanity: the clean build works
            epc_artifact_node(kind, _fields(kind, broadening=0.005, ui_plot_style="dark"))


def test_synthetic_displacement_cannot_be_the_phonon_of_a_g():
    nodes = build_dag()
    with pytest.raises(EpcDagError, match="must be one of"):
        epc_artifact_node(
            "pao_projected_mode_coupling",
            _fields("pao_projected_mode_coupling"),
            {
                "perturbation": nodes["pao_covariant_response"],
                "phonon": nodes[SYNTHETIC_TEST_DISPLACEMENT],
                "eigenspace_k": nodes["electronic_eigenspace"],
                "eigenspace_k_plus_q": nodes["electronic_eigenspace"],
            },
        )


def test_synthetic_displacement_cannot_hide_in_the_ancestry_of_a_physical_artifact():
    nodes = build_dag(mode_kind=SYNTHETIC_TEST_DISPLACEMENT)
    assert SYNTHETIC_TEST_DISPLACEMENT in nodes["raw_derivative"]["ancestor_kinds"]
    with pytest.raises(EpcDagError, match=SYNTHETIC_TEST_DISPLACEMENT):
        epc_artifact_node(
            "pao_covariant_response",
            _fields("pao_covariant_response"),
            {"raw_derivative": nodes["raw_derivative"], "basis_response": nodes["basis_response"]},
        )


def test_a_physical_mode_is_accepted_and_distinguishable_from_a_synthetic_one():
    physical = build_dag(mode_kind=PHYSICAL_PHONON)
    synthetic = build_dag(mode_kind=SYNTHETIC_TEST_DISPLACEMENT)
    assert PHYSICAL_PHONON in physical["raw_derivative"]["ancestor_kinds"]
    assert (
        physical["raw_derivative"]["signature_sha256"]
        != synthetic["raw_derivative"]["signature_sha256"]
        != build_dag()["raw_derivative"]["signature_sha256"]
    )
    # the physical mode propagates all the way to g
    assert physical["pao_projected_mode_coupling"]["signature_sha256"] != build_dag()["pao_projected_mode_coupling"]["signature_sha256"]


def test_contract_violations_fail_closed():
    with pytest.raises(EpcDagError, match="unknown EPC artifact kind"):
        epc_artifact_node("g", {})
    with pytest.raises(EpcDagError, match="missing="):
        epc_artifact_node("geometry", {"cell": [], "species": [], "positions_sha256": "p"})
    with pytest.raises(EpcDagError, match="missing required dependency"):
        epc_artifact_node("electronic_hamiltonian", _fields("electronic_hamiltonian"))
    with pytest.raises(EpcDagError, match="undeclared dependencies"):
        epc_artifact_node(
            "electronic_hamiltonian",
            _fields("electronic_hamiltonian"),
            {"geometry": build_dag()["geometry"], "phonon": build_dag()[PHYSICAL_PHONON]},
        )
    with pytest.raises(EpcDagError, match="not an EPC artifact node"):
        epc_artifact_node("electronic_hamiltonian", _fields("electronic_hamiltonian"), {"geometry": "x"})


def test_basis_response_must_name_its_representation():
    nodes = build_dag()
    parents = {"geometry": nodes["geometry"], "overlap": nodes["overlap"]}
    for representation in ("D_S", "S_L_S_R", "covariant_basis_connection"):
        node = epc_artifact_node(
            "basis_response", _fields("basis_response", representation=representation), parents
        )
        assert node["fields"]["representation"] == representation
    with pytest.raises(EpcDagError, match="representation must be one of"):
        epc_artifact_node(
            "basis_response", _fields("basis_response", representation="ignore_overlap"), parents
        )


def test_mixed_backends_must_declare_themselves_hybrid():
    nodes = build_dag()
    perturbation = epc_artifact_node(
        "pao_covariant_response",
        _fields("pao_covariant_response", electronic_derivative_backend="graph2mat"),
        {"raw_derivative": nodes["raw_derivative"], "basis_response": nodes["basis_response"]},
    )
    parents = {
        "perturbation": perturbation,
        "phonon": nodes[PHYSICAL_PHONON],
        "eigenspace_k": nodes["electronic_eigenspace"],
        "eigenspace_k_plus_q": nodes["electronic_eigenspace"],
    }
    with pytest.raises(EpcDagError, match="hybrid"):
        epc_artifact_node(
            "pao_projected_mode_coupling", _fields("pao_projected_mode_coupling", epc_backend_class="graph2mat"), parents
        )
    node = epc_artifact_node(
        "pao_projected_mode_coupling", _fields("pao_projected_mode_coupling", epc_backend_class="hybrid"), parents
    )
    assert node["fields"]["epc_backend_class"] == "hybrid"


# -- occupations / chemical potential (roadmap XII, E-F_001-S7) --------------

OCCUPATION_DEPENDENT = {
    "electronic_occupations",
    "deformation_potential_relative_to_mu",
    "fermi_surface_response",
}


def test_resmearing_invalidates_the_occupation_responses_and_nothing_else():
    """The acceptance criterion of S7: g does not move when the smearing does."""
    before = build_dag()
    after = build_dag(smearing=0.100)
    for kind in OCCUPATION_DEPENDENT:
        assert before[kind]["signature_sha256"] != after[kind]["signature_sha256"], kind
    for kind in set(EPC_ARTIFACT_KINDS) - OCCUPATION_DEPENDENT:
        assert before[kind]["signature_sha256"] == after[kind]["signature_sha256"], kind


def test_an_eigenspace_declares_its_filling_but_never_a_smearing():
    fields = _fields("electronic_eigenspace")
    assert set(fields["occupations"]) == {"electron_count", "spin_degeneracy", "mu_convention"}
    nodes = build_dag()
    parents = {"hamiltonian": nodes["electronic_hamiltonian"], "overlap": nodes["overlap"]}
    for broken in (
        {**fields["occupations"], "smearing_width_ev": 0.025},  # would drag g along
        {"electron_count": 8.0},  # incomplete: no mu convention
        "fermi_dirac",  # not even a mapping
    ):
        with pytest.raises(EpcDagError, match="occupations"):
            epc_artifact_node(
                "electronic_eigenspace", _fields("electronic_eigenspace", occupations=broken), parents
            )


def test_an_occupation_contract_must_match_the_states_it_fills():
    nodes = build_dag()
    with pytest.raises(EpcDagError, match="same electrons"):
        epc_artifact_node(
            "electronic_occupations",
            _fields("electronic_occupations", electron_count=7.0),
            {"eigenspaces": [nodes["electronic_eigenspace"]]},
        )
    for bad in ({"mu_policy": "whatever"}, {"occupation_function": "lorentzian"}):
        with pytest.raises(EpcDagError, match="must be one of"):
            epc_artifact_node(
                "electronic_occupations",
                _fields("electronic_occupations", **bad),
                {"eigenspaces": [nodes["electronic_eigenspace"]]},
            )
    with pytest.raises(EpcDagError, match="needs a numeric mu_ev"):
        epc_artifact_node(
            "electronic_occupations",
            _fields("electronic_occupations", mu_policy="fixed_mu", mu_ev=None),
            {"eigenspaces": [nodes["electronic_eigenspace"]]},
        )


def test_only_fixed_N_may_subtract_a_chemical_potential_response():
    nodes = build_dag()
    fixed_mu = epc_artifact_node(
        "electronic_occupations",
        _fields("electronic_occupations", mu_policy="fixed_mu", mu_ev=-0.2),
        {"eigenspaces": [nodes["electronic_eigenspace"]]},
    )
    with pytest.raises(EpcDagError, match="identically zero"):
        epc_artifact_node(
            "deformation_potential_relative_to_mu",
            _fields("deformation_potential_relative_to_mu"),
            {
                "eigenvalue_derivative": nodes["eigenvalue_directional_derivative"],
                "occupations": fixed_mu,
            },
        )
    with pytest.raises(EpcDagError, match="identically zero"):
        epc_artifact_node(
            "fermi_surface_response",
            _fields("fermi_surface_response"),
            {
                "eigenvalue_derivatives": [nodes["eigenvalue_directional_derivative"]],
                "occupations": fixed_mu,
            },
        )
    # fixed-N is the branch that has something to say
    assert build_dag(mu_policy="fixed_N")["fermi_surface_response"]["fields"]["response_quantity"] == "d_mu"


def test_the_four_electronic_artifacts_of_section_XII_are_distinct():
    nodes = build_dag()
    kinds = (
        "pao_projected_mode_coupling",
        "eigenvalue_directional_derivative",
        "deformation_potential_relative_to_mu",
        "fermi_surface_response",
    )
    assert len({nodes[kind]["signature_sha256"] for kind in kinds}) == 4
    # d(mu) reaches exactly the two artifacts that asked for it
    for kind in ("pao_projected_mode_coupling", "eigenvalue_directional_derivative"):
        assert "electronic_occupations" not in nodes[kind]["ancestor_kinds"], kind
    for kind in ("deformation_potential_relative_to_mu", "fermi_surface_response"):
        assert "electronic_occupations" in nodes[kind]["ancestor_kinds"], kind
    assert nodes["eigenvalue_directional_derivative"]["fields"]["mu_subtracted"] is False


def test_node_signature_drives_the_existing_derivative_cache(tmp_path):
    node = build_dag()["raw_derivative"]
    npz, meta = _write_cache(tmp_path, node["signature_sha256"])
    assert epc_reuse_status(node, npz, meta) == CACHE_VALID
    stale = build_dag(geometry_positions="pos-displaced")["raw_derivative"]
    assert epc_reuse_status(stale, npz, meta) == CACHE_SIGNATURE_MISMATCH
