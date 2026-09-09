"""Deterministic input signatures for cached derivative artifacts (audit Fase 5).

A cached ``.npz`` may only be reused when its sidecar metadata carries an
``input_signature_sha256`` that matches the signature recomputed from the
CURRENT inputs (checkpoint, code, structure, direction, dtype, method).
Anything else — no metadata, no signature (legacy), mismatch, unreadable or
non-finite payload — must be recomputed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

INPUT_SIGNATURE_SCHEMA = "derivative_input_signature_v1"

CACHE_VALID = "valid"
CACHE_LEGACY_UNVERIFIED = "legacy_unverified"
CACHE_SIGNATURE_MISMATCH = "signature_mismatch"
CACHE_MISSING_METADATA = "missing_metadata"
CACHE_UNREADABLE = "unreadable"
CACHE_NON_FINITE = "non_finite"


def file_sha256(path: str | Path | None) -> str | None:
    if path in (None, ""):
        return None
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_signature_sha256(payload: dict[str, Any]) -> str:
    """Canonical sha256 over the signature payload (order-independent)."""
    encoded = json.dumps(
        {"schema": INPUT_SIGNATURE_SCHEMA, **payload},
        sort_keys=True,
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def cached_result_status(
    npz_path: str | Path,
    metadata_path: str | Path,
    expected_signature: str,
) -> str:
    """Classify an existing cached derivative for reuse (never raises)."""
    npz_path = Path(npz_path)
    metadata_path = Path(metadata_path)
    if not metadata_path.is_file():
        return CACHE_MISSING_METADATA
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return CACHE_UNREADABLE
    stored = metadata.get("input_signature_sha256")
    if not stored:
        return CACHE_LEGACY_UNVERIFIED
    if str(stored) != str(expected_signature):
        return CACHE_SIGNATURE_MISMATCH
    try:
        import numpy as np
        from scipy import sparse

        matrix = sparse.load_npz(npz_path)
        if matrix.data.size and not bool(np.all(np.isfinite(matrix.data))):
            return CACHE_NON_FINITE
        expected_shape = metadata.get("matrix_shape")
        if expected_shape and [int(x) for x in expected_shape] != [int(x) for x in matrix.shape]:
            return CACHE_SIGNATURE_MISMATCH
    except Exception:  # noqa: BLE001 - unreadable/corrupt payloads must not be reused
        return CACHE_UNREADABLE
    return CACHE_VALID


# ---------------------------------------------------------------------------
# EPC artifact DAG (C-DAG / E-F_001-S3)
# ---------------------------------------------------------------------------
# One opaque hash per pipeline is useless for EPC: re-plotting alpha^2F would
# invalidate the JVPs, and a moved atom would not obviously invalidate g. So a
# signature is computed per node from (its OWN declared fields) + (the
# signatures of its declared parents). Consequences, all of them mechanical:
#
#   * geometry sits at the root, so touching it changes every descendant hash;
#   * broadening/mesh live ONLY on ``integrated_observable``, and a field the
#     kind does not declare is rejected — nothing figure-shaped can be smuggled
#     into ``pao_projected_mode_coupling``, and the UI is not a node at all;
#   * smearing lives ONLY on ``electronic_occupations`` and its two dependants,
#     never on an eigenspace or on ``pao_projected_mode_coupling`` (roadmap XII);
#   * ``physical_phonon`` and ``synthetic_test_displacement`` are distinct
#     kinds with distinct fields. ``pao_projected_mode_coupling`` accepts only the
#     former as its ``phonon`` parent, and no physical node may have a
#     synthetic displacement anywhere in its ancestry (roadmap B7).
#
# This is not a second cache manager: the produced ``signature_sha256`` is the
# same ``input_signature_sha256`` value the existing producers already store as
# ``input_signature_sha256`` in their sidecar metadata, so ``epc_reuse_status``
# is just ``cached_result_status`` with the node's hash.

EPC_ARTIFACT_SIGNATURE_SCHEMA = "epc_artifact_signature_v1"

PHYSICAL_PHONON = "physical_phonon"
SYNTHETIC_TEST_DISPLACEMENT = "synthetic_test_displacement"

# GO-1 decides which of these the formalism actually requires; the artifact has
# to say which one it is instead of leaving "some overlap response" implicit.
BASIS_RESPONSE_REPRESENTATIONS = ("D_S", "S_L_S_R", "covariant_basis_connection")

EPC_BACKEND_CLASS_HYBRID = "hybrid"

# Occupations / chemical potential vocabulary (roadmap XII, E-F_001-S7).
# ``Comparison/scripts/epc_occupations.py`` is the physics that consumes it.
PAO_PROJECTED_MODE_COUPLING = "pao_projected_mode_coupling"

MU_FIXED_N = "fixed_N"
MU_FIXED_MU = "fixed_mu"
MU_POLICIES = (MU_FIXED_N, MU_FIXED_MU)

OCCUPATION_FUNCTIONS = ("fermi_dirac", "gaussian", "zero_temperature_step")
ZERO_TEMPERATURE_STEP = "zero_temperature_step"

# What an *electronic* artifact declares: what its eigenvalues mean and how many
# electrons fill them. Smearing is deliberately not here — it is the whole point
# of the split, so that re-smearing cannot invalidate a raw matrix element.
EIGENSPACE_OCCUPATION_FIELDS = ("electron_count", "spin_degeneracy", "mu_convention")

_MODE_KINDS = (PHYSICAL_PHONON, SYNTHETIC_TEST_DISPLACEMENT)

# kind -> declared fields, declared parents, and which parents may be omitted
# (a pure dH/dR artifact has no overlap input) or repeated (a lambda integrates
# a whole set of g).
EPC_ARTIFACT_KINDS: dict[str, dict[str, Any]] = {
    "geometry": {
        "fields": ("cell", "species", "positions_sha256", "coordinate_convention"),
        "deps": {},
    },
    "electronic_hamiltonian": {
        "fields": (
            "source_backend",  # graph2mat | siesta | tight_binding | ...
            "model_or_binary_sha256",
            "code_version",
            "basis",
            "neighbor_cutoff_policy",
            "topology_sha256",
            "dtype",
            "mapping_version",
        ),
        "deps": {"geometry": ("geometry",)},
    },
    "overlap": {
        "fields": (
            "pao_basis",
            "siesta_runtime_ref",  # C01 record: VERIFIED_SOURCE | VERIFIED_BINARY_ONLY
            "fdf_physics_sha256",
            "orb_indx_sha256",
            "overlap_export_convention",
            "units",
        ),
        "deps": {"geometry": ("geometry",)},
    },
    "electronic_eigenspace": {
        "fields": (
            "k",
            "solver_backend",
            "solver_version",
            "window",
            "shift",
            "tolerances",
            "state_count",
            "dtype",
            "occupations",  # EIGENSPACE_OCCUPATION_FIELDS; no smearing (roadmap XII)
        ),
        "deps": {"hamiltonian": ("electronic_hamiltonian",), "overlap": ("overlap",)},
    },
    PHYSICAL_PHONON: {
        "fields": (
            "provider",
            "provider_version",
            "force_source",
            "primitive_mapping",
            "fc_range",
            "q",
            "frequency_ev",
            "eigenvector_sha256",
            "masses",
            "normalization",
            "asr_policy",
            "fourier_convention",
            "units",
        ),
        "deps": {"geometry": ("geometry",)},
    },
    SYNTHETIC_TEST_DISPLACEMENT: {
        # Deliberately NOT a phonon: no omega, no eigenvector, no masses, no ASR.
        # Benchmark-only (roadmap 10b/D06).
        "fields": ("label", "displacement_sha256", "normalization", "units"),
        "deps": {"geometry": ("geometry",)},
    },
    "raw_derivative": {
        "fields": (
            "derivative_backend",
            "derivative_method",  # jvp | frozen_central_difference | siesta_fc_dHS
            "perturbation_definition",
            "q",
            "delta_or_jvp",
            "topology_sha256",
            "convention",
            "dtype",
            "units",
        ),
        "deps": {
            "hamiltonian": ("electronic_hamiltonian",),
            "overlap": ("overlap",),
            "mode": _MODE_KINDS,
            "kernel_table": ("raw_derivative_kernel_table",),
        },
        "optional_deps": ("overlap", "mode", "kernel_table"),
    },
    "raw_derivative_kernel_table": {
        # E-F_001-S33/S40: the q-independent real-space derivative kernel
        # (route Q-C, epc_qb_qc_prototypes.kernel_table): {shell: (A_l, B_l)}
        # built once per direction and reused for every (k, q) query
        # afterwards. Deliberately has no "q" or "mode" field -- that
        # independence from q is the entire point of this kind (S33 Sec. 2,
        # "signature_gap"); a per-(k, q) raw_derivative built by contracting
        # this table over Bloch phases may declare it as its "kernel_table"
        # dependency so a restart reuses the table instead of repeating the
        # finite differences that built it.
        "fields": (
            "derivative_backend",
            "derivative_method",  # graph2mat_jvp_per_shell | frozen_central_difference_per_shell
            "direction_hash",
            "topology_sha256",
            "delta_or_jvp",
            "shells",
            "dtype",
            "units",
        ),
        "deps": {"hamiltonian": ("electronic_hamiltonian",)},
    },
    "basis_response": {
        "fields": (
            "formalism_id",
            "representation",
            "backend",
            "method",
            "perturbation_definition",
            "q",
            "convention",
            "dtype",
            "units",
        ),
        "deps": {
            "geometry": ("geometry",),
            "overlap": ("overlap",),
            "mode": _MODE_KINDS,
        },
        "optional_deps": ("mode",),
    },
    "pao_covariant_response": {
        "fields": (
            "formalism_id",
            "basis_response_convention",
            "electronic_derivative_backend",
            "basis_response_backend",
            "units",
        ),
        "deps": {"raw_derivative": ("raw_derivative",), "basis_response": ("basis_response",)},
        "requires_non_synthetic_ancestry": True,
    },
    "electronic_occupations": {
        # The smearing lives here and only here. It fills a set of eigenspaces;
        # nothing upstream of it (H, S, eigenvalues, D_H, g) depends on it.
        "fields": (
            "electron_count",
            "spin_degeneracy",
            "occupation_function",
            "smearing_width_ev",
            "mu_convention",
            "mu_policy",  # fixed_N | fixed_mu
            "mu_ev",
            "mu_source",
        ),
        "deps": {"eigenspaces": ("electronic_eigenspace",)},
        "multi_deps": ("eigenspaces",),
    },
    "eigenvalue_directional_derivative": {
        # diag of (F2). Needs the declared energy zero, never an occupation:
        # d(mu) is NOT subtracted here, which is what separates it from
        # ``deformation_potential_relative_to_mu``.
        "fields": ("k", "band_indices", "mu_convention", "mu_subtracted", "units"),
        "deps": {
            "perturbation": ("pao_covariant_response",),
            "eigenspace": ("electronic_eigenspace",),
        },
    },
    "deformation_potential_relative_to_mu": {
        # d(eps_n) - d(mu): only defined at fixed N, and only for observables
        # that ask for it.
        "fields": ("k", "band_indices", "units"),
        "deps": {
            "eigenvalue_derivative": ("eigenvalue_directional_derivative",),
            "occupations": ("electronic_occupations",),
        },
        "requires_non_synthetic_ancestry": True,
    },
    "fermi_surface_response": {
        # d(mu) itself, from the whole mesh: sum(w f' deps) / sum(w f').
        "fields": ("response_quantity", "kpoint_weights_sha256", "units"),
        "deps": {
            "eigenvalue_derivatives": ("eigenvalue_directional_derivative",),
            "occupations": ("electronic_occupations",),
        },
        "multi_deps": ("eigenvalue_derivatives",),
        "requires_non_synthetic_ancestry": True,
    },
    PAO_PROJECTED_MODE_COUPLING: {
        "fields": (
            "k",
            "k_plus_q",
            "fourier_convention",
            "phonon_normalization",
            "phonon_backend",
            "epc_backend_class",
            "units",
        ),
        "deps": {
            "perturbation": ("pao_covariant_response",),
            "phonon": (PHYSICAL_PHONON,),
            "eigenspace_k": ("electronic_eigenspace",),
            "eigenspace_k_plus_q": ("electronic_eigenspace",),
        },
        "requires_non_synthetic_ancestry": True,
    },
    "integrated_observable": {
        # Everything figure-shaped lives here and nowhere upstream.
        "fields": (
            "observable",
            "weights",
            "mesh",
            "occupations",
            "smearing",
            "broadening",
            "integration_convention",
        ),
        "deps": {"g_set": (PAO_PROJECTED_MODE_COUPLING,)},
        "multi_deps": ("g_set",),
        "requires_non_synthetic_ancestry": True,
    },
}


class EpcDagError(ValueError):
    """A node violates the EPC artifact contract (fields, parents or ancestry)."""


def _is_epc_node(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and value.get("schema") == EPC_ARTIFACT_SIGNATURE_SCHEMA
        and isinstance(value.get("kind"), str)
        and isinstance(value.get("signature_sha256"), str)
    )


def _check_occupation_rules(kind: str, fields: dict[str, Any], parents: dict[str, Any]) -> None:
    """Roadmap XII: who may declare a smearing, and who may subtract d(mu)."""
    if kind == "electronic_eigenspace":
        occupations = fields.get("occupations")
        if not isinstance(occupations, dict) or set(occupations) != set(EIGENSPACE_OCCUPATION_FIELDS):
            raise EpcDagError(
                f"electronic_eigenspace: 'occupations' must declare exactly "
                f"{list(EIGENSPACE_OCCUPATION_FIELDS)}, got {occupations!r}. A smearing here would "
                f"make re-smearing invalidate every {PAO_PROJECTED_MODE_COUPLING} built on it (roadmap XII)"
            )

    if kind == "electronic_occupations":
        if fields["mu_policy"] not in MU_POLICIES:
            raise EpcDagError(f"electronic_occupations: mu_policy must be one of {MU_POLICIES}")
        if fields["occupation_function"] not in OCCUPATION_FUNCTIONS:
            raise EpcDagError(
                f"electronic_occupations: occupation_function must be one of {OCCUPATION_FUNCTIONS}"
            )
        if fields["mu_policy"] == MU_FIXED_MU and fields["mu_ev"] is None:
            raise EpcDagError("electronic_occupations: mu_policy 'fixed_mu' needs a numeric mu_ev")
        declared = {key: fields[key] for key in EIGENSPACE_OCCUPATION_FIELDS}
        for eigenspace in parents["eigenspaces"]:
            if eigenspace["fields"]["occupations"] != declared:
                raise EpcDagError(
                    f"electronic_occupations: eigenspace declares "
                    f"{eigenspace['fields']['occupations']!r} but the occupation contract says "
                    f"{declared!r}; they must fill the same states with the same electrons"
                )

    if kind in ("deformation_potential_relative_to_mu", "fermi_surface_response"):
        policy = parents["occupations"]["fields"]["mu_policy"]
        if policy != MU_FIXED_N:
            raise EpcDagError(
                f"{kind}: d(mu) is identically zero under mu_policy {policy!r}, so this artifact "
                f"would just restate its parent. Only {MU_FIXED_N!r} moves the chemical potential"
            )


def _check_kind_rules(kind: str, fields: dict[str, Any], parents: dict[str, Any]) -> None:
    """Contract checks that need the field values, not just their presence."""
    _check_occupation_rules(kind, fields, parents)
    if kind == "basis_response":
        representation = fields.get("representation")
        if representation not in BASIS_RESPONSE_REPRESENTATIONS:
            raise EpcDagError(
                f"basis_response: representation must be one of {BASIS_RESPONSE_REPRESENTATIONS}, "
                f"got {representation!r} (GO-1 has to name what the response actually is)"
            )
    if kind == PAO_PROJECTED_MODE_COUPLING:
        perturbation_fields = parents["perturbation"]["fields"]
        backends = {
            str(perturbation_fields["electronic_derivative_backend"]),
            str(perturbation_fields["basis_response_backend"]),
            str(fields["phonon_backend"]),
        }
        declared = str(fields["epc_backend_class"])
        if len(backends) > 1 and declared != EPC_BACKEND_CLASS_HYBRID:
            raise EpcDagError(
                f"{PAO_PROJECTED_MODE_COUPLING}: derivative/basis-response/phonon come from {sorted(backends)}, "
                f"so epc_backend_class must be {EPC_BACKEND_CLASS_HYBRID!r}, not {declared!r}"
            )


def epc_artifact_node(
    kind: str,
    fields: dict[str, Any],
    dependencies: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one signed node of the EPC artifact DAG.

    ``fields`` must be exactly the fields the kind declares — a missing one is
    unrecorded provenance, an extra one is a dependency the invalidation rules
    do not know about. ``dependencies`` maps the kind's declared parent names to
    nodes returned by this same function. The result is JSON-serialisable and
    meant to be embedded in the producer's existing manifest.
    """
    spec = EPC_ARTIFACT_KINDS.get(kind)
    if spec is None:
        raise EpcDagError(f"unknown EPC artifact kind {kind!r}; known: {sorted(EPC_ARTIFACT_KINDS)}")

    declared = set(spec["fields"])
    given = dict(fields)
    missing = sorted(declared - set(given))
    extra = sorted(set(given) - declared)
    if missing or extra:
        raise EpcDagError(
            f"{kind}: field contract violated (missing={missing}, unexpected={extra}); "
            f"declared fields are {sorted(declared)}"
        )

    supplied = dict(dependencies or {})
    optional = set(spec.get("optional_deps", ()))
    multi = set(spec.get("multi_deps", ()))
    parents: dict[str, Any] = {}
    resolved: dict[str, Any] = {}
    ancestor_kinds = {kind}

    for name, allowed in spec["deps"].items():
        value = supplied.pop(name, None)
        if name in multi and value is not None and not isinstance(value, (list, tuple)):
            raise EpcDagError(f"{kind}: dependency {name!r} takes a list of nodes")
        nodes = [] if value is None else (list(value) if name in multi else [value])
        if not nodes:
            if name not in optional:
                raise EpcDagError(f"{kind}: missing required dependency {name!r} (expected {'|'.join(allowed)})")
            resolved[name] = [] if name in multi else None
            continue
        signatures = []
        for node in nodes:
            if not _is_epc_node(node):
                raise EpcDagError(f"{kind}: dependency {name!r} is not an EPC artifact node")
            if node["kind"] not in allowed:
                raise EpcDagError(
                    f"{kind}: dependency {name!r} must be one of {allowed}, got {node['kind']!r}"
                )
            signatures.append(str(node["signature_sha256"]))
            ancestor_kinds.update(node.get("ancestor_kinds", ()))
        parents[name] = nodes[0] if name not in multi else nodes
        resolved[name] = sorted(signatures) if name in multi else signatures[0]

    if supplied:
        raise EpcDagError(f"{kind}: undeclared dependencies {sorted(supplied)}")

    if spec.get("requires_non_synthetic_ancestry") and SYNTHETIC_TEST_DISPLACEMENT in ancestor_kinds:
        raise EpcDagError(
            f"{kind}: a {SYNTHETIC_TEST_DISPLACEMENT!r} is in the ancestry of a PAO-covariant coupling "
            f"artifact; it carries no frequency, eigenvector, mass or normalisation and can "
            f"never satisfy a {PHYSICAL_PHONON!r} dependency (roadmap B7)"
        )

    _check_kind_rules(kind, given, parents)

    return {
        "schema": EPC_ARTIFACT_SIGNATURE_SCHEMA,
        "kind": kind,
        "signature_sha256": input_signature_sha256(
            {
                "epc_schema": EPC_ARTIFACT_SIGNATURE_SCHEMA,
                "kind": kind,
                "fields": given,
                "dependencies": resolved,
            }
        ),
        "fields": given,
        "dependencies": resolved,
        "ancestor_kinds": sorted(ancestor_kinds),
    }


def epc_reuse_status(
    node: dict[str, Any],
    npz_path: str | Path,
    metadata_path: str | Path,
) -> str:
    """Reuse verdict for a cached payload produced by ``node`` (same cache, DAG hash)."""
    if not _is_epc_node(node):
        raise EpcDagError("expected an EPC artifact node")
    return cached_result_status(npz_path, metadata_path, node["signature_sha256"])
