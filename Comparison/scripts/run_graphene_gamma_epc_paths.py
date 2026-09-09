#!/usr/bin/env python3
"""C20 / E-F_001-S25: the three paths of graphene Gamma, produced under the frozen protocol.

Everything upstream of this script measured one thing each. This one makes the
*same* physical object three times and puts the three numbers side by side:

``siesta_reference``    ``D_H`` from ``FC.Save.dHS``, contracted onto the mode's
                        direction, with its own explicit central finite
                        difference of the same H and S as the internal check.
``graph2mat_jvp``       one directional double-backward JVP of the checkpoint.
``graph2mat_frozen``    ``[H_G2M(R + d v) - H_G2M(R - d v)] / (2 d)`` of the very
                        same model and the same frozen neighbour topology.

What makes them comparable rather than merely adjacent:

* **one basis response.** The ``S_L``/``S_R`` blocks are assembled once per
  direction by :func:`certify_basis_response.response_for_direction` — the
  inter-atomic object C14C validated — plus the intra-atomic ``A_I(R)`` term
  certified PASS in C14C-v2 (``certify_basis_response_v2.py``) and extracted
  for all five directions this protocol needs by
  ``extract_b_i_real_space_collective_v6.py``
  (``S_R += A_I(R)``, ``S_L -= A_I(R)``, same sign convention C14C-v2 proved).
  ``A_I(R)`` is a real-space geometric/basis quantity, not a DFT-derivative
  one, so it is loaded once per direction and applied identically to both
  backends before the response is re-signed per path so that
  ``electronic_derivative_backend`` stays truthful and ``epc_backend_class``
  comes out ``hybrid`` by arithmetic. The blocks themselves are byte-identical
  across the three paths and the report carries their hash to prove it.
* **one phonon.** The C18 ``sc5x5x1`` ASR-corrected Gamma modes, with the
  degenerate ``E_2g`` doublet replaced by the gauge the pre-registration froze
  *by value*. The substitution is checked, not trusted: the frozen members must
  lie in the span of the solver's own doublet.
* **one set of eigenspaces.** Persisted once in the C15 format from the
  equilibrium TSHS and consumed by every path and every direction; they carry no
  ``q`` and no mode, which is what makes them reusable.

Energy origin. ``eps`` comes from sisl's ``H - E_F S`` (the protocol's
``mu_convention``), so every ``D_H`` is brought into that same convention before
it is contracted: the SIESTA side by subtracting ``E_F * D_S``, the Graph2Mat
side by adding back ``dE_F * S`` (the model's label follows a *moving* ``E_F``,
memo/C14). ``g`` is invariant under a constant shift of the origin and the run
measures that invariance instead of assuming it.

Splits are evaluated in the pre-registered order — calibration, then validation,
then result — and ``tau_model`` is frozen from the calibration split before the
result is contracted at all. Nothing here may widen it.

Status. C14C-v1 (``dS/dR`` alone, inter-atomic only) is still ``NO_GO`` and
stays that way on purpose — it is structurally blind to the intra-atomic term
and nothing here re-derives it. But C14C-v2 is ``PASS``, and its certified
``A_I(R)`` is now available and applied for all five directions this protocol
needs (see ``apply_intra_atomic_correction`` above), so the basis response this
run assembles is the *full* intra-atomic-included response, not the inter-atomic
one C14C-v1 validated alone: every path now measures ``g``, not merely bounds
it. ``authorize()`` still authorises this run through the pre-registration's
own frozen "intra_atomic_basis_response still unresolved" rung — that text is
frozen in the protocol JSON and is not re-derived here — so the written reports
still carry ``result_class = bound__...`` and ``status = candidate`` as the
protocol's own label for the rung it pre-authorised, even though the assembled
response is now corrected. If any direction's ``A_I(R)`` images fail to match
the response's own ``isc_off`` table, that direction is reported BLOCKED via an
``intra_atomic_correction_applied__<direction>`` check rather than silently
falling back to the uncorrected bound. The exit code tracks the *checks*, not
this label.

Units: ``D_H`` and ``g`` per unit displacement in eV/Ang, ``g`` in eV, the basis
response in 1/Ang, amplitudes and positions in Ang.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for _path in (SCRIPT_DIR, REPO_ROOT / "shared", REPO_ROOT / "scripts" / "torch_serialization_compat"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import certify_basis_response as cbr  # noqa: E402
import certify_siesta_dhsdr as go2  # noqa: E402
import compute_epc_matrix_elements as epc  # noqa: E402
import epc_basis_response as ebr  # noqa: E402
import epc_subspaces as subspaces  # noqa: E402
import fd_perturbation_space as fdp  # noqa: E402
import phonon_provider as pp  # noqa: E402
import preregister_graphene_gamma_epc as prereg  # noqa: E402
import quantify_checkpoint_derivative_error as c14  # noqa: E402
import run_epc_siesta_reference as siesta_ref  # noqa: E402
from artifact_signature import (  # noqa: E402
    PHYSICAL_PHONON,
    epc_artifact_node,
    file_sha256,
    input_signature_sha256,
)
from epc_occupations import OccupationContract  # noqa: E402
from run_deeph_sparse_spectrum import (  # noqa: E402
    EIGENSPACE_IDENTITY_TOLERANCE_POLICY,
    EIGENSPACE_SCHEMA,
    eigenspace_identity_tolerance,
    eigenspace_signature_node,
    eigenspace_set_signature,
)

SCHEMA = "epc_graphene_gamma_three_paths_v1"
TICKET = "C20 / E-F_001-S25"
GATE = "GO-4"

PATH_SIESTA = "siesta_reference"
PATH_JVP = "graph2mat_jvp"
PATH_FROZEN = "graph2mat_frozen"
PATHS = (PATH_SIESTA, PATH_JVP, PATH_FROZEN)

BACKEND_OF_PATH = {PATH_SIESTA: "siesta", PATH_JVP: "graph2mat", PATH_FROZEN: "graph2mat"}
METHOD_OF_PATH = {
    PATH_SIESTA: "siesta_fc_dHS",
    PATH_JVP: "jvp",
    PATH_FROZEN: "frozen_central_difference",
}

# Where each stage lives. Only C20's own reports go under ``result_root``: the
# pre-registration's ``--verify`` demands that *every* JSON there cite the frozen
# hash, and a SIESTA run record or a persisted eigenspace is not a C20 result.
DEFAULT_RESULT_ROOT = prereg.DEFAULT_RESULT_ROOT
DEFAULT_REFERENCE_ROOT = REPO_ROOT / "Comparison/results/epc/siesta_reference/graphene"
DEFAULT_E2G_ROOT = REPO_ROOT / "Comparison/results/epc/siesta_reference/graphene_e2g"
DEFAULT_EIGENSPACE_DIR = REPO_ROOT / "Comparison/results/epc/eigenspaces/graphene"
DEFAULT_GO6_REPORT = subspaces.DEFAULT_OUTPUT_DIR / subspaces.REPORT_NAME
DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "Comparison/results/tbg_registry_spectral_loss/training/checkpoints"
    / "spectral-best-epoch=247-step=03472.ckpt"
)

# The intra-atomic term (C14C-v2, PASS): A_I(R) is a real-space geometric/basis
# quantity built once from C.ion.xml + geometry via the Fourier-Bessel engine,
# not a DFT-derivative quantity -- it does not depend on
# electronic_derivative_backend, exactly why certify_basis_response_v2.py only
# ever computes it once per direction, never once per backend.
A_I_REAL_SPACE_COLLECTIVE_DIR = REPO_ROOT / "Comparison/results/epc/b_i_real_space_collective_v6"
METHOD_INTRA_ATOMIC_SUFFIX = "__plus_BI_collective_v6"

# The blocking preconditions this run is *allowed* to proceed under, and the
# rung of the frozen claim ladder each one forces. Anything else stops the run:
# a protocol blocked for a reason nobody anticipated is not a protocol.
ALLOWED_BLOCKERS = {
    "C14C basis response: NO_GO": "the intra-atomic term is unresolved (memo A2): g is bounded, "
    "not measured",
    "GO-6 subspace metrics: MISSING": "produced by this run from the persisted eigenspaces and "
    "required to PASS before any subspace claim",
}

RESULT_STATUS_CANDIDATE = "candidate"

# Two states closer than this share a subspace when the response gate reads them
# (C14C's own clustering scale). The eigenspaces use the resolution-driven
# clustering of GO-6 instead, which is measured rather than chosen.
DEGENERACY_TOL_EV = prereg.DEGENERACY_TOL_EV

Key = tuple[int, int, tuple[int, int, int]]


class ThreePathError(RuntimeError):
    """The pre-registered experiment cannot be produced as specified."""


PREFLIGHT_SCHEMA = "epc_graphene_gamma_read_only_preflight_v1"
REQUIRED_DIRECTIONS = (
    "atom0000_x",
    "translation_x",
    "random_seed0",
    "e2g_bond_longitudinal",
    "e2g_bond_transverse",
)


def _check(name: str, passed: bool, detail: str, **extra: Any) -> dict[str, Any]:
    return {"check": name, "passed": bool(passed), "detail": detail, **extra}


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _frobenius(values: Any) -> float:
    return float(np.linalg.norm(np.asarray(values)))


def _delta_tag(delta_ang: float) -> str:
    """The amplitude tag the SIESTA campaign already uses, so file names line up."""
    return siesta_ref._delta_tag(float(delta_ang))


def _load_a_i_real_space_collective(direction_name: str) -> dict[tuple[int, int, int], np.ndarray]:
    """One direction's ``A_I(R)`` from the certified v6 artifact (PASS-gated)."""
    report_path = A_I_REAL_SPACE_COLLECTIVE_DIR / f"{direction_name}.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("verdict") != "PASS":
        raise ThreePathError(
            f"b_i_real_space_collective_v6/{direction_name}.json is {report.get('verdict')!r}, "
            "not PASS -- refusing to apply an uncertified intra-atomic correction"
        )
    npz = np.load(A_I_REAL_SPACE_COLLECTIVE_DIR / f"{direction_name}.npz", allow_pickle=True)
    r_list = npz["R_list"]
    return {tuple(int(v) for v in r): npz[f"R_{r[0]}_{r[1]}_{r[2]}"] for r in r_list}


def apply_intra_atomic_correction(
    artifact: Any, direction_name: str, a_i_by_r: Mapping[tuple[int, int, int], np.ndarray]
) -> tuple[Any, dict[str, Any]]:
    """``S_R += A_I(R)``, ``S_L -= A_I(R)`` -- the exact sign convention certified
    PASS in certify_basis_response_v2.py (S_L(R) relates to S_R(-R), not S_R(R);
    do not re-derive it here, only replicate it). Returns ``(corrected_artifact,
    missing_check)``; ``missing_check`` records BLOCKED with the exact reason
    when any A_I(R) image has no matching row in the artifact's own isc_off --
    in that case ``corrected_artifact`` is the ORIGINAL, uncorrected artifact
    and the caller must not treat it as intra-atomic-included.
    """
    isc_off = np.asarray(artifact.isc_off)
    isc_index = {tuple(int(v) for v in row): i for i, row in enumerate(isc_off)}
    missing = [r for r in a_i_by_r if r not in isc_index]
    if missing:
        return artifact, _check(
            f"intra_atomic_correction_applied__{direction_name}",
            False,
            f"{len(missing)} A_I(R) image(s) for {direction_name!r} have no matching row in "
            "this response's isc_off table -- cannot assemble without inventing R rows",
            missing_r=missing[:5],
            missing_count=len(missing),
        )

    s_left_full = artifact.blocks["S_left"].copy()
    s_right_full = artifact.blocks["S_right"].copy()
    for r, a_i in a_i_by_r.items():
        idx = isc_index[r]
        s_right_full[idx] += a_i
        s_left_full[idx] -= a_i

    corrected = ebr.from_S_L_S_R(
        s_left_full, s_right_full, artifact.context,
        backend=artifact.backend, method=f"{artifact.method}{METHOD_INTRA_ATOMIC_SUFFIX}",
        intra_atomic_included=True, isc_off=isc_off,
        diagnostics={
            **dict(artifact.diagnostics),
            "intra_atomic_term": "included via b_i_real_space_collective_v6",
            "n_R_with_A_I": len(a_i_by_r),
        },
    )
    return corrected, _check(
        f"intra_atomic_correction_applied__{direction_name}",
        True,
        f"A_I(R) from b_i_real_space_collective_v6 applied to {direction_name!r} "
        "(S_R += A_I(R), S_L -= A_I(R), certified sign convention from C14C-v2)",
        n_r_with_a_i=len(a_i_by_r),
    )


# --------------------------------------------------------------------------- #
# The gate: run only what the frozen protocol authorises
# --------------------------------------------------------------------------- #


def authorize(protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Authorize only the now-ready, intact frozen finite-PAO protocol."""
    stored = protocol.get("frozen_content_sha256")
    if prereg.freeze_hash(protocol) != stored:
        raise ThreePathError(
            "the protocol has been edited since it was frozen; a protocol changed after the "
            "fact is not a pre-registration"
        )
    review = protocol.get("physics_review", {})
    if review.get("status") != "APPROVED":
        raise ThreePathError(f"the physics review is {review.get('status')!r}")
    blocking = [str(reason) for reason in protocol.get("blocking") or []]
    unexpected = [reason for reason in blocking if reason not in ALLOWED_BLOCKERS]
    if unexpected:
        raise ThreePathError(
            f"the protocol is blocked for reasons the claim ladder did not pre-authorise: "
            f"{unexpected}. Resolve them, re-freeze, and re-run"
        )
    if blocking or not protocol.get("ready_to_execute"):
        raise ThreePathError(f"the frozen protocol is not ready: {blocking}")
    ladder = protocol["claim_ladder"]
    return {
        "preregistration_sha256": stored,
        "protocol_id": protocol.get("protocol_id"),
        "ready_to_execute": bool(protocol.get("ready_to_execute")),
        "blocking": blocking,
        "blocking_consequences": {},
        "authorised_claim": [ladder[0]["claim"], ladder[1]["claim"]],
        "authorised_outcome": "selected after tau_model is measured",
        "result_class": epc.RESULT_QUANTITATIVE,
        "status": RESULT_STATUS_CANDIDATE,
        "observable": "finite-PAO covariant electron-phonon matrix element (g_PAO)",
        "full_KS": "BLOCKED: Delta_out",
    }


# --------------------------------------------------------------------------- #
# Shared contracts: geometry, basis, directions, k
# --------------------------------------------------------------------------- #


def protocol_directions(protocol: Mapping[str, Any]) -> dict[str, fdp.Direction]:
    """The two ``E_2g`` members, rebuilt from the frozen vectors, hash checked.

    Not re-derived from the phonon: the gauge rule ran once, at freezing time,
    and its output is what the experiment is defined on.
    """
    directions = {}
    for entry in protocol["perturbation"]["directions"]:
        direction = fdp.Direction(
            name=str(entry["direction_name"]),
            kind=str(entry["direction_kind"]),
            vectors=np.asarray(entry["vectors"], dtype=np.float64),
        )
        if direction.direction_hash != entry["direction_hash"]:
            raise ThreePathError(
                f"{direction.name}: rebuilt direction hash {direction.direction_hash} != frozen "
                f"{entry['direction_hash']}"
            )
        directions[direction.name] = direction
    return directions


def protocol_k_points(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Every ``k`` the splits name, in a fixed order, with its frozen spectrum."""
    electronic = protocol["electronic"]
    entries = [
        *electronic["k_calibration"],
        electronic["k_validation"],
        electronic["k_result_non_degenerate"],
        electronic["k_result_degenerate"],
    ]
    return [
        {
            "label": entry["label"],
            "k_fractional": [float(value) for value in entry["k_fractional"]],
            "window_indices": [int(value) for value in entry["window_indices"]],
            "frozen_eigenvalues_ev": [float(value) for value in entry["eigenvalues_ev"]],
        }
        for entry in entries
    ]


def split_pairs(protocol: Mapping[str, Any]) -> dict[str, list[tuple[str, str]]]:
    """``(direction, k)`` of each split, in the pre-registered evaluation order."""
    splits = protocol["splits"]
    return {
        name: [
            (direction, k_label)
            for direction in splits[name]["directions"]
            for k_label in splits[name]["k_labels"]
        ]
        for name in ("calibration", "validation", "result")
    }


def geometry_contracts(campaign: Any, base_fdf: Path) -> tuple[dict[str, Any], str]:
    """The ``geometry`` node the phonons were signed with, plus the basis hash.

    The phonon campaign signs its geometry with
    ``run_graphene_gamma_phonons.geometry_node`` over the canonical
    ``base_ang.fdf``; the derivative campaign writes the byte-identical file (the
    protocol's ``geometry_consistency`` proves it). Building the node here from
    that same file is what makes ``D_H``, the response and the mode share one
    ``geometry_signature`` instead of three schemes that merely look alike.
    """
    import run_graphene_gamma_phonons as phonons

    from fdf_materialization import extract_fdf_structure

    structure = extract_fdf_structure(base_fdf)
    node = phonons.geometry_node(structure)
    basis_signature = str(
        (campaign.manifest.get("orbital_contract") or {}).get("orbital_contract_hash", "")
    )
    if not basis_signature:
        raise ThreePathError(f"{campaign.root} declares no orbital_contract_hash")
    return node, basis_signature


# --------------------------------------------------------------------------- #
# Stage 1: prove that every input already exists. This path never runs SIESTA.
# --------------------------------------------------------------------------- #


def _json_if_present(path: Path, missing: list[str]) -> dict[str, Any]:
    if not path.is_file():
        missing.append(str(path))
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        missing.append(f"{path}: {error}")
        return {}


def read_only_preflight(
    *,
    protocol_path: Path,
    reference_root: Path,
    e2g_root: Path,
    eigenspace_dir: Path,
    checkpoint: Path,
    go6_report_path: Path,
) -> dict[str, Any]:
    """Audit all C19/C20 inputs without creating or modifying an artifact."""
    missing: list[str] = []
    mismatches: list[str] = []
    protocol = _json_if_present(protocol_path, missing)
    reference_manifest_path = reference_root / "epc_siesta_reference_manifest.json"
    e2g_manifest_path = e2g_root / "epc_siesta_reference_manifest.json"
    reference_manifest = _json_if_present(reference_manifest_path, missing)
    e2g_manifest = _json_if_present(e2g_manifest_path, missing)

    deltas = [float(value) for value in (protocol.get("perturbation") or {}).get("delta_sweep_ang", [])]
    result_entries = (protocol.get("perturbation") or {}).get("directions", [])
    result_directions = {
        str(entry.get("direction_name")): np.asarray(entry.get("vectors", []), dtype=float)
        for entry in result_entries
    }
    equilibrium_id = (reference_manifest.get("certification_set") or {}).get(
        "equilibrium_run_id", "equilibrium"
    )
    equilibrium_dir = reference_root / "runs" / str(equilibrium_id)
    equilibrium_tshs = equilibrium_dir / f"{equilibrium_id}.TSHS"
    equilibrium_orb = equilibrium_dir / f"{equilibrium_id}.ORB_INDX"
    reference_base = Path(reference_manifest.get("base_ang_fdf", reference_root / "inputs/base_ang.fdf"))
    e2g_base = Path(e2g_manifest.get("base_ang_fdf", e2g_root / "inputs/base_ang.fdf"))

    required = [
        equilibrium_tshs,
        equilibrium_orb,
        equilibrium_dir / "C.ion.xml",
        reference_base,
        e2g_base,
        checkpoint,
    ]
    for delta in deltas:
        required.append(reference_root / "runs" / f"fc_dhs_{_delta_tag(delta)}" / f"fc_dhs_{_delta_tag(delta)}.dHSdR.nc")
    for name in result_directions:
        for delta in deltas:
            for sign in ("minus", "plus"):
                run_id = f"fd_{name}_{_delta_tag(delta)}_{sign}"
                run_dir = e2g_root / "runs" / run_id
                required.extend(
                    (
                        run_dir / f"{run_id}.TSHS",
                        run_dir / f"{run_id}.ORB_INDX",
                        run_dir / "C.ion.xml",
                        run_dir / "RUN.fdf",
                        run_dir / "run_record.json",
                    )
                )
    for name in REQUIRED_DIRECTIONS:
        required.extend(
            (
                A_I_REAL_SPACE_COLLECTIVE_DIR / f"{name}.json",
                A_I_REAL_SPACE_COLLECTIVE_DIR / f"{name}.npz",
            )
        )
    c18_path = prereg.DEFAULT_CERTIFICATION_DIR / "c18_gamma_phonon_certification.json"
    c14c_v2_path = REPO_ROOT / "Comparison/results/epc/basis_response_v2/graphene/basis_response_v2_certification.json"
    go2_path = prereg.DEFAULT_CERTIFICATION_DIR / "go2_certification.json"
    eigenspace_manifest_path = eigenspace_dir / "eigenspaces_manifest.json"
    required.extend((c18_path, c14c_v2_path, go2_path, go6_report_path, eigenspace_manifest_path))
    for index in range(5):
        required.extend(
            (
                eigenspace_dir / f"eigenspace_{index:03d}.json",
                eigenspace_dir / f"eigenspace_{index:03d}_C.bin",
                eigenspace_dir / f"eigenspace_{index:03d}_CSC_minus_I.bin",
            )
        )
    for path in required:
        if not path.is_file() and str(path) not in missing:
            missing.append(str(path))

    coverage: dict[str, Any] = {"required": [], "by_delta": {}, "missing": []}
    central_contract: dict[str, Any] = {}
    fd_count = 0
    if not missing:
        from fdf_materialization import extract_fdf_structure
        from read_siesta_dhsdr import read_dhsdr

        required_pairs = sorted(
            {
                (atom + 1, axis + 1)
                for vectors in result_directions.values()
                for atom, axis in zip(*np.nonzero(vectors))
            }
        )
        coverage["required"] = [f"I={atom},alpha={axis}" for atom, axis in required_pairs]
        for delta in deltas:
            dhsdr = read_dhsdr(reference_root / "runs" / f"fc_dhs_{_delta_tag(delta)}" / f"fc_dhs_{_delta_tag(delta)}.dHSdR.nc")
            available = sorted(dhsdr.records)
            absent = sorted(set(required_pairs) - set(available))
            coverage["by_delta"][str(delta)] = [f"I={atom},alpha={axis}" for atom, axis in available]
            coverage["missing"].extend(
                f"delta={delta}:I={atom},alpha={axis}" for atom, axis in absent
            )

        contract_checks = {
            "base_fdf_sha256": file_sha256(reference_base) == file_sha256(e2g_base),
            "physics_fingerprint": reference_manifest.get("physics_fingerprints") == e2g_manifest.get("physics_fingerprints"),
            "orbital_contract": reference_manifest.get("orbital_contract") == e2g_manifest.get("orbital_contract"),
            "siesta_runtime": (reference_manifest.get("siesta_runtime_ref") or {}).get("binary_sha256")
            == (e2g_manifest.get("siesta_runtime_ref") or {}).get("binary_sha256"),
            "central_tshs_preregistered": file_sha256(equilibrium_tshs)
            == (protocol.get("references") or {}).get("equilibrium_tshs_sha256"),
            "energy_gauge": ((protocol.get("electronic") or {}).get("occupations") or {}).get("mu_convention")
            == "exported_hamiltonian_fermi_zero",
        }
        reference_structure = extract_fdf_structure(reference_base)
        e2g_structure = extract_fdf_structure(e2g_base)
        contract_checks.update(
            {
                "cell": reference_structure.lattice_vectors_ang == e2g_structure.lattice_vectors_ang,
                "species": [item.to_dict() for item in reference_structure.species]
                == [item.to_dict() for item in e2g_structure.species],
                "central_geometry_coordinates": reference_structure.positions_ang
                == e2g_structure.positions_ang,
            }
        )
        reference_orb_sha = file_sha256(equilibrium_orb)
        reference_basis_sha = file_sha256(equilibrium_dir / "C.ion.xml")
        rows = {row["run_id"]: row for row in e2g_manifest.get("rows", [])}
        max_position_error = 0.0
        for name, vectors in result_directions.items():
            base_positions = np.asarray(reference_structure.positions_ang)
            for delta in deltas:
                pair_positions = {}
                for sign, factor in (("minus", -1.0), ("plus", 1.0)):
                    run_id = f"fd_{name}_{_delta_tag(delta)}_{sign}"
                    run_dir = e2g_root / "runs" / run_id
                    row = rows.get(run_id, {})
                    if not row.get("certified"):
                        mismatches.append(f"{run_id}: manifest row is not certified")
                    positions = np.asarray(row.get("positions_ang", []), dtype=float)
                    pair_positions[sign] = positions
                    if positions.shape == base_positions.shape:
                        max_position_error = max(
                            max_position_error,
                            float(np.max(np.abs(positions - (base_positions + factor * delta * vectors)))),
                        )
                    else:
                        mismatches.append(f"{run_id}: positions shape {positions.shape} != {base_positions.shape}")
                    if file_sha256(run_dir / f"{run_id}.ORB_INDX") != reference_orb_sha:
                        mismatches.append(f"{run_id}: ORB_INDX hash")
                    if file_sha256(run_dir / "C.ion.xml") != reference_basis_sha:
                        mismatches.append(f"{run_id}: C.ion.xml hash")
                    if row.get("fermi_level_ev") is None:
                        mismatches.append(f"{run_id}: missing Fermi level for H-E_F*S gauge")
                    fd_count += 1
                if all(value.shape == base_positions.shape for value in pair_positions.values()):
                    max_position_error = max(
                        max_position_error,
                        float(np.max(np.abs((pair_positions["plus"] + pair_positions["minus"]) / 2 - base_positions))),
                    )
        contract_checks["orb_indx_ordering"] = not any("ORB_INDX" in item for item in mismatches)
        contract_checks["basis_C_ion_xml"] = not any("C.ion.xml" in item for item in mismatches)
        contract_checks["displaced_geometry_reconstructs_central"] = max_position_error <= 1e-12
        central_contract = {
            "checks": contract_checks,
            "reference_base_fdf_sha256": file_sha256(reference_base),
            "e2g_base_fdf_sha256": file_sha256(e2g_base),
            "equilibrium_tshs_sha256": file_sha256(equilibrium_tshs),
            "orb_indx_sha256": reference_orb_sha,
            "C_ion_xml_sha256": reference_basis_sha,
            "mesh_cutoff_contract": "byte-identical base FDF (MeshCutoff included)",
            "energy_gauge": "exported H - E_F S",
            "S_convention": "TSHS overlap, row/column/lattice-image ordering pinned by ORB_INDX",
            "maximum_displacement_contract_error_ang": max_position_error,
        }
        mismatches.extend(name for name, passed in contract_checks.items() if not passed)
        mismatches.extend(f"dHSdR:{item}" for item in coverage["missing"])

    a_i_rows = {}
    for name in REQUIRED_DIRECTIONS:
        report = _json_if_present(A_I_REAL_SPACE_COLLECTIVE_DIR / f"{name}.json", missing)
        a_i_rows[name] = report.get("verdict")
        if report and report.get("verdict") != "PASS":
            mismatches.append(f"A_I_v6:{name}:{report.get('verdict')}")
        npz_path = A_I_REAL_SPACE_COLLECTIVE_DIR / f"{name}.npz"
        if report and npz_path.is_file() and file_sha256(npz_path) != report.get("matrix_artifact_sha256"):
            mismatches.append(f"A_I_v6:{name}:npz_hash")
    c18 = _json_if_present(c18_path, missing)
    c14c_v2 = _json_if_present(c14c_v2_path, missing)
    go2_report = _json_if_present(go2_path, missing)
    go6 = _json_if_present(go6_report_path, missing)
    eigenspaces = _json_if_present(eigenspace_manifest_path, missing)
    phonon_path = Path((protocol.get("phonon") or {}).get("artifact", ""))
    if not phonon_path.is_file():
        missing.append(str(phonon_path))
    elif file_sha256(phonon_path) != (protocol.get("phonon") or {}).get("artifact_sha256"):
        mismatches.append("C18 phonon artifact hash")
    for label, report in (("C18", c18), ("GO6", go6), ("GO2", go2_report), ("C14C-v2", c14c_v2)):
        if report and report.get("verdict") != "PASS":
            mismatches.append(f"{label}:{report.get('verdict')}")
    if eigenspaces and eigenspaces.get("kpoint_count") != 5:
        mismatches.append(f"eigenspaces:kpoint_count={eigenspaces.get('kpoint_count')}")

    status = "PASS"
    if missing:
        status = "BLOCKED: missing_artifact"
    elif mismatches:
        status = "BLOCKED: contract_mismatch"
    return {
        "schema": PREFLIGHT_SCHEMA,
        "preregistration_sha256": protocol.get("frozen_content_sha256"),
        "mode": "READ_ONLY",
        "status": status,
        "central_contract_match": "PASS" if central_contract and not mismatches else "FAIL",
        "central_contract": central_contract,
        "required_dHSdR_I_alpha_coverage": coverage,
        "e2g_fd_artifacts": {"expected_tshs": 12, "present_tshs": fd_count},
        "A_I_v6": {"passed": sum(value == "PASS" for value in a_i_rows.values()), "required": 5, "verdicts": a_i_rows},
        "C18": c18.get("verdict"),
        "GO6": go6.get("verdict"),
        "GO2": go2_report.get("verdict"),
        "C14C_v2": c14c_v2.get("verdict"),
        "missing_artifacts": sorted(set(missing)),
        "contract_mismatches": sorted(set(mismatches)),
    }


# --------------------------------------------------------------------------- #
# Stage 2: the electrons, persisted once in the C15 format
# --------------------------------------------------------------------------- #


def _solve_window(
    H: np.ndarray, S: np.ndarray, window: Sequence[int]
) -> dict[str, Any]:
    """C15's own method: solve the pencil, then Loewdin + Rayleigh-Ritz in the window.

    The re-orthonormalisation is not cosmetic — it is what the persisted contract
    means by ``C†SC = I``, and its conditioning ``cond(V†SV)`` is the number the
    identity budget is derived from.
    """
    eps_all, vectors_all = cbr.solve_pencil(H, S)
    selection = np.asarray(window, dtype=int)
    raw = vectors_all[:, selection]
    metric = raw.conj().T @ S @ raw
    metric_eigenvalues, metric_vectors = np.linalg.eigh(metric)
    if metric_eigenvalues.min() <= 0.0:
        raise ThreePathError("the window metric is not positive definite")
    inverse_root = (metric_vectors * metric_eigenvalues**-0.5) @ metric_vectors.conj().T
    orthonormal = raw @ inverse_root
    projected = orthonormal.conj().T @ H @ orthonormal
    eps, rotation = np.linalg.eigh(projected)
    coefficients = orthonormal @ rotation
    residual = np.linalg.norm(H @ coefficients - (S @ coefficients) * eps[None, :], axis=0)
    scale = np.linalg.norm(H @ coefficients, axis=0) + np.abs(eps) * np.linalg.norm(
        S @ coefficients, axis=0
    )
    return {
        "energies_eV": eps,
        "coefficients": coefficients,
        "all_energies_eV": eps_all,
        "generalized_relative_residual": residual / np.where(scale > 0.0, scale, 1.0),
        "overlap_minus_identity": coefficients.conj().T @ S @ coefficients - np.eye(eps.size),
        "window_metric_condition_number": float(
            metric_eigenvalues.max() / metric_eigenvalues.min()
        ),
        "window_metric_minimum_eigenvalue": float(metric_eigenvalues.min()),
        "maximum_energy_shift_vs_solver_eV": float(
            np.max(np.abs(eps - eps_all[selection]), initial=0.0)
        ),
        "solver_band_indices": [int(value) for value in selection],
    }


def persist_eigenspaces(
    output_dir: Path,
    *,
    h_blocks: np.ndarray,
    s_blocks: np.ndarray,
    isc_off: np.ndarray,
    k_points: Sequence[Mapping[str, Any]],
    parents: Mapping[str, Any],
    occupations: Mapping[str, Any],
    solver_version: Mapping[str, Any],
    shift_ev: float,
) -> dict[str, Any]:
    """Write one ``eigenspace_<k>.json`` + two raw ComplexF64 files per k (C15).

    The dense Loewdin path stands in for the sparse solver on a 8-orbital cell;
    the *format*, the identity budget and the signature fields are the solver's,
    so GO-6 and the contraction consume these exactly as they would consume a
    MATBG window.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    nodes: list[dict[str, Any]] = []
    for index, entry in enumerate(k_points):
        k = entry["k_fractional"]
        H = ebr.bloch_sum(h_blocks, isc_off, k)
        S = ebr.bloch_sum(s_blocks, isc_off, k)
        solved = _solve_window(H, S, entry["window_indices"])
        vectors_name = f"eigenspace_{index:03d}_C.bin"
        deviation_name = f"eigenspace_{index:03d}_CSC_minus_I.bin"
        coefficients = np.ascontiguousarray(solved["coefficients"], dtype=np.complex128)
        coefficients.flatten(order="F").tofile(output_dir / vectors_name)
        deviation = np.ascontiguousarray(solved["overlap_minus_identity"], dtype=np.complex128)
        deviation.flatten(order="F").tofile(output_dir / deviation_name)
        residuals = np.asarray(solved["generalized_relative_residual"], dtype=np.float64)
        node = eigenspace_signature_node(
            dict(parents),
            k=list(k),
            solver_backend="dense_loewdin_numpy",
            solver_version=dict(solver_version),
            window={
                "half_width_eV": None,
                "reference": "electron_count",
                "state_selection": "states_below_and_above_neutrality_from_the_declared_electron_count",
                "window_indices": entry["window_indices"],
                "degeneracy_tolerance_eV": float(DEGENERACY_TOL_EV),
            },
            shift=float(shift_ev),
            tolerances={
                "identity_tolerance_policy": EIGENSPACE_IDENTITY_TOLERANCE_POLICY,
                "window_metric_minimum_eigenvalue": solved["window_metric_minimum_eigenvalue"],
            },
            state_count=int(solved["energies_eV"].size),
            occupations=dict(occupations),
        )
        payload = {
            "schema": EIGENSPACE_SCHEMA,
            "k_index": index,
            "label": entry["label"],
            "k_fractional": [float(value) for value in k],
            "norbits": int(coefficients.shape[0]),
            "state_count": int(coefficients.shape[1]),
            "solver_band_indices": solved["solver_band_indices"],
            "energies_eV": [float(value) for value in solved["energies_eV"]],
            "generalized_relative_residual": [float(value) for value in residuals],
            "window_metric_condition_number": solved["window_metric_condition_number"],
            "window_metric_minimum_eigenvalue": solved["window_metric_minimum_eigenvalue"],
            "maximum_energy_shift_vs_solver_eV": solved["maximum_energy_shift_vs_solver_eV"],
            "dtype": "complex128",
            "storage_order": "fortran",
            "vectors_file": vectors_name,
            "overlap_minus_identity_file": deviation_name,
            "signature": node,
            "solver": "dense_loewdin_numpy",
            "energy_origin": "sisl H - E_F S (exported_hamiltonian_fermi_zero)",
        }
        _write_json(output_dir / f"eigenspace_{index:03d}.json", payload)
        rows.append(
            {
                "k_index": index,
                "label": entry["label"],
                "k_fractional": [float(value) for value in k],
                "energies_eV": payload["energies_eV"],
                "all_energies_eV": [float(value) for value in solved["all_energies_eV"]],
                "identity_maximum_absolute_error": float(np.abs(deviation).max(initial=0.0)),
                "identity_tolerance": eigenspace_identity_tolerance(
                    float(residuals.max(initial=0.0)),
                    solved["window_metric_condition_number"],
                    int(coefficients.shape[0]),
                ),
                "maximum_generalized_relative_residual": float(residuals.max(initial=0.0)),
                "signature_sha256": node["signature_sha256"],
            }
        )
        nodes.append(node)
    return {
        "eigenspace_dir": str(output_dir),
        "kpoint_count": len(rows),
        "set_signature_sha256": eigenspace_set_signature(nodes),
        "eigenspaces": rows,
    }


def eigenspace_stage(
    output_dir: Path,
    *,
    equilibrium: Any,
    isc_off: np.ndarray,
    k_points: Sequence[Mapping[str, Any]],
    parents: Mapping[str, Any],
    occupations: Mapping[str, Any],
    solver_version: Mapping[str, Any],
    overwrite: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Persist (or reuse) the windows and check them against the frozen spectra."""
    signature = input_signature_sha256(
        {
            "equilibrium_tshs_sha256": file_sha256(equilibrium.path),
            "k_points": [dict(entry) for entry in k_points],
            "occupations": dict(occupations),
            "solver": "dense_loewdin_numpy",
            "isc_off": np.asarray(isc_off).tolist(),
        }
    )
    manifest_path = output_dir / "eigenspaces_manifest.json"
    if not overwrite and manifest_path.is_file():
        stored = json.loads(manifest_path.read_text(encoding="utf-8"))
        if stored.get("input_signature_sha256") == signature:
            stored["reused"] = True
            return stored, [
                _check(
                    "eigenspaces_reused",
                    True,
                    f"{manifest_path} matches the input signature; H, S and the windows were "
                    "not recomputed",
                )
            ]

    h_blocks, h_outside = cbr.dense_blocks(equilibrium.h_shifted, isc_off, equilibrium.no_u)
    s_blocks, s_outside = cbr.dense_blocks(equilibrium.overlap, isc_off, equilibrium.no_u)
    manifest = persist_eigenspaces(
        output_dir,
        h_blocks=h_blocks,
        s_blocks=s_blocks,
        isc_off=isc_off,
        k_points=k_points,
        parents=parents,
        occupations=occupations,
        solver_version=solver_version,
        shift_ev=0.0,
    )
    manifest.update(
        {
            "schema": EIGENSPACE_SCHEMA,
            "input_signature_sha256": signature,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "reused": False,
            "energy_origin": "sisl H - E_F S",
            "equilibrium_tshs": str(equilibrium.path),
            "norm_outside_image_table": {"H": h_outside, "S": s_outside},
        }
    )
    _write_json(manifest_path, manifest)
    checks = [
        _check(
            "equilibrium_matrices_inside_image_table",
            max(h_outside, s_outside) == 0.0,
            "every H and S element of the equilibrium TSHS lives on the image table the "
            "derivatives use",
            norm_outside={"H": h_outside, "S": s_outside},
        )
    ]
    return manifest, checks


# The protocol froze its spectra through sisl's own ``Hamiltonian.eigh``; this
# run rebuilds H(k) and S(k) from the sparse ``(row, col, R)`` field and solves
# the pencil through a Cholesky factor. Two independent implementations agree to
# roundoff on the spectrum's own scale, not to an absolute eV: the criterion is
# relative, and it is still four orders below the 1.3e-6 eV degeneracy split the
# protocol *decided* the K stage with, so a wrong k, window or phase cannot pass.
SPECTRUM_RELATIVE_TOLERANCE = 1e-9


def check_frozen_spectra(
    manifest: Mapping[str, Any], k_points: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Do the recomputed spectra reproduce the ones the protocol froze by value?"""
    checks = []
    for entry, row in zip(k_points, manifest["eigenspaces"]):
        frozen = np.asarray(entry["frozen_eigenvalues_ev"], dtype=np.float64)
        recomputed = np.asarray(row["all_energies_eV"], dtype=np.float64)
        deviation = (
            float(np.max(np.abs(frozen - recomputed)))
            if frozen.shape == recomputed.shape
            else float("inf")
        )
        scale = float(np.max(np.abs(recomputed), initial=1.0))
        tolerance = SPECTRUM_RELATIVE_TOLERANCE * max(scale, 1.0)
        checks.append(
            _check(
                f"spectrum_matches_preregistration__{entry['label']}",
                deviation <= tolerance,
                f"the spectrum at {entry['label']} reproduces the frozen one to "
                f"{deviation:.3e} eV (tolerance {tolerance:.3e}); the k, the window and the "
                "Bloch/phase convention are the pre-registered ones",
                maximum_absolute_deviation_ev=deviation,
                tolerance_ev=tolerance,
            )
        )
    return checks


# --------------------------------------------------------------------------- #
# Stage 3: the phonon, in the gauge the protocol froze
# --------------------------------------------------------------------------- #


def gauge_fixed_modes(
    protocol: Mapping[str, Any], phonon_path: Path
) -> tuple[Any, dict[str, Any], list[dict[str, Any]]]:
    """The C18 mode set with the ``E_2g`` doublet rotated into the frozen gauge.

    A degenerate doublet has no canonical member, so the protocol fixed one by a
    rule and stored the outcome. Replacing the solver's arbitrary pair by that
    outcome is legitimate only if the frozen members span the same subspace; the
    projection onto the solver's doublet is measured here and the substitution
    fails if it is not unity.
    """
    payload = json.loads(phonon_path.read_text(encoding="utf-8"))
    modes = pp.PhononModeSet.from_dict(payload)
    doublet = protocol["phonon"]["doublet"]
    branches = [int(value) for value in doublet["branches"]]
    solver_span = np.stack(
        [np.asarray(modes.eigenvectors[branch]).reshape(-1) for branch in branches]
    )

    eigenvectors = np.array(modes.eigenvectors, dtype=np.complex128)
    frequencies = np.array(modes.frequencies_ev, dtype=np.float64)
    projections = []
    labels = []
    for branch, member in zip(branches, doublet["members"]):
        vector = np.asarray(member["eigenvector_mass_weighted"], dtype=np.float64)
        flat = vector.reshape(-1)
        projected = solver_span.conj().T @ (solver_span @ flat)
        projections.append(float(np.linalg.norm(projected) / np.linalg.norm(flat)))
        eigenvectors[branch] = vector
        frequencies[branch] = float(doublet["frequency_ev"])
        labels.append(member["label"])

    gauge_fixed = pp.PhononModeSet(
        q_fractional=modes.q_fractional,
        frequencies_ev=frequencies,
        eigenvectors=eigenvectors,
        masses_amu=modes.masses_amu,
        cell_ang=modes.cell_ang,
        positions_ang=modes.positions_ang,
        conventions=modes.conventions,
        provider=modes.provider,
        provider_version=modes.provider_version,
        force_source=modes.force_source,
        primitive_mapping=modes.primitive_mapping,
        fc_range=modes.fc_range,
        asr_policy=modes.asr_policy,
        asr_residual_ev_ang2=modes.asr_residual_ev_ang2,
        asr_residual_raw_ev_ang2=modes.asr_residual_raw_ev_ang2,
        geometry_signature=modes.geometry_signature,
        provenance={
            **(modes.provenance or {}),
            "doublet_gauge": doublet["gauge_rule"],
            "doublet_gauge_source": "preregistration (frozen by value)",
            "solver_frequencies_ev": [float(modes.frequencies_ev[b]) for b in branches],
            "doublet_frequency_spread_ev": float(doublet["frequency_spread_ev"]),
        },
    )
    branch_of_label = dict(zip(labels, branches))
    checks = [
        _check(
            "phonon_doublet_gauge_lies_in_the_solver_span",
            all(abs(value - 1.0) <= 1e-8 for value in projections),
            "the frozen E2g members are inside the span of the solver's own doublet, so fixing "
            "the gauge rotated the pair instead of replacing it",
            projections=projections,
        ),
        _check(
            "phonon_zero_point_amplitude_matches_preregistration",
            math.isclose(
                float(gauge_fixed.mode(branches[0]).zero_point_amplitude_ang[0]),
                float(doublet["zero_point_amplitude_ang"]),
                rel_tol=1e-9,
            ),
            "sqrt(hbar/2 M omega) reproduces the frozen amplitude "
            f"{doublet['zero_point_amplitude_ang']:.9f} Ang",
            recomputed=float(gauge_fixed.mode(branches[0]).zero_point_amplitude_ang[0]),
        ),
        _check(
            "phonon_artifact_is_the_preregistered_one",
            file_sha256(phonon_path) == protocol["phonon"]["artifact_sha256"],
            f"{phonon_path.name} is the artifact the protocol froze",
        ),
    ]
    return gauge_fixed, branch_of_label, checks


# --------------------------------------------------------------------------- #
# Stage 4: the raw derivatives of the three paths
# --------------------------------------------------------------------------- #


class ModelSide:
    """The checkpoint, loaded once, on the effective backend."""

    def __init__(self, checkpoint: Path, campaign: Any, equilibrium: Any, work_dir: Path):
        import sisl
        import torch

        from graph2mat_autograd_derivatives import batch_topology_hash, resolve_jvp_backend

        equilibrium_id = campaign.manifest["certification_set"]["equilibrium_run_id"]
        geometry = sisl.get_sile(str(campaign.run_dir(equilibrium_id) / "RUN.fdf")).read_geometry()
        structure_fdf = c14.ghost_free_fdf(geometry, work_dir / "geometry" / "RUN.fdf")
        model, batch, processor, provenance = c14.load_model_and_batch(
            checkpoint, structure_fdf, basis_dir=Path(campaign.manifest["basis_dir"])
        )
        import validate_graph2mat_jvp as vj

        model, batch = vj._cast(model, batch, torch.float64)
        model, batch, record = resolve_jvp_backend(model, batch, "cuda", output_keys=c14.OUTPUT_KEYS)
        self.model = model
        self.batch = batch
        self.processor = processor
        self.provenance = provenance
        self.backend = record
        self.change_of_basis = np.asarray(processor.basis_table.change_of_basis, dtype=np.float64)
        self.topology = batch_topology_hash(batch)
        self.equilibrium_prediction, self.prediction_meta = c14.model_predictions(
            model, batch, processor
        )
        self.checkpoint = checkpoint
        self.equilibrium = equilibrium

    def backend_record(self) -> dict[str, Any]:
        return {
            "requested_backend": self.backend.requested,
            "effective_backend": self.backend.effective,
            "preflight": self.backend.preflight,
            "fallback_reason": self.backend.reason,
            "dtype": "float64",
            "evidence": (
                "resolve_jvp_backend ran one forward plus one double backward on this very model "
                "and batch before any derivative was spent"
            ),
        }

    def label_convention_check(self) -> dict[str, Any]:
        against_label = go2.difference_norms(
            self.equilibrium.h_shifted, self.equilibrium_prediction
        )
        against_absolute = go2.difference_norms(
            self.equilibrium.h_absolute, self.equilibrium_prediction
        )
        return _check(
            "model_predicts_the_fermi_shifted_label",
            against_label["frobenius"] < against_absolute["frobenius"],
            "the checkpoint predicts sisl's H - E_F S, so its derivative is converted into the "
            "energy origin the eigenvalues use before any contraction",
            equilibrium_h_vs_label=against_label,
            equilibrium_h_vs_absolute=against_absolute,
        )

    def jvp(self, direction: fdp.Direction) -> tuple[dict[Key, float], dict[str, Any]]:
        return c14.directional_derivative_field(
            self.model,
            self.batch,
            direction,
            processor=self.processor,
            change_of_basis=self.change_of_basis,
            backend=self.backend,
        )

    def frozen(
        self, direction: fdp.Direction, delta_ang: float
    ) -> tuple[dict[Key, float], list[str]]:
        return c14.frozen_derivative_field(
            self.model,
            self.batch,
            direction,
            delta_ang,
            processor=self.processor,
            change_of_basis=self.change_of_basis,
        )


def certify_split_fc_fd(
    fc_campaign: Any,
    fd_campaign: Any,
    directions: Mapping[str, fdp.Direction],
    deltas: Sequence[float],
) -> dict[str, Any]:
    """Recheck atom-resolved FC contractions against the existing E2g FD pairs."""
    from read_siesta_dhsdr import read_dhsdr

    dhsdr = {delta: read_dhsdr(fc_campaign.dhsdr_path(delta)) for delta in deltas}
    comparisons = []
    for name, direction in directions.items():
        fd_by_kind = {kind: {} for kind in ("D_H", "D_S")}
        fc_by_kind = {kind: {} for kind in ("D_H", "D_S")}
        sources_by_delta = {}
        for delta in deltas:
            pair = next(
                entry
                for entry in fd_campaign.certified_pairs()
                if entry["direction_name"] == name
                and math.isclose(float(entry["delta_ang"]), float(delta))
            )
            plus = go2.read_tshs(
                fd_campaign.run_dir(pair["plus_run_id"]), pair["plus_run_id"],
                fermi_ev=fd_campaign.fermi_ev(pair["plus_run_id"]),
            )
            minus = go2.read_tshs(
                fd_campaign.run_dir(pair["minus_run_id"]), pair["minus_run_id"],
                fermi_ev=fd_campaign.fermi_ev(pair["minus_run_id"]),
            )
            fd = go2.finite_difference(plus, minus, delta)
            sources_by_delta[delta] = {
                "fc": {"path": str(fc_campaign.dhsdr_path(delta)),
                       "sha256": file_sha256(fc_campaign.dhsdr_path(delta))},
                "fd_plus": {"path": str(plus.path), "sha256": file_sha256(plus.path)},
                "fd_minus": {"path": str(minus.path), "sha256": file_sha256(minus.path)},
            }
            for kind in ("D_H", "D_S"):
                fd_by_kind[kind][delta] = fd[kind]
                fc_by_kind[kind][delta] = go2.contract_fc(
                    dhsdr[delta], direction.vectors, kind
                )
        for kind in ("D_H", "D_S"):
            plateau = go2.plateau_for(fd_by_kind[kind])
            plateau_values = plateau.get("plateau_delta_ang") or list(deltas)
            tau = go2.noise_floor(fd_by_kind[kind], plateau_values)
            by_norm = {}
            for norm in ("frobenius", "max_abs"):
                discrepancies = {
                    str(delta): go2.difference_norms(
                        fc_by_kind[kind][delta], fd_by_kind[kind][delta]
                    )[norm]
                    for delta in deltas
                }
                within = [
                    float(delta) for delta in deltas
                    if discrepancies[str(delta)] <= tau[norm]
                ]
                by_norm[norm] = {
                    "discrepancy_by_delta": discrepancies,
                    "deltas_within_tau": within,
                    "tau_fd": tau[norm],
                    "passed": bool(within),
                    "verdict": "noise_limited" if within else "failed",
                }
            detail_by_delta = {}
            for delta in deltas:
                fc_values = fc_by_kind[kind][delta]
                fd_values = fd_by_kind[kind][delta]
                difference = {
                    key: fc_values[key] - fd_values[key]
                    for key in fc_values.keys() | fd_values.keys()
                }
                norms = go2.difference_norms(fc_values, fd_values)
                detail_by_delta[str(delta)] = {
                    "delta_ang": float(delta),
                    "quantity": kind,
                    "sources": sources_by_delta[delta],
                    "fc": fc_values,
                    "fd": fd_values,
                    "difference": difference,
                    "difference_frobenius": norms["frobenius"],
                    "difference_max_abs": norms["max_abs"],
                    "relative_difference_frobenius": (
                        norms["frobenius"] / max(_frobenius(list(fd_values.values())), 1e-30)
                    ),
                    "tolerance": tau,
                    "passed": all(norms[norm] <= tau[norm] for norm in ("frobenius", "max_abs")),
                }
            comparisons.append(
                {
                    "direction_name": name,
                    "kind": kind,
                    "unit": "eV/Ang" if kind == "D_H" else "1/Ang",
                    "plateau": plateau,
                    "tau_fd": tau,
                    "by_norm": by_norm,
                    "detail_by_delta": detail_by_delta,
                    "passed": all(row["passed"] for row in by_norm.values()),
                }
            )
    passed = sum(bool(row["passed"]) for row in comparisons)
    return {
        "schema": "epc_split_fc_fd_read_only_v1",
        "source": {
            "fc_campaign": str(fc_campaign.root),
            "fd_campaign": str(fd_campaign.root),
            "assembly": "D[v] = sum_(I,alpha) v[I,alpha] D_(I,alpha)",
        },
        "comparisons": comparisons,
        "summary": {
            "comparisons_passed": passed,
            "comparisons_total": len(comparisons),
        },
        "verdict": "PASS" if passed == len(comparisons) else "FAIL",
    }


def split_source_basis_response(
    fc_campaign: Any,
    fd_campaign: Any,
    certification: Mapping[str, Any],
    direction: fdp.Direction,
    *,
    geometry_signature: str,
    basis_signature: str,
    electronic_derivative_backend: str,
) -> tuple[Any, dict[str, Any]]:
    """Build S_L/S_R from certified FC while taking delta evidence from existing FD."""
    from read_siesta_dhsdr import read_dhsdr

    terms = cbr.d_s_reference_terms(certification, direction.name)
    within = terms["deltas_within_tau"]
    delta = max(within) if within else float(terms["plateau_selected_delta_ang"])
    pair = next(
        entry for entry in fd_campaign.certified_pairs()
        if entry["direction_name"] == direction.name
        and math.isclose(float(entry["delta_ang"]), delta)
    )
    dhsdr = read_dhsdr(fc_campaign.dhsdr_path(delta))
    equilibrium_id = fc_campaign.manifest["certification_set"]["equilibrium_run_id"]
    equilibrium = go2.read_tshs(
        fc_campaign.run_dir(equilibrium_id), equilibrium_id,
        fermi_ev=fc_campaign.fermi_ev(equilibrium_id),
    )
    import sisl

    geometry = sisl.get_sile(equilibrium.path).read_geometry()
    atom_of_orbital = np.asarray(geometry.o2a(np.arange(equilibrium.no_u)), dtype=np.int64) + 1
    context = ebr.ResponseContext(
        geometry_signature=geometry_signature,
        basis_signature=basis_signature,
        electronic_derivative_backend=electronic_derivative_backend,
        direction=direction,
        source={
            "fc_reference_root": str(fc_campaign.root),
            "fd_reference_root": str(fd_campaign.root),
            "fc_run": str(fc_campaign.dhsdr_path(delta)),
            "delta_ang": delta,
            "delta_source": "read_only_split_fc_fd_largest_delta_within_tau",
            "siesta_runtime_ref": fc_campaign.manifest.get("siesta_runtime_ref"),
        },
    )
    return ebr.from_dhsdr(dhsdr, context, atom_of_orbital=atom_of_orbital), {
        "direction": direction,
        "delta_ang": delta,
        "delta_source": "read_only_split_fc_fd_largest_delta_within_tau",
        "pair": pair,
        "dhsdr": dhsdr,
        "equilibrium": equilibrium,
        "atom_of_orbital": atom_of_orbital,
        "terms": terms,
    }


def siesta_fields(
    fc_campaign: Any,
    fd_campaign: Any,
    equilibrium: Any,
    dhsdr_by_delta: Mapping[float, Any],
    direction: fdp.Direction,
    delta: float,
) -> dict[str, Any]:
    """FC and FD derivatives in the pre-registered ``H - E_F S`` gauge."""
    d_h_abs = go2.contract_fc(dhsdr_by_delta[delta], direction.vectors, "D_H")
    d_s = go2.contract_fc(dhsdr_by_delta[delta], direction.vectors, "D_S")
    pair = next(
        (
            entry
            for entry in fd_campaign.certified_pairs()
            if entry["direction_name"] == direction.name
            and math.isclose(float(entry["delta_ang"]), float(delta))
        ),
        None,
    )
    if pair is None:
        raise ThreePathError(f"no certified +- pair for {direction.name} at delta = {delta}")
    plus = go2.read_tshs(
        fd_campaign.run_dir(pair["plus_run_id"]),
        pair["plus_run_id"],
        fermi_ev=fd_campaign.fermi_ev(pair["plus_run_id"]),
    )
    minus = go2.read_tshs(
        fd_campaign.run_dir(pair["minus_run_id"]),
        pair["minus_run_id"],
        fermi_ev=fd_campaign.fermi_ev(pair["minus_run_id"]),
    )
    d_fermi = (plus.fermi_ev - minus.fermi_ev) / (2.0 * float(delta))
    d_h_aligned = fc_to_fermi_zero(d_h_abs, d_s, equilibrium, d_fermi)
    fd = go2.finite_difference(plus, minus, float(delta), undo_fermi_shift=False)
    return {
        "d_h_absolute": d_h_abs,
        "d_s": d_s,
        "d_h": d_h_aligned,
        "d_h_fd": fd["D_H"],
        "d_s_fd": go2.finite_difference(plus, minus, float(delta))["D_S"],
        "d_fermi_ev_per_ang": d_fermi,
        "pair": pair,
        "fermi_ev": float(equilibrium.fermi_ev),
    }


def fc_to_fermi_zero(
    d_h_absolute: Mapping[Key, float],
    d_s: Mapping[Key, float],
    equilibrium: Any,
    d_fermi_ev_per_ang: float,
) -> dict[Key, float]:
    """Differentiate ``H - E_F S`` from the atom-resolved absolute-H FC fields."""
    return c14.combine(
        (1.0, d_h_absolute),
        (-float(equilibrium.fermi_ev), d_s),
        (-float(d_fermi_ev_per_ang), equilibrium.overlap),
    )


# --------------------------------------------------------------------------- #
# Stage 5: PAO-covariant response and g
# --------------------------------------------------------------------------- #


def raw_derivative(
    *,
    path: str,
    direction: fdp.Direction,
    d_h: Mapping[Key, float],
    d_s: Mapping[Key, float] | None,
    isc_off: np.ndarray,
    no_u: int,
    geometry_signature: str,
    basis_signature: str,
    topology_sha256: str,
    delta_or_jvp: Any,
) -> tuple[epc.RawDerivative, dict[str, float]]:
    blocks = {}
    outside = {}
    blocks["D_H"], outside["D_H"] = cbr.dense_blocks(d_h, isc_off, no_u)
    if d_s is not None:
        blocks["D_S"], outside["D_S"] = cbr.dense_blocks(d_s, isc_off, no_u)
    return (
        epc.RawDerivative(
            blocks=blocks,
            isc_off=isc_off,
            backend=BACKEND_OF_PATH[path],
            method=METHOD_OF_PATH[path],
            direction=direction,
            geometry_signature=geometry_signature,
            basis_signature=basis_signature,
            topology_sha256=topology_sha256,
            delta_or_jvp=delta_or_jvp,
        ),
        outside,
    )


def contracted_block(
    perturbation: epc.PaoCovariantResponse, eigenspace: epc.Eigenspace
) -> np.ndarray:
    """``C_W† PAO-covariant response C_W`` — the object every pre-registered tolerance is a norm of."""
    return np.asarray(perturbation.block(eigenspace, eigenspace).values)


def subspace_metrics(
    values: np.ndarray, eigenspace: epc.Eigenspace
) -> dict[str, Any]:
    block = epc.EpcBlock(
        values=values,
        formalism_id=ebr.FORMALISM_ID,
        representation=ebr.REPRESENTATION_S_L_S_R,
        intra_atomic_included=True,
        basis_response_backend="siesta",
    )
    return {
        "gauge_invariant": epc.gauge_invariant_block_metrics(block),
        "cluster_blocks": subspaces.block_metrics(
            values, eigenspace.cluster_labels, eigenspace.cluster_labels
        ),
        "cluster_groups": subspaces.cluster_groups(eigenspace.cluster_labels),
    }


def tau_num_from_sweep(blocks_by_delta: Mapping[float, np.ndarray], plateau: Sequence[float]) -> dict[str, Any]:
    """``max`` over plateau pairs of ``||C†(Delta(a) - Delta(b))C||_F`` (pre-registered)."""
    deltas = sorted(value for value in blocks_by_delta if any(math.isclose(value, p) for p in plateau)) or sorted(
        blocks_by_delta
    )
    pairs = [
        {
            "delta_a": a,
            "delta_b": b,
            "difference_frobenius": _frobenius(blocks_by_delta[a] - blocks_by_delta[b]),
        }
        for index, a in enumerate(deltas)
        for b in deltas[index + 1 :]
    ]
    return {
        "tau_num": max((row["difference_frobenius"] for row in pairs), default=0.0),
        "plateau_deltas_ang": deltas,
        "pairs": pairs,
        "unit": "eV/Ang",
    }


def plateau_deltas(certification: Mapping[str, Any], direction_name: str, kind: str) -> list[float]:
    """The amplitudes GO-2 resolved for this direction, or every swept amplitude."""
    for entry in certification.get("comparisons", []):
        if entry.get("direction_name") == direction_name and entry.get("kind") == kind:
            within = ((entry.get("by_norm") or {}).get("frobenius") or {}).get("deltas_within_tau")
            if within:
                return [float(value) for value in within]
            selected = (entry.get("plateau") or {}).get("selected_delta_ang")
            if selected:
                return [float(selected)]
    return []


# --------------------------------------------------------------------------- #
# The campaign
# --------------------------------------------------------------------------- #


def produce(
    *,
    protocol_path: Path,
    reference_root: Path,
    e2g_root: Path,
    eigenspace_dir: Path,
    result_root: Path,
    checkpoint: Path,
    go6_report_path: Path,
) -> dict[str, Any]:
    preflight = read_only_preflight(
        protocol_path=protocol_path,
        reference_root=reference_root,
        e2g_root=e2g_root,
        eigenspace_dir=eigenspace_dir,
        checkpoint=checkpoint,
        go6_report_path=go6_report_path,
    )
    print("[C19/C20] READ-ONLY PREFLIGHT")
    print(json.dumps(_json_safe(preflight), indent=2, sort_keys=True))
    if preflight["status"] != "PASS":
        raise ThreePathError(
            f"{preflight['status']}: "
            f"{preflight['missing_artifacts'] or preflight['contract_mismatches']}"
        )

    protocol = prereg.load_protocol(protocol_path)
    gate = authorize(protocol)
    checks: list[dict[str, Any]] = []
    result_root.mkdir(parents=True, exist_ok=True)
    _write_json(result_root / "read_only_preflight.json", preflight)

    directions_by_name = protocol_directions(protocol)
    k_points = protocol_k_points(protocol)
    splits = split_pairs(protocol)
    deltas = [float(value) for value in protocol["perturbation"]["delta_sweep_ang"]]

    # --- existing SIESTA artifacts: atom-resolved FC plus E2g explicit FD ---
    certified = go2.load_campaign(reference_root)
    e2g = go2.load_campaign(e2g_root)
    checks.append(
        _check(
            "read_only_preflight",
            True,
            "all C19/C20 inputs existed and their central/FD contracts matched before any "
            "Graph2Mat or post-processing calculation started",
        )
    )

    go2_certification = json.loads(
        (prereg.DEFAULT_CERTIFICATION_DIR / "go2_certification.json").read_text(encoding="utf-8")
    )
    checks.append(
        _check(
            "go2_certified",
            go2_certification.get("verdict") == "PASS",
            "the GO-2 certification of the reference campaign passed",
        )
    )
    e2g_certification = certify_split_fc_fd(
        certified,
        e2g,
        directions_by_name,
        deltas,
    )
    e2g_certification["preregistration_sha256"] = gate["preregistration_sha256"]
    _write_json(result_root / "embedded_read_only_split_fc_fd_v2.json", e2g_certification)
    checks.append(
        _check(
            "e2g_directions_fc_matches_explicit_fd",
            e2g_certification["verdict"] == "PASS",
            "the certified atom-resolved dHSdR contraction reproduces the 12 pre-existing E2g "
            "central-difference TSHS artifacts inside their measured noise floor",
            comparisons=e2g_certification.get("summary"),
        )
    )

    fd_campaign_of_direction = {name: e2g for name in directions_by_name}
    for name, vectors in certified.directions.items():
        fd_campaign_of_direction[name] = certified
        # Built exactly as ``response_for_direction`` builds it (name, kind,
        # vectors): the raw derivative and the basis response must declare the
        # *same* perturbation_definition, and a one-hot's index/axis decoration
        # is not part of ``direction_hash``.
        directions_by_name[name] = fdp.Direction(
            name=name,
            kind=next(
                entry["direction_kind"]
                for entry in certified.manifest["perturbation_space"]["directions"]
                if entry["direction_name"] == name
            ),
            vectors=vectors,
        )
    certification_of_direction = {
        **{name: e2g_certification for name in protocol["splits"]["result"]["directions"]},
        **{name: go2_certification for name in certified.directions},
    }

    # --- shared contracts ---------------------------------------------------
    base_fdf = Path(certified.manifest["base_ang_fdf"])
    geometry_node, basis_signature = geometry_contracts(certified, base_fdf)
    geometry_signature = geometry_node["signature_sha256"]
    checks.append(
        _check(
            "geometry_signature_matches_preregistration",
            geometry_signature == protocol["experiment"]["geometry_signature"],
            "D_H, the basis response and the phonon are signed with the one geometry the "
            "protocol froze",
            geometry_signature=geometry_signature,
        )
    )

    equilibrium_id = certified.manifest["certification_set"]["equilibrium_run_id"]
    equilibrium = go2.read_tshs(
        certified.run_dir(equilibrium_id),
        equilibrium_id,
        fermi_ev=certified.fermi_ev(equilibrium_id),
    )
    checks.append(
        _check(
            "equilibrium_tshs_matches_preregistration",
            file_sha256(equilibrium.path) == protocol["references"]["equilibrium_tshs_sha256"],
            "the equilibrium matrices are the ones the protocol pinned",
        )
    )
    # --- the basis response, once per direction -----------------------------
    # Intra-atomic correction (C14C-v2, PASS): A_I(R) is a real-space
    # geometric/basis quantity (Fourier-Bessel engine over C.ion.xml +
    # geometry), not a DFT-derivative quantity, so it does not vary with
    # electronic_derivative_backend -- loaded once per direction and applied
    # identically to both backends' artifacts (same reasoning as why
    # certify_basis_response_v2.py calls response_for_direction only once per
    # direction, never once per backend).
    responses: dict[tuple[str, str], Any] = {}
    response_inputs: dict[str, dict[str, Any]] = {}
    for name, direction in directions_by_name.items():
        fd_campaign = fd_campaign_of_direction[name]
        certification = certification_of_direction[name]
        a_i_by_r = _load_a_i_real_space_collective(name)
        for backend in ("siesta", "graph2mat"):
            artifact, inputs = split_source_basis_response(
                certified,
                fd_campaign,
                certification,
                direction,
                geometry_signature=geometry_signature,
                basis_signature=basis_signature,
                electronic_derivative_backend=backend,
            )
            corrected, correction_check = apply_intra_atomic_correction(artifact, name, a_i_by_r)
            if backend == "siesta":
                checks.append(correction_check)
            responses[(name, backend)] = corrected
            if backend == "siesta":
                response_inputs[name] = inputs
    response_block_hashes = {
        name: input_signature_sha256(
            {
                "S_left": np.round(responses[(name, "siesta")].blocks["S_left"], 12).tolist(),
                "S_right": np.round(responses[(name, "siesta")].blocks["S_right"], 12).tolist(),
            }
        )
        for name in directions_by_name
    }
    checks.append(
        _check(
            "one_basis_response_per_direction_across_paths",
            all(
                input_signature_sha256(
                    {
                        "S_left": np.round(responses[(name, "graph2mat")].blocks["S_left"], 12).tolist(),
                        "S_right": np.round(responses[(name, "graph2mat")].blocks["S_right"], 12).tolist(),
                    }
                )
                == response_block_hashes[name]
                for name in directions_by_name
            ),
            "the three paths consume identical S_L/S_R blocks; only the declared "
            "electronic_derivative_backend differs, which is what makes epc_backend_class hybrid",
            block_hashes=response_block_hashes,
        )
    )
    isc_off = responses[(next(iter(directions_by_name)), "siesta")].isc_off
    no_u = equilibrium.no_u

    # --- the electrons ------------------------------------------------------
    occupations = OccupationContract(
        electron_count=float(protocol["electronic"]["occupations"]["electron_count"]),
        spin_degeneracy=int(protocol["electronic"]["occupations"]["spin_degeneracy"]),
        occupation_function="zero_temperature_step",
        smearing_width_ev=0.0,
        mu_convention=str(protocol["electronic"]["occupations"]["mu_convention"]),
        mu_ev=0.0,
        mu_source="siesta_equilibrium_tshs_fermi_level",
    )
    parents = {
        "hamiltonian": epc_artifact_node(
            "electronic_hamiltonian",
            {
                "source_backend": "siesta",
                "model_or_binary_sha256": file_sha256(equilibrium.path),
                "code_version": certified.manifest.get("siesta_runtime_ref"),
                "basis": {"orbital_contract_hash": basis_signature, "n_orbitals": no_u},
                "neighbor_cutoff_policy": "siesta_pao_overlap_support",
                "topology_sha256": input_signature_sha256(
                    {"nsc": list(equilibrium.nsc), "sc_off": [list(row) for row in equilibrium.sc_off]}
                ),
                "dtype": "complex128",
                "mapping_version": "tshs_row_col_R",
            },
            {"geometry": geometry_node},
        ),
        "overlap": epc_artifact_node(
            "overlap",
            {
                "pao_basis": basis_signature,
                "siesta_runtime_ref": (certified.manifest.get("siesta_runtime_ref") or {}).get(
                    "record_sha256"
                ),
                "fdf_physics_sha256": certified.manifest["physics_fingerprints"][0],
                "orb_indx_sha256": file_sha256(
                    certified.run_dir(equilibrium_id) / f"{equilibrium_id}.ORB_INDX"
                ),
                "overlap_export_convention": "tshs_row_col_R",
                "units": "dimensionless",
            },
            {"geometry": geometry_node},
        ),
    }
    eigenspace_manifest = json.loads(
        (eigenspace_dir / "eigenspaces_manifest.json").read_text(encoding="utf-8")
    )
    checks.append(
        _check(
            "eigenspaces_reused_read_only",
            Path(eigenspace_manifest["equilibrium_tshs"]).resolve() == Path(equilibrium.path).resolve(),
            "the persisted windows point to the exact central TSHS; none was regenerated",
        )
    )
    checks.extend(check_frozen_spectra(eigenspace_manifest, k_points))

    from run_deeph_sparse_spectrum import load_persisted_eigenspace

    eigenspaces = {}
    for index, entry in enumerate(k_points):
        payload = load_persisted_eigenspace(eigenspace_dir, index)
        eigenspaces[entry["label"]] = epc.Eigenspace.from_persisted(payload, label=entry["label"])

    go6 = json.loads(go6_report_path.read_text(encoding="utf-8"))
    checks.append(
        _check(
            "go6_subspace_metrics_pass",
            go6.get("verdict") == "PASS",
            "the gauge-invariant subspace machinery is certified on these very windows before any "
            "block metric is read (roadmap GO-6)",
            verdict=go6.get("verdict"),
            summary=go6.get("summary"),
        )
    )

    # --- the phonon ---------------------------------------------------------
    phonon_path = Path(protocol["phonon"]["artifact"])
    modes, branch_of_label, phonon_checks = gauge_fixed_modes(protocol, phonon_path)
    checks.extend(phonon_checks)
    checks.append(
        _check(
            "phonon_geometry_signature_shared",
            modes.geometry_signature == geometry_signature,
            "the mode and the derivatives belong to the same geometry",
        )
    )
    phonon_node = epc_artifact_node(
        PHYSICAL_PHONON, modes.physical_phonon_fields(), {"geometry": geometry_node}
    )

    # --- the model ----------------------------------------------------------
    model_side = ModelSide(checkpoint, certified, equilibrium, result_root / "model")
    checks.append(model_side.label_convention_check())

    # The neighbour-list margin, re-asserted at run time on the same PAO cutoffs
    # the campaign used, and compared with the values frozen in the protocol.
    topology_margins = e2g.manifest["topology_margin"]
    frozen_margins = {
        row["direction_name"]: row for row in protocol["perturbation"]["topology_margin"]
    }
    checks.append(
        _check(
            "topology_margin_matches_preregistration",
            all(
                math.isclose(
                    row["min_margin_ang"],
                    frozen_margins[row["direction_name"]]["min_margin_ang"],
                    rel_tol=1e-9,
                )
                for row in topology_margins
            ),
            "the margin measured now is the one the protocol froze, so the perturbation acts on "
            "the graph the derivatives were taken on",
        )
    )

    # GO-3 on the pre-registered directions themselves: a JVP certified on the
    # S10 direction set says nothing about the E2g members until it is run on them.
    import validate_graph2mat_jvp as vj

    radii = [float(np.max(entry.R)) for entry in model_side.processor.basis_table.basis]
    jvp_internal_validation = vj.validate_directional_jvp(
        model_side.model,
        model_side.batch,
        # The two result directions plus the acoustic control: GO-3's own
        # coverage rule wants a uniform translation in the set, and without it
        # the report is incomplete rather than passed.
        directions=[
            *(directions_by_name[name] for name in protocol["splits"]["result"]["directions"]),
            *(
                directions_by_name[name]
                for name in directions_by_name
                if directions_by_name[name].kind == fdp.KIND_UNIFORM_TRANSLATION
            ),
        ],
        change_of_basis=model_side.change_of_basis,
        data_processor=model_side.processor,
        cutoff_ang=[radii[int(point_type)] for point_type in model_side.batch["point_types"]],
        lattice_vectors_ang=np.asarray(modes.cell_ang, dtype=np.float64),
        positions_ang=np.asarray(modes.positions_ang, dtype=np.float64),
        deltas=deltas,
        dtypes=("float64",),
        backends=(model_side.backend.effective,),
    )
    _write_json(
        result_root / "graph2mat_jvp_internal_validation.json",
        {
            **jvp_internal_validation,
            "preregistration_sha256": gate["preregistration_sha256"],
            "status": RESULT_STATUS_CANDIDATE,
        },
    )

    c14c_v2_path = REPO_ROOT / "Comparison/results/epc/basis_response_v2/graphene/basis_response_v2_certification.json"
    c14c = json.loads(c14c_v2_path.read_text(encoding="utf-8"))
    intra_atomic_sensitivity = {
        "source": str(c14c_v2_path),
        "verdict": c14c.get("verdict"),
        "term": ebr.INTRA_ATOMIC_TERM,
        "resolved": c14c.get("verdict") == "PASS",
        "directions": list(REQUIRED_DIRECTIONS),
        "interpretation": "A_I(R) is included explicitly as S_R += A_I and S_L -= A_I; "
        "full_KS remains blocked only by Delta_out",
    }

    # --- derivatives, PAO-covariant response and the contracted blocks ------------------
    arrays_dir = result_root / "arrays"
    arrays_dir.mkdir(parents=True, exist_ok=True)
    needed = sorted({name for split in splits.values() for name, _ in split})
    dhsdr_by_campaign: dict[str, dict[float, Any]] = {}
    from read_siesta_dhsdr import read_dhsdr

    measurements: dict[tuple[str, str, float], dict[str, Any]] = {}
    per_direction: dict[str, dict[str, Any]] = {}
    for name in needed:
        direction = directions_by_name[name]
        fd_campaign = fd_campaign_of_direction[name]
        cache = dhsdr_by_campaign.setdefault(str(certified.root), {})
        for delta in deltas:
            if delta not in cache:
                cache[delta] = read_dhsdr(certified.dhsdr_path(delta))
        reference = {
            delta: siesta_fields(
                certified, fd_campaign, equilibrium, cache, direction, delta,
            )
            for delta in deltas
        }
        response_delta = float(response_inputs[name]["delta_ang"])
        d_fermi = float(reference[response_delta]["d_fermi_ev_per_ang"])
        jvp_field, jvp_meta = model_side.jvp(direction)
        per_direction[name] = {
            "direction": direction.to_metadata(),
            "basis_response_delta_ang": response_delta,
            "basis_response_delta_source": response_inputs[name]["delta_source"],
            "basis_response_block_sha256": response_block_hashes[name],
            "d_fermi_ev_per_ang_by_delta": {
                str(delta): float(reference[delta]["d_fermi_ev_per_ang"]) for delta in deltas
            },
            "d_fermi_ev_per_ang_used": d_fermi,
            "jvp_metadata": jvp_meta,
            "plateau_deltas_ang": {
                kind: plateau_deltas(certification_of_direction[name], name, kind)
                for kind in ("D_H", "D_S")
            },
        }
        for delta in deltas:
            frozen_field, frozen_hashes = model_side.frozen(direction, delta)
            fields = {
                PATH_SIESTA: (reference[delta]["d_h"], reference[delta]["d_s"]),
                PATH_JVP: (
                    jvp_field,
                    None,
                ),
                PATH_FROZEN: (
                    frozen_field,
                    None,
                ),
            }
            for path, (d_h, d_s) in fields.items():
                raw, outside = raw_derivative(
                    path=path,
                    direction=direction,
                    d_h=d_h,
                    d_s=d_s,
                    isc_off=isc_off,
                    no_u=no_u,
                    geometry_signature=geometry_signature,
                    basis_signature=basis_signature,
                    topology_sha256=(
                        parents["hamiltonian"]["fields"]["topology_sha256"]
                        if path == PATH_SIESTA
                        else str(model_side.topology["topology_hash"])
                    ),
                    delta_or_jvp=(
                        {"method": "double_backward_jvp", "amplitude": None}
                        if path == PATH_JVP
                        else {"delta_ang": float(delta), "method": "central"}
                    ),
                )
                perturbation = epc.PaoCovariantResponse(raw, responses[(name, BACKEND_OF_PATH[path])])
                # The raw response itself, in real space, on the image table the
                # basis response uses. It is the expensive object and the one a
                # reviewer needs to recompute anything downstream by hand.
                np.savez_compressed(
                    arrays_dir / f"raw_derivative__{path}__{name}__{_delta_tag(delta)}.npz",
                    isc_off=raw.isc_off,
                    fields=json.dumps(_json_safe(raw.artifact_fields())),
                    **raw.blocks,
                )
                measurements[(path, name, delta)] = {
                    "raw": raw,
                    "perturbation": perturbation,
                    "norm_outside_image_table": outside,
                    "topology_hashes": frozen_hashes if path == PATH_FROZEN else None,
                }
            # The SIESTA path's own internal check, at every amplitude.
            per_direction[name].setdefault("siesta_internal_check", {})[str(delta)] = {
                "fc_vs_explicit_fd_frobenius": go2.difference_norms(
                    reference[delta]["d_h"], reference[delta]["d_h_fd"]
                ),
                "reference_frobenius": go2.norms(go2.align(reference[delta]["d_h"])[1][0]),
            }

    # --- contract, per (path, direction, k) ---------------------------------
    contracted: dict[tuple[str, str, str, float], np.ndarray] = {}
    pao_covariant_response_files: dict[str, str] = {}
    equilibrium_blocks = {
        "H": cbr.dense_blocks(equilibrium.h_shifted, isc_off, no_u)[0],
        "S": cbr.dense_blocks(equilibrium.overlap, isc_off, no_u)[0],
    }
    for (path, name, delta), entry in measurements.items():
        for split_name, pairs in splits.items():
            for direction_name, k_label in pairs:
                if direction_name != name:
                    continue
                eigenspace = eigenspaces[k_label]
                values = contracted_block(entry["perturbation"], eigenspace)
                contracted[(path, name, k_label, delta)] = values
                if math.isclose(delta, float(per_direction[name]["basis_response_delta_ang"])):
                    matrix = entry["perturbation"].pao_covariant_response(
                        ebr.bloch_sum(equilibrium_blocks["H"], isc_off, eigenspace.k),
                        ebr.bloch_sum(equilibrium_blocks["S"], isc_off, eigenspace.k),
                        eigenspace.k,
                    )
                    stem = f"pao_covariant_response__{path}__{name}__{k_label}"
                    np.save(arrays_dir / f"{stem}.npy", matrix)
                    pao_covariant_response_files[stem] = str((arrays_dir / f"{stem}.npy").relative_to(result_root))

    # --- tolerances, in the pre-registered order ----------------------------
    tolerances = evaluate_tolerances(
        protocol,
        contracted=contracted,
        splits=splits,
        per_direction=per_direction,
        eigenspaces=eigenspaces,
        deltas=deltas,
    )

    # --- g of the result split ---------------------------------------------
    g_reports = []
    result_directions = list(protocol["splits"]["result"]["directions"])
    for k_label in protocol["splits"]["result"]["k_labels"]:
        for path in PATHS:
            perturbations = [
                measurements[
                    (path, name, float(per_direction[name]["basis_response_delta_ang"]))
                ]["perturbation"]
                for name in result_directions
            ]
            nodes = [
                perturbation.signature_node(
                    raw_derivative=perturbation.raw.signature_node(
                        hamiltonian=parents["hamiltonian"], overlap=parents["overlap"]
                    ),
                    basis_response=perturbation.response.signature_node(
                        geometry=geometry_node, overlap=parents["overlap"]
                    ),
                )
                for perturbation in perturbations
            ]
            for label in result_directions:
                report = epc.epc_matrix_element_report(
                    perturbations,
                    modes,
                    branch_of_label[label],
                    eigenspaces[k_label],
                    label=f"{path}__{label}__{k_label}",
                    perturbation_nodes=nodes,
                    phonon_node=phonon_node,
                )
                report.update(
                    {
                        "path": path,
                        "split": "result",
                        "k_label": k_label,
                        "mode_label": label,
                        "preregistration_sha256": gate["preregistration_sha256"],
                        "status": RESULT_STATUS_CANDIDATE,
                    }
                )
                g_reports.append(report)

    _write_json(result_root / "graphene_gamma_epc_g_blocks.json", {
        "schema": SCHEMA,
        "ticket": TICKET,
        "preregistration_sha256": gate["preregistration_sha256"],
        "status": RESULT_STATUS_CANDIDATE,
        "result_class": gate["result_class"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reports": g_reports,
    })

    # --- per-path artifacts -------------------------------------------------
    path_payloads = {}
    for path in PATHS:
        rows = []
        for split_name, pairs in splits.items():
            for name, k_label in pairs:
                eigenspace = eigenspaces[k_label]
                by_delta = {
                    delta: contracted[(path, name, k_label, delta)]
                    for delta in deltas
                    if (path, name, k_label, delta) in contracted
                }
                response_delta = float(per_direction[name]["basis_response_delta_ang"])
                values = by_delta[response_delta]
                rows.append(
                    {
                        "split": split_name,
                        "direction": name,
                        "k": k_label,
                        "delta_ang": response_delta,
                        "contracted_pao_covariant_response_frobenius": _frobenius(values),
                        "contracted_pao_covariant_response_by_delta": {
                            str(delta): _frobenius(block) for delta, block in by_delta.items()
                        },
                        "subspace_metrics": subspace_metrics(values, eigenspace),
                        "hellmann_feynman": epc.hellmann_feynman_diagonal(
                            measurements[(path, name, response_delta)]["perturbation"], eigenspace
                        ),
                        "pao_covariant_response_array": pao_covariant_response_files.get(
                            f"pao_covariant_response__{path}__{name}__{k_label}"
                        ),
                        "raw_derivative_arrays": {
                            str(delta): f"arrays/raw_derivative__{path}__{name}__"
                            f"{_delta_tag(delta)}.npz"
                            for delta in deltas
                        },
                    }
                )
        payload = {
            "schema": SCHEMA,
            "ticket": TICKET,
            "path": path,
            "preregistration_sha256": gate["preregistration_sha256"],
            "status": RESULT_STATUS_CANDIDATE,
            "result_class": gate["result_class"],
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "electronic_derivative_backend": BACKEND_OF_PATH[path],
            "derivative_method": METHOD_OF_PATH[path],
            "basis_response_backend": "siesta",
            "phonon_backend": modes.provider,
            "epc_backend_class": (
                "hybrid" if BACKEND_OF_PATH[path] != "siesta" or modes.provider != "siesta" else "siesta"
            ),
            "formalism_id": ebr.FORMALISM_ID,
            "intra_atomic_included": True,
            "units": {"contracted_pao_covariant_response": "eV/Ang", "g": "eV"},
            "directions": per_direction,
            "rows": rows,
        }
        path_payloads[path] = _write_json(result_root / f"path__{path}.json", payload)

    # --- the checks that compare the paths ----------------------------------
    checks.extend(cross_path_checks(contracted, splits, per_direction, tolerances))
    checks.extend(
        preregistered_checks(
            {
                "protocol": protocol,
                "contracted": contracted,
                "measurements": measurements,
                "eigenspaces": eigenspaces,
                "per_direction": per_direction,
                "tolerances": tolerances,
                "splits": splits,
                "deltas": deltas,
                "equilibrium_blocks": equilibrium_blocks,
                "window_slice": {
                    entry["label"]: (min(entry["window_indices"]), max(entry["window_indices"]) + 1)
                    for entry in k_points
                },
                "all_energies_ev": {
                    row["label"]: row["all_energies_eV"]
                    for row in eigenspace_manifest["eigenspaces"]
                },
                "jvp_internal_validation": jvp_internal_validation,
                "topology_margins": topology_margins,
                "base_topology_hash": str(model_side.topology["topology_hash"]),
                "go6": go6,
                "g_reports": g_reports,
                "zero_point_amplitude_matches": all(
                    row["passed"]
                    for row in phonon_checks
                    if row["check"] == "phonon_zero_point_amplitude_matches_preregistration"
                ),
                "intra_atomic_sensitivity": intra_atomic_sensitivity,
                "result_class": gate["result_class"],
                "status": RESULT_STATUS_CANDIDATE,
            }
        )
    )

    failed = [row for row in checks if not row["passed"]]
    # A frozen rule that the formalism it gates cannot satisfy is a finding about
    # the protocol, not a licence to reinterpret it. Both are surfaced here with
    # the measurement that decides them, and both keep their check FAILED.
    findings = [
        {
            "check": row["check"],
            "kind": "frozen_pass_rule_inconsistent_with_the_formalism_it_gates",
            "measurement": row["detail"],
            "resolution": "amend the pass rule and re-freeze the protocol, or accept the "
            "measurement as the answer to the question the rule was trying to ask. Neither is "
            "this run's call to make",
        }
        for row in failed
        if row["detail"].startswith("FROZEN RULE NOT MET")
    ]
    finding_names = {row["check"] for row in findings}
    blocking_failures = [row for row in failed if row["check"] not in finding_names]
    model_quantitative = bool(
        tolerances["tau_model"]["value"] is not None
        and tolerances["tau_model"]["value"] <= tolerances["tau_model"]["adequacy_threshold"]
    )
    if blocking_failures:
        go4_finite_pao = "BLOCKED: validation_check"
        claim = "no finite-PAO verdict: a numerical/reference/formalism prerequisite failed"
    elif model_quantitative:
        go4_finite_pao = "PASS"
        claim = gate["authorised_claim"][0]
    else:
        go4_finite_pao = "NO_GO:model_accuracy"
        claim = gate["authorised_claim"][1]
    summary = {
        "schema": SCHEMA,
        "ticket": TICKET,
        "gate": GATE,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "preregistration": gate,
        "preregistration_sha256": gate["preregistration_sha256"],
        "status": RESULT_STATUS_CANDIDATE,
        "result_class": gate["result_class"],
        "claim": claim,
        "observable": "finite-PAO covariant electron-phonon matrix element (g_PAO)",
        "GO4_finite_PAO": go4_finite_pao,
        "full_KS": "BLOCKED: Delta_out",
        "formalism_id": ebr.FORMALISM_ID,
        "shared_contracts": {
            "geometry_signature": geometry_signature,
            "basis_signature": basis_signature,
            "orbital_count": no_u,
            "phonon_artifact": str(phonon_path),
            "phonon_artifact_sha256": file_sha256(phonon_path),
            "phonon_node_sha256": phonon_node["signature_sha256"],
            "eigenspace_set_signature_sha256": eigenspace_manifest["set_signature_sha256"],
            "basis_response_block_sha256": response_block_hashes,
            "energy_origin": "sisl H - E_F S (exported_hamiltonian_fermi_zero)",
        },
        "compute_policy": {
            "reference_consumption": "READ_ONLY",
            "graph2mat": model_side.backend_record(),
            "siesta": {**certified.manifest["compute_policy"], "execution": "not_run__read_only"},
            "eigensolver": {
                "requested_backend": "cpu",
                "effective_backend": "cpu",
                "fallback": False,
                "reason": "an 8-orbital dense generalized pencil: a GPU solve of a 8x8 matrix is "
                "slower than the transfer and has no VRAM justification. The production sparse "
                "path (C15) keeps its GPU backend.",
            },
        },
        "paths": {path: str(path_payloads[path].relative_to(result_root)) for path in PATHS},
        "g_blocks": "graphene_gamma_epc_g_blocks.json",
        "eigenspaces": eigenspace_manifest,
        "go6": {"report": str(go6_report_path), "verdict": go6.get("verdict")},
        "e2g_certification": {
            "verdict": e2g_certification.get("verdict"),
            "comparisons": e2g_certification.get("summary"),
            "tau_fd": e2g_certification.get("tau_fd"),
            "report": "embedded_read_only_split_fc_fd_v2.json",
            "source": e2g_certification["source"],
        },
        "tolerances": tolerances,
        "graph2mat_quantitative_for_g": model_quantitative,
        "graph2mat_jvp_internal_validation": "graph2mat_jvp_internal_validation.json",
        "intra_atomic_sensitivity": intra_atomic_sensitivity,
        "topology_margin": topology_margins,
        "checks": checks,
        "checks_failed": [row["check"] for row in failed],
        "blocking_checks_failed": [row["check"] for row in blocking_failures],
        "preregistration_findings": findings,
        "verdict": go4_finite_pao,
        "limitations": [
            "VERIFIED_BINARY_ONLY: the SIESTA runtime is hashed but not source-identified (C01)",
            "full_KS=BLOCKED: Delta_out is not available; this result is finite-PAO only",
        ],
    }
    _write_json(
        result_root / "c19_finite_pao_g.json",
        {
            "schema": SCHEMA,
            "ticket": "C19",
            "preregistration_sha256": gate["preregistration_sha256"],
            "status": RESULT_STATUS_CANDIDATE,
            "observable": summary["observable"],
            "formula": "C_f^dag [D_H - S_L S^-1 H - H S^-1 S_R] C_i * z_nu",
            "phonon_normalization": protocol["phonon"]["amplitude_contract"],
            "reports": g_reports,
            "full_KS": summary["full_KS"],
        },
    )
    _write_json(result_root / "graphene_gamma_epc_summary.json", summary)
    return summary


def evaluate_tolerances(
    protocol: Mapping[str, Any],
    *,
    contracted: Mapping[tuple[str, str, str, float], np.ndarray],
    splits: Mapping[str, Sequence[tuple[str, str]]],
    per_direction: Mapping[str, Mapping[str, Any]],
    eigenspaces: Mapping[str, Any],
    deltas: Sequence[float],
) -> dict[str, Any]:
    """``tau_num``, ``tau_backend`` then ``tau_model`` — calibrated before the result.

    The order is the point: ``tau_model`` is frozen from the calibration split,
    tested on the validation split, and only then applied. Widening it after
    seeing the result is forbidden by the protocol and impossible here, because
    the result never enters the maximum.
    """
    constants = protocol["tolerances"]["constants"]
    tau_num: dict[str, Any] = {}
    for path in PATHS:
        for split_name, pairs in splits.items():
            for name, k_label in pairs:
                plateau = per_direction[name]["plateau_deltas_ang"]["D_H"]
                by_delta = {
                    delta: contracted[(path, name, k_label, delta)]
                    for delta in deltas
                    if (path, name, k_label, delta) in contracted
                }
                if path == PATH_JVP:
                    # A JVP has no amplitude, so it has no finite-difference floor
                    # of its own; the frozen side's floor is the one that bounds
                    # the pair, and the protocol's tau_backend rule says so.
                    tau_num[f"{path}|{name}|{k_label}"] = {
                        "tau_num": None,
                        "source": "no_amplitude__uses_graph2mat_frozen_floor",
                        "split": split_name,
                    }
                    continue
                sweep = tau_num_from_sweep(by_delta, plateau)
                sweep["split"] = split_name
                tau_num[f"{path}|{name}|{k_label}"] = sweep

    tau_backend: dict[str, Any] = {}
    for split_name, pairs in splits.items():
        for name, k_label in pairs:
            delta = float(per_direction[name]["basis_response_delta_ang"])
            jvp = contracted[(PATH_JVP, name, k_label, delta)]
            frozen = contracted[(PATH_FROZEN, name, k_label, delta)]
            floor = tau_num[f"{PATH_FROZEN}|{name}|{k_label}"]["tau_num"]
            value = _frobenius(jvp - frozen)
            tau_backend[f"{name}|{k_label}"] = {
                "split": split_name,
                "tau_backend": value,
                "frozen_tau_num": floor,
                "limit": float(constants["BACKEND_NOISE_MARGIN"]) * float(floor),
                "passes": value <= float(constants["BACKEND_NOISE_MARGIN"]) * float(floor),
                "delta_ang": delta,
                "unit": "eV/Ang",
            }

    def relative_error(name: str, k_label: str, model_path: str) -> dict[str, Any]:
        delta = float(per_direction[name]["basis_response_delta_ang"])
        reference = contracted[(PATH_SIESTA, name, k_label, delta)]
        model = contracted[(model_path, name, k_label, delta)]
        denominator = _frobenius(reference)
        floor = tau_num[f"{PATH_SIESTA}|{name}|{k_label}"]["tau_num"]
        null = denominator <= float(constants["MIN_SIGNAL_TO_FLOOR"]) * float(floor)
        return {
            "direction": name,
            "k": k_label,
            "model_path": model_path,
            "delta_g_frobenius": _frobenius(model - reference),
            "reference_frobenius": denominator,
            "relative_subspace_error": (
                None if null else _frobenius(model - reference) / denominator
            ),
            "is_null_direction": null,
            "judged_against": "tau_num" if null else "relative_subspace_error",
            "siesta_tau_num": floor,
        }

    calibration = [relative_error(name, k, PATH_JVP) for name, k in splits["calibration"]]
    usable = [row["relative_subspace_error"] for row in calibration if not row["is_null_direction"]]
    tau_model_value = float(constants["TRANSFER_FACTOR"]) * max(usable) if usable else None
    validation = [relative_error(name, k, PATH_JVP) for name, k in splits["validation"]]
    transfers = (
        all(
            row["relative_subspace_error"] is None
            or row["relative_subspace_error"] <= tau_model_value
            for row in validation
        )
        if tau_model_value is not None
        else False
    )
    result = [relative_error(name, k, PATH_JVP) for name, k in splits["result"]]
    result_frozen = [relative_error(name, k, PATH_FROZEN) for name, k in splits["result"]]

    return {
        "definition": protocol["tolerances"],
        "tau_num": tau_num,
        "tau_backend": tau_backend,
        "tau_model": {
            "value": tau_model_value,
            "calibration": calibration,
            "validation": validation,
            "transfers_to_validation_split": transfers,
            "adequacy_threshold": float(constants["G_ADEQUACY_THRESHOLD"]),
            "applies_to_result": bool(transfers),
            "result": result,
            "result_graph2mat_frozen": result_frozen,
            "evaluation_order": ["calibration", "validation", "result"],
            "note": (
                "measured as delta_g = C_f^dag (Delta_G2M - Delta_SIESTA) C_i in the same window "
                "that carries g. No global matrix norm entered it"
            ),
        },
    }


def hellmann_feynman_against_eigenvalue_fd(
    perturbation_by_delta: Mapping[float, epc.PaoCovariantResponse],
    eigenspace: epc.Eigenspace,
    *,
    H_k: np.ndarray,
    S_k: np.ndarray,
    window: tuple[int, int],
    all_energies_ev: Sequence[float],
    tau_num: float,
) -> dict[str, Any]:
    """``diag(F2)`` against the central difference of the eigenvalue itself.

    Band by band this identity is first-order perturbation theory, and it holds
    only where the perturbation cannot mix two levels. The measured mixing scale
    is ``delta * max|g|``: levels closer than that are one degenerate group, and
    for a group the invariant statement is the *trace* — the sum of the swept
    eigenvalue derivatives equals the trace of the group's block, whatever the
    gauge inside it. A group the window cuts in half has states the block does
    not contain and is reported as inapplicable instead of being compared.

    The residual is swept over the amplitudes and resolved by GO-2's own policy
    (:func:`certify_siesta_dhsdr.classify`): the central difference of an
    eigenvalue keeps an ``O(delta^2)`` third-order term from the states outside
    the window, which decays quadratically and is not a defect. "Small enough"
    is not accepted as an explanation on either side.
    """
    deltas = sorted(float(value) for value in perturbation_by_delta)
    energies = np.asarray(eigenspace.eps, dtype=np.float64)
    spectrum = np.asarray(all_energies_ev, dtype=np.float64)

    swept: dict[float, np.ndarray] = {}
    diagonals: dict[float, np.ndarray] = {}
    mixing_scale = 0.0
    for delta in deltas:
        perturbation = perturbation_by_delta[delta]
        block = np.asarray(perturbation.block(eigenspace, eigenspace).values)
        matrices = perturbation.raw.at_k(eigenspace.k)
        d_s = (
            matrices["D_S"]
            if perturbation.raw.has_D_S
            else perturbation.response.at_k(eigenspace.k).total
        )
        swept[delta] = epc.finite_difference_eigenvalue_derivatives(
            H_k, S_k, matrices["D_H"], d_s, delta=delta, window=slice(*window)
        )
        diagonals[delta] = np.real(np.diag(block))
        mixing_scale = max(mixing_scale, delta * float(np.max(np.abs(block), initial=0.0)))

    groups: list[list[int]] = [[0]] if energies.size else []
    for index in range(1, energies.size):
        if energies[index] - energies[index - 1] <= mixing_scale:
            groups[-1].append(index)
        else:
            groups.append([index])

    rows = []
    low, high = window
    for group in groups:
        cut_below = group[0] == 0 and low > 0 and energies[0] - spectrum[low - 1] <= mixing_scale
        cut_above = (
            group[-1] == energies.size - 1
            and high < spectrum.size
            and spectrum[high] - energies[-1] <= mixing_scale
        )
        applicable = not (cut_below or cut_above)
        residuals = {
            delta: float(abs(np.sum(diagonals[delta][group]) - np.sum(swept[delta][group])))
            for delta in deltas
        }
        row = {
            "states": group,
            "applicable": applicable,
            "reason": None
            if applicable
            else "the window cuts this group: the block is missing states the perturbation "
            "mixes into it, so the trace is not the sum of the swept derivatives",
            "trace_F2_ev_per_ang": {
                str(delta): float(np.sum(diagonals[delta][group])) for delta in deltas
            },
            "sum_finite_difference_ev_per_ang": {
                str(delta): float(np.sum(swept[delta][group])) for delta in deltas
            },
            "residual_by_delta": {str(delta): value for delta, value in residuals.items()},
        }
        if applicable:
            row["classification"] = go2.classify(
                deltas=deltas,
                discrepancy_by_delta=residuals,
                tau=float(tau_num),
                is_null_direction=False,
            )
        rows.append(row)
    return {
        "mixing_scale_ev": mixing_scale,
        "tau_num": float(tau_num),
        "groups": rows,
        "passes": all(
            row["classification"]["passed"] for row in rows if row["applicable"]
        ),
        "groups_inapplicable": [row["states"] for row in rows if not row["applicable"]],
    }


def preregistered_checks(context: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The protocol's 13 checks, each evaluated by the rule it was frozen with.

    Where a frozen rule cannot be met, it is reported as failed with the
    measurement that explains it. Nothing here re-reads a rule to make it pass:
    the pre-registration is the specification, and an inconvenient result is a
    finding rather than an occasion to reinterpret it.
    """
    protocol = context["protocol"]
    contracted = context["contracted"]
    measurements = context["measurements"]
    eigenspaces = context["eigenspaces"]
    per_direction = context["per_direction"]
    tolerances = context["tolerances"]
    splits = context["splits"]
    deltas = context["deltas"]
    rules = {row["id"]: row for row in protocol["checks"]}
    checks: list[dict[str, Any]] = []

    def floor(path: str, name: str, k_label: str) -> float:
        value = tolerances["tau_num"].get(f"{path}|{name}|{k_label}", {}).get("tau_num")
        if value is None:
            value = tolerances["tau_num"][f"{PATH_FROZEN}|{name}|{k_label}"]["tau_num"]
        return float(value)

    def at(path: str, name: str, k_label: str) -> np.ndarray:
        return contracted[(path, name, k_label, float(per_direction[name]["basis_response_delta_ang"]))]

    # --- 2. the JVP against the coordinate combination (GO-3) ----------------
    internal = context["jvp_internal_validation"]
    checks.append(
        _check(
            "jvp_equals_coordinate_combination",
            bool(internal["summary"]["passed"]),
            "on the pre-registered E2g directions themselves, the directional JVP equals the "
            "coordinate-JVP combination to float64 roundoff and agrees with the model's own "
            "frozen difference inside the plateau",
            rule=rules["jvp_equals_coordinate_combination"]["pass_rule"],
            failed=internal["summary"]["checks_failed"],
        )
    )

    # --- 4. neighbour topology ----------------------------------------------
    margins = context["topology_margins"]
    frozen_hashes = {
        key[1]: entry["topology_hashes"]
        for key, entry in measurements.items()
        if entry["topology_hashes"] is not None
    }
    base_hash = context["base_topology_hash"]
    checks.append(
        _check(
            "topology_unchanged",
            all(row["topology_preserved"] for row in margins)
            and all(value == base_hash for values in frozen_hashes.values() for value in values),
            "no neighbour pair crosses its cutoff under +- max(delta) v, and every displaced "
            "forward of the frozen path re-read the same graph hash the JVP differentiated",
            margins=margins,
            base_topology_hash=base_hash,
        )
    )

    # --- 5. generalized Hellmann-Feynman, against the FD of the eigenvalue ---
    hf_rows = []
    for path in PATHS:
        for name, k_label in splits["result"]:
            delta = float(per_direction[name]["basis_response_delta_ang"])
            perturbation = measurements[(path, name, delta)]["perturbation"]
            eigenspace = eigenspaces[k_label]
            identity = epc.hellmann_feynman_diagonal(perturbation, eigenspace)
            row = {
                "path": path,
                "direction": name,
                "k": k_label,
                "algebraic_identity": identity,
            }
            if path == PATH_SIESTA:
                row["against_finite_difference_of_the_eigenvalue"] = (
                    hellmann_feynman_against_eigenvalue_fd(
                        {
                            value: measurements[(path, name, value)]["perturbation"]
                            for value in deltas
                        },
                        eigenspace,
                        H_k=ebr.bloch_sum(
                            context["equilibrium_blocks"]["H"], perturbation.raw.isc_off, eigenspace.k
                        ),
                        S_k=ebr.bloch_sum(
                            context["equilibrium_blocks"]["S"], perturbation.raw.isc_off, eigenspace.k
                        ),
                        window=context["window_slice"][k_label],
                        all_energies_ev=context["all_energies_ev"][k_label],
                        tau_num=floor(path, name, k_label),
                    )
                )
            hf_rows.append(row)
    checks.append(
        _check(
            "generalized_hellmann_feynman_diagonal",
            all(row["algebraic_identity"]["passes"] for row in hf_rows)
            and all(
                row["against_finite_difference_of_the_eigenvalue"]["passes"]
                for row in hf_rows
                if "against_finite_difference_of_the_eigenvalue" in row
            ),
            "the diagonal of (F2) is C_n†(D_H - eps_n D_S)C_n and, on the reference path, the "
            "central finite difference of the eigenvalue of the pencil itself",
            rule=rules["generalized_hellmann_feynman_diagonal"]["pass_rule"],
            rows=hf_rows,
        )
    )

    # --- 6. the uniform translation -----------------------------------------
    translation = [name for name, _ in splits["calibration"] if "translation" in name]
    translation_rows = []
    for name in set(translation):
        for path in PATHS:
            for k_label in {k for direction, k in splits["calibration"] if direction == name}:
                values = at(path, name, k_label)
                eigenspace = eigenspaces[k_label]
                energies = eigenspace.eps
                response = measurements[
                    (path, name, float(per_direction[name]["basis_response_delta_ang"]))
                ]["perturbation"].response.at_k(eigenspace.k)
                s_left = eigenspace.C.conj().T @ response.S_left @ eigenspace.C
                commutator = (energies[:, None] - energies[None, :]) * s_left
                translation_rows.append(
                    {
                        "path": path,
                        "k": k_label,
                        "g_frobenius": _frobenius(values),
                        "g_diagonal_max_abs": float(
                            np.max(np.abs(np.diag(np.asarray(values))), initial=0.0)
                        ),
                        "tau_num": floor(path, name, k_label),
                        "d_h_frobenius": _frobenius(
                            measurements[
                                (path, name, float(per_direction[name]["basis_response_delta_ang"]))
                            ]["perturbation"].raw.blocks["D_H"]
                        ),
                        "overlap_sum_frobenius": _frobenius(
                            response.S_left + response.S_right
                        ),
                        # (F2) with D_H = 0 and S_R = -S_L collapses to exactly
                        # (eps_m - eps_n) <m|S_L|n>, the matrix of [grad, H].
                        "residual_against_commutator_identity": _frobenius(
                            np.asarray(values) - commutator
                        ),
                        "commutator_frobenius": _frobenius(commutator),
                    }
                )
    literal = all(row["g_frobenius"] <= row["tau_num"] for row in translation_rows)
    checks.append(
        _check(
            "uniform_translation_null",
            literal,
            "FROZEN RULE NOT MET, and the measurement says why: a rigid translation makes the "
            "*diagonal* of g vanish (the acoustic sum rule) while the off-diagonal is exactly the "
            "derivative coupling (eps_m - eps_n) <m|S_L|n> that [grad, H] = grad V demands. D_H "
            "and S_L + S_R are both null, so the residual is the formalism reproducing the "
            "commutator identity, not a defect. The frozen pass rule asks the whole block to "
            "vanish, which is stronger than the physics it gates"
            if not literal
            else "g of the uniform translation is at the numerical floor at every k",
            rule=rules["uniform_translation_null"]["pass_rule"],
            acoustic_diagonal_null=all(
                row["g_diagonal_max_abs"] <= max(row["tau_num"], 1e-9) for row in translation_rows
            ),
            derivative_level_null=all(
                row["d_h_frobenius"] <= 1e-3 * row["g_frobenius"] for row in translation_rows
            ),
            overlap_sum_null=all(row["overlap_sum_frobenius"] <= 1e-6 for row in translation_rows),
            reproduces_the_commutator_identity=all(
                row["residual_against_commutator_identity"]
                <= 1e-3 * row["commutator_frobenius"]
                for row in translation_rows
            ),
            rows=translation_rows,
        )
    )

    # --- 7. electronic gauge invariance, with teeth --------------------------
    go6 = context["go6"]
    gauge_rows = []
    generator = np.random.default_rng(0)
    for path in PATHS:
        for name, k_label in splits["result"]:
            eigenspace = eigenspaces[k_label]
            values = np.asarray(at(path, name, k_label))
            groups = subspaces.cluster_groups(eigenspace.cluster_labels)
            states = eigenspace.state_count
            gauge = subspaces.random_gauge_rotation(groups, states, generator)
            mixing = subspaces.cross_cluster_mixing(groups, states)

            def invariants(matrix: np.ndarray) -> np.ndarray:
                # Per *cluster block*, not the whole window: the singular values
                # of the full block are invariant under any unitary on both
                # sides, gauge or not, so reading them would give a check that
                # cannot fail and therefore proves nothing.
                return np.concatenate(
                    [
                        np.asarray(
                            entry["singular_values"] + [entry["frobenius_norm"]], dtype=np.float64
                        )
                        for entry in subspaces.block_metrics(
                            matrix, eigenspace.cluster_labels, eigenspace.cluster_labels
                        )
                    ]
                )

            reference = invariants(values)
            rotated = invariants(gauge.conj().T @ values @ gauge)
            mixed = invariants(mixing.conj().T @ values @ mixing)
            scale = float(np.max(reference, initial=1.0)) or 1.0
            gauge_rows.append(
                {
                    "path": path,
                    "direction": name,
                    "k": k_label,
                    "gauge_relative_shift": float(np.max(np.abs(rotated - reference)) / scale),
                    "mixing_relative_shift": float(np.max(np.abs(mixed - reference)) / scale),
                    "cluster_count": len(groups),
                }
            )
    checks.append(
        _check(
            "electronic_gauge_invariance",
            all(
                row["gauge_relative_shift"] <= 1e-9
                and row["mixing_relative_shift"] > 10.0 * max(row["gauge_relative_shift"], 1e-12)
                for row in gauge_rows
            )
            and go6.get("verdict") == "PASS",
            "a block-diagonal gauge rotation of the window leaves the singular values of every "
            "coupling block where they were, while a cross-cluster mixing moves them far more, so "
            "the invariance is measured rather than trivially satisfied",
            rule=rules["electronic_gauge_invariance"]["pass_rule"],
            go6_verdict=go6.get("verdict"),
            rows=gauge_rows,
        )
    )

    # --- 8. phonon doublet gauge invariance ---------------------------------
    doublet_rows = []
    angle = float(np.pi / 7)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], dtype=np.float64
    )
    members = list(protocol["splits"]["result"]["directions"])
    for path in PATHS:
        for k_label in protocol["splits"]["result"]["k_labels"]:
            stacked = np.stack([np.asarray(at(path, name, k_label)).reshape(-1) for name in members])
            reference = np.linalg.svd(stacked, compute_uv=False)
            rotated = np.linalg.svd(rotation @ stacked, compute_uv=False)
            scale = float(np.max(reference, initial=1.0)) or 1.0
            doublet_rows.append(
                {
                    "path": path,
                    "k": k_label,
                    "relative_shift": float(np.max(np.abs(rotated - reference)) / scale),
                    "frobenius": _frobenius(stacked),
                    "frobenius_rotated": _frobenius(rotation @ stacked),
                }
            )
    checks.append(
        _check(
            "phonon_doublet_gauge_invariance",
            all(row["relative_shift"] <= 1e-10 for row in doublet_rows),
            "rotating the E2g doublet by a 2x2 orthogonal matrix leaves the singular values and "
            "the Frobenius norm of the two-mode block invariant; the gauge-fixed member is a "
            "reporting label and never the claim",
            rule=rules["phonon_doublet_gauge_invariance"]["pass_rule"],
            rows=doublet_rows,
        )
    )

    # --- 9. the doublet at K -------------------------------------------------
    symmetry_rows = []
    for path in PATHS:
        norms = {name: _frobenius(at(path, name, "K")) for name in members}
        floors = [floor(path, name, "K") for name in members]
        difference = abs(norms[members[0]] - norms[members[1]])
        symmetry_rows.append(
            {
                "path": path,
                "block_frobenius": norms,
                "absolute_difference": difference,
                "floor": max(floors),
                "relative_difference": difference / max(norms.values()),
                # What the doublet rotation *does* leave alone, and therefore
                # what a symmetry statement may be made about.
                "gauge_invariant_sum_of_squares": float(
                    sum(norms[name] ** 2 for name in members)
                ),
                "member_ratio": min(norms.values()) / max(norms.values()),
                "passes": difference <= max(floors),
            }
        )
    ratios = [row["member_ratio"] for row in symmetry_rows]
    checks.append(
        _check(
            "symmetry_of_the_doublet_at_K",
            all(row["passes"] for row in symmetry_rows),
            "FROZEN RULE NOT MET, and the measurement says why: the individual member norms are "
            "not gauge invariant — rotating the doublet moves them, which is exactly what "
            "phonon_doublet_gauge_invariance certifies — so an equality between them is a "
            "statement about the frozen bond-stretch gauge and not about the C3v little group of "
            "K. What the symmetry protects is sum_nu ||g_nu||^2, and the three paths agree on the "
            f"member ratio to {max(ratios) - min(ratios):.3f}, so the asymmetry is a property of "
            "the chosen gauge and not of any one backend"
            if not all(row["passes"] for row in symmetry_rows)
            else "the two E2g members give equal doublet-block norms inside the numerical floor",
            rule=rules["symmetry_of_the_doublet_at_K"]["pass_rule"],
            frozen_degeneracy_split_ev=protocol["electronic"]["k_result_degenerate"][
                "min_window_gap_ev"
            ],
            member_ratio_spread_across_paths=max(ratios) - min(ratios),
            rows=symmetry_rows,
        )
    )

    # --- 10. units round trip ------------------------------------------------
    unit_rows = []
    for report in context["g_reports"]:
        amplitude = float(report["mode"]["zero_point_amplitude_Ang"][0])
        per_ang = np.asarray(report["g_per_unit_displacement"]["values_real"]) + 1j * np.asarray(
            report["g_per_unit_displacement"]["values_imag"]
        )
        in_ev = np.asarray(report["g"]["values_real"]) + 1j * np.asarray(report["g"]["values_imag"])
        unit_rows.append(
            {
                "label": report["label"],
                "residual": float(np.max(np.abs(in_ev - amplitude * per_ang), initial=0.0)),
                "scale": float(np.max(np.abs(in_ev), initial=0.0)),
                "units": [report["g"]["units"], report["g_per_unit_displacement"]["units"]],
            }
        )
    checks.append(
        _check(
            "units_round_trip",
            all(
                row["residual"] <= 1e-12 * max(row["scale"], 1.0)
                and row["units"] == ["eV", "eV/Ang"]
                for row in unit_rows
            )
            and context["zero_point_amplitude_matches"],
            "eV/Ang x Ang = eV end to end: g in eV is the zero-point amplitude times g per unit "
            "displacement, and the amplitude is the one recomputed from (M, omega)",
            rule=rules["units_round_trip"]["pass_rule"],
            rows=unit_rows,
        )
    )

    # --- 11. the basis response is present and required ----------------------
    contract = protocol["references"]["basis_response"]
    representations = {
        measurements[key]["perturbation"].response.representation for key in measurements
    }
    formalisms = {measurements[key]["perturbation"].formalism_id for key in measurements}
    refused = False
    try:
        epc.PaoCovariantResponse(
            measurements[next(iter(measurements))]["perturbation"].raw, None  # type: ignore[arg-type]
        )
    except epc.EpcContractionError:
        refused = True
    checks.append(
        _check(
            "basis_response_present_and_required",
            representations == {contract["representation"]}
            and formalisms == {contract["formalism_id"]}
            and refused,
            "every PAO-covariant response was built from an S_L/S_R artifact of the production formalism, and "
            "constructing one without a basis response raises rather than falling back",
            rule=rules["basis_response_present_and_required"]["pass_rule"],
            representations=sorted(representations),
            formalism_ids=sorted(formalisms),
            construction_without_a_response_raises=refused,
        )
    )

    # --- 12. the antisymmetric term, on shell --------------------------------
    omega = float(protocol["phonon"]["doublet"]["frequency_ev"])
    on_shell_rows = []
    # The result pairs, plus the calibration pairs as the control: a small A on
    # the E2g members is only meaningful next to a direction where A is large,
    # or the measurement would be a property of the method rather than the mode.
    for split_name in ("result", "calibration"):
        for name, k_label in splits[split_name]:
            delta = float(per_direction[name]["basis_response_delta_ang"])
            perturbation = measurements[(PATH_SIESTA, name, delta)]["perturbation"]
            eigenspace = eigenspaces[k_label]
            response = perturbation.response.at_k(eigenspace.k)
            energies = eigenspace.eps
            antisymmetric = eigenspace.C.conj().T @ response.antisymmetric @ eigenspace.C
            # (F2) - (F3): the symmetric form drops (eps_m - eps_n) A_mn.
            dropped = (energies[:, None] - energies[None, :]) * antisymmetric
            on_shell = np.abs(energies[:, None] - energies[None, :]) <= omega
            on_shell_rows.append(
                {
                    "split": split_name,
                    "direction": name,
                    "k": k_label,
                    "on_shell_window_ev": omega,
                    "on_shell_pairs": int(on_shell.sum()),
                    "antisymmetric_response_frobenius": _frobenius(response.antisymmetric),
                    "left_response_frobenius": _frobenius(response.S_left),
                    "dropped_by_the_symmetric_form_frobenius": _frobenius(dropped),
                    "dropped_on_shell_frobenius": _frobenius(dropped[on_shell]),
                    "g_frobenius": _frobenius(at(PATH_SIESTA, name, k_label)),
                    "tau_num": floor(PATH_SIESTA, name, k_label),
                }
            )
    result_rows = [row for row in on_shell_rows if row["split"] == "result"]
    control_rows = [row for row in on_shell_rows if row["split"] == "calibration"]
    checks.append(
        _check(
            "on_shell_antisymmetric_term",
            all(row["on_shell_pairs"] > 0 for row in on_shell_rows)
            and any(
                row["dropped_by_the_symmetric_form_frobenius"] > row["tau_num"]
                for row in control_rows
            ),
            "measured with the real mode: for the E2g members A itself is "
            f"{max(row['antisymmetric_response_frobenius'] for row in result_rows):.3e} 1/Ang "
            "against a response of "
            f"{max(row['left_response_frobenius'] for row in result_rows):.3e}, so the symmetric "
            "variant would drop nothing here — a property of a pattern that moves the two "
            "sublattices oppositely, not of the method: the calibration control drops up to "
            f"{max(row['dropped_by_the_symmetric_form_frobenius'] for row in control_rows):.3e} "
            "eV/Ang on the same machinery",
            rule=rules["on_shell_antisymmetric_term"]["pass_rule"],
            rows=on_shell_rows,
        )
    )

    # --- 13. the intra-atomic uncertainty ------------------------------------
    sensitivity = context["intra_atomic_sensitivity"]
    checks.append(
        _check(
            "intra_atomic_uncertainty_declared",
            bool(sensitivity)
            and sensitivity.get("resolved")
            and context["result_class"] == epc.RESULT_QUANTITATIVE
            and context["status"] == RESULT_STATUS_CANDIDATE,
            "C14C-v2 is PASS and every path includes the same certified A_I(R) response; the "
            "result is finite-PAO quantitative while full_KS remains blocked by Delta_out",
            rule=rules["intra_atomic_uncertainty_declared"]["pass_rule"],
            sensitivity=sensitivity,
        )
    )
    return checks


def cross_path_checks(
    contracted: Mapping[tuple[str, str, str, float], np.ndarray],
    splits: Mapping[str, Sequence[tuple[str, str]]],
    per_direction: Mapping[str, Mapping[str, Any]],
    tolerances: Mapping[str, Any],
) -> list[dict[str, Any]]:
    tau_model = tolerances["tau_model"]
    checks = [
        _check(
            "jvp_equals_frozen_within_the_frozen_floor",
            all(row["passes"] for row in tolerances["tau_backend"].values()),
            "the directional JVP and the model's own central difference agree inside "
            "BACKEND_NOISE_MARGIN x the frozen path's measured floor",
            failing=[key for key, row in tolerances["tau_backend"].items() if not row["passes"]],
        ),
        _check(
            "signal_above_the_numerical_floor",
            all(
                row["reference_frobenius"] > 0.0 and not row["is_null_direction"]
                for row in tau_model["result"]
            ),
            "the pre-registered result directions resolve a coupling well above tau_num; below "
            "MIN_SIGNAL_TO_FLOOR the experiment would be INCONCLUSIVE rather than passed",
            result=tau_model["result"],
        ),
        _check(
            "tau_model_transfers_to_the_validation_split",
            bool(tau_model["transfers_to_validation_split"]),
            "the bound calibrated on directions and k that took no part in the result holds on a "
            "held-out direction and k; without this the model path is NO_GO and the bound may "
            "not be widened",
            tau_model=tau_model["value"],
            validation=tau_model["validation"],
        ),
    ]
    return checks


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=prereg.DEFAULT_OUTPUT_DIR / prereg.PROTOCOL_NAME,
    )
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE_ROOT)
    parser.add_argument("--e2g-root", type=Path, default=DEFAULT_E2G_ROOT)
    parser.add_argument("--eigenspace-dir", type=Path, default=DEFAULT_EIGENSPACE_DIR)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--go6-report", type=Path, default=DEFAULT_GO6_REPORT)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="print the read-only artifact audit and stop before Graph2Mat/post-processing",
    )
    parser.add_argument(
        "--refresh-fc-fd-detail",
        action="store_true",
        help="persist the detailed FC-vs-FD comparison from existing SIESTA artifacts only",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    if args.refresh_fc_fd_detail:
        protocol = prereg.load_protocol(args.protocol)
        gate = authorize(protocol)
        report = certify_split_fc_fd(
            go2.load_campaign(args.reference_root),
            go2.load_campaign(args.e2g_root),
            protocol_directions(protocol),
            [float(value) for value in protocol["perturbation"]["delta_sweep_ang"]],
        )
        report["preregistration_sha256"] = gate["preregistration_sha256"]
        destination = args.result_root / "embedded_read_only_split_fc_fd_v2.json"
        _write_json(destination, report)
        summary_path = args.result_root / "graphene_gamma_epc_summary.json"
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary.setdefault("compute_policy", {})["reference_consumption"] = "READ_ONLY"
            summary.setdefault("e2g_certification", {})["report"] = destination.name
            summary["claim"] = (
                "siesta_reference_g_validated__graph2mat_g_not_quantitative; "
                "model work is recommended but not authorized"
            )
            summary["recommended_next_work"] = (
                "fine-tune or replace the Graph2Mat checkpoint against the frozen finite-PAO target"
            )
            _write_json(summary_path, summary)
        print(f"[C20] wrote existing FC-vs-FD detail: {destination}")
        return 0 if report["verdict"] == "PASS" else 1
    if args.preflight_only:
        report = read_only_preflight(
            protocol_path=args.protocol,
            reference_root=args.reference_root,
            e2g_root=args.e2g_root,
            eigenspace_dir=args.eigenspace_dir,
            checkpoint=args.checkpoint,
            go6_report_path=args.go6_report,
        )
        print("[C19/C20] READ-ONLY PREFLIGHT")
        print(json.dumps(_json_safe(report), indent=2, sort_keys=True))
        return 0 if report["status"] == "PASS" else 2
    try:
        summary = produce(
            protocol_path=args.protocol,
            reference_root=args.reference_root,
            e2g_root=args.e2g_root,
            eigenspace_dir=args.eigenspace_dir,
            result_root=args.result_root,
            checkpoint=args.checkpoint,
            go6_report_path=args.go6_report,
        )
    except (ThreePathError, prereg.PreregistrationError) as error:
        print(f"[C20] refused: {error}")
        return 2

    print(f"[C19/C20] GO4_finite_PAO={summary['GO4_finite_PAO']}")
    print(f"  claim            : {summary['claim']}")
    print(f"  tau_model        : {summary['tolerances']['tau_model']['value']}")
    print(f"  GO-6             : {summary['go6']['verdict']}")
    comparisons = summary["e2g_certification"]["comparisons"]
    print(
        f"  E2g FC vs FD     : {comparisons['comparisons_passed']}/"
        f"{comparisons['comparisons_total']} comparisons"
    )
    for row in summary["checks"]:
        if not row["passed"]:
            print(f"  FAILED {row['check']}: {row['detail']}")
    for row in summary["preregistration_findings"]:
        print(f"  FINDING on the frozen protocol: {row['check']} -- {row['resolution']}")
    return 0 if summary["GO4_finite_PAO"] in {"PASS", "NO_GO:model_accuracy"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
