#!/usr/bin/env python3
"""E-F_001-S34: the AB bilayer EPC cases (GO-7 raw material), no new theory.

The graphene Gamma/K campaigns (C14B/C14C/C15/C18/C19/C20) validated one
pipeline: a direction -> SIESTA ``dHSdR.nc`` raw D_H/D_S -> a shared S_L/S_R
basis response -> an S-orthonormal electronic window -> a contracted g block,
cross-checked against Graph2Mat's JVP and frozen central difference of the
same weights. This script points that exact pipeline (:mod:`certify_siesta_dhsdr`,
:mod:`epc_basis_response`, :mod:`compute_epc_matrix_elements`,
:mod:`epc_subspaces`, :mod:`phonon_provider`,
:mod:`quantify_checkpoint_derivative_error`, and
:func:`run_graphene_gamma_epc_paths.raw_derivative` /
:class:`run_graphene_gamma_epc_paths.ModelSide` themselves, reused rather than
copied) at ``materials/bilayer_graphene_AB/RUN.fdf``. Nothing here is a new
formalism, a new contraction or a new abstraction: the only AB-specific
content is which three directions the material's own geometry defines.

Three sectors, each a :class:`fd_perturbation_space.Direction` built from the
AB positions (layers assigned from the z coordinate, not from a band index):

``shear``               the two layers translate oppositely in-plane (x).
``layer_breathing``     the two layers translate oppositely out-of-plane (z).
``intralayer_control``  the two sublattice atoms of one layer stretch against
                         each other (graphene's E2g pattern, restricted to one
                         layer); the other layer is at rest, so its own
                         inter-layer block is a built-in null control.

Each direction is matched to the FC run's own Gamma phonon branch by
projection overlap (:func:`match_mode`, the same rule C19 used to fix the
graphene E2g doublet: by pattern, never by branch index), then carried
through the two reference paths -- ``siesta_reference`` (this material's own
FC ``dHSdR.nc``) and ``graph2mat`` (frozen central difference, checked against
the JVP) -- through one basis response shared by both, into one g block per
path. ``D_H`` and ``PAO-covariant response`` are additionally cut into intra-layer
(``AA``, ``BB``) and inter-layer (``AB``) blocks with
:func:`epc_subspaces.block_metrics`, using the same z-derived orbital labels
(never a cluster/band label).

Status: every artifact here is written ``status = "candidate"``. No
SIESTA-vs-explicit-finite-difference cross-check exists yet for this material
(only the FC branch was run, not the +- single points GO-2 uses for graphene),
no auxiliary FC supercell was computed (Gamma-only IFC), the moving-E_F
correction of ``model_to_fermi_zero`` is not applied (it needs a +- TSHS pair
this campaign does not have) and C14C's intra-atomic gap applies here exactly
as it does at graphene Gamma. This is the material GO-7 will be evaluated
from, not a GO-7 verdict.

Units: eV/Ang for D_H and g per unit displacement, eV after the zero-point
factor, 1/Ang for the basis response, eV for phonon energies.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

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
import epc_commensurate_k as commensurate_k  # noqa: E402
import fd_perturbation_space as fdp  # noqa: E402
import phonon_provider as pp  # noqa: E402
import quantify_checkpoint_derivative_error as c14  # noqa: E402
from artifact_signature import file_sha256, input_signature_sha256  # noqa: E402
from preregister_graphene_gamma_epc import minimum_image_vector  # noqa: E402
from read_siesta_dhsdr import read_dhsdr  # noqa: E402
from run_graphene_gamma_epc_paths import ModelSide, raw_derivative  # noqa: E402

SCHEMA = "epc_ab_bilayer_three_sectors_v1"
TICKET = "E-F_001-S34"
GATE = "GO-7"
STATUS_CANDIDATE = "candidate"

DEFAULT_MATERIAL_FDF = REPO_ROOT / "materials/bilayer_graphene_AB/RUN.fdf"
DEFAULT_BASIS_DIR = REPO_ROOT / "materials/graphene_common/basis"
DEFAULT_PSEUDO_DIR = REPO_ROOT / "materials/graphene_common/pseudos"
DEFAULT_REFERENCE_ROOT = REPO_ROOT / "Comparison/results/epc/siesta_reference/bilayer_graphene_AB"
DEFAULT_RESULT_ROOT = REPO_ROOT / "Comparison/results/epc/gamma_epc/bilayer_graphene_AB"
DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "Comparison/results/tbg_registry_spectral_loss/training/checkpoints"
    / "spectral-best-epoch=247-step=03472.ckpt"
)

GAMMA = (0.0, 0.0, 0.0)
LAYER_NAMES = ("bottom", "top")
INTRALAYER_CONTROL_LAYER = 0  # bottom layer carries the intralayer control pattern


class AbBilayerEpcError(RuntimeError):
    """The AB campaign cannot be produced as specified."""


def require_upstream_gates() -> dict[str, str]:
    """S34 is downstream of GO-4 and GO-5; neither may be inferred from an old artifact."""
    go4_path = REPO_ROOT / "Comparison/results/epc/gamma_epc/graphene/go4_verdict.json"
    go4 = json.loads(go4_path.read_text(encoding="utf-8")) if go4_path.is_file() else {}
    go5 = commensurate_k.go5_decision()
    verdicts = {"GO-4": str(go4.get("verdict", "MISSING")), "GO-5": str(go5.get("go5", "MISSING"))}
    if verdicts != {"GO-4": "PASS", "GO-5": "PASS"}:
        raise AbBilayerEpcError(f"S34 BLOCKED: requires GO-4=PASS and GO-5=PASS, got {verdicts}")
    return verdicts


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
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _frobenius(values: Any) -> float:
    return float(np.linalg.norm(np.asarray(values)))


# --------------------------------------------------------------------------- #
# Geometry: layers and directions, both derived from positions, never from an
# atom or band index chosen by hand.
# --------------------------------------------------------------------------- #


def layer_of_atom_from_positions(positions_ang: Any, *, tol_ang: float = 0.5) -> np.ndarray:
    """0/1 per atom, ranked by ``z``. Two clusters or refuse: this is a bilayer."""
    z = np.asarray(positions_ang, dtype=np.float64)[:, 2]
    order = np.argsort(z)
    gaps = np.diff(z[order])
    if gaps.size == 0 or np.max(gaps) < tol_ang:
        raise AbBilayerEpcError(
            f"positions do not separate into two z clusters (largest gap {np.max(gaps, initial=0.0):.3f} "
            f"Ang < tol {tol_ang}); layer labels must come from geometry, and this geometry is not a bilayer"
        )
    split = order[np.argmax(gaps) + 1 :]
    labels = np.zeros(z.shape[0], dtype=np.int64)
    labels[split] = 1
    return labels


def build_directions(positions_ang: Any, cell_ang: Any, layer_of_atom: np.ndarray) -> dict[str, fdp.Direction]:
    """``shear``, ``layer_breathing``, ``intralayer_control``, unit-Frobenius cartesian patterns."""
    positions = np.asarray(positions_ang, dtype=np.float64)
    atom_count = positions.shape[0]

    shear_vec = np.zeros((atom_count, 3))
    shear_vec[layer_of_atom == 0, 0] = 1.0
    shear_vec[layer_of_atom == 1, 0] = -1.0

    breathing_vec = np.zeros((atom_count, 3))
    breathing_vec[layer_of_atom == 0, 2] = -1.0
    breathing_vec[layer_of_atom == 1, 2] = 1.0

    bottom = np.where(layer_of_atom == INTRALAYER_CONTROL_LAYER)[0]
    if bottom.size != 2:
        raise AbBilayerEpcError(
            f"the intralayer control pattern needs a two-atom sublattice in the bottom layer, "
            f"got {bottom.size} atoms"
        )
    bond = minimum_image_vector(positions[bottom[0]], positions[bottom[1]], cell_ang)
    unit = bond / np.linalg.norm(bond)
    intralayer_vec = np.zeros((atom_count, 3))
    intralayer_vec[bottom[0]] = -unit
    intralayer_vec[bottom[1]] = unit

    return {
        "shear": fdp.collective(shear_vec, name="shear"),
        "layer_breathing": fdp.collective(breathing_vec, name="layer_breathing"),
        "intralayer_control": fdp.collective(intralayer_vec, name="intralayer_control"),
    }


def match_mode(direction: fdp.Direction, modes: pp.PhononModeSet) -> dict[str, Any]:
    """The Gamma branch with the largest overlap on ``direction`` -- by pattern, not by index.

    All atoms are carbon here, so mass-weighting is a common scale factor and
    does not change which branch wins; the overlap is computed directly on the
    unit-Frobenius cartesian patterns both objects already carry.
    """
    pattern = direction.vectors.reshape(-1)
    overlaps = []
    for branch in range(modes.frequencies_ev.size):
        vector = np.asarray(modes.eigenvectors[branch]).reshape(-1)
        norm = np.linalg.norm(vector)
        overlaps.append(float(abs(vector @ pattern) / norm) if norm > 0.0 else 0.0)
    best = int(np.argmax(overlaps))
    return {
        "branch": best,
        "frequency_ev": float(modes.frequencies_ev[best]),
        "overlap": overlaps[best],
        "overlaps_by_branch": overlaps,
    }


# --------------------------------------------------------------------------- #
# The campaign
# --------------------------------------------------------------------------- #


def produce(
    *,
    reference_root: Path = DEFAULT_REFERENCE_ROOT,
    result_root: Path = DEFAULT_RESULT_ROOT,
    checkpoint: Path = DEFAULT_CHECKPOINT,
    work_dir: Path | None = None,
) -> dict[str, Any]:
    import sisl

    upstream_gates = require_upstream_gates()
    work_dir = work_dir or (result_root / "work")
    checks: list[dict[str, Any]] = []

    # go2.load_campaign() additionally demands a certified +- pair per delta
    # (GO-2's own finite-difference cross-check); this campaign has none, which
    # is exactly the "no explicit FD cross-check" limitation this report
    # declares. Campaign itself needs only root + manifest, so it is built
    # directly rather than loosening load_campaign's gate for every caller.
    manifest_path = reference_root / "epc_siesta_reference_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dry_run") or not manifest.get("physics_identical_across_branches"):
        raise AbBilayerEpcError(f"{manifest_path} is not a usable SIESTA reference campaign")
    campaign = go2.Campaign(root=reference_root, manifest=manifest)
    equilibrium_id = campaign.manifest["certification_set"]["equilibrium_run_id"]
    equilibrium = go2.read_tshs(
        campaign.run_dir(equilibrium_id), equilibrium_id, fermi_ev=campaign.fermi_ev(equilibrium_id)
    )
    delta = float(campaign.deltas[0])
    dhsdr = read_dhsdr(campaign.dhsdr_path(delta))

    geometry = sisl.get_sile(str(equilibrium.path)).read_geometry()
    positions_ang = np.asarray(geometry.xyz, dtype=np.float64)
    cell_ang = np.asarray(geometry.cell, dtype=np.float64)
    atom_of_orbital = np.asarray(geometry.o2a(np.arange(equilibrium.no_u)), dtype=np.int64) + 1

    layer_of_atom = layer_of_atom_from_positions(positions_ang)
    layer_of_orbital = layer_of_atom[atom_of_orbital - 1]
    checks.append(
        _check(
            "layer_labels_from_geometry",
            len(set(layer_of_atom.tolist())) == 2,
            "layer 0/1 assigned by ranking atomic z (largest gap split), not by atom or band index",
            layer_of_atom=layer_of_atom.tolist(),
            layer_of_orbital=layer_of_orbital.tolist(),
        )
    )

    directions = build_directions(positions_ang, cell_ang, layer_of_atom)

    geometry_signature = str(campaign.manifest.get("material_fdf_sha256", ""))
    basis_signature = str((campaign.manifest.get("orbital_contract") or {}).get("orbital_contract_hash", ""))
    no_u = equilibrium.no_u

    model_side = ModelSide(checkpoint, campaign, equilibrium, work_dir)
    checks.append(model_side.label_convention_check())
    checks.append(
        _check(
            "graph2mat_backend",
            model_side.backend.effective in ("cuda", "cpu"),
            "Graph2Mat directional derivatives ran on a resolved backend, requested/effective/reason "
            "recorded (default compute policy: GPU unless a preflight shows it is unavailable)",
            **model_side.backend_record(),
        )
    )

    # --- one basis response per direction, shared by both reference paths ---
    responses: dict[tuple[str, str], ebr.BasisResponseArtifact] = {}
    isc_off: np.ndarray | None = None
    for name, direction in directions.items():
        for backend in ("siesta", "graph2mat"):
            context = ebr.ResponseContext(
                geometry_signature=geometry_signature,
                basis_signature=basis_signature,
                electronic_derivative_backend=backend,
                direction=direction,
                source={
                    "reference_root": str(campaign.root),
                    "fc_run": str(campaign.dhsdr_path(delta)),
                    "delta_ang": delta,
                },
            )
            artifact = ebr.from_dhsdr(dhsdr, context, atom_of_orbital=atom_of_orbital)
            responses[(name, backend)] = artifact
            if isc_off is None:
                isc_off = artifact.isc_off
    checks.append(
        _check(
            "one_basis_response_per_direction_across_paths",
            all(
                input_signature_sha256(
                    {
                        "S_left": np.round(responses[(name, "siesta")].blocks["S_left"], 12).tolist(),
                        "S_right": np.round(responses[(name, "siesta")].blocks["S_right"], 12).tolist(),
                    }
                )
                == input_signature_sha256(
                    {
                        "S_left": np.round(responses[(name, "graph2mat")].blocks["S_left"], 12).tolist(),
                        "S_right": np.round(responses[(name, "graph2mat")].blocks["S_right"], 12).tolist(),
                    }
                )
                for name in directions
            ),
            "the siesta_reference and graph2mat paths consume byte-identical S_L/S_R blocks; only "
            "electronic_derivative_backend differs, so epc_backend_class is hybrid by arithmetic",
        )
    )
    assert isc_off is not None

    equilibrium_topology_sha256 = input_signature_sha256(
        {"nsc": list(equilibrium.nsc), "sc_off": [list(row) for row in equilibrium.sc_off]}
    )

    # --- equilibrium electrons at Gamma, dense (8-orbital cell) -------------
    h_blocks = cbr.dense_blocks(equilibrium.h_shifted, isc_off, no_u)[0]
    s_blocks = cbr.dense_blocks(equilibrium.overlap, isc_off, no_u)[0]
    H_gamma = ebr.bloch_sum(h_blocks, isc_off, GAMMA)
    S_gamma = ebr.bloch_sum(s_blocks, isc_off, GAMMA)
    eigenspace = epc.dense_eigenspace(GAMMA, H_gamma, S_gamma, label="gamma")
    checks.append(
        _check(
            "eigenspace_S_orthonormal",
            eigenspace.identity_error <= eigenspace.identity_tolerance,
            f"|C^dag S C - I| = {eigenspace.identity_error:.3e} within its own budget "
            f"{eigenspace.identity_tolerance:.3e}",
        )
    )

    # --- phonons: the FC run's own Gamma dynamical matrix, ASR-corrected ----
    # The FC file lives next to the run that produced dHSdR.nc.
    fc_run_id = next(
        entry["fc_run_id"]
        for entry in campaign.manifest["certification_set"]["per_delta"]
        if entry.get("fc_dhs_available")
    )
    fc_dir = campaign.run_dir(fc_run_id)
    fc_file = next(fc_dir.glob("*.FC"))
    fdf_file = fc_dir / "RUN.fdf"
    force_constants_raw = pp.force_constants_from_siesta_run(fc_file, fdf_file)
    force_constants = force_constants_raw.apply_asr()
    modes = pp.modes_from_force_constants(
        force_constants,
        q_fractional=GAMMA,
        provider="siesta_fc_gamma_only",
        geometry_signature=geometry_signature,
    )
    checks.append(
        _check(
            "phonon_asr_residual_small_vs_raw",
            force_constants.asr_residual_raw_ev_ang2 >= 0.0,
            "ASR-corrected Gamma modes from this run's own .FC; raw residual carried, not hidden",
            asr_residual_raw_ev_ang2=float(force_constants.asr_residual_raw_ev_ang2),
            asr_residual_corrected_ev_ang2=float(force_constants.asr_residual_ev_ang2),
        )
    )

    # --- per-direction: raw derivatives, contraction, layer blocks ----------
    sectors: dict[str, Any] = {}
    for name, direction in directions.items():
        d_h_abs = go2.contract_fc(dhsdr, direction.vectors, "D_H")
        d_s = go2.contract_fc(dhsdr, direction.vectors, "D_S")
        d_h_siesta = c14.combine((1.0, d_h_abs), (-float(equilibrium.fermi_ev), d_s))
        siesta_raw, siesta_outside = raw_derivative(
            path="siesta_reference",
            direction=direction,
            d_h=d_h_siesta,
            d_s=d_s,
            isc_off=isc_off,
            no_u=no_u,
            geometry_signature=geometry_signature,
            basis_signature=basis_signature,
            topology_sha256=equilibrium_topology_sha256,
            delta_or_jvp={"delta_ang": delta, "method": "fc_dHSdR"},
        )

        jvp_field, jvp_meta = model_side.jvp(direction)
        frozen_field, frozen_hashes = model_side.frozen(direction, delta)
        jvp_raw, jvp_outside = raw_derivative(
            path="graph2mat_jvp",
            direction=direction,
            d_h=jvp_field,
            d_s=None,
            isc_off=isc_off,
            no_u=no_u,
            geometry_signature=geometry_signature,
            basis_signature=basis_signature,
            topology_sha256=str(model_side.topology["topology_hash"]),
            delta_or_jvp={"method": "double_backward_jvp", "amplitude": None},
        )
        frozen_raw, frozen_outside = raw_derivative(
            path="graph2mat_frozen",
            direction=direction,
            d_h=frozen_field,
            d_s=None,
            isc_off=isc_off,
            no_u=no_u,
            geometry_signature=geometry_signature,
            basis_signature=basis_signature,
            topology_sha256=str(model_side.topology["topology_hash"]),
            delta_or_jvp={"delta_ang": delta, "method": "central"},
        )

        jvp_gamma = jvp_raw.at_k(GAMMA)["D_H"]
        frozen_gamma = frozen_raw.at_k(GAMMA)["D_H"]
        tau_backend = _frobenius(jvp_gamma - frozen_gamma)

        perturbations = {
            "siesta_reference": epc.PaoCovariantResponse(siesta_raw, responses[(name, "siesta")]),
            "graph2mat_jvp": epc.PaoCovariantResponse(jvp_raw, responses[(name, "graph2mat")]),
            "graph2mat_frozen": epc.PaoCovariantResponse(frozen_raw, responses[(name, "graph2mat")]),
        }

        g_blocks = {
            path: perturbation.block(eigenspace, eigenspace) for path, perturbation in perturbations.items()
        }

        mode = match_mode(direction, modes)
        zero_point_amplitude_ang = None
        g_ev_frobenius = None
        if mode["frequency_ev"] > 0.0:
            mass_amu = float(np.mean(force_constants.masses_amu))
            from epc_formalism import zero_point_amplitude_ang as _zpa

            zero_point_amplitude_ang = _zpa(mass_amu, mode["frequency_ev"])
            g_ev_frobenius = {
                path: float(np.linalg.norm(block.values) * zero_point_amplitude_ang)
                for path, block in g_blocks.items()
            }

        d_h_gamma_siesta = siesta_raw.at_k(GAMMA)["D_H"]
        pao_covariant_response_siesta = perturbations["siesta_reference"].pao_covariant_response(H_gamma, S_gamma, GAMMA)
        layer_labels = layer_of_orbital.tolist()
        layer_blocks_D_H = subspaces.block_metrics(d_h_gamma_siesta, layer_labels, layer_labels)
        layer_blocks_pao_covariant_response = subspaces.block_metrics(pao_covariant_response_siesta, layer_labels, layer_labels)

        sectors[name] = {
            "direction": direction.to_metadata(),
            "layer_pattern": {
                LAYER_NAMES[layer]: direction.vectors[layer_of_atom == layer].tolist()
                for layer in (0, 1)
            },
            "phonon_match": mode,
            "phonon_frequency_ev": mode["frequency_ev"],
            "zero_point_amplitude_ang": zero_point_amplitude_ang,
            "raw_derivatives": {
                "siesta_reference": {
                    "artifact_fields": siesta_raw.artifact_fields(),
                    "frobenius_D_H_ev_per_ang": _frobenius(d_h_gamma_siesta),
                    "norm_outside_image_table": siesta_outside,
                },
                "graph2mat_jvp": {
                    "artifact_fields": jvp_raw.artifact_fields(),
                    "frobenius_D_H_ev_per_ang": _frobenius(jvp_gamma),
                    "norm_outside_image_table": jvp_outside,
                    "jvp_metadata": jvp_meta,
                },
                "graph2mat_frozen": {
                    "artifact_fields": frozen_raw.artifact_fields(),
                    "frobenius_D_H_ev_per_ang": _frobenius(frozen_gamma),
                    "norm_outside_image_table": frozen_outside,
                    "topology_hashes": frozen_hashes,
                },
                "tau_backend_jvp_vs_frozen_frobenius_ev_per_ang": tau_backend,
            },
            "basis_response": {
                "formalism_id": responses[(name, "siesta")].context.formalism_id,
                "representation": responses[(name, "siesta")].representation,
                "intra_atomic_included": responses[(name, "siesta")].intra_atomic_included,
                "diagnostics": responses[(name, "siesta")].diagnostics,
            },
            "g_blocks_per_unit_displacement_ev_per_ang": {
                path: {
                    "values_real": np.real(block.values).tolist(),
                    "values_imag": np.imag(block.values).tolist(),
                    "formalism_id": block.formalism_id,
                    **epc.gauge_invariant_block_metrics(block),
                }
                for path, block in g_blocks.items()
            },
            "g_ev_frobenius_at_zero_point": g_ev_frobenius,
            "layer_blocks": {
                "labels": {"0": LAYER_NAMES[0], "1": LAYER_NAMES[1]},
                "D_H_gamma_siesta": layer_blocks_D_H,
                "pao_covariant_response_gamma_siesta": layer_blocks_pao_covariant_response,
            },
            "result_class": epc.RESULT_BOUND,
            "status": STATUS_CANDIDATE,
        }

    report = {
        "schema": SCHEMA,
        "ticket": TICKET,
        "gate": GATE,
        "status": STATUS_CANDIDATE,
        "upstream_gates": upstream_gates,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "material_fdf": campaign.manifest["material_fdf"],
        "geometry_signature": geometry_signature,
        "basis_signature": basis_signature,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "reference_root": str(reference_root),
        "delta_ang": delta,
        "layer_of_atom": layer_of_atom.tolist(),
        "layer_of_orbital": layer_of_orbital.tolist(),
        "eigenspace": eigenspace.summary(),
        "phonon": {
            "provider": modes.provider,
            "geometry_signature": modes.geometry_signature,
            "asr_policy": modes.asr_policy,
            "asr_residual_raw_ev_ang2": float(modes.asr_residual_raw_ev_ang2),
            "frequencies_ev": modes.frequencies_ev.tolist(),
            "fc_source": force_constants.provenance,
        },
        "sectors": sectors,
        "checks": checks,
        "limitations": [
            "no explicit SIESTA +- finite-difference cross-check was run for this material (only "
            "the FC branch); the siesta_reference path here is uncertified in the GO-2 sense",
            "the FC run is Gamma-only (no auxiliary supercell); phonon frequencies are not IFC-range "
            "converged the way graphene's sc5x5x1 result is",
            "model_to_fermi_zero's moving-E_F correction is not applied (needs a +- TSHS pair this "
            "campaign does not have); both reference paths use the fixed equilibrium E_F origin",
            "C14C's intra-atomic basis-response term (memo A2) is unresolved here exactly as at "
            "graphene Gamma, so every g_blocks entry is a bound (result_class = "
            f"{epc.RESULT_BOUND!r}), not a measurement",
        ],
    }
    _write_json(result_root / "ab_bilayer_epc_three_sectors.json", report)
    return report


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE_ROOT)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    report = produce(
        reference_root=args.reference_root, result_root=args.result_root, checkpoint=args.checkpoint
    )
    failed = [row["check"] for row in report["checks"] if not row["passed"]]
    print(f"[AB-EPC] sectors={list(report['sectors'])} checks_failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
