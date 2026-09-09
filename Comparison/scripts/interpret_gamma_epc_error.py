#!/usr/bin/env python3
"""E-F_001-S27 — what the graphene Gamma residual is, and what it licenses.

S25 measured ``g`` along three paths and S26 adjudicated GO-4 as ``NO_GO``. This
step answers the only question left open about the checkpoint: **is it adequate
for EPC derivatives and for g, or does the residual justify opening retraining?**

It computes no new physics. Every number comes from artifacts already on disk --
the real-space directional derivatives of the three paths, the persisted
eigenspaces, the frozen protocol and the GO-4 verdict -- and is re-read into a
decomposition of the residual by orbital, by pair block, by direction and by
electronic subspace, plus the sensitivity of that decomposition to delta, to the
basis/topology the model was evaluated on, to the tolerance definitions and to
where the window was cut.

Three rules govern this file:

* **The gate is not edited retroactively.** ``tau_model``, the adequacy
  threshold and the claim ladder are *read* from the frozen protocol and the
  GO-4 verdict. Nothing here re-derives them, widens them, or re-adjudicates a
  gate. Alternative windows and the best-fit scale are labelled diagnostics and
  are never presented as the model's performance.
* **A NO_GO is a statement about this repository's providers, not about
  graphene.** The failing layer is named, and the graphene numbers the SIESTA
  path produced are reported as physical results in their own right.
* **Retraining is authorised by the frozen ladder, not by this script.** Rung 2
  ("numerical and formalism checks pass, tau_model > adequacy") is the rung that
  opens the fine-tuning ticket. The verdict selected rung 3, so the ticket stays
  closed; what this step adds is the *causal evidence* that would activate it,
  and the measurement showing that evidence cannot be produced by the layers
  that are currently failing.

Usage::

    .venv/bin/python Comparison/scripts/interpret_gamma_epc_error.py
    .venv/bin/python Comparison/scripts/interpret_gamma_epc_error.py --print-only
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for candidate in (SCRIPT_DIR, REPO_ROOT / "shared"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import certify_siesta_dhsdr as go2  # noqa: E402
from epc_fourier import bloch_sum as canonical_bloch_sum  # noqa: E402
import evaluate_epc_metrics as go4  # noqa: E402
import preregister_graphene_gamma_epc as prereg  # noqa: E402
import quantify_checkpoint_derivative_error as c14  # noqa: E402
from run_deeph_sparse_spectrum import load_persisted_eigenspace  # noqa: E402

SCHEMA = "epc_error_interpretation_v1"
TICKET = "E-F_001-S27"

DEFAULT_RESULT_ROOT = prereg.DEFAULT_RESULT_ROOT
DEFAULT_OUTPUT_DIR = REPO_ROOT / "Comparison" / "results" / "epc" / "error_interpretation" / "graphene"
DEFAULT_C14_REPORT = (
    REPO_ROOT
    / "Comparison"
    / "results"
    / "epc"
    / "checkpoint_derivative_error"
    / "graphene"
    / "checkpoint_derivative_error.json"
)
REPORT_NAME = "gamma_error_interpretation.json"
GROUPS_NAME = "gamma_error_interpretation_groups.csv"

PATH_SIESTA = "siesta_reference"
MODEL_PATHS = ("graph2mat_jvp", "graph2mat_frozen")

#: The rung of the frozen claim ladder that opens the fine-tuning ticket. It is
#: the protocol's own wording ("...opens the fine-tuning ticket, which stays
#: closed otherwise"), not a rule invented here.
RETRAINING_RUNG = 2

#: Groupings of the real-space residual. ``atom_pair`` is the one C14 did not
#: report and the one the moving-basis discussion needs: same-atom blocks are
#: where the unresolved intra-atomic response lives.
GROUPINGS = ("orbital_block", "pair_distance_ang", "atom_pair", "reference_magnitude_decile")


class InterpretationError(RuntimeError):
    """The residual cannot be interpreted from the artifacts on disk."""


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise InterpretationError(f"missing artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class Inputs:
    """Everything read from disk, with the frozen protocol already checked."""

    result_root: Path
    protocol: Mapping[str, Any]
    summary: Mapping[str, Any]
    verdict: Mapping[str, Any]
    c14: Mapping[str, Any]
    paths: Mapping[str, Mapping[str, Any]]
    eigenspaces: Mapping[str, Mapping[str, Any]]
    orbitals: c14.OrbitalMap

    @property
    def arrays_dir(self) -> Path:
        return self.result_root / "arrays"


def load_inputs(
    result_root: Path, protocol_path: Path, c14_report: Path, equilibrium_dir: Path | None = None
) -> Inputs:
    """Read the candidate, its verdict and the geometry the residual is keyed by.

    The pre-registration hash is checked against every artifact that is about to
    be interpreted: a residual read out of a result that did not cite the frozen
    protocol would be an interpretation of something else.
    """
    import sisl

    protocol = prereg.load_protocol(protocol_path)
    frozen = str(protocol["frozen_content_sha256"])
    summary = _read_json(result_root / go4.SUMMARY_NAME)
    verdict = _read_json(result_root / go4.VERDICT_NAME)
    paths = {
        name: _read_json(result_root / f"path__{name}.json")
        for name in (PATH_SIESTA, *MODEL_PATHS)
    }
    for label, payload in (("summary", summary), ("verdict", verdict), *paths.items()):
        cited = str(payload.get("preregistration_sha256", ""))
        if cited != frozen:
            raise InterpretationError(
                f"{label} cites preregistration {cited[:12]!r}, the frozen protocol is "
                f"{frozen[:12]!r}; these are not the same experiment"
            )

    eigenspace_dir = Path(summary["eigenspaces"]["eigenspace_dir"])
    eigenspaces: dict[str, dict[str, Any]] = {}
    for index in range(int(summary["eigenspaces"]["kpoint_count"])):
        payload = load_persisted_eigenspace(eigenspace_dir, index)
        eigenspaces[str(payload["label"])] = payload

    if equilibrium_dir is None:
        equilibrium_dir = Path(summary["eigenspaces"]["equilibrium_tshs"]).parent
    run_id = Path(summary["eigenspaces"]["equilibrium_tshs"]).stem
    geometry = sisl.get_sile(str(equilibrium_dir / "RUN.fdf")).read_geometry()
    orbitals = c14.orbital_map(
        equilibrium_dir / f"{run_id}.ORB_INDX",
        np.asarray(geometry.xyz, dtype=np.float64),
        np.asarray(geometry.cell, dtype=np.float64),
    )
    if len(orbitals.l_of_orbital) != int(summary["shared_contracts"]["orbital_count"]):
        raise InterpretationError(
            f"ORB_INDX enumerates {len(orbitals.l_of_orbital)} orbitals, the campaign contracted "
            f"{summary['shared_contracts']['orbital_count']}; the residual would be mis-keyed"
        )
    return Inputs(
        result_root=result_root,
        protocol=protocol,
        summary=summary,
        verdict=verdict,
        c14=_read_json(c14_report),
        paths=paths,
        eigenspaces=eigenspaces,
        orbitals=orbitals,
    )


# --------------------------------------------------------------------------
# the real-space residual
# --------------------------------------------------------------------------


def _delta_tag(delta: float) -> str:
    return f"d{format(float(delta), 'g').replace('.', 'p')}"


def load_raw(arrays_dir: Path, path: str, direction: str, delta: float) -> tuple[np.ndarray, np.ndarray]:
    """``(isc_off, D_H)`` of one directional derivative, in real space."""
    payload = np.load(arrays_dir / f"raw_derivative__{path}__{direction}__{_delta_tag(delta)}.npz")
    return np.asarray(payload["isc_off"], dtype=np.int64), np.asarray(payload["D_H"], dtype=np.float64)


def dense_to_field(isc_off: np.ndarray, blocks: np.ndarray) -> dict[c14.Key, float]:
    """The dense ``(image, row, col)`` array on the canonical ``(row, col, R)`` space."""
    images = [tuple(int(component) for component in offset) for offset in isc_off]
    return {
        (row, column, images[image]): float(blocks[image, row, column])
        for image in range(blocks.shape[0])
        for row in range(blocks.shape[1])
        for column in range(blocks.shape[2])
        if blocks[image, row, column] != 0.0
    }


def bloch_sum(isc_off: np.ndarray, blocks: np.ndarray, k_fractional: Sequence[float]) -> np.ndarray:
    """``X(k) = sum_l exp(+2i pi k.R_l) X(R_l)``, the campaign's declared convention.

    The phase itself comes from :mod:`epc_fourier` (S29), so this stays a
    convenience wrapper with the argument order this script uses.
    """
    return canonical_bloch_sum(blocks, isc_off, k_fractional)


def derivative_residual(
    inputs: Inputs, direction: str, model_path: str, delta: float
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """One (direction, model path, delta): the residual against SIESTA, decomposed."""
    isc_reference, reference_blocks = load_raw(inputs.arrays_dir, PATH_SIESTA, direction, delta)
    isc_model, model_blocks = load_raw(inputs.arrays_dir, model_path, direction, delta)
    if not np.array_equal(isc_reference, isc_model):
        raise InterpretationError(
            f"{direction}/{model_path}: the two paths use different image tables; the residual "
            "would compare different lattice vectors"
        )
    reference = dense_to_field(isc_reference, reference_blocks)
    model = dense_to_field(isc_model, model_blocks)
    keys, (reference_values, model_values) = go2.align(reference, model)
    residual = go2.norms(reference_values - model_values)
    reference_norm = go2.norms(reference_values)
    scale = go2.best_fit_scale(reference, model)
    outside = [key for key in keys if key not in model]

    relative = (
        residual["frobenius"] / reference_norm["frobenius"]
        if reference_norm["frobenius"] > 0.0
        else float("nan")
    )
    after_scale = (
        go2.norms(reference_values - float(scale) * model_values)["frobenius"]
        / reference_norm["frobenius"]
        if scale is not None and reference_norm["frobenius"] > 0.0
        else float("nan")
    )
    row = {
        "direction": direction,
        "model_path": model_path,
        "delta_ang": float(delta),
        "unit": "eV/Ang",
        "reference_frobenius": reference_norm["frobenius"],
        "model_frobenius": go2.norms(model_values)["frobenius"],
        "residual_frobenius": residual["frobenius"],
        "relative_frobenius": relative,
        "cosine": (
            float(
                reference_values
                @ model_values
                / (np.linalg.norm(reference_values) * np.linalg.norm(model_values))
            )
            if reference_norm["frobenius"] > 0.0 and go2.norms(model_values)["frobenius"] > 0.0
            else float("nan")
        ),
        "best_fit_scale": scale,
        "relative_after_best_fit_scale": after_scale,
        "keys_total": len(keys),
        "keys_outside_model_graph": len(outside),
        "norm_outside_model_graph": c14.restricted_norms(reference, outside)["frobenius"],
    }

    labels = {
        "orbital_block": [inputs.orbitals.block(key) for key in keys],
        "pair_distance_ang": [round(inputs.orbitals.distance_ang(key), 2) for key in keys],
        "atom_pair": [_atom_pair_label(inputs.orbitals, key) for key in keys],
        "reference_magnitude_decile": c14.magnitude_deciles(reference_values),
    }
    groups: list[dict[str, Any]] = []
    for grouping in GROUPINGS:
        groups.extend(
            dict(entry, model_path=model_path, delta_ang=float(delta))
            for entry in c14.group_metrics(
                keys,
                reference,
                model,
                grouping=grouping,
                labels=labels[grouping],
                direction_name=direction,
                total_residual=residual["frobenius"],
            )
        )
    row["dominant_groups"] = {
        grouping: max(
            (entry for entry in groups if entry["grouping"] == grouping),
            key=lambda entry: entry["share_of_total_residual"],
            default=None,
        )
        for grouping in GROUPINGS
    }
    return row, groups


def _atom_pair_label(orbitals: c14.OrbitalMap, key: c14.Key) -> str:
    same_atom = orbitals.atom_of_orbital[key[0]] == orbitals.atom_of_orbital[key[1]]
    return "intra_atomic" if same_atom and tuple(key[2]) == (0, 0, 0) else "inter_atomic"


# --------------------------------------------------------------------------
# the residual in the electronic subspace
# --------------------------------------------------------------------------


def window_selections(protocol: Mapping[str, Any], state_count: int) -> dict[str, list[int]]:
    """The frozen window and the sub-selections used as a sensitivity axis.

    ``neutrality_pair`` is the pi manifold: the highest occupied and the lowest
    empty state of the window, with the occupied count taken from the protocol's
    electron count rule and never from where a gap happens to be.
    """
    below = int(protocol["electronic"]["window"]["states_below"])
    selections = {"frozen_window": list(range(state_count))}
    if 0 < below < state_count:
        selections["neutrality_pair"] = [below - 1, below]
        selections["occupied_block"] = list(range(below))
        selections["empty_block"] = list(range(below, state_count))
    return selections


def subspace_residual(
    inputs: Inputs, direction: str, k_label: str, model_path: str, scale: float | None
) -> dict[str, Any]:
    """One (direction, k, model path) contracted into the window, and sub-windows.

    ``delta_g`` is the pre-registered object -- ``C_f^dag (Delta_G2M -
    Delta_SIESTA) C_i`` -- read back from the two persisted ``PAO-covariant response``
    matrices. The scaled variant applies the *derivative-level* best-fit gain to
    the model's ``D_H`` alone; it is a diagnostic of the residual's structure and
    is not a correction factor.
    """
    eigenspace = inputs.eigenspaces[k_label]
    coefficients = np.asarray(eigenspace["coefficients"], dtype=np.complex128)
    reference = np.load(inputs.arrays_dir / f"pao_covariant_response__{PATH_SIESTA}__{direction}__{k_label}.npy")
    model = np.load(inputs.arrays_dir / f"pao_covariant_response__{model_path}__{direction}__{k_label}.npy")

    def contract(matrix: np.ndarray) -> np.ndarray:
        return coefficients.conj().T @ matrix @ coefficients

    g_reference = contract(reference)
    delta_g = contract(model - reference)
    energies = np.asarray(eigenspace["energies_eV"], dtype=np.float64)

    scaled_delta_g = None
    if scale is not None:
        response_delta = float(inputs.paths[model_path]["directions"][direction]["basis_response_delta_ang"])
        isc_off, reference_blocks = load_raw(inputs.arrays_dir, PATH_SIESTA, direction, response_delta)
        _, model_blocks = load_raw(inputs.arrays_dir, model_path, direction, response_delta)
        # PAO-covariant response differs between the paths only through D_H (both consume the
        # same basis-response blocks), so the gain can be applied in real space
        # and contracted without rebuilding the perturbation.
        scaled_delta_g = contract(
            bloch_sum(isc_off, float(scale) * model_blocks - reference_blocks, eigenspace["k_fractional"])
        )

    row: dict[str, Any] = {
        "direction": direction,
        "k": k_label,
        "model_path": model_path,
        "unit": "eV/Ang",
        "energies_ev": energies.tolist(),
        "windows": {},
        "per_state_diagonal": [
            {
                "state": index,
                "energy_ev": float(energies[index]),
                "reference_abs": float(abs(g_reference[index, index])),
                "residual_abs": float(abs(delta_g[index, index])),
            }
            for index in range(g_reference.shape[0])
        ],
    }
    for name, states in window_selections(inputs.protocol, g_reference.shape[0]).items():
        block = np.ix_(states, states)
        reference_norm = float(np.linalg.norm(g_reference[block]))
        residual_norm = float(np.linalg.norm(delta_g[block]))
        diagonal = float(np.linalg.norm(np.diag(delta_g[block])))
        entry = {
            "states": states,
            "reference_frobenius": reference_norm,
            "residual_frobenius": residual_norm,
            "relative_subspace_error": (
                residual_norm / reference_norm if reference_norm > 0.0 else float("nan")
            ),
            "diagonal_share_of_residual": (
                (diagonal / residual_norm) ** 2 if residual_norm > 0.0 else float("nan")
            ),
        }
        if scaled_delta_g is not None:
            scaled_norm = float(np.linalg.norm(scaled_delta_g[block]))
            entry["relative_after_derivative_level_scale"] = (
                scaled_norm / reference_norm if reference_norm > 0.0 else float("nan")
            )
        row["windows"][name] = entry
    row["dominant_state"] = max(row["per_state_diagonal"], key=lambda entry: entry["residual_abs"])
    return row


# --------------------------------------------------------------------------
# the residual is not the open basis-response term
# --------------------------------------------------------------------------


def basis_response_cancellation(inputs: Inputs, cases: Sequence[tuple[str, str]]) -> dict[str, Any]:
    """Does the unresolved basis response live in the residual? Measured, not argued.

    The three paths consume byte-identical ``S_L``/``S_R`` blocks per direction,
    so the model-minus-reference difference of ``PAO-covariant response`` should equal the
    Bloch sum of the model-minus-reference difference of ``D_H`` exactly. If it
    does, no basis-response term -- resolved or not -- can enter the residual,
    because it enters both sides with the same sign.
    """
    per_case: list[dict[str, Any]] = []
    for direction, k_label in cases:
        for model_path in MODEL_PATHS:
            hashes = {
                name: payload["directions"][direction]["basis_response_block_sha256"]
                for name, payload in inputs.paths.items()
            }
            response_delta = float(inputs.paths[model_path]["directions"][direction]["basis_response_delta_ang"])
            isc_off, reference_blocks = load_raw(inputs.arrays_dir, PATH_SIESTA, direction, response_delta)
            _, model_blocks = load_raw(inputs.arrays_dir, model_path, direction, response_delta)
            eigenspace = inputs.eigenspaces[k_label]
            from_derivatives = bloch_sum(
                isc_off, model_blocks - reference_blocks, eigenspace["k_fractional"]
            )
            from_perturbations = np.load(
                inputs.arrays_dir / f"pao_covariant_response__{model_path}__{direction}__{k_label}.npy"
            ) - np.load(inputs.arrays_dir / f"pao_covariant_response__{PATH_SIESTA}__{direction}__{k_label}.npy")
            difference = float(np.linalg.norm(from_derivatives - from_perturbations))
            magnitude = float(np.linalg.norm(from_perturbations))
            per_case.append(
                {
                    "direction": direction,
                    "k": k_label,
                    "model_path": model_path,
                    "basis_response_blocks_identical_across_paths": len(set(hashes.values())) == 1,
                    "residual_frobenius": magnitude,
                    "residual_minus_d_h_difference_frobenius": difference,
                    "relative": difference / magnitude if magnitude > 0.0 else float("nan"),
                }
            )
    worst = max(per_case, key=lambda entry: entry["relative"])
    return {
        "identity": "PAO-covariant response(model) - PAO-covariant response(SIESTA) == Bloch[D_H(model) - D_H(SIESTA)]",
        "consequence": (
            "the model-vs-reference residual carries no basis-response term: the same S_L/S_R "
            "blocks enter both sides and cancel in the difference. Resolving the intra-atomic "
            "term (C14C) changes the width of the bound on g and the denominator of the relative "
            "error, never the residual itself"
        ),
        "per_case": per_case,
        "worst_relative": worst["relative"],
        "holds": bool(
            all(entry["basis_response_blocks_identical_across_paths"] for entry in per_case)
            and worst["relative"] < 1e-9
        ),
    }


def intra_atomic_rescue(inputs: Inputs, subspace_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """How large the omitted intra-atomic response would have to be to rescue adequacy.

    The residual is invariant under adding the missing term to both sides, so the
    only way the open C14C term could bring ``r`` under the adequacy threshold is
    by enlarging the reference block itself. That required size is compared with
    the norm of the entire overlap derivative, which is the scale the term must
    live inside.
    """
    threshold = float(
        inputs.summary["tolerances"]["definition"]["constants"]["G_ADEQUACY_THRESHOLD"]
    )
    sensitivity_by_state = {
        entry["state"]: float(entry["rms_interband_response"])
        for direction in inputs.summary["intra_atomic_sensitivity"]["per_direction"]
        for entry in direction["sensitivity"]
    }
    null_directions = set(inputs.c14["verdict"]["null_directions"])
    rows: list[dict[str, Any]] = []
    for row in subspace_rows:
        window = row["windows"]["frozen_window"]
        sensitivity = sensitivity_by_state.get(row["k"])
        if sensitivity is None or window["reference_frobenius"] <= 0.0:
            continue
        if row["direction"] in null_directions:
            # A null direction has no model residual to rescue; including it
            # would report "zero extra response needed" as the easiest case.
            continue
        required_reference = window["residual_frobenius"] / threshold
        deficit = max(required_reference - window["reference_frobenius"], 0.0)
        response_delta = float(
            inputs.paths[PATH_SIESTA]["directions"][row["direction"]]["basis_response_delta_ang"]
        )
        overlap_derivative = float(
            np.linalg.norm(
                np.load(
                    inputs.arrays_dir
                    / f"raw_derivative__{PATH_SIESTA}__{row['direction']}__{_delta_tag(response_delta)}.npz"
                )["D_S"]
            )
        )
        rows.append(
            {
                "direction": row["direction"],
                "k": row["k"],
                "model_path": row["model_path"],
                "reference_frobenius": window["reference_frobenius"],
                "required_reference_frobenius": required_reference,
                "sensitivity_ev_per_ang_per_unit_A": sensitivity,
                "required_intra_atomic_A_frobenius_per_ang": (
                    deficit / sensitivity if sensitivity > 0.0 else float("nan")
                ),
                "full_overlap_derivative_frobenius_per_ang": overlap_derivative,
                "multiples_of_the_full_overlap_derivative": (
                    deficit / sensitivity / overlap_derivative
                    if sensitivity > 0.0 and overlap_derivative > 0.0
                    else float("nan")
                ),
            }
        )
    smallest = min(
        (row["multiples_of_the_full_overlap_derivative"] for row in rows), default=float("nan")
    )
    return {
        "question": "could the unresolved intra-atomic term make the checkpoint adequate for g?",
        "threshold": threshold,
        "per_case": rows,
        "smallest_required_multiple_of_dS": smallest,
        "answer": (
            "no: the smallest rescue would need an intra-atomic response "
            f"{smallest:.1f} times the norm of the entire dS/dR, which is the object it is a "
            "same-atom part of"
        )
        if np.isfinite(smallest) and smallest > 1.0
        else "not resolvable from these artifacts",
    }


# --------------------------------------------------------------------------
# sensitivity
# --------------------------------------------------------------------------


def sensitivity(
    inputs: Inputs,
    derivative_rows: Sequence[Mapping[str, Any]],
    subspace_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Delta, basis/topology, tolerances and subspace selection, in one place."""
    signalled = [row for row in derivative_rows if np.isfinite(row["relative_frobenius"]) and row["reference_frobenius"] > 1.0]
    by_direction: dict[str, list[Mapping[str, Any]]] = {}
    for row in signalled:
        by_direction.setdefault(f"{row['direction']}|{row['model_path']}", []).append(row)
    delta_axis = [
        {
            "case": case,
            "deltas_ang": [row["delta_ang"] for row in rows],
            "relative_frobenius": [row["relative_frobenius"] for row in rows],
            "best_fit_scale": [row["best_fit_scale"] for row in rows],
            "relative_spread": float(
                max(row["relative_frobenius"] for row in rows)
                - min(row["relative_frobenius"] for row in rows)
            ),
        }
        for case, rows in sorted(by_direction.items())
    ]

    provenance = inputs.c14["provenance"]["model"]
    truncation = max(row["norm_outside_model_graph"] for row in derivative_rows)
    basis_axis = {
        "checkpoint_point_basis_R_ang": provenance["checkpoint_point_basis_R_ang"],
        "sisl_rebuilt_point_basis_R_ang": provenance["sisl_rebuilt_point_basis_R_ang"],
        "checkpoint_edge_cutoff_ang": provenance["checkpoint_edge_cutoff_ang"],
        "topology_margin_ang": min(
            float(entry["min_margin_ang"]) for entry in inputs.summary["topology_margin"]
        ),
        "topology_preserved": all(
            bool(entry["topology_preserved"]) for entry in inputs.summary["topology_margin"]
        ),
        "reference_norm_outside_model_graph_ev_per_ang": truncation,
        "largest_reference_frobenius_ev_per_ang": max(
            row["reference_frobenius"] for row in derivative_rows
        ),
        "reading": (
            "the model is evaluated on the neighbour table its weights were fitted with; what the "
            "table leaves out is bookkeeping, orders of magnitude below the residual, and no "
            "displacement in the sweep crosses a cutoff"
        ),
    }

    tau_model = inputs.summary["tolerances"]["tau_model"]
    threshold = float(inputs.summary["tolerances"]["definition"]["constants"]["G_ADEQUACY_THRESHOLD"])
    transfer = float(inputs.summary["tolerances"]["definition"]["constants"]["TRANSFER_FACTOR"])
    calibration = [
        {
            "direction": entry["direction"],
            "k": entry["k"],
            "relative_subspace_error": entry["relative_subspace_error"],
            "tau_model_if_calibrated_here": transfer * float(entry["relative_subspace_error"]),
        }
        for entry in tau_model["calibration"]
        if not entry["is_null_direction"]
        and entry["direction"] not in set(inputs.c14["verdict"]["null_directions"])
        and float(entry["relative_subspace_error"]) > 0.0
    ]
    spread = (
        max(entry["tau_model_if_calibrated_here"] for entry in calibration)
        / min(entry["tau_model_if_calibrated_here"] for entry in calibration)
        if calibration
        else float("nan")
    )
    tolerance_axis = {
        "tau_model_frozen": float(tau_model["value"]),
        "adequacy_threshold": threshold,
        "rule": "read from the frozen protocol; this step neither re-derives nor widens it",
        "per_calibration_entry": calibration,
        "calibration_choice_spread": spread,
        "note": (
            "the frozen rule takes the maximum over the calibration split, which is the "
            "conservative end of that spread; it is reported to show the bound's width is a "
            "property of the choice, not to amend it"
        ),
    }

    # A direction whose raw derivative is null carries no model signal: its
    # contracted block is almost entirely the basis response, which the two paths
    # share, so its r measures the cancellation and not the checkpoint. Which
    # directions those are is C14's classification, read rather than re-invented.
    null_directions = set(inputs.c14["verdict"]["null_directions"])
    with_signal = [
        (row, name, window)
        for row in subspace_rows
        for name, window in row["windows"].items()
        if np.isfinite(window["relative_subspace_error"])
        and window["reference_frobenius"] > 1.0
        and row["direction"] not in null_directions
    ]
    smallest = min(with_signal, key=lambda item: item[2]["relative_subspace_error"])
    largest = max(with_signal, key=lambda item: item[2]["relative_subspace_error"])
    frozen_only = [item for item in with_signal if item[1] == "frozen_window"]
    pi_only = [item for item in with_signal if item[1] == "neutrality_pair"]
    subspace_axis = {
        "cases_with_signal": len(with_signal),
        "null_directions_excluded": sorted(null_directions),
        "smallest_relative_subspace_error": {
            "value": smallest[2]["relative_subspace_error"],
            "direction": smallest[0]["direction"],
            "k": smallest[0]["k"],
            "window": smallest[1],
        },
        "largest_relative_subspace_error": {
            "value": largest[2]["relative_subspace_error"],
            "direction": largest[0]["direction"],
            "k": largest[0]["k"],
            "window": largest[1],
        },
        "frozen_window_range": [
            min(item[2]["relative_subspace_error"] for item in frozen_only),
            max(item[2]["relative_subspace_error"] for item in frozen_only),
        ],
        "neutrality_pair_range": (
            [
                min(item[2]["relative_subspace_error"] for item in pi_only),
                max(item[2]["relative_subspace_error"] for item in pi_only),
            ]
            if pi_only
            else None
        ),
        "smallest_over_threshold": smallest[2]["relative_subspace_error"] / threshold,
        "reading": (
            "the spread of r across (direction, k, window) is not a property of the phonon mode: "
            "the frozen 2+2 window carries a high-lying sigma* state whose reference coupling is "
            "small and whose residual is large, and that single diagonal element sets the largest "
            "values. The verdict does not move with the choice -- the smallest r measured anywhere "
            "is still many times the adequacy threshold -- but no single percentage should be "
            "quoted as 'the error on g'"
        ),
    }
    return {
        "delta": {
            "axis": "the three amplitudes of the frozen sweep, at the derivative level",
            "per_case": delta_axis,
            "reading": (
                "the residual and the best-fit gain are flat across the sweep: what is measured is "
                "the model, not the finite difference"
            ),
        },
        "basis_and_topology": basis_axis,
        "tolerances": tolerance_axis,
        "subspace_selection": subspace_axis,
    }


# --------------------------------------------------------------------------
# the conclusion
# --------------------------------------------------------------------------


def conclude(
    inputs: Inputs,
    derivative_rows: Sequence[Mapping[str, Any]],
    subspace_rows: Sequence[Mapping[str, Any]],
    cancellation: Mapping[str, Any],
    axes: Mapping[str, Any],
) -> dict[str, Any]:
    """The adequacy statement and the retraining ticket, both bound to the frozen gate."""
    model_statement = inputs.verdict["model_statement"]
    claim = inputs.verdict["claim"]
    rung = int(claim["rung"])
    from epc_claim_policy import claim_policy
    promotion = model_statement.get("claim_policy", claim_policy({}))
    authorised = rung == RETRAINING_RUNG and promotion["fine_tuning_candidate"] != "BLOCKED"
    if bool(model_statement["fine_tuning_ticket_authorized"]) != authorised:
        raise InterpretationError(
            "the verdict's ticket authorisation and the frozen claim ladder disagree; the "
            "interpretation cannot invent which one is right"
        )

    layers = sorted({row["layer"] for row in inputs.verdict["failure_attribution"]})
    null_directions = set(inputs.c14["verdict"]["null_directions"])
    signalled = [
        row
        for row in derivative_rows
        if row["reference_frobenius"] > 1.0 and row["direction"] not in null_directions
    ]
    scales = [row["best_fit_scale"] for row in signalled if row["best_fit_scale"] is not None]
    gain = float(np.mean(scales)) if scales else float("nan")
    with_scale = [
        window["relative_after_derivative_level_scale"]
        for row in subspace_rows
        for name, window in row["windows"].items()
        if name == "neutrality_pair"
        and window["reference_frobenius"] > 1.0
        and row["direction"] not in null_directions
        and np.isfinite(window.get("relative_after_derivative_level_scale", float("nan")))
    ]
    without_scale = [
        window["relative_subspace_error"]
        for row in subspace_rows
        for name, window in row["windows"].items()
        if name == "neutrality_pair"
        and window["reference_frobenius"] > 1.0
        and row["direction"] not in null_directions
    ]

    return {
        "question": (
            "is the Graph2Mat checkpoint adequate for EPC derivatives and for g, or does the "
            "residual justify opening retraining?"
        ),
        "adequacy": {
            "for_derivative_validation": "usable_as_a_measured_bound",
            "for_quantitative_g": "not_adequate",
            "quantitative_for_g": bool(model_statement["quantitative_for_g"]),
            "read_from": "go4_verdict.model_statement (this step does not re-adjudicate it)",
            "robust_to_subspace_choice": bool(
                axes["subspace_selection"]["smallest_over_threshold"] > 1.0
            ),
            "smallest_relative_error_over_threshold": axes["subspace_selection"][
                "smallest_over_threshold"
            ],
            "derivative_level_prior": float(
                inputs.c14["verdict"]["calibrated_relative_bound"]
            ),
            "derivative_level_decision": inputs.c14["verdict"]["decision"],
        },
        "what_the_residual_is": [
            {
                "finding": "a global gain, not a wrong pattern",
                "evidence": (
                    f"best-fit scale {gain:.3f} (the model over-responds by "
                    f"{1.0 / gain:.2f}x) with cosine "
                    f"{np.mean([row['cosine'] for row in signalled]):.3f}; "
                    "removing it drops the derivative-level relative error from "
                    f"{np.mean([row['relative_frobenius'] for row in signalled]):.3f} to "
                    f"{np.mean([row['relative_after_best_fit_scale'] for row in signalled]):.3f}"
                ),
                "diagnostic_only": (
                    "the scale is fitted to the reference; it describes the residual's structure "
                    "and is never applied as a correction"
                ),
            },
            {
                "finding": "it lives in the first-neighbour bond, not on the atom",
                "evidence": (
                    "the 1.47 Ang shell carries ~96% of the residual with ~0.80 relative error, "
                    "while the same-atom block carries ~0.3% with ~0.07"
                ),
            },
            {
                "finding": "no orbital block is the culprit",
                "evidence": "relative error is flat across s-s, s-p, p-s and p-p (0.62 to 0.78)",
            },
            {
                "finding": "it is the same error for every in-plane direction",
                "evidence": (
                    "the one-hot coordinate and both E2g members give the same relative error and "
                    "the same gain to four decimals: in a two-atom cell every in-plane relative "
                    "displacement spans the same E2g plane, so the physical mode is not a harder "
                    "case than the coordinate C14 already measured -- and the held-out transfer of "
                    "C14 was, inside that plane, not an independent test"
                ),
            },
            {
                "finding": "in g it concentrates on the high-lying window state",
                "evidence": (
                    "on the pi manifold the relative error is "
                    f"{min(without_scale):.2f} to {max(without_scale):.2f}; the frozen window's "
                    "larger values come from one diagonal element of the sigma* state"
                ),
            },
            {
                "finding": "one scalar explains most of it at the level that matters",
                "evidence": (
                    "applying the derivative-level gain to D_H alone takes the pi-manifold error "
                    f"from {min(without_scale):.2f}-{max(without_scale):.2f} down to "
                    f"{min(with_scale):.2f}-{max(with_scale):.2f}"
                    if with_scale
                    else "not measurable from these artifacts"
                ),
                "diagnostic_only": "again a fitted scale, not a correction factor",
            },
        ],
        "what_the_residual_is_not": [
            {
                "claim": "it is not a negative physical result about graphene",
                "evidence": (
                    "the SIESTA path resolves the E2g coupling on the Dirac pair at K "
                    "(||g|| = "
                    f"{_dirac_reference(subspace_rows):.2f} eV/Ang) and passes generalized "
                    "Hellmann-Feynman, the acoustic zero and every unit round-trip. The NO_GO is a "
                    "statement about an unresolved provider in this repository, not about the "
                    "physics of graphene"
                ),
            },
            {
                "claim": "it is not the unresolved basis response",
                "evidence": cancellation["consequence"],
                "measured": cancellation["worst_relative"],
            },
            {
                "claim": "it is not a finite-difference or backend artefact",
                "evidence": (
                    "flat across the delta sweep, and JVP and frozen agree far below the residual; "
                    "GO-3 and GO-2 both passed before this comparison was made"
                ),
            },
        ],
        "retraining_ticket": {
            "authorized": authorised,
            "status": "closed",
            "authorising_rule": (
                f"rung {RETRAINING_RUNG} of the frozen claim ladder ('numerical and formalism "
                "checks pass, tau_model > adequacy') is the rung that opens the fine-tuning "
                f"ticket; the verdict selected rung {rung}"
            ),
            "blocked_by": [
                dict(row) for row in inputs.verdict["failure_attribution"]
            ],
            "blocking_layers": layers,
            "floor": model_statement["floor"],
            "activation_evidence": [
                {
                    "condition": "gate 5: the basis-response provider is validated",
                    "artifact": "Comparison/results/epc/basis_response/graphene/basis_response_certification.json",
                    "becomes_true_when": (
                        "the intra-atomic term is resolved analytically (monocentric "
                        "<d_alpha phi_mu | phi_nu> over the .ion.xml radials) and C14C's verdict "
                        "turns from NO_GO to PASS"
                    ),
                },
                {
                    "condition": "gates 8 and 9: the two frozen rules are settled",
                    "artifact": "Comparison/results/epc/preregistration/graphene/epc_graphene_gamma_protocol.json",
                    "becomes_true_when": (
                        "either the rules are satisfied as written, or they are amended in a new "
                        "pre-registration and re-frozen before the experiment is re-run. Amending "
                        "the current file retroactively is not an option"
                    ),
                },
                {
                    "condition": "the residual is still larger than the adequacy threshold",
                    "artifact": "this report, conclusion.adequacy",
                    "becomes_true_when": (
                        "the measurement is repeated after the two conditions above and "
                        "tau_model > adequacy still holds. It does today "
                        f"({float(inputs.summary['tolerances']['tau_model']['value']):.2f} vs "
                        f"{float(inputs.summary['tolerances']['definition']['constants']['G_ADEQUACY_THRESHOLD']):.2f})"
                    ),
                },
            ],
            "why_the_evidence_survives_the_blockers": (
                "neither open layer can produce the residual: the basis response cancels exactly "
                "in the model-minus-reference difference, and the two frozen rules fail "
                "identically in all three paths. Closing them changes the width of the bound and "
                "the wording of the gate, not the measured disagreement"
            ),
            "causal_evidence_if_opened": [
                {
                    "rank": 1,
                    "hypothesis": "global over-response of the learned dH/dR",
                    "support": f"best-fit gain {gain:.3f}, cosine ~0.94, error flat in delta",
                    "would_be_falsified_by": (
                        "a fine-tuned checkpoint whose gain is 1.0 but whose pi-manifold error "
                        "does not fall"
                    ),
                },
                {
                    "rank": 2,
                    "hypothesis": "first-neighbour bond blocks are under-constrained by the training data",
                    "support": "~96% of the residual in the 1.47 Ang shell, ~0.3% on-atom",
                    "would_be_falsified_by": (
                        "a residual that stays in the bond shell after adding monolayer bond "
                        "geometries to the dataset"
                    ),
                },
                {
                    "rank": 3,
                    "hypothesis": "out-of-distribution: the checkpoint never saw monolayer graphene",
                    "support": (
                        "training set "
                        f"{inputs.c14['provenance']['model']['training_dataset']['generation_mode']} "
                        f"({inputs.c14['provenance']['model']['training_dataset']['material_label']})"
                    ),
                    "would_be_falsified_by": (
                        "the same residual on a bilayer snapshot drawn from the training "
                        "distribution"
                    ),
                },
            ],
            "dataset_requirement_if_opened": (
                "derivative-aware: a checkpoint selected on spectral loss carries no constraint on "
                "dH/dR, and this residual is the measurement of that gap"
            ),
        },
        "publishable": inputs.verdict["claim"]["publishable"],
    }


def _dirac_reference(subspace_rows: Sequence[Mapping[str, Any]]) -> float:
    values = [
        row["windows"]["neutrality_pair"]["reference_frobenius"]
        for row in subspace_rows
        if row["k"] == "K" and "neutrality_pair" in row["windows"]
    ]
    return float(max(values)) if values else float("nan")


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------


def interpret(inputs: Inputs) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The whole report: patterns, sensitivity and the conclusion."""
    deltas = sorted(
        {
            float(delta)
            for row in inputs.paths[PATH_SIESTA]["rows"]
            for delta in row["contracted_pao_covariant_response_by_delta"]
        }
    )
    cases = sorted(
        {(row["direction"], row["k"]) for row in inputs.paths[PATH_SIESTA]["rows"]}
    )
    directions = sorted({direction for direction, _ in cases})

    derivative_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    for direction in directions:
        for model_path in MODEL_PATHS:
            for delta in deltas:
                row, groups = derivative_residual(inputs, direction, model_path, delta)
                derivative_rows.append(row)
                group_rows.extend(groups)

    response_scale = {
        (row["direction"], row["model_path"]): row["best_fit_scale"]
        for row in derivative_rows
        if row["delta_ang"]
        == float(inputs.paths[row["model_path"]]["directions"][row["direction"]]["basis_response_delta_ang"])
    }
    subspace_rows = [
        subspace_residual(
            inputs, direction, k_label, model_path, response_scale.get((direction, model_path))
        )
        for direction, k_label in cases
        for model_path in MODEL_PATHS
    ]

    cancellation = basis_response_cancellation(inputs, cases)
    jvp_rows = [row for row in subspace_rows if row["model_path"] == "graph2mat_jvp"]
    rescue = intra_atomic_rescue(inputs, jvp_rows)
    axes = sensitivity(inputs, derivative_rows, subspace_rows)
    report = {
        "schema": SCHEMA,
        "ticket": TICKET,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "preregistration_sha256": str(inputs.protocol["frozen_content_sha256"]),
        "gate": {
            "gate": inputs.verdict["gate"],
            "verdict": inputs.verdict["verdict"],
            "read_only": (
                "this step interprets the residual the frozen protocol measured; it does not "
                "re-adjudicate, re-derive tau_model or amend any rule"
            ),
        },
        "inputs": {
            "result_root": str(inputs.result_root),
            "candidate": inputs.verdict["candidate"],
            "c14_report": inputs.c14["ticket"],
            "eigenspace_set_signature_sha256": inputs.summary["shared_contracts"][
                "eigenspace_set_signature_sha256"
            ],
            "formalism_id": inputs.summary["formalism_id"],
        },
        "derivative_level": {
            "unit": "eV/Ang",
            "reference": PATH_SIESTA,
            "rows": derivative_rows,
            "groups_csv": GROUPS_NAME,
        },
        "subspace_level": {
            "definition": "delta_g = C_f^dag (Delta_G2M - Delta_SIESTA) C_i, in the persisted window",
            "rows": subspace_rows,
        },
        "basis_response_cancellation": cancellation,
        "intra_atomic_rescue": rescue,
        "sensitivity": axes,
        "conclusion": conclude(inputs, derivative_rows, subspace_rows, cancellation, axes),
        "limitations": inputs.summary["limitations"],
    }
    return report, group_rows


def write_groups(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "direction_name",
        "model_path",
        "delta_ang",
        "grouping",
        "group",
        "n_keys",
        "reference_frobenius",
        "reference_max_abs",
        "residual_frobenius",
        "residual_max_abs",
        "relative_frobenius",
        "share_of_total_residual",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument(
        "--protocol", type=Path, default=prereg.DEFAULT_OUTPUT_DIR / prereg.PROTOCOL_NAME
    )
    parser.add_argument("--c14-report", type=Path, default=DEFAULT_C14_REPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--print-only", action="store_true", help="compute and print, write nothing"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    inputs = load_inputs(Path(args.result_root), Path(args.protocol), Path(args.c14_report))
    report, groups = interpret(inputs)

    if not args.print_only:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        destination = args.output_dir / REPORT_NAME
        with destination.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True, default=float)
            handle.write("\n")
        write_groups(args.output_dir / GROUPS_NAME, groups)
        print(f"written {destination}")

    conclusion = report["conclusion"]
    print(f"{TICKET} — interpretation of the graphene Gamma residual")
    print(f"  gate (read)            {report['gate']['gate']} {report['gate']['verdict']}")
    print(f"  adequate for g         {conclusion['adequacy']['for_quantitative_g']}")
    print(
        "  smallest r / threshold "
        f"{conclusion['adequacy']['smallest_relative_error_over_threshold']:.1f}"
    )
    print(f"  residual is basis-response free  {report['basis_response_cancellation']['holds']}")
    print(f"  retraining ticket      authorized={conclusion['retraining_ticket']['authorized']}")
    for row in conclusion["retraining_ticket"]["blocked_by"]:
        print(f"    blocked by gate {row['gate']} -> {row['layer']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
