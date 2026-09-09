#!/usr/bin/env python3
"""E-F_001-S35: validate the AB bilayer candidate (S34) and decide GO-7.

This script computes no physics: every number it judges was measured by
``run_ab_bilayer_epc_paths.py`` and written into ``ab_bilayer_epc_three_sectors.json``.
Re-deriving any of it here would be measuring the result a second time with the
freedom to measure it differently -- the same rule ``evaluate_epc_metrics.py``
(GO-4) follows. What is new is five checks the roadmap's Fase 8 names
explicitly and S34 did not adjudicate on its own:

``layer_swap``
    the interlayer ``(bottom, top)`` and ``(top, bottom)`` blocks of ``D_H``
    must carry the same Frobenius norm (Hermiticity), so no result secretly
    depends on which physical layer ``layer_of_atom_from_positions`` happened
    to label 0.
``shear_direction``
    the shear pattern is purely in-plane, bottom and top move exactly opposite
    and equal in magnitude, and the pattern carries no net rigid translation.
``breathing_parity``
    the layer-breathing pattern is purely out-of-plane and equal-and-opposite,
    and the two intralayer (AA, BB) blocks it induces agree within a loose
    tolerance -- AB stacking has no exact atom-index symmetry under a uniform
    per-layer pattern, so exact equality is not expected, but a large
    asymmetry would be.
``interlayer_derivative_blocks``
    shear and layer_breathing are built to be interlayer-sensitive: their
    interlayer block must be comparable to or larger than their own intralayer
    blocks. ``intralayer_control`` is built to be a null control: its
    interlayer block must be small relative to its own (large) intralayer
    block.
``parameter_sensitivity``
    each sector's phonon match must be a clear winner over the runner-up
    branch; a match that is barely ahead of its neighbour is one dHSdR/basis
    perturbation away from flipping to a different branch, and no g measured
    against an unstable match can be trusted.

Usage::

    .venv/bin/python Comparison/scripts/evaluate_ab_bilayer_epc.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for _path in (SCRIPT_DIR, REPO_ROOT / "shared"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import run_ab_bilayer_epc_paths as ab  # noqa: E402
import certify_basis_response as cbr  # noqa: E402
import certify_siesta_dhsdr as go2  # noqa: E402
import epc_basis_response as ebr  # noqa: E402
import orbital_contract as oc  # noqa: E402
from read_siesta_dhsdr import read_dhsdr  # noqa: E402

SCHEMA = "epc_go7_adjudication_v1"
TICKET = "E-F_001-S35"
GATE = "GO-7"
VERDICT_NAME = "go7_verdict.json"

DEFAULT_REPORT_PATH = ab.DEFAULT_RESULT_ROOT / "ab_bilayer_epc_three_sectors.json"

INTERLAYER_SENSITIVE_SECTORS = ("shear", "layer_breathing")
CONTROL_SECTOR = "intralayer_control"
D_H_SOURCE = "D_H_gamma_siesta"

#: Every tolerance judged below, named so a failure can be read off this table
#: rather than re-derived from the numbers in each check function.
LAYER_SWAP_REL_TOL = 1e-6
NET_TRANSLATION_ABS_TOL = 1e-9
BREATHING_PARITY_REL_TOL = 0.20
CONTROL_NULL_MAX_RATIO = 0.05
INTERLAYER_DOMINANT_MIN_RATIO = 1.0
MODE_MATCH_MIN_SEPARATION = 1.2

#: The layer a failure is charged to -- infrastructure/formalism, never "the
#: model", since none of these five checks touch Graph2Mat vs SIESTA agreement.
LAYERS = ("geometry_construction", "reference", "phonons")


class Go7AdjudicationError(RuntimeError):
    """The candidate cannot be judged: evidence this adjudication needs is missing."""


def _check(name: str, passed: bool, detail: str, layer: str, **extra: Any) -> dict[str, Any]:
    if layer not in LAYERS:
        raise Go7AdjudicationError(f"check {name!r} names an unknown layer {layer!r}")
    return {"check": name, "passed": bool(passed), "detail": detail, "layer": layer, **extra}


def _block(sector: Mapping[str, Any], row: int, col: int) -> dict[str, Any]:
    for entry in sector["layer_blocks"][D_H_SOURCE]:
        if entry["row_cluster"] == row and entry["column_cluster"] == col:
            return entry
    raise Go7AdjudicationError(f"no ({row}, {col}) D_H block recorded in {D_H_SOURCE!r}")


def _pattern(sector: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    pattern = sector["layer_pattern"]
    return np.asarray(pattern["bottom"], dtype=np.float64), np.asarray(pattern["top"], dtype=np.float64)


def derive_spatial_layer_swap(report: Mapping[str, Any]) -> dict[str, Any]:
    """Derive and apply the real AB inversion; no Hermitian-block shortcut."""
    import sisl

    root = Path(report["reference_root"])
    manifest = json.loads((root / "epc_siesta_reference_manifest.json").read_text(encoding="utf-8"))
    campaign = go2.Campaign(root=root, manifest=manifest)
    run_id = manifest["certification_set"]["equilibrium_run_id"]
    equilibrium = go2.read_tshs(campaign.run_dir(run_id), run_id, fermi_ev=campaign.fermi_ev(run_id))
    geometry = sisl.get_sile(str(equilibrium.path)).read_geometry()
    positions, cell = np.asarray(geometry.xyz), np.asarray(geometry.cell)
    inverse_cell = np.linalg.inv(cell)
    center = permutation = translations = None
    for i in range(len(positions)):
        for j in range(len(positions)):
            candidate = 0.5 * (positions[i] + positions[j])
            trial, shifts = [], []
            for position in positions:
                fractional = (2.0 * candidate - position - positions) @ inverse_cell
                residual = fractional - np.round(fractional)
                matches = np.flatnonzero(np.linalg.norm(residual, axis=1) < 1e-3)
                if len(matches) != 1:
                    break
                trial.append(int(matches[0]))
                shifts.append(np.round(fractional[matches[0]]).astype(int).tolist())
            if len(trial) == len(positions) and len(set(trial)) == len(positions):
                center, permutation, translations = candidate, trial, shifts
                break
        if permutation is not None:
            break
    if permutation is None:
        return {"residuals": [{"sector": "geometry", "passes": False}], "reason": "no inversion"}

    orb = oc.parse_orb_indx(campaign.run_dir(run_id) / f"{run_id}.ORB_INDX")
    rows = sorted(orb["unit_cell_rows"], key=lambda row: row["io"])
    atom_of = np.asarray([row["ia"] - 1 for row in rows])
    first = {atom: int(np.flatnonzero(atom_of == atom)[0]) for atom in set(atom_of.tolist())}
    orbital_permutation, parity = [], []
    for orbital, row in enumerate(rows):
        atom = int(atom_of[orbital])
        orbital_permutation.append(first[permutation[atom]] + orbital - first[atom])
        parity.append((-1.0) ** int(row["l"]))

    dhsdr = read_dhsdr(campaign.dhsdr_path(float(campaign.deltas[0])))
    isc_off = next(iter(dhsdr.records.values()))["D_H"].isc_off
    layers = ab.layer_of_atom_from_positions(positions)
    directions = ab.build_directions(positions, cell, layers)

    def residual(name: str, kind: str) -> float:
        field = go2.contract_fc(dhsdr, directions[name].vectors, kind)
        blocks = cbr.dense_blocks(field, isc_off, equilibrium.no_u)[0]
        matrix = ebr.bloch_sum(blocks, isc_off, (0.0, 0.0, 0.0))
        if kind == "D_H":
            overlap = ebr.bloch_sum(
                cbr.dense_blocks(go2.contract_fc(dhsdr, directions[name].vectors, "D_S"), isc_off, equilibrium.no_u)[0],
                isc_off, (0.0, 0.0, 0.0),
            )
            matrix = matrix - float(equilibrium.fermi_ev) * overlap
        transformed = np.zeros_like(matrix)
        for a in range(equilibrium.no_u):
            for b in range(equilibrium.no_u):
                transformed[orbital_permutation[a], orbital_permutation[b]] = parity[a] * parity[b] * matrix[a, b]
        return float(np.linalg.norm(transformed - matrix) / max(np.linalg.norm(matrix), 1e-30))

    control = residual("intralayer_control", "D_H")
    evidence = []
    for name in ("shear", "layer_breathing"):
        ds, dh = residual(name, "D_S"), residual(name, "D_H")
        evidence.append({"sector": name, "D_S_relative_residual": ds, "D_H_relative_residual": dh,
                         "passes": ds < 1e-6 and dh < 0.5 and dh < 0.5 * control})
    return {
        "cell": cell.tolist(), "positions": positions.tolist(), "images": np.asarray(isc_off).tolist(),
        "orbitals": {"permutation": orbital_permutation, "parity": parity},
        "inversion_center_ang": center.tolist(), "atom_permutation": permutation,
        "atom_lattice_translations": translations, "k_fractional": [0.0, 0.0, 0.0],
        "q_fractional": [0.0, 0.0, 0.0], "negative_control_D_H_relative_residual": control,
        "residuals": evidence,
    }


# --------------------------------------------------------------------------
# the five checks
# --------------------------------------------------------------------------


def check_spatial_layer_swap(report: Mapping[str, Any]) -> dict[str, Any]:
    evidence = report.get("spatial_layer_swap") or {}
    required = ("cell", "positions", "images", "orbitals", "k_fractional", "q_fractional", "residuals")
    complete = all(key in evidence for key in required)
    rows = evidence.get("residuals") or []
    return _check(
        "spatial_layer_swap_from_geometry",
        complete and bool(rows) and all(bool(row.get("passes")) for row in rows),
        "mandatory independent AB spatial transform derived from cell, positions, image mapping, "
        "orbital action and declared k/q; conjugate-block norms alone are not evidence",
        "geometry_construction",
        evidence_complete=complete,
        rows=rows,
    )


def check_shear_direction(sectors: Mapping[str, Any]) -> dict[str, Any]:
    bottom, top = _pattern(sectors["shear"])
    in_plane_only = bool(np.allclose(bottom[:, 2], 0.0) and np.allclose(top[:, 2], 0.0))
    antiparallel = bool(np.allclose(bottom, -top))
    net = bottom.sum(axis=0) + top.sum(axis=0)
    passed = in_plane_only and antiparallel and float(np.linalg.norm(net)) <= NET_TRANSLATION_ABS_TOL
    return _check(
        "shear_direction_in_plane_antiparallel",
        passed,
        "the shear pattern moves the two layers in-plane, exactly opposite and equal in "
        "magnitude, with no net rigid translation of the bilayer",
        "geometry_construction",
        in_plane_only=in_plane_only,
        bottom_equals_minus_top=antiparallel,
        net_translation=net.tolist(),
    )


def check_breathing_parity(sectors: Mapping[str, Any]) -> dict[str, Any]:
    sector = sectors["layer_breathing"]
    bottom, top = _pattern(sector)
    out_of_plane_only = bool(np.allclose(bottom[:, :2], 0.0) and np.allclose(top[:, :2], 0.0))
    antiparallel = bool(np.allclose(bottom, -top))
    net = bottom.sum(axis=0) + top.sum(axis=0)
    aa = _block(sector, 0, 0)["frobenius_norm"]
    bb = _block(sector, 1, 1)["frobenius_norm"]
    relative = abs(aa - bb) / max(aa, bb, 1e-30)
    passed = (
        out_of_plane_only
        and antiparallel
        and float(np.linalg.norm(net)) <= NET_TRANSLATION_ABS_TOL
        and relative <= BREATHING_PARITY_REL_TOL
    )
    return _check(
        "breathing_parity_layers_equal_and_opposite",
        passed,
        "the layer-breathing pattern is purely out-of-plane and equal-and-opposite, and the AA "
        "vs BB intralayer blocks it induces agree within a loose tolerance (AB stacking has no "
        "exact atom-index symmetry under a uniform per-layer pattern)",
        "reference",
        out_of_plane_only=out_of_plane_only,
        bottom_equals_minus_top=antiparallel,
        net_translation=net.tolist(),
        intralayer_AA_frobenius=aa,
        intralayer_BB_frobenius=bb,
        relative_difference=relative,
        tolerance=BREATHING_PARITY_REL_TOL,
    )


def check_interlayer_derivative_blocks(sectors: Mapping[str, Any]) -> dict[str, Any]:
    rows = []
    for name, sector in sectors.items():
        aa = _block(sector, 0, 0)["frobenius_norm"]
        bb = _block(sector, 1, 1)["frobenius_norm"]
        ab_block = _block(sector, 0, 1)["frobenius_norm"]
        intralayer_avg = 0.5 * (aa + bb)
        ratio = ab_block / max(intralayer_avg, 1e-30)
        sensitive = name in INTERLAYER_SENSITIVE_SECTORS
        passes = ratio >= INTERLAYER_DOMINANT_MIN_RATIO if sensitive else ratio <= CONTROL_NULL_MAX_RATIO
        rows.append(
            {
                "sector": name,
                "role": "interlayer_sensitive" if sensitive else "null_control",
                "interlayer_frobenius": ab_block,
                "intralayer_average_frobenius": intralayer_avg,
                "interlayer_over_intralayer_ratio": ratio,
                "rule": (
                    f"ratio >= {INTERLAYER_DOMINANT_MIN_RATIO}"
                    if sensitive
                    else f"ratio <= {CONTROL_NULL_MAX_RATIO}"
                ),
                "passes": passes,
            }
        )
    return _check(
        "interlayer_derivative_blocks_match_sector_role",
        all(row["passes"] for row in rows),
        "shear and layer_breathing show an interlayer block at least as large as their own "
        "intralayer response; intralayer_control's interlayer block stays a small fraction of "
        "its own (large) intralayer response, the built-in null control S34 documents",
        "reference",
        rows=rows,
    )


def check_parameter_sensitivity(sectors: Mapping[str, Any]) -> dict[str, Any]:
    rows = []
    for name, sector in sectors.items():
        overlaps = sorted(sector["phonon_match"]["overlaps_by_branch"], reverse=True)
        top_overlap = overlaps[0]
        runner_up = overlaps[1] if len(overlaps) > 1 else 0.0
        separation = top_overlap / max(runner_up, 1e-30)
        rows.append(
            {
                "sector": name,
                "top_overlap": top_overlap,
                "runner_up_overlap": runner_up,
                "separation_ratio": separation,
                "passes": separation >= MODE_MATCH_MIN_SEPARATION,
            }
        )
    return _check(
        "phonon_mode_match_robust_to_branch_ambiguity",
        all(row["passes"] for row in rows),
        f"each sector's matched Gamma branch leads its runner-up by at least "
        f"{MODE_MATCH_MIN_SEPARATION}x in projection overlap; a smaller margin means the match "
        "is one small perturbation of the FC/basis away from flipping branches",
        "phonons",
        rows=rows,
    )


CHECKS = (
    check_shear_direction,
    check_breathing_parity,
    check_interlayer_derivative_blocks,
    check_parameter_sensitivity,
)


# --------------------------------------------------------------------------
# adjudication
# --------------------------------------------------------------------------


def adjudicate(report: Mapping[str, Any]) -> dict[str, Any]:
    report = dict(report)
    # Every caller must derive the spatial witness from the reference geometry.
    try:
        report["spatial_layer_swap"] = derive_spatial_layer_swap(report)
    except (KeyError, OSError, ValueError) as exc:
        report["spatial_layer_swap"] = {"reason": str(exc)}
    sectors = report.get("sectors")
    if not sectors:
        raise Go7AdjudicationError("the candidate records no sectors; GO-7 has no evidence")
    upstream = report.get("upstream_gates") or {}
    checks = [
        _check(
            "upstream_go4_go5_pass",
            upstream == {"GO-4": "PASS", "GO-5": "PASS"},
            "S35 fails closed unless the candidate records GO-4=PASS and GO-5=PASS",
            "reference",
            upstream_gates=upstream,
        ),
        check_spatial_layer_swap(report),
        *[fn(sectors) for fn in CHECKS],
    ]
    failed = [row["check"] for row in checks if not row["passed"]]
    verdict = "PASS" if not failed else "NO_GO"
    return {
        "schema": SCHEMA,
        "ticket": TICKET,
        "gate": GATE,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "candidate": report.get("ticket"),
        "candidate_status": report.get("status"),
        "verdict": verdict,
        "checks": checks,
        "checks_failed": failed,
        "failure_attribution": [
            {"check": row["check"], "layer": row["layer"]} for row in checks if not row["passed"]
        ],
        # A GO-7 PASS validates the five structural/sensitivity checks Fase 8 names. It does not
        # promote result_class or status: S34's own limitations (no explicit SIESTA +- FD
        # cross-check for this material, Gamma-only FC, no moving-E_F correction, C14C's
        # unresolved intra-atomic term) still make every g here a bound, not a measurement.
        "candidate_limitations_carried_forward": report.get("limitations"),
        "label": None,
        "label_rule": "GO-7 judges layer/shear/breathing/interlayer/sensitivity structure only; "
        "it never promotes a sector's own result_class or status",
    }


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--result-root", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    verdict = adjudicate(report)

    result_root = args.result_root or Path(args.report).parent
    destination = result_root / VERDICT_NAME
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"{GATE} adjudication of {verdict['candidate']}: {verdict['verdict']}")
    for row in verdict["checks"]:
        print(f"  {row['check']:<45} {'PASS' if row['passed'] else 'FAIL'}  ({row['layer']})")
    print(f"  written {destination}")
    return 0 if verdict["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
