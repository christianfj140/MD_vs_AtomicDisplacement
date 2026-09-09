#!/usr/bin/env python3
"""Gate 1: delta_plateau_topology_certification (C1:x) + graphene_Gamma_E2g_delta_convergence.

Delta(h) = D5 K(h) - S_L(h) S0^-1 K0 - K0 S0^-1 S_R(h), reconstructed end-to-end
for each of the five pre-registered h in
``delta_translation_campaign_preregistration_v1.json``. S_L(h)/S_R(h) are NOT
reimplemented here: they come from the already-certified
``production_basis_connection`` route (``certify_production_basis_connection.
_connections``), which computes them from real-space PAO grid quadrature and is
valid for any h, including 0.000625 -- if that route cannot produce a value for
some h, this script fails closed rather than approximate or interpolate. K(h)
is built per geometry in the certified PAO gauge (K = H_abs - c_vac(R) S) from
real SIESTA TSHS/VT artifacts of the twelve new + four historical PROD-SZ runs,
then combined with the same D5 stencil. The TSHS support topology {(i,j,R)} is
independently re-verified here from the real files, not assumed from any prior
manual check.

``graphene_Gamma_E2g_delta_convergence`` is reported separately and is NOT
promoted from a C1:x PASS: no certified fixed (C_i, C_f) electronic coefficient
artifact exists yet for the E2g contraction, so it is honestly BLOCKED rather
than approximated.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO_ROOT / "Comparison/scripts"), str(REPO_ROOT / "shared")]

import sisl  # noqa: E402

from artifact_signature import file_sha256, input_signature_sha256  # noqa: E402
from certify_epc_energy_zero import vacuum_level  # noqa: E402
from certify_full_basis_connection_semantic_v5 import solve  # noqa: E402
from certify_production_basis_connection import _connections  # noqa: E402
from cross_basis_projection_preflight import K_POINTS, _basis_geometry  # noqa: E402
from nested_basis_identity_preflight import PROD_CENTRAL  # noqa: E402
from run_nested_basis_ladder import _rms  # noqa: E402
from run_nested_basis_sentinel import _matrix_data  # noqa: E402

OUTPUT = REPO_ROOT / "Comparison/results/epc/delta_plateau_topology_certification"
PROD_SZ_ROOT = PROD_CENTRAL.parent
PREREG_PATH = (
    REPO_ROOT / "Comparison/results/epc/delta_translation_campaign_preregistration"
    / "delta_translation_campaign_preregistration_v1.json"
)

H_LADDER = (0.000625, 0.00125, 0.0025, 0.005, 0.01)
DECISIVE_WINDOW = (0.005, 0.0025, 0.00125)
RMS_LIMIT_EV_PER_ANG = 5e-3
REL_LIMIT = 0.01
MAX_LIMIT_EV_PER_ANG = 15e-3
D5_WEIGHTS = {-2: 1.0, -1: -8.0, 1: 8.0, 2: -1.0}
D2_WEIGHTS = {-1: -1.0, 1: 1.0}

_HISTORICAL_C1X = {-0.01: "m2", -0.005: "m1", 0.005: "p1", 0.01: "p2"}
_ALL_TRANSLATION_TAGS = ("d0p01_minus", "d0p005_minus", "d0p005_plus", "d0p01_plus")


def _tag(delta_ang: float) -> str:
    magnitude = "d" + f"{abs(delta_ang):.6g}".replace(".", "p")
    return f"{magnitude}_{'plus' if delta_ang > 0 else 'minus'}"


def _run_dir(direction: str, delta_ang: float) -> Path:
    delta_ang = round(delta_ang, 10)
    if direction == "C1_x" and delta_ang in _HISTORICAL_C1X:
        return PROD_SZ_ROOT / "C1_x" / _HISTORICAL_C1X[delta_ang]
    return PROD_SZ_ROOT / direction / _tag(delta_ang)


def _label(run_dir: Path) -> str:
    for line in (run_dir / "RUN.fdf").read_text(encoding="utf-8").splitlines():
        if line.strip().lower().startswith("systemlabel"):
            return line.split(None, 1)[1].strip()
    raise ValueError(f"no SystemLabel in {run_dir}/RUN.fdf")


def _tshs_path(run_dir: Path) -> Path:
    return run_dir / f"{_label(run_dir)}.TSHS"


def _point(run_dir: Path):
    """(hamiltonian, fermi_ev, vacuum_ev, tshs_path) for one real SIESTA run."""
    label = _label(run_dir)
    tshs_path = run_dir / f"{label}.TSHS"
    tshs = sisl.get_sile(str(tshs_path))
    h = tshs.read_hamiltonian()
    fermi = float(tshs.read_fermi_level())
    vacuum, _ = vacuum_level(sisl.get_sile(str(run_dir / f"{label}.VT")).read_grid().grid)
    return h, fermi, vacuum, tshs_path


def _k_matrix(h, fermi: float, vacuum: float, k):
    """K(R) = H_abs(R) - c_vac(R) S(R), in the certified PAO gauge."""
    s = np.asarray(h.Sk(k=k, format="array"))
    h_abs = np.asarray(h.Hk(k=k, format="array")) + fermi * s
    return h_abs - vacuum * s


def _support_pairs(tshs_path: Path) -> list:
    h = sisl.get_sile(str(tshs_path)).read_hamiltonian()
    geom = h.geometry
    rows, cols = h.nonzero()[:2]
    atom_i = geom.o2a(rows)
    atom_j = geom.o2a(cols % geom.no)
    isc = geom.o2isc(cols)
    return sorted(set(zip(atom_i.tolist(), atom_j.tolist(), map(tuple, isc.tolist()))))


def _d5(points: dict, h_ang: float) -> np.ndarray:
    total = sum(D5_WEIGHTS[step] * points[step] for step in D5_WEIGHTS)
    return total / (12.0 * h_ang)


def _d2(points: dict, h_ang: float) -> np.ndarray:
    total = sum(D2_WEIGHTS[step] * points[step] for step in D2_WEIGHTS)
    return total / (2.0 * h_ang)


def _compare(a: list, b: list, s0_by_k: list) -> dict:
    """Whitened RMS/rel/max of (a - b), worst-case across K_POINTS."""
    worst_rms = worst_rel = worst_max = 0.0
    per_k = []
    for ik in range(len(a)):
        diff = a[ik] - b[ik]
        rms, mx = _rms(diff, s0_by_k[ik])
        signal_rms, _ = _rms(a[ik], s0_by_k[ik])
        rel = rms / signal_rms if signal_rms > np.finfo(float).tiny else float("inf")
        worst_rms, worst_rel, worst_max = max(worst_rms, rms), max(worst_rel, rel), max(worst_max, mx)
        per_k.append({"rms_ev_per_ang": rms, "relative": rel, "max_ev_per_ang": mx})
    return {
        "worst_rms_ev_per_ang": worst_rms, "worst_relative": worst_rel, "worst_max_ev_per_ang": worst_max,
        "per_k": per_k,
        "passed": worst_rms < RMS_LIMIT_EV_PER_ANG and worst_rel < REL_LIMIT and worst_max < MAX_LIMIT_EV_PER_ANG,
    }


def main() -> int:
    prereg = json.loads(PREREG_PATH.read_text())
    if not prereg.get("frozen"):
        raise RuntimeError("preregistration is not frozen; refusing to adjudicate against it")

    central_label, central_h, central_fermi, central_vacuum, atom = _matrix_data(PROD_CENTRAL)
    shape = tuple(int(x) for x in sisl.get_sile(str(PROD_CENTRAL / f"{central_label}.VT")).read_grid().shape)
    base = _basis_geometry(central_h.geometry, PROD_CENTRAL / "C.ion.xml")
    dvolume = abs(np.linalg.det(base.cell)) / np.prod(shape)

    k0_by_k = [_k_matrix(central_h, central_fermi, central_vacuum, k) for k in K_POINTS]
    s0_by_k = [np.asarray(central_h.Sk(k=k, format="array")) for k in K_POINTS]

    # -- topology: independently re-verify {(i,j,R)} across every geometry ---
    topology_dirs = [PROD_CENTRAL] + [
        _run_dir("C1_x", sign * h) for h in H_LADDER for sign in (-2, -1, 1, 2)
    ] + [PROD_SZ_ROOT / "translation_x" / tag for tag in _ALL_TRANSLATION_TAGS]
    reference_pairs = _support_pairs(_tshs_path(PROD_CENTRAL))
    reference_hash = input_signature_sha256({"kind": "support_pairs", "pairs": reference_pairs})
    topology_rows = []
    topology_constant = True
    per_h_support_sets: list[set] = []
    for run_dir in topology_dirs:
        tshs_path = _tshs_path(run_dir)
        pairs = _support_pairs(tshs_path)
        pair_hash = input_signature_sha256({"kind": "support_pairs", "pairs": pairs})
        matches = pair_hash == reference_hash
        topology_constant &= matches
        topology_rows.append({
            "run_dir": str(run_dir.relative_to(REPO_ROOT)),
            "tshs_sha256": file_sha256(tshs_path),
            "support_pair_count": len(pairs),
            "support_sha256": pair_hash,
            "matches_reference": matches,
        })
        per_h_support_sets.append(set(pairs))

    # -- siesta_sparse_support_signature: per-h SIESTA TSHS sparse support ---
    # {(i,j,R): H_ij(R) exists in the sparse TSHS}, independent from
    # `graph2mat_graph_topology_hash`-style fields (Graph2Mat graph
    # connectivity, computed elsewhere by batch_topology_hash in
    # graph2mat_autograd_derivatives.py -- NOT this). This block documents the
    # SIESTA-specific sparse-support semantics explicitly so it is never
    # confused with a requirement that SIESTA itself have a Graph2Mat-style
    # graph topology: it is a support-SET signature (which (i,j,R) blocks are
    # populated in the sparse TSHS at all), not a graph-topology claim. Added
    # alongside the existing `topology` block above (not replacing it) to
    # preserve backward compatibility of already-PASSing artifacts.
    support_union: set = set().union(*per_h_support_sets) if per_h_support_sets else set()
    support_intersection: set = (
        set.intersection(*per_h_support_sets) if per_h_support_sets else set()
    )
    appearing_or_disappearing = sorted(support_union - support_intersection)
    siesta_sparse_support_signature = {
        "semantics": (
            "per-h SIESTA sparse-support-set signature: {(i,j,R): H_ij(R) exists in "
            "the sparse TSHS}, union/intersection across the h-ladder (incl. PROD_CENTRAL "
            "and translation_x runs). This is a sparse-support-signature check, not a "
            "graph-topology-hash check; it says nothing about (and does not require) "
            "SIESTA having a Graph2Mat-style graph topology."
        ),
        "n_runs": len(per_h_support_sets),
        "union_count": len(support_union),
        "intersection_count": len(support_intersection),
        "union_equals_intersection": support_union == support_intersection,
        "appearing_or_disappearing_blocks": [list(t) for t in appearing_or_disappearing],
        "n_appearing_or_disappearing_blocks": len(appearing_or_disappearing),
        "union_sha256": input_signature_sha256({"kind": "support_union", "pairs": sorted(support_union)}),
        "intersection_sha256": input_signature_sha256(
            {"kind": "support_intersection", "pairs": sorted(support_intersection)}
        ),
        "not_to_be_confused_with": "graph2mat_graph_topology_hash (Graph2Mat graph connectivity; unrelated field)",
    }

    # -- Delta(h) end-to-end for every h in the ladder -----------------------
    delta_by_h: dict[float, list] = {}
    provenance_by_h: dict[float, dict] = {}
    d2_diagnostic_by_h: dict[float, dict] = {}
    blocked_reason = None
    try:
        for h in H_LADDER:
            points_by_step = {}
            provenance = {}
            for step in D5_WEIGHTS:
                run_dir = _run_dir("C1_x", step * h)
                hamiltonian, fermi, vacuum, tshs_path = _point(run_dir)
                points_by_step[step] = hamiltonian, fermi, vacuum
                provenance[step] = {
                    "run_dir": str(run_dir.relative_to(REPO_ROOT)),
                    "tshs_sha256": file_sha256(tshs_path),
                    "fermi_ev": fermi, "vacuum_ev": vacuum,
                }
            provenance_by_h[h] = provenance
            connections = _connections("C1_x", h, base, atom, shape, dvolume, K_POINTS)

            deltas = []
            d2_norms = []
            for ik, k in enumerate(K_POINTS):
                k_by_step = {step: _k_matrix(*points_by_step[step], k) for step in points_by_step}
                d5k = _d5(k_by_step, h)
                d2k = _d2({s: k_by_step[s] for s in D2_WEIGHTS}, h)
                s_l, s_r = connections[ik]
                left_term, _ = solve(s0_by_k[ik], k0_by_k[ik])
                right_term, _ = solve(s0_by_k[ik], s_r)
                delta = d5k - s_l @ left_term - k0_by_k[ik] @ right_term
                deltas.append(delta)
                d2_rms, d2_max = _rms(d5k - d2k, s0_by_k[ik])
                d2_norms.append({"k_reduced": list(k), "d5_vs_d2_K_rms_ev_per_ang": d2_rms,
                                  "d5_vs_d2_K_max_ev_per_ang": d2_max})
            delta_by_h[h] = deltas
            d2_diagnostic_by_h[h] = d2_norms
    except (KeyError, FileNotFoundError, ValueError, np.linalg.LinAlgError) as exc:
        blocked_reason = f"canonical connection route could not produce a value: {exc}"

    if blocked_reason is not None:
        report = {
            "schema": "delta_plateau_topology_certification_v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "delta_plateau_topology_certification": "BLOCKED",
            "C1x_delta_plateau": "BLOCKED",
            "graphene_Gamma_E2g_delta_convergence": "BLOCKED",
            "blocked_reason": blocked_reason,
            "topology": {"constant": topology_constant, "rows": topology_rows},
            "siesta_sparse_support_signature": siesta_sparse_support_signature,
        }
    else:
        e12 = _compare(delta_by_h[0.005], delta_by_h[0.0025], s0_by_k)
        e23 = _compare(delta_by_h[0.0025], delta_by_h[0.00125], s0_by_k)
        r1 = [(16.0 * delta_by_h[0.0025][ik] - delta_by_h[0.005][ik]) / 15.0 for ik in range(len(K_POINTS))]
        r2 = [(16.0 * delta_by_h[0.00125][ik] - delta_by_h[0.0025][ik]) / 15.0 for ik in range(len(K_POINTS))]
        richardson_stability = _compare(r1, r2, s0_by_k)
        production_vs_extrapolated = _compare(delta_by_h[0.005], r2, s0_by_k)

        c1x_plateau_passed = (
            topology_constant and e12["passed"] and e23["passed"]
            and richardson_stability["passed"] and production_vs_extrapolated["passed"]
        )

        report = {
            "schema": "delta_plateau_topology_certification_v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "gate": "delta_plateau_topology_certification",
            "h_ladder_ang": list(H_LADDER),
            "decisive_window_ang": list(DECISIVE_WINDOW),
            "thresholds": {"rms_ev_per_ang": RMS_LIMIT_EV_PER_ANG, "relative": REL_LIMIT,
                            "max_ev_per_ang": MAX_LIMIT_EV_PER_ANG},
            "topology": {"constant": topology_constant, "reference_sha256": reference_hash, "rows": topology_rows},
            "siesta_sparse_support_signature": siesta_sparse_support_signature,
            "C1x_delta_plateau": {
                "verdict": "PASS" if c1x_plateau_passed else "NO_GO",
                "e12_0p005_to_0p0025": e12,
                "e23_0p0025_to_0p00125": e23,
                "richardson": {
                    "r1_formula": "(16*Delta(0.0025) - Delta(0.005)) / 15",
                    "r2_formula": "(16*Delta(0.00125) - Delta(0.0025)) / 15",
                    "r1_vs_r2": richardson_stability,
                    "production_0p005_vs_r2": production_vs_extrapolated,
                },
                "d2_vs_d5_K_diagnostic_only": {str(h): rows for h, rows in d2_diagnostic_by_h.items()},
                "h_0p000625_role": "noise-floor probe; computed but not required to improve monotonically",
                "run_provenance_by_h": {str(h): provenance_by_h[h] for h in H_LADDER},
            },
            "graphene_Gamma_E2g_delta_convergence": {
                "verdict": "BLOCKED",
                "reason": (
                    "no certified fixed (C_i, C_f) electronic coefficient artifact exists for the "
                    "E2g contraction, and no exact certified symmetry relation reduces it to C1:x; "
                    "not approximated or interpolated"
                ),
            },
            "delta_plateau_topology_certification": "PASS" if c1x_plateau_passed else "NO_GO",
            "gates_translation_gate_on": "C1x_delta_plateau",
            "unaffected_by_this_gate": {
                "checkpoint_lineage": "NO_GO", "fine_tuning_candidate": "BLOCKED",
                "delta_out_closure": "NO_GO", "full_KS": "BLOCKED",
            },
        }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "delta_plateau_topology_certification.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True, default=float) + "\n")
    print(f"delta_plateau_topology_certification: {report['delta_plateau_topology_certification']} -> {path}")
    return 0 if report["delta_plateau_topology_certification"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
