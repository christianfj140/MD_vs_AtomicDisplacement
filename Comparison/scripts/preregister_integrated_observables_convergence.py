#!/usr/bin/env python3
"""E-F_001-S53: freeze the convergence protocol for integrated rigid-model observables.

Roadmap Fase 12 (``|g|^2, gamma_qnu, lambda_qnu, lambda, alpha2F(omega)``) is explicitly
``DEFERRED``: "Solo despues de GO-10 o, para un resultado explicitamente rigid-model,
despues de GO-9." ``shared/artifact_signature.py``'s own DAG makes the dependency
mechanical, not just prose: the ``integrated_observable`` node's only dependency is
``g_set`` (one or more ``pao_projected_mode_coupling`` artifacts). Read live today:

    certify_rigid_gamma_campaign.py   (S43) -> go9_status == "NO-GO-9"
    certify_qneq0_rigid_campaign.py   (S45) -> go9_status == "NO-GO-9"

so ``g_set`` is empty -- there is no PAO-projected coupling anywhere in this repository, on
either branch. Sweeping a k/q mesh, a broadening width or a DOS normalization against
a coupling constant that does not exist would fabricate a plateau. What this ticket
does instead, in the pattern S41/S44/S50 already established for a gated ticket: freeze
the numerical protocol a future execution must use --

    * a Gamma-centered Monkhorst-Pack k/q mesh generator (2D, shared by k and q: both
      live in the same moire Brillouin zone, roadmap Section XII/Fase 12),
    * the modal-completeness rule (``3 * n_atoms_per_cell`` branches per q; "selected
      modes aislados no bastan" per this ticket's own Requirements),
    * the smearing contract, reused verbatim from ``epc_occupations.py`` (roadmap XII)
      rather than re-declared,
    * a broadening contract for the observable-level energy-conserving delta (distinct
      from electronic smearing -- ``artifact_signature.py``'s ``integrated_observable``
      node already carries both as separate fields),
    * the spin/valley weight convention (``spin_degeneracy`` from the occupations
      contract times ``VALLEY_DEGENERACY = 2``, graphene/TBG's two inequivalent K/K'
      valleys),
    * the 2D DOS normalization convention,
    * the plateau/uncertainty-evidence rule this ticket's acceptance criteria requires
      ("cada parametro tiene evidencia de plateau o incertidumbre declarada"),

and validate that machinery against synthetic test functions with a known limit (never
against a fabricated ``g``), while gating any real execution behind a live read of GO-9.
``Tc`` stays out of scope, per the roadmap and per this ticket's acceptance criteria.

Units: fractional reciprocal-lattice coordinates for k/q (2D, in [0, 1)); eV for
energies/widths; Ang^2 for cell area; DOS in states/eV/Ang^2.
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

import certify_qneq0_rigid_campaign as go9_qneq0_cert  # noqa: E402
import certify_rigid_gamma_campaign as go9_gamma_cert  # noqa: E402
import epc_occupations  # noqa: E402
import preregister_graphene_gamma_epc as prereg  # noqa: E402
import preregister_matbg_gamma_rigid_epc as gamma_prereg  # noqa: E402

PreregistrationError = prereg.PreregistrationError

SCHEMA = "epc_preregistration_v1"
PROTOCOL_ID = "integrated_observables_convergence_v1"
TICKET = "E-F_001-S53"
GATE = "Fase-12 convergence-protocol pre-registration; execution additionally requires a " \
    "GO-9-certified g_set (either branch) or GO-10"

DEFAULT_OUTPUT_DIR = REPO_ROOT / "Comparison/results/epc/preregistration/matbg"
PROTOCOL_NAME = "epc_integrated_observables_convergence_protocol.json"
DEFAULT_MATBG_CERTIFICATION_DIR = gamma_prereg.DEFAULT_MATBG_CERTIFICATION_DIR

TARGET_ATOM_COUNT = gamma_prereg.TARGET_ATOM_COUNT

# --------------------------------------------------------------------------- #
# Modal completeness (roadmap Fase 12: "mode completeness"; this ticket's own
# Requirements: "selected modes aislados no bastan").
# --------------------------------------------------------------------------- #

PHONON_BRANCHES_PER_ATOM = 3


def required_phonon_branches(n_atoms_per_cell: int) -> int:
    return PHONON_BRANCHES_PER_ATOM * int(n_atoms_per_cell)


def modal_completeness_report(
    present_branch_count: int, *, n_atoms_per_cell: int = TARGET_ATOM_COUNT
) -> dict[str, Any]:
    """Reject any campaign that claims completeness from a handful of selected modes."""
    required = required_phonon_branches(n_atoms_per_cell)
    complete = int(present_branch_count) >= required
    return {
        "n_atoms_per_cell": int(n_atoms_per_cell),
        "required_branches": required,
        "present_branches": int(present_branch_count),
        "complete": complete,
        "rule": "3 * n_atoms_per_cell branches per q; a selected-mode subset (roadmap Fase 10d) "
        "may validate physics but never stands in for the modal sum an integrated observable needs",
    }


# --------------------------------------------------------------------------- #
# k/q mesh: shared 2D Gamma-centered Monkhorst-Pack generator. k and q live in
# the same moire BZ (roadmap Fase 12), so one generator serves both.
# --------------------------------------------------------------------------- #


def monkhorst_pack_2d(nx: int, ny: int) -> tuple[np.ndarray, np.ndarray]:
    """Gamma-centered fractional mesh in [0, 1)^2 with uniform weights summing to 1."""
    if nx < 1 or ny < 1:
        raise PreregistrationError(f"mesh density must be >= 1, got ({nx}, {ny})")
    fx = np.arange(nx, dtype=np.float64) / nx
    fy = np.arange(ny, dtype=np.float64) / ny
    gx, gy = np.meshgrid(fx, fy, indexing="ij")
    points = np.stack([gx.ravel(), gy.ravel()], axis=1)
    weights = np.full(nx * ny, 1.0 / (nx * ny), dtype=np.float64)
    return points, weights


# --------------------------------------------------------------------------- #
# Smearing: reused verbatim from epc_occupations.py (roadmap XII) -- not
# redeclared. Broadening: the *other* delta function, for the observable-level
# energy conservation (e.g. delta(eps - mu) in a DOS, delta(eps_f - eps_i -
# hbar*omega) in gamma_qnu) -- artifact_signature.py's integrated_observable
# node already carries "smearing" and "broadening" as two separate fields.
# --------------------------------------------------------------------------- #

BROADENING_FUNCTIONS = ("gaussian", "lorentzian")
VALLEY_DEGENERACY = 2  # graphene/TBG: two inequivalent K, K' valleys (Piscanec et al. 2004)

# A convergence ladder, not a chosen final value -- the point of this ticket is that no
# single width is picked without plateau evidence (acceptance criteria).
DEFAULT_BROADENING_LADDER_EV = (0.2, 0.1, 0.05, 0.025, 0.0125)
DEFAULT_MESH_DENSITY_LADDER = (6, 12, 24, 48)
DEFAULT_PLATEAU_RELATIVE_TOL = 0.01


def spin_valley_weight(spin_degeneracy: int) -> int:
    return int(spin_degeneracy) * VALLEY_DEGENERACY


def broadening_kernel(delta_e: np.ndarray, width_ev: float, kind: str) -> np.ndarray:
    if width_ev <= 0.0:
        raise PreregistrationError(f"broadening width must be positive, got {width_ev}")
    x = np.asarray(delta_e, dtype=np.float64) / width_ev
    if kind == "gaussian":
        return np.exp(-0.5 * x**2) / (width_ev * np.sqrt(2.0 * np.pi))
    if kind == "lorentzian":
        return (width_ev / np.pi) / (np.asarray(delta_e, dtype=np.float64) ** 2 + width_ev**2)
    raise PreregistrationError(f"broadening kind must be one of {BROADENING_FUNCTIONS}, got {kind!r}")


# --------------------------------------------------------------------------- #
# 2D DOS normalization convention.
# --------------------------------------------------------------------------- #


def gaussian_broadened_dos(
    eigenvalues_ev: np.ndarray,
    k_weights: np.ndarray,
    energy_grid_ev: np.ndarray,
    *,
    width_ev: float,
    cell_area_ang2: float,
    spin_valley_mult: int,
    kind: str = "gaussian",
) -> np.ndarray:
    """g(E) = (spin_valley_mult / cell_area) * sum_k w_k * sum_n kernel(E - eps_kn, width).

    ``eigenvalues_ev`` is (n_k, n_bands); ``k_weights`` sums to 1 over the mesh (a
    :func:`monkhorst_pack_2d` output). Units: states / eV / Ang^2.
    """
    if cell_area_ang2 <= 0.0:
        raise PreregistrationError(f"cell_area_ang2 must be positive, got {cell_area_ang2}")
    eigs = np.asarray(eigenvalues_ev, dtype=np.float64)
    weights = np.asarray(k_weights, dtype=np.float64)
    if eigs.shape[0] != weights.shape[0]:
        raise PreregistrationError(f"{eigs.shape[0]} k-points but {weights.shape[0]} weights")
    grid = np.asarray(energy_grid_ev, dtype=np.float64)
    delta = grid[:, None, None] - eigs[None, :, :]  # (n_grid, n_k, n_bands)
    kernel = broadening_kernel(delta, width_ev, kind)
    per_k = kernel.sum(axis=2)  # (n_grid, n_k)
    dos = (per_k * weights[None, :]).sum(axis=1)
    return dos * (spin_valley_mult / cell_area_ang2)


def dos_normalization_contract() -> dict[str, Any]:
    return {
        "formula": "g(E) = (spin_degeneracy * valley_degeneracy / cell_area_ang2) * "
        "sum_k weight_k * sum_n kernel(E - eps_kn, broadening_width_ev)",
        "units": "states / eV / Ang^2",
        "mesh_weight_normalization": "weights from monkhorst_pack_2d sum to 1 over the full mesh",
        "valley_degeneracy": VALLEY_DEGENERACY,
        "broadening_functions": list(BROADENING_FUNCTIONS),
        "sum_rule": "integral g(E) dE == spin_degeneracy * valley_degeneracy / cell_area_ang2, "
        "independent of broadening width or kind (self-check in this module's demo())",
    }


# --------------------------------------------------------------------------- #
# Plateau / uncertainty evidence -- the acceptance-criteria mechanism itself.
# Generic over any already-evaluated convergence ladder (mesh density, broadening
# width, or FC range alike); never assumes what it is converging.
# --------------------------------------------------------------------------- #


def mesh_convergence_report(
    name: str, densities: Sequence[float], values: Sequence[float], *, tol: float = DEFAULT_PLATEAU_RELATIVE_TOL
) -> dict[str, Any]:
    if len(densities) != len(values):
        raise PreregistrationError(f"{len(densities)} densities but {len(values)} values")
    if len(values) < 2:
        raise PreregistrationError("need at least two points on the ladder to judge convergence")
    vals = np.asarray(values, dtype=np.float64)
    diffs = np.abs(np.diff(vals)) / np.maximum(np.abs(vals[1:]), 1e-300)
    plateau = bool(diffs[-1] <= tol)
    return {
        "name": name,
        "densities": list(densities),
        "values": [float(v) for v in vals],
        "relative_diffs": [float(d) for d in diffs],
        "tol": float(tol),
        "evidence": "plateau" if plateau else "uncertainty_declared",
        "declared_uncertainty": None if plateau else float(diffs[-1]),
    }


# --------------------------------------------------------------------------- #
# Preconditions: g_set needs at least one GO-9-certified branch. Read live.
# --------------------------------------------------------------------------- #


def preconditions(paths: Mapping[str, Path]) -> list[dict[str, Any]]:
    go9_gamma = prereg._read_json(paths["go9_gamma"])
    go9_qneq0 = prereg._read_json(paths["go9_qneq0"])
    return [
        prereg._precondition(
            "GO-9 rigid-Gamma campaign (S43)", paths["go9_gamma"], go9_gamma,
            verdict="PASS" if (go9_gamma or {}).get("go9_status") == "GO-9" else (go9_gamma or {}).get("go9_status"),
            required=False,
        ),
        prereg._precondition(
            "GO-9 rigid q!=0 campaign (S45)", paths["go9_qneq0"], go9_qneq0,
            verdict="PASS" if (go9_qneq0 or {}).get("go9_status") == "GO-9" else (go9_qneq0 or {}).get("go9_status"),
            required=False,
        ),
    ]


def physics_review(gates: list[dict[str, Any]]) -> dict[str, Any]:
    any_go9 = any(row["status"] == "PASS" for row in gates)
    items = [
        {
            "item": "mesh_generator",
            "decision": "approved",
            "evidence": {"generator": "monkhorst_pack_2d", "shared_by": ["k", "q"]},
            "basis": "pure numerical machinery, testable now against synthetic functions with a "
            "known limit; does not depend on any PAO-projected coupling",
        },
        {
            "item": "modal_completeness_rule",
            "decision": "approved",
            "evidence": {"required_branches": required_phonon_branches(TARGET_ATOM_COUNT)},
            "basis": "3 * n_atoms_per_cell, rejects any selected-mode subset as a substitute",
        },
        {
            "item": "smearing_contract",
            "decision": "approved",
            "evidence": {"reused_from": "epc_occupations.py", "schema": epc_occupations.occupations_contract()["schema"]},
            "basis": "roadmap XII; not redeclared here",
        },
        {
            "item": "broadening_and_dos_normalization",
            "decision": "approved",
            "evidence": dos_normalization_contract(),
            "basis": "separate from electronic smearing per artifact_signature.py's "
            "integrated_observable node fields",
        },
        {
            "item": "g_set",
            "decision": "approved" if any_go9 else "blocked",
            "evidence": {row["name"]: row["status"] for row in gates},
            "basis": "artifact_signature.py: integrated_observable depends only on g_set "
            "(pao_projected_mode_coupling, physical); with both GO-9 branches NO-GO-9 today, g_set is "
            "empty and no sweep over a real coupling constant is authorised",
        },
    ]
    return {
        "reviewer": "automated_precondition_review (evidence-bound)",
        "items": items,
        "status": "APPROVED" if all(row["decision"] == "approved" for row in items) else "BLOCKED",
        "human_sign_off": None,
    }


def build_protocol(
    *,
    gamma_campaign_certification: Path = DEFAULT_MATBG_CERTIFICATION_DIR / go9_gamma_cert.REPORT_NAME,
    qneq0_campaign_certification: Path = DEFAULT_MATBG_CERTIFICATION_DIR / go9_qneq0_cert.REPORT_NAME,
) -> dict[str, Any]:
    gate_paths = {"go9_gamma": gamma_campaign_certification, "go9_qneq0": qneq0_campaign_certification}
    gates = preconditions(gate_paths)
    review = physics_review(gates)

    blocking = [f"{row['name']}: {row['status']}" for row in gates if row["status"] != "PASS"]
    if not any(row["status"] == "PASS" for row in gates):
        blocking.append(
            "g_set_empty: no pao_projected_mode_coupling artifact exists on either branch; "
            "integrated_observable (artifact_signature.py) has nothing to depend on yet"
        )

    protocol: dict[str, Any] = {
        "schema": SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "ticket": TICKET,
        "gate": GATE,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "frozen_content_sha256": None,
        "scope": {
            "in_scope": ["k_mesh", "q_mesh", "modal_completeness", "smearing", "broadening",
                         "occupations", "spin_valley_weights", "dos_normalization_2d"],
            "explicitly_out_of_scope": ["Tc"],
        },
        "mesh": {
            "generator": "monkhorst_pack_2d(nx, ny)",
            "convention": "Gamma-centered, fractional [0, 1)^2, shared by k and q (same moire BZ)",
            "default_density_ladder": list(DEFAULT_MESH_DENSITY_LADDER),
        },
        "modal_completeness": modal_completeness_report(0, n_atoms_per_cell=TARGET_ATOM_COUNT),
        "smearing": epc_occupations.occupations_contract(),
        "broadening": {
            "functions": list(BROADENING_FUNCTIONS),
            "default_width_ladder_ev": list(DEFAULT_BROADENING_LADDER_EV),
            "distinct_from_smearing": "smearing fills electronic states (roadmap XII); broadening "
            "is the observable-level energy-conserving delta (integrated_observable node)",
        },
        "spin_valley": {"valley_degeneracy": VALLEY_DEGENERACY, "formula": "spin_degeneracy * valley_degeneracy"},
        "dos_normalization": dos_normalization_contract(),
        "convergence_evidence_rule": {
            "method": "mesh_convergence_report(name, densities, values, tol)",
            "default_relative_tol": DEFAULT_PLATEAU_RELATIVE_TOL,
            "rule": "plateau requires the last relative difference on the ladder <= tol; otherwise "
            "the residual itself is the declared uncertainty -- never silently dropped",
        },
        "preconditions": gates,
        "physics_review": review,
        "blocking": blocking,
        "ready_to_execute": not blocking and review["status"] == "APPROVED",
        "backend": {
            "requested_backend": "cpu",
            "effective_backend": "cpu",
            "backend_preflight": "not_required",
            "backend_fallback_reason": "this ticket only freezes mesh/smearing/broadening/DOS math "
            "and validates it against synthetic functions (negligible cost); the GPU preflight "
            "applies to the future g_set campaign this protocol gates, not to this ticket's own math",
        },
    }
    protocol["frozen_content_sha256"] = prereg.freeze_hash(protocol)
    return prereg._json_safe(protocol)


def load_protocol(path: Path = DEFAULT_OUTPUT_DIR / PROTOCOL_NAME) -> dict[str, Any]:
    payload = prereg._read_json(Path(path))
    if payload is None:
        raise PreregistrationError(f"no pre-registered protocol at {path}")
    return payload


def verify(path: Path = DEFAULT_OUTPUT_DIR / PROTOCOL_NAME) -> dict[str, Any]:
    protocol = load_protocol(path)
    recomputed = prereg.freeze_hash(protocol)
    stored = protocol.get("frozen_content_sha256")
    return {
        "protocol": str(path),
        "protocol_id": protocol.get("protocol_id"),
        "frozen_content_sha256": stored,
        "recomputed_sha256": recomputed,
        "intact": stored == recomputed,
        "verified": stored == recomputed,
    }


def require_frozen_protocol(path: Path = DEFAULT_OUTPUT_DIR / PROTOCOL_NAME) -> dict[str, Any]:
    protocol = load_protocol(path)
    if prereg.freeze_hash(protocol) != protocol.get("frozen_content_sha256"):
        raise PreregistrationError(f"{path} has been edited since it was frozen")
    if not protocol.get("ready_to_execute"):
        raise PreregistrationError(
            f"the protocol is frozen but its preconditions are not met: {protocol.get('blocking')}"
        )
    return protocol


def _selfcheck() -> None:
    """The one runnable check this module's non-trivial math needs (ponytail)."""
    points, weights = monkhorst_pack_2d(12, 12)
    assert points.shape == (144, 2) and np.isclose(weights.sum(), 1.0)

    assert not modal_completeness_report(4, n_atoms_per_cell=11164)["complete"]
    assert modal_completeness_report(33492, n_atoms_per_cell=11164)["complete"]

    # DOS sum rule: integral over a wide grid recovers spin_valley/area, any width/kind.
    rng = np.random.default_rng(0)
    eigs = rng.uniform(-1.0, 1.0, size=(200, 4))
    _, w = monkhorst_pack_2d(20, 10)
    grid = np.linspace(-5.0, 5.0, 4001)
    for kind in BROADENING_FUNCTIONS:
        dos = gaussian_broadened_dos(eigs, w, grid, width_ev=0.05, cell_area_ang2=3.0,
                                      spin_valley_mult=4, kind=kind)
        integral = np.trapezoid(dos, grid)
        expected = 4 * eigs.shape[1] / 3.0
        # Lorentzian tails are heavy: a +-5 eV grid at 0.05 eV width still misses ~0.6%.
        tol = 5e-3 if kind == "gaussian" else 1e-2
        assert abs(integral - expected) / expected < tol, (kind, integral, expected)

    # Convergence detector: a smooth sum over an increasing mesh must plateau.
    values = []
    for n in DEFAULT_MESH_DENSITY_LADDER:
        pts, wts = monkhorst_pack_2d(n, n)
        values.append(float(np.sum(wts * np.cos(2 * np.pi * pts[:, 0]) ** 2)))
    report = mesh_convergence_report("cos2_test", DEFAULT_MESH_DENSITY_LADDER, values, tol=0.01)
    assert report["evidence"] == "plateau", report

    # A ladder that has not settled must declare an uncertainty, not a false plateau.
    diverging = mesh_convergence_report("not_converged", [1, 2], [1.0, 2.0], tol=0.01)
    assert diverging["evidence"] == "uncertainty_declared" and diverging["declared_uncertainty"] is not None


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--verify", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    output = args.output_dir / PROTOCOL_NAME

    if args.verify:
        report = verify(output)
        print(f"pre-registration {report['protocol_id']}: {'INTACT' if report['intact'] else 'TAMPERED'} "
              f"{report['frozen_content_sha256']}")
        return 0 if report["verified"] else 1

    _selfcheck()
    protocol = build_protocol()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(protocol, indent=2, sort_keys=True), encoding="utf-8")

    print(f"pre-registration {protocol['protocol_id']} -> {output}")
    print(f"  frozen_content_sha256 : {protocol['frozen_content_sha256']}")
    print(f"  physics review        : {protocol['physics_review']['status']}")
    print(f"  ready to execute      : {protocol['ready_to_execute']}")
    for reason in protocol["blocking"]:
        print(f"    blocked by: {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
