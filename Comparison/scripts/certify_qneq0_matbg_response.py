#!/usr/bin/env python3
"""E-F_001-S40: GO-8b certification of the scalable MATBG ``q != 0`` response.

S33 (``docs/epc_s33_ruta_qneq0_produccion.md``) named Q-C as the route that
*would* become primary for MATBG ``q != 0`` production (Q-B kept as a
per-``q`` cross-check) *if and when* GO-5 clears -- ``production_route_
decision()`` itself reports ``primary_route=None`` while GO-5 is NO_GO, and
this module reads that value rather than repeating the name, so it cannot
drift out of sync with S33 again. S33 also pre-registered eight things
``additional_tests_required_before_go8b`` had to demonstrate first. This
ticket's own kernel mechanics happen to be built against Q-C's real-space
kernel regardless of whether Q-C is ever authorised -- that is a fact about
what this module tests, not a claim that Q-C has been selected. It runs that
exact list
against :mod:`epc_qb_qc_graph2mat_kernel` (the real-production-JVP
implementation S40 adds) and reports the result the same two-suite way
``certify_matbg_phonon_provider.py`` (S39, GO-8a) reports its own gate:

* ``route_mechanics_suite`` runs unconditionally today, because none of it
  needs a physical basis-response validation to be true -- it is entirely
  about whether the chosen route's *machinery* (real JVP, not finite
  differences; topology margin; the five S31-style invariance checks;
  cache/restart; GPU preflight; sparse serialisation; shell-scaling) works,
  independent of whether GO-4/GO-5 have cleared yet.
* ``graphene_k_agreement_suite`` is the one piece GO-8b's own item 4 asks
  for -- agreement against Q-A on *real* graphene K -- and it is NOT_RUN,
  blocked by GO-5's own ``NO_GO`` (``docs/epc_s31_graphene_k_go5_verdict.md``,
  itself blocked by GO-4/C14C's unresolved intra-atomic basis-response term).
  Running it anyway would only re-report that same open gate under a
  different label, not exercise anything new about ``q``-space mechanics
  (the same reasoning ``epc_qb_qc_prototypes`` and ``epc_commensurate_k``
  already used to defer their own real-graphene runs).

``go8b_status()`` is therefore honestly ``NO-GO-8b`` until GO-4/GO-5 clear --
this ticket's job is to make sure nothing *else* is still missing when they
do, and to make a Gamma-JVP-reuse-for-q!=0 an automatic, structural failure
rather than a promise in a docstring (GO-8b's own acceptance criterion).
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
for _path in (SCRIPT_DIR, REPO_ROOT / "shared"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import epc_commensurate_k as ck  # noqa: E402
import epc_qb_qc_graph2mat_kernel as g2m_kernel  # noqa: E402
import epc_qb_qc_prototypes as prototypes  # noqa: E402

SCHEMA = "matbg_qneq0_response_go8b_certification_v1"
TICKET = "E-F_001-S40"
MEMO = "docs/epc_s40_matbg_qneq0_response_certification.md"
GATE = "GO-8b"

DEFAULT_OUTPUT_DIR = REPO_ROOT / "Comparison/results/epc/certification/matbg"
REPORT_NAME = "s40_matbg_qneq0_response_go8b_certification.json"

# S33's own pre-registered proxy sweep range (docs/epc_s33_ruta_qneq0_produccion.md
# Sec. 4 item 7): a range of shells, since no real MATBG edge count exists yet
# (epc_qb_qc_graph2mat_kernel.MATBG_EDGE_COUNT_BLOCKING_REASON).
DEFAULT_SHELLS_SWEEP = (1, 4, 16, 64)


class MatbgQneq0ResponseCertificationError(RuntimeError):
    """A GO-8b certification was built, read, or trusted in a way this gate forbids."""


def check(name: str, passed: bool, detail: str, *, applicable: bool = True, **extra: Any) -> dict[str, Any]:
    return {
        "check": name,
        "passed": bool(passed) if applicable else True,
        "applicable": bool(applicable),
        "detail": detail,
        **extra,
    }


# --------------------------------------------------------------------------- #
# route_mechanics_suite: runs unconditionally against the real production JVP
# --------------------------------------------------------------------------- #


def _certify_route_mechanics() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    # S33 items 1 + 3: the chosen route (Q-C, plus Q-B as cross-check)
    # implemented against the real Graph2Mat production JVP, and the five
    # S31-style invariance checks repeated on it.
    real_checks = g2m_kernel.real_kernel_checks()
    for entry in real_checks["details"]:
        checks.append(
            check(
                entry["name"], entry["status"] == "PASS",
                f"worst_abs_diff={entry.get('worst_abs_diff', entry.get('round_trip_max_abs_diff'))!r}",
                **{key: value for key, value in entry.items() if key not in ("name", "status")},
            )
        )

    # S33 item 2: topology margin, reusing fd_perturbation_space.topology_margin.
    margin = g2m_kernel.topology_margin_report()
    checks.append(
        check(
            "topology_margin_preserved", bool(margin["topology_preserved"]),
            f"min_margin_ang={margin['min_margin_ang']:.4g} at cutoff {margin['cutoff_ang']:.4g}, "
            f"{len(margin['crossing_pairs'])} crossing pair(s)",
            min_margin_ang=margin["min_margin_ang"], crossing_pairs=margin["crossing_pairs"],
        )
    )

    # S33 item 6: GPU preflight, extending resolve_jvp_backend rather than
    # repeating it (Default compute policy: never a silent CPU fallback).
    preflight = g2m_kernel.gpu_preflight_report()
    checks.append(
        check(
            "gpu_preflight_recorded",
            preflight["backend_preflight"] in ("passed", "failed", "not_required"),
            f"requested={preflight['requested_backend']!r} effective={preflight['effective_backend']!r} "
            f"preflight={preflight['backend_preflight']!r} reason={preflight['backend_fallback_reason']!r}",
            **preflight,
        )
    )

    # S33 item 8: sparse serialisation round trip through the existing
    # Graph2Mat -> sparse mapping (C11).
    sparse = g2m_kernel.sparse_round_trip_check()
    checks.append(
        check(
            "sparse_serialization_round_trip",
            bool(sparse["reproducible_across_calls"] and sparse["matches_result_metadata_nnz"]),
            f"nnz={sparse['nnz']} shape={sparse['matrix_shape']} reproducible={sparse['reproducible_across_calls']}",
            **sparse,
        )
    )

    # S33 item 5: raw_derivative_kernel_table cache/restart -- kill the
    # process after kernel_table() completes (simulated: discard the
    # in-memory table), resume from disk, confirm no finite difference /
    # JVP reruns and the reloaded table reproduces the same D_H(k+q, k).
    restart = _certify_cache_restart()
    checks.append(restart)

    # S33 item 7 / GO-8b requirement: a scaling sweep, at the real MATBG
    # shell/edge count where possible, otherwise an honestly-labelled proxy
    # (no real MATBG edge count has ever been measured, S37 OOM'd first).
    sweep = g2m_kernel.scaling_sweep_graph2mat(DEFAULT_SHELLS_SWEEP)
    checks.append(
        check(
            "jvp_calls_independent_of_shells", bool(sweep["jvp_calls_independent_of_shells"]),
            f"jvp_calls constant across shells={DEFAULT_SHELLS_SWEEP}: {sweep['jvp_calls_independent_of_shells']}; "
            f"real MATBG edge count measured: {sweep['matbg_edge_count_measured']} "
            f"({sweep['matbg_edge_count_blocking_reason']})",
            scaling_sweep=sweep,
        )
    )

    # GO-8b acceptance criterion: a Gamma JVP reused for q != 0 must fail
    # automatically, not just by convention.
    guard = _certify_gamma_jvp_reuse_guard()
    checks.append(guard)

    passed = all(entry["passed"] for entry in checks if entry["applicable"])
    return {"status": "PASS" if passed else "FAIL", "checks": checks}


def _certify_cache_restart() -> dict[str, Any]:
    import tempfile

    table = g2m_kernel.kernel_table_graph2mat()
    node = g2m_kernel.kernel_table_dag_node(table)
    with tempfile.TemporaryDirectory() as tmpdir:
        npz_path = Path(tmpdir) / "kernel_table.npz"
        metadata_path = Path(tmpdir) / "kernel_table.json"
        g2m_kernel.save_kernel_table_cache(table, node, npz_path, metadata_path)
        loaded, status = g2m_kernel.load_kernel_table_cache(node, npz_path, metadata_path)
        if loaded is None:
            return check(
                "kernel_table_cache_restart", False,
                f"reload after 'process kill' returned status={status!r}, expected valid", cache_status=status,
            )
        before = g2m_kernel.k_to_k_plus_q_kernel_graph2mat(table, (0.15, 0.0, 0.0), (0.2, 0.0, 0.0))
        after = g2m_kernel.k_to_k_plus_q_kernel_graph2mat(loaded, (0.15, 0.0, 0.0), (0.2, 0.0, 0.0))
        diff = float(np.abs(before["D_H"] - after["D_H"]).max())
        # A geometry/direction change must invalidate the cache -- not just
        # succeed once. Confirms the restart path is signature-checked, not
        # a blind reuse.
        stale_table = g2m_kernel.kernel_table_graph2mat(direction=[[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]])
        stale_node = g2m_kernel.kernel_table_dag_node(stale_table)
        _, stale_status = g2m_kernel.load_kernel_table_cache(stale_node, npz_path, metadata_path)
    passed = diff < 1e-9 and status == "valid" and stale_status != "valid"
    return check(
        "kernel_table_cache_restart", passed,
        f"reload status={status!r}, D_H(k+q,k) diff after restart={diff:.3e}, stale-direction reload "
        f"correctly rejected as {stale_status!r}",
        cache_status=status, restart_diff=diff, stale_direction_status=stale_status,
    )


def _certify_gamma_jvp_reuse_guard() -> dict[str, Any]:
    import torch

    from graph2mat_autograd_derivatives import compute_graph2mat_directional_derivative

    model = g2m_kernel.PeriodicShellGraphModel()
    home0 = torch.as_tensor(np.asarray(prototypes.PRIMITIVE_POSITIONS_ANG, dtype=np.float64), dtype=torch.float64)
    batch = g2m_kernel.KernelBatch(positions=torch.cat([home0, home0.clone()]))
    dvec = torch.as_tensor(g2m_kernel.DEFAULT_DIRECTION_VECTORS, dtype=torch.float64)
    zero = torch.zeros_like(dvec)
    gamma_result = compute_graph2mat_directional_derivative(model, batch, torch.cat([zero, dvec]))
    fake_table = {"schema": gamma_result.metadata["schema"]}
    try:
        g2m_kernel.k_to_k_plus_q_kernel_graph2mat(fake_table, (0.1, 0.0, 0.0), (0.2, 0.0, 0.0))
        rejected = False
        error_text = None
    except g2m_kernel.GammaJvpReuseError as error:
        rejected = True
        error_text = str(error)
    return check(
        "gamma_jvp_reuse_for_qneq0_rejected", rejected,
        "a plain compute_graph2mat_directional_derivative (q=0 Gamma-periodic JVP) payload fed "
        f"into k_to_k_plus_q_kernel_graph2mat for q != 0 {'raised GammaJvpReuseError as required' if rejected else 'was NOT rejected -- GO-8b violation'}",
        error=error_text,
    )


# --------------------------------------------------------------------------- #
# graphene_k_agreement_suite: item 4, blocked by GO-5 NO_GO
# --------------------------------------------------------------------------- #


def _blocked_graphene_k_agreement_suite() -> dict[str, Any]:
    go5 = ck.go5_decision()
    reason = f"blocked by {go5['go5']}: {go5['blocked_by']} ({go5['blocking_gate']})"
    return {
        "status": "NOT_RUN",
        "checks": [
            check(
                "route_agrees_with_q_a_on_real_graphene_k", True, reason, applicable=False,
                requirement="S33 additional_tests_required_before_go8b item 4",
            )
        ],
        "blocked_by": go5["go5"],
        "go5_decision": go5,
    }


# --------------------------------------------------------------------------- #
# The gate
# --------------------------------------------------------------------------- #


def certify_qneq0_matbg_response_go8b() -> dict[str, Any]:
    """The GO-8b / NO-GO-8b certification, as data. Never fabricates agreement."""
    route_mechanics = _certify_route_mechanics()
    graphene_k_agreement = _blocked_graphene_k_agreement_suite()

    go8b = route_mechanics["status"] == "PASS" and graphene_k_agreement["status"] == "PASS"
    # S33's own decision is the single source of truth for which route is
    # "selected" -- it is None while GO-5 is NO_GO. Hardcoding "Q-C" here
    # (this module's own mechanics happen to be built against Q-C's kernel,
    # which is a real, separate fact) previously kept describing Q-C as an
    # already-selected primary route even after S33 itself was corrected to
    # report no route selected, which an audit correctly read as the DAG
    # being violated by this module, not just worded loosely.
    route_decision = prototypes.production_route_decision()
    return {
        "schema": SCHEMA,
        "ticket": TICKET,
        "memo": MEMO,
        "gate": GATE,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "primary_route": route_decision.get("primary_route"),
        "validation_only_route": route_decision.get("validation_only_route"),
        "route_decision_no_production_started": route_decision.get("no_production_started"),
        "route_decision_blocked_by": route_decision.get("blocked_by"),
        "kernel_mechanics_built_against_route": "Q-C",
        "route_decision_memo": "docs/epc_s33_ruta_qneq0_produccion.md",
        "route_mechanics_suite": route_mechanics,
        "graphene_k_agreement_suite": graphene_k_agreement,
        "go8b_status": "GO-8b" if go8b else "NO-GO-8b",
        "matbg_qneq0_production_authorized": bool(go8b),
    }


def require_go8b_certified_qneq0_response(certification: Mapping[str, Any]) -> Mapping[str, Any]:
    """Gate for a MATBG q != 0 production consumer: a NO-GO-8b stops here."""
    if certification.get("gate") != GATE or certification.get("go8b_status") != "GO-8b":
        raise MatbgQneq0ResponseCertificationError(
            f"GO-8b is not certified ({certification.get('go8b_status')!r}): no MATBG q != 0 "
            "response may be produced as physical (roadmap Fase 10aB, "
            "'ninguna reutilizacion del JVP Gamma como sustituto silencioso de una perturbacion "
            "q != 0', and no unverified Q-C/Q-B agreement may be assumed)"
        )
    return certification


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, Path):
        return str(value)
    return value


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    certification = certify_qneq0_matbg_response_go8b()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / REPORT_NAME
    output.write_text(json.dumps(_json_safe(certification), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"S40 MATBG q!=0 response {GATE} certification: {certification['go8b_status']}  ->  {output}")
    for suite_name in ("route_mechanics_suite", "graphene_k_agreement_suite"):
        suite = certification[suite_name]
        print(f"  {suite_name}: {suite['status']}")
        for entry in suite["checks"]:
            state = "n/a " if not entry["applicable"] else ("ok  " if entry["passed"] else "FAIL")
            print(f"    [{state}] {entry['check']}: {entry['detail']}")
    return 0 if certification["go8b_status"] == "GO-8b" else 1


if __name__ == "__main__":
    raise SystemExit(main())
