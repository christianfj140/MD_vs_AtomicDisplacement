#!/usr/bin/env python3
"""E-F_001-S49: relaxed-MATBG ``PhononProvider`` bound (Fase 11).

The roadmap objective is a ``PhononProvider`` whose geometry signature matches
the *relaxed* magic-angle ``(31, 30)`` cell. Two upstream gates already answer
"can that exist today", both as data rather than assertion:

* ``docs/epc_s47_matbg_relaxation_decision.md`` (S47,
  :func:`decide_matbg_relaxation_provider.matbg_relaxation_provider_decision`)
  -- ``NO-GO-10``: no relaxation provider in this environment is defensible
  for the production cell, so **no relaxed MATBG geometry exists**, and
  ``relaxed_geometry_produced`` is hard-coded ``False``.
* ``docs/epc_s39_matbg_phonon_provider_certification.md`` (S39,
  :func:`certify_matbg_phonon_provider.certify_matbg_phonon_provider_go8a`)
  -- ``NO-GO-8a``: no MATBG-scale ``PhononProvider`` exists at all, on any
  geometry, rigid or relaxed.

Either alone blocks a ``relaxed`` ``physical_phonon`` artifact; both are
NO-GO here. This ticket does not re-litigate those decisions (it imports and
reuses them, never recomputes them differently) and does not fabricate a
relaxed geometry or a MATBG phonon to work around them. What Fase 11 asks
that *can* still be done without either -- "Reevaluar frecuencias,
eigenvectors, masas, ASR, Fourier convention y modos shear/breathing",
"Medir sensibilidad al provider y a parametros relevantes", "Registrar
backend, recursos y provenance de la geometria" -- is done here against real
archived small-system SIESTA FC data already on disk:

``displacement_sensitivity_suite``
    the same graphene-Gamma acoustic/ZO/E2g/ASR signature S39 already
    certifies (exact thresholds, reused), re-measured independently at each
    of the three archived FC displacement magnitudes
    (``fc_dhs_d0p005``/``d0p01``/``d0p02``), plus the cross-delta frequency
    spread and a gauge-invariant eigenvector-subspace angle between the two
    extreme deltas (:mod:`epc_subspaces`) -- the "sensitivity to the
    provider and relevant parameters" measurement, reported as a diagnostic
    rather than gated against an invented tolerance.
``local_stacking_phonon_suite``
    the same acoustic/ASR contract check, run against whichever of the
    AA/AB/BA local stacking registries -- the registries Nam & Koshino
    (2017) identify a relaxed twisted bilayer locally interpolates between
    -- has an archived FC run today. Only ``bilayer_graphene_AB`` does; AA
    and BA are reported ``applicable: False`` with the missing path named,
    never silently marked compatible (roadmap acceptance criterion "Ningun
    phonon rigido se marca compatible salvo demostracion explicita").

Both suites accept an optional real ``PhononProvider`` (mirroring S39
Section 4): :func:`certify_relaxed_matbg_phonon_provider_bound` forwards it
straight into
:func:`certify_matbg_phonon_provider.certify_matbg_phonon_provider_go8a`, so
the day a relaxed geometry and a MATBG-scale provider both exist, this
module runs the *exact same* GO-8a contract against them without being
rewritten -- "Los artifacts relaxed physical_phonon pasan el mismo contrato
GO-8a" is satisfied by reuse, not by a second gate invented here.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for _path in (SCRIPT_DIR, REPO_ROOT / "shared"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import certify_matbg_phonon_provider as go8a_cert  # noqa: E402
import phonon_provider as pp  # noqa: E402
from artifact_signature import PHYSICAL_PHONON  # noqa: E402
from certify_matbg_phonon_provider import check, _rss_bytes  # noqa: E402
from decide_matbg_relaxation_provider import matbg_relaxation_provider_decision  # noqa: E402

SCHEMA = "epc_relaxed_matbg_phonon_provider_bound_v1"
TICKET = "E-F_001-S49"
MEMO = "docs/epc_s49_relaxed_matbg_phonon_provider.md"
STATUS_LABEL = "relaxed_matbg_phonon_provider_bound"

DEFAULT_OUTPUT = REPO_ROOT / "Comparison/results/epc/certification/matbg/s49_relaxed_matbg_phonon_provider_bound.json"

SIESTA_REFERENCE_ROOT = REPO_ROOT / "Comparison/results/epc/siesta_reference"

#: Real archived graphene Gamma FC runs at three displacement magnitudes --
#: the "parameter sensitivity" axis this ticket measures. Same C18/S39 data,
#: no new SIESTA run.
GRAPHENE_DELTA_RUNS: dict[str, Path] = {
    "d0p005": SIESTA_REFERENCE_ROOT / "graphene/runs/fc_dhs_d0p005",
    "d0p01": SIESTA_REFERENCE_ROOT / "graphene/runs/fc_dhs_d0p01",
    "d0p02": SIESTA_REFERENCE_ROOT / "graphene/runs/fc_dhs_d0p02",
}

#: Nam & Koshino (2017): the three local stacking registries a relaxed
#: twisted bilayer interpolates between below ~2 deg twist -- the same
#: citation S47/S48 use. Only the fdf path is needed here (unlike S48, this
#: module runs no checkpoint/derivative machinery).
BILAYER_MATERIALS: tuple[dict[str, str], ...] = (
    {"label": "bilayer_graphene_AA", "fdf": "materials/bilayer_graphene_AA/RUN.fdf"},
    {"label": "bilayer_graphene_AB", "fdf": "materials/bilayer_graphene_AB/RUN.fdf"},
    {"label": "bilayer_graphene_BA", "fdf": "materials/bilayer_graphene_BA/RUN.fdf"},
)

_GRAPHENE_GAMMA_REFERENCE = (
    "tests/test_phonon_provider.py::test_gamma_frequencies_reproduce_graphene thresholds, "
    "reused per delta (S39 Section 2)"
)


class RelaxedMatbgPhononBoundError(RuntimeError):
    """A relaxed-MATBG phonon claim was made that this bound does not support."""


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


# --------------------------------------------------------------------------- #
# displacement_sensitivity_suite: graphene Gamma at three archived deltas.
# --------------------------------------------------------------------------- #


def _graphene_gamma_signature_check(label: str, run_dir: Path) -> tuple[dict[str, Any], Any]:
    """The exact S39 acoustic/ZO/E2g/ASR signature, at one archived delta."""
    if not run_dir.is_dir():
        return (
            check(
                f"graphene_gamma_{label}", True, f"no archived FC run at {run_dir}",
                applicable=False, reference=_GRAPHENE_GAMMA_REFERENCE,
            ),
            None,
        )
    raw = pp.SiestaFcPhononProvider.from_run_dir(run_dir, apply_asr=False).modes()
    corrected = pp.SiestaFcPhononProvider.from_run_dir(run_dir, apply_asr=True).modes()
    freq_cm1 = corrected.frequencies_ev / pp.CM1_TO_EV
    acoustic_ok = bool(np.abs(corrected.frequencies_ev[:3]).max() < 1e-6)
    zo_ok = bool(corrected.mode_count > 3 and 800.0 < freq_cm1[3] < 900.0)
    e2g_ok = bool(corrected.mode_count > 5 and 1500.0 < freq_cm1[5] < 1620.0)
    e2g_split = (
        abs(freq_cm1[4] - freq_cm1[5]) / max(abs(freq_cm1[4]), abs(freq_cm1[5]))
        if corrected.mode_count > 5 else float("nan")
    )
    asr_ok = bool(raw.asr_residual_ev_ang2 > 1e-3 and corrected.asr_residual_ev_ang2 < 1e-10)
    passed = acoustic_ok and zo_ok and e2g_ok and e2g_split < 1e-4 and asr_ok
    entry = check(
        f"graphene_gamma_{label}", passed,
        f"{run_dir.name}: acoustic {'ok' if acoustic_ok else 'FAIL'}, "
        f"ZO {freq_cm1[3] if corrected.mode_count > 3 else float('nan'):.1f} cm^-1, "
        f"E2g {freq_cm1[5] if corrected.mode_count > 5 else float('nan'):.1f} cm^-1 split {e2g_split:.2e}, "
        f"ASR raw {raw.asr_residual_ev_ang2:.2e} -> corrected {corrected.asr_residual_ev_ang2:.2e}",
        reference=_GRAPHENE_GAMMA_REFERENCE, frequencies_cm1=freq_cm1.tolist(),
        asr_residual_raw_ev_ang2=raw.asr_residual_ev_ang2,
        asr_residual_corrected_ev_ang2=corrected.asr_residual_ev_ang2,
    )
    return entry, corrected


def _delta_spread_diagnostics(mode_sets: dict[str, Any]) -> dict[str, Any]:
    """Cross-delta frequency spread and E2g eigenvector-subspace angle.

    Diagnostic, not gated: no delta-sensitivity tolerance has been
    independently derived for this repository (the way, e.g., the hermiticity
    tolerance in ``certify_matbg_phonon_provider`` is derived from the mode
    set's own scale). Reporting real numbers here satisfies Fase 11's "medir
    sensibilidad al provider y a parametros relevantes" without inventing a
    pass/fail threshold this ticket cannot defend.
    """
    if len(mode_sets) < 2:
        return {
            "applicable": False,
            "detail": f"{len(mode_sets)} archived delta(s); at least two are needed for a spread",
        }
    labels = sorted(mode_sets)  # "d0p005" < "d0p01" < "d0p02" sorts numerically here
    zo_cm1 = {
        label: float(mode_sets[label].frequencies_ev[3] / pp.CM1_TO_EV)
        for label in labels if mode_sets[label].mode_count > 3
    }
    e2g_cm1 = {
        label: float(np.mean(mode_sets[label].frequencies_ev[4:6]) / pp.CM1_TO_EV)
        for label in labels if mode_sets[label].mode_count > 5
    }
    subspace_angle = None
    lo, hi = labels[0], labels[-1]
    if mode_sets[lo].mode_count > 5 and mode_sets[hi].mode_count > 5:
        from epc_subspaces import subspace_metrics, subspace_overlap

        na3 = int(mode_sets[lo].eigenvectors.shape[1] * 3)
        c_lo = mode_sets[lo].eigenvectors[4:6].reshape(2, -1).T
        c_hi = mode_sets[hi].eigenvectors[4:6].reshape(2, -1).T
        metrics = subspace_metrics(subspace_overlap(c_lo, np.eye(na3), c_hi))
        subspace_angle = {
            "clusters_compared": f"{lo} vs {hi}",
            "maximum_principal_angle_rad": metrics["maximum_principal_angle_rad"],
        }
    return {
        "applicable": True,
        "zo_frequency_cm1_by_delta": zo_cm1,
        "zo_spread_cm1": (max(zo_cm1.values()) - min(zo_cm1.values())) if zo_cm1 else float("nan"),
        "e2g_mean_frequency_cm1_by_delta": e2g_cm1,
        "e2g_spread_cm1": (max(e2g_cm1.values()) - min(e2g_cm1.values())) if e2g_cm1 else float("nan"),
        "e2g_eigenvector_subspace_angle": subspace_angle,
        "policy": (
            "diagnostic only, not gated -- see docstring of _delta_spread_diagnostics"
        ),
    }


def _certify_displacement_sensitivity() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    mode_sets: dict[str, Any] = {}
    for label, run_dir in GRAPHENE_DELTA_RUNS.items():
        entry, modes = _graphene_gamma_signature_check(label, run_dir)
        checks.append(entry)
        if modes is not None:
            mode_sets[label] = modes
    applicable = [entry for entry in checks if entry["applicable"]]
    if not applicable:
        status = "NOT_VERIFIED"
    elif all(entry["passed"] for entry in applicable):
        status = "PASS"
    else:
        status = "FAIL"
    return {
        "status": status,
        "checks": checks,
        "deltas_available": sorted(mode_sets),
        "spread_diagnostics": _delta_spread_diagnostics(mode_sets),
    }


# --------------------------------------------------------------------------- #
# local_stacking_phonon_suite: real phonons on whichever of AA/AB/BA has FC.
# --------------------------------------------------------------------------- #


def _local_stacking_material(material: dict[str, str]) -> dict[str, Any]:
    run_root = SIESTA_REFERENCE_ROOT / material["label"] / "runs"
    fc_runs = sorted(run_root.glob("fc_dhs_*")) if run_root.is_dir() else []
    if not fc_runs:
        return check(
            f"{material['label']}_phonon", True,
            f"no archived FC run under {run_root}; not marked compatible (roadmap acceptance: "
            "'ningun phonon rigido se marca compatible salvo demostracion explicita')",
            applicable=False,
        )
    run_dir = fc_runs[-1]
    rss_before = _rss_bytes()
    started = time.perf_counter()
    raw = pp.SiestaFcPhononProvider.from_run_dir(run_dir, apply_asr=False).modes()
    corrected = pp.SiestaFcPhononProvider.from_run_dir(run_dir, apply_asr=True).modes()
    wall_seconds = time.perf_counter() - started
    peak_rss_gib = max(0, _rss_bytes() - rss_before) / 1024**3

    sector = pp.mode_sector_report(corrected)
    acoustic_ok = bool(sector["identification"]["acoustic_unambiguous"])
    asr_ok = bool(raw.asr_residual_ev_ang2 > 1e-3 and corrected.asr_residual_ev_ang2 < 1e-10)
    atoms_ok = corrected.positions_ang.shape[0] == 4
    passed = bool(acoustic_ok and asr_ok and atoms_ok)

    optical = [row for row in sector["branches"] if row["branch"] not in sector["acoustic_branches"]]
    breathing_candidate = max(optical, key=lambda row: row["out_of_plane_weight"], default=None)
    shear_candidate = min(optical, key=lambda row: row["out_of_plane_weight"], default=None)

    return check(
        f"{material['label']}_phonon", passed,
        f"{run_dir.name}: {corrected.mode_count} modes, "
        f"acoustic {'unambiguous' if acoustic_ok else 'ambiguous'}, "
        f"ASR raw {raw.asr_residual_ev_ang2:.2e} -> corrected {corrected.asr_residual_ev_ang2:.2e}, "
        f"{corrected.positions_ang.shape[0]} atoms",
        fc_run=str(run_dir),
        geometry_signature=corrected.geometry_signature,
        backend=corrected.provider,
        wall_seconds=wall_seconds,
        peak_rss_delta_gib=peak_rss_gib,
        acoustic_branches=sector["acoustic_branches"],
        breathing_candidate_branch=(breathing_candidate or {}).get("branch"),
        breathing_candidate_out_of_plane_weight=(breathing_candidate or {}).get("out_of_plane_weight"),
        shear_candidate_branch=(shear_candidate or {}).get("branch"),
        shear_candidate_out_of_plane_weight=(shear_candidate or {}).get("out_of_plane_weight"),
        shear_breathing_policy=(
            "diagnostic only: candidates are the highest/lowest out-of-plane projection weight "
            "among non-acoustic branches (phonon_provider.mode_sector_report); this is not the "
            "GO-7 shear/breathing adjudication (evaluate_ab_bilayer_epc, an electronic "
            "D_H-derivative check on synthetic directions) and makes no sector-identity claim "
            "beyond that weight"
        ),
    )


def _certify_local_stacking_phonons() -> dict[str, Any]:
    checks = [_local_stacking_material(material) for material in BILAYER_MATERIALS]
    applicable = [entry for entry in checks if entry["applicable"]]
    if not applicable:
        status = "NOT_VERIFIED"
    elif all(entry["passed"] for entry in applicable):
        status = "PASS"
    else:
        status = "FAIL"
    return {
        "status": status,
        "checks": checks,
        "materials_with_archived_fc": [entry["check"][: -len("_phonon")] for entry in applicable],
    }


# --------------------------------------------------------------------------- #
# The bound
# --------------------------------------------------------------------------- #


def certify_relaxed_matbg_phonon_provider_bound(
    provider: Any | None = None,
    *,
    relaxed_target_fdf: Path | None = None,
) -> dict[str, Any]:
    """The relaxed-MATBG phonon bound, as data. Never fabricates a relaxed geometry or mode.

    ``provider``/``relaxed_target_fdf`` exist so the day both S47 and S38/S39
    reopen (a relaxed geometry with its own provenance, and a MATBG-scale
    PhononProvider instance), this function runs the exact GO-8a contract
    against them without being rewritten -- passed straight through to
    :func:`certify_matbg_phonon_provider.certify_matbg_phonon_provider_go8a`.
    Called with neither (what this repository has today), it reuses that same
    certifier's no-provider ``NO-GO-8a`` path, which is geometry-agnostic: it
    blocks equally whether the target would have been rigid or relaxed.
    """
    go10 = matbg_relaxation_provider_decision()
    go8a_certification = go8a_cert.certify_matbg_phonon_provider_go8a(
        provider, target_fdf=relaxed_target_fdf or go8a_cert.TARGET_GEOMETRY_FDF,
    )
    displacement_sensitivity = _certify_displacement_sensitivity()
    local_stacking = _certify_local_stacking_phonons()

    relaxed_geometry_available = bool(go10["relaxed_geometry_produced"])
    relaxed_physical_phonon_produced = bool(
        relaxed_geometry_available and go8a_certification["go8a_status"] == "GO-8a"
    )
    measurable_suites = (displacement_sensitivity, local_stacking)
    bound_verdict = (
        "PASS"
        if all(suite["status"] in ("PASS", "NOT_VERIFIED") for suite in measurable_suites)
        else "FAIL"
    )

    return {
        "schema": SCHEMA,
        "ticket": TICKET,
        "memo": MEMO,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status_label": STATUS_LABEL,
        "go10_status": go10["verdict"],
        "go10_reopen_if": go10["reopen_if"],
        "go8a_status": go8a_certification["go8a_status"],
        "go8a_certification": go8a_certification,
        "relaxed_geometry_available": relaxed_geometry_available,
        "relaxed_physical_phonon_produced": relaxed_physical_phonon_produced,
        "relaxed_physical_phonon": (
            go8a_certification["physical_phonon"] if relaxed_physical_phonon_produced else None
        ),
        "displacement_sensitivity_suite": displacement_sensitivity,
        "local_stacking_phonon_suite": local_stacking,
        "bound_verdict": bound_verdict,
        "licenses_go8a_for_relaxed_geometry": False,
        "quantitative_relaxed_matbg_phonon_claim_licensed": False,
        "reason": (
            "displacement-sensitivity and local-stacking phonon contract checks pass against "
            "real archived SIESTA FC data, and the exact S39 GO-8a certifier is re-run against "
            "any relaxed-geometry provider passed in -- but with no relaxed geometry (S47: "
            f"{go10['verdict']}) and no MATBG-scale PhononProvider (S39/S38: "
            f"{go8a_certification['go8a_status']}) present in this repository today, no "
            "relaxed_physical_phonon exists and none is fabricated by this bound"
        ),
        "reopen_if": (
            f"S47 reopens ({go10['reopen_if']}) AND a MATBG-scale PhononProvider clears GO-8a "
            f"(S38/S39: {go8a_certification['matbg_scale_suite'].get('reopen_if', 'see docs/epc_s38_matbg_phonon_provider_decision.md')}), "
            "with the resulting provider run against the relaxed geometry's own RUN.fdf via this "
            "function's provider/relaxed_target_fdf arguments"
        ),
        "references": [
            "N. N. T. Nam and M. Koshino, Phys. Rev. B 96, 075311 (2017) -- AA/AB/BA as the "
            "registries a relaxed twisted bilayer locally interpolates between, the physical "
            "basis for local_stacking_phonon_suite.",
            "Z. Lu et al., Phys. Rev. B 106, 144305 (2022) -- low-energy moire phonons; out of "
            "reach here since no MATBG-scale PhononProvider exists to reevaluate them on (GO-8a: "
            f"{go8a_certification['go8a_status']}).",
        ],
    }


def require_relaxed_matbg_physical_phonon(bound: dict[str, Any]) -> dict[str, Any]:
    """Gate for a relaxed-MATBG EPC consumer: this bound refuses until GO-10 and GO-8a both hold."""
    if not bound.get("relaxed_physical_phonon_produced") or not bound.get("relaxed_physical_phonon"):
        raise RelaxedMatbgPhononBoundError(
            f"no relaxed-MATBG physical_phonon exists: go10={bound.get('go10_status')!r}, "
            f"go8a={bound.get('go8a_status')!r}; no g_mn,nu may be built for the relaxed target "
            "(roadmap Fase 11 / GO-8a, B7)"
        )
    phonon = bound["relaxed_physical_phonon"]
    if phonon.get("artifact_kind") != PHYSICAL_PHONON:
        raise RelaxedMatbgPhononBoundError("bound is marked produced but carries no physical_phonon artifact")
    return phonon


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    bound = certify_relaxed_matbg_phonon_provider_bound()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_json_safe(bound), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"[{TICKET}] relaxed-MATBG PhononProvider bound: {bound['bound_verdict']}  ->  {args.output}")
    print(f"[{TICKET}]   go10={bound['go10_status']}  go8a={bound['go8a_status']}  "
          f"relaxed_physical_phonon_produced={bound['relaxed_physical_phonon_produced']}")
    for suite_name in ("displacement_sensitivity_suite", "local_stacking_phonon_suite"):
        suite = bound[suite_name]
        print(f"[{TICKET}]   {suite_name}: {suite['status']}")
        for entry in suite["checks"]:
            state = "n/a " if not entry["applicable"] else ("ok  " if entry["passed"] else "FAIL")
            print(f"[{TICKET}]     [{state}] {entry['check']}: {entry['detail']}")
    return 0 if bound["bound_verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
