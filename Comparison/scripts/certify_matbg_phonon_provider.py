#!/usr/bin/env python3
"""E-F_001-S39: GO-8a certification for the scalable MATBG ``PhononProvider``.

S38 (:mod:`decide_matbg_phonon_provider`) answers *which* candidate; this
ticket is the certification a candidate's modes on the rigid ``(31, 30)``
target would have to clear before they may be published as a
``physical_phonon`` artifact rather than a ``synthetic_test_displacement``
(roadmap Fase 10a). The suite run here is exactly the one S38 Section 8 /
``benchmark_suite_required_before_go8a`` pre-registered -- nothing invented
after the fact:

* ``small_system`` runs today, against real archived SIESTA FC runs (graphene
  Gamma, AB bilayer) that already exist, with the exact thresholds
  ``tests/test_phonon_provider.py`` already certifies and the exact GO-7
  adjudication ``evaluate_ab_bilayer_epc.adjudicate`` already computes -- no
  new numbers, per S38 Section 8's own instruction;
* ``matbg_scale`` can only run against a real ``PhononProvider`` instance for
  ``materials/twisted_bilayer_graphene_1p084549deg/RUN.fdf``. S38's live
  decision is ``NO-GO-8a`` (no candidate is defensible,
  ``selected_provider`` is ``None``), so this suite is not fabricated: it is
  reported ``NOT_RUN``, with S38's own blocking reasons carried through
  verbatim, so the limitation stays visible to every downstream consumer
  instead of silently disappearing.

:func:`certify_matbg_phonon_provider_go8a` takes an optional ``provider``
argument precisely so that a future ticket which reopens S38 (its own
Section 9: an installed interlayer registry potential, a validated MLIP
checkpoint, or a continuum moire-phonon implementation) has this
certification ready to run against it, rather than needing another module
written from scratch. Calling it with no provider -- what this repository
does today -- returns ``NO-GO-8a``.

:func:`require_go8a_certified_matbg_phonons` is the consumer-side gate, the
MATBG analogue of ``phonon_provider.require_physical_phonon``: any future
D08A/D08B selected-mode campaign step must call it before treating a MATBG
mode as physical, and it raises rather than let a ``NO-GO-8a`` certification
be read as a pass (roadmap Fase 10a: "No se autoriza una physical
selected-mode EPC production campaign antes de GO-8a").
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for _path in (SCRIPT_DIR, REPO_ROOT / "shared"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import decide_matbg_phonon_provider as decide  # noqa: E402
import evaluate_ab_bilayer_epc as ab_adjudicate  # noqa: E402
import phonon_provider as pp  # noqa: E402
from artifact_signature import PHYSICAL_PHONON, SYNTHETIC_TEST_DISPLACEMENT  # noqa: E402
from reference_provenance import geometry_cell_species_sha256  # noqa: E402

SCHEMA = "matbg_phonon_provider_go8a_certification_v1"
TICKET = "E-F_001-S39"
MEMO = "docs/epc_s39_matbg_phonon_provider_certification.md"
GATE = "GO-8a"

TARGET_GEOMETRY_FDF = REPO_ROOT / decide.TARGET_GEOMETRY
TARGET_ATOM_COUNT = decide.TARGET_ATOM_COUNT

GRAPHENE_FC_RUN = REPO_ROOT / "Comparison/results/epc/siesta_reference/graphene/runs/fc_dhs_d0p01"
AB_BILAYER_REPORT = ab_adjudicate.DEFAULT_REPORT_PATH

# ponytail: roadmap Section X's own operational headroom (also used by
# benchmark_matbg_synthetic_displacements.py). Duplicated as two floats rather
# than imported, so this module -- which must import cleanly with no provider
# present at all -- does not pull in that script's torch/solver dependency
# chain just to read two constants.
RAM_MARGIN_GIB = 62.0
VRAM_MARGIN_GIB = 28.0

_EPS = float(np.finfo(np.float64).eps)

DEFAULT_OUTPUT_DIR = REPO_ROOT / "Comparison/results/epc/certification/matbg"
REPORT_NAME = "s39_matbg_phonon_provider_go8a_certification.json"


class MatbgPhononCertificationError(RuntimeError):
    """A GO-8a certification was built, read, or trusted in a way this gate forbids."""


def check(name: str, passed: bool, detail: str, *, applicable: bool = True, **extra: Any) -> dict[str, Any]:
    return {
        "check": name,
        "passed": bool(passed) if applicable else True,
        "applicable": bool(applicable),
        "detail": detail,
        **extra,
    }


def _rss_bytes() -> int:
    # ponytail: ru_maxrss is a high-water mark, not a delta -- same caveat as
    # benchmark_matbg_synthetic_displacements.rss_bytes / run_epc_siesta_reference.py.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


# --------------------------------------------------------------------------- #
# small_system suite: runs today, against archived FC runs already on disk.
# --------------------------------------------------------------------------- #


def _certify_graphene_gamma() -> list[dict[str, Any]]:
    reference = "tests/test_phonon_provider.py::test_gamma_frequencies_reproduce_graphene"
    if not GRAPHENE_FC_RUN.is_dir():
        return [
            check(
                "graphene_gamma_acoustic_and_optical", True,
                f"no archived FC run at {GRAPHENE_FC_RUN}", applicable=False, reference=reference,
            ),
            check(
                "acoustic_sum_rule_raw_vs_corrected", True,
                f"no archived FC run at {GRAPHENE_FC_RUN}", applicable=False,
            ),
        ]
    raw_modes = pp.SiestaFcPhononProvider.from_run_dir(GRAPHENE_FC_RUN, apply_asr=False).modes()
    corrected_modes = pp.SiestaFcPhononProvider.from_run_dir(GRAPHENE_FC_RUN, apply_asr=True).modes()
    freq_cm1 = corrected_modes.frequencies_ev / pp.CM1_TO_EV
    acoustic_ok = bool(np.abs(corrected_modes.frequencies_ev[:3]).max() < 1e-6)
    zo_ok = bool(corrected_modes.mode_count > 3 and 800.0 < freq_cm1[3] < 900.0)
    e2g_ok = bool(corrected_modes.mode_count > 5 and 1500.0 < freq_cm1[5] < 1620.0)
    e2g_split = (
        abs(freq_cm1[4] - freq_cm1[5]) / max(abs(freq_cm1[4]), abs(freq_cm1[5]))
        if corrected_modes.mode_count > 5 else float("nan")
    )
    passed = (
        corrected_modes.mode_count == 6 and corrected_modes.asr_policy == pp.ASR_CORRECTED
        and acoustic_ok and zo_ok and e2g_ok and e2g_split < 1e-4
    )
    gamma_check = check(
        "graphene_gamma_acoustic_and_optical", passed,
        f"3 acoustic branches |omega| < 1e-6 eV after ASR, ZO {freq_cm1[3] if corrected_modes.mode_count > 3 else float('nan'):.1f} cm^-1 "
        f"in (800, 900), E2g {freq_cm1[5] if corrected_modes.mode_count > 5 else float('nan'):.1f} cm^-1 in (1500, 1620) "
        f"split {e2g_split:.2e} < 1e-4",
        reference=reference, frequencies_cm1=freq_cm1.tolist(), e2g_relative_split=e2g_split,
    )
    asr_check = check(
        "acoustic_sum_rule_raw_vs_corrected",
        bool(raw_modes.asr_residual_ev_ang2 > 1e-3 and corrected_modes.asr_residual_ev_ang2 < 1e-10),
        f"raw residual {raw_modes.asr_residual_ev_ang2:.3e} eV/Ang^2 (measurably nonzero), corrected "
        f"{corrected_modes.asr_residual_ev_ang2:.3e} (roundoff); neither number replaces the other "
        f"(phonon_provider.ASR_RAW/ASR_CORRECTED split)",
        raw_residual_ev_ang2=raw_modes.asr_residual_ev_ang2,
        corrected_residual_ev_ang2=corrected_modes.asr_residual_ev_ang2,
    )
    return [gamma_check, asr_check]


def _certify_ab_bilayer_sectors() -> dict[str, Any]:
    reference = (
        "Comparison/scripts/run_ab_bilayer_epc_paths.py sectors 'shear'/'layer_breathing', "
        "evaluate_ab_bilayer_epc.check_shear_direction/check_breathing_parity"
    )
    if not AB_BILAYER_REPORT.is_file():
        return check(
            "ab_bilayer_interlayer_sectors", True,
            f"no persisted GO-7 report at {AB_BILAYER_REPORT}", applicable=False, reference=reference,
        )
    report = json.loads(AB_BILAYER_REPORT.read_text(encoding="utf-8"))
    try:
        adjudication = ab_adjudicate.adjudicate(report)
    except ab_adjudicate.Go7AdjudicationError as error:
        return check("ab_bilayer_interlayer_sectors", False, f"GO-7 adjudication failed: {error}", reference=reference)
    by_name = {row["check"]: row for row in adjudication["checks"]}
    shear = by_name.get("shear_direction_in_plane_antiparallel", {"passed": False})
    breathing = by_name.get("breathing_parity_layers_equal_and_opposite", {"passed": False})
    passed = bool(shear.get("passed") and breathing.get("passed"))
    return check(
        "ab_bilayer_interlayer_sectors", passed,
        f"the candidate's own eigenvectors (not a synthetic direction) show shear "
        f"{'PASS' if shear.get('passed') else 'FAIL'} and breathing "
        f"{'PASS' if breathing.get('passed') else 'FAIL'}; GO-7 verdict {adjudication['verdict']}",
        reference=reference, go7_verdict=adjudication["verdict"],
        shear_direction=shear, breathing_parity=breathing,
    )


def _certify_small_system_suite() -> dict[str, Any]:
    checks = _certify_graphene_gamma() + [_certify_ab_bilayer_sectors()]
    applicable = [entry for entry in checks if entry["applicable"]]
    if not applicable:
        status = "NOT_VERIFIED"
    elif all(entry["passed"] for entry in applicable):
        status = "PASS"
    else:
        status = "FAIL"
    return {"status": status, "checks": checks}


# --------------------------------------------------------------------------- #
# matbg_scale suite: only runs against a real PhononProvider on the target.
# --------------------------------------------------------------------------- #


def _blocked_matbg_scale_suite(decision: Mapping[str, Any]) -> dict[str, Any]:
    """No candidate is defensible: every matbg_scale check is refused, not faked."""
    required = decision["benchmark_suite_required_before_go8a"]["matbg_scale"]
    reason = (
        f"blocked by {decision['verdict']}: {decision['reopen_if']}"
    )
    checks = [
        check(name, True, reason, applicable=False, requirement=requirement)
        for name, requirement in required.items()
    ]
    return {
        "status": "NOT_RUN",
        "checks": checks,
        "blocked_by": decision["verdict"],
        "selected_provider": decision["selected_provider"],
        "candidates_considered": [candidate["id"] for candidate in decision["candidates"]],
        "reopen_if": decision["reopen_if"],
    }


def _certify_matbg_scale_suite(
    provider: Any, *, target_fdf: Path, q_fractional: Sequence[float]
) -> dict[str, Any]:
    """Run the full S38 Section 8 matbg_scale suite against a real provider."""
    expected_signature = geometry_cell_species_sha256(target_fdf)
    rss_before = _rss_bytes()
    started = time.perf_counter()
    mode_set = pp.require_physical_phonon(provider.modes(q_fractional))
    wall_seconds = time.perf_counter() - started
    peak_rss_gib = max(0, _rss_bytes() - rss_before) / 1024**3

    atom_count = int(mode_set.positions_ang.shape[0])
    checks = [
        check(
            "geometry_signature_match",
            mode_set.geometry_signature == expected_signature and atom_count == TARGET_ATOM_COUNT,
            f"provider geometry signature {mode_set.geometry_signature!r} vs target "
            f"{expected_signature!r}; {atom_count} atoms vs target {TARGET_ATOM_COUNT}",
            geometry_signature=mode_set.geometry_signature,
            expected_geometry_signature=expected_signature,
            atom_count=atom_count,
        )
    ]

    hermiticity = float((mode_set.provenance or {}).get("dynamical_matrix_hermiticity_ev_ang2_amu", float("nan")))
    # The dynamical matrix is symmetrised as 0.5*(D + D^dagger) before
    # diagonalisation (phonon_provider.modes_from_force_constants): its
    # asymmetry is bounded by float64 roundoff of that construction, at the
    # matrix's own scale (lambda ~ omega^2 / const) -- the candidate's own
    # noise floor, not a hand-picked constant (S38 Section 8).
    scale = float(np.max(mode_set.frequencies_ev**2)) / pp.ZERO_POINT_CONSTANT_ANG2_AMU_EV if mode_set.mode_count else 0.0
    hermiticity_tolerance = 64.0 * _EPS * max(scale, 1e-12)
    checks.append(
        check(
            "dynamical_matrix_hermiticity",
            hermiticity <= hermiticity_tolerance,
            f"max|D(q) - D(q)^dagger| = {hermiticity:.3e} <= {hermiticity_tolerance:.3e} eV/(Ang^2 amu)",
            hermiticity_ev_ang2_amu=hermiticity, tolerance=hermiticity_tolerance,
        )
    )

    sector = pp.mode_sector_report(mode_set)
    acoustic_frequencies = [sector["branches"][index]["frequency_ev"] for index in sector["acoustic_branches"]]
    checks.append(
        check(
            "acoustic_branches_vanish_at_gamma_after_asr",
            bool(
                sector["identification"]["acoustic_unambiguous"]
                and mode_set.asr_policy == pp.ASR_CORRECTED
                and max((abs(value) for value in acoustic_frequencies), default=1.0) < 1e-6
            ),
            f"{len(acoustic_frequencies)} acoustic branch(es) identified, max |omega| = "
            f"{max((abs(value) for value in acoustic_frequencies), default=float('nan')):.3e} eV after ASR",
            acoustic_branches=sector["acoustic_branches"], acoustic_frequencies_ev=acoustic_frequencies,
        )
    )

    checks.append(
        check(
            "resource_preflight",
            bool(peak_rss_gib <= RAM_MARGIN_GIB),
            f"wall {wall_seconds:.2f}s, peak RSS delta {peak_rss_gib:.2f} GiB <= {RAM_MARGIN_GIB} GiB margin "
            "(roadmap Section X; no VRAM measured here -- a q=Gamma dynamical-matrix diagonalisation "
            "is CPU-only, per tbg_tight_binding_reference)",
            wall_seconds=wall_seconds, peak_rss_delta_gib=peak_rss_gib, ram_margin_gib=RAM_MARGIN_GIB,
        )
    )

    fields = mode_set.physical_phonon_fields()
    provenance_ok = all(
        bool(fields.get(name)) for name in ("provider", "provider_version", "force_source", "primitive_mapping", "fc_range", "masses")
    ) and bool((fields.get("asr_policy") or {}).get("policy"))
    checks.append(
        check(
            "provenance",
            provenance_ok,
            "geometry + provider/version + force/IFC source + primitive mapping + FC range + q + "
            "eigvec hash + masses + normalization + ASR policy (roadmap Section V 'Phonon' firma)",
            fields={name: fields.get(name) for name in ("provider", "provider_version", "force_source", "primitive_mapping", "fc_range", "asr_policy")},
        )
    )

    passed = all(entry["passed"] for entry in checks)
    return {
        "status": "PASS" if passed else "FAIL",
        "checks": checks,
        "mode_set_dict": mode_set.to_dict() if passed else None,
    }


# --------------------------------------------------------------------------- #
# The gate
# --------------------------------------------------------------------------- #


def certify_matbg_phonon_provider_go8a(
    provider: Any | None = None,
    *,
    target_fdf: Path = TARGET_GEOMETRY_FDF,
    q_fractional: Sequence[float] = (0.0, 0.0, 0.0),
) -> dict[str, Any]:
    """The GO-8a / NO-GO-8a certification, as data. Never fabricates a MATBG mode."""
    decision = decide.matbg_phonon_provider_decision()
    small_system = _certify_small_system_suite()
    matbg_scale = (
        _certify_matbg_scale_suite(provider, target_fdf=target_fdf, q_fractional=q_fractional)
        if provider is not None
        else _blocked_matbg_scale_suite(decision)
    )

    go8a = small_system["status"] == "PASS" and matbg_scale["status"] == "PASS"
    physical_phonon = matbg_scale.get("mode_set_dict") if go8a else None
    return {
        "schema": SCHEMA,
        "ticket": TICKET,
        "memo": MEMO,
        "gate": GATE,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target_geometry": str(target_fdf),
        "target_atom_count": TARGET_ATOM_COUNT,
        "decision": decision,
        "small_system_suite": small_system,
        "matbg_scale_suite": matbg_scale,
        "go8a_status": "GO-8a" if go8a else "NO-GO-8a",
        "physical_phonon_produced": bool(go8a),
        "physical_phonon": physical_phonon,
        "artifact_kind": PHYSICAL_PHONON if go8a else None,
        "displacement_class_policy": (
            f"a matbg_scale PASS is required before any mode from this target geometry may be "
            f"tagged {PHYSICAL_PHONON!r}; anything produced before that stays "
            f"{SYNTHETIC_TEST_DISPLACEMENT!r} per roadmap B7"
        ),
    }


def require_go8a_certified_matbg_phonons(certification: Mapping[str, Any]) -> dict[str, Any]:
    """Gate for a MATBG selected-mode consumer: a NO-GO-8a certification stops here."""
    if certification.get("gate") != GATE or certification.get("go8a_status") != "GO-8a" or not certification.get("physical_phonon_produced"):
        raise MatbgPhononCertificationError(
            f"GO-8a is not certified ({certification.get('go8a_status')!r}) for "
            f"{certification.get('target_geometry')!r}: no MATBG mode may be used as a physical "
            "phonon (roadmap Fase 10a, 'No se autoriza una physical selected-mode EPC production "
            "campaign antes de GO-8a')"
        )
    physical_phonon = certification.get("physical_phonon")
    if not physical_phonon or physical_phonon.get("artifact_kind") != PHYSICAL_PHONON:
        raise MatbgPhononCertificationError(
            "certification is marked GO-8a but carries no physical_phonon artifact"
        )
    return physical_phonon


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
    certification = certify_matbg_phonon_provider_go8a()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / REPORT_NAME
    output.write_text(json.dumps(_json_safe(certification), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"S39 MATBG PhononProvider {GATE} certification: {certification['go8a_status']}  ->  {output}")
    for suite_name in ("small_system_suite", "matbg_scale_suite"):
        suite = certification[suite_name]
        print(f"  {suite_name}: {suite['status']}")
        for entry in suite["checks"]:
            state = "n/a " if not entry["applicable"] else ("ok  " if entry["passed"] else "FAIL")
            print(f"    [{state}] {entry['check']}: {entry['detail']}")
    return 0 if certification["go8a_status"] == "GO-8a" else 1


if __name__ == "__main__":
    raise SystemExit(main())
