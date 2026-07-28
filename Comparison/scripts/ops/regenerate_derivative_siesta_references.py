#!/usr/bin/env python
"""Regenerate the contaminated SIESTA derivative references for cross_w90_to_5x5_2delta.

The original campaign wrote stencils that inherited the source MD block
(``MD.TypeOfRun Verlet`` / ``MD.Steps 20`` at 450 K), so SIESTA evolved the geometry
instead of doing a single-point SCF at the +-delta displacement. The stored TSHS sits
0.05-0.33 Ang away from the stencil position while delta is only 0.005-0.01 Ang, i.e.
the contamination is 10-25x the signal, which is why every derivative metric came back
with relative Frobenius > 1 and a near-zero (often negative) cosine.

``single_point=True`` in build_hamiltonian_derivative_stencils fixes new stencils, but
the existing ones still carry the MD block, so both the stencils and the references have
to be rebuilt: that needs ``overwrite: true``, which the committed payloads deliberately
do not set (they are tuned for safe resume). This script derives an overwrite payload per
case at run time rather than duplicating twelve config files.

Resumable: a case whose references already pass the geometry guard is skipped, so an
interrupted run does not redo hours of finished SCF work.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "Comparison/scripts"))
sys.path.insert(0, str(REPO_ROOT / "shared"))

from reference_selection import choose_reference_matrix  # noqa: E402
from run_hamiltonian_derivative_siesta_references import (  # noqa: E402
    reference_output_geometry_error,
)

LAUNCHER = REPO_ROOT / "Comparison/scripts/ops/launch_ui_real_metrics_derivatives.py"
AUTOGRAD_TRAINER = REPO_ROOT / "Comparison/scripts/ops/train_deeph_autograd_models.py"
MAIN_PAYLOAD = REPO_ROOT / "Comparison/config/ui_cross_w90_to_5x5_2delta_payload.json"
PYTHON = REPO_ROOT / ".venv/bin/python"
CAMPAIGN = REPO_ROOT / "Comparison/results/ui_real_metrics_derivatives/cross_w90_to_5x5_2delta"

CASES = [
    "iid20", "iid30", "iid50", "iid60", "iid80", "iid90",
    "iid100", "iid150", "iid200", "iid300", "iid400", "iid500",
]

# How many references to sample when deciding whether a case is already clean. The
# contamination was uniform (93/96 sampled across every size), so a handful is decisive.
GUARD_SAMPLE = 6


def _case_id(case: str) -> str:
    return (
        f"cross_graphene__graphene_w90_scale_{case}"
        f"__to__graphene_5x5__graphene_5x5_scale_{case}"
    )


CASE_IDS = {case: _case_id(case) for case in CASES}


def case_dir(case: str) -> Path:
    return CAMPAIGN / _case_id(case)


def case_is_clean(case: str) -> bool:
    """True when the case's existing references are single-point at the stencil geometry."""
    refs = case_dir(case) / "siesta_hamiltonians"
    if not refs.is_dir():
        return False
    sampled = [d for d in sorted(refs.iterdir()) if d.is_dir()][:GUARD_SAMPLE]
    if not sampled:
        return False
    for ref in sampled:
        selection = choose_reference_matrix(ref)
        if not selection.ok:
            return False
        if reference_output_geometry_error(ref, selection.path):
            return False
    return True


def case_is_done(case: str) -> bool:
    """Clean references AND a completed workflow.

    Guard-clean alone is not completion: the SIESTA stage can finish and the workflow
    still fail downstream (a missing DeepH autograd model does exactly that), which would
    otherwise leave the case skipped forever with stale contaminated metrics.
    """
    return (case_dir(case) / "regeneracion_ok.json").exists() and case_is_clean(case)


def mark_case_done(case: str, reference_workers: int | None) -> None:
    (case_dir(case) / "regeneracion_ok.json").write_text(
        json.dumps(
            {
                "case": case,
                "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "reference_workers": reference_workers,
                "note": "stencils y referencias SIESTA regenerados en single-point y validados por el guard de geometria",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def ensure_deeph_autograd_models(cases: list[str], env: dict[str, str], log) -> None:
    """Train the DeepH autograd models the derivative workflow needs.

    ``predict_derivative_deeph`` preflights on derivative.deeph_model_dir, so a case
    without deeph_autograd_model/train fails instantly. The trainer skips cases that
    already have one, so this is cheap when nothing is missing.
    """
    missing = [
        case
        for case in cases
        if not (case_dir(case) / "deeph_autograd_model/train/best_state_dict.pkl").exists()
    ]
    if not missing:
        log("modelos DeepH autograd: todos presentes.")
        return
    log(f"modelos DeepH autograd ausentes en {', '.join(missing)}; entrenando.")
    for case in missing:
        # One --case-id per call: handing the trainer the whole payload lets it run
        # max_parallel_deeph_autograd_jobs at once, which returns rc=0 without writing
        # the models. Serially it is reliable (~1 min per case).
        result = subprocess.run(
            [
                str(PYTHON), "-u", str(AUTOGRAD_TRAINER), str(MAIN_PAYLOAD),
                "--case-id", CASE_IDS[case],
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            env=env,
        )
        built = (case_dir(case) / "deeph_autograd_model/train/best_state_dict.pkl").exists()
        log(f"  modelo autograd {case}: rc={result.returncode} creado={built}")


def overwrite_payload(case: str, scratch: Path, reference_workers: int | None = None, *, overwrite: bool = True) -> Path:
    """Committed solo payload + overwrite, written to scratch (not into config/)."""
    source = REPO_ROOT / f"Comparison/config/ui_cross_w90_to_5x5_2delta_{case}_solo_payload.json"
    body = json.loads(source.read_text(encoding="utf-8"))
    body["derivative"] = {**(body.get("derivative") or {}), "overwrite": bool(overwrite)}
    note = ""
    if reference_workers is not None:
        # Each SIESTA worker is ~1 core (no mpirun, no libgomp), so the worker count is
        # the core budget: drop it when something else is already running on this box.
        body["derivative"]["reference_workers"] = int(reference_workers)
        note = f" reference_workers bajado a {reference_workers} para correr en paralelo con otra campana."
    body["description"] = (
        f"REGENERACION single-point de {case}: overwrite de stencils y referencias SIESTA "
        "contaminadas por el bloque MD heredado. Derivado de " + source.name + note
    )
    target = scratch / f"regen_{case}{'' if overwrite else '_predict'}_payload.json"
    target.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    return target


def wait_for(pattern: str, log) -> None:
    """Block while any process matching ``pattern`` is alive."""
    while subprocess.run(["pgrep", "-f", pattern], capture_output=True).returncode == 0:
        log(f"esperando a que termine: {pattern}")
        time.sleep(60)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wait-for",
        default="run_5x5_to_w90.sh|run_cross_structure_sweep_payload.py",
        help="pgrep -f pattern to wait on before starting; empty string starts immediately.",
    )
    parser.add_argument("--cases", nargs="+", default=CASES)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument(
        "--force", action="store_true", help="Regenerate even if the guard says a case is clean."
    )
    parser.add_argument(
        "--reference-workers",
        type=int,
        help="Override the payload's SIESTA worker count (~1 core each). Lower it when this "
        "runs alongside another campaign so the total core load stays within budget.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Cap on BLAS/OpenMP/torch threads per child process. reference_workers only "
        "bounds the SIESTA stage; the Graph2Mat/DeepH autograd stages are torch and will "
        "otherwise grab every core (measured: 31 threads, 1662%% CPU, 82 C).",
    )
    args = parser.parse_args()

    args.scratch.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)

    def log(message: str) -> None:
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {message}"
        with args.log.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        print(line, flush=True)

    child_env = {
        **os.environ,
        "OMP_NUM_THREADS": str(args.threads),
        "MKL_NUM_THREADS": str(args.threads),
        "OPENBLAS_NUM_THREADS": str(args.threads),
        "NUMEXPR_NUM_THREADS": str(args.threads),
        "TORCH_NUM_THREADS": str(args.threads),
    }

    if args.wait_for:
        wait_for(args.wait_for, log)
    workers = args.reference_workers if args.reference_workers is not None else "los del payload"
    log(
        f"empieza la regeneracion single-point (workers SIESTA: {workers}, "
        f"hilos por proceso: {args.threads})."
    )

    ensure_deeph_autograd_models(list(args.cases), child_env, log)

    failed: list[str] = []
    for case in args.cases:
        if not args.force and case_is_done(case):
            log(f"{case}: ya regenerado y validado, salto.")
            continue
        done = False
        for attempt in range(1, args.attempts + 1):
            log(f"{case}: intento {attempt}/{args.attempts}")
            started = time.time()
            # Two passes. overwrite=true wipes the case dir, which destroys
            # deeph_autograd_model/ and makes the predict_derivative_deeph preflight fail,
            # so it is confined to the stencil+reference stages (--reference-only). The
            # model is then (re)trained and the prediction/metric stages run with
            # overwrite=false, reusing the freshly regenerated references.
            result = subprocess.run(
                [
                    str(PYTHON), "-u", str(LAUNCHER),
                    str(overwrite_payload(case, args.scratch, args.reference_workers)),
                    "--reference-only",
                ],
                cwd=REPO_ROOT, capture_output=True, text=True, env=child_env,
            )
            log(f"  {case}: fase referencias rc={result.returncode} guard_ok={case_is_clean(case)}")
            if result.returncode == 0 and case_is_clean(case):
                ensure_deeph_autograd_models([case], child_env, log)
                result = subprocess.run(
                    [
                        str(PYTHON), "-u", str(LAUNCHER),
                        str(overwrite_payload(case, args.scratch, args.reference_workers, overwrite=False)),
                    ],
                    cwd=REPO_ROOT, capture_output=True, text=True, env=child_env,
                )
            (args.scratch / f"regen_{case}.log").write_text(
                result.stdout + result.stderr, encoding="utf-8"
            )
            minutes = (time.time() - started) / 60
            if result.returncode == 0 and case_is_clean(case):
                mark_case_done(case, args.reference_workers)
                log(f"{case}: OK en {minutes:.1f} min, guard conforme.")
                done = True
                break
            log(
                f"{case}: fallo (rc={result.returncode}, guard_ok={case_is_clean(case)}) "
                f"tras {minutes:.1f} min"
            )
            time.sleep(30)
        if not done:
            # Keep going: one bad case should not cost the other eleven.
            failed.append(case)
            log(f"{case}: ERROR tras {args.attempts} intentos, continuo con el resto.")

    if failed:
        log(f"TERMINADO con fallos: {', '.join(failed)}")
        return 1
    log("TERMINADO: todas las referencias regeneradas y validadas por el guard.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
