#!/usr/bin/env python3
"""Run a cross-structure *sweep* payload (source×target pairs) from the terminal.

Same body schema as the UI ``/api/cross-testing/launch`` endpoint::

    {
      "action": "preview" | "materialize" | "train" | "predict_metrics",
      "sources": ["<dataset_root>", ...],
      "targets": ["<dataset_root>", ...],
      "models": ["graph2mat", "deeph"],
      "epochs": 40,
      "seed": 0,
      "confirm_incomplete_hamiltonian_semantics": true
    }

For ``action=train`` or ``action=predict_metrics`` each compatible pair drives the
real Graph2Mat/DeepH runner (needs the models installed and, for a full sweep, a
GPU). Incompatible pairs are skipped with a warning; ``training_sweep`` is not
supported in cross-structure.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "Comparison" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import ml_vs_siesta as mvs  # noqa: E402

_POLL_SECONDS = 5.0


def _roots(body: dict[str, Any], key: str) -> list[str]:
    raw = body.get(key)
    if isinstance(raw, dict):
        raw = list(raw.values())
    roots: list[str] = []
    for root in raw or []:
        path = Path(str(root))
        roots.append(str(path if path.is_absolute() else REPO_ROOT / path))
    return roots


def _pairs(body: dict[str, Any]) -> list[tuple[str, str]] | None:
    raw = body.get("pairs")
    if not raw:
        return None
    pairs: list[tuple[str, str]] = []
    for item in raw:
        source = Path(str(item["source"]))
        target = Path(str(item["target"]))
        pairs.append((
            str(source if source.is_absolute() else REPO_ROOT / source),
            str(target if target.is_absolute() else REPO_ROOT / target),
        ))
    return pairs


def _launch_fn(runner_payload: dict[str, Any]) -> dict[str, Any]:
    """Drive the real runner synchronously for one composite dataset.

    Same logic as ``pipeline_ui._cross_testing_launch_fn`` (used by the UI),
    including the common-CSV fallback: without it, metrics that only land in
    the per-model common_metrics CSV (not the in-memory results dict) are
    silently missed and the pair is wrongly reported as failed.
    """
    from g2m_deeph_runner import Graph2MatDeepHBenchmarkRunner  # noqa: PLC0415
    from pipeline_ui import (  # noqa: PLC0415
        _extract_model_h_mae_eV,
        _mixing_metrics_from_common_csv,
        _mixing_run_ok,
    )

    runner = Graph2MatDeepHBenchmarkRunner()
    models = tuple(
        runner_payload.get("models")
        or runner_payload.get("selected_methods")
        or ("graph2mat", "deeph")
    )
    runner.start(runner_payload)
    while runner.status().get("running"):
        time.sleep(_POLL_SECONDS)
    results = runner.results()
    runner_status = results.get("status") or {}
    metrics = _extract_model_h_mae_eV(results, models)
    if len(metrics) < len(models):
        metrics.update(_mixing_metrics_from_common_csv(runner_status.get("run_root"), models))
    ok = _mixing_run_ok(runner_status)
    error: str | None = runner_status.get("error") if isinstance(runner_status, dict) else None
    missing_models = [model for model in models if model not in metrics]
    if ok and missing_models:
        ok = False
        error = error or f"runner produced no h_mae_eV for models: {missing_models}"
    return {
        "ok": ok,
        "error": error,
        "metrics": metrics,
        "missing_models": missing_models,
        "runner_status": runner_status,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("payload", type=Path)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "Comparison/results/ml_vs_siesta_cross_structure_sweep")
    parser.add_argument("--result-json", type=Path)
    args = parser.parse_args()

    body = json.loads(args.payload.read_text(encoding="utf-8"))
    if not isinstance(body, dict):
        raise RuntimeError(f"{args.payload} must contain a JSON object.")
    action = str(body.get("action") or "preview").strip().lower()
    launch_fn = _launch_fn if action in {"train", "predict_metrics"} else None

    # "seeds": [0, 1, 2] runs the whole sweep once per seed into
    # output_root/seed_<s>/ (the layout of the existing *_10seeds campaigns).
    # A bare "seed" keeps the flat single-run layout.
    seeds = [int(s) for s in (body.get("seeds") or [])] or [int(body.get("seed") or 0)]
    multi_seed = len(seeds) > 1

    def _run_one(seed: int, output_root: Path) -> dict[str, Any]:
        return mvs.run_cross_structure_sweep(
            _roots(body, "sources"),
            _roots(body, "targets"),
            output_root,
            pairs=_pairs(body),
            models=tuple(body.get("models") or ("graph2mat", "deeph")),
            epochs=int(body["epochs"]) if body.get("epochs") not in (None, "") else None,
            hyperparams=body.get("hyperparams") or None,
            early_stopping=body.get("early_stopping") or None,
            existing_artifacts=body.get("existing_artifacts") or None,
            performance=body.get("performance") or None,
            seed=seed,
            confirm_ghost_species_exemption=bool(body.get("confirm_ghost_species_exemption")),
            confirm_incomplete_hamiltonian_semantics=bool(body.get("confirm_incomplete_hamiltonian_semantics")),
            strict_dataset_validation=bool(body.get("strict_dataset_validation", True)),
            action=action,
            launch_fn=launch_fn,
        )

    if multi_seed:
        per_seed = {seed: _run_one(seed, args.output_root / f"seed_{seed}") for seed in seeds}
        summary = {
            "seeds": seeds,
            "per_seed_output_roots": {str(s): str(args.output_root / f"seed_{s}") for s in seeds},
            "records": [record for s in seeds for record in per_seed[s].get("records") or []],
            "n_failed": sum(int(per_seed[s].get("n_failed") or 0) for s in seeds),
            "per_seed": {str(s): per_seed[s] for s in seeds},
        }
    else:
        summary = _run_one(seeds[0], args.output_root)
    text = json.dumps(summary, indent=2) + "\n"
    if args.result_json:
        args.result_json.parent.mkdir(parents=True, exist_ok=True)
        args.result_json.write_text(text, encoding="utf-8")
    print(text, end="")
    return 1 if int(summary.get("n_failed") or 0) > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
