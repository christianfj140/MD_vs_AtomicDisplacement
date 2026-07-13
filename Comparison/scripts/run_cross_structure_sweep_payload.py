#!/usr/bin/env python3
"""Run a cross-structure *sweep* payload (source×target pairs) from the terminal.

Same body schema as the UI ``/api/cross-testing/launch`` endpoint::

    {
      "action": "preview" | "materialize" | "train",
      "sources": ["<dataset_root>", ...],
      "targets": ["<dataset_root>", ...],
      "models": ["graph2mat", "deeph"],
      "epochs": 40,
      "seed": 0,
      "confirm_incomplete_hamiltonian_semantics": true
    }

For ``action=train`` each compatible pair drives the real Graph2Mat/DeepH runner
(needs the models installed and, for a full sweep, a GPU). Incompatible pairs are
skipped with a warning; ``training_sweep`` is not supported in cross-structure.
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


def _launch_fn(runner_payload: dict[str, Any]) -> dict[str, Any]:
    """Drive the real runner synchronously for one composite dataset."""
    from g2m_deeph_runner import Graph2MatDeepHBenchmarkRunner  # noqa: PLC0415
    from pipeline_ui import _extract_model_h_mae_eV, _mixing_run_ok  # noqa: PLC0415

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
    ok = _mixing_run_ok(runner_status)
    missing = [model for model in models if model not in metrics]
    if ok and missing:
        ok = False
    return {"ok": ok, "error": runner_status.get("error"), "metrics": metrics}


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
    launch_fn = _launch_fn if action == "train" else None

    summary = mvs.run_cross_structure_sweep(
        _roots(body, "sources"),
        _roots(body, "targets"),
        args.output_root,
        models=tuple(body.get("models") or ("graph2mat", "deeph")),
        epochs=int(body["epochs"]) if body.get("epochs") not in (None, "") else None,
        performance=body.get("performance") or None,
        seed=int(body.get("seed") or 0),
        confirm_ghost_species_exemption=bool(body.get("confirm_ghost_species_exemption")),
        confirm_incomplete_hamiltonian_semantics=bool(body.get("confirm_incomplete_hamiltonian_semantics")),
        action=action,
        launch_fn=launch_fn,
    )
    text = json.dumps(summary, indent=2) + "\n"
    if args.result_json:
        args.result_json.parent.mkdir(parents=True, exist_ok=True)
        args.result_json.write_text(text, encoding="utf-8")
    print(text, end="")
    return 1 if int(summary.get("n_failed") or 0) > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
