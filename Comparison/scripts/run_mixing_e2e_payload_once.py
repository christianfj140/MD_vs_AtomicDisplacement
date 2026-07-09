#!/usr/bin/env python3
"""Run a two-phase mixing E2E payload: generate source datasets, then mix/train."""

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

from g2m_deeph_runner import Graph2MatDeepHBenchmarkRunner  # noqa: E402
import pipeline_ui as ui  # noqa: E402


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} must contain a JSON object.")
    return payload


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def expected_paths_present(paths: list[str | Path]) -> bool:
    return all(resolve_repo_path(path).exists() for path in paths)


def _run_generation_payload(payload_path: Path, poll_seconds: float) -> dict[str, Any]:
    payload = read_json(payload_path)
    runner = Graph2MatDeepHBenchmarkRunner()
    runner.start(payload)
    while True:
        status = runner.status()
        if not status.get("running"):
            break
        time.sleep(max(1.0, float(poll_seconds)))
    results = runner.results()
    status = results.get("status") or {}
    returncode = status.get("returncode")
    if returncode not in (0, 2, None) or status.get("error"):
        raise RuntimeError(f"Generation payload failed: {status.get('error') or returncode}")
    return results


def _run_mixing_payload(payload: dict[str, Any], output_root: Path | None) -> dict[str, Any]:
    small = ui._mixing_roots_from_body(payload, "small")
    large = ui._mixing_roots_from_body(payload, "large")
    modes = tuple(payload.get("modes") or ("add", "replace"))
    ratios = tuple(float(r) for r in (payload.get("ratios") or (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)))
    sizes = [int(s) for s in payload["sizes"]] if payload.get("sizes") else None
    seed = int(payload.get("seed") or 0)
    models = tuple(payload.get("models") or ("graph2mat", "deeph"))
    epochs = int(payload["epochs"]) if payload.get("epochs") not in (None, "") else None
    performance = payload.get("performance") or None
    action = str(payload.get("action") or "preview")
    # Default fixed_common_test: resplit_combined re-splits the test set per
    # ratio, so MAE differences between ratios can come from the test set
    # changing instead of the composition (see docs/ml_vs_siesta_benchmark.md).
    split_policy = str(payload.get("split_policy") or "fixed_common_test")
    confirm_ghost = bool(payload.get("confirm_ghost_species_exemption"))
    root = output_root or ui.MIXING_SWEEP_OUTPUT_ROOT
    if action == "train":
        return ui._run_mixing_sweep_parallel(
            small,
            large,
            root,
            sizes=sizes,
            modes=modes,
            ratios=ratios,
            seed=seed,
            models=models,
            epochs=epochs,
            performance=performance,
            split_policy=split_policy,
            confirm_ghost_species_exemption=confirm_ghost,
            progress_fn=None,
        )
    mvs = ui._ml_vs_siesta_module()
    dry_run = action == "preview"
    return mvs.run_mixing_sweep(
        small,
        large,
        root,
        sizes=sizes,
        modes=modes,
        ratios=ratios,
        seed=seed,
        models=models,
        epochs=epochs,
        performance=performance,
        split_policy=split_policy,
        confirm_ghost_species_exemption=confirm_ghost,
        dry_run=dry_run,
        launch_fn=None,
        progress_fn=None,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("payload", type=Path)
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    args = parser.parse_args()

    payload = read_json(args.payload)
    generation_payload = resolve_repo_path(payload["generation_payload"])
    mixing_payload_path = resolve_repo_path(payload["mixing_payload"])
    mixing_payload = read_json(mixing_payload_path)
    expected = payload.get("expected_paths") or []
    skip_if_present = bool(payload.get("skip_generation_if_present", True))
    mixing_output_root = payload.get("mixing_output_root")
    output_root = resolve_repo_path(mixing_output_root) if mixing_output_root else None

    manifest: dict[str, Any] = {
        "schema": "ml_vs_siesta_mixing_e2e_payload_once_v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "payload": str(args.payload),
        "generation_payload": str(generation_payload),
        "mixing_payload": str(mixing_payload_path),
        "expected_paths": [str(resolve_repo_path(path)) for path in expected],
        "skip_generation_if_present": skip_if_present,
    }
    write_json(args.manifest_json, manifest)

    if skip_if_present and expected_paths_present(expected):
        manifest["generation"] = {"status": "skipped", "reason": "expected_paths_present"}
    else:
        generation_result = _run_generation_payload(generation_payload, args.poll_seconds)
        manifest["generation"] = {
            "status": "completed",
            "results": generation_result,
        }
        write_json(args.manifest_json, manifest)

    if expected and not expected_paths_present(expected):
        raise RuntimeError("Expected generated dataset paths are still missing after generation phase.")

    mixing_summary = _run_mixing_payload_seeds(mixing_payload, output_root)
    manifest["mixing"] = {
        "status": "completed",
        "summary": mixing_summary,
        "output_root": str(output_root or ui.MIXING_SWEEP_OUTPUT_ROOT),
    }
    write_json(args.manifest_json, manifest)
    return 0


def _run_mixing_payload_seeds(payload: dict[str, Any], output_root: Path | None) -> dict[str, Any]:
    """Run the mixing payload once per seed (``seeds`` list) or once (``seed``).

    Multi-seed runs write each sweep under ``<output_root>/seed<N>`` and then
    aggregate one combined MAE-vs-size payload (mean ± std, n_seeds per point)
    at ``output_root``. Fewer than 3 seeds is flagged exploratory by the
    aggregator (audit I5).
    """
    seeds = payload.get("seeds")
    if not seeds:
        return _run_mixing_payload(payload, output_root)
    seeds = [int(s) for s in seeds]
    root = output_root or ui.MIXING_SWEEP_OUTPUT_ROOT
    mvs = ui._ml_vs_siesta_module()
    per_seed: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for seed in seeds:
        seed_payload = dict(payload)
        seed_payload["seed"] = seed
        seed_payload.pop("seeds", None)
        summary = _run_mixing_payload(seed_payload, Path(root) / f"seed{seed}")
        per_seed.append({"seed": seed, "summary": summary})
        records.extend(summary.get("records") or [])
    aggregated = mvs.write_mae_vs_size_outputs(records, root)
    return {
        "schema": "ml_vs_siesta_mixing_multi_seed_summary_v1",
        "seeds": seeds,
        "n_seeds": len(seeds),
        "exploratory": aggregated.get("exploratory"),
        "warnings": aggregated.get("warnings") or [],
        "records": records,
        "mae_vs_size": {k: aggregated[k] for k in ("json_path", "png_path") if k in aggregated},
        "per_seed": per_seed,
    }


if __name__ == "__main__":
    raise SystemExit(main())
