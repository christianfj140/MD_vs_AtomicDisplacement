#!/usr/bin/env python
"""Resume a mixing sweep after an OOM kill: retrain ONLY the (model, permutation)
combos that never finished, at a lower parallelism. No re-materialization, no
re-training of what already succeeded.

Generalizes the one-off Comparison/scripts/refill_mixing_paper_ready_oom.py:
that script hardcoded which permutations had failed by hand. This one reads
the most recent training_sweep_manifest.json under the sweep's parallel_run/
to find which (model, dataset_id) never reached status=="completed", and the
original payload's sizes/modes/ratios/seed to reconstruct the deterministic
perm_N -> size/mode/ratio mapping (same triple loop as mixing_sweep.py), so it
can point manual_runs at the already-materialized size{N}_{mode}_r{ratio}/
directories directly (dataset_mode=reuse_validated).

Usage:
    .venv/bin/python Comparison/scripts/ops/resume_mixing_sweep_oom.py \\
        --payload Comparison/config/<payload>.json \\
        --g2m-parallel 2 --deeph-parallel 1 \\
        --apply   # omit --apply for a dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = REPO_ROOT / "Comparison" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from g2m_deeph_runner import Graph2MatDeepHBenchmarkRunner  # noqa: E402

MIXING_SWEEP_OUTPUT_ROOT = REPO_ROOT / "Comparison" / "results" / "ml_vs_siesta_mixing_sweep"


def _ratio_slug(ratio: float) -> str:
    return f"r{ratio:.3f}".replace(".", "p")


def _ml_vs_siesta_module():
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    import ml_vs_siesta  # noqa: PLC0415

    return ml_vs_siesta


def _roots_from_payload(payload: dict[str, Any], key: str) -> dict[int, str]:
    raw = payload.get(key) or {}
    result: dict[int, str] = {}
    for size, root in raw.items():
        resolved = Path(root)
        if not resolved.is_absolute():
            resolved = REPO_ROOT / resolved
        result[int(size)] = str(resolved)
    return result


def _reconstruct_perm_order(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Same size -> mode -> ratio-part triple loop as mixing_sweep.run_mixing_sweep,
    so perm_N here lines up with the perm_N the original sweep trained against."""
    mvs = _ml_vs_siesta_module()
    small = _roots_from_payload(payload, "small")
    large = _roots_from_payload(payload, "large")
    sizes = [int(s) for s in payload["sizes"]] if payload.get("sizes") else None
    modes = tuple(payload.get("modes") or ("add", "replace"))
    ratios = tuple(float(r) for r in (payload.get("ratios") or (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)))
    seed = int(payload.get("seed") or 0)
    split_policy = str(payload.get("split_policy") or "fixed_common_test")

    plan = mvs.plan_mixing_sweep_from_roots(
        small, large, sizes=sizes, modes=modes, ratios=ratios, seed=seed,
        split_policy=split_policy,
    )
    perms = []
    for i, entry in enumerate(plan.get("permutations") or []):
        size = entry["size"]
        mode = entry["mode"]
        ratio = float(entry["ratio"])
        dirname = f"size{size}_{mode}_{_ratio_slug(ratio)}"
        perms.append({"index": i, "dataset_id": f"perm_{i}", "size": size, "mode": mode,
                       "ratio": ratio, "dirname": dirname})
    return perms


def _latest_training_sweep_manifest(payload: dict[str, Any]) -> Path | None:
    output_root = MIXING_SWEEP_OUTPUT_ROOT
    candidates = sorted(
        (output_root / "parallel_run").glob("*/sweep/training_sweep_manifest.json"),
        key=lambda p: p.stat().st_mtime if p.exists() else 0,
        reverse=True,
    )
    return candidates[0] if candidates else None


def find_incomplete_runs(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], Path | None]:
    """Return (list of {"model","dataset_id","size","mode","ratio","dirname"}, manifest_path)
    for every planned (model, perm) that never reached status "completed"."""
    perms = _reconstruct_perm_order(payload)
    perm_by_id = {p["dataset_id"]: p for p in perms}
    models = tuple(payload.get("models") or ("graph2mat", "deeph"))

    manifest_path = _latest_training_sweep_manifest(payload)
    completed: set[tuple[str, str]] = set()
    if manifest_path is not None and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        for run in manifest.get("runs") or []:
            if run.get("status") == "completed":
                completed.add((str(run.get("model")), str(run.get("dataset_id"))))

    incomplete: list[dict[str, Any]] = []
    for perm in perms:
        merged_dir = MIXING_SWEEP_OUTPUT_ROOT / perm["dirname"]
        if not merged_dir.is_dir():
            # Never materialized in the first place -- nothing to resume here;
            # a full relaunch (not this script) is needed.
            continue
        for model in models:
            if (model, perm["dataset_id"]) in completed:
                continue
            incomplete.append({**perm, "model": model, "dataset_root": str(merged_dir)})
    return incomplete, manifest_path


def _build_resume_payload(
    incomplete: list[dict[str, Any]],
    original_payload: dict[str, Any],
    *,
    g2m_parallel: int,
    deeph_parallel: int,
    output_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    hyperparams = original_payload.get("hyperparams") or {}
    epochs = original_payload.get("epochs")
    early_stopping = original_payload.get("early_stopping")
    training_weighting_policy = str(original_payload.get("training_weighting_policy") or "legacy_elementwise")

    # One dataset_id per unique permutation dir (a perm training both models
    # shares one dataset_id, same convention as refill_mixing_paper_ready_oom.py).
    seen_dirnames: dict[str, str] = {}
    datasets_list: list[dict[str, Any]] = []
    manual_runs: list[dict[str, Any]] = []
    for entry in incomplete:
        dirname = entry["dirname"]
        if dirname not in seen_dirnames:
            seen_dirnames[dirname] = entry["dataset_id"]
            datasets_list.append({"dataset_id": entry["dataset_id"], "dataset_root": entry["dataset_root"]})
        dataset_id = seen_dirnames[dirname]
        overrides: dict[str, Any] = dict(hyperparams.get(entry["model"]) or {})
        if epochs is not None:
            if entry["model"] == "graph2mat":
                overrides["max_epochs"] = int(epochs)
            else:
                overrides["epochs"] = int(epochs)
        if training_weighting_policy not in ("", "legacy_elementwise"):
            if entry["model"] == "graph2mat":
                overrides.setdefault(
                    "loss",
                    "graph2mat.core.data.metrics.block_type_mse_per_structure"
                    if training_weighting_policy == "per_structure"
                    else "graph2mat.core.data.metrics.block_type_mse_per_domain",
                )
            else:
                overrides.setdefault("training_weighting_policy", training_weighting_policy)
        manual_runs.append({
            "model": entry["model"],
            "dataset_id": dataset_id,
            "config_id": f"{entry['model']}_{dataset_id}",
            "seed": int(original_payload.get("seed") or 0),
            "overrides": overrides,
        })

    performance = dict(original_payload.get("performance") or {})
    performance["max_parallel_graph2mat_training_jobs"] = g2m_parallel
    performance["max_parallel_deeph_training_jobs"] = deeph_parallel

    runner_payload: dict[str, Any] = {
        "dataset_mode": "reuse_validated",
        "dataset_root": datasets_list[0]["dataset_root"],
        "output_root": str(output_root),
        "allow_regenerate_siesta": False,
        "strict_dataset_validation": False,
        "performance": performance,
        "training_sweep": {
            "enabled": True,
            "error_policy": "continue_on_error",
            "search_policy": {"strategy": "manual", "random_seed": int(original_payload.get("seed") or 0)},
            "manual_runs": manual_runs,
        },
        "_mixing_datasets_list": datasets_list,
    }
    if early_stopping:
        runner_payload["early_stopping"] = dict(early_stopping)
    if epochs is not None:
        runner_payload["epochs"] = int(epochs)
    return runner_payload, datasets_list


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", type=Path, required=True, help="Original mixing payload JSON.")
    parser.add_argument("--g2m-parallel", type=int, required=True)
    parser.add_argument("--deeph-parallel", type=int, required=True)
    parser.add_argument("--apply", action="store_true", help="Actually launch training (default: dry-run).")
    args = parser.parse_args()

    payload_path = args.payload if args.payload.is_absolute() else REPO_ROOT / args.payload
    original_payload = json.loads(payload_path.read_text())

    incomplete, manifest_path = find_incomplete_runs(original_payload)
    print(f"[resume] latest training_sweep_manifest: {manifest_path}")
    print(f"[resume] incomplete (model, permutation) combos: {len(incomplete)}")
    for entry in incomplete:
        print(f"  - {entry['model']} / {entry['dirname']}")

    if not incomplete:
        print("[resume] Nothing to resume; every planned run already completed.")
        return 0

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_root = MIXING_SWEEP_OUTPUT_ROOT / "resume_run" / f"resume_{timestamp}"
    runner_payload, datasets_list = _build_resume_payload(
        incomplete, original_payload,
        g2m_parallel=args.g2m_parallel, deeph_parallel=args.deeph_parallel,
        output_root=output_root,
    )
    print(f"[resume] output_root: {output_root}")
    print(f"[resume] datasets: {len(datasets_list)} | manual_runs: {len(runner_payload['training_sweep']['manual_runs'])}")
    print(f"[resume] parallelism: graph2mat={args.g2m_parallel} deeph={args.deeph_parallel}")

    missing = [d for d in datasets_list if not Path(d["dataset_root"]).is_dir()]
    if missing:
        print(f"[resume] ABORT: {len(missing)} dataset_root(s) missing on disk: {missing}")
        return 1

    json.dumps(runner_payload)  # must be JSON-serializable end to end
    print("[resume] Payload parses (json.dumps OK).")

    if not args.apply:
        print("[resume] DRY-RUN OK. Re-run with --apply to launch training.")
        return 0

    output_root.mkdir(parents=True, exist_ok=True)
    runner = Graph2MatDeepHBenchmarkRunner()

    def _patched_training_sweep_datasets(_validation: dict[str, Any]) -> list[dict[str, Any]]:
        return datasets_list

    runner._training_sweep_datasets = _patched_training_sweep_datasets  # type: ignore[method-assign]

    print("[resume] Launching runner...")
    runner.start(runner_payload)
    log_offset = 0
    while True:
        try:
            log_payload = runner.logs(since=log_offset, limit=None)
            for line in log_payload.get("lines") or []:
                sys.stdout.write(str(line))
            log_offset = int(log_payload.get("offset") or log_offset)
        except Exception as exc:  # noqa: BLE001
            print(f"[resume][WARN] log read failed: {exc}")
        if not runner.status().get("running"):
            break
        time.sleep(5.0)

    results = runner.results()
    status = results.get("status") or {}
    print(f"[resume] Runner finished. run_root: {status.get('run_root')}")
    print(f"[resume] status: {json.dumps({k: status.get(k) for k in ('running', 'error', 'returncode') if k in status})}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
