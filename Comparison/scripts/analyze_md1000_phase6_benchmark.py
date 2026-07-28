#!/usr/bin/env python3
"""Aggregate and decide the MD1000 Phase-6 Hamiltonian benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "Comparison" / "results"
RESULTS_MD = RESULTS_ROOT / "results_md"
OUTPUT_ROOT = RESULTS_ROOT / "h2o_md1000_phase6_architecture_benchmark"
REPORT_PATH = OUTPUT_ROOT / "final_decision_report.md"
PHASE6_PAYLOAD = REPO_ROOT / "Comparison" / "config" / "h2o_phase6_hamiltonian_architecture_benchmark_payload.json"

BASELINE_RUNS = [
    {
        "method": "default_huber_0p01_honly_sweep015",
        "role": "archived H-only baseline",
        "path": RESULTS_MD
        / "md_dataset1_zzu9ln_sweep015_ep550_lr0p005_l3_c32_i3_corr2_b96_w4_s15_honly_huber_b0p01"
        / "run_20260520_095922",
    },
    {
        "method": "default_huber_0p003_honly_sweep015",
        "role": "archived H-only beta comparator",
        "path": RESULTS_MD
        / "md_dataset1_zzu9ln_sweep015_ep550_lr0p005_l3_c32_i3_corr2_b96_w4_s15_honly_huber_b0p003"
        / "run_20260520_095922",
    },
]

EXPECTED_METHODS = [
    "default_huber_0p01_honly_sweep015",
    "default_huber_0p003_honly_sweep015",
    "default_mse_honly",
    "default_mae_honly",
    "default_block_normalized_huber_0p01_honly",
    "hamiltonian_context_default_huber_0p01",
    "hamiltonian_context_readout_huber_0p01",
    "hamiltonian_context_readout_staged_composite",
    "edge_node_mix_huber_0p01_honly",
    "diagnostic_dense_upper_bound",
]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def as_float(value: Any) -> float | None:
    if value in (None, "", "nan", "NaN", "None"):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def fmt(value: Any, digits: int = 3) -> str:
    number = as_float(value)
    if number is None:
        return "NA"
    return f"{number:.{digits}f}"


def series(rows: list[dict[str, str]], column: str) -> list[float]:
    return [value for row in rows if (value := as_float(row.get(column))) is not None]


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def stdev(values: list[float]) -> float | None:
    if not values:
        return None
    return statistics.stdev(values) if len(values) > 1 else 0.0


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def git_info(path: Path) -> dict[str, Any]:
    def run(args: list[str]) -> str | None:
        completed = subprocess.run(
            ["git", "-C", str(path), *args],
            capture_output=True,
            text=True,
            check=False,
        )
        return completed.stdout.strip() if completed.returncode == 0 else None

    return {
        "commit": run(["rev-parse", "HEAD"]),
        "branch": run(["rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(run(["status", "--porcelain"])),
    }


def failure_counts(metrics_manifest: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in metrics_manifest.get("fatal_errors") or []:
        if isinstance(item, dict):
            kind = str(item.get("kind") or "unknown")
            counts[kind] = counts.get(kind, 0) + 1
    return counts


def classify_run(run_dir: Path) -> tuple[str, str]:
    manifest = read_json(run_dir / "manifest.json")
    metadata = manifest.get("benchmark_metadata")
    if isinstance(metadata, dict):
        method = metadata.get("benchmark_method_id")
        if method:
            return str(method), str(manifest.get("training_plan_display_label") or method)
    label = run_dir.parent.name
    if "phase6_default_mse" in label:
        return "default_mse_honly", "Phase6 default readout H-only block MSE seed0"
    if "phase6_default_mae" in label:
        return "default_mae_honly", "Phase6 default readout H-only block MAE seed0"
    if "phase6_default_block_normalized_huber" in label:
        return (
            "default_block_normalized_huber_0p01_honly",
            "Phase6 default readout H-only block-normalized Huber beta 0.01 seed0",
        )
    if "phase6_context_default_huber" in label:
        return "hamiltonian_context_default_huber_0p01", "Phase6 Hamiltonian context plus default readout"
    if "phase6_context_hamreadout_huber" in label:
        return "hamiltonian_context_readout_huber_0p01", "Phase6 Hamiltonian context plus Hamiltonian readout"
    if "phase6_context_hamreadout_staged" in label:
        return "hamiltonian_context_readout_staged_composite", "Phase6 staged composite"
    if "phase6_dense_upper_bound" in label:
        return "diagnostic_dense_upper_bound", "Phase6 dense diagnostic upper bound"
    if "edge_node_mix_huber_0p01_honly" in label:
        return "edge_node_mix_huber_0p01_honly", "EdgeNodeMix edge readout H-only Huber beta 0.01 seed0"
    return label, label


def phase6_run_entries(run_id: str | None) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    run_dirs: dict[Path, None] = {}
    for pattern in (
        "*phase6*/run_*",
        "*edge_node_mix_huber_0p01_honly*/run_*",
        "*block_normalized_huber*/run_*",
    ):
        for run_dir in sorted(RESULTS_MD.glob(pattern)):
            run_dirs[run_dir] = None
    for run_dir in sorted(run_dirs):
        if run_id and run_dir.name != f"run_{run_id}":
            continue
        method, role = classify_run(run_dir)
        entries.append({"method": method, "role": role, "path": run_dir})
    return entries


def h_only_valid(data: dict[str, Any]) -> bool:
    return (
        data.get("out_matrix") == "hamiltonian"
        and data.get("matrix_component_policy") == "h_only"
        and int(data.get("n_matrix_components", 0) or 0) == 1
        and bool(data.get("symmetric_matrix")) is True
    )


def references(run_dir: Path) -> dict[str, Any]:
    reference_root = run_dir / "siesta_hamiltonians"
    files = [path for path in reference_root.rglob("*") if path.is_file()]
    forbidden = [str(path.relative_to(run_dir)) for path in files if path.name == "ML_prediction.HSX"]
    allowed = [
        path
        for path in files
        if path.name != "ML_prediction.HSX" and path.suffix in {".TSHS", ".HSX"}
    ]
    return {
        "reference_files": len(allowed),
        "forbidden_prediction_references": len(forbidden),
        "forbidden_prediction_reference_paths": forbidden[:20],
    }


def summarize_run(entry: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    run_dir = Path(entry["path"])
    manifest = read_json(run_dir / "manifest.json")
    config = read_yaml(run_dir / "pipeline_config.yaml")
    metrics_manifest = read_json(run_dir / "metrics" / "manifest.json")
    sparse = read_rows(run_dir / "metrics" / "sparse_metrics.csv")
    spectral = read_rows(run_dir / "metrics" / "spectral_metrics.csv")
    dos = read_rows(run_dir / "metrics" / "dos_metrics.csv")
    block = read_rows(run_dir / "metrics" / "block_metrics.csv")
    orbital = read_rows(run_dir / "metrics" / "orbital_pair_summary.csv")

    training = manifest.get("training_hyperparameters") if isinstance(manifest.get("training_hyperparameters"), dict) else {}
    if not training:
        training = config.get("training") if isinstance(config.get("training"), dict) else {}
    model = training.get("model") if isinstance(training.get("model"), dict) else {}
    data = training.get("data") if isinstance(training.get("data"), dict) else {}
    metadata = manifest.get("benchmark_metadata") if isinstance(manifest.get("benchmark_metadata"), dict) else {}
    method = entry["method"]
    loss = metadata.get("loss") or model.get("loss")
    loss_kwargs = metadata.get("loss_kwargs") if isinstance(metadata.get("loss_kwargs"), dict) else model.get("loss_kwargs", {})
    ref_info = references(run_dir)
    failures = failure_counts(metrics_manifest)

    by_sample: dict[str, dict[str, Any]] = {}
    for row in sparse:
        sample = str(row.get("sample") or "")
        by_sample.setdefault(sample, {}).update(
            {
                "method": method,
                "run_id": manifest.get("run_id") or run_dir.name.replace("run_", ""),
                "dataset_label": manifest.get("dataset_label") or run_dir.parent.name,
                "sample": sample,
                "mae_ref_meV": as_float(row.get("mae_ref_meV")),
                "rmse_ref_meV": as_float(row.get("rmse_ref_meV")),
                "relative_frobenius_union": as_float(row.get("relative_frobenius_union")),
                "support_f1": as_float(row.get("support_f1")),
                "hermiticity_pred": as_float(row.get("hermiticity_pred")),
            }
        )
    for row in spectral:
        sample = str(row.get("sample") or "")
        by_sample.setdefault(sample, {"method": method, "sample": sample}).update(
            {
                "global_rmse_eV": as_float(row.get("global_rmse_eV")),
                "low_energy_rmse_eV": as_float(row.get("low_energy_rmse_eV")),
                "fermi_window_rmse_eV": as_float(row.get("fermi_window_rmse_eV")),
                "frontier_window_rmse_eV": as_float(row.get("frontier_window_rmse_eV")),
            }
        )
    for row in dos:
        sample = str(row.get("sample") or "")
        by_sample.setdefault(sample, {"method": method, "sample": sample}).update(
            {"dos_mae_500_fermi_window": as_float(row.get("dos_mae_500_fermi_window"))}
        )
    per_sample = list(by_sample.values())

    row = {
        "method": method,
        "role": entry["role"],
        "run_id": manifest.get("run_id") or run_dir.name.replace("run_", ""),
        "dataset_label": manifest.get("dataset_label") or run_dir.parent.name,
        "test_samples": len(sparse),
        "h_only_valid": h_only_valid(data),
        "loss": loss,
        "loss_kwargs": json.dumps(loss_kwargs or {}, sort_keys=True),
        "architecture": metadata.get("architecture") or "",
        "readout": metadata.get("readout") or "",
        "context_enabled": metadata.get("context_enabled"),
        "diagnostic_only": bool(metadata.get("diagnostic_only")),
        "seed": metadata.get("seed") if metadata.get("seed") is not None else manifest.get("seed"),
        "h_mae_meV_mean": mean(series(sparse, "mae_ref_meV")),
        "h_mae_meV_std": stdev(series(sparse, "mae_ref_meV")),
        "h_mae_meV_p90": percentile(series(sparse, "mae_ref_meV"), 0.9),
        "h_rmse_meV_mean": mean(series(sparse, "rmse_ref_meV")),
        "relative_frobenius_union_mean": mean(series(sparse, "relative_frobenius_union")),
        "support_f1_mean": mean(series(sparse, "support_f1")),
        "hermiticity_pred_mean": mean(series(sparse, "hermiticity_pred")),
        "global_rmse_eV_mean": mean(series(spectral, "global_rmse_eV")),
        "low_energy_rmse_eV_mean": mean(series(spectral, "low_energy_rmse_eV")),
        "fermi_window_rmse_eV_mean": mean(series(spectral, "fermi_window_rmse_eV")),
        "frontier_window_rmse_eV_mean": mean(series(spectral, "frontier_window_rmse_eV")),
        "dos_mae_500_fermi_window_mean": mean(series(dos, "dos_mae_500_fermi_window")),
        "samples_failed": metrics_manifest.get("samples_failed") or sum(failures.values()),
        "failure_summary": json.dumps(failures, sort_keys=True),
        "reference_files": ref_info["reference_files"],
        "forbidden_prediction_references": ref_info["forbidden_prediction_references"],
        "pipeline_commit": manifest.get("pipeline_commit") or (manifest.get("pipeline_git") or {}).get("commit"),
        "graph2mat_commit": manifest.get("graph2mat_commit") or (manifest.get("graph2mat_git") or {}).get("commit"),
        "run_dir": str(run_dir),
    }

    worst_blocks = sorted(
        (
            {
                "method": method,
                "run_id": row["run_id"],
                "sample": item.get("sample"),
                "row_atom": item.get("row_atom"),
                "col_atom": item.get("col_atom"),
                "species_pair": f"{item.get('row_species')}-{item.get('col_species')}",
                "mae_union_meV": (as_float(item.get("mae_union_eV")) or 0.0) * 1000.0,
                "rmse_union_meV": (as_float(item.get("rmse_union_eV")) or 0.0) * 1000.0,
                "max_abs_error_union_meV": (as_float(item.get("max_abs_error_union_eV")) or 0.0) * 1000.0,
                "mean_distance_ang": as_float(item.get("mean_distance_ang")),
            }
            for item in block
        ),
        key=lambda item: item["mae_union_meV"],
        reverse=True,
    )[:50]

    worst_orbitals = sorted(
        (
            {
                "method": method,
                "run_id": row["run_id"],
                "species_pair": item.get("species_pair"),
                "row_orbital_label": item.get("row_orbital_label"),
                "col_orbital_label": item.get("col_orbital_label"),
                "mae_union_meV_mean": as_float(item.get("mae_union_meV_mean")),
                "rmse_union_eV_mean": as_float(item.get("rmse_union_eV_mean")),
                "max_abs_error_union_eV_max": as_float(item.get("max_abs_error_union_eV_max")),
                "n_entries": item.get("n_entries"),
            }
            for item in orbital
        ),
        key=lambda item: item["mae_union_meV_mean"] if item["mae_union_meV_mean"] is not None else -1.0,
        reverse=True,
    )[:50]
    return row, per_sample, worst_blocks, worst_orbitals


def markdown_table(rows: list[dict[str, Any]]) -> str:
    columns = [
        "method",
        "h_mae_meV_mean",
        "h_rmse_meV_mean",
        "low_energy_rmse_eV_mean",
        "fermi_window_rmse_eV_mean",
        "dos_mae_500_fermi_window_mean",
        "diagnostic_only",
    ]
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, sep]
    for row in sorted(rows, key=lambda item: as_float(item.get("h_mae_meV_mean")) or float("inf")):
        values = []
        for column in columns:
            if column == "method":
                values.append(str(row.get(column)))
            elif column == "diagnostic_only":
                values.append(str(row.get(column)))
            else:
                values.append(fmt(row.get(column)))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def report(rows: list[dict[str, Any]], missing: dict[str, Any], run_id: str | None) -> str:
    measured_methods = {str(row["method"]) for row in rows if row.get("h_mae_meV_mean") is not None}
    baseline = next((row for row in rows if row["method"] == "default_huber_0p01_honly_sweep015"), None)
    baseline_mae = as_float(baseline.get("h_mae_meV_mean")) if baseline else None
    production = [
        row
        for row in rows
        if row.get("h_mae_meV_mean") is not None
        and not row.get("diagnostic_only")
        and row.get("h_only_valid") is True
        and int(row.get("forbidden_prediction_references") or 0) == 0
    ]
    best = min(production, key=lambda row: as_float(row.get("h_mae_meV_mean")) or float("inf"), default=None)
    new_arch = [
        row
        for row in production
        if str(row.get("method", "")).startswith("hamiltonian_context")
        or str(row.get("method", "")) == "edge_node_mix_huber_0p01_honly"
    ]
    best_new = min(new_arch, key=lambda row: as_float(row.get("h_mae_meV_mean")) or float("inf"), default=None)
    beats_mae = (
        baseline_mae is not None
        and best_new is not None
        and (as_float(best_new.get("h_mae_meV_mean")) or float("inf")) < baseline_mae
    )
    beats_physics = False
    if baseline is not None and best_new is not None:
        beats_physics = all(
            (as_float(best_new.get(metric)) or float("inf"))
            < (as_float(baseline.get(metric)) or float("inf"))
            for metric in (
                "h_mae_meV_mean",
                "low_energy_rmse_eV_mean",
                "fermi_window_rmse_eV_mean",
            )
        )
    best_mae = as_float(best.get("h_mae_meV_mean")) if best else None
    if not new_arch:
        verdict = "INCOMPLETE: no new architecture candidate has complete metrics yet."
    elif beats_physics:
        verdict = "ITERATE GRAPH2MAT: a new architecture beat the default Huber baseline on seed 0 across H MAE and spectral window metrics; multi-seed stability is still missing."
    elif beats_mae:
        verdict = "MIXED: a new architecture beat default Huber beta=0.01 on H MAE only, but not on low-energy/Fermi metrics; do not promote it yet."
    elif best_new is not None:
        verdict = "DO NOT PROMOTE YET: new architecture candidates did not beat default Huber beta=0.01 on the available seed."
    else:
        verdict = "INCOMPLETE: new architecture candidates ran but produced no complete valid metrics."
    if best_mae is not None and best_mae > 10.0:
        deeph_note = (
            f"Best held-out H MAE is {fmt(best_mae)} meV, still tens of meV and therefore not DeepH-level low-meV accuracy."
        )
    else:
        deeph_note = "Best held-out H MAE is in low-meV territory; this would warrant a deeper DeepH-scale comparison."

    commands = [
        "python3 -m py_compile Comparison/scripts/pipeline_ui.py MD/scripts/md_pipeline_config.py MD/scripts/run_md_training.py Comparison/scripts/analyze_md1000_phase6_benchmark.py",
        "git diff --check",
        "curl -sS -X POST http://127.0.0.1:8770/api/experiment -H 'Content-Type: application/json' --data-binary @Comparison/config/h2o_phase6_hamiltonian_architecture_benchmark_payload.json",
        f"python3 Comparison/scripts/analyze_md1000_phase6_benchmark.py --run-id {run_id}" if run_id else "python3 Comparison/scripts/analyze_md1000_phase6_benchmark.py",
    ]
    return "\n".join(
        [
            "# H2O MD1000 Phase-6 Hamiltonian Benchmark Decision Report",
            "",
            "## Executive Verdict",
            "",
            verdict,
            "",
            deeph_note,
            "",
            "## Experiment Matrix",
            "",
            "\n".join(f"- {method}" for method in EXPECTED_METHODS),
            "",
            "## Data And Reference Validity",
            "",
            "- Dataset policy: reused archived MD-only 1000-snapshot dataset; no regeneration in this benchmark payload.",
            "- Split policy: preserve archived train/validation/test splits.",
            "- H-only policy: valid runs must use `out_matrix=hamiltonian`, `matrix_component_policy=h_only`, `n_matrix_components=1`, `symmetric_matrix=true`.",
            "- Reference policy: real SIESTA `.TSHS`/`.HSX` only; `ML_prediction.HSX` is forbidden as ground truth.",
            "",
            "## Aggregate Results",
            "",
            markdown_table(rows),
            "",
            "## Best Config By Metric",
            "",
            f"- Best production H MAE: `{best.get('method') if best else 'NA'}` at {fmt(best_mae)} meV.",
            f"- Best new architecture: `{best_new.get('method') if best_new else 'NA'}` at {fmt(best_new.get('h_mae_meV_mean') if best_new else None)} meV.",
            f"- New architecture beats default Huber 0.01 on H MAE: `{beats_mae}`.",
            f"- New architecture beats default Huber 0.01 on H MAE + low-energy + Fermi: `{beats_physics}`.",
            "",
            "## Per-Sample Failure Analysis",
            "",
            f"- Missing/incomplete methods: `{', '.join(missing.get('missing_methods', [])) or 'none'}`.",
            f"- Incomplete run directories: `{len(missing.get('incomplete_run_dirs', []))}`.",
            f"- Methods with complete metrics: `{', '.join(sorted(measured_methods)) or 'none'}`.",
            "",
            "## Recommendation",
            "",
            "Use the aggregate table above as the gating result. Do not promote a new architecture unless it beats `default_huber_0p01_honly_sweep015` and survives seeds 1 and 2 on the same split.",
            "",
            "## Risks And Caveats",
            "",
            "- This report is seed-0 only unless additional seeds are present in the archive.",
            "- Dense diagnostic readout is explicitly non-production.",
            "- Any candidate absent from the aggregate table either was not run or did not archive complete metrics.",
            "",
            "## Exact Commands Used",
            "",
            "\n".join(f"- `{command}`" for command in commands),
            "",
            "## Repository State",
            "",
            f"- Pipeline git: `{git_info(REPO_ROOT)}`",
            f"- Graph2Mat git: `{git_info(Path('/home/christian/repositorios/graph2mat'))}`",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=None, help="Only include Phase-6 runs with this run id.")
    args = parser.parse_args()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    entries = [*BASELINE_RUNS, *phase6_run_entries(args.run_id)]

    aggregate: list[dict[str, Any]] = []
    per_sample: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    orbital_rows: list[dict[str, Any]] = []
    incomplete: list[str] = []
    for entry in entries:
        run_dir = Path(entry["path"])
        if not (run_dir / "metrics" / "sparse_metrics.csv").exists():
            incomplete.append(str(run_dir))
            continue
        row, samples, blocks, orbitals = summarize_run(entry)
        aggregate.append(row)
        per_sample.extend(samples)
        block_rows.extend(blocks)
        orbital_rows.extend(orbitals)

    measured = {str(row["method"]) for row in aggregate}
    missing = {
        "missing_methods": [method for method in EXPECTED_METHODS if method not in measured],
        "incomplete_run_dirs": incomplete,
        "run_id_filter": args.run_id,
    }
    phase6_payload = read_json(PHASE6_PAYLOAD)
    requested_runs = list(phase6_payload.get("training_plan") or [])
    requested_seeds = sorted(
        {
            int((row.get("training_settings") or {}).get("seed_everything"))
            for row in requested_runs
            if (row.get("training_settings") or {}).get("seed_everything") is not None
        }
    )
    reusable_ids = sorted(
        {
            str(item)
            for item in phase6_payload.get("reusable_dataset_ids") or []
            if str(item)
        }
    )
    campaign_blockers: list[str] = []
    if len(aggregate) < len(requested_runs):
        campaign_blockers.append(
            f"physical_runs_incomplete:{len(aggregate)}/{len(requested_runs)}"
        )
    if len(requested_seeds) < 5:
        campaign_blockers.append(
            "fewer_than_five_independent_seeds_without_power_justification"
        )
    if reusable_ids:
        campaign_blockers.append(
            "reusable_dataset_ids_not_resolved_to_current_strict_benchmark_manifests:"
            + ",".join(reusable_ids)
        )
    campaign_status = {
        "schema": "phase6_campaign_status_v1",
        "status": "BLOCKED_FAIL_CLOSED" if campaign_blockers else "PASS",
        "claim_allowed": not campaign_blockers,
        "payload": str(PHASE6_PAYLOAD),
        "requested_run_count": len(requested_runs),
        "completed_metric_run_count": len(aggregate),
        "configured_seeds": requested_seeds,
        "minimum_independent_seeds": 5,
        "reusable_dataset_ids": reusable_ids,
        "physical_execution_verified": len(aggregate) == len(requested_runs),
        "paper_level_blockers": campaign_blockers,
        "resume_requirements": [
            "generate or identify a strict benchmark_ready H2O dataset with valid SCF and MD temporal evidence",
            "replace reusable_dataset_ids with its frozen manifest identity",
            "predeclare at least five independent seeds or a power justification",
            "execute the payload and rerun this analyzer",
        ],
    }

    aggregate_fields = [
        "method",
        "role",
        "run_id",
        "dataset_label",
        "test_samples",
        "h_only_valid",
        "loss",
        "loss_kwargs",
        "architecture",
        "readout",
        "context_enabled",
        "diagnostic_only",
        "seed",
        "h_mae_meV_mean",
        "h_mae_meV_std",
        "h_mae_meV_p90",
        "h_rmse_meV_mean",
        "relative_frobenius_union_mean",
        "support_f1_mean",
        "hermiticity_pred_mean",
        "global_rmse_eV_mean",
        "low_energy_rmse_eV_mean",
        "fermi_window_rmse_eV_mean",
        "frontier_window_rmse_eV_mean",
        "dos_mae_500_fermi_window_mean",
        "samples_failed",
        "failure_summary",
        "reference_files",
        "forbidden_prediction_references",
        "pipeline_commit",
        "graph2mat_commit",
        "run_dir",
    ]
    per_sample_fields = [
        "method",
        "run_id",
        "dataset_label",
        "sample",
        "mae_ref_meV",
        "rmse_ref_meV",
        "relative_frobenius_union",
        "support_f1",
        "hermiticity_pred",
        "global_rmse_eV",
        "low_energy_rmse_eV",
        "fermi_window_rmse_eV",
        "frontier_window_rmse_eV",
        "dos_mae_500_fermi_window",
    ]
    block_fields = [
        "method",
        "run_id",
        "sample",
        "row_atom",
        "col_atom",
        "species_pair",
        "mae_union_meV",
        "rmse_union_meV",
        "max_abs_error_union_meV",
        "mean_distance_ang",
    ]
    orbital_fields = [
        "method",
        "run_id",
        "species_pair",
        "row_orbital_label",
        "col_orbital_label",
        "mae_union_meV_mean",
        "rmse_union_eV_mean",
        "max_abs_error_union_eV_max",
        "n_entries",
    ]

    write_rows(OUTPUT_ROOT / "aggregate_results.csv", aggregate, aggregate_fields)
    write_rows(OUTPUT_ROOT / "per_sample_metrics.csv", per_sample, per_sample_fields)
    write_rows(OUTPUT_ROOT / "block_worst.csv", block_rows, block_fields)
    write_rows(OUTPUT_ROOT / "orbital_pair_worst.csv", orbital_rows, orbital_fields)
    (OUTPUT_ROOT / "missing_and_incomplete.json").write_text(
        json.dumps(missing, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (OUTPUT_ROOT / "phase6_campaign_status.json").write_text(
        json.dumps(campaign_status, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    REPORT_PATH.write_text(report(aggregate, missing, args.run_id), encoding="utf-8")
    print(f"Wrote {OUTPUT_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
