#!/usr/bin/env python3
"""Decision analysis for the H2O Hamiltonian architecture reform benchmark."""

from __future__ import annotations

import csv
import json
import math
import statistics
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "Comparison" / "results"
OUTPUT_ROOT = RESULTS_ROOT / "h2o_hamiltonian_architecture_reform"
REPORT_PATH = OUTPUT_ROOT / "final_decision_report.md"

EXPECTED_METHODS = [
    "default_mae_honly",
    "default_huber_0p01_honly",
    "default_mse_honly",
    "hamiltonian_context_default_huber_0p01",
    "hamiltonian_context_readout_huber_0p01",
    "hamiltonian_context_readout_staged_composite",
    "diagnostic_dense_upper_bound",
]

MEASURED_RUNS = [
    {
        "method": "legacy_default_mae_sweep015_two_component",
        "role": "historical baseline; target semantics not H-only",
        "path": RESULTS_ROOT
        / "results_md"
        / "md_dataset1_zzu9ln_sweep015_ep550_lr0p005_l3_c32_i3_corr2_b96_w4"
        / "run_20260518_154600",
    },
    {
        "method": "default_huber_0p01_honly_sweep015",
        "role": "corrected H-only default readout comparator",
        "path": RESULTS_ROOT
        / "results_md"
        / "md_dataset1_zzu9ln_sweep015_ep550_lr0p005_l3_c32_i3_corr2_b96_w4_s15_honly_huber_b0p01"
        / "run_20260520_095922",
    },
    {
        "method": "default_huber_0p003_honly_sweep015",
        "role": "corrected H-only beta comparator",
        "path": RESULTS_ROOT
        / "results_md"
        / "md_dataset1_zzu9ln_sweep015_ep550_lr0p005_l3_c32_i3_corr2_b96_w4_s15_honly_huber_b0p003"
        / "run_20260520_095922",
    },
    {
        "method": "default_huber_0p01_honly_strong_md1140_partial",
        "role": "additional partial archive without top-level manifest",
        "path": RESULTS_ROOT
        / "results_md"
        / "md_dataset5_0dwpgm_epochs450_seed0_huber_0p01_strong_l3_corr3_mace_4e30e5761f"
        / "run_20260519_175335",
    },
]

KNOWN_INCOMPLETE = [
    {
        "method": "default_huber_0p03_honly_sweep015",
        "status": "running_or_unarchived",
        "path": REPO_ROOT
        / "Comparison"
        / "workspaces"
        / "20260520_095922"
        / "md"
        / "md_dataset1_zzu9ln_sweep015_ep550_lr0p005_l3_c32_i3_corr2_b96_w4_s15_honly_huber_b0p03",
    },
    {
        "method": "default_mse_honly_sweep015",
        "status": "planned_not_archived",
        "path": RESULTS_ROOT
        / "results_md"
        / "md_dataset1_zzu9ln_sweep015_ep550_lr0p005_l3_c32_i3_corr2_b96_w4_s15_honly_mse",
    },
    {
        "method": "default_mae_honly_sweep015",
        "status": "planned_not_archived",
        "path": RESULTS_ROOT
        / "results_md"
        / "md_dataset1_zzu9ln_sweep015_ep550_lr0p005_l3_c32_i3_corr2_b96_w4_s15_honly_mae_baseline",
    },
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


def as_float(value: Any) -> float | None:
    if value in (None, "", "nan", "NaN", "None"):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def series(rows: list[dict[str, str]], column: str) -> list[float]:
    return [value for row in rows if (value := as_float(row.get(column))) is not None]


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


def summarize(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "std": None, "p90": None, "min": None, "max": None}
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "p90": percentile(values, 0.9),
        "min": min(values),
        "max": max(values),
    }


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, str):
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "NA"
    return f"{number:.{digits}f}"


def git_info(path: Path) -> dict[str, str | None]:
    def run(args: list[str]) -> str | None:
        completed = subprocess.run(
            ["git", "-C", str(path), *args],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            return None
        return completed.stdout.strip()

    return {
        "commit": run(["rev-parse", "HEAD"]),
        "branch": run(["rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": "yes" if run(["status", "--porcelain"]) else "no",
    }


def classify_loss(model: dict[str, Any], label: str) -> tuple[str, dict[str, Any]]:
    loss = str(model.get("loss") or "")
    kwargs = model.get("loss_kwargs") if isinstance(model.get("loss_kwargs"), dict) else {}
    if not loss and "huber_b0p01" in label:
        loss, kwargs = "graph2mat.core.data.metrics.block_type_huber", {"beta": 0.01}
    return loss, kwargs


def inspect_reference_validity(run_dir: Path, metrics_manifest: dict[str, Any]) -> dict[str, Any]:
    reference_root = run_dir / "siesta_hamiltonians"
    reference_files = [p for p in reference_root.rglob("*") if p.is_file()]
    forbidden = [str(p.relative_to(run_dir)) for p in reference_files if p.name == "ML_prediction.HSX"]
    allowed = [p for p in reference_files if p.suffix in {".TSHS", ".HSX"} and p.name != "ML_prediction.HSX"]
    return {
        "reference_policy": metrics_manifest.get("reference_selection_policy"),
        "reference_files": len(allowed),
        "forbidden_prediction_references": forbidden,
        "samples_failed": metrics_manifest.get("samples_failed"),
        "samples_compared": metrics_manifest.get("samples_compared"),
    }


def failure_summary(metrics_manifest: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    errors = metrics_manifest.get("fatal_errors")
    if not isinstance(errors, list):
        return counts
    for error in errors:
        if not isinstance(error, dict):
            continue
        kind = str(error.get("kind") or "unknown")
        counts[kind] = counts.get(kind, 0) + 1
    return counts


def analyze_run(entry: dict[str, Any]) -> dict[str, Any]:
    run_dir = Path(entry["path"])
    manifest = read_json(run_dir / "manifest.json")
    config = read_yaml(run_dir / "pipeline_config.yaml")
    metrics_manifest = read_json(run_dir / "metrics" / "manifest.json")
    failures = failure_summary(metrics_manifest)
    sparse = read_rows(run_dir / "metrics" / "sparse_metrics.csv")
    spectral = read_rows(run_dir / "metrics" / "spectral_metrics.csv")
    dos = read_rows(run_dir / "metrics" / "dos_metrics.csv")
    block_rows = read_rows(run_dir / "metrics" / "block_metrics.csv")
    orbital_summary = read_rows(run_dir / "metrics" / "orbital_pair_summary.csv")

    training = manifest.get("training_hyperparameters") if isinstance(manifest.get("training_hyperparameters"), dict) else {}
    if not training:
        training = config.get("training") if isinstance(config.get("training"), dict) else {}
    model = training.get("model") if isinstance(training.get("model"), dict) else {}
    data = training.get("data") if isinstance(training.get("data"), dict) else {}
    loss, loss_kwargs = classify_loss(model, str(entry["method"]))
    reference = inspect_reference_validity(run_dir, metrics_manifest)

    sparse_stats = {column: summarize(series(sparse, column)) for column in (
        "mae_ref_meV",
        "rmse_ref_meV",
        "relative_frobenius_union",
        "support_f1",
        "hermiticity_pred",
    )}
    spectral_stats = {column: summarize(series(spectral, column)) for column in (
        "global_rmse_eV",
        "low_energy_rmse_eV",
        "fermi_window_rmse_eV",
        "frontier_window_rmse_eV",
    )}
    dos_stats = {"dos_mae_500_fermi_window": summarize(series(dos, "dos_mae_500_fermi_window"))}

    worst_samples = sorted(
        (
            {
                "method": entry["method"],
                "sample": row.get("sample"),
                "mae_ref_meV": as_float(row.get("mae_ref_meV")),
                "rmse_ref_meV": as_float(row.get("rmse_ref_meV")),
                "relative_frobenius_union": as_float(row.get("relative_frobenius_union")),
            }
            for row in sparse
        ),
        key=lambda row: row["mae_ref_meV"] if row["mae_ref_meV"] is not None else -1.0,
        reverse=True,
    )[:10]
    worst_blocks = sorted(
        (
            {
                "method": entry["method"],
                "sample": row.get("sample"),
                "species_pair": f"{row.get('row_species')}-{row.get('col_species')}",
                "mae_union_meV": (as_float(row.get("mae_union_eV")) or 0.0) * 1000.0,
                "distance_ang": as_float(row.get("mean_distance_ang")),
            }
            for row in block_rows
            if as_float(row.get("mae_union_eV")) is not None
        ),
        key=lambda row: row["mae_union_meV"],
        reverse=True,
    )[:10]
    worst_orbitals = sorted(
        (
            {
                "method": entry["method"],
                "species_pair": row.get("species_pair"),
                "row_orbital": row.get("row_orbital_label"),
                "col_orbital": row.get("col_orbital_label"),
                "mae_union_meV_mean": as_float(row.get("mae_union_meV_mean")),
            }
            for row in orbital_summary
        ),
        key=lambda row: row["mae_union_meV_mean"] if row["mae_union_meV_mean"] is not None else -1.0,
        reverse=True,
    )[:10]

    n_components = data.get("n_matrix_components")
    h_only_valid = (
        data.get("out_matrix") == "hamiltonian"
        and data.get("matrix_component_policy", "h_only" if n_components == 1 else None) == "h_only"
        and int(n_components or 0) == 1
        and bool(data.get("symmetric_matrix")) is True
    )

    return {
        "method": entry["method"],
        "role": entry["role"],
        "run_dir": str(run_dir),
        "exists": run_dir.exists(),
        "has_manifest": (run_dir / "manifest.json").exists(),
        "has_metrics": bool(sparse),
        "returncode": manifest.get("returncode"),
        "run_id": manifest.get("run_id") or run_dir.name.removeprefix("run_"),
        "dataset_label": manifest.get("dataset_label") or run_dir.parent.name,
        "effective_dataset_size": manifest.get("effective_dataset_size"),
        "test_samples": len(sparse),
        "samples_compared": reference["samples_compared"],
        "samples_failed": reference["samples_failed"],
        "loss": loss,
        "loss_kwargs": loss_kwargs,
        "seed": training.get("seed_everything") or manifest.get("training_seed") or manifest.get("seed"),
        "out_matrix": data.get("out_matrix"),
        "matrix_component_policy": data.get("matrix_component_policy"),
        "n_matrix_components": n_components,
        "symmetric_matrix": data.get("symmetric_matrix"),
        "h_only_valid": h_only_valid,
        "reference_validity": reference,
        "failure_summary": failures,
        "pipeline_commit": manifest.get("pipeline_commit"),
        "graph2mat_commit": manifest.get("graph2mat_commit"),
        "sparse_stats": sparse_stats,
        "spectral_stats": spectral_stats,
        "dos_stats": dos_stats,
        "deeph_comparability_status": metrics_manifest.get("deeph_comparability_status"),
        "worst_samples": worst_samples,
        "worst_blocks": worst_blocks,
        "worst_orbitals": worst_orbitals,
        "train_test_gap_status": "unavailable: archive has held-out metrics but no train-set evaluation CSV",
    }


def aggregate_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        sparse = result["sparse_stats"]
        spectral = result["spectral_stats"]
        dos = result["dos_stats"]
        rows.append(
            {
                "method": result["method"],
                "role": result["role"],
                "run_id": result["run_id"],
                "dataset_label": result["dataset_label"],
                "test_samples": result["test_samples"],
                "h_only_valid": result["h_only_valid"],
                "loss": result["loss"],
                "loss_kwargs": json.dumps(result["loss_kwargs"], sort_keys=True),
                "h_mae_meV_mean": sparse["mae_ref_meV"]["mean"],
                "h_mae_meV_std": sparse["mae_ref_meV"]["std"],
                "h_mae_meV_p90": sparse["mae_ref_meV"]["p90"],
                "h_rmse_meV_mean": sparse["rmse_ref_meV"]["mean"],
                "relative_frobenius_union_mean": sparse["relative_frobenius_union"]["mean"],
                "support_f1_mean": sparse["support_f1"]["mean"],
                "hermiticity_pred_mean": sparse["hermiticity_pred"]["mean"],
                "global_rmse_eV_mean": spectral["global_rmse_eV"]["mean"],
                "low_energy_rmse_eV_mean": spectral["low_energy_rmse_eV"]["mean"],
                "fermi_window_rmse_eV_mean": spectral["fermi_window_rmse_eV"]["mean"],
                "frontier_window_rmse_eV_mean": spectral["frontier_window_rmse_eV"]["mean"],
                "dos_mae_500_fermi_window_mean": dos["dos_mae_500_fermi_window"]["mean"],
                "samples_failed": result["samples_failed"],
                "failure_summary": json.dumps(result["failure_summary"], sort_keys=True),
                "forbidden_prediction_references": len(result["reference_validity"]["forbidden_prediction_references"]),
                "pipeline_commit": result["pipeline_commit"],
                "graph2mat_commit": result["graph2mat_commit"],
                "run_dir": result["run_dir"],
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def md_table(rows: list[dict[str, Any]]) -> str:
    header = (
        "| method | H-only valid | failed | H MAE mean meV | H RMSE mean meV | "
        "low-energy RMSE eV | global RMSE eV | DOS window MAE | support F1 | notes |"
    )
    sep = "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"
    lines = [header, sep]
    for row in rows:
        lines.append(
            "| {method} | {honly} | {failed} | {mae} | {rmse} | {low} | {glob} | {dos} | {f1} | {role} |".format(
                method=row["method"],
                honly="yes" if row["h_only_valid"] else "no",
                failed=row["samples_failed"],
                mae=fmt(row["h_mae_meV_mean"]),
                rmse=fmt(row["h_rmse_meV_mean"]),
                low=fmt(row["low_energy_rmse_eV_mean"]),
                glob=fmt(row["global_rmse_eV_mean"]),
                dos=fmt(row["dos_mae_500_fermi_window_mean"]),
                f1=fmt(row["support_f1_mean"], 4),
                role=row["role"],
            )
        )
    return "\n".join(lines)


def best_by_metric(rows: list[dict[str, Any]], metric: str) -> dict[str, Any] | None:
    valid = [row for row in rows if row.get(metric) is not None]
    if not valid:
        return None
    return min(valid, key=lambda row: float(row[metric]))


def write_report(results: list[dict[str, Any]], rows: list[dict[str, Any]], git: dict[str, Any]) -> None:
    complete_honly = [row for row in rows if row["h_only_valid"] and row["h_mae_meV_mean"] is not None]
    best_h = best_by_metric(complete_honly, "h_mae_meV_mean")
    best_low = best_by_metric(complete_honly, "low_energy_rmse_eV_mean")
    best_dos = best_by_metric(complete_honly, "dos_mae_500_fermi_window_mean")
    phase6_outputs = list((RESULTS_ROOT / "results_md").glob("phase6_*/*"))
    missing_methods = [
        method for method in EXPECTED_METHODS
        if not any(method in row["method"] for row in rows)
    ]
    incomplete = [
        {
            **item,
            "exists": Path(item["path"]).exists(),
        }
        for item in KNOWN_INCOMPLETE
    ]
    worst_lines = []
    for result in results:
        if not result["has_metrics"]:
            continue
        worst = result["worst_samples"][:3]
        if not worst:
            continue
        bits = ", ".join(
            f"{item['sample']}={fmt(item['mae_ref_meV'])} meV"
            for item in worst
        )
        worst_lines.append(f"- `{result['method']}`: {bits}")
    orbital_lines = []
    for result in results:
        worst = result["worst_orbitals"][:3]
        if not worst:
            continue
        bits = ", ".join(
            f"{item['species_pair']} {item['row_orbital']}-{item['col_orbital']}={fmt(item['mae_union_meV_mean'])} meV"
            for item in worst
        )
        orbital_lines.append(f"- `{result['method']}`: {bits}")
    failure_lines = []
    for result in results:
        if result["failure_summary"]:
            failure_lines.append(
                f"- `{result['method']}`: "
                + ", ".join(f"{key}={value}" for key, value in sorted(result["failure_summary"].items()))
            )

    report = f"""# H2O Hamiltonian Architecture Reform Decision Report

Generated: {datetime.now().isoformat(timespec="seconds")}

## 1. Executive Verdict

**Verdict: C - do not promote the Graph2Mat Hamiltonian reform yet; use DeepH as the scientific backend for Hamiltonian accuracy work, while keeping Graph2Mat only as an experimental branch.**

Reason: the full architecture-reform matrix has not produced archived results in this repository. The only completed corrected H-only runs available here are default-readout Huber variants. The best completed H-only result remains far above a low-meV DeepH-style target: `{fmt(best_h['h_mae_meV_mean'] if best_h else None)}` meV H MAE and `{fmt(best_h['h_rmse_meV_mean'] if best_h else None)}` meV H RMSE on the held-out test set. New context/readout/staged/dense candidates cannot be promoted because no archived metrics exist for them.

## 2. Experiment Matrix

Expected phase-6 candidates:

{chr(10).join(f"- `{method}`" for method in EXPECTED_METHODS)}

Observed completed or partial result directories:

{chr(10).join(f"- `{row['method']}` -> `{row['run_dir']}`" for row in rows)}

Missing from archived results:

{chr(10).join(f"- `{method}`" for method in missing_methods) if missing_methods else "- none"}

Known incomplete/planned entries:

{chr(10).join(f"- `{item['method']}`: {item['status']}; exists={item['exists']}; path=`{item['path']}`" for item in incomplete)}

Phase-6 output directories found: `{len(phase6_outputs)}`.

## 3. Data And Reference Validity Check

- Completed H-only runs use `out_matrix=hamiltonian`, `matrix_component_policy=h_only`, `n_matrix_components=1`, and `symmetric_matrix=true`.
- The historical sweep015 baseline uses `n_matrix_components=2`, so it is not a valid corrected H-only benchmark. It is retained only as historical context.
- No archived reference directory inspected by this script contains `ML_prediction.HSX` under `siesta_hamiltonians`.
- Metrics manifests record strict single-reference policy where available.
- `pipeline_commit` and `graph2mat_commit` are missing in older manifests; this was added to the harness after these runs.

## 4. Aggregate Results Table

{md_table(rows)}

Full CSV: `aggregate_results.csv`.

## 5. Per-Sample Failure Analysis

Samples failed in completed metrics: see `samples_failed` in `aggregate_results.csv`; completed H-only Huber runs report zero failed samples in their metrics manifests.

Fatal/skipped failure summary:

{chr(10).join(failure_lines) if failure_lines else "- No fatal errors recorded in analyzed completed metrics."}

Worst samples by H MAE:

{chr(10).join(worst_lines) if worst_lines else "- No per-sample sparse metrics available."}

Worst orbital-pair residuals:

{chr(10).join(orbital_lines) if orbital_lines else "- No orbital-pair summaries available."}

Train/test gap: unavailable from archived outputs. The archives contain held-out test metrics and training logs, but not train-set Hamiltonian evaluation CSVs. Therefore this report cannot distinguish generalization gap from uniformly high error using train/test H metrics.

## 6. Best Config By Metric

- Best H MAE among completed H-only runs: `{best_h['method'] if best_h else 'NA'}` with `{fmt(best_h['h_mae_meV_mean'] if best_h else None)}` meV.
- Best low-energy spectral RMSE among completed H-only runs: `{best_low['method'] if best_low else 'NA'}` with `{fmt(best_low['low_energy_rmse_eV_mean'] if best_low else None)}` eV.
- Best DOS 500-Fermi-window MAE among completed H-only runs: `{best_dos['method'] if best_dos else 'NA'}` with `{fmt(best_dos['dos_mae_500_fermi_window_mean'] if best_dos else None)}`.

## 7. Comparison To DeepH Target Scale

Local repository docs mark these metrics as DeepH-comparable diagnostics, not exact DeepH H-prime reproduction. Caveats include raw global Hamiltonian metrics rather than DeepH local-coordinate H-prime blocks, and molecular H2O DOS not matching DeepH's original material examples.

External primary references put the target scale much lower than these runs: the original DeepH Nature Computational Science paper reports an example matrix-element MAE of 6.6 meV for a nearest-neighbor 1s element, and DeepH-E3/Nat. Commun. describes sub-meV Hamiltonian prediction accuracy in its benchmark setting. These are not one-to-one H2O comparisons, but they set the order-of-magnitude target. Current completed Graph2Mat H-only runs are still tens of meV.

Sources:

- DeepH original paper: https://www.nature.com/articles/s43588-022-00265-6
- DeepH-E3 paper: https://www.nature.com/articles/s41467-023-38468-8
- Local metric caveats: `Comparison/METRICS.md`

## 8. Recommendation

**Use DeepH as the backend for scientific Hamiltonian accuracy.**

Do not promote the Graph2Mat Hamiltonian architecture reform based on current archived evidence. The required new-architecture candidates are absent, and the completed default H-only Huber runs remain above the 50-100 meV warning band for some matrix/spectral criteria or remain far from low-meV targets. If Graph2Mat work continues, it should be a bounded architecture research task, not the production scientific path.

## 9. Risks And Caveats

- The phase-6 benchmark harness exists, but no phase-6 results are archived.
- Seed stability cannot be estimated: completed corrected H-only runs are single-seed variants, not three-seed candidate groups.
- Train/test gap cannot be estimated from available Hamiltonian CSVs.
- One-sample architecture checks are not present as local archived outputs under `Comparison/results`, so the <1 meV one-sample criterion cannot be verified here.
- One Huber beta=0.03 run is currently running or unarchived and is excluded from numeric decisions.
- The partial `md1140` run lacks a top-level manifest; it is treated as auxiliary evidence only.

## 10. Exact Commands And Results Used

Commands:

```bash
python3 -m py_compile Comparison/scripts/analyze_hamiltonian_architecture_reform.py
python3 Comparison/scripts/analyze_hamiltonian_architecture_reform.py
```

Repository state at analysis:

- pipeline commit: `{git['pipeline'].get('commit')}`
- pipeline branch: `{git['pipeline'].get('branch')}`
- pipeline dirty: `{git['pipeline'].get('dirty')}`

Generated files:

- `aggregate_results.csv`
- `per_sample_worst.csv`
- `block_worst.csv`
- `orbital_pair_worst.csv`
- `missing_and_incomplete.json`
- `final_decision_report.md`
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    results = [analyze_run(entry) for entry in MEASURED_RUNS if Path(entry["path"]).exists()]
    rows = aggregate_rows(results)
    write_csv(OUTPUT_ROOT / "aggregate_results.csv", rows)
    write_csv(
        OUTPUT_ROOT / "per_sample_worst.csv",
        [item for result in results for item in result["worst_samples"]],
    )
    write_csv(
        OUTPUT_ROOT / "block_worst.csv",
        [item for result in results for item in result["worst_blocks"]],
    )
    write_csv(
        OUTPUT_ROOT / "orbital_pair_worst.csv",
        [item for result in results for item in result["worst_orbitals"]],
    )
    incomplete_payload = {
        "expected_methods": EXPECTED_METHODS,
        "known_incomplete": [
            {**item, "path": str(item["path"]), "exists": Path(item["path"]).exists()}
            for item in KNOWN_INCOMPLETE
        ],
        "phase6_output_dirs": [str(path) for path in (RESULTS_ROOT / "results_md").glob("phase6_*/*")],
    }
    (OUTPUT_ROOT / "missing_and_incomplete.json").write_text(
        json.dumps(incomplete_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    git = {"pipeline": git_info(REPO_ROOT)}
    write_report(results, rows, git)
    print(f"Wrote {REPORT_PATH}")
    print(f"Analyzed {len(results)} result directories")


if __name__ == "__main__":
    main()
