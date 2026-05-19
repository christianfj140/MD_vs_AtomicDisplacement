#!/usr/bin/env python3
"""Write the staged H2O Hamiltonian loss-hardening experiment plan.

The plan is intentionally explicit rather than changing Comparison defaults.
It enumerates the loss/readout combinations needed to test whether the
MAE/L1 objective geometry is the original H2O overfit bottleneck.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "Comparison"
    / "results"
    / "diagnostics"
    / "h2o_hamiltonian"
    / "hamiltonian_loss_hardening"
)

STAGES = [
    {
        "stage": "one_sample",
        "dataset_family": "atom_displacement",
        "train_n": 1,
        "val_n": 1,
        "test_n": 1,
        "purpose": "sanity overfit; train/test may intentionally point at the same structure",
    },
    {
        "stage": "five_samples",
        "dataset_family": "atom_displacement",
        "train_n": 5,
        "val_n": 2,
        "test_n": 2,
        "purpose": "tiny local generalization probe",
    },
    {
        "stage": "twenty_samples",
        "dataset_family": "atom_displacement",
        "train_n": 20,
        "val_n": 5,
        "test_n": 5,
        "purpose": "small but nontrivial AtomicDisplacement probe",
    },
    {
        "stage": "small_md_dataset",
        "dataset_family": "md",
        "train_n": 20,
        "val_n": 5,
        "test_n": 5,
        "purpose": "nearby MD structures with temporal/geometric correlation",
    },
    {
        "stage": "small_atomic_displacement_dataset",
        "dataset_family": "atom_displacement",
        "train_n": 40,
        "val_n": 10,
        "test_n": 10,
        "purpose": "small production-like AtomicDisplacement comparison",
    },
]

METHODS = [
    {
        "method": "default_readout_block_type_mae",
        "readout": "default",
        "node_block_readout": "graph2mat.bindings.e3nn.E3nnSimpleNodeBlock",
        "edge_block_readout": "graph2mat.bindings.e3nn.E3nnSimpleEdgeBlock",
        "loss": "graph2mat.metrics.block_type_mae",
        "loss_kwargs": {},
        "return_coefficients": False,
        "production_candidate": False,
        "notes": "current H2O baseline; retained for compatibility",
    },
    {
        "method": "default_readout_block_type_mse",
        "readout": "default",
        "node_block_readout": "graph2mat.bindings.e3nn.E3nnSimpleNodeBlock",
        "edge_block_readout": "graph2mat.bindings.e3nn.E3nnSimpleEdgeBlock",
        "loss": "graph2mat.metrics.block_type_mse",
        "loss_kwargs": {},
        "return_coefficients": False,
        "production_candidate": True,
        "notes": "candidate Hamiltonian objective after coefficient-space audit",
    },
    {
        "method": "default_readout_block_type_huber_beta_0p01",
        "readout": "default",
        "node_block_readout": "graph2mat.bindings.e3nn.E3nnSimpleNodeBlock",
        "edge_block_readout": "graph2mat.bindings.e3nn.E3nnSimpleEdgeBlock",
        "loss": "graph2mat.metrics.block_type_huber",
        "loss_kwargs": {"beta": 0.01},
        "return_coefficients": False,
        "production_candidate": True,
        "notes": "candidate compromise between MAE robustness and MSE conditioning",
    },
    {
        "method": "dense_readout_block_type_mae",
        "readout": "diagnostic_dense",
        "node_block_readout": "graph2mat.bindings.e3nn.E3nnDiagnosticDenseNodeBlock",
        "edge_block_readout": "graph2mat.bindings.e3nn.E3nnDiagnosticDenseEdgeBlock",
        "loss": "graph2mat.metrics.block_type_mae",
        "loss_kwargs": {},
        "return_coefficients": False,
        "production_candidate": False,
        "notes": "non-equivariant diagnostic control",
    },
    {
        "method": "dense_readout_block_type_mse",
        "readout": "diagnostic_dense",
        "node_block_readout": "graph2mat.bindings.e3nn.E3nnDiagnosticDenseNodeBlock",
        "edge_block_readout": "graph2mat.bindings.e3nn.E3nnDiagnosticDenseEdgeBlock",
        "loss": "graph2mat.metrics.block_type_mse",
        "loss_kwargs": {},
        "return_coefficients": False,
        "production_candidate": False,
        "notes": "non-equivariant diagnostic control that should overfit if objective is the blocker",
    },
    {
        "method": "default_readout_coefficient_space_mse",
        "readout": "default",
        "node_block_readout": "graph2mat.bindings.e3nn.E3nnSimpleNodeBlock",
        "edge_block_readout": "graph2mat.bindings.e3nn.E3nnSimpleEdgeBlock",
        "loss": "graph2mat.metrics.coefficient_space_mse",
        "loss_kwargs": {},
        "return_coefficients": True,
        "production_candidate": False,
        "notes": "experimental opt-in diagnostic; requires return_coefficients=True",
    },
]

METRICS = [
    "train_loss",
    "val_loss",
    "mae_union_meV",
    "rmse_union_meV",
    "relative_frobenius_union",
    "fermi_window_rmse_eV",
    "dos_mae_500_fermi_window",
    "support_f1",
    "sparse_threshold_sweep",
]


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def experiment_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stage in STAGES:
        for method in METHODS:
            rows.append({**stage, **method, "status": "planned"})
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "stage",
        "dataset_family",
        "train_n",
        "val_n",
        "test_n",
        "method",
        "readout",
        "loss",
        "loss_kwargs",
        "return_coefficients",
        "node_block_readout",
        "edge_block_readout",
        "production_candidate",
        "status",
        "purpose",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            encoded = dict(row)
            encoded["loss_kwargs"] = json.dumps(row["loss_kwargs"], sort_keys=True)
            writer.writerow(encoded)


def write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Hamiltonian Loss Hardening Experiment Plan",
        "",
        "This plan turns the coefficient-space loss audit into a staged, opt-in",
        "Comparison experiment. It does not change production defaults.",
        "",
        "## Loss Audit",
        "",
        "- `block_type_mae`: mean absolute node error plus mean absolute edge error.",
        "- `block_type_mse`: mean squared node error plus mean squared edge error.",
        "- `block_type_huber`: configurable Smooth L1/Huber node loss plus edge loss.",
        "- `coefficient_space_mse`: experimental diagnostic comparing irreps coefficients;",
        "  it requires `return_coefficients=True` and is not a default candidate yet.",
        "",
        "## Support Metric Audit",
        "",
        "The canonical sparse support threshold in `evaluate_hamiltonian_metrics.py` is",
        "`1e-12`. That is a numerical/container sparsity diagnostic, not a physical",
        "Hamiltonian significance threshold. Treat `support_f1` as a fail-closed",
        "pattern check and inspect `sparse_threshold_sweep.csv` at `1e-12`, `1e-10`,",
        "`1e-8`, and `1e-6` before drawing physical conclusions.",
        "",
        "## Experiment Matrix",
        "",
        "| stage | family | train/val/test | method | loss | readout | status |",
        "|---|---|---:|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['stage']}` | `{row['dataset_family']}` | "
            f"{row['train_n']}/{row['val_n']}/{row['test_n']} | "
            f"`{row['method']}` | `{row['loss']}` | `{row['readout']}` | `{row['status']}` |"
        )
    lines.extend(
        [
            "",
            "## Metrics To Collect",
            "",
            *[f"- `{metric}`" for metric in METRICS],
            "",
            "## Interpretation Gate",
            "",
            "- If `block_type_mse` and/or `block_type_huber` close the train/test H gap",
            "  while `block_type_mae` remains worse, the original issue is MAE/L1",
            "  objective geometry for Hamiltonian targets.",
            "- If dense readout helps but default readout does not, the remaining issue is",
            "  readout/feature capacity.",
            "- If coefficient-space MSE helps while label-space MSE does not, revisit the",
            "  label/coefficient projection. The current one-sample audit did not show",
            "  that mismatch.",
            "",
            "## Output Files",
            "",
            "- `experiment_matrix.csv`: concrete stage x method grid.",
            "- `experiment_plan.json`: machine-readable plan with config fragments.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = experiment_rows()
    payload = {
        "status": "planned_not_executed",
        "output_dir": str(output_dir),
        "stages": STAGES,
        "methods": METHODS,
        "metrics": METRICS,
        "rows": rows,
    }
    (output_dir / "experiment_plan.json").write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(output_dir / "experiment_matrix.csv", rows)
    write_report(output_dir / "report.md", rows)
    print(f"Wrote {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
