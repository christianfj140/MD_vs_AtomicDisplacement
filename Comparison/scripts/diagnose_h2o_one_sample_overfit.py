#!/usr/bin/env python3
"""Prepare or run an isolated one-sample H2O H-only overfit diagnostic."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIAGNOSTIC_ROOT = (
    REPO_ROOT
    / "Comparison"
    / "results"
    / "diagnostics"
    / "h2o_hamiltonian"
    / "one_sample_overfit"
)
DEFAULT_BASE_CONFIG = REPO_ROOT / "configs" / "config_md.yaml"
PRODUCTION_OUTPUT_ROOTS = (
    REPO_ROOT / "Comparison" / "results" / "results_md",
    REPO_ROOT / "Comparison" / "results" / "results_atomdisp",
    REPO_ROOT / "Comparison" / "results" / "results_random_cartesian",
    REPO_ROOT / "Comparison" / "workspaces",
    REPO_ROOT / "MD" / "dataset",
    REPO_ROOT / "MD" / "training",
    REPO_ROOT / "AtomDisplacement" / "dataset",
    REPO_ROOT / "AtomDisplacement" / "training",
)


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"Base Graph2Mat config does not exist: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Base Graph2Mat config must be a mapping: {path}")
    return payload


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, default_flow_style=False, allow_unicode=False),
        encoding="utf-8",
    )


def resolve_workspace(path: Path | None, sample: str) -> Path:
    if path is not None:
        return path.resolve()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    safe_sample = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in sample)
    return (DEFAULT_DIAGNOSTIC_ROOT / f"run_{stamp}_{safe_sample}").resolve()


def ensure_isolated_workspace(workspace: Path, *, overwrite: bool, allow_non_diagnostic_output: bool) -> None:
    resolved = workspace.resolve()
    if not allow_non_diagnostic_output:
        for root in PRODUCTION_OUTPUT_ROOTS:
            try:
                resolved.relative_to(root.resolve())
            except ValueError:
                continue
            raise RuntimeError(
                "Refusing to write one-sample diagnostic under production output root "
                f"{root}. Use --allow-non-diagnostic-output only if you really intend this."
            )
    if resolved.exists() and any(resolved.iterdir()):
        if not overwrite:
            raise RuntimeError(
                f"Diagnostic workspace already exists and is not empty: {resolved}. "
                "Use --overwrite to replace an existing diagnostic workspace."
            )
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)


def find_reference_matrix(sample_dir: Path) -> Path | None:
    candidates = sorted(sample_dir.glob("*.TSHS")) + sorted(sample_dir.glob("*.HSX"))
    candidates = [path for path in candidates if path.name != "ML_prediction.HSX"]
    return candidates[0] if candidates else None


def resolve_sample_inputs(evaluation_root: Path, sample: str) -> dict[str, Path]:
    root = evaluation_root.resolve()
    structure = root / "structures" / sample / "RUN.fdf"
    reference = find_reference_matrix(root / "siesta_hamiltonians" / sample)
    missing: list[str] = []
    if not structure.exists():
        missing.append(f"structures/{sample}/RUN.fdf")
    if reference is None or not reference.exists():
        missing.append(f"siesta_hamiltonians/{sample}/*.TSHS|*.HSX")
    if missing:
        raise RuntimeError(
            f"Missing required one-sample inputs under {root}: " + ", ".join(missing)
        )
    return {"structure": structure, "reference": reference}


def resolve_basis_files(evaluation_root: Path, basis_glob: str | None) -> list[Path]:
    patterns: list[str] = []
    if basis_glob:
        patterns.append(basis_glob)
    else:
        patterns.extend(
            [
                str(evaluation_root / "basis" / "*.ion.xml"),
                str(evaluation_root / "material_basis" / "*.ion.xml"),
                str(evaluation_root / "structures" / "basis" / "*.ion.xml"),
            ]
        )
    matches: list[Path] = []
    for pattern in patterns:
        pattern_path = Path(pattern)
        if pattern_path.is_absolute():
            expanded = list(pattern_path.parent.glob(pattern_path.name))
        else:
            expanded = list(REPO_ROOT.glob(pattern))
        matches.extend(path.resolve() for path in expanded if path.is_file())
    unique = sorted({path for path in matches})
    if not unique:
        source = basis_glob or "default basis search under evaluation root"
        raise RuntimeError(f"No basis .ion.xml files found for {source}")
    return unique


def copy_side_inputs(source_dir: Path, destination_dir: Path) -> None:
    for pattern in ("*.psf", "*.ion", "*.ion.xml", "*.XV", "*.STRUCT_OUT", "*.ORB_INDX"):
        for source in source_dir.glob(pattern):
            if source.name == "ML_prediction.HSX":
                continue
            destination = destination_dir / source.name
            if not destination.exists():
                shutil.copy2(source, destination)


def prepare_split_sample(
    *,
    split_dir: Path,
    sample: str,
    structure: Path,
    reference: Path,
) -> Path:
    destination = split_dir / sample
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(structure, destination / "RUN.fdf")
    shutil.copy2(reference, destination / reference.name)
    copy_side_inputs(structure.parent, destination)
    copy_side_inputs(reference.parent, destination)
    return destination / "RUN.fdf"


def copy_basis_files(basis_files: list[Path], workspace: Path) -> list[Path]:
    basis_dir = workspace / "basis"
    basis_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for source in basis_files:
        destination = basis_dir / source.name
        shutil.copy2(source, destination)
        copied.append(destination)
    return copied


def h_only_config(
    base_config: dict[str, Any],
    *,
    workspace: Path,
    max_epochs: int,
    accelerator: str,
    seed: int,
) -> dict[str, Any]:
    config = dict(base_config)
    data = dict(config.get("data") or {})
    model = dict(config.get("model") or {})
    trainer = dict(config.get("trainer") or {})
    logger = dict(trainer.get("logger") or {})
    logger_args = dict(logger.get("init_args") or {})

    data.update(
        {
            "out_matrix": "hamiltonian",
            "matrix_component_policy": "h_only",
            "n_matrix_components": 1,
            "symmetric_matrix": True,
            "sub_point_matrix": False,
            "batch_size": 1,
            "store_in_memory": True,
            "basis_files": "../basis/*.ion.xml",
            "train_runs": "../dataset/splits/train/*/RUN.fdf",
            "val_runs": "../dataset/splits/validation/*/RUN.fdf",
            "test_runs": "../dataset/splits/test/*/RUN.fdf",
        }
    )
    trainer.update({"max_epochs": int(max_epochs), "accelerator": accelerator})
    logger_args.update({"name": "one_sample_h_only", "save_dir": "lightning_logs"})
    logger.update({"class_path": logger.get("class_path", "TensorBoardLogger"), "init_args": logger_args})
    trainer["logger"] = logger
    config["data"] = data
    config["model"] = model
    config["trainer"] = trainer
    config["seed_everything"] = int(seed)
    config["diagnostic"] = {
        "kind": "h2o_one_sample_overfit",
        "acceptance_threshold_is_diagnostic_not_publication_metric": True,
    }
    return config


def validate_h_only_config(config: dict[str, Any]) -> None:
    data = config.get("data")
    if not isinstance(data, dict):
        raise RuntimeError("Generated config is missing data section")
    expected = {
        "out_matrix": "hamiltonian",
        "matrix_component_policy": "h_only",
        "n_matrix_components": 1,
        "symmetric_matrix": True,
        "batch_size": 1,
    }
    mismatches = {
        key: {"expected": value, "actual": data.get(key)}
        for key, value in expected.items()
        if data.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Generated one-sample config is not H-only safe: {mismatches}")


def write_manifest_csv(path: Path, sample: str, structure_path: Path, reference_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "structure_path", "reference_path"])
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": sample,
                "structure_path": str(structure_path),
                "reference_path": str(reference_path),
            }
        )


def latest_checkpoint(training_dir: Path) -> Path:
    candidates = sorted(
        list(training_dir.glob("lightning_logs/**/checkpoints/best-*.ckpt"))
        + list(training_dir.glob("lightning_logs/**/checkpoints/last.ckpt")),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise RuntimeError(f"No checkpoint found under {training_dir / 'lightning_logs'}")
    return candidates[-1]


def run_checked(command: list[str], *, cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n\n")
        completed = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            text=True,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with return code {completed.returncode}: {' '.join(command)}")


def read_metric_summary(evaluation_dir: Path) -> dict[str, Any]:
    sparse_path = evaluation_dir / "metrics" / "sparse_metrics.csv"
    spectral_path = evaluation_dir / "metrics" / "spectral_metrics.csv"
    summary: dict[str, Any] = {
        "sparse_metrics_csv": str(sparse_path),
        "spectral_metrics_csv": str(spectral_path),
    }
    if sparse_path.exists():
        rows = list(csv.DictReader(sparse_path.open(encoding="utf-8")))
        if rows:
            row = rows[0]
            for key in ("mae_union_eV", "rmse_union_eV", "relative_frobenius_union", "support_f1"):
                if key in row:
                    try:
                        summary[key] = float(row[key])
                    except ValueError:
                        summary[key] = row[key]
    if spectral_path.exists():
        rows = list(csv.DictReader(spectral_path.open(encoding="utf-8")))
        if rows:
            row = rows[0]
            for key in ("global_rmse_eV", "low_energy_rmse_eV", "fermi_window_rmse_eV"):
                if key in row:
                    try:
                        summary[key] = float(row[key])
                    except ValueError:
                        summary[key] = row[key]
    return summary


def write_report(path: Path, payload: dict[str, Any]) -> None:
    metrics = payload.get("metrics", {})
    direct = payload.get("direct_label_diagnostic") if isinstance(payload.get("direct_label_diagnostic"), dict) else {}
    node = direct.get("node_labels") if isinstance(direct.get("node_labels"), dict) else {}
    edge = direct.get("edge_labels") if isinstance(direct.get("edge_labels"), dict) else {}
    lines = [
        "# One-Sample H2O H-only Overfit Diagnostic",
        "",
        f"Status: **{payload['status']}**",
        f"Workspace: `{payload['workspace']}`",
        f"Sample: `{payload['sample']}`",
        "",
        "## Purpose",
        "",
        "This is a minimal reproducible diagnostic, not a publication metric. Passing means the "
        "end-to-end H-only label/model/reconstruction/evaluation path can memorize one sample.",
        "",
        "## H-only Target Policy",
        "",
        "- `out_matrix`: `hamiltonian`",
        "- `matrix_component_policy`: `h_only`",
        "- `n_matrix_components`: `1`",
        "- `batch_size`: `1`",
        "",
        "## Inputs",
        "",
        f"- Structure: `{payload['inputs']['structure']}`",
        f"- Reference: `{payload['inputs']['reference']}`",
        f"- Basis files: {len(payload['inputs']['basis_files'])}",
        "",
        "## Commands",
        "",
        "```bash",
        *(" ".join(command) for command in payload["commands"].values()),
        "```",
        "",
        "## Metrics",
        "",
        "```json",
        json.dumps(metrics, indent=2, sort_keys=True),
        "```",
        "",
        "## Direct H-only Node/Edge Label Fit",
        "",
        "These values come from `diagnose_h2o_overfit_direct_labels.py` when the real",
        "`--execute` path is run. They compare `model(batch)` directly against the",
        "single training batch before MatrixWriter reconstruction.",
        "",
        "| target | MAE | RMSE | max abs |",
        "|---|---:|---:|---:|",
        f"| node H labels | {node.get('mae')} | {node.get('rmse')} | {node.get('max_abs')} |",
        f"| edge H labels | {edge.get('mae')} | {edge.get('rmse')} | {edge.get('max_abs')} |",
        "",
        "## Acceptance",
        "",
        f"Diagnostic target H MAE: `< {payload['acceptance']['h_mae_meV_threshold']} meV`.",
        "This threshold is a bug-finding target, not a formal scientific benchmark.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_workspace(args: argparse.Namespace) -> dict[str, Any]:
    evaluation_root = args.evaluation_root.resolve()
    if not evaluation_root.exists():
        raise RuntimeError(f"Evaluation root does not exist: {evaluation_root}")
    workspace = resolve_workspace(args.workspace, args.sample)
    ensure_isolated_workspace(
        workspace,
        overwrite=args.overwrite,
        allow_non_diagnostic_output=args.allow_non_diagnostic_output,
    )

    inputs = resolve_sample_inputs(evaluation_root, args.sample)
    basis_files = resolve_basis_files(evaluation_root, args.basis_glob)
    copied_basis = copy_basis_files(basis_files, workspace)

    split_paths: dict[str, str] = {}
    for split in ("train", "validation", "test", "predict"):
        run_fdf = prepare_split_sample(
            split_dir=workspace / "dataset" / "splits" / split,
            sample=args.sample,
            structure=inputs["structure"],
            reference=inputs["reference"],
        )
        split_paths[split] = str(run_fdf)

    evaluation_dir = workspace / "evaluation"
    eval_structure = prepare_split_sample(
        split_dir=evaluation_dir / "structures",
        sample=args.sample,
        structure=inputs["structure"],
        reference=inputs["reference"],
    )
    reference_dir = evaluation_dir / "siesta_hamiltonians" / args.sample
    reference_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(inputs["reference"], reference_dir / inputs["reference"].name)

    training_dir = workspace / "training"
    training_dir.mkdir(parents=True, exist_ok=True)
    config = h_only_config(
        load_yaml(args.base_config.resolve()),
        workspace=workspace,
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        seed=args.seed,
    )
    validate_h_only_config(config)
    config_path = training_dir / "config.yaml"
    write_yaml(config_path, config)

    test_manifest = workspace / "test_manifest.csv"
    write_manifest_csv(test_manifest, args.sample, eval_structure, reference_dir / inputs["reference"].name)

    train_command = [args.graph2mat_command, "models", "mace", "main", "fit", "-c", "config.yaml"]
    predict_command = [
        sys.executable,
        str(REPO_ROOT / "Comparison" / "scripts" / "predict_model_on_dataset.py"),
        "--checkpoint",
        "<best-checkpoint>",
        "--train-method",
        "md",
        "--test-set",
        "one_sample_overfit",
        "--test-manifest",
        str(test_manifest),
        "--basis-files",
        str(workspace / "basis" / "*.ion.xml"),
        "--output-dir",
        str(evaluation_dir),
        "--out-matrix",
        "hamiltonian",
        "--symmetric-matrix",
        "--no-sub-point-matrix",
        "--store-in-memory",
        "--accelerator",
        args.accelerator,
        "--matrix-component-policy",
        "h_only",
        "--n-matrix-components",
        "1",
    ]
    evaluate_command = [
        sys.executable,
        str(REPO_ROOT / "Comparison" / "scripts" / "evaluate_hamiltonian_metrics.py"),
        str(evaluation_dir),
        "--workers",
        "1",
    ]
    direct_label_command = [
        sys.executable,
        str(REPO_ROOT / "Comparison" / "scripts" / "diagnose_h2o_overfit_direct_labels.py"),
        "--workspace",
        str(workspace),
        "--training-dir",
        str(training_dir),
        "--config",
        str(config_path),
        "--checkpoint",
        "<best-checkpoint>",
        "--output-dir",
        str(workspace / "direct_label_diagnostic"),
    ]
    payload = {
        "status": "prepared",
        "dry_run": not args.execute,
        "workspace": str(workspace),
        "sample": args.sample,
        "created_at_unix": time.time(),
        "command": " ".join([sys.executable, *sys.argv]),
        "inputs": {
            "evaluation_root": str(evaluation_root),
            "structure": str(inputs["structure"]),
            "structure_sha256": sha256_file(inputs["structure"]),
            "reference": str(inputs["reference"]),
            "reference_sha256": sha256_file(inputs["reference"]),
            "basis_files": [str(path) for path in copied_basis],
            "basis_sha256": {path.name: sha256_file(path) for path in copied_basis},
        },
        "config": {
            "path": str(config_path),
            "base_config": str(args.base_config.resolve()),
            "max_epochs": args.max_epochs,
            "accelerator": args.accelerator,
            "seed": args.seed,
            "h_only_target_policy": {
                "out_matrix": "hamiltonian",
                "matrix_component_policy": "h_only",
                "n_matrix_components": 1,
                "symmetric_matrix": True,
                "batch_size": 1,
            },
        },
        "paths": {
            "training_dir": str(training_dir),
            "evaluation_dir": str(evaluation_dir),
            "test_manifest": str(test_manifest),
            "split_run_fdfs": split_paths,
        },
        "commands": {
            "train": train_command,
            "direct_label_diagnostic_template": direct_label_command,
            "predict_template": predict_command,
            "evaluate": evaluate_command,
        },
        "acceptance": {
            "h_mae_meV_threshold": args.acceptance_mae_mev,
            "threshold_role": "diagnostic_bug_finding_target_not_publication_metric",
        },
        "metrics": {},
    }
    (workspace / "provenance.json").write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_report(workspace / "report.md", payload)
    return payload


def execute_diagnostic(payload: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    workspace = Path(payload["workspace"])
    training_dir = Path(payload["paths"]["training_dir"])
    evaluation_dir = Path(payload["paths"]["evaluation_dir"])
    logs_dir = workspace / "logs"
    run_checked(payload["commands"]["train"], cwd=training_dir, log_path=logs_dir / "train.log")
    checkpoint = latest_checkpoint(training_dir)
    direct_label_command = [
        str(checkpoint) if item == "<best-checkpoint>" else item
        for item in payload["commands"]["direct_label_diagnostic_template"]
    ]
    payload["commands"]["direct_label_diagnostic"] = direct_label_command
    run_checked(
        direct_label_command,
        cwd=workspace,
        log_path=logs_dir / "direct_label_diagnostic.log",
    )
    predict_command = [
        str(checkpoint) if item == "<best-checkpoint>" else item
        for item in payload["commands"]["predict_template"]
    ]
    payload["commands"]["predict"] = predict_command
    payload["checkpoint"] = str(checkpoint)
    run_checked(predict_command, cwd=workspace, log_path=logs_dir / "predict.log")
    run_checked(payload["commands"]["evaluate"], cwd=workspace, log_path=logs_dir / "evaluate.log")
    payload["metrics"] = read_metric_summary(evaluation_dir)
    direct_label_json = workspace / "direct_label_diagnostic" / "direct_label_diagnostic.json"
    if direct_label_json.exists():
        payload["direct_label_diagnostic"] = json.loads(direct_label_json.read_text(encoding="utf-8"))
    mae_eV = payload["metrics"].get("mae_union_eV")
    pass_threshold = False
    if isinstance(mae_eV, float):
        pass_threshold = mae_eV * 1000.0 < float(args.acceptance_mae_mev)
    payload["status"] = "PASS" if pass_threshold else "FAIL"
    payload["dry_run"] = False
    (workspace / "provenance.json").write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_report(workspace / "report.md", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--basis-glob", default=None)
    parser.add_argument("--workspace", type=Path, default=None)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--max-epochs", type=int, default=1500)
    parser.add_argument("--accelerator", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--graph2mat-command", default="graph2mat")
    parser.add_argument("--acceptance-mae-mev", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true", help="Prepare files only; this is the default.")
    parser.add_argument("--execute", action="store_true", help="Actually run fit, predict, and evaluate.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-non-diagnostic-output", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.execute and args.dry_run:
        raise RuntimeError("Use either --execute or --dry-run, not both.")
    payload = prepare_workspace(args)
    if args.execute:
        payload = execute_diagnostic(payload, args)
    else:
        payload["status"] = "DRY_RUN"
        workspace = Path(payload["workspace"])
        (workspace / "provenance.json").write_text(
            json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        write_report(workspace / "report.md", payload)
    print(json.dumps(json_safe({"workspace": payload["workspace"], "status": payload["status"]}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
