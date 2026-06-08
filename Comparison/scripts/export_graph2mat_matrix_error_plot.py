#!/usr/bin/env python3
"""Export Graph2Mat PlotMatrixError figures with audit logs.

The script offers two execution modes:

* ``cli``: wraps the official ``graph2mat models mace main test`` command with
  ``PlotMatrixError`` and ``SamplewiseMetricsLogger`` callbacks.
* ``programmatic``: imports Graph2Mat, subclasses ``PlotMatrixError`` and saves
  the generated Plotly figures as HTML/PNG/PDF when supported by the local
  backend.

No Graph2Mat source is modified. In both modes stdout/stderr and an auditable
manifest are written under ``--output-dir``.
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_NAME = "Comparison/scripts/export_graph2mat_matrix_error_plot.py"
TORCH_COMPAT_DIR = REPO_ROOT / "scripts" / "torch_serialization_compat"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def optional_file_sha256(path: Path | None) -> str | None:
    if path is None or not path.exists() or not path.is_file():
        return None
    return sha256_file(path)


def split_shell_words(value: str) -> list[str]:
    parts = shlex.split(value)
    return parts or [value]


def resolve_test_run_inputs(patterns: list[str]) -> list[Path]:
    matches: list[Path] = []
    for pattern in patterns:
        expanded = [Path(item) for item in sorted(glob.glob(pattern))] if any(ch in pattern for ch in "*?[]") else []
        if expanded:
            matches.extend(expanded)
            continue
        path = Path(pattern)
        if path.exists():
            matches.append(path)
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in matches:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def compact_hash_summary(paths: list[Path], *, max_entries: int = 25) -> dict[str, Any]:
    items = []
    for path in paths[:max_entries]:
        items.append({"path": str(path), "sha256": optional_file_sha256(path)})
    return {"count": len(paths), "entries": items, "truncated": len(paths) > max_entries}


def graph2mat_version() -> str | None:
    try:
        import importlib.metadata

        return importlib.metadata.version("graph2mat")
    except Exception:
        return None


def package_version(package: str) -> str | None:
    try:
        import importlib.metadata

        return importlib.metadata.version(package)
    except Exception:
        return None


def torch_compat_env(base: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ if base is None else base)
    pythonpath = env.get("PYTHONPATH")
    compat = str(TORCH_COMPAT_DIR)
    env["PYTHONPATH"] = compat if not pythonpath else os.pathsep.join([compat, pythonpath])
    return env


def str_bool(value: bool) -> str:
    return "true" if value else "false"


@dataclass
class Graph2MatCommand:
    command: list[str]
    warnings: list[str]


def build_graph2mat_test_command(args: argparse.Namespace, *, sample_metrics_csv: Path) -> Graph2MatCommand:
    command: list[str] = [
        *split_shell_words(args.graph2mat_command),
        "models",
        "mace",
        "main",
        "test",
    ]
    if args.config_yaml is not None:
        command.extend(["--config", str(args.config_yaml)])
    command.extend(["--ckpt_path", str(args.ckpt_path)])
    for pattern in args.test_runs:
        command.extend(["--data.test_runs", pattern])
    if args.out_matrix is not None:
        command.extend(["--data.out_matrix", args.out_matrix])
    if args.basis_files is not None:
        command.extend(["--data.basis_files", args.basis_files])
    if args.symmetric_matrix is not None:
        command.extend(["--data.symmetric_matrix", str_bool(args.symmetric_matrix)])
    if args.sub_point_matrix is not None:
        command.extend(["--data.sub_point_matrix", str_bool(args.sub_point_matrix)])
    if args.batch_size is not None:
        command.extend(["--data.batch_size", str(args.batch_size)])
    if args.store_in_memory is not None:
        command.extend(["--data.store_in_memory", str_bool(args.store_in_memory)])
    if args.accelerator is not None:
        command.extend(["--trainer.accelerator", args.accelerator])

    warnings: list[str] = []
    if args.plot_matrix_error:
        command.extend(
            [
                "--trainer.callbacks+",
                "PlotMatrixError",
                "--trainer.callbacks.split",
                args.split,
                "--trainer.callbacks.show",
                str_bool(args.show),
                "--trainer.callbacks.store_in_logger",
                str_bool(args.store_in_logger),
            ]
        )
    if args.samplewise_metrics:
        command.extend(
            [
                "--trainer.callbacks+",
                "SamplewiseMetricsLogger",
                "--trainer.callbacks.splits",
                args.split,
                "--trainer.callbacks.output_file",
                str(sample_metrics_csv),
            ]
        )
    if args.mode == "cli" and (args.save_png or args.save_pdf or args.save_html):
        warnings.append(
            "Direct PNG/PDF/HTML export is only guaranteed in --mode programmatic; "
            "CLI mode reproduces the official callback and stores figures through the logger/browser."
        )
    return Graph2MatCommand(command=command, warnings=warnings)


def validate_inputs(args: argparse.Namespace) -> list[Path]:
    if not args.ckpt_path.exists():
        raise RuntimeError(f"Checkpoint not found: {args.ckpt_path}")
    if args.config_yaml is not None and not args.config_yaml.exists():
        raise RuntimeError(f"Config YAML not found: {args.config_yaml}")
    test_runs = resolve_test_run_inputs(args.test_runs)
    if not test_runs:
        raise RuntimeError(f"--test-runs did not match any files: {args.test_runs}")
    return test_runs


def base_manifest(args: argparse.Namespace, test_run_paths: list[Path], *, status: str) -> dict[str, Any]:
    return {
        "script": SCRIPT_NAME,
        "mode": args.mode,
        "created_at": utc_now(),
        "ckpt_path": args.ckpt_path,
        "config_yaml": args.config_yaml,
        "test_runs": args.test_runs,
        "resolved_test_runs": [str(path) for path in test_run_paths],
        "split": args.split,
        "out_matrix": args.out_matrix or "from_config_or_checkpoint",
        "graph2mat_version": graph2mat_version(),
        "pytorch_lightning_version": package_version("pytorch-lightning"),
        "python_version": platform.python_version(),
        "output_dir": args.output_dir,
        "save_flags": {"png": args.save_png, "pdf": args.save_pdf, "html": args.save_html},
        "image_export": {
            "width": args.image_width,
            "height": args.image_height,
            "scale": args.image_scale,
            "stretch_matrix": args.stretch_matrix,
        },
        "show": args.show,
        "store_in_logger": args.store_in_logger,
        "outputs": {},
        "status": status,
        "warnings": [],
        "fatal_errors": [],
        "input_hashes": {
            "checkpoint_sha256": optional_file_sha256(args.ckpt_path),
            "config_yaml_sha256": optional_file_sha256(args.config_yaml),
            "test_runs": compact_hash_summary(test_run_paths),
        },
    }


def run_cli_mode(args: argparse.Namespace, manifest: dict[str, Any]) -> dict[str, Any]:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_metrics_csv = output_dir / "sample_metrics.csv"
    stdout_log = output_dir / "graph2mat_test_stdout.log"
    stderr_log = output_dir / "graph2mat_test_stderr.log"
    command_plan = build_graph2mat_test_command(args, sample_metrics_csv=sample_metrics_csv)
    manifest["command"] = command_plan.command
    manifest["warnings"].extend(command_plan.warnings)
    manifest["outputs"].update(
        {
            "sample_metrics_csv": sample_metrics_csv if sample_metrics_csv.exists() else None,
            "stdout_log": stdout_log,
            "stderr_log": stderr_log,
            "tensorboard_dir": output_dir / "tensorboard",
        }
    )
    if args.dry_run:
        stdout_log.write_text("Dry run: command was not executed.\n", encoding="utf-8")
        stderr_log.write_text("", encoding="utf-8")
        manifest["status"] = "planned_dry_run"
        return manifest

    env = torch_compat_env()
    if args.store_in_logger:
        env["PL_LOGGER_DEFAULTS_ENABLE"] = "1"
        env["PL_LOGGER_SAVE_DIR"] = str((output_dir / "tensorboard").resolve())
        env["PL_LOGGER_NAME"] = "graph2mat_matrix_error"
        env["PL_LOGGER_VERSION"] = "version_0"
    completed = subprocess.run(
        command_plan.command,
        cwd=output_dir,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    stdout_log.write_text(completed.stdout, encoding="utf-8")
    stderr_log.write_text(completed.stderr, encoding="utf-8")
    manifest["returncode"] = completed.returncode
    manifest["outputs"]["sample_metrics_csv"] = sample_metrics_csv if sample_metrics_csv.exists() else None
    if completed.returncode != 0:
        manifest["status"] = "failed"
        manifest["fatal_errors"].append(f"Graph2Mat CLI exited with return code {completed.returncode}.")
        return manifest
    manifest["status"] = "ok"
    return manifest


def load_yaml_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError("PyYAML is required to read --config-yaml") from exc
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def nested_get(payload: dict[str, Any], *keys: str) -> Any:
    cursor: Any = payload
    for key in keys:
        if not isinstance(cursor, dict) or key not in cursor:
            return None
        cursor = cursor[key]
    return cursor


def first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def programmatic_datamodule_kwargs(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    data_cfg = first_present(
        config.get("data") if isinstance(config.get("data"), dict) else None,
        nested_get(config, "training", "data"),
        nested_get(config, "testing", "data"),
        {},
    )
    kwargs = {
        "out_matrix": first_present(args.out_matrix, data_cfg.get("out_matrix"), "hamiltonian"),
        "basis_files": first_present(args.basis_files, data_cfg.get("basis_files")),
        "test_runs": args.test_runs[0] if len(args.test_runs) == 1 else ",".join(args.test_runs),
        "symmetric_matrix": bool(first_present(args.symmetric_matrix, data_cfg.get("symmetric_matrix"), False)),
        "sub_point_matrix": bool(first_present(args.sub_point_matrix, data_cfg.get("sub_point_matrix"), True)),
        "batch_size": int(first_present(args.batch_size, data_cfg.get("batch_size"), 1)),
        "store_in_memory": bool(first_present(args.store_in_memory, data_cfg.get("store_in_memory"), False)),
    }
    for key in ("n_matrix_components", "matrix_component_policy", "loader_threads", "root_dir", "no_basis", "initial_node_feats"):
        if key in data_cfg and data_cfg[key] is not None:
            kwargs[key] = data_cfg[key]
    return kwargs


def run_programmatic_mode(args: argparse.Namespace, manifest: dict[str, Any]) -> dict[str, Any]:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    stdout_log = output_dir / "graph2mat_test_stdout.log"
    stderr_log = output_dir / "graph2mat_test_stderr.log"
    sample_metrics_csv = output_dir / "sample_metrics.csv"
    manifest["outputs"].update({"stdout_log": stdout_log, "stderr_log": stderr_log, "sample_metrics_csv": sample_metrics_csv})
    if args.dry_run:
        stdout_log.write_text("Dry run: programmatic Graph2Mat test was not executed.\n", encoding="utf-8")
        stderr_log.write_text("", encoding="utf-8")
        manifest["status"] = "planned_dry_run"
        manifest["datamodule_kwargs"] = programmatic_datamodule_kwargs(args, load_yaml_config(args.config_yaml))
        return manifest

    with stdout_log.open("w", encoding="utf-8") as stdout_fd, stderr_log.open("w", encoding="utf-8") as stderr_fd:
        with contextlib.redirect_stdout(stdout_fd), contextlib.redirect_stderr(stderr_fd):
            try:
                saved_outputs = _run_programmatic_graph2mat_test(args, sample_metrics_csv=sample_metrics_csv)
            except Exception as exc:
                manifest["status"] = "failed"
                manifest["fatal_errors"].append(str(exc))
                manifest["traceback"] = traceback.format_exc()
                print(traceback.format_exc(), file=sys.stderr)
                return manifest
    save_warnings = saved_outputs.pop("_warnings", None)
    manifest["outputs"].update(saved_outputs)
    if save_warnings:
        manifest["warnings"].extend(save_warnings)
    manifest["status"] = "ok"
    return manifest


def _run_programmatic_graph2mat_test(args: argparse.Namespace, *, sample_metrics_csv: Path) -> dict[str, Any]:
    sys.path.insert(0, str(TORCH_COMPAT_DIR))
    from torch_safe_globals import allow_graph2mat_checkpoint_globals

    allow_graph2mat_checkpoint_globals()

    import numpy as np
    import pytorch_lightning as pl
    import sisl
    from pytorch_lightning.loggers import TensorBoardLogger

    from graph2mat import AtomicTableWithEdges
    from graph2mat.core.data.sparse import nodes_and_edges_to_sparse_orbital
    from graph2mat.tools.lightning import MatrixDataModule, PlotMatrixError, SamplewiseMetricsLogger
    from graph2mat.tools.lightning.models.mace import LitMACEMatrixModel
    from graph2mat.tools.viz import plot_basis_matrix

    class SavingPlotMatrixError(PlotMatrixError):
        def __init__(self, *callback_args: Any, output_dir: Path, **callback_kwargs: Any) -> None:
            super().__init__(*callback_args, **callback_kwargs)
            self.output_dir = output_dir
            self.saved: dict[str, Path] = {}
            self.save_warnings: list[str] = []

        def _save_figure(self, fig: Any, label: str) -> None:
            slug = label.lower()
            fig.update_layout(width=args.image_width, height=args.image_height)
            if args.save_html:
                path = self.output_dir / f"matrix_error_{slug}.html"
                try:
                    fig.write_html(path)
                    self.saved[f"{slug}_html"] = path
                except Exception as exc:
                    message = f"Could not save {label} HTML: {exc}"
                    print(f"[WARN] {message}", file=sys.stderr)
                    self.save_warnings.append(message)
            if args.save_png:
                path = self.output_dir / f"matrix_error_{slug}.png"
                try:
                    fig.write_image(path, width=args.image_width, height=args.image_height, scale=args.image_scale)
                    self.saved[f"{slug}_png"] = path
                except Exception as exc:
                    message = f"Could not save {label} PNG: {exc}"
                    print(f"[WARN] {message}", file=sys.stderr)
                    self.save_warnings.append(message)
            if args.save_pdf:
                path = self.output_dir / f"matrix_error_{slug}.pdf"
                try:
                    fig.write_image(path, width=args.image_width, height=args.image_height, scale=args.image_scale)
                    self.saved[f"{slug}_pdf"] = path
                except Exception as exc:
                    message = f"Could not save {label} PDF: {exc}"
                    print(f"[WARN] {message}", file=sys.stderr)
                    self.save_warnings.append(message)

        def _on_epoch_end(self, trainer: Any, pl_module: Any) -> None:
            matrix_cls = {
                "density_matrix": sisl.DensityMatrix,
                "energy_density_matrix": sisl.EnergyDensityMatrix,
                "hamiltonian": sisl.Hamiltonian,
                "dynamical_matrix": sisl.DynamicalMatrix,
            }[trainer.datamodule.out_matrix]
            basis_table: AtomicTableWithEdges = trainer.datamodule.basis_table
            labels = ["MAE", "RMSE"]
            node_errors = [self.node_running_ae, self.node_running_se]
            edge_errors = [self.edge_running_ae, self.edge_running_se]
            assert self.point_types is not None
            for label, node_error, edge_error in zip(labels, node_errors, edge_errors):
                unique_atoms = basis_table.get_sisl_atoms()
                point_types = np.asarray(self.point_types, dtype=int)
                edge_index = np.asarray(self.edge_index, dtype=int)
                geometry = sisl.Geometry(
                    self.positions,
                    atoms=[unique_atoms[int(at_type)] for at_type in point_types],
                    lattice=self.cell,
                )
                geometry.set_nsc(self.nsc)
                assert node_error is not None
                assert edge_error is not None
                if label == "RMSE":
                    ne = np.sqrt(node_error / self.matrix_count)
                    ee = np.sqrt(edge_error / self.matrix_count)
                else:
                    ne = node_error / self.matrix_count
                    ee = edge_error / self.matrix_count
                matrix = nodes_and_edges_to_sparse_orbital(
                    node_vals=ne,
                    edge_vals=ee,
                    edge_index=edge_index,
                    geometry=geometry,
                    sp_class=matrix_cls,
                    symmetrize_edges=trainer.datamodule.symmetric_matrix,
                    matrix_component_policy=getattr(trainer.datamodule, "matrix_component_policy", None),
                ).tocsr()
                fig = plot_basis_matrix(
                    matrix,
                    configuration=geometry,
                    point_lines=True,
                    basis_lines=True,
                    text=".3f",
                    colorscale="temps",
                ).update_layout(
                    title=f"Graph2Mat {args.split} matrix error: {label}",
                )
                if args.stretch_matrix:
                    fig.update_yaxes(scaleanchor=False, constrain=None)
                    fig.update_xaxes(constrain=None)
                    fig.update_layout(margin={"l": 80, "r": 120, "t": 80, "b": 80})
                self._save_figure(fig, label)
                if self.show:
                    fig.show("browser")
                if self.store_in_logger and trainer.logger is not None:
                    import io
                    import PIL.Image

                    img_bytes = fig.to_image(format="png", width=800, height=600)
                    img_buf = io.BytesIO(img_bytes)
                    img = PIL.Image.open(img_buf, formats=["PNG"])
                    trainer.logger.experiment.add_image(
                        label,
                        np.array(img),
                        dataformats="HWC",
                        global_step=trainer.global_step,
                    )

    config = load_yaml_config(args.config_yaml)
    datamodule_kwargs = programmatic_datamodule_kwargs(args, config)
    model_overrides: dict[str, Any] = {}
    for key in ("basis_files", "root_dir", "n_matrix_components", "symmetric_matrix", "initial_node_feats"):
        value = datamodule_kwargs.get(key)
        if value is not None:
            model_overrides[key] = value
    model = LitMACEMatrixModel.load_from_checkpoint(
        str(args.ckpt_path),
        weights_only=False,
        **model_overrides,
    )
    datamodule = MatrixDataModule(**datamodule_kwargs)
    callbacks: list[Any] = []
    saving_callback: SavingPlotMatrixError | None = None
    if args.plot_matrix_error:
        saving_callback = SavingPlotMatrixError(
            split=args.split,
            show=args.show,
            store_in_logger=args.store_in_logger,
            output_dir=args.output_dir,
        )
        callbacks.append(saving_callback)
    if args.samplewise_metrics:
        callbacks.append(SamplewiseMetricsLogger(splits=[args.split], output_file=sample_metrics_csv))
    logger: Any = False
    if args.store_in_logger:
        logger = TensorBoardLogger(save_dir=str(args.output_dir / "tensorboard"), name="graph2mat_matrix_error")
    accelerator = first_present(args.accelerator, nested_get(config, "trainer", "accelerator"), nested_get(config, "training", "trainer", "accelerator"), "cpu")
    trainer = pl.Trainer(accelerator=accelerator, logger=logger, callbacks=callbacks)
    trainer.test(model, datamodule=datamodule, ckpt_path=None)
    outputs: dict[str, Any] = {}
    if saving_callback is not None:
        outputs.update({key: path for key, path in saving_callback.saved.items()})
        outputs["_warnings"] = saving_callback.save_warnings
    if sample_metrics_csv.exists():
        outputs["sample_metrics_csv"] = sample_metrics_csv
    if args.store_in_logger:
        outputs["tensorboard_dir"] = args.output_dir / "tensorboard"
    return outputs


def write_output_manifest(output_dir: Path, manifest: dict[str, Any]) -> Path:
    outputs = manifest.setdefault("outputs", {})
    for key, value in list(outputs.items()):
        if isinstance(value, Path) and not value.exists():
            outputs[key] = None
    path = output_dir / "matrix_error_manifest.json"
    write_json(path, manifest)
    return path


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    test_run_paths = validate_inputs(args)
    manifest = base_manifest(args, test_run_paths, status="running")
    try:
        if args.mode == "cli":
            manifest = run_cli_mode(args, manifest)
        elif args.mode == "programmatic":
            manifest = run_programmatic_mode(args, manifest)
        else:
            raise RuntimeError(f"Unsupported mode: {args.mode}")
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["fatal_errors"].append(str(exc))
    manifest_path = write_output_manifest(args.output_dir, manifest)
    manifest["manifest_path"] = manifest_path
    if manifest["status"] == "failed":
        raise RuntimeError("; ".join(manifest["fatal_errors"]) or "Graph2Mat matrix error export failed")
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["cli", "programmatic"], default="cli")
    parser.add_argument("--graph2mat-command", default="graph2mat")
    parser.add_argument("--ckpt-path", type=Path, required=True)
    parser.add_argument("--config-yaml", type=Path, default=None)
    parser.add_argument("--test-runs", action="append", required=True, help="RUN.fdf path or glob. Repeatable.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--out-matrix", choices=["density_matrix", "hamiltonian", "energy_density_matrix", "dynamical_matrix"], default=None)
    parser.add_argument("--basis-files", default=None)
    parser.add_argument("--symmetric-matrix", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--sub-point-matrix", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--store-in-memory", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--accelerator", default=None)
    parser.add_argument("--plot-matrix-error", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--samplewise-metrics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--store-in-logger", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-png", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-pdf", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save-html", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--image-width", type=int, default=1600)
    parser.add_argument("--image-height", type=int, default=1000)
    parser.add_argument("--image-scale", type=float, default=2.0)
    parser.add_argument("--stretch-matrix", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = run(args)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"status": manifest["status"], "output_dir": str(args.output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
