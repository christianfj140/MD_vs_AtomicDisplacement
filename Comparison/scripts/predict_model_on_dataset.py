#!/usr/bin/env python3
"""Predict Hamiltonians for a manifest-defined test dataset.

Examples
--------
Run a trained MD model on a frozen AtomDisplacement test set::

    python Comparison/scripts/predict_model_on_dataset.py \
      --checkpoint Comparison/workspaces/exp/md/dataset_10/training/lightning_logs/.../best.ckpt \
      --train-method md \
      --test-set test_atomdisp \
      --test-manifest Comparison/results/exp/common_tests/test_atomdisp/test_manifest.csv \
      --basis-files "../dataset/basis/*.ion.xml" \
      --output-dir Comparison/results/exp/cross_predictions/md_on_test_atomdisp
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    if not fieldnames:
        fieldnames = ["sample_id", "status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def copy_sample_inputs(rows: list[dict[str, str]], workspace: Path) -> list[dict[str, Any]]:
    copied = []
    structs_dir = workspace / "predict_structures"
    if structs_dir.exists():
        shutil.rmtree(structs_dir)
    structs_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        sample_id = row["sample_id"]
        sample_dir = structs_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        src_structure = Path(row["structure_path"])
        if not src_structure.exists():
            raise RuntimeError(f"Missing structure for {sample_id}: {src_structure}")
        dst_structure = sample_dir / "RUN.fdf"
        shutil.copy2(src_structure, dst_structure)
        # Copy only structural/basis side inputs. Reference Hamiltonians are
        # deliberately not copied into the prediction input tree; evaluation
        # reads them through the manifest/reference paths instead.
        source_parent = src_structure.parent
        for pattern in ("*.psf", "*.ion", "*.ion.xml", "*.XV"):
            for src in source_parent.glob(pattern):
                if src.name == "ML_prediction.HSX":
                    continue
                dst = sample_dir / src.name
                if not dst.exists():
                    shutil.copy2(src, dst)
        copied.append(
            {
                **row,
                "prediction_structure_path": str(dst_structure),
                "reference_hamiltonian_copied_to_input": False,
            }
        )
    return copied


def run_prediction(args: argparse.Namespace, predict_glob: str) -> None:
    import inspect
    import pytorch_lightning as pl
    from graph2mat.core.data.processing import MatrixDataProcessor
    from graph2mat.tools.lightning import MatrixDataModule, MatrixWriter
    from graph2mat.tools.lightning.models.mace import LitMACEMatrixModel

    if args.patch_graph2mat_basis_loading:
        original = MatrixDataProcessor.get_config_kwargs

        def patched(self: MatrixDataProcessor, obj: object) -> dict[str, object]:
            kwargs = original(self, obj)
            kwargs.pop("basis", None)
            return kwargs

        MatrixDataProcessor.get_config_kwargs = patched

    model = LitMACEMatrixModel.load_from_checkpoint(str(args.checkpoint))
    datamodule_kwargs: dict[str, Any] = {
        "out_matrix": args.out_matrix,
        "symmetric_matrix": bool(args.symmetric_matrix),
        "sub_point_matrix": bool(args.sub_point_matrix),
        "basis_files": args.basis_files,
        "predict_structs": predict_glob,
        "store_in_memory": bool(args.store_in_memory),
    }
    if "n_matrix_components" in inspect.signature(MatrixDataModule).parameters and args.n_matrix_components:
        datamodule_kwargs["n_matrix_components"] = int(args.n_matrix_components)
    datamodule = MatrixDataModule(**datamodule_kwargs)
    callbacks = [MatrixWriter(output_file="ML_prediction.HSX", splits=["predict"])]
    trainer = pl.Trainer(accelerator=args.accelerator, logger=False, callbacks=callbacks)
    trainer.predict(model, datamodule=datamodule, ckpt_path=str(args.checkpoint))


def checkpoint_training_dir(checkpoint: Path) -> Path:
    for parent in checkpoint.parents:
        if parent.name == "lightning_logs":
            return parent.parent
    return checkpoint.parent


def relative_pattern(pattern: str, base: Path) -> str:
    if os.path.isabs(pattern):
        return os.path.relpath(pattern, base).replace("\\", "/")
    return pattern


def collect_predictions(rows: list[dict[str, Any]], workspace: Path, output_dir: Path) -> list[dict[str, Any]]:
    prediction_rows = []
    prediction_root = output_dir / "predicted_hamiltonians"
    if prediction_root.exists():
        shutil.rmtree(prediction_root)
    prediction_root.mkdir(parents=True, exist_ok=True)
    for row in rows:
        sample_id = row["sample_id"]
        source_sample_dir = workspace / "predict_structures" / sample_id
        source_prediction = source_sample_dir / "ML_prediction.HSX"
        destination_dir = prediction_root / sample_id
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination_prediction = destination_dir / "ML_prediction.HSX"
        status = "predicted" if source_prediction.exists() else "missing_prediction"
        if source_prediction.exists():
            shutil.copy2(source_prediction, destination_prediction)
        prediction_rows.append(
            {
                **row,
                "prediction_path": str(destination_prediction) if destination_prediction.exists() else "",
                "status": status,
            }
        )
    return prediction_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-method", required=True, choices=["md", "atom_displacement"])
    parser.add_argument("--test-set", required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--basis-files", required=True)
    parser.add_argument("--out-matrix", default="hamiltonian")
    parser.add_argument("--symmetric-matrix", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sub-point-matrix", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--store-in-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--accelerator", default="cpu")
    parser.add_argument("--n-matrix-components", type=int, default=None)
    parser.add_argument("--patch-graph2mat-basis-loading", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.checkpoint.exists():
        raise RuntimeError(f"Checkpoint does not exist: {args.checkpoint}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    workspace = args.output_dir / "workspace"
    rows = read_rows(args.test_manifest)
    start = time.time()
    copied_rows = copy_sample_inputs(rows, workspace)
    run_cwd = checkpoint_training_dir(args.checkpoint)
    args.basis_files = relative_pattern(str(args.basis_files), run_cwd)
    predict_glob = os.path.relpath(
        workspace / "predict_structures" / "*" / "RUN.fdf",
        run_cwd,
    ).replace("\\", "/")
    prediction_status = "dry_run"
    error = None
    if not args.dry_run:
        cwd = Path.cwd()
        try:
            os.chdir(run_cwd)
            run_prediction(args, predict_glob)
            prediction_status = "completed"
        except Exception as exc:
            prediction_status = "failed"
            error = str(exc)
        finally:
            os.chdir(cwd)
    prediction_rows = collect_predictions(copied_rows, workspace, args.output_dir)
    for row in prediction_rows:
        row["train_method"] = args.train_method
        row["test_set"] = args.test_set
        row["model_checkpoint"] = str(args.checkpoint)
    write_rows(args.output_dir / "prediction_manifest.csv", prediction_rows)
    summary = {
        "ok": prediction_status in {"completed", "dry_run"} and all(
            row["status"] == "predicted" for row in prediction_rows
        ),
        "status": prediction_status,
        "error": error,
        "train_method": args.train_method,
        "test_set": args.test_set,
        "checkpoint": str(args.checkpoint),
        "samples": len(rows),
        "predicted_samples": sum(1 for row in prediction_rows if row["status"] == "predicted"),
        "prediction_time_seconds": time.time() - start,
        "outputs": {
            "prediction_manifest": str(args.output_dir / "prediction_manifest.csv"),
            "predicted_hamiltonians": str(args.output_dir / "predicted_hamiltonians"),
        },
        "input_reference_files_copied": False,
    }
    (args.output_dir / "prediction_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if summary["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
