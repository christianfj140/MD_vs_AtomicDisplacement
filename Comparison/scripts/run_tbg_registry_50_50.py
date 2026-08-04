#!/usr/bin/env python3
"""Fine-tune Graph2Mat with balanced MD/registry sampling, then run the frozen gate."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


TORCH_COMPAT_DIR = Path(__file__).resolve().parents[2] / "scripts/torch_serialization_compat"
sys.path.insert(0, str(TORCH_COMPAT_DIR))
from torch_safe_globals import allow_graph2mat_checkpoint_globals  # noqa: E402

allow_graph2mat_checkpoint_globals()


REPO = Path(__file__).resolve().parents[2]
DATASET = REPO / "Comparison/datasets/tbg_md_plus_registry/n558"
SOURCE_CHECKPOINT = REPO / (
    "Comparison/results/tbg_registry/training/n558/tbg_registry_n558/sweep/graph2mat/n558/"
    "g2m_tbg_n558_fp32_seed0/graph2mat/training/lightning_logs/my_first_model/"
    "version_0/checkpoints/best-7476.ckpt"
)
OUTPUT = REPO / "Comparison/results/tbg_registry_50_50"
HOLDOUT = REPO / "Comparison/datasets/tbg_registry_holdout_7x7"
MIN_FREE_DISK_PERCENT = 12.0
GATE_EV = 0.010


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def update_status(
    stage: str, state: str = "running", *, output: Path = OUTPUT, **extra: object
) -> None:
    write_json(
        output / "status.json",
        {
            "stage": stage,
            "state": state,
            "running": state == "running",
            "pid": os.getpid() if state == "running" else None,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "minimum_free_disk_percent": MIN_FREE_DISK_PERCENT,
            **extra,
        },
    )


def disk_guard() -> float:
    usage = shutil.disk_usage(REPO)
    free = 100.0 * usage.free / usage.total
    if free < MIN_FREE_DISK_PERCENT:
        raise RuntimeError(f"Disk guard: {free:.2f}% free < {MIN_FREE_DISK_PERCENT:.2f}%")
    return free


def split_paths() -> tuple[list[str], list[bool], list[str], list[str]]:
    manifest = json.loads((DATASET / "frozen_split_manifest.json").read_text(encoding="utf-8"))
    by_split: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
    for row in manifest["rows"]:
        by_split[row["split"]].append(row)

    train = sorted(by_split["train"], key=lambda row: row["sample_id"])
    train_paths = [str(Path(row["structure_path"]).resolve()) for row in train]
    registry_mask = [str(row["sample_id"]).startswith("tbg_registry_grid_12x12__") for row in train]
    val_paths = [str(Path(row["structure_path"]).resolve()) for row in by_split["validation"]]
    test_paths = [str(Path(row["structure_path"]).resolve()) for row in by_split["test"]]
    return train_paths, registry_mask, val_paths, test_paths


def balanced_weights(registry_mask: list[bool]) -> list[float]:
    n_registry = sum(registry_mask)
    n_md = len(registry_mask) - n_registry
    if not n_registry or not n_md:
        raise ValueError(f"Cannot balance train split: md={n_md}, registry={n_registry}")
    weights = [0.5 / (n_registry if is_registry else n_md) for is_registry in registry_mask]
    assert math.isclose(sum(w for w, flag in zip(weights, registry_mask) if flag), 0.5)
    assert math.isclose(sum(w for w, flag in zip(weights, registry_mask) if not flag), 0.5)
    return weights


def prepare_holdout() -> None:
    if (HOLDOUT / "frozen_split_manifest.json").is_file():
        return
    subprocess.run(
        [
            str(REPO / ".venv/bin/python"),
            str(REPO / "Comparison/scripts/build_registry_grid_probe.py"),
            "--output-root",
            str(HOLDOUT),
            "--divisions",
            "7",
            "--jobs",
            "8",
            "--split",
            "test",
        ],
        cwd=REPO,
        check=True,
    )


def run_gate(checkpoints: list[Path], output: Path = OUTPUT) -> dict:
    python = str(REPO / ".venv/bin/python")
    evaluator = str(REPO / "Comparison/scripts/evaluate_checkpoint_spectral_metrics.py")
    basis = str(DATASET / "material_basis/*.ion.xml")
    validation_root = output / "eval_validation"
    validation_report = validation_root / "checkpoint_spectral_metrics_validation.json"
    common = [item for checkpoint in checkpoints for item in ("--checkpoint", str(checkpoint))]

    update_status(
        "validation_gate", output=output, checkpoints=[str(path) for path in checkpoints]
    )
    subprocess.run(
        [python, evaluator, "--dataset", str(DATASET), *common, "--split", "validation",
         "--basis-files", basis, "--output-root", str(validation_root)],
        cwd=REPO,
        check=True,
    )
    disk_guard()

    test_root = output / "eval_registry_holdout"
    update_status("registry_holdout_gate", output=output)
    subprocess.run(
        [python, evaluator, "--dataset", str(HOLDOUT), *common, "--split", "test",
         "--calibration-report", str(validation_report), "--basis-files", basis,
         "--output-root", str(test_root)],
        cwd=REPO,
        check=True,
    )

    report = json.loads((test_root / "checkpoint_spectral_metrics_test.json").read_text(encoding="utf-8"))
    scores: dict[str, float] = {}
    for name, payload in report["checkpoints"].items():
        offset = report["frozen_offsets_eV"][name]
        rows = [
            row for row in payload["per_sample"]
            if row["kpoint"] in ("K", "Kprime") and row["frontier_n_states"]
        ]
        scores[name] = float(np.sqrt(np.mean([
            row["frontier_rmse_eV"] ** 2
            + 2 * offset * row["frontier_global_shift_eV"]
            + offset**2
            for row in rows
        ])))
    winner = min(scores, key=scores.get)
    result = {
        "status": "passed" if scores[winner] < GATE_EV else "failed",
        "threshold_eV": GATE_EV,
        "checkpoint": winner,
        "score_eV": scores[winner],
        "all_scores_eV": scores,
        "validation_report": str(validation_report),
        "test_report": str(test_root / "checkpoint_spectral_metrics_test.json"),
    }
    write_json(output / "precision_gate.json", result)
    return result


def train(args: argparse.Namespace) -> list[Path]:
    import pytorch_lightning as pl
    import torch
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    from pytorch_lightning.loggers import TensorBoardLogger
    from torch.utils.data import WeightedRandomSampler
    from torch_geometric.loader import DataLoader

    from graph2mat import AtomicTableWithEdges
    from graph2mat.tools.lightning import MatrixDataModule
    from graph2mat.tools.lightning.models.mace import LitMACEMatrixModel
    train_paths, registry_mask, val_paths, test_paths = split_paths()
    weights = balanced_weights(registry_mask)
    n_registry = sum(registry_mask)
    n_md = len(registry_mask) - n_registry

    class BalancedEarlyStopping(EarlyStopping):
        @property
        def state_key(self):
            return self._generate_state_key(
                monitor=self.monitor, mode=self.mode, campaign="registry_50_50"
            )

    class BalancedDataModule(MatrixDataModule):
        def train_dataloader(self):
            assert self.train_dataset is not None
            sampler = WeightedRandomSampler(
                weights,
                num_samples=len(weights),
                replacement=True,
                generator=torch.Generator().manual_seed(args.seed),
            )
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                sampler=sampler,
                drop_last=False,
                num_workers=self.hparams.loader_threads,
                persistent_workers=bool(self.hparams.loader_threads),
            )

    basis_glob = str(DATASET / "material_basis/*.ion.xml")
    basis_table = AtomicTableWithEdges.from_basis_glob((DATASET / "material_basis").glob("*.ion.xml"))
    allow_graph2mat_checkpoint_globals()
    model = LitMACEMatrixModel.load_from_checkpoint(
        str(SOURCE_CHECKPOINT),
        weights_only=False,
        basis_table=basis_table,
        basis_files=basis_glob,
        root_dir=str(DATASET),
        optim_lr=args.learning_rate,
    )
    model.hparams.optim_lr = args.learning_rate

    datamodule = BalancedDataModule(
        out_matrix="hamiltonian",
        symmetric_matrix=True,
        sub_point_matrix=False,
        matrix_component_policy="h_only",
        n_matrix_components=1,
        basis_table=basis_table,
        basis_files=basis_glob,
        root_dir=str(DATASET),
        train_runs=train_paths,
        val_runs=val_paths,
        test_runs=test_paths,
        batch_size=args.batch_size,
        loader_threads=args.loader_threads,
        store_in_memory=True,
    )

    checkpoint_dir = OUTPUT / "training/checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="best-{epoch:03d}-{step:05d}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    callbacks = [
        checkpoint,
        BalancedEarlyStopping(
            monitor="val_loss", mode="min", patience=args.patience, min_delta=1e-5
        ),
    ]
    logger = TensorBoardLogger(save_dir=str(OUTPUT / "training/lightning_logs"), name="finetune_50_50")

    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    pl.seed_everything(args.seed, workers=True)
    update_status(
        "training_50_50",
        source_checkpoint=str(SOURCE_CHECKPOINT),
        physical_train_samples=len(train_paths),
        md_samples=n_md,
        registry_samples=n_registry,
        expected_sampling_fraction={"md": 0.5, "registry": 0.5},
        learning_rate=args.learning_rate,
        max_epochs=args.max_epochs,
        free_disk_percent=disk_guard(),
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision="32-true",
        max_epochs=args.max_epochs,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=1,
        check_val_every_n_epoch=1,
        enable_progress_bar=True,
    )
    resume_checkpoint = checkpoint_dir / "last.ckpt"
    trainer.fit(
        model,
        datamodule=datamodule,
        ckpt_path=str(resume_checkpoint) if args.resume and resume_checkpoint.is_file() else None,
    )
    paths = [Path(checkpoint.best_model_path), checkpoint_dir / "last.ckpt"]
    if not paths[0].is_file() or not paths[1].is_file():
        raise RuntimeError(f"Fine-tuning produced incomplete checkpoints: {paths}")
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--max-epochs", type=int, default=250)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--loader-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if not SOURCE_CHECKPOINT.is_file():
        raise FileNotFoundError(SOURCE_CHECKPOINT)
    train_paths, registry_mask, val_paths, test_paths = split_paths()
    weights = balanced_weights(registry_mask)
    summary = {
        "physical_train_samples": len(train_paths),
        "md_samples": len(train_paths) - sum(registry_mask),
        "registry_samples": sum(registry_mask),
        "validation_samples": len(val_paths),
        "test_samples": len(test_paths),
        "sampling_mass_md": sum(w for w, flag in zip(weights, registry_mask) if not flag),
        "sampling_mass_registry": sum(w for w, flag in zip(weights, registry_mask) if flag),
        "source_checkpoint": str(SOURCE_CHECKPOINT),
    }
    write_json(OUTPUT / "sampling_plan.json", summary)
    if args.prepare_only:
        print(json.dumps(summary, indent=2))
        return 0

    prepare_holdout()
    checkpoints = train(args)
    result = run_gate(checkpoints)
    update_status("complete", result["status"], precision_gate=result)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        update_status("failed", "failed", error=str(exc))
        raise
