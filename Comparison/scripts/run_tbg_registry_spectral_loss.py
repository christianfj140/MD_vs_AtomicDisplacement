#!/usr/bin/env python3
"""Fine-tune Graph2Mat with a K-frontier loss and apply the frozen 10 meV gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from run_tbg_registry_50_50 import (
    DATASET,
    MIN_FREE_DISK_PERCENT,
    balanced_weights,
    disk_guard,
    prepare_holdout,
    run_gate,
    split_paths,
    update_status,
    write_json,
)
from tbg_spectral_training import SpectralLitMACEMatrixModel, build_reference_targets

REPO = Path(__file__).resolve().parents[2]
SOURCE_CHECKPOINT = (
    REPO
    / "Comparison/results/tbg_registry_50_50/training/checkpoints/"
    "best-epoch=242-step=03402.ckpt"
)
OUTPUT = REPO / "Comparison/results/tbg_registry_spectral_loss"


def train(args: argparse.Namespace, targets) -> list[Path]:
    import pytorch_lightning as pl
    import torch
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    from pytorch_lightning.loggers import TensorBoardLogger
    from torch.utils.data import WeightedRandomSampler
    from torch_geometric.loader import DataLoader

    from graph2mat import AtomicTableWithEdges
    from graph2mat.tools.lightning import MatrixDataModule

    train_paths, registry_mask, val_paths, test_paths = split_paths()
    weights = balanced_weights(registry_mask)

    class BalancedDataModule(MatrixDataModule):
        def train_dataloader(self):
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
    model = SpectralLitMACEMatrixModel.load_from_checkpoint(
        str(SOURCE_CHECKPOINT),
        weights_only=False,
        basis_table=basis_table,
        basis_files=basis_glob,
        root_dir=str(DATASET),
        optim_lr=args.learning_rate,
    )
    model.hparams.optim_lr = args.learning_rate
    model.configure_spectral_loss(targets, weight=args.spectral_weight, beta_eV=args.spectral_beta)

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
    metric = "val_spectral_frontier_aligned_rmse_eV"
    best = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="spectral-best-{epoch:03d}-{step:05d}",
        monitor=metric,
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    periodic = ModelCheckpoint(
        dirpath=checkpoint_dir / "periodic",
        filename="epoch-{epoch:03d}-{step:05d}",
        every_n_epochs=25,
        save_top_k=-1,
    )
    callbacks = [
        best,
        periodic,
        EarlyStopping(monitor=metric, mode="min", patience=args.patience, min_delta=1e-4),
    ]
    logger = TensorBoardLogger(save_dir=str(OUTPUT / "training/lightning_logs"), name="spectral_loss")

    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    pl.seed_everything(args.seed, workers=True)
    update_status(
        "spectral_training",
        output=OUTPUT,
        source_checkpoint=str(SOURCE_CHECKPOINT),
        spectral_weight=args.spectral_weight,
        spectral_beta_eV=args.spectral_beta,
        checkpoint_monitor=metric,
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
    trainer.fit(model, datamodule=datamodule)
    # Checkpoint identity is fixed by validation only; the holdout must never select it.
    checkpoints = [Path(best.best_model_path)]
    if not all(path.is_file() for path in checkpoints):
        raise RuntimeError(f"Missing spectral checkpoints: {checkpoints}")
    return checkpoints


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--gate-only-checkpoint", type=Path)
    parser.add_argument("--max-epochs", type=int, default=250)
    parser.add_argument("--patience", type=int, default=150)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--spectral-weight", type=float, default=0.25)
    parser.add_argument("--spectral-beta", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--loader-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not SOURCE_CHECKPOINT.is_file():
        raise FileNotFoundError(SOURCE_CHECKPOINT)
    disk_guard()
    if args.gate_only_checkpoint:
        checkpoint = args.gate_only_checkpoint.resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        prepare_holdout()
        result = run_gate([checkpoint], output=OUTPUT)
        update_status("complete", result["status"], output=OUTPUT, precision_gate=result)
        print(json.dumps(result, indent=2))
        return 0 if result["status"] == "passed" else 2

    train_paths, registry_mask, val_paths, _ = split_paths()
    targets = build_reference_targets(
        train_paths + val_paths, OUTPUT / "spectral_targets_k.npz"
    )
    plan = {
        "source_checkpoint": str(SOURCE_CHECKPOINT),
        "physical_train_samples": len(train_paths),
        "registry_samples": sum(registry_mask),
        "validation_samples": len(val_paths),
        "sampling_mass": {"md": 0.5, "registry": 0.5},
        "spectral_kpoint": [1 / 3, 1 / 3, 0.0],
        "frontier_states": 4,
        "spectral_weight": args.spectral_weight,
        "spectral_beta_eV": args.spectral_beta,
        "checkpoint_monitor": "val_spectral_frontier_aligned_rmse_eV",
        "periodic_checkpoint_epochs": 25,
        "minimum_free_disk_percent": MIN_FREE_DISK_PERCENT,
    }
    write_json(OUTPUT / "training_plan.json", plan)
    if args.prepare_only:
        print(json.dumps(plan, indent=2))
        return 0

    prepare_holdout()
    checkpoints = train(args, targets)
    result = run_gate(checkpoints, output=OUTPUT)
    update_status("complete", result["status"], output=OUTPUT, precision_gate=result)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        update_status("failed", "failed", output=OUTPUT, error=f"{type(exc).__name__}: {exc}")
        raise
