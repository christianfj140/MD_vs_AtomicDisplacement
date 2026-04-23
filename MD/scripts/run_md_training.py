#!/usr/bin/env python3
"""Automatiza el entrenamiento MACE según MD/command_history.txt.

Flujo replicado (hardcodeado):
1) cd ..
2) mkdir training
3) cd training
4) crear config.yaml
5) graph2mat models mace main fit -c config.yaml
6) (opcional) tensorboard --logdir lightning_logs
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINING_DIR = REPO_ROOT / "MD" / "training"
CONFIG_PATH = TRAINING_DIR / "config.yaml"

# Config hardcodeado para esta fase inicial (basado en el config actual del repo).
CONFIG_YAML_CONTENT = """# pytorch_lightning==2.6.1
seed_everything: 0
trainer:
  accelerator: auto
  strategy: auto
  devices: auto
  num_nodes: 1
  precision: null
  logger: false
  callbacks:
  - class_path: pytorch_lightning.callbacks.ModelCheckpoint
    init_args:
      dirpath: null
      filename: last-{step:02d}
      monitor: step
      verbose: false
      save_last: true
      save_top_k: 1
      save_on_exception: false
      save_weights_only: false
      mode: max
      auto_insert_metric_name: false
      every_n_train_steps: null
      train_time_interval: null
      every_n_epochs: null
      save_on_train_epoch_end: null
      enable_version_counter: true
  - class_path: pytorch_lightning.callbacks.ModelCheckpoint
    init_args:
      dirpath: null
      filename: best-{step:02d}
      monitor: val_loss
      verbose: false
      save_last: null
      save_top_k: 1
      save_on_exception: false
      save_weights_only: false
      mode: min
      auto_insert_metric_name: false
      every_n_train_steps: null
      train_time_interval: null
      every_n_epochs: null
      save_on_train_epoch_end: null
      enable_version_counter: true
  - class_path: graph2mat.tools.lightning.SamplewiseMetricsLogger
    init_args:
      metrics: null
      splits:
      - train
      - val
      - test
      output_file: sample_metrics.csv
  fast_dev_run: false
  max_epochs: null
  min_epochs: null
  max_steps: -1
  min_steps: null
  max_time: null
  limit_train_batches: null
  limit_val_batches: null
  limit_test_batches: 1.0
  limit_predict_batches: null
  overfit_batches: 0.0
  val_check_interval: null
  check_val_every_n_epoch: 1
  num_sanity_val_steps: null
  log_every_n_steps: null
  enable_checkpointing: null
  enable_progress_bar: null
  enable_model_summary: null
  accumulate_grad_batches: 1
  gradient_clip_val: null
  gradient_clip_algorithm: null
  deterministic: null
  benchmark: null
  inference_mode: true
  use_distributed_sampler: true
  profiler: null
  detect_anomaly: false
  barebones: false
  plugins: null
  sync_batchnorm: false
  reload_dataloaders_every_n_epochs: 0
  default_root_dir: null
  enable_autolog_hparams: true
  model_registry: null
model:
  num_bessel: 10
  num_polynomial_cutoff: 3
  max_ell: 2
  interaction_cls: mace.modules.RealAgnosticResidualInteractionBlock
  interaction_cls_first: mace.modules.RealAgnosticResidualInteractionBlock
  num_interactions: 1
  hidden_irreps: 10x0e + 10x1o + 10x2e
  edge_hidden_irreps: 4x0e+4x1o+4x2e
  avg_num_neighbors: 1.0
  correlation: 1
  basis_grouping: point_type
  preprocessing_nodes: null
  preprocessing_edges: graph2mat.bindings.e3nn.E3nnEdgeMessageBlock
  preprocessing_edges_reuse_nodes: false
  node_block_readout: graph2mat.bindings.e3nn.E3nnSimpleNodeBlock
  edge_block_readout: graph2mat.bindings.e3nn.E3nnSimpleEdgeBlock
  readout_per_interaction: false
  optim_wdecay: 5.0e-07
  optim_amsgrad: true
  optim_lr: 0.005
  loss: graph2mat.core.data.metrics.block_type_mae
  version: new
data:
  out_matrix: hamiltonian
  basis_files: ../dataset/MD_steps/basis/*.ion.xml
  no_basis: null
  basis_table: null
  root_dir: .
  train_runs: ../dataset/MD_steps/*/RUN.fdf
  val_runs: null
  test_runs: ../dataset/MD_steps/25/RUN.fdf
  predict_structs: null
  runs_json: null
  symmetric_matrix: true
  sub_point_matrix: true
  n_matrix_components: 2
  batch_size: 10
  loader_threads: 1
  copy_root_to_tmp: false
  store_in_memory: true
  rotating_pool_size: null
  initial_node_feats: OneHotZ
multiprocessing_sharing_strategy: ''
optimizer: null
lr_scheduler: null
ckpt_path: lightning_logs/my_first_model/version_0/checkpoints/best-960.ckpt
verbose: true
weights_only: null
"""

FIT_COMMAND = ["graph2mat", "models", "mace", "main", "fit", "-c", "config.yaml"]


def require_command(command_name: str) -> None:
    if shutil.which(command_name) is None:
        raise RuntimeError(
            f"No se encontró '{command_name}' en PATH. "
            "Activa tu entorno antes de ejecutar este script."
        )


def run_command(cmd: list[str], cwd: Path) -> None:
    print(f"\n[RUN] {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"El comando falló con código {result.returncode}: {' '.join(cmd)}"
        )


def ensure_training_dir() -> None:
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)


def write_config_yaml() -> None:
    # Para reproducibilidad en esta fase, sobreescribimos config.yaml.
    CONFIG_PATH.write_text(CONFIG_YAML_CONTENT, encoding="utf-8")
    print(f"[OK] Config escrito en {CONFIG_PATH}")


def main() -> int:
    print("=== Pipeline MD (fase entrenamiento): fit del modelo ===")
    print(f"Repositorio: {REPO_ROOT}")
    print(f"Training dir: {TRAINING_DIR}")

    require_command("graph2mat")

    ensure_training_dir()
    write_config_yaml()
    run_command(FIT_COMMAND, cwd=TRAINING_DIR)

    print("[INFO] Si quieres ver métricas: tensorboard --logdir lightning_logs")
    print("\n=== Entrenamiento completado correctamente ===")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
