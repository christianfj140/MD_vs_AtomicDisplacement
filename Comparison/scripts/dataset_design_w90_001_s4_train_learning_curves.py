#!/usr/bin/env python3
"""DATASET-DESIGN-W90-001-S4 -- Graph2Mat (+ DeepH-gated) learning curves.

Trains Graph2Mat on the W90 graphene SIESTA references produced by S3
(``run_w90_displacement_siesta_pilot.py``) at nested training sizes
N_train in {16, 32, 64}, one model per (family, dimensionality, k, N_train,
seed), all with the *same* architecture/hyperparameters (S4 fairness
requirement) -- and records H-MAE/H-RMSE/relative-Frobenius/low-energy
spectral error/Hermiticity on a held-out validation split, plus the
effective training backend.

Cites and reuses, rather than reimplements (S1 section 4):
  - ``docs/dataset_design_w90_001_s1_convention_manifest.md`` (S1) and
    ``Comparison/scripts/run_w90_displacement_siesta_pilot.py`` (S3) for the
    sample inventory and units/PBC/ordering conventions.
  - The ``graph2mat`` CLI itself (``graph2mat models mace main fit`` /
    ``predict_model_on_dataset.py``) as the trainer/predictor -- the same
    tool ``Comparison/scripts/write_graph2mat_configs.py`` and
    ``Comparison/scripts/run_tbg_pure_graph2mat_campaign.py`` drive, and the
    same ``data`` block schema (``out_matrix``/``symmetric_matrix``/
    ``matrix_component_policy``/``train_runs``/``val_runs`` glob-of-RUN.fdf)
    already used by ``MD/pipeline_config.yaml``.
  - ``Comparison/scripts/deeph_raw_global_equivalence_preflight.py`` for the
    DeepH basis/units/ordering gate named in the task.
  - ``sisl`` (already a dependency) for reading Hk/eigenvalues from the
    reference and predicted matrices, exactly as S3's
    ``gamma_hamiltonian``/``verify_hermiticity`` already do.

Deliberately NOT used: the full ``Graph2MatDeepHBenchmarkRunner`` /
``g2m_deeph_end_to_end_pipeline.py`` orchestrator. That system's
``reuse_validated``/``full_strict_pipeline`` dataset modes are built around
its own MD/random-cartesian generators' dataset-manifest schema
(``benchmark_dataset_manifest.json`` + ``frozen_split_manifest.json`` +
temporal-block splitting) for campaigns of hundreds-to-thousands of MD
snapshots. Wiring S2/S3's six small geometric sampler families into that
schema is a separate, much larger integration project than this pilot step;
using the bare ``graph2mat`` CLI directly on S3's existing RUN.fdf +
TSHS/HSX pairs is the smaller, equally-real, reuse-respecting path for a
pilot with at most 128 samples per group.

DeepH: the raw/global equivalence preflight requires a DeepH-processed
representation of each sample (DeepH-pack's own raw-format conversion) --
that conversion has not been run for the S2/S3 W90 pilot samples in this
step, and standing up DeepH-pack preprocessing for a brand-new sampler
family is out of scope for this training step. Every group's DeepH gate is
therefore recorded as ``not_evaluated`` with that reason (task-permitted:
"if gates fail, record the reason and exclude DeepH from quantitative
comparison").

N_train availability: S3 generated far fewer samples for the deterministic
geometric families (``axial_radial``: 4-12 per group; ``local_pair_modes``:
6-18 per group) than for the stochastic families (``sobol_sparse``,
``random_cartesian``, ``angular_shell``: 128 per group, pooled over the two
pilot amplitudes). A (family, dim, k, N_train) point whose pooled sample
pool cannot cover N_train + its validation split is recorded with
``status=insufficient_samples`` and the available count, never fabricated.

MD family: excluded. S2's ``md_family_contract()`` and S3's docstring both
reserve MD frame generation for a later step; no S3 samples exist for it.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for _path in (SCRIPT_DIR, REPO_ROOT / "shared"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from reference_selection import choose_reference_matrix  # noqa: E402
import run_w90_displacement_siesta_pilot as pilot  # noqa: E402

DEFAULT_S3_ROOT = REPO_ROOT / "Comparison/results/dataset_design_w90_001_s3"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "Comparison/results/dataset_design_w90_001_s4"
DEFAULT_MD_CONFIG = REPO_ROOT / "MD/pipeline_config.yaml"
DEFAULT_BASIS_GLOB = str(REPO_ROOT / "materials/graphene/basis/*.ion.xml")
PYTHON = REPO_ROOT / ".venv/bin/python"
GRAPH2MAT_BIN = REPO_ROOT / ".venv/bin/graph2mat"

NESTED_N_TRAIN: tuple[int, ...] = (16, 32, 64)
SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
VAL_FRACTION = 0.25  # val_count = round(VAL_FRACTION * n_train), floor 4
MIN_VAL_COUNT = 4
# Validation is drawn once per (pool, seed) at the size needed for the
# largest nested N_train point, then held fixed across every N_train in the
# curve -- otherwise each N_train would draw its own val split (and, via the
# old n_train-keyed RNG seed, its own *train* split too), breaking the
# required train16 subset train32 subset train64 nesting and making
# "SIESTA cost" undercount the (fixed) validation structures. See S1/S4
# audit: "select_split independently draws each N_train/seed".
VAL_COUNT = max(MIN_VAL_COUNT, round(VAL_FRACTION * max(NESTED_N_TRAIN)))

# Fixed architecture/training hyperparameters, identical for every run at
# every N_train (S4 fairness requirement). Reused verbatim from
# MD/pipeline_config.yaml's ``training.model`` block; only dataset paths,
# logger name and max_epochs (small on purpose -- 8-orbital pilot system,
# not a paper-ready run) differ per run.
MODEL_OVERRIDES: dict[str, Any] = {
    "num_interactions": 1,
    "correlation": 1,
    "max_ell": 2,
    "hidden_irreps": "10x0e + 10x1o + 10x2e",
    "loss": "graph2mat.metrics.block_type_mae",
    "optim_lr": 0.005,
}
PILOT_MAX_EPOCHS = 60

DEEPH_NOT_EVALUATED_REASON = (
    "DeepH raw/global equivalence preflight "
    "(deeph_raw_global_equivalence_preflight.py) requires a DeepH-processed "
    "representation of each sample; that conversion has not been run for "
    "the S2/S3 W90 sampler-family pilot samples, so the basis/units/ordering "
    "gate cannot be evaluated for this step. DeepH is excluded from "
    "quantitative comparison for every group pending that preprocessing run."
)

CSV_FIELDNAMES = [
    "family",
    "dim",
    "k",
    "N_train",
    "seed",
    "H_MAE",
    "H_RMSE",
    "rel_Frob",
    "spectral_err",
    "hermiticity",
    "backend",
    "status",
    "available_n",
    "val_n",
    "checkpoint",
    "deeph_included",
    "deeph_exclusion_reason",
    "R_train_max",
    "density_level",
    "sampler_seed",
    "split_seed",
    "training_seed",
]

# Below this pooled-sample count a (family, dim, k) group is a "sparse" design
# (deterministic geometric families: 4-18 samples/group); at or above it, a
# "dense" one (stochastic families: 128 samples/group). Used only to keep two
# distinct designs sharing the same (family, dim, k) from silently colliding
# in the resume cache (see ``cache_domain_matches``).
DENSITY_SPARSE_MAX = 20


@dataclass(frozen=True)
class Sample:
    sample_id: str
    family: str
    dim: str
    k: int
    amplitude_ang: float
    run_fdf: Path
    reference_dir: Path
    reference_matrix: Path
    # Effective per-atom max displacement norm (Ang) actually realized by this
    # geometry (``Configuration.metadata["max_displacement_ang"]``), distinct
    # from ``amplitude_ang`` (the nominal batch/generator parameter). Box
    # samplers (random_cartesian/sobol_sparse) can realize any effective norm
    # up to (and usually well below) their nominal amplitude, so OOD
    # classification must use this field, not the nominal batch label.
    # Optional/defaulted so pre-existing call sites are unaffected.
    max_displacement_ang: float | None = None


# --------------------------------------------------------------------------
# Sample inventory (pure, no subprocess/training)
# --------------------------------------------------------------------------


def load_s3_samples(s3_root: Path = DEFAULT_S3_ROOT) -> list[Sample]:
    """One Sample per usable, non-reference S3 structure (S1/S3 conventions)."""

    structures_root = s3_root / "structures"
    hamiltonians_root = s3_root / "siesta_hamiltonians"
    samples: list[Sample] = []
    for struct_dir in sorted(structures_root.iterdir()):
        if not struct_dir.is_dir() or struct_dir.name == "reference":
            continue
        metadata = json.loads((struct_dir / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("family") in (None, "reference"):
            continue
        reference_dir = hamiltonians_root / struct_dir.name
        selection = choose_reference_matrix(reference_dir, require_positive_provenance=False)
        if not selection.ok or selection.path is None:
            continue
        samples.append(
            Sample(
                sample_id=struct_dir.name,
                family=str(metadata["family"]),
                dim=str(metadata["dimensionality"]),
                k=int(metadata["k"]),
                amplitude_ang=float(metadata["amplitude_ang"]),
                run_fdf=struct_dir / "RUN.fdf",
                reference_dir=reference_dir,
                reference_matrix=selection.path,
                max_displacement_ang=(
                    float(metadata["max_displacement_ang"]) if "max_displacement_ang" in metadata else None
                ),
            )
        )
    return samples


def group_key(sample: Sample) -> tuple[str, str, int]:
    return (sample.family, sample.dim, sample.k)


def group_samples(samples: list[Sample]) -> dict[tuple[str, str, int], list[Sample]]:
    """Pool samples across both pilot amplitudes per (family, dim, k)."""

    groups: dict[tuple[str, str, int], list[Sample]] = {}
    for sample in samples:
        groups.setdefault(group_key(sample), []).append(sample)
    for pool in groups.values():
        pool.sort(key=lambda s: s.sample_id)
    return groups


def val_count_for(n_train: int) -> int:
    return max(MIN_VAL_COUNT, round(VAL_FRACTION * n_train))


def _stable_split_seed(seed: int, pool: list[Sample]) -> int:
    """SHA-256-derived RNG seed: reproducible across processes.

    Python's builtin ``hash()`` on a tuple containing strings is salted per
    process (``PYTHONHASHSEED``) unless explicitly pinned, so the previous
    ``(seed, tuple(...)).__hash__()`` gave a *different* split in every fresh
    interpreter -- silently non-reproducible across process boundaries (e.g.
    a resumed/parallelized run). SHA-256 of a plain string is stable by
    construction.
    """

    key = f"{seed}|" + "|".join(sample.sample_id for sample in pool)
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def select_split(pool: list[Sample], n_train: int, seed: int) -> tuple[list[Sample], list[Sample]] | None:
    """Deterministic seeded (train, val) selection from ``pool``, or None if too small.

    Nested by construction: the shuffle order depends only on ``(seed, pool)``
    -- never on ``n_train`` -- so the fixed-size validation split is identical
    across every N_train drawn from the same (pool, seed), and
    ``train(16) subset train(32) subset train(64)`` because each is a prefix
    of the same post-validation ordering.
    """

    needed = n_train + VAL_COUNT
    if len(pool) < needed:
        return None
    rng = random.Random(_stable_split_seed(seed, pool))
    shuffled = list(pool)
    rng.shuffle(shuffled)
    val = shuffled[:VAL_COUNT]
    train_pool = shuffled[VAL_COUNT:]
    return train_pool[:n_train], val


def r_train_max_for_pool(pool: list[Sample]) -> float:
    return max((sample.amplitude_ang for sample in pool), default=0.0)


def density_level_for_pool(pool: list[Sample]) -> str:
    return "sparse" if len(pool) < DENSITY_SPARSE_MAX else "dense"


def cache_domain_matches(cached_row: dict[str, str], pool: list[Sample]) -> bool:
    """False if a resumed CSV row was computed from a different design domain.

    S4's resume key is ``(family, dim, k, N_train, seed)`` -- two distinct
    designs (different amplitude domain / sample density) can share that
    tuple and must not be silently treated as the same cached result. Legacy
    rows written before this fix carry no ``R_train_max``/``density_level``
    and are treated as matching (preserves old resume behavior for them).
    """

    cached_r_train_max = cached_row.get("R_train_max", "")
    cached_density = cached_row.get("density_level", "")
    if cached_r_train_max == "" or cached_density == "":
        return True
    try:
        r_train_max_ok = abs(float(cached_r_train_max) - r_train_max_for_pool(pool)) < 1e-9
    except ValueError:
        return True
    return r_train_max_ok and cached_density == density_level_for_pool(pool)


# --------------------------------------------------------------------------
# Graph2Mat config + training (reuses the MD training-block schema/CLI)
# --------------------------------------------------------------------------


def _symlink_samples(dest_dir: Path, samples: list[Sample]) -> None:
    """Symlink each sample's whole SIESTA reference dir (RUN.fdf + TSHS/HSX/...).

    Graph2Mat's dataset loader resolves the reference matrix relative to
    RUN.fdf's own directory, so the two must live together -- exactly the
    layout S3 already wrote under ``siesta_hamiltonians/<sample_id>/``.
    """

    dest_dir.mkdir(parents=True, exist_ok=True)
    wanted = {sample.sample_id for sample in samples}
    for existing in list(dest_dir.iterdir()):
        if existing.name in wanted:
            continue
        if existing.is_symlink() or existing.is_file():
            existing.unlink()
        else:
            shutil.rmtree(existing)
    for sample in samples:
        link = dest_dir / sample.sample_id
        if not link.exists():
            link.symlink_to(sample.reference_dir)


def build_graph2mat_config(
    run_dir: Path,
    train_samples: list[Sample],
    val_samples: list[Sample],
    *,
    run_name: str,
    max_epochs: int = PILOT_MAX_EPOCHS,
    accelerator: str = "cpu",
    training_seed: int = 0,
    md_config_path: Path = DEFAULT_MD_CONFIG,
    model_overrides: dict[str, Any] | None = None,
    batch_size: int | None = None,
    optim_lr: float | None = None,
    early_stopping_patience: int | None = None,
    precision: str | None = None,
    torch_float32_matmul_precision: str | None = None,
    loader_threads: int | None = None,
) -> Path:
    """Write a Graph2Mat training config sharing ``MODEL_OVERRIDES`` across every run."""

    md_config = yaml.safe_load(md_config_path.read_text(encoding="utf-8"))
    block = md_config["training"]

    train_dir = run_dir / "train_samples"
    val_dir = run_dir / "validation_samples"
    _symlink_samples(train_dir, train_samples)
    _symlink_samples(val_dir, val_samples)

    # graph2mat's CLI resolves data-module glob paths relative to root_dir
    # (the config's own directory, since we run with cwd=run_dir) -- an
    # absolute pattern hits pathlib's "Non-relative patterns are
    # unsupported", matching the relative-glob convention already used by
    # MD/pipeline_config.yaml and write_graph2mat_configs.py.
    def _relative(path: Path) -> str:
        return os.path.relpath(path, run_dir)

    data = dict(block["data"])
    data.update(
        {
            "basis_files": _relative(Path(DEFAULT_BASIS_GLOB)),
            "train_runs": _relative(train_dir) + "/*/RUN.fdf",
            "val_runs": _relative(val_dir) + "/*/RUN.fdf",
            "batch_size": min(batch_size or 10, len(train_samples)),
        }
    )
    if loader_threads is not None:
        data["loader_threads"] = int(loader_threads)
    model = dict(model_overrides or MODEL_OVERRIDES)
    if optim_lr is not None:
        model["optim_lr"] = optim_lr
    trainer = dict(block["trainer"])
    trainer["accelerator"] = accelerator
    trainer["max_epochs"] = int(max_epochs)
    if early_stopping_patience is not None:
        # Matches Comparison/results/ml_vs_siesta_cross_structure_sweep/*/graph2mat/
        # pipeline_config.yaml's callbacks verbatim (monitor=val_loss, mode=min,
        # min_delta=0.0, strict=true) so max_epochs acts as a cap, not a fixed budget.
        trainer["callbacks"] = [
            {"class_path": "EarlyStopping", "init_args": {
                "monitor": "val_loss", "mode": "min", "patience": int(early_stopping_patience),
                "min_delta": 0.0, "strict": True,
            }},
        ]
    trainer["logger"] = {
        "class_path": "TensorBoardLogger",
        "init_args": {"name": run_name, "save_dir": "lightning_logs"},
    }
    if precision is not None:
        trainer["precision"] = precision

    # Distinct from sampler_seed (S3 geometry generation) and split_seed (S4
    # train/val selection): the graph2mat CLI's LightningCLI-style top-level
    # ``seed_everything`` key, so training is itself reproducible instead of
    # implicitly unseeded.
    config: dict[str, Any] = {"seed_everything": int(training_seed), "data": data, "model": model, "trainer": trainer}
    if torch_float32_matmul_precision is not None:
        # NOTE: unlike ``precision`` (a real trainer.fit CLI arg), the graph2mat CLI's
        # "fit" subcommand rejects an unrecognized top-level "torch_float32_matmul_precision"
        # key outright ("error: Subcommand 'fit' does not accept option
        # ...") -- verified empirically. In the repo's other sections this key lives in
        # MD/pipeline_config.yaml's orchestrator-level config (g2m_deeph_runner.py), which
        # calls torch.set_float32_matmul_precision(...) itself in-process before invoking
        # graph2mat as a subprocess; since we invoke the graph2mat CLI directly, that
        # setting has no CLI-config equivalent to pass through here.
        pass
    config_path = run_dir / "config.yaml"
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "# Generated by dataset_design_w90_001_s4_train_learning_curves.py\n"
        + yaml.safe_dump(config, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    return config_path


def torch_backend_preflight() -> dict[str, Any]:
    """Cheap CUDA availability check (default compute policy: prefer GPU, record evidence)."""

    try:
        import torch

        available = bool(torch.cuda.is_available())
        return {
            "requested_backend": "gpu" if available else "cpu",
            "effective_backend": "gpu" if available else "cpu",
            "cuda_available": available,
            "device_name": torch.cuda.get_device_name(0) if available else None,
            "reason": None if available else "torch.cuda.is_available() is False",
        }
    except Exception as exc:  # noqa: BLE001 - a broken torch import is a legitimate CPU fallback
        return {
            "requested_backend": "gpu",
            "effective_backend": "cpu",
            "cuda_available": False,
            "device_name": None,
            "reason": f"{type(exc).__name__}: {exc}",
        }


TORCH_COMPAT_DIR = REPO_ROOT / "scripts" / "torch_serialization_compat"


def run_graph2mat_training(config_path: Path, run_dir: Path) -> Path:
    """Invoke the real ``graph2mat`` CLI trainer; return the newest checkpoint written.

    The bare ``graph2mat`` CLI is a separate process, so the e3nn/PyTorch
    checkpoint-loading safe-globals fix (S1 §4 --
    ``torch_safe_globals.allow_graph2mat_checkpoint_globals``, must run
    before e3nn is imported) is applied via its ``sitecustomize.py`` on
    ``PYTHONPATH``, the same mechanism ``predict_model_on_dataset.py``
    already relies on for its own subprocess entrypoints.
    """

    log_path = run_dir / "train.log"
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(TORCH_COMPAT_DIR), env.get("PYTHONPATH", "")])
    )
    with log_path.open("w", encoding="utf-8") as log_file:
        completed = subprocess.run(
            [str(GRAPH2MAT_BIN), "models", "mace", "main", "fit", "-c", str(config_path.name)],
            cwd=run_dir,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"graph2mat training failed (rc={completed.returncode}); see {log_path}")
    checkpoints = sorted(run_dir.rglob("*.ckpt"), key=lambda p: p.stat().st_mtime)
    if not checkpoints:
        raise RuntimeError(f"graph2mat training produced no checkpoint under {run_dir}")
    return checkpoints[-1]


# --------------------------------------------------------------------------
# Prediction + metrics (reuses predict_model_on_dataset.py and sisl)
# --------------------------------------------------------------------------


def write_val_manifest(manifest_path: Path, val_samples: list[Sample]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "sample_id", "method", "source_run", "source_sample_id", "structure_path",
                "hamiltonian_path", "run_out_path", "metadata_path", "valid", "split",
                "status", "sample_dir",
            ]
        )
        for sample in val_samples:
            writer.writerow(
                [
                    sample.sample_id, "graph2mat_w90_pilot", str(sample.run_fdf), sample.sample_id,
                    str(sample.run_fdf), str(sample.reference_matrix),
                    str(sample.reference_dir / "RUN.out"), str(sample.reference_dir / "metadata.json"),
                    "True", "test", "ok", str(sample.reference_dir),
                ]
            )


def run_prediction(checkpoint: Path, manifest_path: Path, output_dir: Path, accelerator: str) -> Path:
    completed = subprocess.run(
        [
            str(PYTHON), str(SCRIPT_DIR / "predict_model_on_dataset.py"),
            "--checkpoint", str(checkpoint),
            "--train-method", "md",
            "--test-set", "w90_s4_validation",
            "--test-manifest", str(manifest_path),
            "--output-dir", str(output_dir),
            "--basis-files", DEFAULT_BASIS_GLOB,
            "--matrix-component-policy", "h_only",
            "--n-matrix-components", "1",
            "--accelerator", accelerator,
            "--no-store-in-memory",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    log_path = output_dir / "predict.log"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path.write_text((completed.stdout or "") + (completed.stderr or ""), encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"predict_model_on_dataset.py failed (rc={completed.returncode}); see {log_path}")
    return output_dir / "predicted_hamiltonians"


def gamma_hk(matrix_path: Path) -> Any:
    import numpy as np
    import sisl

    hamiltonian = sisl.get_sile(str(matrix_path)).read_hamiltonian()
    return np.asarray(hamiltonian.Hk(k=[0.0, 0.0, 0.0], format="array"))


def gamma_eigenvalues(matrix_path: Path) -> Any:
    import sisl

    hamiltonian = sisl.get_sile(str(matrix_path)).read_hamiltonian()
    return hamiltonian.eigh(k=[0.0, 0.0, 0.0])


def fermi_level_ev(matrix_path: Path) -> float:
    """Reuses ``run_epc_siesta_reference.fermi_level_ev``'s sisl call; 0.0 if unreadable."""

    import sisl

    try:
        return float(sisl.get_sile(str(matrix_path)).read_fermi_level())
    except Exception:  # noqa: BLE001 - matches fermi_level_ev's own fail-soft contract
        return 0.0


def compute_validation_metrics(
    val_samples: list[Sample],
    predicted_root: Path,
    *,
    spectral_window_ev: float = 2.0,
) -> dict[str, float]:
    """H-MAE, H-RMSE, relative Frobenius, low-energy spectral RMSE, Hermiticity -- averaged."""

    import numpy as np

    mae, rmse, rel_frob, spectral, hermiticity = [], [], [], [], []
    for sample in val_samples:
        predicted_path = predicted_root / sample.sample_id / "ML_prediction.HSX"
        if not predicted_path.exists():
            continue
        h_pred = gamma_hk(predicted_path)
        h_ref = gamma_hk(sample.reference_matrix)
        error = h_pred - h_ref
        mae.append(float(np.mean(np.abs(error))))
        rmse.append(float(np.sqrt(np.mean(np.abs(error) ** 2))))
        ref_norm = float(np.linalg.norm(h_ref))
        rel_frob.append(float(np.linalg.norm(error) / ref_norm) if ref_norm else float("nan"))
        hermiticity.append(float(np.max(np.abs(h_pred - h_pred.conj().T))))

        eig_pred = gamma_eigenvalues(predicted_path)
        eig_ref = gamma_eigenvalues(sample.reference_matrix)
        # Reference E_F: predicted matrices are ML output, not SIESTA state,
        # so they carry no independent Fermi level to read.
        e_fermi = fermi_level_ev(sample.reference_matrix)
        window_mask = np.abs(eig_ref - e_fermi) <= spectral_window_ev
        if np.any(window_mask):
            spectral.append(float(np.sqrt(np.mean((eig_pred[window_mask] - eig_ref[window_mask]) ** 2))))

    def _mean(values: list[float]) -> float:
        return float(np.mean(values)) if values else float("nan")

    return {
        "H_MAE": _mean(mae),
        "H_RMSE": _mean(rmse),
        "rel_Frob": _mean(rel_frob),
        "spectral_err": _mean(spectral),
        "hermiticity": _mean(hermiticity),
        "n_evaluated": len(mae),
    }


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def read_existing_rows(csv_path: Path) -> dict[tuple[str, str, str, str, str], dict[str, str]]:
    if not csv_path.exists():
        return {}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {(r["family"], r["dim"], r["k"], r["N_train"], r["seed"]): r for r in rows}


def write_csv(csv_path: Path, rows: list[dict[str, Any]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in CSV_FIELDNAMES})


def run_one_combo(
    group: tuple[str, str, int],
    pool: list[Sample],
    n_train: int,
    seed: int,
    *,
    output_root: Path,
) -> dict[str, Any]:
    family, dim, k = group
    row: dict[str, Any] = {
        "family": family, "dim": dim, "k": k, "N_train": n_train, "seed": seed,
        "available_n": len(pool),
        "R_train_max": r_train_max_for_pool(pool),
        "density_level": density_level_for_pool(pool),
        # Three distinct seeds, previously collapsed into the single reused
        # ``seed`` loop variable: sampler_seed generated the S3 geometries
        # (fixed pilot constant), split_seed picks train/val from that pool,
        # training_seed seeds graph2mat training itself.
        "sampler_seed": pilot.PILOT_SEED,
        "split_seed": seed,
        "training_seed": seed,
        "deeph_included": False,
        "deeph_exclusion_reason": DEEPH_NOT_EVALUATED_REASON,
    }
    split = select_split(pool, n_train, seed)
    if split is None:
        row.update(status="insufficient_samples", backend="", val_n=0)
        return row
    train_samples, val_samples = split
    row["val_n"] = len(val_samples)

    run_name = f"{family}__{dim}__k{k}__n{n_train}__seed{seed}"
    run_dir = output_root / "runs" / run_name

    backend_info = torch_backend_preflight()
    accelerator = backend_info["effective_backend"]
    row["backend"] = accelerator

    try:
        config_path = build_graph2mat_config(
            run_dir, train_samples, val_samples, run_name=run_name, accelerator=accelerator,
            training_seed=seed,
        )
        checkpoint = run_graph2mat_training(config_path, run_dir)
        row["checkpoint"] = str(checkpoint)

        manifest_path = run_dir / "validation_manifest.csv"
        write_val_manifest(manifest_path, val_samples)
        predicted_root = run_prediction(checkpoint, manifest_path, run_dir / "prediction", accelerator)

        metrics = compute_validation_metrics(val_samples, predicted_root)
        row.update(metrics)
        row["status"] = "ok" if metrics["n_evaluated"] > 0 else "prediction_incomplete"
    except Exception as exc:  # noqa: BLE001 - a failed run is a recorded row, not a crash
        row["status"] = f"error: {type(exc).__name__}: {exc}"[:500]
    return row


def build_plan(samples: list[Sample]) -> dict[tuple[str, str, int], list[Sample]]:
    return group_samples(samples)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s3-root", type=Path, default=DEFAULT_S3_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--only-family", default=None, help="Debug: restrict to one family.")
    parser.add_argument("--only-n-train", type=int, default=None, help="Debug: restrict to one N_train.")
    parser.add_argument("--only-seed", type=int, default=None, help="Debug: restrict to one seed.")
    parser.add_argument("--max-combos", type=int, default=None, help="Debug/testing cap on total combos run.")
    args = parser.parse_args()

    samples = load_s3_samples(args.s3_root)
    groups = build_plan(samples)
    csv_path = args.output_root / "learning_curves.csv"
    existing = read_existing_rows(csv_path)
    rows = list(existing.values())

    n_trains = (args.only_n_train,) if args.only_n_train else NESTED_N_TRAIN
    seeds = (args.only_seed,) if args.only_seed is not None else SEEDS
    combos_run = 0
    for group, pool in sorted(groups.items()):
        family, dim, k = group
        if args.only_family and family != args.only_family:
            continue
        for n_train in n_trains:
            for seed in seeds:
                key = (family, dim, str(k), str(n_train), str(seed))
                stale_cache_row = existing.get(key)
                if stale_cache_row is not None and cache_domain_matches(stale_cache_row, pool):
                    continue
                if args.max_combos is not None and combos_run >= args.max_combos:
                    write_csv(csv_path, rows)
                    print(json.dumps({"combos_run": combos_run, "csv": str(csv_path), "status": "max_combos_reached"}))
                    return 0
                row = run_one_combo(group, pool, n_train, seed, output_root=args.output_root)
                if stale_cache_row is not None and stale_cache_row in rows:
                    rows.remove(stale_cache_row)
                rows.append(row)
                write_csv(csv_path, rows)
                combos_run += 1
                print(json.dumps({k: row.get(k) for k in ("family", "dim", "N_train", "seed", "status", "backend", "H_MAE")}))

    write_csv(csv_path, rows)
    print(json.dumps({"combos_run": combos_run, "total_rows": len(rows), "csv": str(csv_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
