#!/usr/bin/env python3
"""E-F_001-S37 / D06: synthetic-displacement resource benchmark for rigid MATBG.

Measures, on the rigid ``(31, 30)`` magic-angle cell, the wall time / RSS /
VRAM / sparse NNZ / disk cost of the pieces a physical selected-mode EPC
campaign will need: the Graph2Mat directional JVP (+ its sparse
serialisation), the "contract before materialise" VJP alternative, and one
persisted electronic eigenspace from the real sparse solver. It also compares
"materialize-once" (:func:`graph2mat_autograd_derivatives.compute_graph2mat_directional_derivative`,
reused ``nk * band_pairs`` times) against "contract-before-materialize"
(:func:`graph2mat_autograd_derivatives.compute_graph2mat_vjp_contraction`, one
call per requested block) over a grid of ``(nk, band_pairs)``.

Every direction here is a :mod:`fd_perturbation_space` coordinate pattern
(breathing/shear/optical/random-normalised), not a diagonalised
dynamical-matrix eigenvector: no mass weighting, no frequency, no phonon
provider. Roadmap Fase 10b/GO-8 requires these stay tagged
:data:`DISPLACEMENT_CLASS` and never ``phonon_eigenmode`` or a PAO-projected coupling;
:func:`assert_no_forbidden_labels` is the mechanical guard for that, run on
every artifact this script writes.

This is a cost-model measurement, not a PAO-covariant coupling calculation: GO-8a
(a validated MATBG PhononProvider) and GO-1/GO-4 (the moving-PAO formalism)
are still open, so nothing here is contracted into a Hamiltonian-electron
coupling.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for _path in (SCRIPT_DIR, REPO_ROOT / "shared", REPO_ROOT / "scripts" / "torch_serialization_compat"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import fd_perturbation_space as fdp  # noqa: E402
from artifact_signature import file_sha256, input_signature_sha256  # noqa: E402
from run_deeph_sparse_spectrum import (  # noqa: E402
    available_memory_bytes,
    free_disk_percent,
    gpu_status,
)

DISPLACEMENT_CLASS = "synthetic_test_displacement"
FORBIDDEN_LABELS = ("phonon_eigenmode", "physical_g", "g_mn_nu", "g_mnnu", "physical_phonon")
SCHEMA = "matbg_synthetic_displacement_benchmark_v1"

# Roadmap Section X operational headroom: never plan past these.
VRAM_MARGIN_GIB = 28.0
RAM_MARGIN_GIB = 62.0

DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "Comparison/results/tbg_registry_spectral_loss/training/checkpoints"
    / "spectral-best-epoch=247-step=03472.ckpt"
)
DEFAULT_STRUCTURE_FDF = REPO_ROOT / "materials/twisted_bilayer_graphene_1p084549deg/RUN.fdf"
DEFAULT_BASIS_FILES = REPO_ROOT / "Comparison/results/tbg_pure_graph2mat/target/material_basis/*.ion.xml"
DEFAULT_SOLVER_INPUT = REPO_ROOT / "Comparison/results/tbg_pure_graph2mat/prediction/solver_input"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "Comparison/results/tbg_pure_graph2mat/epc_synthetic_benchmark"
DEFAULT_NK_GRID = (1, 4, 16)
DEFAULT_BAND_PAIRS_GRID = (1, 10, 50)


class BenchmarkError(RuntimeError):
    """A precondition (checkpoint/structure/margin) this benchmark needs is missing."""


def assert_no_forbidden_labels(payload: Any, *, path: str = "$") -> None:
    """Recursively refuse ``phonon_eigenmode``/physical-``g`` labels anywhere in a payload."""
    if isinstance(payload, str):
        lowered = payload.lower()
        for forbidden in FORBIDDEN_LABELS:
            if forbidden in lowered:
                raise BenchmarkError(f"{path}: forbidden label {forbidden!r} found in {payload!r}")
        return
    if isinstance(payload, dict):
        for key, value in payload.items():
            assert_no_forbidden_labels(key, path=f"{path}.{key}")
            assert_no_forbidden_labels(value, path=f"{path}.{key}")
        return
    if isinstance(payload, (list, tuple)):
        for index, item in enumerate(payload):
            assert_no_forbidden_labels(item, path=f"{path}[{index}]")


def write_json(path: Path, payload: dict) -> None:
    assert_no_forbidden_labels(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def rss_bytes() -> int:
    """Peak RSS of this process so far (Linux: ru_maxrss is already in KiB)."""
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def total_rss_bytes(*, child_rss_delta_kib: float = 0.0) -> int:
    """Peak RSS of this process plus every reaped child (e.g. the Julia solver).

    ``rss_bytes()`` alone is blind to ``benchmark_eigenspace``'s subprocess,
    which is exactly the piece GO-8's 62 GB margin exists to bound.
    """
    return rss_bytes() + int(max(0.0, child_rss_delta_kib) * 1024)


def sparse_bytes(matrix: Any) -> int:
    return int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)


def timed(fn: Callable[[], Any]) -> tuple[Any, float]:
    started = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - started


def vram_snapshot(torch_module, device: str) -> dict[str, float] | None:
    if device != "cuda" or torch_module is None or not torch_module.cuda.is_available():
        return None
    return {
        "peak_allocated_bytes": float(torch_module.cuda.max_memory_allocated()),
        "peak_reserved_bytes": float(torch_module.cuda.max_memory_reserved()),
    }


# --------------------------------------------------------------------------- #
# Stage 1: load the real checkpoint + (31, 30) structure in-process.
# --------------------------------------------------------------------------- #


def load_checkpoint_and_batch(
    checkpoint: Path,
    structure_fdf: Path,
    basis_files: str,
    *,
    matrix_component_policy: str = "h_only",
    n_matrix_components: int = 1,
    tmp_dir: Path,
):
    """Model + one-structure batch, mirroring the GO-3 test loader (no chunking:
    the directional JVP is one double-backward, not the batched-VJP jacobian
    route that needed MACE node/edge chunking to bound memory).
    """
    # Resolved to absolute *before* the os.chdir below, or a relative CLI path
    # would be re-anchored against the checkpoint's training directory instead
    # of the caller's cwd.
    checkpoint = checkpoint.expanduser().resolve(strict=False)
    structure_fdf = structure_fdf.expanduser().resolve(strict=False)
    if not checkpoint.is_file():
        raise BenchmarkError(f"No Graph2Mat checkpoint at {checkpoint}")
    if not structure_fdf.is_file():
        raise BenchmarkError(f"No structure RUN.fdf at {structure_fdf}")

    from torch_safe_globals import allow_graph2mat_checkpoint_globals

    allow_graph2mat_checkpoint_globals()

    import torch
    from graph2mat.tools.lightning import MatrixDataModule
    from graph2mat.tools.lightning.models.mace import LitMACEMatrixModel

    from predict_model_on_dataset import checkpoint_training_dir, normalize_pattern_for_workdir

    run_cwd = checkpoint_training_dir(checkpoint)
    source_cwd = Path.cwd()
    tmp_dir = tmp_dir.expanduser().resolve(strict=False)
    runs_json = tmp_dir / "benchmark_runs.json"
    runs_json.parent.mkdir(parents=True, exist_ok=True)
    runs_json.write_text(json.dumps({"predict": [str(structure_fdf)]}), encoding="utf-8")
    basis_files = normalize_pattern_for_workdir(basis_files, source_cwd=source_cwd, target_cwd=run_cwd)
    try:
        os.chdir(run_cwd)
        model = LitMACEMatrixModel.load_from_checkpoint(
            str(checkpoint), map_location="cpu", weights_only=False
        )
        model.eval()
        datamodule = MatrixDataModule(
            out_matrix="hamiltonian",
            symmetric_matrix=True,
            sub_point_matrix=False,
            basis_files=basis_files,
            runs_json=str(runs_json),
            store_in_memory=True,
            batch_size=1,
            n_matrix_components=n_matrix_components,
            matrix_component_policy=matrix_component_policy,
        )
        datamodule.setup("predict")
        batch = next(iter(datamodule.predict_dataloader()))
    finally:
        os.chdir(source_cwd)
    return torch, model.model, batch, datamodule.data_processor


# --------------------------------------------------------------------------- #
# Stage 2: directional JVP (materialize-once) + VJP contraction
# (contract-before-materialize), per synthetic direction.
# --------------------------------------------------------------------------- #


def direction_signature(
    *, checkpoint_sha256: str, structure_sha256: str, direction, backend_effective: str
) -> str:
    return input_signature_sha256(
        {
            "benchmark_schema": SCHEMA,
            "checkpoint_sha256": checkpoint_sha256,
            "structure_sha256": structure_sha256,
            "direction_hash": direction.direction_hash,
            "direction_kind": direction.kind,
            "displacement_class": DISPLACEMENT_CLASS,
            "backend_effective": backend_effective,
        }
    )


def benchmark_direction(
    torch_module,
    model,
    batch,
    data_processor,
    direction,
    *,
    change_of_basis,
    backend_record,
    seed: int,
) -> dict[str, Any]:
    from graph2mat_autograd_derivatives import (
        compute_graph2mat_directional_derivative,
        compute_graph2mat_vjp_contraction,
    )

    if backend_record.effective == "cuda":
        torch_module.cuda.reset_peak_memory_stats()
    rss_before = rss_bytes()
    result, jvp_wall_seconds = timed(
        lambda: compute_graph2mat_directional_derivative(
            model,
            batch,
            direction,
            change_of_basis=change_of_basis,
            data_processor=data_processor,
            backend=backend_record,
        )
    )
    jvp_vram = vram_snapshot(torch_module, backend_record.effective)
    rss_after_jvp = rss_bytes()
    matrix = result.matrices[0]

    # Contract-before-materialize: a synthetic per-block cotangent (this is a
    # resource-cost proxy for "one requested electronic block's weight", not a
    # physically meaningful contraction weight -- see module docstring).
    generator = np.random.default_rng(hash((direction.name, seed)) % (2**32))
    cotangent = generator.standard_normal(result.metadata["n_outputs"])
    if backend_record.effective == "cuda":
        torch_module.cuda.reset_peak_memory_stats()
    vjp_result, vjp_wall_seconds = timed(
        lambda: compute_graph2mat_vjp_contraction(
            model, batch, cotangent, direction, change_of_basis=change_of_basis, backend=backend_record
        )
    )
    vjp_vram = vram_snapshot(torch_module, backend_record.effective)
    rss_after_vjp = rss_bytes()

    return {
        "displacement_class": DISPLACEMENT_CLASS,
        "direction_name": direction.name,
        "direction_kind": direction.kind,
        "direction_hash": direction.direction_hash,
        "materialize_once": {
            "method": result.metadata["derivative_method"],
            "wall_seconds": jvp_wall_seconds,
            "rss_delta_bytes": rss_after_jvp - rss_before,
            "vram": jvp_vram,
            "nnz": result.metadata.get("nnz"),
            "matrix_shape": result.metadata.get("matrix_shape"),
            "sparse_bytes": sparse_bytes(matrix),
        },
        "contract_before_materialize": {
            "method": vjp_result.metadata["derivative_method"],
            "wall_seconds": vjp_wall_seconds,
            "rss_delta_bytes": rss_after_vjp - rss_after_jvp,
            "vram": vjp_vram,
            "value": vjp_result.value,
        },
        "backend": backend_record.to_metadata(),
    }


# --------------------------------------------------------------------------- #
# Stage 3: A/B crossover over a (nk, band_pairs) grid.
# --------------------------------------------------------------------------- #


def dense_contract_rate_seconds_per_block(matrix, *, n_orbitals: int, band_pairs_samples: tuple[int, ...]) -> float:
    """Real wall time of ``C_f^T @ D_H @ C_i`` per requested block (random C)."""
    generator = np.random.default_rng(0)
    rates = []
    for band_pairs in band_pairs_samples:
        c_row = generator.standard_normal((n_orbitals, band_pairs))
        c_col = generator.standard_normal((n_orbitals, band_pairs))
        _, wall_seconds = timed(lambda: (c_row.T @ (matrix @ c_col)))
        rates.append(wall_seconds / band_pairs)
    return float(np.mean(rates))


def crossover_table(
    *, t_materialize_seconds: float, t_vjp_contract_seconds: float, dense_contract_rate: float,
    nk_grid: tuple[int, ...], band_pairs_grid: tuple[int, ...],
) -> dict[str, Any]:
    rows = []
    for nk in nk_grid:
        for band_pairs in band_pairs_grid:
            n_blocks = nk * band_pairs
            t_a = t_materialize_seconds + n_blocks * dense_contract_rate
            t_b = n_blocks * t_vjp_contract_seconds
            rows.append(
                {
                    "nk": nk,
                    "band_pairs": band_pairs,
                    "n_blocks": n_blocks,
                    "materialize_once_seconds": t_a,
                    "contract_before_materialize_seconds": t_b,
                    "cheaper": "materialize_once" if t_a <= t_b else "contract_before_materialize",
                }
            )
    denominator = t_vjp_contract_seconds - dense_contract_rate
    crossover_n_blocks = (t_materialize_seconds / denominator) if denominator > 0 else None
    return {
        "t_materialize_seconds": t_materialize_seconds,
        "t_vjp_contract_seconds": t_vjp_contract_seconds,
        "dense_contract_rate_seconds_per_block": dense_contract_rate,
        "crossover_n_blocks": crossover_n_blocks,
        "crossover_policy": (
            "n_blocks below this favors materialize_once reuse; above it favors "
            "contract_before_materialize (roadmap Section X)"
        ),
        "grid": rows,
    }


# --------------------------------------------------------------------------- #
# Stage 4: one persisted electronic eigenspace, via the real sparse solver
# subprocess (Julia/cuDSS backend already bootstrapped in this repo).
# --------------------------------------------------------------------------- #


def benchmark_eigenspace(
    solver_input: Path,
    output_dir: Path,
    *,
    kmesh: tuple[int, int, int],
    num_bands: int,
    window_ev: float,
    backend: str,
    gpu_memory_limit_gib: float,
) -> dict[str, Any]:
    if not (solver_input / "hamiltonians_pred.h5").is_file():
        return {"status": "skipped", "reason": f"no solver input at {solver_input}"}
    command = [
        str(REPO_ROOT / ".venv/bin/python"),
        str(SCRIPT_DIR / "run_deeph_sparse_spectrum.py"),
        "--input-dir", str(solver_input),
        "--output-dir", str(output_dir),
        "--job", "dos",
        "--fermi-level", "0.0",
        "--num-bands", str(num_bands),
        "--kmesh", str(kmesh[0]), str(kmesh[1]), str(kmesh[2]),
        "--backend", backend,
        "--gpu-memory-limit-gib", str(gpu_memory_limit_gib),
        "--persist-eigenspaces",
        "--eigenspace-window-ev", str(window_ev),
    ]
    before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    wall_seconds = time.perf_counter() - started
    after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    if completed.returncode != 0:
        return {
            "status": "failed",
            "reason": completed.stderr[-4000:],
            "command": command,
            "wall_seconds": wall_seconds,
        }
    manifest = read_json(output_dir / "solver_manifest.json")
    return {
        "status": "completed",
        "command": command,
        "wall_seconds": wall_seconds,
        # ponytail: RUSAGE_CHILDREN.ru_maxrss is a high-water mark over all
        # children this process has ever reaped, not just this one -- a lower
        # bound on this call's own peak, matching the convention already used
        # by run_epc_siesta_reference.py / run_graphene_gamma_phonons.py.
        "rss_delta_kib": max(0, after - before),
        "backend_effective": manifest.get("backend_effective"),
        "resources": manifest.get("resources"),
        "cudss_timings": manifest.get("cudss_timings"),
        "eigenvectors_persisted": manifest.get("eigenvectors_persisted"),
        "solver_status": manifest.get("status"),
    }


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def resource_margins(*, peak_rss_bytes: int = 0, peak_vram_bytes: float = 0.0) -> dict[str, Any]:
    """Snapshot of headroom *and* what this run itself consumed against it.

    ``ram_available_gib``/``vram_free_gib`` are the machine's state before/around
    this run (context); ``peak_rss_gib``/``peak_vram_gib`` are what this run
    itself measured, which is what "within margin" (roadmap Section X: no plan
    past 28 GiB VRAM / 62 GB RAM) actually has to be checked against.

    ``peak_rss_bytes`` must already include reaped children (pass
    :func:`total_rss_bytes`, not :func:`rss_bytes`) or the Julia solver
    subprocess is invisible to the RAM margin. ``peak_vram_bytes`` is the
    PyTorch-allocator reading; the margin check below takes the max against
    ``nvidia-smi``'s system-wide ``used_bytes`` so VRAM held by that same
    subprocess (outside PyTorch's allocator) is not silently dropped.
    """
    gpu = gpu_status()
    disk = free_disk_percent(REPO_ROOT)
    ram_available_gib = available_memory_bytes() / 1024**3
    vram_free_gib = (gpu["free_bytes"] / 1024**3) if gpu else None
    gpu_used_gib = (gpu["used_bytes"] / 1024**3) if gpu else 0.0
    peak_rss_gib = peak_rss_bytes / 1024**3
    peak_vram_gib = max(peak_vram_bytes / 1024**3, gpu_used_gib)
    return {
        "ram_available_gib": ram_available_gib,
        "ram_margin_gib": RAM_MARGIN_GIB,
        "peak_rss_gib": peak_rss_gib,
        "ram_within_margin": peak_rss_gib <= RAM_MARGIN_GIB,
        "vram_free_gib": vram_free_gib,
        "vram_margin_gib": VRAM_MARGIN_GIB,
        "peak_vram_gib": peak_vram_gib,
        "vram_within_margin": peak_vram_gib <= VRAM_MARGIN_GIB,
        "free_disk_percent": disk,
        "gpu": gpu,
    }


def go8_verdict(*, direction_results: list[dict], eigenspace_result: dict, margins: dict) -> dict[str, Any]:
    reasons = []
    if not direction_results:
        reasons.append("no synthetic direction was measured")
    if eigenspace_result.get("status") == "failed":
        reasons.append(f"eigenspace stage failed: {eigenspace_result.get('reason', '')[:200]}")
    if margins["vram_free_gib"] is not None and margins["vram_free_gib"] < 1.0:
        reasons.append("less than 1 GiB VRAM free before the run")
    if margins["free_disk_percent"] < 12.0:
        reasons.append(f"free disk {margins['free_disk_percent']:.2f}% below the 12% campaign guard")
    if not margins.get("ram_within_margin", True):
        reasons.append(
            f"measured peak RSS {margins.get('peak_rss_gib', 0):.2f} GiB exceeds the "
            f"{margins.get('ram_margin_gib')} GiB roadmap margin"
        )
    if not margins.get("vram_within_margin", True):
        reasons.append(
            f"measured peak VRAM {margins.get('peak_vram_gib', 0):.2f} GiB exceeds the "
            f"{margins.get('vram_margin_gib')} GiB roadmap margin"
        )
    status = "cost_model_measured_no_blocking_issue" if not reasons else "NO_GO"
    return {
        "status": status,
        "reasons": reasons,
        "note": (
            "This is a per-unit cost model (one JVP, one VJP contraction, one "
            "persisted eigenspace k-point). Extrapolating to a full selected-mode "
            "campaign total needs N_q/N_nu from a validated GO-8a PhononProvider, "
            "which is not yet available; multiply the measured per-unit costs by "
            "that count once it exists instead of guessing it here."
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_dir / "tmp"

    checkpoint_sha256 = file_sha256(args.checkpoint)
    structure_sha256 = file_sha256(args.structure_fdf)

    torch_module, model, batch, data_processor = load_checkpoint_and_batch(
        args.checkpoint,
        args.structure_fdf,
        args.basis_files,
        tmp_dir=tmp_dir,
    )
    n_atoms = int(batch["positions"].shape[0])
    positions = batch["positions"].detach().cpu().numpy()

    from graph2mat_autograd_derivatives import resolve_jvp_backend

    model, batch, backend_record = resolve_jvp_backend(model, batch, args.accelerator)

    change_of_basis = np.asarray(data_processor.basis_table.change_of_basis, dtype=np.float64)
    # positions are stored in the e3nn frame; the direction generators below
    # build patterns in the physical cartesian frame of the fdf structure.
    cartesian_positions = positions @ np.linalg.inv(change_of_basis).T

    directions_dir = output_dir / "directions"
    direction_results: list[dict[str, Any]] = []
    for direction in fdp.synthetic_benchmark_direction_set(cartesian_positions, seed=args.seed):
        signature = direction_signature(
            checkpoint_sha256=checkpoint_sha256,
            structure_sha256=structure_sha256,
            direction=direction,
            backend_effective=backend_record.effective,
        )
        cache_path = directions_dir / f"{direction.name}.json"
        cached = read_json(cache_path)
        if not args.overwrite and cached.get("input_signature_sha256") == signature:
            cached["cache_hit"] = True
            direction_results.append(cached)
            continue
        payload = benchmark_direction(
            torch_module,
            model,
            batch,
            data_processor,
            direction,
            change_of_basis=change_of_basis,
            backend_record=backend_record,
            seed=args.seed,
        )
        payload["input_signature_sha256"] = signature
        payload["cache_hit"] = False
        write_json(cache_path, payload)
        direction_results.append(payload)

    disk_before_eigenspace = free_disk_percent(output_dir)
    eigenspace_result = benchmark_eigenspace(
        args.solver_input,
        output_dir / "eigenspace_probe",
        kmesh=tuple(args.eigenspace_kmesh),
        num_bands=args.eigenspace_num_bands,
        window_ev=args.eigenspace_window_ev,
        backend=args.eigenspace_backend,
        gpu_memory_limit_gib=args.eigenspace_gpu_memory_limit_gib,
    ) if not args.skip_eigenspace_stage else {"status": "skipped", "reason": "--skip-eigenspace-stage"}
    disk_after_eigenspace = free_disk_percent(output_dir)

    materialize_wall = [row["materialize_once"]["wall_seconds"] for row in direction_results if not row.get("cache_hit")]
    contract_wall = [row["contract_before_materialize"]["wall_seconds"] for row in direction_results if not row.get("cache_hit")]
    matrix_shape = next(
        (row["materialize_once"]["matrix_shape"] for row in direction_results if row["materialize_once"].get("matrix_shape")),
        None,
    )
    n_orbitals = int(matrix_shape[0]) if matrix_shape else n_atoms
    crossover = None
    if materialize_wall and contract_wall and matrix_shape:
        from scipy import sparse as scipy_sparse

        # benchmark_direction persists only the sparse matrix's metadata (nnz,
        # shape), not the matrix itself, so this stage never re-runs a JVP just
        # to time a matmul: a same-shaped random sparse matrix at the measured
        # density stands in for the dense-contraction-rate probe.
        density = (direction_results[0]["materialize_once"]["nnz"] or 1) / max(1, n_orbitals * n_orbitals)
        rate = dense_contract_rate_seconds_per_block(
            scipy_sparse.random(n_orbitals, n_orbitals, density=min(density, 1.0), format="csr"),
            n_orbitals=n_orbitals,
            band_pairs_samples=(1, 4),
        )
        crossover = crossover_table(
            t_materialize_seconds=float(np.mean(materialize_wall)),
            t_vjp_contract_seconds=float(np.mean(contract_wall)),
            dense_contract_rate=rate,
            nk_grid=tuple(args.nk_grid),
            band_pairs_grid=tuple(args.band_pairs_grid),
        )

    peak_vram_bytes = max(
        (
            (row.get(stage, {}).get("vram") or {}).get("peak_allocated_bytes", 0.0)
            for row in direction_results
            for stage in ("materialize_once", "contract_before_materialize")
        ),
        default=0.0,
    )
    margins = resource_margins(
        peak_rss_bytes=total_rss_bytes(
            child_rss_delta_kib=eigenspace_result.get("rss_delta_kib", 0.0)
        ),
        peak_vram_bytes=peak_vram_bytes,
    )
    verdict = go8_verdict(direction_results=direction_results, eigenspace_result=eigenspace_result, margins=margins)

    report = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "displacement_class": DISPLACEMENT_CLASS,
        "physical_phonon_labels_used": False,
        "system": {
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": checkpoint_sha256,
            "structure_fdf": str(args.structure_fdf),
            "structure_sha256": structure_sha256,
            "n_atoms": n_atoms,
            "n_orbitals": n_orbitals,
        },
        "backend_preflight": backend_record.to_metadata(),
        "directions": direction_results,
        "crossover": crossover,
        "eigenspace_probe": eigenspace_result,
        "resource_margins": margins,
        "disk_free_percent_before_eigenspace_stage": disk_before_eigenspace,
        "disk_free_percent_after_eigenspace_stage": disk_after_eigenspace,
        "cache": {
            "hits": sum(1 for row in direction_results if row.get("cache_hit")),
            "total_directions": len(direction_results),
            "checkpoint_granularity": "per_direction_json_and_per_solver_subprocess_call",
        },
        "restart_loss_upper_bound_seconds": max(
            [*materialize_wall, *contract_wall, eigenspace_result.get("wall_seconds", 0.0) or 0.0],
            default=0.0,
        ),
        "go8_verdict": verdict,
    }
    write_json(output_dir / "matbg_synthetic_benchmark_report.json", report)
    return report


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--structure-fdf", type=Path, default=DEFAULT_STRUCTURE_FDF)
    parser.add_argument("--basis-files", default=str(DEFAULT_BASIS_FILES))
    parser.add_argument("--solver-input", type=Path, default=DEFAULT_SOLVER_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--accelerator", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nk-grid", type=int, nargs="+", default=list(DEFAULT_NK_GRID))
    parser.add_argument("--band-pairs-grid", type=int, nargs="+", default=list(DEFAULT_BAND_PAIRS_GRID))
    parser.add_argument("--eigenspace-kmesh", type=int, nargs=3, default=(1, 1, 1))
    parser.add_argument("--eigenspace-num-bands", type=int, default=4)
    parser.add_argument("--eigenspace-window-ev", type=float, default=0.05)
    parser.add_argument("--eigenspace-backend", choices=["cpu_mkl_pardiso", "gpu_cudss"], default="gpu_cudss")
    parser.add_argument("--eigenspace-gpu-memory-limit-gib", type=float, default=VRAM_MARGIN_GIB)
    parser.add_argument("--skip-eigenspace-stage", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_argument_parser().parse_args()
    report = run(args)
    print(json.dumps(report["go8_verdict"], indent=2))
    return 0 if report["go8_verdict"]["status"] != "NO_GO" else 1


if __name__ == "__main__":
    raise SystemExit(main())
