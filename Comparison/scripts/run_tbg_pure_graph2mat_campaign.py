#!/usr/bin/env python3
"""Run the gated pure-TBG Graph2Mat campaign from training to UI artifacts."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
ROOT = REPO / "Comparison/results/tbg_pure_graph2mat"
DATASET = REPO / "Comparison/datasets/tbg_pure_md_nested/n474"
BASE_PAYLOAD = REPO / "Comparison/config/tbg_pure_n30_train_payload.json"
TARGET_FDF = REPO / "materials/twisted_bilayer_graphene_1p084549deg/RUN.fdf"
PYTHON = REPO / ".venv/bin/python"
GATE_EV = 0.010
MIN_FREE_DISK_PERCENT = 12.0

sys.path.insert(0, str(SCRIPTS))
from export_siesta_hamiltonian_to_deeph import export as export_hamiltonian  # noqa: E402
from generate_siesta_overlap_only import generate as generate_overlap  # noqa: E402
from run_graphene_hbn_moire_spectral_campaign import _link_exact_overlap_inputs  # noqa: E402
from run_deeph_sparse_spectrum import projected_dos_observables  # noqa: E402
from run_graphene_unfolded_spectrum import run_layer as run_unfolded_layer  # noqa: E402

# E-F_001-S42: the rigid-Gamma raw-derivative stage. Kept as plain imports of
# the same generic modules run_graphene_gamma_epc_paths.py already exercises
# for graphene, not a parallel campaign runner.
import graph2mat_autograd_derivatives as epc_g2m_deriv  # noqa: E402
import preregister_matbg_gamma_rigid_epc as epc_prereg  # noqa: E402
import quantify_checkpoint_derivative_error as epc_c14  # noqa: E402
from artifact_signature import SYNTHETIC_TEST_DISPLACEMENT, file_sha256, input_signature_sha256  # noqa: E402
from certify_siesta_dhsdr import difference_norms, norms  # noqa: E402


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def update_status(stage: str, state: str = "running", **extra) -> None:
    status = read_json(ROOT / "status.json")
    status.update(
        {
            "state": state,
            "stage": stage,
            "running": state == "running",
            "pid": os.getpid() if state == "running" else None,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "deep_h_excluded": True,
            "minimum_free_disk_percent": MIN_FREE_DISK_PERCENT,
            # update() merges, so a stale "error" from an earlier failed run would
            # survive a later success and misreport a healthy campaign as broken.
            **({} if state == "failed" else {"error": None}),
            **extra,
        }
    )
    write_json(ROOT / "status.json", status)


def disk_guard() -> None:
    usage = shutil.disk_usage(REPO)
    free = 100.0 * usage.free / usage.total
    if free < MIN_FREE_DISK_PERCENT:
        update_status("disk_guard", "resource_blocked", free_disk_percent=free)
        raise RuntimeError(f"Disk guard: {free:.2f}% free < {MIN_FREE_DISK_PERCENT:.2f}%")


def run(command: list[str], name: str, *, env: dict[str, str] | None = None) -> None:
    disk_guard()
    update_status(name, command=command)
    log_dir = ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / f"{name}.stdout.log").open("a", encoding="utf-8") as stdout, (
        log_dir / f"{name}.stderr.log"
    ).open("a", encoding="utf-8") as stderr:
        completed = subprocess.run(
            command,
            cwd=REPO,
            env={**os.environ, **(env or {})},
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    if completed.returncode:
        raise RuntimeError(f"{name} failed with return code {completed.returncode}")


def training_payload() -> Path:
    path = ROOT / "training/control/payload.json"
    payload = read_json(BASE_PAYLOAD)
    payload.update(
        {
            "description": "Pure TBG Graph2Mat only, n_train=474, seed 0; DeepH excluded.",
            "dataset_root": str(DATASET),
            "output_root": str(ROOT / "training/n474"),
            "run_id": "tbg_pure_n474",
            "reuse_run_root": True,
            "resume_training_sweep": True,
        }
    )
    manual = payload["training_sweep"]["manual_runs"][0]
    manual.update({"id": "g2m_tbg_n474_seed0", "config_id": "g2m_tbg_n474_seed0"})
    payload["training_sweep"].update({"max_runs": 1, "manual_runs": [manual]})
    write_json(path, payload)
    return path


def train() -> Path:
    control = ROOT / "training/control"
    runner_status = read_json(control / "status.json").get("status", {})
    if runner_status.get("returncode") != 0:
        run(
            [
                str(PYTHON),
                str(SCRIPTS / "run_g2m_deeph_payload_once.py"),
                str(training_payload()),
                "--status-json",
                str(control / "status.json"),
                "--manifest-json",
                str(control / "runner_manifest.json"),
            ],
            "train_n474",
        )
    checkpoints = ROOT / "training/n474/tbg_pure_n474/sweep/graph2mat/n474/g2m_tbg_n474_seed0/graph2mat/training/lightning_logs/my_first_model/version_0/checkpoints"
    if not list(checkpoints.glob("best-*.ckpt")):
        raise RuntimeError(f"No Graph2Mat checkpoint found in {checkpoints}")
    return checkpoints


def evaluate_gate(checkpoint_dir: Path) -> Path | None:
    output = ROOT / "checkpoint_spectral_eval"
    report_path = output / "checkpoint_spectral_metrics_validation.json"
    checkpoints = sorted(checkpoint_dir.glob("best-*.ckpt")) + sorted(checkpoint_dir.glob("last-[0-9]*.ckpt"))
    if not report_path.exists():
        command = [
            str(PYTHON),
            str(SCRIPTS / "evaluate_checkpoint_spectral_metrics.py"),
            "--dataset",
            str(DATASET),
            "--output-root",
            str(output),
            "--basis-files",
            str(DATASET / "material_basis/*.ion.xml"),
        ]
        for checkpoint in checkpoints:
            command.extend(["--checkpoint", str(checkpoint)])
        run(command, "spectral_gate")
    report = read_json(report_path)
    scores = report.get("selection", {}).get("mean_eV", {})
    finite = {name: float(value) for name, value in scores.items() if value is not None}
    if not finite:
        raise RuntimeError("Spectral gate produced no finite frontier RMSE")
    winner = min(finite, key=finite.get)
    score = finite[winner]
    gate = {
        "status": "passed" if score <= GATE_EV else "failed",
        "metric": "validation K/Kprime four-frontier-state RMSE",
        "threshold_eV": GATE_EV,
        "score_eV": score,
        "checkpoint": winner,
        "all_scores_eV": finite,
        "report": str(report_path),
    }
    write_json(ROOT / "precision_gate.json", gate)
    if score > GATE_EV:
        update_status("precision_gate", "gate_failed", precision_gate=gate)
        publish_summary(None, gate)
        return None
    return next(path for path in checkpoints if path.name == winner)


def prepare_target() -> tuple[Path, Path]:
    sample = ROOT / "target/splits/test/0"
    sample.mkdir(parents=True, exist_ok=True)
    shutil.copy2(TARGET_FDF, sample / "RUN.fdf")
    shutil.copy2(REPO / "materials/graphene_common/basis/C.ion.xml", sample / "C.ion.xml")
    basis = ROOT / "target/material_basis"
    basis.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPO / "materials/graphene_common/basis/C.ion.xml", basis / "C.ion.xml")
    metadata = {
        "status": "geometry_only",
        "material_system": "pure_twisted_bilayer_graphene",
        "twist_angle_deg": 1.084549049,
        "materialized_twist_angle_deg": 1.0845490491576433,
        "geometry_inplane_lattice_ang": 2.48,
        "commensurate_cell_index": 2791,
        "layer1_supercell_matrix": [[61, 31], [30, 61]],
        "layer2_supercell_matrix": [[61, 30], [31, 61]],
        "commensurate_index": [31, 30],
        "num_atoms": 11164,
        "expected_orbitals": 44656,
        "reference_hamiltonian_available": False,
    }
    write_json(sample / "metadata.json", metadata)
    manifest = ROOT / "target/splits/test_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["sample_id", "method", "source_run", "source_sample_id", "structure_path", "hamiltonian_path", "run_out_path", "metadata_path", "valid", "split", "status", "sample_dir"]
        )
        writer.writerow(
            ["tbg_pure_0", "static_moire_geometry_only", str(TARGET_FDF), "tbg_pure_0", str(sample / "RUN.fdf"), "", "", str(sample / "metadata.json"), "True", "test", "geometry_only", str(sample)]
        )
    write_json(ROOT / "target/moire_geometry.json", metadata)
    return sample / "RUN.fdf", manifest


def overlap(target: Path) -> Path:
    output = ROOT / "overlap"
    existing = read_json(output / "overlap_manifest.json")
    if existing.get("status") != "completed":
        disk_guard()
        update_status("build_overlap")
        generated = generate_overlap(
            target,
            output,
            preset="bilayer_graphene_AA",
            siesta_command="/home/christian/bin/siesta",
            kgrid=3,
            overwrite=True,
        )
        if generated.get("status") != "completed":
            raise RuntimeError("Exact overlap validation failed")
    return output


def predict(checkpoint: Path, manifest: Path, overlap_root: Path) -> Path:
    raw = ROOT / "prediction/raw"
    predicted = raw / "predicted_hamiltonians/tbg_pure_0/ML_prediction.HSX"
    if not predicted.exists():
        run(
            [
                str(PYTHON),
                str(SCRIPTS / "predict_model_on_dataset.py"),
                "--checkpoint", str(checkpoint),
                "--train-method", "md",
                "--test-set", "tbg_pure_1p084549",
                "--test-manifest", str(manifest),
                "--output-dir", str(raw),
                "--basis-files", str(ROOT / "target/material_basis/*.ion.xml"),
                "--matrix-component-policy", "h_only",
                "--n-matrix-components", "1",
                "--accelerator", "gpu",
                "--precision", "bf16-mixed",
                "--mace-node-chunk-size", "512",
                "--mace-edge-chunk-size", "8192",
                "--graph2mat-edge-chunk-size", "8192",
                "--graph2mat-node-chunk-size", "512",
                "--loader-threads", "1",
                "--no-store-in-memory",
                "--torch-float32-matmul-precision", "high",
            ],
            "predict_target",
            env={"NVIDIA_TF32_OVERRIDE": "0", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
        )
    solver_input = ROOT / "prediction/solver_input"
    _link_exact_overlap_inputs(overlap_root, solver_input)
    h5 = solver_input / "hamiltonians_pred.h5"
    if not h5.exists():
        orb_indx = next(overlap_root.glob("*.ORB_INDX"))
        export_hamiltonian(predicted, orb_indx, h5)
    return solver_input


def solve_bands(solver_input: Path) -> dict:
    output = ROOT / "spectra/production"
    manifest = read_json(output / "solver_manifest.json")
    if manifest.get("status") != "completed":
        run(
            [
                str(PYTHON),
                str(SCRIPTS / "run_deeph_sparse_spectrum.py"),
                "--input-dir", str(solver_input),
                "--output-dir", str(output),
                "--job", "band",
                "--fermi-level", "0.0",
                "--num-bands", "16",
                "--points-per-segment", "11",
                "--backend", "gpu_cudss",
                "--gpu-memory-limit-gib", "28",
                "--project-mulliken",
                "--band-path", "k-gamma-m-k",
            ],
            "solve_bands",
            env={
                "OPENBLAS_NUM_THREADS": "8",
                "OMP_NUM_THREADS": "8",
                "MKL_NUM_THREADS": "8",
                "CUDSS_HOST_THREADS": "1",
            },
        )
        manifest = read_json(output / "solver_manifest.json")
    if manifest.get("status") != "completed" or not manifest.get("bands"):
        raise RuntimeError("Band solver did not produce a completed non-empty result")
    return manifest


DOS_TIERS = {
    # nombre: (subdirectorio, bandas, malla, ventana en meV, ensanchamiento en meV)
    "8x8": ("spectra/dos_8x8", "32", ("8", "8", "1"), "100", "1.0"),
    "16x16_wide": ("spectra/dos_16x16_wide", "55", ("16", "16", "1"), "500", "2.0"),
}


def solve_dos(solver_input: Path, tier: str = "8x8") -> dict:
    subdir, bands, kmesh, window, broadening = DOS_TIERS[tier]
    output = ROOT / subdir
    manifest = read_json(output / "solver_manifest.json")
    if manifest.get("status") != "completed":
        run(
            [
                str(PYTHON), str(SCRIPTS / "run_deeph_sparse_spectrum.py"),
                "--input-dir", str(solver_input), "--output-dir", str(output),
                "--job", "dos", "--fermi-level", "0.0", "--num-bands", bands,
                "--kmesh", *kmesh, "--backend", "gpu_cudss",
                "--gpu-memory-limit-gib", "28", "--project-mulliken",
                "--dos-broadening-mev", broadening, "--dos-energy-window-mev", window,
            ],
            "solve_dos",
            env={"OPENBLAS_NUM_THREADS": "8", "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8", "CUDSS_HOST_THREADS": "1"},
        )
        manifest = read_json(output / "solver_manifest.json")
    if manifest.get("status") != "completed" or not manifest.get("projected_dos"):
        raise RuntimeError("DOS solver did not produce a completed non-empty result")
    return manifest


def solve_fermi(solver_input: Path) -> dict:
    output = ROOT / "spectra/fermi_inertia_8x8"
    manifest = read_json(output / "solver_manifest.json")
    if not manifest.get("neutrality_reference", {}).get("chemical_potential_available"):
        run(
            [
                str(PYTHON), str(SCRIPTS / "run_deeph_sparse_spectrum.py"),
                "--input-dir", str(solver_input), "--output-dir", str(output),
                "--job", "dos", "--fermi-level", "0.0", "--num-bands", "32",
                "--kmesh", "8", "8", "1", "--backend", "gpu_cudss",
                "--gpu-memory-limit-gib", "28", "--neutral-electrons", "44656",
                "--spin-degeneracy", "2",
            ],
            "solve_fermi",
            env={"OPENBLAS_NUM_THREADS": "8", "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8", "CUDSS_HOST_THREADS": "1"},
        )
        manifest = read_json(output / "solver_manifest.json")
    if manifest.get("status") != "completed" or not manifest.get("neutrality_reference", {}).get("chemical_potential_available"):
        raise RuntimeError("Inertia solver did not produce a neutral chemical potential")
    write_json(ROOT / "neutrality_estimate.json", manifest["neutrality_reference"])
    return manifest


def solve_unfolding(solver_input: Path) -> dict:
    layers = {}
    for layer in ("bottom", "top"):
        output = ROOT / f"spectra/unfolded_{layer}"
        result = read_json(output / "solver_manifest.json")
        if result.get("status") != "completed":
            update_status(f"unfold_{layer}")
            disk_guard()
            result = run_unfolded_layer(
                root=ROOT,
                input_dir=solver_input,
                output_dir=output,
                layer=layer,
                num_bands=16,
                points_per_segment=16,
                backend="gpu_cudss",
                gpu_memory_limit_gib=28,
            )
        if result.get("status") != "completed":
            raise RuntimeError(f"{layer} unfolding failed: {result.get('reason')}")
        layers[layer] = result
    return {"status": "completed", "layers": layers}


# --------------------------------------------------------------------------- #
# E-F_001-S42: rigid-Gamma raw-derivative stage
#
# preregister_matbg_gamma_rigid_epc.py (S41) already froze the experiment --
# the five candidate directions, the delta sweep, the two allowed reference
# paths (graph2mat_jvp / graph2mat_frozen; no SIESTA reference exists at this
# cell's scale, S38) and the claim ladder rung this stage is authorised to
# produce: "derivative-level checks pass ... but GO-8a is still NO-GO-8a" ->
# graph2mat_derivative_validated_no_full_ks_coupling. Building PAO-covariant response and g needs
# a basis-response artifact (S_L/S_R); the only producer this repository has
# (epc_basis_response.py / certify_basis_response.py) requires a certified
# SIESTA FC.Save.dHS campaign object, and S38 measured that campaign
# intractable here (66984 displaced SCF solves). That gap is recorded on every
# artifact this stage writes, not silently skipped.
# --------------------------------------------------------------------------- #

EPC_ROOT = epc_prereg.DEFAULT_RESULT_ROOT
EPC_ARRAYS_DIR = EPC_ROOT / "arrays"
EPC_STATUS_PATH = EPC_ROOT / "artifact_status.json"
EPC_RESULT_STATUS = "candidate_rigid_tbg"

# Prefixes of the S41 protocol's own frozen ``blocking`` reasons this stage is
# pre-authorised to run under (its claim ladder names this exact set). Any
# other blocking reason means the protocol is stuck for something nobody
# accounted for, and this stage must stop rather than guess at it.
EPC_ALLOWED_BLOCKER_PREFIXES = (
    "GO-8a MATBG PhononProvider",
    "C14C basis response (global formalism gate)",
    "k_selection",
    "moire_hexagonal_convention_checked",
)


class EpcGammaRigidError(RuntimeError):
    """The S41 pre-registered protocol cannot be executed as specified."""


def epc_authorize(protocol: dict) -> dict:
    """S41's own gate, minus the one rung its frozen claim ladder pre-authorises.

    Mirrors ``run_graphene_gamma_epc_paths.authorize()``: a *quantitative* g
    needs GO-8a and C14C closed, and
    ``preregister_matbg_gamma_rigid_epc.require_frozen_protocol`` correctly
    refuses to run anything while ``ready_to_execute`` is false. The frozen
    claim ladder names this exact blocking set explicitly and authorises the
    two-path raw-derivative validation in exchange; this grants exactly that
    and nothing wider.
    """
    stored = protocol.get("frozen_content_sha256")
    if epc_prereg.prereg.freeze_hash(protocol) != stored:
        raise EpcGammaRigidError("the protocol has been edited since it was frozen")
    blocking = [str(reason) for reason in protocol.get("blocking") or []]
    unexpected = [
        reason
        for reason in blocking
        if not any(reason.startswith(prefix) for prefix in EPC_ALLOWED_BLOCKER_PREFIXES)
    ]
    if unexpected:
        raise EpcGammaRigidError(
            f"the protocol is blocked for reasons the claim ladder did not pre-authorise: "
            f"{unexpected}; resolve them, re-freeze and re-run"
        )
    ladder = protocol["claim_ladder"]
    rung = next(
        row
        for row in ladder
        if "derivative-level checks pass" in row["outcome"] and "NO-GO-8a" in row["outcome"]
    )
    return {
        "preregistration_sha256": stored,
        "protocol_id": protocol.get("protocol_id"),
        "blocking": blocking,
        "authorised_claim": rung["claim"],
        "authorised_outcome": rung["outcome"],
        "result_status": EPC_RESULT_STATUS,
    }


def epc_artifact_signature(*, checkpoint: Path, direction_hash: str, deltas: list[float], topology_hash: str) -> str:
    return input_signature_sha256(
        {
            "schema": "epc_gamma_rigid_matbg_raw_derivative_v1",
            "checkpoint_sha256": file_sha256(checkpoint),
            "direction_hash": direction_hash,
            "delta_sweep_ang": [round(float(value), 9) for value in deltas],
            "topology_hash": topology_hash,
            "code_sha256": file_sha256(Path(__file__)),
        }
    )


def _write_sparse_field_npz(path: Path, **fields) -> None:
    payload: dict[str, np.ndarray] = {}
    for field_name, field in fields.items():
        keys = list(field)
        payload[f"{field_name}__rows"] = np.array([key[0] for key in keys], dtype=np.int64)
        payload[f"{field_name}__cols"] = np.array([key[1] for key in keys], dtype=np.int64)
        payload[f"{field_name}__images"] = np.array([key[2] for key in keys], dtype=np.int64)
        payload[f"{field_name}__values"] = np.array([field[key] for key in keys], dtype=np.float64)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def epc_raw_derivative_for_direction(
    *,
    name: str,
    direction,
    model,
    batch,
    processor,
    change_of_basis: np.ndarray,
    deltas: list[float],
    checkpoint: Path,
    topology_hash: str,
    backend,
    status: dict,
) -> dict:
    """``D_H[v]`` of one pre-registered direction, JVP and frozen, reusing one model load."""
    artifact_id = f"raw_derivative__{name}"
    signature = epc_artifact_signature(
        checkpoint=checkpoint, direction_hash=direction.direction_hash, deltas=deltas, topology_hash=topology_hash
    )
    existing = status["artifacts"].get(artifact_id, {})
    if existing.get("state") == "completed" and existing.get("signature_sha256") == signature:
        return {**existing, "reused": True}

    status["artifacts"][artifact_id] = {
        "state": "running",
        "signature_sha256": signature,
        "result_status": EPC_RESULT_STATUS,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(EPC_STATUS_PATH, status)
    try:
        d_h_jvp, jvp_meta = epc_c14.directional_derivative_field(
            model, batch, direction, processor=processor, change_of_basis=change_of_basis, backend=backend
        )
        d_h_frozen_by_delta: dict[float, dict] = {}
        frozen_topology_hashes: list[str] = []
        for delta in deltas:
            d_h_frozen, hashes = epc_c14.frozen_derivative_field(
                model, batch, direction, delta, processor=processor, change_of_basis=change_of_basis
            )
            d_h_frozen_by_delta[delta] = d_h_frozen
            frozen_topology_hashes.extend(hashes)

        reference_delta = max(deltas)
        agreement = difference_norms(d_h_jvp, d_h_frozen_by_delta[reference_delta])
        frozen_scale = norms(
            np.array(list(d_h_frozen_by_delta[reference_delta].values()), dtype=np.float64)
        )
        # tau_backend rule (S41 TOLERANCES): the two numerical routes to the
        # same model's own derivative must agree well inside the frozen side's
        # own delta-plateau spread, not to an absolute eV number.
        plateau = [
            difference_norms(d_h_frozen_by_delta[deltas[i]], d_h_frozen_by_delta[deltas[j]])["frobenius"]
            for i in range(len(deltas))
            for j in range(i + 1, len(deltas))
        ]
        tau_num = max(plateau, default=0.0)
        jvp_equals_frozen = agreement["frobenius"] <= epc_prereg.BACKEND_NOISE_MARGIN * max(tau_num, 1e-12)

        npz_path = EPC_ARRAYS_DIR / f"{artifact_id}.npz"
        _write_sparse_field_npz(npz_path, d_h_jvp=d_h_jvp, d_h_frozen=d_h_frozen_by_delta[reference_delta])

        result = {
            "state": "completed",
            "signature_sha256": signature,
            "result_status": EPC_RESULT_STATUS,
            "artifact_kind": SYNTHETIC_TEST_DISPLACEMENT,
            "direction_hash": direction.direction_hash,
            "direction_kind": direction.kind,
            "reference_delta_ang": reference_delta,
            "delta_sweep_ang": list(deltas),
            "jvp_metadata": jvp_meta,
            "frozen_topology_hashes": frozen_topology_hashes,
            "checks": {
                "jvp_equals_frozen_within_backend_margin": bool(jvp_equals_frozen),
                "jvp_vs_frozen_frobenius": agreement,
                "frozen_delta_plateau_tau_num": tau_num,
                "frozen_reference_scale": frozen_scale,
                "topology_unchanged": "not_measured: real MATBG neighbour edge count has never "
                "been measured in this repository (S40 item 7); this check blocks a claim above "
                "graph2mat_derivative_validated_no_full_ks_coupling, it is not assumed",
            },
            "array_path": str(npz_path),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as exc:  # noqa: BLE001 - a failed artifact is a recorded state, not a crash
        result = {
            "state": "failed",
            "signature_sha256": signature,
            "result_status": EPC_RESULT_STATUS,
            "error": f"{type(exc).__name__}: {exc}"[:2000],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    status["artifacts"][artifact_id] = result
    write_json(EPC_STATUS_PATH, status)
    return result


def epc_gamma_rigid(checkpoint: Path, target_fdf: Path, basis_dir: Path, requested_backend: str = "cuda") -> dict:
    """E-F_001-S42: raw ``D_H`` responses of the five S41 pre-registered rigid-Gamma directions.

    Reuses the checkpoint, the equilibrium geometry and the overlap-stage
    positions this campaign already produced -- one model load, one CUDA
    preflight, no SIESTA call, no eigensolve, nothing recomputed per
    direction. ``PAO-covariant response`` and ``g`` are not computable today (no
    basis-response provider exists at this cell's scale); every direction's
    ``basis_response``/``eigenspace``/``pao_covariant_response``/``g`` are recorded
    ``status: "not_attempted"`` with the reason, rather than fabricated or
    silently dropped. Idempotent: an artifact whose signature (checkpoint,
    direction, delta sweep, neighbour topology, this file's own hash) already
    matches a ``completed`` entry in ``artifact_status.json`` is reused as is.
    """
    EPC_ROOT.mkdir(parents=True, exist_ok=True)
    EPC_ARRAYS_DIR.mkdir(parents=True, exist_ok=True)

    protocol = epc_prereg.load_protocol()
    gate = epc_authorize(protocol)

    status = read_json(EPC_STATUS_PATH)
    status.setdefault("artifacts", {})

    positions = epc_prereg.read_xyz_positions_ang(epc_prereg.DEFAULT_POSITIONS_XYZ)
    directions = epc_prereg.build_candidate_directions(positions)
    frozen_sectors = protocol["phonon"]["candidate_sectors"]
    for direction_name, direction in directions.items():
        if direction.direction_hash != frozen_sectors[direction_name]["direction_hash"]:
            raise EpcGammaRigidError(
                f"{direction_name}: rebuilt direction hash != the S41 frozen protocol's hash; "
                "the geometry or fd_perturbation_space has drifted since the protocol was frozen"
            )

    model, batch, processor, _provenance = epc_c14.load_model_and_batch(
        checkpoint, target_fdf, basis_dir=basis_dir
    )
    model, batch, backend = epc_g2m_deriv.resolve_jvp_backend(
        model, batch, requested_backend, output_keys=epc_c14.OUTPUT_KEYS
    )
    topology = epc_g2m_deriv.batch_topology_hash(batch)
    change_of_basis = np.asarray(processor.basis_table.change_of_basis, dtype=np.float64)
    deltas = [float(value) for value in protocol["perturbation"]["delta_sweep_ang"]]

    direction_results = {
        direction_name: epc_raw_derivative_for_direction(
            name=direction_name,
            direction=direction,
            model=model,
            batch=batch,
            processor=processor,
            change_of_basis=change_of_basis,
            deltas=deltas,
            checkpoint=checkpoint,
            topology_hash=topology["topology_hash"],
            backend=backend,
            status=status,
        )
        for direction_name, direction in directions.items()
    }
    write_json(EPC_STATUS_PATH, status)

    all_completed = all(row["state"] == "completed" for row in direction_results.values())
    checks_passed = all_completed and all(
        row["checks"]["jvp_equals_frozen_within_backend_margin"] for row in direction_results.values()
    )
    claim = gate["authorised_claim"] if checks_passed else "NO_GO"

    summary = {
        "schema": "epc_gamma_rigid_matbg_v1",
        "ticket": "E-F_001-S42",
        "gate": "S41 claim ladder rung: derivative-level validation while GO-8a is NO-GO-8a",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "result_status": EPC_RESULT_STATUS,
        "artifact_kind": SYNTHETIC_TEST_DISPLACEMENT,
        "preregistration": gate,
        "backend": backend.to_metadata(),
        "topology": topology,
        "checkpoint_sha256": file_sha256(checkpoint),
        "formalism_id": protocol["references"]["formalism_id"],
        "directions": direction_results,
        "blocked": {
            "basis_response": {
                "status": "not_attempted",
                "reason": "no SIESTA basis-response provider is tractable at this cell's scale "
                "(S38: an exact FC.Save.dHS campaign needs 66984 displaced SCF solves); the "
                f"{protocol['references']['basis_response']['representation']} representation "
                f"formalism_id {protocol['references']['formalism_id']} requires has no producer "
                "here yet",
            },
            "eigenspace": {
                "status": "not_attempted",
                "reason": "no electronic k has passed k_selection (protocol status: "
                "pending_measurement); persisting an eigenspace before a k is selected would "
                "pre-empt that measurement instead of reusing it",
            },
            "pao_covariant_response": {"status": "not_attempted", "reason": "blocked by basis_response"},
            "g": {"status": "not_attempted", "reason": "blocked by pao_covariant_response"},
            "phonon": {
                "status": "not_attempted",
                "reason": "GO-8a is NO-GO-8a (certify_matbg_phonon_provider); every direction "
                "here stays a synthetic_test_displacement candidate pattern, never a phonon "
                "eigenvector (roadmap B7)",
            },
        },
        "claim": claim,
        "forbidden_publication_labels": [
            "quantitative_relaxed_matbg_prediction",
            "g_mn_nu",
            "phonon_eigenmode",
        ],
    }
    write_json(EPC_ROOT / "epc_gamma_rigid_summary.json", summary)
    return summary


def _recenter(rows: list[dict], fermi_level_eV: float) -> list[dict]:
    return [
        {**row, "energy_aligned_eV": float(row["energy_eV"]) - fermi_level_eV}
        for row in rows
    ]


def publish_summary(  # noqa: PLR0913
    solver: dict | None,
    gate: dict,
    training_size: int = 474,
    dos: dict | None = None,
    fermi: dict | None = None,
    unfolding: dict | None = None,
    dos_tier: str = "8x8",
) -> None:
    spectra = []
    if solver:
        solver_view = copy.deepcopy(solver)
        dos_view = copy.deepcopy(dos) if dos else None
        neutrality = (fermi or {}).get("neutrality_reference")
        if neutrality:
            energy = float(neutrality["energy_eV"])
            solver_view["bands"] = _recenter(solver_view.get("bands", []), energy)
            for representation in solver_view.get("band_representations", {}).values():
                if "bands" in representation:
                    representation["bands"] = _recenter(representation["bands"], energy)
            solver_view.update({"fermi_level_eV": energy, "neutrality_reference": neutrality})
            if dos_view:
                for key in ("low_energy_dos", "projected_dos"):
                    dos_view[key] = _recenter(dos_view.get(key, []), energy)
                dos_view["dos_observables"] = projected_dos_observables(dos_view["projected_dos"])
        spectra.append(
            {
                **solver_view,
                **({key: dos_view[key] for key in ("low_energy_dos", "projected_dos", "dos_observables", "dos_projection") if key in dos_view} if dos_view else {}),
                **({"dos_projection": dos_view["projection"]} if dos_view and "projection" in dos_view else {}),
                "model": "graph2mat",
                "training_size": training_size,
                "requested_training_size": training_size,
                "seed": 0,
                "material_system": "pure_tbg",
                "twist_angle_deg": 1.084549049,
                "visible_band_tier": "pure_tbg_production",
                "manifest_path": str(ROOT / "spectra/production/solver_manifest.json"),
                "dos_manifest_path": str(ROOT / DOS_TIERS[dos_tier][0] / "solver_manifest.json") if dos else None,
                "visible_dos_tier": dos_tier if dos else None,
                "fermi_manifest_path": str(ROOT / "spectra/fermi_inertia_8x8/solver_manifest.json") if fermi else None,
                "unfolding": unfolding,
                "scientific_status": "prediction_only_validation_gate_passed",
            }
        )
    write_json(
        ROOT / "summary/spectral_results.json",
        {
            "campaign_kind": "pure_tbg_graph2mat_spectral_prediction",
            "scientific_status": "completed" if solver else "stopped_by_precision_gate",
            "target_contract": "geometry_plus_exact_overlap_no_reference_hamiltonian",
            "target_reference_metrics_available": False,
            "deep_h_excluded": True,
            "precision_gate": gate,
            "neutrality_reference": (fermi or {}).get("neutrality_reference"),
            "spectra": spectra,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--precision-gate", type=Path)
    parser.add_argument("--training-size", type=int, default=474)
    parser.add_argument("--dos-tier", choices=tuple(DOS_TIERS), default="8x8",
                        help="Which DOS calculation the summary publishes to the UI.")
    parser.add_argument(
        "--epc-gamma-rigid", action="store_true",
        help="E-F_001-S42: after the spectral pipeline, run the S41 pre-registered rigid-Gamma "
        "raw-derivative stage (reuses this campaign's own checkpoint/H/S; no SIESTA, no eigensolve).",
    )
    parser.add_argument("--epc-backend", default="cuda", choices=("cpu", "cuda"))
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    try:
        update_status("starting")
        if args.checkpoint:
            checkpoint = args.checkpoint.resolve()
            gate = read_json(args.precision_gate.resolve()) if args.precision_gate else {}
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            if gate.get("status") != "passed" or float(gate.get("score_eV", 1.0)) > GATE_EV:
                raise RuntimeError("External checkpoint lacks a passing frozen precision gate")
            write_json(ROOT / "precision_gate.json", gate)
            update_status(
                "external_gate_accepted",
                checkpoint=str(checkpoint),
                precision_gate=gate,
            )
        else:
            checkpoints = train()
            checkpoint = evaluate_gate(checkpoints)
            if checkpoint is None:
                return 0
            gate = read_json(ROOT / "precision_gate.json")
        target, manifest = prepare_target()
        overlap_root = overlap(target)
        solver_input = predict(checkpoint, manifest, overlap_root)
        solver = solve_bands(solver_input)
        dos = solve_dos(solver_input, args.dos_tier)
        fermi = solve_fermi(solver_input)
        unfolding = solve_unfolding(solver_input)
        publish_summary(solver, gate, training_size=args.training_size, dos=dos, fermi=fermi, unfolding=unfolding, dos_tier=args.dos_tier)
        epc_claim = None
        if args.epc_gamma_rigid:
            update_status("epc_gamma_rigid")
            epc_summary = epc_gamma_rigid(checkpoint, target, ROOT / "target/material_basis", args.epc_backend)
            epc_claim = epc_summary["claim"]
        update_status("completed", "completed", precision_gate=gate, ui_ready=True, epc_gamma_rigid_claim=epc_claim)
        return 0
    except Exception as exc:
        update_status("failed", "failed", error=str(exc))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
