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
        update_status("completed", "completed", precision_gate=gate, ui_ready=True)
        return 0
    except Exception as exc:
        update_status("failed", "failed", error=str(exc))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
