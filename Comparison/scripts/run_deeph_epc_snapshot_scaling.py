#!/usr/bin/env python3
"""Evaluate existing DeepH W90/5x5 checkpoints on frozen graphene Gamma-E2g pairs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from evaluate_hamiltonian_derivative_metrics import load_hamiltonian_matrix
import epc_basis_response as ebr
from hamiltonian_derivative_stencil import derivative_sparse_metrics, finite_difference_derivative_pair
from run_deeph_sparse_spectrum import load_persisted_eigenspace
import compute_epc_matrix_elements as epc
from run_hamiltonian_derivative_predictions import run_derivative_predictions
from screen_epc_snapshot_scaling import DIRECTIONS, ROOT, SIZES, write_ui_payload

REFERENCE = ROOT / "Comparison/results/epc/siesta_reference/graphene_e2g"
OUTPUT = ROOT / "Comparison/results/epc/deeph_snapshot_scaling_20260909"
DEEPH_CLI = Path("/home/christian/repositorios/DeepH-pack/.venv/bin/deeph-inference")
DELTA = 0.02


def model_dir(family: str, size: int) -> Path:
    if family == "w90":
        root = ROOT / "Comparison/results/ml_vs_siesta_cross_structure_vacancy/seed_1"
        pattern = (
            f"graphene__graphene_w90_scale_iid{size}__to__graphene_5x5_vacancy/"
            "training/**/deeph/train/best_state_dict.pkl"
        )
    else:
        root = ROOT / "Comparison/results/graphene_5x5_snapshot_scaling_complete_paper_ready"
        pattern = f"**/deeph/graphene_5x5_scale_iid{size}/**/deeph/train/best_state_dict.pkl"
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"expected one DeepH {family}-{size} checkpoint, found {len(matches)}")
    return matches[0].parent


def _link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        destination.symlink_to(os.path.relpath(source, destination.parent), target_is_directory=source.is_dir())


def stage_frozen_e2g(root: Path) -> dict[str, dict[str, Path]]:
    manifest = json.loads((REFERENCE / "epc_siesta_reference_manifest.json").read_text())
    selected: dict[str, dict[str, Path]] = {direction: {} for direction in DIRECTIONS}
    for row in manifest["rows"]:
        direction = row.get("direction_name")
        if direction not in selected or not np.isclose(float(row.get("delta_ang", -1)), DELTA):
            continue
        sign = str(row["sign_label"])
        source = Path(row["run_dir"])
        sample_id = f"{direction}_{sign}"
        _link(source, root / "structures" / sample_id)
        _link(source, root / "siesta_hamiltonians" / sample_id)
        selected[direction][sign] = source
    missing = {key: sorted({"plus", "minus"} - set(value)) for key, value in selected.items() if len(value) != 2}
    if missing:
        raise RuntimeError(f"missing frozen E2g references: {missing}")
    return selected


def _complex_matrix(payload: dict[str, object]) -> np.ndarray:
    return np.asarray(payload["values_real"]) + 1j * np.asarray(payload["values_imag"])


def diagnostic_epc_g(family: str, size: int, run: Path) -> list[dict[str, object]]:
    """Reuse Graph2Mat's certified PAO response, replacing only its electronic dH/dR."""
    graph_run = ROOT / "Comparison/results/epc/snapshot_scaling_20260908/runs" / f"{family}_{size}"
    base = json.loads((graph_run / "graphene_gamma_epc_g_blocks.json").read_text())
    eigenspace_dir = ROOT / "Comparison/results/epc/eigenspaces/graphene"
    manifest = json.loads((eigenspace_dir / "eigenspaces_manifest.json").read_text())
    eigenspaces = {
        row["label"]: epc.Eigenspace.from_persisted(
            load_persisted_eigenspace(eigenspace_dir, index), label=row["label"]
        )
        for index, row in enumerate(manifest["eigenspaces"])
    }
    corrections = {}
    for direction in DIRECTIONS:
        plus = load_hamiltonian_matrix(run / "predicted_hamiltonians" / f"{direction}_plus" / "ML_prediction.HSX")
        minus = load_hamiltonian_matrix(run / "predicted_hamiltonians" / f"{direction}_minus" / "ML_prediction.HSX")
        deeph = ((plus - minus) / (2 * DELTA)).toarray()
        source = np.load(graph_run / "arrays" / f"raw_derivative__graph2mat_frozen__{direction}__d0p02.npz")
        isc_off, graph2mat = source["isc_off"], source["D_H"]
        no_u = graph2mat.shape[1]
        if deeph.shape != (no_u, len(isc_off) * no_u):
            raise RuntimeError(f"unexpected DeepH sparse layout {deeph.shape} for {direction}")
        deeph_blocks = deeph.reshape(no_u, len(isc_off), no_u).transpose(1, 0, 2)
        corrections[direction] = (deeph_blocks - graph2mat, isc_off)

    siesta = {
        (row["mode_label"], row["k_label"]): _complex_matrix(row["g"])
        for row in base["reports"] if row["path"] == "siesta_reference"
    }
    rows = []
    for report in base["reports"]:
        if report["path"] != "graph2mat_frozen":
            continue
        eigenspace = eigenspaces[report["k_label"]]
        delta_orbital = np.zeros((eigenspace.no_u, eigenspace.no_u), dtype=np.complex128)
        coefficients = np.asarray(report["expansion"]["coefficients_real"])
        for coefficient, direction in zip(coefficients, report["expansion"]["direction_names"]):
            blocks, isc_off = corrections[direction]
            delta_orbital += coefficient * ebr.bloch_sum(blocks, isc_off, eigenspace.k)
        values = _complex_matrix(report["g"]) + eigenspace.C.conj().T @ delta_orbital @ eigenspace.C
        reference = siesta[(report["mode_label"], report["k_label"])]
        rows.append({
            "mode_label": report["mode_label"], "k_label": report["k_label"],
            "g_frobenius_eV": float(np.linalg.norm(values)),
            "g_relative_frobenius_vs_siesta": float(np.linalg.norm(values - reference) / np.linalg.norm(reference)),
            "g_values_real": values.real.tolist(), "g_values_imag": values.imag.tolist(),
            "scientific_status": "diagnostic_only",
        })
    return rows


def evaluate(family: str, size: int) -> dict[str, object]:
    run = OUTPUT / "runs" / f"{family}_{size}"
    references = stage_frozen_e2g(run)
    predictions = run / "predicted_hamiltonians"
    previous_force_cpu = os.environ.get("DEEPH_FORCE_CPU")
    os.environ["DEEPH_FORCE_CPU"] = "1"
    try:
        manifest = run_derivative_predictions(
            stencil_root=run,
            model="deeph",
            output_root=predictions,
            model_dir=model_dir(family, size),
            diagnostic_only=True,
            deeph_command=str(DEEPH_CLI),
            python_executable=str(DEEPH_CLI.with_name("python")),
        )
    finally:
        if previous_force_cpu is None:
            os.environ.pop("DEEPH_FORCE_CPU", None)
        else:
            os.environ["DEEPH_FORCE_CPU"] = previous_force_cpu
    failed = [row for row in manifest["rows"] if row["status"] == "error"]
    if failed:
        raise RuntimeError(f"DeepH {family}-{size} failed for {len(failed)} E2g structures")
    rows = []
    for direction, pair in references.items():
        reference_plus = load_hamiltonian_matrix(next(pair["plus"].glob("*.TSHS")))
        reference_minus = load_hamiltonian_matrix(next(pair["minus"].glob("*.TSHS")))
        predicted_plus = load_hamiltonian_matrix(predictions / f"{direction}_plus" / "ML_prediction.HSX")
        predicted_minus = load_hamiltonian_matrix(predictions / f"{direction}_minus" / "ML_prediction.HSX")
        comparison = finite_difference_derivative_pair(
            method="central", delta_ang=DELTA,
            reference_plus=reference_plus, reference_minus=reference_minus,
            predicted_plus=predicted_plus, predicted_minus=predicted_minus,
            predicted_source="deeph",
        )
        rows.append({
            "direction": direction,
            **derivative_sparse_metrics(
                comparison.reference.matrix,
                comparison.predicted.matrix,
                sample=direction,
                source_model="deeph",
            ),
        })
    metric_keys = (
        "dh_mae_union_eV_per_Ang", "dh_rmse_union_eV_per_Ang",
        "dh_relative_frobenius_ref", "dh_support_f1", "dh_false_zero_rate",
        "dh_false_nonzero_rate", "dh_relative_frobenius_union_robust",
        "dh_relative_l1_union_robust", "dh_pearson_union",
        "dh_residual_mean_union_eV_per_Ang", "dh_residual_std_union_eV_per_Ang",
        "dh_residual_median_union_eV_per_Ang", "dh_residual_bias_over_mae_union",
        "dh_residual_abs_p90_union_eV_per_Ang", "dh_residual_abs_p95_union_eV_per_Ang",
        "dh_residual_abs_p99_union_eV_per_Ang",
    )
    metrics = {key: float(np.mean([row[key] for row in rows])) for key in metric_keys}
    adapter_records = [
        json.loads(path.read_text())
        for path in sorted(predictions.glob("*/deeph_adapter_result.json"))
    ]
    adapter_statuses = sorted({str(row.get("adapter_equivalence_status") or "missing") for row in adapter_records})
    equivalence_statuses = sorted({str(row.get("equivalence_status") or "missing") for row in adapter_records})
    epc_g_rows = diagnostic_epc_g(family, size, run)
    result = {
        "family": family, "snapshots": size, "model": "deeph", **metrics,
        "directions": rows, "model_dir": str(model_dir(family, size)),
        "adapter_statuses": adapter_statuses, "equivalence_statuses": equivalence_statuses,
        "scientific_status": "diagnostic_only",
        "epc_g_status": "diagnostic_only_until_raw_global_orbital_equivalence_is_proven",
        "g_relative_frobenius_K": float(np.mean([row["g_relative_frobenius_vs_siesta"] for row in epc_g_rows if row["k_label"] == "K"])),
        "g_relative_frobenius_k_generic_1": float(np.mean([row["g_relative_frobenius_vs_siesta"] for row in epc_g_rows if row["k_label"] == "k_generic_1"])),
        "epc_g": epc_g_rows,
    }
    (run / "deeph_e2g_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("w90", "5x5"), action="append")
    parser.add_argument("--size", type=int, action="append")
    args = parser.parse_args()
    families = args.family or ["w90", "5x5"]
    sizes = args.size or list(SIZES)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    results = []
    for family in families:
        for size in sizes:
            result = evaluate(family, size)
            results.append(result)
            print(f"{family}-{size}: MAE={result['dh_mae_union_eV_per_Ang']:.6g}", flush=True)
    (OUTPUT / "deeph_epc_snapshot_scaling_metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    graph2mat_path = ROOT / "Comparison/results/epc/snapshot_scaling_20260908/epc_snapshot_scaling_metrics.json"
    write_ui_payload(json.loads(graph2mat_path.read_text()) + results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
