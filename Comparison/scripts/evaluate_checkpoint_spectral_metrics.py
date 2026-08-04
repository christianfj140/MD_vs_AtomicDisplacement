#!/usr/bin/env python3
"""Score Graph2Mat checkpoints by validation-split frontier-state error.

Selection uses the validation split only. The test split may be reported for
documentation but never drives checkpoint choice.

Two things this deliberately does NOT do:

* It does not score "the N eigenvalues nearest zero" for large N. These cells
  carry 42 orbitals, so N=32 spans roughly -20..+46 eV and is dominated by the
  polarization orbitals that live in the near-null space of S. That measures
  global spectral sensitivity, not the bands near E_F.
* It does not average over k-points that have no low-energy states. In these
  primitive cells only K and K' hold reference states inside +/-0.5 eV; Gamma
  and M hold none, so including them would let far-from-E_F error decide the
  checkpoint. Gamma and M are still reported, just excluded from selection.

Alongside the eigenvalue error it measures the perturbation projected onto the
reference low-energy subspace, C_low^dagger dH C_low, with S-orthonormal C. The
frontier states are near-degenerate, so the first-order shifts are that block's
EIGENVALUES, not its diagonal — the diagonal depends on which basis the solver
picked inside a degenerate subspace and is not reportable. Those eigenvalues are
the quantity that has to be small for minibands of a few meV to survive.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import scipy.linalg

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_hamiltonian_metrics import (  # noqa: E402
    kpoint_hamiltonian_matrix,
    kpoint_overlap_matrix,
    complex_generalized_eigenvalues,
    symmetrized_hermitian_dense,
)

DEFAULT_DATASET = REPO_ROOT / "Comparison/datasets/graphene_hbn_bilayer_md_nested/n480"
DEFAULT_CHECKPOINT_DIR = (
    REPO_ROOT / "Comparison/results/graphene_hbn_magic_angle_spectral/training/n480/n480"
    "/sweep/graph2mat/n480/g2m_n480_seed0/graph2mat/training/lightning_logs"
    "/my_first_model/version_0/checkpoints"
)
# K and K' are the only points holding states near E_F in these cells; Gamma and
# M are measured for context and excluded from selection by SELECTION_KPOINTS.
KPOINTS = {
    "Gamma": [0.0, 0.0, 0.0],
    "K": [1 / 3, 1 / 3, 0.0],
    "Kprime": [2 / 3, 2 / 3, 0.0],
    "M": [0.5, 0.0, 0.0],
}
SELECTION_KPOINTS = ("K", "Kprime")
FRONTIER_STATES = 4
WINDOW_eV = 0.5
MANIFEST_FIELDS = [
    "sample_id", "method", "source_run", "source_sample_id", "structure_path",
    "hamiltonian_path", "run_out_path", "metadata_path", "valid", "split", "status", "sample_dir",
]


def split_rows(dataset: Path, split: str, limit: int | None) -> list[dict[str, Any]]:
    manifest = json.loads((dataset / "frozen_split_manifest.json").read_text(encoding="utf-8"))
    rows = [row for row in manifest.get("rows", []) if row.get("split") == split]
    if not rows:
        raise RuntimeError(f"No rows with split={split!r} in {dataset}")
    return rows[:limit] if limit else rows


def write_manifest(rows: list[dict[str, Any]], path: Path, split: str) -> None:
    """Record the true split. predict_model_on_dataset does not read this column."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(MANIFEST_FIELDS)
        for row in rows:
            writer.writerow([
                row["sample_id"], "md", "", row["source_sample_id"], row["structure_path"],
                "", "", row["metadata_path"], "True", split, "ok", row["sample_dir"],
            ])


def run_prediction(checkpoint: Path, manifest: Path, output_dir: Path, basis_glob: str) -> None:
    command = [
        str(REPO_ROOT / ".venv/bin/python"),
        str(SCRIPT_DIR / "predict_model_on_dataset.py"),
        "--checkpoint", str(checkpoint),
        "--train-method", "md",
        "--test-set", "checkpoint_spectral_eval",
        "--test-manifest", str(manifest),
        "--output-dir", str(output_dir),
        "--basis-files", basis_glob,
        "--matrix-component-policy", "h_only",
        "--n-matrix-components", "1",
        "--accelerator", "cpu",
    ]
    completed = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Prediction failed for {checkpoint.name}:\n{completed.stderr[-2000:]}")


def inverse_sqrt(overlap: np.ndarray) -> tuple[np.ndarray, float, float]:
    """S^-1/2 by eigendecomposition, plus the smallest eigenvalue and condition number."""
    values, vectors = np.linalg.eigh(symmetrized_hermitian_dense(overlap))
    if values.min() <= 0.0:
        raise RuntimeError(f"Overlap is not positive definite: min eigenvalue {values.min():.3e}")
    return (vectors * values**-0.5) @ vectors.conj().T, float(values.min()), float(values.max() / values.min())


def subspace_metrics(
    reference: np.ndarray,
    predicted: np.ndarray,
    vectors: np.ndarray,
    delta_h: np.ndarray,
    selection: np.ndarray,
) -> dict[str, Any]:
    """Eigenvalue error and projected perturbation over one set of states."""
    if selection.size == 0:
        return {"n_states": 0}
    selection = np.sort(selection)
    delta = predicted[selection] - reference[selection]
    shift = float(np.mean(delta))
    aligned = delta - shift
    # C is S-orthonormal, so C^dag dH C is the perturbation restricted to the
    # subspace. Frontier states are near-degenerate, and there the first-order
    # shifts are this block's EIGENVALUES: its diagonal depends on which basis
    # the solver happened to pick inside a degenerate subspace, so it is not a
    # reportable quantity. For a Hermitian block max|eigenvalue| is its spectral
    # norm, already reported below.
    projected = vectors[:, selection].conj().T @ delta_h @ vectors[:, selection]
    first_order = np.linalg.eigvalsh((projected + projected.conj().T) / 2.0)
    return {
        "n_states": int(selection.size),
        "reference_min_eV": float(reference[selection].min()),
        "reference_max_eV": float(reference[selection].max()),
        "rmse_eV": float(np.sqrt(np.mean(delta**2))),
        "mae_eV": float(np.mean(np.abs(delta))),
        "max_abs_error_eV": float(np.max(np.abs(delta))),
        "median_abs_error_eV": float(np.median(np.abs(delta))),
        "global_shift_eV": -shift,
        "aligned_rmse_eV": float(np.sqrt(np.mean(aligned**2))),
        "aligned_median_abs_error_eV": float(np.median(np.abs(aligned))),
        "projected_frobenius_eV": float(np.linalg.norm(projected, ord="fro")),
        "projected_spectral_eV": float(np.linalg.norm(projected, ord=2)),
        "projected_first_order_shifts_eV": [float(value) for value in first_order],
        "projected_max_abs_first_order_shift_eV": float(np.max(np.abs(first_order))),
    }


def score_sample(prediction: Path, sample_dir: Path) -> list[dict[str, Any]]:
    import sisl

    predicted_obj = sisl.get_sile(str(prediction)).read_hamiltonian()
    reference_obj = sisl.get_sile(str(next(sample_dir.glob("*.TSHS")))).read_hamiltonian()
    rows = []
    for label, kpoint in KPOINTS.items():
        h_pred = kpoint_hamiltonian_matrix(predicted_obj, kpoint)
        h_ref = kpoint_hamiltonian_matrix(reference_obj, kpoint)
        s_ref = kpoint_overlap_matrix(reference_obj, kpoint)
        if s_ref is None:
            raise RuntimeError(f"{sample_dir}: reference has no overlap; cannot score spectrally.")
        # The model predicts H only, so both spectra use the exact reference overlap.
        eig_ref, vec_ref = scipy.linalg.eigh(
            symmetrized_hermitian_dense(h_ref), symmetrized_hermitian_dense(s_ref),
            check_finite=False,
        )
        eig_pred = np.sort(complex_generalized_eigenvalues(h_pred, s_ref))
        delta_h = symmetrized_hermitian_dense(h_pred) - symmetrized_hermitian_dense(h_ref)
        s_inv_sqrt, s_min, s_cond = inverse_sqrt(s_ref)
        delta_h_tilde = s_inv_sqrt @ delta_h @ s_inv_sqrt
        row: dict[str, Any] = {
            "kpoint": label,
            "delta_h_frobenius_eV": float(np.linalg.norm(delta_h, ord="fro")),
            "delta_h_spectral_eV": float(np.linalg.norm(delta_h, ord=2)),
            "delta_h_tilde_frobenius_eV": float(np.linalg.norm(delta_h_tilde, ord="fro")),
            "delta_h_tilde_spectral_eV": float(np.linalg.norm(delta_h_tilde, ord=2)),
            "overlap_min_eigenvalue": s_min,
            "overlap_condition_number": s_cond,
            "states_within_window": int((np.abs(eig_ref) <= WINDOW_eV).sum()),
        }
        selections = {
            "frontier": np.argsort(np.abs(eig_ref))[:FRONTIER_STATES],
            "window": np.flatnonzero(np.abs(eig_ref) <= WINDOW_eV),
            "all": np.arange(eig_ref.size),
        }
        for name, selection in selections.items():
            for key, value in subspace_metrics(eig_ref, eig_pred, vec_ref, delta_h, selection).items():
                row[f"{name}_{key}"] = value
        rows.append(row)
    return rows


def evaluate(checkpoints: list[Path], dataset: Path, split: str, limit: int | None,
             output_root: Path, basis_glob: str) -> dict[str, Any]:
    rows = split_rows(dataset, split, limit)
    report: dict[str, Any] = {
        "split_used_for_selection": split,
        "sample_count": len(rows),
        "kpoints_fractional": KPOINTS,
        "selection_kpoints": list(SELECTION_KPOINTS),
        "frontier_states": FRONTIER_STATES,
        "window_eV": WINDOW_eV,
        "checkpoints": {},
    }
    with tempfile.TemporaryDirectory() as tmp:
        manifest = Path(tmp) / "manifest.csv"
        write_manifest(rows, manifest, split)
        for checkpoint in checkpoints:
            predictions = output_root / f"pred_{checkpoint.stem}"
            run_prediction(checkpoint, manifest, predictions, basis_glob)
            per_sample = []
            for row in rows:
                path = predictions / "predicted_hamiltonians" / row["sample_id"] / "ML_prediction.HSX"
                for scored in score_sample(path, Path(row["sample_dir"])):
                    per_sample.append({"sample_id": row["sample_id"], **scored})
            report["checkpoints"][checkpoint.name] = {
                "path": str(checkpoint),
                "per_sample": per_sample,
                "per_kpoint": {
                    label: {
                        key: float(np.mean([r[key] for r in items]))
                        for key in items[0] if isinstance(items[0][key], (int, float))
                    }
                    for label in KPOINTS
                    if (items := [r for r in per_sample if r["kpoint"] == label])
                },
                "explains_frontier_error": frontier_correlations(per_sample),
                "single_offset": single_offset_metrics(per_sample),
            }
    return report


def frontier_correlations(per_sample: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Which norm tracks the frontier-state error, over the k-points that have one."""
    usable = [r for r in per_sample if r["kpoint"] in SELECTION_KPOINTS and r["frontier_n_states"]]
    if len(usable) < 3:
        return {}
    norms = ("delta_h_spectral_eV", "delta_h_tilde_spectral_eV",
             "delta_h_frobenius_eV", "delta_h_tilde_frobenius_eV",
             "frontier_projected_spectral_eV")
    return {
        target: {
            norm: float(np.corrcoef([r[norm] for r in usable], [r[target] for r in usable])[0, 1])
            for norm in norms
        }
        for target in ("frontier_rmse_eV", "all_rmse_eV")
    }


def single_offset_metrics(per_sample: list[dict[str, Any]]) -> dict[str, Any]:
    """One frozen offset for the whole split, not one per sample.

    ``frontier_aligned_rmse_eV`` fits a separate shift to every sample using the
    reference eigenvalues, so it is an optimistic bound that cannot be reproduced
    on a target with no reference (the moire). The transferable protocol is to fit
    a single offset here and apply it unchanged elsewhere, which is what
    ``offset_eV`` is for: freeze it from the validation run and pass it to the test
    run via --calibration-report.
    """
    rows = [r for r in per_sample if r["kpoint"] in SELECTION_KPOINTS and r["frontier_n_states"]]
    if not rows:
        return {}
    # global_shift = -mean(delta), so the offset that centres the errors is its mean.
    offset = float(np.mean([-r["frontier_global_shift_eV"] for r in rows]))
    return {
        "offset_eV": offset,
        "rmse_raw_eV": rmse_with_offset(rows, 0.0),
        "rmse_with_this_split_offset_eV": rmse_with_offset(rows, offset),
        "per_sample_aligned_rmse_eV": float(np.sqrt(np.mean(
            [r["frontier_aligned_rmse_eV"] ** 2 for r in rows]
        ))),
        "per_sample_alignment_is_not_transferable": True,
    }


def frozen_offsets(calibration_report: Path | None) -> dict[str, float]:
    """Per-checkpoint offsets frozen from an earlier split's report."""
    if calibration_report is None:
        return {}
    payload = json.loads(calibration_report.read_text(encoding="utf-8"))
    offsets = {}
    for name, entry in (payload.get("checkpoints") or {}).items():
        offset = (entry.get("single_offset") or {}).get("offset_eV")
        if offset is not None:
            offsets[name] = float(offset)
    if not offsets:
        raise RuntimeError(f"{calibration_report}: no single_offset.offset_eV to freeze.")
    return offsets


def rmse_with_offset(rows: list[dict[str, Any]], offset: float) -> float:
    """Pooled frontier RMSE after subtracting one fixed offset from every error."""
    return float(np.sqrt(np.mean([
        r["frontier_rmse_eV"] ** 2 + 2 * offset * r["frontier_global_shift_eV"] + offset**2
        for r in rows
    ])))


def selection_verdict(report: dict[str, Any], metric: str = "frontier_rmse_eV") -> dict[str, Any]:
    scores = {}
    for name, payload in report["checkpoints"].items():
        values = [r[metric] for r in payload["per_sample"]
                  if r["kpoint"] in SELECTION_KPOINTS and r["frontier_n_states"]]
        scores[name] = float(np.mean(values)) if values else float("nan")
    finite = {k: v for k, v in scores.items() if np.isfinite(v)}
    ordered = sorted(finite.values())
    separated = len(ordered) < 2 or (ordered[1] - ordered[0]) > 0.05 * ordered[1]
    lowest = min(finite, key=finite.get) if finite else None
    return {
        "metric": metric,
        "kpoints": list(SELECTION_KPOINTS),
        "mean_eV": scores,
        # A winner only when the gap is larger than the noise; otherwise the
        # lowest number is recorded but must not be read as a recommendation.
        "spectrally_best": lowest if separated else None,
        "numerically_lowest": lowest,
        "separation_is_meaningful": bool(separated),
    }


def print_report(report: dict[str, Any]) -> None:
    print(f"\nsplit de seleccion: {report['split_used_for_selection']} "
          f"({report['sample_count']} muestras) · seleccion solo en {report['selection_kpoints']}\n")
    for name, payload in report["checkpoints"].items():
        print(f"=== {name} ===")
        header = (f"{'k':8s} {'n~0':>4s} {'frontera RMSE':>14s} {'front. alin.':>13s} "
                  f"{'1er orden':>11s} {'todo RMSE':>10s} {'||dH~||_2':>10s}")
        print(header)
        print("-" * len(header))
        for label, metrics in payload["per_kpoint"].items():
            mark = "*" if label in report["selection_kpoints"] else " "
            print(f"{mark}{label:7s} {metrics['states_within_window']:4.1f} "
                  f"{metrics['frontier_rmse_eV'] * 1000:11.1f} meV "
                  f"{metrics['frontier_aligned_rmse_eV'] * 1000:9.1f} meV "
                  f"{metrics['frontier_projected_spectral_eV'] * 1000:7.1f} meV "
                  f"{metrics['all_rmse_eV']:10.3f} {metrics['delta_h_tilde_spectral_eV']:10.2f}")
        print("  correlacion en K/K' (norma vs error):")
        for target, correlations in payload["explains_frontier_error"].items():
            pairs = "  ".join(f"{norm.replace('_eV','').replace('delta_h','dH')}={value:+.3f}"
                              for norm, value in correlations.items())
            print(f"    {target:18s} {pairs}")
        print()
    for name, payload in report["checkpoints"].items():
        so = payload.get("single_offset") or {}
        if so:
            print(f"  {name}: crudo {so['rmse_raw_eV']*1000:.1f} meV | offset de este split "
                  f"{so['offset_eV']*1000:+.1f} meV -> {so['rmse_with_this_split_offset_eV']*1000:.1f} meV"
                  f" | por-muestra {so['per_sample_aligned_rmse_eV']*1000:.1f} meV (optimista, NO transferible)")
            frozen = (report.get("frozen_offsets_eV") or {}).get(name)
            if frozen is not None:
                value = rmse_with_offset(
                    [r for r in payload["per_sample"]
                     if r["kpoint"] in SELECTION_KPOINTS and r["frontier_n_states"]], frozen)
                print(f"      con el offset CONGELADO {frozen*1000:+.1f} meV -> {value*1000:.1f} meV  <-- cifra honesta")
    print()
    verdict = report["selection"]
    print(f"seleccion por {verdict['metric']} en {verdict['kpoints']} (solo validacion):")
    for name, score in sorted(verdict["mean_eV"].items(), key=lambda item: item[1]):
        print(f"   {name:20s} {score * 1000:8.2f} meV")
    if not verdict["separation_is_meaningful"]:
        print("   -> diferencia < 5%: NO hay ganador claro; no cambiar de checkpoint por esto.")
    else:
        print(f"   -> mejor: {verdict['spectrally_best']}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, action="append")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--split", default="validation", choices=("validation", "test"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--basis-files", default=None)
    parser.add_argument(
        "--calibration-report", type=Path, default=None,
        help="Report JSON from an earlier split (normally validation). Each checkpoint's "
             "own single_offset.offset_eV is frozen from it and applied here unchanged. "
             "This is the only protocol transferable to a target with no SIESTA reference; "
             "a single shared number would be wrong because every checkpoint has its own bias.",
    )
    args = parser.parse_args()

    checkpoints = args.checkpoint or sorted(DEFAULT_CHECKPOINT_DIR.glob("best-*.ckpt")) + sorted(
        DEFAULT_CHECKPOINT_DIR.glob("last-*.ckpt")
    )
    if not checkpoints:
        raise RuntimeError("No checkpoints found; pass --checkpoint explicitly.")
    basis_glob = args.basis_files or str(args.dataset / "splits/test/4/*.ion.xml")
    args.output_root.mkdir(parents=True, exist_ok=True)

    report = evaluate(checkpoints, args.dataset.resolve(), args.split, args.limit,
                      args.output_root.resolve(), basis_glob)
    report["frozen_offsets_eV"] = frozen_offsets(args.calibration_report)
    report["selection"] = selection_verdict(report)
    destination = args.output_root / f"checkpoint_spectral_metrics_{args.split}.json"
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print_report(report)
    print(f"escrito: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
