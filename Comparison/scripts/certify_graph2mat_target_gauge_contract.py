#!/usr/bin/env python3
"""Identify the checkpoint's electronic target gauge from real label artifacts."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "Comparison/scripts"), str(ROOT / "shared"),
                str(ROOT / "scripts/torch_serialization_compat")]

from artifact_signature import file_sha256
from certify_epc_energy_zero import vacuum_level
from run_nested_basis_sentinel import _matrix_data, _production_run

CHECKPOINT = ROOT / "Comparison/results/tbg_registry_spectral_loss/training/checkpoints/spectral-best-epoch=247-step=03472.ckpt"
TRAIN_RUN = ROOT / "Comparison/datasets/tbg_md_plus_registry/n558/splits/train/0/RUN.fdf"
OUTPUT = ROOT / "Comparison/results/epc/pao_flow_audit/graph2mat_target_gauge_contract.json"
K_POINTS = ([0., 0., 0.], [1/3, 1/3, 0.], [.29, .11, 0.])
WEIGHTS = {-2: 1., -1: -8., 1: 8., 2: -1.}
H = .005
TARGET_RELATIVE_LIMIT = 1e-4
SEPARATION_FACTOR = 100.


def _processor():
    import torch
    from torch_safe_globals import allow_graph2mat_checkpoint_globals
    allow_graph2mat_checkpoint_globals()
    raw = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    hp, dm = raw["hyper_parameters"], raw["datamodule_hyper_parameters"]
    from graph2mat import MatrixDataProcessor
    return MatrixDataProcessor(
        basis_table=hp["basis_table"], out_matrix=dm["out_matrix"],
        symmetric_matrix=dm["symmetric_matrix"], sub_point_matrix=dm["sub_point_matrix"],
        n_matrix_components=dm.get("n_matrix_components", 1),
        matrix_component_policy=dm.get("matrix_component_policy", "h_only")), dm


def _processed_label(source, processor):
    import sisl
    from graph2mat.core.data.processing import BasisMatrixData
    source = source.copy()
    source._geometry = sisl.Geometry(source.geometry.xyz,
        atoms=[processor.basis_table.atoms[0]]*source.geometry.na, lattice=source.geometry.lattice)
    data = BasisMatrixData.new(source, data_processor=processor, labels=True)
    return processor.matrix_from_data(data)


def _errors(label, hamiltonian, fermi, vacuum=None):
    rows = []
    for k in K_POINTS:
        target = np.asarray(label.Hk(k=k, format="array"))
        shifted = np.asarray(hamiltonian.Hk(k=k, format="array"))
        overlap = np.asarray(hamiltonian.Sk(k=k, format="array"))
        options = {"H_sisl_equals_H_abs_minus_EfS": shifted,
                   "H_absolute": shifted + fermi*overlap}
        if vacuum is not None:
            options["H_absolute_minus_cvacS"] = shifted + (fermi-vacuum)*overlap
        denom = max(np.linalg.norm(target), np.finfo(float).tiny)
        errors = {name: float(np.linalg.norm(target-value)/denom) for name, value in options.items()}
        winner = min(errors, key=errors.get)
        ordered = sorted(errors.values())
        rows.append({"k_reduced": k, "relative_frobenius": errors, "closest": winner,
                     "separation_to_second_best": ordered[1]/max(ordered[0], np.finfo(float).tiny)})
    return rows


def _run_case(run: Path, processor, vacuum=None):
    import sisl
    sile = sisl.get_sile(str(run))
    hamiltonian = sile.read_hamiltonian()
    label = _processed_label(hamiltonian, processor)
    fermi = float(sile.read_fermi_level())
    rows = _errors(label, hamiltonian, fermi, vacuum)
    tshs = next(run.parent.glob("*.TSHS"))
    return {"run_fdf": str(run), "run_fdf_sha256": file_sha256(run),
            "tshs": str(tshs), "tshs_sha256": file_sha256(tshs),
            "fermi_ev": fermi, "vacuum_ev": vacuum, "k_checks": rows}


def main() -> int:
    processor, dm = _processor()
    cases = [_run_case(TRAIN_RUN, processor)]
    levels = {}
    for step in (-2, -1, 0, 1, 2):
        run_dir = _production_run("C1_x", step)
        label, _, fermi, vacuum, _ = _matrix_data(run_dir)
        levels[step] = {"fermi": fermi, "vacuum": vacuum}
        cases.append(_run_case(run_dir / f"{label}.TSHS", processor, vacuum))
    passed = all(
        row["closest"] == "H_sisl_equals_H_abs_minus_EfS"
        and row["relative_frobenius"][row["closest"]] < TARGET_RELATIVE_LIMIT
        and row["separation_to_second_best"] > SEPARATION_FACTOR
        for case in cases for row in case["k_checks"])
    derivative = lambda key: sum(w*levels[j][key] for j, w in WEIGHTS.items())/(12*H)
    report = {
        "schema": "graph2mat_target_gauge_contract_v1", "gate": "graph2mat_target_gauge_contract",
        "generated_at": datetime.now(timezone.utc).isoformat(), "verdict": "PASS" if passed else "NO_GO",
        "target_contract": "Graph2Mat labels are sisl read_hamiltonian(): H_absolute-E_F(R)S",
        "comparison_contract": "convert model derivative to H_absolute-c_vac(R)S before comparing Delta_PAO_cov",
        "thresholds": {"target_relative_frobenius": TARGET_RELATIVE_LIMIT,
                       "separation_to_second_best": SEPARATION_FACTOR},
        "checkpoint": str(CHECKPOINT), "checkpoint_sha256": file_sha256(CHECKPOINT),
        "checkpoint_data_policy": {key: dm.get(key) for key in
            ("out_matrix", "matrix_component_policy", "n_matrix_components", "sub_point_matrix", "symmetric_matrix")},
        "real_artifact_checks": cases,
        "C1_x_five_point_derivatives_ev_per_ang": {
            "d_Ef": derivative("fermi"), "d_c_vac": derivative("vacuum"),
            "d_cvac_minus_Ef": derivative("vacuum")-derivative("fermi")},
        "scope": "contract identification only; does not certify numerical convergence, lineage or full_KS",
        "status_note": "threshold confirmation follows prior exploratory evidence and is not a blind holdout",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    print(report["verdict"], report["C1_x_five_point_derivatives_ev_per_ang"])
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
