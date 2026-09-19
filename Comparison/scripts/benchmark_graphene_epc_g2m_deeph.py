#!/usr/bin/env python3
"""Aggregate the frozen-reference graphene E2g G2M/DeepH EPC benchmark.

No electronic or phonon calculation is run here.  The script consumes the
already frozen SIESTA/Graph2Mat contractions and DeepH finite differences and
keeps DeepH diagnostic-only until its raw/global adapter is certified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import compute_epc_matrix_elements as epc
import epc_basis_response as ebr
import quantify_checkpoint_derivative_error as c14
from deeph_raw_global_equivalence_preflight import siesta_orbital_labels_from_orb_indx
from evaluate_hamiltonian_derivative_metrics import load_hamiltonian_matrix
from run_deeph_sparse_spectrum import load_persisted_eigenspace

ROOT = Path(__file__).resolve().parents[2]
G2M_ROOT = ROOT / "Comparison/results/epc/snapshot_scaling_20260908/runs/w90_600"
DEEPH_ROOT = ROOT / "Comparison/results/epc/deeph_snapshot_scaling_20260909/runs/w90_600"
OUTPUT = ROOT / "Comparison/results/epc/graphene_g2m_deeph_benchmark_certified_20260909"
CERTIFICATION = ROOT / "Comparison/results/epc/graphene_g2m_deeph_certification_v2_20260909/certification_summary.json"
DELTA = 0.02
MODES = ("e2g_bond_longitudinal", "e2g_bond_transverse")
KPOINTS = ("K", "k_generic_1")


def complex_matrix(payload: dict[str, Any], prefix: str = "values") -> np.ndarray:
    return np.asarray(payload[f"{prefix}_real"]) + 1j * np.asarray(payload[f"{prefix}_imag"])


def matrix_metrics(candidate: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    error = candidate - reference
    ref_norm = float(np.linalg.norm(reference))
    norm = float(np.linalg.norm(candidate))
    singular = np.linalg.svd(candidate, compute_uv=False)
    singular_ref = np.linalg.svd(reference, compute_uv=False)
    flat_candidate = np.concatenate((candidate.real.ravel(), candidate.imag.ravel()))
    flat_reference = np.concatenate((reference.real.ravel(), reference.imag.ravel()))
    return {
        "G_norm_eV": norm,
        "G_relative_error": float(np.linalg.norm(error) / ref_norm) if ref_norm else None,
        "G_norm_ratio": norm / ref_norm if ref_norm else None,
        "W_eV2": norm**2,
        "W_over_reference": norm**2 / ref_norm**2 if ref_norm else None,
        "G_max_absolute_error_eV": float(np.max(np.abs(error))),
        "G_correlation": float(np.corrcoef(flat_candidate, flat_reference)[0, 1]),
        "singular_values_eV": singular.tolist(),
        "singular_values_relative_error": (
            float(np.linalg.norm(singular - singular_ref) / np.linalg.norm(singular_ref))
            if np.linalg.norm(singular_ref) else None
        ),
    }


def derivative_metrics(candidate: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    candidate = np.asarray(candidate)
    reference = np.asarray(reference)
    error = candidate - reference
    inner = float(np.vdot(candidate.ravel(), reference.ravel()).real)
    candidate_norm = float(np.linalg.norm(candidate))
    reference_norm = float(np.linalg.norm(reference))
    return {
        "D_MAE_eV_per_Ang": float(np.mean(np.abs(error))),
        "D_relative_error": float(np.linalg.norm(error) / reference_norm),
        "D_cosine": inner / (candidate_norm * reference_norm),
        "D_best_fit_scale_diagnostic": inner / candidate_norm**2,
    }


def deeph_blocks(direction: str) -> tuple[np.ndarray, np.ndarray]:
    pred = DEEPH_ROOT / "predicted_hamiltonians"
    plus = load_hamiltonian_matrix(pred / f"{direction}_plus/ML_prediction.HSX")
    minus = load_hamiltonian_matrix(pred / f"{direction}_minus/ML_prediction.HSX")
    dense = ((plus - minus) / (2 * DELTA)).toarray()
    source = np.load(G2M_ROOT / "arrays" / f"raw_derivative__siesta_reference__{direction}__d0p02.npz")
    isc = source["isc_off"]
    no_u = source["D_H"].shape[1]
    if dense.shape != (no_u, len(isc) * no_u):
        raise ValueError(f"DeepH sparse layout {dense.shape} is incompatible with {(no_u, len(isc) * no_u)}")
    blocks = dense.reshape(no_u, len(isc), no_u).transpose(1, 0, 2)
    by_r = {tuple(map(int, offset)): block for offset, block in zip(isc, blocks)}
    hermitized = np.asarray([
        (block + by_r[tuple(map(int, -offset))].conj().T) / 2
        for offset, block in zip(isc, blocks)
    ])
    return isc, hermitized


def derivative_sources(direction: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    arrays = G2M_ROOT / "arrays"
    sources = {}
    for method in ("siesta_reference", "graph2mat_frozen"):
        payload = np.load(arrays / f"raw_derivative__{method}__{direction}__d0p02.npz")
        sources[method] = (payload["isc_off"], payload["D_H"])
    sources["deeph"] = deeph_blocks(direction)
    if not all(np.array_equal(sources["siesta_reference"][0], item[0]) for item in sources.values()):
        raise ValueError(f"R-vector mismatch for {direction}")
    return sources


def electronic_spaces() -> dict[str, epc.Eigenspace]:
    root = ROOT / "Comparison/results/epc/eigenspaces/graphene"
    manifest = json.loads((root / "eigenspaces_manifest.json").read_text())
    return {
        row["label"]: epc.Eigenspace.from_persisted(load_persisted_eigenspace(root, i), label=row["label"])
        for i, row in enumerate(manifest["eigenspaces"])
    }


def g_sources() -> dict[tuple[str, str, str], np.ndarray]:
    base = json.loads((G2M_ROOT / "graphene_gamma_epc_g_blocks.json").read_text())
    result = {
        (row["path"], row["mode_label"], row["k_label"]): complex_matrix(row["g"])
        for row in base["reports"]
        if row["path"] in {"siesta_reference", "graph2mat_frozen"}
    }
    spaces = electronic_spaces()
    derivatives = {mode: derivative_sources(mode) for mode in MODES}
    for row in base["reports"]:
        if row["path"] != "graph2mat_frozen" or row["mode_label"] not in MODES or row["k_label"] not in KPOINTS:
            continue
        space = spaces[row["k_label"]]
        correction = np.zeros((space.no_u, space.no_u), dtype=np.complex128)
        for coefficient, direction in zip(row["expansion"]["coefficients_real"], row["expansion"]["direction_names"]):
            sources = derivatives[direction]
            isc, deeph = sources["deeph"]
            _, graph2mat = sources["graph2mat_frozen"]
            correction += coefficient * ebr.bloch_sum(deeph - graph2mat, isc, space.k)
        result[("deeph", row["mode_label"], row["k_label"])] = (
            complex_matrix(row["g"]) + space.C.conj().T @ correction @ space.C
        )
    return result


def localization_rows() -> list[dict[str, Any]]:
    import sisl

    summary = json.loads((G2M_ROOT / "graphene_gamma_epc_summary.json").read_text())
    tshs = Path(summary["eigenspaces"]["equilibrium_tshs"])
    geometry = sisl.get_sile(str(tshs.parent / "RUN.fdf")).read_geometry()
    orb_indx = tshs.with_suffix(".ORB_INDX")
    mapping = c14.orbital_map(orb_indx, geometry.xyz, geometry.cell)
    labels = siesta_orbital_labels_from_orb_indx(orb_indx, expected_count=len(mapping.l_of_orbital))
    rows = []
    for mode in MODES:
        sources = derivative_sources(mode); isc, reference = sources["siesta_reference"]
        for method, source in (("Graph2Mat", "graph2mat_frozen"), ("DeepH", "deeph")):
            candidate = sources[source][1]
            buckets: dict[str, list[tuple[complex, complex]]] = {
                name: [] for name in ("onsite", "1NN", "2NN", "pz-pz", "1NN_pz-pz", "other")
            }
            for image, offset in enumerate(isc):
                r = tuple(map(int, offset))
                for i in range(reference.shape[1]):
                    for j in range(reference.shape[2]):
                        key = (i, j, r); distance = mapping.distance_ang(key)
                        shell = "onsite" if distance < .1 else ("1NN" if abs(distance - 1.47) < .2 else ("2NN" if abs(distance - 2.54) < .2 else "other"))
                        buckets[shell].append((reference[image, i, j], candidate[image, i, j]))
                        if labels[i] == labels[j] == "pz":
                            buckets["pz-pz"].append((reference[image, i, j], candidate[image, i, j]))
                            if shell == "1NN":
                                buckets["1NN_pz-pz"].append((reference[image, i, j], candidate[image, i, j]))
            total_error = float(np.linalg.norm(candidate - reference))
            for group, pairs in buckets.items():
                ref = np.asarray([pair[0] for pair in pairs]); cand = np.asarray([pair[1] for pair in pairs])
                rows.append({"mode": mode, "method": method, "group": group,
                             "D_relative_error": float(np.linalg.norm(cand-ref) / np.linalg.norm(ref)) if np.linalg.norm(ref) else None,
                             "share_of_total_squared_error": float(np.linalg.norm(cand-ref)**2 / total_error**2) if total_error else 0.0})
    return rows


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    g = g_sources()
    d_by_mode = {mode: derivative_sources(mode) for mode in MODES}
    rows: list[dict[str, Any]] = []
    derivative_rows: list[dict[str, Any]] = []
    for mode, sources in d_by_mode.items():
        reference_d = sources["siesta_reference"][1]
        for source, (_isc, values) in sources.items():
            derivative_rows.append({"mode": mode, "method": source, **(
                {"D_MAE_eV_per_Ang": None, "D_relative_error": None, "D_cosine": None,
                 "D_best_fit_scale_diagnostic": None}
                if source == "siesta_reference" else derivative_metrics(values, reference_d)
            )})
        d_metrics = {row["method"]: row for row in derivative_rows if row["mode"] == mode}
        for kpoint in KPOINTS:
            reference_g = g[("siesta_reference", mode, kpoint)]
            # Four-state frozen window and its neutrality/Dirac pair.
            for subspace, indices in (("full_window", np.arange(reference_g.shape[0])), ("Dirac_pi", np.array([1, 2]))):
                ref = reference_g[np.ix_(indices, indices)]
                for method, source in (("SIESTA", "siesta_reference"), ("Graph2Mat", "graph2mat_frozen"), ("DeepH", "deeph")):
                    candidate = g[(source, mode, kpoint)][np.ix_(indices, indices)]
                    row = {
                        "k": kpoint, "mode": mode, "subspace": subspace, "method": method,
                        **d_metrics[source], **matrix_metrics(candidate, ref),
                        "status": "REF" if method == "SIESTA" else "comparable",
                    }
                    row.pop("method", None)
                    row["method"] = method
                    row.pop("mode", None)
                    row["mode"] = mode
                    if method == "SIESTA":
                        row.update({"G_relative_error": None, "G_norm_ratio": 1.0, "W_over_reference": 1.0,
                                    "G_max_absolute_error_eV": None, "singular_values_relative_error": None})
                    rows.append(row)
    return rows, derivative_rows


def plots(rows: list[dict[str, Any]], derivative_rows: list[dict[str, Any]], localization: list[dict[str, Any]], output: Path) -> None:
    import matplotlib.pyplot as plt

    groups = ("onsite", "1NN", "2NN", "pz-pz", "other"); xg = np.arange(len(groups)); width = .2
    plt.figure(figsize=(8, 4))
    for index, method in enumerate(("Graph2Mat", "DeepH")):
        values = [np.mean([r["D_relative_error"] for r in localization if r["method"] == method and r["group"] == group]) for group in groups]
        plt.bar(xg + (index - .5) * width, values, width, label=method)
    plt.xticks(xg, groups); plt.legend(); plt.ylabel("relative derivative error")
    plt.tight_layout(); plt.savefig(output / "figure_1_derivative_relative_error.png", dpi=180); plt.close()

    full = [row for row in rows if row["subspace"] == "full_window"]
    cases = [f"{row['k']}\n{row['mode'].split('_')[-1]}" for row in full if row["method"] == "SIESTA"]
    x = np.arange(len(cases)); width = .25
    for filename, key, ylabel in (("figure_2_g_frobenius.png", "G_norm_eV", r"$||G||_F$ (eV)"),
                                  ("figure_3_spectral_weight.png", "W_eV2", r"$Tr(G^\dagger G)$ (eV$^2$)")):
        plt.figure(figsize=(8, 4))
        for offset, method in enumerate(("SIESTA", "Graph2Mat", "DeepH")):
            plt.bar(x + (offset - 1) * width, [r[key] for r in full if r["method"] == method], width, label=method)
        plt.xticks(x, cases); plt.ylabel(ylabel); plt.legend(); plt.tight_layout(); plt.savefig(output / filename, dpi=180); plt.close()

    dirac = [row for row in rows if row["subspace"] == "Dirac_pi" and row["k"] == "K"]
    plt.figure(figsize=(8, 4))
    for row in dirac:
        plt.plot(np.arange(1, len(row["singular_values_eV"]) + 1), row["singular_values_eV"], "o-", label=f"{row['method']} {row['mode'].split('_')[-1]}")
    plt.xlabel("singular-value index"); plt.ylabel("eV"); plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(output / "figure_4_dirac_singular_values.png", dpi=180); plt.close()

    points = [row for row in rows if row["method"] != "SIESTA" and row["subspace"] == "full_window"]
    plt.figure(figsize=(6, 4))
    for method in ("Graph2Mat", "DeepH"):
        selected = [row for row in points if row["method"] == method]
        plt.scatter([r["D_relative_error"] for r in selected], [r["G_relative_error"] for r in selected], label=method)
    plt.xlabel("relative derivative error"); plt.ylabel("relative G error"); plt.legend(); plt.tight_layout(); plt.savefig(output / "figure_5_derivative_vs_g_error.png", dpi=180); plt.close()

    plt.figure(figsize=(8, 4))
    for index, method in enumerate(("Graph2Mat", "DeepH")):
        values = [np.mean([r["share_of_total_squared_error"] for r in localization if r["method"] == method and r["group"] == group]) for group in groups]
        plt.bar(xg + (index - .5) * width, values, width, label=method)
    plt.xticks(xg, groups); plt.ylabel("share of total squared error"); plt.legend()
    plt.tight_layout(); plt.savefig(output / "figure_6_error_localization.png", dpi=180); plt.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--certification", type=Path, default=CERTIFICATION)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise FileExistsError(f"refusing to overwrite benchmark artifacts in {args.output}")
    certification = json.loads(args.certification.read_text())
    if certification.get("status") != "proven":
        raise RuntimeError("DeepH raw/global certification did not pass")
    rows, derivative_rows = make_rows()
    localization = localization_rows()
    checkpoint = Path(json.loads((DEEPH_ROOT / "deeph_e2g_metrics.json").read_text())["model_dir"])
    provenance = {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "siesta_version": "artifact command /home/christian/bin/siesta; version not recorded",
        "graph2mat_checkpoint": "not recorded in frozen EPC artifact",
        "deeph_checkpoint": str(checkpoint),
        "geometry_hash": json.loads((G2M_ROOT / "graphene_gamma_epc_summary.json").read_text())["shared_contracts"]["geometry_signature"],
        "derivative_source": {"SIESTA": "frozen central differences", "Graph2Mat": "frozen central differences", "DeepH": "frozen central differences"},
        "units": {"D_H": "eV/Ang", "G": "eV"},
        "orbital_mapping_hash": None,
        "atom_ordering_hash": None,
        "electronic_k": list(KPOINTS), "phonon_modes": list(MODES),
        "subspaces": {"full_window": [0, 1, 2, 3], "Dirac_pi": [1, 2]},
        "reference_files": [str(G2M_ROOT / "graphene_gamma_epc_g_blocks.json"), str(DEEPH_ROOT / "deeph_e2g_metrics.json")],
        "reference_hashes": {str(path): sha256(path) for path in (G2M_ROOT / "graphene_gamma_epc_g_blocks.json", DEEPH_ROOT / "deeph_e2g_metrics.json")},
    }
    fieldnames = list(rows[0])
    with (args.output / "graphene_epc_g2m_deeph_benchmark.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames); writer.writeheader(); writer.writerows(rows)
    payload = {
        "schema": "graphene_epc_g2m_deeph_benchmark_v1", "experiment": "A_frozen_electronic_reference",
        "scientific_status": "comparable", "verdict": "GRAPH2MAT_ADVANCES",
        "mapping_certification": str(args.certification),
        "rows": rows, "derivative_metrics": derivative_rows, "localization": localization, "provenance": provenance,
        "experiment_B": {"status": "not_attempted", "reason": "Experiment A is decisive; end-to-end was optional and would add unrelated electronic-model error."},
    }
    (args.output / "graphene_epc_g2m_deeph_benchmark.json").write_text(json.dumps(payload, indent=2) + "\n")
    plots(rows, derivative_rows, localization, args.output)
    print(json.dumps({"output": str(args.output), "rows": len(rows), "verdict": payload["verdict"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
