#!/usr/bin/env python3
"""Free diagnostics for the uniform_translation_PAO_covariance_v1 NO_GO at
K_POINTS[1] = (1/3, 1/3, 0) -- a point this module's docstring originally
mislabelled as "K". Direct diagonalisation shows (1/3, 1/3, 0) has a minimum
band gap of ~1.15 eV: it is NOT the Dirac point for PROD-SZ's cell. The true
K/K' are at (1/3, 2/3, 0) / (2/3, 1/3, 0), which show ~1e-6 eV degeneracies --
see ``cross_basis_projection_preflight.K_POINTS_HISTORICAL_LABELS``.

Does NOT relabel, reinterpret, or change any threshold of v1 -- that verdict
stays historical. Runs two diagnostics against the *existing* TSHS artifacts,
no new SIESTA:

(A) Real-space decomposition: builds D5K(R) directly from the real-space
    H(R)/S(R) blocks of the four translation_x TSHS files (not the Bloch sum),
    reconstructs D5K(k) = sum_R D5K(R) exp(i k.R) at every K_POINTS reduced
    k-point, cross-checks it against the k-space construction already used by
    the certifiers, and reports which R blocks dominate the norm -- evidence
    for or against "a small real-space (grid/egg-box) residual that adds up
    constructively away from Gamma".

(B) Degenerate-subspace check: at a genuinely degenerate point, the diagonal
    of C^dagger Delta C in one arbitrary eigenbasis choice is not, by itself,
    independent evidence that the physically meaningful (degenerate-subspace)
    part of Delta_PAO_cov_trans is small -- any rotation within a degenerate
    subspace is an equally valid eigenbasis. This computes the full block
    C_D^dagger Delta_PAO_cov_trans C_D for whichever eigenvalue cluster forms
    at each k, using a data-driven (not tuned-after-seeing-the-answer)
    degeneracy tolerance. K_POINTS[1] itself has no such cluster (gaps > 1eV),
    so this diagnostic is empty there by construction; it matters for any
    future certification actually run at true K/K'.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.linalg import eigh

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO_ROOT / "Comparison/scripts"), str(REPO_ROOT / "shared")]

import sisl  # noqa: E402

import fd_perturbation_space as fdp  # noqa: E402
from artifact_signature import file_sha256  # noqa: E402
from certify_delta_plateau_topology import D5_WEIGHTS, PROD_SZ_ROOT, _d5, _label  # noqa: E402
from certify_full_basis_connection_semantic_v5 import solve  # noqa: E402
from certify_uniform_translation_pao_covariance import (  # noqa: E402
    H_PRODUCTION, _translation_connections,
)
from cross_basis_projection_preflight import K_POINTS, _basis_geometry  # noqa: E402
from nested_basis_identity_preflight import PROD_CENTRAL  # noqa: E402
from run_nested_basis_sentinel import _matrix_data  # noqa: E402

OUTPUT = REPO_ROOT / "Comparison/results/epc/translation_k_point_residual_diagnostic"

# A degenerate cluster is a run of consecutive eigenvalues whose gaps are all
# below this fraction of the *median* gap across the whole spectrum -- an
# objective, data-driven rule fixed before looking at any cluster's content.
DEGENERACY_GAP_FRACTION = 0.05


def _run_dir(step: int) -> Path:
    tag = f"d0p{'01' if abs(step) == 2 else '005'}_{'plus' if step > 0 else 'minus'}"
    return PROD_SZ_ROOT / "translation_x" / tag


def _real_space_blocks(run_dir: Path):
    label = _label(run_dir)
    h = sisl.get_sile(str(run_dir / f"{label}.TSHS")).read_hamiltonian()
    fermi = float(sisl.get_sile(str(run_dir / f"{label}.TSHS")).read_fermi_level())
    from certify_epc_energy_zero import vacuum_level
    vacuum, _ = vacuum_level(sisl.get_sile(str(run_dir / f"{label}.VT")).read_grid().grid)
    geom = h.geometry
    no = geom.no
    h_csr = h.tocsr()
    s_csr = h.tocsr(-1)
    blocks = {}
    for isc in range(geom.n_s):
        r = tuple(int(v) for v in geom.sc_off[isc])
        h_block = h_csr[:, isc * no:(isc + 1) * no].toarray()
        s_block = s_csr[:, isc * no:(isc + 1) * no].toarray()
        blocks[r] = h_block + fermi * s_block - vacuum * s_block
    return blocks


def diagnostic_a() -> dict:
    per_step_blocks = {step: _real_space_blocks(_run_dir(step)) for step in D5_WEIGHTS}
    common_r = set.intersection(*(set(blocks) for blocks in per_step_blocks.values()))
    d5k_r = {
        r: sum(D5_WEIGHTS[step] * per_step_blocks[step][r] for step in D5_WEIGHTS) / (12.0 * H_PRODUCTION)
        for r in common_r
    }
    ranked = sorted(d5k_r.items(), key=lambda item: -np.linalg.norm(item[1]))
    top_contributors = [{"R": list(r), "frobenius_norm_ev_per_ang": float(np.linalg.norm(block))}
                         for r, block in ranked[:8]]

    from certify_delta_plateau_topology import _k_matrix, _point
    points_by_step = {step: _point(_run_dir(step))[:3] for step in D5_WEIGHTS}

    checks = []
    for k in K_POINTS:
        reconstructed = sum(block * np.exp(2j * np.pi * np.dot(k, r)) for r, block in d5k_r.items())
        # Direct k-space construction, independent of the real-space extraction above.
        direct_by_step = {step: _k_matrix(*points_by_step[step], k) for step in D5_WEIGHTS}
        direct = _d5(direct_by_step, H_PRODUCTION)
        mismatch = float(np.abs(reconstructed - direct).max())
        checks.append({
            "k_reduced": list(k), "real_space_vs_kspace_max_diff_ev_per_ang": mismatch,
            "self_consistent": mismatch < 1e-9,
        })
    return {"common_R_count": len(common_r), "top_contributors_by_norm": top_contributors, "consistency_checks": checks}


def diagnostic_b() -> dict:
    central_label, central_h, central_fermi, central_vacuum, atom = _matrix_data(PROD_CENTRAL)
    shape = tuple(int(x) for x in sisl.get_sile(str(PROD_CENTRAL / f"{central_label}.VT")).read_grid().shape)
    base = _basis_geometry(central_h.geometry, PROD_CENTRAL / "C.ion.xml")
    dvolume = abs(np.linalg.det(base.cell)) / np.prod(shape)
    direction = fdp.uniform_translation(base.na, "x")
    connections = _translation_connections(direction, H_PRODUCTION, base, atom, shape, dvolume, K_POINTS)

    from certify_delta_plateau_topology import _k_matrix, _point
    points_by_step = {}
    for step in D5_WEIGHTS:
        run_dir = _run_dir(step)
        hamiltonian, fermi, vacuum, _ = _point(run_dir)
        points_by_step[step] = hamiltonian, fermi, vacuum

    rows = []
    for ik, k in enumerate(K_POINTS):
        s0 = np.asarray(central_h.Sk(k=k, format="array"))
        k0 = _k_matrix(central_h, central_fermi, central_vacuum, k)
        k_by_step = {step: _k_matrix(*points_by_step[step], k) for step in points_by_step}
        d5k_trans = _d5(k_by_step, H_PRODUCTION)
        s_l, s_r = connections[ik]
        left_term, _ = solve(s0, k0)
        right_term, _ = solve(s0, s_r)
        delta_trans = d5k_trans - s_l @ left_term - k0 @ right_term

        eigvals, c = eigh(k0, s0)
        gaps = np.diff(eigvals)
        median_gap = float(np.median(gaps)) if len(gaps) else 0.0
        threshold = DEGENERACY_GAP_FRACTION * median_gap

        clusters, current = [], [0]
        for i, gap in enumerate(gaps):
            if gap < threshold:
                current.append(i + 1)
            else:
                clusters.append(current)
                current = [i + 1]
        clusters.append(current)

        cluster_rows = []
        for cluster in clusters:
            if len(cluster) < 2:
                continue
            c_d = c[:, cluster]
            block = c_d.conj().T @ delta_trans @ c_d
            cluster_rows.append({
                "eigenvalue_indices": cluster,
                "eigenvalues_ev": [float(eigvals[i]) for i in cluster],
                "block_frobenius_norm_ev_per_ang": float(np.linalg.norm(block) / np.sqrt(block.size)),
                "block_max_abs_ev_per_ang": float(np.abs(block).max()),
                "block_diag_only_max_abs_ev_per_ang": float(np.abs(np.diag(block)).max()),
            })
        rows.append({
            "k_reduced": list(k), "median_eigenvalue_gap_ev": median_gap,
            "degeneracy_threshold_ev": threshold, "degenerate_clusters": cluster_rows,
        })
    return {"gap_fraction_rule": DEGENERACY_GAP_FRACTION, "rows": rows}


def main() -> int:
    report = {
        "schema": "translation_k_point_residual_diagnostic_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "free diagnostics only; does not change uniform_translation_PAO_covariance_v1's verdict",
        "diagnostic_A_real_space_decomposition": diagnostic_a(),
        "diagnostic_B_degenerate_subspace_block": diagnostic_b(),
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "translation_k_point_residual_diagnostic.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True, default=float) + "\n")
    print(f"diagnostic written -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
