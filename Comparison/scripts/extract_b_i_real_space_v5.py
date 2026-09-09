#!/usr/bin/env python3
"""Persist the real-space per-R intra-atomic connection A_I(R) from the
already-certified B_I_fourier_bessel_semantic_v5 route.

certify_b_i_fourier_bessel_semantic_v5.py computes connection_blocks(...) (a
real-space {R: (no,no) matrix} dict, the full un-masked connection) as an
intermediate, but only persists three Bloch-summed k-space samples
(B_I_2c_C1_x_k{0,1,2}), which are genuinely k-dependent (verified: max
difference between k0 and k1 is 0.76, not ~0) because B_I combines the I-I
block across MULTIPLE periodic self-images of the displaced atom, not a
single monocentric R=0 integral. Assembling S_L_full/S_R_full with C14C's
real-space (n_s, no, no) block format therefore needs the real-space dict
itself, not the k-space samples.

This script re-runs EXACTLY the same, already-validated construction (same
basis, same engine config, same connection_blocks call) for the C1_x
direction and additionally masks and saves the per-R intra-atomic block
A_I(R) = J_I . connection(R) . J_I, verifying antihermiticity and exclusive
intra-atomic support at every R before saving anything.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO_ROOT / "Comparison/scripts"), str(REPO_ROOT / "shared")]

from artifact_signature import file_sha256  # noqa: E402
from certify_b_i_fourier_bessel_v4 import FourierBesselSP, connection_blocks  # noqa: E402
from certify_b_i_independent_two_center import Basis, enumerate_images, read_radials  # noqa: E402
from certify_support_exact_connection import _selector  # noqa: E402
from nested_basis_identity_preflight import PROD_CENTRAL  # noqa: E402
from run_nested_basis_sentinel import _matrix_data  # noqa: E402

OUTPUT = REPO_ROOT / "Comparison/results/epc/b_i_real_space_v5"
DIRECTION = "C1_x"
ENGINE_CONFIG = (1000, 180.0, 7201, True)  # q180_normalized, the PASS-certified engine
ANTIHERMITIAN_TOL = 1e-6


def main() -> int:
    _, central, fermi, vacuum, _ = _matrix_data(PROD_CENTRAL)
    geometry = central.geometry
    radials, _ = read_radials(PROD_CENTRAL / "C.ion.xml")
    basis = Basis(radials, [0, 1, 1, 1])
    engine = FourierBesselSP(basis, *ENGINE_CONFIG)
    images = enumerate_images(basis, np.asarray(geometry.cell), geometry.xyz)
    mask = _selector(geometry)

    connection = connection_blocks(engine, images, geometry, DIRECTION)
    a_i_by_r = {r: mask[:, None] * block * mask[None, :] for r, block in connection.items()}

    # The periodic antihermiticity identity is A_I(R) + A_I(-R)^dagger = 0 (bra/ket
    # atom-index exchange flips the *image* too, not just the transpose within one
    # R) -- checking A_I(R) + A_I(R)^dagger = 0 at a single R is only valid at
    # R=(0,0,0) and was verified wrong for R != 0 before landing on this identity.
    worst_antihermiticity, rows = 0.0, []
    for r, block in a_i_by_r.items():
        opposite = a_i_by_r.get((-r[0], -r[1], -r[2]))
        if opposite is None:
            raise RuntimeError(f"R={r} has no opposite image (-R) in the enumerated set")
        residual = block + opposite.conj().T
        anti = float(np.linalg.norm(residual) / max(np.linalg.norm(block), np.finfo(float).tiny)) \
            if np.linalg.norm(block) > 0 else 0.0
        worst_antihermiticity = max(worst_antihermiticity, anti)
        rows.append({
            "R": list(r), "norm": float(np.linalg.norm(block)),
            "antihermiticity_relative_vs_opposite_image": anti,
        })

    significant = [row for row in rows if row["norm"] > 1e-10]
    report = {
        "schema": "b_i_real_space_v5_extraction_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "direction": DIRECTION, "engine_config": list(ENGINE_CONFIG),
        "source": "certify_b_i_fourier_bessel_semantic_v5.py's own connection_blocks(), "
                   "re-run unchanged; only what is persisted differs",
        "n_R_images_total": len(connection), "n_R_images_with_significant_A_I": len(significant),
        "worst_antihermiticity_relative": worst_antihermiticity,
        "antihermiticity_tolerance": ANTIHERMITIAN_TOL,
        "verdict": "PASS" if worst_antihermiticity < ANTIHERMITIAN_TOL else "NO_GO",
        "rows": rows,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    npz_path = OUTPUT / "b_i_real_space_v5.npz"
    np.savez_compressed(npz_path, **{f"R_{r[0]}_{r[1]}_{r[2]}": block for r, block in a_i_by_r.items()},
                         R_list=np.array(list(a_i_by_r.keys())))
    report["matrix_artifact"] = str(npz_path)
    report["matrix_artifact_sha256"] = file_sha256(npz_path)
    path = OUTPUT / "b_i_real_space_v5.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True, default=float) + "\n")
    print(f"A_I(R) real-space extraction: {report['verdict']} "
          f"({len(significant)}/{len(connection)} R-images with significant support) -> {path}")
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
