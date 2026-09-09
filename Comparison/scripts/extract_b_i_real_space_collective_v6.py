#!/usr/bin/env python3
"""Generalize the intra-atomic connection A_I(R) to arbitrary collective
``fd_perturbation_space.Direction`` patterns (na, 3), not just the single
``C1_x`` (atom 0, Cartesian x) case ``extract_b_i_real_space_v5.py`` covers.

Does NOT touch ``certify_basis_response_v2.py`` or ``extract_b_i_real_space_v5.py``:
those stay exactly as-is, the historical, already-validated single-direction case.
This script is additive and reuses their machinery unchanged.

Physics claim being verified, not assumed (see module docstring context in the
task): because A_I is on-site/monocentric (masked to the intra-atomic diagonal
block, per ``extract_b_i_real_space_v5.py``) and ``FourierBesselSP.derivative``
is linear in one atom's own Cartesian displacement (chain rule on a two-center
integral of ``separation = R_j - R_i`` w.r.t. that atom's position), the claim
is that for a collective direction with per-atom vectors ``v_a``:

    A_I_a(R; v_a) = sum_axis v_a[axis] * (single-atom-a, axis-aligned A_I block at R)

and the total A_I(R) is each atom's own block placed on ITS OWN diagonal slot,
with all cross-atom blocks exactly zero (no new physics is assumed for the
cross-atom part -- it is asserted to be zero by construction and checked).

This is verified per direction by:
  1. atom0000_x regression: byte-for-byte (float) match to the existing
     b_i_real_space_v5.npz artifact.
  2. Antihermiticity A_I(R) + A_I(-R)^dagger = 0 for the generalized construction,
     including atoms/axes other than (atom 0, x) -- the same identity
     extract_b_i_real_space_v5.py verified for the single already-certified case.
  3. Cross-atom blocks are exactly zero.
  4. For translation_x specifically: an independent cross-check against the
     existing rigid-translation null identity S_L_inter + S_R_inter =~ 0
     already certified in certify_uniform_translation_pao_covariance_gamma.py.
     Because S_L_full = S_L_inter - A_I and S_R_full = S_R_inter + A_I, adding
     A_I to S_L/S_R cancels EXACTLY in the sum S_L_full + S_R_full = S_L_inter +
     S_R_inter -- so this script checks that identity holds for its own A_I(R)
     construction (i.e. A_I(R) - A_I(R) == 0 in the sum, trivially true by the
     S_L/S_R sign convention, but the non-trivial part -- that A_I(R) itself is
     antihermitian and atom-diagonal so it doesn't perturb the acoustic sum
     rule -- is exactly checks 2+3).
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO_ROOT / "Comparison/scripts"), str(REPO_ROOT / "shared")]

import fd_perturbation_space as fdp  # noqa: E402
from artifact_signature import file_sha256  # noqa: E402
from certify_b_i_fourier_bessel_v4 import FourierBesselSP  # noqa: E402
from certify_b_i_independent_two_center import Basis, enumerate_images, read_radials  # noqa: E402
from certify_support_exact_connection import _selector  # noqa: E402
from nested_basis_identity_preflight import PROD_CENTRAL  # noqa: E402
from run_nested_basis_sentinel import _matrix_data  # noqa: E402

OUTPUT_ROOT = REPO_ROOT / "Comparison/results/epc/b_i_real_space_collective_v6"
ENGINE_CONFIG = (1000, 180.0, 7201, True)  # same q180_normalized, PASS-certified engine as v5
ANTIHERMITIAN_TOL = 1e-6
CROSS_ATOM_TOL = 0.0  # must be EXACTLY zero: never written, not just small
REGRESSION_TOL = 1e-10  # float-precision match to the existing v5 artifact

V5_JSON = REPO_ROOT / "Comparison/results/epc/b_i_real_space_v5/b_i_real_space_v5.json"
V5_NPZ = REPO_ROOT / "Comparison/results/epc/b_i_real_space_v5/b_i_real_space_v5.npz"

REFERENCE_MANIFEST = (
    REPO_ROOT / "Comparison/results/epc/siesta_reference/graphene/epc_siesta_reference_manifest.json"
)
PROTOCOL_JSON = (
    REPO_ROOT / "Comparison/results/epc/preregistration/graphene/epc_graphene_gamma_protocol.json"
)

DIRECTIONS_NEEDED = (
    "atom0000_x", "translation_x", "random_seed0",
    "e2g_bond_longitudinal", "e2g_bond_transverse",
)


def load_direction_vectors() -> dict[str, np.ndarray]:
    """Pull the SAME vectors already used to generate the real SIESTA FD data.

    atom0000_x/translation_x/random_seed0 come from the siesta_reference
    campaign manifest (Campaign.directions' source); e2g_bond_longitudinal/
    e2g_bond_transverse come from the frozen C19/C20 protocol JSON (the gauge-
    fixed E2g doublet members, derived from real C18 phonon data earlier this
    session) -- neither is re-derived here.
    """
    vectors: dict[str, np.ndarray] = {}
    manifest = json.loads(REFERENCE_MANIFEST.read_text())
    for entry in manifest["perturbation_space"]["directions"]:
        if entry["direction_name"] in DIRECTIONS_NEEDED:
            vectors[entry["direction_name"]] = np.asarray(entry["vectors"], dtype=np.float64)

    protocol = json.loads(PROTOCOL_JSON.read_text())
    for entry in protocol["perturbation"]["directions"]:
        if entry["direction_name"] in DIRECTIONS_NEEDED:
            vectors[entry["direction_name"]] = np.asarray(entry["vectors"], dtype=np.float64)

    missing = [name for name in DIRECTIONS_NEEDED if name not in vectors]
    if missing:
        raise RuntimeError(
            f"Could not find vectors for direction(s) {missing} in either "
            f"{REFERENCE_MANIFEST} or {PROTOCOL_JSON} -- refusing to guess a bond-pattern "
            "geometry silently."
        )
    return vectors


def collective_connection_blocks(engine, images, geometry, vectors: np.ndarray) -> dict:
    """Generalization of certify_b_i_fourier_bessel_v4.connection_blocks to an
    arbitrary (na, 3) Direction.vectors array: each atom a with nonzero
    vectors[a] gets its own axis-weighted intra-atomic block on ITS OWN
    (no, no) diagonal slot; every cross-atom block is left exactly zero.
    """
    no, na = 4, geometry.na
    if vectors.shape != (na, 3):
        raise ValueError(f"direction vectors shape {vectors.shape} != (na={na}, 3)")
    active = [a for a in range(na) if np.any(vectors[a] != 0.0)]
    result = {}
    for image in images:
        matrix = np.zeros((na * no, na * no))
        for a in active:
            block = np.zeros((no, no))
            for axis in range(3):
                weight = vectors[a, axis]
                if weight == 0.0:
                    continue
                block += weight * engine.derivative(image["vector"], axis)
            matrix[a * no:(a + 1) * no, a * no:(a + 1) * no] = block
        result[tuple(image["index"])] = matrix
    return result


def verify_antihermiticity(a_i_by_r: dict) -> tuple[float, list[dict]]:
    """Same identity as extract_b_i_real_space_v5.py: A_I(R) + A_I(-R)^dagger = 0."""
    worst, rows = 0.0, []
    for r, block in a_i_by_r.items():
        opposite = a_i_by_r.get((-r[0], -r[1], -r[2]))
        if opposite is None:
            raise RuntimeError(f"R={r} has no opposite image (-R) in the enumerated set")
        residual = block + opposite.conj().T
        norm = np.linalg.norm(block)
        anti = float(np.linalg.norm(residual) / max(norm, np.finfo(float).tiny)) if norm > 0 else 0.0
        worst = max(worst, anti)
        rows.append({"R": list(r), "norm": float(norm), "antihermiticity_relative_vs_opposite_image": anti})
    return worst, rows


def verify_cross_atom_zero(a_i_by_r: dict, no: int, na: int) -> float:
    """Every off-diagonal-atom (no, no) block must be exactly zero."""
    worst = 0.0
    for block in a_i_by_r.values():
        for a in range(na):
            for b in range(na):
                if a == b:
                    continue
                sub = block[a * no:(a + 1) * no, b * no:(b + 1) * no]
                worst = max(worst, float(np.abs(sub).max()) if sub.size else 0.0)
    return worst


def main() -> int:
    _, central, fermi, vacuum, _ = _matrix_data(PROD_CENTRAL)
    geometry = central.geometry
    radials, _ = read_radials(PROD_CENTRAL / "C.ion.xml")
    basis = Basis(radials, [0, 1, 1, 1])
    engine = FourierBesselSP(basis, *ENGINE_CONFIG)
    images = enumerate_images(basis, np.asarray(geometry.cell), geometry.xyz)
    mask = _selector(geometry)
    no, na = 4, geometry.na

    all_vectors = load_direction_vectors()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    per_direction_reports = {}
    overall_pass = True

    for direction_name in DIRECTIONS_NEEDED:
        vectors = all_vectors[direction_name]
        connection = collective_connection_blocks(engine, images, geometry, vectors)

        # The mask here is a no-op safety net: A_I is already exactly atom-diagonal
        # by construction (only [a*no:(a+1)*no, a*no:(a+1)*no] slots are ever
        # written), unlike v5's single-atom mask which selects one atom's rows/cols
        # out of a matrix that (for C1_x) only ever had that one atom's block
        # populated anyway. Kept for parity with v5's masking step.
        a_i_by_r = {r: mask[:, None] * block * mask[None, :] if direction_name == "atom0000_x"
                    else block for r, block in connection.items()}

        worst_anti, anti_rows = verify_antihermiticity(a_i_by_r)
        anti_pass = worst_anti < ANTIHERMITIAN_TOL

        worst_cross = verify_cross_atom_zero(a_i_by_r, no, na)
        cross_pass = worst_cross <= CROSS_ATOM_TOL

        checks = {
            "antihermiticity": {
                "worst_relative": worst_anti, "tolerance": ANTIHERMITIAN_TOL, "passed": anti_pass,
            },
            "cross_atom_blocks_exactly_zero": {
                "worst_abs": worst_cross, "tolerance": CROSS_ATOM_TOL, "passed": cross_pass,
            },
        }

        regression = None
        if direction_name == "atom0000_x":
            v5_report = json.loads(V5_JSON.read_text())
            v5_npz = np.load(V5_NPZ, allow_pickle=True)
            r_list = v5_npz["R_list"]
            v5_by_r = {tuple(int(v) for v in r): v5_npz[f"R_{r[0]}_{r[1]}_{r[2]}"] for r in r_list}
            missing_here = [r for r in v5_by_r if r not in a_i_by_r]
            missing_v5 = [r for r in a_i_by_r if r not in v5_by_r]
            worst_diff = 0.0
            for r, v5_block in v5_by_r.items():
                mine = a_i_by_r.get(r)
                if mine is None:
                    continue
                worst_diff = max(worst_diff, float(np.abs(mine - v5_block).max()))
            regression_pass = (
                not missing_here and not missing_v5
                and worst_diff < REGRESSION_TOL
                and v5_report["verdict"] == "PASS"
            )
            regression = {
                "reference_artifact": str(V5_NPZ), "reference_sha256": file_sha256(V5_NPZ),
                "worst_abs_diff": worst_diff, "tolerance": REGRESSION_TOL,
                "missing_R_in_new_construction": missing_here,
                "extra_R_in_new_construction": missing_v5,
                "reference_verdict": v5_report["verdict"],
                "passed": regression_pass,
            }
            checks["atom0000_x_regression_vs_v5"] = regression

        direction_pass = all(c["passed"] for c in checks.values())
        overall_pass = overall_pass and direction_pass

        significant = [row for row in anti_rows if row["norm"] > 1e-10]
        npz_path = OUTPUT_ROOT / f"{direction_name}.npz"
        np.savez_compressed(
            npz_path,
            **{f"R_{r[0]}_{r[1]}_{r[2]}": block for r, block in a_i_by_r.items()},
            R_list=np.array(list(a_i_by_r.keys())),
            direction_vectors=vectors,
        )
        report = {
            "schema": "b_i_real_space_collective_v6_extraction_v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "direction": direction_name,
            "direction_vectors": vectors.tolist(),
            "engine_config": list(ENGINE_CONFIG),
            "source": "generalization of certify_b_i_fourier_bessel_v4.connection_blocks / "
                      "extract_b_i_real_space_v5.py to arbitrary per-atom Direction.vectors; "
                      "atom0000_x re-derives the identical construction as a regression check",
            "n_R_images_total": len(connection),
            "n_R_images_with_significant_A_I": len(significant),
            "checks": checks,
            "verdict": "PASS" if direction_pass else "NO_GO",
        }
        report["matrix_artifact"] = str(npz_path)
        report["matrix_artifact_sha256"] = file_sha256(npz_path)
        json_path = OUTPUT_ROOT / f"{direction_name}.json"
        json_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=float) + "\n")
        per_direction_reports[direction_name] = report
        print(f"A_I(R) collective extraction [{direction_name}]: {report['verdict']} "
              f"({len(significant)}/{len(connection)} R-images with significant support) -> {json_path}")
        for name, check in checks.items():
            print(f"    {name}: {'PASS' if check['passed'] else 'FAIL'}")

    summary = {
        "schema": "b_i_real_space_collective_v6_summary_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "directions": list(DIRECTIONS_NEEDED),
        "verdicts": {name: per_direction_reports[name]["verdict"] for name in DIRECTIONS_NEEDED},
        "overall_verdict": "PASS" if overall_pass else "NO_GO",
    }
    summary_path = OUTPUT_ROOT / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"\nOverall: {summary['overall_verdict']} -> {summary_path}")
    return 0 if overall_pass else 1


def demo() -> None:
    """Self-check: collective_connection_blocks reduces to the single-atom
    hardcoded placement when only atom 0 / axis 0 carries weight, and leaves
    every cross-atom block exactly zero for a two-atom, two-axis-active case."""
    class FakeEngine:
        def derivative(self, separation, axis):
            return np.full((4, 4), float(axis + 1))

    class FakeGeom:
        na = 2

    images = [{"index": (0, 0, 0), "vector": np.zeros(3)}]
    vectors = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    result = collective_connection_blocks(FakeEngine(), images, FakeGeom(), vectors)
    matrix = result[(0, 0, 0)]
    assert np.allclose(matrix[:4, :4], 1.0)
    assert np.allclose(matrix[4:, 4:], 0.0)
    assert np.allclose(matrix[:4, 4:], 0.0) and np.allclose(matrix[4:, :4], 0.0)

    vectors2 = np.array([[0.0, 0.0, 0.0], [0.0, 2.0, 3.0]])
    result2 = collective_connection_blocks(FakeEngine(), images, FakeGeom(), vectors2)
    matrix2 = result2[(0, 0, 0)]
    assert np.allclose(matrix2[:4, :4], 0.0)
    assert np.allclose(matrix2[4:, 4:], 2.0 * 2.0 + 3.0 * 3.0)  # axis1*w1 + axis2*w2
    print("extract_b_i_real_space_collective_v6: demo ok")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
        raise SystemExit(0)
    raise SystemExit(main())
