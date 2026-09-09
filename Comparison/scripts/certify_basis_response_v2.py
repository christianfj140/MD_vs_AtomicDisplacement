#!/usr/bin/env python3
"""C14C-v2: assemble the inter-atomic dS/dR provider with the certified
intra-atomic B_I_v5 term and re-run C14C's own identities.

C14C-v1 (certify_basis_response.py) stays untouched and historical -- its
NO_GO is correct for what it measures (dS/dR structurally cannot see the
monocentric/self-image intra-atomic term, method=dhsdr_atom_resolved_split__
inter_atomic_only__intra_atomic_omitted). This script builds a SEPARATE,
versioned artifact:

  S_R_full(R) = S_R_inter(R) + A_I(R)
  S_L_full(R) = S_L_inter(R) + A_I(R)^dagger = S_L_inter(R) - A_I(R)

using A_I(R) extracted in extract_b_i_real_space_v5.py (verified: exact
antihermiticity A_I(R) + A_I(-R)^dagger = 0, exact match to the original
Bloch-summed k-space B_I_v5 artifact at all three K_POINTS) and S_L_inter/
S_R_inter from response_for_direction() -- the SAME function, SAME GO-2
campaign C14C-v1 uses, not reimplemented.

Checks run, in order, before any PASS is emitted:
  1. contract/provenance (documented, not re-derived here -- see module notes)
  2. A_I(R) + A_I(-R)^dagger ~ 0            -- verified in extract_b_i_real_space_v5.py
  3. A_I(R) has support only in the I-I block -- guaranteed by construction (masked)
  4. D_u S ~ S_L_full + S_R_full              -- checked below
  5. S_R_full(Gamma) vs certify_production_basis_connection's independent,
     already-certified full (intra-atomic-included) connection
  6. Delta_full - Delta_inter ~ A_I S^-1 K0 - K0 S^-1 A_I  (pure algebra, no new
     finite-difference data)
  7. re-run C14C-v1's own electronic-state / gauge / covariance identities via
     epc_basis_response.validate_basis_response, unchanged tolerances

If any prerequisite is missing or a check fails, this reports BLOCKED/NO_GO
with the specific reason -- it does not loosen anything to force a PASS.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO_ROOT / "Comparison/scripts"), str(REPO_ROOT / "shared")]

import sisl  # noqa: E402

import certify_siesta_dhsdr as go2  # noqa: E402
import epc_basis_response as ebr  # noqa: E402
from artifact_signature import file_sha256  # noqa: E402
from certify_basis_response import (  # noqa: E402
    DEFAULT_DEGENERACY_TOL_EV, DEFAULT_GO2_REPORT, DEFAULT_REFERENCE_ROOT,
    DEFAULT_STATE_K, dense_blocks, electronic_states, load_go2_report,
    response_for_direction,
)
from certify_delta_plateau_topology import _k_matrix  # noqa: E402
from certify_full_basis_connection_semantic_v5 import solve  # noqa: E402
from certify_production_basis_connection import _connections  # noqa: E402
from cross_basis_projection_preflight import _basis_geometry  # noqa: E402
from nested_basis_identity_preflight import PROD_CENTRAL  # noqa: E402
from run_nested_basis_sentinel import _matrix_data  # noqa: E402

OUTPUT = REPO_ROOT / "Comparison/results/epc/basis_response_v2/graphene"
A_I_REAL_SPACE_PATH = REPO_ROOT / "Comparison/results/epc/b_i_real_space_v5/b_i_real_space_v5.json"
A_I_REAL_SPACE_NPZ = REPO_ROOT / "Comparison/results/epc/b_i_real_space_v5/b_i_real_space_v5.npz"
DIRECTION_NAME = "atom0000_x"
METHOD_V2 = "dhsdr_atom_resolved_split__inter_atomic_plus_BI_v5_intra_atomic"
GAMMA = (0.0, 0.0, 0.0)
DS_IDENTITY_TOL = 1e-6
CROSS_CHECK_RMS_TOL_EV_PER_ANG = 5e-3
ALGEBRAIC_CORRECTION_TOL_EV_PER_ANG = 1e-3


def _load_a_i_real_space() -> dict[tuple[int, int, int], np.ndarray]:
    a_i_report = json.loads(A_I_REAL_SPACE_PATH.read_text())
    if a_i_report["verdict"] != "PASS":
        raise RuntimeError(f"A_I real-space extraction is {a_i_report['verdict']!r}, not PASS")
    npz = np.load(A_I_REAL_SPACE_NPZ, allow_pickle=True)
    r_list = npz["R_list"]
    return {tuple(int(v) for v in r): npz[f"R_{r[0]}_{r[1]}_{r[2]}"] for r in r_list}


def main() -> int:
    a_i_by_r = _load_a_i_real_space()

    campaign = go2.load_campaign(DEFAULT_REFERENCE_ROOT)
    report = load_go2_report(DEFAULT_GO2_REPORT)
    vectors = campaign.directions[DIRECTION_NAME]
    artifact_inter, inputs = response_for_direction(campaign, report, DIRECTION_NAME, vectors)

    isc_off = np.asarray(artifact_inter.isc_off)
    isc_index = {tuple(int(v) for v in row): i for i, row in enumerate(isc_off)}

    missing = [r for r in a_i_by_r if r not in isc_index]
    if missing:
        report_out = {
            "schema": "basis_response_v2_graphene_v1", "gate": "C14C_basis_response_v2",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "verdict": "BLOCKED",
            "reason": f"{len(missing)} A_I(R) image(s) have no matching row in from_dhsdr's "
                      f"isc_off table: {missing[:5]}... -- cannot assemble without inventing R rows",
        }
        OUTPUT.mkdir(parents=True, exist_ok=True)
        path = OUTPUT / "basis_response_v2_certification.json"
        path.write_text(json.dumps(report_out, indent=2, sort_keys=True) + "\n")
        print(f"C14C_basis_response_v2: BLOCKED -> {path}")
        return 1

    s_left_full = artifact_inter.blocks["S_left"].copy()
    s_right_full = artifact_inter.blocks["S_right"].copy()
    # S_left(R) relates to S_right(-R), not S_right(R): verified empirically
    # (||S_L(R)-S_R(-R)^H|| ~ 1e-15 vs ||S_L(R)-S_R(R)^H|| ~ O(1) on the real
    # inter-atomic artifact). Preserving S_left_full(R) = S_right_full(-R)^H
    # given S_right_full(R) = S_right_inter(R) + A_I(R) requires
    # S_left_full(R) = S_left_inter(R) + A_I(-R)^H = S_left_inter(R) - A_I(R)
    # (using the verified A_I(-R)^H = -A_I(R)), NOT S_left_inter(R) + A_I(R)^H
    # at the same R -- those coincide only at R=(0,0,0).
    for r, a_i in a_i_by_r.items():
        idx = isc_index[r]
        s_right_full[idx] += a_i
        s_left_full[idx] -= a_i

    new_context = ebr.ResponseContext(
        geometry_signature=artifact_inter.context.geometry_signature,
        basis_signature=artifact_inter.context.basis_signature,
        electronic_derivative_backend=artifact_inter.context.electronic_derivative_backend,
        direction=artifact_inter.context.direction,
        q=artifact_inter.context.q,
        q_aware_source=artifact_inter.context.q_aware_source,
        source={**dict(artifact_inter.context.source),
                "intra_atomic_source": "certify_b_i_fourier_bessel_semantic_v5.py via "
                                        "extract_b_i_real_space_v5.py",
                "intra_atomic_artifact_sha256": file_sha256(A_I_REAL_SPACE_NPZ)},
        formalism_id=artifact_inter.context.formalism_id,
    )
    artifact_full = ebr.from_S_L_S_R(
        s_left_full, s_right_full, new_context,
        backend=artifact_inter.backend, method=METHOD_V2, intra_atomic_included=True,
        isc_off=isc_off,
        diagnostics={**dict(artifact_inter.diagnostics),
                     "intra_atomic_term": "included via B_I_v5 (real-space, all R with support)",
                     "n_R_with_A_I": len(a_i_by_r)},
    )

    checks: dict[str, dict] = {}

    # -- check 4: D_u S ~ S_L_full + S_R_full --------------------------------
    direction = inputs["direction"]
    dhsdr = inputs["dhsdr"]
    pair = inputs["pair"]
    fd = go2.finite_difference(
        go2.read_tshs(campaign.run_dir(pair["plus_run_id"]), pair["plus_run_id"],
                      fermi_ev=campaign.fermi_ev(pair["plus_run_id"])),
        go2.read_tshs(campaign.run_dir(pair["minus_run_id"]), pair["minus_run_id"],
                      fermi_ev=campaign.fermi_ev(pair["minus_run_id"])),
        float(inputs["delta_ang"]),
    )
    reference_d_s, _outside = dense_blocks(fd["D_S"], isc_off, artifact_full.no_u)
    residual_d_s = reference_d_s - (s_left_full + s_right_full)
    checks["D_uS_identity"] = {
        "max_abs_residual_inv_ang": float(np.abs(residual_d_s).max()),
        "tolerance_inv_ang": DS_IDENTITY_TOL,
        "passed": float(np.abs(residual_d_s).max()) < DS_IDENTITY_TOL,
        "note": "D_uS must be UNCHANGED by adding A_I -- A_I(R)+A_I(-R)^dagger=0 makes the "
                "S_L_full+S_R_full sum identical to S_L_inter+S_R_inter at every k",
    }

    # -- check 5: cross-check against certify_production_basis_connection ----
    central_label, central_h, central_fermi, central_vacuum, atom = _matrix_data(PROD_CENTRAL)
    shape = tuple(int(x) for x in sisl.get_sile(str(PROD_CENTRAL / f"{central_label}.VT")).read_grid().shape)
    base = _basis_geometry(central_h.geometry, PROD_CENTRAL / "C.ion.xml")
    dvolume = abs(np.linalg.det(base.cell)) / np.prod(shape)
    s0 = np.asarray(central_h.Sk(k=GAMMA, format="array"))
    k0 = _k_matrix(central_h, central_fermi, central_vacuum, GAMMA)
    connections_canonical = _connections("C1_x", 0.005, base, atom, shape, dvolume, [GAMMA])
    _s_l_canonical, s_r_canonical = connections_canonical[0]
    s_r_full_gamma = ebr.bloch_sum(s_right_full, isc_off, GAMMA)
    cross_check_diff = s_r_full_gamma - s_r_canonical
    cross_check_rms = float(np.linalg.norm(cross_check_diff) / np.sqrt(cross_check_diff.size))
    checks["cross_check_vs_production_basis_connection"] = {
        "rms_ev_equivalent": cross_check_rms,  # S_R units are 1/Ang; reported as-is
        "rms_inv_ang": cross_check_rms,
        "tolerance_inv_ang": CROSS_CHECK_RMS_TOL_EV_PER_ANG,
        "passed": cross_check_rms < CROSS_CHECK_RMS_TOL_EV_PER_ANG,
        "note": "certify_production_basis_connection.py builds S_R independently via real-space "
                "grid quadrature of the full geometry (intra-atomic term included by construction, "
                "not assembled from dS/dR+B_I) -- an independent witness for this assembly",
    }

    # -- check 6: algebraic correction Delta_full - Delta_inter --------------
    left_term, _ = solve(s0, k0)
    a_i_gamma = sum(block * np.exp(2j * np.pi * np.dot(GAMMA, r)) for r, block in a_i_by_r.items())
    predicted_correction = a_i_gamma @ left_term - k0 @ solve(s0, a_i_gamma)[0]
    s_r_inter_gamma = ebr.bloch_sum(artifact_inter.blocks["S_right"], isc_off, GAMMA)
    s_l_inter_gamma = ebr.bloch_sum(artifact_inter.blocks["S_left"], isc_off, GAMMA)
    delta_inter = -(s_l_inter_gamma @ left_term + k0 @ solve(s0, s_r_inter_gamma)[0])
    s_l_full_gamma = ebr.bloch_sum(s_left_full, isc_off, GAMMA)
    delta_full = -(s_l_full_gamma @ left_term + k0 @ solve(s0, s_r_full_gamma)[0])
    measured_correction = delta_full - delta_inter
    correction_residual = measured_correction - predicted_correction
    correction_rms = float(np.linalg.norm(correction_residual) / np.sqrt(correction_residual.size))
    checks["algebraic_correction_identity"] = {
        "formula": "Delta_full - Delta_inter == A_I(Gamma) S0^-1 K0 - K0 S0^-1 A_I(Gamma)",
        "rms_residual_ev_per_ang": correction_rms,
        "tolerance_ev_per_ang": ALGEBRAIC_CORRECTION_TOL_EV_PER_ANG,
        "passed": correction_rms < ALGEBRAIC_CORRECTION_TOL_EV_PER_ANG,
    }

    # -- check 7: re-run C14C-v1's own identities on the assembled artifact --
    states, diagnostics = electronic_states(
        inputs["equilibrium"], go2.contract_fc(dhsdr, direction.vectors, "D_H"),
        isc_off, DEFAULT_STATE_K, DEFAULT_DEGENERACY_TOL_EV,
    )
    validation = ebr.validate_basis_response(
        artifact_full, D_S_reference=reference_d_s,
        reference_name=f"central_fd_of_S({pair['plus_run_id']}, {pair['minus_run_id']})",
        tau_reference=inputs["terms"]["tau_fd_frobenius"], states=states,
        atom_of_orbital=inputs["atom_of_orbital"], degeneracy_tol_ev=DEFAULT_DEGENERACY_TOL_EV,
    )

    all_new_checks_passed = all(c["passed"] for c in checks.values())
    verdict = "PASS" if (all_new_checks_passed and validation["verdict"] == "PASS") else "NO_GO"

    report_out = {
        "schema": "basis_response_v2_graphene_v1",
        "gate": "C14C_basis_response_v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "direction": DIRECTION_NAME,
        "method": METHOD_V2,
        "provenance_note": (
            "GO-2 reference campaign (siesta_reference/graphene) and PROD-SZ (B_I_v5 source) "
            "share identical C.ion.xml/C.psf hash, geometry, MeshCutoff (600 Ry), k-grid "
            "(20x20x1). DM.Tolerance differs (GO-2: 1e-4, PROD-SZ: 1e-6) but S/dS/dR do not "
            "depend on the converged SCF density, only on geometry+basis, so this does not "
            "affect the assembly. A_I is delta-independent (an analytic monocentric+self-image "
            "quadrature, not a finite difference), so no delta-matching was needed either."
        ),
        "n_R_images_with_A_I": len(a_i_by_r),
        "checks": checks,
        "c14c_v1_identities_rerun": {
            "verdict": validation["verdict"],
            "checks": validation.get("checks"),
        },
        "verdict": verdict,
        "historical_c14c_v1": "NO_GO (unchanged, dhsdr-only method, correct for its own scope)",
        "unaffected_by_this_gate": {
            "checkpoint_lineage": "NO_GO", "fine_tuning_candidate": "BLOCKED",
            "delta_out_closure": "NO_GO", "full_KS": "BLOCKED",
        },
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "basis_response_v2_certification.json"
    path.write_text(json.dumps(report_out, indent=2, sort_keys=True, default=float) + "\n")
    print(f"C14C_basis_response_v2: {verdict} -> {path}")
    for name, check in checks.items():
        print(f"  {name}: {'PASS' if check['passed'] else 'FAIL'}")
    print(f"  c14c_v1_identities_rerun: {validation['verdict']}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
