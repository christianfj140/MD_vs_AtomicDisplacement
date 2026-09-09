#!/usr/bin/env python3
"""C14C / E-F_001-S17: run the basis-response gate on the certified SIESTA reference.

S16 built the provider; this script points it at the only reference the
repository owns that can judge it — the graphene campaign GO-2 already certified
(``certify_siesta_dhsdr.py``) — and writes the verdict. It runs no SCF and reads
no fdf: it consumes the reference manifest, the FC ``*.dHSdR.nc`` and the ``+-``
TSHS pairs that were already produced, and it refuses to run at all if GO-2 did
not pass, because a basis response validated against an uncertified derivative
certifies nothing.

For each direction of the perturbation space:

``S_L, S_R``   assembled by :func:`epc_basis_response.from_dhsdr` from the FC
               file at the largest amplitude GO-2 actually *resolved* for
               ``D_S`` (its ``deltas_within_tau``) — not the plateau's, which is
               flat in the signal and, for a collective direction, still carries
               the ``O(delta^2)`` cross term of the central difference above the
               reference's own floor. Charging that truncation to the provider
               would be an attribution error.
``D_S`` reference
               a central difference of the *overlap* matrices of the ``+-`` TSHS
               pair — an independent measurement of the same object, with GO-2's
               measured ``tau_FD`` as its noise floor. The overlap needs no Fermi
               correction (that one is a property of ``H``, not ``S``), so this
               side is free of the convention that dominates the ``D_H`` gate.
``electrons``  ``H`` and ``S`` of the equilibrium run at Gamma and at K, solved
               as the generalized pencil, with ``H`` restored to absolute
               energies so that ``eps`` and ``dHSdR``'s unshifted ``dH/dR`` share
               one energy origin. K is included because it is where graphene has
               the degeneracy the gauge-covariance check needs.

What comes out is :func:`epc_basis_response.validate_basis_response`'s report per
direction: the identities, the size of the basis response split into diagonal and
interband, what each approximation drops, and the terms still unresolved. The
expected verdict today is ``NO_GO``: ``dS/dR`` cannot see the intra-atomic term
(memo assumption A2) and nothing in this repository computes it yet. The exit
code therefore tracks the *checks*, not the verdict — a failing check means the
provider is wrong, while a NO_GO with all checks passing means the provider is
right and incomplete.

Units: ``1/Ang`` for the response, ``eV`` for energies, ``eV/Ang`` for ``D_H``
and ``g``, ``Ang`` for amplitudes.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for path in (SCRIPT_DIR, REPO_ROOT / "shared"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import certify_siesta_dhsdr as go2  # noqa: E402
import epc_basis_response as ebr  # noqa: E402
from read_siesta_dhsdr import read_dhsdr  # noqa: E402

SCHEMA = "epc_basis_response_certification_v1"

DEFAULT_REFERENCE_ROOT = REPO_ROOT / "Comparison/results/epc/siesta_reference/graphene"
DEFAULT_GO2_REPORT = REPO_ROOT / "Comparison/results/epc/certification/graphene/go2_certification.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "Comparison/results/epc/basis_response/graphene"

# Gamma and the graphene K corner: K carries the Dirac degeneracy, which is what
# gives the gauge-covariance check something to rotate. (2/3, 1/3, 0) is K for
# the 60-degree hexagonal primitive cell of materials/graphene; on any other
# lattice the label is wrong and the check simply reports no degeneracy, which
# is a missing test, never a false claim.
DEFAULT_STATE_K = {"gamma": (0.0, 0.0, 0.0), "K": (2.0 / 3.0, 1.0 / 3.0, 0.0)}

# Two Kohn-Sham states closer than this are one subspace. Exact degeneracy is a
# symmetry statement; a real SCF spectrum only realises it to the level its grid
# breaks the symmetry at, which for this campaign is ~1e-4 eV. The covariance
# check carries that spread in its own tolerance, so this is a clustering scale,
# not a slackened gate.
DEFAULT_DEGENERACY_TOL_EV = 1e-3


class BasisResponseCertificationError(RuntimeError):
    """The gate cannot be evaluated at all (missing, unusable or uncertified inputs)."""


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #


def load_go2_report(path: Path) -> dict[str, Any]:
    """The GO-2 certification, which must have passed before anything here runs."""
    if not path.is_file():
        raise BasisResponseCertificationError(
            f"{path} is missing. Run certify_siesta_dhsdr.py first: the basis response is only "
            "as good as the derivative it is split out of."
        )
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("verdict") != "PASS":
        raise BasisResponseCertificationError(
            f"{path} reports GO-2 = {report.get('verdict')!r}. A basis response validated against "
            "an uncertified dS/dR would certify nothing."
        )
    return report


def d_s_reference_terms(report: Mapping[str, Any], direction_name: str) -> dict[str, Any]:
    """GO-2's own plateau amplitude and noise floor for ``D_S`` in one direction."""
    for entry in report.get("comparisons", []):
        if entry.get("direction_name") == direction_name and entry.get("kind") == "D_S":
            plateau = entry.get("plateau") or {}
            frobenius = (entry.get("by_norm") or {}).get("frobenius") or {}
            return {
                "go2_passed": bool(entry.get("passed")),
                "go2_verdict": frobenius.get("verdict"),
                "is_null_direction": bool(plateau.get("is_null_direction")),
                "plateau_selected_delta_ang": plateau.get("selected_delta_ang"),
                "deltas_within_tau": [float(value) for value in frobenius.get("deltas_within_tau") or []],
                "discrepancy_by_delta": frobenius.get("discrepancy_by_delta"),
                "tau_fd_frobenius": float((entry.get("tau_fd") or {}).get("frobenius", 0.0)),
                "unit": entry.get("unit"),
            }
    raise BasisResponseCertificationError(
        f"GO-2 has no D_S comparison for direction {direction_name!r}."
    )


def dense_blocks(
    field: Mapping[tuple[int, int, tuple[int, int, int]], float],
    isc_off: np.ndarray,
    no_u: int,
) -> tuple[np.ndarray, float]:
    """A sparse ``(row, col, R)`` field as ``(n_s, no_u, no_u)`` on a given image table.

    Elements whose image is absent from the table are not dropped in silence:
    their norm is returned, so a reference that lives partly outside the
    response's supercell is visible instead of being quietly truncated.
    """
    index = {tuple(int(value) for value in row): position for position, row in enumerate(isc_off)}
    blocks = np.zeros((isc_off.shape[0], no_u, no_u), dtype=np.float64)
    outside = 0.0
    for (row, col, image), value in field.items():
        position = index.get(tuple(int(component) for component in image))
        if position is None:
            outside += float(value) ** 2
            continue
        blocks[position, int(row), int(col)] = float(value)
    return blocks, math.sqrt(outside)


def solve_pencil(h: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``H C = S C eps`` with ``C^dag S C = I``, through a Cholesky factor of ``S``."""
    inverse = np.linalg.inv(np.linalg.cholesky(s))
    eps, vectors = np.linalg.eigh(inverse @ h @ inverse.conj().T)
    return eps, inverse.conj().T @ vectors


def electronic_states(
    equilibrium: Any,
    d_h_field: Mapping[tuple[int, int, tuple[int, int, int]], float],
    isc_off: np.ndarray,
    k_points: Mapping[str, Sequence[float]],
    degeneracy_tol_ev: float,
) -> tuple[list[ebr.ElectronicState], list[dict[str, Any]]]:
    """One :class:`ElectronicState` per ``k``, plus the orthonormality diagnostics.

    ``H`` is the absolute Hamiltonian (sisl's Fermi shift undone), which is the
    energy origin ``dHSdR.nc`` differentiates; ``eps`` therefore belongs to the
    same convention as ``D_H`` and the contraction is consistent.
    """
    no_u = equilibrium.no_u
    h_blocks, _ = dense_blocks(equilibrium.h_absolute, isc_off, no_u)
    s_blocks, _ = dense_blocks(equilibrium.overlap, isc_off, no_u)
    d_h_blocks, outside = dense_blocks(d_h_field, isc_off, no_u)

    states: list[ebr.ElectronicState] = []
    diagnostics: list[dict[str, Any]] = []
    for label, k in k_points.items():
        h = ebr.bloch_sum(h_blocks, isc_off, k)
        s = ebr.bloch_sum(s_blocks, isc_off, k)
        eps, c = solve_pencil(h, s)
        states.append(
            ebr.ElectronicState(
                label=label, k=tuple(float(value) for value in k),
                D_H=ebr.bloch_sum(d_h_blocks, isc_off, k), C=c, eps=eps,
            )
        )
        diagnostics.append(
            {
                "state": label,
                "k": [float(value) for value in k],
                "eps_ev": [float(value) for value in eps],
                "S_orthonormality_residual": float(
                    np.linalg.norm(c.conj().T @ s @ c - np.eye(no_u))
                ),
                "eigen_residual": float(np.linalg.norm(h @ c - s @ c @ np.diag(eps))),
                "degenerate_groups": ebr.degenerate_groups(eps, degeneracy_tol_ev),
                "energy_origin": "absolute H (sisl E_F shift undone), as in dHSdR.nc",
            }
        )
    if outside > 0.0:
        diagnostics.append({"warning": "D_H carries elements outside the image table", "norm": outside})
    return states, diagnostics


# --------------------------------------------------------------------------- #
# One direction
# --------------------------------------------------------------------------- #


def response_for_direction(
    campaign: Any,
    report: Mapping[str, Any],
    direction_name: str,
    vectors: np.ndarray,
    *,
    geometry_signature: str | None = None,
    basis_signature: str | None = None,
    electronic_derivative_backend: str = "siesta",
) -> tuple[Any, dict[str, Any]]:
    """The ``S_L``/``S_R`` artifact of one direction, at the amplitude GO-2 resolved.

    Split out of :func:`certify_direction` so that C20 contracts the *same*
    response object it validated instead of assembling a second one. The two
    signature arguments exist because a consumer that also carries phonons must
    declare the geometry under the scheme those phonons were signed with; the
    defaults keep the C14C artifact byte-identical.
    """
    import fd_perturbation_space as fdp  # scipy lands here, not in the provider

    terms = d_s_reference_terms(report, direction_name)
    # The largest amplitude at which GO-2 actually resolved D_S^FC against
    # D_S^FD. Not the plateau's: the plateau is flat in the *signal*, and a
    # collective central difference still carries its O(delta^2) cross term on
    # top of a contraction of one-atom derivatives, which at the largest
    # amplitude exceeds the reference's own floor (GO-2 certifies it as
    # truncation, not as a defect -- fitted exponent 2). Building the response
    # there would charge that truncation to the provider.
    within_tau = terms["deltas_within_tau"]
    delta, delta_source = (max(within_tau), "go2_largest_delta_within_tau") if within_tau else (
        terms["plateau_selected_delta_ang"] or max(campaign.deltas),
        "go2_plateau" if terms["plateau_selected_delta_ang"] else "largest_certified_amplitude",
    )

    pair = next(
        (
            entry
            for entry in campaign.certified_pairs()
            if entry["direction_name"] == direction_name
            and math.isclose(float(entry["delta_ang"]), float(delta))
        ),
        None,
    )
    if pair is None:
        raise BasisResponseCertificationError(
            f"No certified +- pair for {direction_name!r} at delta = {delta}."
        )

    direction = fdp.Direction(
        name=direction_name,
        kind=next(
            entry["direction_kind"]
            for entry in campaign.manifest["perturbation_space"]["directions"]
            if entry["direction_name"] == direction_name
        ),
        vectors=vectors,
    )

    dhsdr = read_dhsdr(campaign.dhsdr_path(delta))
    equilibrium = go2.read_tshs(
        campaign.run_dir(campaign.manifest["certification_set"]["equilibrium_run_id"]),
        campaign.manifest["certification_set"]["equilibrium_run_id"],
        fermi_ev=campaign.fermi_ev(campaign.manifest["certification_set"]["equilibrium_run_id"]),
    )

    import sisl  # only for the orbital -> atom map

    geometry = sisl.get_sile(equilibrium.path).read_geometry()
    atom_of_orbital = np.asarray(geometry.o2a(np.arange(equilibrium.no_u)), dtype=np.int64) + 1

    context = ebr.ResponseContext(
        geometry_signature=str(
            geometry_signature
            if geometry_signature is not None
            else campaign.manifest.get("material_fdf_sha256", "")
        ),
        basis_signature=str(
            basis_signature
            if basis_signature is not None
            else (campaign.manifest.get("orbital_contract") or {}).get(
                "orbital_contract_hash", "unknown"
            )
        ),
        electronic_derivative_backend=electronic_derivative_backend,
        direction=direction,
        source={
            "reference_root": str(campaign.root),
            "fc_run": str(campaign.dhsdr_path(delta)),
            "delta_ang": float(delta),
            "delta_source": delta_source,
            "siesta_runtime_ref": campaign.manifest.get("siesta_runtime_ref"),
        },
    )
    artifact = ebr.from_dhsdr(dhsdr, context, atom_of_orbital=atom_of_orbital)
    inputs = {
        "direction": direction,
        "delta_ang": float(delta),
        "delta_source": delta_source,
        "pair": pair,
        "dhsdr": dhsdr,
        "equilibrium": equilibrium,
        "atom_of_orbital": atom_of_orbital,
        "terms": terms,
    }
    return artifact, inputs


def certify_direction(
    campaign: Any,
    report: Mapping[str, Any],
    direction_name: str,
    vectors: np.ndarray,
    *,
    k_points: Mapping[str, Sequence[float]] = DEFAULT_STATE_K,
    degeneracy_tol_ev: float = DEFAULT_DEGENERACY_TOL_EV,
) -> dict[str, Any]:
    """Build the response for one direction and run the C14C gate on it."""
    artifact, inputs = response_for_direction(campaign, report, direction_name, vectors)
    direction = inputs["direction"]
    delta, delta_source = inputs["delta_ang"], inputs["delta_source"]
    pair, dhsdr = inputs["pair"], inputs["dhsdr"]
    equilibrium, atom_of_orbital = inputs["equilibrium"], inputs["atom_of_orbital"]
    terms = inputs["terms"]

    fd = go2.finite_difference(
        go2.read_tshs(campaign.run_dir(pair["plus_run_id"]), pair["plus_run_id"], fermi_ev=campaign.fermi_ev(pair["plus_run_id"])),
        go2.read_tshs(campaign.run_dir(pair["minus_run_id"]), pair["minus_run_id"], fermi_ev=campaign.fermi_ev(pair["minus_run_id"])),
        float(delta),
    )
    reference, outside = dense_blocks(fd["D_S"], artifact.isc_off, artifact.no_u)

    states, diagnostics = electronic_states(
        equilibrium,
        go2.contract_fc(dhsdr, direction.vectors, "D_H"),
        artifact.isc_off,
        k_points,
        degeneracy_tol_ev,
    )

    validation = ebr.validate_basis_response(
        artifact,
        D_S_reference=reference,
        reference_name=f"central_fd_of_S({pair['plus_run_id']}, {pair['minus_run_id']})",
        tau_reference=terms["tau_fd_frobenius"],
        states=states,
        atom_of_orbital=atom_of_orbital,
        degeneracy_tol_ev=degeneracy_tol_ev,
    )
    return {
        "direction": direction_name,
        "direction_kind": direction.kind,
        "delta_ang": float(delta),
        "delta_source": delta_source,
        "reference": {
            **terms,
            "plus_run_id": pair["plus_run_id"],
            "minus_run_id": pair["minus_run_id"],
            "reference_norm_outside_image_table": outside,
        },
        "electronic_states": diagnostics,
        "validation": validation,
    }


# --------------------------------------------------------------------------- #
# Campaign
# --------------------------------------------------------------------------- #


def certify(
    reference_root: Path,
    go2_report_path: Path,
    *,
    directions: Sequence[str] = (),
    degeneracy_tol_ev: float = DEFAULT_DEGENERACY_TOL_EV,
) -> dict[str, Any]:
    campaign = go2.load_campaign(reference_root)
    report = load_go2_report(go2_report_path)
    wanted = set(directions) or set(campaign.directions)

    rows = []
    for name, vectors in campaign.directions.items():
        if name not in wanted:
            continue
        try:
            rows.append(
                certify_direction(
                    campaign, report, name, vectors, degeneracy_tol_ev=degeneracy_tol_ev
                )
            )
        except (ebr.BasisResponseError, BasisResponseCertificationError) as error:
            rows.append({"direction": name, "blocked": str(error)})

    evaluated = [row for row in rows if "validation" in row]
    failed_checks = [
        {"direction": row["direction"], "check": check["check"], "detail": check["detail"]}
        for row in evaluated
        for check in row["validation"]["checks"]
        if check["applicable"] and not check["passed"]
    ]
    blocked = [row for row in rows if "blocked" in row]
    verdicts = sorted({row["validation"]["verdict"] for row in evaluated})
    return {
        "schema": SCHEMA,
        "ticket": "C14C / E-F_001-S17",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reference_root": str(reference_root),
        "go2_report": str(go2_report_path),
        "go2_verdict": report.get("verdict"),
        "siesta_runtime_ref": campaign.manifest.get("siesta_runtime_ref"),
        "contract": ebr.basis_response_contract(),
        "degeneracy_tol_ev": float(degeneracy_tol_ev),
        "k_points": {label: list(k) for label, k in DEFAULT_STATE_K.items()},
        "directions": rows,
        "summary": {
            "directions_evaluated": len(evaluated),
            "directions_blocked": [row["direction"] for row in blocked],
            "failed_checks": failed_checks,
            "verdicts": verdicts,
        },
        # The provider is wrong only if a check failed; a NO_GO with every check
        # passing is the unresolved intra-atomic term, which is a gate, not a bug.
        "provider_checks_passed": not failed_checks and not blocked,
        "verdict": "PASS" if verdicts == ["PASS"] and not blocked else "NO_GO",
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, Path):
        return str(value)
    return value


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE_ROOT)
    parser.add_argument("--go2-report", type=Path, default=DEFAULT_GO2_REPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--direction", action="append", default=[], help="restrict to these directions")
    parser.add_argument(
        "--degeneracy-tol-ev",
        type=float,
        default=DEFAULT_DEGENERACY_TOL_EV,
        help="states closer than this are treated as one subspace",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    report = certify(
        args.reference_root,
        args.go2_report,
        directions=args.direction,
        degeneracy_tol_ev=args.degeneracy_tol_ev,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "basis_response_certification.json"
    output.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True), encoding="utf-8")

    print(f"C14C basis response: {report['verdict']}  ->  {output}")
    for row in report["directions"]:
        if "blocked" in row:
            print(f"  {row['direction']:<16} BLOCKED  {row['blocked']}")
            continue
        validation = row["validation"]
        contribution = validation["contributions"][0] if validation["contributions"] else {}
        response = contribution.get("basis_response_contribution", {})
        print(
            f"  {row['direction']:<16} {validation['verdict']:<6} delta={row['delta_ang']:g} "
            f"basis response |diag|={response.get('diagonal_frobenius')} "
            f"|interband|={response.get('interband_frobenius')}"
        )
        for reason in validation["reasons"]:
            print(f"      reason: {reason}")
    for failure in report["summary"]["failed_checks"]:
        print(f"  FAILED CHECK {failure['direction']}: {failure['check']}: {failure['detail']}")
    return 0 if report["provider_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
