#!/usr/bin/env python3
"""C09 / E-F_001-S12: GO-2 -- certify ``dHSdR.nc`` against central finite differences.

GO-2 asks whether the numbers SIESTA writes into ``*dHSdR.nc`` are the
derivative of *the same* H and S that the same binary writes into a TSHS. S11
produced both sides from one frozen set of physics inputs
(``run_epc_siesta_reference.py``); this script is the automatic verdict on them.
It runs no SCF and reads no fdf: it consumes the reference manifest and the
artifacts it certified, and exits non-zero when GO-2 fails.

The two sides of the comparison
-------------------------------

``D^FC[v]``  the FC file's own derivatives, contracted onto a direction::

        D^FC[v] = sum_{I,alpha} v[I,alpha] (dH/dR_{I,alpha})

    read through :mod:`read_siesta_dhsdr` (C06), which is what fixes atom/axis
    indexing, the supercell rule and the Ry/Bohr -> eV/Ang factor.

``D^FD[v]``  an explicit central difference of the TSHS matrices::

        D^FD[v] = (M(R + delta v) - M(R - delta v)) / (2 delta)

    over the *same* ``(row, col, R)`` index space.

One correction is mandatory and is the reason a naive comparison fails by 22%:
sisl hands back ``H - E_F S``, ``dHSdR.nc`` stores the unshifted ``dH/dR``, and
E_F itself moves with the displacement. The certified finite difference is
therefore

    D^FD_H = [ (H_sisl(+) + E_F(+) S(+)) - (H_sisl(-) + E_F(-) S(-)) ] / (2 delta)

with each run's own ``E_F`` taken from the reference manifest's row. The
uncorrected variant is computed too, as :func:`fermi_control`: a gate that the
wrong convention would also pass certifies nothing, so the report has to show
that it does not.

What "compatible" is made to mean
---------------------------------

``tau_FD`` is measured, never assumed. For each (direction, kind) the plateau is
the run of amplitudes over which ``||D^FD(delta)||_F`` is flat
(:func:`fd_perturbation_space.select_delta_from_plateau`), and

    tau_FD = max over plateau pairs (i, j) of || D^FD(delta_i) - D^FD(delta_j) ||

in the same norm the discrepancy is measured in -- the residual variation of the
finite difference itself, i.e. what this method can resolve at all. A
``|D^FC - D^FD|`` smaller than that is indistinguishable from the noise floor.

Two discrepancies are legitimately *not* noise and are classified rather than
excused:

``truncation_limited``
    A collective direction moves several atoms at once, so its central
    difference carries third-derivative cross terms that the sum of one-atom FC
    derivatives does not. The difference is O(delta^2) -- for D_S the measured
    exponent is 2.00 -- and vanishes as delta -> 0, so it cannot be an index,
    phase or unit defect. Accepted only when the fitted exponent is near 2.

``null_direction``
    A uniform translation has no derivative to reproduce: D_H and D_S both
    vanish by translational invariance. Its ``||D^FD||`` sits *below* the noise
    floor of the directions that do carry signal, so no plateau exists and the
    relative metrics are meaningless. It is certified against an absolute
    floor instead -- an acoustic sum rule, not a ratio.

Anything else fails. That is the discriminating power the gate needs: a wrong
unit is a fixed ratio (0.53, 1.89, 13.6) independent of delta, a wrong index or
phase is an O(1) error, and neither shrinks as delta^2 nor hides under a noise
floor two orders of magnitude smaller than the signal.

Structural checks run unconditionally and are not tolerance-based: the R/image
sets of ``dHSdR.nc`` and the TSHS must coincide *as sets* (their orderings do
not, so index-order alignment would be silently wrong), real-space hermiticity
``D(mu, nu, R) = D(nu, mu, -R)`` must hold, and the best-fit scale between the
two sides must be 1.

Sparsification is compared on/off from two campaigns that differ only in
``FC.dHdR.Tolerance`` / ``FC.dSdR.Tolerance``. The release's documented
asymmetry is checked numerically rather than believed: dH is unfiltered, dS is a
pure threshold that drops elements without perturbing the survivors.

Provenance: the runtime is ``VERIFIED_BINARY_ONLY`` (C01), which is exactly the
case in which Fase 2 makes this certification mandatory. The classification is
propagated into the report, so no downstream artifact can lose it.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for _path in (SCRIPT_DIR, REPO_ROOT / "shared"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import fd_perturbation_space as fdp  # noqa: E402
from artifact_signature import file_sha256, input_signature_sha256  # noqa: E402
from read_siesta_dhsdr import (  # noqa: E402
    RY_TO_EV,
    SIGN_CONVENTION,
    SUPERCELL_RULE,
    DhsdrSchemaError,
    read_dhsdr,
)
from run_inventory import validate_siesta_runtime_ref  # noqa: E402

from fdf_materialization import BOHR_TO_ANG  # noqa: E402  (isort: skip)

SCHEMA = "epc_siesta_derivative_certification_v1"
REFERENCE_SCHEMA = "epc_siesta_reference_v1"

KINDS = ("D_H", "D_S")
CANONICAL_UNITS = {"D_H": "eV/Ang", "D_S": "1/Ang"}

DEFAULT_REFERENCE_ROOT = REPO_ROOT / "Comparison/results/epc/siesta_reference/graphene"
DEFAULT_SPARSIFIED_ROOT = REPO_ROOT / "Comparison/results/epc/siesta_reference/graphene_sparsified"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "Comparison/results/epc/certification/graphene"

# A collective central difference and a contraction of one-atom derivatives
# differ at O(delta^2). Accept that classification only when the measured
# exponent really is quadratic; a unit or index defect has exponent 0.
TRUNCATION_EXPONENT_RANGE = (1.5, 2.5)
# Best-fit scale between the two sides. The wrong-unit alternatives are 0.529
# (Ry/Bohr read as Ry/Ang), 1.889 and 13.606, so a 1% window separates them by
# more than an order of magnitude.
UNIT_SCALE_TOLERANCE = 0.01
# The Fermi control must fail by a wide margin, not marginally.
FERMI_CONTROL_MIN_RATIO = 10.0

NORMS = ("max_abs", "frobenius")

Key = tuple[int, int, tuple[int, int, int]]


class CertificationError(RuntimeError):
    """The certification cannot be evaluated at all (missing or unusable inputs)."""


# --------------------------------------------------------------------------- #
# Sparse fields on a canonical (row, col, R) index space
# --------------------------------------------------------------------------- #


def align(*fields: Mapping[Key, float]) -> tuple[list[Key], list[np.ndarray]]:
    """Put several sparse fields on their shared, sorted index space.

    Absent keys are exact zeros: a sparsified field and a dense one are then
    comparable elementwise without either side inventing support.
    """
    keys = sorted(set().union(*(set(field) for field in fields))) if fields else []
    return keys, [np.array([field.get(key, 0.0) for key in keys], dtype=np.float64) for field in fields]


def norms(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"max_abs": 0.0, "frobenius": 0.0}
    return {"max_abs": float(np.abs(values).max()), "frobenius": float(np.linalg.norm(values))}


def difference_norms(left: Mapping[Key, float], right: Mapping[Key, float]) -> dict[str, float]:
    _, (a, b) = align(left, right)
    return norms(a - b)


def best_fit_scale(reference: Mapping[Key, float], candidate: Mapping[Key, float]) -> float | None:
    """``alpha`` minimising ``||reference - alpha * candidate||``.

    This is the numerical unit certification: the attribute in the NetCDF file
    says nothing (5.4.2 labels a Ry/Bohr derivative "Ry" and gives dS no unit at
    all), but the scale that maps the file onto an independent finite difference
    of the same matrices is measured, not read.
    """
    _, (a, b) = align(reference, candidate)
    denominator = float(b @ b)
    if denominator <= 0.0:
        return None
    return float(a @ b) / denominator


def hermiticity_residual(field: Mapping[Key, float]) -> dict[str, Any]:
    """``D(mu, nu, R) - D(nu, mu, -R)``, the real-space hermiticity of a derivative.

    H and S are hermitian in real space and differentiation does not change
    that, so the transpose partner of every stored element must exist and carry
    the same value.
    """
    residual = 0.0
    missing = 0
    for (row, col, isc), value in field.items():
        partner = (col, row, (-isc[0], -isc[1], -isc[2]))
        if partner in field:
            residual = max(residual, abs(value - field[partner]))
        else:
            missing += 1
    return {"max_abs_residual": residual, "missing_transpose_partners": missing}


# --------------------------------------------------------------------------- #
# The two sides
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TshsMatrices:
    """H and S of one run, on the canonical index space, plus its Fermi level."""

    run_id: str
    path: str
    h_shifted: dict[Key, float]  # what sisl returns: H - E_F S
    overlap: dict[Key, float]
    fermi_ev: float
    no_u: int
    nsc: tuple[int, int, int]
    sc_off: tuple[tuple[int, int, int], ...]

    @property
    def h_absolute(self) -> dict[Key, float]:
        """``H``, with sisl's Fermi shift undone -- what ``dHSdR.nc`` differentiates."""
        return {key: value + self.fermi_ev * self.overlap.get(key, 0.0) for key, value in self.h_shifted.items()}


def read_tshs(run_dir: Path, run_id: str, *, fermi_ev: float | None = None) -> TshsMatrices:
    import sisl  # lazy: only this reader needs sisl

    path = run_dir / f"{run_id}.TSHS"
    if not path.is_file():
        raise CertificationError(f"{path} is missing; the reference campaign did not certify {run_id}.")
    sile = sisl.get_sile(str(path))
    hamiltonian = sile.read_hamiltonian()
    overlap = sile.read_overlap()
    lattice = hamiltonian.geometry.lattice
    sc_off = [tuple(int(component) for component in offset) for offset in lattice.sc_off]
    no_u = int(hamiltonian.no)

    def field(matrix: Any) -> dict[Key, float]:
        coo = matrix.tocsr(0).tocoo()
        return {
            (int(row), int(col % no_u), sc_off[int(col) // no_u]): float(value)
            for row, col, value in zip(coo.row, coo.col, coo.data)
        }

    stored_fermi = float(sile.read_fermi_level())
    if fermi_ev is not None and not math.isclose(stored_fermi, float(fermi_ev), rel_tol=0.0, abs_tol=1e-9):
        raise CertificationError(
            f"{run_id}: E_F in the TSHS ({stored_fermi!r}) is not the one the reference manifest "
            f"recorded ({fermi_ev!r}); the artifact and the manifest describe different runs."
        )
    return TshsMatrices(
        run_id=run_id,
        path=str(path),
        h_shifted=field(hamiltonian),
        overlap=field(overlap),
        fermi_ev=stored_fermi,
        no_u=no_u,
        nsc=tuple(int(component) for component in lattice.nsc),  # type: ignore[arg-type]
        sc_off=tuple(sc_off),
    )


def finite_difference(
    plus: TshsMatrices, minus: TshsMatrices, delta_ang: float, *, undo_fermi_shift: bool = True
) -> dict[str, dict[Key, float]]:
    """Central difference of the TSHS matrices of one ``+-`` pair.

    ``undo_fermi_shift=False`` is the control: it differentiates sisl's
    ``H - E_F S`` instead of H, which is the convention error the report has to
    demonstrate this gate would catch.
    """
    if delta_ang <= 0:
        raise CertificationError(f"delta must be positive, got {delta_ang!r}.")
    h_plus = plus.h_absolute if undo_fermi_shift else plus.h_shifted
    h_minus = minus.h_absolute if undo_fermi_shift else minus.h_shifted
    scale = 1.0 / (2.0 * delta_ang)
    keys_h, (a, b) = align(h_plus, h_minus)
    keys_s, (c, d) = align(plus.overlap, minus.overlap)
    return {
        "D_H": dict(zip(keys_h, ((a - b) * scale).tolist())),
        "D_S": dict(zip(keys_s, ((c - d) * scale).tolist())),
    }


def richardson_extrapolate(fd_h: Mapping[Key, float], fd_2h: Mapping[Key, float]) -> dict[Key, float]:
    """Combine two already-certified central differences into an O(h^4) estimate.

    ``D_h = f' + (h^2/6) f''' + O(h^4)`` and ``D_2h = f' + (4h^2/6) f''' + O(h^4)``
    share the same leading error term, so ``(4 D_h - D_2h) / 3`` cancels it and
    leaves O(h^4) -- the standard 5-point stencil, built from two 2-point central
    differences instead of four fresh matrix reads (GO-2 / E-F_001 Richardson
    requirement: a two-point FD compared only against another two-point FD
    shares its truncation error and cannot certify it away).
    """
    keys, (a, b) = align(fd_h, fd_2h)
    return dict(zip(keys, ((4.0 * a - b) / 3.0).tolist()))


def contract_fc(dhsdr: Any, vectors: np.ndarray, kind: str, *, spin: int = 0) -> dict[Key, float]:
    """``sum_{I,alpha} v[I,alpha] D_{I,alpha}`` from one ``dHSdR.nc``.

    The FC file stores one derivative per displaced atom and cartesian axis;
    a direction is a linear combination of exactly those, so the contraction is
    the directional derivative without any new numerics.
    """
    accumulated: dict[Key, float] = {}
    for (atom, axis), records in dhsdr.records.items():
        weight = float(vectors[atom - 1, axis - 1])
        if weight == 0.0:
            continue
        record = records[kind]
        values = record.values[:, spin]
        for index in range(record.nnz):
            key = (
                int(record.row[index]),
                int(record.col[index]),
                (int(record.isc[index, 0]), int(record.isc[index, 1]), int(record.isc[index, 2])),
            )
            accumulated[key] = accumulated.get(key, 0.0) + weight * float(values[index])
    return accumulated


# --------------------------------------------------------------------------- #
# Plateau, noise floor and classification
# --------------------------------------------------------------------------- #


def plateau_for(fd_by_delta: Mapping[float, Mapping[Key, float]]) -> dict[str, Any]:
    """The amplitudes over which the finite difference is flat, or why there are none."""
    measurements = {delta: norms(align(field)[1][0])["frobenius"] for delta, field in fd_by_delta.items()}
    try:
        selection = fdp.select_delta_from_plateau(measurements)
    except fdp.FdPerturbationError as exc:
        return {"has_plateau": False, "reason": str(exc), "fd_frobenius_by_delta": measurements}
    return {"has_plateau": True, "fd_frobenius_by_delta": measurements, **selection}


def noise_floor(fd_by_delta: Mapping[float, Mapping[Key, float]], plateau_deltas: Sequence[float]) -> dict[str, float]:
    """``tau_FD``: the residual variation of the finite difference inside the plateau.

    Every pair of plateau amplitudes estimates the same derivative, so the
    largest disagreement among them is what this finite difference can resolve.
    """
    tau = {norm: 0.0 for norm in NORMS}
    deltas = sorted(plateau_deltas)
    for first in range(len(deltas)):
        for second in range(first + 1, len(deltas)):
            measured = difference_norms(fd_by_delta[deltas[first]], fd_by_delta[deltas[second]])
            for norm in NORMS:
                tau[norm] = max(tau[norm], measured[norm])
    return tau


# (4 D_h - D_2h) / 3 does not cancel the *noise* in D_h and D_2h the way it
# cancels their shared O(h^2) truncation term: worst case (triangle
# inequality) the combination's resolution is |4/3| + |-1/3| times tau_FD of
# the two-point method it is built from. Reusing tau_FD unscaled would call
# amplified SCF/DM noise a "defect" -- exactly the failure mode GO-2 measures
# noise floors instead of assuming them to avoid.
RICHARDSON_NOISE_AMPLIFICATION = 4.0 / 3.0 + 1.0 / 3.0


def doubling_pairs(deltas: Sequence[float]) -> list[tuple[float, float]]:
    """Every ``(h, 2h)`` pair present in a delta sweep, smaller value first."""
    values = sorted(float(value) for value in deltas)
    pairs: list[tuple[float, float]] = []
    for h in values:
        for other in values:
            if math.isclose(2.0 * h, other, rel_tol=1e-6):
                pairs.append((h, other))
                break
    return pairs


def fitted_exponent(deltas: Sequence[float], discrepancies: Sequence[float]) -> float | None:
    """Slope of ``log(discrepancy)`` against ``log(delta)``.

    O(delta^2) truncation gives 2; a unit or index defect is delta-independent
    and gives 0.
    """
    points = [
        (math.log(float(delta)), math.log(float(value)))
        for delta, value in zip(deltas, discrepancies)
        if float(delta) > 0.0 and float(value) > 0.0
    ]
    if len(points) < 2:
        return None
    x = np.array([point[0] for point in points])
    y = np.array([point[1] for point in points])
    if float(np.ptp(x)) == 0.0:
        return None
    return float(np.polyfit(x, y, 1)[0])


def classify(
    *,
    deltas: Sequence[float],
    discrepancy_by_delta: Mapping[float, float],
    tau: float,
    is_null_direction: bool,
) -> dict[str, Any]:
    """Resolve a FC-vs-FD discrepancy as noise, truncation, or a defect.

    PASS needs a positive explanation: at or below the noise floor somewhere in
    the plateau, or a quadratic decay towards zero. "Small enough" on its own is
    not one, which is why the exponent is fitted rather than assumed.
    """
    ordered = sorted(deltas)
    values = [float(discrepancy_by_delta[delta]) for delta in ordered]
    below = [delta for delta, value in zip(ordered, values) if value <= tau]
    exponent = fitted_exponent(ordered, values)
    low, high = TRUNCATION_EXPONENT_RANGE
    truncation = exponent is not None and low <= exponent <= high

    if below:
        verdict, reason = "noise_limited", (
            f"|D^FC - D^FD| <= tau_FD = {tau:.6g} at delta = {below} (of {ordered})"
        )
    elif truncation:
        verdict, reason = "truncation_limited", (
            f"discrepancy decays as delta^{exponent:.2f} towards zero: the collective central "
            "difference carries third-derivative cross terms that a contraction of one-atom "
            "derivatives does not; it is not an index, phase or unit defect"
        )
    else:
        verdict, reason = "unresolved", (
            f"discrepancy exceeds tau_FD = {tau:.6g} at every plateau delta and does not decay as "
            f"delta^2 (fitted exponent {exponent!r}); index, phase, unit or sparsification defect "
            "is not excluded"
        )
    return {
        "verdict": verdict,
        "passed": verdict != "unresolved",
        "reason": reason,
        "tau_fd": tau,
        "deltas": ordered,
        "discrepancy_by_delta": {str(delta): value for delta, value in zip(ordered, values)},
        "deltas_within_tau": below,
        "fitted_exponent": exponent,
        "truncation_exponent_range": list(TRUNCATION_EXPONENT_RANGE),
        "is_null_direction": is_null_direction,
    }


# --------------------------------------------------------------------------- #
# Reference campaign inputs
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Campaign:
    """One ``run_epc_siesta_reference.py`` output root, loaded and validated."""

    root: Path
    manifest: dict[str, Any]

    @property
    def directions(self) -> dict[str, np.ndarray]:
        return {
            entry["direction_name"]: np.asarray(entry["vectors"], dtype=np.float64)
            for entry in self.manifest["perturbation_space"]["directions"]
        }

    @property
    def deltas(self) -> list[float]:
        return [float(value) for value in self.manifest["perturbation_space"]["delta_ang_values"]]

    @property
    def fc_tolerances(self) -> dict[str, str]:
        return dict(self.manifest["fc"]["tolerances"])

    def run_dir(self, run_id: str) -> Path:
        return self.root / "runs" / run_id

    def fermi_ev(self, run_id: str) -> float | None:
        for row in self.manifest["rows"]:
            if row["run_id"] == run_id:
                value = row.get("fermi_level_ev")
                return None if value is None else float(value)
        return None

    def certified_pairs(self) -> list[dict[str, Any]]:
        return [
            pair
            for pair in self.manifest["certification_set"]["fd_pairs"]
            if pair.get("ready_for_central_difference")
        ]

    def dhsdr_path(self, delta_ang: float) -> Path:
        for entry in self.manifest["certification_set"]["per_delta"]:
            if math.isclose(float(entry["delta_ang"]), float(delta_ang)) and entry.get("fc_dhs_available"):
                run_id = entry["fc_run_id"]
                return self.run_dir(run_id) / f"{run_id}.dHSdR.nc"
        raise CertificationError(f"No certified FC run at delta = {delta_ang} in {self.root}.")


def load_campaign(root: Path) -> Campaign:
    manifest_path = root / "epc_siesta_reference_manifest.json"
    if not manifest_path.is_file():
        raise CertificationError(
            f"{manifest_path} is missing. Run run_epc_siesta_reference.py --output-root {root} first."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != REFERENCE_SCHEMA:
        raise CertificationError(f"{manifest_path} is schema {manifest.get('schema')!r}, expected {REFERENCE_SCHEMA!r}.")
    if manifest.get("dry_run"):
        raise CertificationError(f"{manifest_path} is a dry run: it certifies no matrices.")
    if not manifest.get("physics_identical_across_branches"):
        raise CertificationError(
            f"{manifest_path}: the branches did not run the same physics, so a finite difference of "
            "one against the other is not a derivative of anything."
        )
    if not manifest["certification_set"].get("all_deltas_comparable"):
        raise CertificationError(f"{manifest_path}: not every swept delta has both a certified FC run and a +- pair.")
    return Campaign(root=root, manifest=manifest)


# --------------------------------------------------------------------------- #
# Structural checks
# --------------------------------------------------------------------------- #


def check(name: str, passed: bool, detail: str, **extra: Any) -> dict[str, Any]:
    return {"check": name, "passed": bool(passed), "detail": detail, **extra}


def structural_checks(
    *,
    dhsdr: Any,
    equilibrium: TshsMatrices,
    orbital_contract: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Index, R/image mapping and orbital-contract agreement, before any tolerance."""
    checks: list[dict[str, Any]] = []
    record = next(iter(dhsdr.records.values()))["D_H"]

    checks.append(
        check(
            "orbital_count_matches_equilibrium",
            dhsdr.no_u == equilibrium.no_u == int(orbital_contract["orbital_count"]),
            f"dHSdR no_u = {dhsdr.no_u}, TSHS no_u = {equilibrium.no_u}, "
            f"orbital contract = {orbital_contract['orbital_count']}",
        )
    )
    checks.append(
        check(
            "displaced_atoms_match_contract",
            len(dhsdr.displaced_atoms) == int(orbital_contract["atom_count"]),
            f"dHSdR displaced atoms {list(dhsdr.displaced_atoms)}; the contract declares "
            f"{orbital_contract['atom_count']} atoms, so every degree of freedom is present",
        )
    )
    checks.append(
        check(
            "nsc_matches_equilibrium",
            tuple(int(value) for value in record.nsc) == equilibrium.nsc,
            f"dHSdR nsc = {record.nsc.tolist()}, TSHS nsc = {list(equilibrium.nsc)}",
        )
    )

    # The R/image mapping. Both files enumerate the same lattice offsets, but in
    # different orders -- aligning by stored index instead of by the offset
    # itself would silently permute the images, so this records both facts.
    file_offsets = {tuple(int(component) for component in offset) for offset in record.isc_off}
    tshs_offsets = set(equilibrium.sc_off)
    orderings_identical = [tuple(int(c) for c in offset) for offset in record.isc_off] == list(equilibrium.sc_off)
    # A boolean verdict is not itself evidence -- a reviewer cannot tell from
    # `true` alone which topology was actually compared, or reproduce the
    # comparison against a different run. Publish the same kind of canonical
    # hash the `raw_derivative`/`electronic_hamiltonian` artifact contracts
    # already declare as `topology_sha256` (shared/artifact_signature.py),
    # over the verified-identical offset set itself (sorted, so the hash does
    # not depend on file-internal ordering, matching `orderings_identical`
    # already recording that fact separately).
    topology_sha256 = input_signature_sha256(
        {"kind": "r_image_offsets", "offsets": sorted(file_offsets)}
    )
    checks.append(
        check(
            "r_image_sets_identical",
            file_offsets == tshs_offsets,
            f"{len(file_offsets)} lattice offsets in dHSdR, {len(tshs_offsets)} in the TSHS; "
            f"symmetric difference {sorted(file_offsets ^ tshs_offsets)}",
            supercell_rule=SUPERCELL_RULE,
            orderings_identical=orderings_identical,
            topology_sha256=topology_sha256,
            alignment="by lattice offset (row, col, R), never by stored image index",
            note=(
                "the two enumerations of the same offsets are in different orders; "
                "aligning by stored index would permute the images"
                if not orderings_identical
                else "orderings coincide here, but alignment is still by offset"
            ),
        )
    )

    dh_keys = set(contract_fc(dhsdr, _one_hot_like(dhsdr, record.atom, record.axis), "D_H"))
    checks.append(
        check(
            "dh_covers_the_tshs_sparsity_pattern",
            dh_keys == set(equilibrium.h_shifted),
            f"dH has {len(dh_keys)} elements, the equilibrium H has {len(equilibrium.h_shifted)}; "
            f"{len(dh_keys - set(equilibrium.h_shifted))} outside the pattern, "
            f"{len(set(equilibrium.h_shifted) - dh_keys)} missing",
        )
    )
    checks.extend(
        check(f"reader_{entry['check']}", entry["outcome"] == "pass", entry.get("detail", ""))
        for entry in dhsdr.checks
        if entry["outcome"] == "fail"
    )
    return checks


def merge_structural_checks(per_delta: Mapping[float, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    """Collapse the same check run over several FC files into one verdict each.

    A check passes only if it passed for every amplitude; the per-amplitude
    details are kept so a single anomalous FC run stays identifiable.
    """
    merged: dict[str, dict[str, Any]] = {}
    for delta, checks in sorted(per_delta.items()):
        for entry in checks:
            name = str(entry["check"])
            slot = merged.setdefault(name, {"check": name, "passed": True, "detail": entry["detail"], "per_delta": {}})
            slot["passed"] = bool(slot["passed"]) and bool(entry["passed"])
            slot["per_delta"][str(delta)] = {
                key: value for key, value in entry.items() if key not in {"check"}
            }
            if not entry["passed"]:
                slot["detail"] = f"delta = {delta}: {entry['detail']}"
    return list(merged.values())


def _one_hot_like(dhsdr: Any, atom: int, axis: int) -> np.ndarray:
    vectors = np.zeros((dhsdr.na_u, 3), dtype=np.float64)
    vectors[atom - 1, axis - 1] = 1.0
    return vectors


# --------------------------------------------------------------------------- #
# Sparsification
# --------------------------------------------------------------------------- #


def sparsification_comparison(
    *, off: Campaign, on: Campaign, deltas: Sequence[float]
) -> dict[str, Any]:
    """Thresholds on vs off, on FC runs that differ in nothing else.

    The release documents dH as unfiltered and dS as thresholded. Neither is
    taken on faith: dH must come back bit-identical, and dS must be a pure
    threshold -- survivors bit-identical, every dropped element below the
    declared tolerance and every kept element above it.
    """
    tolerances_on = on.fc_tolerances
    thresholds = {
        "D_H": _tolerance_value(tolerances_on["dhdr"]) * RY_TO_EV / BOHR_TO_ANG,
        "D_S": _tolerance_value(tolerances_on["dsdr"]) / BOHR_TO_ANG,
    }
    rows: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []

    for delta in deltas:
        file_off = read_dhsdr(off.dhsdr_path(delta))
        file_on = read_dhsdr(on.dhsdr_path(delta))
        for (atom, axis) in sorted(file_off.records):
            for kind in KINDS:
                vectors = _one_hot_like(file_off, atom, axis)
                field_off = contract_fc(file_off, vectors, kind)
                field_on = contract_fc(file_on, vectors, kind)
                dropped = {key: field_off[key] for key in set(field_off) - set(field_on)}
                extra = sorted(set(field_on) - set(field_off))
                survivors = set(field_off) & set(field_on)
                survivor_delta = max((abs(field_off[key] - field_on[key]) for key in survivors), default=0.0)
                rows.append(
                    {
                        "delta_ang": delta,
                        "atom": atom,
                        "axis": axis,
                        "kind": kind,
                        "unit": CANONICAL_UNITS[kind],
                        "threshold_canonical": thresholds[kind],
                        "nnz_tolerance_off": len(field_off),
                        "nnz_tolerance_on": len(field_on),
                        "dropped": len(dropped),
                        "introduced": len(extra),
                        "max_abs_change_on_survivors": survivor_delta,
                        "max_abs_dropped": max((abs(value) for value in dropped.values()), default=0.0),
                        "min_abs_kept": min((abs(value) for value in field_on.values()), default=0.0),
                    }
                )

    def rows_for(kind: str) -> list[dict[str, Any]]:
        return [row for row in rows if row["kind"] == kind]

    dh_rows = rows_for("D_H")
    checks.append(
        check(
            "dh_tolerance_is_a_noop",
            all(row["dropped"] == 0 and row["introduced"] == 0 and row["max_abs_change_on_survivors"] == 0.0 for row in dh_rows),
            f"FC.dHdR.Tolerance = {tolerances_on['dhdr']} drops "
            f"{max((row['dropped'] for row in dh_rows), default=0)} elements and changes survivors by "
            f"{max((row['max_abs_change_on_survivors'] for row in dh_rows), default=0.0):.3e} eV/Ang: dH is "
            "unfiltered in this release, exactly as the C06 reader records",
        )
    )

    ds_rows = rows_for("D_S")
    checks.append(
        check(
            "ds_sparsification_does_not_perturb_survivors",
            all(row["max_abs_change_on_survivors"] == 0.0 and row["introduced"] == 0 for row in ds_rows),
            f"surviving dS elements are bit-identical with the threshold on "
            f"(max change {max((row['max_abs_change_on_survivors'] for row in ds_rows), default=0.0):.3e} 1/Ang) "
            "and the threshold introduces no element",
        )
    )
    checks.append(
        check(
            "ds_sparsification_is_a_pure_threshold",
            # The threshold must separate the two populations cleanly: nothing
            # above it was dropped, nothing below it was kept.
            all(
                row["max_abs_dropped"] <= row["threshold_canonical"] <= row["min_abs_kept"]
                for row in ds_rows
            ),
            f"FC.dSdR.Tolerance = {tolerances_on['dsdr']} = {thresholds['D_S']:.6g} 1/Ang separates "
            f"max|dropped| = {max((row['max_abs_dropped'] for row in ds_rows), default=0.0):.6g} from "
            f"min|kept| = {min((row['min_abs_kept'] for row in ds_rows), default=0.0):.6g}",
        )
    )
    return {
        "tolerances_off": off.fc_tolerances,
        "tolerances_on": tolerances_on,
        "thresholds_canonical": thresholds,
        "rows": rows,
        "checks": checks,
        "passed": all(entry["passed"] for entry in checks),
    }


def _tolerance_value(text: str) -> float:
    """The number in an fdf tolerance like ``1.0e-3 Ry/Bohr``."""
    try:
        return float(str(text).split()[0])
    except (ValueError, IndexError) as exc:
        raise CertificationError(f"Cannot read a tolerance value out of {text!r}.") from exc


# --------------------------------------------------------------------------- #
# The certification
# --------------------------------------------------------------------------- #


def certify(
    *,
    reference_root: Path = DEFAULT_REFERENCE_ROOT,
    sparsified_root: Path | None = DEFAULT_SPARSIFIED_ROOT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    campaign = load_campaign(reference_root)
    directions = campaign.directions
    deltas = campaign.deltas

    equilibrium_id = campaign.manifest["certification_set"]["equilibrium_run_id"]
    if not equilibrium_id:
        raise CertificationError(f"{reference_root} has no certified equilibrium run.")
    equilibrium = read_tshs(
        campaign.run_dir(equilibrium_id), equilibrium_id, fermi_ev=campaign.fermi_ev(equilibrium_id)
    )

    # --- both sides, for every certified (direction, delta) --------------------
    dhsdr_by_delta = {delta: read_dhsdr(campaign.dhsdr_path(delta)) for delta in deltas}
    fd: dict[tuple[str, float, str], dict[Key, float]] = {}
    fd_uncorrected: dict[tuple[str, float, str], dict[Key, float]] = {}
    fc: dict[tuple[str, float, str], dict[Key, float]] = {}
    pattern_notes: list[dict[str, Any]] = []

    for pair in campaign.certified_pairs():
        name, delta = pair["direction_name"], float(pair["delta_ang"])
        if name not in directions:
            raise CertificationError(f"Direction {name!r} is certified but absent from the perturbation space.")
        plus = read_tshs(campaign.run_dir(pair["plus_run_id"]), pair["plus_run_id"], fermi_ev=campaign.fermi_ev(pair["plus_run_id"]))
        minus = read_tshs(campaign.run_dir(pair["minus_run_id"]), pair["minus_run_id"], fermi_ev=campaign.fermi_ev(pair["minus_run_id"]))
        pattern_notes.append(
            {
                "direction_name": name,
                "delta_ang": delta,
                "sparsity_pattern_identical": set(plus.h_shifted) == set(minus.h_shifted) == set(equilibrium.h_shifted),
                "nsc_identical": plus.nsc == minus.nsc == equilibrium.nsc,
                "fermi_ev": {"plus": plus.fermi_ev, "minus": minus.fermi_ev, "equilibrium": equilibrium.fermi_ev},
            }
        )
        corrected = finite_difference(plus, minus, delta)
        uncorrected = finite_difference(plus, minus, delta, undo_fermi_shift=False)
        for kind in KINDS:
            fd[(name, delta, kind)] = corrected[kind]
            fd_uncorrected[(name, delta, kind)] = uncorrected[kind]
            fc[(name, delta, kind)] = contract_fc(dhsdr_by_delta[delta], directions[name], kind)

    names = sorted({name for name, _, _ in fd})

    # --- plateau and tau_FD ----------------------------------------------------
    plateaus: dict[tuple[str, str], dict[str, Any]] = {}
    taus: dict[tuple[str, str], dict[str, float]] = {}
    for name in names:
        for kind in KINDS:
            by_delta = {delta: fd[(name, delta, kind)] for delta in deltas if (name, delta, kind) in fd}
            plateau = plateau_for(by_delta)
            plateaus[(name, kind)] = plateau
            if plateau["has_plateau"]:
                taus[(name, kind)] = noise_floor(by_delta, plateau["plateau_delta_ang"])

    # A direction with no plateau is only acceptable when it has no derivative
    # to reproduce: its whole signal must sit under the floor measured on the
    # directions that do carry one.
    global_tau = {
        kind: {norm: max((taus[(n, k)][norm] for (n, k) in taus if k == kind), default=0.0) for norm in NORMS}
        for kind in KINDS
    }
    for name in names:
        for kind in KINDS:
            plateau = plateaus[(name, kind)]
            signal = {
                norm: max(
                    (norms(align(fd[(name, delta, kind)])[1][0])[norm] for delta in deltas if (name, delta, kind) in fd),
                    default=0.0,
                )
                for norm in NORMS
            }
            plateau["fd_signal"] = signal
            # Null means "no plateau *because* there is no signal". A direction
            # that does have a plateau is certified on its own terms, so it
            # never takes the absolute-floor shortcut or skips the unit check.
            plateau["is_null_direction"] = not plateau["has_plateau"] and all(
                signal[norm] <= global_tau[kind][norm] for norm in NORMS
            )
            if plateau["is_null_direction"]:
                # No plateau can exist because there is no signal: certify it
                # against the absolute floor instead (an acoustic sum rule).
                plateau["plateau_delta_ang"] = list(deltas)
                plateau["selected_delta_ang"] = max(deltas)
                plateau["criterion"] = (
                    "null direction: ||D^FD|| lies under tau_FD of the directions that do carry "
                    "signal, so translational invariance -- not a ratio -- is what is certified"
                )
                taus[(name, kind)] = dict(global_tau[kind])

    # --- the verdict per (direction, kind, norm) ------------------------------
    rows: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    for name in names:
        for kind in KINDS:
            plateau = plateaus[(name, kind)]
            if (name, kind) not in taus:
                comparisons.append(
                    {
                        "direction_name": name,
                        "kind": kind,
                        "passed": False,
                        "plateau": plateau,
                        "reason": (
                            "no plateau over the swept amplitudes and the direction is not null: "
                            "tau_FD is undefined, so no discrepancy can be called compatible with it"
                        ),
                    }
                )
                continue
            plateau_deltas = [float(value) for value in plateau["plateau_delta_ang"]]
            tau = taus[(name, kind)]
            per_norm: dict[str, Any] = {}
            for norm in NORMS:
                discrepancy = {
                    delta: difference_norms(fc[(name, delta, kind)], fd[(name, delta, kind)])[norm]
                    for delta in plateau_deltas
                }
                per_norm[norm] = classify(
                    deltas=plateau_deltas,
                    discrepancy_by_delta=discrepancy,
                    tau=tau[norm],
                    is_null_direction=bool(plateau["is_null_direction"]),
                )
            scale = best_fit_scale(fd[(name, plateau["selected_delta_ang"], kind)], fc[(name, plateau["selected_delta_ang"], kind)])
            # A null direction has nothing to fit a scale against; both sides are
            # zero, so the unit check is carried by the directions with signal.
            scale_ok = bool(plateau["is_null_direction"]) or (
                scale is not None and abs(scale - 1.0) <= UNIT_SCALE_TOLERANCE
            )
            comparison = {
                "direction_name": name,
                "kind": kind,
                "unit": CANONICAL_UNITS[kind],
                "plateau": plateau,
                "tau_fd": tau,
                "by_norm": per_norm,
                "unit_scale": {
                    "best_fit_scale": scale,
                    "tolerance": UNIT_SCALE_TOLERANCE,
                    "passed": scale_ok,
                    "detail": (
                        "alpha minimising ||D^FD - alpha D^FC||; the wrong-unit alternatives for D_H are "
                        f"{BOHR_TO_ANG:.4f} (Ry/Bohr read as Ry/Ang), {1 / BOHR_TO_ANG:.4f} and {RY_TO_EV:.4f}, "
                        "all outside the window by more than an order of magnitude"
                        if not plateau["is_null_direction"]
                        else "null direction: both sides vanish, so no scale is determined and none is required"
                    ),
                },
                "passed": all(per_norm[norm]["passed"] for norm in NORMS) and scale_ok,
            }
            comparisons.append(comparison)
            for norm in NORMS:
                for delta in plateau_deltas:
                    rows.append(
                        {
                            "direction_name": name,
                            "kind": kind,
                            "unit": CANONICAL_UNITS[kind],
                            "norm": norm,
                            "delta_ang": delta,
                            "fd_signal": norms(align(fd[(name, delta, kind)])[1][0])[norm],
                            "fc_signal": norms(align(fc[(name, delta, kind)])[1][0])[norm],
                            "discrepancy": per_norm[norm]["discrepancy_by_delta"][str(delta)],
                            "tau_fd": tau[norm],
                            "within_tau": per_norm[norm]["discrepancy_by_delta"][str(delta)] <= tau[norm],
                            "verdict": per_norm[norm]["verdict"],
                            "fitted_exponent": per_norm[norm]["fitted_exponent"],
                            "is_null_direction": bool(plateau["is_null_direction"]),
                            "best_fit_scale": comparison["unit_scale"]["best_fit_scale"],
                        }
                    )

    # --- checks that do not depend on a tolerance ------------------------------
    # Every FC file, not just the first: no_u, nsc and the R/image table are
    # per-run properties, and one anomalous amplitude would otherwise pass
    # unlooked-at.
    checks = merge_structural_checks(
        {
            delta: structural_checks(
                dhsdr=dhsdr_by_delta[delta],
                equilibrium=equilibrium,
                orbital_contract=campaign.manifest["orbital_contract"],
            )
            for delta in deltas
        }
    )
    checks.append(
        check(
            "sparsity_pattern_stable_across_displacements",
            all(note["sparsity_pattern_identical"] and note["nsc_identical"] for note in pattern_notes),
            "every +- run shares the equilibrium sparsity pattern and supercell, so the finite "
            "difference is taken over one fixed index space",
            rows=pattern_notes,
        )
    )

    hermiticity = []
    for name in names:
        for kind in KINDS:
            for source, field_map in (("fc", fc), ("fd", fd)):
                for delta in deltas:
                    if (name, delta, kind) not in field_map:
                        continue
                    residual = hermiticity_residual(field_map[(name, delta, kind)])
                    scale_ = norms(align(field_map[(name, delta, kind)])[1][0])["max_abs"]
                    hermiticity.append(
                        {
                            "direction_name": name,
                            "kind": kind,
                            "source": source,
                            "delta_ang": delta,
                            "signal_max_abs": scale_,
                            **residual,
                            "passed": residual["missing_transpose_partners"] == 0
                            and residual["max_abs_residual"] <= max(1e-8 * scale_, 1e-10),
                        }
                    )
    checks.append(
        check(
            "real_space_hermiticity",
            all(entry["passed"] for entry in hermiticity),
            "D(mu, nu, R) = D(nu, mu, -R) holds on both sides for every direction, kind and amplitude; "
            f"largest residual {max((entry['max_abs_residual'] for entry in hermiticity), default=0.0):.3e}",
            rows=hermiticity,
        )
    )

    # The control. If differentiating sisl's H - E_F S also passed, the gate
    # would be measuring nothing.
    control = fermi_control(fd=fd, fd_uncorrected=fd_uncorrected, fc=fc, taus=taus, plateaus=plateaus)
    checks.append(
        check(
            "fermi_shift_correction_is_required",
            control["passed"],
            control["detail"],
            **{key: value for key, value in control.items() if key not in {"passed", "detail"}},
        )
    )

    sparsification = None
    if sparsified_root is not None:
        sparsification = sparsification_comparison(
            off=campaign, on=load_campaign(Path(sparsified_root)), deltas=deltas
        )
        checks.extend(sparsification["checks"])
    else:
        checks.append(
            check(
                "sparsification_compared",
                False,
                "no sparsified campaign supplied; GO-2 requires thresholds on and off to be compared",
            )
        )

    # --- provenance ------------------------------------------------------------
    runtime_ref = campaign.manifest["siesta_runtime_ref"]
    runtime_validation = validate_siesta_runtime_ref(runtime_ref, effective_sha256=runtime_ref.get("binary_sha256"))
    classification = runtime_ref.get("classification")
    checks.append(
        check(
            "siesta_runtime_reference_valid",
            bool(runtime_validation.get("valid")),
            f"runtime is {classification}; reasons {runtime_validation.get('reasons')}",
        )
    )

    # --- Richardson (h, 2h) reference: separates O(delta^2) truncation of the
    # two-point FD from tau_FD noise, using an O(delta^4) combination of the
    # same central differences instead of a second two-point FD that would
    # share the same truncation error (GO-2 / E-F_001 requirement).
    richardson_available_pairs = doubling_pairs(deltas)
    richardson_rows: list[dict[str, Any]] = []
    for name in names:
        for kind in KINDS:
            if (name, kind) not in taus:
                continue
            tau = taus[(name, kind)]["frobenius"] * RICHARDSON_NOISE_AMPLIFICATION
            discrepancy_by_h: dict[float, float] = {}
            for h, two_h in richardson_available_pairs:
                if (name, h, kind) not in fd or (name, two_h, kind) not in fd or (name, h, kind) not in fc:
                    continue
                estimate = richardson_extrapolate(fd[(name, h, kind)], fd[(name, two_h, kind)])
                discrepancy_by_h[h] = difference_norms(fc[(name, h, kind)], estimate)["frobenius"]
            if not discrepancy_by_h:
                continue
            richardson_verdict = classify(
                deltas=sorted(discrepancy_by_h),
                discrepancy_by_delta=discrepancy_by_h,
                tau=tau,
                is_null_direction=bool(plateaus[(name, kind)]["is_null_direction"]),
            )
            richardson_rows.append({"direction_name": name, "kind": kind, **richardson_verdict})
            checks.append(
                check(
                    f"richardson_reference_{name}_{kind}",
                    richardson_verdict["passed"],
                    "FC vs O(delta^4) Richardson (h, 2h) extrapolation of the certified TSHS finite "
                    f"differences: {richardson_verdict['reason']}",
                    base_deltas_ang=sorted(discrepancy_by_h),
                )
            )
    if not richardson_available_pairs:
        checks.append(
            check(
                "richardson_reference_available",
                False,
                f"no (h, 2h) doubling pair in the swept deltas {deltas}; GO-2 requires a higher-order "
                "reference independent of the two-point FD's own truncation error",
            )
        )

    comparisons_passed = all(entry["passed"] for entry in comparisons)
    checks_passed = all(entry["passed"] for entry in checks)
    verdict = "PASS" if comparisons_passed and checks_passed and comparisons else "FAIL"

    report = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "gate": "GO-2",
        "ticket": "C09 / E-F_001-S12",
        "verdict": verdict,
        "reference_root": str(reference_root),
        "reference_manifest_sha256": file_sha256(reference_root / "epc_siesta_reference_manifest.json"),
        "sparsified_root": str(sparsified_root) if sparsified_root else None,
        "provenance": {
            "siesta_runtime_ref": runtime_ref,
            "classification": classification,
            "validation": {
                "valid": runtime_validation.get("valid"),
                "reasons": runtime_validation.get("reasons"),
                "warnings": runtime_validation.get("warnings"),
            },
            "limitation": (
                "VERIFIED_BINARY_ONLY: the binary is hashed, versioned and its build recorded, but no "
                "source/tag/commit is bound to it. Fase 2 makes this exhaustive finite-difference "
                "certification mandatory in that state, and the limitation propagates to every artifact "
                "derived from these derivatives -- it is not discharged by this PASS."
                if classification == "VERIFIED_BINARY_ONLY"
                else None
            ),
            "propagates_to_derived_artifacts": classification != "VERIFIED_SOURCE",
        },
        "conventions": {
            "sign_convention": SIGN_CONVENTION,
            "supercell_rule": SUPERCELL_RULE,
            "alignment": "by (row orbital, column orbital, lattice offset R), never by stored image index",
            "fermi_shift": (
                "sisl returns H - E_F S and dHSdR.nc stores the unshifted dH/dR, so the certified finite "
                "difference is d(H_sisl + E_F S) with each run's own E_F; E_F moves with the displacement"
            ),
            "units": {
                "D_H": {"file": "Ry/Bohr", "canonical": "eV/Ang", "factor": RY_TO_EV / BOHR_TO_ANG},
                "D_S": {"file": "1/Bohr", "canonical": "1/Ang", "factor": 1.0 / BOHR_TO_ANG},
                "constants": {"RY_TO_EV": RY_TO_EV, "BOHR_TO_ANG": BOHR_TO_ANG},
                "certified_numerically": True,
            },
        },
        "perturbation_space": {
            "directions": sorted(names),
            "delta_ang_values": deltas,
            "method": campaign.manifest["perturbation_space"]["method"],
            "normalization": campaign.manifest["perturbation_space"]["normalization"],
        },
        "tau_fd": {
            "definition": (
                "max over plateau amplitude pairs of ||D^FD(delta_i) - D^FD(delta_j)||, in the same norm "
                "the discrepancy is measured in: the residual variation of the finite difference itself"
            ),
            "per_direction": {f"{name}/{kind}": taus.get((name, kind)) for name in names for kind in KINDS},
            "global_by_kind": global_tau,
        },
        "comparisons": comparisons,
        "richardson_reference": {
            "definition": (
                "(4*D_h - D_2h)/3 from the already-certified central differences at h and 2h; the "
                "shared O(h^2) leading error cancels, leaving O(h^4). Compared against the FC "
                "contraction at h so the O(h^2) truncation of the two-point FD cannot masquerade as "
                "agreement -- a second two-point FD would share that truncation and prove nothing. "
                f"tau is tau_FD scaled by {RICHARDSON_NOISE_AMPLIFICATION:.4g} (=4/3+1/3), the worst-case "
                "propagation of the two-point method's own noise floor through this linear combination"
            ),
            "pairs_available_ang": richardson_available_pairs,
            "rows": richardson_rows,
        },
        "structural_checks": checks,
        "sparsification": sparsification,
        "summary": {
            "comparisons_total": len(comparisons),
            "comparisons_passed": len([entry for entry in comparisons if entry["passed"]]),
            "checks_total": len(checks),
            "checks_failed": [entry["check"] for entry in checks if not entry["passed"]],
            "verdicts": _tally(rows),
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "go2_certification.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=_json_safe) + "\n", encoding="utf-8"
    )
    _write_csv(output_dir / "go2_certification.csv", rows)
    return report


def fermi_control(
    *,
    fd: Mapping[tuple[str, float, str], Mapping[Key, float]],
    fd_uncorrected: Mapping[tuple[str, float, str], Mapping[Key, float]],
    fc: Mapping[tuple[str, float, str], Mapping[Key, float]],
    taus: Mapping[tuple[str, str], Mapping[str, float]],
    plateaus: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    """Show that the gate rejects the wrong Fermi convention.

    Only directions that carry signal can demonstrate this: on a null direction
    ``E_F dS`` vanishes with everything else, so the two conventions agree there
    and prove nothing.
    """
    rows: list[dict[str, Any]] = []
    for (name, delta, kind), corrected in fd.items():
        if kind != "D_H" or plateaus[(name, kind)].get("is_null_direction"):
            continue
        tau = taus.get((name, kind), {}).get("frobenius")
        if tau is None:
            continue
        with_correction = difference_norms(fc[(name, delta, kind)], corrected)["frobenius"]
        without = difference_norms(fc[(name, delta, kind)], fd_uncorrected[(name, delta, kind)])["frobenius"]
        rows.append(
            {
                "direction_name": name,
                "delta_ang": delta,
                "tau_fd_frobenius": tau,
                "discrepancy_corrected": with_correction,
                "discrepancy_uncorrected": without,
                "ratio_to_tau": without / tau if tau > 0 else float("inf"),
                "rejected": without > FERMI_CONTROL_MIN_RATIO * tau,
            }
        )
    passed = bool(rows) and all(row["rejected"] for row in rows)
    worst = min((row["ratio_to_tau"] for row in rows), default=0.0)
    return {
        "passed": passed,
        "detail": (
            f"differentiating sisl's H - E_F S instead of H exceeds tau_FD by at least {worst:.1f}x on every "
            f"direction that carries signal (required: {FERMI_CONTROL_MIN_RATIO}x), so this gate does "
            "discriminate the convention rather than absorbing it"
            if passed
            else "the uncorrected Fermi convention is not rejected by this gate; the agreement it reports "
            "cannot be attributed to the derivatives being right"
        ),
        "min_ratio_to_tau": worst,
        "required_ratio": FERMI_CONTROL_MIN_RATIO,
        "rows": rows,
    }


def _tally(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    tally: dict[str, int] = {}
    for row in rows:
        tally[str(row["verdict"])] = tally.get(str(row["verdict"]), 0) + 1
    return tally


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "direction_name",
        "kind",
        "unit",
        "norm",
        "delta_ang",
        "fd_signal",
        "fc_signal",
        "discrepancy",
        "tau_fd",
        "within_tau",
        "verdict",
        "fitted_exponent",
        "is_null_direction",
        "best_fit_scale",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE_ROOT)
    parser.add_argument(
        "--sparsified-root",
        type=Path,
        default=DEFAULT_SPARSIFIED_ROOT,
        help="A second campaign differing only in FC.dHdR/dSdR.Tolerance.",
    )
    parser.add_argument("--no-sparsification", action="store_true", help="Skip the on/off comparison (GO-2 then fails).")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        report = certify(
            reference_root=args.reference_root,
            sparsified_root=None if args.no_sparsification else args.sparsified_root,
            output_dir=args.output_dir,
        )
    except (CertificationError, DhsdrSchemaError, fdp.FdPerturbationError) as exc:
        print(f"[GO-2][ERROR] {exc}", file=sys.stderr)
        return 2

    for comparison in report["comparisons"]:
        for norm in NORMS:
            entry = comparison.get("by_norm", {}).get(norm)
            if entry is None:
                continue
            print(
                f"[GO-2] {comparison['direction_name']:>14} {comparison['kind']} {norm:>9}  "
                f"tau_FD={entry['tau_fd']:.3e}  {entry['verdict']}"
            )
    for failed in report["summary"]["checks_failed"]:
        print(f"[GO-2][CHECK-FAILED] {failed}", file=sys.stderr)
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "provenance": report["provenance"]["classification"],
                **report["summary"],
            },
            sort_keys=True,
        )
    )
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
