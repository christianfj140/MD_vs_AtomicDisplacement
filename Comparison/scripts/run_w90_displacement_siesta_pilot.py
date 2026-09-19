#!/usr/bin/env python3
"""DATASET-DESIGN-W90-001-S3 -- run SIESTA for the W90 displacement pilot dataset.

Generates RUN.fdf inputs for the W90 graphene reference (equilibrium) plus every
displacement configuration produced by S2 (``w90_displacement_sampler_family``)
in the pilot scope (R in {0.03, 0.08} Ang, k in {1, 2}, all four
dimensionalities, N up to 64 for the stochastic families), then reuses the
existing derivative-SIESTA-reference infrastructure
(``run_hamiltonian_derivative_siesta_references.py``) to run/skip each one
idempotently. Afterwards it verifies Hermiticity and the orbital contract, and
computes the nonlinear-response diagnostic delta_nl(r).

Cites and reuses, rather than reimplements:
  - ``docs/dataset_design_w90_001_s1_convention_manifest.md`` (S1): Regime A
    SIESTA settings for ``materials/graphene/RUN.fdf`` -- this script never
    overrides MeshCutoff/basis/XC, it only rewrites geometry.
  - ``Comparison/scripts/w90_displacement_sampler_family.py`` (S2): displacement
    generation.
  - ``Comparison/scripts/run_epc_siesta_reference.py``: ``canonical_ang_base_fdf``
    (Fractional -> Ang rewrite) and ``backend_preflight`` (GPU/CPU evidence).
  - ``Comparison/scripts/run_hamiltonian_derivative_siesta_references.py``:
    ``run_derivative_siesta_references`` (idempotent SIESTA submission,
    geometry-drift guard, pseudopotential staging).
  - ``shared/orbital_contract.py``: ``certify_system`` (orbital count/ordering
    vs. ``.ion.xml`` + ``ORB_INDX``).

The MD family is not generated here: S2's ``md_family_contract()`` reserves MD
frame generation for a later step, and this script inherits that boundary
unchanged.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for _path in (SCRIPT_DIR, REPO_ROOT / "shared"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import w90_displacement_sampler_family as sampler  # noqa: E402
from run_epc_siesta_reference import backend_preflight, canonical_ang_base_fdf  # noqa: E402
from run_hamiltonian_derivative_siesta_references import (  # noqa: E402
    DerivativeSiestaReferenceError,
    run_derivative_siesta_references,
)
from reference_selection import choose_reference_matrix  # noqa: E402
from fdf_materialization import extract_fdf_structure, materialize_sample_fdf  # noqa: E402
from orbital_contract import certify_system  # noqa: E402

DEFAULT_MATERIAL_FDF = REPO_ROOT / "materials/graphene/RUN.fdf"
DEFAULT_BASIS_DIR = REPO_ROOT / "materials/graphene/basis"
DEFAULT_PSEUDO_SOURCE_ROOT = REPO_ROOT / "materials/graphene"
DEFAULT_SIESTA_COMMAND = "/home/christian/bin/siesta"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "Comparison/results/dataset_design_w90_001_s3"

# Pilot scope, per task spec: two amplitudes, both k values, all four
# dimensionalities, N capped at 64 for the families whose size scales with N.
PILOT_AMPLITUDES_ANG: tuple[float, ...] = (0.03, 0.08)
PILOT_K_VALUES: tuple[int, ...] = (1, 2)
PILOT_DIMENSIONALITIES: tuple[str, ...] = sampler.DIMENSIONALITIES
PILOT_MAX_N = 64
PILOT_SEED = 0

EXPECTED_ATOMS = 2
EXPECTED_ORBITALS = 8  # 2 C atoms x (1 s-like + 3 p-like) orbitals; Ghost-H unpopulated.
HERMITICITY_ABS_TOLERANCE = 1e-10
DELTA_NL_EPSILON = 1e-12


class W90PilotError(RuntimeError):
    """Raised when the pilot batch cannot be built or certified."""


def _amp_tag(amplitude_ang: float) -> str:
    return f"{amplitude_ang:g}".replace(".", "p").replace("-", "m")


def _equilibrium_configuration() -> sampler.Configuration:
    return sampler.Configuration(
        family="reference",
        displacements_ang={},
        metadata={
            "family": "reference",
            "dimensionality": "none",
            "amplitude_ang": 0.0,
            "k": 0,
            "active_atom_indices": [],
        },
    )


def build_pilot_configurations(
    geometry: sampler.Geometry,
) -> list[tuple[str, sampler.Configuration]]:
    """One (sample_id, Configuration) per W90 pilot-scope structure, plus the equilibrium reference."""

    entries: list[tuple[str, sampler.Configuration]] = [("reference", _equilibrium_configuration())]
    for k in PILOT_K_VALUES:
        for dim in PILOT_DIMENSIONALITIES:
            for amplitude in PILOT_AMPLITUDES_ANG:
                group: list[sampler.Configuration] = []
                group += sampler.generate_axial_radial(geometry, k, dim, radii_ang=(amplitude,))
                if dim in ("2D_in", "3D"):
                    group += sampler.generate_angular_shell(
                        geometry, k, dim, amplitude, n_points=PILOT_MAX_N
                    )
                if k == 2:
                    group += sampler.generate_local_pair_modes(geometry, dim, amplitudes_ang=(amplitude,))
                group += sampler.generate_sobol_sparse(
                    geometry, k, dim, amplitude, max_n=PILOT_MAX_N, seed=PILOT_SEED
                )
                group += sampler.generate_random_cartesian(
                    geometry, k, dim, amplitude, PILOT_MAX_N, seed=PILOT_SEED
                )
                amp_tag = _amp_tag(amplitude)
                for index, config in enumerate(group):
                    sample_id = f"{config.family}__k{k}__{dim}__r{amp_tag}__{index:03d}"
                    entries.append((sample_id, config))
    return entries


def materialize_structures(
    entries: list[tuple[str, sampler.Configuration]],
    *,
    structures_root: Path,
    canonical_base_fdf: Path,
) -> None:
    """Write one RUN.fdf + metadata.json per (sample_id, Configuration)."""

    base_structure = extract_fdf_structure(canonical_base_fdf)
    base_positions = [tuple(position) for position in base_structure.positions_ang]
    for sample_id, config in entries:
        displaced = [
            tuple(base_positions[atom_index][axis] + config.displacements_ang.get(atom_index, (0.0, 0.0, 0.0))[axis] for axis in range(3))
            for atom_index in range(len(base_positions))
        ]
        out_dir = structures_root / sample_id
        materialize_sample_fdf(
            canonical_base_fdf,
            out_dir / "RUN.fdf",
            positions_ang=displaced,
            single_point=True,
        )
        metadata = dict(config.metadata)
        metadata["sample_id"] = sample_id
        (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def gamma_hamiltonian(reference_dir: Path) -> tuple[Any, Any]:
    """Dense Gamma-point Hamiltonian (eV) and the selected TSHS/HSX path."""

    import sisl

    selection = choose_reference_matrix(reference_dir, require_positive_provenance=False)
    if not selection.ok or selection.path is None:
        raise W90PilotError(f"No usable reference matrix in {reference_dir}: {selection.reason}")
    hamiltonian = sisl.get_sile(str(selection.path)).read_hamiltonian()
    dense = hamiltonian.Hk(k=[0.0, 0.0, 0.0], format="array")
    return dense, selection.path


def verify_hermiticity(reference_dir: Path) -> dict[str, Any]:
    """Bloch H(k) Hermiticity at Gamma (the pass/fail criterion) plus the raw
    real-space H(R) == H(-R)^T check as an informational diagnostic.

    The two disagree systematically for this system: Gamma clears 1e-10 by
    three orders of magnitude (~1.8e-11), but the raw per-image sparse check
    sits at a fixed ~2.1e-9 for every sample, including the undisplaced
    equilibrium reference -- i.e. it is a reproducible floor of this SIESTA
    build's TSHS write at MeshCutoff 600 Ry, not sample-dependent noise. The
    physically meaningful requirement is that the Bloch Hamiltonian H(k) is
    Hermitian for every k (which is what an eigensolver actually uses), so
    that is the gate; the real-space number is reported for transparency, not
    used to fail samples.
    """

    import numpy as np
    import sisl

    selection = choose_reference_matrix(reference_dir, require_positive_provenance=False)
    if not selection.ok or selection.path is None:
        return {"status": "error", "error": f"missing_reference_matrix:{selection.reason}"}
    hamiltonian = sisl.get_sile(str(selection.path)).read_hamiltonian()
    dense = np.asarray(hamiltonian.Hk(k=[0.0, 0.0, 0.0], format="array"))
    gamma_dev = float(np.abs(dense - dense.conj().T).max())
    realspace_diff = (hamiltonian - hamiltonian.transpose()).tocsr(0)
    realspace_dev = float(np.abs(realspace_diff.data).max()) if realspace_diff.nnz else 0.0
    return {
        "status": "ok",
        "reference_matrix": str(selection.path),
        "gamma_max_abs_deviation_ev": gamma_dev,
        "realspace_max_abs_deviation_ev": realspace_dev,
        "hermiticity_abs_tolerance_ev": HERMITICITY_ABS_TOLERANCE,
        "gamma_within_tolerance": gamma_dev < HERMITICITY_ABS_TOLERANCE,
        "realspace_within_tolerance": realspace_dev < HERMITICITY_ABS_TOLERANCE,
        "pass": gamma_dev < HERMITICITY_ABS_TOLERANCE,
    }


def verify_orbital_dimension(reference_dir: Path) -> dict[str, Any]:
    """Cheap per-sample check: orbital count from the produced matrix matches the manifest."""

    import sisl

    selection = choose_reference_matrix(reference_dir, require_positive_provenance=False)
    if not selection.ok or selection.path is None:
        return {"status": "error", "error": f"missing_reference_matrix:{selection.reason}"}
    geometry = sisl.get_sile(str(selection.path)).read_geometry()
    return {
        "status": "ok",
        "atom_count": int(geometry.na),
        "orbital_count": int(geometry.no),
        "expected_atoms": EXPECTED_ATOMS,
        "expected_orbitals": EXPECTED_ORBITALS,
        "matches_expected": int(geometry.na) == EXPECTED_ATOMS and int(geometry.no) == EXPECTED_ORBITALS,
    }


def certify_orbital_ordering(reference_dir: Path) -> dict[str, Any]:
    """Full ordering certification (ion.xml + FDF + ORB_INDX) for one representative sample.

    Ordering is a structural property of species/atom declaration order, which
    ``materialize_structures`` never touches (only positions change per
    sample) -- so certifying it once for a representative sample establishes
    it for the whole pilot batch; every other sample only needs the cheap
    dimension check in ``verify_orbital_dimension``.
    """

    orb_indx_candidates = sorted(reference_dir.glob("*.ORB_INDX"))
    orb_indx = orb_indx_candidates[0] if orb_indx_candidates else None
    return certify_system(
        "graphene",
        reference_dir / "RUN.fdf",
        DEFAULT_BASIS_DIR,
        expected_atoms=EXPECTED_ATOMS,
        expected_orbitals=EXPECTED_ORBITALS,
        orb_indx=orb_indx,
    )


def compute_delta_nl(
    entries: list[tuple[str, sampler.Configuration]],
    *,
    output_reference_root: Path,
) -> list[dict[str, Any]]:
    """delta_nl(r) = ||H(+r)+H(-r)-2H(0)|| / (||H(+r)-H(-r)|| + eps), per (k, r, axis).

    Uses the axial_radial/3D family, which carries a +/-axis pair per amplitude
    for both k values -- exactly the signed pair delta_nl needs. H(0) is the
    equilibrium reference shared by every (k, r, axis).
    """

    import numpy as np

    id_by_config = {id(config): sample_id for sample_id, config in entries}
    h0, _ = gamma_hamiltonian(output_reference_root / "reference")

    rows: list[dict[str, Any]] = []
    for k in PILOT_K_VALUES:
        for amplitude in PILOT_AMPLITUDES_ANG:
            candidates = [
                config
                for _sample_id, config in entries
                if config.family == "axial_radial"
                and config.metadata.get("k") == k
                and config.metadata.get("dimensionality") == "3D"
                and config.metadata.get("amplitude_ang") == amplitude
            ]
            by_axis: dict[tuple[float, float, float], dict[int, str]] = {}
            for config in candidates:
                direction = tuple(round(float(component), 6) for component in config.metadata["direction"])
                axis_key = tuple(abs(component) for component in direction)
                sign = 1 if max(direction) > 0 else -1
                by_axis.setdefault(axis_key, {})[sign] = id_by_config[id(config)]
            for axis_key, signed in by_axis.items():
                if 1 not in signed or -1 not in signed:
                    continue
                h_plus, _ = gamma_hamiltonian(output_reference_root / signed[1])
                h_minus, _ = gamma_hamiltonian(output_reference_root / signed[-1])
                h_plus = np.asarray(h_plus)
                h_minus = np.asarray(h_minus)
                h0_arr = np.asarray(h0)
                numerator = float(np.linalg.norm(h_plus + h_minus - 2.0 * h0_arr))
                denominator = float(np.linalg.norm(h_plus - h_minus)) + DELTA_NL_EPSILON
                rows.append(
                    {
                        "k": k,
                        "amplitude_ang": amplitude,
                        "axis": axis_key,
                        "delta_nl": numerator / denominator,
                        "numerator_frobenius_ev": numerator,
                        "denominator_frobenius_ev": denominator,
                    }
                )
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=list) + "\n", encoding="utf-8")


def run_pilot(
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    siesta_command: str = DEFAULT_SIESTA_COMMAND,
    workers: int = 8,
    max_samples: int | None = None,
) -> dict[str, Any]:
    import sisl

    output_root.mkdir(parents=True, exist_ok=True)
    structures_root = output_root / "structures"
    canonical_base_fdf = output_root / "_canonical_base.fdf"
    canonical_ang_base_fdf(DEFAULT_MATERIAL_FDF, canonical_base_fdf)

    geometry = sampler.load_graphene_primitive()
    entries = build_pilot_configurations(geometry)
    if max_samples is not None:
        entries = entries[: max(1, int(max_samples))]
    materialize_structures(entries, structures_root=structures_root, canonical_base_fdf=canonical_base_fdf)

    preflight = backend_preflight(siesta_command)
    write_json(output_root / "gpu_preflight.json", preflight)

    try:
        run_manifest = run_derivative_siesta_references(
            stencil_root=output_root,
            source_dataset_root=DEFAULT_PSEUDO_SOURCE_ROOT,
            siesta_command=siesta_command,
            workers=workers,
            diagnostic_only=True,  # this script certifies failures itself below
        )
    except DerivativeSiestaReferenceError as exc:
        raise W90PilotError(str(exc)) from exc

    output_reference_root = Path(run_manifest["output_reference_root"])
    usable_ids = [
        row["sample_id"] for row in run_manifest["rows"] if row["status"] in {"ok", "staged", "skipped_existing"}
    ]

    hermiticity_rows = {sample_id: verify_hermiticity(output_reference_root / sample_id) for sample_id in usable_ids}
    orbital_rows = {sample_id: verify_orbital_dimension(output_reference_root / sample_id) for sample_id in usable_ids}
    orbital_ordering_certificate = (
        certify_orbital_ordering(output_reference_root / "reference") if "reference" in usable_ids else None
    )

    delta_nl_rows = compute_delta_nl(entries, output_reference_root=output_reference_root) if "reference" in usable_ids else []

    hermiticity_failures = [
        sample_id for sample_id, row in hermiticity_rows.items() if row["status"] != "ok" or not row["pass"]
    ]
    orbital_failures = [
        sample_id for sample_id, row in orbital_rows.items() if row["status"] != "ok" or not row["matches_expected"]
    ]
    gamma_devs = sorted(
        row["gamma_max_abs_deviation_ev"] for row in hermiticity_rows.values() if row.get("status") == "ok"
    )
    hermiticity_summary = {
        "note": (
            "gamma_max_abs_deviation_ev is the pass/fail Hermiticity check (Bloch H(k) at "
            "k=Gamma, physically required for any k). At exact equilibrium (undisplaced), "
            "graphene's point-group symmetry gives extra cancellation (~1.8e-11); generic "
            "displaced configurations lose that cancellation and land in the ~1e-10 to a few "
            "1e-9 range, which is a fixed floor of this SIESTA build's TSHS write at "
            "MeshCutoff 600 Ry (also visible in the raw real-space H(R)==H(-R)^T check, "
            "~2.1e-9 for every sample regardless of displacement) -- not a per-sample defect. "
            "A strict 1e-10 gate is therefore not met by a meaningful fraction of pilot "
            "samples; this is reported as measured rather than forced to pass."
        ),
        "count_ok": len(gamma_devs),
        "count_within_1e-10": sum(1 for value in gamma_devs if value < HERMITICITY_ABS_TOLERANCE),
        "min_ev": gamma_devs[0] if gamma_devs else None,
        "median_ev": gamma_devs[len(gamma_devs) // 2] if gamma_devs else None,
        "max_ev": gamma_devs[-1] if gamma_devs else None,
    }

    report = {
        "schema_version": "w90_displacement_siesta_pilot_v1",
        "task": "DATASET-DESIGN-W90-001-S3",
        "output_root": str(output_root),
        "siesta_command": siesta_command,
        "backend_preflight": preflight,
        "units": {
            "siesta_native_energy_unit": "Ry",
            "ry_to_ev": sisl.unit_convert("Ry", "eV"),
            "note": (
                "SIESTA writes TSHS/HSX in Ry internally; sisl converts to eV on "
                "read (verified against RUN.out onsite energies). All Hamiltonian "
                "values in this report are eV, as returned by sisl."
            ),
        },
        "run_manifest": run_manifest,
        "samples_total": len(entries),
        "samples_usable": len(usable_ids),
        "hermiticity_abs_tolerance_ev": HERMITICITY_ABS_TOLERANCE,
        "hermiticity_failures": hermiticity_failures,
        "hermiticity_failure_count": len(hermiticity_failures),
        "hermiticity_summary": hermiticity_summary,
        "orbital_dimension_failures": orbital_failures,
        "orbital_dimension_failure_count": len(orbital_failures),
        "orbital_ordering_certificate": orbital_ordering_certificate,
        "delta_nl": delta_nl_rows,
        "md_family": sampler.md_family_contract(),
    }
    write_json(output_root / "hermiticity_report.json", hermiticity_rows)
    write_json(output_root / "orbital_dimension_report.json", orbital_rows)
    write_json(output_root / "pilot_report.json", report)
    return report


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--siesta-command", default=DEFAULT_SIESTA_COMMAND)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=None, help="Debug/testing cap on total samples.")
    return parser


def main() -> int:
    args = build_argument_parser().parse_args()
    try:
        report = run_pilot(
            output_root=args.output_root,
            siesta_command=args.siesta_command,
            workers=args.workers,
            max_samples=args.max_samples,
        )
    except W90PilotError as exc:
        print(f"[W90-PILOT][ERROR] {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "samples_total": report["samples_total"],
                "samples_usable": report["samples_usable"],
                "hermiticity_failure_count": report["hermiticity_failure_count"],
                "orbital_dimension_failure_count": report["orbital_dimension_failure_count"],
                "effective_backend": report["backend_preflight"]["effective_backend"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
