#!/usr/bin/env python3
"""E-F_001-S48: Graph2Mat structure/H/derivative transferability across the local stacking
registries a relaxed MATBG cell would locally contain, as a bound before any quantitative
relaxed-MATBG claim.

``decide_matbg_relaxation_provider`` (S47) found NO-GO-10: no relaxation provider in this
environment is defensible for the production ``(31, 30)`` cell, so no relaxed MATBG geometry
exists and none is produced here either. Fase 11 of the roadmap still names one thing that
can be checked without a relaxed cell: "directional-derivative validation Graph2Mat en
muestras locales/sistemas pequenos representativos" -- whether the checkpoint's structure
handling, H prediction and directional-derivative machinery (GO-3) transfer across the local
registries relaxation itself redistributes area among.

Nam & Koshino (Phys. Rev. B 96, 075311, 2017) identify AA, AB and BA as exactly the stacking
registries a relaxed twisted bilayer locally interpolates between below ~2 deg twist (the
same citation S47 already used to reject every relaxation candidate that cannot represent
interlayer registry physics). This repository already has all three as independent
commensurate bilayer materials (``materials/bilayer_graphene_A{A,B}``,
``materials/bilayer_graphene_BA``). Certifying the checkpoint across those three is a real,
cheap test of local-environment transfer -- it is not a substitute for GO-10, which stays
blocked until a relaxed geometry with its own provenance exists and is compared against
SIESTA (S47 Sec. 6's acceptance criteria).

Three checks per material, every one reusing existing certified machinery rather than a new
one:

``structure``   :func:`orbital_contract.certify_systems` -- the basis/species/orbital
                contract the checkpoint was trained under, cross-checked against this
                material's own FDF/.ion.xml/ORB_INDX (C03).
``h``           the checkpoint's forward Hamiltonian prediction on this material's geometry
                is finite and produces a well-formed sparse ``(row, col, R)`` support --
                reusing :func:`quantify_checkpoint_derivative_error.load_model_and_batch` /
                ``model_predictions``. No SIESTA reference exists for AA/BA (only AB has a
                ``dHSdR.nc`` campaign, S34/S35), so this is a well-formedness check, not a
                DFT-accuracy one; the DFT-accuracy side is exactly what GO-10 additionally
                requires once a relaxed geometry exists.
``derivative``  GO-3 (:func:`validate_graph2mat_jvp.validate_directional_jvp`): the
                directional JVP equals the coordinate-JVP combination and the model's own
                frozen central difference, on this material's own topology, for the default
                direction set plus this material's stacking-specific directions (shear,
                layer breathing, intralayer control -- reused, unmodified, from
                :mod:`run_ab_bilayer_epc_paths`, which built them generically from atomic z,
                not from anything AB-specific).

Status label this ticket can license: ``local_stacking_transferability_bound``. Never
``relaxed_geometry_transferability`` and never a substitute for GO-10.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for _path in (SCRIPT_DIR, REPO_ROOT / "shared", REPO_ROOT / "scripts" / "torch_serialization_compat"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import fd_perturbation_space as fdp  # noqa: E402
import orbital_contract as oc  # noqa: E402
from decide_matbg_relaxation_provider import matbg_relaxation_provider_decision  # noqa: E402
from quantify_checkpoint_derivative_error import load_model_and_batch, model_predictions  # noqa: E402
from run_ab_bilayer_epc_paths import build_directions, layer_of_atom_from_positions  # noqa: E402

SCHEMA = "epc_relaxed_geometry_transferability_bound_v1"
TICKET = "E-F_001-S48"
MEMO = "docs/epc_s48_relaxed_geometry_transferability.md"
STATUS_LABEL = "local_stacking_transferability_bound"

DEFAULT_BASIS_DIR = REPO_ROOT / "materials/graphene_common/basis"
DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "Comparison/results/tbg_registry_spectral_loss/training/checkpoints"
    / "spectral-best-epoch=247-step=03472.ckpt"
)
DEFAULT_OUTPUT = REPO_ROOT / "Comparison/results/epc/relaxed_transferability/local_stacking_bound.json"

#: Nam & Koshino (2017): relaxation below ~2 deg twist redistributes area among exactly these
#: three registries. They are the "local samples...representativos" Fase 11 asks for in the
#: absence of a relaxed geometry (S47: NO-GO-10).
MATERIALS: tuple[dict[str, Any], ...] = (
    {
        "label": "bilayer_graphene_AA",
        "fdf": "materials/bilayer_graphene_AA/RUN.fdf",
        "basis_dir": "materials/graphene_common/basis",
        "expected_atoms": 4,
        "expected_orbitals": 16,
        "orb_indx_glob": "Comparison/datasets/bilayer_graphene_AA_md30/**/bilayer_graphene_AA.ORB_INDX",
    },
    {
        "label": "bilayer_graphene_AB",
        "fdf": "materials/bilayer_graphene_AB/RUN.fdf",
        "basis_dir": "materials/graphene_common/basis",
        "expected_atoms": 4,
        "expected_orbitals": 16,
        "orb_indx_glob": "Comparison/datasets/bilayer_graphene_AB_md30/**/bilayer_graphene_AB.ORB_INDX",
    },
    {
        "label": "bilayer_graphene_BA",
        "fdf": "materials/bilayer_graphene_BA/RUN.fdf",
        "basis_dir": "materials/graphene_common/basis",
        "expected_atoms": 4,
        "expected_orbitals": 16,
        "orb_indx_glob": "Comparison/datasets/bilayer_graphene_BA_md30/**/bilayer_graphene_BA.ORB_INDX",
    },
)


class RelaxedTransferabilityError(RuntimeError):
    """The bound cannot be computed as specified."""


def _check(name: str, passed: bool, detail: str, **extra: Any) -> dict[str, Any]:
    return {"check": name, "passed": bool(passed), "detail": detail, **extra}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, Path):
        return str(value)
    return value


def _material_directions(positions_ang: np.ndarray, cell_ang: np.ndarray, n_atoms: int) -> list[Any]:
    """The GO-3 default set plus this material's own stacking-specific patterns."""
    layer_of_atom = layer_of_atom_from_positions(positions_ang)
    stacking = build_directions(positions_ang, cell_ang, layer_of_atom)
    return [*fdp.default_direction_set(n_atoms, seed=0), *stacking.values()]


def certify_material(
    material: dict[str, Any],
    *,
    structure: dict[str, Any],
    checkpoint: Path,
    basis_dir: Path,
    backend: str,
) -> dict[str, Any]:
    """Structure (already certified), H well-formedness and GO-3 for one local sample."""
    import sisl
    import torch

    from graph2mat_autograd_derivatives import batch_topology_hash, resolve_jvp_backend
    import validate_graph2mat_jvp as vj

    fdf = REPO_ROOT / material["fdf"]
    geometry = sisl.get_sile(str(fdf)).read_geometry()
    positions_ang = np.asarray(geometry.xyz, dtype=np.float64)
    cell_ang = np.asarray(geometry.cell, dtype=np.float64)

    # oc.DECLARED_ONLY is not a failure: shared/run_inventory.py itself maps it to
    # PRESENT_UNVERIFIED, and every material already in production (graphene, AB,
    # the rigid TBG target) gets this same status against this checkpoint, from the
    # known checkpoint PointBasis.R / edge-cutoff drift, not from anything specific
    # to this material. Only oc.INVALID -- a real species/count/order mismatch --
    # blocks this check.
    structure_check = _check(
        "structure_orbital_contract",
        structure["status"] != oc.INVALID,
        f"{material['label']}: orbital contract status {structure['status']}",
        status=structure["status"],
    )

    model, batch, processor, provenance = load_model_and_batch(checkpoint, fdf, basis_dir=basis_dir)
    change_of_basis = np.asarray(processor.basis_table.change_of_basis, dtype=np.float64)
    model, batch = vj._cast(model, batch, torch.float64)
    model, batch, backend_record = resolve_jvp_backend(model, batch, backend, output_keys=vj.DEFAULT_OUTPUT_KEYS)
    topology = batch_topology_hash(batch)

    h_field, h_meta = model_predictions(model, batch, processor)
    values = np.array(list(h_field.values()), dtype=np.float64)
    h_check = _check(
        "h_prediction_finite_and_nonempty",
        bool(values.size > 0 and np.all(np.isfinite(values))),
        f"{material['label']}: {h_meta['nnz']} nonzeros, shape {h_meta['shape']}, all finite"
        if values.size > 0 and np.all(np.isfinite(values))
        else f"{material['label']}: prediction empty or non-finite",
        **h_meta,
    )

    n_atoms = int(batch["positions"].shape[0])
    radii = [float(np.max(entry.R)) for entry in processor.basis_table.basis]
    directions = _material_directions(positions_ang, cell_ang, n_atoms)
    go3 = vj.validate_directional_jvp(
        model,
        batch,
        directions=directions,
        deltas=fdp.delta_sweep(0.004, count=5),
        dtypes=("float64",),
        backends=(backend_record.effective,),
        change_of_basis=change_of_basis,
        data_processor=processor,
        cutoff_ang=[radii[int(point_type)] for point_type in batch["point_types"]],
        positions_ang=positions_ang,
        lattice_vectors_ang=cell_ang,
    )
    go3_check = _check(
        "go3_directional_jvp_internal_validation",
        bool(go3["summary"]["passed"]),
        (
            f"{material['label']}: JVP == coordinate combination == frozen central difference "
            f"on {len(directions)} directions (default set + this material's stacking patterns)"
            if go3["summary"]["passed"]
            else f"{material['label']}: GO-3 failed: {go3['summary']['checks_failed']}"
        ),
        directions=[direction.name for direction in directions],
    )

    checks = [structure_check, h_check, go3_check]
    return {
        "label": material["label"],
        "fdf": str(fdf),
        "checks": checks,
        "status": "PASS" if all(row["passed"] for row in checks) else "FAIL",
        "backend": backend_record.to_metadata(),
        "topology": topology,
        "model_provenance": provenance,
        "go3_report": go3,
    }


def certify(
    *,
    checkpoint: Path = DEFAULT_CHECKPOINT,
    basis_dir: Path = DEFAULT_BASIS_DIR,
    backend: str = "cuda",
    output: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    go10 = matbg_relaxation_provider_decision()
    structure_report = oc.certify_systems(systems=MATERIALS, checkpoint=checkpoint)
    structure_by_label = {row["label"]: row for row in structure_report["systems"]}
    missing = [material["label"] for material in MATERIALS if material["label"] not in structure_by_label]
    if missing:
        raise RelaxedTransferabilityError(f"orbital-contract certification produced no record for {missing}.")

    materials = [
        certify_material(
            material,
            structure=structure_by_label[material["label"]],
            checkpoint=checkpoint,
            basis_dir=basis_dir,
            backend=backend,
        )
        for material in MATERIALS
    ]
    all_pass = bool(materials) and all(entry["status"] == "PASS" for entry in materials)

    report = {
        "schema": SCHEMA,
        "ticket": TICKET,
        "memo": MEMO,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status_label": STATUS_LABEL,
        "structure_report_status": structure_report["status"],
        "materials": materials,
        "verdict": "PASS" if all_pass else "FAIL",
        "go10_status": go10["verdict"],
        "licenses_go10": False,
        "quantitative_relaxed_matbg_claim_licensed": False,
        "reason": (
            "structure, H well-formedness and GO-3 directional-derivative internal "
            "consistency transfer across AA/AB/BA -- the local registries a relaxed MATBG "
            "cell interpolates between (Nam & Koshino, Phys. Rev. B 96, 075311, 2017) -- but "
            "this is checkpoint self-consistency on existing small commensurate cells, not a "
            "model-vs-SIESTA accuracy comparison on an actual relaxed geometry; GO-10 stays "
            f"{go10['verdict']} (docs/epc_s47_matbg_relaxation_decision.md) and no "
            "quantitative relaxed-MATBG claim is licensed by this report"
        ),
        "references": [
            "N. N. T. Nam and M. Koshino, Phys. Rev. B 96, 075311 (2017) -- AA/AB/BA as the "
            "registries relaxation redistributes area among, the physical basis for treating "
            "them as local samples representative of a relaxed cell.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--basis-dir", type=Path, default=DEFAULT_BASIS_DIR)
    parser.add_argument(
        "--backend",
        choices=("cuda", "cpu"),
        default="cuda",
        help="requested JVP backend; a failed CUDA preflight falls back to CPU with a recorded reason",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    report = certify(checkpoint=args.checkpoint, basis_dir=args.basis_dir, backend=args.backend, output=args.output)
    for material in report["materials"]:
        print(f"[{TICKET}] {material['label']:<24} status={material['status']}")
        for row in material["checks"]:
            print(f"[{TICKET}]   {row['passed']!s:>5} {row['check']:<32} {row['detail'][:100]}")
    print(f"[{TICKET}] verdict: {report['verdict']}  (GO-10: {report['go10_status']})")
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
