#!/usr/bin/env python3
"""Close (or refuse to close) ``checkpoint_lineage`` and ``final_case_exclusion``.

Both gates are adjudicated from real artifacts only: the checkpoint file, the
frozen split manifest the trainer actually reads, and the declared EPC
evaluation structures. Nothing is inferred from a filename.

**Gate ``checkpoint_lineage``.** The checkpoint identified by the
preregistration is re-hashed, its selection metric checked validation-only,
and every one of the manifest's rows is linked individually: each row must
carry a ``sample_id``, a ``split``, a ``source_run``, a readable
``structure_path`` and a sha256 for every declared artifact, and every one of
those sha256 values is recomputed from the file on disk. A row that is
unreadable or that lacks provenance is counted as *unlinked* and can never be
counted as excluded. The manifest's own ``split_counts`` must reconcile with
the trainer's declared ``physical_train_samples``/``validation_samples``.

Additionally, the split is checked for *internal* consistency: a checkpoint
selected on a validation metric whose validation split contains rows that are
byte-identical to training rows was not selected on held-out data. This is a
lineage defect of the checkpoint itself and blocks the gate.

**Gate ``final_case_exclusion``.** Two distinct claims are computed and
reported separately, never merged:

``exact_exclusion``
    No evaluation structure is byte-identical (artifact sha256) or
    geometrically identical (coordinates within threshold) to any linked
    train/validation row. This is a strong claim and it is decided per row.

``family_exclusion``
    No evaluation structure shares the structural family (atom count and
    species multiset) of any linked train/validation row. This is a weaker
    claim: it says the two sets cannot be compared pairwise at all, which is
    why the earlier audit's ``pairs_checked = 0`` must never be read as
    ``exact_exclusion``.

Cases already opened during this investigation are marked
``previously_inspected = true`` and are explicitly denied blind-holdout
status: a structure inspected while tolerances were being set is not a blind
holdout, no matter what it is called afterwards.

No SIESTA is run, no dataset is modified, no threshold is set after seeing a
result.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "Comparison/scripts"), str(ROOT / "shared")]

from artifact_signature import file_sha256  # noqa: E402
from check_geometry_leakage import parse_run_fdf_geometry  # noqa: E402

SCHEMA = "checkpoint_lineage_and_final_case_exclusion_v1"
DEFAULT_PREREGISTRATION = ROOT / "docs/epc_preregistration_v2.json"
DEFAULT_OUTPUT = ROOT / "Comparison/results/epc/checkpoint_lineage_exclusion"
TRAINING_PLAN = ROOT / "Comparison/results/tbg_registry_spectral_loss/training_plan.json"

# Pre-registered, never tuned after seeing a result.
GEOMETRY_IDENTITY_ANG = 1e-4

# Artifacts every manifest row must carry a verifiable sha256 for. A row
# missing any of these has incomplete provenance and stays unlinked.
REQUIRED_ARTIFACTS = ("reference_tshs", "run_fdf", "orb_indx", "struct_out", "xv", "metadata")
REQUIRED_FIELDS = ("sample_id", "split", "source_run", "structure_path")

# Only these artifacts identify a *geometry*. ORB_INDX is a basis-orbital
# descriptor and metadata.json is bookkeeping: both repeat verbatim across rows
# that are physically different samples (ORB_INDX takes 2 distinct values over
# all 738 rows), so matching on them would report every row as a duplicate of
# every other. Identity claims use geometry-bearing artifacts only.
IDENTITY_ARTIFACTS = ("run_fdf", "xv", "struct_out", "reference_tshs")

# Splits that participate in selecting the checkpoint. The test split does not
# select anything, so it is linked and reported but is not a selection input.
SELECTION_SPLITS = ("train", "validation")

# Structures this investigation has already opened. certify_graph2mat_target_
# gauge_contract.py reads splits/train/0/RUN.fdf directly, and both EPC
# evaluation structures have been used repeatedly while tolerances were being
# fixed. Declared here rather than guessed, so the report can never call one of
# them a blind holdout retrospectively.
PREVIOUSLY_INSPECTED = {
    "graphene_primitive_cell_equilibrium": (
        "opened repeatedly during tolerance setting: certify_basis_response.py, "
        "run_epc_convergence_sweeps.py, quantify_checkpoint_derivative_error.py, "
        "run_epc_siesta_reference.py"
    ),
    "matbg_rigid_31_30_magic_angle": (
        "opened during the MATBG response and phonon-provider work "
        "(benchmark_matbg_synthetic_displacements.py, certify_matbg_phonon_provider.py)"
    ),
}
PREVIOUSLY_INSPECTED_ROWS = {
    "n474__bilayer_graphene_aa_md200__md_0": (
        "certify_graph2mat_target_gauge_contract.py reads this row's RUN.fdf "
        "(splits/train/0) as its TRAIN_RUN label case"
    ),
}


def _sha_or_none(path_text: str | None) -> str | None:
    return file_sha256(path_text) if path_text else None


def link_row(row: dict[str, Any]) -> dict[str, Any]:
    """Link one manifest row to real files, or say exactly why it cannot be."""
    problems: list[str] = []
    for field in REQUIRED_FIELDS:
        if not row.get(field):
            problems.append(f"missing field {field!r}")

    declared = row.get("artifact_sha256") or {}
    paths = row.get("artifact_paths") or {}
    verified: dict[str, str] = {}
    for artifact in REQUIRED_ARTIFACTS:
        want = declared.get(artifact)
        path = paths.get(artifact)
        if not want:
            problems.append(f"{artifact}: no declared sha256")
            continue
        if not path:
            problems.append(f"{artifact}: no declared path")
            continue
        got = _sha_or_none(path)
        if got is None:
            problems.append(f"{artifact}: file absent, sha256 unverifiable")
        elif got != want:
            problems.append(f"{artifact}: sha256 {got} != declared {want}")
        else:
            verified[artifact] = got

    structure_path = row.get("structure_path")
    atom_count: int | None = None
    species: list[str] = []
    coords: list[tuple[float, float, float]] = []
    if structure_path and Path(structure_path).is_file():
        try:
            species, coords = parse_run_fdf_geometry(Path(structure_path))
        except OSError as exc:
            problems.append(f"structure_path unreadable: {exc}")
        else:
            atom_count = len(coords) or None
            if atom_count is None:
                problems.append("structure_path parsed to zero atoms")
    else:
        problems.append("structure_path does not exist")

    return {
        "sample_id": row.get("sample_id"),
        "split": row.get("split"),
        "source_run": row.get("source_run"),
        "source_stacking": row.get("source_stacking"),
        "structure_path": structure_path,
        "atom_count": atom_count,
        "species_multiset": sorted(species),
        "verified_artifact_sha256": verified,
        "linked": not problems,
        "provenance_problems": problems,
        "previously_inspected": str(row.get("sample_id")) in PREVIOUSLY_INSPECTED_ROWS,
        "_coords": coords,
    }


def audit_lineage(manifest_path: Path, declared_sha: str | None) -> dict[str, Any]:
    actual_sha = _sha_or_none(str(manifest_path))
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = [link_row(row) for row in payload.get("rows") or []]

    unlinked = [row for row in rows if not row["linked"]]
    by_split: dict[str, int] = {}
    for row in rows:
        by_split[str(row["split"])] = by_split.get(str(row["split"]), 0) + 1

    findings: list[str] = []
    if actual_sha is None:
        findings.append(f"manifest {manifest_path} is unreadable")
    elif declared_sha and actual_sha != declared_sha:
        findings.append(f"manifest sha256 {actual_sha} != preregistered {declared_sha}")
    if unlinked:
        findings.append(
            f"{len(unlinked)} of {len(rows)} rows lack complete provenance and are NOT counted as excluded"
        )
    if by_split != payload.get("split_counts"):
        findings.append(f"row splits {by_split} disagree with declared split_counts {payload.get('split_counts')}")

    # The trainer's own declaration must reconcile with the manifest it reads.
    plan_reconciliation: dict[str, Any] = {}
    if TRAINING_PLAN.is_file():
        plan = json.loads(TRAINING_PLAN.read_text(encoding="utf-8"))
        plan_reconciliation = {
            "training_plan": str(TRAINING_PLAN),
            "training_plan_sha256": _sha_or_none(str(TRAINING_PLAN)),
            "physical_train_samples_declared": plan.get("physical_train_samples"),
            "validation_samples_declared": plan.get("validation_samples"),
            "manifest_train_rows": by_split.get("train"),
            "manifest_validation_rows": by_split.get("validation"),
            "checkpoint_monitor": plan.get("checkpoint_monitor"),
            "source_checkpoint": plan.get("source_checkpoint"),
        }
        if plan.get("physical_train_samples") != by_split.get("train"):
            findings.append(
                f"training_plan physical_train_samples={plan.get('physical_train_samples')} "
                f"!= manifest train rows {by_split.get('train')}"
            )
        if plan.get("validation_samples") != by_split.get("validation"):
            findings.append(
                f"training_plan validation_samples={plan.get('validation_samples')} "
                f"!= manifest validation rows {by_split.get('validation')}"
            )
    else:
        findings.append(f"training plan {TRAINING_PLAN} absent; trainer declaration unreconciled")

    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": actual_sha,
        "manifest_sha256_preregistered": declared_sha,
        "manifest_split_hash": payload.get("split_hash"),
        "manifest_valid_flag": payload.get("valid"),
        "declared_split_counts": payload.get("split_counts"),
        "rows_total": len(rows),
        "rows_linked_individually": len(rows) - len(unlinked),
        "rows_unlinked": len(unlinked),
        "unlinked_details": [
            {"sample_id": row["sample_id"], "split": row["split"], "problems": row["provenance_problems"]}
            for row in unlinked
        ],
        "artifact_sha256_verified_count": sum(len(row["verified_artifact_sha256"]) for row in rows),
        "rows_by_split": by_split,
        "training_plan_reconciliation": plan_reconciliation,
        "findings": findings,
        "_rows": rows,
    }


def selection_integrity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Is the validation split actually held out from the training split?

    A checkpoint selected on ``val_*`` whose validation rows are byte-identical
    to training rows was not selected on held-out data. Reported per row, not
    as a count.
    """
    train_by_sha: dict[str, list[str]] = {}
    for row in rows:
        if row["split"] != "train":
            continue
        for artifact in IDENTITY_ARTIFACTS:
            sha = row["verified_artifact_sha256"].get(artifact)
            if sha:
                train_by_sha.setdefault(f"{artifact}:{sha}", []).append(str(row["sample_id"]))

    collisions: list[dict[str, Any]] = []
    for row in rows:
        if row["split"] != "validation":
            continue
        hits = {
            artifact: train_by_sha[f"{artifact}:{sha}"]
            for artifact in IDENTITY_ARTIFACTS
            if (sha := row["verified_artifact_sha256"].get(artifact))
            and f"{artifact}:{sha}" in train_by_sha
        }
        if hits:
            collisions.append(
                {
                    "validation_sample_id": row["sample_id"],
                    "identical_train_sample_ids": sorted({s for ids in hits.values() for s in ids}),
                    "identical_artifacts": sorted(hits),
                }
            )
    return {
        "validation_rows_identical_to_train_rows": len(collisions),
        "collisions": collisions,
        "held_out_for_selection": not collisions,
    }


def _coords_identical(left, right) -> bool:
    if not left or not right or len(left) != len(right):
        return False
    return all(
        abs(a - b) <= GEOMETRY_IDENTITY_ANG
        for row_a, row_b in zip(left, right)
        for a, b in zip(row_a, row_b)
    )


def audit_exclusion(rows: list[dict[str, Any]], structures: list[dict[str, Any]]) -> dict[str, Any]:
    """Exact exclusion and family exclusion, computed and reported separately."""
    linked = [row for row in rows if row["linked"] and row["split"] in SELECTION_SPLITS]
    unlinked = [row for row in rows if not row["linked"] and row["split"] in SELECTION_SPLITS]

    per_structure: list[dict[str, Any]] = []
    for structure in structures:
        label = str(structure.get("label"))
        path = ROOT / str(structure.get("path"))
        actual = _sha_or_none(str(path))
        declared = structure.get("sha256")
        problems: list[str] = []
        if actual is None:
            problems.append(f"evaluation structure {path} is unreadable")
        elif declared and actual != declared:
            problems.append(f"sha256 {actual} != preregistered {declared}")

        species: list[str] = []
        coords: list[tuple[float, float, float]] = []
        if path.is_file():
            species, coords = parse_run_fdf_geometry(path)
        atom_count = len(coords) or None
        if atom_count is None:
            problems.append("evaluation structure parsed to zero atoms; exclusion unproven")

        # Exact exclusion, decided per row. The evaluation artifact is a
        # RUN.fdf, so byte identity is only meaningful against each row's own
        # run_fdf; geometry identity below covers the rest.
        sha_hits = [
            {"sample_id": row["sample_id"], "split": row["split"], "artifact": "run_fdf"}
            for row in linked
            if actual is not None and row["verified_artifact_sha256"].get("run_fdf") == actual
        ]
        geometry_hits = [
            {"sample_id": row["sample_id"], "split": row["split"]}
            for row in linked
            if _coords_identical(coords, row["_coords"])
        ]
        # Family exclusion: same atom count AND same species multiset.
        family_hits = [
            {"sample_id": row["sample_id"], "split": row["split"], "atom_count": row["atom_count"]}
            for row in linked
            if row["atom_count"] == atom_count and row["species_multiset"] == sorted(species)
        ]

        exact_ok = not problems and not sha_hits and not geometry_hits
        per_structure.append(
            {
                "label": label,
                "path": str(path),
                "sha256": actual,
                "sha256_preregistered": declared,
                "atom_count": atom_count,
                "species_multiset": sorted(set(species)),
                "rows_compared_for_exact_exclusion": len(linked),
                "rows_not_compared_because_unlinked": len(unlinked),
                "exact_exclusion": {
                    "claim": "no linked train/validation row is byte-identical or geometrically identical",
                    "artifact_sha256_collisions": sha_hits,
                    "geometry_identity_collisions": geometry_hits,
                    "geometry_identity_threshold_ang": GEOMETRY_IDENTITY_ANG,
                    "holds": exact_ok,
                },
                "family_exclusion": {
                    "claim": "no linked train/validation row shares atom count and species multiset",
                    "note": (
                        "STRICTLY WEAKER than exact_exclusion. When this holds, the two sets "
                        "cannot be compared pairwise at all, so a pairwise leakage check "
                        "reporting zero pairs examined is evidence of family separation ONLY "
                        "and must never be reported as exact exclusion."
                    ),
                    "same_family_rows": len(family_hits),
                    "examples": family_hits[:5],
                    "holds": not family_hits and atom_count is not None,
                },
                "previously_inspected": label in PREVIOUSLY_INSPECTED,
                "previously_inspected_reason": PREVIOUSLY_INSPECTED.get(label),
                "blind_holdout": False,
                "blind_holdout_reason": (
                    "inspected during tolerance setting; cannot be relabelled a blind holdout"
                    if label in PREVIOUSLY_INSPECTED
                    else "not declared as a pre-registered blind holdout"
                ),
                "problems": problems,
            }
        )

    exact_all = bool(per_structure) and all(item["exact_exclusion"]["holds"] for item in per_structure)
    family_all = bool(per_structure) and all(item["family_exclusion"]["holds"] for item in per_structure)
    return {
        "linked_rows_used": len(linked),
        "unlinked_rows_excluded_from_the_claim": len(unlinked),
        "per_structure": per_structure,
        "exact_exclusion_holds_for_all": exact_all,
        "family_exclusion_holds_for_all": family_all,
        "distinction": (
            "exact_exclusion and family_exclusion are separate claims. family_exclusion "
            "holding does not imply the evaluation structures were ever compared to a "
            "training row; it implies they could not be."
        ),
    }


def build_report(preregistration: Path) -> dict[str, Any]:
    doc = json.loads(preregistration.read_text(encoding="utf-8"))
    entry = (doc.get("checkpoint_lineage") or [{}])[0]

    checkpoint_path = ROOT / str(entry.get("checkpoint_path"))
    checkpoint_sha = _sha_or_none(str(checkpoint_path))
    declared_ckpt_sha = entry.get("checkpoint_sha256")
    selection = entry.get("checkpoint_selection") or {}
    metric = str(selection.get("monitor_metric") or "")

    lineage_findings: list[str] = []
    if checkpoint_sha is None:
        lineage_findings.append(f"checkpoint {checkpoint_path} is absent; lineage unverifiable")
    elif declared_ckpt_sha and checkpoint_sha != declared_ckpt_sha:
        lineage_findings.append(f"checkpoint sha256 {checkpoint_sha} != preregistered {declared_ckpt_sha}")
    lowered = metric.lower()
    if not metric:
        lineage_findings.append("checkpoint_selection.monitor_metric is not declared")
    elif "test" in lowered or "holdout" in lowered:
        lineage_findings.append(f"selection metric {metric!r} references test/holdout data")
    elif "val" not in lowered:
        lineage_findings.append(f"selection metric {metric!r} cannot be confirmed validation-only")

    # The manifest split_paths() actually reads.
    manifests = entry.get("known_training_manifests") or []
    primary = next(
        (m for m in manifests if "tbg_md_plus_registry/n558" in str(m.get("path"))),
        manifests[0] if manifests else {},
    )
    manifest_path = ROOT / str(primary.get("path"))
    lineage = audit_lineage(manifest_path, primary.get("sha256"))
    rows = lineage.pop("_rows")

    integrity = selection_integrity(rows)
    if not integrity["held_out_for_selection"]:
        lineage_findings.append(
            f"{integrity['validation_rows_identical_to_train_rows']} validation rows are "
            f"byte-identical to training rows; the checkpoint's {metric!r} selection metric "
            "was not evaluated on data held out from training"
        )
    lineage_findings.extend(lineage["findings"])

    # The upstream checkpoint this one was fine-tuned from is not re-traced.
    upstream = entry.get("upstream_checkpoint") or {}
    upstream_sha = _sha_or_none(str(ROOT / str(upstream.get("path")))) if upstream.get("path") else None

    exclusion = audit_exclusion(rows, entry.get("epc_evaluation_structures") or [])

    lineage_verdict = "PASS" if not lineage_findings else "NO_GO"
    if lineage["rows_unlinked"] and lineage_verdict == "PASS":
        lineage_verdict = "UNKNOWN"

    if exclusion["per_structure"] and any(item["problems"] for item in exclusion["per_structure"]):
        exclusion_verdict = "UNKNOWN"
    elif exclusion["unlinked_rows_excluded_from_the_claim"]:
        exclusion_verdict = "UNKNOWN"
    elif exclusion["exact_exclusion_holds_for_all"]:
        exclusion_verdict = "PASS"
    else:
        exclusion_verdict = "NO_GO"

    return {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "gates": {
            "checkpoint_lineage": lineage_verdict,
            "final_case_exclusion": exclusion_verdict,
        },
        "preregistration": {
            "path": str(preregistration),
            "sha256": _sha_or_none(str(preregistration)),
            "content_sha256_declared": doc.get("content_sha256"),
        },
        "checkpoint": {
            "checkpoint_id": entry.get("checkpoint_id"),
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha,
            "sha256_preregistered": declared_ckpt_sha,
            "selection_metric": metric,
            "selection_metric_source": selection.get("monitor_source"),
            "upstream_checkpoint_path": upstream.get("path"),
            "upstream_checkpoint_sha256": upstream_sha,
            "upstream_lineage": "NOT_RE_TRACED",
        },
        "checkpoint_lineage": {
            "verdict": lineage_verdict,
            **lineage,
            "selection_split_integrity": integrity,
            "findings": lineage_findings,
        },
        "final_case_exclusion": {"verdict": exclusion_verdict, **exclusion},
        "thresholds": {
            "geometry_identity_ang": GEOMETRY_IDENTITY_ANG,
            "required_row_fields": list(REQUIRED_FIELDS),
            "required_row_artifacts": list(REQUIRED_ARTIFACTS),
            "note": "pre-registered before execution; not adjusted after seeing any result",
        },
        "preserved_verdicts": {
            "graph2mat_target_gauge_contract": "PASS",
            "gauge_derivative_audit": "PASS",
            "numerical_PAO_convergence": "PASS",
            "delta_out_closure": "NO_GO",
            "full_KS": "BLOCKED",
            "full_basis_connection_v1": "NO_GO",
            "onlyS_semantic_preflight_v1": "NO_GO",
            "gate_1_support_structure": "PASS",
        },
        "numerical_convergence_margin": {
            "limit_rms_mev_per_ang": 5.0,
            "scf_rms_mev_per_ang": 1.4568,
            "mesh_rms_mev_per_ang": 4.4551,
            "egg_box_rms_mev_per_ang": 4.8305,
            "statement": (
                "converged within the established budget, NOT with a large margin: the "
                "mesh (4.455) and egg-box (4.830) stages sit close to the 5 meV/Ang limit. "
                "SCF (1.457) is comfortable. Thresholds unchanged."
            ),
            "source": "Comparison/results/epc/pao_numerical_convergence_v2/pao_numerical_convergence_v2.json",
        },
        "gauge_constraint": {
            "graph2mat_target": "H_G2M = H_abs - E_F(R).S",
            "primary_pao_reference": "K_ref = H_abs - c_vac(R).S",
            "d_cvac_minus_dEf_ev_per_ang": 0.0034869062475910892,
            "d_cvac_minus_dEf_mev_per_ang": 3.4869062475910892,
            "source": "Comparison/results/epc/pao_flow_audit/graph2mat_target_gauge_contract.json",
            "constraint": (
                "gauge_derivative_audit = PASS proves the conversion is known; it does NOT "
                "license comparing the two matrices as they leave their respective routes. "
                "No final residual or fine_tuning_candidate may be computed by directly "
                "comparing quantities in these two gauges: 3.4869 meV/Ang is not negligible "
                "against the 5 meV/Ang convergence budget."
            ),
        },
        "scope": (
            "checkpoint_lineage and final_case_exclusion only. No SIESTA was run, no new "
            "basis opened, no dataset or state.db modified. Delta_out, q != 0, MATBG "
            "campaigns, GO-5 and GO-7 are untouched and their verdicts stand."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, default=DEFAULT_PREREGISTRATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    report = build_report(args.preregistration)
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "checkpoint_lineage_exclusion.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lineage = report["checkpoint_lineage"]
    print(f"checkpoint_lineage    = {report['gates']['checkpoint_lineage']}")
    print(f"final_case_exclusion  = {report['gates']['final_case_exclusion']}")
    print(
        f"rows linked individually: {lineage['rows_linked_individually']}/{lineage['rows_total']} "
        f"({lineage['artifact_sha256_verified_count']} artifact sha256 verified), "
        f"unlinked: {lineage['rows_unlinked']}"
    )
    for finding in lineage["findings"]:
        print(f"  lineage: {finding}")
    for item in report["final_case_exclusion"]["per_structure"]:
        print(
            f"  {item['label']}: exact={item['exact_exclusion']['holds']} "
            f"family={item['family_exclusion']['holds']} "
            f"previously_inspected={item['previously_inspected']}"
        )
    print(f"-> {path}")
    return 0 if all(v == "PASS" for v in report["gates"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
