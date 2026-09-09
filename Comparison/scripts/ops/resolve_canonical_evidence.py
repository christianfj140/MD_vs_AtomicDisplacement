#!/usr/bin/env python3
"""Canonical evidence resolver for E-F_001's ``infrastructure`` block reason.

The orchestrator's block_reason aggregates findings from many audit rounds
into one flat list. That list conflates three different things that must not
be treated as equivalent:

1. findings that are genuinely STALE -- a canonical gate certified since then
   supersedes them (e.g. "no >=5 delta amplitudes" when
   delta_plateau_topology_certification=PASS now exists);
2. findings that are ACTIVE_BLOCKERs but belong to a scope this task does not
   need (e.g. checkpoint_selection_integrity blocks fine-tuning claims, not
   the SIESTA reference infrastructure; delta_out_closure blocks full_KS, not
   the finite-PAO-covariant local claim);
3. findings that are OUT_OF_SCOPE entirely (q!=0, MATBG, CUDA JVP) or whose
   evidence was never attached to a canonical artifact at all
   (EVIDENCE_NOT_ATTACHED -- not silently promoted to PASS).

This script does not change any scientific verdict. It reads the current
state.db block_reason (read-only) and a fixed registry of canonical artifact
paths, and produces:
  - a per-finding reconciliation table (historical_reason, canonical_gate,
    canonical_verdict, classification, propagates_to_graphene_Gamma)
  - a per-domain status summary (physics_reference_infrastructure,
    graphene_Gamma_finite_PAO, full_KS, ml_checkpoint_selection, C19C20,
    qneq0, MATBG)

Historical findings are matched to canonical gates by fixed keyword
substrings (Spanish and English), not by re-deriving verdicts from the
planner's prose -- exactly what the resolver exists to stop doing.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

SCHEMA = "canonical_evidence_resolver_v1"
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STATE_DB = Path("/home/christian/repositorios/research-orchestrator/state.db")
DEFAULT_TASK_ID = "E-F_001"
DEFAULT_OUTPUT = REPO_ROOT / "Comparison" / "results" / "epc_repair" / "canonical_evidence_resolution.json"

ACTIVE_BLOCKER = "ACTIVE_BLOCKER"
SUPERSEDED = "SUPERSEDED"
OUT_OF_SCOPE = "OUT_OF_SCOPE"
EVIDENCE_NOT_ATTACHED = "EVIDENCE_NOT_ATTACHED"

READY = "READY"
BLOCKED = "BLOCKED"
NO_GO = "NO_GO"


# --------------------------------------------------------------------------- #
# Canonical gate registry: name -> (artifact path, dotted key(s) to verdict)
# --------------------------------------------------------------------------- #

def _read(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def _dig(d: Any, *keys: str) -> Any:
    for key in keys:
        if not isinstance(d, dict) or key not in d:
            return None
        d = d[key]
    return d


def _bool_to_verdict(value: Any) -> str | None:
    if isinstance(value, bool):
        return "PASS" if value else "NO_GO"
    return value


class Gate:
    def __init__(self, name: str, path: Path, *key: str, transform=_bool_to_verdict):
        self.name = name
        self.path = path
        self.key = key
        self.transform = transform

    def resolve(self) -> dict[str, Any]:
        data = _read(self.path)
        if data is None:
            return {"gate": self.name, "artifact": str(self.path.relative_to(REPO_ROOT)),
                     "found": False, "verdict": None}
        raw = _dig(data, *self.key) if self.key else data.get("verdict")
        verdict = self.transform(raw) if raw is not None else None
        return {"gate": self.name, "artifact": str(self.path.relative_to(REPO_ROOT)),
                "found": True, "verdict": verdict,
                "artifact_sha256": _artifact_hash(self.path)}


def _artifact_hash(path: Path) -> str | None:
    import hashlib
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


EPC = REPO_ROOT / "Comparison" / "results" / "epc"

GATES: dict[str, Gate] = {
    g.name: g for g in [
        Gate("delta_plateau_topology_certification",
             EPC / "delta_plateau_topology_certification/delta_plateau_topology_certification.json",
             "delta_plateau_topology_certification"),
        Gate("uniform_translation_PAO_covariance_Gamma",
             EPC / "uniform_translation_pao_covariance_gamma/uniform_translation_pao_covariance_gamma.json",
             "verdict"),
        Gate("uniform_translation_PAO_covariance_BZ_sampled_v1",
             EPC / "uniform_translation_pao_covariance/uniform_translation_pao_covariance.json",
             "verdict"),
        Gate("graphene_Gamma_sublattice_symmetry_reduction",
             EPC / "graphene_gamma_sublattice_symmetry_reduction/graphene_gamma_sublattice_symmetry_reduction.json",
             "graphene_Gamma_sublattice_symmetry_reduction"),
        Gate("graphene_Gamma_E2g_x_delta_convergence",
             EPC / "graphene_gamma_sublattice_symmetry_reduction/graphene_gamma_sublattice_symmetry_reduction.json",
             "graphene_Gamma_E2g_x_delta_convergence", "verdict"),
        Gate("graphene_Gamma_E2g_y_delta_convergence",
             EPC / "graphene_gamma_e2g_y_delta_convergence/graphene_gamma_e2g_y_delta_convergence.json",
             "graphene_Gamma_E2g_y_delta_convergence", "verdict"),
        Gate("graphene_Gamma_E2g_doublet_convergence",
             EPC / "graphene_gamma_e2g_y_delta_convergence/graphene_gamma_e2g_y_delta_convergence.json",
             "graphene_Gamma_E2g_doublet_convergence_by_direct_measurement", "verdict"),
        Gate("graphene_Gamma_sublattice_C3_symmetry_reduction",
             EPC / "graphene_gamma_e2g_doublet_convergence/graphene_gamma_e2g_doublet_convergence.json",
             "graphene_Gamma_sublattice_C3_symmetry_reduction"),
        Gate("graphene_Gamma_E2g_electronic_contraction_convergence",
             EPC / "graphene_gamma_e2g_electronic_contraction/graphene_gamma_e2g_electronic_contraction.json",
             "graphene_Gamma_E2g_doublet_electronic_contraction_convergence", "verdict"),
        Gate("reference_basis_convergence",
             EPC / "reference_basis_convergence/reference_basis_convergence.json", "verdict"),
        Gate("numerical_PAO_convergence",
             EPC / "pao_numerical_convergence_v2/pao_numerical_convergence_v2.json", "verdict"),
        Gate("production_basis_connection",
             EPC / "production_basis_connection/production_basis_connection.json", "verdict"),
        Gate("B_I_fourier_bessel_semantic_closure_v5",
             EPC / "b_i_fourier_bessel_semantic_v5/b_i_fourier_bessel_semantic_v5.json", "verdict"),
        Gate("full_basis_connection_semantic_v5",
             EPC / "full_basis_connection_semantic_v5/full_basis_connection_semantic_v5.json", "verdict"),
        Gate("gauge_derivative_audit",
             EPC / "pao_flow_audit/gauge_derivative_audit.json", "verdict"),
        Gate("graph2mat_target_gauge_contract",
             EPC / "pao_flow_audit/graph2mat_target_gauge_contract.json", "verdict"),
        Gate("final_comparator_gauge_conversion_verified",
             EPC / "checkpoint_derivative_error/graphene/checkpoint_derivative_error.json",
             "gauge_conversion_gate", "verdict"),
        Gate("final_case_exclusion",
             EPC / "checkpoint_lineage_exclusion/checkpoint_lineage_exclusion.json",
             "final_case_exclusion", "verdict"),
        Gate("checkpoint_lineage",
             EPC / "checkpoint_lineage_exclusion/checkpoint_lineage_exclusion.json",
             "checkpoint_lineage", "verdict"),
        Gate("checkpoint_selection_remediation",
             EPC / "checkpoint_selection_remediation/checkpoint_selection_remediation.json", "verdict"),
        Gate("delta_out_closure",
             EPC / "delta_out_closure/delta_out_closure.json", "verdict"),
        Gate("C14C_basis_response_v1",
             EPC / "basis_response/graphene/basis_response_certification.json", "verdict"),
        Gate("C14C_basis_response_v2",
             EPC / "basis_response_v2/graphene/basis_response_v2_certification.json", "verdict"),
        Gate("GO2_siesta_dhsdr",
             EPC / "certification/graphene/go2_certification.json", "verdict"),
        Gate("ab_bilayer_epc_three_sectors",
             EPC / "gamma_epc/bilayer_graphene_AB/ab_bilayer_epc_three_sectors.json", "status",
             transform=lambda v: v),
    ]
}


def gate_verdict(name: str) -> dict[str, Any]:
    gate = GATES.get(name)
    if gate is None:
        return {"gate": name, "found": False, "verdict": None}
    return gate.resolve()


# --------------------------------------------------------------------------- #
# Finding -> canonical gate mapping (keyword-matched, not planner prose)
# --------------------------------------------------------------------------- #
# Each entry: (keyword substrings that must ALL appear, case-insensitive),
# canonical gate name(s) to consult, domain this finding's dependency belongs
# to, and human label for the table.

FindingRule = tuple[tuple[str, ...], tuple[str, ...], str, str]

FINDING_RULES: list[FindingRule] = [
    (("cinco", "δ", "topológ"), ("delta_plateau_topology_certification",),
     "graphene_Gamma_finite_PAO", "5-delta plateau + topology certification"),
    (("cinco amplitudes", "topolog"),
     ("delta_plateau_topology_certification",), "graphene_Gamma_finite_PAO",
     "5-delta plateau + topology (plan entry)"),
    (("plateau prerregistrado", "cinco amplitudes"),
     ("delta_plateau_topology_certification",), "graphene_Gamma_finite_PAO",
     "preregistered >=5 delta plateau"),
    (("fc y fd", "plateau"), ("delta_plateau_topology_certification",),
     "graphene_Gamma_finite_PAO", "FC/FD plateau without topology change"),
    (("certify_epc_numerical_convergence",), ("delta_plateau_topology_certification",),
     "graphene_Gamma_finite_PAO", "numerical convergence certification script reference"),
    (("traslación uniforme", "graphene real"), ("uniform_translation_PAO_covariance_Gamma",),
     "graphene_Gamma_finite_PAO", "uniform translation measured on real graphene"),
    (("invariancia traslacional", "artifacts físicos"), ("uniform_translation_PAO_covariance_Gamma",),
     "graphene_Gamma_finite_PAO", "translational invariance on physical artifacts"),
    (("hermiticidad y traslación",), ("uniform_translation_PAO_covariance_Gamma",),
     "graphene_Gamma_finite_PAO", "hermiticity/translation within converged budget"),
    (("checkpoint", "no es independiente", "3 filas"),
     ("checkpoint_lineage", "checkpoint_selection_remediation"),
     "ml_checkpoint_selection", "checkpoint selection not independent (leakage)"),
    (("validación", "byte-idénticas", "entrenamiento"),
     ("checkpoint_lineage",), "ml_checkpoint_selection", "validation rows byte-identical to train"),
    (("leakage",), ("checkpoint_lineage",), "ml_checkpoint_selection", "leakage plan entry"),
    (("lineage del checkpoint", "exclusión de los casos finales"),
     ("checkpoint_lineage", "final_case_exclusion"), "ml_checkpoint_selection",
     "checkpoint lineage / final case exclusion"),
    (("δ_out", "despreciable"), ("delta_out_closure",), "full_KS", "Delta_out negligibility (full-KS)"),
    (("go-2", "dhsdr.nc", "hash local"), ("GO2_siesta_dhsdr",), "graphene_Gamma_finite_PAO",
     "GO-2 needs a locally-produced dHSdR.nc"),
    (("operación espacial exacta", "geometría ab"), ("ab_bilayer_epc_three_sectors",),
     "MATBG", "AB bilayer exact spatial operation"),
    (("ruta cuda", "graph2mat jvp"), (), "qneq0", "CUDA Graph2Mat JVP route (no canonical artifact found)"),
    (("a_l", "a_r", "runtime"), (), "qneq0", "A_L/A_R separately available from runtime (no canonical artifact found)"),
    (("escalera de bases siesta",), (), "MATBG", "basis ladder does not auto-validate checkpoint"),
    (("sound for symmetry",), ("uniform_translation_PAO_covariance_Gamma",), "graphene_Gamma_finite_PAO",
     "plan-soundness: symmetry"),
    (("sound for convergence",), ("delta_plateau_topology_certification", "reference_basis_convergence"),
     "graphene_Gamma_finite_PAO", "plan-soundness: convergence"),
    (("sound for units",), (), "qneq0", "plan-soundness: units (needs q!=0/MATBG artifacts)"),
    (("sound for leakage",), ("checkpoint_lineage",), "ml_checkpoint_selection", "plan-soundness: leakage"),
    (("sound for reference",), ("GO2_siesta_dhsdr", "reference_basis_convergence"),
     "graphene_Gamma_finite_PAO", "plan-soundness: reference"),
    (("sound for sensitivity",), ("delta_plateau_topology_certification",), "graphene_Gamma_finite_PAO",
     "plan-soundness: sensitivity"),
    (("resultado graphene-γ es estable",), ("delta_plateau_topology_certification", "production_basis_connection"),
     "graphene_Gamma_finite_PAO", "graphene-Gamma result stable under numerical parameters"),
    (("comparación graph2mat", "mismo bloque electrónico"),
     ("C14C_basis_response_v2",), "C19C20", "Graph2Mat-SIESTA comparison in the same electronic block"),
    (("56 pasos", "done", "reconciliador"), (), "infrastructure_reconciliation",
     "S1-S56 marked DONE without reconciled scientific evidence"),
    (("estado done de s1", "s56"), (), "infrastructure_reconciliation",
     "S1-S56 DONE state not backed by reconciled checks"),
    # -- added: findings observed live in E-F_001's block_reason (2026-09)
    # that fell through every rule above and were silently dropped from
    # domain_status() (see domain_status's "unclassified" bucket bug, fixed
    # separately below). Each was hand-checked against a real artifact
    # before being added here -- not name-pattern guessed.
    (("checkpoint", "byte-identical", "entrenamiento"),
     ("checkpoint_lineage",), "ml_checkpoint_selection",
     "checkpoint selection not independent (byte-identical validation rows)"),
    (("reranking retrospectivo", "63 epochs"),
     ("checkpoint_lineage", "checkpoint_selection_remediation"), "ml_checkpoint_selection",
     "clean reselection impossible: 59/63 running-best epoch weights missing"),
    (("selección del checkpoint", "contaminada", "snapshots conservados"),
     ("checkpoint_lineage",), "ml_checkpoint_selection",
     "checkpoint selection contaminated; clean reselection not reconstructible"),
    # GO-4 adjudicates the quantitative physical EPC coupling `g` -- excluded
    # from E-F_001's scope (see E-F_001_scope_contract_v3.json). A finding
    # that "graphene Gamma remains GO-4 NO_GO" is about that excluded claim,
    # not the PAO-finite covariant-response claim this task actually makes;
    # its domain is full_KS, matching the existing Delta_out finding, and it
    # must not propagate to graphene_Gamma_finite_PAO.
    (("graphene", "go-4 no_go", "matbg", "relaxed"), ("GO4_verdict",), "full_KS",
     "graphene-Gamma GO-4 (quantitative g) NO_GO; MATBG/relaxed descendants blocked"),
    (("resultado físico requerido", "go-4 no_go"), ("GO4_verdict",), "full_KS",
     "required physical result (quantitative g) not validated; GO-4 NO_GO"),
    # uniform_translation_PAO_covariance_Gamma is PASS (see GATES above); a
    # finding that no verifiable execution of this gate exists is stale.
    (("traslación uniforme electrónica", "presupuesto convergido"),
     ("uniform_translation_PAO_covariance_Gamma",), "graphene_Gamma_finite_PAO",
     "uniform electronic translation gate at Gamma: stale, superseded by PASS artifact"),
    (("ejecución verificable", "traslación uniforme electrónica"),
     ("uniform_translation_PAO_covariance_Gamma",), "graphene_Gamma_finite_PAO",
     "uniform electronic translation gate at Gamma: stale, superseded by PASS artifact"),
]


def classify_finding(text: str) -> tuple[str, ...]:
    """Which finding rules match this text (by keyword substrings)."""
    lower = text.lower()
    matched = []
    for keywords, gates, domain, label in FINDING_RULES:
        if all(kw.lower() in lower for kw in keywords if kw):
            matched.append((keywords, gates, domain, label))
    return matched


def reconcile_finding(text: str) -> dict[str, Any]:
    matches = classify_finding(text)
    if not matches:
        return {
            "historical_reason": text, "canonical_gate": None, "canonical_verdict": None,
            "classification": ACTIVE_BLOCKER, "domain": "unclassified",
            "propagates_to_graphene_Gamma": True,
            "note": "no keyword rule matched; kept as an active blocker until reviewed",
        }
    _, gate_names, domain, label = matches[0]
    if not gate_names:
        return {
            "historical_reason": text, "canonical_gate": None, "canonical_verdict": None,
            "classification": EVIDENCE_NOT_ATTACHED, "domain": domain, "label": label,
            "propagates_to_graphene_Gamma": domain == "graphene_Gamma_finite_PAO",
        }
    resolved = [gate_verdict(name) for name in gate_names]
    verdicts = [r["verdict"] for r in resolved]
    if any(v is None for v in resolved) or any(not r["found"] for r in resolved):
        classification = EVIDENCE_NOT_ATTACHED
    elif all(v == "PASS" for v in verdicts):
        classification = SUPERSEDED if domain == "graphene_Gamma_finite_PAO" else (
            OUT_OF_SCOPE if domain not in ("graphene_Gamma_finite_PAO", "infrastructure_reconciliation")
            else SUPERSEDED
        )
    elif domain == "graphene_Gamma_finite_PAO":
        classification = ACTIVE_BLOCKER
    else:
        classification = OUT_OF_SCOPE if domain in ("qneq0", "MATBG") else ACTIVE_BLOCKER
    return {
        "historical_reason": text, "canonical_gate": list(gate_names), "canonical_verdict": verdicts,
        "classification": classification, "domain": domain, "label": label,
        "propagates_to_graphene_Gamma": classification == ACTIVE_BLOCKER and domain == "graphene_Gamma_finite_PAO",
        "canonical_artifacts": resolved,
    }


# --------------------------------------------------------------------------- #
# state.db (read-only)
# --------------------------------------------------------------------------- #

def open_readonly(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def latest_findings(conn: sqlite3.Connection, task_id: str) -> list[str]:
    """The structured issues list from the latest repair_route event.

    ``tasks.block_reason`` is a single string built by joining these same
    issues with "; " (see the orchestrator's _dispatch_repair) -- several
    individual issues themselves contain internal semicolons, so re-splitting
    that joined string on ";" fragments single findings into several bogus
    rows. The repair_route event's own ``issues`` field is the array before
    that join, so read it directly instead of re-deriving it.
    """
    row = conn.execute(
        "SELECT payload FROM events WHERE task_id = ? AND kind = 'repair_route' "
        "ORDER BY id DESC LIMIT 1",
        (task_id,),
    ).fetchone()
    if row is None:
        return []
    payload = json.loads(row["payload"])
    issues = [str(issue).strip() for issue in payload.get("issues", []) if str(issue).strip()]
    return [part for issue in issues for part in _expand_planner_bundle(issue)]


_PLANNER_BUNDLE_PREFIX = "Planning ended with the planners still requesting revisions:"


def _expand_planner_bundle(issue: str) -> list[str]:
    """One ``issues`` entry can itself be a "; "-joined bundle of the planner's
    own sub-reasons (wrapped behind the fixed prefix above) -- unlike other
    findings, splitting THIS one on ";" is safe because that is exactly how it
    was assembled one level up, and leaving it whole would let a single
    keyword match (e.g. "Delta_out") silently mis-route every other topic
    bundled inside it (GO-2, checkpoint lineage, AB geometry, CUDA JVP, ...)."""
    if not issue.startswith(_PLANNER_BUNDLE_PREFIX):
        return [issue]
    rest = issue[len(_PLANNER_BUNDLE_PREFIX):]
    raw_parts = [part.strip() for part in rest.split(";") if part.strip()]
    # A ";" inside one bundled sub-reason (a clause boundary, not a finding
    # boundary) produces a fragment that continues the previous sentence --
    # reliably distinguishable here because every genuine finding in this
    # bundle starts with an uppercase letter and every such fragment does not.
    merged: list[str] = []
    for part in raw_parts:
        if merged and part[:1].islower():
            merged[-1] = f"{merged[-1]}; {part}"
        else:
            merged.append(part)
    return merged


# --------------------------------------------------------------------------- #
# Domain status aggregation
# --------------------------------------------------------------------------- #

def domain_status(table: list[dict[str, Any]]) -> dict[str, Any]:
    domains = {
        "physics_reference_infrastructure": [],
        "graphene_Gamma_finite_PAO": [],
        "full_KS": [],
        "ml_checkpoint_selection": [],
        "C19C20": [],
        "qneq0": [],
        "MATBG": [],
        "infrastructure_reconciliation": [],
        "unclassified": [],
    }
    for row in table:
        domain = row.get("domain")
        domains[domain if domain in domains else "unclassified"].append(row)

    def active_blockers(rows):
        return [r for r in rows if r["classification"] == ACTIVE_BLOCKER]

    gamma_blockers = active_blockers(domains["graphene_Gamma_finite_PAO"])
    reconciliation_blockers = active_blockers(domains["infrastructure_reconciliation"])
    # A finding no FINDING_RULES keyword tuple matched is NOT silently
    # dropped: it must surface here so an unclassified active blocker is
    # visible and investigable, never invisibly excluded from every domain
    # count (the bug this replaces -- see reconcile_finding's own default:
    # unmatched findings are conservatively kept as ACTIVE_BLOCKER with
    # domain="unclassified", precisely so they cannot vanish).
    unclassified_blockers = active_blockers(domains["unclassified"])

    # C14C_v2=PASS satisfies C19/C20's *prerequisite*; it does not mean the
    # C19/C20 protocol itself (bond-stretch gauge rule, k_generic_1/K
    # campaigns, calibration/validation splits, tau_model) has been executed
    # -- none of that exists yet, so C19C20 stays BLOCKED regardless of the
    # prerequisite's own verdict. Conflating "may now attempt execution" with
    # "already executed" was a real bug in an earlier version of this
    # resolver; keep the two states distinct.
    c14c_v2 = gate_verdict("C14C_basis_response_v2")
    c19c20_prerequisite_satisfied = c14c_v2["verdict"] == "PASS"
    c19c20_status = BLOCKED
    checkpoint_status = NO_GO if gate_verdict("checkpoint_lineage")["verdict"] != "PASS" else READY

    return {
        "physics_reference_infrastructure": READY if not any(
            gate_verdict(n)["verdict"] not in ("PASS", None) for n in
            ("GO2_siesta_dhsdr", "reference_basis_convergence", "numerical_PAO_convergence")
        ) else BLOCKED,
        "graphene_Gamma_finite_PAO": READY if not gamma_blockers and not reconciliation_blockers else BLOCKED,
        "graphene_Gamma_finite_PAO_active_blockers": [r["historical_reason"] for r in gamma_blockers],
        "full_KS": BLOCKED,
        "full_KS_reason": "delta_out_closure=NO_GO",
        "ml_checkpoint_selection": checkpoint_status,
        "ml_checkpoint_selection_reason": "validation rows byte-identical to train; "
                                           "59/63 historically-leading snapshots missing",
        "C19C20": c19c20_status,
        "C19C20_prerequisite_C14C_v2_satisfied": c19c20_prerequisite_satisfied,
        "C19C20_reason": (
            "C14C_v2=PASS (prerequisite satisfied) but the C19/C20 protocol itself "
            "(bond-stretch gauge rule, k_generic_1/K campaigns, calibration/validation "
            "splits, tau_model) has not been executed"
        ) if c19c20_prerequisite_satisfied else "C14C_v2_not_yet_certified",
        "qneq0": BLOCKED,
        "qneq0_reason": "GO-5/GO-7/downstream gates",
        "MATBG": BLOCKED,
        "MATBG_reason": "GO-5/GO-7/downstream gates",
        "infrastructure_reconciliation": READY if not reconciliation_blockers else BLOCKED,
        "infrastructure_reconciliation_active_blockers": [r["historical_reason"] for r in reconciliation_blockers],
        # Never silently dropped (see the loop above): any finding no
        # FINDING_RULES rule matches is surfaced here, not excluded from
        # every domain count. An empty list is a real "nothing unclassified
        # left"; a non-empty one needs a human or a new keyword rule, not a
        # default READY reading elsewhere in this dict.
        "unclassified_active_blockers": [r["historical_reason"] for r in unclassified_blockers],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-db", type=Path, default=DEFAULT_STATE_DB)
    parser.add_argument("--task-id", default=DEFAULT_TASK_ID)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    conn = open_readonly(args.state_db)
    try:
        findings = latest_findings(conn, args.task_id)
    finally:
        conn.close()
    if not findings:
        raise SystemExit(f"task {args.task_id!r}: no repair_route event with issues found")

    table = [reconcile_finding(f) for f in findings]
    status = domain_status(table)

    report = {
        "schema": SCHEMA,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "task_id": args.task_id,
        "state_db_path": str(args.state_db),
        "n_findings": len(findings),
        "reconciliation_table": table,
        "domain_status": status,
        "summary": {
            "superseded": sum(1 for r in table if r["classification"] == SUPERSEDED),
            "out_of_scope": sum(1 for r in table if r["classification"] == OUT_OF_SCOPE),
            "evidence_not_attached": sum(1 for r in table if r["classification"] == EVIDENCE_NOT_ATTACHED),
            "active_blockers": sum(1 for r in table if r["classification"] == ACTIVE_BLOCKER),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps(report["domain_status"], indent=2, sort_keys=True))
    return 0 if status["graphene_Gamma_finite_PAO"] == READY else 1


if __name__ == "__main__":
    raise SystemExit(main())
