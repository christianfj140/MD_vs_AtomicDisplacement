#!/usr/bin/env python3
"""Scope-drift linter for E-F_001's actual claim (see
``Comparison/results/epc_repair/E-F_001_scope_contract_v3.json``).

Scans a configurable set of artifact/doc paths for AFFIRMATIVE over-claims of
topics explicitly excluded from E-F_001's scope: physical EPC, full-KS EPC,
`g` as the physical EPC coupling constant, `lambda`/`alpha2F` in the EPC
coupling sense, MATBG, graphene-K as a physical result.

A mention is not itself a violation. This repo's own historical reports
correctly say things like "full-KS EPC is not claimed" or "MATBG is
explicitly out of scope" -- those are exactly the sentences this task must
NOT flag. Only affirmative claims (asserting the excluded thing as true,
resolved, or produced) count, so each pattern is paired with a negation/
out-of-scope-label guard checked against the same line.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO_ROOT / "shared")]
from artifact_signature import file_sha256  # noqa: E402

SCHEMA = "epc_e_f_001_claim_scope_lint_v1"
DEFAULT_OUTPUT = REPO_ROOT / "Comparison/results/epc_repair/claim_scope_lint.json"

DEFAULT_SCAN_ROOTS = [
    REPO_ROOT / "Comparison/results/epc",
    REPO_ROOT / "Comparison/results/epc_repair",
]
DEFAULT_DOC_GLOB = REPO_ROOT / "docs"

# Directories that are themselves a DIFFERENT, already out-of-scope subsystem
# by construction (their own dedicated MATBG/AB/qneq0 result tree, named as
# such). Linting E-F_001's graphene-Gamma claim against another subsystem's
# own artifacts would just relabel "MATBG's own MATBG file" as a scope
# violation -- noise, not signal. These subsystems have their own tasks/gates
# (S29-S56) and are already tracked OUT_OF_SCOPE by the reconciler.
_EXCLUDED_DIR_NAMES = {"matbg", "ab", "bilayer_graphene_ab", "qneq0"}


def _is_excluded(path: Path, scan_roots: list[Path]) -> bool:
    for root in scan_roots:
        try:
            rel_parts = path.relative_to(root).parts
        except ValueError:
            continue
        if any(part.lower() in _EXCLUDED_DIR_NAMES for part in rel_parts[:-1]):
            return True
    return False

# Negation / scope-label guards: if any of these appear on the same line as a
# forbidden-pattern match, the line is NOT a violation -- it is a correctly
# scoped historical/negative/out-of-scope statement.
_NEGATION_GUARDS = re.compile(
    r"(not claimed|is not claimed|no claim|not certified|out of scope|"
    r"explicitly excluded|excluded from|does not (certify|claim)|"
    r"not (a )?full-KS|not retroactively|no full-KS|unbound|unresolved|"
    r"NO_GO|NO-GO|BLOCKED|not launched|not executed|remains? (open|unclosed)|"
    r"phase 2|blocks? full_KS|blocked by|not the PAO-finite claim|"
    r"tracked separately|untouched|no .{0,20} was (read|written|"
    r"re-adjudicated)|does not depend on any physical|continues? blocked|"
    r"sigue sin estar demostrada|bloqueado por|is a pattern label, not a claim)",
    re.IGNORECASE,
)

# Files whose whole purpose is to name forbidden tokens as labels (this
# linter's own output, the scope contract that enumerates excluded topics) --
# scanning them would just flag the linter/contract for doing its job.
_SELF_EXEMPT_NAMES = {
    "claim_scope_lint.json", "E-F_001_scope_contract_v3.json",
    # the reconciler/resolver's own classification output: its job is to
    # enumerate excluded topics as domain/scope labels for steps it has
    # already classified OUT_OF_SCOPE -- not an affirmative claim about them.
    "s1_s56_reconciliation_v2.json", "canonical_evidence_resolution.json",
}

# Forbidden pattern -> human label. Word-boundaried, case-sensitive where the
# token itself is case-sensitive physics notation (g, lambda as EPC symbol).
_FORBIDDEN_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bphysical EPC\b", re.IGNORECASE), "physical EPC"),
    (re.compile(r"\bfull[- ]KS EPC\b", re.IGNORECASE), "full-KS EPC"),
    (re.compile(r"\bfull KS coupling\b", re.IGNORECASE), "full KS coupling"),
    (re.compile(r"\balpha2F\b"), "alpha2F"),
    (re.compile(r"\bMATBG\b"), "MATBG"),
    # `g` as the physical EPC coupling constant: only flag it spelled out
    # ("g physical", "physical g", "g coupling constant") -- a bare "g" is
    # far too common (grams, geometry `g`, etc.) to regex reliably.
    (re.compile(r"\bg[_ ]?physical\b|\bphysical g\b|\bg coupling constant\b", re.IGNORECASE),
     "g (physical EPC coupling constant)"),
    # standalone lambda/alpha2F as the EPC coupling constant -- exclude the
    # "relative lambda" mode-weight sense used elsewhere in this repo.
    (re.compile(r"(?<!relative )\blambda\b(?!\s*=\s*1e)", re.IGNORECASE), "lambda (EPC coupling constant sense)"),
    (re.compile(r"\bgraphene[- ]K\b.{0,40}\b(physical|qualified|resolved)\b", re.IGNORECASE),
     "graphene-K physical qualification"),
]

# `lambda` also appears as: a LaTeX subscript/eigenvalue index (\lambda, C
# lambda under a sqrt = a phonon IFC eigenvalue, not the EPC constant) and as
# a bare mode-weight ("relative lambda=1e-3", already guarded above). These
# are additional non-EPC senses this repo uses on the same lines the plain
# regex would otherwise flag.
_LAMBDA_NON_EPC_SENSE = re.compile(r"sqrt\(C[ ·]*lambda\)|\\lambda|_\\lambda|\\phi_")
# MATBG as a scope-exclusion list entry or a scalability aside ("escalable a
# MATBG", "candidata ... escalable"), not an affirmative claim about MATBG.
_MATBG_NON_CLAIM_SENSE = re.compile(
    r"not_promoted_to|escalable a MATBG|scalable to MATBG|candidata.{0,60}escalable|"
    r"flagged an explicit open item|response and phonon-provider work|"
    r"no .{0,120} was (read|written|re-adjudicated)|"
    r"importa para MATBG|campaña MATBG, no del toy",
    re.IGNORECASE,
)


def _iter_files(scan_roots: list[Path], doc_root: Path) -> list[Path]:
    files: list[Path] = []
    for root in scan_roots:
        if root.is_dir():
            files.extend(p for p in root.rglob("*.json") if p.is_file())
            files.extend(p for p in root.rglob("*.md") if p.is_file())
    if doc_root.is_dir():
        files.extend(
            p for p in doc_root.glob("epc_*.md")
            if p.is_file() and not any(tag in p.name.lower() for tag in _EXCLUDED_DIR_NAMES)
        )
    files = [p for p in files if not _is_excluded(p, scan_roots) and p.name not in _SELF_EXEMPT_NAMES]
    return sorted(set(files))


def lint_file(path: Path) -> list[dict[str, Any]]:
    violations = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return violations
    lines = text.splitlines()
    for lineno, line in enumerate(lines, start=1):
        if _NEGATION_GUARDS.search(line):
            continue
        # A bare `"TOKEN"` JSON array element two lines below a
        # `"not_promoted_to"` (or similarly named) key is a scope-exclusion
        # list entry, not a claim -- look back a couple of lines for that key.
        if re.fullmatch(r'"[A-Za-z0-9!=_ -]+",?', line.strip()) and any(
            "not_promoted_to" in lines[j] or "out_of_scope" in lines[j].lower()
            for j in range(max(0, lineno - 8), lineno - 1)
        ):
            continue
        for pattern, label in _FORBIDDEN_PATTERNS:
            if label.startswith("lambda") and _LAMBDA_NON_EPC_SENSE.search(line):
                continue
            if label == "MATBG" and _MATBG_NON_CLAIM_SENSE.search(line):
                continue
            match = pattern.search(line)
            if match:
                try:
                    rel = str(path.relative_to(REPO_ROOT))
                except ValueError:
                    rel = str(path)
                violations.append({
                    "file": rel,
                    "line": lineno,
                    "pattern": label,
                    "text": line.strip()[:300],
                })
    return violations


def build_report(scan_roots: list[Path], doc_root: Path) -> dict[str, Any]:
    files = _iter_files(scan_roots, doc_root)
    violations: list[dict[str, Any]] = []
    for path in files:
        violations.extend(lint_file(path))
    return {
        "schema": SCHEMA,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "scan_roots": [str(r.relative_to(REPO_ROOT)) for r in scan_roots],
        "doc_root_glob": f"{doc_root.relative_to(REPO_ROOT)}/epc_*.md",
        "n_files_scanned": len(files),
        "n_violations": len(violations),
        "verdict": "FAIL" if violations else "PASS",
        "violations": violations,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-root", action="append", type=Path, default=None,
                         help="extra directory to scan (repeatable); defaults to "
                              "Comparison/results/epc and Comparison/results/epc_repair")
    parser.add_argument("--doc-root", type=Path, default=DEFAULT_DOC_GLOB)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    scan_roots = args.scan_root if args.scan_root else DEFAULT_SCAN_ROOTS
    report = build_report(scan_roots, args.doc_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"claim_scope_lint: {report['verdict']} "
          f"({report['n_violations']} violation(s) in {report['n_files_scanned']} file(s))")
    for v in report["violations"][:50]:
        print(f"  {v['file']}:{v['line']}: [{v['pattern']}] {v['text']}")
    print(f"-> {args.output}")
    return 0 if report["verdict"] == "PASS" else 1


def _demo() -> None:
    """Self-check: negation guard suppresses a legitimate historical line;
    an affirmative claim on an otherwise-identical line is still caught."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        clean = Path(tmp) / "clean.md"
        clean.write_text("full-KS EPC is not claimed; MATBG is explicitly out of scope.\n")
        dirty = Path(tmp) / "dirty.md"
        dirty.write_text("The physical EPC coupling g_physical is resolved for graphene-Gamma.\n")
        assert lint_file(clean) == [], "negation-guarded line must not be flagged"
        hits = lint_file(dirty)
        assert hits and hits[0]["pattern"] == "physical EPC", f"affirmative claim must be flagged, got {hits}"
    print("lint_epc_e_f_001_claim_scope self-check: OK")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--self-check":
        _demo()
        raise SystemExit(0)
    raise SystemExit(main())
