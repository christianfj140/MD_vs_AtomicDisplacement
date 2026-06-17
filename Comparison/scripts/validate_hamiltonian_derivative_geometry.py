#!/usr/bin/env python3
"""Validate finite-displacement derivative stencil geometries without computing metrics."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from hamiltonian_derivative_stencil import (  # noqa: E402
    DEFAULT_GEOMETRY_TOLERANCE_ANG,
    discover_derivative_stencils,
    validate_derivative_geometry,
    validation_errors,
)


FIELDS = [
    "sample",
    "status",
    "finite_difference_method",
    "base_sample_id",
    "plus_sample_id",
    "minus_sample_id",
    "atom_index_zero_based",
    "axis",
    "axis_index",
    "delta_ang",
    "issue_codes",
    "issue_messages",
]


class DerivativeGeometryValidationError(RuntimeError):
    """Raised when derivative geometry validation cannot complete."""


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def validate_derivative_geometry_outputs(
    result_dir: Path,
    *,
    output_dir: Path | None = None,
    method: str = "central",
    source_model: str = "graph2mat",
    split: str = "all",
    require_central: bool = False,
    tolerance_ang: float = DEFAULT_GEOMETRY_TOLERANCE_ANG,
) -> dict[str, Any]:
    output_dir = output_dir or result_dir
    discoveries = discover_derivative_stencils(
        result_dir,
        method=source_model,
        split=split,
        finite_difference_method=method,
        require_central=require_central,
    )
    rows = []
    for discovery in discoveries:
        issues = validate_derivative_geometry(discovery, tolerance_ang=tolerance_ang)
        metadata = discovery.stencil.metadata if discovery.stencil else None
        errors = validation_errors(issues)
        warnings = [issue for issue in issues if not issue.is_error]
        rows.append(
            {
                "sample": metadata.sample_id if metadata else "|".join(discovery.sample_ids),
                "status": "error" if errors else "warning" if warnings else "ok",
                "finite_difference_method": discovery.method,
                "base_sample_id": metadata.base_sample_id if metadata else None,
                "plus_sample_id": metadata.plus_sample_id if metadata else None,
                "minus_sample_id": metadata.minus_sample_id if metadata else None,
                "atom_index_zero_based": metadata.atom_index_zero_based if metadata else None,
                "axis": metadata.axis if metadata else None,
                "axis_index": metadata.axis_index if metadata else None,
                "delta_ang": metadata.delta_ang if metadata else None,
                "issue_codes": ";".join(issue.code for issue in issues),
                "issue_messages": "; ".join(issue.message for issue in issues),
            }
        )
    summary = {
        "schema_version": "derivative_geometry_validation_v1",
        "result_dir": str(result_dir),
        "split": split,
        "finite_difference_method": method,
        "source_model": source_model,
        "tolerance_ang": tolerance_ang,
        "total": len(rows),
        "ok": len([row for row in rows if row["status"] == "ok"]),
        "warnings": len([row for row in rows if row["status"] == "warning"]),
        "errors": len([row for row in rows if row["status"] == "error"]),
        "outputs": {
            "csv": str(output_dir / "derivative_geometry_validation.csv"),
            "json": str(output_dir / "derivative_geometry_validation.json"),
        },
        "rows": rows,
    }
    write_csv(output_dir / "derivative_geometry_validation.csv", rows)
    write_json(output_dir / "derivative_geometry_validation.json", summary)
    return summary


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--method", choices=["central", "forward", "backward"], default="central")
    parser.add_argument("--source-model", choices=["graph2mat", "deeph"], default="graph2mat")
    parser.add_argument("--split", default="all")
    parser.add_argument("--require-central", action="store_true")
    parser.add_argument("--tolerance-ang", type=float, default=DEFAULT_GEOMETRY_TOLERANCE_ANG)
    return parser


def main() -> int:
    args = build_argument_parser().parse_args()
    summary = validate_derivative_geometry_outputs(
        args.result_dir,
        output_dir=args.output_dir,
        method=args.method,
        source_model=args.source_model,
        split=args.split,
        require_central=args.require_central,
        tolerance_ang=args.tolerance_ang,
    )
    print(json.dumps({"total": summary["total"], "ok": summary["ok"], "errors": summary["errors"]}, sort_keys=True))
    return 1 if summary["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
