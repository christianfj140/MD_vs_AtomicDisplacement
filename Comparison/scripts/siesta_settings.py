#!/usr/bin/env python3
"""Compare the SIESTA settings used by MD and AtomDisplacement configs.

This module deliberately does not rewrite pipeline configs. It provides a strict
hash/comparison layer so the UI can warn when a run is not a clean comparison of
dataset-generation strategy alone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SHARED = REPO_ROOT / "Comparison" / "config" / "shared_siesta_settings.yaml"
DEFAULT_MD_CONFIG = REPO_ROOT / "MD" / "pipeline_config.yaml"
DEFAULT_ATOM_CONFIG = REPO_ROOT / "AtomDisplacement" / "pipeline_config.yaml"


COMMON_KEYS = [
    "lattice_constant",
    "lattice_vectors",
    "PAO.BasisType",
    "PAO.BasisSize",
    "PAO.EnergyShift",
    "MeshCutoff",
    "XC.functional",
    "XC.authors",
    "MaxSCFIterations",
    "SolutionMethod",
    "DM.MixingWeight",
    "DM.NumberPulay",
    "DM.Tolerance",
    "DM.Require.Energy.Convergence",
    "DM.Energy.Tolerance",
    "SpinPolarized",
    "FixSpin",
    "NonCollinearSpin",
    "ForceAuxCell",
    "Save.HS",
    "TS.HS.Save",
    "TS.DE.Save",
    "XML.Write",
]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data if isinstance(data, dict) else {}


def normalize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): normalize_value(item) for key, item in sorted(value.items())}
    if isinstance(value, list):
        return [normalize_value(item) for item in value]
    if isinstance(value, tuple):
        return [normalize_value(item) for item in value]
    if isinstance(value, str):
        return " ".join(value.strip().split())
    return value


def canonical_json(data: dict[str, Any]) -> str:
    return json.dumps(normalize_value(data), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def settings_hash(settings: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(settings).encode("utf-8")).hexdigest()


def md_siesta_settings(config: dict[str, Any]) -> dict[str, Any]:
    md = config.get("md", {}) or {}
    return {
        "lattice_constant": md.get("lattice_constant"),
        "lattice_vectors": md.get("lattice_vectors"),
        "MeshCutoff": md.get("mesh_cutoff"),
        "XC.functional": md.get("xc_functional"),
        "XC.authors": md.get("xc_authors"),
        "PAO.BasisType": md.get("basis_type"),
        "PAO.BasisSize": md.get("basis_size"),
        "PAO.EnergyShift": md.get("energy_shift"),
        "MaxSCFIterations": md.get("max_scf_iterations"),
        "SolutionMethod": md.get("solution_method"),
        "DM.MixingWeight": md.get("dm_mixing_weight"),
        "DM.NumberPulay": md.get("dm_number_pulay"),
        "DM.Tolerance": md.get("dm_tolerance"),
        "DM.Require.Energy.Convergence": md.get("dm_require_energy_convergence"),
        "DM.Energy.Tolerance": md.get("dm_energy_tolerance"),
        "SpinPolarized": md.get("spin_polarized"),
        "FixSpin": md.get("fix_spin"),
        "NonCollinearSpin": md.get("non_collinear_spin"),
        "ForceAuxCell": "T" if bool(md.get("force_aux_cell", False)) else "F",
        "Save.HS": "T" if bool(md.get("save_hs_file", False)) else "F",
        "TS.HS.Save": "T" if bool(md.get("save_hs", False)) else "F",
        "TS.DE.Save": "T" if bool(md.get("save_de", False)) else "F",
        "XML.Write": "T" if bool(md.get("xml_write", False)) else "F",
    }


def atom_siesta_settings(config: dict[str, Any]) -> dict[str, Any]:
    structure = config.get("structure", {}) or {}
    siesta = dict(structure.get("siesta", {}) or {})
    force_constants = structure.get("force_constants", {}) or {}
    return {
        "lattice_constant": structure.get("lattice_constant"),
        "lattice_vectors": structure.get("lattice_vectors"),
        "TS.HS.Save": "T" if bool(force_constants.get("save_tshs", siesta.get("TS.HS.Save", True))) else "F",
        "TS.DE.Save": "T" if bool(force_constants.get("save_tsde", siesta.get("TS.DE.Save", True))) else "F",
        **siesta,
    }


def compare_settings(
    md_config: dict[str, Any],
    atom_config: dict[str, Any],
    shared_settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    md_settings = md_siesta_settings(md_config)
    atom_settings = atom_siesta_settings(atom_config)
    shared_settings = shared_settings or {}
    mismatches = []
    for key in COMMON_KEYS:
        md_value = normalize_value(md_settings.get(key))
        atom_value = normalize_value(atom_settings.get(key))
        shared_value = normalize_value(shared_settings.get(key))
        if md_value != atom_value:
            mismatches.append(
                {
                    "key": key,
                    "md": md_value,
                    "atom_displacement": atom_value,
                    "shared_reference": shared_value,
                }
            )
    return {
        "ok": not mismatches,
        "siesta_settings_hash": settings_hash({"md": md_settings, "atom_displacement": atom_settings}),
        "md_siesta_settings_hash": settings_hash(md_settings),
        "atom_displacement_siesta_settings_hash": settings_hash(atom_settings),
        "shared_siesta_settings_hash": settings_hash(shared_settings) if shared_settings else None,
        "mismatches": mismatches,
        "warning": "" if not mismatches else "MD and AtomDisplacement SIESTA settings differ; comparison is not strict.",
    }


def file_digest(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(path for path in paths if path.exists() and path.is_file()):
        digest.update(str(path).encode("utf-8", errors="ignore"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--md-config", type=Path, default=DEFAULT_MD_CONFIG)
    parser.add_argument("--atom-config", type=Path, default=DEFAULT_ATOM_CONFIG)
    parser.add_argument("--shared-settings", type=Path, default=DEFAULT_SHARED)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = compare_settings(
        load_yaml(args.md_config),
        load_yaml(args.atom_config),
        load_yaml(args.shared_settings) if args.shared_settings.exists() else {},
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
