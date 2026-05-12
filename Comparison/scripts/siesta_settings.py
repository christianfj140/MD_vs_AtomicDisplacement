#!/usr/bin/env python3
"""Compare SIESTA settings used by the scientific methods.

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

from method_registry import normalize_method_id


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
    "overlap_policy",
]

PHYSICS_RELEVANT_KEYS = {
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
    "overlap_policy",
}

OUTPUT_ARTIFACT_KEYS = {
    "Save.HS",
    "TS.HS.Save",
    "TS.DE.Save",
    "XML.Write",
}


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


def random_cartesian_siesta_settings(config: dict[str, Any]) -> dict[str, Any]:
    structure = config.get("structure", {}) or {}
    random_cartesian = structure.get("random_cartesian", {}) or {}
    settings = dict(atom_siesta_settings(config))
    random_siesta = random_cartesian.get("siesta", {}) or {}
    if isinstance(random_siesta, dict):
        settings.update(random_siesta)
    if random_cartesian.get("overlap_policy") not in (None, ""):
        settings["overlap_policy"] = random_cartesian.get("overlap_policy")
    return settings


def method_siesta_settings(method: str, config: dict[str, Any]) -> dict[str, Any]:
    method_id = normalize_method_id(method, allow_unknown=True)
    if method_id == "md":
        return md_siesta_settings(config)
    if method_id == "siesta_fc_cartesian":
        return atom_siesta_settings(config)
    if method_id == "random_cartesian":
        return random_cartesian_siesta_settings(config)
    raise ValueError(f"Unsupported SIESTA settings method: {method!r}")


def files_content_hash(paths: list[Path]) -> str:
    content_hashes = []
    for path in paths:
        if path.exists() and path.is_file():
            content_hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
    if not content_hashes:
        return ""
    digest = hashlib.sha256()
    for content_hash in sorted(content_hashes):
        digest.update(content_hash.encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def artifact_hash_payload(
    *,
    basis_files: list[Path] | None = None,
    pseudopotential_files: list[Path] | None = None,
) -> dict[str, str]:
    return {
        "basis_hash": files_content_hash(basis_files or []),
        "pseudopotential_hash": files_content_hash(pseudopotential_files or []),
    }


def mismatch_payload(
    *,
    method_a: str,
    method_b: str,
    key: str,
    value_a: Any,
    value_b: Any,
    shared_value: Any = None,
    severity: str,
    mismatch_type: str = "siesta_setting",
) -> dict[str, Any]:
    return {
        "type": mismatch_type,
        "key": key,
        "methods": [method_a, method_b],
        "values": {
            method_a: value_a,
            method_b: value_b,
        },
        "shared_reference": shared_value,
        "severity": severity,
        "scientifically_relevant": severity == "severe",
    }


def pairwise_mismatch_report(
    method_settings: dict[str, dict[str, Any]],
    shared_settings: dict[str, Any],
    artifact_hashes_by_method: dict[str, dict[str, str]] | None = None,
) -> list[dict[str, Any]]:
    artifact_hashes_by_method = artifact_hashes_by_method or {}
    methods = sorted(method_settings)
    mismatches: list[dict[str, Any]] = []
    for index, method_a in enumerate(methods):
        for method_b in methods[index + 1 :]:
            settings_a = method_settings[method_a]
            settings_b = method_settings[method_b]
            for key in COMMON_KEYS:
                value_a = normalize_value(settings_a.get(key))
                value_b = normalize_value(settings_b.get(key))
                if value_a == value_b:
                    continue
                severity = "severe" if key in PHYSICS_RELEVANT_KEYS else "warning"
                mismatches.append(
                    mismatch_payload(
                        method_a=method_a,
                        method_b=method_b,
                        key=key,
                        value_a=value_a,
                        value_b=value_b,
                        shared_value=normalize_value(shared_settings.get(key)),
                        severity=severity,
                    )
                )
            artifacts_a = artifact_hashes_by_method.get(method_a, {}) or {}
            artifacts_b = artifact_hashes_by_method.get(method_b, {}) or {}
            for artifact_key in ("basis_hash", "pseudopotential_hash"):
                value_a = artifacts_a.get(artifact_key) or ""
                value_b = artifacts_b.get(artifact_key) or ""
                if not value_a or not value_b or value_a == value_b:
                    continue
                mismatches.append(
                    mismatch_payload(
                        method_a=method_a,
                        method_b=method_b,
                        key=artifact_key,
                        value_a=value_a,
                        value_b=value_b,
                        severity="severe",
                        mismatch_type="basis_pseudopotential",
                    )
                )
    return mismatches


def compare_method_settings(
    configs_by_method: dict[str, dict[str, Any]],
    shared_settings: dict[str, Any] | None = None,
    *,
    artifact_hashes_by_method: dict[str, dict[str, str]] | None = None,
    selected_methods: list[str] | None = None,
) -> dict[str, Any]:
    shared_settings = shared_settings or {}
    normalized_configs = {
        normalize_method_id(method, allow_unknown=True): config
        for method, config in configs_by_method.items()
    }
    methods = selected_methods or list(normalized_configs)
    canonical_methods = []
    for method in methods:
        method_id = normalize_method_id(method, allow_unknown=True)
        if method_id in normalized_configs and method_id not in canonical_methods:
            canonical_methods.append(method_id)
    method_settings = {
        method_id: method_siesta_settings(method_id, normalized_configs[method_id])
        for method_id in canonical_methods
    }
    artifact_hashes_by_method = {
        normalize_method_id(method, allow_unknown=True): dict(payload or {})
        for method, payload in (artifact_hashes_by_method or {}).items()
    }
    pairwise_mismatches = pairwise_mismatch_report(
        method_settings,
        shared_settings,
        artifact_hashes_by_method,
    )
    severe_mismatches = [mismatch for mismatch in pairwise_mismatches if mismatch.get("severity") == "severe"]
    warning_mismatches = [mismatch for mismatch in pairwise_mismatches if mismatch.get("severity") != "severe"]
    hash_by_method = {
        method_id: settings_hash(settings)
        for method_id, settings in method_settings.items()
    }
    basis_hash_by_method = {
        method_id: (artifact_hashes_by_method.get(method_id, {}) or {}).get("basis_hash", "")
        for method_id in canonical_methods
    }
    pseudopotential_hash_by_method = {
        method_id: (artifact_hashes_by_method.get(method_id, {}) or {}).get("pseudopotential_hash", "")
        for method_id in canonical_methods
    }
    warning = (
        "SIESTA settings or basis/pseudopotential artifacts differ across selected methods."
        if pairwise_mismatches
        else ""
    )
    severe_warning = (
        "Physics-relevant SIESTA settings or basis/pseudopotential artifacts differ across selected methods."
        if severe_mismatches
        else ""
    )
    report = {
        "ok": not severe_mismatches,
        "selected_methods": canonical_methods,
        "siesta_settings_hash": settings_hash(method_settings),
        "siesta_settings_hash_by_method": hash_by_method,
        "method_siesta_settings": method_settings,
        "basis_hash_by_method": basis_hash_by_method,
        "pseudopotential_hash_by_method": pseudopotential_hash_by_method,
        "shared_siesta_settings_hash": settings_hash(shared_settings) if shared_settings else None,
        "pairwise_mismatch_report": pairwise_mismatches,
        "severe_mismatches": severe_mismatches,
        "warning_mismatches": warning_mismatches,
        "warning": warning,
        "severe_warning": severe_warning,
        "basis_pseudopotential_warning": (
            "Basis or pseudopotential content hashes differ across selected methods."
            if any(mismatch.get("type") == "basis_pseudopotential" for mismatch in severe_mismatches)
            else ""
        ),
    }
    if "md" in hash_by_method:
        report["md_siesta_settings_hash"] = hash_by_method["md"]
    if "siesta_fc_cartesian" in hash_by_method:
        report["siesta_fc_cartesian_siesta_settings_hash"] = hash_by_method["siesta_fc_cartesian"]
        report["atom_displacement_siesta_settings_hash"] = hash_by_method["siesta_fc_cartesian"]
    if "random_cartesian" in hash_by_method:
        report["random_cartesian_siesta_settings_hash"] = hash_by_method["random_cartesian"]
    return report


def compare_settings(
    md_config: dict[str, Any],
    atom_config: dict[str, Any],
    shared_settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    report = compare_method_settings(
        {"md": md_config, "siesta_fc_cartesian": atom_config},
        shared_settings,
        selected_methods=["md", "siesta_fc_cartesian"],
    )
    legacy_mismatches = []
    for mismatch in report["pairwise_mismatch_report"]:
        values = mismatch.get("values", {})
        legacy_mismatches.append(
            {
                "key": mismatch.get("key"),
                "md": values.get("md"),
                "atom_displacement": values.get("siesta_fc_cartesian"),
                "shared_reference": mismatch.get("shared_reference"),
                "severity": mismatch.get("severity"),
            }
        )
    report["mismatches"] = legacy_mismatches
    if report["warning"]:
        report["warning"] = "MD and AtomDisplacement SIESTA settings differ; comparison is not strict."
    return report


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
