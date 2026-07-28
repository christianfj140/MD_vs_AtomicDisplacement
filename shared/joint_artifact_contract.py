"""Validate joint Graph2Mat/DeepH SIESTA benchmark artifacts.

The contract is intentionally filesystem-only: it never runs SIESTA and never
repairs a dataset. It answers whether an existing snapshot has the artifacts
needed to be used as shared SIESTA ground truth by both Graph2Mat and DeepH.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from siesta_output_status import parse_siesta_output


CONTRACT_NAME = "joint_graph2mat_deeph_artifact_contract_v1"
G2M_DEEPH_BENCHMARK_PROFILE = "g2m_deeph_benchmark"

SYSTEM_LABEL_SUFFIXES = (
    ".TSHS",
    ".TSDE",
    ".HSX",
    ".STRUCT_OUT",
    ".XV",
    ".ORB_INDX",
)
FORBIDDEN_REFERENCE_NAMES = {"ML_prediction.HSX"}


@dataclass(frozen=True)
class ArtifactRequirement:
    key: str
    description: str
    required: bool = True
    filenames: tuple[str, ...] = ()
    system_label_suffix: str | None = None
    category: str = "snapshot"


@dataclass
class SnapshotValidationResult:
    snapshot_dir: Path
    contract_name: str = CONTRACT_NAME
    valid: bool = False
    repair_required: bool = False
    system_label: str | None = None
    missing_required: list[str] = field(default_factory=list)
    present_artifacts: dict[str, str] = field(default_factory=dict)
    siesta_run_status: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["snapshot_dir"] = str(self.snapshot_dir)
        return data


@dataclass
class DatasetValidationResult:
    dataset_root: Path
    contract_name: str = CONTRACT_NAME
    valid: bool = False
    total_snapshots: int = 0
    valid_snapshots: int = 0
    invalid_snapshots: int = 0
    repair_required_snapshots: int = 0
    basis_present: bool | None = None
    pseudopotential_provenance_present: bool | None = None
    material_identity_present: bool | None = None
    siesta_input_provenance_present: bool | None = None
    siesta_version_provenance_present: bool | None = None
    siesta_command_line_provenance_present: bool | None = None
    siesta_environment_provenance_present: bool | None = None
    siesta_execution_log_present: bool | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    snapshots: list[SnapshotValidationResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["dataset_root"] = str(self.dataset_root)
        data["snapshots"] = [result.to_dict() for result in self.snapshots]
        return data


def read_system_label_from_fdf(path: Path) -> str | None:
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        clean = line.split("#", 1)[0].strip()
        if not clean:
            continue
        parts = clean.split()
        if len(parts) >= 2 and parts[0].lower() == "systemlabel":
            return parts[1]
    return None


def read_system_label_from_metadata(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(metadata, dict):
        return None
    for key in ("system_label", "siesta_system_label", "SystemLabel"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _system_labels_from_filenames(snapshot_dir: Path) -> set[str]:
    labels: set[str] = set()
    if not snapshot_dir.exists():
        return labels
    for path in snapshot_dir.iterdir():
        if not path.is_file() or path.name in FORBIDDEN_REFERENCE_NAMES:
            continue
        for suffix in SYSTEM_LABEL_SUFFIXES:
            if path.name.endswith(suffix) and path.name != suffix:
                labels.add(path.name[: -len(suffix)])
                break
    return labels


def resolve_system_label(snapshot_dir: Path, default: str | None = None) -> tuple[str | None, list[str], list[str]]:
    """Resolve SystemLabel from RUN.fdf, metadata, filenames or explicit default."""

    errors: list[str] = []
    warnings: list[str] = []
    labels_by_source: dict[str, str] = {}

    run_fdf_label = read_system_label_from_fdf(snapshot_dir / "RUN.fdf")
    if run_fdf_label:
        labels_by_source["RUN.fdf"] = run_fdf_label

    metadata_label = read_system_label_from_metadata(snapshot_dir / "metadata.json")
    if metadata_label:
        labels_by_source["metadata.json"] = metadata_label

    filename_labels = _system_labels_from_filenames(snapshot_dir)
    if len(filename_labels) == 1:
        labels_by_source["filenames"] = next(iter(filename_labels))
    elif len(filename_labels) > 1:
        errors.append(f"ambiguous SystemLabel from filenames: {sorted(filename_labels)}")

    if default:
        labels_by_source["default"] = default

    if errors:
        return None, errors, warnings

    unique_labels = sorted(set(labels_by_source.values()))
    if len(unique_labels) > 1:
        errors.append(f"ambiguous SystemLabel across sources: {labels_by_source}")
        return None, errors, warnings
    if not unique_labels:
        errors.append("could not resolve SystemLabel from RUN.fdf, metadata, filenames or default")
        return None, errors, warnings
    return unique_labels[0], errors, warnings


def snapshot_requirements(
    system_label: str,
    *,
    require_tshs: bool = True,
    require_tsde: bool = True,
    require_run_output: bool = True,
) -> list[ArtifactRequirement]:
    requirements = [
        ArtifactRequirement("run_fdf", "SIESTA input FDF", filenames=("RUN.fdf",)),
        ArtifactRequirement("metadata", "snapshot metadata", filenames=("metadata.json",)),
        ArtifactRequirement(
            "run_output",
            "SIESTA text output",
            required=require_run_output,
            filenames=("RUN.out", "siesta.out"),
        ),
        ArtifactRequirement("hsx", "DeepH SIESTA Hamiltonian/overlap input", system_label_suffix=".HSX"),
        ArtifactRequirement("struct_out", "DeepH SIESTA structure output", system_label_suffix=".STRUCT_OUT"),
        ArtifactRequirement("xv", "SIESTA XV geometry/velocity output", system_label_suffix=".XV"),
        ArtifactRequirement("orb_indx", "DeepH SIESTA orbital index file", system_label_suffix=".ORB_INDX"),
        ArtifactRequirement(
            "tshs",
            "Graph2Mat/evaluator transport Hamiltonian",
            required=require_tshs,
            system_label_suffix=".TSHS",
        ),
        ArtifactRequirement(
            "tsde",
            "Graph2Mat transport density/energy artifact",
            required=require_tsde,
            system_label_suffix=".TSDE",
        ),
    ]
    # Materialize the label into descriptions only indirectly; matching happens
    # in find_artifact so tests can inspect requirement metadata.
    return requirements


def find_artifact(snapshot_dir: Path, requirement: ArtifactRequirement, system_label: str) -> Path | None:
    for name in requirement.filenames:
        candidate = snapshot_dir / name
        if candidate.exists():
            return candidate
    if requirement.system_label_suffix:
        candidate = snapshot_dir / f"{system_label}{requirement.system_label_suffix}"
        if candidate.exists():
            return candidate
    return None


def validate_snapshot(
    snapshot_dir: Path,
    *,
    system_label: str | None = None,
    require_tshs: bool = True,
    require_tsde: bool = True,
    require_run_output: bool = True,
) -> SnapshotValidationResult:
    snapshot_dir = Path(snapshot_dir)
    result = SnapshotValidationResult(snapshot_dir=snapshot_dir)
    if not snapshot_dir.exists():
        result.errors.append(f"snapshot directory does not exist: {snapshot_dir}")
        result.repair_required = True
        return result
    if not snapshot_dir.is_dir():
        result.errors.append(f"snapshot path is not a directory: {snapshot_dir}")
        result.repair_required = True
        return result

    resolved_label, label_errors, label_warnings = resolve_system_label(snapshot_dir, default=system_label)
    result.errors.extend(label_errors)
    result.warnings.extend(label_warnings)
    result.system_label = resolved_label
    if resolved_label is None:
        result.repair_required = True
        return result

    for forbidden_name in sorted(FORBIDDEN_REFERENCE_NAMES):
        if (snapshot_dir / forbidden_name).exists():
            result.warnings.append(f"forbidden prediction artifact present but ignored: {forbidden_name}")

    for requirement in snapshot_requirements(
        resolved_label,
        require_tshs=require_tshs,
        require_tsde=require_tsde,
        require_run_output=require_run_output,
    ):
        artifact = find_artifact(snapshot_dir, requirement, resolved_label)
        if artifact is not None:
            result.present_artifacts[requirement.key] = str(artifact)
        elif requirement.required:
            result.missing_required.append(requirement.key)

    if require_run_output and "run_output" in result.present_artifacts:
        result.siesta_run_status = parse_siesta_output(
            Path(result.present_artifacts["run_output"]),
            Path(result.present_artifacts["run_fdf"]) if "run_fdf" in result.present_artifacts else None,
        )
        if not result.siesta_run_status["valid"]:
            result.errors.append(
                "invalid SIESTA execution evidence: "
                + str(result.siesta_run_status["parser_status"])
            )

    result.valid = not result.errors and not result.missing_required
    result.repair_required = not result.valid
    return result


def validate_recorded_snapshots(
    artifact_validation: dict[str, Any],
    *,
    base_dir: Path,
    cache: dict[Path, SnapshotValidationResult] | None = None,
) -> tuple[list[SnapshotValidationResult], list[str]]:
    """Revalidate snapshot paths recorded in a manifest instead of trusting it."""

    results: list[SnapshotValidationResult] = []
    errors: list[str] = []
    seen: set[Path] = set()
    snapshots = artifact_validation.get("snapshots")
    if not isinstance(snapshots, list) or not snapshots:
        return results, ["artifact validation contains no snapshot records"]
    for index, snapshot in enumerate(snapshots):
        if not isinstance(snapshot, dict) or not snapshot.get("snapshot_dir"):
            errors.append(f"snapshots[{index}] has no snapshot_dir")
            continue
        path = Path(str(snapshot["snapshot_dir"]))
        if not path.is_absolute():
            cwd_path = Path.cwd() / path
            path = cwd_path if cwd_path.exists() else Path(base_dir) / path
        path = path.resolve()
        if path in seen:
            continue
        seen.add(path)
        result = cache.get(path) if cache is not None else None
        if result is None:
            result = validate_snapshot(path)
            if cache is not None:
                cache[path] = result
        results.append(result)
        if not result.valid:
            errors.append(
                f"{path}: "
                + "; ".join(result.missing_required + result.errors)
            )
    return results, errors


def md_temporal_evidence_errors(dataset_root: Path) -> list[str]:
    """Return publication blockers for MD datasets; non-MD datasets are N/A."""

    dataset_root = Path(dataset_root)
    if not (dataset_root / "MD_steps").exists():
        return []
    path = dataset_root / "md_temporal_diagnostics.json"
    if not path.exists():
        return ["MD dataset has no md_temporal_diagnostics.json"]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"invalid md_temporal_diagnostics.json: {exc}"]
    if not isinstance(payload, dict):
        return ["md_temporal_diagnostics.json root is not an object"]
    if payload.get("paper_ready") is True and not payload.get("blockers"):
        return []
    blockers = [str(item) for item in payload.get("blockers") or []]
    return blockers or ["MD temporal evidence is not paper_ready"]


def material_profile_errors(
    dataset_root: Path,
    benchmark_manifest: dict[str, Any] | None = None,
) -> list[str]:
    """Return publication blockers for missing or non-production profiles."""

    dataset_root = Path(dataset_root)
    manifest = benchmark_manifest or {}
    material = manifest.get("material_source")
    if not isinstance(material, dict) or not material:
        payloads = _json_objects([dataset_root / "material_provenance.json"])
        material = payloads[0] if payloads else {}
    profile = str(
        manifest.get("material_profile")
        or material.get("profile")
        or ""
    ).strip().lower()
    if profile == "production":
        return []
    return [
        "material profile is not publication eligible: "
        f"{profile or 'missing'}"
    ]


def discover_snapshot_dirs(dataset_root: Path) -> list[Path]:
    dataset_root = Path(dataset_root)
    if (dataset_root / "RUN.fdf").exists() or (dataset_root / "metadata.json").exists():
        return [dataset_root]
    if not dataset_root.exists():
        return []
    return sorted(
        path
        for path in dataset_root.iterdir()
        if path.is_dir() and ((path / "RUN.fdf").exists() or (path / "metadata.json").exists())
    )


def _has_files(path: Path) -> bool:
    return path.exists() and path.is_dir() and any(child.is_file() for child in path.iterdir())


def _json_objects(paths: list[Path]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists() or not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            payloads.append(payload)
    return payloads


def _non_empty_mapping(payloads: list[dict[str, Any]], *keys: str) -> bool:
    for payload in payloads:
        for key in keys:
            value = payload.get(key)
            if isinstance(value, dict) and bool(value):
                return True
    return False


def _non_empty_text(payloads: list[dict[str, Any]], *keys: str) -> bool:
    for payload in payloads:
        for key in keys:
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return True
    return False


def _non_empty_text_or_sequence(payloads: list[dict[str, Any]], *keys: str) -> bool:
    for payload in payloads:
        for key in keys:
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return True
            if isinstance(value, (list, tuple)) and any(str(item).strip() for item in value):
                return True
    return False


def _existing_path_value(payloads: list[dict[str, Any]], dataset_root: Path, *keys: str) -> bool:
    for payload in payloads:
        for key in keys:
            value = payload.get(key)
            if not isinstance(value, str) or not value.strip():
                continue
            path = Path(value)
            if not path.is_absolute():
                path = dataset_root / path
            if path.exists() and path.is_file():
                return True
    return False


def _environment_provenance_present(payloads: list[dict[str, Any]]) -> bool:
    for payload in payloads:
        environment = payload.get("environment")
        if not isinstance(environment, dict):
            continue
        python_version = environment.get("python_version")
        platform_value = environment.get("platform")
        if isinstance(python_version, str) and python_version.strip() and isinstance(platform_value, str) and platform_value.strip():
            return True
    return False


def validate_dataset(
    dataset_root: Path,
    *,
    snapshot_dirs: list[Path] | None = None,
    system_label: str | None = None,
    require_tshs: bool = True,
    require_tsde: bool = True,
    require_run_output: bool = True,
    basis_dirs: list[Path] | None = None,
    pseudopotential_provenance_paths: list[Path] | None = None,
    material_identity_paths: list[Path] | None = None,
    siesta_input_paths: list[Path] | None = None,
    require_basis: bool = False,
    require_pseudopotential_provenance: bool = False,
    require_material_identity: bool = False,
    require_siesta_input_provenance: bool = False,
    require_siesta_version_provenance: bool = False,
    require_siesta_command_line_provenance: bool = False,
    require_siesta_environment_provenance: bool = False,
    require_siesta_execution_log: bool = False,
    require_dataset_provenance: bool = False,
    validation_profile: str | None = None,
) -> DatasetValidationResult:
    dataset_root = Path(dataset_root)
    if validation_profile == G2M_DEEPH_BENCHMARK_PROFILE or require_dataset_provenance:
        require_basis = True
        require_pseudopotential_provenance = True
        require_material_identity = True
        require_siesta_input_provenance = True
        require_siesta_version_provenance = True
        require_siesta_command_line_provenance = True
        require_siesta_environment_provenance = True
        require_siesta_execution_log = True
    result = DatasetValidationResult(dataset_root=dataset_root)
    snapshots = snapshot_dirs if snapshot_dirs is not None else discover_snapshot_dirs(dataset_root)
    result.snapshots = [
        validate_snapshot(
            path,
            system_label=system_label,
            require_tshs=require_tshs,
            require_tsde=require_tsde,
            require_run_output=require_run_output,
        )
        for path in snapshots
    ]
    result.total_snapshots = len(result.snapshots)
    result.valid_snapshots = sum(1 for item in result.snapshots if item.valid)
    result.invalid_snapshots = result.total_snapshots - result.valid_snapshots
    result.repair_required_snapshots = sum(1 for item in result.snapshots if item.repair_required)
    if result.total_snapshots == 0:
        result.errors.append(f"no snapshot directories found under {dataset_root}")

    default_material_provenance = dataset_root / "material_provenance.json"
    basis_candidates = [Path(path) for path in (basis_dirs or [])] or [dataset_root / "basis"]
    pseudo_candidates = [Path(path) for path in (pseudopotential_provenance_paths or [])] or [
        default_material_provenance
    ]
    material_candidates = [Path(path) for path in (material_identity_paths or [])] or [
        default_material_provenance
    ]
    siesta_candidates = [Path(path) for path in (siesta_input_paths or [])] or [
        dataset_root / "RUN.fdf",
        default_material_provenance,
    ]
    provenance_payloads = _json_objects(
        sorted({*pseudo_candidates, *material_candidates, *siesta_candidates}, key=lambda path: str(path))
    )

    result.basis_present = any(_has_files(path) for path in basis_candidates) or _non_empty_mapping(
        provenance_payloads,
        "basis_file_sha256",
        "basis_hashes",
    )
    result.pseudopotential_provenance_present = _non_empty_mapping(
        provenance_payloads,
        "pseudopotential_sha256",
        "pseudopotential_hashes",
        "pseudopotential_sha256_by_species",
    )
    result.material_identity_present = _non_empty_text(
        provenance_payloads,
        "label",
        "material_label",
        "material_id",
    )
    result.siesta_input_provenance_present = any(path.exists() and path.is_file() for path in siesta_candidates) or _non_empty_text(
        provenance_payloads,
        "fdf_sha256",
        "siesta_input_sha256",
    )
    result.siesta_version_provenance_present = _non_empty_text(
        provenance_payloads,
        "siesta_version",
    ) or _existing_path_value(
        provenance_payloads,
        dataset_root,
        "siesta_version_source_file",
    )
    result.siesta_command_line_provenance_present = _non_empty_text_or_sequence(
        provenance_payloads,
        "siesta_command_line",
    )
    result.siesta_environment_provenance_present = _environment_provenance_present(provenance_payloads)
    result.siesta_execution_log_present = _existing_path_value(
        provenance_payloads,
        dataset_root,
        "siesta_stdout_path",
        "run_out_path",
    )

    if require_basis and not result.basis_present:
        result.errors.append("dataset-level basis provenance or basis file hashes are missing")
    if require_pseudopotential_provenance and not result.pseudopotential_provenance_present:
        result.errors.append("dataset-level pseudopotential provenance or hashes are missing")
    if require_material_identity and not result.material_identity_present:
        result.errors.append("dataset-level material identity is missing")
    if require_siesta_input_provenance and not result.siesta_input_provenance_present:
        result.errors.append("dataset-level SIESTA/FDF input provenance is missing")
    if require_siesta_version_provenance and not result.siesta_version_provenance_present:
        result.errors.append("dataset-level SIESTA version provenance is missing")
    if require_siesta_command_line_provenance and not result.siesta_command_line_provenance_present:
        result.errors.append("dataset-level SIESTA command-line provenance is missing")
    if require_siesta_environment_provenance and not result.siesta_environment_provenance_present:
        result.errors.append("dataset-level execution environment provenance is missing")
    if require_siesta_execution_log and not result.siesta_execution_log_present:
        result.errors.append("dataset-level SIESTA execution log provenance is missing")

    result.valid = not result.errors and result.invalid_snapshots == 0
    if result.invalid_snapshots:
        result.warnings.append(f"{result.invalid_snapshots} snapshots are not benchmark-ready")
    return result
