#!/usr/bin/env python3
"""Generate the molecular-dynamics dataset from pipeline_config.yaml."""

from __future__ import annotations

import os
import csv
import copy
import json
import platform
import shutil
import subprocess
import sys
from importlib import metadata as importlib_metadata
from pathlib import Path

from md_pipeline_config import (
    command,
    config_dir,
    load_pipeline_config,
    md_temperature_blocks,
    md_total_steps,
    paths,
    render_run_fdf,
)
from material_bundle import file_sha256 as material_file_sha256
from material_presets import resolve_material_bundle
from graph2mat_material_config import copy_graph2mat_basis_files, resolve_graph2mat_basis_files
from joint_artifact_contract import (
    CONTRACT_NAME,
    G2M_DEEPH_BENCHMARK_PROFILE,
    find_artifact,
    resolve_system_label,
    snapshot_requirements,
    validate_dataset,
)
from benchmark_manifest import write_benchmark_manifests

BOHR_TO_ANG = 0.529177210903
JOINT_GRAPH2MAT_DEEPH_STORE_FILES = "*fdf *TSHS *TSDE *XV *HSX *STRUCT_OUT *ORB_INDX *out"
JOINT_REQUIRED_FDF_OUTPUT_FLAGS = (
    "SaveHS",
    "Save.HS",
    "TS.HS.Save",
    "TS.DE.Save",
    "XML.Write",
    "Write.OrbitalIndex",
)
SPREAD_SPLIT_WARNING = (
    "MD split strategy 'spread' interleaves trajectory frames across "
    "train/validation/test and is exploratory/debug only; use "
    "'blocked_with_gap' with a positive temporal_gap for scientific comparisons."
)
MANIFEST_FIELDS = [
    "sample_id",
    "method",
    "source_run",
    "frame_index",
    "time_index",
    "displacement_amplitude",
    "displacement_magnitude",
    "displaced_atom",
    "displacement_axis",
    "displacement_sign",
    "displacement_family",
    "structure_path",
    "hamiltonian_path",
    "output_path",
    "run_out_path",
    "metadata_path",
    "valid",
    "validation_reason",
    "split",
    "split_strategy",
    "temporal_gap",
    "source_frame_index",
    "excluded_gap_reason",
    "seed",
    "status",
    "sample_dir",
    "recipe_id",
    "recipe_label",
    "block_id",
    "block_label",
    "generation_parameters_json",
    "sample_index_within_block",
    "global_sample_id",
    "temperature_K",
    "md_block_id",
    "md_block_label",
    "md_source_block_dir",
    "md_source_frame_index",
    "timestep_fs",
    "md_type_of_run",
]


def require_command(command_name: str) -> None:
    """Verifica que un ejecutable exista en PATH."""
    if shutil.which(command_name) is None:
        raise RuntimeError(
            f"No se encontró '{command_name}' en PATH. "
            "Activa tu entorno antes de ejecutar este script."
        )


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in ("numpy", "sisl", "torch", "graph2mat", "deeph"):
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            continue
    return versions


def execution_environment_provenance() -> dict[str, object]:
    """Return a small, whitelisted execution environment summary."""

    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "executable": sys.executable,
        "package_versions": _package_versions(),
    }


def probe_siesta_version(siesta_command: str) -> dict[str, object]:
    """Best-effort SIESTA version probe; unknown versions remain unknown."""

    attempts: list[dict[str, object]] = []
    for flag in ("--version", "-V", "-v"):
        cmd = [siesta_command, flag]
        try:
            result = subprocess.run(
                cmd,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            attempts.append({"command": cmd, "error": str(exc)})
            continue
        output = (result.stdout or "").strip()
        attempts.append({"command": cmd, "returncode": result.returncode, "output": output[:2000]})
        if result.returncode == 0 and output:
            first_line = next((line.strip() for line in output.splitlines() if line.strip()), "")
            if first_line:
                return {
                    "siesta_version": first_line,
                    "siesta_build_info": output,
                    "siesta_version_probe": {"status": "detected", "attempts": attempts},
                }
    return {
        "siesta_version": "",
        "siesta_build_info": "",
        "siesta_version_probe": {"status": "unavailable", "attempts": attempts},
    }


def siesta_command_line(config: dict) -> str:
    pipeline_paths = paths(config)
    venv_activate = pipeline_paths["venv_activate"]
    bash_cmd = (
        f"source '{venv_activate}' "
        f"&& {command(config, 'siesta')} < {pipeline_paths['run_fdf_path'].name}"
    )
    return f"{command(config, 'shell')} -lc {json.dumps(bash_cmd)}"


def siesta_run_provenance(config: dict, *, returncode: int | None = None) -> dict[str, object]:
    pipeline_paths = paths(config)
    try:
        siesta_executable = command(config, "siesta")
        command_line = siesta_command_line(config)
    except KeyError:
        return {"environment": execution_environment_provenance()}
    payload: dict[str, object] = {
        "siesta_executable": siesta_executable,
        "siesta_command_line": command_line,
        "siesta_stdout_path": str(pipeline_paths["run_out_path"]),
        "run_out_path": str(pipeline_paths["run_out_path"]),
        "environment": execution_environment_provenance(),
    }
    payload.update(probe_siesta_version(siesta_executable))
    if returncode is not None:
        payload["siesta_returncode"] = returncode
    return payload


def update_material_provenance(config: dict, updates: dict[str, object]) -> None:
    material_path = paths(config)["dataset_dir"] / "material_provenance.json"
    current: dict[str, object] = {}
    if material_path.exists():
        try:
            payload = json.loads(material_path.read_text(encoding="utf-8"))
            current = payload if isinstance(payload, dict) else {}
        except json.JSONDecodeError:
            current = {}
    current.update(updates)
    write_json(material_path, current)


def prepare_material_inputs(config: dict) -> dict:
    pipeline_paths = paths(config)
    dataset_dir = pipeline_paths["dataset_dir"]
    dataset_dir.mkdir(parents=True, exist_ok=True)
    resolved = resolve_material_bundle(config, base_dir=config_dir(config))
    validated = resolved.validated
    copied: dict[str, str] = {}
    verified: dict[str, str] = {}
    for label, source in sorted(validated.pseudopotentials.items()):
        destination = dataset_dir / source.name
        if destination.exists():
            if not destination.is_file():
                raise RuntimeError(f"MD pseudopotential path is not a file: {destination}")
            if material_file_sha256(destination) != material_file_sha256(source):
                raise RuntimeError(
                    f"MD pseudopotential for species {label!r} differs from material bundle: {destination}"
                )
            verified[label] = destination.name
            continue
        shutil.copy2(source, destination)
        copied[label] = destination.name
    manifest = resolved.to_manifest_dict()
    copied_basis = copy_graph2mat_basis_files(
        resolve_graph2mat_basis_files(validated),
        dataset_dir / "material_basis",
    )
    manifest.update(
        {
            "pseudopotentials_copied_to_dataset": copied,
            "pseudopotentials_verified_in_dataset": verified,
            "graph2mat_basis_files": copied_basis,
        }
    )
    manifest.update(siesta_run_provenance(config))
    write_json(dataset_dir / "material_provenance.json", manifest)
    return manifest


def recipe_manifest_fields(config: dict, *, sample_index: str = "") -> dict[str, str]:
    recipe = config.get("dataset_recipe") or {}
    generation_parameters = recipe.get("generation_parameters_json")
    if generation_parameters in (None, "") and recipe.get("generation_parameters") is not None:
        generation_parameters = json.dumps(
            recipe.get("generation_parameters"),
            sort_keys=True,
            ensure_ascii=False,
        )
    return {
        "recipe_id": str(recipe.get("recipe_id") or ""),
        "recipe_label": str(recipe.get("recipe_label") or ""),
        "block_id": str(recipe.get("block_id") or ""),
        "block_label": str(recipe.get("block_label") or ""),
        "generation_parameters_json": str(generation_parameters or ""),
        "sample_index_within_block": str(sample_index),
        "global_sample_id": "",
    }


def run_command(cmd: list[str], cwd: Path) -> None:
    """Ejecuta un comando y falla con mensaje claro si retorna código != 0."""
    print(f"\n[RUN] {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"El comando falló con código {result.returncode}: {' '.join(cmd)}"
        )


def performance_env(config: dict) -> dict[str, str]:
    env = os.environ.copy()
    mapping = {
        "omp_num_threads": "OMP_NUM_THREADS",
        "mkl_num_threads": "MKL_NUM_THREADS",
        "openblas_num_threads": "OPENBLAS_NUM_THREADS",
        "numexpr_num_threads": "NUMEXPR_NUM_THREADS",
    }
    for key, env_name in mapping.items():
        value = (config.get("performance") or {}).get(key)
        if value in (None, "", "null"):
            continue
        threads = int(value)
        if threads <= 0:
            raise RuntimeError(f"performance.{key} debe ser un entero positivo.")
        env[env_name] = str(threads)
    return env


def setup_store(config: dict) -> None:
    pipeline_paths = paths(config)
    # Asumimos que `graph2mat siesta md setup-store` puede re-ejecutarse sobre el
    # mismo directorio. Si el contenido ya existe, el comportamiento depende de
    # graph2mat y se respeta tal cual.
    run_command(
        [
            command(config, "graph2mat"),
            "siesta",
            "md",
            "setup-store",
            "--files",
            JOINT_GRAPH2MAT_DEEPH_STORE_FILES,
        ],
        cwd=pipeline_paths["dataset_dir"],
    )


def validate_joint_benchmark_artifacts(config: dict, steps_dir: Path | None = None) -> None:
    pipeline_paths = paths(config)
    steps_dir = steps_dir or pipeline_paths["dataset_dir"] / "MD_steps"
    summary_path = steps_dir.parent / "artifact_validation.json"
    if not steps_dir.exists():
        write_json(
            summary_path,
            {
                "contract_name": CONTRACT_NAME,
                "valid": False,
                "benchmark_ready": False,
                "scientific_status": "missing_md_steps",
                "steps_dir": str(steps_dir),
                "errors": [f"missing MD_steps directory: {steps_dir}"],
            },
        )
        raise RuntimeError(f"{CONTRACT_NAME} failed: missing MD_steps directory: {steps_dir}")
    snapshot_dirs = sorted(
        (path for path in steps_dir.iterdir() if path.is_dir() and path.name.isdigit()),
        key=lambda path: int(path.name),
    )
    dataset_dir = pipeline_paths["dataset_dir"]
    material_provenance = dataset_dir / "material_provenance.json"
    result = validate_dataset(
        steps_dir,
        snapshot_dirs=snapshot_dirs,
        basis_dirs=[dataset_dir / "basis", steps_dir / "basis"],
        pseudopotential_provenance_paths=[material_provenance],
        material_identity_paths=[material_provenance],
        siesta_input_paths=[dataset_dir / "RUN.fdf", material_provenance],
        validation_profile=G2M_DEEPH_BENCHMARK_PROFILE,
    )
    update_joint_snapshot_validation_metadata(result)
    summary = result.to_dict()
    summary.update(
        {
            "benchmark_ready": result.valid,
            "scientific_status": "benchmark_ready" if result.valid else "repair_required",
            "steps_dir": str(steps_dir),
            "artifact_validation_path": str(summary_path),
            "generation_contract_diagnostics": {
                "required_fdf_output_flags": list(JOINT_REQUIRED_FDF_OUTPUT_FLAGS),
                "store_files": JOINT_GRAPH2MAT_DEEPH_STORE_FILES,
                "store_file_patterns": JOINT_GRAPH2MAT_DEEPH_STORE_FILES.split(),
            },
        }
    )
    write_json(summary_path, summary)
    if result.valid:
        print(
            f"[OK] {CONTRACT_NAME}: {result.valid_snapshots}/{result.total_snapshots} "
            f"snapshots listos para Graph2Mat+DeepH. Summary: {summary_path}"
        )
        return
    dataset_errors = [str(error) for error in result.errors]
    details = []
    for snapshot in result.snapshots:
        if snapshot.valid:
            continue
        reason = ", ".join(snapshot.missing_required or snapshot.errors or ["invalid"])
        flags = read_joint_fdf_output_flags(snapshot.snapshot_dir / "RUN.fdf")
        details.append(
            f"{snapshot.snapshot_dir.name}: {reason}; "
            f"SystemLabel={snapshot.system_label or 'unknown'}; "
            f"FDF flags={flags}; store_files={JOINT_GRAPH2MAT_DEEPH_STORE_FILES}"
        )
        if len(details) >= 5:
            break
    detail_parts = []
    if dataset_errors:
        detail_parts.append("Dataset-level errors: " + "; ".join(dataset_errors))
    if details:
        detail_parts.append("Snapshot examples: " + "; ".join(details))
    detail_text = " ".join(detail_parts) or "No extra diagnostics available."
    raise RuntimeError(
        f"{CONTRACT_NAME} failed for {steps_dir}: "
        f"{result.invalid_snapshots}/{result.total_snapshots} snapshots are incomplete. "
        f"{detail_text}"
    )


def write_run_fdf(config: dict, block: dict | None = None) -> None:
    pipeline_paths = paths(config)
    # Asumimos que queremos un RUN.fdf determinista: lo sobreescribimos siempre.
    pipeline_paths["run_fdf_path"].write_text(render_run_fdf(config, block=block), encoding="utf-8")
    print(f"[OK] RUN.fdf escrito en {pipeline_paths['run_fdf_path']}")


def run_siesta_with_venv(config: dict) -> None:
    pipeline_paths = paths(config)
    venv_activate = pipeline_paths["venv_activate"]
    if not venv_activate.exists():
        raise RuntimeError(
            "No se encontró el script de activación del entorno virtual esperado en "
            f"{venv_activate}."
        )

    bash_cmd = (
        f"source '{venv_activate}' "
        f"&& {command(config, 'siesta')} < {pipeline_paths['run_fdf_path'].name}"
    )

    print(f"\n[RUN] bash -lc \"{bash_cmd}\"")
    with pipeline_paths["run_out_path"].open("w", encoding="utf-8") as run_out:
        process = subprocess.Popen(
            [command(config, "shell"), "-lc", bash_cmd],
            cwd=pipeline_paths["dataset_dir"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=performance_env(config),
        )

        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            run_out.write(line)

        return_code = process.wait()

    update_material_provenance(
        config,
        {
            "siesta_stdout_path": str(pipeline_paths["run_out_path"]),
            "run_out_path": str(pipeline_paths["run_out_path"]),
            "siesta_returncode": return_code,
        },
    )
    if return_code != 0:
        raise RuntimeError(f"siesta terminó con código {return_code}.")

    print(f"[OK] Salida guardada en {pipeline_paths['run_out_path']}")


def _split_counts(total: int) -> tuple[int, int, int]:
    train = int(total * 0.8)
    validation = int(total * 0.1)
    test = total - train - validation
    if total >= 3 and test == 0:
        test = 1
        train = max(1, train - 1)
    return train, validation, test


def _select_spread(items: list[Path], count: int) -> list[Path]:
    if count <= 0:
        return []
    if count >= len(items):
        return list(items)

    used: set[int] = set()
    selected: list[int] = []
    for index in range(count):
        target = min(len(items) - 1, int((index + 0.5) * len(items) / count))
        if target in used:
            target = min(
                (candidate for candidate in range(len(items)) if candidate not in used),
                key=lambda candidate: abs(candidate - target),
            )
        used.add(target)
        selected.append(target)
    return [items[index] for index in sorted(selected)]


def _split_spread(items: list[Path], train_count: int, validation_count: int, test_count: int) -> dict[str, list[Path]]:
    test = _select_spread(items, test_count)
    remaining = [item for item in items if item not in set(test)]
    validation = _select_spread(remaining, validation_count)
    train = [item for item in remaining if item not in set(validation)]
    if len(train) > train_count:
        train = _select_spread(train, train_count)
    return {"train": train, "validation": validation, "test": test}


def _split_block(items: list[Path], train_count: int, validation_count: int, test_count: int) -> dict[str, list[Path]]:
    requested = train_count + validation_count + test_count
    selected = list(items[:requested])
    train = selected[:train_count]
    validation = selected[train_count : train_count + validation_count]
    test = selected[train_count + validation_count : train_count + validation_count + test_count]
    return {"train": train, "validation": validation, "test": test}


def _parse_block_order(value: object) -> list[str]:
    if isinstance(value, str):
        order = [item.strip().lower() for item in value.replace(";", ",").split(",") if item.strip()]
    elif isinstance(value, list):
        order = [str(item).strip().lower() for item in value if str(item).strip()]
    else:
        order = ["train", "validation", "test"]
    if sorted(order) != ["test", "train", "validation"]:
        raise RuntimeError(
            "splits.block_order debe contener exactamente train, validation y test."
        )
    return order


def _split_blocked_with_gap(
    items: list[Path],
    counts: dict[str, int],
    *,
    temporal_gap: int,
    block_order: list[str],
) -> tuple[dict[str, list[Path]], list[tuple[Path, str]]]:
    nonempty_blocks = [name for name in block_order if counts[name] > 0]
    required = sum(counts.values()) + max(0, len(nonempty_blocks) - 1) * temporal_gap
    if required > len(items):
        raise RuntimeError(
            "El split MD blocked_with_gap necesita mas frames de los disponibles: "
            f"{required} > {len(items)} (gap={temporal_gap}, counts={counts})."
        )
    split_ranges = {"train": [], "validation": [], "test": []}
    excluded: list[tuple[Path, str]] = []
    cursor = 0
    for block_index, split_name in enumerate(nonempty_blocks):
        count = counts[split_name]
        split_ranges[split_name] = list(items[cursor : cursor + count])
        cursor += count
        if block_index < len(nonempty_blocks) - 1 and temporal_gap > 0:
            next_split = nonempty_blocks[block_index + 1]
            for sample in items[cursor : cursor + temporal_gap]:
                excluded.append((sample, f"temporal_gap_between_{split_name}_and_{next_split}"))
            cursor += temporal_gap
    return split_ranges, excluded


def _sample_names(samples: list[Path]) -> str:
    return ", ".join(path.name for path in samples) if samples else "-"


def _link_or_copy_file(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        try:
            if src.resolve(strict=True) == dst.resolve(strict=True):
                return
        except OSError:
            pass
        try:
            if src.is_file() and dst.exists() and dst.is_file() and material_file_sha256(src) == material_file_sha256(dst):
                return
        except OSError:
            pass
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(os.path.relpath(src, dst.parent), dst)
    except OSError:
        shutil.copy2(src, dst)


def _graph2mat_basis_files_for_dataset(dataset_dir: Path) -> list[Path]:
    by_name: dict[str, Path] = {}
    for basis_dir in (
        dataset_dir / "material_basis",
        dataset_dir / "basis",
        dataset_dir / "MD_steps" / "basis",
    ):
        if not basis_dir.exists():
            continue
        for basis_file in sorted(basis_dir.glob("*.ion.xml")):
            by_name.setdefault(basis_file.name, basis_file)
    return [by_name[name] for name in sorted(by_name)]


def _materialize_graph2mat_basis_files(dataset_dir: Path, sample_dirs: list[Path]) -> int:
    basis_files = _graph2mat_basis_files_for_dataset(dataset_dir)
    if not basis_files:
        return 0
    materialized = 0
    for sample_dir in sample_dirs:
        if not sample_dir.exists() or not sample_dir.is_dir():
            continue
        for basis_file in basis_files:
            _link_or_copy_file(basis_file, sample_dir / basis_file.name)
            materialized += 1
    return materialized


def _prepare_split_sample(src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src in src_dir.iterdir():
        if src.is_file():
            _link_or_copy_file(src, dst_dir / src.name)


def _parse_float_triplet(line: str, *, path: Path, line_number: int) -> tuple[float, float, float]:
    parts = line.split()
    if len(parts) < 3:
        raise RuntimeError(f"{path}:{line_number}: expected at least 3 floats.")
    try:
        return float(parts[0]), float(parts[1]), float(parts[2])
    except ValueError as exc:
        raise RuntimeError(f"{path}:{line_number}: invalid float triplet.") from exc


def parse_xv_geometry(xv_path: Path) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, float, float, float]]]:
    """Read the geometry written by SIESTA in XV format.

    SIESTA XV coordinates and lattice vectors are in Bohr. The per-frame FDF
    files written below use Angstrom to match the generated RUN.fdf template.
    """

    lines = [line for line in xv_path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]
    if len(lines) < 4:
        raise RuntimeError(f"{xv_path}: XV file is too short to contain a geometry.")
    lattice = []
    for index in range(3):
        lattice.append(
            tuple(value * BOHR_TO_ANG for value in _parse_float_triplet(lines[index], path=xv_path, line_number=index + 1))
        )
    try:
        atom_count = int(lines[3].split()[0])
    except (IndexError, ValueError) as exc:
        raise RuntimeError(f"{xv_path}: invalid atom count line.") from exc
    if len(lines) < 4 + atom_count:
        raise RuntimeError(f"{xv_path}: expected {atom_count} atom rows, found {max(0, len(lines) - 4)}.")

    atoms: list[tuple[int, int, float, float, float]] = []
    for offset, line in enumerate(lines[4 : 4 + atom_count], start=5):
        parts = line.split()
        if len(parts) < 5:
            raise RuntimeError(f"{xv_path}:{offset}: invalid XV atom row.")
        try:
            species_index = int(parts[0])
            atomic_number = int(parts[1])
            x, y, z = (float(parts[col]) * BOHR_TO_ANG for col in (2, 3, 4))
        except ValueError as exc:
            raise RuntimeError(f"{xv_path}:{offset}: invalid XV atom values.") from exc
        atoms.append((species_index, atomic_number, x, y, z))
    return lattice, atoms


def _replace_or_append_block(text: str, block_name: str, block_lines: list[str]) -> str:
    lines = text.splitlines()
    lower_name = block_name.lower()
    output: list[str] = []
    index = 0
    replaced = False
    while index < len(lines):
        clean = lines[index].split("#", 1)[0].strip().lower()
        if clean == f"%block {lower_name}":
            output.append(f"%block {block_name}")
            output.extend(block_lines)
            output.append(f"%endblock {block_name}")
            index += 1
            while index < len(lines):
                end_clean = lines[index].split("#", 1)[0].strip().lower()
                index += 1
                if end_clean == f"%endblock {lower_name}":
                    break
            replaced = True
            continue
        output.append(lines[index])
        index += 1

    if not replaced:
        if output and output[-1].strip():
            output.append("")
        output.append(f"%block {block_name}")
        output.extend(block_lines)
        output.append(f"%endblock {block_name}")
    return "\n".join(output).rstrip() + "\n"


def _set_fdf_directive(text: str, key: str, value: str) -> str:
    lines = text.splitlines()
    output: list[str] = []
    inserted = False
    lower_key = key.lower()
    for line in lines:
        clean = line.split("#", 1)[0].strip()
        first = clean.split(None, 1)[0].lower() if clean else ""
        if first == lower_key:
            if not inserted:
                output.append(f"{key} {value}")
                inserted = True
            continue
        output.append(line)
    if not inserted:
        output.append(f"{key} {value}")
    return "\n".join(output).rstrip() + "\n"


MD_RUN_FDF_XV_MARKER = (
    "# Graph2Mat MD geometry materialized from siesta.XV after SIESTA; "
    "reference matrix timestamp may predate this RUN.fdf rewrite."
)


def rewrite_run_fdf_from_xv(run_fdf_path: Path, xv_path: Path) -> None:
    lattice, atoms = parse_xv_geometry(xv_path)
    lattice_lines = [f"{x:.12f} {y:.12f} {z:.12f}" for x, y, z in lattice]
    atom_lines = [
        f"{x:.12f} {y:.12f} {z:.12f} {species_index}"
        for species_index, _atomic_number, x, y, z in atoms
    ]
    text = run_fdf_path.read_text(encoding="utf-8", errors="ignore")
    text = _set_fdf_directive(text, "LatticeConstant", "1.0 Ang")
    text = _set_fdf_directive(text, "AtomicCoordinatesFormat", "Ang")
    text = _replace_or_append_block(text, "LatticeVectors", lattice_lines)
    text = _replace_or_append_block(text, "AtomicCoordinatesAndAtomicSpecies", atom_lines)
    if MD_RUN_FDF_XV_MARKER not in text:
        text = text.rstrip() + "\n\n" + MD_RUN_FDF_XV_MARKER + "\n"
    run_fdf_path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def read_sample_metadata(sample_dir: Path) -> dict:
    metadata_path = sample_dir / "metadata.json"
    if not metadata_path.exists():
        return {}
    try:
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def read_joint_fdf_output_flags(run_fdf_path: Path) -> dict[str, str]:
    """Return the joint benchmark output directives present in a RUN.fdf."""

    if not run_fdf_path.exists():
        return {}
    wanted = {flag.lower(): flag for flag in JOINT_REQUIRED_FDF_OUTPUT_FLAGS}
    found: dict[str, str] = {}
    for line in run_fdf_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        clean = line.split("#", 1)[0].strip()
        if not clean:
            continue
        parts = clean.split(None, 1)
        key = wanted.get(parts[0].lower())
        if key:
            found[key] = parts[1].strip() if len(parts) > 1 else ""
    return found


def _expected_artifact_name(requirement, system_label: str) -> str:
    if requirement.filenames:
        return requirement.filenames[0]
    if requirement.system_label_suffix:
        return f"{system_label}{requirement.system_label_suffix}"
    return requirement.key


def write_joint_snapshot_metadata(
    sample_dir: Path,
    config: dict,
    *,
    extra: dict | None = None,
    validation_status: str = "pending_joint_artifact_validation",
) -> dict:
    """Write per-snapshot provenance for the joint Graph2Mat/DeepH contract."""

    metadata = read_sample_metadata(sample_dir)
    if extra:
        metadata.update(extra)

    system_label, label_errors, label_warnings = resolve_system_label(sample_dir)
    if system_label is None:
        fallback_label = str((config.get("md") or {}).get("system_label") or "").strip() or None
        system_label, label_errors, label_warnings = resolve_system_label(
            sample_dir,
            default=fallback_label,
        )
    system_label = system_label or str((config.get("md") or {}).get("system_label") or "siesta")

    artifacts: dict[str, dict[str, object]] = {}
    for requirement in snapshot_requirements(system_label):
        expected_name = _expected_artifact_name(requirement, system_label)
        artifact_path = (
            sample_dir / "metadata.json"
            if requirement.key == "metadata"
            else find_artifact(sample_dir, requirement, system_label)
        )
        artifact_info: dict[str, object] = {
            "filename": artifact_path.name if artifact_path else expected_name,
            "path": str(artifact_path if artifact_path else sample_dir / expected_name),
            "required": requirement.required,
            "present": bool(requirement.key == "metadata" or (artifact_path and artifact_path.exists())),
        }
        if artifact_path and artifact_path.exists() and artifact_path.is_file() and requirement.key != "metadata":
            artifact_info["sha256"] = material_file_sha256(artifact_path)
        artifacts[requirement.key] = artifact_info

    run_fdf_path = sample_dir / "RUN.fdf"
    source_run_fdf = Path(str(metadata.get("source_run_fdf") or ""))
    if source_run_fdf and not source_run_fdf.is_absolute():
        source_run_fdf = sample_dir / source_run_fdf
    frame_index = metadata.get("frame_index", metadata.get("source_frame_index", sample_dir.name))
    metadata.update(
        {
            "sample_id": str(metadata.get("sample_id") or metadata.get("global_sample_id") or f"md_{sample_dir.name}"),
            "snapshot_dir": str(sample_dir),
            "system_label": system_label,
            "siesta_system_label": system_label,
            "artifact_contract_name": CONTRACT_NAME,
            "artifact_contract_version": CONTRACT_NAME,
            "artifact_contract_validation_status": validation_status,
            "generation_mode": "clean_one_pass",
            "source": "graph2mat_vs_deeph_dataset_generation",
            "artifacts": artifacts,
            "joint_store_files": JOINT_GRAPH2MAT_DEEPH_STORE_FILES,
            "joint_store_file_patterns": JOINT_GRAPH2MAT_DEEPH_STORE_FILES.split(),
            "fdf_output_flags": read_joint_fdf_output_flags(run_fdf_path),
            "fdf_output_flags_required": list(JOINT_REQUIRED_FDF_OUTPUT_FLAGS),
            "frame_index": str(frame_index),
            "time_index": str(metadata.get("time_index", frame_index)),
        }
    )
    if label_errors:
        metadata["system_label_resolution_errors"] = label_errors
    if label_warnings:
        metadata["system_label_resolution_warnings"] = label_warnings
    if run_fdf_path.exists():
        metadata["run_fdf_sha256"] = material_file_sha256(run_fdf_path)
    if source_run_fdf.exists() and source_run_fdf.is_file():
        metadata["source_run_fdf_sha256"] = material_file_sha256(source_run_fdf)

    write_json(sample_dir / "metadata.json", metadata)
    return metadata


def update_joint_snapshot_validation_metadata(result) -> None:
    for snapshot in result.snapshots:
        metadata = read_sample_metadata(snapshot.snapshot_dir)
        if not metadata:
            continue
        metadata["artifact_contract_validation_status"] = "valid" if snapshot.valid else "invalid"
        metadata["artifact_contract_validation"] = snapshot.to_dict()
        write_json(snapshot.snapshot_dir / "metadata.json", metadata)


def md_sample_manifest_fields(sample_dir: Path) -> dict[str, str]:
    metadata = read_sample_metadata(sample_dir)
    def text(key: str) -> str:
        value = metadata.get(key)
        return "" if value is None else str(value)
    return {
        "metadata_path": str(sample_dir / "metadata.json") if metadata else "",
        "seed": text("seed"),
        "temperature_K": text("temperature_K"),
        "md_block_id": text("source_block_id"),
        "md_block_label": text("source_block_label"),
        "md_source_block_dir": text("source_block_dir"),
        "md_source_frame_index": text("source_frame_index"),
        "timestep_fs": text("timestep_fs"),
        "md_type_of_run": text("type_of_run"),
        "block_id": text("source_block_id"),
        "block_label": text("source_block_label"),
        "sample_index_within_block": text("sample_index_within_block"),
    }


def effective_fdf_geometry_signature(run_fdf_path: Path) -> tuple[str, ...]:
    """Return the geometry blocks Graph2Mat can see in RUN.fdf.

    Tests use this to catch the historical failure mode where all MD frames had
    identical effective input geometries despite different ``siesta.XV`` files.
    """

    text = run_fdf_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    capture = None
    blocks: list[str] = []
    for line in text:
        clean = line.split("#", 1)[0].strip()
        lower = clean.lower()
        if lower in {"%block latticevectors", "%block atomiccoordinatesandatomicspecies"}:
            capture = lower
            blocks.append(lower)
            continue
        if lower in {"%endblock latticevectors", "%endblock atomiccoordinatesandatomicspecies"}:
            capture = None
            continue
        if capture and clean:
            blocks.append(" ".join(clean.split()))
    return tuple(blocks)


def xv_geometry_signature(xv_path: Path) -> tuple[str, ...]:
    lattice, atoms = parse_xv_geometry(xv_path)
    return tuple(
        [f"L {x:.10f} {y:.10f} {z:.10f}" for x, y, z in lattice]
        + [
            f"A {species_index} {atomic_number} {x:.10f} {y:.10f} {z:.10f}"
            for species_index, atomic_number, x, y, z in atoms
        ]
    )


def md_step_xv_path(step_dir: Path) -> Path:
    preferred = step_dir / "siesta.XV"
    if preferred.exists():
        return preferred
    candidates = sorted(step_dir.glob("*.XV"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise RuntimeError(
            f"Missing per-step XV geometry: expected {preferred} or a single *.XV in {step_dir}"
        )
    names = ", ".join(path.name for path in candidates)
    raise RuntimeError(f"Ambiguous per-step XV geometry in {step_dir}: {names}")


def refresh_md_step_geometries(config: dict) -> None:
    pipeline_paths = paths(config)
    steps_dir = pipeline_paths["dataset_dir"] / "MD_steps"
    step_dirs = sorted(
        (path for path in steps_dir.iterdir() if path.is_dir() and path.name.isdigit()),
        key=lambda path: int(path.name),
    )
    if not step_dirs:
        raise RuntimeError(f"No MD step directories found in {steps_dir}.")

    rewritten = 0
    xv_signatures: set[tuple[str, ...]] = set()
    for step_dir in step_dirs:
        run_fdf_path = step_dir / "RUN.fdf"
        if not run_fdf_path.exists():
            raise RuntimeError(f"Missing per-step RUN.fdf: {run_fdf_path}")
        xv_path = md_step_xv_path(step_dir)
        xv_signatures.add(xv_geometry_signature(xv_path))
        rewrite_run_fdf_from_xv(run_fdf_path, xv_path)
        metadata = read_sample_metadata(step_dir)
        metadata.update(
            {
                "run_fdf_geometry_source": xv_path.name,
                "run_fdf_rewritten_from_xv": True,
                "run_fdf_rewrite_time_policy": "post_siesta_geometry_materialization",
            }
        )
        write_joint_snapshot_metadata(step_dir, config, extra=metadata)
        rewritten += 1

    signatures = {effective_fdf_geometry_signature(step_dir / "RUN.fdf") for step_dir in step_dirs}
    if len(xv_signatures) > 1 and len(signatures) <= 1:
        raise RuntimeError(
            "MD geometry validation failed: multiple MD frames still expose the same "
            "effective RUN.fdf geometry to Graph2Mat."
        )
    materialized = _materialize_graph2mat_basis_files(pipeline_paths["dataset_dir"], step_dirs)
    print(f"[OK] Geometrias MD por frame escritas en RUN.fdf: {rewritten} muestras.")
    if materialized:
        print(f"[OK] Basis Graph2Mat materializada en snapshots MD: {materialized} enlaces/copias.")


def _copy_pseudopotentials_for_block(source_dir: Path, block_dir: Path) -> None:
    block_dir.mkdir(parents=True, exist_ok=True)
    for pseudo in sorted(list(source_dir.glob("*.psf")) + list(source_dir.glob("*.psml"))):
        shutil.copy2(pseudo, block_dir / pseudo.name)
    material_provenance = source_dir / "material_provenance.json"
    if material_provenance.exists():
        shutil.copy2(material_provenance, block_dir / material_provenance.name)
    material_basis = source_dir / "material_basis"
    if material_basis.exists():
        shutil.copytree(material_basis, block_dir / "material_basis", dirs_exist_ok=True)


def _block_config(config: dict, block_dir: Path, block: dict) -> dict:
    block_config = copy.deepcopy(config)
    block_config["paths"]["dataset_dir"] = str(block_dir)
    block_config["md"]["steps"] = int(block["n_snapshots"])
    block_config["md"].pop("temperature_blocks", None)
    block_config["md"].pop("blocks", None)
    return block_config


def run_temperature_block(config: dict, block: dict, block_dir: Path) -> None:
    pipeline_paths = paths(config)
    block_dir.mkdir(parents=True, exist_ok=True)
    _copy_pseudopotentials_for_block(pipeline_paths["dataset_dir"], block_dir)
    block_config = _block_config(config, block_dir, block)
    setup_store(block_config)
    write_run_fdf(block_config, block=block)
    run_siesta_with_venv(block_config)
    refresh_md_step_geometries(block_config)
    validate_joint_benchmark_artifacts(block_config)


def combine_temperature_blocks(config: dict, blocks: list[dict]) -> None:
    pipeline_paths = paths(config)
    dataset_dir = pipeline_paths["dataset_dir"]
    blocks_root = dataset_dir / "md_temperature_blocks"
    final_steps_dir = dataset_dir / "MD_steps"
    if final_steps_dir.exists():
        shutil.rmtree(final_steps_dir)
    final_steps_dir.mkdir(parents=True, exist_ok=True)

    samples = []
    global_index = 0
    basis_copied = False
    combined_run_out = []
    for block_index, block in enumerate(blocks):
        block_id = str(block.get("block_id") or f"md_block_{block_index + 1}")
        block_label = str(block.get("label") or block_id)
        block_dir = blocks_root / block_id
        source_steps_dir = block_dir / "MD_steps"
        if not source_steps_dir.exists():
            raise RuntimeError(f"Bloque MD sin MD_steps: {source_steps_dir}")
        if not basis_copied and (source_steps_dir / "basis").exists():
            shutil.copytree(source_steps_dir / "basis", final_steps_dir / "basis")
            basis_copied = True
        step_dirs = sorted(
            (path for path in source_steps_dir.iterdir() if path.is_dir() and path.name.isdigit()),
            key=lambda path: int(path.name),
        )
        expected = int(block["n_snapshots"])
        if len(step_dirs) < expected:
            raise RuntimeError(
                f"Bloque MD {block_id} genero menos snapshots de los pedidos: {len(step_dirs)} < {expected}."
            )
        for sample_index, source_sample in enumerate(step_dirs[:expected]):
            target_sample = final_steps_dir / str(global_index)
            shutil.copytree(source_sample, target_sample)
            metadata = read_sample_metadata(source_sample)
            metadata.update(
                {
                    "generation_method": "md_temperature_block",
                    "method": "md",
                    "temperature_K": block.get("temperature_K"),
                    "n_snapshots_in_block": expected,
                    "source_block_id": block_id,
                    "source_block_label": block_label,
                    "source_block_dir": str(block_dir),
                    "source_frame_index": source_sample.name,
                    "sample_index_within_block": sample_index,
                    "global_sample_id": str(global_index),
                    "seed": block.get("seed"),
                    "timestep_fs": block.get("timestep_fs", config["md"].get("timestep_fs")),
                    "ensemble": block.get("ensemble", config["md"].get("ensemble", "nve")),
                    "thermostat": block.get("thermostat", config["md"].get("thermostat")),
                    "type_of_run": block.get("type_of_run", config["md"].get("type_of_run", "Verlet")),
                    "source_run_fdf": str(block_dir / "RUN.fdf"),
                    "source_run_out": str(block_dir / "RUN.out"),
                    "run_fdf_geometry_source": read_sample_metadata(source_sample).get(
                        "run_fdf_geometry_source",
                        "XV",
                    ),
                    "run_fdf_rewritten_from_xv": True,
                    "run_fdf_rewrite_time_policy": "post_siesta_geometry_materialization",
                }
            )
            metadata = write_joint_snapshot_metadata(target_sample, config, extra=metadata)
            samples.append(metadata)
            global_index += 1
        run_out = block_dir / "RUN.out"
        if run_out.exists():
            combined_run_out.append(f"\n# ==== MD block {block_id} ({block_label}) ====\n")
            combined_run_out.append(run_out.read_text(encoding="utf-8", errors="replace"))

    if global_index != md_total_steps(config):
        raise RuntimeError(f"Total MD combinado incorrecto: {global_index} != {md_total_steps(config)}")
    materialized = _materialize_graph2mat_basis_files(
        dataset_dir,
        sorted((path for path in final_steps_dir.iterdir() if path.is_dir() and path.name.isdigit()), key=lambda path: int(path.name)),
    )
    if materialized:
        print(f"[OK] Basis Graph2Mat materializada en dataset combinado: {materialized} enlaces/copias.")
    pipeline_paths["run_out_path"].write_text("".join(combined_run_out), encoding="utf-8")
    pipeline_paths["run_fdf_path"].write_text(render_run_fdf(config, block={"n_snapshots": md_total_steps(config)}), encoding="utf-8")
    validate_joint_benchmark_artifacts(config, final_steps_dir)
    write_json(
        dataset_dir / "md_temperature_blocks_manifest.json",
        {
            "method": "md",
            "generation_method": "md_temperature_blocks",
            "total_snapshots": global_index,
            "blocks": blocks,
            "samples": samples,
        },
    )
    print(f"[OK] Bloques MD combinados: {global_index} snapshots en {final_steps_dir}.")


def run_temperature_block_dataset(config: dict) -> None:
    blocks = md_temperature_blocks(config)
    if not blocks:
        return
    pipeline_paths = paths(config)
    dataset_dir = pipeline_paths["dataset_dir"]
    blocks_root = dataset_dir / "md_temperature_blocks"
    if blocks_root.exists():
        shutil.rmtree(blocks_root)
    blocks_root.mkdir(parents=True, exist_ok=True)
    for block in blocks:
        block_id = str(block.get("block_id") or "md_block")
        print(
            "[INFO] MD block "
            f"{block_id}: {block['n_snapshots']} snapshots, "
            f"T={block.get('temperature_K', config['md'].get('temperature_K', config['md'].get('initial_temperature_K', 300)))} K"
        )
        run_temperature_block(config, block, blocks_root / block_id)
    combine_temperature_blocks(config, blocks)


def _find_hamiltonian(sample_dir: Path) -> Path | None:
    for name in ("siesta.TSHS", "siesta.HSX"):
        path = sample_dir / name
        if path.exists():
            return path
    candidates = sorted(
        [
            path
            for path in list(sample_dir.glob("*.TSHS")) + list(sample_dir.glob("*.HSX"))
            if path.name != "ML_prediction.HSX"
        ]
    )
    return candidates[0] if candidates else None


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_excluded_gap_manifest(
    config: dict,
    split_root: Path,
    excluded_samples: list[tuple[Path, str]],
    *,
    strategy: str,
    temporal_gap: int,
) -> None:
    pipeline_paths = paths(config)
    dataset_dir = pipeline_paths["dataset_dir"]
    rows = [
        {
            "sample_id": f"md_{sample.name}",
            "method": "md",
            "source_run": str(dataset_dir),
            "frame_index": sample.name,
            "time_index": sample.name,
            "displacement_amplitude": "",
            "displacement_magnitude": "",
            "displaced_atom": "",
            "displacement_axis": "",
            "displacement_sign": "",
            "displacement_family": "",
            "structure_path": str(sample / "RUN.fdf"),
            "hamiltonian_path": str(_find_hamiltonian(sample) or ""),
            "output_path": str(pipeline_paths["run_out_path"]),
            "run_out_path": str(pipeline_paths["run_out_path"]),
            "metadata_path": str(sample / "metadata.json") if (sample / "metadata.json").exists() else "",
            "valid": False,
            "validation_reason": "excluded_temporal_gap",
            "split": "excluded_gap",
            "split_strategy": strategy,
            "temporal_gap": str(temporal_gap),
            "source_frame_index": sample.name,
            "excluded_gap_reason": reason,
            "seed": "",
            "status": "excluded",
            "sample_dir": str(sample),
            **recipe_manifest_fields(config, sample_index=sample.name),
            **md_sample_manifest_fields(sample),
        }
        for sample, reason in excluded_samples
    ]
    _write_manifest(split_root / "excluded_gap_manifest.csv", rows)


def write_split_manifests(
    config: dict,
    split_root: Path,
    split_ranges: dict[str, list[Path]],
    *,
    strategy: str,
    temporal_gap: int,
) -> None:
    pipeline_paths = paths(config)
    dataset_dir = pipeline_paths["dataset_dir"]
    run_out_path = pipeline_paths["run_out_path"]
    for split_name, source_samples in split_ranges.items():
        rows = []
        for source_sample in source_samples:
            sample_dir = split_root / split_name / source_sample.name
            structure_path = sample_dir / "RUN.fdf"
            hamiltonian_path = _find_hamiltonian(sample_dir)
            sample_metadata = md_sample_manifest_fields(sample_dir)
            rows.append(
                {
                    "sample_id": f"md_{source_sample.name}",
                    "method": "md",
                    "source_run": str(dataset_dir),
                    "frame_index": source_sample.name,
                    "time_index": source_sample.name,
                    "displacement_amplitude": "",
                    "displacement_magnitude": "",
                    "displaced_atom": "",
                    "displacement_axis": "",
                    "displacement_sign": "",
                    "displacement_family": "",
                    "structure_path": str(structure_path),
                    "hamiltonian_path": str(hamiltonian_path or ""),
                    "output_path": str(run_out_path),
                    "run_out_path": str(run_out_path),
                    "metadata_path": sample_metadata.get("metadata_path", ""),
                    "valid": bool(structure_path.exists() and hamiltonian_path and run_out_path.exists()),
                    "validation_reason": "ok" if structure_path.exists() and hamiltonian_path and run_out_path.exists() else "missing_run_fdf_or_matrix_or_output",
                    "split": split_name,
                    "split_strategy": strategy,
                    "temporal_gap": str(temporal_gap),
                    "source_frame_index": source_sample.name,
                    "excluded_gap_reason": "",
                    "seed": sample_metadata.get("seed", ""),
                    "status": "completed" if structure_path.exists() and hamiltonian_path and run_out_path.exists() else "incomplete",
                    "sample_dir": str(sample_dir),
                    **recipe_manifest_fields(config, sample_index=source_sample.name),
                    **sample_metadata,
                }
            )
        _write_manifest(split_root / f"{split_name}_manifest.csv", rows)


def write_split_summary(
    split_root: Path,
    split_ranges: dict[str, list[Path]],
    excluded_samples: list[tuple[Path, str]],
    *,
    strategy: str,
    temporal_gap: int,
    warnings: list[str],
) -> None:
    split_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "strategy": strategy,
        "temporal_gap": temporal_gap,
        "counts": {split_name: len(samples) for split_name, samples in split_ranges.items()},
        "excluded_gap_count": len(excluded_samples),
        "excluded_gap_samples": [
            {
                "sample_id": f"md_{sample.name}",
                "frame_index": sample.name,
                "excluded_gap_reason": reason,
            }
            for sample, reason in excluded_samples
        ],
        "warnings": warnings,
        "scientific_status": (
            "exploratory_temporal_leakage_risk"
            if strategy == "spread"
            else "temporal_gap_split" if strategy == "blocked_with_gap" else "blocked_split"
        ),
    }
    (split_root / "split_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def prepare_dataset_splits(config: dict) -> None:
    split_config = config.get("splits", {})
    if not bool(split_config.get("enabled", False)):
        return

    pipeline_paths = paths(config)
    steps_dir = pipeline_paths["dataset_dir"] / "MD_steps"
    split_root = pipeline_paths["dataset_dir"] / "splits"
    step_dirs = sorted(
        (path for path in steps_dir.iterdir() if path.is_dir() and path.name.isdigit()),
        key=lambda path: int(path.name),
    )
    total = md_total_steps(config)
    if len(step_dirs) < total:
        raise RuntimeError(
            f"Se esperaban {total} muestras MD, pero solo hay {len(step_dirs)} en {steps_dir}."
        )

    default_train, default_validation, default_test = _split_counts(total)
    train_count = int(split_config.get("train", default_train))
    validation_count = int(split_config.get("validation", default_validation))
    test_count = int(split_config.get("test", default_test))
    requested = train_count + validation_count + test_count
    if requested > len(step_dirs):
        raise RuntimeError(
            "El split MD pide mas muestras de las disponibles: "
            f"{requested} > {len(step_dirs)}."
        )

    if split_root.exists():
        shutil.rmtree(split_root)

    strategy = str(split_config.get("strategy", "blocked_with_gap")).strip().lower()
    temporal_gap_default = 1 if strategy == "blocked_with_gap" else 0
    temporal_gap = int(split_config.get("temporal_gap", temporal_gap_default) or 0)
    if temporal_gap < 0:
        raise RuntimeError("splits.temporal_gap debe ser >= 0.")
    warnings: list[str] = []
    excluded_gap_samples: list[tuple[Path, str]] = []
    if strategy == "spread":
        warnings.append(SPREAD_SPLIT_WARNING)
        print(f"[WARN] {SPREAD_SPLIT_WARNING}")
        selected = _select_spread(step_dirs, requested)
        split_ranges = _split_spread(selected, train_count, validation_count, test_count)
    elif strategy == "block":
        selected = _select_spread(step_dirs, requested)
        split_ranges = _split_block(selected, train_count, validation_count, test_count)
    elif strategy == "blocked_with_gap":
        if temporal_gap <= 0:
            raise RuntimeError("splits.temporal_gap debe ser > 0 para blocked_with_gap.")
        counts = {"train": train_count, "validation": validation_count, "test": test_count}
        split_ranges, excluded_gap_samples = _split_blocked_with_gap(
            step_dirs,
            counts,
            temporal_gap=temporal_gap,
            block_order=_parse_block_order(split_config.get("block_order", "train,validation,test")),
        )
    else:
        raise RuntimeError(f"Estrategia de split MD no soportada: {strategy!r}.")
    for split_name, samples in split_ranges.items():
        for sample_dir in samples:
            _prepare_split_sample(sample_dir, split_root / split_name / sample_dir.name)
    split_sample_dirs = sorted(path for path in split_root.glob("*/*") if path.is_dir())
    materialized = _materialize_graph2mat_basis_files(pipeline_paths["dataset_dir"], split_sample_dirs)
    if materialized:
        print(f"[OK] Basis Graph2Mat materializada en splits MD: {materialized} enlaces/copias.")
    write_split_manifests(config, split_root, split_ranges, strategy=strategy, temporal_gap=temporal_gap)
    if excluded_gap_samples:
        write_excluded_gap_manifest(
            config,
            split_root,
            excluded_gap_samples,
            strategy=strategy,
            temporal_gap=temporal_gap,
        )
    write_split_summary(
        split_root,
        split_ranges,
        excluded_gap_samples,
        strategy=strategy,
        temporal_gap=temporal_gap,
        warnings=warnings,
    )
    print(
        "[OK] Split MD preparado: "
        f"{train_count} train, {test_count} test, {validation_count} validation "
        f"en {split_root} (strategy={strategy})"
    )
    artifact_validation_path = pipeline_paths["dataset_dir"] / "artifact_validation.json"
    if artifact_validation_path.exists():
        dataset_manifest, frozen_split = write_benchmark_manifests(
            dataset_root=pipeline_paths["dataset_dir"],
            split_root=split_root,
            generation_mode="clean_one_pass",
            strict_paper_ready_provenance=True,
        )
        print(
            "[OK] Benchmark dataset congelado: "
            f"{dataset_manifest['benchmark_dataset_id']} split_hash={frozen_split['split_hash']}"
        )
    else:
        print(
            "[INFO] No se escribe benchmark_dataset_manifest.json: "
            f"no existe {artifact_validation_path}."
        )
    print(f"[INFO] MD train samples: {_sample_names(split_ranges['train'])}")
    print(f"[INFO] MD test samples: {_sample_names(split_ranges['test'])}")
    print(f"[INFO] MD validation samples: {_sample_names(split_ranges['validation'])}")


def main() -> int:
    config = load_pipeline_config()
    pipeline_paths = paths(config)

    print("=== Pipeline MD (fase 1): generación de dataset ===")
    print(f"Repositorio: {pipeline_paths['dataset_dir'].parent}")
    print(f"Dataset dir: {pipeline_paths['dataset_dir']}")

    require_command(command(config, "graph2mat"))
    require_command(command(config, "shell"))
    require_command(command(config, "siesta"))

    pipeline_paths["dataset_dir"].mkdir(parents=True, exist_ok=True)
    prepare_material_inputs(config)
    if md_temperature_blocks(config):
        run_temperature_block_dataset(config)
    else:
        setup_store(config)
        write_run_fdf(config)
        run_siesta_with_venv(config)
        refresh_md_step_geometries(config)
        validate_joint_benchmark_artifacts(config)
    prepare_dataset_splits(config)

    print("\n=== Pipeline completado correctamente ===")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
