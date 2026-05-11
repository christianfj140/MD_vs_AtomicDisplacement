#!/usr/bin/env python3
"""Generate the molecular-dynamics dataset from pipeline_config.yaml."""

from __future__ import annotations

import os
import csv
import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path

from md_pipeline_config import (
    command,
    load_pipeline_config,
    md_temperature_blocks,
    md_total_steps,
    paths,
    render_run_fdf,
)

BOHR_TO_ANG = 0.529177210903
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
        [command(config, "graph2mat"), "siesta", "md", "setup-store"],
        cwd=pipeline_paths["dataset_dir"],
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
        dst.unlink()
    try:
        os.symlink(os.path.relpath(src, dst.parent), dst)
    except OSError:
        shutil.copy2(src, dst)


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
        xv_path = step_dir / "siesta.XV"
        if not run_fdf_path.exists():
            raise RuntimeError(f"Missing per-step RUN.fdf: {run_fdf_path}")
        if not xv_path.exists():
            raise RuntimeError(f"Missing per-step siesta.XV geometry: {xv_path}")
        xv_signatures.add(xv_geometry_signature(xv_path))
        rewrite_run_fdf_from_xv(run_fdf_path, xv_path)
        metadata = read_sample_metadata(step_dir)
        metadata.update(
            {
                "run_fdf_geometry_source": "siesta.XV",
                "run_fdf_rewritten_from_xv": True,
                "run_fdf_rewrite_time_policy": "post_siesta_geometry_materialization",
            }
        )
        write_json(step_dir / "metadata.json", metadata)
        rewritten += 1

    signatures = {effective_fdf_geometry_signature(step_dir / "RUN.fdf") for step_dir in step_dirs}
    if len(xv_signatures) > 1 and len(signatures) <= 1:
        raise RuntimeError(
            "MD geometry validation failed: multiple MD frames still expose the same "
            "effective RUN.fdf geometry to Graph2Mat."
        )
    print(f"[OK] Geometrias MD por frame escritas en RUN.fdf: {rewritten} muestras.")


def _copy_pseudopotentials_for_block(source_dir: Path, block_dir: Path) -> None:
    for pseudo in sorted(list(source_dir.glob("*.psf")) + list(source_dir.glob("*.psml"))):
        shutil.copy2(pseudo, block_dir / pseudo.name)


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
            metadata = {
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
                "run_fdf_geometry_source": "siesta.XV",
                "run_fdf_rewritten_from_xv": True,
                "run_fdf_rewrite_time_policy": "post_siesta_geometry_materialization",
            }
            write_json(target_sample / "metadata.json", metadata)
            samples.append(metadata)
            global_index += 1
        run_out = block_dir / "RUN.out"
        if run_out.exists():
            combined_run_out.append(f"\n# ==== MD block {block_id} ({block_label}) ====\n")
            combined_run_out.append(run_out.read_text(encoding="utf-8", errors="replace"))

    if global_index != md_total_steps(config):
        raise RuntimeError(f"Total MD combinado incorrecto: {global_index} != {md_total_steps(config)}")
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
    pipeline_paths["run_out_path"].write_text("".join(combined_run_out), encoding="utf-8")
    pipeline_paths["run_fdf_path"].write_text(render_run_fdf(config, block={"n_snapshots": md_total_steps(config)}), encoding="utf-8")
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

    strategy = str(split_config.get("strategy", "block")).strip().lower()
    temporal_gap = int(split_config.get("temporal_gap", 0) or 0)
    if temporal_gap < 0:
        raise RuntimeError("splits.temporal_gap debe ser >= 0.")
    excluded_gap_samples: list[tuple[Path, str]] = []
    if strategy == "spread":
        selected = _select_spread(step_dirs, requested)
        split_ranges = _split_spread(selected, train_count, validation_count, test_count)
    elif strategy == "block":
        selected = _select_spread(step_dirs, requested)
        split_ranges = _split_block(selected, train_count, validation_count, test_count)
    elif strategy == "blocked_with_gap":
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
    write_split_manifests(config, split_root, split_ranges, strategy=strategy, temporal_gap=temporal_gap)
    if excluded_gap_samples:
        write_excluded_gap_manifest(
            config,
            split_root,
            excluded_gap_samples,
            strategy=strategy,
            temporal_gap=temporal_gap,
        )

    print(
        "[OK] Split MD preparado: "
        f"{train_count} train, {test_count} test, {validation_count} validation "
        f"en {split_root} (strategy={strategy})"
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
    if md_temperature_blocks(config):
        run_temperature_block_dataset(config)
    else:
        setup_store(config)
        write_run_fdf(config)
        run_siesta_with_venv(config)
        refresh_md_step_geometries(config)
    prepare_dataset_splits(config)

    print("\n=== Pipeline completado correctamente ===")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
