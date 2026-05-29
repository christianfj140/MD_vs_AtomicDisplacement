#!/usr/bin/env python3
"""DeepH configuration helpers for the joint Graph2Mat/DeepH benchmark."""

from __future__ import annotations

import configparser
import json
import os
import re
import shutil
import sys
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SHARED_DIR = REPO_ROOT / "shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

from joint_artifact_contract import validate_snapshot  # noqa: E402


DEEPh_REQUIRED_SIESTA_KEYS = ("hsx", "struct_out", "xv", "orb_indx")
DEEPh_PREPROCESS_INTERFACE = "siesta"
DEEPh_TRAIN_INTERFACE = "h5"
DEEPh_TARGET = "hamiltonian"
DEEPh_DEFAULT_DATASET_NAME = "graphene_w90_joint"


@dataclass(frozen=True)
class DeepHPaths:
    root: Path
    raw_dir: Path
    processed_dir: Path
    graph_dir: Path
    save_dir: Path
    inference_dir: Path
    config_dir: Path
    preprocess_config: Path
    train_config: Path
    manifest_path: Path


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"Required JSON file is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON file must contain an object: {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_ini(path: Path, config: configparser.ConfigParser) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        config.write(handle)


def ensure_path_inside(path: Path, root: Path, *, label: str) -> Path:
    path = Path(path).resolve(strict=False)
    root = Path(root).resolve(strict=False)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"{label} must stay inside benchmark workspace: {path} not under {root}") from exc
    return path


def safe_sample_id(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.]+", "_", str(value).strip())
    text = text.strip("._")
    return text or "sample"


def split_rows(frozen_split: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    grouped = {"train": [], "validation": [], "test": []}
    for row in frozen_split.get("rows") or []:
        split = row.get("split")
        if split in grouped:
            grouped[str(split)].append(dict(row))
    return grouped


def ratio_string(count: int, total: int) -> str:
    if total <= 0:
        raise RuntimeError("DeepH split ratio total must be positive.")
    if count < 0 or count > total:
        raise RuntimeError(f"Invalid DeepH split count {count}/{total}.")
    if count == 0:
        return "0.0"
    value = count / total
    for epsilon in (0.0, 1e-15, 1e-12, 1e-10):
        candidate = format(value + epsilon, ".16g")
        if int(float(candidate) * total) == count:
            return candidate
    raise RuntimeError(f"Could not encode DeepH split ratio for {count}/{total}.")


def deeph_split_ratios(frozen_split: dict[str, Any]) -> dict[str, str]:
    grouped = split_rows(frozen_split)
    total = sum(len(rows) for rows in grouped.values())
    return {
        "train_ratio": ratio_string(len(grouped["train"]), total),
        "val_ratio": ratio_string(len(grouped["validation"]), total),
        "test_ratio": ratio_string(len(grouped["test"]), total),
    }


def ordered_rows_for_deeph_split(frozen_split: dict[str, Any], *, seed: int) -> list[dict[str, Any]]:
    grouped = split_rows(frozen_split)
    total = sum(len(rows) for rows in grouped.values())
    if total <= 0:
        raise RuntimeError("Frozen split manifest has no rows for DeepH.")
    for split, rows in grouped.items():
        if not rows:
            raise RuntimeError(f"Frozen split manifest has no {split} rows for DeepH.")

    indices = list(range(total))
    try:
        import numpy as np  # type: ignore[import-not-found]

        np.random.seed(int(seed))
        np.random.shuffle(indices)
    except ImportError:
        random.Random(int(seed)).shuffle(indices)
    counts = {split: len(rows) for split, rows in grouped.items()}
    desired_by_index: dict[int, str] = {}
    cursor = 0
    for split in ("train", "validation", "test"):
        for index in indices[cursor : cursor + counts[split]]:
            desired_by_index[int(index)] = split
        cursor += counts[split]
    queues = {split: list(rows) for split, rows in grouped.items()}
    ordered: list[dict[str, Any]] = []
    for index in range(total):
        split = desired_by_index[index]
        row = dict(queues[split].pop(0))
        row["deeph_sorted_index"] = index
        row["deeph_expected_split"] = split
        ordered.append(row)
    return ordered


def validate_deeph_siesta_sample(sample_dir: Path) -> dict[str, Any]:
    validation = validate_snapshot(sample_dir)
    missing = [key for key in DEEPh_REQUIRED_SIESTA_KEYS if key not in validation.present_artifacts]
    if missing or validation.errors:
        raise RuntimeError(
            "DeepH SIESTA preprocess requires HSX/STRUCT_OUT/XV/ORB_INDX; "
            f"{sample_dir} is missing {missing} errors={validation.errors}"
        )
    return validation.to_dict()


def _link_or_copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(os.path.relpath(src, dst.parent), dst)
    except OSError:
        shutil.copy2(src, dst)


def build_deeph_raw_mirror(
    frozen_split_manifest: dict[str, Any],
    *,
    raw_dir: Path,
    workspace_root: Path,
    seed: int,
) -> dict[str, Any]:
    raw_dir = ensure_path_inside(raw_dir, workspace_root, label="DeepH raw_dir")
    if raw_dir.exists():
        shutil.rmtree(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for ordered_row in ordered_rows_for_deeph_split(frozen_split_manifest, seed=seed):
        source_dir = Path(str(ordered_row.get("sample_dir") or ""))
        sample_id = str(ordered_row.get("sample_id") or source_dir.name)
        validate_deeph_siesta_sample(source_dir)
        mirror_name = f"{int(ordered_row['deeph_sorted_index']):06d}_{safe_sample_id(sample_id)}"
        mirror_dir = raw_dir / mirror_name
        mirror_dir.mkdir(parents=True, exist_ok=True)
        for artifact in sorted(path for path in source_dir.iterdir() if path.is_file()):
            if artifact.name == "ML_prediction.HSX":
                continue
            _link_or_copy_file(artifact, mirror_dir / artifact.name)
        rows.append(
            {
                "sample_id": sample_id,
                "split": ordered_row.get("split"),
                "deeph_expected_split": ordered_row["deeph_expected_split"],
                "deeph_sorted_index": ordered_row["deeph_sorted_index"],
                "source_dir": str(source_dir),
                "raw_dir": str(mirror_dir),
            }
        )
    return {
        "raw_dir": str(raw_dir),
        "seed": int(seed),
        "rows": rows,
        "split_ratios": deeph_split_ratios(frozen_split_manifest),
    }


def _first_matching_file(sample_dir: Path, suffix: str) -> Path:
    matches = sorted(sample_dir.glob(f"*{suffix}"))
    if not matches:
        raise RuntimeError(f"DeepH orbital mask derivation requires *{suffix} under {sample_dir}")
    return matches[0]


def _atomic_numbers_from_struct_out(path: Path) -> dict[int, int]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 5:
        raise RuntimeError(f"STRUCT_OUT is too short for DeepH orbital mask derivation: {path}")
    try:
        num_atoms = int(lines[3].split()[0])
    except (IndexError, ValueError) as exc:
        raise RuntimeError(f"Could not parse atom count from STRUCT_OUT: {path}") from exc
    atomic_numbers: dict[int, int] = {}
    for index in range(num_atoms):
        try:
            parts = lines[4 + index].split()
            atomic_numbers[index + 1] = int(parts[1])
        except (IndexError, ValueError) as exc:
            raise RuntimeError(f"Could not parse atomic number for atom {index + 1} from {path}") from exc
    return atomic_numbers


def deeph_orbital_list_from_siesta_sample(sample_dir: Path) -> list[dict[str, list[int]]]:
    """Build DeepH basic.orbital entries from the actual SIESTA basis in ORB_INDX.

    DeepH's default.ini contains a very broad carbon orbital mask. That default is
    not valid for small SIESTA bases such as graphene SZP/DZP variants with only
    s+p orbitals per atom, so benchmark configs must write a basis-derived mask.
    """

    sample_dir = Path(sample_dir)
    struct_out = _first_matching_file(sample_dir, ".STRUCT_OUT")
    orb_indx = _first_matching_file(sample_dir, ".ORB_INDX")
    atomic_numbers = _atomic_numbers_from_struct_out(struct_out)
    header_parts = orb_indx.read_text(encoding="utf-8", errors="replace").splitlines()[0].split()
    try:
        unit_cell_orbitals = int(header_parts[0])
    except (IndexError, ValueError) as exc:
        raise RuntimeError(f"Could not parse unit-cell orbital count from {orb_indx}") from exc

    orbital_counts_by_atom: dict[int, int] = {}
    for raw_line in orb_indx.read_text(encoding="utf-8", errors="replace").splitlines()[3:]:
        parts = raw_line.split()
        if len(parts) < 16:
            continue
        try:
            orbital_id = int(parts[0])
            atom_id = int(parts[1])
        except ValueError:
            continue
        if orbital_id > unit_cell_orbitals:
            continue
        if atom_id not in atomic_numbers:
            continue
        orbital_counts_by_atom[atom_id] = orbital_counts_by_atom.get(atom_id, 0) + 1

    if not orbital_counts_by_atom:
        raise RuntimeError(f"No unit-cell orbitals could be parsed from {orb_indx}")

    orbital_counts_by_atomic_number: dict[int, int] = {}
    for atom_id, count in sorted(orbital_counts_by_atom.items()):
        atomic_number = atomic_numbers[atom_id]
        previous = orbital_counts_by_atomic_number.get(atomic_number)
        if previous is not None and previous != count:
            raise RuntimeError(
                "DeepH benchmark currently requires a consistent orbital count per element; "
                f"Z={atomic_number} has both {previous} and {count} orbitals."
            )
        orbital_counts_by_atomic_number[atomic_number] = count

    entries: list[dict[str, list[int]]] = []
    for atomic_number_i, count_i in sorted(orbital_counts_by_atomic_number.items()):
        for atomic_number_j, count_j in sorted(orbital_counts_by_atomic_number.items()):
            key = f"{atomic_number_i} {atomic_number_j}"
            for orbital_i in range(count_i):
                for orbital_j in range(count_j):
                    entries.append({key: [orbital_i, orbital_j]})
    if not entries:
        raise RuntimeError(f"DeepH orbital mask derivation produced no entries from {sample_dir}")
    return entries


def deeph_orbital_json_from_raw_mirror(raw_mirror: dict[str, Any]) -> str:
    rows = raw_mirror.get("rows") if isinstance(raw_mirror.get("rows"), list) else []
    if not rows:
        raise RuntimeError("DeepH raw mirror has no rows for orbital mask derivation.")
    sample_dir = Path(str(rows[0].get("raw_dir") or rows[0].get("source_dir") or ""))
    return json.dumps(deeph_orbital_list_from_siesta_sample(sample_dir), separators=(",", ":"))


def render_preprocess_config(
    path: Path,
    *,
    raw_dir: Path,
    processed_dir: Path,
    multiprocessing: int = 0,
    local_coordinate: bool = True,
    get_s: bool = True,
    radius: float = -1.0,
    julia_interpreter: str = "julia",
) -> Path:
    config = configparser.ConfigParser()
    config["basic"] = {
        "raw_dir": str(raw_dir),
        "processed_dir": str(processed_dir),
        "target": DEEPh_TARGET,
        "interface": DEEPh_PREPROCESS_INTERFACE,
        "multiprocessing": str(int(multiprocessing)),
        "local_coordinate": str(bool(local_coordinate)),
        "get_S": str(bool(get_s)),
    }
    config["interpreter"] = {"julia_interpreter": julia_interpreter}
    config["graph"] = {
        "radius": str(float(radius)),
        "create_from_DFT": "True",
        "r2_rand": "False",
    }
    config["magnetic_moment"] = {
        "parse_magnetic_moment": "False",
        "magnetic_element": '["Cr", "Mn", "Fe", "Co", "Ni"]',
    }
    _write_ini(path, config)
    return path


def render_train_config(
    path: Path,
    *,
    processed_dir: Path,
    graph_dir: Path,
    save_dir: Path,
    dataset_name: str,
    split_ratios: dict[str, str],
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    disable_cuda: bool,
    device: str,
    num_threads: int = -1,
    multiprocessing: int = 0,
    radius: float = -1.0,
    orbital: str | list[dict[str, list[int]]] | None = None,
    early_stopping_loss: float | None = None,
    early_stopping_loss_epoch: list[float | int] | None = None,
    overrides: dict[str, Any] | None = None,
) -> Path:
    config = configparser.ConfigParser()
    config["basic"] = {
        "graph_dir": str(graph_dir),
        "save_dir": str(save_dir),
        "raw_dir": str(processed_dir),
        "dataset_name": dataset_name,
        "interface": DEEPh_TRAIN_INTERFACE,
        "target": DEEPh_TARGET,
        "disable_cuda": str(bool(disable_cuda)),
        "device": device,
        "num_threads": str(int(num_threads)),
        "save_to_time_folder": "False",
        "save_csv": "False",
        "tb_writer": "True",
        "seed": str(int(seed)),
        "multiprocessing": str(int(multiprocessing)),
    }
    if orbital is not None:
        config["basic"]["orbital"] = orbital if isinstance(orbital, str) else json.dumps(orbital, separators=(",", ":"))
    config["graph"] = {
        "radius": str(float(radius)),
        "create_from_DFT": "True",
    }
    config["train"] = {
        "epochs": str(int(epochs)),
        "train_ratio": split_ratios["train_ratio"],
        "val_ratio": split_ratios["val_ratio"],
        "test_ratio": split_ratios["test_ratio"],
    }
    if early_stopping_loss is not None:
        config["train"]["early_stopping_loss"] = str(float(early_stopping_loss))
    if early_stopping_loss_epoch is not None:
        config["train"]["early_stopping_loss_epoch"] = json.dumps(list(early_stopping_loss_epoch))
    config["hyperparameter"] = {
        "batch_size": str(int(batch_size)),
        "learning_rate": str(float(learning_rate)),
    }
    overrides = dict(overrides or {})
    section_map = {
        "optimizer": ("hyperparameter", "optimizer"),
        "weight_decay": ("hyperparameter", "weight_decay"),
        "criterion": ("hyperparameter", "criterion"),
        "retain_edge_fea": ("hyperparameter", "retain_edge_fea"),
        "atom_fea_len": ("network", "atom_fea_len"),
        "edge_fea_len": ("network", "edge_fea_len"),
        "gauss_stop": ("network", "gauss_stop"),
        "num_l": ("network", "num_l"),
        "if_edge_update": ("network", "if_edge_update"),
        "if_lcmp": ("network", "if_lcmp"),
        "normalization": ("network", "normalization"),
        "atom_update_net": ("network", "atom_update_net"),
    }
    unknown = sorted(set(overrides) - set(section_map) - {"epochs", "batch_size", "learning_rate", "seed"})
    if unknown:
        raise RuntimeError(f"Unsupported DeepH train override keys: {', '.join(unknown)}")
    for key in ("epochs", "batch_size", "learning_rate", "seed"):
        if key not in overrides:
            continue
        if key == "epochs":
            config["train"]["epochs"] = str(int(overrides[key]))
        elif key == "batch_size":
            config["hyperparameter"]["batch_size"] = str(int(overrides[key]))
        elif key == "learning_rate":
            config["hyperparameter"]["learning_rate"] = str(float(overrides[key]))
        elif key == "seed":
            config["basic"]["seed"] = str(int(overrides[key]))
    for key, value in overrides.items():
        if key not in section_map:
            continue
        section, option = section_map[key]
        if section not in config:
            config[section] = {}
        config[section][option] = str(value)
    _write_ini(path, config)
    return path


def render_inference_config(
    path: Path,
    *,
    work_dir: Path,
    trained_model_dir: Path,
    python_interpreter: str,
    interface: str = "openmx",
    task: list[int] | None = None,
    disable_cuda: bool = True,
    device: str = "cpu",
    huge_structure: bool = True,
    restore_blocks_py: bool = True,
    radius: float = -1.0,
) -> Path:
    task = task or [3, 4]
    config = configparser.ConfigParser()
    config["basic"] = {
        "work_dir": str(work_dir),
        "OLP_dir": str(work_dir),
        "interface": interface,
        "trained_model_dir": json.dumps([str(trained_model_dir)]),
        "task": json.dumps(task),
        "sparse_calc_config": "",
        "eigen_solver": "dense_py",
        "disable_cuda": str(bool(disable_cuda)),
        "device": device,
        "huge_structure": str(bool(huge_structure)),
        "restore_blocks_py": str(bool(restore_blocks_py)),
        "gen_rc_idx": "False",
        "gen_rc_by_idx": "",
        "with_grad": "False",
    }
    config["interpreter"] = {
        "julia_interpreter": "",
        "python_interpreter": python_interpreter,
    }
    config["graph"] = {
        "radius": str(float(radius)),
        "create_from_DFT": "True",
    }
    _write_ini(path, config)
    return path


def default_deeph_paths(run_root: Path) -> DeepHPaths:
    root = Path(run_root) / "deeph"
    config_dir = root / "config"
    return DeepHPaths(
        root=root,
        raw_dir=root / "raw",
        processed_dir=root / "processed",
        graph_dir=root / "graph",
        save_dir=root / "train",
        inference_dir=root / "inference",
        config_dir=config_dir,
        preprocess_config=config_dir / "preprocess.ini",
        train_config=config_dir / "train.ini",
        manifest_path=root / "deeph_manifest.json",
    )
