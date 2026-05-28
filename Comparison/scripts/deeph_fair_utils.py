#!/usr/bin/env python3
"""Utilities for fair Graph2Mat-vs-DeepH SIESTA benchmark harnesses."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
from configparser import ConfigParser
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
COMPARISON_ROOT = REPO_ROOT / "Comparison"
DEFAULT_DEEPH_REPO = Path(os.environ["DEEPH_PACK_ROOT"]).expanduser() if os.environ.get("DEEPH_PACK_ROOT") else Path("deeph-pack")
FORBIDDEN_REFERENCE_NAMES = {"ML_prediction.HSX"}
DEEPH_REQUIRED_SIESTA_SUFFIXES = (".HSX", ".STRUCT_OUT", ".XV", ".ORB_INDX")


class DeepHFairBenchmarkError(RuntimeError):
    """Raised when the fair-comparison contract cannot be satisfied."""


@dataclass(frozen=True)
class SplitSample:
    sample: str
    split: str
    sample_id: str
    sample_dir: Path
    structure_path: Path | None
    hamiltonian_path: Path | None
    metadata_path: Path | None
    source_row: dict[str, Any]


def sha256_file(path: Path | None) -> str | None:
    if path is None or not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row}) or ["status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows([{key: json_safe(value) for key, value in row.items()} for row in rows])


def run_git_commit(repo: Path) -> dict[str, Any]:
    if not (repo / ".git").exists():
        return {"path": str(repo), "commit": None, "dirty": None}
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=False,
        text=True,
        capture_output=True,
    )
    dirty = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain"],
        check=False,
        text=True,
        capture_output=True,
    )
    branch = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--abbrev-ref", "HEAD"],
        check=False,
        text=True,
        capture_output=True,
    )
    return {
        "path": str(repo),
        "commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "branch": branch.stdout.strip() if branch.returncode == 0 else None,
        "dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
    }


def deeph_subprocess_env(deeph_repo: Path) -> dict[str, str]:
    """Return an environment that imports the editable DeepH checkout first."""
    env = os.environ.copy()
    repo_path = str(deeph_repo.resolve())
    current = env.get("PYTHONPATH")
    env["PYTHONPATH"] = repo_path if not current else repo_path + os.pathsep + current
    env["PYTHONUNBUFFERED"] = "1"
    return env


def run_subprocess_streaming(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    stdout_path: Path,
    stderr_path: Path,
    prefix: str,
    epoch_total: int | None = None,
) -> int:
    """Run a subprocess while teeing combined output to disk and parent stdout."""
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    merged_env = dict(env)
    merged_env["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=merged_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    epoch_pattern = re.compile(r"Epoch\s*#\s*(\d+)", re.IGNORECASE)
    with stdout_path.open("w", encoding="utf-8") as stdout_log:
        stdout_log.write("$ " + " ".join(command) + "\n")
        stdout_log.flush()
        if process.stdout is not None:
            for line in process.stdout:
                stdout_log.write(line)
                stdout_log.flush()
                print(f"{prefix}{line}", end="", flush=True)
                if epoch_total is not None:
                    match = epoch_pattern.search(line)
                    if match:
                        print(
                            f"[DEEPh-FAIR][epoch] reported_epoch={match.group(1)}/{epoch_total}",
                            flush=True,
                        )
            process.stdout.close()
    returncode = process.wait()
    stderr_path.write_text(
        "stderr was merged into stdout for live UI streaming; see "
        f"{stdout_path.name}\nreturncode={returncode}\n",
        encoding="utf-8",
    )
    return returncode


def stable_sample_from_row(row: dict[str, str]) -> str:
    sample_id = str(row.get("sample_id") or "").strip()
    if sample_id:
        return sample_id
    for key in ("frame_index", "source_frame_index", "global_sample_id"):
        value = str(row.get(key) or "").strip()
        if value and value.lower() != "nan":
            return value
    for key in ("sample_dir", "structure_path", "hamiltonian_path"):
        value = str(row.get(key) or "").strip()
        if value:
            return Path(value).parent.name if key != "sample_dir" else Path(value).name
    match = re.search(r"(\d+)$", sample_id)
    if match:
        return match.group(1)
    raise DeepHFairBenchmarkError("Cannot infer stable sample id from split manifest row.")


def split_manifest_paths(graph2mat_result_dir: Path) -> dict[str, Path]:
    split_root = graph2mat_result_dir / "splits"
    return {
        "train": split_root / "train_manifest.csv",
        "validation": split_root / "validation_manifest.csv",
        "test": split_root / "test_manifest.csv",
    }


def load_split_samples(graph2mat_result_dir: Path) -> list[SplitSample]:
    samples: list[SplitSample] = []
    for split, manifest_path in split_manifest_paths(graph2mat_result_dir).items():
        rows = read_csv_rows(manifest_path)
        if not rows:
            raise DeepHFairBenchmarkError(f"Missing or empty split manifest: {manifest_path}")
        for row in rows:
            sample = stable_sample_from_row(row)
            sample_dir = Path(row.get("sample_dir") or graph2mat_result_dir / "structures" / sample)
            structure_path = Path(row["structure_path"]) if row.get("structure_path") else sample_dir / "RUN.fdf"
            hamiltonian_path = Path(row["hamiltonian_path"]) if row.get("hamiltonian_path") else None
            metadata_path = Path(row["metadata_path"]) if row.get("metadata_path") else sample_dir / "metadata.json"
            samples.append(
                SplitSample(
                    sample=sample,
                    split=split,
                    sample_id=row.get("sample_id") or sample,
                    sample_dir=sample_dir,
                    structure_path=structure_path,
                    hamiltonian_path=hamiltonian_path,
                    metadata_path=metadata_path,
                    source_row=row,
                )
            )
    return samples


def sample_limit_by_split(samples: list[SplitSample], limit: int | None) -> list[SplitSample]:
    if limit is None:
        return samples
    selected: list[SplitSample] = []
    counts: dict[str, int] = {}
    for sample in samples:
        count = counts.get(sample.split, 0)
        if count < limit:
            selected.append(sample)
            counts[sample.split] = count + 1
    return selected


def find_named_file(search_dirs: Iterable[Path], suffix: str, system_label: str = "graphene") -> Path | None:
    preferred = f"{system_label}{suffix}"
    for directory in search_dirs:
        candidate = directory / preferred
        if candidate.exists() and candidate.is_file():
            return candidate
    for directory in search_dirs:
        matches = sorted(path for path in directory.glob(f"*{suffix}") if path.name not in FORBIDDEN_REFERENCE_NAMES)
        if matches:
            return matches[0]
    return None


def detect_forbidden_references(paths: Iterable[Path | None]) -> list[str]:
    return [str(path) for path in paths if path is not None and path.name in FORBIDDEN_REFERENCE_NAMES]


def copy_or_symlink(src: Path, dst: Path, *, symlink: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if symlink:
        os.symlink(src, dst)
    else:
        shutil.copy2(src, dst)


def ensure_static_siesta_flags(run_fdf: Path) -> None:
    """Add DeepH/SIESTA export flags without changing physics controls."""
    text = run_fdf.read_text(encoding="utf-8", errors="replace")
    additions = []
    lower = text.lower()
    if "savehs" not in lower:
        additions.append("SaveHS                           true")
    if "save.hs" not in lower:
        additions.append("Save.HS                          T")
    if "ts.hs.save" not in lower:
        additions.append("TS.HS.Save                       T")
    if "xml.write" not in lower:
        additions.append("XML.Write                        T")
    if additions:
        run_fdf.write_text(text.rstrip() + "\n\n# DeepH fair-comparison export flags\n" + "\n".join(additions) + "\n", encoding="utf-8")


def infer_pseudo_dir_from_manifest(graph2mat_result_dir: Path) -> Path | None:
    manifest = read_json(graph2mat_result_dir / "manifest.json")
    provenance = manifest.get("material_provenance") if isinstance(manifest.get("material_provenance"), dict) else {}
    bundle_path = provenance.get("material_bundle_path") or manifest.get("material_bundle_path")
    if bundle_path:
        candidate = Path(bundle_path).parent / "pseudos"
        if candidate.exists():
            return candidate
    candidate = REPO_ROOT / "materials" / "graphene" / "pseudos"
    return candidate if candidate.exists() else None


def run_siesta_static(work_dir: Path, siesta_command: str, graph2mat_result_dir: Path) -> dict[str, Any]:
    pseudo_dir = infer_pseudo_dir_from_manifest(graph2mat_result_dir)
    if pseudo_dir is not None:
        for pseudo in pseudo_dir.iterdir():
            if pseudo.is_file() and pseudo.suffix.lower() in {".psf", ".psml", ".psp"}:
                copy_or_symlink(pseudo, work_dir / pseudo.name, symlink=False)
    ensure_static_siesta_flags(work_dir / "RUN.fdf")
    with (work_dir / "RUN.out").open("w", encoding="utf-8") as stdout:
        completed = subprocess.run(
            siesta_command,
            cwd=work_dir,
            stdin=(work_dir / "RUN.fdf").open("r", encoding="utf-8"),
            stdout=stdout,
            stderr=subprocess.STDOUT,
            shell=True,
            check=False,
            text=True,
        )
    return {"command": siesta_command, "returncode": completed.returncode}


def count_orbitals_from_orbital_types(path: Path) -> list[int]:
    counts: list[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        tokens = [token for token in line.split() if token.strip()]
        if not tokens:
            continue
        counts.append(sum(2 * int(token) + 1 for token in tokens))
    return counts


def orbital_config_from_processed_sample(sample_dir: Path) -> list[dict[str, list[int]]]:
    elements = [int(float(value)) for value in (sample_dir / "element.dat").read_text(encoding="utf-8").split()]
    orbital_counts = count_orbitals_from_orbital_types(sample_dir / "orbital_types.dat")
    if len(elements) != len(orbital_counts):
        raise DeepHFairBenchmarkError(
            f"element/orbital count mismatch in {sample_dir}: {len(elements)} vs {len(orbital_counts)}"
        )
    max_orbitals_by_element: dict[int, int] = {}
    for element, count in zip(elements, orbital_counts, strict=True):
        max_orbitals_by_element[element] = max(max_orbitals_by_element.get(element, 0), count)
    entries: list[dict[str, list[int]]] = []
    for row_element in sorted(max_orbitals_by_element):
        for col_element in sorted(max_orbitals_by_element):
            pair = f"{row_element} {col_element}"
            for row_orbital in range(max_orbitals_by_element[row_element]):
                for col_orbital in range(max_orbitals_by_element[col_element]):
                    entries.append({pair: [row_orbital, col_orbital]})
    return entries


def max_l_from_orbital_types(path: Path) -> int:
    values: list[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        values.extend(int(token) for token in line.split())
    return max(values) if values else 0


def configparser_to_dict(config: ConfigParser) -> dict[str, dict[str, str]]:
    return {section: dict(config.items(section)) for section in config.sections()}


def write_deeph_config(path: Path, config: ConfigParser) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        config.write(handle)


def make_split_ordered_processed_dir(
    *,
    processed_dir: Path,
    ordered_dir: Path,
    samples: list[SplitSample],
    seed: int,
    symlink: bool = True,
    shuffled_indices: list[int] | None = None,
) -> dict[str, Any]:
    split_groups = {split: [sample.sample for sample in samples if sample.split == split] for split in ("train", "validation", "test")}
    total = sum(len(values) for values in split_groups.values())
    if total == 0:
        raise DeepHFairBenchmarkError("No split samples available for DeepH split ordering.")
    train_size = len(split_groups["train"])
    validation_size = len(split_groups["validation"])
    test_size = len(split_groups["test"])
    if shuffled_indices is None:
        try:
            import numpy as np
        except ModuleNotFoundError as exc:
            raise DeepHFairBenchmarkError(
                "numpy is required to reproduce DeepH's np.random.shuffle split ordering; "
                "run this step in the DeepH environment or pass explicit shuffled_indices."
            ) from exc
        indices = list(range(total))
        np.random.seed(seed)
        np.random.shuffle(indices)
        indices = [int(index) for index in indices]
    else:
        indices = list(shuffled_indices)
        if sorted(indices) != list(range(total)):
            raise DeepHFairBenchmarkError(
                f"Invalid shuffled_indices for {total} samples: expected a permutation of 0..{total - 1}."
            )
    desired_by_index: dict[int, str] = {}
    for index, sample_id in zip(indices[:train_size], split_groups["train"], strict=True):
        desired_by_index[index] = sample_id
    offset = train_size
    for index, sample_id in zip(indices[offset:offset + validation_size], split_groups["validation"], strict=True):
        desired_by_index[index] = sample_id
    offset += validation_size
    for index, sample_id in zip(indices[offset:offset + test_size], split_groups["test"], strict=True):
        desired_by_index[index] = sample_id
    ordered_dir.mkdir(parents=True, exist_ok=True)
    mapping_rows: list[dict[str, Any]] = []
    for index in range(total):
        sample_id = desired_by_index[index]
        src = processed_dir / sample_id
        if not src.exists():
            raise DeepHFairBenchmarkError(f"Processed DeepH sample missing: {src}")
        dst = ordered_dir / f"{index:06d}__sample_{sample_id}"
        if dst.exists() or dst.is_symlink():
            if dst.is_dir() and not dst.is_symlink():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        if symlink:
            dst.mkdir(parents=True, exist_ok=True)
            for child in src.iterdir():
                os.symlink(child, dst / child.name, target_is_directory=child.is_dir())
        else:
            shutil.copytree(src, dst)
        split = next(split for split, ids in split_groups.items() if sample_id in ids)
        mapping_rows.append({"ordered_index": index, "ordered_name": dst.name, "sample": sample_id, "split": split, "source": str(src)})
    return {
        "seed": seed,
        "ordered_dir": str(ordered_dir),
        "train_size": train_size,
        "validation_size": validation_size,
        "test_size": test_size,
        "total": total,
        "train_ratio": train_size / total,
        "validation_ratio": validation_size / total,
        "test_ratio": test_size / total,
        "mapping_rows": mapping_rows,
    }
