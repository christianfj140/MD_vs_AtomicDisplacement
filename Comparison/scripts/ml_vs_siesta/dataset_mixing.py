"""Dataset partitioning / mixing manifests (no training).

A "dataset" here is just a list of sample dicts. Each sample should carry an
identifier and, for size classification, an atom count. These helpers only build
manifests and per-ratio config stubs; they never load models or train.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any, Iterable

DEFAULT_RATIOS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
_ATOM_COUNT_KEYS = ("n_atoms", "num_atoms", "atom_count", "natoms")
_ID_KEYS = ("id", "sample_id", "name", "path")


def _sample_atom_count(sample: dict[str, Any]) -> int | None:
    for key in _ATOM_COUNT_KEYS:
        if key in sample and sample[key] is not None:
            return int(sample[key])
    metadata = sample.get("metadata")
    if isinstance(metadata, dict):
        for key in _ATOM_COUNT_KEYS:
            if key in metadata and metadata[key] is not None:
                return int(metadata[key])
    return None


def _sample_id(sample: dict[str, Any], fallback: int) -> str:
    for key in _ID_KEYS:
        if key in sample and sample[key] not in (None, ""):
            return str(sample[key])
    return f"sample_{fallback}"


def classify_dataset_by_size(
    dataset: Iterable[dict[str, Any]],
    threshold_atoms: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split ``dataset`` into ``(small, large)`` by atom count.

    Samples with ``n_atoms < threshold_atoms`` are "small", the rest "large".
    Raises if a sample has no discoverable atom count.
    """
    threshold = int(threshold_atoms)
    small: list[dict[str, Any]] = []
    large: list[dict[str, Any]] = []
    for index, sample in enumerate(dataset):
        count = _sample_atom_count(sample)
        if count is None:
            raise ValueError(
                f"Sample {_sample_id(sample, index)!r} has no atom count "
                f"(expected one of {_ATOM_COUNT_KEYS}); cannot classify by size."
            )
        (small if count < threshold else large).append(sample)
    return small, large


def _select_manifest_for_ratio(
    small_ids: list[str],
    large_ids: list[str],
    ratio: float,
    mode: str,
    rng: random.Random,
) -> dict[str, Any]:
    if mode == "add":
        # Keep all small samples, add a fraction of the large ones.
        n_large = round(ratio * len(large_ids))
        chosen_large = sorted(rng.sample(large_ids, n_large)) if n_large else []
        selected = list(small_ids) + chosen_large
    elif mode == "replace":
        # Keep total size constant: replace a fraction of small samples with large.
        total = len(small_ids)
        n_replace = min(round(ratio * total), total, len(large_ids))
        chosen_large = sorted(rng.sample(large_ids, n_replace)) if n_replace else []
        kept_small = list(small_ids[: total - n_replace])
        selected = kept_small + chosen_large
    else:
        raise ValueError(f"Unknown mode {mode!r}; use 'add' or 'replace'.")
    return {
        "ratio": ratio,
        "mode": mode,
        "n_selected": len(selected),
        "n_large_selected": len(chosen_large),
        "selected_ids": selected,
    }


def make_mixed_dataset_manifest(
    small_dataset: Iterable[dict[str, Any]],
    large_dataset: Iterable[dict[str, Any]],
    ratios: Iterable[float] = DEFAULT_RATIOS,
    mode: str = "add",
    output_path: str | Path | None = None,
    *,
    seed: int = 0,
) -> dict[str, Any]:
    """Build a reproducible small/large mixing manifest.

    ``mode="add"`` appends a fraction of large samples to all small samples;
    ``mode="replace"`` swaps a fraction of small samples for large ones while
    keeping the total size constant. Output is written as JSON/YAML/CSV inferred
    from ``output_path``'s suffix (CSV = one row per ratio×sample).
    """
    small_ids = [
        _sample_id(sample, i) for i, sample in enumerate(small_dataset)
    ]
    large_ids = [
        _sample_id(sample, i) for i, sample in enumerate(large_dataset)
    ]
    ratios_list = [float(r) for r in ratios]
    for ratio in ratios_list:
        if not 0.0 <= ratio <= 1.0:
            raise ValueError(f"ratio {ratio} outside [0, 1].")

    partitions: list[dict[str, Any]] = []
    for index, ratio in enumerate(ratios_list):
        # Deterministic but ratio-dependent seed for reproducibility.
        rng = random.Random(seed * 1000 + index)
        entry = _select_manifest_for_ratio(small_ids, large_ids, ratio, mode, rng)
        entry["label"] = f"D{index}"
        partitions.append(entry)

    manifest = {
        "schema": "ml_vs_siesta_mixed_dataset_manifest_v1",
        "mode": mode,
        "seed": seed,
        "n_small": len(small_ids),
        "n_large": len(large_ids),
        "ratios": ratios_list,
        "partitions": partitions,
    }

    if output_path is not None:
        _write_manifest(manifest, Path(output_path))
        manifest["output_path"] = str(Path(output_path))
    return manifest


def _write_manifest(manifest: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        import yaml

        path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    elif suffix == ".csv":
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["label", "ratio", "mode", "sample_id"])
            for partition in manifest["partitions"]:
                for sample_id in partition["selected_ids"]:
                    writer.writerow(
                        [partition["label"], partition["ratio"], manifest["mode"], sample_id]
                    )
    else:
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def generate_mixed_dataset_configs(
    manifest: dict[str, Any],
    output_dir: str | Path,
    *,
    base_config: dict[str, Any] | None = None,
    manifest_path: str | Path | None = None,
) -> list[str]:
    """Emit one ``D<i>.yaml`` per partition (ratio/mode/seed + manifest ref).

    Purely declarative: no training is launched. Returns the written paths.
    """
    import yaml

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_ref = (
        str(manifest_path)
        if manifest_path is not None
        else manifest.get("output_path")
    )
    written: list[str] = []
    for partition in manifest.get("partitions", []):
        config: dict[str, Any] = dict(base_config or {})
        config.update(
            {
                "label": partition["label"],
                "ratio": partition["ratio"],
                "mode": manifest.get("mode"),
                "seed": manifest.get("seed"),
                "manifest": manifest_ref,
                "n_selected": partition.get("n_selected"),
            }
        )
        config_path = out_dir / f"{partition['label']}.yaml"
        config_path.write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
        written.append(str(config_path))
    return written
