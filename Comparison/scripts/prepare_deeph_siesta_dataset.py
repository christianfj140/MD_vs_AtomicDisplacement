#!/usr/bin/env python3
"""Prepare fail-closed raw SIESTA folders for a fair DeepH benchmark."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

from deeph_fair_utils import (
    DEFAULT_DEEPH_REPO,
    DEEPH_REQUIRED_SIESTA_SUFFIXES,
    DeepHFairBenchmarkError,
    copy_or_symlink,
    detect_forbidden_references,
    find_named_file,
    load_split_samples,
    read_json,
    run_git_commit,
    run_siesta_static,
    sample_limit_by_split,
    sha256_file,
    write_csv_rows,
    write_json,
)


def candidate_dirs(sample, graph2mat_result_dir: Path) -> list[Path]:
    dirs = [
        sample.sample_dir,
        sample.structure_path.parent if sample.structure_path else None,
        sample.hamiltonian_path.parent if sample.hamiltonian_path else None,
        graph2mat_result_dir / "structures" / sample.sample,
        graph2mat_result_dir / "siesta_hamiltonians" / sample.sample,
    ]
    manifest = read_json(graph2mat_result_dir / "manifest.json")
    dataset_dir = manifest.get("dataset_dir") or manifest.get("training_base_dataset_dir")
    if dataset_dir:
        dirs.extend(
            [
                Path(dataset_dir) / "MD_steps" / sample.sample,
                Path(dataset_dir) / "splits" / sample.split / sample.sample,
            ]
        )
    unique: list[Path] = []
    for directory in dirs:
        if directory is not None and directory.exists() and directory not in unique:
            unique.append(directory)
    return unique


def prepare_one_sample(
    *,
    sample,
    raw_root: Path,
    graph2mat_result_dir: Path,
    system_label: str,
    symlink: bool,
    allow_regenerate_siesta: bool,
    siesta_command: str,
    dry_run: bool,
) -> dict[str, Any]:
    out_dir = raw_root / sample.sample
    dirs = candidate_dirs(sample, graph2mat_result_dir)
    files: dict[str, Path | None] = {
        suffix: find_named_file(dirs, suffix, system_label=system_label)
        for suffix in DEEPH_REQUIRED_SIESTA_SUFFIXES
    }
    run_fdf = sample.structure_path if sample.structure_path and sample.structure_path.exists() else find_named_file(dirs, ".fdf", system_label="RUN")
    forbidden_candidates = sorted(
        {
            str(path)
            for directory in dirs
            for path in directory.glob("ML_prediction.HSX")
        }
    )
    forbidden = list(forbidden_candidates)
    forbidden.extend(item for item in detect_forbidden_references(files.values()) if item not in forbidden)
    missing = [suffix for suffix, path in files.items() if path is None]
    regeneration: dict[str, Any] | None = None

    if missing and allow_regenerate_siesta and run_fdf is not None and not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        copy_or_symlink(run_fdf, out_dir / "RUN.fdf", symlink=False)
        regeneration = run_siesta_static(out_dir, siesta_command, graph2mat_result_dir)
        dirs = [out_dir, *dirs]
        files = {
            suffix: find_named_file(dirs, suffix, system_label=system_label)
            for suffix in DEEPH_REQUIRED_SIESTA_SUFFIXES
        }
        missing = [suffix for suffix, path in files.items() if path is None]
        forbidden = detect_forbidden_references(files.values())
        if missing:
            forbidden.extend(item for item in forbidden_candidates if item not in forbidden)

    status = "ok" if not missing and not forbidden and run_fdf is not None else "missing_required_artifacts"
    row: dict[str, Any] = {
        "sample": sample.sample,
        "sample_id": sample.sample_id,
        "split": sample.split,
        "status": status,
        "raw_dir": str(out_dir),
        "run_fdf": str(run_fdf) if run_fdf else None,
        "run_fdf_sha256": sha256_file(run_fdf),
        "missing_suffixes": missing,
        "forbidden_references": forbidden,
        "candidate_dirs": [str(path) for path in dirs],
        "regeneration": regeneration,
    }
    for suffix, src in files.items():
        key = suffix.lower().removeprefix(".")
        row[f"{key}_source"] = str(src) if src else None
        row[f"{key}_sha256"] = sha256_file(src)

    if status != "ok" or dry_run:
        return row

    if regeneration is not None:
        if sample.metadata_path and sample.metadata_path.exists():
            copy_or_symlink(sample.metadata_path, out_dir / "metadata.json", symlink=False)
        return row

    if out_dir.exists():
        for item in out_dir.iterdir():
            if item.name != "RUN.out":
                if item.is_dir() and not item.is_symlink():
                    shutil.rmtree(item)
                else:
                    item.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)
    copy_or_symlink(run_fdf, out_dir / "RUN.fdf", symlink=symlink)
    for suffix, src in files.items():
        assert src is not None
        copy_or_symlink(src, out_dir / f"{system_label}{suffix}", symlink=symlink)
    if sample.metadata_path and sample.metadata_path.exists():
        copy_or_symlink(sample.metadata_path, out_dir / "metadata.json", symlink=symlink)
    return row


def run(args: argparse.Namespace) -> dict[str, Any]:
    graph2mat_result_dir = args.graph2mat_result_dir.resolve()
    output_dir = args.output_dir.resolve()
    raw_root = output_dir / "raw"
    samples = sample_limit_by_split(load_split_samples(graph2mat_result_dir), args.sample_limit_per_split)
    rows: list[dict[str, Any]] = []
    for index, sample in enumerate(samples, start=1):
        print(
            f"[DEEPh-FAIR][raw] preparing {index}/{len(samples)} "
            f"sample={sample.sample} split={sample.split}",
            flush=True,
        )
        row = prepare_one_sample(
            sample=sample,
            raw_root=raw_root,
            graph2mat_result_dir=graph2mat_result_dir,
            system_label=args.system_label,
            symlink=not args.copy,
            allow_regenerate_siesta=args.allow_regenerate_siesta,
            siesta_command=args.siesta_command,
            dry_run=args.dry_run,
        )
        rows.append(row)
        missing = ",".join(row["missing_suffixes"]) if row["missing_suffixes"] else "none"
        print(
            f"[DEEPh-FAIR][raw] done {index}/{len(samples)} "
            f"sample={sample.sample} status={row['status']} missing={missing}",
            flush=True,
        )
    failed = [row for row in rows if row["status"] != "ok"]
    manifest = {
        "stage": "deeph_raw_siesta_prepare",
        "graph2mat_result_dir": str(graph2mat_result_dir),
        "output_dir": str(output_dir),
        "raw_dir": str(raw_root),
        "deepH_repo": str(args.deeph_repo.resolve()),
        "system_label": args.system_label,
        "dry_run": args.dry_run,
        "allow_regenerate_siesta": args.allow_regenerate_siesta,
        "siesta_command": args.siesta_command if args.allow_regenerate_siesta else None,
        "samples_requested": len(samples),
        "samples_ready": len(rows) - len(failed),
        "samples_failed": len(failed),
        "required_siesta_suffixes": list(DEEPH_REQUIRED_SIESTA_SUFFIXES),
        "forbidden_reference_filenames": ["ML_prediction.HSX"],
        "pipeline_git": run_git_commit(Path.cwd()),
        "deepH_git": run_git_commit(args.deeph_repo.resolve()),
        "rows": rows,
    }
    write_csv_rows(output_dir / "deeph_raw_samples.csv", rows)
    write_json(output_dir / "deeph_raw_manifest.json", manifest)
    if failed and args.fail_closed:
        missing = output_dir / "deeph_raw_manifest.json"
        raise DeepHFairBenchmarkError(
            f"DeepH raw dataset is incomplete for {len(failed)} samples. See {missing}"
        )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph2mat-result-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--deeph-repo", type=Path, default=DEFAULT_DEEPH_REPO)
    parser.add_argument("--system-label", default="graphene")
    parser.add_argument("--sample-limit-per-split", type=int, default=None)
    parser.add_argument("--allow-regenerate-siesta", action="store_true")
    parser.add_argument("--siesta-command", default="siesta")
    parser.add_argument("--copy", action="store_true", help="Copy files instead of symlinking.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-closed", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    try:
        manifest = run(parse_args())
    except DeepHFairBenchmarkError as exc:
        raise SystemExit(f"[DEEPh-FAIR][ERROR] {exc}") from exc
    print(
        "[DEEPh-FAIR] raw prepare complete: "
        f"{manifest['samples_ready']}/{manifest['samples_requested']} ready; "
        f"manifest={Path(manifest['output_dir']) / 'deeph_raw_manifest.json'}"
    )


if __name__ == "__main__":
    main()
