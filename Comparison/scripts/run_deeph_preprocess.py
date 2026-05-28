#!/usr/bin/env python3
"""Run DeepH SIESTA preprocessing for the fair benchmark harness."""

from __future__ import annotations

import argparse
from configparser import ConfigParser
from pathlib import Path

from deeph_fair_utils import (
    DEFAULT_DEEPH_REPO,
    DeepHFairBenchmarkError,
    configparser_to_dict,
    deeph_subprocess_env,
    read_json,
    run_git_commit,
    run_subprocess_streaming,
    write_deeph_config,
    write_json,
)


def build_config(args: argparse.Namespace) -> ConfigParser:
    config = ConfigParser()
    config.read(args.deeph_repo / "deeph" / "preprocess" / "preprocess_default.ini")
    config.set("basic", "raw_dir", str(args.raw_dir.resolve()))
    config.set("basic", "processed_dir", str(args.processed_dir.resolve()))
    config.set("basic", "target", "hamiltonian")
    config.set("basic", "interface", "siesta")
    config.set("basic", "multiprocessing", str(args.multiprocessing))
    config.set("basic", "local_coordinate", str(args.local_coordinate))
    config.set("basic", "get_S", "True")
    config.set("graph", "radius", str(args.radius))
    config.set("graph", "create_from_DFT", "True")
    config.set("graph", "r2_rand", "False")
    return config


def run(args: argparse.Namespace) -> dict:
    if not args.raw_dir.exists():
        raise DeepHFairBenchmarkError(f"Raw DeepH SIESTA directory does not exist: {args.raw_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = build_config(args)
    config_path = args.output_dir / "deeph_preprocess_config.ini"
    write_deeph_config(config_path, config)
    command = [
        str(args.python),
        "-u",
        str(args.deeph_repo / "deeph" / "scripts" / "preprocess.py"),
        "--config",
        str(config_path.resolve()),
    ]
    returncode = run_subprocess_streaming(
        command,
        cwd=args.deeph_repo,
        env=deeph_subprocess_env(args.deeph_repo),
        stdout_path=args.output_dir / "deeph_preprocess_stdout.log",
        stderr_path=args.output_dir / "deeph_preprocess_stderr.log",
        prefix="[DEEPh][preprocess] ",
    )
    processed_samples = sorted(
        str(path.relative_to(args.processed_dir))
        for path in args.processed_dir.rglob("*")
        if path.is_dir() and (path / "hamiltonians.h5").exists()
    ) if args.processed_dir.exists() else []
    raw_manifest = read_json(args.raw_manifest) if args.raw_manifest else {}
    manifest = {
        "stage": "deeph_preprocess_siesta",
        "command": command,
        "returncode": returncode,
        "raw_dir": str(args.raw_dir.resolve()),
        "processed_dir": str(args.processed_dir.resolve()),
        "processed_sample_count": len(processed_samples),
        "processed_samples": processed_samples,
        "config_path": str(config_path),
        "config": configparser_to_dict(config),
        "raw_manifest": raw_manifest,
        "pipeline_git": run_git_commit(Path.cwd()),
        "deepH_git": run_git_commit(args.deeph_repo.resolve()),
    }
    write_json(args.output_dir / "deeph_preprocess_manifest.json", manifest)
    if returncode != 0 and args.fail_closed:
        raise DeepHFairBenchmarkError(
            f"DeepH preprocessing failed with return code {returncode}; see {args.output_dir}"
        )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--raw-manifest", type=Path, default=None)
    parser.add_argument("--deeph-repo", type=Path, default=DEFAULT_DEEPH_REPO)
    parser.add_argument("--python", type=Path, default=DEFAULT_DEEPH_REPO / ".venv" / "bin" / "python")
    parser.add_argument("--multiprocessing", type=int, default=0)
    parser.add_argument("--radius", type=float, default=-1.0)
    parser.add_argument("--local-coordinate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fail-closed", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    try:
        manifest = run(parse_args())
    except DeepHFairBenchmarkError as exc:
        raise SystemExit(f"[DEEPh-FAIR][ERROR] {exc}") from exc
    print(
        "[DEEPh-FAIR] preprocess complete: "
        f"returncode={manifest['returncode']} processed={manifest['processed_sample_count']} "
        f"manifest={Path(manifest['config_path']).parent / 'deeph_preprocess_manifest.json'}"
    )


if __name__ == "__main__":
    main()
