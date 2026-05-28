#!/usr/bin/env python3
"""Train DeepH on a split-ordered processed SIESTA dataset."""

from __future__ import annotations

import argparse
import json
from configparser import ConfigParser
from pathlib import Path

from deeph_fair_utils import (
    DEFAULT_DEEPH_REPO,
    DeepHFairBenchmarkError,
    configparser_to_dict,
    deeph_subprocess_env,
    load_split_samples,
    make_split_ordered_processed_dir,
    max_l_from_orbital_types,
    orbital_config_from_processed_sample,
    run_git_commit,
    run_subprocess_streaming,
    sample_limit_by_split,
    write_csv_rows,
    write_deeph_config,
    write_json,
)


def first_processed_sample(processed_dir: Path) -> Path:
    candidates = sorted(path for path in processed_dir.iterdir() if path.is_dir() and (path / "orbital_types.dat").exists())
    if not candidates:
        raise DeepHFairBenchmarkError(f"No processed DeepH samples found in {processed_dir}")
    return candidates[0]


def build_train_config(args: argparse.Namespace, ordered: dict, first_sample: Path) -> ConfigParser:
    config = ConfigParser()
    config.read(args.deeph_repo / "deeph" / "default.ini")
    save_dir = args.output_dir / "training"
    graph_dir = args.output_dir / "graph_cache"
    config.set("basic", "graph_dir", str(graph_dir.resolve()))
    config.set("basic", "save_dir", str(save_dir.resolve()))
    config.set("basic", "raw_dir", str(Path(ordered["ordered_dir"]).resolve()))
    config.set("basic", "dataset_name", args.dataset_name)
    config.set("basic", "interface", "h5")
    config.set("basic", "target", "hamiltonian")
    config.set("basic", "disable_cuda", str(args.disable_cuda))
    config.set("basic", "device", args.device)
    config.set("basic", "save_to_time_folder", "False")
    config.set("basic", "save_csv", "True")
    config.set("basic", "tb_writer", str(args.tensorboard))
    config.set("basic", "seed", str(args.seed))
    config.set("basic", "multiprocessing", str(args.multiprocessing))
    config.set("basic", "orbital", json.dumps(orbital_config_from_processed_sample(first_sample)))
    config.set("basic", "max_element", str(args.max_element))
    config.set("graph", "radius", str(args.radius))
    config.set("graph", "create_from_DFT", "True")
    config.set("graph", "if_lcmp_graph", "True")
    config.set("train", "epochs", str(args.epochs))
    config.set("train", "train_ratio", f"{ordered['train_ratio']:.17g}")
    config.set("train", "val_ratio", f"{ordered['validation_ratio']:.17g}")
    config.set("train", "test_ratio", f"{ordered['test_ratio']:.17g}")
    config.set("hyperparameter", "batch_size", str(args.batch_size))
    config.set("hyperparameter", "learning_rate", str(args.learning_rate))
    config.set("network", "num_l", str(max_l_from_orbital_types(first_sample / "orbital_types.dat") + 1))
    return config


def run(args: argparse.Namespace) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples = sample_limit_by_split(load_split_samples(args.graph2mat_result_dir.resolve()), args.sample_limit_per_split)
    ordered = make_split_ordered_processed_dir(
        processed_dir=args.processed_dir.resolve(),
        ordered_dir=args.output_dir / "processed_split_ordered",
        samples=samples,
        seed=args.seed,
        symlink=not args.copy,
    )
    write_csv_rows(args.output_dir / "deeph_split_order_mapping.csv", ordered["mapping_rows"])
    first_sample = first_processed_sample(args.processed_dir.resolve())
    config = build_train_config(args, ordered, first_sample)
    config_path = args.output_dir / "deeph_train_config.ini"
    write_deeph_config(config_path, config)
    command = [
        str(args.python),
        "-u",
        str(args.deeph_repo / "deeph" / "scripts" / "train.py"),
        "--config",
        str(config_path.resolve()),
    ]
    if args.dry_run:
        returncode = None
        (args.output_dir / "deeph_train_stdout.log").write_text("", encoding="utf-8")
        (args.output_dir / "deeph_train_stderr.log").write_text("", encoding="utf-8")
    else:
        returncode = run_subprocess_streaming(
            command,
            cwd=args.deeph_repo,
            env=deeph_subprocess_env(args.deeph_repo),
            stdout_path=args.output_dir / "deeph_train_stdout.log",
            stderr_path=args.output_dir / "deeph_train_stderr.log",
            prefix="[DEEPh][train] ",
            epoch_total=args.epochs,
        )
    manifest = {
        "stage": "deeph_train",
        "dry_run": args.dry_run,
        "command": command,
        "returncode": returncode,
        "graph2mat_result_dir": str(args.graph2mat_result_dir.resolve()),
        "processed_dir": str(args.processed_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "config_path": str(config_path),
        "config": configparser_to_dict(config),
        "split_ordering_policy": "deeph_random_split_reproduced_by_ordered_symlink_dataset",
        "split_order": {key: value for key, value in ordered.items() if key != "mapping_rows"},
        "pipeline_git": run_git_commit(Path.cwd()),
        "deepH_git": run_git_commit(args.deeph_repo.resolve()),
    }
    write_json(args.output_dir / "deeph_train_manifest.json", manifest)
    if returncode not in (0, None) and args.fail_closed:
        raise DeepHFairBenchmarkError(f"DeepH training failed with return code {returncode}; see {args.output_dir}")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph2mat-result-dir", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--deeph-repo", type=Path, default=DEFAULT_DEEPH_REPO)
    parser.add_argument("--python", type=Path, default=DEFAULT_DEEPH_REPO / ".venv" / "bin" / "python")
    parser.add_argument("--dataset-name", default="graphene_w90_siesta_fair")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=4e-4)
    parser.add_argument("--radius", type=float, default=-1.0)
    parser.add_argument("--max-element", type=int, default=6)
    parser.add_argument("--multiprocessing", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--disable-cuda", action="store_true")
    parser.add_argument("--tensorboard", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sample-limit-per-split", type=int, default=None)
    parser.add_argument("--copy", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-closed", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    try:
        manifest = run(parse_args())
    except DeepHFairBenchmarkError as exc:
        raise SystemExit(f"[DEEPh-FAIR][ERROR] {exc}") from exc
    print(
        "[DEEPh-FAIR] train stage complete: "
        f"returncode={manifest['returncode']} manifest={Path(manifest['output_dir']) / 'deeph_train_manifest.json'}"
    )


if __name__ == "__main__":
    main()
