"""Minimal CLI for the ML vs SIESTA benchmark infrastructure.

Subcommands:
- ``generate-siesta-displacements``: write reference + ±h displacement FDFs.
- ``benchmark-dry-run``: validate the whole plan without running anything.
- ``mix-datasets``: build a small/large mixing manifest (+ optional configs).
- ``inspect-species``: run the species-support diagnostic.

Nothing here launches SIESTA or trains a model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .config import load_benchmark_config
from .dataset_mixing import (
    generate_mixed_dataset_configs,
    make_mixed_dataset_manifest,
)
from .fdf_io import generate_siesta_displacement_inputs
from .pipeline import benchmark_dry_run
from .species_transfer import inspect_species_support


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2))


def _cmd_generate(args: argparse.Namespace) -> int:
    config = load_benchmark_config(args.config)
    metadata = generate_siesta_displacement_inputs(
        config,
        args.output,
        base_fdf=args.base_fdf,
        dry_run=args.dry_run,
    )
    _print_json(metadata)
    return 0


def _cmd_dry_run(args: argparse.Namespace) -> int:
    config = load_benchmark_config(args.config)
    summary = benchmark_dry_run(config, siesta_output_dir=args.output)
    _print_json(summary)
    return 0 if summary["ok"] else 1


def _load_dataset(path: str | Path) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "samples" in data:
        data = data["samples"]
    if not isinstance(data, list):
        raise ValueError(f"Dataset {path} must be a list (or have a 'samples' list).")
    return data


def _cmd_mix(args: argparse.Namespace) -> int:
    small = _load_dataset(args.small)
    large = _load_dataset(args.large)
    ratios = [float(r) for r in args.ratios.split(",")] if args.ratios else None
    manifest = make_mixed_dataset_manifest(
        small,
        large,
        ratios=ratios if ratios else (0.0, 0.1, 0.25, 0.5, 0.75, 1.0),
        mode=args.mode,
        output_path=args.output,
        seed=args.seed,
    )
    if args.configs_dir:
        written = generate_mixed_dataset_configs(
            manifest, args.configs_dir, manifest_path=args.output
        )
        manifest["generated_configs"] = written
    _print_json(manifest)
    return 0


def _cmd_inspect_species(args: argparse.Namespace) -> int:
    new_species = args.new_species.split(",") if args.new_species else None
    report = inspect_species_support(args.config, new_species=new_species)
    _print_json(report.to_dict())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ml-vs-siesta",
        description="ML vs SIESTA benchmark infrastructure (no SIESTA, no training).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser(
        "generate-siesta-displacements",
        help="Write reference + ±h displacement FDFs and metadata.json.",
    )
    gen.add_argument("--config", required=True, help="Benchmark YAML/JSON config.")
    gen.add_argument("--output", required=True, help="Output directory.")
    gen.add_argument("--base-fdf", default=None, help="Override system.input_structure.")
    gen.add_argument("--dry-run", action="store_true", help="List files without writing.")
    gen.set_defaults(func=_cmd_generate)

    dry = sub.add_parser("benchmark-dry-run", help="Validate the full plan.")
    dry.add_argument("--config", required=True, help="Benchmark YAML/JSON config.")
    dry.add_argument("--output", default=None, help="Optional SIESTA output dir for paths.")
    dry.set_defaults(func=_cmd_dry_run)

    mix = sub.add_parser("mix-datasets", help="Build a small/large mixing manifest.")
    mix.add_argument("--small", required=True, help="Small dataset JSON.")
    mix.add_argument("--large", required=True, help="Large dataset JSON.")
    mix.add_argument("--mode", choices=["add", "replace"], default="add")
    mix.add_argument("--ratios", default=None, help="Comma-separated ratios.")
    mix.add_argument("--seed", type=int, default=0)
    mix.add_argument("--output", default=None, help="Manifest output path (json/yaml/csv).")
    mix.add_argument("--configs-dir", default=None, help="Also emit D<i>.yaml configs here.")
    mix.set_defaults(func=_cmd_mix)

    insp = sub.add_parser("inspect-species", help="Species-support diagnostic.")
    insp.add_argument("--config", required=True, help="Model/species config (YAML/JSON).")
    insp.add_argument("--new-species", default=None, help="Comma-separated new species.")
    insp.set_defaults(func=_cmd_inspect_species)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
