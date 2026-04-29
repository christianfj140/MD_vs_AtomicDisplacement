#!/usr/bin/env python3
"""Archive SIESTA FC output and optionally run VIBRA."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def copy_outputs(fc_run_dir: Path, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    patterns = ("*.FC", "*.fdf", "*.out", "*.xml", "*.xyz", "*.FA", "*.XV")
    for pattern in patterns:
        for src in sorted(fc_run_dir.glob(pattern)):
            dst = output_dir / src.name
            shutil.copy2(src, dst)
            copied.append(dst)
    return copied


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preserve .FC and run VIBRA when available.")
    parser.add_argument("--fc-run-dir", type=Path, default=REPO_ROOT / "AtomDisplacement" / "dataset")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results" / "comparison" / "phonons")
    parser.add_argument("--vibra-bin", default="vibra")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.fc_run_dir.exists():
        raise RuntimeError(f"No existe el directorio FC: {args.fc_run_dir}")
    fc_files = sorted(args.fc_run_dir.glob("*.FC"))
    if not fc_files:
        raise RuntimeError(f"No se encontro ningun .FC en {args.fc_run_dir}")

    vibra_path = shutil.which(args.vibra_bin)
    report = {
        "fc_run_dir": str(args.fc_run_dir),
        "output_dir": str(args.output_dir),
        "fc_files": [str(path) for path in fc_files],
        "vibra_bin": args.vibra_bin,
        "vibra_available": vibra_path is not None,
        "vibra_returncode": None,
        "copied_outputs": [],
    }

    if args.dry_run:
        print(json.dumps(report, indent=2))
        return 0

    copied = copy_outputs(args.fc_run_dir, args.output_dir)
    report["copied_outputs"] = [str(path) for path in copied]

    if vibra_path is not None:
        log_path = args.output_dir / "vibra.log"
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.run(
                [vibra_path],
                cwd=args.output_dir,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        report["vibra_returncode"] = process.returncode
        report["vibra_log"] = str(log_path)
    else:
        report["warning"] = "VIBRA no esta en PATH; se conservaron los outputs FC sin postproceso."

    report_path = args.output_dir / "phonon_manifest.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"[OK] Manifest fonones escrito en {report_path}")
    return 0 if report.get("vibra_returncode") in (None, 0) else int(report["vibra_returncode"])


if __name__ == "__main__":
    raise SystemExit(main())
