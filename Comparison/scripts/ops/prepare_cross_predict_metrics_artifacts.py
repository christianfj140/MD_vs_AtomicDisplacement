#!/usr/bin/env python3
"""Stage only the checkpoint/model files needed by cross predict_metrics."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]


def repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def _tar_members(archive: Path) -> list[str]:
    proc = subprocess.run(
        ["tar", "--zstd", "-tf", str(archive)],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return proc.stdout.splitlines()


def _prepared(dest: Path, key: str) -> bool:
    if key == "graph2mat":
        return (dest / "checkpoint_manifest.json").exists() and any(dest.rglob("*.ckpt"))
    return (dest / "config.ini").exists() and (dest / "best_state_dict.pkl").exists()


def _extract_members(
    archive: Path,
    prefix: str,
    dest: Path,
    wanted: tuple[str, ...],
    members_by_archive: dict[Path, list[str]],
) -> None:
    if _prepared(dest, "graph2mat" if wanted[0] == "checkpoint_manifest.json" else "deeph"):
        return
    members_by_archive.setdefault(archive, _tar_members(archive))
    members = [
        member
        for member in members_by_archive[archive]
        if member.startswith(prefix.rstrip("/") + "/")
        and not member.endswith("/")
        and (
            member.endswith("/checkpoint_manifest.json")
            or "/checkpoints/" in member
            or member.endswith("/config.ini")
            or member.endswith("/best_state_dict.pkl")
            or member.endswith("/best_model.pt")
            or member.endswith("/state_dict.pkl")
            or member.endswith("/model.pt")
        )
    ]
    if not members:
        raise RuntimeError(f"No matching artifact members under {prefix} in {archive}")
    tmp = dest.with_suffix(dest.suffix + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)
    strip = len(Path(prefix).parts)
    subprocess.run(
        ["tar", "--zstd", "-xf", str(archive), "-C", str(tmp), f"--strip-components={strip}", *members],
        cwd=REPO_ROOT,
        check=True,
    )
    shutil.rmtree(dest, ignore_errors=True)
    tmp.rename(dest)


def prepare(payload_path: Path) -> dict[str, Any]:
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    prepared: list[dict[str, str]] = []
    members_by_archive: dict[Path, list[str]] = {}
    for source_id, artifacts in (payload.get("existing_artifacts") or {}).items():
        for key in ("graph2mat", "deeph"):
            archive_key = f"{key}_archive"
            prefix_key = f"{key}_archive_prefix"
            dest_key = "graph2mat_training_dir" if key == "graph2mat" else "deeph_save_dir"
            archive_value = artifacts.get(archive_key)
            if not archive_value:
                continue
            archive = repo_path(archive_value)
            dest = repo_path(artifacts[dest_key])
            wanted = (
                ("checkpoint_manifest.json", "lightning_logs")
                if key == "graph2mat"
                else ("config.ini", "best_state_dict.pkl")
            )
            _extract_members(archive, str(artifacts[prefix_key]), dest, wanted, members_by_archive)
            prepared.append({"source_id": str(source_id), "model": key, "dest": str(dest)})
    return {"prepared": prepared, "count": len(prepared)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("payload", type=Path)
    parser.add_argument("--result-json", type=Path)
    args = parser.parse_args()
    result = prepare(repo_path(args.payload))
    text = json.dumps(result, indent=2) + "\n"
    if args.result_json:
        out = repo_path(args.result_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
