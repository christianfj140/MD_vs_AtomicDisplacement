#!/usr/bin/env python3
"""Write the final three-repository and numerical-environment inventory."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
SHARED_DIR = REPO_ROOT / "shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

from run_inventory import collect_run_inventory  # noqa: E402

DEFAULT_OUTPUT = (
    REPO_ROOT
    / "Comparison"
    / "results"
    / "audit_remediation_20260728"
    / "reproducibility_inventory.json"
)
DEEPH_PYTHON = REPO_ROOT.parent / "DeepH-pack" / ".venv" / "bin" / "python"
ENV_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "SIESTA_COMMAND",
)
PACKAGE_NAMES = ("numpy", "scipy", "torch", "sisl", "graph2mat", "netCDF4", "h5py")
PIN_NAMES = (
    "pyproject.toml",
    "requirements.txt",
    "requirements-dev.txt",
    "environment.yml",
    "environment.yaml",
    "uv.lock",
    "poetry.lock",
)


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(command: list[str]) -> dict[str, Any]:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
    except Exception as exc:  # noqa: BLE001 - inventory is fail-closed, not fatal
        return {"command": command, "returncode": None, "error": repr(exc)}
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def executable_record(name: str, path: str | Path | None, version_args: list[str]) -> dict[str, Any]:
    requested = Path(path).absolute() if path else None
    resolved = requested.resolve() if requested else None
    return {
        "name": name,
        "path": str(requested) if requested else None,
        "resolved_path": str(resolved) if resolved else None,
        "sha256": sha256(resolved) if resolved else None,
        "version_probe": command_output([str(requested), *version_args]) if requested else None,
    }


def dependency_pins() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root in (REPO_ROOT, REPO_ROOT.parent / "graph2mat", REPO_ROOT.parent / "DeepH-pack"):
        for name in PIN_NAMES:
            path = root / name
            if path.is_file():
                rows.append({"path": str(path), "sha256": sha256(path)})
    return rows


def netcdf_numpy_probe() -> dict[str, Any]:
    script = """
import tempfile, warnings
import numpy as np
warnings.simplefilter("always")
import netCDF4
with tempfile.NamedTemporaryFile(suffix=".nc") as handle:
    with netCDF4.Dataset(handle.name, "w") as dataset:
        dataset.createDimension("n", 3)
        dataset.createVariable("x", "f8", ("n",))[:] = [1.0, 2.0, 3.0]
    with netCDF4.Dataset(handle.name) as dataset:
        values = np.asarray(dataset.variables["x"][:])
assert values.tolist() == [1.0, 2.0, 3.0]
"""
    probe = command_output([sys.executable, "-c", script])
    result: dict[str, Any] = {
        "roundtrip_ok": probe.get("returncode") == 0,
        "warnings": [
            line
            for line in str(probe.get("stderr") or "").splitlines()
            if "Warning:" in line
        ],
        "probe": probe,
    }
    result["classification"] = (
        "warning_reproduced_but_minimal_numeric_roundtrip_passed"
        if result["roundtrip_ok"] and result["warnings"]
        else "no_runtime_incompatibility_observed"
        if result["roundtrip_ok"]
        else "runtime_incompatibility_or_probe_failure"
    )
    return result


def build_inventory() -> dict[str, Any]:
    siesta = shutil.which("siesta")
    nvidia_smi = shutil.which("nvidia-smi")
    package_versions = {}
    for name in PACKAGE_NAMES:
        try:
            package_versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            package_versions[name] = None
    inventory = collect_run_inventory(
        deeph_python=DEEPH_PYTHON if DEEPH_PYTHON.exists() else None,
    )
    inventory.update(
        {
            "schema": "scientific_reproducibility_inventory_v1",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "package_versions": package_versions,
            "interpreters": {
                "pipeline": executable_record("pipeline_python", sys.executable, ["--version"]),
                "deeph": executable_record(
                    "deeph_python",
                    DEEPH_PYTHON if DEEPH_PYTHON.exists() else None,
                    ["--version"],
                ),
            },
            "executables": {
                "siesta": executable_record("siesta", siesta, ["--version"]),
                "nvidia_smi": executable_record("nvidia-smi", nvidia_smi, ["--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]),
            },
            "siesta_linkage": command_output(["ldd", siesta]) if siesta else None,
            "cuda": {
                "torch_cuda_version": getattr(__import__("torch").version, "cuda", None),
                "torch_cuda_available": bool(__import__("torch").cuda.is_available()),
            },
            "selected_environment": {key: os.environ.get(key) for key in ENV_KEYS},
            "dependency_pins": dependency_pins(),
            "pip_check": command_output([sys.executable, "-m", "pip", "check"]),
            "numpy_netcdf4_abi_probe": netcdf_numpy_probe(),
            "paper_ready_allowed": inventory.get("reproducibility_status") == "pinned_clean",
            "paper_level_blockers": (
                []
                if inventory.get("reproducibility_status") == "pinned_clean"
                else [f"reproducibility_status={inventory.get('reproducibility_status')}"]
            ),
            "clean_freeze_instructions": [
                "Review and intentionally commit the desired changes in each repository.",
                "Re-run this inventory and require reproducibility_status=pinned_clean.",
                "Record the resulting three commit SHAs in every final campaign manifest.",
                "Do not alter historical scientific artifacts; rerun final campaigns into new directories.",
            ],
        }
    )
    return inventory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    payload = build_inventory()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "reproducibility_status": payload["reproducibility_status"]}))
    return 0 if payload["numpy_netcdf4_abi_probe"]["roundtrip_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
