"""Reproducible run inventory: which code (repo SHAs + imported checkouts) ran.

Single source of truth for the three-repo provenance block required by every
scientific manifest (training, derivatives, mixing, metrics, UI payloads).

``reproducibility_status`` semantics:

- ``pinned_clean``: every repository has a resolvable SHA and a clean tree.
- ``pinned_dirty``: SHAs resolve but at least one tree has local changes.
- ``unpinned``: at least one repository path exists but has no resolvable SHA.
- ``unavailable``: at least one repository could not be inspected at all.

Only ``pinned_clean`` may aspire to ``paper_ready``; everything else is
diagnostic and must surface a visible warning downstream.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPOSITORIES = {
    "MD_vs_AtomicDisplacement": REPO_ROOT,
    "graph2mat": REPO_ROOT.parent / "graph2mat",
    "DeepH-pack": REPO_ROOT.parent / "DeepH-pack",
}

RUN_INVENTORY_SCHEMA = "run_inventory_v1"


def _git_output(path: Path, args: list[str], timeout: float = 10.0) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(path), *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except Exception:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def git_repository_state(path: str | Path | None) -> dict[str, Any]:
    """Commit/branch/dirty for one repository path (never raises)."""
    if path in (None, ""):
        return {"path": None, "commit": None, "branch": None, "dirty": None, "error": "no_path"}
    candidate = Path(path).expanduser()
    if not candidate.exists():
        return {
            "path": str(candidate),
            "commit": None,
            "branch": None,
            "dirty": None,
            "error": "path_missing",
        }
    root_text = _git_output(candidate, ["rev-parse", "--show-toplevel"])
    if not root_text:
        return {
            "path": str(candidate),
            "commit": None,
            "branch": None,
            "dirty": None,
            "error": "not_git_repo",
        }
    root = Path(root_text)
    commit = _git_output(root, ["rev-parse", "HEAD"])
    dirty_text = _git_output(root, ["status", "--porcelain"], timeout=30.0)
    return {
        "path": str(root),
        "commit": commit,
        "branch": _git_output(root, ["rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(dirty_text) if dirty_text is not None else None,
    }


def module_import_state(module_name: str, python_executable: str | Path | None = None) -> dict[str, Any]:
    """Where ``module_name`` actually imports from (this or another python)."""
    if python_executable in (None, "", str(sys.executable)):
        import importlib

        try:
            module = importlib.import_module(module_name)
            module_path = str(Path(getattr(module, "__file__", "") or "").resolve())
            return {"module": module_name, "module_path": module_path or None}
        except Exception as exc:  # noqa: BLE001 - inventory must never crash the run
            return {"module": module_name, "module_path": None, "error": repr(exc)}
    script = (
        f"import importlib, pathlib; "
        f"m = importlib.import_module({module_name!r}); "
        f"print(pathlib.Path(m.__file__).resolve())"
    )
    try:
        completed = subprocess.run(
            [str(python_executable), "-c", script],
            check=False,
            capture_output=True,
            text=True,
            timeout=60.0,
        )
    except Exception as exc:  # noqa: BLE001
        return {"module": module_name, "module_path": None, "error": repr(exc)}
    if completed.returncode != 0:
        return {
            "module": module_name,
            "module_path": None,
            "error": completed.stderr.strip() or "import_failed",
        }
    return {"module": module_name, "module_path": completed.stdout.strip()}


def _import_matches_repo(import_state: dict[str, Any], repo_state: dict[str, Any]) -> bool | None:
    module_path = import_state.get("module_path")
    repo_path = repo_state.get("path")
    if not module_path or not repo_path:
        return None
    return str(module_path).startswith(str(repo_path).rstrip("/") + "/")


def reproducibility_status(repositories: dict[str, dict[str, Any]]) -> str:
    statuses = list(repositories.values())
    if not statuses:
        return "unavailable"
    if any(s.get("error") in ("path_missing", "no_path") for s in statuses):
        return "unavailable"
    if any(not s.get("commit") for s in statuses):
        return "unpinned"
    if any(s.get("dirty") for s in statuses):
        return "pinned_dirty"
    if any(s.get("dirty") is None for s in statuses):
        return "unpinned"
    return "pinned_clean"


def collect_run_inventory(
    repositories: dict[str, str | Path] | None = None,
    *,
    deeph_python: str | Path | None = None,
    graph2mat_python: str | Path | None = None,
) -> dict[str, Any]:
    """Full run inventory (repos + python + real import locations).

    ``deeph_python`` / ``graph2mat_python`` point at the interpreters that
    actually run each backend when they differ from ``sys.executable``.
    """
    repo_paths = {k: Path(v) for k, v in (repositories or DEFAULT_REPOSITORIES).items()}
    repo_states = {name: git_repository_state(path) for name, path in repo_paths.items()}

    try:
        import torch

        torch_version = torch.__version__
        default_dtype = str(torch.get_default_dtype()).replace("torch.", "")
    except Exception:  # noqa: BLE001 - torch-less callers still get an inventory
        torch_version = None
        default_dtype = None

    imports: dict[str, Any] = {}
    if "graph2mat" in repo_states:
        state = module_import_state("graph2mat", graph2mat_python)
        state["matches_inspected_repo"] = _import_matches_repo(state, repo_states["graph2mat"])
        imports["graph2mat"] = state
    if "DeepH-pack" in repo_states:
        state = module_import_state("deeph", deeph_python)
        state["matches_inspected_repo"] = _import_matches_repo(state, repo_states["DeepH-pack"])
        imports["deeph"] = state

    inventory = {
        "schema": RUN_INVENTORY_SCHEMA,
        "repositories": repo_states,
        "python": {
            "executable": sys.executable,
            "version": sys.version.split()[0],
            "torch_version": torch_version,
            "default_dtype": default_dtype,
        },
        "imports": imports,
        "reproducibility_status": reproducibility_status(repo_states),
    }
    mismatches = [
        name for name, state in imports.items() if state.get("matches_inspected_repo") is False
    ]
    if mismatches:
        inventory["warnings"] = [
            f"module '{name}' imports from outside the inspected repository "
            f"({imports[name].get('module_path')}); the inspected SHA does not "
            "describe the executed code"
            for name in mismatches
        ]
    return inventory


def main() -> int:
    print(json.dumps(collect_run_inventory(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
