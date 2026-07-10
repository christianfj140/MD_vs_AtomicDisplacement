"""Fase 0: reproducible run inventory (repo SHAs, dirty state, import paths)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "shared"))

from run_inventory import (  # noqa: E402
    collect_run_inventory,
    git_repository_state,
    module_import_state,
    reproducibility_status,
)


def _init_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    (path / "a.txt").write_text("x", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "-A"], check=True)
    subprocess.run(
        ["git", "-C", str(path), "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init"],
        check=True,
    )


def test_clean_repo_is_pinned_clean(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    state = git_repository_state(repo)
    assert state["commit"] and len(state["commit"]) == 40
    assert state["dirty"] is False
    assert reproducibility_status({"r": state}) == "pinned_clean"


def test_dirty_repo_is_pinned_dirty(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    (repo / "b.txt").write_text("dirty", encoding="utf-8")
    state = git_repository_state(repo)
    assert state["dirty"] is True
    assert reproducibility_status({"r": state}) == "pinned_dirty"


def test_non_git_directory_is_unpinned(tmp_path):
    plain = tmp_path / "plain"
    plain.mkdir()
    state = git_repository_state(plain)
    assert state["error"] == "not_git_repo"
    assert state["commit"] is None
    assert reproducibility_status({"r": state}) == "unpinned"


def test_missing_path_is_unavailable(tmp_path):
    state = git_repository_state(tmp_path / "nope")
    assert state["error"] == "path_missing"
    assert reproducibility_status({"r": state}) == "unavailable"


def test_import_from_unexpected_path_is_flagged(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    inventory = collect_run_inventory({"MD_vs_AtomicDisplacement": REPO_ROOT, "graph2mat": repo})
    g2m = inventory["imports"]["graph2mat"]
    # graph2mat really imports from its own checkout, not from tmp repo.
    if g2m.get("module_path"):
        assert g2m["matches_inspected_repo"] is False
        assert inventory.get("warnings")


def test_module_import_state_reports_this_interpreter():
    state = module_import_state("json")
    assert state["module_path"].endswith("json/__init__.py")


def test_full_inventory_has_required_blocks():
    inventory = collect_run_inventory()
    assert inventory["schema"] == "run_inventory_v1"
    assert set(inventory["repositories"]) == {"MD_vs_AtomicDisplacement", "graph2mat", "DeepH-pack"}
    for state in inventory["repositories"].values():
        assert "commit" in state and "dirty" in state
    assert inventory["python"]["executable"]
    assert inventory["reproducibility_status"] in {
        "pinned_clean",
        "pinned_dirty",
        "unpinned",
        "unavailable",
    }
