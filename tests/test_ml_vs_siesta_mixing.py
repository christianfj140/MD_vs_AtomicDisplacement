"""CPU-only tests for the small/large dataset mixing sweep.

Uses tiny fake datasets (a handful of snapshots with fake SIESTA artifacts that
share a fake carbon basis). No SIESTA, no training, no GPU, no large data.
"""

from __future__ import annotations

import hashlib
import json
import sys
import threading
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "Comparison" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import ml_vs_siesta as mvs  # noqa: E402

_REQUIRED_SUFFIXES = (".HSX", ".STRUCT_OUT", ".XV", ".ORB_INDX", ".TSHS", ".TSDE")


def _make_snapshot(dirpath: Path, *, label: str, n_atoms: int, tag: str) -> None:
    dirpath.mkdir(parents=True, exist_ok=True)
    (dirpath / "RUN.fdf").write_text(f"SystemLabel {label}\nNumberOfAtoms {n_atoms}\n", encoding="utf-8")
    (dirpath / "metadata.json").write_text(
        json.dumps({"system_label": label, "n_atoms": n_atoms}), encoding="utf-8"
    )
    (dirpath / "RUN.out").write_text(f"fake run out {tag}\n", encoding="utf-8")
    for suffix in _REQUIRED_SUFFIXES:
        (dirpath / f"{label}{suffix}").write_text(f"fake {suffix} {tag}", encoding="utf-8")


def _make_dataset(
    root: Path,
    *,
    n_snapshots: int,
    n_atoms: int,
    label: str = "graphene",
    basis_content: str = "fake basis",
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "RUN.fdf").write_text(f"SystemLabel {label}\n", encoding="utf-8")
    (root / "RUN.out").write_text("fake run out\n", encoding="utf-8")
    (root / "C.psf").write_text("fake psf", encoding="utf-8")
    basis = root / "material_basis"
    basis.mkdir(exist_ok=True)
    (basis / "C.ion.xml").write_text(basis_content, encoding="utf-8")
    basis_hash = hashlib.sha256(basis_content.encode()).hexdigest()
    prov = {
        "label": label,
        "species": [
            {"index": 1, "atomic_number": 6, "label": "C"},
            {"index": 2, "atomic_number": -1, "label": "Ghost-H"},
        ],
        "basis_file_sha256": {"C.ion.xml": basis_hash},
        "pseudopotential_sha256": {"C": hashlib.sha256(b"fake psf").hexdigest()},
        "fdf_sha256": hashlib.sha256(b"x").hexdigest(),
    }
    (root / "material_provenance.json").write_text(json.dumps(prov), encoding="utf-8")
    rows = []
    md = root / "MD_steps"
    for i in range(n_snapshots):
        snap = md / str(i)
        _make_snapshot(snap, label=label, n_atoms=n_atoms, tag=f"{label}{i}")
        rows.append(
            {
                "sample_id": f"md_{i}",
                "sample_dir": str(snap.resolve()),
                "split": "train",
                "system_label": label,
                "valid": True,
            }
        )
    frozen = {
        "schema": "joint_graph2mat_deeph_frozen_split_manifest_v1",
        "valid": True,
        "rows": rows,
        "split_counts": {"train": n_snapshots, "validation": 0, "test": 0},
    }
    (root / "frozen_split_manifest.json").write_text(json.dumps(frozen), encoding="utf-8")
    return root


@pytest.fixture
def small_large(tmp_path):
    small = _make_dataset(tmp_path / "small", n_snapshots=8, n_atoms=2)
    large = _make_dataset(tmp_path / "large", n_snapshots=8, n_atoms=50)
    return small, large


# --------------------------------------------------------------------------- #
# materialize
# --------------------------------------------------------------------------- #
def test_materialize_produces_runner_ready_dataset(small_large, tmp_path):
    small, large = small_large
    small_ids = [s.sample_id for s in mvs.read_dataset_samples(small)]
    large_ids = [s.sample_id for s in mvs.read_dataset_samples(large)]
    out = tmp_path / "merged"
    summary = mvs.materialize_mixed_dataset(
        small, large, selected_small_ids=small_ids, selected_large_ids=large_ids[:4],
        output_root=out, seed=1,
    )
    assert summary["total"] == 12
    assert not (out / "RUN.fdf").exists()
    frozen = json.loads((out / "frozen_split_manifest.json").read_text())
    assert frozen["valid"] is True
    assert all(frozen["split_counts"][s] > 0 for s in ("train", "validation", "test"))
    bench = json.loads((out / "benchmark_dataset_manifest.json").read_text())
    assert bench["benchmark_ready"] is True
    for split in ("train", "validation", "test"):
        assert list((out / "splits" / split).glob("*/RUN.fdf"))


def test_runner_validates_mixed_dataset_splits_from_frozen_manifest(small_large, tmp_path):
    import pipeline_ui as ui

    small, large = small_large
    small_ids = [s.sample_id for s in mvs.read_dataset_samples(small)]
    large_ids = [s.sample_id for s in mvs.read_dataset_samples(large)]
    out = tmp_path / "merged"
    mvs.materialize_mixed_dataset(
        small, large, selected_small_ids=small_ids, selected_large_ids=large_ids[:4],
        output_root=out, seed=1,
    )
    payload = ui.Graph2MatDeepHBenchmarkRunner().validate_dataset_payload(
        {"dataset_root": str(out), "strict_dataset_validation": False}
    )
    assert payload["artifact_summary"]["total_snapshots"] == 12
    assert payload["artifact_summary"]["invalid_snapshots"] == 0


def test_materialize_split_proportions_roughly_match(small_large, tmp_path):
    small, large = small_large
    small_ids = [s.sample_id for s in mvs.read_dataset_samples(small)]
    large_ids = [s.sample_id for s in mvs.read_dataset_samples(large)]
    summary = mvs.materialize_mixed_dataset(
        small, large, selected_small_ids=small_ids, selected_large_ids=large_ids,
        output_root=tmp_path / "merged", seed=0,
    )
    counts = summary["split_counts"]
    assert counts["train"] + counts["validation"] + counts["test"] == 16
    assert counts["train"] >= counts["validation"]
    assert counts["train"] >= counts["test"]


def test_materialize_writes_self_contained_provenance(small_large, tmp_path):
    small, large = small_large
    small_ids = [s.sample_id for s in mvs.read_dataset_samples(small)]
    large_ids = [s.sample_id for s in mvs.read_dataset_samples(large)]
    out = tmp_path / "merged"
    mvs.materialize_mixed_dataset(
        small, large, selected_small_ids=small_ids, selected_large_ids=large_ids[:4],
        output_root=out, seed=1, mode="add", ratio=0.5,
    )
    prov = json.loads((out / "mixed_dataset_provenance.json").read_text())
    assert prov["schema"] == "ml_vs_siesta_mixed_dataset_provenance_v1"
    assert prov["mode"] == "add"
    assert prov["ratio"] == 0.5
    assert prov["ratio_semantics"] == "fraction_of_large_pool"
    assert prov["seed"] == 1
    assert prov["small_root"] == str(small)
    assert prov["large_root"] == str(large)
    assert prov["split_policy"] == "resplit_combined"
    assert prov["split_fractions"] == [0.7, 0.15, 0.15]
    assert prov["compatibility"]["compatible"] is True
    # The recorded selection reconstructs exactly what was requested.
    assert prov["selected_small_ids"] == small_ids
    assert prov["selected_large_ids"] == large_ids[:4]
    # mode/ratio are optional: omitted -> null, no breakage.
    out2 = tmp_path / "merged2"
    mvs.materialize_mixed_dataset(
        small, large, selected_small_ids=small_ids, selected_large_ids=[],
        output_root=out2, seed=1,
    )
    prov2 = json.loads((out2 / "mixed_dataset_provenance.json").read_text())
    assert prov2["mode"] is None and prov2["ratio"] is None


def _test_split_ids(dataset_root: Path) -> set[str]:
    frozen = json.loads((dataset_root / "frozen_split_manifest.json").read_text())
    return {row["sample_id"] for row in frozen["rows"] if row["split"] == "test"}


def test_fixed_common_test_split_is_ratio_independent(small_large, tmp_path):
    small, large = small_large
    small_ids = [s.sample_id for s in mvs.read_dataset_samples(small)]
    large_ids = [s.sample_id for s in mvs.read_dataset_samples(large)]
    test_sets = []
    for name, n_large in (("r_low", 2), ("r_high", 6)):
        out = tmp_path / name
        mvs.materialize_mixed_dataset(
            small, large, selected_small_ids=small_ids,
            selected_large_ids=large_ids[:n_large],
            output_root=out, seed=3, split_policy="fixed_common_test",
        )
        test_sets.append(_test_split_ids(out))
    assert test_sets[0] == test_sets[1]
    assert test_sets[0]  # non-empty
    assert all(sid.startswith("small__") for sid in test_sets[0])


def test_fixed_common_test_replace_preserves_same_test_set(small_large, tmp_path):
    small, large = small_large
    summary = mvs.run_mixing_sweep(
        {8: small}, {8: large}, tmp_path / "out",
        modes=("replace",), ratios=(0.25, 0.75), seed=3,
        split_policy="fixed_common_test", dry_run=False,
    )
    assert summary["n_permutations"] == 2
    test_sets = [
        _test_split_ids(tmp_path / "out" / "size8_replace_r0p250"),
        _test_split_ids(tmp_path / "out" / "size8_replace_r0p750"),
    ]
    assert test_sets[0] == test_sets[1]
    assert test_sets[0]
    assert all(sid.startswith("small__") for sid in test_sets[0])


def test_fixed_common_test_differs_from_resplit_and_default_unchanged(small_large, tmp_path):
    small, large = small_large
    small_ids = [s.sample_id for s in mvs.read_dataset_samples(small)]
    large_ids = [s.sample_id for s in mvs.read_dataset_samples(large)]
    # Default policy regression: same counts + all splits populated as before.
    summary = mvs.materialize_mixed_dataset(
        small, large, selected_small_ids=small_ids, selected_large_ids=large_ids[:4],
        output_root=tmp_path / "default", seed=1,
    )
    assert sum(summary["split_counts"].values()) == 12
    assert all(summary["split_counts"][s] > 0 for s in ("train", "validation", "test"))
    # Unknown policy fails loudly.
    with pytest.raises(mvs.DatasetMaterializeError):
        mvs.materialize_mixed_dataset(
            small, large, selected_small_ids=small_ids, selected_large_ids=[],
            output_root=tmp_path / "bad", seed=1, split_policy="nonsense",
        )


def test_materialize_refuses_existing_output_without_overwrite(small_large, tmp_path):
    small, large = small_large
    small_ids = [s.sample_id for s in mvs.read_dataset_samples(small)]
    out = tmp_path / "merged"
    kwargs = dict(
        selected_small_ids=small_ids, selected_large_ids=[], output_root=out, seed=0,
    )
    mvs.materialize_mixed_dataset(small, large, **kwargs)
    with pytest.raises(mvs.DatasetMaterializeError, match="already exists"):
        mvs.materialize_mixed_dataset(small, large, **kwargs)
    # overwrite=True keeps the run_mixing_sweep contract working.
    mvs.materialize_mixed_dataset(small, large, overwrite=True, **kwargs)


def test_materialize_rejects_output_outside_safe_roots(small_large):
    small, large = small_large
    small_ids = [s.sample_id for s in mvs.read_dataset_samples(small)]
    for dangerous in (
        Path.home() / "some_merged_dataset",
        Path("/"),
        Path(__file__).resolve().parents[1],  # repo root
        Path(__file__).resolve().parents[1] / "Comparison" / "results",
    ):
        with pytest.raises(mvs.DatasetMaterializeError, match="safe working roots"):
            mvs.materialize_mixed_dataset(
                small, large, selected_small_ids=small_ids, selected_large_ids=[],
                output_root=dangerous, seed=0, overwrite=True,
            )


def test_incompatible_basis_raises(tmp_path):
    small = _make_dataset(tmp_path / "small", n_snapshots=4, n_atoms=2, basis_content="basis A")
    large = _make_dataset(tmp_path / "large", n_snapshots=4, n_atoms=50, basis_content="basis B")
    with pytest.raises(mvs.DatasetCompatibilityError):
        mvs.validate_datasets_compatible(small, large)
    small_ids = [s.sample_id for s in mvs.read_dataset_samples(small)]
    with pytest.raises(mvs.DatasetCompatibilityError):
        mvs.materialize_mixed_dataset(
            small, large, selected_small_ids=small_ids, selected_large_ids=[],
            output_root=tmp_path / "merged", seed=0,
        )


def test_incompatible_real_species_raises(tmp_path):
    # A genuine *real* species mismatch (adds N) must still raise.
    small = _make_dataset(tmp_path / "small", n_snapshots=4, n_atoms=2)
    large = _make_dataset(tmp_path / "large", n_snapshots=4, n_atoms=50)
    prov = json.loads((large / "material_provenance.json").read_text())
    prov["species"] = [
        {"index": 1, "atomic_number": 6, "label": "C"},
        {"index": 2, "atomic_number": 7, "label": "N"},
    ]
    (large / "material_provenance.json").write_text(json.dumps(prov), encoding="utf-8")
    with pytest.raises(mvs.DatasetCompatibilityError):
        mvs.validate_datasets_compatible(small, large)


def test_ghost_species_do_not_block_compatibility(tmp_path):
    # 2-atom cell has [C, Ghost-H]; 5x5x1 supercell often has only [C].
    small = _make_dataset(tmp_path / "small", n_snapshots=4, n_atoms=2)
    large = _make_dataset(tmp_path / "large", n_snapshots=4, n_atoms=50)
    prov = json.loads((large / "material_provenance.json").read_text())
    prov["species"] = [{"index": 1, "atomic_number": 6, "label": "C"}]
    prov["basis_file_sha256"] = {"C.ion.xml": small_basis_hash(small)}
    (large / "material_provenance.json").write_text(json.dumps(prov), encoding="utf-8")
    result = mvs.validate_datasets_compatible(small, large)
    assert result["compatible"] is True
    assert result["species"] == ["C"]
    assert result["ghost_species_ignored"] == ["Ghost-H"]


def small_basis_hash(dataset_root: Path) -> str:
    prov = json.loads((dataset_root / "material_provenance.json").read_text())
    return prov["basis_file_sha256"]["C.ion.xml"]


def test_dataset_atom_count(small_large):
    small, large = small_large
    assert mvs.dataset_atom_count(small) == 2
    assert mvs.dataset_atom_count(large) == 50


def test_dataset_atom_count_falls_back_to_fdf(tmp_path):
    # Real datasets omit n_atoms from metadata.json; NumberOfAtoms in RUN.fdf is
    # the reliable fallback used for small/large classification.
    dataset = _make_dataset(tmp_path / "ds", n_snapshots=3, n_atoms=50)
    for meta in dataset.glob("MD_steps/*/metadata.json"):
        payload = json.loads(meta.read_text())
        payload.pop("n_atoms", None)
        meta.write_text(json.dumps(payload), encoding="utf-8")
    assert mvs.dataset_atom_count(dataset) == 50


# --------------------------------------------------------------------------- #
# manifest selection (replace mode)
# --------------------------------------------------------------------------- #
def _replace_manifest(seed: int):
    small = [{"id": f"s{i:02d}"} for i in range(20)]
    large = [{"id": f"l{i:02d}"} for i in range(10)]
    return mvs.make_mixed_dataset_manifest(
        small, large, ratios=(0.5,), mode="replace", seed=seed
    )


def test_replace_kept_small_is_deterministic():
    a = _replace_manifest(seed=3)["partitions"][0]["selected_ids"]
    b = _replace_manifest(seed=3)["partitions"][0]["selected_ids"]
    assert a == b


def test_replace_kept_small_is_sampled_not_prefix():
    part = _replace_manifest(seed=3)["partitions"][0]
    kept_small = [sid for sid in part["selected_ids"] if sid.startswith("s")]
    n_keep = 20 - part["n_large_selected"]
    assert len(kept_small) == n_keep
    prefix = [f"s{i:02d}" for i in range(n_keep)]
    assert sorted(kept_small) != prefix


def test_ratio_semantics_and_large_capped_exposed():
    small = [{"id": f"s{i}"} for i in range(10)]
    large = [{"id": f"l{i}"} for i in range(4)]  # fewer large than small
    for mode in ("add", "replace"):
        manifest = mvs.make_mixed_dataset_manifest(
            small, large, ratios=(0.0, 1.0), mode=mode, seed=0
        )
        for part in manifest["partitions"]:
            assert part["ratio_semantics"] == "fraction_of_large_pool"
            assert "large_capped" in part
    # replace ratio=1.0 wants 10 large but only 4 exist -> capped, and flagged.
    replace = mvs.make_mixed_dataset_manifest(
        small, large, ratios=(1.0,), mode="replace", seed=0
    )["partitions"][0]
    assert replace["large_capped"] is True
    assert replace["n_large_selected"] == 4
    # plan propagates both fields.
    plan = mvs.plan_mixing_sweep({8: 10}, {8: 4}, modes=("replace",), ratios=(1.0,), seed=0)
    perm = plan["permutations"][0]
    assert perm["ratio_semantics"] == "fraction_of_large_pool"
    assert perm["large_capped"] is True


# --------------------------------------------------------------------------- #
# plan
# --------------------------------------------------------------------------- #
def test_plan_add_grows_replace_constant():
    plan = mvs.plan_mixing_sweep(
        {20: 20, 40: 40}, {20: 20, 40: 40},
        modes=("add", "replace"), ratios=(0.0, 0.5, 1.0), seed=0,
    )
    assert plan["n_permutations"] == 2 * 2 * 3
    by_key = {(p["size"], p["mode"], p["ratio"]): p for p in plan["permutations"]}
    assert by_key[(20, "add", 0.0)]["total_size"] == 20
    assert by_key[(20, "add", 1.0)]["total_size"] == 40
    assert by_key[(20, "replace", 0.0)]["total_size"] == 20
    assert by_key[(20, "replace", 1.0)]["total_size"] == 20
    assert by_key[(20, "replace", 1.0)]["n_small_selected"] == 0


def test_plan_skips_missing_size():
    plan = mvs.plan_mixing_sweep({20: 20}, {40: 40}, sizes=[20, 40])
    assert plan["n_permutations"] == 0
    assert len(plan["warnings"]) == 2


def test_plan_reproducible():
    a = mvs.plan_mixing_sweep({30: 30}, {30: 60}, modes=("add",), ratios=(0.5,), seed=7)
    b = mvs.plan_mixing_sweep({30: 30}, {30: 60}, modes=("add",), ratios=(0.5,), seed=7)
    assert a["permutations"] == b["permutations"]


# --------------------------------------------------------------------------- #
# run_mixing_sweep (with fake launch_fn)
# --------------------------------------------------------------------------- #
def test_run_mixing_sweep_dry_run(small_large, tmp_path):
    small, large = small_large
    summary = mvs.run_mixing_sweep(
        {8: small}, {8: large}, tmp_path / "out",
        modes=("add",), ratios=(0.0, 1.0), seed=0, dry_run=True,
    )
    assert summary["dry_run"] is True
    assert all(p["status"] == "planned" for p in summary["permutations"])
    assert not summary["records"]


def test_run_mixing_sweep_materialize_and_train(small_large, tmp_path):
    small, large = small_large

    def fake_launch(payload):
        assert "system_label" not in payload
        assert payload["epochs"] == 10
        assert payload["graph2mat_overrides"]["max_epochs"] == 10
        assert payload["deeph"]["epochs"] == 10
        assert payload["deeph"]["num_threads"] == 4
        frozen = json.loads((Path(payload["dataset_root"]) / "frozen_split_manifest.json").read_text())
        total = sum(frozen["split_counts"].values())
        return {"metrics": {"graph2mat": {"h_mae_eV": 1.0 / total},
                            "deeph": {"h_mae_eV": 2.0 / total}}}

    summary = mvs.run_mixing_sweep(
        {8: small}, {8: large}, tmp_path / "out",
        modes=("add",), ratios=(0.0, 1.0), seed=0,
        models=("graph2mat", "deeph"),
        epochs=10,
        performance={"torch_num_threads": 4},
        dry_run=False,
        launch_fn=fake_launch,
    )
    assert summary["dry_run"] is False
    assert len(summary["records"]) == 2 * 2  # 2 permutations x 2 models
    assert summary["n_trained"] == 2
    assert summary["n_failed"] == 0
    assert all(p["status"] == "trained" for p in summary["permutations"])
    assert (tmp_path / "out" / "mixing_sweep_summary.json").is_file()


def test_run_mixing_sweep_marks_failed_when_launch_fails(small_large, tmp_path):
    small, large = small_large

    def failing_launch(payload):
        # Runner ran but produced no MAE / signalled failure.
        return {"ok": False, "error": "runner returncode 1", "metrics": {}}

    summary = mvs.run_mixing_sweep(
        {8: small}, {8: large}, tmp_path / "out",
        modes=("add",), ratios=(0.0, 1.0), seed=0,
        models=("graph2mat", "deeph"), dry_run=False, launch_fn=failing_launch,
    )
    assert summary["n_failed"] == 2
    assert summary["n_trained"] == 0
    assert not summary["records"]
    assert all(p["status"] == "failed" and p.get("error") for p in summary["permutations"])


def test_run_mixing_sweep_marks_partial_when_one_model_missing(small_large, tmp_path):
    small, large = small_large

    def partial_launch(payload):
        # Only graph2mat produced MAE; deeph is missing -> partial.
        return {"ok": True, "metrics": {"graph2mat": {"h_mae_eV": 0.05}}}

    summary = mvs.run_mixing_sweep(
        {8: small}, {8: large}, tmp_path / "out",
        modes=("add",), ratios=(1.0,), seed=0,
        models=("graph2mat", "deeph"), dry_run=False, launch_fn=partial_launch,
    )
    assert summary["n_partial"] == 1
    assert summary["permutations"][0]["status"] == "partial"
    assert "deeph" in summary["permutations"][0]["error"]


# --------------------------------------------------------------------------- #
# aggregation / plot
# --------------------------------------------------------------------------- #
def test_aggregate_mae_vs_size():
    records = [
        {"size": 20, "mode": "add", "ratio": 0.0, "total_size": 20, "model": "graph2mat", "h_mae_eV": 0.05},
        {"size": 40, "mode": "add", "ratio": 0.0, "total_size": 40, "model": "graph2mat", "h_mae_eV": 0.03},
        {"size": 20, "mode": "add", "ratio": 0.0, "total_size": 20, "model": "deeph", "h_mae_eV": 0.09},
    ]
    agg = mvs.aggregate_mae_vs_size(records)
    assert agg["n_curves"] == 2
    assert len(agg["payloads"]) == 2
    assert all(point.get("payload_id") for curve in agg["curves"] for point in curve["points"])
    g2m = next(c for c in agg["curves"] if c["model"] == "graph2mat")
    assert [p["total_size"] for p in g2m["points"]] == [20, 40]


def test_write_mae_vs_size_outputs(tmp_path):
    records = [
        {"size": 20, "mode": "add", "ratio": 0.0, "total_size": 20, "model": "graph2mat", "h_mae_eV": 0.05},
        {"size": 40, "mode": "add", "ratio": 0.0, "total_size": 40, "model": "graph2mat", "h_mae_eV": 0.03},
    ]
    agg = mvs.write_mae_vs_size_outputs(records, tmp_path / "plots")
    assert (tmp_path / "plots" / "mae_vs_size.json").is_file()
    assert Path(agg["json_path"]).is_file()
    if agg.get("png_path"):
        assert Path(agg["png_path"]).is_file()


def test_build_mae_vs_size_from_sweep(small_large, tmp_path):
    small, large = small_large

    def fake_launch(payload):
        return {"metrics": {"graph2mat": {"h_mae_eV": 0.04}}}

    summary = mvs.run_mixing_sweep(
        {8: small}, {8: large}, tmp_path / "out",
        modes=("add",), ratios=(1.0,), seed=0,
        models=("graph2mat",), dry_run=False, launch_fn=fake_launch,
    )
    agg = mvs.build_mae_vs_size_from_sweep(summary)
    assert agg["n_curves"] == 1


# --------------------------------------------------------------------------- #
# backend payloads + HTTP endpoints (pipeline_ui bridge)
# --------------------------------------------------------------------------- #
def test_backend_mixing_plan_and_demo(small_large):
    import pipeline_ui as ui

    small, large = small_large
    plan = ui.mixing_plan_payload(
        {
            "small": {"8": str(small)},
            "large": {"8": str(large)},
            "modes": ["add", "replace"],
            "ratios": [0.0, 1.0],
        }
    )
    assert plan["n_permutations"] == 4
    demo = ui.mixing_metrics_demo_payload()
    assert demo["n_curves"] > 0


def test_backend_mixing_discover(small_large, monkeypatch):
    import pipeline_ui as ui

    small, large = small_large
    # Point discovery at a scratch datasets root containing our two fakes.
    datasets_root = small.parent
    monkeypatch.setattr(ui, "DATASETS_ROOT", datasets_root)
    payload = ui.mixing_discover_payload(threshold_atoms=10)
    small_atoms = [d["n_atoms"] for d in payload["small"]]
    large_atoms = [d["n_atoms"] for d in payload["large"]]
    assert 2 in small_atoms  # 2-atom cell classified as small
    assert 50 in large_atoms  # 50-atom supercell classified as large


def test_backend_mixing_launch_dry_run_and_status(small_large, tmp_path, monkeypatch):
    import pipeline_ui as ui

    small, large = small_large
    # Isolate from any real mixing-sweep history on disk (e.g. a developer's
    # own past runs) so the persisted-summary fallback in metrics() only
    # ever sees this test's own payloads.
    monkeypatch.setattr(ui, "MIXING_SWEEP_OUTPUT_ROOT", tmp_path / "mixing_sweep_output")
    monkeypatch.setattr(ui, "RESULTS_ROOT", tmp_path / "results")
    runner = ui.MixingSweepRunner()  # fresh instance, does not touch the singleton
    runner.start(
        {
            "small": {"8": str(small)},
            "large": {"8": str(large)},
            "modes": ["add"],
            "ratios": [0.0, 1.0],
            "action": "preview",  # dry-run: no training, no writes
        }
    )
    for _ in range(100):
        status = runner.status()
        if status["state"] in ("completed", "error"):
            break
        time.sleep(0.05)
    status = runner.status()
    assert status["state"] == "completed", status
    assert status["n_permutations"] == 2
    assert not status["summary"]["records"]  # preview never trains
    metrics = runner.metrics()
    assert len(metrics["payloads"]) == 2
    assert {item["size"] for item in metrics["payloads"]} == {8}


def test_mixing_metrics_surfaces_historical_payload_without_records(tmp_path, monkeypatch):
    """A payload materialized in a previous session (e.g. size=50, no training
    yet) must still show up in the payload selector, just without metrics."""
    import pipeline_ui as ui

    output_root = tmp_path / "mixing_sweep_output"
    monkeypatch.setattr(ui, "MIXING_SWEEP_OUTPUT_ROOT", output_root)
    monkeypatch.setattr(ui, "RESULTS_ROOT", tmp_path / "results")
    output_root.mkdir(parents=True)
    (output_root / "mixing_sweep_summary.json").write_text(
        json.dumps(
            {
                "records": [],
                "permutations": [
                    {
                        "size": 50,
                        "mode": "add",
                        "ratio": 0.0,
                        "total_size": 48,
                        "status": "materialized",
                        "output_root": str(output_root / "size50_add_r0p000"),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    runner = ui.MixingSweepRunner()  # fresh, idle instance; never started
    metrics = runner.metrics()
    assert len(metrics["payloads"]) == 1
    payload = metrics["payloads"][0]
    assert payload["size"] == 50
    assert payload["status"] == "materialized"
    assert not any(curve["points"] for curve in metrics["curves"])


def test_mixing_metrics_payload_status_trained_when_metrics_exist(tmp_path, monkeypatch):
    """A payload with real MAE records should be marked as trained, while a
    sibling payload from the same historical summary with no records yet
    keeps its own (non-trained) status."""
    import pipeline_ui as ui

    output_root = tmp_path / "mixing_sweep_output"
    monkeypatch.setattr(ui, "MIXING_SWEEP_OUTPUT_ROOT", output_root)
    monkeypatch.setattr(ui, "RESULTS_ROOT", tmp_path / "results")
    output_root.mkdir(parents=True)
    (output_root / "mixing_sweep_summary.json").write_text(
        json.dumps(
            {
                "records": [
                    {
                        "size": 50,
                        "mode": "add",
                        "ratio": 0.0,
                        "total_size": 48,
                        "model": "graph2mat",
                        "h_mae_eV": 0.42,
                    }
                ],
                "permutations": [
                    {
                        "size": 50,
                        "mode": "add",
                        "ratio": 0.0,
                        "total_size": 48,
                        "status": "materialized",
                        "output_root": str(output_root / "size50_add_r0p000"),
                    },
                    {
                        "size": 80,
                        "mode": "add",
                        "ratio": 0.0,
                        "total_size": 78,
                        "status": "planned",
                        "output_root": str(output_root / "size80_add_r0p000"),
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    runner = ui.MixingSweepRunner()
    metrics = runner.metrics()
    by_size = {item["size"]: item for item in metrics["payloads"]}
    assert by_size[50]["status"] == "trained"
    assert by_size[80]["status"] == "planned"
    assert any(curve["points"] for curve in metrics["curves"])


class _FakeParallelBenchmarkRunner:
    """Stands in for Graph2MatDeepHBenchmarkRunner: finishes instantly, no training."""

    def start(self, payload):
        self.payload = payload

    def status(self):
        return {"running": False}

    def results(self):
        return {"status": {}}


def _write_common_metrics_csv(run_root: Path, model: str, h_mae: float) -> None:
    summary_dir = run_root / "common_metrics" / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    (summary_dir / "common_method_metrics.csv").write_text(
        f"method,h_mae_eV_mean\n{model},{h_mae}\n", encoding="utf-8"
    )


def test_parallel_sweep_marks_partial_when_one_model_missing(small_large, tmp_path, monkeypatch):
    import pipeline_ui as ui

    small, large = small_large
    monkeypatch.setattr(ui, "Graph2MatDeepHBenchmarkRunner", _FakeParallelBenchmarkRunner)
    out = tmp_path / "out"
    # Only graph2mat produces MAE for perm_0; deeph is missing -> partial.
    _write_common_metrics_csv(
        out / "parallel_run" / "sweep" / "graph2mat" / "perm_0", "graph2mat", 0.05
    )
    summary = ui._run_mixing_sweep_parallel(
        {8: str(small)}, {8: str(large)}, out,
        sizes=None, modes=("add",), ratios=(1.0,), seed=0,
        models=("graph2mat", "deeph"), epochs=None, performance=None, progress_fn=None,
    )
    assert summary["n_partial"] == 1
    assert summary["n_trained"] == 0
    assert summary["n_failed"] == 0
    perm = summary["permutations"][0]
    assert perm["status"] == "partial"
    assert "deeph" in perm["error"]


def test_parallel_sweep_marks_failed_without_any_mae(small_large, tmp_path, monkeypatch):
    import pipeline_ui as ui

    small, large = small_large
    monkeypatch.setattr(ui, "Graph2MatDeepHBenchmarkRunner", _FakeParallelBenchmarkRunner)
    summary = ui._run_mixing_sweep_parallel(
        {8: str(small)}, {8: str(large)}, tmp_path / "out",
        sizes=None, modes=("add",), ratios=(1.0,), seed=0,
        models=("graph2mat", "deeph"), epochs=None, performance=None, progress_fn=None,
    )
    assert summary["n_failed"] == 1
    assert summary["n_trained"] == 0
    perm = summary["permutations"][0]
    assert perm["status"] == "failed"
    assert perm["error"] == "no h_mae_eV produced"


def test_backend_mixing_launch_rejects_concurrent(small_large):
    import pipeline_ui as ui

    small, large = small_large
    runner = ui.MixingSweepRunner()
    body = {"small": {"8": str(small)}, "large": {"8": str(large)}, "action": "preview"}
    # Force a "running" state and confirm a second start is rejected.
    with runner._lock:
        runner._thread = threading.current_thread()
        runner._status = {"state": "running"}
    with pytest.raises(RuntimeError):
        runner.start(body)


def test_extract_model_h_mae_eV_shapes():
    import pipeline_ui as ui

    keyed = {"results": {"graph2mat": {"h_mae_eV": 0.03}, "deeph": {"h_mae_eV": 0.07}}}
    assert ui._extract_model_h_mae_eV(keyed, ("graph2mat", "deeph")) == {
        "graph2mat": {"h_mae_eV": 0.03},
        "deeph": {"h_mae_eV": 0.07},
    }
    listed = {"results": [{"method": "graph2mat", "h_mae_eV": 0.05}]}
    assert ui._extract_model_h_mae_eV(listed, ("graph2mat",)) == {"graph2mat": {"h_mae_eV": 0.05}}
    assert ui._extract_model_h_mae_eV({"results": None}, ("graph2mat",)) == {}


def test_mixing_metrics_from_common_csv(tmp_path):
    import pipeline_ui as ui

    summary = tmp_path / "run" / "common_metrics" / "summary"
    summary.mkdir(parents=True)
    (summary / "common_method_metrics.csv").write_text(
        "method,h_mae_eV_mean\n"
        "graph2mat,0.12\n"
        "deeph,0.34\n",
        encoding="utf-8",
    )
    assert ui._mixing_metrics_from_common_csv(tmp_path / "run", ("graph2mat", "deeph")) == {
        "graph2mat": {"h_mae_eV": 0.12},
        "deeph": {"h_mae_eV": 0.34},
    }


def test_mixing_metrics_from_run_metrics_csv(tmp_path):
    import pipeline_ui as ui

    metrics = tmp_path / "run" / "metrics" / "graph2mat" / "eval" / "metrics"
    metrics.mkdir(parents=True)
    (metrics / "kpoint_matrix_metrics.csv").write_text(
        "sample,row_type,h_mae_eV\n"
        "s0,per_k,0.10\n"
        "s0,per_k,0.20\n",
        encoding="utf-8",
    )

    assert ui._mixing_metrics_from_run_metrics(tmp_path / "run", "graph2mat") == {
        "graph2mat": {"h_mae_eV": pytest.approx(0.15)}
    }


def test_http_dispatch_mixing_routes(small_large):
    """Exercise the real do_GET/do_POST route wiring over a live server."""
    import http.client
    import json as _json
    from http.server import ThreadingHTTPServer

    import pipeline_ui as ui

    small, large = small_large
    original_datasets_root = ui.DATASETS_ROOT
    ui.DATASETS_ROOT = small.parent
    server = ThreadingHTTPServer(("127.0.0.1", 0), ui.ComparisonUIHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)

        conn.request("GET", "/api/mixing/discover?threshold_atoms=10")
        discover = _json.loads(conn.getresponse().read())
        assert [d["n_atoms"] for d in discover["small"]] == [2]
        assert [d["n_atoms"] for d in discover["large"]] == [50]

        conn.request("GET", "/api/mixing/metrics-demo")
        demo = _json.loads(conn.getresponse().read())
        assert demo["n_curves"] > 0

        body = _json.dumps(
            {"small": {"8": str(small)}, "large": {"8": str(large)}, "modes": ["add"], "ratios": [0.0, 1.0]}
        )
        conn.request("POST", "/api/mixing/plan", body, {"Content-Type": "application/json"})
        plan = _json.loads(conn.getresponse().read())
        assert plan["n_permutations"] == 2

        # Launch (preview action = dry-run, no training) + poll status over HTTP.
        launch_body = _json.dumps(
            {
                "small": {"8": str(small)},
                "large": {"8": str(large)},
                "modes": ["add"],
                "ratios": [0.0, 1.0],
                "action": "preview",
            }
        )
        conn.request("POST", "/api/mixing/launch", launch_body, {"Content-Type": "application/json"})
        launch_resp = conn.getresponse()
        assert launch_resp.status == 202
        assert _json.loads(launch_resp.read())["state"] == "started"

        final_status = None
        for _ in range(100):
            conn.request("GET", "/api/mixing/status")
            final_status = _json.loads(conn.getresponse().read())
            if final_status["state"] in ("completed", "error"):
                break
            time.sleep(0.05)
        assert final_status is not None and final_status["state"] == "completed", final_status
        assert final_status["n_permutations"] == 2
        conn.close()
    finally:
        ui.DATASETS_ROOT = original_datasets_root
        server.shutdown()
        server.server_close()


def test_mixing_run_ok_helper():
    import pipeline_ui as ui

    assert ui._mixing_run_ok({"returncode": 0}) is True
    assert ui._mixing_run_ok({"returncode": 2}) is True  # runner treats 2 as ok
    assert ui._mixing_run_ok({"returncode": 1}) is False
    assert ui._mixing_run_ok({"returncode": 0, "error": "boom"}) is False
    assert ui._mixing_run_ok({"returncode": 0, "stop_requested": True}) is False
    assert ui._mixing_run_ok(None) is False
