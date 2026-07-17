"""CPU-only tests for the small/large dataset mixing sweep.

Uses tiny fake datasets (a handful of snapshots with fake SIESTA artifacts that
share a fake carbon basis). No SIESTA, no training, no GPU, no large data.
"""

from __future__ import annotations

import csv
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
    assert prov["schema"] == "ml_vs_siesta_mixed_dataset_provenance_v2"
    # v2 additions (audit Fases 0/7/8): run inventory, scope and composition.
    assert prov["run_inventory"]["reproducibility_status"] in {
        "pinned_clean", "pinned_dirty", "unpinned", "unavailable"
    }
    assert prov["evaluation_scope"] == "small_only"
    composition = prov["composition"]
    assert composition["actual_train_size"] == prov["composition"]["n_small_train"] + composition["n_large_train"]
    assert composition["materialized_total_size"] == 12
    assert composition["test_large_size"] == 0
    assert prov["mode"] == "add"
    assert prov["ratio"] == 0.5
    assert prov["ratio_semantics"] == "fraction_of_large_pool_added"
    assert prov["seed"] == 1
    assert prov["small_root"] == str(small)
    assert prov["large_root"] == str(large)
    # Default split policy is the scientifically safe one (C1 audit fix).
    assert prov["split_policy"] == "fixed_common_test"
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


def test_ghost_species_mismatch_requires_explicit_exemption(tmp_path):
    """Audit I2: the ghost exemption is unverified in code -> explicit opt-in."""
    # 2-atom cell has [C, Ghost-H]; 5x5x1 supercell often has only [C].
    small = _make_dataset(tmp_path / "small", n_snapshots=4, n_atoms=2)
    large = _make_dataset(tmp_path / "large", n_snapshots=4, n_atoms=50)
    prov = json.loads((large / "material_provenance.json").read_text())
    prov["species"] = [{"index": 1, "atomic_number": 6, "label": "C"}]
    prov["basis_file_sha256"] = {"C.ion.xml": small_basis_hash(small)}
    (large / "material_provenance.json").write_text(json.dumps(prov), encoding="utf-8")

    # Without confirmation, differing ghost species refuse to mix.
    with pytest.raises(mvs.DatasetCompatibilityError, match="Ghost species differ"):
        mvs.validate_datasets_compatible(small, large)

    # With explicit confirmation the mix is allowed and the exemption recorded.
    result = mvs.validate_datasets_compatible(
        small, large, confirm_ghost_species_exemption=True
    )
    assert result["compatible"] is True
    assert result["species"] == ["C"]
    assert result["ghost_species_ignored"] == ["Ghost-H"]
    exemption = result["ghost_species_exemption"]
    assert exemption["required"] is True
    assert exemption["confirmed"] is True
    assert exemption["verified_in_code"] is False
    assert "UNVERIFIED" in exemption["note"]


def test_merged_provenance_is_honest_for_heterogeneous_pool(tmp_path):
    """Audit I3: the merged dataset must not masquerade as the small material."""
    small = _make_dataset(tmp_path / "small", n_snapshots=4, n_atoms=2)
    large = _make_dataset(tmp_path / "large", n_snapshots=4, n_atoms=50)
    prov = json.loads((large / "material_provenance.json").read_text())
    prov["species"] = [{"index": 1, "atomic_number": 6, "label": "C"}]
    prov["basis_file_sha256"] = {"C.ion.xml": small_basis_hash(small)}
    prov["label"] = "graphene_5x5"
    (large / "material_provenance.json").write_text(json.dumps(prov), encoding="utf-8")

    small_ids = [s.sample_id for s in mvs.read_dataset_samples(small)]
    large_ids = [s.sample_id for s in mvs.read_dataset_samples(large)]
    out = tmp_path / "merged"
    mvs.materialize_mixed_dataset(
        small, large, selected_small_ids=small_ids, selected_large_ids=large_ids[:2],
        output_root=out, seed=0, confirm_ghost_species_exemption=True,
    )
    merged = json.loads((out / "material_provenance.json").read_text())
    assert merged["material_source"] == "mixed_dataset"
    assert merged["heterogeneous_material_pool"] is True
    assert merged["provenance_source_of_truth"] == "mixed_dataset_provenance.json"
    assert merged["label"] == "mixed(graphene+graphene_5x5)"
    # Only the validated real species are asserted; ghosts are per-source.
    assert [s["label"] for s in merged["species"]] == ["C"]
    assert merged["ghost_species_by_source"] == {"small": ["Ghost-H"], "large": []}
    assert merged["fdf_sha256_semantics"] == "sha256_of_source_fdf_sha256_pair"
    assert any("UNVERIFIED" in w for w in merged["warnings"])


def test_merged_provenance_copies_verbatim_for_homogeneous_pool(small_large, tmp_path):
    small, large = small_large
    small_ids = [s.sample_id for s in mvs.read_dataset_samples(small)]
    out = tmp_path / "merged"
    mvs.materialize_mixed_dataset(
        small, large, selected_small_ids=small_ids, selected_large_ids=[],
        output_root=out, seed=0,
    )
    merged = json.loads((out / "material_provenance.json").read_text())
    source = json.loads((small / "material_provenance.json").read_text())
    assert merged == source


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
    expected_semantics = {
        "add": "fraction_of_large_pool_added",
        "replace": (
            "fraction_of_small_pool_replaced_capped_by_available_large_and_reserved_small_test"
        ),
    }
    for mode in ("add", "replace"):
        manifest = mvs.make_mixed_dataset_manifest(
            small, large, ratios=(0.0, 1.0), mode=mode, seed=0
        )
        for part in manifest["partitions"]:
            assert part["ratio_semantics"] == expected_semantics[mode]
            assert "large_capped" in part
    # replace ratio=1.0 wants 10 large but only 4 exist -> capped, and flagged.
    replace = mvs.make_mixed_dataset_manifest(
        small, large, ratios=(1.0,), mode="replace", seed=0
    )["partitions"][0]
    assert replace["large_capped"] is True
    assert replace["n_large_selected"] == 4
    # plan propagates both fields with the per-mode semantics.
    plan = mvs.plan_mixing_sweep({8: 10}, {8: 4}, modes=("replace",), ratios=(1.0,), seed=0)
    perm = plan["permutations"][0]
    assert perm["ratio_semantics"] == expected_semantics["replace"]
    assert perm["large_capped"] is True
    reserved = mvs.make_mixed_dataset_manifest(
        small,
        large,
        ratios=(1.0,),
        mode="replace",
        seed=0,
        reserved_small_ids={"s0", "s1"},
    )["partitions"][0]
    assert reserved["n_reserved_small"] == 2
    assert set(reserved["replace_cap_reasons"]) == {"available_large", "reserved_small_test"}


def _compo(permutations):
    return {
        (p["mode"], p["ratio"]): (
            p["n_small_selected"], p["n_large_selected"], p["total_size"]
        )
        for p in permutations
    }


def test_preview_matches_materialization_for_fixed_common_test_replace(small_large, tmp_path):
    import pipeline_ui as ui

    small, large = small_large
    body = {
        "small": {"8": str(small)},
        "large": {"8": str(large)},
        "modes": ["replace"],
        "ratios": [0.5, 1.0],
        "seed": 3,
        "split_policy": "fixed_common_test",
    }
    preview = _compo(ui.mixing_plan_payload(body)["permutations"])
    summary = mvs.run_mixing_sweep(
        {8: str(small)}, {8: str(large)}, tmp_path / "out",
        modes=("replace",), ratios=(0.5, 1.0), seed=3,
        split_policy="fixed_common_test", dry_run=False,
    )
    assert preview == _compo(summary["permutations"])
    # Non-trivial: the reserved common-test small id caps replace at ratio=1.0,
    # so it is NOT 100% large (this is exactly what would mismatch without the
    # split_policy-aware preview).
    assert preview[("replace", 1.0)][1] < 8


def test_mixing_plan_rejects_invalid_split_policy(small_large):
    import pipeline_ui as ui

    small, large = small_large
    with pytest.raises(ValueError, match="Unknown split_policy"):
        mvs.plan_mixing_sweep_from_roots(
            {8: str(small)}, {8: str(large)}, split_policy="bogus"
        )
    with pytest.raises(ValueError, match="Unknown split_policy"):
        ui.mixing_plan_payload(
            {"small": {"8": str(small)}, "large": {"8": str(large)}, "split_policy": "bogus"}
        )


def test_cli_mixing_sweep_accepts_split_policy(small_large, tmp_path):
    from ml_vs_siesta.cli import build_parser

    small, large = small_large
    out = tmp_path / "cli_out"
    args = build_parser().parse_args(
        [
            "mixing-sweep",
            "--small", f"8={small}",
            "--large", f"8={large}",
            "--modes", "replace",
            "--ratios", "1.0",
            "--seed", "3",
            "--split-policy", "fixed_common_test",
            "--output-root", str(out),
        ]
    )
    assert args.split_policy == "fixed_common_test"
    assert args.func(args) == 0
    prov = json.loads(
        (out / "size8_replace_r1p000" / "mixed_dataset_provenance.json").read_text()
    )
    assert prov["split_policy"] == "fixed_common_test"


def test_fixed_common_test_reuses_source_test_split_when_available(small_large, tmp_path):
    """Audit C2: the source's temporally blocked test split is reused verbatim."""
    small, large = small_large
    frozen_path = small / "frozen_split_manifest.json"
    frozen = json.loads(frozen_path.read_text())
    # Mark the source's own (tail) test split, as real datasets have.
    source_test = {"md_6", "md_7"}
    for row in frozen["rows"]:
        row["split"] = "test" if row["sample_id"] in source_test else "train"
    frozen_path.write_text(json.dumps(frozen), encoding="utf-8")

    small_ids = [s.sample_id for s in mvs.read_dataset_samples(small)]
    large_ids = [s.sample_id for s in mvs.read_dataset_samples(large)]
    out = tmp_path / "merged"
    mvs.materialize_mixed_dataset(
        small, large, selected_small_ids=small_ids, selected_large_ids=large_ids[:4],
        output_root=out, seed=2, split_policy="fixed_common_test",
    )
    assert _test_split_ids(out) == {f"small__{sid}" for sid in source_test}


def test_fixed_common_test_without_source_split_uses_temporal_tail(small_large, tmp_path):
    """No source test split -> test is the temporal TAIL, never interleaved."""
    small, large = small_large
    small_samples = mvs.read_dataset_samples(small)
    test_ids = mvs.fixed_common_test_ids(
        small_samples, mvs.DEFAULT_SPLIT_FRACTIONS, seed=3
    )
    # 8 snapshots, 15% test -> 1 snapshot: the temporally last frame.
    assert test_ids == {"md_7"}

    # After materialization no train/validation frame is temporally later than
    # any test frame (test is a contiguous tail: single temporal boundary).
    out = tmp_path / "merged"
    mvs.materialize_mixed_dataset(
        small, large,
        selected_small_ids=[s.sample_id for s in small_samples],
        selected_large_ids=[],
        output_root=out, seed=3, split_policy="fixed_common_test",
    )
    frozen = json.loads((out / "frozen_split_manifest.json").read_text())
    frame = lambda sid: int(sid.rsplit("_", 1)[1])  # noqa: E731
    test_frames = [frame(r["sample_id"]) for r in frozen["rows"] if r["split"] == "test"]
    fit_frames = [frame(r["sample_id"]) for r in frozen["rows"] if r["split"] != "test"]
    assert test_frames and fit_frames
    assert min(test_frames) > max(fit_frames)


def test_fixed_common_test_empty_validation_fails_loudly(small_large, tmp_path):
    small, large = small_large
    small_samples = mvs.read_dataset_samples(small)
    reserved = mvs.fixed_common_test_ids(small_samples, mvs.DEFAULT_SPLIT_FRACTIONS)
    # Reserved test + one extra snapshot: nothing left for validation.
    selection = sorted(reserved) + [
        next(s.sample_id for s in small_samples if s.sample_id not in reserved)
    ]
    with pytest.raises(mvs.DatasetMaterializeError, match="train and validation"):
        mvs.materialize_mixed_dataset(
            small, large,
            selected_small_ids=selection, selected_large_ids=[],
            output_root=tmp_path / "novalid", seed=0,
            split_policy="fixed_common_test",
        )


def test_fixed_common_test_raises_when_selection_excludes_reserved(small_large, tmp_path):
    small, large = small_large
    small_ids = [s.sample_id for s in mvs.read_dataset_samples(small)]
    large_ids = [s.sample_id for s in mvs.read_dataset_samples(large)]
    reserved = mvs.fixed_common_test_ids(
        sorted(small_ids), mvs.DEFAULT_SPLIT_FRACTIONS, seed=5
    )
    non_reserved = [sid for sid in small_ids if sid not in reserved]
    assert non_reserved and reserved  # sanity: the split is non-degenerate
    with pytest.raises(mvs.DatasetMaterializeError, match="empty test split"):
        mvs.materialize_mixed_dataset(
            small, large,
            selected_small_ids=non_reserved,  # deliberately drops the reserved test ids
            selected_large_ids=large_ids[:2],
            output_root=tmp_path / "empty_test", seed=5,
            split_policy="fixed_common_test",
        )


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


def test_aggregate_mae_vs_size_multi_seed_mean_std_n():
    """Audit I5: points aggregate per-seed replicates as mean±std with N."""
    base = {"size": 20, "mode": "add", "ratio": 0.0, "total_size": 20, "model": "graph2mat"}
    records = [
        {**base, "seed": 0, "h_mae_eV": 0.04},
        {**base, "seed": 1, "h_mae_eV": 0.06},
        {**base, "seed": 2, "h_mae_eV": 0.05},
    ]
    agg = mvs.aggregate_mae_vs_size(records)
    point = agg["curves"][0]["points"][0]
    assert point["mae"] == pytest.approx(0.05)
    assert point["mae_std"] == pytest.approx(0.01)
    assert point["n_seeds"] == 3
    assert point["exploratory"] is False
    assert agg["curves"][0]["exploratory"] is False
    assert agg["exploratory"] is False
    assert agg["warnings"] == []
    assert agg["min_seeds_for_claims"] == 3


def test_aggregate_mae_vs_size_single_seed_is_exploratory():
    records = [
        {"size": 20, "mode": "add", "ratio": 0.0, "seed": 0, "total_size": 20,
         "model": "graph2mat", "h_mae_eV": 0.05},
    ]
    agg = mvs.aggregate_mae_vs_size(records)
    point = agg["curves"][0]["points"][0]
    assert point["n_seeds"] == 1
    assert point["mae_std"] is None
    assert point["exploratory"] is True
    assert agg["exploratory"] is True
    assert any("EXPLORATORY" in w for w in agg["warnings"])


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


def test_parallel_sweep_seed_overrides_payload_hyperparam_seed(small_large, tmp_path, monkeypatch):
    """The sweep seed must reach the training overrides, beating any payload
    hyperparam. Otherwise every replicate trains from an identical init and the
    error bars only measure dataset selection, not optimization variance."""
    import pipeline_ui as ui

    small, large = small_large
    captured = []

    class _Capturing(_FakeParallelBenchmarkRunner):
        def start(self, payload):
            captured.append(payload)
            super().start(payload)

    monkeypatch.setattr(ui, "Graph2MatDeepHBenchmarkRunner", _Capturing)
    ui._run_mixing_sweep_parallel(
        {8: str(small)}, {8: str(large)}, tmp_path / "out",
        sizes=None, modes=("add",), ratios=(1.0,), seed=7,
        models=("graph2mat", "deeph"), epochs=None, performance=None,
        # Pinned seeds, exactly as the shipped mixing payloads used to carry.
        hyperparams={"graph2mat": {"seed_everything": 1}, "deeph": {"seed": 1}},
        progress_fn=None,
    )
    runs = captured[0]["training_sweep"]["manual_runs"]
    by_model = {r["model"]: r["overrides"] for r in runs}
    assert by_model["graph2mat"]["seed_everything"] == 7
    assert by_model["deeph"]["seed"] == 7


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


def test_parallel_sweep_skips_reconstructed_permutations(small_large, tmp_path, monkeypatch):
    import pipeline_ui as ui

    small, large = small_large
    monkeypatch.setattr(ui, "Graph2MatDeepHBenchmarkRunner", _FakeParallelBenchmarkRunner)
    out = tmp_path / "out"
    # ratio=0 is reconstructed, so the only trained dataset is ratio=1 -> perm_0.
    _write_common_metrics_csv(
        out / "parallel_run" / "sweep" / "graph2mat" / "perm_0", "graph2mat", 0.05
    )
    _write_common_metrics_csv(
        out / "parallel_run" / "sweep" / "deeph" / "perm_0", "deeph", 0.08
    )
    reconstructed = [
        {"size": 8, "mode": "add", "ratio": 0.0, "total_size": 8, "model": "graph2mat", "h_mae_eV": 0.01},
        {"size": 8, "mode": "add", "ratio": 0.0, "total_size": 8, "model": "deeph", "h_mae_eV": 0.02},
    ]
    summary = ui._run_mixing_sweep_parallel(
        {8: str(small)}, {8: str(large)}, out,
        sizes=None, modes=("add",), ratios=(0.0, 1.0), seed=0,
        models=("graph2mat", "deeph"), epochs=None, performance=None,
        reconstructed_records=reconstructed, progress_fn=None,
    )
    by_ratio = {perm["ratio"]: perm for perm in summary["permutations"]}
    assert by_ratio[0.0]["status"] == "reconstructed"
    assert by_ratio[1.0]["status"] == "trained"
    assert len(summary["records"]) == 4


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


def test_extract_model_h_mae_eV_ignores_non_test_splits():
    import pipeline_ui as ui

    payload = {
        "results": {
            "graph2mat": {
                "train": {"h_mae_eV": 0.001},
                "validation": {"h_mae_eV": 0.002},
                "test": {"h_mae_eV": 0.5},
            }
        }
    }
    assert ui._extract_model_h_mae_eV(payload, ("graph2mat",)) == {
        "graph2mat": {"h_mae_eV": 0.5}
    }
    # Explicit split field on the node also filters.
    payload = {
        "rows": [
            {"method": "deeph", "split": "train", "h_mae_eV": 0.001},
            {"method": "deeph", "split": "test", "h_mae_eV": 0.7},
        ]
    }
    assert ui._extract_model_h_mae_eV(payload, ("deeph",)) == {"deeph": {"h_mae_eV": 0.7}}


def test_extract_model_h_mae_eV_ambiguous_distinct_values_fail_loudly():
    import pipeline_ui as ui

    payload = {
        "summary": {"graph2mat": {"h_mae_eV": 0.10}},
        "plots": {"graph2mat": {"h_mae_eV": 0.99}},
    }
    with pytest.raises(RuntimeError, match="Ambiguous h_mae_eV"):
        ui._extract_model_h_mae_eV(payload, ("graph2mat",))
    # Duplicated but IDENTICAL values are fine (same metric surfaced twice).
    payload = {
        "summary": {"graph2mat": {"h_mae_eV": 0.10}},
        "plots": {"graph2mat": {"h_mae_eV": 0.10}},
    }
    assert ui._extract_model_h_mae_eV(payload, ("graph2mat",)) == {
        "graph2mat": {"h_mae_eV": 0.10}
    }


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

    def _write_csv(metrics_dir, rows, manifest_split=None):
        metrics_dir.mkdir(parents=True, exist_ok=True)
        (metrics_dir / "kpoint_matrix_metrics.csv").write_text(
            "sample,row_type,h_mae_eV\n" + "".join(f"{r}\n" for r in rows),
            encoding="utf-8",
        )
        if manifest_split is not None:
            (metrics_dir / "manifest.json").write_text(
                json.dumps({"split": manifest_split}), encoding="utf-8"
            )

    # Manifest declares the test split -> used; per_k rows are ignored
    # (weighted_sample rows only, no double counting).
    run = tmp_path / "run"
    _write_csv(
        run / "metrics" / "graph2mat" / "eval" / "metrics",
        ["s0,per_k,0.90", "s0,weighted_sample,0.10", "s1,weighted_sample,0.20"],
        manifest_split="test",
    )
    assert ui._mixing_metrics_from_run_metrics(run, "graph2mat") == {
        "graph2mat": {"h_mae_eV": pytest.approx(0.15)}
    }

    # Train/validation CSVs are never averaged into the metric.
    run2 = tmp_path / "run2"
    _write_csv(
        run2 / "cross_evaluations" / "on__test_md" / "metrics",
        ["s0,weighted_sample,0.30"],
    )
    _write_csv(
        run2 / "cross_evaluations" / "on__train_md" / "metrics",
        ["s0,weighted_sample,0.001"],
    )
    _write_csv(
        run2 / "cross_evaluations" / "on__validation_md" / "metrics",
        ["s0,weighted_sample,0.002"],
    )
    assert ui._mixing_metrics_from_run_metrics(run2, "graph2mat") == {
        "graph2mat": {"h_mae_eV": pytest.approx(0.30)}
    }

    # No split evidence at all -> skipped, metric reported missing (not wrong).
    run3 = tmp_path / "run3"
    _write_csv(run3 / "metrics" / "eval" / "metrics", ["s0,weighted_sample,0.40"])
    assert ui._mixing_metrics_from_run_metrics(run3, "graph2mat") == {}


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


# --------------------------------------------------------------------------- #
# Fase 7/8 (audit): fixed_stratified_test, no-leakage guard, composition
# --------------------------------------------------------------------------- #
def test_fixed_stratified_test_contains_both_domains(small_large, tmp_path):
    small, large = small_large
    small_ids = [s.sample_id for s in mvs.read_dataset_samples(small)]
    large_ids = [s.sample_id for s in mvs.read_dataset_samples(large)]
    out = tmp_path / "stratified"
    summary = mvs.materialize_mixed_dataset(
        small, large, selected_small_ids=small_ids, selected_large_ids=large_ids,
        output_root=out, seed=0, split_policy="fixed_stratified_test",
    )
    composition = summary["composition"]
    assert composition["test_small_size"] > 0
    assert composition["test_large_size"] > 0
    assert summary["evaluation_scope"] == "small_and_large"
    # Test rows come from both origins.
    rows = list(csv.DictReader((out / "splits" / "test_manifest.csv").open()))
    origins = {row["origin"] for row in rows}
    assert origins == {"small", "large"}


def test_fixed_stratified_test_is_identical_across_ratios_and_seeds(small_large, tmp_path):
    small, large = small_large
    test_sets = []
    for i, (ratio_ids, seed) in enumerate([(4, 0), (8, 0), (8, 7)]):
        large_ids = [s.sample_id for s in mvs.read_dataset_samples(large)][:ratio_ids]
        small_ids = [s.sample_id for s in mvs.read_dataset_samples(small)]
        out = tmp_path / f"strat{i}"
        mvs.materialize_mixed_dataset(
            small, large, selected_small_ids=small_ids,
            # include reserved large test ids (temporal tail) in every selection
            selected_large_ids=sorted(set(large_ids) | mvs.fixed_common_test_ids(
                mvs.read_dataset_samples(large), mvs.DEFAULT_SPLIT_FRACTIONS, 0
            )),
            output_root=out, seed=seed, split_policy="fixed_stratified_test",
        )
        rows = list(csv.DictReader((out / "splits" / "test_manifest.csv").open()))
        test_sets.append(sorted(row["sample_id"] for row in rows))
    assert test_sets[0] == test_sets[1] == test_sets[2]


def test_blocked_stratified_gap_ratio_zero_keeps_same_composition(small_large, tmp_path):
    small, large = small_large
    summary = mvs.run_mixing_sweep(
        {8: small},
        {8: large},
        tmp_path / "blocked",
        sizes=[8],
        modes=("add",),
        ratios=(0.0,),
        seed=0,
        split_policy="blocked_stratified_gap",
        split_fractions=(0.8, 0.1, 0.1),
        temporal_gap=1,
        dry_run=False,
        launch_fn=None,
    )
    perm = summary["permutations"][0]
    composition = perm["materialize"]["composition"]
    assert perm["n_large_selected"] == 0
    assert composition["n_large_train"] == 0
    assert composition["test_large_size"] == 0
    assert composition["evaluation_scope"] == "small_and_large"


def test_blocked_stratified_gap_uses_fresh_split_not_source_labels(small_large, tmp_path):
    small, large = small_large
    manifest_path = small / "frozen_split_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["rows"][0]["split"] = "test"
    manifest_path.write_text(json.dumps(manifest))
    small_ids = [s.sample_id for s in mvs.read_dataset_samples(small)]
    large_ids = [s.sample_id for s in mvs.read_dataset_samples(large)]
    summary = mvs.materialize_mixed_dataset(
        small,
        large,
        selected_small_ids=small_ids,
        selected_large_ids=large_ids[:4],
        output_root=tmp_path / "fresh",
        split_policy="blocked_stratified_gap",
        split_fractions=(0.8, 0.1, 0.1),
        temporal_gap=1,
    )
    assert summary["split_policy"] == "blocked_stratified_gap"


def test_source_test_snapshot_never_trains(small_large, tmp_path):
    small, large = small_large
    # Mark every small snapshot as source-test: any train/validation assignment
    # under resplit_combined must fail closed.
    manifest_path = small / "frozen_split_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for row in manifest["rows"]:
        row["split"] = "test"
    manifest_path.write_text(json.dumps(manifest))
    small_ids = [s.sample_id for s in mvs.read_dataset_samples(small)]
    large_ids = [s.sample_id for s in mvs.read_dataset_samples(large)]
    with pytest.raises(mvs.DatasetMaterializeError, match="leakage"):
        mvs.materialize_mixed_dataset(
            small, large, selected_small_ids=small_ids, selected_large_ids=large_ids[:4],
            output_root=tmp_path / "leaky", seed=0, split_policy="resplit_combined",
        )
    # Explicit non-scientific override still works.
    mvs.materialize_mixed_dataset(
        small, large, selected_small_ids=small_ids, selected_large_ids=large_ids[:4],
        output_root=tmp_path / "leaky_ok", seed=0, split_policy="resplit_combined",
        allow_source_test_in_train=True,
    )


def test_ratio_applies_to_training_pool_not_total():
    from ml_vs_siesta.dataset_mixing import make_mixed_dataset_manifest

    small = [{"id": f"s{i}"} for i in range(10)]
    large = [{"id": f"l{i}"} for i in range(10)]
    reserved_large = {"l8", "l9"}
    manifest = make_mixed_dataset_manifest(
        small, large, ratios=[0.5], mode="add", seed=0,
        reserved_large_ids=reserved_large,
    )
    part = manifest["partitions"][0]
    # ratio 0.5 of the 8-id large TRAINING pool -> 4 train + 2 reserved test.
    assert part["n_large_selected"] == 6
    assert part["n_reserved_large"] == 2
    assert set(reserved_large) <= set(part["selected_ids"])
    assert part["requested_count_float"] == 4.0
    assert part["rounding_policy"] == "python_round_half_even"


def test_composition_metrics_fractions(small_large, tmp_path):
    small, large = small_large
    small_ids = [s.sample_id for s in mvs.read_dataset_samples(small)]
    large_ids = [s.sample_id for s in mvs.read_dataset_samples(large)]
    summary = mvs.materialize_mixed_dataset(
        small, large, selected_small_ids=small_ids, selected_large_ids=large_ids[:4],
        output_root=tmp_path / "comp", seed=1,
    )
    composition = summary["composition"]
    total_train = composition["n_small_train"] + composition["n_large_train"]
    assert composition["actual_train_size"] == total_train
    assert composition["materialized_total_size"] == 12
    # atoms: small snapshots have 2 atoms, large 50.
    assert composition["small_atoms_total"] == 2 * composition["n_small_train"]
    assert composition["large_atoms_total"] == 50 * composition["n_large_train"]
    expected = composition["large_atoms_total"] / (
        composition["large_atoms_total"] + composition["small_atoms_total"]
    )
    assert abs(composition["actual_large_fraction_by_atoms"] - expected) < 1e-12


# --------------------------------------------------------------------------- #
# Fase 10 (audit): fail-closed transactional materialization
# --------------------------------------------------------------------------- #
def test_invalid_snapshot_blocks_materialization(small_large, tmp_path):
    small, large = small_large
    # Break one source snapshot: remove a required artifact.
    victim = next((small / "MD_steps" / "3").glob("*.HSX"))
    victim.unlink()
    small_ids = [s.sample_id for s in mvs.read_dataset_samples(small)]
    large_ids = [s.sample_id for s in mvs.read_dataset_samples(large)]
    out = tmp_path / "merged_invalid"
    with pytest.raises(mvs.DatasetMaterializeError, match="NOT materialized"):
        mvs.materialize_mixed_dataset(
            small, large, selected_small_ids=small_ids, selected_large_ids=large_ids[:4],
            output_root=out, seed=1,
        )
    # No apparently-complete dataset: final root absent, no manifests anywhere.
    assert not out.exists()
    partials = list(out.parent.glob(f"{out.name}.partial-*"))
    assert partials, "diagnostic partial should be kept"
    assert (partials[0] / "MATERIALIZATION_FAILED.json").exists()
    assert not (partials[0] / "frozen_split_manifest.json").exists()
    assert not (partials[0] / "benchmark_dataset_manifest.json").exists()


def test_successful_materialization_leaves_no_partial(small_large, tmp_path):
    small, large = small_large
    small_ids = [s.sample_id for s in mvs.read_dataset_samples(small)]
    large_ids = [s.sample_id for s in mvs.read_dataset_samples(large)]
    out = tmp_path / "merged_ok"
    mvs.materialize_mixed_dataset(
        small, large, selected_small_ids=small_ids, selected_large_ids=large_ids[:4],
        output_root=out, seed=1,
    )
    assert out.exists()
    assert not list(out.parent.glob(f"{out.name}.partial-*"))
    validation = json.loads((out / "artifact_validation.json").read_text())
    assert validation["valid"] is True and validation["status"] == "validated"
    # Manifest rows point at the FINAL paths, not the partial ones.
    frozen = json.loads((out / "frozen_split_manifest.json").read_text())
    for row in frozen["rows"]:
        assert ".partial-" not in row["sample_dir"]
        assert Path(row["sample_dir"]).exists()
