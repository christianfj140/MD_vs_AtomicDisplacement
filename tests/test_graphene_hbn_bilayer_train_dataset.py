"""Graphene-hBN bilayer material bundles + the 3-stacking train merge builder."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "Comparison" / "scripts"
SHARED = REPO_ROOT / "shared"
for path in (SCRIPTS, SHARED):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import build_graphene_hbn_bilayer_train_dataset as bilayer  # noqa: E402
from material_presets import resolve_material_bundle  # noqa: E402


def _fake_source(root: Path, stacking: str, *, basis_hash: str, n_train: int, n_val: int) -> Path:
    """Minimal materialized dataset: provenance + frozen manifest + sample dirs."""
    dataset = root / stacking
    (dataset / "material_basis").mkdir(parents=True)
    (dataset / "material_basis" / "C.ion.xml").write_text("basis", encoding="utf-8")
    (dataset / "C.psf").write_text("pseudo", encoding="utf-8")
    provenance = {
        "label": stacking,
        "basis_file_sha256": {"C.ion.xml": basis_hash},
        "pseudopotential_sha256": {"C.psf": "pseudo-hash"},
        "environment": {},
        "species": [{"atomic_number": 6, "index": 1, "label": "C"}],
    }
    (dataset / "material_provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    rows = []
    for split, count in (("train", n_train), ("validation", n_val)):
        for i in range(count):
            sample_dir = dataset / "splits" / split / str(i)
            sample_dir.mkdir(parents=True)
            (sample_dir / "RUN.fdf").write_text(f"SystemLabel {stacking}\n", encoding="utf-8")
            (sample_dir / "RUN.out").write_text("ok\n", encoding="utf-8")
            (sample_dir / f"{stacking}.TSHS").write_bytes(b"tshs")
            rows.append({"sample_id": f"md_{split}_{i}", "split": split, "sample_dir": str(sample_dir)})
    (dataset / "frozen_split_manifest.json").write_text(json.dumps({"rows": rows}), encoding="utf-8")
    return dataset


def _patch_finalizers(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Skip the SIESTA-artifact-heavy validation; capture what the merge produced."""
    captured: dict = {}

    class _Result:
        valid = True
        errors: list = []

        def to_dict(self):
            return {"valid": True}

    def fake_validate(output_root, **_kwargs):
        captured["snapshot_dirs"] = _kwargs.get("snapshot_dirs")
        return _Result()

    def fake_manifests(*, dataset_root, split_root, **_kwargs):
        rows = []
        counts = {"train": 0, "validation": 0, "test": 0}
        for split in ("train", "validation"):
            manifest = split_root / f"{split}_manifest.csv"
            n = max(0, len(manifest.read_text().splitlines()) - 1)
            counts[split] = n
            rows.extend([split] * n)
        captured["split_counts"] = counts
        return ({"benchmark_dataset_id": "test", "benchmark_ready": True}, {"split_counts": counts})

    monkeypatch.setattr(bilayer, "validate_dataset", fake_validate)
    monkeypatch.setattr(bilayer, "write_benchmark_manifests", fake_manifests)
    return captured


def test_committed_bilayer_presets_share_basis_and_have_three_species() -> None:
    hashes = {}
    for preset in bilayer.STACKING_PRESETS:
        validated = resolve_material_bundle({"material": {"preset": preset}}).validated
        labels = sorted(sp.label for sp in validated.species)
        assert labels == ["B", "C", "N"], (preset, labels)
        hashes[preset] = validated.basis_file_sha256
    reference = hashes[bilayer.STACKING_PRESETS[0]]
    assert all(value == reference for value in hashes.values())


def test_merge_sums_samples_without_id_collision(tmp_path: pytest.MonkeyPatch, monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _patch_finalizers(monkeypatch)
    sources = [
        _fake_source(tmp_path, "graphene_hBN_AA", basis_hash="h", n_train=3, n_val=1),
        _fake_source(tmp_path, "graphene_hBN_AB1", basis_hash="h", n_train=2, n_val=2),
        _fake_source(tmp_path, "graphene_hBN_AB2", basis_hash="h", n_train=4, n_val=1),
    ]
    output = tmp_path / "merged"
    result = bilayer.build_dataset(sources, output, overwrite=False)
    assert result["n_samples"] == (3 + 2 + 4) + (1 + 2 + 1)
    assert captured["split_counts"]["train"] == 3 + 2 + 4
    assert captured["split_counts"]["validation"] == 1 + 2 + 1

    ids = []
    for split in ("train", "validation"):
        for row in (output / "splits" / f"{split}_manifest.csv").read_text().splitlines()[1:]:
            ids.append(row.split(",", 1)[0])
    assert len(ids) == len(set(ids)), "sample ids collided across stackings"
    assert all(any(s in i for s in ("AA", "AB1", "AB2")) for i in ids)


def test_bilayer_payload_is_prediction_only() -> None:
    path = REPO_ROOT / "Comparison/config/graphene_hbn_bilayer_to_moire_predict_metrics_payload.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["action"] == "predict_metrics"
    assert payload["models"] == ["graph2mat", "deeph"]
    assert len(payload["pairs"]) == 1
    pair = payload["pairs"][0]
    assert pair["direction"] == "bilayer_to_moire"
    assert pair["source"] == "Comparison/datasets/graphene_hBN_bilayer_train"
    assert pair["target"].startswith("Comparison/datasets/graphene_hBN_moire")
    # existing_artifacts keyed by source basename (what the runner looks up).
    assert "graphene_hBN_bilayer_train" in payload["existing_artifacts"]
    art = payload["existing_artifacts"]["graphene_hBN_bilayer_train"]
    assert "graph2mat_training_dir" in art and "deeph_save_dir" in art
    for key in ("hyperparams", "early_stopping", "performance"):
        assert payload.get(key), key


def test_bilayer_ui_subsection_is_additive_and_last() -> None:
    html = (REPO_ROOT / "Comparison/ui/index.html").read_text(encoding="utf-8")
    app = (REPO_ROOT / "Comparison/ui/app.js").read_text(encoding="utf-8")
    panel_start = html.index('aria-labelledby="ct-bilayer-heading"')
    panel = html[panel_start:html.index("</section>", panel_start)]
    assert "Cross testing bicapa grafeno/hBN" in panel
    assert "Checkpoints existentes · bicapa→moiré" in panel
    assert "ct-bilayer-evaluate" in panel
    assert "ct-bilayer-mae-chart" in panel
    assert "ct-bilayer-log" in panel
    # Additive: appears after both the normal and the vacancy subsections.
    assert panel_start > html.index('id="ct-mae-chart"')
    assert panel_start > html.index('aria-labelledby="ct-vacancy-heading"')
    # JS wiring hits the bilayer routes and leaves vacancy/sweep untouched.
    assert 'request("/api/cross-testing/bilayer/launch"' in app
    assert 'request("/api/cross-testing/bilayer/metrics")' in app
    assert 'request("/api/cross-testing/bilayer/status")' in app
    assert '"ct-bilayer-mae-chart"' in app


def test_bilayer_runner_is_independent() -> None:
    import pipeline_ui as ui

    assert ui.CROSS_TESTING_BILAYER_RUNNER is not ui.CROSS_TESTING_RUNNER
    assert ui.CROSS_TESTING_BILAYER_RUNNER is not ui.CROSS_TESTING_VACANCY_RUNNER
    assert ui.CROSS_TESTING_BILAYER_RUNNER._output_root != ui.CROSS_TESTING_VACANCY_RUNNER._output_root
    assert ui.CROSS_TESTING_BILAYER_RUNNER._output_root != ui.CROSS_TESTING_RUNNER._output_root


def test_merge_fails_closed_on_basis_hash_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_finalizers(monkeypatch)
    sources = [
        _fake_source(tmp_path, "graphene_hBN_AA", basis_hash="h", n_train=2, n_val=1),
        _fake_source(tmp_path, "graphene_hBN_AB1", basis_hash="DIFFERENT", n_train=2, n_val=1),
        _fake_source(tmp_path, "graphene_hBN_AB2", basis_hash="h", n_train=2, n_val=1),
    ]
    with pytest.raises(RuntimeError, match="Basis hashes differ"):
        bilayer.build_dataset(sources, tmp_path / "merged", overwrite=False)
    assert not (tmp_path / "merged").exists()
