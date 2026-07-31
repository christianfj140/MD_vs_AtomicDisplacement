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


def _bilayer_fdf(stacking: str) -> str:
    return f"""SystemLabel {stacking}
NumberOfAtoms 6
NumberOfSpecies 3
%block ChemicalSpeciesLabel
1 6 C
2 5 B
3 7 N
%endblock ChemicalSpeciesLabel
LatticeConstant 1.0 Ang
%block LatticeVectors
2.48 0 0
-1.24 2.147743 0
0 0 20
%endblock LatticeVectors
AtomicCoordinatesFormat Ang
%block AtomicCoordinatesAndAtomicSpecies
0 0 6 2
0 1.4318286667 6 3
0 0 9.35 1
0 1.4318286667 9.35 1
0 1.4318286667 12.7 1
1.24 0.7159143333 12.7 1
%endblock AtomicCoordinatesAndAtomicSpecies
"""


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
            (sample_dir / "RUN.fdf").write_text(_bilayer_fdf(stacking), encoding="utf-8")
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
        _fake_source(tmp_path, "bilayer_graphene_hBN_AA", basis_hash="h", n_train=3, n_val=1),
        _fake_source(tmp_path, "bilayer_graphene_hBN_AB1", basis_hash="h", n_train=2, n_val=2),
        _fake_source(tmp_path, "bilayer_graphene_hBN_AB2", basis_hash="h", n_train=4, n_val=1),
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
    assert "B. Predicción espectral ML-only" in panel
    assert "No target H reference" in panel
    assert "ct-spectral-bands-chart" in panel
    assert "ct-spectral-dos-chart" in panel
    assert "ct-spectral-reference-validation" in panel
    assert "ct-spectral-downloads" in panel
    assert "ct-spectral-visible-bands" in panel
    assert "ct-spectral-energy-window" in panel
    assert "ct-spectral-plot-mode" in panel
    assert "ct-spectral-weight" in panel
    assert "ct-spectral-artifact" in panel
    assert "ct-spectral-diagnostic-window" in panel
    assert "projected_progress" in app
    assert "Smoke legacy aislado" in panel
    spectral_plot = app[app.index("function ctSpectralBandPlot"):app.index("async function ctSpectralRender")]
    assert "point.band_index" in spectral_plot
    assert "showlegend: index === 0" in spectral_plot
    assert 'tickmode: "array"' in spectral_plot
    assert 'yref: "y"' in spectral_plot
    assert 'mode === "projected"' in spectral_plot
    assert "point.weight_c_pz" in spectral_plot
    assert "point.layer_polarization" in spectral_plot
    assert "range: [-energyWindow, energyWindow]" in spectral_plot
    for route in ("plan", "launch", "results", "stop", "artifact"):
        assert f"/api/cross-testing/bilayer/spectral/{route}" in app
    for route in ("plan", "launch", "status", "results", "stop"):
        assert f"/api/cross-testing/bilayer/spectral/{route}" in (
            REPO_ROOT / "Comparison/scripts/pipeline_ui.py"
        ).read_text(encoding="utf-8")


def test_bilayer_runner_is_independent() -> None:
    import pipeline_ui as ui

    assert ui.CROSS_TESTING_BILAYER_RUNNER is not ui.CROSS_TESTING_RUNNER
    assert ui.CROSS_TESTING_BILAYER_RUNNER is not ui.CROSS_TESTING_VACANCY_RUNNER
    assert ui.CROSS_TESTING_BILAYER_RUNNER._output_root != ui.CROSS_TESTING_VACANCY_RUNNER._output_root
    assert ui.CROSS_TESTING_BILAYER_RUNNER._output_root != ui.CROSS_TESTING_RUNNER._output_root


def test_magic_angle_projected_solver_config_has_safe_smoke_and_production_settings() -> None:
    config = json.loads((
        REPO_ROOT / "Comparison/config/graphene_hbn_magic_angle_spectral_campaign.json"
    ).read_text(encoding="utf-8"))
    solver = config["solver"]
    assert solver["compute_backend"] == "gpu_cudss"
    assert solver["project_mulliken"] is True
    assert solver["tier_b_bands"] == 256
    assert solver["tier_b_points_per_segment"] == 11
    assert solver["production_points_per_segment"] == 51
    assert solver["tier_c_kmesh"] == [6, 6, 1]
    assert solver["dos_production_kmesh"] == [24, 24, 1]
    assert solver["auto_launch_dos_production"] is False


def test_spectral_artifact_download_is_confined_to_campaign(tmp_path: Path) -> None:
    import pipeline_ui as ui

    runner = ui.MoireSpectralCampaignRunner()
    runner.root = tmp_path / "campaign"
    assert runner.status()["disk"]["free_percent"] > 0
    artifact = runner.root / "summary" / "result.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}\n", encoding="utf-8")

    assert runner.artifact_path("summary/result.json") == artifact
    with pytest.raises(RuntimeError, match="fuera de la campaña"):
        runner.artifact_path("../secret.json")
    with pytest.raises(RuntimeError, match="Tipo de artefacto"):
        runner.artifact_path("overlaps.h5")


def test_spectral_results_prioritizes_live_training_over_stale_manifest(tmp_path: Path) -> None:
    import pipeline_ui as ui

    runner = ui.MoireSpectralCampaignRunner()
    runner.root = tmp_path / "campaign"
    (runner.root / "training/n30/control").mkdir(parents=True)
    (runner.root / "status.json").write_text(
        '{"running": true, "current_stage": "train"}\n',
        encoding="utf-8",
    )
    (runner.root / "training/training_campaign_manifest.json").write_text(
        '{"status": "failed"}\n',
        encoding="utf-8",
    )
    (runner.root / "training/n30/control/status.json").write_text(
        '{"status": {"running": true, "stage": "training_sweep"}}\n',
        encoding="utf-8",
    )

    training = runner.results()["training"]
    assert training["status"] == "running"
    assert training["previous_status"] == "failed"
    assert training["live_controls"][0]["status"]["stage"] == "training_sweep"


def test_merge_fails_closed_on_basis_hash_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_finalizers(monkeypatch)
    sources = [
        _fake_source(tmp_path, "bilayer_graphene_hBN_AA", basis_hash="h", n_train=2, n_val=1),
        _fake_source(tmp_path, "bilayer_graphene_hBN_AB1", basis_hash="DIFFERENT", n_train=2, n_val=1),
        _fake_source(tmp_path, "bilayer_graphene_hBN_AB2", basis_hash="h", n_train=2, n_val=1),
    ]
    with pytest.raises(RuntimeError, match="Basis hashes differ"):
        bilayer.build_dataset(sources, tmp_path / "merged", overwrite=False)
    assert not (tmp_path / "merged").exists()


def test_merge_rejects_legacy_four_atom_cell(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_finalizers(monkeypatch)
    sources = [
        _fake_source(tmp_path, preset, basis_hash="h", n_train=2, n_val=1)
        for preset in bilayer.STACKING_PRESETS
    ]
    bad_fdf = sources[0] / "splits" / "train" / "0" / "RUN.fdf"
    bad_fdf.write_text(
        _bilayer_fdf(sources[0].name).replace("NumberOfAtoms 6", "NumberOfAtoms 4").replace(
            "0 1.4318286667 12.7 1\n1.24 0.7159143333 12.7 1\n", ""
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="six-atom C4BN"):
        bilayer.build_dataset(sources, tmp_path / "merged", overwrite=False)


def test_balanced_train_limit_is_a_nested_prefix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_finalizers(monkeypatch)
    sources = [
        _fake_source(tmp_path, preset, basis_hash="h", n_train=4, n_val=1)
        for preset in bilayer.STACKING_PRESETS
    ]
    small = tmp_path / "small"
    large = tmp_path / "large"
    bilayer.build_dataset(sources, small, overwrite=False, train_size=6)
    bilayer.build_dataset(sources, large, overwrite=False, train_size=9)

    def ids(dataset: Path) -> set[str]:
        rows = (dataset / "splits/train_manifest.csv").read_text(encoding="utf-8").splitlines()[1:]
        return {row.split(",", 1)[0] for row in rows}

    assert len(ids(small)) == 6
    assert len(ids(large)) == 9
    assert ids(small) <= ids(large)


def test_train_limit_must_balance_across_stackings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_finalizers(monkeypatch)
    sources = [
        _fake_source(tmp_path, preset, basis_hash="h", n_train=2, n_val=1)
        for preset in bilayer.STACKING_PRESETS
    ]
    with pytest.raises(RuntimeError, match="divisible by 3"):
        bilayer.build_dataset(sources, tmp_path / "merged", overwrite=False, train_size=5)


def test_explicit_train_quotas_support_weighted_nested_campaign(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_finalizers(monkeypatch)
    sources = [
        _fake_source(tmp_path, f"source_{index}", basis_hash="h", n_train=4, n_val=1)
        for index in range(3)
    ]
    result = bilayer.build_dataset(
        sources,
        tmp_path / "merged",
        overwrite=False,
        train_size=6,
        train_quotas=[1, 3, 2],
    )
    assert result["requested_train_size"] == 6
    assert result["train_quotas"] == [1, 3, 2]
