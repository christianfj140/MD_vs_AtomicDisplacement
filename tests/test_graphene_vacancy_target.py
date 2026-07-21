"""Graphene 5x5 monovacancy material, target builder, and UI payload safety."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "Comparison" / "scripts"
SHARED = REPO_ROOT / "shared"
for path in (SCRIPTS, SHARED):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import build_graphene_5x5_vacancy_target as vacancy  # noqa: E402
import pipeline_ui as ui  # noqa: E402
from fdf_materialization import extract_fdf_structure  # noqa: E402
from material_bundle import read_fdf_block  # noqa: E402


def _source_fdf(path: Path) -> Path:
    rows = [f" {i + 0.1:.12f}  0.000000000000  0.000000000000  1" for i in range(50)]
    rows[24] = " 30.000000000000  30.000000000000  0.000000000000  1"
    text = "\n".join(
        [
            "SystemName graphene_5x5",
            "SystemLabel graphene_5x5",
            "NumberOfAtoms 50",
            "NumberOfSpecies 1",
            "%block ChemicalSpeciesLabel",
            "1 6 C",
            "%endblock ChemicalSpeciesLabel",
            "LatticeConstant 1.0 Ang",
            "%block LatticeVectors",
            "60 0 0",
            "0 60 0",
            "0 0 20",
            "%endblock LatticeVectors",
            "AtomicCoordinatesFormat Ang",
            "%block AtomicCoordinatesAndAtomicSpecies",
            *rows,
            "%endblock AtomicCoordinatesAndAtomicSpecies",
            "MD.TypeOfRun Verlet",
            "MD.Steps 4",
            "WriteMDHistory T",
            "Lua.Script md_store.lua",
        ]
    )
    path.write_text(text + "\n", encoding="utf-8")
    return path


def test_committed_vacancy_material_has_49_carbons() -> None:
    path = REPO_ROOT / "materials/graphene_5x5_vacancy/RUN.fdf"
    rows = read_fdf_block(path, "AtomicCoordinatesAndAtomicSpecies")
    assert len(rows) == 49
    assert "NumberOfAtoms      49" in path.read_text(encoding="utf-8")
    assert not any(tuple(map(float, row.split()[:3])) == (0.5, 0.5, 0.0) for row in rows)


def test_vacancy_transform_preserves_other_atoms_and_writes_metadata(tmp_path: Path) -> None:
    source = _source_fdf(tmp_path / "RUN.fdf")
    source_structure = extract_fdf_structure(source)
    text, metadata = vacancy.vacancy_fdf(source, 24)
    output = tmp_path / "vacancy.fdf"
    output.write_text(text, encoding="utf-8")
    target = extract_fdf_structure(output)
    assert target.atom_count == 49
    assert target.positions_ang == source_structure.positions_ang[:24] + source_structure.positions_ang[25:]
    assert target.lattice_vectors_ang == source_structure.lattice_vectors_ang
    assert metadata["removed_atom_index"] == 24
    assert metadata["removed_atom_species"] == "C"
    assert metadata["removed_atom_position_fractional"] == [0.5, 0.5, 0.0]
    assert "MD.TypeOfRun" not in text
    assert "Lua.Script" not in text
    with pytest.raises(RuntimeError, match="not the central vacancy site"):
        vacancy.vacancy_fdf(source, 23)


def test_dry_run_never_invokes_siesta(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sample_dir = tmp_path / "source"
    sample_dir.mkdir()
    _source_fdf(sample_dir / "RUN.fdf")
    sample = SimpleNamespace(sample_id="md_1", sample_dir=sample_dir, split="test")
    monkeypatch.setattr(vacancy, "_selected_samples", lambda *_args: [sample])
    monkeypatch.setattr(vacancy, "run_siesta", lambda *_args, **_kwargs: pytest.fail("SIESTA invoked"))
    plan = vacancy.dry_run_plan(tmp_path, "test", 1, 24)
    assert plan["n_selected"] == 1
    assert plan["siesta_invoked"] is False


def test_siesta_failure_leaves_no_target(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "source_dataset"
    sample_dir = source / "sample"
    sample_dir.mkdir(parents=True)
    _source_fdf(sample_dir / "RUN.fdf")
    sample = SimpleNamespace(sample_id="md_1", sample_dir=sample_dir, split="test")
    monkeypatch.setattr(vacancy, "_selected_samples", lambda *_args: [sample])
    monkeypatch.setattr(vacancy, "_copy_basis", lambda *_args: None)
    monkeypatch.setattr(vacancy, "stage_required_pseudopotentials", lambda **_kwargs: [])
    monkeypatch.setattr(vacancy, "run_siesta", lambda *_args, **_kwargs: {"returncode": 1})
    output = tmp_path / "failed_target"
    with pytest.raises(RuntimeError, match="SIESTA reference failed"):
        vacancy.build_target(
            source,
            output,
            source_split="test",
            limit=1,
            atom_index=24,
            siesta_command="false",
            overwrite=False,
        )
    assert not output.exists()


def test_cross_testing_payload_path_is_confined(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = tmp_path / "config"
    config.mkdir()
    payload = config / "campaign.json"
    payload.write_text(json.dumps({"action": "preview", "models": ["graph2mat"]}), encoding="utf-8")
    monkeypatch.setattr(ui, "CROSS_TESTING_CONFIG_ROOT", config)
    resolved = ui._cross_testing_resolve_body({"payload_path": str(payload), "action": "predict_metrics"})
    assert resolved["action"] == "predict_metrics"
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Comparison/config"):
        ui._cross_testing_resolve_body({"payload_path": str(outside)})
    with pytest.raises(RuntimeError, match="Comparison/config"):
        ui._cross_testing_resolve_body({"payload_path": str(config / "../outside.json")})


def test_vacancy_campaign_is_prediction_only() -> None:
    path = REPO_ROOT / "Comparison/config/graphene_w90_5x5_to_vacancy_predict_metrics_payload.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    normal = json.loads(
        (REPO_ROOT / "Comparison/config/graphene_w90_5x5_cross_structure_predict_metrics_payload.json")
        .read_text(encoding="utf-8")
    )
    assert payload["action"] == "predict_metrics"
    assert payload["models"] == ["graph2mat", "deeph"]
    assert len(payload["pairs"]) == 26
    assert {pair["direction"] for pair in payload["pairs"]} == {
        "w90_to_vacancy", "5x5_to_vacancy",
    }
    assert {pair["target"] for pair in payload["pairs"]} == {
        "Comparison/datasets/graphene_5x5_vacancy",
    }
    for key in ("sizes", "models", "epochs", "seed", "seeds", "early_stopping", "hyperparams"):
        assert payload[key] == normal[key]
    assert payload["performance"]["cross_model_schedule"] == "deeph_then_graph2mat"
    assert payload["performance"]["max_parallel_deeph_training_jobs"] == 5
    assert payload["performance"]["max_parallel_graph2mat_training_jobs"] == 7


def test_vacancy_ui_uses_predict_metrics_without_epochs() -> None:
    html = (REPO_ROOT / "Comparison/ui/index.html").read_text(encoding="utf-8")
    app = (REPO_ROOT / "Comparison/ui/app.js").read_text(encoding="utf-8")
    panel = html[html.index('aria-labelledby="ct-vacancy-heading"'):html.index("</section>", html.index('aria-labelledby="ct-vacancy-heading"'))]
    assert "Cross testing con vacante" in panel
    assert "ct-vacancy-evaluate" in panel
    assert "ct-vacancy-mae-chart" in panel
    assert "ct-vacancy-frobenius-chart" in panel
    assert "ct-vacancy-log" in panel
    assert "ct-vacancy-matrix-run" in panel
    assert "ct-vacancy-matrix-frame" in panel
    assert "ct-epochs" not in panel
    assert html.index('aria-labelledby="ct-vacancy-heading"') > html.index('id="ct-mae-chart"')
    function = app[app.index("async function ctVacancyEvaluate"):app.index("async function ctLaunch")]
    assert 'ctVacancyBody("predict_metrics")' in function
    assert 'request("/api/cross-testing/vacancy/launch"' in function
    assert 'ctLaunch("train"' not in function
    preview = app[app.index("async function ctVacancyPreview"):app.index("async function ctVacancyEvaluate")]
    assert "ctSetMetricsPayload" not in preview
    assert 'request("/api/cross-testing/vacancy/metrics")' in app
    assert 'request("/api/cross-testing/vacancy/matrix-errors")' in app
    assert 'request("/api/cross-testing/vacancy/matrix-error"' in app
    assert '"ct-vacancy-mae-chart"' in app
    assert '"ct-vacancy-frobenius-chart"' in app
    assert 'ctVacancyRenderChart(payload, "relative_frobenius")' in app


def test_normal_cross_testing_plot_keeps_direction_markers() -> None:
    app = (REPO_ROOT / "Comparison/ui/app.js").read_text(encoding="utf-8")
    function = app[app.index("async function ctRenderChart"):app.index("async function ctLoadMetrics")]
    assert 'const markerSymbol = (point)' in function
    assert 'const directionLabel = (point)' in function
    assert 'symbol: points.map(markerSymbol)' in function
    assert 'mode: "lines+markers"' in function


def test_metrics_backfill_frobenius_from_completed_model_launches() -> None:
    records = [{
        "source_id": "source",
        "target_id": "vacancy",
        "payload_id": "source__to__vacancy",
        "source_n_snapshots": 20,
        "model": "graph2mat",
        "seed": 0,
        "h_mae_eV": 0.01,
    }]
    summary = {"permutations": [{
        "payload_id": "source__to__vacancy",
        "model_launches": {"graph2mat": {
            "metrics": {"graph2mat": {"relative_frobenius": 0.125}},
        }},
    }]}
    payload = ui._cross_testing_metrics_payload(records, summary)
    assert payload["curves"][0]["points"][0]["relative_frobenius"] == 0.125


def test_vacancy_runner_has_independent_state_and_output() -> None:
    assert ui.CROSS_TESTING_RUNNER is not ui.CROSS_TESTING_VACANCY_RUNNER
    assert ui.CROSS_TESTING_RUNNER._output_root != ui.CROSS_TESTING_VACANCY_RUNNER._output_root


def test_vacancy_matrix_error_inventory_groups_real_graph2mat_runs(tmp_path: Path) -> None:
    import pipeline_ui as ui

    run_root = tmp_path / "payload_a" / "training" / "graph2mat" / "g2m_deeph_1"
    training = run_root / "graph2mat" / "training"
    training.mkdir(parents=True)
    checkpoint = training / "best.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    (training / "checkpoint_manifest.json").write_text(
        json.dumps({"checkpoint_path": str(checkpoint)}), encoding="utf-8"
    )
    (run_root / "graph2mat" / "pipeline_config.yaml").write_text("training: {}\n", encoding="utf-8")
    test_run = run_root / "graph2mat" / "prediction_structures" / "test" / "sample" / "RUN.fdf"
    test_run.parent.mkdir(parents=True)
    test_run.write_text("SystemLabel vacancy\n", encoding="utf-8")
    (test_run.parent / "ML_prediction.HSX").write_bytes(b"prediction")

    runs = ui._vacancy_matrix_error_inventory(tmp_path)

    assert len(runs) == 1
    assert runs[0]["payload_id"] == "payload_a"
    assert runs[0]["run_name"] == "g2m_deeph_1"
    assert runs[0]["sample_count"] == 1
    assert runs[0]["cached"] is False
