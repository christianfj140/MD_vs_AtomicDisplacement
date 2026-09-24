"""DATASET-DESIGN-W90-001-S8: Dataset Design UI tab backend (pipeline_ui.py).

Covers the estimate/smoke/reuse-existing payload builders directly (no HTTP
server): estimate counts must come from the real sampler generators (never a
guessed formula), active-atom subset selection must work for N>2 without
assuming N==2, and reuse-existing must read the S4/S7 artifacts verbatim.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "Comparison" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import pipeline_ui  # noqa: E402
import w90_displacement_sampler_family as dataset_design_sampler  # noqa: E402


def test_options_payload_has_no_second_hardcoded_sampler_list(monkeypatch):
    """Fase 4, item 1: pipeline_ui must not keep its own copy of the sampler
    catalog -- editing the single canonical source (and nothing else) must be
    enough for the UI-facing payload to change."""
    monkeypatch.setattr(dataset_design_sampler, "SAMPLER_IDS", ("axial_radial", "brand_new_sampler"))
    payload = pipeline_ui.dataset_design_options_payload()
    assert payload["samplers"] == ["axial_radial", "brand_new_sampler"]
    assert "angular_shell" not in payload["samplers"]


def test_options_payload_dims_and_amplitudes_come_from_canonical_source():
    payload = pipeline_ui.dataset_design_options_payload()
    assert payload["dimensionalities"] == list(dataset_design_sampler.DIMENSIONALITIES)
    assert payload["amplitudes_ang"] == list(dataset_design_sampler.ALL_AMPLITUDES_ANG)
    assert payload["samplers"] == list(dataset_design_sampler.SAMPLER_IDS)


def test_estimate_counts_match_real_generators():
    payload = pipeline_ui.dataset_design_estimate_payload({
        "material": "w90",
        "samplers": ["axial_radial"],
        "dimensionalities": ["2D_in"],
        "amplitudes_ang": [0.01, 0.03, 0.05],
        "k": 1,
        "seeds": [0],
        "n_train_values": [16],
        "models": ["graph2mat"],
    })
    combo = payload["combinations"][0]
    # axial_radial emits len(radii) * len(axial_directions(dim)) configs; 2D_in has 2 axes -> 4 directions.
    assert combo["n_configs_per_seed"] == 3 * 4
    assert combo["n_configs_total"] == combo["n_configs_per_seed"]  # deterministic sampler: no seed multiplier
    assert payload["dataset_points_estimate"] == combo["n_configs_total"]
    assert payload["training_runs_estimate"] == 1
    assert payload["explosion_warning"] is None


def test_estimate_seed_dependent_sampler_multiplies_by_seeds():
    payload = pipeline_ui.dataset_design_estimate_payload({
        "material": "w90",
        "samplers": ["sobol_sparse"],
        "dimensionalities": ["2D_in"],
        "amplitudes_ang": [0.03],
        "k": 1,
        "seeds": [0, 1, 2],
        "n_train_values": [16],
        "models": ["graph2mat"],
    })
    combo = payload["combinations"][0]
    assert combo["n_configs_per_seed"] == 16
    assert combo["seed_multiplier"] == 3
    assert combo["n_configs_total"] == 48


def test_estimate_flags_combinatorial_explosion():
    payload = pipeline_ui.dataset_design_estimate_payload({
        "material": "w90",
        "samplers": ["sobol_sparse"],
        "dimensionalities": ["2D_in"],
        "amplitudes_ang": [0.03],
        "k": 1,
        "seeds": list(range(10)),
        "n_train_values": [300],
        "models": ["graph2mat"],
    })
    assert payload["dataset_points_estimate"] > pipeline_ui.DATASET_DESIGN_EXPLOSION_DATASET_POINTS
    assert payload["explosion_warning"] is not None


def test_active_atom_selection_works_for_n_greater_than_2():
    payload = pipeline_ui.dataset_design_estimate_payload({
        "material": "5x5",
        "samplers": ["axial_radial"],
        "dimensionalities": ["2D_in"],
        "k": 2,
        "center_index": 10,
        "pair_mode": "bonded",
    })
    preview = payload["active_atom_preview"]
    assert preview["n_atoms_total"] == 50
    assert preview["k"] == 2
    assert len(preview["active_atom_indices"]) == 2


def test_smoke_mode_generates_real_rows_without_siesta():
    payload = pipeline_ui.dataset_design_run_smoke_payload({
        "material": "w90",
        "samplers": ["axial_radial"],
        "dimensionalities": ["2D_in"],
        "amplitudes_ang": [0.01, 0.03],
        "k": 1,
        "seeds": [0],
        "n_train_values": [8],
        "models": ["graph2mat"],
    })
    assert payload["mode"] == "smoke"
    assert payload["results_table"], "smoke mode must produce at least one row"
    row = payload["results_table"][0]
    assert row["family"] == "axial_radial"
    assert row["dimensionality"] == "2D_in"


def test_local_pair_modes_requires_k_equals_2():
    payload = pipeline_ui.dataset_design_estimate_payload({
        "material": "w90",
        "samplers": ["local_pair_modes"],
        "dimensionalities": ["2D_in"],
        "k": 1,
    })
    assert payload["combinations"] == []
    assert payload["skipped_combinations"][0]["sampler"] == "local_pair_modes"


def test_reuse_existing_reads_s4_s7_artifacts_verbatim():
    if not (pipeline_ui.DATASET_DESIGN_S4_ROOT / "learning_curves.csv").exists():
        pytest.skip("S4 artifacts not present in this checkout")
    payload = pipeline_ui.dataset_design_results_payload()
    assert payload["available"] is True
    assert payload["results_table"]
    assert payload["figures"]
    assert "conclusions_markdown" in payload


def test_followup_real_figures_expose_all_methods_and_model_ids():
    root = pipeline_ui.DATASET_DESIGN_CURVES_ROOT
    if not (root / "label_budget_6x6/model_results.csv").exists():
        pytest.skip("follow-up artifacts not present in this checkout")
    figures = {figure["name"]: figure for figure in pipeline_ui.dataset_design_followup_payload()["figures"]}
    assert "6x6_training_budget" in figures
    assert "label_budget_h_mae_synthetic_test" in figures
    assert "latin_hypercube" in str(figures["campaign_learning_6x6"])
    assert figures["label_budget_h_mae_synthetic_test"]["data"][0]["customdata"][0][0].startswith("N4__V8")
    assert figures["label_budget_h_mae_synthetic_test"]["data"][1]["customdata"][0][0].startswith("md__N4__V8")
    assert figures["label_budget_h_mae"]["data"][0]["customdata"][0][0].startswith("synthetic__N4__V8")
    assert figures["label_budget_test_reliability"]["data"][0]["customdata"][0][0].startswith("N4__V8")
    frontier = figures["label_budget_frontier"]
    assert "fuera de los márgenes estrictos" in frontier["caption"]
    assert not any("no pasa" in trace["name"] for trace in frontier["data"])
    assert {"Diseñado · equivalente", "Diseñado · fuera del umbral", "Diseñado · Ntest/ordenación no robustos"} <= {
        trace["name"] for trace in frontier["data"]}
    decisions = figures["label_budget_decisions"]
    assert "Equivalencia" in decisions["layout"]["title"]
    assert "Clasificación" in decisions["data"][0]["header"]["values"]
    heatmaps = []
    for name in ("method_amplitude_crosstest_H_MAE_meV", "method_amplitude_crosstest_rel_Frob"):
        boxes = figures[name]["layout"]["shapes"]
        assert len(boxes) == 12
        assert all(box["x0"] == box["y0"] and box["x1"] == box["y1"] for box in boxes)
        heatmap = figures[name]["data"][0]
        heatmaps.append(heatmap)
        assert heatmap["colorscale"] == "Inferno"
        views = figures[name]["views"]
        assert [view["label"] for view in views] == ["Heatmap", "Superficie 3D"]
        assert views[1]["data"][0]["type"] == "surface"
        assert views[1]["data"][0]["z"] == heatmap["text"]
        assert views[1]["data"][0]["surfacecolor"] == heatmap["z"]
        assert views[1]["data"][1]["type"] == "scatter3d"
        assert views[1]["data"][1]["line"]["color"] == "#111827"
    assert heatmaps[0]["z"][0][0] == pytest.approx(math.log10(heatmaps[0]["text"][0][0]))
    assert heatmaps[1]["z"][0][0] == pytest.approx(math.log10(2.5 * heatmaps[1]["text"][0][0]))
    assert (heatmaps[0]["zmin"], heatmaps[0]["zmax"]) == (heatmaps[1]["zmin"], heatmaps[1]["zmax"])
    cross_table = figures["method_amplitude_crosstest_table"]["data"][0]
    assert {"H seed 0", "H seed 1", "Frob. seed 0 (%)", "Frob. seed 1 (%)"} <= set(cross_table["header"]["values"])
    assert {trace["name"] for trace in figures["md6x6_paired_difference"]["data"]} == {
        "sobre MD-Test48", "sobre Test48 sintético"}
    for name in ("w90_training_budget", "6x6_training_budget"):
        budget = figures[name]
        assert budget["layout"]["xaxis"]["tickvals"] == [4, 8, 16, 32, 64]
        assert budget["layout"]["xaxis2"]["tickvals"] == [4, 8, 16, 32, 64]
        assert budget["layout"]["yaxis2"]["overlaying"] == "y"
        assert budget["layout"]["yaxis4"]["overlaying"] == "y3"
        assert {trace["yaxis"] for trace in budget["data"]} == {"y", "y2", "y3", "y4"}
    badge = next(item for item in figures["campaign_learning_6x6"]["layout"]["annotations"]
                 if item["text"] == "<b>OBSOLETA</b>")
    assert badge["font"] == {"size": 26, "color": "#000000"}
    assert badge["bordercolor"] == "#000000" and badge["bgcolor"] == "#fecaca"
    for current in ("w90_training_budget", "6x6_training_budget", "label_budget_h_mae",
                    "label_budget_test_reliability"):
        assert "OBSOLETA" not in str(figures[current]["layout"].get("annotations", []))


def test_followup_payload_exposes_available_tables_figures_and_report(tmp_path, monkeypatch):
    root = tmp_path / "curves"
    summary = root / "w90_coverage_physics" / "summary.csv"
    summary.parent.mkdir(parents=True)
    summary.write_text(
        "recipe,domain,n_seeds,H_MAE_meV_mean,H_MAE_meV_sd,rel_Frob_mean,rel_Frob_sd,band_rmse_meV_mean,band_rmse_meV_sd,dos_rel_L1_mean,dos_rel_L1_sd\n"
        "three_d,all,5,18.6,3.9,0.008,0.001,34.1,8.9,0.051,0.014\n",
        encoding="utf-8",
    )
    (root / "final_table.csv").write_text("Sistema,Dataset,N\nw90,designed,64\n", encoding="utf-8")
    figure = root / "figure.png"
    figure.write_bytes(b"png")
    report = tmp_path / "report.md"
    report.write_text("# Informe final", encoding="utf-8")
    budget_report = tmp_path / "budget-report.md"
    budget_report.write_text("# Presupuesto 6x6", encoding="utf-8")
    monkeypatch.setattr(pipeline_ui, "DATASET_DESIGN_CURVES_ROOT", root)
    monkeypatch.setattr(pipeline_ui, "DATASET_DESIGN_FOLLOWUP_REPORT", report)
    monkeypatch.setattr(pipeline_ui, "DATASET_DESIGN_LABEL_BUDGET_REPORT", budget_report)
    monkeypatch.setattr(pipeline_ui, "DATASET_DESIGN_MD_BASELINE_ROOT", tmp_path / "missing-md")
    monkeypatch.setattr(pipeline_ui, "DATASET_DESIGN_MD_BASELINE_REPORT", tmp_path / "missing-md-report.md")
    monkeypatch.setattr(pipeline_ui, "DATASET_DESIGN_METHOD_CROSSTEST_REPORT", tmp_path / "missing-cross-report.md")
    monkeypatch.setattr(pipeline_ui, "DATASET_DESIGN_ACTIVE_K_REPORT", tmp_path / "missing-k-report.md")
    monkeypatch.setattr(pipeline_ui, "DATASET_DESIGN_FOLLOWUP_FIGURES",
                        {"result": (figure, "Resultado", "Descripción")})

    payload = pipeline_ui.dataset_design_followup_payload()
    assert payload["available"] is True
    assert payload["campaign_table"] == [{"Sistema": "w90", "Dataset": "designed", "N": "64"}]
    assert payload["coverage_table"][0]["H-MAE (meV)"] == "18.60 ± 3.90"
    assert payload["coverage_table"][0]["Frobenius relativo (%)"] == "0.80 ± 0.10"
    assert payload["coverage_table"][0]["DOS L1 (%)"] == "5.10 ± 1.40"
    assert payload["training_table"] == []
    assert payload["figures"][0]["name"] == "w90_dataset_comparison"
    assert payload["figures"][0]["kind"] == "plotly"
    assert {trace["yaxis"] for trace in payload["figures"][0]["data"][:2]} == {"y", "y4"}
    assert payload["figures"][0]["layout"]["yaxis"]["tickfont"]["color"] == "#d62728"
    assert payload["figures"][0]["layout"]["yaxis4"]["tickfont"]["color"] == "#1f77b4"
    assert payload["report_markdown"] == "# Informe final\n\n---\n\n# Presupuesto 6x6"


def test_followup_payload_shows_6x6_loss_diagnostic(tmp_path, monkeypatch):
    root = tmp_path / "curves"
    for seed, value in enumerate((1.96, 1.98, 2.00, 2.02, 2.04)):
        path = root / "runs/6x6" / f"loss-{seed}" / "result.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({
            "stage": "loss_long",
            "config_patch": {"model": {"loss": "graph2mat.metrics.elementwise_mse"}},
            "val_metrics": {"H_MAE_meV": value},
        }), encoding="utf-8")
    monkeypatch.setattr(pipeline_ui, "DATASET_DESIGN_CURVES_ROOT", root)
    monkeypatch.setattr(pipeline_ui, "DATASET_DESIGN_FOLLOWUP_FIGURES", {})

    payload = pipeline_ui.dataset_design_followup_payload()

    assert payload["training_table"][0]["H-MAE validación (meV)"] == "2.00 ± 0.03"
    assert payload["training_table"][0]["Semillas"] == "5/5"


def test_followup_figures_include_every_available_method(tmp_path, monkeypatch):
    root = tmp_path / "curves"
    root.mkdir()
    (root / "final_configs.csv").write_text(
        "system,recipe_id,family,N,n_seeds,H_MAE_meV,seed_sd,rel_Frob,band_rmse_meV,dos_rel_L1\n"
        "6x6,sobol_sparse__3D__R0.08__d64,sobol_sparse,4,3,8,1,0.03,60,0.1\n"
        "6x6,sobol_sparse__3D__R0.08__d64,sobol_sparse,64,3,6,1,0.02,50,0.09\n"
        "6x6,random_cartesian__1D_in__R0.12__d64,random_cartesian,64,3,7,1,0.025,55,0.095\n"
        "6x6,latin_hypercube__2D_in__R0.08__d64,latin_hypercube,64,3,9,1,0.04,70,0.12\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(pipeline_ui, "DATASET_DESIGN_CURVES_ROOT", root)

    comparison = pipeline_ui._dd_metric_panels(
        root / "missing.csv", "comparison", "comparison", "caption", "comparison", "6x6")
    curve = pipeline_ui._dd_metric_panels(
        root / "missing.csv", "curve", "curve", "caption", "curve", "6x6")

    assert comparison is not None
    assert set(comparison["layout"]["xaxis"]["categoryarray"]) == {
        "Sobol 3D R=0.08 Å", "random 1D in R=0.12 Å", "LHS 2D in R=0.08 Å",
    }
    assert curve is not None
    assert {trace["name"] for trace in curve["data"]} >= {
        "Sobol 3D R=0.08 Å", "random 1D in R=0.12 Å", "LHS 2D in R=0.08 Å",
    }
    assert curve["data"][0]["x"] == [4, 64]


def test_6x6_validation_diagnostics_include_curve_and_lhs_methods():
    results = [
        {"stage": "curve", "recipe_id": "random_cartesian__1D_in__R0.12__d64", "N": 64,
         "val_metrics": {"H_MAE_meV": 2.0, "rel_Frob": 0.02}},
        {"stage": "lhs", "recipe_id": "latin_hypercube__2D_in__R0.08__d64", "N": 64,
         "val_metrics": {"H_MAE_meV": 3.0, "rel_Frob": 0.03}},
    ]

    _, plot = pipeline_ui._dataset_design_6x6_training(results)

    assert {row["label"] for row in plot} == {
        "random 1D in R=0.12 Å · block_type_mae · campaña original",
        "LHS 2D in R=0.08 Å · block_type_mae · campaña original",
    }
