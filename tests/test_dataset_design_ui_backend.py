"""DATASET-DESIGN-W90-001-S8: Dataset Design UI tab backend (pipeline_ui.py).

Covers the estimate/smoke/reuse-existing payload builders directly (no HTTP
server): estimate counts must come from the real sampler generators (never a
guessed formula), active-atom subset selection must work for N>2 without
assuming N==2, and reuse-existing must read the S4/S7 artifacts verbatim.
"""

from __future__ import annotations

import json
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
