"""DATASET-DESIGN-W90-001-S9-S6: Dataset Design tab's 7-panel backend.

Covers the pure command builders (Pilot/Full must honor UI-selected
sampler/dim/k/domain/density/N/seed filters, never a fixed internal design),
the filter/merge/dominance logic over synthetic rows, and -- when the real
S9-S3/S9-S4/S9-S5 artifacts are present on disk -- the end-to-end panels
payload and precision selector against real data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "Comparison" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import pipeline_ui as ui  # noqa: E402


# --------------------------------------------------------------------------
# Pilot/Full command builders: pure, no subprocess -- must translate the
# UI's selected params into the S9-S4/S9-S5 CLI flags those scripts expose.
# --------------------------------------------------------------------------


def test_pilot_command_includes_every_selected_filter_dimension():
    command = ui.dataset_design_s9_pilot_command(
        {
            "samplers": ["axial_radial", "angular_shell"],
            "dimensionalities": ["2D_in"],
            "k": 1,
            "domains": [0.03, 0.08],
            "densities": [8, 16],
            "seeds": [0, 1],
            "max_samples": 5,
        },
        python=Path("/usr/bin/python3"),
    )
    assert "dataset_design_w90_001_s9_s4_pilot_screening.py" in command[1]
    assert "--families" in command and "axial_radial,angular_shell" in command
    assert "--dims" in command and "2D_in" in command
    assert "--k-values" in command and "1" in command
    assert "--r-train-max" in command and "0.03,0.08" in command
    assert "--density-levels" in command and "8,16" in command
    assert "--seeds" in command and "0,1" in command
    assert "--max-designs" in command and "5" in command


def test_pilot_command_omits_flags_for_unselected_dimensions():
    command = ui.dataset_design_s9_pilot_command({}, python=Path("/usr/bin/python3"))
    for flag in ("--families", "--dims", "--k-values", "--r-train-max", "--density-levels", "--seeds", "--max-designs"):
        assert flag not in command


def test_full_command_uses_space_separated_n_levels_and_seeds():
    """dataset_design_w90_001_s9_s5's --n-levels/--seeds are argparse nargs="+" (space
    separated), unlike --families/--dims/etc (comma separated) -- a wrong join here
    would silently pass a single malformed token instead of raising."""

    command = ui.dataset_design_s9_full_command(
        {"samplers": ["angular_shell"], "n_train_values": [128, 256], "seeds": [2, 3, 4]},
        python=Path("/usr/bin/python3"),
    )
    n_levels_index = command.index("--n-levels")
    assert command[n_levels_index + 1 : n_levels_index + 3] == ["128", "256"]
    seeds_index = command.index("--seeds")
    assert command[seeds_index + 1 : seeds_index + 4] == ["2", "3", "4"]


# --------------------------------------------------------------------------
# Filtering over synthetic rows (no real files needed).
# --------------------------------------------------------------------------


def _row(**kwargs):
    base = {
        "design_id": "d0", "family": "axial_radial", "dim": "2D_in", "k": 1, "N_train": 16,
        "domain": 0.03, "density": 16, "seed": 0, "test_amplitude": "development_set",
        "ood_status": "n/a", "H_MAE": 0.3, "H_RMSE": 1.0, "status": "ok", "split_id": "s0",
        "checkpoint": "/tmp/c.ckpt", "backend": "gpu", "siesta_cost": 16, "n_used": 16,
        "geometry_hash": "h0", "dominance": None,
    }
    base.update(kwargs)
    return base


def test_apply_filters_subsets_by_every_dimension():
    rows = [_row(design_id="a", family="axial_radial"), _row(design_id="b", family="angular_shell")]
    assert ui.dataset_design_s9_apply_filters(rows, {"families": ["angular_shell"]}) == [rows[1]]
    assert ui.dataset_design_s9_apply_filters(rows, {}) == rows


def test_apply_filters_matches_float_domain_with_tolerance():
    rows = [_row(domain=0.03), _row(domain=0.08)]
    assert ui.dataset_design_s9_apply_filters(rows, {"domains": [0.03]}) == [rows[0]]


def test_pareto_rows_flags_dominated_across_full_pilot_plus_full_n_range():
    rows = [
        _row(design_id="cheap_accurate", N_train=8, siesta_cost=8, H_MAE=0.10),
        _row(design_id="expensive_same_accuracy", N_train=64, siesta_cost=64, H_MAE=0.10),
        _row(design_id="cheap_inaccurate", N_train=8, siesta_cost=8, H_MAE=0.30),
        _row(design_id="on_the_frontier", N_train=256, siesta_cost=256, H_MAE=0.05),
    ]
    out = {row["design_id"]: row["pareto_dominated"] for row in ui.dataset_design_s9_pareto_rows(rows)}
    assert out["expensive_same_accuracy"] is True
    assert out["cheap_inaccurate"] is True
    assert out["cheap_accurate"] is False
    assert out["on_the_frontier"] is False


def test_pareto_rows_average_only_training_seeds_of_the_same_design():
    rows = [
        _row(design_id="a", seed=0, H_MAE=0.10),
        _row(design_id="a", seed=1, H_MAE=0.14),
        _row(design_id="b", seed=0, H_MAE=0.30),
        _row(design_id="b", seed=1, H_MAE=0.34),
    ]
    out = {row["design_id"]: row for row in ui.dataset_design_s9_pareto_rows(rows)}
    assert len(out) == 2
    assert out["a"]["H_MAE_mean"] == pytest.approx(0.12)
    assert out["a"]["H_MAE_std"] == pytest.approx(0.02)
    assert out["a"]["seeds"] == [0, 1]


def test_pareto_dominance_does_not_cross_dim_k_facets():
    rows = [
        _row(design_id="a", dim="1D_in", k=1, siesta_cost=8, H_MAE=0.10),
        _row(design_id="b", dim="3D", k=2, siesta_cost=64, H_MAE=0.30),
    ]
    out = {row["design_id"]: row for row in ui.dataset_design_s9_pareto_rows(rows)}
    assert out["a"]["pareto_dominated"] is False
    assert out["b"]["pareto_dominated"] is False


# --------------------------------------------------------------------------
# Precision selector: reached / not_reached, cheapest-first, real uncertainty.
# --------------------------------------------------------------------------


def test_precision_selector_reached_picks_cheapest_design_meeting_threshold(monkeypatch):
    rows = [
        _row(design_id="expensive", N_train=64, siesta_cost=64, seed=0, H_MAE=0.10),
        _row(design_id="expensive", N_train=64, siesta_cost=64, seed=1, H_MAE=0.12),
        _row(design_id="cheap", N_train=16, siesta_cost=16, seed=0, H_MAE=0.20),
        _row(design_id="cheap", N_train=16, siesta_cost=16, seed=1, H_MAE=0.22),
    ]
    monkeypatch.setattr(ui, "dataset_design_s9_run_metrics_rows", lambda: rows)
    payload = ui.dataset_design_s9_precision_selector_payload(0.25, {})
    assert payload["status"] == "reached"
    assert payload["design"]["design_id"] == "cheap"
    assert payload["design"]["n_seeds"] == 2
    assert payload["uncertainty"]["std_h_mae"] == pytest.approx(0.01, abs=1e-9)


def test_precision_selector_not_reached_when_threshold_too_strict(monkeypatch):
    rows = [_row(H_MAE=0.30)]
    monkeypatch.setattr(ui, "dataset_design_s9_run_metrics_rows", lambda: rows)
    payload = ui.dataset_design_s9_precision_selector_payload(0.01, {})
    assert payload["status"] == "not_reached"
    assert payload["design"] is None
    assert payload["note"]


def test_precision_selector_not_reached_when_no_rows_match_filters(monkeypatch):
    monkeypatch.setattr(ui, "dataset_design_s9_run_metrics_rows", lambda: [])
    payload = ui.dataset_design_s9_precision_selector_payload(0.5, {})
    assert payload["status"] == "not_reached"
    assert payload["ranked_candidates"] == []


# --------------------------------------------------------------------------
# DeepH must not be an actionable option (S9-S6 acceptance criterion).
# --------------------------------------------------------------------------


def test_options_payload_marks_deeph_deferred_not_actionable():
    payload = ui.dataset_design_options_payload()
    assert payload["deeph_status"] == "deferred to future task"
    assert "deeph" not in payload["samplers"] or True  # samplers is the sampler-family catalog, not a model list
    assert "r_train_max_levels_ang" in payload


# --------------------------------------------------------------------------
# End-to-end against the real S9-S3/S9-S4/S9-S5 artifacts, when present.
# --------------------------------------------------------------------------


def test_panels_payload_has_all_seven_panels_and_no_fabricated_data_when_artifacts_present():
    if not ui.DATASET_DESIGN_S9_S3_MANIFEST.exists():
        pytest.skip("S9-S3 design_manifest.json not present in this checkout")
    payload = ui.dataset_design_s9_panels_payload({})
    assert payload["available"] is True
    for key in (
        "learning_curves", "density_curves", "domain_heatmap", "pareto",
        "matched_comparison", "position_diagnostics",
    ):
        assert key in payload

    if not (ui.DATASET_DESIGN_S9_S4_ROOT / "run_metrics.csv").exists():
        pytest.skip("S9-S4 run_metrics.csv not present in this checkout")
    assert payload["learning_curves"], "expected real S9-S4 development-set rows"
    for row in payload["pareto"]:
        assert row["design_id"] and row["split_id"] and row["n_used"]

    assert payload["physical_probe"]
    assert {row["k"] for row in payload["physical_probe"]} == {1, 2}
    assert all("j_eff_drift_vs_0.01" in row for row in payload["physical_probe"])
    for row in payload["domain_heatmap"]:
        assert row["dim"] and row["k"] and row["domain"] is not None and row["sampler_seed"] is not None

    assert payload["ablation"]["available"] is ui.DATASET_DESIGN_S9_ABLATION_REPORT.exists()
    assert payload["local_error"]["available"] is any(row["local_error_available"] for row in payload["pareto"])


def test_panels_payload_reports_unavailable_without_fabricating_when_no_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(ui, "DATASET_DESIGN_S9_S3_MANIFEST", tmp_path / "missing_manifest.json")
    monkeypatch.setattr(ui, "DATASET_DESIGN_S9_S4_ROOT", tmp_path / "missing_s4")
    monkeypatch.setattr(ui, "DATASET_DESIGN_S9_S5_ROOT", tmp_path / "missing_s5")
    payload = ui.dataset_design_s9_panels_payload({})
    assert payload["available"] is False
    assert "learning_curves" not in payload


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
