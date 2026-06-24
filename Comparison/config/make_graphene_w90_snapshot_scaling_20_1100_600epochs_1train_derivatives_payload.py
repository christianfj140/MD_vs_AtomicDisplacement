#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "Comparison" / "config"

BASE_PAYLOAD = CONFIG_DIR / "graphene_w90_snapshot_scaling_reuse_10_1000_4seeds_payload.json"

RUN_TAG = "graphene_w90_snapshot_scaling_20_1100_600epochs_1train_derivatives"
OUT_PAYLOAD = CONFIG_DIR / f"{RUN_TAG}_payload.json"

SIZES = [20, 40, 60, 80, 100, 200, 300, 500, 800, 1100]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def recipe_for_size(size: int) -> dict[str, Any]:
    n150 = size // 5
    n300 = size - 2 * n150
    n450 = n150
    if n150 + n300 + n450 != size:
        raise RuntimeError(f"Invalid 20/60/20 split for N={size}")

    return {
        "recipe_id": f"graphene_w90_scale_iid{size}",
        "label": f"Graphene W90 IID scale: {size} snapshots",
        "size": size,
        "thermal_regime": "iid_mixed",
        "split_intent": "IID mixed temperature train/validation/test with blocked temporal gap.",
        "blocks": [
            {
                "block_id": f"graphene_w90_scale_iid{size}_T150_{n150}_1",
                "label": f"{n150} snapshots @ 150 K",
                "n_snapshots": n150,
                "temperature_K": 150,
                "seed": 150000 + size,
            },
            {
                "block_id": f"graphene_w90_scale_iid{size}_T300_{n300}_2",
                "label": f"{n300} snapshots @ 300 K",
                "n_snapshots": n300,
                "temperature_K": 300,
                "seed": 300000 + size,
            },
            {
                "block_id": f"graphene_w90_scale_iid{size}_T450_{n450}_3",
                "label": f"{n450} snapshots @ 450 K",
                "n_snapshots": n450,
                "temperature_K": 450,
                "seed": 450000 + size,
            },
        ],
    }


def template_prefix(config_id: str) -> str:
    if "-N" not in config_id:
        raise RuntimeError(f"Unexpected config_id without -N suffix: {config_id}")
    return config_id.rsplit("-N", 1)[0]


def select_600_epoch_templates(base_payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    manual_runs = list(base_payload["training_sweep"]["manual_runs"])

    graph2mat_candidates = [
        run
        for run in manual_runs
        if run.get("model") == "graph2mat"
        and int((run.get("overrides") or {}).get("max_epochs") or 0) == 600
        and int((run.get("overrides") or {}).get("seed_everything") or 0) == 1
    ]
    deeph_candidates = [
        run
        for run in manual_runs
        if run.get("model") == "deeph"
        and int((run.get("overrides") or {}).get("epochs") or 0) == 600
        and int((run.get("overrides") or {}).get("seed") or 0) == 1
    ]

    if not graph2mat_candidates:
        raise RuntimeError("No Graph2Mat 600-epoch seed1 template found in base payload.")
    if not deeph_candidates:
        raise RuntimeError("No DeepH 600-epoch seed1 template found in base payload.")

    return copy.deepcopy(graph2mat_candidates[0]), copy.deepcopy(deeph_candidates[0])


def build_manual_runs(base_payload: dict[str, Any], sizes: list[int]) -> list[dict[str, Any]]:
    g2m_template, dh_template = select_600_epoch_templates(base_payload)

    manual_runs: list[dict[str, Any]] = []
    for size in sizes:
        dataset_id = f"graphene_w90_scale_iid{size}"

        g2m = copy.deepcopy(g2m_template)
        g2m_prefix = template_prefix(str(g2m["config_id"]))
        g2m_id = f"{g2m_prefix}-N{size}"
        g2m["id"] = g2m_id
        g2m["config_id"] = g2m_id
        g2m["dataset_id"] = dataset_id
        g2m["overrides"]["seed_everything"] = 1
        manual_runs.append(g2m)

        dh = copy.deepcopy(dh_template)
        dh_prefix = template_prefix(str(dh["config_id"]))
        dh_id = f"{dh_prefix}-N{size}"
        dh["id"] = dh_id
        dh["config_id"] = dh_id
        dh["dataset_id"] = dataset_id
        dh["overrides"]["seed"] = 1
        manual_runs.append(dh)

    return manual_runs


def set_top_level_model_defaults_from_templates(payload: dict[str, Any], manual_runs: list[dict[str, Any]]) -> None:
    for run in manual_runs:
        if run["model"] == "graph2mat":
            payload["graph2mat_overrides"] = copy.deepcopy(run["overrides"])
            break

    for run in manual_runs:
        if run["model"] == "deeph":
            deeph = copy.deepcopy(payload.get("deeph") or {})
            deeph.update(copy.deepcopy(run["overrides"]))
            payload["deeph"] = deeph
            break


def main() -> None:
    base = load_json(BASE_PAYLOAD)
    payload = copy.deepcopy(base)

    recipes = [recipe_for_size(size) for size in SIZES]
    manual_runs = build_manual_runs(base, SIZES)

    payload["description"] = (
        "Graphene W90 real end-to-end snapshot-scaling payload derived from "
        "graphene_w90_snapshot_scaling_reuse_10_1000_4seeds_payload.json and the W90 dense/full "
        "generation payload logic. It generates datasets for N=20,40,60,80,100,200,300,500,800,1100, "
        "runs one 600-epoch Graph2Mat train and one 600-epoch DeepH train per dataset, then runs "
        "Hamiltonian metrics and finite-difference Hamiltonian derivative metrics."
    )
    payload["source_reference_payloads"] = [
        "Comparison/config/graphene_w90_snapshot_scaling_reuse_10_1000_4seeds_payload.json",
        "Comparison/config/graphene_w90_snapshot_scaling_150_1000_dense_payload.json",
        "Comparison/config/graphene_w90_snapshot_scaling_1100_1300_4seeds_followup_payload.json",
    ]

    payload["run_id"] = RUN_TAG
    payload["dataset_mode"] = "full_strict_pipeline"
    payload["run_mode"] = "full_strict_pipeline"
    payload["dataset_root"] = f"${{REPO_ROOT}}/Comparison/datasets/{RUN_TAG}"
    payload["output_root"] = f"${{REPO_ROOT}}/Comparison/results/{RUN_TAG}"

    # Critical: this is a real generation payload, not reuse.
    payload.pop("reuse_dataset_sweep_from_run_root", None)
    payload.pop("resume_from_run_root", None)
    payload.pop("resume_training_sweep_from_run_root", None)
    payload.pop("resume_training_sweep", None)
    payload["overwrite_datasets"] = False

    payload["dataset_sweep"] = {
        "enabled": True,
        "max_datasets": len(recipes),
        "recipes": recipes,
        "notes": (
            "Real full-strict dataset generation. No reuse_dataset_sweep_from_run_root and no "
            "dataset_sweep.reuse_only. Temperature distribution is 20/60/20 at 150/300/450 K."
        ),
    }

    payload["training_sweep"]["manual_runs"] = manual_runs
    payload["training_sweep"]["max_runs"] = len(manual_runs)
    payload["training_sweep"]["apply_to_datasets"] = ["all"]
    payload["training_sweep"]["error_policy"] = "continue_on_error"
    payload["training_sweep"]["search_policy"] = {
        "strategy": "manual",
        "random_seed": 0,
    }

    set_top_level_model_defaults_from_templates(payload, manual_runs)

    performance = dict(payload.get("performance") or {})
    performance["reuse_validated_siesta_outputs"] = False
    performance["max_parallel_dataset_jobs"] = 1
    performance["max_parallel_derivative_workflows"] = 1
    performance["max_parallel_derivative_reference_jobs"] = 6
    payload["performance"] = performance

    # Top-level workflow keys are intentional: the runner normalizes modular workflow from top-level
    # workflow_mode/stages/derivative.
    payload["workflow_mode"] = "h_then_derivative_full"
    payload["stages"] = {
        "generate_or_validate_dataset": True,
        "freeze_splits": True,
        "train_graph2mat": True,
        "predict_graph2mat": True,
        "train_deeph": True,
        "predict_deeph": True,
        "hamiltonian_metrics": True,
        "build_derivative_stencils": True,
        "validate_derivative_stencils": True,
        "run_derivative_siesta_reference": True,
        "predict_derivative_graph2mat": True,
        "predict_derivative_deeph": True,
        "derivative_metrics_graph2mat": True,
        "derivative_metrics_deeph": True,
        "derivative_gate_check": True,
        "derivative_plots": True,
    }

    payload["derivative"] = {
        "enabled": True,
        # Placeholder required by preflight. The training-sweep derivative handoff overwrites it
        # per dataset group.
        "source_dataset_root": f"${{REPO_ROOT}}/Comparison/datasets/{RUN_TAG}/graphene_w90_scale_iid20",
        "output_root": f"${{REPO_ROOT}}/Comparison/results/{RUN_TAG}/derivative_workflows",
        "method": "central",
        "base_split": "test",
        "base_selection_policy": "adaptive_min_fraction",
        "min_base_snapshots": 20,
        "base_fraction": 0.20,
        "base_selection_seed": 1,
        # Real derivative run: adaptive test bases, two deltas, both physical graphene atoms, all axes.
        # Atom indices are zero-based. Do not include ghost atoms.
        "atoms": ["0-1"],
        "axes": ["x", "y", "z"],
        "delta_ang": [0.005, 0.01],
        "basis_files": f"${{REPO_ROOT}}/Comparison/datasets/{RUN_TAG}/graphene_w90_scale_iid20/material_basis/*.ion.xml",
        "siesta_command": "/home/christian/bin/siesta",
        "reference_workers": 6,
        "deeph_command": "/home/christian/repositorios/DeepH-pack/.venv/bin/deeph-inference",
        "skip_if_exists": True,
        "diagnostic_only": False,
        "require_central": True,
    }

    payload["derivative_metrics"] = {
        "enabled": True,
        "method": "central",
        "require_central": True,
        "diagnostic_only": False,
    }

    payload["notes"] = {
        **(payload.get("notes") or {}),
        "intended_use": "Real end-to-end W90 dataset-size benchmark with derivative metrics.",
        "dataset_generation": "enabled_full_strict_pipeline",
        "dataset_reuse": "disabled",
        "expected_dataset_sizes": SIZES,
        "expected_dataset_count": len(SIZES),
        "expected_training_runs": len(manual_runs),
        "trains_per_dataset": {
            "graph2mat": 1,
            "deeph": 1,
        },
        "training_seed_policy": "single deterministic train per model and dataset; seed=1 retained for reproducibility",
        "epochs_policy": "only 600-epoch templates retained; 750-epoch templates ignored",
        "temperature_distribution": {
            "150K": "20%",
            "300K": "60%",
            "450K": "20%",
        },
        "derivative_policy": {
            "method": "central finite difference",
            "delta_ang": [0.005, 0.01],
            "base_split": "test",
            "base_selection_policy": "adaptive_min_fraction",
            "min_base_snapshots": 20,
            "base_fraction": 0.20,
            "base_selection_seed": 1,
            "base_selection_formula": "K = min(n_test, max(20, ceil(0.20*n_test))); if n_test < 20, use all available test snapshots.",
            "reference_workers": 6,
            "reference_workers_note": (
                "reference_workers controls how many derivative SIESTA processes run in parallel "
                "inside each derivative workflow. Initial recommendation: 6 on a 24-CPU machine; "
                "raise to 8 only if RAM and IO wait remain acceptable."
            ),
            "expected_selected_base_snapshots_by_dataset_size": {
                "N20": 2,
                "N40": 4,
                "N60": 6,
                "N80": 8,
                "N100": 10,
                "N200": 20,
                "N300": 20,
                "N500": 20,
                "N800": 20,
                "N1100": 22,
            },
            "atoms": "0-1 zero-based physical graphene atoms; ghost atoms intentionally excluded",
            "axes": ["x", "y", "z"],
        },
        "cost_warning": (
            "This is intentionally not a smoke test, but derivative bases are capped adaptively: "
            "K = min(n_test, max(20, ceil(0.20*n_test))); if n_test < 20, the full test split is used."
        ),
    }

    # Basic validation.
    ids = [str(run["config_id"]) for run in manual_runs]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate manual run IDs detected.")
    if len(manual_runs) != 2 * len(SIZES):
        raise RuntimeError(f"Expected {2 * len(SIZES)} manual runs, got {len(manual_runs)}.")
    for run in manual_runs:
        overrides = run.get("overrides") or {}
        if run["model"] == "graph2mat" and int(overrides.get("max_epochs") or 0) != 600:
            raise RuntimeError(f"Non-600 Graph2Mat run leaked into plan: {run['config_id']}")
        if run["model"] == "deeph" and int(overrides.get("epochs") or 0) != 600:
            raise RuntimeError(f"Non-600 DeepH run leaked into plan: {run['config_id']}")

    OUT_PAYLOAD.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Wrote payload: {OUT_PAYLOAD}")
    print(f"run_id: {payload['run_id']}")
    print(f"dataset_root: {payload['dataset_root']}")
    print(f"output_root: {payload['output_root']}")
    print(f"sizes: {SIZES}")
    print(f"datasets: {len(SIZES)}")
    print(f"training runs: {len(manual_runs)}")
    print("manual runs:")
    for run in manual_runs:
        print(f"  - {run['config_id']} [{run['model']}] {run['dataset_id']}")


if __name__ == "__main__":
    main()
