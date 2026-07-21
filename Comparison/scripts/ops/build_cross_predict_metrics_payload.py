#!/usr/bin/env python3
"""Build the queued cross-structure predict_metrics payload."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
SIZES = [20, 30, 50, 60, 80, 90, 100, 150, 200, 300, 400, 500, 600]
W90_20 = {20, 60, 80, 100, 200, 300, 500}
W90_12 = {30, 50, 90, 150, 400, 600}


def repo(path: str) -> str:
    return path


def w90_dataset(size: int) -> str:
    root = (
        "graphene_w90_snapshot_scaling_20_1100_600epochs_1train_derivatives"
        if size in W90_20
        else "graphene_w90_snapshot_scaling_12_1300_600epochs_1train_derivatives"
    )
    return repo(f"Comparison/datasets/{root}/graphene_w90_scale_iid{size}")


def x5_dataset(size: int) -> str:
    roots = {
        20: "graphene_5x5_snapshot_scaling_20_50_80",
        30: "graphene_5x5_snapshot_scaling_complete_paper_ready_missing",
        50: "graphene_5x5_snapshot_scaling_20_50_80",
        60: "graphene_5x5_snapshot_scaling_complete_paper_ready_missing",
        80: "graphene_5x5_snapshot_scaling_20_50_80",
        90: "graphene_5x5_snapshot_scaling_complete_paper_ready_missing",
        100: "graphene_5x5_snapshot_scaling_100_500_mixing",
        150: "graphene_5x5_snapshot_scaling_100_500_mixing",
        200: "graphene_5x5_snapshot_scaling_100_500_mixing",
        300: "graphene_5x5_snapshot_scaling_100_500_mixing",
        400: "graphene_5x5_snapshot_scaling_100_500_mixing",
        500: "graphene_5x5_snapshot_scaling_100_500_mixing",
        600: "graphene_5x5_snapshot_scaling_complete_paper_ready_missing",
    }
    return repo(f"Comparison/datasets/{roots[size]}/graphene_5x5_scale_iid{size}")


def w90_archive(size: int) -> tuple[str, str]:
    name = (
        "graphene_w90_snapshot_scaling_20_1100_600epochs_1train_derivatives"
        if size in W90_20
        else "graphene_w90_snapshot_scaling_12_1300_600epochs_1train_derivatives"
    )
    archive = f"Comparison/results_archived/results/{name}.tar.zst"
    root = f"{name}/{name}/sweep"
    return archive, root


def x5_training_dir(size: int) -> str:
    return repo(
        "Comparison/results/graphene_5x5_snapshot_scaling_complete_paper_ready/"
        "graphene_5x5_snapshot_scaling_complete_paper_ready/sweep/graph2mat/"
        f"graphene_5x5_scale_iid{size}/G2M-T1000-03-anchor-G5x5-N{size}/graph2mat/training"
    )


def x5_deeph_dir(size: int) -> str:
    return repo(
        "Comparison/results/graphene_5x5_snapshot_scaling_complete_paper_ready/"
        "graphene_5x5_snapshot_scaling_complete_paper_ready/sweep/deeph/"
        f"graphene_5x5_scale_iid{size}/DH-T1000-04-anchor-G5x5-N{size}/deeph/train"
    )


def existing_artifacts() -> dict[str, Any]:
    artifacts: dict[str, Any] = {}
    for size in SIZES:
        archive, root = w90_archive(size)
        staged = f"Comparison/results/ml_vs_siesta_cross_structure_sweep/predict_metrics_artifacts/w90/N{size}"
        artifacts[f"graphene_w90_scale_iid{size}"] = {
            "graph2mat_training_dir": f"{staged}/graph2mat/training",
            "deeph_save_dir": f"{staged}/deeph/train",
            "graph2mat_archive": archive,
            "graph2mat_archive_prefix": (
                f"{root}/graph2mat/graphene_w90_scale_iid{size}/"
                f"G2M-T1000-03-anchor-N{size}/graph2mat/training"
            ),
            "deeph_archive": archive,
            "deeph_archive_prefix": (
                f"{root}/deeph/graphene_w90_scale_iid{size}/"
                f"DH-T1000-04-anchor-N{size}/deeph/train"
            ),
        }
        artifacts[f"graphene_5x5_scale_iid{size}"] = {
            "graph2mat_training_dir": x5_training_dir(size),
            "deeph_save_dir": x5_deeph_dir(size),
        }
    return artifacts


def build(base: dict[str, Any]) -> dict[str, Any]:
    pairs = []
    for size in SIZES:
        pairs.append({"size": size, "direction": "w90_to_5x5", "source": w90_dataset(size), "target": x5_dataset(size)})
        pairs.append({"size": size, "direction": "5x5_to_w90", "source": x5_dataset(size), "target": w90_dataset(size)})
    return {
        **base,
        "schema": "g2m_deeph_cross_structure_predict_metrics_payload_v1",
        "description": (
            "Predict+metrics-only cross testing for W90<->5x5, same size grid and "
            "hparams/performance inherited from the paper-ready cross payload."
        ),
        "action": "predict_metrics",
        "sources": {},
        "targets": {},
        "pairs": pairs,
        "existing_artifacts": existing_artifacts(),
        "notes": {
            **(base.get("notes") or {}),
            "predict_metrics_only": "Skips training and fails closed if a source checkpoint/model dir is missing.",
            "directions": "Includes w90->5x5 and 5x5->w90. Self baselines are copied separately into the graph.",
        },
    }


def build_vacancy(base: dict[str, Any], target: str) -> dict[str, Any]:
    pairs = []
    for size in SIZES:
        pairs.append({"size": size, "direction": "w90_to_vacancy", "source": w90_dataset(size), "target": target})
        pairs.append({"size": size, "direction": "5x5_to_vacancy", "source": x5_dataset(size), "target": target})
    return {
        **base,
        "schema": "g2m_deeph_cross_structure_predict_metrics_payload_v1",
        "description": "Predict+metrics-only transfer from existing W90/5x5 checkpoints to graphene 5x5 monovacancy.",
        "action": "predict_metrics",
        "sources": {},
        "targets": {},
        "pairs": pairs,
        "existing_artifacts": existing_artifacts(),
        "notes": {
            **(base.get("notes") or {}),
            "predict_metrics_only": "Existing checkpoints are staged; Graph2Mat and DeepH training are skipped.",
            "target": "Unrelaxed, non-spin-polarized 49-carbon graphene 5x5 monovacancy test dataset.",
        },
    }


def _pin_deeph_threads(payload: dict[str, Any], num_threads: int = 2) -> dict[str, Any]:
    # The base paper-ready payload omits deeph.num_threads, so it defaults to -1
    # (every process grabs all cores). Under parallel training that oversubscribes
    # the CPU and slows the graph builder ~30-40x, starving the GPU. Pin it to a
    # small value like the fast snapshot-scaling sweep did.
    hyperparams = dict(payload.get("hyperparams") or {})
    deeph = dict(hyperparams.get("deeph") or {})
    deeph.setdefault("num_threads", num_threads)
    hyperparams["deeph"] = deeph
    payload["hyperparams"] = hyperparams
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base",
        type=Path,
        default=REPO_ROOT / "Comparison/config/graphene_w90_to_5x5_cross_structure_paper_ready_payload.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "Comparison/config/graphene_w90_5x5_cross_structure_predict_metrics_payload.json",
    )
    parser.add_argument(
        "--vacancy-output",
        type=Path,
        default=REPO_ROOT / "Comparison/config/graphene_w90_5x5_to_vacancy_predict_metrics_payload.json",
    )
    parser.add_argument(
        "--vacancy-target",
        default="Comparison/datasets/graphene_5x5_vacancy",
    )
    args = parser.parse_args()
    base = json.loads(args.base.read_text(encoding="utf-8"))
    payload = _pin_deeph_threads(build(base))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    vacancy_payload = _pin_deeph_threads(build_vacancy(base, args.vacancy_target))
    args.vacancy_output.parent.mkdir(parents=True, exist_ok=True)
    args.vacancy_output.write_text(json.dumps(vacancy_payload, indent=2) + "\n", encoding="utf-8")
    print(args.vacancy_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
