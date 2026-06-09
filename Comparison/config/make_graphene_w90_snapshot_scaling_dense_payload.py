from __future__ import annotations

import copy
import json
import time
from pathlib import Path


SRC = Path("Comparison/config/graphene_w90_snapshot_scaling_10_500_paper_ready_like_payload.json")
DST = Path("Comparison/config/graphene_w90_snapshot_scaling_150_1000_dense_payload.json")

RUN_TAG = "graphene_w90_snapshot_scaling_150_1000_dense"

# Denso en 150-400 y en 500-1000.
# 22 datasets × 4 configs = 88 training runs.
SIZES = [
    150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400,
    500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000,
]


def seed_for(temp_k: int, size: int) -> int:
    return int(f"{temp_k}{size:03d}")


def recipe_for_size(size: int) -> dict:
    n_150 = size // 5
    n_450 = size // 5
    n_300 = size - n_150 - n_450

    return {
        "recipe_id": f"graphene_w90_scale_iid{size}",
        "label": f"Graphene W90 IID scale: {size} snapshots",
        "size": size,
        "thermal_regime": "iid_mixed",
        "split_intent": "IID mixed temperature train/validation/test with blocked temporal gap.",
        "blocks": [
            {
                "block_id": f"graphene_w90_scale_iid{size}_T150_{n_150}_1",
                "label": f"{n_150} snapshots @ 150 K",
                "n_snapshots": n_150,
                "temperature_K": 150,
                "seed": seed_for(150, size),
            },
            {
                "block_id": f"graphene_w90_scale_iid{size}_T300_{n_300}_2",
                "label": f"{n_300} snapshots @ 300 K",
                "n_snapshots": n_300,
                "temperature_K": 300,
                "seed": seed_for(300, size),
            },
            {
                "block_id": f"graphene_w90_scale_iid{size}_T450_{n_450}_3",
                "label": f"{n_450} snapshots @ 450 K",
                "n_snapshots": n_450,
                "temperature_K": 450,
                "seed": seed_for(450, size),
            },
        ],
    }


def template_prefix(config_id: str) -> str:
    if "-N" not in config_id:
        raise RuntimeError(f"Unexpected config_id without -N suffix: {config_id}")
    return config_id.rsplit("-N", 1)[0]


def main() -> None:
    payload = json.loads(SRC.read_text(encoding="utf-8"))

    payload["description"] = (
        "Graphene W90 snapshot-scaling dense payload derived from "
        "graphene_w90_snapshot_scaling_10_500_paper_ready_like_payload.json. "
        "It densifies dataset sizes in 150-400 and 500-1000."
    )

    payload["dataset_root"] = "${REPO_ROOT}/Comparison/datasets/graphene_w90_snapshot_scaling_150_1000_dense"
    payload["output_root"] = "${REPO_ROOT}/Comparison/results/graphene_w90_snapshot_scaling_150_1000_dense"
    payload["run_id"] = f"{RUN_TAG}_{time.strftime('%Y%m%d_%H%M%S')}"

    payload["dataset_sweep"]["max_datasets"] = len(SIZES)
    payload["dataset_sweep"]["recipes"] = [recipe_for_size(size) for size in SIZES]

    original_manual_runs = payload["training_sweep"]["manual_runs"]

    # The original payload has four manual configs per dataset.
    # Use the first dataset's four configs as templates.
    templates = copy.deepcopy(original_manual_runs[:4])
    if len(templates) != 4:
        raise RuntimeError("Expected at least four manual run templates.")

    new_manual_runs = []
    for size in SIZES:
        dataset_id = f"graphene_w90_scale_iid{size}"
        for template in templates:
            run = copy.deepcopy(template)
            prefix = template_prefix(str(template["config_id"]))
            new_id = f"{prefix}-N{size}"
            run["id"] = new_id
            run["config_id"] = new_id
            run["dataset_id"] = dataset_id
            new_manual_runs.append(run)

    payload["training_sweep"]["manual_runs"] = new_manual_runs
    payload["training_sweep"]["max_runs"] = len(new_manual_runs)
    payload["training_sweep"]["apply_to_datasets"] = ["all"]

    DST.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Wrote {DST}")
    print(f"run_id: {payload['run_id']}")
    print(f"datasets: {len(SIZES)}")
    print(f"training runs: {len(new_manual_runs)}")
    print(f"sizes: {SIZES}")


if __name__ == "__main__":
    main()