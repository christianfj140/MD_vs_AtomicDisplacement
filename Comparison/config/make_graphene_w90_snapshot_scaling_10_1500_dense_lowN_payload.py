from __future__ import annotations

import copy
import json
import time
from pathlib import Path


SRC = Path("Comparison/config/graphene_w90_snapshot_scaling_10_500_paper_ready_like_payload.json")
DST = Path("Comparison/config/graphene_w90_snapshot_scaling_10_1500_dense_10_30_unit_payload.json")
RUN_TAG = "graphene_w90_snapshot_scaling_10_1500_dense_10_30_unit"

SIZES = [
    # Zona crítica de bajo N: discretización 2 a 2
    10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,

    # Zona baja-media: todavía bastante densa
    35, 40, 45, 50, 55, 60, 65, 70, 75, 80,
    85, 90, 95,

    # Zona hasta 150
    100, 110, 120, 130, 140, 150,

    # Zona media-alta
    200, 250, 300, 400, 500, 600, 700, 800,

    # Zona alta extendida
    900, 1000, 1100, 1200, 1300, 1400, 1500,
]


def seed_for(temp_k: int, size: int) -> int:
    return int(f"{temp_k}{size:04d}")


def recipe_for_size(size: int) -> dict:
    # Mantiene la misma mezcla térmica 20/60/20 usada en los payloads anteriores:
    # 150 K: 20 %, 300 K: 60 %, 450 K: 20 %.
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
    "Graphene W90 snapshot-scaling payload derived from "
    "graphene_w90_snapshot_scaling_10_500_paper_ready_like_payload.json. "
    "It uses unit-step discretization from 10 to 30 snapshots, dense sampling up to 150, "
    "and extends the high-N range to 1500 snapshots."
    )

    payload["source_reference_payloads"] = [
        "Comparison/config/graphene_w90_snapshot_scaling_10_500_paper_ready_like_payload.json",
        "Comparison/config/graphene_w90_snapshot_scaling_150_1000_dense_payload.json",
    ]

    payload["dataset_root"] = (
    "${REPO_ROOT}/Comparison/datasets/"
    "graphene_w90_snapshot_scaling_10_1500_dense_10_30_unit"
    )
    payload["output_root"] = (
        "${REPO_ROOT}/Comparison/results/"
        "graphene_w90_snapshot_scaling_10_1500_dense_10_30_unit"
    )
    payload["run_id"] = f"{RUN_TAG}_{time.strftime('%Y%m%d_%H%M%S')}"

    payload["dataset_sweep"]["max_datasets"] = len(SIZES)
    payload["dataset_sweep"]["recipes"] = [recipe_for_size(size) for size in SIZES]

    original_manual_runs = payload["training_sweep"]["manual_runs"]

    # El payload paper-ready-like usa 4 configs por dataset:
    # G2M-T600-26, G2M-T1000-03, DH-T600-13, DH-T1000-04.
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

    DST.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote: {DST}")
    print(f"run_id: {payload['run_id']}")
    print(f"dataset_root: {payload['dataset_root']}")
    print(f"output_root: {payload['output_root']}")
    print(f"datasets: {len(SIZES)}")
    print(f"training runs: {len(new_manual_runs)}")
    print(f"sizes: {SIZES}")


if __name__ == "__main__":
    main()