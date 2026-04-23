#!/usr/bin/env python3
"""Collect positions, energies and forces from atom-displacement single-point runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from atom_displacement_utils import (
    COLLECTED_DIR,
    SAMPLES_DIR,
    compute_water_geometry_metrics,
    ensure_dir,
    find_first_output,
    parse_fa_file,
    parse_fdf_structure,
    parse_total_energy_ev,
    parse_xv_structure,
    sample_run_status,
    write_json,
)


def load_metadata(sample_dir: Path) -> dict:
    metadata_path = sample_dir / "metadata.json"
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def collect_sample(sample_dir: Path) -> dict:
    metadata = load_metadata(sample_dir)
    input_structure = parse_fdf_structure(sample_dir / "RUN.fdf")
    status = sample_run_status(sample_dir)
    energy_ev = parse_total_energy_ev(sample_dir)
    fa_path = find_first_output(sample_dir, ".FA")
    xv_path = find_first_output(sample_dir, ".XV")
    forces = parse_fa_file(fa_path) if fa_path is not None else None
    output_structure = (
        parse_xv_structure(xv_path, input_structure.species_labels) if xv_path is not None else None
    )
    metrics = compute_water_geometry_metrics(input_structure)

    return {
        "id": sample_dir.name,
        "status": status,
        "species": input_structure.symbols,
        "positions_ang": input_structure.positions_ang,
        "output_positions_ang": output_structure.positions_ang if output_structure else None,
        "lattice_vectors_ang": input_structure.lattice_vectors_ang,
        "energy_ev": energy_ev,
        "forces_ev_ang": forces,
        "geometry_metrics": metrics,
        "reference_source": metadata.get("reference_source"),
        "displacements_ang": metadata.get("displacements_ang"),
        "displacement_norms_ang": metadata.get("displacement_norms_ang"),
        "files": {
            "run_fdf": str(sample_dir / "RUN.fdf"),
            "run_out": str(sample_dir / "RUN.out"),
            "fa": str(fa_path) if fa_path else None,
            "xv": str(xv_path) if xv_path else None,
        },
    }


def write_summary_csv(rows: list[dict], csv_path: Path) -> None:
    fieldnames = [
        "id",
        "job_completed",
        "scf_converged",
        "energy_ev",
        "oh_1_ang",
        "oh_2_ang",
        "hh_ang",
        "hoh_angle_deg",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "id": row["id"],
                    "job_completed": row["status"]["job_completed"],
                    "scf_converged": row["status"]["scf_converged"],
                    "energy_ev": row["energy_ev"],
                    **row["geometry_metrics"],
                }
            )


def main() -> int:
    ensure_dir(COLLECTED_DIR)
    sample_dirs = sorted(path for path in SAMPLES_DIR.glob("sample_*") if path.is_dir())
    dataset_rows = [collect_sample(sample_dir) for sample_dir in sample_dirs]

    payload = {
        "num_samples": len(dataset_rows),
        "num_completed": sum(1 for row in dataset_rows if row["status"]["job_completed"]),
        "num_converged": sum(1 for row in dataset_rows if row["status"]["scf_converged"]),
        "samples": dataset_rows,
    }
    json_path = COLLECTED_DIR / "water_atom_displacement_dataset.json"
    csv_path = COLLECTED_DIR / "water_atom_displacement_summary.csv"
    write_json(json_path, payload)
    write_summary_csv(dataset_rows, csv_path)

    print("=== AtomDisplacement dataset collected ===")
    print(f"[OK] JSON: {json_path}")
    print(f"[OK] CSV:  {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
