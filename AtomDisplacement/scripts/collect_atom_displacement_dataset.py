#!/usr/bin/env python3
"""Collect positions, energies and forces from atom-displacement single-point runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from atom_displacement_utils import (
    COLLECTED_DIR,
    PIPELINE_CONFIG,
    PIPELINE_PATHS,
    compute_water_geometry_metrics,
    ensure_dir,
    find_first_output,
    generated_sample_dirs,
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
    run_fdf_path = sample_dir / PIPELINE_CONFIG["paths"]["run_fdf_name"]
    run_out_path = sample_dir / PIPELINE_CONFIG["paths"]["run_out_name"]
    input_structure = parse_fdf_structure(run_fdf_path)
    status = sample_run_status(sample_dir)
    energy_ev = parse_total_energy_ev(sample_dir)
    fa_path = find_first_output(sample_dir, ".FA")
    xv_path = find_first_output(sample_dir, ".XV")
    fc_path = sample_dir / "FC"
    dhsdr_path = find_first_output(sample_dir, ".dHSdR.nc")
    tshs_path = find_first_output(sample_dir, ".TSHS")
    tsde_path = find_first_output(sample_dir, ".TSDE")
    forces = parse_fa_file(fa_path) if fa_path is not None else None
    output_structure = (
        parse_xv_structure(xv_path, input_structure.species_labels) if xv_path is not None else None
    )
    metrics = compute_water_geometry_metrics(input_structure)

    return {
        "id": metadata.get("id", sample_dir.name),
        "status": status,
        "species": input_structure.symbols,
        "positions_ang": input_structure.positions_ang,
        "output_positions_ang": output_structure.positions_ang if output_structure else None,
        "lattice_vectors_ang": input_structure.lattice_vectors_ang,
        "energy_ev": energy_ev,
        "forces_ev_ang": forces,
        "geometry_metrics": metrics,
        "reference_source": metadata.get("reference_source"),
        "force_constants": metadata.get("force_constants"),
        "files": {
            "run_fdf": str(sample_dir / "RUN.fdf"),
            "run_out": str(run_out_path),
            "fa": str(fa_path) if fa_path else None,
            "xv": str(xv_path) if xv_path else None,
            "fc": str(fc_path) if fc_path.exists() else None,
            "dhsdr": str(dhsdr_path) if dhsdr_path else None,
            "tshs": str(tshs_path) if tshs_path else None,
            "tsde": str(tsde_path) if tsde_path else None,
        },
    }


def write_summary_csv(rows: list[dict], csv_path: Path) -> None:
    fieldnames = [
        "id",
        "sample_dir",
        "structure_path",
        "hamiltonian_path",
        "run_out_path",
        "job_completed",
        "scf_converged",
        "valid",
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
            files = row.get("files", {})
            status = row.get("status", {})

            structure_path = files.get("run_fdf")
            hamiltonian_path = files.get("tshs") or files.get("hsx")
            run_out_path = files.get("run_out")

            writer.writerow(
                {
                    "id": row["id"],
                    "sample_dir": str(Path(structure_path).parent) if structure_path else "",
                    "structure_path": structure_path or "",
                    "hamiltonian_path": hamiltonian_path or "",
                    "run_out_path": run_out_path or "",
                    "job_completed": status.get("job_completed"),
                    "scf_converged": status.get("scf_converged"),
                    "valid": bool(
                        structure_path
                        and hamiltonian_path
                        and status.get("job_completed")
                        and status.get("scf_converged")
                    ),
                    "energy_ev": row["energy_ev"],
                    **row["geometry_metrics"],
                }
            )


def main() -> int:
    ensure_dir(COLLECTED_DIR)
    sample_dirs = generated_sample_dirs()
    dataset_rows = [collect_sample(sample_dir) for sample_dir in sample_dirs]

    payload = {
        "num_samples": len(dataset_rows),
        "num_completed": sum(1 for row in dataset_rows if row["status"]["job_completed"]),
        "num_converged": sum(1 for row in dataset_rows if row["status"]["scf_converged"]),
        "samples": dataset_rows,
    }
    json_path = PIPELINE_PATHS["collected_json_path"]
    csv_path = PIPELINE_PATHS["collected_csv_path"]
    write_json(json_path, payload)
    write_summary_csv(dataset_rows, csv_path)

    print("=== AtomDisplacement dataset collected ===")
    print(f"[OK] JSON: {json_path}")
    print(f"[OK] CSV:  {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
