#!/usr/bin/env python3
"""Compare graphene bands on the Gamma-K-M-Gamma path.

The script is a standalone visualization/evidence tool. It does not select
models, declare winners, or mutate the training workflow. SIESTA bands can be
read from an existing ``SystemLabel.bands`` file, or generated explicitly from
an FDF containing/receiving a native ``%block BandLines`` block. Graph2Mat and
DeepH bands are computed on the same high-symmetry k-path by solving

    H(k) v = E(k) S_ref(k) v

using the SIESTA reference overlap whenever it is available.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import scipy.linalg


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from deeph_prediction_adapter import (  # noqa: E402
    EQUIVALENCE_STATUS_PROVEN,
    find_raw_global_equivalence_evidence,
    validate_raw_global_equivalence_evidence,
)
from evaluate_deeph_kpoint_metrics import assemble_hk  # noqa: E402
from evaluate_hamiltonian_metrics import (  # noqa: E402
    complex_hermiticity_defect,
    kpoint_hamiltonian_matrix,
    kpoint_overlap_matrix,
)
from reference_selection import choose_reference_matrix, file_sha256  # noqa: E402


GRAPHENE_GKM_PATH = [
    ("Gamma", [0.0, 0.0, 0.0]),
    ("K", [1.0 / 3.0, 1.0 / 3.0, 0.0]),
    ("M", [0.5, 0.0, 0.0]),
    ("Gamma", [0.0, 0.0, 0.0]),
]
DISPLAY_LABELS = {"Gamma": "Γ", "K": "K", "M": "M"}
OVERLAP_POLICY = "siesta_reference_overlap_for_all_methods"
KPATH_WARNING = (
    "The default Γ-K-M-Γ coordinates assume the graphene reciprocal-cell "
    "convention used by this repository. Validate against the actual RUN.fdf "
    "lattice or provide --kpath-json."
)
DIRAC_GAP_WARNING_DEFAULT_MEV = 10.0
DIRAC_FERMI_WARNING_DEFAULT_MEV = 50.0


@dataclass(frozen=True)
class KPathNode:
    label: str
    k: tuple[float, float, float]


@dataclass(frozen=True)
class KPointRecord:
    k_index: int
    kx: float
    ky: float
    kz: float
    k_distance: float
    k_label: str
    segment: str

    @property
    def k(self) -> tuple[float, float, float]:
        return (self.kx, self.ky, self.kz)


@dataclass
class MethodBands:
    method: str
    sample_id: str
    bands: list[list[float]]
    fermi_level_eV: float | None
    energy_zero_policy: str
    raw_source: str
    hermiticity_defects: list[float]
    overlap_used: bool
    diagonalization_errors: list[str]
    scientific_status: str = "claim_ready"
    warnings: list[str] | None = None

    def n_bands(self) -> int:
        return len(self.bands[0]) if self.bands else 0


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    extra = sorted({str(key) for row in rows for key in row if str(key) not in fieldnames})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*fieldnames, *extra])
        writer.writeheader()
        writer.writerows(rows)


def run_git_commit(repo_root: Path) -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def label_for_display(label: str) -> str:
    return DISPLAY_LABELS.get(label, label)


def parse_fdf_bandlines(path: Path) -> tuple[str, list[KPathNode]]:
    """Parse a SIESTA BandLines block using reciprocal fractional coordinates."""
    text = path.read_text(encoding="utf-8", errors="replace")
    in_block = False
    nodes: list[KPathNode] = []
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        lower = line.lower()
        if lower.startswith("%block") and "bandlines" in lower:
            in_block = True
            continue
        if in_block and lower.startswith("%endblock") and "bandlines" in lower:
            break
        if not in_block or not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            coords = tuple(float(value.replace("D", "E").replace("d", "e")) for value in parts[1:4])
        except ValueError:
            continue
        label = parts[4].lstrip("\\")
        nodes.append(KPathNode(label=label, k=coords))
    if len(nodes) < 2:
        raise RuntimeError(f"No usable %block BandLines found in {path}")
    return f"{path.stem}_bandlines", nodes


def parse_md_initial_temperature(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        parts = line.split()
        if len(parts) >= 2 and parts[0].lower() == "md.initialtemperature":
            try:
                return {"value": float(parts[1]), "unit": parts[2] if len(parts) >= 3 else "K", "source": str(path)}
            except ValueError:
                return {"raw": " ".join(parts[1:]), "source": str(path)}
    return None


def parse_kpath_json(path: Path) -> tuple[str, list[KPathNode]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    points = payload.get("points")
    if not isinstance(points, list) or len(points) < 2:
        raise RuntimeError(f"Invalid k-path JSON: {path}")
    nodes: list[KPathNode] = []
    for item in points:
        if not isinstance(item, dict):
            raise RuntimeError(f"Invalid k-path point in {path}: {item!r}")
        label = str(item.get("label") or "").strip()
        coords = item.get("k")
        if not label or not isinstance(coords, list) or len(coords) != 3:
            raise RuntimeError(f"Invalid k-path point in {path}: {item!r}")
        nodes.append(KPathNode(label=label, k=tuple(float(value) for value in coords)))
    return str(payload.get("name") or path.stem), nodes


def load_kpath(args: argparse.Namespace) -> tuple[str, list[KPathNode]]:
    if getattr(args, "kpath_from_siesta_run_fdf", False):
        if getattr(args, "siesta_run_fdf", None) is None:
            raise RuntimeError("--kpath-from-siesta-run-fdf requires --siesta-run-fdf")
        return parse_fdf_bandlines(args.siesta_run_fdf)
    if args.kpath_json is not None:
        return parse_kpath_json(args.kpath_json)
    if args.kpath != "graphene_gkm":
        raise RuntimeError(f"Unsupported --kpath: {args.kpath}")
    return "graphene_gkm", [KPathNode(label=label, k=tuple(coords)) for label, coords in GRAPHENE_GKM_PATH]


def interpolate_kpath(nodes: list[KPathNode], points_per_segment: int) -> list[KPointRecord]:
    if points_per_segment <= 0:
        raise RuntimeError("--points-per-segment must be positive")
    records: list[KPointRecord] = []
    distance = 0.0
    first = nodes[0]
    records.append(
        KPointRecord(
            k_index=0,
            kx=first.k[0],
            ky=first.k[1],
            kz=first.k[2],
            k_distance=0.0,
            k_label=label_for_display(first.label),
            segment=f"{label_for_display(first.label)}-{label_for_display(nodes[1].label)}",
        )
    )
    previous = np.asarray(first.k, dtype=float)
    for node_index in range(len(nodes) - 1):
        start = np.asarray(nodes[node_index].k, dtype=float)
        end = np.asarray(nodes[node_index + 1].k, dtype=float)
        segment_label = f"{label_for_display(nodes[node_index].label)}-{label_for_display(nodes[node_index + 1].label)}"
        for step in range(1, points_per_segment + 1):
            t = step / float(points_per_segment)
            point = start + t * (end - start)
            distance += float(np.linalg.norm(point - previous))
            previous = point
            endpoint = step == points_per_segment
            records.append(
                KPointRecord(
                    k_index=len(records),
                    kx=float(point[0]),
                    ky=float(point[1]),
                    kz=float(point[2]),
                    k_distance=distance,
                    k_label=label_for_display(nodes[node_index + 1].label) if endpoint else "",
                    segment=segment_label,
                )
            )
    return records


def bandlines_block(nodes: list[KPathNode], points_per_segment: int) -> str:
    lines = ["BandLinesScale ReciprocalLatticeVectors", "", "%block BandLines"]
    first = nodes[0]
    lines.append(
        f"  1    {first.k[0]:.10f}    {first.k[1]:.10f}    {first.k[2]:.10f}    {label_for_display(first.label)}"
    )
    for node in nodes[1:]:
        lines.append(
            f"  {points_per_segment:d}   {node.k[0]:.10f}    {node.k[1]:.10f}    {node.k[2]:.10f}    {label_for_display(node.label)}"
        )
    lines.append("%endblock BandLines")
    return "\n".join(lines) + "\n"


def strip_existing_bandlines(fdf_text: str) -> str:
    output: list[str] = []
    skipping = False
    for line in fdf_text.splitlines():
        lower = line.strip().lower()
        if lower.startswith("bandlinesscale"):
            continue
        if lower.startswith("%block") and "bandlines" in lower:
            skipping = True
            continue
        if skipping:
            if lower.startswith("%endblock") and "bandlines" in lower:
                skipping = False
            continue
        output.append(line)
    return "\n".join(output).rstrip() + "\n"


def inject_bandlines_block(fdf_text: str, nodes: list[KPathNode], points_per_segment: int) -> str:
    return strip_existing_bandlines(fdf_text) + "\n" + bandlines_block(nodes, points_per_segment)


def infer_system_label(fdf_path: Path, fallback: str = "graphene") -> str:
    if not fdf_path.exists():
        return fallback
    for raw_line in fdf_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 2 and parts[0].lower() == "systemlabel":
            return parts[1]
    return fallback


def run_siesta_band_generation(
    args: argparse.Namespace,
    output_dir: Path,
    nodes: list[KPathNode],
    manifest: dict[str, Any],
) -> Path:
    if args.siesta_run_fdf is None:
        raise RuntimeError("--generate-siesta-bands requires --siesta-run-fdf")
    src_fdf = args.siesta_run_fdf
    if not src_fdf.exists():
        raise RuntimeError(f"SIESTA RUN.fdf not found: {src_fdf}")
    siesta_dir = output_dir / "siesta"
    siesta_dir.mkdir(parents=True, exist_ok=True)
    run_fdf = siesta_dir / "RUN.fdf"
    run_fdf.write_text(
        inject_bandlines_block(src_fdf.read_text(encoding="utf-8", errors="replace"), nodes, args.points_per_segment),
        encoding="utf-8",
    )
    system_label = infer_system_label(run_fdf)
    command = shlex.split(args.siesta_command)
    manifest["siesta_generation"] = {
        "source_fdf": str(src_fdf),
        "run_fdf": str(run_fdf),
        "command": command,
        "system_label": system_label,
    }
    if args.dry_run:
        manifest["siesta_generation"]["dry_run"] = True
        return siesta_dir / f"{system_label}.bands"
    with run_fdf.open("rb") as stdin, (siesta_dir / "RUN.out").open("wb") as stdout:
        completed = subprocess.run(command, cwd=siesta_dir, stdin=stdin, stdout=stdout, stderr=subprocess.STDOUT, check=False)
    manifest["siesta_generation"]["returncode"] = completed.returncode
    manifest["siesta_generation"]["stdout"] = str(siesta_dir / "RUN.out")
    if completed.returncode != 0:
        raise RuntimeError(f"SIESTA band generation failed with return code {completed.returncode}")
    bands_path = siesta_dir / f"{system_label}.bands"
    if not bands_path.exists():
        raise RuntimeError(f"SIESTA completed but did not produce {bands_path}")
    return bands_path


def try_run_gnubands(bands_path: Path, output_dir: Path, args: argparse.Namespace, manifest: dict[str, Any]) -> Path | None:
    gnubands = shutil.which(args.gnubands or "gnubands")
    manifest["gnubands"] = {"available": bool(gnubands), "used": False}
    if not gnubands or args.dry_run:
        return None
    siesta_dir = output_dir / "siesta"
    data_path = siesta_dir / "siesta_bands.dat"
    command = [gnubands, "-G", "-o", "siesta_bands", "-F", str(bands_path)]
    completed = subprocess.run(command, cwd=siesta_dir, text=True, capture_output=True, check=False)
    (siesta_dir / "gnubands.stdout").write_text(completed.stdout, encoding="utf-8")
    (siesta_dir / "gnubands.stderr").write_text(completed.stderr, encoding="utf-8")
    manifest["gnubands"].update(
        {
            "used": completed.returncode == 0,
            "command": command,
            "returncode": completed.returncode,
            "stdout": str(siesta_dir / "gnubands.stdout"),
            "stderr": str(siesta_dir / "gnubands.stderr"),
        }
    )
    return data_path if data_path.exists() else None


def numeric_tokens(line: str) -> list[float] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    values: list[float] = []
    for token in stripped.replace(",", " ").split():
        try:
            values.append(float(token.replace("D", "E").replace("d", "e")))
        except ValueError:
            return None
    return values if values else None


def parse_tabulated_band_file(path: Path, method: str, sample_id: str) -> MethodBands:
    if path.suffix.lower() == ".csv":
        by_k_band: dict[tuple[int, int], tuple[float, float]] = {}
        fermi: float | None = None
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                try:
                    k_index = int(float(row.get("k_index") or row.get("k") or 0))
                    band_index = int(float(row.get("band_index") or row.get("band") or 0))
                    energy = float(row.get("energy_eV") or row.get("eigenvalue_eV") or row.get("y"))
                    distance = float(row.get("k_distance") or row.get("distance") or row.get("x") or k_index)
                except (TypeError, ValueError):
                    continue
                if row.get("fermi_level_eV") not in (None, ""):
                    try:
                        fermi = float(row["fermi_level_eV"])
                    except ValueError:
                        pass
                by_k_band[(k_index, band_index)] = (distance, energy)
        if not by_k_band:
            raise RuntimeError(f"No band rows parsed from {path}")
        return bands_from_k_band_map(method, sample_id, by_k_band, fermi, str(path))

    rows: list[list[float] | None] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        rows.append(numeric_tokens(line))
    two_column = [row for row in rows if row and len(row) == 2]
    if two_column and len(two_column) >= max(2, len([row for row in rows if row is not None]) // 2):
        bands: list[list[float]] = []
        current: list[float] = []
        for row in rows:
            if row is None:
                if current:
                    bands.append(current)
                    current = []
                continue
            if len(row) >= 2:
                current.append(row[1])
        if current:
            bands.append(current)
        transposed = [list(values) for values in zip(*bands, strict=False)] if bands else []
        if transposed:
            return MethodBands(method, sample_id, transposed, None, "none", str(path), [], False, [])

    raw = [row for row in rows if row]
    if len(raw) >= 5 and len(raw[0]) == 1:
        fermi = raw[0][0]
        header_index = None
        for index, row in enumerate(raw[1:8], start=1):
            if len(row) >= 3 and all(float(value).is_integer() and value > 0 for value in row[:3]):
                header_index = index
                break
        if header_index is not None:
            nbands = int(raw[header_index][0])
            nspin = int(raw[header_index][1])
            nk = int(raw[header_index][2])
            needed = 1 + nbands * nspin
            cursor = header_index + 1
            by_k_band: dict[tuple[int, int], tuple[float, float]] = {}
            for k_index in range(nk):
                collected: list[float] = []
                while cursor < len(raw) and len(collected) < needed:
                    collected.extend(raw[cursor])
                    cursor += 1
                if len(collected) < needed:
                    break
                distance = collected[0]
                for band_index, energy in enumerate(collected[1:needed]):
                    by_k_band[(k_index, band_index)] = (distance, energy)
            if by_k_band:
                return bands_from_k_band_map(method, sample_id, by_k_band, fermi, str(path))
    raise RuntimeError(
        f"Could not parse {path} as CSV, gnubands/two-column data, or a supported SIESTA .bands layout"
    )


def bands_from_k_band_map(
    method: str,
    sample_id: str,
    by_k_band: dict[tuple[int, int], tuple[float, float]],
    fermi: float | None,
    source: str,
) -> MethodBands:
    k_indices = sorted({key[0] for key in by_k_band})
    band_indices = sorted({key[1] for key in by_k_band})
    bands: list[list[float]] = []
    for k_index in k_indices:
        row: list[float] = []
        for band_index in band_indices:
            item = by_k_band.get((k_index, band_index))
            row.append(item[1] if item else math.nan)
        bands.append(row)
    return MethodBands(method, sample_id, bands, fermi, "fermi" if fermi is not None else "none", source, [], False, [])


def resolve_reference_path(path: Path) -> Path:
    if path.name == "ML_prediction.HSX":
        raise RuntimeError("ML_prediction.HSX must never be used as a SIESTA reference")
    if path.is_dir():
        selection = choose_reference_matrix(path)
        if not selection.ok or selection.path is None:
            raise RuntimeError(
                f"Could not select a strict SIESTA reference from {path}: "
                f"{selection.reason}; candidates={list(selection.candidates)}"
            )
        return selection.path
    if not path.exists():
        raise RuntimeError(f"Reference path does not exist: {path}")
    if path.name == "ML_prediction.HSX":
        raise RuntimeError("ML_prediction.HSX must never be used as a SIESTA reference")
    return path


def read_sisl_hamiltonian(path: Path) -> Any:
    import sisl

    return sisl.get_sile(str(path)).read_hamiltonian()


def solve_generalized_bands(Hk: Any, Sk: Any | None = None) -> np.ndarray:
    H = np.asarray(Hk, dtype=np.complex128)
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise RuntimeError(f"H(k) must be square; got {H.shape}")
    H = 0.5 * (H + H.conj().T)
    if Sk is None:
        values = np.linalg.eigvalsh(H)
    else:
        S = np.asarray(Sk, dtype=np.complex128)
        if S.shape != H.shape:
            raise RuntimeError(f"S(k) shape {S.shape} does not match H(k) shape {H.shape}")
        S = 0.5 * (S + S.conj().T)
        values = scipy.linalg.eigh(H, S, eigvals_only=True, check_finite=False)
    values = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(values)):
        raise RuntimeError("Non-finite eigenvalues produced by generalized solve")
    return values


def matrix_bands_from_sisl(
    *,
    method: str,
    sample_id: str,
    hamiltonian_obj: Any,
    reference_obj: Any,
    kpoints: list[KPointRecord],
    fermi_level: float | None,
    fail_closed: bool,
) -> MethodBands:
    rows: list[list[float]] = []
    hermiticity: list[float] = []
    errors: list[str] = []
    overlap_used = False
    for rec in kpoints:
        try:
            h_k = kpoint_hamiltonian_matrix(hamiltonian_obj, list(rec.k))
            try:
                s_ref_k = kpoint_overlap_matrix(reference_obj, list(rec.k))
            except Exception as exc:
                if fail_closed:
                    raise
                s_ref_k = None
                errors.append(f"k{rec.k_index}: missing overlap, used standard eigensolve: {exc}")
            overlap_used = overlap_used or s_ref_k is not None
            hermiticity.append(complex_hermiticity_defect(h_k))
            rows.append(solve_generalized_bands(h_k, s_ref_k).tolist())
        except Exception as exc:
            errors.append(f"k{rec.k_index}: {exc}")
            if fail_closed:
                raise RuntimeError(f"{method} band solve failed at k-index {rec.k_index}: {exc}") from exc
            rows.append([])
    status = "claim_ready" if not errors else "diagnostic_only"
    return MethodBands(method, sample_id, rows, fermi_level, "fermi" if fermi_level is not None else "none", "sisl_hamiltonian", hermiticity, overlap_used, errors, status)


def fermi_from_reference(path: Path) -> tuple[float | None, str]:
    try:
        import sisl

        value = float(sisl.get_sile(str(path)).read_fermi_level())
        return value, "siesta_file"
    except Exception:
        return None, "unavailable"


def deeph_equivalence_status(
    *,
    prediction: Path,
    processed_dir: Path,
    sample_id: str,
    explicit: Path | None,
) -> tuple[str, str, Path | None]:
    path = explicit
    if path is None:
        path = find_raw_global_equivalence_evidence(work_dir=prediction.parent, processed_sample_dir=processed_dir, sample_id=sample_id)
    if path is None:
        return "unproven", "DeepH raw/global equivalence evidence was not found.", None
    validation = validate_raw_global_equivalence_evidence(path, sample_id=sample_id)
    if validation.get("status") == EQUIVALENCE_STATUS_PROVEN:
        return "proven", "DeepH raw/global equivalence evidence is proven.", path
    return "failed", str(validation.get("reason") or "DeepH equivalence evidence failed validation."), path


def deeph_bands_from_h5(
    *,
    sample_id: str,
    processed_dir: Path,
    prediction: Path,
    overlap: Path,
    kpoints: list[KPointRecord],
    fail_closed: bool,
    equivalence_status: str,
    equivalence_reason: str,
    energy_reference_shift_eV: float = 0.0,
) -> MethodBands:
    rows: list[list[float]] = []
    hermiticity: list[float] = []
    errors: list[str] = []
    for rec in kpoints:
        try:
            h_k = assemble_hk(prediction, processed_dir, rec.k)
            s_k = assemble_hk(overlap, processed_dir, rec.k)
            if energy_reference_shift_eV:
                h_k = h_k + energy_reference_shift_eV * s_k
            hermiticity.append(complex_hermiticity_defect(h_k))
            rows.append(solve_generalized_bands(h_k, s_k).tolist())
        except Exception as exc:
            errors.append(f"k{rec.k_index}: {exc}")
            if fail_closed:
                raise RuntimeError(f"DeepH band solve failed at k-index {rec.k_index}: {exc}") from exc
            rows.append([])
    status = "claim_ready" if equivalence_status == "proven" and not errors else "diagnostic_only"
    warnings = [] if equivalence_status == "proven" else [equivalence_reason]
    return MethodBands("DeepH", sample_id, rows, None, "none", str(prediction), hermiticity, True, errors, status, warnings)


def deeph_energy_reference_shift(evidence_path: Path | None) -> float:
    if evidence_path is None or not evidence_path.exists():
        return 0.0
    try:
        payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    except Exception:
        return 0.0
    alignment = payload.get("energy_reference_alignment")
    if not isinstance(alignment, dict):
        return 0.0
    try:
        return float(alignment.get("shift_eV") or 0.0)
    except Exception:
        return 0.0


def align_energy(value: float, fermi: float | None, policy: str, custom_zero: float | None) -> float:
    if policy == "none":
        return value
    if policy == "custom":
        if custom_zero is None:
            raise RuntimeError("--energy-zero custom requires --custom-energy-zero")
        return value - custom_zero
    if policy == "fermi":
        if fermi is None:
            raise RuntimeError("Fermi level is unavailable")
        return value - fermi
    raise RuntimeError(f"Unknown energy-zero policy: {policy}")


def method_energy_aligned(
    method: MethodBands,
    value: float,
    policy: str,
    zero: float | None,
    custom_zero: float | None = None,
) -> float:
    if policy == "fermi" and method.method != "SIESTA":
        return value
    return align_energy(value, zero, policy, custom_zero)


def normalize_energy_zero(
    methods: list[MethodBands],
    *,
    policy: str,
    fermi_level: float | None,
    custom_zero: float | None,
    fail_closed: bool,
    warnings: list[str],
) -> tuple[str, float | None]:
    if policy == "custom":
        return "custom", custom_zero
    if policy == "none":
        return "none", None
    if fermi_level is not None:
        return "fermi", fermi_level
    for method in methods:
        if method.fermi_level_eV is not None:
            return "fermi", method.fermi_level_eV
    message = "Fermi level is unavailable; cannot use --energy-zero fermi."
    if fail_closed:
        raise RuntimeError(message)
    warnings.append(message + " Falling back to energy-zero=none.")
    return "none", None


def band_rows(method: MethodBands, kpoints: list[KPointRecord], policy: str, zero: float | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rec, energies in zip(kpoints, method.bands, strict=False):
        for band_index, energy in enumerate(energies):
            rows.append(
                {
                    "method": method.method,
                    "sample_id": method.sample_id,
                    "k_index": rec.k_index,
                    "k_distance": rec.k_distance,
                    "kx": rec.kx,
                    "ky": rec.ky,
                    "kz": rec.kz,
                    "k_label": rec.k_label,
                    "segment": rec.segment,
                    "band_index": band_index,
                    "energy_eV": energy,
                    "energy_aligned_eV": method_energy_aligned(
                        method,
                        float(energy),
                        policy,
                        zero,
                        zero if policy == "custom" else None,
                    ),
                    "fermi_level_eV": zero if policy == "fermi" else None,
                    "energy_zero_policy": policy,
                }
            )
    return rows


def error_rows(
    method: MethodBands,
    siesta: MethodBands,
    kpoints: list[KPointRecord],
    policy: str,
    zero: float | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n_k = min(len(method.bands), len(siesta.bands), len(kpoints))
    for k_index in range(n_k):
        rec = kpoints[k_index]
        n_bands = min(len(method.bands[k_index]), len(siesta.bands[k_index]))
        for band_index in range(n_bands):
            ref = align_energy(float(siesta.bands[k_index][band_index]), zero, policy, zero if policy == "custom" else None)
            pred = method_energy_aligned(
                method,
                float(method.bands[k_index][band_index]),
                policy,
                zero,
                zero if policy == "custom" else None,
            )
            delta = pred - ref
            rows.append(
                {
                    "method": method.method,
                    "sample_id": method.sample_id,
                    "k_index": rec.k_index,
                    "k_distance": rec.k_distance,
                    "band_index": band_index,
                    "siesta_energy_eV": ref,
                    "predicted_energy_eV": pred,
                    "error_eV": delta,
                    "abs_error_eV": abs(delta),
                    "squared_error_eV": delta * delta,
                    "k_label": rec.k_label,
                    "segment": rec.segment,
                }
            )
    return rows


def metric_summary(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    errors = np.asarray([float(row["error_eV"]) for row in rows if math.isfinite(float(row["error_eV"]))], dtype=float)
    if errors.size == 0:
        return {"band_mae_eV": None, "band_rmse_eV": None, "max_abs_error_eV": None}
    return {
        "band_mae_eV": float(np.mean(np.abs(errors))),
        "band_rmse_eV": float(np.sqrt(np.mean(errors**2))),
        "max_abs_error_eV": float(np.max(np.abs(errors))),
    }


def infer_occupied_bands(siesta: MethodBands, explicit: int | None) -> int:
    if explicit is not None:
        if explicit <= 0:
            raise RuntimeError("--occupied-bands must be positive")
        return explicit
    n_bands = siesta.n_bands()
    if n_bands < 2:
        raise RuntimeError("At least two bands are required for Dirac diagnostics")
    return n_bands // 2


def first_kpoint_with_label(kpoints: list[KPointRecord], label: str) -> int | None:
    normalized = label.strip().lstrip("\\")
    normalized_display = label_for_display(normalized)
    for rec in kpoints:
        rec_label = rec.k_label.strip().lstrip("\\")
        if rec_label in {normalized, normalized_display}:
            return rec.k_index
    return None


def dirac_diagnostic_for_method(
    method: MethodBands,
    *,
    k_index: int,
    occupied_bands: int,
    fermi_level_eV: float | None,
    gap_warning_meV: float,
    fermi_warning_meV: float,
) -> dict[str, Any]:
    if k_index >= len(method.bands):
        return {"status": "unavailable", "reason": f"k-index {k_index} not available"}
    energies = [float(value) for value in method.bands[k_index]]
    valence_index = occupied_bands - 1
    conduction_index = occupied_bands
    if conduction_index >= len(energies):
        return {
            "status": "unavailable",
            "reason": f"occupied_bands={occupied_bands} incompatible with n_bands={len(energies)}",
        }
    valence = energies[valence_index]
    conduction = energies[conduction_index]
    gap = abs(conduction - valence)
    dirac = 0.5 * (conduction + valence)
    if fermi_level_eV is not None:
        dirac_minus_fermi = method_energy_aligned(method, dirac, "fermi", fermi_level_eV)
        dirac_fermi_convention = "prediction_already_fermi_aligned" if method.method != "SIESTA" else "raw_minus_reference_fermi"
    else:
        dirac_minus_fermi = None
        dirac_fermi_convention = "unavailable"

    closest_pair: dict[str, Any] | None = None
    if len(energies) >= 2:
        candidates = []
        for lower_index, (lower, upper) in enumerate(zip(energies, energies[1:], strict=False)):
            candidates.append(
                {
                    "lower_band_index": lower_index,
                    "upper_band_index": lower_index + 1,
                    "gap_eV": abs(upper - lower),
                    "center_energy_eV": 0.5 * (upper + lower),
                    "lower_energy_eV": lower,
                    "upper_energy_eV": upper,
                }
            )
        closest_pair = min(candidates, key=lambda item: item["gap_eV"])

    warnings = []
    if gap * 1000.0 > gap_warning_meV:
        warnings.append(f"K gap {gap * 1000.0:.3f} meV exceeds {gap_warning_meV:.3f} meV")
    if dirac_minus_fermi is not None and abs(dirac_minus_fermi) * 1000.0 > fermi_warning_meV:
        warnings.append(
            f"Dirac-Fermi shift {dirac_minus_fermi * 1000.0:.3f} meV exceeds {fermi_warning_meV:.3f} meV"
        )
    return {
        "status": "ok",
        "method": method.method,
        "k_index": k_index,
        "occupied_bands": occupied_bands,
        "valence_band_index": valence_index,
        "conduction_band_index": conduction_index,
        "valence_energy_eV": valence,
        "conduction_energy_eV": conduction,
        "gap_eV": gap,
        "dirac_energy_eV": dirac,
        "fermi_level_eV": fermi_level_eV,
        "dirac_minus_fermi_eV": dirac_minus_fermi,
        "dirac_fermi_convention": dirac_fermi_convention,
        "closest_adjacent_pair": closest_pair,
        "warnings": warnings,
    }


def compute_dirac_diagnostics(
    methods: list[MethodBands],
    siesta: MethodBands,
    kpoints: list[KPointRecord],
    args: argparse.Namespace,
    fermi_level_eV: float | None,
) -> dict[str, Any]:
    k_index = first_kpoint_with_label(kpoints, args.dirac_k_label)
    if k_index is None:
        return {"status": "unavailable", "reason": f"k-label {args.dirac_k_label!r} not found"}
    occupied_bands = infer_occupied_bands(siesta, args.occupied_bands)
    per_method = {
        method.method.lower(): dirac_diagnostic_for_method(
            method,
            k_index=k_index,
            occupied_bands=occupied_bands,
            fermi_level_eV=fermi_level_eV,
            gap_warning_meV=args.dirac_gap_warning_mev,
            fermi_warning_meV=args.dirac_fermi_warning_mev,
        )
        for method in methods
    }
    siesta_diag = per_method.get("siesta", {})
    return {
        "status": "ok" if siesta_diag.get("status") == "ok" else "diagnostic_only",
        "dirac_k_label": args.dirac_k_label,
        "dirac_k_index": k_index,
        "occupied_bands": occupied_bands,
        "gap_warning_threshold_meV": args.dirac_gap_warning_mev,
        "dirac_fermi_warning_threshold_meV": args.dirac_fermi_warning_mev,
        "methods": per_method,
        "reference_dirac_energy_eV": siesta_diag.get("dirac_energy_eV"),
        "reference_dirac_minus_fermi_eV": siesta_diag.get("dirac_minus_fermi_eV"),
        "warnings": [
            f"{method_key}: {warning}"
            for method_key, payload in per_method.items()
            for warning in payload.get("warnings", [])
        ],
    }


def select_band_indices(siesta: MethodBands, policy: str, zero: float | None, args: argparse.Namespace) -> list[int]:
    n_bands = siesta.n_bands()
    if args.plot_all_bands:
        return list(range(n_bands))
    limit = args.max_bands or args.bands_around_fermi
    scores: list[tuple[float, int]] = []
    for band_index in range(n_bands):
        values = []
        for row in siesta.bands:
            if band_index < len(row):
                values.append(abs(align_energy(float(row[band_index]), zero, policy, zero if policy == "custom" else None)))
        scores.append((float(np.nanmedian(values)) if values else math.inf, band_index))
    return sorted(band for _score, band in sorted(scores)[:limit])


def plot_comparison(
    output_dir: Path,
    methods: list[MethodBands],
    siesta: MethodBands,
    kpoints: list[KPointRecord],
    policy: str,
    zero: float | None,
    args: argparse.Namespace,
    *,
    output_stem: str = "band_comparison",
    y_offset_eV: float = 0.0,
    method_offsets_eV: dict[str, float] | None = None,
    ylabel: str | None = None,
    title_suffix: str = "",
    annotation_lines: list[str] | None = None,
) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    band_ids = select_band_indices(siesta, policy, zero, args)
    colors = {"SIESTA": "black", "Graph2Mat": "#1f77b4", "DeepH": "#d62728"}
    linestyles = {"SIESTA": "-", "Graph2Mat": "--", "DeepH": ":"}
    fig, ax = plt.subplots(figsize=(12.5, 6.8))
    x = [rec.k_distance for rec in kpoints]
    for method in methods:
        used_label = False
        for band_index in band_ids:
            y = []
            x_local = []
            for rec, energies in zip(kpoints, method.bands, strict=False):
                if band_index >= len(energies):
                    continue
                x_local.append(rec.k_distance)
                method_offset = (method_offsets_eV or {}).get(method.method, 0.0)
                y.append(
                    method_energy_aligned(
                        method,
                        float(energies[band_index]),
                        policy,
                        zero,
                        zero if policy == "custom" else None,
                    )
                    - y_offset_eV
                    - method_offset
                )
            if not y:
                continue
            ax.plot(
                x_local,
                y,
                color=colors.get(method.method, "#666666"),
                linestyle=linestyles.get(method.method, "-"),
                linewidth=1.8 if method.method == "SIESTA" else 1.45,
                alpha=0.92,
                label=method.method if not used_label else None,
            )
            used_label = True
    seen_tick_positions: list[float] = []
    seen_tick_labels: list[str] = []
    for rec in kpoints:
        if rec.k_label:
            seen_tick_positions.append(rec.k_distance)
            seen_tick_labels.append(rec.k_label)
            ax.axvline(rec.k_distance, color="#777777", linewidth=0.85, alpha=0.45)
    if policy == "fermi":
        ax.axhline(0.0, color="#4d4d4d", linewidth=0.9, alpha=0.55)
    if x:
        ax.set_xlim(min(x), max(x))
    ax.set_xticks(seen_tick_positions)
    ax.set_xticklabels(seen_tick_labels)
    ax.set_ylabel(ylabel or ("Energy relative to Fermi (eV)" if policy == "fermi" else "Energy (eV)"))
    ax.set_xlabel("Accumulated k-path distance")
    ax.set_title("Graphene band structure: Γ-K-M-Γ\nSIESTA vs Graph2Mat vs DeepH" + title_suffix)
    ax.grid(True, color="#e5e8ef", linewidth=0.8)
    ax.legend(frameon=False, ncols=3)
    if annotation_lines:
        ax.text(
            0.99,
            0.98,
            "\n".join(annotation_lines),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            color="#333333",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#c8ced8", "alpha": 0.88},
        )
    deeph = next((method for method in methods if method.method == "DeepH"), None)
    if deeph is not None and deeph.scientific_status == "diagnostic_only":
        ax.text(
            0.01,
            0.02,
            "DeepH diagnostic-only: raw/global equivalence not proven",
            transform=ax.transAxes,
            color="#8a1f11",
            fontsize=9,
    )
    fig.tight_layout()
    png = output_dir / f"{output_stem}.png"
    pdf = output_dir / f"{output_stem}.pdf"
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)
    return {f"{output_stem}.png": str(png), f"{output_stem}.pdf": str(pdf)}


def write_kpoints_csv(output_dir: Path, kpoints: list[KPointRecord]) -> None:
    write_csv(
        output_dir / "kpoints.csv",
        [
            {
                "k_index": rec.k_index,
                "kx": rec.kx,
                "ky": rec.ky,
                "kz": rec.kz,
                "k_distance": rec.k_distance,
                "k_label": rec.k_label,
                "segment": rec.segment,
            }
            for rec in kpoints
        ],
        ["k_index", "kx", "ky", "kz", "k_distance", "k_label", "segment"],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-id", default="graphene_sample_000")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--kpath", default="graphene_gkm")
    parser.add_argument("--kpath-json", type=Path, default=None)
    parser.add_argument(
        "--kpath-from-siesta-run-fdf",
        action="store_true",
        help="Use the native %block BandLines coordinates from --siesta-run-fdf.",
    )
    parser.add_argument("--points-per-segment", type=int, default=80)

    parser.add_argument("--siesta-bands", type=Path, default=None)
    parser.add_argument("--siesta-band-data", type=Path, default=None, help="Preconverted gnubands/CSV/two-column SIESTA data.")
    parser.add_argument("--siesta-run-fdf", type=Path, default=None)
    parser.add_argument("--siesta-command", default="siesta")
    parser.add_argument("--generate-siesta-bands", action="store_true")
    parser.add_argument("--gnubands", default=None)

    parser.add_argument("--graph2mat-prediction", type=Path, default=None)
    parser.add_argument("--graph2mat-reference", type=Path, default=None)
    parser.add_argument("--graph2mat-band-data", type=Path, default=None)

    parser.add_argument("--deeph-processed-dir", type=Path, default=None)
    parser.add_argument("--deeph-prediction", type=Path, default=None)
    parser.add_argument("--deeph-overlap", type=Path, default=None)
    parser.add_argument("--deeph-predictions-dir", type=Path, default=None)
    parser.add_argument("--deeph-prediction-filename", default="hamiltonians_pred.h5")
    parser.add_argument("--deeph-band-data", type=Path, default=None)
    parser.add_argument("--deeph-equivalence-evidence", type=Path, default=None)

    parser.add_argument("--energy-zero", choices=["fermi", "none", "custom"], default="fermi")
    parser.add_argument("--fermi-level", type=float, default=None)
    parser.add_argument("--custom-energy-zero", type=float, default=None)
    parser.add_argument("--bands-around-fermi", type=int, default=20)
    parser.add_argument("--max-bands", type=int, default=None)
    parser.add_argument("--plot-all-bands", action="store_true")
    parser.add_argument("--occupied-bands", type=int, default=None, help="Occupied band count for the K-point Dirac diagnostic.")
    parser.add_argument("--dirac-k-label", default="K")
    parser.add_argument("--dirac-gap-warning-mev", type=float, default=DIRAC_GAP_WARNING_DEFAULT_MEV)
    parser.add_argument("--dirac-fermi-warning-mev", type=float, default=DIRAC_FERMI_WARNING_DEFAULT_MEV)
    parser.add_argument(
        "--no-dirac-aligned-plot",
        action="store_true",
        help="Do not generate band_comparison_dirac_aligned.*.",
    )
    parser.add_argument(
        "--method-dirac-gauge-aligned-plot",
        action="store_true",
        help="Also plot each predicted method shifted so its K Dirac midpoint matches SIESTA.",
    )
    parser.add_argument("--fail-closed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-plot", action="store_true")
    return parser.parse_args()


def input_hashes(paths: dict[str, Path | None]) -> dict[str, str | None]:
    return {key: file_sha256(path) if path is not None else None for key, path in paths.items()}


def build_methods(args: argparse.Namespace, kpoints: list[KPointRecord], output_dir: Path, manifest: dict[str, Any]) -> list[MethodBands]:
    methods: list[MethodBands] = []
    warnings = manifest.setdefault("warnings", [])

    bands_path = args.siesta_bands
    if args.generate_siesta_bands:
        kpath_name, nodes = load_kpath(args)
        bands_path = run_siesta_band_generation(args, output_dir, nodes, manifest)
    if bands_path is not None:
        if not args.dry_run and not bands_path.exists():
            raise RuntimeError(f"SIESTA .bands file not found: {bands_path}")
        if bands_path.exists():
            siesta_dir = output_dir / "siesta"
            siesta_dir.mkdir(parents=True, exist_ok=True)
            copied_bands = siesta_dir / bands_path.name
            if bands_path.resolve() != copied_bands.resolve():
                shutil.copy2(bands_path, copied_bands)
            converted = try_run_gnubands(copied_bands, output_dir, args, manifest)
            methods.append(parse_tabulated_band_file(converted or copied_bands, "SIESTA", args.sample_id))
    elif args.siesta_band_data is not None:
        methods.append(parse_tabulated_band_file(args.siesta_band_data, "SIESTA", args.sample_id))

    reference_path: Path | None = None
    reference_obj: Any | None = None
    reference_fermi, reference_fermi_source = (None, "unavailable")
    if args.graph2mat_reference is not None:
        reference_path = resolve_reference_path(args.graph2mat_reference)
        reference_obj = read_sisl_hamiltonian(reference_path)
        reference_fermi, reference_fermi_source = fermi_from_reference(reference_path)
        if not any(method.method == "SIESTA" for method in methods):
            methods.append(
                matrix_bands_from_sisl(
                    method="SIESTA",
                    sample_id=args.sample_id,
                    hamiltonian_obj=reference_obj,
                    reference_obj=reference_obj,
                    kpoints=kpoints,
                    fermi_level=reference_fermi,
                    fail_closed=args.fail_closed,
                )
            )
            manifest["siesta_status"] = "computed_from_reference_matrix"
    if args.graph2mat_band_data is not None:
        methods.append(parse_tabulated_band_file(args.graph2mat_band_data, "Graph2Mat", args.sample_id))
    elif args.graph2mat_prediction is not None:
        if reference_obj is None:
            raise RuntimeError("--graph2mat-prediction requires --graph2mat-reference for S_ref(k)")
        if args.graph2mat_prediction.name == "ML_prediction.HSX" and not args.graph2mat_prediction.exists():
            raise RuntimeError(f"Graph2Mat prediction not found: {args.graph2mat_prediction}")
        prediction_obj = read_sisl_hamiltonian(args.graph2mat_prediction)
        methods.append(
            matrix_bands_from_sisl(
                method="Graph2Mat",
                sample_id=args.sample_id,
                hamiltonian_obj=prediction_obj,
                reference_obj=reference_obj,
                kpoints=kpoints,
                fermi_level=reference_fermi,
                fail_closed=args.fail_closed,
            )
        )

    if args.deeph_band_data is not None:
        methods.append(parse_tabulated_band_file(args.deeph_band_data, "DeepH", args.sample_id))
    else:
        prediction = args.deeph_prediction
        if prediction is None and args.deeph_predictions_dir is not None:
            prediction = args.deeph_predictions_dir / args.deeph_prediction_filename
        if prediction is not None or args.deeph_processed_dir is not None:
            if args.deeph_processed_dir is None or prediction is None:
                raise RuntimeError("DeepH bands require --deeph-processed-dir and --deeph-prediction or --deeph-predictions-dir")
            overlap = args.deeph_overlap or (args.deeph_processed_dir / "overlaps.h5")
            for required in (args.deeph_processed_dir / "orbital_types.dat", prediction, overlap):
                if not required.exists():
                    raise RuntimeError(f"Missing DeepH band input: {required}")
            eq_status, eq_reason, eq_path = deeph_equivalence_status(
                prediction=prediction,
                processed_dir=args.deeph_processed_dir,
                sample_id=args.sample_id,
                explicit=args.deeph_equivalence_evidence,
            )
            manifest["deeph_equivalence"] = {"status": eq_status, "reason": eq_reason, "evidence_path": eq_path}
            if args.fail_closed and eq_status != "proven":
                raise RuntimeError(f"DeepH raw/global equivalence is not proven: {eq_reason}")
            deeph_shift_eV = deeph_energy_reference_shift(eq_path)
            manifest["deeph_energy_reference_alignment"] = {
                "policy": "apply_shift_times_reference_overlap",
                "shift_eV": deeph_shift_eV,
                "source": eq_path,
            }
            methods.append(
                deeph_bands_from_h5(
                    sample_id=args.sample_id,
                    processed_dir=args.deeph_processed_dir,
                    prediction=prediction,
                    overlap=overlap,
                    kpoints=kpoints,
                    fail_closed=args.fail_closed,
                    equivalence_status=eq_status,
                    equivalence_reason=eq_reason,
                    energy_reference_shift_eV=deeph_shift_eV,
                )
            )
            if eq_status != "proven":
                warnings.append("DeepH raw/global equivalence was not proven; DeepH plot is diagnostic-only.")

    if args.fermi_level is not None:
        manifest["fermi_level_source"] = "cli"
    elif reference_fermi is not None:
        manifest["fermi_level_source"] = reference_fermi_source
    return methods


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "script": str(Path(__file__).resolve()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repo_commit": run_git_commit(Path.cwd()),
        "output_dir": str(args.output_dir),
        "overlap_policy": OVERLAP_POLICY,
        "warnings": [],
        "fatal_errors": [],
        "outputs": {},
    }
    try:
        kpath_name, nodes = load_kpath(args)
        kpoints = interpolate_kpath(nodes, args.points_per_segment)
        write_kpoints_csv(args.output_dir, kpoints)
        manifest["kpath"] = {
            "name": kpath_name,
            "points_per_segment": args.points_per_segment,
            "labels": [label_for_display(node.label) for node in nodes],
            "n_kpoints": len(kpoints),
            "source": "siesta_run_fdf_bandlines"
            if args.kpath_from_siesta_run_fdf
            else ("json" if args.kpath_json is not None else "built_in_default"),
            "kpath_convention_warning": KPATH_WARNING
            if args.kpath_json is None and not args.kpath_from_siesta_run_fdf
            else "",
        }
        manifest["input_paths"] = {
            "siesta_bands": args.siesta_bands,
            "siesta_band_data": args.siesta_band_data,
            "siesta_run_fdf": args.siesta_run_fdf,
            "graph2mat_prediction": args.graph2mat_prediction,
            "graph2mat_reference": args.graph2mat_reference,
            "graph2mat_band_data": args.graph2mat_band_data,
            "deeph_processed_dir": args.deeph_processed_dir,
            "deeph_prediction": args.deeph_prediction,
            "deeph_overlap": args.deeph_overlap,
            "deeph_band_data": args.deeph_band_data,
            "deeph_equivalence_evidence": args.deeph_equivalence_evidence,
        }
        snapshot_temperature = parse_md_initial_temperature(args.siesta_run_fdf)
        if snapshot_temperature is not None:
            manifest["snapshot_temperature"] = snapshot_temperature
        manifest["input_hashes"] = input_hashes(
            {
                "siesta_bands": args.siesta_bands,
                "siesta_band_data": args.siesta_band_data,
                "siesta_run_fdf": args.siesta_run_fdf,
                "graph2mat_prediction": args.graph2mat_prediction,
                "graph2mat_reference": args.graph2mat_reference,
                "graph2mat_band_data": args.graph2mat_band_data,
                "deeph_prediction": args.deeph_prediction,
                "deeph_overlap": args.deeph_overlap,
                "deeph_band_data": args.deeph_band_data,
                "deeph_equivalence_evidence": args.deeph_equivalence_evidence,
            }
        )
        if args.dry_run:
            manifest["status"] = "dry_run"
            write_json(args.output_dir / "manifest.json", manifest)
            print(json.dumps({"status": "dry_run", "output_dir": str(args.output_dir)}, indent=2))
            return 0

        methods = build_methods(args, kpoints, args.output_dir, manifest)
        if not methods:
            raise RuntimeError("No band inputs were provided.")
        siesta = next((method for method in methods if method.method == "SIESTA"), None)
        if siesta is None:
            raise RuntimeError("A SIESTA reference band source is required.")
        zero_policy, zero_value = normalize_energy_zero(
            methods,
            policy=args.energy_zero,
            fermi_level=args.fermi_level,
            custom_zero=args.custom_energy_zero,
            fail_closed=args.fail_closed,
            warnings=manifest["warnings"],
        )
        manifest["energy_zero_policy"] = zero_policy
        manifest["fermi_level_eV"] = zero_value if zero_policy == "fermi" else None
        band_fieldnames = [
            "method",
            "sample_id",
            "k_index",
            "k_distance",
            "kx",
            "ky",
            "kz",
            "k_label",
            "segment",
            "band_index",
            "energy_eV",
            "energy_aligned_eV",
            "fermi_level_eV",
            "energy_zero_policy",
        ]
        for method in methods:
            output_name = f"bands_{method.method.lower()}.csv"
            write_csv(args.output_dir / output_name, band_rows(method, kpoints, zero_policy, zero_value), band_fieldnames)
            manifest["outputs"][output_name] = str(args.output_dir / output_name)

        error_fieldnames = [
            "method",
            "sample_id",
            "k_index",
            "k_distance",
            "band_index",
            "siesta_energy_eV",
            "predicted_energy_eV",
            "error_eV",
            "abs_error_eV",
            "squared_error_eV",
            "k_label",
            "segment",
        ]
        summaries: dict[str, Any] = {
            "sample_id": args.sample_id,
            "kpath": kpath_name,
            "labels": [label_for_display(node.label) for node in nodes],
            "n_kpoints": len(kpoints),
            "n_bands_siesta": siesta.n_bands(),
        }
        for method in methods:
            key = method.method.lower()
            summaries[f"n_bands_{key}"] = method.n_bands()
            manifest[f"{key}_status"] = {
                "scientific_status": method.scientific_status,
                "overlap_used": method.overlap_used,
                "diagonalization_errors": method.diagonalization_errors,
                "hermiticity_defect_max": max(method.hermiticity_defects, default=None),
                "warnings": method.warnings or [],
            }
            if method.method == "SIESTA":
                continue
            rows = error_rows(method, siesta, kpoints, zero_policy, zero_value)
            output_name = f"band_errors_{key}.csv"
            write_csv(args.output_dir / output_name, rows, error_fieldnames)
            manifest["outputs"][output_name] = str(args.output_dir / output_name)
            summaries[key] = {
                **metric_summary(rows),
                "scientific_status": method.scientific_status,
            }
        dirac_diagnostics = compute_dirac_diagnostics(methods, siesta, kpoints, args, zero_value if zero_policy == "fermi" else None)
        summaries["dirac_diagnostic"] = dirac_diagnostics
        manifest["dirac_diagnostic"] = dirac_diagnostics
        write_json(args.output_dir / "dirac_diagnostic.json", dirac_diagnostics)
        manifest["outputs"]["dirac_diagnostic.json"] = str(args.output_dir / "dirac_diagnostic.json")
        write_json(args.output_dir / "band_summary.json", summaries)
        manifest["outputs"]["band_summary.json"] = str(args.output_dir / "band_summary.json")
        if not args.skip_plot:
            annotation_lines: list[str] = []
            if snapshot_temperature is not None:
                if "value" in snapshot_temperature:
                    annotation_lines.append(f"Snapshot T = {snapshot_temperature['value']:g} {snapshot_temperature.get('unit', 'K')}")
                elif "raw" in snapshot_temperature:
                    annotation_lines.append(f"Snapshot T = {snapshot_temperature['raw']}")
            ref_diag = dirac_diagnostics.get("methods", {}).get("siesta", {})
            if ref_diag.get("status") == "ok":
                gap_mev = float(ref_diag["gap_eV"]) * 1000.0
                shift = ref_diag.get("dirac_minus_fermi_eV")
                annotation_lines.append(f"SIESTA K gap = {gap_mev:.1f} meV")
                if shift is not None:
                    annotation_lines.append(f"SIESTA E_D - E_F = {float(shift):+.3f} eV")
            if dirac_diagnostics.get("warnings"):
                manifest["warnings"].extend(f"Dirac diagnostic: {warning}" for warning in dirac_diagnostics["warnings"])
            manifest["outputs"].update(
                plot_comparison(
                    args.output_dir,
                    methods,
                    siesta,
                    kpoints,
                    zero_policy,
                    zero_value,
                    args,
                    annotation_lines=annotation_lines,
                )
            )
            dirac_offset = dirac_diagnostics.get("reference_dirac_minus_fermi_eV")
            if not args.no_dirac_aligned_plot and dirac_offset is not None:
                dirac_annotation = [*annotation_lines, "Energy zero = SIESTA K Dirac midpoint"]
                manifest["outputs"].update(
                    plot_comparison(
                        args.output_dir,
                        methods,
                        siesta,
                        kpoints,
                        zero_policy,
                        zero_value,
                        args,
                        output_stem="band_comparison_dirac_aligned",
                        y_offset_eV=float(dirac_offset),
                        ylabel="Energy relative to SIESTA Dirac midpoint (eV)",
                        title_suffix="\nEnergy zero shifted to SIESTA K Dirac midpoint",
                        annotation_lines=dirac_annotation,
                    )
                )
            if args.method_dirac_gauge_aligned_plot and ref_diag.get("status") == "ok":
                reference_shift = ref_diag.get("dirac_minus_fermi_eV")
                method_offsets: dict[str, float] = {}
                if reference_shift is not None:
                    for method_name, payload in dirac_diagnostics.get("methods", {}).items():
                        method_shift = payload.get("dirac_minus_fermi_eV")
                        if method_shift is not None:
                            display_name = next(
                                (method.method for method in methods if method.method.lower() == method_name),
                                method_name,
                            )
                            method_offsets[display_name] = float(method_shift) - float(reference_shift)
                gauge_annotation = [
                    *annotation_lines,
                    "Prediction gauges shifted to SIESTA K Dirac midpoint",
                ]
                manifest["method_dirac_gauge_offsets_eV"] = method_offsets
                manifest["outputs"].update(
                    plot_comparison(
                        args.output_dir,
                        methods,
                        siesta,
                        kpoints,
                        zero_policy,
                        zero_value,
                        args,
                        output_stem="band_comparison_method_dirac_gauge_aligned",
                        method_offsets_eV=method_offsets,
                        ylabel="Energy relative to Fermi after Dirac-gauge alignment (eV)"
                        if zero_policy == "fermi"
                        else "Energy after Dirac-gauge alignment (eV)",
                        title_suffix="\nPredictions shifted to SIESTA K Dirac midpoint",
                        annotation_lines=gauge_annotation,
                    )
                )
        else:
            manifest["outputs"]["plot"] = "skipped"
        manifest["status"] = "completed"
        write_json(args.output_dir / "manifest.json", manifest)
        print(json.dumps({"status": "completed", "output_dir": str(args.output_dir)}, indent=2))
        return 0
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["fatal_errors"].append(str(exc))
        write_json(args.output_dir / "manifest.json", manifest)
        print(json.dumps({"status": "failed", "error": str(exc), "output_dir": str(args.output_dir)}, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
