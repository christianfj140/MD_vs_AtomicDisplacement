"""Tests for DATASET-DESIGN-W90-001-S9-S3 (design generation + validation)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "Comparison" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import dataset_design_w90_001_s9_s2_envelope_and_coverage as s2  # noqa: E402
import dataset_design_w90_001_s9_s3_design_generation as s3g  # noqa: E402
import w90_displacement_sampler_family as sampler  # noqa: E402

# "nested_verified"/"nested_reason" are added by build_design_manifest's
# cross-design annotation pass, not by build_manifest_entry in isolation.
REQUIRED_ENTRY_KEYS = {
    "family", "dim", "k", "r_train_max_ang", "density_level", "seeds",
    "n_generated", "n_used", "geometry_hash", "envelope_check_passed",
    "dof", "correlation_summary", "locality_indicator_participation_ratio_mean",
    "diagnostics",
}


@pytest.fixture(scope="module")
def geometry():
    return sampler.load_graphene_primitive()


@pytest.fixture(scope="module")
def external_hashes(geometry):
    return s3g.external_reference_hashes(geometry)


def _one_row_per_family(rows):
    seen = set()
    subset = []
    for row in rows:
        if row["family"] not in seen:
            seen.add(row["family"])
            subset.append(row)
    return subset


def test_applicable_rows_excludes_not_applicable_and_pruned_states():
    rows = applicable = s3g.applicable_rows()
    assert applicable
    assert all(row["state"] in ("pending", "executed") for row in rows)
    all_states = {row["state"] for row in s2.build_coverage_matrix()}
    assert "not_applicable" in all_states and "pruned" in all_states  # sanity: filter had something to remove


def test_applicable_rows_includes_latin_hypercube_with_at_least_three_density_levels():
    rows = s3g.applicable_rows()
    lhs_levels = {row["resolution"] for row in rows if row["family"] == "latin_hypercube"}
    assert len(lhs_levels) >= 3


def test_manifest_entry_schema_and_envelope_for_one_row_per_family(geometry, external_hashes):
    rows = _one_row_per_family(s3g.applicable_rows())
    assert {row["family"] for row in rows} == set(s2.FAMILIES)
    siesta_hashes: set[str] = set()
    for row in rows:
        entry, _point_hashes = s3g.build_manifest_entry(
            geometry, row, external_hashes=external_hashes, siesta_hashes=siesta_hashes
        )
        assert REQUIRED_ENTRY_KEYS.issubset(entry)
        assert entry["envelope_check_passed"] is True
        assert entry["envelope_max_displacement_ang"] <= entry["r_train_max_ang"] + 1e-9
        assert entry["n_generated"] > 0
        assert entry["n_used"] <= entry["n_generated"]
        assert entry["dof"] == entry["k"] * len(sampler.dimensionality_axes(entry["dim"]))


def test_diagnostics_are_finite_and_physically_reasonable(geometry):
    row = next(r for r in s3g.applicable_rows() if r["family"] == "sobol_sparse" and r["dim"] == "3D" and r["k"] == 1)
    entry, _ = s3g.build_manifest_entry(geometry, row, external_hashes=set(), siesta_hashes=set())
    diagnostics = entry["diagnostics"]

    assert sum(diagnostics["radial_histogram"]["counts"]) == entry["n_generated"]
    assert sum(diagnostics["angular_histogram"]["counts"]) == entry["n_generated"]
    assert diagnostics["min_nearest_neighbor_distance_ang"] > 0.0
    assert diagnostics["covering_radius_ang"] > 0.0
    import math
    assert math.isfinite(diagnostics["min_nearest_neighbor_distance_ang"])
    assert math.isfinite(diagnostics["covering_radius_ang"])


def test_no_geometry_hash_overlap_with_frozen_or_validation_sets(geometry, external_hashes):
    assert external_hashes, "expected at least one external (frozen-test/common_validation) geometry hash"
    rows = _one_row_per_family(s3g.applicable_rows())
    for row in rows:
        entry, _ = s3g.build_manifest_entry(geometry, row, external_hashes=external_hashes, siesta_hashes=set())
        assert entry["n_overlap_with_frozen_or_validation"] == 0
        assert entry["n_used"] == entry["n_generated"]


def test_axial_radial_density_levels_are_nested_by_hash_subset(geometry):
    rows = [
        r for r in s3g.applicable_rows()
        if r["family"] == "axial_radial" and r["dim"] == "3D" and r["k"] == 1
        and abs(r["domain_r_train_max_ang"] - 0.08) < 1e-9
    ]
    rows.sort(key=lambda r: r["resolution"])
    manifest = s3g.build_design_manifest(geometry=geometry, coverage_matrix=rows)
    designs = sorted(manifest["designs"], key=lambda e: e["density_level"])
    assert len(designs) >= 2
    for entry in designs:
        assert entry["nested_verified"] is True


def test_latin_hypercube_is_documented_non_nested(geometry):
    rows = [
        r for r in s3g.applicable_rows()
        if r["family"] == "latin_hypercube" and r["dim"] == "3D" and r["k"] == 1
        and abs(r["domain_r_train_max_ang"] - 0.08) < 1e-9
    ]
    manifest = s3g.build_design_manifest(geometry=geometry, coverage_matrix=rows)
    assert manifest["designs"]
    for entry in manifest["designs"]:
        assert entry["nested_verified"] is False
        assert "not a nested sequence" in entry["nested_reason"].lower()


def test_local_pair_modes_dedupes_geometrically_identical_bond_frame_and_lab_axis_modes(geometry):
    """transverse_opp (bond-frame) and antiparallel-z (lab-axis) are the same

    geometry for graphene's planar primitive cell (both reduce to +/-z_hat);
    must be deduplicated within the design, not double-counted or left as a
    zero-distance "nearest neighbor" in the coverage diagnostics.
    """

    row = next(
        r for r in s3g.applicable_rows()
        if r["family"] == "local_pair_modes" and r["dim"] in ("1D_z", "3D") and int(r["k"]) == 2
    )
    entry, _ = s3g.build_manifest_entry(geometry, row, external_hashes=set(), siesta_hashes=set())
    assert entry["n_duplicate_within_design"] > 0
    assert entry["n_used"] == entry["n_generated"] - entry["n_duplicate_within_design"]
    assert entry["diagnostics"]["min_nearest_neighbor_distance_ang"] > 0.0
    assert entry["diagnostics"]["covering_radius_ang"] > 0.0


def test_full_campaign_manifest_has_no_zero_or_nonfinite_diagnostics(geometry):
    """Acceptance criterion: coverage diagnostics finite/reasonable for every

    applicable coverage-matrix cell, not just a spot-checked subset -- runs
    the full 792-design campaign (~6s, pure numpy, no SIESTA).
    """

    import math

    manifest = s3g.build_design_manifest(geometry=geometry)
    assert manifest["leakage_free"] is True
    assert manifest["envelope_all_passed"] is True
    for entry in manifest["designs"]:
        diagnostics = entry["diagnostics"]
        for key in ("min_nearest_neighbor_distance_ang", "covering_radius_ang"):
            value = diagnostics[key]
            assert math.isfinite(value) and value > 0.0, (entry["design_id"], key, value)


def test_correlation_summary_present_only_for_k2_designs(geometry):
    rows = s3g.applicable_rows()
    k1_row = next(r for r in rows if r["family"] == "sobol_sparse" and r["k"] == 1)
    k2_row = next(r for r in rows if r["family"] == "sobol_sparse" and r["k"] == 2)
    k1_entry, _ = s3g.build_manifest_entry(geometry, k1_row, external_hashes=set(), siesta_hashes=set())
    k2_entry, _ = s3g.build_manifest_entry(geometry, k2_row, external_hashes=set(), siesta_hashes=set())
    assert k1_entry["correlation_summary"] is None
    assert k2_entry["correlation_summary"] is not None
    assert {"mean", "std"}.issubset(k2_entry["correlation_summary"])


def test_write_manifest_json_round_trips_and_is_valid(tmp_path, geometry):
    rows = _one_row_per_family(s3g.applicable_rows())
    out = s3g.write_manifest_json(tmp_path / "design_manifest.json", geometry=geometry, coverage_matrix=rows)
    import json

    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["n_designs"] == len(rows)
    assert loaded["leakage_free"] is True
    assert loaded["envelope_all_passed"] is True
    for entry in loaded["designs"]:
        assert REQUIRED_ENTRY_KEYS.issubset(entry)
        assert "nested_verified" in entry


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
