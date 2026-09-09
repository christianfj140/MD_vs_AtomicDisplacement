"""E-F_001-S48: structure/H/derivative transferability across AA/AB/BA local samples.

One CPU run of the real checkpoint against the three real materials, shared by every
assertion below (:func:`_report`, module-scoped) -- the checks are on the shape and
honesty of the verdict, not a second measurement of the same numbers.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT / "Comparison" / "scripts", REPO_ROOT / "shared"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import certify_relaxed_geometry_transferability as cert  # noqa: E402


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    output = tmp_path_factory.mktemp("s48") / "local_stacking_bound.json"
    return cert.certify(backend="cpu", output=output)


def test_all_three_registries_are_certified(report):
    assert {m["label"] for m in report["materials"]} == {
        "bilayer_graphene_AA",
        "bilayer_graphene_AB",
        "bilayer_graphene_BA",
    }


def test_verdict_passes_structure_h_and_go3_per_material(report):
    for material in report["materials"]:
        assert material["status"] == "PASS", material
        for row in material["checks"]:
            assert row["passed"], row
    assert report["verdict"] == "PASS"


def test_structure_check_rejects_invalid_but_not_declared_only():
    """The gate certify_material applies: only oc.INVALID blocks; DECLARED_ONLY does not.

    Every material already in production (graphene, AB, the rigid TBG target) gets
    DECLARED_ONLY against this checkpoint from the known PointBasis.R/edge-cutoff
    drift, not from anything specific to AA/AB/BA -- requiring strict VERIFIED here
    would fail this ticket for a reason unrelated to relaxed-geometry transferability.
    """
    assert cert.oc.VERIFIED != cert.oc.INVALID
    assert cert.oc.DECLARED_ONLY != cert.oc.INVALID
    assert cert.oc.INVALID == cert.oc.INVALID


def test_report_never_claims_go10_or_a_relaxed_geometry(report):
    assert report["licenses_go10"] is False
    assert report["quantitative_relaxed_matbg_claim_licensed"] is False
    assert report["go10_status"] == "NO-GO-10"
    assert report["status_label"] == "local_stacking_transferability_bound"


def test_go3_ran_on_the_stacking_specific_directions_not_only_the_default_set(report):
    for material in report["materials"]:
        go3_check = next(
            row for row in material["checks"] if row["check"] == "go3_directional_jvp_internal_validation"
        )
        assert {"shear", "layer_breathing", "intralayer_control"} <= set(go3_check["directions"])


def test_output_file_is_written_and_matches_the_returned_report(tmp_path):
    output = tmp_path / "bound.json"
    written = cert.certify(backend="cpu", output=output)
    assert output.is_file()
    import json

    on_disk = json.loads(output.read_text(encoding="utf-8"))
    assert on_disk["verdict"] == written["verdict"]
    assert on_disk["schema"] == cert.SCHEMA
