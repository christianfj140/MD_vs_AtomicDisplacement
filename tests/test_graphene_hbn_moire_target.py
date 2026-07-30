"""Graphene-hBN twisted moire target geometry builder."""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "Comparison" / "scripts"
SHARED = REPO_ROOT / "shared"
for path in (SCRIPTS, SHARED):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import build_graphene_hbn_moire_target as moire  # noqa: E402
from fdf_materialization import extract_fdf_structure  # noqa: E402
from g2m_deeph_training_sweep import expand_training_sweep  # noqa: E402
from run_graphene_hbn_moire_spectral_campaign import (  # noqa: E402
    DEFAULT_CONFIG as SPECTRAL_CONFIG,
    build_overlap,
    disk_headroom,
    load_config as load_spectral_config,
    process_resource_blocked,
    spectrum_tiers,
    training_source_dataset,
    training_source_payload,
    training_payload,
)


STACKING_FDF = REPO_ROOT / "materials" / "bilayer_graphene_hBN_AA" / "RUN.fdf"


def test_commensurate_angle_matches_known_values() -> None:
    # (1,2) -> ~21.79 deg is the standard first commensurate hexagonal twist.
    assert math.isclose(moire.commensurate_angle_degrees(1, 2), 21.7867892, abs_tol=1e-4)
    # (1,3) -> ~32.2 deg.
    assert math.isclose(moire.commensurate_angle_degrees(1, 3), 32.2042469, abs_tol=1e-4)
    with pytest.raises(RuntimeError):
        moire.commensurate_angle_degrees(2, 2)


def test_moire_geometry_atom_count_and_species(tmp_path: Path) -> None:
    text, metadata = moire.moire_geometry(STACKING_FDF, approximant=2, m=1, n=2)
    assert metadata["num_atoms"] == 6 * (1 * 1 + 1 * 2 + 2 * 2) == 42
    assert metadata["expected_orbitals"] == 294
    assert metadata["species_counts"] == {"C": 28, "B": 7, "N": 7}
    assert metadata["twisted_sublayer"] == "top_graphene"
    assert metadata["aligned_sublayers"] == ["hBN", "bottom_graphene"]
    assert metadata["reference_hamiltonian_available"] is False
    assert metadata["scientific_scope"] == "rigid strained commensurate geometry"
    assert math.isclose(metadata["materialized_twist_angle_deg"], metadata["twist_angle_deg"])
    expected_strain = (
        metadata["geometry_inplane_lattice_ang"] / metadata["native_hBN_lattice_ang"] - 1.0
    ) * 100.0
    assert math.isclose(metadata["effective_hBN_strain_percent"], expected_strain)
    assert metadata["minimum_periodic_atom_distance_ang"] > 1.2
    out = tmp_path / "RUN.fdf"
    out.write_text(text, encoding="utf-8")
    parsed = extract_fdf_structure(out, structure_type="crystal")
    assert parsed.atom_count == 42
    labels = {sp.label for sp in parsed.species}
    assert labels == {"C", "B", "N"}
    label_by_index = {sp.index: sp.label for sp in parsed.species}
    counts = {label: 0 for label in labels}
    for atom in parsed.atoms:
        counts[label_by_index[atom.species_index]] += 1
    assert counts == {"C": 28, "B": 7, "N": 7}
    assert "MD.TypeOfRun" not in text and "Lua.Script" not in text


def test_kgrid_is_divided_by_approximant_to_match_density(tmp_path: Path) -> None:
    # The index-7 cell is sqrt(7) larger in-plane; 20/sqrt(7) rounds to 8.
    text, _ = moire.moire_geometry(STACKING_FDF, approximant=2, m=1, n=2)
    block = text[text.index("%block kgrid_Monkhorst_Pack"):text.index("%endblock kgrid_Monkhorst_Pack")]
    rows = [line.split() for line in block.splitlines() if line.split() and line.split()[0].lstrip("-").isdigit()]
    assert int(rows[0][0]) == 8
    assert int(rows[1][1]) == 8
    assert int(rows[2][2]) == 1


def test_dry_run_never_invokes_siesta(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(moire, "run_siesta", lambda *_a, **_k: pytest.fail("SIESTA invoked"))
    plan = moire.dry_run_plan(STACKING_FDF, approximant=2, m=1, n=2, limit=1)
    assert plan["siesta_invoked"] is False
    assert plan["num_atoms"] == 42
    assert math.isclose(plan["twist_angle_deg"], 21.7867892, abs_tol=1e-4)
    assert plan["minimum_periodic_atom_distance_ang"] > 1.2


def test_geometry_only_writes_predict_manifest_without_siesta(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import csv

    monkeypatch.setattr(moire, "run_siesta", lambda *_a, **_k: pytest.fail("SIESTA invoked"))
    result = moire.build_geometry_only(
        STACKING_FDF, tmp_path / "moire_geom", approximant=2, m=1, n=2, overwrite=True
    )
    assert result["siesta_invoked"] is False
    assert result["reference_available"] is False
    assert result["num_atoms"] == 42
    manifest = tmp_path / "moire_geom" / "splits" / "test_manifest.csv"
    rows = list(csv.DictReader(manifest.open(encoding="utf-8")))
    assert len(rows) == 1
    row = rows[0]
    # Predict path needs only sample_id + a real structure; reference stays empty.
    assert row["sample_id"] == "moire_0"
    assert row["hamiltonian_path"] == ""
    assert Path(row["structure_path"]).is_file()
    assert extract_fdf_structure(Path(row["structure_path"]), structure_type="crystal").atom_count == 42


def test_magic_angle_geometry_has_expected_trilayer_size() -> None:
    text, metadata = moire.moire_geometry(STACKING_FDF, approximant=2, m=31, n=30)
    assert metadata["commensurate_cell_index"] == 2791
    assert metadata["num_atoms"] == 16746
    assert metadata["expected_orbitals"] == 117222
    assert metadata["species_counts"] == {"C": 11164, "B": 2791, "N": 2791}
    assert math.isclose(metadata["twist_angle_deg"], 1.08454905, abs_tol=1e-6)
    assert re.search(r"^NumberOfAtoms\s+16746$", text, re.MULTILINE)


def test_minimum_distance_guard_rejects_overlap_and_accepts_valid_geometry() -> None:
    lattice = moire.np.diag([10.0, 10.0, 10.0])
    assert math.isclose(
        moire.validate_minimum_atom_distance(moire.np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]), lattice),
        1.5,
    )
    with pytest.raises(RuntimeError, match="minimum periodic atom distance"):
        moire.validate_minimum_atom_distance(
            moire.np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]), lattice
        )


def test_moire_basis_hashes_match_bilayer_train_presets() -> None:
    # The moire must share basis with the flat stackings for the planner to accept it.
    from material_presets import resolve_material_bundle

    stacking = resolve_material_bundle({"material": {"preset": "bilayer_graphene_hBN_AA"}}).validated
    for basis in (moire.COMMON_MATERIAL_ROOT / "basis").glob("*.ion.xml"):
        assert basis.name in stacking.basis_file_sha256


def test_spectral_campaign_runs_seed_zero_smoke_and_retains_three_seed_plan() -> None:
    config = load_spectral_config(SPECTRAL_CONFIG)
    payload = training_payload(config, Path("/tmp/spectral-campaign-test"), 30)
    plan = expand_training_sweep(
        payload["training_sweep"],
        datasets=[{"dataset_id": "n30", "dataset_root": "/tmp/n30"}],
    )
    assert config["model_seeds"] == [0, 1, 2]
    assert config["active_model_seeds"] == [0]
    assert len(plan["planned_runs"]) == 2
    assert {row["model"] for row in plan["planned_runs"]} == {"graph2mat", "deeph"}
    assert {
        row["overrides"].get("seed_everything", row["overrides"].get("seed"))
        for row in plan["planned_runs"]
    } == {0}
    assert "system_label" not in payload
    assert payload["reuse_run_root"] is True
    assert payload["early_stopping"] == {
        "metric": "val_loss",
        "mode": "min",
        "patience": 150,
        "min_delta": 1e-05,
        "max_epochs": 750,
    }
    assert payload["performance"]["graph2mat_min_free_gpu_memory_mb"] == 18000
    assert payload["performance"]["deeph_min_free_gpu_memory_mb"] == 24000


def test_spectral_campaign_recognizes_resource_exhaustion() -> None:
    assert process_resource_blocked(subprocess.CompletedProcess([], -9, "", ""))
    assert process_resource_blocked(
        subprocess.CompletedProcess([], 1, "", "CUDA out of memory")
    )
    assert not process_resource_blocked(subprocess.CompletedProcess([], 1, "", "bad input"))


def test_spectral_campaign_keeps_disk_buffer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    usage = type("Usage", (), {"total": 100, "free": 11})
    monkeypatch.setattr("run_graphene_hbn_moire_spectral_campaign.shutil.disk_usage", lambda _: usage)
    assert disk_headroom(tmp_path)["status"] == "resource_blocked"
    usage.free = 13
    assert disk_headroom(tmp_path)["status"] == "safe"


def test_overlap_resume_promotes_valid_sparse_positivity_gate(tmp_path: Path) -> None:
    output = tmp_path / "overlap"
    output.mkdir()
    overlap_h5 = output / "overlaps.h5"
    overlap_h5.touch()
    (output / "overlap_manifest.json").write_text(
        json.dumps({"status": "completed", "export": {"overlaps_h5": str(overlap_h5)}}),
        encoding="utf-8",
    )
    sparse = {
        "status": "valid",
        "no_identity_overlap": True,
        "kpoints": [
            {
                "label": label,
                "positivity_status": "validated_sparse_arpack",
                "minimum_eigenvalue": 1e-4,
            }
            for label in ("Gamma", "K", "M")
        ],
    }
    (output / "diagnostics_sparse.json").write_text(json.dumps(sparse), encoding="utf-8")

    result = build_overlap({}, tmp_path, resume=True)

    assert result["manifest"]["diagnostics"] == sparse
    assert result["manifest"]["sparse_positivity_gate"]["status"] == "validated"


def test_spectral_tiers_limit_expensive_magic_angle_solves() -> None:
    config = load_spectral_config(SPECTRAL_CONFIG)
    ordinary = {"training_size": 240, "seed": 2}
    representative = {"training_size": 480, "seed": 0}
    assert spectrum_tiers(config, ordinary, "bands") == ["tier_a"]
    assert spectrum_tiers(config, representative, "bands") == ["tier_b"]
    assert spectrum_tiers(config, ordinary, "dos") == []
    assert spectrum_tiers(config, representative, "dos") == ["tier_c"]


def test_spectral_training_source_uses_runner_dataset_slug() -> None:
    config = load_spectral_config(SPECTRAL_CONFIG)
    payload = training_source_payload(
        config,
        Path("/tmp/spectral-campaign-test"),
        "bilayer_graphene_hBN_AA",
        150,
    )
    assert training_source_dataset(payload).name == "bilayer_graphene_hbn_aa_t150_master60"
