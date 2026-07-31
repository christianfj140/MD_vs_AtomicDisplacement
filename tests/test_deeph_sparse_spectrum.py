import json
import os
import sys
import time
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "Comparison" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import run_deeph_sparse_spectrum as sparse_spectrum  # noqa: E402
from run_moire_projected_followup import validated_projection  # noqa: E402
from run_graphene_hbn_moire_spectral_campaign import projected_resolution_stability  # noqa: E402
from run_deeph_sparse_spectrum import (  # noqa: E402
    band_k_data,
    estimated_neutrality_reference,
    kprime_comparison,
    mulliken_projection_groups,
    projected_band_data,
    projected_dos_data,
    projected_dos_observables,
    select_band_path,
)


def test_band_k_data_supports_single_high_symmetry_points() -> None:
    assert band_k_data(1) == [
        "1 0.0 0.0 0.0 0.0 0.0 0.0",
        "1 0.3333333333333333 0.6666666666666666 0.0 0.3333333333333333 0.6666666666666666 0.0",
        "1 0.5 0.5 0.0 0.5 0.5 0.0",
    ]
    assert len(band_k_data(2)) == 4
    detailed = band_k_data(8)
    assert len(detailed) == 22
    assert len(set(detailed)) == 21
    assert detailed[0] == detailed[-1]
    assert detailed.count(
        "1 0.3333333333333333 0.6666666666666666 0.0 0.3333333333333333 0.6666666666666666 0.0"
    ) == 1
    assert detailed.count("1 0.5 0.5 0.0 0.5 0.5 0.0") == 1
    with pytest.raises(ValueError, match="positive"):
        band_k_data(0)
    assert band_k_data(8, gamma_only=True) == ["1 0.0 0.0 0.0 0.0 0.0 0.0"]


def test_runtime_disk_guard_terminates_process(tmp_path, monkeypatch) -> None:
    readings = iter((100.0, 11.0))
    monkeypatch.setattr(sparse_spectrum, "free_disk_percent", lambda _path: next(readings, 11.0))

    started = time.monotonic()
    returncode, reason, minimum, _temperature, _memory = sparse_spectrum.run_with_disk_guard(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        cwd=tmp_path,
        env=os.environ.copy(),
        output_dir=tmp_path,
        poll_seconds=0.01,
    )

    assert reason == "disk_headroom_runtime"
    assert returncode != 0
    assert minimum == 11.0
    assert time.monotonic() - started < 5


def test_runtime_temperature_guard_terminates_process(tmp_path, monkeypatch) -> None:
    readings = iter((55.0, 91.0))
    monkeypatch.setattr(sparse_spectrum, "free_disk_percent", lambda _path: 100.0)
    monkeypatch.setattr(
        sparse_spectrum,
        "cpu_package_temperature_c",
        lambda: next(readings, 91.0),
    )

    returncode, reason, minimum, maximum, _memory = sparse_spectrum.run_with_disk_guard(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        cwd=tmp_path,
        env=os.environ.copy(),
        output_dir=tmp_path,
        poll_seconds=0.01,
    )

    assert returncode != 0
    assert reason == "cpu_temperature"
    assert minimum == 100.0
    assert maximum == 91.0


def test_runtime_memory_guard_terminates_process(tmp_path, monkeypatch) -> None:
    readings = iter((16, 7))
    monkeypatch.setattr(sparse_spectrum, "free_disk_percent", lambda _path: 100.0)
    monkeypatch.setattr(
        sparse_spectrum,
        "available_memory_bytes",
        lambda: next(readings, 7) * sparse_spectrum.GIB,
    )

    returncode, reason, _disk, _temperature, minimum = sparse_spectrum.run_with_disk_guard(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        cwd=tmp_path,
        env=os.environ.copy(),
        output_dir=tmp_path,
        poll_seconds=0.01,
    )

    assert returncode != 0
    assert reason == "memory_headroom_runtime"
    assert minimum == 7 * sparse_spectrum.GIB


def test_runtime_gpu_temperature_guard_terminates_process(tmp_path, monkeypatch) -> None:
    readings = iter(
        (
            {"temperature_c": 40.0, "used_bytes": 0, "free_bytes": 32 * sparse_spectrum.GIB},
            {"temperature_c": 81.0, "used_bytes": 0, "free_bytes": 32 * sparse_spectrum.GIB},
        )
    )
    monkeypatch.setattr(sparse_spectrum, "free_disk_percent", lambda _path: 100.0)
    monkeypatch.setattr(sparse_spectrum, "available_memory_bytes", lambda: 32 * sparse_spectrum.GIB)
    monkeypatch.setattr(sparse_spectrum, "cpu_package_temperature_c", lambda: 40.0)
    monkeypatch.setattr(sparse_spectrum, "gpu_status", lambda: next(readings, None))

    returncode, reason, *_rest = sparse_spectrum.run_with_disk_guard(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        cwd=tmp_path,
        env=os.environ.copy(),
        output_dir=tmp_path,
        poll_seconds=0.01,
        gpu_memory_limit_bytes=28 * sparse_spectrum.GIB,
    )

    assert returncode != 0
    assert reason == "gpu_temperature"


def test_requested_gpu_backend_is_recorded_without_cpu_fallback(tmp_path, monkeypatch) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    for name in ("hamiltonians_pred.h5", "overlaps.h5", "rlat.dat", "site_positions.dat", "orbital_types.dat"):
        (input_dir / name).touch()
    environment = tmp_path / "environment.json"
    environment.write_text(json.dumps({"status": "valid"}), encoding="utf-8")
    monkeypatch.setattr(sparse_spectrum, "free_disk_percent", lambda _path: 100.0)
    monkeypatch.setattr(sparse_spectrum, "available_memory_bytes", lambda: 40 * sparse_spectrum.GIB)
    monkeypatch.setattr(sparse_spectrum, "gpu_status", lambda: None)

    result = sparse_spectrum.run(
        input_dir,
        tmp_path / "output",
        job="band",
        fermi_level=0.0,
        num_bands=2,
        points_per_segment=1,
        kmesh=(1, 1, 1),
        environment_path=environment,
        backend="gpu_cudss",
    )

    assert result["status"] == "resource_blocked"
    assert result["backend_requested"] == "gpu_cudss"
    assert result["backend_effective"] is None


def test_band_tracking_uses_adjacent_eigenvector_overlap(tmp_path) -> None:
    energies = sparse_spectrum.np.asarray([[0.0, 1.0, 2.0], [1.1, 2.1, 0.1]])
    sparse_spectrum.write_json(
        tmp_path / "band_tracking_000.json",
        {"k_index": 0, "energies_eV": energies[0].tolist(), "overlap_from_previous": None},
    )
    sparse_spectrum.write_json(
        tmp_path / "band_tracking_001.json",
        {
            "k_index": 1,
            "energies_eV": energies[1].tolist(),
            "overlap_from_previous": [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
        },
    )

    tracked, metadata = sparse_spectrum.track_bands_by_overlap(tmp_path, energies)

    assert tracked.tolist() == [[0.0, 1.0, 2.0], [0.1, 1.1, 2.1]]
    assert metadata["minimum_assigned_overlap"] == 1.0


def test_band_energies_are_sorted_at_each_k() -> None:
    energies = sparse_spectrum.np.asarray([[2.0, 0.0, 1.0], [-1.0, 3.0, 2.0]])
    assert sparse_spectrum.sort_bands_by_energy(energies).tolist() == [
        [0.0, 1.0, 2.0],
        [-1.0, 2.0, 3.0],
    ]


def test_hexagonal_path_is_selected_from_the_actual_reciprocal_metric() -> None:
    primitive_direct = sparse_spectrum.np.asarray(
        [[2.48, 0.0, 0.0], [-1.24, sparse_spectrum.np.sqrt(3.0) * 1.24, 0.0], [0, 0, 20]]
    )
    primitive_reciprocal = 2 * sparse_spectrum.np.pi * sparse_spectrum.np.linalg.inv(primitive_direct).T
    primitive_path, primitive_check = sparse_spectrum.validated_hexagonal_k_path(primitive_reciprocal)
    assert primitive_check["reciprocal_angle_deg"] == pytest.approx(60.0)
    assert primitive_path[1][1] == (1 / 3, 1 / 3, 0.0)
    moire_matrix = sparse_spectrum.np.asarray([[61, 31, 0], [30, 61, 0], [0, 0, 1]])
    moire_reciprocal = 2 * sparse_spectrum.np.pi * sparse_spectrum.np.linalg.inv(
        moire_matrix @ primitive_direct
    ).T
    moire_path, moire_check = sparse_spectrum.validated_hexagonal_k_path(moire_reciprocal)
    assert moire_check["reciprocal_angle_deg"] == pytest.approx(120.0)
    assert moire_path[1][1] == (1 / 3, 2 / 3, 0.0)
    open_path, _validation = select_band_path(moire_reciprocal, "k-gamma-m")
    assert [label for label, _point in open_path] == ["K", "Γ", "M"]


def test_magic_angle_mulliken_groups_prove_pz_and_partition_the_basis() -> None:
    input_dir = (
        sparse_spectrum.REPO_ROOT
        / "Comparison/results/graphene_hbn_magic_angle_spectral/predictions/graph2mat/n480/seed0/solver_input"
    )
    if not input_dir.is_dir():
        pytest.skip("local N=480 solver input is unavailable")
    payload = mulliken_projection_groups(input_dir)
    groups = payload["groups"]
    mapping = payload["mapping"]
    assert mapping["carbon_pz_local_indices_zero_based"] == [2]
    assert mapping["basis_convention"] == "siesta_spherical=(-py,pz,-px) for l=1"
    assert len(groups["carbon_pz"]) == mapping["species_atom_counts"]["C"]
    assert len(groups["carbon"]) == len(groups["graphene_lower"]) + len(groups["graphene_upper"])
    assert len(groups["carbon"]) + len(groups["boron"]) + len(groups["nitrogen"]) == mapping["total_orbitals"]
    assert mapping["group_orbital_counts"]["carbon_pz"] == mapping["species_atom_counts"]["C"]
    neutrality = estimated_neutrality_reference(input_dir, -5.3926)
    assert neutrality["expected_neutral_valence_electrons"] == 66984
    assert neutrality["spin_degeneracy"] == 2
    assert neutrality["chemical_potential_available"] is False


def test_projected_outputs_follow_solver_indices_after_energy_sorting(tmp_path: Path) -> None:
    weights = {
        "carbon_pz": [0.2, 0.6], "carbon": [0.5, 0.7],
        "graphene_lower": [0.2, 0.4], "graphene_upper": [0.3, 0.3],
        "boron": [0.2, 0.1], "nitrogen": [0.3, 0.2], "hbn": [0.5, 0.3],
    }
    (tmp_path / "mulliken_000.json").write_text(json.dumps({
        "k_index": 0, "energies_eV": [2.0, 1.0], "mulliken_weights": weights,
        "normalization_cdagger_s_c": [1.0, 1.0],
        "generalized_relative_residual": [1e-12, 2e-12],
    }))
    (tmp_path / "matrix_quality_000.json").write_text(json.dumps({
        "h_relative_hermiticity_before_solver_symmetrization": 0.0,
        "s_relative_hermiticity_before_solver_symmetrization": 0.0,
    }))
    rows, diagnostics = projected_band_data(tmp_path, sparse_spectrum.np.asarray([[2.0, 1.0]]))
    assert rows[0][0]["solver_band_index"] == 1
    assert rows[0][0]["weight_c_pz"] == pytest.approx(0.6)
    assert diagnostics["maximum_partition_sum_error"] < 1e-12
    dos, dos_diagnostics = projected_dos_data(
        tmp_path, sparse_spectrum.np.asarray([[2.0, 1.0]]), 1.5, 0.5, 1000.0
    )
    assert len(dos) == 801
    assert all(sparse_spectrum.np.isfinite(row["pdos_c_pz"]) for row in dos)
    assert dos_diagnostics["kpoint_count"] == 1
    assert projected_dos_observables(dos)["gap_claimed"] is False


def test_kprime_comparison_uses_set_ranks_without_claiming_band_identity() -> None:
    bands = [
        {"k_index": k, "band_index": band, "energy_aligned_eV": energy,
         "weight_c_pz": weight, "weight_graphene_lower": weight,
         "weight_graphene_upper": 1 - weight, "weight_hbn": 0.0}
        for k, values in enumerate(([(0.0, 0.2), (0.1, 0.3)], [(0.01, 0.25), (0.11, 0.35)]))
        for band, (energy, weight) in enumerate(values)
    ]
    result = kprime_comparison(bands)
    assert result["pairing"] == "ascending_energy_rank_only_not_band_identity"
    assert result["energy_rank_rmse_meV"] == pytest.approx(10.0)
    assert result["symmetry_equivalence_required"] is False


def test_projected_followup_rejects_scientifically_invalid_smoke() -> None:
    manifest = {"status": "completed", "projection": {"status": "completed", "diagnostics": {
        "status": "valid", "maximum_normalization_error": 1e-12,
        "maximum_partition_sum_error": 1e-12,
        "maximum_generalized_relative_residual": 1e-9,
    }}}
    assert validated_projection(manifest)
    manifest["projection"]["diagnostics"]["maximum_partition_sum_error"] = 1e-3
    assert not validated_projection(manifest)


def test_projected_resolution_stability_compares_only_shared_special_point_ranks() -> None:
    def result(offset: float) -> dict:
        return {"bands": [
            {"k_index": k_index, "k_label": label, "band_index": band,
             "energy_aligned_eV": energy + offset, "weight_c_pz": 0.5}
            for k_index, label in enumerate(("Γ", "K", "M", "Γ"))
            for band, energy in enumerate((-0.01, 0.01))
        ]}
    stability = projected_resolution_stability(result(0.0), result(1e-6))
    assert stability["status"] == "completed"
    assert stability["energy_rank_rmse_meV"] == pytest.approx(0.001)
    assert stability["band_count_stability"] == "not_evaluated_no_additional_solve"
