import json
import os
import sys
import time
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "Comparison" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import run_deeph_sparse_spectrum as sparse_spectrum  # noqa: E402
from run_deeph_sparse_spectrum import band_k_data  # noqa: E402


def test_band_k_data_supports_single_high_symmetry_points() -> None:
    assert band_k_data(1) == [
        "1 0.0 0.0 0.0 0.0 0.0 0.0",
        "1 0.3333333333333333 0.3333333333333333 0.0 0.3333333333333333 0.3333333333333333 0.0",
        "1 0.5 0.0 0.0 0.5 0.0 0.0",
    ]
    assert len(band_k_data(2)) == 4
    detailed = band_k_data(8)
    assert len(detailed) == 22
    assert len(set(detailed)) == 21
    assert detailed[0] == detailed[-1]
    assert detailed.count(
        "1 0.3333333333333333 0.3333333333333333 0.0 0.3333333333333333 0.3333333333333333 0.0"
    ) == 1
    assert detailed.count("1 0.5 0.0 0.0 0.5 0.0 0.0") == 1
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
