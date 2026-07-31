import sys
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Comparison/scripts"))

from run_graphene_unfolded_spectrum import (  # noqa: E402
    folded_kpoints,
    k_distances,
    primitive_samples,
    validate_folding,
)


def test_primitive_graphene_path_folds_into_commensurate_moire_bz() -> None:
    samples = primitive_samples(8)
    assert [samples[index][0] for index in (0, 7, 14, 21)] == ["Γ", "K", "M", "Γ"]
    matrix = np.asarray([[61, 31], [30, 61]])
    folded = folded_kpoints(samples, matrix)
    assert np.allclose(folded[[0, 7, 14, 21]], [
        [0.0, 0.0, 0.0],
        [2.0 / 3.0, 1.0 / 3.0, 0.0],
        [0.5, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ])
    primitive_lattice = np.asarray([[2.48, 0.0], [-1.24, np.sqrt(3.0) * 1.24]])
    validation = validate_folding(samples, matrix, primitive_lattice)
    assert validation["status"] == "valid"
    assert validation["maximum_cartesian_error_inv_ang"] < 1e-10
    distances = k_distances(samples, 2 * np.pi * np.linalg.inv(primitive_lattice).T)
    assert len(distances) == len(samples)
    assert all(right > left for left, right in zip(distances, distances[1:]))


def test_cpu_and_gpu_wrappers_share_the_unfolding_weight_hook() -> None:
    scripts = Path(__file__).resolve().parents[1] / "Comparison/scripts"
    helper = (scripts / "deeph_unfolding_weights.jl").read_text(encoding="utf-8")
    cpu = (scripts / "deeph_sparse_calc_unfolded.jl").read_text(encoding="utf-8")
    gpu = (scripts / "deeph_sparse_calc_gpu.jl").read_text(encoding="utf-8")
    assert "function record_unfolding!" in helper
    assert 'include(joinpath(@__DIR__, "deeph_unfolding_weights.jl"))' in cpu
    assert 'include(joinpath(@__DIR__, "deeph_unfolding_weights.jl"))' in gpu
