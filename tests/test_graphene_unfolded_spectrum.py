import sys
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Comparison/scripts"))

from run_graphene_unfolded_spectrum import (  # noqa: E402
    folded_kpoints,
    graphene_layer_indices,
    k_distances,
    primitive_samples,
    validate_folding,
)


def test_layer_selection_accepts_pure_tbg_and_tbg_hbn() -> None:
    pure = np.asarray([[0, 0, 9.35], [1, 0, 9.35], [0, 0, 12.7], [1, 0, 12.7]])
    supported = np.vstack((np.asarray([[0, 0, 6.0]]), pure))
    assert graphene_layer_indices(pure, "bottom").tolist() == [0, 1]
    assert graphene_layer_indices(pure, "top").tolist() == [2, 3]
    assert graphene_layer_indices(supported, "bottom").tolist() == [1, 2]
    assert graphene_layer_indices(supported, "top").tolist() == [3, 4]


def test_primitive_graphene_path_folds_into_commensurate_moire_bz() -> None:
    samples = primitive_samples(8)
    assert [samples[index][0] for index in (0, 7, 14, 21)] == ["K", "Γ", "M", "K"]
    matrix = np.asarray([[61, 31], [30, 61]])
    folded = folded_kpoints(samples, matrix)
    assert np.allclose(folded[[0, 7, 14, 21]], [
        [2.0 / 3.0, 1.0 / 3.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [2.0 / 3.0, 1.0 / 3.0, 0.0],
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
