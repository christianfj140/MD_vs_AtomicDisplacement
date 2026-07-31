import sys
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Comparison/scripts"))

from run_graphene_unfolded_spectrum import folded_kpoints, primitive_samples  # noqa: E402


def test_primitive_graphene_path_folds_into_commensurate_moire_bz() -> None:
    samples = primitive_samples(8)
    assert [samples[index][0] for index in (0, 7, 14, 21)] == ["Γ", "K", "M", "Γ"]
    matrix = np.asarray([[61, 31], [30, 61]])
    folded = folded_kpoints(samples, matrix)
    assert np.allclose(folded[[0, 7, 14, 21]], [
        [0.0, 0.0, 0.0],
        [0.0, 2.0 / 3.0, 0.0],
        [0.0, 0.5, 0.0],
        [0.0, 0.0, 0.0],
    ])
