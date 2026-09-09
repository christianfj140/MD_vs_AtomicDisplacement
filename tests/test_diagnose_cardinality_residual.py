from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Comparison/scripts"))
from diagnose_cardinality_residual import _block_report  # noqa: E402


def test_block_report_partitions_every_matrix_element():
    orbitals = [{"atom": 0, "l": 0}, {"atom": 0, "l": 1}, {"atom": 1, "l": 0}, {"atom": 1, "l": 1}]
    assert sum(row["squared_norm_fraction"] for row in _block_report(np.ones((4, 4)), orbitals)) == 1.0
