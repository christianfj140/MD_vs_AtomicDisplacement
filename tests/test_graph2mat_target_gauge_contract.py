import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"Comparison/scripts"))
from certify_graph2mat_target_gauge_contract import _errors


class Matrix:
    def __init__(self, h, s): self.h, self.s = h, s
    def Hk(self, **_): return self.h
    def Sk(self, **_): return self.s


def test_contract_classifier_distinguishes_energy_zeros():
    s = np.array([[1., .2], [.2, 1.]])
    h = np.array([[2., .1], [.1, 3.]])
    rows = _errors(Matrix(h, s), Matrix(h, s), 4., 7.)
    assert all(row["closest"] == "H_sisl_equals_H_abs_minus_EfS" for row in rows)
