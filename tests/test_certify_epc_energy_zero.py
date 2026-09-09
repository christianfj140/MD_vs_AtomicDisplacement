from pathlib import Path
import json
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Comparison/scripts"))
import certify_epc_energy_zero as gauge  # noqa: E402
from certify_epc_energy_zero import vacuum_level  # noqa: E402


def test_vacuum_level_uses_the_cell_centred_slab():
    grid = np.zeros((2, 3, 10))
    grid[:, :, 4:6] = (2.0, 4.0)
    assert vacuum_level(grid) == (3.0, 1.0)


def test_real_report_contains_gauge_aligned_richardson():
    if not gauge.DEFAULT_ROOT.exists():
        pytest.skip("real gauge campaign unavailable")
    assert gauge.main([]) == 0
    report = json.loads((gauge.DEFAULT_ROOT / "energy_zero_alignment.json").read_text())
    assert report["richardson"]
    assert all("H-c_vacuum*S" in row["scheme"] for row in report["richardson"])
