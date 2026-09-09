from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "Comparison" / "scripts"
for path in (str(SCRIPTS), str(REPO_ROOT / "shared")):
    if path not in sys.path:
        sys.path.insert(0, path)

pytest.importorskip("sisl")

import onlys_semantic_preflight as preflight  # noqa: E402

RESULT = REPO_ROOT / "Comparison/results/epc/onlys_semantic_preflight"


def test_fdf_uses_documented_keyword_and_duplicates_the_geometry() -> None:
    source = (preflight.PROD_CENTRAL / "RUN.fdf").read_text(encoding="utf-8")
    xyz = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.005, 0.0, 0.0], [1.0, 0.0, 0.0]])
    text = preflight._duplicated_fdf(source, "probe", xyz)
    assert "TS.S.Save                        T" in text
    assert "MaxSCFIterations                 0" in text
    # The legacy keyword must not be reintroduced merely because this build accepts it.
    assert "TS.onlyS" not in text
    assert "NumberOfAtoms                    4" in text
    coordinates = text.split("%block AtomicCoordinatesAndAtomicSpecies")[1]
    coordinates = coordinates.split("%endblock")[0].strip().splitlines()
    assert len(coordinates) == 4
    # Second half is the first half with atom 0 displaced along the axis only.
    assert coordinates[1].split()[:3] == coordinates[3].split()[:3]
    assert coordinates[0].split()[0] != coordinates[2].split()[0]


def test_dense_k_grid_is_the_production_mesh() -> None:
    dense = preflight._dense_k_points()
    assert dense.shape == (400, 3)
    assert np.allclose(dense[0], 0.0)
    assert np.allclose(np.unique(dense[:, 2]), 0.0)
    assert len(np.unique(dense[:, 0])) == 20


def test_cross_derivative_takes_the_inelastica_blocks() -> None:
    """dS_u must come from S[0:N, N:2N] and use h_+ + h_- as the denominator."""

    class _Stub:
        def __init__(self, matrix):
            self._matrix = matrix

        def Sk(self, k, format):  # noqa: A002 - mirrors sisl's signature
            return self._matrix

    no, h = 2, preflight.H_ANG
    plus = np.arange(16, dtype=float).reshape(4, 4)
    minus = np.zeros((4, 4))
    right, left = preflight._cross_derivative(
        {("C1_x", 1): _Stub(plus), ("C1_x", -1): _Stub(minus)}, "C1_x", no, (0.0, 0.0, 0.0)
    )
    assert np.allclose(right, plus[:no, no:] / (2 * h))
    assert np.allclose(left, plus[no:, :no] / (2 * h))


@pytest.mark.slow
def test_recorded_report_keeps_its_preregistered_thresholds_and_verdicts() -> None:
    path = RESULT / "onlys_semantic_preflight.json"
    if not path.is_file():
        pytest.skip("preflight has not been run in this checkout")
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["schema"] == "onlys_semantic_preflight_prod_v1"
    assert report["thresholds"]["P1_propagated_ev_per_ang"] == 5.0e-4
    assert report["thresholds"]["P0_S11_relative_spectral"] == 1e-8
    assert report["siesta_interface"]["keyword_used"] == "TS.S.Save"
    assert report["gate_P0"]["verdict"] == "PASS"
    for row in report["gate_P0"]["runs"]:
        assert row["n_orbitals"] == 2 * report["gate_P0"]["N"]
        assert row["halves_share_orbital_contract"]
        assert row["ordering_matches_central_ORB_INDX"]
        for check in row["S11_checks"]:
            assert check["S11_vs_TSHS_relative_spectral"] < 1e-8
    for case in report["cases"]:
        assert case["max_solve_residual_relative"] < 1e-11
        for row in case["gate_P1_diagnostic_k"]:
            assert row["orientation_S12_is_S_R"] is True
    # B_I must never be claimed certified while any direction's P1 is NO_GO.
    if any(case["gate_P1"] != "PASS" for case in report["cases"]):
        assert report["b_i_status"] == "NOT_DETERMINED"
        assert report["verdict"] != "PASS"


@pytest.mark.slow
def test_extracted_B_I_is_the_selector_projection_and_antihermitian() -> None:
    artifact = RESULT / "onlys_semantic_preflight_matrices.npz"
    if not artifact.is_file():
        pytest.skip("preflight has not been run in this checkout")
    data = np.load(artifact)
    mask = data["orbital_mask_I"]
    assert mask.sum() == 4  # one C atom, SZ: s + 3p
    for direction in ("C1_x", "C1_z"):
        for ik in range(3):
            right = data[f"S_R_onlyS_{direction}_k{ik}"]
            b_i = data[f"B_I_{direction}_k{ik}"]
            assert np.allclose(b_i, mask[:, None] * right * mask[None, :])
            scale = max(np.linalg.norm(b_i), np.finfo(float).tiny)
            assert np.linalg.norm(b_i + b_i.conj().T) / scale < 1e-10
