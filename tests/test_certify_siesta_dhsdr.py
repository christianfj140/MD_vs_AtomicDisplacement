"""C09 / E-F_001-S12: the GO-2 certification of the SIESTA ``dHSdR.nc`` derivatives.

Most of these tests are about the gate's *teeth*, not its agreement. A
certification that says PASS is worth nothing unless it would say FAIL on the
defects it claims to exclude, so the discrepancy classifier is driven directly
with synthetic delta sweeps that stand for each one:

* a delta-independent offset -- what a wrong Ry/Bohr -> eV/Ang factor, a
  permuted image index or a flipped sign produce -- must come out
  ``unresolved``;
* an O(delta^2) decay -- what a collective central difference legitimately adds
  on top of a contraction of one-atom derivatives -- must come out
  ``truncation_limited``;
* anything under the measured noise floor must come out ``noise_limited``.

The finite-difference and contraction primitives are exercised on hand-built
fields where the answer is known in closed form, so a sign or a Fermi-shift
error cannot hide behind SIESTA's numbers. The last test runs the real
certification if S11's artifacts are in the checkout, and skips cleanly if not.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "Comparison" / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "shared"))

from certify_siesta_dhsdr import (  # noqa: E402
    BOHR_TO_ANG,
    RICHARDSON_NOISE_AMPLIFICATION,
    RY_TO_EV,
    TshsMatrices,
    _tolerance_value,
    align,
    best_fit_scale,
    classify,
    contract_fc,
    difference_norms,
    doubling_pairs,
    finite_difference,
    fitted_exponent,
    hermiticity_residual,
    noise_floor,
    richardson_extrapolate,
)

ORIGIN = (0, 0, 0)
RIGHT = (1, 0, 0)
LEFT = (-1, 0, 0)


# ---------------------------------------------------------------------------
# Index space
# ---------------------------------------------------------------------------


def test_absent_keys_are_exact_zeros_not_dropped_support():
    """A sparsified field and a dense one must stay comparable elementwise."""
    dense = {(0, 0, ORIGIN): 1.0, (0, 1, ORIGIN): 2.0}
    sparse = {(0, 0, ORIGIN): 1.0}
    keys, (left, right) = align(dense, sparse)
    assert keys == [(0, 0, ORIGIN), (0, 1, ORIGIN)]
    assert right.tolist() == [1.0, 0.0]
    assert difference_norms(dense, sparse) == {"max_abs": 2.0, "frobenius": 2.0}


def test_alignment_is_by_lattice_offset_not_by_position():
    """The same element under two different R must never be compared."""
    here = {(0, 0, ORIGIN): 5.0}
    there = {(0, 0, RIGHT): 5.0}
    assert difference_norms(here, there)["max_abs"] == 5.0


# ---------------------------------------------------------------------------
# The numerical unit certification
# ---------------------------------------------------------------------------


def test_best_fit_scale_recovers_a_known_factor():
    candidate = {(0, 0, ORIGIN): 2.0, (0, 1, RIGHT): -3.0}
    reference = {key: 7.5 * value for key, value in candidate.items()}
    assert best_fit_scale(reference, candidate) == pytest.approx(7.5)


@pytest.mark.parametrize("wrong", [BOHR_TO_ANG, 1.0 / BOHR_TO_ANG, RY_TO_EV, -1.0])
def test_the_wrong_unit_or_a_flipped_sign_moves_the_scale_far_off_one(wrong):
    """The alternatives a unit defect would produce are nowhere near the window."""
    candidate = {(0, 0, ORIGIN): 2.0, (0, 1, RIGHT): -3.0, (1, 0, LEFT): 0.5}
    reference = {key: wrong * value for key, value in candidate.items()}
    assert abs(best_fit_scale(reference, candidate) - 1.0) > 0.01


def test_best_fit_scale_is_undefined_against_a_vanishing_field():
    assert best_fit_scale({(0, 0, ORIGIN): 1.0}, {(0, 0, ORIGIN): 0.0}) is None


# ---------------------------------------------------------------------------
# Real-space hermiticity
# ---------------------------------------------------------------------------


def test_hermiticity_holds_when_the_transpose_partner_carries_the_same_value():
    field = {(0, 1, RIGHT): 4.0, (1, 0, LEFT): 4.0}
    assert hermiticity_residual(field) == {"max_abs_residual": 0.0, "missing_transpose_partners": 0}


def test_a_broken_transpose_partner_is_measured():
    field = {(0, 1, RIGHT): 4.0, (1, 0, LEFT): 4.25}
    assert hermiticity_residual(field)["max_abs_residual"] == pytest.approx(0.25)


def test_a_missing_transpose_partner_is_counted_not_ignored():
    """Negating R is part of the partner: dropping that gives a missing partner."""
    field = {(0, 1, RIGHT): 4.0, (1, 0, RIGHT): 4.0}
    assert hermiticity_residual(field)["missing_transpose_partners"] == 2


# ---------------------------------------------------------------------------
# Delta scaling and the classifier: the teeth of the gate
# ---------------------------------------------------------------------------


def test_fitted_exponent_reads_a_quadratic_decay():
    deltas = [0.005, 0.01, 0.02]
    assert fitted_exponent(deltas, [3.0 * delta**2 for delta in deltas]) == pytest.approx(2.0)


def test_fitted_exponent_reads_zero_for_a_delta_independent_offset():
    assert fitted_exponent([0.005, 0.01, 0.02], [0.5, 0.5, 0.5]) == pytest.approx(0.0, abs=1e-9)


def test_noise_floor_is_the_largest_disagreement_among_plateau_amplitudes():
    fields = {
        0.005: {(0, 0, ORIGIN): 1.0},
        0.01: {(0, 0, ORIGIN): 1.5},
        0.02: {(0, 0, ORIGIN): 1.2},
    }
    assert noise_floor(fields, [0.005, 0.01, 0.02])["max_abs"] == pytest.approx(0.5)


DELTAS = [0.005, 0.01, 0.02]


def test_a_discrepancy_under_the_noise_floor_passes_as_noise():
    verdict = classify(
        deltas=DELTAS,
        discrepancy_by_delta={0.005: 0.9, 0.01: 0.8, 0.02: 0.95},
        tau=1.0,
        is_null_direction=False,
    )
    assert verdict["verdict"] == "noise_limited"
    assert verdict["passed"]


def test_a_quadratic_discrepancy_above_the_floor_passes_as_truncation():
    """The collective cross term: real, understood, and gone as delta -> 0."""
    verdict = classify(
        deltas=DELTAS,
        discrepancy_by_delta={delta: 100.0 * delta**2 for delta in DELTAS},
        tau=1e-6,
        is_null_direction=False,
    )
    assert verdict["verdict"] == "truncation_limited"
    assert verdict["passed"]
    assert verdict["fitted_exponent"] == pytest.approx(2.0)


def test_a_delta_independent_discrepancy_fails():
    """A wrong unit factor, a permuted image or a flipped sign look like this."""
    verdict = classify(
        deltas=DELTAS,
        discrepancy_by_delta={delta: 4.2 for delta in DELTAS},
        tau=1e-3,
        is_null_direction=False,
    )
    assert verdict["verdict"] == "unresolved"
    assert not verdict["passed"]


def test_a_discrepancy_that_grows_with_delta_fails():
    verdict = classify(
        deltas=DELTAS,
        discrepancy_by_delta={delta: 10.0 * delta for delta in DELTAS},
        tau=1e-6,
        is_null_direction=False,
    )
    assert verdict["verdict"] == "unresolved"
    assert not verdict["passed"]


# ---------------------------------------------------------------------------
# The finite difference, and the Fermi shift it has to undo
# ---------------------------------------------------------------------------


def _tshs(run_id, *, h_shifted, overlap, fermi_ev):
    return TshsMatrices(
        run_id=run_id,
        path=f"{run_id}.TSHS",
        h_shifted=h_shifted,
        overlap=overlap,
        fermi_ev=fermi_ev,
        no_u=1,
        nsc=(1, 1, 1),
        sc_off=(ORIGIN,),
    )


def test_the_central_difference_undoes_the_fermi_shift_with_each_run_s_own_ef():
    """``H`` is what dHSdR differentiates; sisl hands back ``H - E_F S``.

    Built so that the absolute H is 2.0 at ``+delta`` and 1.0 at ``-delta``:
    the certified derivative is exactly 1.0 / (2 * 0.01), and the uncorrected
    one is not, because E_F moves between the two runs.
    """
    key = (0, 0, ORIGIN)
    plus = _tshs("plus", h_shifted={key: 2.0 - (-5.0) * 3.0}, overlap={key: 3.0}, fermi_ev=-5.0)
    minus = _tshs("minus", h_shifted={key: 1.0 - (-4.0) * 2.0}, overlap={key: 2.0}, fermi_ev=-4.0)

    corrected = finite_difference(plus, minus, 0.01)
    assert corrected["D_H"][key] == pytest.approx((2.0 - 1.0) / 0.02)
    assert corrected["D_S"][key] == pytest.approx((3.0 - 2.0) / 0.02)

    uncorrected = finite_difference(plus, minus, 0.01, undo_fermi_shift=False)
    assert uncorrected["D_H"][key] != pytest.approx(corrected["D_H"][key])
    # S carries no Fermi shift, so it is the same either way.
    assert uncorrected["D_S"][key] == pytest.approx(corrected["D_S"][key])


def test_the_overlap_derivative_is_independent_of_the_fermi_level():
    key = (0, 0, ORIGIN)
    plus = _tshs("plus", h_shifted={key: 0.0}, overlap={key: 3.0}, fermi_ev=-5.0)
    minus = _tshs("minus", h_shifted={key: 0.0}, overlap={key: 2.0}, fermi_ev=99.0)
    assert finite_difference(plus, minus, 0.5)["D_S"][key] == pytest.approx(1.0)


@pytest.mark.parametrize("delta", [0.0, -0.01])
def test_a_non_positive_amplitude_is_refused(delta):
    key = (0, 0, ORIGIN)
    matrices = _tshs("run", h_shifted={key: 1.0}, overlap={key: 1.0}, fermi_ev=0.0)
    with pytest.raises(Exception):
        finite_difference(matrices, matrices, delta)


# ---------------------------------------------------------------------------
# Richardson (h, 2h): a higher-order reference independent of the two-point
# FD's own truncation, per the E-F_001 GO-2 requirement.
# ---------------------------------------------------------------------------


def test_doubling_pairs_finds_every_h_2h_ladder():
    assert doubling_pairs([0.005, 0.01, 0.02]) == [(0.005, 0.01), (0.01, 0.02)]


def test_doubling_pairs_is_empty_without_a_x2_ratio():
    assert doubling_pairs([0.005, 0.011, 0.019]) == []


def test_richardson_extrapolate_cancels_a_known_h_squared_error():
    """``D_h = f' + c h^2`` at two amplitudes must combine to exactly ``f'``."""
    key = (0, 0, ORIGIN)
    true_derivative, cubic_coefficient = 10.0, 3.0
    h = 0.01
    fd_h = {key: true_derivative + cubic_coefficient * h**2}
    fd_2h = {key: true_derivative + cubic_coefficient * (2 * h) ** 2}
    assert richardson_extrapolate(fd_h, fd_2h)[key] == pytest.approx(true_derivative)


def test_richardson_extrapolate_treats_absent_keys_as_zero():
    richardson = richardson_extrapolate({(0, 0, ORIGIN): 4.0}, {(0, 1, RIGHT): 1.0})
    assert richardson[(0, 0, ORIGIN)] == pytest.approx(4.0 * 4.0 / 3.0)
    assert richardson[(0, 1, RIGHT)] == pytest.approx(-1.0 / 3.0)


def test_richardson_noise_amplification_is_the_triangle_inequality_bound():
    """The combination's coefficients are 4/3 and -1/3; noise does not cancel."""
    assert RICHARDSON_NOISE_AMPLIFICATION == pytest.approx(4.0 / 3.0 + 1.0 / 3.0)


# ---------------------------------------------------------------------------
# Contracting the FC file onto a direction
# ---------------------------------------------------------------------------


class _Record:
    """The slice of SparseDerivative that :func:`contract_fc` reads."""

    def __init__(self, values, *, row=(0, 0), col=(0, 1), isc=(ORIGIN, RIGHT)):
        self.row = np.asarray(row, dtype=np.int64)
        self.col = np.asarray(col, dtype=np.int64)
        self.isc = np.asarray(isc, dtype=np.int64)
        self.values = np.asarray(values, dtype=np.float64).reshape(-1, 1)
        self.nnz = self.values.shape[0]


class _Dhsdr:
    """Two atoms, three axes, one distinct pair of values per degree of freedom."""

    na_u = 2

    def __init__(self):
        self.records = {
            (atom, axis): {
                "D_H": _Record([atom * 10.0 + axis, -(atom * 10.0 + axis)]),
                "D_S": _Record([atom + 0.1 * axis, 0.0]),
            }
            for atom in (1, 2)
            for axis in (1, 2, 3)
        }


def test_a_one_hot_direction_returns_exactly_that_degree_of_freedom():
    vectors = np.zeros((2, 3))
    vectors[1, 2] = 1.0  # atom 2 (1-based), axis 3
    field = contract_fc(_Dhsdr(), vectors, "D_H")
    assert field[(0, 0, ORIGIN)] == pytest.approx(23.0)
    assert field[(0, 1, RIGHT)] == pytest.approx(-23.0)


def test_the_contraction_is_linear_in_the_direction():
    """``J(a v + b w) = a Jv + b Jw`` -- the directional derivative is a contraction."""
    rng = np.random.default_rng(0)
    dhsdr = _Dhsdr()
    v = rng.normal(size=(2, 3))
    w = rng.normal(size=(2, 3))
    a, b = 1.7, -0.4

    combined = contract_fc(dhsdr, a * v + b * w, "D_H")
    from_parts = contract_fc(dhsdr, v, "D_H")
    other = contract_fc(dhsdr, w, "D_H")
    keys, (left, right, extra) = align(combined, from_parts, other)
    assert left == pytest.approx(a * right + b * extra)


def test_a_zero_weight_contributes_nothing():
    field = contract_fc(_Dhsdr(), np.zeros((2, 3)), "D_S")
    assert field == {}


# ---------------------------------------------------------------------------
# fdf tolerance strings
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [("0.0 Ry/Bohr", 0.0), ("1.0e-3 1/Bohr", 1e-3), ("2.5E-4 Ry/Bohr", 2.5e-4)],
)
def test_a_tolerance_value_is_read_off_its_fdf_string(text, expected):
    assert _tolerance_value(text) == pytest.approx(expected)


def test_an_unreadable_tolerance_is_refused_rather_than_defaulted():
    with pytest.raises(Exception):
        _tolerance_value("Ry/Bohr")


# ---------------------------------------------------------------------------
# The real artifacts, when the checkout has them
# ---------------------------------------------------------------------------

REFERENCE_ROOT = REPO_ROOT / "Comparison/results/epc/siesta_reference/graphene"
SPARSIFIED_ROOT = REPO_ROOT / "Comparison/results/epc/siesta_reference/graphene_sparsified"


@pytest.mark.slow
@pytest.mark.skipif(
    not (REFERENCE_ROOT / "epc_siesta_reference_manifest.json").is_file()
    or not (SPARSIFIED_ROOT / "epc_siesta_reference_manifest.json").is_file(),
    reason="S11 SIESTA reference campaigns are not in this checkout",
)
def test_go2_passes_on_the_graphene_reference_campaign(tmp_path):
    pytest.importorskip("sisl")
    pytest.importorskip("netCDF4")
    from certify_siesta_dhsdr import certify

    report = certify(
        reference_root=REFERENCE_ROOT,
        sparsified_root=SPARSIFIED_ROOT,
        output_dir=tmp_path,
    )
    assert report["verdict"] == "PASS", report["summary"]
    assert report["summary"]["checks_failed"] == []
    # The provenance limitation has to survive the PASS, not be discharged by it.
    assert report["provenance"]["classification"] == "VERIFIED_BINARY_ONLY"
    assert report["provenance"]["propagates_to_derived_artifacts"]
    assert report["provenance"]["limitation"]
    # The unit factors are certified numerically, on the directions that carry
    # signal: a wrong Ry/Bohr conversion would put this at 0.53 or 1.89.
    for comparison in report["comparisons"]:
        if not comparison["plateau"]["is_null_direction"]:
            assert comparison["unit_scale"]["best_fit_scale"] == pytest.approx(1.0, abs=0.01)
    assert (tmp_path / "go2_certification.json").is_file()
    assert (tmp_path / "go2_certification.csv").is_file()
    # A bare boolean is not reproducible evidence: the topology comparison
    # must publish a canonical hash of the verified-identical R-image set,
    # stable across every displacement amplitude (same lattice, same offsets).
    r_image_check = next(
        c for c in report["structural_checks"] if c["check"] == "r_image_sets_identical"
    )
    hashes = {entry["topology_sha256"] for entry in r_image_check["per_delta"].values()}
    assert len(hashes) == 1
    assert len(next(iter(hashes))) == 64  # sha256 hex digest


def _rescaled(original, factor):
    return lambda dhsdr, vectors, kind, **kwargs: {
        key: factor * value for key, value in original(dhsdr, vectors, kind, **kwargs).items()
    }


def _shifted_image(original):
    """Every element attributed to the neighbouring periodic image."""
    return lambda dhsdr, vectors, kind, **kwargs: {
        (row, col, (isc[0] + 1, isc[1], isc[2])): value
        for (row, col, isc), value in original(dhsdr, vectors, kind, **kwargs).items()
    }


def _transposed(original):
    """Row and column swapped without negating R."""
    return lambda dhsdr, vectors, kind, **kwargs: {
        (col, row, isc): value for (row, col, isc), value in original(dhsdr, vectors, kind, **kwargs).items()
    }


@pytest.mark.slow
@pytest.mark.skipif(
    not (REFERENCE_ROOT / "epc_siesta_reference_manifest.json").is_file()
    or not (SPARSIFIED_ROOT / "epc_siesta_reference_manifest.json").is_file(),
    reason="S11 SIESTA reference campaigns are not in this checkout",
)
@pytest.mark.parametrize(
    "defect",
    [
        "unit_ry_bohr_read_as_ry_ang",
        "unit_off_by_bohr",
        "flipped_sign",
        "image_index_shifted",
        "row_col_transposed",
        "fermi_shift_not_undone",
    ],
)
def test_go2_fails_on_an_injected_index_phase_or_unit_defect(tmp_path, monkeypatch, defect):
    """The end-to-end proof that the PASS above is not vacuous.

    Each defect is one of the failure modes GO-2 claims to exclude, injected
    into the real artifacts. A gate that still reported PASS would be measuring
    nothing.
    """
    pytest.importorskip("sisl")
    pytest.importorskip("netCDF4")
    import certify_siesta_dhsdr as module

    original_contract = module.contract_fc
    original_difference = module.finite_difference
    injections = {
        "unit_ry_bohr_read_as_ry_ang": lambda: monkeypatch.setattr(
            module, "contract_fc", _rescaled(original_contract, BOHR_TO_ANG)
        ),
        "unit_off_by_bohr": lambda: monkeypatch.setattr(
            module, "contract_fc", _rescaled(original_contract, 1.0 / BOHR_TO_ANG)
        ),
        "flipped_sign": lambda: monkeypatch.setattr(module, "contract_fc", _rescaled(original_contract, -1.0)),
        "image_index_shifted": lambda: monkeypatch.setattr(
            module, "contract_fc", _shifted_image(original_contract)
        ),
        "row_col_transposed": lambda: monkeypatch.setattr(module, "contract_fc", _transposed(original_contract)),
        "fermi_shift_not_undone": lambda: monkeypatch.setattr(
            module,
            "finite_difference",
            lambda plus, minus, delta, undo_fermi_shift=True: original_difference(
                plus, minus, delta, undo_fermi_shift=False
            ),
        ),
    }
    injections[defect]()

    report = module.certify(reference_root=REFERENCE_ROOT, sparsified_root=SPARSIFIED_ROOT, output_dir=tmp_path)
    assert report["verdict"] == "FAIL", f"{defect} was not caught"
    # It has to fail on the derivative comparison itself, not only on some
    # incidental structural check.
    assert [entry for entry in report["comparisons"] if not entry["passed"]]
