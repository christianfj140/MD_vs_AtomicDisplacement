"""C14C / E-F_001-S17: the basis-response validation gate.

The reference here is *analytic*, not another finite difference: the toy builds
the finite basis explicitly (the same construction the GO-1 memo uses), so
``S_L = V1^dag V0`` and ``S_R = V0^dag V1`` are known exactly and their sum is
the ``D_S`` a perfect measurement would return. That makes it possible to test
what the gate is for -- that it *fails* -- rather than only that it passes:

* a response that reproduces the reference sum can still be a wrong split, so
  the identity ``S_L = S_R^dag`` is checked independently;
* the size of what each approximation drops is measured, split into diagonal
  and interband, and the symmetric form's error is exactly zero on the diagonal
  by construction;
* an artifact that omits the intra-atomic term returns ``NO_GO`` and a
  sensitivity, never a correction factor.

The multi-image half of the file reuses the SIESTA ``dS/dR`` fixtures of
``test_epc_basis_response.py``, so the gate is exercised on the representation
the graphene reference actually produces.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "Comparison" / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "shared"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from epc_basis_response import (  # noqa: E402
    DEFAULT_K_POINTS,
    GAMMA,
    TOLERANCES,
    VERDICT_NO_GO,
    VERDICT_PASS,
    BasisResponseError,
    ElectronicState,
    ResponseContext,
    bloch_sum,
    check_energy_origin_invariance,
    check_gauge_covariance,
    check_split_identity,
    check_sum_against_reference,
    check_uniform_translation,
    contribution_split,
    degenerate_groups,
    from_covariant_connection,
    from_D_S,
    from_S_L_S_R,
    inter_atomic_split,
    intra_atomic_sensitivity,
    omitted_terms,
    real_space_total,
    validate_basis_response,
)
from epc_formalism import (  # noqa: E402
    FORMALISM_ID_IGNORE_OVERLAP,
    FORMALISM_ID_SYMMETRIC,
)
from fd_perturbation_space import collective, uniform_translation  # noqa: E402
from test_epc_basis_response import ISC, ORBITAL_ATOM, _ds_records, _true_split  # noqa: E402

N_EXACT = 9  # dimension of the exact space: the PAO basis below is incomplete
N_BASIS = 5


def _toy(seed: int = 20260902) -> dict:
    """An explicit moving basis: exact ``S_L``, ``S_R``, ``D_S`` and ``D_H``."""
    rng = np.random.default_rng(seed)

    def cplx(rows: int, cols: int) -> np.ndarray:
        return rng.normal(size=(rows, cols)) + 1j * rng.normal(size=(rows, cols))

    v0, v1 = cplx(N_EXACT, N_BASIS), cplx(N_EXACT, N_BASIS)
    op0, op1 = cplx(N_EXACT, N_EXACT), cplx(N_EXACT, N_EXACT)
    op0, op1 = op0 + op0.conj().T, op1 + op1.conj().T

    s = v0.conj().T @ v0
    h = v0.conj().T @ op0 @ v0
    # H C = S C eps with C^dag S C = I, through a Cholesky factor of S.
    chol = np.linalg.inv(np.linalg.cholesky(s))
    eps, u = np.linalg.eigh(chol @ h @ chol.conj().T)
    return {
        "S": s,
        "H": h,
        "S_left": v1.conj().T @ v0,
        "S_right": v0.conj().T @ v1,
        "D_S": v1.conj().T @ v0 + v0.conj().T @ v1,
        "D_H": v1.conj().T @ op0 @ v0 + v0.conj().T @ op1 @ v0 + v0.conj().T @ op0 @ v1,
        "eps": eps,
        "C": chol.conj().T @ u,
    }


def _context(**overrides):
    base = {
        "geometry_signature": "geo-sha",
        "basis_signature": "orbital-contract-sha",
        "electronic_derivative_backend": "analytic_toy",
        "direction": collective(np.eye(2, 3) / np.sqrt(2.0), name="collective_x"),
    }
    return ResponseContext(**{**base, **overrides})


def _artifact(toy: dict, *, intra_atomic_included: bool = True, **overrides):
    return from_S_L_S_R(
        overrides.get("S_left", toy["S_left"]),
        overrides.get("S_right", toy["S_right"]),
        overrides.get("context", _context()),
        backend="analytic_toy",
        method="explicit_moving_basis",
        intra_atomic_included=intra_atomic_included,
    )


def _state(toy: dict, *, label: str = "gamma", eps: np.ndarray | None = None) -> ElectronicState:
    return ElectronicState(
        label=label, k=GAMMA, D_H=toy["D_H"], C=toy["C"], eps=toy["eps"] if eps is None else eps
    )


# ---------------------------------------------------------------------------
# Criterion 1: does it reproduce the best available reference
# ---------------------------------------------------------------------------


def test_sum_reproduces_the_analytic_reference():
    toy = _toy()
    check = check_sum_against_reference(
        _artifact(toy), toy["D_S"], reference_name="analytic_explicit_basis"
    )
    assert check["passed"]
    assert check["residual_frobenius"] < 1e-12
    assert "hermitian part only" in check["bounds"]


def test_a_response_that_misses_the_reference_fails_even_inside_its_own_noise_floor():
    toy = _toy()
    broken = _artifact(toy, S_left=toy["S_left"] * 1.01)
    assert not check_sum_against_reference(broken, toy["D_S"], reference_name="analytic")["passed"]

    # ... unless the reference itself cannot resolve the difference: tau_reference
    # is the reference's own floor, and below it there is nothing to compare.
    residual = np.linalg.norm(0.01 * toy["S_left"])
    generous = check_sum_against_reference(
        broken, toy["D_S"], reference_name="analytic", tau_reference=2.0 * residual
    )
    assert generous["passed"] and generous["tau_reference"] > generous["residual_frobenius"]


def test_the_sum_check_alone_does_not_certify_the_split():
    """A wrong split with the right sum passes criterion 1 and dies on the identity."""
    toy = _toy()
    rng = np.random.default_rng(5)
    offset = rng.normal(size=(N_BASIS, N_BASIS)) + 1j * rng.normal(size=(N_BASIS, N_BASIS))
    wrong = _artifact(toy, S_left=toy["S_left"] + offset, S_right=toy["S_right"] - offset)

    assert check_sum_against_reference(wrong, toy["D_S"], reference_name="analytic")["passed"]
    assert not check_split_identity(wrong)["passed"]
    assert check_split_identity(_artifact(toy))["passed"]


def test_identity_is_checked_at_generic_k_over_the_whole_image_table():
    left, right = _true_split()
    artifact = from_S_L_S_R(
        left[0], right[0], _context(), backend="siesta", method="m",
        intra_atomic_included=False, isc_off=ISC,
    )
    check = check_split_identity(artifact)
    assert check["passed"]
    assert [entry["k"] for entry in check["by_k"]] == [
        list(np.asarray(k, dtype=float)) for k in DEFAULT_K_POINTS
    ]
    # A real split carries a non-zero antisymmetric part; that is the information
    # no D_S reference can ever confirm.
    assert all(entry["antisymmetric_frobenius"] > 0.0 for entry in check["by_k"])


def test_identity_catches_a_wrong_image_partner():
    left, right = _true_split()
    scrambled = right[0][[0, 2, 1], :, :]  # +R and -R swapped: sum survives, identity does not
    artifact = from_S_L_S_R(
        left[0], scrambled, _context(), backend="siesta", method="m",
        intra_atomic_included=False, isc_off=ISC,
    )
    assert not check_split_identity(artifact)["passed"]


def test_covariant_representation_has_no_real_space_sum_and_says_so():
    toy = _toy()
    artifact = from_covariant_connection(
        toy["S_left"], toy["S"], _context(), backend="analytic_pao", method="m",
        intra_atomic_included=True,
    )
    assert real_space_total(artifact) is None
    check = check_sum_against_reference(artifact, toy["D_S"], reference_name="analytic")
    assert not check["applicable"] and not check["passed"]


# ---------------------------------------------------------------------------
# Uniform translation
# ---------------------------------------------------------------------------


def test_uniform_translation_sum_vanishes_while_the_response_does_not():
    toy = _toy()
    context = _context(direction=uniform_translation(2, "x"))
    acoustic = from_S_L_S_R(
        toy["S_left"], -toy["S_left"], context, backend="analytic_toy", method="m",
        intra_atomic_included=True,
    )
    check = check_uniform_translation(acoustic)
    assert check["passed"] and check["applicable"]
    assert check["S_left_frobenius"] > 0.0

    leaking = from_S_L_S_R(
        toy["S_left"], -toy["S_left"] + 0.1 * toy["S_right"], context,
        backend="analytic_toy", method="m", intra_atomic_included=True,
    )
    assert not check_uniform_translation(leaking)["passed"]


def test_translation_check_is_not_applicable_to_other_directions():
    check = check_uniform_translation(_artifact(_toy()))
    assert not check["applicable"] and check["passed"]


# ---------------------------------------------------------------------------
# Criterion 2: the size of what each approximation drops
# ---------------------------------------------------------------------------


def test_basis_response_contribution_is_split_into_diagonal_and_interband():
    toy = _toy()
    contribution = contribution_split(_artifact(toy), _state(toy))
    response = contribution["basis_response_contribution"]

    # It is the distance to the control negative C^dag D_H C, in both channels.
    raw = toy["C"].conj().T @ toy["D_H"] @ toy["C"]
    g = toy["C"].conj().T @ toy["D_H"] @ toy["C"]
    g = g - (toy["C"].conj().T @ toy["S_left"] @ toy["C"]) * toy["eps"][None, :]
    g = g - (toy["C"].conj().T @ toy["S_right"] @ toy["C"]) * toy["eps"][:, None]
    assert np.isclose(response["frobenius"], np.linalg.norm(raw - g))
    assert response["diagonal_frobenius"] > 0.0 and response["interband_frobenius"] > 0.0

    # The control negative's delta *is* that contribution; it is reported twice
    # on purpose, once as a magnitude and once as an approximation error.
    ignored = contribution["approximations"][FORMALISM_ID_IGNORE_OVERLAP]
    assert ignored["status"] == "approximation"
    assert {key: ignored[key] for key in response} == response


def test_symmetric_approximation_is_exact_on_the_diagonal_and_wrong_between_bands():
    """Memo (F3): the dropped term is (eps_m - eps_n) A, so the diagonal is exact."""
    toy = _toy()
    symmetric = contribution_split(_artifact(toy), _state(toy))["approximations"][
        FORMALISM_ID_SYMMETRIC
    ]
    assert symmetric["status"] == "approximation"
    assert symmetric["diagonal_frobenius"] < 1e-12
    assert symmetric["interband_frobenius"] > 0.0
    assert symmetric["relative_interband"] > 0.0


def test_symmetric_error_is_reported_on_shell_as_well_as_over_the_whole_block():
    """`(eps_m - eps_n) A` grows with the gap, so the block norm is not the EPC error."""
    toy = _toy()
    symmetric = contribution_split(_artifact(toy), _state(toy), on_shell_window_ev=0.25)[
        "approximations"
    ][FORMALISM_ID_SYMMETRIC]
    on_shell = symmetric["on_shell"]
    assert on_shell["applicable"] and on_shell["window_ev"] == 0.25
    assert on_shell["frobenius"] <= symmetric["interband_frobenius"]

    # Widen the window until it swallows the spectrum and the two agree.
    span = float(toy["eps"].max() - toy["eps"].min())
    wide = contribution_split(_artifact(toy), _state(toy), on_shell_window_ev=span)[
        "approximations"
    ][FORMALISM_ID_SYMMETRIC]
    assert np.isclose(wide["on_shell"]["frobenius"], wide["interband_frobenius"])


def test_symmetric_error_vanishes_inside_a_degenerate_block():
    toy = _toy()
    eps = toy["eps"].copy()
    eps[1] = eps[0]
    degenerate = contribution_split(_artifact(toy), _state(toy, eps=eps))
    values = degenerate["approximations"][FORMALISM_ID_SYMMETRIC]
    assert [0, 1] in degenerate["degenerate_groups"]
    # The exactness is per element: the 0-1 block of the difference must vanish.
    assert values["interband_frobenius"] > 0.0  # other pairs still differ
    response = _artifact(toy).at_k(GAMMA)
    a_block = toy["C"].conj().T @ response.antisymmetric @ toy["C"]
    assert abs((eps[0] - eps[1]) * a_block[0, 1]) < 1e-12


def test_a_D_S_response_reports_no_symmetric_delta_because_it_is_already_that_form():
    toy = _toy()
    artifact = from_D_S(
        toy["D_S"], _context(formalism_id=FORMALISM_ID_SYMMETRIC), backend="frozen_fd",
        method="central_fd_of_S",
    )
    contribution = contribution_split(artifact, _state(toy))
    assert contribution["production"]["formalism_id"] == FORMALISM_ID_SYMMETRIC
    assert contribution["approximations"][FORMALISM_ID_SYMMETRIC]["frobenius"] == 0.0
    assert contribution["approximations"][FORMALISM_ID_IGNORE_OVERLAP]["frobenius"] > 0.0


# ---------------------------------------------------------------------------
# Gauge
# ---------------------------------------------------------------------------


def test_energy_origin_shift_leaves_g_invariant_against_the_independent_reference():
    toy = _toy()
    artifact = _artifact(toy)
    state = _state(toy)
    assert check_energy_origin_invariance(artifact, state, D_S_at_k=toy["D_S"])["passed"]

    # A response whose halves do not add up to the measured D_S is not invariant.
    broken = _artifact(toy, S_right=toy["S_right"] * 1.05)
    check = check_energy_origin_invariance(broken, state, D_S_at_k=toy["D_S"])
    assert not check["passed"] and check["D_S_source"] == "independent_reference"


def test_gauge_rotation_inside_a_degenerate_block_is_covariant():
    toy = _toy()
    eps = toy["eps"].copy()
    eps[2] = eps[1]
    check = check_gauge_covariance(_artifact(toy), _state(toy, eps=eps))
    assert check["applicable"] and check["passed"]
    assert [1, 2] in check["degenerate_groups"]
    assert check["singular_value_shift"] < 1e-9


def test_gauge_check_declares_itself_inapplicable_without_degeneracy():
    toy = _toy()
    check = check_gauge_covariance(_artifact(toy), _state(toy))
    assert not check["applicable"] and "no degenerate group" in check["detail"]
    assert degenerate_groups(toy["eps"]) == []


# ---------------------------------------------------------------------------
# Criterion 3: unresolved terms are a NO-GO, not a factor
# ---------------------------------------------------------------------------


def test_intra_atomic_sensitivity_is_a_coefficient_not_a_value():
    toy = _toy()
    atoms = np.array([1, 1, 1, 2, 2])
    sensitivity = intra_atomic_sensitivity(_state(toy), atoms)
    assert sensitivity["same_atom_orbital_pairs"] == 3 + 1
    assert sensitivity["max_interband_response"] > 0.0
    assert sensitivity["units"].startswith("eV/Ang of g per unit")
    assert "not its value" in sensitivity["interpretation"]

    # It is a gap effect: a flat spectrum is insensitive to A, which is (F3).
    flat = intra_atomic_sensitivity(_state(toy, eps=np.zeros_like(toy["eps"])), atoms)
    assert flat["max_interband_response"] < 1e-12

    with pytest.raises(BasisResponseError, match="atom_of_orbital"):
        intra_atomic_sensitivity(_state(toy), np.array([1, 2]))


def test_omitted_terms_separate_the_closed_pulay_term_from_the_open_one():
    toy = _toy()
    terms = {term["term"]: term for term in omitted_terms(_artifact(toy, intra_atomic_included=False))}
    assert terms["out_of_span_projection"]["status"] == "closed_non_blocking"
    assert terms["intra_atomic_basis_response"]["status"] == "unresolved"
    assert "no correction factor" in terms["intra_atomic_basis_response"]["policy"]

    included = {term["term"]: term for term in omitted_terms(_artifact(toy))}
    assert included["intra_atomic_basis_response"]["status"] == "included_by_backend"


def test_a_response_without_the_intra_atomic_term_is_NO_GO():
    toy = _toy()
    report = validate_basis_response(
        _artifact(toy, intra_atomic_included=False),
        D_S_reference=toy["D_S"],
        reference_name="analytic_explicit_basis",
        states=[_state(toy)],
        atom_of_orbital=np.array([1, 1, 1, 2, 2]),
    )
    assert report["verdict"] == VERDICT_NO_GO
    assert any("intra_atomic_basis_response" in reason for reason in report["reasons"])
    # Every measurable check still passed: the NO-GO is the missing term, not a failure.
    assert all(check["passed"] for check in report["checks"] if check["applicable"])
    term = next(t for t in report["omitted_terms"] if t["term"] == "intra_atomic_basis_response")
    assert term["sensitivity"][0]["max_interband_response"] > 0.0
    assert "correction_factor" not in term and "value" not in term


def test_a_complete_validated_response_passes():
    toy = _toy()
    report = validate_basis_response(
        _artifact(toy),
        D_S_reference=toy["D_S"],
        reference_name="analytic_explicit_basis",
        states=[_state(toy)],
    )
    assert report["verdict"] == VERDICT_PASS
    assert report["reasons"] == []
    assert report["tolerances"] == TOLERANCES
    assert report["tolerances_pre_registered"]
    assert report["provenance"]["epc_backend_class"] == "analytic_toy"
    assert report["contributions"][0]["units"] == "eV/Ang per unit-Frobenius displacement"


def test_without_a_reference_or_a_state_nothing_is_certified():
    toy = _toy()
    report = validate_basis_response(_artifact(toy))
    assert report["verdict"] == VERDICT_NO_GO
    assert any("never compared" in reason for reason in report["reasons"])
    assert any("not quantified" in reason for reason in report["reasons"])


def test_a_degraded_formalism_can_never_pass():
    toy = _toy()
    report = validate_basis_response(
        from_D_S(
            toy["D_S"], _context(formalism_id=FORMALISM_ID_SYMMETRIC), backend="frozen_fd",
            method="central_fd_of_S",
        ),
        D_S_reference=toy["D_S"],
        reference_name="analytic_explicit_basis",
        states=[_state(toy)],
    )
    assert report["verdict"] == VERDICT_NO_GO
    assert any("degraded variant" in reason for reason in report["reasons"])
    assert any(
        term["term"] == "antisymmetric_overlap_response_A" and term["status"] == "unresolved"
        for term in report["omitted_terms"]
    )


def test_overridden_tolerances_are_flagged_as_not_pre_registered():
    toy = _toy()
    report = validate_basis_response(
        _artifact(toy),
        D_S_reference=toy["D_S"],
        reference_name="analytic",
        states=[_state(toy)],
        tolerances={"sum_relative": 1e-3},
    )
    assert not report["tolerances_pre_registered"]
    assert report["tolerances"]["sum_relative"] == 1e-3


# ---------------------------------------------------------------------------
# The representation the SIESTA reference actually produces
# ---------------------------------------------------------------------------


def test_the_siesta_style_split_passes_every_measurable_check_and_is_still_NO_GO():
    left, right = _true_split()
    direction = collective(np.array([[0.3, -0.4, 0.1], [0.2, 0.5, -0.6]]), name="v")
    split = inter_atomic_split(_ds_records(left, right), direction.vectors, ORBITAL_ATOM)
    artifact = from_S_L_S_R(
        split["S_left"], split["S_right"], _context(direction=direction), backend="siesta",
        method="dhsdr_atom_resolved_split__inter_atomic_only__intra_atomic_omitted",
        intra_atomic_included=False, isc_off=ISC,
    )

    rng = np.random.default_rng(3)
    no_u = ORBITAL_ATOM.size
    k = (0.25, 0.0, 0.0)
    c = rng.normal(size=(no_u, no_u)) + 1j * rng.normal(size=(no_u, no_u))
    state = ElectronicState(
        label="k_quarter",
        k=k,
        D_H=bloch_sum(rng.normal(size=(ISC.shape[0], no_u, no_u)), ISC, k),
        C=c,
        eps=np.array([-2.0, -0.5, 0.5, 3.0]),
    )
    report = validate_basis_response(
        artifact,
        D_S_reference=split["S_left"] + split["S_right"],
        reference_name="inter_atomic_dS_sum",
        states=[state],
        atom_of_orbital=ORBITAL_ATOM,
    )
    assert all(check["passed"] for check in report["checks"] if check["applicable"])
    assert report["verdict"] == VERDICT_NO_GO
    assert report["provenance"]["epc_backend_class"] == "hybrid"


# ---------------------------------------------------------------------------
# The runner over the certified SIESTA reference
# ---------------------------------------------------------------------------

REFERENCE_ROOT = REPO_ROOT / "Comparison/results/epc/siesta_reference/graphene"
GO2_REPORT = REPO_ROOT / "Comparison/results/epc/certification/graphene/go2_certification.json"


def test_the_runner_refuses_to_validate_against_an_uncertified_reference(tmp_path):
    import certify_basis_response as runner

    report = tmp_path / "go2.json"
    report.write_text('{"verdict": "FAIL", "comparisons": []}', encoding="utf-8")
    with pytest.raises(runner.BasisResponseCertificationError, match="certify nothing"):
        runner.load_go2_report(report)
    with pytest.raises(runner.BasisResponseCertificationError, match="is missing"):
        runner.load_go2_report(tmp_path / "absent.json")


def test_the_runner_puts_a_sparse_field_on_the_response_image_table():
    import certify_basis_response as runner

    field = {(0, 1, (0, 0, 0)): 2.0, (1, 0, (1, 0, 0)): -3.0, (0, 0, (7, 7, 7)): 5.0}
    blocks, outside = runner.dense_blocks(field, ISC, ORBITAL_ATOM.size)
    assert blocks.shape == (ISC.shape[0], ORBITAL_ATOM.size, ORBITAL_ATOM.size)
    assert blocks[0, 0, 1] == 2.0 and blocks[1, 1, 0] == -3.0
    # An element outside the table is reported, never silently truncated.
    assert outside == 5.0


def test_the_runner_solves_the_pencil_S_orthonormally():
    import certify_basis_response as runner

    toy = _toy()
    eps, c = runner.solve_pencil(toy["H"], toy["S"])
    assert np.allclose(c.conj().T @ toy["S"] @ c, np.eye(N_BASIS), atol=1e-10)
    assert np.allclose(toy["H"] @ c, toy["S"] @ c @ np.diag(eps), atol=1e-10)


@pytest.mark.slow
@pytest.mark.skipif(
    not (REFERENCE_ROOT / "epc_siesta_reference_manifest.json").is_file() or not GO2_REPORT.is_file(),
    reason="the S11 SIESTA reference campaign / GO-2 report is not in this checkout",
)
def test_graphene_reference_passes_every_check_and_is_NO_GO_on_the_intra_atomic_term():
    """The real gate: the SIESTA-derived split is right, and it is incomplete."""
    pytest.importorskip("sisl")
    pytest.importorskip("netCDF4")
    import certify_basis_response as runner

    report = runner.certify(REFERENCE_ROOT, GO2_REPORT)
    assert report["summary"]["failed_checks"] == []
    assert report["summary"]["directions_blocked"] == []
    assert report["verdict"] == "NO_GO"
    for row in report["directions"]:
        terms = {term["term"]: term for term in row["validation"]["omitted_terms"]}
        assert terms["intra_atomic_basis_response"]["status"] == "unresolved"
        # The two measurable approximations must carry a number, not a promise.
        contribution = row["validation"]["contributions"][0]["approximations"]
        assert contribution[FORMALISM_ID_SYMMETRIC]["interband_frobenius"] >= 0.0
        assert contribution[FORMALISM_ID_IGNORE_OVERLAP]["frobenius"] > 0.0
