"""E-F_001-S7: occupations, chemical potential and who is allowed to move it.

The single formula under test is

    d(mu) = sum_n w_n f'_n d(eps_n) / sum_n w_n f'_n ,

and it is not checked against itself: ``mu(t)`` is re-solved from ``N(mu) = N``
at ``t = +-h`` and differentiated numerically, which is the definition the
formula is derived from. The rest of the file is about the contract — which
artifact may see a smearing, and which may subtract ``d(mu)``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "Comparison" / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "shared"))

from epc_formalism import (  # noqa: E402
    FORMALISM_ID,
    basis_response_S_L_S_R,
    epc_matrix_elements,
)
from epc_occupations import (  # noqa: E402
    FERMI_DIRAC,
    GAUSSIAN,
    MU_CONVENTION_HAMILTONIAN_FERMI_ZERO,
    EpcOccupationError,
    OccupationContract,
    chemical_potential_response,
    contract_from_neutrality_reference,
    deformation_potentials_relative_to_mu,
    eigenvalue_directional_derivatives,
    electron_count,
    occupation_derivative,
    occupations,
    occupations_contract,
    resolve_chemical_potential,
    with_smearing,
)

N_STATES = 41
WEIGHTS = np.full(N_STATES, 1.0 / N_STATES)  # one band, 41 k points


def _metal(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """A half-filled band and a directional derivative of it."""
    rng = np.random.default_rng(seed)
    eps = np.linspace(-3.0, 3.0, N_STATES)
    return eps, rng.normal(scale=0.5, size=N_STATES)


def _contract(**overrides) -> OccupationContract:
    base = {
        "electron_count": 1.0,  # spin 2 x half of one band
        "spin_degeneracy": 2,
        "occupation_function": FERMI_DIRAC,
        "smearing_width_ev": 0.15,
        "mu_convention": MU_CONVENTION_HAMILTONIAN_FERMI_ZERO,
    }
    return OccupationContract(**{**base, **overrides})


# -- the formula ------------------------------------------------------------


@pytest.mark.parametrize("function", [FERMI_DIRAC, GAUSSIAN])
def test_dmu_is_the_derivative_of_the_mu_that_holds_N_fixed(function):
    contract = _contract(occupation_function=function)
    eps, d_eps = _metal()
    analytic = chemical_potential_response(contract, eps, d_eps, weights=WEIGHTS)["d_mu"]

    h = 1e-4
    moved = [
        resolve_chemical_potential(contract, eps + sign * h * d_eps, WEIGHTS)["mu_ev"]
        for sign in (+1, -1)
    ]
    numeric = (moved[0] - moved[1]) / (2 * h)
    assert analytic == pytest.approx(numeric, abs=1e-7)


def test_a_rigid_shift_of_the_spectrum_moves_mu_by_exactly_that_shift():
    """H -> H + cS: every eigenvalue moves by c, so mu does too and nothing is left."""
    contract = _contract()
    eps, _ = _metal()
    shift = np.full(N_STATES, 0.37)
    response = chemical_potential_response(contract, eps, shift, weights=WEIGHTS)
    assert response["d_mu"] == pytest.approx(0.37, abs=1e-12)
    deformation = deformation_potentials_relative_to_mu(contract, eps, shift, weights=WEIGHTS)
    assert np.allclose(deformation["values"], 0.0, atol=1e-12)


def test_the_resolved_mu_actually_holds_the_declared_electron_count():
    contract = _contract()
    eps, _ = _metal()
    resolved = resolve_chemical_potential(contract, eps, WEIGHTS)
    assert resolved["method"] == "solved_from_electron_count"
    assert electron_count(contract, eps, resolved["mu_ev"], WEIGHTS) == pytest.approx(1.0, abs=1e-10)


def test_occupations_and_their_derivative_are_physical():
    contract = _contract()
    eps, _ = _metal()
    mu = resolve_chemical_potential(contract, eps, WEIGHTS)["mu_ev"]
    f = occupations(contract, eps, mu)
    assert np.all((f >= 0.0) & (f <= 1.0))
    assert f[0] > 0.99 and f[-1] < 0.01  # deep states full, high states empty
    assert np.all(occupation_derivative(contract, eps, mu) <= 0.0)
    # f' against a real derivative needs a grid finer than the smearing width
    fine = np.linspace(mu - 1.0, mu + 1.0, 2001)
    numeric = np.gradient(occupations(contract, fine, mu), fine)[1:-1]  # drop one-sided edges
    assert np.allclose(occupation_derivative(contract, fine, mu)[1:-1], numeric, atol=1e-6)


# -- fixed-N versus fixed-mu ------------------------------------------------


def test_fixed_mu_refuses_to_produce_a_chemical_potential_response():
    contract = _contract(mu_policy="fixed_mu", mu_ev=0.0)
    eps, d_eps = _metal()
    assert resolve_chemical_potential(contract)["method"] == "externally_fixed"
    for call in (chemical_potential_response, deformation_potentials_relative_to_mu):
        with pytest.raises(EpcOccupationError, match="identically zero"):
            call(contract, eps, d_eps, weights=WEIGHTS)


def test_fixed_N_at_zero_temperature_reuses_the_solvers_neutrality_estimate():
    eps, d_eps = _metal()
    blind = OccupationContract(
        electron_count=1.0,
        spin_degeneracy=2,
        occupation_function="zero_temperature_step",
        smearing_width_ev=0.0,
        mu_convention=MU_CONVENTION_HAMILTONIAN_FERMI_ZERO,
    )
    with pytest.raises(EpcOccupationError, match="neutrality reference"):
        resolve_chemical_potential(blind, eps, WEIGHTS)

    from_solver = contract_from_neutrality_reference(
        {
            "method": "uniform_kmesh_generalized_inertia_zero_temperature",
            "energy_eV": 0.05,
            "chemical_potential_available": True,
            "neutral_electrons": 1,
            "spin_degeneracy": 2,
        }
    )
    assert resolve_chemical_potential(from_solver, eps, WEIGHTS)["mu_ev"] == 0.05
    # ... but a step function still has no Fermi surface to respond with
    with pytest.raises(EpcOccupationError, match="no Fermi surface"):
        chemical_potential_response(from_solver, eps, d_eps, weights=WEIGHTS)


def test_a_gapped_spectrum_has_no_dmu_but_keeps_its_eigenvalue_derivatives():
    contract = _contract(electron_count=2.0, smearing_width_ev=0.01)  # both valence states full
    eps = np.array([-5.0, -5.0, 5.0, 5.0])
    d_eps = np.array([0.1, -0.2, 0.3, 0.4])
    weights = np.full(4, 0.5)
    with pytest.raises(EpcOccupationError, match="no Fermi surface"):
        chemical_potential_response(contract, eps, d_eps, weights=weights)
    assert eigenvalue_directional_derivatives(_block(), contract)["mu_subtracted"] is False


def test_the_neutrality_reference_without_a_chemical_potential_stays_unresolved():
    evidence_only = contract_from_neutrality_reference(
        {
            "method": "exported_hamiltonian_is_already_aligned_to_E_F_zero",
            "expected_neutral_valence_electrons": 8,
            "spin_degeneracy": 2,
            "chemical_potential_available": False,
            "limitation": "no converged 2-D chemical potential exists",
        },
        occupation_function=FERMI_DIRAC,
        smearing_width_ev=0.075,
    )
    assert evidence_only.mu_ev is None and evidence_only.electron_count == 8.0
    with pytest.raises(EpcOccupationError, match="no electron count"):
        contract_from_neutrality_reference({"limitation": "element.dat is unavailable"})


# -- what each artifact is allowed to know ----------------------------------


def _block(seed: int = 3, size: int = 4):
    rng = np.random.default_rng(seed)
    d_h = rng.normal(size=(size, size))
    d_h = d_h + d_h.T
    s_left = rng.normal(size=(size, size))
    response = basis_response_S_L_S_R(
        s_left, s_left.T, intra_atomic_included=True, backend="toy"
    )
    c = np.eye(size)
    eps = np.linspace(-1.0, 1.0, size)
    return epc_matrix_elements(d_h, response, C_row=c, C_col=c, eps_row=eps, eps_col=eps)


def test_eigenvalue_derivatives_declare_mu_but_never_a_smearing():
    block = _block()
    hot = eigenvalue_directional_derivatives(block, _contract(smearing_width_ev=0.5))
    cold = eigenvalue_directional_derivatives(block, _contract(smearing_width_ev=0.01))
    assert np.array_equal(hot["values"], cold["values"])  # re-smearing changes nothing here
    assert hot["mu_subtracted"] is False
    assert hot["mu_convention"] == MU_CONVENTION_HAMILTONIAN_FERMI_ZERO
    assert hot["formalism_id"] == FORMALISM_ID
    assert np.allclose(hot["values"], np.real(np.diag(block.values)))


def test_the_overlap_free_control_negative_has_no_eigenvalue_derivative():
    rng = np.random.default_rng(1)
    d_h = rng.normal(size=(3, 3))
    control = epc_matrix_elements(
        d_h + d_h.T,
        None,
        C_row=np.eye(3),
        C_col=np.eye(3),
        eps_row=np.zeros(3),
        eps_col=np.zeros(3),
        control_negative=True,
    )
    with pytest.raises(EpcOccupationError, match="control negative"):
        eigenvalue_directional_derivatives(control, _contract())


def test_deformation_potentials_subtract_dmu_once_and_say_so():
    contract = _contract()
    eps, d_eps = _metal()
    d_mu = chemical_potential_response(contract, eps, d_eps, weights=WEIGHTS)["d_mu"]
    bands = [10, 20]
    result = deformation_potentials_relative_to_mu(
        contract, eps, d_eps, bands=bands, d_eps_bands=d_eps[bands], weights=WEIGHTS
    )
    assert np.allclose(result["values"], d_eps[bands] - d_mu)
    assert result["smearing_width_ev"] == contract.smearing_width_ev
    assert result["mu_policy"] == "fixed_N"


def test_re_smearing_changes_the_occupation_response_and_only_that():
    contract = _contract()
    eps, d_eps = _metal()
    other = with_smearing(contract, FERMI_DIRAC, 0.4)
    assert other.electron_count == contract.electron_count
    assert other.eigenspace_fields() == contract.eigenspace_fields()  # what g depends on
    assert other.artifact_fields() != contract.artifact_fields()
    first = chemical_potential_response(contract, eps, d_eps, weights=WEIGHTS)["d_mu"]
    second = chemical_potential_response(other, eps, d_eps, weights=WEIGHTS)["d_mu"]
    assert first != second


def test_the_contract_fails_closed_on_incoherent_declarations():
    for broken in (
        {"occupation_function": "lorentzian"},
        {"mu_policy": "whatever"},
        {"occupation_function": "zero_temperature_step"},  # width 0.15 contradicts it
        {"smearing_width_ev": 0.0},  # fermi_dirac at exactly T = 0
        {"mu_convention": "  "},
        {"spin_degeneracy": 3},
        {"electron_count": 0},
        {"mu_policy": "fixed_mu"},  # without mu_ev
    ):
        with pytest.raises(EpcOccupationError):
            _contract(**broken)


def test_the_published_contract_names_the_four_artifacts():
    published = occupations_contract()
    assert set(published["artifacts"]) == {
        "pao_projected_mode_coupling",
        "eigenvalue_directional_derivative",
        "deformation_potential_relative_to_mu",
        "fermi_surface_response",
    }
    assert published["eigenspace_declares"] == ["electron_count", "spin_degeneracy", "mu_convention"]
