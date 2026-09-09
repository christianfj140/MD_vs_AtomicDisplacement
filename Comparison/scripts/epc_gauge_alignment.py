"""Align the Graph2Mat and PAO electronic gauges *per geometry*, before differencing.

Graph2Mat's target is what ``sisl.read_hamiltonian()`` returns,

    K^{E_F}(R) = H_abs(R) - E_F(R) . S(R)

while the primary PAO reference is referred to the vacuum level,

    K^{c_vac}(R) = H_abs(R) - c_vac(R) . S(R)

so the two differ by a geometry-dependent multiple of the overlap:

    K^{E_F}(R) = K^{c_vac}(R) + [c_vac(R) - E_F(R)] . S(R)                (1)

``gauge_derivative_audit = PASS`` established that this conversion is *known*.
It did not apply it. The comparator (``quantify_checkpoint_derivative_error``)
still forms its model-side derivative in one gauge and its reference in the
other, and says so itself: its ``decision`` is ``UNDECIDED`` with
``"common electrostatic gauge not yet wired into this comparison artifact"``
listed first among ``decision_blockers``.

**Why it must be applied per geometry, not afterwards.** Differentiating (1)
along a displacement direction ``v`` gives

    dK^{E_F} = dK^{c_vac} + [c_vac' - E_F'] . S + [c_vac - E_F] . dS        (2)

Both correction terms are first order. An a-posteriori shift of the finished
derivative can only ever supply the first one, and only if ``c_vac' - E_F'`` is
treated as a constant -- which is precisely the approximation the audit refused.
Applying (1) to each displaced geometry *before* the five-point stencil runs
makes the stencil produce (2) exactly, with no term dropped and no linearity
assumed: the stencil is linear, so differencing the converted Hamiltonians is
the same operation as converting the exact derivative, to the stencil's own
order.

The certified scalar for the equilibrium graphene C1_x direction is

    d(c_vac - E_F)/dR = 3.4869062475910892e-3 eV/Ang

(``Comparison/results/epc/pao_flow_audit/graph2mat_target_gauge_contract.json``,
``C1_x_five_point_derivatives_ev_per_ang.d_cvac_minus_Ef``). That is not
negligible against the 5 meV/Ang numerical convergence budget, which is why
``epc_claim_policy`` keeps ``final_comparator_gauge_conversion_verified`` as a
separate, non-optional condition.

**This module computes no residual and adjudicates nothing.** It supplies the
transformation and the stencil identity; wiring it into the comparator's verdict
is a separate step, and ``final_comparator_gauge_conversion_verified`` stays
false until that happens.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

Key = Any

# Central five-point first-derivative weights, w_j for j in {-2,-1,1,2}, over 12h.
# Same stencil the gauge contract used to certify d_cvac_minus_Ef.
FIVE_POINT_WEIGHTS: dict[int, float] = {-2: 1.0, -1: -8.0, 1: 8.0, 2: -1.0}
FIVE_POINT_DENOMINATOR = 12.0


def _combine(*terms: tuple[float, Mapping[Key, float]]) -> dict[Key, float]:
    """``sum_i c_i F_i`` over the union of the fields' index spaces."""
    total: dict[Key, float] = {}
    for coefficient, field in terms:
        for key, value in field.items():
            total[key] = total.get(key, 0.0) + float(coefficient) * float(value)
    return total


def to_fermi_gauge(
    k_cvac: Mapping[Key, float],
    overlap: Mapping[Key, float],
    *,
    c_vac_ev: float,
    fermi_ev: float,
) -> dict[Key, float]:
    """``K^{E_F}(R) = K^{c_vac}(R) + [c_vac(R) - E_F(R)] . S(R)`` at ONE geometry.

    Both levels belong to the same geometry ``R``; passing levels from different
    geometries is the mistake this function exists to prevent.
    """
    return _combine((1.0, k_cvac), (float(c_vac_ev) - float(fermi_ev), overlap))


def to_vacuum_gauge(
    k_fermi: Mapping[Key, float],
    overlap: Mapping[Key, float],
    *,
    c_vac_ev: float,
    fermi_ev: float,
) -> dict[Key, float]:
    """The inverse of :func:`to_fermi_gauge`, at one geometry."""
    return _combine((1.0, k_fermi), (float(fermi_ev) - float(c_vac_ev), overlap))


def five_point_derivative(
    samples: Mapping[int, Mapping[Key, float]], *, step_ang: float
) -> dict[Key, float]:
    """Central five-point derivative of a sparse field sampled at ``j . h``."""
    missing = sorted(set(FIVE_POINT_WEIGHTS) - set(samples))
    if missing:
        raise ValueError(f"five-point stencil needs offsets {sorted(FIVE_POINT_WEIGHTS)}, missing {missing}")
    scale = FIVE_POINT_DENOMINATOR * float(step_ang)
    return _combine(
        *((weight / scale, samples[offset]) for offset, weight in FIVE_POINT_WEIGHTS.items())
    )


def five_point_scalar_derivative(values: Mapping[int, float], *, step_ang: float) -> float:
    """The same stencil on a scalar (used for ``c_vac`` and ``E_F`` themselves)."""
    return sum(
        weight * float(values[offset]) for offset, weight in FIVE_POINT_WEIGHTS.items()
    ) / (FIVE_POINT_DENOMINATOR * float(step_ang))


def aligned_derivative(
    samples: Sequence[Mapping[str, Any]], *, step_ang: float
) -> dict[str, Any]:
    """Gauge-align every geometry, THEN differentiate. The order is the point.

    ``samples`` is one entry per stencil offset, each carrying that geometry's
    own ``offset``, ``k_cvac``, ``overlap``, ``c_vac_ev`` and ``fermi_ev``.

    Returns the aligned derivative together with the two decomposition terms of
    equation (2), so a caller can see that neither was dropped -- but it draws
    no conclusion from them.
    """
    by_offset = {int(entry["offset"]): entry for entry in samples}

    converted = {
        offset: to_fermi_gauge(
            entry["k_cvac"],
            entry["overlap"],
            c_vac_ev=entry["c_vac_ev"],
            fermi_ev=entry["fermi_ev"],
        )
        for offset, entry in by_offset.items()
    }
    aligned = five_point_derivative(converted, step_ang=step_ang)

    # The same stencil applied to the unconverted field and to the scalars, so
    # the identity (2) can be checked term by term rather than asserted.
    unconverted = five_point_derivative(
        {offset: entry["k_cvac"] for offset, entry in by_offset.items()}, step_ang=step_ang
    )
    d_gauge = five_point_scalar_derivative(
        {
            offset: float(entry["c_vac_ev"]) - float(entry["fermi_ev"])
            for offset, entry in by_offset.items()
        },
        step_ang=step_ang,
    )
    d_overlap = five_point_derivative(
        {offset: entry["overlap"] for offset, entry in by_offset.items()}, step_ang=step_ang
    )
    equilibrium = by_offset.get(0)
    gauge_at_equilibrium = (
        float(equilibrium["c_vac_ev"]) - float(equilibrium["fermi_ev"])
        if equilibrium is not None
        else None
    )

    return {
        "aligned_derivative": aligned,
        "unconverted_derivative": unconverted,
        "d_cvac_minus_Ef_ev_per_ang": d_gauge,
        "d_overlap": d_overlap,
        "cvac_minus_Ef_at_equilibrium_ev": gauge_at_equilibrium,
        "applied": "per geometry, before the stencil",
        "computes_no_residual": True,
    }
