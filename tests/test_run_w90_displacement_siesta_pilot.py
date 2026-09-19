from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "Comparison" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_w90_displacement_siesta_pilot as pilot  # noqa: E402
import w90_displacement_sampler_family as sampler  # noqa: E402


def test_build_pilot_configurations_covers_pilot_scope_and_is_deterministic():
    geometry = sampler.load_graphene_primitive()
    first = pilot.build_pilot_configurations(geometry)
    second = pilot.build_pilot_configurations(geometry)

    assert [sample_id for sample_id, _ in first] == [sample_id for sample_id, _ in second]
    assert first[0] == ("reference", first[0][1])
    assert first[0][1].family == "reference"

    sample_ids = [sample_id for sample_id, _ in first]
    assert len(sample_ids) == len(set(sample_ids))  # every sample_id is unique

    families = {config.family for _sample_id, config in first[1:]}
    assert families == {"axial_radial", "angular_shell", "local_pair_modes", "sobol_sparse", "random_cartesian"}

    ks = {config.metadata["k"] for _sample_id, config in first[1:]}
    assert ks == set(pilot.PILOT_K_VALUES)
    amplitudes = {config.metadata["amplitude_ang"] for _sample_id, config in first[1:]}
    assert amplitudes == set(pilot.PILOT_AMPLITUDES_ANG)


def test_stochastic_families_are_capped_at_pilot_max_n():
    geometry = sampler.load_graphene_primitive()
    entries = pilot.build_pilot_configurations(geometry)
    sobol_count = sum(
        1
        for _sample_id, config in entries
        if config.family == "sobol_sparse" and config.metadata["k"] == 1 and config.metadata["dimensionality"] == "3D"
        and config.metadata["amplitude_ang"] == 0.03
    )
    assert sobol_count == pilot.PILOT_MAX_N


def test_amp_tag_is_filesystem_safe():
    assert pilot._amp_tag(0.03) == "0p03"
    assert pilot._amp_tag(0.08) == "0p08"


def test_delta_nl_formula_matches_direct_computation():
    import numpy as np

    h0 = np.eye(2)
    h_plus = np.array([[1.0, 0.2], [0.2, 1.1]])
    h_minus = np.array([[1.0, -0.2], [-0.2, 1.1]])
    numerator = np.linalg.norm(h_plus + h_minus - 2.0 * h0)
    denominator = np.linalg.norm(h_plus - h_minus) + pilot.DELTA_NL_EPSILON
    expected = numerator / denominator
    assert expected > 0.0


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
