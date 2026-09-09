"""C19-C20 / E-F_001-S23: the graphene Gamma pre-registration, checked.

A pre-registration is only worth the freeze if the freeze holds, so what is
tested here is not "does the script run" but the four properties that make it a
pre-registration at all:

* **no leakage.** The three splits must be disjoint, and -- because the stage-1
  ``k`` is chosen by a rule over spectra measured at freeze time -- the chosen
  ``k`` must be the declared result ``k`` and not one a tolerance is fitted on.
  The static table cannot see that second failure mode, so it is tested
  separately;
* **the mode is fixed by a rule, not by an index.** Feeding the doublet in a
  rotated gauge must produce the same two members, since "band 4" is the
  solver's bookkeeping and "the one that stretches the bond" is not;
* **the tolerances live in the electronic subspace** and ``tau_model`` names the
  sources it refuses -- C14's matrix-level 0.67 among them;
* **the hash is a freeze.** Editing any non-volatile field must break
  ``require_frozen_protocol``, and a result that does not cite the hash must be
  caught by ``verify``.

Everything runs without SIESTA: spectra are injected through the provider hook
that ``build_protocol`` takes for exactly this reason. The produced protocol,
when it exists, is then checked as an artifact.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "Comparison" / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "shared"))

import fd_perturbation_space as fdp  # noqa: E402
import preregister_graphene_gamma_epc as prereg  # noqa: E402

CELL = np.array([[2.2005, -1.2704592674, 0.0], [2.2005, 1.2704592674, 0.0], [0.0, 0.0, 29.34]])
POSITIONS = np.array([[1.467, 0.0, 0.0], [2.934, 0.0, 0.0]])
MASSES = np.array([12.0107, 12.0107])

PROTOCOL = prereg.DEFAULT_OUTPUT_DIR / prereg.PROTOCOL_NAME


# --------------------------------------------------------------------------- #
# Splits: the leakage guards
# --------------------------------------------------------------------------- #


def test_frozen_splits_are_disjoint() -> None:
    prereg.assert_splits_disjoint()


def test_overlapping_splits_are_refused() -> None:
    splits = {
        "calibration": {"directions": ("a",), "k_labels": ("gamma",), "fc_ranges": ()},
        "result": {"directions": ("a",), "k_labels": ("gamma",), "fc_ranges": ()},
    }
    with pytest.raises(prereg.PreregistrationError, match="leakage|share"):
        prereg.assert_splits_disjoint(splits)


def test_shared_direction_alone_is_leakage() -> None:
    """Same direction at a different k still means a floor measured on the result."""
    splits = {
        "calibration": {"directions": ("a",), "k_labels": ("gamma",), "fc_ranges": ()},
        "result": {"directions": ("a",), "k_labels": ("K",), "fc_ranges": ()},
    }
    with pytest.raises(prereg.PreregistrationError, match="share directions"):
        prereg.assert_splits_disjoint(splits)


def test_result_k_of_the_frozen_table_passes_the_selection_guard() -> None:
    for label in prereg.SPLITS["result"]["k_labels"]:
        prereg.assert_selection_outside_fitted_splits(label)


@pytest.mark.parametrize("label", ["gamma", "M", "k_generic_2"])
def test_selecting_a_fitted_k_is_refused(label: str) -> None:
    with pytest.raises(prereg.PreregistrationError, match="leakage|does not"):
        prereg.assert_selection_outside_fitted_splits(label)


def test_selection_rule_cannot_promote_the_validation_k() -> None:
    """If candidate 1 failed the rule, candidate 2 is the validation k: refuse.

    This is the failure the static disjointness check is blind to, and the one
    that would quietly fit and measure a tolerance in the same place.
    """
    spectra = [
        {"label": "k_generic_1", "min_window_gap_ev": 0.001,
         "min_gap_to_states_outside_window_ev": 1.0},
        {"label": "k_generic_2", "min_window_gap_ev": 0.9,
         "min_gap_to_states_outside_window_ev": 1.0},
    ]
    candidates = [row for row in prereg.K_CANDIDATES if row["label"] in {"k_generic_1", "k_generic_2"}]
    with pytest.raises(prereg.PreregistrationError, match="validation split"):
        prereg.select_result_k(spectra, candidates)


def test_selection_rule_skips_high_symmetry_and_unresolved_windows() -> None:
    spectra = [
        {"label": "M", "min_window_gap_ev": 5.0, "min_gap_to_states_outside_window_ev": 5.0},
        {"label": "k_generic_1", "min_window_gap_ev": 0.8,
         "min_gap_to_states_outside_window_ev": 0.8},
    ]
    candidates = [row for row in prereg.K_CANDIDATES if row["label"] in {"M", "k_generic_1"}]
    selection = prereg.select_result_k(spectra, candidates)
    assert selection["selected"] == "k_generic_1"
    rejected = {row["label"]: row["rejected_because"] for row in selection["candidates"]}
    assert "symmetry" in rejected["M"]


def test_no_candidate_is_an_error_not_a_relaxation() -> None:
    spectra = [{"label": "k_generic_1", "min_window_gap_ev": 1e-4,
                "min_gap_to_states_outside_window_ev": 1e-4}]
    candidates = [row for row in prereg.K_CANDIDATES if row["label"] == "k_generic_1"]
    with pytest.raises(prereg.PreregistrationError, match="not to be relaxed"):
        prereg.select_result_k(spectra, candidates)


# --------------------------------------------------------------------------- #
# The mode: the gauge rule
# --------------------------------------------------------------------------- #


def test_bond_reference_is_mass_weighted_unit_norm_and_opposed() -> None:
    reference = prereg.bond_reference_pattern(POSITIONS, CELL, MASSES)
    pattern = np.asarray(reference["pattern_mass_weighted"])
    assert pattern.shape == (2, 3)
    assert np.isclose(np.linalg.norm(pattern), 1.0)
    assert np.allclose(pattern[0], -pattern[1])  # optical, not acoustic
    assert np.isclose(np.dot(pattern[0], [0.0, 0.0, 1.0]), 0.0)  # in plane


def test_bond_reference_takes_the_minimum_image() -> None:
    """The 1.467 Ang bond, not the 2.934 Ang one that ignores periodicity."""
    reference = prereg.bond_reference_pattern(POSITIONS, CELL, MASSES)
    assert np.linalg.norm(reference["bond_vector_ang"]) == pytest.approx(1.467, abs=1e-6)


def test_bond_reference_refuses_a_cell_that_is_not_the_two_atom_one() -> None:
    with pytest.raises(prereg.PreregistrationError, match="two-atom cell"):
        prereg.bond_reference_pattern(np.zeros((4, 3)), CELL, np.full(4, 12.0107))


def _doublet_spanning_the_reference(angle: float) -> np.ndarray:
    """A synthetic 6-branch eigenvector set whose branches 4,5 span the E2g plane."""
    reference = prereg.bond_reference_pattern(POSITIONS, CELL, MASSES)
    longitudinal = np.asarray(reference["pattern_mass_weighted"]).reshape(-1)
    normal = np.asarray(reference["in_plane_normal"], dtype=np.float64)
    transverse = np.stack([-normal, normal]).reshape(-1)
    transverse /= np.linalg.norm(transverse)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    span = rotation @ np.stack([longitudinal, transverse])
    vectors = np.zeros((6, 2, 3))
    vectors[4] = span[0].reshape(2, 3)
    vectors[5] = span[1].reshape(2, 3)
    return vectors


def test_gauge_rule_is_invariant_under_rotations_of_the_doublet() -> None:
    """Band 4 vs band 5 is the solver's gauge; the frozen member must not be."""
    reference = prereg.bond_reference_pattern(POSITIONS, CELL, MASSES)
    members = [
        prereg.gauge_fix_doublet(_doublet_spanning_the_reference(angle), (4, 5), reference)[
            "members"
        ]
        for angle in (0.0, 0.4, 1.1, -2.3)
    ]
    for rotated in members[1:]:
        for first, other in zip(members[0], rotated):
            assert first["label"] == other["label"]
            # to float64 roundoff: the sha256 is of the exact floats, so it
            # identifies the frozen value and is not itself a gauge invariant
            assert np.allclose(
                first["eigenvector_mass_weighted"], other["eigenvector_mass_weighted"], atol=1e-12
            )


def test_gauge_rule_puts_the_bond_stretch_first() -> None:
    reference = prereg.bond_reference_pattern(POSITIONS, CELL, MASSES)
    fixed = prereg.gauge_fix_doublet(_doublet_spanning_the_reference(0.7), (4, 5), reference)
    longitudinal, transverse = fixed["members"]
    assert abs(longitudinal["overlap_with_reference"]) == pytest.approx(1.0, abs=1e-10)
    assert abs(transverse["overlap_with_reference"]) < 1e-10
    assert fixed["reference_lies_in_the_doublet"] is True
    # orthonormal inside the span
    first = np.asarray(longitudinal["eigenvector_mass_weighted"]).reshape(-1)
    second = np.asarray(transverse["eigenvector_mass_weighted"]).reshape(-1)
    assert abs(float(first @ second)) < 1e-12
    assert np.isclose(np.linalg.norm(second), 1.0)


def test_gauge_rule_refuses_a_sector_that_is_not_a_doublet() -> None:
    reference = prereg.bond_reference_pattern(POSITIONS, CELL, MASSES)
    with pytest.raises(prereg.PreregistrationError, match="doublet"):
        prereg.gauge_fix_doublet(_doublet_spanning_the_reference(0.0), (3, 4, 5), reference)


def test_gauge_rule_refuses_to_guess_when_the_reference_misses_the_span() -> None:
    reference = prereg.bond_reference_pattern(POSITIONS, CELL, MASSES)
    out_of_plane = np.zeros((6, 2, 3))
    out_of_plane[4, 0, 2] = 1.0
    out_of_plane[5, 1, 2] = 1.0
    with pytest.raises(prereg.PreregistrationError, match="cannot fix a member"):
        prereg.gauge_fix_doublet(out_of_plane, (4, 5), reference)


def test_mode_direction_is_a_unit_frobenius_cartesian_pattern() -> None:
    reference = prereg.bond_reference_pattern(POSITIONS, CELL, MASSES)
    fixed = prereg.gauge_fix_doublet(_doublet_spanning_the_reference(0.0), (4, 5), reference)
    direction = prereg.mode_direction(fixed["members"][0], MASSES, name="e2g_bond_longitudinal")
    assert direction.kind == fdp.KIND_COLLECTIVE
    assert np.isclose(np.linalg.norm(direction.vectors), 1.0)
    # equal masses: u ~ e, so the cartesian pattern is still the opposed bond stretch
    assert np.allclose(direction.vectors[0], -direction.vectors[1])


# --------------------------------------------------------------------------- #
# The electronic window
# --------------------------------------------------------------------------- #


def test_window_straddles_the_neutrality_point_by_electron_count() -> None:
    values = np.arange(8.0)
    indices = prereg.window_indices(values)
    occupied = prereg.ELECTRON_COUNT // prereg.SPIN_DEGENERACY
    assert indices == [occupied - 2, occupied - 1, occupied, occupied + 1]


def test_window_that_does_not_fit_is_an_error() -> None:
    with pytest.raises(prereg.PreregistrationError, match="does not fit"):
        prereg.window_indices(np.arange(4.0))


def test_spectrum_report_measures_the_gaps_the_rule_reads() -> None:
    values = np.array([-9.0, -8.0, -3.0, -1.0, 1.0, 4.0, 7.0, 9.0])
    report = prereg.spectrum_report("probe", (0.1, 0.2, 0.0), values)
    assert report["window_indices"] == [2, 3, 4, 5]
    assert report["min_window_gap_ev"] == pytest.approx(2.0)
    assert report["min_gap_to_states_outside_window_ev"] == pytest.approx(3.0)


# --------------------------------------------------------------------------- #
# The error budget
# --------------------------------------------------------------------------- #


def test_every_tolerance_is_a_norm_in_the_electronic_subspace() -> None:
    for name in ("tau_num", "tau_backend", "tau_model"):
        assert prereg.TOLERANCES[name]["contracted_in"]
    assert "C_f^dag" in prereg.TOLERANCES["tau_model"]["definition"]


def test_tau_model_refuses_the_global_matrix_norm_shortcut() -> None:
    forbidden = " ".join(prereg.FORBIDDEN_TAU_MODEL_SOURCES).lower()
    assert "frobenius" in forbidden and "mae" in forbidden
    assert prereg.TOLERANCES["tau_model"]["forbidden_sources"] == prereg.FORBIDDEN_TAU_MODEL_SOURCES
    # calibrated on one split, transfer-tested on another, applied to a third
    assert "calibration" in prereg.TOLERANCES["tau_model"]["calibration"]
    assert "validation split" in prereg.TOLERANCES["tau_model"]["validation"]


def test_error_budget_is_complete_for_the_frozen_result_k() -> None:
    budget = prereg.error_budget_report(prereg.SPLITS["result"]["k_labels"][0])
    assert budget["complete"] is True
    assert budget["leakage"] is None


def test_error_budget_is_blocked_when_the_k_leaks() -> None:
    """The review item must be able to say no; before, it could not."""
    budget = prereg.error_budget_report("k_generic_2")
    assert budget["complete"] is False
    assert "validation split" in budget["leakage"]


def test_error_budget_is_blocked_when_a_tolerance_leaves_the_subspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patched = {name: dict(row) for name, row in prereg.TOLERANCES.items()}
    patched["tau_model"].pop("contracted_in")
    monkeypatch.setattr(prereg, "TOLERANCES", patched)
    assert prereg.error_budget_report(prereg.SPLITS["result"]["k_labels"][0])["complete"] is False


# --------------------------------------------------------------------------- #
# The freeze
# --------------------------------------------------------------------------- #


def _minimal_protocol() -> dict[str, object]:
    protocol = {
        "schema": prereg.SCHEMA,
        "protocol_id": prereg.PROTOCOL_ID,
        "generated_at": "2026-01-01T00:00:00+00:00",
        "frozen_content_sha256": None,
        "tolerances": {"tau_num": {"unit": "eV/Ang"}},
        "physics_review": {"status": "APPROVED", "items": []},
        "ready_to_execute": True,
        "blocking": [],
    }
    protocol["frozen_content_sha256"] = prereg.freeze_hash(protocol)
    return protocol


def test_freeze_hash_ignores_only_the_volatile_fields() -> None:
    protocol = _minimal_protocol()
    stamped = {**protocol, "generated_at": "2030-06-06T12:00:00+00:00"}
    assert prereg.freeze_hash(stamped) == prereg.freeze_hash(protocol)
    edited = {**protocol, "tolerances": {"tau_num": {"unit": "eV"}}}
    assert prereg.freeze_hash(edited) != prereg.freeze_hash(protocol)


def test_require_frozen_protocol_detects_an_edit(tmp_path: Path) -> None:
    protocol = _minimal_protocol()
    path = tmp_path / prereg.PROTOCOL_NAME
    path.write_text(json.dumps(protocol), encoding="utf-8")
    assert prereg.require_frozen_protocol(path)["protocol_id"] == prereg.PROTOCOL_ID

    protocol["tolerances"]["tau_num"]["unit"] = "meV"  # type: ignore[index]
    path.write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(prereg.PreregistrationError, match="edited since it was frozen"):
        prereg.require_frozen_protocol(path)


def test_require_frozen_protocol_refuses_a_blocked_review(tmp_path: Path) -> None:
    protocol = _minimal_protocol()
    protocol["physics_review"] = {"status": "BLOCKED", "items": [{"decision": "blocked"}]}
    protocol["frozen_content_sha256"] = prereg.freeze_hash(protocol)
    path = tmp_path / prereg.PROTOCOL_NAME
    path.write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(prereg.PreregistrationError, match="BLOCKED"):
        prereg.require_frozen_protocol(path)


def test_require_frozen_protocol_refuses_unmet_preconditions(tmp_path: Path) -> None:
    protocol = _minimal_protocol()
    protocol["ready_to_execute"] = False
    protocol["blocking"] = ["C14C basis response: NO_GO"]
    protocol["frozen_content_sha256"] = prereg.freeze_hash(protocol)
    path = tmp_path / prereg.PROTOCOL_NAME
    path.write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(prereg.PreregistrationError, match="preconditions are not met"):
        prereg.require_frozen_protocol(path)


def test_verify_catches_a_result_that_does_not_cite_the_protocol(tmp_path: Path) -> None:
    protocol = _minimal_protocol()
    path = tmp_path / prereg.PROTOCOL_NAME
    path.write_text(json.dumps(protocol), encoding="utf-8")
    results = tmp_path / "results"
    results.mkdir()

    assert prereg.verify(path, result_root=results)["verified"] is True

    (results / "stray.json").write_text(json.dumps({"g": 1.0}), encoding="utf-8")
    report = prereg.verify(path, result_root=results)
    assert report["intact"] is True
    assert report["verified"] is False
    assert report["results_not_citing_the_protocol"] == [str(results / "stray.json")]

    (results / "stray.json").write_text(
        json.dumps({"preregistration_sha256": protocol["frozen_content_sha256"]}), encoding="utf-8"
    )
    assert prereg.verify(path, result_root=results)["verified"] is True


# --------------------------------------------------------------------------- #
# The produced protocol, when the upstream artifacts are there
# --------------------------------------------------------------------------- #


def test_build_protocol_assembles_and_freezes_without_a_solver() -> None:
    """The assembly itself, with the spectra injected instead of diagonalised.

    ``build_protocol`` takes the provider hook precisely so the freeze can be
    exercised where sisl and the TSHS are not: everything else (mode, gauge,
    directions, topology, gates, review, hash) is the real path.
    """
    if not (prereg.DEFAULT_PHONON_ROOT / "graphene_gamma_phonon_manifest.json").is_file():
        pytest.skip("no upstream phonon/reference artifacts in this checkout")

    def spectra(_tshs: Path, points) -> list[dict]:
        # generic k: no degeneracy anywhere in the window; K: a Dirac pair
        rows = []
        for point in points:
            values = np.array([-9.0, -7.0, -3.0, -1.0, 1.0, 3.0, 6.0, 8.0])
            if point["label"] == "K":
                values[3] = values[4] = 0.0
            rows.append(prereg.spectrum_report(point["label"], point["k"], values))
        return rows

    protocol = prereg.build_protocol(spectra_provider=spectra)
    assert protocol["frozen_content_sha256"] == prereg.freeze_hash(protocol)
    assert protocol["electronic"]["k_selection"]["selected"] in prereg.SPLITS["result"]["k_labels"]
    assert [member["label"] for member in protocol["phonon"]["doublet"]["members"]] == list(
        prereg.SPLITS["result"]["directions"]
    )
    assert protocol["physics_review"]["status"] == "APPROVED"


@pytest.fixture(scope="module")
def protocol() -> dict:
    if not PROTOCOL.is_file():
        pytest.skip(f"no frozen protocol at {PROTOCOL}; run preregister_graphene_gamma_epc.py")
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


def test_frozen_file_is_intact(protocol: dict) -> None:
    assert prereg.freeze_hash(protocol) == protocol["frozen_content_sha256"]


def test_frozen_experiment_is_the_gamma_e2g_one(protocol: dict) -> None:
    assert protocol["experiment"]["q_label"] == "Gamma"
    assert protocol["experiment"]["q_fractional"] == [0.0, 0.0, 0.0]
    assert protocol["phonon"]["fc_range"] == prereg.RESULT_FC_RANGE
    assert len(protocol["phonon"]["doublet"]["members"]) == 2
    assert protocol["phonon"]["doublet"]["frequency_ev"] > 0.0
    assert protocol["phonon"]["doublet"]["reference_lies_in_the_doublet"] is True


def test_frozen_k_is_generic_and_resolved(protocol: dict) -> None:
    stage_1 = protocol["electronic"]["k_result_non_degenerate"]
    assert stage_1["label"] in prereg.SPLITS["result"]["k_labels"]
    assert stage_1["min_window_gap_ev"] >= prereg.MIN_WINDOW_GAP_EV
    assert protocol["electronic"]["k_result_degenerate"]["label"] == prereg.K_DEGENERATE["label"]


def test_frozen_directions_keep_the_neighbour_topology(protocol: dict) -> None:
    assert protocol["perturbation"]["topology_preserved"] is True
    assert protocol["perturbation"]["normalization"] == fdp.NORMALIZATION
    assert protocol["perturbation"]["delta_sweep_ang"] == list(prereg.DELTA_SWEEP_ANG)


def test_frozen_protocol_declares_three_paths_and_one_basis_response(protocol: dict) -> None:
    paths = {row["path"] for row in protocol["references"]["paths"]}
    assert paths == {"siesta_reference", "graph2mat_jvp", "graph2mat_frozen"}
    contract = protocol["references"]["basis_response"]
    assert contract["formalism_id"] == protocol["references"]["formalism_id"]
    assert contract["diagnostic_variants_are_never_the_result"] is True


def test_frozen_protocol_states_pass_fail_before_the_result(protocol: dict) -> None:
    assert protocol["claim_ladder"]
    assert {row["id"] for row in protocol["checks"]} >= {
        "generalized_hellmann_feynman_diagonal",
        "uniform_translation_null",
        "electronic_gauge_invariance",
        "phonon_doublet_gauge_invariance",
        "units_round_trip",
        "basis_response_present_and_required",
    }
    assert {row["axis"] for row in protocol["sensitivity"]} >= {"fc_range", "asr_policy", "delta"}


def test_the_result_does_not_exist_yet(protocol: dict) -> None:
    """"Pre" is a fact about time, and this is how it stays checkable."""
    report = prereg.verify(PROTOCOL, result_root=Path(protocol["result_root"]))
    assert report["intact"] is True
    assert report["results_not_citing_the_protocol"] == []


def test_blocked_preconditions_are_recorded_not_hidden(protocol: dict) -> None:
    required = [row for row in protocol["preconditions"] if row["required_before_execution"]]
    assert required
    unmet = [row["name"] for row in required if row["status"] != "PASS"]
    assert protocol["ready_to_execute"] == (not unmet and not protocol["blocking"])
