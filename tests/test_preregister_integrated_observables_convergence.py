"""E-F_001-S53: convergence protocol for integrated rigid-model observables (Fase 12)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT / "Comparison" / "scripts", REPO_ROOT / "shared"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import preregister_integrated_observables_convergence as mod  # noqa: E402


def test_monkhorst_pack_2d_weights_sum_to_one_and_point_count():
    points, weights = mod.monkhorst_pack_2d(12, 8)
    assert points.shape == (96, 2)
    assert np.isclose(weights.sum(), 1.0)
    assert np.all((points >= 0.0) & (points < 1.0))


def test_monkhorst_pack_2d_rejects_nonpositive_density():
    with pytest.raises(mod.PreregistrationError):
        mod.monkhorst_pack_2d(0, 4)


def test_modal_completeness_rejects_selected_modes_and_accepts_full_sum():
    partial = mod.modal_completeness_report(4, n_atoms_per_cell=11164)
    assert partial["required_branches"] == 33492
    assert partial["complete"] is False

    full = mod.modal_completeness_report(33492, n_atoms_per_cell=11164)
    assert full["complete"] is True


def test_dos_sum_rule_independent_of_broadening_kind():
    rng = np.random.default_rng(1)
    eigs = rng.uniform(-1.0, 1.0, size=(50, 3))
    _, weights = mod.monkhorst_pack_2d(10, 5)
    grid = np.linspace(-5.0, 5.0, 2001)
    expected = 4 * eigs.shape[1] / 2.0
    for kind in mod.BROADENING_FUNCTIONS:
        dos = mod.gaussian_broadened_dos(
            eigs, weights, grid, width_ev=0.05, cell_area_ang2=2.0, spin_valley_mult=4, kind=kind
        )
        integral = np.trapezoid(dos, grid)
        tol = 5e-3 if kind == "gaussian" else 1e-2
        assert abs(integral - expected) / expected < tol


def test_dos_rejects_mismatched_weight_count():
    eigs = np.zeros((5, 2))
    with pytest.raises(mod.PreregistrationError):
        mod.gaussian_broadened_dos(eigs, np.ones(4), np.array([0.0]), width_ev=0.1, cell_area_ang2=1.0,
                                    spin_valley_mult=4)


def test_mesh_convergence_report_detects_plateau_and_declares_uncertainty():
    plateau = mod.mesh_convergence_report("plateau_case", [10, 20, 40], [1.0, 1.001, 1.0011], tol=0.01)
    assert plateau["evidence"] == "plateau"
    assert plateau["declared_uncertainty"] is None

    not_converged = mod.mesh_convergence_report("open_case", [1, 2], [1.0, 2.0], tol=0.01)
    assert not_converged["evidence"] == "uncertainty_declared"
    assert not_converged["declared_uncertainty"] == pytest.approx(0.5)


def test_spin_valley_weight_includes_two_valleys():
    assert mod.spin_valley_weight(2) == 4
    assert mod.spin_valley_weight(1) == 2


def test_build_protocol_reads_go9_live_and_stays_blocked_until_a_g_set_exists():
    protocol = mod.build_protocol()
    assert protocol["ready_to_execute"] is False
    assert protocol["physics_review"]["status"] == "BLOCKED"
    names = {row["name"] for row in protocol["preconditions"]}
    assert names == {"GO-9 rigid-Gamma campaign (S43)", "GO-9 rigid q!=0 campaign (S45)"}
    assert any("g_set_empty" in reason for reason in protocol["blocking"])
    assert protocol["scope"]["explicitly_out_of_scope"] == ["Tc"]


def test_protocol_freeze_hash_is_stable_and_tamper_evident():
    protocol = mod.build_protocol()
    recomputed = mod.prereg.freeze_hash(protocol)
    assert recomputed == protocol["frozen_content_sha256"]

    tampered = dict(protocol)
    tampered["ready_to_execute"] = True
    assert mod.prereg.freeze_hash(tampered) != protocol["frozen_content_sha256"]


def test_require_frozen_protocol_round_trip(tmp_path):
    output_dir = tmp_path / "matbg"
    protocol = mod.build_protocol()
    output_dir.mkdir(parents=True)
    path = output_dir / mod.PROTOCOL_NAME
    import json

    path.write_text(json.dumps(protocol, indent=2, sort_keys=True), encoding="utf-8")

    with pytest.raises(mod.PreregistrationError):
        mod.require_frozen_protocol(path)

    verified = mod.verify(path)
    assert verified["intact"] is True
    assert verified["verified"] is True


def test_selfcheck_runs_clean():
    mod._selfcheck()
