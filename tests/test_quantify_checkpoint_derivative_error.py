"""C14 / E-F_001-S15: the checkpoint-vs-certified-SIESTA derivative error report.

Synthetic fields only. What is under test is the accounting -- index mapping,
convention algebra, distributions and the verdict rules -- not Graph2Mat and not
SIESTA. The one test that touches the real artifacts is marked ``slow`` and skips
when they are absent.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "Comparison" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import fd_perturbation_space as fdp  # noqa: E402
import quantify_checkpoint_derivative_error as qc  # noqa: E402


class FakeMatrix:
    """Minimal stand-in for the ``(no, no * n_s)`` sparse layout."""

    def __init__(self, dense: np.ndarray) -> None:
        self.dense = np.asarray(dense, dtype=np.float64)
        self.shape = self.dense.shape

    def tocoo(self):
        from scipy import sparse

        return sparse.coo_matrix(self.dense)


class SparseFieldTests(unittest.TestCase):
    def test_columns_decompose_into_orbital_and_lattice_vector(self):
        dense = np.zeros((2, 4))
        dense[0, 0] = 1.5  # (row 0, col 0, R = (0,0,0))
        dense[1, 3] = -2.5  # (row 1, col 1, R = (1,0,0))
        field = qc.sparse_field(FakeMatrix(dense), [(0, 0, 0), (1, 0, 0)])
        self.assertEqual(field[(0, 0, (0, 0, 0))], 1.5)
        self.assertEqual(field[(1, 1, (1, 0, 0))], -2.5)
        self.assertEqual(len(field), 2)

    def test_supercell_order_inconsistent_with_the_shape_is_refused(self):
        with self.assertRaises(qc.CheckpointDerivativeError):
            qc.sparse_field(FakeMatrix(np.zeros((2, 4))), [(0, 0, 0)])


class ConventionAlgebraTests(unittest.TestCase):
    """``D_label = D_H - E_F D_S - (dE_F) S`` on the shared index space."""

    def test_label_reference_is_the_derivative_of_the_shifted_matrix(self):
        key_a = (0, 0, (0, 0, 0))
        key_b = (0, 1, (1, 0, 0))
        d_h = {key_a: 3.0, key_b: -1.0}
        d_s = {key_a: 0.5, key_b: 0.25}
        overlap = {key_a: 1.0, key_b: 0.1}
        fermi, d_fermi = -5.0, 0.2
        label = qc.combine((1.0, d_h), (-fermi, d_s), (-d_fermi, overlap))
        self.assertAlmostEqual(label[key_a], 3.0 + 5.0 * 0.5 - 0.2 * 1.0)
        self.assertAlmostEqual(label[key_b], -1.0 + 5.0 * 0.25 - 0.2 * 0.1)

    def test_combine_unions_index_spaces_instead_of_dropping_keys(self):
        left = {(0, 0, (0, 0, 0)): 1.0}
        right = {(1, 1, (0, 0, 0)): 2.0}
        combined = qc.combine((1.0, left), (3.0, right))
        self.assertEqual(combined[(0, 0, (0, 0, 0))], 1.0)
        self.assertEqual(combined[(1, 1, (0, 0, 0))], 6.0)


class OrbitalMapTests(unittest.TestCase):
    def setUp(self):
        # Two atoms, four orbitals each (s + p), 5 Ang cubic cell.
        self.mapping = qc.OrbitalMap(
            atom_of_orbital=[0, 0, 0, 0, 1, 1, 1, 1],
            l_of_orbital=[0, 1, 1, 1, 0, 1, 1, 1],
            positions_ang=np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]),
            cell_ang=np.eye(3) * 5.0,
        )

    def test_block_labels_come_from_the_orbital_angular_momenta(self):
        self.assertEqual(self.mapping.block((0, 0, (0, 0, 0))), "s-s")
        self.assertEqual(self.mapping.block((0, 5, (0, 0, 0))), "s-p")
        self.assertEqual(self.mapping.block((1, 5, (0, 0, 0))), "p-p")

    def test_distance_uses_the_periodic_image_of_the_column(self):
        self.assertAlmostEqual(self.mapping.distance_ang((0, 4, (0, 0, 0))), 1.5)
        self.assertAlmostEqual(self.mapping.distance_ang((0, 4, (-1, 0, 0))), 3.5)
        self.assertAlmostEqual(self.mapping.distance_ang((0, 0, (0, 0, 0))), 0.0)


class DistributionTests(unittest.TestCase):
    def test_group_shares_add_up_to_the_total_residual(self):
        keys = [(0, index, (0, 0, 0)) for index in range(6)]
        reference = {key: float(index + 1) for index, key in enumerate(keys)}
        candidate = {key: float(index + 1) + 0.5 for index, key in enumerate(keys)}
        total = float(np.linalg.norm(np.full(len(keys), 0.5)))
        rows = qc.group_metrics(
            keys,
            reference,
            candidate,
            grouping="orbital_block",
            labels=["s-s", "s-s", "s-p", "s-p", "p-p", "p-p"],
            direction_name="v",
            total_residual=total,
        )
        self.assertEqual({row["group"] for row in rows}, {"s-s", "s-p", "p-p"})
        self.assertAlmostEqual(sum(row["share_of_total_residual"] for row in rows), 1.0)

    def test_a_key_missing_from_the_candidate_counts_as_zero_not_as_absent(self):
        keys = [(0, 0, (0, 0, 0)), (0, 1, (0, 0, 0))]
        reference = {keys[0]: 1.0, keys[1]: 4.0}
        rows = qc.group_metrics(
            keys,
            reference,
            {keys[0]: 1.0},  # the model has no entry for the second key
            grouping="orbital_block",
            labels=["s-s", "s-s"],
            direction_name="v",
            total_residual=4.0,
        )
        self.assertAlmostEqual(rows[0]["residual_frobenius"], 4.0)

    def test_magnitude_deciles_span_ten_buckets(self):
        labels = qc.magnitude_deciles(np.linspace(-1.0, 1.0, 100))
        self.assertEqual(len(set(labels)), 10)
        self.assertEqual(len(labels), 100)


def _row(name, kind, reference, residual, over_explained=100.0):
    return {
        "direction_name": name,
        "direction_kind": kind,
        "reference": {"frobenius": reference, "max_abs": reference},
        "relative_frobenius": residual / reference if reference else float("nan"),
        "residual": {"frobenius": residual, "max_abs": residual},
        "error_budget": {"residual_over_explained": over_explained},
    }


class VerdictTests(unittest.TestCase):
    def test_a_small_error_on_every_direction_is_adequate(self):
        verdict = qc.decide(
            [
                _row("atom0000_x", "one_hot", 40.0, 0.4),
                _row("random_seed0", "collective", 20.0, 0.3),
                _row("translation_x", "uniform_translation", 0.004, 0.004),
            ],
            calibration=("atom0000_x", "translation_x"),
        )
        self.assertEqual(verdict["decision"], "UNDECIDED")
        self.assertEqual(verdict["null_directions"], ["translation_x"])

    def test_a_large_error_stays_undecided_until_reference_is_closed(self):
        verdict = qc.decide(
            [
                _row("atom0000_x", "one_hot", 40.0, 26.0),
                _row("random_seed0", "collective", 20.0, 11.0),
            ],
            calibration=("atom0000_x",),
        )
        self.assertEqual(verdict["decision"], "UNDECIDED")
        self.assertGreater(verdict["worst_relative_frobenius"], qc.ADEQUACY_RELATIVE_FROBENIUS)

    def test_the_null_direction_never_sets_the_worst_relative_error(self):
        """Its reference is zero, so ``residual / reference`` is meaningless."""
        verdict = qc.decide(
            [
                _row("atom0000_x", "one_hot", 40.0, 0.4),
                _row("translation_x", "uniform_translation", 0.004, 0.004),
            ],
            calibration=("atom0000_x", "translation_x"),
        )
        self.assertAlmostEqual(verdict["worst_relative_frobenius"], 0.01)
        self.assertEqual(verdict["decision"], "UNDECIDED")

    def test_a_held_out_direction_far_above_the_calibrated_bound_is_flagged(self):
        verdict = qc.decide(
            [
                _row("atom0000_x", "one_hot", 40.0, 4.0),  # 10%
                _row("random_seed0", "collective", 20.0, 12.0),  # 60%
            ],
            calibration=("atom0000_x",),
        )
        self.assertAlmostEqual(verdict["calibrated_relative_bound"], 0.1)
        self.assertFalse(verdict["held_out_within_calibrated_bound"])

    def test_calibration_bound_ignores_the_held_out_direction(self):
        """Leakage guard: the tolerance may not be fitted on the case it judges."""
        verdict = qc.decide(
            [
                _row("atom0000_x", "one_hot", 40.0, 4.0),
                _row("random_seed0", "collective", 20.0, 12.0),
            ],
            calibration=("atom0000_x",),
        )
        self.assertAlmostEqual(verdict["calibrated_relative_bound"], 0.1)

    def test_a_residual_inside_the_numerical_floors_is_not_attributed_to_the_model(self):
        verdict = qc.decide(
            [_row("atom0000_x", "one_hot", 40.0, 0.4, over_explained=0.5)],
            calibration=("atom0000_x",),
        )
        self.assertFalse(verdict["residual_exceeds_every_numerical_floor"])


class PreconditionTests(unittest.TestCase):
    def test_a_failed_precondition_downgrades_the_decision_to_not_evaluable(self):
        verdict = qc.finalize_verdict(
            {"decision": "eligible"},
            [
                {"check": "go2_certified", "passed": True},
                {"check": "go3_internal_validation_passed", "passed": False},
            ],
        )
        self.assertEqual(verdict["decision"], "not_evaluable")
        self.assertIn("go3_internal_validation_passed", verdict["failed_preconditions"])

    def test_all_preconditions_passing_leaves_the_decision_alone(self):
        verdict = qc.finalize_verdict(
            {"decision": "eligible"},
            [{"check": "go2_certified", "passed": True}],
        )
        self.assertEqual(verdict["decision"], "eligible")
        self.assertNotIn("failed_preconditions", verdict)


class DirectionTests(unittest.TestCase):
    def test_rebuilt_direction_reproduces_the_campaign_hash(self):
        original = fdp.one_hot(2, 0, "x")
        entry = original.to_dict()
        rebuilt = qc.direction_from_manifest(entry)
        self.assertEqual(rebuilt.direction_hash, entry["direction_hash"])
        self.assertEqual(rebuilt.kind, fdp.KIND_ONE_HOT)

    def test_rebuilding_a_one_hot_as_collective_would_hash_differently(self):
        """Why the kind is carried: the hash covers it."""
        original = fdp.one_hot(2, 0, "x")
        as_collective = fdp.collective(original.vectors, name=original.name)
        self.assertNotEqual(original.direction_hash, as_collective.direction_hash)


class GhostFreeFdfTests(unittest.TestCase):
    def test_unpopulated_species_are_dropped_and_the_geometry_survives(self, ):
        sisl = pytest.importorskip("sisl")
        source = REPO_ROOT / "materials/graphene/RUN.fdf"
        if not source.is_file():
            self.skipTest(f"{source} is absent")
        geometry = sisl.get_sile(str(source)).read_geometry()
        destination = Path(
            __import__("tempfile").mkdtemp(prefix="ghost_free_")
        ) / "RUN.fdf"
        qc.ghost_free_fdf(geometry, destination)
        text = destination.read_text(encoding="utf-8")
        self.assertNotIn("Ghost-H", text)
        rebuilt = sisl.get_sile(str(destination)).read_geometry()
        np.testing.assert_allclose(rebuilt.xyz, geometry.xyz, atol=1e-9)
        np.testing.assert_allclose(rebuilt.cell, geometry.cell, atol=1e-9)


@pytest.mark.slow
def test_c14_on_the_real_certified_artifacts(tmp_path):
    """End to end on the graphene campaign, when GO-2 and the checkpoint are present."""
    if not qc.DEFAULT_CERTIFICATION.is_file():
        pytest.skip(f"no GO-2 certification at {qc.DEFAULT_CERTIFICATION}")
    if not qc.DEFAULT_CHECKPOINT.is_file():
        pytest.skip(f"no checkpoint at {qc.DEFAULT_CHECKPOINT}")
    pytest.importorskip("graph2mat")

    report = qc.quantify(output_dir=tmp_path, run_internal_validation=False)
    failed = [row["check"] for row in report["checks"] if not row["passed"]]
    assert not failed, failed
    assert report["verdict"]["decision"] == "UNDECIDED"
    assert report["verdict"]["decision_blockers"]
    # Every direction the campaign certified is compared, and each carries its
    # full budget: a bare residual with no floors is not a C14 result.
    assert {row["direction_name"] for row in report["directions"]} == {
        "atom0000_x",
        "random_seed0",
        "translation_x",
    }
    for row in report["directions"]:
        assert set(row["error_budget"]) >= {
            "tau_num_reference_go2",
            "tau_num_model_frozen_plateau",
            "tau_backend_jvp_vs_frozen",
            "support_truncation_outside_model_graph",
        }
    assert json.loads((tmp_path / "checkpoint_derivative_error.json").read_text())["schema"] == qc.SCHEMA


if __name__ == "__main__":
    unittest.main()
