"""C12 + C13 / E-F_001-S14: the GO-3 internal validation of the directional JVP.

Analytic stand-in models only: a translation-invariant pair model (the honest
one), the same model with a deliberately scaled backward (a wrong JVP that is
still *small*), and a stiff one whose central difference never plateaus. What is
under test is the validator's verdict, not Graph2Mat.
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "Comparison" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import fd_perturbation_space as fdp  # noqa: E402
import validate_graph2mat_jvp as vj  # noqa: E402


DELTAS = fdp.delta_sweep(0.004, count=5)


class Batch(dict):
    """Minimal torch_geometric-like batch (dict + clone + to)."""

    def clone(self) -> "Batch":
        return Batch(self)

    def to(self, device) -> "Batch":
        moved = Batch(self)
        for key, value in list(moved.items()):
            if isinstance(value, torch.Tensor):
                moved[key] = value.to(device)
        return moved


class PairModel(torch.nn.Module):
    """Smooth, translation-invariant function of the interatomic vectors."""

    def forward(self, data):
        positions = data["positions"]
        delta = positions[:, None, :] - positions[None, :, :]
        return {
            "node_labels": torch.exp(-delta.pow(2).sum(-1)).sum(1),
            "edge_labels": torch.sin(delta).reshape(-1),
        }


class _ScaledBackward(torch.autograd.Function):
    """Identity forward, gradient multiplied by ``scale``: a wrong-but-small JVP."""

    @staticmethod
    def forward(ctx, tensor, scale):
        ctx.scale = float(scale)
        return tensor.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None


class WrongGradientPairModel(PairModel):
    """Correct labels, derivative off by 10%: only a finite difference can see it."""

    def forward(self, data):
        data = dict(data)
        data["positions"] = _ScaledBackward.apply(data["positions"], 1.1)
        return super().forward(data)


class StiffModel(torch.nn.Module):
    """Oscillates on the scale of the swept amplitudes: no central-difference plateau."""

    def forward(self, data):
        positions = data["positions"]
        delta = positions[:, None, :] - positions[None, :, :]
        return {
            "node_labels": torch.sin(900.0 * delta.pow(2).sum(-1)).sum(1),
            "edge_labels": torch.sin(900.0 * delta).reshape(-1),
        }


def make_batch(dtype: torch.dtype = torch.float64) -> Batch:
    return Batch(
        positions=torch.tensor(
            [[0.0, 0.0, 0.0], [1.4, 0.2, 0.0], [0.7, 1.2, 0.1]], dtype=dtype
        )
    )


def checks_named(report: dict, name: str) -> list[dict]:
    return [row for row in report["checks"] if row["check"] == name]


def run(model=None, **kwargs) -> dict:
    kwargs.setdefault("cutoff_ang", 6.0)
    kwargs.setdefault("deltas", DELTAS)
    return vj.validate_directional_jvp(model or PairModel(), make_batch(), **kwargs)


class PathAgreementTests(unittest.TestCase):
    """D_jvp[v] vs the coordinate combination and vs the frozen central difference."""

    def test_go3_passes_on_a_correct_model_in_float64_and_float32(self) -> None:
        report = run(dtypes=("float64", "float32"))
        self.assertTrue(report["summary"]["passed"], report["summary"]["checks_failed"])
        # Errors are reported per dtype, never merged into one number.
        by_dtype = {
            row["dtype"]: row["relative_frobenius"]
            for row in checks_named(report, "coordinate_combination")
            if row["direction_name"] == "random_seed0"
        }
        self.assertEqual(set(by_dtype), {"float64", "float32"})
        self.assertLess(by_dtype["float64"], vj.ROUNDOFF_TOLERANCE["float64"])
        self.assertLess(by_dtype["float32"], vj.ROUNDOFF_TOLERANCE["float32"])
        self.assertGreater(by_dtype["float32"], by_dtype["float64"])

    def test_coordinate_combination_is_exact_under_a_rotated_frame(self) -> None:
        # p_batch = C @ p_cart with a non-identity C: both paths must apply the
        # same convention or the contraction stops matching.
        angle = 0.7
        cob = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        model, batch = PairModel(), make_batch()
        direction = fdp.random_collective(3, seed=4)
        jvp, _ = vj.directional_jvp(model, batch, direction, change_of_basis=cob)
        combination, _ = vj.coordinate_combination(model, batch, direction, change_of_basis=cob)
        measured = vj.discrepancy(combination, jvp)
        self.assertLess(measured["relative_frobenius"], vj.ROUNDOFF_TOLERANCE["float64"])
        # And the rotated frame really is a different derivative from the raw one.
        raw, _ = vj.directional_jvp(model, batch, direction)
        self.assertGreater(vj.discrepancy(raw, jvp)["relative_frobenius"], 1e-3)

    def test_linearity_holds_for_a_combination_of_two_directions(self) -> None:
        report = run()
        (row,) = checks_named(report, "linearity")
        self.assertTrue(row["passed"], row["detail"])
        self.assertLess(row["relative_frobenius"], vj.ROUNDOFF_TOLERANCE["float64"])

    def test_uniform_translation_annihilates_both_routes(self) -> None:
        report = run()
        (row,) = checks_named(report, "uniform_translation_invariance")
        self.assertTrue(row["passed"], row["detail"])
        self.assertEqual(row["status"], "evaluated")
        self.assertLess(row["relative_to_signal"], vj.TRANSLATION_TOLERANCE["float64"])
        # The coordinate combination of a null direction is judged against the
        # magnitude that cancelled, not against its own zero.
        (coordinate,) = [
            row
            for row in checks_named(report, "coordinate_combination")
            if row["direction_name"] == "translation_x"
        ]
        self.assertTrue(coordinate["is_null_direction"])
        self.assertEqual(coordinate["relative_to"], "cancellation_scale")
        self.assertTrue(coordinate["passed"])

    def test_frozen_central_difference_converges_to_the_jvp(self) -> None:
        report = run()
        rows = [
            row for row in checks_named(report, "jvp_vs_frozen") if row["passed"]
        ]
        self.assertEqual(len(rows), 3)  # every non-translation direction
        for row in rows:
            self.assertIn(row["verdict"], ("noise_limited", "truncation_limited"))
            self.assertTrue(row["has_plateau"])
            self.assertFalse(row["smoke_tolerance_only"])
            # Far below the legacy smoke tolerance, and explained, not just small.
            self.assertLess(row["best_relative_frobenius"], 1e-3)


class RefusalTests(unittest.TestCase):
    """What GO-3 must refuse."""

    def test_wrong_gradient_inside_the_smoke_tolerance_fails(self) -> None:
        # A 10% wrong JVP passes the legacy `rel_frobenius <= 0.25` smoke test
        # and must still fail GO-3.
        report = run(WrongGradientPairModel())
        rows = checks_named(report, "jvp_vs_frozen")
        self.assertTrue(rows)
        for row in rows:
            self.assertFalse(row["passed"], row["detail"])
            self.assertEqual(row["verdict"], "unresolved")
            self.assertLessEqual(row["best_relative_frobenius"], vj.SMOKE_TOLERANCE)
            self.assertTrue(row["smoke_tolerance_only"])
        self.assertFalse(report["summary"]["passed"])
        self.assertIn("jvp_vs_frozen", report["summary"]["checks_failed"])

    def test_amplitude_that_crosses_the_neighbour_cutoff_fails(self) -> None:
        # 1.4 Ang pair with a 1.5 Ang cutoff: a 0.128 Ang one-hot displacement
        # takes it across, so the derivative is taken across a graph discontinuity.
        report = run(cutoff_ang=1.5, deltas=fdp.delta_sweep(0.032, count=5))
        rows = checks_named(report, "topology_fixed")
        self.assertTrue(any(not row["passed"] for row in rows))
        self.assertIn("topology_fixed", report["summary"]["checks_failed"])
        self.assertFalse(report["summary"]["passed"])

    def test_missing_cutoff_cannot_certify_topology(self) -> None:
        report = run(cutoff_ang=None)
        for row in checks_named(report, "topology_fixed"):
            self.assertFalse(row["passed"])
            self.assertEqual(row["status"], "not_evaluable")
        self.assertFalse(report["summary"]["passed"])

    def test_no_frozen_plateau_fails_instead_of_picking_an_amplitude(self) -> None:
        report = run(StiffModel())
        rows = [row for row in checks_named(report, "jvp_vs_frozen")]
        self.assertTrue(rows)
        self.assertTrue(any(not row["passed"] for row in rows))
        self.assertTrue(any(row.get("has_plateau") is False for row in rows))

    def test_a_single_amplitude_cannot_produce_a_plateau(self) -> None:
        report = run(deltas=(0.01,))
        (row,) = checks_named(report, "delta_sweep_plateau_capable")
        self.assertFalse(row["passed"])
        self.assertEqual(row["delta_sweep_status"], "insufficient_for_plateau")
        # And the comparison itself refuses rather than picking that amplitude.
        for frozen in checks_named(report, "jvp_vs_frozen"):
            self.assertFalse(frozen["passed"])
        self.assertFalse(report["summary"]["passed"])

    def test_coverage_refuses_a_one_sided_direction_set(self) -> None:
        report = run(directions=[fdp.random_collective(3, seed=0), fdp.random_collective(3, seed=1)])
        (row,) = checks_named(report, "coverage")
        self.assertFalse(row["passed"])
        self.assertEqual(row["translation_directions"], 0)

    def test_unknown_dtype_is_refused(self) -> None:
        with self.assertRaises(vj.JvpValidationError):
            run(dtypes=("float16",))  # no pre-registered tolerance

    def test_mismatched_direction_length_is_refused(self) -> None:
        from graph2mat_autograd_derivatives import Graph2MatAutogradDerivativeError

        with self.assertRaises(Graph2MatAutogradDerivativeError):
            vj.directional_jvp(PairModel(), make_batch(), np.ones((5, 3)))


class BackendTests(unittest.TestCase):
    """CPU/CUDA: compared only when the preflight ran, documented otherwise."""

    def test_cuda_fallback_is_documented_and_not_silent(self) -> None:
        available = torch.cuda.is_available
        torch.cuda.is_available = lambda: False
        try:
            report = run(backends=("cuda",))
        finally:
            torch.cuda.is_available = available
        (row,) = checks_named(report, "backend_preflight_documented")
        self.assertTrue(row["passed"])
        self.assertEqual(row["requested_backend"], "cuda")
        self.assertEqual(row["effective_backend"], "cpu")
        self.assertEqual(row["backend_preflight"], "failed")
        self.assertIn("cuda_unavailable", row["backend_fallback_reason"])
        # No equivalence is claimed when CUDA never ran.
        self.assertEqual(checks_named(report, "backend_equivalence"), [])
        self.assertTrue(report["summary"]["passed"], report["summary"]["checks_failed"])

    @unittest.skipUnless(torch.cuda.is_available(), "no CUDA device")
    def test_cuda_and_cpu_agree_when_the_preflight_passes(self) -> None:
        report = run(backends=("cpu", "cuda"), dtypes=("float64",))
        rows = checks_named(report, "backend_equivalence")
        self.assertTrue(rows)
        for row in rows:
            self.assertEqual(row["status"], "compared")
            self.assertTrue(row["passed"], row["detail"])
        self.assertTrue(report["summary"]["passed"], report["summary"]["checks_failed"])


class HermiticityTests(unittest.TestCase):
    """Blockwise hermiticity of the serialised D_H[v], through the real converter."""

    class FakeProcessor:
        """Serialises whatever labels it is given into a fixed supercell layout."""

        sub_point_matrix = False
        default_out_format = "scipy_csr"

        def __init__(self, hermitian: bool = True) -> None:
            self.hermitian = hermitian

        class _SislLike:
            """Just enough of a sisl matrix: ``tocsr(dim)`` and an R-vector order."""

            class geometry:
                class lattice:
                    sc_off = [(0, 0, 0), (1, 0, 0), (-1, 0, 0)]

            def __init__(self, matrix):
                self._matrix = matrix

            def tocsr(self, dim=0):
                return self._matrix

        def yield_from_batch(self, batch, predictions=None, as_matrix=False):
            self.predictions = predictions
            yield "example"

        def labels_to(self, out_format, data=None, threshold=None):
            from scipy import sparse

            rng = np.random.default_rng(3)
            onsite = rng.normal(size=(2, 2))
            onsite = onsite + onsite.T
            plus = rng.normal(size=(2, 2))
            minus = plus.T if self.hermitian else rng.normal(size=(2, 2))
            return self._SislLike(sparse.csr_matrix(np.hstack([onsite, plus, minus])))

    def _hermiticity_rows(self, hermitian: bool) -> list[dict]:
        report = run(data_processor=self.FakeProcessor(hermitian))
        return checks_named(report, "blockwise_hermiticity")

    def test_hermitian_derivative_passes(self) -> None:
        rows = self._hermiticity_rows(True)
        self.assertTrue(rows)
        for row in rows:
            self.assertTrue(row["passed"], row["detail"])
            self.assertLess(row["hermiticity_defect"], vj.HERMITICITY_TOLERANCE["float64"])

    def test_broken_transpose_partner_fails(self) -> None:
        rows = self._hermiticity_rows(False)
        self.assertTrue(rows)
        self.assertTrue(all(not row["passed"] for row in rows))


class ReportTests(unittest.TestCase):
    def test_report_carries_the_gate_tolerances_and_provenance(self) -> None:
        report = run()
        self.assertEqual(report["schema"], vj.SCHEMA)
        self.assertEqual(report["gate"], "GO-3")
        self.assertEqual(report["tolerances"]["refused_smoke_tolerance"], 0.25)
        self.assertEqual(report["delta_ang_values"], sorted(DELTAS))
        (run_entry,) = report["runs"]
        for direction in run_entry["directions"]:
            metadata = direction["jvp_metadata"]
            self.assertEqual(metadata["jvp_calls"], 1)
            self.assertFalse(metadata["materialized_jacobian"])
            self.assertEqual(direction["direction_hash"], metadata["direction_hash"])

    def test_every_evaluation_ran_on_the_same_graph(self) -> None:
        report = run()
        for row in checks_named(report, "batch_topology_unchanged"):
            self.assertTrue(row["passed"], row["detail"])
            self.assertEqual(len(row["topology_hashes"]), 1)


# --------------------------------------------------------------------------- #
# The same gate on the real thing. Skips cleanly when no checkpoint is around;
# this is the run that replaces the legacy `rel_frobenius <= 0.25` smoke test.
# --------------------------------------------------------------------------- #
_SMOKE_SWEEP_ROOT = (
    REPO_ROOT / "Comparison" / "results" / "e2e_smoke_12snap_20ep"
    / "e2e_smoke_12snap_20ep" / "sweep"
)
_DEFAULT_CKPT = (
    _SMOKE_SWEEP_ROOT / "graph2mat" / "graphene_w90_scale_iid12" / "G2M-E2E12-20EP"
    / "graph2mat" / "training" / "lightning_logs" / "my_first_model" / "version_0"
    / "checkpoints" / "best-160.ckpt"
)
_DEFAULT_STRUCTURE = (
    _SMOKE_SWEEP_ROOT / "derivative_workflows" / "graphene_w90_scale_iid12"
    / "structures" / "md_11_base"
)


def _load_real_model_and_batch(tmp_path):
    """Checkpoint + single-structure batch, or ``pytest.skip``."""
    checkpoint = Path(os.environ.get("G2M_AUTOGRAD_SMOKE_CKPT") or _DEFAULT_CKPT)
    structure_dir = Path(os.environ.get("G2M_AUTOGRAD_SMOKE_STRUCTURE") or _DEFAULT_STRUCTURE)
    if not checkpoint.is_file():
        pytest.skip(f"No Graph2Mat checkpoint available: {checkpoint}")
    if not (structure_dir / "RUN.fdf").is_file():
        pytest.skip(f"No base structure RUN.fdf available: {structure_dir}")
    basis_glob = os.environ.get("G2M_AUTOGRAD_SMOKE_BASIS") or str(structure_dir / "*.ion.xml")
    try:
        compat_dir = REPO_ROOT / "scripts" / "torch_serialization_compat"
        if str(compat_dir) not in sys.path:
            sys.path.insert(0, str(compat_dir))
        from torch_safe_globals import allow_graph2mat_checkpoint_globals

        # Importing graph2mat/mace already torch.load()s data files, so the safe
        # globals must be registered before the imports.
        allow_graph2mat_checkpoint_globals()

        from graph2mat.tools.lightning import MatrixDataModule
        from graph2mat.tools.lightning.models.mace import LitMACEMatrixModel

        from predict_model_on_dataset import checkpoint_training_dir, normalize_pattern_for_workdir
    except ImportError as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"graph2mat stack not importable: {exc}")

    run_cwd = checkpoint_training_dir(checkpoint)
    source_cwd = Path.cwd()
    runs_json = tmp_path / "go3_runs.json"
    runs_json.write_text(json.dumps({"predict": [str(structure_dir / "RUN.fdf")]}), encoding="utf-8")
    basis_files = normalize_pattern_for_workdir(
        basis_glob, source_cwd=source_cwd, target_cwd=run_cwd
    )
    try:
        os.chdir(run_cwd)
        model = LitMACEMatrixModel.load_from_checkpoint(
            str(checkpoint), map_location="cpu", weights_only=False
        )
        model.eval()
        datamodule = MatrixDataModule(
            out_matrix="hamiltonian",
            symmetric_matrix=True,
            sub_point_matrix=False,
            basis_files=basis_files,
            runs_json=str(runs_json),
            store_in_memory=True,
            batch_size=1,
            n_matrix_components=1,
            matrix_component_policy="h_only",
        )
        datamodule.setup("predict")
        batch = next(iter(datamodule.predict_dataloader()))
    finally:
        os.chdir(source_cwd)
    return model.model, batch, datamodule.data_processor


def _pao_pair_cutoffs(batch, data_processor) -> list[float]:
    """Per-atom PAO radius: the neighbour rule of the graph is ``d_ij <= R_i + R_j``."""
    basis_table = data_processor.basis_table
    radii = [float(np.max(entry.R)) for entry in basis_table.basis]
    return [radii[int(point_type)] for point_type in batch["point_types"]]


@pytest.mark.slow
def test_go3_on_a_real_graph2mat_checkpoint(tmp_path):
    """JVP vs coordinate combination vs frozen Graph2Mat, on the real model."""
    model, batch, data_processor = _load_real_model_and_batch(tmp_path)
    cob = np.asarray(data_processor.basis_table.change_of_basis, dtype=np.float64)
    # Positions and cell live in the e3nn frame (p_batch = C p_cart); the
    # topology gate works in the physical cartesian frame of the fdf structure.
    cell = np.asarray(batch["cell"], dtype=np.float64).reshape(3, 3) @ np.linalg.inv(cob).T

    report = vj.validate_directional_jvp(
        model,
        batch,
        change_of_basis=cob,
        cutoff_ang=_pao_pair_cutoffs(batch, data_processor),
        lattice_vectors_ang=cell,
        deltas=fdp.delta_sweep(0.004, count=5),
        dtypes=("float32",),
        # CUDA is requested, not assumed: resolve_jvp_backend runs the real
        # double backward on this model first and falls back to CPU with a
        # recorded reason if it cannot. (The documented MACE/TorchScript CUDA
        # failure is of the *vectorized batched* backward, which this path does
        # not use.)
        backends=("cpu", "cuda"),
        # No data_processor: this smoke checkpoint carries its own (smaller)
        # basis table, so its labels do not slice into this structure's sparse
        # layout. Blockwise hermiticity of the serialised D_H[v] is covered by
        # HermiticityTests; point the env overrides at a basis-consistent
        # checkpoint to exercise it here too.
    )
    for row in report["checks"]:
        print(
            f"[GO-3] {row['passed']!s:>5} {row['check']:<28} "
            f"{str(row.get('direction_name') or ''):<14} {row['detail'][:120]}"
        )
    assert report["summary"]["passed"], report["summary"]["checks_failed"]


if __name__ == "__main__":
    unittest.main()
