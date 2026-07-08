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
TORCH_COMPAT_DIR = REPO_ROOT / "scripts" / "torch_serialization_compat"
if str(TORCH_COMPAT_DIR) not in sys.path:
    sys.path.insert(0, str(TORCH_COMPAT_DIR))

from graph2mat_autograd_derivatives import (  # noqa: E402
    Graph2MatAutogradDerivativeError,
    compute_graph2mat_position_jacobian,
    flatten_graph2mat_predictions,
    graph2mat_forward_labels,
    require_single_structure_batch,
    select_derivative_prediction_from_jacobian,
    unflatten_graph2mat_prediction_vector,
)
from run_graph2mat_autograd_derivative_predictions import (  # noqa: E402
    _append_missing_structure_rows,
)


N_ATOMS = 4
N_NODE_OUTPUTS = 6
N_EDGE_OUTPUTS = 10


class FakeBatch(dict):
    """Minimal stand-in for a torch_geometric batch (dict + clone)."""

    def clone(self) -> "FakeBatch":
        return FakeBatch(self)


def make_batch(n_atoms: int = N_ATOMS, num_graphs: int | None = None) -> FakeBatch:
    generator = torch.Generator().manual_seed(7)
    batch = FakeBatch(
        positions=torch.randn(n_atoms, 3, generator=generator, dtype=torch.float64)
    )
    if num_graphs is not None:
        batch["ptr"] = torch.arange(num_graphs + 1) * n_atoms
    return batch


class LinearFakeModel(torch.nn.Module):
    """node_labels = A @ p_flat, edge_labels = B @ p_flat (analytic jacobian)."""

    def __init__(self) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(11)
        self.A = torch.randn(
            N_NODE_OUTPUTS, N_ATOMS * 3, generator=generator, dtype=torch.float64
        )
        self.B = torch.randn(
            N_EDGE_OUTPUTS, N_ATOMS * 3, generator=generator, dtype=torch.float64
        )

    def forward(self, data) -> dict[str, torch.Tensor]:
        flat = data["positions"].reshape(-1)
        return {"node_labels": self.A @ flat, "edge_labels": self.B @ flat}

    def analytic_jacobian(self) -> torch.Tensor:
        return torch.cat([self.A, self.B]).reshape(-1, N_ATOMS, 3)


class NonlinearFakeModel(torch.nn.Module):
    """node_labels = sum_axis(p^2) per atom, edge_labels = sin(p).flatten()."""

    def forward(self, data) -> dict[str, torch.Tensor]:
        positions = data["positions"]
        return {
            "node_labels": positions.pow(2).sum(dim=1),
            "edge_labels": torch.sin(positions).reshape(-1),
        }

    @staticmethod
    def analytic_jacobian(positions: torch.Tensor) -> torch.Tensor:
        n_atoms = positions.shape[0]
        n_outputs = n_atoms + n_atoms * 3
        jacobian = torch.zeros(n_outputs, n_atoms, 3, dtype=positions.dtype)
        for atom in range(n_atoms):
            jacobian[atom, atom, :] = 2.0 * positions[atom]
        for atom in range(n_atoms):
            for axis in range(3):
                out_index = n_atoms + atom * 3 + axis
                jacobian[out_index, atom, axis] = torch.cos(positions[atom, axis])
        return jacobian


class DetachedFakeModel(torch.nn.Module):
    def forward(self, data) -> dict[str, torch.Tensor]:
        positions = data["positions"].detach()
        return {
            "node_labels": positions.sum(dim=1),
            "edge_labels": positions.reshape(-1),
        }


class MultiComponentFakeModel(torch.nn.Module):
    """2D labels (n_matrix_components > 1) to exercise flatten/unflatten shapes."""

    def forward(self, data) -> dict[str, torch.Tensor]:
        positions = data["positions"]
        return {
            "node_labels": torch.stack([positions.sum(dim=1), positions.prod(dim=1)], dim=1),
            "edge_labels": positions.reshape(-1, 3),
        }


class FlattenSpecTests(unittest.TestCase):
    def test_flatten_and_unflatten_roundtrip(self) -> None:
        batch = make_batch()
        model = MultiComponentFakeModel()
        out = model(batch)
        flat, spec = flatten_graph2mat_predictions(out)

        self.assertEqual(flat.numel(), out["node_labels"].numel() + out["edge_labels"].numel())
        self.assertEqual(spec.n_outputs, flat.numel())
        rebuilt = unflatten_graph2mat_prediction_vector(flat, spec)
        self.assertEqual(tuple(rebuilt["node_labels"].shape), tuple(out["node_labels"].shape))
        self.assertEqual(tuple(rebuilt["edge_labels"].shape), tuple(out["edge_labels"].shape))
        torch.testing.assert_close(rebuilt["node_labels"], out["node_labels"])
        torch.testing.assert_close(rebuilt["edge_labels"], out["edge_labels"])

    def test_unflatten_rejects_wrong_size(self) -> None:
        batch = make_batch()
        out = LinearFakeModel()(batch)
        _, spec = flatten_graph2mat_predictions(out)
        with self.assertRaises(Graph2MatAutogradDerivativeError):
            unflatten_graph2mat_prediction_vector(torch.zeros(spec.n_outputs + 1), spec)

    def test_missing_output_key_is_rejected(self) -> None:
        with self.assertRaises(Graph2MatAutogradDerivativeError):
            flatten_graph2mat_predictions({"node_labels": torch.zeros(3)})


class ForwardLabelsTests(unittest.TestCase):
    def test_forward_keeps_gradient_connection(self) -> None:
        batch = make_batch()
        model = LinearFakeModel()
        positions = batch["positions"].detach().clone().requires_grad_(True)
        flat, spec = graph2mat_forward_labels(model, batch, positions)

        self.assertTrue(flat.requires_grad)
        self.assertEqual(spec.n_outputs, N_NODE_OUTPUTS + N_EDGE_OUTPUTS)
        # The original batch positions must not be mutated by the closure.
        self.assertFalse(batch["positions"].requires_grad)

    def test_single_structure_guard(self) -> None:
        batch = make_batch(num_graphs=2)
        with self.assertRaises(Graph2MatAutogradDerivativeError):
            require_single_structure_batch(batch)
        with self.assertRaises(Graph2MatAutogradDerivativeError):
            compute_graph2mat_position_jacobian(LinearFakeModel(), batch)


class PositionJacobianTests(unittest.TestCase):
    def test_linear_model_matches_analytic_jacobian(self) -> None:
        batch = make_batch()
        model = LinearFakeModel()
        result = compute_graph2mat_position_jacobian(
            model, batch, method="vmap_vjp_chunked", chunk_size=5
        )

        expected = model.analytic_jacobian()
        self.assertEqual(
            tuple(result.jacobian.shape), (N_NODE_OUTPUTS + N_EDGE_OUTPUTS, N_ATOMS, 3)
        )
        torch.testing.assert_close(result.jacobian, expected, rtol=1e-9, atol=1e-12)
        self.assertEqual(result.method, "vmap_vjp_chunked")
        self.assertEqual(result.chunk_size, 5)
        self.assertGreater(result.jacobian.abs().max().item(), 0.0)

    def test_nonlinear_model_matches_analytic_jacobian(self) -> None:
        batch = make_batch()
        model = NonlinearFakeModel()
        result = compute_graph2mat_position_jacobian(model, batch, method="auto")

        expected = model.analytic_jacobian(batch["positions"])
        torch.testing.assert_close(result.jacobian, expected, rtol=1e-9, atol=1e-12)
        torch.testing.assert_close(
            result.base_predictions["node_labels"],
            batch["positions"].pow(2).sum(dim=1),
        )

    def test_methods_agree_on_small_case(self) -> None:
        batch = make_batch()
        model = NonlinearFakeModel()
        reference = compute_graph2mat_position_jacobian(
            model, batch, method="vmap_vjp_chunked", chunk_size=3
        ).jacobian

        for method in ("jacrev", "jacfwd", "autograd_jacobian"):
            with self.subTest(method=method):
                jacobian = compute_graph2mat_position_jacobian(
                    model, batch, method=method
                ).jacobian
                torch.testing.assert_close(jacobian, reference, rtol=1e-8, atol=1e-10)

    def test_detached_model_fails_loudly(self) -> None:
        batch = make_batch()
        with self.assertRaises(Graph2MatAutogradDerivativeError):
            compute_graph2mat_position_jacobian(DetachedFakeModel(), batch)

    def test_unknown_method_is_rejected(self) -> None:
        with self.assertRaises(Graph2MatAutogradDerivativeError):
            compute_graph2mat_position_jacobian(
                LinearFakeModel(), make_batch(), method="finite_difference"
            )


class SelectDerivativeTests(unittest.TestCase):
    def test_selected_column_shapes_and_values(self) -> None:
        batch = make_batch()
        model = LinearFakeModel()
        result = compute_graph2mat_position_jacobian(model, batch)

        atom_index, axis_index = 2, 1
        derivative = select_derivative_prediction_from_jacobian(
            result.jacobian, result.spec, atom_index, axis_index
        )

        self.assertEqual(tuple(derivative["node_labels"].shape), (N_NODE_OUTPUTS,))
        self.assertEqual(tuple(derivative["edge_labels"].shape), (N_EDGE_OUTPUTS,))
        flat_index = atom_index * 3 + axis_index
        torch.testing.assert_close(derivative["node_labels"], model.A[:, flat_index])
        torch.testing.assert_close(derivative["edge_labels"], model.B[:, flat_index])

    def test_selected_column_matches_numeric_finite_difference(self) -> None:
        # Numeric FD is only a sanity check of the test itself, not the
        # scientific route.
        batch = make_batch()
        model = NonlinearFakeModel()
        result = compute_graph2mat_position_jacobian(model, batch)

        atom_index, axis_index = 1, 2
        delta = 1e-6
        plus = batch.clone()
        minus = batch.clone()
        plus["positions"] = batch["positions"].clone()
        minus["positions"] = batch["positions"].clone()
        plus["positions"][atom_index, axis_index] += delta
        minus["positions"][atom_index, axis_index] -= delta
        numeric = {
            key: (model(plus)[key] - model(minus)[key]) / (2 * delta)
            for key in ("node_labels", "edge_labels")
        }

        derivative = select_derivative_prediction_from_jacobian(
            result.jacobian, result.spec, atom_index, axis_index
        )
        torch.testing.assert_close(
            derivative["node_labels"], numeric["node_labels"], rtol=1e-6, atol=1e-8
        )
        torch.testing.assert_close(
            derivative["edge_labels"], numeric["edge_labels"], rtol=1e-6, atol=1e-8
        )

    def test_change_of_basis_contraction_gives_cartesian_directional_derivative(self) -> None:
        # Graph2Mat stores batch positions as p_batch = C @ p_cart (e3nn change
        # of basis). The cartesian derivative along axis a must therefore be
        # J[:, atom, :] @ C[:, a], not the raw batch-frame column.
        batch = make_batch()
        model = NonlinearFakeModel()
        result = compute_graph2mat_position_jacobian(model, batch)

        # e3nn cartesian -> spherical-harmonics convention: (x, y, z) -> (y, z, x).
        cob = torch.tensor(
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=torch.float64
        )
        atom_index, axis_index = 1, 0
        derivative = select_derivative_prediction_from_jacobian(
            result.jacobian, result.spec, atom_index, axis_index, change_of_basis=cob
        )

        # Numeric check: displace the *cartesian* position, map through C, and
        # finite-difference the model on batch-frame inputs.
        delta = 1e-6
        direction = cob[:, axis_index]
        plus = batch.clone()
        minus = batch.clone()
        plus["positions"] = batch["positions"].clone()
        minus["positions"] = batch["positions"].clone()
        plus["positions"][atom_index] += delta * direction
        minus["positions"][atom_index] -= delta * direction
        for key in ("node_labels", "edge_labels"):
            numeric = (model(plus)[key] - model(minus)[key]) / (2 * delta)
            torch.testing.assert_close(derivative[key], numeric, rtol=1e-6, atol=1e-8)

    def test_change_of_basis_identity_matches_raw_column(self) -> None:
        batch = make_batch()
        result = compute_graph2mat_position_jacobian(LinearFakeModel(), batch)
        raw = select_derivative_prediction_from_jacobian(result.jacobian, result.spec, 0, 1)
        identity = select_derivative_prediction_from_jacobian(
            result.jacobian, result.spec, 0, 1, change_of_basis=torch.eye(3, dtype=torch.float64)
        )
        torch.testing.assert_close(raw["node_labels"], identity["node_labels"])
        torch.testing.assert_close(raw["edge_labels"], identity["edge_labels"])

    def test_out_of_range_selection_is_rejected(self) -> None:
        batch = make_batch()
        result = compute_graph2mat_position_jacobian(LinearFakeModel(), batch)
        with self.assertRaises(Graph2MatAutogradDerivativeError):
            select_derivative_prediction_from_jacobian(
                result.jacobian, result.spec, N_ATOMS, 0
            )
        with self.assertRaises(Graph2MatAutogradDerivativeError):
            select_derivative_prediction_from_jacobian(result.jacobian, result.spec, 0, 3)


class SparseConversionTests(unittest.TestCase):
    def test_sub_point_matrix_is_disabled_for_derivative_conversion(self) -> None:
        from graph2mat_autograd_derivatives import derivative_prediction_to_sparse_matrices

        recorded: dict[str, object] = {}

        class FakeProcessor:
            sub_point_matrix = True
            default_out_format = "scipy_csr"

            def copy(self, **kwargs):
                clone = FakeProcessor()
                for key, value in kwargs.items():
                    setattr(clone, key, value)
                recorded["copy_kwargs"] = kwargs
                return clone

            def yield_from_batch(self, batch, predictions=None, as_matrix=False):
                recorded["predictions"] = predictions
                yield "example"

            def labels_to(self, out_format, data=None, threshold=None):
                recorded["out_format"] = out_format
                recorded["threshold"] = threshold
                from scipy import sparse

                return sparse.csr_matrix(np.eye(2))

        derivative = {
            "node_labels": torch.ones(3, requires_grad=True),
            "edge_labels": torch.ones(4, requires_grad=True),
        }
        matrices = derivative_prediction_to_sparse_matrices(
            FakeProcessor(), FakeBatch(), derivative
        )

        self.assertEqual(len(matrices), 1)
        self.assertEqual(recorded["copy_kwargs"], {"sub_point_matrix": False})
        self.assertIsNone(recorded["threshold"])
        self.assertFalse(recorded["predictions"]["node_labels"].requires_grad)


# --------------------------------------------------------------------------- #
# Real-checkpoint smoke: autograd jacobian vs finite-difference of the model
# itself (never SIESTA). Skips cleanly when no checkpoint is available.
# --------------------------------------------------------------------------- #
_SMOKE_SWEEP_ROOT = (
    REPO_ROOT / "Comparison" / "results" / "e2e_smoke_12snap_20ep" / "e2e_smoke_12snap_20ep" / "sweep"
)
_DEFAULT_SMOKE_CKPT = (
    _SMOKE_SWEEP_ROOT / "graph2mat" / "graphene_w90_scale_iid12" / "G2M-E2E12-20EP"
    / "graph2mat" / "training" / "lightning_logs" / "my_first_model" / "version_0"
    / "checkpoints" / "best-160.ckpt"
)
_DEFAULT_SMOKE_STRUCTURE = (
    _SMOKE_SWEEP_ROOT / "derivative_workflows" / "graphene_w90_scale_iid12"
    / "structures" / "md_11_base"
)


def _flat_labels(predictions: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat(
        [predictions[key].detach().reshape(-1) for key in ("node_labels", "edge_labels")]
    ).to(torch.float64)


@pytest.mark.slow
def test_autograd_jacobian_matches_model_finite_difference_real_checkpoint(tmp_path):
    """dH_pred/dR (autograd, 1 atom, 1 axis) vs central FD of the model itself."""
    checkpoint = Path(os.environ.get("G2M_AUTOGRAD_SMOKE_CKPT") or _DEFAULT_SMOKE_CKPT)
    structure_dir = Path(
        os.environ.get("G2M_AUTOGRAD_SMOKE_STRUCTURE") or _DEFAULT_SMOKE_STRUCTURE
    )
    if not checkpoint.is_file():
        pytest.skip(f"No Graph2Mat checkpoint available: {checkpoint}")
    if not (structure_dir / "RUN.fdf").is_file():
        pytest.skip(f"No base structure RUN.fdf available: {structure_dir}")
    basis_glob = os.environ.get("G2M_AUTOGRAD_SMOKE_BASIS") or str(structure_dir / "*.ion.xml")
    import glob as _glob

    if not _glob.glob(basis_glob):
        pytest.skip(f"No basis .ion.xml files match: {basis_glob}")
    try:
        from torch_safe_globals import allow_graph2mat_checkpoint_globals

        # Importing graph2mat/mace already torch.load()s data files, so the
        # safe globals must be registered before the imports.
        allow_graph2mat_checkpoint_globals()

        from graph2mat.tools.lightning import MatrixDataModule
        from graph2mat.tools.lightning.models.mace import LitMACEMatrixModel

        from predict_model_on_dataset import (
            checkpoint_training_dir,
            normalize_pattern_for_workdir,
        )
    except ImportError as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"graph2mat stack not importable: {exc}")
    run_cwd = checkpoint_training_dir(checkpoint)
    source_cwd = Path.cwd()
    runs_json = tmp_path / "smoke_runs.json"
    runs_json.write_text(
        json.dumps({"predict": [str(structure_dir / "RUN.fdf")]}), encoding="utf-8"
    )
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
        cob = torch.as_tensor(
            datamodule.data_processor.basis_table.change_of_basis, dtype=torch.float64
        )
    finally:
        os.chdir(source_cwd)

    atom_index, axis_index = 0, 0
    result = compute_graph2mat_position_jacobian(model.model, batch)
    derivative = select_derivative_prediction_from_jacobian(
        result.jacobian, result.spec, atom_index, axis_index, change_of_basis=cob
    )
    autograd_flat = _flat_labels(derivative)
    assert autograd_flat.abs().max().item() > 0.0

    positions = batch["positions"]
    # Batch positions live in the e3nn frame (p_batch = C @ p_cart), so a
    # cartesian displacement along axis a maps to the direction C[:, a].
    direction = cob.to(positions.dtype)[:, axis_index]
    for delta in (0.003, 0.01):
        displaced = {}
        for sign in (1.0, -1.0):
            shifted = batch.clone()
            shifted_positions = positions.detach().clone()
            shifted_positions[atom_index] += sign * delta * direction
            shifted["positions"] = shifted_positions
            with torch.no_grad():
                displaced[sign] = _flat_labels(model.model(shifted))
        fd_flat = (displaced[1.0] - displaced[-1.0]) / (2.0 * delta)

        cos = torch.dot(autograd_flat, fd_flat) / (
            autograd_flat.norm() * fd_flat.norm()
        )
        rel_frobenius = (autograd_flat - fd_flat).norm() / fd_flat.norm()
        mae = (autograd_flat - fd_flat).abs().mean()
        print(
            f"[autograd-smoke] delta={delta} cos={cos.item():.6f} "
            f"rel_frobenius={rel_frobenius.item():.3e} mae_eV_per_ang={mae.item():.3e}"
        )
        assert cos.item() >= 0.999, (
            f"autograd vs model finite-difference cosine {cos.item():.6f} < 0.999 "
            f"(delta={delta}, rel_frobenius={rel_frobenius.item():.3e})"
        )
        assert rel_frobenius.item() <= 0.25, (
            f"autograd vs model finite-difference relative Frobenius "
            f"{rel_frobenius.item():.3e} > 2.5e-1 (delta={delta})"
        )


class AutogradPredictionScriptTests(unittest.TestCase):
    def test_missing_dataloader_structure_becomes_failed_row(self) -> None:
        rows = []
        requests = {
            "base_0_base": {
                "base_sample_id": "base_0",
                "pairs": {(0, "x"): {0.01}},
            }
        }

        _append_missing_structure_rows(rows, requests, seen_structure_ids=set())

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "error")
        self.assertEqual(rows[0]["base_structure_sample_id"], "base_0_base")
        self.assertEqual(rows[0]["error"], "missing_base_structure_from_graph2mat_dataloader")


if __name__ == "__main__":
    unittest.main()
