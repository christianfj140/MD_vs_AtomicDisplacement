import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Comparison/scripts"))

from tbg_spectral_training import assemble_k_matrix, frontier_terms


def test_assemble_k_matrix_folds_periodic_edges_and_backpropagates() -> None:
    node = torch.tensor([2.0], requires_grad=True)
    edge = torch.tensor([3.0], requires_grad=True)
    matrix = assemble_k_matrix(
        node,
        edge,
        [1],
        torch.tensor([[0], [0]]),
        torch.tensor([[1.0, 0.0, 0.0]]),
        torch.eye(3),
        kpoint=(0.5, 0.0, 0.0),
    )
    assert matrix.item().real == pytest.approx(-4.0)
    matrix.real.sum().backward()
    assert node.grad.item() == pytest.approx(1.0)
    assert edge.grad.item() == pytest.approx(-2.0)


def test_frontier_loss_ignores_a_rigid_shift_but_raw_rmse_reports_it() -> None:
    reference = torch.tensor([-5.0, -0.2, -0.1, 0.1, 0.2, 4.0])
    predicted = reference + 0.03
    loss, aligned, raw = frontier_terms(predicted, reference)
    assert loss.item() == pytest.approx(0.0, abs=1e-12)
    assert aligned.item() == pytest.approx(0.0, abs=1e-7)
    assert raw.item() == pytest.approx(0.03, abs=1e-7)
