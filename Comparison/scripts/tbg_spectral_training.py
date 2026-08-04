"""Differentiable low-energy spectral objective for primitive bilayer graphene."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import scipy.linalg
import torch
import torch.nn.functional as F

TORCH_COMPAT_DIR = Path(__file__).resolve().parents[2] / "scripts/torch_serialization_compat"
sys.path.insert(0, str(TORCH_COMPAT_DIR))
from torch_safe_globals import allow_graph2mat_checkpoint_globals  # noqa: E402

allow_graph2mat_checkpoint_globals()

from graph2mat.tools.lightning.models.mace import LitMACEMatrixModel

K_POINT = (1 / 3, 1 / 3, 0.0)


def assemble_k_matrix(
    node_labels: torch.Tensor,
    edge_labels: torch.Tensor,
    orbitals: list[int],
    edge_index: torch.Tensor,
    shifts: torch.Tensor,
    cell: torch.Tensor,
    kpoint=K_POINT,
) -> torch.Tensor:
    """Assemble H(k) from full node blocks and one edge from each Hermitian pair."""
    first = np.cumsum([0, *orbitals])
    matrix = node_labels.new_zeros((int(first[-1]), int(first[-1])), dtype=torch.complex64)
    node_offset = 0
    for atom, size in enumerate(orbitals):
        block = node_labels[node_offset : node_offset + size * size].reshape(size, size)
        sl = slice(int(first[atom]), int(first[atom + 1]))
        matrix[sl, sl] = matrix[sl, sl] + block
        node_offset += size * size

    inverse_cell = torch.linalg.inv(cell)
    k = torch.as_tensor(kpoint, dtype=cell.dtype, device=cell.device)
    edge_offset = 0
    for edge, shift in zip(edge_index.T, shifts):
        i, j = (int(edge[0]), int(edge[1]))
        ni, nj = orbitals[i], orbitals[j]
        block = edge_labels[edge_offset : edge_offset + ni * nj].reshape(ni, nj)
        phase = torch.exp(2j * torch.pi * ((shift @ inverse_cell) @ k))
        si = slice(int(first[i]), int(first[i + 1]))
        sj = slice(int(first[j]), int(first[j + 1]))
        matrix[si, sj] = matrix[si, sj] + block * phase
        matrix[sj, si] = matrix[sj, si] + block.T * phase.conj()
        edge_offset += ni * nj

    if node_offset != node_labels.numel() or edge_offset != edge_labels.numel():
        raise ValueError(
            f"Unconsumed matrix labels: nodes {node_offset}/{node_labels.numel()}, "
            f"edges {edge_offset}/{edge_labels.numel()}"
        )
    return (matrix + matrix.mH) / 2


def graph_k_matrix(example, node_labels, edge_labels, basis_table) -> torch.Tensor:
    """Apply Graph2Mat's exact symmetric-edge convention before assembling H(K)."""
    processor = example.metadata["data_processor"]
    mask = processor._get_symmetric_unique_edge_mask(
        example.edge_types.detach().cpu().numpy(),
        expected_nlabels=edge_labels.numel(),
        edge_index=example.edge_index.detach().cpu().numpy(),
        point_types=example.point_types.detach().cpu().numpy(),
    )
    mask = torch.as_tensor(mask, dtype=torch.bool, device=example.edge_index.device)
    orbitals = [basis_table.basis[int(kind)].basis_size for kind in example.point_types]
    return assemble_k_matrix(
        node_labels,
        edge_labels,
        orbitals,
        example.edge_index[:, mask],
        example.shifts[mask],
        example.cell,
    )


def batch_k_matrices(batch, node_labels, edge_labels, basis_table) -> torch.Tensor:
    """Vectorized H(K) assembly while preserving each graph's edge ordering."""
    count = batch.num_graphs
    all_indices, all_values = [], []
    node_offset = edge_offset = 0
    total_orbitals = None
    for graph in range(count):
        example = batch.get_example(graph)
        processor = example.metadata["data_processor"]
        mask = processor._get_symmetric_unique_edge_mask(
            example.edge_types.detach().cpu().numpy(),
            expected_nlabels=example.edge_labels.numel(),
            edge_index=example.edge_index.detach().cpu().numpy(),
            point_types=example.point_types.detach().cpu().numpy(),
        )
        mask = torch.as_tensor(mask, dtype=torch.bool, device=node_labels.device)
        orbitals = [basis_table.basis[int(kind)].basis_size for kind in example.point_types]
        if len(set(orbitals)) != 1:
            raise ValueError(f"Vectorized spectral loss requires one orbital size, got {orbitals}")
        size, atoms = orbitals[0], len(orbitals)
        graph_orbitals = atoms * size
        total_orbitals = total_orbitals or graph_orbitals
        if graph_orbitals != total_orbitals:
            raise ValueError("All spectral-loss graphs must have the same orbital dimension")

        block_rows, block_cols = torch.meshgrid(
            torch.arange(size, device=node_labels.device),
            torch.arange(size, device=node_labels.device),
            indexing="ij",
        )
        node_indices = torch.cat([
            ((atom * size + block_rows) * total_orbitals + atom * size + block_cols).reshape(-1)
            for atom in range(atoms)
        ])
        local_edges = example.edge_index[:, mask]
        rows = local_edges[0, :, None, None] * size + block_rows
        cols = local_edges[1, :, None, None] * size + block_cols
        edge_indices = (rows * total_orbitals + cols).reshape(-1)
        reverse_indices = (cols * total_orbitals + rows).reshape(-1)

        n_node, n_edge = example.point_labels.numel(), example.edge_labels.numel()
        graph_node_values = node_labels[node_offset : node_offset + n_node].to(torch.complex64)
        graph_edge_values = edge_labels[edge_offset : edge_offset + n_edge].to(torch.complex64)
        k = torch.as_tensor(K_POINT, dtype=example.cell.dtype, device=example.cell.device)
        phases = torch.exp(
            2j * torch.pi * (((example.shifts[mask] @ torch.linalg.inv(example.cell)) @ k))
        ).repeat_interleave(size * size)
        all_values.extend(
            [graph_node_values, graph_edge_values * phases, graph_edge_values * phases.conj()]
        )
        base = graph * total_orbitals * total_orbitals
        all_indices.extend([node_indices + base, edge_indices + base, reverse_indices + base])
        node_offset += n_node
        edge_offset += n_edge

    if node_offset != node_labels.numel() or edge_offset != edge_labels.numel():
        raise ValueError("Spectral minibatch labels were not fully consumed")
    matrices = torch.cat(all_values).new_zeros(count * total_orbitals * total_orbitals)
    matrices.scatter_add_(0, torch.cat(all_indices), torch.cat(all_values))
    matrices = matrices.reshape(count, total_orbitals, total_orbitals)
    return (matrices + matrices.mH) / 2


def frontier_terms(
    predicted: torch.Tensor,
    reference: torch.Tensor,
    n_states: int = 4,
    beta_eV: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return shift-invariant Smooth-L1, aligned RMSE and raw RMSE."""
    selection = torch.argsort(reference.abs())[:n_states]
    delta = predicted[selection] - reference[selection]
    aligned = delta - delta.mean()
    return (
        F.smooth_l1_loss(aligned, torch.zeros_like(aligned), beta=beta_eV),
        torch.sqrt(torch.mean(aligned.square())),
        torch.sqrt(torch.mean(delta.square())),
    )


def build_reference_targets(paths: list[str], cache: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Cache reference eigenvalues and S^-1/2 at K for every primitive sample."""
    unique_paths = sorted(set(map(str, paths)))
    if cache.is_file():
        saved = np.load(cache, allow_pickle=False)
        if saved["paths"].tolist() == unique_paths:
            return {
                path: (saved["eigenvalues"][i], saved["inverse_sqrt_overlap"][i])
                for i, path in enumerate(unique_paths)
            }

    import sisl
    from evaluate_hamiltonian_metrics import (
        kpoint_hamiltonian_matrix,
        kpoint_overlap_matrix,
        symmetrized_hermitian_dense,
    )

    eigenvalues, roots = [], []
    for run in unique_paths:
        tshs = next(Path(run).parent.glob("*.TSHS"))
        hamiltonian = sisl.get_sile(str(tshs)).read_hamiltonian()
        h = symmetrized_hermitian_dense(kpoint_hamiltonian_matrix(hamiltonian, K_POINT))
        s = symmetrized_hermitian_dense(kpoint_overlap_matrix(hamiltonian, K_POINT))
        values, vectors = np.linalg.eigh(s)
        if values.min() <= 0:
            raise RuntimeError(f"Non-positive overlap in {tshs}: {values.min():.3e}")
        root = (vectors * values**-0.5) @ vectors.conj().T
        eigenvalues.append(scipy.linalg.eigvalsh(h, s, check_finite=False))
        roots.append(root)

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        paths=np.asarray(unique_paths),
        eigenvalues=np.asarray(eigenvalues),
        inverse_sqrt_overlap=np.asarray(roots),
    )
    return {path: (eigenvalues[i], roots[i]) for i, path in enumerate(unique_paths)}


class SpectralLitMACEMatrixModel(LitMACEMatrixModel):
    """Graph2Mat Lightning wrapper with an auxiliary frontier-state objective."""

    def configure_spectral_loss(self, targets, weight: float = 0.25, beta_eV: float = 0.01):
        self.spectral_targets = targets
        self.spectral_weight = float(weight)
        self.spectral_beta_eV = float(beta_eV)

    def _spectral_terms(self, batch, out):
        h = batch_k_matrices(
            batch, out["node_labels"], out["edge_labels"], self.basis_table
        )
        pairs = [self.spectral_targets[str(path)] for path in batch.metadata["path"]]
        reference = torch.as_tensor(
            np.stack([pair[0] for pair in pairs]), dtype=h.real.dtype, device=h.device
        )
        roots = torch.as_tensor(
            np.stack([pair[1] for pair in pairs]), dtype=h.dtype, device=h.device
        )
        predicted = torch.linalg.eigvalsh(roots @ h @ roots)
        selection = torch.argsort(reference.abs(), dim=1)[:, :4]
        delta = torch.gather(predicted, 1, selection) - torch.gather(reference, 1, selection)
        aligned = delta - delta.mean()
        return (
            F.smooth_l1_loss(aligned, torch.zeros_like(aligned), beta=self.spectral_beta_eV),
            torch.sqrt(torch.mean(aligned.square())),
            torch.sqrt(torch.mean(delta.square())),
        )

    def _shared_step(self, batch, stage: str):
        out = self.model(batch)
        self._validate_pred_ref_shapes(out, batch)
        matrix_loss, stats = self.loss_fn(
            nodes_pred=out["node_labels"],
            nodes_ref=batch["point_labels"],
            edges_pred=out["edge_labels"],
            edges_ref=batch["edge_labels"],
            batch=batch,
            basis_table=self.basis_table,
            log_verbose=stage != "train",
            out=out,
            model=self.model,
        )
        spectral_loss, aligned_rmse, raw_rmse = self._spectral_terms(batch, out)
        loss = matrix_loss + self.spectral_weight * spectral_loss
        on_step = stage == "train"
        self.log(f"{stage}_loss", loss, on_step=on_step, on_epoch=True, prog_bar=True, batch_size=batch.num_graphs)
        self.log(f"{stage}_matrix_loss", matrix_loss, on_step=on_step, on_epoch=True, batch_size=batch.num_graphs)
        self.log(f"{stage}_spectral_frontier_aligned_rmse_eV", aligned_rmse, on_step=on_step, on_epoch=True, prog_bar=stage == "val", batch_size=batch.num_graphs)
        self.log(f"{stage}_spectral_frontier_raw_rmse_eV", raw_rmse, on_step=on_step, on_epoch=True, batch_size=batch.num_graphs)
        for key, value in stats.items():
            self.log(f"{stage}_{key}", value, on_step=on_step, on_epoch=True, batch_size=batch.num_graphs)
        return {**out, "loss": loss}

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        result = self._shared_step(batch, "val")
        self.log("hp_metric", result["loss"], batch_size=batch.num_graphs)
        return result
