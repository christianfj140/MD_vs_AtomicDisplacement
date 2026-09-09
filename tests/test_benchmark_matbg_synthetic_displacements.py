"""E-F_001-S37 / D06: the MATBG synthetic-displacement resource benchmark.

Exercises the pure, checkpoint-free pieces (labeling guard, cost-model
arithmetic, the per-direction measurement on a fake model) without requiring
the real (31, 30) checkpoint, SIESTA overlap or Julia solver -- those are only
reachable through ``run()`` against real artifacts and are covered by the
script's own ``--skip-eigenspace-stage``/cache-hit paths at run time.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from scipy import sparse

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "Comparison" / "scripts"
for _path in (SCRIPTS_DIR, REPO_ROOT / "shared", REPO_ROOT / "scripts" / "torch_serialization_compat"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import fd_perturbation_space as fdp  # noqa: E402
import benchmark_matbg_synthetic_displacements as bench  # noqa: E402
from graph2mat_autograd_derivatives import CPU_BACKEND_RECORD  # noqa: E402


class Batch(dict):
    def clone(self) -> "Batch":
        return Batch(self)


N_ATOMS = 4


class LinearModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(1)
        self.a = torch.randn(3, N_ATOMS * 3, generator=generator, dtype=torch.float64)
        self.b = torch.randn(5, N_ATOMS * 3, generator=generator, dtype=torch.float64)

    def forward(self, data):
        flat = data["positions"].reshape(-1)
        return {"node_labels": self.a @ flat, "edge_labels": self.b @ flat}


class FakeDataProcessor:
    """Minimal stand-in: converts any derivative to one fixed-shape sparse matrix."""

    sub_point_matrix = False
    default_out_format = "scipy_csr"

    def copy(self, **kwargs):
        clone = FakeDataProcessor()
        for key, value in kwargs.items():
            setattr(clone, key, value)
        return clone

    def yield_from_batch(self, batch, predictions=None, as_matrix=False):
        yield "example"

    def labels_to(self, out_format, data=None, threshold=None):
        return sparse.csr_matrix(np.eye(4))


def make_batch() -> Batch:
    generator = torch.Generator().manual_seed(2)
    return Batch(positions=torch.randn(N_ATOMS, 3, generator=generator, dtype=torch.float64))


# --------------------------------------------------------------------------- #
# Labeling guard
# --------------------------------------------------------------------------- #


class LabelGuardTests(unittest.TestCase):
    def test_clean_payload_passes(self) -> None:
        bench.assert_no_forbidden_labels(
            {"displacement_class": bench.DISPLACEMENT_CLASS, "value": 1.0, "nested": ["ok"]}
        )

    def test_phonon_eigenmode_label_is_refused(self) -> None:
        with self.assertRaises(bench.BenchmarkError):
            bench.assert_no_forbidden_labels({"schema": "phonon_eigenmode_v1"})

    def test_physical_g_label_is_refused_in_nested_value(self) -> None:
        with self.assertRaises(bench.BenchmarkError):
            bench.assert_no_forbidden_labels({"result": {"kind": "physical_g"}})

    def test_physical_phonon_label_is_refused(self) -> None:
        with self.assertRaises(bench.BenchmarkError):
            bench.assert_no_forbidden_labels({"kind": "physical_phonon"})

    def test_write_json_runs_the_guard(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.json"
            with self.assertRaises(bench.BenchmarkError):
                bench.write_json(path, {"schema": "phonon_eigenmode_v1"})
            self.assertFalse(path.exists())
            bench.write_json(path, {"displacement_class": bench.DISPLACEMENT_CLASS})
            self.assertTrue(path.exists())


# --------------------------------------------------------------------------- #
# Per-direction measurement (materialize-once + contract-before-materialize)
# --------------------------------------------------------------------------- #


class BenchmarkDirectionTests(unittest.TestCase):
    def test_reports_both_paths_and_stays_synthetic_labeled(self) -> None:
        model = LinearModel()
        batch = make_batch()
        direction = fdp.random_collective(N_ATOMS, seed=0)

        payload = bench.benchmark_direction(
            torch,
            model,
            batch,
            FakeDataProcessor(),
            direction,
            change_of_basis=None,
            backend_record=CPU_BACKEND_RECORD,
            seed=0,
        )

        self.assertEqual(payload["displacement_class"], bench.DISPLACEMENT_CLASS)
        self.assertEqual(payload["direction_kind"], fdp.KIND_COLLECTIVE)
        self.assertIn("wall_seconds", payload["materialize_once"])
        self.assertIn("wall_seconds", payload["contract_before_materialize"])
        self.assertEqual(payload["materialize_once"]["nnz"], 4)
        bench.assert_no_forbidden_labels(payload)  # never raises


# --------------------------------------------------------------------------- #
# A/B crossover arithmetic
# --------------------------------------------------------------------------- #


class CrossoverTableTests(unittest.TestCase):
    def test_cheap_materialize_favors_reuse_at_large_block_counts(self) -> None:
        table = bench.crossover_table(
            t_materialize_seconds=1.0,
            t_vjp_contract_seconds=0.1,
            dense_contract_rate=0.0001,
            nk_grid=(1, 100),
            band_pairs_grid=(1, 100),
        )
        big = next(row for row in table["grid"] if row["nk"] == 100 and row["band_pairs"] == 100)
        self.assertEqual(big["cheaper"], "materialize_once")
        small = next(row for row in table["grid"] if row["nk"] == 1 and row["band_pairs"] == 1)
        self.assertEqual(small["cheaper"], "contract_before_materialize")

    def test_crossover_n_blocks_matches_the_grid_switch(self) -> None:
        table = bench.crossover_table(
            t_materialize_seconds=2.0,
            t_vjp_contract_seconds=0.05,
            dense_contract_rate=0.0,
            nk_grid=(1,),
            band_pairs_grid=(1,),
        )
        # crossover: n_blocks where t_materialize == n_blocks * t_vjp_contract
        self.assertAlmostEqual(table["crossover_n_blocks"], 2.0 / 0.05, places=6)

    def test_degenerate_rate_leaves_crossover_undefined(self) -> None:
        table = bench.crossover_table(
            t_materialize_seconds=1.0,
            t_vjp_contract_seconds=0.01,
            dense_contract_rate=0.01,
            nk_grid=(1,),
            band_pairs_grid=(1,),
        )
        self.assertIsNone(table["crossover_n_blocks"])


class DenseContractRateTests(unittest.TestCase):
    def test_rate_is_positive_and_finite(self) -> None:
        matrix = sparse.random(50, 50, density=0.05, format="csr")
        rate = bench.dense_contract_rate_seconds_per_block(
            matrix, n_orbitals=50, band_pairs_samples=(1, 4)
        )
        self.assertGreater(rate, 0.0)
        self.assertTrue(np.isfinite(rate))


class SparseBytesTests(unittest.TestCase):
    def test_matches_the_three_backing_arrays(self) -> None:
        matrix = sparse.csr_matrix(np.eye(6))
        expected = matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
        self.assertEqual(bench.sparse_bytes(matrix), expected)


# --------------------------------------------------------------------------- #
# GO-8 verdict
# --------------------------------------------------------------------------- #


class ResourceMarginsTests(unittest.TestCase):
    def test_margin_flags_are_computed_from_this_runs_own_peaks_not_machine_totals(self) -> None:
        within = bench.resource_margins(peak_rss_bytes=1 * 1024**3, peak_vram_bytes=1 * 1024**3)
        self.assertTrue(within["ram_within_margin"])
        self.assertTrue(within["vram_within_margin"])

        over_vram = bench.resource_margins(
            peak_rss_bytes=1 * 1024**3, peak_vram_bytes=(bench.VRAM_MARGIN_GIB + 5) * 1024**3
        )
        self.assertFalse(over_vram["vram_within_margin"])
        self.assertTrue(over_vram["ram_within_margin"])

    def test_vram_margin_catches_usage_outside_the_pytorch_allocator(self) -> None:
        # The Julia/cuDSS subprocess holds VRAM the PyTorch allocator never
        # sees; the system-wide nvidia-smi reading must still trip the guard.
        real_gpu_status = bench.gpu_status
        bench.gpu_status = lambda: {
            "temperature_c": 50.0,
            "used_bytes": (bench.VRAM_MARGIN_GIB + 5) * 1024**3,
            "free_bytes": 1 * 1024**3,
        }
        try:
            margins = bench.resource_margins(peak_rss_bytes=1 * 1024**3, peak_vram_bytes=0.0)
        finally:
            bench.gpu_status = real_gpu_status
        self.assertFalse(margins["vram_within_margin"])

    def test_total_rss_bytes_includes_the_eigenspace_subprocess(self) -> None:
        self_only = bench.total_rss_bytes()
        with_child = bench.total_rss_bytes(child_rss_delta_kib=5 * 1024**2)
        self.assertGreater(with_child, self_only)
        self.assertAlmostEqual(with_child - self_only, 5 * 1024**2 * 1024, delta=1024)


class Go8VerdictTests(unittest.TestCase):
    _MARGINS_OK = {"vram_free_gib": 30.0, "free_disk_percent": 50.0}

    def test_no_go_on_low_disk(self) -> None:
        verdict = bench.go8_verdict(
            direction_results=[{"ok": True}],
            eigenspace_result={"status": "completed"},
            margins={"vram_free_gib": 30.0, "free_disk_percent": 5.0},
        )
        self.assertEqual(verdict["status"], "NO_GO")
        self.assertTrue(any("disk" in reason for reason in verdict["reasons"]))

    def test_no_go_on_failed_eigenspace_stage(self) -> None:
        verdict = bench.go8_verdict(
            direction_results=[{"ok": True}],
            eigenspace_result={"status": "failed", "reason": "julia crashed"},
            margins=self._MARGINS_OK,
        )
        self.assertEqual(verdict["status"], "NO_GO")

    def test_clean_run_is_not_a_no_go(self) -> None:
        verdict = bench.go8_verdict(
            direction_results=[{"ok": True}],
            eigenspace_result={"status": "completed"},
            margins=self._MARGINS_OK,
        )
        self.assertNotEqual(verdict["status"], "NO_GO")
        self.assertEqual(verdict["reasons"], [])


if __name__ == "__main__":
    unittest.main()
