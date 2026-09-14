import csv
import math
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "Comparison" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import compare_graph2mat_deeph as compare_mod  # noqa: E402
import g2m_deeph_metrics as canon  # noqa: E402


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class SummarizeMethodMatchesCanonicalTests(unittest.TestCase):
    """h_mae_eV_mean (Fase 3, item 3): compare_graph2mat_deeph.py must report the
    same value as the canonical g2m_deeph_metrics.py, including the sparse
    (union) fallback for a run with no dense k-point matrix rows."""

    def test_dense_matrix_rows_are_used_when_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_csv(
                root / "kpoint_matrix_metrics.csv",
                ["row_type", "h_mae_eV"],
                [
                    {"row_type": "weighted_sample", "h_mae_eV": "0.10"},
                    {"row_type": "weighted_sample", "h_mae_eV": "0.20"},
                    {"row_type": "other", "h_mae_eV": "999"},
                ],
            )
            (root / "manifest.json").write_text("{}", encoding="utf-8")

            summary = compare_mod.summarize_method("x", root.parent, metrics_root=root)
            canonical = canon.mean(
                [canon.number(row.get("h_mae_eV")) for row in canon.weighted_sample_rows(root)]
            )
            self.assertAlmostEqual(summary["h_mae_eV_mean"], 0.15)
            self.assertAlmostEqual(summary["h_mae_eV_mean"], canonical)

    def test_falls_back_to_sparse_union_when_no_matrix_rows(self):
        """Before this fix: an empty kpoint_matrix_metrics.csv made this script
        report NaN even though the run has a real sparse-union H-MAE — while the
        canonical g2m_deeph_metrics.py already fell back to it. Same run, two
        different reported numbers. This is the bug Fase 3 closes."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_csv(root / "kpoint_matrix_metrics.csv", ["row_type", "h_mae_eV"], [])
            write_csv(
                root / "sparse_metrics.csv",
                ["mae_union_eV", "rmse_union_eV", "relative_frobenius_union"],
                [
                    {"mae_union_eV": "0.01", "rmse_union_eV": "0.02", "relative_frobenius_union": "0.03"},
                    {"mae_union_eV": "0.03", "rmse_union_eV": "0.04", "relative_frobenius_union": "0.05"},
                ],
            )
            (root / "manifest.json").write_text("{}", encoding="utf-8")

            summary = compare_mod.summarize_method("x", root.parent, metrics_root=root)
            canonical = canon.mean(
                [canon.number(row.get("h_mae_eV")) for row in canon.weighted_sample_rows(root)]
                or [
                    canon.number(row.get("mae_union_eV"))
                    for row in csv.DictReader((root / "sparse_metrics.csv").open(encoding="utf-8"))
                ]
            )
            self.assertFalse(math.isnan(summary["h_mae_eV_mean"]))
            self.assertAlmostEqual(summary["h_mae_eV_mean"], 0.02)
            self.assertAlmostEqual(summary["h_mae_eV_mean"], canonical)

    def test_nan_when_neither_matrix_nor_sparse_rows_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_csv(root / "kpoint_matrix_metrics.csv", ["row_type", "h_mae_eV"], [])
            (root / "manifest.json").write_text("{}", encoding="utf-8")

            summary = compare_mod.summarize_method("x", root.parent, metrics_root=root)
            self.assertTrue(math.isnan(summary["h_mae_eV_mean"]))


if __name__ == "__main__":
    unittest.main()
