from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "Comparison" / "scripts"
SHARED_DIR = REPO_ROOT / "shared"


def load_script_module(name: str, relative_path: str):
    for path in (SCRIPTS_DIR, SHARED_DIR):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / relative_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


aggregate = load_script_module("aggregate_cross_metrics_material_tests", "aggregate_cross_metrics.py")
material_provenance = load_script_module("material_provenance_tests", "material_provenance.py")
analyze_winners = load_script_module("analyze_winners_material_tests", "analyze_winners.py")

if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))
from material_presets import resolve_material_bundle  # noqa: E402


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class MaterialProvenanceAggregationTests(unittest.TestCase):
    def make_cross_result(self, root: Path, manifest: dict[str, object]) -> Path:
        result_dir = root / "cross_md_on_test_mixed"
        metrics_dir = result_dir / "metrics"
        metrics_dir.mkdir(parents=True)
        (result_dir / "cross_evaluation_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )
        (result_dir / "prediction_summary.json").write_text(
            json.dumps({"prediction_time_seconds": 1.25}) + "\n",
            encoding="utf-8",
        )
        (metrics_dir / "manifest.json").write_text(
            json.dumps({"samples_seen": 1, "warnings": [], "fatal_errors": []}) + "\n",
            encoding="utf-8",
        )
        write_csv(
            metrics_dir / "sparse_metrics.csv",
            [{"sample": "sample_001", "global_rmse_eV": 0.1}],
        )
        write_csv(metrics_dir / "spectral_metrics.csv", [])
        write_csv(metrics_dir / "dos_metrics.csv", [])
        return result_dir

    def test_h2o_preset_flattens_to_manifest_ready_material_fields(self) -> None:
        resolved = resolve_material_bundle({"material": {"preset": "h2o"}}, base_dir=REPO_ROOT)

        flattened = material_provenance.flatten_material_provenance({"material": resolved.to_manifest_dict()})

        self.assertEqual(flattened["material_label"], "h2o")
        self.assertEqual(flattened["material_preset"], "h2o")
        self.assertIn("fdf_sha256", flattened)
        self.assertIn("material_identity_hash", flattened)
        self.assertIn("O", flattened["pseudopotential_sha256_by_species"])

    def test_aggregate_rows_include_material_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result_dir = self.make_cross_result(
                Path(tmp),
                {
                    "train_method": "md",
                    "test_set": "test_mixed",
                    "material_provenance": {
                        "material_label": "sic",
                        "material_structure_type": "crystal",
                        "material_species": [{"index": 1, "atomic_number": 14, "label": "Si"}],
                        "fdf_sha256": "fdf123",
                        "pseudopotential_sha256_by_species": {"Si": "pseudo123"},
                        "basis_sha256_by_species": {"Si": "basis123"},
                        "graph2mat_config_hash": "g2m123",
                        "split_manifest_hash": "split123",
                        "reference_matrix_sha256": "ref123",
                        "prediction_matrix_sha256": "pred123",
                    },
                },
            )

            rows = aggregate.aggregate_one(result_dir, "exp_material")

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["material_label"], "sic")
        self.assertEqual(row["fdf_sha256"], "fdf123")
        self.assertEqual(json.loads(row["pseudopotential_sha256_by_species"]), {"Si": "pseudo123"})
        self.assertEqual(row["graph2mat_config_hash"], "g2m123")
        self.assertEqual(row["reference_matrix_sha256"], "ref123")
        self.assertEqual(row["prediction_matrix_sha256"], "pred123")

    def test_legacy_manifest_without_material_fields_still_aggregates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result_dir = self.make_cross_result(
                Path(tmp),
                {
                    "train_method": "md",
                    "test_set": "test_mixed",
                    "dataset_size": 10,
                },
            )

            rows = aggregate.aggregate_one(result_dir, "exp_legacy")

        self.assertEqual(len(rows), 1)
        self.assertIsNone(rows[0]["material_label"])
        self.assertEqual(json.loads(rows[0]["material_label_by_method"]), {})

    def test_incompatible_material_hashes_become_severe_aggregate_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result_dir = self.make_cross_result(
                Path(tmp),
                {
                    "train_method": "md",
                    "test_set": "test_mixed",
                    "method_provenance": {
                        "md": {
                            "material_provenance": {
                                "material_label": "h2o",
                                "material_compatibility_hash": "hash_h2o",
                            }
                        },
                        "siesta_fc_cartesian": {
                            "material_provenance": {
                                "material_label": "sic",
                                "material_compatibility_hash": "hash_sic",
                            }
                        },
                    },
                },
            )

            row = aggregate.aggregate_one(result_dir, "exp_bad_material")[0]

        self.assertIn("INCOMPATIBLE_MATERIAL_PROVENANCE", row["material_compatibility_warning"])
        self.assertIn("INCOMPATIBLE_MATERIAL_PROVENANCE", row["severe_warnings"])
        by_method = json.loads(row["material_label_by_method"])
        self.assertEqual(by_method["md"], "h2o")
        self.assertEqual(by_method["siesta_fc_cartesian"], "sic")

    def test_material_compatibility_warning_blocks_winner_recommendation(self) -> None:
        rows = []
        for method in ("md", "siesta_fc_cartesian"):
            for test_set in ("test_md", "test_siesta_fc_cartesian", "test_mixed"):
                rows.append(
                    {
                        "train_method": method,
                        "test_set": test_set,
                        "seed": 1,
                        "low_energy_rmse_eV": 0.1 if method == "md" else 0.2,
                        "material_compatibility_warning": (
                            "INCOMPATIBLE_MATERIAL_PROVENANCE: material compatibility hashes differ"
                            if method == "md" and test_set == "test_md"
                            else ""
                        ),
                    }
                )

        recommendation = analyze_winners.build_recommendation(
            rows,
            [],
            [],
            primary_metric="low_energy_rmse_eV",
            minimum_robust_seeds=1,
        )

        self.assertIsNone(recommendation["winner"])
        self.assertIn(recommendation["scientific_status"], {"scientifically_inconclusive", "not_scientifically_valid"})
        self.assertTrue(
            any(
                "INCOMPATIBLE_MATERIAL_PROVENANCE" in warning
                for warning in recommendation["severe_warnings"]
            )
        )


if __name__ == "__main__":
    unittest.main()
