from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "Comparison" / "scripts" / "ops" / "audit_geometry_independence.py"
SPEC = importlib.util.spec_from_file_location("audit_geometry_independence", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_manifest(path: Path, hashes: list[str]) -> None:
    path.write_text(
        json.dumps({"samples": [{"geometry_sha256": value} for value in hashes]}),
        encoding="utf-8",
    )


class GeometryIndependenceAuditTests(unittest.TestCase):
    def test_duplicate_seed_geometries_are_not_independent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            left, right = root / "left.json", root / "right.json"
            write_manifest(left, ["a", "b"])
            write_manifest(right, ["b", "c"])

            audit = MODULE.build_audit([left, right])

            self.assertFalse(audit["independent_replica_claim_allowed"])
            self.assertEqual(audit["pairs"][0]["duplicate_geometry_count"], 1)
            self.assertAlmostEqual(audit["pairs"][0]["jaccard"], 1 / 3)

    def test_disjoint_geometry_sets_are_independent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            left, right = root / "left.json", root / "right.json"
            write_manifest(left, ["a"])
            write_manifest(right, ["b"])

            self.assertTrue(MODULE.build_audit([left, right])["independent_replica_claim_allowed"])


if __name__ == "__main__":
    unittest.main()
