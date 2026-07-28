from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "Comparison" / "scripts" / "ops" / "revalidate_artifact_manifests.py"
SPEC = importlib.util.spec_from_file_location("revalidate_artifact_manifests", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_snapshot(path: Path, *, complete: bool) -> None:
    path.mkdir(parents=True)
    (path / "RUN.fdf").write_text("SystemLabel x\n", encoding="utf-8")
    (path / "metadata.json").write_text('{"system_label":"x"}\n', encoding="utf-8")
    for suffix in (".TSHS", ".TSDE", ".HSX", ".STRUCT_OUT", ".XV", ".ORB_INDX"):
        (path / f"x{suffix}").write_text("x\n", encoding="utf-8")
    (path / "RUN.out").write_text(
        "iscf     Eharris\nSCF cycle converged\nJob completed\n" if complete else "startup only\n",
        encoding="utf-8",
    )


class ArtifactManifestRevalidationTests(unittest.TestCase):
    def test_declared_valid_truncated_output_is_quarantined(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample = root / "dataset" / "sample"
            write_snapshot(sample, complete=False)
            manifest = root / "dataset" / "artifact_validation.json"
            manifest.write_text(
                json.dumps({"valid": True, "snapshots": [{"snapshot_dir": str(sample), "valid": True}]}),
                encoding="utf-8",
            )

            report = MODULE.build_report(root)

            self.assertEqual(report["status_counts"], {"quarantined": 1})
            self.assertEqual(report["manifests"][0]["parser_status_counts"], {"scf_not_started": 1})


if __name__ == "__main__":
    unittest.main()
