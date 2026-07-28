import json
import random
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "Comparison" / "scripts"
for path in (SCRIPTS_DIR,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from deeph_config import build_deeph_raw_mirror, render_train_config  # noqa: E402
from deeph_split_audit import (  # noqa: E402
    STATUS_INCOMPATIBLE,
    STATUS_UNVERIFIED,
    STATUS_VALID,
    audit_deeph_split,
)


class _FakeNumpyRandom:
    def __init__(self) -> None:
        self._rng = random.Random(0)

    def seed(self, seed: int) -> None:
        self._rng = random.Random(int(seed))

    def shuffle(self, values: list[int]) -> None:
        self._rng.shuffle(values)


class _FakeNumpy:
    def __init__(self) -> None:
        self.random = _FakeNumpyRandom()


@contextmanager
def fake_numpy():
    previous = sys.modules.get("numpy")
    sys.modules["numpy"] = _FakeNumpy()  # type: ignore[assignment]
    try:
        yield
    finally:
        if previous is None:
            sys.modules.pop("numpy", None)
        else:
            sys.modules["numpy"] = previous


def write_snapshot(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "RUN.fdf").write_text("SystemLabel graphene\n", encoding="utf-8")
    (path / "RUN.out").write_text(
        "iscf     Eharris\nSCF cycle converged\nJob completed\n",
        encoding="utf-8",
    )
    (path / "metadata.json").write_text(json.dumps({"system_label": "graphene"}) + "\n", encoding="utf-8")
    for suffix in (".TSHS", ".TSDE", ".HSX", ".STRUCT_OUT", ".XV", ".ORB_INDX"):
        (path / f"graphene{suffix}").write_text(f"{suffix}\n", encoding="utf-8")


def frozen_split_for(root: Path, splits: list[str]) -> dict:
    rows = []
    for index, split in enumerate(splits):
        sample_dir = root / "source" / str(index)
        write_snapshot(sample_dir)
        rows.append(
            {
                "sample_id": f"sample{index}",
                "split": split,
                "sample_dir": str(sample_dir),
            }
        )
    return {
        "valid": True,
        "split_hash": "unit-split",
        "split_counts": {
            "train": sum(1 for split in splits if split == "train"),
            "validation": sum(1 for split in splits if split == "validation"),
            "test": sum(1 for split in splits if split == "test"),
        },
        "rows": rows,
    }


def write_train_config(root: Path, raw_mirror: dict, *, seed: int) -> Path:
    path = root / "config" / "train.ini"
    render_train_config(
        path,
        processed_dir=root / "processed",
        graph_dir=root / "graph",
        save_dir=root / "train",
        dataset_name="graphene_unit",
        split_ratios=raw_mirror["split_ratios"],
        seed=seed,
        epochs=1,
        batch_size=1,
        learning_rate=0.001,
        disable_cuda=True,
        device="cpu",
    )
    return path


def write_processed_from_mirror(raw_mirror: dict, processed_dir: Path) -> None:
    raw_root = Path(str(raw_mirror["raw_dir"])).resolve(strict=False)
    for row in raw_mirror["rows"]:
        raw_dir = Path(str(row["raw_dir"])).resolve(strict=False)
        relative = raw_dir.relative_to(raw_root)
        target = processed_dir / relative
        target.mkdir(parents=True, exist_ok=True)
        (target / "rc.h5").write_text("rc\n", encoding="utf-8")


class DeepHSplitAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def build_case(self, *, seed: int = 123) -> tuple[dict, dict, Path]:
        frozen = frozen_split_for(self.root, ["train", "validation", "test"])
        raw_mirror = build_deeph_raw_mirror(
            frozen,
            raw_dir=self.root / "deeph" / "raw",
            workspace_root=self.root / "deeph",
            seed=seed,
        )
        train_config = write_train_config(self.root / "deeph", raw_mirror, seed=seed)
        write_processed_from_mirror(raw_mirror, self.root / "deeph" / "processed")
        return frozen, raw_mirror, train_config

    def test_matching_deeph_split_audit_passes(self) -> None:
        with fake_numpy():
            frozen, raw_mirror, train_config = self.build_case()

            audit = audit_deeph_split(
                frozen_split_manifest=frozen,
                raw_mirror=raw_mirror,
                processed_dir=self.root / "deeph" / "processed",
                train_config_path=train_config,
                output_json=self.root / "deeph" / "deeph_split_audit.json",
                output_csv=self.root / "deeph" / "deeph_split_audit.csv",
            )

        self.assertEqual(audit["status"], STATUS_VALID)
        self.assertTrue(audit["robust_winner_allowed"])
        self.assertTrue((self.root / "deeph" / "deeph_split_audit.json").exists())
        self.assertTrue((self.root / "deeph" / "deeph_split_audit.csv").exists())

    def test_swapped_frozen_split_fails(self) -> None:
        with fake_numpy():
            frozen, raw_mirror, train_config = self.build_case()
            swapped = {**frozen, "rows": [dict(row) for row in frozen["rows"]]}
            swapped["rows"][0]["split"] = "test"
            swapped["rows"][2]["split"] = "train"

            audit = audit_deeph_split(
                frozen_split_manifest=swapped,
                raw_mirror=raw_mirror,
                processed_dir=self.root / "deeph" / "processed",
                train_config_path=train_config,
            )

        self.assertEqual(audit["status"], STATUS_INCOMPATIBLE)
        self.assertFalse(audit["robust_winner_allowed"])
        self.assertTrue(audit["mismatched_rows"])

    def test_unknown_processed_ordering_is_unverified(self) -> None:
        frozen, raw_mirror, train_config = self.build_case()
        processed_dir = self.root / "deeph" / "processed"
        for rc_file in processed_dir.rglob("rc.h5"):
            rc_file.unlink()

        audit = audit_deeph_split(
            frozen_split_manifest=frozen,
            raw_mirror=raw_mirror,
            processed_dir=processed_dir,
            train_config_path=train_config,
        )

        self.assertEqual(audit["status"], STATUS_UNVERIFIED)
        self.assertEqual(audit["comparability_status"], STATUS_UNVERIFIED)
        self.assertFalse(audit["robust_winner_allowed"])

    def test_different_train_seed_is_invalid(self) -> None:
        with fake_numpy():
            frozen, raw_mirror, _train_config = self.build_case(seed=123)
            train_config = write_train_config(self.root / "deeph", raw_mirror, seed=456)

            audit = audit_deeph_split(
                frozen_split_manifest=frozen,
                raw_mirror=raw_mirror,
                processed_dir=self.root / "deeph" / "processed",
                train_config_path=train_config,
            )

        self.assertIn(audit["status"], {STATUS_INCOMPATIBLE, STATUS_UNVERIFIED})
        self.assertFalse(audit["robust_winner_allowed"])
        self.assertTrue(any("seed" in error for error in audit["errors"]))


if __name__ == "__main__":
    unittest.main()
