import csv
import configparser
import json
import random
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "Comparison" / "scripts"
SHARED_DIR = REPO_ROOT / "shared"
for path in (SCRIPTS_DIR, SHARED_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark_manifest import write_benchmark_manifests  # noqa: E402
from g2m_deeph_runner import Graph2MatDeepHBenchmarkRunner  # noqa: E402
from joint_artifact_contract import validate_dataset  # noqa: E402

try:
    import h5py  # type: ignore[import-not-found]
    import numpy as np  # type: ignore[import-not-found]

    H5PY_AVAILABLE = True
except ImportError:
    h5py = None
    np = None
    H5PY_AVAILABLE = False


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
def fake_numpy_for_split_audit():
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
    (path / "RUN.out").write_text("Job completed\n", encoding="utf-8")
    (path / "metadata.json").write_text(json.dumps({"system_label": "graphene"}) + "\n", encoding="utf-8")
    for suffix in (".TSHS", ".TSDE", ".HSX", ".XV"):
        (path / f"graphene{suffix}").write_text(f"{suffix}\n", encoding="utf-8")
    (path / "graphene.STRUCT_OUT").write_text(
        "1 0 0\n0 1 0\n0 0 1\n1\n1 6 0 0 0\n",
        encoding="utf-8",
    )
    (path / "graphene.ORB_INDX").write_text(
        "      2     2 = orbitals in unit cell and supercell. See end of file.\n\n"
        "io ia is spec iao n l m z p sym rc isc iuo\n"
        "1 1 1 C 1 2 0 0 1 F s 4.0 0 0 0 1\n"
        "2 1 1 C 2 2 1 -1 1 F py 4.0 0 0 0 2\n",
        encoding="utf-8",
    )


def write_split_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["sample_id", "split", "sample_dir", "structure_path", "hamiltonian_path", "metadata_path"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_h5(path: Path, blocks: dict[str, object]) -> None:
    assert h5py is not None
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        for key, value in blocks.items():
            handle[key] = np.asarray(value, dtype=float)


def write_processed_deeph_blocks(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "orbital_types.dat").write_text("0\n0\n", encoding="utf-8")
    (path / "info.json").write_text(json.dumps({"isspinful": False, "isorthogonal": False}) + "\n", encoding="utf-8")
    write_h5(
        path / "hamiltonians.h5",
        {
            "[0, 0, 0, 1, 1]": [[0.0]],
            "[0, 0, 0, 1, 2]": [[1.0]],
            "[0, 0, 0, 2, 1]": [[1.0]],
            "[0, 0, 0, 2, 2]": [[0.0]],
        },
    )
    write_h5(
        path / "overlaps.h5",
        {
            "[0, 0, 0, 1, 1]": [[1.0]],
            "[0, 0, 0, 1, 2]": [[0.0]],
            "[0, 0, 0, 2, 1]": [[0.0]],
            "[0, 0, 0, 2, 2]": [[1.0]],
        },
    )


def write_processed_split_markers(deeph_context) -> None:
    raw_root = Path(str(deeph_context.raw_mirror["raw_dir"])).resolve(strict=False)
    for row in deeph_context.raw_mirror["rows"]:
        raw_dir = Path(str(row["raw_dir"])).resolve(strict=False)
        processed_sample = deeph_context.processed_dir / raw_dir.relative_to(raw_root)
        processed_sample.mkdir(parents=True, exist_ok=True)
        (processed_sample / "rc.h5").write_text("rc\n", encoding="utf-8")


class DeepHRunnerIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.dataset = self.root / "dataset"
        self.output_root = self.root / "results"
        self.dataset.mkdir(parents=True)
        (self.dataset / "RUN.fdf").write_text("SystemLabel graphene\nSave.HS T\n", encoding="utf-8")
        (self.dataset / "material_provenance.json").write_text(
            json.dumps(
                {
                    "label": "graphene",
                    "basis_file_sha256": {"C.ion.xml": "basis"},
                    "pseudopotential_sha256": {"C": "pseudo"},
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def prepare_manifest_dataset(self) -> None:
        split_root = self.dataset / "splits"
        steps = self.dataset / "MD_steps"
        for index, split in enumerate(("train", "validation", "test")):
            step = steps / str(index)
            split_sample = split_root / split / str(index)
            write_snapshot(step)
            write_snapshot(split_sample)
            write_split_manifest(
                split_root / f"{split}_manifest.csv",
                [
                    {
                        "sample_id": f"md_{index}",
                        "split": split,
                        "sample_dir": str(split_sample),
                        "structure_path": str(split_sample / "RUN.fdf"),
                        "hamiltonian_path": str(split_sample / "graphene.TSHS"),
                        "metadata_path": str(split_sample / "metadata.json"),
                    }
                ],
            )
        artifact_validation = validate_dataset(
            steps,
            snapshot_dirs=[steps / "0", steps / "1", steps / "2"],
        ).to_dict()
        (self.dataset / "artifact_validation.json").write_text(
            json.dumps(artifact_validation, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        write_benchmark_manifests(dataset_root=self.dataset, split_root=split_root)

    def prepare_contexts(self):
        self.prepare_manifest_dataset()
        runner = Graph2MatDeepHBenchmarkRunner()
        validation = runner.validate_dataset_payload({"dataset_root": str(self.dataset)})
        graph2mat_context = runner._prepare_graph2mat_context(
            {
                "dataset_root": str(self.dataset),
                "output_root": str(self.output_root),
                "run_id": "unit_deeph",
                "dry_run": True,
            },
            validation,
        )
        deeph_context = runner._prepare_deeph_context(
            {
                "dataset_root": str(self.dataset),
                "output_root": str(self.output_root),
                "run_id": "unit_deeph",
                "dry_run": True,
                "deeph": {"epochs": 7, "batch_size": 2, "learning_rate": 0.004},
            },
            graph2mat_context,
        )
        return runner, graph2mat_context, deeph_context

    def test_prepare_deeph_context_renders_configs_and_manifest(self) -> None:
        _, graph2mat_context, deeph_context = self.prepare_contexts()

        self.assertTrue(deeph_context.preprocess_config.exists())
        self.assertTrue(deeph_context.train_config.exists())
        self.assertEqual(len(deeph_context.inference_configs), 1)
        self.assertTrue(deeph_context.inference_configs[0].exists())
        self.assertTrue(deeph_context.manifest_path.exists())
        self.assertTrue(str(deeph_context.root).startswith(str(graph2mat_context.run_root)))

        manifest = json.loads(deeph_context.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema"], "graph2mat_deeph_deeph_stage_manifest_v1")
        self.assertEqual(manifest["context"]["split_hash"], graph2mat_context.split_hash)
        self.assertTrue(manifest["config_sha256"]["preprocess"])
        self.assertEqual(manifest["split_audit_status"], "pending")

    def test_prepare_deeph_context_preserves_zero_seed_for_split_audit(self) -> None:
        self.prepare_manifest_dataset()
        runner = Graph2MatDeepHBenchmarkRunner()
        validation = runner.validate_dataset_payload({"dataset_root": str(self.dataset)})
        graph2mat_context = runner._prepare_graph2mat_context(
            {
                "dataset_root": str(self.dataset),
                "output_root": str(self.output_root),
                "run_id": "unit_deeph_seed0",
                "dry_run": True,
            },
            validation,
        )
        deeph_context = runner._prepare_deeph_context(
            {
                "dataset_root": str(self.dataset),
                "output_root": str(self.output_root),
                "run_id": "unit_deeph_seed0",
                "dry_run": True,
                "deeph": {"seed": 0, "epochs": 1, "batch_size": 1, "learning_rate": 0.001},
            },
            graph2mat_context,
        )

        config = configparser.ConfigParser()
        config.read(deeph_context.train_config)
        self.assertEqual(deeph_context.raw_mirror["seed"], 0)
        self.assertEqual(config.getint("basic", "seed"), 0)

    def test_deeph_split_audit_matches_frozen_split_and_manifest(self) -> None:
        with fake_numpy_for_split_audit():
            runner, graph2mat_context, deeph_context = self.prepare_contexts()
            write_processed_split_markers(deeph_context)

            audit = runner._audit_deeph_split(deeph_context, graph2mat_context)
            manifest = runner._write_deeph_manifest(deeph_context, split_audit=audit)

        self.assertTrue(audit["valid"])
        self.assertEqual(manifest["split_audit_status"], "valid")
        self.assertEqual(Path(manifest["split_audit_path"]), deeph_context.split_audit_path)
        self.assertTrue(deeph_context.split_audit_path.exists())

    def test_deeph_split_audit_failure_blocks_runner(self) -> None:
        runner, graph2mat_context, deeph_context = self.prepare_contexts()

        with self.assertRaisesRegex(RuntimeError, "DeepH split audit failed"):
            runner._audit_deeph_split(deeph_context, graph2mat_context)
        manifest = json.loads(deeph_context.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["split_audit_status"], "invalid_unverified_deeph_split")
        self.assertTrue(deeph_context.split_audit_path.exists())

    def test_missing_configured_deeph_cli_fails_clearly(self) -> None:
        runner = Graph2MatDeepHBenchmarkRunner()

        with self.assertRaisesRegex(RuntimeError, "DeepH CLI"):
            runner._deeph_command(
                {"deeph": {"commands": {"deeph-preprocess": str(self.root / "missing-cli")}}},
                "deeph-preprocess",
            )

    def test_stage_deeph_inference_inputs_after_preprocess(self) -> None:
        runner, _, deeph_context = self.prepare_contexts()
        test_row = next(row for row in deeph_context.raw_mirror["rows"] if row["split"] == "test")
        processed_sample = deeph_context.processed_dir / Path(test_row["raw_dir"]).name
        processed_sample.mkdir(parents=True, exist_ok=True)
        (processed_sample / "site_positions.dat").write_text("positions\n", encoding="utf-8")
        (processed_sample / "orbital_types.dat").write_text("orbitals\n", encoding="utf-8")

        staged = runner._stage_deeph_inference_inputs(deeph_context)

        self.assertEqual(staged["count"], 1)
        work_dir = Path(staged["rows"][0]["work_dir"])
        self.assertTrue((work_dir / "site_positions.dat").exists())
        self.assertTrue((work_dir / "orbital_types.dat").exists())

    def test_stage_deeph_inference_inputs_requires_processed_samples(self) -> None:
        runner, _, deeph_context = self.prepare_contexts()

        with self.assertRaisesRegex(RuntimeError, "Missing DeepH processed test samples"):
            runner._stage_deeph_inference_inputs(deeph_context)

    @unittest.skipUnless(H5PY_AVAILABLE, "h5py/numpy are required for DeepH adapter validation")
    def test_validate_deeph_prediction_outputs_writes_adapter_manifest(self) -> None:
        runner, _, deeph_context = self.prepare_contexts()
        test_work_dir = deeph_context.inference_work_dirs[0]
        processed_sample = deeph_context.processed_dir / test_work_dir.name
        write_processed_deeph_blocks(processed_sample)
        test_work_dir.mkdir(parents=True, exist_ok=True)
        write_h5(
            test_work_dir / "hamiltonians_pred.h5",
            {
                "[0, 0, 0, 1, 1]": [[0.0]],
                "[0, 0, 0, 1, 2]": [[1.1]],
                "[0, 0, 0, 2, 1]": [[1.1]],
                "[0, 0, 0, 2, 2]": [[0.0]],
            },
        )

        outputs = runner._validate_deeph_prediction_outputs(deeph_context)

        adapter_manifest = Path(outputs["adapter_manifest"])
        self.assertTrue(adapter_manifest.exists())
        self.assertEqual(outputs["adapter_summary"]["metrics_ready_count"], 1)
        self.assertEqual(outputs["adapter_summary"]["diagnostic_only_count"], 1)

    def test_dry_run_workflow_completes_deeph_command_chain_without_training(self) -> None:
        self.prepare_manifest_dataset()
        runner = Graph2MatDeepHBenchmarkRunner()
        status = runner.start(
            {
                "dataset_root": str(self.dataset),
                "output_root": str(self.output_root),
                "run_id": "dry_run_deeph_workflow",
                "dry_run": True,
                "deeph": {"epochs": 3, "batch_size": 1},
            }
        )
        self.assertTrue(status["running"])
        runner._thread.join(timeout=5)

        final_status = runner.status()
        self.assertFalse(final_status["running"])
        self.assertEqual(final_status["returncode"], 0)
        self.assertEqual(final_status["stage"], "complete")
        self.assertTrue(
            (self.output_root / "dry_run_deeph_workflow" / "deeph" / "deeph_manifest.json").exists()
        )
        results = runner.results()["results"]
        self.assertIsNotNone(results["deeph"])
        self.assertIn("preprocess_config", results["deeph"])


if __name__ == "__main__":
    unittest.main()
