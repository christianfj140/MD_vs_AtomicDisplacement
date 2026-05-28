from __future__ import annotations

import contextlib
import io
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "Comparison" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def load_script_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


utils = load_script_module("deeph_fair_utils_tests", "deeph_fair_utils.py")
prepare = load_script_module("prepare_deeph_siesta_dataset_tests", "prepare_deeph_siesta_dataset.py")


def write_split_manifest(root: Path, sample: str, sample_dir: Path, split: str = "train") -> None:
    split_dir = root / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    for name in ("train", "validation", "test"):
        path = split_dir / f"{name}_manifest.csv"
        chosen_sample = sample if name == split else f"{sample}_{name}"
        chosen_dir = sample_dir if name == split else sample_dir.parent / chosen_sample
        chosen_dir.mkdir(parents=True, exist_ok=True)
        (chosen_dir / "RUN.fdf").write_text("SystemLabel graphene\n", encoding="utf-8")
        (chosen_dir / "metadata.json").write_text("{}\n", encoding="utf-8")
        path.write_text(
            "sample_id,sample_dir,structure_path,hamiltonian_path,metadata_path,split,frame_index\n"
            f"md_{chosen_sample},{chosen_dir},{chosen_dir / 'RUN.fdf'},{chosen_dir / 'graphene.TSHS'},{chosen_dir / 'metadata.json'},{name},{chosen_sample}\n",
            encoding="utf-8",
        )


class DeepHFairBenchmarkTests(unittest.TestCase):
    def test_stable_sample_prefers_frozen_sample_id_over_frame_index(self) -> None:
        sample = utils.stable_sample_from_row(
            {
                "sample_id": "md_18",
                "frame_index": "18",
                "sample_dir": "/tmp/dataset/splits/test/18",
            }
        )

        self.assertEqual(sample, "md_18")

    def test_deeph_subprocess_streams_output_and_epoch_progress(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stdout_path = root / "stdout.log"
            stderr_path = root / "stderr.log"
            buffer = io.StringIO()
            command = [
                sys.executable,
                "-u",
                "-c",
                "print('Epoch #3 \\t| Train loss: 0.1', flush=True)",
            ]

            with contextlib.redirect_stdout(buffer):
                returncode = utils.run_subprocess_streaming(
                    command,
                    cwd=root,
                    env=os.environ.copy(),
                    stdout_path=stdout_path,
                    stderr_path=stderr_path,
                    prefix="[DEEPh][train] ",
                    epoch_total=5,
                )

            self.assertEqual(returncode, 0)
            self.assertIn("[DEEPh][train] Epoch #3", buffer.getvalue())
            self.assertIn("[DEEPh-FAIR][epoch] reported_epoch=3/5", buffer.getvalue())
            self.assertIn("Epoch #3", stdout_path.read_text(encoding="utf-8"))
            self.assertIn("stderr was merged into stdout", stderr_path.read_text(encoding="utf-8"))

    def test_graph2mat_deeph_selector_keeps_successful_returncode_zero(self) -> None:
        pipeline_ui = load_script_module("pipeline_ui_selector_tests", "pipeline_ui.py")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_dir = root / "graph2mat_result"
            metrics_dir = result_dir / "metrics"
            metrics_dir.mkdir(parents=True)
            (metrics_dir / "kpoint_spectral_metrics.csv").write_text(
                "sample,low_energy_rmse_eV\n0,0.12\n1,0.08\n",
                encoding="utf-8",
            )
            manifest = {
                "runs": [
                    {
                        "pipeline": "md",
                        "returncode": 0,
                        "dataset_label": "candidate_ok",
                        "result_dir": str(result_dir),
                        "hamiltonian_evaluation": {"samples_compared": 2},
                    }
                ]
            }

            selected = pipeline_ui.ExperimentRunner()._select_graph2mat_top_candidates(
                manifest,
                primary_metric="low_energy_rmse_eV",
                top_percent=10.0,
                top_count=1,
            )

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["dataset_label"], "candidate_ok")
        self.assertAlmostEqual(selected[0]["_deeph_selection_score"], 0.1)

    def test_deeph_comparison_options_accept_manual_graph2mat_result_dirs(self) -> None:
        pipeline_ui = load_script_module("pipeline_ui_manual_deeph_tests", "pipeline_ui.py")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            deeph_repo = root / "DeepH-pack"
            deeph_repo.mkdir()
            deeph_python = root / "deeph-python"
            deeph_python.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
            pipeline_python = root / "pipeline-python"
            pipeline_python.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
            first = root / "run_a"
            second = root / "run_b"
            first.mkdir()
            second.mkdir()

            options = pipeline_ui.parse_deeph_comparison_options(
                {
                    "graph2mat_result_dirs": f"{first}\n{second}\n{first}",
                    "deeph_repo": str(deeph_repo),
                    "deeph_python": str(deeph_python),
                    "pipeline_python": str(pipeline_python),
                }
            )

        self.assertEqual(options["graph2mat_result_dir"], "")
        self.assertEqual(options["graph2mat_result_dirs"], [str(first), str(second)])

    def test_deeph_candidate_summary_csv_uses_top_score_rows(self) -> None:
        pipeline_ui = load_script_module("pipeline_ui_summary_deeph_tests", "pipeline_ui.py")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            best = root / "best" / "run"
            middle = root / "middle" / "run"
            low = root / "low" / "run"
            for path in (best, middle, low):
                path.mkdir(parents=True)
            summary = root / "summary.csv"
            summary.write_text(
                "method_id,run_dir,score_0_10\n"
                f"low,{low},1.0\n"
                f"best,{best},9.0\n"
                f"middle,{middle},5.0\n",
                encoding="utf-8",
            )

            selected = pipeline_ui.ExperimentRunner()._select_graph2mat_candidates_from_summary_csv(
                summary,
                top_count=2,
            )

        self.assertEqual([item["dataset_label"] for item in selected], ["best", "middle"])
        self.assertEqual([item["result_dir"] for item in selected], [str(best), str(middle)])

    def test_raw_prepare_rejects_ml_prediction_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = root / "result"
            sample_dir = root / "sample" / "0"
            sample_dir.mkdir(parents=True)
            write_split_manifest(result, "0", sample_dir)
            for suffix in (".STRUCT_OUT", ".XV", ".ORB_INDX"):
                (sample_dir / f"graphene{suffix}").write_text("x\n", encoding="utf-8")
            (sample_dir / "ML_prediction.HSX").write_text("not a reference\n", encoding="utf-8")
            samples = utils.load_split_samples(result)

            row = prepare.prepare_one_sample(
                sample=samples[0],
                raw_root=root / "out" / "raw",
                graph2mat_result_dir=result,
                system_label="graphene",
                symlink=True,
                allow_regenerate_siesta=False,
                siesta_command="siesta",
                dry_run=True,
            )

        self.assertNotEqual(row["status"], "ok")
        self.assertTrue(row["forbidden_references"])
        self.assertIn(".HSX", row["missing_suffixes"])

    def test_raw_prepare_reports_missing_deeph_required_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = root / "result"
            sample_dir = root / "sample" / "0"
            sample_dir.mkdir(parents=True)
            write_split_manifest(result, "0", sample_dir)
            samples = utils.load_split_samples(result)

            row = prepare.prepare_one_sample(
                sample=samples[0],
                raw_root=root / "out" / "raw",
                graph2mat_result_dir=result,
                system_label="graphene",
                symlink=True,
                allow_regenerate_siesta=False,
                siesta_command="siesta",
                dry_run=True,
            )

        self.assertEqual(row["status"], "missing_required_artifacts")
        self.assertEqual(set(row["missing_suffixes"]), {".HSX", ".STRUCT_OUT", ".XV", ".ORB_INDX"})

    def test_split_ordering_recreates_requested_split_sizes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            processed = root / "processed"
            samples = []
            for split, ids in {"train": ["0", "1"], "validation": ["2"], "test": ["3"]}.items():
                for sample_id in ids:
                    sample_dir = processed / sample_id
                    sample_dir.mkdir(parents=True)
                    samples.append(
                        utils.SplitSample(
                            sample=sample_id,
                            split=split,
                            sample_id=f"md_{sample_id}",
                            sample_dir=sample_dir,
                            structure_path=None,
                            hamiltonian_path=None,
                            metadata_path=None,
                            source_row={},
                        )
                    )

            manifest = utils.make_split_ordered_processed_dir(
                processed_dir=processed,
                ordered_dir=root / "ordered",
                samples=samples,
                seed=7,
                symlink=True,
                shuffled_indices=[2, 0, 3, 1],
            )

        self.assertEqual(manifest["train_size"], 2)
        self.assertEqual(manifest["validation_size"], 1)
        self.assertEqual(manifest["test_size"], 1)
        self.assertEqual(len(manifest["mapping_rows"]), 4)

    def test_orbital_config_matches_s_p_carbon_basis(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sample_dir = Path(tmp) / "0"
            sample_dir.mkdir()
            (sample_dir / "element.dat").write_text("6\n6\n", encoding="utf-8")
            (sample_dir / "orbital_types.dat").write_text("0 1\n0 1\n", encoding="utf-8")

            orbital = utils.orbital_config_from_processed_sample(sample_dir)

        self.assertEqual(len(orbital), 16)
        self.assertEqual(orbital[0], {"6 6": [0, 0]})
        self.assertEqual(orbital[-1], {"6 6": [3, 3]})

    def test_manifest_json_writer_handles_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            utils.write_json(path, {"path": Path(tmp), "items": [Path(tmp) / "x"]})
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(payload["path"], tmp)
        self.assertTrue(payload["items"][0].endswith("/x"))


if __name__ == "__main__":
    unittest.main()
