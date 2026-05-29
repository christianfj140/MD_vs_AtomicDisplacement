import json
import sys
import tempfile
import time
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "Comparison" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from g2m_deeph_telemetry import (  # noqa: E402
    GpuTelemetryMonitor,
    ProcResourceMonitor,
    TELEMETRY_SCHEMA,
    best_validation_cost_from_events,
    classify_failure,
    compute_gpu_hours,
    compute_throughput,
    optimizer_update_accounting,
    parse_proc_meminfo,
    parse_proc_status,
    summarize_run_telemetry,
    write_telemetry,
)
from g2m_deeph_rank_runs import row_from_training_record  # noqa: E402


class Graph2MatDeepHTelemetryTests(unittest.TestCase):
    def test_gpu_hours_calculation(self) -> None:
        self.assertAlmostEqual(compute_gpu_hours(7200, 2), 4.0)
        self.assertIsNone(compute_gpu_hours(None, 2))
        self.assertIsNone(compute_gpu_hours(10, 0))

    def test_throughput_calculation(self) -> None:
        values = compute_throughput(samples=100, matrix_blocks=500, elapsed_seconds=20)

        self.assertEqual(values["samples_per_second"], 5.0)
        self.assertEqual(values["matrix_blocks_per_second"], 25.0)
        self.assertEqual(
            compute_throughput(samples=100, matrix_blocks=500, elapsed_seconds=0),
            {"samples_per_second": None, "matrix_blocks_per_second": None},
        )

    def test_optimizer_update_accounting_flags_low_update_runs(self) -> None:
        accounting = optimizer_update_accounting(
            train_samples=100,
            batch_size=64,
            max_epochs=10,
            gradient_accumulation=1,
        )

        self.assertEqual(accounting["steps_per_epoch"], 2)
        self.assertEqual(accounting["total_optimizer_updates"], 20)
        self.assertTrue(accounting["possible_undertraining"])
        self.assertIn("below paper-ready threshold", accounting["warnings"][0])

        adequate = optimizer_update_accounting(train_samples=1000, batch_size=32, max_epochs=100)
        self.assertFalse(adequate["possible_undertraining"])
        self.assertEqual(adequate["optimizer_updates_per_epoch"], 32)

    def test_best_validation_cost_extraction(self) -> None:
        events = [
            {"epoch": 1, "step": 10, "value": 0.5, "wall_time": 100.0},
            {"epoch": 2, "step": 20, "value": 0.2, "wall_time": 130.0},
            {"epoch": 3, "step": 30, "value": 0.3, "wall_time": 150.0},
        ]

        best = best_validation_cost_from_events(events, mode="min", run_started_at=90.0)

        self.assertEqual(best["best_validation_epoch"], 2)
        self.assertEqual(best["best_validation_step"], 20)
        self.assertAlmostEqual(best["wall_clock_seconds_to_best_validation"], 40.0)

    def test_failure_classification_cuda_oom_and_generic_nonzero(self) -> None:
        cuda = classify_failure(returncode=1, output_excerpt="RuntimeError: CUDA out of memory")
        generic = classify_failure(returncode=2, output_excerpt="validation failed")
        missing = classify_failure(returncode=127, output_excerpt="/bin/sh: deeph-train: command not found")

        self.assertEqual(cuda["failure_category"], "cuda_oom_detected")
        self.assertIn("CUDA out of memory", cuda["failure_evidence_excerpt"])
        self.assertEqual(generic["failure_category"], "nonzero_exit")
        self.assertEqual(missing["failure_category"], "missing_dependency")

    def test_proc_parsers_work_on_synthetic_text(self) -> None:
        status = parse_proc_status("Name:\tpython\nVmRSS:\t2048 kB\nVmHWM:\t4096 kB\n")
        meminfo = parse_proc_meminfo("MemTotal: 1048576 kB\nMemAvailable: 524288 kB\n")

        self.assertEqual(status["VmRSS_mb"], 2.0)
        self.assertEqual(status["VmHWM_mb"], 4.0)
        self.assertEqual(meminfo["MemTotal_mb"], 1024.0)
        self.assertEqual(meminfo["MemAvailable_mb"], 512.0)

    def test_gpu_monitor_uses_mocked_process_memory(self) -> None:
        calls = []

        def query():
            calls.append(True)
            return ([{"pid": 123, "used_memory_mb": 512, "gpu_uuid": "gpu0"}], None)

        monitor = GpuTelemetryMonitor(
            poll_interval_seconds=0.01,
            query_processes=query,
            pid_tree=lambda root: {root},
        )
        monitor.start(123)
        time.sleep(0.04)
        telemetry = monitor.stop()

        self.assertTrue(calls)
        self.assertEqual(telemetry["peak_gpu_memory_mb"], 512)
        self.assertEqual(telemetry["observed_gpu_count"], 1)
        self.assertGreater(telemetry["gpu_active_seconds"], 0)

    def test_proc_monitor_unavailable_path_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            monitor = ProcResourceMonitor(
                poll_interval_seconds=0.01,
                proc_root=Path(tmp) / "missing_proc",
                pid_tree=lambda root: {root},
            )
            monitor.start(123456)
            time.sleep(0.02)
            telemetry = monitor.stop()

        self.assertIsNone(telemetry["peak_rss_mb"])
        self.assertTrue(any("unavailable" in warning for warning in telemetry["warnings"]))

    def test_unavailable_telemetry_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            split = root / "frozen_split_manifest.json"
            split.write_text(json.dumps({"split_counts": {"train": 8, "validation": 2}}), encoding="utf-8")

            telemetry = summarize_run_telemetry(
                model="graph2mat",
                run_root=root,
                training_dir=root / "training",
                frozen_split_manifest_path=split,
                train_run={"elapsed_seconds": 10, "telemetry": {"warnings": ["nvidia-smi unavailable"]}},
                optimizer_accounting=optimizer_update_accounting(train_samples=8, batch_size=4, max_epochs=2),
            )

            self.assertEqual(telemetry["schema"], TELEMETRY_SCHEMA)
            self.assertEqual(telemetry["telemetry_status"], "partial")
            self.assertIsNone(telemetry["gpu_hours_total"])
            self.assertIn("peak_rss_mb", telemetry)
            self.assertIn("cpu_time_seconds_total", telemetry)
            self.assertEqual(telemetry["total_optimizer_updates"], 4)
            self.assertIn("gpu_hours_total unavailable", telemetry["telemetry_warnings"])
            self.assertIn("train: nvidia-smi unavailable", telemetry["telemetry_warnings"])

    def test_artifact_serialization_and_ranking_old_artifact_compatibility(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            telemetry_path = root / "telemetry" / "graph2mat.json"
            write_telemetry(
                telemetry_path,
                {
                    "schema": TELEMETRY_SCHEMA,
                    "telemetry_status": "partial",
                    "gpu_hours_total": 1.5,
                    "peak_gpu_memory_mb": 2048,
                },
            )
            record = {
                "model": "graph2mat",
                "config_id": "cfg",
                "status": "completed",
                "run_root": str(root),
                "telemetry_path": str(telemetry_path),
            }

            row = row_from_training_record(record)

            self.assertEqual(row["telemetry_status"], "partial")
            self.assertEqual(row["gpu_hours_total"], 1.5)

            old_row = row_from_training_record(
                {"model": "graph2mat", "config_id": "old", "status": "completed", "run_root": str(root / "old")}
            )
            self.assertEqual(old_row["telemetry_status"], "unavailable")
            self.assertIn("telemetry unavailable", old_row["telemetry_warnings"])


if __name__ == "__main__":
    unittest.main()
