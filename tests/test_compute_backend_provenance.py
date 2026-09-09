from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
import json


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "Comparison" / "scripts" / "audit_epc_backends.py"
SPEC = importlib.util.spec_from_file_location("audit_epc_backends", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class BackendCandidateDetectionTests(unittest.TestCase):
    def test_ternary_fallback_with_no_nearby_marker_is_flagged_silent(self) -> None:
        text = (
            "def pick(is_cuda_ok):\n"
            "    accelerator = 'gpu' if is_cuda_ok else 'cpu'\n"
            "    return accelerator\n"
        )
        candidates = MODULE.find_candidates(text)
        by_line = {c["cuda_line"]: c for c in candidates}
        self.assertIn(2, by_line)
        self.assertFalse(by_line[2]["explicit"])

    def test_fallback_next_to_a_record_call_is_not_flagged_silent(self) -> None:
        text = (
            "def resolve(requested):\n"
            "    if not torch.cuda.is_available():\n"
            "        return move_to_cpu(model), JvpBackendRecord(\n"
            "            requested, 'cpu', 'failed', 'unavailable'\n"
            "        )\n"
        )
        candidates = MODULE.find_candidates(text)
        by_line = {c["cuda_line"]: c for c in candidates}
        self.assertIn(2, by_line)
        self.assertTrue(by_line[2]["explicit"])

    def test_fallback_next_to_effective_backend_field_is_not_flagged_silent(self) -> None:
        text = (
            "backend = {\n"
            "    'requested_backend': 'cuda',\n"
            "    'effective_backend': 'cpu',\n"
            "    'backend_fallback_reason': 'no GPU visible',\n"
            "}\n"
        )
        candidates = MODULE.find_candidates(text)
        self.assertEqual(len(candidates), 1)
        self.assertTrue(candidates[0]["explicit"])

    def test_cuda_reference_with_no_cpu_fallback_is_not_a_candidate(self) -> None:
        text = "device = torch.device('cuda:0')\nmodel.to(device)\n"
        self.assertEqual(MODULE.find_candidates(text), [])

    def test_cpu_literal_far_beyond_the_fallback_window_is_not_paired(self) -> None:
        text = "x = 'cuda'\n" + "\n".join(f"line{i}" for i in range(MODULE.FALLBACK_WINDOW + 5)) + "\ny = 'cpu'\n"
        self.assertEqual(MODULE.find_candidates(text), [])

    def test_backend_enum_and_prose_are_not_fallbacks(self) -> None:
        self.assertEqual(MODULE.find_candidates('VALID_BACKENDS = ("cpu", "cuda")\n'), [])
        self.assertEqual(
            MODULE.find_candidates('"""Compare CPU vs CUDA where both actually ran."""\n'),
            [],
        )

class ScanAndReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def write(self, relative: str, content: str) -> None:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def test_build_report_finds_silent_and_explicit_sites_across_files(self) -> None:
        self.write("silent.py", "device = 'gpu' if cuda_ok else 'cpu'\n")
        self.write(
            "explicit.py",
            "device = 'gpu' if cuda_ok else 'cpu'  # logger.warning('cpu fallback: no GPU visible')\n",
        )
        report = MODULE.build_report(roots=[self.root])
        self.assertEqual(report["summary"]["silent_count"], 1)
        self.assertEqual(report["summary"]["explicit_count"], 1)
        self.assertFalse(report["summary"]["no_silent_fallbacks_found"])
        self.assertEqual(report["files_scanned"], 2)

    def test_pycache_directories_are_skipped(self) -> None:
        self.write("__pycache__/junk.py", "device = 'gpu' if cuda_ok else 'cpu'\n")
        report = MODULE.build_report(roots=[self.root])
        self.assertEqual(report["files_scanned"], 0)

    def test_clean_tree_reports_no_silent_fallbacks(self) -> None:
        self.write("clean.py", "print('no cuda mentions here')\n")
        report = MODULE.build_report(roots=[self.root])
        self.assertTrue(report["summary"]["no_silent_fallbacks_found"])

    def test_runtime_manifest_requires_device_and_precision(self) -> None:
        path = self.root / "runtime.json"
        path.write_text(json.dumps({"schema": "graph2mat_jvp_runtime_preflight_v1", "generated_at": "2026-09-06T00:00:00Z", "requested_backend": "cuda", "effective_backend": "cuda", "backend_preflight": "passed", "backend_fallback_reason": None, "device": "cuda:0", "precision": "torch.float64"}), encoding="utf-8")
        report = MODULE.build_report(roots=[self.root], runtime_manifests=[path])
        self.assertTrue(report["summary"]["runtime_evidence_valid"])
        del_payload = json.loads(path.read_text(encoding="utf-8"))
        del del_payload["precision"]
        path.write_text(json.dumps(del_payload), encoding="utf-8")
        report = MODULE.build_report(roots=[self.root], runtime_manifests=[path])
        self.assertFalse(report["summary"]["runtime_evidence_valid"])

    def test_main_fail_on_silent_cpu_flag_controls_exit_code(self) -> None:
        self.write("silent.py", "device = 'gpu' if cuda_ok else 'cpu'\n")
        output = self.root / "report.json"
        exit_default = MODULE.main(["--scan-root", str(self.root), "--output", str(output)])
        self.assertEqual(exit_default, 0)
        exit_strict = MODULE.main(
            ["--scan-root", str(self.root), "--output", str(output), "--fail-on-silent-cpu"]
        )
        self.assertEqual(exit_strict, 1)


class RealRepositoryScanTests(unittest.TestCase):
    """Smoke-tests the scanner against the real repository, not a fixture.

    This does not assert a specific silent/explicit count (the codebase keeps
    changing); it only asserts the scan runs cleanly over real files and that
    the known-good explicit pattern in graph2mat_autograd_derivatives.py is
    recognised, so a regression in the marker regex would be caught here.
    """

    def test_scans_the_real_scripts_directory_without_error(self) -> None:
        report = MODULE.build_report()
        self.assertGreater(report["files_scanned"], 50)
        self.assertIn("summary", report)

    def test_known_explicit_jvp_backend_record_pattern_is_recognised(self) -> None:
        path = REPO_ROOT / "Comparison" / "scripts" / "graph2mat_autograd_derivatives.py"
        text = path.read_text(encoding="utf-8")
        candidates = MODULE.find_candidates(text)
        explicit_lines = {c["cuda_line"] for c in candidates if c["explicit"]}
        self.assertTrue(explicit_lines, "expected at least one explicit fallback site to be recognised")

    def test_real_repository_has_no_silent_cpu_fallback_candidates(self) -> None:
        report = MODULE.build_report()
        self.assertEqual(report["silent_cpu_fallback_candidates"], [])

if __name__ == "__main__":
    unittest.main()
