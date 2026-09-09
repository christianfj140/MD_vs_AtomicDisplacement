from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / "Comparison/scripts"
sys.path.insert(0, str(SCRIPTS))

from run_hamiltonian_derivative_predictions import deeph_runtime_settings  # noqa: E402


class DeepHEpcSnapshotScalingTests(unittest.TestCase):
    def test_force_cpu_overrides_checkpoint_cuda_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            (model / "config.ini").write_text(
                "[basic]\ndisable_cuda = false\ndevice = cuda:0\n[graph]\nradius = -1\n"
            )
            with patch.dict(os.environ, {"DEEPH_FORCE_CPU": "1"}):
                settings = deeph_runtime_settings(model, python_executable=sys.executable, command_template=None)
            self.assertTrue(settings["disable_cuda"])
            self.assertEqual(settings["device"], "cpu")
            self.assertEqual(settings["backend_fallback_reason"], "DEEPH_FORCE_CPU=1")


if __name__ == "__main__":
    unittest.main()
