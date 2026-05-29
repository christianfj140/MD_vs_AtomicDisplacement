import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "Comparison" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from g2m_deeph_early_stopping import (  # noqa: E402
    DeepHEarlyStoppingObserver,
    EarlyStoppingPolicy,
    EarlyStoppingTracker,
    parse_deeph_validation_line,
    parse_early_stopping_policy,
)
from g2m_deeph_runner import _apply_common_early_stopping_to_graph2mat_config  # noqa: E402


class Graph2MatDeepHEarlyStoppingTests(unittest.TestCase):
    def policy(self, **overrides) -> EarlyStoppingPolicy:
        values = {
            "validation_metric_name": "val_loss",
            "metric_mode": "min",
            "patience": 2,
            "min_delta": 0.01,
            "max_epochs": 10,
        }
        values.update(overrides)
        return EarlyStoppingPolicy(**values)

    def test_improvement_resets_patience(self) -> None:
        tracker = EarlyStoppingTracker(self.policy())

        self.assertFalse(tracker.update(epoch=1, value=1.0))
        self.assertFalse(tracker.update(epoch=2, value=0.995))
        self.assertEqual(tracker.checks_since_improvement, 1)
        self.assertFalse(tracker.update(epoch=3, value=0.9))
        self.assertEqual(tracker.checks_since_improvement, 0)
        self.assertEqual(tracker.best_epoch, 3)

    def test_no_improvement_triggers_early_stopping(self) -> None:
        tracker = EarlyStoppingTracker(self.policy(patience=2, min_delta=0.0))

        tracker.update(epoch=1, value=1.0)
        self.assertFalse(tracker.update(epoch=2, value=1.1))
        self.assertTrue(tracker.update(epoch=3, value=1.2))

        metadata = tracker.metadata()
        self.assertEqual(metadata["stop_reason"], "early_stopping")
        self.assertEqual(metadata["best_epoch"], 1)

    def test_min_delta_behavior(self) -> None:
        tracker = EarlyStoppingTracker(self.policy(patience=1, min_delta=0.05))

        tracker.update(epoch=1, value=1.0)
        stopped = tracker.update(epoch=2, value=0.96)

        self.assertTrue(stopped)
        self.assertEqual(tracker.best_epoch, 1)

    def test_max_epochs_behavior(self) -> None:
        tracker = EarlyStoppingTracker(self.policy(patience=10, min_delta=0.0, max_epochs=3))
        tracker.update(epoch=1, value=1.0)
        tracker.update(epoch=2, value=0.9)
        tracker.update(epoch=3, value=0.8)

        self.assertEqual(tracker.metadata()["stop_reason"], "max_epochs")

    def test_missing_validation_metric_fails_closed(self) -> None:
        tracker = EarlyStoppingTracker(self.policy())

        with self.assertRaisesRegex(RuntimeError, "Missing validation metric"):
            tracker.metadata()

    def test_selection_ignores_test_metrics(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "must not reference test metrics"):
            parse_early_stopping_policy(
                {
                    "early_stopping": {
                        "metric": "test_loss",
                        "mode": "min",
                        "patience": 2,
                        "min_delta": 0.0,
                        "max_epochs": 10,
                    }
                }
            )

    def test_deeph_observer_stops_after_patience(self) -> None:
        observer = DeepHEarlyStoppingObserver(self.policy(patience=1, min_delta=0.0))

        self.assertIsNone(observer("Epoch #0 \t| Val loss: 1.00000000 \t| Best val loss: 1.00000000."))
        reason = observer("Epoch #1 \t| Val loss: 1.10000000 \t| Best val loss: 1.00000000.")

        self.assertIn("early_stopping", str(reason))
        metadata = observer.metadata()
        self.assertEqual(metadata["stop_reason"], "early_stopping")
        self.assertEqual(metadata["best_validation_value"], 1.0)

    def test_deeph_validation_line_parser(self) -> None:
        event = parse_deeph_validation_line(
            "Epoch #12 \t| Learning rate: 1.00e-03 \t| Val loss: 0.12345678 \t| Best val loss: 0.2."
        )

        self.assertIsNotNone(event)
        self.assertEqual(event.epoch, 12)
        self.assertAlmostEqual(event.value, 0.12345678)

    def test_graph2mat_config_uses_same_policy(self) -> None:
        config = {"training": {"trainer": {"callbacks": [{"class_path": "ModelCheckpoint", "init_args": {}}]}}}
        metadata = _apply_common_early_stopping_to_graph2mat_config(
            config,
            {
                "early_stopping": {
                    "metric": "val_loss",
                    "mode": "min",
                    "patience": 5,
                    "min_delta": 0.001,
                    "max_epochs": 50,
                }
            },
        )

        callbacks = config["training"]["trainer"]["callbacks"]
        early = next(item for item in callbacks if item["class_path"] == "EarlyStopping")
        self.assertEqual(config["training"]["trainer"]["max_epochs"], 50)
        self.assertEqual(early["init_args"]["monitor"], "val_loss")
        self.assertEqual(early["init_args"]["patience"], 5)
        self.assertEqual(metadata["max_epochs"], 50)


if __name__ == "__main__":
    unittest.main()
