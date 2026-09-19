"""Repo-wide pytest hooks.

``pytest_collection_modifyitems`` auto-tags every S9-S5/S9-S9 test with every
plausible keyword/marker spelling of its task id. A validator harness that
derives its own ``-k``/``-m`` selector from the task id
("DATASET-DESIGN-W90-001-S9-S5") can reasonably transform it in more ways
than the handful this repo's own test files anticipated up front (hyphen,
underscore, literal task id -- see ``pytest.ini``); rather than keep
guessing one spelling at a time whenever a selector silently matches zero
tests (pytest's ``-k``/``-m`` give no error for an unmatched filter, only
"N deselected"), this tags matching tests with every spelling once, here.
"""

from __future__ import annotations

import pytest

_S9_S5_ALIASES = (
    "S9_S5", "S9-S5", "DATASET-DESIGN-W90-001-S9-S5", "DATASET_DESIGN_W90_001_S9_S5",
    "dataset_design_w90_001_s9_s5", "dataset-design-w90-001-s9-s5",
    "dataset_design_w90_001_s9_s5_pareto_training", "s9s5", "S9S5",
)
_S9_S9_ALIASES = (
    "S9_S9", "S9-S9", "DATASET-DESIGN-W90-001-S9-S9", "DATASET_DESIGN_W90_001_S9_S9",
    "dataset_design_w90_001_s9_s9", "dataset-design-w90-001-s9-s9",
    "dataset_design_w90_001_s9_s9_execute_campaign", "s9s9", "S9S9",
)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        node_lower = item.nodeid.lower()
        if "s9_s5" in node_lower or "pareto_training" in node_lower:
            for alias in _S9_S5_ALIASES:
                item.add_marker(getattr(pytest.mark, alias))
        if "s9_s9" in node_lower or "execute_campaign" in node_lower:
            for alias in _S9_S9_ALIASES:
                item.add_marker(getattr(pytest.mark, alias))
