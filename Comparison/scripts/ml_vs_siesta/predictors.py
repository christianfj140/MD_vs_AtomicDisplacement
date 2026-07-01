"""Common predictor interface + thin Graph2Mat / DeepH adapters.

These adapters never train and never mutate model internals. They wrap an
existing inference callable (so real Graph2Mat/DeepH runners can be plugged in
without changing this package). When no callable is wired, ``predict`` raises a
clear :class:`NotImplementedError` pointing at the existing runner scripts.
"""

from __future__ import annotations

from typing import Any, Callable

from .matrices import MatrixData

PredictFn = Callable[[Any, list[str]], dict[str, MatrixData]]


class MatrixPredictor:
    """Abstract matrix predictor.

    Subclasses implement :meth:`predict`, returning one :class:`MatrixData` per
    requested target.
    """

    name: str = "predictor"

    def predict(self, structure, targets: list[str]) -> dict[str, MatrixData]:
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}(name={self.name!r})"


class FunctionMatrixPredictor(MatrixPredictor):
    """Wrap any ``callable(structure, targets) -> dict[str, MatrixData]``.

    Handy for tests, dry-runs and for adapting already-loaded models.
    """

    def __init__(self, predict_fn: PredictFn, *, name: str = "function"):
        self._predict_fn = predict_fn
        self.name = name

    def predict(self, structure, targets: list[str]) -> dict[str, MatrixData]:
        result = self._predict_fn(structure, list(targets))
        if not isinstance(result, dict):
            raise TypeError(
                f"{self.name} predict_fn must return a dict[str, MatrixData]."
            )
        return result


class _WrappedModelPredictor(MatrixPredictor):
    """Shared base for Graph2Mat/DeepH adapters wrapping an inference callable."""

    runner_hint = ""

    def __init__(
        self,
        *,
        predict_fn: PredictFn | None = None,
        checkpoint: Any = None,
        name: str | None = None,
        **options: Any,
    ):
        self._predict_fn = predict_fn
        self.checkpoint = checkpoint
        self.options = options
        if name is not None:
            self.name = name

    def predict(self, structure, targets: list[str]) -> dict[str, MatrixData]:
        if self._predict_fn is None:
            raise NotImplementedError(
                f"{self.name} inference is not wired in this thin adapter. "
                f"Pass predict_fn=... (a callable(structure, targets) -> "
                f"dict[str, MatrixData]) or run inference via {self.runner_hint}."
            )
        return self._predict_fn(structure, list(targets))


class Graph2MatPredictor(_WrappedModelPredictor):
    """Thin adapter around an existing Graph2Mat model / inference callable."""

    name = "graph2mat"
    runner_hint = (
        "Comparison/scripts/predict_model_on_dataset.py / "
        "run_hamiltonian_derivative_predictions.py"
    )


class DeepHPredictor(_WrappedModelPredictor):
    """Thin adapter around an existing DeepH model / inference callable.

    DeepH inference/base-conversion lives in ``deeph_prediction_adapter.py``.
    Without a wired ``predict_fn`` this raises ``NotImplementedError``.
    """

    name = "deeph"
    runner_hint = "Comparison/scripts/deeph_prediction_adapter.py"


_PREDICTOR_CLASSES = {
    "graph2mat": Graph2MatPredictor,
    "deeph": DeepHPredictor,
}


def build_predictor(model: str, **kwargs: Any) -> MatrixPredictor:
    """Instantiate a predictor by model name (``graph2mat`` / ``deeph``)."""
    key = str(model).strip().lower()
    if key not in _PREDICTOR_CLASSES:
        raise ValueError(
            f"Unknown model {model!r}; available: {sorted(_PREDICTOR_CLASSES)}."
        )
    return _PREDICTOR_CLASSES[key](**kwargs)
