from __future__ import annotations

from pathlib import Path
from unittest import mock

import Comparison.scripts.run_mixing_e2e_payload_once as e2e
from Comparison.scripts.run_mixing_e2e_payload_once import expected_paths_present, resolve_repo_path


def test_resolve_repo_path_maps_relative_paths_inside_repo() -> None:
    resolved = resolve_repo_path("Comparison/config/ml_vs_siesta_mixing_e2e_20_50_80_payload.json")
    assert resolved.is_absolute()
    assert resolved.name == "ml_vs_siesta_mixing_e2e_20_50_80_payload.json"


def test_expected_paths_present_requires_all_paths(tmp_path: Path) -> None:
    existing = tmp_path / "exists"
    existing.mkdir()
    missing = tmp_path / "missing"
    assert expected_paths_present([existing]) is True
    assert expected_paths_present([existing, missing]) is False


def test_stream_runner_logs_prints_internal_runner_lines(capsys) -> None:
    class FakeRunner:
        def logs(self, *, since: int, limit: int | None):
            assert since == 0
            assert limit is None
            return {"offset": 2, "lines": ["Epoch 1/750\n", "[DeepH] loss=0.1\n"]}

    assert e2e._stream_runner_logs(FakeRunner(), 0) == 2
    assert "Epoch 1/750" in capsys.readouterr().out


def _base_payload(**extra: object) -> dict:
    payload = {
        "small": {"20": "/tmp/small20"},
        "large": {"20": "/tmp/large20"},
        "sizes": [20],
        "modes": ["add"],
        "ratios": [0.0, 0.5],
        "seed": 0,
        "models": ["graph2mat"],
    }
    payload.update(extra)
    return payload


def _run_and_capture_mixing_kwargs(payload: dict, tmp_path: Path) -> dict[str, object]:
    """Run _run_mixing_payload with both runner paths mocked; return captured kwargs."""
    captured: dict[str, object] = {}

    def fake_parallel(*args, **kwargs):
        captured["parallel"] = kwargs
        return {"n_permutations": 0, "permutations": []}

    def fake_sweep(*args, **kwargs):
        captured["sweep"] = kwargs
        return {"n_permutations": 0, "permutations": []}

    fake_mvs = mock.Mock()
    fake_mvs.run_mixing_sweep = fake_sweep
    with mock.patch.object(e2e.ui, "_run_mixing_sweep_parallel", fake_parallel), \
            mock.patch.object(e2e.ui, "_ml_vs_siesta_module", return_value=fake_mvs):
        e2e._run_mixing_payload(payload, tmp_path)
    return captured


def test_mixing_payload_defaults_to_fixed_common_test(tmp_path: Path) -> None:
    # Train path (parallel runner).
    captured = _run_and_capture_mixing_kwargs(_base_payload(action="train"), tmp_path)
    assert captured["parallel"]["split_policy"] == "fixed_common_test"
    # Preview path (library sweep).
    captured = _run_and_capture_mixing_kwargs(_base_payload(action="preview"), tmp_path)
    assert captured["sweep"]["split_policy"] == "fixed_common_test"


def test_mixing_payload_explicit_resplit_combined_is_respected(tmp_path: Path) -> None:
    payload = _base_payload(action="train", split_policy="resplit_combined")
    captured = _run_and_capture_mixing_kwargs(payload, tmp_path)
    assert captured["parallel"]["split_policy"] == "resplit_combined"


def test_mixing_payload_forwards_split_fractions_and_gap(tmp_path: Path) -> None:
    payload = _base_payload(
        action="train",
        split_policy="blocked_stratified_gap",
        split_fractions=[0.8, 0.1, 0.1],
        temporal_gap=1,
    )
    captured = _run_and_capture_mixing_kwargs(payload, tmp_path)
    assert captured["parallel"]["split_fractions"] == (0.8, 0.1, 0.1)
    assert captured["parallel"]["temporal_gap"] == 1


def test_mixing_payload_hyperparams_reach_parallel_runner(tmp_path: Path) -> None:
    hyperparams = {"graph2mat": {"hidden_irreps": "64x0e + 64x1o + 64x2e + 64x3o", "optim_lr": 0.0018}}
    payload = _base_payload(action="train", hyperparams=hyperparams)
    captured: dict[str, object] = {}

    def fake_parallel(*args, **kwargs):
        captured["hyperparams"] = kwargs["hyperparams"]
        return {"n_permutations": 0, "permutations": []}

    with mock.patch.object(e2e.ui, "_run_mixing_sweep_parallel", fake_parallel):
        e2e._run_mixing_payload(payload, tmp_path)
    assert captured["hyperparams"] == hyperparams


def test_mixing_payload_without_hyperparams_passes_none(tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_parallel(*args, **kwargs):
        captured["hyperparams"] = kwargs["hyperparams"]
        return {"n_permutations": 0, "permutations": []}

    with mock.patch.object(e2e.ui, "_run_mixing_sweep_parallel", fake_parallel):
        e2e._run_mixing_payload(_base_payload(action="train"), tmp_path)
    assert captured["hyperparams"] is None


def test_production_train_payload_declares_fixed_common_test() -> None:
    import json

    path = resolve_repo_path(
        "Comparison/config/ml_vs_siesta_mixing_sweep_100_500_train_payload.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["split_policy"] == "fixed_common_test"
    # Audit I2/I5: the heterogeneous ghost pool is declared explicitly and the
    # sweep runs >= 3 seeds so the curves carry error bars.
    assert payload["confirm_ghost_species_exemption"] is True
    assert len(payload["seeds"]) >= 3


def test_paper_ready_20_500_payload_declares_single_pass_with_anchor_hyperparams() -> None:
    import json

    path = resolve_repo_path(
        "Comparison/config/ml_vs_siesta_mixing_sweep_20_500_paper_ready_train_payload.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["split_policy"] == "fixed_common_test"
    assert payload["sizes"] == [20, 50, 100, 150, 200, 300, 400, 500]
    # Single training pass, no seed replication (not publication-grade, see notes).
    assert "seeds" not in payload
    assert payload["seed"] == 0
    assert payload["epochs"] == 750
    # Hyperparams must match the paper-ready T600 anchors verbatim, not the
    # small MD/pipeline_config.yaml defaults (10x0e+10x1o+10x2e, batch_size=10).
    anchor_path = resolve_repo_path(
        "Comparison/config/mixing_sanity_size500_add_r0p000_paper_ready_anchors_payload.json"
    )
    anchor = json.loads(anchor_path.read_text(encoding="utf-8"))
    anchor_runs = {r["model"]: r["overrides"] for r in anchor["training_sweep"]["manual_runs"]}
    assert payload["hyperparams"]["graph2mat"]["hidden_irreps"] == anchor_runs["graph2mat"]["hidden_irreps"]
    assert payload["hyperparams"]["graph2mat"]["optim_lr"] == anchor_runs["graph2mat"]["optim_lr"]
    assert payload["hyperparams"]["deeph"]["learning_rate"] == anchor_runs["deeph"]["learning_rate"]
    assert payload["hyperparams"]["deeph"]["atom_fea_len"] == anchor_runs["deeph"]["atom_fea_len"]


def test_multi_seed_payload_runs_each_seed_and_aggregates(tmp_path: Path) -> None:
    payload = _base_payload(action="train", seeds=[0, 1, 2])
    calls: list[tuple[int, str]] = []

    def fake_single(single_payload: dict, output_root: Path) -> dict:
        seed = int(single_payload["seed"])
        calls.append((seed, str(output_root)))
        assert "seeds" not in single_payload
        return {
            "records": [
                {"size": 20, "mode": "add", "ratio": 0.0, "seed": seed,
                 "total_size": 20, "model": "graph2mat", "h_mae_eV": 0.05 + 0.01 * seed},
            ]
        }

    with mock.patch.object(e2e, "_run_mixing_payload", side_effect=fake_single):
        summary = e2e._run_mixing_payload_seeds(payload, tmp_path)

    assert [seed for seed, _root in calls] == [0, 1, 2]
    assert all(root.endswith(f"seed{seed}") for seed, root in calls)
    assert summary["n_seeds"] == 3
    assert summary["exploratory"] is False
    assert len(summary["records"]) == 3
    assert (tmp_path / "mae_vs_size.json").is_file()


def test_payload_without_seeds_runs_single_pass(tmp_path: Path) -> None:
    payload = _base_payload(action="preview")
    with mock.patch.object(e2e, "_run_mixing_payload", return_value={"records": []}) as single:
        summary = e2e._run_mixing_payload_seeds(payload, tmp_path)
    single.assert_called_once_with(payload, tmp_path)
    assert summary == {"records": []}
