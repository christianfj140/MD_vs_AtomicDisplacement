from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "Comparison" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_graphene_hbn_moire_spectral_campaign as hbn  # noqa: E402
import run_tbg_pure_graph2mat_campaign as tbg  # noqa: E402


def test_hbn_config_is_used_for_payload_and_manifest(tmp_path: Path, monkeypatch) -> None:
    base_payload = json.loads(hbn.DEFAULT_TRAINING_PAYLOAD.read_text(encoding="utf-8"))
    base_payload["phase4_marker"] = "custom payload selected"
    payload_path = tmp_path / "training.json"
    payload_path.write_text(json.dumps(base_payload), encoding="utf-8")

    config_path = tmp_path / "campaign.json"
    config = json.loads(hbn.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    config["training_payload"] = str(payload_path)
    config_path.write_text(json.dumps(config), encoding="utf-8")

    loaded = hbn.load_config(config_path)
    payload = hbn.training_payload(loaded, tmp_path / "results", 30)
    assert payload["phase4_marker"] == "custom payload selected"
    monkeypatch.setattr(hbn, "environment_inventory", lambda _config: {})
    monkeypatch.setattr(hbn, "legacy_inventory", lambda _root: {})
    manifest = hbn.plan(loaded, tmp_path / "plan", config_path)
    assert manifest["config"] == str(config_path)
    assert manifest["config_sha256"] == hbn.sha256(config_path)


def test_tbg_cli_defaults_and_input_overrides(tmp_path: Path, monkeypatch) -> None:
    defaults = tbg.parse_args([])
    assert defaults.training_size == 474
    assert defaults.base_payload == tbg.DEFAULT_BASE_PAYLOAD
    assert defaults.output_root == tbg.DEFAULT_ROOT
    assert defaults.dataset is None

    dataset = tmp_path / "dataset"
    base_payload = tmp_path / "payload.json"
    output = tmp_path / "output"
    base_payload.write_text(tbg.DEFAULT_BASE_PAYLOAD.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(tbg, "ROOT", tbg.DEFAULT_ROOT)
    monkeypatch.setattr(tbg, "DATASET", tbg.DEFAULT_DATASET)
    monkeypatch.setattr(tbg, "BASE_PAYLOAD", tbg.DEFAULT_BASE_PAYLOAD)

    tbg.configure_campaign_paths(defaults)
    assert tbg.ROOT == tbg.DEFAULT_ROOT
    assert tbg.DATASET == tbg.DEFAULT_DATASET
    assert tbg.BASE_PAYLOAD == tbg.DEFAULT_BASE_PAYLOAD

    args = tbg.parse_args([
        "--training-size", "12",
        "--dataset", str(dataset),
        "--base-payload", str(base_payload),
        "--output-root", str(output),
    ])
    tbg.configure_campaign_paths(args)
    payload_path = tbg.training_payload(args.training_size)
    payload = json.loads(payload_path.read_text(encoding="utf-8"))

    assert tbg.DATASET == dataset
    assert payload["dataset_root"] == str(dataset)
    assert payload["output_root"] == str(output / "training" / "n12")
    assert payload["run_id"] == "tbg_pure_n12"
    assert payload["training_sweep"]["manual_runs"][0]["id"] == "g2m_tbg_n12_seed0"
