import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "Comparison" / "scripts"
sys.path.insert(0, str(SCRIPTS))

from run_graphene_hbn_moire_spectral_campaign import aggregate, write_json  # noqa: E402


def test_aggregate_publishes_completed_tier_before_campaign_finishes(tmp_path):
    run_root = tmp_path / "spectra/graph2mat/n30/seed0"
    write_json(
        run_root / "solver_manifest.json",
        {
            "status": "resource_blocked",
            "bands_status": "resource_blocked",
            "model": "graph2mat",
            "training_size": 30,
            "seed": 0,
        },
    )
    write_json(
        run_root / "tier_a/solver_manifest.json",
        {
            "status": "completed",
            "bands": [
                {
                    "k_index": 0,
                    "band_index": 0,
                    "k_distance": 0.0,
                    "energy_aligned_eV": -0.01,
                },
                {
                    "k_index": 0,
                    "band_index": 1,
                    "k_distance": 0.0,
                    "energy_aligned_eV": 0.02,
                },
            ],
        },
    )

    summary = aggregate(
        {"campaign_kind": "test", "target_contract": {"reference_hamiltonian": False}},
        tmp_path,
    )

    assert summary["spectra"][0]["status"] == "completed"
    assert summary["spectra"][0]["visible_band_tier"] == "tier_a"
    assert len(summary["spectra"][0]["bands"]) == 2
