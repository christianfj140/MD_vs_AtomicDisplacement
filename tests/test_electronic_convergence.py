from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "shared"))

from electronic_convergence import convergence_status, pooling_errors  # noqa: E402


def evidence(label: str, equivalence_class: str = "graphene-converged") -> dict:
    studies = {
        name: {
            "converged": True,
            "points": [
                {"parameter": value, "observable": observable}
                for value, observable in ((1, -1.0), (2, -1.01), (3, -1.011))
            ],
        }
        for name in ("mesh_cutoff", "kpoint_density", "scf_tolerance")
    }
    return {
        "schema": "electronic_convergence_evidence_v1",
        "material_label": label,
        "predeclared": True,
        "observable": {"name": "energy_per_atom", "unit": "eV/atom"},
        "tolerance": {
            "value": 0.002,
            "unit": "eV/atom",
            "justification": "predeclared material-specific accuracy target",
        },
        "basis_convergence_required": False,
        "studies": studies,
        "converged": True,
        "equivalence_class_hash": equivalence_class,
    }


def test_missing_evidence_limits_exact_configuration_and_blocks_pooling(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()

    assert convergence_status(first)["status"] == "exact_configuration_only"
    assert pooling_errors([(first, "graphene"), (second, "graphene")])


def test_matching_valid_evidence_allows_pooling(tmp_path: Path) -> None:
    datasets = []
    for name in ("primitive", "supercell"):
        root = tmp_path / name
        root.mkdir()
        (root / "electronic_convergence.json").write_text(
            json.dumps(evidence("graphene")) + "\n",
            encoding="utf-8",
        )
        datasets.append((root, "graphene"))

    assert pooling_errors(datasets) == []


def test_raw_equal_kpoint_count_cannot_replace_convergence_evidence(tmp_path: Path) -> None:
    for name in ("primitive", "supercell"):
        root = tmp_path / name
        root.mkdir()
        (root / "RUN.fdf").write_text(
            "%block kgrid_Monkhorst_Pack\n8 0 0 0\n0 8 0 0\n0 0 1 0\n%endblock kgrid_Monkhorst_Pack\n",
            encoding="utf-8",
        )

    assert any(
        "electronic_convergence.json is missing" in error
        for error in pooling_errors(
            [(tmp_path / "primitive", "graphene"), (tmp_path / "supercell", "graphene")]
        )
    )
