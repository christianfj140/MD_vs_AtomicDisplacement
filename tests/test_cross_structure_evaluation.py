from __future__ import annotations

import hashlib
import argparse
import importlib
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "Comparison" / "scripts"
SHARED_DIR = REPO_ROOT / "shared"
for path in (SCRIPTS_DIR, SHARED_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import ml_vs_siesta as mvs  # noqa: E402
import ml_vs_siesta.cross_structure_materialize as csm  # noqa: E402
from benchmark_manifest import write_benchmark_manifests  # noqa: E402
from g2m_deeph_metrics import aggregate_common_metrics  # noqa: E402
from g2m_deeph_runner import DeepHBenchmarkContext, Graph2MatBenchmarkContext, Graph2MatDeepHBenchmarkRunner  # noqa: E402
from joint_artifact_contract import validate_snapshot  # noqa: E402

_SUFFIXES = (".HSX", ".STRUCT_OUT", ".XV", ".ORB_INDX", ".TSHS", ".TSDE")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _fdf(
    label: str,
    n_atoms: int,
    *,
    species: str = "C",
    xc: str = "GGA",
    kgrid: int = 10,
    lattice_scale: float = 1.0,
    ghost_active: bool = False,
) -> str:
    species_block = f"1 6 {species}"
    coords_block = "0 0 0 1"
    if ghost_active:
        species_block += "\n2 -1 Ghost-H"
        coords_block += "\n0.1 0.1 0 2"
    return f"""SystemLabel {label}
NumberOfAtoms {n_atoms}
XC.functional {xc}
MeshCutoff 200 Ry
ElectronicTemperature 300 K
DM.Tolerance 1.d-4
SpinPolarized false
LatticeConstant 1.0 Ang
%block LatticeVectors
{lattice_scale} 0 0
0 {lattice_scale} 0
0 0 10
%endblock LatticeVectors
%block kgrid_Monkhorst_Pack
{kgrid} 0 0 0
0 {kgrid} 0 0
0 0 1 0
%endblock kgrid_Monkhorst_Pack
%block ChemicalSpeciesLabel
{species_block}
%endblock ChemicalSpeciesLabel
%block AtomicCoordinatesAndAtomicSpecies
{coords_block}
%endblock AtomicCoordinatesAndAtomicSpecies
"""


def _make_snapshot(
    root: Path,
    *,
    label: str,
    n_atoms: int,
    tag: str,
    species: str = "C",
    xc: str = "GGA",
    kgrid: int = 10,
    lattice_scale: float = 1.0,
    ghost_active: bool = False,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "RUN.fdf").write_text(
        _fdf(
            label,
            n_atoms,
            species=species,
            xc=xc,
            kgrid=kgrid,
            lattice_scale=lattice_scale,
            ghost_active=ghost_active,
        ),
        encoding="utf-8",
    )
    (root / "metadata.json").write_text(json.dumps({"system_label": label, "n_atoms": n_atoms}), encoding="utf-8")
    (root / "RUN.out").write_text(
        f"run out {tag}\niscf     Eharris\nSCF cycle converged\nJob completed\n",
        encoding="utf-8",
    )
    for suffix in _SUFFIXES:
        name = f"{label}{suffix}"
        content = f"{suffix} {tag}\n"
        if suffix == ".ORB_INDX":
            content = f"1 1 = orbitals in unit cell\n1 1 1 {species} 1\n"
            if ghost_active:
                content = f"2 2 = orbitals in unit cell\n1 1 1 {species} 1\n2 2 2 Ghost-H 1\n"
        (root / name).write_text(content, encoding="utf-8")


def _make_dataset(
    root: Path,
    *,
    n_atoms: int,
    label: str,
    species: str = "C",
    basis: str = "basis-v1",
    pseudo: str = "pseudo-v1",
    xc: str = "GGA",
    kgrid: int = 10,
    lattice_scale: float = 1.0,
    ghost_active: bool = False,
    include_hamiltonian_semantics: bool = True,
    hamiltonian_semantics: dict | None = None,
    counts: dict[str, int] | None = None,
) -> Path:
    counts = counts or {"train": 2, "validation": 1, "test": 1}
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{species}.psf").write_text(pseudo, encoding="utf-8")
    basis_dir = root / "material_basis"
    basis_dir.mkdir(exist_ok=True)
    (basis_dir / f"{species}.ion.xml").write_text(basis, encoding="utf-8")
    if ghost_active:
        (basis_dir / "Ghost-H.ion.xml").write_text("ghost-basis", encoding="utf-8")
    provenance = {
        "profile": "production",
        "label": label,
        "species": [
            {"index": 1, "atomic_number": 6, "label": species},
            *([{"index": 2, "atomic_number": -1, "label": "Ghost-H"}] if ghost_active else []),
        ],
        "basis_file_sha256": {
            f"{species}.ion.xml": _sha(basis),
            **({"Ghost-H.ion.xml": _sha("ghost-basis")} if ghost_active else {}),
        },
        "pseudopotential_sha256": {species: _sha(pseudo)},
        "fdf_sha256": _sha(label),
    }
    if include_hamiltonian_semantics:
        provenance["hamiltonian_target_semantics"] = hamiltonian_semantics or {
            "matrix_component_policy": "h_only",
            "n_matrix_components": 1,
            "real_complex_representation": "real",
        }
    (root / "material_provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    split_rows = {split: [] for split in ("train", "validation", "test")}
    for split, n in counts.items():
        for i in range(n):
            sample_id = f"{split}_{i}"
            sample_dir = root / "samples" / sample_id
            _make_snapshot(
                sample_dir,
                label=label,
                n_atoms=n_atoms,
                tag=sample_id,
                species=species,
                xc=xc,
                kgrid=kgrid,
                lattice_scale=lattice_scale,
                ghost_active=ghost_active,
            )
            split_rows[split].append(
                {
                    "sample_id": sample_id,
                    "sample_dir": str(sample_dir),
                    "split": split,
                    "system_label": label,
                }
            )
    snapshots = []
    for rows in split_rows.values():
        for row in rows:
            result = validate_snapshot(Path(row["sample_dir"]))
            snapshots.append(result.to_dict())
    (root / "artifact_validation.json").write_text(
        json.dumps(
            {
                "contract_name": "joint_graph2mat_deeph_artifact_contract_v1",
                "valid": all(item["valid"] for item in snapshots),
                "warnings": [],
                "snapshots": snapshots,
            }
        ),
        encoding="utf-8",
    )
    import csv

    split_root = root / "splits"
    split_root.mkdir(exist_ok=True)
    for split, rows in split_rows.items():
        with (split_root / f"{split}_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["sample_id", "sample_dir", "split", "system_label"])
            writer.writeheader()
            writer.writerows(rows)
    write_benchmark_manifests(dataset_root=root, split_root=split_root, generation_mode="fake")
    return root


@pytest.fixture
def source_target(tmp_path: Path) -> tuple[Path, Path]:
    return (
        _make_dataset(tmp_path / "source", n_atoms=2, label="graphene2"),
        _make_dataset(tmp_path / "target", n_atoms=50, label="graphene50", kgrid=2, lattice_scale=5.0),
    )


def test_cross_structure_materialization_exact_splits(source_target: tuple[Path, Path], tmp_path: Path) -> None:
    source, target = source_target
    out = tmp_path / "cross"
    summary = mvs.materialize_cross_structure_dataset(source, target, out)
    assert summary["split_counts"] == {"train": 2, "validation": 1, "test": 1}
    frozen = json.loads((out / "frozen_split_manifest.json").read_text())
    rows = frozen["rows"]
    assert frozen["valid"] is True
    assert {row["sample_id"] for row in rows} == {
        "source_train__train_0",
        "source_train__train_1",
        "source_validation__validation_0",
        "target_test__test_0",
    }
    assert all(row["role"] == "source" for row in rows if row["split"] in {"train", "validation"})
    assert all(row["role"] == "target" for row in rows if row["split"] == "test")
    assert not any(row["original_split"] == "test" and row["role"] == "source" for row in rows)
    assert not any(row["original_split"] in {"train", "validation"} and row["role"] == "target" for row in rows)
    assert all(row["evaluation_mode"] == "cross_structure" for row in rows)
    assert {row["transfer_direction"] for row in rows} == {"2_atoms_to_50_atoms"}
    assert all(row.get("source_artifact_identity") for row in rows)
    bench = json.loads((out / "benchmark_dataset_manifest.json").read_text())
    assert bench["benchmark_ready"] is True
    validation = Graph2MatDeepHBenchmarkRunner().validate_dataset_payload(
        {"dataset_root": str(out), "dataset_mode": "reuse_validated", "strict_dataset_validation": False}
    )
    assert validation["benchmark_ready"] is True
    assert validation["artifact_summary"]["invalid_snapshots"] == 0
    provenance = json.loads((out / "cross_structure_dataset_provenance.json").read_text())
    assert provenance["evaluation_scope"] == "target_structure_only"
    assert provenance["validation_scope"] == "source_structure_only"
    assert provenance["source_atom_counts"] == [2]
    assert provenance["target_atom_counts"] == [50]
    assert provenance["leakage_check"]["passed"] is True
    assert provenance["hamiltonian_target_semantics"]["h_only_policy"] == "h_only"
    assert provenance["materialization"]["linked_artifacts"] > 0


def test_cross_structure_test_membership_stable(source_target: tuple[Path, Path], tmp_path: Path) -> None:
    source, target = source_target
    first = mvs.materialize_cross_structure_dataset(source, target, tmp_path / "one")
    second = mvs.materialize_cross_structure_dataset(source, target, tmp_path / "two")
    assert first["frozen_split_hash"] == second["frozen_split_hash"]
    one = json.loads((tmp_path / "one" / "cross_structure_dataset_provenance.json").read_text())
    two = json.loads((tmp_path / "two" / "cross_structure_dataset_provenance.json").read_text())
    assert one["target_test_ids"] == two["target_test_ids"] == ["test_0"]


def test_cross_structure_reuses_existing_composite_for_train(source_target: tuple[Path, Path], tmp_path: Path) -> None:
    source, target = source_target
    out = tmp_path / "dataset"
    first = mvs.materialize_cross_structure_dataset(source, target, out)
    assert first.get("reused_existing") is None
    seen: list[dict] = []
    result = mvs.run_cross_structure_payload(
        {
            "action": "train",
            "source_dataset_root": str(source),
            "target_dataset_root": str(target),
            "composite_dataset_root": str(out),
            "run_output_root": str(tmp_path / "run"),
        },
        launch_fn=lambda payload: seen.append(payload) or {"running": False, "returncode": 0},
    )
    assert result["materialized"]["reused_existing"] is True
    assert seen[0]["dataset_root"] == str(out)
    assert seen[0]["cross_structure_metadata"]["evaluation_mode"] == "cross_structure"


def test_cross_structure_reuse_revalidates_current_frozen_rows(source_target: tuple[Path, Path], tmp_path: Path) -> None:
    source, target = source_target
    out = tmp_path / "dataset"
    mvs.materialize_cross_structure_dataset(source, target, out)
    frozen_path = out / "frozen_split_manifest.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    frozen["rows"][0]["role"] = "target"
    frozen_path.write_text(json.dumps(frozen), encoding="utf-8")
    with pytest.raises(mvs.DatasetMaterializeError, match="leakage"):
        mvs.materialize_or_reuse_cross_structure_dataset(source, target, out)


def test_cross_structure_reuse_rejects_missing_source_artifact_identity(
    source_target: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    source, target = source_target
    out = tmp_path / "dataset"
    mvs.materialize_cross_structure_dataset(source, target, out)
    frozen_path = out / "frozen_split_manifest.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    frozen["rows"][0].pop("source_artifact_identity", None)
    frozen_path.write_text(json.dumps(frozen), encoding="utf-8")
    with pytest.raises(mvs.DatasetMaterializeError, match="source_artifact_identity"):
        mvs.materialize_or_reuse_cross_structure_dataset(source, target, out)


def test_cross_structure_records_actual_copy_mode(source_target: tuple[Path, Path], tmp_path: Path) -> None:
    source, target = source_target
    out = tmp_path / "copied"
    mvs.materialize_cross_structure_dataset(source, target, out, link=False)
    provenance = json.loads((out / "cross_structure_dataset_provenance.json").read_text())
    materialization = provenance["materialization"]
    assert materialization["linked_or_copied"] == "copied"
    assert materialization["copied_artifacts"] > 0
    assert materialization["linked_artifacts"] == 0


@pytest.mark.parametrize(
    "side,key,match",
    [
        ("source", "basis_file_sha256", "Missing orbital basis hash"),
        ("target", "basis_file_sha256", "Missing orbital basis hash"),
        ("source", "pseudopotential_sha256", "Missing pseudopotential hash"),
        ("target", "pseudopotential_sha256", "Missing pseudopotential hash"),
    ],
)
def test_cross_structure_missing_basis_or_pseudopotential_hashes_fail_closed(
    tmp_path: Path,
    side: str,
    key: str,
    match: str,
) -> None:
    source = _make_dataset(tmp_path / "source", n_atoms=2, label="source")
    target = _make_dataset(tmp_path / "target", n_atoms=50, label="target", kgrid=2, lattice_scale=5.0)
    root = source if side == "source" else target
    provenance_path = root / "material_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance[key] = {}
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(mvs.DatasetCompatibilityError, match=match):
        mvs.plan_cross_structure_dataset(source, target)


def test_cross_structure_materialization_failure_removes_partial_and_output(
    source_target: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, target = source_target
    out = tmp_path / "dataset"
    source_frozen_before = (source / "frozen_split_manifest.json").read_text(encoding="utf-8")
    target_frozen_before = (target / "frozen_split_manifest.json").read_text(encoding="utf-8")

    def fail_validation(_path: Path):
        raise RuntimeError("forced validation failure")

    monkeypatch.setattr(csm, "validate_snapshot", fail_validation)
    with pytest.raises(RuntimeError, match="forced validation failure"):
        mvs.materialize_cross_structure_dataset(source, target, out)

    assert not out.exists()
    assert not list(tmp_path.glob("dataset.partial-*"))
    assert (source / "frozen_split_manifest.json").read_text(encoding="utf-8") == source_frozen_before
    assert (target / "frozen_split_manifest.json").read_text(encoding="utf-8") == target_frozen_before


def test_cross_structure_hamiltonian_semantics_fail_closed_without_confirmation(tmp_path: Path) -> None:
    source = _make_dataset(
        tmp_path / "source",
        n_atoms=2,
        label="source",
        include_hamiltonian_semantics=False,
    )
    target = _make_dataset(
        tmp_path / "target",
        n_atoms=50,
        label="target",
        include_hamiltonian_semantics=False,
        kgrid=2,
        lattice_scale=5.0,
    )
    with pytest.raises(mvs.DatasetCompatibilityError, match="Incomplete Hamiltonian target semantics"):
        mvs.plan_cross_structure_dataset(source, target)
    preview = mvs.plan_cross_structure_dataset(
        source,
        target,
        confirm_incomplete_hamiltonian_semantics=True,
    )
    semantics = preview["hamiltonian_target_semantics"]
    assert semantics["confirmed_incomplete_hamiltonian_semantics"] is True
    assert semantics["blocking_errors"]


@pytest.mark.parametrize(
    "target_semantics,match",
    [
        (
            {"matrix_component_policy": "h_only", "n_matrix_components": 1, "real_complex_representation": "complex_supported"},
            "real/complex representation",
        ),
        (
            {"matrix_component_policy": "h_and_overlap", "n_matrix_components": 1, "real_complex_representation": "real"},
            "H-only policy",
        ),
        (
            {"matrix_component_policy": "h_only", "n_matrix_components": 2, "real_complex_representation": "real"},
            "matrix component count",
        ),
    ],
)
def test_cross_structure_hamiltonian_semantic_mismatch_fails_even_with_confirmation(
    tmp_path: Path,
    target_semantics: dict,
    match: str,
) -> None:
    source = _make_dataset(tmp_path / "source", n_atoms=2, label="source")
    target = _make_dataset(
        tmp_path / "target",
        n_atoms=50,
        label="target",
        kgrid=2,
        lattice_scale=5.0,
        hamiltonian_semantics=target_semantics,
    )
    with pytest.raises(mvs.DatasetCompatibilityError, match=match):
        mvs.plan_cross_structure_dataset(
            source,
            target,
            confirm_incomplete_hamiltonian_semantics=True,
        )


def test_cross_structure_matching_hamiltonian_semantics_pass(source_target: tuple[Path, Path]) -> None:
    source, target = source_target
    preview = mvs.plan_cross_structure_dataset(source, target)
    assert preview["hamiltonian_target_semantics"]["blocking_errors"] == []


def test_cross_structure_preview_writes_nothing(source_target: tuple[Path, Path], tmp_path: Path) -> None:
    source, target = source_target
    out = tmp_path / "preview_out"
    result = mvs.run_cross_structure_payload(
        {
            "action": "preview",
            "source_dataset_root": str(source),
            "target_dataset_root": str(target),
            "composite_dataset_root": str(out),
        }
    )
    assert result["preview"]["split_counts"] == {"train": 2, "validation": 1, "test": 1}
    assert result["preview"]["source_atom_counts"] == [2]
    assert result["preview"]["target_atom_counts"] == [50]
    assert not out.exists()


def test_cross_structure_materialize_action_writes_dataset_without_runner(source_target: tuple[Path, Path], tmp_path: Path) -> None:
    source, target = source_target
    out = tmp_path / "materialized"
    result = mvs.run_cross_structure_payload(
        {
            "action": "materialize",
            "source_dataset_root": str(source),
            "target_dataset_root": str(target),
            "composite_dataset_root": str(out),
        },
        launch_fn=lambda _payload: pytest.fail("materialize must not launch runner"),
    )
    assert out.exists()
    assert result["materialized"]["split_counts"] == {"train": 2, "validation": 1, "test": 1}
    assert "runner_result" not in result


def test_cross_structure_train_payload_forces_reuse_and_passes_settings(source_target: tuple[Path, Path], tmp_path: Path) -> None:
    source, target = source_target
    seen: list[dict] = []
    result = mvs.run_cross_structure_payload(
        {
            "action": "train",
            "source_dataset_root": str(source),
            "target_dataset_root": str(target),
            "composite_dataset_root": str(tmp_path / "dataset"),
            "run_output_root": str(tmp_path / "run"),
            "runner_payload": {
                "selected_methods": ["graph2mat", "deeph"],
                "graph2mat_overrides": {"max_epochs": 3},
                "deeph": {"epochs": 3},
                "performance": {"max_parallel_graph2mat_training_jobs": 1},
            },
        },
        launch_fn=lambda payload: seen.append(payload) or {"running": True},
    )
    runner_payload = seen[0]
    assert runner_payload["dataset_mode"] == "reuse_validated"
    assert runner_payload["dataset_root"] == str(tmp_path / "dataset")
    assert runner_payload["output_root"] == str(tmp_path / "run")
    assert runner_payload["allow_regenerate_siesta"] is False
    assert runner_payload["selected_methods"] == ["graph2mat", "deeph"]
    assert runner_payload["graph2mat_overrides"] == {"max_epochs": 3}
    assert result["runner_result"] == {"running": True}


def test_cross_structure_payload_rejects_protected_runner_fields(source_target: tuple[Path, Path], tmp_path: Path) -> None:
    source, target = source_target
    with pytest.raises(mvs.DatasetMaterializeError, match="protected"):
        mvs.run_cross_structure_payload(
            {
                "action": "train",
                "source_dataset_root": str(source),
                "target_dataset_root": str(target),
                "composite_dataset_root": str(tmp_path / "dataset"),
                "runner_payload": {"dataset_root": "/bad"},
            },
            launch_fn=lambda _payload: None,
        )


def test_cross_structure_train_launch_failure_propagates(source_target: tuple[Path, Path], tmp_path: Path) -> None:
    source, target = source_target

    def fail(_payload: dict) -> None:
        raise RuntimeError("launch failed")

    with pytest.raises(RuntimeError, match="launch failed"):
        mvs.run_cross_structure_payload(
            {
                "action": "train",
                "source_dataset_root": str(source),
                "target_dataset_root": str(target),
                "composite_dataset_root": str(tmp_path / "dataset"),
            },
            launch_fn=fail,
        )


def test_cross_structure_rejects_training_sweep(source_target: tuple[Path, Path], tmp_path: Path) -> None:
    source, target = source_target
    with pytest.raises(mvs.DatasetMaterializeError, match="training_sweep"):
        mvs.run_cross_structure_payload(
            {
                "action": "train",
                "source_dataset_root": str(source),
                "target_dataset_root": str(target),
                "composite_dataset_root": str(tmp_path / "dataset"),
                "runner_payload": {"training_sweep": {"enabled": True}},
            },
            launch_fn=lambda _payload: None,
        )


def test_cross_structure_cli_train_polls_and_writes_status(source_target: tuple[Path, Path], tmp_path: Path, monkeypatch) -> None:
    source, target = source_target
    cli = importlib.import_module("run_cross_structure_payload")

    class FakeRunner:
        def __init__(self) -> None:
            self.calls = 0

        def start(self, payload: dict) -> dict:
            self.payload = payload
            return {"running": True, "returncode": None}

        def status(self) -> dict:
            self.calls += 1
            return {"running": self.calls < 2, "returncode": 0, "dataset_root": self.payload["dataset_root"]}

        def write_incremental_manifest(self, path: Path) -> None:
            path.write_text(json.dumps({"written": True}), encoding="utf-8")

        def results(self) -> dict:
            return {"status": {"returncode": 0}}

    monkeypatch.setattr(cli, "Graph2MatDeepHBenchmarkRunner", FakeRunner)
    monkeypatch.setattr(cli.time, "sleep", lambda _seconds: None)
    payload = {
        "action": "train",
        "source_dataset_root": str(source),
        "target_dataset_root": str(target),
        "composite_dataset_root": str(tmp_path / "dataset"),
        "run_output_root": str(tmp_path / "run"),
    }
    args = argparse.Namespace(
        status_json=tmp_path / "status.json",
        manifest_json=tmp_path / "manifest.json",
        poll_seconds=1,
    )
    result, returncode = cli._run_train(payload, args)
    assert returncode == 0
    assert result["runner_result"]["running"] is False
    assert json.loads(args.status_json.read_text())["status"]["returncode"] == 0
    assert json.loads(args.manifest_json.read_text()) == {"written": True}


def test_leakage_report_detects_duplicate_source_identity(tmp_path: Path) -> None:
    root = tmp_path / "source"
    sample_dir = root / "sample"
    sample_dir.mkdir(parents=True)
    sample = csm.DatasetSample("same", sample_dir, "sys", root, n_atoms=2, split="train")
    report = csm._leakage_report({"train": [("source", sample)], "validation": [("source", sample)], "test": []})
    assert report["passed"] is False
    assert any("canonical source artifact identity reused" in error for error in report["errors"])


@pytest.mark.parametrize(
    "source_kwargs,target_kwargs,match",
    [
        ({"species": "C"}, {"species": "Si"}, "Real species differ"),
        ({"basis": "basis-a"}, {"basis": "basis-b"}, "Orbital basis differs"),
        ({"pseudo": "pseudo-a"}, {"pseudo": "pseudo-b"}, "pseudopotential"),
        ({"xc": "GGA"}, {"xc": "LDA"}, "XC.functional differs"),
        ({}, {"ghost_active": True}, "Ghost species are ACTIVE|ghost_compatibility"),
    ],
)
def test_cross_structure_blocks_incompatibilities(tmp_path: Path, source_kwargs: dict, target_kwargs: dict, match: str) -> None:
    source = _make_dataset(tmp_path / "source", n_atoms=2, label="source", **source_kwargs)
    target = _make_dataset(tmp_path / "target", n_atoms=50, label="target", **target_kwargs)
    with pytest.raises((mvs.DatasetMaterializeError, mvs.DatasetCompatibilityError), match=match):
        mvs.plan_cross_structure_dataset(source, target)


def test_cross_structure_missing_required_splits_fail(tmp_path: Path) -> None:
    source = _make_dataset(tmp_path / "source", n_atoms=2, label="source", counts={"train": 1, "validation": 0, "test": 1})
    target = _make_dataset(tmp_path / "target", n_atoms=50, label="target", counts={"train": 1, "validation": 1, "test": 0})
    with pytest.raises(mvs.DatasetMaterializeError, match="validation"):
        mvs.plan_cross_structure_dataset(source, target)
    source_ok = _make_dataset(tmp_path / "source_ok", n_atoms=2, label="source")
    with pytest.raises(mvs.DatasetMaterializeError, match="test"):
        mvs.plan_cross_structure_dataset(source_ok, target)


def test_cross_structure_invalid_frozen_manifest_fails(tmp_path: Path) -> None:
    source = _make_dataset(tmp_path / "source", n_atoms=2, label="source")
    target = _make_dataset(tmp_path / "target", n_atoms=50, label="target")
    (target / "frozen_split_manifest.json").write_text(json.dumps({"rows": []}), encoding="utf-8")
    with pytest.raises(mvs.DatasetMaterializeError, match="no rows"):
        mvs.plan_cross_structure_dataset(source, target)


def test_cross_structure_allows_different_atom_counts_and_raw_kgrid(source_target: tuple[Path, Path]) -> None:
    source, target = source_target
    preview = mvs.plan_cross_structure_dataset(source, target)
    assert preview["source_atom_counts"] == [2]
    assert preview["target_atom_counts"] == [50]
    assert preview["compatibility"]["compatible"] is True
    sampling = preview["compatibility"]["compatibility_report"]["sampling_differences"]
    assert any("raw Monkhorst-Pack grids differ" in item for item in sampling)
    assert any("lattice vectors/cell dimensions differ" in item for item in sampling)


def test_cross_structure_metadata_reaches_runner_and_common_metrics(
    source_target: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    source, target = source_target
    dataset = tmp_path / "dataset"
    mvs.materialize_cross_structure_dataset(source, target, dataset)
    run_root = tmp_path / "run"
    training_dir = run_root / "graph2mat" / "training"
    training_dir.mkdir(parents=True)
    (training_dir / "config.yaml").write_text("{}\n", encoding="utf-8")
    (training_dir / "runs.json").write_text("{}\n", encoding="utf-8")
    context = Graph2MatBenchmarkContext(
        dataset_root=dataset,
        run_root=run_root,
        graph2mat_root=run_root / "graph2mat",
        training_dir=training_dir,
        prediction_structs_dir=run_root / "graph2mat" / "prediction_structures" / "test",
        config_path=run_root / "graph2mat" / "pipeline_config.yaml",
        graph2mat_config_path=training_dir / "config.yaml",
        graph2mat_manifest_path=run_root / "graph2mat" / "graph2mat_manifest.json",
        frozen_split_manifest_path=dataset / "frozen_split_manifest.json",
        benchmark_dataset_manifest_path=dataset / "benchmark_dataset_manifest.json",
        runs_json_path=training_dir / "runs.json",
        runs_json_counts={},
        train_glob="",
        validation_glob="",
        predict_glob="",
        output_file="ML_prediction.HSX",
        test_sample_ids=["target_test__test_0"],
        split_hash=json.loads((dataset / "frozen_split_manifest.json").read_text())["split_hash"],
        prediction_split="test",
        dry_run=False,
    )
    runner = Graph2MatDeepHBenchmarkRunner()
    manifest = runner._write_graph2mat_manifest(context)
    assert manifest["cross_structure_metadata"]["transfer_direction"] == "2_atoms_to_50_atoms"
    cross_metadata = json.loads((dataset / "cross_structure_dataset_provenance.json").read_text())["cross_structure_metadata"]
    deeph_context = DeepHBenchmarkContext(
        root=run_root / "deeph",
        raw_dir=run_root / "deeph" / "raw",
        processed_dir=run_root / "deeph" / "processed",
        graph_dir=run_root / "deeph" / "graph",
        save_dir=run_root / "deeph" / "save",
        inference_dir=run_root / "deeph" / "inference",
        preprocess_config=run_root / "deeph" / "preprocess.ini",
        train_config=run_root / "deeph" / "train.ini",
        inference_configs=[],
        inference_work_dirs=[],
        manifest_path=run_root / "deeph" / "deeph_manifest.json",
        deeph_discovery={},
        split_audit_path=run_root / "deeph" / "deeph_split_audit.json",
        split_audit_csv_path=run_root / "deeph" / "deeph_split_audit.csv",
        split_hash=context.split_hash,
        raw_mirror={"cross_structure_metadata": cross_metadata},
        inference_split="test",
        dry_run=False,
    )
    deeph_manifest = runner._write_deeph_manifest(deeph_context)
    assert deeph_manifest["cross_structure_metadata"]["evaluation_mode"] == "cross_structure"
    assert deeph_manifest["cross_structure_metadata"]["transfer_direction"] == "2_atoms_to_50_atoms"
    assert deeph_manifest["cross_structure_metadata"]["source_atom_counts"] == [2]
    assert deeph_manifest["cross_structure_metadata"]["target_atom_counts"] == [50]
    assert deeph_manifest["cross_structure_metadata"]["source_split_hash"]
    assert deeph_manifest["cross_structure_metadata"]["target_split_hash"]
    assert deeph_manifest["cross_structure_metadata"]["composite_split_hash"]

    def write_metrics(root: Path, sample: str) -> None:
        root.mkdir(parents=True)
        import csv

        with (root / "kpoint_matrix_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["sample", "row_type", "h_mae_eV", "h_rmse_eV"])
            writer.writeheader()
            writer.writerow({"sample": sample, "row_type": "weighted_sample", "h_mae_eV": "0.1", "h_rmse_eV": "0.2"})
        (root / "manifest.json").write_text(
            json.dumps(
                {
                    "kpoint_metrics_enabled": True,
                    "uses_reference_overlap_k": True,
                    "raw_global_equivalence_proven": True,
                }
            ),
            encoding="utf-8",
        )

    write_metrics(tmp_path / "g2m_metrics", "target_test__test_0")
    write_metrics(tmp_path / "deeph_metrics", "target_test__test_0")
    common = aggregate_common_metrics(
        graph2mat_metrics_root=tmp_path / "g2m_metrics",
        deeph_metrics_root=tmp_path / "deeph_metrics",
        output_dir=tmp_path / "summary",
        frozen_split_manifest_path=dataset / "frozen_split_manifest.json",
        dataset_manifest_path=dataset / "benchmark_dataset_manifest.json",
    )
    assert common["cross_structure_metadata"]["evaluation_mode"] == "cross_structure"
    assert common["cross_structure_metadata"]["source_split_hash"]
    assert common["cross_structure_metadata"]["target_split_hash"]
