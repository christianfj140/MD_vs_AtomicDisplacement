"""Small, CPU-only tests for the ML vs SIESTA benchmark toolkit.

No SIESTA, no training, no GPU, no large data. Everything uses tiny fixtures.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "Comparison" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import ml_vs_siesta as mvs  # noqa: E402
from ml_vs_siesta.config import BenchmarkConfigError, parse_benchmark_config  # noqa: E402
from ml_vs_siesta.predictors import FunctionMatrixPredictor  # noqa: E402
from ml_vs_siesta.species_transfer import (  # noqa: E402
    SpeciesTransferConfigError,
    load_species_transfer_config,
    prepare_species_expansion,
)
from ml_vs_siesta.structure import BenchmarkStructure  # noqa: E402


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #
def _two_atom_structure() -> BenchmarkStructure:
    return BenchmarkStructure(
        symbols=["C", "C"],
        positions=np.array([[0.0, 0.0, 10.0], [1.23, 0.71, 10.0]]),
        cell=np.array([[2.46, 0.0, 0.0], [1.23, 2.13, 0.0], [0.0, 0.0, 20.0]]),
        species_index=[1, 1],
    )


def _minimal_config_payload(structure_path: str | None = None) -> dict:
    return {
        "system": {
            "input_structure": structure_path,
            "supercell": [5, 5, 1],
            "central_atom": "auto",
        },
        "derivatives": {"enabled": True, "displacement": 0.01, "directions": ["x", "y", "z"]},
        "models": {"enabled": ["graph2mat", "deeph"]},
        "matrices": {"targets": ["hamiltonian", "density_matrix", "overlap"]},
        "dataset_mixing": {"enabled": False},
        "species_transfer": {"enabled": False, "base_species": ["C"], "new_species": ["H"]},
        "ui": {"enable_matrix_viewer": True},
    }


# --------------------------------------------------------------------------- #
# Phase 1 — config
# --------------------------------------------------------------------------- #
def test_config_parses_and_validates():
    config = parse_benchmark_config(_minimal_config_payload())
    assert config.system.supercell == (5, 5, 1)
    assert config.system.central_atom == "auto"
    assert config.derivatives.enabled is True
    assert config.models == ("graph2mat", "deeph")
    assert config.targets == ("hamiltonian", "density_matrix", "overlap")
    round_trip = config.to_dict()
    assert round_trip["system"]["supercell"] == [5, 5, 1]


def test_config_example_file_loads():
    config = mvs.load_benchmark_config(
        REPO_ROOT / "Comparison" / "config" / "ml_vs_siesta_benchmark_example.yaml"
    )
    assert config.ui_enable_matrix_viewer is True
    assert config.species_transfer.new_species == ("H",)


@pytest.mark.parametrize(
    "payload",
    [
        {"system": {"supercell": [5, 5]}},
        {"models": {"enabled": ["not_a_model"]}},
        {"matrices": {"targets": ["hamiltonian", "bogus"]}},
        {"derivatives": {"displacement": -1}},
        {"system": {"central_atom": -3}},
    ],
)
def test_config_rejects_bad_values(payload):
    with pytest.raises(BenchmarkConfigError):
        parse_benchmark_config(payload)


# --------------------------------------------------------------------------- #
# Phase 2 — supercell
# --------------------------------------------------------------------------- #
def test_make_supercell_counts_and_cell():
    structure = _two_atom_structure()
    supercell = mvs.make_supercell(structure, (5, 5, 1))
    assert supercell.n_atoms == 2 * 5 * 5 * 1
    np.testing.assert_allclose(supercell.cell[0], structure.cell[0] * 5)
    np.testing.assert_allclose(supercell.cell[1], structure.cell[1] * 5)
    np.testing.assert_allclose(supercell.cell[2], structure.cell[2] * 1)
    # Traceability: every replica maps to a valid original atom.
    assert set(supercell.source_atom) == {0, 1}
    assert supercell.symbols.count("C") == supercell.n_atoms


def test_make_supercell_rejects_bad_reps():
    with pytest.raises(mvs.StructureError):
        mvs.make_supercell(_two_atom_structure(), (0, 1, 1))


# --------------------------------------------------------------------------- #
# Phase 3 — central atom
# --------------------------------------------------------------------------- #
def test_find_central_atom_unambiguous():
    # Atom 1 sits exactly at the centroid; it must be picked.
    structure = BenchmarkStructure(
        symbols=["C", "C", "C"],
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        cell=np.eye(3) * 10,
    )
    assert mvs.find_central_atom(structure) == 1


def test_find_central_atom_species_filter():
    structure = BenchmarkStructure(
        symbols=["H", "C", "H"],
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        cell=np.eye(3) * 10,
    )
    assert mvs.find_central_atom(structure, species="C") == 1


# --------------------------------------------------------------------------- #
# Phase 4 — displacements
# --------------------------------------------------------------------------- #
def test_make_displaced_structures_moves_one_coord():
    structure = _two_atom_structure()
    original = structure.positions.copy()
    plus, minus = mvs.make_displaced_structures(structure, 1, "y", 0.02)
    # Original untouched.
    np.testing.assert_array_equal(structure.positions, original)
    # Exactly +h / -h on the single coordinate.
    assert plus.positions[1, 1] == pytest.approx(original[1, 1] + 0.02)
    assert minus.positions[1, 1] == pytest.approx(original[1, 1] - 0.02)
    # Nothing else changed.
    delta = plus.positions - original
    assert np.count_nonzero(np.abs(delta) > 1e-12) == 1


@pytest.mark.parametrize("direction", ["x", "y", "z", 0, 1, 2])
def test_make_displaced_structures_direction_tokens(direction):
    plus, minus = mvs.make_displaced_structures(_two_atom_structure(), 0, direction, 0.01)
    assert plus.n_atoms == 2


def test_make_displaced_structures_validations():
    structure = _two_atom_structure()
    with pytest.raises(mvs.StructureError):
        mvs.make_displaced_structures(structure, 5, "x", 0.01)
    with pytest.raises(mvs.StructureError):
        mvs.make_displaced_structures(structure, 0, "w", 0.01)
    with pytest.raises(mvs.StructureError):
        mvs.make_displaced_structures(structure, 0, "x", -0.01)


# --------------------------------------------------------------------------- #
# Phase 5 — fdf generation (dry-run)
# --------------------------------------------------------------------------- #
def test_generate_siesta_displacement_inputs_dry_run():
    structure_path = REPO_ROOT / "Comparison" / "config" / "ml_vs_siesta_example_structure.fdf"
    config = parse_benchmark_config(_minimal_config_payload(str(structure_path)))
    metadata = mvs.generate_siesta_displacement_inputs(config, "/tmp/does-not-exist", dry_run=True)
    assert metadata["dry_run"] is True
    assert set(metadata["generated_files"]) == {
        "reference",
        "x_plus",
        "x_minus",
        "y_plus",
        "y_minus",
        "z_plus",
        "z_minus",
    }
    assert metadata["supercell_atom_count"] == 50
    assert metadata["central_atom_symbol"] == "C"


def test_generate_siesta_displacement_inputs_writes(tmp_path):
    structure_path = REPO_ROOT / "Comparison" / "config" / "ml_vs_siesta_example_structure.fdf"
    config = parse_benchmark_config(_minimal_config_payload(str(structure_path)))
    metadata = mvs.generate_siesta_displacement_inputs(config, tmp_path)
    assert (tmp_path / "metadata.json").is_file()
    assert (tmp_path / "reference" / "RUN.fdf").is_file()
    for label in metadata["displacement_labels"]:
        assert (tmp_path / label / "RUN.fdf").is_file()


# --------------------------------------------------------------------------- #
# Phase 8 — MatrixData + compatibility
# --------------------------------------------------------------------------- #
def test_validate_matrix_compatible_ok_and_fail():
    a = mvs.MatrixData(values=np.zeros((3, 3)), target="hamiltonian")
    b = mvs.MatrixData(values=np.ones((3, 3)), target="hamiltonian")
    mvs.validate_matrix_compatible(a, b)  # ok

    with pytest.raises(mvs.MatrixCompatibilityError):
        mvs.validate_matrix_compatible(a, mvs.MatrixData(np.zeros((2, 2)), "hamiltonian"))
    with pytest.raises(mvs.MatrixCompatibilityError):
        mvs.validate_matrix_compatible(a, mvs.MatrixData(np.zeros((3, 3)), "overlap"))
    with pytest.raises(mvs.MatrixCompatibilityError):
        mvs.validate_matrix_compatible(
            mvs.MatrixData(np.zeros((3, 3)), "hamiltonian", metadata={"units": "eV"}),
            mvs.MatrixData(np.zeros((3, 3)), "hamiltonian", metadata={"units": "Ry"}),
        )


# --------------------------------------------------------------------------- #
# Phase 9 — SIESTA matrix loader
# --------------------------------------------------------------------------- #
def test_load_siesta_matrices_npy(tmp_path):
    np.save(tmp_path / "hamiltonian.npy", np.eye(4))
    (tmp_path / "hamiltonian.meta.json").write_text(
        json.dumps({"units": "eV", "atom_order": [0, 1]}), encoding="utf-8"
    )
    loaded = mvs.load_siesta_matrices(tmp_path, ["hamiltonian"])
    assert loaded["hamiltonian"].shape == (4, 4)
    assert loaded["hamiltonian"].metadata["units"] == "eV"


def test_load_siesta_matrices_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        mvs.load_siesta_matrices(tmp_path, ["overlap"])


# --------------------------------------------------------------------------- #
# Phase 10 — matrix error
# --------------------------------------------------------------------------- #
def test_compute_matrix_error_obvious_values():
    ref = mvs.MatrixData(values=np.array([[0.0, 0.0], [0.0, 0.0]]), target="hamiltonian")
    pred = mvs.MatrixData(values=np.array([[1.0, -1.0], [2.0, -2.0]]), target="hamiltonian")
    summary = mvs.compute_matrix_error(ref, pred)
    assert summary.mae == pytest.approx(1.5)
    assert summary.max_abs_error == pytest.approx(2.0)
    assert summary.rmse == pytest.approx(np.sqrt((1 + 1 + 4 + 4) / 4))
    assert summary.relative_mae is None  # zero reference


# --------------------------------------------------------------------------- #
# Phase 7 — predictors
# --------------------------------------------------------------------------- #
def test_graph2mat_predictor_instantiates_without_training():
    predictor = mvs.Graph2MatPredictor(checkpoint="ckpt.pt")
    assert predictor.name == "graph2mat"


def test_graph2mat_predictor_predict_stub_raises():
    predictor = mvs.Graph2MatPredictor()
    with pytest.raises(NotImplementedError):
        predictor.predict(_two_atom_structure(), ["hamiltonian"])


def test_deeph_predictor_predict_stub_raises():
    with pytest.raises(NotImplementedError):
        mvs.DeepHPredictor().predict(_two_atom_structure(), ["hamiltonian"])


# --------------------------------------------------------------------------- #
# Phase 11 — compare model to SIESTA
# --------------------------------------------------------------------------- #
def test_compare_model_to_siesta_with_fake_predictor(tmp_path):
    np.save(tmp_path / "hamiltonian.npy", np.zeros((3, 3)))

    def predict(structure, targets):
        return {t: mvs.MatrixData(values=np.ones((3, 3)), target=t) for t in targets}

    predictor = FunctionMatrixPredictor(predict, name="fake")
    result = mvs.compare_model_to_siesta(tmp_path, predictor, _two_atom_structure(), ["hamiltonian"])
    assert result["model"] == "fake"
    assert result["errors"]["hamiltonian"]["mae"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Phase 12 — finite difference derivative
# --------------------------------------------------------------------------- #
def test_finite_difference_matrix_derivative_exact_for_linear():
    # Matrix element equals the displaced atom's x coordinate → derivative == 1.
    def predict(structure, targets):
        x = structure.positions[0, 0]
        return {t: mvs.MatrixData(values=np.array([[x, 2 * x]]), target=t) for t in targets}

    predictor = FunctionMatrixPredictor(predict, name="linear")
    derivative = mvs.finite_difference_matrix_derivative(
        predictor, _two_atom_structure(), 0, "x", 0.01, ["hamiltonian"]
    )
    np.testing.assert_allclose(derivative["hamiltonian"].values, np.array([[1.0, 2.0]]))


# --------------------------------------------------------------------------- #
# Phase 13 — torch finite difference (optional)
# --------------------------------------------------------------------------- #
def test_torch_finite_difference_derivative():
    torch = pytest.importorskip("torch")

    weight = torch.tensor(3.0, requires_grad=True)

    def model_fn(positions):
        return weight * positions[0, 0].reshape(1, 1)

    derivative = mvs.torch_finite_difference_matrix_derivative(
        model_fn, torch.zeros((2, 3)), 0, "x", 0.01
    )
    assert torch.is_tensor(derivative)
    assert derivative.requires_grad  # differentiable w.r.t. model params
    assert derivative.item() == pytest.approx(3.0)


# --------------------------------------------------------------------------- #
# Phase 14 — compare derivatives to SIESTA
# --------------------------------------------------------------------------- #
def test_compare_derivatives_to_siesta(tmp_path):
    # SIESTA derivative fixture: constant 1.0 along x.
    (tmp_path / "x").mkdir()
    np.save(tmp_path / "x" / "hamiltonian.npy", np.array([[1.0, 2.0]]))

    def predict(structure, targets):
        x = structure.positions[0, 0]
        return {t: mvs.MatrixData(values=np.array([[x, 2 * x]]), target=t) for t in targets}

    predictor = FunctionMatrixPredictor(predict, name="linear")
    result = mvs.compare_derivatives_to_siesta(
        tmp_path, predictor, _two_atom_structure(), 0, ["x"], 0.01, ["hamiltonian"]
    )
    assert result["directions"]["x"]["errors"]["hamiltonian"]["mae"] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Phase 15/16/17 — dataset mixing
# --------------------------------------------------------------------------- #
def test_classify_dataset_by_size():
    dataset = [{"id": "a", "n_atoms": 5}, {"id": "b", "n_atoms": 50}]
    small, large = mvs.classify_dataset_by_size(dataset, threshold_atoms=10)
    assert [s["id"] for s in small] == ["a"]
    assert [s["id"] for s in large] == ["b"]


def test_classify_dataset_by_size_missing_count():
    with pytest.raises(ValueError):
        mvs.classify_dataset_by_size([{"id": "a"}], threshold_atoms=10)


def test_make_mixed_dataset_manifest_modes():
    small = [{"id": f"s{i}"} for i in range(4)]
    large = [{"id": f"l{i}"} for i in range(4)]

    add = mvs.make_mixed_dataset_manifest(small, large, [0.0, 0.5, 1.0], mode="add", seed=1)
    assert add["partitions"][0]["n_selected"] == 4  # ratio 0 → just small
    assert add["partitions"][2]["n_selected"] == 8  # ratio 1 → small + all large

    replace = mvs.make_mixed_dataset_manifest(small, large, [0.0, 0.5, 1.0], mode="replace", seed=1)
    for partition in replace["partitions"]:
        assert partition["n_selected"] == 4  # total constant


def test_make_mixed_dataset_manifest_reproducible():
    small = [{"id": f"s{i}"} for i in range(3)]
    large = [{"id": f"l{i}"} for i in range(6)]
    a = mvs.make_mixed_dataset_manifest(small, large, [0.5], mode="add", seed=7)
    b = mvs.make_mixed_dataset_manifest(small, large, [0.5], mode="add", seed=7)
    assert a["partitions"][0]["selected_ids"] == b["partitions"][0]["selected_ids"]


def test_generate_mixed_dataset_configs(tmp_path):
    small = [{"id": "s0"}]
    large = [{"id": "l0"}]
    manifest = mvs.make_mixed_dataset_manifest(
        small, large, [0.0, 1.0], mode="add", output_path=tmp_path / "manifest.json", seed=0
    )
    written = mvs.generate_mixed_dataset_configs(manifest, tmp_path / "configs")
    assert len(written) == 2
    assert (tmp_path / "configs" / "D0.yaml").is_file()


# --------------------------------------------------------------------------- #
# Phase 18/19/20 — species transfer
# --------------------------------------------------------------------------- #
def test_inspect_species_support_c_only_add_h():
    report = mvs.inspect_species_support(
        {"supported_species": ["C"], "orbital_basis": {"C": ["2s", "2p"]}},
        new_species=["H"],
    )
    assert report.missing_species == ["H"]
    assert report.requires_new_embeddings is True
    assert report.status == "not_implemented"


def test_inspect_species_support_expandable():
    report = mvs.inspect_species_support(
        {"supported_species": ["C"], "expandable": True}, new_species="H"
    )
    assert report.status == "partially_supported"


def test_load_species_transfer_config_detects_new():
    config = load_species_transfer_config(
        {"species_transfer": {"base_species": ["C"], "new_species": ["C", "H"]}}
    )
    assert config["detected_new_species"] == ["H"]


def test_load_species_transfer_config_requires_base():
    with pytest.raises(SpeciesTransferConfigError):
        load_species_transfer_config({"species_transfer": {"new_species": ["H"]}})


def test_prepare_species_expansion_unsupported_raises_clearly():
    class DummyModel:
        pass

    with pytest.raises(NotImplementedError) as exc:
        prepare_species_expansion(DummyModel(), {"supported_species": ["C"]}, {"new_species": ["H"]})
    assert hasattr(exc.value, "report")


def test_prepare_species_expansion_delegates_to_hook():
    class ExpandableModel:
        def expand_species(self, old, new):
            return {"expanded": True}

    result = prepare_species_expansion(
        ExpandableModel(), {"supported_species": ["C"]}, {"new_species": ["H"]}
    )
    assert result["status"] == "delegated"


# --------------------------------------------------------------------------- #
# Phase 21 — species-pair errors
# --------------------------------------------------------------------------- #
def test_compute_error_by_species_pair():
    # 3 orbitals: atom0 (C) → orbital 0, atom1 (H) → orbitals 1,2.
    ref = mvs.MatrixData(
        values=np.zeros((3, 3)),
        target="hamiltonian",
        metadata={"orbital_atom_index": [0, 1, 1]},
    )
    pred = mvs.MatrixData(values=np.ones((3, 3)), target="hamiltonian")
    structure = BenchmarkStructure(
        symbols=["C", "H"], positions=np.zeros((2, 3)), cell=np.eye(3)
    )
    result = mvs.compute_error_by_species_pair(ref, pred, structure)
    assert set(result["pairs"]) == {"C-C", "C-H", "H-H"}
    assert result["pairs"]["C-C"]["mae"] == pytest.approx(1.0)


def test_compute_error_by_species_pair_warns_without_mapping():
    ref = mvs.MatrixData(values=np.zeros((3, 3)), target="hamiltonian")
    pred = mvs.MatrixData(values=np.ones((3, 3)), target="hamiltonian")
    structure = BenchmarkStructure(symbols=["C", "H"], positions=np.zeros((2, 3)), cell=np.eye(3))
    result = mvs.compute_error_by_species_pair(ref, pred, structure)
    assert result["pairs"] == {}
    assert result["warnings"]


# --------------------------------------------------------------------------- #
# Phase 22/23/24/25 — viewer payloads
# --------------------------------------------------------------------------- #
def test_prepare_matrix_plot_payload_raw_and_error():
    matrix = mvs.MatrixData(values=np.array([[1.0, 2.0], [3.0, 4.0]]), target="hamiltonian")
    error = np.array([[0.1, -0.1], [0.2, -0.2]])
    payload = mvs.prepare_matrix_plot_payload(matrix, error, target="hamiltonian")
    assert payload["matrix"]["shape"] == [2, 2]
    assert payload["error"]["max_abs_error"] == pytest.approx(0.2)
    assert "log_abs" in payload["scales"]


def test_build_matrix_viewer_payload():
    siesta = mvs.MatrixData(values=np.zeros((2, 2)), target="hamiltonian")
    g2m = mvs.MatrixData(values=np.ones((2, 2)), target="hamiltonian")
    payload = mvs.build_matrix_viewer_payload(target="hamiltonian", siesta=siesta, graph2mat=g2m)
    assert "graph2mat" in payload["available"]
    assert payload["metrics"]["graph2mat"]["mae"] == pytest.approx(1.0)
    assert "graph2mat_minus_siesta" in payload["differences"]


def test_build_derivative_viewer_payload():
    ml = mvs.MatrixData(values=np.ones((2, 2)), target="hamiltonian", metadata={"model": "graph2mat"})
    siesta = mvs.MatrixData(values=np.zeros((2, 2)), target="hamiltonian")
    payload = mvs.build_derivative_viewer_payload(
        target="hamiltonian",
        atom_index=3,
        direction="x",
        displacement=0.01,
        ml_derivative=ml,
        siesta_derivative=siesta,
    )
    assert payload["metrics"]["mae"] == pytest.approx(1.0)
    assert payload["direction"] == "x"


# --------------------------------------------------------------------------- #
# Phase 30 — dry-run pipeline
# --------------------------------------------------------------------------- #
def test_benchmark_dry_run_with_example_config():
    structure_path = REPO_ROOT / "Comparison" / "config" / "ml_vs_siesta_example_structure.fdf"
    config = parse_benchmark_config(_minimal_config_payload(str(structure_path)))
    summary = mvs.benchmark_dry_run(config, siesta_output_dir="/tmp/example")
    assert summary["ok"] is True
    assert summary["checks"]["supercell"]["detail"]["supercell_atoms"] == 50
    assert set(summary["checks"]["siesta_paths"]["detail"]) >= {"reference", "x_plus"}


def test_benchmark_dry_run_missing_structure_warns():
    config = parse_benchmark_config(_minimal_config_payload(None))
    summary = mvs.benchmark_dry_run(config)
    assert summary["checks"]["structure"]["ok"] is False
    assert summary["warnings"]
