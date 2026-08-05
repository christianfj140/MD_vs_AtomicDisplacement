from __future__ import annotations

import sys
import subprocess
from pathlib import Path

import numpy as np
import pytest


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "Comparison/scripts"))

import run_tbg_tight_binding as tb  # noqa: E402


def test_published_hopping_signs_and_values() -> None:
    intralayer = tb.hopping(np.array([1.4318286667]), np.array([0.0]))[0]
    vertical = tb.hopping(np.array([3.35]), np.array([3.35]))[0]
    assert -2.64 < intralayer < -2.62
    assert vertical == tb.V_PP_SIGMA_0


def test_runtime_guards_are_preventive(monkeypatch) -> None:
    assert tb.SOLVER_THREADS <= 4
    monkeypatch.setattr(tb, "free_disk_percent", lambda _path: 11.9)
    monkeypatch.setattr(tb, "cpu_package_temperature_c", lambda: 30.0)
    with pytest.raises(tb.ResourceGuardError, match="disk headroom"):
        tb.guard_resources("test", cooldown=False)
    monkeypatch.setattr(tb, "free_disk_percent", lambda _path: 20.0)
    monkeypatch.setattr(tb, "cpu_package_temperature_c", lambda: 82.0)
    with pytest.raises(tb.ResourceGuardError, match="preventive stop"):
        tb.guard_resources("test", cooldown=False)


def test_small_sparse_spectrum_and_inertia_match_dense() -> None:
    model = tb.PzTightBinding(tb.monolayer_geometry().tile(4, 0).tile(4, 1))
    k = [0.11, 0.37, 0.0]
    dense = np.linalg.eigvalsh(model.hk(k).toarray())
    assert model.hermiticity_error(k) < 1e-12
    reversed_dense = np.linalg.eigvalsh(model.hk([-value for value in k]).toarray())
    assert np.max(np.abs(dense - reversed_dense)) < 1e-9
    for sigma in (-0.37, 0.05, 1.31):
        result = model.solve(k, sigma, 12, vectors=True)
        assert result["states_below_sigma"] == int((dense < sigma).sum())
        expected = dense[result["first_index"] : result["first_index"] + 12]
        assert np.max(np.abs(expected - result["energies"])) < 1e-9
        assert result["inertia_diagnostic"]["symmetric_row_column_permutation"] is True


def test_mesh_observables_are_distinct_from_path_scope() -> None:
    occupied = 5
    first = np.array([occupied - 3, occupied - 3])
    windows = np.array(
        [
            [-0.08, -0.02, -0.01, 0.01, 0.02, 0.08],
            [-0.09, -0.03, -0.015, 0.012, 0.025, 0.09],
        ]
    )
    result = tb.mesh_observables_from_cache(windows, first, occupied, [1, 2, 1])
    assert result["status"] == "computed_from_cached_eigenvalues"
    assert np.isclose(result["global_manifold_width_eV"], 0.055)
    assert result["mesh"] == [1, 2, 1]
    # La etiqueta debe salir de la malla real, no de un 12x12 fijo.
    assert result["convergence_status"] == "single_1x2_mesh_not_cross_mesh_converged"


def test_partial_dos_never_publishes_truncated_edges() -> None:
    eigenvalues = np.array([[-0.4, -0.2, 0.0, 0.2, 0.3], [-0.35, -0.1, 0.0, 0.1, 0.45]])
    result = tb.partial_dos_from_cache(
        eigenvalues, 0.0, 10, broadening_eV=0.01, requested_window_eV=0.5
    )
    maximum = max(abs(point["energy_aligned_eV"]) for point in result["low_energy_dos"])
    assert np.isclose(maximum, result["published_half_width_eV"])
    assert maximum <= result["fully_covered_half_width_eV"] - 5 * result["broadening_eV"]
    assert all("dos_per_atom" in point for point in result["low_energy_dos"])


def test_ui_groups_tb_by_absolute_index_and_uses_linear_style() -> None:
    source = (REPO / "Comparison/ui/app.js").read_text(encoding="utf-8")
    helper = source[
        source.index("function ctSpectralDirectBandTraces") :
        source.index("function ctSpectralManifoldSplit")
    ]
    assert "point.absolute_band_index" in helper
    assert 'shape: "linear"' in helper
    assert 'model === "tight_binding"' in helper
    assert "shape: \"spline\"" not in helper
    script = r'''
const fs = require("fs");
const vm = require("vm");
global.methodDisplayLabel = (value) => value;
const source = fs.readFileSync(process.argv[2], "utf8");
vm.runInThisContext(source.slice(
  source.indexOf("function ctSpectralRowLabel"),
  source.indexOf("function ctSpectralManifoldSplit"),
));
const row = {model: "tight_binding", training_size: null, seed: null};
const points = [
  {k_index: 0, k_distance: 0, band_index: 0, solver_band_index: 0, absolute_band_index: 5580, energy_aligned_eV: 0},
  {k_index: 1, k_distance: 1, band_index: 0, solver_band_index: 0, absolute_band_index: 5581, energy_aligned_eV: 0.001},
  {k_index: 1, k_distance: 1, band_index: 1, solver_band_index: 1, absolute_band_index: 5580, energy_aligned_eV: 0.002},
];
const traces = ctSpectralDirectBandTraces(row, points, 4, 0.8, false);
if (traces.length !== 2) throw Error(`expected 2 absolute bands, got ${traces.length}`);
for (const trace of traces) {
  const absolute = new Set(trace.customdata.map((item) => item[0]));
  if (absolute.size !== 1) throw Error("trace mixed absolute band indices");
  if (trace.line.shape !== "linear") throw Error("TB trace is not linear");
}
if (traces[0].name.includes("null")) throw Error("null leaked into TB legend");
'''
    subprocess.run(
        ["node", "-", str(REPO / "Comparison/ui/app.js")],
        input=script,
        text=True,
        check=True,
    )


def test_target_contract_and_hashes_are_deterministic() -> None:
    geometry = tb.read_geometry(tb.TARGET_FDF)
    contract = tb.validate_target_geometry(geometry)
    assert contract["atoms_per_layer"] == [5582, 5582]
    assert contract["carbon_only"] is True
    first = tb.stage_contract("test", value=1)
    second = tb.stage_contract("test", value=1)
    assert first["contract_sha256"] == second["contract_sha256"]


def test_pipeline_merges_one_tb_row_and_confines_artifacts(tmp_path, monkeypatch) -> None:
    import pipeline_ui

    results = tmp_path / "results"
    campaign = results / "graphene_hbn_magic_angle_spectral"
    pure = results / "tbg_pure_graph2mat"
    tight_binding = results / "tbg_tight_binding"
    for root, payload in (
        (campaign, {"spectra": [{"model": "graph2mat", "material_system": "legacy"}]}),
        (pure, {"spectra": [{"model": "graph2mat", "material_system": "pure_tbg"}]}),
        (tight_binding, {"spectra": [{"model": "tight_binding", "material_system": "pure_tbg"}]}),
    ):
        (root / "summary").mkdir(parents=True)
        (root / "summary/spectral_results.json").write_text(
            __import__("json").dumps(payload), encoding="utf-8"
        )
    (tight_binding / "stages").mkdir()
    (tight_binding / "stages/preflight.json").write_text('{"status":"passed"}', encoding="utf-8")
    monkeypatch.setattr(pipeline_ui, "RESULTS_ROOT", results)
    runner = pipeline_ui.MoireSpectralCampaignRunner()
    spectra = runner.results()["summary"]["spectra"]
    assert sum(row.get("model") == "tight_binding" for row in spectra) == 1
    artifact = tight_binding / "stages/preflight.json"
    assert runner.artifact_path(str(artifact)) == artifact
    try:
        runner.artifact_path(str(tight_binding / "stages/mesh_eigenvalues.npz"))
    except RuntimeError as error:
        assert "Tipo de artefacto" in str(error)
    else:
        raise AssertionError("binary TB cache must not be downloadable through the JSON endpoint")


# --------------------------------------------------------------------------------------
# Regresiones de la auditoría 2026-08-05


def test_single_atom_lattice_translation_leaves_the_spectrum_invariant() -> None:
    """P2-01: con reach=1 (el de la celda moiré) un átomo fuera de celda perdía enlaces.

    Antes del wrap, desplazar el átomo 0 por -a2 quitaba 56 enlaces y movía el espectro
    1.4 eV. Trasladar un átomo por un vector de red describe el mismo cristal.
    """
    base_geometry = tb.read_geometry(REPO / "materials/bilayer_graphene_AB/RUN.fdf")
    supercell = base_geometry.tile(3, 0).tile(3, 1)
    reference_model = tb.PzTightBinding(supercell)
    assert reference_model.periodic_reach == [1, 1]
    k = [0.11, 0.37, 0.0]
    reference = np.linalg.eigvalsh(reference_model.hk(k).toarray())
    cell = np.asarray(supercell.cell)
    for translation in (cell[0], -cell[1], cell[0] - 2 * cell[1]):
        moved = supercell.copy()
        moved.xyz[0] = moved.xyz[0] + translation
        model = tb.PzTightBinding(moved)
        assert model.bond_count == reference_model.bond_count
        spectrum = np.linalg.eigvalsh(model.hk(k).toarray())
        assert np.max(np.abs(spectrum - reference)) < 1e-10


def test_target_geometry_is_unaffected_by_the_wrap() -> None:
    """El wrap no debe alterar la geometría objetivo, que ya está dentro de la celda."""
    model = tb.PzTightBinding(tb.read_geometry(tb.TARGET_FDF))
    assert model.wrap_shift_max_ang < 1e-9
    assert model.bond_count == 679208


def test_dos_contract_binds_the_fermi_level_and_the_spectral_cache(monkeypatch) -> None:
    """P1-02: cambiar E_F debe invalidar la DOS, no dejarla reutilizable."""
    import argparse

    args = argparse.Namespace(mesh=12, states=64, band_states=32)
    real = tb.read_stage("neutrality")
    if not real:
        pytest.skip("no neutrality stage available")
    baseline = tb.contract_for_stage("dos", args)
    monkeypatch.setattr(
        tb, "read_stage",
        lambda name: ({**real, "energy_eV": 9.0} if name == "neutrality" else real),
    )
    shifted = tb.contract_for_stage("dos", args)
    assert baseline["contract_sha256"] != shifted["contract_sha256"]
    assert "fermi_level_eV" in baseline["settings"]
    assert "mesh_cache_sha256" in baseline["settings"]


def test_neutrality_contract_binds_sigma_from_preflight(monkeypatch) -> None:
    """No reportado por la auditoría: sigma viene de preflight y no ataba el contrato."""
    import argparse

    args = argparse.Namespace(mesh=12, states=64, band_states=32)
    real = tb.read_stage("preflight")
    if not real:
        pytest.skip("no preflight stage available")
    baseline = tb.contract_for_stage("neutrality", args)
    monkeypatch.setattr(
        tb, "read_stage",
        lambda name: ({**real, "sigma_eV": 5.0} if name == "preflight" else real),
    )
    shifted = tb.contract_for_stage("neutrality", args)
    assert baseline["contract_sha256"] != shifted["contract_sha256"]


def test_every_stage_contract_carries_an_algorithm_version() -> None:
    """P1-03: sin versión de algoritmo, un cambio numérico no invalida las cachés."""
    contract = tb.stage_contract("probe", setting=1)
    assert contract["algorithm_version"] == tb.ALGORITHM_VERSION


def test_mesh_cache_without_signature_is_rejected(tmp_path, monkeypatch) -> None:
    """P1-03: un NPZ ajeno con las mismas dimensiones no debe aceptarse."""
    monkeypatch.setattr(tb, "RESULTS", tmp_path)
    (tmp_path / "stages").mkdir(parents=True)
    target = tmp_path / "stages" / tb.MESH_CACHE
    np.savez(target, eigenvalues=np.zeros((2, 4)), first_indices=np.zeros(2, dtype=int))
    with pytest.raises(SystemExit, match="no signature"):
        tb.load_mesh_cache()
    tb.atomic_savez(
        target,
        eigenvalues=np.zeros((2, 4)),
        first_indices=np.zeros(2, dtype=int),
        signature=np.asarray("abc"),
    )
    with pytest.raises(SystemExit, match="does not match"):
        tb.load_mesh_cache("expected-other-signature")
    assert tb.load_mesh_cache("abc")["signature"] == "abc"


def test_aggregate_does_not_rewrite_upstream_stages() -> None:
    """P1-04: agregar debe leer las etapas, nunca resellarlas con el SHA actual."""
    source = (REPO / "Comparison/scripts/run_tbg_tight_binding.py").read_text(encoding="utf-8")
    body = source[source.index("def stage_aggregate"): source.index("def self_test")]
    for stage in ("neutrality.json", "bands.json", "preflight.json", "dos.json"):
        assert f'atomic_write_json(RESULTS / "stages" / "{stage}"' not in body
    assert "aggregation_implementation" in body
    assert "source_implementation" in body


def test_atomic_write_rejects_non_finite_numbers(tmp_path) -> None:
    """P3-03: NaN no es JSON válido; publicarlo esconde un fallo numérico aguas arriba."""
    with pytest.raises(ValueError):
        tb.atomic_write_json(tmp_path / "bad.json", {"value": float("nan")})
    assert not list(tmp_path.iterdir())  # el temporal no queda huérfano


def test_gap_signs_ignore_values_below_the_residual() -> None:
    """P2-02: el signo de un gap menor que el residuo no significa nada."""
    occupied = 5
    indices = np.arange(occupied - 3, occupied + 3)

    def probe(gap: float) -> dict:
        energies = np.array([-0.08, -0.02, -0.01, -0.01 + gap, 0.02, 0.08])
        return {
            label: {"energies": energies, "first_index": occupied - 3}
            for label in ("Γ", "K", "M")
        }

    reference, candidate = probe(1e-12), probe(-1e-12)
    strict = tb.compare_probe_spectra(reference, candidate, occupied)
    tolerant = tb.compare_probe_spectra(
        reference, candidate, occupied, sign_tolerance_eV=1e-9
    )
    assert strict["gap_signs_preserved"] is False
    assert tolerant["gap_signs_preserved"] is True
    assert "sampled_indirect_neutrality_gap_eV" in tolerant["gaps_below_sign_tolerance"]
    assert indices.size == 6


def test_ui_labels_and_benchmarks_are_honest() -> None:
    """P2-05/07/08: sin N0·s0, sin NaN en hover ortogonal, benchmark con v_F medido."""
    source = (REPO / "Comparison/ui/app.js").read_text(encoding="utf-8")
    label = source[source.index("function ctSpectralRowLabel"):]
    label = label[: label.index("\n}\n")]
    assert "row.training_size != null" in label and "row.seed != null" in label
    assert "hbar_v_fermi_eV_ang" in source and "Benchmark monocapa" in source
    assert "base ortogonal, S = I" in source
    html = (REPO / "Comparison/ui/index.html").read_text(encoding="utf-8")
    benchmark = html[html.index('id="ct-spectral-dirac-reach"'):]
    benchmark = benchmark[: benchmark.index("</select>")]
    assert '<option value="0" selected>' in benchmark  # oculto por defecto


def test_inertia_certification_reads_the_sign_it_certifies() -> None:
    """El guard debe certificar sign(Re d_i) por pivote, no contra el pivote mayor.

    El criterio global rechazaba 2 de 144 puntos k de producción cuyo conteo se verificó
    correcto contra diagonalización densa (5580 en ambos). El margen por pivote observado
    en la malla es 1.5e-6, seis órdenes por debajo de la ambigüedad.
    """
    source = (REPO / "Comparison/scripts/run_tbg_tight_binding.py").read_text(encoding="utf-8")
    body = source[source.index("def solve"): source.index("def read_geometry")]
    assert "pivot_sign_margin" in body
    assert "np.abs(diagonal_u.imag) / np.maximum(np.abs(diagonal_u.real)" in body
    # El criterio global ya no puede ser el que aborta.
    assert "relative_pivot_imaginary > 1e-7" not in body

    model = tb.PzTightBinding(tb.monolayer_geometry().tile(4, 0).tile(4, 1))
    k = [0.11, 0.37, 0.0]
    dense = np.linalg.eigvalsh(model.hk(k).toarray())
    for sigma in (-0.37, 0.05, 1.31):
        result = model.solve(k, sigma, 12)
        assert result["states_below_sigma"] == int((dense < sigma).sum())
        diagnostic = result["inertia_diagnostic"]
        assert diagnostic["pivot_sign_margin"] < 1e-3
        assert diagnostic["smallest_absolute_real_pivot"] > 0.0
