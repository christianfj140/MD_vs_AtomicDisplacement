from __future__ import annotations

import re
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
global.spectralModelStyle = () => ({color: "#0c5da5", dash: "dash", width: 2.4});
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


def test_direct_band_limit_never_hides_one_side_of_the_spectrum() -> None:
    """Recortar bandas por |E| medio borraba el lado positivo de Graph2Mat.

    En Graph2Mat `band_index` es el rango energético dentro de la ventana shift-invert,
    no una banda física: los rangos bajos bajan mucho en negativo, ganaban la media y se
    perdían los estados de +8 a +49 meV. El recorte debe ser simétrico.
    """
    summary = REPO / "Comparison/results/tbg_pure_graph2mat/summary/spectral_results.json"
    if not summary.exists():
        pytest.skip("no Graph2Mat summary available")
    script = r'''
const fs = require("fs");
const vm = require("vm");
global.methodDisplayLabel = (value) => value;
global.spectralModelStyle = () => ({color: "#0c5da5", dash: "dash", width: 2.4});
const source = fs.readFileSync(process.argv[2], "utf8");
vm.runInThisContext(source.slice(
  source.indexOf("function ctSpectralRowLabel"),
  source.indexOf("function ctSpectralManifoldSplit"),
));
const row = JSON.parse(fs.readFileSync(process.argv[3], "utf8")).spectra[0];
for (const windowMeV of [25, 50, 75]) {
  const points = (row.bands || []).filter((point) => {
    const energy = 1000 * Number(point.energy_aligned_eV ?? point.energy_eV);
    return Number.isFinite(energy) && Math.abs(energy) <= windowMeV;
  });
  for (const limit of [0, 4, 6, 8]) {
    const traces = ctSpectralDirectBandTraces(row, points, 4, 0.8, false, limit);
    const values = traces.flatMap((trace) => trace.y.filter((value) => value !== null));
    if (!values.some((value) => value > 0.5)) {
      throw Error(`ventana ${windowMeV} limite ${limit}: sin bandas de energia positiva`);
    }
    if (!values.some((value) => value < -0.5)) {
      throw Error(`ventana ${windowMeV} limite ${limit}: sin bandas de energia negativa`);
    }
    if (limit > 0 && traces.length > limit) throw Error("el limite no se respeta");
  }
}
'''
    subprocess.run(
        ["node", "-", str(REPO / "Comparison/ui/app.js"), str(summary)],
        input=script, text=True, check=True,
    )
    html = (REPO / "Comparison/ui/index.html").read_text(encoding="utf-8")
    control = html[html.index('id="ct-spectral-visible-bands"'):]
    control = control[: control.index("</select>")]
    # Por defecto no se debe ocultar nada en el gráfico científico principal.
    assert '<option value="0" selected>' in control


def test_api_compresses_large_payloads(tmp_path) -> None:
    """El endpoint espectral son decenas de MB y congelaba la UI al cargar."""
    import gzip as gzip_module
    import json as json_module
    import threading
    import urllib.request
    from http.server import HTTPServer

    import pipeline_ui

    payload = {"rows": [{"index": i, "value": i * 1.5, "label": "x" * 40} for i in range(20000)]}
    compact = json_module.dumps(payload, separators=(",", ":")).encode()
    assert len(compact) > pipeline_ui.GZIP_MINIMUM_BYTES

    class Handler(pipeline_ui.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            pipeline_ui.json_response(self, payload)

        def log_message(self, *_args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_port}/"
    try:
        with urllib.request.urlopen(urllib.request.Request(url)) as response:
            plain, plain_encoding = response.read(), response.headers.get("Content-Encoding")
        request = urllib.request.Request(url, headers={"Accept-Encoding": "gzip"})
        with urllib.request.urlopen(request) as response:
            packed, packed_encoding = response.read(), response.headers.get("Content-Encoding")
    finally:
        server.shutdown()

    assert plain_encoding is None
    assert packed_encoding == "gzip"
    assert len(packed) < len(plain) / 2
    assert json_module.loads(gzip_module.decompress(packed)) == payload
    assert json_module.loads(plain) == payload
    # Compacto, no pretty-printed: el indent=2 añadía ~50 % de bytes.
    assert b'": ' not in plain[:200]


def test_spectral_plots_use_a_paper_style_and_contrasting_model_colours() -> None:
    """Estilo tipo SciencePlots y TB en azul frente a Graph2Mat en rojo."""
    source = (REPO / "Comparison/ui/app.js").read_text(encoding="utf-8")
    helper = source[source.index("function scienceAxis"): source.index("function spectralModelStyle")]
    for expected in ('mirror: "ticks"', 'ticks: "inside"', "showgrid: false", "minor:"):
        assert expected in helper, expected
    assert "serif" in source[source.index("SCIENCE_PLOT_FONT_FAMILY"):][:400]
    assert "scienceLayout({" in source  # el helper compartido lo aplica

    style = source[source.index("const SPECTRAL_MODEL_STYLE"):]
    style = style[: style.index("};")]
    # Tight binding en rojo y Graph2Mat en azul, y cada uno con su propio trazo.
    tight_binding = style[style.index("tight_binding:"): style.index("graph2mat:")]
    graph2mat = style[style.index("graph2mat:"): style.index("deeph:")]
    assert "#e8000b" in tight_binding and '"dash"' in tight_binding
    assert "#0c5da5" in graph2mat and '"solid"' in graph2mat

    # Todo lo legible al doble del tamaño por defecto de Plotly (12 px).
    for name, minimum in (
        ("SCIENCE_PLOT_TICK_SIZE", 20),
        ("SCIENCE_PLOT_AXIS_TITLE_SIZE", 24),
        ("SCIENCE_PLOT_TITLE_SIZE", 24),
        ("SCIENCE_PLOT_LEGEND_SIZE", 18),
    ):
        value = int(source.split(f"const {name} = ")[1].split(";")[0])
        assert value >= minimum, f"{name} = {value}, se esperaba >= {minimum}"

    css = (REPO / "Comparison/ui/styles.css").read_text(encoding="utf-8")
    card = css[css.index(".plot-card {"): css.index(".plot-card.js-plotly-plot")]
    height = int(card.split("min-height:")[1].split("px")[0].strip())
    assert height >= 860, f"las gráficas deben ser más altas que las 480px previas: {height}"
    assert ".plot-card.is-loading::after" in css
    assert 'content: "Cargando…"' in css


def test_api_drops_payload_no_client_reads() -> None:
    """`band_representations` son 6.3 MB que ningún cliente lee; fuera de la respuesta."""
    import pipeline_ui

    for client in ("Comparison/ui/app.js", "Comparison/ui/index.html", "Comparison/ui/lite.html"):
        assert "band_representations" not in (REPO / client).read_text(encoding="utf-8")
    trimmed = pipeline_ui.drop_unused_spectral_payload(
        {"spectra": [{"bands": [1], "band_representations": {"raw": [1, 2, 3]}}], "other": 1}
    )
    assert trimmed["spectra"] == [{"bands": [1]}]
    assert trimmed["other"] == 1
    # No debe romperse con resúmenes sin `spectra`.
    assert pipeline_ui.drop_unused_spectral_payload({"a": 1}) == {"a": 1}


def test_plot_loading_indicator_always_clears() -> None:
    """El indicador debe quitarse aunque la petición falle, o la UI se queda colgada."""
    source = (REPO / "Comparison/ui/app.js").read_text(encoding="utf-8")
    body = source[source.index("async function ctSpectralRefresh()"):]
    body = body[: body.index("async function ctSpectralRefreshInner")]
    assert "ctSpectralBusy(true)" in body
    assert "finally" in body and "ctSpectralBusy(false)" in body


def test_dos_reserves_red_for_the_tight_binding_reference() -> None:
    """Graph2Mat entra por la rama projected_dos, así que no toma el color de modelo.

    Sus cinco componentes salían del ciclo Tol, cuyo azul competía con la línea del TB.
    Ahora llevan color fijo y ninguna es roja: el rojo es solo del tight binding.
    """
    source = (REPO / "Comparison/ui/app.js").read_text(encoding="utf-8")
    block = source[source.index("const PDOS_COMPONENTS"):]
    block = block[: block.index("];")]
    colours = re.findall(r'"(#[0-9a-fA-F]{6})"', block)
    assert len(colours) == 5, colours

    def channels(value: str) -> tuple[int, int, int]:
        return tuple(int(value[index:index + 2], 16) for index in (1, 3, 5))

    for colour in colours:
        red, green, blue = channels(colour)
        assert not (red > 150 and red > green + 60 and red > blue + 60), (
            f"{colour} es rojizo y compite con la serie del tight binding"
        )
    tight_binding_red = channels("#e8000b")
    assert tight_binding_red[0] > 200 and tight_binding_red[1] < 80
