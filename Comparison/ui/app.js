const pipelines = [
  { key: "md", label: "MD", logId: "md-log", dotId: "md-status-dot", textId: "md-status-text" },
  {
    key: "atom_displacement",
    label: "AtomDisplacement",
    logId: "atom-displacement-log",
    dotId: "atom-displacement-status-dot",
    textId: "atom-displacement-status-text",
  },
];

const state = {
  offsets: Object.fromEntries(pipelines.map((pipeline) => [pipeline.key, 0])),
  experimentOffset: 0,
  polling: null,
  plotsEnabled: false,
  plotData: null,
  fcMaxPerDisplacement: null,
};

function showToast(message) {
  const toast = document.getElementById("toast");
  toast.textContent = message;
  toast.classList.add("visible");
  setTimeout(() => toast.classList.remove("visible"), 2600);
}

async function request(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const payload = await response.json();
  if (!response.ok || payload.error) {
    throw new Error(payload.error || response.statusText);
  }
  return payload;
}

function statusText(status) {
  if (status.running) {
    const elapsed = status.elapsed_seconds == null ? "" : ` · ${formatDuration(status.elapsed_seconds)}`;
    const eta = status.eta_seconds == null ? "" : ` · ETA ${formatDuration(status.eta_seconds)}`;
    return `Running${elapsed}${eta}`;
  }
  if (status.returncode == null || status.returncode === 0) return "Idle";
  return `Exit ${status.returncode}`;
}

function formatDuration(value) {
  if (value == null || !Number.isFinite(Number(value))) return "sin estimacion";
  const total = Math.max(0, Math.round(Number(value)));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const seconds = total % 60;
  if (hours > 0) return `${hours}h ${String(minutes).padStart(2, "0")}m ${String(seconds).padStart(2, "0")}s`;
  if (minutes > 0) return `${minutes}m ${String(seconds).padStart(2, "0")}s`;
  return `${seconds}s`;
}

function updatePipelineStatus(status) {
  const pipeline = pipelines.find((item) => item.key === status.key);
  if (!pipeline) return;
  const dot = document.getElementById(pipeline.dotId);
  const text = document.getElementById(pipeline.textId);
  dot.classList.toggle("running", Boolean(status.running));
  dot.classList.toggle("error", status.returncode != null && status.returncode !== 0);
  text.textContent = statusText(status);
}

function updateGlobalStatus(payload) {
  const statuses = Object.values(payload.pipelines || {});
  const running = statuses.some((status) => status.running);
  const failed = statuses.some((status) => status.returncode != null && status.returncode !== 0);
  const dot = document.getElementById("global-status-dot");
  const text = document.getElementById("global-status-text");
  dot.classList.toggle("running", running);
  dot.classList.toggle("error", !running && failed);
  text.textContent = running ? "Running" : failed ? "Finished with errors" : "Idle";
  statuses.forEach(updatePipelineStatus);
}

async function runAll() {
  for (const pipeline of pipelines) {
    state.offsets[pipeline.key] = 0;
    document.getElementById(pipeline.logId).textContent = "";
  }
  const payload = await request("/api/run", { method: "POST", body: "{}" });
  updateGlobalStatus(payload);
  if (payload.errors && Object.keys(payload.errors).length) {
    showToast(Object.values(payload.errors).join(" | "));
  } else {
    showToast("Both pipelines started");
  }
}

async function stopAll() {
  const payload = await request("/api/run/stop", { method: "POST", body: "{}" });
  updateGlobalStatus(payload);
  showToast("Stop requested");
}

async function pollStatus() {
  const payload = await request("/api/run/status");
  updateGlobalStatus(payload);
}

async function pollLogs() {
  await Promise.all(
    pipelines.map(async (pipeline) => {
      const payload = await request(
        `/api/run/logs?pipeline=${pipeline.key}&since=${state.offsets[pipeline.key]}`,
      );
      state.offsets[pipeline.key] = payload.offset;
      updatePipelineStatus(payload.status);
      if (payload.lines.length) {
        const output = document.getElementById(pipeline.logId);
        output.textContent += payload.lines.join("");
        output.scrollTop = output.scrollHeight;
      }
    }),
  );
  await pollStatus();
  await pollExperimentLogs();
}

function parseSizesInput(id) {
  return document
    .getElementById(id)
    .value.split(/[;,]/)
    .map((item) => item.trim())
    .filter(Boolean)
    .map((item) => Number(item));
}

function parseTextListInput(id) {
  return document
    .getElementById(id)
    .value.split(/[;,]/)
    .map((item) => item.trim())
    .filter(Boolean);
}

function parsePositiveIntegerListInput(id) {
  return parseTextListInput(id)
    .map((item) => Number(item))
    .filter((value) => Number.isInteger(value) && value > 0);
}

function parseSplitRatios() {
  return {
    train: Number(document.getElementById("split-train").value),
    validation: Number(document.getElementById("split-validation").value),
    test: Number(document.getElementById("split-test").value),
  };
}

function parseFcDisplacementOptionsText() {
  const rows = document
    .getElementById("fc-displacement-options")
    .value.split(/\n+/)
    .map((item) => item.trim())
    .filter(Boolean);
  const options = {};
  for (const [index, row] of rows.entries()) {
    const parts = row.split(":");
    if (parts.length < 2) {
      throw new Error(
        `Displacement options: la fila ${index + 1} no tiene el formato "magnitud: n1, n2, ...".`,
      );
    }
    const displacement = parts.shift().trim();
    if (!displacement) {
      throw new Error(`Displacement options: la fila ${index + 1} tiene la magnitud vacia.`);
    }
    if (Object.prototype.hasOwnProperty.call(options, displacement)) {
      throw new Error(
        `Displacement options: la magnitud "${displacement}" esta repetida; usa una sola fila por magnitud.`,
      );
    }
    const counts = parts
      .join(":")
      .split(/[;,]/)
      .map((item) => Number(item.trim()))
      .filter((value) => Number.isInteger(value) && value > 0);
    if (!counts.length) {
      throw new Error(
        `Displacement options: la fila ${index + 1} no tiene enteros positivos (ej. 2, 3, 4).`,
      );
    }
    options[displacement] = counts;
  }
  return options;
}

function formatFcDisplacementOptions(options) {
  return Object.entries(options || {})
    .map(([displacement, counts]) => `${displacement}: ${(counts || []).join(", ")}`)
    .join("\n");
}

function fcAlignedSpecs() {
  const options = parseFcDisplacementOptionsText();
  const entries = Object.entries(options);
  if (!entries.length) return [];
  const lengths = entries.map(([, counts]) => counts.length);
  const uniqueLengths = new Set(lengths);
  if (uniqueLengths.size !== 1) {
    return [];
  }
  const total = lengths[0] || 0;
  const specs = [];
  for (let index = 0; index < total; index += 1) {
    specs.push({
      size: entries.reduce((sum, [, counts]) => sum + counts[index], 0),
    });
  }
  return specs;
}

function updateAtomSizesFromFcPlan() {
  try {
    const specs = fcAlignedSpecs();
    const sizes = specs.map((spec) => spec.size);
    const sizesText = sizes.join(", ");
    document.getElementById("atom-sizes").value = sizesText;
    document.getElementById("md-sizes").value = sizesText;
    document.getElementById("atom-combination-count").value = specs.length
      ? `${specs.length} aligned datasets`
      : "invalid aligned lists";
  } catch (_error) {
    document.getElementById("atom-sizes").value = "";
    document.getElementById("md-sizes").value = "";
    document.getElementById("atom-combination-count").value = "invalid aligned lists";
  }
}

async function loadFcConfig() {
  const config = await request("/api/atom-fc-config");
  state.fcMaxPerDisplacement = config.max_per_displacement;
  const limit = document.getElementById("fc-limit");
  limit.textContent =
    config.max_per_displacement == null
      ? "Max per displacement: -"
      : `Max per displacement: ${config.max_per_displacement}`;
  document.getElementById("fc-displacement-options").value = formatFcDisplacementOptions(
    config.displacement_options || { "0.05 Ang": [2, 4, 6] },
  );
  document.getElementById("fc-max-datasets").value = config.max_datasets ?? 100;
  document.getElementById("fc-random-seed").value = config.random_seed ?? 42;
  document.getElementById("split-train").value = config.splits?.train ?? 0.8;
  document.getElementById("split-validation").value = config.splits?.validation ?? 0.1;
  document.getElementById("split-test").value = config.splits?.test ?? 0.1;
  updateAtomSizesFromFcPlan();
}

function updateExperimentStatus(status) {
  const text = document.getElementById("experiment-status-text");
  const root = document.getElementById("experiment-results-root");
  if (status.running && status.current) {
    const elapsed = formatDuration(status.current.elapsed_seconds);
    const eta = formatDuration(status.current.eta_seconds);
    const label = status.current.dataset_label || `dataset_${status.current.size}`;
    text.textContent = `${status.current.pipeline} ${label} · ${elapsed} · ETA ${eta}`;
  } else if (status.running) {
    text.textContent = "Running";
  } else if (status.returncode == null || status.returncode === 0) {
    text.textContent = "Idle";
  } else {
    text.textContent = `Exit ${status.returncode}`;
  }
  root.textContent = status.results_root || "Comparison/results";
  renderExperimentResults(status.results || []);
}

function renderExperimentResults(results) {
  const container = document.getElementById("experiment-results");
  container.innerHTML = "";
  for (const result of results) {
    const item = document.createElement("div");
    item.className = "result-pill";
    const label = result.dataset_label || `dataset_${result.dataset_size}`;
    item.innerHTML = `
      <strong>${result.pipeline} ${label}</strong>
      <span>${result.predicted_hamiltonians} predicted Hamiltonians</span>
      <span>${result.siesta_hamiltonians} SIESTA Hamiltonians</span>
      <code>${result.result_dir}</code>
    `;
    container.appendChild(item);
  }
}

async function runExperiment() {
  const mdSizes = parseSizesInput("md-sizes");
  const fcDisplacementOptions = parseFcDisplacementOptionsText();
  if (!Object.keys(fcDisplacementOptions).length) {
    throw new Error("Define al menos una magnitud FC con opciones.");
  }
  const badDatasets = fcAlignedSpecs()
    .map((spec) => spec.size)
    .filter((size) => !Number.isInteger(size) || size < 3);
  if (badDatasets.length) {
    throw new Error(
      `Con train/validation/test se requieren datasets >= 3. Tamaños invalidos: ${badDatasets.join(", ")}.`,
    );
  }
  const atomSizes = parseSizesInput("atom-sizes");
  const splitRatios = parseSplitRatios();
  const randomSeed = Number(document.getElementById("fc-random-seed").value);
  const maxDatasets = Number(document.getElementById("fc-max-datasets").value);
  state.experimentOffset = 0;
  document.getElementById("experiment-log").textContent = "";
  const payload = await request("/api/experiment", {
    method: "POST",
    body: JSON.stringify({
      md_sizes: mdSizes,
      atom_sizes: atomSizes,
      fc_displacement_options: fcDisplacementOptions,
      sync_md_sizes: true,
      splits: splitRatios,
      random_seed: Number.isInteger(randomSeed) ? randomSeed : 42,
      max_datasets: Number.isInteger(maxDatasets) ? maxDatasets : 100,
    }),
  });
  updateExperimentStatus(payload);
  showToast("Experiment started");
}

async function stopExperiment() {
  const payload = await request("/api/experiment/stop", { method: "POST", body: "{}" });
  updateExperimentStatus(payload);
  showToast("Experiment stop requested");
}

async function pollExperimentLogs() {
  const payload = await request(`/api/experiment/logs?since=${state.experimentOffset}`);
  state.experimentOffset = payload.offset;
  updateExperimentStatus(payload.status);
  if (payload.lines.length) {
    const output = document.getElementById("experiment-log");
    output.textContent += payload.lines.join("");
    output.scrollTop = output.scrollHeight;
  }
}

async function loadResults() {
  const results = await request("/api/results");
  const grid = document.getElementById("results-grid");
  grid.innerHTML = "";
  for (const pipeline of pipelines) {
    const data = results[pipeline.key];
    const panel = document.createElement("section");
    panel.className = "panel result-row";
    panel.innerHTML = `
      <div>
        <p class="eyebrow">Pipeline</p>
        <h3>${pipeline.label}</h3>
      </div>
      <p><strong>${data.predictions}</strong> prediction files</p>
      <p><strong>Metrics:</strong> ${data.metrics_exists ? "available" : "missing"}</p>
      <code>${data.metrics}</code>
      <code>${data.prediction_glob}</code>
    `;
    grid.appendChild(panel);
  }
  const archived = results.archived || {};
  for (const pipeline of pipelines) {
    const items = archived[pipeline.key] || [];
    const panel = document.createElement("section");
    panel.className = "panel result-row";
    panel.innerHTML = `
      <div>
        <p class="eyebrow">Archived</p>
        <h3>${pipeline.label}</h3>
      </div>
      <p><strong>${items.length}</strong> archived experiment runs</p>
      <code>Comparison/results/${pipeline.key === "md" ? "results_md" : "results_atomdisp"}</code>
    `;
    grid.appendChild(panel);
  }
  if (state.plotsEnabled) {
    await loadPlots();
  }
}

function metricValue(run, group, metric) {
  const value = run?.means?.[group]?.[metric];
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function sampleMetricValues(run, group, metric) {
  return (run?.samples?.[group] || [])
    .map((row) => row[metric])
    .filter((value) => typeof value === "number" && Number.isFinite(value));
}

function missingFermiSummary(runs) {
  const pieces = [];
  for (const [pipeline, items] of groupedRuns(runs)) {
    const label = pipelines.find((item) => item.key === pipeline)?.label || pipeline;
    const missing = items.reduce(
      (sum, run) => sum + Number(run.diagnostics?.missing_fermi_level_samples || 0),
      0,
    );
    const unavailable = items.reduce(
      (sum, run) => sum + Number(run.diagnostics?.unavailable_fermi_source_samples || 0),
      0,
    );
    const available = items.reduce(
      (sum, run) => sum + Number(run.diagnostics?.fermi_window_samples || 0),
      0,
    );
    const totalMissing = Math.max(missing, unavailable);
    if (totalMissing > 0) {
      pieces.push(`${label}: ${totalMissing} samples sin Fermi SIESTA (${available} con metrica Fermi)`);
    }
  }
  return pieces.length ? ` | ${pieces.join(" | ")}` : "";
}

function groupedRuns(runs) {
  const groups = new Map();
  for (const run of runs) {
    if (!groups.has(run.pipeline)) {
      groups.set(run.pipeline, []);
    }
    groups.get(run.pipeline).push(run);
  }
  for (const items of groups.values()) {
    items.sort((a, b) => a.dataset_size - b.dataset_size || String(a.run_id).localeCompare(String(b.run_id)));
  }
  return groups;
}

function lineTraces(runs, group, metrics) {
  const traces = [];
  for (const [pipeline, items] of groupedRuns(runs)) {
    const label = pipelines.find((item) => item.key === pipeline)?.label || pipeline;
    for (const metric of metrics) {
      const points = items
        .map((run) => ({ x: run.dataset_size, y: metricValue(run, group, metric.key), text: run.run_id }))
        .filter((point) => point.y != null);
      if (!points.length) continue;
      traces.push({
        type: "scatter",
        mode: "lines+markers",
        name: metrics.length > 1 ? `${label} · ${metric.label}` : label,
        x: points.map((point) => point.x),
        y: points.map((point) => point.y),
        text: points.map((point) => point.text),
        hovertemplate: "dataset %{x}<br>%{y:.4g}<br>run %{text}<extra>%{fullData.name}</extra>",
      });
    }
  }
  return traces;
}

function plotLayout(title, yTitle, extra = {}) {
  return {
    title: { text: title, x: 0.02, xanchor: "left", font: { size: 15 } },
    margin: { l: 56, r: 18, t: 46, b: 48 },
    paper_bgcolor: "#ffffff",
    plot_bgcolor: "#ffffff",
    xaxis: { title: "Dataset size", gridcolor: "#edf1f4", zeroline: false },
    yaxis: { title: yTitle, gridcolor: "#edf1f4", zeroline: false },
    legend: { orientation: "h", y: -0.25 },
    font: { family: "Inter, sans-serif", color: "#17202a" },
    ...extra,
  };
}

function emptyPlotAnnotation(message) {
  return {
    text: message,
    xref: "paper",
    yref: "paper",
    x: 0.5,
    y: 0.5,
    showarrow: false,
    font: { size: 13, color: "#6b7280" },
  };
}

function renderLinePlot(id, runs, group, metrics, title, yTitle) {
  const traces = lineTraces(runs, group, metrics);
  const layout = plotLayout(title, yTitle);
  if (!traces.length) {
    layout.annotations = [emptyPlotAnnotation("No hay valores finitos para esta metrica.")];
  }
  Plotly.react(id, traces, layout, { responsive: true, displaylogo: false });
}

function renderBoxPlot(id, runs) {
  const traces = [];
  for (const run of runs) {
    const spectral = sampleMetricValues(run, "spectral", "fermi_window_rmse_eV");
    if (!spectral.length) continue;
    traces.push({
      type: "box",
      name: `${run.label} ${run.dataset_size}`,
      y: spectral,
      boxpoints: "all",
      jitter: 0.35,
      pointpos: 0,
      hovertemplate: "%{y:.4g} eV<extra>%{fullData.name}</extra>",
    });
  }
  const layout = plotLayout("Distribucion por muestra: RMSE cerca de Fermi", "RMSE eV", {
    xaxis: { title: "", tickangle: -25 },
    showlegend: false,
  });
  if (!traces.length) {
    layout.annotations = [
      emptyPlotAnnotation("No hay RMSE cerca de Fermi con Fermi real de SIESTA."),
    ];
  }
  Plotly.react(
    id,
    traces,
    layout,
    { responsive: true, displaylogo: false },
  );
}

function renderScatterPlot(id, runs) {
  const traces = [];
  for (const [pipeline, items] of groupedRuns(runs)) {
    const label = pipelines.find((item) => item.key === pipeline)?.label || pipeline;
    const x = [];
    const y = [];
    const text = [];
    for (const run of items) {
      let relationshipRows = run.samples?.matrix_spectrum || [];
      if (!relationshipRows.length) {
        const sparseRows = run.samples?.sparse || [];
        const spectralRows = run.samples?.spectral || [];
        const spectralBySample = new Map(spectralRows.map((row) => [String(row.sample), row]));
        relationshipRows = sparseRows.map((row) => {
          const spectral = spectralBySample.get(String(row.sample)) || {};
          return {
            sample: row.sample,
            relative_frobenius_union: row.relative_frobenius_union,
            global_rmse_eV: spectral.global_rmse_eV,
            fermi_window_rmse_eV: spectral.fermi_window_rmse_eV,
            fermi_level_source: spectral.fermi_level_source,
          };
        });
      }
      for (const row of relationshipRows) {
        const xValue = row.relative_frobenius_union;
        const yValue = row.global_rmse_eV;
        if (typeof xValue !== "number" || typeof yValue !== "number") continue;
        x.push(xValue);
        y.push(yValue);
        const fermiValue =
          typeof row.fermi_window_rmse_eV === "number" && Number.isFinite(row.fermi_window_rmse_eV)
            ? `${row.fermi_window_rmse_eV.toPrecision(4)} eV`
            : "no disponible";
        text.push(
          `dataset_${run.dataset_size} - sample ${row.sample}<br>` +
            `Fermi source: ${row.fermi_level_source || "unknown"}<br>` +
            `Fermi RMSE: ${fermiValue}`,
        );
      }
    }
    if (!x.length) continue;
    traces.push({
      type: "scatter",
      mode: "markers",
      name: label,
      x,
      y,
      text,
      marker: { size: 9, opacity: 0.78 },
      hovertemplate: "%{text}<br>Frobenius %{x:.4g}<br>Global spectral RMSE %{y:.4g} eV<extra>%{fullData.name}</extra>",
    });
  }
  const layout = plotLayout("Relacion matriz-espectro", "Global spectral RMSE eV", {
    xaxis: { title: "Relative Frobenius error", gridcolor: "#edf1f4", zeroline: false },
    legend: { orientation: "h", y: -0.25 },
  });
  if (!traces.length) {
    layout.annotations = [
      emptyPlotAnnotation("No hay pares matriz-espectro comparables."),
    ];
  }
  Plotly.react(
    id,
    traces,
    layout,
    { responsive: true, displaylogo: false },
  );
}

function renderHeatmap(id, runs) {
  const metrics = [
    ["sparse", "mae_ref_eV", "MAE ref"],
    ["sparse", "relative_frobenius_union", "Frobenius rel."],
    ["sparse", "support_f1", "Support F1"],
    ["spectral", "fermi_window_rmse_eV", "Fermi RMSE"],
    ["spectral", "gap_abs_error_eV", "Gap error"],
    ["dos", "dos_wasserstein_eV", "DOS W1"],
    ["run", "pipeline_elapsed_seconds", "Time s"],
  ];
  const rows = runs
    .filter((run) => metrics.some(([group, metric]) => metricValue(run, group, metric) != null))
    .sort((a, b) => a.pipeline.localeCompare(b.pipeline) || a.dataset_size - b.dataset_size);
  const z = rows.map((run) =>
    metrics.map(([group, metric]) => {
      const value = metricValue(run, group, metric);
      return value == null ? null : value;
    }),
  );
  Plotly.react(
    id,
    [
      {
        type: "heatmap",
        z,
        x: metrics.map((item) => item[2]),
        y: rows.map((run) => `${run.label} ${run.dataset_size}`),
        colorscale: "Viridis",
        hoverongaps: false,
        hovertemplate: "%{y}<br>%{x}: %{z:.4g}<extra></extra>",
      },
    ],
    {
      title: { text: "Resumen compacto de metricas", x: 0.02, xanchor: "left", font: { size: 15 } },
      margin: { l: 120, r: 18, t: 46, b: 72 },
      paper_bgcolor: "#ffffff",
      plot_bgcolor: "#ffffff",
      font: { family: "Inter, sans-serif", color: "#17202a" },
    },
    { responsive: true, displaylogo: false },
  );
}

function renderPlots(payload) {
  const panel = document.getElementById("plots-panel");
  const status = document.getElementById("plots-status");
  panel.classList.toggle("hidden", !state.plotsEnabled);
  if (!state.plotsEnabled) {
    status.textContent = "Plots disabled";
    return;
  }
  if (!window.Plotly) {
    status.textContent = "Plotly no esta disponible";
    return;
  }
  const runs = payload?.runs || [];
  status.textContent = runs.length
    ? `${runs.length} runs con metricas${missingFermiSummary(runs)}`
    : "No hay metricas archivadas";
  renderLinePlot("plot-fermi", runs, "spectral", [{ key: "fermi_window_rmse_eV", label: "Fermi RMSE" }], "Error cerca del Fermi", "RMSE eV");
  renderLinePlot("plot-sparse", runs, "sparse", [{ key: "relative_frobenius_union", label: "Frobenius rel." }], "Error sparse matricial", "Relative Frobenius");
  renderLinePlot("plot-dos", runs, "dos", [{ key: "dos_wasserstein_eV", label: "Wasserstein" }], "Distancia DOS total", "Wasserstein eV");
  renderLinePlot("plot-gap", runs, "spectral", [{ key: "gap_abs_error_eV", label: "Gap error" }], "Error de gap", "Abs error eV");
  renderBoxPlot("plot-box", runs);
  renderScatterPlot("plot-scatter", runs);
  renderHeatmap("plot-heatmap", runs);
}

async function loadPlots() {
  const payload = await request("/api/plots");
  state.plotData = payload;
  renderPlots(payload);
}

function setupTabs() {
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((item) => item.classList.remove("active"));
      document.querySelectorAll(".view").forEach((item) => item.classList.remove("active"));
      tab.classList.add("active");
      document.getElementById(`view-${tab.dataset.view}`).classList.add("active");
      if (tab.dataset.view === "results") {
        loadResults().catch((error) => showToast(error.message));
      }
    });
  });
}

function setupEvents() {
  document.getElementById("run-all").addEventListener("click", () => {
    runAll().catch((error) => showToast(error.message));
  });
  document.getElementById("stop-all").addEventListener("click", () => {
    stopAll().catch((error) => showToast(error.message));
  });
  document.getElementById("refresh-results").addEventListener("click", () => {
    loadResults().then(() => showToast("Results refreshed")).catch((error) => showToast(error.message));
  });
  document.getElementById("show-plots").addEventListener("change", (event) => {
    state.plotsEnabled = event.target.checked;
    if (state.plotsEnabled) {
      loadPlots().catch((error) => showToast(error.message));
    } else {
      renderPlots(state.plotData);
    }
  });
  document.getElementById("run-experiment").addEventListener("click", () => {
    runExperiment().catch((error) => showToast(error.message));
  });
  document.getElementById("stop-experiment").addEventListener("click", () => {
    stopExperiment().catch((error) => showToast(error.message));
  });
  document.getElementById("fc-displacement-options").addEventListener("input", updateAtomSizesFromFcPlan);
}

async function boot() {
  setupTabs();
  setupEvents();
  await loadFcConfig();
  await pollLogs();
  await loadResults();
  state.polling = setInterval(() => {
    pollLogs().catch((error) => showToast(error.message));
  }, 1200);
}

boot().catch((error) => showToast(error.message));
