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

const resultPipelines = [
  { key: "md", label: "MD", resultsDir: "results_md" },
  { key: "atom_displacement", label: "AtomDisplacement", resultsDir: "results_atomdisp" },
  { key: "random_cartesian", label: "Random Cartesian", resultsDir: "results_random_cartesian" },
];

const DEFAULT_VENV_ACTIVATE_COMMAND = "source ${REPO_ROOT}/.venv/bin/activate";

const state = {
  offsets: Object.fromEntries(pipelines.map((pipeline) => [pipeline.key, 0])),
  experimentOffset: 0,
  polling: null,
  plotsEnabled: false,
  plotData: null,
  fcMaxPerDisplacement: null,
  experimentWasRunning: false,
};

function showToast(message) {
  const toast = document.getElementById("toast");
  toast.textContent = message;
  toast.classList.add("visible");
  setTimeout(() => toast.classList.remove("visible"), 2600);
}

function shellQuote(value) {
  return `"${String(value).replace(/"/g, '\\"')}"`;
}

function fallbackCopyText(text) {
  const el = document.createElement("textarea");
  el.value = text;
  el.setAttribute("readonly", "readonly");
  el.style.position = "absolute";
  el.style.left = "-9999px";
  document.body.appendChild(el);
  el.select();
  const ok = document.execCommand("copy");
  document.body.removeChild(el);
  return ok;
}

async function copyText(text) {
  if (navigator?.clipboard?.writeText) {
    await navigator.clipboard.writeText(text);
    return true;
  }
  return fallbackCopyText(text);
}

function extractMissingVenvPath(text) {
  const lines = String(text || "").split("\n");
  for (let index = lines.length - 1; index >= 0; index -= 1) {
    const line = lines[index];
    const match = line.match(/no se encontro el entorno virtual:\s*(\S+)/i);
    if (match) return match[1].trim();
  }
  return null;
}

function venvCommandFromPath(path) {
  if (!path) return null;
  return `source ${shellQuote(path)}`;
}

function preferredMissingVenvPath() {
  const ids = ["experiment-log", "md-log", "atom-displacement-log"];
  for (const id of ids) {
    const node = document.getElementById(id);
    const path = extractMissingVenvPath(node?.textContent || "");
    if (path) return path;
  }
  return null;
}

function updateVenvCommandPreview() {
  const preview = document.getElementById("venv-command-preview");
  if (!preview) return;
  const path = preferredMissingVenvPath();
  const command = venvCommandFromPath(path);
  preview.textContent = command || 'source "/ruta/al/venv/bin/activate"';
}

async function copyVenvActivationCommand() {
  const path = preferredMissingVenvPath();
  const command = venvCommandFromPath(path);
  if (!command) {
    showToast("No detecte un error de venv en los logs todavia.");
    return;
  }
  await copyText(command);
  showToast("Comando de activacion copiado al portapapeles.");
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
  const venvActivateCommandInput = document.getElementById("venv-activate-command");
  const venvActivateCommand = String(venvActivateCommandInput?.value || "").trim();
  const payload = await request("/api/run", {
    method: "POST",
    body: JSON.stringify({
      venv_activate_command: venvActivateCommand || DEFAULT_VENV_ACTIVATE_COMMAND,
    }),
  });
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
  updateVenvCommandPreview();
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
    const rawCounts = parts
      .join(":")
      .split(/[;,]/)
      .map((item) => item.trim())
      .filter(Boolean);
    const counts = rawCounts.map((item) => {
      const value = Number(item);
      if (!Number.isInteger(value) || value <= 0) {
        throw new Error(
          `Displacement options: "${item}" en la fila ${index + 1} no es un entero positivo.`,
        );
      }
      return value;
    });
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

function displacementSortValue(value) {
  const match = String(value).match(/[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?/);
  return match ? Number(match[0]) : Number.POSITIVE_INFINITY;
}

function sortedFcEntries(options) {
  return Object.entries(options).sort(
    ([left], [right]) =>
      displacementSortValue(left) - displacementSortValue(right) || String(left).localeCompare(String(right)),
  );
}

function currentCombinationMode() {
  return document.getElementById("fc-combination-mode")?.value || "aligned";
}

function selectedTestSets() {
  return Array.from(document.getElementById("test-sets")?.selectedOptions || []).map((option) => option.value);
}

function selectedMethods() {
  return Array.from(document.querySelectorAll(".method-checkbox:checked")).map((item) => item.value);
}

function pipelineLabel(key) {
  return resultPipelines.find((item) => item.key === key)?.label || key;
}

function parseRandomCartesianOptions(methods) {
  const input = document.getElementById("random-cartesian-n-structures");
  const rawItems = String(input?.value || "")
    .split(/[;,]/)
    .map((item) => item.trim())
    .filter(Boolean);
  const sizes = rawItems.map((item) => Number(item));
  if (methods.includes("random_cartesian")) {
    if (!sizes.length || sizes.some((size) => !Number.isInteger(size) || size < 3)) {
      throw new Error("Random Cartesian requiere tamanos enteros >= 3, separados por comas.");
    }
  }
  if (!sizes.length || sizes.some((size) => !Number.isInteger(size) || size <= 0)) return {};
  return sizes.length === 1 ? { n_structures: sizes[0] } : { n_structures: sizes };
}

function syncRandomCartesianSizeFromAtomPlan(sizes) {
  const input = document.getElementById("random-cartesian-n-structures");
  if (!input) return;
  const validSizes = sizes.filter((size) => Number.isInteger(size) && size >= 3);
  if (validSizes.length) {
    input.value = validSizes.join(", ");
  }
}

function cartesianProduct(arrays) {
  return arrays.reduce(
    (acc, values) => acc.flatMap((prefix) => values.map((value) => [...prefix, value])),
    [[]],
  );
}

function fcDatasetSpecs() {
  const options = parseFcDisplacementOptionsText();
  const entries = sortedFcEntries(options);
  if (!entries.length) return [];
  const mode = currentCombinationMode();
  const specs = [];
  if (mode === "aligned") {
    const lengths = entries.map(([, counts]) => counts.length);
    const uniqueLengths = new Set(lengths);
    if (uniqueLengths.size !== 1) {
      throw new Error(
        `Aligned mode requires equal list lengths. Lengths: ${entries
          .map(([displacement, counts]) => `${displacement}=${counts.length}`)
          .join(", ")}.`,
      );
    }
    const total = lengths[0] || 0;
    for (let index = 0; index < total; index += 1) {
      const displacements = entries.map(([displacement, counts]) => ({
        value: displacement,
        n_structures: counts[index],
      }));
      specs.push({
        mode,
        index,
        size: displacements.reduce((sum, item) => sum + item.n_structures, 0),
        displacements,
      });
    }
    return specs;
  }
  if (mode !== "cartesian") {
    throw new Error(`Unsupported combination mode: ${mode}`);
  }
  const combinations = cartesianProduct(entries.map(([, counts]) => counts));
  for (const [index, combo] of combinations.entries()) {
    const displacements = entries.map(([displacement], displacementIndex) => ({
      value: displacement,
      n_structures: combo[displacementIndex],
    }));
    specs.push({
      mode,
      index,
      size: displacements.reduce((sum, item) => sum + item.n_structures, 0),
      displacements,
    });
  }
  return specs;
}

function validateFcPreviewSpecs(specs) {
  const maxDatasets = Number(document.getElementById("fc-max-datasets").value);
  if (Number.isInteger(maxDatasets) && maxDatasets > 0 && specs.length > maxDatasets) {
    throw new Error(`The plan creates ${specs.length} datasets but max_datasets is ${maxDatasets}.`);
  }
  if (state.fcMaxPerDisplacement != null) {
    for (const spec of specs) {
      for (const entry of spec.displacements) {
        if (entry.n_structures > state.fcMaxPerDisplacement) {
          throw new Error(
            `${entry.value} requests ${entry.n_structures} structures, above the FC limit ${state.fcMaxPerDisplacement}.`,
          );
        }
      }
    }
  }
}

function renderFcPreview(specs) {
  const preview = document.getElementById("fc-dataset-preview");
  if (!preview) return;
  preview.innerHTML = "";
  const limit = 18;
  for (const spec of specs.slice(0, limit)) {
    const item = document.createElement("div");
    item.className = "preview-row";
    const details = spec.displacements
      .map((entry) => `${entry.value}: ${entry.n_structures}`)
      .join(" | ");
    item.innerHTML = `<strong>dataset_${spec.index}</strong><span>${spec.size} structures</span><code>${details}</code>`;
    preview.appendChild(item);
  }
  if (specs.length > limit) {
    const item = document.createElement("div");
    item.className = "preview-row muted-preview";
    item.textContent = `... ${specs.length - limit} more datasets not shown`;
    preview.appendChild(item);
  }
}

function updateAtomSizesFromFcPlan() {
  try {
    const specs = fcDatasetSpecs();
    validateFcPreviewSpecs(specs);
    const sizes = specs.map((spec) => spec.size);
    const sizesText = sizes.join(", ");
    document.getElementById("atom-sizes").value = sizesText;
    document.getElementById("md-sizes").value = [...new Set(sizes)].join(", ");
    syncRandomCartesianSizeFromAtomPlan(sizes);
    renderFcPreview(specs);
    const modeLabel = currentCombinationMode() === "cartesian" ? "cartesian datasets" : "aligned datasets";
    document.getElementById("atom-combination-count").value = specs.length
      ? `${specs.length} ${modeLabel}`
      : "invalid plan";
  } catch (error) {
    document.getElementById("atom-sizes").value = "";
    document.getElementById("md-sizes").value = "";
    document.getElementById("atom-combination-count").value = `invalid plan: ${error.message}`;
    renderFcPreview([]);
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
  document.getElementById("fc-combination-mode").value = config.combination_mode || "aligned";
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
  const wasRunning = state.experimentWasRunning;
  state.experimentWasRunning = Boolean(status.running);
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
  if (wasRunning && !status.running && state.plotsEnabled) {
    loadPlots().catch((error) => showToast(error.message));
  }
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
  const methods = selectedMethods();
  if (!methods.length) {
    throw new Error("Selecciona al menos un metodo.");
  }
  const mdSizes = parseSizesInput("md-sizes");
  let fcDisplacementOptions = {};
  let specs = [];
  if (methods.includes("siesta_fc_cartesian")) {
    fcDisplacementOptions = parseFcDisplacementOptionsText();
    if (!Object.keys(fcDisplacementOptions).length) {
      throw new Error("Define al menos una magnitud FC con opciones.");
    }
    specs = fcDatasetSpecs();
    validateFcPreviewSpecs(specs);
    const badDatasets = specs
      .map((spec) => spec.size)
      .filter((size) => !Number.isInteger(size) || size < 3);
    if (badDatasets.length) {
      throw new Error(
        `Con train/validation/test se requieren datasets >= 3. Tamaños invalidos: ${badDatasets.join(", ")}.`,
      );
    }
  }
  const atomSizes = parseSizesInput("atom-sizes");
  const randomCartesianOptions = parseRandomCartesianOptions(methods);
  const splitRatios = parseSplitRatios();
  const randomSeed = Number(document.getElementById("fc-random-seed").value);
  const maxDatasets = Number(document.getElementById("fc-max-datasets").value);
  const venvActivateCommandInput = document.getElementById("venv-activate-command");
  const venvActivateCommand = String(venvActivateCommandInput?.value || "").trim();
  state.experimentOffset = 0;
  document.getElementById("experiment-log").textContent = "";
  const payload = await request("/api/experiment", {
    method: "POST",
    body: JSON.stringify({
      md_sizes: mdSizes,
      atom_sizes: atomSizes,
      selected_methods: methods,
      run_mode: document.getElementById("run-mode").value,
      fc_displacement_options: fcDisplacementOptions,
      random_cartesian_options: randomCartesianOptions,
      combination_mode: currentCombinationMode(),
      sync_md_sizes: true,
      splits: splitRatios,
      split_mode: document.getElementById("split-mode").value,
      test_sets: selectedTestSets(),
      primary_metric: document.getElementById("primary-metric").value,
      compute_budget_mode: document.getElementById("compute-budget-mode").value,
      compute_accelerator: document.getElementById("compute-accelerator").value,
      random_seed: Number.isInteger(randomSeed) ? randomSeed : 42,
      max_datasets: Number.isInteger(maxDatasets) ? maxDatasets : 100,
      venv_activate_command: venvActivateCommand || DEFAULT_VENV_ACTIVATE_COMMAND,
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
  updateVenvCommandPreview();
}

async function loadResults() {
  const results = await request("/api/results");
  const grid = document.getElementById("results-grid");
  grid.innerHTML = "";
  for (const pipeline of resultPipelines) {
    const data = results[pipeline.key];
    if (!data) continue;
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
  for (const pipeline of resultPipelines) {
    const items = archived[pipeline.key] || [];
    const panel = document.createElement("section");
    panel.className = "panel result-row";
    panel.innerHTML = `
      <div>
        <p class="eyebrow">Archived</p>
        <h3>${pipeline.label}</h3>
      </div>
      <p><strong>${items.length}</strong> archived experiment runs</p>
      <code>Comparison/results/${pipeline.resultsDir}</code>
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

function sampleMetricValuesAny(run, group, metrics) {
  for (const metric of metrics) {
    const values = sampleMetricValues(run, group, metric);
    if (values.length) return values;
  }
  return [];
}

function missingFermiSummary(runs) {
  const pieces = [];
  for (const [pipeline, items] of groupedRuns(runs)) {
    const label = pipelineLabel(pipeline);
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
    const label = pipelineLabel(pipeline);
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
    const spectral = sampleMetricValuesAny(run, "spectral", [
      "fermi_window_rmse_eV",
      "frontier_window_rmse_eV",
      "low_energy_rmse_eV",
    ]);
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
      emptyPlotAnnotation("No hay RMSE Fermi/frontier con valores finitos."),
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
    const label = pipelineLabel(pipeline);
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
            frontier_window_rmse_eV: spectral.frontier_window_rmse_eV,
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
        const frontierValue =
          typeof row.frontier_window_rmse_eV === "number" && Number.isFinite(row.frontier_window_rmse_eV)
            ? `${row.frontier_window_rmse_eV.toPrecision(4)} eV`
            : "no disponible";
        text.push(
          `dataset_${run.dataset_size} - sample ${row.sample}<br>` +
            `Fermi source: ${row.fermi_level_source || "unknown"}<br>` +
            `Fermi RMSE: ${fermiValue}<br>` +
            `Frontier RMSE: ${frontierValue}`,
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
    ["spectral", "low_energy_rmse_eV", "Low-energy RMSE"],
    ["spectral", "fermi_window_rmse_eV", "Fermi RMSE"],
    ["spectral", "frontier_window_rmse_eV", "Frontier RMSE"],
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

function renderSensitivitySweeps(id, runs) {
  const traces = [];
  for (const [pipeline, items] of groupedRuns(runs)) {
    const label = pipelineLabel(pipeline);
    const sparseSweepRows = items.flatMap((run) => run.samples?.sparse_sweep || []);
    const dosSweepRows = items.flatMap((run) => run.samples?.dos_sweep || []);
    const sparseByThreshold = new Map();
    for (const row of sparseSweepRows) {
      const t = Number(row.support_threshold);
      const v = Number(row.rmse_union_eV);
      if (!Number.isFinite(t) || !Number.isFinite(v)) continue;
      if (!sparseByThreshold.has(t)) sparseByThreshold.set(t, []);
      sparseByThreshold.get(t).push(v);
    }
    const dosBySigma = new Map();
    for (const row of dosSweepRows) {
      const s = Number(row.dos_sigma_eV);
      const v = Number(row.dos_wasserstein_eV);
      if (!Number.isFinite(s) || !Number.isFinite(v)) continue;
      if (!dosBySigma.has(s)) dosBySigma.set(s, []);
      dosBySigma.get(s).push(v);
    }
    const ts = Array.from(sparseByThreshold.keys()).sort((a, b) => a - b);
    if (ts.length) {
      traces.push({
        type: "scatter",
        mode: "lines+markers",
        name: `${label} sparse-threshold RMSE`,
        x: ts,
        y: ts.map((t) => sparseByThreshold.get(t).reduce((s, v) => s + v, 0) / sparseByThreshold.get(t).length),
        xaxis: "x1",
        yaxis: "y1",
      });
    }
    const ss = Array.from(dosBySigma.keys()).sort((a, b) => a - b);
    if (ss.length) {
      traces.push({
        type: "scatter",
        mode: "lines+markers",
        name: `${label} DOS sigma W1`,
        x: ss,
        y: ss.map((s) => dosBySigma.get(s).reduce((sum, v) => sum + v, 0) / dosBySigma.get(s).length),
        xaxis: "x2",
        yaxis: "y2",
      });
    }
  }
  const layout = {
    title: { text: "Sensitivity sweeps", x: 0.02, xanchor: "left", font: { size: 15 } },
    grid: { rows: 1, columns: 2, pattern: "independent" },
    xaxis: { title: "Support threshold", type: "log" },
    yaxis: { title: "RMSE union (eV)" },
    xaxis2: { title: "DOS sigma (eV)" },
    yaxis2: { title: "DOS Wasserstein (eV)" },
    margin: { l: 56, r: 18, t: 46, b: 48 },
    legend: { orientation: "h", y: -0.25 },
    paper_bgcolor: "#ffffff",
    plot_bgcolor: "#ffffff",
  };
  if (!traces.length) layout.annotations = [emptyPlotAnnotation("No hay datos de sensitivity sweep.")];
  Plotly.react(id, traces, layout, { responsive: true, displaylogo: false });
}

function latestCrossExperiment(payload) {
  const experiments = payload?.cross_experiments || [];
  if (!experiments.length) return null;
  const latest = experiments[experiments.length - 1];
  const sourceExperiments = experiments.map((experiment) => ({
    experiment_id: experiment.experiment_id,
    rows: (experiment.metrics || []).length,
    outputs: experiment.outputs,
  }));
  return {
    ...latest,
    metrics: latest.metrics || [],
    source_experiments: sourceExperiments,
    multi_experiment_available: experiments.length > 1,
    isolation_warning:
      experiments.length > 1
        ? "Mostrando solo el experimento cross mas reciente. La comparacion multi-experimento debe seleccionarse explicitamente."
        : "",
  };
}

const CROSS_METRIC_FALLBACKS = [
  "frontier_window_rmse_eV",
  "low_energy_rmse_eV",
  "gap_abs_error_eV",
  "fermi_window_rmse_eV",
  "global_rmse_eV",
  "relative_frobenius_union",
  "mae_ref_eV",
];

function hasFiniteCrossMetric(experiment, metric) {
  return (experiment?.metrics || []).some((row) => {
    const value = row[metric];
    return typeof value === "number" && Number.isFinite(value);
  });
}

function primaryCrossMetric(experiment) {
  const requested =
    experiment?.recommendation?.primary_metric ||
    experiment?.manifest?.selected_metrics?.primary_metric ||
    "frontier_window_rmse_eV";
  const candidates = [requested, ...CROSS_METRIC_FALLBACKS].filter(Boolean);
  for (const metric of Array.from(new Set(candidates))) {
    if (hasFiniteCrossMetric(experiment, metric)) return metric;
  }
  return requested;
}

function crossTrainMethods(experiment) {
  const rows = experiment?.metrics || [];
  const methods = Array.from(new Set(rows.map((row) => row.train_method).filter(Boolean))).sort();
  return methods.length ? methods : ["md", "atom_displacement"];
}

function crossTestSets(experiment) {
  const rows = experiment?.metrics || [];
  const sets = Array.from(new Set(rows.map((row) => row.test_set).filter(Boolean))).sort();
  return sets.length ? sets : ["test_md", "test_atomdisp", "test_mixed"];
}

function groupedCrossMeans(rows, metric) {
  const groups = new Map();
  for (const row of rows || []) {
    const value = row[metric];
    if (typeof value !== "number" || !Number.isFinite(value)) continue;
    const mdDatasetSize = Number(row.md_dataset_size ?? row.dataset_size);
    const atomDatasetSize = Number(row.atom_dataset_size ?? row.dataset_size);
    const trainDatasetSize = Number(row.train_dataset_size ?? row.dataset_size);
    const key = [mdDatasetSize, atomDatasetSize, trainDatasetSize, row.train_method, row.test_set].join("||");
    if (!groups.has(key)) {
      groups.set(key, {
        dataset_size: trainDatasetSize,
        md_dataset_size: mdDatasetSize,
        atom_dataset_size: atomDatasetSize,
        train_method: row.train_method,
        test_set: row.test_set,
        values: [],
        times: [],
      });
    }
    groups.get(key).values.push(value);
    if (typeof row.total_time_seconds === "number" && Number.isFinite(row.total_time_seconds)) {
      groups.get(key).times.push(row.total_time_seconds);
    }
  }
  return Array.from(groups.values()).map((group) => ({
    dataset_size: Number(group.dataset_size),
    md_dataset_size: Number(group.md_dataset_size),
    atom_dataset_size: Number(group.atom_dataset_size),
    train_method: group.train_method,
    test_set: group.test_set,
    mean: group.values.reduce((sum, value) => sum + value, 0) / group.values.length,
    time:
      group.times.length > 0
        ? group.times.reduce((sum, value) => sum + value, 0) / group.times.length
        : null,
  }));
}

function renderCrossHeatmap(id, experiment) {
  const metric = primaryCrossMetric(experiment);
  const means = groupedCrossMeans(experiment?.metrics || [], metric);
  const trainMethods = crossTrainMethods(experiment);
  const testSets = crossTestSets(experiment);
  const sizeLabels = Array.from(
    new Set(
      means
        .map((row) => `${row.test_set} · MD ${row.md_dataset_size} / AD ${row.atom_dataset_size}`)
        .filter(Boolean),
    ),
  );
  const z = sizeLabels.map((label) =>
    trainMethods.map((method) => {
      const row = means.find((item) =>
        `${item.test_set} · MD ${item.md_dataset_size} / AD ${item.atom_dataset_size}` === label &&
        item.train_method === method
      );
      return row ? row.mean : null;
    }),
  );
  const layout = plotLayout(`Cross-evaluation heatmap (${metric})`, metric, {
    xaxis: { title: "Training method", gridcolor: "#edf1f4", zeroline: false },
    yaxis: { title: "Frozen test set / pair size", automargin: true },
  });
  if (!means.length) {
    layout.annotations = [emptyPlotAnnotation("No hay tabla cross_evaluation_metrics.csv completa.")];
  }
  Plotly.react(
    id,
    [{ type: "heatmap", z, x: trainMethods, y: sizeLabels, colorscale: "Viridis", hoverongaps: false }],
    layout,
    { responsive: true, displaylogo: false },
  );
}

function renderCrossLearning(id, experiment) {
  const metric = primaryCrossMetric(experiment);
  const means = groupedCrossMeans(experiment?.metrics || [], metric);
  const traces = [];
  for (const method of crossTrainMethods(experiment)) {
    for (const testSet of crossTestSets(experiment)) {
      const points = means
        .filter((row) => row.train_method === method && row.test_set === testSet)
        .sort((a, b) => a.dataset_size - b.dataset_size);
      if (!points.length) continue;
      traces.push({
        type: "scatter",
        mode: "lines+markers",
        name: `${method} on ${testSet}`,
        x: points.map((row) => row.dataset_size),
        y: points.map((row) => row.mean),
        hovertemplate: "dataset %{x}<br>%{y:.4g}<extra>%{fullData.name}</extra>",
      });
    }
  }
  const layout = plotLayout(`Learning curves (${metric})`, metric);
  if (!traces.length) layout.annotations = [emptyPlotAnnotation("No hay curvas cruzadas disponibles.")];
  Plotly.react(id, traces, layout, { responsive: true, displaylogo: false });
}

function renderCrossCompute(id, experiment) {
  const metric = primaryCrossMetric(experiment);
  const means = groupedCrossMeans(experiment?.metrics || [], metric).filter((row) => row.time != null);
  const traces = [];
  for (const method of crossTrainMethods(experiment)) {
    const points = means.filter((row) => row.train_method === method).sort((a, b) => a.time - b.time);
    if (!points.length) continue;
    traces.push({
      type: "scatter",
      mode: "markers",
      name: method,
      x: points.map((row) => row.time),
      y: points.map((row) => row.mean),
      text: points.map((row) => `${row.test_set}, dataset_${row.dataset_size}`),
      hovertemplate: "%{text}<br>%{x:.2f}s<br>%{y:.4g}<extra>%{fullData.name}</extra>",
    });
  }
  const layout = plotLayout(`Metric vs total compute time (${metric})`, metric, {
    xaxis: { title: "Total compute seconds", gridcolor: "#edf1f4", zeroline: false },
  });
  if (!traces.length) layout.annotations = [emptyPlotAnnotation("No hay timing cruzado disponible.")];
  Plotly.react(id, traces, layout, { responsive: true, displaylogo: false });
}

function renderWinnerMap(id, experiment) {
  const metric = primaryCrossMetric(experiment);
  const scientificStatus = experiment?.recommendation?.scientific_status;
  const means = groupedCrossMeans(experiment?.metrics || [], metric);
  const mdSizes = Array.from(new Set(means.map((row) => row.md_dataset_size).filter(Number.isFinite))).sort((a, b) => a - b);
  const atomSizes = Array.from(new Set(means.map((row) => row.atom_dataset_size).filter(Number.isFinite))).sort((a, b) => a - b);
  const testSets = ["test_md", "test_atomdisp", "test_mixed"];
  const rows = testSets.flatMap((testSet) => atomSizes.map((atomSize) => ({ testSet, atomSize })));
  const labels = new Map([
    [-1, "MD"],
    [0, "Tie"],
    [1, "AtomDisplacement"],
  ]);
  const shortLabels = new Map([
    [-1, "MD"],
    [0, "="],
    [1, "AD"],
  ]);
  const z = rows.map(({ testSet, atomSize }) => mdSizes.map((mdSize) => {
    const md = means.find((row) =>
      row.md_dataset_size === mdSize &&
      row.atom_dataset_size === atomSize &&
      row.test_set === testSet &&
      row.train_method === "md"
    );
    const atom = means.find((row) =>
      row.md_dataset_size === mdSize &&
      row.atom_dataset_size === atomSize &&
      row.test_set === testSet &&
      ["atom_displacement", "siesta_fc_cartesian"].includes(row.train_method)
    );
    if (!md || !atom) return null;
    if (Math.abs(md.mean - atom.mean) < 1e-12) return 0;
    return atom.mean < md.mean ? 1 : -1;
  }));
  const yLabels = rows.map(({ testSet, atomSize }) => `${testSet} · AD ${atomSize}`);
  const text = z.map((row) => row.map((value) => (value == null ? "" : shortLabels.get(value))));
  const customdata = rows.map(({ testSet, atomSize }, rowIndex) => mdSizes.map((mdSize, colIndex) => {
    const md = means.find((row) =>
      row.md_dataset_size === mdSize &&
      row.atom_dataset_size === atomSize &&
      row.test_set === testSet &&
      row.train_method === "md"
    );
    const atom = means.find((row) =>
      row.md_dataset_size === mdSize &&
      row.atom_dataset_size === atomSize &&
      row.test_set === testSet &&
      ["atom_displacement", "siesta_fc_cartesian"].includes(row.train_method)
    );
    return {
      winner: z[rowIndex][colIndex] == null ? "No data" : labels.get(z[rowIndex][colIndex]),
      testSet,
      mdSize,
      atomSize,
      md: md?.mean,
      atom: atom?.mean,
    };
  }));
  const annotations = [];
  if (scientificStatus && scientificStatus !== "robust_comparison") {
    const blockers = recommendationBlockers(experiment?.recommendation).slice(0, 3).join(" | ");
    annotations.push({
      xref: "paper",
      yref: "paper",
      x: 0,
      y: 1.12,
      xanchor: "left",
      yanchor: "bottom",
      text: `Mapa exploratorio: ${scientificStatus}${blockers ? ` · ${blockers}` : ""}`,
      showarrow: false,
      font: { size: 12, color: "#9f5b00" },
    });
  }
  rows.forEach((row, rowIndex) => {
    mdSizes.forEach((mdSize, colIndex) => {
      const label = text[rowIndex][colIndex];
      if (!label) return;
      annotations.push({
        x: mdSize,
        y: yLabels[rowIndex],
        text: label,
        showarrow: false,
        font: { size: 11, color: "#17202a" },
      });
    });
  });
  const layout = plotLayout(`Winner map (${metric})`, "Winner", {
    xaxis: { title: "MD train size", tickmode: "array", tickvals: mdSizes, ticktext: mdSizes.map(String) },
    yaxis: { title: "Frozen test set / AtomDisplacement train size", automargin: true },
    annotations,
  });
  if (!mdSizes.length || !atomSizes.length) layout.annotations = [emptyPlotAnnotation("No hay pares MD/AtomDisplacement en el mismo test set.")];
  Plotly.react(
    id,
    [{
      type: "heatmap",
      z,
      x: mdSizes,
      y: yLabels,
      customdata,
      zmin: -1,
      zmax: 1,
      colorscale: [[0, "#4b6f8f"], [0.5, "#d7dee5"], [1, "#2a7f62"]],
      colorbar: { tickvals: [-1, 0, 1], ticktext: ["MD", "Tie", "AtomDisp"] },
      hovertemplate:
        "MD size %{customdata.mdSize}<br>AtomDisp size %{customdata.atomSize}<br>%{customdata.testSet}<br>winner: %{customdata.winner}<br>MD: %{customdata.md:.4g}<br>AtomDisp: %{customdata.atom:.4g}<extra></extra>",
    }],
    layout,
    { responsive: true, displaylogo: false },
  );
}

function recommendationBlockers(recommendation) {
  if (!recommendation) return [];
  const blockers = [];
  if ((recommendation.missing_required_cells || []).length) blockers.push("missing cells");
  if (recommendation.insufficient_robust_seeds || recommendation.single_seed_warning) blockers.push("insufficient seeds");
  if ((recommendation.missing_primary_metric_cells || []).length) blockers.push("missing primary metric");
  const warnings = recommendation.severe_warnings || [];
  const addWarning = (label, pattern) => {
    if (warnings.some((warning) => String(warning).toLowerCase().includes(pattern))) blockers.push(label);
  };
  addWarning("leakage", "leakage");
  addWarning("SIESTA/model/basis mismatch", "mismatch");
  addWarning("SIESTA/model/basis mismatch", "basis");
  addWarning("checkpoint fallback", "checkpoint");
  addWarning("reproducibility warning", "absolute");
  addWarning("reproducibility warning", "reproducibility");
  return Array.from(new Set(blockers.concat(warnings.map((warning) => String(warning)))));
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
  const crossExperiment = latestCrossExperiment(payload);
  const recommendation = crossExperiment?.recommendation;
  const crossRows = crossExperiment?.metrics?.length || 0;
  const crossSources = crossExperiment?.source_experiments?.length || 0;
  const isolationText = crossExperiment?.isolation_warning ? ` | ${crossExperiment.isolation_warning}` : "";
  const blockerText = recommendation ? recommendationBlockers(recommendation).slice(0, 6).join(" | ") : "";
  const crossText = recommendation?.status
    ? ` | cross: ${crossRows} filas del experimento seleccionado (${crossSources} disponibles) | scientific: ${recommendation.scientific_status || "unknown"} | blockers: ${blockerText || "none"} | ${recommendation.status} - ${recommendation.reason || ""}${isolationText}`
    : "";
  status.textContent = runs.length
    ? `${runs.length} runs con metricas${missingFermiSummary(runs)}`
    : "No hay metricas archivadas";
  status.textContent += crossText;
  renderLinePlot(
    "plot-fermi",
    runs,
    "spectral",
    [
      { key: "fermi_window_rmse_eV", label: "Fermi-window RMSE" },
      { key: "frontier_window_rmse_eV", label: "Frontier RMSE" },
    ],
    "Error cerca de Fermi/frontier",
    "RMSE eV",
  );
  renderLinePlot("plot-low-energy", runs, "spectral", [{ key: "low_energy_rmse_eV", label: "Low-energy RMSE" }], "Low-energy eigenvalues", "RMSE eV");
  renderLinePlot("plot-sparse", runs, "sparse", [{ key: "relative_frobenius_union", label: "Frobenius rel." }], "Error sparse matricial", "Relative Frobenius");
  renderLinePlot("plot-dos", runs, "dos", [{ key: "dos_wasserstein_eV", label: "Wasserstein" }], "Distancia DOS total", "Wasserstein eV");
  renderLinePlot("plot-gap", runs, "spectral", [{ key: "gap_abs_error_eV", label: "Gap error" }], "Error de gap", "Abs error eV");
  renderBoxPlot("plot-box", runs);
  renderScatterPlot("plot-scatter", runs);
  renderHeatmap("plot-heatmap", runs);
  renderLinePlot("plot-frontier", runs, "spectral", [{ key: "frontier_window_rmse_eV", label: "Frontier RMSE" }], "Frontier window", "RMSE eV");
  renderLinePlot("plot-aligned", runs, "spectral", [{ key: "align_global_rmse_eV", label: "Aligned global RMSE" }], "Spectral aligned RMSE", "RMSE eV");
  renderSensitivitySweeps("plot-sweeps", runs);
  renderCrossHeatmap("plot-cross-heatmap", crossExperiment);
  renderCrossLearning("plot-learning", crossExperiment);
  renderCrossCompute("plot-compute", crossExperiment);
  renderWinnerMap("plot-winner", crossExperiment);
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
  document.getElementById("copy-venv-command").addEventListener("click", () => {
    copyVenvActivationCommand().catch((error) => showToast(error.message));
  });
  document.getElementById("fc-displacement-options").addEventListener("input", updateAtomSizesFromFcPlan);
  document.getElementById("fc-combination-mode").addEventListener("change", updateAtomSizesFromFcPlan);
  document.getElementById("fc-max-datasets").addEventListener("input", updateAtomSizesFromFcPlan);
  document.getElementById("split-train").addEventListener("input", updateAtomSizesFromFcPlan);
  document.getElementById("split-validation").addEventListener("input", updateAtomSizesFromFcPlan);
  document.getElementById("split-test").addEventListener("input", updateAtomSizesFromFcPlan);
}

async function boot() {
  setupTabs();
  setupEvents();
  const venvActivateInput = document.getElementById("venv-activate-command");
  if (venvActivateInput && !String(venvActivateInput.value || "").trim()) {
    venvActivateInput.value = DEFAULT_VENV_ACTIVATE_COMMAND;
  }
  updateVenvCommandPreview();
  await loadFcConfig();
  await pollLogs();
  await loadResults();
  state.polling = setInterval(() => {
    pollLogs().catch((error) => showToast(error.message));
  }, 1200);
}

boot().catch((error) => showToast(error.message));
