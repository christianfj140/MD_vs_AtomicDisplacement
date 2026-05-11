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
const CROSS_PLOT_METRIC_DEFAULT = "low_energy_rmse_eV";

const state = {
  offsets: Object.fromEntries(pipelines.map((pipeline) => [pipeline.key, 0])),
  experimentOffset: 0,
  polling: null,
  plotsEnabled: false,
  plotData: null,
  fcMaxPerDisplacement: null,
  experimentWasRunning: false,
  performancePresetCatalog: null,
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

function inputValue(id) {
  return String(document.getElementById(id)?.value || "").trim();
}

function splitList(value) {
  return String(value || "")
    .split(/[;,]/)
    .map((item) => item.trim())
    .filter(Boolean);
}

function parseNumberListInput(id, label, { integer = false, min = null, allowEmpty = false } = {}) {
  const rawItems = splitList(inputValue(id));
  if (!rawItems.length) {
    if (allowEmpty) return [];
    throw new Error(`${label} requiere al menos un valor.`);
  }
  return rawItems.map((item) => {
    const value = Number(item);
    const integerOk = !integer || Number.isInteger(value);
    const minOk = min == null || value >= min;
    if (!Number.isFinite(value) || !integerOk || !minOk) {
      const type = integer ? "entero" : "numero";
      const floor = min == null ? "" : ` >= ${min}`;
      throw new Error(`${label}: "${item}" debe ser un ${type}${floor}.`);
    }
    return value;
  });
}

function parseSizesInput(id) {
  return parseNumberListInput(id, "Tamanos de dataset", {
    integer: true,
    min: 3,
    allowEmpty: true,
  });
}

function optionalNumberInput(id, label, { integer = false, min = null } = {}) {
  const raw = inputValue(id);
  if (!raw) return null;
  const value = Number(raw);
  if (!Number.isFinite(value) || (integer && !Number.isInteger(value)) || (min != null && value < min)) {
    const type = integer ? "entero" : "numero";
    const floor = min == null ? "" : ` >= ${min}`;
    throw new Error(`${label} debe ser un ${type}${floor}.`);
  }
  return value;
}

function parseSplitRatios() {
  return {
    train: Number(document.getElementById("split-train").value),
    validation: Number(document.getElementById("split-validation").value),
    test: Number(document.getElementById("split-test").value),
  };
}

function optionalPositiveInteger(id, label) {
  const raw = inputValue(id);
  if (!raw) return null;
  const value = Number(raw);
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`${label} debe ser un entero positivo.`);
  }
  return value;
}

function optionalBooleanSelect(id) {
  const raw = inputValue(id);
  if (raw === "") return null;
  return raw === "true";
}

function performanceSettings() {
  const accelerator = document.getElementById("performance-compute-accelerator")?.value || "cpu";
  return {
    max_parallel_siesta_jobs: optionalPositiveInteger(
      "performance-max-parallel-siesta-jobs",
      "Max parallel SIESTA jobs",
    ) || 1,
    max_parallel_dataset_jobs:
      optionalPositiveInteger("performance-max-parallel-dataset-jobs", "Max dataset jobs") || 1,
    max_parallel_prediction_jobs:
      optionalPositiveInteger("performance-max-parallel-prediction-jobs", "Max prediction jobs") || 1,
    max_parallel_evaluation_jobs:
      optionalPositiveInteger("performance-max-parallel-evaluation-jobs", "Max evaluation jobs") || 1,
    max_parallel_metric_jobs:
      optionalPositiveInteger("performance-max-parallel-metric-jobs", "Max metric jobs") || 1,
    omp_num_threads: optionalPositiveInteger("performance-omp-num-threads", "OMP threads"),
    mkl_num_threads: optionalPositiveInteger("performance-mkl-num-threads", "MKL threads"),
    openblas_num_threads: optionalPositiveInteger("performance-openblas-num-threads", "OpenBLAS threads"),
    numexpr_num_threads: optionalPositiveInteger("performance-numexpr-num-threads", "NumExpr threads"),
    torch_num_threads: optionalPositiveInteger("performance-torch-num-threads", "Torch threads"),
    compute_accelerator: accelerator,
    batch_size: optionalPositiveInteger("performance-batch-size", "Batch size override"),
    store_in_memory: optionalBooleanSelect("performance-store-in-memory"),
    reuse_validated_siesta_outputs: optionalBooleanSelect("performance-reuse-validated-siesta-outputs"),
    enable_experiment_cache: optionalBooleanSelect("performance-enable-experiment-cache"),
    error_policy: document.getElementById("performance-error-policy")?.value || "fail_fast",
    preset: document.getElementById("performance-preset")?.value || null,
    torch_float32_matmul_precision:
      document.getElementById("performance-torch-float32-matmul-precision")?.value || null,
  };
}

function optionalTextInput(id) {
  const value = inputValue(id);
  return value ? value : null;
}

function trainingSettings() {
  const settings = {
    max_epochs: optionalPositiveInteger("training-max-epochs", "Training epochs"),
    optim_lr: optionalNumberInput("training-optim-lr", "Learning rate", { min: 0.000001 }),
    batch_size: optionalPositiveInteger("training-batch-size", "Training batch size"),
    loss: optionalTextInput("training-loss"),
    num_interactions: optionalPositiveInteger("training-num-interactions", "Model interactions"),
    correlation: optionalPositiveInteger("training-correlation", "Model correlation"),
    max_ell: optionalNumberInput("training-max-ell", "Model max ell", { integer: true, min: 0 }),
    hidden_irreps: optionalTextInput("training-hidden-irreps"),
  };
  return Object.fromEntries(
    Object.entries(settings).filter(([, value]) => value !== null && value !== undefined && value !== ""),
  );
}

const performanceFieldMap = {
  max_parallel_siesta_jobs: "performance-max-parallel-siesta-jobs",
  max_parallel_dataset_jobs: "performance-max-parallel-dataset-jobs",
  max_parallel_prediction_jobs: "performance-max-parallel-prediction-jobs",
  max_parallel_evaluation_jobs: "performance-max-parallel-evaluation-jobs",
  max_parallel_metric_jobs: "performance-max-parallel-metric-jobs",
  omp_num_threads: "performance-omp-num-threads",
  mkl_num_threads: "performance-mkl-num-threads",
  openblas_num_threads: "performance-openblas-num-threads",
  numexpr_num_threads: "performance-numexpr-num-threads",
  torch_num_threads: "performance-torch-num-threads",
  compute_accelerator: "performance-compute-accelerator",
  batch_size: "performance-batch-size",
  store_in_memory: "performance-store-in-memory",
  reuse_validated_siesta_outputs: "performance-reuse-validated-siesta-outputs",
  enable_experiment_cache: "performance-enable-experiment-cache",
  error_policy: "performance-error-policy",
  torch_float32_matmul_precision: "performance-torch-float32-matmul-precision",
};

function allPerformancePresetItems() {
  const catalog = state.performancePresetCatalog;
  if (!catalog) return [];
  return [...(catalog.presets || []), ...(catalog.dynamic_profiles || [])];
}

function presetItemById(id) {
  return allPerformancePresetItems().find((item) => item.id === id) || null;
}

function settingsForPresetId(id) {
  if (id === "auto_detect") {
    const choice = state.performancePresetCatalog?.auto_detect_choice;
    return presetItemById(choice)?.settings || {};
  }
  return presetItemById(id)?.settings || {};
}

function setPerformanceInputValue(id, value) {
  const node = document.getElementById(id);
  if (!node) return;
  if (value === null || value === undefined) {
    node.value = "";
    return;
  }
  if (node.tagName === "SELECT") {
    node.value = String(value);
  } else {
    node.value = String(value);
  }
}

function applyPerformancePreset(id) {
  const settings = settingsForPresetId(id);
  for (const [key, inputId] of Object.entries(performanceFieldMap)) {
    if (Object.prototype.hasOwnProperty.call(settings, key)) {
      setPerformanceInputValue(inputId, settings[key]);
    }
  }
  renderPerformancePresetInfo(id);
}

function renderPerformancePresetInfo(id) {
  const item = presetItemById(id);
  const resolved = id === "auto_detect" ? presetItemById(state.performancePresetCatalog?.auto_detect_choice) : item;
  const description = document.getElementById("performance-preset-description");
  if (description) {
    const base = item?.description || "Manual settings.";
    const suffix = id === "auto_detect" && resolved ? ` Resolved profile: ${resolved.label}.` : "";
    description.textContent = `${base}${suffix}`;
  }
  const warningNode = document.getElementById("performance-preset-warnings");
  const warnings = [
    ...((item && item.warnings) || []),
    ...((id === "auto_detect" && resolved && resolved.warnings) || []),
  ].filter(Boolean);
  if (warningNode) {
    warningNode.classList.toggle("hidden", !warnings.length);
    warningNode.textContent = warnings.join(" ");
  }
}

function renderHardwareSummary(hardware) {
  const node = document.getElementById("performance-hardware-summary");
  if (!node || !hardware) return;
  const cpu = `${hardware.cpu_physical_cores || "?"} physical / ${hardware.cpu_logical_cores || "?"} logical CPU cores`;
  const ram = hardware.ram_total_gb ? `${hardware.ram_total_gb} GB RAM` : "RAM unknown";
  const gpu = hardware.gpu_name
    ? `${hardware.gpu_name}${hardware.gpu_vram_total_gb ? ` (${hardware.gpu_vram_total_gb} GB VRAM)` : ""}`
    : "No GPU detected";
  const torch = hardware.torch_cuda_available ? "PyTorch CUDA available" : "PyTorch CUDA not confirmed";
  node.textContent = `${cpu} · ${ram} · ${gpu} · ${torch}`;
}

function populatePerformancePresetSelector(catalog) {
  const select = document.getElementById("performance-preset");
  if (!select || !catalog) return;
  const options = [];
  for (const item of catalog.presets || []) {
    options.push({ value: item.id, label: item.label, description: item.description });
  }
  if ((catalog.dynamic_profiles || []).length) {
    for (const item of catalog.dynamic_profiles) {
      options.push({ value: item.id, label: item.label, description: item.description });
    }
  }
  options.push({ value: "", label: "Manual", description: "Keep the current editable values." });
  select.innerHTML = "";
  for (const option of options) {
    const node = document.createElement("option");
    node.value = option.value;
    node.textContent = option.label;
    node.title = option.description || "";
    if (option.value === (catalog.default_preset || "balanced")) node.selected = true;
    select.appendChild(node);
  }
}

async function loadPerformancePresets() {
  try {
    const catalog = await request("/api/performance-presets");
    state.performancePresetCatalog = catalog;
    populatePerformancePresetSelector(catalog);
    renderHardwareSummary(catalog.hardware);
    const selected = document.getElementById("performance-preset")?.value || catalog.default_preset || "balanced";
    applyPerformancePreset(selected);
  } catch (error) {
    renderPerformancePresetInfo(document.getElementById("performance-preset")?.value || "balanced");
    showToast(`No se pudieron cargar presets de rendimiento: ${error.message}`);
  }
}

function splitDatasetTableGroups(rawText) {
  const groups = [];
  let current = [];
  for (const rawLine of String(rawText || "").split(/\n/)) {
    const line = rawLine.trim();
    if (!line) {
      if (current.length) {
        groups.push(current);
        current = [];
      }
      continue;
    }
    if (line.startsWith("#")) continue;
    const lower = line.toLowerCase();
    if (
      lower.includes("snapshot") ||
      lower.includes("temperatura") ||
      lower.includes("temperature") ||
      lower.includes("desplazamiento") ||
      lower.includes("estructura") ||
      lower.includes("structure") ||
      lower.includes("amplitud") ||
      lower.includes("amplitude")
    ) {
      continue;
    }
    current.push(line);
  }
  if (current.length) groups.push(current);
  return groups;
}

function splitDatasetTableRow(row) {
  const parts = String(row)
    .split(/\s*\|\s*|[,;]\s*/)
    .map((part) => part.trim())
    .filter(Boolean);
  if (parts.length >= 2) return [parts[0], parts.slice(1).join(" ")];
  const whitespaceParts = String(row).trim().split(/\s+/).filter(Boolean);
  if (whitespaceParts.length >= 2) return [whitespaceParts[0], whitespaceParts.slice(1).join(" ")];
  return [];
}

function normalizeDisplacement(value) {
  const text = String(value || "").trim();
  if (!text) return "";
  return /[a-zA-Z]/.test(text) ? text : `${text} Ang`;
}

function parseFcDisplacementOptionsText() {
  const text = document.getElementById("fc-displacement-options").value;
  if (!text.includes(":")) return {};
  const rows = text
    .split(/\n+/)
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
  const entries = sortedFcEntries(options || {});
  const maxLength = Math.max(0, ...entries.map(([, counts]) => (counts || []).length));
  if (!maxLength) return "";
  const groups = [];
  for (let index = 0; index < maxLength; index += 1) {
    const rows = [];
    for (const [displacement, counts] of entries) {
      const count = Number((counts || [])[index]);
      if (Number.isInteger(count) && count > 0) rows.push(`${count} | ${displacement}`);
    }
    if (rows.length) groups.push(rows.join("\n"));
  }
  return groups.join("\n\n");
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

function numericPrefix(value, label) {
  const match = String(value ?? "").match(/[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?/);
  if (!match) throw new Error(`${label}: valor numerico requerido.`);
  return Number(match[0]);
}

const datasetEditorConfigs = {
  md: {
    containerId: "md-dataset-editor",
    sourceId: "md-dataset-table",
    addDatasetId: "md-add-dataset",
    countLabel: "Snapshots",
    valueLabel: "Temperatura (K)",
    defaultCount: "1000",
    defaultValue: "300",
  },
  fc: {
    containerId: "fc-dataset-editor",
    sourceId: "fc-displacement-options",
    addDatasetId: "fc-add-dataset",
    countLabel: "Snapshots",
    valueLabel: "Ang",
    defaultCount: "200",
    defaultValue: "0.02",
  },
  random_cartesian: {
    containerId: "random-dataset-editor",
    sourceId: "random-cartesian-dataset-table",
    addDatasetId: "random-add-dataset",
    countLabel: "Estructuras",
    valueLabel: "Amplitud (Ang)",
    defaultCount: "200",
    defaultValue: "0.03",
  },
};

function datasetEditorConfig(kind) {
  return datasetEditorConfigs[kind];
}

function datasetEditorContainer(kind) {
  const config = datasetEditorConfig(kind);
  return config ? document.getElementById(config.containerId) : null;
}

function sourceTextForKind(kind) {
  const config = datasetEditorConfig(kind);
  return String(document.getElementById(config?.sourceId)?.value || "");
}

function blocksForEditorSpec(kind, spec) {
  if (kind === "md") {
    return (spec.blocks || []).map((block) => ({
      count: block.n_snapshots,
      value: block.temperature_K,
    }));
  }
  if (kind === "fc") {
    return (spec.displacements || []).map((entry) => ({
      count: entry.n_structures,
      value: String(entry.value || "").replace(/\s*Ang\s*$/i, ""),
    }));
  }
  return (spec.blocks || []).map((block) => ({
    count: block.n_structures,
    value: block.amplitude_ang ?? block.max_displacement ?? block.sigma_ang ?? block.uniform_range_ang,
  }));
}

function defaultEditorBlocks(kind) {
  const config = datasetEditorConfig(kind);
  return [{ count: config.defaultCount, value: config.defaultValue }];
}

function specsForEditor(kind) {
  try {
    if (kind === "md") return parseMdDatasetTableSpecsFromText(sourceTextForKind(kind));
    if (kind === "fc") return parseFcDatasetTableSpecsFromText(sourceTextForKind(kind)) || [];
    if (kind === "random_cartesian") return parseRandomCartesianDatasetTableSpecsFromText(sourceTextForKind(kind));
  } catch (error) {
    return [];
  }
  return [];
}

function createDatasetComponentRow(kind, count = "", value = "") {
  const config = datasetEditorConfig(kind);
  const row = document.createElement("div");
  row.className = "dataset-component-row";
  row.innerHTML = `
    <label>
      <span>${config.countLabel}</span>
      <input type="number" min="1" step="1" data-field="count" value="${String(count ?? "")}" />
    </label>
    <label>
      <span>${config.valueLabel}</span>
      <input type="number" min="0" step="0.001" data-field="value" value="${String(value ?? "")}" />
    </label>
    <button class="mini-button danger" type="button" data-action="remove-component">Remove</button>
  `;
  return row;
}

function updateDatasetEditorTotals(kind) {
  const container = datasetEditorContainer(kind);
  if (!container) return;
  for (const [index, card] of Array.from(container.querySelectorAll(".dataset-card")).entries()) {
    card.querySelector(".dataset-card-name").textContent = `Dataset ${index + 1}`;
    const rows = Array.from(card.querySelectorAll(".dataset-component-row"));
    let total = 0;
    for (const row of rows) {
      const count = Number(row.querySelector('[data-field="count"]')?.value);
      if (Number.isInteger(count) && count > 0) total += count;
    }
    const totalNode = card.querySelector(".dataset-card-total");
    if (totalNode) {
      totalNode.textContent = `${total || 0} estructuras · ${rows.length} bloques`;
    }
    rows.forEach((row) => {
      const remove = row.querySelector('[data-action="remove-component"]');
      if (remove) remove.disabled = rows.length <= 1;
    });
    const removeDataset = card.querySelector('[data-action="remove-dataset"]');
    if (removeDataset) removeDataset.disabled = container.querySelectorAll(".dataset-card").length <= 1;
  }
}

function createDatasetCard(kind, blocks) {
  const card = document.createElement("div");
  card.className = "dataset-card";
  card.innerHTML = `
    <div class="dataset-card-header">
      <div class="dataset-card-title">
        <span class="dataset-card-name">Dataset</span>
        <span class="dataset-card-total">0 estructuras · 0 bloques</span>
      </div>
      <button class="mini-button danger" type="button" data-action="remove-dataset">Remove dataset</button>
    </div>
    <div class="dataset-components"></div>
    <div class="button-row">
      <button class="mini-button" type="button" data-action="add-component">+ Component</button>
    </div>
  `;
  const list = card.querySelector(".dataset-components");
  for (const block of blocks.length ? blocks : defaultEditorBlocks(kind)) {
    list.appendChild(createDatasetComponentRow(kind, block.count, block.value));
  }
  return card;
}

function renderDatasetEditor(kind) {
  const container = datasetEditorContainer(kind);
  if (!container) return;
  const specs = specsForEditor(kind);
  const blockGroups = specs.length
    ? specs.map((spec) => blocksForEditorSpec(kind, spec))
    : [defaultEditorBlocks(kind)];
  container.innerHTML = "";
  for (const blocks of blockGroups) {
    container.appendChild(createDatasetCard(kind, blocks));
  }
  bindDatasetEditor(kind);
  syncDatasetEditorText(kind);
}

function syncDatasetEditorText(kind) {
  const config = datasetEditorConfig(kind);
  const container = datasetEditorContainer(kind);
  const source = document.getElementById(config?.sourceId);
  if (!config || !container || !source || !container.querySelector(".dataset-card")) return;
  const groups = Array.from(container.querySelectorAll(".dataset-card")).map((card) =>
    Array.from(card.querySelectorAll(".dataset-component-row"))
      .map((row) => {
        const count = String(row.querySelector('[data-field="count"]')?.value || "").trim();
        const value = String(row.querySelector('[data-field="value"]')?.value || "").trim();
        return `${count} | ${value}`;
      })
      .join("\n"),
  );
  source.value = groups.join("\n\n");
  if (kind === "md") {
    const sizes = groups
      .map((group) =>
        group
          .split(/\n/)
          .map((row) => Number(splitDatasetTableRow(row)[0]))
          .filter((value) => Number.isInteger(value) && value > 0)
          .reduce((sum, value) => sum + value, 0),
      )
      .filter((value) => value > 0);
    const hidden = document.getElementById("md-sizes");
    if (hidden) hidden.value = sizes.join(", ");
  }
  if (kind === "random_cartesian") {
    const sizes = groups
      .map((group) =>
        group
          .split(/\n/)
          .map((row) => Number(splitDatasetTableRow(row)[0]))
          .filter((value) => Number.isInteger(value) && value > 0)
          .reduce((sum, value) => sum + value, 0),
      )
      .filter((value) => value > 0);
    const hidden = document.getElementById("random-cartesian-n-structures");
    if (hidden) hidden.value = sizes.join(", ");
  }
  updateDatasetEditorTotals(kind);
}

function handleDatasetEditorChanged(kind) {
  syncDatasetEditorText(kind);
  if (kind === "fc") updateAtomSizesFromFcPlan();
  else updateDatasetPreview();
}

function bindDatasetEditor(kind) {
  const container = datasetEditorContainer(kind);
  if (!container || container.dataset.bound === "true") return;
  container.dataset.bound = "true";
  container.addEventListener("input", () => handleDatasetEditorChanged(kind));
  container.addEventListener("click", (event) => {
    const action = event.target?.dataset?.action;
    if (!action) return;
    const card = event.target.closest(".dataset-card");
    if (!card) return;
    if (action === "add-component") {
      card.querySelector(".dataset-components").appendChild(createDatasetComponentRow(kind, "", ""));
    } else if (action === "remove-component") {
      const rows = card.querySelectorAll(".dataset-component-row");
      if (rows.length > 1) event.target.closest(".dataset-component-row")?.remove();
    } else if (action === "remove-dataset") {
      const cards = container.querySelectorAll(".dataset-card");
      if (cards.length > 1) card.remove();
    }
    handleDatasetEditorChanged(kind);
  });
}

function setupDatasetEditors() {
  for (const kind of Object.keys(datasetEditorConfigs)) {
    renderDatasetEditor(kind);
    const addDataset = document.getElementById(datasetEditorConfigs[kind].addDatasetId);
    addDataset?.addEventListener("click", () => {
      const container = datasetEditorContainer(kind);
      if (!container) return;
      container.appendChild(createDatasetCard(kind, defaultEditorBlocks(kind)));
      handleDatasetEditorChanged(kind);
    });
  }
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

function slugPart(value) {
  return String(value || "x")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "")
    .slice(0, 48) || "x";
}

function parseMoveAtomsInput() {
  const raw = inputValue("random-cartesian-move-atoms");
  if (!raw || raw.toLowerCase() === "all") return "all";
  return splitList(raw).map((item) => {
    const value = Number(item);
    if (!Number.isInteger(value) || value <= 0) {
      throw new Error(`Átomos movidos: "${item}" debe ser un indice 1-based positivo o usa all.`);
    }
    return value;
  });
}

function parseRandomCartesianOptions(methods) {
  const selected = methods.includes("random_cartesian");
  if (!selected) return {};
  syncDatasetEditorText("random_cartesian");
  const sizes = parseNumberListInput("random-cartesian-n-structures", "Random Cartesian structures", {
    integer: true,
    min: 3,
  });
  if (!sizes.length) return {};
  const distribution = inputValue("random-cartesian-distribution") || "gaussian";
  if (!["gaussian", "uniform"].includes(distribution)) {
    throw new Error("Random Cartesian distribution debe ser gaussian o uniform.");
  }
  const seed = optionalNumberInput("random-cartesian-seed", "Random Cartesian seed", {
    integer: true,
  });
  const sigma = optionalNumberInput("random-cartesian-sigma-ang", "Sigma gaussian", { min: 0 });
  const uniformRange = optionalNumberInput("random-cartesian-uniform-range-ang", "Rango uniforme", {
    min: 0,
  });
  const minDistance = optionalNumberInput("random-cartesian-min-distance-ang", "Min distance", {
    min: 0,
  });
  return {
    n_structures: sizes.length === 1 ? sizes[0] : sizes,
    distribution,
    seed: seed ?? 1234,
    sigma_ang: sigma ?? 0.03,
    uniform_range_ang: uniformRange ?? 0.05,
    min_distance_ang: minDistance ?? 0.65,
    move_atoms: parseMoveAtomsInput(),
    species_filter: splitList(inputValue("random-cartesian-species-filter")),
    remove_center_of_mass_translation: Boolean(
      document.getElementById("random-cartesian-remove-com")?.checked,
    ),
  };
}

function parseMdDatasetTableSpecsFromText(rawText) {
  const raw = String(rawText || "").trim();
  if (!raw) return [];
  return splitDatasetTableGroups(raw).map((group, datasetIndex) => {
    const blocks = group.map((row, blockIndex) => {
      const parts = splitDatasetTableRow(row);
      if (parts.length < 2) {
        throw new Error(`MD dataset ${datasetIndex + 1}, fila ${blockIndex + 1}: usa "snapshots | temperatura".`);
      }
      const nSnapshots = Number(parts[0]);
      const temperature = Number(parts[1]);
      if (!Number.isInteger(nSnapshots) || nSnapshots <= 0) {
        throw new Error(`MD dataset ${datasetIndex + 1}, fila ${blockIndex + 1}: snapshots debe ser entero positivo.`);
      }
      if (!Number.isFinite(temperature) || temperature < 0) {
        throw new Error(`MD dataset ${datasetIndex + 1}, fila ${blockIndex + 1}: temperatura debe ser >= 0 K.`);
      }
      return {
        block_id: `md_d${datasetIndex + 1}_T${slugPart(temperature)}_${blockIndex + 1}_${nSnapshots}`,
        label: `${nSnapshots} snapshots @ ${temperature} K`,
        temperature_K: temperature,
        n_snapshots: nSnapshots,
      };
    });
    return {
      index: datasetIndex,
      size: blocks.reduce((sum, block) => sum + Number(block.n_snapshots), 0),
      blocks,
    };
  });
}

function parseMdDatasetTableSpecs() {
  syncDatasetEditorText("md");
  return parseMdDatasetTableSpecsFromText(document.getElementById("md-dataset-table")?.value || "");
}

function parseMdTemperatureBlocks() {
  const specs = parseMdDatasetTableSpecs();
  if (specs.length === 1) return specs[0].blocks;
  return [];
}

function parseFcDatasetTableSpecsFromText(rawText) {
  const raw = String(rawText || "").trim();
  if (!raw || raw.includes(":")) return null;
  return splitDatasetTableGroups(raw).map((group, datasetIndex) => {
    const displacements = group.map((row, rowIndex) => {
      const parts = splitDatasetTableRow(row);
      if (parts.length < 2) {
        throw new Error(`FC dataset ${datasetIndex + 1}, fila ${rowIndex + 1}: usa "snapshots | desplazamiento".`);
      }
      const count = Number(parts[0]);
      const displacement = normalizeDisplacement(parts[1]);
      if (!Number.isInteger(count) || count <= 0) {
        throw new Error(`FC dataset ${datasetIndex + 1}, fila ${rowIndex + 1}: snapshots debe ser entero positivo.`);
      }
      if (!displacement) {
        throw new Error(`FC dataset ${datasetIndex + 1}, fila ${rowIndex + 1}: desplazamiento vacio.`);
      }
      return { value: displacement, n_structures: count };
    });
    return {
      mode: "explicit_table",
      index: datasetIndex,
      size: displacements.reduce((sum, item) => sum + Number(item.n_structures), 0),
      displacements,
    };
  });
}

function parseFcDatasetTableSpecs() {
  syncDatasetEditorText("fc");
  return parseFcDatasetTableSpecsFromText(document.getElementById("fc-displacement-options")?.value || "");
}

function parseRandomCartesianDatasetTableSpecsFromText(rawText) {
  const raw = String(rawText || "").trim();
  if (!raw) return [];
  return splitDatasetTableGroups(raw).map((group, datasetIndex) => {
    const blocks = group.map((row, rowIndex) => {
      const parts = splitDatasetTableRow(row);
      if (parts.length < 2) {
        throw new Error(`RC dataset ${datasetIndex + 1}, fila ${rowIndex + 1}: usa "estructuras | amplitud".`);
      }
      const count = Number(parts[0]);
      const amplitude = numericPrefix(parts[1], `RC dataset ${datasetIndex + 1}, fila ${rowIndex + 1}`);
      if (!Number.isInteger(count) || count <= 0) {
        throw new Error(`RC dataset ${datasetIndex + 1}, fila ${rowIndex + 1}: estructuras debe ser entero positivo.`);
      }
      if (!Number.isFinite(amplitude) || amplitude < 0) {
        throw new Error(`RC dataset ${datasetIndex + 1}, fila ${rowIndex + 1}: amplitud debe ser >= 0 Ang.`);
      }
      return {
        block_id: `rc_d${datasetIndex + 1}_a${slugPart(amplitude)}_${rowIndex + 1}_${count}`,
        label: `${count} estructuras @ ${amplitude} Ang`,
        n_structures: count,
        amplitude_ang: amplitude,
        max_displacement: `${amplitude} Ang`,
      };
    });
    return {
      index: datasetIndex,
      size: blocks.reduce((sum, block) => sum + Number(block.n_structures), 0),
      blocks,
    };
  });
}

function parseRandomCartesianDatasetTableSpecs() {
  syncDatasetEditorText("random_cartesian");
  return parseRandomCartesianDatasetTableSpecsFromText(
    document.getElementById("random-cartesian-dataset-table")?.value || "",
  );
}

function syncMdTableFromSizes(sizes) {
  const target = document.getElementById("md-dataset-table");
  if (!target) return;
  target.value = sizes.map((size) => `${size} | 300`).join("\n\n");
  renderDatasetEditor("md");
}

function mdSizesFromSpecs(specs) {
  return specs.map((spec) => Number(spec.size)).filter((size) => Number.isInteger(size) && size > 0);
}

function legacyMdSizesFromHidden() {
  return parseSizesInput("md-sizes");
}

function parseDatasetRecipes() {
  const input = document.getElementById("dataset-recipes-json");
  const useJson = Boolean(document.getElementById("use-dataset-recipes-json")?.checked);
  const raw = String(input?.value || "").trim();
  if (!useJson && raw) return null;
  if (useJson && !raw) {
    throw new Error("Activa recetas JSON sólo si el textarea contiene un objeto JSON.");
  }
  if (!raw) return null;
  let parsed;
  try {
    parsed = JSON.parse(raw);
  } catch (error) {
    throw new Error(`Dataset recipes JSON no es valido: ${error.message}`);
  }
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error("Dataset recipes debe ser un objeto con claves md, siesta_fc_cartesian o random_cartesian.");
  }
  return parsed;
}

function randomCartesianSizes(options) {
  const raw = options?.n_structures;
  return Array.isArray(raw) ? raw.map((value) => Number(value)) : raw == null ? [] : [Number(raw)];
}

function builderDatasetRecipes(
  methods,
  mdSizes,
  fcSpecs,
  randomCartesianOptions,
  mdDatasetSpecs = [],
  randomCartesianDatasetSpecs = [],
) {
  const recipes = {};
  if (methods.includes("md")) {
    if (mdDatasetSpecs.length) {
      recipes.md = mdDatasetSpecs.map((spec, index) => ({
        recipe_id: `md_table_${index + 1}_${spec.size}`,
        label: `MD dataset ${index + 1}: ${spec.size} snapshots`,
        blocks: spec.blocks,
      }));
    } else {
      if (!mdSizes.length) {
        throw new Error("MD requiere al menos un tamaño de dataset.");
      }
      recipes.md = mdSizes.map((size, index) => ({
        recipe_id: `md_${size}`,
        label: `MD ${size} snapshots`,
        blocks: [
          {
            block_id: `md_snapshots_${index + 1}_${size}`,
            label: `${size} snapshots`,
            n_snapshots: size,
          },
        ],
      }));
    }
  }
  if (methods.includes("siesta_fc_cartesian")) {
    recipes.siesta_fc_cartesian = fcSpecs.map((spec) => ({
      recipe_id: `fc_${spec.index + 1}_${spec.size}`,
      label: `FC ${spec.size} structures`,
      seed: optionalNumberInput("fc-random-seed", "FC random seed", { integer: true }) ?? 42,
      blocks: spec.displacements.map((entry) => ({
        block_id: `fc_${slugPart(entry.value)}_${entry.n_structures}`,
        label: `${entry.value}: ${entry.n_structures}`,
        displacement: entry.value,
        n_structures: entry.n_structures,
      })),
    }));
  }
  if (methods.includes("random_cartesian")) {
    const base = { ...randomCartesianOptions };
    delete base.n_structures;
    if (randomCartesianDatasetSpecs.length) {
      recipes.random_cartesian = randomCartesianDatasetSpecs.map((spec, index) => ({
        recipe_id: `rc_table_${index + 1}_${spec.size}`,
        label: `Random Cartesian dataset ${index + 1}: ${spec.size} structures`,
        blocks: spec.blocks.map((block) => ({ ...base, ...block })),
      }));
      return recipes;
    }
    const sizes = randomCartesianSizes(randomCartesianOptions);
    if (!sizes.length) {
      throw new Error("Random Cartesian requiere al menos un tamaño de dataset.");
    }
    const amplitude =
      base.distribution === "uniform"
        ? `u${slugPart(base.uniform_range_ang)}`
        : `s${slugPart(base.sigma_ang)}`;
    recipes.random_cartesian = sizes.map((size, index) => ({
      recipe_id: `rc_${amplitude}_${size}`,
      label: `Random Cartesian ${size}`,
      blocks: [
        {
          ...base,
          block_id: `rc_${amplitude}_${index + 1}_${size}`,
          label: `${size} random structures`,
          n_structures: size,
        },
      ],
    }));
  }
  return recipes;
}

function datasetRecipesForRun(
  methods,
  mdSizes,
  fcSpecs,
  randomCartesianOptions,
  mdDatasetSpecs = [],
  randomCartesianDatasetSpecs = [],
) {
  return (
    parseDatasetRecipes() ||
    builderDatasetRecipes(
      methods,
      mdSizes,
      fcSpecs,
      randomCartesianOptions,
      mdDatasetSpecs,
      randomCartesianDatasetSpecs,
    )
  );
}

function exportCurrentDatasetRecipes() {
  const methods = selectedMethods();
  const mdDatasetSpecs = methods.includes("md") ? parseMdDatasetTableSpecs() : [];
  const mdSizes = methods.includes("md") ? (mdDatasetSpecs.length ? mdSizesFromSpecs(mdDatasetSpecs) : legacyMdSizesFromHidden()) : [];
  const fcSpecs = methods.includes("siesta_fc_cartesian") ? fcDatasetSpecs() : [];
  if (methods.includes("siesta_fc_cartesian")) validateFcPreviewSpecs(fcSpecs);
  const randomOptions = parseRandomCartesianOptions(methods);
  const randomDatasetSpecs = methods.includes("random_cartesian") ? parseRandomCartesianDatasetTableSpecs() : [];
  const recipes = builderDatasetRecipes(
    methods,
    mdSizes,
    fcSpecs,
    randomOptions,
    mdDatasetSpecs,
    randomDatasetSpecs,
  );
  const target = document.getElementById("dataset-recipes-json");
  target.value = JSON.stringify(recipes, null, 2);
  const details = document.querySelector(".advanced-recipes");
  if (details) details.open = true;
  showToast("Recetas exportadas al JSON avanzado.");
}

function cartesianProduct(arrays) {
  return arrays.reduce(
    (acc, values) => acc.flatMap((prefix) => values.map((value) => [...prefix, value])),
    [[]],
  );
}

function fcDatasetSpecs() {
  const explicitSpecs = parseFcDatasetTableSpecs();
  if (explicitSpecs !== null) return explicitSpecs;
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

function setPreviewText(id, text, isError = false) {
  const node = document.getElementById(id);
  if (!node) return;
  node.textContent = text;
  node.classList.toggle("error-text", isError);
}

function updateMethodCardStates() {
  document.querySelectorAll(".method-card").forEach((card) => {
    const checkbox = card.querySelector(".method-checkbox");
    card.classList.toggle("disabled-method", checkbox && !checkbox.checked);
  });
}

function updateDatasetPreview() {
  const methods = selectedMethods();
  try {
    if (!methods.includes("md")) {
      setPreviewText("md-preview-summary", "desactivado", false);
    } else {
      const mdSpecs = parseMdDatasetTableSpecs();
      if (mdSpecs.length) {
        const sizes = mdSizesFromSpecs(mdSpecs);
        const blockCount = mdSpecs.reduce((sum, spec) => sum + spec.blocks.length, 0);
        setPreviewText("md-preview-summary", `${mdSpecs.length} datasets · ${sizes.join(", ")} · ${blockCount} T rows`, false);
      } else {
        const mdSizes = legacyMdSizesFromHidden();
        setPreviewText(
          "md-preview-summary",
          mdSizes.length ? `${mdSizes.length} datasets · ${mdSizes.join(", ")}` : "sin datasets",
          false,
        );
      }
    }
  } catch (error) {
    setPreviewText("md-preview-summary", "MD invalido", true);
  }
  try {
    if (!methods.includes("random_cartesian")) {
      setPreviewText("random-preview-summary", "desactivado", false);
      updateMethodCardStates();
      return;
    }
    const randomOptions = parseRandomCartesianOptions(methods);
    const randomSpecs = parseRandomCartesianDatasetTableSpecs();
    const sizes = randomSpecs.length ? randomSpecs.map((spec) => spec.size) : randomCartesianSizes(randomOptions);
    const blockCount = randomSpecs.reduce((sum, spec) => sum + spec.blocks.length, 0);
    const label = sizes.length
      ? `${sizes.length} datasets · ${sizes.join(", ")} · ${blockCount || sizes.length} bloques · ${randomOptions.distribution}`
      : "sin datasets";
    setPreviewText("random-preview-summary", label, false);
  } catch (error) {
    setPreviewText("random-preview-summary", "Random invalido", true);
  }
  updateMethodCardStates();
}

function updateAtomSizesFromFcPlan() {
  try {
    const specs = fcDatasetSpecs();
    validateFcPreviewSpecs(specs);
    const sizes = specs.map((spec) => spec.size);
    const sizesText = sizes.join(", ");
    document.getElementById("atom-sizes").value = sizesText;
    if (document.getElementById("sync-md-sizes")?.checked) {
      document.getElementById("md-sizes").value = [...new Set(sizes)].join(", ");
      syncMdTableFromSizes([...new Set(sizes)]);
    }
    renderFcPreview(specs);
    const modeLabel = currentCombinationMode() === "cartesian" ? "cartesian datasets" : "aligned datasets";
    document.getElementById("atom-combination-count").value = specs.length
      ? `${specs.length} ${modeLabel}`
      : "invalid plan";
  } catch (error) {
    document.getElementById("atom-sizes").value = "";
    if (document.getElementById("sync-md-sizes")?.checked) {
      document.getElementById("md-sizes").value = "";
    }
    document.getElementById("atom-combination-count").value = `invalid plan: ${error.message}`;
    renderFcPreview([]);
  }
  updateDatasetPreview();
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
  renderDatasetEditor("fc");
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
  const mdDatasetSpecs = methods.includes("md") ? parseMdDatasetTableSpecs() : [];
  const mdSizes = methods.includes("md")
    ? (mdDatasetSpecs.length ? mdSizesFromSpecs(mdDatasetSpecs) : legacyMdSizesFromHidden())
    : [];
  let fcDisplacementOptions = {};
  let specs = [];
  if (methods.includes("siesta_fc_cartesian")) {
    specs = fcDatasetSpecs();
    fcDisplacementOptions = parseFcDisplacementOptionsText();
    if (!specs.length) {
      throw new Error("Define al menos una tabla FC con desplazamientos.");
    }
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
  const randomDatasetSpecs = methods.includes("random_cartesian") ? parseRandomCartesianDatasetTableSpecs() : [];
  if (methods.includes("random_cartesian")) {
    if (!randomDatasetSpecs.length) {
      throw new Error("Define al menos un dataset Random Cartesian.");
    }
    const badRandomDatasets = randomDatasetSpecs
      .map((spec) => spec.size)
      .filter((size) => !Number.isInteger(size) || size < 3);
    if (badRandomDatasets.length) {
      throw new Error(
        `Con train/validation/test se requieren datasets RC >= 3. Tamaños invalidos: ${badRandomDatasets.join(", ")}.`,
      );
    }
  }
  const datasetRecipes = datasetRecipesForRun(
    methods,
    mdSizes,
    specs,
    randomCartesianOptions,
    mdDatasetSpecs,
    randomDatasetSpecs,
  );
  const splitRatios = parseSplitRatios();
  const randomSeed = Number(document.getElementById("fc-random-seed").value);
  const maxDatasets = Number(document.getElementById("fc-max-datasets").value);
  const performance = performanceSettings();
  const training = trainingSettings();
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
      dataset_recipes: datasetRecipes,
      combination_mode: currentCombinationMode(),
      sync_md_sizes: Boolean(document.getElementById("sync-md-sizes")?.checked),
      splits: splitRatios,
      split_mode: document.getElementById("split-mode").value,
      test_sets: selectedTestSets(),
      primary_metric: document.getElementById("primary-metric").value,
      compute_budget_mode: document.getElementById("compute-budget-mode").value,
      compute_accelerator: performance.compute_accelerator,
      performance,
      training_settings: training,
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

function finiteNumber(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
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
        mode: "markers",
        name: metrics.length > 1 ? `${label} · ${metric.label}` : label,
        x: points.map((point) => point.x),
        y: points.map((point) => point.y),
        text: points.map((point) => point.text),
        marker: { size: 9, opacity: 0.82 },
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
  const availability = [];
  for (const run of runs) {
    const spectral = sampleMetricValues(run, "spectral", "fermi_window_rmse_eV");
    const fermiAvailability = run.metric_availability?.spectral?.fermi_window_rmse_eV ||
      run.diagnostics?.metric_availability?.spectral?.fermi_window_rmse_eV ||
      {};
    const total = finiteNumber(fermiAvailability.n_total) ?? (run.samples?.spectral || []).length;
    const finite = finiteNumber(fermiAvailability.n_finite) ?? spectral.length;
    availability.push({
      label: `${run.label} ${run.dataset_size}`,
      finite,
      total,
    });
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
  const layout = plotLayout("Distribucion por muestra: Fermi-window RMSE", "RMSE eV", {
    xaxis: { title: "", tickangle: -25 },
    showlegend: false,
    margin: { l: 56, r: 18, t: 82, b: 64 },
  });
  const summaries = availability
    .filter((item) => item.total > 0 || item.finite > 0)
    .map((item) => `${item.label}: ${item.finite}/${item.total}`)
    .slice(0, 10);
  const zeroFinite = availability
    .filter((item) => item.total > 0 && item.finite === 0)
    .map((item) => `${item.label}: sin bandas dentro de ±2 eV de Fermi`)
    .slice(0, 4);
  const annotations = [];
  if (summaries.length) {
    annotations.push({
      text: `Disponibilidad Fermi-window: ${summaries.join(" · ")}`,
      xref: "paper",
      yref: "paper",
      x: 0,
      y: 1.22,
      xanchor: "left",
      yanchor: "bottom",
      showarrow: false,
      font: { size: 11, color: "#56616f" },
    });
  }
  if (zeroFinite.length) {
    annotations.push({
      text: zeroFinite.join(" · "),
      xref: "paper",
      yref: "paper",
      x: 0,
      y: 1.12,
      xanchor: "left",
      yanchor: "bottom",
      showarrow: false,
      font: { size: 11, color: "#9f5b00" },
    });
  }
  if (!traces.length) {
    annotations.push(
      emptyPlotAnnotation("No hay Fermi-window RMSE finito; la metrica primaria no se sustituye."),
    );
  }
  if (annotations.length) layout.annotations = annotations;
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
        mode: "markers",
        name: `${label} sparse-threshold RMSE`,
        x: ts,
        y: ts.map((t) => sparseByThreshold.get(t).reduce((s, v) => s + v, 0) / sparseByThreshold.get(t).length),
        marker: { size: 9, opacity: 0.82 },
        xaxis: "x1",
        yaxis: "y1",
      });
    }
    const ss = Array.from(dosBySigma.keys()).sort((a, b) => a - b);
    if (ss.length) {
      traces.push({
        type: "scatter",
        mode: "markers",
        name: `${label} DOS sigma W1`,
        x: ss,
        y: ss.map((s) => dosBySigma.get(s).reduce((sum, v) => sum + v, 0) / dosBySigma.get(s).length),
        marker: { size: 9, opacity: 0.82 },
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

function selectedCrossExperimentSet(payload) {
  const experiments = payload?.cross_experiments || [];
  if (!experiments.length) return null;
  const defaultSelection = payload?.default_plot_selection?.mode || "all";
  const selection = document.getElementById("plot-cross-selection")?.value || defaultSelection;
  let selected = [];
  let isolationWarning = "";
  if (selection === "latest") {
    selected = [experiments[experiments.length - 1]];
    if (experiments.length > 1) {
      isolationWarning = "Mostrando solo el experimento cross mas reciente por seleccion del usuario.";
    }
  } else if (selection === "all") {
    selected = experiments;
    const compatibilityGroups = new Set(experiments.map((experiment) => experiment.compatibility_group_id || "unknown"));
    isolationWarning =
      compatibilityGroups.size > 1
        ? `Mostrando todos los experimentos; hay ${compatibilityGroups.size} grupos metric_version/compatibilidad distintos. Interpreta rankings como visuales.`
        : experiments.length > 1
          ? "Mostrando todos los experimentos compatibles para visualizacion."
          : "";
  } else {
    const defaultGroupId = payload?.default_plot_selection?.group_id;
    const groupId = defaultGroupId || experiments[experiments.length - 1]?.compatibility_group_id;
    selected = experiments.filter((experiment) => experiment.compatibility_group_id === groupId);
    const hidden = experiments.length - selected.length;
    isolationWarning =
      hidden > 0
        ? `${hidden} experimento(s) ocultos por incompatibilidad/filtro. Selecciona "All experiments" para inspeccionarlos.`
        : "";
  }
  if (!selected.length) selected = [experiments[experiments.length - 1]];
  const latest = selected[selected.length - 1];
  const metrics = selected.flatMap((experiment) =>
    (experiment.metrics || []).map((row) => ({
      ...row,
      experiment_id: row.experiment_id || experiment.experiment_id,
      compatibility_group_id: experiment.compatibility_group_id,
    })),
  );
  return {
    ...latest,
    metrics,
    source_experiments: selected.map((experiment) => ({
      experiment_id: experiment.experiment_id,
      rows: (experiment.metrics || []).length,
      outputs: experiment.outputs,
      compatibility_group_id: experiment.compatibility_group_id,
      compatibility: experiment.compatibility,
    })),
    multi_experiment_available: experiments.length > 1,
    isolation_warning: isolationWarning,
  };
}

function primaryCrossMetric(experiment) {
  return (
    experiment?.recommendation?.primary_metric ||
    experiment?.manifest?.selected_metrics?.primary_metric ||
    "fermi_window_rmse_eV"
  );
}

function selectedCrossMetric() {
  return document.getElementById("plot-cross-metric")?.value || CROSS_PLOT_METRIC_DEFAULT;
}

function missingPrimaryMetricAnnotation(metric) {
  return emptyPlotAnnotation(`La metrica primaria ${metric} no tiene valores finitos; no se sustituye por otra metrica.`);
}

function missingPlotMetricAnnotation(metric) {
  return emptyPlotAnnotation(`La metrica ${metric} no tiene valores finitos en las filas seleccionadas; se muestra como No metric.`);
}

function crossTrainMethods(experiment) {
  const rows = experiment?.metrics || [];
  const methods = Array.from(new Set(rows.map((row) => row.train_method).filter(Boolean))).sort();
  return methods.length ? methods : ["md", "atom_displacement"];
}

function crossTestSets(experiment) {
  const rows = experiment?.metrics || [];
  const sets = Array.from(new Set(rows.map((row) => row.test_set).filter(Boolean))).sort();
  return sets.length ? sets : ["test_md", "test_siesta_fc_cartesian", "test_mixed"];
}

function groupedCrossMetrics(rows, metric) {
  const groups = new Map();
  for (const row of rows || []) {
    const value = finiteNumber(row[metric]);
    const mdDatasetSize = finiteNumber(row.md_dataset_size ?? row.dataset_size);
    const atomDatasetSize = finiteNumber(row.atom_dataset_size ?? row.dataset_size);
    const randomDatasetSize = finiteNumber(row.random_dataset_size);
    const trainDatasetSize = finiteNumber(row.train_dataset_size ?? row.dataset_size);
    const key = [
      row.experiment_id,
      row.recipe_set_hash,
      row.md_recipe_set_hash,
      row.atom_recipe_set_hash,
      row.random_recipe_set_hash,
      mdDatasetSize,
      atomDatasetSize,
      randomDatasetSize,
      trainDatasetSize,
      row.train_method,
      row.test_set,
    ].join("||");
    if (!groups.has(key)) {
      groups.set(key, {
        experiment_id: row.experiment_id,
        recipe_set_hash: row.recipe_set_hash,
        dataset_size: trainDatasetSize,
        md_dataset_size: mdDatasetSize,
        atom_dataset_size: atomDatasetSize,
        random_dataset_size: randomDatasetSize,
        train_dataset_label: row.train_dataset_label || row.train_method,
        md_dataset_label: row.md_dataset_label,
        atom_dataset_label: row.atom_dataset_label,
        random_dataset_label: row.random_dataset_label,
        train_method: row.train_method,
        test_set: row.test_set,
        values: [],
        times: [],
        n_total: 0,
      });
    }
    const group = groups.get(key);
    group.n_total += 1;
    if (value != null) group.values.push(value);
    const totalTime = finiteNumber(row.total_time_seconds);
    if (totalTime != null) {
      group.times.push(totalTime);
    }
  }
  return Array.from(groups.values()).map((group) => ({
    dataset_size: group.dataset_size,
    md_dataset_size: group.md_dataset_size,
    atom_dataset_size: group.atom_dataset_size,
    random_dataset_size: group.random_dataset_size,
    experiment_id: group.experiment_id,
    recipe_set_hash: group.recipe_set_hash,
    train_dataset_label: group.train_dataset_label,
    md_dataset_label: group.md_dataset_label,
    atom_dataset_label: group.atom_dataset_label,
    random_dataset_label: group.random_dataset_label,
    train_method: group.train_method,
    test_set: group.test_set,
    mean: group.values.length
      ? group.values.reduce((sum, value) => sum + value, 0) / group.values.length
      : null,
    n_total: group.n_total,
    n_finite: group.values.length,
    missing_count: Math.max(0, group.n_total - group.values.length),
    metric_available: group.values.length > 0,
    time:
      group.times.length > 0
        ? group.times.reduce((sum, value) => sum + value, 0) / group.times.length
        : null,
  }));
}

function crossDatasetComboLabel(row) {
  return [
    row.md_dataset_size != null ? `MD ${row.md_dataset_size}` : "",
    row.atom_dataset_size != null ? `AD ${row.atom_dataset_size}` : "",
    row.random_dataset_size != null ? `RC ${row.random_dataset_size}` : "",
    row.recipe_set_hash || "",
  ].filter(Boolean).join(" / ") || `dataset ${row.dataset_size ?? "unknown"}`;
}

function crossSizeLabel(row) {
  return `${row.test_set || "test"} · ${crossDatasetComboLabel(row)}`;
}

function metricAvailabilityLabel(row) {
  return `${row.n_finite}/${row.n_total}`;
}

function crossMissingGroupsAnnotation(groups, metric) {
  const missingGroups = groups.filter((row) => !row.metric_available).length;
  if (!missingGroups) return null;
  return {
    xref: "paper",
    yref: "paper",
    x: 0,
    y: 1.12,
    xanchor: "left",
    yanchor: "bottom",
    text: `${missingGroups} grupo(s) sin ${metric}; permanecen marcados como No metric en mapas.`,
    showarrow: false,
    font: { size: 12, color: "#9f5b00" },
  };
}

function renderCrossHeatmap(id, experiment) {
  const metric = selectedCrossMetric(experiment);
  const means = groupedCrossMetrics(experiment?.metrics || [], metric);
  const trainMethods = crossTrainMethods(experiment);
  const sizeLabels = Array.from(
    new Set(
      means
        .map(crossSizeLabel)
        .filter(Boolean),
    ),
  );
  const z = sizeLabels.map((label) =>
    trainMethods.map((method) => {
      const row = means.find((item) =>
        crossSizeLabel(item) === label &&
        item.train_method === method
      );
      return row?.metric_available ? row.mean : null;
    }),
  );
  const text = sizeLabels.map((label) =>
    trainMethods.map((method) => {
      const row = means.find((item) => crossSizeLabel(item) === label && item.train_method === method);
      if (!row) return "";
      return row.metric_available ? "" : "No metric";
    }),
  );
  const customdata = sizeLabels.map((label) =>
    trainMethods.map((method) => {
      const row = means.find((item) => crossSizeLabel(item) === label && item.train_method === method);
      if (!row) {
        return { label, method, valueText: "No row", availability: "0/0" };
      }
      return {
        label,
        method,
        valueText: row.metric_available ? row.mean.toPrecision(4) : "No metric",
        availability: metricAvailabilityLabel(row),
      };
    }),
  );
  const layout = plotLayout(`Cross-evaluation heatmap (${metric})`, metric, {
    xaxis: { title: "Training method", gridcolor: "#edf1f4", zeroline: false },
    yaxis: { title: "Frozen test set / pair size", automargin: true },
  });
  const annotations = [];
  text.forEach((row, rowIndex) => {
    row.forEach((label, colIndex) => {
      if (!label) return;
      annotations.push({
        x: trainMethods[colIndex],
        y: sizeLabels[rowIndex],
        text: label,
        showarrow: false,
        font: { size: 11, color: "#6b7280" },
      });
    });
  });
  const missingAnnotation = crossMissingGroupsAnnotation(means, metric);
  if (missingAnnotation) annotations.push(missingAnnotation);
  if (!means.length) {
    layout.annotations = [(experiment?.metrics || []).length ? missingPlotMetricAnnotation(metric) : emptyPlotAnnotation("No hay tabla cross_evaluation_metrics.csv completa.")];
  } else if (annotations.length) {
    layout.annotations = annotations;
  }
  Plotly.react(
    id,
    [{
      type: "heatmap",
      z,
      text,
      customdata,
      x: trainMethods,
      y: sizeLabels,
      colorscale: "Viridis",
      hoverongaps: false,
      hovertemplate:
        "%{customdata.label}<br>%{customdata.method}<br>" +
        `${metric}: %{customdata.valueText}<br>` +
        "finite rows: %{customdata.availability}<extra></extra>",
    }],
    layout,
    { responsive: true, displaylogo: false },
  );
}

function renderCrossLearning(id, experiment) {
  const metric = selectedCrossMetric(experiment);
  const means = groupedCrossMetrics(experiment?.metrics || [], metric);
  const traces = [];
  for (const method of crossTrainMethods(experiment)) {
    for (const testSet of crossTestSets(experiment)) {
      const points = means
        .filter((row) => row.train_method === method && row.test_set === testSet)
        .filter((row) => row.metric_available)
        .sort((a, b) => (a.dataset_size ?? 0) - (b.dataset_size ?? 0));
      if (!points.length) continue;
      traces.push({
        type: "scatter",
        mode: "markers",
        name: `${method} on ${testSet}`,
        x: points.map((row) => row.dataset_size),
        y: points.map((row) => row.mean),
        text: points.map((row) => `${crossDatasetComboLabel(row)} · ${metricAvailabilityLabel(row)} finite`),
        marker: { size: 9, opacity: 0.82 },
        hovertemplate: "dataset %{x}<br>%{y:.4g}<br>%{text}<extra>%{fullData.name}</extra>",
      });
    }
  }
  const layout = plotLayout(`Learning curves (${metric})`, metric);
  const missingAnnotation = crossMissingGroupsAnnotation(means, metric);
  if (!traces.length) {
    layout.annotations = [(experiment?.metrics || []).length ? missingPlotMetricAnnotation(metric) : emptyPlotAnnotation("No hay curvas cruzadas disponibles.")];
  } else if (missingAnnotation) {
    layout.annotations = [missingAnnotation];
  }
  Plotly.react(id, traces, layout, { responsive: true, displaylogo: false });
}

function renderCrossCompute(id, experiment) {
  const metric = selectedCrossMetric(experiment);
  const means = groupedCrossMetrics(experiment?.metrics || [], metric).filter((row) => row.time != null);
  const traces = [];
  for (const method of crossTrainMethods(experiment)) {
    const points = means
      .filter((row) => row.train_method === method && row.metric_available)
      .sort((a, b) => a.time - b.time);
    if (!points.length) continue;
    traces.push({
      type: "scatter",
      mode: "markers",
      name: method,
      x: points.map((row) => row.time),
      y: points.map((row) => row.mean),
      text: points.map((row) => `${row.test_set}, ${crossDatasetComboLabel(row)} · ${metricAvailabilityLabel(row)} finite`),
      hovertemplate: "%{text}<br>%{x:.2f}s<br>%{y:.4g}<extra>%{fullData.name}</extra>",
    });
  }
  const layout = plotLayout(`Metric vs total compute time (${metric})`, metric, {
    xaxis: { title: "Total compute seconds", gridcolor: "#edf1f4", zeroline: false },
  });
  const missingAnnotation = crossMissingGroupsAnnotation(means, metric);
  if (!traces.length) {
    layout.annotations = [(experiment?.metrics || []).length ? missingPlotMetricAnnotation(metric) : emptyPlotAnnotation("No hay timing cruzado disponible.")];
  } else if (missingAnnotation) {
    layout.annotations = [missingAnnotation];
  }
  Plotly.react(id, traces, layout, { responsive: true, displaylogo: false });
}

function renderWinnerMap(id, experiment) {
  const metric = selectedCrossMetric(experiment);
  const scientificStatus = experiment?.recommendation?.scientific_status;
  const means = groupedCrossMetrics(experiment?.metrics || [], metric);
  const methods = crossTrainMethods(experiment);
  const methodLabel = (method) => ({
    md: "MD",
    siesta_fc_cartesian: "FC",
    atom_displacement: "FC",
    random_cartesian: "RC",
  }[method] || method);
  const comboLabels = Array.from(new Set(means.map(crossDatasetComboLabel))).sort();
  const testSets = crossTestSets(experiment);
  const tieIndex = methods.length;
  const labels = new Map(methods.map((method, index) => [index, methodLabel(method)]));
  labels.set(tieIndex, "Tie");
  const z = testSets.map((testSet) => comboLabels.map((combo) => {
    const candidates = means.filter((row) => row.test_set === testSet && crossDatasetComboLabel(row) === combo);
    if (!candidates.length) return null;
    const available = candidates.filter((row) => row.metric_available);
    if (!available.length) return null;
    const best = Math.min(...available.map((row) => row.mean));
    const winners = available.filter((row) => Math.abs(row.mean - best) < 1e-12);
    if (winners.length !== 1) return tieIndex;
    const index = methods.indexOf(winners[0].train_method);
    return index >= 0 ? index : null;
  }));
  const yLabels = testSets;
  const text = z.map((row) => row.map((value) => (value == null ? "No metric" : labels.get(value))));
  const customdata = testSets.map((testSet, rowIndex) => comboLabels.map((combo, colIndex) => {
    const candidates = means.filter((row) => row.test_set === testSet && crossDatasetComboLabel(row) === combo);
    const values = candidates.map((row) =>
      row.metric_available
        ? `${methodLabel(row.train_method)}=${row.mean.toPrecision(4)} (${metricAvailabilityLabel(row)})`
        : `${methodLabel(row.train_method)}=No metric (${metricAvailabilityLabel(row)})`
    );
    return {
      winner: z[rowIndex][colIndex] == null ? "No metric" : labels.get(z[rowIndex][colIndex]),
      testSet,
      combo,
      values: values.join("; "),
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
  yLabels.forEach((rowLabel, rowIndex) => {
    comboLabels.forEach((combo, colIndex) => {
      const label = text[rowIndex][colIndex];
      if (!label) return;
      const isMissing = label === "No metric";
      annotations.push({
        x: combo,
        y: rowLabel,
        text: label,
        showarrow: false,
        font: { size: 11, color: isMissing ? "#6b7280" : "#17202a" },
      });
    });
  });
  const missingAnnotation = crossMissingGroupsAnnotation(means, metric);
  if (missingAnnotation) annotations.push(missingAnnotation);
  const layout = plotLayout(`Winner map (${metric})`, "Winner", {
    xaxis: { title: "Dataset combination", automargin: true },
    yaxis: { title: "Frozen test set", automargin: true },
    annotations,
  });
  if (!comboLabels.length || !methods.length) {
    layout.annotations = [(experiment?.metrics || []).length ? missingPlotMetricAnnotation(metric) : emptyPlotAnnotation("No hay celdas cross comparables.")];
  }
  const palette = ["#4b6f8f", "#2a7f62", "#9467bd", "#d7a021", "#d7dee5"];
  const maxValue = Math.max(1, tieIndex);
  const colorscale = Array.from({ length: tieIndex + 1 }, (_, index) => {
    const position = maxValue === 0 ? 0 : index / maxValue;
    return [position, palette[index % palette.length]];
  });
  Plotly.react(
    id,
    [{
      type: "heatmap",
      z,
      x: comboLabels,
      y: yLabels,
      customdata,
      zmin: 0,
      zmax: maxValue,
      colorscale,
      colorbar: { tickvals: Array.from(labels.keys()), ticktext: Array.from(labels.values()) },
      hovertemplate:
        "%{customdata.combo}<br>%{customdata.testSet}<br>winner: %{customdata.winner}<br>%{customdata.values}<extra></extra>",
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
  const crossExperiment = selectedCrossExperimentSet(payload);
  const crossMetric = selectedCrossMetric(crossExperiment);
  const primaryMetric = primaryCrossMetric(crossExperiment);
  const recommendation = crossExperiment?.recommendation;
  const crossRows = crossExperiment?.metrics?.length || 0;
  const crossSources = crossExperiment?.source_experiments?.length || 0;
  const isolationText = crossExperiment?.isolation_warning ? ` | ${crossExperiment.isolation_warning}` : "";
  const blockerText = recommendation ? recommendationBlockers(recommendation).slice(0, 6).join(" | ") : "";
  const crossText = recommendation?.status
    ? ` | cross: ${crossRows} filas del experimento seleccionado (${crossSources} disponibles) | plot metric: ${crossMetric} | primary: ${primaryMetric} | scientific: ${recommendation.scientific_status || "unknown"} | blockers: ${blockerText || "none"} | ${recommendation.status} - ${recommendation.reason || ""}${isolationText}`
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
    ],
    "Error Fermi-window",
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
  setupDatasetEditors();
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
  document.getElementById("plot-cross-selection")?.addEventListener("change", () => {
    if (state.plotsEnabled) renderPlots(state.plotData);
  });
  document.getElementById("plot-cross-metric")?.addEventListener("change", () => {
    if (state.plotsEnabled) renderPlots(state.plotData);
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
  document.getElementById("performance-preset")?.addEventListener("change", (event) => {
    applyPerformancePreset(event.target.value);
  });
  document.getElementById("export-dataset-recipes")?.addEventListener("click", () => {
    try {
      exportCurrentDatasetRecipes();
    } catch (error) {
      showToast(error.message);
    }
  });
  document.getElementById("fc-displacement-options").addEventListener("input", updateAtomSizesFromFcPlan);
  document.getElementById("fc-combination-mode").addEventListener("change", updateAtomSizesFromFcPlan);
  document.getElementById("fc-max-datasets").addEventListener("input", updateAtomSizesFromFcPlan);
  document.getElementById("sync-md-sizes").addEventListener("change", updateAtomSizesFromFcPlan);
  document.getElementById("split-train").addEventListener("input", updateAtomSizesFromFcPlan);
  document.getElementById("split-validation").addEventListener("input", updateAtomSizesFromFcPlan);
  document.getElementById("split-test").addEventListener("input", updateAtomSizesFromFcPlan);
  document.getElementById("md-sizes").addEventListener("input", updateDatasetPreview);
  document.getElementById("md-dataset-table")?.addEventListener("input", updateDatasetPreview);
  document.getElementById("random-cartesian-dataset-table")?.addEventListener("input", updateDatasetPreview);
  [
    "random-cartesian-n-structures",
    "random-cartesian-distribution",
    "random-cartesian-seed",
    "random-cartesian-sigma-ang",
    "random-cartesian-uniform-range-ang",
    "random-cartesian-min-distance-ang",
    "random-cartesian-move-atoms",
    "random-cartesian-species-filter",
    "random-cartesian-remove-com",
  ].forEach((id) => {
    const node = document.getElementById(id);
    node?.addEventListener("input", updateDatasetPreview);
    node?.addEventListener("change", updateDatasetPreview);
  });
  document.querySelectorAll(".method-checkbox").forEach((node) => {
    node.addEventListener("change", updateDatasetPreview);
  });
}

async function boot() {
  setupTabs();
  setupEvents();
  const venvActivateInput = document.getElementById("venv-activate-command");
  if (venvActivateInput && !String(venvActivateInput.value || "").trim()) {
    venvActivateInput.value = DEFAULT_VENV_ACTIVATE_COMMAND;
  }
  updateVenvCommandPreview();
  await loadPerformancePresets();
  await loadFcConfig();
  updateDatasetPreview();
  await pollLogs();
  await loadResults();
  state.polling = setInterval(() => {
    pollLogs().catch((error) => showToast(error.message));
  }, 1200);
}

boot().catch((error) => showToast(error.message));
