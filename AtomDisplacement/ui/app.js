const state = {
  config: null,
  raw: "",
  logOffset: 0,
  polling: null,
};

const fieldSpecs = [
  ["paths-form", [
    ["paths.base_dir", "Base dir", "text"],
    ["paths.relaxed_dir", "Relaxed dir", "text"],
    ["paths.dataset_dir", "Dataset dir", "text"],
    ["paths.samples_dir", "Samples dir", "text"],
    ["paths.training_dir", "Training dir", "text"],
    ["paths.venv_activate", "Venv activate", "text"],
    ["commands.graph2mat", "graph2mat", "text"],
    ["commands.siesta", "siesta", "text"],
    ["commands.shell", "Shell", "text"],
  ]],
  ["checkpoint-form", [
    ["checkpoint.path", "Path", "text"],
    ["checkpoint.auto_best", "Auto best", "checkbox"],
    ["checkpoint.search_glob", "Search glob", "text"],
    ["checkpoint.selection", "Selection", "text"],
  ]],
  ["generation-form", [
    ["generation.num_samples", "Samples", "number"],
    ["generation.sigma_ang", "Sigma Ang", "number"],
    ["generation.seed", "Seed", "number"],
    ["generation.max_displacement_norm_ang", "Max displacement", "number"],
    ["generation.sample_id_format", "Sample id format", "text"],
    ["single_points.limit", "Single-point limit", "number"],
    ["single_points.rerun", "Rerun", "checkbox"],
  ]],
  ["filters-form", [
    ["generation.filters.min_oh_ang", "Min OH", "number"],
    ["generation.filters.max_oh_ang", "Max OH", "number"],
    ["generation.filters.min_hh_ang", "Min HH", "number"],
    ["generation.filters.min_angle_deg", "Min angle", "number"],
    ["generation.filters.max_angle_deg", "Max angle", "number"],
  ]],
  ["siesta-form", [
    ["structure.siesta.MeshCutoff", "Mesh cutoff", "text"],
    ["structure.siesta.MaxSCFIterations", "Max SCF", "number"],
    ["structure.siesta.SolutionMethod", "Solution method", "text"],
    ["structure.siesta.SaveHS", "Save HS", "text"],
    ["structure.relaxation.system_label", "Relax label", "text"],
    ["structure.single_point.system_name_template", "SP name template", "text"],
  ]],
  ["training-data-form", [
    ["training.data.out_matrix", "Out matrix", "text"],
    ["training.data.symmetric_matrix", "Symmetric", "checkbox"],
    ["training.data.basis_files", "Basis files", "text"],
    ["training.data.runs_json", "Runs JSON", "text"],
    ["training.data.batch_size", "Batch size", "number"],
    ["training.data.store_in_memory", "Store in memory", "checkbox"],
  ]],
  ["training-model-form", [
    ["training.model.num_interactions", "Interactions", "number"],
    ["training.model.correlation", "Correlation", "number"],
    ["training.model.max_ell", "Max ell", "number"],
    ["training.model.hidden_irreps", "Hidden irreps", "text"],
    ["training.model.loss", "Loss", "text"],
    ["training.model.optim_lr", "Learning rate", "number"],
  ]],
  ["training-trainer-form", [
    ["training.trainer.accelerator", "Accelerator", "text"],
    ["training.trainer.logger.class_path", "Logger", "text"],
    ["training.trainer.logger.init_args.name", "Run name", "text"],
    ["training.trainer.logger.init_args.save_dir", "Log dir", "text"],
    ["training.trainer.max_epochs", "Epochs", "number"],
    ["training.min_completed_samples", "Min samples", "number"],
  ]],
  ["testing-form", [
    ["testing.sample_index", "Sample index", "number"],
    ["testing.data.basis_files", "Basis files", "text"],
    ["testing.data.n_matrix_components", "Matrix components", "number"],
    ["testing.callbacks.plot_matrix_error", "Plot error", "checkbox"],
    ["testing.callbacks.samplewise_metrics_logger", "Metrics logger", "checkbox"],
    ["testing.callbacks.output_file", "Metrics file", "text"],
  ]],
  ["prediction-form", [
    ["prediction.data.predict_structs", "Predict structs", "text"],
    ["prediction.data.basis_files", "Basis files", "text"],
    ["prediction.data.n_matrix_components", "Matrix components", "number"],
    ["prediction.callbacks.matrix_writer", "Matrix writer", "checkbox"],
    ["prediction.callbacks.output_file", "Output file", "text"],
  ]],
];

function splitPath(path) {
  const parts = path.split(".");
  const keys = [];
  let cursor = state.config;
  let index = 0;
  while (index < parts.length) {
    if (cursor == null || typeof cursor !== "object") {
      keys.push(parts.slice(index).join("."));
      break;
    }
    let match = null;
    for (let end = parts.length; end > index; end -= 1) {
      const candidate = parts.slice(index, end).join(".");
      if (Object.prototype.hasOwnProperty.call(cursor, candidate)) {
        match = candidate;
        index = end;
        break;
      }
    }
    if (match == null) {
      keys.push(parts[index]);
      index += 1;
      cursor = undefined;
    } else {
      keys.push(match);
      cursor = cursor[match];
    }
  }
  return keys;
}

function getPath(object, path) {
  return splitPath(path).reduce((value, key) => value == null ? undefined : value[key], object);
}

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
  if (!response.ok || payload.error) throw new Error(payload.error || response.statusText);
  return payload;
}

async function loadConfig() {
  const payload = await request("/api/config");
  state.config = payload.parsed;
  state.raw = payload.raw;
  document.getElementById("yaml-editor").value = payload.raw;
  renderForms();
}

function renderForms() {
  for (const [containerId, specs] of fieldSpecs) {
    const container = document.getElementById(containerId);
    container.innerHTML = "";
    for (const [path, label, type] of specs) {
      const value = getPath(state.config, path);
      const wrapper = document.createElement("div");
      wrapper.className = type === "checkbox" ? "checkbox-field" : "field";
      const input = document.createElement("input");
      input.dataset.path = path;
      input.type = type;
      if (type === "checkbox") {
        input.checked = Boolean(value);
      } else {
        input.value = value == null ? "" : String(value);
        if (type === "number") input.step = "any";
      }
      const labelEl = document.createElement("label");
      labelEl.textContent = label;
      if (type === "checkbox") wrapper.append(input, labelEl);
      else wrapper.append(labelEl, input);
      container.appendChild(wrapper);
    }
  }
}

function collectUpdates() {
  const updates = {};
  document.querySelectorAll("[data-path]").forEach((input) => {
    let value;
    if (input.type === "checkbox") value = input.checked;
    else if (input.type === "number") value = input.value === "" ? null : Number(input.value);
    else value = input.value === "" && input.dataset.path === "checkpoint.path" ? null : input.value;
    updates[input.dataset.path] = value;
  });
  return updates;
}

async function saveForm() {
  const payload = await request("/api/config", {
    method: "PATCH",
    body: JSON.stringify({ updates: collectUpdates() }),
  });
  state.config = payload.parsed;
  state.raw = payload.raw;
  document.getElementById("yaml-editor").value = payload.raw;
  renderForms();
  showToast("Saved");
}

async function saveYaml() {
  const content = document.getElementById("yaml-editor").value;
  const payload = await request("/api/config", {
    method: "PUT",
    body: JSON.stringify({ content }),
  });
  state.config = payload.parsed;
  state.raw = payload.raw;
  renderForms();
  showToast("YAML saved");
}

async function runPipeline() {
  const payload = await request("/api/run", { method: "POST", body: "{}" });
  updateStatus(payload);
  state.logOffset = 0;
  document.getElementById("log-output").textContent = "";
  showToast("Pipeline started");
}

async function stopPipeline() {
  const payload = await request("/api/run/stop", { method: "POST", body: "{}" });
  updateStatus(payload);
  showToast("Stop requested");
}

function updateStatus(status) {
  const dot = document.getElementById("status-dot");
  const text = document.getElementById("status-text");
  dot.classList.toggle("running", Boolean(status.running));
  dot.classList.toggle("error", status.returncode != null && status.returncode !== 0);
  if (status.running) text.textContent = "Running";
  else if (status.returncode == null || status.returncode === 0) text.textContent = "Idle";
  else text.textContent = `Exit ${status.returncode}`;
}

async function pollLogs() {
  try {
    const payload = await request(`/api/run/logs?since=${state.logOffset}`);
    state.logOffset = payload.offset;
    if (payload.lines.length) {
      const output = document.getElementById("log-output");
      output.textContent += payload.lines.join("");
      output.scrollTop = output.scrollHeight;
    }
    updateStatus(payload.status);
  } catch (error) {
    showToast(error.message);
  }
}

function setupTabs() {
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((item) => item.classList.remove("active"));
      document.querySelectorAll(".view").forEach((item) => item.classList.remove("active"));
      tab.classList.add("active");
      document.getElementById(`view-${tab.dataset.view}`).classList.add("active");
    });
  });
}

function setupEvents() {
  document.getElementById("reload-config").addEventListener("click", () => loadConfig().then(() => showToast("Reloaded")).catch((error) => showToast(error.message)));
  document.querySelectorAll(".save-form").forEach((button) => {
    button.addEventListener("click", () => saveForm().catch((error) => showToast(error.message)));
  });
  document.getElementById("save-yaml").addEventListener("click", () => saveYaml().catch((error) => showToast(error.message)));
  document.getElementById("run-pipeline").addEventListener("click", () => runPipeline().catch((error) => showToast(error.message)));
  document.getElementById("stop-pipeline").addEventListener("click", () => stopPipeline().catch((error) => showToast(error.message)));
}

async function boot() {
  setupTabs();
  setupEvents();
  await loadConfig();
  await pollLogs();
  state.polling = setInterval(pollLogs, 1200);
}

boot().catch((error) => showToast(error.message));
