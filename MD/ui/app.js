const state = {
  config: null,
  raw: "",
  logOffset: 0,
  polling: null,
};

const fieldSpecs = [
  ["paths-form", [
    ["paths.dataset_dir", "Dataset dir", "text"],
    ["paths.training_dir", "Training dir", "text"],
    ["paths.run_fdf_name", "RUN fdf", "text"],
    ["paths.run_out_name", "RUN out", "text"],
    ["paths.training_config_name", "Training config", "text"],
    ["paths.venv_activate", "Venv activate", "text"],
    ["commands.graph2mat", "graph2mat", "text"],
    ["commands.siesta", "siesta", "text"],
  ]],
  ["checkpoint-form", [
    ["checkpoint.path", "Path", "text"],
    ["checkpoint.auto_best", "Auto best", "checkbox"],
    ["checkpoint.search_glob", "Search glob", "text"],
    ["checkpoint.selection", "Selection", "text"],
  ]],
  ["md-form", [
    ["md.type_of_run", "Type", "text"],
    ["md.steps", "Steps", "number"],
    ["md.basis_size", "Basis", "text"],
    ["md.save_hs", "Save HS", "checkbox"],
    ["md.save_de", "Save DE", "checkbox"],
    ["md.lua_script", "Lua script", "text"],
    ["md.force_aux_cell", "Force aux cell", "checkbox"],
  ]],
  ["structure-form", [
    ["md.lattice_constant.value", "Lattice constant", "number"],
    ["md.lattice_constant.unit", "Unit", "text"],
    ["md.coordinates_format", "Coordinates", "text"],
  ]],
  ["training-data-form", [
    ["training.data.out_matrix", "Out matrix", "text"],
    ["training.data.symmetric_matrix", "Symmetric", "checkbox"],
    ["training.data.basis_files", "Basis files", "text"],
    ["training.data.train_runs", "Train runs", "text"],
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
    ["testing.test_runs", "Test run", "text"],
    ["prediction.predict_structs", "Predict structs", "text"],
    ["prediction.output_file", "Prediction output", "text"],
  ]],
];

function getPath(object, path) {
  return path.split(".").reduce((value, key) => value == null ? undefined : value[key], object);
}

function setPath(object, path, value) {
  const keys = path.split(".");
  let target = object;
  for (const key of keys.slice(0, -1)) {
    target = target[key];
  }
  target[keys[keys.length - 1]] = value;
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
  if (!response.ok || payload.error) {
    throw new Error(payload.error || response.statusText);
  }
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
      if (type === "checkbox") {
        wrapper.append(input, labelEl);
      } else {
        wrapper.append(labelEl, input);
      }
      container.appendChild(wrapper);
    }
  }
}

function collectUpdates() {
  const updates = {};
  document.querySelectorAll("[data-path]").forEach((input) => {
    let value;
    if (input.type === "checkbox") {
      value = input.checked;
    } else if (input.type === "number") {
      value = input.value === "" ? null : Number(input.value);
    } else {
      value = input.value === "" && input.dataset.path === "checkpoint.path" ? null : input.value;
    }
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
  if (status.running) {
    text.textContent = "Running";
  } else if (status.returncode == null || status.returncode === 0) {
    text.textContent = "Idle";
  } else {
    text.textContent = `Exit ${status.returncode}`;
  }
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
