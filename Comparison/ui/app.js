const pipelines = [
  { key: "md", label: "MD", logId: "md-log", dotId: "md-status-dot", textId: "md-status-text" },
  {
    key: "atom_displacement",
    label: "FC Cartesian",
    logId: "atom-displacement-log",
    dotId: "atom-displacement-status-dot",
    textId: "atom-displacement-status-text",
  },
];

const resultPipelines = [
  { key: "md", label: "MD", resultsDir: "results_md" },
  { key: "atom_displacement", label: "FC Cartesian", resultsDir: "results_atomdisp" },
  { key: "random_cartesian", label: "Random Cartesian", resultsDir: "results_random_cartesian" },
];

const METHOD_ID_ALIASES = {
  md: "md",
  siesta_fc_cartesian: "siesta_fc_cartesian",
  atom_displacement: "siesta_fc_cartesian",
  atomdisp: "siesta_fc_cartesian",
  random_cartesian: "random_cartesian",
};

const TEST_SET_ALIASES = {
  test_atomdisp: "test_siesta_fc_cartesian",
};

const METHOD_DISPLAY_LABELS = {
  md: "MD",
  siesta_fc_cartesian: "FC Cartesian",
  random_cartesian: "Random Cartesian",
};

const TEST_SET_DISPLAY_LABELS = {
  test_md: "MD test (test_md)",
  test_siesta_fc_cartesian: "FC Cartesian test (test_siesta_fc_cartesian)",
  test_random_cartesian: "Random Cartesian test (test_random_cartesian)",
  test_mixed: "Mixed test (test_mixed)",
};

function normalizeMethodId(value) {
  const text = String(value || "").trim();
  return METHOD_ID_ALIASES[text] || text;
}

function normalizeTestSetId(value) {
  const text = String(value || "").trim();
  if (TEST_SET_ALIASES[text]) return TEST_SET_ALIASES[text];
  if (!text.startsWith("test_")) return text;
  const suffix = text.slice("test_".length);
  if (suffix === "mixed") return "test_mixed";
  return `test_${normalizeMethodId(suffix)}`;
}

function methodDisplayLabel(value) {
  const methodId = normalizeMethodId(value);
  return METHOD_DISPLAY_LABELS[methodId] || String(value || methodId || "unknown");
}

function testSetDisplayLabel(value) {
  const testSetId = normalizeTestSetId(value);
  return TEST_SET_DISPLAY_LABELS[testSetId] || testSetId;
}

function canonicalDisplayText(value) {
  return String(value || "")
    .replace(/\btest_atomdisp\b/g, "FC Cartesian legacy alias (test_atomdisp)")
    .replace(/\btest_siesta_fc_cartesian\b/g, TEST_SET_DISPLAY_LABELS.test_siesta_fc_cartesian)
    .replace(/\btest_random_cartesian\b/g, TEST_SET_DISPLAY_LABELS.test_random_cartesian)
    .replace(/\btest_md\b/g, TEST_SET_DISPLAY_LABELS.test_md)
    .replace(/\bAtomDisplacement\b/g, "FC Cartesian")
    .replace(/\batom_displacement\b/g, "FC Cartesian")
    .replace(/\batomdisp\b/gi, "FC Cartesian")
    .replace(/\bAD\b/g, "FC Cartesian")
    .replace(/\bsiesta_fc_cartesian\b/g, "FC Cartesian")
    .replace(/\brandom_cartesian\b/g, "Random Cartesian");
}

const DEFAULT_VENV_ACTIVATE_COMMAND = "source ${REPO_ROOT}/.venv/bin/activate";
const PRIMARY_METRIC_DEFAULT = "low_energy_rmse_eV";
const CROSS_PLOT_METRIC_DEFAULT = "low_energy_rmse_eV";
const LOG_POLL_LIMIT = 2000;
const POLL_INTERVAL_MS = 1200;
const POLL_ERROR_TOAST_INTERVAL_MS = 30000;

const METRIC_HELP = {
  low_energy_rmse_eV: {
    label: "Low-energy RMSE",
    formula: "\\bar{m}=\\frac{1}{N_s}\\sum_s \\operatorname{RMSE}^{(s)}_S,\\quad \\operatorname{RMSE}_S=\\sqrt{\\frac{1}{|S|}\\sum_{i\\in S}(\\varepsilon^{pred}_i-\\varepsilon^{ref}_i)^2}",
    description: "Raiz del error cuadratico medio entre autovalores predichos y referencia en la ventana de baja energia.",
    purpose: "Resume la fidelidad del espectro en la zona que suele dominar comparaciones quimicas y de estructura electronica.",
    direction: "Menor es mejor; al cuadrar los errores penaliza mas los fallos grandes.",
  },
  fermi_window_rmse_eV: {
    label: "Fermi-window RMSE",
    formula: "\\bar{m}=\\frac{1}{N_s}\\sum_s \\operatorname{RMSE}^{(s)}_F,\\quad \\operatorname{RMSE}_F=\\sqrt{\\frac{1}{N_F}\\sum_{|\\varepsilon^{ref}_i-E_F|\\le w}(\\varepsilon^{pred}_i-\\varepsilon^{ref}_i)^2}",
    description: "RMSE de autovalores dentro de la ventana alrededor del nivel de Fermi.",
    purpose: "Sirve para vigilar estados cercanos al borde de ocupacion, donde pequenos errores pueden cambiar propiedades electronicas.",
    direction: "Menor es mejor; si no hay estados en la ventana se marca como no disponible y se usa el diagnostico frontier aparte.",
  },
  frontier_window_rmse_eV: {
    label: "Frontier RMSE",
    formula: "\\bar{m}=\\frac{1}{N_s}\\sum_s \\operatorname{RMSE}^{(s)}_{frontier},\\quad \\operatorname{RMSE}_{frontier}=\\sqrt{\\operatorname{mean}(e_{HOMO}^2,e_{LUMO}^2)}",
    description: "RMSE en estados frontier, normalmente alrededor de HOMO/LUMO, cuando la ventana de Fermi no basta.",
    purpose: "Mantiene un diagnostico local del borde ocupado/no ocupado incluso en sistemas con pocos niveles en la ventana de Fermi.",
    direction: "Menor es mejor; comparalo junto al gap para entender si el borde espectral se conserva.",
  },
  align_global_rmse_eV: {
    label: "Aligned global RMSE",
    formula: "\\bar{m}=\\frac{1}{N_s}\\sum_s \\sqrt{\\frac{1}{N}\\sum_i((\\varepsilon^{pred}_i+\\Delta)-\\varepsilon^{ref}_i)^2},\\quad \\Delta=\\operatorname{mean}_i(\\varepsilon^{ref}_i-\\varepsilon^{pred}_i)",
    description: "RMSE del espectro completo tras corregir un desplazamiento global entre prediccion y referencia.",
    purpose: "Separa errores de forma espectral de un offset casi constante de energia.",
    direction: "Menor es mejor; si baja mucho frente al RMSE global, el problema puede ser principalmente de alineamiento.",
  },
  global_rmse_eV: {
    label: "Global RMSE",
    formula: "\\bar{m}=\\frac{1}{N_s}\\sum_s \\sqrt{\\frac{1}{N}\\sum_i(\\varepsilon^{pred}_i-\\varepsilon^{ref}_i)^2}",
    description: "RMSE sobre el conjunto global de autovalores comparables.",
    purpose: "Da una lectura amplia del error espectral total, sin concentrarse solo en baja energia o Fermi.",
    direction: "Menor es mejor.",
  },
  gap_abs_error_eV: {
    label: "Gap absolute error",
    formula: "\\bar{m}=\\frac{1}{N_s}\\sum_s |g^{pred}_s-g^{ref}_s|,\\quad g=\\varepsilon_{LUMO}-\\varepsilon_{HOMO}",
    description: "Error absoluto del gap HOMO-LUMO o gap electronico equivalente frente a la referencia.",
    purpose: "Ayuda a saber si el modelo conserva separaciones energeticas clave entre estados ocupados y no ocupados.",
    direction: "Menor es mejor; un valor bajo indica que el tamano del gap se reproduce mejor.",
  },
  relative_frobenius_union: {
    label: "Relative Frobenius",
    formula: "\\bar{m}=\\frac{1}{N_s}\\sum_s \\frac{\\|H^{pred}-H^{ref}\\|_{F,union}}{\\|H^{ref}\\|_F},\\quad \\|A\\|_F=\\sqrt{\\sum_{ij}|a_{ij}|^2}",
    description: "Norma Frobenius relativa del error matricial, calculada como energia cuadratica acumulada en la matriz.",
    purpose: "Mide fidelidad global del Hamiltoniano o matriz sparse antes de mirar sus consecuencias espectrales.",
    direction: "Menor es mejor; es sensible a errores repartidos por muchos elementos y a errores grandes.",
  },
  mae_ref_eV: {
    label: "MAE ref",
    formula: "\\bar{m}=\\frac{1}{N_s}\\sum_s \\frac{1}{|R|}\\sum_{ij\\in R}|H^{pred}_{ij}-H^{ref}_{ij}|",
    description: "Error absoluto medio frente a la referencia en elementos de matriz o valores escalares comparables.",
    purpose: "Ofrece una lectura robusta del error tipico porque promedia magnitudes absolutas sin cuadrarlas.",
    direction: "Menor es mejor.",
  },
  mse_ref_eV2: {
    label: "MSE ref",
    formula: "\\bar{m}=\\frac{1}{N_s}\\sum_s \\operatorname{mean}_{ij\\in R}|H^{pred}_{ij}-H^{ref}_{ij}|^2",
    description: "Error cuadratico medio del Hamiltoniano en el soporte no nulo de referencia, con unidades eV^2.",
    purpose: "Diagnostico DeepH-comparable para penalizar mas los errores grandes sin cambiar la politica de soporte sparse.",
    direction: "Menor es mejor; no reemplaza las metricas espectrales primarias.",
  },
  mse_pred_eV2: {
    label: "MSE pred",
    formula: "\\bar{m}=\\frac{1}{N_s}\\sum_s \\operatorname{mean}_{ij\\in P}|H^{pred}_{ij}-H^{ref}_{ij}|^2",
    description: "Error cuadratico medio del Hamiltoniano en el soporte no nulo predicho, con unidades eV^2.",
    purpose: "Ayuda a revisar errores donde el modelo predice acoplamientos activos.",
    direction: "Menor es mejor; diagnostico secundario.",
  },
  mse_union_eV2: {
    label: "MSE union",
    formula: "\\bar{m}=\\frac{1}{N_s}\\sum_s \\operatorname{mean}_{ij\\in R\\cup P}|H^{pred}_{ij}-H^{ref}_{ij}|^2",
    description: "Error cuadratico medio del Hamiltoniano en la union de soportes referencia/prediccion, con unidades eV^2.",
    purpose: "Resume valor y soporte sparse en una metrica cuadratica de matriz.",
    direction: "Menor es mejor; diagnostico secundario.",
  },
  r2_ref: {
    label: "R2 ref",
    formula: "R^2=1-\\frac{\\sum|H^{pred}-H^{ref}|^2}{\\sum|H^{ref}-\\overline{H}^{ref}|^2}",
    description: "Coeficiente de determinacion del Hamiltoniano en el soporte de referencia.",
    purpose: "Complementa MAE/RMSE mostrando cuanta variacion del target queda explicada.",
    direction: "Mayor es mejor; no esta disponible si el target es constante.",
  },
  r2_pred: {
    label: "R2 pred",
    formula: "R^2=1-\\frac{\\sum|H^{pred}-H^{ref}|^2}{\\sum|H^{ref}-\\overline{H}^{ref}|^2}",
    description: "Coeficiente de determinacion del Hamiltoniano en el soporte predicho.",
    purpose: "Diagnostica varianza explicada donde el modelo predice entradas activas.",
    direction: "Mayor es mejor; no esta disponible si el target es constante.",
  },
  r2_union: {
    label: "R2 union",
    formula: "R^2=1-\\frac{\\sum|H^{pred}-H^{ref}|^2}{\\sum|H^{ref}-\\overline{H}^{ref}|^2}",
    description: "Coeficiente de determinacion del Hamiltoniano en la union de soportes.",
    purpose: "Complementa los errores sparse de union con una lectura de varianza explicada.",
    direction: "Mayor es mejor; no esta disponible si el target es constante.",
  },
  mae_ref_meV: {
    label: "MAE ref meV",
    formula: "1000\\times \\operatorname{MAE}_{ref,eV}",
    description: "Alias de MAE ref en meV para reportes estilo DeepH.",
    purpose: "Cambia solo la unidad; la definicion y el soporte son los mismos que MAE ref en eV.",
    direction: "Menor es mejor.",
  },
  rmse_ref_meV: {
    label: "RMSE ref meV",
    formula: "1000\\times \\operatorname{RMSE}_{ref,eV}",
    description: "Alias de RMSE ref en meV para reportes estilo DeepH.",
    purpose: "Cambia solo la unidad; la definicion y el soporte son los mismos que RMSE ref en eV.",
    direction: "Menor es mejor.",
  },
  mae_pred_meV: {
    label: "MAE pred meV",
    formula: "1000\\times \\operatorname{MAE}_{pred,eV}",
    description: "Alias de MAE pred en meV.",
    purpose: "Facilita comparar magnitudes sin alterar la metrica base.",
    direction: "Menor es mejor.",
  },
  rmse_pred_meV: {
    label: "RMSE pred meV",
    formula: "1000\\times \\operatorname{RMSE}_{pred,eV}",
    description: "Alias de RMSE pred en meV.",
    purpose: "Facilita comparar magnitudes sin alterar la metrica base.",
    direction: "Menor es mejor.",
  },
  mae_union_meV: {
    label: "MAE union meV",
    formula: "1000\\times \\operatorname{MAE}_{union,eV}",
    description: "Alias de MAE union en meV.",
    purpose: "Reporta la misma metrica de union sparse en una escala mas comun para DeepH.",
    direction: "Menor es mejor.",
  },
  rmse_union_meV: {
    label: "RMSE union meV",
    formula: "1000\\times \\operatorname{RMSE}_{union,eV}",
    description: "Alias de RMSE union en meV.",
    purpose: "Reporta la misma metrica de union sparse en una escala mas comun para DeepH.",
    direction: "Menor es mejor.",
  },
  support_f1: {
    label: "Support F1",
    formula: "\\bar{m}=\\frac{1}{N_s}\\sum_s F_1^{(s)},\\quad F_1=\\frac{2\\,\\mathrm{precision}\\,\\mathrm{recall}}{\\mathrm{precision}+\\mathrm{recall}}",
    description: "F1 sobre el soporte sparse: combina precision y recall para entradas no nulas o activas.",
    purpose: "Evalua si el modelo conserva donde existe acoplamiento/matriz activa, no solo el valor numerico de lo predicho.",
    direction: "Mayor es mejor; 1 es perfecto y 0 es el peor caso.",
  },
  dos_wasserstein_eV: {
    label: "DOS Wasserstein-1",
    formula: "\\bar{m}=\\frac{1}{N_s}\\sum_s \\int |\\operatorname{CDF}_{ref}(E)-\\operatorname{CDF}_{pred}(E)|\\,dE",
    description: "Distancia Wasserstein-1 entre densidades de estados predicha y de referencia.",
    purpose: "Cuantifica cuanta masa espectral habria que desplazar en energia para transformar una DOS en la otra.",
    direction: "Menor es mejor; cero indica distribuciones DOS indistinguibles en esta metrica.",
  },
  dos_mae_500_fermi_window: {
    label: "DOS MAE 500 Fermi window",
    formula: "\\frac{1}{500}\\sum_{k=1}^{500}|D^{pred}(E_F+x_k)-D^{ref}(E_F+x_k)|,\\quad x_k\\in[-6,6]\\,eV",
    description: "MAE entre DOS predicha y referencia en 500 puntos alrededor del Fermi de SIESTA.",
    purpose: "Diagnostico DOS comparable con DeepH; requiere un Fermi real y no se sustituye si falta.",
    direction: "Menor es mejor; no estima Fermi desde HOMO/LUMO.",
  },
  dos_ks_statistic: {
    label: "DOS KS statistic",
    formula: "\\bar{m}=\\frac{1}{N_s}\\sum_s \\max_E |\\operatorname{CDF}_{ref}(E)-\\operatorname{CDF}_{pred}(E)|",
    description: "Maxima separacion entre distribuciones acumuladas de DOS predicha y referencia.",
    purpose: "Detecta diferencias de forma entre distribuciones, complementando la distancia Wasserstein.",
    direction: "Menor es mejor; valores grandes apuntan a distribuciones acumuladas mas distintas.",
  },
  pipeline_elapsed_seconds: {
    label: "Pipeline elapsed seconds",
    formula: "t_{elapsed}=t_{finished}-t_{started}",
    description: "Tiempo total registrado para una ejecucion o grupo de ejecuciones.",
    purpose: "Permite comparar coste computacional frente a calidad de metrica.",
    direction: "Menor es mejor si la precision es comparable.",
  },
};

const PLOT_HELP_BY_ID = {
  "plot-fermi": {
    title: "Error Fermi-window",
    metricKey: "fermi_window_rmse_eV",
  },
  "plot-low-energy": {
    title: "Low-energy eigenvalues",
    metricKey: "low_energy_rmse_eV",
  },
  "plot-sparse": {
    title: "Error sparse matricial",
    metricKey: "relative_frobenius_union",
  },
  "plot-dos": {
    title: "Distancia DOS total",
    metricKey: "dos_wasserstein_eV",
  },
  "plot-gap": {
    title: "Error de gap",
    metricKey: "gap_abs_error_eV",
  },
  "plot-frontier": {
    title: "Frontier window",
    metricKey: "frontier_window_rmse_eV",
  },
  "plot-aligned": {
    title: "Spectral aligned RMSE",
    metricKey: "align_global_rmse_eV",
  },
  "plot-box": {
    title: "Distribucion Fermi/frontier",
    metric: "Fermi-window RMSE por muestra",
    formula: "\\operatorname{box}\\left(\\operatorname{RMSE}^{(s)}_F\\right);\\quad \\text{fallback } \\operatorname{RMSE}^{(s)}_{frontier}\\text{ si Fermi no esta disponible}",
    description: "Distribucion por muestra del RMSE en ventana de Fermi, con fallback frontier cuando Fermi no esta disponible.",
    purpose: "Sirve para ver dispersion, outliers y estabilidad de cada metodo, no solo el promedio.",
    direction: "Cajas mas bajas y compactas son mejores; puntos altos senalan muestras dificiles.",
  },
  "plot-scatter": {
    title: "Sparse vs spectral",
    metric: "Relative Frobenius vs global spectral RMSE",
    formula: "x=\\operatorname{mean}_s(\\text{relative Frobenius}),\\quad y=\\operatorname{mean}_s(\\text{global RMSE})",
    description: "Relaciona error matricial con error espectral global para cada run.",
    purpose: "Ayuda a detectar si mejorar la matriz tambien mejora los autovalores, o si hay desacople entre ambas cosas.",
    direction: "El cuadrante inferior izquierdo es el deseable: bajo error matricial y bajo error espectral.",
  },
  "plot-sweeps": {
    title: "Sensitivity sweeps",
    metric: "Support threshold RMSE y DOS sigma W1",
    formula: "\\text{cada punto recalcula } RMSE=\\sqrt{\\operatorname{mean}(e^2)}\\text{ o } W_1=\\int |\\operatorname{CDF}_{ref}-\\operatorname{CDF}_{pred}|\\,dE",
    description: "Muestra como cambian metricas al variar umbrales de soporte sparse o suavizado sigma de la DOS.",
    purpose: "Sirve para comprobar robustez: una conclusion fiable no deberia depender de un unico umbral arbitrario.",
    direction: "Curvas mas bajas y estables suelen indicar comportamiento mas robusto.",
  },
  "plot-heatmap": {
    title: "Resumen compacto de metricas",
    metric: "MAE, Frobenius, Support F1, RMSE, gap, DOS W1 y tiempo",
    formula: "z=\\frac{x-\\min(m)}{\\max(m)-\\min(m)},\\quad \\text{si higher-is-better: } z\\leftarrow 1-z",
    description: "Mapa normalizado por metrica para comparar varias senales en una sola vista.",
    purpose: "Ayuda a localizar compromisos: buen espectro, buena matriz, buen soporte sparse y coste razonable.",
    direction: "Verde es mejor dentro de cada columna; Support F1 invierte la escala porque ahi mayor es mejor.",
  },
  "plot-deeph-mev": {
    title: "DeepH-comparable matrix MAE/RMSE",
    metric: "Hamiltonian MAE/RMSE in meV",
    formula: "1000\\times m_{eV}\\quad\\text{en los mismos soportes sparse del repositorio}",
    description: "Aliases en meV para MAE/RMSE del Hamiltoniano en soporte referencia y union.",
    purpose: "Facilita comparar magnitudes con reportes estilo DeepH sin cambiar la definicion de soporte.",
    direction: "Menor es mejor; son diagnosticos sobre la matriz raw/global del repositorio, no H' local transformado.",
  },
  "plot-deeph-mse": {
    title: "DeepH-comparable matrix MSE",
    metricKey: "mse_union_eV2",
  },
  "plot-deeph-r2": {
    title: "Hamiltonian R2 diagnostics",
    metricKey: "r2_union",
  },
  "plot-deeph-dos": {
    title: "DOS MAE 500 Fermi window",
    metricKey: "dos_mae_500_fermi_window",
  },
  "plot-orbital-pair": {
    title: "Orbital-pair MAE heatmap",
    metric: "mae_union_meV_mean by species pair and local orbital indices",
    formula: "z_{ab}=\\operatorname{mean}_s(\\operatorname{MAE}_{union,meV}^{(s)}(a,b))",
    description: "Heatmap diagnostico desde metrics/orbital_pair_summary.csv cuando existe.",
    purpose: "Permite inspeccionar errores orbital-orbital estilo DeepH por species_pair e indices locales.",
    direction: "Menor es mejor; diagnostico repo-compatible en la base Hamiltoniana raw/global, no metrica H' exacta ni winner primario.",
  },
};

const CROSS_PLOT_HELP_BY_ID = {
  "plot-cross-heatmap": {
    title: "Cross-evaluation heatmap",
    purpose: "Compara cada metodo entrenado contra test sets congelados para medir generalizacion cruzada.",
    direction: "Celdas mas bajas son mejores para metricas de error; las celdas No metric no se sustituyen por otra metrica.",
  },
  "plot-learning": {
    title: "Learning curves",
    purpose: "Muestra como cambia la metrica seleccionada al aumentar el tamano de dataset.",
    direction: "Pendientes descendentes indican que mas datos ayudan para ese metodo/test set.",
  },
  "plot-compute": {
    title: "Metric vs total compute time",
    purpose: "Cruza calidad con coste para elegir configuraciones que compensen computacionalmente.",
    direction: "Busca puntos hacia abajo y a la izquierda: menor error y menor tiempo.",
  },
  "plot-winner": {
    title: "Winner map",
    purpose: "Resume que metodo gana para cada combinacion de test/dataset usando la metrica seleccionada.",
    direction: "El ganador es el menor valor si la metrica es de error; revisa empates, No metric y warnings antes de concluir.",
  },
};

const state = {
  offsets: Object.fromEntries(pipelines.map((pipeline) => [pipeline.key, 0])),
  experimentOffset: 0,
  polling: null,
  pollingInFlight: false,
  pollingFailures: 0,
  lastPollingToastAt: 0,
  plotsEnabled: true,
  plotData: null,
  fcMaxPerDisplacement: null,
  experimentWasRunning: false,
  performancePresetCatalog: null,
  datasetTargets: [],
  reusableDatasets: [],
  reusableDatasetsLoaded: false,
  trainingPlan: [],
  trainingPlanNextId: 1,
  materialPresets: [],
  materialValidation: null,
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
  let response;
  try {
    response = await fetch(path, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
  } catch (error) {
    const message = error?.message || "No se pudo conectar con la UI local.";
    throw new Error(message);
  }
  const payload = await response.json();
  if (!response.ok || payload.error) {
    throw new Error(payload.error || response.statusText);
  }
  return payload;
}

function materialPayloadFromControls() {
  const mode = document.getElementById("material-mode")?.value || "preset";
  if (mode === "preset") {
    return {
      mode: "preset",
      preset: String(document.getElementById("material-preset")?.value || "h2o").trim(),
    };
  }
  return {
    mode: "bundle",
    label: String(document.getElementById("material-label")?.value || "").trim(),
    fdf: String(document.getElementById("material-fdf")?.value || "").trim(),
    pseudopotential_dir: String(document.getElementById("material-pseudopotential-dir")?.value || "").trim(),
    basis_dir: String(document.getElementById("material-basis-dir")?.value || "").trim(),
    structure_type: String(document.getElementById("material-structure-type")?.value || "").trim(),
  };
}

function updateMaterialMode() {
  const mode = document.getElementById("material-mode")?.value || "preset";
  document.querySelectorAll(".material-preset-field").forEach((node) => {
    node.classList.toggle("hidden", mode !== "preset");
  });
  document.querySelectorAll(".material-bundle-field").forEach((node) => {
    node.classList.toggle("hidden", mode !== "bundle");
  });
}

function materialSpeciesLabel(species) {
  return (species || [])
    .map((item) => item?.label)
    .filter(Boolean)
    .join(", ") || "unknown";
}

function renderMaterialValidation(payload) {
  const container = document.getElementById("material-validation-status");
  if (!container) return;
  container.classList.remove("valid", "invalid", "muted");
  container.textContent = "";
  if (!payload) {
    container.classList.add("muted");
    container.textContent = "Material not validated yet.";
    return;
  }
  if (!payload.ok) {
    container.classList.add("invalid");
    container.textContent = payload.message || "Material validation failed.";
    return;
  }
  container.classList.add("valid");
  const material = payload.material || {};
  const lines = [
    `OK: ${material.label || "material"} (${material.material_source || "explicit"})`,
    `Species: ${materialSpeciesLabel(payload.species || material.species)}`,
    `Atoms: ${payload.atom_count ?? material.atom_count ?? "unknown"}`,
    `Pseudopotentials: ${Object.keys(payload.pseudopotentials || material.pseudopotentials || {}).length}`,
    `Basis files: ${Object.keys(payload.basis_files || material.basis_file_sha256 || {}).length}`,
  ];
  if (payload.warnings?.length) {
    lines.push(`Warnings: ${payload.warnings.join(" | ")}`);
  }
  for (const line of lines) {
    const row = document.createElement("div");
    row.textContent = line;
    container.appendChild(row);
  }
}

async function loadMaterialPresets() {
  const payload = await request("/api/material/presets");
  state.materialPresets = payload.presets || [];
  const select = document.getElementById("material-preset");
  if (!select) return;
  const selected = select.value || payload.default_preset || "h2o";
  select.innerHTML = "";
  for (const preset of state.materialPresets) {
    const option = document.createElement("option");
    option.value = preset.name;
    option.textContent = preset.valid
      ? `${preset.name} (${materialSpeciesLabel(preset.species)})`
      : `${preset.name} (invalid)`;
    select.appendChild(option);
  }
  if (!state.materialPresets.some((preset) => preset.name === selected)) {
    const option = document.createElement("option");
    option.value = selected;
    option.textContent = selected;
    select.appendChild(option);
  }
  select.value = selected;
}

async function validateMaterialSelection({ silent = false } = {}) {
  const payload = await request("/api/material/validate", {
    method: "POST",
    body: JSON.stringify({ material: materialPayloadFromControls() }),
  });
  state.materialValidation = payload;
  renderMaterialValidation(payload);
  if (!payload.ok) {
    throw new Error(payload.message || "Material validation failed.");
  }
  if (!silent) showToast("Material validado.");
  return payload;
}

function isTransientFetchError(error) {
  const message = String(error?.message || "").toLowerCase();
  return (
    error instanceof TypeError ||
    message.includes("failed to fetch") ||
    message.includes("networkerror") ||
    message.includes("load failed") ||
    message.includes("network")
  );
}

function handlePollingError(error) {
  state.pollingFailures += 1;
  const now = Date.now();
  if (
    state.pollingFailures < 3 &&
    isTransientFetchError(error)
  ) {
    return;
  }
  if (now - state.lastPollingToastAt < POLL_ERROR_TOAST_INTERVAL_MS) {
    return;
  }
  state.lastPollingToastAt = now;
  const message = isTransientFetchError(error)
    ? "Conexion temporal con la UI perdida; reintentando..."
    : error.message;
  showToast(message);
}

function markPollingSuccess() {
  state.pollingFailures = 0;
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
    showToast("Pipelines started");
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
        `/api/run/logs?pipeline=${pipeline.key}&since=${state.offsets[pipeline.key]}&limit=${LOG_POLL_LIMIT}`,
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

function assertOrderedRange(minValue, maxValue, label) {
  if (minValue == null || maxValue == null) return;
  if (minValue > maxValue) {
    throw new Error(`${label}: el minimo no puede ser mayor que el maximo.`);
  }
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

function parseHiddenIrrepsTerms(raw) {
  const text = String(raw || "").trim();
  if (!text) return [];
  return text.split("+").map((part) => {
    const term = part.trim();
    const match = term.match(/^(?:(\d+)\s*x\s*)?(\d+)\s*([eo])$/i);
    if (!match) {
      throw new Error(`Termino "${term}" invalido. Usa formato NxLe, por ejemplo 10x1o.`);
    }
    const mul = match[1] ? Number(match[1]) : 1;
    const ell = Number(match[2]);
    const parity = match[3].toLowerCase();
    if (!Number.isInteger(mul) || mul <= 0) {
      throw new Error(`Termino "${term}": la multiplicidad debe ser un entero positivo.`);
    }
    if (!Number.isInteger(ell) || ell < 0) {
      throw new Error(`Termino "${term}": l debe ser un entero >= 0.`);
    }
    return { mul, ell, parity, term };
  });
}

function expectedIrrepsText(channels, maxEll) {
  const width = Number.isInteger(channels) && channels > 0 ? channels : 10;
  return Array.from({ length: maxEll + 1 }, (_, ell) => {
    const parity = ell % 2 === 0 ? "e" : "o";
    return `${width}x${ell}${parity}`;
  }).join(" + ");
}

function validateHiddenIrrepsText(raw, maxEllRaw) {
  const text = String(raw || "").trim();
  if (!text) return { ok: true };
  const terms = parseHiddenIrrepsTerms(text);
  if (!terms.length) return { ok: true };

  const multipliers = new Set(terms.map((term) => term.mul));
  if (multipliers.size !== 1) {
    throw new Error("Todos los irreps deben tener la misma multiplicidad/canales: 32x0e + 32x1o + 32x2e.");
  }

  const seenEll = new Set();
  for (const term of terms) {
    if (seenEll.has(term.ell)) {
      throw new Error(`l=${term.ell} aparece mas de una vez en Hidden irreps.`);
    }
    seenEll.add(term.ell);
    const expectedParity = term.ell % 2 === 0 ? "e" : "o";
    if (term.parity !== expectedParity) {
      throw new Error(`Paridad no esperada en l=${term.ell}: usa ${term.ell}${expectedParity}.`);
    }
  }

  const maxEll = Number(maxEllRaw);
  if (Number.isFinite(maxEll) && Number.isInteger(maxEll) && maxEll >= 0) {
    const lmax = Math.max(...terms.map((term) => term.ell));
    const channels = terms[0].mul;
    const expected = expectedIrrepsText(channels, maxEll);
    if (lmax !== maxEll) {
      throw new Error(`Hidden irreps tiene lmax=${lmax}, pero Max ell=${maxEll}. Usa: ${expected}`);
    }
    for (let ell = 0; ell <= maxEll; ell += 1) {
      if (!seenEll.has(ell)) {
        throw new Error(`Falta l=${ell} para Max ell=${maxEll}. Usa: ${expected}`);
      }
    }
  }

  const dimension = terms.reduce((total, term) => total + term.mul * (2 * term.ell + 1), 0);
  return { ok: true, dimension, channels: terms[0].mul, lmax: Math.max(...terms.map((term) => term.ell)) };
}

function renderHiddenIrrepsValidation({ throwOnInvalid = false } = {}) {
  const input = document.getElementById("training-hidden-irreps");
  const alert = document.getElementById("training-hidden-irreps-alert");
  if (!input || !alert) return { ok: true };
  const field = input.closest(".field");
  try {
    const result = validateHiddenIrrepsText(input.value, document.getElementById("training-max-ell")?.value);
    field?.classList.remove("invalid");
    alert.classList.add("hidden");
    alert.textContent = "";
    input.setAttribute("aria-invalid", "false");
    return result;
  } catch (error) {
    field?.classList.add("invalid");
    alert.classList.remove("hidden");
    alert.textContent = error.message;
    input.setAttribute("aria-invalid", "true");
    if (throwOnInvalid) throw error;
    return { ok: false, message: error.message };
  }
}

function trainingSettings() {
  renderHiddenIrrepsValidation({ throwOnInvalid: true });
  const settings = {
    max_epochs: optionalPositiveInteger("training-max-epochs", "Training epochs"),
    optim_lr: optionalNumberInput("training-optim-lr", "Learning rate", { min: 0.000001 }),
    batch_size: optionalPositiveInteger("training-batch-size", "Training batch size"),
    loader_threads: optionalPositiveInteger("training-loader-threads", "Training loader threads"),
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

function trainingSettingsSummary(settings) {
  const entries = Object.entries(settings || {});
  if (!entries.length) return "defaults";
  return entries.map(([key, value]) => `${key}=${value}`).join(", ");
}

function reusableDatasetNameById() {
  return new Map((state.reusableDatasets || []).map((item) => [item.id, item.dataset_label || item.id]));
}

function plannedDatasetTargetNameById() {
  return new Map(
    (state.datasetTargets || []).map((item) => [
      item.target_id,
      item.dataset_label || item.recipe_id || item.target_id,
    ]),
  );
}

function defaultTrainingPlanLabel(settings) {
  if (settings?.max_epochs) return `epochs${settings.max_epochs}`;
  return `config${state.trainingPlanNextId}`;
}

function trainingPlanPayload() {
  return state.trainingPlan.map((entry) => ({
    label: entry.label,
    ...(entry.reusable_dataset_ids?.length ? { reusable_dataset_ids: [...entry.reusable_dataset_ids] } : {}),
    ...(entry.dataset_targets?.length ? { dataset_targets: entry.dataset_targets.map((target) => ({ ...target })) } : {}),
    training_settings: { ...entry.training_settings },
  }));
}

function renderTrainingPlan() {
  const body = document.getElementById("training-plan-list");
  const status = document.getElementById("training-plan-status");
  if (!body || !status) return;
  const nameById = reusableDatasetNameById();
  const targetNameById = plannedDatasetTargetNameById();
  body.innerHTML = "";
  status.textContent = state.trainingPlan.length
    ? `${state.trainingPlan.length} queued config${state.trainingPlan.length === 1 ? "" : "s"}`
    : "No queued training configs.";
  for (const entry of state.trainingPlan) {
    const row = document.createElement("tr");
    const reusableNames = (entry.reusable_dataset_ids || []).map((id) => nameById.get(id) || id);
    const targetNames = (entry.dataset_targets || []).map((target) => {
      const targetId = target.target_id || target.id || "";
      return targetNameById.get(targetId) || target.dataset_label || target.recipe_id || targetId;
    });
    const datasetNames = [...reusableNames, ...targetNames];
    const labelCell = document.createElement("td");
    labelCell.textContent = entry.label;
    const datasetCell = document.createElement("td");
    datasetCell.textContent = datasetNames.join(", ");
    const settingsCell = document.createElement("td");
    settingsCell.textContent = trainingSettingsSummary(entry.training_settings);
    const actionCell = document.createElement("td");
    const removeButton = document.createElement("button");
    removeButton.className = "mini-button danger";
    removeButton.type = "button";
    removeButton.dataset.removeTrainingPlan = String(entry.id);
    removeButton.textContent = "Remove";
    actionCell.appendChild(removeButton);
    row.append(labelCell, datasetCell, settingsCell, actionCell);
    body.appendChild(row);
  }
  body.querySelectorAll("[data-remove-training-plan]").forEach((button) => {
    button.addEventListener("click", () => {
      const id = Number(button.getAttribute("data-remove-training-plan"));
      state.trainingPlan = state.trainingPlan.filter((entry) => entry.id !== id);
      renderTrainingPlan();
    });
  });
}

function addTrainingPlanEntry() {
  const runMode = document.getElementById("run-mode")?.value;
  if (!["full_strict_pipeline", "train_test_metrics_plots_only"].includes(runMode)) {
    throw new Error("Training plan solo esta disponible en Full strict o Train/test/metrics/plots only.");
  }
  const datasetIds = runMode === "train_test_metrics_plots_only" ? selectedReusableDatasetIds() : [];
  const datasetTargets = runMode === "full_strict_pipeline" ? selectedPlannedDatasetTargets() : [];
  if (runMode === "train_test_metrics_plots_only" && !datasetIds.length) {
    throw new Error("Selecciona al menos un dataset reusable para esta configuracion.");
  }
  if (runMode === "full_strict_pipeline" && !datasetTargets.length) {
    throw new Error("Selecciona al menos un planned dataset para esta configuracion.");
  }
  const settings = trainingSettings();
  const labelInput = document.getElementById("training-plan-label");
  const rawLabel = String(labelInput?.value || "").trim();
  const label = rawLabel || defaultTrainingPlanLabel(settings);
  state.trainingPlan.push({
    id: state.trainingPlanNextId,
    label,
    reusable_dataset_ids: datasetIds,
    dataset_targets: datasetTargets,
    training_settings: settings,
  });
  state.trainingPlanNextId += 1;
  if (labelInput) labelInput.value = "";
  renderTrainingPlan();
}

function updateTrainingPlanPanel() {
  const panel = document.getElementById("training-plan-panel");
  if (!panel) return;
  const runMode = document.getElementById("run-mode")?.value;
  const enabled = ["full_strict_pipeline", "train_test_metrics_plots_only"].includes(runMode);
  panel.classList.toggle("hidden", !enabled);
  updatePlannedDatasetTargetPanel();
  renderTrainingPlan();
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
    label: "MD",
    containerId: "md-dataset-editor",
    sourceId: "md-dataset-table",
    addDatasetId: "md-add-dataset",
    countLabel: "Snapshots",
    valueLabel: "Temperatura (K)",
    defaultCount: "1000",
    defaultValue: "300",
  },
  fc: {
    label: "FC Cartesian",
    containerId: "fc-dataset-editor",
    sourceId: "fc-displacement-options",
    addDatasetId: "fc-add-dataset",
    countLabel: "Snapshots",
    valueLabel: "Ang",
    defaultCount: "200",
    defaultValue: "0.02",
  },
  random_cartesian: {
    label: "Random Cartesian",
    containerId: "random-dataset-editor",
    sourceId: "random-cartesian-dataset-table",
    addDatasetId: "random-add-dataset",
    countLabel: "Estructuras",
    valueLabel: "",
    defaultCount: "200",
    defaultValue: "",
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
  if (kind === "random_cartesian") {
    return (spec.blocks || []).map((block) => ({
      count: block.n_structures,
      value: "",
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

function datasetSeedPatch(spec) {
  return spec?.seed == null ? {} : { seed: spec.seed };
}

function datasetSeedValue(card) {
  return String(card?.querySelector('[data-field="dataset-seed"]')?.value || "").trim();
}

function parseDatasetSeed(rawValue, label) {
  const raw = String(rawValue ?? "").trim();
  if (!raw) return null;
  const value = Number(raw);
  if (!Number.isInteger(value) || value < 0) {
    throw new Error(`${label}: seed debe ser un entero >= 0.`);
  }
  return value;
}

function applyDatasetSeeds(kind, specs) {
  const container = datasetEditorContainer(kind);
  if (!container || !Array.isArray(specs) || !specs.length) return specs;
  const cards = Array.from(container.querySelectorAll(".dataset-card"));
  const methodLabel = datasetEditorConfig(kind)?.label || kind;
  return specs.map((spec, index) => {
    const card = cards[index];
    if (!card) return spec;
    const seed = parseDatasetSeed(datasetSeedValue(card), `${methodLabel} dataset ${index + 1}`);
    return seed == null ? spec : { ...spec, seed };
  });
}

function cardInputValue(card, field) {
  return String(card?.querySelector(`[data-field="${field}"]`)?.value || "").trim();
}

function cardChecked(card, field) {
  return Boolean(card?.querySelector(`[data-field="${field}"]`)?.checked);
}

function optionalCardNumber(card, field, label, { integer = false, min = null } = {}) {
  const raw = cardInputValue(card, field);
  if (!raw) return null;
  const value = Number(raw);
  if (!Number.isFinite(value) || (integer && !Number.isInteger(value)) || (min != null && value < min)) {
    const type = integer ? "entero" : "numero";
    const floor = min == null ? "" : ` >= ${min}`;
    throw new Error(`${label} debe ser un ${type}${floor}.`);
  }
  return value;
}

function cardDistribution(card, field, label) {
  const distribution = cardInputValue(card, field) || "gaussian";
  if (!["gaussian", "uniform"].includes(distribution)) {
    throw new Error(`${label} debe ser gaussian o uniform.`);
  }
  return distribution;
}

function randomCartesianSettingsFromCard(card, label) {
  const atomEnabled = cardChecked(card, "rc-enable-atom");
  const bondEnabled = cardChecked(card, "rc-enable-bond");
  const angleEnabled = cardChecked(card, "rc-enable-angle");
  if (!atomEnabled && !bondEnabled && !angleEnabled) {
    throw new Error(`${label}: habilita al menos atom, bond o angle displacement.`);
  }
  const bondMinDelta = optionalCardNumber(card, "rc-bond-min-delta", `${label}: bond min delta`);
  const bondMaxDelta = optionalCardNumber(card, "rc-bond-max-delta", `${label}: bond max delta`);
  const minBond = optionalCardNumber(card, "rc-min-bond", `${label}: min O-H bond`, { min: 0 }) ?? 0.7;
  const maxBond = optionalCardNumber(card, "rc-max-bond", `${label}: max O-H bond`, { min: 0 }) ?? 1.3;
  assertOrderedRange(bondMinDelta, bondMaxDelta, `${label}: bond delta`);
  assertOrderedRange(minBond, maxBond, `${label}: O-H bond`);

  const angleMinDelta = optionalCardNumber(card, "rc-angle-min-delta", `${label}: angle min delta`);
  const angleMaxDelta = optionalCardNumber(card, "rc-angle-max-delta", `${label}: angle max delta`);
  const minAngle = optionalCardNumber(card, "rc-min-angle", `${label}: min H-O-H angle`, { min: 0 }) ?? 80.0;
  const maxAngle = optionalCardNumber(card, "rc-max-angle", `${label}: max H-O-H angle`, { min: 0 }) ?? 130.0;
  assertOrderedRange(angleMinDelta, angleMaxDelta, `${label}: angle delta`);
  assertOrderedRange(minAngle, maxAngle, `${label}: H-O-H angle`);
  if (maxAngle > 180) throw new Error(`${label}: max H-O-H angle debe ser <= 180.`);

  const minDistance = optionalCardNumber(card, "rc-min-distance", `${label}: min distance`, { min: 0 }) ?? 0.65;
  const maxRmsd = optionalCardNumber(card, "rc-max-rmsd", `${label}: max RMSD`, { min: 0 });
  const maxAttempts =
    optionalCardNumber(card, "rc-max-attempts", `${label}: max attempts`, { integer: true, min: 1 }) ?? 100;
  const atomDistribution = cardDistribution(card, "rc-atom-distribution", `${label}: atom distribution`);
  const atomSigma = optionalCardNumber(card, "rc-atom-sigma", `${label}: atom sigma`, { min: 0 }) ?? 0.03;
  const atomUniformRange =
    optionalCardNumber(card, "rc-atom-uniform-range", `${label}: atom uniform range`, { min: 0 }) ?? 0.05;
  const moveAtoms = parseMoveAtomsRaw(cardInputValue(card, "rc-move-atoms"));
  const speciesFilter = splitList(cardInputValue(card, "rc-species-filter"));
  const validation = {
    min_distance_ang: minDistance,
    max_rmsd_from_reference_ang: maxRmsd,
    max_attempts_per_structure: maxAttempts,
  };
  const components = {
    atom_displacement: {
      enabled: atomEnabled,
      distribution: atomDistribution,
      sigma_ang: atomSigma,
      uniform_range_ang: atomUniformRange,
      move_atoms: moveAtoms,
      species_filter: speciesFilter,
      remove_center_of_mass_translation: cardChecked(card, "rc-remove-com"),
    },
    bond_displacement: {
      enabled: bondEnabled,
      distribution: cardDistribution(card, "rc-bond-distribution", `${label}: bond distribution`),
      sigma_ang: optionalCardNumber(card, "rc-bond-sigma", `${label}: bond sigma`, { min: 0 }) ?? 0.01,
      uniform_range_ang:
        optionalCardNumber(card, "rc-bond-uniform-range", `${label}: bond uniform range`, { min: 0 }) ?? 0.02,
      min_delta_ang: bondMinDelta,
      max_delta_ang: bondMaxDelta,
      min_bond_ang: minBond,
      max_bond_ang: maxBond,
      bonds: cardInputValue(card, "rc-bonds") || "h2o_oh",
    },
    angle_displacement: {
      enabled: angleEnabled,
      distribution: cardDistribution(card, "rc-angle-distribution", `${label}: angle distribution`),
      sigma_deg: optionalCardNumber(card, "rc-angle-sigma", `${label}: angle sigma`, { min: 0 }) ?? 3.0,
      uniform_range_deg:
        optionalCardNumber(card, "rc-angle-uniform-range", `${label}: angle uniform range`, { min: 0 }) ?? 5.0,
      min_delta_deg: angleMinDelta,
      max_delta_deg: angleMaxDelta,
      min_angle_deg: minAngle,
      max_angle_deg: maxAngle,
      angles: cardInputValue(card, "rc-angles") || "h2o_hoh",
    },
  };
  return {
    components,
    validation,
    legacy: {
      distribution: atomDistribution,
      sigma_ang: atomSigma,
      uniform_range_ang: atomUniformRange,
      min_distance_ang: minDistance,
      max_rmsd_from_reference_ang: maxRmsd,
      max_attempts_per_structure: maxAttempts,
      move_atoms: moveAtoms,
      species_filter: speciesFilter,
      remove_center_of_mass_translation: cardChecked(card, "rc-remove-com"),
    },
  };
}

function applyRandomCartesianDatasetSettings(specs) {
  const container = datasetEditorContainer("random_cartesian");
  if (!container || !Array.isArray(specs) || !specs.length) return specs;
  const cards = Array.from(container.querySelectorAll(".dataset-card"));
  return specs.map((spec, index) => {
    const card = cards[index];
    if (!card) return spec;
    const seed = parseDatasetSeed(datasetSeedValue(card), `Random Cartesian dataset ${index + 1}`);
    const rows = Array.from(card.querySelectorAll(".dataset-component-row"));
    return {
      ...spec,
      ...(seed == null ? {} : { seed }),
      blocks: (spec.blocks || []).map((block, blockIndex) => {
        const row = rows[blockIndex];
        if (!row) return block;
        const settings = randomCartesianSettingsFromCard(
          row,
          `Random Cartesian dataset ${index + 1}, component ${blockIndex + 1}`,
        );
        return {
          ...settings.legacy,
          ...block,
          components: settings.components,
          validation: settings.validation,
        };
      }),
    };
  });
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
  if (kind === "random_cartesian") {
    row.className = "dataset-component-row count-only random-block-row";
    row.innerHTML = `
      <div class="random-block-row-header">
        <label>
          <span>${config.countLabel}</span>
          <input type="number" min="1" step="1" data-field="count" value="${String(count ?? "")}" />
        </label>
        <button class="mini-button danger" type="button" data-action="remove-component">Remove</button>
      </div>
      <details class="random-block-details">
        <summary>Atom / bond / angle parameters</summary>
        ${randomCartesianDatasetControlsHtml()}
      </details>
    `;
    return row;
  }
  row.className = kind === "random_cartesian" ? "dataset-component-row count-only" : "dataset-component-row";
  const valueField = kind === "random_cartesian" ? "" : `
    <label>
      <span>${config.valueLabel}</span>
      <input type="number" min="0" step="0.001" data-field="value" value="${String(value ?? "")}" />
    </label>
  `;
  row.innerHTML = `
    <label>
      <span>${config.countLabel}</span>
      <input type="number" min="1" step="1" data-field="count" value="${String(count ?? "")}" />
    </label>
    ${valueField}
    <button class="mini-button danger" type="button" data-action="remove-component">Remove</button>
  `;
  return row;
}

function randomCartesianDatasetControlsHtml() {
  return `
    <div class="random-dataset-controls">
      <section class="random-component-panel">
        <div class="random-component-header">
          <label class="toggle-field inline-toggle">
            <input data-field="rc-enable-atom" type="checkbox" checked />
            <span>Atom displacement</span>
          </label>
        </div>
        <div class="grid two split-grid">
          <label class="field">
            <span>Atom distribution</span>
            <select data-field="rc-atom-distribution">
              <option value="gaussian" selected>Gaussian</option>
              <option value="uniform">Uniform</option>
            </select>
          </label>
          <label class="field">
            <span>Atom sigma (Ang)</span>
            <input data-field="rc-atom-sigma" type="number" step="0.001" min="0" value="0.03" />
          </label>
          <label class="field">
            <span>Atom uniform range (Ang)</span>
            <input data-field="rc-atom-uniform-range" type="number" step="0.001" min="0" value="0.05" />
          </label>
          <label class="field">
            <span>Átomos movidos</span>
            <input data-field="rc-move-atoms" type="text" value="all" />
          </label>
          <label class="field">
            <span>Filtro de especies</span>
            <input data-field="rc-species-filter" type="text" placeholder="H, O" />
          </label>
          <label class="toggle-field inline-toggle">
            <input data-field="rc-remove-com" type="checkbox" checked />
            <span>Eliminar traslación</span>
          </label>
        </div>
      </section>
      <section class="random-component-panel">
        <div class="random-component-header">
          <label class="toggle-field inline-toggle">
            <input data-field="rc-enable-bond" type="checkbox" />
            <span>Bond displacement</span>
          </label>
        </div>
        <div class="grid two split-grid">
          <label class="field">
            <span>Bonds</span>
            <select data-field="rc-bonds">
              <option value="h2o_oh" selected>H2O O-H</option>
            </select>
          </label>
          <label class="field">
            <span>Bond distribution</span>
            <select data-field="rc-bond-distribution">
              <option value="gaussian" selected>Gaussian</option>
              <option value="uniform">Uniform</option>
            </select>
          </label>
          <label class="field">
            <span>Bond sigma (Ang)</span>
            <input data-field="rc-bond-sigma" type="number" step="0.001" min="0" value="0.01" />
          </label>
          <label class="field">
            <span>Bond uniform range (Ang)</span>
            <input data-field="rc-bond-uniform-range" type="number" step="0.001" min="0" value="0.02" />
          </label>
          <label class="field">
            <span>Bond min delta (Ang)</span>
            <input data-field="rc-bond-min-delta" type="number" step="0.001" placeholder="-range" />
          </label>
          <label class="field">
            <span>Bond max delta (Ang)</span>
            <input data-field="rc-bond-max-delta" type="number" step="0.001" placeholder="+range" />
          </label>
          <label class="field">
            <span>Min O-H bond (Ang)</span>
            <input data-field="rc-min-bond" type="number" step="0.01" min="0" value="0.70" />
          </label>
          <label class="field">
            <span>Max O-H bond (Ang)</span>
            <input data-field="rc-max-bond" type="number" step="0.01" min="0" value="1.30" />
          </label>
        </div>
      </section>
      <section class="random-component-panel">
        <div class="random-component-header">
          <label class="toggle-field inline-toggle">
            <input data-field="rc-enable-angle" type="checkbox" />
            <span>Angle displacement</span>
          </label>
        </div>
        <div class="grid two split-grid">
          <label class="field">
            <span>Angles</span>
            <select data-field="rc-angles">
              <option value="h2o_hoh" selected>H2O H-O-H</option>
            </select>
          </label>
          <label class="field">
            <span>Angle distribution</span>
            <select data-field="rc-angle-distribution">
              <option value="gaussian" selected>Gaussian</option>
              <option value="uniform">Uniform</option>
            </select>
          </label>
          <label class="field">
            <span>Angle sigma (deg)</span>
            <input data-field="rc-angle-sigma" type="number" step="0.1" min="0" value="3.0" />
          </label>
          <label class="field">
            <span>Angle uniform range (deg)</span>
            <input data-field="rc-angle-uniform-range" type="number" step="0.1" min="0" value="5.0" />
          </label>
          <label class="field">
            <span>Angle min delta (deg)</span>
            <input data-field="rc-angle-min-delta" type="number" step="0.1" placeholder="-range" />
          </label>
          <label class="field">
            <span>Angle max delta (deg)</span>
            <input data-field="rc-angle-max-delta" type="number" step="0.1" placeholder="+range" />
          </label>
          <label class="field">
            <span>Min H-O-H angle (deg)</span>
            <input data-field="rc-min-angle" type="number" step="0.1" min="0" max="180" value="80.0" />
          </label>
          <label class="field">
            <span>Max H-O-H angle (deg)</span>
            <input data-field="rc-max-angle" type="number" step="0.1" min="0" max="180" value="130.0" />
          </label>
        </div>
      </section>
      <section class="random-component-panel">
        <div class="random-component-header">
          <span class="random-component-title">Validation</span>
        </div>
        <div class="grid two split-grid">
          <label class="field">
            <span>Min distance (Ang)</span>
            <input data-field="rc-min-distance" type="number" step="0.01" min="0" value="0.65" />
          </label>
          <label class="field">
            <span>Max RMSD (Ang)</span>
            <input data-field="rc-max-rmsd" type="number" step="0.01" min="0" placeholder="optional" />
          </label>
          <label class="field">
            <span>Max attempts / structure</span>
            <input data-field="rc-max-attempts" type="number" step="1" min="1" value="100" />
          </label>
        </div>
      </section>
    </div>
  `;
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
      const seed = datasetSeedValue(card);
      totalNode.textContent = `${total || 0} estructuras · ${rows.length} bloques${seed ? ` · seed ${seed}` : ""}`;
    }
    rows.forEach((row) => {
      const remove = row.querySelector('[data-action="remove-component"]');
      if (remove) remove.disabled = rows.length <= 1;
    });
    const removeDataset = card.querySelector('[data-action="remove-dataset"]');
    if (removeDataset) removeDataset.disabled = container.querySelectorAll(".dataset-card").length <= 1;
  }
}

function createDatasetCard(kind, blocks, seed = "") {
  const card = document.createElement("details");
  card.className = "dataset-card";
  card.open = true;
  card.innerHTML = `
    <summary class="dataset-card-header">
      <div class="dataset-card-title">
        <span class="dataset-card-name">Dataset</span>
        <span class="dataset-card-total">0 estructuras · 0 bloques</span>
      </div>
      <span class="dataset-card-chevron" aria-hidden="true">▾</span>
    </summary>
    <div class="dataset-card-body">
      <div class="dataset-card-meta">
        <label class="dataset-seed-field" title="Seed opcional para este dataset. Si queda vacia se usa la seed global o el comportamiento por defecto.">
          <span>Seed</span>
          <input type="number" min="0" step="1" data-field="dataset-seed" placeholder="global" value="${String(seed ?? "")}" />
        </label>
        <button class="mini-button danger" type="button" data-action="remove-dataset">Remove dataset</button>
      </div>
      <div class="dataset-components"></div>
      <div class="button-row">
        <button class="mini-button" type="button" data-action="add-component">+ Component</button>
      </div>
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
  const datasetGroups = specs.length
    ? specs.map((spec) => ({ blocks: blocksForEditorSpec(kind, spec), seed: spec.seed ?? "" }))
    : [{ blocks: defaultEditorBlocks(kind), seed: "" }];
  container.innerHTML = "";
  for (const group of datasetGroups) {
    container.appendChild(createDatasetCard(kind, group.blocks, group.seed));
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
        if (kind === "random_cartesian") return count;
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
  const controls = document.querySelectorAll(".method-execution-checkbox");
  const selector = controls.length ? ".method-execution-checkbox:checked" : ".method-checkbox:checked";
  return Array.from(document.querySelectorAll(selector)).map((item) => item.value);
}

function setMethodSelected(method, checked, source = null) {
  const selectors = [
    `.method-execution-checkbox[value="${method}"]`,
    `.method-checkbox[value="${method}"]`,
  ];
  for (const selector of selectors) {
    document.querySelectorAll(selector).forEach((node) => {
      if (node !== source) node.checked = checked;
    });
  }
}

function updateMethodSelectionSummary() {
  const node = document.getElementById("method-selection-summary");
  if (!node) return;
  const methods = selectedMethods();
  node.textContent = methods.length
    ? `${methods.length} method${methods.length === 1 ? "" : "s"} selected: ${methods.map(methodDisplayLabel).join(", ")}`
    : "No methods selected";
  node.classList.toggle("error-text", !methods.length);
}

function pipelineLabel(key) {
  return resultPipelines.find((item) => item.key === key)?.label || methodDisplayLabel(key);
}

function runDisplayLabel(run) {
  const detail = run?.training_tag || run?.dataset_label || run?.recipe_id || run?.run_id || "";
  return `${pipelineLabel(run?.pipeline || run?.label)} ${run?.dataset_size ?? ""}${detail ? ` · ${detail}` : ""}`;
}

function slugPart(value) {
  return String(value || "x")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "")
    .slice(0, 48) || "x";
}

function parseMoveAtomsRaw(rawInput) {
  const raw = String(rawInput || "").trim();
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
  const specs = parseRandomCartesianDatasetTableSpecs();
  const sizes = specs.map((spec) => spec.size).filter((size) => Number.isInteger(size) && size > 0);
  if (!sizes.length) return {};
  const firstRandomRow = datasetEditorContainer("random_cartesian")?.querySelector(".dataset-component-row");
  if (firstRandomRow) {
    const settings = randomCartesianSettingsFromCard(firstRandomRow, "Random Cartesian defaults");
    return {
      n_structures: sizes.length === 1 ? sizes[0] : sizes,
      seed: 1234,
      ...settings.legacy,
      components: settings.components,
      validation: settings.validation,
    };
  }
  return {
    n_structures: sizes.length === 1 ? sizes[0] : sizes,
    seed: 1234,
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
  return applyDatasetSeeds(
    "md",
    parseMdDatasetTableSpecsFromText(document.getElementById("md-dataset-table")?.value || ""),
  );
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
        throw new Error(`FC Cartesian dataset ${datasetIndex + 1}, fila ${rowIndex + 1}: usa "snapshots | desplazamiento".`);
      }
      const count = Number(parts[0]);
      const displacement = normalizeDisplacement(parts[1]);
      if (!Number.isInteger(count) || count <= 0) {
        throw new Error(`FC Cartesian dataset ${datasetIndex + 1}, fila ${rowIndex + 1}: snapshots debe ser entero positivo.`);
      }
      if (!displacement) {
        throw new Error(`FC Cartesian dataset ${datasetIndex + 1}, fila ${rowIndex + 1}: desplazamiento vacio.`);
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
  const specs = parseFcDatasetTableSpecsFromText(document.getElementById("fc-displacement-options")?.value || "");
  return specs ? applyDatasetSeeds("fc", specs) : specs;
}

function parseRandomCartesianDatasetTableSpecsFromText(rawText) {
  const raw = String(rawText || "").trim();
  if (!raw) return [];
  return splitDatasetTableGroups(raw).map((group, datasetIndex) => {
    const blocks = group.map((row, rowIndex) => {
      const parts = splitDatasetTableRow(row);
      if (parts.length < 1) {
        throw new Error(`Random Cartesian dataset ${datasetIndex + 1}, fila ${rowIndex + 1}: usa "estructuras".`);
      }
      const count = Number(parts[0]);
      if (!Number.isInteger(count) || count <= 0) {
        throw new Error(`Random Cartesian dataset ${datasetIndex + 1}, fila ${rowIndex + 1}: estructuras debe ser entero positivo.`);
      }
      const block = {
        block_id: `rc_d${datasetIndex + 1}_${rowIndex + 1}_${count}`,
        label: `${count} estructuras`,
        n_structures: count,
      };
      if (parts.length >= 2) {
        const amplitude = numericPrefix(parts[1], `Random Cartesian dataset ${datasetIndex + 1}, fila ${rowIndex + 1}`);
        if (!Number.isFinite(amplitude) || amplitude < 0) {
          throw new Error(`Random Cartesian dataset ${datasetIndex + 1}, fila ${rowIndex + 1}: amplitud debe ser >= 0 Ang.`);
        }
        block.block_id = `rc_d${datasetIndex + 1}_a${slugPart(amplitude)}_${rowIndex + 1}_${count}`;
        block.label = `${count} estructuras @ ${amplitude} Ang`;
        block.amplitude_ang = amplitude;
        block.max_displacement = `${amplitude} Ang`;
      }
      return block;
    });
    return {
      index: datasetIndex,
      size: blocks.reduce((sum, block) => sum + Number(block.n_structures), 0),
      blocks,
    };
  });
}

function parseRandomCartesianDatasetSpecsFromEditor() {
  const container = datasetEditorContainer("random_cartesian");
  if (!container) return null;
  const cards = Array.from(container.querySelectorAll(".dataset-card"));
  if (!cards.length) return null;
  return cards.map((card, datasetIndex) => {
    const seed = parseDatasetSeed(datasetSeedValue(card), `Random Cartesian dataset ${datasetIndex + 1}`);
    const rows = Array.from(card.querySelectorAll(".dataset-component-row"));
    const blocks = rows.map((row, rowIndex) => {
      const rawCount = String(row.querySelector('[data-field="count"]')?.value || "").trim();
      const count = Number(rawCount);
      if (!Number.isInteger(count) || count <= 0) {
        throw new Error(
          `Random Cartesian dataset ${datasetIndex + 1}, component ${rowIndex + 1}: estructuras debe ser entero positivo.`,
        );
      }
      const settings = randomCartesianSettingsFromCard(
        row,
        `Random Cartesian dataset ${datasetIndex + 1}, component ${rowIndex + 1}`,
      );
      return {
        ...settings.legacy,
        block_id: `rc_d${datasetIndex + 1}_${rowIndex + 1}_${count}`,
        label: `${count} estructuras`,
        n_structures: count,
        components: settings.components,
        validation: settings.validation,
      };
    });
    return {
      index: datasetIndex,
      size: blocks.reduce((sum, block) => sum + Number(block.n_structures), 0),
      ...(seed == null ? {} : { seed }),
      blocks,
    };
  });
}

function parseRandomCartesianDatasetTableSpecs() {
  syncDatasetEditorText("random_cartesian");
  const editorSpecs = parseRandomCartesianDatasetSpecsFromEditor();
  if (editorSpecs) return editorSpecs;
  return applyRandomCartesianDatasetSettings(
    parseRandomCartesianDatasetTableSpecsFromText(
      document.getElementById("random-cartesian-dataset-table")?.value || "",
    ),
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
        ...datasetSeedPatch(spec),
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
    const globalFcSeed = optionalNumberInput("fc-random-seed", "FC Cartesian random seed", { integer: true }) ?? 42;
    recipes.siesta_fc_cartesian = fcSpecs.map((spec) => ({
      recipe_id: `fc_${spec.index + 1}_${spec.size}`,
      label: `FC Cartesian ${spec.size} structures`,
      seed: spec.seed ?? globalFcSeed,
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
    const globalRandomSeed = base.seed;
    delete base.n_structures;
    delete base.seed;
    if (randomCartesianDatasetSpecs.length) {
      recipes.random_cartesian = randomCartesianDatasetSpecs.map((spec, index) => ({
        recipe_id: `rc_table_${index + 1}_${spec.size}`,
        label: `Random Cartesian dataset ${index + 1}: ${spec.size} structures`,
        seed: spec.seed ?? globalRandomSeed,
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
      seed: globalRandomSeed,
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

function plannedDatasetTargetId(methodId, recipeId, occurrence = 1) {
  const base = `${normalizeMethodId(methodId)}:${String(recipeId || "").trim()}`;
  return occurrence <= 1 ? base : `${base}#${occurrence}`;
}

function defaultRecipeIdForMethod(methodId, index) {
  if (methodId === "md") return `md_recipe_${index + 1}`;
  if (methodId === "siesta_fc_cartesian") return `fc_recipe_${index + 1}`;
  if (methodId === "random_cartesian") return `rc_recipe_${index + 1}`;
  return `${methodId}_recipe_${index + 1}`;
}

function recipeSize(methodId, recipe) {
  const blocks = Array.isArray(recipe?.blocks) ? recipe.blocks : [];
  return blocks.reduce((sum, block) => {
    const size = methodId === "md" ? Number(block.n_snapshots || block.n_structures || 0) : Number(block.n_structures || 0);
    return sum + (Number.isFinite(size) ? size : 0);
  }, 0);
}

function plannedDatasetTargetsFromRecipes(recipes, methods) {
  const targets = [];
  const seen = new Map();
  for (const methodId of ["md", "siesta_fc_cartesian", "random_cartesian"]) {
    if (!methods.includes(methodId)) continue;
    const methodRecipes = Array.isArray(recipes?.[methodId]) ? recipes[methodId] : [];
    methodRecipes.forEach((recipe, index) => {
      const recipeId = String(recipe?.recipe_id || defaultRecipeIdForMethod(methodId, index));
      const key = `${methodId}:${recipeId}`;
      const occurrence = (seen.get(key) || 0) + 1;
      seen.set(key, occurrence);
      targets.push({
        target_id: plannedDatasetTargetId(methodId, recipeId, occurrence),
        method_id: methodId,
        recipe_id: recipeId,
        dataset_label: recipe?.label || recipeId,
        dataset_size: recipeSize(methodId, recipe),
      });
    });
  }
  return targets;
}

function renderPlannedDatasetTargets(targets = state.datasetTargets) {
  const body = document.getElementById("planned-dataset-target-list");
  const status = document.getElementById("planned-dataset-target-status");
  if (!body || !status) return;
  const hadCheckboxes = document.querySelectorAll(".planned-dataset-target-checkbox").length > 0;
  const selectedIds = new Set(selectedPlannedDatasetTargetIds());
  body.innerHTML = "";
  state.datasetTargets = targets || [];
  status.textContent = state.datasetTargets.length
    ? `${state.datasetTargets.length} planned dataset${state.datasetTargets.length === 1 ? "" : "s"} available`
    : "No planned datasets loaded.";
  for (const target of state.datasetTargets) {
    const row = document.createElement("tr");
    const useCell = document.createElement("td");
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.className = "planned-dataset-target-checkbox";
    checkbox.value = target.target_id;
    checkbox.checked = hadCheckboxes ? selectedIds.has(target.target_id) : true;
    useCell.appendChild(checkbox);
    const methodCell = document.createElement("td");
    methodCell.textContent = methodDisplayLabel(target.method_id);
    const datasetCell = document.createElement("td");
    datasetCell.textContent = target.dataset_label || target.target_id;
    const sizeCell = document.createElement("td");
    sizeCell.textContent = String(target.dataset_size || "-");
    const recipeCell = document.createElement("td");
    recipeCell.textContent = target.recipe_id || target.target_id;
    row.append(useCell, methodCell, datasetCell, sizeCell, recipeCell);
    body.appendChild(row);
  }
}

function currentPlannedDatasetTargets() {
  const methods = selectedMethods();
  const mdDatasetSpecs = methods.includes("md") ? parseMdDatasetTableSpecs() : [];
  const mdSizes = methods.includes("md")
    ? (mdDatasetSpecs.length ? mdSizesFromSpecs(mdDatasetSpecs) : legacyMdSizesFromHidden())
    : [];
  const fcSpecs = methods.includes("siesta_fc_cartesian") ? fcDatasetSpecs() : [];
  const randomOptions = parseRandomCartesianOptions(methods);
  const randomDatasetSpecs = methods.includes("random_cartesian") ? parseRandomCartesianDatasetTableSpecs() : [];
  const recipes = datasetRecipesForRun(
    methods,
    mdSizes,
    fcSpecs,
    randomOptions,
    mdDatasetSpecs,
    randomDatasetSpecs,
  );
  return plannedDatasetTargetsFromRecipes(recipes, methods);
}

function refreshPlannedDatasetTargets({ silent = false } = {}) {
  if (document.getElementById("run-mode")?.value !== "full_strict_pipeline") {
    state.datasetTargets = [];
    renderPlannedDatasetTargets([]);
    return [];
  }
  try {
    const targets = currentPlannedDatasetTargets();
    state.datasetTargets = targets;
    renderPlannedDatasetTargets(targets);
    return targets;
  } catch (error) {
    if (!silent) throw error;
    const status = document.getElementById("planned-dataset-target-status");
    if (status) status.textContent = `Planned datasets unavailable: ${error.message}`;
    return [];
  }
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
            `${entry.value} requests ${entry.n_structures} structures, above the FC Cartesian limit ${state.fcMaxPerDisplacement}.`,
          );
        }
      }
    }
  }
}

function setPreviewText(id, text, isError = false) {
  const node = document.getElementById(id);
  if (!node) return;
  node.textContent = text;
  node.classList.toggle("error-text", isError);
}

function updateMethodCardStates() {
  const methods = selectedMethods();
  document.querySelectorAll(".method-card").forEach((card) => {
    const checkbox = card.querySelector(".method-checkbox");
    const method = checkbox?.value;
    card.classList.toggle("disabled-method", method && !methods.includes(method));
  });
  updateMethodSelectionSummary();
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
    setPreviewText("random-preview-summary", "Random Cartesian invalido", true);
  }
  updateMethodCardStates();
  if (document.getElementById("run-mode")?.value === "full_strict_pipeline") {
    refreshPlannedDatasetTargets({ silent: true });
  }
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
    text.textContent = `${pipelineLabel(status.current.pipeline)} ${label} · ${elapsed} · ETA ${eta}`;
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
      <strong>${pipelineLabel(result.pipeline)} ${label}</strong>
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
      throw new Error("Define al menos una tabla FC Cartesian con desplazamientos.");
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
        `Con train/validation/test se requieren datasets Random Cartesian >= 3. Tamaños invalidos: ${badRandomDatasets.join(", ")}.`,
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
  const runMode = document.getElementById("run-mode").value;
  if (runMode === "full_strict_pipeline") {
    state.datasetTargets = plannedDatasetTargetsFromRecipes(datasetRecipes, methods);
    renderPlannedDatasetTargets(state.datasetTargets);
  }
  const plan = ["full_strict_pipeline", "train_test_metrics_plots_only"].includes(runMode)
    ? trainingPlanPayload()
    : [];
  const reusableDatasetIds = runMode === "train_test_metrics_plots_only"
    ? selectedReusableDatasetIds()
    : [];
  const venvActivateCommandInput = document.getElementById("venv-activate-command");
  const venvActivateCommand = String(venvActivateCommandInput?.value || "").trim();
  const material = materialPayloadFromControls();
  await validateMaterialSelection({ silent: true });
  state.experimentOffset = 0;
  document.getElementById("experiment-log").textContent = "";
  const payload = await request("/api/experiment", {
    method: "POST",
    body: JSON.stringify({
      material,
      md_sizes: mdSizes,
      atom_sizes: atomSizes,
      selected_methods: methods,
      run_mode: runMode,
      reusable_dataset_ids: reusableDatasetIds,
      reusable_split_policy: reusableSplitPolicy(),
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
      training_plan: plan,
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
  const payload = await request(`/api/experiment/logs?since=${state.experimentOffset}&limit=${LOG_POLL_LIMIT}`);
  state.experimentOffset = payload.offset;
  updateExperimentStatus(payload.status);
  if (payload.lines.length) {
    const output = document.getElementById("experiment-log");
    output.textContent += payload.lines.join("");
    output.scrollTop = output.scrollHeight;
  }
  updateVenvCommandPreview();
}

async function pollOnce() {
  if (state.pollingInFlight) return;
  state.pollingInFlight = true;
  try {
    await pollLogs();
    markPollingSuccess();
  } catch (error) {
    handlePollingError(error);
  } finally {
    state.pollingInFlight = false;
  }
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
    const orbitalPairItems = items.filter((item) => item?.diagnostic_outputs?.orbital_pair_metrics?.exists);
    const orbitalPairPath = orbitalPairItems[0]?.diagnostic_outputs?.orbital_pair_metrics?.path ||
      `Comparison/results/${pipeline.resultsDir}/.../metrics/orbital_pair_metrics.csv`;
    const panel = document.createElement("section");
    panel.className = "panel result-row";
    panel.innerHTML = `
      <div>
        <p class="eyebrow">Archived</p>
        <h3>${pipeline.label}</h3>
      </div>
      <p><strong>${items.length}</strong> archived experiment runs</p>
      <p><strong>Orbital-pair diagnostics:</strong> ${orbitalPairItems.length}/${items.length} runs</p>
      <code>Comparison/results/${pipeline.resultsDir}</code>
      <code>${orbitalPairPath}</code>
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

const PLOT_COLORS = ["#4b6f8f", "#2a7f62", "#9467bd", "#d7a021", "#4f8f84", "#b15c5f", "#6370aa"];

function plotColor(index) {
  return PLOT_COLORS[index % PLOT_COLORS.length];
}

function mean(values) {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function aggregateFitPoints(points) {
  const grouped = new Map();
  for (const point of points) {
    const x = finiteNumber(point.x);
    const y = finiteNumber(point.y);
    if (x == null || y == null) continue;
    if (!grouped.has(x)) grouped.set(x, []);
    grouped.get(x).push(y);
  }
  return Array.from(grouped.entries())
    .map(([x, values]) => ({ x, y: mean(values) }))
    .sort((a, b) => a.x - b.x);
}

function solveLinearSystem(matrix, vector) {
  const n = vector.length;
  const a = matrix.map((row, index) => [...row, vector[index]]);
  for (let column = 0; column < n; column += 1) {
    let pivot = column;
    for (let row = column + 1; row < n; row += 1) {
      if (Math.abs(a[row][column]) > Math.abs(a[pivot][column])) pivot = row;
    }
    if (Math.abs(a[pivot][column]) < 1e-12) return null;
    if (pivot !== column) [a[column], a[pivot]] = [a[pivot], a[column]];
    const pivotValue = a[column][column];
    for (let col = column; col <= n; col += 1) a[column][col] /= pivotValue;
    for (let row = 0; row < n; row += 1) {
      if (row === column) continue;
      const factor = a[row][column];
      for (let col = column; col <= n; col += 1) {
        a[row][col] -= factor * a[column][col];
      }
    }
  }
  return a.map((row) => row[n]);
}

function polynomialCoefficients(points, degree) {
  if (points.length <= degree) return null;
  const size = degree + 1;
  const matrix = Array.from({ length: size }, () => Array(size).fill(0));
  const vector = Array(size).fill(0);
  for (const point of points) {
    const powers = Array.from({ length: size * 2 - 1 }, (_, power) => point.x ** power);
    for (let row = 0; row < size; row += 1) {
      for (let col = 0; col < size; col += 1) matrix[row][col] += powers[row + col];
      vector[row] += point.y * powers[row];
    }
  }
  return solveLinearSystem(matrix, vector);
}

function evaluatePolynomial(coefficients, x) {
  return coefficients.reduce((sum, coefficient, power) => sum + coefficient * (x ** power), 0);
}

function fitLinePoints(points, degree) {
  const fitPoints = aggregateFitPoints(points);
  if (fitPoints.length <= degree) return [];
  const coefficients = polynomialCoefficients(fitPoints, degree);
  if (!coefficients) return [];
  const xValues = fitPoints.map((point) => point.x);
  const minX = Math.min(...xValues);
  const maxX = Math.max(...xValues);
  if (!Number.isFinite(minX) || !Number.isFinite(maxX)) return [];
  const lineX = minX === maxX
    ? [minX]
    : Array.from({ length: 80 }, (_, index) => minX + ((maxX - minX) * index) / 79);
  return lineX.map((x) => ({ x, y: evaluatePolynomial(coefficients, x) }));
}

function fitTrace(points, name, color, kind, extra = {}) {
  const degree = kind === "quadratic" ? 2 : 1;
  const linePoints = fitLinePoints(points, degree);
  if (linePoints.length < 2) return null;
  return {
    type: "scatter",
    mode: "lines",
    name: `${name} ${kind} fit`,
    x: linePoints.map((point) => point.x),
    y: linePoints.map((point) => point.y),
    line: {
      color,
      width: 2,
      dash: kind === "quadratic" ? "dash" : "solid",
    },
    opacity: 0.42,
    hoverinfo: "skip",
    visible: kind === "linear",
    showlegend: false,
    meta: { role: "fit", fitKind: kind },
    ...extra,
  };
}

function addFitTraces(traces, points, name, color, extra = {}) {
  const linear = fitTrace(points, name, color, "linear", extra);
  const quadratic = fitTrace(points, name, color, "quadratic", extra);
  if (linear) traces.push(linear);
  if (quadratic) traces.push(quadratic);
}

function fitVisibility(traces, fitKind) {
  return traces.map((trace) => {
    if (trace.meta?.role !== "fit") return true;
    if (fitKind === "none") return false;
    return trace.meta.fitKind === fitKind;
  });
}

function withFitSelector(layout, traces) {
  if (!traces.some((trace) => trace.meta?.role === "fit")) return layout;
  return {
    ...layout,
    margin: { ...layout.margin, t: Math.max(layout.margin?.t || 46, 78) },
    updatemenus: [
      {
        type: "dropdown",
        x: 1,
        y: 1.16,
        xanchor: "right",
        yanchor: "top",
        buttons: [
          { label: "Linear fit", method: "update", args: [{ visible: fitVisibility(traces, "linear") }] },
          { label: "Quadratic fit", method: "update", args: [{ visible: fitVisibility(traces, "quadratic") }] },
          { label: "No fit", method: "update", args: [{ visible: fitVisibility(traces, "none") }] },
        ],
      },
    ],
  };
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

function runTrainingGroupLabel(run) {
  if (run?.training_plan_label) return run.training_plan_label;
  if (run?.training_index != null && run.training_index !== "") return `train${run.training_index}`;
  return "";
}

function groupedRunLabel(groupKey, items) {
  const first = items?.[0] || {};
  const pipeline = String(first.pipeline || groupKey).split("||")[0];
  const trainingLabel = groupKey.includes("||") ? groupKey.split("||").slice(1).join("||") : "";
  return trainingLabel ? `${pipelineLabel(pipeline)} · ${trainingLabel}` : pipelineLabel(pipeline);
}

function groupedRuns(runs, options = {}) {
  const groups = new Map();
  const includeTrainingContext = Boolean(options.includeTrainingContext);
  for (const run of runs) {
    const trainingContext = includeTrainingContext ? runTrainingGroupLabel(run) : "";
    const key = trainingContext ? `${run.pipeline}||${trainingContext}` : run.pipeline;
    if (!groups.has(key)) {
      groups.set(key, []);
    }
    groups.get(key).push(run);
  }
  for (const items of groups.values()) {
    items.sort((a, b) => a.dataset_size - b.dataset_size || String(a.run_id).localeCompare(String(b.run_id)));
  }
  return groups;
}

function lineTraces(runs, group, metrics) {
  const traces = [];
  let traceIndex = 0;
  for (const [groupKey, items] of groupedRuns(runs, { includeTrainingContext: true })) {
    const label = groupedRunLabel(groupKey, items);
    for (const metric of metrics) {
      const points = items
        .map((run) => ({ x: run.dataset_size, y: metricValue(run, group, metric.key), text: run.training_tag || run.run_id }))
        .filter((point) => point.y != null);
      if (!points.length) continue;
      const name = metrics.length > 1 ? `${label} · ${metric.label}` : label;
      const color = plotColor(traceIndex);
      const legendgroup = `${groupKey}-${metric.key}`;
      addFitTraces(traces, points, name, color, { legendgroup });
      traces.push({
        type: "scatter",
        mode: "markers",
        name,
        x: points.map((point) => point.x),
        y: points.map((point) => point.y),
        text: points.map((point) => point.text),
        marker: { size: 9, opacity: 0.86, color },
        legendgroup,
        hovertemplate: "dataset %{x}<br>%{y:.4g}<br>run/tag %{text}<extra>%{fullData.name}</extra>",
      });
      traceIndex += 1;
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

function topPlotAnnotation(message, y = 1.12, color = "#9f5b00") {
  return {
    text: message,
    xref: "paper",
    yref: "paper",
    x: 0,
    y,
    xanchor: "left",
    yanchor: "bottom",
    showarrow: false,
    font: { size: 12, color },
  };
}

function plotNode(id) {
  return typeof id === "string" ? document.getElementById(id) : id;
}

function metricHelp(metricKey) {
  const key = String(metricKey || "").trim();
  if (METRIC_HELP[key]) return METRIC_HELP[key];
  return {
    label: key || "Selected metric",
    formula: "\\bar{m}_{group}=\\operatorname{mean}_{\\text{finite rows}}(m)",
    description: "Metrica seleccionada en la tabla de resultados/cross-evaluation.",
    purpose: "Sirve para comparar los runs con el mismo criterio numerico.",
    direction: "Interpreta la direccion segun la definicion de la metrica en el experimento.",
  };
}

function latexText(value) {
  return String(value || "").replace(/[{}\\]/g, "");
}

function plotInfoFor(plotId) {
  const crossInfo = CROSS_PLOT_HELP_BY_ID[plotId];
  if (crossInfo) {
    const metric = selectedCrossMetric();
    const help = metricHelp(metric);
    return {
      title: crossInfo.title,
      metric: help.label,
      formula: `\\bar{m}_{group}=\\operatorname{mean}_{\\text{finite rows}}(m),\\quad m=\\text{${latexText(help.label || metric)}}`,
      description: help.description,
      purpose: crossInfo.purpose,
      direction: crossInfo.direction || help.direction,
    };
  }
  const info = PLOT_HELP_BY_ID[plotId];
  if (!info) return null;
  if (info.metricKey) {
    const help = metricHelp(info.metricKey);
    return {
      title: info.title,
      metric: help.label,
      formula: help.formula,
      description: help.description,
      purpose: help.purpose,
      direction: help.direction,
    };
  }
  return info;
}

function installPlotInfoBubble(id) {
  const node = plotNode(id);
  if (!node) return;
  node.querySelectorAll(":scope > .plot-info-bubble").forEach((bubble) => bubble.remove());
  const info = plotInfoFor(node.id);
  if (!info) return;

  const bubble = document.createElement("div");
  bubble.className = "plot-info-bubble";

  const button = document.createElement("button");
  button.type = "button";
  button.className = "plot-info-button";
  button.textContent = "i";
  button.setAttribute("aria-label", `Informacion de metrica: ${info.title}`);

  const popover = document.createElement("div");
  popover.className = "plot-info-popover";
  popover.id = `${node.id}-metric-info`;
  popover.setAttribute("role", "tooltip");
  button.setAttribute("aria-describedby", popover.id);

  const heading = document.createElement("h4");
  heading.textContent = info.title;
  popover.appendChild(heading);

  const details = document.createElement("dl");
  [
    ["Metrica", info.metric],
    ["Formula", info.formula],
    ["Que mide", info.description],
    ["Para que sirve", info.purpose],
    ["Como leerlo", info.direction],
  ].forEach(([label, value]) => {
    if (!value) return;
    const term = document.createElement("dt");
    term.textContent = label;
    const definition = document.createElement("dd");
    if (label === "Formula") {
      const formula = document.createElement("div");
      formula.className = "plot-info-formula";
      formula.textContent = `\\[${value}\\]`;
      definition.appendChild(formula);
    } else {
      definition.textContent = value;
    }
    details.append(term, definition);
  });
  popover.appendChild(details);

  bubble.append(button, popover);
  node.appendChild(bubble);
  typesetPlotInfoMath(popover);
}

function typesetPlotInfoMath(node) {
  if (!window.MathJax?.typesetPromise) {
    window.addEventListener("load", () => typesetPlotInfoMath(node), { once: true });
    return;
  }
  window.MathJax.typesetPromise([node]).catch((error) => {
    console.warn("MathJax could not render plot formula", error);
  });
}

function resizePlot(id) {
  const node = plotNode(id);
  if (!node || !window.Plotly?.Plots?.resize) return;
  Plotly.Plots.resize(node);
}

function resizeVisiblePlots() {
  if (!state.plotsEnabled) return;
  document.querySelectorAll(".plot-card.js-plotly-plot").forEach((node) => resizePlot(node));
}

function schedulePlotResize(id = null) {
  requestAnimationFrame(() => {
    if (id) {
      resizePlot(id);
      return;
    }
    resizeVisiblePlots();
  });
}

function renderPlot(id, traces, layout, config = {}) {
  const node = plotNode(id);
  if (!node || !window.Plotly) return;
  const nextLayout = {
    autosize: true,
    ...layout,
  };
  const nextConfig = {
    responsive: true,
    displaylogo: false,
    ...config,
  };
  Plotly.react(node, traces, nextLayout, nextConfig).then(() => {
    installPlotInfoBubble(node);
    schedulePlotResize(node);
  });
}

function renderEmptyPlot(id, title, message, yTitle = "") {
  renderPlot(
    id,
    [],
    plotLayout(title, yTitle, { annotations: [emptyPlotAnnotation(message)] }),
    { responsive: true, displaylogo: false },
  );
}

function metricAvailabilityByPipeline(runs, group, metric) {
  return Array.from(groupedRuns(runs)).map(([pipeline, items]) => {
    const total = items.reduce((sum, run) => sum + (run.samples?.[group] || []).length, 0);
    const finite = items.reduce((sum, run) => sum + sampleMetricValues(run, group, metric).length, 0);
    return {
      pipeline,
      label: pipelineLabel(pipeline),
      runs: items.length,
      total,
      finite,
      missing: Math.max(0, total - finite),
    };
  });
}

function metricGapAnnotation(runs, group, metric) {
  const availability = metricAvailabilityByPipeline(runs, group, metric)
    .filter((item) => item.total > 0 && item.missing > 0);
  if (!availability.length) return null;
  const missingAll = availability
    .filter((item) => item.finite === 0)
    .map((item) => `${item.label}: 0/${item.total} finitos`);
  const partial = availability
    .filter((item) => item.finite > 0)
    .map((item) => `${item.label}: ${item.finite}/${item.total} finitos`);
  const pieces = missingAll.concat(partial).slice(0, 5);
  return topPlotAnnotation(`Disponibilidad ${metric}: ${pieces.join(" | ")}`);
}

function formatMetricDisplay(value, suffix = "") {
  if (value == null || !Number.isFinite(Number(value))) return "No metric";
  const number = Number(value);
  const text = Math.abs(number) >= 1000 || Math.abs(number) < 0.001
    ? number.toExponential(2)
    : number.toPrecision(4);
  return `${text}${suffix}`;
}

function renderLinePlot(id, runs, group, metrics, title, yTitle) {
  const traces = lineTraces(runs, group, metrics);
  let layout = plotLayout(title, yTitle);
  const annotations = [];
  if (metrics.length === 1) {
    const availabilityAnnotation = metricGapAnnotation(runs, group, metrics[0].key);
    if (availabilityAnnotation) annotations.push(availabilityAnnotation);
  }
  if (!traces.length) {
    annotations.push(emptyPlotAnnotation("No hay valores finitos para esta metrica."));
  }
  if (annotations.length) {
    layout.annotations = annotations;
    layout.margin = { ...layout.margin, t: Math.max(layout.margin?.t || 46, 74) };
  }
  layout = withFitSelector(layout, traces);
  renderPlot(id, traces, layout, { responsive: true, displaylogo: false });
}

function renderR2Plot(id, runs) {
  const metrics = [
    { key: "r2_union", label: "R2 union" },
    { key: "r2_ref", label: "R2 ref" },
  ];
  const traces = lineTraces(runs, "sparse", metrics);
  const values = runs.flatMap((run) => metrics.map((metric) => metricValue(run, "sparse", metric.key))).filter((value) => value != null);
  const yaxis = { title: "R2", gridcolor: "#edf1f4", zeroline: true };
  if (values.length) {
    const minValue = Math.min(...values);
    const maxValue = Math.max(...values);
    if (minValue >= 0.8 && maxValue <= 1.05) {
      yaxis.range = [Math.max(0, minValue - 0.03), Math.min(1.05, Math.max(1.01, maxValue + 0.01))];
    }
  }
  let layout = plotLayout("DeepH-comparable matrix R2", "R2", { yaxis });
  const annotations = [];
  const availabilityAnnotation = metricGapAnnotation(runs, "sparse", "r2_union");
  if (availabilityAnnotation) annotations.push(availabilityAnnotation);
  if (!traces.length) {
    annotations.push(emptyPlotAnnotation("No hay R2 finito; puede ser no disponible para targets constantes."));
  }
  if (annotations.length) {
    layout.annotations = annotations;
    layout.margin = { ...layout.margin, t: Math.max(layout.margin?.t || 46, 74) };
  }
  layout = withFitSelector(layout, traces);
  renderPlot(id, traces, layout, { responsive: true, displaylogo: false });
}

function dosWindowUnavailableReasonAnnotation(runs) {
  const counts = new Map();
  for (const run of runs) {
    for (const row of run.samples?.dos || []) {
      const value = finiteNumber(row.dos_mae_500_fermi_window);
      const available = String(row.dos_window_metric_available ?? "").trim().toLowerCase();
      const markedUnavailable = ["false", "0", "no"].includes(available);
      const reason = String(row.dos_window_unavailable_reason || "").trim();
      if (value != null && !markedUnavailable) continue;
      if (!markedUnavailable && !reason) continue;
      const key = reason || "unavailable";
      counts.set(key, (counts.get(key) || 0) + 1);
    }
  }
  if (!counts.size) return null;
  const pieces = Array.from(counts.entries())
    .sort((left, right) => String(left[0]).localeCompare(String(right[0])))
    .map(([reason, count]) => `${reason}: ${count}`)
    .slice(0, 5);
  return topPlotAnnotation(`DOS Fermi-window unavailable: ${pieces.join(" | ")}`, 1.22);
}

function renderDosFermiMaePlot(id, runs) {
  const metrics = [{ key: "dos_mae_500_fermi_window", label: "DOS MAE 500 Fermi window" }];
  const traces = lineTraces(runs, "dos", metrics);
  let layout = plotLayout("DeepH-comparable DOS MAE", "DOS MAE");
  const annotations = [];
  const availabilityAnnotation = metricGapAnnotation(runs, "dos", "dos_mae_500_fermi_window");
  if (availabilityAnnotation) annotations.push(availabilityAnnotation);
  const reasonAnnotation = dosWindowUnavailableReasonAnnotation(runs);
  if (reasonAnnotation) annotations.push(reasonAnnotation);
  if (!traces.length) {
    annotations.push(
      emptyPlotAnnotation("No hay DOS_MAE_500_FermiWindow finito; revisa Fermi real y columnas DOS."),
    );
  }
  if (annotations.length) {
    layout.annotations = annotations;
    layout.margin = { ...layout.margin, t: Math.max(layout.margin?.t || 46, 92) };
  }
  layout = withFitSelector(layout, traces);
  renderPlot(id, traces, layout, { responsive: true, displaylogo: false });
}

function orbitalPairAxisEntry(row, prefix) {
  const species = String(row?.[`${prefix}_species`] || "").trim();
  const index = row?.[`${prefix}_orbital_index`];
  const indexText = index == null ? "" : String(index).replace(/\.0$/, "");
  const label = String(row?.[`${prefix}_orbital_label`] || "").trim() || (indexText ? `orbital_${indexText}` : "orbital");
  return {
    key: `${species}|${indexText}|${label}`,
    label: `${species ? `${species} ` : ""}${label}`,
    species,
    index: finiteNumber(index) ?? Number.POSITIVE_INFINITY,
  };
}

function orbitalPairMetricValue(row) {
  return finiteNumber(row.mae_union_meV_mean) ?? finiteNumber(row.mae_union_meV);
}

function orbitalPairRmseValue(row) {
  return finiteNumber(row.rmse_union_eV_mean) ?? finiteNumber(row.rmse_union_eV);
}

function orbitalPairR2Value(row) {
  return finiteNumber(row.r2_union_mean) ?? finiteNumber(row.r2_union);
}

function sortedOrbitalEntries(entriesByKey) {
  return Array.from(entriesByKey.values()).sort(
    (left, right) =>
      left.species.localeCompare(right.species) ||
      left.index - right.index ||
      left.label.localeCompare(right.label),
  );
}

function orbitalPairHeatmapChoices(runs) {
  const choices = [];
  for (const run of runs) {
    const grouped = new Map();
    for (const row of run.samples?.orbital_pair_summary || []) {
      if (orbitalPairMetricValue(row) == null) continue;
      const speciesPair = String(row.species_pair || `${row.row_species || "?"}-${row.col_species || "?"}`);
      if (!grouped.has(speciesPair)) grouped.set(speciesPair, []);
      grouped.get(speciesPair).push(row);
    }
    for (const [speciesPair, rows] of grouped.entries()) {
      choices.push({
        run,
        speciesPair,
        rows,
        label: `${runDisplayLabel(run)} · ${speciesPair}`,
      });
    }
  }
  return choices;
}

function orbitalPairTrace(choice, visible) {
  const rowEntries = new Map();
  const colEntries = new Map();
  for (const row of choice.rows) {
    const rowEntry = orbitalPairAxisEntry(row, "row");
    const colEntry = orbitalPairAxisEntry(row, "col");
    rowEntries.set(rowEntry.key, rowEntry);
    colEntries.set(colEntry.key, colEntry);
  }
  const yEntries = sortedOrbitalEntries(rowEntries);
  const xEntries = sortedOrbitalEntries(colEntries);
  const yIndex = new Map(yEntries.map((entry, index) => [entry.key, index]));
  const xIndex = new Map(xEntries.map((entry, index) => [entry.key, index]));
  const z = yEntries.map(() => xEntries.map(() => null));
  const customdata = yEntries.map(() => xEntries.map(() => ""));
  for (const row of choice.rows) {
    const value = orbitalPairMetricValue(row);
    if (value == null) continue;
    const rowKey = orbitalPairAxisEntry(row, "row").key;
    const colKey = orbitalPairAxisEntry(row, "col").key;
    const y = yIndex.get(rowKey);
    const x = xIndex.get(colKey);
    if (y == null || x == null) continue;
    z[y][x] = value;
    const nSamples = row.n_samples ?? "-";
    const nEntries = row.n_entries ?? "-";
    const rmse = orbitalPairRmseValue(row);
    const r2 = orbitalPairR2Value(row);
    customdata[y][x] =
      `species_pair ${choice.speciesPair}<br>` +
      `samples ${nSamples}, entries ${nEntries}<br>` +
      `RMSE ${rmse == null ? "No metric" : `${rmse.toPrecision(4)} eV`}<br>` +
      `R2 ${r2 == null ? "No metric" : r2.toPrecision(4)}`;
  }
  return {
    type: "heatmap",
    name: choice.label,
    visible,
    z,
    x: xEntries.map((entry) => entry.label),
    y: yEntries.map((entry) => entry.label),
    customdata,
    colorscale: [
      [0, "#e8f6f3"],
      [0.5, "#f1c453"],
      [1, "#b15c5f"],
    ],
    colorbar: { title: { text: "MAE<br>meV" } },
    hoverongaps: false,
    hovertemplate:
      "row %{y}<br>col %{x}<br>MAE %{z:.4g} meV<br>%{customdata}<extra>%{fullData.name}</extra>",
  };
}

function renderOrbitalPairHeatmap(id, runs) {
  const choices = orbitalPairHeatmapChoices(runs);
  if (!choices.length) {
    const hasDiagnosticFile = runs.some((run) => run.diagnostic_outputs?.orbital_pair_summary?.exists);
    const message = hasDiagnosticFile
      ? "orbital_pair_summary.csv existe, pero no hay mae_union_meV_mean finito para dibujar."
      : "No hay orbital_pair_summary.csv disponible; la salida orbital-pair sigue siendo diagnostica cuando se genere.";
    renderEmptyPlot(id, "Orbital-pair MAE heatmap", message, "MAE meV");
    return;
  }
  const traces = choices.map((choice, index) => orbitalPairTrace(choice, index === 0));
  const titleFor = (choice) => `Orbital-pair MAE meV · ${choice.label}`;
  const layout = {
    ...plotLayout(titleFor(choices[0]), "MAE meV", {
      xaxis: { title: "Column local orbital", gridcolor: "#edf1f4", zeroline: false },
      yaxis: { title: "Row local orbital", gridcolor: "#edf1f4", zeroline: false, autorange: "reversed" },
      margin: { l: 112, r: 32, t: choices.length > 1 ? 96 : 66, b: 96 },
      annotations: [
        topPlotAnnotation(
          "Diagnostico desde orbital_pair_summary.csv; no participa en winner analysis ni representa H' local exacto.",
          choices.length > 1 ? 1.18 : 1.08,
          "#56616f",
        ),
      ],
    }),
  };
  if (choices.length > 1) {
    layout.updatemenus = [
      {
        type: "dropdown",
        x: 1,
        y: 1.18,
        xanchor: "right",
        yanchor: "top",
        buttons: choices.map((choice, index) => ({
          label: choice.label,
          method: "update",
          args: [
            { visible: choices.map((_item, itemIndex) => itemIndex === index) },
            { title: { text: titleFor(choice), x: 0.02, xanchor: "left", font: { size: 15 } } },
          ],
        })),
      },
    ];
  }
  renderPlot(id, traces, layout, { responsive: true, displaylogo: false });
}

function renderBoxPlot(id, runs) {
  const traces = [];
  const availability = [];
  const fallbackRuns = [];
  for (const run of runs) {
    const spectral = sampleMetricValues(run, "spectral", "fermi_window_rmse_eV");
    const frontier = sampleMetricValues(run, "spectral", "frontier_window_rmse_eV");
    const fermiAvailability = run.metric_availability?.spectral?.fermi_window_rmse_eV ||
      run.diagnostics?.metric_availability?.spectral?.fermi_window_rmse_eV ||
      {};
    const total = finiteNumber(fermiAvailability.n_total) ?? (run.samples?.spectral || []).length;
    const finite = finiteNumber(fermiAvailability.n_finite) ?? spectral.length;
    availability.push({
      label: runDisplayLabel(run),
      finite,
      total,
    });
    if (!spectral.length && frontier.length) {
      fallbackRuns.push({
        label: runDisplayLabel(run),
        frontier: frontier.length,
        total,
      });
      traces.push({
        type: "box",
        name: `${runDisplayLabel(run)} · Frontier fallback`,
        y: frontier,
        boxpoints: "all",
        jitter: 0.35,
        pointpos: 0,
        marker: { color: "#d7a021" },
        line: { color: "#9f5b00" },
        hovertemplate: "%{y:.4g} eV<br>Frontier RMSE (HOMO/LUMO fallback)<extra>%{fullData.name}</extra>",
      });
      continue;
    }
    if (!spectral.length) continue;
    traces.push({
      type: "box",
      name: runDisplayLabel(run),
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
    margin: { l: 56, r: 18, t: 136, b: 64 },
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
  const pipelineAvailability = metricAvailabilityByPipeline(runs, "spectral", "fermi_window_rmse_eV")
    .filter((item) => item.total > 0 && item.finite === 0)
    .map((item) => {
      const fallbackCount = fallbackRuns
        .filter((run) => run.label.startsWith(item.label))
        .reduce((sum, run) => sum + run.frontier, 0);
      return fallbackCount
        ? `${item.label}: 0/${item.total} Fermi; box naranja = Frontier RMSE`
        : `${item.label}: 0/${item.total} finitos; no se dibuja box`;
    });
  if (pipelineAvailability.length) {
    annotations.push(topPlotAnnotation(pipelineAvailability.join(" | "), 1.32));
  }
  if (fallbackRuns.length) {
    annotations.push(
      topPlotAnnotation(
        "Fallback explicito: cuando la ventana ±2 eV de Fermi esta vacia, se muestra Frontier RMSE (HOMO/LUMO) en naranja; no se reescribe Fermi-window RMSE.",
        1.42,
        "#9f5b00",
      ),
    );
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
  renderPlot(
    id,
    traces,
    layout,
    { responsive: true, displaylogo: false },
  );
}

function renderScatterPlot(id, runs) {
  const traces = [];
  let traceIndex = 0;
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
    const color = plotColor(traceIndex);
    const points = x.map((value, index) => ({ x: value, y: y[index] }));
    addFitTraces(traces, points, label, color, { legendgroup: pipeline });
    traces.push({
      type: "scatter",
      mode: "markers",
      name: label,
      x,
      y,
      text,
      marker: { size: 9, opacity: 0.82, color },
      legendgroup: pipeline,
      hovertemplate: "%{text}<br>Frobenius %{x:.4g}<br>Global spectral RMSE %{y:.4g} eV<extra>%{fullData.name}</extra>",
    });
    traceIndex += 1;
  }
  let layout = plotLayout("Relacion matriz-espectro", "Global spectral RMSE eV", {
    xaxis: { title: "Relative Frobenius error", gridcolor: "#edf1f4", zeroline: false },
    legend: { orientation: "h", y: -0.25 },
  });
  if (!traces.length) {
    layout.annotations = [
      emptyPlotAnnotation("No hay pares matriz-espectro comparables."),
    ];
  }
  layout = withFitSelector(layout, traces);
  renderPlot(
    id,
    traces,
    layout,
    { responsive: true, displaylogo: false },
  );
}

function renderHeatmap(id, runs) {
  const metrics = [
    { group: "sparse", key: "mae_ref_eV", label: "MAE ref", better: "lower" },
    { group: "sparse", key: "relative_frobenius_union", label: "Frobenius rel.", better: "lower" },
    { group: "sparse", key: "support_f1", label: "Support F1", better: "higher" },
    { group: "spectral", key: "low_energy_rmse_eV", label: "Low-energy RMSE", better: "lower" },
    { group: "spectral", key: "fermi_window_rmse_eV", label: "Fermi RMSE", better: "lower" },
    { group: "spectral", key: "frontier_window_rmse_eV", label: "Frontier RMSE", better: "lower" },
    { group: "spectral", key: "gap_abs_error_eV", label: "Gap error", better: "lower" },
    { group: "dos", key: "dos_wasserstein_eV", label: "DOS W1", better: "lower" },
    {
      group: "run",
      key: "pipeline_elapsed_seconds",
      label: "Time s",
      better: "lower",
      transform: "log10_positive",
      note: "color uses log10(seconds), normalized within this metric",
      suffix: " s",
    },
  ];
  const rows = runs
    .filter((run) => metrics.some((metric) => metricValue(run, metric.group, metric.key) != null))
    .sort((a, b) => a.pipeline.localeCompare(b.pipeline) || a.dataset_size - b.dataset_size);
  if (!rows.length) {
    renderEmptyPlot(id, "Resumen compacto de metricas", "No hay metricas archivadas para resumir.", "");
    return;
  }
  const transformedColumns = metrics.map((metric) =>
    rows
      .map((run) => metricValue(run, metric.group, metric.key))
      .filter((value) => value != null)
      .map((value) => metric.transform === "log10_positive" ? Math.log10(Math.max(value, 1e-12)) : value),
  );
  const ranges = transformedColumns.map((values) => {
    if (!values.length) return { min: null, max: null };
    return { min: Math.min(...values), max: Math.max(...values) };
  });
  const z = rows.map((run) =>
    metrics.map((metric, columnIndex) => {
      const value = metricValue(run, metric.group, metric.key);
      if (value == null) return null;
      const transformed = metric.transform === "log10_positive" ? Math.log10(Math.max(value, 1e-12)) : value;
      const range = ranges[columnIndex];
      if (range.min == null || range.max == null) return null;
      if (Math.abs(range.max - range.min) < 1e-15) return 0;
      const scaled = (transformed - range.min) / (range.max - range.min);
      return metric.better === "higher" ? 1 - scaled : scaled;
    }),
  );
  const text = rows.map((run) =>
    metrics.map((metric) => {
      const value = metricValue(run, metric.group, metric.key);
      return value == null ? "" : formatMetricDisplay(value, metric.suffix || "");
    }),
  );
  const customdata = rows.map((run) =>
    metrics.map((metric) => {
      const value = metricValue(run, metric.group, metric.key);
      return {
        raw: formatMetricDisplay(value, metric.suffix || ""),
        note: metric.note || "color normalized within this metric",
        direction: metric.better === "higher" ? "higher is better" : "lower is better",
      };
    }),
  );
  renderPlot(
    id,
    [
      {
        type: "heatmap",
        z,
        text,
        texttemplate: "%{text}",
        textfont: { size: 10 },
        customdata,
        x: metrics.map((item) => item.label),
        y: rows.map((run) => runDisplayLabel(run)),
        zmin: 0,
        zmax: 1,
        colorscale: [
          [0, "#1f9e89"],
          [0.5, "#f1c453"],
          [1, "#c73e3a"],
        ],
        colorbar: {
          title: { text: "normalizado<br>por metrica" },
          tickvals: [0, 0.5, 1],
          ticktext: ["mejor", "medio", "peor"],
        },
        hoverongaps: false,
        hovertemplate:
          "%{y}<br>%{x}: %{customdata.raw}<br>" +
          "normalizado: %{z:.3f}<br>%{customdata.direction}<br>%{customdata.note}<extra></extra>",
      },
    ],
    {
      title: { text: "Resumen compacto de metricas", x: 0.02, xanchor: "left", font: { size: 15 } },
      margin: { l: 120, r: 18, t: 74, b: 72 },
      paper_bgcolor: "#ffffff",
      plot_bgcolor: "#ffffff",
      font: { family: "Inter, sans-serif", color: "#17202a" },
      annotations: [
        topPlotAnnotation(
          "Color = valor normalizado por metrica; el tiempo usa log10(segundos). Los valores de las celdas son los valores fisicos.",
          1.08,
          "#56616f",
        ),
      ],
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
      const sparsePoints = ts.map((t) => ({
        x: t,
        y: sparseByThreshold.get(t).reduce((s, v) => s + v, 0) / sparseByThreshold.get(t).length,
      }));
      const sparseColor = plotColor(traces.length);
      addFitTraces(traces, sparsePoints, `${label} sparse-threshold RMSE`, sparseColor, {
        xaxis: "x1",
        yaxis: "y1",
        legendgroup: `${pipeline}-sparse-threshold`,
      });
      traces.push({
        type: "scatter",
        mode: "markers",
        name: `${label} sparse-threshold RMSE`,
        x: sparsePoints.map((point) => point.x),
        y: sparsePoints.map((point) => point.y),
        marker: { size: 9, opacity: 0.86, color: sparseColor },
        xaxis: "x1",
        yaxis: "y1",
        legendgroup: `${pipeline}-sparse-threshold`,
      });
    }
    const ss = Array.from(dosBySigma.keys()).sort((a, b) => a - b);
    if (ss.length) {
      const dosPoints = ss.map((s) => ({
        x: s,
        y: dosBySigma.get(s).reduce((sum, v) => sum + v, 0) / dosBySigma.get(s).length,
      }));
      const dosColor = plotColor(traces.length);
      addFitTraces(traces, dosPoints, `${label} DOS sigma W1`, dosColor, {
        xaxis: "x2",
        yaxis: "y2",
        legendgroup: `${pipeline}-dos-sigma`,
      });
      traces.push({
        type: "scatter",
        mode: "markers",
        name: `${label} DOS sigma W1`,
        x: dosPoints.map((point) => point.x),
        y: dosPoints.map((point) => point.y),
        marker: { size: 9, opacity: 0.86, color: dosColor },
        xaxis: "x2",
        yaxis: "y2",
        legendgroup: `${pipeline}-dos-sigma`,
      });
    }
  }
  let layout = {
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
  layout = withFitSelector(layout, traces);
  renderPlot(id, traces, layout, { responsive: true, displaylogo: false });
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
    PRIMARY_METRIC_DEFAULT
  );
}

function selectedCrossMetric() {
  return document.getElementById("plot-cross-metric")?.value || CROSS_PLOT_METRIC_DEFAULT;
}

function metricHigherIsBetter(metric) {
  const key = String(metric || "");
  return key === "support_f1" || key.startsWith("r2_");
}

function crossMethodLabel(method) {
  return methodDisplayLabel(method);
}

function crossUnavailableMessage(payload) {
  const diagnostics = payload?.plot_diagnostics?.cross || [];
  if (!diagnostics.length) {
    return "No hay summary/cross_evaluation_metrics.csv. Ejecuta un experimento full con al menos dos metodos hasta completar cross-evaluation.";
  }
  const latest = diagnostics[diagnostics.length - 1];
  const warning = Array.isArray(latest.warnings) && latest.warnings.length
    ? ` Detalle: ${canonicalDisplayText(latest.warnings[0])}`
    : "";
  const counts = latest.archived_runs
    ? ` Runs archivados detectados: ${latest.archived_runs}.`
    : "";
  return `${canonicalDisplayText(latest.message || "Faltan datos de cross-evaluation.")}${counts}${warning}`;
}

function missingPrimaryMetricAnnotation(metric) {
  return emptyPlotAnnotation(`La metrica primaria ${metric} no tiene valores finitos; no se sustituye por otra metrica.`);
}

function missingPlotMetricAnnotation(metric) {
  return emptyPlotAnnotation(`La metrica ${metric} no tiene valores finitos en las filas seleccionadas; se muestra como No metric.`);
}

function crossTrainMethods(experiment) {
  const rows = experiment?.metrics || [];
  const methods = Array.from(new Set(rows.map((row) => normalizeMethodId(row.train_method)).filter(Boolean))).sort();
  return methods.length ? methods : ["md", "siesta_fc_cartesian", "random_cartesian"];
}

function crossTestSets(experiment) {
  const rows = experiment?.metrics || [];
  const sets = Array.from(new Set(rows.map((row) => normalizeTestSetId(row.test_set)).filter(Boolean))).sort();
  return sets.length ? sets : ["test_md", "test_siesta_fc_cartesian", "test_random_cartesian", "test_mixed"];
}

function groupedCrossMetrics(rows, metric) {
  const groups = new Map();
  for (const row of rows || []) {
    const value = finiteNumber(row[metric]);
    const trainMethod = normalizeMethodId(row.train_method);
    const testSet = normalizeTestSetId(row.test_set);
    const mdDatasetSize = finiteNumber(row.md_dataset_size ?? row.dataset_size);
    const atomDatasetSize = finiteNumber(row.atom_dataset_size ?? row.dataset_size);
    const randomDatasetSize = finiteNumber(row.random_dataset_size);
    const trainDatasetSize = finiteNumber(row.train_dataset_size ?? row.dataset_size);
    const trainTrainingTag = row.train_training_tag || row.training_tag || "";
    const trainTrainingPlanLabel = row.train_training_plan_label || row.training_plan_label || "";
    const trainTrainingPlanSettings = row.train_training_plan_settings || row.training_plan_settings || "";
    const key = [
      row.experiment_id,
      row.recipe_set_hash,
      row.md_recipe_set_hash,
      row.atom_recipe_set_hash,
      row.random_recipe_set_hash,
      row.training_tag_by_method,
      row.training_plan_label_by_method,
      row.training_plan_settings_by_method,
      mdDatasetSize,
      atomDatasetSize,
      randomDatasetSize,
      trainDatasetSize,
      trainTrainingTag,
      trainTrainingPlanLabel,
      trainTrainingPlanSettings,
      trainMethod,
      testSet,
    ].join("||");
    if (!groups.has(key)) {
      groups.set(key, {
        experiment_id: row.experiment_id,
        recipe_set_hash: row.recipe_set_hash,
        dataset_size: trainDatasetSize,
        md_dataset_size: mdDatasetSize,
        atom_dataset_size: atomDatasetSize,
        random_dataset_size: randomDatasetSize,
        train_dataset_label: row.train_dataset_label || trainMethod,
        md_dataset_label: row.md_dataset_label,
        atom_dataset_label: row.atom_dataset_label,
        random_dataset_label: row.random_dataset_label,
        train_training_tag: trainTrainingTag,
        train_training_plan_label: trainTrainingPlanLabel,
        train_training_plan_settings: trainTrainingPlanSettings,
        train_method: trainMethod,
        test_set: testSet,
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
    train_training_tag: group.train_training_tag,
    train_training_plan_label: group.train_training_plan_label,
    train_training_plan_settings: group.train_training_plan_settings,
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
  const trainingLabel = row.train_training_tag || row.train_training_plan_label || "";
  return [
    row.md_dataset_size != null ? `MD ${row.md_dataset_size}` : "",
    row.atom_dataset_size != null ? `FC Cartesian ${row.atom_dataset_size}` : "",
    row.random_dataset_size != null ? `Random Cartesian ${row.random_dataset_size}` : "",
    trainingLabel ? `train ${trainingLabel}` : "",
    row.recipe_set_hash || "",
  ].filter(Boolean).join(" / ") || `dataset ${row.dataset_size ?? "unknown"}`;
}

function crossSizeLabel(row) {
  return `${testSetDisplayLabel(row.test_set || "test")} · ${crossDatasetComboLabel(row)}`;
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

function renderCrossHeatmap(id, experiment, unavailableMessage = "") {
  const metric = selectedCrossMetric(experiment);
  if (!experiment || !(experiment.metrics || []).length) {
    renderEmptyPlot(
      id,
      `Cross-evaluation heatmap (${metric})`,
      unavailableMessage || "No hay tabla cross_evaluation_metrics.csv completa.",
      metric,
    );
    return;
  }
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
  const trainMethodLabels = trainMethods.map(crossMethodLabel);
  const customdata = sizeLabels.map((label) =>
    trainMethods.map((method) => {
      const row = means.find((item) => crossSizeLabel(item) === label && item.train_method === method);
      if (!row) {
        return { label, method: crossMethodLabel(method), valueText: "No row", availability: "0/0" };
      }
      return {
        label,
        method: crossMethodLabel(method),
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
        x: trainMethodLabels[colIndex],
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
  renderPlot(
    id,
    [{
      type: "heatmap",
      z,
      text,
      customdata,
      x: trainMethodLabels,
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

function renderCrossLearning(id, experiment, unavailableMessage = "") {
  const metric = selectedCrossMetric(experiment);
  if (!experiment || !(experiment.metrics || []).length) {
    renderEmptyPlot(
      id,
      `Learning curves (${metric})`,
      unavailableMessage || "No hay cross_evaluation_metrics.csv para construir curvas de aprendizaje.",
      metric,
    );
    return;
  }
  const means = groupedCrossMetrics(experiment?.metrics || [], metric);
  const traces = [];
  let traceIndex = 0;
  for (const method of crossTrainMethods(experiment)) {
    for (const testSet of crossTestSets(experiment)) {
      const points = means
        .filter((row) => row.train_method === method && row.test_set === testSet)
        .filter((row) => row.metric_available)
        .sort((a, b) => (a.dataset_size ?? 0) - (b.dataset_size ?? 0));
      if (!points.length) continue;
      const name = `${crossMethodLabel(method)} on ${testSetDisplayLabel(testSet)}`;
      const color = plotColor(traceIndex);
      const legendgroup = `${method}-${testSet}`;
      addFitTraces(
        traces,
        points.map((row) => ({ x: row.dataset_size, y: row.mean })),
        name,
        color,
        { legendgroup },
      );
      traces.push({
        type: "scatter",
        mode: "markers",
        name,
        x: points.map((row) => row.dataset_size),
        y: points.map((row) => row.mean),
        text: points.map((row) => `${crossDatasetComboLabel(row)} · ${metricAvailabilityLabel(row)} finite`),
        marker: { size: 9, opacity: 0.86, color },
        legendgroup,
        hovertemplate: "dataset %{x}<br>%{y:.4g}<br>%{text}<extra>%{fullData.name}</extra>",
      });
      traceIndex += 1;
    }
  }
  let layout = plotLayout(`Learning curves (${metric})`, metric);
  const missingAnnotation = crossMissingGroupsAnnotation(means, metric);
  if (!traces.length) {
    layout.annotations = [(experiment?.metrics || []).length ? missingPlotMetricAnnotation(metric) : emptyPlotAnnotation("No hay curvas cruzadas disponibles.")];
  } else if (missingAnnotation) {
    layout.annotations = [missingAnnotation];
  }
  layout = withFitSelector(layout, traces);
  renderPlot(id, traces, layout, { responsive: true, displaylogo: false });
}

function renderCrossCompute(id, experiment, unavailableMessage = "") {
  const metric = selectedCrossMetric(experiment);
  if (!experiment || !(experiment.metrics || []).length) {
    renderEmptyPlot(
      id,
      `Metric vs total compute time (${metric})`,
      unavailableMessage || "No hay cross_evaluation_metrics.csv para leer total_time_seconds.",
      metric,
    );
    return;
  }
  const allMeans = groupedCrossMetrics(experiment?.metrics || [], metric);
  const means = allMeans.filter((row) => row.time != null);
  if (!means.length) {
    renderEmptyPlot(
      id,
      `Metric vs total compute time (${metric})`,
      "Falta total_time_seconds finito en cross_evaluation_metrics.csv; no se puede comparar metrica frente a coste total.",
      metric,
    );
    return;
  }
  const traces = [];
  let traceIndex = 0;
  for (const method of crossTrainMethods(experiment)) {
    const points = means
      .filter((row) => row.train_method === method && row.metric_available)
      .sort((a, b) => a.time - b.time);
    if (!points.length) continue;
    const name = crossMethodLabel(method);
    const color = plotColor(traceIndex);
    addFitTraces(
      traces,
      points.map((row) => ({ x: row.time, y: row.mean })),
      name,
      color,
      { legendgroup: method },
    );
    traces.push({
      type: "scatter",
      mode: "markers",
      name,
      x: points.map((row) => row.time),
      y: points.map((row) => row.mean),
      text: points.map((row) => `${testSetDisplayLabel(row.test_set)}, ${crossDatasetComboLabel(row)} · ${metricAvailabilityLabel(row)} finite`),
      marker: { size: 9, opacity: 0.86, color },
      legendgroup: method,
      hovertemplate: "%{text}<br>%{x:.2f}s<br>%{y:.4g}<extra>%{fullData.name}</extra>",
    });
    traceIndex += 1;
  }
  let layout = plotLayout(`Metric vs total compute time (${metric})`, metric, {
    xaxis: { title: "Total compute seconds", gridcolor: "#edf1f4", zeroline: false },
  });
  const missingAnnotation = crossMissingGroupsAnnotation(means, metric);
  if (!traces.length) {
    layout.annotations = [(experiment?.metrics || []).length ? missingPlotMetricAnnotation(metric) : emptyPlotAnnotation("No hay timing cruzado disponible.")];
  } else if (missingAnnotation) {
    layout.annotations = [missingAnnotation];
  }
  layout = withFitSelector(layout, traces);
  renderPlot(id, traces, layout, { responsive: true, displaylogo: false });
}

function renderWinnerMap(id, experiment, unavailableMessage = "") {
  const metric = selectedCrossMetric(experiment);
  if (!experiment || !(experiment.metrics || []).length) {
    renderEmptyPlot(
      id,
      `Winner map (${metric})`,
      unavailableMessage || "No hay celdas cross comparables porque falta cross_evaluation_metrics.csv.",
      "Winner",
    );
    return;
  }
  const scientificStatus = experiment?.recommendation?.scientific_status;
  const means = groupedCrossMetrics(experiment?.metrics || [], metric);
  const higherIsBetter = metricHigherIsBetter(metric);
  const methods = crossTrainMethods(experiment);
  const methodLabel = crossMethodLabel;
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
    const best = higherIsBetter
      ? Math.max(...available.map((row) => row.mean))
      : Math.min(...available.map((row) => row.mean));
    const winners = available.filter((row) => Math.abs(row.mean - best) < 1e-12);
    if (winners.length !== 1) return tieIndex;
    const index = methods.indexOf(winners[0].train_method);
    return index >= 0 ? index : null;
  }));
  const yLabels = testSets.map(testSetDisplayLabel);
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
      testSet: testSetDisplayLabel(testSet),
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
  renderPlot(
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
  return Array.from(new Set(blockers.concat(warnings.map(canonicalDisplayText))));
}

function plotWarningStatusLabel(status) {
  return ({
    exploratory_only: "exploratory_only",
    scientifically_inconclusive: "scientifically_inconclusive",
    invalid_incomplete_grid: "invalid_incomplete_grid",
    invalid_leakage: "invalid_leakage",
    not_scientifically_valid: "not_scientifically_valid",
  }[status] || status || "diagnostic");
}

function plotWarningDetailText(details) {
  if (details == null || details === "") return "";
  if (Array.isArray(details)) return details.map(canonicalDisplayText).join(", ");
  if (typeof details === "object") {
    try {
      return canonicalDisplayText(JSON.stringify(details));
    } catch {
      return canonicalDisplayText(String(details));
    }
  }
  return canonicalDisplayText(details);
}

function warningKey(warning) {
  return [
    warning?.experiment_id || "",
    warning?.code || "",
    warning?.scientific_status || "",
    warning?.message || "",
    plotWarningDetailText(warning?.details || ""),
  ].join("||");
}

function plotWarningEntriesForPayload(payload, crossExperiment) {
  const warnings = [];
  const seen = new Set();
  const addWarning = (warning, experimentId = "") => {
    if (!warning) return;
    const entry = {
      experiment_id: warning.experiment_id || experimentId,
      severity: warning.severity || "warning",
      code: warning.code || "plot_warning",
      scientific_status: warning.scientific_status || crossExperiment?.plot_scientific_status || "",
      message: canonicalDisplayText(warning.message || ""),
      details: warning.details,
    };
    const key = warningKey(entry);
    if (seen.has(key)) return;
    seen.add(key);
    warnings.push(entry);
  };
  if (crossExperiment) {
    for (const warning of crossExperiment.plot_warnings || []) {
      addWarning(warning, crossExperiment.experiment_id);
    }
    const status = crossExperiment.plot_scientific_status || crossExperiment.recommendation?.scientific_status;
    if (status && status !== "robust_comparison") {
      addWarning(
        {
          severity: status.startsWith("invalid") ? "error" : "warning",
          code: "selected_plot_status",
          scientific_status: status,
          message: `Selected plot set is ${plotWarningStatusLabel(status)}; plots remain diagnostic.`,
        },
        crossExperiment.experiment_id,
      );
    }
  }
  for (const warning of payload?.plot_warnings || []) {
    if (!crossExperiment || warning.experiment_id === crossExperiment.experiment_id || warning.code === "visualization_compatibility") {
      addWarning(warning);
    }
  }
  return warnings;
}

function renderPlotWarnings(payload, crossExperiment) {
  const banner = document.getElementById("plot-warnings");
  if (!banner) return;
  const warnings = plotWarningEntriesForPayload(payload, crossExperiment);
  banner.replaceChildren();
  banner.classList.toggle("hidden", warnings.length === 0);
  if (!warnings.length) return;

  const title = document.createElement("strong");
  const status = crossExperiment?.plot_scientific_status || warnings[0]?.scientific_status || "diagnostic";
  title.textContent = `Scientific plot status: ${plotWarningStatusLabel(status)}`;
  banner.appendChild(title);

  const list = document.createElement("ul");
  for (const warning of warnings.slice(0, 8)) {
    const item = document.createElement("li");
    const pieces = [
      warning.experiment_id ? `${warning.experiment_id}:` : "",
      warning.message,
      warning.scientific_status ? `[${plotWarningStatusLabel(warning.scientific_status)}]` : "",
      plotWarningDetailText(warning.details),
    ].filter(Boolean);
    item.textContent = pieces.join(" ");
    list.appendChild(item);
  }
  if (warnings.length > 8) {
    const item = document.createElement("li");
    item.textContent = `${warnings.length - 8} additional plot warning(s) omitted from this banner.`;
    list.appendChild(item);
  }
  banner.appendChild(list);
}

function renderPlots(payload) {
  const panel = document.getElementById("plots-panel");
  const status = document.getElementById("plots-status");
  panel.classList.toggle("hidden", !state.plotsEnabled);
  if (!state.plotsEnabled) {
    status.textContent = "Plots disabled";
    renderPlotWarnings(null, null);
    return;
  }
  if (!window.Plotly) {
    status.textContent = "Plotly no esta disponible";
    renderPlotWarnings(payload, null);
    return;
  }
  const runs = payload?.runs || [];
  const crossExperiment = selectedCrossExperimentSet(payload);
  const crossMetric = selectedCrossMetric(crossExperiment);
  const primaryMetric = primaryCrossMetric(crossExperiment);
  const recommendation = crossExperiment?.recommendation;
  renderPlotWarnings(payload, crossExperiment);
  const crossRows = crossExperiment?.metrics?.length || 0;
  const crossSources = crossExperiment?.source_experiments?.length || 0;
  const isolationText = crossExperiment?.isolation_warning ? ` | ${canonicalDisplayText(crossExperiment.isolation_warning)}` : "";
  const blockerText = recommendation ? recommendationBlockers(recommendation).slice(0, 6).join(" | ") : "";
  const crossMissingText = !crossExperiment ? crossUnavailableMessage(payload) : "";
  const plotScientificStatus = crossExperiment?.plot_scientific_status || recommendation?.scientific_status || "unknown";
  const crossText = recommendation?.status
    ? ` | cross: ${crossRows} filas del experimento seleccionado (${crossSources} disponibles) | plot metric: ${crossMetric} | primary: ${primaryMetric} | scientific: ${plotScientificStatus} | blockers: ${blockerText || "none"} | ${recommendation.status} - ${canonicalDisplayText(recommendation.reason || "")}${isolationText}`
    : crossMissingText
      ? ` | cross: ${crossMissingText}`
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
  renderLinePlot(
    "plot-deeph-mev",
    runs,
    "sparse",
    [
      { key: "mae_union_meV", label: "MAE union" },
      { key: "rmse_union_meV", label: "RMSE union" },
      { key: "mae_ref_meV", label: "MAE ref" },
      { key: "rmse_ref_meV", label: "RMSE ref" },
    ],
    "DeepH-comparable matrix MAE/RMSE",
    "meV",
  );
  renderLinePlot(
    "plot-deeph-mse",
    runs,
    "sparse",
    [
      { key: "mse_union_eV2", label: "MSE union" },
      { key: "mse_ref_eV2", label: "MSE ref" },
    ],
    "DeepH-comparable matrix MSE",
    "MSE eV^2",
  );
  renderR2Plot("plot-deeph-r2", runs);
  renderDosFermiMaePlot("plot-deeph-dos", runs);
  renderOrbitalPairHeatmap("plot-orbital-pair", runs);
  renderLinePlot("plot-frontier", runs, "spectral", [{ key: "frontier_window_rmse_eV", label: "Frontier RMSE" }], "Frontier window", "RMSE eV");
  renderLinePlot("plot-aligned", runs, "spectral", [{ key: "align_global_rmse_eV", label: "Aligned global RMSE" }], "Spectral aligned RMSE", "RMSE eV");
  renderSensitivitySweeps("plot-sweeps", runs);
  renderCrossHeatmap("plot-cross-heatmap", crossExperiment, crossMissingText);
  renderCrossLearning("plot-learning", crossExperiment, crossMissingText);
  renderCrossCompute("plot-compute", crossExperiment, crossMissingText);
  renderWinnerMap("plot-winner", crossExperiment, crossMissingText);
  schedulePlotResize();
}

async function loadPlots() {
  const payload = await request("/api/plots");
  state.plotData = payload;
  renderPlots(payload);
}

function reusableDatasetLabel(item) {
  const tag = item.training_tag ? ` · ${item.training_tag}` : "";
  const run = item.run_id ? ` · run ${item.run_id}` : "";
  const recipe = item.recipe_id ? ` · ${item.recipe_id}` : "";
  return `${item.dataset_label || item.id}${tag}${run}${recipe}`;
}

function renderReusableDatasets(datasets) {
  const body = document.getElementById("reusable-dataset-list");
  const status = document.getElementById("reusable-dataset-status");
  if (!body || !status) return;
  const methods = new Set(selectedMethods());
  const selectedIds = new Set(selectedReusableDatasetIds());
  const activeSelectedIds = new Set(
    datasets
      .filter((item) => item.eligible && methods.has(item.method_id) && selectedIds.has(item.id))
      .map((item) => item.id),
  );
  body.innerHTML = "";
  const eligible = datasets.filter((item) => item.eligible);
  status.textContent = eligible.length
    ? `${eligible.length} reusable dataset${eligible.length === 1 ? "" : "s"} available · ${activeSelectedIds.size} selected`
    : "No reusable archived datasets found.";
  for (const item of datasets) {
    const row = document.createElement("tr");
    row.classList.toggle("muted-text", !item.eligible);

    const selectCell = document.createElement("td");
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.className = "reusable-dataset-checkbox";
    checkbox.value = item.id;
    checkbox.dataset.method = item.method_id;
    checkbox.disabled = !item.eligible || !methods.has(item.method_id);
    checkbox.checked = activeSelectedIds.has(item.id);
    checkbox.setAttribute("aria-label", `Reuse ${item.dataset_label}`);
    checkbox.addEventListener("change", () => renderReusableDatasets(state.reusableDatasets));
    selectCell.appendChild(checkbox);

    const methodCell = document.createElement("td");
    methodCell.textContent = item.method_label || methodDisplayLabel(item.method_id);

    const datasetCell = document.createElement("td");
    datasetCell.textContent = reusableDatasetLabel(item);

    const sizeCell = document.createElement("td");
    sizeCell.textContent = item.dataset_size != null ? `${item.dataset_size}` : "-";

    const runCell = document.createElement("td");
    runCell.textContent = item.run_mode || "-";

    const pathCell = document.createElement("td");
    const pathCode = document.createElement("code");
    pathCode.textContent = item.dataset_dir || item.result_dir || item.source_manifest_path || item.id;
    pathCell.appendChild(pathCode);
    if (item.missing_dataset) {
      const warning = document.createElement("div");
      warning.className = "error-text";
      warning.textContent = "dataset folder missing";
      pathCell.appendChild(warning);
    }

    row.append(selectCell, methodCell, datasetCell, sizeCell, runCell, pathCell);
    body.appendChild(row);
  }
}

async function loadReusableDatasets() {
  const payload = await request("/api/datasets/reusable");
  state.reusableDatasets = payload.datasets || [];
  state.reusableDatasetsLoaded = true;
  renderReusableDatasets(state.reusableDatasets);
}

function selectedReusableDatasetIds() {
  return Array.from(document.querySelectorAll(".reusable-dataset-checkbox:checked")).map((node) => node.value);
}

function selectedPlannedDatasetTargetIds() {
  return Array.from(document.querySelectorAll(".planned-dataset-target-checkbox:checked")).map((node) => node.value);
}

function selectedPlannedDatasetTargets() {
  refreshPlannedDatasetTargets();
  const selectedIds = new Set(selectedPlannedDatasetTargetIds());
  return (state.datasetTargets || [])
    .filter((target) => selectedIds.has(target.target_id))
    .map((target) => ({ ...target }));
}

function reusableSplitPolicy() {
  return document.getElementById("reusable-split-policy")?.value || "preserve_archived_splits";
}

function updatePlannedDatasetTargetPanel() {
  const panel = document.getElementById("planned-dataset-target-panel");
  if (!panel) return;
  const fullStrict = document.getElementById("run-mode")?.value === "full_strict_pipeline";
  panel.classList.toggle("hidden", !fullStrict);
  if (fullStrict) {
    refreshPlannedDatasetTargets({ silent: true });
  }
}

function updateReusableDatasetPanel() {
  const panel = document.getElementById("reusable-dataset-panel");
  if (!panel) return;
  const downstreamOnly = document.getElementById("run-mode")?.value === "train_test_metrics_plots_only";
  panel.classList.toggle("hidden", !downstreamOnly);
  if (downstreamOnly) {
    if (state.reusableDatasetsLoaded) {
      renderReusableDatasets(state.reusableDatasets);
    } else {
      loadReusableDatasets().catch((error) => showToast(error.message));
    }
  }
}

function formatBytes(bytes) {
  const value = Number(bytes);
  if (!Number.isFinite(value) || value <= 0) return "-";
  const units = ["B", "KB", "MB", "GB"];
  let scaled = value;
  let index = 0;
  while (scaled >= 1024 && index < units.length - 1) {
    scaled /= 1024;
    index += 1;
  }
  return `${scaled >= 10 || index === 0 ? scaled.toFixed(0) : scaled.toFixed(1)} ${units[index]}`;
}

function datasetTargetLabel(target) {
  const size = target.dataset_size != null ? ` · ${target.dataset_size}` : "";
  return `${target.dataset_label || target.name}${size} · ${target.relative_path}`;
}

function renderDatasetTargets(targets) {
  const body = document.getElementById("dataset-cleanup-list");
  const status = document.getElementById("dataset-cleanup-status");
  if (!body || !status) return;
  body.innerHTML = "";
  status.textContent = targets.length
    ? `${targets.length} generated artifact${targets.length === 1 ? "" : "s"} found`
    : "No generated datasets found";
  for (const target of targets) {
    const row = document.createElement("tr");
    const selectCell = document.createElement("td");
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.className = "dataset-target-checkbox";
    checkbox.value = target.id;
    checkbox.setAttribute("aria-label", `Select ${target.relative_path}`);
    selectCell.appendChild(checkbox);

    const nameCell = document.createElement("td");
    nameCell.textContent = target.dataset_label || target.name || target.id;

    const methodCell = document.createElement("td");
    methodCell.textContent = methodDisplayLabel(target.method);

    const kindCell = document.createElement("td");
    kindCell.textContent = String(target.kind || "");

    const sizeCell = document.createElement("td");
    sizeCell.textContent = [target.dataset_size != null ? `${target.dataset_size} samples` : "", formatBytes(target.bytes)]
      .filter(Boolean)
      .join(" · ") || "-";

    const modifiedCell = document.createElement("td");
    modifiedCell.textContent = target.modified_at || "-";

    const pathCell = document.createElement("td");
    const pathCode = document.createElement("code");
    pathCode.textContent = target.relative_path || target.path;
    pathCell.appendChild(pathCode);
    if (target.warning) {
      const warning = document.createElement("div");
      warning.className = "muted-text";
      warning.textContent = target.warning;
      pathCell.appendChild(warning);
    }

    row.append(selectCell, nameCell, methodCell, kindCell, sizeCell, modifiedCell, pathCell);
    body.appendChild(row);
  }
}

async function loadDatasetTargets() {
  const payload = await request("/api/datasets/targets");
  state.datasetTargets = payload.targets || [];
  renderDatasetTargets(state.datasetTargets);
}

function selectedDatasetTargetIds() {
  return Array.from(document.querySelectorAll(".dataset-target-checkbox:checked")).map((node) => node.value);
}

function confirmDatasetDeletion(targets, title) {
  if (!targets.length) {
    showToast("Selecciona al menos un dataset generado");
    return false;
  }
  const listed = targets.map((target) => `- ${datasetTargetLabel(target)}`).join("\n");
  return window.confirm(`${title}\n\nSe borrara exactamente:\n${listed}`);
}

async function deleteDatasetTargets(targetIds, { all = false } = {}) {
  const targets = all
    ? state.datasetTargets
    : state.datasetTargets.filter((target) => targetIds.includes(target.id));
  const title = all
    ? "Borrar todos los datasets generados, workspaces y resultados archivados?"
    : "Borrar los datasets generados seleccionados?";
  const confirmed = confirmDatasetDeletion(targets, title);
  if (!confirmed) return;
  const payload = await request("/api/datasets/clear", {
    method: "POST",
    body: JSON.stringify(all ? { all: true, dry_run: false } : { target_ids: targetIds, dry_run: false }),
  });
  state.plotData = null;
  await loadResults();
  await loadDatasetTargets();
  const removed = Array.isArray(payload.removed) ? payload.removed.length : 0;
  showToast(`Datasets borrados: ${removed}`);
}

async function clearGeneratedDatasets() {
  if (!state.datasetTargets.length) await loadDatasetTargets();
  await deleteDatasetTargets([], { all: true });
}

async function deleteSelectedGeneratedDatasets() {
  await deleteDatasetTargets(selectedDatasetTargetIds(), { all: false });
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
        loadDatasetTargets().catch((error) => showToast(error.message));
        schedulePlotResize();
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
    Promise.all([loadResults(), loadDatasetTargets()])
      .then(() => showToast("Results refreshed"))
      .catch((error) => showToast(error.message));
  });
  document.getElementById("refresh-dataset-targets")?.addEventListener("click", () => {
    loadDatasetTargets().then(() => showToast("Dataset list refreshed")).catch((error) => showToast(error.message));
  });
  document.getElementById("refresh-reusable-datasets")?.addEventListener("click", () => {
    loadReusableDatasets().then(() => showToast("Reusable dataset list refreshed")).catch((error) => showToast(error.message));
  });
  document.getElementById("material-mode")?.addEventListener("change", () => {
    state.materialValidation = null;
    updateMaterialMode();
    renderMaterialValidation(null);
  });
  document.getElementById("material-preset")?.addEventListener("change", () => {
    state.materialValidation = null;
    renderMaterialValidation(null);
  });
  document.querySelectorAll(".material-bundle-field input").forEach((node) => {
    node.addEventListener("input", () => {
      state.materialValidation = null;
      renderMaterialValidation(null);
    });
  });
  document.getElementById("validate-material")?.addEventListener("click", () => {
    validateMaterialSelection().catch((error) => showToast(error.message));
  });
  document.getElementById("training-hidden-irreps")?.addEventListener("input", () => {
    renderHiddenIrrepsValidation();
  });
  document.getElementById("training-max-ell")?.addEventListener("input", () => {
    renderHiddenIrrepsValidation();
  });
  document.getElementById("add-training-plan-entry")?.addEventListener("click", () => {
    try {
      addTrainingPlanEntry();
    } catch (error) {
      showToast(error.message);
    }
  });
  document.getElementById("clear-training-plan")?.addEventListener("click", () => {
    state.trainingPlan = [];
    renderTrainingPlan();
  });
  document.getElementById("run-mode")?.addEventListener("change", () => {
    state.trainingPlan = [];
    updateReusableDatasetPanel();
    updateTrainingPlanPanel();
  });
  document.getElementById("delete-selected-datasets")?.addEventListener("click", () => {
    deleteSelectedGeneratedDatasets().catch((error) => showToast(error.message));
  });
  document.getElementById("clear-datasets")?.addEventListener("click", () => {
    clearGeneratedDatasets().catch((error) => showToast(error.message));
  });
  document.getElementById("show-plots").addEventListener("change", (event) => {
    state.plotsEnabled = event.target.checked;
    if (state.plotsEnabled) {
      loadPlots().catch((error) => showToast(error.message));
      schedulePlotResize();
    } else {
      renderPlots(state.plotData);
    }
  });
  document.getElementById("plot-cross-selection")?.addEventListener("change", () => {
    if (state.plotsEnabled) {
      renderPlots(state.plotData);
      schedulePlotResize();
    }
  });
  document.getElementById("plot-cross-metric")?.addEventListener("change", () => {
    if (state.plotsEnabled) {
      renderPlots(state.plotData);
      schedulePlotResize();
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
  document.querySelectorAll(".method-execution-checkbox, .method-checkbox").forEach((node) => {
    node.addEventListener("change", () => {
      setMethodSelected(node.value, node.checked, node);
      updateAtomSizesFromFcPlan();
      updateDatasetPreview();
      updateReusableDatasetPanel();
      updateTrainingPlanPanel();
    });
  });
  window.addEventListener("resize", () => schedulePlotResize());
}

async function boot() {
  setupTabs();
  setupEvents();
  const venvActivateInput = document.getElementById("venv-activate-command");
  if (venvActivateInput && !String(venvActivateInput.value || "").trim()) {
    venvActivateInput.value = DEFAULT_VENV_ACTIVATE_COMMAND;
  }
  updateVenvCommandPreview();
  renderHiddenIrrepsValidation();
  await loadPerformancePresets();
  updateMaterialMode();
  try {
    await loadMaterialPresets();
    await validateMaterialSelection({ silent: true });
  } catch (error) {
    renderMaterialValidation({ ok: false, message: error.message });
  }
  await loadFcConfig();
  updateDatasetPreview();
  updateReusableDatasetPanel();
  updateTrainingPlanPanel();
  await pollOnce();
  await loadResults();
  await loadDatasetTargets();
  state.polling = setInterval(pollOnce, POLL_INTERVAL_MS);
}

boot().catch((error) => showToast(error.message));
