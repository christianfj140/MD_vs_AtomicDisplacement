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
  { key: "deeph_comparison", label: "Graph2Mat vs DeepH", resultsDir: "graphene_w90_deeph_fair_benchmark" },
  { key: "graph2mat_deeph_comparison", label: "Graph2Mat sweep + DeepH", resultsDir: "graphene_w90_deeph_fair_benchmark" },
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

const PLOTLY_SCRIPT_URL = "https://cdn.plot.ly/plotly-2.35.2.min.js";
const MATHJAX_SCRIPT_URL = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js";
let plotlyLoadPromise = null;
let mathJaxLoadPromise = null;

function loadExternalScript(id, src) {
  const existing = document.getElementById(id);
  if (existing?.dataset.loaded === "true") return Promise.resolve();
  if (existing?.dataset.loading === "true") {
    return new Promise((resolve, reject) => {
      const timeout = window.setTimeout(() => reject(new Error(`Timeout cargando ${src}`)), 8000);
      existing.addEventListener("load", resolve, { once: true });
      existing.addEventListener("error", () => reject(new Error(`No se pudo cargar ${src}`)), { once: true });
      existing.addEventListener("load", () => window.clearTimeout(timeout), { once: true });
      existing.addEventListener("error", () => window.clearTimeout(timeout), { once: true });
    });
  }
  return new Promise((resolve, reject) => {
    const script = document.createElement("script");
    const timeout = window.setTimeout(() => reject(new Error(`Timeout cargando ${src}`)), 8000);
    script.id = id;
    script.src = src;
    script.async = true;
    script.dataset.loading = "true";
    script.addEventListener("load", () => {
      window.clearTimeout(timeout);
      script.dataset.loaded = "true";
      script.dataset.loading = "false";
      resolve();
    });
    script.addEventListener("error", () => {
      window.clearTimeout(timeout);
      script.dataset.loading = "false";
      reject(new Error(`No se pudo cargar ${src}`));
    });
    document.head.appendChild(script);
  });
}

function ensurePlotlyLoaded() {
  if (window.Plotly) return Promise.resolve();
  if (!plotlyLoadPromise) {
    plotlyLoadPromise = loadExternalScript("plotly-runtime", PLOTLY_SCRIPT_URL)
      .then(() => {
        if (!window.Plotly) throw new Error("Plotly no esta disponible despues de cargar el script.");
      })
      .catch((error) => {
        plotlyLoadPromise = null;
        throw error;
      });
  }
  return plotlyLoadPromise;
}

function configureMathJax() {
  window.MathJax = {
    ...(window.MathJax || {}),
    tex: {
      inlineMath: [["\\(", "\\)"]],
      displayMath: [["\\[", "\\]"]],
      ...((window.MathJax || {}).tex || {}),
    },
    chtml: {
      scale: 0.92,
      ...((window.MathJax || {}).chtml || {}),
    },
    startup: {
      ...((window.MathJax || {}).startup || {}),
      typeset: false,
    },
  };
}

function ensureMathJaxLoaded() {
  if (window.MathJax?.typesetPromise) return Promise.resolve();
  configureMathJax();
  if (!mathJaxLoadPromise) {
    mathJaxLoadPromise = loadExternalScript("mathjax-runtime", MATHJAX_SCRIPT_URL)
      .then(() => {
        if (!window.MathJax?.typesetPromise) throw new Error("MathJax no esta disponible despues de cargar el script.");
      })
      .catch((error) => {
        mathJaxLoadPromise = null;
        throw error;
      });
  }
  return mathJaxLoadPromise;
}

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
const UNKNOWN_MATERIAL_LABEL = "unknown material";
const LOG_POLL_LIMIT = 2000;
const POLL_INTERVAL_MS = 1200;
const POLL_ERROR_TOAST_INTERVAL_MS = 30000;
const G2M_DEEPH_LIVE_PLOT_REFRESH_MS = 30000;
const TERMINAL_MAX_BLOCKS = 1200;

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
  h_mae_eV: {
    label: "Weighted H(k) MAE",
    formula: "\\bar{m}=\\frac{1}{N_s}\\sum_s\\sum_k w_k\\operatorname{MAE}(|H^{pred}(k)-H^{ref}(k)|)",
    description: "MAE complejo del Hamiltoniano evaluado en la malla de k-points, ponderado por pesos normalizados.",
    purpose: "Mide error matricial periodico sin colapsar un run k-point a una aproximacion gamma.",
    direction: "Menor es mejor; comparalo solo con otros resultados k-point-aware equivalentes.",
  },
  h_rmse_eV: {
    label: "Weighted H(k) RMSE",
    formula: "\\bar{m}=\\frac{1}{N_s}\\sum_s\\sum_k w_k\\operatorname{RMSE}(|H^{pred}(k)-H^{ref}(k)|)",
    description: "RMSE complejo de H(k), ponderado en la malla de k-points.",
    purpose: "Penaliza mas los errores grandes en bloques o k-points concretos.",
    direction: "Menor es mejor; no es equivalente a la metrica sparse gamma-only.",
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
  "plot-kpoint-h": {
    title: "K-point matrix H(k)",
    metricKey: "h_mae_eV",
  },
  "plot-kpoint-low-energy": {
    title: "K-point low-energy spectrum",
    metricKey: "low_energy_rmse_eV",
  },
  "plot-kpoint-dos": {
    title: "K-point DOS",
    metricKey: "dos_mae_500_fermi_window",
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

const G2M_DEEPH_PLOT_HELP_BY_ID = {
  "g2m-deeph-plot-metric_scaling_h_mae": {
    title: "Hamiltonian MAE",
    metric: "Mean absolute error of H(k)",
    formula: "\\operatorname{MAE}_H=\\operatorname{mean}_{s,k,ij}|H^{pred}_{ij}(k)-H^{ref}_{ij}(k)|",
    description: "Error absoluto medio del Hamiltoniano periodico ensamblado en k-points.",
    purpose: "DeepH reporta MAE de bloques del Hamiltoniano en escala meV. Este plot da una lectura parecida de error matricial para Graph2Mat y DeepH, pero debe tratarse como diagnostico hasta que la equivalencia raw/global de DeepH este probada.",
    direction: "Menor es mejor. No declara ganador espectral por si solo: en DeepH el objetivo final es que el Hamiltoniano reproduzca bandas, DOS y propiedades derivadas.",
  },
  "g2m-deeph-plot-metric_scaling_h_rmse": {
    title: "Hamiltonian RMSE",
    metric: "Root mean squared error of H(k)",
    formula: "\\operatorname{RMSE}_H=\\sqrt{\\operatorname{mean}_{s,k,ij}|H^{pred}_{ij}(k)-H^{ref}_{ij}(k)|^2}",
    description: "Error cuadratico medio del Hamiltoniano; penaliza mas los elementos o bloques con errores grandes.",
    purpose: "Sirve para detectar outliers de matriz que una MAE puede suavizar. Es util frente a DeepH porque errores grandes en pocos acoplamientos pueden degradar bandas aunque la media parezca aceptable.",
    direction: "Menor es mejor. Comparalo junto a MAE, Frobenius y metricas espectrales.",
  },
  "g2m-deeph-plot-metric_scaling_h_mse": {
    title: "Hamiltonian MSE",
    metric: "Mean squared error of H(k)",
    formula: "\\operatorname{MSE}_H=\\operatorname{mean}_{s,k,ij}|H^{pred}_{ij}(k)-H^{ref}_{ij}(k)|^2",
    description: "Version cuadratica del error de matriz, en eV^2.",
    purpose: "Ayuda a ver si una configuracion falla por pocos errores grandes. Para comparacion con DeepH es una metrica de soporte, no el endpoint fisico principal.",
    direction: "Menor es mejor; valores altos suelen indicar inestabilidad en bloques concretos.",
  },
  "g2m-deeph-plot-metric_scaling_frobenius": {
    title: "Relative Frobenius",
    metric: "Relative matrix norm error",
    formula: "\\frac{\\|H^{pred}-H^{ref}\\|_F}{\\|H^{ref}\\|_F}",
    description: "Norma global del error de Hamiltoniano normalizada por la norma de referencia.",
    purpose: "Resume si el modelo reproduce la escala total del Hamiltoniano. Es un puente util entre MAE de bloques estilo DeepH y errores espectrales derivados.",
    direction: "Menor es mejor. Si baja pero el espectro no mejora, hay desacoplo matriz-espectro.",
  },
  "g2m-deeph-plot-metric_scaling_hermiticity": {
    title: "Hermiticity residual",
    metric: "Predicted Hamiltonian Hermiticity",
    formula: "\\frac{\\|H^{pred}(k)-H^{pred}(k)^\\dagger\\|_F}{\\|H^{pred}(k)\\|_F}",
    description: "Mide si el Hamiltoniano predicho respeta la simetria Hermitiana esperada.",
    purpose: "DeepH y Graph2Mat solo son comparables fisicamente si el H predicho define un problema de autovalores estable. Un residuo alto puede invalidar bandas o DOS aunque el MAE sea bajo.",
    direction: "Menor es mejor; cero es el comportamiento ideal.",
  },
  "g2m-deeph-plot-metric_scaling_spectral_global": {
    title: "Global spectral RMSE",
    metric: "Global eigenvalue RMSE",
    formula: "\\sqrt{\\operatorname{mean}_{s,k,n}(\\varepsilon^{pred}_{n,k}-\\varepsilon^{ref}_{n,k})^2}",
    description: "RMSE de autovalores sobre el espectro comparable completo.",
    purpose: "DeepH se evalua no solo por matriz, sino por bandas y propiedades electronicas derivadas. Esta metrica pregunta directamente si los autovalores salen bien.",
    direction: "Menor es mejor. Es mas cercano a una comparacion DeepH-style de bandas que H-MAE aislado.",
  },
  "g2m-deeph-plot-metric_scaling_spectral_low_energy": {
    title: "Low-energy spectral RMSE",
    metric: "Low-energy eigenvalue RMSE",
    formula: "\\sqrt{\\operatorname{mean}_{s,k,n\\in W_{low}}(\\varepsilon^{pred}_{n,k}-\\varepsilon^{ref}_{n,k})^2}",
    description: "Error de autovalores en la region de baja energia usada como endpoint principal del protocolo.",
    purpose: "Es clave para comparar con DeepH porque las bandas cercanas a las energias relevantes son las que suelen gobernar estructura electronica y propiedades derivadas.",
    direction: "Menor es mejor. Una mejora en H-MAE no basta si esta metrica no mejora.",
  },
  "g2m-deeph-plot-metric_scaling_spectral_fermi": {
    title: "Fermi-window spectral RMSE",
    metric: "Eigenvalue RMSE near Fermi",
    formula: "\\sqrt{\\operatorname{mean}_{|\\varepsilon^{ref}-E_F|\\le w}(\\varepsilon^{pred}-\\varepsilon^{ref})^2}",
    description: "Error espectral dentro de una ventana alrededor del nivel de Fermi.",
    purpose: "DeepH muestra bandas y DOS cerca de Fermi; esta ventana es especialmente sensible para metales, gaps pequenos y propiedades de transporte.",
    direction: "Menor es mejor. Si falta la ventana de Fermi, usa frontier/gap como diagnostico complementario, no sustituto silencioso.",
  },
  "g2m-deeph-plot-metric_scaling_spectral_frontier": {
    title: "Frontier-window RMSE",
    metric: "HOMO/LUMO or band-edge RMSE",
    formula: "\\sqrt{\\operatorname{mean}(e_{HOMO}^2,e_{LUMO}^2)}",
    description: "Error en estados frontier o borde de banda cuando la ventana de Fermi no captura suficientes niveles.",
    purpose: "Mantiene una comparacion local de borde ocupado/no ocupado, util para juzgar si Graph2Mat conserva los rasgos de banda que DeepH suele mostrar visualmente.",
    direction: "Menor es mejor; leelo junto al error de gap.",
  },
  "g2m-deeph-plot-metric_scaling_dos_mae": {
    title: "DOS Fermi-window MAE",
    metric: "Mean absolute DOS error near Fermi",
    formula: "\\frac{1}{N_E}\\sum_E|D^{pred}(E)-D^{ref}(E)|",
    description: "Error absoluto medio entre densidad de estados predicha y de referencia en la ventana energetica evaluada.",
    purpose: "El paper de DeepH reporta DOS de estructuras no vistas y MAE de DOS en puntos alrededor de Fermi para graphene. Esta es una de las comparaciones mas utiles cuando las unidades y ventanas coinciden.",
    direction: "Menor es mejor. Revisa unidades de DOS y ventana energetica antes de comparar con numeros del paper.",
  },
  "g2m-deeph-plot-metric_scaling_dos_wasserstein": {
    title: "DOS Wasserstein distance",
    metric: "Energy displacement between DOS distributions",
    formula: "W_1=\\int |\\operatorname{CDF}_{pred}(E)-\\operatorname{CDF}_{ref}(E)|\\,dE",
    description: "Distancia entre distribuciones DOS; mide cuanto habria que desplazar masa espectral en energia.",
    purpose: "Complementa la DOS MAE: dos curvas pueden tener MAE parecida pero picos desplazados. DeepH busca reproducir propiedades derivadas del Hamiltoniano, y esta metrica ve desplazamientos de DOS.",
    direction: "Menor es mejor; cero significa DOS indistinguible bajo esta metrica.",
  },
  "g2m-deeph-plot-metric_scaling_validation_rerun": {
    title: "Final-seed validation metric",
    metric: "Validation metric recorded during final-seed reruns",
    formula: "m_{val}=\\mathrm{metric}(\\mathcal{D}_{val};\\theta_{seed})",
    description: "Valor de validacion guardado por cada semilla final. Sirve para ver estabilidad del rerun bloqueado sin abrir test.",
    purpose: "Permite comparar si las configuraciones seleccionadas para Graph2Mat y DeepH mantienen comportamiento razonable al repetir semillas finales. No sustituye el test final.",
    direction: "Normalmente menor es mejor. No declarar winner paper-ready con esta metrica: el winner sale de final_test + final_statistics + gate_check.",
  },
  "g2m-deeph-plot-metric_scaling_deeph_live_loss": {
    title: "DeepH live training loss",
    metric: "Train/validation loss streamed from result.txt",
    formula: "\\mathcal{L}_{train},\\ \\mathcal{L}_{val},\\ \\min_t\\mathcal{L}_{val}(t)",
    description: "Lectura diagnostica del entrenamiento DeepH en curso. Se actualiza desde los logs de entrenamiento, no desde el evaluator Hamiltoniano.",
    purpose: "Sirve para verificar que DeepH esta entrenando y no se ha quedado parado. No es una metrica paper-ready ni sustituye MAE/Frobenius/espectro/DOS.",
    direction: "Menor suele ser mejor, pero no declares ganador con esta curva. Las metricas cientificas aparecen cuando termina la config y se ejecuta el evaluator.",
  },
  "g2m-deeph-plot-metric_scaling_gpu_hours": {
    title: "GPU-hours",
    metric: "Total GPU active time",
    formula: "GPUh=\\sum_g\\int u_g(t)\\,dt/3600",
    description: "Coste de GPU observado durante el entrenamiento de cada run.",
    purpose: "DeepH y Graph2Mat deben compararse tambien por coste. Este plot ayuda a ver si una ventaja de precision compensa el gasto de GPU.",
    direction: "Menor es mejor solo si la precision final es comparable. Los fallos/OOM tambien deben contar en el coste.",
  },
  "g2m-deeph-plot-metric_scaling_peak_gpu_memory": {
    title: "Peak GPU memory",
    metric: "Maximum VRAM observed",
    formula: "M_{peak}=\\max_t M_{GPU}(t)",
    description: "Pico de memoria GPU registrado para cada configuracion/semilla.",
    purpose: "Ayuda a saber que modelo cabe mejor en hardware real y que paralelismo es viable para DeepH y Graph2Mat.",
    direction: "Menor es mejor para escalabilidad, pero debe leerse junto con precision y GPU-hours.",
  },
  "g2m-deeph-plot-metric_scaling_peak_rss": {
    title: "Peak process RAM",
    metric: "Maximum process resident memory",
    formula: "RSS_{peak}=\\max_t RSS(t)",
    description: "Pico de RAM de proceso observado durante el run.",
    purpose: "DeepH puede gastar CPU/RAM en preprocesado y dataloading. Este plot separa cuellos de botella de CPU/RAM de coste GPU.",
    direction: "Menor es mejor para robustez operativa; no es una metrica de precision.",
  },
  "g2m-deeph-plot-metric_scaling_cpu_time": {
    title: "CPU time",
    metric: "Total CPU seconds",
    formula: "t_{CPU}=\\sum_c\\int active_c(t)\\,dt",
    description: "Tiempo acumulado de CPU usado por el proceso.",
    purpose: "Sirve para ver si un metodo aparentemente barato en GPU esta trasladando coste a CPU, preprocesado o dataloading.",
    direction: "Menor es mejor a igualdad de precision y cobertura experimental.",
  },
  "g2m-deeph-plot-metric_scaling_throughput": {
    title: "Training throughput",
    metric: "Samples processed per second",
    formula: "q=N_{train}/t_{train}",
    description: "Tasa de entrenamiento estimada por run.",
    purpose: "Permite detectar configuraciones lentas, saturacion de hardware y diferencias practicas entre DeepH y Graph2Mat.",
    direction: "Mayor es mejor si no degrada las metricas espectrales/DOS finales.",
  },
  "g2m-deeph-plot-timing_scaling": {
    title: "Phase time vs dataset size",
    metric: "Wall-clock seconds by phase",
    formula: "t_{phase}=t_{end}-t_{start},\\quad \\mathrm{s/snapshot}=t_{phase}/N",
    description: "Tiempo registrado para entrenamiento, prediccion, preprocesado DeepH y calculo de metricas.",
    purpose: "DeepH se propone como metodo de alta eficiencia para evitar SCF costoso. Este plot separa precision de coste y ayuda a decidir si una configuracion es Pareto-competitiva.",
    direction: "Menor es mejor solo si la precision espectral/DOS es comparable. Los fallos y OOM deben contar en el coste.",
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
  g2mDeephOffset: 0,
  mixingE2eOffset: 0,
  terminalBlocks: [],
  terminalMixingSignature: "",
  g2mDeephRunId: null,
  g2mDeephWasRunning: false,
  g2mDeephValidation: null,
  g2mDeephResults: null,
  g2mDeephPlotPayload: null,
  g2mDeephDerivativePayload: null,
  g2mDeephPlotRuns: [],
  g2mDeephDefaultPlotRunIds: [],
  g2mDeephSelectedPlotRunIds: [],
  g2mDeephDerivativeRunId: null,
  g2mDeephDerivativeRunIds: [],
  g2mDeephDerivativeDatasetSizeAxis: "n_train",
  g2mDeephDerivativeMaeSeriesIds: null,
  datasetMinimumPayload: null,
  datasetMinimumRunRootSelection: null,
  datasetMinimumThresholdPresetKey: "h_mae_relaxed_10",
  datasetMinimumThresholdUserDefined: false,
  datasetMinimumPreviewCache: null,
  datasetMinimumViewRequestId: 0,
  g2mDeephPlotsInFlight: false,
  g2mDeephLastPlotRefreshAt: 0,
  g2mDeephLastCompletedPlotSignature: null,
  g2mDeephDatasets: [],
  g2mDeephDatasetsLoaded: false,
  polling: null,
  pollingInFlight: false,
  pollingFailures: 0,
  lastPollingToastAt: 0,
  plotsEnabled: false,
  plotData: null,
  fcMaxPerDisplacement: null,
  experimentWasRunning: false,
  performancePresetCatalog: null,
  datasetTargets: [],
  reusableDatasets: [],
  reusableDatasetsLoaded: false,
  trainingPlan: [],
  trainingPlanNextId: 1,
  sweepExcludedIndices: new Set(),
  sweepPreviewPage: 1,
  sweepPreviewSignature: null,
  materialPresets: [],
  materialValidation: null,
};

const SWEEP_PREVIEW_PAGE_SIZE = 8;

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
  await pollG2MDeepHLogs();
  await pollMixingE2ELogs();
  await pollMixingTerminalStatus();
  updateVenvCommandPreview();
}

async function pollMixingE2ELogs() {
  const requestedSince = state.mixingE2eOffset;
  const payload = await request(`/api/mixing-e2e/logs?since=${requestedSince}&limit=${LOG_POLL_LIMIT}`);
  if (Number.isFinite(payload.offset) && payload.offset < requestedSince) {
    terminalClearSource("mixing-e2e");
    state.mixingE2eOffset = 0;
    return pollMixingE2ELogs();
  }
  state.mixingE2eOffset = payload.offset;
  if (payload.lines?.length) {
    terminalAppendBlock("mixing-e2e", payload.lines.join(""));
  }
}

async function pollMixingTerminalStatus() {
  const status = await request("/api/mixing/status");
  const stateName = status.state || "idle";
  const signature = JSON.stringify({
    state: stateName,
    action: status.action || "",
    done: status.permutations_done || 0,
    total: status.n_permutations || 0,
    failed: status.n_failed || 0,
    partial: status.n_partial || 0,
    records: (status.live_records || []).length,
    error: status.error || "",
  });
  if (signature === state.terminalMixingSignature) return;
  state.terminalMixingSignature = signature;
  if (stateName === "idle" && !(status.permutations_done || status.n_permutations || status.error)) return;
  terminalAppendBlock(
    "mixing",
    [
      `state=${stateName}`,
      status.action ? `action=${status.action}` : "",
      `done=${status.permutations_done || 0}/${status.n_permutations || 0}`,
      `records=${(status.live_records || []).length}`,
      status.n_failed ? `failed=${status.n_failed}` : "",
      status.n_partial ? `partial=${status.n_partial}` : "",
      status.error ? `error=${status.error}` : "",
    ].filter(Boolean).join(" | "),
  );
}

function inputValue(id) {
  return String(document.getElementById(id)?.value || "").trim();
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function terminalSourceLabel(source) {
  return {
    experiment: "Experiment",
    "g2m-deeph": "G2M vs DeepH",
    mixing: "Mixing sweep",
    "mixing-e2e": "Mixing E2E",
  }[source] || source;
}

function terminalAppendBlock(source, text) {
  const content = String(text || "");
  if (!content.trim()) return;
  state.terminalBlocks.push({
    source,
    label: terminalSourceLabel(source),
    stamp: new Date().toLocaleTimeString(),
    text: content.endsWith("\n") ? content : `${content}\n`,
  });
  if (state.terminalBlocks.length > TERMINAL_MAX_BLOCKS) {
    state.terminalBlocks.splice(0, state.terminalBlocks.length - TERMINAL_MAX_BLOCKS);
  }
  renderTerminalView();
}

function terminalClearSource(source) {
  state.terminalBlocks = state.terminalBlocks.filter((block) => block.source !== source);
  renderTerminalView();
}

function terminalSelectedSource() {
  return document.getElementById("terminal-source")?.value || "all";
}

function terminalFilteredBlocks() {
  const selected = terminalSelectedSource();
  if (selected === "all") return state.terminalBlocks;
  return state.terminalBlocks.filter((block) => block.source === selected);
}

function scrollTerminalToBottom() {
  const output = document.getElementById("terminal-log");
  if (output) output.scrollTop = output.scrollHeight;
}

function renderTerminalView() {
  const output = document.getElementById("terminal-log");
  const status = document.getElementById("terminal-status");
  if (!output) return;
  const blocks = terminalFilteredBlocks();
  const text = blocks
    .map((block) => `[${block.stamp}] ${block.label}\n${block.text}`)
    .join("\n");
  output.textContent = text || "Esperando procesos activos.";
  if (status) {
    const sources = Array.from(new Set(state.terminalBlocks.map((block) => block.source))).length;
    status.textContent = `${blocks.length}/${state.terminalBlocks.length} bloques · ${sources} origen(es)`;
  }
  scrollTerminalToBottom();
}

function clearTerminalView() {
  state.terminalBlocks = [];
  state.terminalMixingSignature = "";
  renderTerminalView();
}

function numericInputValue(id, fallback = null) {
  const raw = inputValue(id);
  if (!raw) return fallback;
  const value = Number(raw);
  return Number.isFinite(value) ? value : fallback;
}

function g2mDeephDatasetMode() {
  return document.getElementById("g2m-deeph-dataset-mode")?.value || "reuse_validated";
}

function updateG2MDeepHDatasetPickerVisibility() {
  const panel = document.getElementById("g2m-deeph-dataset-picker-panel");
  if (!panel) return;
  panel.hidden = ["generate_new", "full_strict_pipeline"].includes(g2mDeephDatasetMode());
}

function renderG2MDeepHDatasetPicker(payload = null) {
  const list = document.getElementById("g2m-deeph-dataset-picker-list");
  const status = document.getElementById("g2m-deeph-dataset-picker-status");
  if (!list || !status) return;
  updateG2MDeepHDatasetPickerVisibility();
  const datasets = payload?.datasets || state.g2mDeephDatasets || [];
  const ready = datasets.filter((item) => item.benchmark_ready);
  status.textContent = datasets.length
    ? `${ready.length}/${datasets.length} benchmark-ready datasets available`
    : "No joint benchmark datasets found.";
  if (!datasets.length) {
    list.classList.add("muted-text");
    list.textContent = "No datasets found under Comparison/datasets.";
    return;
  }
  list.classList.remove("muted-text");
  const currentRoot = inputValue("g2m-deeph-dataset-root");
  list.innerHTML = `
    <div class="cleanup-table-wrap g2m-deeph-table-wrap">
      <table class="cleanup-table g2m-deeph-table">
        <thead>
          <tr>
            <th>Select</th>
            <th>Dataset</th>
            <th>Snapshots</th>
            <th>Ready</th>
            <th>Path</th>
          </tr>
        </thead>
        <tbody>
          ${datasets
            .map((item) => {
              const checked = item.dataset_root === currentRoot || item.relative_path === currentRoot ? "checked" : "";
              const disabled = item.benchmark_ready ? "" : "disabled";
              const missing = item.missing_required_counts && Object.keys(item.missing_required_counts).length
                ? ` · missing ${escapeHtml(JSON.stringify(item.missing_required_counts))}`
                : "";
              const errors = Array.isArray(item.errors) && item.errors.length
                ? `<div class="muted-text">${escapeHtml(item.errors.slice(0, 2).join(" | "))}</div>`
                : "";
              return `
                <tr>
                  <td>
                    <input
                      class="g2m-deeph-dataset-checkbox"
                      type="checkbox"
                      value="${escapeHtml(item.dataset_root)}"
                      ${checked}
                      ${disabled}
                      aria-label="Select ${escapeHtml(item.label || item.relative_path)}"
                    />
                  </td>
                  <td>${escapeHtml(item.label || item.relative_path || item.dataset_root)}</td>
                  <td>${Number(item.valid_snapshots || 0)}/${Number(item.total_snapshots || 0)}</td>
                  <td>${item.benchmark_ready ? "yes" : `no${missing}`}${errors}</td>
                  <td><code>${escapeHtml(item.relative_path || item.dataset_root)}</code></td>
                </tr>
              `;
            })
            .join("")}
        </tbody>
      </table>
    </div>
  `;
}

async function loadG2MDeepHDatasets() {
  const payload = await request("/api/g2m-deeph/datasets");
  state.g2mDeephDatasets = payload.datasets || [];
  state.g2mDeephDatasetsLoaded = true;
  renderG2MDeepHDatasetPicker(payload);
  return payload;
}

function selectG2MDeepHDatasetFromCheckbox(checkbox) {
  if (!checkbox?.checked) return;
  document.querySelectorAll(".g2m-deeph-dataset-checkbox").forEach((node) => {
    if (node !== checkbox) node.checked = false;
  });
  const root = checkbox.value;
  const rootInput = document.getElementById("g2m-deeph-dataset-root");
  if (rootInput) rootInput.value = root;
  const mode = document.getElementById("g2m-deeph-dataset-mode");
  if (mode && ["generate_new", "full_strict_pipeline"].includes(mode.value)) mode.value = "reuse_validated";
  updateG2MDeepHDatasetPickerVisibility();
  renderG2MDeepHDatasetSweepPreview();
  showToast("Dataset seleccionado para Graph2Mat vs DeepH.");
}

function g2mDeephDatasetSweepRecipes() {
  syncDatasetEditorText("g2m_deeph_md");
  const specs = parseMdDatasetTableSpecsFromText(
    document.getElementById("g2m-deeph-md-sweep-table")?.value || "",
  );
  return applyDatasetSeeds("g2m_deeph_md", specs).map((spec, index) => ({
    recipe_id: `md_sweep_${index + 1}_${spec.size}`,
    label: `MD sweep ${index + 1}: ${spec.size} snapshots`,
    ...datasetSeedPatch(spec),
    blocks: spec.blocks,
  }));
}

function g2mDeephDatasetSweepPayload() {
  const enabled = ["generate_new", "full_strict_pipeline"].includes(g2mDeephDatasetMode());
  const recipes = enabled ? g2mDeephDatasetSweepRecipes() : [];
  if (enabled && !recipes.length) {
    throw new Error("Generate/full strict joint dataset: anade al menos un dataset MD.");
  }
  return {
    enabled,
    max_datasets: numericInputValue("g2m-deeph-dataset-sweep-max", 20),
    recipes,
  };
}

function g2mDeephSplitCounts(size) {
  const ratios = {
    train: numericInputValue("g2m-deeph-split-train", 0.8),
    validation: numericInputValue("g2m-deeph-split-validation", 0.1),
    test: numericInputValue("g2m-deeph-split-test", 0.1),
  };
  const raw = Object.fromEntries(Object.entries(ratios).map(([key, value]) => [key, size * value]));
  const counts = Object.fromEntries(Object.entries(raw).map(([key, value]) => [key, Math.floor(value)]));
  let remainder = size - Object.values(counts).reduce((sum, value) => sum + value, 0);
  const order = Object.keys(counts).sort(
    (left, right) => raw[right] - counts[right] - (raw[left] - counts[left]) || ratios[right] - ratios[left],
  );
  for (const key of order.slice(0, remainder)) counts[key] += 1;
  return counts;
}

function renderG2MDeepHDatasetSweepPreview() {
  const container = document.getElementById("g2m-deeph-dataset-sweep-preview");
  if (!container) return;
  if (!["generate_new", "full_strict_pipeline"].includes(g2mDeephDatasetMode())) {
    container.textContent = "Selecciona Generate new joint dataset o Full strict pipeline para generar datasets MD/SIESTA desde este panel.";
    return;
  }
  try {
    const recipes = g2mDeephDatasetSweepRecipes();
    if (!recipes.length) {
      container.textContent = "Dataset sweep enabled, but no MD groups are defined.";
      return;
    }
    const totalSnapshots = recipes.reduce(
      (sum, recipe) => sum + recipe.blocks.reduce((inner, block) => inner + Number(block.n_snapshots || 0), 0),
      0,
    );
    const warning =
      totalSnapshots >= 1000
        ? `<div class="performance-warning">Total planned snapshots: ${totalSnapshots}. SIESTA generation may take a long time.</div>`
        : "";
    container.innerHTML = `
      ${warning}
      <div class="cleanup-table-wrap g2m-deeph-table-wrap">
        <table class="cleanup-table g2m-deeph-table">
          <thead>
            <tr>
              <th>Recipe</th>
              <th>Total</th>
              <th>Blocks</th>
              <th>Temperatures</th>
              <th>Split counts</th>
            </tr>
          </thead>
          <tbody>
            ${recipes
              .map((recipe) => {
                const total = recipe.blocks.reduce((sum, block) => sum + Number(block.n_snapshots || 0), 0);
                const counts = g2mDeephSplitCounts(total);
                return `
                  <tr>
                    <td>${escapeHtml(recipe.recipe_id)}</td>
                    <td>${total}</td>
                    <td>${recipe.blocks.length}</td>
                    <td>${escapeHtml(recipe.blocks.map((block) => `${block.temperature_K} K`).join(", "))}</td>
                    <td>${counts.train}/${counts.validation}/${counts.test}</td>
                  </tr>
                `;
              })
              .join("")}
          </tbody>
        </table>
      </div>
      <p class="field-help">Every planned dataset will be generated with the joint Graph2Mat+DeepH artifact contract.</p>
    `;
  } catch (error) {
    container.textContent = error.message;
  }
}

function parseSweepBoolList(id) {
  return String(inputValue(id) || "")
    .split(/[\n,;]+/)
    .map((item) => item.trim())
    .filter(Boolean)
    .map((item) => {
      const normalized = item.toLowerCase();
      if (["1", "true", "yes", "y", "on", "si", "sí"].includes(normalized)) return true;
      if (["0", "false", "no", "n", "off"].includes(normalized)) return false;
      throw new Error(`${id}: boolean value expected, got ${item}`);
    });
}

function setPayloadList(target, key, values) {
  if (values.length) target[key] = [...new Set(values.map((value) => JSON.stringify(value)))].map((value) => JSON.parse(value));
}

function g2mDeephTrainingSweepPayload(performance = null) {
  const enabled = Boolean(document.getElementById("g2m-deeph-training-sweep-enabled")?.checked);
  if (!enabled) return { enabled: false };
  if (g2mDeephDatasetMode() === "generate_new") {
    throw new Error("Training sweep: primero genera/valida el dataset joint; despues lanza el sweep sobre Reuse existing validated joint dataset, o usa Full strict pipeline.");
  }
  const common = {};
  setPayloadList(common, "seeds", parseSweepNumberList("g2m-deeph-sweep-common-seeds", "G2M/DeepH sweep seeds", { integer: true, min: 0 }));
  setPayloadList(common, "epochs", parseSweepNumberList("g2m-deeph-sweep-common-epochs", "G2M/DeepH sweep epochs", { integer: true, min: 1 }));
  setPayloadList(common, "learning_rate", parseSweepNumberList("g2m-deeph-sweep-common-lr", "G2M/DeepH sweep learning rate", { min: 0.000001 }));
  setPayloadList(common, "batch_size", parseSweepNumberList("g2m-deeph-sweep-common-batch-size", "G2M/DeepH sweep batch size", { integer: true, min: 1 }));

  const graph2mat = { enabled: Boolean(document.getElementById("g2m-deeph-sweep-graph2mat-enabled")?.checked) };
  setPayloadList(graph2mat, "num_interactions", parseSweepNumberList("g2m-deeph-sweep-g2m-interactions", "G2M sweep interactions", { integer: true, min: 1 }));
  setPayloadList(graph2mat, "correlation", parseSweepNumberList("g2m-deeph-sweep-g2m-correlation", "G2M sweep correlation", { integer: true, min: 1 }));
  setPayloadList(graph2mat, "max_ell", parseSweepNumberList("g2m-deeph-sweep-g2m-max-ell", "G2M sweep max ell", { integer: true, min: 0 }));
  setPayloadList(graph2mat, "hidden_irreps_channels", parseSweepNumberList("g2m-deeph-sweep-g2m-hidden-channels", "G2M sweep hidden channels", { integer: true, min: 1 }));
  setPayloadList(graph2mat, "hidden_irreps", parseSweepTextList("g2m-deeph-sweep-g2m-hidden-irreps"));
  setPayloadList(graph2mat, "batch_size", parseSweepNumberList("g2m-deeph-sweep-g2m-batch-size", "G2M sweep batch size", { integer: true, min: 1 }));
  setPayloadList(graph2mat, "loss", parseSweepTextList("g2m-deeph-sweep-g2m-loss"));
  setPayloadList(graph2mat, "loss_kwargs", parseSweepJsonObjectList("g2m-deeph-sweep-g2m-loss-kwargs", "G2M sweep loss kwargs"));
  setPayloadList(graph2mat, "loader_threads", parseSweepNumberList("g2m-deeph-sweep-g2m-loader-threads", "G2M sweep loader threads", { integer: true, min: 1 }));
  if (graph2mat.enabled && !graph2mat.batch_size?.length && !common.batch_size?.length && performance?.batch_size) {
    graph2mat.batch_size = [performance.batch_size];
  }
  if (graph2mat.hidden_irreps?.length && graph2mat.hidden_irreps_channels?.length) {
    throw new Error("Training sweep Graph2Mat: usa hidden irreps o hidden irreps channels, no ambos.");
  }

  const deeph = { enabled: Boolean(document.getElementById("g2m-deeph-sweep-deeph-enabled")?.checked) };
  setPayloadList(deeph, "optimizer", parseSweepTextList("g2m-deeph-sweep-deeph-optimizer"));
  setPayloadList(deeph, "weight_decay", parseSweepNumberList("g2m-deeph-sweep-deeph-weight-decay", "DeepH sweep weight decay", { min: 0 }));
  setPayloadList(deeph, "criterion", parseSweepTextList("g2m-deeph-sweep-deeph-criterion"));
  setPayloadList(deeph, "atom_fea_len", parseSweepNumberList("g2m-deeph-sweep-deeph-atom-fea-len", "DeepH sweep atom_fea_len", { integer: true, min: 1 }));
  setPayloadList(deeph, "edge_fea_len", parseSweepNumberList("g2m-deeph-sweep-deeph-edge-fea-len", "DeepH sweep edge_fea_len", { integer: true, min: 1 }));
  setPayloadList(deeph, "gauss_stop", parseSweepNumberList("g2m-deeph-sweep-deeph-gauss-stop", "DeepH sweep gauss_stop", { min: 0 }));
  setPayloadList(deeph, "num_l", parseSweepNumberList("g2m-deeph-sweep-deeph-num-l", "DeepH sweep num_l", { integer: true, min: 1 }));
  setPayloadList(deeph, "if_lcmp", parseSweepBoolList("g2m-deeph-sweep-deeph-if-lcmp"));
  setPayloadList(deeph, "normalization", parseSweepTextList("g2m-deeph-sweep-deeph-normalization"));
  setPayloadList(deeph, "atom_update_net", parseSweepTextList("g2m-deeph-sweep-deeph-atom-update-net"));
  setPayloadList(deeph, "retain_edge_fea", parseSweepBoolList("g2m-deeph-sweep-deeph-retain-edge-fea"));
  if (!graph2mat.enabled && !deeph.enabled) {
    throw new Error("Training sweep necesita Graph2Mat o DeepH activado.");
  }
  return {
    enabled: true,
    max_runs: numericInputValue("g2m-deeph-training-sweep-max-runs", 128),
    apply_to_datasets: ["all"],
    error_policy: inputValue("g2m-deeph-training-sweep-error-policy") || "continue_on_error",
    common,
    graph2mat,
    deeph,
  };
}

function g2mDeephSweepGridCount(section) {
  return Object.entries(section || {})
    .filter(([key]) => key !== "enabled")
    .reduce((total, [, values]) => total * Math.max(1, Array.isArray(values) ? values.length : 1), 1);
}

function renderG2MDeepHTrainingSweepPreview() {
  const container = document.getElementById("g2m-deeph-training-sweep-preview");
  if (!container) return;
  try {
    const payload = g2mDeephTrainingSweepPayload();
    if (!payload.enabled) {
      container.textContent = "Training sweep disabled.";
      return;
    }
    const fullStrict = g2mDeephDatasetMode() === "full_strict_pipeline";
    const datasetMultiplier = fullStrict ? Math.max(1, g2mDeephDatasetSweepRecipes().length) : 1;
    const commonCount = g2mDeephSweepGridCount(payload.common);
    const graph2matCount = payload.graph2mat.enabled ? datasetMultiplier * commonCount * g2mDeephSweepGridCount(payload.graph2mat) : 0;
    const deephCount = payload.deeph.enabled ? datasetMultiplier * commonCount * g2mDeephSweepGridCount(payload.deeph) : 0;
    const total = graph2matCount + deephCount;
    const warning =
      total > Number(payload.max_runs)
        ? `<div class="performance-warning">Planned runs ${total} exceed max_runs=${payload.max_runs}.</div>`
        : "";
    container.innerHTML = `
      ${warning}
      <div class="cleanup-table-wrap g2m-deeph-table-wrap">
        <table class="cleanup-table g2m-deeph-table">
          <thead><tr><th>Model</th><th>Planned configs</th><th>Notes</th></tr></thead>
          <tbody>
            <tr><td>Graph2Mat</td><td>${graph2matCount}</td><td>Uses existing Graph2Mat sweep semantics</td></tr>
            <tr><td>DeepH</td><td>${deephCount}</td><td>Uses DeepH train.ini overrides only</td></tr>
            <tr><td>Total</td><td>${total}</td><td>${fullStrict ? `${datasetMultiplier} generated dataset(s) in full strict pipeline before training` : "No SIESTA generation in training sweep"}</td></tr>
          </tbody>
        </table>
      </div>
    `;
  } catch (error) {
    container.textContent = error.message;
  }
}

function g2mDeephPayload() {
  const datasetMode = g2mDeephDatasetMode();
  const fullStrict = datasetMode === "full_strict_pipeline";
  const snapshotRoot = inputValue("g2m-deeph-snapshot-root");
  const runId = inputValue("g2m-deeph-run-id");
  const device = inputValue("g2m-deeph-deeph-device") || "cuda:0";
  const performance = performanceSettings();
  const datasetSweep = g2mDeephDatasetSweepPayload();
  const trainingSweep = g2mDeephTrainingSweepPayload(performance);
  if (fullStrict && !trainingSweep.enabled) {
    throw new Error("Full strict pipeline requiere activar Training sweep.");
  }
  const payload = {
    material_preset: inputValue("g2m-deeph-material-preset") || "graphene",
    dataset_mode: fullStrict ? "full_strict_pipeline" : datasetSweep.enabled ? "generate_new" : datasetMode,
    run_mode: fullStrict ? "full_strict_pipeline" : datasetSweep.enabled || datasetMode === "generate_new" ? "generate_datasets_only" : undefined,
    dataset_root: inputValue("g2m-deeph-dataset-root"),
    system_label: inputValue("g2m-deeph-system-label") || "graphene",
    output_root: inputValue("g2m-deeph-output-root"),
    compute_accelerator: performance.compute_accelerator,
    performance,
    snapshot_count: numericInputValue("g2m-deeph-snapshot-count", null),
    split_mode: inputValue("g2m-deeph-split-mode") || "blocked_with_gap",
    dataset_sweep: datasetSweep,
    training_sweep: trainingSweep,
    splits: {
      train: numericInputValue("g2m-deeph-split-train", 0.8),
      validation: numericInputValue("g2m-deeph-split-validation", 0.1),
      test: numericInputValue("g2m-deeph-split-test", 0.1),
    },
    allow_repair: datasetMode === "repair_expensive",
    repair_mode: datasetMode === "repair_expensive",
    require_tshs: true,
    require_tsde: true,
    require_run_output: true,
    graph2mat_overrides: {
      max_epochs: numericInputValue("g2m-deeph-g2m-epochs", 200),
      optim_lr: numericInputValue("g2m-deeph-g2m-lr", 0.005),
      batch_size: numericInputValue("g2m-deeph-g2m-batch-size", 32),
      hidden_irreps: inputValue("g2m-deeph-g2m-hidden-irreps") || "32x0e + 32x1o + 32x2e",
    },
    deeph: {
      repo_path: optionalTextInput("g2m-deeph-deeph-repo"),
      python: optionalTextInput("g2m-deeph-deeph-python"),
      epochs: numericInputValue("g2m-deeph-deeph-epochs", 200),
      batch_size: numericInputValue("g2m-deeph-deeph-batch-size", 4),
      learning_rate: numericInputValue("g2m-deeph-deeph-lr", 0.001),
      device,
      disable_cuda: device.trim().toLowerCase() === "cpu",
    },
  };
  if (snapshotRoot) payload.snapshot_root = snapshotRoot;
  if (runId) payload.run_id = runId;
  return payload;
}

function clearNode(node) {
  if (node) node.textContent = "";
}

function appendKeyValue(container, label, value) {
  const row = document.createElement("div");
  row.className = "summary-row";
  const key = document.createElement("span");
  key.textContent = label;
  const val = document.createElement("strong");
  val.textContent = value == null || value === "" ? "-" : String(value);
  row.append(key, val);
  container.appendChild(row);
}

function g2mDeephValue(value, digits = 5) {
  const number = finiteNumber(value);
  if (number == null) {
    if (value === true) return "yes";
    if (value === false) return "no";
    return value == null || value === "" ? "-" : String(value);
  }
  if (Math.abs(number) >= 1000 || (Math.abs(number) > 0 && Math.abs(number) < 0.001)) {
    return number.toExponential(3);
  }
  return Number(number).toPrecision(digits);
}

function g2mDeephIntegerValue(value) {
  const number = finiteNumber(value);
  return number == null ? g2mDeephValue(value) : String(Math.round(number));
}

function appendG2MDeepHHeading(container, title, subtitle = "") {
  const block = document.createElement("div");
  block.className = "g2m-deeph-section-heading";
  const heading = document.createElement("h4");
  heading.textContent = title;
  block.appendChild(heading);
  if (subtitle) {
    const text = document.createElement("p");
    text.textContent = subtitle;
    block.appendChild(text);
  }
  container.appendChild(block);
}

function appendG2MDeepHTable(container, title, columns, rows, emptyMessage = "No data available.") {
  appendG2MDeepHHeading(container, title);
  if (!rows?.length) {
    const empty = document.createElement("p");
    empty.className = "muted-text";
    empty.textContent = emptyMessage;
    container.appendChild(empty);
    return;
  }
  const wrap = document.createElement("div");
  wrap.className = "cleanup-table-wrap g2m-deeph-table-wrap";
  if (["Metrics vs dataset size", "Timing vs dataset size", "Scientific gates", "Derivative gate report"].includes(title)) {
    wrap.classList.add("g2m-deeph-scroll-table-wrap");
  }
  const table = document.createElement("table");
  table.className = "cleanup-table g2m-deeph-table";
  const thead = document.createElement("thead");
  const headRow = document.createElement("tr");
  for (const column of columns) {
    const th = document.createElement("th");
    th.textContent = column.label;
    headRow.appendChild(th);
  }
  thead.appendChild(headRow);
  const tbody = document.createElement("tbody");
  for (const row of rows) {
    const tr = document.createElement("tr");
    for (const column of columns) {
      const td = document.createElement("td");
      const raw = typeof column.value === "function" ? column.value(row) : row[column.key];
      td.textContent = column.format ? column.format(raw, row) : g2mDeephValue(raw);
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
  }
  table.append(thead, tbody);
  wrap.appendChild(table);
  container.appendChild(wrap);
}

function renderG2MDeepHArtifactSummary(payload) {
  const container = document.getElementById("g2m-deeph-artifact-summary");
  if (!container) return;
  container.classList.remove("muted-text");
  container.classList.add("g2m-deeph-summary-stack");
  container.textContent = "";
  if (!payload) {
    container.classList.add("muted-text");
    container.classList.remove("g2m-deeph-summary-stack");
    container.textContent = "No dataset validation yet.";
    return;
  }
  const summary = payload.artifact_summary || {};
  appendG2MDeepHTable(
    container,
    "Artifact completeness",
    [
      { key: "contract", label: "Contract" },
      { key: "benchmark_ready", label: "Ready" },
      { key: "valid_snapshots", label: "Valid snapshots" },
      { key: "invalid_snapshots", label: "Invalid" },
      { key: "repair_required_snapshots", label: "Repair required" },
      { key: "basis_present", label: "Basis" },
      { key: "pseudopotential_provenance_present", label: "Pseudos provenance" },
    ],
    [
      {
        contract: payload.contract_name,
        benchmark_ready: payload.benchmark_ready,
        valid_snapshots: `${summary.valid_snapshots || 0}/${summary.total_snapshots || 0}`,
        invalid_snapshots: summary.invalid_snapshots || 0,
        repair_required_snapshots: summary.repair_required_snapshots || 0,
        basis_present: summary.basis_present ? "present" : "not required / missing",
        pseudopotential_provenance_present: summary.pseudopotential_provenance_present ? "present" : "not required / missing",
      },
    ],
  );
  const missingRows = Object.entries(summary.missing_required_counts || {}).map(([artifact, count]) => ({
    artifact,
    count,
  }));
  appendG2MDeepHTable(
    container,
    "Missing required artifacts",
    [
      { key: "artifact", label: "Artifact" },
      { key: "count", label: "Snapshots" },
    ],
    missingRows,
    "No required artifacts are missing.",
  );
}

function renderG2MDeepHRankingSummary(container, ranking) {
  if (!container || !ranking) return;
  const recommendation = ranking.recommendation || {};
  const status = recommendation.status || "unknown";
  const robust = String(status).startsWith("robust_");
  const exploratory = String(status).startsWith("exploratory_");
  const banner = document.createElement("div");
  banner.className = "comparison-status-banner";
  banner.classList.toggle("diagnostic", status === "diagnostic_only" || recommendation.scientific_status === "diagnostic_only");
  banner.classList.toggle("invalid", String(status).startsWith("invalid_") || status === "no_robust_winner");
  banner.textContent = robust
    ? `Robust winner: ${methodDisplayLabel(recommendation.winner || recommendation.winning_model || "unknown")} (${recommendation.primary_metric || "primary metric"})`
    : exploratory
    ? `Exploratory best run: ${methodDisplayLabel(recommendation.winner || recommendation.winning_model || "unknown")} (${recommendation.primary_metric || "primary metric"})`
    : `No robust Graph2Mat/DeepH winner: ${status}. ${recommendation.reason || "Review gates and warnings."}`;
  container.appendChild(banner);

  appendG2MDeepHTable(
    container,
    "Ranking recommendation",
    [
      { key: "status", label: "Status" },
      { key: "scientific_status", label: "Scientific status" },
      { key: "winner", label: "Winner" },
      { key: "primary_metric", label: "Primary metric" },
      { key: "adapter_equivalence_status", label: "Adapter equivalence" },
      { key: "split_audit_status", label: "DeepH split audit" },
      { key: "reason", label: "Reason" },
    ],
    [
      {
        status,
        scientific_status: recommendation.scientific_status || "-",
        winner: recommendation.winner ? methodDisplayLabel(recommendation.winner) : "none",
        primary_metric: recommendation.primary_metric || "-",
        adapter_equivalence_status: recommendation.adapter_equivalence_status || "-",
        split_audit_status: recommendation.split_audit_status || "-",
        reason: recommendation.reason || "-",
      },
    ],
  );

  const bestRows = (ranking.best_runs_by_model || []).filter((row) => row.scope === "global" || row.rank === 1);
  appendG2MDeepHTable(
    container,
    "Best Graph2Mat / DeepH runs",
    [
      { key: "model", label: "Model", format: methodDisplayLabel },
      { key: "scope", label: "Scope" },
      { key: "dataset_id", label: "Dataset" },
      { key: "config_id", label: "Config" },
      { key: "metric", label: "Metric" },
      {
        key: "mean",
        label: "Value",
        format: (value) => {
          const numeric = finiteNumber(value);
          return numeric == null ? "-" : numeric.toPrecision(5);
        },
      },
      { key: "seed_stability_status", label: "Seeds" },
      { key: "scientific_status", label: "Scientific" },
      { key: "adapter_equivalence_status", label: "Adapter" },
      { key: "split_audit_status", label: "Split audit" },
    ],
    bestRows,
    "No best-run ranking available.",
  );

  appendG2MDeepHTable(
    container,
    "Pairwise Graph2Mat vs DeepH",
    [
      { key: "dataset_id", label: "Dataset" },
      { key: "metric", label: "Metric" },
      { key: "winner", label: "Winner", format: (value) => (value ? methodDisplayLabel(value) : "none") },
      { key: "status", label: "Status" },
      {
        key: "percent_improvement_challenger_vs_baseline",
        label: "Improvement %",
        format: (value) => {
          const numeric = finiteNumber(value);
          return numeric == null ? "-" : numeric.toFixed(2);
        },
      },
    ],
    ranking.pairwise_graph2mat_vs_deeph || [],
    "No pairwise comparison available.",
  );

  appendG2MDeepHTable(
    container,
    "Accuracy-vs-time Pareto",
    [
      { key: "model", label: "Model", format: methodDisplayLabel },
      { key: "config_id", label: "Config" },
      { key: "metric", label: "Metric" },
      {
        key: "metric_value",
        label: "Metric value",
        format: (value) => {
          const numeric = finiteNumber(value);
          return numeric == null ? "-" : numeric.toPrecision(5);
        },
      },
      {
        key: "total_time_seconds",
        label: "Seconds",
        format: (value) => {
          const numeric = finiteNumber(value);
          return numeric == null ? "-" : numeric.toFixed(2);
        },
      },
      { key: "pareto_status", label: "Status" },
    ],
    ranking.pareto_accuracy_cost || [],
    "No robust Pareto frontier available.",
  );

  const gateRows = [
    ...(recommendation.gates_failed || []).map((gate) => ({ gate, status: "failed" })),
    ...(recommendation.gates_passed || []).map((gate) => ({ gate, status: "passed" })),
  ];
  appendG2MDeepHTable(
    container,
    "Scientific gates",
    [
      { key: "gate", label: "Gate" },
      { key: "status", label: "Status" },
    ],
    gateRows,
    "No gate information available.",
  );
}

function g2mDeephPhaseLabel(phase) {
  return String(phase || "idle").replace(/_/g, " ");
}

function renderG2MDeepHPhaseProgress(status = {}) {
  const container = document.getElementById("g2m-deeph-phase-progress");
  if (!container) return;
  container.textContent = "";
  const phases = status.phases || [];
  const current = status.stage || "idle";
  const activeIndex = phases.indexOf(current);
  for (const phase of phases) {
    const index = phases.indexOf(phase);
    const chip = document.createElement("span");
    chip.className = "phase-chip";
    if (phase === current) chip.classList.add("active");
    if (activeIndex >= 0 && index < activeIndex) chip.classList.add("done");
    chip.textContent = g2mDeephPhaseLabel(phase);
    container.appendChild(chip);
  }
}

function renderG2MDeepHWarnings({ status = null, validation = null, results = null } = {}) {
  const banner = document.getElementById("g2m-deeph-warning-summary");
  if (!banner) return;
  const common = results?.results?.common_metrics || results?.common_metrics || null;
  const warnings = [
    ...((status && status.warnings) || []),
    ...((validation && validation.errors) || []),
    ...((validation && validation.warnings) || []),
    ...((common && common.warnings) || []),
  ].filter(Boolean);
  banner.classList.toggle("hidden", warnings.length === 0);
  if (!warnings.length) {
    banner.textContent = "";
    return;
  }
  banner.textContent = warnings
    .slice(0, 8)
    .map((warning) => {
      if (typeof warning === "string") return warning;
      return [warning.severity, warning.kind || warning.code, warning.message]
        .filter(Boolean)
        .join(": ");
    })
    .join(" | ");
}

function updateG2MDeepHGlobalStatus(status = {}) {
  if (!status.running && status.returncode == null) return;
  const dot = document.getElementById("global-status-dot");
  const text = document.getElementById("global-status-text");
  if (!dot || !text) return;
  dot.classList.toggle("running", Boolean(status.running));
  dot.classList.toggle("error", !status.running && status.returncode != null && status.returncode !== 0);
  if (status.running) {
    text.textContent = `G2M vs DeepH · ${g2mDeephPhaseLabel(status.stage)}`;
  } else if (status.returncode !== 0) {
    text.textContent = "G2M vs DeepH finished with errors";
  }
}

function updateG2MDeepHStatus(status = {}) {
  const dot = document.getElementById("g2m-deeph-status-dot");
  const text = document.getElementById("g2m-deeph-status-text");
  const title = document.getElementById("g2m-deeph-phase-title");
  const root = document.getElementById("g2m-deeph-run-root");
  const sweepStatus = document.getElementById("g2m-deeph-training-sweep-status");
  if (dot) {
    dot.classList.toggle("running", Boolean(status.running));
    dot.classList.toggle("error", status.returncode != null && status.returncode !== 0);
  }
  if (text) text.textContent = statusText(status);
  if (title) title.textContent = g2mDeephPhaseLabel(status.stage || "idle");
  if (root) root.textContent = status.run_root || "No run root yet";
  if (sweepStatus) {
    const sweep = status.training_sweep || {};
    if (sweep.enabled) {
      const activeRuns = Array.isArray(sweep.active_runs) ? sweep.active_runs.length : 0;
      const pieces = [
        `training sweep ${sweep.completed || 0}/${sweep.total || 0}`,
        `failed ${sweep.failed || 0}`,
        `G2M parallel ${sweep.graph2mat_parallelism || 1}`,
        `DeepH parallel ${sweep.deeph_parallelism || 1}`,
      ];
      if (sweep.active_model) pieces.push(`active ${sweep.active_model}`);
      if (activeRuns) pieces.push(`${activeRuns} active run${activeRuns === 1 ? "" : "s"}`);
      sweepStatus.textContent = pieces.join(" | ");
    } else {
      sweepStatus.textContent = "No training sweep active.";
    }
  }
  renderG2MDeepHPhaseProgress(status);
  renderG2MDeepHArtifactSummary(status.dataset_validation || state.g2mDeephValidation);
  renderG2MDeepHWarnings({ status, validation: state.g2mDeephValidation, results: state.g2mDeephResults });
  updateG2MDeepHGlobalStatus(status);
}

function g2mDeephReadableMetricGroups() {
  return [
    { id: "h_mae", title: "Hamiltonian MAE", y_title: "MAE eV", metrics: [{ key: "h_mae_eV_mean", label: "H MAE" }] },
    { id: "h_rmse", title: "Hamiltonian RMSE", y_title: "RMSE eV", metrics: [{ key: "h_rmse_eV_mean", label: "H RMSE" }] },
    { id: "h_mse", title: "Hamiltonian MSE", y_title: "MSE eV^2", metrics: [{ key: "h_mse_eV2_mean", label: "H MSE" }] },
    { id: "r2", title: "Sparse support R2", y_title: "R2", metrics: [{ key: "r2_mean", label: "R2" }] },
    {
      id: "frobenius",
      title: "Relative Frobenius comparison",
      y_title: "Relative Frobenius",
      metrics: [{ key: "relative_frobenius_mean", label: "Relative Frobenius" }],
    },
    {
      id: "hermiticity",
      title: "Predicted Hamiltonian hermiticity",
      y_title: "Hermiticity residual",
      metrics: [{ key: "hermiticity_pred_mean", label: "Hermiticity residual" }],
    },
    {
      id: "spectral_global",
      title: "Global spectral RMSE",
      y_title: "RMSE eV",
      metrics: [{ key: "global_rmse_eV_mean", label: "Global RMSE" }],
    },
    {
      id: "spectral_low_energy",
      title: "Low-energy spectral RMSE",
      y_title: "RMSE eV",
      metrics: [{ key: "low_energy_rmse_eV_mean", label: "Low-energy RMSE" }],
    },
    {
      id: "spectral_fermi",
      title: "Fermi-window spectral RMSE",
      y_title: "RMSE eV",
      metrics: [{ key: "fermi_window_rmse_eV_mean", label: "Fermi RMSE" }],
    },
    {
      id: "spectral_frontier",
      title: "Frontier-window spectral RMSE",
      y_title: "RMSE eV",
      metrics: [{ key: "frontier_window_rmse_eV_mean", label: "Frontier RMSE" }],
    },
    {
      id: "dos_mae",
      title: "DOS Fermi-window MAE",
      y_title: "DOS MAE",
      metrics: [{ key: "dos_mae_500_fermi_window_mean", label: "DOS MAE" }],
    },
    {
      id: "dos_wasserstein",
      title: "DOS Wasserstein distance",
      y_title: "Wasserstein eV",
      metrics: [{ key: "dos_wasserstein_eV_mean", label: "DOS Wasserstein" }],
    },
    {
      id: "validation_rerun",
      title: "Final-seed validation metric",
      y_title: "Validation metric",
      metrics: [{ key: "validation_metric_value", label: "Validation metric" }],
    },
    {
      id: "deeph_live_loss",
      title: "DeepH live training loss",
      y_title: "Loss",
      metrics: [
        { key: "deeph_live_train_loss", label: "Train loss" },
        { key: "deeph_live_val_loss", label: "Val loss" },
        { key: "deeph_live_best_val_loss", label: "Best val loss" },
      ],
    },
    {
      id: "gpu_hours",
      title: "GPU-hours",
      y_title: "GPU-hours",
      metrics: [{ key: "gpu_hours_total", label: "GPU-hours" }],
    },
    {
      id: "peak_gpu_memory",
      title: "Peak GPU memory",
      y_title: "Peak VRAM MB",
      metrics: [{ key: "peak_gpu_memory_mb", label: "Peak VRAM" }],
    },
    {
      id: "peak_rss",
      title: "Peak process RAM",
      y_title: "Peak RSS MB",
      metrics: [{ key: "peak_rss_mb", label: "Peak RSS" }],
    },
    {
      id: "cpu_time",
      title: "CPU time",
      y_title: "CPU seconds",
      metrics: [{ key: "cpu_time_seconds_total", label: "CPU time" }],
    },
    {
      id: "throughput",
      title: "Training throughput",
      y_title: "Samples/s",
      metrics: [{ key: "samples_per_second", label: "Samples/s" }],
    },
  ];
}

function renderG2MDeepHMetricSummary(payload) {
  const container = document.getElementById("g2m-deeph-metric-summary");
  if (!container) return;
  container.classList.remove("muted-text");
  container.classList.add("g2m-deeph-summary-stack");
  container.textContent = "";
  const plotPayload = payload?.plot_payload || state.g2mDeephPlotPayload || null;
  const common = payload?.results?.common_metrics || plotPayload?.common_metrics || null;
  const ranking = payload?.results?.ranking || plotPayload?.ranking || null;
  const archivedMetricRows = plotPayload?.metric_scaling_rows || [];
  const archivedTimingRows = plotPayload?.timing_scaling_rows || [];
  if (!common && !ranking && !archivedMetricRows.length && !archivedTimingRows.length) {
    container.classList.add("muted-text");
    container.classList.remove("g2m-deeph-summary-stack");
    container.textContent = "No common metrics or ranking yet.";
    return;
  }
  if (ranking) {
    renderG2MDeepHRankingSummary(container, ranking);
    if (!common && !archivedMetricRows.length && !archivedTimingRows.length) return;
  }
  if (!common) {
    const statusBanner = document.createElement("div");
    statusBanner.className = "comparison-status-banner diagnostic";
    const completedRows =
      Number(plotPayload?.live_metric_rows || 0) ||
      (plotPayload?.metric_scaling_rows || []).filter((row) => row.source === "live_training_sweep_metrics").length;
    statusBanner.textContent = completedRows
      ? `Completed Graph2Mat/DeepH metrics: ${completedRows} metric row(s), ${plotPayload?.timing_scaling_rows?.length || 0} timing row(s).`
      : `Archived Graph2Mat/DeepH plots: ${plotPayload?.archived_runs || 0} metric run(s), ${plotPayload?.archived_timing_runs || 0} timing source(s).`;
    container.appendChild(statusBanner);
  }
  if (archivedMetricRows.length) {
    appendG2MDeepHTable(
      container,
      "Metrics vs dataset size",
      [
        { key: "run_id", label: "Run" },
        { key: "dataset_size", label: "Snapshots", format: g2mDeephIntegerValue },
        { key: "method", label: "Method" },
        { key: "epoch_label", label: "Epochs", format: (value, row) => g2mDeephEpochLabel(row) },
        { key: "config_id", label: "Config" },
        { key: "metric_key", label: "Metric" },
        {
          key: "metric_value",
          label: "Value",
          format: (value) => {
            const number = finiteNumber(value);
            return number == null ? "-" : number.toPrecision(5);
          },
        },
        { key: "scientific_status", label: "Status" },
      ],
      archivedMetricRows,
      "No archived metric-vs-size rows available.",
    );
  }
  if (!common) {
    if (archivedTimingRows.length) {
      appendG2MDeepHTable(
        container,
        "Timing vs dataset size",
        [
          { key: "dataset_id", label: "Dataset" },
          { key: "dataset_size", label: "Snapshots", format: g2mDeephIntegerValue },
          { key: "label", label: "Phase" },
          { key: "model", label: "Model" },
          { key: "epoch_label", label: "Epochs", format: (value, row) => g2mDeephEpochLabel(row) },
          { key: "config_id", label: "Config" },
          {
            key: "elapsed_seconds",
            label: "Seconds",
            format: (value) => {
              const number = finiteNumber(value);
              return number == null ? "-" : number.toFixed(2);
            },
          },
          {
            key: "seconds_per_snapshot",
            label: "s/snapshot",
            format: (value) => {
              const number = finiteNumber(value);
              return number == null ? "-" : number.toFixed(4);
            },
          },
        ],
        archivedTimingRows,
        "No timing-vs-size rows available.",
      );
    }
    return;
  }
  const recommendation = common.recommendation || {};
  const scientificStatus = common.status || plotPayload?.scientific_status || recommendation.status || "unknown";
  const statusBanner = document.createElement("div");
  statusBanner.className = "comparison-status-banner";
  statusBanner.classList.toggle("diagnostic", scientificStatus === "diagnostic_only");
  statusBanner.classList.toggle("invalid", String(scientificStatus).startsWith("invalid_"));
  const winner = recommendation.robust_recommendation ? recommendation.winner : null;
  statusBanner.textContent = winner
    ? `Robust candidate: ${winner} (${recommendation.primary_metric || "h_mae_eV_mean"})`
    : `No robust winner: ${scientificStatus}. ${recommendation.reason || "Review comparability warnings."}`;
  container.appendChild(statusBanner);

  appendG2MDeepHTable(
    container,
    "Final recommendation",
    [
      { key: "scientific_status", label: "Scientific status" },
      { key: "winner", label: "Winner" },
      { key: "robust", label: "Robust" },
      { key: "primary_metric", label: "Primary metric" },
      { key: "reason", label: "Reason" },
    ],
    [
      {
        scientific_status: scientificStatus,
        winner: winner || "none",
        robust: Boolean(recommendation.robust_recommendation),
        primary_metric: recommendation.primary_metric || "h_mae_eV_mean",
        reason: recommendation.reason || "-",
      },
    ],
  );

  const metricGroups = plotPayload?.metric_groups || g2mDeephReadableMetricGroups();
  const summaryRows = common.summary_rows || [];
  for (const group of metricGroups) {
    const columns = [
      { key: "method", label: "Method" },
      ...(group.metrics || []).map((metric) => ({
        key: metric.key,
        label: metric.unit ? `${metric.label} (${metric.unit})` : metric.label,
      })),
    ];
    appendG2MDeepHTable(
      container,
      group.title,
      columns,
      summaryRows,
      "No metric rows available.",
    );
  }

  appendG2MDeepHTable(
    container,
    "Phase timing",
    [
      { key: "label", label: "Phase" },
      {
        key: "elapsed_seconds",
        label: "Seconds",
        format: (value) => {
          const number = finiteNumber(value);
          return number == null ? "-" : number.toFixed(2);
        },
      },
      { key: "status", label: "Status" },
      { key: "source", label: "Source" },
    ],
    plotPayload?.timing_rows || common.timing_rows || [],
    "No phase timing rows available.",
  );

  appendG2MDeepHTable(
    container,
    "Timing vs dataset size",
    [
      { key: "dataset_id", label: "Dataset" },
      { key: "dataset_size", label: "Snapshots", format: g2mDeephIntegerValue },
      { key: "label", label: "Phase" },
      { key: "model", label: "Model" },
      { key: "config_id", label: "Config" },
      {
        key: "elapsed_seconds",
        label: "Seconds",
        format: (value) => {
          const number = finiteNumber(value);
          return number == null ? "-" : number.toFixed(2);
        },
      },
      {
        key: "seconds_per_snapshot",
        label: "s/snapshot",
        format: (value) => {
          const number = finiteNumber(value);
          return number == null ? "-" : number.toFixed(4);
        },
      },
    ],
    plotPayload?.timing_scaling_rows || common.timing_scaling_rows || [],
    "No timing-vs-size rows available.",
  );
}

async function validateG2MDeepHDataset() {
  const payload = await request("/api/g2m-deeph/validate-dataset", {
    method: "POST",
    body: JSON.stringify(g2mDeephPayload()),
  });
  state.g2mDeephValidation = payload;
  renderG2MDeepHArtifactSummary(payload);
  renderG2MDeepHWarnings({ validation: payload });
  showToast(payload.benchmark_ready ? "Joint dataset validado." : "Dataset no esta listo para benchmark.");
  return payload;
}

function formatG2MDeepHValidationError(validation, datasetMode) {
  const summary = validation?.artifact_summary || {};
  const errors = Array.isArray(validation?.errors) ? validation.errors.filter(Boolean) : [];
  const missing = summary.missing_required_counts || {};
  const missingText = Object.entries(missing)
    .map(([key, value]) => `${key}=${value}`)
    .join(", ");
  const reasons = [];
  if (datasetMode === "generate_new") {
    reasons.push("pulsa Run para generar un dataset joint nuevo con SIESTA");
  }
  if (!Number(summary.total_snapshots || 0)) {
    reasons.push(`no hay snapshots en ${validation?.snapshot_root || validation?.dataset_root || "dataset_root"}`);
  }
  if (Number(summary.invalid_snapshots || 0)) {
    reasons.push(`${summary.invalid_snapshots} snapshots invalidos`);
  }
  if (missingText) {
    reasons.push(`faltan artefactos requeridos: ${missingText}`);
  }
  reasons.push(...errors);
  return `Dataset no listo para Graph2Mat vs DeepH: ${reasons.join("; ") || "validacion incompleta"}.`;
}

async function runG2MDeepHBenchmark() {
  state.g2mDeephOffset = 0;
  state.g2mDeephRunId = null;
  state.g2mDeephResults = null;
  clearNode(document.getElementById("g2m-deeph-log"));
  renderG2MDeepHMetricSummary(null);
  const datasetMode = g2mDeephDatasetMode();
  if (!["generate_new", "full_strict_pipeline"].includes(datasetMode)) {
    const validation = await validateG2MDeepHDataset();
    if (!validation.benchmark_ready && datasetMode !== "repair_expensive") {
      throw new Error(formatG2MDeepHValidationError(validation, datasetMode));
    }
  } else {
    renderG2MDeepHDatasetSweepPreview();
    if (datasetMode === "full_strict_pipeline") renderG2MDeepHTrainingSweepPreview();
  }
  const payload = await request("/api/g2m-deeph/run", {
    method: "POST",
    body: JSON.stringify(g2mDeephPayload()),
  });
  updateG2MDeepHStatus(payload);
  showToast("Graph2Mat vs DeepH benchmark started.");
}

async function stopG2MDeepHBenchmark() {
  const payload = await request("/api/g2m-deeph/stop", { method: "POST", body: "{}" });
  updateG2MDeepHStatus(payload);
  showToast("Graph2Mat vs DeepH stop requested.");
}

async function loadG2MDeepHResults() {
  const payload = await request("/api/g2m-deeph/results");
  state.g2mDeephResults = payload;
  if (payload.plot_payload) state.g2mDeephPlotPayload = payload.plot_payload;
  updateG2MDeepHStatus(payload.status || {});
  renderG2MDeepHMetricSummary(payload);
  renderG2MDeepHWarnings({
    status: payload.status,
    validation: state.g2mDeephValidation,
    results: payload.results,
  });
  loadG2MDeepHDerivativeMetrics().catch((error) => {
    renderG2MDeepHDerivativePayload({ available: false, not_computed: true, message: error.message });
  });
  return payload;
}

function renderG2MDeepHGroupedBarPlot(card, plot) {
  const rows = plot.rows || [];
  const methods = rows.map((row) => methodDisplayLabel(row.method || "unknown"));
  const metricKeys = (plot.metrics || []).map((metric) => metric.key);
  const traces = (plot.metrics || []).map((metric) => ({
    type: "bar",
    x: methods,
    y: rows.map((row) => metricDisplayValue(metric.key, row[metric.key])),
    name: metricDisplayLabel(metric.key, metric.unit ? `${metric.label} (${metric.unit})` : metric.label),
    hovertemplate: `%{x}<br>%{fullData.name}: %{y:.5g}<extra></extra>`,
  }));
  const finiteValues = traces.flatMap((trace) => (trace.y || []).filter((value) => value != null));
  const missing = plot.missing_metrics || [];
  const annotations = [];
  if (!finiteValues.length) {
    annotations.push(emptyPlotAnnotation("No finite values for this metric group."));
  } else if (missing.length) {
    annotations.push(topPlotAnnotation(`${missing.length} missing metric values`, 1.12, "#9f5b00"));
  }
  renderPlot(
    card,
    traces,
    plotLayout(plot.title || "Graph2Mat vs DeepH", metricDisplayAxisTitle(plot.y_title || "", metricKeys), {
      barmode: "group",
      annotations,
      xaxis: { title: "Method", gridcolor: "#edf1f4", zeroline: false },
      yaxis: { title: metricDisplayAxisTitle(plot.y_title || "", metricKeys), gridcolor: "#edf1f4", zeroline: false },
    }),
  );
}

function g2mDeephIsDeepH(method) {
  return String(method || "").toLowerCase().includes("deeph");
}

function g2mDeephMarkerSymbol(method) {
  return g2mDeephIsDeepH(method) ? "triangle-up" : "circle";
}

function g2mDeephEpochLabel(row = {}) {
  if (row.epoch_label) return String(row.epoch_label);
  if (row.epochs != null && row.epochs !== "") return `${row.epochs} epochs`;
  return "epochs unknown";
}

function valuesAreNonNegative(values) {
  const finiteValues = values.filter((value) => typeof value === "number" && Number.isFinite(value));
  return finiteValues.length > 0 && finiteValues.every((value) => value >= 0);
}

function paddedLinearRange(values, options = {}) {
  const finiteValues = values.filter((value) => typeof value === "number" && Number.isFinite(value));
  if (!finiteValues.length) return null;
  const minValue = Math.min(...finiteValues);
  const maxValue = Math.max(...finiteValues);
  const span = maxValue - minValue;
  const padding = span > 0 ? span * (options.padFraction ?? 0.06) : Math.max(Math.abs(maxValue) * 0.08, 1);
  const minRange = options.forceZeroMin ? 0 : minValue - padding;
  const maxRange = maxValue + padding;
  if (!Number.isFinite(minRange) || !Number.isFinite(maxRange) || minRange === maxRange) return null;
  return [minRange, maxRange];
}

function paddedLogRange(values, options = {}) {
  const finiteValues = values.filter((value) => typeof value === "number" && Number.isFinite(value) && value > 0);
  if (!finiteValues.length) return null;
  const minValue = Math.min(...finiteValues);
  const maxValue = Math.max(...finiteValues);
  if (!Number.isFinite(minValue) || !Number.isFinite(maxValue) || minValue <= 0 || maxValue <= 0) return null;
  const logMin = Math.log10(minValue);
  const logMax = Math.log10(maxValue);
  const span = logMax - logMin;
  const padding = span > 0 ? span * (options.padFraction ?? 0.08) : 0.25;
  return [logMin - padding, logMax + padding];
}

function renderG2MDeepHTimingScalingPlot(card, plot) {
  const rows = (plot.rows || [])
    .map((row) => ({
      ...row,
      dataset_size: finiteNumber(row.dataset_size),
      elapsed_seconds: finiteNumber(row.elapsed_seconds),
    }))
    .filter((row) => row.dataset_size != null && row.elapsed_seconds != null)
    .sort(
      (a, b) =>
        String(a.phase || "").localeCompare(String(b.phase || "")) ||
        String(a.model || "").localeCompare(String(b.model || "")) ||
        String(g2mDeephEpochLabel(a)).localeCompare(String(g2mDeephEpochLabel(b))) ||
        a.dataset_size - b.dataset_size ||
        String(a.config_id || "").localeCompare(String(b.config_id || "")),
    );
  const groups = new Map();
  for (const row of rows) {
    const key = `${row.phase || "unknown"}|${row.model || "all"}|${g2mDeephEpochLabel(row)}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(row);
  }
  const traces = [];
  const fitYMin = valuesAreNonNegative(rows.map((row) => row.elapsed_seconds)) ? 0 : null;
  Array.from(groups.entries()).forEach(([key, group], index) => {
    const [phase, model, epochLabel] = key.split("|");
    const label = group[0]?.label || phase;
    const name = `${label}${model && model !== "all" ? ` · ${methodDisplayLabel(model)}` : ""} · ${epochLabel}`;
    const color = plotColor(index);
    const points = group.map((row) => ({
      x: row.dataset_size,
      y: row.elapsed_seconds,
    }));
    addFitTraces(traces, points, name, color, { legendgroup: key, fitYMin });
    traces.push({
      type: "scatter",
      mode: "markers",
      marker: { symbol: g2mDeephMarkerSymbol(model), size: g2mDeephIsDeepH(model) ? 10 : 8, color },
      x: group.map((row) => row.dataset_size),
      y: group.map((row) => row.elapsed_seconds),
      text: group.map((row) => `${row.dataset_id || "-"} · ${row.config_id || "-"} · ${g2mDeephEpochLabel(row)}`),
      name,
      legendgroup: key,
      hovertemplate:
        "Dataset size: %{x}<br>Seconds: %{y:.3f}<br>%{text}<extra>%{fullData.name}</extra>",
    });
  });
  const annotations = traces.length ? [] : [emptyPlotAnnotation("No timing-vs-size rows with finite values.")];
  let layout = plotLayout(plot.title || "Phase time vs dataset size", plot.y_title || "Seconds", {
    annotations,
    xaxis: {
      title: plot.x_title || "Dataset size (snapshots)",
      gridcolor: "#edf1f4",
      zeroline: false,
      range: paddedLinearRange(rows.map((row) => row.dataset_size)),
    },
    yaxis: {
      title: plot.y_title || "Seconds",
      gridcolor: "#edf1f4",
      zeroline: false,
      ...(fitYMin === 0 ? { range: paddedLinearRange(rows.map((row) => row.elapsed_seconds), { forceZeroMin: true }) } : {}),
    },
    legend: { orientation: "h", y: -0.24 },
  });
  layout = withFitSelector(layout, traces);
  renderPlot(
    card,
    traces,
    layout,
  );
}

function renderG2MDeepHMetricScalingPlot(card, plot) {
  const metricKeys = (plot.metrics || []).map((metric) => metric.key);
  const yTitle = metricDisplayAxisTitle(plot.y_title || "Metric value", metricKeys);
  const yUnit = metricDisplayUnitForKeys(metricKeys);
  const rows = (plot.rows || [])
    .map((row) => ({
      ...row,
      dataset_size: finiteNumber(row.dataset_size),
      metric_value: metricDisplayValue(row.metric_key, row.metric_value),
    }))
    .filter((row) => row.dataset_size != null && row.metric_value != null)
    .sort(
      (a, b) =>
        String(a.metric_key || "").localeCompare(String(b.metric_key || "")) ||
        String(a.method || "").localeCompare(String(b.method || "")) ||
        String(g2mDeephEpochLabel(a)).localeCompare(String(g2mDeephEpochLabel(b))) ||
        a.dataset_size - b.dataset_size ||
        String(a.run_id || "").localeCompare(String(b.run_id || "")),
    );
  const metricLabels = Object.fromEntries((plot.metrics || []).map((metric) => [metric.key, metric.label || metric.key]));
  const groups = new Map();
  for (const row of rows) {
    const key = `${row.metric_key || "metric"}|${row.method || "unknown"}|${g2mDeephEpochLabel(row)}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(row);
  }
  const traces = [];
  const fitYMin = valuesAreNonNegative(rows.map((row) => row.metric_value)) ? 0 : null;
  Array.from(groups.entries()).forEach(([key, group], index) => {
    const [metricKey, method, epochLabel] = key.split("|");
    const name = `${metricLabels[metricKey] || metricKey} · ${methodDisplayLabel(method)} · ${epochLabel}`;
    const color = plotColor(index);
    const points = group.map((row) => ({
      x: row.dataset_size,
      y: row.metric_value,
    }));
    addFitTraces(traces, points, name, color, { legendgroup: key, fitYMin });
    traces.push({
      type: "scatter",
      mode: "markers",
      marker: { symbol: g2mDeephMarkerSymbol(method), size: g2mDeephIsDeepH(method) ? 10 : 8, color },
      x: group.map((row) => row.dataset_size),
      y: group.map((row) => row.metric_value),
      text: group.map((row) => `${row.run_id || "-"} · ${row.config_id || "-"} · ${g2mDeephEpochLabel(row)} · ${row.scientific_status || "-"}`),
      name,
      legendgroup: key,
      hovertemplate:
        `Dataset size: %{x}<br>Value${yUnit ? ` (${yUnit})` : ""}: %{y:.5g}<br>%{text}<extra>%{fullData.name}</extra>`,
    });
  });
  const annotations = traces.length ? [] : [emptyPlotAnnotation("No archived metric-vs-size rows with finite values.")];
  let layout = plotLayout(plot.title || "Metrics vs dataset size", yTitle, {
      annotations,
      xaxis: {
        title: plot.x_title || "Dataset size (snapshots)",
        gridcolor: "#edf1f4",
        zeroline: false,
        range: paddedLinearRange(rows.map((row) => row.dataset_size)),
      },
      yaxis: {
        title: yTitle,
        gridcolor: "#edf1f4",
        zeroline: false,
        ...(fitYMin === 0 ? { range: paddedLinearRange(rows.map((row) => row.metric_value), { forceZeroMin: true }) } : {}),
      },
      legend: { orientation: "h", y: -0.24 },
    });
  const primaryMetricKey = metricKeys[0] || rows[0]?.metric_key || "";
  const references = scaleReferencesForMetric(
    DEEPH_PAPER_REFERENCE_LINES[card.id] || DEEPH_PAPER_REFERENCE_LINES[plot.id],
    primaryMetricKey,
  );
  layout = withHorizontalReferenceLines(
    layout,
    traces,
    references,
    references
      ? "DeepH paper reference lines are diagnostic guides; match basis, units, support and raw/global equivalence before claims."
      : "",
  );
  layout = withFitSelector(layout, traces);
  renderPlot(card, traces, layout);
}

function renderG2MDeepHPlotsPayload(payload) {
  const container = document.getElementById("g2m-deeph-plots");
  if (!container) return;
  container.textContent = "";
  if (!payload.available || !(payload.plots || []).length) {
    const placeholder = document.createElement("div");
    placeholder.className = "plot-card full placeholder-card";
    placeholder.textContent = payload.message || "No benchmark plots available yet.";
    container.appendChild(placeholder);
    return;
  }
  for (const plot of payload.plots) {
    const card = document.createElement("div");
    card.id = `g2m-deeph-plot-${plot.id || container.children.length}`;
    card.className = "plot-card wide";
    container.appendChild(card);
    if (window.Plotly && plot.kind === "grouped_bar") {
      renderG2MDeepHGroupedBarPlot(card, plot);
    } else if (window.Plotly && plot.kind === "timing_scaling") {
      renderG2MDeepHTimingScalingPlot(card, plot);
    } else if (window.Plotly && plot.kind === "metric_scaling") {
      renderG2MDeepHMetricScalingPlot(card, plot);
    } else {
      card.textContent = plot.title || "Graph2Mat vs DeepH plot";
    }
  }
  schedulePlotResize();
}

function preferredG2MDeepHDerivativeRunId() {
  const runs = state.g2mDeephPlotRuns || [];
  if (!runs.length) return state.g2mDeephRunId || null;
  if (state.g2mDeephDerivativeRunId && runs.some((run) => run.run_id === state.g2mDeephDerivativeRunId)) {
    return state.g2mDeephDerivativeRunId;
  }
  if (state.g2mDeephRunId && runs.some((run) => run.run_id === state.g2mDeephRunId)) return state.g2mDeephRunId;
  return runs[0]?.run_id || null;
}

const G2M_DEEPH_DERIVATIVE_TITLE = "Hamiltonian derivative diagnostics";
const G2M_DEEPH_DERIVATIVE_REFERENCE = "Reference: finite differences of SIESTA Hamiltonians";
const G2M_DEEPH_DERIVATIVE_FORCE_CONSTANTS = "SIESTA force constants are not treated as dH/dR";
const G2M_DEEPH_DERIVATIVE_DEFAULT_STATUS = "Default status: diagnostic-only unless all scientific gates pass";
const G2M_DEEPH_DERIVATIVE_OPTIONAL_POSTPROCESSING = "Derivative diagnostics are optional post-processing outputs. If not computed, the benchmark remains valid for H-vs-H metrics.";
const G2M_DEEPH_DERIVATIVE_INTERNAL_ONLY = "Technical internal diagnostic only. No winner claim comes from derivative metrics.";
const G2M_DEEPH_DERIVATIVE_DATASET_SIZE_NOTE = "Dataset-size derivative plots show aggregates over derivative metric rows/stencils; x_dataset_size is usually N_train when available, otherwise N_total.";
const G2M_DEEPH_EXPECTED_DERIVATIVE_PLOTS = [
  ["relative_frobenius_union_robust_by_model", "Robust relative Frobenius by model"],
  ["relative_l1_union_robust_by_model", "Robust relative L1 by model"],
  ["robust_primary_metrics_by_model", "Robust primary metrics by model"],
  ["derivative_correlation_by_model", "Derivative correlation by model"],
  ["derivative_residual_summary_by_model", "Derivative residual summary by model"],
  ["derivative_residual_tail_by_model", "Derivative residual tail by model"],
  ["derivative_error_by_abs_ref_quantile", "Derivative error by |dH_ref| quantile"],
  ["derivative_relative_l1_by_abs_ref_quantile", "Derivative relative L1 by |dH_ref| quantile"],
  ["robust_error_by_displaced_atom", "Robust derivative error by displaced atom"],
  ["robust_error_by_axis", "Robust derivative error by axis"],
  ["robust_error_by_atom_axis", "Robust derivative error by atom and axis"],
  ["onsite_offsite_derivative_error", "Onsite/offsite derivative error by model"],
];
const G2M_DEEPH_DERIVATIVE_MARKER_ONLY_PLOTS = new Set(["error_vs_delta", "graph2mat_vs_deeph_paired_comparison"]);
const G2M_DEEPH_DERIVATIVE_MAE_DATASET_SIZE_PLOT_ID = "dh_mae_vs_dataset_size";
const G2M_DEEPH_DERIVATIVE_MAE_EV_TO_MEV = 1000;
const G2M_DEEPH_DERIVATIVE_DATASET_SIZE_TITLES = {
  dh_mae_vs_dataset_size: "Mean dH MAE vs dataset size",
  dh_mae_vs_dataset_size_by_delta: "Mean dH MAE vs dataset size by delta",
  dh_rmse_vs_dataset_size: "Mean dH RMSE vs dataset size",
  dh_rmse_vs_dataset_size_by_delta: "Mean dH RMSE vs dataset size by delta",
  relative_frobenius_vs_dataset_size: "Mean relative Frobenius vs dataset size",
  relative_frobenius_vs_dataset_size_by_delta: "Mean relative Frobenius vs dataset size by delta",
  signal_to_noise_vs_dataset_size: "Mean dH signal-to-noise ratio vs dataset size",
};
const G2M_DEEPH_BASE_DERIVATIVE_DATASET_SIZE_PLOTS = new Set([
  "dh_mae_vs_dataset_size",
  "dh_rmse_vs_dataset_size",
  "relative_frobenius_vs_dataset_size",
  "signal_to_noise_vs_dataset_size",
  "support_f1_vs_dataset_size",
  "support_error_rates_vs_dataset_size",
]);
const G2M_DEEPH_OPTIONAL_DERIVATIVE_DATASET_SIZE_PLOTS = [
  "robust_relative_frobenius_vs_dataset_size",
  "robust_relative_l1_vs_dataset_size",
  "derivative_correlation_vs_dataset_size",
  "derivative_residual_summary_vs_dataset_size",
  "derivative_residual_tail_vs_dataset_size",
  "dh_mae_vs_dataset_size_by_delta",
  "dh_rmse_vs_dataset_size_by_delta",
  "relative_frobenius_vs_dataset_size_by_delta",
  "dh_mae_vs_dataset_size_by_axis",
  "robust_frobenius_vs_dataset_size_by_axis",
  "dh_mae_vs_dataset_size_by_displaced_atom",
  "derivative_hermiticity_vs_dataset_size",
  "onsite_offsite_derivative_error_vs_dataset_size",
];
const G2M_DEEPH_DERIVATIVE_DATASET_SIZE_PARTIAL_MESSAGE = "Only base dataset-size derivative plots are present. Regenerate derivative_plot_payload.json from current derivative metric CSV/JSON artifacts to show robust, residual, delta, axis/atom, Hermiticity, and onsite/offsite dataset-size plots. If the payload is current, the optional metric columns are missing or all values are unavailable.";

const G2M_DEEPH_DERIVATIVE_METRIC_HELP = {
  dh_mae_union_eV_per_Ang: {
    label: "dH MAE union",
    formula: "\\operatorname{MAE}=\\operatorname{mean}_{ij\\in R\\cup P}|(dH^{pred}/dR)_{ij}-(dH^{ref}/dR)_{ij}|",
    description: "Error absoluto medio entre la derivada del Hamiltoniano predicha y la referencia SIESTA en la union de soportes sparse.",
    purpose: "Da el error tipico de dH/dR sin penalizar excesivamente outliers.",
    direction: "Menor es mejor. Unidades: eV/Ang. Es diagnostico, no decide winners.",
  },
  dh_rmse_union_eV_per_Ang: {
    label: "dH RMSE union",
    formula: "\\operatorname{RMSE}=\\sqrt{\\operatorname{mean}_{ij\\in R\\cup P}\\left((dH^{pred}/dR)_{ij}-(dH^{ref}/dR)_{ij}\\right)^2}",
    description: "Raiz del error cuadratico medio de la derivada del Hamiltoniano en la union de soportes.",
    purpose: "Resalta fallos grandes que una MAE puede suavizar.",
    direction: "Menor es mejor. Si RMSE sube mucho frente a MAE, hay outliers relevantes.",
  },
  dh_relative_frobenius_ref: {
    label: "dH relative Frobenius ref",
    formula: "\\frac{\\|dH^{pred}/dR-dH^{ref}/dR\\|_{F,R}}{\\|dH^{ref}/dR\\|_{F,R}}",
    description: "Norma Frobenius relativa del error de derivada sobre el soporte de referencia.",
    purpose: "Mide el error global normalizado por la escala de la derivada SIESTA.",
    direction: "Menor es mejor; valores cercanos a 0 indican derivadas de matriz mas parecidas. Si dh_signal_to_noise_ratio < 1, un valor alto refleja piso de ruido del modelo, no necesariamente peor derivada.",
  },
  dh_signal_to_noise_ratio: {
    label: "dH signal-to-noise ratio",
    formula: "\\frac{\\|H_{+}-H_{-}\\|_{F}}{\\tfrac12(\\|H^{pred}_{+}-H^{ref}_{+}\\|_{F}+\\|H^{pred}_{-}-H^{ref}_{-}\\|_{F})}",
    description: "Senal fisica del desplazamiento (||H+ - H-||) dividida por el error de prediccion del H absoluto del modelo.",
    purpose: "Contextualiza dh_relative_frobenius_ref: la diferencia finita resta dos H casi identicos, asi que si el error de H absoluto supera la senal, la derivada predicha es ruido.",
    direction: "Mayor es mejor. SNR < 1 (dh_signal_below_noise_floor=true) => la derivada esta enterrada bajo el error del modelo; el error alto no es un bug sino falta de resolucion.",
  },
  dh_relative_frobenius_union_robust: {
    label: "Robust dH relative Frobenius",
    formula: "\\frac{\\|dH^{pred}/dR-dH^{ref}/dR\\|_{F,R\\cup P}}{\\max(\\|dH^{ref}/dR\\|_{F,R\\cup P},\\epsilon)}",
    description: "Version robusta de Frobenius relativo en la union de soportes, protegida frente a normas de referencia casi cero.",
    purpose: "Compara error relativo de dH/dR entre modelos, atomos, ejes o tamanos de dataset.",
    direction: "Menor es mejor. Si la referencia es muy pequena, interpretalo como diagnostico numerico.",
  },
  dh_relative_l1_union_robust: {
    label: "Robust dH relative L1",
    formula: "\\frac{\\|dH^{pred}/dR-dH^{ref}/dR\\|_{1,R\\cup P}}{\\max(\\|dH^{ref}/dR\\|_{1,R\\cup P},\\epsilon)}",
    description: "Error relativo L1 robusto de la derivada del Hamiltoniano en la union de soportes.",
    purpose: "Complementa Frobenius: L1 es menos dominado por pocos errores enormes.",
    direction: "Menor es mejor.",
  },
  dh_support_f1: {
    label: "dH support F1",
    formula: "F_1=\\frac{2\\,precision\\,recall}{precision+recall}",
    description: "F1 del soporte sparse de la derivada: combina si el modelo encuentra entradas dH/dR activas y si evita activaciones falsas.",
    purpose: "Separa errores de patron sparse de errores de valor numerico.",
    direction: "Mayor es mejor; 1 es soporte perfecto.",
  },
  dh_false_zero_rate: {
    label: "dH false-zero rate",
    formula: "\\frac{|\\{ij:(dH^{ref}/dR)_{ij}\\ne0\\land(dH^{pred}/dR)_{ij}=0\\}|}{|R|}",
    description: "Fraccion del soporte de referencia que el modelo deja como cero.",
    purpose: "Detecta acoplamientos derivados que la prediccion pierde.",
    direction: "Menor es mejor.",
  },
  dh_false_nonzero_rate: {
    label: "dH false-nonzero rate",
    formula: "\\frac{|\\{ij:(dH^{pred}/dR)_{ij}\\ne0\\land(dH^{ref}/dR)_{ij}=0\\}|}{|P|}",
    description: "Fraccion del soporte predicho que no existe en la referencia.",
    purpose: "Detecta acoplamientos derivados espurios.",
    direction: "Menor es mejor.",
  },
  dh_pearson_union: {
    label: "dH Pearson correlation",
    formula: "\\rho=\\operatorname{corr}(dH^{pred}/dR,dH^{ref}/dR)",
    description: "Correlacion lineal entre derivadas predichas y de referencia en la union de soportes.",
    purpose: "Mira si el modelo sigue la tendencia de magnitudes y signos.",
    direction: "Mayor es mejor; 1 indica correlacion lineal perfecta.",
  },
  dh_spearman_union: {
    label: "dH Spearman correlation",
    formula: "\\rho_s=\\operatorname{corr}(rank(dH^{pred}/dR),rank(dH^{ref}/dR))",
    description: "Correlacion de rangos entre derivadas predichas y de referencia.",
    purpose: "Comprueba si el modelo ordena bien entradas grandes y pequenas aunque la escala no sea perfecta.",
    direction: "Mayor es mejor; 1 indica mismo orden relativo.",
  },
  dh_residual_mean_union_eV_per_Ang: {
    label: "Mean dH residual",
    formula: "\\operatorname{mean}_{R\\cup P}(dH^{pred}/dR-dH^{ref}/dR)",
    description: "Sesgo medio firmado del error de derivada.",
    purpose: "Detecta si el modelo desplaza sistematicamente las derivadas hacia arriba o abajo.",
    direction: "Cercano a 0 es mejor.",
  },
  dh_residual_std_union_eV_per_Ang: {
    label: "dH residual std",
    formula: "\\operatorname{std}_{R\\cup P}(dH^{pred}/dR-dH^{ref}/dR)",
    description: "Dispersion de los residuos de dH/dR.",
    purpose: "Resume variabilidad del error despues del sesgo medio.",
    direction: "Menor es mejor.",
  },
  dh_residual_median_union_eV_per_Ang: {
    label: "Median dH residual",
    formula: "\\operatorname{median}_{R\\cup P}(dH^{pred}/dR-dH^{ref}/dR)",
    description: "Residuo mediano firmado de la derivada.",
    purpose: "Da una lectura robusta del sesgo central.",
    direction: "Cercano a 0 es mejor.",
  },
  dh_residual_bias_over_mae_union: {
    label: "dH bias over MAE",
    formula: "\\frac{|\\operatorname{mean}(residual)|}{\\operatorname{MAE}}",
    description: "Cuanto del error absoluto medio viene de sesgo sistematico.",
    purpose: "Distingue sesgo global de ruido disperso.",
    direction: "Menor es mejor.",
  },
  dh_residual_abs_p90_union_eV_per_Ang: {
    label: "dH abs residual p90",
    formula: "P_{90}(|dH^{pred}/dR-dH^{ref}/dR|)",
    description: "Percentil 90 del error absoluto de derivada.",
    purpose: "Mide la cola de errores grandes sin depender del maximo.",
    direction: "Menor es mejor.",
  },
  dh_residual_abs_p95_union_eV_per_Ang: {
    label: "dH abs residual p95",
    formula: "P_{95}(|dH^{pred}/dR-dH^{ref}/dR|)",
    description: "Percentil 95 del error absoluto de derivada.",
    purpose: "Vigila outliers fuertes de dH/dR.",
    direction: "Menor es mejor.",
  },
  dh_residual_abs_p99_union_eV_per_Ang: {
    label: "dH abs residual p99",
    formula: "P_{99}(|dH^{pred}/dR-dH^{ref}/dR|)",
    description: "Percentil 99 del error absoluto de derivada.",
    purpose: "Muestra la cola extrema sin usar un maximo posiblemente inestable.",
    direction: "Menor es mejor.",
  },
  dH_pred_hermiticity_defect: {
    label: "Predicted dH Hermiticity defect",
    formula: "\\frac{\\|dH^{pred}/dR-(dH^{pred}/dR)^\\dagger\\|_F}{\\|dH^{pred}/dR\\|_F}",
    description: "Defecto de Hermiticidad de la derivada predicha.",
    purpose: "Comprueba si la derivada predicha respeta la simetria fisica esperada.",
    direction: "Menor es mejor; cero es ideal.",
  },
  dH_hermiticity_error_delta: {
    label: "dH Hermiticity error delta",
    formula: "|h_{pred}-h_{ref}|",
    description: "Diferencia entre defecto de Hermiticidad predicho y de referencia.",
    purpose: "Detecta si el modelo introduce mas no-Hermiticidad que la referencia.",
    direction: "Menor es mejor.",
  },
};

const G2M_DEEPH_DERIVATIVE_PLOT_HELP_BY_ID = {
  error_vs_delta: {
    purpose: "Comprueba estabilidad frente al paso de diferencia finita.",
    direction: "Curvas bajas y estables al cambiar delta son mas fiables.",
  },
  graph2mat_vs_deeph_paired_comparison: {
    metric: "Paired dH MAE",
    formula: "x=\\operatorname{MAE}_{Graph2Mat},\\quad y=\\operatorname{MAE}_{DeepH}",
    description: "Compara errores de dH/dR de ambos modelos sobre los mismos stencils.",
    purpose: "Permite ver pares donde un metodo falla mas que el otro.",
    direction: "Puntos bajo la diagonal favorecen DeepH en MAE; puntos sobre la diagonal favorecen Graph2Mat. Sigue siendo diagnostico.",
  },
  derivative_error_by_abs_ref_quantile: {
    purpose: "Muestra si los errores se concentran en derivadas pequenas, medianas o grandes.",
    direction: "Menor es mejor en todos los cuantiles; colas altas indican regimenes dificiles.",
  },
  derivative_relative_l1_by_abs_ref_quantile: {
    purpose: "Normaliza el error por escala de referencia en cada cuantil.",
    direction: "Menor es mejor; cuantiles pequenos pueden ser ruidosos si la referencia es casi cero.",
  },
  robust_error_by_displaced_atom: {
    purpose: "Localiza atomos desplazados que producen derivadas mas dificiles.",
    direction: "Barras mas bajas son mejores; atomos altos piden inspeccion local.",
  },
  robust_error_by_axis: {
    purpose: "Separa sensibilidad de dH/dR por direccion cartesiana.",
    direction: "Barras mas bajas son mejores; anisotropias grandes pueden indicar geometria o modelo dificil.",
  },
  robust_error_by_atom_axis: {
    purpose: "Combina atomos y ejes para encontrar casos locales problematicos.",
    direction: "Barras mas bajas son mejores.",
  },
  onsite_offsite_derivative_error: {
    purpose: "Separa errores onsite y offsite para distinguir terminos locales de acoplamientos entre atomos.",
    direction: "Menor es mejor; compara onsite/offsite solo con la misma base y convenciones.",
  },
  support_change_false_zero_false_nonzero: {
    metric: "dH support-change diagnostics",
    formula: "support\\ change,\\ false\\ zero,\\ false\\ nonzero",
    description: "Resume cambios de soporte sparse y errores de ceros/no-ceros en dH/dR.",
    purpose: "Distingue fallos de patron sparse de fallos de magnitud.",
    direction: "Menor es mejor para las tres tasas.",
  },
  signal_to_noise_vs_dataset_size: {
    metric: "dH signal-to-noise ratio",
    title: "dH signal-to-noise ratio vs dataset size",
    formula: "\\mathrm{SNR}=\\frac{\\|H_{+}-H_{-}\\|_{F}}{\\tfrac12(\\|H^{pred}_{+}-H^{ref}_{+}\\|_{F}+\\|H^{pred}_{-}-H^{ref}_{-}\\|_{F})}",
    description: "Senal fisica del desplazamiento dividida por el error de prediccion del H absoluto del modelo, promediada por tamano de dataset.",
    purpose: "Contextualiza el Relative Frobenius de arriba: la diferencia finita resta dos H casi identicos, asi que si el error de H absoluto supera la senal la derivada predicha es ruido.",
    direction: "Mayor es mejor. SNR < 1 significa que la derivada esta enterrada bajo el error del modelo: un Relative Frobenius alto refleja falta de resolucion, no necesariamente peor modelo.",
  },
};

function renderG2MDeepHDerivativeRunSelector() {
  const list = document.getElementById("g2m-deeph-derivative-run-list");
  if (!list) return;
  const runs = state.g2mDeephPlotRuns || [];
  const visibleIds = new Set(runs.map((run) => run.run_id || run.id).filter(Boolean));
  let selectedRunIds = (state.g2mDeephDerivativeRunIds || []).filter((id) => visibleIds.has(id));
  if (!selectedRunIds.length) selectedRunIds = (state.g2mDeephSelectedPlotRunIds || []).filter((id) => visibleIds.has(id));
  if (!selectedRunIds.length && preferredG2MDeepHDerivativeRunId()) selectedRunIds = [preferredG2MDeepHDerivativeRunId()];
  state.g2mDeephDerivativeRunIds = selectedRunIds;
  state.g2mDeephDerivativeRunId = selectedRunIds[0] || null;
  list.textContent = "";
  if (!runs.length) {
    list.textContent = "No Graph2Mat vs DeepH runs available yet";
    return;
  }
  for (const run of runs) {
    const value = run.run_id || run.id || "";
    const option = document.createElement("label");
    option.className = "plot-run-option";
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.className = "g2m-deeph-derivative-run-checkbox";
    checkbox.value = value;
    checkbox.checked = selectedRunIds.includes(value);
    checkbox.addEventListener("change", () => {
      const next = new Set(state.g2mDeephDerivativeRunIds || []);
      if (checkbox.checked) next.add(value);
      else next.delete(value);
      state.g2mDeephDerivativeRunIds = Array.from(next);
      state.g2mDeephDerivativeRunId = state.g2mDeephDerivativeRunIds[0] || null;
      renderG2MDeepHDerivativeRunSelector();
      loadG2MDeepHDerivativeMetrics().catch((error) => showToast(error.message));
    });
    const body = document.createElement("span");
    const title = document.createElement("strong");
    title.textContent = run.label || run.run_id || run.id;
    const details = document.createElement("span");
    details.textContent = [(run.models || []).map(methodDisplayLabel).join("+"), (run.dataset_ids || []).join(", "), run.status || ""].filter(Boolean).join(" | ");
    body.appendChild(title);
    body.appendChild(details);
    option.appendChild(checkbox);
    option.appendChild(body);
    list.appendChild(option);
  }
}

function g2mDeephDerivativeStatusText(payload = {}) {
  if (payload.not_computed) return payload.message || G2M_DEEPH_DERIVATIVE_OPTIONAL_POSTPROCESSING;
  const statuses = (payload.status_rows || []).map((row) => row.scientific_status).filter(Boolean);
  const distinct = Array.from(new Set(statuses));
  return distinct.length ? `Derivative status: ${distinct.join(", ")}` : "Derivative diagnostics loaded.";
}

function g2mDeephDerivativePlotLabel(row = {}) {
  const run = row.run_label || row.run_id || "";
  const suffix = run ? ` · ${run}` : "";
  if (row.method && row.axis) return `${row.axis} · ${row.method}${suffix}`;
  if (row.method && row.atom_index_zero_based != null) return `atom ${row.atom_index_zero_based} · ${row.method}${suffix}`;
  if (row.method) return `${row.method}${suffix}`;
  if (row.axis) return `${row.axis}${suffix}`;
  if (row.atom_index_zero_based != null) return `atom ${row.atom_index_zero_based}${suffix}`;
  return `${row.sample || "value"}${suffix}`;
}

function g2mDeephDerivativeMetricHelp(metricKey) {
  const key = String(metricKey || "").trim();
  if (G2M_DEEPH_DERIVATIVE_METRIC_HELP[key]) return G2M_DEEPH_DERIVATIVE_METRIC_HELP[key];
  return {
    label: key || "Derivative metric",
    formula: "m=\\operatorname{mean}_{\\text{finite derivative rows}}(metric)",
    description: "Metrica diagnostica calculada sobre diferencias finitas de Hamiltonianos: dH_pred/dR frente a dH_ref/dR.",
    purpose: "Sirve para inspeccionar el comportamiento de derivadas sin cambiar el winner de metricas H-vs-H.",
    direction: "Interpreta la direccion segun el nombre de la metrica; errores y tasas bajan, correlaciones y F1 suben.",
  };
}

function g2mDeephDerivativeIsDatasetSizePlot(plot = {}) {
  return Boolean(plot.dataset_size_plot || String(plot.id || "").includes("_vs_dataset_size"));
}

function g2mDeephDerivativePlotTitle(plot = {}) {
  return G2M_DEEPH_DERIVATIVE_DATASET_SIZE_TITLES[plot.id] || plot.title || "Derivative diagnostic";
}

function g2mDeephDerivativeHoverText(row = {}) {
  const listText = (value) => Array.isArray(value) ? value.join(",") : value;
  return [
    row.dataset_ids?.join?.(","),
    row.dataset_id,
    row.x_dataset_size != null ? `${row.x_dataset_size_kind || "dataset size"} ${row.x_dataset_size}` : "",
    row.n_train != null ? `N_train ${row.n_train}` : "",
    row.n_total != null ? `N_total ${row.n_total}` : "",
    row.dataset_size_source ? `source ${row.dataset_size_source}` : "",
    row.delta_ang != null ? `delta ${row.delta_ang}` : "",
    listText(row.delta_values) ? `deltas ${listText(row.delta_values)}` : "",
    row.axes ? `axes ${listText(row.axes)}` : row.axis,
    row.atom_indices ? `atoms ${listText(row.atom_indices)}` : (row.atom_index_zero_based != null ? `atom ${row.atom_index_zero_based}` : ""),
    row.finite_difference_method,
    row.n_rows ? `rows ${row.n_rows}` : "",
    row.n_stencils ? `stencils ${row.n_stencils}` : "",
    row.sample,
  ].filter((item) => item !== undefined && item !== "").join(" | ");
}

function g2mDeephDerivativePlotInfo(plot = {}) {
  const metrics = (plot.metrics || []).filter((metric) => metric?.key);
  const primaryMetric = metrics[0]?.key || plot.y_key || "";
  const help = g2mDeephDerivativeMetricHelp(primaryMetric);
  const explicit = G2M_DEEPH_DERIVATIVE_PLOT_HELP_BY_ID[plot.id] || {};
  const datasetSizePlot = g2mDeephDerivativeIsDatasetSizePlot(plot);
  const metricLabel = metrics.length > 1
    ? metrics.map((metric) => g2mDeephDerivativeMetricHelp(metric.key).label || metric.label || metric.key).join(", ")
    : (explicit.metric || help.label || plot.y_title || primaryMetric);
  return {
    title: explicit.title || g2mDeephDerivativePlotTitle(plot),
    metric: metricLabel,
    formula: explicit.formula || help.formula,
    description: datasetSizePlot ? G2M_DEEPH_DERIVATIVE_DATASET_SIZE_NOTE : (explicit.description || help.description),
    purpose: explicit.purpose || (datasetSizePlot ? "Prioritizes how mean derivative errors change with dataset size before lower-level diagnostics." : help.purpose),
    direction: explicit.direction || help.direction || G2M_DEEPH_DERIVATIVE_INTERNAL_ONLY,
  };
}

function renderG2MDeepHDerivativeGroupedBarPlot(card, plot) {
  const rows = plot.rows || [];
  const labels = rows.map((row) => g2mDeephDerivativePlotLabel(row));
  const traces = (plot.metrics || []).map((metric) => ({
    type: "bar",
    x: labels,
    y: rows.map((row) => finiteNumber(row[metric.key])),
    name: metric.label || metric.key,
    hovertemplate: `%{x}<br>%{fullData.name}: %{y:.5g}<extra></extra>`,
  }));
  const finiteValues = traces.flatMap((trace) => (trace.y || []).filter((value) => value != null));
  renderPlot(
    card,
    traces,
    plotLayout(g2mDeephDerivativePlotTitle(plot), plot.metrics?.[0]?.unit ? `${plot.metrics[0].label} (${plot.metrics[0].unit})` : "", {
      barmode: "group",
      annotations: finiteValues.length ? [] : [emptyPlotAnnotation("No derivative values available for this plot.")],
      xaxis: { title: "Group", gridcolor: "#edf1f4", zeroline: false },
      yaxis: { title: plot.metrics?.[0]?.unit ? `${plot.metrics[0].label} (${plot.metrics[0].unit})` : plot.title || "", gridcolor: "#edf1f4", zeroline: false },
    }),
    { plotInfo: g2mDeephDerivativePlotInfo(plot) },
  );
}

function g2mDeephDerivativeDatasetSizeAxisField() {
  return state.g2mDeephDerivativeDatasetSizeAxis === "n_total" ? "n_total" : "n_train";
}

function g2mDeephDerivativeXValue(row, plot) {
  if (g2mDeephDerivativeIsDatasetSizePlot(plot)) {
    const field = g2mDeephDerivativeDatasetSizeAxisField();
    const value = finiteNumber(row[field]);
    if (value != null) return value;
  }
  return finiteNumber(row[plot.x_key || "x"]);
}

function g2mDeephDerivativeXAxisTitle(plot) {
  if (g2mDeephDerivativeIsDatasetSizePlot(plot)) {
    return g2mDeephDerivativeDatasetSizeAxisField() === "n_total"
      ? "N_total snapshots (train+val+test)"
      : "N_train snapshots";
  }
  return plot.x_title || "x";
}

function renderG2MDeepHDerivativeScatterPlot(card, plot) {
  const grouped = new Map();
  for (const row of plot.rows || []) {
    const seriesKey = plot.series_key || "method";
    const key = String(row.combined_series || row[seriesKey] || row.method || row.finite_difference_method || "diagnostic");
    if (!grouped.has(key)) grouped.set(key, []);
    grouped.get(key).push(row);
  }
  const metrics = (plot.metrics && plot.metrics.length) ? plot.metrics : [{ key: plot.y_key, label: plot.y_title || plot.y_key }];
  const mode = G2M_DEEPH_DERIVATIVE_MARKER_ONLY_PLOTS.has(plot.id) ? "markers" : null;
  const series = Array.from(grouped.entries()).flatMap(([key, rows], index) => {
    const sortedRows = [...rows].sort((left, right) => {
      const leftX = g2mDeephDerivativeXValue(left, plot);
      const rightX = g2mDeephDerivativeXValue(right, plot);
      if (leftX == null && rightX == null) return 0;
      if (leftX == null) return 1;
      if (rightX == null) return -1;
      return leftX - rightX;
    });
    return metrics.map((metric, metricIndex) => ({
      key,
      rows: sortedRows,
      metric,
      color: plotColor(index),
      name: metrics.length > 1 ? `${key} · ${metric.label || metric.key}` : key,
      dash: metricIndex ? "dash" : "solid",
      xValues: sortedRows.map((row) => g2mDeephDerivativeXValue(row, plot)),
      yValues: sortedRows.map((row) => finiteNumber(row[metric.key])),
    }));
  });
  const allYValues = series.flatMap((entry) => entry.yValues.filter((value) => value != null));
  const fitYMin = valuesAreNonNegative(allYValues) ? 0 : null;
  const traces = [];
  for (const entry of series) {
    traces.push({
      type: "scatter",
      mode: mode || (entry.rows.length > 1 ? "lines+markers" : "markers"),
      name: entry.name,
      marker: { symbol: g2mDeephIsDeepH(entry.key) ? "triangle-up" : "circle", size: 9, color: entry.color },
      line: { color: entry.color, dash: entry.dash },
      x: entry.xValues,
      y: entry.yValues,
      text: entry.rows.map((row) => g2mDeephDerivativeHoverText(row)),
      hovertemplate: `%{x:.5g}, %{y:.5g}<br>%{text}<extra>%{fullData.name}</extra>`,
    });
  }
  // Fits are computed on the pooled data of every visible sweep/run sharing
  // the same metric AND model (graph2mat vs deeph stay separate fits).
  const combinedFitGroups = new Map();
  for (const entry of series) {
    const modelKey = String(entry.rows[0]?.model || entry.rows[0]?.model_label || entry.key);
    const modelLabel = String(entry.rows[0]?.model_label || modelKey);
    const groupKey = `${entry.metric.key}|${modelKey}`;
    if (!combinedFitGroups.has(groupKey)) {
      combinedFitGroups.set(groupKey, { metric: entry.metric, modelLabel, points: [] });
    }
    const bucket = combinedFitGroups.get(groupKey);
    entry.xValues.forEach((x, position) => {
      const y = entry.yValues[position];
      if (x != null && y != null) bucket.points.push({ x, y });
    });
  }
  const COMBINED_FIT_COLORS = ["#374151", "#7c3aed", "#0f766e", "#b45309"];
  Array.from(combinedFitGroups.values()).forEach((group, index) => {
    const name = metrics.length > 1
      ? `${group.modelLabel} · ${group.metric.label || group.metric.key} combined`
      : `${group.modelLabel} combined`;
    addFitTraces(traces, group.points, name, COMBINED_FIT_COLORS[index % COMBINED_FIT_COLORS.length], {
      legendgroup: `combined|${group.metric.key}|${group.modelLabel}`,
      fitYMin,
    });
  });
  const diagonalShapes = [];
  const diagonalAnnotations = [];
  let pairedAxisRange = null;
  if (plot.id === "graph2mat_vs_deeph_paired_comparison") {
    const allXValues = series.flatMap((entry) => entry.xValues.filter((value) => value != null));
    if (allXValues.length && allYValues.length) {
      const diagonalMax = Math.max(0, ...allXValues, ...allYValues);
      if (Number.isFinite(diagonalMax) && diagonalMax > 0) {
        pairedAxisRange = [0, diagonalMax * 1.06];
        diagonalShapes.push({
          type: "line",
          xref: "x",
          yref: "y",
          x0: 0,
          y0: 0,
          x1: diagonalMax,
          y1: diagonalMax,
          line: { color: "#9ca3af", width: 1.5, dash: "dot" },
          layer: "below",
        });
        diagonalAnnotations.push({
          x: diagonalMax,
          y: diagonalMax,
          xref: "x",
          yref: "y",
          text: "y = x",
          showarrow: false,
          xanchor: "left",
          yanchor: "bottom",
          font: { size: 12 },
        });
      }
    }
  }
  let layout = plotLayout(g2mDeephDerivativePlotTitle(plot), plot.y_title || "", {
    annotations: traces.length ? diagonalAnnotations : [emptyPlotAnnotation("No derivative scatter rows available.")],
    shapes: diagonalShapes,
    xaxis: {
      title: g2mDeephDerivativeXAxisTitle(plot),
      gridcolor: "#edf1f4",
      zeroline: false,
      ...(pairedAxisRange ? { range: pairedAxisRange } : {}),
    },
    yaxis: {
      title: plot.y_title || "y",
      gridcolor: "#edf1f4",
      zeroline: false,
      ...(pairedAxisRange ? { range: pairedAxisRange } : (fitYMin === 0 ? { range: paddedLinearRange(allYValues, { forceZeroMin: true }) } : {})),
    },
    legend: { orientation: "h", y: -0.24 },
  });
  layout = withFitSelector(layout, traces);
  renderPlot(
    card,
    traces,
    layout,
    { plotInfo: g2mDeephDerivativePlotInfo(plot) },
  );
}

function g2mDeephDerivativePairedAboveBelowCounts(plot) {
  const metricKey = plot.metrics?.[0]?.key || plot.y_key;
  let below = 0;
  let above = 0;
  let onLine = 0;
  for (const row of plot.rows || []) {
    const x = finiteNumber(row[plot.x_key || "x"]);
    const y = finiteNumber(row[metricKey]);
    if (x == null || y == null) continue;
    if (y < x) below += 1;
    else if (y > x) above += 1;
    else onLine += 1;
  }
  return { below, above, onLine };
}

function renderG2MDeepHDerivativePairedAboveBelowPlot(card, plot) {
  const counts = g2mDeephDerivativePairedAboveBelowCounts(plot);
  const total = counts.below + counts.above + counts.onLine;
  const labels = ["Below y=x<br>(DeepH better)", "Above y=x<br>(Graph2Mat better)"];
  const values = [counts.below, counts.above];
  const colors = [plotColor(1), plotColor(0)];
  const trace = {
    type: "bar",
    x: labels,
    y: values,
    marker: { color: colors },
    text: values.map((value) => String(value)),
    textposition: "outside",
    hovertemplate: "%{x}<br>points: %{y}<extra></extra>",
  };
  const subtitle = total
    ? `${counts.below} below, ${counts.above} above${counts.onLine ? `, ${counts.onLine} on the line` : ""} (of ${total} stencils)`
    : "";
  renderPlot(
    card,
    [trace],
    plotLayout("Points above vs below y = x", "Stencil count", {
      annotations: total ? [] : [emptyPlotAnnotation("No paired comparison rows available.")],
      xaxis: { title: "", gridcolor: "#edf1f4", zeroline: false },
      yaxis: { title: "Stencil count", gridcolor: "#edf1f4", zeroline: false, rangemode: "tozero" },
      showlegend: false,
      margin: { l: 56, r: 16, t: 50, b: 70 },
    }),
    {
      plotInfo: {
        title: "Points above vs below y = x",
        metric: "Stencil count",
        description: subtitle || "Counts how many graph2mat-vs-deeph paired stencils fall below vs above the y = x diagonal.",
        purpose: "Summarizes the paired-comparison scatter into a single per-model win/loss tally on dH MAE.",
        direction: "Below the line favors DeepH; above the line favors Graph2Mat. Diagnostic only.",
      },
    },
  );
}

function renderG2MDeepHDerivativePlotSummary(card, plot, message = "") {
  card.textContent = "";
  const title = document.createElement("h4");
  title.textContent = g2mDeephDerivativePlotTitle(plot) || plot.id || "Derivative diagnostic plot";
  card.appendChild(title);
  const datasetSizeNote = g2mDeephDerivativeIsDatasetSizePlot(plot) ? G2M_DEEPH_DERIVATIVE_DATASET_SIZE_NOTE : "";
  for (const text of [plot.subtitle, plot.reference_label, datasetSizeNote, message].filter(Boolean)) {
    const paragraph = document.createElement("p");
    paragraph.className = "field-help";
    paragraph.textContent = text;
    card.appendChild(paragraph);
  }
  if (!message) {
    const details = document.createElement("p");
    details.className = "field-help";
    details.textContent = [
      plot.id ? `id: ${plot.id}` : "",
      plot.kind ? `kind: ${plot.kind}` : "",
      plot.x_key ? `x: ${plot.x_key}` : "",
      plot.y_key ? `y: ${plot.y_key}` : "",
      plot.series_key ? `series: ${plot.series_key}` : "",
      (plot.metrics || []).length ? `metrics: ${(plot.metrics || []).map((metric) => metric.label || metric.key).join(", ")}` : "",
      `rows: ${(plot.rows || []).length}`,
    ].filter(Boolean).join(" | ");
    card.appendChild(details);
  }
  installPlotInfoBubble(card, g2mDeephDerivativePlotInfo(plot));
}

function g2mDeephDerivativePlotSections(plotPayload = {}) {
  const plots = [...(plotPayload.plots || [])];
  const seen = new Set(plots.map((plot) => plot.id).filter(Boolean));
  for (const [id, title] of G2M_DEEPH_EXPECTED_DERIVATIVE_PLOTS) {
    if (seen.has(id)) continue;
    plots.push({
      id,
      title,
      kind: "grouped_bar",
      rows: [],
      unavailable_message: "No data available for this metric. Regenerate derivative_plot_payload.json from derivative metric CSV/JSON artifacts.",
    });
  }
  const primaryIds = new Set(plotPayload.primary_plot_ids || []);
  const diagnosticIds = new Set(plotPayload.diagnostic_plot_ids || []);
  const datasetSizeIds = new Set(plotPayload.dataset_size_plot_ids || []);
  const isDatasetSizePlot = (plot) => datasetSizeIds.has(plot.id) || g2mDeephDerivativeIsDatasetSizePlot(plot);
  return [
    { title: "Dataset-size derivative metrics", plots: plots.filter(isDatasetSizePlot) },
    { title: "Primary derivative metrics", plots: plots.filter((plot) => !isDatasetSizePlot(plot) && primaryIds.has(plot.id)) },
    { title: "Secondary derivative metrics", plots: plots.filter((plot) => !isDatasetSizePlot(plot) && !primaryIds.has(plot.id) && !diagnosticIds.has(plot.id)) },
    { title: "Diagnostic derivative metrics", plots: plots.filter((plot) => !isDatasetSizePlot(plot) && diagnosticIds.has(plot.id)) },
  ];
}

function g2mDeephDerivativeDatasetSizeNotice(plotPayload = {}) {
  const datasetSizeIds = new Set(plotPayload.dataset_size_plot_ids || []);
  if (!datasetSizeIds.size) return "";
  const hasBase = Array.from(G2M_DEEPH_BASE_DERIVATIVE_DATASET_SIZE_PLOTS).some((id) => datasetSizeIds.has(id));
  const hasOptional = G2M_DEEPH_OPTIONAL_DERIVATIVE_DATASET_SIZE_PLOTS.some((id) => datasetSizeIds.has(id));
  return hasBase && !hasOptional ? G2M_DEEPH_DERIVATIVE_DATASET_SIZE_PARTIAL_MESSAGE : G2M_DEEPH_DERIVATIVE_DATASET_SIZE_NOTE;
}

function renderG2MDeepHDerivativeDatasetSizeAxisControl(container, plotPayload) {
  if (!(plotPayload.dataset_size_plot_ids || []).length) return;
  const bar = document.createElement("div");
  bar.className = "plot-card full dataset-size-axis-control";
  const label = document.createElement("label");
  label.setAttribute("for", "g2m-deeph-derivative-dataset-size-axis");
  label.textContent = "Dataset-size x-axis: ";
  const select = document.createElement("select");
  select.id = "g2m-deeph-derivative-dataset-size-axis";
  for (const [value, text] of [
    ["n_train", "N_train (training snapshots)"],
    ["n_total", "N_total (train + validation + test snapshots)"],
  ]) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = text;
    if (g2mDeephDerivativeDatasetSizeAxisField() === value) option.selected = true;
    select.appendChild(option);
  }
  select.addEventListener("change", () => {
    state.g2mDeephDerivativeDatasetSizeAxis = select.value;
    renderG2MDeepHDerivativePlotsPayload(state.g2mDeephDerivativePayload || {});
  });
  bar.append(label, select);
  container.appendChild(bar);
}

function g2mDeephDerivativeMaePlot(payload = {}) {
  const plots = payload.plot_payload?.plots || payload.plots || [];
  return plots.find((plot) => plot.id === G2M_DEEPH_DERIVATIVE_MAE_DATASET_SIZE_PLOT_ID) || null;
}

function g2mDeephDerivativeHasMaePlot(payload = {}) {
  const plot = g2mDeephDerivativeMaePlot(payload);
  return Boolean(payload.available && plot && (plot.rows || []).length);
}

function g2mDeephDerivativeMaeSeriesLabel(row = {}, plot = {}) {
  const mode = row.mode ?? row.mixing_mode ?? row.combination_mode;
  const ratio = row.ratio ?? row.mixing_ratio ?? row.large_ratio;
  const model = row.model_label || methodDisplayLabel(row.model || row.source_model || "model");
  if (mode != null || ratio != null) {
    return `mode=${mode ?? "-"} · ratio=${ratio ?? "-"} · ${model}`;
  }
  const seriesKey = plot.series_key || "model_label";
  return String(row.combined_series || row[seriesKey] || model || "Derivative MAE");
}

function g2mDeephDerivativeMaeSeriesId(row = {}, plot = {}) {
  return g2mDeephDerivativeMaeSeriesLabel(row, plot);
}

function g2mDeephDerivativeMaeSeries(plot = {}) {
  const byId = new Map();
  for (const row of plot.rows || []) {
    const id = g2mDeephDerivativeMaeSeriesId(row, plot);
    if (!byId.has(id)) byId.set(id, { id, label: id, count: 0 });
    byId.get(id).count += 1;
  }
  return Array.from(byId.values()).sort((left, right) => left.label.localeCompare(right.label));
}

function renderG2MDeepHDerivativeMaeSeriesSelector(plot) {
  const status = document.getElementById("g2m-deeph-derivative-mae-series-status");
  const list = document.getElementById("g2m-deeph-derivative-mae-series-list");
  if (!status || !list) return;
  const series = plot ? g2mDeephDerivativeMaeSeries(plot) : [];
  list.textContent = "";
  if (!series.length) {
    status.textContent = "No derivative MAE dataset-size series available yet.";
    state.g2mDeephDerivativeMaeSeriesIds = null;
    return;
  }
  const visibleIds = new Set(series.map((item) => item.id));
  let selected = state.g2mDeephDerivativeMaeSeriesIds;
  selected = selected == null ? series.map((item) => item.id) : selected.filter((id) => visibleIds.has(id));
  state.g2mDeephDerivativeMaeSeriesIds = selected;
  status.textContent = `${selected.length}/${series.length} curve(s) selected.`;
  for (const item of series) {
    const option = document.createElement("label");
    option.className = "plot-run-option";
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.className = "g2m-deeph-derivative-mae-series-checkbox";
    checkbox.value = item.id;
    checkbox.checked = selected.includes(item.id);
    checkbox.addEventListener("change", () => {
      const next = new Set(state.g2mDeephDerivativeMaeSeriesIds || []);
      if (checkbox.checked) next.add(item.id);
      else next.delete(item.id);
      state.g2mDeephDerivativeMaeSeriesIds = Array.from(next);
      renderG2MDeepHDerivativeMaeDatasetPlot(state.g2mDeephDerivativePayload || {});
    });
    const body = document.createElement("span");
    const title = document.createElement("strong");
    title.textContent = item.label;
    const details = document.createElement("span");
    details.textContent = `${item.count} point(s)`;
    body.append(title, details);
    option.append(checkbox, body);
    list.appendChild(option);
  }
}

function setG2MDeepHDerivativeMaeSeriesSelection(selected) {
  const plot = g2mDeephDerivativeMaePlot(state.g2mDeephDerivativePayload || {});
  const ids = g2mDeephDerivativeMaeSeries(plot || {}).map((item) => item.id);
  state.g2mDeephDerivativeMaeSeriesIds = selected ? ids : [];
  renderG2MDeepHDerivativeMaeDatasetPlot(state.g2mDeephDerivativePayload || {});
}

function renderG2MDeepHDerivativeMaeDatasetPlot(payload = {}) {
  const host = document.getElementById("g2m-deeph-derivative-mae-dataset-chart");
  if (!host) return;
  const plot = g2mDeephDerivativeMaePlot(payload);
  renderG2MDeepHDerivativeMaeSeriesSelector(plot);
  if (!payload.available || !plot || !(plot.rows || []).length) {
    host.textContent = payload.not_computed
      ? (payload.message || G2M_DEEPH_DERIVATIVE_OPTIONAL_POSTPROCESSING)
      : "No derivative MAE vs dataset size data available yet.";
    return;
  }
  const selected = new Set(state.g2mDeephDerivativeMaeSeriesIds || []);
  const rows = (plot.rows || [])
    .map((row) => ({
      ...row,
      combined_series: g2mDeephDerivativeMaeSeriesLabel(row, plot),
      dh_mae_union_meV_per_Ang: finiteNumber(row.dh_mae_union_eV_per_Ang) == null
        ? null
        : finiteNumber(row.dh_mae_union_eV_per_Ang) * G2M_DEEPH_DERIVATIVE_MAE_EV_TO_MEV,
    }))
    .filter((row) => selected.has(row.combined_series));
  if (!rows.length) {
    host.textContent = "Selecciona al menos una curva para ver el MAE de derivadas.";
    return;
  }
  const filteredPlot = {
    ...plot,
    title: "dH MAE vs dataset size",
    y_key: "dh_mae_union_meV_per_Ang",
    y_title: "dH MAE (meV/Ang)",
    series_key: "combined_series",
    rows,
  };
  if (window.Plotly) {
    renderG2MDeepHDerivativeScatterPlot(host, filteredPlot);
  } else {
    host.textContent = "Cargando Plotly...";
    ensurePlotlyLoaded()
      .then(() => renderG2MDeepHDerivativeScatterPlot(host, filteredPlot))
      .catch(() => renderG2MDeepHDerivativePlotSummary(host, filteredPlot));
  }
}

function renderG2MDeepHDerivativePlotsPayload(payload = {}) {
  const container = document.getElementById("g2m-deeph-derivative-plots");
  if (!container) return;
  container.textContent = "";
  const plotPayload = payload.plot_payload || {};
  if (!payload.available || !plotPayload.available || !(plotPayload.plots || []).length) {
    renderG2MDeepHDerivativeMaeDatasetPlot(payload);
    const placeholder = document.createElement("div");
    placeholder.className = "plot-card full placeholder-card";
    placeholder.textContent = payload.not_computed
      ? (payload.message || G2M_DEEPH_DERIVATIVE_OPTIONAL_POSTPROCESSING)
      : "No derivative plots available yet.";
    container.appendChild(placeholder);
    return;
  }
  renderG2MDeepHDerivativeMaeDatasetPlot(payload);
  renderG2MDeepHDerivativeDatasetSizeAxisControl(container, plotPayload);
  const datasetSizeNotice = g2mDeephDerivativeDatasetSizeNotice(plotPayload);
  if (datasetSizeNotice) {
    const note = document.createElement("div");
    note.className = "plot-card full placeholder-card";
    note.textContent = datasetSizeNotice;
    container.appendChild(note);
  }
  for (const section of g2mDeephDerivativePlotSections(plotPayload)) {
    const heading = document.createElement("h3");
    heading.className = "plot-section-title";
    heading.textContent = section.title;
    container.appendChild(heading);
    if (!section.plots.length) {
      const placeholder = document.createElement("div");
      placeholder.className = "plot-card full placeholder-card";
      placeholder.textContent = "No data available for this metric";
      container.appendChild(placeholder);
      continue;
    }
    for (const plot of section.plots) {
      const hasSideBySideCompanion = plot.id === "graph2mat_vs_deeph_paired_comparison" && (plot.rows || []).length;
      let host = container;
      if (hasSideBySideCompanion) {
        const row = document.createElement("div");
        row.className = "plot-card-row";
        container.appendChild(row);
        host = row;
      }
      const card = document.createElement("div");
      card.id = `g2m-deeph-derivative-plot-${plot.id || container.children.length}`;
      card.className = hasSideBySideCompanion ? "plot-card wide plot-card-row-main" : "plot-card wide";
      host.appendChild(card);
      if (!(plot.rows || []).length) {
        renderG2MDeepHDerivativePlotSummary(card, plot, plot.unavailable_message || "No data available for this metric");
      } else if (window.Plotly && plot.kind === "grouped_bar") {
        renderG2MDeepHDerivativeGroupedBarPlot(card, plot);
      } else if (window.Plotly && (plot.kind === "scatter" || plot.kind === "line")) {
        renderG2MDeepHDerivativeScatterPlot(card, plot);
      } else {
        renderG2MDeepHDerivativePlotSummary(card, plot);
      }
      if (hasSideBySideCompanion && window.Plotly) {
        const sideCard = document.createElement("div");
        sideCard.id = `g2m-deeph-derivative-plot-${plot.id}-above-below`;
        sideCard.className = "plot-card plot-card-row-side";
        host.appendChild(sideCard);
        renderG2MDeepHDerivativePairedAboveBelowPlot(sideCard, plot);
      }
    }
  }
  schedulePlotResize();
}

function renderG2MDeepHDerivativeArtifacts(payload = {}) {
  const container = document.getElementById("g2m-deeph-derivative-artifacts");
  if (!container) return;
  container.textContent = "";
  const rows = payload.artifact_rows || [];
  if (!rows.length) {
    container.className = "result-list muted-text";
    container.textContent = payload.not_computed
      ? (payload.message || G2M_DEEPH_DERIVATIVE_OPTIONAL_POSTPROCESSING)
      : "No derivative CSV or manifest links available.";
    return;
  }
  container.className = "result-list g2m-deeph-derivative-artifact-list";
  for (const row of rows) {
    const line = document.createElement("div");
    if (row.exists && row.url) {
      const link = document.createElement("a");
      link.href = row.url;
      link.textContent = row.label || row.kind;
      link.className = "g2m-deeph-derivative-artifact-link";
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      line.appendChild(link);
    } else {
      const text = document.createElement("span");
      text.textContent = `${row.label || row.kind}: not available`;
      line.appendChild(text);
    }
    container.appendChild(line);
  }
}

function renderG2MDeepHDerivativeGateReport(payload = {}) {
  const container = document.getElementById("g2m-deeph-derivative-gate-report");
  if (!container) return;
  container.textContent = "";
  const gateReport = payload.gate_report || {};
  if (!payload.available && !Object.keys(gateReport).length) {
    container.className = "result-list muted-text";
    container.textContent = payload.not_computed
      ? (payload.message || G2M_DEEPH_DERIVATIVE_OPTIONAL_POSTPROCESSING)
      : "No derivative gate report available.";
    return;
  }
  container.className = "result-list";
  const banner = document.createElement("div");
  banner.className = `comparison-status-banner ${gateReport.scientific_status === "blocked" ? "invalid" : "diagnostic"}`;
  banner.textContent = gateReport.message || "diagnostic-only / no winner claim";
  container.appendChild(banner);
  appendG2MDeepHTable(
    container,
    "Derivative gate report",
    [
      { key: "common_metrics_status", label: "Common metrics status" },
      { key: "common_recommendation_status", label: "Common recommendation" },
      { key: "ranking_status", label: "Ranking status" },
      { key: "ranking_scientific_status", label: "Scientific status" },
      { key: "scientific_status", label: "Derivative gate status" },
      { key: "winner", label: "Benchmark winner", format: (value) => (value ? methodDisplayLabel(value) : "none") },
      { key: "primary_metric", label: "Primary metric" },
      { key: "derivative_winner_claim", label: "Derivative winner claim" },
      { key: "ranking_reason", label: "Gate report reason" },
    ],
    [gateReport],
    "No derivative gate report available.",
  );
  appendG2MDeepHTable(
    container,
    "Scientific gates",
    [
      { key: "gate", label: "Gate" },
      { key: "status", label: "Status" },
      { key: "severity", label: "Severity" },
      { key: "message", label: "Message" },
    ],
    gateReport.gate_rows || [],
    "No gate information available.",
  );
}

function renderG2MDeepHDerivativePayload(payload = null) {
  const statusEl = document.getElementById("g2m-deeph-derivative-status");
  const summaryEl = document.getElementById("g2m-deeph-derivative-summary");
  const issuesEl = document.getElementById("g2m-deeph-derivative-issues");
  const comparisonEl = document.getElementById("g2m-deeph-derivative-comparison");
  const gateReportEl = document.getElementById("g2m-deeph-derivative-gate-report");
  if (!statusEl || !summaryEl || !issuesEl || !comparisonEl || !gateReportEl) return;
  statusEl.className = "result-list";
  statusEl.textContent = "";
  summaryEl.textContent = "";
  comparisonEl.textContent = "";
  issuesEl.textContent = "";
  gateReportEl.textContent = "";
  if (!payload) {
    statusEl.className = "result-list muted-text";
    statusEl.textContent = "Derivative diagnostics not loaded yet.";
    renderG2MDeepHDerivativeGateReport({});
    renderG2MDeepHDerivativePlotsPayload({});
    renderG2MDeepHDerivativeArtifacts({});
    return;
  }
  const banner = document.createElement("div");
  banner.className = "comparison-status-banner diagnostic";
  banner.textContent = payload.not_computed
    ? (payload.message || G2M_DEEPH_DERIVATIVE_OPTIONAL_POSTPROCESSING)
    : `${g2mDeephDerivativeStatusText(payload)} diagnostic-only / no winner claim.`;
  statusEl.appendChild(banner);
  appendKeyValue(statusEl, "Panel", payload.title || G2M_DEEPH_DERIVATIVE_TITLE);
  appendKeyValue(statusEl, "Reference", payload.reference_label || G2M_DEEPH_DERIVATIVE_REFERENCE);
  appendKeyValue(statusEl, "Force constants", payload.force_constants_label || G2M_DEEPH_DERIVATIVE_FORCE_CONSTANTS);
  appendKeyValue(statusEl, "Default status", payload.default_status_text || G2M_DEEPH_DERIVATIVE_DEFAULT_STATUS);
  appendKeyValue(statusEl, "Post-processing", G2M_DEEPH_DERIVATIVE_OPTIONAL_POSTPROCESSING);
  appendKeyValue(statusEl, "Claim scope", payload.message || G2M_DEEPH_DERIVATIVE_INTERNAL_ONLY);
  appendKeyValue(statusEl, "Run", payload.run_id || "-");
  appendKeyValue(statusEl, "finite difference method", (payload.status_rows || []).map((row) => row.finite_difference_method).filter(Boolean).join(" | ") || "-");
  appendKeyValue(statusEl, "delta_ang", (payload.status_rows || []).map((row) => row.delta_ang).filter(Boolean).join(" | ") || "-");
  appendKeyValue(statusEl, "derivative_units", (payload.status_rows || []).map((row) => row.derivative_units).filter(Boolean).join(" | ") || "-");
  if ((payload.prominent_issue_rows || []).length) {
    const prominent = document.createElement("div");
    prominent.className = "comparison-status-banner diagnostic";
    prominent.textContent = "Derivative metadata/order diagnostics need attention before any stronger interpretation.";
    statusEl.appendChild(prominent);
  }
  const issueRows = [];
  const seenIssueKeys = new Set();
  for (const row of [...(payload.prominent_issue_rows || []), ...(payload.issue_rows || [])]) {
    const key = [row.model || "", row.code || "", row.sample || "", row.message || ""].join("|");
    if (seenIssueKeys.has(key)) continue;
    seenIssueKeys.add(key);
    issueRows.push(row);
  }
  appendG2MDeepHTable(
    summaryEl,
    "Derivative status summary",
    [
      { key: "method", label: "Method" },
      { key: "scientific_status", label: "Status" },
      { key: "finite_difference_method", label: "finite difference method" },
      { key: "delta_ang", label: "delta_ang (Ang)" },
      { key: "derivative_units", label: "derivative_units" },
      { key: "stencils_ok", label: "stencils ok", format: g2mDeephIntegerValue },
      { key: "stencils_failed", label: "stencils failed", format: g2mDeephIntegerValue },
    ],
    payload.status_rows || [],
    payload.not_computed ? (payload.message || G2M_DEEPH_DERIVATIVE_OPTIONAL_POSTPROCESSING) : "No derivative status rows available.",
  );
  appendG2MDeepHTable(
    comparisonEl,
    "Model comparison table Graph2Mat vs DeepH",
    [
      { key: "method", label: "Method" },
      { key: "dh_mae_union_eV_per_Ang", label: "dH MAE (eV/Ang)" },
      { key: "dh_rmse_union_eV_per_Ang", label: "dH RMSE (eV/Ang)" },
      { key: "dh_relative_frobenius_ref", label: "Relative Frobenius" },
      { key: "dH_pred_hermiticity_defect", label: "Predicted hermiticity defect" },
      { key: "dH_hermiticity_error_delta", label: "Hermiticity error delta" },
    ],
    payload.comparison_rows || [],
    payload.not_computed ? (payload.message || G2M_DEEPH_DERIVATIVE_OPTIONAL_POSTPROCESSING) : "No derivative model-comparison rows available.",
  );
  appendG2MDeepHTable(
    issuesEl,
    "Warning/fatal error table",
    [
      { key: "model", label: "Method" },
      { key: "severity", label: "Severity" },
      { key: "code", label: "Code" },
      { key: "sample", label: "Sample" },
      { key: "message", label: "Message" },
    ],
    issueRows,
    payload.not_computed ? (payload.message || G2M_DEEPH_DERIVATIVE_OPTIONAL_POSTPROCESSING) : "No derivative warnings or fatal errors.",
  );
  renderG2MDeepHDerivativeGateReport(payload);
  renderG2MDeepHDerivativePlotsPayload(payload);
  renderG2MDeepHDerivativeArtifacts(payload);
}

async function loadG2MDeepHDerivativeMetrics({ runId = null } = {}) {
  if (!state.g2mDeephPlotRuns?.length) {
    await loadG2MDeepHPlotRuns({ preserveSelection: true });
  }
  renderG2MDeepHDerivativeRunSelector();
  let selectedRunIds = runId ? [runId] : (state.g2mDeephDerivativeRunIds || []);
  if (!selectedRunIds.length) selectedRunIds = [preferredG2MDeepHDerivativeRunId()].filter(Boolean);
  state.g2mDeephDerivativeRunIds = selectedRunIds;
  state.g2mDeephDerivativeRunId = selectedRunIds[0] || null;
  const params = new URLSearchParams();
  selectedRunIds.forEach((id) => params.append("run_id", id));
  const query = params.toString() ? `?${params.toString()}` : "";
  let payload = await request(`/api/g2m-deeph/derivative-metrics${query}`);
  if (!runId && selectedRunIds.length <= 1 && !g2mDeephDerivativeHasMaePlot(payload)) {
    const current = selectedRunIds[0] || "";
    for (const run of state.g2mDeephPlotRuns || []) {
      const candidate = run.run_id || run.id;
      if (!candidate || candidate === current) continue;
      const fallback = await request(`/api/g2m-deeph/derivative-metrics?run_id=${encodeURIComponent(candidate)}`);
      if (!g2mDeephDerivativeHasMaePlot(fallback)) continue;
      selectedRunIds = [candidate];
      state.g2mDeephDerivativeRunIds = selectedRunIds;
      state.g2mDeephDerivativeRunId = candidate;
      payload = fallback;
      break;
    }
  }
  state.g2mDeephDerivativePayload = payload;
  renderG2MDeepHDerivativeRunSelector();
  renderG2MDeepHDerivativePayload(payload);
  return payload;
}

const DATASET_MINIMUM_CRITERIA = {
  N_min_abs: { label: "N_min_abs", dash: "dash", width: 1.8 },
  N_min_rel_tol: { label: "N_min_rel_tol", dash: "dot", width: 1.2 },
  N_min_plateau: { label: "N_min_plateau", dash: "dashdot", width: 1.2 },
  N_min_cost_eff: { label: "N_min_cost_eff", dash: "longdash", width: 1.2 },
};

const DATASET_MINIMUM_LEGACY_CRITERIA = {
  N_min_rel95: "N_min_rel_tol",
};

function datasetMinimumCanonicalCriterion(criterion) {
  return DATASET_MINIMUM_LEGACY_CRITERIA[criterion] || criterion;
}

function datasetMinimumCriterionValue(container, criterion) {
  const canonical = datasetMinimumCanonicalCriterion(criterion);
  if (!container) return undefined;
  if (container[canonical] !== undefined) return container[canonical];
  const legacy = Object.entries(DATASET_MINIMUM_LEGACY_CRITERIA)
    .find(([_legacy, target]) => target === canonical)?.[0];
  return legacy ? container[legacy] : undefined;
}

function datasetMinimumSelectedCriterion() {
  return datasetMinimumControlValue("dataset-minimum-criterion", "all");
}

function datasetMinimumCriteriaToPlot() {
  const selected = datasetMinimumSelectedCriterion();
  if (selected === "all") return Object.keys(DATASET_MINIMUM_CRITERIA);
  const canonical = datasetMinimumCanonicalCriterion(selected);
  return DATASET_MINIMUM_CRITERIA[canonical] ? [canonical] : Object.keys(DATASET_MINIMUM_CRITERIA);
}

function datasetMinimumSelectedRunRoots() {
  const domSelected = Array.from(document.querySelectorAll(".dataset-minimum-run-root:checked"))
    .map((node) => String(node.value || "").trim())
    .filter(Boolean);
  if (domSelected.length) {
    setDatasetMinimumRunRootSelection(domSelected);
    return domSelected;
  }
  if (Array.isArray(state.datasetMinimumRunRootSelection)) {
    return state.datasetMinimumRunRootSelection
      .map((value) => String(value || "").trim())
      .filter(Boolean);
  }
  return [];
}

function setDatasetMinimumRunRootSelection(runRoots) {
  state.datasetMinimumRunRootSelection = Array.from(new Set(
    (runRoots || []).map((value) => String(value || "").trim()).filter(Boolean),
  ));
}

function syncDatasetMinimumRunRootSelectionFromDom() {
  const selected = Array.from(document.querySelectorAll(".dataset-minimum-run-root:checked"))
    .map((node) => String(node.value || "").trim())
    .filter(Boolean);
  setDatasetMinimumRunRootSelection(selected);
  return selected;
}

function populateDatasetMinimumThresholdPresets(payload = {}) {
  const select = document.getElementById("dataset-minimum-threshold-preset");
  const input = document.getElementById("dataset-minimum-threshold");
  const warning = document.getElementById("dataset-minimum-threshold-warning");
  if (!select) return;
  const currentMetric = datasetMinimumSelectedMetric();
  const currentThreshold = datasetMinimumSelectedThreshold();
  const presets = DATASET_MINIMUM_THRESHOLD_PRESETS[currentMetric] || [];
  const activePreset = presets.find((item) => item.key === state.datasetMinimumThresholdPresetKey) || presets[0] || null;
  select.innerHTML = "";
  for (const preset of presets) {
    const option = document.createElement("option");
    option.value = String(preset.key);
    option.textContent = preset.label;
    if (!state.datasetMinimumThresholdUserDefined && activePreset && preset.key === activePreset.key) {
      option.selected = true;
    }
    select.appendChild(option);
  }
  if (state.datasetMinimumThresholdUserDefined) {
    const manual = document.createElement("option");
    manual.value = DATASET_MINIMUM_THRESHOLD_MANUAL_KEY;
    manual.textContent = "Manual exploratory threshold";
    manual.selected = true;
    select.appendChild(manual);
  }
  if (input) {
    if (state.datasetMinimumThresholdUserDefined) {
      if (currentThreshold == null && activePreset) input.value = String(activePreset.threshold_mev);
    } else if (activePreset) {
      input.value = String(activePreset.threshold_mev);
      state.datasetMinimumThresholdPresetKey = activePreset.key;
    }
  }
  if (warning) {
    warning.textContent = state.datasetMinimumThresholdUserDefined
      ? "Manual threshold: marked as user_defined_exploratory. Visible warning: this blocks paper-level use unless separately justified."
      : `Preset threshold for ${DATASET_MINIMUM_METRIC_LABELS[currentMetric] || currentMetric}. 20 meV is not universal; presets are metric-specific and exploratory unless separately justified.`;
  }
}

function syncDatasetMinimumThresholdFromPreset() {
  const select = document.getElementById("dataset-minimum-threshold-preset");
  const input = document.getElementById("dataset-minimum-threshold");
  if (!select || !input) return;
  const currentMetric = datasetMinimumSelectedMetric();
  const preset = (DATASET_MINIMUM_THRESHOLD_PRESETS[currentMetric] || []).find((item) => item.key === select.value);
  if (!preset) return;
  state.datasetMinimumThresholdUserDefined = false;
  state.datasetMinimumThresholdPresetKey = preset.key;
  input.value = String(preset.threshold_mev);
  populateDatasetMinimumThresholdPresets();
}

function syncDatasetMinimumThresholdPresetFromInput() {
  const select = document.getElementById("dataset-minimum-threshold-preset");
  const input = document.getElementById("dataset-minimum-threshold");
  if (!select || !input) return;
  const value = finiteNumber(input.value);
  if (value == null) return;
  state.datasetMinimumThresholdUserDefined = true;
  state.datasetMinimumThresholdPresetKey = DATASET_MINIMUM_THRESHOLD_MANUAL_KEY;
  populateDatasetMinimumThresholdPresets();
}

function renderDatasetMinimumRunSources(payload = {}) {
  const container = document.getElementById("dataset-minimum-run-sources");
  if (!container) return;
  container.textContent = "";
  const sources = payload.run_root_sources || [];
  if (!sources.length) {
    container.textContent = payload?.run_root_discovery_error
      || "No se detectaron sweeps terminados con summary/ranking/normalized_run_metrics.json.";
    return;
  }
  const selectableRoots = sources
    .filter((source) => source.selectable)
    .map((source) => String(source.run_root || ""));
  const previousSelection = datasetMinimumSelectedRunRoots()
    .filter((runRoot) => selectableRoots.includes(runRoot));
  const defaultRoots = (payload.default_run_roots || [])
    .map((value) => String(value || ""))
    .filter((runRoot) => selectableRoots.includes(runRoot));
  const selectedRoots = new Set(
    previousSelection.length
      ? previousSelection
      : defaultRoots.length
        ? defaultRoots
        : selectableRoots.slice(0, 1),
  );
  setDatasetMinimumRunRootSelection(Array.from(selectedRoots));
  for (const source of sources) {
    const row = document.createElement("label");
    row.className = `dataset-minimum-run-source${source.selectable ? "" : " dataset-minimum-run-source-blocked"}`;
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.className = "dataset-minimum-run-root";
    checkbox.value = String(source.run_root || "");
    checkbox.disabled = !source.selectable;
    checkbox.checked = source.selectable && selectedRoots.has(String(source.run_root || ""));
    checkbox.addEventListener("change", () => {
      syncDatasetMinimumRunRootSelectionFromDom();
      state.datasetMinimumPreviewCache = null;
      datasetMinimumInvalidatePreferredOutputIfStale();
      refreshDatasetMinimumView();
    });
    const meta = document.createElement("div");
    const sizes = Array.isArray(source.dataset_sizes) ? source.dataset_sizes.join(", ") : "-";
    const blocked = source.blocked_reason
      ? `<div class="dataset-minimum-run-source-meta">Bloqueado: ${escapeHtml(String(source.blocked_reason))}</div>`
      : "";
    meta.innerHTML = `
      <strong>${escapeHtml(String(source.label || source.run_root || "sweep"))}</strong>
      <div class="dataset-minimum-run-source-meta">${escapeHtml(String(source.run_root || ""))}</div>
      <div class="dataset-minimum-run-source-meta">${source.metric_rows || 0} filas · N: ${escapeHtml(sizes || "-")}</div>
      ${blocked}
    `;
    row.appendChild(checkbox);
    row.appendChild(meta);
    container.appendChild(row);
  }
}

async function runDatasetMinimumAnalysis() {
  const runRoots = datasetMinimumSelectedRunRoots();
  if (!runRoots.length) {
    showToast("Selecciona al menos un sweep terminado.");
    return;
  }
  const button = document.getElementById("g2m-deeph-dataset-minimum-run");
  if (button) button.disabled = true;
  try {
    const selectedFit = datasetMinimumSelectedFit();

    const payload = {
      run_roots: runRoots,
      primary_metric: datasetMinimumSelectedMetric(),
      threshold_mev: datasetMinimumSelectedThreshold(),
      threshold_preset_key: datasetMinimumSelectedThresholdPresetKey(),
      threshold_is_user_defined: datasetMinimumThresholdIsUserDefined(),
      x_axis: datasetMinimumSelectedXAxis(),
      fit_models: [
        "linear",
        "quadratic",
        "inverse",
        "inverse_square",
        "power_law_floor",
        "power_law",
        "lowess_logx",
        "lowess_logx_robust",
        "monotone_lowess_logx",
        "moving_average",
        "cumulative_best",
        "none",
      ].join(","),
      n_min_source: datasetMinimumSelectedNMinSource
        ? datasetMinimumSelectedNMinSource()
        : "fit",
      n_min_fit_model: datasetMinimumBackendFitModel(selectedFit),
      moving_average_window: datasetMinimumSelectedMovingAverageWindow(),
      aggregation_mode: datasetMinimumSelectedAggregationMode(),
      cost_basis: datasetMinimumSelectedCostBasis(),
      claim_mode: datasetMinimumSelectedClaimMode(),
      threshold_protocol_file: datasetMinimumSelectedThresholdProtocolFile(),
      bootstrap_replicates: datasetMinimumSelectedBootstrapReplicates(),
      bootstrap_seed: 12345,
      ci_level: datasetMinimumSelectedCiLevel(),
    };
    
    const response = await request("/api/g2m-deeph/dataset-size-minimum/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    
    state.datasetMinimumPreferredOutputDir = datasetMinimumNormalizePath(
      response.output_dir || response.summary?.output_dir || "",
    );
    
    if (response.status === "no_usable_metric_rows") {
      const warnings = response.summary?.warnings || response.warnings || [];
      const detail = warnings.length ? ` ${warnings.slice(0, 3).join("; ")}` : "";
      showToast(`No usable metric rows for this selection.${detail}`);
    } else {
      showToast("Dataset-size-minimum analysis completed");
    }
    
    await loadDatasetMinimum();


  } catch (error) {
    showToast(error.message || String(error));
  } finally {
    if (button) button.disabled = false;
  }
}

const DATASET_MINIMUM_METRIC_LABELS = {
  h_mae_eV_mean: "H-MAE",
  h_rmse_eV: "H-RMSE",
  low_energy_rmse_eV: "Low-energy RMSE",
  fermi_window_rmse_eV: "Fermi RMSE",
};

const DATASET_MINIMUM_THRESHOLD_PRESETS = {
  h_mae_eV_mean: [
    {
      key: "h_mae_relaxed_10",
      threshold_mev: 10,
      label: "H-MAE exploratory 10 meV",
      reference: "DATASET_SIZE_MINIMUM.md#metric-specific-threshold-presets",
    },
    {
      key: "h_mae_relaxed_20",
      threshold_mev: 20,
      label: "H-MAE exploratory 20 meV",
      reference: "DATASET_SIZE_MINIMUM.md#metric-specific-threshold-presets",
    },
  ],
  h_rmse_eV: [
    {
      key: "h_rmse_relaxed_15",
      threshold_mev: 15,
      label: "H-RMSE exploratory 15 meV",
      reference: "DATASET_SIZE_MINIMUM.md#metric-specific-threshold-presets",
    },
    {
      key: "h_rmse_relaxed_25",
      threshold_mev: 25,
      label: "H-RMSE exploratory 25 meV",
      reference: "DATASET_SIZE_MINIMUM.md#metric-specific-threshold-presets",
    },
  ],
  low_energy_rmse_eV: [
    {
      key: "low_energy_rmse_exploratory_20",
      threshold_mev: 20,
      label: "Low-energy RMSE exploratory 20 meV",
      reference: "DATASET_SIZE_MINIMUM.md#metric-specific-threshold-presets",
    },
    {
      key: "low_energy_rmse_exploratory_40",
      threshold_mev: 40,
      label: "Low-energy RMSE exploratory 40 meV",
      reference: "DATASET_SIZE_MINIMUM.md#metric-specific-threshold-presets",
    },
  ],
  fermi_window_rmse_eV: [
    {
      key: "fermi_window_rmse_exploratory_15",
      threshold_mev: 15,
      label: "Fermi RMSE exploratory 15 meV",
      reference: "DATASET_SIZE_MINIMUM.md#metric-specific-threshold-presets",
    },
    {
      key: "fermi_window_rmse_exploratory_30",
      threshold_mev: 30,
      label: "Fermi RMSE exploratory 30 meV",
      reference: "DATASET_SIZE_MINIMUM.md#metric-specific-threshold-presets",
    },
  ],
};

const DATASET_MINIMUM_THRESHOLD_MANUAL_KEY = "manual";

const DATASET_MINIMUM_METHOD_COLORS = {
  deeph: "#d62728",
  graph2mat: "#1f77b4",
};

function datasetMinimumCanonicalFitModel(fitKind) {
  if (fitKind === "power_law" || fitKind === "power_law_floor") return "power_law_floor";
  return String(fitKind || "power_law_floor");
}

function datasetMinimumFitModelsEquivalent(left, right) {
  return datasetMinimumCanonicalFitModel(left) === datasetMinimumCanonicalFitModel(right);
}

function datasetMinimumBackendFitModel(fitKind) {
  const allowed = new Set([
    "linear",
    "quadratic",
    "inverse",
    "inverse_square",
    "power_law",
    "power_law_floor",
    "lowess_logx",
    "lowess_logx_robust",
    "monotone_lowess_logx",
    "moving_average",
    "cumulative_best",
    "none",
  ]);

  if (allowed.has(fitKind)) {
    return datasetMinimumCanonicalFitModel(fitKind);
  }

  return "power_law_floor";
}


function updateDatasetMinimumMovingAverageVisibility() {
  const field = document.getElementById("dataset-minimum-moving-average-window")?.closest(".field");
  if (!field) return;

  const isMovingAverage = datasetMinimumSelectedFit() === "moving_average";
  field.hidden = !isMovingAverage;

  const input = document.getElementById("dataset-minimum-moving-average-window");
  if (input) input.disabled = !isMovingAverage;
}


function datasetMinimumControlValue(id, fallback = "") {
  const node = document.getElementById(id);
  return node ? node.value : fallback;
}

function datasetMinimumSelectedMetric() {
  return datasetMinimumControlValue("dataset-minimum-metric", "h_mae_eV_mean");
}

function datasetMinimumSelectedThreshold() {
  return finiteNumber(datasetMinimumControlValue("dataset-minimum-threshold", "10"));
}

function datasetMinimumSelectedThresholdPresetKey() {
  return state.datasetMinimumThresholdUserDefined
    ? DATASET_MINIMUM_THRESHOLD_MANUAL_KEY
    : (state.datasetMinimumThresholdPresetKey || datasetMinimumControlValue("dataset-minimum-threshold-preset", "h_mae_relaxed_10"));
}

function datasetMinimumThresholdIsUserDefined() {
  return Boolean(state.datasetMinimumThresholdUserDefined);
}

function datasetMinimumSelectedThresholdProtocolFile() {
  return datasetMinimumControlValue("dataset-minimum-threshold-protocol-file", "").trim();
}

function datasetMinimumSelectedXAxis() {
  return datasetMinimumControlValue("dataset-minimum-x-axis", "n_train");
}

function datasetMinimumSelectedCostBasis() {
  return datasetMinimumControlValue("dataset-minimum-cost-basis", "per_seed_mean");
}

function datasetMinimumSelectedClaimMode() {
  return datasetMinimumControlValue("dataset-minimum-claim-mode", "diagnostic");
}

function datasetMinimumSelectedFit() {
  return datasetMinimumControlValue("dataset-minimum-fit", "power_law_floor");
}

function datasetMinimumSelectedMovingAverageWindow() {
  const value = finiteNumber(
    datasetMinimumControlValue("dataset-minimum-moving-average-window", "3"),
  );
  if (value == null) return 3;
  return Math.max(1, Math.round(value));
}

function datasetMinimumOutputLabel(output = {}) {
  const threshold = finiteNumber(output.threshold_mev);
  const metric = DATASET_MINIMUM_METRIC_LABELS[output.primary_metric] || output.primary_metric || "metric";
  const axis = output.x_axis || "n_train";
  return `${metric} · ${threshold == null ? "threshold ?" : `${formatCompactNumber(threshold)} meV`} · ${axis}`;
}

function formatCompactNumber(value) {
  const number = finiteNumber(value);
  if (number == null) return "-";
  return Number.isInteger(number) ? String(number) : Number(number.toPrecision(5)).toString();
}

function datasetMinimumSelectedNMinSource() {
  return datasetMinimumControlValue("dataset-minimum-nmin-source", "fit");
}

function datasetMinimumSelectedAggregationMode() {
  return datasetMinimumControlValue("dataset-minimum-aggregation-mode", "mean_seeds_per_config");
}

function datasetMinimumAggregationModeLabel(mode) {
  if (mode === "mean_seeds_per_config") return "Mean seeds per config — paper-level";
  if (mode === "best_config_mean") return "Best config by seed mean — paper-level if config policy is locked";
  if (mode === "best_config") return "Best single run — diagnostic only";
  if (mode === "mean_replicates") return "Mean all replicates — diagnostic/mixed";
  return String(mode || "unknown");
}

function datasetMinimumAggregationModeClassification(mode) {
  if (mode === "mean_seeds_per_config") {
    return {
      classification: "paper_candidate",
      reason: "paper_ready_seed_mean_per_config",
    };
  }
  if (mode === "best_config_mean") {
    return {
      classification: "paper_candidate",
      reason: "paper_candidate_only_if_config_selection_policy_is_locked",
    };
  }
  if (mode === "best_config") {
    return {
      classification: "diagnostic_only",
      reason: "best_single_run_is_not_a_paper_level_protocol",
    };
  }
  if (mode === "mean_replicates") {
    return {
      classification: "diagnostic_only",
      reason: "replicate_mean_mixes_configs_or_seeds_without_locked_paper_protocol",
    };
  }
  return {
    classification: "diagnostic_only",
    reason: `unknown_aggregation_mode:${mode || "unknown"}`,
  };
}

function datasetMinimumUsesBackendAggregationMode(mode) {
  return mode === "mean_replicates"
    || mode === "mean_seeds_per_config"
    || mode === "best_config_mean";
}

function datasetMinimumAggregationSeriesSuffix(mode) {
  if (mode === "best_config") return "best config";
  if (mode === "best_config_mean") return "best config (seed mean)";
  if (mode === "mean_seeds_per_config") return "seed mean";
  return "mean";
}

function datasetMinimumSelectedBootstrapReplicates() {
  const value = finiteNumber(datasetMinimumControlValue("dataset-minimum-bootstrap-replicates", "0"));
  return value == null ? 0 : Math.max(0, Math.round(value));
}

function datasetMinimumSelectedCiLevel() {
  const value = finiteNumber(datasetMinimumControlValue("dataset-minimum-ci-level", "0.95"));
  return value == null ? 0.95 : value;
}

function datasetMinimumShowRawReplicates() {
  return Boolean(document.getElementById("dataset-minimum-show-raw-replicates")?.checked);
}

function datasetMinimumOutputBootstrapReplicates(output = {}) {
  const value = finiteNumber(output.bootstrap_replicates);
  return value == null ? 0 : Math.round(value);
}

function datasetMinimumOutputCiLevel(output = {}) {
  const value = finiteNumber(output.ci_level);
  return value == null ? 0.95 : value;
}

function datasetMinimumEffectiveAggregationMode(output = {}) {
  if (output.aggregation_mode) return String(output.aggregation_mode);
  const roots = datasetMinimumNormalizeRoots(output.run_roots || []);
  return roots.length > 1 ? "mean_replicates" : "best_config";
}

function datasetMinimumOutputNMinFitModel(output = {}) {
  return datasetMinimumCanonicalFitModel(
    output.actual_fit_model || output.requested_fit_model || output.n_min_fit_model || "power_law_floor",
  );
}

function datasetMinimumOutputMovingAverageWindow(output = {}) {
  const value = finiteNumber(output.moving_average_window);
  return value == null ? 3 : Math.max(1, Math.round(value));
}

function datasetMinimumOutputCostBasis(output = {}) {
  return String(output.cost_basis || "per_seed_mean");
}

function datasetMinimumCostBasisLabel(costBasis) {
  return costBasis === "protocol_total"
    ? "protocol total GPU-hours"
    : "per-seed mean GPU-hours";
}

function datasetMinimumRowCost(row = {}, costBasis = "per_seed_mean") {
  const keys = costBasis === "protocol_total"
    ? ["gpu_hours_protocol_total", "gpu_hours_total_sum", "gpu_hours_total_mean", "gpu_hours_total"]
    : ["gpu_hours_per_seed_mean", "gpu_hours_total_mean", "gpu_hours_total", "gpu_hours_protocol_total"];
  for (const key of keys) {
    const value = finiteNumber(row[key]);
    if (value != null) return value;
  }
  return null;
}

function datasetMinimumOutputClaimModeRequested(output = {}) {
  return String(output.claim_mode_requested || "diagnostic");
}

function datasetMinimumOutputClaimModeActual(output = {}) {
  return String(output.claim_mode_actual || "diagnostic");
}

function datasetMinimumOutputAggregationMetadata(output = {}) {
  const requested = output.requested_aggregation_mode;
  const actual = String(
    output.actual_aggregation_mode
    || output.aggregation_mode
    || (Array.isArray(output.run_roots) && output.run_roots.length > 1 ? "mean_replicates" : "best_config"),
  );
  const classification = output.aggregation_mode_classification
    || datasetMinimumAggregationModeClassification(actual).classification;
  const reason = output.aggregation_mode_classification_reason
    || datasetMinimumAggregationModeClassification(actual).reason;
  const legacyInferred = Boolean(
    output.aggregation_mode_legacy_inferred
    || (!output.actual_aggregation_mode && !output.requested_aggregation_mode && !output.aggregation_mode),
  );
  return {
    requested,
    actual,
    classification,
    reason,
    legacyInferred,
  };
}

function datasetMinimumOutputMatchesCurrentSelection(output = {}) {
  if (output.primary_metric !== datasetMinimumSelectedMetric()) return false;
  if ((output.x_axis || "n_train") !== datasetMinimumSelectedXAxis()) return false;

  const threshold = datasetMinimumSelectedThreshold();
  const outThreshold = finiteNumber(output.threshold_mev);
  if (
    threshold == null
    || outThreshold == null
    || Math.abs(threshold - outThreshold) >= 1e-9
  ) {
    return false;
  }
  if (Boolean(output.threshold_is_user_defined) !== datasetMinimumThresholdIsUserDefined()) {
    return false;
  }
  if (!datasetMinimumThresholdIsUserDefined()) {
    const outputPresetKey = String(output.threshold_preset_key || "");
    if (outputPresetKey !== String(datasetMinimumSelectedThresholdPresetKey() || "")) {
      return false;
    }
  }

  const requestedSource = String(
    output.requested_n_min_source || output.n_min_source || "observed",
  );
  if (requestedSource !== datasetMinimumSelectedNMinSource()) return false;

  const selectedFit = datasetMinimumBackendFitModel(datasetMinimumSelectedFit());
  const outputFit = output.requested_fit_model || output.n_min_fit_model || output.actual_fit_model;
  if (!datasetMinimumFitModelsEquivalent(outputFit, selectedFit)) return false;

  if (
    datasetMinimumEffectiveAggregationMode(output)
    !== datasetMinimumSelectedAggregationMode()
  ) {
    return false;
  }

  if (datasetMinimumOutputCostBasis(output) !== datasetMinimumSelectedCostBasis()) {
    return false;
  }
  if (datasetMinimumOutputClaimModeRequested(output) !== datasetMinimumSelectedClaimMode()) {
    return false;
  }
  const outputThresholdProtocolFile = datasetMinimumNormalizePath(output.threshold_protocol_file || "");
  const selectedThresholdProtocolFile = datasetMinimumNormalizePath(
    datasetMinimumSelectedThresholdProtocolFile(),
  );
  if (outputThresholdProtocolFile !== selectedThresholdProtocolFile) {
    return false;
  }

  if (selectedFit === "moving_average") {
    if (
      datasetMinimumOutputMovingAverageWindow(output)
      !== datasetMinimumSelectedMovingAverageWindow()
    ) {
      return false;
    }
  }

  if (
    datasetMinimumOutputBootstrapReplicates(output)
    !== datasetMinimumSelectedBootstrapReplicates()
  ) {
    return false;
  }

  if (
    Math.abs(datasetMinimumOutputCiLevel(output) - datasetMinimumSelectedCiLevel()) > 1e-9
  ) {
    return false;
  }

  return datasetMinimumOutputMatchesSelection(output);
}

function datasetMinimumInvalidatePreferredOutputIfStale() {
  const preferred = datasetMinimumNormalizePath(state.datasetMinimumPreferredOutputDir || "");
  if (!preferred) return;

  const payload = state.datasetMinimumPayload || {};
  const output = (payload.outputs || []).find(
    (item) => datasetMinimumNormalizePath(item.output_dir) === preferred,
  );
  if (!output || !datasetMinimumOutputMatchesCurrentSelection(output)) {
    state.datasetMinimumPreferredOutputDir = null;
  }
}

function datasetMinimumNormalizeRoots(roots = []) {
  return Array.from(
    new Set(
      (roots || [])
        .map(datasetMinimumNormalizePath)
        .filter(Boolean),
    ),
  ).sort();
}

function datasetMinimumRootKey(roots = []) {
  return datasetMinimumNormalizeRoots(roots).join("\n");
}

function datasetMinimumOutputMatchesSelection(output = {}) {
  const selectedRoots = datasetMinimumNormalizeRoots(datasetMinimumSelectedRunRoots());

  // Sin seleccion explicita, no filtrar por roots.
  if (!selectedRoots.length) return true;

  const outputRoots = datasetMinimumNormalizeRoots(output.run_roots || []);
  if (!outputRoots.length) return false;

  return datasetMinimumRootKey(outputRoots) === datasetMinimumRootKey(selectedRoots);
}

function datasetMinimumNormalizePath(value) {
  return String(value || "")
    .trim()
    .replace(/\\/g, "/")
    .replace(/\/+$/, "");
}

function datasetMinimumFindOutput(payload = {}) {
  const preferredOutputDir = datasetMinimumNormalizePath(
    state.datasetMinimumPreferredOutputDir || "",
  );

  const compatible = (payload.outputs || [])
    .filter((output) => ["ok", "no_usable_metric_rows"].includes(String(output.status || "")))
    .filter(datasetMinimumOutputMatchesCurrentSelection);

  if (!compatible.length) return null;

  // 1. Prioridad maxima: el output recien generado por Run analysis.
  if (preferredOutputDir) {
    const preferred = compatible.find(
      (output) => datasetMinimumNormalizePath(output.output_dir) === preferredOutputDir,
    );
    if (preferred) return preferred;
  }

  // 2. Coincidencia exacta mas reciente para la seleccion actual.
  return compatible.sort((left, right) => {
    const leftTime = finiteNumber(left.modified_at) ?? 0;
    const rightTime = finiteNumber(right.modified_at) ?? 0;
    return rightTime - leftTime;
  })[0];
}

function datasetMinimumAxisField(axis) {
  return axis === "n_total" ? "dataset_size_total" : "dataset_size_train";
}

function datasetMinimumAxisLabel(axis) {
  return axis === "n_total" ? "N_total snapshots" : "N_train snapshots";
}

function datasetMinimumRowX(row = {}, axis = "n_train") {
  return finiteNumber(row[datasetMinimumAxisField(axis)] ?? row.dataset_size_x);
}

function datasetMinimumDisplayN(output, method, nValue, axis) {
  const value = finiteNumber(nValue);
  if (value == null) return null;
  if ((output.x_axis || "n_train") === axis) return value;
  const sourceField = datasetMinimumAxisField(output.x_axis || "n_train");
  const targetField = datasetMinimumAxisField(axis);
  const row = (output.best_rows || []).find(
    (item) =>
      String(item.method || "") === String(method || "") &&
      finiteNumber(item[sourceField]) != null &&
      Math.abs(finiteNumber(item[sourceField]) - value) < 1e-9,
  );
  return row ? finiteNumber(row[targetField]) : value;
}

function datasetMinimumFormatN(value) {
  const number = finiteNumber(value);
  return number == null ? "-" : String(Math.round(number));
}

function datasetMinimumFormatMeV(value) {
  const number = finiteNumber(value);
  return number == null ? "-" : `${number.toFixed(3)} meV`;
}

function datasetMinimumRowErrorBarDelta(row = {}) {
  const sem = finiteNumber(row.primary_metric_mev_sem);
  if (sem != null && sem > 0) return sem;
  const std = finiteNumber(row.primary_metric_mev_std);
  if (std != null && std > 0) return std;
  return null;
}

function datasetMinimumListField(value) {
  if (Array.isArray(value)) return value.filter(Boolean).join(", ") || "-";
  return value == null || value === "" ? "-" : String(value);
}

function datasetMinimumAggregatedRowHover(row = {}, axis = "n_train") {
  const lines = [
    `method: ${row.method || "-"}`,
    `${axis === "n_total" ? "N_total" : "N_train"}: ${row.x_value ?? "-"}`,
    `mean: ${row.y_value == null ? "-" : `${row.y_value.toFixed(3)} meV`}`,
  ];
  const std = finiteNumber(row.primary_metric_mev_std);
  const sem = finiteNumber(row.primary_metric_mev_sem);
  if (std != null) lines.push(`std: ${std.toFixed(3)} meV`);
  if (sem != null) lines.push(`sem: ${sem.toFixed(3)} meV`);
  if (sem == null && std != null) lines.push("error bar uses std (sem unavailable)");
  const replicateCount = row.replicate_count || row.row_count || row.source_count;
  if (replicateCount) lines.push(`replicate_count: ${replicateCount}`);
  lines.push(`seeds: ${datasetMinimumListField(row.seeds || row.seed)}`);
  lines.push(`config_ids: ${datasetMinimumListField(row.config_ids || row.config_id)}`);
  lines.push(`source_run_roots: ${datasetMinimumListField(row.source_run_roots || row.source_run_root)}`);
  if (row.y_min != null && row.y_max != null) {
    lines.push(`y_range: [${Number(row.y_min).toFixed(3)}, ${Number(row.y_max).toFixed(3)}] meV`);
  }
  return lines.join("<br>");
}

function datasetMinimumFormatNMinWithCi(output, method, criterion, axis) {
  const point = datasetMinimumDisplayN(
    output,
    method,
    datasetMinimumCriterionValue(output?.thresholds?.[method], criterion),
    axis,
  );
  const bootstrap = datasetMinimumReplicateBootstrap(output);
  const ci = bootstrap?.enabled
    ? datasetMinimumCriterionValue(bootstrap?.by_method?.[method], criterion)
    : null;
  const hasInterval = ci
    && ci.median != null
    && ci.lower != null
    && ci.upper != null
    && ci.n_bootstrap_successful >= 2;

  if (hasInterval) {
    const median = datasetMinimumFormatN(
      datasetMinimumDisplayN(output, method, ci.median, axis),
    );
    const lower = datasetMinimumFormatN(
      datasetMinimumDisplayN(output, method, ci.lower, axis),
    );
    const upper = datasetMinimumFormatN(
      datasetMinimumDisplayN(output, method, ci.upper, axis),
    );
    return `${median} [${lower}, ${upper}]`;
  }

  if (point != null) {
    return `${datasetMinimumFormatN(point)} <span class="muted-text">(no replicate-resampling CI)</span>`;
  }
  return "-";
}

function datasetMinimumRawReplicateTraces(output, axis) {
  if (!datasetMinimumShowRawReplicates() || output?.is_preview) return [];
  const rawRows = output?.normalized_rows || [];
  if (!rawRows.length) return [];

  const traces = [];
  const methods = Array.from(new Set(rawRows.map((row) => String(row.method || "unknown")))).sort();
  methods.forEach((method, index) => {
    const color = DATASET_MINIMUM_METHOD_COLORS[method] || plotColor(index);
    const methodRows = rawRows
      .filter((row) => String(row.method || "") === method)
      .map((row) => ({
        x: datasetMinimumRowX(row, axis),
        y: finiteNumber(row.primary_metric_mev_mean ?? row.primary_metric_mev),
      }))
      .filter((row) => row.x != null && row.y != null);
    if (!methodRows.length) return;
    traces.push({
      type: "scatter",
      mode: "markers",
      x: methodRows.map((row) => row.x),
      y: methodRows.map((row) => row.y),
      marker: {
        symbol: g2mDeephMarkerSymbol(method),
        size: 6,
        color,
        opacity: 0.28,
        line: { width: 0 },
      },
      name: `${methodDisplayLabel(method)} raw replicates`,
      hovertemplate: `${axis === "n_total" ? "N_total" : "N_train"}: %{x}<br>Error: %{y:.3f} meV<extra>raw replicate</extra>`,
      showlegend: false,
    });
  });
  return traces;
}

function datasetMinimumBootstrapNMinOverlays(output, axis) {
  const shapes = [];
  const annotations = [];
  const bootstrap = datasetMinimumReplicateBootstrap(output);
  if (!bootstrap?.enabled) return { shapes, annotations };

  const thresholds = output.thresholds || {};
  Object.keys(thresholds).sort().forEach((method, methodIndex) => {
    const color = DATASET_MINIMUM_METHOD_COLORS[method] || plotColor(methodIndex);
    for (const criterionKey of ["N_min_abs", "N_min_rel_tol", "N_min_plateau"]) {
      if (!datasetMinimumCriteriaToPlot().includes(criterionKey)) continue;
      const ci = datasetMinimumCriterionValue(bootstrap?.by_method?.[method], criterionKey);
      if (!ci || ci.lower == null || ci.upper == null) continue;

      const lower = finiteNumber(datasetMinimumDisplayN(output, method, ci.lower, axis));
      const upper = finiteNumber(datasetMinimumDisplayN(output, method, ci.upper, axis));
      const median = finiteNumber(datasetMinimumDisplayN(output, method, ci.median, axis));
      if (lower == null || upper == null) continue;

      const x0 = Math.min(lower, upper);
      const x1 = Math.max(lower, upper);
      shapes.push({
        type: "rect",
        xref: "x",
        yref: "paper",
        x0,
        x1,
        y0: 0,
        y1: 1,
        fillcolor: color,
        opacity: 0.08,
        line: { width: 0 },
      });
      shapes.push({
        type: "line",
        xref: "x",
        x0: x0,
        x1: x0,
        yref: "paper",
        y0: 0,
        y1: 1,
        line: { color, width: 1, dash: "dot" },
      });
      shapes.push({
        type: "line",
        xref: "x",
        x0: x1,
        x1: x1,
        yref: "paper",
        y0: 0,
        y1: 1,
        line: { color, width: 1, dash: "dot" },
      });
      if (median != null) {
        shapes.push({
          type: "line",
          xref: "x",
          x0: median,
          x1: median,
          yref: "paper",
          y0: 0,
          y1: 1,
          line: { color, width: 2.2, dash: "solid" },
        });
        annotations.push({
          text: `${methodDisplayLabel(method)} ${criterionKey} replicate CI ${Math.round(median)} [${Math.round(x0)}, ${Math.round(x1)}]`,
          xref: "x",
          x: median,
          yref: "paper",
          y: 1.04 + methodIndex * 0.04,
          showarrow: false,
          font: { size: 10, color },
          bgcolor: "rgba(255,255,255,0.82)",
        });
      }
    }
  });
  return { shapes, annotations };
}

function renderDatasetMinimumTable(output, axis) {
  const container = document.getElementById("dataset-minimum-table");
  if (!container) return;
  container.textContent = "";
  const blockers = Array.isArray(output?.paper_level_blockers) ? output.paper_level_blockers.filter(Boolean) : [];
  const claimModeActual = datasetMinimumOutputClaimModeActual(output);
  const claimStatus = String(output?.scientific_claim_status || "diagnostic_only");
  const paperReadyNominal =
    claimModeActual === "paper_candidate" &&
    claimStatus === "paper_candidate_nominal_with_n_eff_diagnostic" &&
    !blockers.length;
  const statusBanner = document.createElement("div");
  statusBanner.className = "comparison-status-banner";
  statusBanner.classList.toggle("diagnostic", !paperReadyNominal);
  statusBanner.classList.toggle("invalid", !paperReadyNominal);
  statusBanner.textContent = paperReadyNominal
    ? "Paper-candidate nominal N_min under the audited protocol. Effective-N values remain diagnostic only and do not replace nominal N_min."
    : `Diagnostic only: paper-ready N_min blocked${blockers.length ? ` (${blockers.slice(0, 4).join("; ")})` : ""}. Effective-N values remain diagnostic only and N_min_effective_diagnostic does not replace nominal N_min.`;
  container.appendChild(statusBanner);

  const thresholds = output?.thresholds || {};
  const methods = Object.keys(thresholds).sort();
  if (!methods.length) {
    container.textContent = "No N_min threshold table available.";
    return;
  }
  const table = document.createElement("table");
  table.className = "summary-table";
  const axisLabel = axis === "n_total" ? "N_total" : "N_train";
  table.innerHTML = `
    <thead>
      <tr>
        <th>Threshold</th>
        <th>Method</th>
        <th>Best observed</th>
        <th>N_min_abs (${axisLabel})</th>
        <th>N_min_rel_tol (${axisLabel})</th>
        <th>N_min_plateau (${axisLabel})</th>
        <th>N_min_cost_eff (${axisLabel})</th>
      </tr>
    </thead>
    <tbody></tbody>
  `;
  const body = table.querySelector("tbody");
  const threshold = finiteNumber(output.threshold_mev);
  for (const method of methods) {
    const row = thresholds[method] || {};
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${threshold == null ? "-" : `${formatCompactNumber(threshold)} meV`}</td>
      <td>${escapeHtml(methodDisplayLabel(method))}</td>
      <td>${datasetMinimumFormatMeV(row.best_observed_mev)}</td>
      <td>${datasetMinimumFormatNMinWithCi(output, method, "N_min_abs", axis)}</td>
      <td>${datasetMinimumFormatNMinWithCi(output, method, "N_min_rel_tol", axis)}</td>
      <td>${datasetMinimumFormatNMinWithCi(output, method, "N_min_plateau", axis)}</td>
      <td>${datasetMinimumFormatN(datasetMinimumDisplayN(output, method, row.N_min_cost_eff, axis))} <span class="muted-text">(diagnostic observed cost-error behavior; no replicate-resampling CI)</span></td>
    `;
    body.appendChild(tr);
  }
  container.appendChild(table);

  const temporal = output?.temporal_diagnostics || {};
  const nEffBySize = temporal.N_eff_by_dataset_size || output?.N_eff_by_dataset_size || {};
  const ratioBySize = temporal.N_eff_over_N_by_dataset_size || output?.N_eff_over_N_by_dataset_size || {};
  const availabilityBySize = temporal.autocorrelation_available_by_dataset_size || output?.autocorrelation_available_by_dataset_size || {};
  const blockBySize = temporal.temporal_block_diagnostics_by_dataset_size || output?.temporal_block_diagnostics_by_dataset_size || {};
  const sizeKeys = Array.from(new Set([
    ...Object.keys(nEffBySize),
    ...Object.keys(blockBySize),
  ])).sort((left, right) => Number(left) - Number(right));
  if (!sizeKeys.length) return;

  const temporalTitle = document.createElement("div");
  temporalTitle.className = "section-subtitle";
  temporalTitle.textContent = "N nominal vs N_eff (diagnostic only)";
  container.appendChild(temporalTitle);

  const temporalTable = document.createElement("table");
  temporalTable.className = "summary-table";
  temporalTable.innerHTML = `
    <thead>
      <tr>
        <th>Dataset size (nominal N_train)</th>
        <th>N_eff diagnostic</th>
        <th>N_eff/N_nominal</th>
        <th>Autocorrelation available</th>
        <th>Blocks / datasets</th>
      </tr>
    </thead>
    <tbody></tbody>
  `;
  const temporalBody = temporalTable.querySelector("tbody");
  sizeKeys.forEach((sizeKey) => {
    const diag = blockBySize[sizeKey] || {};
    const datasets = Array.isArray(diag.datasets) ? diag.datasets : [];
    const blockEntries = datasets.reduce(
      (sum, item) => sum + Object.keys(item?.block_diagnostics || {}).length,
      0,
    );
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${escapeHtml(sizeKey)}</td>
      <td>${datasetMinimumFormatN(nEffBySize[sizeKey])}</td>
      <td>${ratioBySize[sizeKey] == null ? "-" : formatCompactNumber(ratioBySize[sizeKey])}</td>
      <td>${availabilityBySize[sizeKey] ? "yes" : "no"}</td>
      <td>${escapeHtml(`${diag.n_datasets || 0} dataset(s), ${blockEntries} block entry(ies)`)}</td>
    `;
    temporalBody.appendChild(tr);
  });
  container.appendChild(temporalTable);
}

function datasetMinimumPowerLawFitLinePoints(points) {
  const fitPoints = aggregateFitPoints(points).filter((point) => point.x > 0 && point.y != null);
  if (fitPoints.length < 3) return [];
  let best = null;
  for (let index = 0; index < 160; index += 1) {
    const alpha = 0.05 + ((4.0 - 0.05) * index) / 159;
    const transformed = fitPoints.map((point) => ({ x: point.x ** (-alpha), y: point.y }));
    const coefficients = polynomialCoefficients(transformed, 1);
    if (!coefficients) continue;
    const sse = transformed.reduce((sum, point) => {
      const predicted = evaluatePolynomial(coefficients, point.x);
      return sum + ((point.y - predicted) ** 2);
    }, 0);
    if (!best || sse < best.sse) best = { alpha, coefficients, sse };
  }
  if (!best) return [];
  const xValues = fitPoints.map((point) => point.x);
  const minX = Math.min(...xValues);
  const maxX = Math.max(...xValues);
  if (!Number.isFinite(minX) || !Number.isFinite(maxX)) return [];
  const lineX = minX === maxX
    ? [minX]
    : Array.from({ length: 100 }, (_, index) => minX + ((maxX - minX) * index) / 99);
  return lineX.map((x) => ({
    x,
    y: best.coefficients[0] + best.coefficients[1] * (x ** (-best.alpha)),
  }));
}

function tricubeWeight(u) {
  const value = Math.max(0, Math.min(1, Math.abs(u)));
  return (1 - value ** 3) ** 3;
}

function median(values) {
  const clean = values
    .filter((value) => Number.isFinite(value))
    .slice()
    .sort((a, b) => a - b);
  if (!clean.length) return null;
  const mid = Math.floor(clean.length / 2);
  return clean.length % 2 ? clean[mid] : 0.5 * (clean[mid - 1] + clean[mid]);
}

function weightedLocalLinearEstimate(points, x0, robustWeights, frac = 0.45) {
  const distances = points
    .map((point) => Math.abs(point.tx - x0))
    .sort((a, b) => a - b);

  const k = Math.max(2, Math.ceil(frac * points.length));
  const bandwidth = distances[Math.min(k - 1, distances.length - 1)] || distances[distances.length - 1] || 1;

  let sw = 0;
  let swx = 0;
  let swy = 0;
  let swxx = 0;
  let swxy = 0;

  points.forEach((point, index) => {
    const distance = Math.abs(point.tx - x0);
    const localWeight = bandwidth > 0 ? tricubeWeight(distance / bandwidth) : 1;
    const robustWeight = robustWeights?.[index] ?? 1;
    const weight = localWeight * robustWeight;

    if (!Number.isFinite(weight) || weight <= 0) return;

    const dx = point.tx - x0;
    sw += weight;
    swx += weight * dx;
    swy += weight * point.y;
    swxx += weight * dx * dx;
    swxy += weight * dx * point.y;
  });

  if (sw <= 0) return null;

  const denom = sw * swxx - swx * swx;
  if (Math.abs(denom) < 1e-14) {
    return swy / sw;
  }

  const slope = (sw * swxy - swx * swy) / denom;
  const intercept = (swy - slope * swx) / sw;
  return intercept;
}


function datasetMinimumPowerLawFloorFitLinePoints(points) {
  const clean = points
    .map((point) => ({
      x: finiteNumber(point.x),
      y: finiteNumber(point.y),
    }))
    .filter((point) => point.x != null && point.x > 0 && point.y != null && point.y >= 0)
    .sort((a, b) => a.x - b.x);

  if (clean.length < 4) return [];

  let best = null;

  for (let alphaIndex = 0; alphaIndex < 160; alphaIndex += 1) {
    const alpha = 0.05 + ((3.5 - 0.05) * alphaIndex) / 159;
    const transformed = clean.map((point) => ({
      x: point.x ** (-alpha),
      y: point.y,
    }));

    const coeffs = polynomialCoefficients(transformed, 1);
    if (!coeffs) continue;

    const c = coeffs[0];
    const a = coeffs[1];

    if (!Number.isFinite(c) || !Number.isFinite(a) || c < 0 || a < 0) continue;

    const sse = clean.reduce((sum, point) => {
      const predicted = c + a * point.x ** (-alpha);
      return sum + (point.y - predicted) ** 2;
    }, 0);

    if (!best || sse < best.sse) {
      best = { alpha, c, a, sse };
    }
  }

  if (!best) return [];

  const minX = clean[0].x;
  const maxX = clean[clean.length - 1].x;
  return Array.from({ length: 160 }, (_, index) => {
    const x = Math.exp(Math.log(minX) + ((Math.log(maxX) - Math.log(minX)) * index) / 159);
    return {
      x,
      y: best.c + best.a * x ** (-best.alpha),
    };
  });
}


function datasetMinimumLowessLinePoints(points, options = {}) {
  const frac = options.frac ?? 0.45;
  const robustIterations = options.robustIterations ?? 2;
  const monotone = Boolean(options.monotone);

  const clean = points
    .map((point) => ({
      x: finiteNumber(point.x),
      y: finiteNumber(point.y),
    }))
    .filter((point) => point.x != null && point.x > 0 && point.y != null)
    .sort((a, b) => a.x - b.x)
    .map((point) => ({
      ...point,
      tx: Math.log(point.x),
    }));

  if (clean.length < 3) return [];

  let robustWeights = clean.map(() => 1);

  for (let iteration = 0; iteration < robustIterations; iteration += 1) {
    const fitted = clean.map((point) =>
      weightedLocalLinearEstimate(clean, point.tx, robustWeights, frac),
    );

    const residuals = clean.map((point, index) => {
      const fit = fitted[index];
      return fit == null ? null : Math.abs(point.y - fit);
    });

    const mad = median(residuals);
    if (mad == null || mad <= 1e-12) break;

    robustWeights = residuals.map((residual) => {
      if (residual == null) return 0;
      const u = residual / (6 * mad);
      if (u >= 1) return 0;
      return (1 - u * u) ** 2;
    });
  }

  const txMin = clean[0].tx;
  const txMax = clean[clean.length - 1].tx;
  const gridSize = 160;

  let line = Array.from({ length: gridSize }, (_, index) => {
    const tx = txMin + ((txMax - txMin) * index) / (gridSize - 1);
    const y = weightedLocalLinearEstimate(clean, tx, robustWeights, frac);
    return {
      x: Math.exp(tx),
      y,
    };
  }).filter((point) => point.y != null && Number.isFinite(point.y));

  if (monotone) {
    let best = Number.POSITIVE_INFINITY;
    line = line.map((point) => {
      best = Math.min(best, point.y);
      return { ...point, y: best };
    });
  }

  return line;
}

function datasetMinimumCumulativeBestLinePoints(points) {
  const clean = points
    .map((point) => ({
      x: finiteNumber(point.x),
      y: finiteNumber(point.y),
    }))
    .filter((point) => point.x != null && point.y != null)
    .sort((a, b) => a.x - b.x);

  let best = Number.POSITIVE_INFINITY;
  return clean.map((point) => {
    best = Math.min(best, point.y);
    return {
      x: point.x,
      y: best,
    };
  });
}

function datasetMinimumMovingAverageLinePoints(points, windowSize) {
  const clean = aggregateFitPoints(points)
    .map((point) => ({
      x: finiteNumber(point.x),
      y: finiteNumber(point.y),
    }))
    .filter((point) => point.x != null && point.y != null)
    .sort((a, b) => a.x - b.x);

  if (!clean.length) return [];

  const window = Math.max(1, Math.min(Math.round(windowSize || 3), clean.length));
  const leftSpan = Math.floor((window - 1) / 2);
  const rightSpan = window - leftSpan - 1;

  return clean.map((point, index) => {
    const start = Math.max(0, index - leftSpan);
    const end = Math.min(clean.length, index + rightSpan + 1);
    const slice = clean.slice(start, end);
    const y = slice.reduce((sum, item) => sum + item.y, 0) / slice.length;

    return {
      x: point.x,
      y,
      window_count: slice.length,
    };
  });
}


function datasetMinimumFitLinePoints(points, fitKind, options = {}) {
  if (fitKind === "none") return [];
  if (fitKind === "moving_average") {
    return datasetMinimumMovingAverageLinePoints(
      points,
      options.movingAverageWindow ?? datasetMinimumSelectedMovingAverageWindow(),
    );
  }
  if (fitKind === "lowess_logx") {
    return datasetMinimumLowessLinePoints(points, {
      frac: 0.45,
      robustIterations: 0,
      monotone: false,
    });
  }

  if (fitKind === "lowess_logx_robust") {
    return datasetMinimumLowessLinePoints(points, {
      frac: 0.45,
      robustIterations: 2,
      monotone: false,
    });
  }

  if (fitKind === "monotone_lowess_logx") {
    return datasetMinimumLowessLinePoints(points, {
      frac: 0.45,
      robustIterations: 2,
      monotone: true,
    });
  }

  if (fitKind === "cumulative_best") {
    return datasetMinimumCumulativeBestLinePoints(points);
  }

  if (fitKind === "power_law" || fitKind === "power_law_floor") {
    return datasetMinimumPowerLawFitLinePoints(points);
  }

  const reciprocalPower = fitKindReciprocalPower(fitKind);
  if (reciprocalPower != null) return reciprocalFitLinePoints(points, reciprocalPower);

  return fitLinePoints(points, fitKindDegree(fitKind));
}

function datasetMinimumFitLabel(fitKind) {
  if (fitKind === "lowess_logx") return "LOWESS log-N (diagnostic)";
  if (fitKind === "lowess_logx_robust") return "robust LOWESS log-N (diagnostic)";
  if (fitKind === "monotone_lowess_logx") return "monotone LOWESS log-N (diagnostic)";
  if (fitKind === "moving_average") return "moving average (diagnostic)";
  if (fitKind === "cumulative_best") return "cumulative best (diagnostic)";
  if (fitKind === "power_law_floor") return "power law + floor (paper candidate if fit valid)";
  if (fitKind === "power_law") return "power law + floor (legacy alias; paper candidate if fit valid)";
  if (fitKind === "none") return "no fit (diagnostic observed thresholds)";
  if (["linear", "quadratic", "inverse", "inverse_square"].includes(fitKind)) {
    return `${fitKindLabel(fitKind)} (diagnostic)`;
  }
  return `${fitKindLabel(fitKind)} (diagnostic)`;
}

function datasetMinimumReplicateBootstrap(output = {}) {
  return output.replicate_bootstrap || output.bootstrap || {};
}

function datasetMinimumHierarchicalUncertainty(output = {}) {
  return output.hierarchical_uncertainty || output.paper_uncertainty || {};
}

function datasetMinimumPreviewCacheKey() {
  return [
    datasetMinimumRootKey(datasetMinimumSelectedRunRoots()),
    datasetMinimumSelectedMetric(),
    datasetMinimumSelectedXAxis(),
    datasetMinimumSelectedAggregationMode(),
  ].join("|");
}

function datasetMinimumSourceLabel(runRoot) {
  const normalized = datasetMinimumNormalizePath(runRoot);
  const sources = state.datasetMinimumPayload?.run_root_sources || [];
  const match = sources.find(
    (item) => datasetMinimumNormalizePath(item.run_root) === normalized,
  );
  return match?.label || normalized.split("/").filter(Boolean).pop() || normalized;
}

function datasetMinimumPlotSeriesKey(row = {}, multiSweep = false) {
  const method = String(row.method || "unknown");
  if (!multiSweep) return method;
  const sweep = String(row.sweep_label || row.source_run_root || "").trim();
  return sweep ? `${sweep}::${method}` : method;
}

function datasetMinimumPlotSeriesLabel(row = {}, multiSweep = false, aggregated = false) {
  const method = methodDisplayLabel(row.method);
  if (aggregated) return `${method} mean`;
  if (!multiSweep) return `${method} best observed`;
  const sweep = String(row.sweep_label || datasetMinimumSourceLabel(row.source_run_root) || "sweep");
  return `${sweep} · ${method} best observed`;
}

function datasetMinimumShouldAggregateRows(rows = [], selectedRootCount = 0) {
  if (selectedRootCount > 1) return true;
  const sourceRoots = new Set(
    rows.map((row) => String(row.source_run_root || "").trim()).filter(Boolean),
  );
  if (sourceRoots.size > 1) return true;
  const sweepLabels = new Set(
    rows
      .map((row) => String(row.sweep_label || "").trim())
      .filter((label) => label && !label.startsWith("mean (")),
  );
  return sweepLabels.size > 1;
}

function datasetMinimumAggregateRowsByMethod(rows = [], axis = "n_train") {
  const buckets = new Map();
  for (const row of rows) {
    const method = String(row.method || "unknown");
    const xValue = row.x_value != null ? finiteNumber(row.x_value) : datasetMinimumRowX(row, axis);
    const yValue = row.y_value != null ? finiteNumber(row.y_value) : finiteNumber(row.primary_metric_mev_mean);
    if (xValue == null || yValue == null) continue;
    const key = `${method}\u0000${xValue}`;
    if (!buckets.has(key)) {
      buckets.set(key, { method, xValue, yValues: [], sources: [], sample: row });
    }
    const bucket = buckets.get(key);
    bucket.yValues.push(yValue);
    const source = String(row.sweep_label || row.source_run_root || "").trim();
    if (source && !source.startsWith("mean (")) bucket.sources.push(source);
  }
  return Array.from(buckets.values())
    .map((bucket) => {
      const yMean = bucket.yValues.reduce((sum, value) => sum + value, 0) / bucket.yValues.length;
      const uniqueSources = Array.from(new Set(bucket.sources));
      return {
        ...bucket.sample,
        method: bucket.method,
        x_value: bucket.xValue,
        y_value: yMean,
        primary_metric_mev_mean: yMean,
        source_count: bucket.yValues.length,
        aggregated_sources: uniqueSources,
        config_id: bucket.yValues.length > 1 ? "aggregated_mean" : (bucket.sample.config_id || "-"),
      };
    })
    .sort((left, right) => left.x_value - right.x_value || String(left.method).localeCompare(String(right.method)));
}

function datasetMinimumAggregateCostRowsByMethod(rows = []) {
  const buckets = new Map();
  for (const row of rows) {
    const method = String(row.method || "unknown");
    const xValue = finiteNumber(row.x_value);
    const yValue = finiteNumber(row.y_value);
    const costValue = finiteNumber(row.cost_value);
    if (xValue == null || yValue == null || costValue == null) continue;
    const key = `${method}\u0000${xValue}`;
    if (!buckets.has(key)) {
      buckets.set(key, {
        method,
        xValue,
        yValues: [],
        costValues: [],
        sources: [],
        sample: row,
      });
    }
    const bucket = buckets.get(key);
    bucket.yValues.push(yValue);
    bucket.costValues.push(costValue);
    const source = String(row.sweep_label || row.source_run_root || "").trim();
    if (source && !source.startsWith("mean (")) bucket.sources.push(source);
  }
  return Array.from(buckets.values())
    .map((bucket) => {
      const yMean = bucket.yValues.reduce((sum, value) => sum + value, 0) / bucket.yValues.length;
      const costMean = bucket.costValues.reduce((sum, value) => sum + value, 0) / bucket.costValues.length;
      return {
        ...bucket.sample,
        method: bucket.method,
        x_value: bucket.xValue,
        y_value: yMean,
        primary_metric_mev_mean: yMean,
        cost_value: costMean,
        source_count: bucket.yValues.length,
        aggregated_sources: Array.from(new Set(bucket.sources)),
        config_id: bucket.yValues.length > 1 ? "aggregated_mean" : (bucket.sample.config_id || "-"),
      };
    })
    .sort((left, right) => left.x_value - right.x_value || String(left.method).localeCompare(String(right.method)));
}

function datasetMinimumPlotRowsFromOutput(output = {}) {
  if (!output) return [];
  if (Array.isArray(output.aggregated_rows) && output.aggregated_rows.length) {
    return output.aggregated_rows;
  }
  return output.best_rows || [];
}

function datasetMinimumResolvePlotOutput(output, preview) {
  const exactMatch = output && datasetMinimumOutputMatchesCurrentSelection(output);
  if (exactMatch) {
    const plotRows = datasetMinimumPlotRowsFromOutput(output);
    return {
      ...output,
      best_rows: plotRows,
      aggregated_rows: plotRows,
      aggregated: datasetMinimumUsesBackendAggregationMode(output.aggregation_mode)
        || Boolean(output.aggregated),
      is_preview: false,
    };
  }

  const previewRows = preview?.best_rows || [];
  if (!previewRows.length) return output || preview || null;

  return {
    ...(preview || {}),
    thresholds: {},
    threshold_mev: datasetMinimumSelectedThreshold(),
    primary_metric: datasetMinimumSelectedMetric(),
    x_axis: datasetMinimumSelectedXAxis(),
    aggregation_mode: datasetMinimumSelectedAggregationMode(),
    best_rows: previewRows,
    aggregated_rows: preview?.aggregated_rows || previewRows,
    is_preview: true,
  };
}

let datasetMinimumViewRefreshPromise = null;
let datasetMinimumViewRefreshPending = false;

async function refreshDatasetMinimumView() {
  if (datasetMinimumViewRefreshPromise) {
    datasetMinimumViewRefreshPending = true;
    return datasetMinimumViewRefreshPromise;
  }

  const requestId = ++state.datasetMinimumViewRequestId;
  const payload = state.datasetMinimumPayload || {};
  const output = datasetMinimumFindOutput(payload);
  const selectedRoots = datasetMinimumSelectedRunRoots();

  datasetMinimumViewRefreshPromise = (async () => {
    let plotOutput = output;

    if (output && datasetMinimumOutputMatchesCurrentSelection(output)) {
      plotOutput = datasetMinimumResolvePlotOutput(output, null);
    } else if (selectedRoots.length) {
      const cacheKey = datasetMinimumPreviewCacheKey();
      let preview = state.datasetMinimumPreviewCache?.key === cacheKey
        ? state.datasetMinimumPreviewCache.data
        : null;

      if (!preview) {
        try {
          if (!window.Plotly) await ensurePlotlyLoaded();
          preview = await request("/api/g2m-deeph/dataset-size-minimum/preview", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              run_roots: selectedRoots,
              primary_metric: datasetMinimumSelectedMetric(),
              x_axis: datasetMinimumSelectedXAxis(),
              aggregation_mode: datasetMinimumSelectedAggregationMode(),
            }),
          });
          state.datasetMinimumPreviewCache = { key: cacheKey, data: preview };
        } catch (error) {
          if (requestId !== state.datasetMinimumViewRequestId) return;
          renderDatasetMinimumSelectedOutput(payload, {
            output,
            plotOutput: null,
            previewError: error,
          });
          return;
        }
      }

      if (requestId !== state.datasetMinimumViewRequestId) return;
      plotOutput = datasetMinimumResolvePlotOutput(output, preview);
    }

    if (requestId !== state.datasetMinimumViewRequestId) return;
    renderDatasetMinimumSelectedOutput(payload, { output, plotOutput });
  })().finally(() => {
    datasetMinimumViewRefreshPromise = null;
    if (datasetMinimumViewRefreshPending) {
      datasetMinimumViewRefreshPending = false;
      refreshDatasetMinimumView().catch((error) => showToast(error.message));
    }
  });

  return datasetMinimumViewRefreshPromise;
}

function renderDatasetMinimumControls() {
  const payload = state.datasetMinimumPayload || {};
  populateDatasetMinimumThresholdPresets(payload);
  updateDatasetMinimumMovingAverageVisibility();
  datasetMinimumInvalidatePreferredOutputIfStale();
  refreshDatasetMinimumView();
}

function datasetMinimumBackendFitLine(method, output, selectedFit) {
  const backendFit = output?.fits?.[method]?.[datasetMinimumBackendFitModel(selectedFit)];
  if (!backendFit || backendFit.status !== "ok") return [];

  const curvePoints = backendFit.curve_points || [];
  if (curvePoints.length > 1) {
    return curvePoints
      .map((point) => ({
        x: finiteNumber(point.x),
        y: finiteNumber(point.y),
      }))
      .filter((point) => point.x != null && point.y != null);
  }

  const coefficients = backendFit.coefficients || [];
  if (!coefficients.length) return [];

  const rows = datasetMinimumPlotRowsFromOutput(output)
    .filter((row) => String(row.method || "") === String(method || ""));
  const xValues = rows
    .map((row) => finiteNumber(row.dataset_size_x))
    .filter((value) => value != null && value > 0)
    .sort((a, b) => a - b);
  if (xValues.length < 2) return [];

  const minX = xValues[0];
  const maxX = xValues[xValues.length - 1];
  const grid = minX === maxX
    ? [minX]
    : Array.from({ length: 100 }, (_, index) => minX + ((maxX - minX) * index) / 99);

  if (
    (selectedFit === "power_law" || selectedFit === "power_law_floor")
    && coefficients.length >= 3
  ) {
    const [eInf, amplitude, alpha] = coefficients;
    return grid.map((x) => ({ x, y: eInf + amplitude * (x ** (-alpha)) }));
  }

  return datasetMinimumFitLinePoints(
    rows.map((row) => ({
      x: finiteNumber(row.dataset_size_x),
      y: finiteNumber(row.primary_metric_mev_mean),
    })).filter((point) => point.x != null && point.y != null),
    selectedFit,
    { movingAverageWindow: datasetMinimumOutputMovingAverageWindow(output) },
  );
}

function renderDatasetMinimumPlot(output, axis) {
  const card = document.getElementById("dataset-minimum-plot");
  if (!card) return;
  if (!window.Plotly) {
    card.textContent = "Plotly no esta cargado.";
    return;
  }
  const bootstrap = datasetMinimumReplicateBootstrap(output);
  const sourceRows = datasetMinimumPlotRowsFromOutput(output);
  const rows = sourceRows
    .map((row) => ({
      ...row,
      x_value: datasetMinimumRowX(row, axis),
      y_value: finiteNumber(row.primary_metric_mev_mean),
      cost: finiteNumber(row.gpu_hours_total_mean),
    }))
    .filter((row) => row.x_value != null && row.y_value != null)
    .sort((a, b) => a.x_value - b.x_value || String(a.method || "").localeCompare(String(b.method || "")));
  if (!rows.length) {
    renderPlot(card, [], plotLayout("Dataset size minimum", "H error (meV)", {
      annotations: [emptyPlotAnnotation("No finite rows for selected dataset-size-minimum output.")],
    }));
    return;
  }
  const selectedFit = datasetMinimumSelectedFit();
  const backendFitModel = datasetMinimumBackendFitModel(selectedFit);
  const traces = [...datasetMinimumRawReplicateTraces(output, axis)];
  const aggregationMode = output?.aggregation_mode
    || (output?.is_preview ? datasetMinimumSelectedAggregationMode() : "mean_replicates");
  const selectedRootCount = datasetMinimumSelectedRunRoots().length;
  const useBackendAggregation = !output?.is_preview && (
    datasetMinimumUsesBackendAggregationMode(aggregationMode)
    || rows.some((row) => row.is_aggregated_mean || (row.replicate_count || 0) > 1)
  );
  const aggregated = datasetMinimumUsesBackendAggregationMode(aggregationMode)
    || useBackendAggregation
    || Boolean(output?.aggregated)
    || datasetMinimumShouldAggregateRows(rows, selectedRootCount);
  const plotRows = useBackendAggregation
    || datasetMinimumUsesBackendAggregationMode(aggregationMode)
    || !output?.is_preview
    ? rows
    : aggregated
      ? datasetMinimumAggregateRowsByMethod(rows, axis)
      : rows;

  const sweepLabels = new Set(
    plotRows.map((row) => String(row.sweep_label || row.source_run_root || "").trim()).filter(Boolean),
  );
  const multiSweep = !aggregated && sweepLabels.size > 1;
  const seriesKeys = aggregated
    ? Array.from(new Set(plotRows.map((row) => String(row.method || "unknown")))).sort()
    : Array.from(new Set(plotRows.map((row) => datasetMinimumPlotSeriesKey(row, multiSweep)))).sort();
  seriesKeys.forEach((seriesKey, index) => {
    const group = aggregated
      ? plotRows.filter((row) => String(row.method || "unknown") === seriesKey)
      : plotRows.filter((row) => datasetMinimumPlotSeriesKey(row, multiSweep) === seriesKey);
    const sample = group[0] || {};
    const method = String(sample.method || "unknown");
    const color = DATASET_MINIMUM_METHOD_COLORS[method] || plotColor(index);
    const seriesLabel = aggregated
      ? `${methodDisplayLabel(method)} ${datasetMinimumAggregationSeriesSuffix(aggregationMode)}`
      : datasetMinimumPlotSeriesLabel(sample, multiSweep, aggregated);
    const errorValues = group.map((row) => datasetMinimumRowErrorBarDelta(row));
    const hasErrorBars = errorValues.some((value) => value != null && value > 0);
    traces.push({
      type: "scatter",
      mode: "lines+markers",
      x: group.map((row) => row.x_value),
      y: group.map((row) => row.y_value),
      marker: { symbol: g2mDeephMarkerSymbol(method), size: g2mDeephIsDeepH(method) ? 10 : 8, color },
      line: { color, width: 2 },
      name: seriesLabel,
      text: group.map((row) => datasetMinimumAggregatedRowHover(row, axis)),
      hovertemplate: "%{text}<extra>%{fullData.name}</extra>",
      error_y: hasErrorBars
        ? {
          type: "data",
          array: errorValues.map((value) => value || 0),
          visible: true,
          color,
          thickness: 1.2,
          width: 4,
        }
        : undefined,
    });
    const fitPoints = group.map((row) => ({ x: row.x_value, y: row.y_value }));
    const fitLine = !output?.is_preview && output?.fits?.[method]?.[backendFitModel]?.status === "ok"
      ? datasetMinimumBackendFitLine(method, output, selectedFit)
      : datasetMinimumFitLinePoints(fitPoints, selectedFit, {
        movingAverageWindow: output?.is_preview
          ? datasetMinimumSelectedMovingAverageWindow()
          : datasetMinimumOutputMovingAverageWindow(output),
      });
    if (fitLine.length > 1) {
      traces.push({
        type: "scatter",
        mode: "lines",
        x: fitLine.map((point) => point.x),
        y: fitLine.map((point) => point.y),
        name: `${seriesLabel} ${datasetMinimumFitLabel(selectedFit)} fit`,
        line: { color, width: 2, dash: fitKindDash(selectedFit) },
        opacity: 0.45,
        hoverinfo: "skip",
      });
    }
  });
  const threshold = finiteNumber(output.threshold_mev);
  const shapes = [];
  const annotations = [];
  if (threshold != null) {
    shapes.push({
      type: "line",
      xref: "paper",
      x0: 0,
      x1: 1,
      yref: "y",
      y0: threshold,
      y1: threshold,
      line: { color: "#111827", width: 1.6, dash: "dot" },
    });
    annotations.push({
      text: `threshold ${formatCompactNumber(threshold)} meV`,
      xref: "paper",
      x: 1,
      xanchor: "right",
      yref: "y",
      y: threshold,
      yanchor: "bottom",
      showarrow: false,
      font: { size: 13, color: "#111827" },
      bgcolor: "rgba(255,255,255,0.82)",
    });
  }
  const bootstrapOverlays = datasetMinimumBootstrapNMinOverlays(output, axis);
  shapes.push(...bootstrapOverlays.shapes);
  annotations.push(...bootstrapOverlays.annotations);

  const thresholds = output.thresholds || {};
  let annotationOffset = bootstrapOverlays.annotations.length;
  Object.keys(thresholds).sort().forEach((method, index) => {
    const color = DATASET_MINIMUM_METHOD_COLORS[method] || plotColor(index);
    for (const criterionKey of datasetMinimumCriteriaToPlot()) {
      const criterion = DATASET_MINIMUM_CRITERIA[criterionKey];
      if (!criterion) continue;
      const nValue = finiteNumber(datasetMinimumDisplayN(output, method, datasetMinimumCriterionValue(thresholds[method], criterionKey), axis));
      if (nValue == null) continue;
      const hasBootstrapBand = Boolean(
        bootstrap?.enabled
        && datasetMinimumCriterionValue(bootstrap?.by_method?.[method], criterionKey)?.lower != null,
      );
      if (!hasBootstrapBand) {
        shapes.push({
          type: "line",
          xref: "x",
          x0: nValue,
          x1: nValue,
          yref: "paper",
          y0: 0,
          y1: 1,
          line: { color, width: criterion.width, dash: criterion.dash },
        });
      }
      annotations.push({
        text: `${methodDisplayLabel(method)} ${criterion.label}=${Math.round(nValue)}`,
        xref: "x",
        x: nValue,
        yref: "paper",
        y: 1.02 + (index * 0.055) + (annotationOffset * 0.028),
        showarrow: false,
        font: { size: 11, color },
        bgcolor: "rgba(255,255,255,0.82)",
      });
      annotationOffset += 1;
    }
  });
  const yValues = plotRows.map((row) => row.y_value).concat(threshold == null ? [] : [threshold]);
  const modeLabel = datasetMinimumAggregationModeLabel(
    output?.aggregation_mode || datasetMinimumSelectedAggregationMode(),
  );
  const title = `Dataset size minimum · ${DATASET_MINIMUM_METRIC_LABELS[output.primary_metric] || output.primary_metric || "metric"} · ${modeLabel}`;
  const layout = plotLayout(title, "Error (meV)", {
    shapes,
    annotations,
    xaxis: {
      title: datasetMinimumAxisLabel(axis),
      gridcolor: "#edf1f4",
      zeroline: false,
      range: paddedLinearRange(plotRows.map((row) => row.x_value), { forceZeroMin: true }),
    },
    yaxis: {
      title: "Error (meV)",
      gridcolor: "#edf1f4",
      zeroline: false,
      range: paddedLinearRange(yValues, { forceZeroMin: true }),
    },
    legend: { orientation: "h", y: -0.22 },
    margin: { t: 84 + Math.min(annotationOffset * 10, 96), r: 48, b: 70, l: 72 },
  });
  renderPlot(card, traces, layout, {
    toImageButtonOptions: { filename: "dataset_size_minimum" },
  });
}

function datasetMinimumCostPlotRows(output, axis) {
  const costBasis = datasetMinimumOutputCostBasis(output);
  const rows = datasetMinimumPlotRowsFromOutput(output)
    .map((row) => ({
      ...row,
      x_value: datasetMinimumRowX(row, axis),
      y_value: finiteNumber(row.primary_metric_mev_mean),
      cost_value: datasetMinimumRowCost(row, costBasis),
    }))
    .filter((row) => row.x_value != null && row.y_value != null && row.cost_value != null)
    .sort((a, b) => a.x_value - b.x_value || String(a.method || "").localeCompare(String(b.method || "")));
  if (!rows.length) return [];

  const aggregationMode = output?.aggregation_mode
    || (output?.is_preview ? datasetMinimumSelectedAggregationMode() : "mean_replicates");
  const selectedRootCount = datasetMinimumSelectedRunRoots().length;
  const useBackendAggregation = !output?.is_preview && (
    datasetMinimumUsesBackendAggregationMode(aggregationMode)
    || rows.some((row) => row.is_aggregated_mean || (row.replicate_count || 0) > 1)
  );
  const aggregated = datasetMinimumUsesBackendAggregationMode(aggregationMode)
    || useBackendAggregation
    || Boolean(output?.aggregated)
    || datasetMinimumShouldAggregateRows(rows, selectedRootCount);
  return useBackendAggregation
    || datasetMinimumUsesBackendAggregationMode(aggregationMode)
    || !output?.is_preview
    ? rows
    : aggregated
      ? datasetMinimumAggregateCostRowsByMethod(rows)
      : rows;
}

function datasetMinimumCostPerErrorPlotRows(output, axis) {
  return datasetMinimumCostPlotRows(output, axis)
    .map((row) => ({
      ...row,
      efficiency_value: row.y_value > 0 ? row.cost_value / row.y_value : null,
    }))
    .filter((row) => row.efficiency_value != null && Number.isFinite(row.efficiency_value) && row.efficiency_value > 0);
}

function renderDatasetMinimumCostPlot(output, axis) {
  const card = document.getElementById("dataset-minimum-cost-plot");
  if (!card) return;
  if (!window.Plotly) {
    card.textContent = "Plotly no esta cargado.";
    return;
  }
  const rows = datasetMinimumCostPlotRows(output, axis);
  const costBasis = datasetMinimumOutputCostBasis(output);
  const costLabel = datasetMinimumCostBasisLabel(costBasis);
  if (!rows.length) {
    renderPlot(card, [], plotLayout("Dataset size minimum cost efficiency", "GPU-hours", {
      annotations: [emptyPlotAnnotation(`No finite cost rows for selected cost basis (${costLabel}).`)],
    }));
    return;
  }

  const methods = Array.from(new Set(rows.map((row) => String(row.method || "unknown")))).sort();
  const traces = [];
  methods.forEach((method, index) => {
    const group = rows.filter((row) => String(row.method || "unknown") === method);
    const color = DATASET_MINIMUM_METHOD_COLORS[method] || plotColor(index);
    const label = methodDisplayLabel(method);
    const hoverText = group.map((row) => [
      `method: ${label}`,
      `${datasetMinimumAxisLabel(axis)}: ${formatCompactNumber(row.x_value)}`,
      `error: ${formatCompactNumber(row.y_value)} meV`,
      `cost: ${formatCompactNumber(row.cost_value)} GPU-hours`,
      `cost_basis: ${costBasis}`,
      `config_id: ${row.config_id || "-"}`,
      `seeds: ${datasetMinimumListField(row.seeds || row.seed)}`,
      `source_run_roots: ${datasetMinimumListField(row.source_run_roots || row.source_run_root)}`,
    ].join("<br>"));

    traces.push({
      type: "scatter",
      mode: "lines+markers",
      x: group.map((row) => row.x_value),
      y: group.map((row) => row.cost_value),
      marker: { symbol: g2mDeephMarkerSymbol(method), size: g2mDeephIsDeepH(method) ? 10 : 8, color },
      line: { color, width: 2 },
      name: `${label} cost`,
      legendgroup: method,
      text: hoverText,
      hovertemplate: "%{text}<extra>%{fullData.name}</extra>",
      xaxis: "x",
      yaxis: "y",
    });
    traces.push({
      type: "scatter",
      mode: "markers+text",
      x: group.map((row) => row.cost_value),
      y: group.map((row) => row.y_value),
      text: group.map((row) => formatCompactNumber(row.x_value)),
      customdata: hoverText,
      marker: {
        symbol: g2mDeephMarkerSymbol(method),
        size: g2mDeephIsDeepH(method) ? 11 : 9,
        color,
        opacity: 0.88,
      },
      textposition: "top center",
      textfont: { size: 10, color },
      name: `${label} accuracy/cost`,
      legendgroup: method,
      showlegend: false,
      hovertemplate: "%{customdata}<extra>%{fullData.name}</extra>",
      xaxis: "x2",
      yaxis: "y2",
    });
  });

  const title = `Dataset size minimum cost efficiency · ${costLabel}`;
  const layout = plotLayout(title, "GPU-hours", {
    grid: undefined,
    xaxis: {
      title: datasetMinimumAxisLabel(axis),
      domain: [0, 0.44],
      gridcolor: "#edf1f4",
      zeroline: false,
      range: paddedLinearRange(rows.map((row) => row.x_value), { forceZeroMin: true }),
    },
    yaxis: {
      title: costLabel,
      anchor: "x",
      gridcolor: "#edf1f4",
      zeroline: false,
      range: paddedLinearRange(rows.map((row) => row.cost_value), { forceZeroMin: true }),
    },
    xaxis2: {
      title: costLabel,
      domain: [0.58, 1],
      anchor: "y2",
      gridcolor: "#edf1f4",
      zeroline: false,
      range: paddedLinearRange(rows.map((row) => row.cost_value), { forceZeroMin: true }),
    },
    yaxis2: {
      title: "Error (meV)",
      anchor: "x2",
      gridcolor: "#edf1f4",
      zeroline: false,
      range: paddedLinearRange(rows.map((row) => row.y_value), { forceZeroMin: true }),
    },
    annotations: [
      {
        text: "Cost vs dataset size",
        xref: "paper",
        yref: "paper",
        x: 0.22,
        y: 1.08,
        showarrow: false,
        font: { size: 13, color: "#374151" },
      },
      {
        text: "Accuracy vs cost",
        xref: "paper",
        yref: "paper",
        x: 0.79,
        y: 1.08,
        showarrow: false,
        font: { size: 13, color: "#374151" },
      },
    ],
    legend: { orientation: "h", y: -0.2 },
    margin: { t: 92, r: 48, b: 76, l: 78 },
  });
  renderPlot(card, traces, layout, {
    toImageButtonOptions: { filename: "dataset_size_minimum_cost_efficiency_interactive" },
  });
}

function renderDatasetMinimumCostPerErrorPlot(output, axis) {
  const card = document.getElementById("dataset-minimum-cost-per-error-plot");
  if (!card) return;
  if (!window.Plotly) {
    card.textContent = "Plotly no esta cargado.";
    return;
  }
  const rows = datasetMinimumCostPerErrorPlotRows(output, axis);
  const costBasis = datasetMinimumOutputCostBasis(output);
  const costLabel = datasetMinimumCostBasisLabel(costBasis);
  if (!rows.length) {
    renderPlot(card, [], plotLayout("Dataset size minimum GPU-hours per error", "GPU-hours / error (meV)", {
      annotations: [emptyPlotAnnotation(`No finite GPU-hours / error rows for selected cost basis (${costLabel}).`)],
    }));
    return;
  }

  const methods = Array.from(new Set(rows.map((row) => String(row.method || "unknown")))).sort();
  const traces = [];
  methods.forEach((method, index) => {
    const group = rows.filter((row) => String(row.method || "unknown") === method);
    const color = DATASET_MINIMUM_METHOD_COLORS[method] || plotColor(index);
    const label = methodDisplayLabel(method);
    traces.push({
      type: "scatter",
      mode: "lines+markers",
      x: group.map((row) => row.x_value),
      y: group.map((row) => row.efficiency_value),
      marker: { symbol: g2mDeephMarkerSymbol(method), size: g2mDeephIsDeepH(method) ? 10 : 8, color },
      line: { color, width: 2 },
      name: label,
      text: group.map((row) => [
        `method: ${label}`,
        `${datasetMinimumAxisLabel(axis)}: ${formatCompactNumber(row.x_value)}`,
        `cost/error: ${formatCompactNumber(row.efficiency_value)} GPU-hours per meV`,
        `cost: ${formatCompactNumber(row.cost_value)} GPU-hours`,
        `error: ${formatCompactNumber(row.y_value)} meV`,
        `cost_basis: ${costBasis}`,
        `config_id: ${row.config_id || "-"}`,
        `seeds: ${datasetMinimumListField(row.seeds || row.seed)}`,
        `source_run_roots: ${datasetMinimumListField(row.source_run_roots || row.source_run_root)}`,
      ].join("<br>")),
      hovertemplate: "%{text}<extra>%{fullData.name}</extra>",
    });
  });

  const title = `Dataset size minimum GPU-hours per error · ${costLabel}`;
  const layout = plotLayout(title, "GPU-hours / error (meV)", {
    xaxis: {
      title: datasetMinimumAxisLabel(axis),
      gridcolor: "#edf1f4",
      zeroline: false,
      range: paddedLinearRange(rows.map((row) => row.x_value), { forceZeroMin: true }),
    },
    yaxis: {
      title: "GPU-hours / error (meV)",
      type: "log",
      gridcolor: "#edf1f4",
      zeroline: false,
      range: paddedLogRange(rows.map((row) => row.efficiency_value)),
    },
    annotations: [
      {
        text: "Y axis in log scale",
        xref: "paper",
        yref: "paper",
        x: 0,
        y: 1.08,
        xanchor: "left",
        showarrow: false,
        font: { size: 13, color: "#6b7280" },
      },
    ],
    legend: { orientation: "h", y: -0.2 },
    margin: { t: 82, r: 48, b: 76, l: 78 },
  });
  renderPlot(card, traces, layout, {
    toImageButtonOptions: { filename: "dataset_size_minimum_gpu_hours_per_error" },
  });
}

function renderDatasetMinimumStatus(output, payload, plotOutput = null) {
  const status = document.getElementById("dataset-minimum-status");
  if (!status) return;

  status.className = "comparison-status-banner diagnostic";

  if (!payload?.available) {
    status.classList.add("invalid");
    status.textContent = payload?.message || "No dataset-size-minimum outputs found.";
    return;
  }

  const selectedRunRoots = datasetMinimumNormalizeRoots(datasetMinimumSelectedRunRoots());
  const selectedAggregation = datasetMinimumSelectedAggregationMode();
  const selectedAggregationMeta = datasetMinimumAggregationModeClassification(selectedAggregation);

  if (!output) {
    const metric = datasetMinimumSelectedMetric();
    const threshold = datasetMinimumSelectedThreshold();
    const previewRows = plotOutput?.best_rows || [];

    if (previewRows.length && selectedRunRoots.length) {
      status.textContent =
        `Preview from ${selectedRunRoots.length} sweep(s) for ${DATASET_MINIMUM_METRIC_LABELS[metric] || metric}. ` +
        `Run analysis required for this exact selection (aggregation, bootstrap, fit, threshold).`;
      return;
    }

    status.classList.add("invalid");
    status.textContent =
      `Run analysis required for this exact selection. ` +
      `No stored output matches metric=${DATASET_MINIMUM_METRIC_LABELS[metric] || metric}, ` +
      `threshold=${threshold ?? "-"} meV, aggregation=${datasetMinimumAggregationModeLabel(datasetMinimumSelectedAggregationMode())}, ` +
      `bootstrap=${datasetMinimumSelectedBootstrapReplicates()}, ci=${datasetMinimumSelectedCiLevel()}.` +
      `${selectedAggregationMeta.classification === "diagnostic_only" ? ` Selected aggregation mode is diagnostic-only (${selectedAggregationMeta.reason}).` : ""}`;

    return;
  }

  const requestedThreshold = datasetMinimumSelectedThreshold();
  const outputThreshold = finiteNumber(output.threshold_mev);
  const thresholdPolicyNote = output.threshold_basis
    ? ` Threshold policy: basis=${output.threshold_basis}, reference=${output.threshold_reference || "n/a"}, reference_type=${output.threshold_protocol_reference_type || "n/a"}, family=${output.threshold_metric_family || "unknown"}, user_defined=${Boolean(output.threshold_is_user_defined)}.`
    : "";
  const thresholdWarningNote = output.threshold_is_user_defined
    ? " WARNING: manual threshold is user_defined_exploratory."
    : "";

  const thresholdNote =
    requestedThreshold != null &&
    outputThreshold != null &&
    Math.abs(requestedThreshold - outputThreshold) > 1e-9
      ? ` Mostrando output precomputado mas cercano (${formatCompactNumber(outputThreshold)} meV).`
      : "";

  const warnings = (output.warnings || []).filter(Boolean);

  const axis = datasetMinimumSelectedXAxis();
  const axisNote =
    axis !== (output.x_axis || "n_train")
      ? ` N_min fue calculado sobre ${output.x_axis || "n_train"} y se remapea visualmente a ${axis}.`
      : "";

  const outputRunRoots = datasetMinimumNormalizeRoots(output.run_roots || []);
  const exactRootMatch =
    datasetMinimumRootKey(outputRunRoots) === datasetMinimumRootKey(selectedRunRoots);
  const aggregationMeta = datasetMinimumOutputAggregationMetadata(output);
  const aggregationMode = aggregationMeta.actual;
  const aggregationNote = ` Aggregation: ${datasetMinimumAggregationModeLabel(aggregationMode)}.`;
  const aggregationProtocolNote =
    ` Aggregation protocol: requested=${aggregationMeta.requested || "legacy/inferred"}, actual=${aggregationMeta.actual}, ` +
    `classification=${aggregationMeta.classification} (${aggregationMeta.reason}).`;
  const aggregationLegacyNote = aggregationMeta.legacyInferred
    ? " WARNING: aggregation mode is legacy/inferred because the original summary did not record an explicit requested mode."
    : "";
  const bestConfigWarning = aggregationMode === "best_config"
    ? " Warning: best_config is diagnostic only, not recommended for paper-level reporting."
    : aggregationMode === "mean_replicates"
      ? " Warning: mean_replicates mixes configs/seeds and is diagnostic only."
      : "";
  const aggregationDiagnosticWarning = aggregationMeta.classification === "diagnostic_only"
    ? " WARNING: selected aggregation mode is diagnostic-only."
    : "";
  const requestedSource = output.requested_n_min_source || output.n_min_source || "observed";
  const actualSource = output.actual_n_min_source || output.n_min_source || requestedSource;
  const nMinSourceNote = output.fallback_used
    ? ` N_min source: requested=${requestedSource}, actual=${actualSource} (FIT FAILED; explicit observed fallback).`
    : ` N_min source: ${actualSource}.`;
  const costBasis = datasetMinimumOutputCostBasis(output);
  const costBasisNote = ` Cost basis: ${datasetMinimumCostBasisLabel(costBasis)}.`;
  const requestedClaimMode = datasetMinimumOutputClaimModeRequested(output);
  const actualClaimMode = datasetMinimumOutputClaimModeActual(output);
  const claimModeNote = ` Claim mode: requested=${requestedClaimMode}, actual=${actualClaimMode}.`;
  const claimModeWarning = requestedClaimMode === "paper_candidate" && actualClaimMode !== "paper_candidate"
    ? " Diagnostic only: do not use as a paper-level minimum snapshot claim."
    : actualClaimMode === "paper_candidate"
      ? " Paper-candidate only for nominal N_min under the audited sweep protocol; not a validated independent-sample minimum."
      : "";
  const fitNote = ` Fit: ${datasetMinimumFitLabel(datasetMinimumSelectedFit())} (canonical: ${output.canonical_fit_model || datasetMinimumOutputNMinFitModel(output)}).`;
  const fallbackNote = output.fallback_used
    ? ` WARNING: canonical fit failed (${output.fallback_reason || "unknown"}); thresholds use observed points.`
    : "";
  const windowNote = datasetMinimumOutputNMinFitModel(output) === "moving_average"
    ? ` Moving-average window: ${datasetMinimumOutputMovingAverageWindow(output)}.`
    : "";
  const bootstrap = datasetMinimumReplicateBootstrap(output);
  const bootstrapNote = bootstrap.enabled
    ? ` Replicate-resampling CI: enabled (${bootstrap.replicates_requested ?? output.bootstrap_replicates ?? 0} resamples, CI ${output.ci_level ?? datasetMinimumOutputCiLevel(output)}, successful ${bootstrap.replicates_successful ?? 0}; row-level only, not temporal/block bootstrap; does not model temporal autocorrelation, model-selection uncertainty, hyperparameter-selection uncertainty, or dependence between dataset sizes).`
    : " Replicate-resampling CI: disabled.";
  const bootstrapWarningNote = Array.isArray(bootstrap.warnings) && bootstrap.warnings.length
    ? ` Replicate-resampling CI warnings: ${bootstrap.warnings.slice(0, 4).join("; ")}.`
    : "";
  const costEffBootstrapNote = bootstrap.cost_eff_ci_available === false
    ? ` N_min_cost_eff is diagnostic observed cost-error behavior only and has no replicate-resampling CI (${bootstrap.cost_eff_ci_reason || "cost and metric are not jointly bootstrapped"}).`
    : "";
  const hierarchical = datasetMinimumHierarchicalUncertainty(output);
  const hierarchicalLevels = hierarchical.levels || {};
  const hierarchicalSummary = Object.entries(hierarchicalLevels)
    .map(([name, level]) => `${name}: available=${Boolean(level?.available)}, sufficient=${Boolean(level?.sufficient)}`)
    .join("; ");
  const hierarchicalNote = hierarchical.status
    ? ` Hierarchical uncertainty: ${hierarchical.display_label || "hierarchical uncertainty"} (${hierarchical.status}; paper-ready=${Boolean(hierarchical.paper_ready)}${hierarchicalSummary ? `; ${hierarchicalSummary}` : ""}).`
    : "";
  const hierarchicalBlockerNote = Array.isArray(hierarchical.paper_level_blockers) && hierarchical.paper_level_blockers.length
    ? ` Hierarchical uncertainty blockers: ${hierarchical.paper_level_blockers.slice(0, 4).join("; ")}.`
    : "";

  const temporal = output.temporal_diagnostics || {};
  let temporalNote = "";
  const nMinNominalWarning =
    " N_min uses nominal N. If MD snapshots are autocorrelated, independent sample count can be lower. Check N_eff before using this as a paper-level claim.";
  if (temporal.status_message) {
    temporalNote = ` ${temporal.status_message}.`;
  } else if (output.estimated_n_eff_train != null) {
    temporalNote = ` Estimated N_eff range: ${JSON.stringify(output.estimated_n_eff_train)}.`;
  } else {
    temporalNote = " N is nominal; N_eff not estimated.";
  }
  temporalNote += nMinNominalWarning;
  if (output.scientific_claim_status) {
    temporalNote += ` Scientific claim status: ${output.scientific_claim_status}.`;
  }
  if (Array.isArray(output.paper_level_blockers) && output.paper_level_blockers.length) {
    temporalNote += ` Paper-level blockers: ${output.paper_level_blockers.slice(0, 4).join("; ")}.`;
  }
  if (output.N_eff_over_N_nominal != null) {
    temporalNote += ` N_eff/N_nominal: ${formatCompactNumber(output.N_eff_over_N_nominal)}.`;
  }
  if (Object.keys(temporal.N_eff_by_dataset_size || {}).length) {
    temporalNote += " Per-size N_eff diagnostics are shown in the N nominal vs N_eff table below.";
  }
  if (temporal.n_eff_convention) {
    temporalNote += ` Convention: ${temporal.n_eff_convention}.`;
  }
  if ((temporal.warnings || []).some((item) => String(item).includes("n_eff_much_smaller_than_nominal"))) {
    temporalNote += " WARNING: N_eff much smaller than nominal N.";
  }
  if (output.autocorrelation_available === false && (temporal.warnings || []).length) {
    const temporalWarnings = (temporal.warnings || []).filter((item) =>
      String(item).includes("autocorrelation") || String(item).includes("temporal"),
    );
    if (temporalWarnings.length) {
      temporalNote += ` Temporal warnings: ${temporalWarnings.slice(0, 3).join("; ")}.`;
    }
  }

  const outputNote = outputRunRoots.length
    ? ` Output from ${outputRunRoots.length} sweep(s)${
        selectedRunRoots.length
          ? exactRootMatch
            ? " matching selection."
            : " different from current selection."
          : "."
      }`
    : "";

  status.textContent =
    `${aggregationNote}${aggregationProtocolNote}${aggregationLegacyNote}${bestConfigWarning}${aggregationDiagnosticWarning}${nMinSourceNote}${costBasisNote}${claimModeNote}${claimModeWarning}${fitNote}${fallbackNote}${windowNote}${bootstrapNote}${bootstrapWarningNote}${costEffBootstrapNote}${hierarchicalNote}${hierarchicalBlockerNote}${temporalNote}` +
    `${thresholdPolicyNote}${thresholdWarningNote}${thresholdNote}${outputNote}${axisNote}` +
    `${warnings.length ? ` Warnings: ${warnings.join("; ")}` : ""}` +
    `${payload.diagnostic_warning ? ` ${payload.diagnostic_warning}` : ""}`;
}



function renderDatasetMinimumSelectedOutput(
  payload = state.datasetMinimumPayload,
  view = {},
) {
  const currentPayload = payload || {};
  const output = view.output !== undefined ? view.output : datasetMinimumFindOutput(currentPayload);
  const plotOutput = view.plotOutput !== undefined ? view.plotOutput : output;
  const axis = datasetMinimumSelectedXAxis();
  const selectedCount = datasetMinimumSelectedRunRoots().length;
  const table = document.getElementById("dataset-minimum-table");
  const card = document.getElementById("dataset-minimum-plot");
  const costCard = document.getElementById("dataset-minimum-cost-plot");
  const costPerErrorCard = document.getElementById("dataset-minimum-cost-per-error-plot");
  const artifactsNode = document.getElementById("dataset-minimum-artifacts");

  if (view.previewError) {
    renderDatasetMinimumStatus(output, currentPayload, plotOutput);
    if (table) {
      table.textContent = output
        ? "N_min table requires a completed analysis for this exact sweep selection."
        : "No N_min threshold table available.";
    }
    if (card) {
      card.textContent = `No plot available: ${view.previewError.message || String(view.previewError)}`;
    }
    if (costCard) {
      costCard.textContent = `No cost plot available: ${view.previewError.message || String(view.previewError)}`;
    }
    if (costPerErrorCard) {
      costPerErrorCard.textContent = `No GPU-hours / error plot available: ${view.previewError.message || String(view.previewError)}`;
    }
    if (artifactsNode) {
      artifactsNode.textContent = "";
    }
    return;
  }

  if (output?.status === "no_usable_metric_rows") {
    renderDatasetMinimumStatus(output, currentPayload, plotOutput);
    if (table) {
      const warnings = output.warnings || [];
      table.innerHTML = `
        <div class="comparison-status-banner diagnostic invalid">
          No usable metric rows for this exact selection.
          ${warnings.length ? `<br><small>${escapeHtml(warnings.slice(0, 8).join("; "))}</small>` : ""}
        </div>
      `;
    }
    if (card && !(plotOutput?.best_rows || []).length) {
      card.textContent =
        "No plot available: selected sweeps do not contain usable rows for the selected metric/x-axis.";
    }
    renderDatasetMinimumArtifacts(output);
    if ((plotOutput?.best_rows || []).length) {
      renderDatasetMinimumPlot(plotOutput, axis);
      renderDatasetMinimumCostPlot(plotOutput, axis);
      renderDatasetMinimumCostPerErrorPlot(plotOutput, axis);
      schedulePlotResize("dataset-minimum-plot");
      schedulePlotResize("dataset-minimum-cost-plot");
      schedulePlotResize("dataset-minimum-cost-per-error-plot");
    } else if (costCard) {
      costCard.textContent =
        "No cost plot available: selected sweeps do not contain usable rows for the selected metric/x-axis.";
      if (costPerErrorCard) {
        costPerErrorCard.textContent =
          "No GPU-hours / error plot available: selected sweeps do not contain usable rows for the selected metric/x-axis.";
      }
    }
    return;
  }

  renderDatasetMinimumStatus(output, currentPayload, plotOutput);

  if (output) {
    renderDatasetMinimumTable(output, axis);
  } else if (table) {
    table.textContent = selectedCount
      ? "No N_min table for this sweep selection yet. Run analysis to compute thresholds."
      : "No N_min threshold table available.";
  }
  renderDatasetMinimumArtifacts(output);

  const plotRows = plotOutput?.best_rows || [];
  if (!plotRows.length) {
    if (card) {
      card.textContent = selectedCount
        ? "No plot available: selected sweeps do not contain usable rows for the selected metric/x-axis."
        : "No plot available for selected metric/threshold.";
    }
    if (costCard) {
      costCard.textContent = selectedCount
        ? "No cost plot available: selected sweeps do not contain usable rows for the selected metric/x-axis."
        : "No cost plot available for selected metric/threshold.";
    }
    if (costPerErrorCard) {
      costPerErrorCard.textContent = selectedCount
        ? "No GPU-hours / error plot available: selected sweeps do not contain usable rows for the selected metric/x-axis."
        : "No GPU-hours / error plot available for selected metric/threshold.";
    }
    return;
  }

  renderDatasetMinimumPlot(plotOutput, axis);
  renderDatasetMinimumCostPlot(plotOutput, axis);
  renderDatasetMinimumCostPerErrorPlot(plotOutput, axis);
  schedulePlotResize("dataset-minimum-plot");
  schedulePlotResize("dataset-minimum-cost-plot");
  schedulePlotResize("dataset-minimum-cost-per-error-plot");
}

function renderDatasetMinimumArtifacts(output = null) {
  const container = document.getElementById("dataset-minimum-artifacts");
  if (!container) return;
  const artifacts = Array.isArray(output?.artifact_outputs) ? output.artifact_outputs : [];
  if (!artifacts.length) {
    container.textContent = "";
    return;
  }

  const linkArtifacts = artifacts.filter((item) => item?.url);

  const cards = [`
    <section class="dataset-minimum-artifact-card">
      <div>
        <p class="eyebrow">Downloads</p>
        <h3>Export files</h3>
      </div>
      <div class="dataset-minimum-artifact-links">
        ${linkArtifacts.map((artifact) => `
          <a class="dataset-minimum-artifact-link" href="${escapeHtml(artifact.url)}" target="_blank" rel="noopener noreferrer">
            ${escapeHtml(artifact.label || artifact.name || "artifact")}
          </a>
        `).join("")}
      </div>
    </section>
  `];

  container.className = "dataset-minimum-artifacts";
  container.innerHTML = cards.join("");
}

function renderDatasetMinimum(payload = state.datasetMinimumPayload) {
  state.datasetMinimumPayload = payload || null;
  state.datasetMinimumPreviewCache = null;
  populateDatasetMinimumThresholdPresets(payload || {});
  renderDatasetMinimumRunSources(payload || {});
  refreshDatasetMinimumView();
}


function renderDatasetMinimumLoadError(error) {
  const status = document.getElementById("dataset-minimum-status");
  if (status) {
    status.className = "comparison-status-banner diagnostic invalid";
    status.textContent = `Error loading dataset-size-minimum outputs: ${error?.message || String(error)}`;
  }

  const sourcesNode = document.getElementById("dataset-minimum-run-sources");
  if (sourcesNode) {
    sourcesNode.textContent = "No se pudieron cargar los sweeps fuente.";
  }

  const card = document.getElementById("dataset-minimum-plot");
  if (card) {
    card.textContent = "No plot available because dataset-size-minimum loading failed.";
  }
  const costCard = document.getElementById("dataset-minimum-cost-plot");
  if (costCard) {
    costCard.textContent = "No cost plot available because dataset-size-minimum loading failed.";
  }
  const costPerErrorCard = document.getElementById("dataset-minimum-cost-per-error-plot");
  if (costPerErrorCard) {
    costPerErrorCard.textContent = "No GPU-hours / error plot available because dataset-size-minimum loading failed.";
  }
}

async function loadDatasetMinimum() {
  const sourcesNode = document.getElementById("dataset-minimum-run-sources");
  const statusNode = document.getElementById("dataset-minimum-status");

  if (sourcesNode) sourcesNode.textContent = "Cargando sweeps disponibles...";
  if (statusNode) {
    statusNode.className = "comparison-status-banner diagnostic";
    statusNode.textContent = "Loading dataset-size-minimum outputs...";
  }

  try {
    if (!window.Plotly) await ensurePlotlyLoaded();
    const payload = await request("/api/g2m-deeph/dataset-size-minimum");
    renderDatasetMinimum(payload);
    return payload;
  } catch (error) {
    renderDatasetMinimumLoadError(error);
    throw error;
  }
}

function formatG2MDeepHPlotRunTime(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric) || numeric <= 0) return "";
  return new Date(numeric * 1000).toLocaleString();
}

function setG2MDeepHSelectedPlotRuns(ids) {
  const visibleIds = new Set((state.g2mDeephPlotRuns || []).map((run) => run.id));
  state.g2mDeephSelectedPlotRunIds = Array.from(new Set(ids || [])).filter((id) => visibleIds.has(id));
}

function g2mDeepHPlotsQuery() {
  if (state.g2mDeephSelectedPlotRunIds == null) return "";
  const params = new URLSearchParams();
  if (!state.g2mDeephSelectedPlotRunIds.length) {
    params.set("run_ids", "");
    return `?${params.toString()}`;
  }
  state.g2mDeephSelectedPlotRunIds.forEach((id) => params.append("run_id", id));
  return `?${params.toString()}`;
}

function renderG2MDeepHPlotRunSelector() {
  const status = document.getElementById("g2m-deeph-plot-run-status");
  const list = document.getElementById("g2m-deeph-plot-run-list");
  if (!status || !list) return;
  const runs = state.g2mDeephPlotRuns || [];
  const selected = new Set(state.g2mDeephSelectedPlotRunIds || []);
  list.textContent = "";
  if (!runs.length) {
    status.textContent = "No previous Graph2Mat/DeepH runs found yet.";
    return;
  }
  status.textContent = `${selected.size}/${runs.length} run(s) selected for plots. Active runs appear once the backend writes runner status.`;
  for (const run of runs) {
    const option = document.createElement("label");
    option.className = "plot-run-option";
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.className = "g2m-deeph-plot-run-checkbox";
    checkbox.value = run.id;
    checkbox.checked = selected.has(run.id);
    checkbox.addEventListener("change", () => {
      const next = new Set(state.g2mDeephSelectedPlotRunIds || []);
      if (checkbox.checked) next.add(run.id);
      else next.delete(run.id);
      setG2MDeepHSelectedPlotRuns(Array.from(next));
      renderG2MDeepHPlotRunSelector();
      loadG2MDeepHPlots().catch((error) => showToast(error.message));
    });
    const body = document.createElement("span");
    const title = document.createElement("strong");
    title.textContent = run.label || run.run_id || run.id;
    const details = document.createElement("span");
    const models = (run.models || []).map(methodDisplayLabel).join("+") || "no model";
    const datasets = (run.dataset_ids || []).join(", ") || "no dataset";
    const counts = `${run.completed_runs || 0}/${run.planned_runs || 0} done`;
    const time = formatG2MDeepHPlotRunTime(run.modified_at);
    details.textContent = [models, datasets, counts, run.status || "", time].filter(Boolean).join(" | ");
    body.appendChild(title);
    body.appendChild(details);
    option.appendChild(checkbox);
    option.appendChild(body);
    list.appendChild(option);
  }
}

async function loadG2MDeepHPlotRuns({ preserveSelection = true } = {}) {
  let payload;
  try {
    payload = await request("/api/g2m-deeph/plot-runs");
  } catch (error) {
    state.g2mDeephPlotRuns = [];
    state.g2mDeephDefaultPlotRunIds = [];
    state.g2mDeephSelectedPlotRunIds = [];
    const status = document.getElementById("g2m-deeph-plot-run-status");
    const list = document.getElementById("g2m-deeph-plot-run-list");
    if (status) status.textContent = "Run selector will be available after the UI backend is restarted.";
    if (list) list.textContent = "";
    return { runs: [], default_selected_run_ids: [], unavailable: true, error: error.message };
  }
  state.g2mDeephPlotRuns = payload.runs || [];
  state.g2mDeephDefaultPlotRunIds = payload.default_selected_run_ids || [];
  const visibleIds = new Set(state.g2mDeephPlotRuns.map((run) => run.id));
  const metricRunIds = new Set(
    state.g2mDeephPlotRuns
      .filter((run) => run.has_metric_rows)
      .map((run) => run.id),
  );
  const preservedSelection = (state.g2mDeephSelectedPlotRunIds || []).filter((id) => visibleIds.has(id));
  const preservedHasMetrics = preservedSelection.some((id) => metricRunIds.has(id));
  if (!preserveSelection || (preservedSelection.length && !preservedHasMetrics)) {
    const defaults = state.g2mDeephDefaultPlotRunIds.filter((id) => visibleIds.has(id));
    setG2MDeepHSelectedPlotRuns(defaults);
  } else {
    setG2MDeepHSelectedPlotRuns(state.g2mDeephSelectedPlotRunIds);
  }
  renderG2MDeepHPlotRunSelector();
  renderG2MDeepHDerivativeRunSelector();
  return payload;
}

function normalizeG2MDeepHMetricPlots(payload = {}) {
  const rows = dedupeMetricScalingRows(payload.metric_scaling_rows || []);
  if (!rows.length) {
    return {
      ...payload,
      metric_groups: g2mDeephReadableMetricGroups(),
    };
  }
  const readableGroups = g2mDeephReadableMetricGroups();
  const readableMetricPlots = [];
  const scaledMetricGroupIds = new Set();
  for (const group of readableGroups) {
    const metricKeys = new Set((group.metrics || []).map((metric) => metric.key));
    const groupRows = rows.filter((row) => metricKeys.has(row.metric_key));
    if (!groupRows.length) continue;
    scaledMetricGroupIds.add(group.id);
    readableMetricPlots.push({
      id: `metric_scaling_${group.id}`,
      kind: "metric_scaling",
      title: `${group.title} vs dataset size`,
      x_title: "Dataset size (snapshots)",
      y_title: group.y_title || "Metric value",
      metrics: group.metrics || [],
      rows: groupRows,
    });
  }
  const nonMetricPlots = (payload.plots || []).filter(
    (plot) => plot.kind !== "metric_scaling" && (plot.kind !== "grouped_bar" || !scaledMetricGroupIds.has(plot.id)),
  );
  return {
    ...payload,
    available: Boolean(payload.available || readableMetricPlots.length || nonMetricPlots.length),
    metric_groups: readableGroups,
    plots: [...readableMetricPlots, ...nonMetricPlots],
  };
}

function metricScalingRowKey(row = {}) {
  return [
    row.run_id || "",
    row.dataset_root || row.dataset_id || "",
    row.dataset_size || "",
    row.method || "",
    row.config_id || "",
    row.seed || "",
    g2mDeephEpochLabel(row),
    row.metric_key || "",
  ].join("|");
}

function metricScalingEquivalentValueKey(row = {}) {
  const value = finiteNumber(row.metric_value);
  return value == null ? "" : value.toPrecision(14);
}

function metricScalingDuplicateValueKey(row = {}) {
  return [
    row.run_id || "",
    row.dataset_root || row.dataset_id || "",
    row.dataset_size || "",
    row.method || "",
    row.config_id || "",
    g2mDeephEpochLabel(row),
    row.metric_key || "",
    metricScalingEquivalentValueKey(row),
  ].join("|");
}

function dedupeMetricScalingRows(rows = []) {
  const seen = new Set();
  const seenEquivalentValues = new Set();
  const result = [];
  for (const row of rows) {
    const key = metricScalingRowKey(row);
    const valueKey = metricScalingDuplicateValueKey(row);
    if (seen.has(key) || seenEquivalentValues.has(valueKey)) continue;
    seen.add(key);
    if (valueKey) seenEquivalentValues.add(valueKey);
    result.push(row);
  }
  return result;
}

function timingScalingRowKey(row = {}) {
  return [
    row.run_id || "",
    row.dataset_id || "",
    row.model || "",
    row.config_id || "",
    row.phase || "",
    row.epoch_label || "",
    row.source || "",
  ].join("|");
}

function dedupeTimingScalingRows(rows = []) {
  const seen = new Set();
  const result = [];
  for (const row of rows) {
    const key = timingScalingRowKey(row);
    if (seen.has(key)) continue;
    seen.add(key);
    result.push(row);
  }
  return result;
}

function mergeG2MDeepHLivePlotPayload(payload, livePayload) {
  const liveRows = livePayload?.metric_scaling_rows || [];
  const liveTimingRows = livePayload?.timing_scaling_rows || [];
  if (!liveRows.length && !liveTimingRows.length) return payload;
  const metricRows = dedupeMetricScalingRows([...(payload.metric_scaling_rows || []), ...liveRows]);
  const timingRows = dedupeTimingScalingRows([...(payload.timing_scaling_rows || []), ...liveTimingRows]);
  const primaryPlots = (payload.plots || []).filter((plot) => plot.kind !== "metric_scaling");
  const liveMetricPlots = (livePayload.plots || []).filter((plot) => plot.kind === "metric_scaling");
  const timingPlots = primaryPlots.map((plot) =>
    plot.kind === "timing_scaling" ? { ...plot, rows: timingRows } : plot,
  );
  return {
    ...payload,
    available: true,
    metric_scaling_rows: metricRows,
    timing_scaling_rows: timingRows,
    live_metric_rows: liveRows.length,
    live_timing_rows: liveTimingRows.length,
    live_metrics_source: livePayload.source || "sidecar",
    plots: [...liveMetricPlots, ...timingPlots],
    message: payload.message || livePayload.message,
  };
}

async function maybeLoadG2MDeepHLiveMetrics(payload) {
  return payload;
}

async function loadG2MDeepHPlots() {
  if (!window.Plotly) await ensurePlotlyLoaded();
  if (!state.g2mDeephPlotRuns?.length) {
    await loadG2MDeepHPlotRuns({ preserveSelection: true });
  }
  const payload = normalizeG2MDeepHMetricPlots(await request(`/api/g2m-deeph/plots${g2mDeepHPlotsQuery()}`));
  state.g2mDeephPlotPayload = payload;
  renderG2MDeepHMetricSummary({
    available: payload.available,
    results: { common_metrics: payload.common_metrics },
    plot_payload: payload,
    status: payload.status,
  });
  renderG2MDeepHWarnings({
    status: payload.status,
    validation: state.g2mDeephValidation,
    results: { common_metrics: payload.common_metrics },
  });
  renderG2MDeepHPlotsPayload(payload);
  return payload;
}

async function maybeRefreshG2MDeepHLivePlots(status = {}) {
  if (!status?.running || !state.g2mDeephSelectedPlotRunIds?.length || !state.plotsEnabled) return status;
  const now = Date.now();
  if (state.g2mDeephPlotsInFlight || now - state.g2mDeephLastPlotRefreshAt < G2M_DEEPH_LIVE_PLOT_REFRESH_MS) {
    return status;
  }
  state.g2mDeephPlotsInFlight = true;
  state.g2mDeephLastPlotRefreshAt = now;
  try {
    await loadG2MDeepHPlots();
  } catch (error) {
    const statusEl = document.getElementById("g2m-deeph-plot-run-status");
    if (statusEl) statusEl.textContent = `Plot refresh pending: ${error.message}`;
  } finally {
    state.g2mDeephPlotsInFlight = false;
  }
  return status;
}

async function pollG2MDeepHStatus() {
  const payload = await request("/api/g2m-deeph/status");
  updateG2MDeepHStatus(payload);
  return payload;
}

function scrollG2MDeepHLogToBottom() {
  const output = document.getElementById("g2m-deeph-log");
  if (output) output.scrollTop = output.scrollHeight;
}

function clearG2MDeepHLogView() {
  const output = document.getElementById("g2m-deeph-log");
  if (output) output.textContent = "";
}

async function pollG2MDeepHLogs() {
  const requestedSince = state.g2mDeephOffset;
  const payload = await request(`/api/g2m-deeph/logs?since=${requestedSince}&limit=${LOG_POLL_LIMIT}`);
  const currentRunId = payload.status?.run_id || null;
  if (
    (currentRunId && state.g2mDeephRunId && currentRunId !== state.g2mDeephRunId)
    || (Number.isFinite(payload.offset) && payload.offset < requestedSince)
  ) {
    clearG2MDeepHLogView();
    state.g2mDeephOffset = 0;
    state.g2mDeephRunId = currentRunId;
    return pollG2MDeepHLogs();
  }
  if (currentRunId) state.g2mDeephRunId = currentRunId;
  state.g2mDeephOffset = payload.offset;
  updateG2MDeepHStatus(payload.status || {});
  if (payload.lines.length) {
    const output = document.getElementById("g2m-deeph-log");
    output.textContent += payload.lines.join("");
    scrollG2MDeepHLogToBottom();
    terminalAppendBlock("g2m-deeph", payload.lines.join(""));
  }
  const wasRunning = state.g2mDeephWasRunning;
  state.g2mDeephWasRunning = Boolean(payload.status?.running);
  if (wasRunning && !payload.status?.running) {
    await loadG2MDeepHResults();
    await loadG2MDeepHPlotRuns();
    await loadG2MDeepHDerivativeMetrics();
  } else if (payload.status?.running) {
    await maybeRefreshG2MDeepHLivePlots(payload.status);
  }
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
    max_parallel_graph2mat_training_jobs:
      optionalPositiveInteger("performance-max-parallel-graph2mat-training-jobs", "Max Graph2Mat training jobs") || 1,
    max_parallel_deeph_training_jobs:
      optionalPositiveInteger("performance-max-parallel-deeph-training-jobs", "Max DeepH training jobs") || 1,
    omp_num_threads: optionalPositiveInteger("performance-omp-num-threads", "OMP threads"),
    mkl_num_threads: optionalPositiveInteger("performance-mkl-num-threads", "MKL threads"),
    openblas_num_threads: optionalPositiveInteger("performance-openblas-num-threads", "OpenBLAS threads"),
    numexpr_num_threads: optionalPositiveInteger("performance-numexpr-num-threads", "NumExpr threads"),
    torch_num_threads: optionalPositiveInteger("performance-torch-num-threads", "Torch threads"),
    compute_accelerator: accelerator,
    batch_size: optionalPositiveInteger("performance-batch-size", "Batch size override"),
    graph2mat_log_every_n_steps: optionalPositiveInteger(
      "performance-graph2mat-log-every-n-steps",
      "Graph2Mat log every steps",
    ),
    graph2mat_check_val_every_n_epoch: optionalPositiveInteger(
      "performance-graph2mat-check-val-every-n-epoch",
      "Graph2Mat validate every epochs",
    ),
    graph2mat_checkpoint_every_n_epochs: optionalPositiveInteger(
      "performance-graph2mat-checkpoint-every-n-epochs",
      "Graph2Mat checkpoint every epochs",
    ),
    graph2mat_require_cuequivariance: optionalBooleanSelect("performance-graph2mat-require-cuequivariance"),
    store_in_memory: optionalBooleanSelect("performance-store-in-memory"),
    reuse_validated_siesta_outputs: optionalBooleanSelect("performance-reuse-validated-siesta-outputs"),
    enable_experiment_cache: optionalBooleanSelect("performance-enable-experiment-cache"),
    error_policy: document.getElementById("performance-error-policy")?.value || "fail_fast",
    preset: document.getElementById("performance-preset")?.value || null,
    torch_float32_matmul_precision:
      document.getElementById("performance-torch-float32-matmul-precision")?.value || null,
    torch_mixed_precision:
      document.getElementById("performance-torch-mixed-precision")?.value || null,
  };
}

function optionalTextInput(id) {
  const value = inputValue(id);
  return value ? value : null;
}

function positiveFloatInput(id, label, fallback) {
  const raw = inputValue(id);
  if (!raw && fallback != null) return fallback;
  const value = Number(raw);
  if (!Number.isFinite(value) || value <= 0) {
    throw new Error(`${label} debe ser un numero positivo.`);
  }
  return value;
}

function nonNegativeIntegerInput(id, label, fallback) {
  const raw = inputValue(id);
  if (!raw && fallback != null) return fallback;
  const value = Number(raw);
  if (!Number.isInteger(value) || value < 0) {
    throw new Error(`${label} debe ser un entero >= 0.`);
  }
  return value;
}

function deephComparisonPayload() {
  return {
    graph2mat_result_dir: optionalTextInput("deeph-graph2mat-result-dir"),
    graph2mat_result_dirs: optionalTextInput("deeph-graph2mat-result-dirs"),
    graph2mat_candidate_summary_csv: optionalTextInput("deeph-graph2mat-candidate-summary"),
    deeph_repo: optionalTextInput("deeph-repo"),
    deeph_python: optionalTextInput("deeph-python"),
    siesta_command: optionalTextInput("deeph-siesta-command") || "siesta",
    epochs: optionalPositiveInteger("deeph-epochs", "DeepH epochs") || 200,
    batch_size: optionalPositiveInteger("deeph-batch-size", "DeepH batch size") || 4,
    learning_rate: positiveFloatInput("deeph-learning-rate", "DeepH learning rate", 0.001),
    sample_limit_per_split: optionalPositiveInteger("deeph-sample-limit-per-split", "Sample limit per split"),
    seed: nonNegativeIntegerInput("deeph-seed", "DeepH seed", 0),
    device: optionalTextInput("deeph-device") || "cuda:0",
    graph2mat_top_count: optionalPositiveInteger("deeph-graph2mat-top-count", "Graph2Mat top count"),
    allow_regenerate_siesta: Boolean(document.getElementById("deeph-allow-regenerate-siesta")?.checked),
  };
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
    loss_kwargs: optionalJsonObjectInput("training-loss-kwargs", "Loss kwargs JSON"),
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
  return entries.map(([key, value]) => {
    const text = value && typeof value === "object" ? JSON.stringify(value) : value;
    return `${key}=${text}`;
  }).join(", ");
}

function splitSweepList(value) {
  return String(value || "")
    .split(/[\n,;]+/)
    .map((item) => item.trim())
    .filter(Boolean);
}

function parseSweepNumberList(id, label, { integer = false, min = null } = {}) {
  return splitSweepList(inputValue(id)).map((item) => {
    const value = Number(item);
    if (!Number.isFinite(value) || (integer && !Number.isInteger(value)) || (min != null && value < min)) {
      const type = integer ? "entero" : "numero";
      const floor = min == null ? "" : ` >= ${min}`;
      throw new Error(`${label}: "${item}" debe ser un ${type}${floor}.`);
    }
    return value;
  });
}

function parseSweepTextList(id) {
  return splitSweepList(inputValue(id));
}

function parseJsonObject(raw, label) {
  const text = String(raw || "").trim();
  if (!text) return null;
  let parsed;
  try {
    parsed = JSON.parse(text);
  } catch (error) {
    throw new Error(`${label}: JSON invalido.`);
  }
  if (!parsed || Array.isArray(parsed) || typeof parsed !== "object") {
    throw new Error(`${label}: debe ser un objeto JSON.`);
  }
  return parsed;
}

function optionalJsonObjectInput(id, label) {
  return parseJsonObject(inputValue(id), label);
}

function parseSweepJsonObjectList(id, label) {
  return String(inputValue(id) || "")
    .split(/\n+/)
    .map((item) => item.trim())
    .filter(Boolean)
    .map((item) => parseJsonObject(item, label));
}

function hiddenIrrepsDimension(raw) {
  const terms = parseHiddenIrrepsTerms(raw);
  return terms.reduce((total, term) => total + term.mul * (2 * term.ell + 1), 0);
}

function sweepParametersFromControls() {
  const parameters = {};
  const setIfAny = (key, values) => {
    if (values.length) parameters[key] = [...new Set(values)];
  };
  setIfAny("max_epochs", parseSweepNumberList("sweep-max-epochs", "Sweep epochs", { integer: true, min: 1 }));
  setIfAny("optim_lr", parseSweepNumberList("sweep-optim-lr", "Sweep learning rate", { min: 0.000001 }));
  setIfAny("batch_size", parseSweepNumberList("sweep-batch-size", "Sweep batch size", { integer: true, min: 1 }));
  setIfAny("loader_threads", parseSweepNumberList("sweep-loader-threads", "Sweep loader threads", { integer: true, min: 1 }));
  setIfAny("loss", parseSweepTextList("sweep-loss"));
  setIfAny("loss_kwargs", parseSweepJsonObjectList("sweep-loss-kwargs", "Sweep loss kwargs"));
  setIfAny(
    "num_interactions",
    parseSweepNumberList("sweep-num-interactions", "Sweep interactions", { integer: true, min: 1 }),
  );
  setIfAny("correlation", parseSweepNumberList("sweep-correlation", "Sweep correlation", { integer: true, min: 1 }));
  setIfAny("max_ell", parseSweepNumberList("sweep-max-ell", "Sweep max ell", { integer: true, min: 0 }));
  setIfAny("hidden_irreps", parseSweepTextList("sweep-hidden-irreps"));
  setIfAny(
    "hidden_irreps_channels",
    parseSweepNumberList("sweep-hidden-irreps-channels", "Sweep hidden irreps channels", {
      integer: true,
      min: 1,
    }),
  );
  if (parameters.hidden_irreps?.length && parameters.hidden_irreps_channels?.length) {
    throw new Error("Usa Hidden irreps list o Hidden irreps channels, no ambos.");
  }
  return parameters;
}

function sweepParameterSignature(parameters) {
  return JSON.stringify(parameters || {});
}

function sortedSweepExcludedIndices() {
  return [...state.sweepExcludedIndices].sort((a, b) => a - b);
}

function syncSweepExclusionsForParameters(parameters) {
  const signature = sweepParameterSignature(parameters);
  if (state.sweepPreviewSignature !== signature) {
    state.sweepExcludedIndices = new Set();
    state.sweepPreviewPage = 1;
    state.sweepPreviewSignature = signature;
  }
  return sortedSweepExcludedIndices();
}

function hyperparameterSweepPayload({ includeTargets = false } = {}) {
  const enabled = Boolean(document.getElementById("sweep-enabled")?.checked);
  if (!enabled) return { enabled: false };
  if (state.trainingPlan.length) {
    throw new Error("No mezcles Hyperparameter sweep con Training plan manual.");
  }
  const maxConfigs = optionalPositiveInteger("sweep-max-configs", "Sweep max configs") || 256;
  const parameters = sweepParametersFromControls();
  if (!Object.keys(parameters).length) {
    throw new Error("Hyperparameter sweep necesita al menos una lista de valores.");
  }
  const excludedIndices = syncSweepExclusionsForParameters(parameters);
  const payload = {
    enabled: true,
    mode: "cartesian",
    label_prefix: optionalTextInput("sweep-label-prefix") || "sweep",
    max_configs: maxConfigs,
    parameters,
  };
  if (excludedIndices.length) payload.excluded_indices = excludedIndices;
  if (includeTargets) {
    const runMode = document.getElementById("run-mode")?.value;
    if (runMode === "dataset_only") {
      throw new Error("Hyperparameter sweep no esta disponible con dataset_only.");
    }
    if (["train_test_metrics_plots_only", "graph2mat_deeph_comparison"].includes(runMode)) {
      const ids = selectedReusableDatasetIds();
      if (!ids.length) {
        throw new Error("Selecciona al menos un dataset reusable para el sweep.");
      }
      payload.reusable_dataset_ids = ids;
    }
    if (runMode === "full_strict_pipeline") {
      const targets = selectedPlannedDatasetTargets();
      if (!targets.length) {
        throw new Error("Selecciona al menos un planned dataset para el sweep.");
      }
      payload.dataset_targets = targets;
    }
  }
  return payload;
}

function sweepConfigCount(parameters) {
  return Object.values(parameters || {}).reduce((total, values) => total * Math.max(1, values.length), 1);
}

function sweepPreviewLabel(index, settings, sweepParameters, prefix) {
  const parts = [`${prefix}${String(index).padStart(3, "0")}`];
  if ("max_epochs" in sweepParameters) parts.push(`ep${settings.max_epochs}`);
  if ("optim_lr" in sweepParameters) parts.push(`lr${slugPart(settings.optim_lr)}`);
  if ("max_ell" in sweepParameters) parts.push(`l${settings.max_ell}`);
  if ("hidden_irreps_channels" in sweepParameters) parts.push(`c${sweepParameters.hidden_irreps_channels}`);
  if ("num_interactions" in sweepParameters) parts.push(`i${settings.num_interactions}`);
  if ("correlation" in sweepParameters) parts.push(`corr${settings.correlation}`);
  if ("batch_size" in sweepParameters) parts.push(`b${settings.batch_size}`);
  if ("loader_threads" in sweepParameters) parts.push(`w${settings.loader_threads}`);
  return parts.join("_");
}

function expandSweepPreview(payload, baseSettings) {
  const parameters = payload.parameters || {};
  const keys = Object.keys(parameters).sort();
  const count = sweepConfigCount(parameters);
  const excludedSet = new Set((payload.excluded_indices || []).map((value) => Number(value)));
  const outOfRange = [...excludedSet].filter((index) => index > count);
  if (outOfRange.length) {
    throw new Error(`Exclusiones de sweep fuera de rango: ${outOfRange.slice(0, 10).join(", ")}.`);
  }
  const activeCount = count - excludedSet.size;
  if (activeCount <= 0) {
    return { count, activeCount, excludedCount: excludedSet.size, rows: [], warning: "Todas las configuraciones del sweep estan excluidas." };
  }
  if (activeCount > payload.max_configs) {
    return {
      count,
      activeCount,
      excludedCount: excludedSet.size,
      rows: [],
      warning: `El sweep genera ${activeCount} configuraciones activas (${count} totales, ${excludedSet.size} excluidas); max_configs=${payload.max_configs}.`,
    };
  }
  const rows = [];
  const combos = cartesianProduct(keys.map((key) => parameters[key]));
  for (const [zeroIndex, combo] of combos.entries()) {
    const sweepParameters = Object.fromEntries(combo.map((value, i) => [keys[i], value]));
    const settings = { ...baseSettings };
    for (const [key, value] of Object.entries(sweepParameters)) {
      if (key !== "hidden_irreps_channels") settings[key] = value;
    }
    if ("hidden_irreps_channels" in sweepParameters) {
      const maxEll = settings.max_ell;
      if (!Number.isInteger(Number(maxEll)) || Number(maxEll) < 0) {
        throw new Error("hidden_irreps_channels requiere Max ell en el sweep o en los controles.");
      }
      settings.hidden_irreps = expectedIrrepsText(Number(sweepParameters.hidden_irreps_channels), Number(maxEll));
    }
    let dimension = "";
    if (settings.hidden_irreps) {
      validateHiddenIrrepsText(settings.hidden_irreps, settings.max_ell);
      dimension = hiddenIrrepsDimension(settings.hidden_irreps);
    }
    rows.push({
      index: zeroIndex + 1,
      label: sweepPreviewLabel(zeroIndex + 1, settings, sweepParameters, payload.label_prefix || "sweep"),
      settings,
      sweepParameters,
      dimension,
      excluded: excludedSet.has(zeroIndex + 1),
    });
  }
  return { count, activeCount, excludedCount: excludedSet.size, rows };
}

function renderHyperparameterSweepPreview() {
  const status = document.getElementById("sweep-status");
  const body = document.getElementById("sweep-preview-list");
  const nav = document.getElementById("sweep-preview-nav");
  const pageLabel = document.getElementById("sweep-preview-page");
  if (!status || !body) return;
  body.innerHTML = "";
  if (nav) nav.classList.add("hidden");
  try {
    const payload = hyperparameterSweepPayload();
    if (!payload.enabled) {
      status.textContent = "Sweep disabled.";
      state.sweepPreviewPage = 1;
      return;
    }
    const preview = expandSweepPreview(payload, trainingSettings());
    if (preview.warning) {
      status.textContent = preview.warning;
      return;
    }
    const totalPages = Math.max(1, Math.ceil(preview.rows.length / SWEEP_PREVIEW_PAGE_SIZE));
    state.sweepPreviewPage = Math.min(Math.max(1, state.sweepPreviewPage), totalPages);
    const start = (state.sweepPreviewPage - 1) * SWEEP_PREVIEW_PAGE_SIZE;
    const pageRows = preview.rows.slice(start, start + SWEEP_PREVIEW_PAGE_SIZE);
    status.textContent = `${preview.activeCount} active / ${preview.count} total configuration${preview.count === 1 ? "" : "s"}; ${preview.excludedCount} excluded. Uncheck Run to exclude a combo.`;
    if (nav) nav.classList.toggle("hidden", totalPages <= 1);
    if (pageLabel) pageLabel.textContent = `Page ${state.sweepPreviewPage} / ${totalPages}`;
    for (const row of pageRows) {
      const tr = document.createElement("tr");
      const runCell = document.createElement("td");
      const includeCheckbox = document.createElement("input");
      includeCheckbox.type = "checkbox";
      includeCheckbox.checked = !row.excluded;
      includeCheckbox.dataset.sweepIncludeIndex = String(row.index);
      includeCheckbox.setAttribute("aria-label", `Run sweep configuration ${row.index}`);
      runCell.appendChild(includeCheckbox);
      const indexCell = document.createElement("td");
      indexCell.textContent = String(row.index);
      const labelCell = document.createElement("td");
      labelCell.textContent = row.label;
      const overridesCell = document.createElement("td");
      overridesCell.textContent = trainingSettingsSummary(row.settings);
      const dimCell = document.createElement("td");
      dimCell.textContent = row.dimension ? String(row.dimension) : "-";
      tr.append(runCell, indexCell, labelCell, overridesCell, dimCell);
      body.appendChild(tr);
    }
    body.querySelectorAll("[data-sweep-include-index]").forEach((checkbox) => {
      checkbox.addEventListener("change", () => {
        const index = Number(checkbox.getAttribute("data-sweep-include-index"));
        if (!Number.isInteger(index) || index <= 0) return;
        if (checkbox.checked) {
          state.sweepExcludedIndices.delete(index);
        } else {
          state.sweepExcludedIndices.add(index);
        }
        renderHyperparameterSweepPreview();
      });
    });
    document.getElementById("sweep-preview-first")?.toggleAttribute("disabled", state.sweepPreviewPage <= 1);
    document.getElementById("sweep-preview-prev")?.toggleAttribute("disabled", state.sweepPreviewPage <= 1);
    document.getElementById("sweep-preview-next")?.toggleAttribute("disabled", state.sweepPreviewPage >= totalPages);
    document.getElementById("sweep-preview-last")?.toggleAttribute("disabled", state.sweepPreviewPage >= totalPages);
  } catch (error) {
    status.textContent = error.message;
  }
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
  const plannedTargetModes = ["full_strict_pipeline"];
  const reusableDatasetModes = ["train_test_metrics_plots_only", "graph2mat_deeph_comparison"];
  if (![...plannedTargetModes, ...reusableDatasetModes].includes(runMode)) {
    throw new Error("Training plan solo esta disponible en Full strict, modo combinado o Train/test/metrics/plots only.");
  }
  const datasetIds = reusableDatasetModes.includes(runMode) ? selectedReusableDatasetIds() : [];
  const datasetTargets = plannedTargetModes.includes(runMode) ? selectedPlannedDatasetTargets() : [];
  if (reusableDatasetModes.includes(runMode) && !datasetIds.length) {
    throw new Error("Selecciona al menos un dataset reusable para esta configuracion.");
  }
  if (plannedTargetModes.includes(runMode) && !datasetTargets.length) {
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
  const enabled = ["full_strict_pipeline", "graph2mat_deeph_comparison", "train_test_metrics_plots_only"].includes(runMode);
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
  max_parallel_graph2mat_training_jobs: "performance-max-parallel-graph2mat-training-jobs",
  max_parallel_deeph_training_jobs: "performance-max-parallel-deeph-training-jobs",
  omp_num_threads: "performance-omp-num-threads",
  mkl_num_threads: "performance-mkl-num-threads",
  openblas_num_threads: "performance-openblas-num-threads",
  numexpr_num_threads: "performance-numexpr-num-threads",
  torch_num_threads: "performance-torch-num-threads",
  compute_accelerator: "performance-compute-accelerator",
  batch_size: "performance-batch-size",
  graph2mat_log_every_n_steps: "performance-graph2mat-log-every-n-steps",
  graph2mat_check_val_every_n_epoch: "performance-graph2mat-check-val-every-n-epoch",
  graph2mat_checkpoint_every_n_epochs: "performance-graph2mat-checkpoint-every-n-epochs",
  graph2mat_require_cuequivariance: "performance-graph2mat-require-cuequivariance",
  store_in_memory: "performance-store-in-memory",
  reuse_validated_siesta_outputs: "performance-reuse-validated-siesta-outputs",
  enable_experiment_cache: "performance-enable-experiment-cache",
  error_policy: "performance-error-policy",
  torch_float32_matmul_precision: "performance-torch-float32-matmul-precision",
  torch_mixed_precision: "performance-torch-mixed-precision",
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
  g2m_deeph_md: {
    label: "Graph2Mat vs DeepH MD",
    containerId: "g2m-deeph-md-dataset-editor",
    sourceId: "g2m-deeph-md-sweep-table",
    addDatasetId: "g2m-deeph-md-add-dataset",
    countLabel: "Snapshots",
    valueLabel: "Temperatura (K)",
    defaultCount: "600",
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

function isMdDatasetEditorKind(kind) {
  return kind === "md" || kind === "g2m_deeph_md";
}

function sourceTextForKind(kind) {
  const config = datasetEditorConfig(kind);
  return String(document.getElementById(config?.sourceId)?.value || "");
}

function blocksForEditorSpec(kind, spec) {
  if (isMdDatasetEditorKind(kind)) {
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
    if (isMdDatasetEditorKind(kind)) return parseMdDatasetTableSpecsFromText(sourceTextForKind(kind));
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
  if (isMdDatasetEditorKind(kind)) {
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
  if (kind === "g2m_deeph_md") renderG2MDeepHDatasetSweepPreview();
  else if (kind === "fc") updateAtomSizesFromFcPlan();
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

function parseJsonObject(value) {
  if (value && typeof value === "object" && !Array.isArray(value)) return value;
  if (typeof value !== "string" || !value.trim()) return {};
  try {
    const parsed = JSON.parse(value);
    return parsed && typeof parsed === "object" && !Array.isArray(parsed) ? parsed : {};
  } catch {
    return {};
  }
}

function isKnownMaterialLabel(label) {
  const text = String(label || "").trim();
  return Boolean(text) && text !== UNKNOWN_MATERIAL_LABEL;
}

function materialDisplayLabel(item) {
  if (typeof item === "string") return item.trim() || UNKNOWN_MATERIAL_LABEL;
  const label = item?.material_display_label || item?.material_label || item?.material_preset || "";
  if (!isKnownMaterialLabel(label)) return UNKNOWN_MATERIAL_LABEL;
  return String(label);
}

function runMaterialLabel(run) {
  return materialDisplayLabel(run);
}

function rowMaterialLabels(row) {
  const labels = new Set();
  const addLabel = (value) => {
    if (isKnownMaterialLabel(value)) labels.add(String(value));
  };
  addLabel(row?.material_display_label);
  addLabel(row?.material_label);
  for (const value of Object.values(parseJsonObject(row?.material_label_by_method))) {
    addLabel(value);
  }
  if (!labels.size) labels.add(UNKNOWN_MATERIAL_LABEL);
  return Array.from(labels).sort();
}

function rowMaterialDisplayLabel(row) {
  const labels = rowMaterialLabels(row);
  if (labels.length <= 2) return labels.join(", ");
  return `${labels.slice(0, 2).join(", ")} +${labels.length - 2}`;
}

function materialContextText(item) {
  const label = materialDisplayLabel(item);
  const structureType = typeof item === "object" ? item?.material_structure_type : "";
  const structure = isKnownMaterialLabel(label) && structureType ? ` (${structureType})` : "";
  return `material ${label}${structure}`;
}

function metricSpaceLabel(run) {
  const space = String(run?.metric_space || "").trim();
  if (space === "kpoint_sampled" || run?.kpoint_metrics_enabled) {
    const mesh = Array.isArray(run?.kpoint_mesh) ? run.kpoint_mesh.join("x") : "";
    return mesh ? `k-point ${mesh}` : "k-point";
  }
  if (space === "gamma_only") return "gamma-only";
  return "";
}

function metricContextText(run) {
  const label = metricSpaceLabel(run);
  if (!label) return "";
  if (run?.kpoint_metrics_enabled) {
    const overlap = run?.uses_reference_overlap_k ? "S_ref(k)" : "overlap unknown";
    const count = run?.kpoint_count ? `${run.kpoint_count} k-points` : "";
    return [label, count, overlap].filter(Boolean).join(" · ");
  }
  return label;
}

function predictionSafetyStatus(run) {
  const explicitStatus = String(run?.prediction_artifact_safety_status || "").trim();
  if (explicitStatus) return explicitStatus;
  if (run?.prediction_artifacts_standalone_safe === true) return "safe";
  if (run?.prediction_artifacts_standalone_safe === false) return "unsafe";
  const unsafeSamples = finiteNumber(run?.prediction_self_contained_hsx_unsafe_samples);
  if (unsafeSamples != null && unsafeSamples > 0) return "unsafe";
  const safeSamples = finiteNumber(run?.prediction_self_contained_hsx_safe_samples);
  if (safeSamples != null && safeSamples > 0) return "safe";
  return "unknown";
}

function runHasSevereWarnings(run) {
  const severeCount = finiteNumber(run?.severe_warning_count);
  if (severeCount != null && severeCount > 0) return true;
  const severeKinds = run?.severe_warning_kinds;
  if (Array.isArray(severeKinds) && severeKinds.length > 0) return true;
  const diagnostics = run?.diagnostics || {};
  const errors = diagnostics.errors || [];
  return Array.isArray(errors) && errors.length > 0;
}

function runScientificSafetyStatus(run) {
  const scientificStatus = String(run?.scientific_status || run?.summary?.scientific_status || "").trim();
  const predictionStatus = predictionSafetyStatus(run);
  if (predictionStatus === "unsafe" || runHasSevereWarnings(run)) return "unsafe";
  if (scientificStatus === "non_comparative") return "non_comparative";
  if (scientificStatus && scientificStatus !== "robust_comparison" && scientificStatus !== "analysis_completed") {
    return "exploratory";
  }
  if (predictionStatus === "unknown" || String(run?.target_component_policy || "unknown") === "unknown") {
    return "unknown";
  }
  return "safe";
}

function safetyStatusLabel(status) {
  return ({
    safe: "safe",
    unsafe: "unsafe/severe",
    unknown: "unknown safety",
    exploratory: "exploratory",
    non_comparative: "non-comparative",
  }[status] || status || "unknown safety");
}

function runSafetyContextText(run) {
  const policy = run?.target_component_policy || "target unknown";
  const components = run?.n_matrix_components != null ? `${run.n_matrix_components} component(s)` : "components unknown";
  const overlap = run?.overlap_source || (run?.uses_reference_overlap_k ? "siesta_reference" : "overlap unknown");
  const prediction = predictionSafetyStatus(run);
  const severe = finiteNumber(run?.severe_warning_count);
  const severeText = severe != null && severe > 0 ? `${severe} severe warning(s)` : "";
  return [
    `target ${policy}`,
    components,
    `overlap ${overlap}`,
    `prediction HSX ${prediction}`,
    safetyStatusLabel(runScientificSafetyStatus(run)),
    severeText,
  ].filter(Boolean).join(" · ");
}

function selectedPlotSafety() {
  return document.getElementById("plot-safety-filter")?.value || "all";
}

function syncSafetyFilterOptions(payload) {
  const select = document.getElementById("plot-safety-filter");
  if (!select) return;
  const previous = select.value || "all";
  const counts = new Map();
  for (const run of payload?.runs || []) {
    const status = runScientificSafetyStatus(run);
    counts.set(status, (counts.get(status) || 0) + 1);
  }
  select.replaceChildren();
  const all = document.createElement("option");
  all.value = "all";
  all.textContent = "All safety states";
  select.appendChild(all);
  for (const status of ["safe", "unsafe", "exploratory", "non_comparative", "unknown"]) {
    const count = counts.get(status) || 0;
    if (!count) continue;
    const option = document.createElement("option");
    option.value = status;
    option.textContent = `${safetyStatusLabel(status)} (${count})`;
    select.appendChild(option);
  }
  const available = new Set(["all", ...Array.from(counts.keys())]);
  select.value = available.has(previous) ? previous : "all";
}

function filterRunsBySafety(runs) {
  const selection = selectedPlotSafety();
  if (selection === "all") return runs || [];
  return (runs || []).filter((run) => runScientificSafetyStatus(run) === selection);
}

function rowMaterialContextText(row) {
  return `material ${rowMaterialDisplayLabel(row)}`;
}

function availableMaterialLabels(payload) {
  const labels = new Set();
  let hasUnknown = false;
  const addLabel = (value) => {
    if (isKnownMaterialLabel(value)) labels.add(String(value));
    else hasUnknown = true;
  };
  for (const run of payload?.runs || []) addLabel(runMaterialLabel(run));
  for (const experiment of payload?.cross_experiments || []) {
    for (const label of experiment?.material_summary?.material_display_labels || []) addLabel(label);
    for (const row of experiment?.metrics || []) {
      for (const label of rowMaterialLabels(row)) addLabel(label);
    }
  }
  const ordered = Array.from(labels).sort((a, b) => a.localeCompare(b));
  if (hasUnknown) ordered.push(UNKNOWN_MATERIAL_LABEL);
  return ordered;
}

function selectedPlotMaterial() {
  return document.getElementById("plot-material-filter")?.value || "all";
}

const RUN_FAMILY_OPTIONS = [
  { id: "graphene_fair_deeph", label: "Graphene fair DeepH" },
  { id: "graphene_w90_md1000", label: "Graphene W90 MD1000" },
  { id: "graphene_w90_kpoint", label: "Graphene W90 k-point" },
  { id: "h2o", label: "H2O" },
  { id: "debug_partial", label: "Debug / partial" },
  { id: "other", label: "Other" },
];

function runSearchText(run) {
  return [
    run?.pipeline,
    run?.method_id,
    run?.dataset_label,
    run?.training_tag,
    run?.training_plan_label,
    run?.training_plan_display_label,
    run?.result_dir,
    run?.run_id,
  ]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();
}

function runFamilyId(run) {
  const text = runSearchText(run);
  const material = runMaterialLabel(run).toLowerCase();
  const failed = Number(run?.sample_row_counts?.kpoint_matrix?.total || 0) === 0 &&
    (String(run?.diagnostics?.severity || "").toLowerCase() === "error" || text.includes("partial"));
  if (run?.pipeline === "deeph_comparison" || text.includes("graph2mat_vs_deeph")) return "graphene_fair_deeph";
  if (material === "h2o" || text.includes("h2o")) return "h2o";
  if (material === "graphene" || text.includes("graphene")) {
    if (text.includes("md1000") || text.includes("xuqnco")) return "graphene_w90_md1000";
    if (text.includes("w90") || run?.metric_space === "kpoint_sampled" || run?.kpoint_metrics_enabled) return "graphene_w90_kpoint";
  }
  if (failed || text.includes("smoke") || text.includes("dryrun") || text.includes("unsupported_kpoint")) return "debug_partial";
  return "other";
}

function runFamilyLabel(familyId) {
  return RUN_FAMILY_OPTIONS.find((item) => item.id === familyId)?.label || "Other";
}

function selectedRunFamily() {
  return document.getElementById("plot-family-filter")?.value || "all";
}

function syncRunFamilyFilterOptions(payload) {
  const select = document.getElementById("plot-family-filter");
  if (!select) return;
  const previous = select.value || "all";
  const counts = new Map();
  for (const run of payload?.runs || []) {
    const id = runFamilyId(run);
    counts.set(id, (counts.get(id) || 0) + 1);
  }
  select.replaceChildren();
  const all = document.createElement("option");
  all.value = "all";
  all.textContent = "All run families";
  select.appendChild(all);
  for (const optionDef of RUN_FAMILY_OPTIONS) {
    const count = counts.get(optionDef.id) || 0;
    if (!count) continue;
    const option = document.createElement("option");
    option.value = optionDef.id;
    option.textContent = `${optionDef.label} (${count})`;
    select.appendChild(option);
  }
  const available = new Set(["all", ...Array.from(counts.keys())]);
  select.value = available.has(previous) ? previous : "all";
}

function filterRunsByFamily(runs) {
  const selection = selectedRunFamily();
  if (selection === "all") return runs || [];
  return (runs || []).filter((run) => runFamilyId(run) === selection);
}

function materialSelectionMatches(labels, selection = selectedPlotMaterial()) {
  if (selection === "all") return true;
  return labels.includes(selection);
}

function filterRunsByMaterial(runs) {
  const selection = selectedPlotMaterial();
  if (selection === "all") return runs || [];
  return (runs || []).filter((run) => materialSelectionMatches([runMaterialLabel(run)], selection));
}

function filterCrossExperimentByMaterial(experiment) {
  const selection = selectedPlotMaterial();
  if (!experiment || selection === "all") return experiment;
  const metrics = (experiment.metrics || []).filter((row) =>
    materialSelectionMatches(rowMaterialLabels(row), selection),
  );
  const hashes = new Set(
    metrics
      .map((row) => row.material_compatibility_hash || Object.values(parseJsonObject(row.material_compatibility_hash_by_method)).sort().join(","))
      .filter(Boolean),
  );
  return {
    ...experiment,
    metrics,
    material_filter: selection,
    material_summary: {
      ...(experiment.material_summary || {}),
      material_display_labels: [selection],
      material_compatibility_hashes: Array.from(hashes).sort(),
      mixed_materials: hashes.size > 1,
    },
  };
}

function syncMaterialFilterOptions(payload) {
  const select = document.getElementById("plot-material-filter");
  if (!select) return;
  const previous = select.value || "all";
  const labels = availableMaterialLabels(payload);
  select.replaceChildren();
  const all = document.createElement("option");
  all.value = "all";
  all.textContent = "All materials";
  select.appendChild(all);
  for (const label of labels) {
    const option = document.createElement("option");
    option.value = label;
    option.textContent = label;
    select.appendChild(option);
  }
  select.value = previous === "all" || labels.includes(previous) ? previous : "all";
}

function runDisplayLabel(run) {
  const detail = run?.training_tag || run?.dataset_label || run?.recipe_id || run?.run_id || "";
  const size = run?.dataset_size ?? "";
  const material = runMaterialLabel(run);
  return [
    `${pipelineLabel(run?.pipeline || run?.label)} ${size}`.trim(),
    isKnownMaterialLabel(material) ? material : "",
    metricSpaceLabel(run),
    detail,
  ].filter(Boolean).join(" · ");
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
    const detailRows = [];
    if (result.predicted_hamiltonians !== undefined && result.predicted_hamiltonians !== "") {
      detailRows.push(`<span>${result.predicted_hamiltonians} predicted Hamiltonians</span>`);
    }
    if (result.siesta_hamiltonians !== undefined && result.siesta_hamiltonians !== "") {
      detailRows.push(`<span>${result.siesta_hamiltonians} SIESTA Hamiltonians</span>`);
    }
    if (result.comparison_report) {
      detailRows.push(`<span>Report: <code>${result.comparison_report}</code></span>`);
    }
    if (result.aggregate_csv) {
      detailRows.push(`<span>Aggregate: <code>${result.aggregate_csv}</code></span>`);
    }
    const safetyStatus = runScientificSafetyStatus(result);
    detailRows.push(`<span>Scientific status: <strong>${safetyStatusLabel(safetyStatus)}</strong></span>`);
    detailRows.push(
      `<span>Target: ${result.target_component_policy || "unknown"} / ` +
      `${result.n_matrix_components ?? "?"} component(s)</span>`,
    );
    detailRows.push(`<span>Overlap: ${result.overlap_source || "unknown"}</span>`);
    detailRows.push(`<span>Prediction HSX standalone: ${predictionSafetyStatus(result)}</span>`);
    const severeCount = finiteNumber(result.severe_warning_count);
    if (severeCount != null && severeCount > 0) {
      const kinds = Array.isArray(result.severe_warning_kinds) && result.severe_warning_kinds.length
        ? ` (${result.severe_warning_kinds.slice(0, 4).join(", ")})`
        : "";
      detailRows.push(`<span>Severe warnings: ${severeCount}${kinds}</span>`);
    }
    item.innerHTML = `
      <strong>${pipelineLabel(result.pipeline)} ${label}</strong>
      ${detailRows.join("")}
      <code>${result.result_dir}</code>
    `;
    container.appendChild(item);
  }
}

async function runExperiment() {
  const runMode = document.getElementById("run-mode").value;
  if (runMode === "deeph_comparison") {
    state.experimentOffset = 0;
    document.getElementById("experiment-log").textContent = "";
    const payload = await request("/api/experiment", {
      method: "POST",
      body: JSON.stringify({
        run_mode: runMode,
        deeph_comparison: deephComparisonPayload(),
      }),
    });
    updateExperimentStatus(payload);
    showToast("DeepH comparison started");
    return;
  }
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
  if (runMode === "full_strict_pipeline") {
    state.datasetTargets = plannedDatasetTargetsFromRecipes(datasetRecipes, methods);
    renderPlannedDatasetTargets(state.datasetTargets);
  }
  const sweep = hyperparameterSweepPayload({ includeTargets: true });
  const plan = !sweep.enabled && ["full_strict_pipeline", "graph2mat_deeph_comparison", "train_test_metrics_plots_only"].includes(runMode)
    ? trainingPlanPayload()
    : [];
  const reusableDatasetIds = ["train_test_metrics_plots_only", "graph2mat_deeph_comparison"].includes(runMode)
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
      hyperparameter_sweep: sweep,
      deeph_comparison: runMode === "graph2mat_deeph_comparison" ? deephComparisonPayload() : undefined,
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
    terminalAppendBlock("experiment", payload.lines.join(""));
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
    const kpointItems = items.filter((item) => item?.diagnostic_outputs?.kpoint_spectral_metrics?.exists);
    const safetyCounts = items.reduce((counts, item) => {
      const status = runScientificSafetyStatus(item);
      counts[status] = (counts[status] || 0) + 1;
      return counts;
    }, {});
    const safetyText = Object.entries(safetyCounts)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([status, count]) => `${safetyStatusLabel(status)}: ${count}`)
      .join(" · ") || "none";
    const orbitalPairPath = orbitalPairItems[0]?.diagnostic_outputs?.orbital_pair_metrics?.path ||
      `Comparison/results/${pipeline.resultsDir}/.../metrics/orbital_pair_metrics.csv`;
    const kpointPath = kpointItems[0]?.diagnostic_outputs?.kpoint_spectral_metrics?.path ||
      `Comparison/results/${pipeline.resultsDir}/.../metrics/kpoint_spectral_metrics.csv`;
    const panel = document.createElement("section");
    panel.className = "panel result-row";
    panel.innerHTML = `
      <div>
        <p class="eyebrow">Archived</p>
        <h3>${pipeline.label}</h3>
      </div>
      <p><strong>${items.length}</strong> archived experiment runs</p>
      <p><strong>Orbital-pair diagnostics:</strong> ${orbitalPairItems.length}/${items.length} runs</p>
      <p><strong>K-point-aware metrics:</strong> ${kpointItems.length}/${items.length} runs</p>
      <p><strong>Safety:</strong> ${safetyText}</p>
      <code>Comparison/results/${pipeline.resultsDir}</code>
      <code>${orbitalPairPath}</code>
      <code>${kpointPath}</code>
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
  if (value === null || value === undefined || value === "") return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function metricDisplayUnitInfo(metricKeyOrTitle) {
  const text = String(metricKeyOrTitle || "").trim();
  const lower = text.toLowerCase();
  if (!lower) return { scale: 1, unit: "", rawUnit: "", converted: false };
  if (lower.includes("mev")) {
    return lower.includes("mev2") || lower.includes("mev^2") || lower.includes("mev²")
      ? { scale: 1, unit: "meV²", rawUnit: "meV²", converted: false }
      : { scale: 1, unit: "meV", rawUnit: "meV", converted: false };
  }
  if (
    /(^|_)ev2(_|$)/i.test(text) ||
    /\bev\^?2\b/i.test(text) ||
    /\bev²\b/i.test(text)
  ) {
    return { scale: 1_000_000, unit: "meV²", rawUnit: "eV²", converted: true };
  }
  if (/(^|_)ev(_|$)/i.test(text) || /\bev\b/i.test(text)) {
    return { scale: 1000, unit: "meV", rawUnit: "eV", converted: true };
  }
  return { scale: 1, unit: "", rawUnit: "", converted: false };
}

function metricDisplayValue(metricKey, value) {
  const number = finiteNumber(value);
  if (number == null) return null;
  return number * metricDisplayUnitInfo(metricKey).scale;
}

function metricDisplayUnitForKeys(metricKeys = []) {
  const units = Array.from(
    new Set(
      metricKeys
        .map((key) => metricDisplayUnitInfo(key).unit)
        .filter(Boolean),
    ),
  );
  return units.length === 1 ? units[0] : "";
}

function metricDisplayAxisTitle(title, metricKeys = []) {
  const keys = Array.isArray(metricKeys) ? metricKeys : [metricKeys];
  const unit = metricDisplayUnitForKeys(keys);
  let text = String(title || "");
  if (unit === "meV²") {
    text = text
      .replace(/_eV2(?=_|$)/g, "_meV2")
      .replace(/\beV\^2\b/g, "meV²")
      .replace(/\beV2\b/g, "meV²")
      .replace(/\beV²\b/g, "meV²");
  } else if (unit === "meV") {
    text = text.replace(/_eV(?=_|$)/g, "_meV").replace(/\beV\b/g, "meV");
  } else {
    const titleUnit = metricDisplayUnitInfo(text);
    if (titleUnit.unit === "meV²") {
      text = text
        .replace(/_eV2(?=_|$)/g, "_meV2")
        .replace(/\beV\^2\b/g, "meV²")
        .replace(/\beV2\b/g, "meV²")
        .replace(/\beV²\b/g, "meV²");
    } else if (titleUnit.unit === "meV") {
      text = text.replace(/_eV(?=_|$)/g, "_meV").replace(/\beV\b/g, "meV");
    }
  }
  if (!text && unit) return unit;
  if (unit && !text.toLowerCase().includes(unit.toLowerCase().replace("²", ""))) return `${text} (${unit})`;
  return text;
}

function metricDisplayLabel(metricKey, label = "") {
  const unit = metricDisplayUnitInfo(metricKey).unit;
  const text = String(label || metricKey || "Metric value");
  if (!unit || text.toLowerCase().includes(unit.toLowerCase().replace("²", ""))) return text;
  return `${text} (${unit})`;
}

function scaleReferencesForMetric(references, metricKey) {
  const scale = metricDisplayUnitInfo(metricKey).scale;
  if (!Array.isArray(references) || scale === 1) return references;
  return references.map((reference) => {
    const value = finiteNumber(reference?.value);
    return value == null ? reference : { ...reference, value: value * scale };
  });
}

function formatDisplayedMetric(value, metricKey, precision = 4) {
  const number = metricDisplayValue(metricKey, value);
  if (number == null) return "No metric";
  const unit = metricDisplayUnitInfo(metricKey).unit;
  return formatMetricDisplay(number, unit ? ` ${unit}` : "", precision);
}

function themeCssVar(name, fallback) {
  if (typeof window === "undefined" || !window.getComputedStyle) return fallback;
  const value = window.getComputedStyle(document.documentElement).getPropertyValue(name);
  return value && value.trim() ? value.trim() : fallback;
}

function currentPlotPaperColor() {
  return themeCssVar("--plot-bg", "#ffffff");
}

function currentPlotGridColor() {
  return themeCssVar("--plot-grid", "rgba(17, 24, 39, 0.10)");
}

function currentPlotAxisColor() {
  return themeCssVar("--plot-axis-text", "#111827");
}

// SCIENCE_PLOT_AXIS_COLOR / SCIENCE_PLOT_GRID_COLOR are getters so Plotly layouts
// always reflect the active theme (dark by default, light via the UI toggle).
Object.defineProperty(globalThis, "SCIENCE_PLOT_AXIS_COLOR", { get: currentPlotAxisColor });
Object.defineProperty(globalThis, "SCIENCE_PLOT_GRID_COLOR", { get: currentPlotGridColor });
const SCIENCE_PLOT_FONT_FAMILY = '"Inter", "STIX Two Text", "Latin Modern Roman", "Times New Roman", Georgia, sans-serif';
const PLOT_COLORS = [
  "#4477aa",
  "#228833",
  "#cc6677",
  "#ee7733",
  "#66ccee",
  "#aa3377",
  "#bbbbbb",
  "#009988",
  "#997700",
];
const DEEPH_REFERENCE_COLORS = ["#cc3311", "#882255", "#332288", "#117733", "#666666"];
const DEEPH_PAPER_REFERENCE_LINES = {
  "g2m-deeph-plot-metric_scaling_h_mae": [
    {
      value: 0.0019,
      label: "DeepH graphene gen. 1.9 meV",
      detail: "Graphene generalization MAE of local transformed H' blocks for 2,000 unseen 100-400 K configurations; diagnostic guide, not an identical H(k) MAE.",
    },
    {
      value: 0.0021,
      label: "DeepH graphene avg 2.1 meV",
      detail: "Graphene H' MAE averaged over all 13x13 orbital combinations in the paper.",
    },
    {
      value: 0.0085,
      label: "DeepH graphene high 8.5 meV",
      detail: "Upper end of reported graphene orbital-combination MAE range; shown as a visual diagnostic guide.",
    },
  ],
  "g2m-deeph-plot-metric_scaling_r2": [
    {
      value: 0.9994,
      label: "DeepH graphene R2 0.9994",
      detail: "Reported coefficient of determination for graphene nearest-neighbor 1s Hamiltonian element.",
    },
  ],
  "g2m-deeph-plot-metric_scaling_dos_mae": [
    {
      value: 0.0001,
      label: "DeepH DOS MAE ~1e-4",
      detail: "Graphene DOS MAE reported as about 0.1 x 10^-3 eV^-1 A^-2 over 500 points from -6 to +6 eV around Fermi; compare units and window carefully.",
    },
  ],
  "plot-kpoint-h": [
    {
      value: 0.0019,
      label: "DeepH graphene gen. 1.9 meV",
      detail: "Paper graphene generalization MAE of local transformed H' blocks; shown here as an eV-scale visual guide, not an identical H(k) metric.",
    },
    {
      value: 0.0021,
      label: "DeepH graphene avg 2.1 meV",
      detail: "Paper graphene H' orbital-combination MAE averaged over all 13x13 orbital pairs; not identical to repository weighted H(k) MAE.",
    },
    {
      value: 0.0066,
      label: "DeepH graphene NN 6.6 meV",
      detail: "Paper nearest-neighbor 1s Hamiltonian element MAE; visual guide only.",
    },
    {
      value: 0.0085,
      label: "DeepH graphene high 8.5 meV",
      detail: "Upper end of the reported graphene orbital-combination MAE range; visual guide only.",
    },
  ],
  "plot-kpoint-dos": [
    {
      value: 0.0001,
      label: "DeepH DOS ~1e-4",
      detail: "Paper graphene DOS MAE is about 0.1 x 10^-3 eV^-1 A^-2 over 500 points from -6 to +6 eV around Fermi; compare units carefully.",
    },
  ],
  "plot-deeph-mev": [
    {
      value: 1.9,
      label: "DeepH graphene gen. 1.9 meV",
      detail: "Graphene generalization MAE of H' for 2,000 unseen 100-400 K configurations.",
    },
    {
      value: 1.3,
      label: "DeepH MoS2 pairs 1.3 meV",
      detail: "MoS2 atom-pair H' MAEs reported in the paper span 0.7-1.3 meV.",
    },
    {
      value: 2.1,
      label: "DeepH graphene avg 2.1 meV",
      detail: "Graphene H' orbital-combination MAE averaged over all 13x13 orbital pairs.",
    },
    {
      value: 3.5,
      label: "DeepH CNT d>2nm 3.5 meV",
      detail: "CNT Hamiltonian MAE reported below this value for nanotube diameter above 2 nm.",
    },
    {
      value: 6.6,
      label: "DeepH graphene NN 6.6 meV",
      detail: "Graphene nearest-neighbor 1s Hamiltonian element MAE.",
    },
    {
      value: 8.5,
      label: "DeepH graphene max 8.5 meV",
      detail: "Upper end of the reported graphene orbital-combination MAE range.",
    },
  ],
  "plot-deeph-r2": [
    {
      value: 0.9994,
      label: "DeepH graphene R2 0.9994",
      detail: "Reported coefficient of determination for the graphene nearest-neighbor 1s element.",
    },
  ],
  "plot-deeph-dos": [
    {
      value: 0.0001,
      label: "DeepH DOS MAE 1e-4",
      detail: "Graphene DOS MAE reported as about 0.1 x 10^-3 eV^-1 A^-2.",
    },
  ],
};

function plotColor(index) {
  return PLOT_COLORS[index % PLOT_COLORS.length];
}

function deephReferenceColor(index) {
  return DEEPH_REFERENCE_COLORS[index % DEEPH_REFERENCE_COLORS.length];
}

function sciencePlotAxis(axis = {}, fallbackTitle = "") {
  const rawTitle = axis.title == null && fallbackTitle ? fallbackTitle : axis.title;
  const title = typeof rawTitle === "object"
    ? {
        ...rawTitle,
        font: { size: 16, color: SCIENCE_PLOT_AXIS_COLOR, ...(rawTitle.font || {}) },
      }
    : { text: rawTitle || "", font: { size: 16, color: SCIENCE_PLOT_AXIS_COLOR } };
  return {
    showline: true,
    linecolor: SCIENCE_PLOT_AXIS_COLOR,
    linewidth: 1.1,
    mirror: false,
    ticks: "outside",
    ticklen: 4,
    tickwidth: 1,
    tickcolor: SCIENCE_PLOT_AXIS_COLOR,
    tickfont: { size: 14, color: SCIENCE_PLOT_AXIS_COLOR },
    zeroline: false,
    showgrid: true,
    gridcolor: SCIENCE_PLOT_GRID_COLOR,
    gridwidth: 0.7,
    automargin: true,
    ...axis,
    title,
  };
}

function sciencePlotLayout(layout = {}) {
  const rawTitle = typeof layout.title === "object" ? layout.title : { text: layout.title || "" };
  const themedPaperColor = currentPlotPaperColor();
  const themedAxisColor = currentPlotAxisColor();
  const annotations = (layout.annotations || []).map((annotation) => ({
    font: { size: 15, color: themedAxisColor, ...(annotation.font || {}) },
    ...annotation,
  }));
  return {
    colorway: PLOT_COLORS,
    autosize: true,
    font: { family: SCIENCE_PLOT_FONT_FAMILY, color: themedAxisColor, size: 15 },
    ...layout,
    // Theme colors always win over any caller-supplied bgcolor so plots stay
    // legible when the user toggles dark/light mode (rendering, not data, logic).
    paper_bgcolor: themedPaperColor,
    plot_bgcolor: themedPaperColor,
    margin: { l: 64, r: 24, t: 50, b: 58, ...(layout.margin || {}) },
    title: {
      x: 0.02,
      xanchor: "left",
      ...rawTitle,
      font: { size: 18, color: themedAxisColor, ...(rawTitle.font || {}) },
    },
    legend: {
      orientation: "h",
      y: -0.22,
      x: 0,
      xanchor: "left",
      yanchor: "top",
      bgcolor: "rgba(0,0,0,0)",
      borderwidth: 0,
      font: { size: 14, color: themedAxisColor },
      ...(layout.legend || {}),
    },
    hoverlabel: {
      bgcolor: themedPaperColor,
      bordercolor: "rgba(128, 128, 128, 0.32)",
      font: { family: SCIENCE_PLOT_FONT_FAMILY, size: 15, color: themedAxisColor },
      ...(layout.hoverlabel || {}),
    },
    annotations,
    xaxis: sciencePlotAxis(layout.xaxis, "Dataset size"),
    yaxis: sciencePlotAxis(layout.yaxis),
  };
}

function sciencePlotTrace(trace = {}, index = 0) {
  const color = trace.marker?.color || trace.line?.color || plotColor(index);
  const marker = trace.marker
    ? {
        size: 8,
        opacity: 0.88,
        color,
        line: { color: "#ffffff", width: 0.7, ...(trace.marker.line || {}) },
        ...trace.marker,
      }
    : undefined;
  const line = trace.line
    ? {
        color,
        width: trace.type === "bar" ? undefined : 1.65,
        ...trace.line,
      }
    : undefined;
  const styled = {
    ...trace,
    ...(marker ? { marker } : {}),
    ...(line ? { line } : {}),
  };
  if (trace.type === "bar") {
    styled.marker = {
      color,
      opacity: 0.84,
      line: { color: "rgba(17,24,39,0.34)", width: 0.7 },
      ...(trace.marker || {}),
    };
  }
  return styled;
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
  const xValues = fitPoints.map((point) => point.x);
  const minX = Math.min(...xValues);
  const maxX = Math.max(...xValues);
  if (!Number.isFinite(minX) || !Number.isFinite(maxX)) return [];
  const scale = maxX - minX;
  const center = (maxX + minX) / 2;
  const normalizedFitPoints = fitPoints.map((point) => ({
    x: scale > 0 ? (2 * (point.x - center)) / scale : 0,
    y: point.y,
  }));
  const coefficients = polynomialCoefficients(normalizedFitPoints, degree);
  if (!coefficients) return [];
  const lineX = minX === maxX
    ? [minX]
    : Array.from({ length: 80 }, (_, index) => minX + ((maxX - minX) * index) / 79);
  return lineX.map((x) => {
    const normalizedX = scale > 0 ? (2 * (x - center)) / scale : 0;
    return { x, y: evaluatePolynomial(coefficients, normalizedX) };
  });
}

function reciprocalFitLinePoints(points, power) {
  const fitPoints = aggregateFitPoints(points).filter((point) => point.x !== 0);
  if (fitPoints.length < 2) return [];
  const transformedPoints = fitPoints
    .map((point) => ({
      x: 1 / (point.x ** power),
      y: point.y,
    }))
    .filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y));
  if (transformedPoints.length < 2) return [];
  const coefficients = polynomialCoefficients(transformedPoints, 1);
  if (!coefficients) return [];
  const xValues = fitPoints.map((point) => point.x);
  const minX = Math.min(...xValues);
  const maxX = Math.max(...xValues);
  if (!Number.isFinite(minX) || !Number.isFinite(maxX)) return [];
  const lineX = minX === maxX
    ? [minX]
    : Array.from({ length: 80 }, (_, index) => minX + ((maxX - minX) * index) / 79);
  return lineX
    .filter((x) => x !== 0)
    .map((x) => ({
      x,
      y: evaluatePolynomial(coefficients, 1 / (x ** power)),
    }));
}

function fitKindReciprocalPower(kind) {
  if (kind === "inverse") return 1;
  if (kind === "inverse_square") return 2;
  return null;
}

function fitKindDegree(kind) {
  if (kind === "quadratic") return 2;
  if (kind === "linear") return 1;
  const match = String(kind || "").match(/^degree_(\d+)$/);
  if (match) return Number(match[1]);
  return 1;
}

function fitKindForDegree(degree) {
  if (degree === 1) return "linear";
  if (degree === 2) return "quadratic";
  return `degree_${degree}`;
}

function fitKindLabel(kind) {
  if (kind === "inverse") return "1/x";
  if (kind === "inverse_square") return "1/x^2";
  const degree = fitKindDegree(kind);
  if (degree === 1) return "linear";
  if (degree === 2) return "quadratic";
  return `degree ${degree}`;
}

function fitKindDash(kind) {
  if (kind === "inverse") return "longdash";
  if (kind === "inverse_square") return "dashdot";
  const degree = fitKindDegree(kind);
  if (degree === 1) return "solid";
  if (degree === 2) return "dash";
  return "dot";
}

function fitKindOrder(kind) {
  if (kind === "linear") return 1;
  if (kind === "quadratic") return 2;
  if (kind === "inverse") return 3;
  if (kind === "inverse_square") return 4;
  return 10 + fitKindDegree(kind);
}

function fitTrace(points, name, color, kind, extra = {}) {
  const { fitYMin = null, ...traceExtra } = extra;
  const degree = fitKindDegree(kind);
  const reciprocalPower = fitKindReciprocalPower(kind);
  const linePoints = reciprocalPower == null
    ? fitLinePoints(points, degree)
    : reciprocalFitLinePoints(points, reciprocalPower);
  if (linePoints.length < 2) return null;
  const yValues = linePoints.map((point) => {
    const value = point.y;
    return Number.isFinite(fitYMin) && value < fitYMin ? null : value;
  });
  return {
    type: "scatter",
    mode: "lines",
    name: `${name} ${fitKindLabel(kind)} fit`,
    x: linePoints.map((point) => point.x),
    y: yValues,
    line: {
      color,
      width: 2,
      dash: fitKindDash(kind),
    },
    opacity: 1,
    hoverinfo: "skip",
    visible: false,
    showlegend: false,
    meta: {
      role: "fit",
      fitKind: kind,
      fitDegree: degree,
      fitLabel: `${fitKindLabel(kind)} fit`,
      fitModel: reciprocalPower == null ? "polynomial" : `a+b/x^${reciprocalPower}`,
    },
    ...traceExtra,
  };
}

const MAX_POLYNOMIAL_FIT_DEGREE = 12;

function addFitTraces(traces, points, name, color, extra = {}) {
  for (const kind of ["linear", "quadratic", "inverse", "inverse_square"]) {
    const trace = fitTrace(points, name, color, kind, extra);
    if (trace) traces.push(trace);
  }
  for (let degree = 3; degree <= MAX_POLYNOMIAL_FIT_DEGREE; degree += 1) {
    const trace = fitTrace(points, name, color, fitKindForDegree(degree), extra);
    if (trace) traces.push(trace);
  }
}

const FIT_ACTIVE_DATA_OPACITY = 0.65;

function fitVisibility(traces, fitKind) {
  return traces.map((trace) => {
    if (trace.meta?.role !== "fit") return true;
    if (fitKind === "none") return false;
    return trace.meta.fitKind === fitKind;
  });
}

function fitOpacities(traces, fitKind) {
  const fitActive = Boolean(fitKind) && fitKind !== "none";
  return traces.map((trace) => {
    if (trace.meta?.role === "fit") return typeof trace.opacity === "number" ? trace.opacity : 1;
    return fitActive ? FIT_ACTIVE_DATA_OPACITY : 1;
  });
}

function fitModes(traces, fitKind) {
  const fitActive = Boolean(fitKind) && fitKind !== "none";
  return traces.map((trace) => {
    if (trace.meta?.role === "fit") return trace.mode || "lines";
    if (!fitActive) return trace.mode;
    return trace.mode === "markers" ? "markers" : "lines+markers";
  });
}

function fitShowLegend(traces, fitKind) {
  return traces.map((trace) => {
    if (trace.meta?.role !== "fit") return trace.showlegend !== false;
    if (fitKind === "none") return false;
    return trace.meta.fitKind === fitKind;
  });
}

function fitSelectorArgs(traces, fitKind) {
  return [{
    visible: fitVisibility(traces, fitKind),
    opacity: fitOpacities(traces, fitKind),
    mode: fitModes(traces, fitKind),
    showlegend: fitShowLegend(traces, fitKind),
  }];
}

function withFitSelector(layout, traces) {
  const fitKinds = Array.from(
    new Map(
      traces
        .filter((trace) => trace.meta?.role === "fit")
        .map((trace) => [trace.meta.fitKind, trace.meta.fitLabel || `${trace.meta.fitKind} fit`]),
    ).entries(),
  ).sort((left, right) => fitKindOrder(left[0]) - fitKindOrder(right[0]));
  if (!fitKinds.length) return layout;
  const standardFitKinds = fitKinds.filter(([fitKind]) => fitKindOrder(fitKind) <= 4);
  const polynomialFitKinds = fitKinds.filter(([fitKind]) => fitKindOrder(fitKind) > 4);
  const updatemenus = [
    {
      type: "dropdown",
      x: polynomialFitKinds.length ? 0.79 : 1,
      y: 1.16,
      xanchor: "right",
      yanchor: "top",
      active: standardFitKinds.length,
      buttons: [
        ...standardFitKinds.map(([fitKind, label]) => ({
          label: label.replace(/\bfit$/i, "fit"),
          method: "update",
          args: fitSelectorArgs(traces, fitKind),
        })),
        { label: "No fit", method: "update", args: fitSelectorArgs(traces, "none") },
      ],
    },
  ];
  if (polynomialFitKinds.length) {
    updatemenus.push({
      type: "dropdown",
      x: 1,
      y: 1.16,
      xanchor: "right",
      yanchor: "top",
      active: -1,
      buttons: polynomialFitKinds.map(([fitKind, label]) => ({
        label: label.replace(/\bfit$/i, "fit"),
        method: "update",
        args: fitSelectorArgs(traces, fitKind),
      })),
    });
  }
  return {
    ...layout,
    margin: { ...layout.margin, t: Math.max(layout.margin?.t || 46, 78) },
    updatemenus,
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
    const label = groupedRunLabel(pipeline, items);
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
  const trainingLabel = runTrainingGroupLabel(first);
  const material = runMaterialLabel(first);
  return [
    pipelineLabel(pipeline),
    isKnownMaterialLabel(material) ? material : "",
    trainingLabel,
  ].filter(Boolean).join(" · ");
}

function groupedRuns(runs, options = {}) {
  const groups = new Map();
  const includeTrainingContext = Boolean(options.includeTrainingContext);
  for (const run of runs) {
    const trainingContext = includeTrainingContext ? runTrainingGroupLabel(run) : "";
    const material = runMaterialLabel(run);
    const key = [
      run.pipeline,
      isKnownMaterialLabel(material) ? `material=${material}` : "",
      trainingContext ? `training=${trainingContext}` : "",
    ].filter(Boolean).join("||");
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
        .map((run) => ({
          x: run.dataset_size,
          y: metricDisplayValue(metric.key, metricValue(run, group, metric.key)),
          text: [
            run.training_tag || run.run_id || "",
            materialContextText(run),
            metricContextText(run),
            runSafetyContextText(run),
          ]
            .filter(Boolean)
            .join(" · "),
        }))
        .filter((point) => point.y != null);
      if (!points.length) continue;
      const name = metrics.length > 1 ? `${label} · ${metric.label}` : label;
      const color = plotColor(traceIndex);
      const legendgroup = `${groupKey}-${metric.key}`;
      const fitYMin = valuesAreNonNegative(points.map((point) => point.y)) ? 0 : null;
      addFitTraces(traces, points, name, color, { legendgroup, fitYMin });
      traces.push({
        type: "scatter",
        mode: "markers",
        name,
        x: points.map((point) => point.x),
        y: points.map((point) => point.y),
        text: points.map((point) => point.text),
        marker: { size: 9, opacity: 0.86, color },
        legendgroup,
        hovertemplate:
          `dataset %{x}<br>${metricDisplayLabel(metric.key, metric.label)}: %{y:.4g}` +
          `<br>run/tag %{text}<extra>%{fullData.name}</extra>`,
      });
      traceIndex += 1;
    }
  }
  return traces;
}

function plotLayout(title, yTitle, extra = {}) {
  return sciencePlotLayout({
    title: { text: title, x: 0.02, xanchor: "left", font: { size: 18 } },
    margin: { l: 64, r: 24, t: 50, b: 58 },
    xaxis: { title: "Dataset size", showgrid: false, zeroline: false },
    yaxis: { title: yTitle, gridcolor: SCIENCE_PLOT_GRID_COLOR, zeroline: false },
    legend: { orientation: "h", y: -0.25, x: 0 },
    font: { family: SCIENCE_PLOT_FONT_FAMILY, color: SCIENCE_PLOT_AXIS_COLOR },
    ...extra,
  });
}

function traceYValues(traces) {
  const values = [];
  for (const trace of traces || []) {
    if (!Array.isArray(trace.y)) continue;
    for (const value of trace.y) {
      const number = finiteNumber(value);
      if (number != null) values.push(number);
    }
  }
  return values;
}

function yAxisRangeIncludingReferences(layout, traces, references) {
  const values = traceYValues(traces);
  for (const reference of references || []) {
    const value = finiteNumber(reference.value);
    if (value != null) values.push(value);
  }
  if (!values.length) return null;
  const minValue = Math.min(...values);
  const maxValue = Math.max(...values);
  if (!Number.isFinite(minValue) || !Number.isFinite(maxValue)) return null;
  const currentRange = layout?.yaxis?.range;
  const baseMin = Array.isArray(currentRange) && currentRange.length === 2
    ? Math.min(finiteNumber(currentRange[0]) ?? minValue, minValue)
    : minValue;
  const baseMax = Array.isArray(currentRange) && currentRange.length === 2
    ? Math.max(finiteNumber(currentRange[1]) ?? maxValue, maxValue)
    : maxValue;
  if (baseMin === baseMax) {
    const pad = Math.max(Math.abs(baseMin) * 0.05, 1e-6);
    return [baseMin - pad, baseMax + pad];
  }
  const pad = Math.max((baseMax - baseMin) * 0.06, 1e-6);
  return [baseMin - pad, baseMax + pad];
}

function withHorizontalReferenceLines(layout, traces, references, note = "") {
  const validReferences = (references || [])
    .map((reference) => ({ ...reference, value: finiteNumber(reference.value) }))
    .filter((reference) => reference.value != null);
  if (!validReferences.length) return layout;
  const shapes = [...(layout.shapes || [])];
  const annotations = [...(layout.annotations || [])];
  validReferences.forEach((reference, index) => {
    const color = reference.color || deephReferenceColor(index);
    shapes.push({
      type: "line",
      xref: "paper",
      x0: 0,
      x1: 1,
      yref: "y",
      y0: reference.value,
      y1: reference.value,
      line: { color, width: 1.5, dash: reference.dash || "dash" },
    });
    annotations.push({
      text: reference.label,
      xref: "paper",
      x: 1,
      xanchor: "right",
      yref: "y",
      y: reference.value,
      yanchor: "bottom",
      showarrow: false,
      font: { size: 14, color },
      bgcolor: "rgba(255, 255, 255, 0.82)",
      bordercolor: "rgba(148, 163, 184, 0.55)",
      borderwidth: 1,
      borderpad: 2,
      hovertext: reference.detail || reference.label,
      hoverlabel: { bgcolor: "#ffffff", bordercolor: color, font: { color: "#17202a" } },
    });
  });
  if (note) {
    annotations.push(topPlotAnnotation(note, 1.3, "#56616f"));
  }
  const yRange = yAxisRangeIncludingReferences(layout, traces, validReferences);
  return {
    ...layout,
    shapes,
    annotations,
    margin: {
      ...layout.margin,
      r: Math.max(layout.margin?.r || 18, 112),
      t: Math.max(layout.margin?.t || 46, note ? 96 : 74),
    },
    yaxis: {
      ...(layout.yaxis || {}),
      ...(yRange ? { range: yRange } : {}),
    },
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
    font: { size: 16, color: "#6b7280" },
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
    font: { size: 15, color },
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

function plotInfoFor(plotId, explicitInfo = null) {
  if (explicitInfo) return explicitInfo;
  const g2mDeepHInfo = G2M_DEEPH_PLOT_HELP_BY_ID[plotId];
  if (g2mDeepHInfo) return g2mDeepHInfo;
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

function installPlotInfoBubble(id, explicitInfo = null) {
  const node = plotNode(id);
  if (!node) return;
  node.querySelectorAll(":scope > .plot-info-bubble").forEach((bubble) => bubble.remove());
  const info = plotInfoFor(node.id, explicitInfo);
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
  if (!node) return;
  ensureMathJaxLoaded()
    .then(() => window.MathJax.typesetPromise([node]))
    .catch((error) => {
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

function repaintPlotsForTheme() {
  if (!window.Plotly?.relayout) return;
  const paperColor = currentPlotPaperColor();
  const axisColor = currentPlotAxisColor();
  const gridColor = currentPlotGridColor();
  document.querySelectorAll(".plot-card.js-plotly-plot").forEach((node) => {
    Plotly.relayout(node, {
      paper_bgcolor: paperColor,
      plot_bgcolor: paperColor,
      "font.color": axisColor,
      "xaxis.color": axisColor,
      "xaxis.linecolor": axisColor,
      "xaxis.tickcolor": axisColor,
      "xaxis.tickfont.color": axisColor,
      "xaxis.title.font.color": axisColor,
      "xaxis.gridcolor": gridColor,
      "yaxis.color": axisColor,
      "yaxis.linecolor": axisColor,
      "yaxis.tickcolor": axisColor,
      "yaxis.tickfont.color": axisColor,
      "yaxis.title.font.color": axisColor,
      "yaxis.gridcolor": gridColor,
      "xaxis2.color": axisColor,
      "xaxis2.linecolor": axisColor,
      "xaxis2.tickcolor": axisColor,
      "xaxis2.tickfont.color": axisColor,
      "xaxis2.title.font.color": axisColor,
      "xaxis2.gridcolor": gridColor,
      "yaxis2.color": axisColor,
      "yaxis2.linecolor": axisColor,
      "yaxis2.tickcolor": axisColor,
      "yaxis2.tickfont.color": axisColor,
      "yaxis2.title.font.color": axisColor,
      "yaxis2.gridcolor": gridColor,
      "legend.font.color": axisColor,
      "hoverlabel.bgcolor": paperColor,
      "hoverlabel.font.color": axisColor,
    }).catch(() => {});
  });
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
  const { plotInfo = null, ...plotlyConfig } = config || {};
  const nextLayout = sciencePlotLayout(layout || {});
  const nextTraces = (traces || []).map((trace, index) => sciencePlotTrace(trace, index));
  const nextConfig = {
    responsive: true,
    displaylogo: false,
    modeBarButtonsToRemove: ["lasso2d", "select2d"],
    toImageButtonOptions: {
      format: "svg",
      filename: "graph2mat_deeph_plot",
      scale: 2,
      ...(plotlyConfig.toImageButtonOptions || {}),
    },
    ...plotlyConfig,
  };
  Plotly.react(node, nextTraces, nextLayout, nextConfig).then(() => {
    installPlotInfoBubble(node, plotInfo);
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
      label: groupedRunLabel(pipeline, items),
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
  const metricKeys = (metrics || []).map((metric) => metric.key);
  let layout = plotLayout(title, metricDisplayAxisTitle(yTitle, metricKeys));
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
  layout = withHorizontalReferenceLines(
    layout,
    traces,
    scaleReferencesForMetric(DEEPH_PAPER_REFERENCE_LINES[id], metricKeys[0]),
    DEEPH_PAPER_REFERENCE_LINES[id]
      ? "DeepH paper reference lines are diagnostic guides; match metric definitions before final claims."
      : "",
  );
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
  layout = withHorizontalReferenceLines(
    layout,
    traces,
    DEEPH_PAPER_REFERENCE_LINES[id],
    "DeepH paper reference line is diagnostic; this plot uses repository sparse supports.",
  );
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
  layout = withHorizontalReferenceLines(
    layout,
    traces,
    DEEPH_PAPER_REFERENCE_LINES[id],
    "DeepH DOS reference uses graphene paper units; treat as a visual guide.",
  );
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
  return finiteNumber(row.rmse_union_meV_mean) ??
    finiteNumber(row.rmse_union_meV) ??
    metricDisplayValue("rmse_union_eV", finiteNumber(row.rmse_union_eV_mean) ?? finiteNumber(row.rmse_union_eV));
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
      `${materialContextText(choice.run)}<br>` +
      `samples ${nSamples}, entries ${nEntries}<br>` +
      `RMSE ${rmse == null ? "No metric" : `${rmse.toPrecision(4)} meV`}<br>` +
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
            { title: { text: titleFor(choice), x: 0.02, xanchor: "left", font: { size: 18 } } },
          ],
        })),
      },
    ];
  }
  renderPlot(id, traces, layout, { responsive: true, displaylogo: false });
}

function combinedCrossMaterialSummary(experiments) {
  const labels = new Set();
  const hashes = new Set();
  const warnings = [];
  let hasUnknown = false;
  let mixed = false;
  for (const experiment of experiments || []) {
    const summary = experiment?.material_summary || {};
    for (const label of summary.material_display_labels || summary.material_labels || []) {
      if (isKnownMaterialLabel(label)) labels.add(String(label));
    }
    for (const hash of summary.material_compatibility_hashes || []) {
      if (hash) hashes.add(String(hash));
    }
    for (const warning of summary.material_compatibility_warnings || []) {
      if (warning && !warnings.includes(warning)) warnings.push(warning);
    }
    hasUnknown = hasUnknown || Boolean(summary.has_unknown_material);
    mixed = mixed || Boolean(summary.mixed_materials);
  }
  if (!labels.size && hasUnknown) labels.add(UNKNOWN_MATERIAL_LABEL);
  return {
    material_display_labels: Array.from(labels).sort(),
    material_compatibility_hashes: Array.from(hashes).sort(),
    material_compatibility_warnings: warnings,
    has_unknown_material: hasUnknown,
    mixed_materials: mixed || hashes.size > 1 || (!hashes.size && labels.size > 1),
  };
}

function renderBoxPlot(id, runs) {
  const traces = [];
  const availability = [];
  const fallbackRuns = [];
  for (const run of runs) {
    const spectral = sampleMetricValues(run, "spectral", "fermi_window_rmse_eV")
      .map((value) => metricDisplayValue("fermi_window_rmse_eV", value));
    const frontier = sampleMetricValues(run, "spectral", "frontier_window_rmse_eV")
      .map((value) => metricDisplayValue("frontier_window_rmse_eV", value));
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
        hovertemplate: "%{y:.4g} meV<br>Frontier RMSE (HOMO/LUMO fallback)<extra>%{fullData.name}</extra>",
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
      hovertemplate: "%{y:.4g} meV<extra>%{fullData.name}</extra>",
    });
  }
  const layout = plotLayout("Distribucion por muestra: Fermi-window RMSE", "RMSE meV", {
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
      font: { size: 14, color: "#56616f" },
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
      font: { size: 14, color: "#9f5b00" },
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
    const label = groupedRunLabel(pipeline, items);
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
        y.push(metricDisplayValue("global_rmse_eV", yValue));
        const fermiValue =
          typeof row.fermi_window_rmse_eV === "number" && Number.isFinite(row.fermi_window_rmse_eV)
            ? `${metricDisplayValue("fermi_window_rmse_eV", row.fermi_window_rmse_eV).toPrecision(4)} meV`
            : "no disponible";
        const frontierValue =
          typeof row.frontier_window_rmse_eV === "number" && Number.isFinite(row.frontier_window_rmse_eV)
            ? `${metricDisplayValue("frontier_window_rmse_eV", row.frontier_window_rmse_eV).toPrecision(4)} meV`
            : "no disponible";
        text.push(
          `dataset_${run.dataset_size} - sample ${row.sample}<br>` +
            `${materialContextText(run)}<br>` +
            `Fermi source: ${row.fermi_level_source || "unknown"}<br>` +
            `Fermi RMSE: ${fermiValue}<br>` +
            `Frontier RMSE: ${frontierValue}`,
        );
      }
    }
    if (!x.length) continue;
    const color = plotColor(traceIndex);
    const points = x.map((value, index) => ({ x: value, y: y[index] }));
    const fitYMin = valuesAreNonNegative(points.map((point) => point.y)) ? 0 : null;
    addFitTraces(traces, points, label, color, { legendgroup: pipeline, fitYMin });
    traces.push({
      type: "scatter",
      mode: "markers",
      name: label,
      x,
      y,
      text,
      marker: { size: 9, opacity: 0.82, color },
      legendgroup: pipeline,
      hovertemplate: "%{text}<br>Frobenius %{x:.4g}<br>Global spectral RMSE %{y:.4g} meV<extra>%{fullData.name}</extra>",
    });
    traceIndex += 1;
  }
  let layout = plotLayout("Relacion matriz-espectro", "Global spectral RMSE meV", {
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
    { group: "kpoint_matrix", key: "h_mae_eV", label: "H(k) MAE", better: "lower" },
    { group: "kpoint_spectral", key: "low_energy_rmse_eV", label: "K low-energy", better: "lower" },
    { group: "kpoint_dos", key: "dos_mae_500_fermi_window", label: "K DOS MAE", better: "lower" },
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
      .map((run) => metricDisplayValue(metric.key, metricValue(run, metric.group, metric.key)))
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
      const displayValue = metricDisplayValue(metric.key, value);
      const transformed = metric.transform === "log10_positive" ? Math.log10(Math.max(displayValue, 1e-12)) : displayValue;
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
      return value == null
        ? ""
        : formatMetricDisplay(metricDisplayValue(metric.key, value), metricDisplayUnitInfo(metric.key).unit ? ` ${metricDisplayUnitInfo(metric.key).unit}` : metric.suffix || "");
    }),
  );
  const customdata = rows.map((run) =>
    metrics.map((metric) => {
      const value = metricValue(run, metric.group, metric.key);
      return {
        raw: value == null
          ? "No metric"
          : formatMetricDisplay(metricDisplayValue(metric.key, value), metricDisplayUnitInfo(metric.key).unit ? ` ${metricDisplayUnitInfo(metric.key).unit}` : metric.suffix || ""),
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
        textfont: { size: 13 },
        customdata,
        x: metrics.map((item) => metricDisplayLabel(item.key, item.label)),
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
      title: { text: "Resumen compacto de metricas", x: 0.02, xanchor: "left", font: { size: 18 } },
      margin: { l: 120, r: 18, t: 74, b: 72 },
      font: { family: "Inter, sans-serif", color: currentPlotAxisColor() },
      annotations: [
        topPlotAnnotation(
          "Color = valor normalizado por metrica; el tiempo usa log10(segundos). Los valores de las celdas son los valores fisicos.",
          1.08,
          currentPlotAxisColor(),
        ),
      ],
    },
    { responsive: true, displaylogo: false },
  );
}

function renderSensitivitySweeps(id, runs) {
  const traces = [];
  for (const [pipeline, items] of groupedRuns(runs)) {
    const label = groupedRunLabel(pipeline, items);
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
        x: t * 1000,
        y: metricDisplayValue("rmse_union_eV", sparseByThreshold.get(t).reduce((s, v) => s + v, 0) / sparseByThreshold.get(t).length),
      }));
      const sparseColor = plotColor(traces.length);
      addFitTraces(traces, sparsePoints, `${label} sparse-threshold RMSE`, sparseColor, {
        xaxis: "x1",
        yaxis: "y1",
        legendgroup: `${pipeline}-sparse-threshold`,
        fitYMin: valuesAreNonNegative(sparsePoints.map((point) => point.y)) ? 0 : null,
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
        x: s * 1000,
        y: metricDisplayValue("dos_wasserstein_eV", dosBySigma.get(s).reduce((sum, v) => sum + v, 0) / dosBySigma.get(s).length),
      }));
      const dosColor = plotColor(traces.length);
      addFitTraces(traces, dosPoints, `${label} DOS sigma W1`, dosColor, {
        xaxis: "x2",
        yaxis: "y2",
        legendgroup: `${pipeline}-dos-sigma`,
        fitYMin: valuesAreNonNegative(dosPoints.map((point) => point.y)) ? 0 : null,
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
    title: { text: "Sensitivity sweeps", x: 0.02, xanchor: "left", font: { size: 18 } },
    grid: { rows: 1, columns: 2, pattern: "independent" },
    xaxis: { title: "Support threshold (meV)", type: "log" },
    yaxis: { title: "RMSE union (meV)" },
    xaxis2: { title: "DOS sigma (meV)" },
    yaxis2: { title: "DOS Wasserstein (meV)" },
    margin: { l: 56, r: 18, t: 46, b: 48 },
    legend: { orientation: "h", y: -0.25 },
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
  const materialSummary = combinedCrossMaterialSummary(selected);
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
    plot_warnings: selected.flatMap((experiment) =>
      (experiment.plot_warnings || []).map((warning) => ({
        ...warning,
        experiment_id: warning.experiment_id || experiment.experiment_id,
      })),
    ),
    source_experiments: selected.map((experiment) => ({
      experiment_id: experiment.experiment_id,
      rows: (experiment.metrics || []).length,
      outputs: experiment.outputs,
      compatibility_group_id: experiment.compatibility_group_id,
      compatibility: experiment.compatibility,
      material_summary: experiment.material_summary,
    })),
    multi_experiment_available: experiments.length > 1,
    isolation_warning: isolationWarning,
    material_summary: materialSummary,
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
    const materialLabels = rowMaterialLabels(row).join(", ");
    const materialIdentityHash = row.material_identity_hash || Object.values(parseJsonObject(row.material_identity_hash_by_method)).sort().join(",");
    const materialCompatibilityHash = row.material_compatibility_hash || Object.values(parseJsonObject(row.material_compatibility_hash_by_method)).sort().join(",");
    const materialWarning = row.material_compatibility_warning || "";
    const key = [
      row.experiment_id,
      materialLabels,
      materialIdentityHash,
      materialCompatibilityHash,
      materialWarning,
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
        material_label: materialLabels || UNKNOWN_MATERIAL_LABEL,
        material_identity_hash: materialIdentityHash,
        material_compatibility_hash: materialCompatibilityHash,
        material_compatibility_warning: materialWarning,
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
    material_label: group.material_label,
    material_identity_hash: group.material_identity_hash,
    material_compatibility_hash: group.material_compatibility_hash,
    material_compatibility_warning: group.material_compatibility_warning,
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
  const material = isKnownMaterialLabel(row.material_label) ? `material ${row.material_label}` : "";
  return [
    material,
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
    font: { size: 15, color: "#9f5b00" },
  };
}

function renderCrossHeatmap(id, experiment, unavailableMessage = "") {
  const metric = selectedCrossMetric(experiment);
  const metricAxis = metricDisplayAxisTitle(metric, [metric]);
  if (!experiment || !(experiment.metrics || []).length) {
    renderEmptyPlot(
      id,
      `Cross-evaluation heatmap (${metricAxis})`,
      unavailableMessage || "No hay tabla cross_evaluation_metrics.csv completa.",
      metricAxis,
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
      return row?.metric_available ? metricDisplayValue(metric, row.mean) : null;
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
        return { label, method: crossMethodLabel(method), valueText: "No row", availability: "0/0", material: UNKNOWN_MATERIAL_LABEL };
      }
      return {
        label,
        method: crossMethodLabel(method),
        valueText: row.metric_available ? formatDisplayedMetric(row.mean, metric) : "No metric",
        availability: metricAvailabilityLabel(row),
        material: row.material_label || UNKNOWN_MATERIAL_LABEL,
      };
    }),
  );
  const layout = plotLayout(`Cross-evaluation heatmap (${metricAxis})`, metricAxis, {
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
        font: { size: 14, color: "#6b7280" },
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
        "material: %{customdata.material}<br>" +
        `${metricAxis}: %{customdata.valueText}<br>` +
        "finite rows: %{customdata.availability}<extra></extra>",
    }],
    layout,
    { responsive: true, displaylogo: false },
  );
}

function renderCrossLearning(id, experiment, unavailableMessage = "") {
  const metric = selectedCrossMetric(experiment);
  const metricAxis = metricDisplayAxisTitle(metric, [metric]);
  if (!experiment || !(experiment.metrics || []).length) {
    renderEmptyPlot(
      id,
      `Learning curves (${metricAxis})`,
      unavailableMessage || "No hay cross_evaluation_metrics.csv para construir curvas de aprendizaje.",
      metricAxis,
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
        points.map((row) => ({ x: row.dataset_size, y: metricDisplayValue(metric, row.mean) })),
        name,
        color,
        {
          legendgroup,
          fitYMin: valuesAreNonNegative(points.map((row) => metricDisplayValue(metric, row.mean))) ? 0 : null,
        },
      );
      traces.push({
        type: "scatter",
        mode: "markers",
        name,
        x: points.map((row) => row.dataset_size),
        y: points.map((row) => metricDisplayValue(metric, row.mean)),
        text: points.map((row) => `${crossDatasetComboLabel(row)} · ${metricAvailabilityLabel(row)} finite`),
        marker: { size: 9, opacity: 0.86, color },
        legendgroup,
        hovertemplate: `dataset %{x}<br>${metricAxis}: %{y:.4g}<br>%{text}<extra>%{fullData.name}</extra>`,
      });
      traceIndex += 1;
    }
  }
  let layout = plotLayout(`Learning curves (${metricAxis})`, metricAxis);
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
  const metricAxis = metricDisplayAxisTitle(metric, [metric]);
  if (!experiment || !(experiment.metrics || []).length) {
    renderEmptyPlot(
      id,
      `Metric vs total compute time (${metricAxis})`,
      unavailableMessage || "No hay cross_evaluation_metrics.csv para leer total_time_seconds.",
      metricAxis,
    );
    return;
  }
  const allMeans = groupedCrossMetrics(experiment?.metrics || [], metric);
  const means = allMeans.filter((row) => row.time != null);
  if (!means.length) {
    renderEmptyPlot(
      id,
      `Metric vs total compute time (${metricAxis})`,
      "Falta total_time_seconds finito en cross_evaluation_metrics.csv; no se puede comparar metrica frente a coste total.",
      metricAxis,
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
      points.map((row) => ({ x: row.time, y: metricDisplayValue(metric, row.mean) })),
      name,
      color,
      {
        legendgroup: method,
        fitYMin: valuesAreNonNegative(points.map((row) => metricDisplayValue(metric, row.mean))) ? 0 : null,
      },
    );
    traces.push({
      type: "scatter",
      mode: "markers",
      name,
      x: points.map((row) => row.time),
      y: points.map((row) => metricDisplayValue(metric, row.mean)),
      text: points.map((row) => `${testSetDisplayLabel(row.test_set)}, ${crossDatasetComboLabel(row)} · ${metricAvailabilityLabel(row)} finite`),
      marker: { size: 9, opacity: 0.86, color },
      legendgroup: method,
      hovertemplate: `%{text}<br>%{x:.2f}s<br>${metricAxis}: %{y:.4g}<extra>%{fullData.name}</extra>`,
    });
    traceIndex += 1;
  }
  let layout = plotLayout(`Metric vs total compute time (${metricAxis})`, metricAxis, {
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
  const metricAxis = metricDisplayAxisTitle(metric, [metric]);
  if (!experiment || !(experiment.metrics || []).length) {
    renderEmptyPlot(
      id,
      `Winner map (${metricAxis})`,
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
        ? `${methodLabel(row.train_method)}=${formatDisplayedMetric(row.mean, metric)} (${metricAvailabilityLabel(row)})`
        : `${methodLabel(row.train_method)}=No metric (${metricAvailabilityLabel(row)})`
    );
    return {
      winner: z[rowIndex][colIndex] == null ? "No metric" : labels.get(z[rowIndex][colIndex]),
      testSet: testSetDisplayLabel(testSet),
      combo,
      material: Array.from(new Set(candidates.map((row) => row.material_label || UNKNOWN_MATERIAL_LABEL))).join(", "),
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
      font: { size: 15, color: "#9f5b00" },
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
        font: { size: 14, color: isMissing ? "#6b7280" : "#17202a" },
      });
    });
  });
  const missingAnnotation = crossMissingGroupsAnnotation(means, metric);
  if (missingAnnotation) annotations.push(missingAnnotation);
  const layout = plotLayout(`Winner map (${metricAxis})`, "Winner", {
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
        "%{customdata.combo}<br>%{customdata.testSet}<br>material: %{customdata.material}<br>winner: %{customdata.winner}<br>%{customdata.values}<extra></extra>",
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

function visibleMaterialWarning(runs, crossExperiment) {
  const runHashes = new Set(
    (runs || [])
      .map((run) => run.material_compatibility_hash)
      .filter((value) => value != null && value !== ""),
  );
  const runLabels = new Set(
    (runs || [])
      .map(runMaterialLabel)
      .filter(isKnownMaterialLabel),
  );
  const summary = crossExperiment?.material_summary || {};
  const crossHashes = new Set(summary.material_compatibility_hashes || []);
  const mixed = Boolean(summary.mixed_materials) ||
    runHashes.size > 1 ||
    crossHashes.size > 1 ||
    (!runHashes.size && !crossHashes.size && runLabels.size > 1);
  if (!mixed) return null;
  return {
    severity: "warning",
    code: "mixed_material_groups",
    scientific_status: "scientifically_inconclusive",
    message: "Multiple material groups are shown; interpret plots as diagnostics, not a pooled benchmark.",
    details: {
      run_material_labels: Array.from(runLabels).sort(),
      run_material_compatibility_hashes: Array.from(runHashes).sort(),
      cross_material_labels: summary.material_display_labels || summary.material_labels || [],
      cross_material_compatibility_hashes: Array.from(crossHashes).sort(),
    },
  };
}

function visibleRunSafetyWarning(runs) {
  const counts = new Map();
  for (const run of runs || []) {
    const status = runScientificSafetyStatus(run);
    counts.set(status, (counts.get(status) || 0) + 1);
  }
  const unsafe = counts.get("unsafe") || 0;
  const unknown = counts.get("unknown") || 0;
  const exploratory = counts.get("exploratory") || 0;
  const nonComparative = counts.get("non_comparative") || 0;
  if (!unsafe && !unknown && !exploratory && !nonComparative) return null;
  return {
    severity: unsafe ? "error" : "warning",
    code: "visible_run_safety_states",
    scientific_status: unsafe ? "scientifically_inconclusive" : "exploratory_only",
    message: "Visible plot points include unsafe, unknown, exploratory or non-comparative runs; they remain visible for audit.",
    details: Object.fromEntries(Array.from(counts.entries()).sort()),
  };
}

function plotWarningEntriesForPayload(payload, crossExperiment, visibleRuns = null) {
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
  addWarning(visibleMaterialWarning(visibleRuns || payload?.runs || [], crossExperiment));
  addWarning(visibleRunSafetyWarning(visibleRuns || payload?.runs || []));
  for (const warning of payload?.plot_warnings || []) {
    if (!crossExperiment || warning.experiment_id === crossExperiment.experiment_id || warning.code === "visualization_compatibility") {
      addWarning(warning);
    }
  }
  return warnings;
}

function renderPlotWarnings(payload, crossExperiment, visibleRuns = null) {
  const banner = document.getElementById("plot-warnings");
  if (!banner) return;
  const warnings = plotWarningEntriesForPayload(payload, crossExperiment, visibleRuns);
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
  syncMaterialFilterOptions(payload);
  syncRunFamilyFilterOptions(payload);
  syncSafetyFilterOptions(payload);
  const allRuns = payload?.runs || [];
  const runs = filterRunsBySafety(filterRunsByFamily(filterRunsByMaterial(allRuns)));
  const materialFilter = selectedPlotMaterial();
  const familyFilter = selectedRunFamily();
  const safetyFilter = selectedPlotSafety();
  const crossExperiment = filterCrossExperimentByMaterial(selectedCrossExperimentSet(payload));
  const crossMetric = selectedCrossMetric(crossExperiment);
  const primaryMetric = primaryCrossMetric(crossExperiment);
  const recommendation = crossExperiment?.recommendation;
  renderPlotWarnings(payload, crossExperiment, runs);
  const crossRows = crossExperiment?.metrics?.length || 0;
  const crossSources = crossExperiment?.source_experiments?.length || 0;
  const isolationText = crossExperiment?.isolation_warning ? ` | ${canonicalDisplayText(crossExperiment.isolation_warning)}` : "";
  const materialFilterText = materialFilter === "all" ? "all materials" : materialFilter;
  const familyFilterText = familyFilter === "all" ? "all families" : runFamilyLabel(familyFilter);
  const safetyFilterText = safetyFilter === "all" ? "all safety states" : safetyStatusLabel(safetyFilter);
  const blockerText = recommendation ? recommendationBlockers(recommendation).slice(0, 6).join(" | ") : "";
  const crossMissingText = !crossExperiment ? crossUnavailableMessage(payload) : "";
  const plotScientificStatus = crossExperiment?.plot_scientific_status || recommendation?.scientific_status || "unknown";
  const crossText = recommendation?.status
    ? ` | material: ${materialFilterText} | family: ${familyFilterText} | cross: ${crossRows} filas del experimento seleccionado (${crossSources} disponibles) | plot metric: ${crossMetric} | primary: ${primaryMetric} | scientific: ${plotScientificStatus} | blockers: ${blockerText || "none"} | ${recommendation.status} - ${canonicalDisplayText(recommendation.reason || "")}${isolationText}`
    : crossMissingText
      ? ` | cross: ${crossMissingText}`
    : "";
  status.textContent = runs.length
    ? `${runs.length}/${allRuns.length} runs con metricas${missingFermiSummary(runs)}`
    : "No hay metricas archivadas";
  status.textContent += ` | family: ${familyFilterText}`;
  status.textContent += ` | safety filter: ${safetyFilterText}`;
  const kpointRunCount = runs.filter((run) => run.metric_space === "kpoint_sampled" || run.kpoint_metrics_enabled).length;
  if (kpointRunCount) status.textContent += ` | ${kpointRunCount} k-point-aware`;
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
  renderLinePlot(
    "plot-kpoint-h",
    runs,
    "kpoint_matrix",
    [
      { key: "h_mae_eV", label: "H(k) MAE" },
      { key: "h_rmse_eV", label: "H(k) RMSE" },
    ],
    "K-point weighted matrix error",
    "eV",
  );
  renderLinePlot(
    "plot-kpoint-low-energy",
    runs,
    "kpoint_spectral",
    [
      { key: "low_energy_rmse_eV", label: "Low-energy" },
      { key: "fermi_window_rmse_eV", label: "Fermi-window" },
      { key: "frontier_window_rmse_eV", label: "Frontier" },
    ],
    "K-point weighted spectral error",
    "RMSE eV",
  );
  renderLinePlot(
    "plot-kpoint-dos",
    runs,
    "kpoint_dos",
    [
      { key: "dos_mae_500_fermi_window", label: "DOS Fermi MAE" },
      { key: "dos_wasserstein_eV", label: "DOS W1" },
    ],
    "K-point weighted DOS",
    "error",
  );
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
  const status = document.getElementById("plots-status");
  if (state.plotsEnabled && !window.Plotly) {
    if (status) status.textContent = "Cargando Plotly...";
    await ensurePlotlyLoaded();
  }
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

function setVisible(selector, visible) {
  document.querySelectorAll(selector).forEach((node) => {
    node.classList.toggle("hidden", !visible);
  });
}

function updateDatasetBuilderPanel() {
  const panel = document.getElementById("dataset-builder-panel");
  if (!panel) return;
  const runMode = document.getElementById("run-mode")?.value;
  const enabled = ["dataset_only", "full_strict_pipeline"].includes(runMode);
  panel.classList.toggle("hidden", !enabled);
  panel.open = enabled;
}

function updateReusableDatasetPanel() {
  const panel = document.getElementById("reusable-dataset-panel");
  if (!panel) return;
  const downstreamOnly = ["train_test_metrics_plots_only", "graph2mat_deeph_comparison"].includes(
    document.getElementById("run-mode")?.value,
  );
  panel.classList.toggle("hidden", !downstreamOnly);
  if (downstreamOnly) {
    if (state.reusableDatasetsLoaded) {
      renderReusableDatasets(state.reusableDatasets);
    } else {
      loadReusableDatasets().catch((error) => showToast(error.message));
    }
  }
}

function updateDeepHComparisonPanel() {
  const panel = document.getElementById("deeph-comparison-panel");
  if (!panel) return;
  const enabled = ["deeph_comparison", "graph2mat_deeph_comparison"].includes(
    document.getElementById("run-mode")?.value,
  );
  panel.classList.toggle("hidden", !enabled);
}

function updateExperimentModePanels() {
  const runMode = document.getElementById("run-mode")?.value;
  const deepHOnly = runMode === "deeph_comparison";
  const graph2matModes = ["dataset_only", "full_strict_pipeline", "train_test_metrics_plots_only", "graph2mat_deeph_comparison"];
  const trainingModes = ["full_strict_pipeline", "train_test_metrics_plots_only", "graph2mat_deeph_comparison"];
  const showGraph2MatControls = graph2matModes.includes(runMode);
  const showTrainingControls = trainingModes.includes(runMode);

  setVisible("#method-selection-panel", showGraph2MatControls);
  setVisible(".material-selector-panel", !deepHOnly);
  setVisible(".run-mode-extra-field", !deepHOnly);
  setVisible("#split-controls-panel", !deepHOnly);
  setVisible("#venv-activation-panel", !deepHOnly);
  setVisible("#training-config-block", showTrainingControls);
  setVisible("#hyperparameter-sweep-panel", showTrainingControls);
  updateDatasetBuilderPanel();
  updateDeepHComparisonPanel();
  updateReusableDatasetPanel();
  updateTrainingPlanPanel();
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
  const run = target.run_id ? ` · run ${target.run_id}` : "";
  return `${target.dataset_label || target.name}${size}${run} · ${target.relative_path}`;
}

function renderDatasetTargets(targets) {
  const body = document.getElementById("dataset-cleanup-list");
  const status = document.getElementById("dataset-cleanup-status");
  if (!body || !status) return;
  body.innerHTML = "";
  status.textContent = targets.length
    ? `${targets.length} plot-visible result run${targets.length === 1 ? "" : "s"} found`
    : "No plot-visible result runs found";
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
    if (Array.isArray(target.metric_files) && target.metric_files.length) {
      const metrics = document.createElement("div");
      metrics.className = "muted-text";
      metrics.textContent = `Plot metrics: ${target.metric_files.join(", ")}`;
      pathCell.appendChild(metrics);
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
    ? "Borrar todos los runs que alimentan los plots?"
    : "Borrar los runs seleccionados que alimentan los plots?";
  const confirmed = confirmDatasetDeletion(targets, title);
  if (!confirmed) return;
  const ids = all ? targets.map((target) => target.id) : targetIds;
  const payload = await request("/api/datasets/clear", {
    method: "POST",
    body: JSON.stringify({ target_ids: ids, dry_run: false }),
  });
  state.plotData = null;
  await loadResults();
  await loadDatasetTargets();
  const removed = Array.isArray(payload.removed) ? payload.removed.length : 0;
  showToast(`Datasets borrados: ${removed}`);
}

async function clearGeneratedDatasets() {
  if (!state.datasetTargets.length) await loadDatasetTargets();
  if (!state.datasetTargets.length) {
    showToast("No hay runs visibles en plots para borrar");
    return;
  }
  await deleteDatasetTargets([], { all: true });
}

async function deleteSelectedGeneratedDatasets() {
  await deleteDatasetTargets(selectedDatasetTargetIds(), { all: false });
}

const THEME_STORAGE_KEY = "ui-theme";

function applyTheme(theme) {
  const normalized = theme === "light" ? "light" : "dark";
  if (normalized === "dark") {
    document.documentElement.removeAttribute("data-theme");
  } else {
    document.documentElement.setAttribute("data-theme", "light");
  }
  const toggle = document.getElementById("theme-toggle");
  const label = document.getElementById("theme-toggle-label");
  if (toggle) toggle.setAttribute("aria-pressed", String(normalized === "light"));
  if (label) label.textContent = normalized === "light" ? "Dark mode" : "Light mode";
  repaintPlotsForTheme();
}

function setupThemeToggle() {
  const toggle = document.getElementById("theme-toggle");
  if (!toggle) return;
  let stored = "dark";
  try {
    stored = localStorage.getItem(THEME_STORAGE_KEY) || "dark";
  } catch (error) {
    stored = "dark";
  }
  applyTheme(stored);
  toggle.addEventListener("click", () => {
    const isLight = document.documentElement.getAttribute("data-theme") === "light";
    const next = isLight ? "dark" : "light";
    applyTheme(next);
    try {
      localStorage.setItem(THEME_STORAGE_KEY, next);
    } catch (error) {
      /* localStorage unavailable; theme still applies for this session */
    }
  });
}

// --------------------------------------------------------------------------- //
// ML vs SIESTA benchmark view (Comparación ML vs SIESTA).
// Thin client over /api/ml-vs-siesta/* endpoints. No SIESTA, no training.
// --------------------------------------------------------------------------- //
let mlVsSiestaMatrixPayload = null;
let mlVsSiestaTemplateLoaded = false;

function mvsValue(id, fallback = "") {
  const node = document.getElementById(id);
  return node ? String(node.value ?? "").trim() : fallback;
}

function mvsParseIntList(text) {
  return String(text || "")
    .split(",")
    .map((token) => token.trim())
    .filter((token) => token.length)
    .map((token) => parseInt(token, 10));
}

function mvsParseIdList(text) {
  return String(text || "")
    .split(",")
    .map((token) => token.trim())
    .filter((token) => token.length)
    .map((id) => ({ id }));
}

function mvsCollectConfig() {
  const supercell = mvsParseIntList(mvsValue("mvs-supercell", "5,5,1"));
  const centralRaw = mvsValue("mvs-central-atom", "auto");
  const directions = String(mvsValue("mvs-directions", "x,y,z"))
    .split(",")
    .map((token) => token.trim().toLowerCase())
    .filter((token) => token.length);
  return {
    system: {
      input_structure: mvsValue("mvs-input-structure") || null,
      supercell: supercell.length === 3 ? supercell : [5, 5, 1],
      central_atom: centralRaw === "auto" || centralRaw === "" ? "auto" : Number(centralRaw),
    },
    derivatives: {
      enabled: true,
      displacement: Number(mvsValue("mvs-displacement", "0.01")) || 0.01,
      directions: directions.length ? directions : ["x", "y", "z"],
    },
    models: { enabled: ["graph2mat", "deeph"] },
    matrices: { targets: ["hamiltonian", "density_matrix", "overlap"] },
    dataset_mixing: { enabled: false },
    species_transfer: { enabled: false, base_species: ["C"], new_species: ["H"] },
    ui: { enable_matrix_viewer: true },
  };
}

function mvsRenderPills(containerId, entries) {
  const container = document.getElementById(containerId);
  if (!container) return;
  container.innerHTML = "";
  entries.forEach(([label, value]) => {
    const pill = document.createElement("div");
    pill.className = "result-pill";
    const strong = document.createElement("strong");
    strong.textContent = label;
    const span = document.createElement("span");
    span.textContent = value;
    pill.append(strong, span);
    container.appendChild(pill);
  });
}

async function loadMlVsSiestaTemplate() {
  if (mlVsSiestaTemplateLoaded) return;
  const payload = await request("/api/ml-vs-siesta/config-template");
  mlVsSiestaTemplateLoaded = true;
  const output = document.getElementById("mvs-generate-output");
  if (output) {
    output.textContent =
      `Plantilla cargada (${payload.example_config_path}). ` +
      `Dry-run del plan: ok=${payload.dry_run?.ok}. Pulsa "Dry-run" o "Cargar demo".`;
  }
  mvsUpdateDerivativeInfo(payload.dry_run);
}

function mvsUpdateDerivativeInfo(dryRun) {
  const central = dryRun?.checks?.central_atom?.detail;
  const disp = dryRun?.checks?.displacements?.detail;
  const targets = dryRun?.checks?.targets?.detail;
  mvsRenderPills("mvs-derivative-info", [
    ["Átomo desplazado", central ? `${central.index} (${central.symbol ?? "?"})` : "—"],
    ["Direcciones", disp?.directions ? disp.directions.join(", ") : "—"],
    ["Displacement h (Å)", disp?.displacement ?? "—"],
    ["Targets", Array.isArray(targets) ? targets.join(", ") : "—"],
    ["Render", "Reutiliza el Matrix Viewer (sección 2) para la matriz derivada."],
  ]);
}

async function mvsGenerateDryRun() {
  const config = mvsCollectConfig();
  const payload = await request("/api/ml-vs-siesta/generate-displacements", {
    method: "POST",
    body: JSON.stringify({ config, dry_run: true }),
  });
  const output = document.getElementById("mvs-generate-output");
  if (output) {
    const files = Object.keys(payload.generated_files || {});
    output.textContent =
      `Átomo central: ${payload.central_atom_index} (${payload.central_atom_symbol}). ` +
      `Supercell atoms: ${payload.supercell_atom_count}.\n` +
      `Ficheros que se generarían (${files.length}):\n  ` +
      files.map((label) => payload.generated_files[label].fdf).join("\n  ");
  }
}

async function mvsValidatePlan() {
  const config = mvsCollectConfig();
  const payload = await request("/api/ml-vs-siesta/dry-run", {
    method: "POST",
    body: JSON.stringify({ config }),
  });
  const output = document.getElementById("mvs-generate-output");
  if (output) {
    const lines = Object.entries(payload.checks || {}).map(
      ([name, entry]) => `  ${entry.ok ? "✓" : "✗"} ${name}`
    );
    output.textContent =
      `Plan válido: ${payload.ok}. ` +
      (payload.warnings?.length ? `Warnings: ${payload.warnings.join("; ")}\n` : "\n") +
      lines.join("\n");
  }
  mvsUpdateDerivativeInfo(payload);
}

function mvsSelectedMatrixValues(payload, key) {
  if (payload.matrices && payload.matrices[key]?.matrix) {
    return payload.matrices[key].matrix.values;
  }
  if (payload.differences && payload.differences[key]?.matrix) {
    return payload.differences[key].matrix.values;
  }
  return null;
}

function mvsApplyScale(values, scale) {
  if (scale !== "log_abs") return values;
  return values.map((row) => row.map((v) => Math.log10(Math.abs(v) + 1e-9)));
}

async function mvsRenderMatrix() {
  const payload = mlVsSiestaMatrixPayload;
  if (!payload) return;
  const key = mvsValue("mvs-matrix-select", "graph2mat");
  const scale = mvsValue("mvs-scale-select", "linear");
  const values = mvsSelectedMatrixValues(payload, key);
  const metricsHost = document.getElementById("mvs-matrix-metrics");
  if (metricsHost) {
    const entries = [["Target", payload.target]];
    Object.entries(payload.metrics || {}).forEach(([model, m]) => {
      entries.push([`${model} MAE`, Number(m.mae).toExponential(3)]);
      entries.push([`${model} RMSE`, Number(m.rmse).toExponential(3)]);
      entries.push([`${model} max|err|`, Number(m.max_abs_error).toExponential(3)]);
    });
    mvsRenderPills("mvs-matrix-metrics", entries);
  }
  const host = document.getElementById("mvs-matrix-heatmap");
  if (!host || !values) return;
  await ensurePlotlyLoaded();
  const z = mvsApplyScale(values, scale);
  window.Plotly.newPlot(
    host,
    [{ z, type: "heatmap", colorscale: "RdBu", reversescale: true }],
    {
      title: `${key} · ${payload.target} · ${scale}`,
      margin: { l: 40, r: 20, t: 40, b: 40 },
      height: 420,
    },
    { displayModeBar: false, responsive: true }
  );
}

async function mvsLoadDemo() {
  mlVsSiestaMatrixPayload = await request("/api/ml-vs-siesta/matrix-viewer-demo");
  await mvsRenderMatrix();
  showToast("Matrix Viewer demo cargado (payload sintético).");
}

async function mvsMixDatasets() {
  const body = {
    small: mvsParseIdList(mvsValue("mvs-small-ids")),
    large: mvsParseIdList(mvsValue("mvs-large-ids")),
    mode: mvsValue("mvs-mix-mode", "add"),
    ratios: String(mvsValue("mvs-ratios", "0.0,0.5,1.0"))
      .split(",")
      .map((token) => Number(token.trim()))
      .filter((value) => !Number.isNaN(value)),
    seed: Number(mvsValue("mvs-seed", "0")) || 0,
  };
  const payload = await request("/api/ml-vs-siesta/mix-datasets", {
    method: "POST",
    body: JSON.stringify(body),
  });
  const output = document.getElementById("mvs-mix-output");
  if (output) {
    const lines = (payload.partitions || []).map(
      (p) => `  ${p.label} ratio=${p.ratio} → ${p.n_selected} (${p.n_large_selected} large)`
    );
    output.textContent =
      `mode=${payload.mode} seed=${payload.seed} small=${payload.n_small} large=${payload.n_large}\n` +
      lines.join("\n");
  }
}

async function mvsInspectSpecies() {
  const base = mvsValue("mvs-base-species", "C")
    .split(",")
    .map((s) => s.trim())
    .filter((s) => s.length);
  const newSpecies = mvsValue("mvs-new-species", "H")
    .split(",")
    .map((s) => s.trim())
    .filter((s) => s.length);
  const expandable = mvsValue("mvs-expandable", "false") === "true";
  const payload = await request("/api/ml-vs-siesta/inspect-species", {
    method: "POST",
    body: JSON.stringify({
      config: { supported_species: base, expandable },
      new_species: newSpecies,
    }),
  });
  mvsRenderPills("mvs-species-output", [
    ["Estado", payload.status],
    ["Soportadas", (payload.supported_species || []).join(", ") || "—"],
    ["Faltan (especies)", (payload.missing_species || []).join(", ") || "—"],
    ["Pares faltantes", (payload.missing_species_pairs || []).map((p) => p.join("-")).join(", ") || "—"],
    ["Requiere embeddings", String(payload.requires_new_embeddings)],
    ["Requiere heads", String(payload.requires_new_heads)],
  ]);
}

function mvsBind(id, event, handler) {
  const node = document.getElementById(id);
  if (node) {
    node.addEventListener(event, () => {
      Promise.resolve(handler()).catch((error) => showToast(error.message));
    });
  }
}

function setupMlVsSiesta() {
  mvsBind("mvs-load-template", "click", () => {
    mlVsSiestaTemplateLoaded = false;
    return loadMlVsSiestaTemplate();
  });
  mvsBind("mvs-generate-dryrun", "click", mvsGenerateDryRun);
  mvsBind("mvs-validate-plan", "click", mvsValidatePlan);
  mvsBind("mvs-load-demo", "click", mvsLoadDemo);
  mvsBind("mvs-matrix-select", "change", mvsRenderMatrix);
  mvsBind("mvs-scale-select", "change", mvsRenderMatrix);
  mvsBind("mvs-mix", "click", mvsMixDatasets);
  mvsBind("mvs-inspect-species", "click", mvsInspectSpecies);
}

// --------------------------------------------------------------------------- //
// Mixing datasets view: small + 5x5x1 large sweep -> MAE vs dataset size.
// --------------------------------------------------------------------------- //
let mixDiscoverLoaded = false;
let mixStatusTimer = null;
let mixLastStatusSignature = "";
let mixMetricsPayload = null;
let mixSelectedPayloadIds = new Set();
let mixPayloadSelectionInitialized = false;
let mixKnownPayloadIds = new Set();
let mixOpenPayloadGroups = new Set();
const MIXING_MAE_EV_TO_MEV = 1000;

function mixParseMap(text) {
  const map = {};
  String(text || "")
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line.length && line.includes("="))
    .forEach((line) => {
      const [size, root] = line.split("=", 2);
      const n = parseInt(size.trim(), 10);
      if (!Number.isNaN(n)) map[n] = root.trim();
    });
  return map;
}

function mixCollectBody() {
  const modes = [];
  if (document.getElementById("mix-mode-add")?.checked) modes.push("add");
  if (document.getElementById("mix-mode-replace")?.checked) modes.push("replace");
  const models = [];
  if (document.getElementById("mix-model-g2m")?.checked) models.push("graph2mat");
  if (document.getElementById("mix-model-deeph")?.checked) models.push("deeph");
  const ratios = String(mvsValue("mix-ratios", "0.0,0.2,0.4,0.6,0.8,1.0"))
    .split(",")
    .map((token) => Number(token.trim()))
    .filter((value) => !Number.isNaN(value));
  const sizesText = mvsValue("mix-sizes", "");
  const sizes = sizesText
    ? sizesText.split(",").map((token) => parseInt(token.trim(), 10)).filter((v) => !Number.isNaN(v))
    : null;
  const body = {
    small: mixParseMap(mvsValue("mix-small-map")),
    large: mixParseMap(mvsValue("mix-large-map")),
    modes: modes.length ? modes : ["add", "replace"],
    ratios: ratios.length ? ratios : [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    seed: Number(mvsValue("mix-seed", "0")) || 0,
    split_policy: mvsValue("mix-split-policy", "fixed_common_test") || "fixed_common_test",
    models: models.length ? models : ["graph2mat", "deeph"],
    performance: {
      compute_accelerator: "gpu",
      max_parallel_graph2mat_training_jobs: 7,
      max_parallel_deeph_training_jobs: 5,
      model_batch_schedule: "alternating",
      model_batch_start: "deeph",
      omp_num_threads: 2,
      mkl_num_threads: 2,
      openblas_num_threads: 2,
      numexpr_num_threads: 2,
      torch_num_threads: 2,
      torch_float32_matmul_precision: "high",
      torch_mixed_precision: "bf16-mixed",
      graph2mat_log_every_n_steps: 1,
      graph2mat_check_val_every_n_epoch: 1,
      graph2mat_checkpoint_every_n_epochs: 1,
      graph2mat_require_cuequivariance: true,
    },
  };
  if (sizes && sizes.length) body.sizes = sizes;
  return body;
}

function mixSetStatus(text, state) {
  const dot = document.getElementById("mix-status-dot");
  const label = document.getElementById("mix-status-text");
  if (label) label.textContent = text;
  if (dot) {
    dot.classList.toggle("running", state === "running" || state === "ok");
    dot.classList.toggle("error", state === "error");
  }
}

function mixAppendPayload(label, payload) {
  const log = document.getElementById("mix-payload-log");
  if (!log) return;
  const text = JSON.stringify(payload, null, 2);
  const stamp = new Date().toLocaleTimeString();
  const prefix = log.textContent.trim() === "Esperando acciones en Mixing datasets." ? "" : `${log.textContent}\n\n`;
  log.textContent = `${prefix}[${stamp}] ${label}\n${text}`;
  log.scrollTop = log.scrollHeight;
}

function mixScrollPayloadLogToBottom() {
  const log = document.getElementById("mix-payload-log");
  if (log) log.scrollTop = log.scrollHeight;
}

function mixClearPayloadLog() {
  const log = document.getElementById("mix-payload-log");
  if (log) log.textContent = "Esperando acciones en Mixing datasets.";
}

function mixPayloadsForMetrics(payload) {
  const fromPayload = Array.isArray(payload?.payloads) ? payload.payloads : [];
  if (fromPayload.length) return fromPayload;
  const seen = new Map();
  for (const curve of payload?.curves || []) {
    for (const point of curve.points || []) {
      const id = point.payload_id;
      if (!id || seen.has(id)) continue;
      seen.set(id, { id, label: id, total_size: point.total_size });
    }
  }
  return Array.from(seen.values());
}

function mixPayloadModels(payload, payloadId) {
  return Array.from(
    new Set(
      (payload?.curves || [])
        .filter((curve) => (curve.points || []).some((point) => point.payload_id === payloadId))
        .map((curve) => curve.model)
        .filter(Boolean)
    )
  );
}

function mixRatioSlug(ratio) {
  return `r${Number(ratio).toFixed(3)}`.replace(".", "p");
}

// Ids must match the backend's canonical (size, mode, ratio) scheme
// (see pipeline_ui._mixing_payload_id) so plan-preview payloads merge with
// payloads coming from /api/mixing/metrics instead of appearing as duplicates.
function mixPlanPayloads(plan) {
  return (plan?.permutations || []).map((item) => ({
    id: `size${item.size}_${item.mode}_${mixRatioSlug(item.ratio)}`,
    label: `size=${item.size} ${item.mode} ratio=${item.ratio}`,
    size: item.size,
    mode: item.mode,
    ratio: item.ratio,
    total_size: item.total_size,
    status: item.status || "planned",
    output_root: item.output_root,
  }));
}

function mixMergePayloadLists(existing, extra) {
  const byId = new Map();
  for (const item of [...(existing || []), ...(extra || [])]) {
    if (item?.id == null) continue;
    byId.set(String(item.id), { ...(byId.get(String(item.id)) || {}), ...item, id: String(item.id) });
  }
  return Array.from(byId.values()).sort((a, b) => (
    (Number(a.size || 0) - Number(b.size || 0)) ||
    String(a.mode || "").localeCompare(String(b.mode || "")) ||
    (Number(a.ratio || 0) - Number(b.ratio || 0)) ||
    String(a.id || "").localeCompare(String(b.id || ""))
  ));
}

function mixSetMetricsPayload(payload, extraPayloads = []) {
  mixMetricsPayload = {
    ...(payload || {}),
    payloads: mixMergePayloadLists(payload?.payloads || [], extraPayloads),
  };
  mixRenderPayloadSelector(mixMetricsPayload);
  return mixRenderChart(mixMetricsPayload);
}

function mixPayloadGroups(payload, payloads) {
  const byGroup = new Map();
  for (const item of payloads) {
    const key = item.size !== undefined ? `size:${item.size}` : "mixing";
    const group = byGroup.get(key) || {
      key,
      label: item.size !== undefined ? `size=${item.size}` : "Mixing sweep",
      items: [],
    };
    group.items.push(item);
    byGroup.set(key, group);
  }
  return Array.from(byGroup.values()).map((group) => {
    const ids = group.items.map((item) => String(item.id));
    const models = Array.from(
      new Set(ids.flatMap((id) => mixPayloadModels(payload, id).map(methodDisplayLabel)))
    ).join("+") || "sin metricas";
    const statuses = Array.from(new Set(group.items.map((item) => item.status).filter(Boolean))).join(", ");
    const totals = group.items.map((item) => Number(item.total_size)).filter((value) => Number.isFinite(value));
    const totalText = totals.length ? `total=${Math.min(...totals)}-${Math.max(...totals)}` : "";
    return { ...group, ids, models, statuses, totalText };
  });
}

function mixUpdatePayloadSelectionUi(payload, groups) {
  const status = document.getElementById("mix-payload-status");
  if (status) {
    const total = (groups || []).reduce((sum, group) => sum + group.items.length, 0);
    status.textContent = `${mixSelectedPayloadIds.size}/${total} run(s) seleccionados.`;
  }
  for (const group of groups || []) {
    const selected = group.ids.filter((id) => mixSelectedPayloadIds.has(id)).length;
    const checkbox = document.querySelector(`.mix-payload-group-checkbox[data-group="${CSS.escape(group.key)}"]`);
    if (checkbox) {
      checkbox.checked = selected === group.ids.length;
      checkbox.indeterminate = selected > 0 && selected < group.ids.length;
    }
    const meta = document.querySelector(`.mix-payload-group-meta[data-group="${CSS.escape(group.key)}"]`);
    if (meta) {
      meta.textContent = [
        `${selected}/${group.items.length} runs`,
        group.models,
        group.statuses ? `status=${group.statuses}` : "",
        group.totalText,
      ].filter(Boolean).join(" | ");
    }
  }
}

function mixRenderPayloadSelector(payload) {
  const list = document.getElementById("mix-payload-list");
  const status = document.getElementById("mix-payload-status");
  if (!list || !status) return;
  const payloads = mixPayloadsForMetrics(payload);
  const ids = payloads.map((item) => String(item.id));
  if (!payloads.length) {
    list.innerHTML = "";
    status.textContent = "No hay payloads disponibles todavia.";
    mixSelectedPayloadIds = new Set();
    mixKnownPayloadIds = new Set();
    return;
  }
  const previous = mixSelectedPayloadIds;
  const allKnownSelected = mixKnownPayloadIds.size > 0 && Array.from(mixKnownPayloadIds).every((id) => previous.has(id));
  if (!mixPayloadSelectionInitialized) {
    mixSelectedPayloadIds = new Set(ids);
    mixPayloadSelectionInitialized = true;
  } else if (allKnownSelected) {
    mixSelectedPayloadIds = new Set(ids);
  } else {
    mixSelectedPayloadIds = new Set(ids.filter((id) => previous.has(id)));
  }
  mixKnownPayloadIds = new Set(ids);
  const groups = mixPayloadGroups(payload, payloads);
  list.innerHTML = groups
    .map((group) => `
      <details class="mix-payload-group" data-group="${escapeHtml(group.key)}" ${mixOpenPayloadGroups.has(group.key) ? "open" : ""}>
        <summary>
          <input class="mix-payload-group-checkbox" data-group="${escapeHtml(group.key)}" type="checkbox" />
          <span>
            <strong>${escapeHtml(group.label)}</strong>
            <span class="mix-payload-group-meta" data-group="${escapeHtml(group.key)}"></span>
          </span>
        </summary>
        <div class="mix-run-list">
          ${group.items.map((item) => {
            const id = String(item.id);
            const label = item.label || id;
            const models = mixPayloadModels(payload, id).map(methodDisplayLabel).join("+") || "sin metricas";
            const detail = [
              models,
              item.status ? `status=${item.status}` : "",
              item.mode ? `mode=${item.mode}` : "",
              item.ratio !== undefined ? `ratio=${item.ratio}` : "",
              item.total_size !== undefined ? `total=${item.total_size}` : "",
              item.output_root || "",
            ].filter(Boolean).join(" | ");
            return `
              <label class="mix-run-option">
                <input class="mix-payload-checkbox" type="checkbox" value="${escapeHtml(id)}" ${mixSelectedPayloadIds.has(id) ? "checked" : ""} />
                <span><strong>${escapeHtml(label)}</strong><span>${escapeHtml(detail)}</span></span>
              </label>
            `;
          }).join("")}
        </div>
      </details>
    `)
    .join("");
  mixUpdatePayloadSelectionUi(payload, groups);
}

async function mixDiscover() {
  const payload = await request("/api/mixing/discover");
  mixAppendPayload("GET /api/mixing/discover", payload);
  mixDiscoverLoaded = true;
  mvsRenderPills("mix-discover-output", [
    ["Small (2 átomos)", String(payload.small.length)],
    ["Large (5×5×1)", String(payload.large.length)],
    ["Umbral átomos", String(payload.threshold_atoms)],
    ["Datasets root", payload.datasets_root],
  ]);
  const smallBox = document.getElementById("mix-small-map");
  if (smallBox && !smallBox.value.trim() && payload.small.length) {
    smallBox.value = payload.small
      .slice(0, 40)
      .map((d) => `${d.n_snapshots}=${d.root}`)
      .join("\n");
  }
  const largeBox = document.getElementById("mix-large-map");
  if (largeBox && !largeBox.value.trim() && payload.large.length) {
    largeBox.value = payload.large.map((d) => `${d.n_snapshots}=${d.root}`).join("\n");
  }
  await mixRefreshAvailablePayloads({ silent: true });
}

async function mixPreview() {
  const body = mixCollectBody();
  mixAppendPayload("POST /api/mixing/plan request", body);
  const plan = await request("/api/mixing/plan", {
    method: "POST",
    body: JSON.stringify(body),
  });
  mixAppendPayload("POST /api/mixing/plan response", plan);
  const output = document.getElementById("mix-plan-output");
  if (output) {
    const lines = (plan.permutations || []).map(
      (p) =>
        `  size=${p.size} ${p.mode} ratio=${p.ratio} → total=${p.total_size} ` +
        `(small=${p.n_small_selected}, large=${p.n_large_selected})`
    );
    output.textContent =
      `Permutaciones: ${plan.n_permutations} · sizes=${(plan.sizes || []).join(",")}\n` +
      (plan.warnings?.length ? `Warnings: ${plan.warnings.join("; ")}\n` : "") +
      lines.join("\n");
  }
  await mixSetMetricsPayload(mixMetricsPayload || {}, mixPlanPayloads(plan));
}

async function mixLaunch(action, statusText) {
  const body = mixCollectBody();
  body.action = action;
  mixAppendPayload("POST /api/mixing/launch request", body);
  const payload = await request("/api/mixing/launch", { method: "POST", body: JSON.stringify(body) });
  mixAppendPayload("POST /api/mixing/launch response", payload);
  mixLastStatusSignature = "";
  mixSetStatus(statusText, "running");
  if (mixStatusTimer) clearInterval(mixStatusTimer);
  mixStatusTimer = setInterval(() => {
    mixPollStatus().catch(() => {});
  }, 1500);
}

async function mixMaterialize() {
  await mixLaunch("materialize", "Materializando…");
}

async function mixTrain() {
  if (!window.confirm(
    "Entrenar el sweep lanza Graph2Mat/DeepH reales por permutación " +
      "(requiere modelos instalados y GPU). ¿Continuar?"
  )) {
    return;
  }
  await mixLaunch("train", "Entrenando sweep…");
}

async function mixPollStatus() {
  const status = await request("/api/mixing/status");
  const signature = JSON.stringify({
    state: status.state,
    action: status.action,
    done: status.permutations_done,
    failed: status.n_failed,
    partial: status.n_partial,
    records: (status.live_records || []).length,
    error: status.error || "",
  });
  if (signature !== mixLastStatusSignature) {
    mixAppendPayload("GET /api/mixing/status", status);
    mixLastStatusSignature = signature;
  }
  const done = status.permutations_done || 0;
  if (status.state === "completed") {
    const failed = status.n_failed || 0;
    const partial = status.n_partial || 0;
    let text = `Completado (${status.n_permutations} permutaciones)`;
    let level = "ok";
    if (failed || partial) {
      text += ` · ${failed} fallidas, ${partial} parciales`;
      level = failed ? "error" : "warning";
    }
    mixSetStatus(text, level);
    if (mixStatusTimer) clearInterval(mixStatusTimer);
    mixStatusTimer = null;
    // Refresh regardless of action: materialize updates payload status/output_root,
    // train additionally produces new MAE records.
    mixLoadMetrics(false).catch(() => {});
  } else if (status.state === "error") {
    mixSetStatus(`Error: ${status.error || ""}`, "error");
    if (mixStatusTimer) clearInterval(mixStatusTimer);
    mixStatusTimer = null;
  } else if (status.state === "running" || status.state === "starting") {
    const trained = (status.live_records || []).length;
    mixSetStatus(`En curso… ${done} permutaciones · ${trained} MAE registrados`, "running");
    if (status.action === "train" && trained > 0) {
      mixLoadMetrics(false).catch(() => {});
    }
  } else {
    mixSetStatus("Idle", "");
  }
}

// metric: "abs" -> h_mae in meV, "rel" -> relative_frobenius in %.
async function mixRenderMetricChart(payload, hostId, metric) {
  const host = document.getElementById(hostId);
  if (!host) return;
  await ensurePlotlyLoaded();
  const isRel = metric === "rel";
  const traces = (payload.curves || [])
    .map((curve) => ({
      curve,
      points: (curve.points || [])
        .filter((point) => mixSelectedPayloadIds.has(point.payload_id))
        // rel plot: skip points without a relative_frobenius value.
        .filter((point) => !isRel || point.relative_frobenius != null),
    }))
    .filter((item) => item.points.length)
    .map(({ curve, points }) => ({
      // Prefer the real training size when the backend recorded it (audit
      // Fase 8): "dataset size" must mean actual_train_size, not train+test.
      x: points.map((p) => (p.actual_train_size != null ? p.actual_train_size : p.total_size)),
      y: points.map((p) => (isRel ? Number(p.relative_frobenius) * 100 : Number(p.mae) * MIXING_MAE_EV_TO_MEV)),
      mode: "lines+markers",
      name: curve.label,
      line: { dash: curve.mode === "replace" ? "dash" : "solid" },
      text: points.map((p) => [
        `train real: ${p.actual_train_size != null ? p.actual_train_size : "?"}`,
        `total materializado: ${p.total_size}`,
        p.n_large_train != null ? `large en train: ${p.n_large_train}` : "",
        p.actual_large_fraction_by_snapshots != null
          ? `fracción large real: ${(p.actual_large_fraction_by_snapshots * 100).toFixed(1)}%`
          : "",
        `seeds: ${p.n_seeds != null ? p.n_seeds : "?"}${p.exploratory ? " (EXPLORATORY)" : ""}`,
        curve.evaluation_scope ? `test scope: ${curve.evaluation_scope}` : "",
        curve.training_weighting_policy ? `loss: ${curve.training_weighting_policy}` : "",
      ].filter(Boolean).join("<br>")),
      hovertemplate: isRel
        ? "rel. Frobenius %{y:.3f} %<br>%{text}<extra>%{fullData.name}</extra>"
        : "MAE %{y:.3f} meV<br>%{text}<extra>%{fullData.name}</extra>",
    }));
  if (!traces.length) {
    window.Plotly.purge(host);
    host.innerHTML = `<p class="field-help">${
      !(payload.curves || []).length
        ? "Sin datos todavía (entrena el sweep o carga la demo)."
        : isRel
          ? "Los payloads seleccionados no tienen error relativo disponible todavia."
          : mixSelectedPayloadIds.size
            ? "Los payloads seleccionados no tienen metricas disponibles todavia."
            : "Selecciona al menos un payload para ver sus metricas en el plot."
    }</p>`;
    return;
  }
  window.Plotly.newPlot(
    host,
    traces,
    {
      title: isRel
        ? "Error relativo (Frobenius) vs tamaño de dataset (mixing)"
        : "MAE absoluto vs tamaño de dataset (mixing)",
      xaxis: { title: "Total dataset size (snapshots)" },
      yaxis: { title: isRel ? "Relative Frobenius error (%)" : "Hamiltonian MAE (meV)" },
      margin: { l: 60, r: 20, t: 40, b: 50 },
      height: 460,
    },
    { displayModeBar: false, responsive: true }
  );
}

async function mixRenderChart(payload) {
  await mixRenderMetricChart(payload, "mix-mae-chart", "abs");
  await mixRenderMetricChart(payload, "mix-rel-chart", "rel");
}

async function mixLoadMetrics(demo, { silent = false } = {}) {
  const path = demo ? "/api/mixing/metrics-demo" : "/api/mixing/metrics";
  const payload = await request(path);
  mixAppendPayload(`GET ${path}`, payload);
  await mixSetMetricsPayload(payload);
  if (!silent && !demo && !(payload.curves || []).length) {
    showToast("Sin métricas reales todavía; usa la demo o entrena el sweep.");
  }
}

async function mixRefreshAvailablePayloads({ silent = false } = {}) {
  await mixLoadMetrics(false, { silent });
  const body = mixCollectBody();
  if (!Object.keys(body.small || {}).length || !Object.keys(body.large || {}).length) return;
  const plan = await request("/api/mixing/plan", {
    method: "POST",
    body: JSON.stringify(body),
  });
  if (!silent) mixAppendPayload("POST /api/mixing/plan response", plan);
  await mixSetMetricsPayload(mixMetricsPayload || {}, mixPlanPayloads(plan));
}

function mixSetAllPayloads(selected) {
  const payloads = mixPayloadsForMetrics(mixMetricsPayload);
  mixPayloadSelectionInitialized = true;
  mixSelectedPayloadIds = new Set(selected ? payloads.map((item) => String(item.id)) : []);
  mixRenderPayloadSelector(mixMetricsPayload);
  mixRenderChart(mixMetricsPayload || {}).catch((error) => showToast(error.message));
}

function setupMixingDatasets() {
  mvsBind("mix-discover", "click", mixDiscover);
  mvsBind("mix-preview", "click", mixPreview);
  mvsBind("mix-materialize", "click", mixMaterialize);
  mvsBind("mix-train", "click", mixTrain);
  mvsBind("mix-metrics-demo", "click", () => mixLoadMetrics(true));
  mvsBind("mix-metrics-real", "click", () => mixLoadMetrics(false));
  mvsBind("mix-payload-bottom", "click", mixScrollPayloadLogToBottom);
  mvsBind("mix-payload-clear", "click", mixClearPayloadLog);
  mvsBind("mix-payloads-all", "click", () => mixSetAllPayloads(true));
  mvsBind("mix-payloads-clear", "click", () => mixSetAllPayloads(false));
  document.getElementById("mix-payload-list")?.addEventListener("change", (event) => {
    const target = event.target;
    const payloads = mixPayloadsForMetrics(mixMetricsPayload);
    const groups = mixPayloadGroups(mixMetricsPayload, payloads);
    mixPayloadSelectionInitialized = true;
    if (target?.classList?.contains("mix-payload-group-checkbox")) {
      const details = target.closest(".mix-payload-group");
      details?.querySelectorAll(".mix-payload-checkbox").forEach((node) => {
        node.checked = target.checked;
        if (target.checked) mixSelectedPayloadIds.add(node.value);
        else mixSelectedPayloadIds.delete(node.value);
      });
    } else if (target?.classList?.contains("mix-payload-checkbox")) {
      if (target.checked) mixSelectedPayloadIds.add(target.value);
      else mixSelectedPayloadIds.delete(target.value);
    } else {
      return;
    }
    mixUpdatePayloadSelectionUi(mixMetricsPayload, groups);
    mixRenderChart(mixMetricsPayload || {}).catch((error) => showToast(error.message));
  });
  document.getElementById("mix-payload-list")?.addEventListener("toggle", (event) => {
    const target = event.target;
    if (!target?.classList?.contains("mix-payload-group")) return;
    const key = target.dataset.group;
    if (!key) return;
    if (target.open) mixOpenPayloadGroups.add(key);
    else mixOpenPayloadGroups.delete(key);
  }, true);
  document.getElementById("mix-payload-list")?.addEventListener("click", (event) => {
    if (event.target?.classList?.contains("mix-payload-group-checkbox")) event.stopPropagation();
  });
}

// --------------------------------------------------------------------------- //
// Cross testing view: source×target pairs -> MAE vs training source (source).
// Curve identity is (target, model); x = source training snapshots.
// --------------------------------------------------------------------------- //
let ctDiscoverLoaded = false;
let ctStatusTimer = null;
let ctLastStatusSignature = "";
let ctMetricsPayload = null;
let ctSelectedPayloadIds = new Set();
let ctPayloadSelectionInitialized = false;
let ctKnownPayloadIds = new Set();
let ctOpenPayloadGroups = new Set();
// Vacancy selector state (independent from the w90→5x5 selector above): its
// payloads are grouped by seed (id prefix "seed{N}::") instead of by target.
let ctVacancyMetricsPayload = null;
let ctVacancySelectedPayloadIds = new Set();
let ctVacancySelectionInitialized = false;
let ctVacancyKnownPayloadIds = new Set();
let ctVacancyOpenGroups = new Set();
let ctVacancyStatusTimer = null;
let ctVacancyLastStatusSignature = "";
let ctBilayerStatusTimer = null;
let ctBilayerLastStatusSignature = "";

function ctParseRoots(text) {
  return String(text || "")
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line.length);
}

function ctCollectBody() {
  const models = [];
  if (document.getElementById("ct-model-g2m")?.checked) models.push("graph2mat");
  if (document.getElementById("ct-model-deeph")?.checked) models.push("deeph");
  const epochsText = mvsValue("ct-epochs", "");
  const body = {
    sources: ctParseRoots(mvsValue("ct-sources")),
    targets: ctParseRoots(mvsValue("ct-targets")),
    models: models.length ? models : ["graph2mat", "deeph"],
    seed: Number(mvsValue("ct-seed", "0")) || 0,
    confirm_incomplete_hamiltonian_semantics: !!document.getElementById("ct-confirm-hamiltonian")?.checked,
    performance: {
      compute_accelerator: "gpu",
      max_parallel_graph2mat_training_jobs: 7,
      max_parallel_deeph_training_jobs: 5,
      omp_num_threads: 2,
      torch_num_threads: 2,
      torch_float32_matmul_precision: "high",
      torch_mixed_precision: "bf16-mixed",
      graph2mat_require_cuequivariance: true,
    },
  };
  if (epochsText !== "") {
    const epochs = parseInt(epochsText, 10);
    if (!Number.isNaN(epochs)) body.epochs = epochs;
  }
  return body;
}

function ctVacancyBody(action) {
  const body = { payload_path: mvsValue("ct-vacancy-payload", "").trim() };
  if (action) body.action = action;
  return body;
}

function ctSetVacancyStatus(text) {
  const status = document.getElementById("ct-vacancy-status");
  if (status) status.textContent = text;
}

function ctAppendVacancyPayload(label, payload) {
  const log = document.getElementById("ct-vacancy-log");
  if (!log) return;
  const initial = "Esperando acciones del cross testing con vacante.";
  const prefix = log.textContent.trim() === initial ? "" : `${log.textContent}\n\n`;
  log.textContent = `${prefix}[${new Date().toLocaleTimeString()}] ${label}\n${JSON.stringify(payload, null, 2)}`;
  log.scrollTop = log.scrollHeight;
}

function ctSetStatus(text, state) {
  const dot = document.getElementById("ct-status-dot");
  const label = document.getElementById("ct-status-text");
  if (label) label.textContent = text;
  if (dot) {
    dot.classList.toggle("running", state === "running" || state === "ok");
    dot.classList.toggle("error", state === "error");
  }
}

function ctAppendPayload(label, payload) {
  const log = document.getElementById("ct-payload-log");
  if (!log) return;
  const text = JSON.stringify(payload, null, 2);
  const stamp = new Date().toLocaleTimeString();
  const prefix = log.textContent.trim() === "Esperando acciones en Cross testing." ? "" : `${log.textContent}\n\n`;
  log.textContent = `${prefix}[${stamp}] ${label}\n${text}`;
  log.scrollTop = log.scrollHeight;
}

function ctScrollPayloadLogToBottom() {
  const log = document.getElementById("ct-payload-log");
  if (log) log.scrollTop = log.scrollHeight;
}

function ctClearPayloadLog() {
  const log = document.getElementById("ct-payload-log");
  if (log) log.textContent = "Esperando acciones en Cross testing.";
}

function ctPayloadsForMetrics(payload) {
  const fromPayload = Array.isArray(payload?.payloads) ? payload.payloads : [];
  if (fromPayload.length) return fromPayload;
  const seen = new Map();
  for (const curve of payload?.curves || []) {
    for (const point of curve.points || []) {
      const id = point.payload_id;
      if (!id || seen.has(id)) continue;
      seen.set(id, { id, label: id, target_id: point.target_id, source_n_snapshots: point.x });
    }
  }
  return Array.from(seen.values());
}

function ctPayloadModels(payload, payloadId) {
  return Array.from(
    new Set(
      (payload?.curves || [])
        .filter((curve) => (curve.points || []).some((point) => point.payload_id === payloadId))
        .map((curve) => curve.model)
        .filter(Boolean)
    )
  );
}

// Ids match the backend's canonical source_id__to__target_id scheme so
// plan-preview payloads merge with /api/cross-testing/metrics payloads.
function ctPlanPayloads(plan) {
  return (plan?.permutations || []).map((item) => ({
    id: item.payload_id || `${item.source_id}__to__${item.target_id}`,
    label: `${item.source_id} → ${item.target_id}`,
    source_id: item.source_id,
    target_id: item.target_id,
    source_system_label: item.source_system_label,
    target_system_label: item.target_system_label,
    source_n_snapshots: item.source_n_snapshots,
    status: item.status || "planned",
    output_root: item.output_root,
    reason: item.reason,
  }));
}

function ctMergePayloadLists(existing, extra) {
  const byId = new Map();
  for (const item of [...(existing || []), ...(extra || [])]) {
    if (item?.id == null) continue;
    byId.set(String(item.id), { ...(byId.get(String(item.id)) || {}), ...item, id: String(item.id) });
  }
  return Array.from(byId.values()).sort((a, b) => (
    String(a.target_id || "").localeCompare(String(b.target_id || "")) ||
    (Number(a.source_n_snapshots || 0) - Number(b.source_n_snapshots || 0)) ||
    String(a.id || "").localeCompare(String(b.id || ""))
  ));
}

function ctSetMetricsPayload(payload, extraPayloads = []) {
  ctMetricsPayload = {
    ...(payload || {}),
    payloads: ctMergePayloadLists(payload?.payloads || [], extraPayloads),
  };
  ctRenderPayloadSelector(ctMetricsPayload);
  return ctRenderChart(ctMetricsPayload);
}

function ctPayloadGroups(payload, payloads) {
  const byGroup = new Map();
  for (const item of payloads) {
    const key = item.target_id !== undefined ? `target:${item.target_id}` : "cross";
    const group = byGroup.get(key) || {
      key,
      label: item.target_id !== undefined ? `→ ${item.target_id}` : "Cross testing",
      items: [],
    };
    group.items.push(item);
    byGroup.set(key, group);
  }
  return Array.from(byGroup.values()).map((group) => {
    const ids = group.items.map((item) => String(item.id));
    const models = Array.from(
      new Set(ids.flatMap((id) => ctPayloadModels(payload, id).map(methodDisplayLabel)))
    ).join("+") || "sin metricas";
    const statuses = Array.from(new Set(group.items.map((item) => item.status).filter(Boolean))).join(", ");
    return { ...group, ids, models, statuses };
  });
}

function ctUpdatePayloadSelectionUi(payload, groups) {
  const status = document.getElementById("ct-payload-status");
  if (status) {
    const total = (groups || []).reduce((sum, group) => sum + group.items.length, 0);
    status.textContent = `${ctSelectedPayloadIds.size}/${total} par(es) seleccionados.`;
  }
  for (const group of groups || []) {
    const selected = group.ids.filter((id) => ctSelectedPayloadIds.has(id)).length;
    const checkbox = document.querySelector(`.ct-payload-group-checkbox[data-group="${CSS.escape(group.key)}"]`);
    if (checkbox) {
      checkbox.checked = selected === group.ids.length;
      checkbox.indeterminate = selected > 0 && selected < group.ids.length;
    }
    const meta = document.querySelector(`.ct-payload-group-meta[data-group="${CSS.escape(group.key)}"]`);
    if (meta) {
      meta.textContent = [
        `${selected}/${group.items.length} pares`,
        group.models,
        group.statuses ? `status=${group.statuses}` : "",
      ].filter(Boolean).join(" | ");
    }
  }
}

function ctRenderPayloadSelector(payload) {
  const list = document.getElementById("ct-payload-list");
  const status = document.getElementById("ct-payload-status");
  if (!list || !status) return;
  const payloads = ctPayloadsForMetrics(payload);
  const ids = payloads.map((item) => String(item.id));
  if (!payloads.length) {
    list.innerHTML = "";
    status.textContent = "No hay payloads disponibles todavia.";
    ctSelectedPayloadIds = new Set();
    ctKnownPayloadIds = new Set();
    return;
  }
  const previous = ctSelectedPayloadIds;
  const allKnownSelected = ctKnownPayloadIds.size > 0 && Array.from(ctKnownPayloadIds).every((id) => previous.has(id));
  if (!ctPayloadSelectionInitialized) {
    ctSelectedPayloadIds = new Set(ids);
    ctPayloadSelectionInitialized = true;
  } else if (allKnownSelected) {
    ctSelectedPayloadIds = new Set(ids);
  } else {
    ctSelectedPayloadIds = new Set(ids.filter((id) => previous.has(id)));
  }
  ctKnownPayloadIds = new Set(ids);
  const groups = ctPayloadGroups(payload, payloads);
  list.innerHTML = groups
    .map((group) => `
      <details class="mix-payload-group" data-group="${escapeHtml(group.key)}" ${ctOpenPayloadGroups.has(group.key) ? "open" : ""}>
        <summary>
          <input class="ct-payload-group-checkbox" data-group="${escapeHtml(group.key)}" type="checkbox" />
          <span>
            <strong>${escapeHtml(group.label)}</strong>
            <span class="ct-payload-group-meta" data-group="${escapeHtml(group.key)}"></span>
          </span>
        </summary>
        <div class="mix-run-list">
          ${group.items.map((item) => {
            const id = String(item.id);
            const label = item.label || id;
            const models = ctPayloadModels(payload, id).map(methodDisplayLabel).join("+") || "sin metricas";
            const detail = [
              models,
              item.status ? `status=${item.status}` : "",
              item.source_n_snapshots !== undefined ? `train=${item.source_n_snapshots}` : "",
              item.reason ? `motivo=${item.reason}` : "",
              item.output_root || "",
            ].filter(Boolean).join(" | ");
            return `
              <label class="mix-run-option">
                <input class="ct-payload-checkbox" type="checkbox" value="${escapeHtml(id)}" ${ctSelectedPayloadIds.has(id) ? "checked" : ""} />
                <span><strong>${escapeHtml(label)}</strong><span>${escapeHtml(detail)}</span></span>
              </label>
            `;
          }).join("")}
        </div>
      </details>
    `)
    .join("");
  ctUpdatePayloadSelectionUi(payload, groups);
}

async function ctDiscover() {
  const payload = await request("/api/cross-testing/discover");
  ctAppendPayload("GET /api/cross-testing/discover", payload);
  ctDiscoverLoaded = true;
  mvsRenderPills("ct-discover-output", [
    ["Datasets", String((payload.datasets || []).length)],
    ["Datasets root", payload.datasets_root],
  ]);
  const sourcesBox = document.getElementById("ct-sources");
  if (sourcesBox && !sourcesBox.value.trim() && (payload.datasets || []).length) {
    sourcesBox.value = payload.datasets.slice(0, 40).map((d) => d.root).join("\n");
  }
  await ctRefreshAvailablePayloads({ silent: true });
}

async function ctPreview() {
  const body = ctCollectBody();
  ctAppendPayload("POST /api/cross-testing/plan request", body);
  const plan = await request("/api/cross-testing/plan", { method: "POST", body: JSON.stringify(body) });
  ctAppendPayload("POST /api/cross-testing/plan response", plan);
  const output = document.getElementById("ct-plan-output");
  if (output) {
    const lines = (plan.permutations || []).map(
      (p) => `  ${p.source_id} → ${p.target_id} · ${p.status}` +
        (p.status === "compatible" ? ` (train=${p.source_n_snapshots})` : ` (${p.reason || ""})`)
    );
    output.textContent =
      `Pares: ${plan.n_permutations} · ${plan.n_compatible} compatibles · ${plan.n_incompatible} incompatibles\n` +
      (plan.warnings?.length ? `Warnings: ${plan.warnings.join("; ")}\n` : "") +
      lines.join("\n");
  }
  await ctSetMetricsPayload(ctMetricsPayload || {}, ctPlanPayloads(plan));
}

async function ctVacancyPreview() {
  const body = ctVacancyBody();
  ctAppendVacancyPayload("POST /api/cross-testing/plan request", body);
  try {
    const plan = await request("/api/cross-testing/plan", { method: "POST", body: JSON.stringify(body) });
    ctAppendVacancyPayload("POST /api/cross-testing/plan response", plan);
    ctSetVacancyStatus(`${plan.n_compatible || 0} compatibles · ${plan.n_incompatible || 0} incompatibles`);
  } catch (error) {
    ctAppendVacancyPayload("POST /api/cross-testing/plan error", { error: error.message });
    ctSetVacancyStatus(`Error: ${error.message}`);
    throw error;
  }
}

async function ctVacancyEvaluate() {
  const body = ctVacancyBody("predict_metrics");
  ctAppendVacancyPayload("POST /api/cross-testing/vacancy/launch request", body);
  try {
    const payload = await request("/api/cross-testing/vacancy/launch", { method: "POST", body: JSON.stringify(body) });
    ctAppendVacancyPayload("POST /api/cross-testing/vacancy/launch response", payload);
    ctVacancyLastStatusSignature = "";
    ctSetVacancyStatus("Evaluando checkpoints existentes…");
    if (ctVacancyStatusTimer) clearInterval(ctVacancyStatusTimer);
    ctVacancyStatusTimer = setInterval(() => ctVacancyPollStatus().catch(() => {}), 1500);
  } catch (error) {
    ctAppendVacancyPayload("POST /api/cross-testing/vacancy/launch error", { error: error.message });
    ctSetVacancyStatus(`Error: ${error.message}`);
    throw error;
  }
}

async function ctLaunch(action, statusText) {
  const body = ctCollectBody();
  body.action = action;
  ctAppendPayload("POST /api/cross-testing/launch request", body);
  const payload = await request("/api/cross-testing/launch", { method: "POST", body: JSON.stringify(body) });
  ctAppendPayload("POST /api/cross-testing/launch response", payload);
  ctLastStatusSignature = "";
  ctSetStatus(statusText, "running");
  if (ctStatusTimer) clearInterval(ctStatusTimer);
  ctStatusTimer = setInterval(() => {
    ctPollStatus().catch(() => {});
  }, 1500);
}

async function ctMaterialize() {
  await ctLaunch("materialize", "Materializando…");
}

async function ctTrain() {
  if (!window.confirm(
    "Entrenar el sweep lanza Graph2Mat/DeepH reales por par source→target " +
      "(requiere modelos instalados y GPU). ¿Continuar?"
  )) {
    return;
  }
  await ctLaunch("train", "Entrenando sweep…");
}

async function ctPollStatus() {
  const status = await request("/api/cross-testing/status");
  const signature = JSON.stringify({
    state: status.state,
    action: status.action,
    done: status.permutations_done,
    failed: status.n_failed,
    partial: status.n_partial,
    incompatible: status.n_incompatible,
    records: (status.live_records || []).length,
    error: status.error || "",
  });
  if (signature !== ctLastStatusSignature) {
    ctAppendPayload("GET /api/cross-testing/status", status);
    ctLastStatusSignature = signature;
  }
  const done = status.permutations_done || 0;
  if (status.state === "completed") {
    const failed = status.n_failed || 0;
    const partial = status.n_partial || 0;
    let text = `Completado (${status.n_permutations} pares)`;
    let level = "ok";
    if (failed || partial) {
      text += ` · ${failed} fallidos, ${partial} parciales`;
      level = failed ? "error" : "warning";
    }
    ctSetStatus(text, level);
    if (ctStatusTimer) clearInterval(ctStatusTimer);
    ctStatusTimer = null;
    ctLoadMetrics(false).catch(() => {});
  } else if (status.state === "error") {
    ctSetStatus(`Error: ${status.error || ""}`, "error");
    if (ctStatusTimer) clearInterval(ctStatusTimer);
    ctStatusTimer = null;
  } else if (status.state === "running" || status.state === "starting") {
    const trained = (status.live_records || []).length;
    ctSetStatus(`En curso… ${done} pares · ${trained} MAE registrados`, "running");
    if (status.action === "train" && trained > 0) {
      ctLoadMetrics(false).catch(() => {});
    }
  } else {
    ctSetStatus("Idle", "");
  }
}

async function ctVacancyPollStatus() {
  const status = await request("/api/cross-testing/vacancy/status");
  const signature = JSON.stringify({
    state: status.state,
    done: status.permutations_done,
    evaluated: status.n_evaluated,
    incompatible: status.n_incompatible,
    records: (status.live_records || []).length,
    error: status.error || "",
  });
  if (signature !== ctVacancyLastStatusSignature) {
    ctAppendVacancyPayload("GET /api/cross-testing/vacancy/status", status);
    ctVacancyLastStatusSignature = signature;
  }
  const records = (status.live_records || []).length;
  if (status.state === "completed") {
    ctSetVacancyStatus(
      `Completado · ${status.n_evaluated || 0} evaluados · ${status.n_incompatible || 0} incompatibles`
    );
    if (ctVacancyStatusTimer) clearInterval(ctVacancyStatusTimer);
    ctVacancyStatusTimer = null;
    ctVacancyLoadMetrics().catch(() => {});
  } else if (status.state === "error") {
    ctSetVacancyStatus(`Error: ${status.error || ""}`);
    if (ctVacancyStatusTimer) clearInterval(ctVacancyStatusTimer);
    ctVacancyStatusTimer = null;
  } else if (status.state === "running" || status.state === "starting") {
    ctSetVacancyStatus(`En curso · ${status.permutations_done || 0} pares · ${records} MAE`);
    if (records > 0) ctVacancyLoadMetrics().catch(() => {});
  }
}

// Vacancy selector: same shape as the w90→5x5 selector but grouped by seed
// (payload ids come prefixed "seed{N}::" from the seed-aware backend). Kept as a
// separate set of functions with its own state so it can never affect w90→5x5.
function ctVacancyPayloadGroups(payload, payloads) {
  const byGroup = new Map();
  for (const item of payloads) {
    const seed = item.seed !== undefined && item.seed !== null ? item.seed : "?";
    const key = `seed:${seed}`;
    const group = byGroup.get(key) || { key, label: `Seed ${seed}`, items: [] };
    group.items.push(item);
    byGroup.set(key, group);
  }
  return Array.from(byGroup.values())
    .sort((a, b) => a.label.localeCompare(b.label))
    .map((group) => {
      const ids = group.items.map((item) => String(item.id));
      const models = Array.from(
        new Set(ids.flatMap((id) => ctPayloadModels(payload, id).map(methodDisplayLabel)))
      ).join("+") || "sin metricas";
      const statuses = Array.from(new Set(group.items.map((item) => item.status).filter(Boolean))).join(", ");
      return { ...group, ids, models, statuses };
    });
}

function ctVacancyUpdateSelectionUi(payload, groups) {
  const status = document.getElementById("ct-vacancy-payload-status");
  if (status) {
    const total = (groups || []).reduce((sum, group) => sum + group.items.length, 0);
    status.textContent = `${ctVacancySelectedPayloadIds.size}/${total} par(es) seleccionados.`;
  }
  for (const group of groups || []) {
    const selected = group.ids.filter((id) => ctVacancySelectedPayloadIds.has(id)).length;
    const checkbox = document.querySelector(`.ct-vacancy-payload-group-checkbox[data-group="${CSS.escape(group.key)}"]`);
    if (checkbox) {
      checkbox.checked = selected === group.ids.length;
      checkbox.indeterminate = selected > 0 && selected < group.ids.length;
    }
    const meta = document.querySelector(`.ct-vacancy-payload-group-meta[data-group="${CSS.escape(group.key)}"]`);
    if (meta) {
      meta.textContent = [
        `${selected}/${group.items.length} pares`,
        group.models,
        group.statuses ? `status=${group.statuses}` : "",
      ].filter(Boolean).join(" | ");
    }
  }
}

function ctVacancyRenderPayloadSelector(payload) {
  const list = document.getElementById("ct-vacancy-payload-list");
  const status = document.getElementById("ct-vacancy-payload-status");
  if (!list || !status) return;
  const payloads = ctPayloadsForMetrics(payload);
  const ids = payloads.map((item) => String(item.id));
  if (!payloads.length) {
    list.innerHTML = "";
    status.textContent = "No hay runs de vacante todavia.";
    ctVacancySelectedPayloadIds = new Set();
    ctVacancyKnownPayloadIds = new Set();
    return;
  }
  const previous = ctVacancySelectedPayloadIds;
  const allKnownSelected = ctVacancyKnownPayloadIds.size > 0 && Array.from(ctVacancyKnownPayloadIds).every((id) => previous.has(id));
  if (!ctVacancySelectionInitialized) {
    ctVacancySelectedPayloadIds = new Set(ids);
    ctVacancySelectionInitialized = true;
  } else if (allKnownSelected) {
    ctVacancySelectedPayloadIds = new Set(ids);
  } else {
    ctVacancySelectedPayloadIds = new Set(ids.filter((id) => previous.has(id)));
  }
  ctVacancyKnownPayloadIds = new Set(ids);
  const groups = ctVacancyPayloadGroups(payload, payloads);
  list.innerHTML = groups
    .map((group) => `
      <details class="mix-payload-group" data-group="${escapeHtml(group.key)}" ${ctVacancyOpenGroups.has(group.key) ? "open" : ""}>
        <summary>
          <input class="ct-vacancy-payload-group-checkbox" data-group="${escapeHtml(group.key)}" type="checkbox" />
          <span>
            <strong>${escapeHtml(group.label)}</strong>
            <span class="ct-vacancy-payload-group-meta" data-group="${escapeHtml(group.key)}"></span>
          </span>
        </summary>
        <div class="mix-run-list">
          ${group.items.map((item) => {
            const id = String(item.id);
            const label = item.label || id;
            const models = ctPayloadModels(payload, id).map(methodDisplayLabel).join("+") || "sin metricas";
            const detail = [
              models,
              item.status ? `status=${item.status}` : "",
              item.source_n_snapshots !== undefined ? `train=${item.source_n_snapshots}` : "",
              item.output_root || "",
            ].filter(Boolean).join(" | ");
            return `
              <label class="mix-run-option">
                <input class="ct-vacancy-payload-checkbox" type="checkbox" value="${escapeHtml(id)}" ${ctVacancySelectedPayloadIds.has(id) ? "checked" : ""} />
                <span><strong>${escapeHtml(label)}</strong><span>${escapeHtml(detail)}</span></span>
              </label>
            `;
          }).join("")}
        </div>
      </details>
    `)
    .join("");
  ctVacancyUpdateSelectionUi(payload, groups);
}

function ctVacancySetAllPayloads(selected) {
  const payloads = ctPayloadsForMetrics(ctVacancyMetricsPayload);
  ctVacancySelectionInitialized = true;
  ctVacancySelectedPayloadIds = new Set(selected ? payloads.map((item) => String(item.id)) : []);
  ctVacancyRenderPayloadSelector(ctVacancyMetricsPayload);
  ctVacancyRenderChart(ctVacancyMetricsPayload || {}).catch((error) => showToast(error.message));
  ctVacancyRenderChart(ctVacancyMetricsPayload || {}, "relative_frobenius").catch((error) => showToast(error.message));
}

async function ctVacancyRenderChart(payload, metric = "mae") {
  const isFrobenius = metric === "relative_frobenius";
  const host = document.getElementById(
    isFrobenius ? "ct-vacancy-frobenius-chart" : "ct-vacancy-mae-chart"
  );
  if (!host) return;
  await ensurePlotlyLoaded();
  // Only plot points whose seed-aware payload_id is checked in the selector.
  // When the selector hasn't been populated yet, ctVacancyKnownPayloadIds is
  // empty and we plot everything (initial load before "Cargar métricas").
  const hasSelector = ctVacancyKnownPayloadIds.size > 0;
  const traces = (payload.curves || []).map((curve) => {
    const points = [...(curve.points || [])]
      .filter((point) => !isFrobenius || point.relative_frobenius != null)
      .filter((point) => !hasSelector || ctVacancySelectedPayloadIds.has(String(point.payload_id)))
      .sort((a, b) => (a.x ?? 0) - (b.x ?? 0));
    if (!points.length) return null;
    const source = curve.source_system_label || points[0].source_system_label || "source";
    const target = points[0].target_system_label || curve.target_id || "graphene_5x5_vacancy";
    const seed = curve.seed !== undefined && curve.seed !== null ? ` · seed ${curve.seed}` : "";
    return {
      x: points.map((point) => point.x),
      y: points.map((point) => isFrobenius
        ? Number(point.relative_frobenius) * 100
        : Number(point.mae) * 1000),
      mode: "lines+markers",
      name: `${methodDisplayLabel(curve.model)} · ${source} → ${target}${seed}`,
      marker: { size: 9 },
      error_y: {
        type: "data",
        array: points.map((point) => isFrobenius
          ? Number(point.relative_frobenius_std || 0) * 100
          : Number(point.mae_std || 0) * 1000),
        visible: points.some((point) => isFrobenius
          ? point.relative_frobenius_std != null
          : point.mae_std != null),
      },
      text: points.map((point) => `source: ${point.source_id}<br>seeds: ${point.n_seeds ?? "?"}`),
      hovertemplate: isFrobenius
        ? "Frobenius relativo %{y:.3f} %<br>%{text}<extra>%{fullData.name}</extra>"
        : "MAE %{y:.3f} meV<br>%{text}<extra>%{fullData.name}</extra>",
    };
  }).filter(Boolean);
  if (!traces.length) {
    window.Plotly.purge(host);
    host.innerHTML = hasSelector
      ? '<p class="field-help">Ningún run seleccionado. Marca uno o más seeds/pares arriba.</p>'
      : '<p class="field-help">Sin métricas de vacante todavía.</p>';
    return;
  }
  window.Plotly.newPlot(host, traces, {
    title: isFrobenius
      ? "Error relativo de Frobenius sobre graphene_5x5_vacancy"
      : "Cross testing sobre graphene_5x5_vacancy",
    xaxis: { title: "Snapshots de entrenamiento (source)" },
    yaxis: { title: isFrobenius ? "Relative Frobenius error (%)" : "Hamiltonian MAE (meV)" },
    margin: { l: 60, r: 20, t: 40, b: 50 },
    height: 460,
  }, { displayModeBar: false, responsive: true });
}

async function ctVacancyLoadMetrics() {
  const payload = await request("/api/cross-testing/vacancy/metrics");
  ctAppendVacancyPayload("GET /api/cross-testing/vacancy/metrics", payload);
  ctVacancyMetricsPayload = payload;
  ctVacancyRenderPayloadSelector(payload);
  await ctVacancyRenderChart(payload);
  await ctVacancyRenderChart(payload, "relative_frobenius");
  await ctVacancyLoadMatrixRuns();
}

// ---- Cross testing bilayer -> moire (independent subsection) ---------------
function ctBilayerBody(action) {
  const body = { payload_path: mvsValue("ct-bilayer-payload", "").trim() };
  if (action) body.action = action;
  return body;
}

function ctSetBilayerStatus(text) {
  const status = document.getElementById("ct-bilayer-status");
  if (status) status.textContent = text;
}

function ctAppendBilayerPayload(label, payload) {
  const log = document.getElementById("ct-bilayer-log");
  if (!log) return;
  const initial = "Esperando acciones del cross testing bicapa→moiré.";
  const prefix = log.textContent.trim() === initial ? "" : `${log.textContent}\n\n`;
  log.textContent = `${prefix}[${new Date().toLocaleTimeString()}] ${label}\n${JSON.stringify(payload, null, 2)}`;
  log.scrollTop = log.scrollHeight;
}

async function ctBilayerPreview() {
  const body = ctBilayerBody();
  ctAppendBilayerPayload("POST /api/cross-testing/plan request", body);
  try {
    const plan = await request("/api/cross-testing/plan", { method: "POST", body: JSON.stringify(body) });
    ctAppendBilayerPayload("POST /api/cross-testing/plan response", plan);
    ctSetBilayerStatus(`${plan.n_compatible || 0} compatibles · ${plan.n_incompatible || 0} incompatibles`);
  } catch (error) {
    ctAppendBilayerPayload("POST /api/cross-testing/plan error", { error: error.message });
    ctSetBilayerStatus(`Error: ${error.message}`);
    throw error;
  }
}

async function ctBilayerEvaluate() {
  const body = ctBilayerBody("predict_metrics");
  ctAppendBilayerPayload("POST /api/cross-testing/bilayer/launch request", body);
  try {
    const payload = await request("/api/cross-testing/bilayer/launch", { method: "POST", body: JSON.stringify(body) });
    ctAppendBilayerPayload("POST /api/cross-testing/bilayer/launch response", payload);
    ctBilayerLastStatusSignature = "";
    ctSetBilayerStatus("Evaluando checkpoints existentes…");
    if (ctBilayerStatusTimer) clearInterval(ctBilayerStatusTimer);
    ctBilayerStatusTimer = setInterval(() => ctBilayerPollStatus().catch(() => {}), 1500);
  } catch (error) {
    ctAppendBilayerPayload("POST /api/cross-testing/bilayer/launch error", { error: error.message });
    ctSetBilayerStatus(`Error: ${error.message}`);
    throw error;
  }
}

async function ctBilayerPollStatus() {
  const status = await request("/api/cross-testing/bilayer/status");
  const signature = JSON.stringify({
    state: status.state,
    done: status.permutations_done,
    evaluated: status.n_evaluated,
    incompatible: status.n_incompatible,
    records: (status.live_records || []).length,
    error: status.error || "",
  });
  if (signature !== ctBilayerLastStatusSignature) {
    ctAppendBilayerPayload("GET /api/cross-testing/bilayer/status", status);
    ctBilayerLastStatusSignature = signature;
  }
  const records = (status.live_records || []).length;
  if (status.state === "completed") {
    ctSetBilayerStatus(
      `Completado · ${status.n_evaluated || 0} evaluados · ${status.n_incompatible || 0} incompatibles`
    );
    if (ctBilayerStatusTimer) clearInterval(ctBilayerStatusTimer);
    ctBilayerStatusTimer = null;
    ctBilayerLoadMetrics().catch(() => {});
  } else if (status.state === "error") {
    ctSetBilayerStatus(`Error: ${status.error || ""}`);
    if (ctBilayerStatusTimer) clearInterval(ctBilayerStatusTimer);
    ctBilayerStatusTimer = null;
  } else if (status.state === "running" || status.state === "starting") {
    ctSetBilayerStatus(`En curso · ${status.permutations_done || 0} pares · ${records} MAE`);
    if (records > 0) ctBilayerLoadMetrics().catch(() => {});
  }
}

async function ctBilayerRenderChart(payload) {
  const host = document.getElementById("ct-bilayer-mae-chart");
  if (!host) return;
  await ensurePlotlyLoaded();
  const traces = (payload.curves || []).map((curve) => {
    const points = [...(curve.points || [])].sort((a, b) => (a.x ?? 0) - (b.x ?? 0));
    if (!points.length) return null;
    const source = curve.source_system_label || points[0].source_system_label || "graphene_hBN_bilayer";
    const target = points[0].target_system_label || curve.target_id || "graphene_hBN_moire";
    return {
      x: points.map((point) => point.x),
      y: points.map((point) => Number(point.mae) * 1000),
      mode: "lines+markers",
      name: `${methodDisplayLabel(curve.model)} · ${source} → ${target}`,
      marker: { size: 9 },
      error_y: {
        type: "data",
        array: points.map((point) => Number(point.mae_std || 0) * 1000),
        visible: points.some((point) => point.mae_std != null),
      },
      text: points.map((point) => `source: ${point.source_id}<br>seeds: ${point.n_seeds ?? "?"}`),
      hovertemplate: "MAE %{y:.3f} meV<br>%{text}<extra>%{fullData.name}</extra>",
    };
  }).filter(Boolean);
  if (!traces.length) {
    window.Plotly.purge(host);
    host.innerHTML = '<p class="field-help">Sin métricas de moiré todavía.</p>';
    return;
  }
  window.Plotly.newPlot(host, traces, {
    title: "Cross testing bicapa grafeno/hBN → moiré rotado",
    xaxis: { title: "Snapshots de entrenamiento (source)" },
    yaxis: { title: "Hamiltonian MAE (meV)" },
    margin: { l: 60, r: 20, t: 40, b: 50 },
    height: 460,
  }, { displayModeBar: false, responsive: true });
}

async function ctBilayerLoadMetrics() {
  const payload = await request("/api/cross-testing/bilayer/metrics");
  ctAppendBilayerPayload("GET /api/cross-testing/bilayer/metrics", payload);
  await ctBilayerRenderChart(payload);
}

async function ctVacancyLoadMatrixRuns() {
  const select = document.getElementById("ct-vacancy-matrix-run");
  const status = document.getElementById("ct-vacancy-matrix-status");
  if (!select) return;
  const selected = select.value;
  const payload = await request("/api/cross-testing/vacancy/matrix-errors");
  select.innerHTML = '<option value="">Selecciona un run…</option>';
  const groups = new Map();
  (payload.runs || []).forEach((run) => {
    if (!groups.has(run.payload_id)) {
      const group = document.createElement("optgroup");
      group.label = run.payload_id;
      groups.set(run.payload_id, group);
      select.appendChild(group);
    }
    const option = document.createElement("option");
    option.value = run.id;
    option.textContent = `${run.run_name} · media de ${run.sample_count} predicciones${run.cached ? " · cached" : ""}`;
    groups.get(run.payload_id).appendChild(option);
  });
  if ([...select.options].some((option) => option.value === selected)) select.value = selected;
  if (status) status.textContent = payload.n_runs
    ? `${payload.n_runs} datasets Graph2Mat disponibles.`
    : "Todavía no hay datasets Graph2Mat con predicciones de vacante.";
}

async function ctVacancyShowMatrixError() {
  const select = document.getElementById("ct-vacancy-matrix-run");
  const status = document.getElementById("ct-vacancy-matrix-status");
  const frame = document.getElementById("ct-vacancy-matrix-frame");
  const placeholder = document.getElementById("ct-vacancy-matrix-placeholder");
  const runId = select?.value || "";
  if (!runId || !frame) {
    if (frame) frame.hidden = true;
    if (placeholder) placeholder.hidden = false;
    return;
  }
  frame.hidden = true;
  if (placeholder) {
    placeholder.hidden = false;
    placeholder.textContent = "Promediando las matrices del dataset…";
  }
  if (status) status.textContent = "Calculando el MAE medio desde las predicciones Graph2Mat…";
  try {
    const result = await request("/api/cross-testing/vacancy/matrix-error", {
      method: "POST",
      body: JSON.stringify({ run_id: runId }),
    });
    frame.onload = () => {
      if (status) status.textContent = `MAE orbital · ${result.label}`;
    };
    frame.src = `${result.artifact_url}&t=${Date.now()}`;
    frame.hidden = false;
    if (placeholder) placeholder.hidden = true;
    await ctVacancyLoadMatrixRuns();
  } catch (error) {
    if (placeholder) placeholder.textContent = `No se pudo generar la matriz: ${error.message}`;
    if (status) status.textContent = "Error generando PlotMatrixError.";
  }
}

async function ctRenderChart(payload) {
  const host = document.getElementById("ct-mae-chart");
  if (!host) return;
  await ensurePlotlyLoaded();
  const markerSymbol = (point) =>
    String(point.source_id || "").includes("5x5") && String(point.target_id || "").includes("w90")
      ? "x"
      : "circle";
  const directionLabel = (point) => markerSymbol(point) === "x" ? "5x5 -> w90" : "w90 -> 5x5";
  const grouped = new Map();
  (payload.curves || []).forEach((curve) => {
    (curve.points || [])
      .filter((point) => ctSelectedPayloadIds.has(point.payload_id))
      .forEach((point) => {
        const direction = directionLabel(point);
        const key = `${curve.model}::${direction}`;
        if (!grouped.has(key)) grouped.set(key, { curve, direction, points: [] });
        grouped.get(key).points.push(point);
      });
  });
  const traces = Array.from(grouped.values()).map(({ curve, direction, points }) => {
    points.sort((a, b) => (a.x ?? a.total_size ?? 0) - (b.x ?? b.total_size ?? 0));
    return {
      x: points.map((p) => (p.x != null ? p.x : p.total_size)),
      y: points.map((p) => Number(p.mae) * 1000),
      mode: "lines+markers",
      name: `${curve.model} · ${direction}`,
      marker: {
        symbol: points.map(markerSymbol),
        size: 9,
      },
      text: points.map((p) => [
        `source: ${p.source_id}`,
        `target: ${p.target_id}`,
        `direction: ${directionLabel(p)}`,
        `snapshots train: ${p.x != null ? p.x : "?"}`,
        `model: ${curve.model}`,
        `seeds: ${p.n_seeds != null ? p.n_seeds : "?"}${p.exploratory ? " (EXPLORATORY)" : ""}`,
      ].filter(Boolean).join("<br>")),
      hovertemplate: "MAE %{y:.3f} meV<br>%{text}<extra>%{fullData.name}</extra>",
    };
  });
  if (!traces.length) {
    window.Plotly.purge(host);
    host.innerHTML = `<p class="field-help">${
      !(payload.curves || []).length
        ? "Sin datos de MAE todavía (entrena el sweep o carga la demo)."
        : ctSelectedPayloadIds.size
          ? "Los payloads seleccionados no tienen metricas disponibles todavia."
          : "Selecciona al menos un payload para ver sus metricas en el plot."
    }</p>`;
    return;
  }
  window.Plotly.newPlot(
    host,
    traces,
    {
      title: "MAE vs dataset de entrenamiento (cross testing)",
      xaxis: { title: "Snapshots de entrenamiento (source)" },
      yaxis: { title: "Hamiltonian MAE (meV)" },
      margin: { l: 60, r: 20, t: 40, b: 50 },
      height: 460,
    },
    { displayModeBar: false, responsive: true }
  );
}

async function ctLoadMetrics(demo, { silent = false } = {}) {
  const path = demo ? "/api/cross-testing/metrics-demo" : "/api/cross-testing/metrics";
  const payload = await request(path);
  ctAppendPayload(`GET ${path}`, payload);
  await ctSetMetricsPayload(payload);
  if (!silent && !demo && !(payload.curves || []).length) {
    showToast("Sin métricas reales todavía; usa la demo o entrena el sweep.");
  }
}

async function ctRefreshAvailablePayloads({ silent = false } = {}) {
  await ctLoadMetrics(false, { silent });
  const body = ctCollectBody();
  if (!(body.sources || []).length || !(body.targets || []).length) return;
  const plan = await request("/api/cross-testing/plan", { method: "POST", body: JSON.stringify(body) });
  if (!silent) ctAppendPayload("POST /api/cross-testing/plan response", plan);
  await ctSetMetricsPayload(ctMetricsPayload || {}, ctPlanPayloads(plan));
}

function ctSetAllPayloads(selected) {
  const payloads = ctPayloadsForMetrics(ctMetricsPayload);
  ctPayloadSelectionInitialized = true;
  ctSelectedPayloadIds = new Set(selected ? payloads.map((item) => String(item.id)) : []);
  ctRenderPayloadSelector(ctMetricsPayload);
  ctRenderChart(ctMetricsPayload || {}).catch((error) => showToast(error.message));
}

function setupCrossTesting() {
  mvsBind("ct-discover", "click", ctDiscover);
  mvsBind("ct-preview", "click", ctPreview);
  mvsBind("ct-materialize", "click", ctMaterialize);
  mvsBind("ct-train", "click", ctTrain);
  mvsBind("ct-vacancy-preview", "click", ctVacancyPreview);
  mvsBind("ct-vacancy-evaluate", "click", ctVacancyEvaluate);
  mvsBind("ct-vacancy-metrics", "click", ctVacancyLoadMetrics);
  mvsBind("ct-vacancy-matrix-run", "change", ctVacancyShowMatrixError);
  mvsBind("ct-bilayer-preview", "click", ctBilayerPreview);
  mvsBind("ct-bilayer-evaluate", "click", ctBilayerEvaluate);
  mvsBind("ct-bilayer-metrics", "click", ctBilayerLoadMetrics);
  mvsBind("ct-metrics-demo", "click", () => ctLoadMetrics(true));
  mvsBind("ct-metrics-real", "click", () => ctLoadMetrics(false));
  mvsBind("ct-payload-bottom", "click", ctScrollPayloadLogToBottom);
  mvsBind("ct-payload-clear", "click", ctClearPayloadLog);
  mvsBind("ct-payloads-all", "click", () => ctSetAllPayloads(true));
  mvsBind("ct-payloads-clear", "click", () => ctSetAllPayloads(false));
  document.getElementById("ct-payload-list")?.addEventListener("change", (event) => {
    const target = event.target;
    const payloads = ctPayloadsForMetrics(ctMetricsPayload);
    const groups = ctPayloadGroups(ctMetricsPayload, payloads);
    ctPayloadSelectionInitialized = true;
    if (target?.classList?.contains("ct-payload-group-checkbox")) {
      const details = target.closest(".mix-payload-group");
      details?.querySelectorAll(".ct-payload-checkbox").forEach((node) => {
        node.checked = target.checked;
        if (target.checked) ctSelectedPayloadIds.add(node.value);
        else ctSelectedPayloadIds.delete(node.value);
      });
    } else if (target?.classList?.contains("ct-payload-checkbox")) {
      if (target.checked) ctSelectedPayloadIds.add(target.value);
      else ctSelectedPayloadIds.delete(target.value);
    } else {
      return;
    }
    ctUpdatePayloadSelectionUi(ctMetricsPayload, groups);
    ctRenderChart(ctMetricsPayload || {}).catch((error) => showToast(error.message));
  });
  document.getElementById("ct-payload-list")?.addEventListener("toggle", (event) => {
    const target = event.target;
    if (!target?.classList?.contains("mix-payload-group")) return;
    const key = target.dataset.group;
    if (!key) return;
    if (target.open) ctOpenPayloadGroups.add(key);
    else ctOpenPayloadGroups.delete(key);
  }, true);
  document.getElementById("ct-payload-list")?.addEventListener("click", (event) => {
    if (event.target?.classList?.contains("ct-payload-group-checkbox")) event.stopPropagation();
  });
  mvsBind("ct-vacancy-payloads-all", "click", () => ctVacancySetAllPayloads(true));
  mvsBind("ct-vacancy-payloads-clear", "click", () => ctVacancySetAllPayloads(false));
  document.getElementById("ct-vacancy-payload-list")?.addEventListener("change", (event) => {
    const target = event.target;
    const payloads = ctPayloadsForMetrics(ctVacancyMetricsPayload);
    const groups = ctVacancyPayloadGroups(ctVacancyMetricsPayload, payloads);
    ctVacancySelectionInitialized = true;
    if (target?.classList?.contains("ct-vacancy-payload-group-checkbox")) {
      const details = target.closest(".mix-payload-group");
      details?.querySelectorAll(".ct-vacancy-payload-checkbox").forEach((node) => {
        node.checked = target.checked;
        if (target.checked) ctVacancySelectedPayloadIds.add(node.value);
        else ctVacancySelectedPayloadIds.delete(node.value);
      });
    } else if (target?.classList?.contains("ct-vacancy-payload-checkbox")) {
      if (target.checked) ctVacancySelectedPayloadIds.add(target.value);
      else ctVacancySelectedPayloadIds.delete(target.value);
    } else {
      return;
    }
    ctVacancyUpdateSelectionUi(ctVacancyMetricsPayload, groups);
    ctVacancyRenderChart(ctVacancyMetricsPayload || {}).catch((error) => showToast(error.message));
    ctVacancyRenderChart(ctVacancyMetricsPayload || {}, "relative_frobenius").catch((error) => showToast(error.message));
  });
  document.getElementById("ct-vacancy-payload-list")?.addEventListener("toggle", (event) => {
    const target = event.target;
    if (!target?.classList?.contains("mix-payload-group")) return;
    const key = target.dataset.group;
    if (!key) return;
    if (target.open) ctVacancyOpenGroups.add(key);
    else ctVacancyOpenGroups.delete(key);
  }, true);
  document.getElementById("ct-vacancy-payload-list")?.addEventListener("click", (event) => {
    if (event.target?.classList?.contains("ct-vacancy-payload-group-checkbox")) event.stopPropagation();
  });
  ctVacancyLoadMatrixRuns().catch(() => {});
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
      } else if (tab.dataset.view === "g2m-deeph") {
        pollG2MDeepHStatus().catch((error) => showToast(error.message));
        loadG2MDeepHDatasets().catch((error) => showToast(error.message));
        loadG2MDeepHResults().catch((error) => showToast(error.message));
        loadG2MDeepHPlotRuns({ preserveSelection: true })
          .then(() => loadG2MDeepHDerivativeMetrics())
          .catch((error) => showToast(error.message));
        loadDatasetMinimum().catch((error) => showToast(error.message));
      } else if (tab.dataset.view === "ml-vs-siesta") {
        loadMlVsSiestaTemplate().catch((error) => showToast(error.message));
      } else if (tab.dataset.view === "mixing-datasets") {
        if (!mixDiscoverLoaded) {
          mixDiscover().catch((error) => showToast(error.message));
        } else {
          mixRefreshAvailablePayloads({ silent: true }).catch((error) => showToast(error.message));
        }
        loadG2MDeepHPlotRuns({ preserveSelection: true })
          .then(() => loadG2MDeepHDerivativeMetrics())
          .catch((error) => showToast(error.message));
      } else if (tab.dataset.view === "cross-testing") {
        if (!ctDiscoverLoaded) {
          ctDiscover().catch((error) => showToast(error.message));
        } else {
          ctRefreshAvailablePayloads({ silent: true }).catch((error) => showToast(error.message));
        }
      } else if (tab.dataset.view === "terminal") {
        renderTerminalView();
        Promise.all([pollMixingE2ELogs(), pollMixingTerminalStatus()])
          .catch((error) => showToast(error.message));
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
    renderHyperparameterSweepPreview();
  });
  document.getElementById("training-max-ell")?.addEventListener("input", () => {
    renderHiddenIrrepsValidation();
    renderHyperparameterSweepPreview();
  });
  [
    "training-max-epochs",
    "training-optim-lr",
    "training-batch-size",
    "training-loader-threads",
    "training-loss",
    "training-loss-kwargs",
    "training-num-interactions",
    "training-correlation",
  ].forEach((id) => {
    document.getElementById(id)?.addEventListener("input", () => renderHyperparameterSweepPreview());
  });
  [
    "sweep-enabled",
    "sweep-label-prefix",
    "sweep-max-configs",
    "sweep-max-epochs",
    "sweep-optim-lr",
    "sweep-batch-size",
    "sweep-loader-threads",
    "sweep-loss",
    "sweep-loss-kwargs",
    "sweep-num-interactions",
    "sweep-correlation",
    "sweep-max-ell",
    "sweep-hidden-irreps",
    "sweep-hidden-irreps-channels",
  ].forEach((id) => {
    document.getElementById(id)?.addEventListener("input", () => renderHyperparameterSweepPreview());
    document.getElementById(id)?.addEventListener("change", () => renderHyperparameterSweepPreview());
  });
  document.getElementById("sweep-preview-first")?.addEventListener("click", () => {
    state.sweepPreviewPage = 1;
    renderHyperparameterSweepPreview();
  });
  document.getElementById("sweep-preview-prev")?.addEventListener("click", () => {
    state.sweepPreviewPage = Math.max(1, state.sweepPreviewPage - 1);
    renderHyperparameterSweepPreview();
  });
  document.getElementById("sweep-preview-next")?.addEventListener("click", () => {
    state.sweepPreviewPage += 1;
    renderHyperparameterSweepPreview();
  });
  document.getElementById("sweep-preview-last")?.addEventListener("click", () => {
    try {
      const payload = hyperparameterSweepPayload();
      const preview = expandSweepPreview(payload, trainingSettings());
      state.sweepPreviewPage = Math.max(1, Math.ceil(preview.rows.length / SWEEP_PREVIEW_PAGE_SIZE));
    } catch {
      state.sweepPreviewPage = 1;
    }
    renderHyperparameterSweepPreview();
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
    updateExperimentModePanels();
    renderHyperparameterSweepPreview();
  });
  document.addEventListener("change", (event) => {
    const target = event.target;
    if (target?.classList?.contains("reusable-dataset-checkbox") || target?.classList?.contains("planned-dataset-target-checkbox")) {
      renderHyperparameterSweepPreview();
    }
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
  document.getElementById("plot-material-filter")?.addEventListener("change", () => {
    if (state.plotsEnabled) {
      renderPlots(state.plotData);
      schedulePlotResize();
    }
  });
  document.getElementById("plot-family-filter")?.addEventListener("change", () => {
    if (state.plotsEnabled) {
      renderPlots(state.plotData);
      schedulePlotResize();
    }
  });
  document.getElementById("plot-safety-filter")?.addEventListener("change", () => {
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
  document.getElementById("g2m-deeph-validate")?.addEventListener("click", () => {
    validateG2MDeepHDataset().catch((error) => showToast(error.message));
  });
  document.getElementById("g2m-deeph-run")?.addEventListener("click", () => {
    runG2MDeepHBenchmark().catch((error) => showToast(error.message));
  });
  document.getElementById("g2m-deeph-stop")?.addEventListener("click", () => {
    stopG2MDeepHBenchmark().catch((error) => showToast(error.message));
  });
  document.getElementById("g2m-deeph-refresh-results")?.addEventListener("click", () => {
    Promise.all([
      loadG2MDeepHResults(),
      loadG2MDeepHPlotRuns().then(() => Promise.all([loadG2MDeepHPlots(), loadG2MDeepHDerivativeMetrics()])),
      loadDatasetMinimum(),
    ])
      .then(() => showToast("Graph2Mat vs DeepH refreshed"))
      .catch((error) => showToast(error.message));
  });
  document.getElementById("g2m-deeph-derivative-refresh")?.addEventListener("click", () => {
    loadG2MDeepHDerivativeMetrics().catch((error) => showToast(error.message));
  });
  document.getElementById("g2m-deeph-derivative-mae-series-all")?.addEventListener("click", () => {
    setG2MDeepHDerivativeMaeSeriesSelection(true);
  });
  document.getElementById("g2m-deeph-derivative-mae-series-clear")?.addEventListener("click", () => {
    setG2MDeepHDerivativeMaeSeriesSelection(false);
  });
  document.getElementById("g2m-deeph-dataset-minimum-refresh")?.addEventListener("click", () => {
    loadDatasetMinimum()
      .then(() => showToast("Dataset-size minimum refreshed"))
      .catch((error) => showToast(error.message));
  });
  document.getElementById("g2m-deeph-dataset-minimum-run")?.addEventListener("click", () => {
    runDatasetMinimumAnalysis().catch((error) => showToast(error.message));
  });
  
  [
      "dataset-minimum-metric",
      "dataset-minimum-threshold",
      "dataset-minimum-threshold-preset",
      "dataset-minimum-x-axis",
      "dataset-minimum-cost-basis",
      "dataset-minimum-claim-mode",
      "dataset-minimum-fit",
      "dataset-minimum-moving-average-window",
      "dataset-minimum-nmin-source",
      "dataset-minimum-aggregation-mode",
      "dataset-minimum-bootstrap-replicates",
      "dataset-minimum-ci-level",
      "dataset-minimum-show-raw-replicates",
      "dataset-minimum-criterion",
    ].forEach((id) => {
    const node = document.getElementById(id);
    node?.addEventListener("change", () => {
      if (id === "dataset-minimum-metric") {
        state.datasetMinimumThresholdUserDefined = false;
        const presets = DATASET_MINIMUM_THRESHOLD_PRESETS[datasetMinimumSelectedMetric()] || [];
        state.datasetMinimumThresholdPresetKey = presets[0]?.key || null;
        populateDatasetMinimumThresholdPresets(state.datasetMinimumPayload || {});
      }
      if (id === "dataset-minimum-threshold-preset") syncDatasetMinimumThresholdFromPreset();
      if (id === "dataset-minimum-threshold") syncDatasetMinimumThresholdPresetFromInput();
      if (id === "dataset-minimum-fit") updateDatasetMinimumMovingAverageVisibility();
      state.datasetMinimumPreviewCache = null;
      datasetMinimumInvalidatePreferredOutputIfStale();
      renderDatasetMinimumControls();
    });
    node?.addEventListener("input", () => {
      if (id === "dataset-minimum-threshold") syncDatasetMinimumThresholdPresetFromInput();
      state.datasetMinimumPreviewCache = null;
      datasetMinimumInvalidatePreferredOutputIfStale();
      renderDatasetMinimumControls();
    });
  });
  document.getElementById("g2m-deeph-plot-runs-default")?.addEventListener("click", () => {
    const recentMetricRuns = (state.g2mDeephPlotRuns || [])
      .filter((run) => run.has_metric_rows)
      .slice(0, 4)
      .map((run) => run.id);
    setG2MDeepHSelectedPlotRuns(recentMetricRuns);
    renderG2MDeepHPlotRunSelector();
    loadG2MDeepHPlots().catch((error) => showToast(error.message));
  });
  document.getElementById("g2m-deeph-plot-runs-all")?.addEventListener("click", () => {
    setG2MDeepHSelectedPlotRuns((state.g2mDeephPlotRuns || []).map((run) => run.id));
    renderG2MDeepHPlotRunSelector();
    loadG2MDeepHPlots().catch((error) => showToast(error.message));
  });
  document.getElementById("g2m-deeph-plot-runs-clear")?.addEventListener("click", () => {
    setG2MDeepHSelectedPlotRuns([]);
    renderG2MDeepHPlotRunSelector();
    loadG2MDeepHPlots().catch((error) => showToast(error.message));
  });
  document.getElementById("g2m-deeph-log-bottom")?.addEventListener("click", scrollG2MDeepHLogToBottom);
  document.getElementById("g2m-deeph-log-clear")?.addEventListener("click", () => {
    clearG2MDeepHLogView();
    showToast("Graph2Mat vs DeepH log view cleared");
  });
  document.getElementById("terminal-source")?.addEventListener("change", renderTerminalView);
  document.getElementById("terminal-refresh")?.addEventListener("click", () => {
    Promise.all([pollMixingE2ELogs(), pollMixingTerminalStatus()])
      .then(() => renderTerminalView())
      .catch((error) => showToast(error.message));
  });
  document.getElementById("terminal-bottom")?.addEventListener("click", scrollTerminalToBottom);
  document.getElementById("terminal-clear")?.addEventListener("click", () => {
    clearTerminalView();
    showToast("Terminal view cleared");
  });
  document.getElementById("g2m-deeph-refresh-datasets")?.addEventListener("click", () => {
    loadG2MDeepHDatasets()
      .then(() => showToast("Graph2Mat vs DeepH dataset list refreshed"))
      .catch((error) => showToast(error.message));
  });
  document.addEventListener("change", (event) => {
    const target = event.target;
    if (target?.classList?.contains("g2m-deeph-dataset-checkbox")) {
      selectG2MDeepHDatasetFromCheckbox(target);
    }
  });
  [
    "g2m-deeph-md-sweep-table",
    "g2m-deeph-dataset-sweep-max",
    "g2m-deeph-split-mode",
    "g2m-deeph-dataset-mode",
    "g2m-deeph-split-train",
    "g2m-deeph-split-validation",
    "g2m-deeph-split-test",
  ].forEach((id) => {
    document.getElementById(id)?.addEventListener("input", renderG2MDeepHDatasetSweepPreview);
    document.getElementById(id)?.addEventListener("change", () => {
      updateG2MDeepHDatasetPickerVisibility();
      renderG2MDeepHDatasetSweepPreview();
      renderG2MDeepHTrainingSweepPreview();
    });
  });
  [
    "g2m-deeph-training-sweep-enabled",
    "g2m-deeph-training-sweep-max-runs",
    "g2m-deeph-training-sweep-error-policy",
    "g2m-deeph-sweep-common-seeds",
    "g2m-deeph-sweep-common-epochs",
    "g2m-deeph-sweep-common-lr",
    "g2m-deeph-sweep-common-batch-size",
    "g2m-deeph-sweep-graph2mat-enabled",
    "g2m-deeph-sweep-g2m-interactions",
    "g2m-deeph-sweep-g2m-correlation",
    "g2m-deeph-sweep-g2m-max-ell",
    "g2m-deeph-sweep-g2m-hidden-channels",
    "g2m-deeph-sweep-g2m-hidden-irreps",
    "g2m-deeph-sweep-g2m-loss",
    "g2m-deeph-sweep-g2m-loss-kwargs",
    "g2m-deeph-sweep-g2m-loader-threads",
    "g2m-deeph-sweep-deeph-enabled",
    "g2m-deeph-sweep-deeph-optimizer",
    "g2m-deeph-sweep-deeph-weight-decay",
    "g2m-deeph-sweep-deeph-criterion",
    "g2m-deeph-sweep-deeph-atom-fea-len",
    "g2m-deeph-sweep-deeph-edge-fea-len",
    "g2m-deeph-sweep-deeph-gauss-stop",
    "g2m-deeph-sweep-deeph-num-l",
    "g2m-deeph-sweep-deeph-if-lcmp",
    "g2m-deeph-sweep-deeph-normalization",
    "g2m-deeph-sweep-deeph-atom-update-net",
    "g2m-deeph-sweep-deeph-retain-edge-fea",
  ].forEach((id) => {
    document.getElementById(id)?.addEventListener("input", renderG2MDeepHTrainingSweepPreview);
    document.getElementById(id)?.addEventListener("change", renderG2MDeepHTrainingSweepPreview);
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
      updateExperimentModePanels();
      renderHyperparameterSweepPreview();
    });
  });
  document.querySelectorAll(
    ".panel-summary button, .panel-summary code, .subpanel-summary button, .subpanel-summary input, .subpanel-summary label",
  ).forEach((node) => {
    node.addEventListener("click", (event) => event.stopPropagation());
  });
  window.addEventListener("resize", () => schedulePlotResize());
}

async function boot() {
  setupThemeToggle();
  setupTabs();
  setupEvents();
  setupMlVsSiesta();
  setupMixingDatasets();
  setupCrossTesting();
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
  updateExperimentModePanels();
  renderHyperparameterSweepPreview();
  updateG2MDeepHDatasetPickerVisibility();
  renderG2MDeepHDatasetSweepPreview();
  renderG2MDeepHTrainingSweepPreview();
  renderG2MDeepHArtifactSummary(null);
  renderG2MDeepHMetricSummary(null);

  await pollOnce();
  state.polling = setInterval(pollOnce, POLL_INTERVAL_MS);
}

boot().catch((error) => showToast(error.message));
