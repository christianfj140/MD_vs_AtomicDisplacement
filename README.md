# MD_vs_OnlyAtomDisplacement

Repositorio para comparar las predicciones de `graph2mat` entrenado con dos tipos de datasets sobre la misma molecula de agua:

1. `MD/`: dataset generado a partir de dinamica molecular.
2. `AtomDisplacement/`: dataset generado con SIESTA `MD.TypeOfRun FC`
   (frozen phonons / force constants) alrededor de una geometria relajada.

El objetivo es estudiar como cambia la calidad de las predicciones del Hamiltoniano cuando el modelo se entrena con trayectorias MD frente a un muestreo local del entorno geometrico de equilibrio.

## Ruta de comparacion cientifica

Los scripts standalone de `MD/` y `AtomDisplacement/` siguen siendo utiles para
desarrollo y exploracion. Para una conclusion MD vs AtomicDisplacement, la ruta
valida es la de `Comparison/scripts/pipeline_ui.py`: construye tests congelados,
ejecuta la evaluacion cruzada 2x3, agrega metricas y termina en
`Comparison/scripts/analyze_winners.py`.

La recomendacion final solo se considera `robust_comparison` si hay al menos
tres seeds, las seis celdas cruzadas estan presentes, la metrica primaria existe
en todas ellas y no hay warnings severos de leakage, settings, budget,
checkpoint o reproducibilidad. Menos de tres seeds se reporta como
`exploratory`; cualquier warning severo deja el resultado como
`scientifically_inconclusive`.

Los artefactos principales de trazabilidad son:

- `frozen_test_manifest.json` con hashes SHA256 de estructuras, referencias y
  salidas SIESTA del test congelado.
- `training/checkpoint_manifest.json` con checkpoint elegido, hash y razon de
  seleccion.
- `cross_evaluation_manifest.json` y `recommendation.json` con hashes,
  warnings y `scientific_status`.
- `metrics/block_metrics.csv`, `metrics/species_pair_metrics.csv` y
  `metrics/distance_bin_metrics.csv` como diagnosticos estructurales del error
  sparse usando la base `.ion.xml` real. La base orbital es obligatoria; si no
  esta archivada o no cuadra con la dimension del Hamiltoniano, la ruta estricta
  aborta.

## Estructura del repositorio

```text
MD_vs_OnlyAtomDisplacement/
├── MD/
│   ├── dataset/
│   ├── scripts/
│   └── training/
├── AtomDisplacement/
│   ├── base/
│   ├── relaxed/
│   ├── dataset/
│   ├── scripts/
│   └── training/
├── README.md
└── readme.txt
```

## Dependencias

Este repo asume que existen al menos estas herramientas:

- `siesta`
- `graph2mat`
- un entorno virtual local en `.venv`

Para crear un entorno portable dentro del repo:

```bash
./scripts/create_graph2mat_venv.sh
```

La UI de comparacion usa por defecto:

```bash
source ${REPO_ROOT}/.venv/bin/activate
```

Si `graph2mat` no se puede instalar desde la URL configurada en
`requirements-graph2mat.txt`, activa `.venv` e instala tu copia local:

```bash
source .venv/bin/activate
python -m pip install -e /ruta/a/graph2mat
```

Los scripts activan automáticamente ese entorno cuando llaman a `graph2mat` o a `siesta`.

## Flujo `MD`

La carpeta `MD/` contiene el pipeline original basado en dinamica molecular.

### Script principal

- [main_md.py](MD/scripts/main_md.py)

### Orden del pipeline

1. generar dataset MD
2. entrenar
3. testear
4. predecir

### Ejecucion

Desde la raiz del repo:

```bash
python3 MD/scripts/main_md.py
```

## Flujo `AtomDisplacement`

La carpeta `AtomDisplacement/` contiene un pipeline nuevo para construir un dataset de agua deformada localmente y entrenar/testear/predecir sobre el Hamiltoniano resultante.

### Idea general

1. relajar una molecula de agua de referencia
2. ejecutar SIESTA con `MD.TypeOfRun FC`, `FC.First`, `FC.Last` y `FC.Displacement`
3. conservar el output raw de fonones (`*.FC`) y Hamiltonianos por desplazamiento
4. normalizar cada desplazamiento en `dataset/FC_steps/<step>/RUN.fdf`
5. entrenar, testear y predecir con `graph2mat`

## Carpetas importantes en `AtomDisplacement`

- `base/`: entrada base de relajacion y pseudopotenciales
- `relaxed/`: salida de la relajacion de referencia
- `dataset/AtDis_steps/`: snapshots raw copiados durante el run FC de SIESTA
- `dataset/FC_steps/`: una carpeta normalizada por desplazamiento FC
- `dataset/collected/`: resumen del dataset en JSON y CSV
- `training/`: artefactos de entrenamiento y test

## Pipeline de dataset en `AtomDisplacement`

### Script principal

- [main_atom_displacement.py](AtomDisplacement/scripts/main_atom_displacement.py)

### Scripts individuales

- [run_relaxation.py](AtomDisplacement/scripts/run_relaxation.py)
- [generate_atom_displacement_dataset.py](AtomDisplacement/scripts/generate_atom_displacement_dataset.py)
- [run_single_points.py](AtomDisplacement/scripts/run_single_points.py)
- [collect_atom_displacement_dataset.py](AtomDisplacement/scripts/collect_atom_displacement_dataset.py)

### Ejecucion completa

```bash
python3 AtomDisplacement/scripts/main_atom_displacement.py
```

### Ejecucion paso a paso

```bash
python3 AtomDisplacement/scripts/run_relaxation.py
python3 AtomDisplacement/scripts/generate_atom_displacement_dataset.py
python3 AtomDisplacement/scripts/run_single_points.py
python3 AtomDisplacement/scripts/normalize_fc_steps.py
python3 AtomDisplacement/scripts/collect_atom_displacement_dataset.py
```

## Variables importantes del dataset de `AtomDisplacement`

Los parametros FC estan en `structure.force_constants` dentro de
[pipeline_config.yaml](AtomDisplacement/pipeline_config.yaml):

- `displacement`: amplitud del desplazamiento, por ejemplo `0.04 Bohr` o `0.05 Ang`
- `first_atom`: primer atomo desplazado por SIESTA FC
- `last_atom`: ultimo atomo desplazado; si es `null`, se usa el ultimo atomo de la molecula

Para H2O con 3 atomos desplazados, el numero esperado de estructuras FC es
`1 + 6 * 3 = 19`, contando la geometria de referencia.
Si se pide un dataset mayor, el pipeline mantiene ese FC canonico para fonones
y expande el dataset con amplitudes adicionales del mismo patron cartesiano
(`2 * displacement`, `3 * displacement`, ...). Esas geometrías extra se calculan
como single-points SIESTA para obtener Hamiltonianos compatibles con Graph2Mat.

Los parametros del `.fdf` base de relajacion estan en [RUN.fdf](AtomDisplacement/base/RUN.fdf).

## Pipeline de entrenamiento en `AtomDisplacement`

### Script principal

- [main_atdisp.py](AtomDisplacement/scripts/main_atdisp.py)

### Scripts individuales

- [run_atdisp_training.py](AtomDisplacement/scripts/run_atdisp_training.py)
- [run_atdisp_testing.py](AtomDisplacement/scripts/run_atdisp_testing.py)
- [run_atdisp_prediction.py](AtomDisplacement/scripts/run_atdisp_prediction.py)

### Ejecucion completa

```bash
python3 AtomDisplacement/scripts/main_atdisp.py
```

### Ejecucion paso a paso

```bash
python3 AtomDisplacement/scripts/run_atdisp_training.py
python3 AtomDisplacement/scripts/run_atdisp_testing.py
python3 AtomDisplacement/scripts/run_atdisp_prediction.py
```

## Artefactos generados en `AtomDisplacement`

### Dataset

- [samples_manifest.json](AtomDisplacement/dataset/samples_manifest.json)
- [water_atom_displacement_dataset.json](AtomDisplacement/dataset/collected/water_atom_displacement_dataset.json)
- [water_atom_displacement_summary.csv](AtomDisplacement/dataset/collected/water_atom_displacement_summary.csv)

### Entrenamiento

- [config.yaml](AtomDisplacement/training/config.yaml)
- [sample_metrics.csv](AtomDisplacement/training/sample_metrics.csv)
- `training/lightning_logs/atom_displacement_model/...`

## Estado actual del repositorio

Actualmente el flujo `AtomDisplacement` ya fue ejecutado con una prueba de 20 muestras:

- 20 estructuras generadas
- 20 calculos single-point completados
- entrenamiento, test y prediccion ejecutados
- checkpoint disponible en `AtomDisplacement/training/lightning_logs/atom_displacement_model/version_0/checkpoints/`

## Notas

- En `AtomDisplacement/scripts/` todavia existen algunos scripts `run_md_*` heredados del flujo antiguo. No forman parte del pipeline nuevo de desplazamientos atomicos.
- Si se quiere un modelo mas util fisicamente, conviene aumentar el numero de muestras antes de sacar conclusiones del entrenamiento.

## UI de comparacion

Para lanzar `MD` y `AtomDisplacement` con un solo click:

```bash
python3 Comparison/scripts/pipeline_ui.py
```

Abre `http://127.0.0.1:8770`. La pantalla ejecuta ambos pipelines, muestra logs separados y resume los artefactos que se usaran para comparar resultados.

### Flujo cientifico recomendado

La UI es el punto de entrada recomendado para sacar conclusiones comparativas.
Los scripts sueltos siguen siendo utiles para depuracion, pero pueden omitir
partes del protocolo experimental. Antes de entrenar o evaluar, el flujo de UI
valida que cada muestra tenga `RUN.fdf`, una unica referencia `.TSHS`/`.HSX`,
`RUN.out`, completion normal de SIESTA y SCF convergido. Las salidas de
validacion quedan en `valid_samples.csv`, `invalid_samples.csv` y
`validation_summary.json`; si hay muestras invalidas, los manifests validos son
los que deben usarse para entrenamiento/evaluacion.

La comparacion MD vs AtomDisplacement solo debe interpretarse como valida si:

- ambos modelos se evaluan contra el mismo test congelado con referencias SIESTA;
- las settings SIESTA de MD y AtomDisplacement no muestran mismatch en el
  manifest; en modo de comparacion estricta la UI aborta si hay mismatch;
- los hiperparametros Graph2Mat comparables coinciden; rutas de dataset, basis
  y logger pueden diferir, pero arquitectura, loss, batch size, epocas, target
  y simetria no deben cambiar salvo en una ablation explicita;
- la metrica primaria existe para ambos metodos, preferiblemente
  `fermi_window_rmse_eV`, despues `occupied_rmse_eV`,
  `relative_frobenius_union` o `dos_wasserstein_eV`;
- las semillas aparecen explicitamente y no hay conclusion robusta basada en
  una sola seed;
- el modo de presupuesto (`equal_sample_count`, `equal_siesta_budget` o `both`)
  esta indicado y no hay aviso grave de mismatch;
- no hay avisos fuertes de leakage geometrico, muestras invalidas o datos
  espectrales ausentes.

Los entrypoints standalone de `AtomDisplacement` son conservadores: el
entrenamiento requiere `dataset/splits/train_manifest.csv` o
`train_valid_manifest.csv`; el test/predict usan el split `test` y no caen al
dataset completo salvo que se active una bandera `*_debug` en el YAML. Esto
mantiene compatibilidad para depuracion, pero evita que el flujo cientifico
entrene o prediga accidentalmente sobre todas las muestras.

La pestaña `Experiment` permite barrer tamaños de dataset, por ejemplo:

- `MD`: `50, 100, 200, 500`
- `AtomDisplacement`: `100, 1000, 10000`

Cada tamaño se ejecuta en un workspace aislado dentro de `Comparison/workspaces/` y se archiva en:

- `Comparison/results/results_md/dataset_<N>/run_<timestamp>/`
- `Comparison/results/results_atomdisp/dataset_<N>/run_<timestamp>/`

Cada carpeta de resultados guarda `structures/`, `predicted_hamiltonians/`, `siesta_hamiltonians/`, `sample_metrics.csv` cuando existe, `pipeline_config.yaml`, `run.log` y `manifest.json`. Tambien ejecuta una evaluacion post-procesado con metricas sparse, espectrales y DOS total. Las salidas principales son `metrics/sparse_metrics.csv`, `metrics/spectral_metrics.csv`, `metrics/dos_metrics.csv`, `eigenvalues/siesta/`, `eigenvalues/predicted/`, `eigenvalues/band_errors/`, `dos/` y `eigenvalues/overlap_summary.csv`. El espectro de los Hamiltonianos predichos se calcula con la matriz de solape de referencia SIESTA para evaluar el error de `H` en el problema generalizado `Hc = ESc`. En la pestaña `Results`, la casilla `Show plots` despliega graficas Plotly para comparar error espectral, error sparse, DOS, gap, distribuciones por muestra, relacion matriz-espectro y un heatmap resumen.

Scripts auxiliares de comparacion:

```bash
python3 Comparison/scripts/verify_dataset_integrity.py --dry-run
python3 Comparison/scripts/validate_sample_bundle.py --help
python3 Comparison/scripts/check_geometry_leakage.py --help
python3 Comparison/scripts/analyze_phonons.py --fc-run-dir AtomDisplacement/dataset
python3 Comparison/scripts/evaluate_cross.py
```

`evaluate_cross.py` escribe `Comparison/results/comparison/metrics.csv`. Si se le pasan
resultados cruzados explicitos (`--md-on-fc`, `--fc-on-md`, etc.), rellena tambien
las celdas Modelo/Test correspondientes.

### Smoke test minimo

Desde la raiz:

```bash
python3 -m unittest tests/test_comparison_workflow.py
python3 Comparison/scripts/pipeline_ui.py
```

Para un experimento pequeño desde la UI, usa tamaños iguales y reducidos
(`MD=10`, `AtomDisplacement=10`), `compute_budget_mode=both`, una seed fija y
revisa primero `experiment_manifest.yaml`, `validation_summary.json`,
`cross_evaluation_metrics.csv`, `winner_summary.csv` y `recommendation.json`
antes de confiar en las graficas.

### Recetas de dataset y plots

La UI de comparacion acepta ahora `dataset_recipes` opcional en el experimento.
Los campos simples (`md_sizes`, opciones FC y `random_cartesian.n_structures`)
siguen funcionando y se convierten internamente a recetas versionadas. En modo
avanzado, el tamaño ya no es la identidad del dataset: cada run queda etiquetado
por `recipe_id`, `block_id` y hash de receta, por ejemplo
`md_md_plain_md_9_9` o `rc_rc_sigma_rc_sigma_0p03_100`.

Ejemplo minimo:

```json
{
  "md": [
    {"recipe_id": "md_100", "blocks": [{"block_id": "md_plain", "n_snapshots": 100}]}
  ],
  "siesta_fc_cartesian": [
    {
      "recipe_id": "fc_mixed",
      "blocks": [
        {"block_id": "fc_0p02", "displacement": "0.02 Ang", "n_structures": 20},
        {"block_id": "fc_0p05", "displacement": "0.05 Ang", "n_structures": 20}
      ]
    }
  ],
  "random_cartesian": [
    {
      "recipe_id": "rc_sigma_0p03",
      "blocks": [
        {"block_id": "rc_100", "n_structures": 100, "distribution": "gaussian", "sigma_ang": 0.03, "seed": 7}
      ]
    }
  ]
}
```

Las recetas MD aceptan por ahora bloques de `n_snapshots`. Los campos de
temperatura/termostato se rechazan con un error claro hasta que se verifiquen y
se rendericen keywords SIESTA 5.4 compatibles en el flujo local; no se generan
controles fisicos falsos.

Los manifiestos de muestras y runs propagan metadatos de receta (`recipe_id`,
`block_id`, parametros de generacion y seed) para poder agrupar errores
prediccion-vs-SIESTA por familia. `/api/plots` devuelve todos los runs y los
experimentos cross agrupados por compatibilidad. La UI ya no oculta
silenciosamente experimentos antiguos: el selector de plots permite ver el grupo
compatible mas reciente, solo el ultimo experimento o todos los experimentos con
warning.

### Rendimiento

La pestaña `Rendimiento` de la UI controla solo opciones cableadas al backend y
registradas en `experiment_manifest.yaml` y `performance_report.json/csv`.
Por defecto el flujo conserva el comportamiento cientifico actual: jobs de
dataset, prediccion, evaluacion y metricas en serie, sin cache experimental, y
reutilizando solo salidas SIESTA locales que pasen la validacion estricta.

Controles que afectan a SIESTA:

- `max_parallel_siesta_jobs`: paraleliza single-points independientes de
  AtomDisplacement/Random Cartesian. Cada muestra se ejecuta en su propio
  directorio y se valida despues. La trayectoria MD no se paraleliza porque es
  un calculo acoplado.
- `omp_num_threads`, `mkl_num_threads`, `openblas_num_threads`,
  `numexpr_num_threads`: se aplican como variables de entorno a los
  subprocesos. Evita sobre-suscripcion: si subes `max_parallel_siesta_jobs`,
  baja los threads por job.
- `reuse_validated_siesta_outputs`: permite saltar single-points locales solo si
  `RUN.fdf`, `RUN.out` y la matriz pasan la validacion; no acepta matrices por
  mera existencia.

Controles que afectan a Graph2Mat/PyTorch:

- `compute_accelerator`: `cpu`, `gpu` o `auto`. `gpu` falla si CUDA no esta
  disponible; `auto` usa GPU si existe y si no cae a CPU con warning.
- `batch_size` y `store_in_memory`: se escriben en el YAML temporal de cada
  workspace antes de entrenar/evaluar.
- `torch_num_threads` y `torch_float32_matmul_precision`: se aplican al entorno
  de PyTorch mediante el wrapper de compatibilidad.

Controles de orquestacion:

- `max_parallel_dataset_jobs`: ejecuta jobs independientes metodo/tamaño en
  paralelo solo mediante snapshots `PIPELINE_CONFIG_PATH` y workspaces aislados.
  Con una sola GPU, el runner fuerza serializacion para evitar entrenamientos
  simultaneos en el mismo dispositivo.
- `max_parallel_prediction_jobs`, `max_parallel_evaluation_jobs` y
  `max_parallel_metric_jobs` limitan predicciones cruzadas, evaluaciones
  cruzadas y metricas Hamiltonianas por muestra cuando sus directorios de salida
  son unicos. Las partes que escriben artefactos agregados siguen en el proceso
  padre para conservar orden determinista.
- `error_policy`: `fail_fast` aborta al primer fallo; `continue_on_error`
  permite terminar tareas pendientes y marca el experimento como parcial, nunca
  como exitoso.
- `enable_experiment_cache`: reservado y desactivado por defecto. Si se activa,
  el backend falla de forma explicita porque aun no existe una clave de hash
  completa para reutilizar entrenamiento, predicciones o metricas con seguridad.

Para una estacion local grande, empieza con valores conservadores: 2-4
single-points SIESTA en paralelo, 1-4 threads por job, dataset jobs en serie si
usas GPU, y predicciones/evaluaciones en serie hasta confirmar que los outputs
son independientes. El preset `aggressive_local` rellena valores prudentes a
partir de los nucleos disponibles, pero sigue evitando paralelismo de GPU no
aislado.

### Limitaciones actuales

- La validacion estricta puede invalidar datasets antiguos que solo guardaban
  matrices sin `RUN.out`; deben regenerarse o validarse con los outputs reales.
- La unificacion SIESTA se implementa como hash/comparacion y warning; no fuerza
  todavia la regeneracion automatica de ambos `.fdf` desde un unico template.
  En la UI de comparacion estricta el warning se convierte en error.
- Las trayectorias MD cortas y los desplazamientos cartesianos locales pueden
  producir leakage o distribuciones de test poco representativas; usa
  `check_geometry_leakage.py` y multiples seeds.
- Algunas metricas cerca de Fermi son delicadas en moleculas y dependen de que
  SIESTA proporcione un Fermi level real.
- No se implementa cache global de SIESTA, checkpoints, predicciones ni metricas
  sin validacion por hash completo; hacerlo por existencia de archivos seria
  rapido pero cientificamente inseguro.
