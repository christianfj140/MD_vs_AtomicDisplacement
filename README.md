# MD_vs_OnlyAtomDisplacement

Repositorio para comparar las predicciones de `graph2mat` entrenado con dos tipos de datasets sobre la misma molecula de agua:

1. `MD/`: dataset generado a partir de dinamica molecular.
2. `AtomDisplacement/`: dataset generado con SIESTA `MD.TypeOfRun FC`
   (frozen phonons / force constants) alrededor de una geometria relajada.

El objetivo es estudiar como cambia la calidad de las predicciones del Hamiltoniano cuando el modelo se entrena con trayectorias MD frente a un muestreo local del entorno geometrico de equilibrio.

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
- un entorno virtual en `/home/christian/graph2mat-env`

Los scripts de `AtomDisplacement` activan automáticamente ese entorno cuando llaman a `graph2mat` o a `siesta`.

## Flujo `MD`

La carpeta `MD/` contiene el pipeline original basado en dinamica molecular.

### Script principal

- [main_md.py](/home/christian/Escritorio/CINN/repositorios/MD_vs_OnlyAtomDisplacement/MD/scripts/main_md.py)

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

- [main_atom_displacement.py](/home/christian/Escritorio/CINN/repositorios/MD_vs_OnlyAtomDisplacement/AtomDisplacement/scripts/main_atom_displacement.py)

### Scripts individuales

- [run_relaxation.py](/home/christian/Escritorio/CINN/repositorios/MD_vs_OnlyAtomDisplacement/AtomDisplacement/scripts/run_relaxation.py)
- [generate_atom_displacement_dataset.py](/home/christian/Escritorio/CINN/repositorios/MD_vs_OnlyAtomDisplacement/AtomDisplacement/scripts/generate_atom_displacement_dataset.py)
- [run_single_points.py](/home/christian/Escritorio/CINN/repositorios/MD_vs_OnlyAtomDisplacement/AtomDisplacement/scripts/run_single_points.py)
- [collect_atom_displacement_dataset.py](/home/christian/Escritorio/CINN/repositorios/MD_vs_OnlyAtomDisplacement/AtomDisplacement/scripts/collect_atom_displacement_dataset.py)

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
[pipeline_config.yaml](/home/christian/Escritorio/CINN/repositorios/MD_vs_OnlyAtomDisplacement/AtomDisplacement/pipeline_config.yaml):

- `displacement`: amplitud del desplazamiento, por ejemplo `0.04 Bohr` o `0.05 Ang`
- `first_atom`: primer atomo desplazado por SIESTA FC
- `last_atom`: ultimo atomo desplazado; si es `null`, se usa el ultimo atomo de la molecula

Para H2O con 3 atomos desplazados, el numero esperado de estructuras FC es
`1 + 6 * 3 = 19`, contando la geometria de referencia.
Si se pide un dataset mayor, el pipeline mantiene ese FC canonico para fonones
y expande el dataset con amplitudes adicionales del mismo patron cartesiano
(`2 * displacement`, `3 * displacement`, ...). Esas geometrías extra se calculan
como single-points SIESTA para obtener Hamiltonianos compatibles con Graph2Mat.

Los parametros del `.fdf` base de relajacion estan en [RUN.fdf](/home/christian/Escritorio/CINN/repositorios/MD_vs_OnlyAtomDisplacement/AtomDisplacement/base/RUN.fdf).

## Pipeline de entrenamiento en `AtomDisplacement`

### Script principal

- [main_atdisp.py](/home/christian/Escritorio/CINN/repositorios/MD_vs_OnlyAtomDisplacement/AtomDisplacement/scripts/main_atdisp.py)

### Scripts individuales

- [run_atdisp_training.py](/home/christian/Escritorio/CINN/repositorios/MD_vs_OnlyAtomDisplacement/AtomDisplacement/scripts/run_atdisp_training.py)
- [run_atdisp_testing.py](/home/christian/Escritorio/CINN/repositorios/MD_vs_OnlyAtomDisplacement/AtomDisplacement/scripts/run_atdisp_testing.py)
- [run_atdisp_prediction.py](/home/christian/Escritorio/CINN/repositorios/MD_vs_OnlyAtomDisplacement/AtomDisplacement/scripts/run_atdisp_prediction.py)

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

- [samples_manifest.json](/home/christian/Escritorio/CINN/repositorios/MD_vs_OnlyAtomDisplacement/AtomDisplacement/dataset/samples_manifest.json)
- [water_atom_displacement_dataset.json](/home/christian/Escritorio/CINN/repositorios/MD_vs_OnlyAtomDisplacement/AtomDisplacement/dataset/collected/water_atom_displacement_dataset.json)
- [water_atom_displacement_summary.csv](/home/christian/Escritorio/CINN/repositorios/MD_vs_OnlyAtomDisplacement/AtomDisplacement/dataset/collected/water_atom_displacement_summary.csv)

### Entrenamiento

- [config.yaml](/home/christian/Escritorio/CINN/repositorios/MD_vs_OnlyAtomDisplacement/AtomDisplacement/training/config.yaml)
- [sample_metrics.csv](/home/christian/Escritorio/CINN/repositorios/MD_vs_OnlyAtomDisplacement/AtomDisplacement/training/sample_metrics.csv)
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
  manifest;
- la metrica primaria existe para ambos metodos, preferiblemente
  `fermi_window_rmse_eV`, despues `occupied_rmse_eV`,
  `relative_frobenius_union` o `dos_wasserstein_eV`;
- las semillas aparecen explicitamente y no hay conclusion robusta basada en
  una sola seed;
- el modo de presupuesto (`equal_sample_count`, `equal_siesta_budget` o `both`)
  esta indicado y no hay aviso grave de mismatch;
- no hay avisos fuertes de leakage geometrico, muestras invalidas o datos
  espectrales ausentes.

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

### Limitaciones actuales

- La validacion estricta puede invalidar datasets antiguos que solo guardaban
  matrices sin `RUN.out`; deben regenerarse o validarse con los outputs reales.
- La unificacion SIESTA se implementa como hash/comparacion y warning; no fuerza
  todavia la regeneracion automatica de ambos `.fdf` desde un unico template.
- Las trayectorias MD cortas y los desplazamientos cartesianos locales pueden
  producir leakage o distribuciones de test poco representativas; usa
  `check_geometry_leakage.py` y multiples seeds.
- Algunas metricas cerca de Fermi son delicadas en moleculas y dependen de que
  SIESTA proporcione un Fermi level real.
