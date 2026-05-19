# MD_vs_AtomicDisplacement

Repositorio para comparar predicciones de Hamiltonianos de `graph2mat` sobre
agua usando datasets generados con SIESTA. El flujo actual ya no es solo
`MD vs AtomDisplacement`: la ruta de comparacion admite tres metodos
canonicos y permite ejecutar cualquier subconjunto de ellos.

## Metodos soportados

La fuente de verdad de identificadores esta en
`Comparison/scripts/method_registry.py`.

| ID canonico | Nombre UI | Dataset / idea | Resultados |
| --- | --- | --- | --- |
| `md` | MD | Trayectoria de dinamica molecular | `Comparison/results/results_md/` |
| `siesta_fc_cartesian` | SIESTA FC Cartesian | Desplazamientos cartesianos generados con `MD.TypeOfRun FC` | `Comparison/results/results_atomdisp/` |
| `random_cartesian` | Random Cartesian | Perturbaciones cartesianas aleatorias alrededor de la geometria relajada | `Comparison/results/results_random_cartesian/` |

Aliases legacy aceptados: `atom_displacement` y `atomdisp` se normalizan a
`siesta_fc_cartesian`.

## Punto de entrada recomendado

La ruta recomendada para experimentos comparables es la UI de `Comparison`:

```bash
python3 Comparison/scripts/pipeline_ui.py
```

Abre `http://127.0.0.1:8770`.

Desde la pestaña `Experiment` se puede:

- seleccionar uno, dos o los tres metodos;
- elegir `dataset_only`, `full_strict_pipeline` o
  `train_test_metrics_plots_only`;
- editar recetas de datasets MD, FC Cartesian y Random Cartesian;
- fijar splits, test sets, metrica primaria, rendimiento y parametros de
  entrenamiento;
- guardar todo en manifests auditables dentro de `Comparison/results/<run_id>/`.

Si no se selecciona ningun metodo, la UI y el backend rechazan el experimento.
Los modos legacy siguen existiendo: cuando no llega `selected_methods`, el
backend usa el default historico `["md", "siesta_fc_cartesian"]`.

## Estructura actual del repositorio

```text
MD_vs_AtomicDisplacement/
├── MD/
│   ├── dataset/                  # inputs, pseudopotenciales y salidas MD generadas
│   ├── scripts/                  # pipeline standalone MD
│   ├── ui/                       # UI standalone de depuracion MD
│   └── pipeline_config.yaml
├── AtomDisplacement/
│   ├── base/                     # RUN.fdf base y pseudopotenciales
│   ├── relaxed/                  # geometria relajada y basis .ion.xml
│   ├── dataset/                  # FC_steps, RandomCartesian_steps, splits y collected
│   ├── scripts/                  # FC Cartesian, Random Cartesian, single-points y Graph2Mat
│   ├── ui/                       # UI standalone de depuracion AtomDisplacement
│   └── pipeline_config.yaml
├── Comparison/
│   ├── config/                   # settings compartidas de comparacion
│   ├── dataset_recipes/          # recetas versionadas para experimentos
│   ├── results/                  # resultados archivados y manifests cientificos
│   ├── scripts/                  # UI, evaluacion cruzada, metricas y analisis de winners
│   ├── ui/                       # frontend de la UI principal
│   ├── workspaces/               # workspaces temporales por experimento
│   ├── METRICS.md
│   └── PERFORMANCE.md
├── configs/                      # configs Graph2Mat auxiliares
├── scripts/                      # utilidades de entorno y compatibilidad Torch
├── shared/                       # helpers compartidos SIESTA
├── tests/
├── requirements-graph2mat.txt
└── README.md
```

## Dependencias

Se espera un entorno con:

- SIESTA disponible como `siesta`;
- `graph2mat`;
- Python 3.12 o compatible con el entorno local;
- un virtualenv local en `.venv`.

Crear el entorno portable:

```bash
./scripts/create_graph2mat_venv.sh
```

La UI de comparacion activa por defecto:

```bash
source ${REPO_ROOT}/.venv/bin/activate
```

Si `graph2mat` no puede instalarse desde `requirements-graph2mat.txt`, instala
tu copia local:

```bash
source .venv/bin/activate
python -m pip install -e /ruta/a/graph2mat
```

## Materiales y presets

El flujo historico sigue usando H2O, pero ahora esta declarado como preset de
material en `materials/h2o/material.yaml`. Los configs principales lo seleccionan
explicitamente con:

```yaml
material:
  preset: h2o
```

Ese preset apunta al `RUN.fdf` base, pseudopotenciales y basis ya versionados
para H2O. La validacion de bundles vive en `shared/material_bundle.py` y la capa
de presets/fallback legacy en `shared/material_presets.py`. Por compatibilidad,
un config antiguo sin seccion `material` puede resolverse al preset `h2o` con una
advertencia de migracion; los nuevos materiales deberan declarar su propio bundle
antes de conectarse al pipeline en fases posteriores.

Un bundle explicito usa las mismas claves que valida el backend:

```yaml
material:
  label: sic
  fdf: materials/sic/RUN.fdf
  pseudopotential_dir: materials/sic/pseudos
  basis_dir: materials/sic/basis
  structure_type: crystal
```

En la UI, la pestaña `Experiment` incluye `Material bundle`: puedes elegir el
preset `h2o` o introducir las rutas de un bundle, pulsar `Validate material` y
ver especies, cobertura de pseudopotenciales, basis y warnings antes de lanzar
el experimento. Si seleccionas un bundle custom invalido, la UI/API no vuelve a
H2O de forma silenciosa; el inicio del experimento falla con el error de
validacion del backend.

Tambien existe una primera receta material-agnostica para AtomicDisplacement:
`AtomDisplacement/scripts/generate_generic_cartesian_displacement_dataset.py`.
Lee el bundle validado y genera estructuras `+/-x`, `+/-y`, `+/-z` para cada
atomo seleccionado, usando `atomic_displacement.recipe: generic_cartesian`. Las
recetas historicas de enlaces/angulos siguen siendo especificas de H2O y se
mantienen separadas.

## Flujo cientifico de `Comparison`

`Comparison/scripts/pipeline_ui.py` orquesta el experimento completo:

1. crea workspaces aislados por metodo y dataset;
2. genera o prepara datasets segun recetas;
3. valida muestras SIESTA antes de usarlas;
4. entrena/testea/predice en modo `full_strict_pipeline` o
   `train_test_metrics_plots_only`;
5. archiva estructuras, Hamiltonianos predichos, referencias SIESTA, configs,
   logs y manifests;
6. construye tests congelados;
7. ejecuta evaluacion cruzada metodo/test set;
8. calcula metricas sparse, espectrales, DOS y relacion matriz-espectro;
9. agrega resultados y escribe `recommendation.json`.

En `dataset_only` se generan y validan datasets, pero se omiten entrenamiento,
prediccion, evaluacion cruzada y analisis de winners. En
`train_test_metrics_plots_only` se reutiliza un dataset ya archivado con la
misma seleccion; si faltan sus carpetas, splits o referencias SIESTA, el
experimento falla antes de entrenar. En ese modo la UI muestra una tabla de
datasets archivados reutilizables; puedes marcar explicitamente los datasets
que quieras entrenar de nuevo. Si no marcas ninguno, el backend usa la
coincidencia automatica por metodo, tamano, etiqueta/receta.

Por defecto ese modo respeta los splits archivados. Si quieres mantener el
dataset fijo pero cambiar train/validation/test, selecciona `Rebuild splits
from controls` en `Split source`; la pipeline copia el dataset al workspace del
nuevo run y reconstruye los splits con los ratios y el `Split mode` elegidos,
sin regenerar SIESTA ni sobrescribir el dataset original.

Para MD, el `Split mode` por defecto es `blocked_with_gap`: separa bloques
contiguos de train, validation y test y deja un frame temporal fuera entre
particiones. El modo `spread` sigue disponible para exploracion/debug, pero se
marca con warning porque puede colocar frames MD temporalmente adyacentes en
particiones distintas.

Los YAML de entrenamiento generados pasan la validacion a Graph2Mat de forma
explicita: MD usa `training.data.val_runs` y AtomDisplacement/Random Cartesian
usan `runs.json` con la clave `val` o `val_runs` cuando el split copiado lo
permite. El `checkpoint_manifest.json` guarda la fuente de validacion,
`val_loss` como criterio de seleccion de `best-*.ckpt`, y si la seleccion queda
respaldada por un split de validacion.

Cada entrenamiento con metricas queda etiquetado en su `manifest.json` con
`training_tag`, `training_index` y el dataset base usado para entrenar. Por
ejemplo, varios entrenamientos sobre el mismo dataset aparecen como
`dataset_1000_train1`, `dataset_1000_train2`, etc.; la UI usa ese tag en plots
y tablas para distinguir reentrenamientos con hiperparametros o splits
distintos.

Para lanzar varios entrenamientos sobre datasets reutilizables, usa
`Train/test/metrics/plots only`, selecciona los datasets en `Reusable archived
datasets`, ajusta los campos de `Training parameters` y pulsa `Add current
config` en `Training plan`. Puedes repetirlo con otros hiperparametros y otra
seleccion de datasets. Al ejecutar el experimento, las configuraciones del plan
se procesan secuencialmente; cada entrada del plan crea un run independiente por
dataset seleccionado.

Un experimento con un solo metodo es valido para generar datos y diagnosticos,
pero queda marcado como `non_comparative` porque no puede producir winner
robusto.

## Artefactos de trazabilidad

Los artefactos importantes quedan bajo `Comparison/results/`:

- `Comparison/results/<run_id>/experiment_manifest.yaml`
- `Comparison/results/<run_id>/performance_report.json`
- `Comparison/results/<run_id>/summary/recommendation.json`
- `Comparison/results/<run_id>/summary/cross_evaluation_metrics.csv`
- `Comparison/results/<run_id>/common_tests/*/frozen_test_manifest.json`
- `Comparison/results/results_<method>/<dataset_label>/run_<run_id>/manifest.json`
- `metrics/sparse_metrics.csv`
- `metrics/spectral_metrics.csv`
- `metrics/dos_metrics.csv`
- `metrics/matrix_spectrum_relationship.csv`
- `metrics/orbital_pair_metrics.csv`
- `metrics/orbital_pair_summary.csv`

La recomendacion final solo debe tratarse como robusta cuando el manifest no
contiene warnings severos de leakage, settings, checkpoint, presupuesto,
metrica primaria incompleta o reproducibilidad, y hay suficientes seeds para el
criterio configurado. Experimentos de una sola seed son exploratorios.

Los CSV `orbital_pair_*` son diagnosticos para comparar mapas orbital-orbital
tipo DeepH: usa `mae_union_meV` por `species_pair`, `row_orbital_index` y
`col_orbital_index`. No son metricas H' locales exactas ni cambian los winners
por defecto.

## Datasets y recetas

La UI acepta recetas de datasets versionadas en JSON. Hay ejemplos en
`Comparison/dataset_recipes/`.

Ejemplo minimo:

```json
{
  "md": [
    {
      "recipe_id": "md_100",
      "blocks": [
        {"block_id": "md_plain", "n_snapshots": 100}
      ]
    }
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
        {
          "block_id": "rc_100",
          "n_structures": 100,
          "distribution": "gaussian",
          "sigma_ang": 0.03,
          "seed": 7
        }
      ]
    }
  ]
}
```

El tamaño del dataset ya no es la identidad cientifica completa: los manifests
propagan `recipe_id`, `block_id`, parametros de generacion, seed y hash de
receta.

`random_cartesian` mantiene compatibilidad con los bloques legacy
(`distribution`, `sigma_ang`, `uniform_range_ang`, `move_atoms`, `seed`, etc.),
pero tambien acepta componentes composables por bloque:

```json
{
  "block_id": "rc_local_h2o",
  "n_structures": 100,
  "components": {
    "atom_displacement": {"enabled": true, "sigma_ang": 0.03},
    "bond_displacement": {
      "enabled": true,
      "distribution": "uniform",
      "min_delta_ang": -0.02,
      "max_delta_ang": 0.02,
      "min_bond_ang": 0.70,
      "max_bond_ang": 1.30
    },
    "angle_displacement": {
      "enabled": true,
      "distribution": "gaussian",
      "sigma_deg": 3.0,
      "min_angle_deg": 80.0,
      "max_angle_deg": 130.0
    }
  },
  "validation": {
    "min_distance_ang": 0.65,
    "max_rmsd_from_reference_ang": null,
    "max_attempts_per_structure": 100
  }
}
```

Los componentes de enlace y angulo son, por ahora, explicitos para H2O
(`h2o_oh` y `h2o_hoh`). El metodo es una perturbacion local restringida no-MD;
no representa un ensamble termodinamico.
La pestaña `Experiment` expone estos componentes dentro de cada fila/bloque de
cada tarjeta de dataset Random Cartesian. Un dataset puede sumar varios bloques
de estructuras, y cada bloque puede activar una combinacion distinta y tener
amplitudes, rangos, limites geometricos y validacion propios.

Para materiales arbitrarios, Random Cartesian tambien acepta la receta explicita
`generic_cartesian_noise`, que lee el bundle `material`, perturba coordenadas
atomicas con una semilla determinista y escribe grupos `split_group_id` para no
separar variantes correlacionadas entre train/validation/test:

```yaml
random_cartesian:
  recipe: generic_cartesian_noise
  n_structures: 100
  max_displacement_ang: 0.05
  selected_species: null
  min_interatomic_distance_ang: 0.6
  remove_center_of_mass_translation: true
  seed: 12345
  variants_per_family: 1
```

## UI de resultados

La pestaña `Results` muestra tablas y plots Plotly. Los plots de dispersion
mantienen los puntos reales y añaden lineas de ajuste por serie:

- los puntos no se eliminan;
- los valores se ordenan por eje X;
- X duplicados se agregan por media antes de ajustar;
- NaN y datos incompletos se ignoran;
- el ajuste lineal se muestra por defecto;
- el menu dentro de cada plot permite cambiar a ajuste cuadratico o ocultar el
  ajuste;
- no se dibujan lineas que conecten punto a punto los scatter reales;
- cuando existe procedencia de material, las etiquetas/hover de los plots
  muestran el material y el selector `Material` permite filtrar por `All
  materials`, `h2o`, `sic` u otros labels archivados.

Si se muestran varios grupos de compatibilidad de material, la UI marca esos
plots como diagnosticos: no deben interpretarse como un benchmark agrupado. Las
comparaciones robustas requieren hashes compatibles de material, basis,
pseudopotenciales y ajustes SIESTA; los runs antiguos sin procedencia aparecen
como `unknown material`.

Debajo de los plots hay una seccion destructiva para datasets generados. Permite
listar artefactos, seleccionar uno o varios y borrar solo esos, o borrar todos
los generados. El backend exige IDs concretos o `all=true`, valida rutas y
rechaza enlaces simbolicos antes de borrar.

## Pipelines standalone

Los pipelines standalone siguen siendo utiles para depuracion, pero no son la
ruta recomendada para conclusiones cientificas comparativas.

MD:

```bash
python3 MD/scripts/main_md.py
```

AtomDisplacement FC Cartesian:

```bash
python3 AtomDisplacement/scripts/main_atom_displacement.py
python3 AtomDisplacement/scripts/main_atdisp.py
```

Random Cartesian se genera desde la ruta de `Comparison` o directamente con:

```bash
python3 AtomDisplacement/scripts/generate_random_cartesian_dataset.py
```

Los single-points de AtomDisplacement/Random Cartesian validan matrices SIESTA
de forma estricta por defecto: una `.TSHS`/`.HSX` solo se reutiliza si el
`RUN.out` correspondiente demuestra completion, convergencia SCF y no es stale
respecto al `RUN.fdf` cuando el repositorio puede comprobarlo. La opcion
`--allow-unvalidated-matrices` queda reservada para depuracion local y marca los
resumenes con `UNSAFE_UNVALIDATED_MATRIX_REFERENCE`.

## Scripts utiles de comparacion

```bash
python3 Comparison/scripts/material_agnostic_smoke.py --case both
python3 Comparison/scripts/verify_dataset_integrity.py --dry-run
python3 Comparison/scripts/validate_sample_bundle.py --help
python3 Comparison/scripts/check_geometry_leakage.py --help
python3 Comparison/scripts/evaluate_hamiltonian_metrics.py --help
python3 Comparison/scripts/evaluate_cross.py --help
python3 Comparison/scripts/analyze_winners.py --help
python3 Comparison/scripts/cleanup_generated_datasets.py --dry-run
```

El cleanup puede escribir `Comparison/generated_dataset_cleanup_manifest.json`
como log local generado. Ese archivo no es fuente de verdad portable y queda
ignorado por git.

## Tests y validacion local

```bash
python3 -m unittest tests/test_comparison_workflow.py
python3 -m unittest tests/test_analyze_winners_three_methods.py
python3 -m unittest tests/test_method_provenance_fairness.py
python3 -m unittest tests/test_material_agnostic_smoke.py
python3 -m unittest tests/test_three_method_scientific_smoke.py
```

Chequeos rapidos de la UI:

```bash
python3 -m py_compile Comparison/scripts/pipeline_ui.py Comparison/scripts/cleanup_generated_datasets.py
node --check Comparison/ui/app.js
```

## Documentacion relacionada

- `Comparison/METRICS.md`: definicion de metricas sparse, espectrales y DOS.
- `Comparison/PERFORMANCE.md`: controles de rendimiento disponibles en la UI.

## Limitaciones actuales

- La comparacion robusta requiere seeds suficientes; una sola seed queda como
  diagnostico exploratorio.
- Las metricas dependientes de Fermi solo son autoritativas si SIESTA proporciona
  un Fermi level real.
- Las metricas comparables con DeepH son analogos del repositorio sobre la base
  Hamiltoniana archivada. No reproducen todavia H' local, k-path bands,
  SOC/complejos, optica/Berry/shift-current, incertidumbre de ensembles ni
  escalado DeepH-vs-DFT por tamano de sistema; ver `Comparison/METRICS.md`.
- La cache experimental global sigue desactivada hasta tener claves de hash
  completas para datasets, entrenamiento, prediccion y metricas.
- Los scripts standalone pueden omitir validaciones que la ruta `Comparison`
  aplica de forma estricta.
