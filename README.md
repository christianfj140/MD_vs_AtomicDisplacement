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
- elegir `dataset_only` o `full_strict_pipeline`;
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

## Flujo cientifico de `Comparison`

`Comparison/scripts/pipeline_ui.py` orquesta el experimento completo:

1. crea workspaces aislados por metodo y dataset;
2. genera o prepara datasets segun recetas;
3. valida muestras SIESTA antes de usarlas;
4. entrena/testea/predice en modo `full_strict_pipeline`;
5. archiva estructuras, Hamiltonianos predichos, referencias SIESTA, configs,
   logs y manifests;
6. construye tests congelados;
7. ejecuta evaluacion cruzada metodo/test set;
8. calcula metricas sparse, espectrales, DOS y relacion matriz-espectro;
9. agrega resultados y escribe `recommendation.json`.

En `dataset_only` se generan y validan datasets, pero se omiten entrenamiento,
prediccion, evaluacion cruzada y analisis de winners.

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

La recomendacion final solo debe tratarse como robusta cuando el manifest no
contiene warnings severos de leakage, settings, checkpoint, presupuesto,
metrica primaria incompleta o reproducibilidad, y hay suficientes seeds para el
criterio configurado. Experimentos de una sola seed son exploratorios.

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
- no se dibujan lineas que conecten punto a punto los scatter reales.

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

## Scripts utiles de comparacion

```bash
python3 Comparison/scripts/verify_dataset_integrity.py --dry-run
python3 Comparison/scripts/validate_sample_bundle.py --help
python3 Comparison/scripts/check_geometry_leakage.py --help
python3 Comparison/scripts/evaluate_hamiltonian_metrics.py --help
python3 Comparison/scripts/evaluate_cross.py --help
python3 Comparison/scripts/analyze_winners.py --help
python3 Comparison/scripts/cleanup_generated_datasets.py --dry-run
```

## Tests y validacion local

```bash
python3 -m unittest tests/test_comparison_workflow.py
python3 -m unittest tests/test_analyze_winners_three_methods.py
python3 -m unittest tests/test_method_provenance_fairness.py
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
- La cache experimental global sigue desactivada hasta tener claves de hash
  completas para datasets, entrenamiento, prediccion y metricas.
- Los scripts standalone pueden omitir validaciones que la ruta `Comparison`
  aplica de forma estricta.
