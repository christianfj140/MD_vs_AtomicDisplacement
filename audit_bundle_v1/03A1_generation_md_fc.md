# Dossier 1A — Generación MD, FC y cartesiana genérica

## Objeto de revisión

Auditar cómo se construyen MD, FC y desplazamientos aleatorios; comprobar unidades, condiciones de contorno, independencia de splits, geometrías, validaciones y procedencia SIESTA.

## Condiciones del contexto

- Contexto ciego: no contiene informes de auditoría previos ni prompts históricos.
- Las líneas prefijadas `NNNNN |` conservan la numeración del archivo fuente.
- Los extractos Python omiten funciones operativas sin semántica científica directa.
- Ausencia de un archivo o función en este dossier significa `no evaluable`, no `correcto`.

## `README.md`

SHA-256: `bfe1981754d7546992bf6388c6a6a648a8dcf15e1a0dde7ac3d60bd3c244bb10`

```md
00001 | # MD_vs_AtomicDisplacement
00002 | 
00003 | Repositorio para comparar predicciones de Hamiltonianos de `graph2mat` sobre
00004 | agua usando datasets generados con SIESTA. El flujo actual ya no es solo
00005 | `MD vs AtomDisplacement`: la ruta de comparacion admite tres metodos
00006 | canonicos y permite ejecutar cualquier subconjunto de ellos.
00007 | 
00008 | ## Documentation map
00009 | 
00010 | - [Architecture](docs/architecture.md)
00011 | - [Workflows](docs/workflows.md)
00012 | - [Data and outputs](docs/data_and_outputs.md)
00013 | - [Development](docs/development.md)
00014 | - [Known limitations](docs/known_limitations.md)
00015 | - [Graph2Mat vs DeepH benchmark runbook](docs/graph2mat_deeph_benchmark.md)
00016 | - [ML vs SIESTA benchmark toolkit](docs/ml_vs_siesta_benchmark.md)
00017 | - [Phase 6 H2O Hamiltonian architecture benchmark](docs/phase6_hamiltonian_architecture_benchmark.md)
00018 | - [Derivative smoke validation note](docs/derivative_smoke_validation_note.md)
00019 | 
00020 | ## Current scope
00021 | 
00022 | This repository now covers more than the original `MD vs AtomDisplacement`
00023 | comparison:
00024 | 
00025 | - the main `Comparison` UI can run `md`, `siesta_fc_cartesian`, and
00026 |   `random_cartesian` experiments;
00027 | - the same UI also exposes `Graph2Mat vs DeepH`, `Graph2Mat sweep + DeepH`,
00028 |   `ML vs SIESTA`, and dataset-size-minimum analysis surfaces;
00029 | - the repository ships versioned material bundles for `h2o`, `graphene`,
00030 |   `graphene_5x2`, `graphene_5x5`, `si_amorphous`, and `si_vacancy`.
00031 | 
00032 | ## Quick start
00033 | 
00034 | 1. Create the local environment:
00035 | 
00036 |    ```bash
00037 |    ./scripts/create_graph2mat_venv.sh
00038 |    ```
00039 | 
00040 | 2. Activate it:
00041 | 
00042 |    ```bash
00043 |    source .venv/bin/activate
00044 |    ```
00045 | 
00046 | 3. Start the main comparison UI:
00047 | 
00048 |    ```bash
00049 |    python3 Comparison/scripts/pipeline_ui.py
00050 |    ```
00051 | 
00052 | 4. Open the printed URL, then use the `Experiment` tab to select methods,
00053 |    datasets, and run mode.
00054 | 
00055 | ## Metodos soportados
00056 | 
00057 | La fuente de verdad de identificadores esta en
00058 | `Comparison/scripts/method_registry.py`.
00059 | 
00060 | | ID canonico | Nombre UI | Dataset / idea | Resultados |
00061 | | --- | --- | --- | --- |
00062 | | `md` | MD | Trayectoria de dinamica molecular | `Comparison/results/results_md/` |
00063 | | `siesta_fc_cartesian` | SIESTA FC Cartesian | Desplazamientos cartesianos generados con `MD.TypeOfRun FC` | `Comparison/results/results_atomdisp/` |
00064 | | `random_cartesian` | Random Cartesian | Perturbaciones cartesianas aleatorias alrededor de la geometria relajada | `Comparison/results/results_random_cartesian/` |
00065 | 
00066 | Aliases legacy aceptados: `atom_displacement` y `atomdisp` se normalizan a
00067 | `siesta_fc_cartesian`.
00068 | 
00069 | ## Punto de entrada recomendado
00070 | 
00071 | La ruta recomendada para experimentos comparables es la UI de `Comparison`:
00072 | 
00073 | ```bash
00074 | python3 Comparison/scripts/pipeline_ui.py
00075 | ```
00076 | 
00077 | Abre `http://127.0.0.1:8770`.
00078 | 
00079 | Desde la pestaña `Experiment` se puede:
00080 | 
00081 | - seleccionar uno, dos o los tres metodos;
00082 | - elegir `dataset_only`, `full_strict_pipeline` o
00083 |   `train_test_metrics_plots_only`;
00084 | - editar recetas de datasets MD, FC Cartesian y Random Cartesian;
00085 | - fijar splits, test sets, metrica primaria, rendimiento y parametros de
00086 |   entrenamiento;
00087 | - guardar todo en manifests auditables dentro de `Comparison/results/<run_id>/`.
00088 | 
00089 | Si no se selecciona ningun metodo, la UI y el backend rechazan el experimento.
00090 | Los modos legacy siguen existiendo: cuando no llega `selected_methods`, el
00091 | backend usa el default historico `["md", "siesta_fc_cartesian"]`.
00092 | 
00093 | ## Graph2Mat vs DeepH para grafeno
00094 | 
00095 | El flujo dedicado `G2M vs DeepH` vive en una pestaña propia de la UI. Su guia
00096 | operativa y el checklist paper-ready estan en
00097 | `docs/graph2mat_deeph_benchmark.md`.
00098 | 
00099 | La misma UI expone tambien:
00100 | 
00101 | - `DeepH comparison`: ejecuta el flujo justo de comparacion Graph2Mat vs DeepH;
00102 | - `Graph2Mat sweep + DeepH comparison`: combina barrido Graph2Mat y comparacion
00103 |   posterior contra DeepH;
00104 | - `ML vs SIESTA`: toolkit ligero para preparar entradas, validaciones y
00105 |   payloads de UI sin lanzar SIESTA ni entrenamientos pesados;
00106 | - `dataset size minimum`: analisis postproceso sobre barridos archivados para
00107 |   estimar el tamano minimo de dataset segun metrica, presupuesto y criterio de
00108 |   claim.
00109 | 
00110 | Los datasets reutilizables de este flujo viven por defecto en
00111 | `Comparison/datasets/`, separados de `Comparison/workspaces/` y
00112 | `Comparison/results/`. La ruta preseleccionada para grafeno es
00113 | `Comparison/datasets/graphene_w90_joint`.
00114 | 
00115 | La idea central es evitar el fallo historico en el que un dataset valido para
00116 | Graph2Mat no contenia `HSX`, `STRUCT_OUT` ni `ORB_INDX`, obligando a reruns
00117 | SIESTA por snapshot para DeepH. Los datasets joint deben generarse en una sola
00118 | pasada SIESTA y archivar, como minimo, `RUN.fdf`, `RUN.out` o `siesta.out`,
00119 | `SystemLabel.TSHS`, `SystemLabel.TSDE`, `SystemLabel.HSX`,
00120 | `SystemLabel.STRUCT_OUT`, `SystemLabel.XV`, `SystemLabel.ORB_INDX` y
00121 | `metadata.json`.
00122 | 
00123 | El modo normal valida y falla si faltan artefactos; no hay reparacion SIESTA
00124 | silenciosa. Cualquier reparacion debe ser explicita, lenta/costosa y quedar
00125 | marcada en los manifests. `ML_prediction.HSX` nunca es ground truth y las
00126 | metricas espectrales usan `S_ref`/`S_ref(k)` cuando esta disponible. Si la
00127 | equivalencia DeepH/Graph2Mat no esta probada en base, unidades, orden orbital,
00128 | convencion R-vector o frame, la comparacion se marca `diagnostic_only`.
00129 | Las metricas del paper DeepH solo son contexto externo: no son un baseline
00130 | directo para claims robustos en este pipeline.
00131 | 
00132 | ## Estructura actual del repositorio
00133 | 
00134 | ```text
00135 | MD_vs_AtomicDisplacement/
00136 | ├── MD/
00137 | │   ├── dataset/                  # inputs, pseudopotenciales y salidas MD generadas
00138 | │   ├── scripts/                  # pipeline standalone MD
00139 | │   ├── ui/                       # UI standalone de depuracion MD
00140 | │   └── pipeline_config.yaml
00141 | ├── AtomDisplacement/
00142 | │   ├── base/                     # RUN.fdf base y pseudopotenciales
00143 | │   ├── relaxed/                  # geometria relajada y basis .ion.xml
00144 | │   ├── dataset/                  # FC_steps, RandomCartesian_steps, splits y collected
00145 | │   ├── scripts/                  # FC Cartesian, Random Cartesian, single-points y Graph2Mat
00146 | │   ├── ui/                       # UI standalone de depuracion AtomDisplacement
00147 | │   └── pipeline_config.yaml
00148 | ├── Comparison/
00149 | │   ├── config/                   # settings compartidas de comparacion
00150 | │   ├── dataset_recipes/          # recetas versionadas para experimentos
00151 | │   ├── results/                  # resultados archivados y manifests cientificos
00152 | │   ├── scripts/                  # UI, evaluacion cruzada, metricas y analisis de winners
00153 | │   ├── ui/                       # frontend de la UI principal
00154 | │   ├── workspaces/               # workspaces temporales por experimento
00155 | │   ├── METRICS.md
00156 | │   └── PERFORMANCE.md
00157 | ├── configs/                      # configs Graph2Mat auxiliares
00158 | ├── scripts/                      # utilidades de entorno y compatibilidad Torch
00159 | ├── shared/                       # helpers compartidos SIESTA
00160 | ├── tests/
00161 | ├── requirements-graph2mat.txt
00162 | └── README.md
00163 | ```
00164 | 
00165 | ## Dependencias
00166 | 
00167 | Se espera un entorno con:
00168 | 
00169 | - SIESTA disponible como `siesta`;
00170 | - `graph2mat`;
00171 | - Python 3.12 o compatible con el entorno local;
00172 | - un virtualenv local en `.venv`.
00173 | 
00174 | Crear el entorno portable:
00175 | 
00176 | ```bash
00177 | ./scripts/create_graph2mat_venv.sh
00178 | ```
00179 | 
00180 | La UI de comparacion activa por defecto:
00181 | 
00182 | ```bash
00183 | source ${REPO_ROOT}/.venv/bin/activate
00184 | ```
00185 | 
00186 | Si `graph2mat` no puede instalarse desde `requirements-graph2mat.txt`, instala
00187 | tu copia local:
00188 | 
00189 | ```bash
00190 | source .venv/bin/activate
00191 | python -m pip install -e /ruta/a/graph2mat
00192 | ```
00193 | 
00194 | ## Materiales y presets
00195 | 
00196 | El flujo historico sigue usando H2O, pero ahora esta declarado como preset de
00197 | material en `materials/h2o/material.yaml`. Los configs principales lo seleccionan
00198 | explicitamente con:
00199 | 
00200 | ```yaml
00201 | material:
00202 |   preset: h2o
00203 | ```
00204 | 
00205 | Ese preset apunta al `RUN.fdf` base, pseudopotenciales y basis ya versionados
00206 | para H2O. La validacion de bundles vive en `shared/material_bundle.py` y la capa
00207 | de presets/fallback legacy en `shared/material_presets.py`. Por compatibilidad,
00208 | un config antiguo sin seccion `material` puede resolverse al preset `h2o` con una
00209 | advertencia de migracion; los nuevos materiales deberan declarar su propio bundle
00210 | antes de conectarse al pipeline en fases posteriores.
00211 | 
00212 | Un bundle explicito usa las mismas claves que valida el backend:
00213 | 
00214 | ```yaml
00215 | material:
00216 |   label: sic
00217 |   fdf: materials/sic/RUN.fdf
00218 |   pseudopotential_dir: materials/sic/pseudos
00219 |   basis_dir: materials/sic/basis
00220 |   structure_type: crystal
00221 | ```
00222 | 
00223 | En la UI, la pestaña `Experiment` incluye `Material bundle`: puedes elegir el
00224 | preset `h2o` o introducir las rutas de un bundle, pulsar `Validate material` y
00225 | ver especies, cobertura de pseudopotenciales, basis y warnings antes de lanzar
00226 | el experimento. Si seleccionas un bundle custom invalido, la UI/API no vuelve a
00227 | H2O de forma silenciosa; el inicio del experimento falla con el error de
00228 | validacion del backend.
00229 | 
00230 | Presets versionados actualmente en `materials/`:
00231 | 
00232 | - `h2o`
00233 | - `graphene`
00234 | - `graphene_5x2`
00235 | - `graphene_5x5`
00236 | - `si_amorphous`
00237 | - `si_vacancy`
00238 | 
00239 | Tambien existe una primera receta material-agnostica para AtomicDisplacement:
00240 | `AtomDisplacement/scripts/generate_generic_cartesian_displacement_dataset.py`.
00241 | Lee el bundle validado y genera estructuras `+/-x`, `+/-y`, `+/-z` para cada
00242 | atomo seleccionado, usando `atomic_displacement.recipe: generic_cartesian`. Las
00243 | recetas historicas de enlaces/angulos siguen siendo especificas de H2O y se
00244 | mantienen separadas.
00245 | 
00246 | ## Flujo cientifico de `Comparison`
00247 | 
00248 | `Comparison/scripts/pipeline_ui.py` orquesta el experimento completo:
00249 | 
00250 | 1. crea workspaces aislados por metodo y dataset;
00251 | 2. genera o prepara datasets segun recetas;
00252 | 3. valida muestras SIESTA antes de usarlas;
00253 | 4. entrena/testea/predice en modo `full_strict_pipeline` o
00254 |    `train_test_metrics_plots_only`;
00255 | 5. archiva estructuras, Hamiltonianos predichos, referencias SIESTA, configs,
00256 |    logs y manifests;
00257 | 6. construye tests congelados;
00258 | 7. ejecuta evaluacion cruzada metodo/test set;
00259 | 8. calcula metricas sparse, espectrales, DOS y relacion matriz-espectro;
00260 | 9. agrega resultados y escribe `recommendation.json`.
00261 | 
00262 | En `dataset_only` se generan y validan datasets, pero se omiten entrenamiento,
00263 | prediccion, evaluacion cruzada y analisis de winners. En
00264 | `train_test_metrics_plots_only` se reutiliza un dataset ya archivado con la
00265 | misma seleccion; si faltan sus carpetas, splits o referencias SIESTA, el
00266 | experimento falla antes de entrenar. En ese modo la UI muestra una tabla de
00267 | datasets archivados reutilizables; puedes marcar explicitamente los datasets
00268 | que quieras entrenar de nuevo. Si no marcas ninguno, el backend usa la
00269 | coincidencia automatica por metodo, tamano, etiqueta/receta.
00270 | 
00271 | Por defecto ese modo respeta los splits archivados. Si quieres mantener el
00272 | dataset fijo pero cambiar train/validation/test, selecciona `Rebuild splits
00273 | from controls` en `Split source`; la pipeline copia el dataset al workspace del
00274 | nuevo run y reconstruye los splits con los ratios y el `Split mode` elegidos,
00275 | sin regenerar SIESTA ni sobrescribir el dataset original.
00276 | 
00277 | Para MD, el `Split mode` por defecto es `blocked_with_gap`: separa bloques
00278 | contiguos de train, validation y test y deja un frame temporal fuera entre
00279 | particiones. El modo `spread` sigue disponible para exploracion/debug, pero se
00280 | marca con warning porque puede colocar frames MD temporalmente adyacentes en
00281 | particiones distintas.
00282 | 
00283 | Los YAML de entrenamiento generados pasan la validacion a Graph2Mat de forma
00284 | explicita: MD usa `training.data.val_runs` y AtomDisplacement/Random Cartesian
00285 | usan `runs.json` con la clave `val` o `val_runs` cuando el split copiado lo
00286 | permite. El `checkpoint_manifest.json` guarda la fuente de validacion,
00287 | `val_loss` como criterio de seleccion de `best-*.ckpt`, y si la seleccion queda
00288 | respaldada por un split de validacion.
00289 | 
00290 | Cada entrenamiento con metricas queda etiquetado en su `manifest.json` con
00291 | `training_tag`, `training_index` y el dataset base usado para entrenar. Por
00292 | ejemplo, varios entrenamientos sobre el mismo dataset aparecen como
00293 | `dataset_1000_train1`, `dataset_1000_train2`, etc.; la UI usa ese tag en plots
00294 | y tablas para distinguir reentrenamientos con hiperparametros o splits
00295 | distintos.
00296 | 
00297 | Para lanzar varios entrenamientos sobre datasets reutilizables, usa
00298 | `Train/test/metrics/plots only`, selecciona los datasets en `Reusable archived
00299 | datasets`, ajusta los campos de `Training parameters` y pulsa `Add current
00300 | config` en `Training plan`. Puedes repetirlo con otros hiperparametros y otra
00301 | seleccion de datasets. Al ejecutar el experimento, las configuraciones del plan
00302 | se procesan secuencialmente; cada entrada del plan crea un run independiente por
00303 | dataset seleccionado.
00304 | 
00305 | Un experimento con un solo metodo es valido para generar datos y diagnosticos,
00306 | pero queda marcado como `non_comparative` porque no puede producir winner
00307 | robusto.
00308 | 
00309 | ## Semantica oficial H-only y seguridad de `ML_prediction.HSX`
00310 | 
00311 | El benchmark oficial de Hamiltonianos no entrena ni evalua el overlap S como si
00312 | fuera una segunda componente fisica del Hamiltoniano. Para las rutas actuales de
00313 | Graph2Mat, toda config oficial debe declarar:
00314 | 
00315 | ```yaml
00316 | data:
00317 |   out_matrix: hamiltonian
00318 |   matrix_component_policy: h_only
00319 |   n_matrix_components: 1
00320 |   symmetric_matrix: true
00321 | ```
00322 | 
00323 | En SIESTA no ortogonal, los contenedores raw pueden exponer `(H, S)`. En este
00324 | repo, `matrix_component_policy: h_only` significa que la componente 0 es el
00325 | target Hamiltoniano H y que S no entra como canal de perdida Hamiltoniana. Las
00326 | configs generadas, los manifests y la evaluacion registran
00327 | `target_component_policy`, `n_matrix_components`, `reference_component_count` y
00328 | `prediction_component_count` para que no se mezclen resultados ambiguos con los
00329 | runs corregidos.
00330 | 
00331 | `ML_prediction.HSX` tampoco debe interpretarse automaticamente como un archivo
00332 | SIESTA standalone fisicamente seguro. Graph2Mat puede escribir contenedores con
00333 | componentes auxiliares/spin-like o un overlap propio que no coincide con el
00334 | overlap de referencia. Las metricas espectrales y DOS oficiales resuelven el
00335 | problema generalizado con el overlap SIESTA de referencia, `S_ref`, salvo que el
00336 | overlap predicho haya sido validado explicitamente. Los manifests registran:
00337 | 
00338 | - `overlap_source`
00339 | - `prediction_own_overlap_used` o
00340 |   `prediction_own_overlap_used_for_spectra`
00341 | - `prediction_overlap_relative_frobenius_vs_reference`
00342 | - `prediction_self_contained_hsx_safe`
00343 | - `prediction_artifact_semantics`
00344 | - `graph2mat_auxiliary_component_ignored`
00345 | 
00346 | Si `prediction_self_contained_hsx_safe=false`, usa las metricas del evaluador
00347 | como resultado oficial y no abras el `ML_prediction.HSX` predicho con su propio
00348 | overlap para extraer bandas o DOS cientificas.
00349 | 
00350 | ## Artefactos de trazabilidad
00351 | 
00352 | Los artefactos importantes quedan bajo `Comparison/results/`:
00353 | 
00354 | - `Comparison/results/<run_id>/experiment_manifest.yaml`
00355 | - `Comparison/results/<run_id>/performance_report.json`
00356 | - `Comparison/results/<run_id>/summary/recommendation.json`
00357 | - `Comparison/results/<run_id>/summary/cross_evaluation_metrics.csv`
00358 | - `Comparison/results/<run_id>/common_tests/*/frozen_test_manifest.json`
00359 | - `Comparison/results/results_<method>/<dataset_label>/run_<run_id>/manifest.json`
00360 | - `metrics/sparse_metrics.csv`
00361 | - `metrics/spectral_metrics.csv`
00362 | - `metrics/dos_metrics.csv`
00363 | - `metrics/matrix_spectrum_relationship.csv`
00364 | - `metrics/orbital_pair_metrics.csv`
00365 | - `metrics/orbital_pair_summary.csv`
00366 | 
00367 | La recomendacion final solo debe tratarse como robusta cuando el manifest no
00368 | contiene warnings severos de leakage, settings, checkpoint, presupuesto,
00369 | metrica primaria incompleta o reproducibilidad, y hay suficientes seeds para el
00370 | criterio configurado. Tambien bloquean o degradan la conclusion los warnings de
00371 | target semantics, prediction HSX safety, overlap source desconocido, Fermi level
00372 | ausente para una metrica primaria near-Fermi, incompatibilidad material/SIESTA o
00373 | checkpoint mismatch. Experimentos de una sola seed son exploratorios.
00374 | 
00375 | Los CSV `orbital_pair_*` son diagnosticos para comparar mapas orbital-orbital
00376 | tipo DeepH: usa `mae_union_meV` por `species_pair`, `row_orbital_index` y
00377 | `col_orbital_index`. No son metricas H' locales exactas ni cambian los winners
00378 | por defecto.
00379 | 
00380 | ## Datasets y recetas
00381 | 
00382 | La UI acepta recetas de datasets versionadas en JSON. Hay ejemplos en
00383 | `Comparison/dataset_recipes/`.
00384 | 
00385 | Ejemplo minimo:
00386 | 
00387 | ```json
00388 | {
00389 |   "md": [
00390 |     {
00391 |       "recipe_id": "md_100",
00392 |       "blocks": [
00393 |         {"block_id": "md_plain", "n_snapshots": 100}
00394 |       ]
00395 |     }
00396 |   ],
00397 |   "siesta_fc_cartesian": [
00398 |     {
00399 |       "recipe_id": "fc_mixed",
00400 |       "blocks": [
00401 |         {"block_id": "fc_0p02", "displacement": "0.02 Ang", "n_structures": 20},
00402 |         {"block_id": "fc_0p05", "displacement": "0.05 Ang", "n_structures": 20}
00403 |       ]
00404 |     }
00405 |   ],
00406 |   "random_cartesian": [
00407 |     {
00408 |       "recipe_id": "rc_sigma_0p03",
00409 |       "blocks": [
00410 |         {
00411 |           "block_id": "rc_100",
00412 |           "n_structures": 100,
00413 |           "distribution": "gaussian",
00414 |           "sigma_ang": 0.03,
00415 |           "seed": 7
00416 |         }
00417 |       ]
00418 |     }
00419 |   ]
00420 | }
00421 | ```
00422 | 
00423 | El tamaño del dataset ya no es la identidad cientifica completa: los manifests
00424 | propagan `recipe_id`, `block_id`, parametros de generacion, seed y hash de
00425 | receta.
00426 | 
00427 | `random_cartesian` mantiene compatibilidad con los bloques legacy
00428 | (`distribution`, `sigma_ang`, `uniform_range_ang`, `move_atoms`, `seed`, etc.),
00429 | pero tambien acepta componentes composables por bloque:
00430 | 
00431 | ```json
00432 | {
00433 |   "block_id": "rc_local_h2o",
00434 |   "n_structures": 100,
00435 |   "components": {
00436 |     "atom_displacement": {"enabled": true, "sigma_ang": 0.03},
00437 |     "bond_displacement": {
00438 |       "enabled": true,
00439 |       "distribution": "uniform",
00440 |       "min_delta_ang": -0.02,
00441 |       "max_delta_ang": 0.02,
00442 |       "min_bond_ang": 0.70,
00443 |       "max_bond_ang": 1.30
00444 |     },
00445 |     "angle_displacement": {
00446 |       "enabled": true,
00447 |       "distribution": "gaussian",
00448 |       "sigma_deg": 3.0,
00449 |       "min_angle_deg": 80.0,
00450 |       "max_angle_deg": 130.0
00451 |     }
00452 |   },
00453 |   "validation": {
00454 |     "min_distance_ang": 0.65,
00455 |     "max_rmsd_from_reference_ang": null,
00456 |     "max_attempts_per_structure": 100
00457 |   }
00458 | }
00459 | ```
00460 | 
00461 | Los componentes de enlace y angulo son, por ahora, explicitos para H2O
00462 | (`h2o_oh` y `h2o_hoh`). El metodo es una perturbacion local restringida no-MD;
00463 | no representa un ensamble termodinamico.
00464 | La pestaña `Experiment` expone estos componentes dentro de cada fila/bloque de
00465 | cada tarjeta de dataset Random Cartesian. Un dataset puede sumar varios bloques
00466 | de estructuras, y cada bloque puede activar una combinacion distinta y tener
00467 | amplitudes, rangos, limites geometricos y validacion propios.
00468 | 
00469 | Para materiales arbitrarios, Random Cartesian tambien acepta la receta explicita
00470 | `generic_cartesian_noise`, que lee el bundle `material`, perturba coordenadas
00471 | atomicas con una semilla determinista y escribe grupos `split_group_id` para no
00472 | separar variantes correlacionadas entre train/validation/test:
00473 | 
00474 | ```yaml
00475 | random_cartesian:
00476 |   recipe: generic_cartesian_noise
00477 |   n_structures: 100
00478 |   max_displacement_ang: 0.05
00479 |   selected_species: null
00480 |   min_interatomic_distance_ang: 0.6
00481 |   remove_center_of_mass_translation: true
00482 |   seed: 12345
00483 |   variants_per_family: 1
00484 | ```
00485 | 
00486 | ## UI de resultados
00487 | 
00488 | La pestaña `Results` muestra tablas y plots Plotly. Los plots de dispersion
00489 | mantienen los puntos reales y añaden lineas de ajuste por serie:
00490 | 
00491 | - los puntos no se eliminan;
00492 | - los valores se ordenan por eje X;
00493 | - X duplicados se agregan por media antes de ajustar;
00494 | - NaN y datos incompletos se ignoran;
00495 | - el ajuste lineal se muestra por defecto;
00496 | - el menu dentro de cada plot permite cambiar a ajuste cuadratico o ocultar el
00497 |   ajuste;
00498 | - no se dibujan lineas que conecten punto a punto los scatter reales;
00499 | - cuando existe procedencia de material, las etiquetas/hover de los plots
00500 |   muestran el material y el selector `Material` permite filtrar por `All
00501 |   materials`, `h2o`, `sic` u otros labels archivados.
00502 | 
00503 | Si se muestran varios grupos de compatibilidad de material, la UI marca esos
00504 | plots como diagnosticos: no deben interpretarse como un benchmark agrupado. Las
00505 | comparaciones robustas requieren hashes compatibles de material, basis,
00506 | pseudopotenciales y ajustes SIESTA; los runs antiguos sin procedencia aparecen
00507 | como `unknown material`.
00508 | 
00509 | El selector `Safety` de la UI no borra resultados archivados. Por defecto se
00510 | muestran todos los estados, incluidos `unsafe`, `unknown`, `exploratory` y
00511 | `non-comparative`, para auditoria. Si filtras a un subconjunto, el estado del
00512 | filtro queda visible en la linea de estado de plots.
00513 | 
00514 | Debajo de los plots hay una seccion destructiva para datasets generados. Permite
00515 | listar artefactos, seleccionar uno o varios y borrar solo esos, o borrar todos
00516 | los generados. El backend exige IDs concretos o `all=true`, valida rutas y
00517 | rechaza enlaces simbolicos antes de borrar.
00518 | 
00519 | ## Pipelines standalone
00520 | 
00521 | Los pipelines standalone siguen siendo utiles para depuracion, pero no son la
00522 | ruta recomendada para conclusiones cientificas comparativas.
00523 | 
00524 | MD:
00525 | 
00526 | ```bash
00527 | python3 MD/scripts/main_md.py
00528 | ```
00529 | 
00530 | AtomDisplacement FC Cartesian:
00531 | 
00532 | ```bash
00533 | python3 AtomDisplacement/scripts/main_atom_displacement.py
00534 | python3 AtomDisplacement/scripts/main_atdisp.py
00535 | ```
00536 | 
00537 | Random Cartesian se genera desde la ruta de `Comparison` o directamente con:
00538 | 
00539 | ```bash
00540 | python3 AtomDisplacement/scripts/generate_random_cartesian_dataset.py
00541 | ```
00542 | 
00543 | Los single-points de AtomDisplacement/Random Cartesian validan matrices SIESTA
00544 | de forma estricta por defecto: una `.TSHS`/`.HSX` solo se reutiliza si el
00545 | `RUN.out` correspondiente demuestra completion, convergencia SCF y no es stale
00546 | respecto al `RUN.fdf` cuando el repositorio puede comprobarlo. La opcion
00547 | `--allow-unvalidated-matrices` queda reservada para depuracion local y marca los
00548 | resumenes con `UNSAFE_UNVALIDATED_MATRIX_REFERENCE`.
00549 | 
00550 | ## Scripts utiles de comparacion
00551 | 
00552 | ```bash
00553 | python3 Comparison/scripts/material_agnostic_smoke.py --case both
00554 | python3 Comparison/scripts/g2m_deeph_smoke.py --dry-run
00555 | python3 Comparison/scripts/g2m_deeph_final_workflow.py --help
00556 | python3 Comparison/scripts/ml_vs_siesta_benchmark.py --help
00557 | python3 Comparison/scripts/g2m_deeph_dataset_size_minimum.py --help
00558 | python3 Comparison/scripts/verify_dataset_integrity.py --dry-run
00559 | python3 Comparison/scripts/validate_sample_bundle.py --help
00560 | python3 Comparison/scripts/check_geometry_leakage.py --help
00561 | python3 Comparison/scripts/evaluate_hamiltonian_metrics.py --help
00562 | python3 Comparison/scripts/evaluate_cross.py --help
00563 | python3 Comparison/scripts/analyze_winners.py --help
00564 | python3 Comparison/scripts/cleanup_generated_datasets.py --dry-run
00565 | ```
00566 | 
00567 | Re-evaluacion post-H-only/S_ref de un run archivado:
00568 | 
00569 | ```bash
00570 | python3 Comparison/scripts/evaluate_hamiltonian_metrics.py <result_dir> --overwrite
00571 | ```
00572 | 
00573 | Para grafeno u otros runs periodicos con malla Monkhorst-Pack no-gamma, la ruta
00574 | k-point-aware es opt-in:
00575 | 
00576 | ```bash
00577 | python3 Comparison/scripts/evaluate_hamiltonian_metrics.py <result_dir> \
00578 |   --enable-kpoint-metrics \
00579 |   --overwrite
00580 | ```
00581 | 
00582 | Usa `--overwrite` solo cuando quieras reemplazar metricas antiguas de ese
00583 | `result_dir` con metricas regeneradas bajo la semantica H-only/S_ref actual.
00584 | 
00585 | El cleanup puede escribir `Comparison/generated_dataset_cleanup_manifest.json`
00586 | como log local generado. Ese archivo no es fuente de verdad portable y queda
00587 | ignorado por git.
00588 | 
00589 | ## Tests y validacion local
00590 | 
00591 | ```bash
00592 | python3 -m unittest tests/test_comparison_workflow.py
00593 | python3 -m unittest tests/test_analyze_winners_three_methods.py
00594 | python3 -m unittest tests/test_method_provenance_fairness.py
00595 | python3 -m unittest tests/test_material_agnostic_smoke.py
00596 | python3 -m unittest tests/test_three_method_scientific_smoke.py
00597 | python3 -m unittest tests/test_metrics_material_compatibility.py
00598 | python3 -m unittest tests/test_material_ui_api.py
00599 | python3 -m unittest tests/test_g2m_deeph_docs.py
00600 | python3 -m unittest tests/test_g2m_deeph_documentation.py
00601 | ```
00602 | 
00603 | Chequeos rapidos de la UI:
00604 | 
00605 | ```bash
00606 | python3 -m py_compile Comparison/scripts/pipeline_ui.py Comparison/scripts/evaluate_hamiltonian_metrics.py Comparison/scripts/cleanup_generated_datasets.py
00607 | node --check Comparison/ui/app.js
00608 | ```
00609 | 
00610 | ## Documentacion relacionada
00611 | 
00612 | - `Comparison/METRICS.md`: definicion de metricas sparse, espectrales y DOS.
00613 | - `Comparison/PERFORMANCE.md`: controles de rendimiento disponibles en la UI.
00614 | 
00615 | ## Limitaciones actuales
00616 | 
00617 | - La comparacion robusta requiere seeds suficientes; una sola seed queda como
00618 |   diagnostico exploratorio.
00619 | - Las metricas dependientes de Fermi solo son autoritativas si SIESTA proporciona
00620 |   un Fermi level real.
00621 | - `ML_prediction.HSX` puede no ser un contenedor Hamiltoniano+overlap autonomo:
00622 |   para referencias no ortogonales las metricas espectrales oficiales usan
00623 |   `S_ref` y el manifest registra `prediction_artifact_semantics` para indicar si
00624 |   el HSX predicho es seguro o no como problema generalizado standalone.
00625 | - Las metricas antiguas sin `metrics_schema_version=h_only_sref_v2` ni
00626 |   `metrics_provenance_generation=post_h_only_sref_prediction_safety` son
00627 |   legado pre/post-fix desconocido: no deben mezclarse con resultados H-only
00628 |   actuales hasta re-evaluarlas explicitamente con
00629 |   `Comparison/scripts/evaluate_hamiltonian_metrics.py --overwrite`.
00630 | - Las metricas comparables con DeepH son analogos del repositorio sobre la base
00631 |   Hamiltoniana archivada. No reproducen todavia H' local, k-path bands,
00632 |   SOC/complejos, optica/Berry/shift-current, incertidumbre de ensembles ni
00633 |   escalado DeepH-vs-DFT por tamano de sistema; ver `Comparison/METRICS.md`.
00634 | - La cache experimental global sigue desactivada hasta tener claves de hash
00635 |   completas para datasets, entrenamiento, prediccion y metricas.
00636 | - Los scripts standalone pueden omitir validaciones que la ruta `Comparison`
00637 |   aplica de forma estricta.
```

## `MD/pipeline_config.yaml`

SHA-256: `668b7c5adf47e1e10480c5cd8d1ac68c93e9dd130e20b6a96eefa470c55ec240`

```yaml
00001 | paths:
00002 |   dataset_dir: ${REPO_ROOT}/MD/dataset
00003 |   training_dir: ${REPO_ROOT}/MD/training
00004 |   run_fdf_name: RUN.fdf
00005 |   run_out_name: RUN.out
00006 |   training_config_name: config.yaml
00007 |   venv_activate: ${REPO_ROOT}/.venv/bin/activate
00008 | material:
00009 |   preset: graphene
00010 | commands:
00011 |   graph2mat: graph2mat
00012 |   siesta: siesta
00013 |   shell: bash
00014 | md:
00015 |   system_name: Graphene MD dataset
00016 |   system_label: siesta
00017 |   run_fdf_template: ${REPO_ROOT}/materials/graphene/RUN.fdf
00018 |   type_of_run: verlet
00019 |   # 70 = 6+1+1 split frames + 2*30 temporal-gap frames (see splits below).
00020 |   steps: 70
00021 |   temperature_K: 300.0
00022 |   timestep_fs: 1.0
00023 |   ensemble: nve
00024 |   thermostat: none
00025 |   write_md_history: true
00026 |   temperature_blocks: []
00027 |   basis_size: SZP
00028 |   basis_type: split
00029 |   energy_shift: 0.275 eV
00030 |   mesh_cutoff: 250 Ry
00031 |   xc_functional: LDA
00032 |   xc_authors: PZ
00033 |   max_scf_iterations: 200
00034 |   solution_method: diagon
00035 |   dm_mixing_weight: 0.02
00036 |   dm_number_pulay: 3
00037 |   dm_tolerance: 1.d-5
00038 |   dm_require_energy_convergence: T
00039 |   dm_energy_tolerance: 1.e-5 eV
00040 |   spin_polarized: F
00041 |   fix_spin: F
00042 |   non_collinear_spin: F
00043 |   xml_write: true
00044 |   save_hs_file: true
00045 |   save_hs: true
00046 |   save_de: true
00047 |   lua_script: md_store.lua
00048 |   force_aux_cell: true
00049 |   lattice_constant:
00050 |     value: 1.0
00051 |     unit: Ang
00052 |   lattice_vectors:
00053 |   - - 6.5025
00054 |     - 3.75422012541
00055 |     - 0.0
00056 |   - - 2.1675
00057 |     - -1.25140670847
00058 |     - 0.0
00059 |   - - 0.0
00060 |     - 0.0
00061 |     - 14.45
00062 |   species:
00063 |   - index: 1
00064 |     atomic_number: 6
00065 |     symbol: C
00066 |   coordinates_format: Ang
00067 |   kgrid_monkhorst_pack:
00068 |   - - 1
00069 |     - 0
00070 |     - 0
00071 |     - 0.0
00072 |   - - 0
00073 |     - 1
00074 |     - 0
00075 |     - 0.0
00076 |   - - 0
00077 |     - 0
00078 |     - 1
00079 |     - 0.0
00080 |   atoms:
00081 |   - label: C
00082 |     species_index: 1
00083 |     position:
00084 |     - 0.0
00085 |     - 0.0
00086 |     - 0.0
00087 |   - label: C
00088 |     species_index: 1
00089 |     position:
00090 |     - 1.445
00091 |     - 0.0
00092 |     - 0.0
00093 |   - label: C
00094 |     species_index: 1
00095 |     position:
00096 |     - 2.1675
00097 |     - 1.25141
00098 |     - 0.0
00099 |   - label: C
00100 |     species_index: 1
00101 |     position:
00102 |     - 3.6125
00103 |     - 1.25141
00104 |     - 0.0
00105 |   - label: C
00106 |     species_index: 1
00107 |     position:
00108 |     - 4.335
00109 |     - 2.50281
00110 |     - 0.0
00111 |   - label: C
00112 |     species_index: 1
00113 |     position:
00114 |     - 5.78
00115 |     - 2.50281
00116 |     - 0.0
00117 | training:
00118 |   torch_float32_matmul_precision: null
00119 |   data:
00120 |     out_matrix: hamiltonian
00121 |     symmetric_matrix: true
00122 |     sub_point_matrix: false
00123 |     matrix_component_policy: h_only
00124 |     n_matrix_components: 1
00125 |     basis_files: ../dataset/MD_steps/basis/*.ion.xml
00126 |     train_runs: ../dataset/splits/train/*/RUN.fdf
00127 |     val_runs: ../dataset/splits/validation/*/RUN.fdf
00128 |     batch_size: 10
00129 |     store_in_memory: true
00130 |   model:
00131 |     num_interactions: 1
00132 |     correlation: 1
00133 |     max_ell: 2
00134 |     hidden_irreps: 10x0e + 10x1o + 10x2e
00135 |     loss: graph2mat.metrics.block_type_mae
00136 |     optim_lr: 0.005
00137 |   trainer:
00138 |     accelerator: cpu
00139 |     logger:
00140 |       class_path: TensorBoardLogger
00141 |       init_args:
00142 |         name: my_first_model
00143 |         save_dir: lightning_logs
00144 |     max_epochs: 200
00145 | checkpoint:
00146 |   path: null
00147 |   auto_best: true
00148 |   search_glob: lightning_logs/**/checkpoints/best-*.ckpt
00149 |   selection: latest_version
00150 | testing:
00151 |   test_runs: ../dataset/splits/test/*/RUN.fdf
00152 |   callbacks:
00153 |     plot_matrix_error: false
00154 |     show_plot: false
00155 |     samplewise_metrics_logger: false
00156 | prediction:
00157 |   predict_structs: ../dataset/splits/test/*/RUN.fdf
00158 |   output_file: ML_prediction.HSX
00159 |   callbacks:
00160 |     matrix_writer: true
00161 | pipeline:
00162 |   skip_model_test: false
00163 |   steps:
00164 |   - generate_md_dataset
00165 |   - run_md_training
00166 |   - run_md_testing
00167 |   - run_md_prediction
00168 | performance:
00169 |   max_parallel_siesta_jobs: 1
00170 |   max_parallel_dataset_jobs: 1
00171 |   max_parallel_prediction_jobs: 1
00172 |   max_parallel_evaluation_jobs: 1
00173 |   max_parallel_metric_jobs: 1
00174 |   omp_num_threads: null
00175 |   mkl_num_threads: null
00176 |   openblas_num_threads: null
00177 |   numexpr_num_threads: null
00178 |   torch_num_threads: null
00179 |   compute_accelerator: cpu
00180 |   batch_size: null
00181 |   store_in_memory: null
00182 |   reuse_validated_siesta_outputs: true
00183 |   enable_experiment_cache: false
00184 |   error_policy: fail_fast
00185 |   preset: null
00186 |   torch_float32_matmul_precision: null
00187 | splits:
00188 |   enabled: true
00189 |   # Leakage-safer scientific default for MD trajectories: consecutive frames
00190 |   # are 1 fs apart, and carbon vibrational periods are ~20-40 fs, so a gap of
00191 |   # 30 frames (~30 fs, about one vibrational period) is the minimum for the
00192 |   # buffered blocks to be meaningfully decorrelated. A gap of 1 frame (1 fs)
00193 |   # is physically meaningless. md.steps must cover
00194 |   # train+validation+test + 2*temporal_gap frames.
00195 |   strategy: blocked_with_gap
00196 |   temporal_gap: 30
00197 |   block_order: train,validation,test
00198 |   train: 6
00199 |   validation: 1
00200 |   test: 1
```

## `AtomDisplacement/pipeline_config.yaml`

SHA-256: `7904a92df49cf33a117252e6371c051b404b2c8924c1a759c956551f7ae5abed`

```yaml
00001 | paths:
00002 |   base_dir: base
00003 |   relaxed_dir: relaxed
00004 |   dataset_dir: dataset
00005 |   samples_dir: dataset/samples
00006 |   collected_dir: dataset/collected
00007 |   training_dir: training
00008 |   run_fdf_name: RUN.fdf
00009 |   run_out_name: RUN.out
00010 |   training_config_name: config.yaml
00011 |   runs_json_name: runs.json
00012 |   samples_manifest_name: samples_manifest.json
00013 |   run_summary_name: run_summary.json
00014 |   collected_json_name: water_atom_displacement_dataset.json
00015 |   collected_csv_name: water_atom_displacement_summary.csv
00016 |   venv_activate: ${REPO_ROOT}/.venv/bin/activate
00017 | material:
00018 |   preset: h2o
00019 | atomic_displacement:
00020 |   recipe: generic_cartesian
00021 |   amplitude_ang: 0.03
00022 |   selected_species: null
00023 |   include_base: true
00024 |   overwrite: false
00025 | commands:
00026 |   graph2mat: graph2mat
00027 |   siesta: siesta
00028 |   shell: bash
00029 |   python: python
00030 | generation:
00031 |   sample_id_format: sample_{index:04d}
00032 | structure:
00033 |   molecule_name: H2O
00034 |   relaxation:
00035 |     system_name: H2O molecule relaxation
00036 |     system_label: h2o_relax
00037 |   single_point:
00038 |     system_name_template: H2O {sample_id}
00039 |     title: Force-constant calculation for H2O atom displacements
00040 |   force_constants:
00041 |     enabled: true
00042 |     displacement: 0.05 Ang
00043 |     target_count: null
00044 |     include_reference: true
00045 |     expand_amplitudes: false
00046 |     subsampling:
00047 |       method: spread
00048 |       seed: 0
00049 |     random_seed: 42
00050 |     combination_mode: aligned
00051 |     max_datasets: 100
00052 |     displacements:
00053 |       0.02 Ang:
00054 |       - 5
00055 |       - 7
00056 |       - 9
00057 |       0.03 Ang:
00058 |       - 4
00059 |       - 6
00060 |       - 8
00061 |       0.05 Ang:
00062 |       - 2
00063 |       - 10
00064 |       - 13
00065 |     allow_missing_matrix: true
00066 |     first_atom: 1
00067 |     last_atom: null
00068 |     lua_script: md_store.lua
00069 |     save_tshs: true
00070 |     save_tsde: true
00071 |     save_dhs: true
00072 |     dHdR_tolerance: -1 Ry/Bohr
00073 |     dSdR_tolerance: -1 1/Bohr
00074 |   random_cartesian:
00075 |     enabled: false
00076 |     recipe: legacy_components
00077 |     n_structures: 100
00078 |     seed: 1234
00079 |     distribution: gaussian
00080 |     sigma_ang: 0.03
00081 |     uniform_range_ang: 0.05
00082 |     move_atoms: all
00083 |     species_filter: []
00084 |     min_distance_ang: 0.65
00085 |     max_attempts_per_structure: 100
00086 |     remove_center_of_mass_translation: true
00087 |     components:
00088 |       atom_displacement:
00089 |         enabled: true
00090 |         distribution: gaussian
00091 |         sigma_ang: 0.03
00092 |         uniform_range_ang: 0.05
00093 |         move_atoms: all
00094 |         species_filter: []
00095 |         remove_center_of_mass_translation: true
00096 |       bond_displacement:
00097 |         enabled: false
00098 |         distribution: gaussian
00099 |         sigma_ang: 0.01
00100 |         uniform_range_ang: 0.02
00101 |         min_delta_ang: null
00102 |         max_delta_ang: null
00103 |         min_bond_ang: 0.70
00104 |         max_bond_ang: 1.30
00105 |         bonds: h2o_oh
00106 |       angle_displacement:
00107 |         enabled: false
00108 |         distribution: gaussian
00109 |         sigma_deg: 3.0
00110 |         uniform_range_deg: 5.0
00111 |         min_delta_deg: null
00112 |         max_delta_deg: null
00113 |         min_angle_deg: 80.0
00114 |         max_angle_deg: 130.0
00115 |         angles: h2o_hoh
00116 |     validation:
00117 |       min_distance_ang: 0.65
00118 |       max_rmsd_from_reference_ang: null
00119 |       max_attempts_per_structure: 100
00120 |   lattice_constant:
00121 |     value: 1.0
00122 |     unit: Ang
00123 |   lattice_vectors:
00124 |   - - 15.0
00125 |     - 0.0
00126 |     - 0.0
00127 |   - - 0.0
00128 |     - 15.0
00129 |     - 0.0
00130 |   - - 0.0
00131 |     - 0.0
00132 |     - 15.0
00133 |   coordinates_format: Ang
00134 |   species:
00135 |   - index: 1
00136 |     atomic_number: 8
00137 |     symbol: O
00138 |   - index: 2
00139 |     atomic_number: 1
00140 |     symbol: H
00141 |   atoms:
00142 |   - label: O
00143 |     species_index: 1
00144 |     position:
00145 |     - 7.5
00146 |     - 7.5
00147 |     - 7.619262
00148 |   - label: H
00149 |     species_index: 2
00150 |     position:
00151 |     - 7.5
00152 |     - 8.263239
00153 |     - 7.022953
00154 |   - label: H
00155 |     species_index: 2
00156 |     position:
00157 |     - 7.5
00158 |     - 6.836839
00159 |     - 7.022953
00160 |   kgrid_monkhorst_pack:
00161 |   - - 1
00162 |     - 0
00163 |     - 0
00164 |     - 0.0
00165 |   - - 0
00166 |     - 1
00167 |     - 0
00168 |     - 0.0
00169 |   - - 0
00170 |     - 0
00171 |     - 1
00172 |     - 0.0
00173 |   siesta:
00174 |     ForceAuxCell: T
00175 |     MeshCutoff: 200 Ry
00176 |     PAO.BasisType: split
00177 |     PAO.BasisSize: DZP
00178 |     PAO.EnergyShift: 0.03 eV
00179 |     XC.functional: GGA
00180 |     XC.authors: PBE
00181 |     MaxSCFIterations: 200
00182 |     SolutionMethod: diagon
00183 |     DM.MixingWeight: 0.02
00184 |     DM.NumberPulay: 3
00185 |     DM.Tolerance: 1.d-5
00186 |     DM.Require.Energy.Convergence: T
00187 |     DM.Energy.Tolerance: 1.e-5 eV
00188 |     SCF.MixAfterConvergence: F
00189 |     SpinPolarized: F
00190 |     FixSpin: F
00191 |     NonCollinearSpin: F
00192 |     DM.InitSpinAF: F
00193 |     ON.UseSaveLWF: T
00194 |     DM.UseSaveDM: T
00195 |     UseSaveData: T
00196 |     LongOutput: T
00197 |     WriteCoorXmol: T
00198 |     WriteCoorStep: T
00199 |     WriteForces: T
00200 |     Save.HS: T
00201 |     XML.Write: T
00202 |   relaxation_md:
00203 |     MD.TypeOfRun: CG
00204 |     MD.NumCGsteps: 200
00205 |     MD.MaxForceTol: 0.04 eV/Ang
00206 |     MD.MaxCGDispl: 0.03 Bohr
00207 |     MD.VariableCell: F
00208 |     MD.ConstantVolume: T
00209 |     MD.UseSaveXV: T
00210 |     MD.UseSaveCG: T
00211 |     WriteMDHistory: T
00212 |   single_point_overrides:
00213 |     DM.UseSaveDM: F
00214 | splits:
00215 |   train: 0.8
00216 |   validation: 0.1
00217 |   test: 0.1
00218 | single_points:
00219 |   limit: null
00220 |   rerun: false
00221 |   # Scientific hardening default: references are reusable only when RUN.out
00222 |   # proves job completion and SCF convergence. Use the CLI debug flag only for
00223 |   # unsafe local recovery/debug workflows.
00224 |   allow_unvalidated_matrices: false
00225 |   workers: 1
00226 | training:
00227 |   torch_float32_matmul_precision: null
00228 |   data:
00229 |     out_matrix: hamiltonian
00230 |     symmetric_matrix: true
00231 |     sub_point_matrix: false
00232 |     matrix_component_policy: h_only
00233 |     n_matrix_components: 1
00234 |     basis_files: ../dataset/FC_steps/basis/*.ion.xml
00235 |     runs_json: runs.json
00236 |     val_runs: ../dataset/validation_samples/*/RUN.fdf
00237 |     batch_size: 10
00238 |     store_in_memory: true
00239 |   model:
00240 |     num_interactions: 1
00241 |     correlation: 1
00242 |     max_ell: 2
00243 |     hidden_irreps: 10x0e + 10x1o + 10x2e
00244 |     loss: graph2mat.metrics.block_type_mae
00245 |     optim_lr: 0.005
00246 |   trainer:
00247 |     accelerator: cpu
00248 |     logger:
00249 |       class_path: TensorBoardLogger
00250 |       init_args:
00251 |         name: atom_displacement_model
00252 |         save_dir: lightning_logs
00253 |     max_epochs: 200
00254 |   fit_args:
00255 |   - models
00256 |   - mace
00257 |   - main
00258 |   - fit
00259 |   - -c
00260 |   - config.yaml
00261 |   min_completed_samples: 2
00262 |   tensorboard_hint: tensorboard --logdir lightning_logs
00263 | checkpoint:
00264 |   path: null
00265 |   auto_best: true
00266 |   search_glob: lightning_logs/**/checkpoints/best-*.ckpt
00267 |   selection: latest_version
00268 | testing:
00269 |   sample_index: 0
00270 |   data:
00271 |     out_matrix: hamiltonian
00272 |     symmetric_matrix: true
00273 |     sub_point_matrix: false
00274 |     basis_files: ../dataset/FC_steps/basis/*.ion.xml
00275 |     store_in_memory: true
00276 |     matrix_component_policy: h_only
00277 |     n_matrix_components: 1
00278 |   callbacks:
00279 |     plot_matrix_error: true
00280 |     show_plot: false
00281 |     store_plot_in_logger: false
00282 |     samplewise_metrics_logger: true
00283 |     output_file: sample_metrics.csv
00284 | prediction:
00285 |   data:
00286 |     out_matrix: hamiltonian
00287 |     symmetric_matrix: true
00288 |     sub_point_matrix: false
00289 |     basis_files: ../dataset/FC_steps/basis/*.ion.xml
00290 |     predict_structs: ../dataset/FC_steps/*/RUN.fdf
00291 |     store_in_memory: true
00292 |     matrix_component_policy: h_only
00293 |     n_matrix_components: 1
00294 |   callbacks:
00295 |     matrix_writer: true
00296 |     output_file: ML_prediction.HSX
00297 | pipeline:
00298 |   skip_model_test: false
00299 |   steps:
00300 |   - render_inputs
00301 |   - generate_atom_displacement_dataset
00302 |   - run_single_points
00303 |   - normalize_fc_steps
00304 |   - collect_atom_displacement_dataset
00305 |   - run_atdisp_training
00306 |   - run_atdisp_testing
00307 |   - run_atdisp_prediction
00308 | performance:
00309 |   max_parallel_siesta_jobs: 1
00310 |   max_parallel_dataset_jobs: 1
00311 |   max_parallel_prediction_jobs: 1
00312 |   max_parallel_evaluation_jobs: 1
00313 |   max_parallel_metric_jobs: 1
00314 |   omp_num_threads: null
00315 |   mkl_num_threads: null
00316 |   openblas_num_threads: null
00317 |   numexpr_num_threads: null
00318 |   torch_num_threads: null
00319 |   compute_accelerator: cpu
00320 |   batch_size: null
00321 |   store_in_memory: null
00322 |   reuse_validated_siesta_outputs: true
00323 |   enable_experiment_cache: false
00324 |   error_policy: fail_fast
00325 |   preset: null
00326 |   torch_float32_matmul_precision: null
```

## `Comparison/config/shared_siesta_settings.yaml`

SHA-256: `0286aa0a6903f71e5edf97756c73285ec5bbe2d3f15780f84beef89b73f8a6ee`

```yaml
00001 | # Shared SIESTA settings used as the strict-comparison reference for MD vs
00002 | # AtomicDisplacement experiments. The current pass uses this file for hashing and
00003 | # mismatch warnings; generation code still reads each pipeline config directly.
00004 | lattice_constant:
00005 |   value: 1.0
00006 |   unit: Ang
00007 | lattice_vectors:
00008 |   - [15.0, 0.0, 0.0]
00009 |   - [0.0, 15.0, 0.0]
00010 |   - [0.0, 0.0, 15.0]
00011 | PAO.BasisType: split
00012 | PAO.BasisSize: DZP
00013 | PAO.EnergyShift: 0.03 eV
00014 | MeshCutoff: 200 Ry
00015 | XC.functional: GGA
00016 | XC.authors: PBE
00017 | MaxSCFIterations: 200
00018 | SolutionMethod: diagon
00019 | DM.MixingWeight: 0.02
00020 | DM.NumberPulay: 3
00021 | DM.Tolerance: 1.d-5
00022 | DM.Require.Energy.Convergence: T
00023 | DM.Energy.Tolerance: 1.e-5 eV
00024 | SpinPolarized: F
00025 | FixSpin: F
00026 | NonCollinearSpin: F
00027 | ForceAuxCell: T
00028 | Save.HS: T
00029 | TS.HS.Save: T
00030 | TS.DE.Save: T
00031 | XML.Write: T
```

## `AtomDisplacement/scripts/generate_atom_displacement_dataset.py`

SHA-256: `9e2dfaa24f28ae84f60814db8a0499697748845ecf88675dbfafa66750e0ab86`

```py
00001 | #!/usr/bin/env python3
00002 | """Generate one or more SIESTA FC inputs for atom-displacement datasets."""
00003 | 
00004 | from __future__ import annotations
00005 | 
00006 | import copy
00007 | import re
00008 | import shutil
00009 | from pathlib import Path
00010 | from typing import Any
00011 | 
00012 | from atom_displacement_utils import (
00013 |     BASE_DIR,
00014 |     DATASET_DIR,
00015 |     FC_RUNS_DIR_NAME,
00016 |     PIPELINE_CONFIG,
00017 |     PIPELINE_PATHS,
00018 |     compute_max_fc_structures,
00019 |     compute_water_geometry_metrics,
00020 |     copy_pseudopotentials,
00021 |     ensure_dir,
00022 |     fc_displaced_atom_count,
00023 |     load_reference_structure,
00024 |     run_command_in_venv,
00025 |     write_json,
00026 | )
00027 | from pipeline_config_utils import command, render_single_point_fdf
00028 | 
00029 | 
00030 | STORE_DIR_NAME = "AtDis_steps"
00031 | 
00032 | 
00033 | def _unit_from_displacement(value: str) -> str:
00034 |     match = re.fullmatch(r"\s*[-+0-9.Ee]+\s*([A-Za-z/]+)?\s*", str(value))
00035 |     return (match.group(1) if match and match.group(1) else "Ang")
00036 | 
00037 | 
00038 | def _format_displacement_value(value: Any, default_unit: str) -> str:
00039 |     if isinstance(value, (int, float)):
00040 |         return f"{value} {default_unit}"
00041 |     text = str(value).strip()
00042 |     if re.fullmatch(r"[-+0-9.Ee]+", text):
00043 |         return f"{text} {default_unit}"
00044 |     return text
00045 | 
00046 | 
00047 | def _safe_slug(value: str) -> str:
00048 |     slug = re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_").lower()
00049 |     return slug or "displacement"
00050 | 
00051 | 
00052 | def setup_lua_store(run_dir: Path, lua_script_name: str) -> None:
00053 |     """Prepare Graph2Mat's SIESTA store script inside one raw FC run directory."""
00054 | 
00055 |     run_command_in_venv(
00056 |         [command(PIPELINE_CONFIG, "graph2mat"), "siesta", "md", "setup-store"],
00057 |         cwd=run_dir,
00058 |     )
00059 |     lua_script = run_dir / lua_script_name
00060 |     text = lua_script.read_text(encoding="utf-8")
00061 |     text = text.replace('local store_dir = "MD_steps"', f'local store_dir = "{STORE_DIR_NAME}"')
00062 |     lua_script.write_text(text, encoding="utf-8")
00063 | 
00064 | 
00065 | def configured_displacements(force_constants: dict[str, Any]) -> list[dict[str, Any]]:
00066 |     """Return normalized displacement entries while preserving legacy config.
00067 | 
00068 |     New configs can define ``structure.force_constants.displacements`` as a list
00069 |     of ``{value, n_structures}`` objects. If that list is absent, the legacy
00070 |     single ``displacement``/``target_count`` pair is converted into one entry.
00071 |     """
00072 | 
00073 |     default_unit = _unit_from_displacement(force_constants.get("displacement", "0.05 Ang"))
00074 |     raw_entries = force_constants.get("displacements")
00075 |     if not raw_entries:
00076 |         raw_entries = [
00077 |             {
00078 |                 "value": force_constants.get("displacement", "0.05 Ang"),
00079 |                 "n_structures": force_constants.get("target_count"),
00080 |             }
00081 |         ]
00082 |     elif isinstance(raw_entries, dict):
00083 |         def max_requested_count(value: Any) -> int | None:
00084 |             if value in (None, ""):
00085 |                 return None
00086 |             if isinstance(value, (int, float)):
00087 |                 return int(value)
00088 |             return max(int(count) for count in value) if value else None
00089 | 
00090 |         raw_entries = [
00091 |             {
00092 |                 "value": displacement,
00093 |                 # Standalone generation prepares enough selected FC steps for
00094 |                 # the largest requested option. The comparison UI expands exact
00095 |                 # per-dataset aligned/cartesian combinations before this script
00096 |                 # is called.
00097 |                 "n_structures": max_requested_count(counts),
00098 |             }
00099 |             for displacement, counts in sorted(
00100 |                 raw_entries.items(),
00101 |                 key=lambda item: str(item[0]),
00102 |             )
00103 |         ]
00104 | 
00105 |     entries: list[dict[str, Any]] = []
00106 |     for index, raw_entry in enumerate(raw_entries):
00107 |         if isinstance(raw_entry, dict):
00108 |             value = raw_entry.get("value", force_constants.get("displacement", "0.05 Ang"))
00109 |             n_structures = raw_entry.get("n_structures", force_constants.get("target_count"))
00110 |             label = raw_entry.get("label")
00111 |             unit = raw_entry.get("unit", default_unit)
00112 |         else:
00113 |             value = raw_entry
00114 |             n_structures = force_constants.get("target_count")
00115 |             label = None
00116 |             unit = default_unit
00117 |         displacement = _format_displacement_value(value, unit)
00118 |         entries.append(
00119 |             {
00120 |                 "index": index,
00121 |                 "label": str(label) if label else f"disp_{index:03d}_{_safe_slug(displacement)}",
00122 |                 "value": displacement,
00123 |                 "n_structures": None if n_structures is None else int(n_structures),
00124 |             }
00125 |         )
00126 |     return entries
00127 | 
00128 | 
00129 | def build_run_config(displacement_value: str, system_label: str) -> dict[str, Any]:
00130 |     config = copy.deepcopy(PIPELINE_CONFIG)
00131 |     molecule_name = config["structure"].get("molecule_name", "system")
00132 |     config["structure"]["force_constants"]["displacement"] = displacement_value
00133 |     config["structure"]["single_point"]["system_name_template"] = f"{molecule_name} {{sample_id}}"
00134 |     config["structure"]["single_point"]["title"] = (
00135 |         f"Force-constant calculation for {system_label}"
00136 |     )
00137 |     return config
00138 | 
00139 | 
00140 | def write_fc_run(
00141 |     *,
00142 |     run_dir: Path,
00143 |     reference: Any,
00144 |     system_label: str,
00145 |     displacement_value: str,
00146 |     force_constants: dict[str, Any],
00147 | ) -> None:
00148 |     ensure_dir(run_dir)
00149 |     setup_lua_store(run_dir, force_constants["lua_script"])
00150 |     copy_pseudopotentials(BASE_DIR, run_dir)
00151 |     run_config = build_run_config(displacement_value, system_label)
00152 |     content = render_single_point_fdf(
00153 |         run_config,
00154 |         positions_ang=reference.positions_ang,
00155 |         atom_species=reference.atom_species,
00156 |         sample_id=system_label,
00157 |     )
00158 |     (run_dir / PIPELINE_CONFIG["paths"]["run_fdf_name"]).write_text(content, encoding="utf-8")
00159 | 
00160 | 
00161 | def main() -> int:
00162 |     ensure_dir(DATASET_DIR)
00163 |     fc_runs_dir = DATASET_DIR / FC_RUNS_DIR_NAME
00164 |     if fc_runs_dir.exists():
00165 |         shutil.rmtree(fc_runs_dir)
00166 |     ensure_dir(fc_runs_dir)
00167 | 
00168 |     reference, source_path = load_reference_structure()
00169 |     metrics = compute_water_geometry_metrics(reference)
00170 |     force_constants = PIPELINE_CONFIG["structure"]["force_constants"]
00171 |     include_reference = bool(force_constants.get("include_reference", True))
00172 |     first_atom = int(force_constants.get("first_atom", 1))
00173 |     last_atom = force_constants.get("last_atom")
00174 |     last_atom = len(reference.atom_species) if last_atom is None else int(last_atom)
00175 |     displaced_atoms = fc_displaced_atom_count(
00176 |         len(reference.atom_species),
00177 |         first_atom,
00178 |         last_atom,
00179 |     )
00180 |     max_structures = compute_max_fc_structures(displaced_atoms, include_reference)
00181 |     displacement_entries = configured_displacements(force_constants)
00182 | 
00183 |     print("=== AtomDisplacement dataset generation ===")
00184 |     print(f"[INFO] Geometria de referencia: {source_path}")
00185 |     print("[INFO] Modo de desplazamiento: SIESTA MD.TypeOfRun FC")
00186 |     print(f"[INFO] Rango de atomos FC: {first_atom}-{last_atom}")
00187 |     print(
00188 |         "[INFO] Limite FC por magnitud: "
00189 |         f"6N{' + 1 referencia' if include_reference else ''} = {max_structures} "
00190 |         f"(N desplazados={displaced_atoms})"
00191 |     )
00192 | 
00193 |     runs = []
00194 |     for entry in displacement_entries:
00195 |         requested = entry["n_structures"] if entry["n_structures"] is not None else max_structures
00196 |         if requested > max_structures:
00197 |             raise ValueError(
00198 |                 "FC cannot generate the requested number of structures for one "
00199 |                 f"displacement magnitude: requested={requested}, max={max_structures}. "
00200 |                 "The FC method is limited to +/- displacements along 3 Cartesian "
00201 |                 "directions for each selected atom."
00202 |             )
00203 | 
00204 |         system_label = f"fc_{entry['index']:03d}_{_safe_slug(entry['value'])}"
00205 |         run_dir = fc_runs_dir / entry["label"]
00206 |         print(
00207 |             f"[INFO] FC run {entry['index']}: displacement={entry['value']}, "
00208 |             f"requested={requested}, run_dir={run_dir}"
00209 |         )
00210 |         write_fc_run(
00211 |             run_dir=run_dir,
00212 |             reference=reference,
00213 |             system_label=system_label,
00214 |             displacement_value=entry["value"],
00215 |             force_constants=force_constants,
00216 |         )
00217 | 
00218 |         force_constants_metadata = {
00219 |             "md_type_of_run": "FC",
00220 |             "displacement": entry["value"],
00221 |             "requested_structures": requested,
00222 |             "max_structures": max_structures,
00223 |             "include_reference": include_reference,
00224 |             "first_atom": first_atom,
00225 |             "last_atom": last_atom,
00226 |             "lua_script": force_constants.get("lua_script"),
00227 |             "save_tshs": bool(force_constants.get("save_tshs", True)),
00228 |             "save_tsde": bool(force_constants.get("save_tsde", True)),
00229 |             "save_dhs": bool(force_constants.get("save_dhs", True)),
00230 |             "dHdR_tolerance": force_constants.get("dHdR_tolerance"),
00231 |             "dSdR_tolerance": force_constants.get("dSdR_tolerance"),
00232 |         }
00233 |         metadata = {
00234 |             "id": system_label,
00235 |             "generation_mode": "siesta_fc_run",
00236 |             "method": "siesta_fc_cartesian",
00237 |             "dataset_recipe": PIPELINE_CONFIG.get("dataset_recipe") or {},
00238 |             "recipe_id": (PIPELINE_CONFIG.get("dataset_recipe") or {}).get("recipe_id"),
00239 |             "recipe_label": (PIPELINE_CONFIG.get("dataset_recipe") or {}).get("recipe_label"),
00240 |             "block_id": (PIPELINE_CONFIG.get("dataset_recipe") or {}).get("block_id"),
00241 |             "block_label": (PIPELINE_CONFIG.get("dataset_recipe") or {}).get("block_label"),
00242 |             "generation_parameters_json": (PIPELINE_CONFIG.get("dataset_recipe") or {}).get("generation_parameters_json"),
00243 |             "reference_source": source_path,
00244 |             "positions_ang": reference.positions_ang,
00245 |             "geometry_metrics": metrics,
00246 |             "force_constants": force_constants_metadata,
00247 |             "expected_outputs": {
00248 |                 "force_constants": "FC",
00249 |                 "hamiltonian_derivatives": f"{system_label}.dHSdR.nc",
00250 |             },
00251 |         }
00252 |         write_json(run_dir / "metadata.json", metadata)
00253 |         runs.append(
00254 |             {
00255 |                 "index": entry["index"],
00256 |                 "id": system_label,
00257 |                 "label": entry["label"],
00258 |                 "run_dir": str(run_dir),
00259 |                 "run_fdf": str(run_dir / PIPELINE_CONFIG["paths"]["run_fdf_name"]),
00260 |                 "displacement": entry["value"],
00261 |                 "requested_structures": requested,
00262 |                 "max_structures": max_structures,
00263 |                 "include_reference": include_reference,
00264 |                 "first_atom": first_atom,
00265 |                 "last_atom": last_atom,
00266 |                 "recipe_id": metadata.get("recipe_id"),
00267 |                 "recipe_label": metadata.get("recipe_label"),
00268 |                 "block_id": metadata.get("block_id"),
00269 |                 "block_label": metadata.get("block_label"),
00270 |                 "generation_parameters_json": metadata.get("generation_parameters_json"),
00271 |             }
00272 |         )
00273 | 
00274 |     manifest = {
00275 |         "generation_mode": "siesta_fc_multi_run",
00276 |         "dataset_recipe": PIPELINE_CONFIG.get("dataset_recipe") or {},
00277 |         "reference_source": source_path,
00278 |         "fc_runs_dir": str(fc_runs_dir),
00279 |         "subsampling": force_constants.get("subsampling", {"method": "spread", "seed": 0}),
00280 |         "force_constants": {
00281 |             "first_atom": first_atom,
00282 |             "last_atom": last_atom,
00283 |             "include_reference": include_reference,
00284 |             "max_structures_per_displacement": max_structures,
00285 |         },
00286 |         "runs": runs,
00287 |     }
00288 |     write_json(PIPELINE_PATHS["samples_manifest_path"], manifest)
00289 |     print(f"[OK] Entradas FC generadas en {fc_runs_dir}")
00290 |     return 0
00291 | 
00292 | 
00293 | if __name__ == "__main__":
00294 |     raise SystemExit(main())
```

## `AtomDisplacement/scripts/generate_generic_cartesian_displacement_dataset.py`

SHA-256: `fb7149bb1204d3e5f74f67fb2f8301b060a201eefd490925584ae2791f0f5e29`

```py
00001 | #!/usr/bin/env python3
00002 | """Generate material-aware Cartesian atom-displacement samples.
00003 | 
00004 | This generator is intentionally separate from the historical SIESTA FC/H2O
00005 | flow. It starts from a validated material bundle and writes explicit +/-x/y/z
00006 | single-point FDF inputs for the selected atoms.
00007 | """
00008 | 
00009 | from __future__ import annotations
00010 | 
00011 | import argparse
00012 | import json
00013 | import shutil
00014 | import sys
00015 | from dataclasses import dataclass
00016 | from pathlib import Path
00017 | from typing import Any
00018 | 
00019 | 
00020 | ATOM_ROOT = Path(__file__).resolve().parents[1]
00021 | REPO_ROOT = ATOM_ROOT.parent
00022 | SHARED_DIR = REPO_ROOT / "shared"
00023 | for candidate in (ATOM_ROOT / "scripts", SHARED_DIR):
00024 |     if str(candidate) not in sys.path:
00025 |         sys.path.insert(0, str(candidate))
00026 | 
00027 | from fdf_materialization import extract_bundle_structure, materialize_sample_fdf  # noqa: E402
00028 | from material_bundle import BASIS_EXTENSIONS, ValidatedMaterialBundle, file_sha256  # noqa: E402
00029 | from material_presets import resolve_material_bundle  # noqa: E402
00030 | from pipeline_config_utils import load_pipeline_config, paths  # noqa: E402
00031 | 
00032 | 
00033 | AXES: tuple[tuple[str, int], ...] = (("x", 0), ("y", 1), ("z", 2))
00034 | SIGNS: tuple[int, ...] = (1, -1)
00035 | GENERATION_MODE = "generic_cartesian_displacement"
00036 | METHOD_ID = "siesta_fc_cartesian"
00037 | 
00038 | 
00039 | class GenericCartesianDisplacementError(RuntimeError):
00040 |     """Raised when the generic Cartesian recipe cannot be generated safely."""
00041 | 
00042 | 
00043 | @dataclass(frozen=True)
00044 | class GenericCartesianSettings:
00045 |     recipe: str
00046 |     amplitude_ang: float
00047 |     selected_species: set[str] | None
00048 |     include_base: bool
00049 |     overwrite: bool
00050 | 
00051 | 
00052 | def _write_json(path: Path, payload: Any) -> None:
00053 |     path.parent.mkdir(parents=True, exist_ok=True)
00054 |     path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
00055 | 
00056 | 
00057 | def _parse_amplitude_ang(value: Any) -> float:
00058 |     if isinstance(value, (int, float)):
00059 |         amplitude = float(value)
00060 |     else:
00061 |         parts = str(value).strip().split()
00062 |         if not parts:
00063 |             raise GenericCartesianDisplacementError("atomic_displacement.amplitude_ang cannot be empty.")
00064 |         try:
00065 |             amplitude = float(parts[0])
00066 |         except ValueError as exc:
00067 |             raise GenericCartesianDisplacementError(
00068 |                 f"atomic_displacement.amplitude_ang must be numeric in Ang: {value!r}"
00069 |             ) from exc
00070 |         if len(parts) > 1 and parts[1].lower() not in {"ang", "angstrom", "angstroms"}:
00071 |             raise GenericCartesianDisplacementError(
00072 |                 "atomic_displacement.amplitude_ang only supports Ang units."
00073 |             )
00074 |     if amplitude <= 0:
00075 |         raise GenericCartesianDisplacementError(
00076 |             "atomic_displacement.amplitude_ang must be positive."
00077 |         )
00078 |     return amplitude
00079 | 
00080 | 
00081 | def _parse_selected_species(value: Any) -> set[str] | None:
00082 |     if value in (None, "", "all"):
00083 |         return None
00084 |     if isinstance(value, str):
00085 |         items = [item.strip() for item in value.split(",")]
00086 |     elif isinstance(value, (list, tuple, set)):
00087 |         items = [str(item).strip() for item in value]
00088 |     else:
00089 |         raise GenericCartesianDisplacementError(
00090 |             "atomic_displacement.selected_species must be null, a string, or a list."
00091 |         )
00092 |     selected = {item for item in items if item}
00093 |     if not selected:
00094 |         raise GenericCartesianDisplacementError(
00095 |             "atomic_displacement.selected_species cannot be empty; use null to select all species."
00096 |         )
00097 |     return selected
00098 | 
00099 | 
00100 | def _recipe_config(config: dict[str, Any]) -> dict[str, Any]:
00101 |     raw = config.get("atomic_displacement")
00102 |     if raw is None:
00103 |         raw = config.get("structure", {}).get("atomic_displacement")
00104 |     if raw is None:
00105 |         raw = {}
00106 |     if not isinstance(raw, dict):
00107 |         raise GenericCartesianDisplacementError("atomic_displacement config must be a mapping.")
00108 |     return raw
00109 | 
00110 | 
00111 | def generic_cartesian_settings(config: dict[str, Any]) -> GenericCartesianSettings:
00112 |     raw = _recipe_config(config)
00113 |     recipe = str(raw.get("recipe", "generic_cartesian")).strip()
00114 |     if recipe != "generic_cartesian":
00115 |         raise GenericCartesianDisplacementError(
00116 |             "generate_generic_cartesian_displacement_dataset.py only supports "
00117 |             "atomic_displacement.recipe='generic_cartesian'. H2O-specific "
00118 |             "bond/angle or SIESTA FC recipes remain in their existing generators."
00119 |         )
00120 |     return GenericCartesianSettings(
00121 |         recipe=recipe,
00122 |         amplitude_ang=_parse_amplitude_ang(raw.get("amplitude_ang", 0.03)),
00123 |         selected_species=_parse_selected_species(raw.get("selected_species")),
00124 |         include_base=bool(raw.get("include_base", True)),
00125 |         overwrite=bool(raw.get("overwrite", False)),
00126 |     )
00127 | 
00128 | 
00129 | def _sample_id(config: dict[str, Any], index: int) -> str:
00130 |     template = str(config.get("generation", {}).get("sample_id_format", "sample_{index:04d}"))
00131 |     try:
00132 |         sample_id = template.format(index=index)
00133 |     except (KeyError, IndexError, ValueError) as exc:
00134 |         raise GenericCartesianDisplacementError(
00135 |             f"Invalid generation.sample_id_format for generic Cartesian samples: {template!r}"
00136 |         ) from exc
00137 |     if not sample_id or "/" in sample_id or "\\" in sample_id or sample_id in {".", ".."}:
00138 |         raise GenericCartesianDisplacementError(
00139 |             f"generation.sample_id_format produced an unsafe sample id: {sample_id!r}"
00140 |         )
00141 |     return sample_id
00142 | 
00143 | 
00144 | def _prepare_output_root(output_root: Path, *, overwrite: bool) -> None:
00145 |     if output_root.exists():
00146 |         existing = list(output_root.iterdir())
00147 |         if existing and not overwrite:
00148 |             raise GenericCartesianDisplacementError(
00149 |                 f"Output sample directory already exists and is not empty: {output_root}. "
00150 |                 "Set atomic_displacement.overwrite=true or pass --overwrite to regenerate it."
00151 |             )
00152 |         if overwrite:
00153 |             shutil.rmtree(output_root)
00154 |     output_root.mkdir(parents=True, exist_ok=True)
00155 | 
00156 | 
00157 | def _copy_material_inputs(validated: ValidatedMaterialBundle, sample_dir: Path) -> dict[str, str]:
00158 |     copied: dict[str, str] = {}
00159 |     for label, pseudo_path in sorted(validated.pseudopotentials.items()):
00160 |         target = sample_dir / pseudo_path.name
00161 |         shutil.copy2(pseudo_path, target)
00162 |         copied[label] = target.name
00163 |     return copied
00164 | 
00165 | 
00166 | def _copy_basis_files(validated: ValidatedMaterialBundle, output_root: Path) -> dict[str, str]:
00167 |     if validated.bundle.basis_dir is None:
00168 |         return {}
00169 |     target_dir = output_root / "basis"
00170 |     target_dir.mkdir(parents=True, exist_ok=True)
00171 |     copied: dict[str, str] = {}
00172 |     for path in sorted(item for item in validated.bundle.basis_dir.iterdir() if item.is_file()):
00173 |         if any(path.name.endswith(extension) for extension in BASIS_EXTENSIONS):
00174 |             target = target_dir / path.name
00175 |             shutil.copy2(path, target)
00176 |             copied[path.name] = file_sha256(target)
00177 |     return copied
00178 | 
00179 | 
00180 | def _species_by_index(structure: Any) -> dict[int, str]:
00181 |     return {int(species.index): str(species.label) for species in structure.species}
00182 | 
00183 | 
00184 | def _selected_atom_records(structure: Any, selected_species: set[str] | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
00185 |     species_by_index = _species_by_index(structure)
00186 |     selected: list[dict[str, Any]] = []
00187 |     skipped: list[dict[str, Any]] = []
00188 |     for index, atom in enumerate(structure.atoms, start=1):
00189 |         label = species_by_index[int(atom.species_index)]
00190 |         record = {
00191 |             "atom_index": index,
00192 |             "atom_index_zero_based": index - 1,
00193 |             "species": label,
00194 |             "species_index": int(atom.species_index),
00195 |         }
00196 |         if selected_species is None or label in selected_species:
00197 |             selected.append(record)
00198 |         else:
00199 |             skipped.append(record)
00200 |     if not selected:
00201 |         expected = ", ".join(sorted(selected_species or []))
00202 |         raise GenericCartesianDisplacementError(
00203 |             f"atomic_displacement.selected_species matched no atoms: {expected}"
00204 |         )
00205 |     return selected, skipped
00206 | 
00207 | 
00208 | def _with_displacement(
00209 |     positions: list[tuple[float, float, float]],
00210 |     *,
00211 |     atom_index_zero_based: int,
00212 |     axis_index: int,
00213 |     delta_ang: float,
00214 | ) -> list[list[float]]:
00215 |     updated = [list(position) for position in positions]
00216 |     updated[atom_index_zero_based][axis_index] += delta_ang
00217 |     return updated
00218 | 
00219 | 
00220 | def _base_metadata(
00221 |     *,
00222 |     sample_id: str,
00223 |     material_label: str,
00224 |     settings: GenericCartesianSettings,
00225 |     materialized_metadata: dict[str, Any],
00226 | ) -> dict[str, Any]:
00227 |     metadata = {
00228 |         "id": sample_id,
00229 |         "sample_id": sample_id,
00230 |         "generation_mode": GENERATION_MODE,
00231 |         "generation_method": "generic_cartesian",
00232 |         "method": METHOD_ID,
00233 |         "recipe": settings.recipe,
00234 |         "material_label": material_label,
00235 |         "is_reference": True,
00236 |         "atom_index": None,
00237 |         "atom_index_zero_based": None,
00238 |         "species": None,
00239 |         "axis": None,
00240 |         "axis_index": None,
00241 |         "sign": None,
00242 |         "sign_label": None,
00243 |         "amplitude_ang": 0.0,
00244 |         "displacement_ang": [0.0, 0.0, 0.0],
00245 |         "split_group_id": f"{GENERATION_MODE}:{material_label}:reference",
00246 |     }
00247 |     metadata.update(materialized_metadata)
00248 |     return metadata
00249 | 
00250 | 
00251 | def _displacement_metadata(
00252 |     *,
00253 |     sample_id: str,
00254 |     material_label: str,
00255 |     settings: GenericCartesianSettings,
00256 |     atom_record: dict[str, Any],
00257 |     axis: str,
00258 |     axis_index: int,
00259 |     sign: int,
00260 |     materialized_metadata: dict[str, Any],
00261 | ) -> dict[str, Any]:
00262 |     displacement = [0.0, 0.0, 0.0]
00263 |     displacement[axis_index] = sign * settings.amplitude_ang
00264 |     metadata = {
00265 |         "id": sample_id,
00266 |         "sample_id": sample_id,
00267 |         "generation_mode": GENERATION_MODE,
00268 |         "generation_method": "generic_cartesian",
00269 |         "method": METHOD_ID,
00270 |         "recipe": settings.recipe,
00271 |         "material_label": material_label,
00272 |         "is_reference": False,
00273 |         "atom_index": atom_record["atom_index"],
00274 |         "atom_index_zero_based": atom_record["atom_index_zero_based"],
00275 |         "species": atom_record["species"],
00276 |         "species_index": atom_record["species_index"],
00277 |         "axis": axis,
00278 |         "axis_index": axis_index,
00279 |         "sign": sign,
00280 |         "sign_label": "+" if sign > 0 else "-",
00281 |         "amplitude_ang": settings.amplitude_ang,
00282 |         "displacement_ang": displacement,
00283 |         "split_group_id": (
00284 |             f"{GENERATION_MODE}:{material_label}:atom_{atom_record['atom_index']:04d}"
00285 |         ),
00286 |     }
00287 |     metadata.update(materialized_metadata)
00288 |     return metadata
00289 | 
00290 | 
00291 | def _write_sample(
00292 |     *,
00293 |     sample_dir: Path,
00294 |     sample_id: str,
00295 |     positions_ang: list[list[float]] | list[tuple[float, float, float]],
00296 |     validated: ValidatedMaterialBundle,
00297 |     structure: Any,
00298 |     metadata: dict[str, Any],
00299 | ) -> dict[str, Any]:
00300 |     sample_dir.mkdir(parents=True, exist_ok=False)
00301 |     copied_pseudos = _copy_material_inputs(validated, sample_dir)
00302 |     materialized = materialize_sample_fdf(
00303 |         validated.bundle.fdf,
00304 |         sample_dir / "RUN.fdf",
00305 |         positions_ang=positions_ang,
00306 |         atom_species=structure.atom_species,
00307 |         lattice_vectors_ang=structure.lattice_vectors_ang,
00308 |         system_label=sample_id,
00309 |         system_name=f"{validated.bundle.label} {sample_id}",
00310 |         structure_type=validated.bundle.structure_type,
00311 |     )
00312 |     materialized_metadata = dict(materialized.metadata)
00313 |     materialized_metadata["structure_species"] = materialized_metadata.pop("species", [])
00314 |     metadata = {**materialized_metadata, **metadata}
00315 |     metadata["pseudopotentials_copied"] = copied_pseudos
00316 |     _write_json(sample_dir / "metadata.json", metadata)
00317 |     return {
00318 |         "id": sample_id,
00319 |         "sample_id": sample_id,
00320 |         "sample_dir": str(sample_dir),
00321 |         "run_fdf": str(sample_dir / "RUN.fdf"),
00322 |         "metadata_path": str(sample_dir / "metadata.json"),
00323 |         "is_reference": metadata["is_reference"],
00324 |         "atom_index": metadata["atom_index"],
00325 |         "atom_index_zero_based": metadata["atom_index_zero_based"],
00326 |         "species": metadata["species"],
00327 |         "axis": metadata["axis"],
00328 |         "axis_index": metadata["axis_index"],
00329 |         "sign": metadata["sign"],
00330 |         "sign_label": metadata["sign_label"],
00331 |         "amplitude_ang": metadata["amplitude_ang"],
00332 |         "displacement_ang": metadata["displacement_ang"],
00333 |         "split_group_id": metadata["split_group_id"],
00334 |         "materialized_fdf_sha256": metadata["materialized_fdf_sha256"],
00335 |     }
00336 | 
00337 | 
00338 | def generate_dataset(
00339 |     config: dict[str, Any],
00340 |     *,
00341 |     output_dir: str | Path | None = None,
00342 |     base_dir: str | Path = REPO_ROOT,
00343 |     overwrite: bool | None = None,
00344 | ) -> dict[str, Any]:
00345 |     """Generate generic +/- Cartesian displacement samples from a material FDF."""
00346 | 
00347 |     settings = generic_cartesian_settings(config)
00348 |     if overwrite is not None:
00349 |         settings = GenericCartesianSettings(
00350 |             recipe=settings.recipe,
00351 |             amplitude_ang=settings.amplitude_ang,
00352 |             selected_species=settings.selected_species,
00353 |             include_base=settings.include_base,
00354 |             overwrite=bool(overwrite),
00355 |         )
00356 |     if output_dir is None:
00357 |         pipeline_paths = paths(config)
00358 |         output_root = pipeline_paths["samples_dir"]
00359 |         manifest_path = pipeline_paths["samples_manifest_path"]
00360 |     else:
00361 |         output_root = Path(output_dir)
00362 |         manifest_path = output_root.parent / "samples_manifest.json"
00363 | 
00364 |     resolved = resolve_material_bundle(config, base_dir=base_dir)
00365 |     validated = resolved.validated
00366 |     structure = extract_bundle_structure(validated)
00367 |     selected_atoms, skipped_atoms = _selected_atom_records(structure, settings.selected_species)
00368 | 
00369 |     _prepare_output_root(output_root, overwrite=settings.overwrite)
00370 |     copied_basis_hashes = _copy_basis_files(validated, output_root)
00371 | 
00372 |     material_label = validated.bundle.label
00373 |     sample_records: list[dict[str, Any]] = []
00374 |     sample_index = 0
00375 |     used_ids: set[str] = set()
00376 | 
00377 |     def next_sample_id() -> str:
00378 |         nonlocal sample_index
00379 |         sample_id = _sample_id(config, sample_index)
00380 |         sample_index += 1
00381 |         if sample_id in used_ids:
00382 |             raise GenericCartesianDisplacementError(f"Duplicate generated sample id: {sample_id}")
00383 |         used_ids.add(sample_id)
00384 |         return sample_id
00385 | 
00386 |     if settings.include_base:
00387 |         sample_id = next_sample_id()
00388 |         materialized_preview = {
00389 |             "structure_type": validated.bundle.structure_type,
00390 |         }
00391 |         metadata = _base_metadata(
00392 |             sample_id=sample_id,
00393 |             material_label=material_label,
00394 |             settings=settings,
00395 |             materialized_metadata=materialized_preview,
00396 |         )
00397 |         sample_records.append(
00398 |             _write_sample(
00399 |                 sample_dir=output_root / sample_id,
00400 |                 sample_id=sample_id,
00401 |                 positions_ang=structure.positions_ang,
00402 |                 validated=validated,
00403 |                 structure=structure,
00404 |                 metadata=metadata,
00405 |             )
00406 |         )
00407 | 
00408 |     for atom_record in selected_atoms:
00409 |         for axis, axis_index in AXES:
00410 |             for sign in SIGNS:
00411 |                 sample_id = next_sample_id()
00412 |                 delta = sign * settings.amplitude_ang
00413 |                 positions = _with_displacement(
00414 |                     structure.positions_ang,
00415 |                     atom_index_zero_based=atom_record["atom_index_zero_based"],
00416 |                     axis_index=axis_index,
00417 |                     delta_ang=delta,
00418 |                 )
00419 |                 metadata = _displacement_metadata(
00420 |                     sample_id=sample_id,
00421 |                     material_label=material_label,
00422 |                     settings=settings,
00423 |                     atom_record=atom_record,
00424 |                     axis=axis,
00425 |                     axis_index=axis_index,
00426 |                     sign=sign,
00427 |                     materialized_metadata={"structure_type": validated.bundle.structure_type},
00428 |                 )
00429 |                 sample_records.append(
00430 |                     _write_sample(
00431 |                         sample_dir=output_root / sample_id,
00432 |                         sample_id=sample_id,
00433 |                         positions_ang=positions,
00434 |                         validated=validated,
00435 |                         structure=structure,
00436 |                         metadata=metadata,
00437 |                     )
00438 |                 )
00439 | 
00440 |     manifest = {
00441 |         "generation_mode": GENERATION_MODE,
00442 |         "generation_method": "generic_cartesian",
00443 |         "method": METHOD_ID,
00444 |         "recipe": settings.recipe,
00445 |         "material": resolved.to_manifest_dict(),
00446 |         "reference_source": str(validated.bundle.fdf),
00447 |         "sample_root": str(output_root),
00448 |         "requested_structures": len(sample_records),
00449 |         "generated_structures": len(sample_records),
00450 |         "include_base": settings.include_base,
00451 |         "amplitude_ang": settings.amplitude_ang,
00452 |         "selected_species": sorted(settings.selected_species) if settings.selected_species else None,
00453 |         "selected_atoms": selected_atoms,
00454 |         "skipped_atoms": skipped_atoms,
00455 |         "axis_order": [axis for axis, _axis_index in AXES],
00456 |         "sign_order": list(SIGNS),
00457 |         "basis_file_sha256": copied_basis_hashes,
00458 |         "samples": sample_records,
00459 |     }
00460 |     _write_json(manifest_path, manifest)
00461 |     _write_json(output_root / "dataset_manifest.json", manifest)
00462 |     return manifest
00463 | 
00464 | 
00465 | def build_argument_parser() -> argparse.ArgumentParser:
00466 |     parser = argparse.ArgumentParser(
00467 |         description="Generate generic Cartesian AtomicDisplacement samples from a material bundle."
00468 |     )
00469 |     parser.add_argument("--config", type=Path, default=None, help="Pipeline YAML config path.")
00470 |     parser.add_argument("--output-dir", type=Path, default=None, help="Output sample directory.")
00471 |     parser.add_argument("--material-base-dir", type=Path, default=REPO_ROOT)
00472 |     parser.add_argument("--amplitude-ang", type=float, default=None)
00473 |     parser.add_argument("--selected-species", nargs="*", default=None)
00474 |     parser.add_argument("--include-base", action=argparse.BooleanOptionalAction, default=None)
00475 |     parser.add_argument("--overwrite", action="store_true")
00476 |     return parser
00477 | 
00478 | 
00479 | def main() -> int:
00480 |     args = build_argument_parser().parse_args()
00481 |     config = load_pipeline_config(args.config)
00482 |     recipe = dict(_recipe_config(config))
00483 |     if args.amplitude_ang is not None:
00484 |         recipe["amplitude_ang"] = args.amplitude_ang
00485 |     if args.selected_species is not None:
00486 |         recipe["selected_species"] = args.selected_species
00487 |     if args.include_base is not None:
00488 |         recipe["include_base"] = args.include_base
00489 |     if args.overwrite:
00490 |         recipe["overwrite"] = True
00491 |     config["atomic_displacement"] = recipe
00492 | 
00493 |     manifest = generate_dataset(
00494 |         config,
00495 |         output_dir=args.output_dir,
00496 |         base_dir=args.material_base_dir,
00497 |         overwrite=args.overwrite or None,
00498 |     )
00499 |     print(
00500 |         "[OK] Generic Cartesian AtomicDisplacement samples generated: "
00501 |         f"{manifest['generated_structures']} in {manifest['sample_root']}"
00502 |     )
00503 |     return 0
00504 | 
00505 | 
00506 | if __name__ == "__main__":
00507 |     raise SystemExit(main())
```

## `shared/material_bundle.py`

SHA-256: `5d0bc5ee4d5a7947842b1295f44f4cb332cfd24fef8ddf9dd02d3722b4aa06f0`

```py
00001 | """Validation helpers for user-provided SIESTA material input bundles.
00002 | 
00003 | The bundle validator is intentionally small: it checks paths, extracts species
00004 | from the ``ChemicalSpeciesLabel`` block, verifies pseudopotential coverage, and
00005 | returns hashes/provenance. It does not try to be a complete FDF parser.
00006 | """
00007 | 
00008 | from __future__ import annotations
00009 | 
00010 | import hashlib
00011 | import json
00012 | import re
00013 | from dataclasses import dataclass
00014 | from pathlib import Path
00015 | from typing import Any
00016 | 
00017 | 
00018 | PSEUDOPOTENTIAL_EXTENSIONS = (".psf", ".psml")
00019 | BASIS_EXTENSIONS = (".ion.xml", ".ion")
00020 | LABEL_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
00021 | 
00022 | 
00023 | class MaterialBundleError(RuntimeError):
00024 |     """Raised when a material bundle is incomplete or inconsistent."""
00025 | 
00026 | 
00027 | @dataclass(frozen=True)
00028 | class MaterialSpecies:
00029 |     index: int
00030 |     atomic_number: int
00031 |     label: str
00032 | 
00033 |     def to_dict(self) -> dict[str, Any]:
00034 |         return {
00035 |             "index": self.index,
00036 |             "atomic_number": self.atomic_number,
00037 |             "label": self.label,
00038 |         }
00039 | 
00040 | 
00041 | @dataclass(frozen=True)
00042 | class MaterialBundle:
00043 |     label: str
00044 |     fdf: Path
00045 |     pseudopotential_dir: Path
00046 |     basis_dir: Path | None = None
00047 |     structure_type: str | None = None
00048 |     source_paths_absolute: bool = False
00049 | 
00050 | 
00051 | @dataclass(frozen=True)
00052 | class ValidatedMaterialBundle:
00053 |     bundle: MaterialBundle
00054 |     species: list[MaterialSpecies]
00055 |     pseudopotentials: dict[str, Path]
00056 |     fdf_sha256: str
00057 |     pseudopotential_sha256: dict[str, str]
00058 |     basis_file_sha256: dict[str, str]
00059 |     absolute_paths_used: bool
00060 | 
00061 |     def to_manifest_dict(self) -> dict[str, Any]:
00062 |         return {
00063 |             "label": self.bundle.label,
00064 |             "structure_type": self.bundle.structure_type,
00065 |             "fdf": str(self.bundle.fdf),
00066 |             "fdf_sha256": self.fdf_sha256,
00067 |             "pseudopotential_dir": str(self.bundle.pseudopotential_dir),
00068 |             "basis_dir": str(self.bundle.basis_dir) if self.bundle.basis_dir else None,
00069 |             "species": [species.to_dict() for species in self.species],
00070 |             "pseudopotentials": {
00071 |                 label: str(path)
00072 |                 for label, path in sorted(self.pseudopotentials.items())
00073 |             },
00074 |             "pseudopotential_sha256": dict(sorted(self.pseudopotential_sha256.items())),
00075 |             "basis_file_sha256": dict(sorted(self.basis_file_sha256.items())),
00076 |             "absolute_paths_used": self.absolute_paths_used,
00077 |         }
00078 | 
00079 | 
00080 | def file_sha256(path: Path) -> str:
00081 |     digest = hashlib.sha256()
00082 |     with path.open("rb") as handle:
00083 |         for chunk in iter(lambda: handle.read(1024 * 1024), b""):
00084 |             digest.update(chunk)
00085 |     return digest.hexdigest()
00086 | 
00087 | 
00088 | def _is_relative_to(path: Path, root: Path) -> bool:
00089 |     try:
00090 |         path.relative_to(root)
00091 |     except ValueError:
00092 |         return False
00093 |     return True
00094 | 
00095 | 
00096 | def _resolve_bundle_path(value: str | Path, root_dir: Path, *, allow_absolute: bool = True) -> tuple[Path, bool]:
00097 |     if value in (None, ""):
00098 |         raise MaterialBundleError("Material bundle path cannot be empty.")
00099 |     raw_path = Path(value).expanduser()
00100 |     if raw_path.is_absolute():
00101 |         if not allow_absolute:
00102 |             raise MaterialBundleError(f"Absolute material bundle paths are not allowed: {raw_path}")
00103 |         return raw_path.resolve(), True
00104 | 
00105 |     root = root_dir.expanduser().resolve()
00106 |     resolved = (root / raw_path).resolve()
00107 |     if not _is_relative_to(resolved, root):
00108 |         raise MaterialBundleError(
00109 |             f"Material bundle path escapes its root: {value!r} resolved under {root}"
00110 |         )
00111 |     return resolved, False
00112 | 
00113 | 
00114 | def _validate_label(label: Any) -> str:
00115 |     text = str(label or "").strip()
00116 |     if not text:
00117 |         raise MaterialBundleError("material.label must be non-empty.")
00118 |     if not LABEL_PATTERN.fullmatch(text):
00119 |         raise MaterialBundleError(
00120 |             "material.label must contain only letters, numbers, '.', '_' or '-' "
00121 |             "and must start with a letter or number."
00122 |         )
00123 |     return text
00124 | 
00125 | 
00126 | def material_bundle_from_config(
00127 |     config: dict[str, Any],
00128 |     *,
00129 |     base_dir: str | Path,
00130 |     allow_absolute_paths: bool = True,
00131 | ) -> MaterialBundle:
00132 |     raw_material = config.get("material", config)
00133 |     if not isinstance(raw_material, dict):
00134 |         raise MaterialBundleError("material config must be a mapping.")
00135 | 
00136 |     label = _validate_label(raw_material.get("label"))
00137 |     root_dir = Path(base_dir)
00138 |     source_paths_absolute = False
00139 |     if raw_material.get("root_dir") not in (None, ""):
00140 |         root_dir, root_absolute = _resolve_bundle_path(
00141 |             raw_material["root_dir"],
00142 |             Path(base_dir),
00143 |             allow_absolute=allow_absolute_paths,
00144 |         )
00145 |         source_paths_absolute = source_paths_absolute or root_absolute
00146 | 
00147 |     fdf, fdf_absolute = _resolve_bundle_path(
00148 |         raw_material.get("fdf", ""),
00149 |         root_dir,
00150 |         allow_absolute=allow_absolute_paths,
00151 |     )
00152 |     source_paths_absolute = source_paths_absolute or fdf_absolute
00153 |     pseudo_dir, pseudo_absolute = _resolve_bundle_path(
00154 |         raw_material.get("pseudopotential_dir", ""),
00155 |         root_dir,
00156 |         allow_absolute=allow_absolute_paths,
00157 |     )
00158 |     source_paths_absolute = source_paths_absolute or pseudo_absolute
00159 |     basis_dir = None
00160 |     if raw_material.get("basis_dir") not in (None, ""):
00161 |         basis_dir, basis_absolute = _resolve_bundle_path(
00162 |             raw_material["basis_dir"],
00163 |             root_dir,
00164 |             allow_absolute=allow_absolute_paths,
00165 |         )
00166 |         source_paths_absolute = source_paths_absolute or basis_absolute
00167 | 
00168 |     structure_type = raw_material.get("structure_type")
00169 |     structure_type = str(structure_type).strip() if structure_type not in (None, "") else None
00170 |     return MaterialBundle(
00171 |         label=label,
00172 |         fdf=fdf,
00173 |         pseudopotential_dir=pseudo_dir,
00174 |         basis_dir=basis_dir,
00175 |         structure_type=structure_type,
00176 |         source_paths_absolute=source_paths_absolute,
00177 |     )
00178 | 
00179 | 
00180 | def _strip_fdf_comment(line: str) -> str:
00181 |     return line.split("#", 1)[0].strip()
00182 | 
00183 | 
00184 | def read_fdf_block(fdf_path: Path, block_name: str) -> list[str]:
00185 |     lower_name = block_name.lower()
00186 |     lines = fdf_path.read_text(encoding="utf-8", errors="ignore").splitlines()
00187 |     start: int | None = None
00188 |     for index, raw_line in enumerate(lines):
00189 |         clean = _strip_fdf_comment(raw_line).lower()
00190 |         if clean == f"%block {lower_name}":
00191 |             start = index + 1
00192 |             break
00193 |     if start is None:
00194 |         return []
00195 | 
00196 |     block: list[str] = []
00197 |     end_marker = f"%endblock {lower_name}"
00198 |     for raw_line in lines[start:]:
00199 |         clean = _strip_fdf_comment(raw_line)
00200 |         if clean.lower() == end_marker:
00201 |             return block
00202 |         if clean:
00203 |             block.append(clean)
00204 |     raise MaterialBundleError(f"FDF block {block_name!r} is not closed in {fdf_path}.")
00205 | 
00206 | 
00207 | def extract_chemical_species(fdf_path: Path) -> list[MaterialSpecies]:
00208 |     rows = read_fdf_block(fdf_path, "ChemicalSpeciesLabel")
00209 |     if not rows:
00210 |         raise MaterialBundleError(
00211 |             f"{fdf_path} does not define a ChemicalSpeciesLabel block; "
00212 |             "cannot validate pseudopotential coverage."
00213 |         )
00214 | 
00215 |     species: list[MaterialSpecies] = []
00216 |     seen_indices: set[int] = set()
00217 |     for line in rows:
00218 |         parts = line.split()
00219 |         if len(parts) < 3:
00220 |             raise MaterialBundleError(f"Invalid ChemicalSpeciesLabel row in {fdf_path}: {line!r}")
00221 |         try:
00222 |             index = int(parts[0])
00223 |             atomic_number = int(parts[1])
00224 |         except ValueError as exc:
00225 |             raise MaterialBundleError(
00226 |                 f"Invalid ChemicalSpeciesLabel numeric fields in {fdf_path}: {line!r}"
00227 |             ) from exc
00228 |         label = str(parts[2]).strip()
00229 |         if not label:
00230 |             raise MaterialBundleError(f"Empty species label in {fdf_path}: {line!r}")
00231 |         if index in seen_indices:
00232 |             raise MaterialBundleError(f"Duplicate species index {index} in {fdf_path}.")
00233 |         seen_indices.add(index)
00234 |         species.append(MaterialSpecies(index=index, atomic_number=atomic_number, label=label))
00235 |     return species
00236 | 
00237 | 
00238 | def extract_coordinate_species_indices(fdf_path: Path) -> list[int]:
00239 |     rows = read_fdf_block(fdf_path, "AtomicCoordinatesAndAtomicSpecies")
00240 |     indices: list[int] = []
00241 |     for line in rows:
00242 |         parts = line.split()
00243 |         if len(parts) < 4:
00244 |             raise MaterialBundleError(
00245 |                 f"Invalid AtomicCoordinatesAndAtomicSpecies row in {fdf_path}: {line!r}"
00246 |             )
00247 |         try:
00248 |             indices.append(int(parts[3]))
00249 |         except ValueError as exc:
00250 |             raise MaterialBundleError(
00251 |                 f"Invalid atomic species index in {fdf_path}: {line!r}"
00252 |             ) from exc
00253 |     return indices
00254 | 
00255 | 
00256 | def validate_fdf_species_consistency(fdf_path: Path, species: list[MaterialSpecies]) -> None:
00257 |     declared = {item.index for item in species}
00258 |     used = set(extract_coordinate_species_indices(fdf_path))
00259 |     missing = sorted(used - declared)
00260 |     if missing:
00261 |         raise MaterialBundleError(
00262 |             f"{fdf_path} uses undeclared species indices in AtomicCoordinatesAndAtomicSpecies: {missing}"
00263 |         )
00264 | 
00265 | 
00266 | def resolve_pseudopotentials(
00267 |     pseudopotential_dir: Path,
00268 |     species: list[MaterialSpecies],
00269 | ) -> dict[str, Path]:
00270 |     resolved: dict[str, Path] = {}
00271 |     for item in species:
00272 |         candidates = [
00273 |             pseudopotential_dir / f"{item.label}{extension}"
00274 |             for extension in PSEUDOPOTENTIAL_EXTENSIONS
00275 |             if (pseudopotential_dir / f"{item.label}{extension}").is_file()
00276 |         ]
00277 |         if item.atomic_number < 0 and not candidates:
00278 |             continue
00279 |         if not candidates:
00280 |             extensions = ", ".join(PSEUDOPOTENTIAL_EXTENSIONS)
00281 |             raise MaterialBundleError(
00282 |                 f"Missing pseudopotential for species {item.label!r} in "
00283 |                 f"{pseudopotential_dir}; expected {item.label} with one of: {extensions}."
00284 |             )
00285 |         if len(candidates) > 1:
00286 |             names = ", ".join(path.name for path in candidates)
00287 |             raise MaterialBundleError(
00288 |                 f"Ambiguous pseudopotential for species {item.label!r} in "
00289 |                 f"{pseudopotential_dir}: {names}"
00290 |             )
00291 |         resolved[item.label] = candidates[0].resolve()
00292 |     return resolved
00293 | 
00294 | 
00295 | def _basis_hashes(basis_dir: Path | None) -> dict[str, str]:
00296 |     if basis_dir is None:
00297 |         return {}
00298 |     hashes: dict[str, str] = {}
00299 |     for path in sorted(path for path in basis_dir.iterdir() if path.is_file()):
00300 |         if any(path.name.endswith(extension) for extension in BASIS_EXTENSIONS):
00301 |             hashes[path.name] = file_sha256(path)
00302 |     return hashes
00303 | 
00304 | 
00305 | def validate_material_bundle(bundle: MaterialBundle) -> ValidatedMaterialBundle:
00306 |     if not bundle.fdf.is_file():
00307 |         raise MaterialBundleError(f"Material FDF does not exist or is not a file: {bundle.fdf}")
00308 |     if not bundle.pseudopotential_dir.is_dir():
00309 |         raise MaterialBundleError(
00310 |             f"Material pseudopotential directory does not exist: {bundle.pseudopotential_dir}"
00311 |         )
00312 |     if bundle.basis_dir is not None and not bundle.basis_dir.is_dir():
00313 |         raise MaterialBundleError(f"Material basis directory does not exist: {bundle.basis_dir}")
00314 | 
00315 |     species = extract_chemical_species(bundle.fdf)
00316 |     validate_fdf_species_consistency(bundle.fdf, species)
00317 |     pseudos = resolve_pseudopotentials(bundle.pseudopotential_dir, species)
00318 |     pseudo_hashes = {
00319 |         label: file_sha256(path)
00320 |         for label, path in sorted(pseudos.items())
00321 |     }
00322 |     return ValidatedMaterialBundle(
00323 |         bundle=bundle,
00324 |         species=species,
00325 |         pseudopotentials=pseudos,
00326 |         fdf_sha256=file_sha256(bundle.fdf),
00327 |         pseudopotential_sha256=pseudo_hashes,
00328 |         basis_file_sha256=_basis_hashes(bundle.basis_dir),
00329 |         absolute_paths_used=bundle.source_paths_absolute,
00330 |     )
00331 | 
00332 | 
00333 | def validate_material_config(
00334 |     config: dict[str, Any],
00335 |     *,
00336 |     base_dir: str | Path,
00337 |     allow_absolute_paths: bool = True,
00338 | ) -> ValidatedMaterialBundle:
00339 |     bundle = material_bundle_from_config(
00340 |         config,
00341 |         base_dir=base_dir,
00342 |         allow_absolute_paths=allow_absolute_paths,
00343 |     )
00344 |     return validate_material_bundle(bundle)
00345 | 
00346 | 
00347 | def manifest_json(validated: ValidatedMaterialBundle) -> str:
00348 |     return json.dumps(
00349 |         validated.to_manifest_dict(),
00350 |         indent=2,
00351 |         ensure_ascii=False,
00352 |         sort_keys=True,
00353 |     )
```

## `shared/fdf_materialization.py`

SHA-256: `bbb56d392cbbf6d879269f19fe75991ae596e3a8aae2f47fd53d4e86117925dd`

```py
00001 | """Minimal SIESTA FDF extraction and per-sample materialization utilities."""
00002 | 
00003 | from __future__ import annotations
00004 | 
00005 | from dataclasses import dataclass
00006 | from pathlib import Path
00007 | from typing import Any
00008 | 
00009 | from material_bundle import (
00010 |     MaterialSpecies,
00011 |     ValidatedMaterialBundle,
00012 |     extract_chemical_species,
00013 |     file_sha256,
00014 |     read_fdf_block,
00015 | )
00016 | 
00017 | 
00018 | SUPPORTED_COORDINATE_FORMATS = {"ang", "angstrom", "angstroms"}
00019 | DEFAULT_REQUIRED_OUTPUT_FLAGS = {
00020 |     "SaveHS": "true",
00021 |     "Save.HS": "T",
00022 |     "TS.HS.Save": "T",
00023 |     "TS.DE.Save": "T",
00024 |     "XML.Write": "T",
00025 |     "Write.OrbitalIndex": "T",
00026 | }
00027 | BOHR_TO_ANG = 0.529177210903
00028 | 
00029 | 
00030 | class FdfMaterializationError(RuntimeError):
00031 |     """Raised when an FDF cannot be safely materialized."""
00032 | 
00033 | 
00034 | @dataclass(frozen=True)
00035 | class FdfAtom:
00036 |     position_ang: tuple[float, float, float]
00037 |     species_index: int
00038 | 
00039 |     def to_dict(self) -> dict[str, Any]:
00040 |         return {
00041 |             "position_ang": list(self.position_ang),
00042 |             "species_index": self.species_index,
00043 |         }
00044 | 
00045 | 
00046 | @dataclass(frozen=True)
00047 | class FdfStructure:
00048 |     fdf_path: Path
00049 |     species: list[MaterialSpecies]
00050 |     atoms: list[FdfAtom]
00051 |     lattice_vectors_ang: list[tuple[float, float, float]]
00052 |     coordinate_format: str
00053 |     number_of_atoms_declared: int | None
00054 |     structure_type: str | None = None
00055 | 
00056 |     @property
00057 |     def atom_count(self) -> int:
00058 |         return len(self.atoms)
00059 | 
00060 |     @property
00061 |     def atom_species(self) -> list[int]:
00062 |         return [atom.species_index for atom in self.atoms]
00063 | 
00064 |     @property
00065 |     def positions_ang(self) -> list[tuple[float, float, float]]:
00066 |         return [atom.position_ang for atom in self.atoms]
00067 | 
00068 |     def to_manifest_dict(self) -> dict[str, Any]:
00069 |         return {
00070 |             "fdf": str(self.fdf_path),
00071 |             "structure_type": self.structure_type,
00072 |             "coordinate_format": self.coordinate_format,
00073 |             "atom_count": self.atom_count,
00074 |             "number_of_atoms_declared": self.number_of_atoms_declared,
00075 |             "species": [species.to_dict() for species in self.species],
00076 |             "lattice_vectors_ang": [list(vector) for vector in self.lattice_vectors_ang],
00077 |         }
00078 | 
00079 | 
00080 | @dataclass(frozen=True)
00081 | class MaterializedFdf:
00082 |     path: Path
00083 |     metadata: dict[str, Any]
00084 | 
00085 | 
00086 | def _strip_comment(line: str) -> str:
00087 |     return line.split("#", 1)[0].strip()
00088 | 
00089 | 
00090 | def _first_directive_value(text: str, key: str) -> str | None:
00091 |     lower_key = key.lower()
00092 |     for raw_line in text.splitlines():
00093 |         clean = _strip_comment(raw_line)
00094 |         if not clean:
00095 |             continue
00096 |         parts = clean.split(None, 1)
00097 |         if parts and parts[0].lower() == lower_key:
00098 |             return parts[1].strip() if len(parts) > 1 else ""
00099 |     return None
00100 | 
00101 | 
00102 | def _parse_optional_int_directive(text: str, key: str, path: Path) -> int | None:
00103 |     value = _first_directive_value(text, key)
00104 |     if value in (None, ""):
00105 |         return None
00106 |     try:
00107 |         return int(value.split()[0])
00108 |     except ValueError as exc:
00109 |         raise FdfMaterializationError(f"{path}: invalid integer directive {key}: {value!r}") from exc
00110 | 
00111 | 
00112 | def _coordinate_format(text: str, path: Path) -> str:
00113 |     value = _first_directive_value(text, "AtomicCoordinatesFormat")
00114 |     if value in (None, ""):
00115 |         raise FdfMaterializationError(
00116 |             f"{path}: missing AtomicCoordinatesFormat; only explicit Ang coordinates are supported."
00117 |         )
00118 |     token = value.split()[0].strip().lower()
00119 |     if token not in SUPPORTED_COORDINATE_FORMATS:
00120 |         raise FdfMaterializationError(
00121 |             f"{path}: unsupported AtomicCoordinatesFormat {value!r}; only Ang is supported."
00122 |         )
00123 |     return token
00124 | 
00125 | 
00126 | def _parse_float_triplet(parts: list[str], path: Path, row: str) -> tuple[float, float, float]:
00127 |     if len(parts) < 3:
00128 |         raise FdfMaterializationError(f"{path}: invalid coordinate/lattice row: {row!r}")
00129 |     try:
00130 |         return float(parts[0]), float(parts[1]), float(parts[2])
00131 |     except ValueError as exc:
00132 |         raise FdfMaterializationError(f"{path}: invalid numeric row: {row!r}") from exc
00133 | 
00134 | 
00135 | def _lattice_constant_scale_ang(text: str, path: Path) -> float:
00136 |     value = _first_directive_value(text, "LatticeConstant")
00137 |     if value in (None, ""):
00138 |         return 1.0
00139 |     parts = value.split()
00140 |     try:
00141 |         magnitude = float(parts[0])
00142 |     except (IndexError, ValueError) as exc:
00143 |         raise FdfMaterializationError(f"{path}: invalid LatticeConstant: {value!r}") from exc
00144 |     unit = parts[1].lower() if len(parts) > 1 else "ang"
00145 |     if unit in {"ang", "angstrom", "angstroms"}:
00146 |         return magnitude
00147 |     if unit in {"bohr", "bohrs"}:
00148 |         return magnitude * BOHR_TO_ANG
00149 |     raise FdfMaterializationError(
00150 |         f"{path}: unsupported LatticeConstant unit {unit!r}; only Ang and Bohr are supported."
00151 |     )
00152 | 
00153 | 
00154 | def _parse_lattice_vectors(fdf_path: Path, text: str) -> list[tuple[float, float, float]]:
00155 |     scale = _lattice_constant_scale_ang(text, fdf_path)
00156 |     vectors = []
00157 |     for row in read_fdf_block(fdf_path, "LatticeVectors"):
00158 |         vector = _parse_float_triplet(row.split(), fdf_path, row)
00159 |         vectors.append(tuple(component * scale for component in vector))
00160 |     return vectors
00161 | 
00162 | 
00163 | def _parse_atoms(fdf_path: Path, declared_species: set[int]) -> list[FdfAtom]:
00164 |     rows = read_fdf_block(fdf_path, "AtomicCoordinatesAndAtomicSpecies")
00165 |     if not rows:
00166 |         raise FdfMaterializationError(
00167 |             f"{fdf_path}: missing AtomicCoordinatesAndAtomicSpecies block."
00168 |         )
00169 |     atoms = []
00170 |     for row in rows:
00171 |         parts = row.split()
00172 |         if len(parts) < 4:
00173 |             raise FdfMaterializationError(
00174 |                 f"{fdf_path}: invalid AtomicCoordinatesAndAtomicSpecies row: {row!r}"
00175 |             )
00176 |         position = _parse_float_triplet(parts, fdf_path, row)
00177 |         try:
00178 |             species_index = int(parts[3])
00179 |         except ValueError as exc:
00180 |             raise FdfMaterializationError(
00181 |                 f"{fdf_path}: invalid atomic species index in row: {row!r}"
00182 |             ) from exc
00183 |         if species_index not in declared_species:
00184 |             raise FdfMaterializationError(
00185 |                 f"{fdf_path}: atom row uses undeclared species index {species_index}."
00186 |             )
00187 |         atoms.append(FdfAtom(position_ang=position, species_index=species_index))
00188 |     return atoms
00189 | 
00190 | 
00191 | def extract_fdf_structure(fdf_path: Path, *, structure_type: str | None = None) -> FdfStructure:
00192 |     if not fdf_path.is_file():
00193 |         raise FdfMaterializationError(f"FDF does not exist or is not a file: {fdf_path}")
00194 |     text = fdf_path.read_text(encoding="utf-8", errors="ignore")
00195 |     coordinate_format = _coordinate_format(text, fdf_path)
00196 |     try:
00197 |         species = extract_chemical_species(fdf_path)
00198 |     except RuntimeError as exc:
00199 |         raise FdfMaterializationError(str(exc)) from exc
00200 |     declared_species = {item.index for item in species}
00201 |     atoms = _parse_atoms(fdf_path, declared_species)
00202 |     declared_atom_count = _parse_optional_int_directive(text, "NumberOfAtoms", fdf_path)
00203 |     if declared_atom_count is not None and declared_atom_count != len(atoms):
00204 |         raise FdfMaterializationError(
00205 |             f"{fdf_path}: NumberOfAtoms={declared_atom_count} does not match "
00206 |             f"AtomicCoordinatesAndAtomicSpecies rows={len(atoms)}."
00207 |         )
00208 |     return FdfStructure(
00209 |         fdf_path=fdf_path,
00210 |         species=species,
00211 |         atoms=atoms,
00212 |         lattice_vectors_ang=_parse_lattice_vectors(fdf_path, text),
00213 |         coordinate_format=coordinate_format,
00214 |         number_of_atoms_declared=declared_atom_count,
00215 |         structure_type=structure_type,
00216 |     )
00217 | 
00218 | 
00219 | def extract_bundle_structure(validated: ValidatedMaterialBundle) -> FdfStructure:
00220 |     return extract_fdf_structure(
00221 |         validated.bundle.fdf,
00222 |         structure_type=validated.bundle.structure_type,
00223 |     )
00224 | 
00225 | 
00226 | def _format_float(value: float) -> str:
00227 |     return f"{float(value):.12f}"
00228 | 
00229 | 
00230 | def _replace_or_append_block(text: str, block_name: str, block_lines: list[str]) -> str:
00231 |     lines = text.splitlines()
00232 |     lower_name = block_name.lower()
00233 |     output: list[str] = []
00234 |     index = 0
00235 |     replaced = False
00236 |     while index < len(lines):
00237 |         clean = _strip_comment(lines[index]).lower()
00238 |         if clean == f"%block {lower_name}":
00239 |             output.append(f"%block {block_name}")
00240 |             output.extend(block_lines)
00241 |             output.append(f"%endblock {block_name}")
00242 |             index += 1
00243 |             while index < len(lines):
00244 |                 end_clean = _strip_comment(lines[index]).lower()
00245 |                 index += 1
00246 |                 if end_clean == f"%endblock {lower_name}":
00247 |                     break
00248 |             replaced = True
00249 |             continue
00250 |         output.append(lines[index])
00251 |         index += 1
00252 |     if not replaced:
00253 |         if output and output[-1].strip():
00254 |             output.append("")
00255 |         output.append(f"%block {block_name}")
00256 |         output.extend(block_lines)
00257 |         output.append(f"%endblock {block_name}")
00258 |     return "\n".join(output).rstrip() + "\n"
00259 | 
00260 | 
00261 | def _set_fdf_directive(text: str, key: str, value: str) -> str:
00262 |     lines = text.splitlines()
00263 |     output: list[str] = []
00264 |     inserted = False
00265 |     lower_key = key.lower()
00266 |     for line in lines:
00267 |         clean = _strip_comment(line)
00268 |         first = clean.split(None, 1)[0].lower() if clean else ""
00269 |         if first == lower_key:
00270 |             if not inserted:
00271 |                 output.append(f"{key:<32} {value}")
00272 |                 inserted = True
00273 |             continue
00274 |         output.append(line)
00275 |     if not inserted:
00276 |         if output and output[-1].strip():
00277 |             output.append("")
00278 |         output.append(f"{key:<32} {value}")
00279 |     return "\n".join(output).rstrip() + "\n"
00280 | 
00281 | 
00282 | def ensure_required_output_flags(
00283 |     text: str,
00284 |     required_flags: dict[str, str] | None = None,
00285 | ) -> str:
00286 |     updated = text
00287 |     for key, value in (required_flags or DEFAULT_REQUIRED_OUTPUT_FLAGS).items():
00288 |         updated = _set_fdf_directive(updated, key, value)
00289 |     return updated
00290 | 
00291 | 
00292 | def _normalized_positions(positions_ang: list[list[float]] | list[tuple[float, float, float]]) -> list[tuple[float, float, float]]:
00293 |     positions = []
00294 |     for position in positions_ang:
00295 |         if len(position) != 3:
00296 |             raise FdfMaterializationError(f"Each position must contain exactly three values: {position!r}")
00297 |         positions.append((float(position[0]), float(position[1]), float(position[2])))
00298 |     return positions
00299 | 
00300 | 
00301 | def _coordinate_block_lines(
00302 |     positions: list[tuple[float, float, float]],
00303 |     atom_species: list[int],
00304 |     species_by_index: dict[int, MaterialSpecies],
00305 | ) -> list[str]:
00306 |     lines = []
00307 |     for position, species_index in zip(positions, atom_species):
00308 |         species = species_by_index[int(species_index)]
00309 |         lines.append(
00310 |             f" {_format_float(position[0])}  {_format_float(position[1])}  "
00311 |             f"{_format_float(position[2])}  {int(species_index)}  # {species.label}"
00312 |         )
00313 |     return lines
00314 | 
00315 | 
00316 | _SINGLE_POINT_STRIPPED_KEYS = {"lua.script", "writemdhistory"}
00317 | 
00318 | 
00319 | def strip_to_single_point(text: str) -> str:
00320 |     """Drop MD/Lua directives so SIESTA runs one SCF at the given geometry.
00321 | 
00322 |     Derivative stencils need the Hamiltonian at exactly the written positions.
00323 |     A base fdf inherited from an MD dataset carries ``MD.TypeOfRun Verlet`` and
00324 |     ``Lua.Script``: SIESTA then evolves the structure for MD.Steps before the
00325 |     stored TSHS, silently invalidating the finite-difference reference.
00326 |     """
00327 |     lines: list[str] = []
00328 |     for line in text.splitlines():
00329 |         clean = _strip_comment(line)
00330 |         first = clean.split(None, 1)[0].lower() if clean else ""
00331 |         if first.startswith("md.") or first in _SINGLE_POINT_STRIPPED_KEYS:
00332 |             continue
00333 |         lines.append(line)
00334 |     text = "\n".join(lines)
00335 |     if not text.endswith("\n"):
00336 |         text += "\n"
00337 |     text = _set_fdf_directive(text, "MD.TypeOfRun", "CG")
00338 |     return _set_fdf_directive(text, "MD.NumCGsteps", "0")
00339 | 
00340 | 
00341 | def materialize_fdf_text(
00342 |     base_text: str,
00343 |     structure: FdfStructure,
00344 |     *,
00345 |     positions_ang: list[list[float]] | list[tuple[float, float, float]],
00346 |     atom_species: list[int] | None = None,
00347 |     lattice_vectors_ang: list[list[float]] | list[tuple[float, float, float]] | None = None,
00348 |     system_label: str | None = None,
00349 |     system_name: str | None = None,
00350 |     required_output_flags: dict[str, str] | None = None,
00351 |     single_point: bool = False,
00352 | ) -> str:
00353 |     positions = _normalized_positions(positions_ang)
00354 |     species_indices = list(atom_species or structure.atom_species)
00355 |     if len(positions) != len(species_indices):
00356 |         raise FdfMaterializationError(
00357 |             f"positions_ang length {len(positions)} does not match atom_species length {len(species_indices)}."
00358 |         )
00359 |     species_by_index = {item.index: item for item in structure.species}
00360 |     missing_species = sorted({int(index) for index in species_indices} - set(species_by_index))
00361 |     if missing_species:
00362 |         raise FdfMaterializationError(f"Cannot materialize atoms with undeclared species: {missing_species}")
00363 | 
00364 |     text = base_text
00365 |     if system_label is not None:
00366 |         text = _set_fdf_directive(text, "SystemLabel", str(system_label))
00367 |     if system_name is not None:
00368 |         text = _set_fdf_directive(text, "SystemName", str(system_name))
00369 |     text = _set_fdf_directive(text, "NumberOfAtoms", str(len(positions)))
00370 |     text = _set_fdf_directive(text, "AtomicCoordinatesFormat", "Ang")
00371 |     if lattice_vectors_ang is not None:
00372 |         vectors = _normalized_positions(lattice_vectors_ang)
00373 |         lattice_lines = [
00374 |             f" {_format_float(vector[0])}  {_format_float(vector[1])}  {_format_float(vector[2])}"
00375 |             for vector in vectors
00376 |         ]
00377 |         text = _set_fdf_directive(text, "LatticeConstant", "1.0 Ang")
00378 |         text = _replace_or_append_block(text, "LatticeVectors", lattice_lines)
00379 |     text = _replace_or_append_block(
00380 |         text,
00381 |         "AtomicCoordinatesAndAtomicSpecies",
00382 |         _coordinate_block_lines(positions, species_indices, species_by_index),
00383 |     )
00384 |     if single_point:
00385 |         text = strip_to_single_point(text)
00386 |     return ensure_required_output_flags(text, required_output_flags)
00387 | 
00388 | 
00389 | def materialize_sample_fdf(
00390 |     base_fdf: Path,
00391 |     output_fdf: Path,
00392 |     *,
00393 |     positions_ang: list[list[float]] | list[tuple[float, float, float]],
00394 |     atom_species: list[int] | None = None,
00395 |     lattice_vectors_ang: list[list[float]] | list[tuple[float, float, float]] | None = None,
00396 |     system_label: str | None = None,
00397 |     system_name: str | None = None,
00398 |     structure_type: str | None = None,
00399 |     required_output_flags: dict[str, str] | None = None,
00400 |     single_point: bool = False,
00401 | ) -> MaterializedFdf:
00402 |     structure = extract_fdf_structure(base_fdf, structure_type=structure_type)
00403 |     base_text = base_fdf.read_text(encoding="utf-8", errors="ignore")
00404 |     materialized_text = materialize_fdf_text(
00405 |         base_text,
00406 |         structure,
00407 |         positions_ang=positions_ang,
00408 |         atom_species=atom_species,
00409 |         lattice_vectors_ang=lattice_vectors_ang,
00410 |         system_label=system_label,
00411 |         system_name=system_name,
00412 |         required_output_flags=required_output_flags,
00413 |         single_point=single_point,
00414 |     )
00415 |     output_fdf.parent.mkdir(parents=True, exist_ok=True)
00416 |     output_fdf.write_text(materialized_text, encoding="utf-8")
00417 |     output_structure = extract_fdf_structure(output_fdf, structure_type=structure_type)
00418 |     metadata = output_structure.to_manifest_dict()
00419 |     metadata.update(
00420 |         {
00421 |             "base_fdf": str(base_fdf),
00422 |             "base_fdf_sha256": file_sha256(base_fdf),
00423 |             "materialized_fdf_sha256": file_sha256(output_fdf),
00424 |             "required_output_flags": required_output_flags or DEFAULT_REQUIRED_OUTPUT_FLAGS,
00425 |             "single_point": bool(single_point),
00426 |         }
00427 |     )
00428 |     return MaterializedFdf(path=output_fdf, metadata=metadata)
```

## `shared/siesta_run_fdf.py`

SHA-256: `ef26afc36ac755966a58aa9e5813e7ed2b450b37b0a451a055064ad45b30d5b0`

```py
00001 | """Layered SIESTA RUN.fdf rendering shared by dataset generators."""
00002 | 
00003 | from __future__ import annotations
00004 | 
00005 | from typing import Any
00006 | 
00007 | 
00008 | GENERATED_HEADER = "# Generated from pipeline_config.yaml using shared RUN.fdf layers"
00009 | 
00010 | 
00011 | def fdf_bool(value: Any) -> str:
00012 |     if isinstance(value, str):
00013 |         text = value.strip()
00014 |         if text.lower() in {"t", "true", ".true."}:
00015 |             return "T"
00016 |         if text.lower() in {"f", "false", ".false."}:
00017 |             return "F"
00018 |         return text
00019 |     return "T" if bool(value) else "F"
00020 | 
00021 | 
00022 | def format_float(value: Any) -> str:
00023 |     return f"{float(value):.8f}"
00024 | 
00025 | 
00026 | def format_value(value: Any) -> str:
00027 |     if isinstance(value, bool):
00028 |         return fdf_bool(value)
00029 |     return str(value)
00030 | 
00031 | 
00032 | def species_map(species: list[dict[str, Any]]) -> dict[int, tuple[int, str]]:
00033 |     return {
00034 |         int(item["index"]): (int(item["atomic_number"]), str(item["symbol"]))
00035 |         for item in species
00036 |     }
00037 | 
00038 | 
00039 | def md_common_settings(md: dict[str, Any]) -> dict[str, Any]:
00040 |     return {
00041 |         "ForceAuxCell": fdf_bool(md.get("force_aux_cell", True)),
00042 |         "MeshCutoff": md.get("mesh_cutoff", "200 Ry"),
00043 |         "PAO.BasisType": md.get("basis_type", "split"),
00044 |         "PAO.BasisSize": md.get("basis_size", "DZP"),
00045 |         "PAO.EnergyShift": md.get("energy_shift", "0.03 eV"),
00046 |         "XC.functional": md.get("xc_functional", "GGA"),
00047 |         "XC.authors": md.get("xc_authors", "PBE"),
00048 |         "MaxSCFIterations": md.get("max_scf_iterations", 200),
00049 |         "SolutionMethod": md.get("solution_method", "diagon"),
00050 |         "DM.MixingWeight": md.get("dm_mixing_weight", 0.02),
00051 |         "DM.NumberPulay": md.get("dm_number_pulay", 3),
00052 |         "DM.Tolerance": md.get("dm_tolerance", "1.d-5"),
00053 |         "DM.Require.Energy.Convergence": md.get("dm_require_energy_convergence", "T"),
00054 |         "DM.Energy.Tolerance": md.get("dm_energy_tolerance", "1.e-5 eV"),
00055 |         "SpinPolarized": md.get("spin_polarized", "F"),
00056 |         "FixSpin": md.get("fix_spin", "F"),
00057 |         "NonCollinearSpin": md.get("non_collinear_spin", "F"),
00058 |         "SaveHS": "true" if bool(md.get("save_hs_file", True)) else "false",
00059 |         "Save.HS": fdf_bool(md.get("save_hs_file", True)),
00060 |         "TS.HS.Save": fdf_bool(md.get("save_hs", True)),
00061 |         "TS.DE.Save": fdf_bool(md.get("save_de", True)),
00062 |         "XML.Write": fdf_bool(md.get("xml_write", True)),
00063 |         "Write.OrbitalIndex": "T",
00064 |     }
00065 | 
00066 | 
00067 | def render_common_run_fdf(
00068 |     *,
00069 |     system_name: str,
00070 |     system_label: str,
00071 |     lattice_constant: dict[str, Any],
00072 |     lattice_vectors: list[list[float]],
00073 |     species: list[dict[str, Any]],
00074 |     coordinates_format: str,
00075 |     atoms: list[dict[str, Any]] | None = None,
00076 |     positions: list[list[float]] | None = None,
00077 |     atom_species: list[int] | None = None,
00078 |     kgrid_monkhorst_pack: list[list[Any]] | None = None,
00079 |     siesta_settings: dict[str, Any],
00080 |     header: str,
00081 | ) -> str:
00082 |     species_by_index = species_map(species)
00083 |     if atoms is not None:
00084 |         positions = [atom["position"] for atom in atoms]
00085 |         atom_species = [int(atom["species_index"]) for atom in atoms]
00086 |     if positions is None or atom_species is None:
00087 |         raise RuntimeError("render_common_run_fdf requires atoms or positions+atom_species.")
00088 | 
00089 |     lines = [
00090 |         GENERATED_HEADER,
00091 |         f"# Common base: {header}",
00092 |         "",
00093 |         f"SystemName   {system_name}",
00094 |         f"SystemLabel  {system_label}",
00095 |         "",
00096 |         f"NumberOfSpecies  {len(species_by_index)}",
00097 |         f"NumberOfAtoms    {len(atom_species)}",
00098 |         "",
00099 |         "%block ChemicalSpeciesLabel",
00100 |     ]
00101 |     for index, (atomic_number, symbol) in sorted(species_by_index.items()):
00102 |         lines.append(f" {index:>1}  {atomic_number:>2}  {symbol}")
00103 |     lines.extend(
00104 |         [
00105 |             "%endblock ChemicalSpeciesLabel",
00106 |             "",
00107 |             f"LatticeConstant  {lattice_constant['value']} {lattice_constant['unit']}",
00108 |             "%block LatticeVectors",
00109 |         ]
00110 |     )
00111 |     for vector in lattice_vectors:
00112 |         lines.append(f" {format_float(vector[0])}   {format_float(vector[1])}   {format_float(vector[2])}")
00113 |     lines.extend(
00114 |         [
00115 |             "%endblock LatticeVectors",
00116 |             "",
00117 |             f"AtomicCoordinatesFormat {coordinates_format}",
00118 |             "%block AtomicCoordinatesAndAtomicSpecies",
00119 |         ]
00120 |     )
00121 |     for position, species_index in zip(positions, atom_species):
00122 |         symbol = species_by_index[int(species_index)][1]
00123 |         lines.append(
00124 |             f" {format_float(position[0])}  {format_float(position[1])}  "
00125 |             f"{format_float(position[2])}  {int(species_index)}  # {symbol}"
00126 |         )
00127 |     lines.extend(["%endblock AtomicCoordinatesAndAtomicSpecies", ""])
00128 |     if kgrid_monkhorst_pack:
00129 |         lines.append("%block kgrid_Monkhorst_Pack")
00130 |         for row in kgrid_monkhorst_pack:
00131 |             lines.append(f" {row[0]}  {row[1]}  {row[2]}  {row[3]}")
00132 |         lines.extend(["%endblock kgrid_Monkhorst_Pack", ""])
00133 |     for key, value in siesta_settings.items():
00134 |         lines.append(f"{key:<32} {format_value(value)}")
00135 |     return "\n".join(lines).rstrip() + "\n"
00136 | 
00137 | 
00138 | def render_md_layer(md: dict[str, Any], block: dict[str, Any] | None = None) -> str:
00139 |     block = dict(block or {})
00140 |     steps = int(block.get("n_snapshots") or block.get("steps") or md.get("steps") or 0)
00141 |     if steps <= 0:
00142 |         raise RuntimeError("MD layer requires a positive number of snapshots/steps.")
00143 |     temperature = block.get("temperature_K", md.get("temperature_K", md.get("initial_temperature_K", 300.0)))
00144 |     timestep_fs = block.get("timestep_fs", md.get("timestep_fs", 1.0))
00145 |     ensemble = str(block.get("ensemble", md.get("ensemble", "nve"))).strip().lower()
00146 |     thermostat = str(block.get("thermostat", md.get("thermostat", ""))).strip().lower()
00147 |     type_of_run = str(block.get("type_of_run", md.get("type_of_run", "Verlet"))).strip()
00148 |     if ensemble == "nvt" or thermostat == "nose":
00149 |         type_of_run = "Nose"
00150 |     elif type_of_run.lower() == "verlet":
00151 |         type_of_run = "Verlet"
00152 |     lines = [
00153 |         "",
00154 |         "# MD layer.",
00155 |         f"{'MD.TypeOfRun':<32} {type_of_run}",
00156 |         f"{'MD.Steps':<32} {steps}",
00157 |         f"{'MD.InitialTimeStep':<32} 1",
00158 |         f"{'MD.FinalTimeStep':<32} {steps}",
00159 |         f"{'MD.LengthTimeStep':<32} {float(timestep_fs):g} fs",
00160 |         f"{'MD.InitialTemperature':<32} {float(temperature):g} K",
00161 |     ]
00162 |     if type_of_run.lower() in {"nose", "noseparrinellorahman"}:
00163 |         lines.append(f"{'MD.TargetTemperature':<32} {float(temperature):g} K")
00164 |         lines.append(f"{'MD.NoseMass':<32} {block.get('nose_mass', md.get('nose_mass', '100.0 Ry*fs**2'))}")
00165 |     lines.append(f"{'WriteMDHistory':<32} {fdf_bool(md.get('write_md_history', True))}")
00166 |     lua_script = block.get("lua_script", md.get("lua_script"))
00167 |     if lua_script:
00168 |         lines.extend(["", "# Store Hamiltonians for each MD step.", f"{'Lua.Script':<32} {lua_script}"])
00169 |     return "\n".join(lines).rstrip() + "\n"
00170 | 
00171 | 
00172 | def render_fc_layer(force_constants: dict[str, Any], atom_count: int) -> str:
00173 |     first_atom = int(force_constants.get("first_atom", 1))
00174 |     last_atom = force_constants.get("last_atom")
00175 |     if last_atom is None:
00176 |         last_atom = atom_count
00177 |     lines = ["", "# SIESTA force-constants layer."]
00178 |     if "lua_script" in force_constants:
00179 |         lines.append(f"{'Lua.Script':<32} {force_constants['lua_script']}")
00180 |     lines.append(f"{'TS.HS.Save':<32} {fdf_bool(force_constants.get('save_tshs', True))}")
00181 |     lines.append(f"{'TS.DE.Save':<32} {fdf_bool(force_constants.get('save_tsde', True))}")
00182 |     lines.append(f"{'MD.TypeOfRun':<32} FC")
00183 |     lines.append(f"{'FC.Displacement':<32} {force_constants['displacement']}")
00184 |     lines.append(f"{'FC.First':<32} {first_atom}")
00185 |     lines.append(f"{'FC.Last':<32} {int(last_atom)}")
00186 |     lines.append(f"{'FC.Save.dHS':<32} {fdf_bool(force_constants.get('save_dhs', True))}")
00187 |     if "dHdR_tolerance" in force_constants:
00188 |         lines.append(f"{'FC.dHdR.Tolerance':<32} {force_constants['dHdR_tolerance']}")
00189 |     if "dSdR_tolerance" in force_constants:
00190 |         lines.append(f"{'FC.dSdR.Tolerance':<32} {force_constants['dSdR_tolerance']}")
00191 |     return "\n".join(lines).rstrip() + "\n"
```

## `tests/test_generic_cartesian_displacement.py`

SHA-256: `694ef5cbaf0a45b194a1b783b8de8e6fb7493195e5852031b696fd3f11c7c1b3`

```py
00001 | from __future__ import annotations
00002 | 
00003 | import importlib.util
00004 | import json
00005 | import sys
00006 | import tempfile
00007 | import unittest
00008 | from pathlib import Path
00009 | 
00010 | 
00011 | REPO_ROOT = Path(__file__).resolve().parents[1]
00012 | SHARED_DIR = REPO_ROOT / "shared"
00013 | if str(SHARED_DIR) not in sys.path:
00014 |     sys.path.insert(0, str(SHARED_DIR))
00015 | 
00016 | from fdf_materialization import extract_fdf_structure  # noqa: E402
00017 | 
00018 | 
00019 | def load_generic_cartesian_module():
00020 |     scripts_dir = REPO_ROOT / "AtomDisplacement" / "scripts"
00021 |     if str(scripts_dir) not in sys.path:
00022 |         sys.path.insert(0, str(scripts_dir))
00023 |     spec = importlib.util.spec_from_file_location(
00024 |         "generate_generic_cartesian_displacement_dataset_test",
00025 |         scripts_dir / "generate_generic_cartesian_displacement_dataset.py",
00026 |     )
00027 |     assert spec and spec.loader
00028 |     module = importlib.util.module_from_spec(spec)
00029 |     sys.modules[spec.name] = module
00030 |     spec.loader.exec_module(module)
00031 |     return module
00032 | 
00033 | 
00034 | def synthetic_fdf_text() -> str:
00035 |     return "\n".join(
00036 |         [
00037 |             "SystemName synthetic crystal",
00038 |             "SystemLabel synthetic",
00039 |             "NumberOfSpecies 2",
00040 |             "NumberOfAtoms 4",
00041 |             "%block ChemicalSpeciesLabel",
00042 |             " 1 14 Si",
00043 |             " 2 6 C",
00044 |             "%endblock ChemicalSpeciesLabel",
00045 |             "LatticeConstant 1.0 Ang",
00046 |             "%block LatticeVectors",
00047 |             " 5.0 0.0 0.0",
00048 |             " 0.0 5.0 0.0",
00049 |             " 0.0 0.0 5.0",
00050 |             "%endblock LatticeVectors",
00051 |             "AtomicCoordinatesFormat Ang",
00052 |             "%block AtomicCoordinatesAndAtomicSpecies",
00053 |             " 0.0 0.0 0.0 1",
00054 |             " 1.0 0.0 0.0 2",
00055 |             " 0.0 1.0 0.0 1",
00056 |             " 0.0 0.0 1.0 2",
00057 |             "%endblock AtomicCoordinatesAndAtomicSpecies",
00058 |             "MeshCutoff 200 Ry",
00059 |             "",
00060 |         ]
00061 |     )
00062 | 
00063 | 
00064 | class GenericCartesianDisplacementTests(unittest.TestCase):
00065 |     def setUp(self) -> None:
00066 |         self.tmp = tempfile.TemporaryDirectory()
00067 |         self.root = Path(self.tmp.name)
00068 |         self.module = load_generic_cartesian_module()
00069 |         self.write_material()
00070 | 
00071 |     def tearDown(self) -> None:
00072 |         self.tmp.cleanup()
00073 | 
00074 |     def write_material(self) -> None:
00075 |         material_root = self.root / "materials" / "sic"
00076 |         material_root.mkdir(parents=True)
00077 |         (material_root / "RUN.fdf").write_text(synthetic_fdf_text(), encoding="utf-8")
00078 |         pseudo_dir = material_root / "pseudos"
00079 |         pseudo_dir.mkdir()
00080 |         (pseudo_dir / "Si.psf").write_text("si pseudo\n", encoding="utf-8")
00081 |         (pseudo_dir / "C.psml").write_text("c pseudo\n", encoding="utf-8")
00082 |         basis_dir = material_root / "basis"
00083 |         basis_dir.mkdir()
00084 |         (basis_dir / "Si.ion.xml").write_text("<ion />\n", encoding="utf-8")
00085 | 
00086 |     def config(self, **atomic_overrides) -> dict:
00087 |         atomic = {
00088 |             "recipe": "generic_cartesian",
00089 |             "amplitude_ang": 0.03,
00090 |             "selected_species": None,
00091 |             "include_base": False,
00092 |         }
00093 |         atomic.update(atomic_overrides)
00094 |         return {
00095 |             "generation": {"sample_id_format": "sample_{index:04d}"},
00096 |             "material": {
00097 |                 "label": "sic",
00098 |                 "fdf": "materials/sic/RUN.fdf",
00099 |                 "pseudopotential_dir": "materials/sic/pseudos",
00100 |                 "basis_dir": "materials/sic/basis",
00101 |                 "structure_type": "crystal",
00102 |             },
00103 |             "atomic_displacement": atomic,
00104 |         }
00105 | 
00106 |     def test_non_h2o_structure_generates_expected_6n_cartesian_samples(self) -> None:
00107 |         output_dir = self.root / "dataset" / "samples"
00108 | 
00109 |         manifest = self.module.generate_dataset(
00110 |             self.config(),
00111 |             output_dir=output_dir,
00112 |             base_dir=self.root,
00113 |         )
00114 | 
00115 |         self.assertEqual(manifest["generated_structures"], 24)
00116 |         self.assertEqual(manifest["recipe"], "generic_cartesian")
00117 |         self.assertEqual(manifest["axis_order"], ["x", "y", "z"])
00118 |         self.assertEqual(manifest["sign_order"], [1, -1])
00119 |         first = manifest["samples"][0]
00120 |         self.assertEqual(first["sample_id"], "sample_0000")
00121 |         self.assertEqual(first["atom_index"], 1)
00122 |         self.assertEqual(first["species"], "Si")
00123 |         self.assertEqual(first["axis"], "x")
00124 |         self.assertEqual(first["sign"], 1)
00125 |         self.assertEqual(first["amplitude_ang"], 0.03)
00126 |         self.assertEqual(first["split_group_id"], "generic_cartesian_displacement:sic:atom_0001")
00127 |         self.assertTrue((output_dir / "sample_0000" / "metadata.json").exists())
00128 |         self.assertTrue((output_dir.parent / "samples_manifest.json").exists())
00129 |         self.assertTrue((output_dir / "dataset_manifest.json").exists())
00130 |         self.assertTrue((output_dir / "basis" / "Si.ion.xml").exists())
00131 | 
00132 |         structure = extract_fdf_structure(output_dir / "sample_0000" / "RUN.fdf")
00133 |         self.assertEqual(structure.positions_ang[0], (0.03, 0.0, 0.0))
00134 |         self.assertEqual(structure.positions_ang[1], (1.0, 0.0, 0.0))
00135 |         self.assertEqual(structure.positions_ang[2], (0.0, 1.0, 0.0))
00136 |         self.assertEqual(structure.positions_ang[3], (0.0, 0.0, 1.0))
00137 |         run_text = (output_dir / "sample_0000" / "RUN.fdf").read_text(encoding="utf-8")
00138 |         self.assertIn("SaveHS", run_text)
00139 |         self.assertIn("Save.HS", run_text)
00140 |         self.assertIn("TS.HS.Save", run_text)
00141 | 
00142 |     def test_species_filter_restricts_selected_atoms(self) -> None:
00143 |         output_dir = self.root / "dataset" / "samples"
00144 | 
00145 |         manifest = self.module.generate_dataset(
00146 |             self.config(selected_species=["C"]),
00147 |             output_dir=output_dir,
00148 |             base_dir=self.root,
00149 |         )
00150 | 
00151 |         self.assertEqual(manifest["generated_structures"], 12)
00152 |         self.assertEqual([atom["atom_index"] for atom in manifest["selected_atoms"]], [2, 4])
00153 |         self.assertEqual([atom["atom_index"] for atom in manifest["skipped_atoms"]], [1, 3])
00154 |         self.assertTrue(all(sample["species"] == "C" for sample in manifest["samples"]))
00155 | 
00156 |     def test_sample_metadata_records_displacement_recipe_fields(self) -> None:
00157 |         output_dir = self.root / "dataset" / "samples"
00158 | 
00159 |         manifest = self.module.generate_dataset(
00160 |             self.config(selected_species="Si", amplitude_ang="0.05 Ang"),
00161 |             output_dir=output_dir,
00162 |             base_dir=self.root,
00163 |         )
00164 |         metadata = json.loads(
00165 |             (output_dir / manifest["samples"][1]["sample_id"] / "metadata.json").read_text(
00166 |                 encoding="utf-8"
00167 |             )
00168 |         )
00169 | 
00170 |         self.assertEqual(metadata["generation_method"], "generic_cartesian")
00171 |         self.assertEqual(metadata["atom_index"], 1)
00172 |         self.assertEqual(metadata["species"], "Si")
00173 |         self.assertEqual(metadata["axis"], "x")
00174 |         self.assertEqual(metadata["sign"], -1)
00175 |         self.assertEqual(metadata["amplitude_ang"], 0.05)
00176 |         self.assertEqual(metadata["displacement_ang"], [-0.05, 0.0, 0.0])
00177 |         self.assertEqual(metadata["method"], "siesta_fc_cartesian")
00178 | 
00179 |     def test_include_base_adds_reference_sample(self) -> None:
00180 |         output_dir = self.root / "dataset" / "samples"
00181 | 
00182 |         manifest = self.module.generate_dataset(
00183 |             self.config(include_base=True, selected_species=["Si"]),
00184 |             output_dir=output_dir,
00185 |             base_dir=self.root,
00186 |         )
00187 | 
00188 |         self.assertEqual(manifest["generated_structures"], 13)
00189 |         self.assertTrue(manifest["samples"][0]["is_reference"])
00190 |         self.assertEqual(
00191 |             manifest["samples"][0]["split_group_id"],
00192 |             "generic_cartesian_displacement:sic:reference",
00193 |         )
00194 | 
00195 |     def test_invalid_amplitude_fails(self) -> None:
00196 |         with self.assertRaisesRegex(
00197 |             self.module.GenericCartesianDisplacementError,
00198 |             "amplitude_ang must be positive",
00199 |         ):
00200 |             self.module.generate_dataset(
00201 |                 self.config(amplitude_ang=0.0),
00202 |                 output_dir=self.root / "dataset" / "samples",
00203 |                 base_dir=self.root,
00204 |             )
00205 | 
00206 |     def test_empty_species_selection_fails(self) -> None:
00207 |         with self.assertRaisesRegex(
00208 |             self.module.GenericCartesianDisplacementError,
00209 |             "selected_species cannot be empty",
00210 |         ):
00211 |             self.module.generate_dataset(
00212 |                 self.config(selected_species=[]),
00213 |                 output_dir=self.root / "dataset" / "samples",
00214 |                 base_dir=self.root,
00215 |             )
00216 | 
00217 |     def test_h2o_specific_recipe_is_not_generalized_here(self) -> None:
00218 |         with self.assertRaisesRegex(
00219 |             self.module.GenericCartesianDisplacementError,
00220 |             "only supports atomic_displacement.recipe='generic_cartesian'",
00221 |         ):
00222 |             self.module.generate_dataset(
00223 |                 self.config(recipe="h2o_hoh"),
00224 |                 output_dir=self.root / "dataset" / "samples",
00225 |                 base_dir=self.root,
00226 |             )
00227 | 
00228 |     def test_existing_output_requires_explicit_overwrite(self) -> None:
00229 |         output_dir = self.root / "dataset" / "samples"
00230 |         output_dir.mkdir(parents=True)
00231 |         (output_dir / "old.txt").write_text("old\n", encoding="utf-8")
00232 | 
00233 |         with self.assertRaisesRegex(
00234 |             self.module.GenericCartesianDisplacementError,
00235 |             "already exists and is not empty",
00236 |         ):
00237 |             self.module.generate_dataset(
00238 |                 self.config(),
00239 |                 output_dir=output_dir,
00240 |                 base_dir=self.root,
00241 |             )
00242 | 
00243 | 
00244 | if __name__ == "__main__":
00245 |     unittest.main()
```

## `tests/test_material_bundle.py`

SHA-256: `5110f3a9b163885b35775ac587fc92f5b9117c3b4e48cbcccc1048dbd29e7c9d`

```py
00001 | from __future__ import annotations
00002 | 
00003 | import sys
00004 | import tempfile
00005 | import unittest
00006 | from pathlib import Path
00007 | 
00008 | 
00009 | REPO_ROOT = Path(__file__).resolve().parents[1]
00010 | SHARED_DIR = REPO_ROOT / "shared"
00011 | if str(SHARED_DIR) not in sys.path:
00012 |     sys.path.insert(0, str(SHARED_DIR))
00013 | 
00014 | from material_bundle import (  # noqa: E402
00015 |     MaterialBundleError,
00016 |     validate_material_config,
00017 | )
00018 | 
00019 | 
00020 | def write_fdf(path: Path, *, species_rows: list[str] | None = None, coordinate_rows: list[str] | None = None) -> None:
00021 |     species_rows = species_rows or [
00022 |         "1 14 Si",
00023 |         "2 6 C",
00024 |     ]
00025 |     coordinate_rows = coordinate_rows or [
00026 |         "0.0 0.0 0.0 1",
00027 |         "1.0 1.0 1.0 2",
00028 |     ]
00029 |     path.parent.mkdir(parents=True, exist_ok=True)
00030 |     path.write_text(
00031 |         "\n".join(
00032 |             [
00033 |                 "SystemName material fixture",
00034 |                 "SystemLabel material_fixture",
00035 |                 "%block ChemicalSpeciesLabel",
00036 |                 *species_rows,
00037 |                 "%endblock ChemicalSpeciesLabel",
00038 |                 "%block AtomicCoordinatesAndAtomicSpecies",
00039 |                 *coordinate_rows,
00040 |                 "%endblock AtomicCoordinatesAndAtomicSpecies",
00041 |                 "",
00042 |             ]
00043 |         ),
00044 |         encoding="utf-8",
00045 |     )
00046 | 
00047 | 
00048 | class MaterialBundleValidationTests(unittest.TestCase):
00049 |     def setUp(self) -> None:
00050 |         self.tmp = tempfile.TemporaryDirectory()
00051 |         self.root = Path(self.tmp.name)
00052 | 
00053 |     def tearDown(self) -> None:
00054 |         self.tmp.cleanup()
00055 | 
00056 |     def write_valid_bundle(self) -> dict:
00057 |         material_root = self.root / "materials" / "sic"
00058 |         write_fdf(material_root / "RUN.fdf")
00059 |         pseudo_dir = material_root / "pseudos"
00060 |         pseudo_dir.mkdir(parents=True)
00061 |         (pseudo_dir / "Si.psf").write_text("pseudo Si\n", encoding="utf-8")
00062 |         (pseudo_dir / "C.psml").write_text("pseudo C\n", encoding="utf-8")
00063 |         return {
00064 |             "material": {
00065 |                 "label": "sic_test",
00066 |                 "fdf": "materials/sic/RUN.fdf",
00067 |                 "pseudopotential_dir": "materials/sic/pseudos",
00068 |                 "structure_type": "crystal",
00069 |             }
00070 |         }
00071 | 
00072 |     def test_valid_minimal_material_bundle_records_species_pseudos_and_hashes(self) -> None:
00073 |         config = self.write_valid_bundle()
00074 | 
00075 |         validated = validate_material_config(config, base_dir=self.root)
00076 |         manifest = validated.to_manifest_dict()
00077 | 
00078 |         self.assertEqual(manifest["label"], "sic_test")
00079 |         self.assertEqual([row["label"] for row in manifest["species"]], ["Si", "C"])
00080 |         self.assertEqual(sorted(manifest["pseudopotentials"]), ["C", "Si"])
00081 |         self.assertEqual(len(manifest["fdf_sha256"]), 64)
00082 |         self.assertEqual(len(manifest["pseudopotential_sha256"]["Si"]), 64)
00083 |         self.assertEqual(
00084 |             validated.to_manifest_dict(),
00085 |             validate_material_config(config, base_dir=self.root).to_manifest_dict(),
00086 |         )
00087 | 
00088 |     def test_missing_fdf_fails_with_clear_message(self) -> None:
00089 |         config = self.write_valid_bundle()
00090 |         Path(self.root / "materials" / "sic" / "RUN.fdf").unlink()
00091 | 
00092 |         with self.assertRaisesRegex(MaterialBundleError, "FDF does not exist"):
00093 |             validate_material_config(config, base_dir=self.root)
00094 | 
00095 |     def test_missing_pseudopotential_directory_fails(self) -> None:
00096 |         config = self.write_valid_bundle()
00097 |         pseudo_dir = self.root / "materials" / "sic" / "pseudos"
00098 |         for child in pseudo_dir.iterdir():
00099 |             child.unlink()
00100 |         pseudo_dir.rmdir()
00101 | 
00102 |         with self.assertRaisesRegex(MaterialBundleError, "pseudopotential directory does not exist"):
00103 |             validate_material_config(config, base_dir=self.root)
00104 | 
00105 |     def test_missing_pseudo_for_species_fails(self) -> None:
00106 |         config = self.write_valid_bundle()
00107 |         (self.root / "materials" / "sic" / "pseudos" / "C.psml").unlink()
00108 | 
00109 |         with self.assertRaisesRegex(MaterialBundleError, "Missing pseudopotential for species 'C'"):
00110 |             validate_material_config(config, base_dir=self.root)
00111 | 
00112 |     def test_duplicate_pseudo_for_species_fails_as_ambiguous(self) -> None:
00113 |         config = self.write_valid_bundle()
00114 |         (self.root / "materials" / "sic" / "pseudos" / "Si.psml").write_text(
00115 |             "pseudo Si duplicate\n",
00116 |             encoding="utf-8",
00117 |         )
00118 | 
00119 |         with self.assertRaisesRegex(MaterialBundleError, "Ambiguous pseudopotential for species 'Si'"):
00120 |             validate_material_config(config, base_dir=self.root)
00121 | 
00122 |     def test_optional_basis_directory_is_validated_and_hashed(self) -> None:
00123 |         config = self.write_valid_bundle()
00124 |         basis_dir = self.root / "materials" / "sic" / "basis"
00125 |         basis_dir.mkdir()
00126 |         (basis_dir / "Si.ion.xml").write_text("<ion><symbol>Si</symbol></ion>\n", encoding="utf-8")
00127 |         config["material"]["basis_dir"] = "materials/sic/basis"
00128 | 
00129 |         validated = validate_material_config(config, base_dir=self.root)
00130 | 
00131 |         self.assertIn("Si.ion.xml", validated.to_manifest_dict()["basis_file_sha256"])
00132 | 
00133 |     def test_missing_optional_basis_directory_fails_when_configured(self) -> None:
00134 |         config = self.write_valid_bundle()
00135 |         config["material"]["basis_dir"] = "materials/sic/missing_basis"
00136 | 
00137 |         with self.assertRaisesRegex(MaterialBundleError, "basis directory does not exist"):
00138 |             validate_material_config(config, base_dir=self.root)
00139 | 
00140 |     def test_path_traversal_outside_base_dir_fails(self) -> None:
00141 |         config = self.write_valid_bundle()
00142 |         config["material"]["fdf"] = "../outside/RUN.fdf"
00143 | 
00144 |         with self.assertRaisesRegex(MaterialBundleError, "escapes its root"):
00145 |             validate_material_config(config, base_dir=self.root)
00146 | 
00147 |     def test_absolute_paths_are_allowed_by_default_and_recorded(self) -> None:
00148 |         config = self.write_valid_bundle()
00149 |         config["material"]["fdf"] = str(self.root / "materials" / "sic" / "RUN.fdf")
00150 | 
00151 |         validated = validate_material_config(config, base_dir=self.root)
00152 | 
00153 |         self.assertTrue(validated.to_manifest_dict()["absolute_paths_used"])
00154 |         with self.assertRaisesRegex(MaterialBundleError, "Absolute material bundle paths"):
00155 |             validate_material_config(config, base_dir=self.root, allow_absolute_paths=False)
00156 | 
00157 |     def test_coordinates_using_undeclared_species_fail(self) -> None:
00158 |         config = self.write_valid_bundle()
00159 |         write_fdf(
00160 |             self.root / "materials" / "sic" / "RUN.fdf",
00161 |             coordinate_rows=["0.0 0.0 0.0 1", "1.0 1.0 1.0 3"],
00162 |         )
00163 | 
00164 |         with self.assertRaisesRegex(MaterialBundleError, "undeclared species indices"):
00165 |             validate_material_config(config, base_dir=self.root)
00166 | 
00167 |     def test_unsafe_label_fails(self) -> None:
00168 |         config = self.write_valid_bundle()
00169 |         config["material"]["label"] = "../sic"
00170 | 
00171 |         with self.assertRaisesRegex(MaterialBundleError, "material.label"):
00172 |             validate_material_config(config, base_dir=self.root)
00173 | 
00174 | 
00175 | if __name__ == "__main__":
00176 |     unittest.main()
```

## `tests/test_fdf_materialization.py`

SHA-256: `aeafcdd1cae397a69844fc7932608c1bfd0f053a7f72eceabc46c7f0535b13ea`

```py
00001 | from __future__ import annotations
00002 | 
00003 | import sys
00004 | import tempfile
00005 | import unittest
00006 | from pathlib import Path
00007 | 
00008 | 
00009 | REPO_ROOT = Path(__file__).resolve().parents[1]
00010 | SHARED_DIR = REPO_ROOT / "shared"
00011 | if str(SHARED_DIR) not in sys.path:
00012 |     sys.path.insert(0, str(SHARED_DIR))
00013 | 
00014 | from fdf_materialization import (  # noqa: E402
00015 |     DEFAULT_REQUIRED_OUTPUT_FLAGS,
00016 |     FdfMaterializationError,
00017 |     extract_bundle_structure,
00018 |     extract_fdf_structure,
00019 |     materialize_sample_fdf,
00020 | )
00021 | from material_bundle import validate_material_config  # noqa: E402
00022 | from material_presets import resolve_material_bundle  # noqa: E402
00023 | 
00024 | 
00025 | def simple_fdf_text(
00026 |     *,
00027 |     coordinate_format: str = "Ang",
00028 |     include_species: bool = True,
00029 |     output_flags: bool = False,
00030 | ) -> str:
00031 |     lines = [
00032 |         "# user comment preserved",
00033 |         "SystemName   synthetic material",
00034 |         "SystemLabel  synthetic",
00035 |         "NumberOfSpecies  2",
00036 |         "NumberOfAtoms    2",
00037 |     ]
00038 |     if include_species:
00039 |         lines.extend(
00040 |             [
00041 |                 "%block ChemicalSpeciesLabel",
00042 |                 " 1  14  Si",
00043 |                 " 2   6  C",
00044 |                 "%endblock ChemicalSpeciesLabel",
00045 |             ]
00046 |         )
00047 |     lines.extend(
00048 |         [
00049 |             "LatticeConstant 1.0 Ang",
00050 |             "%block LatticeVectors",
00051 |             " 4.0 0.0 0.0",
00052 |             " 0.0 4.0 0.0",
00053 |             " 0.0 0.0 4.0",
00054 |             "%endblock LatticeVectors",
00055 |             f"AtomicCoordinatesFormat {coordinate_format}",
00056 |             "%block AtomicCoordinatesAndAtomicSpecies",
00057 |             " 0.0 0.0 0.0 1 # Si",
00058 |             " 1.0 1.0 1.0 2 # C",
00059 |             "%endblock AtomicCoordinatesAndAtomicSpecies",
00060 |             "MeshCutoff 300 Ry",
00061 |             "Custom.User.Setting keep-me",
00062 |         ]
00063 |     )
00064 |     if output_flags:
00065 |         lines.extend(
00066 |             [
00067 |                 "SaveHS false",
00068 |                 "Save.HS F",
00069 |                 "XML.Write F",
00070 |             ]
00071 |         )
00072 |     return "\n".join(lines) + "\n"
00073 | 
00074 | 
00075 | class FdfMaterializationTests(unittest.TestCase):
00076 |     def setUp(self) -> None:
00077 |         self.tmp = tempfile.TemporaryDirectory()
00078 |         self.root = Path(self.tmp.name)
00079 | 
00080 |     def tearDown(self) -> None:
00081 |         self.tmp.cleanup()
00082 | 
00083 |     def write_base_fdf(self, text: str | None = None) -> Path:
00084 |         path = self.root / "RUN.fdf"
00085 |         path.write_text(text or simple_fdf_text(), encoding="utf-8")
00086 |         return path
00087 | 
00088 |     def test_valid_simple_fdf_extracts_species_atoms_and_lattice(self) -> None:
00089 |         path = self.write_base_fdf()
00090 | 
00091 |         structure = extract_fdf_structure(path, structure_type="crystal")
00092 | 
00093 |         self.assertEqual(structure.atom_count, 2)
00094 |         self.assertEqual([item.label for item in structure.species], ["Si", "C"])
00095 |         self.assertEqual(structure.atom_species, [1, 2])
00096 |         self.assertEqual(len(structure.lattice_vectors_ang), 3)
00097 |         self.assertEqual(structure.structure_type, "crystal")
00098 | 
00099 |     def test_lattice_vectors_are_scaled_to_angstrom(self) -> None:
00100 |         path = self.write_base_fdf(simple_fdf_text().replace("LatticeConstant 1.0 Ang", "LatticeConstant 2.0 Ang"))
00101 | 
00102 |         structure = extract_fdf_structure(path)
00103 | 
00104 |         self.assertEqual(structure.lattice_vectors_ang[0], (8.0, 0.0, 0.0))
00105 | 
00106 |     def test_materialized_sample_updates_coordinates_and_preserves_settings(self) -> None:
00107 |         output_flags_text = simple_fdf_text(output_flags=True)
00108 |         base = self.write_base_fdf(output_flags_text)
00109 |         self.assertIn("Custom.User.Setting keep-me", output_flags_text)
00110 |         output = self.root / "sample" / "RUN.fdf"
00111 | 
00112 |         result = materialize_sample_fdf(
00113 |             base,
00114 |             output,
00115 |             positions_ang=[[0.2, 0.3, 0.4], [1.2, 1.3, 1.4]],
00116 |             system_label="sample_001",
00117 |             system_name="Synthetic sample 001",
00118 |         )
00119 |         text = output.read_text(encoding="utf-8")
00120 | 
00121 |         self.assertIn("# user comment preserved", text)
00122 |         self.assertIn("Custom.User.Setting keep-me", text)
00123 |         self.assertIn("SystemLabel                      sample_001", text)
00124 |         self.assertIn("0.200000000000", text)
00125 |         self.assertNotIn(" 0.0 0.0 0.0 1 # Si", text)
00126 |         self.assertEqual(result.metadata["atom_count"], 2)
00127 |         self.assertEqual(len(result.metadata["base_fdf_sha256"]), 64)
00128 |         self.assertEqual(len(result.metadata["materialized_fdf_sha256"]), 64)
00129 | 
00130 |     def test_single_point_strips_md_and_lua_directives(self) -> None:
00131 |         base = self.write_base_fdf(
00132 |             simple_fdf_text()
00133 |             + "MD.TypeOfRun Verlet\nMD.Steps 20\nMD.InitialTemperature 450 K\nWriteMDHistory T\nLua.Script md_store.lua\n"
00134 |         )
00135 |         output = self.root / "single_point" / "RUN.fdf"
00136 | 
00137 |         result = materialize_sample_fdf(
00138 |             base,
00139 |             output,
00140 |             positions_ang=[[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]],
00141 |             single_point=True,
00142 |         )
00143 |         text = output.read_text(encoding="utf-8")
00144 | 
00145 |         self.assertNotIn("Verlet", text)
00146 |         self.assertNotIn("MD.Steps", text)
00147 |         self.assertNotIn("MD.InitialTemperature", text)
00148 |         self.assertNotIn("WriteMDHistory", text)
00149 |         self.assertNotIn("Lua.Script", text)
00150 |         self.assertIn("MD.TypeOfRun", text)
00151 |         self.assertIn("CG", text)
00152 |         self.assertIn("MD.NumCGsteps", text)
00153 |         self.assertTrue(result.metadata["single_point"])
00154 | 
00155 |     def test_default_materialization_preserves_md_directives(self) -> None:
00156 |         base = self.write_base_fdf(simple_fdf_text() + "MD.TypeOfRun Verlet\nMD.Steps 20\nLua.Script md_store.lua\n")
00157 |         output = self.root / "with_md" / "RUN.fdf"
00158 | 
00159 |         result = materialize_sample_fdf(base, output, positions_ang=[[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]])
00160 |         text = output.read_text(encoding="utf-8")
00161 | 
00162 |         self.assertIn("Verlet", text)
00163 |         self.assertIn("Lua.Script", text)
00164 |         self.assertFalse(result.metadata["single_point"])
00165 | 
00166 |     def test_unsupported_coordinate_format_fails_clearly(self) -> None:
00167 |         path = self.write_base_fdf(simple_fdf_text(coordinate_format="Fractional"))
00168 | 
00169 |         with self.assertRaisesRegex(FdfMaterializationError, "unsupported AtomicCoordinatesFormat"):
00170 |             extract_fdf_structure(path)
00171 | 
00172 |     def test_missing_species_block_fails_clearly(self) -> None:
00173 |         path = self.write_base_fdf(simple_fdf_text(include_species=False))
00174 | 
00175 |         with self.assertRaisesRegex(FdfMaterializationError, "ChemicalSpeciesLabel"):
00176 |             extract_fdf_structure(path)
00177 | 
00178 |     def test_required_output_flags_are_inserted_or_replaced(self) -> None:
00179 |         base = self.write_base_fdf(simple_fdf_text(output_flags=True))
00180 |         output = self.root / "sample" / "RUN.fdf"
00181 | 
00182 |         materialize_sample_fdf(
00183 |             base,
00184 |             output,
00185 |             positions_ang=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
00186 |         )
00187 |         text = output.read_text(encoding="utf-8")
00188 | 
00189 |         for key, value in DEFAULT_REQUIRED_OUTPUT_FLAGS.items():
00190 |             self.assertIn(f"{key:<32} {value}", text)
00191 |         self.assertNotIn("SaveHS false", text)
00192 |         self.assertNotIn("Save.HS F", text)
00193 |         self.assertNotIn("XML.Write F", text)
00194 | 
00195 |     def test_h2o_preset_fixture_can_be_extracted_and_materialized(self) -> None:
00196 |         resolved = resolve_material_bundle({"material": {"preset": "h2o"}}, base_dir=REPO_ROOT)
00197 |         structure = extract_bundle_structure(resolved.validated)
00198 |         output = self.root / "h2o_sample" / "RUN.fdf"
00199 |         shifted = [
00200 |             [position[0] + 0.01, position[1], position[2]]
00201 |             for position in structure.positions_ang
00202 |         ]
00203 | 
00204 |         result = materialize_sample_fdf(
00205 |             resolved.validated.bundle.fdf,
00206 |             output,
00207 |             positions_ang=shifted,
00208 |             structure_type=resolved.validated.bundle.structure_type,
00209 |         )
00210 | 
00211 |         self.assertEqual(structure.atom_count, 3)
00212 |         self.assertEqual([item.label for item in structure.species], ["O", "H"])
00213 |         self.assertEqual(result.metadata["structure_type"], "molecule")
00214 |         text = output.read_text(encoding="utf-8")
00215 |         self.assertIn("SaveHS", text)
00216 |         self.assertIn("Save.HS", text)
00217 | 
00218 |     def test_non_h2o_validated_bundle_can_be_materialized(self) -> None:
00219 |         material_root = self.root / "materials" / "sic"
00220 |         fdf = material_root / "RUN.fdf"
00221 |         fdf.parent.mkdir(parents=True)
00222 |         fdf.write_text(simple_fdf_text(), encoding="utf-8")
00223 |         pseudo_dir = material_root / "pseudos"
00224 |         pseudo_dir.mkdir()
00225 |         (pseudo_dir / "Si.psf").write_text("si\n", encoding="utf-8")
00226 |         (pseudo_dir / "C.psml").write_text("c\n", encoding="utf-8")
00227 |         validated = validate_material_config(
00228 |             {
00229 |                 "material": {
00230 |                     "label": "sic",
00231 |                     "fdf": "materials/sic/RUN.fdf",
00232 |                     "pseudopotential_dir": "materials/sic/pseudos",
00233 |                     "structure_type": "crystal",
00234 |                 }
00235 |             },
00236 |             base_dir=self.root,
00237 |         )
00238 |         structure = extract_bundle_structure(validated)
00239 |         output = self.root / "sic_sample" / "RUN.fdf"
00240 | 
00241 |         result = materialize_sample_fdf(
00242 |             validated.bundle.fdf,
00243 |             output,
00244 |             positions_ang=structure.positions_ang,
00245 |             structure_type=validated.bundle.structure_type,
00246 |         )
00247 | 
00248 |         self.assertEqual([item["label"] for item in result.metadata["species"]], ["Si", "C"])
00249 |         self.assertEqual(result.metadata["structure_type"], "crystal")
00250 | 
00251 | 
00252 | if __name__ == "__main__":
00253 |     unittest.main()
```

## `tests/test_siesta_material_provenance.py`

SHA-256: `a0c9df3cdbb43d7d76ad4ef4313c2d219fb06ba70ef99293efbff43c1ea00559`

```py
00001 | from __future__ import annotations
00002 | 
00003 | import importlib.util
00004 | import json
00005 | import os
00006 | import sys
00007 | import tempfile
00008 | import unittest
00009 | from pathlib import Path
00010 | 
00011 | 
00012 | REPO_ROOT = Path(__file__).resolve().parents[1]
00013 | SCRIPTS_DIR = REPO_ROOT / "AtomDisplacement" / "scripts"
00014 | 
00015 | 
00016 | def load_atom_utils():
00017 |     if str(SCRIPTS_DIR) not in sys.path:
00018 |         sys.path.insert(0, str(SCRIPTS_DIR))
00019 |     spec = importlib.util.spec_from_file_location(
00020 |         "atom_displacement_utils_material_provenance_test",
00021 |         SCRIPTS_DIR / "atom_displacement_utils.py",
00022 |     )
00023 |     assert spec and spec.loader
00024 |     module = importlib.util.module_from_spec(spec)
00025 |     sys.modules[spec.name] = module
00026 |     spec.loader.exec_module(module)
00027 |     return module
00028 | 
00029 | 
00030 | def load_md_generator():
00031 |     md_scripts = REPO_ROOT / "MD" / "scripts"
00032 |     if str(md_scripts) not in sys.path:
00033 |         sys.path.insert(0, str(md_scripts))
00034 |     spec = importlib.util.spec_from_file_location(
00035 |         "generate_md_dataset_material_provenance_test",
00036 |         md_scripts / "generate_md_dataset.py",
00037 |     )
00038 |     assert spec and spec.loader
00039 |     module = importlib.util.module_from_spec(spec)
00040 |     sys.modules[spec.name] = module
00041 |     spec.loader.exec_module(module)
00042 |     return module
00043 | 
00044 | 
00045 | def synthetic_fdf_text() -> str:
00046 |     return "\n".join(
00047 |         [
00048 |             "SystemName synthetic",
00049 |             "SystemLabel synthetic",
00050 |             "NumberOfSpecies 2",
00051 |             "NumberOfAtoms 2",
00052 |             "%block ChemicalSpeciesLabel",
00053 |             " 1 14 Si",
00054 |             " 2 6 C",
00055 |             "%endblock ChemicalSpeciesLabel",
00056 |             "LatticeConstant 1.0 Ang",
00057 |             "%block LatticeVectors",
00058 |             " 5.0 0.0 0.0",
00059 |             " 0.0 5.0 0.0",
00060 |             " 0.0 0.0 5.0",
00061 |             "%endblock LatticeVectors",
00062 |             "AtomicCoordinatesFormat Ang",
00063 |             "%block AtomicCoordinatesAndAtomicSpecies",
00064 |             " 0.0 0.0 0.0 1",
00065 |             " 1.0 0.0 0.0 2",
00066 |             "%endblock AtomicCoordinatesAndAtomicSpecies",
00067 |             "Save.HS T",
00068 |             "XML.Write T",
00069 |             "",
00070 |         ]
00071 |     )
00072 | 
00073 | 
00074 | class SiestaMaterialProvenanceTests(unittest.TestCase):
00075 |     def setUp(self) -> None:
00076 |         self.tmp = tempfile.TemporaryDirectory()
00077 |         self.root = Path(self.tmp.name)
00078 |         self.module = load_atom_utils()
00079 |         self.write_material()
00080 | 
00081 |     def tearDown(self) -> None:
00082 |         self.tmp.cleanup()
00083 | 
00084 |     def write_material(self) -> None:
00085 |         material_root = self.root / "materials" / "sic"
00086 |         material_root.mkdir(parents=True)
00087 |         (material_root / "RUN.fdf").write_text(synthetic_fdf_text(), encoding="utf-8")
00088 |         pseudo_dir = material_root / "pseudos"
00089 |         pseudo_dir.mkdir()
00090 |         (pseudo_dir / "Si.psf").write_text("si pseudo\n", encoding="utf-8")
00091 |         (pseudo_dir / "C.psml").write_text("c pseudo\n", encoding="utf-8")
00092 |         basis_dir = material_root / "basis"
00093 |         basis_dir.mkdir()
00094 |         (basis_dir / "Si.ion.xml").write_text("<ion />\n", encoding="utf-8")
00095 |         (basis_dir / "C.ion.xml").write_text("<ion />\n", encoding="utf-8")
00096 | 
00097 |     def config(self) -> dict:
00098 |         return {
00099 |             "paths": {
00100 |                 "run_fdf_name": "RUN.fdf",
00101 |                 "run_out_name": "RUN.out",
00102 |             },
00103 |             "material": {
00104 |                 "label": "sic",
00105 |                 "fdf": "materials/sic/RUN.fdf",
00106 |                 "pseudopotential_dir": "materials/sic/pseudos",
00107 |                 "basis_dir": "materials/sic/basis",
00108 |                 "structure_type": "crystal",
00109 |             },
00110 |         }
00111 | 
00112 |     def write_sample(self, name: str = "sample_001") -> Path:
00113 |         sample = self.root / name
00114 |         sample.mkdir()
00115 |         (sample / "RUN.fdf").write_text(synthetic_fdf_text(), encoding="utf-8")
00116 |         return sample
00117 | 
00118 |     def make_valid_outputs(self, sample: Path) -> None:
00119 |         (sample / "RUN.out").write_text("Job completed\nSCF cycle converged\n", encoding="utf-8")
00120 |         (sample / "siesta.TSHS").write_bytes(b"matrix")
00121 | 
00122 |     def test_sample_preparation_copies_synthetic_material_pseudos(self) -> None:
00123 |         sample = self.write_sample()
00124 | 
00125 |         manifest = self.module.prepare_sample_material_inputs(
00126 |             sample,
00127 |             self.config(),
00128 |             base_dir=self.root,
00129 |         )
00130 | 
00131 |         self.assertEqual(manifest["label"], "sic")
00132 |         self.assertEqual(sorted(manifest["pseudopotentials_copied_to_sample"].values()), ["C.psml", "Si.psf"])
00133 |         self.assertTrue((sample / "Si.psf").exists())
00134 |         self.assertTrue((sample / "C.psml").exists())
00135 |         self.assertEqual(len(manifest["fdf_sha256"]), 64)
00136 | 
00137 |     def test_missing_pseudo_fails_before_execution(self) -> None:
00138 |         sample = self.write_sample()
00139 |         (self.root / "materials" / "sic" / "pseudos" / "C.psml").unlink()
00140 | 
00141 |         with self.assertRaisesRegex(RuntimeError, "Missing pseudopotential for species 'C'"):
00142 |             self.module.prepare_sample_material_inputs(
00143 |                 sample,
00144 |                 self.config(),
00145 |                 base_dir=self.root,
00146 |             )
00147 | 
00148 |     def test_execution_metadata_records_material_hashes_flags_scf_and_matrix(self) -> None:
00149 |         sample = self.write_sample()
00150 |         self.module.prepare_sample_material_inputs(sample, self.config(), base_dir=self.root)
00151 |         self.make_valid_outputs(sample)
00152 | 
00153 |         validation = self.module.validate_sample_dir(sample)
00154 |         self.assertTrue(validation["valid"], validation["validation_reason"])
00155 |         metadata = self.module.update_sample_execution_metadata(
00156 |             sample,
00157 |             validation,
00158 |             {"status": "completed", "wall_time_seconds": 0.1},
00159 |             self.config(),
00160 |             base_dir=self.root,
00161 |         )
00162 | 
00163 |         self.assertEqual(metadata["material"]["label"], "sic")
00164 |         self.assertEqual(len(metadata["material"]["fdf_sha256"]), 64)
00165 |         self.assertEqual(sorted(metadata["pseudopotential_sha256"]), ["C", "Si"])
00166 |         self.assertEqual(sorted(metadata["basis_file_sha256"]), ["C.ion.xml", "Si.ion.xml"])
00167 |         self.assertTrue(metadata["siesta_output_flags"]["valid"])
00168 |         self.assertTrue(metadata["siesta_execution"]["job_completed"])
00169 |         self.assertTrue(metadata["siesta_execution"]["scf_converged"])
00170 |         self.assertTrue(metadata["reference_matrix"]["path"].endswith("siesta.TSHS"))
00171 |         self.assertEqual(len(metadata["reference_matrix"]["sha256"]), 64)
00172 | 
00173 |     def test_acceptance_rejects_missing_output_failed_scf_and_stale_matrix(self) -> None:
00174 |         missing_output = self.write_sample("missing_output")
00175 |         (missing_output / "siesta.TSHS").write_bytes(b"matrix")
00176 |         self.assertFalse(self.module.validate_sample_dir(missing_output)["valid"])
00177 |         self.assertIn("missing_output", self.module.validate_sample_dir(missing_output)["validation_reason"])
00178 | 
00179 |         failed_scf = self.write_sample("failed_scf")
00180 |         (failed_scf / "RUN.out").write_text("Job completed\n", encoding="utf-8")
00181 |         (failed_scf / "siesta.TSHS").write_bytes(b"matrix")
00182 |         self.assertFalse(self.module.validate_sample_dir(failed_scf)["valid"])
00183 |         self.assertIn("scf_not_converged", self.module.validate_sample_dir(failed_scf)["validation_reason"])
00184 | 
00185 |         stale = self.write_sample("stale_matrix")
00186 |         self.make_valid_outputs(stale)
00187 |         os.utime(stale / "siesta.TSHS", (1000, 1000))
00188 |         os.utime(stale / "RUN.fdf", (2000, 2000))
00189 |         os.utime(stale / "RUN.out", (3000, 3000))
00190 |         self.assertFalse(self.module.validate_sample_dir(stale)["valid"])
00191 |         self.assertIn("stale_matrix", self.module.validate_sample_dir(stale)["validation_reason"])
00192 | 
00193 |     def test_acceptance_rejects_missing_hamiltonian_output_flags(self) -> None:
00194 |         sample = self.write_sample("missing_flags")
00195 |         text = (sample / "RUN.fdf").read_text(encoding="utf-8")
00196 |         text = text.replace("Save.HS T\n", "").replace("XML.Write T\n", "")
00197 |         (sample / "RUN.fdf").write_text(text, encoding="utf-8")
00198 |         self.make_valid_outputs(sample)
00199 | 
00200 |         validation = self.module.validate_sample_dir(sample)
00201 | 
00202 |         self.assertFalse(validation["valid"])
00203 |         self.assertIn("missing_hamiltonian_output_flag", validation["validation_reason"])
00204 |         self.assertIn("missing_xml_write_flag", validation["validation_reason"])
00205 | 
00206 |     def test_h2o_preset_still_prepares_pseudopotentials(self) -> None:
00207 |         sample = self.write_sample()
00208 | 
00209 |         manifest = self.module.prepare_sample_material_inputs(
00210 |             sample,
00211 |             {"material": {"preset": "h2o"}},
00212 |             base_dir=REPO_ROOT,
00213 |         )
00214 | 
00215 |         self.assertEqual(manifest["label"], "h2o")
00216 |         self.assertTrue((sample / "H.psf").exists())
00217 |         self.assertTrue((sample / "O.psf").exists())
00218 | 
00219 |     def test_md_material_preparation_copies_pseudos_and_writes_provenance(self) -> None:
00220 |         md_module = load_md_generator()
00221 |         config = {
00222 |             "_config_dir": self.root,
00223 |             "paths": {
00224 |                 "dataset_dir": str(self.root / "MD" / "dataset"),
00225 |                 "training_dir": str(self.root / "MD" / "training"),
00226 |                 "run_fdf_name": "RUN.fdf",
00227 |                 "run_out_name": "RUN.out",
00228 |                 "training_config_name": "config.yaml",
00229 |                 "venv_activate": str(self.root / ".venv" / "bin" / "activate"),
00230 |             },
00231 |             "material": self.config()["material"],
00232 |         }
00233 | 
00234 |         manifest = md_module.prepare_material_inputs(config)
00235 |         dataset_dir = self.root / "MD" / "dataset"
00236 | 
00237 |         self.assertEqual(manifest["label"], "sic")
00238 |         self.assertTrue((dataset_dir / "Si.psf").exists())
00239 |         self.assertTrue((dataset_dir / "C.psml").exists())
00240 |         recorded = json.loads((dataset_dir / "material_provenance.json").read_text(encoding="utf-8"))
00241 |         self.assertEqual(recorded["label"], "sic")
00242 |         self.assertEqual(sorted(recorded["pseudopotential_sha256"]), ["C", "Si"])
00243 | 
00244 | 
00245 | if __name__ == "__main__":
00246 |     unittest.main()
```

## `MD/scripts/generate_md_dataset.py` — extractos seleccionados

SHA-256 del archivo completo: `c7d14908ca96f11379a228727a1602107b96cf13473dc300e0e2575876aefad4`

### Cabecera, imports y constantes iniciales

```py
00001 | #!/usr/bin/env python3
00002 | """Generate the molecular-dynamics dataset from pipeline_config.yaml."""
00003 | 
00004 | from __future__ import annotations
00005 | 
00006 | import os
00007 | import csv
00008 | import copy
00009 | import json
00010 | import platform
00011 | import shutil
00012 | import subprocess
00013 | import sys
00014 | from concurrent.futures import ThreadPoolExecutor
00015 | from importlib import metadata as importlib_metadata
00016 | from pathlib import Path
00017 | 
00018 | from md_pipeline_config import (
00019 |     command,
00020 |     config_dir,
00021 |     load_pipeline_config,
00022 |     md_temperature_blocks,
00023 |     md_total_steps,
00024 |     paths,
00025 |     render_run_fdf,
00026 | )
00027 | from material_bundle import file_sha256 as material_file_sha256
00028 | from material_presets import resolve_material_bundle
00029 | from graph2mat_material_config import copy_graph2mat_basis_files, resolve_graph2mat_basis_files
00030 | from joint_artifact_contract import (
00031 |     CONTRACT_NAME,
00032 |     G2M_DEEPH_BENCHMARK_PROFILE,
00033 |     find_artifact,
00034 |     resolve_system_label,
00035 |     snapshot_requirements,
00036 |     validate_dataset,
00037 | )
00038 | from benchmark_manifest import extract_siesta_version_from_text, write_benchmark_manifests
00039 | 
00040 | BOHR_TO_ANG = 0.529177210903
00041 | JOINT_GRAPH2MAT_DEEPH_STORE_FILES = "*fdf *TSHS *TSDE *XV *HSX *STRUCT_OUT *ORB_INDX *out"
00042 | JOINT_REQUIRED_FDF_OUTPUT_FLAGS = (
00043 |     "SaveHS",
00044 |     "Save.HS",
00045 |     "TS.HS.Save",
00046 |     "TS.DE.Save",
00047 |     "XML.Write",
00048 |     "Write.OrbitalIndex",
00049 | )
00050 | SPREAD_SPLIT_WARNING = (
00051 |     "MD split strategy 'spread' interleaves trajectory frames across "
00052 |     "train/validation/test and is exploratory/debug only; use "
00053 |     "'blocked_with_gap' with a positive temporal_gap for scientific comparisons."
00054 | )
00055 | MANIFEST_FIELDS = [
00056 |     "sample_id",
00057 |     "method",
00058 |     "source_run",
00059 |     "frame_index",
00060 |     "time_index",
00061 |     "displacement_amplitude",
00062 |     "displacement_magnitude",
00063 |     "displaced_atom",
00064 |     "displacement_axis",
00065 |     "displacement_sign",
00066 |     "displacement_family",
00067 |     "structure_path",
00068 |     "hamiltonian_path",
00069 |     "output_path",
00070 |     "run_out_path",
00071 |     "metadata_path",
00072 |     "valid",
00073 |     "validation_reason",
00074 |     "split",
00075 |     "split_strategy",
00076 |     "temporal_gap",
00077 |     "source_frame_index",
00078 |     "excluded_gap_reason",
00079 |     "seed",
00080 |     "status",
00081 |     "sample_dir",
00082 |     "recipe_id",
00083 |     "recipe_label",
00084 |     "block_id",
00085 |     "block_label",
00086 |     "generation_parameters_json",
00087 |     "sample_index_within_block",
00088 |     "global_sample_id",
00089 |     "temperature_K",
00090 |     "md_block_id",
00091 |     "md_block_label",
00092 |     "md_source_block_dir",
00093 |     "md_source_frame_index",
00094 |     "timestep_fs",
00095 |     "md_type_of_run",
00096 | ]
00097 | 
```

### `execution_environment_provenance` — líneas 118–126

```py
00118 | def execution_environment_provenance() -> dict[str, object]:
00119 |     """Return a small, whitelisted execution environment summary."""
00120 | 
00121 |     return {
00122 |         "python_version": sys.version.split()[0],
00123 |         "platform": platform.platform(),
00124 |         "executable": sys.executable,
00125 |         "package_versions": _package_versions(),
00126 |     }
```

### `probe_siesta_version` — líneas 129–175

```py
00129 | def probe_siesta_version(siesta_command: str) -> dict[str, object]:
00130 |     """Best-effort SIESTA version probe; unknown versions remain unknown.
00131 | 
00132 |     The output must contain a line that validates as a real version (the
00133 |     build-info ``Version : X.Y...`` line, or a line with an ``X.Y`` token).
00134 |     Environment noise on stdout/stderr (e.g. X11 "Authorization required")
00135 |     is never recorded as a version: without a validated version the probe
00136 |     reports ``status: "unverified"`` and leaves ``siesta_version`` empty.
00137 |     """
00138 | 
00139 |     attempts: list[dict[str, object]] = []
00140 |     saw_output = False
00141 |     for flag in ("--version", "-V", "-v"):
00142 |         cmd = [siesta_command, flag]
00143 |         try:
00144 |             result = subprocess.run(
00145 |                 cmd,
00146 |                 check=False,
00147 |                 stdout=subprocess.PIPE,
00148 |                 stderr=subprocess.STDOUT,
00149 |                 text=True,
00150 |                 timeout=10,
00151 |             )
00152 |         except (OSError, subprocess.SubprocessError) as exc:
00153 |             attempts.append({"command": cmd, "error": str(exc)})
00154 |             continue
00155 |         output = (result.stdout or "").strip()
00156 |         attempts.append({"command": cmd, "returncode": result.returncode, "output": output[:2000]})
00157 |         if result.returncode == 0 and output:
00158 |             saw_output = True
00159 |             version = extract_siesta_version_from_text(output)
00160 |             if version is not None:
00161 |                 return {
00162 |                     "siesta_version": version,
00163 |                     "siesta_build_info": output,
00164 |                     "siesta_version_probe": {"status": "detected", "attempts": attempts},
00165 |                 }
00166 |     return {
00167 |         "siesta_version": "",
00168 |         "siesta_build_info": "",
00169 |         "siesta_version_probe": {
00170 |             # "unverified": the command ran and produced output, but nothing
00171 |             # validated as a version; "unavailable": no usable output at all.
00172 |             "status": "unverified" if saw_output else "unavailable",
00173 |             "attempts": attempts,
00174 |         },
00175 |     }
```

### `prepare_material_inputs` — líneas 221–256

```py
00221 | def prepare_material_inputs(config: dict) -> dict:
00222 |     pipeline_paths = paths(config)
00223 |     dataset_dir = pipeline_paths["dataset_dir"]
00224 |     dataset_dir.mkdir(parents=True, exist_ok=True)
00225 |     resolved = resolve_material_bundle(config, base_dir=config_dir(config))
00226 |     validated = resolved.validated
00227 |     copied: dict[str, str] = {}
00228 |     verified: dict[str, str] = {}
00229 |     for label, source in sorted(validated.pseudopotentials.items()):
00230 |         destination = dataset_dir / source.name
00231 |         if destination.exists():
00232 |             if not destination.is_file():
00233 |                 raise RuntimeError(f"MD pseudopotential path is not a file: {destination}")
00234 |             if material_file_sha256(destination) != material_file_sha256(source):
00235 |                 raise RuntimeError(
00236 |                     f"MD pseudopotential for species {label!r} differs from material bundle: {destination}"
00237 |                 )
00238 |             verified[label] = destination.name
00239 |             continue
00240 |         shutil.copy2(source, destination)
00241 |         copied[label] = destination.name
00242 |     manifest = resolved.to_manifest_dict()
00243 |     copied_basis = copy_graph2mat_basis_files(
00244 |         resolve_graph2mat_basis_files(validated),
00245 |         dataset_dir / "material_basis",
00246 |     )
00247 |     manifest.update(
00248 |         {
00249 |             "pseudopotentials_copied_to_dataset": copied,
00250 |             "pseudopotentials_verified_in_dataset": verified,
00251 |             "graph2mat_basis_files": copied_basis,
00252 |         }
00253 |     )
00254 |     manifest.update(siesta_run_provenance(config))
00255 |     write_json(dataset_dir / "material_provenance.json", manifest)
00256 |     return manifest
```

### `write_run_fdf` — líneas 420–424

```py
00420 | def write_run_fdf(config: dict, block: dict | None = None) -> None:
00421 |     pipeline_paths = paths(config)
00422 |     # Asumimos que queremos un RUN.fdf determinista: lo sobreescribimos siempre.
00423 |     pipeline_paths["run_fdf_path"].write_text(render_run_fdf(config, block=block), encoding="utf-8")
00424 |     print(f"[OK] RUN.fdf escrito en {pipeline_paths['run_fdf_path']}")
```

### `_split_counts` — líneas 473–480

```py
00473 | def _split_counts(total: int) -> tuple[int, int, int]:
00474 |     train = int(total * 0.8)
00475 |     validation = int(total * 0.1)
00476 |     test = total - train - validation
00477 |     if total >= 3 and test == 0:
00478 |         test = 1
00479 |         train = max(1, train - 1)
00480 |     return train, validation, test
```

### `_select_spread` — líneas 483–500

```py
00483 | def _select_spread(items: list[Path], count: int) -> list[Path]:
00484 |     if count <= 0:
00485 |         return []
00486 |     if count >= len(items):
00487 |         return list(items)
00488 | 
00489 |     used: set[int] = set()
00490 |     selected: list[int] = []
00491 |     for index in range(count):
00492 |         target = min(len(items) - 1, int((index + 0.5) * len(items) / count))
00493 |         if target in used:
00494 |             target = min(
00495 |                 (candidate for candidate in range(len(items)) if candidate not in used),
00496 |                 key=lambda candidate: abs(candidate - target),
00497 |             )
00498 |         used.add(target)
00499 |         selected.append(target)
00500 |     return [items[index] for index in sorted(selected)]
```

### `_split_spread` — líneas 503–510

```py
00503 | def _split_spread(items: list[Path], train_count: int, validation_count: int, test_count: int) -> dict[str, list[Path]]:
00504 |     test = _select_spread(items, test_count)
00505 |     remaining = [item for item in items if item not in set(test)]
00506 |     validation = _select_spread(remaining, validation_count)
00507 |     train = [item for item in remaining if item not in set(validation)]
00508 |     if len(train) > train_count:
00509 |         train = _select_spread(train, train_count)
00510 |     return {"train": train, "validation": validation, "test": test}
```

### `_split_block` — líneas 513–519

```py
00513 | def _split_block(items: list[Path], train_count: int, validation_count: int, test_count: int) -> dict[str, list[Path]]:
00514 |     requested = train_count + validation_count + test_count
00515 |     selected = list(items[:requested])
00516 |     train = selected[:train_count]
00517 |     validation = selected[train_count : train_count + validation_count]
00518 |     test = selected[train_count + validation_count : train_count + validation_count + test_count]
00519 |     return {"train": train, "validation": validation, "test": test}
```

### `_split_blocked_with_gap` — líneas 536–562

```py
00536 | def _split_blocked_with_gap(
00537 |     items: list[Path],
00538 |     counts: dict[str, int],
00539 |     *,
00540 |     temporal_gap: int,
00541 |     block_order: list[str],
00542 | ) -> tuple[dict[str, list[Path]], list[tuple[Path, str]]]:
00543 |     nonempty_blocks = [name for name in block_order if counts[name] > 0]
00544 |     required = sum(counts.values()) + max(0, len(nonempty_blocks) - 1) * temporal_gap
00545 |     if required > len(items):
00546 |         raise RuntimeError(
00547 |             "El split MD blocked_with_gap necesita mas frames de los disponibles: "
00548 |             f"{required} > {len(items)} (gap={temporal_gap}, counts={counts})."
00549 |         )
00550 |     split_ranges = {"train": [], "validation": [], "test": []}
00551 |     excluded: list[tuple[Path, str]] = []
00552 |     cursor = 0
00553 |     for block_index, split_name in enumerate(nonempty_blocks):
00554 |         count = counts[split_name]
00555 |         split_ranges[split_name] = list(items[cursor : cursor + count])
00556 |         cursor += count
00557 |         if block_index < len(nonempty_blocks) - 1 and temporal_gap > 0:
00558 |             next_split = nonempty_blocks[block_index + 1]
00559 |             for sample in items[cursor : cursor + temporal_gap]:
00560 |                 excluded.append((sample, f"temporal_gap_between_{split_name}_and_{next_split}"))
00561 |             cursor += temporal_gap
00562 |     return split_ranges, excluded
```

### `parse_xv_geometry` — líneas 634–668

```py
00634 | def parse_xv_geometry(xv_path: Path) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, float, float, float]]]:
00635 |     """Read the geometry written by SIESTA in XV format.
00636 | 
00637 |     SIESTA XV coordinates and lattice vectors are in Bohr. The per-frame FDF
00638 |     files written below use Angstrom to match the generated RUN.fdf template.
00639 |     """
00640 | 
00641 |     lines = [line for line in xv_path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]
00642 |     if len(lines) < 4:
00643 |         raise RuntimeError(f"{xv_path}: XV file is too short to contain a geometry.")
00644 |     lattice = []
00645 |     for index in range(3):
00646 |         lattice.append(
00647 |             tuple(value * BOHR_TO_ANG for value in _parse_float_triplet(lines[index], path=xv_path, line_number=index + 1))
00648 |         )
00649 |     try:
00650 |         atom_count = int(lines[3].split()[0])
00651 |     except (IndexError, ValueError) as exc:
00652 |         raise RuntimeError(f"{xv_path}: invalid atom count line.") from exc
00653 |     if len(lines) < 4 + atom_count:
00654 |         raise RuntimeError(f"{xv_path}: expected {atom_count} atom rows, found {max(0, len(lines) - 4)}.")
00655 | 
00656 |     atoms: list[tuple[int, int, float, float, float]] = []
00657 |     for offset, line in enumerate(lines[4 : 4 + atom_count], start=5):
00658 |         parts = line.split()
00659 |         if len(parts) < 5:
00660 |             raise RuntimeError(f"{xv_path}:{offset}: invalid XV atom row.")
00661 |         try:
00662 |             species_index = int(parts[0])
00663 |             atomic_number = int(parts[1])
00664 |             x, y, z = (float(parts[col]) * BOHR_TO_ANG for col in (2, 3, 4))
00665 |         except ValueError as exc:
00666 |             raise RuntimeError(f"{xv_path}:{offset}: invalid XV atom values.") from exc
00667 |         atoms.append((species_index, atomic_number, x, y, z))
00668 |     return lattice, atoms
```

### `rewrite_run_fdf_from_xv` — líneas 728–742

```py
00728 | def rewrite_run_fdf_from_xv(run_fdf_path: Path, xv_path: Path) -> None:
00729 |     lattice, atoms = parse_xv_geometry(xv_path)
00730 |     lattice_lines = [f"{x:.12f} {y:.12f} {z:.12f}" for x, y, z in lattice]
00731 |     atom_lines = [
00732 |         f"{x:.12f} {y:.12f} {z:.12f} {species_index}"
00733 |         for species_index, _atomic_number, x, y, z in atoms
00734 |     ]
00735 |     text = run_fdf_path.read_text(encoding="utf-8", errors="ignore")
00736 |     text = _set_fdf_directive(text, "LatticeConstant", "1.0 Ang")
00737 |     text = _set_fdf_directive(text, "AtomicCoordinatesFormat", "Ang")
00738 |     text = _replace_or_append_block(text, "LatticeVectors", lattice_lines)
00739 |     text = _replace_or_append_block(text, "AtomicCoordinatesAndAtomicSpecies", atom_lines)
00740 |     if MD_RUN_FDF_XV_MARKER not in text:
00741 |         text = text.rstrip() + "\n\n" + MD_RUN_FDF_XV_MARKER + "\n"
00742 |     run_fdf_path.write_text(text, encoding="utf-8")
```

### `write_joint_snapshot_metadata` — líneas 786–861

```py
00786 | def write_joint_snapshot_metadata(
00787 |     sample_dir: Path,
00788 |     config: dict,
00789 |     *,
00790 |     extra: dict | None = None,
00791 |     validation_status: str = "pending_joint_artifact_validation",
00792 | ) -> dict:
00793 |     """Write per-snapshot provenance for the joint Graph2Mat/DeepH contract."""
00794 | 
00795 |     metadata = read_sample_metadata(sample_dir)
00796 |     if extra:
00797 |         metadata.update(extra)
00798 | 
00799 |     system_label, label_errors, label_warnings = resolve_system_label(sample_dir)
00800 |     if system_label is None:
00801 |         fallback_label = str((config.get("md") or {}).get("system_label") or "").strip() or None
00802 |         system_label, label_errors, label_warnings = resolve_system_label(
00803 |             sample_dir,
00804 |             default=fallback_label,
00805 |         )
00806 |     system_label = system_label or str((config.get("md") or {}).get("system_label") or "siesta")
00807 | 
00808 |     artifacts: dict[str, dict[str, object]] = {}
00809 |     for requirement in snapshot_requirements(system_label):
00810 |         expected_name = _expected_artifact_name(requirement, system_label)
00811 |         artifact_path = (
00812 |             sample_dir / "metadata.json"
00813 |             if requirement.key == "metadata"
00814 |             else find_artifact(sample_dir, requirement, system_label)
00815 |         )
00816 |         artifact_info: dict[str, object] = {
00817 |             "filename": artifact_path.name if artifact_path else expected_name,
00818 |             "path": str(artifact_path if artifact_path else sample_dir / expected_name),
00819 |             "required": requirement.required,
00820 |             "present": bool(requirement.key == "metadata" or (artifact_path and artifact_path.exists())),
00821 |         }
00822 |         if artifact_path and artifact_path.exists() and artifact_path.is_file() and requirement.key != "metadata":
00823 |             artifact_info["sha256"] = material_file_sha256(artifact_path)
00824 |         artifacts[requirement.key] = artifact_info
00825 | 
00826 |     run_fdf_path = sample_dir / "RUN.fdf"
00827 |     source_run_fdf = Path(str(metadata.get("source_run_fdf") or ""))
00828 |     if source_run_fdf and not source_run_fdf.is_absolute():
00829 |         source_run_fdf = sample_dir / source_run_fdf
00830 |     frame_index = metadata.get("frame_index", metadata.get("source_frame_index", sample_dir.name))
00831 |     metadata.update(
00832 |         {
00833 |             "sample_id": str(metadata.get("sample_id") or metadata.get("global_sample_id") or f"md_{sample_dir.name}"),
00834 |             "snapshot_dir": str(sample_dir),
00835 |             "system_label": system_label,
00836 |             "siesta_system_label": system_label,
00837 |             "artifact_contract_name": CONTRACT_NAME,
00838 |             "artifact_contract_version": CONTRACT_NAME,
00839 |             "artifact_contract_validation_status": validation_status,
00840 |             "generation_mode": "clean_one_pass",
00841 |             "source": "graph2mat_vs_deeph_dataset_generation",
00842 |             "artifacts": artifacts,
00843 |             "joint_store_files": JOINT_GRAPH2MAT_DEEPH_STORE_FILES,
00844 |             "joint_store_file_patterns": JOINT_GRAPH2MAT_DEEPH_STORE_FILES.split(),
00845 |             "fdf_output_flags": read_joint_fdf_output_flags(run_fdf_path),
00846 |             "fdf_output_flags_required": list(JOINT_REQUIRED_FDF_OUTPUT_FLAGS),
00847 |             "frame_index": str(frame_index),
00848 |             "time_index": str(metadata.get("time_index", frame_index)),
00849 |         }
00850 |     )
00851 |     if label_errors:
00852 |         metadata["system_label_resolution_errors"] = label_errors
00853 |     if label_warnings:
00854 |         metadata["system_label_resolution_warnings"] = label_warnings
00855 |     if run_fdf_path.exists():
00856 |         metadata["run_fdf_sha256"] = material_file_sha256(run_fdf_path)
00857 |     if source_run_fdf.exists() and source_run_fdf.is_file():
00858 |         metadata["source_run_fdf_sha256"] = material_file_sha256(source_run_fdf)
00859 | 
00860 |     write_json(sample_dir / "metadata.json", metadata)
00861 |     return metadata
```

### `effective_fdf_geometry_signature` — líneas 895–917

```py
00895 | def effective_fdf_geometry_signature(run_fdf_path: Path) -> tuple[str, ...]:
00896 |     """Return the geometry blocks Graph2Mat can see in RUN.fdf.
00897 | 
00898 |     Tests use this to catch the historical failure mode where all MD frames had
00899 |     identical effective input geometries despite different ``siesta.XV`` files.
00900 |     """
00901 | 
00902 |     text = run_fdf_path.read_text(encoding="utf-8", errors="ignore").splitlines()
00903 |     capture = None
00904 |     blocks: list[str] = []
00905 |     for line in text:
00906 |         clean = line.split("#", 1)[0].strip()
00907 |         lower = clean.lower()
00908 |         if lower in {"%block latticevectors", "%block atomiccoordinatesandatomicspecies"}:
00909 |             capture = lower
00910 |             blocks.append(lower)
00911 |             continue
00912 |         if lower in {"%endblock latticevectors", "%endblock atomiccoordinatesandatomicspecies"}:
00913 |             capture = None
00914 |             continue
00915 |         if capture and clean:
00916 |             blocks.append(" ".join(clean.split()))
00917 |     return tuple(blocks)
```

### `xv_geometry_signature` — líneas 920–928

```py
00920 | def xv_geometry_signature(xv_path: Path) -> tuple[str, ...]:
00921 |     lattice, atoms = parse_xv_geometry(xv_path)
00922 |     return tuple(
00923 |         [f"L {x:.10f} {y:.10f} {z:.10f}" for x, y, z in lattice]
00924 |         + [
00925 |             f"A {species_index} {atomic_number} {x:.10f} {y:.10f} {z:.10f}"
00926 |             for species_index, atomic_number, x, y, z in atoms
00927 |         ]
00928 |     )
```

### `run_temperature_block` — líneas 1009–1018

```py
01009 | def run_temperature_block(config: dict, block: dict, block_dir: Path) -> None:
01010 |     pipeline_paths = paths(config)
01011 |     block_dir.mkdir(parents=True, exist_ok=True)
01012 |     _copy_pseudopotentials_for_block(pipeline_paths["dataset_dir"], block_dir)
01013 |     block_config = _block_config(config, block_dir, block)
01014 |     setup_store(block_config)
01015 |     write_run_fdf(block_config, block=block)
01016 |     run_siesta_with_venv(block_config)
01017 |     refresh_md_step_geometries(block_config)
01018 |     validate_joint_benchmark_artifacts(block_config)
```

### `combine_temperature_blocks` — líneas 1021–1127

```py
01021 | def combine_temperature_blocks(config: dict, blocks: list[dict]) -> None:
01022 |     # Semantics note (audit C2): datasets built from temperature blocks are
01023 |     # often *named* "iid" in payloads, but each block is one MD trajectory of
01024 |     # consecutive frames (1 fs apart) — the samples are temporally correlated,
01025 |     # NOT independent draws. Splits must respect blocked_with_gap with a
01026 |     # physically meaningful temporal_gap; see docs/known_limitations.md.
01027 |     pipeline_paths = paths(config)
01028 |     dataset_dir = pipeline_paths["dataset_dir"]
01029 |     blocks_root = dataset_dir / "md_temperature_blocks"
01030 |     final_steps_dir = dataset_dir / "MD_steps"
01031 |     if final_steps_dir.exists():
01032 |         shutil.rmtree(final_steps_dir)
01033 |     final_steps_dir.mkdir(parents=True, exist_ok=True)
01034 | 
01035 |     samples = []
01036 |     global_index = 0
01037 |     basis_copied = False
01038 |     combined_run_out = []
01039 |     for block_index, block in enumerate(blocks):
01040 |         block_id = str(block.get("block_id") or f"md_block_{block_index + 1}")
01041 |         block_label = str(block.get("label") or block_id)
01042 |         block_dir = blocks_root / block_id
01043 |         source_steps_dir = block_dir / "MD_steps"
01044 |         if not source_steps_dir.exists():
01045 |             raise RuntimeError(f"Bloque MD sin MD_steps: {source_steps_dir}")
01046 |         if not basis_copied and (source_steps_dir / "basis").exists():
01047 |             shutil.copytree(source_steps_dir / "basis", final_steps_dir / "basis")
01048 |             basis_copied = True
01049 |         step_dirs = sorted(
01050 |             (path for path in source_steps_dir.iterdir() if path.is_dir() and path.name.isdigit()),
01051 |             key=lambda path: int(path.name),
01052 |         )
01053 |         expected = int(block["n_snapshots"])
01054 |         if len(step_dirs) < expected:
01055 |             raise RuntimeError(
01056 |                 f"Bloque MD {block_id} genero menos snapshots de los pedidos: {len(step_dirs)} < {expected}."
01057 |             )
01058 |         for sample_index, source_sample in enumerate(step_dirs[:expected]):
01059 |             target_sample = final_steps_dir / str(global_index)
01060 |             shutil.copytree(source_sample, target_sample)
01061 |             metadata = read_sample_metadata(source_sample)
01062 |             metadata.update(
01063 |                 {
01064 |                     "generation_method": "md_temperature_block",
01065 |                     "method": "md",
01066 |                     # Audit I1: per-block sample_ids (md_0..md_N) repeat across
01067 |                     # temperature blocks, while the frozen split manifest keys
01068 |                     # samples globally. The primary sample_id must be the
01069 |                     # global one so frozen-manifest <-> metadata joins are 1:1;
01070 |                     # the block-local id survives as block_local_sample_id.
01071 |                     "sample_id": f"md_{global_index}",
01072 |                     "block_local_sample_id": metadata.get("sample_id"),
01073 |                     "temperature_K": block.get("temperature_K"),
01074 |                     "n_snapshots_in_block": expected,
01075 |                     "source_block_id": block_id,
01076 |                     "source_block_label": block_label,
01077 |                     "source_block_dir": str(block_dir),
01078 |                     "source_frame_index": source_sample.name,
01079 |                     "sample_index_within_block": sample_index,
01080 |                     "global_sample_id": str(global_index),
01081 |                     "seed": block.get("seed"),
01082 |                     "timestep_fs": block.get("timestep_fs", config["md"].get("timestep_fs")),
01083 |                     "ensemble": block.get("ensemble", config["md"].get("ensemble", "nve")),
01084 |                     "thermostat": block.get("thermostat", config["md"].get("thermostat")),
01085 |                     "type_of_run": block.get("type_of_run", config["md"].get("type_of_run", "Verlet")),
01086 |                     "source_run_fdf": str(block_dir / "RUN.fdf"),
01087 |                     "source_run_out": str(block_dir / "RUN.out"),
01088 |                     "run_fdf_geometry_source": read_sample_metadata(source_sample).get(
01089 |                         "run_fdf_geometry_source",
01090 |                         "XV",
01091 |                     ),
01092 |                     "run_fdf_rewritten_from_xv": True,
01093 |                     "run_fdf_rewrite_time_policy": "post_siesta_geometry_materialization",
01094 |                 }
01095 |             )
01096 |             metadata = write_joint_snapshot_metadata(target_sample, config, extra=metadata)
01097 |             samples.append(metadata)
01098 |             global_index += 1
01099 |         run_out = block_dir / "RUN.out"
01100 |         if run_out.exists():
01101 |             combined_run_out.append(f"\n# ==== MD block {block_id} ({block_label}) ====\n")
01102 |             combined_run_out.append(run_out.read_text(encoding="utf-8", errors="replace"))
01103 | 
01104 |     if global_index != md_total_steps(config):
01105 |         raise RuntimeError(f"Total MD combinado incorrecto: {global_index} != {md_total_steps(config)}")
01106 |     materialized = _materialize_graph2mat_basis_files(
01107 |         dataset_dir,
01108 |         sorted((path for path in final_steps_dir.iterdir() if path.is_dir() and path.name.isdigit()), key=lambda path: int(path.name)),
01109 |     )
01110 |     if materialized:
01111 |         print(f"[OK] Basis Graph2Mat materializada en dataset combinado: {materialized} enlaces/copias.")
01112 |     pipeline_paths["run_out_path"].write_text("".join(combined_run_out), encoding="utf-8")
01113 |     pipeline_paths["run_fdf_path"].write_text(render_run_fdf(config, block={"n_snapshots": md_total_steps(config)}), encoding="utf-8")
01114 |     validate_joint_benchmark_artifacts(config, final_steps_dir)
01115 |     write_json(
01116 |         dataset_dir / "md_temperature_blocks_manifest.json",
01117 |         {
01118 |             "method": "md",
01119 |             "generation_method": "md_temperature_blocks",
01120 |             "total_snapshots": global_index,
01121 |             "temperature_block_workers": temperature_block_workers(config),
01122 |             "parallel_execution_enabled": temperature_block_workers(config) > 1 and len(blocks) > 1,
01123 |             "blocks": blocks,
01124 |             "samples": samples,
01125 |         },
01126 |     )
01127 |     print(f"[OK] Bloques MD combinados: {global_index} snapshots en {final_steps_dir}.")
```

### `run_temperature_block_dataset` — líneas 1130–1179

```py
01130 | def run_temperature_block_dataset(config: dict) -> None:
01131 |     blocks = md_temperature_blocks(config)
01132 |     if not blocks:
01133 |         return
01134 |     workers = temperature_block_workers(config)
01135 |     pipeline_paths = paths(config)
01136 |     dataset_dir = pipeline_paths["dataset_dir"]
01137 |     blocks_root = dataset_dir / "md_temperature_blocks"
01138 |     if blocks_root.exists():
01139 |         shutil.rmtree(blocks_root)
01140 |     blocks_root.mkdir(parents=True, exist_ok=True)
01141 |     for block in blocks:
01142 |         block_id = str(block.get("block_id") or "md_block")
01143 |         print(
01144 |             "[INFO] MD block "
01145 |             f"{block_id}: {block['n_snapshots']} snapshots, "
01146 |             f"T={block.get('temperature_K', config['md'].get('temperature_K', config['md'].get('initial_temperature_K', 300)))} K"
01147 |         )
01148 |     if workers <= 1 or len(blocks) <= 1:
01149 |         for block in blocks:
01150 |             block_id = str(block.get("block_id") or "md_block")
01151 |             run_temperature_block(config, block, blocks_root / block_id)
01152 |     else:
01153 |         print(f"[INFO] Ejecutando {len(blocks)} bloques MD con workers={workers}.")
01154 |         failures: list[tuple[str, Exception]] = []
01155 |         with ThreadPoolExecutor(max_workers=workers) as executor:
01156 |             futures = [
01157 |                 (
01158 |                     str(block.get("block_id") or "md_block"),
01159 |                     executor.submit(
01160 |                         run_temperature_block,
01161 |                         config,
01162 |                         block,
01163 |                         blocks_root / str(block.get("block_id") or "md_block"),
01164 |                     ),
01165 |                 )
01166 |                 for block in blocks
01167 |             ]
01168 |             for block_id, future in futures:
01169 |                 try:
01170 |                     future.result()
01171 |                 except Exception as exc:
01172 |                     failures.append((block_id, exc))
01173 |         if failures:
01174 |             failed_block_ids = ", ".join(block_id for block_id, _ in failures)
01175 |             first_block_id, first_exc = failures[0]
01176 |             raise RuntimeError(
01177 |                 f"MD temperature block(s) failed: {failed_block_ids}. First failure in {first_block_id}: {first_exc}"
01178 |             ) from first_exc
01179 |     combine_temperature_blocks(config, blocks)
```

### `write_excluded_gap_manifest` — líneas 1206–1249

```py
01206 | def write_excluded_gap_manifest(
01207 |     config: dict,
01208 |     split_root: Path,
01209 |     excluded_samples: list[tuple[Path, str]],
01210 |     *,
01211 |     strategy: str,
01212 |     temporal_gap: int,
01213 | ) -> None:
01214 |     pipeline_paths = paths(config)
01215 |     dataset_dir = pipeline_paths["dataset_dir"]
01216 |     rows = [
01217 |         {
01218 |             "sample_id": f"md_{sample.name}",
01219 |             "method": "md",
01220 |             "source_run": str(dataset_dir),
01221 |             "frame_index": sample.name,
01222 |             "time_index": sample.name,
01223 |             "displacement_amplitude": "",
01224 |             "displacement_magnitude": "",
01225 |             "displaced_atom": "",
01226 |             "displacement_axis": "",
01227 |             "displacement_sign": "",
01228 |             "displacement_family": "",
01229 |             "structure_path": str(sample / "RUN.fdf"),
01230 |             "hamiltonian_path": str(_find_hamiltonian(sample) or ""),
01231 |             "output_path": str(pipeline_paths["run_out_path"]),
01232 |             "run_out_path": str(pipeline_paths["run_out_path"]),
01233 |             "metadata_path": str(sample / "metadata.json") if (sample / "metadata.json").exists() else "",
01234 |             "valid": False,
01235 |             "validation_reason": "excluded_temporal_gap",
01236 |             "split": "excluded_gap",
01237 |             "split_strategy": strategy,
01238 |             "temporal_gap": str(temporal_gap),
01239 |             "source_frame_index": sample.name,
01240 |             "excluded_gap_reason": reason,
01241 |             "seed": "",
01242 |             "status": "excluded",
01243 |             "sample_dir": str(sample),
01244 |             **recipe_manifest_fields(config, sample_index=sample.name),
01245 |             **md_sample_manifest_fields(sample),
01246 |         }
01247 |         for sample, reason in excluded_samples
01248 |     ]
01249 |     _write_manifest(split_root / "excluded_gap_manifest.csv", rows)
```

### `write_split_manifests` — líneas 1252–1302

```py
01252 | def write_split_manifests(
01253 |     config: dict,
01254 |     split_root: Path,
01255 |     split_ranges: dict[str, list[Path]],
01256 |     *,
01257 |     strategy: str,
01258 |     temporal_gap: int,
01259 | ) -> None:
01260 |     pipeline_paths = paths(config)
01261 |     dataset_dir = pipeline_paths["dataset_dir"]
01262 |     run_out_path = pipeline_paths["run_out_path"]
01263 |     for split_name, source_samples in split_ranges.items():
01264 |         rows = []
01265 |         for source_sample in source_samples:
01266 |             sample_dir = split_root / split_name / source_sample.name
01267 |             structure_path = sample_dir / "RUN.fdf"
01268 |             hamiltonian_path = _find_hamiltonian(sample_dir)
01269 |             sample_metadata = md_sample_manifest_fields(sample_dir)
01270 |             rows.append(
01271 |                 {
01272 |                     "sample_id": f"md_{source_sample.name}",
01273 |                     "method": "md",
01274 |                     "source_run": str(dataset_dir),
01275 |                     "frame_index": source_sample.name,
01276 |                     "time_index": source_sample.name,
01277 |                     "displacement_amplitude": "",
01278 |                     "displacement_magnitude": "",
01279 |                     "displaced_atom": "",
01280 |                     "displacement_axis": "",
01281 |                     "displacement_sign": "",
01282 |                     "displacement_family": "",
01283 |                     "structure_path": str(structure_path),
01284 |                     "hamiltonian_path": str(hamiltonian_path or ""),
01285 |                     "output_path": str(run_out_path),
01286 |                     "run_out_path": str(run_out_path),
01287 |                     "metadata_path": sample_metadata.get("metadata_path", ""),
01288 |                     "valid": bool(structure_path.exists() and hamiltonian_path and run_out_path.exists()),
01289 |                     "validation_reason": "ok" if structure_path.exists() and hamiltonian_path and run_out_path.exists() else "missing_run_fdf_or_matrix_or_output",
01290 |                     "split": split_name,
01291 |                     "split_strategy": strategy,
01292 |                     "temporal_gap": str(temporal_gap),
01293 |                     "source_frame_index": source_sample.name,
01294 |                     "excluded_gap_reason": "",
01295 |                     "seed": sample_metadata.get("seed", ""),
01296 |                     "status": "completed" if structure_path.exists() and hamiltonian_path and run_out_path.exists() else "incomplete",
01297 |                     "sample_dir": str(sample_dir),
01298 |                     **recipe_manifest_fields(config, sample_index=source_sample.name),
01299 |                     **sample_metadata,
01300 |                 }
01301 |             )
01302 |         _write_manifest(split_root / f"{split_name}_manifest.csv", rows)
```

### `write_split_summary` — líneas 1305–1338

```py
01305 | def write_split_summary(
01306 |     split_root: Path,
01307 |     split_ranges: dict[str, list[Path]],
01308 |     excluded_samples: list[tuple[Path, str]],
01309 |     *,
01310 |     strategy: str,
01311 |     temporal_gap: int,
01312 |     warnings: list[str],
01313 | ) -> None:
01314 |     split_root.mkdir(parents=True, exist_ok=True)
01315 |     summary = {
01316 |         "strategy": strategy,
01317 |         "temporal_gap": temporal_gap,
01318 |         "counts": {split_name: len(samples) for split_name, samples in split_ranges.items()},
01319 |         "excluded_gap_count": len(excluded_samples),
01320 |         "excluded_gap_samples": [
01321 |             {
01322 |                 "sample_id": f"md_{sample.name}",
01323 |                 "frame_index": sample.name,
01324 |                 "excluded_gap_reason": reason,
01325 |             }
01326 |             for sample, reason in excluded_samples
01327 |         ],
01328 |         "warnings": warnings,
01329 |         "scientific_status": (
01330 |             "exploratory_temporal_leakage_risk"
01331 |             if strategy == "spread"
01332 |             else "temporal_gap_split" if strategy == "blocked_with_gap" else "blocked_split"
01333 |         ),
01334 |     }
01335 |     (split_root / "split_summary.json").write_text(
01336 |         json.dumps(summary, indent=2, sort_keys=True),
01337 |         encoding="utf-8",
01338 |     )
```

### `prepare_dataset_splits` — líneas 1341–1451

```py
01341 | def prepare_dataset_splits(config: dict) -> None:
01342 |     split_config = config.get("splits", {})
01343 |     if not bool(split_config.get("enabled", False)):
01344 |         return
01345 | 
01346 |     pipeline_paths = paths(config)
01347 |     steps_dir = pipeline_paths["dataset_dir"] / "MD_steps"
01348 |     split_root = pipeline_paths["dataset_dir"] / "splits"
01349 |     step_dirs = sorted(
01350 |         (path for path in steps_dir.iterdir() if path.is_dir() and path.name.isdigit()),
01351 |         key=lambda path: int(path.name),
01352 |     )
01353 |     total = md_total_steps(config)
01354 |     if len(step_dirs) < total:
01355 |         raise RuntimeError(
01356 |             f"Se esperaban {total} muestras MD, pero solo hay {len(step_dirs)} en {steps_dir}."
01357 |         )
01358 | 
01359 |     default_train, default_validation, default_test = _split_counts(total)
01360 |     train_count = int(split_config.get("train", default_train))
01361 |     validation_count = int(split_config.get("validation", default_validation))
01362 |     test_count = int(split_config.get("test", default_test))
01363 |     requested = train_count + validation_count + test_count
01364 |     if requested > len(step_dirs):
01365 |         raise RuntimeError(
01366 |             "El split MD pide mas muestras de las disponibles: "
01367 |             f"{requested} > {len(step_dirs)}."
01368 |         )
01369 | 
01370 |     if split_root.exists():
01371 |         shutil.rmtree(split_root)
01372 | 
01373 |     strategy = str(split_config.get("strategy", "blocked_with_gap")).strip().lower()
01374 |     # 30 frames @ 1 fs ≈ one carbon vibrational period (~20-40 fs); a 1-frame
01375 |     # gap does not decorrelate adjacent MD frames (audit finding C2). Explicit
01376 |     # splits.temporal_gap in the config still overrides this default.
01377 |     temporal_gap_default = 30 if strategy == "blocked_with_gap" else 0
01378 |     temporal_gap = int(split_config.get("temporal_gap", temporal_gap_default) or 0)
01379 |     if temporal_gap < 0:
01380 |         raise RuntimeError("splits.temporal_gap debe ser >= 0.")
01381 |     warnings: list[str] = []
01382 |     excluded_gap_samples: list[tuple[Path, str]] = []
01383 |     if strategy == "spread":
01384 |         warnings.append(SPREAD_SPLIT_WARNING)
01385 |         print(f"[WARN] {SPREAD_SPLIT_WARNING}")
01386 |         selected = _select_spread(step_dirs, requested)
01387 |         split_ranges = _split_spread(selected, train_count, validation_count, test_count)
01388 |     elif strategy == "block":
01389 |         selected = _select_spread(step_dirs, requested)
01390 |         split_ranges = _split_block(selected, train_count, validation_count, test_count)
01391 |     elif strategy == "blocked_with_gap":
01392 |         if temporal_gap <= 0:
01393 |             raise RuntimeError("splits.temporal_gap debe ser > 0 para blocked_with_gap.")
01394 |         counts = {"train": train_count, "validation": validation_count, "test": test_count}
01395 |         split_ranges, excluded_gap_samples = _split_blocked_with_gap(
01396 |             step_dirs,
01397 |             counts,
01398 |             temporal_gap=temporal_gap,
01399 |             block_order=_parse_block_order(split_config.get("block_order", "train,validation,test")),
01400 |         )
01401 |     else:
01402 |         raise RuntimeError(f"Estrategia de split MD no soportada: {strategy!r}.")
01403 |     for split_name, samples in split_ranges.items():
01404 |         for sample_dir in samples:
01405 |             _prepare_split_sample(sample_dir, split_root / split_name / sample_dir.name)
01406 |     split_sample_dirs = sorted(path for path in split_root.glob("*/*") if path.is_dir())
01407 |     materialized = _materialize_graph2mat_basis_files(pipeline_paths["dataset_dir"], split_sample_dirs)
01408 |     if materialized:
01409 |         print(f"[OK] Basis Graph2Mat materializada en splits MD: {materialized} enlaces/copias.")
01410 |     write_split_manifests(config, split_root, split_ranges, strategy=strategy, temporal_gap=temporal_gap)
01411 |     if excluded_gap_samples:
01412 |         write_excluded_gap_manifest(
01413 |             config,
01414 |             split_root,
01415 |             excluded_gap_samples,
01416 |             strategy=strategy,
01417 |             temporal_gap=temporal_gap,
01418 |         )
01419 |     write_split_summary(
01420 |         split_root,
01421 |         split_ranges,
01422 |         excluded_gap_samples,
01423 |         strategy=strategy,
01424 |         temporal_gap=temporal_gap,
01425 |         warnings=warnings,
01426 |     )
01427 |     print(
01428 |         "[OK] Split MD preparado: "
01429 |         f"{train_count} train, {test_count} test, {validation_count} validation "
01430 |         f"en {split_root} (strategy={strategy})"
01431 |     )
01432 |     artifact_validation_path = pipeline_paths["dataset_dir"] / "artifact_validation.json"
01433 |     if artifact_validation_path.exists():
01434 |         dataset_manifest, frozen_split = write_benchmark_manifests(
01435 |             dataset_root=pipeline_paths["dataset_dir"],
01436 |             split_root=split_root,
01437 |             generation_mode="clean_one_pass",
01438 |             strict_paper_ready_provenance=True,
01439 |         )
01440 |         print(
01441 |             "[OK] Benchmark dataset congelado: "
01442 |             f"{dataset_manifest['benchmark_dataset_id']} split_hash={frozen_split['split_hash']}"
01443 |         )
01444 |     else:
01445 |         print(
01446 |             "[INFO] No se escribe benchmark_dataset_manifest.json: "
01447 |             f"no existe {artifact_validation_path}."
01448 |         )
01449 |     print(f"[INFO] MD train samples: {_sample_names(split_ranges['train'])}")
01450 |     print(f"[INFO] MD test samples: {_sample_names(split_ranges['test'])}")
01451 |     print(f"[INFO] MD validation samples: {_sample_names(split_ranges['validation'])}")
```
