
## GOAL

Auditar de forma crítica y honesta TODO lo que se implementó para el cross testing de la
bicapa grafeno/hBN contra su versión rotada (moiré), commiteado en `main` en:

- `81cc26b` — pipeline bicapa→moiré (bundles, builders, payloads, fix runner, UI, tests, doc)

El objetivo NO es confirmar que "funciona el flujo" (ya se probó un smoke end-to-end), sino
determinar si los **resultados serían científicamente válidos y reproducibles**, si el
**código es correcto y no introduce regresiones**, y si la **procedencia es honesta**. Trata
esto como un revisor de paper hostil: busca lo que está mal, exagerado, o no defendible.

Entregable: un informe priorizado (Crítico / Alto / Medio / Bajo) donde cada hallazgo lleve
(a) archivo:línea o comando que lo evidencia, (b) por qué importa, (c) fix mínimo propuesto.
Termina con un veredicto claro: ¿es esto publishable, exploratorio, o roto?

---

## ALCANCE (los artefactos a auditar)

Código nuevo:
- `Comparison/scripts/build_graphene_hbn_moire_target.py` (458 líneas) — builder del target moiré
- `Comparison/scripts/build_graphene_hbn_bilayer_train_dataset.py` (302 líneas) — merge de 3 apilamientos
- `Comparison/scripts/g2m_deeph_runner.py` — el helper nuevo `_bind_graph2mat_material_to_dataset`
  (def ~línea 1030, llamada ~línea 3233) y su integración en `_prepare_graph2mat_context`
- `Comparison/scripts/ops/build_cross_predict_metrics_payload.py` — soporte `--bilayer-*`
- `Comparison/scripts/pipeline_ui.py` — runner + rutas `/api/cross-testing/bilayer/*`
- `Comparison/ui/app.js`, `Comparison/ui/index.html` — subsección `ct-bilayer`

Config/datos:
- `materials/graphene_hBN_{AA,AB1,AB2}/` (bundles) + `materials/graphene_hBN_common/` (symlinks)
- `Comparison/config/graphene_hbn_*_payload.json` + `graphene_hbn_bilayer_md_pipeline_config.yaml`
- Datasets generados (gitignored): `Comparison/datasets/graphene_hBN_{AA,AB1,AB2}_md30`,
  `graphene_hBN_bilayer_train`, `graphene_hBN_moire_22deg`
- Resultados smoke: `Comparison/results/ml_vs_siesta_cross_structure_bilayer_moire/`

Tests: `tests/test_graphene_hbn_bilayer_train_dataset.py`, `tests/test_graphene_hbn_moire_target.py`

---

## EJES DE AUDITORÍA (ataca cada uno; son donde está el riesgo real)

### 1. VALIDEZ FÍSICA DEL TARGET MOIRÉ (prioridad máxima — probablemente el punto más débil)
El provenance de `graphene_hBN_moire_22deg` declara:
`approximation = "rigid commensurate-angle twist of hBN on the shared graphene lattice; true
incommensurate moire is NOT resolved"`, `effective_hBN_strain_percent = 1.8`,
`twist_angle_deg ≈ 21.79`, `commensurate_index = (1,2)`, `num_atoms = 16`, `relaxed = false`.

Cuestiona con dureza:
- ¿Es 21.79° un ángulo de twist **físicamente interesante** o solo el que sale de forzar
  conmensurabilidad (1,2) en una celda de 16 átomos? Un moiré real de interés está a ángulos
  pequeños (~1–5°). Verifica si el ángulo es un artefacto de la aproximación, no una elección
  científica.
- El **1.8% de strain impuesto al hBN** para hacerlo conmensurado con el grafeno: ¿distorsiona
  el Hamiltoniano de referencia hasta el punto de que "predecir el moiré" mide sobre todo la
  respuesta del modelo al strain, no a la rotación? ¿Es esto lo que el usuario quería medir?
- ¿La geometría generada es realmente una bicapa con las dos capas en la misma celda, o el
  builder aplicó la rotación de forma que rompe el registro entre capas / solapa átomos?
  Comprueba distancias interatómicas mínimas en la fdf generada (no debe haber átomos a <0.8 Å).
- `relaxed=false`: ¿es defendible usar una geometría sin relajar (con strain) como referencia
  SIESTA de test? ¿O el H de referencia es de una estructura que físicamente no existiría?
- Reproduce el cálculo del ángulo y del strain desde `build_graphene_hbn_moire_target.py` y
  confirma que los números del provenance salen de la geometría real, no hardcodeados.

Concluye explícitamente: ¿este target mide "generalización a bicapa rotada" o mide otra cosa?

### 2. CORRECCIÓN DEL MERGE (dataset combinado de 3 apilamientos)
- Verifica que `graphene_hBN_bilayer_train` contiene EXACTAMENTE los snapshots de los 3
  datasets md30, sin duplicados ni pérdidas (cuenta por apilamiento vs total).
- **Leakage de splits**: los 3 md30 se generaron con `blocked_with_gap` y su propio train/val.
  Al fusionar, ¿puede un snapshot de validación de un apilamiento acabar en train del pool (o
  viceversa) de forma que rompa la independencia? ¿El gap temporal se respeta dentro de cada
  apilamiento tras el reindexado?
- Confirma que los hashes de base/pseudo son idénticos entre los 3 (el merge dice fallar
  cerrado si difieren): fuerza el camino de fallo y comprueba que realmente aborta.
- ¿La procedencia registra los 3 datasets fuente + hashes, o solo dice "mezcla"? Verifica que
  es auditable a posteriori.

### 3. EL FIX DEL RUNNER — RIESGO DE REGRESIÓN (ruta compartida)
`_bind_graph2mat_material_to_dataset` se inyecta en `_prepare_graph2mat_context`, que es la
ruta de preparación de entrenamiento G2M para **TODOS** los materiales, no solo la bicapa.
- Confirma que para datasets con `material.preset` legítimo (grafeno, h2o, si, w90) el bind
  NO cambia el comportamiento previo: si `material_provenance.json` trae `basis_dir`, ¿se
  sobreescribe el material del template con uno equivalente o con uno distinto? Compara el
  `config.yaml` de G2M generado ANTES (git stash del fix) y DESPUÉS para un dataset grafeno
  existente. Si difiere, es una regresión.
- El slug de label (`re.sub`) puede colisionar dos labels distintos en el mismo slug. ¿Importa
  aguas abajo (nombres de checkpoint, manifests)?
- El fallback de `fdf`/`pseudopotential_dir` vía `source_dataset_root`: ¿qué pasa si el source
  ya no existe en disco (dataset movido/borrado)? ¿Falla ruidoso o silencioso?
- Ejecuta la suite completa de `tests/` (no solo los 2 nuevos) y confirma 0 regresiones. En
  especial `tests/test_g2m_deeph_runner.py`, `tests/test_cross_structure_sweep.py`.

### 4. LOS NÚMEROS DE MAE — ¿SMOKE O ROTO? (sanity científico)
El smoke reportó `h_mae_eV`: Graph2Mat=0.856, DeepH=1.056. Para un Hamiltoniano en eV, **~1 eV
de MAE es enorme** (los sweeps de grafeno del repo rondan 1e-2–5e-2 eV). Investiga si es solo
"10 épocas + 66 snapshots" o si hay un bug real:
- Compara con el MAE a época 1 de un training grafeno del repo: ¿0.85 eV es consistente con
  "poco entrenado" o está fuera de escala?
- ¿La evaluación compara H_pred vs H_ref en la **misma base y con el mismo orden de átomos/
  orbitales**? El target moiré tiene 16 átomos y 3 especies; el pool tiene 4 átomos. Verifica
  que el mapping de bloques orbital→orbital no está desalineado (esto ya mordió a DeepH en el
  repo: ver la nota de memoria sobre desalineamiento de base DeepH).
- ¿El MAE se calcula sobre elementos de H reales presentes, o incluye ceros/padding que lo
  inflarían o desinflarían artificialmente?
- Sube epochs a, p.ej., 100 en una corrida corta (si hay GPU) y mira si el MAE baja de forma
  coherente. Si NO baja apreciablemente, hay algo estructural mal, no falta de entrenamiento.

### 5. COMPATIBILIDAD Y CONTRATO DEL CROSS-SWEEP
- El planner marcó el par compatible. Reconfirma POR QUÉ: especies/base/pseudos idénticos.
  ¿Se comparó k-grid por espaciado recíproco correctamente tras el ajuste 20×20→10×10 que se
  hizo en el builder? Verifica que ese ajuste de k-mesh no degrada la referencia SIESTA del
  moiré (menos k-points = menos preciso). ¿Es 10×10×1 suficiente para 16 átomos?
- El `payload_id` real es larguísimo
  (`graphene_hBN_bilayer__graphene_hBN_bilayer_train__to__graphene_hBN_moire__graphene_hBN_moire_22deg`).
  ¿Rompe algo (límites de longitud de ruta, colisiones de basename, el gotcha conocido de
  "Unknown case ids" si un predict_metrics vacío clobbea el summary)?

### 6. UI — AISLAMIENTO Y CORRECCIÓN
- Confirma que la subsección `ct-bilayer` es puramente ADITIVA: el resto de la pestaña Cross
  testing (sweep normal, vacancy) se comporta idéntico. Diffea el HTML/JS para asegurar que no
  se tocaron los binds existentes.
- Las 3 rutas `/api/cross-testing/bilayer/*` reusan `CrossStructureSweepRunner` con otro
  output-root. Verifica que no comparten estado mutable con los runners de sweep/vacancy
  (nada de globals compartidos que se pisen).
- La curva se pinta leyendo `payload.curves[].points[]`. Con 1 solo punto (1 par, 1 tamaño),
  ¿la gráfica MAE-vs-snapshots tiene sentido, o una curva de 1 punto es engañosa? Reporta si la
  visualización comunica bien "esto es un único punto exploratorio".

### 7. REPRODUCIBILIDAD Y PROCEDENCIA
- ¿Se puede regenerar TODO desde cero con los comandos documentados (bundles → 3 md30 → merge →
  moiré → train)? Sigue el `Comparison/scripts/ops/smoke_bilayer_moire_pipeline.sh` y anota
  cualquier paso que no sea reproducible o que dependa de estado no commiteado.
- Seeds: ¿el smoke fija seed? ¿Una segunda corrida da el mismo MAE? Si no, la comparación
  G2M vs DeepH sobre 1 seed no es concluyente (el repo exige ≥3 seeds para claims).
- ¿Los datasets/resultados son gitignored (no se commitearon)? Confirma que el commit no metió
  binarios pesados por error.

---

## MÉTODO

- Ejecuta comprobaciones reales (lee ficheros, corre `python`, `pytest`, inspecciona geometrías,
  golpea endpoints), no audites de memoria.
- Para el eje 3 (regresión), usa `git stash`/comparación real del `config.yaml` generado, no
  razonamiento a ojo.
- Cita evidencia concreta (archivo:línea, salida de comando) en cada hallazgo.
- Distingue "esto está mal" (bug) de "esto es una limitación asumida y documentada" (aceptable
  si el provenance es honesto) de "esto se vende como más de lo que es" (deshonestidad de
  procedencia — trátalo como Alto/Crítico).

## RESTRICCIONES
- NO apliques fixes en este turno. Solo audita y reporta. Propón el fix mínimo por hallazgo.
- NO detengas ni interfieras con procesos en curso (hay sweeps de derivadas y de vacancy seeds
  corriendo; no toques sus procesos ni sus dirs de resultados).
- NO reinicies la UI ni mates procesos.
- Sé honesto sobre lo que NO pudiste verificar (p.ej. si algo requiere una corrida larga).

## VEREDICTO FINAL (obligatorio)
Cierra con una clasificación explícita del pipeline bicapa→moiré:
- **Publishable** (resultados defendibles en paper), o
- **Exploratorio** (el flujo es correcto pero los resultados/target no son claim-grade), o
- **Roto** (hay un bug que invalida las métricas),
y la lista corta de qué habría que arreglar para subir de categoría.
