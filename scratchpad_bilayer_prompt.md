# MEGA-PROMPT (modo GOAL) — Pipeline "Bicapa grafeno/hBN → moiré rotado" (cross testing)

> Pégale esto tal cual a Claude Code en este repo. Está en modo *goal*: describe el
> resultado y las restricciones, no un guion línea a línea. Reutiliza al máximo la
> maquinaria existente; casi todo ya está construido.

---

## GOAL

Implementar, end-to-end y reflejado en la UI, un pipeline nuevo que:

1. Genera snapshots de dinámica molecular (SIESTA MD) para las **3 configuraciones de
   apilamiento** de la bicapa grafeno/hBN que ya existen en el repo:
   - `materials/graphene_hBN_AA.fdf`
   - `materials/graphene_hBN_AB1.fdf`
   - `materials/graphene_hBN_AB2.fdf`
2. Combina los snapshots de los 3 apilamientos en **UN ÚNICO dataset de entrenamiento**
   (train+validation) y entrena sobre él **un modelo Graph2Mat y un modelo DeepH**.
3. Construye un **target de test independiente**: una **supercelda moiré de la bicapa
   rotada** (twist entre capas → N ≫ 4 átomos), con referencias SIESTA propias (split
   `test`).
4. Hace **cross testing** de esos modelos (entrenados en la bicapa no rotada) contra el
   moiré rotado, produciendo curvas MAE (y Frobenius) por modelo — exactamente el mismo
   tipo de análisis que el cross testing actual, pero bicapa→bicapa-rotada.
5. Expone TODO esto en la pestaña **"Cross testing"** de la UI como una **subsección
   nueva e independiente al final de la pestaña**, sin tocar nada del resto de la
   sección.

Decisiones de diseño ya fijadas por el usuario (no re-preguntar):
- **Target = twist/moiré (supercelda grande, N≫4 átomos)**, no rotación rígida de la celda.
- **Dataset de entrenamiento = uno solo, combinando AA+AB1+AB2** → 1 curva por modelo en
  el cross-test (source combinado → target moiré).

---

## PRINCIPIO RECTOR: REUTILIZAR, NO REINVENTAR

El motor de cross-structure sweep **es totalmente agnóstico de material** — nada
hardcodea grafeno más allá de rutas de dataset y labels. Reutiliza SIN MODIFICAR:

- `Comparison/scripts/ml_vs_siesta/cross_structure_sweep.py` (planner + orquestador)
- `Comparison/scripts/ml_vs_siesta/cross_structure_materialize.py` (compone source→target)
- `Comparison/scripts/run_cross_structure_sweep_payload.py` (runner CLI de payloads)
- `Comparison/scripts/g2m_deeph_runner.py` (`Graph2MatDeepHBenchmarkRunner`, incl. modo
  `predict_metrics_only` con `existing_model_artifacts`)
- `MD/scripts/generate_md_dataset.py` (genera dataset MD desde un bundle de material)
- `shared/material_presets.py` / `shared/material_bundle.py` / `shared/fdf_materialization.py`
- El validador de artefactos conjunto (`joint_artifact_contract`) y los manifests
  (`benchmark_manifest`).

**El contrato source→target del cross-sweep** (respétalo, no lo cambies):
- `train` ← split train del/los *source*; `validation` ← validation del source;
  `test` ← split test del *target*. IDs con prefijo de rol: `source_train__<id>`,
  `source_validation__<id>`, `target_test__<id>`.
- El planner **falla cerrado** si difieren: especies reales, hashes de base, hashes de
  pseudopotenciales, espacios ghost activos, ajustes DFT bloqueantes, contrato conjunto
  G2M/DeepH, o semántica de Hamiltoniano incompleto (bypass explícito con
  `confirm_incomplete_hamiltonian_semantics=true`).
- **SÍ pueden diferir**: nº de átomos, dimensiones de celda, vectores de red, dims del H
  crudo, system label, enteros MP (k-grids se comparan por espaciado recíproco).
  → Por eso el moiré (más átomos, otra celda) es compatible **siempre que** comparta
  especies + base + pseudos con los 3 apilamientos. Esto es una **precondición dura**:
  el moiré DEBE usar exactamente los mismos ficheros de base (`C.ion.xml`, `B.ion.xml`,
  `N.ion.xml`) y pseudos (`C.psf` de `materials/graphene/pseudos`, `B.psml`/`N.psml` de
  `materials/bn/pseudos`) y el mismo bloque `PAO.Basis` que las fdf de apilamiento.

`payload_id` se autoderiva como `<source_id>__to__<target_id>`; las curvas se agrupan por
`(source_system_label, target_id, model)`, con x = nº de snapshots de train del source.

---

## PLAN DE TRABAJO (fases)

### FASE 0 — Bundles de material para las 3 fdf de apilamiento

Las 3 fdf están sueltas en `materials/` pero el pipeline consume **bundles**
(`materials/<preset>/material.yaml` + fdf). Crea 3 bundles nuevos, uno por apilamiento,
espejo de `materials/graphene_5x5/material.yaml`:

```
materials/graphene_hBN_AA/   material.yaml + RUN.fdf   (mueve/copia graphene_hBN_AA.fdf → RUN.fdf)
materials/graphene_hBN_AB1/  material.yaml + RUN.fdf
materials/graphene_hBN_AB2/  material.yaml + RUN.fdf
```

`material.yaml` (por apilamiento), con `structure_type: crystal`. Ojo: la bicapa tiene
**3 especies (C, B, N)** cuyos pseudos/base viven en **dos carpetas distintas**
(`materials/graphene/{pseudos,basis}` para C, `materials/bn/{pseudos,basis}` para B y N).
Revisa cómo `resolve_material_bundle` / `MaterialBundle` esperan `pseudopotential_dir` y
`basis_dir` (¿una sola carpeta?). Si el bundle solo admite UN `pseudopotential_dir`,
resuelve por la vía más simple: crea `materials/graphene_hBN_common/{pseudos,basis}` con
enlaces/symlinks a `C.psf`, `B.psml`, `N.psml`, `C.ion.xml`, `B.ion.xml`, `N.ion.xml`, y
apunta los 3 `material.yaml` a esa carpeta común. Valida con `material_bundle` que la
cobertura de pseudopotenciales para las 3 especies es correcta antes de seguir.

> `ponytail`: no inventes un bundle multi-carpeta si el formato ya soporta una carpeta
> con los 6 ficheros. La carpeta común es la solución más corta.

**Check de fase**: `python3 -c "from material_presets import resolve_material_bundle; ..."`
resuelve cada preset y el validador de bundle reporta las 3 especies con pseudos+base.

### FASE 1 — Datasets MD por apilamiento (SOLO payload, cero código nuevo)

Genera 3 datasets MD (uno por apilamiento). **No hace falta código nuevo de pipeline**: el
motor es agnóstico de material. El camino es un payload por material → `Graph2MatDeepHBenchmarkRunner`
→ que expande `dataset_sweep.recipes[].blocks[]` en `temperature_blocks` y llama a
`MD/scripts/generate_md_dataset.py`, el cual:
- genera `md_store.lua` solo (vía `graph2mat siesta md setup-store` — **no** copies un lua a
  mano, se crea solo),
- corre SIESTA MD por bloque de temperatura, combina los `MD_steps/`, y arma
  `splits/{train,validation,test}` + manifests.

Plantillas: `configs/config_md.yaml`, `MD/pipeline_config.yaml`, y cualquier
`*_snapshot_scaling_*_payload.json` de `Comparison/config/`. El material se elige con
`material.preset: graphene_hBN_AA` (o `material_preset`). El nº de snapshots se fija en los
`dataset_sweep.recipes[].blocks[]` (`n_snapshots`, `temperature_K`, `seed`). Empieza pequeño
para smoke (~30–60 snapshots por apilamiento; deja el tamaño como parámetro). Driver más simple
para correr un payload de punta a punta: `Comparison/scripts/run_g2m_deeph_payload_once.py <payload>`.

**Gotcha física (3 especies)**: DeepH necesita la máscara orbital correcta para B y N (la
default de carbono no vale). Esto ya se resuelve solo en
`deeph_config.deeph_orbital_list_from_siesta_sample`, que la deriva de `*.ORB_INDX` +
`*.STRUCT_OUT` — **siempre que** la RUN.fdf renderizada emita `Write.OrbitalIndex T`,
`XML.Write T` y `SaveHS T`. Las 3 fdf ya los traen; verifica que `ensure_required_output_flags`
no los pisa. No toques la máscara a mano.

- Cada dataset produce el layout estándar: `splits/{train,validation}`,
  `material_provenance.json`, `frozen_split_manifest.json`, `benchmark_dataset_manifest.json`,
  `material_basis/`, pseudos. Ubícalos en `Comparison/datasets/graphene_hBN_<config>_mdN...`.

**Check de fase**: los 3 datasets pasan `validate_dataset` (joint artifact contract), tienen
splits no vacíos de train/validation, y en los snapshots existen `*.HSX`, `*.ORB_INDX`,
`*.STRUCT_OUT` para las 3 especies.

### FASE 2 — Combinar los 3 apilamientos en un dataset único de entrenamiento

`generate_md_dataset.py` resuelve **un** bundle por dataset, así que "combinar los 3" NO es
nativo. Es la **única pieza de lógica realmente nueva**. Impleméntala como el helper más
pequeño que funcione:

- Nuevo script `Comparison/scripts/build_graphene_hbn_bilayer_train_dataset.py` que toma los
  3 datasets MD de la Fase 1 y **fusiona sus snapshots** en un dataset compuesto
  `Comparison/datasets/graphene_hBN_bilayer_train/` con un split `train`/`validation`
  reindexado (prefija los sample ids por apilamiento para no colisionar). Reutiliza
  `read_dataset_samples`, la escritura de manifests (`write_benchmark_manifests`), y el
  validador conjunto — igual que hace `build_graphene_5x5_vacancy_target.py`. NO reejecutes
  SIESTA: copia/enlaza los artefactos H/S/RUN.out ya generados.
- Precondición: los 3 apilamientos comparten especies/base/pseudos (lo garantiza la Fase 0),
  así que la mezcla es legal para el contrato conjunto. Verifica hashes de base/pseudo
  iguales entre los 3 antes de fusionar; falla cerrado si difieren.
- `material_provenance.json` del compuesto: label `graphene_hBN_bilayer`, y registra en
  provenance que es una **mezcla de 3 apilamientos** (lista los 3 datasets fuente y sus
  hashes). Esto es honestidad de procedencia, no opcional.

> `ponytail`: reusa el patrón exacto de `build_graphene_5x5_vacancy_target.py`
> (`_copy_basis`, `_write_test_manifest`, `write_benchmark_manifests`,
> `validate_dataset`). El "merge" es copiar sample dirs + reindexar splits + reescribir
> manifests. Nada de un framework de datasets nuevo.

**Check de fase**: `validate_dataset` OK sobre el compuesto; `read_dataset_samples` devuelve
la suma de snapshots de los 3 con splits train/validation coherentes. Deja un
`test_*.py` mínimo (assert-based) espejo de `tests/test_graphene_vacancy_target.py` que
verifique: nº de samples = suma de los 3, sin colisión de ids, hashes de base idénticos.

### FASE 3 — Entrenar G2M + DeepH sobre el dataset combinado

Entrena 1 Graph2Mat y 1 DeepH sobre `graphene_hBN_bilayer_train` reutilizando el mismo
camino de entrenamiento que el resto (`Graph2MatDeepHBenchmarkRunner` vía
`run_g2m_deeph_payload_once.py <payload>`, o un payload de snapshot-scaling con un solo tamaño
apuntando `dataset_root` al compuesto de la Fase 2). Bajo el capó: Graph2Mat lee los snapshots
SIESTA en sitio (fit vía `MD/scripts/run_md_training.py` → `graph2mat models mace main fit`);
DeepH construye su mirror raw + INIs con `Comparison/scripts/deeph_config.py` y corre
`deeph-preprocess`/`deeph-train`. Copia hiperparámetros de un payload existente (p.ej. los
`hyperparams.graph2mat` / `hyperparams.deeph` del
`graphene_w90_5x5_to_vacancy_predict_metrics_payload.json`). Deja los epochs como parámetro
(bajo para smoke).

Guarda los checkpoints en rutas estables y anótalas: necesitarás
`graph2mat_training_dir` (con `checkpoint_manifest.json` + `*.ckpt`) y `deeph_save_dir`
(con `config.ini` + `best_state_dict.pkl`) para el `existing_artifacts` de la Fase 5.

**Check de fase**: existen `checkpoint_manifest.json` (G2M) y `best_state_dict.pkl` (DeepH)
en los dirs esperados.

### FASE 4 — Target moiré (bicapa rotada) — split de test

Construye el dataset target: **supercelda moiré commensurada** de la bicapa con twist entre
la capa de grafeno y la de hBN. Nuevo script
`Comparison/scripts/build_graphene_hbn_moire_target.py`, modelado sobre
`build_graphene_5x5_vacancy_target.py` (misma estructura: genera geometría → `run_siesta`
estático por snapshot → `validate_snapshot`/`validate_dataset` → manifests → split `test`
frozen). Diferencias específicas del moiré:

- **Geometría**: parte de una de las fdf de apilamiento (celda primitiva 4 átomos) y
  construye una supercelda commensurada aplicando un twist. Parametriza el ángulo/índice de
  commensurabilidad `(m, n)` (ángulo de twist → tamaño de supercelda). Reutiliza
  `extract_fdf_structure` + `materialize_fdf_text` para escribir la fdf resultante (nuevos
  vectores de red + posiciones + species del supercell).
- **AVISO FÍSICO que debes dejar explícito en el código y en la doc**: grafeno y hBN tienen
  ~1.8% de desajuste de red, así que un moiré con twist arbitrario es **incommensurado**;
  una celda periódica finita exige o bien un **ángulo commensurado específico** o bien un
  **approximant con deformación (strain) pequeña**. Expón un parámetro `--commensurate-angle`
  o `--approximant (m,n)` y **documenta la aproximación usada** (strain aplicado, ángulo
  efectivo) en `material_provenance.json`. No lo escondas: es exactamente el tipo de
  calibración física que un modelo mínimo no ve.
  > `ponytail`: deja la perilla de commensurabilidad/strain, no solo menos código.
- **Especies/base/pseudos**: idénticos a los apilamientos (precondición de compatibilidad).
- **Snapshots**: puedes generar un pequeño MD del moiré o unos pocos static references
  (empieza con `--limit` pequeño). Solo se necesita split `test`.
- Salida en `Comparison/datasets/graphene_hBN_moire_<ang>/` con
  `frozen_split_manifest.json`, `material_provenance.json`, `benchmark_dataset_manifest.json`,
  `artifact_validation.json`, split `test`.

**Check de fase**: `validate_dataset` OK; el planner del cross-sweep (dry/preview) marca el
par `graphene_hBN_bilayer → graphene_hBN_moire_<ang>` como **compatible** (si sale
`incompatible`, el motivo estará en especies/base/pseudos → corrígelo, no lo bypassees a la
ligera). Deja un `test_*.py` mínimo que compruebe: N_átomos del moiré = el esperado para
`(m,n)`, especies = {C,B,N}, hashes de base == los del dataset de train.

### FASE 5 — Payload de cross testing (predict_metrics) bicapa→moiré

Crea el payload de campaña, espejo de
`Comparison/config/graphene_w90_5x5_to_vacancy_predict_metrics_payload.json`:

`Comparison/config/graphene_hbn_bilayer_to_moire_predict_metrics_payload.json`

- `schema: "g2m_deeph_cross_structure_predict_metrics_payload_v1"`, `action: "predict_metrics"`.
- `pairs`: **un solo par** `{ "source": "Comparison/datasets/graphene_hBN_bilayer_train",
  "target": "Comparison/datasets/graphene_hBN_moire_<ang>", "direction": "bilayer_to_moire" }`.
  (Dataset de train único → 1 curva por modelo, como pidió el usuario.)
- `models: ["graph2mat", "deeph"]`, `seed`/`seeds` como el vacancy.
- `existing_artifacts`: keyed por el **basename del dir source** (`graphene_hBN_bilayer_train`)
  → `{ "graph2mat_training_dir": "<...>/graph2mat/training", "deeph_save_dir": "<...>/deeph/train" }`
  apuntando a los checkpoints de la Fase 3.
- Hereda `hyperparams`, `early_stopping`, `performance`, `notes` de un payload existente.
- Reusa `ops/build_cross_predict_metrics_payload.py` como referencia de cómo se genera el
  payload de forma reproducible (idealmente añade una función/flag análoga para el bilayer en
  ese script, para no editar el JSON a mano).

**Check de fase**: `run_cross_structure_sweep_payload.py <payload> --action preview` planea 1
permutación compatible. Luego `--action predict_metrics --output-root
Comparison/results/ml_vs_siesta_cross_structure_bilayer_moire` produce
`cross_structure_sweep_summary.json` con `records[]` que tienen `h_mae_eV` para g2m y deeph,
`payload_id = graphene_hBN_bilayer__to__graphene_hBN_moire_<ang>`.

### FASE 6 — UI: subsección nueva e independiente al final de "Cross testing"

**Restricción dura del usuario**: NO tocar nada del resto de la pestaña. Solo AÑADIR una
subsección nueva al final. Replica exactamente el patrón de la **subsección de vacante**, que
ya es un flujo independiente con su propio runner backend.

Anatomía a copiar (la subsección de vacante es la plantilla 1:1):

- **Backend** (`Comparison/scripts/pipeline_ui.py`):
  - Añade una constante de output root nueva, p.ej.
    `CROSS_TESTING_BILAYER_OUTPUT_ROOT = RESULTS_ROOT / "ml_vs_siesta_cross_structure_bilayer_moire"`
    (junto a `CROSS_TESTING_SWEEP_OUTPUT_ROOT` / `CROSS_TESTING_VACANCY_OUTPUT_ROOT`).
  - Instancia un **tercer runner**: `CROSS_TESTING_BILAYER_RUNNER =
    CrossStructureSweepRunner(CROSS_TESTING_BILAYER_OUTPUT_ROOT)` (misma clase; solo cambia
    el dir de salida — igual que el vacancy runner).
  - Añade rutas nuevas, calcadas de las de vacancy, con prefijo `bilayer`:
    `POST /api/cross-testing/bilayer/launch` → `CROSS_TESTING_BILAYER_RUNNER.start(payload)`
    `GET  /api/cross-testing/bilayer/status` → `.status()`
    `GET  /api/cross-testing/bilayer/metrics` → `.metrics()`
    (Opcional, si quieres el PlotMatrixError como en vacancy: rutas
    `bilayer/matrix-errors` y `bilayer/matrix-error` reusando
    `vacancy_matrix_error_*` generalizado por output-root.)
  - El input de payload debe forzar `.json` bajo `Comparison/config` (reusa
    `_cross_testing_resolve_body`), apuntando por defecto al payload de la Fase 5.
  - NO modifiques las rutas ni runners existentes de `sweep`/`vacancy`.

- **HTML** (`Comparison/ui/index.html`): inserta una `<section class="panel">` nueva
  **entre la línea 1863 (cierre de la subsección de vacante) y la 1864 (cierre de
  `#view-cross-testing`)**. Cópiala de la subsección vacante (1821–1863), cambiando ids
  `ct-vacancy-*` → `ct-bilayer-*`, textos, y el valor por defecto del input al payload de la
  Fase 5. Reutiliza las mismas clases genéricas (`.panel`, `.panel-heading`, `.eyebrow`,
  `.field`, `.button-row`, `.terminal-frame` + `.log-output mix-payload-log`,
  `.plot-card full`). Copy sugerido: eyebrow "Checkpoints existentes · bicapa→moiré",
  h3 "Cross testing bicapa grafeno/hBN → moiré rotado".

- **JS** (`Comparison/ui/app.js`): en el bloque "Cross testing view" (desde 13456), duplica
  el conjunto de funciones `ctVacancy*` como `ctBilayer*` (`ctBilayerPreview`,
  `ctBilayerEvaluate`/`launch`, `ctBilayerPollStatus`, `ctBilayerLoadMetrics`,
  `ctBilayerRenderChart`), apuntando a las rutas `/api/cross-testing/bilayer/*` y a los ids
  `ct-bilayer-*`. Engánchalas en `setupCrossTesting()` (14113–14161) SIN tocar los binds
  existentes. Render de la curva: mismo contrato que el resto — lee `payload.curves[].points[]`
  con `x`, `mae`, `mae_std`, `relative_frobenius`; plotea meV vs snapshots de train,
  1 curva por modelo (g2m, deeph). Target al `<div id="ct-bilayer-mae-chart" class="plot-card full">`.

> `ponytail`: la subsección de vacante ya prueba que "segundo runner + rutas paralelas +
> ids `ct-*` propios + `<section class=panel>`" es el patrón bendecido. NO abstraigas un
> "framework de subsecciones"; duplica el patrón vacancy con prefijo `bilayer`. Es el diff
> más corto y aislado.

**Check de fase**: arranca la UI (`python3 Comparison/scripts/pipeline_ui.py`), abre la
pestaña "Cross testing", baja al final → aparece la subsección bicapa; "Previsualizar" planea
1 par compatible; "Cargar métricas" pinta 1 curva g2m + 1 deeph de MAE vs snapshots. El resto
de la pestaña queda idéntico (verifícalo).

---

## VERIFICACIÓN GLOBAL (end-to-end, no solo tests)

Ejecuta el flujo real de punta a punta con tamaños pequeños (smoke) y observa el
comportamiento, no solo que pasen los tests:

1. Fase 0–2 → dataset combinado válido.
2. Fase 3 → checkpoints g2m+deeph presentes.
3. Fase 4 → target moiré válido y **compatible** en `preview`.
4. Fase 5 → `predict_metrics` genera `records[]` con `h_mae_eV` finitos para ambos modelos.
5. Fase 6 → la subsección de la UI muestra la curva y el resto de "Cross testing" intacto.

Deja el smoke reproducible como un script `Comparison/scripts/ops/` (o un `Makefile`/target)
que encadene las fases con parámetros pequeños, espejo de los `ops/*` existentes.

---

## ENTREGABLES

- `materials/graphene_hBN_{AA,AB1,AB2}/` (bundles) + carpeta común de pseudos/base.
- `Comparison/scripts/build_graphene_hbn_bilayer_train_dataset.py` (+ test mínimo).
- `Comparison/scripts/build_graphene_hbn_moire_target.py` (+ test mínimo).
- `Comparison/config/graphene_hbn_bilayer_to_moire_predict_metrics_payload.json`
  (+ soporte en `ops/build_cross_predict_metrics_payload.py`).
- Cambios ADITIVOS en `pipeline_ui.py`, `index.html`, `app.js` (subsección `bilayer`).
- Doc: sección nueva en `docs/cross_structure_evaluation.md` describiendo la variante
  bicapa→moiré (contrato de splits idéntico; nota física de commensurabilidad/strain).
- Script de smoke end-to-end en `Comparison/scripts/ops/`.

## RESTRICCIONES / GOTCHAS (no negociables)

- Reutiliza el motor de cross-sweep **sin modificarlo**; es agnóstico de material.
- El moiré DEBE compartir especies+base+pseudos con los apilamientos, o el planner lo marca
  `incompatible`. No uses `confirm_incomplete_hamiltonian_semantics` para tapar un mismatch
  real de base/pseudo; eso sería falsear la comparación.
- Honestidad de procedencia: registra en `material_provenance.json` la mezcla de 3
  apilamientos (Fase 2) y la aproximación de commensurabilidad/strain del moiré (Fase 4).
- La UI: solo AÑADIR al final de la pestaña "Cross testing". Cero cambios al resto.
- `payload_id` se autoderiva; no lo hardcodees en la UI: léelo del summary/metrics como hace
  el resto (gotcha conocido: un `predict_metrics` vacío puede clobbear el summary y dar
  "Unknown case ids" — asegúrate de que el runner escribe `records[]` no vacíos).
- Deja checks runnables (assert-based) en cada script no trivial; smoke pequeño antes de
  cualquier corrida grande.
