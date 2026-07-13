# GOAL: pestaña "Cross testing" en la UI — configurar, lanzar y graficar en vivo sweeps de cross-structure (MAE vs dataset de entrenamiento)

## Contexto y estado actual (ya verificado en el repo)

La evaluación cross-structure **ya está implementada a nivel de backend, pero solo para un par único source→target y solo por CLI/JSON**. NO hay: concepto de *sweep* (varios datasets de entrenamiento), descubrimiento de pares, API HTTP, ni pestaña en la UI. Este GOAL añade exactamente eso, replicando el patrón de la pestaña "Mixing datasets".

Piezas que YA existen y debes reutilizar (no reimplementar):
- `Comparison/scripts/ml_vs_siesta/cross_structure_materialize.py`: `plan_cross_structure_dataset()`, `materialize_or_reuse_cross_structure_dataset()`, `run_cross_structure_payload(payload, *, launch_fn=None)`. Este último ya hace preview/materialize/train para **un** par. La materialización congela splits: `train`/`validation` del source, `test` del target, con chequeos fail-closed (especies, hashes de base/pseudopotencial, semántica de Hamiltoniano, fugas de datos). Documentado en `docs/cross_structure_evaluation.md`.
- `Comparison/scripts/run_cross_structure_payload.py`: CLI que envuelve lo anterior (acciones `preview`/`materialize`/`train`, `--status-json`, `--manifest-json`, `--poll-seconds`).
- Ejemplo committeado: `Comparison/config/graphene_w90_to_5x5_cross_structure_preview_payload.json` (schema `g2m_deeph_cross_structure_run_v1`).
- `Graph2MatDeepHBenchmarkRunner` (en `g2m_deeph_runner.py`), con `.start()` / `.status()` / `.results()`, ya usado como `launch_fn`.

El patrón de UI a **imitar 1:1** es "Mixing datasets", cuya implementación completa es tu plantilla:
- Frontend: `Comparison/ui/app.js`, bloque `setupMixingDatasets()` y funciones `mix*` (aprox. líneas 12906–13440): `mixDiscover`, `mixCollectBody`, `mixPreview`, `mixMaterialize`, `mixTrain`, `mixPollStatus`, `mixRenderChart`, `mixLoadMetrics`, `mixRenderPayloadSelector`, `mixSetAllPayloads`, selector de payloads por checkbox/grupo con `<details>`.
- HTML: `Comparison/ui/index.html`, `<section id="view-mixing-datasets" class="view">` (desde ~línea 1520) y el `<button class="tab" data-view="mixing-datasets">` (~línea 35). Registro de tab en `setupTabs()` (~línea 13442) y arranque en `DOMContentLoaded` (`setupMixingDatasets()` ~línea 13866).
- Backend: `Comparison/scripts/pipeline_ui.py`: endpoints `/api/mixing/discover|plan|launch|status|metrics|metrics-demo` (GET en ~17417–17443, POST en ~17543–17548), clase `MixingSweepRunner` (~17091), helpers `mixing_discover_payload`, `mixing_plan_payload`, `_mixing_payload_id`, `_mixing_metrics_payload`, y `aggregate_mae_vs_size` (en `ml_vs_siesta/plot_mixing_mae_vs_size.py`).

Helpers de UI compartidos ya disponibles: `request()`, `mvsBind()`, `mvsValue()`, `ensurePlotlyLoaded()`, `methodDisplayLabel()`, `showToast()`, `escapeHtml()`.

## Diferencia conceptual crítica (léela antes de tocar nada)

En "Mixing datasets", una permutación se identifica por `(size, mode, ratio)` y el eje X del plot es el **tamaño total del dataset**. En "Cross testing" NO hay ratio/mode/size mezclado: una permutación es un **par `(source_dataset, target_dataset)`**. El usuario quiere:

- **Eje X del plot = el dataset de ENTRENAMIENTO (source)**, no el de test. El target (donde se predice) queda fijo o agrupado por curva.
- Un **sweep** = un target fijo (o varios) × una lista de sources. Ejemplo canónico: entrenar en grafeno w90 (celda pequeña) y testear en la celda 5×5. Otro punto de la curva: entrenar en 5×5 y testear en 5×5, etc.

Por tanto la identidad de payload es `(source_id, target_id)`, y el eje X es una magnitud del **source**: nº de snapshots de entrenamiento (preferido) o nº de átomos del source. La agrupación por curva es por `(target, model)` — así cada curva responde "cómo cambia el MAE sobre el target Y según en qué source X entreno".

## Qué construir (fases)

### Fase 1 — Sweep runner + orquestación (backend, `ml_vs_siesta/`)
Crea la lógica de sweep que hoy no existe. Añade en el paquete `ml_vs_siesta` (nuevo módulo, p.ej. `cross_structure_sweep.py`) funciones análogas a `mixing_sweep`:
- `plan_cross_structure_sweep(sources, targets, ...)`: producto de pares `(source, target)`; para cada par corre `plan_cross_structure_dataset()` en modo dry (sin escribir) y devuelve `{ "permutations": [ {source_root, target_root, source_id, target_id, source_n_snapshots, source_n_atoms, target_n_atoms, status, output_root, ...} ], "warnings": [...] }`. Reutiliza los chequeos de compatibilidad existentes; un par incompatible NO aborta el sweep entero: márcalo con `status: "incompatible"` + razón y sigue.
- `run_cross_structure_sweep(sources, targets, output_root, *, models, epochs, performance, split_policy, launch_fn, progress_fn, dry_run, ...)`: por cada par compatible, materializa (reusa si ya existe) y, si `action=train`, lanza el runner real vía `run_cross_structure_payload(..., launch_fn=...)`. Emite records incrementales por `progress_fn` con la MISMA forma que consume el agregador (ver Fase 2). Persiste un `cross_structure_sweep_summary.json` (análogo a `mixing_sweep_summary.json`) con `records` + `permutations`.
- Respeta el bloqueo existente: `training_sweep` NO se soporta en cross-structure (la función ya lanza error); mantenlo.

Exporta lo nuevo en `ml_vs_siesta/__init__.py` junto a los símbolos cross-structure ya exportados.

### Fase 2 — Agregación MAE-vs-source (backend)
Los records de cross-testing NO encajan en `aggregate_mae_vs_size` (asume `mode/ratio/size/total_size`). Escribe un agregador propio `aggregate_cross_structure_mae(records)` que devuelva la MISMA estructura de salida que consume el frontend de mixing (`{ curves: [{ label, model, points: [...] }], payloads: [{id, label, ...}] }`) pero con:
- `payload_id = f"{source_id}__to__{target_id}"` (estable, independiente de rutas tmp — mismo principio que el comentario de `_mixing_payload_id`).
- clave de curva = `(target_id, model)`; `label` = p.ej. `f"→ {target_id} · {model}"`.
- cada punto lleva `x = source_n_snapshots` (train real; fallback `source_n_atoms`), `mae` (meV en el front), y en `text`/hover: source, target, nº snapshots train, model, seeds.
- agregación por seeds mean±std igual que el existente (reutiliza la lógica de `_point` si puedes factorizarla; si no, cópiala mínima).

### Fase 3 — API HTTP (backend, `pipeline_ui.py`)
Añade una clase `CrossStructureSweepRunner` calcada de `MixingSweepRunner` (thread daemon, `_lock`, `status()`, `metrics()`, `start(body)`, acumulación live de records vía `progress`), y su singleton. Añade endpoints espejo de los de mixing:
- GET `/api/cross-testing/discover` → lista de datasets candidatos (source y target) con `{root, n_snapshots, n_atoms}`. Reutiliza el barrido de `mixing_discover_payload` (recorre `DATASETS_ROOT` por `frozen_split_manifest.json`, usa `read_dataset_samples` + `dataset_atom_count`), pero SIN separar por umbral small/large: aquí cualquier dataset validado puede ser source o target. Devuelve una única lista `datasets`.
- POST `/api/cross-testing/plan` → `plan_cross_structure_sweep(...)`.
- POST `/api/cross-testing/launch` → `CROSS_TESTING_RUNNER.start(body)` con `202 ACCEPTED` (acciones `preview`/`materialize`/`train`).
- GET `/api/cross-testing/status` → `.status()`.
- GET `/api/cross-testing/metrics` → `.metrics()` (usa `aggregate_cross_structure_mae` + merge con permutaciones del plan/summary, igual que `_mixing_metrics_payload`).
- GET `/api/cross-testing/metrics-demo` → curvas sintéticas para que el plot renderice antes de entrenar (espejo de `mixing_metrics_demo_payload`).

El `body` del launch debe aceptarse **tanto desde la UI como desde un JSON** (mismo esquema), para que `Comparison/scripts/run_cross_structure_payload.py` pueda extenderse a un modo sweep o se cree un `run_cross_structure_sweep_payload.py` hermano que POSTee o llame directo al runner. Requisito del usuario: lanzable por botón en la UI **y** por terminal.

### Fase 4 — Frontend: pestaña "Cross testing"
- `index.html`: nuevo `<button class="tab" data-view="cross-testing" data-icon="CT">Cross testing</button>` junto al de mixing, y una `<section id="view-cross-testing" class="view">` que replica la estructura de paneles de mixing: (1) descubrir datasets, (2) configurar sweep, (3) plan/preview, (4) selector de payloads + gráfica. Config mínima editable en la UI: multiselección de sources y de un/varios targets (o textareas `source=root` / `target=root` como hace mixing con `mix-small-map`), modelos (graph2mat/deeph), epochs, performance, `confirm_incomplete_hamiltonian_semantics` (checkbox, porque datasets viejos lo requieren), y un `<pre>` de log de payloads con botones Bottom/Clear.
- `app.js`: bloque `setupCrossTesting()` y funciones `ct*` calcadas de las `mix*`: `ctDiscover`, `ctCollectBody`, `ctPreview`, `ctMaterialize`, `ctTrain` (con `window.confirm` igual que `mixTrain`), `ctPollStatus` (intervalo 1500 ms, para al completar/error), `ctLoadMetrics`, `ctRenderPayloadSelector` (checkboxes agrupados por target con `<details>`), `ctSetAllPayloads`, y `ctRenderChart`.
- **Gráfica**: `Plotly.newPlot` con `title: "MAE vs dataset de entrenamiento (cross testing)"`, `xaxis.title: "Snapshots de entrenamiento (source)"`, `yaxis.title: "Hamiltonian MAE (meV)"`. Solo dibuja puntos cuyo `payload_id` esté en la selección (igual que mixRenderChart filtra por `mixSelectedPayloadIds`). Los resultados **se van ploteando conforme el sweep avanza**: en `ctPollStatus`, si `action==="train"` y hay records live, llama a `ctLoadMetrics(false)` — idéntico al comportamiento de mixing.
- Registro: añade el `case` en `setupTabs()` (carga discover+metrics al activar la pestaña la primera vez, patrón `mixDiscoverLoaded`) y llama `setupCrossTesting()` en el arranque `DOMContentLoaded`.

### Fase 5 — Tests + verificación
- Tests backend (pytest, sin GPU, sin entrenamiento real): `plan_cross_structure_sweep` produce el producto correcto de pares y marca incompatibles sin abortar; `aggregate_cross_structure_mae` agrupa por `(target, model)` y pone `x = source_n_snapshots`; el `CrossStructureSweepRunner` acumula records live y `metrics()` los fusiona con las permutaciones. Reusa fixtures de `tests/test_cross_structure_evaluation.py` (`_make_dataset`) y el estilo de `tests/test_ml_vs_siesta_mixing.py`.
- Verificación manual descrita: `discover` lista los datasets locales; `preview` de un sweep {source: w90 pequeño, 5x5} × {target: 5x5} devuelve permutaciones con status; `metrics-demo` renderiza la curva; un `train` corto (si hay entorno) va ploteando por par.

## Restricciones y criterios de aceptación (modo lazy)

- **Reutiliza, no dupliques.** Toda la anti-fuga, materialización y compatibilidad ya existen en `cross_structure_materialize.py`: el sweep es un bucle sobre pares que llama a `run_cross_structure_payload`/`materialize_or_reuse_cross_structure_dataset`. No reimplementes chequeos de splits ni compatibilidad.
- **Copia el patrón de mixing** para UI/API/runner en vez de inventar uno nuevo; los helpers `request/mvsBind/mvsValue/ensurePlotlyLoaded/methodDisplayLabel/showToast/escapeHtml` ya están.
- **Sin dependencias nuevas.** Plotly, http.server y threading ya se usan.
- Un par incompatible produce un warning visible en el plan y se **omite** del entrenamiento; nunca rompe el sweep entero.
- Mantén el bloqueo de `training_sweep` en cross-structure.
- Cada pieza de lógica no trivial (planner, agregador, runner) deja **un** check runnable (assert en `__main__` o un `test_*.py` mínimo), sin frameworks extra.

## Definición de "hecho"
Una pestaña "Cross testing" donde el usuario: descubre datasets, arma un payload de sweep source×target (por UI o por JSON), lo lanza con un botón (o por terminal), y ve el **MAE vs dataset de entrenamiento** ploteándose en vivo por par entrenado, con selector de runs (checkboxes) idéntico al de "Mixing datasets".
