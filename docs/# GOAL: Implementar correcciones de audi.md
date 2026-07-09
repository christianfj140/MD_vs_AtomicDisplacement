# GOAL: Implementar correcciones de auditoría científica y de software — MD_vs_AtomicDisplacement

## Contexto y decisiones ya tomadas (NO re-litigar)
Dos auditorías independientes convergieron en los mismos hallazgos. Implementa las correcciones. Decisiones del usuario ya fijadas:
- **Datasets ya archivados**: C3 (provenance) SÍ se re-graba in-place (recuperable parseando siesta_build_info). C2 (split temporal) NO se regenera — solo se documenta como limitación. NO re-correr SIESTA.
- **Alcance**: solo CRÍTICOS (C1-C3) e IMPORTANTES (I1-I7). Ignora los MENORES (M1-M9).
- **Split default**: cambiar a `fixed_common_test` en todos los puntos de entrada.

## Reglas de ejecución (token-eficiencia sin perder rigor)
1. NO releas archivos ya citados con archivo:línea salvo para editar. Los números de línea de abajo son de la auditoría; ábrelos con `offset`/`limit` (±40 líneas), no el archivo entero, salvo pipeline_ui.py que requiere grep dirigido.
2. Trabaja fase a fase. Cada fase: (a) lee solo lo necesario, (b) edita, (c) añade/actualiza SU test, (d) marca la fase. NO avances con una fase rota.
3. Reutiliza helpers de shared/ (file_sha256, etc.) — ya existen; no dupliques.
4. Un test runnable por cambio no trivial. Usa el venv del repo: `.venv/bin/python -m pytest`. El python del sistema no tiene pytest.
5. Al final de CADA fase corre solo los tests de esa fase, no toda la suite. Suite completa solo en la fase final.
6. Si un hallazgo ya está corregido al abrirlo (auditorías con líneas ligeramente distintas: una dice app.js:9215, otra :12785), verifica el estado real y ajusta; no asumas.

---

## FASE 1 — C1: split_policy configurable y default fixed_common_test
Objetivo: que el sweep de mixing NUNCA corra con resplit_combined por omisión, y que split_policy sea seleccionable extremo a extremo.

Puntos a tocar (verifica cada uno antes de editar):
- `Comparison/scripts/run_mixing_e2e_payload_once.py:73-116` (`_run_mixing_payload`): leer `split_policy` del payload y reenviarlo al runner. Default `fixed_common_test` si ausente.
- `Comparison/scripts/pipeline_ui.py:~16629` (`_run_mixing_sweep_parallel`) y `~16881` (`/api/mixing/launch`): cambiar default `resplit_combined` → `fixed_common_test`; propagar split_policy del payload.
- `Comparison/ui/app.js`: exponer selector split_policy en el tab Mixing (buscar el bloque de lanzamiento de mixing; NO confundir con `reusable_split_policy` de app.js:~9215 que es otra cosa). Incluir split_policy en el payload que la UI envía.
- `Comparison/config/ml_vs_siesta_mixing_sweep_100_500_train_payload.json` y `graphene_5x5_snapshot_scaling_100_500_mixing_payload.json`: añadir `"split_policy": "fixed_common_test"` explícito.
- `docs/ml_vs_siesta_benchmark.md:183-207`: actualizar el flujo documentado para reflejar que split_policy ahora ES alcanzable desde UI/wrapper y que el default es fixed_common_test. Resolver la divergencia doc↔código (I7).

Test: extender `tests/test_run_mixing_e2e_payload_once.py` — payload sin split_policy → fixed_common_test; payload con resplit_combined explícito → se respeta. Verifica que el valor llega al runner (mock/captura del arg).

## FASE 2 — I4: extracción de h_mae_eV robusta (misma zona pipeline_ui)
Aprovecha que pipeline_ui.py ya está abierto de la Fase 1.
- `pipeline_ui.py:16469-16498` (`_extract_model_h_mae_eV`): dejar de usar "last-match-wins". Filtrar explícitamente por split final/test; si hay ambigüedad, fallar ruidoso (no elegir en silencio).
- `pipeline_ui.py:16544-16562` (`_mixing_metrics_from_run_metrics`): al hacer rglob de kpoint_matrix_metrics.csv, filtrar por split (test/final), no promediar train+val+test mezclados.

Test: nuevo test con un árbol de resultados sintético que contenga métricas de varios splits → confirma que se selecciona test y que múltiples matches ambiguos lanzan error en vez de coger el último.

## FASE 3 — C3: probe de versión SIESTA endurecido + re-grabado in-place
Parte A (código, previene futuros):
- `MD/scripts/generate_md_dataset.py:129-161` (`probe_siesta_version`): no aceptar la primera línea no vacía. Validar que la línea parece una versión real (p.ej. contiene "Version" o patrón `\d+\.\d+`); la versión real vive en `siesta_build_info` ("Version : 5.4.2..."). Extraer de ahí. Si no se puede validar → status distinto de "detected" (p.ej. "unverified"), NO texto basura.
- `shared/benchmark_manifest.py:329-353`: el gate no debe pasar con "texto no vacío". Exigir que siesta_version cumpla el mismo patrón de versión válida; fail-closed si no.

Parte B (re-grabado de datasets vivos):
- Escribir `MD/scripts/regrab_siesta_provenance.py`: recorre los material_provenance.json bajo Comparison/datasets/, y para cada uno con siesta_version inválido pero con siesta_build_info conteniendo "Version : X.Y.Z", corrige siesta_version in-place re-parseando build_info. Idempotente, dry-run por defecto con flag `--apply`. Loggea qué cambió.
- Ejecutarlo con --apply sobre los datasets archivados y verificar que graphene_w90_scale_iid100 y graphene_5x5_scale_iid100 quedan con versión válida.

Test: `tests/` — probe con stdout basura (ruido X11) → no "detected"; probe con build_info válido → extrae "5.4.2"; gate de benchmark_manifest rechaza versión inválida. Y un test del regrab script sobre un provenance.json de fixture.

## FASE 4 — C2: documentar fuga temporal (sin regenerar)
Solo documentación, decisión del usuario. Ser preciso y honesto.
- `docs/known_limitations.md`: añadir sección declarando que en los datasets snapshot-scaling (*_iid*): (1) train no contiene 450K mientras val/test SÍ → el MAE reportado es error de EXTRAPOLACIÓN a 450K, no in-distribution; (2) val y test son la misma trayectoria 450K con gap temporal de ~1 fs (físicamente nulo frente a periodos vibracionales del C ~20-40 fs) → selección de checkpoint contaminada por gemela temporal del test; (3) el nombre *_iid* es un misnomer (frames MD consecutivos, no iid). Declarar que cualquier número ya publicado con estos splits necesita esta nota.
- `MD/pipeline_config.yaml:186-191`: subir el default de `temporal_gap` a un valor físicamente significativo (≥ ~20-40 frames a 1fs, justifícalo con un comentario citando el periodo vibracional). Esto arregla FUTUROS datasets aunque no los viejos.
- Añadir comentario en el generador y/o en los payloads que dicen "IID mixed temperature" aclarando la semántica real.

Test: no aplica (docs+config). Verifica que el yaml sigue siendo válido cargándolo.

## FASE 5 — C2-agravante: mixing preserva blocked_with_gap
- `mixed_dataset_materialize.py:399-417`: al re-splitear el pool combinado, ambas políticas (`_split_pool` baraja todo; `fixed_common_test_ids` muestrea test al azar del small) DESTRUYEN el blocked_with_gap de origen → frames de test a 1 fs de frames de train. Hacer que el re-split respete el orden temporal/bloques del dataset de origen, o como mínimo invocar el `check_geometry_leakage.py` existente (que hoy NO se llama en la ruta de mixing) y fallar/avisar si detecta vecindad temporal train↔test.
- `mixed_dataset_materialize.py:283-289` (`_fixed_common_test_split`, I6-menor pero barato aquí): guardar el caso validation vacío (n_rest ≤ 1) con error claro, igual que ya se guarda test vacío.

Test: extender `tests/test_ml_vs_siesta_mixing.py` — pool con estructura temporal conocida → tras split, ningún test_id es vecino temporal (dentro del gap) de un train_id. Validation vacía → error claro.

## FASE 6 — I3 + I2: provenance del merge honesta + evidencia ghosts
- `mixed_dataset_materialize.py:315-329` (`_copy_dataset_lineage`): NO presentar la provenance del small como la del dataset mezclado. Si el pool mezcla small (con Ghost-H) y 5x5 (sin ghosts), el material_provenance.json resultante debe reflejar la mezcla (o al menos marcar que es heterogénea y apuntar a mixed_dataset_provenance.json como fuente de verdad). No heredar ciegamente material/basis/pseudos del small.
- `mixed_dataset_materialize.py:169-207` (exención Ghost-H): la afirmación "los ghosts no son parte de la representación G2M/DeepH" hoy no tiene respaldo en código. Añadir verificación: comprobar que el basis table del run mezclado trata los ghosts consistentemente (o que el pipeline de entrenamiento efectivamente los excluye). Si no es verificable en código, degradar la exención a un check explícito que falle si un dataset small con Ghost-H.ion.xml entra al pool sin la exención confirmada. NO dejar la afirmación como comentario sin verificación.
  (Nota del usuario: ghosts es problema de provenance/evidencia, NO crítico; no bloquees el pipeline sin evidencia numérica de impacto — basta con hacer la exención explícita y verificable.)

Test: `tests/test_ml_vs_siesta_mixing.py` — merge heterogéneo → provenance resultante no afirma falsamente "single material with Ghost-H"; dataset small con ghosts sin exención confirmada → check falla.

## FASE 7 — I1: colisión de sample_id entre bloques de temperatura
- `MD/scripts/generate_md_dataset.py:772`: los metadata.json conservan sample_id local (md_0..md_19 repetido entre bloques) además de global_sample_id. El frozen split usa global (md_0..md_99). Cualquier join frozen↔metadata por sample_id casa mal.
- Fix: en metadata.json escribir el global_sample_id como identificador primario (o renombrar el local a algo inequívoco tipo `block_local_sample_id` y hacer que el sample_id "oficial" sea el global). Asegurar que auditorías/stencils que cruzan por sample_id usen el global.

Test: generar (o fixture) dos bloques de temperatura → ningún sample_id primario colisiona; join frozen↔metadata es 1:1.

## FASE 8 — I5: réplicas por seed + barras de error en curvas de mixing
- Payload de producción (`ml_vs_siesta_mixing_sweep_100_500_train_payload.json`) fija seed:0, una sola tanda. El repo ya exige ≥3 seeds en otro sitio (`--min-final-seeds 3` en final_stats.py). 
- `plot_mixing_mae_vs_size.py:17-93`: hoy promedia duplicados pero no calcula std/CI ni exige N mínimo. Añadir: agregación por seed, cálculo de media±std (o CI), y N por punto en la salida. Que el plot muestre barras de error.
- Payload: parametrizar para ≥3 seeds (o documentar cómo lanzarlo con múltiples seeds). Añadir gate opcional que avise si una curva se genera con <3 seeds ("exploratorio, no publicable").

Test: `plot_mixing_mae_vs_size` con datos de 3 seeds → produce media, std y N por punto; con 1 seed → marca el resultado como exploratorio.

## FASE 9 — I6: gate autograd DeepH más estricto y honesto sobre dtype
- `tests/test_deeph_autograd_derivatives.py:315-322`: el assert pasa si el MEJOR δ baja de 25% (cherry-pick del δ). Endurecer: (a) reportar el δ óptimo, no solo el mejor caso favorable; (b) el criterio de convergencia (errors[0]*1.2) tolera que reducir δ no mejore — exigir que exista una ventana de δ donde el error decrezca antes de saturar (comportamiento de diferencias finitas correcto: truncamiento arriba, cancelación abajo). (c) Documentar explícitamente en el test que valida float32 (producción), y que la validación float64 previa NO se re-ejecuta aquí — o re-ejecutar un caso float64 como referencia.
- `tests/test_deeph_autograd_derivatives.py:240,252,318`: el default hardcodeado `/home/christian/repositorios/DeepH-pack/.venv/bin/deeph-inference` debe venir de env var/config con skip claro si no está disponible (no petar en otra máquina).

Test: es un test; ajusta sus asserts. Verifica que sigue pasando en float64 y que el criterio de δ-window funciona con datos sintéticos de derivada conocida.

## FASE FINAL — Verificación global
1. `.venv/bin/python -m pytest -q` sobre la suite completa. Objetivo: mantener o superar el baseline (auditoría 1 vio 116 passed; auditoría 2 vio 74 en un subconjunto). Cero regresiones.
2. Resumen final: tabla hallazgo → archivo:línea cambiado → test que lo cubre → estado. Lista explícita de lo NO tocado (C2 regeneración, M1-M9) y por qué.
3. NO commitear salvo que el usuario lo pida.

## Orden y por qué (no lo cambies sin razón)
Fases 1-2 comparten pipeline_ui.py (una sola pasada). 3-4-5 son el eje datasets/provenance/split. 6-7 tocan generate_md_dataset.py y mixed_dataset_materialize.py (reutiliza lo abierto). 8-9 son tests/plots aislados. Esto minimiza reaperturas de los dos archivos gigantes (pipeline_ui.py, generate_md_dataset.py).
