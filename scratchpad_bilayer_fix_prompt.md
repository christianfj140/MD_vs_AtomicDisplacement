

## CONTEXTO

Una auditoría previa dictaminó **ROTO** el target moiré `graphene_hBN_moire_22deg` generado por
`Comparison/scripts/build_graphene_hbn_moire_target.py`. El software del pipeline (merge,
k-mesh, UI, bind del runner, tests) es correcto; lo que invalida las métricas es **la geometría
del target**. Hay que arreglar los hallazgos Crítico y Alto. Ignora el tema de seeds: 1 seed
está bien para explorar.

Decisiones ya fijadas por el usuario (NO re-preguntar):
- **Ángulo de twist = ~21.79°** (el conmensurado exacto de índice (m,n)=(1,2), celda de 16
  átomos). Es viable con SIESTA local. NO uses 1.08° (su periodo moiré son ~131 Å ≈ 11k átomos,
  inviable). El objetivo aquí es un **target de test pequeño y físicamente válido** para
  explorar transfer, no un moiré paper-ready de ángulo pequeño.
- 1 seed.

---

## GOAL

Que `build_graphene_hbn_moire_target.py` produzca, a 21.79° (índice (1,2), 16 átomos), un
target de test **físicamente válido**: una bicapa grafeno/hBN twisteada donde **ninguna capa
tenga átomos solapados**, con el strain **realmente calculado** (no una etiqueta hardcoded), un
**guard** que aborte antes de correr SIESTA si la geometría es inválida, y un provenance
honesto. Regenerar el dataset y re-verificar el cross-testing smoke end to end.

Criterio de éxito duro: la distancia interatómica mínima (con imágenes periódicas) de la
geometría generada es **> 1.2 Å** para las tres especies, y el strain reportado en el
provenance coincide con el que realmente se aplicó a la geometría.

---

## PROBLEMAS A ARREGLAR (con su evidencia de la auditoría)

### C1 — Geometría rota: la capa hBN tiene átomos solapados (CRÍTICO)
- **Evidencia**: en el `RUN.fdf` generado, la distancia mínima interatómica con imágenes
  periódicas es **0.937 Å entre dos átomos de Boro** (muy por debajo del enlace B–N ideal ~1.45
  Å; por debajo de ~1 Å = átomos solapados). La red de hBN queda destrozada.
- **Causa raíz** (`moire_geometry`, ~líneas 160–174 de `build_graphene_hbn_moire_target.py`):
  el "twist" **rota SOLO la subcapa hBN** sobre el centro de la supercelda y hace `wrap` dentro
  de la MISMA celda. Rotar una sublattice + wrap NO produce una red conmensurada válida salvo
  para ángulos del grupo de simetría de la supercelda; rompe la periodicidad del hBN.
- **Qué debe quedar bien**: la geometría twisteada debe ser **coherente y periódica** para las
  dos capas. Opciones (elige la más simple que dé dist_min > 1.2 Å y sea realmente conmensurada
  a 21.79° con la celda de 16 átomos):
  (a) construir la supercelda conmensurada twisteada con la **construcción estándar de bicapa
      twisteada** (vectores de superred definidos por (m,n); la capa 1 usa la base (m,n) y la
      capa 2 la base (n,m)), de modo que AMBAS capas son periódicas en la misma superred; o
  (b) si se mantiene el enfoque "una capa fija + otra rotada", rotar de forma que el ángulo sea
      exactamente una operación del grupo de la supercelda (para (1,2) en red hexagonal el
      ángulo 21.79° debe mapear la sublattice en sí misma sin wrap que rompa distancias).
  Lo que NO vale: rotar la sublattice y hacer módulo/ wrap ciego (el bug actual).
- **Verificación obligatoria**: calcular `dist_min` con imágenes periódicas (3×3 en el plano)
  ANTES de escribir el fdf y ANTES de SIESTA. Si `dist_min < 1.2 Å`, **abortar con error claro**
  (guard, ver abajo).

### A1 — El strain 1.8% está hardcoded (ALTO)
- **Evidencia**: `build_graphene_hbn_moire_target.py:198` → `lattice_mismatch_percent = 1.8`, y
  `effective_hBN_strain_percent = lattice_mismatch_percent`. Es una constante literal que va al
  provenance como si fuera una medida.
- **Qué debe quedar bien**: el strain reportado debe **calcularse de la geometría real**: la
  deformación efectiva impuesta a la capa hBN al forzarla en la superred (p.ej. razón entre el
  parámetro de red nativo del hBN y el que realmente ocupa en la celda generada, en %). Si la
  construcción elegida NO impone strain (porque ambas capas comparten la misma red de grafeno
  por diseño), entonces el strain que corresponde reportar es el **desajuste nativo
  grafeno/hBN sin acomodar** y hay que dejar EXPLÍCITO que la geometría usa la red de grafeno
  para ambas capas (y por tanto el hBN está comprimido ese %). En cualquier caso: el número del
  provenance = el que de verdad describe la geometría, no un literal suelto. Si no se puede
  cuantificar limpiamente, retira el campo y di "no cuantificado" en vez de inventarlo.

### A2 — El ángulo/celda no se validan contra la geometría (ALTO, ligado a C1)
- **Qué debe quedar bien**: reproducir en el provenance que `twist_angle_deg` sale de
  `commensurate_angle_degrees(1,2)` Y que la geometría escrita realmente lo materializa
  (no solo la etiqueta). Registrar `num_atoms` real de la geometría (contado, no `4*p^2`
  asumido) y que coincide con lo esperado.

---

## GUARD (nuevo, obligatorio)
Añade una comprobación de geometría reutilizable que se ejecute SIEMPRE antes de correr SIESTA
(y en el `--dry-run`): calcula la distancia interatómica mínima con imágenes periódicas y
**aborta con RuntimeError** si es menor que un umbral (`--min-atom-distance`, default 1.2 Å).
Deja un check runnable (assert-based) que verifique el guard: una geometría con solape debe
abortar; una válida debe pasar. Es exactamente el tipo de red de seguridad física que el bug
actual no tenía.

---

## MÉTODO / REUTILIZACIÓN
- Reutiliza `extract_fdf_structure` + `materialize_fdf_text` (shared/fdf_materialization) para
  leer la celda de apilamiento y escribir la geometría resultante, como ya hace el builder.
- Reutiliza el resto del builder sin tocar: `run_siesta`, `validate_snapshot`,
  `validate_dataset`, escritura de manifests, `_rescale_inplane_kgrid` (el k-mesh 20→10 es
  correcto). Solo cambia la construcción de la geometría (`moire_geometry`), el cálculo del
  strain y el guard.
- El resto del pipeline (merge, cross-sweep, payloads, UI) está bien; NO lo toques.

## VERIFICACIÓN (end to end, no solo asserts)
1. `python build_graphene_hbn_moire_target.py ... --dry-run` reporta `dist_min > 1.2 Å` y el
   strain calculado; sin solapes.
2. Regenerar el dataset real (`--overwrite`) con SIESTA (celda de 16 átomos, viable local).
   Debe pasar `validate_dataset` y el planner del cross-sweep debe marcar el par
   `graphene_hBN_bilayer → graphene_hBN_moire_*` COMPATIBLE en preview.
3. Re-lanzar el smoke de entrenamiento+cross-eval (10 épocas, 1 seed) con el payload existente
   `graphene_hbn_bilayer_to_moire_train_smoke_payload.json` y confirmar que produce `records[]`
   con `h_mae_eV` finitos para g2m y deeph. **Compara el nuevo MAE con el viejo (0.856/1.056):**
   sobre una geometría física válida el número debería tener sentido (aunque siga alto por ser
   10 épocas). Reporta el valor real, no lo maquilles.
4. Actualiza el test `tests/test_graphene_hbn_moire_target.py` para que verifique el guard de
   distancia mínima y que el strain del provenance es coherente con la geometría (no un literal).
5. Actualiza la nota física en `docs/cross_structure_evaluation.md` para reflejar la geometría
   corregida (qué construcción se usa, qué strain real hay, que se valida dist_min).

## RESTRICCIONES
- NO toques ni interfieras con procesos en curso (hay sweeps de derivadas y de vacancy seeds
  corriendo). NO reinicies la UI ni mates procesos. Usa el output-root del bilayer, aislado.
- Trabaja en rama; NO commitees hasta que la verificación end-to-end pase. Si algo de SIESTA
  falla o es lento, documenta el estado real y no fabriques métricas.
- Honestidad de procedencia: el provenance debe describir la geometría REAL. Nada hardcoded que
  se venda como medido.
- 1 seed está bien; no añadas la maquinaria de multi-seed.

## ENTREGABLE
`build_graphene_hbn_moire_target.py` con geometría coherente + strain calculado + guard;
dataset `graphene_hBN_moire_22deg` regenerado y válido; smoke re-verificado con el MAE real
reportado honestamente; test y doc actualizados. Un resumen final de: qué estaba roto, qué se
arregló, el `dist_min` nuevo, el strain real, y el MAE antes vs después.
