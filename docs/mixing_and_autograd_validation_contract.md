# Contrato de validación: mezcla small/large y derivadas autograd

Contrato técnico implementado en la corrección de auditoría de 2026-07-10.
Define qué significa cada opción, qué garantiza cada gate y qué claims puede
sostener cada resultado.

## 1. Semántica de `add`

`ratio` = fracción del **pool de entrenamiento large** que se añade
(`fraction_of_large_pool_added`). Los snapshots de test reservados (small y
large bajo `fixed_stratified_test`) quedan fuera de la aritmética del ratio:
`n_large = round(ratio * len(large_pool_sin_test))`.

## 2. Semántica de `replace`

`ratio` = fracción del pool small que se sustituye por large manteniendo
constante el total entrenable. Cap por large disponible y por los ids small
reservados como test (`replace_cap_reasons`: `available_large`,
`reserved_small_test`; flag `large_capped`).

## 3. Ratio nominal vs composición real

Cada materialización persiste en `mixed_dataset_provenance.json` (schema
`ml_vs_siesta_mixed_dataset_provenance_v2`) el bloque `composition`:
`actual_train_size`, `n_small_train`, `n_large_train`,
`actual_large_fraction_by_snapshots`, `_by_atoms`, `_by_blocks` (node blocks =
átomos; edge blocks no se reportan porque requieren listas de vecinos),
`_by_matrix_elements` (desde la cabecera de `ORB_INDX`), tamaños de
validación y test por dominio, `requested_count_float` y `rounding_policy`
(`python_round_half_even`).

Medido en el smoke real 20/20: ratio nominal `add 0.4` ⇒ 46% large por
snapshots pero **95.5% por átomos y 99.5% por elementos de matriz**. El ratio
nominal jamás debe presentarse como composición efectiva.

## 4. Train size vs total materializado

El eje "dataset size" significa `actual_train_size` (train real). El total
materializado (`materialized_total_size`) incluye test fijo y validación y no
es un eje científico. Los records de las curvas llevan ambos.

## 5. Políticas de test (`split_policy`)

- `fixed_stratified_test` (recomendada): test fijo small+large — el split de
  test del dataset fuente cuando existe, o la cola temporal de cada pool.
  Idéntico entre ratios, modos y seeds. `evaluation_scope = small_and_large`.
- `fixed_common_test` (= `fixed_common_test_small_only`): test fijo solo
  small. Válida únicamente para la pregunta "¿cómo afecta añadir datos large
  al rendimiento en el dominio small?". `evaluation_scope = small_only`.
- `resplit_combined` (legacy): el test cambia con cada selección; solo
  exploratorio. `evaluation_scope = combined_resplit_legacy`.

Guard de no-leakage: un snapshot con `source_split == "test"` no puede caer en
train/validation; solo se anula con `allow_source_test_in_train=True`
(explícitamente no científico). Cada snapshot combinado conserva `origin`,
`source_root`, `source_sample_id`, `source_split` y `n_atoms`.

## 6. Riesgo de ponderación

Con la loss legacy elementwise, una estructura de 50 átomos aporta ~4× más
elementos de nodo y ~25× más bloques que una de 2 átomos: el mismo error
unitario cuesta 4× menos en la estructura pequeña
(demostrado en `graph2mat .../tests/test_per_structure_losses.py::test_legacy_elementwise_bias_toward_large_structures`
y `DeepH-pack/tests/test_training_weighting_policy.py::test_legacy_elementwise_bias`).

## 7. Políticas de loss (`training_weighting_policy`)

- `legacy_elementwise`: comportamiento histórico, reproducible.
- `per_structure`: media dentro de cada estructura, después media de
  estructuras. Graph2Mat: `block_type_mse_per_structure` (mantiene términos
  node/edge separados, `L_s = w_n L_node + w_e L_edge`). DeepH:
  `masked_mse_per_graph` + media por grafo (respeta la máscara orbital).
- `per_domain`: media small y media large ponderadas
  (`small_domain_weight`/`large_domain_weight`); requiere
  `domain_threshold_atoms` explícito.

La política viaja en el payload de mezcla (`training_weighting_policy` +
`domain_weighting`) y se registra en el summary y en cada record; curvas con
políticas distintas nunca se agregan (la política forma parte de la identidad
de la curva).

## 8. Compatibilidad orbital

`dataset_compatibility_report.json` compara orbitales por átomo por especie
(desde `ORB_INDX`), hashes de base para especies reales, pseudopotenciales por
especie compartida (mismatch ⇒ bloqueo), escalares DFT (XC.functional,
XC.authors, MeshCutoff, ElectronicTemperature, DM.Tolerance, spin) y densidad
de k-points por eje (`|b_i|/N_i`, tolerancia 10% — la primitiva 20×20×1 y la
supercelda 5×5 con 4×4×1 son equivalentes).

## 9. Compatibilidad ghost

`ghost_compatibility_status` ∈ `not_applicable`, `proven_inactive`,
`proven_compatible`, `unproven`, `incompatible`, derivado de artefactos
activos (átomos en `AtomicCoordinatesAndAtomicSpecies` + orbitales en
`ORB_INDX`), no de un booleano manual:

- Ghost declarada pero sin átomos ni orbitales ⇒ `proven_inactive` (el caso
  real Ghost-H del grafeno primitivo): mezcla sin override.
- Ghost activa en un solo dataset ⇒ `incompatible`: bloquea siempre.
- Sin evidencia ⇒ `unproven`: requiere `confirm_ghost_species_exemption=True`.

## 10. Graph2Mat autograd

Jacobiano completo `[n_outputs, n_atoms, 3]` por VJP vectorizadas
(`vmap_vjp_chunked`); derivada cartesiana por chain rule con
`basis_table.change_of_basis` (frame e3nn). Validado contra FD del propio
modelo con checkpoint real
(`tests/test_graph2mat_autograd_derivatives.py`). CUDA no soportado por el
backward vectorizado con MACE TorchScript: fallo explícito, sin fallback.

## 11. DeepH forward-mode JVP

`predict_with_grad` usa `torch.autograd.forward_ad` (un forward dual por
dirección). Contrato de capacidades
`deeph.inference.capability.autograd_capability()`
(`deeph_autograd_capability_v1`): verifica firma con selección de
átomos/ejes, ejecuta un smoke JVP analítico real y declara
`output_schema=hamiltonians_grad_pred_v2`. El runner MD ejecuta este preflight
antes de lanzar inferencia (`capability_unavailable` si falla). Un backend con
el placeholder histórico NaN no supera el preflight.

`hamiltonians_grad_pred_v2`: las direcciones no calculadas quedan en NaN
(sentinel); `info.json` lista `grad_computed_atom_indices`/`_axis_indices`.
El runner rechaza NaN o direcciones ausentes en lo solicitado
(`deeph_autograd_non_finite` / `deeph_autograd_direction_not_computed`).
Spinful: `NotImplementedError` antes de calcular (`supports_spinful=false`).

## 12. Autograd vs finite differences

Autograd deriva el modelo exactamente (`predicted_delta_ang = null`); FD lo
aproxima con paso δ y hereda su ruido/discontinuidades. La dependencia con δ
pertenece al método FD, nunca al autograd.

## 13. Tres comparaciones de derivadas

- **A** `model_autograd_vs_model_fd`: consistencia matemática (sin SIESTA).
- **B** `model_fd_vs_siesta_fd`: error del modelo con el mismo método.
- **C** `model_autograd_vs_siesta_fd`: resultado final.

`evaluate_hamiltonian_derivative_metrics` etiqueta su manifest con
`comparison_kind` (B o C); A se reporta en
`DeepH-pack/work/autograd_sanity_*/autograd_fd_report.json` y en los tests de
Graph2Mat. Nunca se combinan en una sola métrica.

## 14. Dtype

La validación A corre en float64 y en el dtype de producción (float32).
Medido (2026-07-10): float64 error relativo Frobenius 1.1e-9 (δ=1e-5, cosine
1.0); float32 3.26e-2 en δ=1e-4 (límite de cancelación de FD en float32, no
un defecto del autograd). El dtype se registra en reports y firmas.

## 15. Topología fija

El JVP/jacobiano asume lista de vecinos fija. El report A comprueba las
claves de `rc.h5` y saltos de matrices de rotación entre ±δ
(`comparison_status`: `ok` / `frame_discontinuous` / `topology_changed`); los
puntos no-`ok` se excluyen del gate. En grafeno, δ≥1e-3 Å reordena vecinos
casi degenerados: FD inválido ahí.

## 16. Claims y gates

Escalera (`derivative_claim_status.py`):
`invalid < diagnostic_only < validated_model_derivative <
validated_against_siesta < paper_ready`.

- `validated_model_derivative`: gate A en `pass`.
- `validated_against_siesta`: además equivalencia base probada
  (`deeph_raw_global_equivalence_evidence_v1`, transformación ORB_INDX
  compartida entre H base y dH/dR con `basis_transform_sha256`) y SIESTA
  convergido.
- `paper_ready`: además repos `pinned_clean`, test fijo, ≥3 seeds y sin
  advertencias bloqueantes. Los claims se degradan automáticamente.

## 17. Versionado de repositorios

Todo manifest científico incluye `run_inventory`
(`shared/run_inventory.py`): SHA/rama/dirty de los tres repos, Python
efectivo, ruta real del módulo importado (detecta checkouts divergentes) y
`reproducibility_status` ∈ `pinned_clean | pinned_dirty | unpinned |
unavailable`. Solo `pinned_clean` puede aspirar a `paper_ready`.

Los caches de derivadas llevan `input_signature_sha256`
(`shared/artifact_signature.py`): checkpoint, commits/dirty de repos,
estructura (`RUN.fdf`), base, método, dtype, dirección. Cambiar cualquiera
invalida el cache; artefactos sin firma son `legacy_unverified` y se
recalculan.

## 18. Limitaciones pendientes

- Los edge blocks no entran en la composición efectiva (requieren listas de
  vecinos); los elementos de matriz de `ORB_INDX` son el proxy más fiel.
- `per_domain` en DeepH no soporta el optimizador lbfgs (fallo explícito).
- Las métricas micro/macro/domain están implementadas para derivadas
  (`_micro_macro_domain`); las métricas H absolutas conservan su agregación
  histórica (las curvas de mixing separan dominios vía
  `fixed_stratified_test`).
- La validación A de DeepH float32 solo es concluyente en δ≈1e-4.
- Los archivados `*_iid*` con fuga temporal documentada no se regeneraron
  (decisión previa; ver `docs/known_limitations.md`).
