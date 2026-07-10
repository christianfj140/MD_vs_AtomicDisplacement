# Informe de implementación: correcciones de auditoría (mezcla + autograd)

Fecha: 2026-07-10. Goal: `docs/# GOAL: corregir y validar end-to-end la.md`.
Contrato técnico: `docs/mixing_and_autograd_validation_contract.md`.

## 1. Estado inicial

| Repo | SHA | Rama | Dirty |
|---|---|---|---|
| MD_vs_AtomicDisplacement | `d318394f` | main | sí (trabajo en curso del usuario: logs streaming, hyperparams, meV, --split) |
| DeepH-pack | `4fd2f435` | agent/deeph-autograd-jvp | sí (pred_ham.py new_sp ValueError, acsf.py; tests/ sin trackear) |
| graph2mat | `1a131f1c` | hamiltonian-spin-colineal-support | limpio |

Imports efectivos verificados: `.venv` principal (Python 3.12.3, torch
2.11.0+cu130, float32) importa graph2mat desde el checkout inspeccionado;
`DeepH-pack/.venv` importa el checkout de DeepH inspeccionado. Capability
DeepH encontrada: JVP forward-mode real (commit 4fd2f43), sin placeholder.

## 2. Revisión de la auditoría

| Hallazgo | Estado real | Acción |
|---|---|---|
| DeepH placeholder NaN | falso para el checkout actual (JVP real desde 4fd2f43) | contrato de capacidades + preflight fail-closed |
| Ceros silenciosos en direcciones no calculadas | confirmado | sentinel NaN (`hamiltonians_grad_pred_v2`) + máscara en info.json + rechazo en runner |
| Spinful assert tardío | confirmado | `NotImplementedError` antes de calcular; forma spinful muerta eliminada |
| DeepH autograd vs FD | ya corregido (test standalone) | añadido report JSON/CSV, chequeo topología, gates pass/warning/fail/not_run |
| Alineación base DeepH-SIESTA | mayormente corregido (adapter + evidencia 10 checks) | añadidos `basis_transform_sha256`, hashes ORB_INDX/orbital_types y `supercell_order` al layout |
| Graph2Mat jacobiano completo | ya corregido | añadidas regla de suma traslacional + hermiticidad por bloques D(R)=D(−R)† |
| Cache stale sin firma | confirmado | `input_signature_sha256` + `cached_result_status` en ambos runners |
| Ghost binario manual | confirmado | 5 estados con evidencia física; el caso real es `proven_inactive` sin override |
| Fingerprints target/DFT/k-density | confirmado (no existían) | `dataset_compatibility.py` + report + bloqueos |
| Test small-only / leakage | parcialmente corregido (fixed_common_test previo) | `fixed_stratified_test`, `evaluation_scope`, guard source-test, provenance por snapshot |
| Ratio sobre total, sin composición real | confirmado | ratio sobre train pool; bloque `composition` completo; rounding registrado |
| Loss sesgada a estructuras grandes | confirmado (medido 4×) | `training_weighting_policy` en G2M y DeepH + cableado E2E |
| Materialización no transaccional | confirmado (`all_valid` no bloqueaba) | `.partial-<uuid>` + rename tras validar; fallo ⇒ sin dataset aparente |
| Claims sin escalera | parcialmente (gate propio) | `derivative_claim_status.py` (5 niveles) + `comparison_kind` en manifests |
| Métricas dominadas por large | confirmado | `_micro_macro_domain` + Frobenius normalizado por elemento |
| Payloads sin schema | confirmado | `mixing_payload_schema_v2` + migración v1 + prevalidación fail-closed |

## 3. Cambios por repositorio

### MD_vs_AtomicDisplacement (nuevos)
`shared/run_inventory.py`, `shared/artifact_signature.py`,
`Comparison/scripts/ml_vs_siesta/dataset_compatibility.py`,
`Comparison/scripts/ml_vs_siesta/mixing_payload_schema.py`,
`Comparison/scripts/derivative_claim_status.py`,
`docs/mixing_and_autograd_validation_contract.md`, este informe,
`Comparison/config/ml_vs_siesta_mixing_sweep_20_500_stratified_per_structure_payload.json`,
tests: `test_run_inventory.py`, `test_artifact_signature.py`,
`test_dataset_compatibility.py`, `test_derivative_claim_status.py`,
`test_mixing_payload_schema.py`, `test_cross_repo_integration.py`.

### MD_vs_AtomicDisplacement (modificados)
`ml_vs_siesta/{mixed_dataset_materialize,dataset_mixing,mixing_sweep,plot_mixing_mae_vs_size}.py`,
`run_deeph_autograd_derivative_predictions.py` (preflight, finitud, firma,
inventario, hermiticidad por bloques),
`run_graph2mat_autograd_derivative_predictions.py` (firma, inventario,
traslación, hermiticidad por bloques),
`run_hamiltonian_derivative_predictions.py` (hashes de transformación,
supercell_order), `hamiltonian_derivative_stencil.py`
(`sparse_blockwise_hermiticity_defect`, Frobenius normalizado,
`dh_matrix_rows`), `evaluate_hamiltonian_derivative_metrics.py`
(`comparison_kind`, inventario, `_micro_macro_domain`),
`graph2mat_autograd_derivatives.py` (`translation_sum_rule_metrics`,
`supercell_order_from_sisl_matrix`), `deeph_config.py` y
`g2m_deeph_runner.py` (overrides de weighting), `pipeline_ui.py`
(`/api/deeph/capabilities`, `/api/run-inventory`, weighting E2E,
prevalidación), `run_mixing_e2e_payload_once.py` (ídem), `ui/app.js`
(x=train real + tooltip composición/seeds/scope), tests actualizados.

### DeepH-pack
`deeph/inference/capability.py` (nuevo), `deeph/inference/pred_ham.py`
(sentinel NaN, finitud, spinful fail-closed, info.json v2),
`deeph/utils.py` (`masked_mse_per_graph`), `deeph/kernel.py`
(`training_weighting_policy`), `deeph/default.ini`,
`tests/test_autograd_capability.py`, `tests/test_training_weighting_policy.py`,
`tests/test_autograd_vs_finite_difference.py` (report+topología+gate).

### graph2mat
`src/graph2mat/core/data/metrics.py` (`block_type_mse_per_structure`,
`block_type_mse_per_domain`, `loss_kwargs` funcionales),
`src/graph2mat/core/data/tests/test_per_structure_losses.py`.

## 4-5. Cambios matemáticos y físicos

- Hermiticidad en espacio real por bloques: D_ij(R) = D_ji(−R)† sobre el
  layout rectangular de supercelda (la métrica cuadrada devolvía NaN).
- Regla de suma traslacional Σ_I ∂H/∂R_Iα ≈ 0 (frame-independiente).
- Frobenius normalizado por elemento (‖err‖_F/√N) comparable entre tamaños.
- k-density por eje |b_i|/N_i (20×20 primitiva ≡ 4×4 en supercelda 5×5).
- Loss per-structure/per-domain matemáticamente L = (1/N_s)Σ_s L_s.

## 6-8. Datasets, métricas, UI

Ver contrato §3-§9. UI: eje x = train real, tooltip con composición/seeds/
scope/política, endpoints de capabilities e inventario.

## 9. Tests ejecutados

| Comando | Resultado |
|---|---|
| `pytest tests/` (19 archivos dirigidos, repo MD) | 336 passed, 1 skipped (+2 arreglos de fakes tras el preflight) |
| `DeepH-pack: pytest tests/ (sin standalone)` | 13 passed |
| `DeepH-pack: python tests/test_autograd_vs_finite_difference.py` | OK — float64 1.03e-7/1.14e-9/4.13e-9 (δ=1e-4/1e-5/1e-6, cosine≈1, topología ok, gate pass); float32 3.26e-2 (δ=1e-4) |
| `graph2mat: pytest core/data/tests/test_per_structure_losses.py + test_hamiltonian_training_losses.py` | 28 passed |
| `pytest tests/test_cross_repo_integration.py` (backend DeepH real, sin mocks) | 4 passed |

## 10. Tests no ejecutados / fallos preexistentes

- `graph2mat core/data/tests`: 13 fallos + 1 error de colección
  (`test_h_dimension.py` con ruta absoluta inexistente) **preexistentes en el
  HEAD limpio de la rama del usuario** (verificado restaurando metrics.py).
- `tests/test_g2m_deeph_ui.py`: 3 fallos que comparan `index.html` (sin
  modificar) con expectativas del test modificado por el usuario —
  preexistentes en el working tree.
- No se ejecutaron campañas SIESTA, entrenamientos completos ni sweeps reales
  (prohibido por coste).

## 11. Resultados del smoke

- **DeepH A (autograd vs FD, checkpoints smoke reales)**: gate `pass` float64
  (mejor 1.1e-9); float32 3.3e-2; reports en
  `DeepH-pack/work/autograd_sanity_f64|f32/autograd_fd_report.{json,csv}`.
- **Graph2Mat A**: test con checkpoint real
  (`test_autograd_jacobian_matches_model_finite_difference_real_checkpoint`) pasado en esta sesión.
- **Mixing real (20 small / 20 large de grafeno, `fixed_stratified_test`)**:
  train 13 (7 small + 6 large), test 2+2, scope `small_and_large`, sin
  leakage, `proven_inactive` sin override, compat report sin bloqueos,
  validación `validated`, provenance v2 con inventario. Composición real del
  `add 0.4`: 46% large por snapshots, 95.5% por átomos, **99.5% por elementos**.
- El primer intento del smoke falló correctamente (fail-closed) por no incluir
  los ids de test reservados.

## 12. Riesgos pendientes

Ver contrato §18. Además: los payloads paper-ready del usuario siguen con
`legacy_elementwise` + `fixed_common_test` (intención preservada; el payload
`..._stratified_per_structure_payload.json` es la variante recomendada);
resultados `paper_ready` requieren repos `pinned_clean` (hoy `pinned_dirty`).

## 13. Commits locales creados

Ninguno (cambios dejados en el working tree para revisión; los tres repos
tenían trabajo del usuario sin commitear que no debía mezclarse).

## 14. Veredicto actualizado

| Subsistema | Veredicto |
|---|---|
| Mezcla `add` | correcto (ratio sobre train pool, composición real persistida) |
| Mezcla `replace` | correcto con limitaciones (cap registrado; total entrenable constante) |
| Graph2Mat autograd | correcto (A validado con checkpoint real; invariancias medidas) |
| DeepH autograd | correcto con limitaciones (float64 probado; float32 solo δ≈1e-4) |
| SIESTA FD | correcto (stencils centrales previos + hashes; sin cambios de fondo) |
| Métricas | correcto con limitaciones (derivadas micro/macro/domain; H absoluta legacy) |
| UI | correcto con limitaciones (composición/capabilities expuestas; panel completo de badges pendiente de frontend) |
| Workflow end-to-end | correcto (prevalidación → materialización transaccional → entrenamiento con política → métricas etiquetadas) |
