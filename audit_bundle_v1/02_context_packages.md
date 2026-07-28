# Punto 2 — Paquetes de contexto

La auditoría se divide en cinco dossiers. Los cuatro primeros se entregan al
especialista en física computacional por separado. El quinto se audita
localmente como software. Las rutas son relativas a la raíz del repositorio.

Reglas comunes:

- conservar ruta y números de línea;
- incluir archivos completos cuando sean razonables;
- para archivos de más de 3000 líneas, incluir mapa de símbolos y extractos
  completos de las funciones relevantes;
- no incluir checkpoints, matrices, pseudopotenciales binarios ni datasets
  completos en la primera pasada;
- seleccionar evidencia antes de copiarla: una ejecución representativa por
  material, método, tamaño y seed;
- marcar cualquier archivo no versionado como `WORKTREE_ONLY`;
- objetivo de 20–65 mil tokens por subdossier.

## Dossier 1 — Generación física, materiales y muestreo

Preguntas: ¿son físicamente válidas las estructuras y trayectorias?, ¿hay
convergencia, independencia y cobertura configuracional?, ¿las unidades y
condiciones de contorno son consistentes?

Contexto base:

```text
README.md
docs/architecture.md
docs/workflows.md
docs/data_and_outputs.md
docs/known_limitations.md
requirements-graph2mat.txt
```

Código y configuración:

```text
MD/pipeline_config.yaml
MD/scripts/md_pipeline_config.py
MD/scripts/generate_md_dataset.py
AtomDisplacement/pipeline_config.yaml
AtomDisplacement/scripts/atom_displacement_utils.py
AtomDisplacement/scripts/generate_atom_displacement_dataset.py
AtomDisplacement/scripts/generate_generic_cartesian_displacement_dataset.py
AtomDisplacement/scripts/generate_random_cartesian_dataset.py
AtomDisplacement/scripts/normalize_fc_steps.py
AtomDisplacement/scripts/collect_atom_displacement_dataset.py
AtomDisplacement/scripts/run_single_points.py
shared/material_bundle.py
shared/material_presets.py
shared/fdf_materialization.py
shared/siesta_run_fdf.py
shared/graph2mat_material_config.py
materials/*/material.yaml
materials/*/RUN*.fdf
```

Pruebas mínimas:

```text
tests/test_generic_cartesian_displacement.py
tests/test_generic_random_cartesian.py
tests/test_material_bundle.py
tests/test_material_presets.py
tests/test_fdf_materialization.py
tests/test_siesta_material_provenance.py
tests/test_siesta_version_provenance_hardening.py
tests/test_dataset_recipe_helpers.py
tests/test_graphene_vacancy_target.py
tests/test_graphene_hbn_bilayer_train_dataset.py
```

Evidencia que debe añadirse después de seleccionar campañas:

```text
RUN.fdf efectivo y stdout SIESTA
versión y hash de pseudopotenciales y bases
geometrías iniciales/finales representativas
parámetros SCF, malla, k-points y tolerancias
manifiesto de generación
split congelado
distribución de temperaturas, desplazamientos y distancias
```

## Dossier 2 — Equidad Graph2Mat–DeepH y procedencia

Preguntas: ¿ambos modelos reciben el mismo problema físico?, ¿coinciden base,
orden orbital, vectores R, espín, overlap y splits?, ¿la comparación puede
sostener un claim?

Contexto:

```text
docs/graph2mat_deeph_benchmark.md
docs/mixing_and_autograd_validation_contract.md
docs/cross_structure_evaluation.md
Comparison/scripts/g2m_deeph_protocol.py
Comparison/scripts/deeph_config.py
Comparison/scripts/deeph_fair_utils.py
Comparison/scripts/deeph_prediction_adapter.py
Comparison/scripts/deeph_raw_global_equivalence_preflight.py
Comparison/scripts/deeph_split_audit.py
Comparison/scripts/g2m_deeph_test_blindness.py
Comparison/scripts/g2m_deeph_verify_protocol_datasets.py
Comparison/scripts/reference_selection.py
Comparison/scripts/material_provenance.py
shared/joint_artifact_contract.py
shared/benchmark_manifest.py
shared/artifact_signature.py
shared/run_inventory.py
```

Extraer de archivos grandes:

```text
Comparison/scripts/g2m_deeph_runner.py
Comparison/scripts/g2m_deeph_end_to_end_pipeline.py
Comparison/scripts/g2m_deeph_final_workflow.py
```

Solo deben incluirse las funciones que deciden datasets, splits, referencias,
conversión de matrices, configuración de backend y promoción de claims.

Pruebas mínimas:

```text
tests/test_g2m_deeph_protocol.py
tests/test_g2m_deeph_test_blindness.py
tests/test_deeph_split_audit.py
tests/test_deeph_raw_global_equivalence_preflight.py
tests/test_method_provenance_fairness.py
tests/test_joint_artifact_contract.py
tests/test_artifact_signature.py
tests/test_run_inventory.py
tests/test_dataset_compatibility.py
tests/test_cross_repo_integration.py
```

Evidencia:

```text
artifact_validation.json
benchmark_dataset_manifest.json
frozen_split_manifest.json
ORB_INDX y metadatos de base
hashes de estructura, material y dataset
configuración efectiva de ambos backends
mapa de conversión entre representaciones
```

## Dossier 3 — Hamiltonianos, espectros y derivadas

Preguntas: ¿las métricas representan el error físico pretendido?, ¿se trata
correctamente el problema generalizado Hc=ESc?, ¿las derivadas tienen signo,
unidades, stencil y delta correctos?

Código:

```text
Comparison/scripts/evaluate_hamiltonian_metrics.py
Comparison/scripts/g2m_deeph_metrics.py
Comparison/scripts/evaluate_deeph_kpoint_metrics.py
Comparison/scripts/compare_graphene_bands_siesta_g2m_deeph.py
Comparison/scripts/report_graphene_kpoint_evaluation.py
Comparison/scripts/extract_eigenvalues.py
Comparison/scripts/hamiltonian_derivative_stencil.py
Comparison/scripts/build_hamiltonian_derivative_stencils.py
Comparison/scripts/evaluate_hamiltonian_derivative_metrics.py
Comparison/scripts/validate_hamiltonian_derivative_geometry.py
Comparison/scripts/validate_derivative_workflow_artifacts.py
Comparison/scripts/graph2mat_autograd_derivatives.py
Comparison/scripts/run_graph2mat_autograd_derivative_predictions.py
Comparison/scripts/run_deeph_autograd_derivative_predictions.py
Comparison/scripts/g2m_deeph_derivative_gate_check.py
Comparison/scripts/derivative_claim_status.py
Comparison/scripts/ml_vs_siesta/matrices.py
Comparison/scripts/ml_vs_siesta/compare.py
```

Pruebas mínimas:

```text
tests/test_g2m_deeph_metrics.py
tests/test_evaluate_hamiltonian_derivative_metrics.py
tests/test_hamiltonian_derivative_stencil.py
tests/test_hamiltonian_derivative_geometry_validation.py
tests/test_graph2mat_autograd_derivatives.py
tests/test_deeph_autograd_derivatives.py
tests/test_g2m_deeph_derivative_gate_check.py
tests/test_graphene_band_comparison.py
tests/test_metrics_material_compatibility.py
tests/test_incremental_derivative_metrics.py
```

Evidencia:

```text
matrices H_ref, H_pred y S_ref de unas pocas muestras
metadatos de unidades, base, espín, k-point y vector R
CSV de métricas absolutas y relativas
geometrías +/-delta y manifiesto del stencil
resultado autograd y diferencias finitas para los mismos grados de libertad
hermiticidad y residuos del problema generalizado
```

## Dossier 4 — Claims, estadística y resultados archivados

Preguntas: ¿qué afirmaciones están respaldadas?, ¿hay suficientes seeds?,
¿cómo afecta la selección de checkpoint?, ¿existen leakage, multiplicidad o
agregaciones engañosas?

Contexto y código:

```text
docs/known_limitations.md
docs/phase6_hamiltonian_architecture_benchmark.md
docs/derivative_smoke_validation_note.md
Comparison/scripts/g2m_deeph_dataset_size_minimum.py
Comparison/scripts/g2m_deeph_rank_runs.py
Comparison/scripts/g2m_deeph_final_stats.py
Comparison/scripts/g2m_deeph_paper_diagnostics.py
Comparison/scripts/g2m_deeph_report.py
Comparison/scripts/g2m_deeph_gate_check.py
Comparison/scripts/g2m_deeph_topk.py
Comparison/scripts/g2m_deeph_early_stopping.py
Comparison/scripts/g2m_deeph_training_sweep.py
Comparison/scripts/g2m_deeph_release_manifest.py
Comparison/scripts/g2m_deeph_budget.py
Comparison/scripts/analyze_winners.py
Comparison/scripts/aggregate_cross_metrics.py
```

Pruebas mínimas:

```text
tests/test_g2m_deeph_dataset_size_minimum.py
tests/test_g2m_deeph_rank_runs.py
tests/test_g2m_deeph_final_stats.py
tests/test_g2m_deeph_paper_diagnostics.py
tests/test_g2m_deeph_report.py
tests/test_g2m_deeph_gate_check.py
tests/test_g2m_deeph_topk.py
tests/test_g2m_deeph_early_stopping.py
tests/test_g2m_deeph_release_manifest.py
tests/test_cross_structure_evaluation.py
```

Selección de evidencia: construir una tabla con una fila por claim y estas
columnas:

```text
claim_id
material
dataset_hash
split_hash
modelo y commit
seeds
checkpoint_policy
métrica primaria
media, dispersión e intervalo
estado de procedencia
estado del gate
rutas de evidencia
```

No se deben enviar los 316 directorios de resultados. Primero se seleccionan
los claims que aparecen en documentación, UI o informes; después se adjuntan
solo sus manifiestos, métricas y logs relevantes.

## Dossier 5 — Programación y operabilidad

Este dossier se revisa contra el repositorio completo, no con un modelo de
física. Incluye correctitud, seguridad local, errores, llamadas, pruebas,
dependencias, rendimiento y complejidad.

Entradas principales:

```text
docs/architecture.md
docs/development.md
requirements-graph2mat.txt
MD/scripts/main_md.py
AtomDisplacement/scripts/main_atom_displacement.py
Comparison/scripts/pipeline_ui.py
Comparison/scripts/g2m_deeph_runner.py
Comparison/ui/app.js
shared/
tests/
```

Por su tamaño, `pipeline_ui.py`, `app.js`, `g2m_deeph_runner.py` y
`tests/test_comparison_workflow.py` requieren primero un mapa de símbolos,
callers, endpoints, escrituras y subprocessos. No se deben refactorizar solo
por ser grandes: cada hallazgo debe demostrar un defecto, riesgo o reducción
concreta.

## Orden de entrega

1. Dossier 1 para validar que los datos de entrada representan la física
   pretendida.
2. Dossier 2 para validar que la comparación es justa.
3. Dossier 3 para validar cálculos y métricas.
4. Dossier 4 para decidir qué claims sobreviven.
5. Dossier 5 en paralelo, sin mezclar mantenibilidad con validez científica.

Los informes históricos
`docs/audit_corrections_implementation_report.md` y los prompts `scratchpad_*`
se excluyen de la primera pasada para evitar anclaje. Se incorporan después,
durante la reconciliación de hallazgos.
