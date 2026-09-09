# E-F_001 v2 — estado de la cualificación prospectiva

Alcance: registrar, antes de generar cualquier label confirmatorio nuevo, si existe una ruta
válida para cualificar prospectivamente `validated_covariant_PAO_matrix_derivative` sobre el
checkpoint fijo de graphene-Γ. No se lanzó DFT, reentrenamiento ni orquestador. No se reordenaron
los 12 snapshots supervivientes. `docs/epc_preregistration_v1.json` no se modifica: sigue siendo
la entrada histórica que ya leen `go4_verdict.json` y artifacts previos.

## Resultado

`docs/epc_preregistration_v2.json` (bloque `prospective_qualification_v2`) declara:

- **`historical_selection_status = CONTAMINATED`**, **`GO-4_status = NO_GO`** — sin cambios; ambos
  ya estaban así en `Comparison/results/epc/gamma_epc/graphene/go4_verdict.json`.
- **`route_evaluation.legacy_fixed_candidate.eligible = false`**: la infraestructura ya existente
  (`certify_checkpoint_lineage_and_exclusion.py`) marca ambas estructuras de evaluación EPC
  (`graphene_primitive_cell_equilibrium`, `matbg_rigid_31_30_magic_angle`) como
  `previously_inspected=true` y `blind_holdout=false` en
  `Comparison/results/epc/checkpoint_lineage_exclusion/checkpoint_lineage_exclusion.json`. La ruta
  legado exige que el holdout nunca haya sido usado en inspección; ya lo fue (S25/S26,
  `quantify_checkpoint_derivative_error.py`, sweeps de convergencia), así que la ruta legado queda
  inelegible con independencia del resultado de leakage geométrico.
- **`route_evaluation.clean_reselection.status = not_initiated`**: elegible en principio, pero
  reentrenar con splits `train/selection_validation/threshold_calibration/final_test` separados
  por lineage está fuera del alcance de este documento.
- **`blinding_status = prospective_unblinded`**, **`sealing.external_custody = false`**: el único
  mecanismo de sellado disponible (`frozen_before_commit`) es una referencia de commit local, no
  custodia externa ni timestamp independiente.
- **`final_state = NO_GO_LINEAGE`**: ninguna ruta ejecutable en este repositorio hoy soporta abrir
  un `final_test` confirmatorio válidamente ciego para esta afirmación. Esto no cambia ningún gate
  ya producido (GO-4, `delta_out_closure`, `checkpoint_lineage_exclusion` conservan su veredicto).

## Verificación (sin cambio de comportamiento)

```text
.venv/bin/python -m pytest tests/test_epc_lineage.py tests/test_checkpoint_lineage_exclusion.py -q
  39 passed

.venv/bin/python Comparison/scripts/audit_epc_lineage.py --preregistration docs/epc_preregistration_v2.json \
  --fail-on-unknown --output Comparison/results/epc/pao_flow_audit/lineage_audit_v2.json
  unknown_count=1 (checkpoint sin metrica de seleccion solo-validacion) -- exit 1, igual que con v1

.venv/bin/python Comparison/scripts/certify_checkpoint_lineage_and_exclusion.py
  checkpoint_lineage=NO_GO, final_case_exclusion=PASS, previously_inspected=True para ambas
  estructuras -- exit 0, igual que antes de este cambio
```

Referencias de código actualizadas de `docs/epc_preregistration_v1.json` a
`docs/epc_preregistration_v2.json`: `certify_checkpoint_lineage_and_exclusion.py`
(`DEFAULT_PREREGISTRATION`), `audit_epc_lineage.py` (docstring), `tests/test_epc_lineage.py` y
`tests/test_checkpoint_lineage_exclusion.py` (`REAL_PREREGISTRATION`). Documentos fechados
existentes (`docs/pao_flow_audit_20260906.md`, `docs/epc_overnight_remediation_20260905.md`) no se
reescriben: son registro histórico de comandos ya ejecutados contra v1.

## Qué haría falta para avanzar

1. Ejecutar `clean_reselection` genuino (entrenamiento desde cero con splits separados por
   lineage) si se quiere una afirmación confirmatoria ciega sobre graphene-Γ; o
2. Identificar una estructura/checkpoint que sí cumpla la precondición de no-inspección de la ruta
   legado, en vez de reusar las dos ya evaluadas.

Ninguna de las dos se inició aquí.
