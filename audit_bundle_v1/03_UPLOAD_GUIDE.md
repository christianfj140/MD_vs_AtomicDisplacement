# Punto 3 — Entrega al especialista en física computacional

Los contextos están divididos para evitar que código, física y resultados
compitan por atención. No se debe enviar todo en un único mensaje.

## Archivos comunes

Adjuntar en todas las conversaciones:

1. `01_snapshot_and_evidence.md`
2. `03_CONTEXT_INDEX.md`
3. uno de los subdossiers `03A1`–`03E2`

`02_context_packages.md` es una guía interna de selección y no es necesario
adjuntarlo al especialista.

## Orden recomendado

1. `03A1_generation_md_fc.md`
2. `03A2_expanded_recipes.md`
3. `03A3_random_sampling.md`
4. `03B_material_inputs.md`
5. `03C1_protocol_provenance.md`
6. `03C2_splits_provenance.md`
7. `03C3_basis_equivalence.md`
8. `03D1_hamiltonian_spectra.md`
9. `03D2_derivatives.md`
10. `03E1_claims_gates.md`
11. `03E2_dataset_size_statistics.md`

Abrir una conversación nueva para cada dossier. Pedir primero una revisión
independiente y conservar las respuestas sin enseñar las conclusiones de los
dossiers anteriores. La reconciliación cruzada se hace después.

## Qué contiene cada dossier

| Dossier | Contenido |
| --- | --- |
| `03A1` | Generación MD, FC, cartesiana genérica, splits y validación |
| `03A2` | Recetas expandidas, tamaños, temperaturas, seeds y bloques |
| `03A3` | Muestreo cartesiano aleatorio y aislamiento por familias |
| `03B` | Todos los `material.yaml` y `RUN*.fdf`, más manifiestos compactados |
| `03C1` | Protocolo científico y política de evaluación |
| `03C2` | Splits, referencias, contratos y procedencia |
| `03C3` | Equivalencia de base y matrices Graph2Mat–DeepH |
| `03D1` | H/S, k-points, espectros, DOS y métricas |
| `03D2` | Derivadas FD/autograd, geometría, unidades y gates |
| `03E1` | Seeds, intervalos, rankings y promoción de claims |
| `03E2` | Dataset-size minimum, agregación, umbrales y fits |

## Exclusiones deliberadas

No se incluyen en esta primera pasada:

- `docs/audit_corrections_implementation_report.md`;
- `docs/known_limitations.md`;
- `docs/mixing_and_autograd_validation_contract.md`;
- `scratchpad_*`;
- prompts históricos;
- matrices, checkpoints y datasets completos;
- código de UI sin efecto científico directo;
- los repositorios completos de Graph2Mat y DeepH.

Las exclusiones evitan anclaje y exceso de contexto. Si un hallazgo depende de
una pieza ausente, el especialista debe marcarlo como `no evaluable` y pedir la
evidencia exacta; esa evidencia se añadirá en una segunda ronda.

## Reproducción

```bash
python3 audit_bundle_v1/build_blind_contexts.py
python3 audit_bundle_v1/build_blind_contexts.py --check
```

El generador conserva rutas, números de línea y SHA-256. Los JSON enormes se
presentan como vistas compactas con longitud y muestras de sus listas; el hash
del original permite identificar el artefacto completo.
