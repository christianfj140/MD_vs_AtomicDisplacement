```markdown
# GOAL: Corregir implementación cross-structure hasta cumplir el prompt original

Trabaja en:

`/home/christian/repositorios/MD_vs_AtomicDisplacement`

Usa como especificación base:

`docs/# GOAL: Implement strict cross-structure.md`

Ya existe una implementación inicial de cross-structure. Audítala y corrige los incumplimientos detectados. No lances entrenamientos reales ni SIESTA.

## Cambios obligatorios

### 1. Arreglar `train` real por CLI

Actualmente `Comparison/scripts/run_cross_structure_payload.py` llama `runner.start()` y sale enseguida. El runner usa un thread `daemon=True`, así que el entrenamiento puede morir al salir el proceso.

Implementa un comportamiento robusto similar a:

`Comparison/scripts/run_g2m_deeph_payload_once.py`

Para `action="train"`:

- mantener vivo el proceso hasta que el runner termine;
- hacer polling de `runner.status()`;
- persistir status JSON y manifest/logs si procede;
- devolver el returncode real;
- no perder errores;
- no usar `shell=True`.

Añade CLI args razonables tipo:

```bash
--status-json
--manifest-json
--poll-seconds
```

o reutiliza exactamente el patrón existente si encaja mejor.

### 2. Reutilizar composite existente

El prompt pide: “Materialize or reuse the validated composite dataset”.

Ahora `train` siempre intenta rematerializar y falla si `composite_dataset_root` existe con `overwrite=false`.

Implementa:

- si `composite_dataset_root` no existe: materializar;
- si existe y `overwrite=false`: validar que es un composite cross-structure compatible y runner-ready;
- si existe y es válido: reutilizarlo;
- si existe pero no es válido: fallar claro;
- si `overwrite=true`: reconstruir con la política segura existente.

Añade tests.

### 3. Corregir leakage provenance

El leakage report persistido debe probar las identidades fuente, no las rutas ya materializadas dentro del composite.

Corrige:

- persistir el leakage report calculado sobre `source_dataset_root` y `target_dataset_root`;
- guardar una identidad estable por muestra fuente;
- no devolver contadores hardcodeados a `0`;
- verificar:
  - no target en train/validation;
  - no source en test;
  - no `sample_id` materializado duplicado;
  - no identidad canónica fuente duplicada entre splits.

Añade tests que fallen si una identidad fuente se duplica entre splits.

### 4. Registrar sampling differences

El prompt pide registrar diferencias no bloqueantes.

Actualmente `dataset_compatibility.py` deja `sampling_differences=[]` aunque los raw k-grids/lattices difieran pero la densidad sea compatible.

Añade un registro no bloqueante cuando:

- raw Monkhorst-Pack grids difieren;
- lattice vectors/cell dimensions difieren;
- k-point spacing está dentro de tolerancia.

No bloquees si la densidad es compatible. Añade test con primitive vs supercell donde raw k-grid difiere pero spacing es compatible y queda registrado en `sampling_differences`.

### 5. Registrar link/copy real

La provenance actual dice `"linked"` si `link=true`, aunque haya fallback a copy.

Corrige para registrar por muestra o resumen real:

- cuántos artefactos fueron symlink;
- cuántos fueron copy;
- por split o por muestra si es fácil;
- si hubo fallback de symlink a copy.

Añade test usando `link=false` y verifica que provenance dice copied de verdad.

### 6. Hamiltonian target semantics

Revisa si ya hay metadatos disponibles para comprobar explícitamente:

- H-only policy;
- matrix component count;
- spin semantics;
- real/complex representation constraints.

Si están representados en artifacts/provenance/manifests, añade checks fail-closed. Si no están representados, documenta exactamente qué queda no verificable y añade warning/provenance claro.

No inventes metadatos falsos.

### 7. No romper mixing ni runner

Preserva:

- `tests/test_ml_vs_siesta_mixing.py`;
- `tests/test_g2m_deeph_runner.py -k "dataset or reuse or split"`;
- `tests/test_benchmark_manifest.py`;
- el comportamiento existente de mixed datasets.

No toques UI.

## Validación mínima

Ejecuta y reporta:

```bash
.venv/bin/python -m pytest -q tests/test_cross_structure_evaluation.py
.venv/bin/python -m pytest -q tests/test_ml_vs_siesta_mixing.py
.venv/bin/python -m pytest -q tests/test_g2m_deeph_runner.py -k "dataset or reuse or split"
.venv/bin/python -m pytest -q tests/test_benchmark_manifest.py
.venv/bin/python -m py_compile Comparison/scripts/ml_vs_siesta/cross_structure_materialize.py Comparison/scripts/run_cross_structure_payload.py shared/benchmark_manifest.py tests/test_cross_structure_evaluation.py
```

Ejecuta también el payload ejemplo en preview:

```bash
rm -rf Comparison/results/graphene_w90_to_5x5_cross_structure
.venv/bin/python Comparison/scripts/run_cross_structure_payload.py \
  Comparison/config/graphene_w90_to_5x5_cross_structure_preview_payload.json
```

Verifica:

- no se crea output;
- detecta 2 -> 50 átomos;
- reporta counts esperados;
- compatibilidad explícita;
- no lanza SIESTA ni training.

## Criterio de cierre

La tarea solo está terminada cuando:

- los 6 puntos anteriores están corregidos o documentados con evidencia clara si algún metadato no existe;
- los tests nuevos cubren los fallos detectados;
- las regresiones pasan;
- no se lanzó entrenamiento real ni SIESTA;
- el resumen final lista archivos cambiados, tests ejecutados y cualquier limitación real.
```