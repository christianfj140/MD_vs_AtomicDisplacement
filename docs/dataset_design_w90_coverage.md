# w90: cobertura del dataset a presupuesto común

Protocolo de seguimiento fijado antes de entrenar esta tanda (2026-09-21).
Objetivo: comparar datasets diseñados y MD en precisión física y coste, una
vez corregido el truncamiento del entrenamiento. No seguir aumentando épocas
antes de comprobar qué cambia al cambiar los datos.

## Comparación

Cuatro brazos, N=64, semillas de entrenamiento 0–4:

1. Random 1D_in R=0.08: reutilizar los cinco modelos de 4000 épocas.
2. Random 3D R=0.08: reutilizar el pool del finalista eficiente, seed de muestreo 0.
3. Random 3D R=0.12: escalar por 1.5 los desplazamientos del brazo 2, sin
   cambiar direcciones ni orden. Generar 64 etiquetas SIESTA nuevas. R es el
   parámetro de amplitud; registrar también las amplitudes reales de cada muestra.
4. MD: reutilizar las 64 estructuras del split temporal auditado, con nuevos
   modelos. Las trayectorias disponibles siguen siendo deterministas y su
   cobertura llega aproximadamente a 0.025 Å.

Los cuatro brazos comparten arquitectura, pérdida elementwise_mse, precisión
bf16, batch 32, 4000 épocas completas (8000 actualizaciones), cosine T_max=4000,
sin EarlyStopping y checkpoint de menor val_loss. **La validación es el mismo
desarrollo común de 24 estructuras para todos, incluido MD**. Se comparan
datasets con distinto origen, no pipelines con distinta validación.

Contrastes previstos: 3D R0.12 frente a 3D R0.08 (amplitud con direcciones
emparejadas); 3D R0.08 frente a 1D R0.08 (recetas, también cambia la semilla de
muestreo); cada diseñado frente a MD. Cinco semillas miden variabilidad de
entrenamiento sobre pools fijos, no incertidumbre por remuestreo del dataset.

## Evaluación y límites

Evaluar los 20 modelos sobre las mismas 64 estructuras existentes, con bandas
y DOS para todos. Informar por separado 48 sintéticas y 16 frames MD. Mantener
las convenciones de `band_dos_metrics`: bandas Γ–K–M–Γ a ±2 eV, DOS con
σ=0.1 eV a ±3 eV, S de referencia. Reportar todas las semillas, medias y sd.
Revisar amplitudes grandes y dimensionalidad al interpretar diferencias.

Registrar CPU·h SIESTA de entrenamiento y tiempo de pared de entrenamiento
por modelo (GPU compartida, no tiempo activo exclusivo). La comparación
económica debe distinguir coste reutilizado, coste nuevo, validación común y
test común. No afirmar equivalencia con MD sin un margen físico fijado.

El test ya ha sido consultado: son resultados exploratorios, no una nueva
confirmación independiente. No seleccionar checkpoints o semillas por test.
La amplitud adicional se elige tras detectar debilidad a 0.12 Å, por lo que
la adaptación al benchmark debe quedar explícita.

Si una receta ofrece una combinación útil de coste y precisión, el siguiente
paso es su curva N=4,8,16,32,64 a presupuesto de actualizaciones comparable,
manteniendo pools anidados. Una confirmación final requerirá datos nuevos
independientes; no ampliar ahora el barrido indiscriminadamente.

## Ejecución

```bash
.venv/bin/python Comparison/scripts/run_w90_coverage.py
```

La secuencia prepara/verifica el pool, entrena 15 modelos, evalúa los cuatro
brazos y genera `w90_coverage_physics/per_seed.csv` y `summary.csv` dentro de
`Comparison/results/dataset_design_curves_v1/`. Reutiliza los marcadores de
finalización existentes y se detiene antes de evaluar si falla un entrenamiento.
Log de la tanda: `coverage_4k.log` en el mismo directorio de resultados.

## Resultados

La tanda terminó completa: 15/15 entrenamientos nuevos y 20/20 evaluaciones,
sin fallos. Media ± sd entre cinco semillas sobre el test común completo:

| Dataset | H-MAE (meV) | Bandas RMSE (meV) | DOS L1 (%) | CPU·h etiquetas | min/modelo |
|---|---:|---:|---:|---:|---:|
| 1D R0.08 | 34.36 ± 3.98 | 49.73 ± 10.23 | 9.07 ± 1.51 | 0.025 | 35.0 |
| 3D R0.08 | 21.49 ± 3.04 | 38.86 ± 12.20 | 7.25 ± 1.72 | 0.027 | 35.2 |
| **3D R0.12** | **18.67 ± 3.90** | **34.08 ± 8.95** | **5.10 ± 1.40** | **0.023** | 37.1 |
| MD | 49.68 ± 11.30 | 133.97 ± 27.62 | 12.39 ± 2.72 | 0.080 | 36.1 |

El coste SIESTA de R0.12 ligeramente menor que R0.08 es variación de tiempo de
ejecución, no una ventaja de diseño; ambos contienen 64 cálculos equivalentes.
El coste incremental de R0.12 fue 0.023 CPU·h; los otros pools ya existían.

Bootstrap emparejado por semilla y estructura (10 000 réplicas, A−B; negativo
= A mejor):

| A − B, test completo | Δ H-MAE [IC95] meV | Δ bandas [IC95] meV | Δ DOS [IC95] puntos |
|---|---:|---:|---:|
| 3D R0.12 − 3D R0.08 | −2.82 [−5.75, −0.32] | −4.78 [−13.66, +4.72] | −2.15 [−3.94, −0.76] |
| 3D R0.08 − 1D R0.08 | −12.87 [−20.88, −5.78] | −10.86 [−29.26, +7.33] | −1.82 [−4.63, +1.12] |
| 3D R0.12 − MD | **−31.01 [−47.79, −18.13]** | **−99.89 [−231.62, −27.15]** | **−7.29 [−10.54, −4.60]** |

R0.12 mejora H y DOS frente a R0.08 en las cinco semillas; bandas mejora en
3/5 y el intervalo cruza cero. La dimensionalidad 3D reduce H frente a 1D en
5/5 semillas, pero bandas/DOS siguen limitadas por la variación de entrenamiento.
Frente a MD, R0.12 gana las tres métricas en 5/5 semillas.

En los 16 frames MD, R0.12 y MD empatan en H dentro de la incertidumbre
(−2.82 [−5.72, +0.53] meV), mientras R0.12 mejora bandas en −20.34
[−37.32, −5.11] meV y DOS en −4.30 [−7.05, −2.27] puntos. Debe recordarse que
el baseline MD proviene de solo seis trayectorias deterministas y que todos los
brazos seleccionan checkpoint con desarrollo sintético común.

En A=0.12 Å, R0.12 baja H de 49.71 a 33.20 meV y DOS de 17.36% a 9.38%
respecto a R0.08, pero bandas no mejoran (98.72 frente a 107.19 meV). Aumentar
la amplitud de entrenamiento resuelve buena parte del Hamiltoniano y DOS, no
todo el problema espectral de la frontera del dominio.

La frontera reproducible queda encabezada por 3D R0.12: mejor media en las tres
métricas, coste de etiquetas equivalente a los otros diseños y ~3.4× menor que
MD. En coste incremental, 3D R0.08 sigue siendo relevante porque sus etiquetas
ya estaban disponibles; R0.12 compra una mejora adicional por 0.023 CPU·h.

Tablas: `w90_coverage_physics/summary.csv`, `per_seed.csv`,
`paired_comparisons.csv` y `by_domain_axis.csv`. Figura:
`w90_coverage_physics/dataset_comparison.png` (también PDF).

## Curva de aprendizaje de la receta ganadora

Fijada antes de entrenar la siguiente tanda: usar prefijos exactos del pool
3D R0.12 para N={4,8,16,32,64}. N=64 reutiliza los cinco modelos anteriores;
N<64 se entrena con semillas 0–4. Cada tamaño recibe aproximadamente 8000
actualizaciones y 4000 validaciones: N≤32 usa 8000 épocas y valida cada dos;
N=64 usa 4000 épocas y valida cada una. Misma pérdida, arquitectura, precisión,
desarrollo y política de checkpoint. Sin EarlyStopping.

No se añade N=12 a priori; se hará únicamente si aparece una caída localizada
entre N=8 y N=16. La decisión primaria es el menor N que mantenga H, bandas y
DOS próximos a N=64 teniendo en cuenta la dispersión entre cinco semillas.
No se generan etiquetas nuevas. La evaluación reutiliza el test ya consultado,
por lo que la curva es exploratoria y necesitará confirmación independiente si
se convierte en una afirmación principal.
