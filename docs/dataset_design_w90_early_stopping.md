# w90: comprobación de la parada temprana (2026-09-21)

Hipótesis exploratoria: la paciencia de 80 validaciones interrumpe algunos
entrenamientos antes de una mejora tardía con el schedule cosine de 2000 épocas.
La ejecución original con semilla 3 alcanzó 30.90 meV en desarrollo en la época
1293; las otras cuatro alcanzaron 74.52–82.71 meV y pararon entre 315 y 574.

Comparación emparejada: semillas 0–4, mismo pool N=64 de random 1D_in R=0.08,
mismas 24 estructuras de desarrollo, arquitectura, batch 32, precisión bf16,
elementwise_mse y cosine (T_max=2000). La única intervención es retirar
EarlyStopping. Se entrenan 2000 épocas completas y se conserva el checkpoint
de menor val_loss, como en el control. No se seleccionan semillas por resultado.

Analizar H-MAE por semilla, diferencias emparejadas, media y dispersión,
época del mejor checkpoint y tiempo de entrenamiento. Verificar que cada
ejecución termina en la época 1999 y que conserva los hashes de entrenamiento.
Contrastar las curvas anteriores y nuevas durante su tramo común para detectar
variación de ejecución. El test final no interviene en este diagnóstico.

Una mejora consistente apoyaría que la parada temprana limita esta receta;
un resultado negativo no descartaría otras dificultades de optimización.
Los resultados son de desarrollo y no demuestran superioridad sobre MD.

Ejecución reanudable:

```bash
.venv/bin/python Comparison/scripts/dataset_design_curves_v1.py train --stage no_early_stop --systems w90 --parallel 3
```

Los cinco trabajos tienen sufijo `__long_elemmse__no_early_stop`, separado de
los controles existentes, bajo `Comparison/results/dataset_design_curves_v1/runs/w90/`.
Log de la tanda: `Comparison/results/dataset_design_curves_v1/no_early_stop.log`.

## Resultado de 2000 épocas y ampliación a 4000

Las cinco ejecuciones completaron la época 1999. H-MAE en desarrollo:

| Semilla | Parada temprana | 2000 completas | Mejor época nueva |
|---|---:|---:|---:|
| 0 | 74.52 | 38.49 | 1959 |
| 1 | 82.71 | 69.53 | 1993 |
| 2 | 78.10 | 49.32 | 1999 |
| 3 | 30.90 | 22.77 | 1813 |
| 4 | 75.86 | 30.18 | 1919 |

Media ± desviación entre semillas: 68.42 ± 21.20 → 42.06 ± 18.26 meV
(−38.5%). Tiempo medio: 5.8 → 18.3 minutos por modelo. Los hashes de
entrenamiento coinciden, pero las curvas del tramo común no son idénticas:
hay variación numérica entre ejecuciones además de la intervención.

El criterio de posible truncamiento señala a las semillas **0, 1 y 2**, cuyos
mejores checkpoints están en las últimas 41 épocas, no solamente a las de
mayor error. Para evitar seleccionar por resultado, se repiten **las cinco**
desde el inicio con 4000 épocas, sin EarlyStopping y con cosine T_max=4000.
Se conserva todo lo demás y el checkpoint de mejor val_loss. Este contraste
mide presupuesto y horizonte del schedule conjuntamente, no épocas aisladas.
Es exploratorio sobre desarrollo; no se abre el test final.

```bash
.venv/bin/python Comparison/scripts/dataset_design_curves_v1.py train --stage no_early_stop_4k --systems w90 --parallel 3
```

Sufijo nuevo: `__no_early_stop__4k`. Log: `no_early_stop_4k.log` dentro del
directorio de resultados de la campaña. Comparar las cinco semillas con los
controles de 2000 épocas, incluyendo coste, dispersión y proximidad del mejor
checkpoint al nuevo límite antes de concluir si se ha alcanzado convergencia.

## Resultado de 4000 épocas

Completadas las cinco ejecuciones hasta la época 3999. H-MAE en desarrollo:
28.82, 31.33, 27.75, 21.05 y 24.47 meV (semillas 0–4). Media ± sd:
**26.68 ± 4.00 meV**, frente a 42.06 ± 18.26 a 2000 épocas. Mejora media
36.6%; tiempo medio 35.0 frente a 18.3 minutos/modelo. Mejores épocas:
3527, 3631, 3838, 3778 y 3670. Ninguna está en las últimas 41 épocas;
esto reduce los indicios de truncamiento, pero no prueba convergencia.

## Generalización, bandas y DOS sobre el test existente

Evaluados los diez checkpoints (cinco de cada presupuesto) sobre las mismas
64 estructuras: 48 sintéticas y 16 frames MD. Se verificaron los hashes de
checkpoints y la ausencia de geometrías de entrenamiento en el test.
Esta es una evaluación exploratoria de seguimiento: el test fue consultado
en etapas anteriores de la campaña, no es una confirmación independiente nueva.
Los checkpoints se seleccionaron por val_loss de desarrollo, no por este test.

Se mantienen las métricas anteriores: RMSE de bandas por estructura en
Γ–K–M–Γ (60 puntos, estados de referencia a ±2 eV de E_F); DOS a ±3 eV,
malla 24×24×1 y ensanchamiento gaussiano σ=0.1 eV, error L1 relativo.
Los autovalores predichos se calculan con **S de referencia**; se evalúa la
calidad de H, no una predicción independiente de H y S. Cada estructura pesa
igual, y se promedian las métricas por semilla. ± indica sd entre cinco semillas.

| Métrica, test completo | 2000 épocas | 4000 épocas | Reducción media |
|---|---:|---:|---:|
| H-MAE (meV) | 50.74 ± 18.35 | 34.36 ± 3.98 | 32.3% |
| RMSE bandas (meV) | 58.13 ± 4.49 | 49.73 ± 10.23 | 14.5% |
| DOS L1 relativo (%) | 10.80 ± 0.58 | 9.07 ± 1.51 | 16.1% |

Las cinco semillas mejoran en las tres métricas agregadas del test completo.
La dispersión del H-MAE disminuye, pero la de bandas y DOS **aumenta**:
no hay una mejora universal de reproducibilidad física.

| Dominio | H-MAE 2k → 4k (meV) | Bandas 2k → 4k (meV) | DOS 2k → 4k (%) |
|---|---:|---:|---:|
| Sintético (48) | 56.17 → 39.26 | 68.10 → 57.45 | 11.95 → 10.15 |
| Frames MD (16) | 34.45 → 19.68 | 28.23 → 26.54 | 7.36 → 5.82 |

En frames MD la mejora media de bandas es solo 6.0%, y solo 3/5 semillas
mejoran bandas y DOS. La reducción del H-MAE no se traduce proporcionalmente
al espectro. Ejemplo a 4000 épocas: semilla 3 tiene el menor H-MAE del test
(28.37 meV), pero 59.18 meV de bandas; semilla 1 tiene 38.51 meV de H-MAE
y el menor error de bandas, 35.34 meV. No elegir una semilla por este resultado
de test; el ejemplo ilustra que el ranking depende del observable.

La cobertura sigue limitando la generalización: a amplitud sintética 0.12 Å,
el RMSE de bandas baja de 186.49 a 157.88 meV, pero sigue muy por encima de
los 25.67–27.68 meV a 0.02–0.04 Å. La receta entrena con random 1D_in R=0.08.
Cada amplitud contiene 8 estructuras, por lo que son diagnósticos descriptivos.

Conclusión: aumentar el presupuesto de optimización y el horizonte cosine
mejora H y también los observables espectrales en este test, a ~1.9× el tiempo
de entrenamiento. No demuestra equivalencia/superioridad frente a entrenar
con MD: eso requiere comparar estrategias con la misma receta y presupuesto.
El siguiente estudio de datasets debería usar la receta estabilizada y medir
bandas/DOS por dominio, especialmente la región de amplitudes grandes.

Reproducción (no requiere entrenamiento ni SIESTA nuevos):

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python Comparison/scripts/dataset_design_curves_v1.py evaluate-final --systems w90 --stages no_early_stop no_early_stop_4k --parallel 2
.venv/bin/python Comparison/scripts/analyze_w90_training_budget.py
```

Tablas: `Comparison/results/dataset_design_curves_v1/w90_budget_physics/`
(`per_seed.csv`, `summary.csv`, `by_amplitude.csv`). Figura emparejada por
semilla: `paired_physics.png` y `paired_physics.pdf` en el mismo directorio.
