# Baseline MD ab initio 6×6 para el presupuesto de etiquetas

Generado el 2026-09-23 13:54 CEST.

## Resultado ejecutivo

- Mínimo exploratorio MD: **Ntrain=64, Nval=48, Ntest=4**.
- Referencia MD N64/V48/Test48: H-MAE **1.661 meV**, bandas **11.53 meV**, DOS L1 **0.0255**.
- Seed de entrenamiento: **0**, igual que el producto cartesiano sintético comparado.
- Se compararon modelos sintéticos y MD sobre exactamente el mismo MD-Test48 congelado.

## Diseño físico

Se reutilizaron tres trayectorias post-NVT de grafeno 6×6 (72 C), una por 150/300/450 K. Los snapshots consecutivos se repartieron en bloques temporales sin geometrías repetidas: Train64 (22/21/21), Validation48 (16 por temperatura) y Test48 (16 por temperatura). No se aplicó thinning por autocorrelación: MD actúa aquí como método de construcción de dataset comparable con Sobol/LHS, igual que en el estudio MD 5×5 previo.

Los splits comparten trayectoria dentro de cada temperatura, pero nunca el mismo snapshot. Por ello el resultado es un **benchmark exploratorio de muestreo MD**, no una estimación de generalización a trayectorias independientes.

La prueba sostenida con tres trayectorias alcanzó 75 °C y el watchdog las pausó; a partir de esa evidencia la producción quedó limitada a dos procesos. No se operó deliberadamente por encima de 75 °C.

## Resultados sobre MD-Test48

| Dataset | Ntrain | Nval | H-MAE (meV) | Band RMSE (meV) | DOS L1 |
|---|---:|---:|---:|---:|---:|
| md | 4 | 8 | 1.747 | 12.53 | 0.0299 |
| md | 4 | 24 | 1.827 | 16.48 | 0.0397 |
| md | 4 | 48 | 1.803 | 7.47 | 0.0253 |
| md | 8 | 8 | 1.880 | 8.28 | 0.0217 |
| md | 8 | 24 | 1.803 | 17.96 | 0.0413 |
| md | 8 | 48 | 1.808 | 18.94 | 0.0432 |
| md | 16 | 8 | 1.886 | 21.88 | 0.0497 |
| md | 16 | 24 | 1.886 | 19.52 | 0.0454 |
| md | 16 | 48 | 1.811 | 10.07 | 0.0280 |
| md | 32 | 8 | 1.913 | 14.61 | 0.0388 |
| md | 32 | 24 | 1.848 | 20.68 | 0.0485 |
| md | 32 | 48 | 1.898 | 17.27 | 0.0444 |
| md | 64 | 8 | 1.724 | 10.44 | 0.0313 |
| md | 64 | 24 | 1.609 | 10.93 | 0.0284 |
| md | 64 | 48 | 1.661 | 11.53 | 0.0255 |
| synthetic | 4 | 8 | 1.886 | 10.79 | 0.0256 |
| synthetic | 4 | 24 | 1.972 | 18.22 | 0.0418 |
| synthetic | 4 | 48 | 1.847 | 9.90 | 0.0232 |
| synthetic | 8 | 8 | 1.754 | 30.84 | 0.0687 |
| synthetic | 8 | 24 | 1.623 | 30.65 | 0.0672 |
| synthetic | 8 | 48 | 1.735 | 28.22 | 0.0630 |
| synthetic | 16 | 8 | 1.590 | 19.21 | 0.0440 |
| synthetic | 16 | 24 | 1.665 | 30.70 | 0.0674 |
| synthetic | 16 | 48 | 1.608 | 24.41 | 0.0541 |
| synthetic | 32 | 8 | 1.552 | 20.68 | 0.0472 |
| synthetic | 32 | 24 | 1.530 | 29.98 | 0.0668 |
| synthetic | 32 | 48 | 1.505 | 32.23 | 0.0714 |
| synthetic | 64 | 8 | 1.476 | 27.35 | 0.0611 |
| synthetic | 64 | 24 | 1.626 | 37.33 | 0.0831 |
| synthetic | 64 | 48 | 1.514 | 32.69 | 0.0728 |

## Coste

La frontera de coste usa CPU·h SIESTA de construcción (train+validación) y separa equilibrado de producción. Los RUN.out copiados dentro de frames no se vuelven a contabilizar. `cost_frontier.csv` contiene 30 puntos; test, pasos intermedios y descartados se informan aparte en `trajectory_diagnostics.csv`.

## Interpretación

Los intervalos asociados a Ntest son **intervalos empíricos de submuestreo**, no intervalos de confianza de generalización. Los 15 modelos MD usan exactamente batch 16, `elementwise_mse`, 8000 actualizaciones, 2000 validaciones, scheduler cosine y mejor checkpoint por `val_loss`, como el producto cartesiano sintético.

## Artefactos

Todos los artefactos están en `Comparison/results/dataset_design_curves_v1/md_label_budget_6x6/`; la UI Dataset Design consume directamente esos CSV/JSON.
