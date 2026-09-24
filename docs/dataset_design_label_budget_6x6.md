# Presupuesto de etiquetas 6×6 para Graph2Mat

Generado el 2026-09-22 17:41 CEST. Estudio exploratorio; no se generaron etiquetas SIESTA nuevas.

## Resultado ejecutivo

- Combinación mínima según las reglas congeladas: **Ntrain=4, Nval=8, Ntest=16 (Ntotal=28)**.
- Cumplen el criterio estricto de equivalencia **38/90** celdas del producto cartesiano.
- Referencia N64/V48/Test48: H-MAE **1.433 meV**, bandas **31.56 meV**, DOS L1 **0.0691**.
- Menor H-MAE observado: N32/V24, **1.378 meV**.
- Ntrain=32 fue **incluido** por la regla previa basada en Test48.

## Método

Se cruzaron Ntrain=[4, 8, 16, 32, 64], Nval=[8, 24, 48] y Ntest=[4, 8, 16, 24, 32, 48] con seed de entrenamiento 0. V8⊂V24⊂V48 se eligió solo por familia, dimensionalidad y amplitud. Todos los modelos usan el mismo pool Sobol 3D R=0.12, `elementwise_mse`, batch 16, 8000 actualizaciones, 2000 validaciones y scheduler cosine.

Cada modelo se evaluó una vez en Test48. Los tamaños de test se obtuvieron mediante 10.000 permutaciones emparejadas y anidadas, equilibradas por dimensionalidad. Los percentiles descritos son **intervalos empíricos de submuestreo**, no intervalos de confianza de generalización.

Criterio operativo de equivalencia: H≤1.05×referencia, bandas≤1.10×, DOS≤1.10×, q90(|Δ H|)≤10 % y ausencia de inversión sistemática (más del 50 %) entre modelos cuya diferencia completa supera el 5 %. Quedar fuera de estos márgenes no invalida un resultado ni significa que el modelo sea malo; solo impide considerarlo equivalente a N64/V48 para seleccionar el presupuesto mínimo. Las diferencias menores no se usan para forzar un ranking.

## Test48 por modelo

| Ntrain | Nval | H-MAE (meV) | Band RMSE (meV) | DOS L1 | Mejor update |
|---:|---:|---:|---:|---:|---:|
| 4 | 8 | 1.495 | 14.28 | 0.0283 | 7916 |
| 4 | 24 | 1.495 | 20.73 | 0.0420 | 7888 |
| 4 | 48 | 1.444 | 12.93 | 0.0252 | 8000 |
| 8 | 8 | 1.522 | 31.06 | 0.0665 | 7700 |
| 8 | 24 | 1.489 | 28.97 | 0.0629 | 7720 |
| 8 | 48 | 1.458 | 28.37 | 0.0605 | 8000 |
| 16 | 8 | 1.540 | 20.69 | 0.0464 | 7808 |
| 16 | 24 | 1.680 | 29.99 | 0.0666 | 7796 |
| 16 | 48 | 1.622 | 24.94 | 0.0551 | 7760 |
| 32 | 8 | 1.382 | 21.25 | 0.0467 | 7996 |
| 32 | 24 | 1.378 | 29.64 | 0.0649 | 7976 |
| 32 | 48 | 1.414 | 31.14 | 0.0684 | 7900 |
| 64 | 8 | 1.392 | 26.79 | 0.0594 | 7864 |
| 64 | 24 | 1.452 | 35.98 | 0.0791 | 7644 |
| 64 | 48 | 1.433 | 31.56 | 0.0691 | 7904 |

## Interpretación y límites

El mínimo reduce el número de etiquetas SIESTA necesarias para train+val+test dentro de este universo sintético 6×6. El test se ha consultado históricamente y la selección usa una sola seed: el resultado debe llamarse **mínimo exploratorio para grafeno 6×6 sintético**. Una trayectoria MD 6×6 independiente y una segunda seed del mínimo serían la confirmación posterior; no forman parte de este barrido.

## Artefactos

- `Comparison/results/dataset_design_curves_v1/label_budget_6x6/manifest.json`
- `model_results.csv`, `test_subsampling.csv`, `cartesian_results.csv`, `minimum.json`
- UI: Dataset Design → Resultados científicos consolidados.
