# Informe final — construcción del dataset para Graph2Mat

Generado automáticamente el 2026-09-22 14:55 CEST después de completar las campañas w90 y 6×6.

## 1. Pregunta científica

El objetivo es construir el conjunto de entrenamiento más barato que permita a Graph2Mat reproducir Hamiltonianos y observables electrónicos con precisión próxima a la referencia SIESTA/MD. Se estudia la frontera entre coste de etiquetado, número de estructuras, cobertura geométrica y precisión; no se pretende comparar directamente los errores absolutos de w90 (celda primitiva) y 6×6 (72 átomos).

## 2. Resumen ejecutivo

- En **w90**, el mejor dataset probado es **random 3D R=0.12 Å**, con H-MAE **18.67 ± 3.90 meV**, Frobenius relativo **0.84 ± 0.12%**, bandas **34.08 ± 8.95 meV** y DOS L1 **5.10 ± 1.40%**.
- Ese dataset cuesta **0.023 CPU·h** de etiquetas, frente a **0.080 CPU·h** del baseline MD. La comparación emparejada con MD se resume abajo y favorece al dataset diseñado en las tres métricas globales.
- La curva w90 sitúa el primer N dentro del 5% de N=64 en **N=64** para H-MAE. Deben mirarse también bandas y DOS: una H similar no garantiza el mismo espectro.
- En **6×6**, la comparación a presupuesto común selecciona **Sobol 3D R=0.12 Å**. Su curva sitúa el primer N dentro del 5% de N=64 en **N=32**.
- La intervención de entrenamiento decisiva fue cambiar `block_type_mae` por **`elementwise_mse`**: en 6×6 redujo el H-MAE de aproximadamente 5.3 a 2.0 meV. Aumentar ancho, profundidad o resolución angular no produjo una mejora robusta.

## 3. Protocolo común

- Etiquetas electrónicas: SIESTA; modelo: Graph2Mat/MACE.
- Comparaciones finales: cinco semillas de entrenamiento.
- Pérdida: `elementwise_mse`; scheduler cosine.
- Curvas nuevas: **8000 actualizaciones de gradiente y 2000 validaciones** para cada N, evitando confundir épocas con cantidad de optimización.
- Selección de receta con desarrollo común; evaluación posterior sobre el mismo test congelado.
- Métricas: H-MAE, RMSE de bandas Γ–K–M–Γ en ±2 eV de E_F y diferencia L1 relativa de DOS en ±3 eV.
- Incertidumbre: dispersión entre semillas y bootstrap emparejado por semilla y estructura.

## 4. Campaña principal

| Sistema | Dataset | N | H-MAE (meV) | Bandas RMSE (meV) | CPU·h SIESTA |
| --- | --- | --- | --- | --- | --- |
| w90 | precision: random_cartesian__1D_in__k2__r0.080__res64__seed2 | 64 | 90.94 | 101.1 | 0.025 |
| w90 | efficient: random_cartesian__3D__k2__r0.080__res64__seed0 | 64 | 81.62 | 102.9 | 0.027 |
| w90 | alternative: sobol_sparse__3D__k2__r0.050__res64__seed2 | 64 | 83.22 | 130.7 | 0.028 |
| w90 | LHS: latin_hypercube__3D__k2__r0.030__res64__seed2 | 64 | 129.43 | — | 0.025 |
| w90 | MD baseline: md_w90 | 64 | 102.79 | 154.3 | 0.08 |
| 6x6 | precision: sobol_sparse__1D_in__R0.12__d64 | 64 | 6.31 | 39.7 | 1.086 |
| 6x6 | efficient: sobol_sparse__3D__R0.08__d64 | 64 | 7.79 | 69.5 | 1.024 |
| 6x6 | alternative: random_cartesian__1D_in__R0.12__d64 | 64 | 6.39 | 44.1 | 1.07 |
| 6x6 | LHS: latin_hypercube__2D_in__R0.08__d64 | 64 | 7.29 | 68.1 | 0.909 |

Esta campaña estableció la frontera inicial, pero empleaba `block_type_mae`. Sus cifras sirven como historial y diagnóstico, no como estimación final de la mejor receta.

## 5. Diagnósticos de entrenamiento

- **Función de pérdida:** `elementwise_mse` mejoró 6×6 por un factor aproximado de 2.6 y redujo fuertemente la dispersión.
- **Presupuesto w90:** duplicar el horizonte de 2000 a 4000 épocas mejoró las cinco semillas, aunque la ganancia espectral fue menor que la ganancia en H.
- **Scheduler:** cosine redujo la varianza; se mantuvo.
- **Ancho, profundidad y `max_ell`:** no mostraron mejora robusta. El modelo 6×6 ancho llegó a ~30.8 GiB de VRAM.
- **Calidad de etiquetas:** el ruido estimado (~0.01 meV) es varios órdenes inferior al error del modelo; no explica el techo.

| Métrica | 2000 épocas | 4000 épocas | Reducción | Semillas que mejoran |
| --- | --- | --- | --- | --- |
| H_MAE_meV | 50.740 | 34.365 | 32.3% | 5/5 |
| rel_Frob | 0.021 | 0.014 | 33.6% | 5/5 |
| band_rmse_meV | 58.132 | 49.726 | 14.5% | 5/5 |
| dos_rel_L1 | 0.108 | 0.091 | 16.1% | 5/5 |

## 6. Cobertura del dataset w90

| Dataset | N | semillas | H-MAE (meV) | Frobenius relativo (%) | Bandas RMSE (meV) | DOS L1 (%) |
| --- | --- | --- | --- | --- | --- | --- |
| MD | 64 | 5 | 49.68 ± 11.30 | 1.89 ± 0.35 | 133.97 ± 27.62 | 12.39 ± 2.72 |
| random 1D in-plane R=0.08 Å | 64 | 5 | 34.36 ± 3.98 | 1.38 ± 0.16 | 49.73 ± 10.23 | 9.07 ± 1.51 |
| random 3D R=0.08 Å | 64 | 5 | 21.49 ± 3.04 | 0.90 ± 0.12 | 38.86 ± 12.20 | 7.25 ± 1.72 |
| random 3D R=0.12 Å | 64 | 5 | 18.67 ± 3.90 | 0.84 ± 0.12 | 34.08 ± 8.95 | 5.10 ± 1.40 |

Comparación emparejada del mejor diseño frente a MD:

| Métrica | Δ 3D R=0.12 − MD | IC95 | Semillas favorables |
| --- | --- | --- | --- |
| H_MAE_meV | -31.010 | [-47.788, -18.127] | 5/5 |
| rel_Frob | -0.010 | [-0.017, -0.006] | 5/5 |
| band_rmse_meV | -99.888 | [-231.625, -27.149] | 5/5 |
| dos_rel_L1 | -0.073 | [-0.105, -0.046] | 5/5 |

La ampliación 3D de R=0.08 a R=0.12 mejora de forma resuelta H y DOS. La mejora media de bandas frente a 3D R=0.08 no queda resuelta por el intervalo de confianza. En los propios frames MD la ventaja en H se convierte en empate: la ventaja global proviene principalmente de cubrir desplazamientos fuera del dominio estrecho de estas trayectorias MD.

## 7. Curva coste–precisión w90

| Dataset | N | semillas | H-MAE (meV) | Frobenius relativo (%) | Bandas RMSE (meV) | DOS L1 (%) |
| --- | --- | --- | --- | --- | --- | --- |
| random 3D R=0.12 Å | 4 | 5 | 33.81 ± 3.90 | 1.38 ± 0.18 | 51.85 ± 27.28 | 8.71 ± 2.31 |
| random 3D R=0.12 Å | 8 | 5 | 24.28 ± 3.98 | 1.01 ± 0.18 | 38.47 ± 12.44 | 6.91 ± 2.07 |
| random 3D R=0.12 Å | 16 | 5 | 23.51 ± 1.31 | 0.98 ± 0.07 | 30.75 ± 2.93 | 5.72 ± 0.93 |
| random 3D R=0.12 Å | 32 | 5 | 21.13 ± 4.52 | 0.87 ± 0.12 | 41.47 ± 14.20 | 6.58 ± 2.32 |
| random 3D R=0.12 Å | 64 | 5 | 18.67 ± 3.90 | 0.84 ± 0.12 | 34.08 ± 8.95 | 5.10 ± 1.40 |

Esta tabla es la base para elegir N: debe usarse el menor N cuyo error y dispersión sean aceptables en **las tres métricas**, no solo H-MAE.

## 8. Comparación y curva 6×6

| Dataset | N | semillas | H-MAE (meV) | Frobenius relativo (%) | Bandas RMSE (meV) | DOS L1 (%) |
| --- | --- | --- | --- | --- | --- | --- |
| Sobol 1D in-plane R=0.12 Å | 64 | 5 | 1.90 ± 0.09 | 0.78 ± 0.03 | 20.15 ± 4.10 | 4.37 ± 0.92 |
| Sobol 3D R=0.08 Å | 64 | 5 | 2.18 ± 0.22 | 0.90 ± 0.08 | 27.59 ± 10.22 | 5.07 ± 1.82 |
| Sobol 3D R=0.12 Å | 64 | 5 | 1.57 ± 0.10 | 0.61 ± 0.05 | 23.01 ± 6.18 | 4.77 ± 1.68 |

| Dataset | N | semillas | H-MAE (meV) | Frobenius relativo (%) | Bandas RMSE (meV) | DOS L1 (%) |
| --- | --- | --- | --- | --- | --- | --- |
| Sobol 3D R=0.12 Å | 4 | 5 | 1.75 ± 0.20 | 0.68 ± 0.08 | 20.46 ± 6.55 | 4.32 ± 1.75 |
| Sobol 3D R=0.12 Å | 8 | 5 | 1.76 ± 0.19 | 0.67 ± 0.07 | 20.92 ± 4.87 | 4.26 ± 1.18 |
| Sobol 3D R=0.12 Å | 16 | 5 | 1.67 ± 0.15 | 0.64 ± 0.04 | 17.89 ± 4.34 | 3.75 ± 1.06 |
| Sobol 3D R=0.12 Å | 32 | 5 | 1.65 ± 0.17 | 0.63 ± 0.05 | 19.74 ± 6.69 | 4.06 ± 1.59 |
| Sobol 3D R=0.12 Å | 64 | 5 | 1.57 ± 0.10 | 0.61 ± 0.05 | 23.01 ± 6.18 | 4.77 ± 1.68 |

No se incluye un baseline MD 6×6 porque el repositorio no contiene trayectorias de esa supercelda. Inventarlo o comparar con MD w90 no sería físicamente válido.

## 9. Qué significan los resultados

1. Una vez estabilizada la optimización, **la cobertura geométrica y la amplitud importan más que añadir configuraciones redundantes**.
2. El coste de SIESTA puede reducirse seleccionando un N próximo al punto de saturación; el ahorro exacto debe leerse junto con bandas y DOS.
3. Cinco semillas son necesarias: varios efectos que parecían prometedores con tres semillas desaparecieron al ampliar la réplica.
4. H-MAE es útil para seleccionar, pero solo correlaciona de forma moderada con bandas/DOS; los observables físicos deben validarse explícitamente.

## 10. Limitaciones

- Los seguimientos son exploratorios y reutilizan un test ya consultado; una publicación requeriría un segundo test independiente.
- Las 410 configuraciones MD disponibles proceden de solo seis trayectorias deterministas; no representan seis réplicas estadísticas independientes.
- El dominio MD llega aproximadamente a 0.025 Å, mientras los datasets diseñados cubren hasta 0.08–0.12 Å: parte de la comparación es también una comparación de dominio.
- w90 y 6×6 tienen dimensionalidad y distribución de bloques distintas; no se deben comparar sus H-MAE absolutos como si fueran el mismo problema.
- No se demuestra equivalencia dinámica con MD: se demuestra precisión electrónica sobre un test compartido y una frontera coste–precisión favorable.

## 11. Guion recomendado para explicarlo

1. **Objetivo:** aproximar la referencia electrónica reduciendo el número de cálculos SIESTA.
2. **Control metodológico:** mismo test, cinco semillas, mismo presupuesto en actualizaciones y tres métricas físicas.
3. **Hallazgo de entrenamiento:** `elementwise_mse` elimina el principal techo en 6×6.
4. **Hallazgo de dataset:** 3D y R=0.12 mejoran w90; más cobertura es más útil que densidad redundante.
5. **Frontera coste–precisión:** mostrar las dos curvas y escoger N según el margen físico aceptable.
6. **Cautelas:** MD limitado, seguimiento exploratorio y necesidad de confirmación independiente.

## 12. Artefactos

- UI: pestaña **Dataset Design → Resultados científicos consolidados**.
- Tablas w90: `w90_budget_physics/`, `w90_coverage_physics/`, `w90_coverage_curve/`.
- Tablas 6×6: `coverage_6x6_physics/`, `coverage_6x6_curve/`.
- Figuras principales: `Comparison/results/dataset_design_curves_v1/figures/`.
- Pre-registro y metodología completa: `docs/dataset_design_curves_v1_preregistration.md`.
