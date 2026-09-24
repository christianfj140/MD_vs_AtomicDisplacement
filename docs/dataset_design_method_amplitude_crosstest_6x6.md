# Cross-testing de método y amplitud en grafeno 6×6

## Protocolo

Doce dominios (cuatro métodos × tres amplitudes), Train64, Validation48 común, Test32 por dominio y dos semillas. Todos los modelos usan `elementwise_mse`, batch 16, 8000 actualizaciones, 2000 evaluaciones de validación, scheduler coseno y el mejor checkpoint por `val_loss`. Los intervalos no se interpretan inferencialmente con n=2.

**Nota de interpretación:** `R` es la escala o envolvente propia de cada generador, no un RMS de desplazamiento igualado entre métodos. Por tanto, la comparación entre métodos incluye también diferencias en su distribución radial efectiva; la comparación entre amplitudes dentro de un método sí está emparejada por patrón normalizado.

## Resultados principales

- Mejor método medio: **angular_shell_collective**, H-MAE media 3.029 meV.
- Mejor peor caso: **angular_shell_collective R=0.12 Å**, peor H-MAE 2.454 meV.
- Media diagonal (mismo método y amplitud): 1.920 meV.
- Media fuera de diagonal: 3.951 meV.
- Penalización media fuera de diagonal: 2.00× respecto a la diagonal del modelo.
- Mayor penalización: sobol_sparse__R0.03 → angular_shell_collective__R0.12, 14.25× (+33.303 meV).
- Transferencia hacia amplitudes mayores: 7.373 meV; hacia menores: 1.563 meV.

## Resultado por dominio de entrenamiento

| Método | Rtrain (Å) | Diagonal seed 0 | Diagonal seed 1 | Media 12 tests | Peor test |
|---|---:|---:|---:|---:|---:|
| sobol_sparse | 0.03 | 2.117 | 2.909 | 7.914 | 35.815 |
| sobol_sparse | 0.08 | 1.620 | 1.709 | 2.951 | 13.249 |
| sobol_sparse | 0.12 | 1.588 | 1.759 | 1.800 | 5.883 |
| latin_hypercube | 0.03 | 1.242 | 2.667 | 5.633 | 22.339 |
| latin_hypercube | 0.08 | 1.415 | 1.342 | 3.099 | 13.045 |
| latin_hypercube | 0.12 | 1.484 | 1.826 | 2.321 | 7.198 |
| random_cartesian | 0.03 | 2.142 | 3.384 | 7.937 | 35.800 |
| random_cartesian | 0.08 | 1.551 | 1.726 | 2.827 | 12.549 |
| random_cartesian | 0.12 | 1.646 | 1.729 | 1.813 | 5.611 |
| angular_shell_collective | 0.03 | 1.727 | 1.836 | 5.110 | 24.600 |
| angular_shell_collective | 0.08 | 1.840 | 1.915 | 2.030 | 5.373 |
| angular_shell_collective | 0.12 | 2.208 | 2.700 | 1.947 | 2.454 |

## Mayores asimetrías de transferencia

Δ = H-MAE(A→B) − H-MAE(B→A); el signo indica qué dirección transfiere peor.

| A | B | A→B (meV) | B→A (meV) | Δ (meV) |
|---|---|---:|---:|---:|
| sobol_sparse__R0.03 | angular_shell_collective__R0.12 | 35.815 | 1.829 | +33.987 |
| random_cartesian__R0.03 | angular_shell_collective__R0.12 | 35.800 | 1.829 | +33.970 |
| angular_shell_collective__R0.03 | angular_shell_collective__R0.12 | 24.600 | 1.851 | +22.749 |
| latin_hypercube__R0.03 | angular_shell_collective__R0.12 | 22.339 | 1.807 | +20.533 |
| sobol_sparse__R0.03 | angular_shell_collective__R0.08 | 14.552 | 1.633 | +12.918 |
| random_cartesian__R0.03 | angular_shell_collective__R0.08 | 14.379 | 1.633 | +12.746 |
| sobol_sparse__R0.08 | angular_shell_collective__R0.12 | 13.249 | 1.890 | +11.359 |
| latin_hypercube__R0.08 | angular_shell_collective__R0.12 | 13.045 | 1.834 | +11.212 |
| random_cartesian__R0.08 | angular_shell_collective__R0.12 | 12.549 | 1.888 | +10.661 |
| sobol_sparse__R0.03 | sobol_sparse__R0.12 | 10.304 | 1.259 | +9.045 |

Los CSV conservan ambas semillas y las 32 estructuras de cada test. Menor H-MAE y menor Frobenius relativo son mejores.
