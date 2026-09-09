# C14 / E-F_001-S15 — error derivative del checkpoint frente a la referencia SIESTA certificada

Artifacts:
[`Comparison/results/epc/checkpoint_derivative_error/graphene/checkpoint_derivative_error.json`](../Comparison/results/epc/checkpoint_derivative_error/graphene/checkpoint_derivative_error.json)
y `checkpoint_derivative_error_groups.csv` (66 filas de distribución).
Productor: [`Comparison/scripts/quantify_checkpoint_derivative_error.py`](../Comparison/scripts/quantify_checkpoint_derivative_error.py).
Tests: [`tests/test_quantify_checkpoint_derivative_error.py`](../tests/test_quantify_checkpoint_derivative_error.py).

Convenciones de evidencia: **[CODE]** verificado en este repositorio, **[MEAS]** medido en
este run, **[DESIGN]** decisión pre-registrada.

---

## 1. Veredicto

```text
decision = UNDECIDED
```
La atribución al checkpoint permanece indecidida hasta que el artifact adjunte
el gauge electrostático común, la convergencia de referencia y una cota de
completitud de la base/`Delta_out`.
**[MEAS]** El error del checkpoint sobre `D_H[v]` es del **60–67 % relativo en Frobenius**,
entre **162x y 509x** por encima de la suma en cuadratura de todo lo que no es el modelo
(τ_num de ambos lados, τ_backend y truncación de soporte). Umbral pre-registrado de
adecuación: 5 %.

Las 11 precondiciones numéricas históricas pasan, pero ya no bastan para atribuir el residuo
al modelo: no cierran el gauge común, la convergencia de referencia ni `Delta_out`. Si una
precondición básica falla, el script degrada además el veredicto a `not_evaluable`
([`finalize_verdict`]).

---

## 2. Qué se compara exactamente

| | |
| --- | --- |
| Referencia | `dHSdR.nc` contraído sobre `v` (C06 + C09), en el δ del plateau que GO-2 seleccionó (0.02 Å) |
| Modelo | JVP direccional Graph2Mat (C10/C11), float64, backend **CUDA** (preflight pasado, sin fallback) |
| Espacio índice | `(fila, columna_celda, R)` canónico, el mismo que `read_tshs`; ausencia = cero exacto |
| Direcciones | las 3 certificadas por la campaña, con el `direction_hash` reconstruido y comparado |

**[MEAS] Convención.** El checkpoint predice `H − E_F·S` (la que devuelve sisl), no `H`:
sobre la H de equilibrio el error relativo es 0.2065 contra la primera y 0.4010 contra la
segunda. La referencia se convierte por tanto a

```math
D^{\rm label}[v] = D_H[v] - E_F\,D_S[v] - (\partial_v E_F)\,S,
```

y esa conversión no se asume: coincide con la diferencia central directa de `H − E_F·S`
dentro del ruido GO-2 propagado (residuo 0.0162 vs suelo 0.0331 eV/Å para `atom0000_x`).
Comparar contra el `D_H` sin convertir daría 21.4 eV/Å de discrepancia extra — se conserva
como control en `control_against_unshifted_D_H`.

Tamaño de los términos de convención para `atom0000_x` (Frobenius, eV/Å):
`D_H` 48.26, `E_F·D_S` 11.48, `(∂_v E_F)·S` 0.33, `D_label` 38.72.

---

## 3. Presupuesto de error (eV/Å, Frobenius)

| dirección | ‖D_ref‖ | ‖residuo‖ | rel | τ_num ref (GO-2) | τ_num modelo | τ_backend | soporte |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `atom0000_x` (one-hot) | 38.72 | 25.96 | 0.671 | 0.0309 | 0.1072 | 0.1144 | 0.0015 |
| `random_seed0` (colectiva, held-out) | 18.76 | 11.27 | 0.601 | 0.0202 | 0.0063 | 0.0067 | 0.0007 |
| `translation_x` (acústica, nula) | 0.0036 | 0.0036 | — | 0.0309 | 0 | 0 | 0 |

- **τ_num** = ruido de las dos diferencias finitas: el `tau_FD` medido por GO-2 en el lado
  SIESTA y la dispersión de la diferencia central del propio modelo a lo largo del plateau.
- **τ_backend** = ‖D_jvp − D_frozen‖ del *mismo* modelo. Máximo 0.114 eV/Å, es decir 0.3 %
  de la señal: los dos caminos numéricos del modelo coinciden, luego el desacuerdo con
  SIESTA no es del JVP.
- **soporte**: el grafo del modelo (cutoff de aristas 4.973 Å) contiene 800 de las 1184
  entradas del TSHS; lo que queda fuera pesa 0.0015 eV/Å. La truncación es contabilidad,
  no aprendizaje.
- **dirección nula**: `D_H[traslación uniforme]` vale exactamente 0 en el JVP (invariancia
  traslacional a topología fija) y 0.0036 en la referencia, por debajo del propio τ_num de
  GO-2. Se clasifica como nula y no entra en ninguna estadística relativa.

---

## 4. Distribuciones (dónde está el error)

**[MEAS] Por distancia de par** — el 96 % del residuo vive en el primer vecino:

| d (Å) | n | ‖D_ref‖ | ‖residuo‖ | rel | cuota del residuo |
| --- | --- | --- | --- | --- | --- |
| 0.00 (onsite) | 32 | 20.53 | 1.52 | 0.074 | 0.003 |
| 1.47 (1er vecino) | 96 | 31.90 | 25.45 | **0.798** | **0.961** |
| 2.54 | 192 | 4.53 | 2.25 | 0.495 | 0.007 |
| 2.93 | 96 | 5.97 | 4.36 | 0.730 | 0.028 |
| 3.88 | 192 | 2.00 | 0.30 | 0.149 | 0.000 |
| ≥ 4.40 | 576 | 0.05 | 0.05 | — | 0.000 |

(`atom0000_x`; la dirección held-out da el mismo patrón: 0.959 de cuota en 1.47 Å.)

**Por bloque orbital**: el error relativo es casi plano, 0.62 (p-p) – 0.78 (s-s); ningún
bloque es el culpable aislado. **Por magnitud**: el decil superior de `|D_ref|` concentra
el 92 % del residuo. Es decir, el modelo falla justo donde la derivada es grande, no en la
cola.

**[MEAS] Escala.** El mejor ajuste `α` que minimiza `‖D_ref − α·D_jvp‖` vale 0.617 y 0.657:
el modelo **sobre-responde ~1.5x**, con coseno 0.945 / 0.937 contra la referencia. Quitando
esa escala el residuo baja de 25.96 a 12.63 (y de 11.27 a 6.53). Aproximadamente la mitad
del error en norma es una ganancia global mal calibrada; el resto es error de patrón.

---

## 5. Calibración y fuga

**[DESIGN]** El umbral de adecuación (5 %) está pre-registrado como constante del módulo,
fijado antes de medir. La *transferibilidad* se calibra sólo sobre
`{atom0000_x, translation_x}` y se evalúa sobre `random_seed0`:

- cota calibrada 0.671; held-out 0.601 ⇒ **dentro** de la cota (factor 1.5). El error medido
  en las direcciones de calibración predice el de una dirección colectiva que no vio.
- El caso final de C20 (autovector fonónico `E_2g` en Γ) **no está en ninguno de los dos
  conjuntos**; la campaña no contiene ningún `phonon_eigenmode`, y el check
  `calibration_and_final_case_disjoint` lo afirma en vez de suponerlo.

**No se convierte a un porcentaje de `g`.** La tolerancia τ_model de `g` debe salir de
propagar `δΔ` por el mismo subespacio electrónico (`δg = C_f† δΔ C_i`) en C20; el propio
artifact lo declara en `verdict.not_a_tolerance_on_g`.

---

## 6. Contexto de provenance que condiciona la lectura

**[CODE]** El checkpoint (`tbg_registry_spectral_loss/…/spectral-best-epoch=247`) se entrenó
sobre `Comparison/datasets/tbg_md_plus_registry/n558`, `generation_mode =
merged_bilayer_stackings_md`: instantáneas MD de bicapa, **no** la celda primitiva de
grafeno monocapa. Esta medida es por tanto de *transferibilidad fuera de distribución*, y
así queda registrada en `provenance.model.training_dataset`.

**[CODE]** Se usa la `basis_table` del propio checkpoint (`PointBasis.R` máx 2.4865 Å,
cutoff de aristas 4.973 Å) y no la que el sisl instalado deriva hoy del mismo `C.ion.xml`
(2.5763 Å → 5.153 Å). No es cosmético: con la tabla actual el grafo pasa de 48 a 60 aristas
en esta geometría. Se compara el modelo bajo la topología con la que se ajustaron sus pesos;
ambos valores quedan en el artifact (`provenance.model.checkpoint_point_basis_R_ang` y
`sisl_rebuilt_point_basis_R_ang`); el check `checkpoint_radial_cutoff_<especie>` de
`shared/orbital_contract.py` es quien detecta la deriva.

**[CODE]** Runtime SIESTA `VERIFIED_BINARY_ONLY`; la limitación se propaga desde el reporte
GO-2 al campo `provenance.siesta_runtime`.

---

## 7. Consecuencias

1. **No autoriza retraining ni abre un ticket de fine-tuning.** La decisión permanece
   `UNDECIDED` hasta cerrar gauge, referencia y completitud PAO/`Delta_out`.
2. Las métricas históricas de sobre-respuesta y distribución por bloques se conservan como
   diagnóstico, no como atribución causal al checkpoint.
3. C20 solo puede usarse como diagnóstico PAO bloqueado; no habilita un claim físico ni una
   decisión downstream mientras sus gates obligatorios no pasen.
