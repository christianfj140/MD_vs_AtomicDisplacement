# DATASET-DESIGN-CURVES-V1 — Resultados

Campaña ejecutada el 2026-09-19/20 siguiendo `docs/gola_para_claude.md`, con las
reglas congeladas antes de entrenar en
`docs/dataset_design_curves_v1_preregistration.md` (commit `5043fdba`; las
desviaciones posteriores están fechadas al final de ese documento).

Código: `Comparison/scripts/dataset_design_curves_v1.py`.
Datos: `Comparison/results/dataset_design_curves_v1/` (figuras en `figures/`,
tablas en `final_table.csv`, `final_configs.csv`, `paired_comparisons.csv`,
`learning_curves.csv`, `pareto_points.csv`, `existing_results_160.csv`).

**Métrica primaria:** H-MAE = MAE sobre los elementos de H(Γ) por estructura,
promediado entre estructuras, en meV. Los valores absolutos de `w90` (2 átomos)
y `6x6` (72 átomos) **no son comparables entre sí**: en `6x6` la mayor parte de
los elementos de H(Γ) corresponden a pares lejanos y son casi nulos.

---

## 1. Resumen para el supervisor

1. **En `6x6` el tamaño del dataset deja de importar muy pronto.** Sobre el test
   congelado y con 3 semillas por punto, el error es plano dentro del ruido
   desde N=4: random 1D_in da 7.10 / 5.81 / 7.14 / 6.07 / 6.39 meV para
   N = 4, 8, 16, 32, 64 (sd entre semillas 0.45–1.94), y Sobol 3D da
   7.55 / 8.45 / 11.55 / 8.04 / 7.79. Cada estructura de 72 átomos aporta 72
   entornos desplazados; cuatro estructuras ya saturan el modelo. **Con la
   receta corregida de §11.9 la curva sigue plana pero a 2 meV en vez de 6**
   (§11.7–§11.8).
2. **En `w90` (celda de 2 átomos) pasa lo contrario:** N=64 es el mejor punto de
   todas las recetas y la variabilidad entre semillas es enorme (la misma receta
   y el mismo dataset dan 61, 101 y 76 meV en desarrollo). Con 2 átomos por
   estructura hacen falta muchas más estructuras para el mismo número de
   entornos.
3. **El mejor dataset diseñado supera a MD, y ahora con 10 semillas es
   concluyente**: −28.7 meV [−45.7, −14.5] sobre el test completo (§11.1). La
   ventaja está fuera del dominio de la MD; sobre los propios frames MD
   empatan. Ojo: eso vale para la receta **estable** (random 3D R=0.08); la otra
   finalista colapsa en 1 de cada 10 semillas y no supera a MD.
4. **Sobol y random son indistinguibles** a igual coste: la diferencia cambia de
   signo con N (§6).
5. **La cobertura del dominio importa mucho más que N o que la familia.** Una
   receta 1D_z (solo desplazamientos fuera del plano) da ~300 meV frente a
   ~80 meV de una 3D, con el mismo N y el mismo coste (§7).
6. **Los resultados previos del repositorio estaban sesgados por dos
   convenciones y un error de checkpoint** que ya están corregidos (§2).
7. **El error residual no viene de los datos, ni de la etiqueta, ni del modelo**
   (§11.3–§11.6): etiqueta reproducible a 0.01 meV; doblar el ancho, añadir una
   capa o subir `max_ell` no mejoran nada. Los siguientes sospechosos son la
   función de pérdida, el radio de la base y la dependencia de E_F con la
   estructura (§11.6).
8. **El ruido entre semillas es nuestro, no de la física**: un schedule cosine
   de learning rate reduce la dispersión 5× en `6x6` y fp32 la reduce 2.6× en
   `w90`, sin cambiar la media (§11.2).

---

## 2. Correcciones de método encontradas durante la campaña

Todas afectan a resultados ya publicados en el repositorio, no solo a esta
campaña.

| Hallazgo | Efecto | Estado |
|---|---|---|
| Ninguno de los 160 checkpoints S9 era el de mejor `val_loss`: todos eran la última época (mejor + 80 de paciencia) | `val_loss` 1.41× (`w90`) y 1.57× (`6x6`) peor que el mejor, mediana | Corregido: `ModelCheckpoint(monitor=val_loss, save_top_k=1)` en el constructor de configs compartido (S4), heredado por todos los scripts S9 |
| Las predicciones `h_only` de Graph2Mat llevan **S = identidad** | El `spectral_err` ≈ 4 eV de todas las campañas S9 no medía nada | Corregido: los autovalores predichos se resuelven con la S de referencia |
| La ventana espectral restaba E_F otra vez, sobre autovalores que `sisl` ya da respecto a E_F | Ventana centrada 5.7 eV por debajo de E_F | Corregido: centrada en 0; en `w90` no hay estados en Γ a ±2 eV (el cono está en K), así que ahí se usan bandas en Γ–K–M–Γ |
| Los 410 bloques MD de `Comparison/datasets/graphene_w90_*` son copias de **6 trayectorias** (la semilla del bloque no cambia las velocidades iniciales de SIESTA) | Cualquier split "por semilla" mete la misma trayectoria en train y test | La partición MD de esta campaña es por bloques temporales dentro de cada trayectoria |

Ninguna de estas correcciones afecta al H-MAE ni a ninguna regla de selección:
H-MAE compara H con H en la misma convención.

---

## 3. Fase 1–2: qué se pudo reutilizar

- **Pools N=64:** 42 por sistema, todos con las 64 etiquetas SIESTA presentes y
  con la cadena de prefijos verificada por hash (D₄ ⊂ D₈ ⊂ D₁₆ ⊂ D₃₂ ⊂ D₆₄).
  Las curvas no necesitaron **ningún** cálculo SIESTA nuevo.
- **N=16 históricos:** en `w90` solo 11 de 38 son prefijo exacto de un pool
  N=64; el resto usa otra `sampler_seed` y queda como resultado histórico, no
  como punto de curva. En `6x6` son compatibles los 26 de Sobol/random.
- **Desarrollo común `w90`:** se mantuvo el de S5 (24 estructuras LHS). Es
  idéntico para los 80 modelos, disjunto por hash de todos los pools y con
  referencia SIESTA completa.
- **Desarrollo común `6x6`:** nuevo, `common_dev_6x6_v1`, 48 estructuras
  (4 dims × 4 estratos de amplitud × 3 familias). Solo 6 necesitaron SIESTA
  nuevo (2D_in y 3D por encima de 0.08 Å no existían).
- **El ranking de `6x6` cambia al usar un desarrollo común:** ρ de Spearman
  0.51 frente al ranking anterior, en el que cada receta se medía sobre sus
  propias 8 estructuras (que además eran su conjunto de early stopping). Por
  ejemplo, `random 1D_in R0.12 d16` pasa del puesto 42 al 7.
- El H-MAE del pilot (~279–290 meV, arquitectura pequeña) predice mal el de
  producción: ρ = 0.62 entre ambos rankings en `w90`.

---

## 4. Curvas de aprendizaje (Fases 4–7)

Recetas seleccionadas por sistema (regla congelada: A = mejor del desarrollo
común, B = mejor con otra dimensionalidad, C = control de cobertura elegido solo
a partir de A y B):

| Sistema | Familia | A | B | C (control) |
|---|---|---|---|---|
| `w90` | Sobol | 3D R=0.05 | 2D_in R=0.08 | 1D_z R=0.03 |
| `w90` | random | 3D R=0.08 | 1D_in R=0.08 | 1D_z R=0.03 |
| `6x6` | Sobol | 1D_in R=0.12 | 3D R=0.08 | 1D_z R=0.03 |
| `6x6` | random | 1D_in R=0.12 | 3D R=0.05 | 1D_z R=0.03 |

**N\* confirmado = 64 en los seis finalistas de los dos sistemas.** Las reglas
de la Fase 7 (media y mediana dentro del 5%, que se mantenga al quitar
cualquier semilla, y dispersión entre semillas comparable a la diferencia) no
certifican ningún N menor, porque la dispersión entre semillas es del mismo
orden que las diferencias entre tamaños. El escalado fue 8 → 16 → 32 → 64 en
`w90` (finalista eficiente) y 4 → 8 → 16 → 32 → 64 en los dos finalistas
pequeños de `6x6`.

Esto es un resultado sobre el **procedimiento**, y conviene leerlo junto al
descriptivo: en `6x6` las medias de 3 semillas de N=4 a N=64 se solapan entre
sí (§1, punto 1), es decir, no se puede certificar N\*<64 no porque haga falta más
dato, sino porque el ruido de entrenamiento tapa cualquier diferencia. En
`w90`, en cambio, N=64 sí es sistemáticamente el mejor punto de cada receta.

**LHS queda descartada en los dos sistemas** por el criterio congelado (media de
3 semillas dentro del 5% del mejor Sobol/random N=64): en `w90` da 97.0 meV
frente a 61.1, y en `6x6` 6.97 frente a 6.35.

**N=12** se añadió en `6x6` a `sobol 3D R0.08`, la única receta que disparó la
regla pre-registrada. El disparo viene de un N=8 ruidoso (9.35 meV con una
semilla), no de una rodilla real: con 3 semillas, N=4 y N=64 ya son
equivalentes. El punto N=12 (7.59 meV) confirma que no hay rodilla.

---

## 5. Diseñado frente a MD (`w90`)

El baseline MD usa exactamente la misma arquitectura, loss, optimizador,
early stopping, política de checkpoint, semillas y test final. La única
diferencia deliberada es que cada estrategia valida con datos de su propio
proceso generador.

Comprobaciones previas: los ajustes SIESTA de los frames MD son idénticos a los
de las estructuras sintéticas salvo el bloque MD (diff de `RUN.fdf`); la
geometría del TSHS coincide con la del `RUN.fdf` del frame (≤3e-10 Å); y
reetiquetar 4 frames MD como punto simple reproduce su H con **≤0.35 meV de
diferencia máxima** (media ≤0.07 meV), tres órdenes de magnitud por debajo del
error de los modelos.

Diferencias emparejadas por estructura sobre el test final congelado (64
estructuras: 48 sintéticas + 16 frames MD), media de 3 semillas por
configuración, negativo = diseñado mejor:

| Comparación (N=64) | Δ (meV) | IC95 estructuras | IC95 semillas+estructuras |
|---|---:|---|---|
| precisión (random 1D_in R0.08) − MD | −11.9 | [−22.1, −3.1] | [−42.6, +17.1] |
| eficiente (random 3D R0.08) − MD | −21.2 | [−29.0, −14.6] | [−46.1, −3.0] |
| alternativo (Sobol 3D R0.05) − MD | −19.6 | [−23.8, −15.7] | [−44.2, −0.2] |
| … solo estructuras sintéticas (eficiente) | −27.1 | [−36.7, −18.7] | [−56.5, −5.8] |
| … solo frames MD (eficiente) | −3.3 | [−5.0, −1.7] | [−14.0, +6.1] |

Lectura honesta: sobre estructuras sintéticas el dataset diseñado gana con
claridad; sobre los propios frames MD —que son territorio del modelo MD— los
dos empatan dentro de la incertidumbre de semillas. **No se fija margen de no
inferioridad** (el plan lo prohíbe sin una tolerancia física declarada), así
que no se usa la palabra «no inferior».

**Esta tabla es la pre-registrada, con 3 semillas. §11.1 la repite con 10 y
cambia el veredicto por receta**: con 10 semillas la diferencia del finalista
eficiente se confirma (−28.7 meV, intervalo bajo cero incluso remuestreando
semillas) y la del finalista de precisión desaparece.

El motivo físico del empate en frames MD y de la ventaja global está en §7: los
frames MD de este repositorio solo cubren amplitudes ≤0.025 Å, mientras que el
test llega a 0.12 Å.

---

## 5b. Tabla final (test congelado, media de 3 semillas)

| Sistema | Dataset | Familia | N | H-MAE (meV) | IC95 estructuras | sd semillas | Rel. Frob | Bandas RMSE (meV) | CPU·h SIESTA | GPU·h/modelo |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| `w90` | precisión: random 1D_in R0.08 | random | 64 | 90.9 | [82.6, 100.9] | 20.7 | 0.041 | 101 | 0.025 | 0.090 |
| `w90` | eficiente: random 3D R0.08 | random | 64 | **81.6** | [75.5, 91.1] | 3.9 | 0.040 | 103 | 0.027 | 0.090 |
| `w90` | alternativo: Sobol 3D R0.05 | Sobol | 64 | 83.2 | [74.5, 96.4] | 7.9 | 0.040 | 131 | 0.028 | 0.111 |
| `w90` | LHS 3D R0.03 | LHS | 64 | 129.4 | [108.9, 155.2] | 21.9 | 0.056 | — | 0.025 | 0.073 |
| `w90` | **baseline MD** | MD | 64 | 102.8 | [90.8, 119.3] | 22.6 | 0.047 | 154 | 0.080 | 0.069 |
| `6x6` | precisión: Sobol 1D_in R0.12 | Sobol | 64 | **6.31** | [6.16, 6.46] | 1.7 | 0.026 | 40 | 1.09 | 0.217 |
| `6x6` | alternativo: random 1D_in R0.12 | random | 64 | 6.39 | [6.23, 6.56] | 1.4 | 0.027 | 44 | 1.07 | 0.144 |
| `6x6` | LHS 2D_in R0.08 | LHS | 64 | 7.29 | [6.88, 7.76] | 0.6 | 0.031 | 68 | 0.91 | 0.177 |
| `6x6` | eficiente: Sobol 3D R0.08 | Sobol | 64 | 7.79 | [7.23, 8.43] | 1.1 | 0.033 | 70 | 1.02 | 0.123 |

Fuente: `final_table.csv` y `final_configs.csv`. En `6x6` no hay columna «vs MD»
porque no existen frames MD de esa supercelda (§9.3).

## 5c. Frontera coste–precisión

Puntos no dominados en (CPU·h SIESTA de las etiquetas de entrenamiento, H-MAE
del test final), `pareto_points.csv` y `fig2_pareto.png`:

- `w90`: random 3D R0.08 N=4 (0.002 CPU·h, 113 meV) → Sobol 2D_in N=8
  (0.003 CPU·h, 93 meV) → random 3D R0.08 N=64 (0.027 CPU·h, **81.6 meV**).
  El baseline MD queda **fuera** de la frontera: cuesta 3× más SIESTA
  (0.080 CPU·h, porque hay que correr la trayectoria hasta cada frame) y da
  102.8 meV.
- `6x6`: random 1D_in R0.12 N=4 (0.065 CPU·h, 7.1 meV) → N=8 (0.135 CPU·h,
  **5.8 meV**). Con 0.14 CPU·h de SIESTA se llega al mismo sitio que con 1.1.

En `w90` el coste de generar etiquetas es despreciable (1.4 s/estructura) y lo
que domina es el tiempo de GPU; en `6x6` (49 s/estructura) sí manda SIESTA, y
ahí la conclusión de que N=4–8 basta es económicamente importante: **8×
menos coste de etiquetado** para el mismo error.

---

## 6. Sobol frente a random

Diferencia emparejada Sobol − random (media de las recetas de cada familia,
semilla 0, test final; negativo = Sobol mejor):

| N | `w90` (3 recetas) | `w90` (solo A+B) | `6x6` (3 recetas) | `6x6` (solo A+B) |
|---:|---:|---:|---:|---:|
| 4 | −10.2 | −18.2 | +1.8 | +1.7 |
| 8 | +2.9 | +6.6 | +1.2 | +1.5 |
| 16 | −15.9 | −4.3 | +0.8 | +1.4 |
| 32 | +23.6 | +32.9 | −0.5 | −0.7 |
| 64 | +11.4 | +11.9 | −0.9 | −1.1 |

**No hay ganador consistente.** El signo cambia con N en los dos sistemas. En
`w90` la magnitud del cambio (hasta 33 meV) es del mismo orden que la
variabilidad entre semillas de una sola receta (sd hasta 21 meV), así que estas
diferencias no son atribuibles a la familia. En `6x6` la tendencia es más
regular (random mejor a N pequeño, Sobol mejor a N grande) pero el efecto es de
~1 meV sobre ~6, y estos puntos son de una sola semilla.

Respuesta operativa: **elegir por coste y por cobertura del dominio, no por
familia.**

---

## 7. Generalización: el dominio manda

Heatmap `fig4_generalization_heatmap.png` (H-MAE frente a la amplitud real del
test, modelos N=64):

- El modelo MD, entrenado con amplitudes ≤0.025 Å, pasa de 70 meV a A=0.02 Å a
  **194 meV a A=0.12 Å**.
- Las recetas 3D con R = 0.05–0.08 Å se mantienen entre 64 y 84 meV hasta
  A=0.10 Å y suben a ~100–152 meV en A=0.12 Å.
- Las recetas 1D_z (solo z) están entre 115 y 549 meV: nunca ven
  desplazamientos en el plano, y eso no se arregla con más N.

En `6x6` el patrón es el mismo y aún más limpio:

- Las recetas con R=0.12 Å (1D_in) van de 5.8 meV en A=0.02 Å a 7.3 meV en
  A=0.12 Å: **planas en todo el rango del test**.
- Las de R=0.05–0.08 Å empiezan igual de bien (6.1–7.1 meV) pero suben a ~11 meV
  en A=0.12 Å, fuera de su dominio.
- Las 1D_z van de 11 a 39 meV.

Es decir: **cubrir el dominio de desplazamientos que se va a usar es mucho más
importante que el tamaño del dataset o la familia de muestreo.** La receta
ganadora en los dos sistemas es la que entrena en el dominio más ancho.

---

## 8. Coste

- SIESTA nuevo en toda la campaña: 7 (desarrollo `6x6`; 6 usados y 1 descartado
  al corregir la regla de estrato de LHS) + 48 (test `w90`) + 48 (test `6x6`) +
  4 (control de etiqueta MD) = **107 cálculos**, por debajo del presupuesto del
  plan (128). Las curvas no costaron **ningún** SIESTA.
- Entrenamientos nuevos: **106** (`w90`: 30 curvas + 3 LHS + 12 confirmación +
  3 MD; `6x6`: 30 curvas + 3 LHS + 24 confirmación + 1 N=12), frente a los
  96–102 previstos. La diferencia son los escalones de N\* de la Fase 7.
- Coste SIESTA por punto (reproducible): en `w90`, 1.4 s por estructura →
  0.025 CPU·h para N=64; el baseline MD cuesta 0.080 CPU·h porque cada frame
  exige correr la trayectoria hasta él. En `6x6`, ~49 s por estructura.
- GPU: 11.3 h de pared en total, con 1–4 entrenamientos compartiendo una
  RTX 5090.

---

## 8b. Relación H-MAE ↔ bandas (entrada para fijar un margen)

Sobre las evaluaciones de desarrollo de todos los modelos (1080 pares
estructura-modelo en `w90`, 2784 en `6x6`):

| | `w90` | `6x6` |
|---|---:|---:|
| ρ Spearman H-MAE ↔ RMSE de bandas | 0.65 | 0.65 |
| ρ Spearman H-MAE ↔ L1 relativo de DOS | 0.60 | 0.58 |
| RMSE de bandas por meV de H-MAE (mediana) | 0.64 | 6.3 |
| Cuartiles del RMSE de bandas (meV) | 36 / 61 / 112 | 25 / 42 / 75 |

La correlación es clara pero no ajustada (ρ≈0.65): **el H-MAE es un proxy
imperfecto del error de bandas**, así que un margen de no inferioridad debería
fijarse sobre la magnitud física que interese (bandas o DOS), no sobre H-MAE.
Con estos números, un objetivo de, por ejemplo, 50 meV de RMSE de bandas cerca
de E_F ya está al alcance de la mitad de los modelos en `6x6` y de una minoría
en `w90`.

---

## 9. Qué haría falta para cerrar las preguntas que quedan abiertas

1. **Más semillas, no más datos.** En los dos sistemas la incertidumbre
   dominante es el entrenamiento, no el muestreo. Con 5–10 semillas por
   configuración, N\* bajaría casi seguro de 64 en `6x6`.
2. **Margen físico.** Para poder decir «equivalente a MD» hace falta declarar
   antes qué degradación de bandas/DOS es aceptable. En `final_configs.csv`
   están los errores de bandas (Γ–K–M–Γ, ±2 eV de E_F) y DOS de cada
   configuración reportada para poder fijarlo.
3. **MD con trayectorias realmente independientes.** Las actuales son 6
   trayectorias deterministas; para un baseline MD fuerte habría que variar la
   semilla de velocidades de SIESTA, no la del manifiesto.
4. **Amplitudes MD comparables.** Un baseline MD a 0.05–0.12 Å exigiría
   temperaturas mucho más altas o desplazamientos impuestos; tal como está, la
   comparación diseñado-vs-MD es también una comparación de dominios.

---

## 10. Respuestas a las siete preguntas del plan

1. **¿Cómo disminuye el error al aumentar N?** En `w90` baja de ~115 a ~82 meV
   entre N=4 y N=64, de forma no monótona porque el ruido de entrenamiento es
   comparable al efecto. En `6x6` **no baja**: es plano dentro del ruido desde
   N=4.
2. **¿Menor dataset próximo a N=64?** Por las reglas congeladas, ninguno: N\*=64
   en los seis finalistas. Descriptivamente, en `6x6` N=4–8 ya iguala a N=64
   (frontera de Pareto en §5c) y en `w90` no.
3. **¿Sobol o random?** Indistinguibles (§6).
4. **¿Se mantiene con distintas semillas?** En `w90` no: sd de hasta 21 meV
   sobre ~90, con casos de 61 vs 101 meV. En `6x6` la dispersión relativa es
   menor (sd 0.6–1.7 sobre ~6–8 meV) pero sigue tapando las diferencias entre N.
5. **¿El dataset diseñado se acerca a MD?** Sobre el mismo test, lo supera por
   12–21 meV; contando la incertidumbre de semillas, dos de los tres finalistas
   mantienen el intervalo por debajo de cero y el tercero no. Sobre los propios
   frames MD, empate. No se afirma equivalencia (§5).
6. **¿Cuánto cuesta cada alternativa?** §8 y §5c. Lo relevante: en `6x6` el
   etiquetado SIESTA domina y se puede dividir por 8; en `w90` domina la GPU.
7. **¿Qué configuraciones forman la frontera de Pareto?** §5c y
   `fig2_pareto.png`.

## 11. Seguimiento (2026-09-20): semillas, receta, suelo y capacidad

Cuatro experimentos posteriores al test congelado. Ninguno cambia una regla de
selección; los tres primeros aumentan la precisión de lo ya reportado y el
cuarto diagnostica el techo.

### 11.1 Diez semillas: quién gana a MD y quién no

Con 10 semillas por configuración (en lugar de 3), sobre el mismo test:

| Comparación (N=64) | Δ vs MD | IC estructuras | IC semillas+estructuras |
|---|---:|---|---|
| eficiente (random 3D R0.08), test completo | **−28.7** | [−38.3, −20.2] | **[−45.7, −14.5]** |
| … solo sintéticas (0.02–0.12 Å) | −36.1 | [−48.1, −25.3] | [−56.7, −19.5] |
| … solo frames MD | −6.3 | [−8.7, −4.3] | [−14.7, +1.7] |
| precisión (random 1D_in R0.08), test completo | +3.8 | [−7.9, +14.5] | [−26.5, +42.4] |

Dos conclusiones que con 3 semillas no se podían sacar:

1. La receta **estable** supera a MD con el intervalo por debajo de cero incluso
   remuestreando semillas. La ventaja está **fuera del dominio de la MD**: en los
   propios frames MD empatan.
2. La receta llamada «de precisión» **no supera a MD**. Sus 10 semillas en
   desarrollo son 61, 101, 76, **233**, 78, 97, 100, 81, 76, 84 meV: una de cada
   diez colapsa. Su ventaja anterior era un artefacto de haber sorteado 3
   semillas buenas.

Consecuencia práctica: una receta debe reportarse **con su dispersión entre
semillas**, no con su mejor ejecución. El orden de mérito cambia al hacerlo.

### 11.2 El ruido entre semillas es de la receta de entrenamiento, y se arregla

Ablación sobre el finalista de precisión, N=64, 3 semillas por variante
(media ± sd en desarrollo):

| Variante | `w90` | `6x6` |
|---|---|---|
| baseline (la congelada) | 79.4 ± 20.4 | 6.22 ± 1.67 |
| máx. épocas 1200 | 89.8 ± 17.8 | 6.35 ± 1.27 |
| cosine LR | 87.1 ± 23.0 | 6.42 ± **0.31** |
| cosine + 1200 épocas | 82.5 ± 10.2 | 6.13 ± 0.73 |
| fp32 (en `6x6` no cabe en 32 GB) | 82.1 ± **7.9** | — |

La media no mejora; **la dispersión cae 2.6× (`w90`, fp32) y 5× (`6x6`,
cosine)**, y ninguna de las 12 ejecuciones con variante colapsó. Para cualquier
campaña futura: schedule cosine siempre, y fp32 donde quepa. Es reproducibilidad
gratis, y es justo lo que impedía concluir en §11.1.

`StochasticWeightAveraging` no es utilizable: Lightning copia el modelo y el
módulo de graph2mat no es serializable («cannot pickle 'module' object»).

### 11.3 El suelo no es la etiqueta

Reetiquetando 8 estructuras por sistema con `DM.Tolerance` 1e-6 en vez de 1e-4:

| Sistema | MAE de etiqueta (mediana) | máximo |
|---|---:|---:|
| `w90` | 0.011 meV | 0.39 meV |
| `6x6` | 0.0024 meV | 0.66 meV |

La etiqueta SIESTA es reproducible a ~0.01 meV, **tres o cuatro órdenes de
magnitud por debajo del error de los modelos** (6 y 80 meV). La convergencia
SCF no limita nada.

### 11.4 El techo tampoco es el ancho del modelo

Doblando `hidden_irreps` (48→96) a datos fijos:

| | base | 2× ancho |
|---|---:|---:|
| `w90` N=8 | 97.2 | 97.1 |
| `w90` N=64 | 81.3 | 86.7 |
| `6x6` N=64 | 7.09 | 7.49 |

Ninguna mejora (el `6x6` ancho llegó a 30.8 GB de VRAM).

### 11.5 Tampoco es la profundidad ni la resolución angular

Campo receptivo y resolución angular, a datos fijos y con schedule cosine (el
radio de corte lo fijan los radios del `.ion.xml`, no es un parámetro, así que
el campo receptivo solo crece por capas). Control y variante con el mismo batch
y el mismo schedule:

| | control | profundidad 4 | Δ (IC95 Welch) |
|---|---|---|---|
| `w90` N=64 | 79.5 ± 15.6 (n=7) | 77.7 ± 5.8 (n=7) | −1.9 [−14.2, +10.4] |
| `6x6` N=64 | 5.55 ± 2.06 (n=5) | 6.24 ± 0.65 (n=5) | +0.7 [−1.2, +2.6] |
| `6x6` N=4 | 7.22 ± 1.18 (n=5) | 7.57 ± 0.86 (n=5) | +0.4 [−0.9, +1.6] |

`num_interactions=5` rompe la optimización (`w90`: 152 ± 115 meV, una ejecución
en 285) y `max_ell=4` no aporta (`w90` 80.2 ± 14.0; `6x6` 6.97 ± 0.21).

**Ninguna intervención de arquitectura mejora la precisión.** Con 2–3 semillas
la profundidad 4 parecía dar −15 %; con 5–7 semillas el efecto desaparece y los
controles bajan (6.16→5.55 en `6x6`, 87.1→79.5 en `w90`): era regresión a la
media. Lo que sí es consistente en los tres casos es que la cuarta capa
**reduce la dispersión entre semillas** (15.6→5.8, 2.06→0.65, 1.18→0.86), igual
que el schedule cosine: mejora la reproducibilidad, no el error.

### 11.6 Dónde queda el techo

Resumen del diagnóstico, todo medido:

| Candidato | Veredicto | Evidencia |
|---|---|---|
| Cantidad de datos | descartado en `6x6` | curva plana N=4…64 con 3 semillas (§1) |
| Calidad de la etiqueta | descartado | 0.01 meV, 3–4 órdenes por debajo (§11.3) |
| Ancho del modelo | descartado | 2× ancho, sin mejora (§11.4) |
| Profundidad / `max_ell` | descartado | §11.5 |
| Receta de entrenamiento | afecta a la **varianza**, no a la media | §11.2, §11.5 |

Lo que queda sin probar, por orden de plausibilidad: (a) la **función de
pérdida** (`block_type_mae` pesa todos los bloques por igual, y en `6x6` la
mayoría son pares lejanos casi nulos); (b) el **radio de la base** (`.ion.xml`),
que fija el grafo y no se puede ampliar sin regenerar las etiquetas; (c) que el
objetivo aprendido sea `H − E_F·S`, donde **E_F depende de la estructura**, de
modo que el modelo tiene que predecir una magnitud global a partir de entornos
locales. (c) es contrastable a bajo coste: reentrenar sobre `H` sin desplazar
y comparar.

Dos efectos "prometedores" se evaporaron al añadir semillas en esta campaña (la
ventaja del finalista de precisión sobre MD, §11.1, y la profundidad 4). **Con
esta arquitectura y este ruido, 3 semillas no bastan para afirmar nada**: es el
resultado metodológico más transferible de todo el trabajo.

### 11.7 Sí era la función de pérdida: 3× menos error en `6x6`

`block_type_mae` promedia por tipo de bloque, así que en una supercelda los
miles de bloques de pares lejanos casi nulos pesan igual que los enlazantes.
Cambiando a `elementwise_mse` (pesa por elemento) en el finalista de precisión
de `6x6`, N=64:

| Pérdida | H-MAE (meV) | mejor época |
|---|---|---|
| `block_type_mae`, 600 ép | 5.55 ± 2.06 (n=5) | 281 |
| `block_type_mae`, 2000 ép | 5.29 ± 0.80 (n=5) | 241 |
| **`elementwise_mse`, 2000 ép** | **2.00 ± 0.04 (n=5)** | 1211 |

**Un factor 2.6 de mejora y la dispersión entre semillas dividida por 20.** Es
la única intervención de toda la campaña que mueve el techo, después de
descartar datos, etiqueta, ancho, profundidad y resolución angular.

El primer intento con 600 épocas no lo vio: las pérdidas alternativas seguían
mejorando en la época 596 de 600, mientras el control convergía en la 281. Con
la receta congelada, la pérdida buena estaba estrangulada por el tope de épocas.

En `w90` el efecto es menor y no concluyente (68.4 ± 21.2 frente a 79.5 ± 15.6,
n=5/7), aunque una semilla llegó a 30.9 meV, menos de la mitad del mejor
resultado de toda la campaña.

### 11.8 La curva sigue plana, pero hay que medirla a pasos iguales

Con la pérdida nueva, a **épocas** iguales la curva de `6x6` parecía tener
pendiente (4.88 → 4.54 → 4.55 → 2.55 → 2.00 para N=4…64). No la tiene: con
batch 16, una época son ⌈N/16⌉ pasos, así que N=64 recibía **4× más
actualizaciones** que N=16, y la caída coincidía exactamente con el punto donde
sube el número de pasos.

Repitiendo con ~8000 pasos de gradiente y 2000 validaciones para todos los
tamaños (`check_val_every_n_epoch` escalado), de modo que solo cambien los datos:

| N | H-MAE (meV), 3 semillas | mejor época |
|---|---|---|
| 4 | 2.13 ± 0.28 | 4214 |
| 8 | 1.94 ± 0.09 | 5344 |
| 16 | 2.14 ± 0.26 | 4202 |
| 32 | 2.04 ± 0.12 | 2560 |
| 64 | 2.00 ± 0.04 | 1211 |

**Plana dentro del ruido: 1.94–2.14 meV de N=4 a N=64.** La conclusión original
(§1) se confirma y se refuerza: cuatro estructuras de 72 átomos bastan. Lo que
cambia es el nivel, de ~6 a ~2 meV, y el motivo por el que antes parecía haber
pendiente.

**Aviso metodológico:** un presupuesto fijo en épocas confunde N con pasos de
gradiente. Las curvas de §4 (y cualquier curva medida así) tienen ese sesgo —
ahí es un factor 2 porque el batch era 32, no 4, pero conviene rehacerlas a
pasos iguales antes de sacar conclusiones finas sobre la forma de la curva.

### 11.9 Receta recomendada para futuras campañas

De todo lo medido en §11:

- pérdida `elementwise_mse` (no `block_type_mae`) — factor 2.6 en `6x6`;
- schedule cosine de learning rate — sd entre semillas ÷5;
- presupuesto en **pasos de gradiente**, no en épocas (~8000), con
  `check_val_every_n_epoch` escalado;
- fp32 donde quepa en memoria — sd ÷2.6 en `w90`;
- **≥5 semillas** antes de afirmar cualquier diferencia;
- N=4–8 estructuras de supercelda son suficientes: el coste de etiquetado baja
  16× sin perder precisión.

---

### 11.10 Seguimiento w90: presupuesto de optimización y validación espectral

La receta `elementwise_mse` sin parada temprana se ha comparado a 2000 y
4000 épocas, cinco semillas por presupuesto y cosine adaptado al horizonte.
Sobre el test común existente, H-MAE pasa de **50.74 ± 18.35 a 34.36 ± 3.98
meV**, bandas de **58.13 ± 4.49 a 49.73 ± 10.23 meV**, y DOS L1 relativo de
**10.80 ± 0.58 a 9.07 ± 1.51%**. ± es sd entre semillas. Las cinco mejoran
las tres métricas globales; el tiempo medio pasa de 18.3 a 35.0 min/modelo.

La mejora espectral es menor que la de H y la dispersión de bandas/DOS
aumenta. En frames MD las bandas solo mejoran 6% en media (3/5 semillas),
y a 0.12 Å todavía dan 157.88 meV. Es un seguimiento exploratorio sobre un
test ya consultado, no una nueva confirmación independiente ni una comparación
justa con el baseline MD entrenado con otra receta. Datos, protocolo, resultados
por dominio y figura en [el informe de seguimiento](dataset_design_w90_early_stopping.md).

### 11.11 Cobertura w90 con receta y presupuesto comunes

Con `elementwise_mse`, 4000 épocas completas, desarrollo común y cinco semillas,
el dataset **3D R=0.12** es el mejor punto probado: H-MAE 18.67 ± 3.90 meV,
bandas 34.08 ± 8.95 meV y DOS L1 5.10 ± 1.40%, frente a 49.68 ± 11.30,
133.97 ± 27.62 y 12.39 ± 2.72% para MD. Frente a 3D R=0.08, ampliar el dominio
mejora H y DOS con IC95 bajo cero; la mejora media de bandas no queda resuelta.

El resultado modifica la prioridad práctica: una vez estabilizado el
entrenamiento, **cobertura tridimensional y amplitud** dominan la construcción
del dataset `w90`. El pool R=0.12 costó 0.023 CPU·h de SIESTA, frente a 0.080
del pool MD. Protocolo, cautelas, comparaciones emparejadas y tablas en
[dataset_design_w90_coverage.md](dataset_design_w90_coverage.md). Se inicia la
curva N=4…64 de esta receta a igual número de actualizaciones y cinco semillas.

## 12. Figuras

| Figura | Qué responde |
|---|---|
| `fig1_learning_curves.png` | Cuántas estructuras necesita cada estrategia y dónde deja de compensar |
| `fig2_pareto.png` | Qué dataset da la mejor precisión por presupuesto (dos fronteras: reproducible e incremental) |
| `fig3_designed_vs_md.png` | Diseñado − MD por estructura, con IC de estructuras y de semillas+estructuras |
| `fig4_generalization_heatmap.png` | Hasta qué amplitud generaliza cada dominio de entrenamiento |
| `fig5_seed_stability.png` | Si las mejoras superan la variabilidad del entrenamiento |
