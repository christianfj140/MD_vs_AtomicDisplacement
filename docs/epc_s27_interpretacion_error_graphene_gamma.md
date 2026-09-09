# E-F_001-S27 — interpretación del error de graphene Γ

S25 midió `g` por tres caminos y S26 adjudicó GO-4 como `NO_GO`. Queda una pregunta abierta sobre el
checkpoint, y es la que este paso responde: **¿es adecuado para derivadas EPC y para `g`, o el residuo
justifica abrir retraining?**

Este paso no calcula física nueva. Todo sale de artifacts ya en disco —las derivadas direccionales en
espacio real de los tres paths, los eigenspaces persistidos, el protocolo congelado y el veredicto
GO-4— releídos como una descomposición del residuo por orbital, bloque, dirección y subespacio, más la
sensibilidad de esa descomposición a δ, a la base/topología, a las tolerancias y a dónde se cortó la
ventana.

Productor: [`Comparison/scripts/interpret_gamma_epc_error.py`](../Comparison/scripts/interpret_gamma_epc_error.py).
Tests: [`tests/test_interpret_gamma_epc_error.py`](../tests/test_interpret_gamma_epc_error.py).
Artifacts: `Comparison/results/epc/error_interpretation/graphene/gamma_error_interpretation.json` y
`gamma_error_interpretation_groups.csv` (720 filas de distribución).

Convenciones: **[CODE]** verificado en este repositorio, **[MEAS]** medido en este paso, **[READ]**
leído de un artifact anterior y no re-derivado aquí.

---

## 1. Veredicto

| pregunta | respuesta |
| --- | --- |
| ¿adecuado para la validación de derivadas a tres paths? | sí, como **cota medida** — es lo que ya sostiene el escalón 5 |
| ¿adecuado para un `g` cuantitativo? | **no** |
| ¿el residuo es atribuible al checkpoint? | `UNDECIDED`: faltan gauge, convergencia de referencia y cota de `Delta_out` |
| ¿se abre el ticket de retraining? | **no**: la escalera congelada lo autoriza en su escalón 2, y el veredicto seleccionó el 3 |

La adecuación no se re-adjudica: `quantitative_for_g = False` se **lee** de
`go4_verdict.model_statement`. Lo que este paso añade es *por qué* el residuo es lo que es, que la
respuesta no cambia con ninguna de las elecciones que podrían haberla cambiado, y qué evidencia
concreta activaría el ticket. **[READ]**

Que el ticket siga cerrado no es una duda sobre la medida: es el suelo del roadmap («si 1–5 fallan, no
se culpa al ML») aplicado a una situación en la que **las capas que fallan no pueden producir este
residuo** (§5).

## 2. Reglas de este paso

1. **El gate no se edita retroactivamente.** `tau_model`, el umbral de adecuación y la escalera de
   claims se leen. Las ventanas alternativas y el factor de escala de mejor ajuste van etiquetados como
   diagnósticos y nunca se presentan como el rendimiento del modelo ni como factores de corrección.
   **[CODE]**
2. **Un `NO_GO` es una afirmación sobre los proveedores de este repositorio, no sobre graphene.**
3. **Se comprueba que se está interpretando lo mismo que se midió.** Los nueve `r` que la
   descomposición recalcula reproducen los del artifact congelado a `1e-9` relativo; el test lo exige
   entrada por entrada. Si la interpretación diera otro número, sería una segunda medida, no una
   lectura. **[MEAS]**

## 3. Qué es el residuo

### 3.1 Nivel derivada — una ganancia global, no un patrón equivocado

Contra la referencia SIESTA certificada, en el mismo espacio `(fila, columna, R)`:

| dirección | ‖D_ref‖ (eV/Å) | rel | α de mejor ajuste | coseno | rel tras quitar α |
| --- | --- | --- | --- | --- | --- |
| `atom0000_x` | 38.72 | 0.671 | 0.617 | 0.945 | 0.326 |
| `e2g_bond_longitudinal` | 54.76 | 0.671 | 0.617 | 0.945 | 0.326 |
| `e2g_bond_transverse` | 54.76 | 0.670 | 0.617 | 0.945 | 0.326 |
| `random_seed0` | 18.76 | 0.601 | 0.657 | 0.937 | 0.348 |

**[MEAS]** El modelo **sobre-responde ×1.62** y apunta casi en la dirección correcta (coseno 0.94).
Quitar esa única constante elimina el 76 % del residuo en cuadratura. La constante está *ajustada
contra la referencia*: describe la estructura del error, no lo corrige.

### 3.2 Las tres direcciones en el plano dan el mismo número

`atom0000_x` y los dos miembros del E2g coinciden en `rel` y en `α` **a cuatro decimales**. No es
casualidad: en una celda de dos átomos, todo desplazamiento relativo en el plano vive en el mismo plano
E2g, y la parte acústica no responde. Consecuencias, las dos incómodas y las dos medidas:

- el modo físico **no** es un caso más difícil que la coordenada que C14 ya había medido;
- la transferencia «held-out» de C14 (`random_seed0`, con componentes fuera del plano, `rel` 0.601)
  sigue siendo una comprobación válida, pero *dentro del plano* no era una prueba independiente.
  **[MEAS]**

### 3.3 Dónde vive: el enlace, no el átomo

`e2g_bond_longitudinal`, δ = 0.02 Å (el `..._groups.csv` trae las 720 filas, con las cuatro agrupaciones para cada dirección, path y δ):

| agrupación | grupo | n | ‖D_ref‖ | rel | cuota del residuo |
| --- | --- | --- | --- | --- | --- |
| `pair_distance_ang` | 1.47 (1er vecino) | 96 | 45.11 | **0.798** | **0.961** |
| `pair_distance_ang` | 2.93 | 96 | 8.45 | 0.730 | 0.028 |
| `pair_distance_ang` | 0.00 (onsite) | 32 | 29.03 | 0.074 | 0.003 |
| `atom_pair` | `inter_atomic` | 1152 | 46.43 | 0.790 | **0.997** |
| `atom_pair` | `intra_atomic` | 32 | 29.03 | 0.074 | 0.003 |
| `orbital_block` | s-s | 74 | 24.86 | 0.783 | 0.281 |
| `orbital_block` | p-p | 666 | 31.93 | 0.616 | 0.287 |

**[MEAS]** Mismo patrón que C14 midió para la coordenada, ahora sobre el modo físico: el 96 % del
residuo está en el primer vecino, ningún bloque orbital es el culpable aislado (0.62–0.78, plano), y el
bloque **intra-atómico** —justo donde vive el término que C14C no resuelve— es el que el modelo acierta
(rel 0.074, 0.3 % del residuo).

### 3.4 Nivel subespacio — el `r` grande es un estado σ*, no el modo

`δg = C_f†(Δ_G2M − Δ_SIESTA)C_i`, path JVP, ventana congelada 2+2 y su sub-bloque π (los dos estados
adyacentes a la neutralidad, con el recuento de ocupados tomado del protocolo):

| dirección | k | ‖g_ref‖ | `r` ventana | `r` manifold π | cuota diagonal | estado dominante |
| --- | --- | --- | --- | --- | --- | --- |
| `atom0000_x` | Γ | 44.83 | 0.327 | 0.346 | 1.00 | s2, +8.69 eV |
| `atom0000_x` | M | 13.02 | 1.895 | 0.615 | 1.00 | s3, +8.14 eV |
| `random_seed0` | `k_generic_2` | 13.87 | 0.280 | 0.610 | 0.80 | s3, +10.94 eV |
| `e2g_bond_longitudinal` | `k_generic_1` | 22.92 | **2.411** | 0.442 | 0.99 | s3, +10.00 eV |
| `e2g_bond_longitudinal` | K | 24.30 | 1.568 | 0.769 | 0.89 | s3, +14.36 eV |
| `e2g_bond_transverse` | `k_generic_1` | 12.22 | 0.597 | 0.594 | 0.59 | s0, −5.99 eV |
| `e2g_bond_transverse` | K | 37.49 | 1.722 | 0.769 | 0.94 | s3, +14.36 eV |

**[MEAS]** El `r = 2.41` que fija el caso peor del gate es, en el 99 %, **un solo elemento diagonal**:
el estado alto de la ventana (σ*, +10 eV), cuyo acoplamiento de referencia es pequeño y cuyo residuo es
grande. En el manifold π —los estados que hacen la física EPC de graphene— el error relativo cae al
rango 0.34–0.77, coherente con el 0.67 de nivel derivada y sin la dispersión.

En `K`, el bloque del par de Dirac es el acoplamiento E2g de la anomalía de Kohn: ‖g‖ = **13.53 eV/Å**
por el path SIESTA, y el modelo lo sobre-estima en un 77 %, idéntico para los dos miembros del doblete
(0.7689 vs 0.7688) — otra vez la simetría, no un backend.

### 3.5 Una constante explica casi todo donde importa

Aplicando la ganancia de nivel derivada (α = 0.617) **sólo a `D_H` del modelo** y volviendo a contraer:

| caso | `r` manifold π | tras α |
| --- | --- | --- |
| `atom0000_x` × Γ | 0.346 | 0.077 |
| `atom0000_x` × M | 0.615 | 0.053 |
| `e2g_bond_longitudinal` × K | 0.769 | **0.092** |
| `e2g_bond_transverse` × K | 0.769 | **0.092** |
| `e2g_bond_longitudinal` × `k_generic_1` | 0.442 | 0.145 |
| `random_seed0` × `k_generic_2` | 0.610 | 0.192 |

**[MEAS]** Un escalar ajustado a nivel de matrices se lleva la mayor parte del error de `g` en el
manifold π. Esto **no** es un checkpoint corregido —α está ajustado contra la referencia que se quiere
predecir— pero sí es la caracterización del defecto: *calibración de ganancia*, no falta de estructura.
Es exactamente la clase de defecto que un fine-tuning puede atacar, y por eso es la evidencia causal
del §6.

## 4. Sensibilidad: nada de lo elegible cambia el veredicto

| eje | qué se varió | resultado |
| --- | --- | --- |
| **δ** | los tres δ del barrido congelado (0.005, 0.01, 0.02 Å) | `rel` se mueve ≤ 9e-4 y α ≤ 1e-3; lo medido es el modelo, no la diferencia finita **[MEAS]** |
| **base / topología** | tabla del checkpoint (`PointBasis.R` 2.4865 Å, cutoff 4.973 Å) frente a la que sisl reconstruye hoy (2.5763 Å) | el modelo se evalúa sobre la topología con la que se ajustaron sus pesos; lo que la tabla deja fuera pesa **0.0085 eV/Å** sobre 54.8, y ningún desplazamiento cruza un cutoff (margen 0.071 Å) **[MEAS]** |
| **tolerancias** | dónde se calibró `tau_model` | calibrar en Γ en vez de en M daría 0.491 en vez de 2.843: la anchura de la cota depende **5.8×** de esa elección. La regla congelada toma el máximo, que es el extremo conservador. Se reporta; no se enmienda **[MEAS]** |
| **selección de subespacio** | ventana congelada, manifold π, bloque ocupado, bloque vacío, y los 5 `k` disponibles | `r` recorre 0.28 – 5.40 según dónde se corte. El **menor `r` medido en cualquier parte** es 0.280, todavía **5.6×** el umbral de adecuación de 0.05 **[MEAS]** |

La lectura conjunta: el **número** no es robusto (por eso no debe citarse «el error de `g` es del X %»),
pero el **veredicto** sí lo es. No existe elección de δ, de base, de tolerancia ni de ventana, dentro de
lo que estos artifacts permiten, que haga cuantitativo a este checkpoint.

## 5. El residuo no es la respuesta de base, y ninguna capa abierta puede fabricarlo

Los tres paths consumen bloques `S_L`/`S_R` **byte-idénticos** por dirección (mismo
`basis_response_block_sha256`). Por tanto

```math
\Delta^{\rm phys}_{\rm G2M} - \Delta^{\rm phys}_{\rm SIESTA}
= \sum_l e^{2\pi i\,k\cdot R_l}\left[D_H^{\rm G2M}(R_l) - D_H^{\rm SIESTA}(R_l)\right],
```

comprobado a **4.2e-13** relativo en los 18 casos. **[MEAS]** La respuesta de base entra en los dos
lados con el mismo signo y **se cancela exactamente en el residuo**. Resolver el término intra-atómico
(C14C) cambia la anchura de la cota sobre `g` y el denominador del error relativo; no cambia el residuo.

¿Podría el término abierto rescatar la adecuación por el denominador? Se mide: para que `r` bajara al
umbral de 0.05 haría falta agrandar el bloque de referencia hasta 293–1 292 eV/Å, lo que con la
sensibilidad intra-atómica medida en S25 (19.8 eV/Å en Γ, 30.6 en K por unidad de Frobenius de `A`)
exige un `A` de **6.2× a 14.4× la norma de toda la dS/dR** — de la que `A` es la parte monocéntrica.
**[MEAS]** No.

Los gates 8 y 9 (traslación uniforme, simetría del doblete en K) fallan **idénticamente en los tres
paths**, que es precisamente lo que impide cargarlos a un backend. Tampoco pueden producir una
diferencia entre paths.

Conclusión: las tres capas que bloquean GO-4 —`basis_response` y dos reglas congeladas de
`formalism`— son **incapaces de generar este residuo**. Cerrarlas cambiará el veredicto del gate y la
anchura de la cota, no la medida. El ticket sigue cerrado por la regla, no porque la evidencia sea
dudosa.

## 6. El ticket de retraining: cerrado, con su disparador escrito

`authorized = False`, `status = closed`. La autorización no la da este script: la da el escalón 2 de la
escalera congelada —«numerical and formalism checks pass, tau_model > adequacy; opens the fine-tuning
ticket, which stays closed otherwise»—, y el veredicto seleccionó el escalón 3. El script exige que la
escalera y `fine_tuning_ticket_authorized` coincidan, y se detiene si no. **[CODE]**

### Evidencia que lo activaría

| condición | artifact | se vuelve cierta cuando |
| --- | --- | --- |
| gate 5: proveedor de respuesta de base **validado** | `basis_response/graphene/basis_response_certification.json` | el término intra-atómico se resuelve analíticamente (`⟨d_α φ_μ|φ_ν⟩` monocéntrico sobre los radiales del `.ion.xml`) y C14C pasa de `NO_GO` a `PASS` |
| gates 8 y 9: las dos reglas congeladas, resueltas | `preregistration/graphene/epc_graphene_gamma_protocol.json` | o se satisfacen tal como están escritas, o se enmiendan **en un pre-registro nuevo** y se re-congelan antes de repetir el experimento. Enmendar el fichero actual a posteriori no es una opción |
| el residuo sigue por encima del umbral | este informe, `conclusion.adequacy` | se repite la medida tras lo anterior y `tau_model > adequacy` sigue siendo cierto. Hoy lo es: 2.84 frente a 0.05 |

### Hipótesis causales, ordenadas por el soporte medido

1. **Sobre-respuesta global de la `dH/dR` aprendida** — α = 0.617, coseno 0.94, plano en δ; en el
   manifold π el error cae a 0.05–0.21 al quitar ese escalar. *Se falsaría* con un checkpoint afinado
   cuya ganancia sea 1.0 y cuyo error en el manifold π no baje.
2. **Bloques de primer vecino poco restringidos por los datos** — 96 % del residuo en la capa de 1.47 Å,
   0.3 % sobre el átomo. *Se falsaría* si el residuo siguiera en la capa de enlace tras añadir
   geometrías de monocapa al dataset.
3. **Fuera de distribución** — el checkpoint se entrenó sobre `merged_bilayer_stackings_md`
   (`graphene_hBN_bilayer`), nunca vio grafeno monocapa. *Se falsaría* con el mismo residuo sobre una
   instantánea de bicapa tomada de la distribución de entrenamiento.

Requisito de dataset si se abre: **derivative-aware**. Un checkpoint seleccionado por pérdida espectral
no lleva ninguna restricción sobre `∂H/∂R`, y este residuo es la medida de ese hueco.

## 7. Lo que este `NO_GO` no dice

No dice que graphene acople mal, ni que la física del E2g esté mal resuelta. El path SIESTA entrega el
acoplamiento E2g sobre el par de Dirac en `K` (‖g‖ = 13.53 eV/Å), satisface Hellmann–Feynman
generalizado a `1e-13`, da el cero acústico a `2.9e-11` eV/Å y cierra los round-trips de unidades. El
`NO_GO` es una afirmación sobre un proveedor sin validar dentro de este repositorio —y sobre dos reglas
congeladas cuyo alcance S25 y S26 ya documentaron—, no sobre la física del material.

Lo publicable sigue siendo el escalón 5: la validación de derivadas a tres paths más una **cota** sobre
`g` cuya anchura es la sensibilidad intra-atómica. Este paso añade a esa cota su interpretación: de qué
está hecho el residuo, y qué haría falta para tener derecho a culpar al modelo.

## 8. Limitaciones que viajan con esta lectura

1. `VERIFIED_BINARY_ONLY`: el runtime SIESTA está hasheado pero no identificado por source (C01).
2. α es un ajuste contra la referencia. Todo número «tras α» es diagnóstico y no describe el
   rendimiento del checkpoint.
3. Los `r` fuera de la ventana congelada (manifold π, bloques ocupado/vacío) son ejes de sensibilidad
   pre-registrados, no el gate. El gate sigue siendo la ventana 2+2.
4. Todo esto es Γ y una celda de dos átomos: la degeneración de las direcciones en el plano (§3.2) es
   propiedad de esta celda y no debe extrapolarse a `q ≠ 0` ni a la bicapa.

## 9. Reproducir

```bash
.venv/bin/python Comparison/scripts/interpret_gamma_epc_error.py
.venv/bin/python -m pytest tests/test_interpret_gamma_epc_error.py
```
