# D01/D03 / E-F_001-S30 — la referencia conmensurable de grafeno K (ruta Q-A)

Grafeno `K` es el primer gate `q != 0` de EPC. Este documento fija cómo se construye la
referencia transparente (ruta Q-A: la celda crece hasta que `q` es periódica) y qué garantiza
la implementación antes de que GO-5 la promueva.

Productor: [`Comparison/scripts/epc_commensurate_k.py`](../Comparison/scripts/epc_commensurate_k.py)
(`schema = epc_commensurate_q_reference_v1`).
Tests: [`tests/test_epc_commensurate_k.py`](../tests/test_epc_commensurate_k.py) (cadena
periódica de dos átomos, tripled a `q = (1/3, 0, 0)`: suficientemente pequeña para diferenciar
a mano, suficientemente estructurada para falsear cada afirmación).
Depende de: [[epc_s29_convenciones_qspace]] (D01, convención q-space), C16/`epc_subspaces.py`
(GO-6), `compute_epc_matrix_elements.py` (F1/F2), `phonon_provider.py`.

Convenciones de evidencia: **[CODE]** verificado en este repositorio, **[DESIGN]** decisión
congelada aquí, **[OPEN]** pendiente.

---

## 1. Qué resuelve la ruta Q-A

El JVP de Graph2Mat sobre una única celda es Γ-periódico: un vector `v` inyectado en las
posiciones de `N` átomos representa una perturbación con la periodicidad de esa celda, nunca
`q != 0`. La ruta Q-A rodea el problema en vez de resolverlo en la celda primitiva: se
construye una supercelda conmensurable `M` tal que `exp(2i pi q.R)` es periódica en ella, y
`q` deja de ser una fase que aplicar a los bloques para convertirse en el propio patrón de
desplazamiento. Todo lo certificado en la cadena Γ (derivadas SIESTA, JVP, basis response,
`(F1)`/`(F2)`) se reutiliza sin tocarlo:

```text
q, M  ->  supercell donde exp(2i pi q.R) es periódica     (CommensurateCell)
e_qnu ->  u_l = u(q) exp(2i pi q.R_l), separado Re / Im    (ComplexMode)
cos, sin  ->  dos D_H + basis response Gamma ordinarios    (cadena existente)
D_H(q) = a_cos D_H[cos] + i a_sin D_H[sin]                 (ComplexModeResponse)
```

**[DESIGN]** Q-A es la referencia contra la que se validan Q-B (JVP con fase por `shift`) y
Q-C (kernel + Fourier), no la ruta productiva de MATBG: replicar la celda moiré para cada `q`
es prohibitivo a esa escala. Aquí solo se demuestra que el número es correcto.

## 2. Tres líneas que este módulo se niega a difuminar

**[CODE] El `q` del artifact es el de la supercelda, y es Γ.** En Q-A el `q` finito vive en el
patrón de desplazamiento, no en la fase de los bloques de Bloch. `ComplexModeResponse` exige
que las dos mitades (`cos`, `sin`) estén firmadas en `q = 0` (`raw.q == (0,0,0)`) y registra el
`q` primitivo por separado en `q_primitive`/`supercell_matrix`. Re-etiquetar un artifact Γ como
`q = K` es la sustitución que GO-8b prohíbe explícitamente; producir el mismo número mediante
una celda más grande es la referencia contra la que GO-8b se valida. `test_a_gamma_phased_block_cannot_carry_the_finite_q_pattern`
comprueba que una mitad fasada a `q != 0` se rechaza con excepción.

**[CODE] `cos` y `sin` permanecen artifacts separados.** Son dos respuestas direccionales
reales, independientes, firmadas por separado (`direction_hash` distinto, cada una con su
propio `RawDerivative`/`BasisResponse`). El modo complejo es su combinación lineal,

```text
D_H(q) = a_cos * D_H[cos] + i * a_sin * D_H[sin]
basis_response(q) = a_cos * basis_response[cos] + i * a_sin * basis_response[sin]
```

con las dos amplitudes de Frobenius en Å que la normalización unitaria de
`fd_perturbation_space.Direction` había retirado (`ComplexMode.cos_amplitude_ang` /
`sin_amplitude_ang`). La combinación es lineal en la respuesta — de ahí que sea exacta y no
aproximada — y se aplica igual a `D_H` que a la respuesta de base, cualquiera que sea su
representación (`D_S` o `S_L`/`S_R`), mediante `ComplexModeResponse.D_H_at_k` /
`basis_response_at_k`.

**[CODE] `k -> k+q` es una afirmación de subespacio, no un índice de banda.** Conmensurable
significa exactamente que `k` y `k+q` primitivos se pliegan al mismo `k` de la supercelda
(`CommensurateCell.folds_together`), así que ambos viven en una única ventana propia y se
distinguen por su carácter de Bloch primitivo, medido en la métrica `S` con la maquinaria GO-6
(`bloch_character_basis`, `project_character`, `epc_subspaces.subspace_overlap`). No hay orden
de bandas, ni matching de autovalores: `project_character` proyecta, re-ortonormaliza (Löwdin)
y re-diagonaliza dentro del subespacio (Rayleigh-Ritz), y rechaza explícitamente cualquier
autovalor del proyector que no sea 0 o 1 dentro de `occupancy_tol` — una ventana que corta a
la mitad el carácter de un estado no puede usarse para esa proyección.

## 3. Objetos y su contrato

| objeto | qué es | invariantes que impone |
| --- | --- | --- |
| `CommensurateCell` | `M`, `q_primitive`, geometría primitiva, imágenes representativas | `M` entera, `det M != 0`, `M @ q` entero (`COMMENSURABILITY_TOL = 1e-9`) |
| `ComplexMode` | `u_l = u(q) exp(2i pi q.R_l)` separado en `cos`/`sin` + dos amplitudes en Å | ninguna mitad puede tener amplitud `<= AMPLITUDE_TOL_ANG` (modo real: no necesita este módulo); la fuente debe ser `phonon_provider.PhononMode` canónico, nunca un `synthetic_test_displacement` (B7) |
| `ComplexModeResponse` | `D_H(q)`, basis response de la combinación, más provenance | ambas mitades en `q=0`, mismo `formalism_id`, misma representación de basis response, mismo `no_u`, mismo `isc_off` |
| `bloch_character_basis` | columnas `e[(mu,l)] = exp(+2i pi k.R_l)/sqrt(N_c)` de un `k` primitivo | signo de Fourier electrónico canónico (S29), gauge de celda |
| `project_character` | ventana proyectada + re-diagonalizada sobre un carácter | autovalores del proyector en `{0,1}` a `occupancy_tol`; sin eso, `CommensurateError` |
| `k_to_k_plus_q_report` | `g` entre los subespacios `k` y `k+q` de una ventana, con métricas | `k` y `k+q` deben plegar al mismo `k` de la ventana; siempre `status = candidate__pending_go5` |

Todo objeto expone su propio hash (`supercell_hash`, `complex_mode_hash`) para que el cache
dependency-aware (Sección V del roadmap) invalide exactamente lo que cambió: cambiar `q` o
`M` invalida la celda y el modo, pero no las respuestas Γ de las que se construyen las mitades.

## 4. Qué certifica el test toy y por qué basta

`tests/test_epc_commensurate_k.py` usa una cadena periódica de dos átomos con hopping y
overlap exponenciales, tripleada a `q = (1/3, 0, 0)`. No es grafeno, pero es suficiente para
falsear cada afirmación de la sección 2 con una cuenta cerrada:

- **conmensurabilidad**: `diagonal_supercell_matrix` + `CommensurateCell` reproducen `M`,
  `N_c` y el mapa de imágenes a mano; una `M` incomensurable se rechaza.
- **reconstrucción compleja**: `mode.pattern_ang` coincide con `u(q) exp(2i pi q.R_l)` evaluado
  directamente, hasta `1e-12`.
- **ley de transformación de `q != 0`**: bajo una traslación primitiva de la red, la matriz sin
  perturbar es invariante pero `D_H(q)` recoge exactamente `exp(2i pi q.R)` — la firma que un
  objeto Γ no puede tener. Esto es lo que distingue una referencia `q != 0` genuina de un
  artifact Γ relabeleado.
- **la respuesta de base sigue la misma combinación que `D_H`**, y `S_L + S_R` reproduce la
  `D_S` de la diferencia finita del patrón complejo completo.
- **`k -> k+q` por subespacio**: en el punto plegado, `k` y `-k` (es decir `k+q`) son
  degenerados y el solver mezcla arbitrariamente sus autovectores — exactamente el escenario
  que impide seleccionar por índice de banda. `unfolding_weights` confirma que el peso de
  carácter de cada estado suma uno sobre toda la estrella de plegado, y
  `k_to_k_plus_q_report` separa las dos ventanas por proyección, con solape de proyectores
  `< 1e-16` entre ellas. `test_the_projection_survives_a_gauge_rotation_of_the_window` confirma
  que la proyección es una propiedad de la ventana, no de la base arbitraria del solver.

**[DESIGN]** El toy certifica el álgebra (la parte que S30 tenía que demostrar); no sustituye
la ejecución sobre grafeno real. Producir la referencia física en la celda `sqrt(3) x sqrt(3)`
de grafeno con SIESTA/Graph2Mat certificados es GO-5 y queda fuera del alcance de este módulo,
que es la maquinaria genérica que GO-5 va a usar.

## 5. Estado

**[CODE]** La maquinaria de la ruta Q-A (celda, modo complejo, respuesta combinada, selección
por subespacio) está implementada y validada algebraicamente. Todo lo que produce queda
etiquetado `candidate__pending_go5` (`CANDIDATE_STATUS`); nada de este módulo se declara a sí
mismo GO-5.

**[OPEN]** Falta ejecutar esta maquinaria sobre la celda conmensurable real de grafeno en `K`
(SIESTA + Graph2Mat certificados, fonón `K` de un `PhononProvider` validado) y comparar el
resultado con Q-B/Q-C. Ese es exactamente el trabajo que GO-5 evalúa, y es lo que promueve —
o no — los artifacts que este módulo sabe construir.
