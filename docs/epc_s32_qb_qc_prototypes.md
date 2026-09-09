# D02 / E-F_001-S32 — prototipos mínimos de las rutas Q-B y Q-C

El roadmap (Seccion II) deja abierto cuál de dos rutas escalables produce una respuesta
`q != 0` sin replicar la celda moiré: **Q-B** (fase inyectada en la tangente de un edge
periódico antes de derivar) y **Q-C** (kernel de derivada real-space, contraído por Fourier).
Este ticket prototipa ambas y las certifica contra **Q-A** (`epc_commensurate_k.py`), la
referencia transparente ya validada algebraicamente en S30/S31.

Productor: [`Comparison/scripts/epc_qb_qc_prototypes.py`](../Comparison/scripts/epc_qb_qc_prototypes.py).
Tests: [`tests/test_epc_qb_qc_prototypes.py`](../tests/test_epc_qb_qc_prototypes.py).
Depende de: [[epc_s30_graphene_k_conmensurable]], [[epc_s31_graphene_k_go5_verdict]],
[[epc_s29_convenciones_qspace]], [[epc_go4_graphene_gamma_verdict]].

Convenciones: **[CODE]** verificado en este repositorio, **[MEAS]** medido aquí, **[DESIGN]**
decisión tomada en este documento.

## 1. Por qué un toy y no grafeno K real

**[CODE]** GO-4 (físico) sigue en `NO_GO` por el término intra-atómico sin resolver de la
respuesta de base (C14C, `docs/epc_go4_graphene_gamma_verdict.md`), y S31 ya estableció que
`epc_commensurate_k.py` heredaría exactamente ese mismo bloqueo sobre grafeno K real
(`docs/epc_s31_graphene_k_go5_verdict.md`, Sección 3). Ejecutar Q-B/Q-C sobre grafeno real
mediría el mismo gate abierto una segunda vez, no la mecánica de `q`-space que este ticket
existe para probar. Por eso los tres caminos se prototipan sobre una cadena periódica 1D de
dos átomos por celda (misma familia funcional que el toy de `test_epc_commensurate_k.py`:
hopping/overlap con decaimiento exponencial, onsite alternante), con estado
`prototype__algebra_only`, no `candidate__pending_go5`.

## 2. La física que hace funcionar ambas rutas

Para un átomo home siempre en imagen relativa `R=0` y un vecino en imagen `R_l`, mover el
átomo home por `v` y el vecino por `v * exp(i q.R_l)` (gauge de celda canónico, `epc_fourier`)
da, por la sola regla de la cadena:

```
dH(R_l)/deps = A_l * exp(i q.R_l) + B_l
A_l = dH(R_l)/dr_vecino . v      (no depende de q)
B_l = dH(R_l)/dr_home . v        (no depende de q)
```

Sumando en Bloch a `k` toda la red perturbada da la identidad real-space-a-k estándar de
electron-phonon, verificada aquí contra Q-A:

```
D_H(k+q, k) = sum_l A_l exp(i (k+q).R_l) + sum_l B_l exp(i k.R_l)
```

**Q-C** (`kernel_table` + `k_to_k_plus_q_kernel`) construye `{A_l, B_l}` **una vez por
dirección** (`q`-independiente) y evalúa la fórmula de arriba para cualquier `(k, q)` a coste
`O(shells)`. **Q-B** (`phase_aware_kernel` + `q_b_response`) hornea la fase de `q` en la
tangente *antes* de diferenciar — un único paso de diferencias finitas (dos reales: parte
`cos`/`sin` de la fase) que produce directamente `A_l exp(i q.R_l) + B_l` para ese `q`, y luego
sólo necesita la suma en `k`. Ninguna ruta materializa nunca un Jacobiano `(n_outputs, N, 3)`;
cada tabla es del tamaño de `shells`, no del tamaño de la supercelda.

## 3. Verificación contra Q-A: sólo donde Q-A puede verificar

**[MEAS]** Ambas rutas coinciden con Q-A y entre sí a nivel de ruido de diferencias finitas
(`~5e-11`, `delta=1e-5`) en los cuatro `k` conmensurables del toy (`0, 1/3, 2/3, -1/3` para
`q=1/3`, supercelda `Nc=3`): `tests/test_epc_qb_qc_prototypes.py::test_qb_and_qc_agree_with_the_commensurate_reference`.

**[CODE] Límite de la referencia, no de Q-B/Q-C.** El sandwich `bloch_character_basis` de Q-A
sólo es una proyección bien definida en `k` conmensurables — los que pliegan sobre los
`k`-puntos discretos de la propia supercelda (`epc_commensurate_k.py`, Sección 2: "`k -> k+q`
es un enunciado de subespacio"). En un `k` incommensurado con `q=1/3` (p. ej. `k=0.11`), la
fase `exp(i k.R_a)` no es invariante bajo la elección de qué imagen periódica representa la
"celda home", así que el número que produce el sandwich en ese `k` no corresponde a ninguna
proyección física — se comprobó numéricamente (`max diff ~0.2`, muy por encima del ruido de
FD) y se abandonó como comparación válida. Q-B y Q-C, en cambio, son sumas reales-espacio-a-`k`
ordinarias y están definidas para cualquier `k` continuo:
`test_qc_and_qb_still_agree_with_each_other_at_incommensurate_k` los verifica de acuerdo entre
sí en `k=0.11`, donde Q-A no puede pronunciarse. `compare_routes()` publica este límite
explícitamente en `reference_limits`, no lo oculta filtrando los `k` malos en silencio.

## 4. Basis response

**[MEAS]** `D_S` (la respuesta del overlap) se comparó con la misma máquina, mismo toy, misma
tolerancia: `max_abs_D_S_qa_vs_qc` y `_qb` están en el reporte de `compare_routes()`, típicamente
`~3e-12`. No se implementó `S_L`/`S_R` por separado en este prototipo — el toy usa un `D_S`
suficiente porque no hay aquí ninguna decisión C14C pendiente que lo exija (el toy no tiene
término intra-atómico); producción real seguirá exigiendo lo que determine `formalism_id`.

## 5. Escalado medido, no supuesto

**[MEAS]** `scaling_report()`/`scaling_sweep()` miden, no asumen, el cruce entre "construir una
vez" (Q-C) y "reconstruir por cada `q`" (Q-B). A `n_k` fijo (5), variando `n_q`:

| `n_q` | Q-C (s) | Q-B (s) |
| --- | --- | --- |
| 1   | 0.00157 | 0.00127 |
| 5   | 0.00346 | 0.00607 |
| 20  | 0.01115 | 0.02407 |
| 80  | 0.04232 | 0.09639 |

El cruce ocurre entre `n_q=1` y `n_q=5` en esta configuración (`shells=4`): para una única
`q`, Q-B es más barato (no hay tabla que mantener); para más de un `q`, la tabla de Q-C se
amortiza y gana con margen creciente. `scaling_sweep()` reporta el primer `n_q` de cruce medido
en cada corrida, no un número fijo en el código — el propio informe advierte que el cruce real
depende de `n_q`, `n_k` y `shells`, exactamente la actitud de la Sección X del roadmap ("el
benchmark decide").

`kernel_table_metrics`/`phase_aware_kernel_metrics` en `compare_routes()` reportan además
`fd_evaluations_to_build` (evaluaciones reales de `pair_block`, no una estimación) y
`sparsity_nonzero_shell_fraction` (fracción de shells dentro del cutoff con bloque no nulo;
`1.0` en el toy porque `CUTOFF_RADIUS_ANG` se fijó exactamente para incluir sólo `shells=1`,
consistente con lo que la supercelda `Nc=3` de Q-A puede representar sin ambigüedad de imagen
mínima).

## 6. GPU

**[DESIGN]** `gpu_preflight()` comprueba `torch.cuda.is_available()` (aquí, `True`) y registra
explícitamente por qué de todos modos se usa CPU: cada kernel es un puñado de bloques `2x2`
complejos indexados por `shells` (microsegundos a milisegundos, medido en `scaling_sweep`), y
este prototipo algebraico no invoca ningún modelo torch/Graph2Mat. La política de cómputo
(preflight antes de decidir, nunca CPU por defecto silencioso) se cumple registrando el
resultado del preflight y la razón, no omitiéndolo porque el problema sea pequeño.

## 7. Qué prueba esto y qué no

- **Prueba:** la mecánica algebraica de Q-B y de Q-C — la fórmula real-space-a-`(k,q)`, su
  independencia de `q` en el caso de Q-C, la equivalencia entre ambas rutas y con Q-A donde
  Q-A puede pronunciarse, y el trade-off de coste computacional entre "construir una tabla" y
  "rehacer la diferencia finita por cada `q`" — es correcta y medible.
- **No prueba:** nada sobre grafeno K físico ni sobre GO-5/GO-8b. Ambas rutas siguen
  bloqueadas por el mismo gate 5 de GO-4 en cuanto se les pida operar sobre un sistema real con
  respuesta de base (C14C). Tampoco decide todavía cuál de Q-B/Q-C es la ruta productiva de
  magic-angle TBG (Sección II: "no se elegirá antes de graphene K"); el cruce medido en la
  Sección 5 es evidencia para esa decisión, no la decisión en sí, y depende del régimen real de
  `n_q`/`n_k`/`shells` de una campaña MATBG, no del toy.

## 8. Reproducir

```bash
.venv/bin/python Comparison/scripts/epc_qb_qc_prototypes.py
.venv/bin/python -m pytest tests/test_epc_qb_qc_prototypes.py -v
```
