# D01 / E-F_001-S31 — valida graphene K y decide GO-5

S30 construyó la maquinaria genérica de la ruta Q-A (`epc_commensurate_k.py`) y la certificó
algebraicamente en una cadena de dos átomos, dejando explícitamente pendiente ["ejecutar esta
maquinaria sobre la celda conmensurable real de grafeno en K ... y comparar el resultado con
Q-B/Q-C. Ese es exactamente el trabajo que GO-5 evalúa"](epc_s30_graphene_k_conmensurable.md).
Este documento cierra esa pendiente: añade las comprobaciones que el roadmap exige para GO-5 y
no estaban en S30, y emite el veredicto.

Productor: [`Comparison/scripts/epc_commensurate_k.py`](../Comparison/scripts/epc_commensurate_k.py)
(`go5_decision()`).
Tests: [`tests/test_epc_commensurate_k.py`](../tests/test_epc_commensurate_k.py) (siete tests
nuevos, sección `E-F_001-S31`).
Depende de: [[epc_s30_graphene_k_conmensurable]], [[epc_s29_convenciones_qspace]],
[[epc_go4_graphene_gamma_verdict]].

Convenciones: **[CODE]** verificado en este repositorio, **[MEAS]** medido aquí, **[DESIGN]**
decisión congelada en este documento.

---

## 1. Veredicto

| capa | resultado |
| --- | --- |
| algebra (S30 + los cinco checks de S31) | **PASS** |
| física (ejecución sobre grafeno K real) | **NOT_ATTEMPTED** |
| GO-5 (global) | **NO_GO**, bloqueado por GO-4 |

`go5_decision()` en `epc_commensurate_k.py` publica exactamente esto como datos, en el mismo
estilo que `commensurate_contract()` y que `evaluate_epc_metrics.py` publica el veredicto de
GO-4: nada de lo que sigue se recalcula fuera de esa función y de sus tests.

## 2. Qué añade S31 sobre S30

El roadmap (Fase 7, §IX.F) pide, además de lo que S30 ya cubría (conmensurabilidad, cos/sin,
`k -> k+q` por subespacio), cinco comprobaciones más. Las cinco están ahora implementadas como
tests sobre la misma cadena de dos átomos de S30, cada una **[MEAS]** a precisión de máquina o
mejor:

| check | test | tolerancia medida |
| --- | --- | --- |
| `q` y `-q` | `test_D_H_of_minus_q_is_the_complex_conjugate_of_D_H_of_q` | `D_H`: `1e-8`; `g`: `1e-6` |
| cell-origin shift | `test_g_is_invariant_under_a_relabeling_that_crosses_the_periodic_boundary` | Frobenius `1e-10`, valores singulares `1e-9` |
| átomo cruzando el borde periódico | (mismo test: la relabeling *es* el cruce) | igual |
| convención de fase atómica alternativa | `test_alternative_atomic_phase_convention_round_trips_to_the_same_pattern` | `1e-12` |
| singular values / projectors / block metrics (GO-6) | los dos tests de arriba leen `report["g"]["singular_values"]` y `["frobenius_norm"]`, no los valores crudos de `D_H` | — |

**[CODE] `q` y `-q`.** S29 fija `X(-q) = conj(X(q))` para bloques reales en tiempo real
(`MINUS_Q_CONVENTION`). `S31` construye la celda y el modo en `-Q` de forma independiente —no
niega el signo a mano— y comprueba que `D_H(-q)` es el conjugado de `D_H(q)` **y** que el bloque
físico `g` (tras la contracción completa `k -> k+q`, GO-6) también lo es. La segunda parte es la
que S30 no había probado: que la identidad sobrevive a la contracción, no solo a la combinación
lineal de las dos mitades.

**[CODE] cell-origin shift / átomo cruzando el borde periódico.** En el anillo de S30 no hay
origen distinguido: renombrar qué celda primitiva es la "celda 0" permutando índices *es*
exactamente ambas operaciones a la vez. La verificación numérica confirma
`_matrices(cell.positions_ang[order]) == _translate(H, cells)` (la relabeling reproduce
`_matrices` exactamente, no es una coincidencia de test), y que el acoplamiento PAO proyectado —normas de
Frobenius y valores singulares gauge-invariantes de GO-6— no se mueve, aunque `D_H` en bruto sí
recoge una fase (`test_complex_D_H_carries_the_q_phase_under_a_primitive_translation`, ya en
S30). Antes de S31 sólo se probaba lo segundo.

**[CODE] convención de fase atómica alternativa.** Se construye un `PhononModeSet` en la
convención `atom` (la que declara `EXTERNAL_CONVENTIONS["dftbephy"]` en `epc_fourier.py`), con el
autovector multiplicado por el factor de fase atómica inverso, y se convierte con
`phonon_provider.to_canonical_conventions`. El patrón complejo que produce `complex_mode()` a
partir del modo convertido coincide con el patrón canónico a `1e-12`. Esto ejercita el límite de
adaptador de S29 (`to_canonical_conventions`) *a través* del límite de S30 (`complex_mode`), que
no estaba probado en conjunto.

## 3. Por qué la capa física queda en `NOT_ATTEMPTED`, no en `FAIL`

El roadmap es explícito: *"Solo cuando éste [Γ] pase se autoriza graphene K"* (§IX.F), y GO-4
(físico) está en **NO_GO** —
[docs/epc_go4_graphene_gamma_verdict.md](epc_go4_graphene_gamma_verdict.md), gate 5— por un
término intra-atómico sin resolver en el proveedor de respuesta de base (C14C). Este módulo
(S30) construye las dos mitades `cos`/`sin` de cualquier modo `q != 0` como **artifacts Gamma
ordinarios de la supercelda** (Sec. 2 de S30): usan el mismo `basis_response_backend` que Γ,
sin excepción. Por tanto, ejecutar la maquinaria sobre grafeno K real heredaría exactamente el
mismo gate abierto, no probaría nada nuevo sobre `q`, y produciría un candidato con la misma
etiqueta `bound__intra_atomic_basis_response_unresolved` que ya tiene el candidato Γ.

Adjudicar esa ejecución como si fuera un experimento nuevo sería medir el mismo fallo dos veces
con la esperanza de que la celda más grande lo disimule — exactamente lo que la sección I del
roadmap prohíbe (*"no se aceptará... hasta demostrar qué conexión/ortogonalización/gauge la
produce"*). `go5_decision()` por eso marca la capa física `NOT_ATTEMPTED`, distinta de `FAIL`: no
es que grafeno K haya fallado una medida, es que la medida heredaría un bloqueo ya conocido y
atribuido, y ejecutarla no lo cierra.

## 4. Qué haría falta para desbloquear GO-5

Exactamente lo que ya haría falta para GO-4 — nada adicional específico de `q`:

1. cerrar C14C (el término intra-atómico monocéntrico de la respuesta de base);
2. resolver las dos reglas congeladas de GO-4 (traslación uniforme, simetría del doblete en K)
   o aceptar la medida que ya dan como respuesta a la pregunta que intentaban hacer.

Sólo entonces tiene sentido generar la referencia SIESTA/Graph2Mat real en la celda
`sqrt(3) x sqrt(3)` de grafeno y comparar contra Q-B/Q-C — el trabajo que S30 dejó pendiente y
que S31 confirma que la maquinaria ya sabe hacer, en cuanto la capa física deje de estar
bloqueada.

## 5. Garantías

- **la ausencia nunca es un PASS**: `go5_decision()` es datos explícitos, `algebra_layer` y
  `physical_layer` se reportan por separado y `go5` nunca se computa mezclándolos en un único
  booleano;
- **el protocolo no se edita para ajustar el resultado**: `blocked_by` cita el documento y el
  gate exactos de GO-4 (`GO4_BLOCKING_GATE`), no una paráfrasis;
- los cinco checks de la Sección 2 son tests, no afirmaciones de prosa: `pytest
  tests/test_epc_commensurate_k.py` los ejecuta a precisión de máquina o mejor en cada corrida.

## 6. Reproducir

```bash
.venv/bin/python -c "import sys; sys.path.insert(0,'Comparison/scripts'); sys.path.insert(0,'shared'); import epc_commensurate_k as ck; import json; print(json.dumps(ck.go5_decision(), indent=2))"
.venv/bin/python -m pytest tests/test_epc_commensurate_k.py
```
