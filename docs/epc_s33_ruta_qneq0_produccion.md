# D02 / E-F_001-S33 — comparación bloqueada de rutas `q != 0`

S32 prototipó y midió Q-A/Q-B/Q-C algebraicamente (`docs/epc_s32_qb_qc_prototypes.md`) pero
dejó la elección de ruta productiva explícitamente pendiente ("no decide todavía cuál de
Q-B/Q-C es la ruta productiva de magic-angle TBG"). GO-5 sigue `NO_GO`, por lo que este ticket no
puede cerrar esa elección; conserva únicamente medidas de candidatos.
decision record, sin arrancar ninguna producción MATBG `q != 0` (roadmap, criterio de
aceptación de S33).

Productor: [`Comparison/scripts/epc_qb_qc_prototypes.py`](../Comparison/scripts/epc_qb_qc_prototypes.py)
(`production_route_decision()`).
Tests: [`tests/test_epc_qb_qc_prototypes.py`](../tests/test_epc_qb_qc_prototypes.py), sección
`E-F_001-S33`.
Depende de: [[epc_s32_qb_qc_prototypes]], [[epc_s31_graphene_k_go5_verdict]],
[[epc_s30_graphene_k_conmensurable]], [[epc_s29_convenciones_qspace]].

Convenciones: **[CODE]** confirmado en este repositorio, **[MEAS]** medido en S32/S33,
**[DESIGN]** decisión tomada en este documento.

## 1. Veredicto

| eje | resultado |
| --- | --- |
| exactitud | empate |
| información de respuesta de base | empate, ortogonal a esta elección |
| coste | Q-C gana a partir de un puñado de `q` |
| cache/restart | decisivo a favor de Q-C |
| **ruta primaria de producción** | **UNDECIDED (BLOCKED por GO-5)** |
| **ruta de validación** | Q-B (cross-check de un solo `q`, no se escala) |

`production_route_decision()` publica exactamente esto como datos, en el mismo estilo que
`go5_decision()` (`epc_commensurate_k.py`) y que `evaluate_epc_metrics.py` publica el veredicto
de GO-4: nada de lo que sigue se recalcula fuera de esa función y de sus tests.

**[DESIGN] Este documento no es un GO.** Es downstream de GO-5 (`NO_GO`, bloqueado por
GO-4/C14C, `docs/epc_s31_graphene_k_go5_verdict.md`). Fija qué ruta usará producción *cuando*
GO-4/GO-5 se desbloqueen; no las desbloquea, y no ejecuta nada sobre grafeno K real ni sobre
MATBG.

## 2. Los cuatro ejes

**[MEAS] Exactitud — empate.** `compare_routes()` ya mostró (S32) que ambas rutas coinciden con
Q-A donde Q-A puede pronunciarse, y entre sí en `k` incommensurado, al ruido de diferencias
finitas (`worst_case_agreement_D_H ~ 5e-11`). Ninguna ruta es más exacta que la otra; este eje no
decide.

**[MEAS] Respuesta de base — empate, ortogonal.** `D_S` se produce con el mismo mecanismo que
`D_H` en ambas rutas (`max_abs_D_S_qa_vs_qc`, `max_abs_D_S_qa_vs_qb`, ambos `~3e-12`). Ninguna
ruta implementa `S_L`/`S_R` por separado. Lo que C14C/GO-1 exija finalmente para el término
intra-atómico es una decisión de `basis_response_backend`, no de Q-B-vs-Q-C: elegir una ruta
aquí no desbloquea el gate 5 de GO-4, y no tiene por qué hacerlo. Si GO-1 exige `S_L`/`S_R` por
separado, ambas rutas necesitarían la misma extensión.

**[MEAS] Coste — Q-C gana más allá de un puñado de `q`.** `scaling_sweep()` mide el cruce entre
`n_q=1` y `n_q=5` (`n_k=5`, `shells=4`, ver tabla en S32 Sección 5): para un único `q`, Q-B es
más barato (no hay tabla que mantener); para varios, la tabla de Q-C se amortiza y gana con
margen creciente. **[DESIGN]** Una campaña selected-mode MATBG (Fase 10d) muestrea varios
sectores de baja energía (Γ/K-like, acoustic/strain, shear/breathing, low-energy moiré),
plausiblemente en más de un `q`, y Fase 12 (`lambda`, `alpha^2F`) exige explícitamente un
`q`-mesh — exactamente el régimen donde Q-C gana. **[OPEN]** El cruce medido es del régimen toy
(`n_k=5`, `shells=4`); no es una constante de MATBG y debe remedirse a la escala real antes de un
preflight de producción (roadmap Sección X, "el benchmark decide").

**[MEAS][DESIGN] Cache/restart — decisivo.** `kernel_table()` es un intermedio genuinamente
`q`-independiente: encaja exactamente en la granularidad de restart que pide el roadmap
(Sección X: `(q, mode_or_test_displacement, derivative, k-block)`) como su propio nodo
persistible, de forma que una campaña reiniciada reutiliza la tabla en vez de repetir
diferencias finitas ya pagadas. `phase_aware_kernel()` (Q-B) no tiene ningún intermedio así:
cada `q` paga su propio paso de diferencias finitas de nuevo, haya o no restart. Este es el eje
que de verdad decide, porque es exactamente uno de los cuatro criterios que pide S33.

**[OPEN] Gap de firma detectado.** `shared/artifact_signature.py` ya cubre el resultado final
por `(k, q)` de cualquiera de las dos rutas con el nodo `raw_derivative` (campos:
`derivative_backend`, `derivative_method`, `perturbation_definition`, `q`, `delta_or_jvp`,
`topology_sha256`, `convention`, `dtype`, `units`), pero no tiene un nodo separado para la tabla
`q`-libre de Q-C. Sin él, un restart de proceso repetiría la construcción de la tabla aunque sea
`q`-independiente: hace falta un nodo tipo `raw_derivative_kernel_table` (clave: dirección +
topología + `delta_or_jvp`), consumido por el nodo `raw_derivative` por `(k, q)`, para que la
amortización de Q-C sobreviva a un restart y no solo a la vida de un único proceso. Esto es un
requisito de implementación para producción, no de este ticket (S33 no implementa producción).

## 3. Rechazo explícito: el JVP Γ no es una ruta `q != 0`

**[DESIGN]** `production_route_decision()` publica `rejects_gamma_jvp_reuse: True` y una razón
explícita. El JVP direccional de Graph2Mat existente
(`graph2mat_autograd_derivatives.py::compute_graph2mat_directional_derivative`) diferencia una
tangente periódica de celda: es `q=0` de esa celda por construcción (roadmap Sección II). No es
sustituible por Q-B ni por Q-C, silenciosamente o no — el blocker B4 nombra exactamente este
fallo, y GO-8b exige explícitamente "ninguna reutilización del JVP Γ como sustituto silencioso de
una perturbación `q≠0`". Cualquier cambio futuro que alimente un resultado del JVP Γ a una
contracción `q≠0` sin pasar por `phase_aware_kernel`/`kernel_table` (o sus equivalentes sobre
Graph2Mat real) es una violación de GO-8b, no una optimización.

## 4. Tests adicionales exigidos antes de GO-8b

`production_route_decision()["additional_tests_required_before_go8b"]` publica la lista como
datos; en prosa:

1. Q-C y Q-B implementadas como candidatos sobre el Graph2Mat real, sin seleccionar una ruta
   (`graph2mat_autograd_derivatives.py`), no solo sobre la cadena toy de este módulo;
2. test de margen de topología: ningún par de átomos que toque el kernel puede cruzar el cutoff
   fijo de `edge_index`/`shifts` del checkpoint (roadmap, "Neighbor topology");
3. los mismos cinco checks que S31 corrió para Q-A (`q`/`-q`, cell-origin shift, átomo cruzando
   el borde periódico, convención de fase atómica alternativa) repetidos sobre el kernel de
   Graph2Mat real, no solo sobre el toy;
4. acuerdo contra Q-A en grafeno K real, en cuanto GO-4/C14C se desbloquee
   (`docs/epc_s31_graphene_k_go5_verdict.md`, Sección 4);
5. un artifact de cache `raw_derivative_kernel_table` (Sección 2) con un test de restart:
   matar el proceso tras completar `kernel_table()`, reanudar, y confirmar que no se repite
   ninguna diferencia finita;
6. preflight de GPU sobre la construcción real del kernel de Graph2Mat (no los bloques densos
   2x2 de este toy); el JVP forward-mode ya es compatible con CUDA según verificación previa, así
   que este test debe extender esa comprobación, no repetirla;
7. un re-run de `scaling_sweep()` a la escala real de shells/edges de MATBG y a los `n_q`, `n_k`
   reales de la campaña, para confirmar que el cruce medido aquí sobre un toy sigue siendo válido
   a escala de producción;
8. round-trip de serialización sparse de las salidas de `kernel_table`/`phase_aware_kernel` a
   través del mapping Graph2Mat -> sparse existente (C11), no solo bloques densos 2x2.

## 5. Qué prueba esto y qué no

- **Prueba:** que, dadas las cuatro medidas que S33 pide (exactitud, información de respuesta de
  base, coste, capacidad de cache/restart), Q-C queda como candidato y Q-B como posible
  cross-check, con evidencia reproducible (`compare_routes()`, `scaling_sweep()`) y sin haber
  arrancado ninguna producción MATBG.
- **No decide:** una ruta productiva hasta que GO-5 sea `PASS`; tampoco prueba que Q-C funcione
  sobre Graph2Mat real, sobre grafeno K real, ni a escala MATBG
  — eso es exactamente la lista de la Sección 4, y sigue bloqueado por GO-4/GO-5 igual que antes
  de este ticket.

## 6. Reproducir

```bash
.venv/bin/python -c "import sys; sys.path.insert(0,'Comparison/scripts'); sys.path.insert(0,'shared'); import epc_qb_qc_prototypes as p; import json; print(json.dumps(p.production_route_decision(), indent=2))"
.venv/bin/python -m pytest tests/test_epc_qb_qc_prototypes.py -v -k production_route
```
