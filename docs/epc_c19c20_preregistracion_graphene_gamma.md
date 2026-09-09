# C19–C20 / E-F_001-S23 — pre-registro del experimento graphene Γ

Congelación del experimento **antes** de calcular `g`: perturbación, `k`, referencias, splits,
tolerancias, checks y escalera de claims quedan en un fichero hasheado que C19/C20 deben citar.

Productor:
[`Comparison/scripts/preregister_graphene_gamma_epc.py`](../Comparison/scripts/preregister_graphene_gamma_epc.py).
Tests: [`tests/test_preregister_graphene_gamma_epc.py`](../tests/test_preregister_graphene_gamma_epc.py).
Artifact: `Comparison/results/epc/preregistration/graphene/epc_graphene_gamma_protocol.json`.

Convenciones de evidencia: **[CODE]** verificado en este repositorio, **[MEAS]** medido al
congelar, **[DESIGN]** decisión pre-registrada.

---

## 1. Por qué un fichero y no una sección de un informe

Todo lo que podría elegirse *después* de ver `g` —qué miembro del doblete, qué `k`, qué δ, qué
umbral— se elige aquí, se mide donde debe medirse por valor y se hashea
(`frozen_content_sha256`, sobre todos los campos salvo `generated_at` y el propio hash).

- `--verify` recalcula el hash y además comprueba que **todo** JSON bajo `result_root` lleva
  `preregistration_sha256` igual al congelado. Un resultado que no lo cita no se produjo bajo
  este protocolo. **[CODE]**
- `require_frozen_protocol()` es la puerta de C19/C20: se niega a computar `g` si el fichero fue
  editado tras congelarse, si la revisión física no está `APPROVED` o si quedan precondiciones
  sin cumplir. **[CODE]**

## 2. Modo: el doblete E2g con gauge fijado por regla

Fuente: el artifact C18 `ranges/sc5x5x1/gamma_modes_asr_corrected.json` (el rango cuyo
IFC tail es 1.3e-3 y que C18 contrastó con `sc1x1x1` **a la misma malla efectiva de k**).

Un doblete degenerado no tiene miembro canónico, así que no se fija por índice sino por regla:
el vector del span con **máximo solapamiento con el patrón bond-stretch** de la celda (átomo 0
contra el enlace, átomo 1 a favor, ponderado por masa y normalizado), y su complemento
ortogonal *dentro del mismo span*, con el signo dado por la normal en el plano.

- solapamiento de la referencia con el span: **1.000** (la referencia vive en el doblete) **[MEAS]**
- ω = 0.193618 eV, spread del doblete 9.8e-6 eV, `A_zp = sqrt(ħ/2Mω)` = 0.029980 Å **[MEAS]**
- la regla es invariante bajo rotaciones 2×2 del doblete de entrada; el test lo comprueba
  alimentando el mismo span en cuatro gauges distintos **[CODE]**

«Banda 4 vs banda 5» es el gauge del solver; «la que estira el enlace» no lo es. El miembro
etiquetado es una etiqueta para el informe: `phonon_doublet_gauge_invariance` exige que las
métricas del bloque no dependan de ella.

## 3. `k`: genérico primero, degenerado después

**Etapa 1** — un `k` *genérico* (grupo pequeño identidad): ningún elemento de matriz forzado a
cero por simetría, así que un desacuerdo no puede esconderse dentro de un cero protegido. Se
elige por regla congelada (primer candidato genérico cuyos espaciados de ventana superen
`MIN_WINDOW_GAP_EV = 0.05 eV`) y se guarda **por valor** junto con el espectro que lo eligió y
con el motivo de rechazo de cada candidato batido.

- seleccionado: `k_generic_1 = (0.29, 0.11, 0)`, gap mínimo de ventana **0.820 eV** **[MEAS]**

**Etapa 2** — `K`, donde el doblete de Dirac convierte la misma medida en una medida de
subespacio, leída sólo a través de invariantes GO-6.

- split de degeneración medido en `K`: **1.3e-6 eV**, que es el suelo de ruptura de simetría del
  SCF y el umbral contra el que se juzga `symmetry_of_the_doublet_at_K` **[MEAS]**

Ventana: 2 estados por debajo y 2 por encima de la neutralidad, con el recuento de ocupados
derivado del número de electrones declarado (8, degeneración de espín 2) y **nunca** de dónde
caiga un gap.

## 4. Splits: dónde se calibra no es dónde se mide

| split | direcciones | k | para qué |
| --- | --- | --- | --- |
| calibración | `atom0000_x`, `translation_x` | Γ, M | fijar suelos FD, plateau de δ y el factor de transferencia |
| validación | `random_seed0` | `k_generic_2` | ¿transfiere la cota calibrada? Se usa una vez |
| resultado | `e2g_bond_longitudinal`, `e2g_bond_transverse` | `k_generic_1`, `K` | el experimento; se evalúa el último |

Disjuntos en pares `(dirección, k)` **y** en direcciones: la misma dirección a otro `k` sigue
siendo un suelo medido sobre el resultado. Dos guardas, no una:

- `assert_splits_disjoint()` sobre la tabla estática; **[CODE]**
- `assert_selection_outside_fitted_splits()` sobre el `k` que la **regla** elige en tiempo de
  congelación. Sin ella, si el candidato 1 fallara la regla, el `k` de validación ascendería en
  silencio a `k` de resultado: ajustar y medir en el mismo sitio. La tabla estática no puede ver
  ese caso. **[CODE]**

## 5. Las tres tolerancias, definidas en el subespacio electrónico

Ninguna es una norma de `ΔΔ` a secas: las tres son normas de la contracción en la ventana.

| tolerancia | definición | regla |
| --- | --- | --- |
| `tau_num` | `max` sobre pares del plateau de `‖C_W† (Δ(δ_a) − Δ(δ_b)) C_W‖_F` | además el resultado es INCONCLUSIVE si `‖C_W† Δ C_W‖_F < 10 · tau_num` |
| `tau_backend` | `‖C_W† (Δ_JVP − Δ_frozen) C_W‖_F` | `≤ 10 × tau_num(frozen)`: son la misma función de los mismos pesos, así que por encima del suelo del lado frozen es un bug, no una tolerancia |
| `tau_model` | `δg = C_f† (Δ_G2M − Δ_SIESTA) C_i`, `r = ‖δg‖_F / ‖C_f† Δ_SIESTA C_i‖_F` | `tau_model = 1.5 × max(r)` sobre el split de calibración; **debe** transferir al split de validación antes de aplicarse al resultado |

`FORBIDDEN_TAU_MODEL_SOURCES` nombra explícitamente lo que no puede convertirse en `tau_model`:
el Frobenius relativo de `D_H` de C14 (0.67), cualquier MAE/Frobenius global de elementos de
matriz, y en general cualquier norma global convertida en porcentaje de `g`. C14 queda registrado
como *prior a nivel derivada*, no como tolerancia sobre `g`.

Ensanchar `tau_model` a posteriori está prohibido: si la cota no transfiere, el veredicto para el
path del modelo es NO_GO.

## 6. Revisión física ligada a evidencia

`physics_review` aprueba cuatro ítems y cada decisión es una condición sobre evidencia medida,
no una firma: `mode` (C18 PASS + doblete estable en el rango IFC + la referencia vive en el
span), `k` (regla satisfecha con el gap medido), `references` (GO-2 PASS + firmas de
geometría/base idénticas entre la campaña de derivadas y la de fonones), `error_budget`
(las tres tolerancias definidas, las tres contraídas en el subespacio, contrato de calibración
completo y **cero leakage**). El slot de firma humana existe, firma el mismo hash y no sustituye
a ninguna de esas condiciones.

Estado al congelar: los cuatro `approved`, `geometry_consistency` consistente. **[MEAS]**

## 7. Estado de las precondiciones

| gate | estado |
| --- | --- |
| GO-2 derivadas SIESTA | PASS |
| C18 fonones Γ | PASS |
| C14C respuesta de base | **NO_GO** (término intra-atómico sin resolver) |
| GO-6 métricas de subespacio | **MISSING** |
| C14 error de derivada del checkpoint | PRESENT (`UNDECIDED`; gauge, referencia y base aún abiertos) |

`ready_to_execute = False`. Es el estado correcto y el que el protocolo debe registrar: el
pre-registro se congela ahora precisamente para que no pueda reescribirse cuando esos dos gates
se cierren. Mientras `intra_atomic_basis_response` siga abierto, la escalera de claims sólo
autoriza publicar la validación de derivadas a tres paths más una **cota** sobre `g` cuya anchura
es la sensibilidad intra-atómica de C14C.

## 8. Qué se compromete el protocolo a comprobar

13 checks (`checks`), separados en `stage: derivative` y `stage: epc`, entre ellos: FC vs FD
explícito de SIESTA, JVP contra la combinación de coordenadas (1e-11), JVP vs frozen,
topología de vecinos invariante bajo ±max(δ), Hellmann–Feynman generalizado diagonal, la
traslación uniforme como cero acústico, invariancia gauge electrónica *con dientes* (mezclar
clusters debe mover las métricas mucho más que rotarlos), invariancia gauge del doblete
fonónico, round-trip de unidades eV/Å·Å = eV, y la prohibición de construir `Δ_phys` sin el
artifact de respuesta de base que exige el `formalism_id`.

5 ejes de sensibilidad (`sensitivity`): rango FC, política ASR (crudo y corregido lado a lado,
nunca fusionados), δ, variante de formalismo (diagnóstico: la diferencia es el tamaño del
término que cada variante tira) y tamaño de ventana (2+2 vs 3+3).

5 escalones de claim (`claim_ladder`), con el suelo explícito: si fallan los checks numéricos o
de formalismo, la culpa no es del ML.

## 9. Reproducir

```bash
.venv/bin/python Comparison/scripts/preregister_graphene_gamma_epc.py            # congelar
.venv/bin/python Comparison/scripts/preregister_graphene_gamma_epc.py --verify   # hash + citas
.venv/bin/python Comparison/scripts/preregister_graphene_gamma_epc.py --sign-off "NOMBRE"
.venv/bin/python -m pytest tests/test_preregister_graphene_gamma_epc.py
```

Congelar es idempotente salvo por `generated_at`: el hash sólo cambia si cambia el protocolo o
alguno de los valores medidos que congela.
