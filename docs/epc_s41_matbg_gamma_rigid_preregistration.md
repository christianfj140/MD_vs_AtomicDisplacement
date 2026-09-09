# E-F_001-S41 — pre-registro de la campaña rígida Γ de MATBG

Fase 10d del roadmap ("Physical selected-mode campaign"), acotada a lo que un pre-registro puede
congelar **hoy**: S38/S39 ya certificaron `NO-GO-8a` (ningún `PhononProvider` MATBG escalable y
validado existe) y C14C sigue `NO_GO` por su término intra-atómico sin resolver. Ninguno de los
dos gates se cierra en este ticket. Lo que se congela es todo lo que una futura ejecución GO-9 no
debe poder elegir **después** de ver `g`: los sectores Γ candidatos, la regla y lista de `k`
electrónicos, el contrato de ventana/ocupación, los splits, las tolerancias y la escalera de
claims — en un fichero hasheado, reutilizando (no reimplementando) las utilidades genéricas de
`preregister_graphene_gamma_epc.py` (C19/C20): `freeze_hash`, las guardas de disjunción de splits,
los helpers de JSON.

Productor: [`Comparison/scripts/preregister_matbg_gamma_rigid_epc.py`](../Comparison/scripts/preregister_matbg_gamma_rigid_epc.py).
Tests: [`tests/test_preregister_matbg_gamma_rigid_epc.py`](../tests/test_preregister_matbg_gamma_rigid_epc.py).
Artifact: `Comparison/results/epc/preregistration/matbg/epc_matbg_rigid_gamma_protocol.json`.

Convenciones de evidencia: **[CODE]** verificado en este repositorio, **[MEAS]** medido al
congelar, **[DESIGN]** decisión pre-registrada, **[OPEN]** abierto explícitamente.

---

## 1. Dos diferencias estructurales frente al protocolo de graphene

**Sin medición en vivo de `k`/modo.** El protocolo de graphene mide los candidatos en vivo
(`eigh` denso de una celda de 2 átomos, segundos). La celda rígida `(31, 30)` tiene 44 656
orbitales; una resolución dispersa generalizada por candidato es exactamente el coste que la
política de cómputo de este repositorio exige preflight antes de correr, no un efecto colateral de
congelar un documento. Este protocolo congela por tanto la **regla** y la **lista ordenada de
candidatos** (mismo formato que `K_CANDIDATES` de graphene) y registra
`k_selection_status = "pending_measurement"`: una ejecución futura aplica
`preregister_graphene_gamma_epc.select_result_k` —ya genérica— a espectros medidos reales; no
re-deriva la regla. **[DESIGN]**

**Sin path de referencia SIESTA.** S38 midió que una campaña `FC.Save.dHS` exacta en esta celda
necesita `6 × 11164 = 66984` resoluciones SCF desplazadas y la calificó de intratable a esta
escala (`docs/epc_s38_matbg_phonon_provider_decision.md` §3). `REFERENCE_PATHS` tiene aquí dos
entradas, no tres: JVP de Graph2Mat contra su propia diferencia central congelada. Un desacuerdo
entre ambas es un bug de backend; nunca se convierte en una afirmación sobre la precisión del
modelo frente a DFT. **[CODE]**

## 2. Los cinco sectores candidatos

Reutilizan exactamente `fd_perturbation_space.synthetic_benchmark_direction_set` (breathing/shear/
optical/random) más `uniform_translation`, aplicadas a las posiciones reales de la celda
`(31, 30)` — la misma llamada que `benchmark_matbg_synthetic_displacements.py` (S37/D06) ya usó
para medir coste. **[CODE]**

| split | direcciones | k | para qué |
| --- | --- | --- | --- |
| calibración | `translation_x` | Γ, M | fijar suelos FD y plateau de δ una vez exista medición real |
| validación | `random_seed0` | `k_generic_2` | ¿transfiere la cota calibrada? |
| resultado | `breathing_like`, `shear_like_x`, `optical_like` | `k_generic_1`, `K` | el experimento pre-registrado |

Cada sector queda etiquetado `synthetic_test_displacement` en todo momento — nunca
`phonon_eigenmode` ni `g_mn_nu` — hasta que exista un `PhononProvider` certificado GO-8a
(roadmap B7 / veredicto S39). **[CODE]**

## 3. `k`: regla congelada, medición diferida

`K_CANDIDATES` (dos `k` genéricos), `K_DEGENERATE` (análogo moiré de K) y `K_CALIBRATION` (Γ, M)
siguen la misma convención fraccionaria hexagonal que graphene. Dos honestidades explícitas:

- el único `k` con un artifact de eigenspace persistido en este repositorio es Γ, y proviene de
  `epc_synthetic_benchmark/eigenspace_probe_manual/` — una sonda manual, no un paso de campaña
  automatizado y firmado; se cita con esa salvedad, no como medición certificada. **[MEAS]**
- la near-degeneración de banda plana en el `K` moiré está medida en este repositorio únicamente
  por el modelo tight-binding Moon–Koshino (`docs/tbg_tight_binding_reference.md`, `S=I`, sin
  cross-check DFT/Graph2Mat) — no por el Hamiltoniano de producción. **[OPEN]**
- qué candidato es realmente "genérico" (grupo puntual identidad) sólo se puede confirmar cuando
  exista un espectro medido; hasta entonces `k_selection_status = "pending_measurement"` es el
  estado correcto, no un `k` elegido a ciegas. **[DESIGN]**

## 4. Contrato de ocupación: medido, no inventado

`occupation_report()` lee `Comparison/results/tbg_pure_graph2mat/neutrality_estimate.json`, un
artifact ya calculado (no producido por este ticket): `matrix_dimension = 44656` (coincide con el
recuento de orbitales del roadmap), `neutral_electrons = 44656`,
`target_occupied_bands_per_k = 22328`, `shift_eV = 0.0` (mismo `MU_CONVENTION` que graphene:
`exported_hamiltonian_fermi_zero`). **[MEAS]**

## 5. `tau_model`: deliberadamente sin definir

A diferencia de graphene, este protocolo **no** define `tau_model` para MATBG: no existe
referencia SIESTA a esta escala (§1) y transferir el `tau_model` de graphene exigiría un argumento
de transferibilidad probado que tampoco existe. `FORBIDDEN_TAU_MODEL_SOURCES` añade explícitamente
el resultado de escalado de coste JVP de S37/S40 — mide tiempo/NNZ, no precisión — a la lista ya
heredada de graphene. Mientras `tau_model` quede indefinido, ningún claim de este sistema puede
superar `graph2mat_derivative_validated_no_full_ks_coupling` en la escalera. **[DESIGN]**

## 6. Estado de las precondiciones al congelar

| gate | estado |
| --- | --- |
| GO-8a `PhononProvider` MATBG | **NO-GO-8a** |
| C14C respuesta de base (gate global) | **NO_GO** |
| GO-6 métricas de subespacio (gate global) | PASS |
| GO-8b respuesta `q≠0` MATBG | NO-GO-8b (informativo; no bloquea una campaña sólo-Γ) |

`ready_to_execute = False`. Bloqueado además por `k_selection` (sin medición en vivo) y por
`moire_hexagonal_convention_checked` (las etiquetas K/M son la convención de graphene trasladada,
sin verificar todavía contra la red recíproca real de esta celda). **[MEAS]**

## 7. Escalera de claims

Techo publicable hoy: **`graph2mat_derivative_validated_no_full_ks_coupling`** si —y sólo si— los checks
de nivel derivada pasan pese a `NO-GO-8a`. La etiqueta de publicación superior,
`rigid_tbg_gamma_epc_validated`, exige además GO-8a y C14C cerrados; la etiqueta de publicación en
sí misma queda fijada a `rigid_tbg_model` y `quantitative_relaxed_matbg_prediction` queda listada
como prohibida explícitamente (roadmap Fase 10, GO-10).

## 8. Reproducir

```bash
.venv/bin/python Comparison/scripts/preregister_matbg_gamma_rigid_epc.py            # congelar
.venv/bin/python Comparison/scripts/preregister_matbg_gamma_rigid_epc.py --verify   # hash + citas
.venv/bin/pytest -q tests/test_preregister_matbg_gamma_rigid_epc.py
```

Congelar es idempotente salvo por `generated_at`: el hash sólo cambia si cambia el protocolo o
alguno de los valores medidos que congela (geometría, ocupación, estado de gates).
