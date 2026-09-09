# D01 / E-F_001-S29 — la convención q-space del repositorio

Una sola convención interna para `R`, `τ`, orientación de bloques, signo de Fourier
electrónico, signo de Fourier fonónico, fase atómica, `k+q` y `q ↔ −q`, más los adapters
que convierten *desde* cualquier proveedor externo *hacia* ella.

Productor: [`Comparison/scripts/epc_fourier.py`](../Comparison/scripts/epc_fourier.py)
(`convention_id = epc_qspace_canonical_v1`).
Tests: [`tests/test_epc_fourier.py`](../tests/test_epc_fourier.py).
Consumidores ya cableados: `epc_basis_response.py`, `compute_epc_matrix_elements.py`,
`phonon_provider.py`.

Convenciones de evidencia: **[CODE]** verificado en este repositorio, **[DESIGN]** decisión
congelada aquí, **[OPEN]** pendiente.

---

## 1. El canon

| objeto | convención canónica |
| --- | --- |
| `R` | imagen entera `l`; `R_l = l @ cell`, con las filas de `cell` = `a1,a2,a3` |
| `k`, `q` | fraccionarias en la base dual: `k·R_l = 2π (k_frac · l)` |
| `τ_κ` | posición atómica; **qué imagen es la de casa lo fija el artifact `geometry`** |
| orientación de bloques | `row_home_cell`: `X_{μν}(R_l) = <φ_μ(0)| x |φ_ν(R_l)>` |
| Fourier electrónico | `+1`: `X(k) = Σ_l exp(+2πi k·R_l) X(R_l)` |
| Fourier fonónico | `+1`: `u_{lκ} = e_κ(q) exp(+2πi q·R_l)` |
| fase atómica | `cell`: sin `exp(i k·τ)` en los orbitales ni en el autovector fonónico |
| `k+q` | estado final en `k+q`, sin plegar; el plegado por `G` es identidad en gauge de celda |
| `q ↔ −q` | no es una convención aparte: con bloques reales, `X(−q) = conj X(q)`, `e(−q) = conj e(q)` |

**[DESIGN]** El mismo `exp(+2πi ·)` en el lado electrónico y en el fonónico es lo que hace
que no sobreviva ningún factor relativo entre `D_H(q)` y el patrón de desplazamiento del
modo. El `1/√N_c` sigue donde lo dejó el memo de GO-1 (fuera del modo y fuera de `g`).

## 2. Las dos gauges y de qué es invariante cada una

Ambas son congruencias `X → U X U†` con `U` diagonal y unitaria, luego **ningún autovalor
generalizado se mueve** en ninguna de las dos. Lo que cambia es qué operación las deja
literalmente iguales:

| operación | gauge de celda (canónica) | gauge de átomo |
| --- | --- | --- |
| desplazar el origen `τ → τ + t` | invariante | fase global, se cancela en `U X U†` |
| reescribir un átomo en otra imagen `τ_κ → τ_κ + T` | `X'(k) = W X(k) W†`, `W = diag(e^{2πi k·T})` | invariante |
| plegar `k → k + G` | identidad (`e^{2πi G·R} = 1`) | `diag(e^{2πi G·τ})` |

**[CODE]** `relabel_home_images()` reindexa los bloques (`X'_{μν}(R) = X_{μν}(R + T_ν − T_μ)`)
y `image_relabel_unitary()` da la `W`; el test comprueba las dos rutas y que los autovalores
de `(H,S)` no se mueven. Un bloque **no nulo** que tuviera que salirse de la tabla de imágenes
lanza excepción: la supercelda auxiliar no es cerrada bajo ese reetiquetado y perder peso en
silencio sería exactamente el error que este módulo existe para impedir.

## 3. Adapters

`external → unitaria/conjugación explícita → canon`, en este orden y sin excepciones:

1. conjugar si el signo efectivo del proveedor es `−1`;
2. deshacer la gauge de átomo con `U X U†`, `U = diag(e^{+2πi k·τ})`.

**[CODE]** `to_canonical_at_k()` / `from_canonical_at_k()` son inversas exactas, y lo mismo
`to_canonical_vector_at_k(..., inverse=)` para autovectores electrónicos y fonónicos. Los
tests verifican las 8 combinaciones (signo × fase atómica × orientación) de dos maneras:
contra un `X(k)` construido directamente desde la definición externa, y por round-trip; el
residuo es `< 1e-13`, que para un producto de fases de módulo 1 es precisión de máquina.

**[CODE] Orientación de bloques ≡ signo de Fourier.** Con
`X'_{μν}(R) = <φ_μ(R)|x|φ_ν(0)> = X_{μν}(−R)`, una tabla `column_home` sumada con `+1` es la
tabla `row_home` sumada con `−1`. Por eso sólo interviene el producto
`effective_fourier_sign()`, y no hay una tercera operación que aprenderse.

**[DESIGN] Bloques vs matrices.** Para bloques en espacio real sólo puede importar la
orientación: un signo de Fourier o una fase atómica son afirmaciones sobre `X(k)`, no sobre
`X(R)`. `matrix_at_k()` (bloques) usa sólo la orientación; `to_canonical_at_k()` (una `X(k)`
ya sumada) usa las tres. Voltear el signo a nivel de matriz es una conjugación y **exige**
bloques reales en espacio real: con `real_space_blocks_real=False` (SOC, campo magnético)
la función se niega y remite a convertir los bloques.

**[CODE] Sin parches de fase por backend.** `EXTERNAL_CONVENTIONS` declara las convenciones
de cada proveedor (sisl/SIESTA, VIBRA, Graph2Mat en el canon; Phonopy y DFTBephy en gauge de
átomo); `external_conventions()` de un backend no declarado **lanza excepción** en vez de
dejar que alguien escriba un `exp(...)` a mano aguas abajo. Añadir un proveedor es añadir una
entrada, no una corrección.

## 4. Qué declara cada artifact

`convention_fields(q, q_aware_source=...)` es la única fuente del bloque q-space que firman
`raw_derivative`, `basis_response` y `pao_projected_mode_coupling`, así que «la misma convención» es
una igualdad de diccionarios y no una promesa en un docstring:

- **[CODE]** `PaoCovariantResponse` llama a `require_same_conventions()` entre `D_H` y la
  respuesta de base. Un `D_H` q-aware con una respuesta de overlap fasada en Γ — el fallo que
  GO-5 nombra explícitamente — no construye el objeto;
- **[CODE]** `convention_fields()` se niega a firmar `q ≠ 0` con `q_aware_source=False`
  (GO-8b) y se niega a firmar nada que no esté ya en el canon: la conversión ocurre en la
  frontera del adapter;
- **[CODE]** el `fourier_convention` del `physical_phonon` lleva el mismo `convention_id`, de
  modo que fonón y `D_H` son comparables sin interpretar dos textos distintos.

## 5. Estado

**[CODE]** La convención está fijada, cableada y testeada; las derivadas y las respuestas de
base la declaran; el contrato de basis-response es q-aware por construcción.

**[OPEN]** No hay todavía una fuente `q ≠ 0` certificada de `D_H` ni de respuesta de base:
`epc_matrix_element_report()` sigue rechazando `q ≠ 0`, ahora indicando que lo que falta es
GO-5 (grafeno K) y no el módulo de convenciones. Este documento es lo que la **revisión
física** debe aprobar antes de producir grafeno K, tal y como pide el criterio de aceptación
de S29.
