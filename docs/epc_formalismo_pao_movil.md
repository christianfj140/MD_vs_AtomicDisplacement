# Formalismo EPC en base PAO móvil (C03b / E-F_001-S5)

Memo de derivación para **GO-1**. Cierra la pregunta previa a cualquier
implementación:

> ¿Qué respuesta covariante PAO debe contraerse entre estados de `H C = eps S C`
> cuando la base PAO no ortogonal se mueve con los átomos, y contienen nuestros
> artifacts `H`, `S`, `dHSdR.nc` y Graph2Mat exactamente la información necesaria?

Estado: **GO-1 evaluable**. La derivación es reproducible, coincide con dos
fuentes primarias independientes y su álgebra está ejecutada en
[`tests/test_epc_moving_pao_formalism.py`](../tests/test_epc_moving_pao_formalism.py)
(9 tests, numpy puro, sin módulo de producción: C04 todavía no existe y GO-1 no
debe presuponerlo).

Convenciones de evidencia: **[CODE]** verificado en este repositorio,
**[LIT]** verificado en la fuente primaria citada (texto/ecuación extraídos, no
recordados), **[DERIV]** derivado aquí y comprobado numéricamente,
**[DESIGN]** decisión de este repositorio, **[OPEN]** pendiente con test asignado.

---

## 1. Veredicto en una página

1. `D_H` **no es** el acoplamiento. La contracción `C_m† D_H C_n` ni siquiera es
   invariante frente a un cambio del cero de energía `H -> H + cS`, que es una
   convención, no física ([test
   `test_energy_origin_shift_leaves_g_invariant_but_moves_raw_D_H`]).
2. La respuesta covariante PAO es la derivada representada dentro del span
   de la base:

   ```math
   \boxed{\;\Delta^{\rm PAO\text{-}cov}[u] \;=\; D_H[u] \;-\; S^{L}[u]\,S^{-1}H \;-\; H\,S^{-1}S^{R}[u]\;}
   \tag{F1}
   ```

   con `S^L[u]_{μν} = ⟨∂_u φ_μ|φ_ν⟩` y `S^R[u]_{μν} = ⟨φ_μ|∂_u φ_ν⟩`.
3. Contraída, y **sin invertir nunca `S`**:

   ```math
   \boxed{\;g_{mn} \;=\; \big(C_m^\dagger D_H C_n\big) \;-\; \epsilon_n \big(C_m^\dagger S^{L} C_n\big) \;-\; \epsilon_m \big(C_m^\dagger S^{R} C_n\big)\;}
   \tag{F2}
   ```
4. La corrección simétrica que el roadmap se negaba a adoptar por intuición
   **es una truncación**, no la respuesta:

   ```math
   g_{mn} = \underbrace{C_m^\dagger\Big[D_H - \tfrac{\epsilon_m+\epsilon_n}{2}D_S\Big]C_n}_{\text{forma simétrica, sólo }D_S} \;+\; (\epsilon_m-\epsilon_n)\,\big(C_m^\dagger A\, C_n\big),
   \qquad A \equiv \tfrac{1}{2}\big(S^{L}-S^{R}\big).
   \tag{F3}
   ```

   Es exacta en la diagonal y dentro de bloques degenerados; el error es
   `O(eps_m - eps_n)`, es decir `O(hbar*omega)` on-shell.
5. Por tanto el basis-response requerido es la **pareja `(S^L, S^R)`**
   (`representation = "S_L_S_R"`), no `D_S`. `D_S = S^L + S^R` solo fija la parte
   hermítica; `A` es información independiente
   ([test `test_D_H_and_D_S_alone_do_not_determine_g`] construye dos sistemas con
   `H`, `S`, `D_H`, `D_S` idénticos y `g` distinto).
6. Consecuencia dura para C14B: **`dHSdR.nc` no basta**. Da la parte
   inter-atómica de `A`, pero la parte intra-atómica (mismo átomo, todas sus
   imágenes) es idénticamente invisible en `dS/dR` y exige una integral
   monocéntrica analítica sobre los radiales PAO (§6).
7. No queda ningún término Pulay/incompletitud escondido: el único supuesto es
   la resolución de la identidad restringida al span, y ése es exactamente el
   supuesto que hace que la diagonal de (F2) reproduzca `d(eps_n)` variacional
   (§4.5). El caso 3 de la §XI del roadmap se cierra en **no bloqueante**.

---

## 2. Convenciones canónicas del repositorio

Todo adapter externo convierte a estas; nada se corrige *ad hoc* aguas abajo.

| Objeto | Convención canónica | Origen |
| --- | --- | --- |
| Bloques real-space | `H_{μν}(R_l) = ⟨φ_μ(0)|Ĥ|φ_ν(R_l)⟩`, `R_l` vector de red entero | [CODE] mapping ORB_INDX/imagen ya usado por el exporter |
| Fase de Bloch | `H(k) = Σ_l e^{+2πi k·R_l} H(R_l)`, `k` fraccional, **gauge de celda** (sin `τ_κ` en la fase) | [CODE] `generate_siesta_overlap_only.py:307`, `evaluate_deeph_kpoint_metrics.py:69`, `run_tbg_tight_binding.py:328` |
| Normalización electrónica | `C(k)† S(k) C(k) = I` | [CODE] `run_deeph_sparse_spectrum.py` (residual y `C†SC` ya se validan) |
| Unidades canónicas | energía `eV`, longitud `Å`, `D_H` en `eV/Å`, `D_S`,`S^L`,`S^R` en `1/Å`, `g` en `eV` | [CODE] Graph2Mat predice `eV` desde `Å` (docstring de `graph2mat_autograd_derivatives.py`) |
| Fase atómica fonónica | **fuera** del eigenvector: `u_{lκ} = e_κ(q) e^{iq·R_l}`, con `e^{iq·τ_κ}` declarado aparte por el `PhononProvider` | [DESIGN], ver §8 y [LIT] Croy §2 (phonopy incluye la fase en `ẽ_s(q) = e_s(q) e^{iq·R_s}`) |
| `1/√N_c` | **no** entra en `g`; aparece sólo como `1/N_q` en observables integrados | [DESIGN], ver §8 |

Índices: `μ,ν` orbitales; `κ` átomo de la celda primitiva; `l` celda; `m,n`
bandas; `u` un desplazamiento colectivo `[N,3]` (Γ) o un patrón `(q, e)` (§8).

---

## 3. Qué es `D_H` y qué es `D_S`, backend por backend

Definiciones (derivada direccional a lo largo de `u`, base **co-móvil**: los
PAO viajan con sus centros, que es lo que hace todo backend de este repo):

```math
D_H[u]_{\mu\nu} \equiv \frac{d}{dt}\Big\langle \phi_\mu(R+tu)\Big|\hat H(R+tu)\Big|\phi_\nu(R+tu)\Big\rangle_{t=0}
= \underbrace{\langle \partial_u\phi_\mu|\hat H|\phi_\nu\rangle}_{H^L}
+ \underbrace{\langle \phi_\mu|\partial_u\hat H|\phi_\nu\rangle}_{\Delta^{\rm phys}}
+ \underbrace{\langle \phi_\mu|\hat H|\partial_u\phi_\nu\rangle}_{H^R}
```

```math
D_S[u] = S^{L}[u] + S^{R}[u], \qquad S^{L} = (S^{R})^{\dagger}\ \text{(identidad exacta, no aproximación).}
```

| Backend | Qué produce realmente | Contiene respuesta SCF | Resolución por átomo |
| --- | --- | --- | --- |
| SIESTA `FC.Save.dHS` -> `*.dHSdR.nc` | `dH/dR` y `dS/dR` por átomo desplazado y eje, sólo celda unidad | Sí: son derivadas del `H` convergido | Sí (`/DISPLACEMENTS/<atom>/<axis>/`) [CODE] `shared/run_inventory.py:288-321` |
| SIESTA FD explícito (C08/C09) | lo mismo por diferencias centrales | Sí | Sí |
| Graph2Mat JVP | `D_H[u]` únicamente (el checkpoint predice `H`, no `S`) | Implícita en el modelo | No: la primitiva devuelve ya la contracción con `u` |
| Graph2Mat frozen | `D_H[u]` por diferencias centrales del propio modelo | Implícita | No |

Consecuencia inmediata: cualquier `g` de este repositorio con `D_H` de
Graph2Mat es **`epc_backend_class = hybrid`**, porque la respuesta de base viene
de otro backend. Ya está previsto en `shared/artifact_signature.py`
(`pao_covariant_response.electronic_derivative_backend` /
`basis_response_backend`).

---

## 4. Derivación

### 4.1 El problema

`|ψ_n⟩ = Σ_μ C_{μn}|φ_μ(R)⟩`. Al mover los átomos cambian *dos* cosas: el
operador `Ĥ` y los propios vectores de base. `D_H` mezcla ambas. El acoplamiento
electrón-fonón es el elemento de matriz de la perturbación **del operador**
entre estados fijos, y esa separación es la totalidad del problema.

### 4.2 Conexión de la base móvil

Descomposición exacta de la derivada de un PAO (ningún supuesto todavía):

```math
|\partial_u \phi_\mu\rangle = \sum_\nu |\phi_\nu\rangle\,\Gamma_{\nu\mu} + |\chi^{\perp}_\mu\rangle,
\qquad \Gamma \equiv S^{-1}S^{R},\qquad \langle\phi_\lambda|\chi^\perp_\mu\rangle = 0 .
```

De aquí, exactamente, `S^R = SΓ` y `S^L = Γ†S`, y por tanto
`D_S = SΓ + Γ†S = 2·Herm(SΓ)`: **`D_S` sólo determina la parte hermítica de
`SΓ`.** Ésta es la raíz de todo lo que sigue.

### 4.3 El único supuesto: proyección sobre el span

```math
\sum_{\mu\nu}|\phi_\mu\rangle (S^{-1})_{\mu\nu}\langle\phi_\nu| \;=\; \hat P \;\approx\; 1
\tag{A1}
```

Con (A1), `H^L = S^L S^{-1}H` y `H^R = H S^{-1} S^R`, y despejando de la
descomposición de `D_H` sale (F1). Sin (A1) el residuo es exactamente

```math
\Pi = \langle\partial_u\phi|\,(1-\hat P)\,\hat H\,|\phi\rangle + \text{h.c.}
```

y nada más: no hay un segundo término Pulay oculto
([test `test_incomplete_basis_residual_is_exactly_the_out_of_span_term`]
construye la base incompleta y verifica el residuo elemento a elemento).
Con base completa (F1) es **exacta**
([test `test_reconstruction_is_exact_for_a_complete_basis`]).

### 4.4 Contracción sin invertir `S`

Usando `H C_n = S C_n eps_n` y `C_m† H = eps_m C_m† S`, los dos `S^{-1}` de (F1)
se cancelan y queda (F2). Esto importa para MATBG: la contracción PAO nunca
necesita factorizar `S` de 44.656 orbitales.

### 4.5 Por qué (F2) y no `⟨ψ_m|∂Ĥ|ψ_n⟩` "exacto"

Porque el sistema electrónico de este repositorio **es** el modelo PAO. La
diagonal de (F2) es

```math
g_{nn} = C_n^\dagger\big(D_H - \epsilon_n D_S\big)C_n = \frac{d\epsilon_n}{du},
```

el teorema de Hellmann–Feynman generalizado, exacto también en base incompleta
([test `test_generalized_hellmann_feynman_diagonal`], comprobado contra
diferencias finitas de los autovalores del pencil). El objeto no proyectado
`⟨ψ_n|∂Ĥ|ψ_n⟩` **no** reproduce `d(eps_n)` con base incompleta: sería
inconsistente con el propio espectro que el solver publica. La proyección no es
una aproximación cómoda; es lo que hace al operador consistente con el modelo.

### 4.6 La forma simétrica y su gauge

Escribiendo `S^L = D_S/2 + A`, `S^R = D_S/2 - A` sale (F3) inmediatamente
([test `test_symmetric_form_differs_only_by_the_antisymmetric_overlap_response`]).
Interpretación: pasar a un marco ortonormal `X(R)` con `X†SX = I` obliga a
`Y + Y† = -X†D_S X` para `Y = X^{-1}∂X`, y deja libre la parte anti-hermítica de
`Y` (una rotación del marco). La forma simétrica es la elección
`Y` hermítica ("sin rotación"); (F2) es la que fija esa libertad con la
**física de la base real**, es decir con `A`.

Notas:

- La forma simétrica **no** es "Löwdin": `Y` de `X = S^{-1/2}` sólo es hermítica
  si `[S, D_S] = 0`. Llamarla Löwdin sería incorrecto.
- La ambigüedad afecta sólo a `eps_m != eps_n`. Diagonal, bloques degenerados y
  el límite adiabático son insensibles a ella.
- On-shell, `eps_m - eps_n = ∓ hbar*omega_{qν}`, así que el término es
  proporcional a la frecuencia del fonón: **pequeño, pero no nulo**. Estimación
  de orden (a medir en C14C, no a citar como resultado): con
  `|A| ~ 1 Å^-1`, `hbar*omega ~ 0.2 eV` y amplitud de punto cero `~0.03 Å`, la
  contribución a `g` es de orden de unos meV frente a `|g| ~ 0.1–0.3 eV` en
  grafeno, es decir el nivel del propio `tau_model`. No es despreciable por
  decreto.

### 4.7 `D_S` no determina `g`

[test `test_D_H_and_D_S_alone_do_not_determine_g`] construye explícitamente dos
modelos con `H`, `S`, `D_H` y `D_S` idénticos a `1e-8` y `g` que difiere en más
del 1 % de su norma, con la diferencia **exactamente** en los elementos
`eps_m != eps_n` y **cero** en la diagonal. La receta: rotar la base dentro del
span con `S X = iK` anti-hermítica (deja `D_S` intacto) y absorber el cambio de
`D_H` en `∂Ĥ`, que no es observable desde datos matriciales. Esto responde
literalmente al requisito del roadmap ("demostrar que `dS` por sí solo no
determina dos matrices independientes").

---

## 5. `formalism_id`

```text
formalism_id = "moving_pao_projected_dual_basis_v1"
```

| Campo del contrato | Valor |
| --- | --- |
| `representation` de `D_H` | derivada direccional de la matriz `H` en base PAO **co-móvil**, `eV/Å` |
| `representation` de `D_S` | `S^L + S^R`, `1/Å`; **no suficiente por sí sola** |
| `basis_response.representation` | `S_L_S_R` (valor ya admitido por `shared/artifact_signature.py`) |
| Construcción de `PAO-covariant response` | (F1); contracción operativa (F2) |
| Gauge electrónico | covariante: `C -> C U` en un bloque degenerado da `g -> U† g U`; los invariantes publicables son valores singulares / normas de bloque, nunca elementos indexados por banda ([test `test_g_block_is_covariant_under_degenerate_subspace_rotation`]) |
| Gauge de marco de base | **no queda libre**: lo fija `A`. No se elige `Y` anti-hermítica por conveniencia |
| Origen de energías | `g` invariante bajo `H -> H + cS` con `c` constante; si `c` depende de `R` (deriva de `E_F` entre configuraciones desplazadas) el término inducido es `(∂c) S`, que contrae a `(∂c) δ_mn`: **sólo diagonal** (§9) |
| Unidades | `g` en `eV` tras multiplicar por la amplitud de punto cero en `Å` (§8) |
| Base fija | prohibida como interpretación por defecto: nuestros backends son co-móviles |

Variantes **degradadas**, admitidas sólo como diagnóstico y obligadas a
etiquetarse como tales (nunca como production truth):

| id | Qué asume | Válido en |
| --- | --- | --- |
| `moving_pao_symmetric_overlap_v1` | `A = 0` (sólo `D_S`) | diagonal, bloques degenerados, límite adiabático |
| `moving_pao_ignore_overlap_v1` | `D_S = 0`, `g = C†D_H C` | nada; ni siquiera es invariante ante el cero de energía. Sólo como control negativo |
| `fixed_pao_basis_v1` | la base no se mueve | ningún artifact de este repo la produce |

---

## 6. Qué debe entregar el basis-response provider (C14B)

Requisito: `S^L[u]` y `S^R[u]` **por separado**, en la misma convención de
bloques/imágenes que `H` y `S`. Reconstruirlos desde su suma está prohibido.

Estructura del objeto, que decide qué fuente sirve:

- **Bloques inter-atómicos** (`atom(μ) != atom(ν)`): `S_{μν}` depende de las
  posiciones sólo a través de `R_l + τ_{atom(ν)} - τ_{atom(μ)}`, así que la
  derivada respecto al átomo desplazado `I` selecciona `S^L` (si `I = atom(μ)`) o
  `S^R` (si `I = atom(ν)`). **El split es recuperable de `dS/dR` resuelto por
  átomo**, que es exactamente lo que `dHSdR.nc` indexa.
- **Bloques intra-atómicos** (`atom(μ) = atom(ν)`, incluidas sus imágenes): el
  bloque es invariante bajo el desplazamiento del átomo, luego
  `∂S/∂R ≡ 0` y `S^L = -S^R`, pero **`S^L` no es cero**: transportar rígidamente
  un orbital `s` lo expresa con componentes `p` del mismo centro. `A` intra-atómico
  es real y **completamente invisible en `dS/dR`**
  ([test `test_intra_atomic_antisymmetric_response_is_invisible_to_D_S`]).
  Es una integral **monocéntrica** `⟨∂_α φ_μ|φ_ν⟩` sobre los radiales del
  `.ion.xml`, no nula sólo para `l' = l ± 1` (en C: `2s`–`2p`, y `2p`–`d` si hay
  polarización). Barata y analítica.

Orden de investigación recomendado (§XI del roadmap, ya decidido por esta
derivación):

1. **Derivada analítica del solape PAO** (dos centros + un centro). Única ruta
   que da `S^L`/`S^R` de forma nativa, sin SCF y escalable a MATBG. Candidata de
   producción.
2. `dHSdR.nc` para la parte inter-atómica + integral monocéntrica para la
   intra-atómica. Ruta de referencia en grafeno.
3. FD frozen de `S`: da sólo `D_S`. **Insuficiente**; sólo alimenta la variante
   degradada.

El artifact debe declarar `representation`, unidades, dirección/`q`, y las
firmas de geometría y base, y debe declarar explícitamente si el término
intra-atómico está incluido o puesto a cero (bandera obligatoria, no silencio).

---

## 7. Correspondencia término a término con los artifacts

| Término de (F1)/(F2) | Artifact | Estado |
| --- | --- | --- |
| `D_H[u]`, referencia | `*.dHSdR.nc` `/DISPLACEMENTS/<atom>/<axis>/`, `Ry/Bohr` -> `eV/Å` | [CODE] esperado desde el source 5.4.2; **no observado**. GO-2 lo certifica contra FD explícito |
| `D_H[u]`, modelo | JVP direccional Graph2Mat (C10/C11), `eV/Å` | [CODE] existe la maquinaria JVP; falta la primitiva direccional |
| `D_S[u]` | mismo `.nc`, `1/Bohr` -> `1/Å` | idem; ojo al `DSDR_ORBITAL_FILTER` |
| `S^L`,`S^R` inter-atómicos | split por átomo desplazado del `dS/dR` anterior | [DERIV] §6; C06 debe confirmar que el filtro conserva ambos triángulos o que la hermiticidad los reconstruye — **[OPEN-S2]** |
| `S^L`,`S^R` intra-atómicos | integral monocéntrica sobre radiales `.ion.xml` | **artifact nuevo**, no existe hoy |
| `H`, `S`, `C`, `eps` | H de Graph2Mat / SIESTA + `S` overlap-only exacto + solver generalizado | [CODE] existentes; C15 debe persistir `C` |
| `eps_m`, `eps_n` | autovalores del mismo pencil, misma ventana | [CODE] |
| Contrato orbital | `orbital_contract_hash` (C03) | [CODE] `shared/orbital_contract.py` |

Nota Graph2Mat [CODE]: las salidas viven en el frame `e3nn` y las derivadas
respecto a posiciones deben contraerse con `change_of_basis`. Esa matriz es
**independiente de `R`** (permutación/signos), luego es un cambio de base `B`
constante y no introduce conexión adicional: `D_H` transforma como `B†D_H B`.
Un `B` que dependiera de `R` sí desplazaría el gauge y rompería (F2) — por eso
el contrato prohíbe recombinaciones de base dependientes de la geometría.

---

## 8. `q != 0`, fonones y normalización

Patrón de desplazamiento canónico y su derivada:

```math
u_{l\kappa} = e_\kappa(q\nu)\,e^{iq\cdot R_l},
\qquad
D_H^{(q)}[e]_{\mu\nu}(R_l) = \sum_{l'\kappa} \frac{\partial H_{\mu\nu}(R_l)}{\partial R_{l'\kappa}}\, e_\kappa\, e^{iq\cdot R_{l'}} .
```

Por invariancia traslacional la perturbación tiene carácter Bloch `q`, conecta
`k` con `k+q`, y en el gauge de celda la suma de Bloch se hace **en `k`** (el `k`
del estado inicial), sin `1/√N_c` residual [DERIV]:

```math
g^{\nu}_{mn}(k,q) = \ell_{q\nu}\Big[ C_m(k{+}q)^\dagger D_H^{(q)}(k) C_n(k)
- \epsilon_n(k)\, C_m^\dagger S^{L,(q)}(k) C_n
- \epsilon_m(k{+}q)\, C_m^\dagger S^{R,(q)}(k) C_n \Big],
```

```math
\ell_{q\nu}\,e_\kappa \;\rightarrow\; \sqrt{\frac{\hbar}{2 M_\kappa \omega_{q\nu}}}\;e_\kappa(q\nu),
\qquad
\frac{\hbar^2}{1\,\text{amu}\cdot 1\,\text{eV}} = 4.1802\times10^{-3}\,\text{Å}^2
```

([test `test_zero_point_amplitude_constant`]; para C con `hbar*omega = 0.196 eV`
da `0.0298 Å`). Con `Σ_κ |e_κ|² = 1` y `e` autovector de la matriz dinámica
mass-weighted, el factor de masa va **dentro** de `ℓ`, átomo por átomo.

Consecuencias de diseño:

- **Γ es el caso `q = 0`** de lo anterior: el JVP actual de Graph2Mat, que
  desplaza las `N` posiciones de la celda con `shifts` fijos, es exactamente
  una perturbación periódica `q = 0` (§II del roadmap). No puede reutilizarse
  para `q != 0` (GO-8b).
- El basis response debe ser **igual de q-aware**: `S^{L,(q)}`, `S^{R,(q)}` con
  la misma fase `e^{iq·R_{l'}}` y el mismo gauge de celda. Un `D_H(q)` correcto
  con un overlap tratado como Γ no pasa GO-5.
- `1/√N_c` no aparece en `g` con esta convención; entra como `1/N_q` al integrar
  observables. Cualquier provider que lo lleve dentro del eigenvector debe
  declararlo y convertirse en el boundary, con round-trip test.

---

## 9. Nivel de Fermi y ocupaciones

[LIT] Frederiksen et al. (2007) Ec. (17) desplaza cada Hamiltoniano desplazado
según `H(Q) := H(Q) - [eps_F(Q) - eps_F^0] S(Q)` para compensar la deriva
artificial de `E_F` en supercelda pequeña. En este formalismo eso se entiende
exactamente:

- `c` constante: `g` es **idénticamente invariante**, sin elegir gauge, porque
  `D_S = S^L + S^R` (verificado en test).
- `c` dependiente de `R`: aparece `(∂c) S`, que contrae a `(∂c) δ_mn`.

Por tanto **la deriva de `mu` afecta sólo a los elementos diagonales**. El
contrato de la §XII del roadmap se concreta así: `pao_projected_mode_coupling`
(`m != n`) es independiente de la convención del cero de energía;
`eigenvalue_directional_derivative` y `deformation_potential_relative_to_mu`
(diagonales) sólo están definidos junto a una convención declarada de `mu`, y
`d(mu)` **no** se resta de todo `g`.

---

## 10. Supuestos que quedan abiertos, y quién los cierra

| id | Supuesto | Cómo se cierra |
| --- | --- | --- |
| **A1** | Resolución de la identidad restringida al span (§4.3) | Estructural: es la definición del modelo PAO. Su consistencia se verifica vía HF diagonal en C09/C20 |
| **A2** | Tratamiento del término intra-atómico de `S^L`,`S^R` | C14B lo implementa analíticamente y C14C mide su contribución; prohibido ponerlo a cero en silencio |
| **A3** | `dHSdR.nc` contiene la derivada del `H` **convergido** (EPC apantallado), no un potencial congelado | GO-2 discrimina: el FD explícito de `H` convergido es apantallado por construcción; si el `.nc` fuese frozen-potential, C09 falla |
| **A4** | Unidades `Ry/Bohr` y `1/Bohr` pese a los atributos NetCDF incorrectos | GO-2 (`GO2_VERIFY_DH_DS_UNITS_NUMERICALLY`), nunca por lectura de atributos |
| **[OPEN-S2]** | El filtro de `dS` sobre orbitales del átomo desplazado conserva ambos triángulos del split | C06, inspección de schema |
| **A5** | El `change_of_basis` de Graph2Mat es independiente de `R` | Test de C10/C11; si dependiera de la geometría, (F2) requeriría un término extra |
| **A6** | Transferibilidad del checkpoint a derivadas | Empírico, C14; ajeno al formalismo |
| **A7** | Runtime SIESTA `VERIFIED_BINARY_ONLY` (C01) | La limitación de provenance viaja con todo artifact derivado; obliga a la validación FD exhaustiva de GO-2 |

---

## 11. Bibliografía primaria y cómo se verificó

Las dos referencias centrales se descargaron y se extrajo su texto; las
ecuaciones citadas están leídas, no recordadas.

1. **Frederiksen, Paulsson, Brandbyge, Jauho, PRB 75, 205413 (2007)**,
   DOI `10.1103/PhysRevB.75.205413` (preprint `arXiv:cond-mat/0611562`).
   Referencia primaria para EPC con base atómica en SIESTA. Sus Ecs. (14)–(16)
   son literalmente (F1):

   > (14) `⟨i|∂Ĥ_e/∂Q_{Iν}|j⟩ = ∂⟨i|Ĥ_e|j⟩/∂Q_{Iν} - ⟨i'|Ĥ_e|j⟩ - ⟨i|Ĥ_e|j'⟩`
   > (15) `Σ_ij |i⟩(S^{-1})_{ij}⟨j| = 1`
   > (16) `= ∂H_ij/∂Q - Σ_kl ⟨i'|k⟩(S^{-1})_{kl}⟨l|Ĥ_e|j⟩ - Σ_kl ⟨i|Ĥ_e|k⟩(S^{-1})_{kl}⟨l|j'⟩`

   con `|i'⟩ ≡ ∂|i⟩/∂Q_{Iν}`. Obtienen `⟨i'|k⟩` y `⟨l|j'⟩` **por separado** por
   diferencias finitas, y su Ec. (17) es el ajuste de `E_F` de la §9.
2. **Croy, Unsal, Biele, Pecchia, J. Comput. Electron. 22, 1231 (2023)**,
   DOI `10.1007/s10825-023-02033-9` (DFTBephy). Derivan lo mismo por la base
   dual/biortogonal `⟨β_μ|φ_ν⟩ = δ_μν`; su Ec. (11) es la identidad
   `⟨φ|∂Ĥ|φ'⟩ = ∂⟨φ|Ĥ|φ'⟩ - ⟨∂φ|Ĥ|φ'⟩ - ⟨φ|Ĥ|∂φ'⟩` y el resultado final
   contiene `∂S/∂R_s S^{-1} H` y `H S^{-1} ∂S/∂R_r̄` con las derivadas asignadas
   al átomo del índice correspondiente, es decir `S^L` y `S^R` separados.
   Advierten en las conclusiones: *"depending on the choice of the basis set,
   i.e., centered at equilibrium positions or co-moving with the displaced atoms,
   one obtains slightly different expressions. The impact of this choice will be
   investigated elsewhere."* Aquí esa diferencia queda cuantificada como el
   término `(eps_m - eps_n) A` de (F3).
3. **Soler et al. (2002)**, DOI `10.1088/0953-8984/14/11/302` y **García et al.
   (2020)**, DOI `10.1063/5.0005077`: base metodológica PAO de SIESTA
   (soporte finito, orbitales centrados en los átomos), que es lo que hace que
   la base sea móvil y que el término intra-atómico de §6 exista.
4. **Manual/source SIESTA 5.4.2** para `FC.Save.dHS`, layout y tolerancias:
   registrado como `source_candidate_expectations` en
   `shared/run_inventory.py`, **nunca como observación** sobre el binario local
   (C01 = `VERIFIED_BINARY_ONLY`).
5. **Piscanec et al. (2004)**, DOI `10.1103/PhysRevLett.93.185503`: benchmark
   físico Γ/K en grafeno para la validación de F6/F7. No se usa para calibrar
   nada del formalismo.

Dos derivaciones independientes (Frederiksen 2007 vía inserción de la identidad;
Croy 2023 vía base dual) y la derivación por conexión de §4.2 dan la misma
expresión (F1). Ninguna de las tres soporta `g = C†D_H C` ni la forma simétrica
como resultado exacto.

---

## 12. Qué habilita este memo

- **C04** puede escribir `epc_formalism.py` con `formalism_id` fijado, la
  transformación raw -> físico de (F1)/(F2) y las variantes degradadas
  etiquetadas.
- **C14B** ya sabe qué debe producir: `S^L`, `S^R`, con el término
  intra-atómico explícito y una bandera obligatoria si se omite.
- **C19** puede rechazar por contrato cualquier `PAO-covariant response` construido sin el
  basis-response que `formalism_id` exige.
- **GO-5** hereda el requisito de que el basis response sea q-aware con la misma
  convención de fase.

Lo que este memo **no** decide: la magnitud real del término `A` en grafeno
(C14C), la transferibilidad del checkpoint (C14) y la ruta `q != 0` productiva
Q-B vs Q-C (posterior a grafeno K).
