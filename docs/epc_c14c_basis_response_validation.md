# C14C / E-F_001-S17 — validación de la respuesta de base

Gate independiente del basis-response provider (C14B) antes de que C19 construya
ningún `PAO-covariant response`.

Productores: [`Comparison/scripts/epc_basis_response.py`](../Comparison/scripts/epc_basis_response.py)
(`validate_basis_response`, métricas puras) y
[`Comparison/scripts/certify_basis_response.py`](../Comparison/scripts/certify_basis_response.py)
(corrida sobre la referencia SIESTA certificada por GO-2).
Tests: [`tests/test_epc_basis_response_validation.py`](../tests/test_epc_basis_response_validation.py).
Artifact: `Comparison/results/epc/basis_response/graphene/basis_response_certification.json`.

Convenciones de evidencia: **[CODE]** verificado en este repositorio, **[DERIV]** derivado
del memo [`docs/epc_formalismo_pao_movil.md`](epc_formalismo_pao_movil.md),
**[MEAS]** medido en el run de grafeno, **[DESIGN]** decisión pre-registrada.

---

## 1. Veredicto

```text
verdict = NO_GO      (todos los checks aplicables en verde)
razón   = intra_atomic_basis_response -> unresolved
```

**[MEAS]** Las tres direcciones certificadas de la campaña de grafeno pasan **todos** los
checks aplicables. El `NO_GO` es exclusivamente el término intra-atómico de `S^L`/`S^R`
(supuesto A2 del memo), que `dS/dR` no puede ver y que nadie ha calculado todavía. Es decir:
el provider es **correcto e incompleto**, y no se aplica ningún factor de corrección para
disimular lo segundo. El código de salida del script sigue a los *checks*, no al veredicto.

---

## 2. Qué se valida y contra qué

El formalismo de producción (`moving_pao_projected_dual_basis_v1`) exige la pareja
`(S^L, S^R)` por separado. Tres preguntas, en este orden:

| # | Pregunta | Cómo se responde |
| --- | --- | --- |
| 1 | ¿Reproduce la mejor referencia disponible? | `‖S^L + S^R − D_S^ref‖` contra una medida **independiente** de `D_S` (diferencia central del solape de un par `±` de TSHS), dentro del `tau_FD` que GO-2 midió para ese mismo `D_S`; más la identidad exacta `S^L(k) = S^R(k)†` en Γ y en dos `k` genéricos; más el control acústico de traslación rígida |
| 2 | ¿Cuánto vale lo que cada aproximación tira? | Contracción (F2) del mismo `D_H` y los mismos electrones con la representación real, con `A = 0` y sin respuesta de base; Frobenius **separado en diagonal e interbanda**, y además restringido a las parejas que un fonón puede conectar |
| 3 | ¿Queda algo sin resolver? | `omitted_terms`: A1 (proyección fuera del span) cerrado estructuralmente por el memo §4.3; A2 (término intra-atómico) **abierto** → `NO_GO` |

**[DERIV]** La comprobación 1 acota **solo la parte hermítica**: `A = (S^L − S^R)/2` es
invisible a cualquier referencia `D_S` (memo 4.7). Por eso la identidad y el control de
traslación son checks independientes: un split falso con la suma correcta pasa el criterio 1
y muere en los otros dos ([`test_the_sum_check_alone_does_not_certify_the_split`]).

---

## 3. Tolerancias pre-registradas

**[DESIGN]** Están en `epc_basis_response.TOLERANCES` y **no** son objetivos de exactitud:
cada una vigila una identidad algebraica exacta, así que lo único que deben absorber es el
ruido de ensamblado en float64. El único suelo real es `tau_reference`, que la corrida toma
de la certificación GO-2 del propio `D_S` con el que compara.

| Clave | Valor | Gate |
| --- | --- | --- |
| `sum_relative` | `1e-9` | `‖S^L + S^R − D_S^ref‖ ≤ max(tau_reference, 1e-9·‖D_S^ref‖)` |
| `identity_relative` | `1e-8` | `max_k ‖S^L(k) − S^R(k)†‖ ≤ max(tau_reference, 1e-8·‖S^L‖)` |
| `translation_relative` | `1e-6` | traslación rígida: `‖S^L + S^R‖ ≤ max(tau_reference, 1e-6·‖S^L‖)`, con `‖S^L‖ > 0` |
| `gauge_relative` | `1e-9` | `g(H + cS) = g(H)`; y `g(CU) = U† g U` con tolerancia `1e-9·‖g‖ + spread·‖C†SC‖` |

Un `tolerances=` explícito marca el informe con `tolerances_pre_registered: false`.

**[DESIGN]** La covariancia de gauge es exacta solo en un bloque *exactamente* degenerado.
Un espectro SCF real solo lo es hasta donde su malla rompe la simetría (aquí `1.2e-4 eV` en Γ
y `3.5e-6 eV` en K, **[MEAS]**), así que la tolerancia lleva encima `spread · ‖C†S^{L,R}C‖`,
calculado del propio bloque rotado, y el clustering usa `1e-3 eV`. Un error real de
contracción (p. ej. `eps_row`/`eps_col` intercambiados) está órdenes de magnitud por encima.

---

## 4. Medidas sobre grafeno

Campaña S11 `Comparison/results/epc/siesta_reference/graphene`, GO-2 `PASS`,
runtime `VERIFIED_BINARY_ONLY`. `δ` = **la mayor amplitud que GO-2 resolvió dentro de su
`tau_FD`**, no la del plateau: el plateau es plano en la *señal*, y una diferencia central
colectiva arrastra su término cruzado `O(δ²)` (GO-2 lo certifica con exponente ajustado
2.0004), que a `δ = 0.02 Å` supera el suelo de la referencia. Cargárselo al provider sería
un error de atribución.

### 4.1 Criterio 1 — reproduce la referencia **[MEAS]**

| Dirección | δ (Å) | `‖S^L+S^R − D_S^FD‖` | `tau_FD` | `max_k ‖S^L − S^R†‖` |
| --- | --- | --- | --- | --- |
| `atom0000_x` (one-hot) | 0.02 | `4.25e-13` | `3.81e-4` | `3.37e-13` |
| `random_seed0` (colectiva) | 0.01 | `4.11e-5` | `5.27e-5` | `1.66e-13` |
| `translation_x` (traslación) | 0.02 | `6.46e-14` | `3.81e-4` | `3.41e-13` |

La identidad `S^L = S^R†` se cumple a nivel de máquina en las tres: el split por átomo
desplazado de `dS/dR` es correcto, no una suma partida por la mitad. El residuo `4.11e-5` de
la dirección colectiva es exactamente la discrepancia FC-vs-FD que GO-2 ya había medido a esa
amplitud (truncación de la FD colectiva), y propagado al subespacio electrónico
(`reference_discrepancy_in_subspace`) vale `4.4e-4` – `7.3e-4 eV/Å` frente a `‖g‖ ≈ 26–43`,
es decir `~2e-5` relativo.

**[MEAS]** Control acústico (`translation_x`): `‖S^L + S^R‖ = 6.4e-14` con `‖S^L‖ = 1.42`.
La suma se anula y la respuesta **no**: para una traslación rígida toda la respuesta de base
es antisimétrica. Consecuencia medida sin pedirla: la diagonal de `g` sale `3.5e-11` en Γ y
`2.7e-6` en K sobre `‖g‖ ≈ 22–38`, que es la regla de suma acústica (`dε_n/du = 0`) cayendo
del Hellmann–Feynman generalizado.

### 4.2 Criterio 2 — magnitud de lo omitido por cada aproximación **[MEAS]**

Contribución de la respuesta de base a `g` (= lo que tira `moving_pao_ignore_overlap_v1`),
en `eV/Å` por desplazamiento de Frobenius unitario:

| Dirección | `k` | `‖g‖` | contrib. diagonal | contrib. interbanda | relativa |
| --- | --- | --- | --- | --- | --- |
| `atom0000_x` | Γ | 48.6 | 11.6 | 16.6 | 0.42 |
| `atom0000_x` | K | 75.5 | 12.7 | 36.6 | 0.51 |
| `random_seed0` | Γ | 26.0 | 0.26 | 14.6 | 0.56 |
| `random_seed0` | K | 42.5 | 0.28 | 30.4 | 0.72 |
| `translation_x` | Γ | 21.6 | 5.9e-13 | 21.6 | 1.00 |
| `translation_x` | K | 38.0 | 8.7e-13 | 38.0 | 1.00 |

`moving_pao_ignore_overlap_v1` (control negativo) se equivoca entre el **42 % y el 100 %**
de `‖g‖`. No es una aproximación utilizable: es la demostración numérica de por qué
`C†D_H C` no es `g`.

`moving_pao_symmetric_overlap_v1` (tira `A`):

| Dirección | `k` | error diagonal | error interbanda (bloque) | on-shell (ventana 0.25 eV) |
| --- | --- | --- | --- | --- |
| `atom0000_x` | Γ | `1.1e-19` | 15.3 (67 %) | `2.7e-18` sobre 4 parejas |
| `atom0000_x` | K | `2.1e-15` | 26.9 (37 %) | `3.1e-6` sobre 6 parejas |
| `random_seed0` | Γ | `1.6e-16` | 12.5 (48 %) | `0` |
| `random_seed0` | K | `4.8e-15` | 27.8 (65 %) | `7.6e-7` |
| `translation_x` | Γ | `3.2e-16` | 21.6 (100 %) | `3.8e-18` |
| `translation_x` | K | `2.8e-15` | 38.0 (100 %) | `4.4e-6` |

**[MEAS]** El error diagonal es cero a nivel de máquina en los seis casos: (F3) es exacta en
la diagonal, comprobado sobre datos reales y no solo en el toy. **[DERIV]** El error de
bloque (37–100 %) está dominado por parejas separadas decenas de eV, que ningún fonón
conecta: el término es `(eps_m − eps_n)·A`. Restringido a `|eps_m − eps_n| ≤ 0.25 eV` (la
escala de `hbar·omega` en grafeno) las únicas parejas que quedan en Γ/K son las degeneradas,
donde la forma simétrica también es exacta, y el error cae a `≤ 4.4e-6`.

> **Aviso de alcance.** Esas columnas on-shell son de un único `k` con `C_row = C_col`. El
> EPC real conecta `k` con `k+q` y sus autovalores **no** coinciden, así que la medida
> on-shell definitiva del término `A` es de C20, con el modo `E_2g` real. Lo que C14C
> establece es el método y la cota de bloque, no un permiso para llamar despreciable a `A`.

### 4.3 Criterio 3 — términos sin resolver

```text
A1  out_of_span_projection        -> closed_non_blocking   (memo 4.3; lo cierra la diagonal HF)
A2  intra_atomic_basis_response   -> unresolved            -> NO_GO
```

**[DERIV]** Los bloques intra-atómicos de `S^L`/`S^R` son idénticamente invisibles en
`dS/dR` (memo §6): `∂S/∂R ≡ 0` ahí, pero `S^L = −S^R ≠ 0`. Lo cierra la integral monocéntrica
analítica sobre los radiales del `.ion.xml`, que **no existe todavía** en este repositorio.

**[DESIGN]** Mientras A2 esté abierto, `validate_basis_response` devuelve `NO_GO` aunque
todos los checks pasen, y no se deriva ningún factor de corrección. Lo único que se publica
es `intra_atomic_sensitivity`: cuánto se mueve `g` **por unidad** de ese término
(anti-hermítico, Frobenius unitario, sobre los pares de orbitales del mismo átomo).
**[MEAS]** En grafeno hay 12 pares de ese tipo y la sensibilidad interbanda es
`max = 24.6 eV/Å` en Γ y `35.3 eV/Å` en K por unidad de `1/Å` (rms `19.8` y `30.6`). Con
`‖g‖ ≈ 26–75`, un `A` intra-atómico de orden `0.1 1/Å` movería el bloque interbanda un ~10 %:
**no es despreciable por decreto**, y por eso la integral monocéntrica es trabajo obligatorio
y no una nota al pie. La sensibilidad es una derivada respecto a algo no calculado; el módulo
no la convierte en corrección ([`test_intra_atomic_sensitivity_is_a_coefficient_not_a_value`]).

`PASS` exigiría simultáneamente: todos los checks aplicables en verde, comparación contra
referencia efectivamente evaluada, `formalism_id` no degradado y ningún término
`unresolved`. Un artifact `D_S`-only nunca puede pasar: le falta `A` por representación.

---

## 5. Cómo se reproduce

```bash
python3 Comparison/scripts/certify_basis_response.py \
    --reference-root Comparison/results/epc/siesta_reference/graphene \
    --go2-report     Comparison/results/epc/certification/graphene/go2_certification.json
```

**[CODE]** No ejecuta SCF ni lee ningún `fdf`: consume los artifacts de S11 y **se niega a
arrancar** si el informe GO-2 no dice `PASS` — una respuesta de base validada contra una
`dS/dR` no certificada no certifica nada. Por dirección:

- `S^L`, `S^R` desde el `*.dHSdR.nc` en el δ descrito en §4;
- referencia `D_S`: diferencia central del **solape** del par `±` de TSHS. El solape no
  necesita la corrección de `E_F` que domina el gate de `D_H`, así que este lado está libre
  de esa convención;
- electrones: `H` absoluto (deshecho el shift de `E_F` de sisl, que es el origen de energías
  que `dHSdR.nc` deriva) y `S` del run de equilibrio, resueltos como pencil generalizado en
  Γ y en K = `(2/3, 1/3, 0)` (`C†SC − I` ≈ `3e-12`, residuo del pencil `≤ 3e-11`, **[MEAS]**).
  K entra porque es donde está la degeneración de Dirac que el check de gauge necesita; en
  otra red la etiqueta sería incorrecta y el check se declararía inaplicable, nunca cierto.

El test `test_graphene_reference_passes_every_check_and_is_NO_GO_on_the_intra_atomic_term`
ejecuta esto end-to-end cuando los artifacts y `sisl`/`netCDF4` están presentes, y se salta
limpiamente si no.

---

## 6. Qué habilita y qué no

- **Habilita** C19 en cuanto A2 se cierre: el contrato de consumo, las tolerancias y las
  métricas ya están fijados y probados sobre datos reales.
- **No habilita** construir `PAO-covariant response` hoy: `validate_basis_response` marca `NO_GO` y
  `epc_matrix_elements` sigue exigiendo un `BasisResponse` explícito.
- **Trabajo que abre**: la integral monocéntrica `⟨∂_α φ_μ|φ_ν⟩` sobre los radiales del
  `.ion.xml` (fuente 1 del memo §6, la misma que da la ruta analítica escalable a MATBG).
