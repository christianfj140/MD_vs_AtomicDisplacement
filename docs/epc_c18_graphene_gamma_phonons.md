# C18 / E-F_001-S21 — fonones Γ de grafeno y rango de la supercelda auxiliar

Producción de los modos Γ de grafeno con **tres rangos de supercelda auxiliar**, conservando
por separado los IFC crudos y los corregidos por ASR.

Productores:
[`Comparison/scripts/run_graphene_gamma_phonons.py`](../Comparison/scripts/run_graphene_gamma_phonons.py)
(campaña SIESTA + artifacts) sobre el contrato C17
[`Comparison/scripts/phonon_provider.py`](../Comparison/scripts/phonon_provider.py).
Tests: [`tests/test_graphene_gamma_phonons.py`](../tests/test_graphene_gamma_phonons.py).
Artifacts: `Comparison/results/epc/phonons/graphene/`.

Convenciones de evidencia: **[CODE]** verificado en este repositorio, **[MEAS]** medido en la
campaña, **[DESIGN]** decisión pre-registrada.

---

## 1. Qué se produce

Un run FC de SIESTA por rango, con la misma física (XC, `MeshCutoff`, base PAO, tolerancias
SCF, temperatura electrónica) y el mismo desplazamiento, sobre la celda replicada
`(2n+1)` veces por eje. Sólo se desplazan los **átomos de la celda unidad**
(`FC.First 1` / `FC.Last 2`): el resto de la supercelda está para recibir fuerzas.

Por rango:

| artifact | contenido |
| --- | --- |
| `ranges/<rango>/run/` | el run SIESTA completo (`RUN.fdf`, `RUN.out`, `*.FC`, `*.ORB_INDX`) |
| `ranges/<rango>/range_record.json` | firma de entrada, coste, log de convergencia SCF |
| `ranges/<rango>/gamma_modes_raw.json` | modos Γ del IFC **crudo** |
| `ranges/<rango>/gamma_modes_asr_corrected.json` | modos Γ del IFC **corregido por ASR** |
| `graphene_gamma_phonon_manifest.json` | preflight de backend, provenance, comparación entre rangos |

**[DESIGN]** Crudo y corregido son dos artifacts distintos con firmas distintas, y ambos
llevan el residuo traslacional *crudo*: «los modos acústicos son cero» nunca es una
afirmación que la corrección haya introducido de tapadillo.

**[CODE]** Cada modo se firma como nodo `physical_phonon` del DAG EPC
(`shared/artifact_signature.py`) colgando del nodo `geometry`. Un
`synthetic_test_displacement` no puede ocupar su lugar.

---

## 2. Mapeo de la supercelda

`supercell_geometry()` escribe los átomos **image-major** — el átomo `l*na + kappa` es el
átomo `kappa` de la imagen `l`, con la celda unidad primero — de modo que la lectura del
`.FC` es un `reshape`, no una suposición sobre el orden de `fcbuild`:

```text
Phi[l, kappa, alpha, kappa', beta] = FC[kappa, alpha, l*na + kappa', beta]
```

**[CODE]** El test escribe un `.FC` con constantes de fuerza conocidas y exige la
reconstrucción exacta; y comprueba que el lector de supercelda y el de celda unidad
coinciden bit a bit sobre el mismo fichero real (`images = [[0,0,0]]`).

**[CODE]** Sólo se aceptan repeticiones impares: `2n+1` es la única replicación cuyo conjunto
de imágenes mínimas es simétrico alrededor de la celda desplazada, y una cola de IFC medida
sobre una capa asimétrica no es una cola.

---

## 3. Por qué Γ ya es correcto en 1×1 y aun así el rango importa

`D(Γ) = sum_l Phi_l`, y un run 1×1 ya suma todas las imágenes implícitamente: desplazar un
átomo desplaza a todas sus imágenes periódicas. La supercelda no añade términos nuevos a
Γ; lo que cambia es **numérico**: el muestreo de `k`, la convergencia de la densidad y el
peso de la anarmonicidad del desplazamiento finito. Por eso la convergencia se *mide*.

**[CODE]** El test `test_gamma_only_sums_the_images_which_is_why_ranges_are_comparable`
comprueba la identidad en Γ y que fuera de Γ los dos objetos **no** coinciden — un run 1×1
es un artifact de Γ y de nada más.

---

## 4. Resultado medido

**[MEAS]** Tres rangos, misma física, mismo desplazamiento (0.01 Å), binario C01. Frecuencias
del IFC **corregido**; el residuo y la traslación son del **crudo**.

| rango | átomos | k del run | k efectiva (`repeats × k`) | wall | residuo ASR crudo (eV/Å²) | traslación cruda | `tail_ratio` | alcance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `sc1x1x1` | 2 | 20×20×1 | 20×20×1 | 10.7 s | 2.311e-1 | 72.3 cm⁻¹ | — (sin cola) | 1.47 Å |
| `sc3x3x1` | 18 | 7×7×1 | 21×21×1 | 109.0 s | 4.602e-2 | 15.4 cm⁻¹ | 2.17e-2 | 5.87 Å |
| `sc5x5x1` | 50 | 4×4×1 | 20×20×1 | 371.6 s | 2.224e-1 | 71.0 cm⁻¹ | 1.31e-3 | 10.27 Å |

Modos Γ del IFC corregido, en cm⁻¹:

| rango | acústicas | ZO | E2g (doblete) | separación del doblete |
| --- | --- | --- | --- | --- |
| `sc1x1x1` | \|ω\| ≤ 0.004 | 848.03 | 1562.45 / 1562.46 | 0.004 cm⁻¹ |
| `sc3x3x1` | \|ω\| ≤ 0.06 | 847.99 | 1531.99 / 1532.07 | 0.073 cm⁻¹ |
| `sc5x5x1` | \|ω\| ≤ 0.001 | 847.94 | 1561.60 / 1561.67 | 0.079 cm⁻¹ |

**[MEAS]** La cola del IFC sí converge con el rango: `tail_ratio` cae 2.17e-2 → 1.31e-3 al pasar
de 3×3 a 5×5. La ZO se mueve 0.09 cm⁻¹ en los tres rangos. El E2g, en cambio, **no** es
monótono (1562.5 → 1532.0 → 1561.6): eso no es el alcance del IFC, y la sección 8 lo separa.

---

## 5. Sectores identificados, no impuestos

**[CODE]** `mode_sector_report()` no etiqueta ramas por índice:

- **acústicas**: peso de proyección del autovector sobre el subespacio de traslaciones
  uniformes pesadas por masa (`t_alpha[kappa] = sqrt(M_kappa) e_alpha`);
- **degeneración**: `epc_subspaces.near_degenerate_clusters` con la resolución
  `sqrt(omega^2 + C*dlambda) - omega`, cuyo suelo de ruido es el residuo ASR crudo del
  propio IFC — nunca un cm^-1 elegido a mano;
- **E2g**: el cluster de tamaño 2, no traslacional, en el plano (el plano lo define la red,
  `a1 x a2`) y de mayor frecuencia.

**[CODE]** El test rota el gauge dentro de cada bloque degenerado con una unitaria aleatoria
y exige la misma identificación: lo que sobrevive a una unitaria fue medido, no impuesto.
El peso traslacional por rama es invariante; el reparto dentro/fuera del plano sólo lo es
por cluster, porque mezclar ZA con LA/TA es exactamente lo que hace un gauge.

---

## 6. Backend

**[MEAS]** `backend_preflight()` interroga a `siesta --version` antes de gastar un SCF. Este
build no declara CUDA/OpenACC/ROCm/MAGMA: no existe backend GPU científicamente equivalente
para este SCF, así que la CPU no es el valor por defecto sino el único camino. El manifest
guarda `requested_backend`, `effective_backend`, el banner y el motivo.

---

## 7. Límites explícitos

1. **[DESIGN]** `FC.Save.dHS` va apagado por defecto en esta campaña: las derivadas de
   matriz son el artifact GO-2 del run primitivo (S11) y escribirlas por cada imagen de una
   supercelda cuesta disco que aquí no se usa. `--save-dhs` las reactiva.
2. **[DESIGN]** La capa de análisis del fdf del material (band lines, PDOS, manifolds de
   Wannier, Mulliken) se elimina del input FC: es postproceso que no decide ninguna fuerza y
   cuyos índices de banda absolutos no sobreviven a la replicación. Queda registrado en
   `analysis_layer_removed`.
3. **[MEAS]** Las mallas de `k` de los distintos rangos no son muestreos idénticos
   (`ceil(20/n)`, nunca más gruesa que la primitiva), así que parte de la diferencia entre
   rangos mide eso y no sólo el alcance del IFC. Ambas mallas están en el manifest.
4. **[DESIGN]** El productor **publica** la comparación entre rangos y no la convierte en
   gate. El umbral vive en el gate S22 (sección 8), pre-registrado antes de mirar los
   números y derivado del uso: `g ~ e/sqrt(omega)`.
5. Sólo `q = Γ`. `q != 0` exige la maquinaria de fase de GO-5 y una supercelda conmensurable,
   fuera del alcance de este ticket.

---

## 8. E-F_001-S22 — gate de normalización y ASR

Productor:
[`Comparison/scripts/certify_gamma_normalization_asr.py`](../Comparison/scripts/certify_gamma_normalization_asr.py).
Tests: [`tests/test_certify_gamma_normalization_asr.py`](../tests/test_certify_gamma_normalization_asr.py).
Artifact: `Comparison/results/epc/certification/graphene/c18_gamma_phonon_certification.json`.

No ejecuta SCF ni lee ningún fdf: consume el manifest, los dos artifacts de modos de cada
rango y, cuando el directorio del run sigue ahí, el propio `.FC`, de modo que **recalcula**
cada número publicado en vez de confiar en él.

### 8.1 Tolerancias pre-registradas **[DESIGN]**

| tolerancia | valor | de dónde sale |
| --- | --- | --- |
| `identity_relative` | `64·eps` = 1.42e-14 | identidades algebraicas en float64 |
| `gauge_angle_rad` | `sqrt(2·64·eps)` = 1.69e-7 | un ángulo principal cerca de cero es `sqrt(2(1-sigma))`: el roundoff llega **raíz cuadrada** |
| `codata_constant_relative` | 1e-6 | 5e-7 sobre `omega`, cuatro órdenes bajo el suelo de ruido del IFC |
| `range_frequency_relative` | 2 % | presupuesto único: el fonón aporta <1 % a `|g|`, y `g ~ 1/sqrt(omega)` |
| `range_subspace_angle_rad` | 0.01 | el mismo 1 %, por el lado del autovector |

La misma raíz cuadrada gobierna el suelo del ASR: `omega = sqrt(C·lambda)`, así que un IFC
corregido exactamente aterriza en ~1e-4 cm⁻¹ y no en 1e-12. El gate usa para eso la propia
política de resolución de C17 (`frequency_resolutions_ev`), no un número a mano.

### 8.2 Qué se comprueba, y con qué se rompería **[CODE]**

| check | contenido | lo que atrapa |
| --- | --- | --- |
| `eigenvectors_are_mutually_orthonormal` | `E†E = I`, no sólo `‖e‖ = 1` | dos ramas que abarcan la misma dirección: cada norma sigue valiendo 1 |
| `zero_point_amplitude_carries_hbar_over_2omega` | `sum_kappa M_kappa |u_kappa|² = hbar/(2·omega)` sobre `displacement_pattern()` | cualquier masa mal colocada, y un `1/sqrt(Nc)` escondido en la amplitud (el test lo inyecta y el check lo mide como `1-1/9`) |
| `zero_point_constant_matches_codata` | `hbar²/(amu·eV·Å²)` contra `scipy.constants` | la constante del repositorio, contra una fuente que no es ella misma |
| `gamma_displacement_pattern_is_cell_independent` | `u(R) = u(0)` en Γ | una fase atómica o un `Nc` dependiente de la imagen |
| `frequencies_invariant_under_unit_system` | re-diagonalización en Ry/Bohr² + masas electrónicas con un `hbar²` construido para ese sistema | una conversión de unidades que se cancela a sí misma |
| `subspaces_invariant_under_unit_system` | ángulos principales por cluster | la segunda diagonalización devuelve **otra** base del doblete: eso es libertad, no error, y por eso se compara el span |
| `correction_touches_only_the_onsite_blocks` | `Phi_corr - Phi_raw` fuera de `R = 0` | una corrección ASR que toca vecinos |
| `translations_are_zero_after_correction` | cociente de Rayleigh de las traslaciones uniformes pesadas por masa | el residuo ASR **en cm⁻¹**, que es la unidad que lee un consumidor de `omega` |
| `doublet_basis_change_is_free` | unitaria de Haar y rotación real sobre el doblete | que algo aguas abajo dependa de *qué miembro* es cuál |
| `doublet_metric_detects_a_forbidden_mixing` | rotar el doblete contra una rama externa | que la métrica anterior no sea trivialmente cero siempre |

### 8.3 El confound de la malla `k`, medido **[MEAS]**

Ensanchar la supercelda FC **también** cambia la malla `k` del run, así que un par de rangos
mide el alcance del IFC sólo si su malla efectiva `repeats × k` coincide. El gate agrupa por
esa malla efectiva y evalúa dentro del grupo; lo demás se publica como término de muestreo.

Aquí `sc1x1x1` y `sc5x5x1` comparten 20×20×1 exactamente, y `sc3x3x1` cae en 21×21×1:

- **a malla fija** (1×1 → 5×5, el alcance pasa de 1.47 Å a 10.27 Å): las frecuencias se
  mueven **0.055 %** (0.86 cm⁻¹) y los subespacios elegidos **2.1e-8 rad**. Ambos dentro de
  tolerancia por dos órdenes o más;
- **entre grupos**: 30 cm⁻¹ (≈2 %) sobre el E2g, en los dos sentidos.

**[MEAS]** 21×21 es la única de las tres mallas divisible por 3, es decir la única que
contiene el punto K. **[LIT]** El E2g de Γ en grafeno tiene una anomalía de Kohn
(Piscanec et al. 2004, DOI 10.1103/PhysRevLett.93.185503), cuya cola depende justamente de si
la malla toca K. La excursión de 30 cm⁻¹ es por tanto **muestreo de `k`, no alcance del IFC**;
queda cuantificada y sin corregir, y su convergencia es una campaña propia — un eje que esta
campaña no controló y que el gate se niega a atribuir al rango.

### 8.4 Veredicto **[MEAS]**

`PASS`, sin checks inaplicables, sobre los tres rangos y ambas políticas ASR. Lo que queda
explícitamente abierto es el eje `k`: ningún consumidor puede tomar el E2g de aquí con una
incertidumbre mejor que los ~30 cm⁻¹ del muestreo mientras no exista esa convergencia.
