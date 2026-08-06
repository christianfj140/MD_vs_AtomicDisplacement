# Referencia tight-binding para TBG puro

Implementación: [`Comparison/scripts/run_tbg_tight_binding.py`](../Comparison/scripts/run_tbg_tight_binding.py)

Resultados locales: `Comparison/results/tbg_tight_binding/`

## Alcance científico

Esta es una referencia atomística empírica independiente para la geometría rígida de TBG
usada por Graph2Mat. No es un Hamiltoniano SIESTA/DFT ni un ground truth. Una discrepancia
TB–Graph2Mat no determina por sí sola qué modelo es correcto.

El modelo es la parametrización de un orbital `pz` de Moon y Koshino, Phys. Rev. B 87,
205404 (2013):

- un orbital `pz` por carbono;
- base ortogonal, `S = I`;
- degeneración de spin 2, sin duplicar la matriz;
- geometría del FDF sin regenerar ni relajar;
- sin SOC, Hartree, magnetismo ni términos dependientes del entorno.

Fuente primaria: <https://arxiv.org/pdf/1302.5218>.

## Parámetros congelados

```text
h(d) = V_pppi(d) [1 - (dz/d)^2] + V_ppsigma(d) (dz/d)^2
V_pppi(d)    = -2.7 exp[-(d-a0)/delta0] eV
V_ppsigma(d) = +0.48 exp[-(d-d0)/delta0] eV

a_ref  = 2.46 Å
a0     = a_ref/sqrt(3)
d0     = 3.35 Å
delta0 = 0.184 a_ref
cutoff = 4 a0 = 5.6811266488 Å, duro
```

El primer vecino de la geometría del repositorio, `d_CC = 1.4318287 Å`, produce
aproximadamente `-2.632 eV`. No se reescala el modelo literal para forzar `-2.7 eV`.

La variante con `a_ref = 2.48 Å` se calcula sólo como sensibilidad y se etiqueta
`moon_koshino_geometry_scaled_diagnostic`. Nunca se escoge por parecido con Graph2Mat.

### ¿Está esta geometría en el ángulo mágico de esta parametrización?

Sí, y se **mide**, no se cita. El preflight publica `moire_fermi_velocity`: la velocidad
de Dirac moiré en K, obtenida del splitting del manifold central a lo largo de K→Γ y
extrapolada a `dk → 0`, dividida por la velocidad de la monocapa del mismo modelo.

```text
hbar*v_F  monocapa   = 5.1515 eV·Å
hbar*v*   moiré      = 0.0627 eV·Å
v*/v_F               = 0.012
```

La velocidad está suprimida un factor ~80. Esa colapso *es* la definición operativa del
ángulo mágico. Una desviación del ~10% en el ángulo deja `v*/v_F` en el rango de decenas
de por ciento, así que el cociente discrimina de forma tajante sin apelar a ningún ángulo
publicado, y sin depender de qué parametrización use cada artículo.

Esto sustituye a la advertencia genérica de "el ángulo mágico depende de la
parametrización" por una cantidad verificable en cada ejecución.

## Geometría y camino k

Entrada:

```text
materials/twisted_bilayer_graphene_1p084549deg/RUN.fdf
```

Contrato validado:

- 11 164 carbonos;
- 5 582 por capa;
- separación 3.35 Å;
- supercelda rígida `(31,30)` de `1.084549049°`.

Con la base recíproca de la supercelda:

```text
K  = (1/3, 2/3)
K' = (2/3, 1/3)
Γ  = (0, 0)
M  = (1/2, 1/2)
```

El camino de comparación es `K–Γ–M–K`, con los mismos 31 puntos que el resultado
Graph2Mat existente.

## Selección electrónica

Hay 11 164 orbitales spinless y 5 582 bandas ocupadas a half filling. El manifold central
se fija antes de mirar la dispersión:

```text
absolute_band_index = {5580, 5581, 5582, 5583}
```

No se seleccionan las ramas más planas. Las líneas TB de la UI se agrupan por
`absolute_band_index`; `solver_band_index` conserva sólo el rango local de ARPACK.

## Solver e inercia

Se usa `scipy.sparse.linalg.eigsh` en shift-invert. Una factorización SuperLU de
`H-sigma I` sirve como `OPinv` y para contar los signos de `diag(U)`.

Esta última parte tiene una cautela explícita: SuperLU es un solver LU general. El conteo
sólo se acepta cuando:

- se usa `SymmetricMode`, `MMD_AT_PLUS_A`, `diag_pivot_thresh=0` y sin equilibrado;
- las permutaciones de filas y columnas son idénticas;
- el margen **por pivote** `|Im d_i| / |Re d_i|` es menor de `1e-3`, es decir, el signo de
  cada pivote —lo único que la inercia lee— está determinado con holgura;
- los tests pequeños coinciden exactamente con diagonalización densa;
- los mismos índices dan los mismos autovalores con `sigma ± 50 meV`.

El criterio es por pivote a propósito. Una razón global contra el pivote mayor mezcla
escalas separadas por nueve órdenes de magnitud y produce falsos positivos; ver la sección
de auditoría.

Por eso el manifest lo llama
`empirically_validated_not_sparse_ldlh_proof`, no una demostración LDLH general. La
cautela sigue la documentación de SuperLU sobre symmetric mode y pivoting:
<https://portal.nersc.gov/project/sparse/superlu/faq.html>.

## Seguridad y reanudación

- BLAS/OpenMP se limita a 4 hilos antes de importar NumPy/SciPy;
- warning térmico a 80 °C y parada preventiva a 82 °C;
- guardrail de disco a 12 % libre, por encima del límite absoluto solicitado del 10 %;
- comprobación de recursos entre puntos k;
- escrituras JSON/NPZ atómicas;
- neutralidad y bandas guardan progreso por punto k;
- las etapas completas se reutilizan cuando coincide su contrato de inputs.

No se guardan Hamiltonianos, LU ni eigenvectores completos.

## Validación ejecutada

### Tests pequeños

- signos de hoppings intraplano/vertical;
- hermiticidad;
- `E(k)=E(-k)`;
- AA y equivalencia AB/BA;
- sparse frente a dense;
- inercia frente a dense para varios puntos y shifts;
- invariancia ante traslación y permutación atómica;
- DOS limitada a su ventana realmente cubierta.

### Preflight de 11 164 átomos

Ejecución reforzada con 4 hilos:

| Comprobación | Resultado |
|---|---:|
| Estado | passed |
| Hermiticidad máxima | `1.06e-13 eV` |
| K frente a K' | `1.20e-12 eV` |
| E(K) frente a E(-K) | `1.40e-12 eV` |
| Residuo absoluto máximo | `7.61e-10 eV` |
| Error máximo al cambiar sigma ±50 meV | `1.81e-12 eV` |
| Cutoff 4a0/5a0, RMS tras un único shift | `0.191 meV` |
| Cutoff 4a0/5a0, error máximo alineado | `0.389 meV` |
| Cambio de anchura muestreada | `-0.301 meV` |
| Temperatura CPU máxima | `70.05 °C` |
| Mínimo disco libre | `16.676 %` |

La sensibilidad `a_ref=2.48 Å` cambia el ancho muestreado en `-3.742 meV`, con RMS
alineado `2.527 meV`. Se conserva como diagnóstico, no como baseline.

## Neutralidad y DOS

La malla existente es Gamma-centrada `12×12×1`, contiene K/K' y usa 64 estados por k.

```text
E_F(T=0) = 0.7674368882 eV en el gauge TB
E_F(0.5 meV) = 0.7674081736 eV
E_F(1.0 meV) = 0.7674314047 eV
```

La variación máxima es `0.0287 meV`; la referencia de neutralidad es estable dentro de
1 meV.

La DOS es parcial. La caché cubre ambos lados de neutralidad hasta `326.419 meV`; con un
margen de cinco veces el broadening de 2 meV se publica únicamente `±316.419 meV`. No se
dibujan como ceros físicos los bordes truncados.

La malla TB es `12×12`; el DOS Graph2Mat disponible usa `16×16`. Su comparación actual es
cualitativa, no una convergencia cuantitativa en la misma malla.

## Observables TB actuales

### Camino K–Γ–M–K, 31 puntos

| Observable | Valor |
|---|---:|
| Ancho del manifold sobre el camino | `36.645 meV` |
| Solape indirecto de neutralidad sobre el camino | `0.0875 meV` |
| Solape con remota de valencia | `2.988 meV` |
| Solape con remota de conducción | `1.002 meV` |
| Splitting de las cuatro centrales en K | `0.0202 meV` |

### Malla 12×12×1

| Observable | Valor |
|---|---:|
| Ancho global muestreado del manifold | `36.645 meV` |
| Solape indirecto de neutralidad | `0.1413 meV` |
| Gap directo mínimo | `6.28e-9 meV`, esencialmente cero |
| Solape con remota de valencia | `2.953 meV` |
| Solape con remota de conducción | `1.107 meV` |

Son valores de una única malla 12×12; no se afirma convergencia 12×12/18×18.

## Comparación permitida con Graph2Mat

Graph2Mat usa cuatro orbitales por C, overlap SIESTA y un problema generalizado. No se
comparan elementos de matriz ni gauges absolutos.

Los artefactos existentes del camino Graph2Mat no guardan `absolute_band_index` por k. La
regla visual por aislamiento falla en 2 de los 31 puntos cuando se contrasta contra los
índices TB. Por ello:

- se permite superponer los espectros, cada uno recentrado por su neutralidad;
- se permiten splittings etiquetados en K, Γ y M;
- no se publican como rigurosos anchos o gaps globales Graph2Mat;
- no se concluye todavía que Graph2Mat reproduzca cuantitativamente el manifold TB.

Recuperar índices Graph2Mat exigiría una ejecución solver-only con inercia en los 31 puntos.
No forma parte de esta implementación y no se reentrena ningún modelo.

## UI

- Graph2Mat: azul continuo;
- TB: rojo discontinuo;
- segmentos lineales y marcadores en los autovalores;
- TB nunca pasa por el matching heurístico de Graph2Mat;
- la UI distingue observables del camino y de la malla;
- aviso visible: referencia empírica rígida, no DFT/SIESTA.

## Comandos

```bash
# Tests pequeños
.venv/bin/python Comparison/scripts/run_tbg_tight_binding.py --self-test
.venv/bin/pytest -q tests/test_tbg_tight_binding.py

# Reutiliza etapas cuyo contrato coincide
.venv/bin/python Comparison/scripts/run_tbg_tight_binding.py --all

# Recalcula explícitamente sólo una etapa
.venv/bin/python Comparison/scripts/run_tbg_tight_binding.py --preflight --force
```

No deben usarse `--force` ni `--all` durante una campaña ajena sin revisar primero disco y
temperatura. El propio script vuelve a comprobar ambos recursos antes de cada solve.

## Auditoría externa 2026-08-05: evaluación y respuesta

Una auditoría independiente reportó 5 hallazgos P1, 9 P2 y 3 P3. Evaluados uno a uno
contra el código: **14 de 15 son correctos y reproducibles; uno es factualmente falso**.
La auditoría reprodujo de forma independiente los hoppings, `ħv_F`, la geometría, la
hermiticidad y la inercia contra 28 509 factorizaciones adversariales — el trabajo de
verificación es sólido. El fallo está en su hallazgo de cabecera.

### P1-01 — REFUTADO

Afirma que la parametrización `-2.7/0.48` tiene su primer ángulo mágico en ~1.21°, no en
1.0845°, y que por tanto la anchura de 36.6 meV "puede deberse a estar fuera del ángulo
mágico". La cita es vaga ("Long et al.") y la afirmación es contrastable sin ella.

Medición directa: `v*/v_F = 0.012`. La velocidad de Fermi moiré está suprimida un factor
~80, que es la firma del ángulo mágico. A 1.21° con esta parametrización el manifold no
estaría colapsado. **La premisa del hallazgo es falsa** y su conclusión —que la anchura
del manifold pueda ser un artefacto de estar fuera del régimen mágico— no se sostiene.

En vez de reetiquetar por precaución se añadió el observable medido al preflight, que es
lo que la auditoría debió pedir.

### P1-05 — correcto pero no es un defecto

Que el gate de Graph2Mat (3.89 meV) supere a varios observables TB (0.02–3 meV) es cierto
y ya estaba documentado. No es un fallo del código. Implementado como aviso cuantificado
en la UI: compara el gate contra la escala de cada observable y los clasifica en
comparables y no comparables, en vez de una advertencia genérica.

### Confirmados y corregidos

| ID | Estado | Corrección |
|---|---|---|
| P1-02 | Confirmado empíricamente: hash de contrato idéntico con `E_F` 0.767 y 9.0 eV | El contrato DOS ata `fermi_level_eV`, `neutrality_sha256` y `mesh_cache_sha256` |
| P1-03 | Confirmado: `mesh_eigenvalues.npz` sólo tenía `eigenvalues` y `first_indices` | Firma dentro del NPZ, verificada al cargar; `ALGORITHM_VERSION` en todo contrato |
| P1-04 | Confirmado: las cuatro etapas resellaadas con el SHA actual | `aggregate` es read-only; el resumen separa `source_implementation` de `aggregation_implementation` |
| P2-01 | Confirmado y **peor de lo reportado** | Ver abajo |
| P2-02 | Confirmado: signo sobre un gap de 5.5e-12 eV con residuo ~1e-9 eV | Tolerancia de signo = 10× el residuo; los gaps por debajo se listan aparte |
| P2-03 | Confirmado | Se valida finitud, rangos e índices antes de aceptar el progreso |
| P2-04 | Confirmado: etiqueta `single_12x12` fija | Se construye desde `mesh`; el test que consolidaba el bug, corregido |
| P2-05 | Confirmado: `Number(null) === 0` → "N0 · s0" | Descarta null antes de convertir |
| P2-06 | Confirmado, pero **la primera corrección causó una regresión** | Ver abajo |
| P2-07 | Confirmado: NaN en hover para TB | Hover específico por método y recuento real de estados en ventana |
| P2-08 | Confirmado | Traza única, `ħv_F` medido del modelo, oculta por defecto, alcance ≤1 eV |
| P2-09 | Confirmado: 73.05 °C del proceso de agregación | Recursos por etapa; sólo se publica máximo global si todas registraron |
| P3-01 | Confirmado | `E_F` es el punto medio entre niveles adyacentes; convención explícita |
| P3-02 | Confirmado | Rejilla k completa con `null` en los huecos |
| P3-03 | Confirmado | `try/finally`, `allow_nan=False`, `fsync` del directorio |

**P2-01 es más grave de lo que reportó.** La auditoría observó "hasta cuatro enlaces
perdidos y 0.595 meV". En una celda con `reach = 1` —el de la celda moiré— desplazar un
átomo por `-a2` pierde **56 enlaces y mueve el espectro 1408 meV**. Corregido plegando
las coordenadas in-plane a la celda antes de replicar. Sobre la geometría objetivo el
pliegue mueve 3.2e-14 Å y deja los 679 208 enlaces intactos: **los resultados publicados
no cambian**.

### Hallazgos no reportados por la auditoría

**1. El contrato de neutralidad tampoco ataba `sigma`**, que procede de preflight. Es el
mismo fallo de P1-02 en otra etapa: cambiar preflight dejaba una neutralidad obsoleta
como reutilizable. La auditoría aisló correctamente el caso de la DOS pero no buscó la
clase de bug en el resto del grafo de etapas. Corregido y con test de regresión.

**2. El guard de certificación de inercia abortaba la producción.** Este es el más grave,
y se le escapó por una razón que la propia auditoría declara: *"No ejecuté una nueva
producción TB del sistema de 11 164 átomos"*. Los artefactos publicados se calcularon
**antes** de que existiera el guard (el NPZ es de las 11:51 y los JSON de las 12:40), así
que el guard nunca había sobrevivido a una ejecución completa. Al regenerar, aborta.

El criterio era:

```text
max|Im d_i| / max|d_i|  >  1e-7   ->  rechazar
```

Compara la contaminación imaginaria contra el pivote **más grande**, no contra el pivote
cuyo signo se está leyendo. En este sistema `max|d| = 4.5e4` mientras que el `|Re d|` más
pequeño es `2.0e-4`: la razón mezcla escalas separadas por nueve órdenes de magnitud y no
dice nada sobre si un signo concreto es determinado.

Verificación sobre los 144 puntos de la malla:

| Cantidad | Valor |
|---|---:|
| Permutación simétrica | **144/144** |
| `max|Im d|/max|d|` peor caso | `5.97e-7` |
| Margen por pivote `max|Im d_i|/|Re d_i|` | `1.49e-6` |
| `|Re d|` más pequeño de toda la malla | `2.01e-4` |
| Puntos rechazados por el criterio antiguo | 2 (k=14, k=142) |

Ambos rechazados se contrastaron contra **diagonalización densa** (54 s cada uno):
SuperLU daba 5580 estados por debajo de σ y el denso da 5580. **Falsos positivos.**

El criterio pasa a ser el margen por pivote `|Im d_i| / |Re d_i| < 1e-3`, que certifica
individualmente el signo de cada pivote —la cantidad que la inercia realmente lee— en vez
de una razón global. En aritmética exacta `D` es real, así que `|Im d_i|` mide el redondeo
local sobre ese mismo número. El peor caso observado deja seis órdenes de margen, y una
degeneración real de LDL^H empuja esa razón hacia 1. Se conserva intacto el requisito
estructural de permutación simétrica, que es el que exige la FAQ de SuperLU.

### P2-06: la primera corrección ocultó medio espectro

Aplicar "número de bandas visibles" al modo directo seleccionando por `|E - E_F|` medio
**borró todo el lado positivo de Graph2Mat**. Con el valor por defecto de 6:

```text
sin límite:  12 trazas,  -47.9 .. +48.9 meV
límite 6:     6 trazas,  -47.9 ..  +3.2 meV   <- desaparecen +8 .. +49 meV
```

La causa es que en Graph2Mat `band_index` no es una banda física, sino el **rango
energético dentro de la ventana shift-invert** en cada k. Los rangos bajos descienden
hasta −47.9 meV, así que su `|E|` medio (13–15 meV) gana a los rangos altos (22–28 meV) y
el recorte se lleva un lado entero.

Corregido en dos partes:

1. El recorte es **simétrico**: mitad de las bandas por debajo de neutralidad y mitad por
   encima, según la energía media con signo, rellenando desde el otro lado si falta.
2. El control gana la opción **"todas las de la ventana"**, que pasa a ser la de por
   defecto. El gráfico científico principal no oculta nada salvo que se pida.

Verificado sobre datos reales para Graph2Mat y tight binding, en ventanas de ±25/±50/±75
meV y límites 0/4/6/8: ambos lados del espectro presentes en las 24 combinaciones. Con
test de regresión que falla si alguna deja de tener bandas de un signo.

Lección: la auditoría clasificó P2-06 como cosmético ("saturación visual y control
incumplido") y ofrecía como alternativa renombrar el control. La alternativa era la opción
segura; aplicar el control sin pensar en la simetría convirtió un nit de UI en pérdida de
datos en la figura principal.
