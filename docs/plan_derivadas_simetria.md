# Plan de implementación detallado para explotar simetría en stencils de derivadas del Hamiltoniano de SIESTA

## 0. Objetivo general

El objetivo es reducir el número de estructuras desplazadas y, por tanto, el número de ejecuciones SIESTA/ML necesarias para calcular derivadas del Hamiltoniano mediante diferencias finitas.

Actualmente, según el contexto del repositorio, se calcula:

[
D_{i\alpha}H
============

\frac{\partial H}{\partial R_{i\alpha}}
\approx
\frac{
H(\mathbf R+\delta \mathbf e_{i\alpha})
---------------------------------------

H(\mathbf R-\delta \mathbf e_{i\alpha})
}{2\delta}
]

para cada átomo (i), cada eje cartesiano (\alpha \in {x,y,z}), cada signo (\pm\delta), y cada valor de (\delta).

Por tanto, para diferencias centrales:

[
N_{\text{runs}}
===============

# 2 \times 3 \times N_{\text{átomos}}

6N
]

por cada valor de (\delta).

La idea de la optimización es explotar que, si dos grados de libertad están relacionados por simetría, sus derivadas del Hamiltoniano no son independientes. En el caso ideal, el coste podría bajar de:

[
6N
]

a:

[
6N_{\text{inequiv}}
]

donde (N_{\text{inequiv}}) es el número de átomos inequivalentes por simetría.

En casos de alta simetría, la ganancia puede ser enorme. Por ejemplo, si una supercelda tiene 64 átomos pero solo 2 átomos inequivalentes:

[
6 \times 64 = 384
]

pasaría a:

[
6 \times 2 = 12
]

ejecuciones SIESTA por valor de (\delta), es decir, una reducción ideal de:

[
\frac{384}{12}=32
]

veces.

---

# 1. Principio físico-matemático

## 1.1. Derivada actual

El código actual calcula derivadas respecto a un único grado de libertad atómico:

[
D_{i\alpha}H
============

\frac{\partial H}{\partial R_{i\alpha}}
]

donde:

* (i) es el índice del átomo;
* (\alpha) es el eje cartesiano;
* (H) es el Hamiltoniano completo generado por SIESTA;
* el resto de átomos se mantiene fijo.

Con diferencias centrales:

[
D_{i\alpha}H
\approx
\frac{
H_{i\alpha}^{+}
---------------

H_{i\alpha}^{-}
}{2\delta}
]

donde:

[
H_{i\alpha}^{+}
===============

H(\mathbf R+\delta\mathbf e_{i\alpha})
]

y:

[
H_{i\alpha}^{-}
===============

H(\mathbf R-\delta\mathbf e_{i\alpha})
]

---

## 1.2. Operaciones de simetría

Una operación de simetría cristalina se puede escribir como:

[
g = (W,\mathbf w)
]

en coordenadas fraccionarias, o como:

[
g = (Q,\mathbf t)
]

en coordenadas cartesianas.

Aquí:

* (W) es la parte rotacional en coordenadas fraccionarias;
* (\mathbf w) es la traslación en coordenadas fraccionarias;
* (Q) es la parte rotacional en coordenadas cartesianas;
* (\mathbf t) es la traslación cartesiana.

`spglib` devuelve operaciones de simetría como pares ((W,\mathbf w)), almacenando rotaciones y traslaciones con el mismo índice, y también proporciona información de átomos equivalentes mediante `equivalent_atoms`.

Si una operación (g) transforma el átomo (i) en el átomo (j), escribimos:

[
g(i)=j
]

Entonces, desplazar el átomo (i) en la dirección (\alpha) se transforma en desplazar el átomo (j) en la dirección:

[
Q\mathbf e_\alpha
]

---

## 1.3. Transformación del Hamiltoniano

El Hamiltoniano no es un escalar. Es una matriz escrita en una base orbital.

Por tanto, bajo una operación de simetría:

[
H(g\mathbf R)
=============

U_g H(\mathbf R) U_g^\dagger
]

donde (U_g) es la representación de la operación de simetría en la base orbital.

Para derivadas:

[
\frac{\partial H}{\partial R_{j\beta}}
======================================

\sum_{\alpha}
Q_{\beta\alpha}
,
U_g
\frac{\partial H}{\partial R_{i\alpha}}
U_g^\dagger
]

Esta es la ecuación central de toda la implementación.

---

## 1.4. Por qué esto es más difícil que con fuerzas

Para fuerzas o constantes de fuerza, la simetría actúa sobre vectores/tensores cartesianos.

Para el Hamiltoniano, la simetría actúa sobre:

1. átomos;
2. orbitales;
3. imágenes periódicas;
4. posibles bloques de espín;
5. ejes cartesianos;
6. bloques dispersos del formato HSX/TSHS.

SIESTA usa orbitales atómicos pseudoatómicos de soporte finito, pseudopotenciales norm-conserving y una malla real para representar densidad, potenciales y elementos de matriz.

Por eso no basta con hacer:

```python
dH_target = dH_source[permuted_indices, permuted_indices]
```

Eso solo sería correcto en casos muy simples, por ejemplo:

* traslaciones puras;
* bases puramente (s);
* ausencia de espín/SOC;
* ausencia de rotaciones que mezclen orbitales;
* formato de matriz sin offsets periódicos complicados.

---

# 2. Estado actual del repositorio según el contexto dado

## 2.1. Archivos relevantes

Según el análisis del agente de programación, los archivos principales son:

```text
Comparison/scripts/build_hamiltonian_derivative_stencils.py
Comparison/scripts/hamiltonian_derivative_stencil.py
Comparison/scripts/run_hamiltonian_derivative_siesta_references.py
Comparison/scripts/run_hamiltonian_derivative_predictions.py
Comparison/scripts/graph2mat_autograd_derivatives.py
docs/workflows.md
```

---

## 2.2. Qué ya hace bien el repo

El repositorio ya implementa correctamente el esquema físico básico:

```text
un cálculo = un átomo desplazado = un eje desplazado = un signo de delta
```

Es decir:

[
\mathbf R_{\text{perturbed}}
============================

\mathbf R_{\text{base}}
]

salvo:

[
R_{i\alpha}
\rightarrow
R_{i\alpha} \pm \delta
]

Además, en cada cálculo se extrae el Hamiltoniano completo, no solo el bloque asociado al átomo desplazado.

Por tanto, al desplazar un único átomo en un único eje, se obtiene una derivada completa:

[
D_{i\alpha}H
]

sobre toda la matriz Hamiltoniana.

Esto ya es eficiente desde el punto de vista de “no desplazar todos los átomos a la vez”. Lo que falta es reducir redundancias por simetría cristalina.

---

## 2.3. Qué no está implementado

Según el agente:

* no hay uso de grupo espacial;
* no hay `spglib`;
* no hay análisis de Wyckoff;
* no hay átomos equivalentes;
* no hay reducción de órbitas;
* no hay deduplicación de grados de libertad;
* no hay reconstrucción del Hamiltoniano por simetría.

El código actual hace fuerza bruta sobre todos los átomos y ejes pasados por `--atoms` y `--axes`.

---

# 3. Estrategia general de implementación

La implementación debe hacerse de forma incremental.

No conviene empezar directamente por la reconstrucción completa del Hamiltoniano, porque esa parte requiere transformar correctamente la base orbital.

La estrategia recomendada es:

```text
Fase 0: Preparación y aislamiento de responsabilidades
Fase 1: Detección y reporte de simetría
Fase 2: Reducción de átomos irreducibles sin reconstrucción completa
Fase 3: Manifest enriquecido con metadatos de simetría
Fase 4: Reconstrucción por simetría para casos seguros
Fase 5: Validación contra cálculo completo
Fase 6: Soporte avanzado de orbitales
Fase 7: Soporte de offsets periódicos HSX/TSHS
Fase 8: Integración completa en workflows SIESTA/ML
Fase 9: Tests, documentación y criterios de activación
```

---

# 4. Diseño de alto nivel

## 4.1. Nueva arquitectura propuesta

Añadiría dos módulos nuevos:

```text
Comparison/scripts/symmetry_utils.py
Comparison/scripts/hamiltonian_symmetry.py
```

Opcionalmente, si se quiere mantener más limpio:

```text
Comparison/scripts/symmetry/
    __init__.py
    geometry.py
    operations.py
    irreducible.py
    manifest.py
    orbital_transforms.py
    hamiltonian_reconstruction.py
    validation.py
```

---

## 4.2. Separación de responsabilidades

### `symmetry_utils.py`

Responsable de simetría geométrica:

* leer estructura;
* convertir posiciones cartesianas/fraccionarias;
* llamar a `spglib`;
* obtener grupo espacial;
* obtener operaciones;
* construir mapas átomo-a-átomo;
* construir órbitas de átomos equivalentes;
* decidir representantes;
* construir órbitas de grados de libertad si procede.

### `hamiltonian_symmetry.py`

Responsable de simetría del Hamiltoniano:

* construir (U_g);
* transformar derivadas;
* transformar Hamiltonianos;
* reconstruir derivadas faltantes;
* comparar derivadas reconstruidas contra explícitas;
* manejar matrices dispersas;
* manejar offsets periódicos;
* detectar casos no soportados.

---

# 5. Fase 0: Preparación

## 5.1. Objetivo

Antes de tocar la lógica del stencil, preparar el código para que la simetría se pueda añadir sin romper el workflow existente.

---

## 5.2. Principio fundamental

El comportamiento actual debe seguir siendo el default.

Es decir:

```bash
python build_hamiltonian_derivative_stencils.py ...
```

sin flags de simetría debe producir exactamente el mismo manifest, las mismas estructuras y los mismos resultados que antes.

La simetría debe ser opt-in:

```bash
--symmetry-report
--use-symmetry
```

---

## 5.3. Flags iniciales recomendados

Añadir al parser de `build_hamiltonian_derivative_stencils.py`:

```python
parser.add_argument(
    "--symmetry-report",
    action="store_true",
    help="Only analyze and report crystal symmetry. Do not generate stencils.",
)

parser.add_argument(
    "--use-symmetry",
    action="store_true",
    help="Enable symmetry-aware stencil generation.",
)

parser.add_argument(
    "--symmetry-mode",
    choices=["off", "report", "atoms", "dofs", "reconstruct"],
    default="off",
    help=(
        "Symmetry mode: off = current brute-force behavior; "
        "report = only report symmetry; "
        "atoms = generate only inequivalent atoms; "
        "dofs = generate only inequivalent atom-axis displacements when safe; "
        "reconstruct = reconstruct missing derivatives by symmetry."
    ),
)

parser.add_argument(
    "--symprec",
    type=float,
    default=1e-3,
    help="Distance tolerance for symmetry detection, in Angstrom-like units.",
)

parser.add_argument(
    "--angle-tolerance",
    type=float,
    default=-1.0,
    help="Angle tolerance for symmetry detection. Use -1 for spglib default.",
)

parser.add_argument(
    "--symmetry-strict",
    action="store_true",
    help="Fail if requested symmetry mode cannot be applied safely.",
)

parser.add_argument(
    "--symmetry-allow-unsafe",
    action="store_true",
    help="Allow experimental symmetry reconstruction paths. Not recommended for production.",
)
```

---

## 5.4. Modo legacy garantizado

Añadir tests de regresión para garantizar que:

```bash
--symmetry-mode off
```

produce el mismo número de stencils que antes:

[
N_{\text{structures}}
=====================

N_{\delta}
\times
N_{\text{atoms}}
\times
N_{\text{axes}}
\times
N_{\text{signs}}
+
N_{\text{base}}
]

---

# 6. Fase 1: Detección y reporte de simetría

## 6.1. Objetivo

Implementar un modo que solo diga:

1. qué simetría tiene la estructura;
2. cuántos átomos son inequivalentes;
3. qué ahorro potencial habría;
4. qué operaciones están disponibles;
5. si la simetría parece segura o sospechosa.

No debe generar stencils ni ejecutar SIESTA.

---

## 6.2. Dependencia recomendada

Usar `spglib`.

`spglib` se instala mediante `pip install spglib` o con `conda install -c conda-forge spglib`, según su documentación de Python.

---

## 6.3. Entrada esperada para `spglib`

La celda se debe proporcionar como:

```python
cell = (lattice, scaled_positions, atomic_numbers)
```

donde:

* `lattice` es la matriz de red;
* `scaled_positions` son posiciones fraccionarias;
* `atomic_numbers` son números atómicos.

---

## 6.4. Función principal

Crear en `symmetry_utils.py`:

```python
from dataclasses import dataclass
import numpy as np
import spglib


@dataclass(frozen=True)
class SymmetryOperation:
    op_id: int
    rotation_frac: np.ndarray
    translation_frac: np.ndarray
    rotation_cart: np.ndarray
    atom_map: tuple[int, ...]


@dataclass(frozen=True)
class SymmetryInfo:
    enabled: bool
    symprec: float
    angle_tolerance: float
    spacegroup_number: int | None
    international_symbol: str | None
    rotations_frac: np.ndarray
    translations_frac: np.ndarray
    equivalent_atoms: np.ndarray
    representative_atoms: tuple[int, ...]
    orbits: dict[int, tuple[int, ...]]
    operations: tuple[SymmetryOperation, ...]
```

---

## 6.5. Conversión de posiciones

Si el código base usa posiciones cartesianas en Å:

[
\mathbf r = A\mathbf f
]

donde (A) contiene los vectores de red como columnas, entonces:

[
\mathbf f = A^{-1}\mathbf r
]

Código:

```python
def cart_to_frac(cart_positions: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    """
    Convert Cartesian positions to fractional coordinates.

    Convention:
        r_cart = A @ f_frac

    lattice is expected to contain lattice vectors as rows or columns depending
    on the repo convention. This function must be adapted and unit-tested.
    """
    A = lattice.T
    frac = np.linalg.solve(A, cart_positions.T).T
    return frac % 1.0
```

Hay que confirmar la convención de `lattice` en el repo. Esta parte debe testearse con una celda cúbica.

---

## 6.6. Detección con `spglib`

```python
def detect_symmetry(
    lattice: np.ndarray,
    frac_positions: np.ndarray,
    atomic_numbers: np.ndarray,
    symprec: float = 1e-3,
    angle_tolerance: float = -1.0,
) -> SymmetryInfo:
    cell = (lattice, frac_positions, atomic_numbers)

    dataset = spglib.get_symmetry_dataset(
        cell,
        symprec=symprec,
        angle_tolerance=angle_tolerance,
    )

    if dataset is None:
        return SymmetryInfo(
            enabled=False,
            symprec=symprec,
            angle_tolerance=angle_tolerance,
            spacegroup_number=None,
            international_symbol=None,
            rotations_frac=np.empty((0, 3, 3), dtype=int),
            translations_frac=np.empty((0, 3), dtype=float),
            equivalent_atoms=np.arange(len(frac_positions)),
            representative_atoms=tuple(range(len(frac_positions))),
            orbits={i: (i,) for i in range(len(frac_positions))},
            operations=tuple(),
        )

    rotations = np.asarray(dataset["rotations"], dtype=int)
    translations = np.asarray(dataset["translations"], dtype=float)
    equivalent_atoms = np.asarray(dataset["equivalent_atoms"], dtype=int)

    reps = tuple(sorted(set(equivalent_atoms.tolist())))
    orbits = {
        int(rep): tuple(np.where(equivalent_atoms == rep)[0].tolist())
        for rep in reps
    }

    operations = []
    for op_id, (W, w) in enumerate(zip(rotations, translations)):
        Q = frac_rotation_to_cartesian(W, lattice)
        atom_map = build_atom_map(
            frac_positions=frac_positions,
            atomic_numbers=atomic_numbers,
            W=W,
            w=w,
            tol=symprec,
        )
        operations.append(
            SymmetryOperation(
                op_id=op_id,
                rotation_frac=W,
                translation_frac=w,
                rotation_cart=Q,
                atom_map=tuple(atom_map),
            )
        )

    return SymmetryInfo(
        enabled=True,
        symprec=symprec,
        angle_tolerance=angle_tolerance,
        spacegroup_number=int(dataset["number"])
            if "number" in dataset else int(dataset["spacegroup_number"]),
        international_symbol=str(dataset["international"])
            if "international" in dataset else str(dataset["international_symbol"]),
        rotations_frac=rotations,
        translations_frac=translations,
        equivalent_atoms=equivalent_atoms,
        representative_atoms=reps,
        orbits=orbits,
        operations=tuple(operations),
    )
```

Nota: según la versión de `spglib`, algunos nombres de campos pueden variar entre `number`/`spacegroup_number` e `international`/`international_symbol`. Conviene encapsular ese acceso en una función auxiliar.

---

## 6.7. Construcción del mapa atómico

```python
def build_atom_map(
    frac_positions: np.ndarray,
    atomic_numbers: np.ndarray,
    W: np.ndarray,
    w: np.ndarray,
    tol: float,
) -> list[int]:
    n_atoms = len(frac_positions)
    atom_map = [-1] * n_atoms

    for i in range(n_atoms):
        f_new = W @ frac_positions[i] + w
        f_new = f_new % 1.0

        candidates = []

        for j in range(n_atoms):
            if atomic_numbers[j] != atomic_numbers[i]:
                continue

            diff = f_new - frac_positions[j]
            diff -= np.round(diff)

            if np.linalg.norm(diff) < tol:
                candidates.append(j)

        if len(candidates) != 1:
            raise RuntimeError(
                f"Could not uniquely map atom {i} under symmetry operation. "
                f"Candidates: {candidates}"
            )

        atom_map[i] = candidates[0]

    return atom_map
```

---

## 6.8. Conversión de rotación fraccionaria a cartesiana

Si:

[
\mathbf r = A \mathbf f
]

entonces:

[
Q = A W A^{-1}
]

Código:

```python
def frac_rotation_to_cartesian(W: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    """
    Convert a fractional-coordinate rotation W to Cartesian coordinates.

    Convention:
        r_cart = A @ f_frac

    If the repo stores lattice vectors as rows, use A = lattice.T.
    If it stores them as columns, use A = lattice.
    """
    A = lattice.T
    return A @ W @ np.linalg.inv(A)
```

Debe validarse con:

* celda cúbica;
* identidad;
* inversión;
* rotación de 90°;
* traslación pura;
* celda no ortogonal.

---

## 6.9. Reporte de simetría

Añadir función:

```python
def print_symmetry_report(
    sym_info: SymmetryInfo,
    requested_atoms: list[int],
    requested_axes: list[int],
    delta_values: list[float],
    method: str,
    include_base: bool,
) -> None:
    ...
```

Salida esperada:

```text
=== Symmetry report ===

Space group:
  number: 225
  symbol: Fm-3m

Tolerances:
  symprec: 1.0e-03
  angle_tolerance: -1.0

Atoms:
  total atoms: 64
  requested atoms: 64
  inequivalent atoms in full structure: 2
  inequivalent atoms within requested set: 2

Orbits:
  representative 0: [0, 4, 8, 12, ...]
  representative 1: [1, 5, 9, 13, ...]

Finite-difference settings:
  method: central
  signs: [+1, -1]
  axes: [x, y, z]
  deltas: [0.01]

Current brute-force structures:
  384

Atom-symmetry-reduced structures:
  12

Potential speedup:
  32.0x

Safety:
  reconstruction of full Hamiltonian: not enabled
  orbital transform required: yes
```

---

## 6.10. Criterios de aceptación de la Fase 1

La Fase 1 está completa cuando:

* `--symmetry-report` funciona sin generar stencils;
* el modo legacy no cambia;
* se imprimen grupo espacial, operaciones, órbitas y ahorro potencial;
* hay tests unitarios para:

  * celda cúbica simple;
  * dos átomos equivalentes;
  * estructura sin simetría;
  * estructura con defecto;
  * estructura con tolerancia demasiado estricta;
  * estructura con tolerancia demasiado laxa.

---

# 7. Fase 2: Reducción por átomos inequivalentes

## 7.1. Objetivo

Permitir generar stencils solo para representantes inequivalentes.

Esto reduce el número de cálculos, pero inicialmente no reconstruye todas las derivadas.

---

## 7.2. Nuevo modo

```bash
--symmetry-mode atoms
```

Comportamiento:

```text
Genera stencils solo para átomos representantes.
Calcula derivadas solo para esos representantes.
No intenta producir derivadas completas para todos los átomos.
```

Debe marcarse explícitamente como salida irreducible.

---

## 7.3. Selección de representantes dentro de `--atoms`

Caso importante: el usuario puede pedir solo un subconjunto de átomos.

Ejemplo:

```bash
--atoms 0 1 2 3 4 5
```

Si las órbitas completas son:

```text
rep 0: [0, 2, 4, 6]
rep 1: [1, 3, 5, 7]
```

y el usuario pidió `[0,1,2,3,4,5]`, los representantes dentro del subconjunto son:

```text
0 y 1
```

Función:

```python
def reduce_requested_atoms_by_symmetry(
    requested_atoms: list[int],
    sym_info: SymmetryInfo,
) -> list[int]:
    requested = set(requested_atoms)
    reps = []

    for rep, orbit in sym_info.orbits.items():
        intersection = [a for a in orbit if a in requested]
        if not intersection:
            continue

        # Prefer the true representative if requested.
        if rep in requested:
            reps.append(rep)
        else:
            # Otherwise choose the smallest requested atom in that orbit.
            reps.append(min(intersection))

    return sorted(reps)
```

---

## 7.4. Modificación en `build_hamiltonian_derivative_stencils.py`

Actualmente:

```python
atom_indices_zero_based = parse_atoms(args.atoms)
```

Nuevo flujo:

```python
atom_indices_zero_based = parse_atoms(args.atoms)

sym_info = None

if args.symmetry_mode in {"report", "atoms", "dofs", "reconstruct"}:
    sym_info = detect_symmetry_from_base_snapshot(...)

if args.symmetry_mode == "report":
    print_symmetry_report(...)
    return

if args.symmetry_mode in {"atoms", "dofs", "reconstruct"}:
    atom_indices_to_displace = reduce_requested_atoms_by_symmetry(
        atom_indices_zero_based,
        sym_info,
    )
else:
    atom_indices_to_displace = atom_indices_zero_based
```

Después:

```python
for delta_ang in delta_ang_values:
    for atom_index in atom_indices_to_displace:
        for axis in axes:
            for sign in signs_for_method(method):
                ...
```

---

## 7.5. Manifest en modo `atoms`

Cada sample debe guardar:

```json
{
  "sample_id": "snapshot_000_atom_000_axis_x_plus_delta_0.01",
  "atom_index": 0,
  "axis_index": 0,
  "sign": 1,
  "delta_ang": 0.01,
  "symmetry": {
    "is_irreducible_sample": true,
    "representative_atom": 0,
    "covers_atoms": [0, 2, 4, 6],
    "reconstruction_required_for_full_output": true
  }
}
```

---

## 7.6. Criterios de aceptación de la Fase 2

La Fase 2 está completa cuando:

* `--symmetry-mode atoms` genera menos stencils;
* el número generado coincide con:

[
N_{\text{structures}}
=====================

N_\delta
\times
N_{\text{rep atoms}}
\times
N_{\text{axes}}
\times
N_{\text{signs}}
+
N_{\text{base}}
]

* el manifest indica que la salida es irreducible;
* el pipeline no confunde esta salida con la derivada completa;
* hay tests que comprueban que el modo `atoms` no se usa accidentalmente como si fuera completo.

---

# 8. Fase 3: Manifest enriquecido con simetría

## 8.1. Objetivo

Guardar suficiente información para reconstruir derivadas faltantes más adelante.

---

## 8.2. Versión del manifest

Añadir un campo de versión:

```json
{
  "manifest_version": 2
}
```

---

## 8.3. Bloque de metadatos globales

Ejemplo:

```json
{
  "metadata": {
    "finite_difference": {
      "method": "central",
      "delta_ang_values": [0.01],
      "axes": [0, 1, 2],
      "include_base": true
    },
    "symmetry": {
      "enabled": true,
      "mode": "atoms",
      "symprec": 0.001,
      "angle_tolerance": -1.0,
      "spacegroup_number": 225,
      "international_symbol": "Fm-3m",
      "equivalent_atoms": [0, 0, 0, 0, 4, 4, 4, 4],
      "representative_atoms": [0, 4],
      "orbits": {
        "0": [0, 1, 2, 3],
        "4": [4, 5, 6, 7]
      },
      "operations": [
        {
          "id": 0,
          "rotation_frac": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
          "translation_frac": [0.0, 0.0, 0.0],
          "rotation_cart": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
          "atom_map": [0, 1, 2, 3, 4, 5, 6, 7]
        }
      ]
    }
  }
}
```

---

## 8.4. Bloque por sample

```json
{
  "sample_id": "snap000_rep000_axis0_plus_delta0.010",
  "base_snapshot_id": "snap000",
  "atom_index_zero_based": 0,
  "axis_index": 0,
  "axis_label": "x",
  "sign": 1,
  "delta_ang": 0.01,
  "is_base": false,
  "symmetry": {
    "is_irreducible": true,
    "representative_atom": 0,
    "representative_axis": 0,
    "covered_atoms": [0, 1, 2, 3],
    "covered_dofs": [
      [0, 0],
      [1, 0],
      [2, 0],
      [3, 0]
    ]
  }
}
```

---

## 8.5. Compatibilidad hacia atrás

Los scripts que lean manifests deben aceptar:

* `manifest_version = 1`, sin simetría;
* `manifest_version = 2`, con bloque opcional `metadata.symmetry`.

Si no existe bloque de simetría:

```python
symmetry_enabled = False
```

---

# 9. Fase 4: Órbitas de grados de libertad

## 9.1. Objetivo

Reducir no solo átomos, sino también ejes cuando sea seguro.

Un grado de libertad es:

[
(i,\alpha)
]

donde (i) es el átomo y (\alpha) es el eje.

La simetría puede relacionar:

[
(i,x) \leftrightarrow (j,y)
]

o incluso:

[
(i,x) \leftrightarrow (j,-x)
]

---

## 9.2. Caso seguro: matrices de permutación con signo

Una rotación cartesiana (Q) es una permutación con signo si cada fila y columna tiene exactamente un elemento no nulo, y ese elemento es (\pm 1).

Ejemplo:

[
Q =
\begin{pmatrix}
0 & 1 & 0 \
-1 & 0 & 0 \
0 & 0 & 1
\end{pmatrix}
]

Esto significa:

[
x \rightarrow -y
]

[
y \rightarrow x
]

[
z \rightarrow z
]

Código:

```python
def is_signed_permutation_matrix(Q: np.ndarray, tol: float = 1e-8) -> bool:
    Q_round = np.round(Q).astype(int)

    if not np.allclose(Q, Q_round, atol=tol):
        return False

    if not np.all(np.isin(Q_round, [-1, 0, 1])):
        return False

    row_ok = np.all(np.sum(np.abs(Q_round), axis=1) == 1)
    col_ok = np.all(np.sum(np.abs(Q_round), axis=0) == 1)

    return bool(row_ok and col_ok)
```

---

## 9.3. Mapeo de ejes

```python
def map_axis_under_signed_permutation(Q: np.ndarray, axis: int) -> tuple[int, int]:
    """
    Returns:
        target_axis, sign

    Meaning:
        Q e_axis = sign * e_target_axis
    """
    Q_round = np.round(Q).astype(int)
    col = Q_round[:, axis]
    nonzero = np.nonzero(col)[0]

    if len(nonzero) != 1:
        raise ValueError("Q is not a signed permutation for this axis.")

    target_axis = int(nonzero[0])
    sign = int(col[target_axis])

    return target_axis, sign
```

---

## 9.4. Construcción de órbitas de grados de libertad

```python
def build_dof_orbits(
    requested_atoms: list[int],
    requested_axes: list[int],
    sym_info: SymmetryInfo,
    tol: float = 1e-8,
) -> dict[tuple[int, int], set[tuple[int, int, int]]]:
    """
    Returns a mapping from representative DOF to a set of covered DOFs.

    A covered DOF is represented as:
        (atom, axis, sign)

    sign means:
        displacement along representative +axis maps to sign * target axis.
    """
    requested_dofs = {
        (atom, axis)
        for atom in requested_atoms
        for axis in requested_axes
    }

    visited = set()
    dof_orbits = {}

    for dof in sorted(requested_dofs):
        if dof in visited:
            continue

        rep_atom, rep_axis = dof
        covered = set()

        for op in sym_info.operations:
            Q = op.rotation_cart

            if not is_signed_permutation_matrix(Q, tol=tol):
                continue

            target_atom = op.atom_map[rep_atom]
            target_axis, axis_sign = map_axis_under_signed_permutation(Q, rep_axis)

            if (target_atom, target_axis) in requested_dofs:
                covered.add((target_atom, target_axis, axis_sign))

        # Mark atoms/axes, ignoring sign for visited.
        for atom, axis, _sign in covered:
            visited.add((atom, axis))

        dof_orbits[dof] = covered

    return dof_orbits
```

---

## 9.5. Precaución

Reducir ejes solo debe activarse si:

1. (Q) es una matriz de permutación con signo;
2. el tratamiento orbital está soportado;
3. se puede transformar el Hamiltoniano;
4. los signos se manejan correctamente;
5. los tests comparan contra cálculo explícito.

Para un MVP, es mejor reducir solo átomos y mantener los tres ejes.

---

# 10. Fase 5: Reconstrucción por simetría para casos seguros

## 10.1. Objetivo

A partir de derivadas calculadas para representantes, reconstruir derivadas para todos los átomos equivalentes.

La fórmula general es:

[
D_{j\beta}H
===========

\sum_{\alpha}
Q_{\beta\alpha}
U_g
D_{i\alpha}H
U_g^\dagger
]

donde:

[
D_{i\alpha}H =
\frac{\partial H}{\partial R_{i\alpha}}
]

---

## 10.2. Función de reconstrucción

```python
def reconstruct_atom_derivatives(
    rep_derivs: dict[int, "sparse matrix"],
    operation: SymmetryOperation,
    orbital_transform,
) -> dict[int, "sparse matrix"]:
    """
    Given derivatives for a representative atom along x,y,z,
    reconstruct derivatives for a symmetry-related target atom.

    rep_derivs:
        {0: dH/dR_rep_x, 1: dH/dR_rep_y, 2: dH/dR_rep_z}

    operation:
        symmetry operation mapping representative atom to target atom

    orbital_transform:
        U_g representation in orbital space

    returns:
        {0: dH/dR_target_x, 1: dH/dR_target_y, 2: dH/dR_target_z}
    """
    Q = operation.rotation_cart
    U = orbital_transform

    transformed_rep_derivs = {}

    for alpha in range(3):
        transformed_rep_derivs[alpha] = U @ rep_derivs[alpha] @ U.T.conjugate()

    target_derivs = {}

    for beta in range(3):
        acc = None

        for alpha in range(3):
            coeff = Q[beta, alpha]
            if abs(coeff) < 1e-14:
                continue

            term = coeff * transformed_rep_derivs[alpha]
            acc = term if acc is None else acc + term

        target_derivs[beta] = acc

    return target_derivs
```

---

## 10.3. Primer caso seguro: traslaciones puras

Una traslación pura tiene:

[
Q = I
]

y no mezcla orbitales.

En ese caso, la transformación es principalmente:

* permutación de átomos;
* permutación de orbitales asociados;
* posible cambio de imagen periódica.

Para sistemas periódicos, incluso las traslaciones puras pueden afectar al índice de celda en el Hamiltoniano almacenado. Por eso deben validarse cuidadosamente.

---

## 10.4. Segundo caso seguro: bases puramente (s)

Si todos los orbitales son tipo (s), las rotaciones no mezclan orbitales angulares.

Entonces (U_g) se reduce a:

* permutación de átomos;
* permutación de orbitales radiales/zeta;
* posible imagen periódica.

Este caso es un excelente banco de pruebas.

---

## 10.5. Casos inicialmente no soportados

Desactivar reconstrucción si se detecta:

* orbitales (p), (d), (f) sin soporte de rotación orbital;
* SOC;
* espín no colineal;
* magnetismo antiferromagnético no tratado;
* offsets periódicos no interpretables;
* HSX antiguo incompleto;
* discrepancias de base entre geometrías desplazadas;
* matriz no compatible con la transformación.

SIESTA incluye capacidades como spin-orbit y tratamientos avanzados; por tanto, el código debe detectar estos casos y no asumir que una simetría espacial ordinaria basta.

---

# 11. Fase 6: Construcción de (U_g)

## 11.1. Objetivo

Construir la representación de una operación de simetría en la base orbital del Hamiltoniano.

[
U_g
]

debe actuar sobre la base de SIESTA.

---

## 11.2. Estructura conceptual de (U_g)

[
U_g
===

P_{\text{átomos}}
\cdot
R_{\text{orbitales}}
\cdot
P_{\text{imágenes}}
\cdot
R_{\text{espín}}
]

donde:

* (P_{\text{átomos}}): permuta átomos equivalentes;
* (R_{\text{orbitales}}): rota orbitales (p,d,f);
* (P_{\text{imágenes}}): transforma offsets de celda;
* (R_{\text{espín}}): transforma espinores si procede.

---

## 11.3. Extracción de base orbital

Usar `sisl` si es compatible con el flujo actual.

La documentación de `sisl` indica que `hsxSileSiesta` es un lector para archivos de Hamiltoniano y solapamiento HSX, con métodos como `read_basis`, `read_geometry`, `read_hamiltonian`, `read_lattice` y `read_overlap`. También advierte que los archivos HSX modernos contienen más información que formatos antiguos.

Función propuesta:

```python
def read_basis_metadata(reference_hsx_path: str) -> BasisMetadata:
    """
    Extract orbital-to-atom mapping and quantum numbers from HSX/TSHS.

    This function should return enough information to build U_g.
    """
    ...
```

---

## 11.4. Estructura de datos para la base

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class OrbitalInfo:
    global_index: int
    atom_index: int
    species: str
    n: int | None
    l: int
    m: int | None
    zeta: int | None
    polarization: bool
    label: str


@dataclass(frozen=True)
class BasisMetadata:
    n_orbitals: int
    orbitals: tuple[OrbitalInfo, ...]
    orbitals_by_atom: dict[int, tuple[int, ...]]
    spin_mode: str
    has_soc: bool
    has_noncolinear: bool
```

---

## 11.5. Validación de compatibilidad de base

Antes de reconstruir:

```python
def assert_basis_compatible_with_symmetry(basis: BasisMetadata) -> None:
    if basis.has_soc:
        raise UnsupportedSymmetryCase("SOC not supported yet.")

    if basis.has_noncolinear:
        raise UnsupportedSymmetryCase("Non-colinear spin not supported yet.")

    for orb in basis.orbitals:
        if orb.l > 0:
            raise UnsupportedSymmetryCase(
                "Orbital rotations for l>0 not implemented yet."
            )
```

Más adelante, se puede ir relajando esta restricción.

---

## 11.6. Construcción de (U_g) para base (s)

Caso más simple:

```python
def build_orbital_transform_s_only(
    operation: SymmetryOperation,
    basis: BasisMetadata,
) -> "sparse matrix":
    """
    Build U_g for a basis containing only s-like orbitals.

    This only permutes atom-centered orbitals.
    """
    rows = []
    cols = []
    data = []

    for orb in basis.orbitals:
        src_idx = orb.global_index
        src_atom = orb.atom_index
        target_atom = operation.atom_map[src_atom]

        target_idx = find_matching_orbital_on_atom(
            basis=basis,
            target_atom=target_atom,
            source_orbital=orb,
        )

        rows.append(target_idx)
        cols.append(src_idx)
        data.append(1.0)

    U = scipy.sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(basis.n_orbitals, basis.n_orbitals),
    )

    return U
```

---

## 11.7. Matching de orbitales

```python
def find_matching_orbital_on_atom(
    basis: BasisMetadata,
    target_atom: int,
    source_orbital: OrbitalInfo,
) -> int:
    candidates = []

    for idx in basis.orbitals_by_atom[target_atom]:
        orb = basis.orbitals[idx]

        if (
            orb.l == source_orbital.l
            and orb.m == source_orbital.m
            and orb.zeta == source_orbital.zeta
            and orb.polarization == source_orbital.polarization
            and orb.label == source_orbital.label
        ):
            candidates.append(orb.global_index)

    if len(candidates) != 1:
        raise RuntimeError(
            f"Could not match orbital {source_orbital} on atom {target_atom}. "
            f"Candidates: {candidates}"
        )

    return candidates[0]
```

---

## 11.8. Extensión a orbitales (p)

Para orbitales (p_x,p_y,p_z), la matriz de rotación orbital es esencialmente (Q), pero hay que asegurarse de que la convención de orbitales reales de SIESTA coincide con la convención usada por el código.

Si la base local tiene orden:

```text
p_x, p_y, p_z
```

entonces:

[
\begin{pmatrix}
p'_x \
p'_y \
p'_z
\end{pmatrix}
=============

Q
\begin{pmatrix}
p_x \
p_y \
p_z
\end{pmatrix}
]

Pero esto debe validarse numéricamente, no asumirse.

---

## 11.9. Extensión a orbitales (d) y (f)

Para (d) y (f), se necesitan matrices de rotación de armónicos esféricos reales:

[
D^{(l)}(Q)
]

con:

* (l=2) para orbitales (d);
* (l=3) para orbitales (f).

Esto implica:

```text
d_xy, d_yz, d_z2, d_xz, d_x2-y2
```

o el orden concreto que use SIESTA.

Esta parte debe implementarse solo después de validar (s) y (p).

---

# 12. Fase 7: Manejo de matrices dispersas HSX/TSHS y offsets periódicos

## 12.1. Problema

Según el contexto, `finite_difference_derivative()` opera sobre matrices dispersas de forma:

[
(n_{\text{orbitals}}, n_{\text{orbitals}} \times n_{\text{supercells}})
]

Esto significa que la matriz no es simplemente:

[
n_{\text{orb}} \times n_{\text{orb}}
]

sino que contiene bloques hacia imágenes periódicas.

---

## 12.2. Forma conceptual

Un elemento puede interpretarse como:

[
H_{\mu 0,\nu T}
]

donde:

* (\mu) es un orbital en la celda base;
* (\nu) es un orbital en una celda imagen;
* (T) es un vector de traslación de celda.

Una operación de simetría transforma:

[
(\mu,0;\nu,T)
\rightarrow
(\mu',0;\nu',T')
]

donde (T') depende de (W), de las posiciones atómicas y de la convención de almacenamiento.

---

## 12.3. Plan de implementación para offsets

Primero, construir una representación explícita de los índices:

```python
@dataclass(frozen=True)
class MatrixColumnIndex:
    orbital_index: int
    cell_offset_index: int
    cell_offset_vector: tuple[int, int, int]
```

Luego construir mapas:

```python
def decode_extended_column(col: int, n_orbitals: int) -> MatrixColumnIndex:
    cell_offset_index = col // n_orbitals
    orbital_index = col % n_orbitals
    ...
```

y:

```python
def encode_extended_column(
    orbital_index: int,
    cell_offset_index: int,
    n_orbitals: int,
) -> int:
    return cell_offset_index * n_orbitals + orbital_index
```

---

## 12.4. Transformación de offset

En coordenadas fraccionarias, una operación transforma un vector de celda (T) como:

[
T' = W T + \Delta
]

donde (\Delta) puede aparecer por el recentrado de átomos dentro de la celda base.

Para cada elemento (H_{\mu 0,\nu T}):

1. identificar átomo de (\mu): (a);
2. identificar átomo de (\nu): (b);
3. aplicar simetría a (a);
4. aplicar simetría a (b+T);
5. reducir ambos a la celda base;
6. determinar el nuevo offset (T');
7. mapear orbitales (\mu\rightarrow\mu'), (\nu\rightarrow\nu').

---

## 12.5. Por qué esta fase debe ir después

Esta fase es delicada porque depende del formato exacto de la matriz SIESTA/HSX/TSHS.

Por tanto, antes de implementarla completamente, conviene:

1. usar `sisl` para inspeccionar la geometría y los offsets;
2. escribir tests en un sistema mínimo;
3. comparar transformación de Hamiltonianos completos, no solo derivadas.

---

# 13. Fase 8: Reconstrucción completa de derivadas

## 13.1. Flujo general

```text
1. Generar stencils irreducibles
2. Ejecutar SIESTA solo en esos stencils
3. Leer Hamiltonianos completos
4. Calcular derivadas por diferencias finitas para representantes
5. Reconstruir derivadas faltantes por simetría
6. Escribir salida completa compatible con el workflow antiguo
```

---

## 13.2. Pseudocódigo global

```python
def compute_derivatives_with_optional_symmetry(manifest, reference_matrices):
    fd_derivatives = compute_finite_differences_for_available_samples(
        manifest=manifest,
        reference_matrices=reference_matrices,
    )

    symmetry_metadata = manifest.metadata.get("symmetry", None)

    if not symmetry_metadata or not symmetry_metadata["enabled"]:
        return fd_derivatives

    if symmetry_metadata["mode"] in {"atoms", "dofs"}:
        return fd_derivatives

    if symmetry_metadata["mode"] == "reconstruct":
        basis = read_basis_metadata(reference_matrices.reference_hsx_path)
        sym_info = load_symmetry_info_from_manifest(manifest)

        reconstructed = reconstruct_full_derivative_set(
            irreducible_derivatives=fd_derivatives,
            sym_info=sym_info,
            basis=basis,
            requested_atoms=manifest.metadata["requested_atoms"],
            requested_axes=manifest.metadata["finite_difference"]["axes"],
        )

        return reconstructed

    raise ValueError(f"Unknown symmetry mode: {symmetry_metadata['mode']}")
```

---

## 13.3. Reconstrucción completa

```python
def reconstruct_full_derivative_set(
    irreducible_derivatives,
    sym_info: SymmetryInfo,
    basis: BasisMetadata,
    requested_atoms: list[int],
    requested_axes: list[int],
):
    all_derivatives = {}

    for rep_atom in sym_info.representative_atoms:
        rep_derivs = {
            axis: irreducible_derivatives[(rep_atom, axis)]
            for axis in [0, 1, 2]
            if (rep_atom, axis) in irreducible_derivatives
        }

        if len(rep_derivs) != 3:
            raise RuntimeError(
                f"Representative atom {rep_atom} does not have all 3 axes."
            )

        orbit = sym_info.orbits[rep_atom]

        for target_atom in orbit:
            op = find_operation_mapping_atom(
                sym_info=sym_info,
                source_atom=rep_atom,
                target_atom=target_atom,
            )

            U = build_orbital_transform(
                operation=op,
                basis=basis,
            )

            target_derivs = reconstruct_atom_derivatives(
                rep_derivs=rep_derivs,
                operation=op,
                orbital_transform=U,
            )

            for axis in requested_axes:
                if target_atom in requested_atoms:
                    all_derivatives[(target_atom, axis)] = target_derivs[axis]

    return all_derivatives
```

---

## 13.4. Elección de operación entre dos átomos

Puede haber varias operaciones que mandan (i) a (j).

```python
def find_operation_mapping_atom(
    sym_info: SymmetryInfo,
    source_atom: int,
    target_atom: int,
) -> SymmetryOperation:
    candidates = [
        op for op in sym_info.operations
        if op.atom_map[source_atom] == target_atom
    ]

    if not candidates:
        raise RuntimeError(
            f"No symmetry operation maps atom {source_atom} to {target_atom}."
        )

    # Prefer simplest operation:
    # 1. translation-like
    # 2. signed permutation
    # 3. smallest rotation deviation
    return choose_preferred_operation(candidates)
```

---

## 13.5. Selección de operación preferida

```python
def choose_preferred_operation(
    candidates: list[SymmetryOperation],
) -> SymmetryOperation:
    def score(op):
        Q = op.rotation_cart

        is_identity = np.allclose(Q, np.eye(3), atol=1e-8)
        is_signed_perm = is_signed_permutation_matrix(Q)

        if is_identity:
            return 0

        if is_signed_perm:
            return 1

        return 2

    return sorted(candidates, key=score)[0]
```

---

# 14. Fase 9: Validación física y numérica

## 14.1. Principio

Nunca activar reconstrucción por simetría como default sin demostrar que reproduce el cálculo completo.

Debe existir un modo de validación:

```bash
--symmetry-validate
```

que haga:

```text
1. cálculo completo brute-force;
2. cálculo reducido por simetría;
3. reconstrucción;
4. comparación matriz a matriz.
```

---

## 14.2. Métrica principal

Para cada derivada:

[
\epsilon_{i\alpha}
==================

\frac{
|D_{i\alpha}^{\text{full}}H
---------------------------

D_{i\alpha}^{\text{sym}}H|
}{
|D_{i\alpha}^{\text{full}}H|+\epsilon
}
]

Código:

```python
def relative_sparse_error(A, B, eps: float = 1e-14) -> float:
    diff = A - B
    numerator = sparse_frobenius_norm(diff)
    denominator = sparse_frobenius_norm(A) + eps
    return float(numerator / denominator)
```

---

## 14.3. Norma de Frobenius dispersa

```python
def sparse_frobenius_norm(A) -> float:
    return float(np.sqrt(np.sum(np.abs(A.data) ** 2)))
```

---

## 14.4. Criterios iniciales

|      Error relativo | Interpretación            |
| ------------------: | ------------------------- |
|          (<10^{-5}) | Excelente                 |
| (10^{-5} - 10^{-4}) | Muy bueno                 |
| (10^{-4} - 10^{-3}) | Aceptable en muchos casos |
| (10^{-3} - 10^{-2}) | Sospechoso                |
|          (>10^{-2}) | No aceptar                |

Estos umbrales deben ajustarse según:

* ruido SCF;
* valor de (\delta);
* malla real;
* tolerancia de simetría;
* convergencia electrónica;
* precisión de lectura/escritura de matrices.

---

## 14.5. Test de geometría

Para cada operación (g):

[
g(\mathbf R) = \mathbf R
]

módulo vectores de red y permutación atómica.

```python
def test_symmetry_operation_maps_structure(sym_info, frac_positions):
    for op in sym_info.operations:
        for i, f_i in enumerate(frac_positions):
            j = op.atom_map[i]

            f_new = op.rotation_frac @ f_i + op.translation_frac
            f_new = f_new % 1.0

            diff = f_new - frac_positions[j]
            diff -= np.round(diff)

            assert np.linalg.norm(diff) < sym_info.symprec
```

---

## 14.6. Test de derivadas reconstruidas

```python
def test_reconstructed_derivatives_match_full(
    full_derivatives,
    reconstructed_derivatives,
):
    for key, D_full in full_derivatives.items():
        D_sym = reconstructed_derivatives[key]

        err = relative_sparse_error(D_full, D_sym)

        assert err < 1e-3, f"{key}: error = {err}"
```

---

## 14.7. Test de Hamiltonianos desplazados

Comparar directamente:

[
H(g\mathbf R_\delta)
]

contra:

[
U_g H(\mathbf R_\delta) U_g^\dagger
]

Esto valida (U_g), no solo la derivada.

```python
def test_hamiltonian_covariance(H_displaced, H_sym_displaced, U):
    H_transformed = U @ H_displaced @ U.T.conjugate()

    err = relative_sparse_error(H_sym_displaced, H_transformed)

    assert err < 1e-4
```

---

## 14.8. Test de varios valores de delta

Correr con:

```text
delta = 0.005 Å
delta = 0.010 Å
delta = 0.020 Å
```

La derivada debe ser estable:

[
D(\delta_1) \approx D(\delta_2)
]

Si la reconstrucción falla solo para ciertos (\delta), puede haber:

* ruido SCF;
* no linealidad;
* falta de convergencia;
* diferencias de malla;
* simetría rota por tolerancias.

---

# 15. Integración con scripts existentes

## 15.1. `build_hamiltonian_derivative_stencils.py`

Cambios:

1. añadir flags;
2. detectar simetría;
3. reducir lista de átomos/ejes según modo;
4. guardar manifest v2;
5. imprimir reporte;
6. mantener modo legacy intacto.

Pseudoflujo:

```python
def build_derivative_stencils(...):
    base_structure = load_base_structure(...)

    atom_indices = parse_atoms(args.atoms)
    axes = parse_axes(args.axes)
    deltas = parse_deltas(args.delta_ang)

    sym_info = None

    if args.symmetry_mode != "off":
        sym_info = detect_symmetry_from_base_structure(
            base_structure,
            symprec=args.symprec,
            angle_tolerance=args.angle_tolerance,
        )

    if args.symmetry_mode == "report":
        print_symmetry_report(...)
        return

    if args.symmetry_mode in {"atoms", "dofs", "reconstruct"}:
        atoms_to_displace = reduce_requested_atoms_by_symmetry(
            atom_indices,
            sym_info,
        )
    else:
        atoms_to_displace = atom_indices

    manifest = []

    for delta in deltas:
        for atom in atoms_to_displace:
            for axis in axes:
                for sign in signs_for_method(method):
                    sample = create_displaced_sample(
                        base_structure=base_structure,
                        atom=atom,
                        axis=axis,
                        sign=sign,
                        delta=delta,
                    )
                    manifest.append(sample)

    write_manifest(
        samples=manifest,
        metadata=build_metadata(..., sym_info=sym_info),
    )
```

---

## 15.2. `hamiltonian_derivative_stencil.py`

No modificaría la función básica:

```python
finite_difference_derivative()
```

Debe seguir haciendo solo:

[
\frac{H^+ - H^-}{2\delta}
]

Añadiría funciones nuevas:

```python
compute_irreducible_derivatives()
reconstruct_derivatives_by_symmetry()
load_symmetry_metadata()
```

Flujo:

```python
irreducible_derivatives = compute_finite_differences(...)
full_derivatives = maybe_reconstruct_by_symmetry(...)
```

---

## 15.3. `run_hamiltonian_derivative_siesta_references.py`

Idealmente no necesita saber física de simetría.

Si el manifest contiene menos estructuras, este script ejecuta menos SIESTA.

Cambios mínimos:

* aceptar manifest v2;
* no asumir que el número de samples es (6N);
* preservar metadatos de simetría en salidas.

---

## 15.4. `run_hamiltonian_derivative_predictions.py`

Mismo principio.

Para finite difference legacy:

```text
menos estructuras en manifest = menos predicciones ML
```

Pero para Graph2Mat autograd puede haber una vía alternativa:

* usar autograd para derivadas completas;
* usar autograd como validación de covariancia por simetría;
* comparar columnas equivalentes del jacobiano.

---

## 15.5. `graph2mat_autograd_derivatives.py`

No es el objetivo principal, pero puede servir para validar.

Como el jacobiano autograd ya da:

[
\frac{\partial H}{\partial R_{i\alpha}}
]

para muchas salidas, se puede comprobar:

[
D_{j\beta}^{\text{autograd}}
\stackrel{?}{=}
\sum_\alpha
Q_{\beta\alpha}
U_g
D_{i\alpha}^{\text{autograd}}
U_g^\dagger
]

Esto permitiría testear simetría sin ejecutar muchos SIESTA.

---

# 16. Nuevos archivos propuestos

## 16.1. `Comparison/scripts/symmetry_utils.py`

Contenido:

```python
SymmetryOperation
SymmetryInfo
detect_symmetry()
frac_rotation_to_cartesian()
cart_to_frac()
frac_to_cart()
build_atom_map()
build_orbits()
reduce_requested_atoms_by_symmetry()
is_signed_permutation_matrix()
map_axis_under_signed_permutation()
build_dof_orbits()
serialize_symmetry_info()
deserialize_symmetry_info()
print_symmetry_report()
```

---

## 16.2. `Comparison/scripts/hamiltonian_symmetry.py`

Contenido:

```python
BasisMetadata
OrbitalInfo
UnsupportedSymmetryCase
read_basis_metadata()
assert_basis_compatible_with_symmetry()
build_orbital_transform()
build_orbital_transform_s_only()
find_matching_orbital_on_atom()
transform_hamiltonian()
transform_hamiltonian_derivative()
reconstruct_atom_derivatives()
reconstruct_full_derivative_set()
relative_sparse_error()
sparse_frobenius_norm()
```

---

## 16.3. `Comparison/scripts/symmetry_validation.py`

Contenido:

```python
validate_geometry_symmetry()
validate_hamiltonian_covariance()
validate_derivative_reconstruction()
generate_validation_report()
```

---

# 17. CLI final propuesta

## 17.1. Solo reporte

```bash
python Comparison/scripts/build_hamiltonian_derivative_stencils.py \
  --input base.fdf \
  --symmetry-mode report \
  --symprec 1e-3
```

---

## 17.2. Generar solo átomos inequivalentes

```bash
python Comparison/scripts/build_hamiltonian_derivative_stencils.py \
  --input base.fdf \
  --method central \
  --delta-ang 0.01 \
  --symmetry-mode atoms \
  --symprec 1e-3
```

---

## 17.3. Reconstrucción completa, modo estricto

```bash
python Comparison/scripts/build_hamiltonian_derivative_stencils.py \
  --input base.fdf \
  --method central \
  --delta-ang 0.01 \
  --symmetry-mode reconstruct \
  --symmetry-strict \
  --symprec 1e-3
```

---

## 17.4. Validación contra brute force

```bash
python Comparison/scripts/validate_hamiltonian_derivative_symmetry.py \
  --full-manifest full/manifest.json \
  --symmetry-manifest sym/manifest.json \
  --tolerance 1e-3
```

---

# 18. Gestión de casos peligrosos

## 18.1. Tabla de riesgos

| Caso                     | Riesgo                                     | Acción recomendada                   |
| ------------------------ | ------------------------------------------ | ------------------------------------ |
| Defectos                 | baja simetría real                         | permitir, pero habrá menos reducción |
| Superficies              | simetría rota en z                         | detectar automáticamente             |
| Adsorbatos               | pocos átomos equivalentes                  | no forzar simetría                   |
| Relajación imperfecta    | spglib puede no detectar grupo correcto    | probar varios `symprec`              |
| `symprec` demasiado laxo | simetrías falsas                           | validar contra brute force           |
| Orbitales (p,d,f)        | mezcla orbital                             | no reconstruir sin (U_g) correcto    |
| SOC                      | requiere espinores                         | desactivar inicialmente              |
| Espín no colineal        | requiere simetría magnética                | desactivar inicialmente              |
| AFM                      | simetría espacial puede no preservar espín | desactivar salvo soporte magnético   |
| HSX antiguo              | puede faltar información                   | exigir HSX moderno o usar fallback   |
| Offsets periódicos       | transformación compleja                    | validar elemento a elemento          |

---

## 18.2. Política de seguridad

Si el usuario pide:

```bash
--symmetry-mode reconstruct
```

y el sistema detecta un caso no soportado, el código debe hacer una de dos cosas:

### Si `--symmetry-strict` está activo

Fallar explícitamente:

```text
ERROR: Symmetry reconstruction requested, but basis contains p/d/f orbitals.
Orbital rotations are not implemented yet.
Use --symmetry-mode atoms or --symmetry-mode off.
```

### Si `--symmetry-strict` no está activo

Hacer fallback:

```text
WARNING: Symmetry reconstruction not safe for this basis.
Falling back to brute-force stencil generation.
```

---

# 19. Validación por etapas

## 19.1. Sistemas de prueba mínimos

Usar varios sistemas:

### Sistema A: molécula o cristal trivial con un átomo y orbital (s)

Objetivo:

* validar lectura;
* validar identidad;
* validar no romper modo legacy.

### Sistema B: cadena 1D con átomos equivalentes

Objetivo:

* validar traslaciones;
* validar reducción por átomos.

### Sistema C: celda cúbica simple

Objetivo:

* validar rotaciones tipo signed permutation.

### Sistema D: cristal con dos subredes

Ejemplo conceptual:

```text
NaCl, diamond, graphene, MoS2
```

Objetivo:

* validar órbitas múltiples;
* validar representantes.

### Sistema E: defecto

Objetivo:

* comprobar que la simetría baja automáticamente.

### Sistema F: base con (p)

Objetivo:

* comprobar que reconstrucción falla de forma controlada si no se implementó rotación orbital.

---

## 19.2. Tests unitarios

```text
tests/test_symmetry_utils.py
tests/test_symmetry_manifest.py
tests/test_irreducible_atoms.py
tests/test_dof_orbits.py
tests/test_hamiltonian_symmetry_s_only.py
tests/test_symmetry_validation.py
```

---

## 19.3. Tests concretos

### Test: identidad

```python
def test_identity_operation_maps_atoms_to_themselves():
    ...
```

### Test: inversión

```python
def test_inversion_is_signed_permutation():
    ...
```

### Test: rotación 90°

```python
def test_rotation_90_maps_x_to_y():
    ...
```

### Test: órbitas

```python
def test_equivalent_atoms_reduce_to_expected_representatives():
    ...
```

### Test: manifest v2

```python
def test_manifest_v2_contains_symmetry_metadata():
    ...
```

### Test: fallback seguro

```python
def test_reconstruct_falls_back_when_basis_has_p_orbitals():
    ...
```

---

# 20. Métricas de rendimiento

## 20.1. Reportar antes de ejecutar

El reporte debe calcular:

[
S_{\text{ideal}}
================

\frac{
N_{\text{brute}}
}{
N_{\text{sym}}
}
]

donde:

[
N_{\text{brute}}
================

N_\delta
N_{\text{atoms}}
N_{\text{axes}}
N_{\text{signs}}
]

y:

[
N_{\text{sym}}
==============

N_\delta
N_{\text{rep atoms}}
N_{\text{axes}}
N_{\text{signs}}
]

---

## 20.2. Ejemplo de salida

```text
Performance estimate:
  brute-force samples: 384
  symmetry-reduced samples: 12
  estimated reduction: 32.0x
  base included: yes
  total directories saved: 372
```

---

## 20.3. Métricas reales

Después de ejecutar:

```text
Timing summary:
  stencil generation time old estimate: ...
  stencil generation time actual: ...
  SIESTA executions skipped: ...
  disk usage saved: ...
  Hamiltonian reads skipped: ...
  reconstruction time: ...
```

---

# 21. Plan de desarrollo por milestones

## Milestone 0: No romper nada

### Tareas

* añadir tests de regresión del modo actual;
* documentar número esperado de stencils;
* congelar un manifest ejemplo.

### Resultado

El repo tiene una línea base clara.

---

## Milestone 1: `symmetry_utils.py`

### Tareas

* añadir dependencia opcional `spglib`;
* implementar detección;
* implementar conversión de coordenadas;
* implementar mapas atómicos;
* implementar órbitas;
* implementar reporte.

### Resultado

Funciona:

```bash
--symmetry-mode report
```

---

## Milestone 2: Reducción `atoms`

### Tareas

* modificar generación de stencils;
* guardar manifest v2;
* marcar salida irreducible;
* impedir uso accidental como derivada completa.

### Resultado

Funciona:

```bash
--symmetry-mode atoms
```

---

## Milestone 3: Validación geométrica

### Tareas

* testear todas las operaciones;
* probar varios `symprec`;
* generar reportes de estabilidad.

### Resultado

Se sabe cuándo la simetría detectada es fiable.

---

## Milestone 4: Reconstrucción para base (s), sin SOC

### Tareas

* leer metadatos de base;
* construir (U_g) por permutación;
* reconstruir derivadas;
* comparar contra brute force.

### Resultado

Funciona:

```bash
--symmetry-mode reconstruct
```

para casos simples.

---

## Milestone 5: Offsets periódicos

### Tareas

* decodificar columnas extendidas;
* mapear offsets;
* transformar Hamiltonianos dispersos;
* validar covariancia.

### Resultado

La reconstrucción funciona para Hamiltonianos periódicos reales del workflow.

---

## Milestone 6: Orbitales (p)

### Tareas

* identificar convención orbital de SIESTA;
* implementar rotación de (p_x,p_y,p_z);
* validar con rotaciones simples.

### Resultado

Reconstrucción soporta bases (s+p).

---

## Milestone 7: Orbitales (d/f)

### Tareas

* implementar matrices reales (D^{(l)}(Q));
* validar orden orbital;
* añadir tests contra rotaciones conocidas.

### Resultado

Reconstrucción soporta bases realistas.

---

## Milestone 8: Espín/SOC

### Tareas

* detectar modos de espín;
* implementar transformación espinorial si procede;
* o mantener fallback seguro.

### Resultado

El código sabe cuándo puede y cuándo no puede aplicar simetría.

---

# 22. Documentación necesaria

## 22.1. Actualizar `docs/workflows.md`

Añadir sección:

```markdown
## Symmetry-aware Hamiltonian derivative stencils
```

Explicar:

* modo legacy;
* modo reporte;
* modo átomos;
* modo reconstrucción;
* limitaciones;
* ejemplos;
* validación.

---

## 22.2. Añadir advertencia importante

```markdown
Symmetry reduction for Hamiltonian derivatives is not equivalent to simply
copying derivatives between equivalent atoms. The Hamiltonian is represented
in an atom-centered orbital basis, so reconstruction requires transforming
both Cartesian displacement directions and orbital indices.
```

---

## 22.3. Documentar fórmula central

[
D_{j\beta}H
===========

\sum_{\alpha}
Q_{\beta\alpha}
U_g
D_{i\alpha}H
U_g^\dagger
]

---

# 23. Decisiones técnicas recomendadas

## 23.1. `spglib` como backend principal

Recomendado.

Motivo:

* especializado en simetría cristalina;
* devuelve operaciones;
* devuelve átomos equivalentes;
* ampliamente usado.

---

## 23.2. `pymatgen` como backend opcional

`pymatgen` puede ser útil si ya se usan sus objetos `Structure`; `SpacegroupAnalyzer` usa `spglib` y expone tolerancias como `symprec` y `angle_tolerance`.

Pero para minimizar dependencias, empezaría con `spglib` directamente.

---

## 23.3. `sisl` para HSX/TSHS

Recomendado para inspección de geometría, base y Hamiltoniano.

`sisl` expone lectores de HSX con métodos para leer geometría, base, Hamiltoniano, solapamiento y red.

---

# 24. Política de compatibilidad

## 24.1. Dependencias opcionales

Si `spglib` no está instalado:

```text
--symmetry-mode off       funciona
--symmetry-mode report    error claro
--symmetry-mode atoms     error claro
--symmetry-mode reconstruct error claro
```

Mensaje:

```text
ERROR: spglib is required for symmetry-aware stencil generation.
Install with: pip install spglib
```

---

## 24.2. Si `sisl` no está instalado

Debe permitir:

```text
--symmetry-mode report
--symmetry-mode atoms
```

pero no:

```text
--symmetry-mode reconstruct
```

Mensaje:

```text
ERROR: sisl is required for Hamiltonian symmetry reconstruction.
Use --symmetry-mode atoms or install sisl.
```

---

# 25. Reglas de activación recomendadas

## 25.1. Activar automáticamente solo reporte

Seguro.

---

## 25.2. No activar reconstrucción por defecto

La reconstrucción debe ser explícita:

```bash
--symmetry-mode reconstruct
```

---

## 25.3. Fallback automático

Si el modo no es estricto:

```text
reconstruct -> atoms -> off
```

según lo que sea seguro.

---

# 26. Validación de tolerancias

## 26.1. Barrido de `symprec`

Añadir script:

```bash
python Comparison/scripts/scan_symmetry_tolerance.py \
  --input base.fdf \
  --symprec-values 1e-5 1e-4 1e-3 1e-2
```

Salida:

```text
symprec     SG      n_ineq     operations
1e-5        P1      64         1
1e-4        P1      64         1
1e-3        Fm-3m   2          192
1e-2        Fm-3m   2          192
```

Si el grupo espacial cambia bruscamente con tolerancia, advertir:

```text
WARNING: symmetry detection is tolerance-sensitive.
Do not use reconstruction without brute-force validation.
```

---

# 27. Checklist de implementación

## 27.1. Checklist de código

* [ ] Añadir `symmetry_utils.py`.
* [ ] Añadir dataclasses de simetría.
* [ ] Añadir detección con `spglib`.
* [ ] Añadir construcción de `atom_map`.
* [ ] Añadir conversión frac/cart.
* [ ] Añadir reporte.
* [ ] Añadir flags CLI.
* [ ] Añadir manifest v2.
* [ ] Añadir modo `atoms`.
* [ ] Añadir tests de modo legacy.
* [ ] Añadir tests de simetría geométrica.
* [ ] Añadir módulo `hamiltonian_symmetry.py`.
* [ ] Añadir lectura de base.
* [ ] Añadir (U_g) para base (s).
* [ ] Añadir reconstrucción.
* [ ] Añadir validación contra brute force.
* [ ] Añadir documentación.
* [ ] Añadir ejemplos.

---

## 27.2. Checklist físico

* [ ] La operación (g) realmente deja invariante la estructura.
* [ ] Los átomos equivalentes tienen misma especie.
* [ ] La base orbital de átomos equivalentes coincide.
* [ ] El eje desplazado se transforma correctamente.
* [ ] El Hamiltoniano se transforma con (U_g H U_g^\dagger).
* [ ] Los offsets periódicos están correctamente mapeados.
* [ ] El modo espín es compatible.
* [ ] La reconstrucción reproduce brute force.
* [ ] La tolerancia de simetría es estable.
* [ ] El SCF no rompe la simetría numéricamente.

---

# 28. Recomendación final de implementación

La ruta más robusta es:

```text
1. Implementar --symmetry-mode report
2. Implementar --symmetry-mode atoms
3. Guardar manifest v2 con simetría completa
4. Validar reducción de número de stencils
5. Implementar reconstrucción solo para casos seguros
6. Comparar contra brute force
7. Extender progresivamente a orbitales p/d/f y offsets periódicos
```

No intentaría implementar directamente el caso general completo, porque el riesgo principal no está en detectar átomos equivalentes, sino en reconstruir correctamente el Hamiltoniano.

La detección geométrica es relativamente sencilla con `spglib`.

La parte difícil es:

[
U_g
D_{i\alpha}H
U_g^\dagger
]

porque (U_g) depende de la base orbital, del formato HSX/TSHS, de los offsets periódicos y del tratamiento de espín.

---

# 29. Veredicto técnico

La optimización es físicamente correcta y potencialmente muy rentable.

El mínimo viable útil sería:

```text
--symmetry-mode report
--symmetry-mode atoms
```

Esto ya permitiría saber cuánta redundancia hay y generar solo stencils irreducibles.

El objetivo completo sería:

```text
--symmetry-mode reconstruct
```

pero debe desarrollarse con mucho cuidado.

La fórmula que debe guiar toda la implementación es:

[
D_{j\beta}H
===========

\sum_{\alpha}
Q_{\beta\alpha}
U_g
D_{i\alpha}H
U_g^\dagger
]

donde:

* (i) es el átomo representante;
* (j) es el átomo equivalente;
* (\alpha) es el eje del representante;
* (\beta) es el eje del átomo objetivo;
* (Q) transforma direcciones cartesianas;
* (U_g) transforma la base orbital del Hamiltoniano.

La implementación debe avanzar desde simetría geométrica simple hacia reconstrucción orbital completa, siempre con validación contra el cálculo brute-force original.
