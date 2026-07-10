# GOAL: corregir y validar end-to-end la mezcla de datasets small/large y las derivadas autograd de Graph2Mat y DeepH

## Rol

Actúa como un agente senior especializado simultáneamente en:

* ingeniería de software científico;
* PyTorch y diferenciación automática;
* Graph2Mat;
* DeepH;
* SIESTA;
* Hamiltonianos electrónicos;
* datasets de dinámica molecular;
* validación matemática de derivadas;
* reproducibilidad computacional;
* diseño experimental y análisis estadístico.

Tu tarea no es producir un parche superficial. Debes **implementar, probar y documentar todas las correcciones confirmadas** derivadas de la auditoría del intervalo:

```text
f3ec12ece5722fe5fda5113fe4a691b419d71924..HEAD
```

del repositorio principal.

Debes trabajar de extremo a extremo sobre los repositorios locales relacionados, porque parte de la funcionalidad reside fuera del repositorio principal.

---

# 1. Repositorios y rutas de trabajo

Repositorios locales esperados:

```text
/home/christian/repositorios/MD_vs_AtomicDisplacement
/home/christian/repositorios/DeepH-pack
/home/christian/repositorios/graph2mat
```

Repositorio principal:

```text
/home/christian/repositorios/MD_vs_AtomicDisplacement
```

Repositorio de DeepH:

```text
/home/christian/repositorios/DeepH-pack
```

Repositorio de Graph2Mat:

```text
/home/christian/repositorios/graph2mat
```

No presupongas que:

* las ramas locales coinciden con GitHub;
* `main` contiene los últimos cambios locales;
* los entornos Python importan el checkout que estás inspeccionando;
* los repositorios están limpios;
* los commits descritos en esta tarea siguen siendo los `HEAD` actuales.

Como hipótesis inicial, que debes verificar y no asumir:

* `MD_vs_AtomicDisplacement` podría estar en `d318394f6cd8569d4606cd92523e7a0ff2d0ddf2`;
* el checkout local de `DeepH-pack` podría contener `4fd2f435d09a73194731200a12fa4a37738586fb`;
* la rama remota `main` de DeepH podría no contener todavía ese commit;
* el entorno DeepH podría importar un checkout distinto del inspeccionado.

---

# 2. Restricciones operativas

## 2.1. Seguridad Git

Antes de modificar nada:

1. Ejecuta en los tres repositorios:

```bash
git status --short
git rev-parse HEAD
git rev-parse --abbrev-ref HEAD
git log -5 --oneline
git remote -v
```

2. No ejecutes:

```bash
git reset --hard
git clean -fd
git checkout -- .
git restore .
```

3. No elimines ni sobrescribas cambios locales del usuario.

4. No hagas `push`.

5. No hagas merge automático contra ramas remotas.

6. Si existen cambios locales no relacionados:

   * consérvalos;
   * evita modificar las mismas líneas cuando sea posible;
   * documenta cualquier conflicto;
   * no abortes toda la tarea salvo que sea imposible continuar con seguridad.

7. Puedes crear commits locales pequeños y coherentes por fase, pero:

   * no es obligatorio;
   * no debes incluir cambios ajenos;
   * debes reportar exactamente qué commits has creado.

## 2.2. Coste computacional

No ejecutes:

* campañas largas de SIESTA;
* entrenamientos completos;
* sweeps reales de cientos de snapshots;
* derivadas de todas las estructuras de 50 átomos;
* test suites completas antes de que pasen los tests dirigidos.

Sí puedes ejecutar:

* tests unitarios;
* tests sintéticos;
* validaciones de configuración;
* materializaciones pequeñas en `tmp_path`;
* una inferencia sobre una estructura mínima ya procesada;
* un smoke real con un checkpoint existente;
* diferencias finitas sobre una inferencia ML, sin SIESTA;
* tests SIESTA únicamente si existe un fixture mínimo ya preparado y su coste es reducido.

## 2.3. Regla de rigor

No implementes una corrección únicamente porque la auditoría la afirmaba.

Para cada hallazgo:

1. verifica el estado actual;
2. clasifícalo como:

   * confirmado;
   * ya corregido;
   * parcialmente corregido;
   * falso para el checkout actual;
   * no verificable;
3. implementa únicamente lo necesario para el estado real;
4. añade una prueba que impida la regresión.

---

# 3. Resultado final requerido

Al terminar deben existir:

1. Correcciones funcionales en los repositorios correspondientes.
2. Tests unitarios y de integración dirigidos.
3. Validación reproducible de Graph2Mat autograd.
4. Validación reproducible de DeepH autograd.
5. Contrato de versiones y capacidades entre repositorios.
6. Materialización de datasets fail-closed.
7. Splits sin leakage y con alcance de evaluación explícito.
8. Métricas que distingan estructuras pequeñas y grandes.
9. Composición efectiva del dataset, no solo ratios nominales.
10. Política de ponderación de pérdidas explícita.
11. Protección contra artefactos stale.
12. UI actualizada.
13. Documentación técnica.
14. Informe final de implementación.

El repositorio no debe declarar ningún resultado como científicamente válido si falta alguna precondición necesaria.

---

# 4. Fase 0 — Inventario reproducible de los tres repositorios

## 4.1. Identificar el código que realmente se ejecuta

En los entornos usados por el proyecto, obtén:

```bash
python - <<'PY'
import inspect
import sys

print("python:", sys.executable)

try:
    import deeph
    print("deeph module:", inspect.getfile(deeph))
except Exception as exc:
    print("deeph import failed:", repr(exc))

try:
    import graph2mat
    print("graph2mat module:", inspect.getfile(graph2mat))
except Exception as exc:
    print("graph2mat import failed:", repr(exc))
PY
```

Repite la comprobación con:

* el Python del entorno principal;
* el Python configurado para DeepH;
* el Python configurado para Graph2Mat, si son distintos.

Debes demostrar qué checkout se importa realmente.

## 4.2. Crear un inventario de ejecución

Implementa o amplía una utilidad común que produzca una estructura similar a:

```json
{
  "repositories": {
    "MD_vs_AtomicDisplacement": {
      "path": "...",
      "commit": "...",
      "branch": "...",
      "dirty": false
    },
    "graph2mat": {
      "path": "...",
      "commit": "...",
      "branch": "...",
      "dirty": false
    },
    "DeepH-pack": {
      "path": "...",
      "commit": "...",
      "branch": "...",
      "dirty": false
    }
  },
  "python": {
    "executable": "...",
    "version": "...",
    "torch_version": "...",
    "default_dtype": "float32"
  },
  "imports": {
    "graph2mat_module_path": "...",
    "deeph_module_path": "..."
  }
}
```

Reutiliza, cuando sea apropiado, la funcionalidad existente que obtiene:

* SHA;
* rama;
* estado dirty.

No dupliques lógica innecesariamente.

## 4.3. Persistencia obligatoria

Incluye este inventario en:

* manifests de entrenamiento;
* manifests de derivadas;
* manifests de mezcla;
* reports de validación;
* outputs de métricas;
* payloads entregados a la UI.

## 4.4. Gate de reproducibilidad

Define:

```text
reproducibility_status =
    pinned_clean
    pinned_dirty
    unpinned
    unavailable
```

Reglas:

* `pinned_clean`: SHA conocido y repo limpio.
* `pinned_dirty`: SHA conocido, pero existen cambios locales.
* `unpinned`: se ejecuta código sin SHA resoluble.
* `unavailable`: no se pudo determinar el origen.

Solo `pinned_clean` puede aspirar a un estado científico equivalente a `paper_ready`.

Los demás pueden ejecutarse en modo diagnóstico, pero deben mostrar advertencias visibles.

## Criterios de aceptación de la fase 0

* Los tres repositorios aparecen en el manifest.
* Se registra la ruta real del módulo importado.
* Se detecta si el Python está importando otro checkout.
* Los tests cubren:

  * repo limpio;
  * repo dirty;
  * directorio sin `.git`;
  * módulo importado desde ruta inesperada.
* Ningún workflow científico puede omitir silenciosamente esta información.

---

# 5. Fase 1 — Contrato de capacidades de DeepH autograd

La auditoría original confundió dos versiones de `DeepH-pack`:

* una versión histórica con placeholder `NaN`;
* una versión posterior con JVP forward-mode.

La solución no es codificar un SHA concreto como única fuente de verdad. Debes implementar un **preflight de capacidades**.

## 5.1. Inspección del estado actual

Revisa al menos:

```text
DeepH-pack/deeph/inference/pred_ham.py
DeepH-pack/deeph/scripts/inference.py
DeepH-pack/deeph/default.ini
MD_vs_AtomicDisplacement/Comparison/scripts/deeph_config.py
MD_vs_AtomicDisplacement/Comparison/scripts/run_deeph_autograd_derivative_predictions.py
MD_vs_AtomicDisplacement/Comparison/scripts/run_hamiltonian_derivative_predictions.py
MD_vs_AtomicDisplacement/Comparison/scripts/g2m_deeph_runner.py
```

Determina:

* si existe `_forward_ad_jvp_blocks`;
* si `predict_with_grad` acepta selección de átomos y ejes;
* si el CLI lee `grad_atom_indices`;
* si el CLI lee `grad_axis_indices`;
* si se ejecuta realmente `fwAD.make_dual`;
* si se extrae la tangente con `fwAD.unpack_dual`;
* si las direcciones solicitadas llegan desde el runner hasta el backend;
* si se calculan todas las direcciones o solo las solicitadas;
* si existe soporte spinful real;
* qué dtype se usa.

## 5.2. Portar o conservar la implementación real

Si el checkout actual ya contiene una implementación equivalente a `4fd2f435`:

* no la reescribas desde cero;
* conserva su lógica;
* corrige únicamente sus defectos pendientes;
* añade tests y contrato de capacidades.

Si no la contiene:

* implementa el forward-mode JVP de manera equivalente;
* no copies ciegamente un diff si el código ha divergido;
* integra la solución según la arquitectura actual.

## 5.3. Capability manifest

Añade una función de introspección y un schema estable:

```json
{
  "schema": "deeph_autograd_capability_v1",
  "available": true,
  "implementation": "torch_forward_ad_jvp",
  "supports_atom_subset": true,
  "supports_axis_subset": true,
  "supports_spinful": false,
  "supports_fixed_topology": true,
  "output_schema": "hamiltonians_grad_pred_v2",
  "dtype": "float64",
  "finite_output_required": true
}
```

No confíes únicamente en nombres de funciones. El preflight debe verificar:

* firma compatible;
* claves de configuración;
* schema producido;
* al menos un smoke sintético real del helper JVP.

Si la capacidad no existe:

* falla de forma explícita antes de lanzar inferencia;
* no generes outputs placeholder;
* no continúes hacia métricas;
* registra `capability_unavailable`.

## 5.4. Direcciones no calculadas

La implementación actual puede inicializar todo el tensor a cero y rellenar solo las direcciones solicitadas.

Eso es ambiguo: un cero puede significar:

* derivada calculada e igual a cero;
* dirección nunca calculada.

Corrígelo mediante una de estas soluciones, por orden de preferencia:

### Opción A — Dataset por dirección

Guardar únicamente las direcciones calculadas, con claves o grupos explícitos:

```text
/block_key/atom_0/axis_0
/block_key/atom_0/axis_1
```

### Opción B — Tensor más máscara

Guardar:

```text
gradient
computed_mask
computed_atom_indices
computed_axis_indices
```

donde:

```python
computed_mask[atom, axis] == True
```

solo para direcciones realmente calculadas.

### Opción C — NaN exclusivamente como sentinel

Las direcciones no calculadas pueden quedar como `NaN`, pero:

* las direcciones calculadas deben ser finitas;
* el schema debe distinguir placeholder no calculado de error;
* el runner debe rechazar cualquier `NaN` en una dirección solicitada.

No mantengas ceros silenciosos en direcciones no calculadas.

## 5.5. Validación de finitud

Antes de marcar una derivada DeepH como `predicted`:

```python
assert requested_values.size > 0
assert np.all(np.isfinite(requested_values))
```

Comprueba también:

* forma esperada;
* dtype;
* correspondencia de átomo y eje;
* presencia de máscara;
* número de bloques;
* ninguna dirección solicitada ausente.

Si falla:

```text
status = error
error_code = deeph_autograd_non_finite
```

No escribas metadata que la presente como resultado válido.

## 5.6. Soporte spinful

Si el código solo es correcto para `spinful=False`:

* reemplaza `assert` por una excepción explícita;
* declara `supports_spinful=false`;
* falla antes del cálculo;
* añade test.

No reserves formas spinful que todavía no estén implementadas.

## 5.7. Tests DeepH mínimos

Añade tests en `DeepH-pack` para:

### Helper JVP analítico

Una función de bloques sencilla:

[
H(\mathbf R)=
\begin{pmatrix}
x^2 & xy\
xy & \sin z
\end{pmatrix}
]

y comprobar sus JVP respecto a varias direcciones.

### Selección parcial

Solicitar:

```text
atom_indices = [0]
axis_indices = [1]
```

y verificar:

* solo esa dirección aparece como calculada;
* las demás no pueden interpretarse como derivadas válidas.

### Finitud

Simular una tangente `NaN` y comprobar que el workflow falla.

### Configuración

Verificar que el CLI transporta:

```text
grad_atom_indices
grad_axis_indices
```

hasta `predict_with_grad`.

### Spinful

Verificar fallo explícito cuando no está soportado.

## Criterios de aceptación de la fase 1

* Una versión con placeholder no puede superar el capability preflight.
* Una versión con JVP sí lo supera.
* Las direcciones no calculadas no se representan como ceros válidos.
* Todo valor solicitado es finito.
* El runner no puede declarar éxito si el backend no calculó la dirección solicitada.
* Los tests no dependen únicamente de mocks de archivos HDF5.

---

# 6. Fase 2 — Validación numérica de DeepH autograd

La existencia de JVP no demuestra su corrección científica.

Debes validar:

[
D^{\mathrm{autograd}}*{\mathrm{DeepH}}
\approx
D^{\mathrm{FD}}*{\mathrm{DeepH}}(\delta).
]

## 6.1. Ruta de finite differences del propio modelo

Implementa una utilidad que:

1. toma una estructura base;
2. selecciona átomo (I) y eje (\alpha);
3. evalúa DeepH en:

   * (R_{I\alpha}+\delta);
   * (R_{I\alpha}-\delta);
4. reconstruye ambos Hamiltonianos en exactamente la misma representación;
5. calcula:

[
D^{FD}_{\mathrm{DeepH}}
=======================

\frac{
H_{\mathrm{DeepH}}(R+\delta)
----------------------------

H_{\mathrm{DeepH}}(R-\delta)
}{
2\delta
}.
]

No uses SIESTA en esta validación.

## 6.2. Delta sweep

Soporta al menos:

```text
0.02 Å
0.01 Å
0.005 Å
0.0025 Å
```

Permite configuración adicional.

Reporta:

* MAE;
* RMSE;
* Frobenius absoluto;
* Frobenius relativo;
* error máximo;
* cosine similarity;
* error por bloque;
* porcentaje de elementos finitos;
* soporte discrepante;
* tiempo autograd;
* tiempo FD.

## 6.3. Dtype

Ejecuta la validación al menos en:

* `float64`;
* dtype usado en producción, probablemente `float32`.

No hardcodees como verdad científica un error histórico concreto. Reprodúcelo.

El report debe incluir:

```json
{
  "dtype": "float32",
  "delta_ang": 0.005,
  "relative_frobenius": 0.0,
  "cosine": 0.0,
  "max_abs": 0.0
}
```

## 6.4. Gates

Crea dos clases de gate:

### Gate matemático sintético

Debe usar tolerancias estrictas y deterministas.

### Gate real de checkpoint

Debe ser configurable y diferenciar:

```text
pass
warning
fail
not_run
```

No declares automáticamente `paper_ready` basándote en una única (\delta).

Exige:

* valores finitos;
* al menos tres deltas;
* existencia de una región razonablemente estable;
* coherencia de signo;
* cosine similarity alta;
* error relativo dentro del umbral configurado.

## 6.5. Topología

Registra si:

* la lista de vecinos de la base;
* la de (+\delta);
* la de (-\delta);

son idénticas.

Si cambia la topología:

```text
comparison_status = topology_changed
```

y no uses ese punto para validar el JVP local de topología fija.

## Criterios de aceptación de la fase 2

* Existe una comparación DeepH-autograd vs DeepH-FD real.
* Se ejecuta sin SIESTA.
* Se comparan varios deltas y dtypes.
* Se detectan cambios de topología.
* El resultado queda en JSON y CSV.
* Una comparación no ejecutada no se marca como superada.

---

# 7. Fase 3 — Alineación de DeepH con SIESTA

Una derivada DeepH puede ser matemáticamente correcta respecto a su propio output y seguir sin ser comparable con SIESTA por:

* orden orbital;
* signos de orbitales;
* bases locales;
* orientación de bloques;
* índices de supercelda;
* layout sparse.

## 7.1. Validar primero el Hamiltoniano base

Antes de comparar derivadas, compara:

[
H_{\mathrm{DeepH}}(R)
\quad\text{con}\quad
H_{\mathrm{SIESTA}}(R).
]

La ruta debe verificar:

* forma;
* número de orbitales;
* orden orbital;
* transformación de permutación;
* signos;
* vectores de traslación;
* bloques on-site;
* bloques off-site;
* soporte;
* orientación (R,i,j);
* unidades.

Reutiliza cuando sea correcto:

```text
derive_deeph_to_siesta_basis_transform
reconstruct_deeph_sparse_layout_prediction
```

pero no presupongas que una reconstrucción sin excepción demuestra equivalencia.

## 7.2. Gate de equivalencia

Define:

```text
deeph_base_equivalence_status =
    proven
    diagnostic_only
    failed
    unavailable
```

Para `proven`, exige:

* transformación explícita;
* hashes del `ORB_INDX`;
* hashes de `orbital_types.dat`;
* forma correcta;
* soporte razonable;
* comparación numérica válida;
* ausencia de bloques perdidos no explicados.

## 7.3. Derivada

La misma transformación aplicada al Hamiltoniano base debe aplicarse a:

[
\frac{\partial H}{\partial R}.
]

Registra el hash de la transformación para demostrar que base y derivada usan exactamente el mismo mapa.

## 7.4. Claims

Si la equivalencia base no está probada:

* permite resultados diagnósticos;
* prohíbe claims de precisión contra SIESTA;
* marca las métricas como `diagnostic_only`;
* muéstralo en la UI.

## Criterios de aceptación de la fase 3

* La derivada no puede declararse comparable si el Hamiltoniano base no lo es.
* Base y derivada comparten transformación y hashes.
* El report explica cualquier bloque descartado.
* Los outputs `diagnostic_only` no se confunden con resultados científicos.

---

# 8. Fase 4 — Fortalecer Graph2Mat autograd

La ruta Graph2Mat parece calcular el Jacobiano completo mediante VJP vectorizadas. No la reemplaces sin necesidad.

Revisa:

```text
Comparison/scripts/graph2mat_autograd_derivatives.py
Comparison/scripts/run_graph2mat_autograd_derivative_predictions.py
Comparison/scripts/run_hamiltonian_derivative_predictions.py
```

## 8.1. Confirmar Jacobiano completo

Mantén tests que demuestren que:

```text
jacobian.shape == [n_outputs, n_atoms, 3]
```

y que no se calcula simplemente:

```python
grad(H.sum(), positions)
```

## 8.2. Validación contra finite differences del modelo

Implementa la misma validación que para DeepH:

[
D^{\mathrm{autograd}}*{\mathrm{Graph2Mat}}
\approx
D^{FD}*{\mathrm{Graph2Mat}}(\delta).
]

Usa:

* estructura base;
* mismo checkpoint;
* misma base;
* misma reconstrucción matricial;
* varios deltas;
* topología comprobada.

## 8.3. Invariancias físicas

Añade tests y métricas para:

### Traslación global

[
\sum_I
\frac{\partial H_{\mu\nu}}
{\partial R_{I\alpha}}
\approx 0.
]

Reporta:

* máximo absoluto;
* Frobenius;
* versión relativa.

### Permutación

Permuta átomos equivalentes y comprueba que:

* los índices atómicos;
* bloques;
* orbitales;

se transformen consistentemente.

### Hermiticidad en espacio real

No uses únicamente:

[
H=H^\dagger
]

sobre una matriz rectangular de supercelda.

Comprueba por bloques:

[
D_{ij}(\mathbf R)
=================

D_{ji}^{\dagger}(-\mathbf R).
]

### Rotación

Añade al menos un test sintético o pequeño que compruebe coherencia del cambio de base y de los ejes cartesianos.

## 8.4. CPU/CUDA

Si CUDA no está soportado por el backward vectorizado actual:

* mantén el fallo explícito;
* decláralo en capabilities;
* no intentes fallback silencioso;
* documenta el coste CPU.

## Criterios de aceptación de la fase 4

* Graph2Mat autograd supera finite differences del propio modelo.
* Se registra estabilidad respecto a delta.
* Existen métricas de traslación y Hermiticidad por bloques.
* El frame cartesiano queda probado mediante tests.

---

# 9. Fase 5 — Firma de artefactos y eliminación de resultados stale

Las rutas actuales pueden reutilizar un `.npz` porque existe, sin garantizar que corresponda al mismo:

* checkpoint;
* código;
* estructura;
* base;
* método;
* dtype;
* topología.

## 9.1. Firma de entrada

Implementa una firma determinista que incluya como mínimo:

```json
{
  "model": "graph2mat | deeph",
  "checkpoint_sha256": "...",
  "checkpoint_config_sha256": "...",
  "repository_commits": {},
  "repository_dirty_states": {},
  "structure_fdf_sha256": "...",
  "structure_coordinates_sha256": "...",
  "basis_sha256": {},
  "orbital_ordering_sha256": "...",
  "neighbor_list_sha256": "...",
  "matrix_component_policy": "...",
  "derivative_method": "...",
  "dtype": "...",
  "atom_index": 0,
  "axis_index": 0,
  "topology_fixed": true
}
```

Calcula:

```text
input_signature_sha256
```

## 9.2. Reutilización

`skip_if_exists` solo puede reutilizar un resultado si:

* existe metadata;
* la firma coincide;
* el archivo se puede leer;
* la forma coincide;
* los valores solicitados son finitos;
* el schema es compatible.

En cualquier otro caso:

* recalcula;
* o falla si el usuario desactivó overwrite.

## 9.3. Migración

Los resultados antiguos sin firma:

```text
cache_status = legacy_unverified
```

No deben reutilizarse automáticamente en una campaña científica.

## Criterios de aceptación de la fase 5

* Cambiar el checkpoint invalida el cache.
* Cambiar una coordenada invalida el cache.
* Cambiar el dtype invalida el cache.
* Cambiar el código importado invalida el cache.
* Un `.npz` sin metadata no se acepta como válido.

---

# 10. Fase 6 — Compatibilidad física de datasets small/large

Revisa:

```text
Comparison/scripts/ml_vs_siesta/mixed_dataset_materialize.py
Comparison/scripts/ml_vs_siesta/dataset_mixing.py
Comparison/scripts/ml_vs_siesta/mixing_sweep.py
shared/benchmark_manifest.py
materials/graphene/RUN.fdf
materials/graphene_5x5/RUN.fdf
```

## 10.1. Distinguir especies declaradas y especies activas

No concluyas que una especie ghost participa únicamente porque aparece en:

```text
ChemicalSpeciesLabel
PAO.Basis
```

Determina:

* si existen átomos con esa especie en `AtomicCoordinatesAndAtomicSpecies`;
* si aparecen orbitales asociados en `ORB_INDX`;
* si aparecen bloques de esa especie en el Hamiltoniano;
* si Graph2Mat o DeepH la incluyen en su representación.

Implementa:

```text
declared_species
active_atomic_species
active_orbital_species
target_block_species
```

## 10.2. Estado de compatibilidad ghost

Sustituye la excepción binaria no verificada por:

```text
ghost_compatibility_status =
    not_applicable
    proven_inactive
    proven_compatible
    unproven
    incompatible
```

Reglas:

* `not_applicable`: no hay ghost en ninguna fuente.
* `proven_inactive`: declarada, pero sin átomos, orbitales o bloques activos.
* `proven_compatible`: activa en ambas con representación equivalente.
* `unproven`: no hay evidencia suficiente.
* `incompatible`: cambia el espacio objetivo.

Solo los tres primeros permiten materialización estricta.

No conviertas un simple booleano manual en una prueba física.

## 10.3. Fingerprint del target

Compara:

* número de orbitales por especie;
* orden orbital;
* hashes de bases;
* dimensión por átomo;
* spin;
* matrix component policy;
* target Hamiltonian type;
* layout sparse;
* convenciones de bloques;
* versión de SIESTA;
* pseudopotenciales.

Las dimensiones globales pueden diferir entre 2 y 50 átomos. Lo que debe ser compatible es la representación local y orbital.

## 10.4. Fingerprint DFT

Registra y compara:

* `XC.functional`;
* `XC.authors`;
* `MeshCutoff`;
* `ElectronicTemperature`;
* tolerancia SCF;
* pseudopotenciales;
* base;
* spin;
* periodicidad;
* espesor de vacío;
* orientación de red;
* densidad de k-points.

No compares el k-grid únicamente como tupla bruta.

Para celdas de distinto tamaño, compara la separación recíproca aproximada:

[
\Delta k_i
\sim
\frac{\lVert \mathbf b_i\rVert}{N_i}.
]

Una primitiva `20×20×1` y una supercelda `5×5` con `4×4×1` pueden representar densidades equivalentes.

## 10.5. Sampling y target

Distingue:

### Incompatibilidad del target

Debe bloquear.

### Diferencia deliberada de sampling

Por ejemplo:

* temperatura;
* amplitud de desplazamientos;
* trayectorias;
* semillas.

Debe registrarse, pero no necesariamente bloquear.

## 10.6. Report

Genera:

```text
dataset_compatibility_report.json
```

con:

```json
{
  "compatible": true,
  "target_compatibility": "proven",
  "dft_compatibility": "proven",
  "ghost_compatibility": "proven_inactive",
  "sampling_differences": [],
  "blocking_errors": [],
  "warnings": []
}
```

## Criterios de aceptación de la fase 6

* No se necesita un override manual cuando la especie ghost está demostrablemente inactiva.
* Una especie ghost activa e incompatible bloquea.
* Bases o pseudopotenciales incompatibles bloquean.
* La equivalencia de k-density acepta correctamente primitiva `20×20` frente a supercelda `4×4`.
* El report es trazable a artefactos reales.

---

# 11. Fase 7 — Splits, leakage y alcance de evaluación

El test actual puede contener únicamente estructuras pequeñas. Eso puede ser válido para una pregunta específica, pero no para afirmar rendimiento general sobre ambos dominios.

## 11.1. Separar conjunto entrenable y conjunto de evaluación

Los ratios de mezcla deben aplicarse al **training pool**, no al conjunto total materializado que incluye test.

Define y reporta por separado:

```text
requested_train_size
actual_train_size
validation_size
test_small_size
test_large_size
materialized_total_size
```

El eje “dataset size” debe significar por defecto:

```text
actual_train_size
```

No debe incluir test fijo.

## 11.2. Preservar procedencia

Cada snapshot combinado debe conservar:

```text
origin
source_root
source_dataset_id
source_sample_id
source_split
source_trajectory_id
source_frame_index
temperature_K
time_fs
source_seed
source_artifact_hashes
```

No pierdas el split original.

## 11.3. Excluir tests de origen

Por defecto:

* un snapshot marcado como test en la fuente no puede acabar en train;
* tampoco puede acabar en validation;
* aplica a small y large.

Añade una opción explícita de migración solo para casos excepcionales y márcala como no científica.

## 11.4. Nuevas políticas

Implementa al menos:

### `fixed_stratified_test`

Política recomendada.

Reserva:

```text
test_small
test_large
```

fijos entre:

* ratios;
* modos;
* seeds de entrenamiento;
* modelos.

### `fixed_common_test_small_only`

Política válida únicamente para estudiar:

> efecto de añadir datos grandes sobre el rendimiento en el dominio pequeño.

Debe etiquetarse así en manifests, plots y UI.

### `resplit_combined`

Mantener como legacy/exploratorio.

Debe mostrar una advertencia porque el test cambia entre composiciones.

## 11.5. Selección temporal

Cuando no exista un test congelado:

* agrupa por trayectoria;
* conserva orden temporal;
* usa colas o bloques;
* incluye gap temporal;
* no interpoles aleatoriamente frames vecinos entre train y test.

No uses únicamente el sufijo numérico global si hay varias temperaturas o trayectorias.

## 11.6. Seeds

Distingue:

```text
selection_seed
training_seed
model_initialization_seed
data_loader_seed
```

Una seed de entrenamiento no debe modificar el test.

## Criterios de aceptación de la fase 7

* Ningún test de origen entra en train.
* `fixed_stratified_test` contiene small y large.
* Los tests son idénticos entre ratios y seeds.
* El tamaño del eje x representa train real.
* La UI identifica el alcance de evaluación.
* Existen tests de no leakage.

---

# 12. Fase 8 — Semántica de ratios y composición efectiva

## 12.1. Ratio nominal

Mantén:

```text
requested_ratio
ratio_semantics
mode
```

## 12.2. Composición real

Calcula y persiste:

```text
n_small_train
n_large_train
actual_large_fraction_by_snapshots
small_atoms_total
large_atoms_total
actual_large_fraction_by_atoms
small_node_blocks
large_node_blocks
small_edge_blocks
large_edge_blocks
actual_large_fraction_by_blocks
small_matrix_elements
large_matrix_elements
actual_large_fraction_by_matrix_elements
large_capped
cap_reasons
```

## 12.3. Add

La semántica debe ser inequívoca.

Considera renombrar:

```text
ratio
```

por:

```text
large_pool_fraction_added
```

manteniendo compatibilidad con payloads anteriores.

## 12.4. Replace

El reemplazo debe mantener fijo:

```text
actual_train_size
```

Los tests fijos no deben formar parte del cálculo del reemplazo.

## 12.5. Redondeo

Registra:

```text
requested_count_float
rounding_policy
actual_count
```

Usa una política determinista.

## Criterios de aceptación de la fase 8

* Preview y materialización producen exactamente los mismos counts.
* Ratio nominal y ratio real aparecen juntos.
* `ratio=1` no se etiqueta como 100 % large si hubo cap.
* Los tests cubren pools desiguales y tamaños pequeños.

---

# 13. Fase 9 — Ponderación efectiva durante el entrenamiento

Una estructura de 50 átomos genera muchos más:

* nodos;
* aristas;
* bloques;
* elementos matriciales;

que una de 2 átomos.

La mezcla por snapshots no garantiza el mismo peso en la loss.

## 13.1. Medición previa

Antes de cambiar la loss, reconstruye para Graph2Mat y DeepH:

* unidad de reducción;
* asociación entre outputs y estructura;
* peso efectivo de cada estructura;
* peso de node y edge terms;
* efecto del batch size;
* efecto de máscaras.

Genera un test con:

* una estructura de 2 átomos;
* una estructura de 50 átomos;
* residuos controlados iguales;

y mide qué contribución tiene cada una.

## 13.2. Políticas configurables

Implementa una opción explícita:

```text
training_weighting_policy
```

con al menos:

### `legacy_elementwise`

Comportamiento anterior.

### `per_structure`

1. calcula la loss dentro de cada estructura;
2. normaliza por sus elementos válidos;
3. promedia las estructuras.

### `per_domain`

1. calcula loss por estructura;
2. obtiene media small;
3. obtiene media large;
4. promedia dominios con pesos configurables.

Soporta:

```text
small_domain_weight
large_domain_weight
```

## 13.3. Graph2Mat

Localiza la loss real y la asociación batch-output.

Preserva, cuando aplique, la distinción entre:

* node loss;
* edge loss;
* on-site;
* off-site.

No implementes `per_structure` promediando indiscriminadamente todos los outputs si eso altera la semántica de `block_type_mse`.

El resultado debe ser conceptualmente similar a:

[
L_s
===

w_n L_{s,\mathrm{node}}
+
w_e L_{s,\mathrm{edge}},
]

[
L
=

\frac{1}{N_s}
\sum_s L_s.
]

## 13.4. DeepH

Localiza:

* `MaskMSELoss`;
* edge-to-graph mapping;
* batch mapping;
* máscara de outputs.

Implementa reducción por grafo sin perder la máscara de orbitales válidos.

No sustituyas el problema por un sampler si la loss sigue dejando que una estructura grande domine internamente.

## 13.5. Backward compatibility

* Mantén `legacy_elementwise` para reproducir runs antiguos.
* Los payloads nuevos de mezcla deben usar una política explícita.
* No dependas de un default silencioso.
* Registra la política en manifests y plots.

## 13.6. Métricas del peso efectivo

Guarda:

```text
effective_training_weight_small
effective_training_weight_large
effective_large_fraction
```

medido según la política real.

## Criterios de aceptación de la fase 9

* Existe un test que demuestra el sesgo legacy.
* `per_structure` da el mismo peso a dos estructuras con el mismo error normalizado.
* `per_domain` respeta pesos small/large.
* Graph2Mat y DeepH registran su política.
* Los resultados de distintas políticas no se mezclan en una misma curva.

---

# 14. Fase 10 — Materialización fail-closed y transaccional

## 14.1. Validación

Después de copiar o enlazar snapshots:

```python
all_valid = all(snapshot["valid"] for snapshot in validation_snapshots)
```

Si `all_valid` es falso:

* no marques la materialización como completada;
* no lances entrenamiento;
* no escribas manifests finales que parezcan válidos.

## 14.2. Escritura transaccional

Materializa en:

```text
<output>.partial-<uuid>
```

Valida todo.

Solo después de superar la validación:

```text
rename parcial → output final
```

Ante error:

* conserva un report diagnóstico si es útil;
* elimina o marca claramente el parcial;
* nunca deja un dataset aparentemente completo.

## 14.3. Estados

Usa:

```text
planned
materializing
validated
failed_validation
ready
training
trained
partial
failed
```

## 14.4. Symlinks

Valida:

* destino existente;
* symlink no roto;
* archivo no vacío;
* artefactos requeridos;
* hashes consistentes.

## Criterios de aceptación de la fase 10

* Un snapshot inválido provoca fallo.
* No se lanza entrenamiento.
* No aparece `benchmark_ready=true`.
* No queda un output final parcialmente escrito.
* Los tests cubren fallo a mitad de materialización.

---

# 15. Fase 11 — Protocolo completo de derivadas

El protocolo debe distinguir tres comparaciones:

## A. Consistencia matemática del modelo

[
D^{autograd}*{ML}
\quad\text{vs}\quad
D^{FD}*{ML}(\delta)
]

## B. Error del modelo usando el mismo método numérico

[
D^{FD}*{ML}(\delta)
\quad\text{vs}\quad
D^{FD}*{SIESTA}(\delta)
]

## C. Resultado final

[
D^{autograd}*{ML}
\quad\text{vs}\quad
D^{FD}*{SIESTA}(\delta)
]

No combines estas tres comparaciones en una única métrica sin identificarlas.

## 15.1. SIESTA

Mantén stencils centrales:

[
D^{FD}_{SIESTA}
===============

\frac{H(R+\delta)-H(R-\delta)}{2\delta}.
]

Valida para cada operand:

* terminación normal;
* convergencia SCF;
* configuración idéntica;
* misma base;
* mismo orden orbital;
* misma forma;
* mismo soporte esperado;
* hashes de comparabilidad;
* desplazamiento real correcto.

## 15.2. Delta stability

Reporta por muestra:

* error vs delta;
* diferencia entre deltas sucesivos;
* estimación de plateau;
* posible ruido numérico;
* cambio de soporte;
* cambio de topología.

## 15.3. Métodos

Registra siempre:

```text
reference_derivative_method
predicted_derivative_method
predicted_delta_ang
reference_delta_ang
```

Autograd debe tener:

```text
predicted_delta_ang = null
```

## 15.4. Claim status

Define:

```text
invalid
diagnostic_only
validated_model_derivative
validated_against_siesta
paper_ready
```

### `validated_model_derivative`

Exige superar ML-autograd vs ML-FD.

### `validated_against_siesta`

Exige además:

* alineación de base;
* SIESTA convergido;
* hashes;
* unidades;
* comparación válida.

### `paper_ready`

Exige además:

* repos pinned y clean;
* protocolo fijado;
* test fijo;
* suficientes seeds;
* ausencia de advertencias bloqueantes.

## Criterios de aceptación de la fase 11

* Los tres tipos de comparación tienen outputs separados.
* La UI no confunde consistencia autograd con precisión física.
* La dependencia con (\delta) se atribuye correctamente a FD.
* Los claims se degradan automáticamente si falta evidencia.

---

# 16. Fase 12 — Métricas robustas para estructuras de distinto tamaño

## 16.1. Micro y macro

Calcula simultáneamente:

### Micro elementwise

Todos los elementos juntos.

### Macro snapshot

Métrica por snapshot y después media.

### Macro domain

Media separada en:

```text
small
large
```

y promedio de dominios.

## 16.2. Hamiltoniano

Reporta como mínimo:

```text
h_mae_micro_eV
h_rmse_micro_eV
h_mae_macro_snapshot_eV
h_rmse_macro_snapshot_eV
h_mae_small_eV
h_mae_large_eV
h_mae_macro_domain_eV
```

## 16.3. Derivadas

Reporta:

```text
dh_mae_micro_eV_per_Ang
dh_rmse_micro_eV_per_Ang
dh_mae_macro_snapshot_eV_per_Ang
dh_mae_small_eV_per_Ang
dh_mae_large_eV_per_Ang
dh_relative_frobenius
dh_normalized_frobenius_per_element
dh_cosine
dh_max_abs
```

## 16.4. Bloques

Separa cuando sea posible:

* on-site;
* off-site;
* por distancia;
* por orbital;
* por átomo desplazado;
* por eje;
* por tamaño estructural.

## 16.5. Frobenius

No uses Frobenius absoluto como métrica principal para comparar 2 y 50 átomos.

Incluye:

[
E_F^{rel}
=========

\frac{
|D_{\mathrm{pred}}-D_{\mathrm{ref}}|*F
}{
|D*{\mathrm{ref}}|_F+\epsilon
}
]

y:

[
E_F^{elem}
==========

\frac{
|D_{\mathrm{pred}}-D_{\mathrm{ref}}|*F
}{
\sqrt{N*{\mathrm{elements}}}
}.
]

## 16.6. Seeds

Para cada punto:

```text
mean
sample_std
n_seeds
confidence_interval
exploratory
```

No combines seeds de:

* distinta política de loss;
* distinto test;
* distinto dtype;
* distinto SHA;
* distinta definición de ratio.

## Criterios de aceptación de la fase 12

* Estructuras grandes no dominan todas las métricas reportadas.
* Se pueden ver small y large por separado.
* Las curvas contienen la política experimental completa.
* Métricas incompatibles no se agregan.

---

# 17. Fase 13 — Plots y payloads científicos

## 17.1. Mezcla

Implementa plots seleccionables frente a:

```text
actual_train_size
n_large_train
actual_large_fraction_by_snapshots
actual_large_fraction_by_atoms
actual_large_fraction_by_blocks
effective_large_training_weight
compute_cost
```

## 17.2. Curvas

Una curva debe estar identificada al menos por:

```text
model
mode
training_weighting_policy
evaluation_scope
metric_reduction
dtype
```

El ratio puede expresarse mediante color, facet o selector, pero no debe ocultar la composición real.

## 17.3. Tooltips o tabla

Muestra:

* requested ratio;
* actual small/large;
* cap;
* train/validation/test;
* test scope;
* seeds;
* repo SHAs;
* claim status;
* warnings.

## 17.4. Derivadas

Añade plots separados:

1. ML-autograd vs ML-FD por delta.
2. ML-FD vs SIESTA-FD por delta.
3. ML-autograd vs SIESTA-FD por delta.
4. Speedup.
5. Regla de suma traslacional.
6. Hermiticidad por bloques.
7. Error por átomo/eje.
8. Error por tamaño estructural.

## Criterios de aceptación de la fase 13

* Ningún plot etiqueta ratio nominal como composición real.
* El dominio del test es visible.
* Los resultados diagnósticos aparecen diferenciados.
* Los resultados fallidos no aparecen como puntos válidos.
* La UI muestra unidades.

---

# 18. Fase 14 — UI

Revisa:

```text
Comparison/scripts/pipeline_ui.py
Comparison/ui/index.html
Comparison/ui/app.js
Comparison/ui/styles.css
```

## 18.1. Panel de mezcla

Añade:

* selector de split policy;
* explicación inequívoca de cada política;
* selector de weighting policy;
* pesos por dominio;
* composición real;
* train size real;
* test small/large;
* cap reasons;
* compatibility status;
* warnings de ghost;
* estado de validación;
* repo SHAs.

## 18.2. DeepH capabilities

Muestra:

```text
DeepH autograd: available / unavailable
implementation: forward_ad_jvp
dtype
supports spinful
commit
dirty state
```

## 18.3. Claims

Usa badges diferenciados:

```text
INVALID
DIAGNOSTIC
MODEL-DERIVATIVE VALIDATED
SIESTA-COMPARABLE
PAPER-READY
```

No uses solo colores: incluye texto.

## 18.4. Errores

Muestra mensajes específicos para:

* capability ausente;
* output no finito;
* cache stale;
* dataset incompatible;
* snapshot inválido;
* leakage detectado;
* base alignment no probado;
* test scope small-only;
* seeds insuficientes.

## 18.5. Backward compatibility

Runs antiguos:

* deben seguir siendo visibles;
* deben marcarse como legacy;
* no deben recibir campos nuevos inventados;
* no deben presentarse como paper-ready.

## Criterios de aceptación de la fase 14

* La UI permite reconstruir qué se entrenó y evaluó.
* El usuario puede distinguir nominal de real.
* El estado DeepH depende del backend real.
* No se ocultan bloqueadores científicos.

---

# 19. Fase 15 — Payloads y configuraciones

Actualiza los payloads de mezcla existentes, incluyendo los de:

```text
20 / 50 / 80
100 / 500
```

Añade explícitamente:

```json
{
  "split_policy": "fixed_stratified_test",
  "evaluation_scope": "small_and_large",
  "training_weighting_policy": "per_structure",
  "selection_seeds": [0, 1, 2],
  "training_seeds": [0, 1, 2],
  "strict_dataset_validation": true,
  "strict_target_compatibility": true,
  "require_repository_provenance": true,
  "require_autograd_validation": true
}
```

No copies literalmente esos valores si la arquitectura actual requiere otra forma; conserva la intención.

## 19.1. Schema

Versiona los payloads:

```text
mixing_payload_schema_v2
derivative_payload_schema_v2
```

Implementa migración desde v1.

## 19.2. Validación previa

Antes de lanzar:

* valida claves;
* valida tipos;
* valida ratios;
* valida seeds;
* valida roots;
* valida modelos;
* valida capabilities;
* valida compatibility report;
* valida test policy.

## Criterios de aceptación de la fase 15

* Un payload inválido falla antes de materializar.
* Los payloads legacy se migran o se rechazan con explicación.
* No existen defaults científicos silenciosos.

---

# 20. Fase 16 — Tests

## 20.1. MD_vs_AtomicDisplacement

Ejecuta primero tests dirigidos similares a:

```bash
pytest -q \
  tests/test_ml_vs_siesta_mixing.py \
  tests/test_run_mixing_e2e_payload_once.py \
  tests/test_graph2mat_autograd_derivatives.py \
  tests/test_hamiltonian_derivative_direct_prediction.py \
  tests/test_run_deeph_autograd_derivative_predictions.py \
  tests/test_deeph_autograd_derivatives.py \
  tests/test_evaluate_hamiltonian_derivative_metrics.py \
  tests/test_g2m_deeph_derivative_gate_check.py \
  tests/test_g2m_deeph_derivative_ui_backend.py \
  tests/test_g2m_deeph_ui.py
```

Ajusta la lista a los archivos existentes.

## 20.2. DeepH-pack

Añade y ejecuta tests para:

* forward AD helper;
* selección de direcciones;
* máscara;
* finitud;
* dtype;
* CLI config;
* spinful fail-closed;
* central finite differences de un modelo sintético.

## 20.3. Graph2Mat

Localiza la suite correspondiente y añade tests para:

* reducción `per_structure`;
* reducción `per_domain`;
* Jacobiano completo;
* FD;
* traslación;
* Hermiticidad por bloques;
* firma de cache.

## 20.4. Integración barata

Crea un test cross-repo que:

1. detecte el checkout de DeepH;
2. lea capabilities;
3. genere una derivada sintética;
4. la consuma desde el runner principal;
5. verifique finitud, metadata y firma.

## 20.5. Tests que no son suficientes

No consideres prueba científica suficiente:

* que exista un archivo;
* que el comando devuelva cero;
* que una forma sea correcta;
* que no haya excepción;
* que un mock escriba un HDF5;
* que un valor sea distinto de `None`.

## 20.6. Suite amplia

Solo después de pasar los tests dirigidos:

* ejecuta una suite más amplia razonable;
* no ejecutes campañas costosas;
* documenta cualquier test omitido.

---

# 21. Fase 17 — Smoke real controlado

Usa recursos existentes, sin entrenar de nuevo, cuando estén disponibles.

## 21.1. Graph2Mat

Con un checkpoint existente:

* una estructura;
* un átomo;
* un eje;
* varios deltas;
* autograd vs FD.

## 21.2. DeepH

Con un checkpoint existente:

* una estructura;
* un átomo;
* un eje;
* dtype registrado;
* autograd vs FD;
* base H equivalence report.

## 21.3. Dataset mixing

Materializa un dataset pequeño de prueba:

```text
small: 4–8 snapshots
large: 4–8 snapshots
```

Comprueba:

* fixed stratified test;
* no leakage;
* composición real;
* fail-closed;
* provenance;
* UI payload.

No ejecutes entrenamiento largo. Puede usarse un launcher fake únicamente para probar orquestación, pero la validación física debe usar outputs reales o analíticos.

---

# 22. Fase 18 — Documentación

Actualiza al menos:

```text
README.md
docs/architecture.md
docs/workflows.md
docs/data_and_outputs.md
docs/known_limitations.md
docs/ml_vs_siesta_benchmark.md
docs/graph2mat_deeph_benchmark.md
Comparison/METRICS.md
```

Añade un documento específico:

```text
docs/mixing_and_autograd_validation_contract.md
```

Debe explicar:

1. Semántica de `add`.
2. Semántica de `replace`.
3. Ratio nominal vs composición real.
4. Train size vs total materializado.
5. Políticas de test.
6. Riesgo de weighting.
7. Políticas de loss.
8. Compatibilidad orbital.
9. Compatibilidad ghost.
10. Graph2Mat autograd.
11. DeepH forward-mode JVP.
12. Diferencia entre autograd y finite differences.
13. Tres comparaciones de derivadas.
14. Dtype.
15. Topología fija.
16. Claims y gates.
17. Versionado de repositorios.
18. Limitaciones pendientes.

---

# 23. Fase 19 — Informe final de implementación

Crea:

```text
docs/audit_corrections_implementation_report.md
```

Estructura obligatoria:

## 1. Estado inicial

* SHA de cada repo;
* ramas;
* dirty;
* paths importados;
* capability DeepH encontrada.

## 2. Revisión de la auditoría

Tabla:

| Hallazgo              | Estado real         | Acción           |
| --------------------- | ------------------- | ---------------- |
| DeepH placeholder NaN | dependiente del SHA | capability + JVP |
| Ghost compatibility   | ...                 | ...              |
| Invalid snapshots     | ...                 | ...              |

## 3. Cambios por repositorio

### MD_vs_AtomicDisplacement

### DeepH-pack

### graph2mat

## 4. Cambios matemáticos

## 5. Cambios físicos

## 6. Cambios en datasets

## 7. Cambios en métricas

## 8. Cambios en UI

## 9. Tests ejecutados

Para cada comando:

* comando;
* resultado;
* duración si está disponible;
* interpretación.

## 10. Tests no ejecutados

Explica por qué.

## 11. Resultados del smoke

## 12. Riesgos pendientes

## 13. Commits locales creados

## 14. Veredicto actualizado

Usa:

```text
correcto
correcto con limitaciones
parcialmente correcto
diagnóstico únicamente
bloqueado
```

para:

* mezcla add;
* mezcla replace;
* Graph2Mat autograd;
* DeepH autograd;
* SIESTA FD;
* métricas;
* UI;
* workflow end-to-end.

---

# 24. Política de ahorro de créditos sin sacrificar calidad

1. Empieza por los archivos nombrados.
2. Usa `rg` para localizar definiciones y consumidores.
3. No leas directorios completos indiscriminadamente.
4. Inspecciona primero firmas y tests.
5. Ejecuta tests dirigidos tras cada fase.
6. No ejecutes la suite completa repetidamente.
7. No abras outputs pesados salvo necesidad.
8. Reutiliza helpers existentes.
9. Evita refactors cosméticos.
10. No reformatees archivos no relacionados.
11. Mantén los diffs pequeños por fase.
12. No reimplementes una funcionalidad ya correcta.
13. Cada cambio debe estar conectado a:

    * un hallazgo;
    * un test;
    * un criterio de aceptación.

---

# 25. Orden de implementación obligatorio

Sigue este orden:

1. Inventario y SHAs.
2. Capability DeepH.
3. Finitud y máscara DeepH.
4. Validación DeepH-autograd vs DeepH-FD.
5. Alineación base DeepH-SIESTA.
6. Validación Graph2Mat-autograd vs Graph2Mat-FD.
7. Firma de artefactos.
8. Compatibilidad small/large.
9. Splits y leakage.
10. Composición efectiva.
11. Ponderación de pérdidas.
12. Materialización fail-closed.
13. Métricas.
14. Plots.
15. UI.
16. Payloads.
17. Documentación.
18. Tests amplios.
19. Informe final.

No empieces por la UI antes de estabilizar los contratos backend.

---

# 26. Puntos de parada obligatorios

Al finalizar cada bloque crítico, verifica antes de continuar.

## Parada A — DeepH capability

Debe estar probado que el backend efectivo tiene JVP real.

## Parada B — DeepH finite differences

Debe existir una comparación reproducible ML-autograd vs ML-FD.

## Parada C — Compatibilidad de datasets

No continúes hacia entrenamiento de mezclas si el target es incompatible.

## Parada D — Splits

No continúes si existe leakage.

## Parada E — Materialización

No continúes si un snapshot inválido puede producir `ready`.

## Parada F — UI

No conectes datos nuevos hasta que los schemas estén estabilizados.

No pidas confirmación al usuario en cada parada. Continúa automáticamente si se cumplen los criterios. Si no se cumplen, implementa la corrección necesaria o documenta con precisión el bloqueo técnico.

---

# 27. Definition of Done

La tarea solo está terminada cuando se cumple todo lo siguiente:

* [ ] Se conocen los SHAs efectivos de los tres repositorios.
* [ ] Se conoce el checkout realmente importado.
* [ ] DeepH autograd se detecta por capacidad, no por suposición.
* [ ] DeepH no acepta placeholders como derivadas.
* [ ] Las direcciones no calculadas no se confunden con ceros físicos.
* [ ] DeepH autograd se compara con DeepH finite differences.
* [ ] Graph2Mat autograd se compara con Graph2Mat finite differences.
* [ ] Se registran dtype y delta.
* [ ] Se comprueba topología.
* [ ] Se comprueba Hamiltoniano base antes de derivadas DeepH-SIESTA.
* [ ] Los caches tienen firma completa.
* [ ] La compatibilidad ghost se determina con artefactos activos.
* [ ] La compatibilidad orbital se verifica.
* [ ] Los pseudopotenciales incompatibles bloquean.
* [ ] Los snapshots inválidos bloquean materialización.
* [ ] Los tests de origen no entran en train.
* [ ] Existe test fijo small+large.
* [ ] El alcance small-only sigue disponible, pero correctamente etiquetado.
* [ ] El tamaño de dataset significa train real.
* [ ] Ratio nominal y composición real están separados.
* [ ] Se reporta composición por snapshots, átomos, bloques y elementos.
* [ ] Existe una política de loss explícita.
* [ ] Existe reducción per-structure.
* [ ] Existe reducción per-domain o una alternativa matemáticamente equivalente.
* [ ] Métricas micro y macro están separadas.
* [ ] Frobenius está normalizado.
* [ ] Seeds incompatibles no se agregan.
* [ ] La UI muestra claims y bloqueadores.
* [ ] Los payloads están versionados.
* [ ] Los tests dirigidos pasan.
* [ ] Existe al menos un smoke real o una explicación precisa de por qué no pudo ejecutarse.
* [ ] La documentación está actualizada.
* [ ] Existe un informe final.
* [ ] No se han perdido cambios locales del usuario.
* [ ] No se ha hecho push.

---

# 28. Respuesta final esperada del agente

Al acabar, responde con:

1. Resumen ejecutivo.
2. SHAs inspeccionados.
3. Estado inicial de cada hallazgo.
4. Correcciones implementadas.
5. Archivos modificados por repositorio.
6. Tests ejecutados y resultados.
7. Smoke tests y métricas obtenidas.
8. Qué quedó únicamente diagnóstico.
9. Riesgos pendientes.
10. Commits locales creados.
11. Comandos exactos para reproducir:

    * tests;
    * validación autograd–FD;
    * materialización pequeña;
    * lanzamiento de UI.
12. Veredicto final por subsistema.

No afirmes que algo está corregido únicamente porque el código compila. Toda corrección crítica debe estar asociada a una prueba.
