# Guía técnica DeepH-only para implementar derivadas autograd vectorizadas de Hamiltonianos respecto a posiciones atómicas

**Fecha de revisión:** 2026-07-07
**Ámbito:** DeepH clásico en `MD_vs_AtomicDisplacement`
**Excluido explícitamente:** DeepH-E3, Graph2Mat, cambios en SIESTA, eliminación de finite-difference

---

## 1. Objetivo de esta guía

Esta guía resume la estrategia técnica para implementar una ruta de derivadas analíticas/autograd vectorizadas para **DeepH clásico** respecto a posiciones atómicas:

```text
d H_deeph(R) / d R_atom,axis
```

El objetivo no es reemplazar toda la infraestructura actual, sino añadir una ruta opt-in equivalente a la ya cerrada para Graph2Mat:

```text
derivative.deeph_prediction_method = "finite_difference" | "autograd_vectorized"
```

La ruta existente debe permanecer intacta:

```text
dH/dR ≈ (H(R + δ) - H(R - δ)) / (2δ)
```

La referencia SIESTA debe seguir calculándose por finite-difference. La predicción DeepH puede seguir usando finite-difference en modo legacy, pero debe añadirse una ruta nueva que calcule la derivada directamente sobre el modelo DeepH cargado en memoria.

---

## 2. Bibliografía y fuentes consultadas

### [R1] Paper original de DeepH

**He Li, Zun Wang, Nianlong Zou, Meng Ye, Runzhang Xu, Xiaoxun Gong, Wenhui Duan, Yong Xu.**
*Deep-learning density functional theory Hamiltonian for efficient ab initio electronic-structure calculation.*
Nature Computational Science 2, 367–377, 2022.
DOI: `10.1038/s43588-022-00265-6`

URL:

```text
https://www.nature.com/articles/s43588-022-00265-6
```

Puntos relevantes:

* DeepH aprende el Hamiltoniano DFT como función de la estructura atómica.
* Usa localidad/nearsightedness.
* Trata la covariancia de gauge/rotación mediante coordenadas locales y transformaciones de base.
* Usa una red de message passing.
* Aprende bloques Hamiltonianos entre átomos vecinos dentro de un cutoff.
* El Hamiltoniano final vive en una base localizada no ortogonal.
* El overlap no se aprende por red neuronal; se obtiene de la base.

### [R2] Repositorio oficial DeepH-pack clásico

Repositorio:

```text
https://github.com/mzjb/DeepH-pack
```

Puntos relevantes:

* Implementación oficial del método DeepH.
* Soporta resultados DFT de ABACUS, OpenMX, FHI-aims y SIESTA.
* Documenta el flujo:

  * prepare dataset
  * preprocess
  * train
  * inference
* Menciona `DeepHKernel` y `make_mask` en `deeph/kernel.py`.
* El keyword `orbital` define qué orbitales o elementos de matriz se predicen.

### [R3] Documentación DeepH-pack: preprocess

URL:

```text
https://deeph-pack.deepmodeling.com/en/latest/preprocess/preprocess.html
```

Puntos relevantes:

* El preprocess convierte unidades.
* Guarda datos en texto/HDF5 por estructura.
* Genera coordenadas locales.
* Realiza transformaciones de base para Hamiltonianos DFT.
* Convenciones de unidades:

  * longitud: Å
  * energía: eV

### [R4] Documentación DeepH-pack: inference

URL:

```text
https://deeph-pack.deepmodeling.com/en/latest/inference/inference.html
```

Puntos relevantes:

* La inferencia predice Hamiltonianos DFT para estructuras grandes.
* Para cálculo de propiedades sparse se necesita overlap matrix.
* El overlap debe calcularse con la misma base y el mismo software DFT usado al preparar el dataset.
* Entrada CLI típica:

```bash
deeph-inference --config ${config_path}
```

### [R5] Documentación DeepH-pack: train

URL:

```text
https://deeph-pack.deepmodeling.com/en/latest/train/train.html
```

Puntos relevantes:

* El keyword `orbital` define qué orbitales/matrix elements son predichos.
* La documentación recomienda mirar `make_mask` en `DeepHKernel`, archivo `DeepH-pack/deeph/kernel.py`, para entender la estructura de orbitales.
* Puede haber modelos múltiples o máscaras distintas para diferentes orbitales/bloques.

### [R6] Documentación DeepH-pack: instalación clásica

URL:

```text
https://deeph-pack.deepmodeling.com/en/latest/installation/installation.html
```

Puntos relevantes:

* La documentación clásica lista dependencias históricas:

  * Python 3.9
  * NumPy
  * SciPy
  * PyTorch 1.9.1
  * PyTorch Geometric 1.7.2
  * e3nn 0.3.5
  * h5py
  * pymatgen
* Aunque aparezca `e3nn` en dependencias, esta guía se restringe a DeepH clásico, no DeepH-E3.
* Hay que inspeccionar la versión instalada en el repositorio real antes de decidir si están disponibles `torch.func.jacrev` y APIs modernas.

### [R7] PyTorch `torch.func.jacrev`

URL:

```text
https://docs.pytorch.org/docs/stable/generated/torch.func.jacrev.html
```

Puntos relevantes:

* Calcula jacobianas usando reverse-mode autodiff.
* Acepta `chunk_size`.
* `chunk_size=None` equivale a un `vmap` grande sobre VJP.
* `chunk_size=1` equivale conceptualmente a calcular fila por fila; debe evitarse como default salvo fallback.
* La documentación advierte que `torch.no_grad()` dentro de la función diferenciada sí es respetado por `jacrev`, por lo que hay que evitar `no_grad` dentro de la closure DeepH.

### [R8] PyTorch `torch.func.jacfwd`

URL:

```text
https://docs.pytorch.org/docs/stable/generated/torch.func.jacfwd.html
```

Puntos relevantes:

* Calcula jacobianas usando forward-mode autodiff.
* Puede ser útil cuando hay pocos outputs y muchos inputs.
* Puede fallar si alguna operación no tiene forward-mode AD implementado; la documentación recomienda `jacrev` como alternativa con mejor cobertura de operadores.

### [R9] PyTorch `torch.func.vjp`

URL:

```text
https://docs.pytorch.org/docs/stable/generated/torch.func.vjp.html
```

Puntos relevantes:

* Devuelve el resultado de la función y una función `vjp_fn`.
* `vjp_fn(cotangents)` calcula productos vector-Jacobian.
* Es útil para construir jacobianas por chunks de outputs con `vmap`.

### [R10] PyTorch `torch.vmap`

URL:

```text
https://docs.pytorch.org/docs/stable/generated/torch.vmap.html
```

Puntos relevantes:

* Vectoriza una función sobre una dimensión.
* Puede usarse para batched gradients y batched VJP.
* Acepta `chunk_size`.
* `chunk_size=1` equivale a un bucle, por lo que no debe ser default para una ruta vectorizada.

### [R11] PyTorch `torch.autograd.functional.jacobian`

URL:

```text
https://docs.pytorch.org/docs/stable/generated/torch.autograd.functional.jacobian.html
```

Puntos relevantes:

* Puede calcular jacobianas con `vectorize=True`.
* La propia documentación recomienda preferir `torch.func.jacrev` o `torch.func.jacfwd` para alternativas menos experimentales y potencialmente más eficientes.
* Útil como baseline o fallback simple para tests pequeños.

### [R12] Nota sobre DeepH-pack moderno/JAX

Fuentes recientes indican que una evolución moderna de DeepH-pack/DeepX puede estar reconstruida sobre JAX. Esta guía, sin embargo, está escrita para la ruta DeepH clásica usada en tu repositorio. Por tanto, el primer paso del implementador debe ser confirmar si la ruta DeepH real del repo usa PyTorch clásico, scripts CLI externos, o un wrapper moderno.

---

## 3. Decisión de alcance

Esta guía cubre exclusivamente:

```text
DeepH clásico
```

Queda fuera:

```text
DeepH-E3
DeepH-2
DeepH-Zero
DeepH-r
xDeepH
Graph2Mat
HPRO
cambios a SIESTA
```

Regla fuerte para el futuro mega-prompt:

```text
No importar e3nn salvo que ya sea dependencia pasiva de DeepH clásico.
No tocar código DeepH-E3.
No reutilizar instrucciones Graph2Mat salvo conceptos generales de autograd/vectorización.
```

---

## 4. Objetivo científico

El problema actual es que las derivadas ML por finite-difference restan dos predicciones independientes:

```text
dH_ML/dR ≈ [H_ML(R + δ) - H_ML(R - δ)] / (2δ)
```

Esto amplifica ruido numérico y ruido de inferencia.

Para DeepH, el objetivo es calcular:

```text
d H_deeph(R) / d R_atom,axis
```

directamente con autograd.

La referencia debe seguir siendo:

```text
dH_ref/dR = finite_difference(SIESTA)
```

La predicción DeepH nueva debe ser:

```text
dH_pred/dR = autograd_vectorized(DeepH)
```

Metadatos mínimos:

```json
{
  "reference_derivative_method": "finite_difference_siesta",
  "predicted_derivative_method": "autograd_deeph_vectorized",
  "reference_delta_ang": 0.01,
  "predicted_delta_ang": null,
  "deeph_prediction_method": "autograd_vectorized",
  "jacobian_method": "vmap_vjp_chunked",
  "jacobian_chunk_size": 128,
  "topology_fixed": true,
  "units": "eV/Angstrom",
  "model_family": "DeepH",
  "not_deeph_e3": true
}
```

---

## 5. Diferencia conceptual entre Graph2Mat y DeepH

Graph2Mat suele tener una ruta más directa:

```text
positions
  -> model.forward
  -> node_labels / edge_labels
  -> data_processor.yield_from_batch
  -> sparse H
```

DeepH clásico suele tener más capas de procesamiento:

```text
positions
  -> graph / neighbours / shifts
  -> distances, edge vectors, local coordinates
  -> basis transformations
  -> model.forward
  -> predicted Hamiltonian blocks
  -> block postprocessing / orbital mask
  -> sparse H / HR / h5 / CSR / HSX
```

Por eso, en DeepH la pregunta principal no es solo:

```text
¿Cómo llamo a jacrev?
```

sino:

```text
¿Existe una closure diferenciable positions -> Hamiltonian blocks?
```

---

## 6. Tres niveles posibles de derivada en DeepH

### Nivel A: derivada de outputs crudos del modelo

```text
d raw_outputs / dR
```

Ventajas:

* Es lo más fácil.
* Probablemente el primer tensor diferenciable disponible.

Desventajas:

* Puede no ser `dH/dR` físico si después hay transformaciones de base o rotaciones dependientes de geometría.
* Puede no mapear directamente al formato sparse evaluado.

### Nivel B: derivada de bloques Hamiltonianos transformados

```text
d H_blocks / dR
```

Ventajas:

* Es la opción recomendada.
* Incluye outputs del modelo y transformaciones necesarias hasta bloques Hamiltonianos comparables.
* Permite ensamblar `dH_sparse` usando el ensamblador DeepH existente.

Desventajas:

* Requiere localizar el punto exacto donde DeepH produce bloques en torch.
* Requiere evitar NumPy/HDF5/Scipy antes de terminar autograd.

### Nivel C: derivada de matriz sparse final

```text
d H_sparse / dR
```

Ventajas:

* Conceptualmente ideal.

Desventajas:

* No debe intentarse diferenciando SciPy CSR, HDF5 o escritura a disco.
* Lo correcto es derivar bloques en torch y ensamblar después.

### Recomendación

El mega-prompt final debe apuntar a:

```text
Nivel B: positions -> DeepH differentiable forward -> transformed Hamiltonian blocks
```

Después:

```text
d_blocks -> ensamblador DeepH normal -> dH_sparse
```

---

## 7. Condición central de viabilidad

Autograd DeepH solo es científicamente válido si la closure conserva la dependencia de posiciones:

```python
def deeph_forward_blocks_from_positions(positions):
    ...
    return blocks_flat
```

Esta closure debe evitar:

```text
torch.no_grad()
.detach()
.cpu().numpy()
np.asarray(...)
scipy
h5py
lectura/escritura de archivos intermedios
.item() usado para construir features geométricas diferenciables
reconstrucción no diferenciable de coordenadas locales
```

Sí puede mantener fijos:

```text
edge_index
neighbour list
lattice shifts
orbital mapping
sparse pattern
```

Esto corresponde a una derivada local con topología fija.

---

## 8. Topología fija

Para una derivada local:

```text
edge_index = fijo
shifts = fijo
cell = fijo, salvo que se quiera derivar respecto a celda
```

Los edge vectors deben recalcularse diferenciablemente desde `positions`:

```python
r_ij = positions[j] + shift_ij @ cell - positions[i]
```

donde:

```text
i, j vienen de edge_index
shift_ij es fijo
cell es fijo
positions requiere gradiente
```

La derivada resultante es:

```text
∂H / ∂R_atom,axis | topology fixed
```

No debe recomputarse la lista de vecinos dentro de la closure.

---

## 9. Preguntas de inspección obligatorias

El chat con acceso al repo debe empezar buscando rutas DeepH:

```bash
grep -R "deeph" -n .
grep -R "DeepH" -n .
grep -R "deeph-inference" -n .
grep -R "DeepHKernel" -n .
grep -R "make_mask" -n .
grep -R "orbital" -n .
grep -R "torch.no_grad" -n .
grep -R "detach" -n .
grep -R "numpy" -n .
grep -R "h5py" -n .
grep -R "scipy" -n .
grep -R "csr" -n .
grep -R "HSX" -n .
```

También debe localizar la instalación externa de DeepH:

```bash
python - <<'PY'
import inspect, os
try:
    import deeph
    print(os.path.dirname(deeph.__file__))
except Exception as exc:
    print("No se pudo importar deeph:", exc)
PY
```

Archivos candidatos a inspeccionar en DeepH-pack clásico:

```text
deeph/kernel.py
deeph/model.py
deeph/inference/
deeph/preprocess/
deeph/graph/
deeph/data/
deeph/utils/
deeph/sparse/
deeph/default.ini
deeph/inference/inference_default.ini
```

No asumir nombres exactos sin inspección, porque la estructura puede variar entre versiones.

---

## 10. Checklist de inspección DeepH

Antes de implementar nada, responder:

```text
1. ¿Cómo se invoca DeepH actualmente desde MD_vs_AtomicDisplacement?
2. ¿Se usa deeph-inference como CLI externo?
3. ¿Se llama a Python API de DeepH?
4. ¿Dónde se carga el checkpoint?
5. ¿Qué clase PyTorch representa el modelo?
6. ¿Dónde está el forward?
7. ¿Qué tensores usa como entrada?
8. ¿Usa positions absolutas?
9. ¿Usa edge vectors?
10. ¿Usa distancias precomputadas?
11. ¿Usa coordenadas locales precomputadas?
12. ¿Usa rotaciones o basis transformations dependientes de geometría?
13. ¿Dónde se ensamblan bloques Hamiltonianos?
14. ¿Dónde se aplica el keyword orbital/mask?
15. ¿Dónde se escribe el resultado final?
16. ¿Qué parte está en torch?
17. ¿Qué parte está en numpy/scipy/h5py?
18. ¿Dónde aparece torch.no_grad()?
19. ¿La ruta de inferencia permite gradiente?
20. ¿Hay batch size > 1?
```

Punto de parada:

```text
No implementar jacobiana hasta localizar el tensor de bloques Hamiltonianos más cercano al H físico y todavía dentro de torch.
```

---

## 11. Rutas posibles según lo que se encuentre

### Caso 1: DeepH ya tiene forward diferenciable desde posiciones

Situación ideal:

```text
positions -> torch geometric features -> model -> blocks
```

Entonces implementar directamente:

```python
positions = base_positions.clone().detach().requires_grad_(True)
blocks = deeph_forward_blocks_from_positions(positions)
```

### Caso 2: DeepH usa edge vectors/distances precomputados

Hay que sustituir el input geométrico por una reconstrucción diferenciable:

```python
edge_vec = positions[dst] + shifts @ cell - positions[src]
edge_len = torch.linalg.norm(edge_vec, dim=-1)
```

Después alimentar esos features al modelo.

### Caso 3: DeepH usa coordenadas locales precomputadas

Hay dos subcasos:

```text
3A. Coordenadas locales solo definen una frame fija del batch base.
    -> Puede aceptarse como derivada parcial con frame fijo, pero hay que documentarlo.

3B. Coordenadas locales deberían cambiar con positions.
    -> Hay que portar la parte mínima a torch o declarar bloqueo científico.
```

La opción preferida es implementar la derivada completa de Nivel B. Si no es viable en una primera etapa, debe guardarse metadato:

```json
{
  "local_coordinates_differentiated": false,
  "derivative_scope": "partial_fixed_local_frame"
}
```

Pero no llamar a eso derivada completa sin aclararlo.

### Caso 4: DeepH actual es solo CLI con archivos

Si `MD_vs_AtomicDisplacement` solo llama:

```bash
deeph-inference --config ...
```

entonces autograd directo requiere construir una API interna:

```text
load_deeph_model(...)
load_deeph_structure_as_batch(...)
forward_deeph_blocks(...)
assemble_deeph_sparse(...)
```

No se puede obtener autograd de una llamada CLI que escribe/lee archivos.

### Caso 5: DeepH instalado es versión moderna JAX

Esta guía no cubre implementación JAX completa. El chat debe documentar:

```text
DeepH backend = JAX
PyTorch autograd no aplica
```

Una futura ruta sería:

```text
jax.jacrev / jax.jacfwd / jax.vmap
```

Pero el mega-prompt DeepH clásico debe bloquear o separar esta ruta.

---

## 12. Diseño de módulo DeepH autograd

Archivo sugerido:

```text
Comparison/scripts/deeph_autograd_derivatives.py
```

Funciones sugeridas:

```python
def load_deeph_model_for_autograd(config):
    """
    Carga DeepH sin torch.no_grad.
    Devuelve modelo, metadata de orbitales, ensamblador y device/dtype.
    """
```

```python
def load_deeph_base_structure_batch(config, sample_index):
    """
    Carga la estructura base correspondiente a un sample.
    No carga R+delta ni R-delta para predicción.
    Debe devolver positions, cell, edge_index, shifts y todo lo necesario
    para reproducir el mapping DeepH normal.
    """
```

```python
def replace_deeph_positions(batch, positions):
    """
    Devuelve una copia/adaptación del batch con positions reemplazadas.
    No detach.
    No numpy.
    """
```

```python
def recompute_deeph_geometric_features_torch(batch, positions):
    """
    Recalcula edge vectors/distances/features geométricos desde positions
    dentro de torch, manteniendo edge_index y shifts fijos.
    """
```

```python
def deeph_forward_blocks(model, batch, positions):
    """
    Closure diferenciable:
        positions -> transformed Hamiltonian blocks

    No usa torch.no_grad.
    No convierte a numpy.
    No escribe archivos.
    """
```

```python
def flatten_deeph_blocks(blocks):
    """
    Convierte una estructura de bloques DeepH en un vector 1D.
    Devuelve outputs_flat y spec para reconstruir.
    """
```

```python
def unflatten_deeph_blocks(flat, spec):
    """
    Reconstruye la estructura de bloques a partir del vector plano.
    """
```

```python
def compute_deeph_position_jacobian(
    model,
    batch,
    *,
    method="vmap_vjp_chunked",
    chunk_size=128,
):
    """
    Calcula J = d outputs_flat / d positions.
    Forma conceptual:
        [n_outputs, n_atoms, 3]
    """
```

```python
def select_deeph_derivative_blocks(
    jacobian,
    spec,
    atom_index,
    axis_index,
):
    """
    Selecciona J[:, atom_index, axis_index] y reconstruye d_blocks.
    """
```

```python
def assemble_deeph_sparse_from_blocks_like_normal_route(
    d_blocks,
    assembly_context,
):
    """
    Reutiliza el ensamblador normal de DeepH para convertir bloques derivados
    en una matriz sparse dH/dR.
    """
```

---

## 13. Closure diferenciable mínima

La closure debe parecerse conceptualmente a esto:

```python
def closure(positions):
    positions = positions.requires_grad_(True)

    batch_pos = replace_deeph_positions(batch, positions)
    batch_geo = recompute_deeph_geometric_features_torch(batch_pos, positions)

    blocks = model_forward_to_transformed_blocks(
        model,
        batch_geo,
        include_basis_transform=True,
        include_orbital_mask=True,
    )

    flat, spec = flatten_deeph_blocks(blocks)
    return flat
```

Reglas:

```text
model.eval()
torch.set_grad_enabled(True)
sin no_grad dentro de closure
sin detach dentro de closure
sin numpy dentro de closure
sin scipy dentro de closure
sin h5py dentro de closure
sin escritura a disco dentro de closure
```

---

## 14. Qué debe incluirse en la derivada

La derivada completa deseada de DeepH clásico debe incluir:

```text
1. Dependencia de edge distances respecto a positions.
2. Dependencia de edge vectors respecto a positions.
3. Dependencia de features gaussianas respecto a distances.
4. Dependencia de orientación/local-coordinate features si se recalculan desde positions.
5. Forward MPNN.
6. Transformación de outputs a bloques Hamiltonianos físicos.
7. Máscara orbital/mapping de outputs a bloques.
```

La derivada no debe incluir:

```text
1. Recomputación de neighbor list.
2. Cambio discontinuo de cutoff.
3. Cambio de sparse pattern.
4. Cambio de número de edges.
5. Derivada respecto a celda, salvo que se implemente explícitamente.
6. Derivada del overlap S, salvo que el proyecto decida hacerlo en otra fase.
```

---

## 15. Estrategia de jacobiana vectorizada

Conceptualmente:

```python
outputs_flat = concat(all_deeph_block_outputs.flatten())
J = d outputs_flat / d positions
```

Forma esperada:

```text
outputs_flat.shape = [n_outputs]
positions.shape    = [n_atoms, 3]
J.shape            = [n_outputs, n_atoms, 3]
```

Luego:

```python
d_outputs = J[:, atom_index, axis_index]
d_blocks = unflatten_deeph_blocks(d_outputs, spec)
dH_sparse = assemble(d_blocks)
```

---

## 16. Métodos de jacobiana

### Opción 1: `torch.func.jacrev`

Uso conceptual:

```python
from torch.func import jacrev

J = jacrev(closure, argnums=0, chunk_size=chunk_size)(positions)
```

Ventajas:

* Reverse-mode.
* Buena cobertura de operadores.
* API moderna.
* Soporta `chunk_size`.

Desventajas:

* Puede consumir mucha memoria si `outputs_flat` es enorme.

### Opción 2: `torch.func.jacfwd`

Uso conceptual:

```python
from torch.func import jacfwd

J = jacfwd(closure, argnums=0)(positions)
```

Ventajas:

* Forward-mode puede ser eficiente si hay pocos outputs y muchos inputs.

Desventajas:

* Puede fallar si faltan reglas forward-mode para alguna operación.
* Probablemente no es primera opción en DeepH clásico con PyTorch/PyG antiguo.

### Opción 3: `torch.autograd.functional.jacobian(vectorize=True)`

Uso conceptual:

```python
J = torch.autograd.functional.jacobian(
    closure,
    positions,
    vectorize=True,
    strategy="reverse-mode",
)
```

Ventajas:

* Simple.
* Bueno para tests pequeños.

Desventajas:

* Documentado como más experimental.
* Puede tener cliffs de rendimiento.
* No ideal como backend principal.

### Opción 4: `vmap(vjp)` por chunks

Uso conceptual:

```python
from torch.func import vjp, vmap

y, vjp_fn = vjp(closure, positions)

def apply_vjp(cotangent_batch):
    grad_pos, = vmap(vjp_fn)(cotangent_batch)
    return grad_pos

# cotangent_batch: [chunk_size, n_outputs]
# resultado: [chunk_size, n_atoms, 3]
```

Ventajas:

* Control fino de memoria.
* Bueno cuando `n_outputs` es muy grande.
* Evita bucles elemento-a-elemento.

Desventajas:

* Más código.
* Hay que manejar chunks y concatenación con cuidado.

### Recomendación para DeepH

Default recomendado:

```text
method = "vmap_vjp_chunked"
chunk_size = 128
```

Fallback:

```text
method = "jacrev"
```

Para tests:

```text
method = "autograd_jacobian"
```

---

## 17. Chunking correcto

Permitido:

```text
bucle por chunks de outputs
```

No permitido como estrategia principal:

```text
bucle por elemento de matriz
bucle por cada label individual
bucle por cada orbital pair individual
```

Ejemplo conceptual correcto:

```python
for start in range(0, n_outputs, chunk_size):
    stop = min(start + chunk_size, n_outputs)
    eye_chunk = torch.eye(n_outputs, device=device, dtype=dtype)[start:stop]
    J_chunk = vmap(vjp_fn)(eye_chunk)[0]
    chunks.append(J_chunk)

J = torch.cat(chunks, dim=0)
```

Aunque hay un bucle Python por chunks, sigue siendo vectorizado dentro del chunk y evita el patrón elemento-a-elemento.

---

## 18. Flatten/unflatten de bloques DeepH

DeepH puede producir bloques en varias estructuras:

```text
dict por tipo de orbital
dict por par de especies
dict por edge
dict por shell
lista de modelos
tensor [n_edges, n_outputs]
tensor [n_edges, n_orb_i, n_orb_j]
tensor complejo real/imag separado
```

Por tanto, `flatten_deeph_blocks` debe preservar una especificación completa:

```json
{
  "entries": [
    {
      "key_path": ["edge_blocks", "C-C"],
      "shape": [128, 4, 4],
      "dtype": "float64",
      "is_complex": false,
      "start": 0,
      "stop": 2048
    }
  ],
  "total_outputs": 2048
}
```

Debe soportar:

```text
dict
list/tuple
torch.Tensor
tensores complejos
pares real/imag si DeepH los usa
```

Para complejos hay dos opciones:

```text
Opción A:
    flatten real e imag por separado.

Opción B:
    usar autograd complejo si todo el pipeline lo soporta.
```

Recomendación primera versión:

```text
Representar complejos como real vector:
    outputs_flat = concat(real.flatten(), imag.flatten())
```

Y reconstruir después:

```python
complex_block = real_block + 1j * imag_block
```

---

## 19. Ensamblado sparse

No reconstruir el mapping orbital a mano salvo bloqueo total.

Ruta deseada:

```text
normal:
    blocks = DeepH(R)
    H_sparse = assemble_deeph(blocks, context)

autograd:
    d_blocks = d DeepH(R) / dR_atom,axis
    dH_sparse = assemble_deeph(d_blocks, same_context)
```

La implementación debe localizar y reutilizar:

```text
función que transforma bloques DeepH a matriz final
función que aplica orbital mask
función que escribe o prepara sparse H
función que convierte a formato HSX/HDF5/CSR
```

Si el ensamblador DeepH solo acepta NumPy, la conversión a NumPy se permite únicamente después de terminar autograd:

```python
d_blocks_np = tree_map(lambda x: x.detach().cpu().numpy(), d_blocks)
```

Nunca antes.

---

## 20. Formato de serialización sugerido

Guardar cada derivada directa como:

```text
dH_deeph_autograd_atom{atom_index}_axis{axis_index}.npz
dH_deeph_autograd_atom{atom_index}_axis{axis_index}.json
```

La matriz `.npz` debe ser CSR o COO de SciPy, siguiendo la convención ya usada en el repo si existe.

Metadatos:

```json
{
  "reference_derivative_method": "finite_difference_siesta",
  "predicted_derivative_method": "autograd_deeph_vectorized",
  "reference_delta_ang": 0.01,
  "predicted_delta_ang": null,
  "deeph_prediction_method": "autograd_vectorized",
  "jacobian_method": "vmap_vjp_chunked",
  "jacobian_chunk_size": 128,
  "atom_index": 0,
  "axis_index": 0,
  "axis_name": "x",
  "units": "eV/Angstrom",
  "topology_fixed": true,
  "edge_index_fixed": true,
  "shifts_fixed": true,
  "cell_derivative": false,
  "overlap_derivative": false,
  "model_family": "DeepH",
  "not_deeph_e3": true,
  "local_coordinates_differentiated": null,
  "basis_transform_differentiated": null,
  "derivative_scope": "full_blocks_if_confirmed"
}
```

Los campos `local_coordinates_differentiated` y `basis_transform_differentiated` deben rellenarse tras inspección real.

---

## 21. Integración con métricas

El evaluador debe aceptar dos rutas DeepH:

### Ruta legacy

```text
predicción DeepH = finite_difference sobre H_deeph(R+δ), H_deeph(R-δ)
```

### Ruta nueva

```text
predicción DeepH = dH_deeph/dR directa desde autograd
```

La referencia SIESTA sigue igual:

```text
referencia = finite_difference_siesta
```

No duplicar métricas. Reutilizar:

```python
derivative_sparse_metrics(dH_ref, dH_pred, ...)
```

Añadir helpers como:

```python
def load_direct_deeph_sparse_derivative(path):
    ...
```

```python
def get_deeph_predicted_derivative_sparse(..., method):
    if method == "finite_difference":
        ...
    elif method == "autograd_vectorized":
        ...
```

---

## 22. Integración con configuración

Añadir:

```text
derivative.deeph_prediction_method = "finite_difference" | "autograd_vectorized"
```

Default:

```text
"finite_difference"
```

para compatibilidad.

Comportamiento:

```text
Si derivative.deeph_prediction_method == "finite_difference":
    ejecutar exactamente la ruta actual

Si derivative.deeph_prediction_method == "autograd_vectorized":
    mantener stencils para SIESTA/reference
    no generar DeepH(R+δ) ni DeepH(R-δ)
    cargar DeepH sobre estructura base
    calcular dH_deeph/dR con autograd
    serializar dH_pred directo
    evaluar contra SIESTA finite-difference
```

---

## 23. Archivos sugeridos para el futuro mega-prompt

Crear:

```text
Comparison/scripts/deeph_autograd_derivatives.py
Comparison/scripts/run_deeph_autograd_derivative_predictions.py
tests/test_deeph_autograd_derivatives.py
tests/test_deeph_direct_derivative_evaluator.py
```

Modificar con cuidado:

```text
Comparison/scripts/run_hamiltonian_derivative_predictions.py
Comparison/scripts/evaluate_hamiltonian_derivative_metrics.py
Comparison/scripts/hamiltonian_derivative_stencil.py
g2m_deeph_runner.py
```

No tocar salvo necesidad:

```text
Graph2Mat ya cerrado
rutas Graph2Mat autograd
rutas SIESTA
rutas DeepH-E3
código externo de DeepH-pack
```

Si hay que tocar DeepH-pack externo, documentar exactamente por qué.

---

## 24. Tests recomendados

### Test 1: modelo DeepH falso diferenciable

Crear un fake DeepH que produzca bloques conocidos:

```python
edge_vec = positions[dst] - positions[src]
edge_scalar = (edge_vec ** 2).sum(dim=-1)
```

Derivadas analíticas:

```text
d ||R_j - R_i||^2 / d R_i = -2 (R_j - R_i)
d ||R_j - R_i||^2 / d R_j =  2 (R_j - R_i)
```

Verificar:

```text
J.shape == [n_outputs, n_atoms, 3]
J coincide con analítica
```

### Test 2: flatten/unflatten

Con bloques sintéticos:

```python
blocks = {
    "onsite": torch.randn(n_atoms, n_orb, n_orb),
    "edge": torch.randn(n_edges, n_orb_i, n_orb_j),
}
```

Verificar:

```text
unflatten(flatten(blocks)) == blocks
```

### Test 3: `jacrev` vs `vmap_vjp_chunked`

Para estructura pequeña:

```text
J_jacrev ≈ J_chunked
```

Tolerancias:

```text
float64: atol 1e-8, rtol 1e-6
float32: atol 1e-5, rtol 1e-4
```

### Test 4: topología fija

Verificar dentro de la closure:

```text
edge_index no cambia
shifts no cambia
n_edges no cambia
```

### Test 5: ensamblado sparse

Con bloques sintéticos y mapping simple, comprobar:

```text
assemble(blocks).shape == assemble(d_blocks).shape
```

y que el patrón sparse esperado se conserva.

### Test 6: evaluador con predicción directa

Construir matrices pequeñas:

```python
dH_ref = scipy.sparse.csr_matrix(...)
dH_pred = scipy.sparse.csr_matrix(...)
```

Guardar en formato directo nuevo.

Verificar:

```text
el evaluador carga dH_pred directo
no busca DeepH(R+δ), DeepH(R-δ)
sí calcula referencia SIESTA finite-difference si el test cubre ese flujo
reutiliza derivative_sparse_metrics
metadatos correctos
```

### Test 7: smoke CLI

Comando conceptual:

```bash
python Comparison/scripts/run_deeph_autograd_derivative_predictions.py \
  --config <config_minima_deeph_autograd> \
  --sample-index 0 \
  --atom-index 0 \
  --axis-index 0
```

O vía runner:

```bash
python g2m_deeph_runner.py \
  --config <config_minima_deeph_autograd>
```

---

## 25. Validación científica

### 25.1 Unidades

Confirmar:

```text
H: eV
R: Å
dH/dR: eV/Å
```

DeepH-pack clásico documenta:

```text
Length = Å
Energy = eV
```

Si alguna ruta usa Bohr, Hartree o escalado interno, aplicar conversión explícita.

Factores útiles:

```text
1 Bohr = 0.529177210903 Å
1 Hartree = 27.211386245988 eV
```

### 25.2 Shapes

Verificar en runtime:

```text
positions.shape == [n_atoms, 3]
outputs_flat.shape == [n_outputs]
J.shape == [n_outputs, n_atoms, 3]
d_blocks tienen mismas shapes que blocks
dH_sparse.shape == H_sparse.shape
```

### 25.3 Hermiticidad

Medir:

```text
||dH - dH†|| / max(||dH||, eps)
```

No forzar hermiticidad silenciosamente salvo que la ruta normal DeepH ya lo haga.

Guardar:

```json
{
  "hermiticity_relative_error": 1.2e-8,
  "hermiticity_checked": true,
  "hermiticity_forced": false
}
```

### 25.4 Topología fija

Guardar:

```json
{
  "topology_fixed": true,
  "edge_index_fixed": true,
  "shifts_fixed": true
}
```

### 25.5 Comparación con finite-difference DeepH

Usar solo como sanity check:

```text
autograd_deeph vs finite_difference_deeph
```

con varios δ:

```text
δ grande: domina error de truncamiento
δ medio: debería haber acuerdo razonable
δ pequeño: puede dominar cancelación/ruido
```

No tratar finite-difference DeepH como fuente de verdad.

---

## 26. Riesgos técnicos específicos de DeepH

### Riesgo 1: DeepH actual es solo CLI

Si el repo llama `deeph-inference` por subprocess, no hay autograd.

Mitigación:

```text
crear wrapper Python interno que cargue modelo y batch en memoria
```

### Riesgo 2: coordenadas locales precomputadas

Si las coordenadas locales vienen de HDF5, autograd no ve su dependencia con positions.

Mitigación:

```text
portar cálculo geométrico mínimo a torch
o documentar derivada parcial con local frame fijo
```

### Riesgo 3: basis transformation fuera de torch

Si la transformación de base se hace en NumPy, la derivada se corta.

Mitigación:

```text
mover transformación necesaria a torch
o derivar antes de la transformación y documentar alcance
```

### Riesgo 4: orbital mask compleja

DeepH puede predecir diferentes orbitales con diferentes modelos/máscaras.

Mitigación:

```text
usar make_mask / DeepHKernel
no reconstruir mapping manualmente
test específico de flatten/unflatten por orbital
```

### Riesgo 5: PyTorch antiguo

DeepH clásico puede depender de PyTorch 1.9.1, donde `torch.func` no existe.

Mitigaciones posibles:

```text
1. si el entorno del repo usa PyTorch moderno, usar torch.func
2. si usa functorch separado, usar functorch.jacrev/vmap/vjp
3. si solo PyTorch antiguo, fallback torch.autograd.grad por chunks batched puede no existir
4. documentar bloqueo si no hay backend vectorizado disponible
```

### Riesgo 6: PyG scatter no compatible con vmap

Algunas operaciones de PyTorch Geometric antiguas pueden no ser compatibles con `vmap`.

Mitigación:

```text
fallback jacrev con chunk_size
fallback autograd.functional.jacobian(vectorize=True)
último fallback por chunks no vectorizados solo para diagnóstico, no ruta final
```

### Riesgo 7: complejos/SOC

Si el modelo produce Hamiltonianos complejos:

```text
representar real/imag explícitamente
test de hermiticidad complejo
```

### Riesgo 8: batch size > 1

Primera versión puede soportar solo batch size 1.

Comportamiento recomendado:

```python
if batch_size != 1:
    raise NotImplementedError("DeepH autograd derivatives currently support batch_size=1")
```

---

## 27. Decisiones para evitar sobreingeniería

No hacer:

```text
no DeepH-E3
no JAX en primera versión
no derivada respecto a celda
no derivada del overlap S
no reconstrucción manual de orbital mapping si hay ensamblador existente
no optimización multi-GPU
no cambios en Graph2Mat ya cerrado
no eliminación de finite-difference
no forzar hermiticidad sin medir
```

Sí hacer:

```text
batch size 1 inicialmente
topología fija
chunks vectorizados
metadatos explícitos
tests con modelo falso
fallbacks documentados
```

---

## 28. Estructura sugerida para el futuro mega-prompt

Cuando se convierta esta guía en mega-prompt para un chat con acceso al repo, dividirlo así:

```text
Fase 1: inspección DeepH-only
Fase 2: localizar forward diferenciable positions -> blocks
Fase 3: refactor mínimo para closure autograd
Fase 4: flatten/unflatten de outputs DeepH
Fase 5: jacobiana vectorizada/chunked
Fase 6: ensamblado sparse de d_blocks
Fase 7: integración con evaluador
Fase 8: integración con workflow/config
Fase 9: tests unitarios y smoke tests
Fase 10: validación científica
```

Punto de parada tras cada fase:

```text
no avanzar sin verificar flujo, shapes y tests mínimos
```

---

## 29. Criterio global de éxito

La implementación DeepH-autograd estará completa solo si:

```text
1. La ruta DeepH finite-difference legacy sigue funcionando.
2. SIESTA sigue usando finite-difference.
3. Existe derivative.deeph_prediction_method.
4. "finite_difference" es el default.
5. "autograd_vectorized" no genera DeepH(R+δ), DeepH(R-δ).
6. La closure DeepH conserva gradiente desde positions hasta bloques.
7. La jacobiana se calcula vectorizada o por chunks vectorizados.
8. Los bloques derivados se ensamblan con el mapping DeepH normal.
9. El evaluador acepta dH_pred directo.
10. Los metadatos distinguen referencia y predicción.
11. Hay tests unitarios con modelo falso.
12. Hay smoke test CLI mínimo.
13. Se documentan unidades, topología fija, hermiticidad y alcance de coordenadas locales.
```

---

## 30. Instrucción clave para el implementador

Antes de escribir código de jacobiana, localizar el punto exacto:

```text
positions -> ... -> DeepH Hamiltonian blocks todavía en torch
```

Ese punto define toda la implementación.

Si no existe, el primer trabajo no es autograd, sino refactorizar la inferencia para crear una closure diferenciable en memoria.

No llamar “derivada autograd DeepH completa” a una derivada que solo ve features precomputadas fijas. En ese caso debe llamarse explícitamente:

```text
partial_fixed_features_derivative
```

o bloquear la fase con explicación técnica.
