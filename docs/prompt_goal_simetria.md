# Prompt para modo Goal — Guía definitiva de implementación de simetría en stencils y derivadas del Hamiltoniano

## 1. Misión

Tu objetivo es **producir una guía de implementación superdetallada, precisa, realista y accionable** para aprovechar simetrías cristalinas en la generación de stencils de diferencias finitas y en el cálculo/reconstrucción de derivadas del Hamiltoniano `dH/dR` dentro del repositorio:

```text
/home/christian/repositorios/MD_vs_AtomicDisplacement
```

La optimización objetivo es reducir el número de estructuras desplazadas (y por tanto de ejecuciones SIESTA y de predicciones ML por diferencias finitas) de `6N` por snapshot base a `~6·N_inequiv`, reconstruyendo las derivadas faltantes mediante la ecuación central:

```text
D_{jβ}H = Σ_α  Q_{βα} · U_g · D_{iα}H · U_g†
```

donde `g` es una operación de simetría que manda el átomo representante `i` al átomo equivalente `j`, `Q` es su parte rotacional cartesiana y `U_g` es su representación en la base orbital del Hamiltoniano.

**El entregable NO es código de producción: es la guía.** Un agente de programación distinto la ejecutará después fase a fase. Tu trabajo es que esa guía sea tan concreta (archivos exactos, firmas de funciones, contratos de datos, tests con nombre y comando, criterios de aceptación) que ese agente no tenga que tomar ninguna decisión de diseño por su cuenta.

## 2. Entregable exacto

Un único documento nuevo:

```text
docs/guia_implementacion_simetria.md
```

- Escrito en español; identificadores, rutas, flags y bloques de código en inglés (convención del repo).
- Autocontenido: no debe requerir leer este prompt para ejecutarse.
- Estructurado por fases incrementales e independientemente aterrizables (cada fase deja el repo en verde y el modo legacy intacto).

No modifiques ningún otro archivo del repositorio. Puedes ejecutar todo el código de inspección de solo lectura que necesites (abrir HSX con sisl, listar datasets, leer manifests, correr tests existentes). Si quieres verificar empíricamente la detección de simetría con spglib, hazlo en un entorno desechable (por ejemplo `pip install spglib` en un venv temporal fuera de `.venv/`), nunca instalando dependencias en el `.venv` del repo; si no es posible, deja ese experimento descrito como primer paso ejecutable de la Fase 1 de tu guía.

## 3. Material de contexto obligatorio

Antes de escribir nada, lee **completos** y en este orden:

1. `docs/derivadas_simetria.md` — contexto físico-matemático general: por qué la reducción por simetría de derivadas del Hamiltoniano no es "copiar ficheros", la ecuación de reconstrucción, la estructura conceptual de `U_g` (permutación de átomos × rotación de armónicos esféricos reales × offsets periódicos × espín), niveles de implementación A–D, riesgos.
2. `docs/plan_derivadas_simetria.md` — un borrador de plan de implementación **generado sin acceso directo al repositorio**. Úsalo como inspiración (fases, flags CLI propuestos, dataclasses, batería de validación), pero NO lo copies: contiene desajustes con el código real que se listan en la sección 6 de este prompt y que tú debes corregir con evidencia del repo. Tu guía debe incluir una sección explícita "Correcciones al plan previo" documentando cada divergencia encontrada.
3. El código real del repositorio (sección 5). Toda afirmación de tu guía sobre "cómo funciona hoy el repo" debe llevar referencia `archivo:línea` verificada por ti.

## 4. Proceso de trabajo exigido

1. Lee los dos documentos de contexto completos.
2. Inspecciona los archivos listados en la sección 5 y todos los que descubras relevantes (tests incluidos). Verifica cada "hecho verificado" de la sección 5 antes de apoyarte en él: si alguno no se cumple, corrígelo en tu guía y señálalo.
3. Traza el flujo completo de una derivada de punta a punta: generación de stencil → ejecución SIESTA → predicción ML (finite-difference y autograd) → discovery → diferencias finitas → métricas. Tu guía debe demostrar ese entendimiento citando las funciones concretas por las que pasa cada dato.
4. Cuantifica la ganancia real por tipo de dataset/material del repo (sección 5.4): la guía debe abrir con un análisis coste/beneficio honesto, incluyendo los casos donde la ganancia es nula.
5. Contrasta el plan previo (`docs/plan_derivadas_simetria.md`) contra la realidad y documenta las correcciones.
6. Escribe la guía conforme a las secciones 7–11.
7. Autoevalúate con el checklist de la sección 12 y corrige antes de terminar.

## 5. Estado real del repositorio: hechos verificados que debes confirmar

Los siguientes hechos fueron verificados el 2026-07-08 inspeccionando el repo. Confírmalos tú mismo (los números de línea pueden derivar) y construye sobre ellos.

### 5.1 Pipeline de derivadas: archivos y responsabilidades

- **`Comparison/scripts/build_hamiltonian_derivative_stencils.py`** (~651 líneas). Genera las estructuras desplazadas. El bucle generador real es `for delta_ang → for atom_index → for axis → for sign in signs_for_method(method)` (líneas ~501–571). `displaced_positions()` (línea ~310) suma `signed_delta` a una componente cartesiana en Å. La CLI real es `--source-dataset-root`, `--output-stencil-root`, `--frozen-split`, `--split`, `--method {central,forward,backward}`, `--delta-ang`, `--atoms` (índices 0-based, admite rangos `0-3`), `--axes` (`x,y,z`), `--include-base`, `--overwrite`, más la familia de selección de snapshots base (`--base-sample-id`, `--max-base-snapshots`, `--base-selection-policy {all,first,adaptive_min_fraction}`, `--min-base-snapshots`, `--base-fraction`, `--base-selection-seed`). Escribe `derivative_stencil_manifest.json` con schema `hamiltonian_derivative_stencil_structures_v1`.
- **`shared/fdf_materialization.py`**. Contrato de estructura: `extract_fdf_structure()` → `FdfStructure` con `lattice_vectors_ang` como **lista de 3 tuplas (vectores de red como FILAS)**, `positions_ang` **cartesianas en Å** (ya normalizadas desde el `coordinate_format` del fdf), y `atom_species` que devuelve **índices de especie del fdf (1-based, campo `species_index` de `FdfAtom`)**, NO números atómicos. `materialize_sample_fdf()` escribe el `RUN.fdf` desplazado.
- **`Comparison/scripts/hamiltonian_derivative_stencil.py`** (~2842 líneas). Módulo core de contratos: dataclasses `DerivativeMetadata`, `DerivativeMatrixInput`, `DerivativeStencil`, `DerivativeStencilDiscovery`, `DerivativeValidationIssue`; `finite_difference_derivative()` (línea ~411) opera sobre matrices dispersas ya cargadas y devuelve `(left − right)/denominador` en CSR con metadatos de validación; `discover_derivative_stencils()` (línea ~1216) **reagrupa los stencils leyendo `structures/<sample>/metadata.json`** (no el manifest global) y espera el layout `structures/`, `siesta_hamiltonians/<sample>/*.HSX|*.TSHS`, `predicted_hamiltonians/<sample>/ML_prediction.HSX`; con `require_ml_predictions=False` soporta el modo de derivada directa (autograd). Incluye utilidades que NO debes reinventar: `sparse_frobenius_norm`, `sparse_hermiticity_defect`, `derivative_signal_to_noise_metrics`, `validate_derivative_geometry`, `direct_derivative_prediction_basename()` → `dH_pred_atom{A}_axis{I}`.
- **`Comparison/scripts/run_hamiltonian_derivative_siesta_references.py`** (~516 líneas). Ejecuta SIESTA por cada sample bajo `structures/`; es agnóstico a la física: si el manifest tiene menos estructuras, ejecuta menos jobs. Idealmente tu guía no lo toca (o casi).
- **`Comparison/scripts/run_hamiltonian_derivative_predictions.py`** (~1257 líneas). Predicciones ML por diferencias finitas (Graph2Mat/DeepH) sobre las estructuras desplazadas. Contiene `siesta_reference_supercell_order()` (línea ~387) que **ya lee el orden de superceldas del HSX de referencia** — punto de partida para el mapeo de offsets periódicos.
- **`Comparison/scripts/run_graph2mat_autograd_derivative_predictions.py`** y **`Comparison/scripts/run_deeph_autograd_derivative_predictions.py`**. Rutas de derivada directa por autograd: escriben un `dH_pred_atom{A}_axis{I}.npz` (CSR, layout `(n_orbitals, n_orbitals × n_supercells)`) + `.json` de provenance por cada par (átomo, eje) requerido, por snapshot base. **Ambas rutas están completas y verificadas** (auditoría del repo 2026-07-08). Son el vehículo barato para validar covariancia por simetría sin gastar SIESTA.
- **`Comparison/scripts/evaluate_hamiltonian_derivative_metrics.py`** (~1317 líneas). `load_hamiltonian_matrix()` (línea ~247) intenta `scipy.sparse.load_npz` primero y cae a `sisl.get_sile(path).read_hamiltonian().tocsr(0)`. Produce `derivative_matrix_metrics.csv`, `derivative_delta_stability.*`, `derivative_hermiticity.csv`, `derivative_summary.json`, etc. **Dato clave**: emite `derivative_onsite_offsite_metrics.json` con `{"available": false, "reason": "orbital_to_atom_mapping_unavailable"}` (línea ~388) — es decir, hoy NO existe en el pipeline un mapeo orbital→átomo; construirlo es prerrequisito compartido entre `U_g` y esa métrica pendiente (beneficio colateral que tu guía debe explotar).
- **`Comparison/scripts/validate_hamiltonian_derivative_geometry.py`** (~149 líneas). CLI fina sobre `discover_derivative_stencils` + `validate_derivative_geometry`. Es el patrón a imitar para un futuro `--symmetry-report` / validador de simetría.
- **`Comparison/scripts/g2m_deeph_runner.py`** (~8000+ líneas). Orquestador: stages `build_derivative_stencils`, `validate_derivative_stencils`, `run_derivative_siesta_reference`, `predict_derivative_graph2mat|deeph`, `derivative_metrics_*`. Normaliza el payload `derivative.{atoms,axes,delta_ang,method,base_selection_policy,...}` (líneas ~1500–1700, con validaciones `_require_derivative_field`) y construye el comando CLI del builder (líneas ~8245–8300). Comprueba el manifest con `_check_derivative_manifest` y calcula `_derivative_cost_summary`. Cualquier flag nuevo del builder necesita su clave espejo en este payload y en `docs/workflows.md`.
- **Tests**: suite `unittest` (NO pytest), se ejecuta `python3 -m unittest tests/test_X.py` (ver `docs/development.md`). Idioma de fixtures: datasets sintéticos en tmp dirs con `synthetic_base_fdf()` y `frozen_split_manifest.json` (ver `tests/test_build_hamiltonian_derivative_stencils.py`). Tests relevantes que tu guía debe mantener en verde y extender: `test_build_hamiltonian_derivative_stencils.py`, `test_hamiltonian_derivative_stencil.py`, `test_hamiltonian_derivative_geometry_validation.py`, `test_run_hamiltonian_derivative_siesta_references.py`, `test_run_hamiltonian_derivative_predictions.py`, `test_evaluate_hamiltonian_derivative_metrics.py`.

### 5.2 Contratos de datos y manifests

- Manifest del builder: `derivative_stencil_manifest.json`, schema `hamiltonian_derivative_stencil_structures_v1`, con listas `samples` (records por estructura) y `stencils` (records por grupo con `plus_sample_id`/`minus_sample_id`, `split_group_id`) y **contadores estrictos** (`stencils_per_base_snapshot`, `expected_structures_per_base_snapshot = (1 si include_base) + signos×átomos×ejes×deltas`, `expected_total_structure_samples`, `sample_count`, `stencil_count`) que tests y runner comprueban. Si reduces átomos por simetría, estos contadores deben seguir siendo coherentes con lo realmente generado — localiza con grep todos los lectores antes de proponer el cambio.
- Metadata por sample (`structures/<sample>/metadata.json`): claves que el discovery usa para reagrupar: `atom_index_zero_based`, `axis`, `axis_index`, `sign`, `delta_ang`, `base_sample_id`, `reference_base_sample_id`, `is_reference`, `split`, `split_group_id`, unidades explícitas (`hamiltonian_units: "eV"`, `displacement_units: "Ang"`, `derivative_units: "eV/Ang"`), `claim_status`, y hashes de compatibilidad heredados (`material_compatibility_hash`, `orbital_ordering_hash`, `neighbor_list_hash`, `sparsity_pattern_hash`, `basis_hash`, `pseudopotential_hash`). **Los metadatos de simetría por sample deben añadirse aquí** (además del manifest), porque el pipeline aguas abajo no relee el manifest global.
- Matrices: CSR con layout extendido `(n_orbitals, n_orbitals × n_supercells)` (formato sisl al leer HSX/TSHS con `.tocsr(0)`). Un elemento es `H_{μ0,νT}` con `T` un offset de celda; columna extendida = `cell_offset_index * n_orbitals + orbital_index`.

### 5.3 Entorno y dependencias

- Venv del repo: `.venv/` con `sisl 0.16.4` (requirement: `sisl[viz]>=0.15.0`), numpy, scipy, torch, e3nn, etc. (`requirements-graph2mat.txt`).
- **`spglib` NO está instalado ni es dependencia; `pymatgen` tampoco.** Tu guía debe introducir spglib como dependencia opcional: `--symmetry-mode off` (default) funciona sin ella; cualquier modo de simetría sin spglib debe fallar con mensaje accionable (`pip install spglib`).

### 5.4 Física y materiales reales

- Material insignia: **grafeno** (datasets `Comparison/datasets/graphene_w90_*`). Base SIESTA real (`PAO.Basis`): C con `n=2 l=0` (1ζ) y `n=2 l=1` (1ζ) → **4 orbitales por átomo (s + px,py,pz)**. Consecuencia dura: la reconstrucción bajo rotaciones no triviales exige rotación orbital `l=1` desde el primer día; el caso "base s-only" del plan previo solo cubre traslaciones puras aquí.
- **Especies ghost**: `ChemicalSpeciesLabel` declara `2 -1 Ghost-H` (Z = −1) en los fdf de grafeno w90 (a veces sin átomos que la usen). Implicación: para spglib usa como "types" los **índices de especie del fdf** (`FdfStructure.atom_species`, positivos y densos, distinguen ghosts), nunca números atómicos crudos.
- **Espín**: los fdf inspeccionados no llevan flags de espín/SOC → spin unpolarized. `U_g` sin parte espinorial en el MVP; la guía debe igualmente definir el guard de detección (leer `spin` vía sisl del HSX) y abortar/fallback si apareciera.
- **CRÍTICO — snapshots base MD**: los snapshots típicos son frames de MD térmicamente desplazados (ejemplo real: `Comparison/datasets/graphene_w90_joint/md_sweep_1_20/MD_steps/5/RUN.fdf` con posiciones tipo `1.476752894842 -0.016406970087 0.004242459738`). Para esos frames spglib con symprec estricto dará **P1 → ganancia CERO** por simetría de sitio. La celda raíz prístina (2 átomos C en (1/3,1/3,0) y (2/3,2/3,0), P6/mmm, 1 sola órbita, ejes x–y relacionados por el grupo puntual) sí es altamente simétrica. Otros materiales en `materials/`: `graphene_5x2`, `graphene_5x5` (supercelda prístina de 50 átomos → potencial ~50× si se desplaza la celda ideal), `h2o` (molécula C2v, sin offsets periódicos → banco de pruebas ideal para `U_g` sin Fase de offsets), `si_vacancy` (simetría parcial), `si_amorphous` (ninguna). Tu guía DEBE cuantificar la ganancia esperada por cada uno y concluir dónde compensa activar cada modo.

### 5.5 Puntos de anclaje arquitectónicos para la simetría (insights que el plan previo no tiene)

1. **Contrato de derivada directa ya existente.** El repo ya sabe consumir derivadas materializadas por (átomo, eje): `dH_pred_atom{A}_axis{I}.npz` + `.json` con discovery `require_ml_predictions=False` y `load_hamiltonian_matrix()` npz-first. La reconstrucción por simetría de la referencia SIESTA puede materializarse con un **contrato espejo del lado referencia** (p. ej. `dH_ref_atom{A}_axis{I}.npz` bajo un directorio tipo `siesta_reconstructed_derivatives/<base>/`), en vez de fabricar HSX sintéticos `H(±δ)`. Tu guía debe comparar explícitamente las dos estrategias de materialización — (a) reconstruir operandos `H(±δ)` equivalentes vs (b) reconstruir directamente `dH` (la transformación es lineal y conmuta con la resta de la diferencia finita) — y justificar la elegida con sus implicaciones para discovery, métricas, delta-stability y SNR.
2. **Validación de covarianza sin SIESTA.** Los jacobianos autograd de Graph2Mat y DeepH ya producen `dH/dR` por (átomo, eje) para el snapshot base. Como los modelos equivariantes (e3nn) son covariantes por construcción, sirven para testear la maquinaria `Q ⊗ U_g` (ecuación central) de forma barata antes de tocar SIESTA. Diseña la batería de validación empezando por ahí.
3. **Mapeo orbital→átomo.** No existe hoy (ver 5.1, `orbital_to_atoms_mapping_unavailable`). La fuente canónica es sisl leyendo el HSX/TSHS de referencia (`read_geometry()` → `geometry.atoms`, `Atom.orbitals` con números cuánticos `n,l,m,ζ` de `AtomicOrbital`) y/o el `.ORB_INDX` de SIESTA (nota: el builder actualmente NO copia `.ORB_INDX` — está en `OUTPUT_SUFFIXES` excluidos, línea ~28). La convención de orden de armónicos esféricos reales de SIESTA/sisl para `l=1` (orden en m: −1,0,+1 → py,pz,px, NO px,py,pz) debe **verificarse empíricamente con sisl en un HSX real del repo**, nunca asumirse; documenta en la guía el experimento concreto para fijarla.
4. **Convención de red y rotación cartesiana.** El repo guarda vectores de red como filas y posiciones cartesianas en Å. La conversión `W` (fraccionaria) → `Q` (cartesiana) depende de esa convención (`Q = Aᵀ W A⁻ᵀ` con A = matriz de filas, o equivalente — derívala tú y exige un unit test con celda no ortogonal, identidad, inversión y rotación de 90° en celda cúbica). El wrap de posiciones fraccionarias y la imagen mínima (`diff -= round(diff)`) son obligatorios en el `atom_map`.
5. **Ruido de referencia ya medido.** `evaluate_hamiltonian_derivative_metrics.py` ya calcula `_reference_noise_summary` y `derivative_signal_to_noise_metrics`. Los umbrales de aceptación de la reconstrucción (ε relativo Frobenius) deben calibrarse contra ese ruido SCF medido, no fijarse a ciegas; reutiliza la escala del plan previo (`<1e-5` excelente … `>1e-2` no aceptar) como punto de partida declarado.

## 6. Divergencias conocidas entre `docs/plan_derivadas_simetria.md` y el repo real

Tu guía debe corregir, como mínimo, estas (verifícalas y añade las que descubras):

1. La CLI propuesta en el plan (`--input base.fdf`) no existe: el builder real consume `--source-dataset-root` + frozen splits y snapshots base múltiples. Los flags de simetría deben integrarse en la CLI real y en el payload del runner (`derivative.symmetry.{mode,symprec,angle_tolerance,strict}` o equivalente).
2. El plan asume que el manifest es la fuente de verdad aguas abajo; en realidad `discover_derivative_stencils` relee `metadata.json` por sample. Los metadatos de simetría deben duplicarse por sample.
3. El plan propone "base s-only" como primer caso seguro representativo; el material insignia real tiene base s+p, así que el orden correcto de casos seguros es: (i) traslaciones puras (permutación de átomos + offsets), (ii) rotaciones signed-permutation con rotación p verificada, (iii) resto.
4. El plan ignora que los snapshots base productivos son frames MD sin simetría (P1) — el análisis coste/beneficio y los criterios "cuándo NO activar" deben liderar la guía, no ser un apéndice.
5. El plan ignora las especies ghost (Z=−1) y el hecho de que `atom_species` son índices de especie, no Z.
6. El plan no conoce el contrato de derivada directa (`dH_pred_atom{A}_axis{I}.npz`, `require_ml_predictions=False`) ni las rutas autograd ya completas — que cambian tanto la estrategia de materialización como la de validación.
7. El plan propone `pytest`-style tests sueltos; el repo usa `unittest` con fixtures sintéticos y comandos `python3 -m unittest tests/test_X.py`.
8. El plan no menciona los contadores estrictos del manifest ni los lectores que los validan.

## 7. Requisitos de contenido de la guía

### 7.1 Estructura obligatoria del documento

1. **Resumen ejecutivo** con decisión go/no-go razonada y la ganancia real cuantificada por dataset/material del repo (tabla: material → grupo espacial esperado → N átomos → órbitas → runs brute-force vs reducidos → speedup → ¿compensa?).
2. **Correcciones al plan previo** (sección 6 de este prompt, con evidencia `archivo:línea`).
3. **Arquitectura propuesta**: módulos nuevos (p. ej. `Comparison/scripts/symmetry_utils.py` para geometría pura y `Comparison/scripts/hamiltonian_symmetry.py` para la parte orbital — justifica la partición o propón otra mejor), dataclasses con campos exactos, serialización en manifest y metadata.json, integración con la CLI del builder y el payload del runner.
4. **Fases de implementación** (7.2), cada una con: objetivo, archivos a crear/tocar (rutas exactas), firmas de funciones concretas, cambios de CLI/manifest/payload, tests nuevos (nombre de archivo + casos), criterios de aceptación verificables por comando, esfuerzo estimado, riesgo, y qué NO se toca.
5. **Detalle matemático operativo**: convenciones de red/posiciones del repo, conversión frac↔cart y `W→Q` derivada para la convención real, `atom_map` con imagen mínima y unicidad, órbitas y representantes (incluida la reducción dentro del subconjunto `--atoms` pedido), selección de operación preferida (identidad > traslación pura > signed-permutation > resto), transformación de ejes con signo, construcción de `U_g` por bloques (permutación atómica × rotación real l=0/l=1 × mapeo de offsets), y el tratamiento del signo del desplazamiento (cuándo `+δ` del representante cubre `−δ` del equivalente).
6. **Estrategia de validación en 4 niveles** (sección 10).
7. **Cuándo NO activar simetría** (tabla de casos con acción: frames MD, si_amorphous, estructuras relajadas imperfectas, symprec inestable, espín/SOC, HSX sin info suficiente...), con política `--symmetry-strict` (fallar) vs default (fallback documentado `reconstruct → atoms → off` con warning).
8. **Riesgos y mitigaciones** específicos del repo (sección 9).
9. **Plan de regresión del modo legacy** (tests que fijan que sin flags nuevos el manifest y las estructuras son idénticos a los actuales).
10. **No-objetivos y puntos de extensión**: orbitales d/f (matrices de Wigner reales `D^(l)(Q)`), espín colineal/no colineal, SOC — fuera del MVP, con los guards y las interfaces donde se enchufarían.

### 7.2 Fases exigidas (esqueleto mínimo — desarróllalas y ajústalas con tu propio criterio tras inspeccionar el repo)

- **Fase 0 — Línea base y arnés de regresión.** Congelar el comportamiento actual con tests (conteos del manifest, ids generados, byte-igualdad de un manifest sintético). Sin funcionalidad nueva.
- **Fase 1 — Detección y reporte (`--symmetry-mode report`).** `symmetry_utils.py` con detección spglib (`cell = (lattice_rows, frac_positions, species_index_types)`), dataclasses `SymmetryOperation`/`SymmetryInfo`, `atom_map`, órbitas, conversión `W→Q`, clasificación de operaciones (identidad/traslación/signed-permutation/general), reporte con ahorro potencial y **scan de symprec** (`1e-5 … 1e-2`) con warning de inestabilidad. Sin cambios en la generación. Dependencia opcional spglib con error accionable.
- **Fase 2 — Reducción por átomos (`--symmetry-mode atoms`).** Generar solo representantes dentro del subconjunto `--atoms` pedido; manifest v2 (bump de schema a `hamiltonian_derivative_stencil_structures_v2` o campo aditivo — decide tras auditar TODOS los lectores con grep) + bloque `symmetry` global y por sample (`is_irreducible`, `representative_atom`, `covered_atoms`, `covered_dofs`, operación que cubre cada DOF); salida marcada como irreducible para que las métricas no la confundan con cobertura completa; integración en runner/payload y `docs/workflows.md`.
- **Fase 3 — Validación geométrica de simetría.** Tests unitarios (identidad, inversión, rotación 90°, celda no ortogonal, órbitas esperadas en grafeno prístino 2 átomos y en un frame MD → P1, ghost species, tolerancias extremas) + CLI de verificación estilo `validate_hamiltonian_derivative_geometry.py`.
- **Fase 4 — Infraestructura orbital compartida.** `BasisMetadata`/`OrbitalInfo` leídos vía sisl del HSX de referencia (mapeo orbital→átomo, `n,l,m,ζ`, spin mode); verificación empírica documentada de la convención de orden de `m` para `l=1`; guard `assert_basis_compatible_with_symmetry`. Beneficio colateral exigido: dejar diseñado cómo esta misma metadata desbloquea `derivative_onsite_offsite_metrics`.
- **Fase 5 — Reconstrucción para casos seguros (`--symmetry-mode reconstruct`).** `hamiltonian_symmetry.py`: `U_g` para permutación atómica + rotación p; reconstrucción `D_{jβ}H` desde representantes; materialización con el contrato elegido en 5.5.1; validación de covarianza primero con autograd (Graph2Mat/DeepH), después con pares SIESTA explícitos; umbrales calibrados contra `reference_noise`. Restricciones del MVP explícitas (sin SOC, sin espín, sin d/f, operaciones soportadas).
- **Fase 6 — Offsets periódicos.** Decode/encode de columnas extendidas, mapeo `T' = W·T + Δ` usando posiciones y wraps, reutilizando `siesta_reference_supercell_order()`; validar covariancia del Hamiltoniano completo (no solo derivadas) en grafeno primitivo; h2o como control sin offsets.
- **Fase 7 — Integración final.** Payload del runner, `docs/workflows.md`, coste (`_derivative_cost_summary` reflejando runs ahorrados), smoke test estilo `smoke_adaptive_derivative_selection.py`, política de fallback, documentación de limitaciones.

Para cada fase, decide y justifica el orden si difiere de este esqueleto.

### 7.3 Nivel de detalle exigido

- Firmas de funciones completas (nombre, parámetros tipados, retorno) para todo lo nuevo; nada de "añadir una función que haga X".
- Contratos JSON exactos (bloques de manifest/metadata con todos los campos y ejemplos).
- Cada test nuevo: archivo, nombre del método de test, qué fixture usa, qué asserta, comando para correrlo.
- Cada criterio de aceptación: un comando ejecutable y su salida esperada.
- Referencias `archivo:línea` del código actual en cada punto de integración.

## 8. Restricciones duras de diseño

1. **El modo legacy es sagrado**: sin flags de simetría, el builder debe producir manifest y estructuras idénticos a los actuales (mismos ids, mismos conteos, mismo schema si optas por campo aditivo). Toda simetría es opt-in.
2. `finite_difference_derivative()` no se modifica: la reconstrucción es una capa posterior separada.
3. spglib es dependencia opcional; sisl ya está y es la vía obligada para leer base orbital y Hamiltonianos (no introduzcas pymatgen).
4. Compatibilidad con ambos `source_model` (`graph2mat`, `deeph`) y con las cuatro rutas de predicción (FD y autograd de cada uno).
5. Todo artefacto reconstruido lleva provenance completo en su `.json` (p. ej. `derivative_provenance: "symmetry_reconstructed"`, operación usada — `W`, `w`, `Q` —, átomo/eje representante, symprec, versión de spglib) y las métricas deben poder distinguir derivadas medidas de reconstruidas.
6. Política de seguridad explícita: con `--symmetry-strict`, cualquier caso no soportado (espín, SOC, l>1, offsets no mapeables, base incompatible, atom_map ambiguo) falla con mensaje claro; sin strict, fallback documentado con warning al siguiente modo seguro.
7. Nunca proponer activar reconstrucción por defecto: el gate para promoverla es pasar la batería de validación de la sección 10 en los materiales de prueba.

## 9. Riesgos que la guía debe tratar explícitamente (mínimo)

- symprec vs estructuras relajadas/MD: detección inestable, simetrías falsas con tolerancia laxa; mitigación: scan de symprec + validación brute-force antes de confiar.
- Frames MD → P1: ganancia nula; la guía debe recomendar activar simetría solo para stencils sobre celdas prístinas/relajadas simétricas (p. ej. diagnósticos tipo fonón, sweeps de δ sobre celda ideal, datasets 5x5 prístinos).
- Ghost species y especies declaradas sin átomos.
- Convención de orden orbital de SIESTA/sisl para p (y el error silencioso si se asume px,py,pz).
- Ruido SCF rompe la igualdad `H(gR) = U_g H(R) U_g†` a nivel numérico: umbrales calibrados contra el ruido de referencia ya medido; k-grid y mesh cutoff como fuentes de asimetría numérica.
- Offsets periódicos: un término `H_{μ0,νT}` puede mapear a `T'` fuera del conjunto de superceldas almacenado → política (error strict / acumular evidencia / descartar con contador).
- Contadores estrictos del manifest y lectores downstream que asumen `6N`.
- Doble cobertura de DOFs (dos operaciones cubren el mismo (átomo, eje) con distinto signo) → determinismo en la elección.
- Delta-stability y SNR: las filas reconstruidas heredan el δ del representante; documentar cómo se reflejan en `derivative_delta_stability` y `dh_signal_to_noise_ratio` sin contaminar los summaries.

## 10. Validación exigida (la guía debe detallarla como fase transversal)

1. **Nivel 0 — geometría pura** (sin matrices): `g(R) = R` módulo red y permutación; unicidad del `atom_map`; `Q` ortogonal; órbitas = `equivalent_atoms` de spglib.
2. **Nivel 1 — covarianza con autograd** (barato, sin SIESTA): comprobar `D_{jβ}H_pred = Σ_α Q_{βα} U_g D_{iα}H_pred U_g†` con los jacobianos Graph2Mat/DeepH existentes sobre una celda prístina.
3. **Nivel 2 — covarianza SIESTA**: `H(g·R_δ)` calculado explícitamente vs `U_g H(R_δ) U_g†` (valida `U_g` directamente, más fuerte que validar solo derivadas).
4. **Nivel 3 — derivadas reconstruidas vs brute force**: en un sistema pequeño (h2o y grafeno primitivo), correr modo completo y modo simetría, comparar con error relativo Frobenius por (átomo, eje); criterios calibrados contra `reference_noise`; barrido de δ (0.005/0.01/0.02 Å) para separar fallos de mapeo de ruido SCF.

## 11. Formato y estilo del documento final

- Markdown, español, code blocks en inglés, ecuaciones en notación de texto clara (evita LaTeX roto como el de los docs de contexto).
- Tabla resumen de fases al principio (fase, objetivo, archivos, riesgo, esfuerzo, dependencias entre fases).
- Referencias tipo `Comparison/scripts/build_hamiltonian_derivative_stencils.py:501` en cada afirmación sobre código existente.
- Sin relleno: cada párrafo debe cambiar lo que haría el implementador.

## 12. Checklist de autoevaluación (respóndelo antes de terminar; si algún punto falla, corrige)

- [ ] ¿He leído los dos docs de contexto completos y el código real de los 8+ archivos del pipeline?
- [ ] ¿Cada afirmación sobre el repo lleva `archivo:línea` verificada?
- [ ] ¿La guía abre con la cuantificación honesta de ganancia por dataset, incluyendo los casos P1 sin ganancia?
- [ ] ¿Incluye la sección "Correcciones al plan previo" con las 8 divergencias mínimas verificadas?
- [ ] ¿Cada fase tiene archivos exactos, firmas completas, tests con nombre y comando, y criterios de aceptación ejecutables?
- [ ] ¿El modo legacy queda protegido por tests de regresión definidos en Fase 0?
- [ ] ¿La estrategia de materialización de derivadas reconstruidas está decidida y justificada frente a la alternativa?
- [ ] ¿La validación usa primero autograd (barato) y después SIESTA, con umbrales calibrados contra el ruido de referencia existente?
- [ ] ¿Están cubiertos ghost species, base s+p, convención de orden orbital p, offsets periódicos y política strict/fallback?
- [ ] ¿Quedan explícitos los no-objetivos (d/f, espín, SOC) con sus puntos de extensión y guards?
- [ ] ¿Un agente de programación podría ejecutar la Fase 1 mañana sin hacerme ninguna pregunta?
