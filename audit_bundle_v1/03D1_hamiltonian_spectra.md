# Dossier 3A — Hamiltonianos, overlap y espectros

## Objeto de revisión

Auditar H y S, compatibilidad de matrices, hermiticidad, problema generalizado, métricas sparse/espectrales, stencils, unidades, delta y autograd.

## Condiciones del contexto

- Contexto ciego: no contiene informes de auditoría previos ni prompts históricos.
- Las líneas prefijadas `NNNNN |` conservan la numeración del archivo fuente.
- Los extractos Python omiten funciones operativas sin semántica científica directa.
- Ausencia de un archivo o función en este dossier significa `no evaluable`, no `correcto`.

## `docs/derivadas_simetria.md`

SHA-256: `dd2da4657995e15476c433254a5d09064da8c2c46b0287627d4154c70191044d`

```md
00001 | Sí: **es posible y potencialmente muy útil**, pero en vuestro caso no es una optimización trivial de “generar menos ficheros” solamente. Como estáis derivando el **Hamiltoniano completo de SIESTA** respecto al desplazamiento de un átomo, explotar simetría implica dos cosas:
00002 | 
00003 | 1. generar menos geometrías desplazadas;
00004 | 2. reconstruir correctamente las derivadas faltantes transformando el Hamiltoniano y su base orbital.
00005 | 
00006 | La parte 1 es fácil. La parte 2 es la delicada.
00007 | 
00008 | ---
00009 | 
00010 | ## 1. Qué está pasando ahora en vuestro código
00011 | 
00012 | Según el resumen de tu agente, el flujo actual es:
00013 | 
00014 | [
00015 | D_{i\alpha}H
00016 | ============
00017 | 
00018 | \frac{\partial H}{\partial R_{i\alpha}}
00019 | \approx
00020 | \frac{
00021 | H(\mathbf R+\delta \mathbf e_{i\alpha})
00022 | ---------------------------------------
00023 | 
00024 | H(\mathbf R-\delta \mathbf e_{i\alpha})
00025 | }{2\delta}.
00026 | ]
00027 | 
00028 | Para cada átomo (i), cada eje (\alpha=x,y,z), y cada signo (+\delta,-\delta), se genera una geometría independiente. Por tanto, con diferencias centrales:
00029 | 
00030 | [
00031 | N_{\text{runs}}
00032 | ===============
00033 | 
00034 | # 2 \times 3 \times N_{\text{átomos}}
00035 | 
00036 | 6N.
00037 | ]
00038 | 
00039 | Esto ya está bien implementado físicamente: mover un átomo en un eje y leer el Hamiltoniano completo te da una derivada completa del Hamiltoniano respecto a ese grado de libertad. El coste viene de repetirlo para todos los átomos y ejes, aunque muchos sean equivalentes por simetría.
00040 | 
00041 | ---
00042 | 
00043 | ## 2. Qué simetría se puede explotar
00044 | 
00045 | La idea física es esta: si dos átomos son equivalentes por una operación de simetría del cristal, sus derivadas no son independientes.
00046 | 
00047 | Una operación de simetría espacial se puede escribir como
00048 | 
00049 | [
00050 | g = (Q, \mathbf t),
00051 | ]
00052 | 
00053 | donde (Q) es una rotación, reflexión, inversión, rotación impropia, etc., y (\mathbf t) es una traslación.
00054 | 
00055 | Si la operación (g) transforma el átomo (i) en el átomo (j),
00056 | 
00057 | [
00058 | g(i)=j,
00059 | ]
00060 | 
00061 | entonces desplazar el átomo (i) en la dirección (\alpha) es equivalente, por simetría, a desplazar el átomo (j) en la dirección transformada (Q\mathbf e_\alpha).
00062 | 
00063 | Para derivadas vectoriales simples, como fuerzas, esto se traduce en relaciones tensoriales relativamente directas. Para el Hamiltoniano, la relación correcta es más rica porque el Hamiltoniano está escrito en una base de orbitales atómicos.
00064 | 
00065 | Formalmente:
00066 | 
00067 | [
00068 | H(g\mathbf R)
00069 | =============
00070 | 
00071 | U_g H(\mathbf R) U_g^\dagger,
00072 | ]
00073 | 
00074 | donde (U_g) representa cómo la operación de simetría actúa sobre la base orbital: permutación de átomos, permutación de orbitales, posible mezcla de orbitales (p,d,f), signos, traslaciones de celda, etc.
00075 | 
00076 | Entonces las derivadas cumplen:
00077 | 
00078 | [
00079 | \frac{\partial H}{\partial R_{j\beta}}
00080 | ======================================
00081 | 
00082 | \sum_\alpha
00083 | Q_{\beta\alpha}
00084 | ,
00085 | U_g
00086 | \frac{\partial H}{\partial R_{i\alpha}}
00087 | U_g^\dagger.
00088 | ]
00089 | 
00090 | Esta es la ecuación central para implementar la reconstrucción por simetría.
00091 | 
00092 | ---
00093 | 
00094 | ## 3. Reducción esperada del número de stencils
00095 | 
00096 | Sin simetría:
00097 | 
00098 | [
00099 | N_{\text{runs}} = 6N.
00100 | ]
00101 | 
00102 | Si solo aprovechas átomos equivalentes pero sigues calculando los tres ejes para cada átomo representante:
00103 | 
00104 | [
00105 | N_{\text{runs}} = 6N_{\text{inequiv}},
00106 | ]
00107 | 
00108 | donde (N_{\text{inequiv}}) es el número de átomos inequivalentes por simetría.
00109 | 
00110 | Por ejemplo, si tienes una supercelda perfecta con 64 átomos pero todos son copias equivalentes de 2 átomos de la celda primitiva, podrías pasar de:
00111 | 
00112 | [
00113 | 6 \times 64 = 384
00114 | ]
00115 | 
00116 | a
00117 | 
00118 | [
00119 | 6 \times 2 = 12
00120 | ]
00121 | 
00122 | cálculos SIESTA por valor de (\delta), siempre que puedas reconstruir correctamente las derivadas faltantes.
00123 | 
00124 | También se puede intentar reducir ejes. Por ejemplo, si la simetría local de un sitio relaciona (x), (y) y (z), quizá no haga falta desplazar las tres direcciones. Pero esta segunda reducción es más delicada. Si una rotación transforma (x) exactamente en (y), es sencillo. Si transforma (x) en una combinación lineal de (x,y,z), entonces necesitas reconstruir mediante combinaciones lineales de derivadas, no simplemente copiando un stencil.
00125 | 
00126 | Finalmente, en algunos casos también puedes reducir el signo (+\delta/-\delta). Si una operación de simetría transforma la geometría (+\delta) en la geometría (-\delta), entonces podrías obtener (H(-\delta)) transformando (H(+\delta)). Pero, de nuevo, para el Hamiltoniano necesitas aplicar (U_g), no solo copiar el fichero.
00127 | 
00128 | ---
00129 | 
00130 | ## 4. Herramienta recomendada para detectar simetría
00131 | 
00132 | La opción más razonable es usar **spglib** o una capa encima como **pymatgen**. Spglib proporciona operaciones de simetría ((W,w)), átomos equivalentes, posiciones de Wyckoff, grupo espacial y órbitas cristalográficas. En su dataset aparecen explícitamente `rotations`, `translations`, `equivalent_atoms`, `wyckoffs`, `site_symmetry_symbols`, etc. ([spglib.readthedocs.io][1])
00133 | 
00134 | La entrada típica de spglib en Python es:
00135 | 
00136 | ```python
00137 | cell = (lattice, scaled_positions, atomic_numbers)
00138 | dataset = spglib.get_symmetry_dataset(cell, symprec=1e-5)
00139 | ```
00140 | 
00141 | La propia documentación de spglib describe el formato `cell = (lattice, positions, numbers)`, con posiciones fraccionarias y números atómicos. ([spglib.readthedocs.io][2])
00142 | 
00143 | Pymatgen también puede ser útil si ya usáis estructuras tipo `Structure`. Su `SpacegroupAnalyzer` usa spglib internamente y permite obtener información de simetría con una tolerancia `symprec`; la documentación menciona que tolerancias más laxas pueden ser necesarias para estructuras relajadas con pequeños desplazamientos numéricos. ([pymatgen][3])
00144 | 
00145 | ---
00146 | 
00147 | ## 5. Nivel mínimo viable de implementación
00148 | 
00149 | Yo no empezaría implementando la reconstrucción completa del Hamiltoniano desde el día uno. Haría esto en fases.
00150 | 
00151 | ### Fase 1: reporte de simetría sin cambiar los cálculos
00152 | 
00153 | Añadiría un modo:
00154 | 
00155 | ```bash
00156 | python build_hamiltonian_derivative_stencils.py \
00157 |   --symmetry-report \
00158 |   --symprec 1e-3
00159 | ```
00160 | 
00161 | Este modo debería imprimir algo como:
00162 | 
00163 | ```text
00164 | Detected space group: Fd-3m
00165 | Number of atoms: 64
00166 | Equivalent atom orbits:
00167 |   orbit 0: atoms [0, 4, 8, ...]
00168 |   orbit 1: atoms [1, 5, 9, ...]
00169 | Current finite-difference runs: 384
00170 | Atom-orbit reduced runs: 12
00171 | Potential speedup: 32x
00172 | ```
00173 | 
00174 | Aquí todavía no reduces nada, solo verificas que la simetría detectada tiene sentido.
00175 | 
00176 | Pseudocódigo:
00177 | 
00178 | ```python
00179 | import spglib
00180 | import numpy as np
00181 | 
00182 | def detect_symmetry(lattice, frac_positions, atomic_numbers, symprec=1e-3):
00183 |     cell = (lattice, frac_positions, atomic_numbers)
00184 |     dataset = spglib.get_symmetry_dataset(cell, symprec=symprec)
00185 | 
00186 |     rotations = np.array(dataset["rotations"])
00187 |     translations = np.array(dataset["translations"])
00188 |     equivalent_atoms = np.array(dataset["equivalent_atoms"])
00189 | 
00190 |     reps = sorted(set(equivalent_atoms.tolist()))
00191 |     orbits = {
00192 |         rep: np.where(equivalent_atoms == rep)[0].tolist()
00193 |         for rep in reps
00194 |     }
00195 | 
00196 |     return {
00197 |         "dataset": dataset,
00198 |         "rotations_frac": rotations,
00199 |         "translations_frac": translations,
00200 |         "equivalent_atoms": equivalent_atoms,
00201 |         "orbits": orbits,
00202 |     }
00203 | ```
00204 | 
00205 | ---
00206 | 
00207 | ## 6. Fase 2: generar stencils solo para átomos representantes
00208 | 
00209 | Modificaría `build_hamiltonian_derivative_stencils.py` para que, opcionalmente, reemplace la lista de átomos por los representantes inequivalentes.
00210 | 
00211 | Actualmente el bucle conceptual es:
00212 | 
00213 | ```python
00214 | for delta_ang in delta_ang_values:
00215 |     for atom_index in atom_indices_zero_based:
00216 |         for axis in axes:
00217 |             for sign in signs_for_method(method):
00218 |                 generate_displaced_structure(atom_index, axis, sign, delta_ang)
00219 | ```
00220 | 
00221 | Con simetría atómica sería:
00222 | 
00223 | ```python
00224 | if use_symmetry:
00225 |     atom_indices_zero_based = get_inequivalent_atom_representatives(...)
00226 | ```
00227 | 
00228 | y luego:
00229 | 
00230 | ```python
00231 | for delta_ang in delta_ang_values:
00232 |     for atom_index in inequivalent_atom_representatives:
00233 |         for axis in axes:
00234 |             for sign in signs_for_method(method):
00235 |                 generate_displaced_structure(atom_index, axis, sign, delta_ang)
00236 | ```
00237 | 
00238 | Esto reduce la generación y el número de runs SIESTA.
00239 | 
00240 | Pero cuidado: esto solo es válido si aguas abajo aceptas que solo tienes derivadas para los representantes. Si el código posterior espera derivadas para todos los átomos, necesitas reconstruirlas.
00241 | 
00242 | ---
00243 | 
00244 | ## 7. Fase 3: guardar metadatos de simetría en el manifest
00245 | 
00246 | El manifest de stencils debería guardar no solo `atom`, `axis`, `sign`, `delta`, sino también información como:
00247 | 
00248 | ```json
00249 | {
00250 |   "symmetry": {
00251 |     "enabled": true,
00252 |     "symprec": 0.001,
00253 |     "spacegroup_number": 225,
00254 |     "international_symbol": "Fm-3m",
00255 |     "equivalent_atoms": [0, 0, 0, 0, 4, 4],
00256 |     "operations": [
00257 |       {
00258 |         "id": 0,
00259 |         "rotation_frac": [[1,0,0],[0,1,0],[0,0,1]],
00260 |         "translation_frac": [0,0,0],
00261 |         "atom_map": [0,1,2,3]
00262 |       }
00263 |     ],
00264 |     "representative_atoms": [0,4]
00265 |   }
00266 | }
00267 | ```
00268 | 
00269 | Necesitas especialmente `atom_map`: para cada operación (g), qué átomo se transforma en cuál.
00270 | 
00271 | Spglib da las operaciones ((W,w)), pero conviene construir explícitamente el mapa atómico:
00272 | 
00273 | ```python
00274 | def build_atom_map(frac_positions, atomic_numbers, W, w, tol=1e-5):
00275 |     n = len(frac_positions)
00276 |     atom_map = [-1] * n
00277 | 
00278 |     for i in range(n):
00279 |         f_new = W @ frac_positions[i] + w
00280 |         f_new = f_new % 1.0
00281 | 
00282 |         candidates = []
00283 |         for j in range(n):
00284 |             if atomic_numbers[j] != atomic_numbers[i]:
00285 |                 continue
00286 | 
00287 |             diff = f_new - frac_positions[j]
00288 |             diff -= np.round(diff)  # minimum image in fractional coords
00289 | 
00290 |             if np.linalg.norm(diff) < tol:
00291 |                 candidates.append(j)
00292 | 
00293 |         if len(candidates) != 1:
00294 |             raise RuntimeError(
00295 |                 f"Could not map atom {i} uniquely under symmetry operation"
00296 |             )
00297 | 
00298 |         atom_map[i] = candidates[0]
00299 | 
00300 |     return atom_map
00301 | ```
00302 | 
00303 | ---
00304 | 
00305 | ## 8. Conversión de rotaciones fraccionarias a cartesianas
00306 | 
00307 | Spglib expresa las operaciones de simetría normalmente en coordenadas fraccionarias. Pero las derivadas de tu código parecen estar en ejes cartesianos (x,y,z), porque `displaced_positions()` modifica una columna de `positions` en Å.
00308 | 
00309 | Por tanto necesitas convertir la rotación fraccionaria (W) a rotación cartesiana (Q).
00310 | 
00311 | Si usas la convención:
00312 | 
00313 | [
00314 | \mathbf r = A \mathbf f,
00315 | ]
00316 | 
00317 | donde (A) tiene como columnas los vectores de red, entonces:
00318 | 
00319 | [
00320 | Q = A W A^{-1}.
00321 | ]
00322 | 
00323 | En código:
00324 | 
00325 | ```python
00326 | def frac_rotation_to_cartesian(W, lattice):
00327 |     # lattice columns are a, b, c
00328 |     A = np.array(lattice).T
00329 |     Q = A @ W @ np.linalg.inv(A)
00330 |     return Q
00331 | ```
00332 | 
00333 | Si en vuestro código la matriz de red está almacenada con vectores como filas, hay que ajustar la fórmula. Esta parte debe testearse con casos simples: identidad, inversión, rotación de 90 grados en una celda cúbica.
00334 | 
00335 | ---
00336 | 
00337 | ## 9. Reconstrucción de derivadas atómicas
00338 | 
00339 | Supongamos que calculas explícitamente:
00340 | 
00341 | [
00342 | D_{i x}H,\quad D_{i y}H,\quad D_{i z}H
00343 | ]
00344 | 
00345 | para un átomo representante (i).
00346 | 
00347 | Ahora quieres la derivada para un átomo equivalente (j). Buscas una operación (g) tal que:
00348 | 
00349 | [
00350 | g(i)=j.
00351 | ]
00352 | 
00353 | Entonces:
00354 | 
00355 | [
00356 | D_{j\beta}H
00357 | ===========
00358 | 
00359 | \sum_{\alpha=x,y,z}
00360 | Q_{\beta\alpha}
00361 | ,
00362 | U_g
00363 | D_{i\alpha}H
00364 | U_g^\dagger.
00365 | ]
00366 | 
00367 | En pseudocódigo:
00368 | 
00369 | ```python
00370 | def reconstruct_atom_derivatives(rep_derivs, operation, orbital_transform):
00371 |     """
00372 |     rep_derivs: dict axis -> sparse matrix dH/dR_rep_axis
00373 |                 axes: 0=x, 1=y, 2=z
00374 | 
00375 |     operation.rotation_cart: Q
00376 |     orbital_transform: U_g
00377 | 
00378 |     returns target_derivs: dict axis -> sparse matrix
00379 |     """
00380 |     Q = operation.rotation_cart
00381 |     U = orbital_transform
00382 | 
00383 |     transformed = {}
00384 |     for alpha in range(3):
00385 |         transformed[alpha] = U @ rep_derivs[alpha] @ U.T.conjugate()
00386 | 
00387 |     target_derivs = {}
00388 |     for beta in range(3):
00389 |         acc = None
00390 |         for alpha in range(3):
00391 |             term = Q[beta, alpha] * transformed[alpha]
00392 |             acc = term if acc is None else acc + term
00393 |         target_derivs[beta] = acc
00394 | 
00395 |     return target_derivs
00396 | ```
00397 | 
00398 | Esta es la reconstrucción conceptualmente correcta.
00399 | 
00400 | ---
00401 | 
00402 | ## 10. El gran problema: construir (U_g)
00403 | 
00404 | Aquí está la dificultad principal.
00405 | 
00406 | Para fuerzas, (U_g) no aparece. Para el Hamiltoniano sí.
00407 | 
00408 | El Hamiltoniano de SIESTA está escrito en una base de orbitales atómicos localizados. SIESTA usa orbitales atómicos numéricos de soporte finito como base, junto con pseudopotenciales norm-conserving y malla real, según la descripción metodológica del código. ([arXiv][4])
00409 | 
00410 | Eso significa que una simetría no solo mueve átomos. También transforma orbitales:
00411 | 
00412 | * los orbitales (s) son escalares;
00413 | * los orbitales (p_x,p_y,p_z) se mezclan como vectores;
00414 | * los orbitales (d) se mezclan mediante una representación (5\times5);
00415 | * los orbitales (f) mediante una representación (7\times7);
00416 | * orbitales con distintas zetas o polarización deben mantenerse separados;
00417 | * si hay espín, SOC o no-colinealidad, la transformación incluye también el espacio de espín.
00418 | 
00419 | Por eso no basta con decir:
00420 | 
00421 | ```python
00422 | H_target = H_source[permuted_indices, permuted_indices]
00423 | ```
00424 | 
00425 | Eso solo sería correcto para orbitales tipo (s) o para simetrías puramente traslacionales que no roten la orientación orbital.
00426 | 
00427 | Para una implementación completa necesitas construir una matriz bloque-diagonal/permutacional (U_g). A nivel conceptual:
00428 | 
00429 | ```text
00430 | U_g =
00431 |   permutación de átomos
00432 |   × permutación de orbitales dentro de cada átomo
00433 |   × rotación de armónicos esféricos reales
00434 |   × posible traslación de celda / imagen periódica
00435 |   × posible parte de espín
00436 | ```
00437 | 
00438 | Este es el motivo por el que recomiendo una implementación por fases.
00439 | 
00440 | ---
00441 | 
00442 | ## 11. Implementación por niveles de dificultad
00443 | 
00444 | ### Nivel A: solo detectar y reportar simetría
00445 | 
00446 | Muy recomendable. Riesgo bajo. No cambia resultados.
00447 | 
00448 | Objetivo:
00449 | 
00450 | ```bash
00451 | --symmetry-report
00452 | ```
00453 | 
00454 | Salida:
00455 | 
00456 | ```text
00457 | Current calculations: 6N
00458 | With atom symmetry: 6N_ineq
00459 | With possible sign symmetry: ...
00460 | ```
00461 | 
00462 | Esto ya te dice si merece la pena implementar más.
00463 | 
00464 | ---
00465 | 
00466 | ### Nivel B: reducción por átomos equivalentes, sin reconstrucción completa
00467 | 
00468 | Útil si solo quieres calcular derivadas para representantes, por ejemplo para diagnóstico, análisis parcial o entrenamiento reducido.
00469 | 
00470 | Añadir:
00471 | 
00472 | ```bash
00473 | --use-symmetry-atoms
00474 | ```
00475 | 
00476 | que internamente cambie:
00477 | 
00478 | ```python
00479 | atom_indices_zero_based
00480 | ```
00481 | 
00482 | por:
00483 | 
00484 | ```python
00485 | representative_atoms
00486 | ```
00487 | 
00488 | Pero habría que documentar claramente:
00489 | 
00490 | > Este modo no produce la matriz completa de derivadas para todos los átomos. Produce solo derivadas irreducibles.
00491 | 
00492 | No debería sustituir al workflow completo salvo que el resto del pipeline se adapte.
00493 | 
00494 | ---
00495 | 
00496 | ### Nivel C: reconstrucción por simetrías simples
00497 | 
00498 | Este sería el primer modo realmente útil para acelerar SIESTA manteniendo una salida completa.
00499 | 
00500 | Restringiría inicialmente las operaciones permitidas a casos seguros:
00501 | 
00502 | 1. traslaciones puras;
00503 | 2. inversión;
00504 | 3. rotaciones/reflexiones que sean matrices de permutación con signo en la base cartesiana;
00505 | 4. orbitales (s), o bases donde podáis implementar signos/permutaciones de (p) de forma controlada;
00506 | 5. sin SOC;
00507 | 6. sin no-colinealidad;
00508 | 7. sin magnetismo complicado.
00509 | 
00510 | Ejemplo de rotación cartesiana tipo permutación con signo:
00511 | 
00512 | [
00513 | Q =
00514 | \begin{pmatrix}
00515 | 0 & 1 & 0 \
00516 | -1 & 0 & 0 \
00517 | 0 & 0 & 1
00518 | \end{pmatrix}.
00519 | ]
00520 | 
00521 | Esto transforma (x\rightarrow -y), (y\rightarrow x), (z\rightarrow z). Aquí la reconstrucción de ejes es relativamente limpia.
00522 | 
00523 | Puedes detectar estas operaciones con:
00524 | 
00525 | ```python
00526 | def is_signed_permutation_matrix(Q, tol=1e-8):
00527 |     Q_round = np.round(Q).astype(int)
00528 |     if not np.allclose(Q, Q_round, atol=tol):
00529 |         return False
00530 | 
00531 |     if not np.all(np.isin(Q_round, [-1, 0, 1])):
00532 |         return False
00533 | 
00534 |     return (
00535 |         np.all(np.sum(np.abs(Q_round), axis=0) == 1)
00536 |         and np.all(np.sum(np.abs(Q_round), axis=1) == 1)
00537 |     )
00538 | ```
00539 | 
00540 | Este nivel puede dar bastante ganancia en cristales simples y evita entrar desde el principio en rotaciones arbitrarias de orbitales (d/f).
00541 | 
00542 | ---
00543 | 
00544 | ### Nivel D: reconstrucción completa del Hamiltoniano
00545 | 
00546 | Este sería el objetivo final.
00547 | 
00548 | Necesitas:
00549 | 
00550 | 1. leer la geometría y la base orbital;
00551 | 2. saber qué orbital pertenece a qué átomo;
00552 | 3. saber sus números cuánticos (l,m,\zeta);
00553 | 4. construir la representación real de la rotación para cada canal (l);
00554 | 5. aplicar la permutación de átomos;
00555 | 6. aplicar el mapeo de imágenes periódicas/superceldas del formato HSX/TSHS;
00556 | 7. transformar matrices dispersas:
00557 | 
00558 | [
00559 | H' = U_g H U_g^\dagger.
00560 | ]
00561 | 
00562 | La librería **sisl** puede ser útil porque tiene soporte para leer Hamiltonianos HSX de SIESTA mediante `read_hamiltonian`, y también expone lectura de geometría/basis en algunos lectores de HSX. ([sisl.readthedocs.io][5])
00563 | 
00564 | ---
00565 | 
00566 | ## 12. Qué archivos tocaría en vuestro repo
00567 | 
00568 | Según tu resumen, tocaría principalmente estos:
00569 | 
00570 | ### `build_hamiltonian_derivative_stencils.py`
00571 | 
00572 | Añadiría:
00573 | 
00574 | ```bash
00575 | --use-symmetry
00576 | --symmetry-report
00577 | --symprec
00578 | --angle-tolerance
00579 | --symmetry-mode atom|dof|full
00580 | --symmetry-strict
00581 | ```
00582 | 
00583 | Responsabilidades nuevas:
00584 | 
00585 | * leer estructura base;
00586 | * detectar simetrías;
00587 | * construir órbitas atómicas;
00588 | * decidir qué desplazamientos son irreducibles;
00589 | * escribir metadatos de simetría al manifest;
00590 | * generar solo las estructuras irreducibles.
00591 | 
00592 | ---
00593 | 
00594 | ### `hamiltonian_derivative_stencil.py`
00595 | 
00596 | Aquí está la fórmula de diferencias finitas. Habría que añadir una capa posterior:
00597 | 
00598 | ```python
00599 | finite_difference_derivative(...)
00600 | symmetry_reconstruct_derivatives(...)
00601 | ```
00602 | 
00603 | Flujo recomendado:
00604 | 
00605 | ```text
00606 | H(+δ), H(-δ)
00607 |         ↓
00608 | derivadas irreducibles D_rep
00609 |         ↓
00610 | reconstrucción por simetría
00611 |         ↓
00612 | derivadas completas D_all
00613 | ```
00614 | 
00615 | No mezclaría la reconstrucción dentro de la función básica de diferencias finitas. Mantendría separadas las responsabilidades.
00616 | 
00617 | ---
00618 | 
00619 | ### `run_hamiltonian_derivative_siesta_references.py`
00620 | 
00621 | Idealmente no debería cambiar demasiado. Este script debería limitarse a ejecutar SIESTA sobre las geometrías existentes en el manifest.
00622 | 
00623 | Si el manifest contiene menos geometrías, ejecutará menos jobs.
00624 | 
00625 | ---
00626 | 
00627 | ### `run_hamiltonian_derivative_predictions.py`
00628 | 
00629 | Igual que SIESTA. Si se usa finite difference legacy, podrá beneficiarse automáticamente de menos estructuras.
00630 | 
00631 | Pero para Graph2Mat tenéis una alternativa aún mejor: el path autograd ya calcula derivadas de forma más directa. Este modo no sustituye a SIESTA como referencia, pero sí puede servir como banco de pruebas para verificar la covariancia por simetría.
00632 | 
00633 | ---
00634 | 
00635 | ## 13. Arquitectura propuesta
00636 | 
00637 | Yo introduciría un módulo nuevo:
00638 | 
00639 | ```text
00640 | Comparison/scripts/symmetry_utils.py
00641 | ```
00642 | 
00643 | con funciones como:
00644 | 
00645 | ```python
00646 | @dataclass
00647 | class SymmetryOperation:
00648 |     op_id: int
00649 |     rotation_frac: np.ndarray
00650 |     translation_frac: np.ndarray
00651 |     rotation_cart: np.ndarray
00652 |     atom_map: list[int]
00653 | 
00654 | @dataclass
00655 | class SymmetryInfo:
00656 |     symprec: float
00657 |     spacegroup_number: int
00658 |     international_symbol: str
00659 |     equivalent_atoms: np.ndarray
00660 |     representative_atoms: list[int]
00661 |     operations: list[SymmetryOperation]
00662 | ```
00663 | 
00664 | Funciones:
00665 | 
00666 | ```python
00667 | def detect_symmetry_from_structure(structure, symprec, angle_tolerance):
00668 |     ...
00669 | 
00670 | def build_atom_maps(frac_positions, atomic_numbers, rotations, translations):
00671 |     ...
00672 | 
00673 | def find_operation_mapping_atom(sym_info, src_atom, dst_atom):
00674 |     ...
00675 | 
00676 | def get_representative_atoms(sym_info):
00677 |     ...
00678 | 
00679 | def reduce_atom_list_by_symmetry(atom_indices, sym_info):
00680 |     ...
00681 | 
00682 | def write_symmetry_metadata(manifest, sym_info):
00683 |     ...
00684 | ```
00685 | 
00686 | Y otro módulo, más avanzado:
00687 | 
00688 | ```text
00689 | Comparison/scripts/hamiltonian_symmetry.py
00690 | ```
00691 | 
00692 | con:
00693 | 
00694 | ```python
00695 | def build_orbital_transform(operation, basis_metadata):
00696 |     ...
00697 | 
00698 | def transform_hamiltonian_derivative(dH, operation, orbital_transform):
00699 |     ...
00700 | 
00701 | def reconstruct_all_derivatives(rep_derivatives, sym_info, basis_metadata):
00702 |     ...
00703 | ```
00704 | 
00705 | Separar estos dos módulos es importante porque `symmetry_utils.py` es geometría pura, mientras que `hamiltonian_symmetry.py` depende de la representación orbital.
00706 | 
00707 | ---
00708 | 
00709 | ## 14. Pseudocódigo del flujo completo
00710 | 
00711 | ```python
00712 | def build_derivative_stencils_with_symmetry(base_structure, atoms, axes, deltas, method):
00713 |     sym_info = detect_symmetry_from_structure(
00714 |         base_structure,
00715 |         symprec=args.symprec,
00716 |         angle_tolerance=args.angle_tolerance,
00717 |     )
00718 | 
00719 |     if args.symmetry_report:
00720 |         print_symmetry_report(sym_info, atoms, axes, deltas, method)
00721 |         return
00722 | 
00723 |     if args.use_symmetry:
00724 |         atoms_to_displace = reduce_atom_list_by_symmetry(atoms, sym_info)
00725 |     else:
00726 |         atoms_to_displace = atoms
00727 | 
00728 |     manifest = []
00729 | 
00730 |     for delta in deltas:
00731 |         for atom in atoms_to_displace:
00732 |             for axis in axes:
00733 |                 for sign in signs_for_method(method):
00734 |                     structure = displaced_positions(
00735 |                         base_structure,
00736 |                         atom_index_zero_based=atom,
00737 |                         axis_index=axis,
00738 |                         signed_delta=sign * delta,
00739 |                     )
00740 | 
00741 |                     manifest.append({
00742 |                         "atom": atom,
00743 |                         "axis": axis,
00744 |                         "sign": sign,
00745 |                         "delta": delta,
00746 |                         "is_irreducible": True,
00747 |                     })
00748 | 
00749 |     manifest_metadata = {
00750 |         "symmetry": serialize_symmetry_info(sym_info),
00751 |         "full_atoms_requested": atoms,
00752 |         "irreducible_atoms_used": atoms_to_displace,
00753 |     }
00754 | 
00755 |     write_manifest(manifest, manifest_metadata)
00756 | ```
00757 | 
00758 | Después, para reconstruir:
00759 | 
00760 | ```python
00761 | def compute_and_reconstruct_derivatives(manifest, matrices, sym_info, basis):
00762 |     rep_derivatives = compute_finite_differences_for_irreducible_atoms(
00763 |         manifest,
00764 |         matrices,
00765 |     )
00766 | 
00767 |     if not manifest["metadata"]["symmetry"]["enabled"]:
00768 |         return rep_derivatives
00769 | 
00770 |     all_derivatives = {}
00771 | 
00772 |     for rep_atom in sym_info.representative_atoms:
00773 |         rep_dH = {
00774 |             axis: rep_derivatives[(rep_atom, axis)]
00775 |             for axis in [0, 1, 2]
00776 |         }
00777 | 
00778 |         for target_atom in sym_info.orbit(rep_atom):
00779 |             op = find_operation_mapping_atom(sym_info, rep_atom, target_atom)
00780 |             U = build_orbital_transform(op, basis)
00781 | 
00782 |             target_dH = reconstruct_atom_derivatives(
00783 |                 rep_dH,
00784 |                 operation=op,
00785 |                 orbital_transform=U,
00786 |             )
00787 | 
00788 |             for axis in [0, 1, 2]:
00789 |                 all_derivatives[(target_atom, axis)] = target_dH[axis]
00790 | 
00791 |     return all_derivatives
00792 | ```
00793 | 
00794 | ---
00795 | 
00796 | ## 15. Validación imprescindible
00797 | 
00798 | No activaría esto por defecto hasta pasar una batería fuerte de tests.
00799 | 
00800 | ### Test 1: simetría geométrica
00801 | 
00802 | Para cada operación (g):
00803 | 
00804 | [
00805 | g(\mathbf R) = \mathbf R
00806 | ]
00807 | 
00808 | módulo vectores de red y permutación de átomos.
00809 | 
00810 | Código conceptual:
00811 | 
00812 | ```python
00813 | for op in sym_info.operations:
00814 |     for i in atoms:
00815 |         j = op.atom_map[i]
00816 |         assert same_position_mod_cell(
00817 |             op.W @ frac_pos[i] + op.w,
00818 |             frac_pos[j],
00819 |             tol=symprec,
00820 |         )
00821 | ```
00822 | 
00823 | ---
00824 | 
00825 | ### Test 2: comparar derivadas reconstruidas contra derivadas calculadas explícitamente
00826 | 
00827 | En un sistema pequeño, correr el modo viejo completo y el modo nuevo con simetría.
00828 | 
00829 | Para cada átomo/eje:
00830 | 
00831 | [
00832 | \epsilon_{i\alpha}
00833 | ==================
00834 | 
00835 | \frac{
00836 | |D^{\text{full}}_{i\alpha}H
00837 | ---------------------------
00838 | 
00839 | D^{\text{sym}}*{i\alpha}H|
00840 | }{
00841 | |D^{\text{full}}*{i\alpha}H|+\epsilon
00842 | }.
00843 | ]
00844 | 
00845 | Usar normas de matrices dispersas, por ejemplo Frobenius:
00846 | 
00847 | ```python
00848 | def relative_sparse_error(A, B, eps=1e-12):
00849 |     diff = A - B
00850 |     return sparse_norm(diff) / (sparse_norm(A) + eps)
00851 | ```
00852 | 
00853 | Criterios razonables iniciales:
00854 | 
00855 | ```text
00856 | relative error < 1e-4  excelente
00857 | relative error < 1e-3  probablemente aceptable
00858 | relative error > 1e-2  sospechoso
00859 | ```
00860 | 
00861 | Depende mucho de convergencia SCF, tolerancia de simetría, delta, malla real y ruido numérico.
00862 | 
00863 | ---
00864 | 
00865 | ### Test 3: covariancia de los Hamiltonianos desplazados
00866 | 
00867 | Para una geometría desplazada explícita y su imagen por simetría:
00868 | 
00869 | [
00870 | H(g\mathbf R_\delta)
00871 | \stackrel{?}{=}
00872 | U_g H(\mathbf R_\delta) U_g^\dagger.
00873 | ]
00874 | 
00875 | Este test valida directamente (U_g). Es más fuerte que probar solo las derivadas.
00876 | 
00877 | ---
00878 | 
00879 | ### Test 4: convergencia con (\delta)
00880 | 
00881 | Comparar varios valores de desplazamiento:
00882 | 
00883 | [
00884 | \delta = 0.005,\ 0.01,\ 0.02\ \text{Å}.
00885 | ]
00886 | 
00887 | La derivada debe ser estable en una ventana razonable. Si la reconstrucción por simetría falla solo para algunos (\delta), probablemente hay ruido SCF o un problema de mapeo de orbitales/imágenes.
00888 | 
00889 | ---
00890 | 
00891 | ## 16. Riesgos y casos problemáticos
00892 | 
00893 | ### 1. Estructuras relajadas imperfectas
00894 | 
00895 | Una estructura relajada puede no estar exactamente en la simetría ideal. Si `symprec` es demasiado estricto, spglib detectará poca simetría. Si es demasiado laxo, inventará simetrías falsas.
00896 | 
00897 | Recomendación:
00898 | 
00899 | ```text
00900 | symprec inicial: 1e-3 Å
00901 | probar también: 1e-4, 1e-2 Å
00902 | ```
00903 | 
00904 | Y guardar siempre en el log qué grupo espacial se detecta.
00905 | 
00906 | ---
00907 | 
00908 | ### 2. Defectos, superficies e interfaces
00909 | 
00910 | En presencia de defectos, adsorbatos, vacantes, superficies o heteroestructuras, la simetría puede reducirse drásticamente.
00911 | 
00912 | No pasa nada: el algoritmo simplemente encontrará más átomos inequivalentes.
00913 | 
00914 | ---
00915 | 
00916 | ### 3. Magnetismo
00917 | 
00918 | Si hay orden ferromagnético colineal simple, algunas simetrías espaciales siguen siendo válidas.
00919 | 
00920 | Si hay antiferromagnetismo, espines no colineales o SOC, las simetrías espaciales ordinarias pueden no ser suficientes. Necesitarías simetría magnética o al menos comprobar que la operación también transforma correctamente los momentos magnéticos.
00921 | 
00922 | Spglib soporta estructuras con momentos magnéticos en el formato de celda extendido, pero la reconstrucción del Hamiltoniano con espín sigue siendo bastante más delicada. ([spglib.readthedocs.io][2])
00923 | 
00924 | ---
00925 | 
00926 | ### 4. Spin-orbit coupling
00927 | 
00928 | Con SOC, (U_g) debe actuar también sobre espinores. No basta con rotar orbitales espaciales.
00929 | 
00930 | Para un MVP, yo desactivaría simetría si detectas SOC.
00931 | 
00932 | ---
00933 | 
00934 | ### 5. Orbitales (p,d,f)
00935 | 
00936 | Este es probablemente el mayor obstáculo técnico. Si la base tiene orbitales con (l>0), las rotaciones pueden mezclar orbitales.
00937 | 
00938 | Una traslación pura no mezcla (p_x,p_y,p_z). Una rotación sí.
00939 | 
00940 | Por eso, para empezar, aceptaría solo operaciones cuya acción orbital podáis implementar y validar.
00941 | 
00942 | ---
00943 | 
00944 | ### 6. Formato periódico del HSX/TSHS
00945 | 
00946 | Según tu agente, las matrices tienen forma dispersa tipo:
00947 | 
00948 | [
00949 | (n_{\text{orb}},\ n_{\text{orb}}\times n_{\text{supercells}}).
00950 | ]
00951 | 
00952 | Esto significa que no solo tienes índices orbitales dentro de la celda base, sino también interacciones con imágenes periódicas. Una operación de simetría puede mandar un término (H_{a0,bT}) a otro término (H_{a'0,b'T'}).
00953 | 
00954 | Por tanto, (U_g) no es solo una matriz (n_{\text{orb}}\times n_{\text{orb}}) si quieres transformar todo el objeto extendido tal como está almacenado. Necesitas entender y transformar también los offsets de celda.
00955 | 
00956 | ---
00957 | 
00958 | ## 17. Recomendación práctica
00959 | 
00960 | Mi recomendación sería implementar en este orden:
00961 | 
00962 | ### Paso 1
00963 | 
00964 | Añadir `--symmetry-report`.
00965 | 
00966 | Esto es rápido, seguro y os dirá el speedup máximo posible.
00967 | 
00968 | ### Paso 2
00969 | 
00970 | Añadir `--use-symmetry-atoms` para generar solo stencils de átomos inequivalentes, pero marcar la salida como “irreducible only”.
00971 | 
00972 | Esto permite empezar a reducir cálculos para análisis parciales.
00973 | 
00974 | ### Paso 3
00975 | 
00976 | Implementar reconstrucción solo para operaciones de simetría simples:
00977 | 
00978 | * traslaciones puras;
00979 | * simetrías que solo permutan átomos equivalentes sin rotar orbitales;
00980 | * quizá inversión/signos si la base lo permite.
00981 | 
00982 | ### Paso 4
00983 | 
00984 | Añadir soporte completo de (U_g) para orbitales (s,p,d,f) y offsets periódicos.
00985 | 
00986 | Este es el paso más caro, pero también el que permitiría explotar de verdad el grupo espacial completo.
00987 | 
00988 | ---
00989 | 
00990 | ## 18. Veredicto final
00991 | 
00992 | Sí, **la simetría puede reducir mucho el tiempo de generación y ejecución de stencils**. En vuestro repo, la optimización más clara sería pasar de:
00993 | 
00994 | [
00995 | 6N
00996 | ]
00997 | 
00998 | a aproximadamente:
00999 | 
01000 | [
01001 | 6N_{\text{inequiv}}
01002 | ]
01003 | 
01004 | runs de SIESTA por valor de (\delta), como primera aproximación.
01005 | 
01006 | Pero para el caso concreto de derivadas del Hamiltoniano, no basta con desplazar menos átomos. Hay que reconstruir correctamente:
01007 | 
01008 | [
01009 | D_{j\beta}H
01010 | ===========
01011 | 
01012 | \sum_\alpha
01013 | Q_{\beta\alpha}
01014 | U_g
01015 | D_{i\alpha}H
01016 | U_g^\dagger.
01017 | ]
01018 | 
01019 | La detección de átomos equivalentes es relativamente sencilla con spglib. Lo difícil es construir y validar (U_g) para el Hamiltoniano de SIESTA, especialmente con orbitales (p,d,f), espín, SOC y términos periódicos del HSX/TSHS.
01020 | 
01021 | La implementación más sensata es incremental: primero reporte, luego reducción por representantes, luego reconstrucción para simetrías simples, y finalmente reconstrucción completa del Hamiltoniano.
01022 | 
01023 | [1]: https://spglib.readthedocs.io/en/stable/dataset.html "Spglib dataset — Spglib v2.7.0"
01024 | [2]: https://spglib.readthedocs.io/en/stable/python-interface.html "Spglib for Python — Spglib v2.7.0"
01025 | [3]: https://pymatgen.org/pymatgen.symmetry.html "pymatgen.symmetry package — pymatgen 2026.5.18 documentation"
01026 | [4]: https://arxiv.org/abs/2006.01270?utm_source=chatgpt.com "SIESTA: recent developments and applications"
01027 | [5]: https://sisl.readthedocs.io/en/latest/api/io/generated/sisl.io.siesta.hsxSileSiesta.html?utm_source=chatgpt.com "sisl.io.siesta.hsxSileSiesta - Read the Docs"
```

## `Comparison/scripts/g2m_deeph_metrics.py`

SHA-256: `5a8c3e100246dde745422f37b20afee32368f9cee18484a1906ad2cf7ea97176`

```py
00001 | #!/usr/bin/env python3
00002 | """Common metric aggregation for the Graph2Mat-vs-DeepH benchmark."""
00003 | 
00004 | from __future__ import annotations
00005 | 
00006 | import argparse
00007 | import csv
00008 | import json
00009 | import math
00010 | import os
00011 | import shutil
00012 | import time
00013 | from dataclasses import dataclass
00014 | from pathlib import Path
00015 | from typing import Any
00016 | 
00017 | from deeph_prediction_adapter import (
00018 |     EQUIVALENCE_INVALID_MISSING_REFERENCE,
00019 |     EQUIVALENCE_PROVEN_RAW_GLOBAL,
00020 |     EQUIVALENCE_STATUS_FAILED,
00021 |     EQUIVALENCE_STATUS_PROVEN,
00022 |     PROVEN_ADAPTER_EQUIVALENCE_STATUSES,
00023 |     equivalence_scope_from_adapter_status,
00024 |     equivalence_status_from_adapter_status,
00025 | )
00026 | 
00027 | 
00028 | SCHEMA = "graph2mat_deeph_common_metrics_v1"
00029 | FORBIDDEN_REFERENCE_NAMES = {"ML_prediction.HSX"}
00030 | STATUS_VALUES = {
00031 |     "valid_joint_one_pass_dataset",
00032 |     "valid_reused_joint_dataset",
00033 |     "valid_repaired_dataset_with_warning",
00034 |     "invalid_missing_artifacts",
00035 |     "invalid_incompatible_splits",
00036 |     "invalid_incompatible_basis_or_pseudos",
00037 |     "invalid_prediction_format",
00038 |     "diagnostic_only",
00039 | }
00040 | PRIMARY_METRIC = "h_mae_eV_mean"
00041 | 
00042 | H_MAE_METRIC_GROUP = {
00043 |     "id": "h_mae",
00044 |     "title": "Hamiltonian MAE",
00045 |     "y_title": "MAE eV",
00046 |     "metrics": [{"key": "h_mae_eV_mean", "label": "H MAE", "unit": "eV", "direction": "lower_is_better"}],
00047 | }
00048 | H_RMSE_METRIC_GROUP = {
00049 |     "id": "h_rmse",
00050 |     "title": "Hamiltonian RMSE",
00051 |     "y_title": "RMSE eV",
00052 |     "metrics": [{"key": "h_rmse_eV_mean", "label": "H RMSE", "unit": "eV", "direction": "lower_is_better"}],
00053 | }
00054 | H_MSE_METRIC_GROUP = {
00055 |     "id": "h_mse",
00056 |     "title": "Hamiltonian MSE",
00057 |     "y_title": "MSE eV^2",
00058 |     "metrics": [{"key": "h_mse_eV2_mean", "label": "H MSE", "unit": "eV^2", "direction": "lower_is_better"}],
00059 | }
00060 | R2_METRIC_GROUP = {
00061 |     "id": "r2",
00062 |     "title": "Sparse support R2",
00063 |     "y_title": "R2",
00064 |     "metrics": [{"key": "r2_mean", "label": "R2", "unit": "", "direction": "higher_is_better"}],
00065 | }
00066 | FROBENIUS_METRIC_GROUP = {
00067 |     "id": "frobenius",
00068 |     "title": "Relative Frobenius",
00069 |     "y_title": "relative error",
00070 |     "metrics": [
00071 |         {
00072 |             "key": "relative_frobenius_mean",
00073 |             "label": "Relative Frobenius",
00074 |             "unit": "",
00075 |             "direction": "lower_is_better",
00076 |         },
00077 |     ],
00078 | }
00079 | HERMITICITY_METRIC_GROUP = {
00080 |     "id": "hermiticity",
00081 |     "title": "Predicted Hamiltonian hermiticity",
00082 |     "y_title": "Hermiticity residual",
00083 |     "metrics": [
00084 |         {
00085 |             "key": "hermiticity_pred_mean",
00086 |             "label": "Hermiticity residual",
00087 |             "unit": "",
00088 |             "direction": "lower_is_better",
00089 |         },
00090 |     ],
00091 | }
00092 | SPECTRAL_GLOBAL_METRIC_GROUP = {
00093 |     "id": "spectral_global",
00094 |     "title": "Global spectral RMSE",
00095 |     "y_title": "RMSE eV",
00096 |     "metrics": [
00097 |         {"key": "global_rmse_eV_mean", "label": "Global RMSE", "unit": "eV", "direction": "lower_is_better"},
00098 |     ],
00099 | }
00100 | SPECTRAL_LOW_ENERGY_METRIC_GROUP = {
00101 |     "id": "spectral_low_energy",
00102 |     "title": "Low-energy spectral RMSE",
00103 |     "y_title": "RMSE eV",
00104 |     "metrics": [
00105 |         {"key": "low_energy_rmse_eV_mean", "label": "Low-energy RMSE", "unit": "eV", "direction": "lower_is_better"},
00106 |     ],
00107 | }
00108 | SPECTRAL_FERMI_METRIC_GROUP = {
00109 |     "id": "spectral_fermi",
00110 |     "title": "Fermi-window spectral RMSE",
00111 |     "y_title": "RMSE eV",
00112 |     "metrics": [
00113 |         {"key": "fermi_window_rmse_eV_mean", "label": "Fermi-window RMSE", "unit": "eV", "direction": "lower_is_better"},
00114 |     ],
00115 | }
00116 | SPECTRAL_FRONTIER_METRIC_GROUP = {
00117 |     "id": "spectral_frontier",
00118 |     "title": "Frontier-window spectral RMSE",
00119 |     "y_title": "RMSE eV",
00120 |     "metrics": [
00121 |         {"key": "frontier_window_rmse_eV_mean", "label": "Frontier RMSE", "unit": "eV", "direction": "lower_is_better"},
00122 |     ],
00123 | }
00124 | DOS_MAE_METRIC_GROUP = {
00125 |     "id": "dos_mae",
00126 |     "title": "DOS Fermi-window MAE",
00127 |     "y_title": "DOS MAE",
00128 |     "metrics": [
00129 |         {
00130 |             "key": "dos_mae_500_fermi_window_mean",
00131 |             "label": "DOS MAE 500 Fermi window",
00132 |             "unit": "",
00133 |             "direction": "lower_is_better",
00134 |         },
00135 |     ],
00136 | }
00137 | DOS_WASSERSTEIN_METRIC_GROUP = {
00138 |     "id": "dos_wasserstein",
00139 |     "title": "DOS Wasserstein distance",
00140 |     "y_title": "Wasserstein eV",
00141 |     "metrics": [
00142 |         {
00143 |             "key": "dos_wasserstein_eV_mean",
00144 |             "label": "DOS Wasserstein",
00145 |             "unit": "eV",
00146 |             "direction": "lower_is_better",
00147 |         },
00148 |     ],
00149 | }
00150 | VALIDATION_RERUN_METRIC_GROUP = {
00151 |     "id": "validation_rerun",
00152 |     "title": "Final-seed validation metric",
00153 |     "y_title": "Validation metric",
00154 |     "metrics": [
00155 |         {
00156 |             "key": "validation_metric_value",
00157 |             "label": "Validation metric",
00158 |             "unit": "",
00159 |             "direction": "lower_is_better",
00160 |         },
00161 |     ],
00162 | }
00163 | DEEPH_LIVE_LOSS_METRIC_GROUP = {
00164 |     "id": "deeph_live_loss",
00165 |     "title": "DeepH live training loss",
00166 |     "y_title": "Loss",
00167 |     "metrics": [
00168 |         {
00169 |             "key": "deeph_live_train_loss",
00170 |             "label": "Train loss",
00171 |             "unit": "",
00172 |             "direction": "lower_is_better",
00173 |         },
00174 |         {
00175 |             "key": "deeph_live_val_loss",
00176 |             "label": "Val loss",
00177 |             "unit": "",
00178 |             "direction": "lower_is_better",
00179 |         },
00180 |         {
00181 |             "key": "deeph_live_best_val_loss",
00182 |             "label": "Best val loss",
00183 |             "unit": "",
00184 |             "direction": "lower_is_better",
00185 |         },
00186 |     ],
00187 | }
00188 | GPU_HOURS_METRIC_GROUP = {
00189 |     "id": "gpu_hours",
00190 |     "title": "GPU-hours",
00191 |     "y_title": "GPU-hours",
00192 |     "metrics": [
00193 |         {
00194 |             "key": "gpu_hours_total",
00195 |             "label": "GPU-hours",
00196 |             "unit": "h",
00197 |             "direction": "lower_is_better",
00198 |         },
00199 |     ],
00200 | }
00201 | PEAK_GPU_MEMORY_METRIC_GROUP = {
00202 |     "id": "peak_gpu_memory",
00203 |     "title": "Peak GPU memory",
00204 |     "y_title": "Peak VRAM MB",
00205 |     "metrics": [
00206 |         {
00207 |             "key": "peak_gpu_memory_mb",
00208 |             "label": "Peak VRAM",
00209 |             "unit": "MB",
00210 |             "direction": "lower_is_better",
00211 |         },
00212 |     ],
00213 | }
00214 | PEAK_RSS_METRIC_GROUP = {
00215 |     "id": "peak_rss",
00216 |     "title": "Peak process RAM",
00217 |     "y_title": "Peak RSS MB",
00218 |     "metrics": [
00219 |         {
00220 |             "key": "peak_rss_mb",
00221 |             "label": "Peak RSS",
00222 |             "unit": "MB",
00223 |             "direction": "lower_is_better",
00224 |         },
00225 |     ],
00226 | }
00227 | CPU_TIME_METRIC_GROUP = {
00228 |     "id": "cpu_time",
00229 |     "title": "CPU time",
00230 |     "y_title": "CPU seconds",
00231 |     "metrics": [
00232 |         {
00233 |             "key": "cpu_time_seconds_total",
00234 |             "label": "CPU time",
00235 |             "unit": "s",
00236 |             "direction": "lower_is_better",
00237 |         },
00238 |     ],
00239 | }
00240 | THROUGHPUT_METRIC_GROUP = {
00241 |     "id": "throughput",
00242 |     "title": "Training throughput",
00243 |     "y_title": "Samples/s",
00244 |     "metrics": [
00245 |         {
00246 |             "key": "samples_per_second",
00247 |             "label": "Samples/s",
00248 |             "unit": "samples/s",
00249 |             "direction": "higher_is_better",
00250 |         },
00251 |     ],
00252 | }
00253 | COMMON_METRIC_GROUPS = [
00254 |     H_MAE_METRIC_GROUP,
00255 |     H_RMSE_METRIC_GROUP,
00256 |     H_MSE_METRIC_GROUP,
00257 |     R2_METRIC_GROUP,
00258 |     FROBENIUS_METRIC_GROUP,
00259 |     HERMITICITY_METRIC_GROUP,
00260 |     SPECTRAL_GLOBAL_METRIC_GROUP,
00261 |     SPECTRAL_LOW_ENERGY_METRIC_GROUP,
00262 |     SPECTRAL_FERMI_METRIC_GROUP,
00263 |     SPECTRAL_FRONTIER_METRIC_GROUP,
00264 |     DOS_MAE_METRIC_GROUP,
00265 |     DOS_WASSERSTEIN_METRIC_GROUP,
00266 |     VALIDATION_RERUN_METRIC_GROUP,
00267 |     DEEPH_LIVE_LOSS_METRIC_GROUP,
00268 |     GPU_HOURS_METRIC_GROUP,
00269 |     PEAK_GPU_MEMORY_METRIC_GROUP,
00270 |     PEAK_RSS_METRIC_GROUP,
00271 |     CPU_TIME_METRIC_GROUP,
00272 |     THROUGHPUT_METRIC_GROUP,
00273 | ]
00274 | DERIVATIVE_METRIC_GROUPS = [
00275 |     {
00276 |         "id": "derivative_mae",
00277 |         "title": "dH/dR MAE",
00278 |         "y_title": "eV/Ang",
00279 |         "metrics": [{"key": "dh_mae_union_eV_per_Ang_mean", "label": "dH MAE", "unit": "eV/Ang", "direction": "lower_is_better"}],
00280 |         "diagnostic_only": True,
00281 |     },
00282 |     {
00283 |         "id": "derivative_rmse",
00284 |         "title": "dH/dR RMSE",
00285 |         "y_title": "eV/Ang",
00286 |         "metrics": [{"key": "dh_rmse_union_eV_per_Ang_mean", "label": "dH RMSE", "unit": "eV/Ang", "direction": "lower_is_better"}],
00287 |         "diagnostic_only": True,
00288 |     },
00289 |     {
00290 |         "id": "derivative_support_f1",
00291 |         "title": "dH/dR Support F1",
00292 |         "y_title": "F1",
00293 |         "metrics": [{"key": "dh_support_f1_mean", "label": "dH support F1", "unit": "", "direction": "higher_is_better"}],
00294 |         "diagnostic_only": True,
00295 |     },
00296 |     {
00297 |         "id": "derivative_relative_frobenius",
00298 |         "title": "dH/dR Relative Frobenius",
00299 |         "y_title": "Relative error",
00300 |         "metrics": [{"key": "dh_relative_frobenius_ref_mean", "label": "dH rel. Frobenius", "unit": "", "direction": "lower_is_better"}],
00301 |         "diagnostic_only": True,
00302 |     },
00303 | ]
00304 | 
00305 | 
00306 | @dataclass(frozen=True)
00307 | class StagedGraph2MatMetrics:
00308 |     result_dir: Path
00309 |     sample_ids: list[str]
00310 | 
00311 | 
00312 | @dataclass(frozen=True)
00313 | class StagedDeepHMetrics:
00314 |     processed_dir: Path
00315 |     predictions_dir: Path
00316 |     sample_ids: list[str]
00317 | 
00318 | 
00319 | def read_json(path: Path) -> dict[str, Any]:
00320 |     if not path.exists():
00321 |         return {}
00322 |     payload = json.loads(path.read_text(encoding="utf-8"))
00323 |     return payload if isinstance(payload, dict) else {}
00324 | 
00325 | 
00326 | def write_json(path: Path, payload: dict[str, Any]) -> None:
00327 |     path.parent.mkdir(parents=True, exist_ok=True)
00328 |     path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
00329 | 
00330 | 
00331 | def json_safe(value: Any) -> Any:
00332 |     if isinstance(value, Path):
00333 |         return str(value)
00334 |     if isinstance(value, float):
00335 |         return value if math.isfinite(value) else None
00336 |     if isinstance(value, dict):
00337 |         return {str(key): json_safe(item) for key, item in value.items()}
00338 |     if isinstance(value, (list, tuple)):
00339 |         return [json_safe(item) for item in value]
00340 |     return value
00341 | 
00342 | 
00343 | def read_csv_rows(path: Path) -> list[dict[str, str]]:
00344 |     if not path.exists():
00345 |         return []
00346 |     with path.open(encoding="utf-8", newline="") as handle:
00347 |         return list(csv.DictReader(handle))
00348 | 
00349 | 
00350 | def write_csv_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
00351 |     path.parent.mkdir(parents=True, exist_ok=True)
00352 |     if fieldnames is None:
00353 |         fieldnames = sorted({key for row in rows for key in row}) or ["status"]
00354 |     with path.open("w", encoding="utf-8", newline="") as handle:
00355 |         writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
00356 |         writer.writeheader()
00357 |         writer.writerows([{key: json_safe(value) for key, value in row.items()} for row in rows])
00358 | 
00359 | 
00360 | def cross_structure_metadata(dataset_manifest_path: Path | None) -> dict[str, Any]:
00361 |     if dataset_manifest_path is None:
00362 |         return {}
00363 |     provenance = read_json(Path(dataset_manifest_path).parent / "cross_structure_dataset_provenance.json")
00364 |     if not provenance:
00365 |         return {}
00366 |     metadata = provenance.get("cross_structure_metadata")
00367 |     if isinstance(metadata, dict) and metadata:
00368 |         return metadata
00369 |     return {
00370 |         "evaluation_mode": provenance.get("evaluation_mode"),
00371 |         "transfer_direction": provenance.get("transfer_direction"),
00372 |         "source_atom_counts": provenance.get("source_atom_counts") or [],
00373 |         "target_atom_counts": provenance.get("target_atom_counts") or [],
00374 |         "source_system_labels": provenance.get("source_system_labels") or [],
00375 |         "target_system_labels": provenance.get("target_system_labels") or [],
00376 |         "source_split_hash": provenance.get("source_split_hash"),
00377 |         "target_split_hash": provenance.get("target_split_hash"),
00378 |         "composite_split_hash": provenance.get("composite_split_hash"),
00379 |     }
00380 | 
00381 | 
00382 | def finite_or_none(value: Any) -> float | None:
00383 |     result = number(value)
00384 |     return result if math.isfinite(result) else None
00385 | 
00386 | 
00387 | def number(value: Any) -> float:
00388 |     try:
00389 |         result = float(value)
00390 |     except (TypeError, ValueError):
00391 |         return math.nan
00392 |     return result if math.isfinite(result) else math.nan
00393 | 
00394 | 
00395 | def mean(values: list[float]) -> float:
00396 |     clean = [value for value in values if math.isfinite(value)]
00397 |     return sum(clean) / len(clean) if clean else math.nan
00398 | 
00399 | 
00400 | def _link_or_copy(src: Path, dst: Path) -> None:
00401 |     dst.parent.mkdir(parents=True, exist_ok=True)
00402 |     if dst.exists() or dst.is_symlink():
00403 |         if dst.is_dir() and not dst.is_symlink():
00404 |             shutil.rmtree(dst)
00405 |         else:
00406 |             dst.unlink()
00407 |     try:
00408 |         os.symlink(os.path.relpath(src, dst.parent), dst)
00409 |     except OSError:
00410 |         if src.is_dir():
00411 |             shutil.copytree(src, dst, symlinks=True)
00412 |         else:
00413 |             shutil.copy2(src, dst)
00414 | 
00415 | 
00416 | def split_rows(frozen_split_manifest: dict[str, Any], split: str = "test") -> list[dict[str, Any]]:
00417 |     return [dict(row) for row in frozen_split_manifest.get("rows") or [] if row.get("split") == split]
00418 | 
00419 | 
00420 | def row_sample_id(row: dict[str, Any]) -> str:
00421 |     value = str(row.get("sample_id") or row.get("graph2mat_sample_id") or row.get("deeph_sample_id") or "").strip()
00422 |     if value:
00423 |         return value
00424 |     sample_dir = str(row.get("sample_dir") or "").strip()
00425 |     if sample_dir:
00426 |         return Path(sample_dir).name
00427 |     raise RuntimeError(f"Frozen split row is missing a stable sample id: {row}")
00428 | 
00429 | 
00430 | def forbidden_reference_paths_from_split(frozen_split_manifest: dict[str, Any]) -> list[str]:
00431 |     forbidden: list[str] = []
00432 |     for row in frozen_split_manifest.get("rows") or []:
00433 |         for key, value in row.items():
00434 |             if not isinstance(value, str) or not value:
00435 |                 continue
00436 |             if Path(value).name in FORBIDDEN_REFERENCE_NAMES and (
00437 |                 "reference" in key.lower() or "hamiltonian" in key.lower()
00438 |             ):
00439 |                 forbidden.append(value)
00440 |     return sorted(set(forbidden))
00441 | 
00442 | 
00443 | def validate_no_forbidden_references(frozen_split_manifest: dict[str, Any]) -> None:
00444 |     forbidden = forbidden_reference_paths_from_split(frozen_split_manifest)
00445 |     if forbidden:
00446 |         raise RuntimeError("ML_prediction.HSX cannot be selected as SIESTA reference: " + ", ".join(forbidden))
00447 | 
00448 | 
00449 | def stage_graph2mat_metric_result(
00450 |     *,
00451 |     frozen_split_manifest: dict[str, Any],
00452 |     prediction_structs_dir: Path,
00453 |     output_dir: Path,
00454 |     dataset_root: Path | None = None,
00455 |     split: str = "test",
00456 | ) -> StagedGraph2MatMetrics:
00457 |     validate_no_forbidden_references(frozen_split_manifest)
00458 |     if output_dir.exists():
00459 |         shutil.rmtree(output_dir)
00460 |     sample_ids: list[str] = []
00461 |     for row in split_rows(frozen_split_manifest, split):
00462 |         sample_id = row_sample_id(row)
00463 |         sample_dir = Path(str(row.get("sample_dir") or ""))
00464 |         if not sample_dir.exists():
00465 |             raise RuntimeError(f"Frozen split sample_dir does not exist: {sample_dir}")
00466 |         prediction_dir = prediction_structs_dir / sample_id
00467 |         prediction = prediction_dir / "ML_prediction.HSX"
00468 |         if not prediction.exists():
00469 |             raise RuntimeError(f"Missing Graph2Mat prediction for common metrics: {prediction}")
00470 |         _link_or_copy(sample_dir / "RUN.fdf", output_dir / "structures" / sample_id / "RUN.fdf")
00471 |         for artifact in sorted(path for path in sample_dir.iterdir() if path.is_file()):
00472 |             if artifact.name in FORBIDDEN_REFERENCE_NAMES:
00473 |                 continue
00474 |             _link_or_copy(artifact, output_dir / "siesta_hamiltonians" / sample_id / artifact.name)
00475 |         _link_or_copy(prediction, output_dir / "predicted_hamiltonians" / sample_id / prediction.name)
00476 |         sample_ids.append(sample_id)
00477 | 
00478 |     if dataset_root is not None:
00479 |         split_root = dataset_root / "splits"
00480 |         if split_root.exists():
00481 |             _link_or_copy(split_root, output_dir / "splits")
00482 |         for basis_dir in (
00483 |             dataset_root / "basis",
00484 |             dataset_root / "material_basis",
00485 |             dataset_root / "MD_steps" / "basis",
00486 |             dataset_root / "materials" / "basis",
00487 |         ):
00488 |             if basis_dir.exists():
00489 |                 _link_or_copy(basis_dir, output_dir / "basis")
00490 |                 break
00491 |     return StagedGraph2MatMetrics(result_dir=output_dir, sample_ids=sample_ids)
00492 | 
00493 | 
00494 | def stage_deeph_metric_inputs(
00495 |     *,
00496 |     raw_mirror: dict[str, Any],
00497 |     processed_dir: Path,
00498 |     inference_dir: Path,
00499 |     output_dir: Path,
00500 |     split: str = "test",
00501 | ) -> StagedDeepHMetrics:
00502 |     if output_dir.exists():
00503 |         shutil.rmtree(output_dir)
00504 |     staged_processed = output_dir / "processed"
00505 |     staged_predictions = output_dir / "predictions"
00506 |     sample_ids: list[str] = []
00507 |     for row in raw_mirror.get("rows") or []:
00508 |         if row.get("split") != split:
00509 |             continue
00510 |         sample_id = str(row.get("sample_id") or "").strip()
00511 |         if not sample_id:
00512 |             raise RuntimeError(f"DeepH raw mirror row is missing sample_id: {row}")
00513 |         raw_name = Path(str(row.get("raw_dir") or "")).name
00514 |         source_processed = processed_dir / raw_name
00515 |         source_prediction = inference_dir / raw_name
00516 |         if not source_processed.exists():
00517 |             raise RuntimeError(f"Missing DeepH processed sample for common metrics: {source_processed}")
00518 |         if not source_prediction.exists():
00519 |             raise RuntimeError(f"Missing DeepH prediction sample for common metrics: {source_prediction}")
00520 |         _link_or_copy(source_processed, staged_processed / sample_id)
00521 |         _link_or_copy(source_prediction, staged_predictions / sample_id)
00522 |         sample_ids.append(sample_id)
00523 |     return StagedDeepHMetrics(staged_processed, staged_predictions, sample_ids)
00524 | 
00525 | 
00526 | def weighted_sample_rows(metrics_root: Path) -> list[dict[str, str]]:
00527 |     return [row for row in read_csv_rows(metrics_root / "kpoint_matrix_metrics.csv") if row.get("row_type") == "weighted_sample"]
00528 | 
00529 | 
00530 | def sample_ids_from_metrics(metrics_root: Path) -> set[str]:
00531 |     ids = {str(row.get("sample") or "").strip() for row in weighted_sample_rows(metrics_root)}
00532 |     if ids:
00533 |         return {sample for sample in ids if sample}
00534 |     return {str(row.get("sample") or "").strip() for row in read_csv_rows(metrics_root / "sample_status.csv") if row.get("sample")}
00535 | 
00536 | 
00537 | def method_has_diagnostic_only(metrics_root: Path, manifest: dict[str, Any]) -> bool:
00538 |     adapter_manifest_path = metrics_root.parent / "adapter_manifest.json"
00539 |     adapter_manifest = read_json(adapter_manifest_path)
00540 |     if int(adapter_manifest.get("diagnostic_only_count") or 0) > 0:
00541 |         return True
00542 |     for name in ("kpoint_matrix_metrics.csv", "kpoint_spectral_metrics.csv", "kpoint_dos_metrics.csv"):
00543 |         for row in read_csv_rows(metrics_root / name):
00544 |             value = str(row.get("deeph_diagnostic_only") or "").strip().lower()
00545 |             if value in {"true", "1", "yes"}:
00546 |                 return True
00547 |     return bool(manifest.get("diagnostic_only"))
00548 | 
00549 | 
00550 | def adapter_equivalence_summary(metrics_root: Path) -> dict[str, Any]:
00551 |     adapter_manifest_path = metrics_root.parent / "adapter_manifest.json"
00552 |     adapter_manifest = read_json(adapter_manifest_path)
00553 |     if not adapter_manifest:
00554 |         return {
00555 |             "adapter_manifest_path": str(adapter_manifest_path),
00556 |             "adapter_equivalence_status": EQUIVALENCE_INVALID_MISSING_REFERENCE,
00557 |             "adapter_equivalence_statuses": [EQUIVALENCE_INVALID_MISSING_REFERENCE],
00558 |             "equivalence_status": EQUIVALENCE_STATUS_FAILED,
00559 |             "equivalence_statuses": [EQUIVALENCE_STATUS_FAILED],
00560 |             "equivalence_scope": "unknown",
00561 |             "equivalence_scopes": ["unknown"],
00562 |             "equivalence_evidence_paths": [],
00563 |             "equivalence_gate": {
00564 |                 "robust_claim_allowed": False,
00565 |                 "diagnostic_only": True,
00566 |                 "diagnostic_only_reason": "DeepH adapter manifest is missing.",
00567 |             },
00568 |             "raw_global_equivalence_proven": False,
00569 |             "robust_matrix_metrics_allowed": False,
00570 |         }
00571 |     statuses = [
00572 |         str(status)
00573 |         for status in adapter_manifest.get("adapter_equivalence_statuses") or []
00574 |         if str(status).strip()
00575 |     ]
00576 |     if not statuses:
00577 |         statuses = sorted(
00578 |             {
00579 |                 str(sample.get("adapter_equivalence_status"))
00580 |                 for sample in adapter_manifest.get("samples") or []
00581 |                 if str(sample.get("adapter_equivalence_status") or "").strip()
00582 |             }
00583 |         )
00584 |     if not statuses:
00585 |         statuses = [EQUIVALENCE_INVALID_MISSING_REFERENCE]
00586 |     equivalence_statuses = [
00587 |         str(status)
00588 |         for status in adapter_manifest.get("equivalence_statuses") or []
00589 |         if str(status).strip()
00590 |     ]
00591 |     if not equivalence_statuses:
00592 |         equivalence_statuses = sorted(
00593 |             {
00594 |                 str(sample.get("equivalence_status"))
00595 |                 for sample in adapter_manifest.get("samples") or []
00596 |                 if str(sample.get("equivalence_status") or "").strip()
00597 |             }
00598 |         )
00599 |     if not equivalence_statuses:
00600 |         equivalence_statuses = sorted({equivalence_status_from_adapter_status(status) for status in statuses})
00601 |     equivalence_scopes = [
00602 |         str(scope)
00603 |         for scope in adapter_manifest.get("equivalence_scopes") or []
00604 |         if str(scope).strip()
00605 |     ]
00606 |     if not equivalence_scopes:
00607 |         equivalence_scopes = sorted(
00608 |             {
00609 |                 str(sample.get("equivalence_scope"))
00610 |                 for sample in adapter_manifest.get("samples") or []
00611 |                 if str(sample.get("equivalence_scope") or "").strip()
00612 |             }
00613 |         )
00614 |     if not equivalence_scopes:
00615 |         equivalence_scopes = sorted({equivalence_scope_from_adapter_status(status) for status in statuses})
00616 |     raw_global_equivalence_proven = (
00617 |         bool(adapter_manifest.get("robust_matrix_metrics_allowed"))
00618 |         and all(status in PROVEN_ADAPTER_EQUIVALENCE_STATUSES for status in statuses)
00619 |         and all(status == EQUIVALENCE_STATUS_PROVEN for status in equivalence_statuses)
00620 |     )
00621 |     primary_status = (
00622 |         EQUIVALENCE_PROVEN_RAW_GLOBAL
00623 |         if raw_global_equivalence_proven
00624 |         else next((status for status in statuses if status not in PROVEN_ADAPTER_EQUIVALENCE_STATUSES), statuses[0])
00625 |     )
00626 |     primary_equivalence_status = (
00627 |         EQUIVALENCE_STATUS_PROVEN
00628 |         if raw_global_equivalence_proven
00629 |         else next((status for status in equivalence_statuses if status != EQUIVALENCE_STATUS_PROVEN), equivalence_statuses[0])
00630 |     )
00631 |     primary_equivalence_scope = equivalence_scopes[0] if len(equivalence_scopes) == 1 else ",".join(equivalence_scopes)
00632 |     equivalence_gate = adapter_manifest.get("equivalence_gate") if isinstance(adapter_manifest.get("equivalence_gate"), dict) else {}
00633 |     return {
00634 |         "adapter_manifest_path": str(adapter_manifest_path),
00635 |         "adapter_equivalence_status": primary_status,
00636 |         "adapter_equivalence_statuses": statuses,
00637 |         "equivalence_status": primary_equivalence_status,
00638 |         "equivalence_statuses": equivalence_statuses,
00639 |         "equivalence_scope": primary_equivalence_scope,
00640 |         "equivalence_scopes": equivalence_scopes,
00641 |         "equivalence_evidence_paths": adapter_manifest.get("equivalence_evidence_paths") or [],
00642 |         "equivalence_gate": equivalence_gate,
00643 |         "raw_global_equivalence_proven": raw_global_equivalence_proven,
00644 |         "robust_matrix_metrics_allowed": raw_global_equivalence_proven,
00645 |     }
00646 | 
00647 | 
00648 | def summarize_method(method: str, metrics_root: Path) -> dict[str, Any]:
00649 |     matrix_rows = weighted_sample_rows(metrics_root)
00650 |     sparse_rows = read_csv_rows(metrics_root / "sparse_metrics.csv")
00651 |     spectral_rows = read_csv_rows(metrics_root / "kpoint_spectral_metrics.csv") or read_csv_rows(metrics_root / "spectral_metrics.csv")
00652 |     dos_rows = read_csv_rows(metrics_root / "kpoint_dos_metrics.csv") or read_csv_rows(metrics_root / "dos_metrics.csv")
00653 |     manifest = read_json(metrics_root / "manifest.json")
00654 |     fatal_errors = manifest.get("fatal_errors") or []
00655 |     warnings = list(manifest.get("warnings") or [])
00656 |     if manifest and manifest.get("uses_reference_overlap_k") is not True:
00657 |         warnings.append({"severity": "severe", "kind": "missing_s_ref", "message": "S_ref(k) was not recorded as overlap source."})
00658 |     if manifest and manifest.get("kpoint_metrics_enabled") is False:
00659 |         warnings.append({"severity": "severe", "kind": "unsupported_kgrid", "message": "k-point metrics were not enabled."})
00660 |     status = "ok"
00661 |     if not matrix_rows and not sparse_rows:
00662 |         status = "missing_metrics"
00663 |     if fatal_errors:
00664 |         status = "fatal_errors"
00665 |     adapter_summary = adapter_equivalence_summary(metrics_root) if method == "deeph" else {}
00666 |     diagnostic_only = method_has_diagnostic_only(metrics_root, manifest)
00667 |     if method == "deeph" and not adapter_summary.get("raw_global_equivalence_proven"):
00668 |         diagnostic_only = True
00669 |         warnings.append(
00670 |             {
00671 |                 "severity": "severe",
00672 |                 "kind": "deeph_adapter_equivalence_not_proven",
00673 |                 "adapter_equivalence_status": adapter_summary.get("adapter_equivalence_status"),
00674 |                 "equivalence_status": adapter_summary.get("equivalence_status"),
00675 |                 "equivalence_scope": adapter_summary.get("equivalence_scope"),
00676 |                 "message": "DeepH prediction equivalence to Graph2Mat raw/global HSX is not proven.",
00677 |             }
00678 |         )
00679 |     row: dict[str, Any] = {
00680 |         "method": method,
00681 |         "metrics_root": str(metrics_root),
00682 |         "method_status": status,
00683 |         "diagnostic_only": diagnostic_only,
00684 |         "samples_compared": manifest.get("samples_compared"),
00685 |         "samples_failed": manifest.get("samples_failed"),
00686 |         "kpoint_metrics_enabled": manifest.get("kpoint_metrics_enabled"),
00687 |         "uses_reference_overlap_k": manifest.get("uses_reference_overlap_k"),
00688 |         "warning_count": len(warnings),
00689 |         "fatal_error_count": len(fatal_errors),
00690 |         **adapter_summary,
00691 |     }
00692 |     row["h_mae_eV_mean"] = mean([number(item.get("h_mae_eV")) for item in matrix_rows] or [number(item.get("mae_union_eV")) for item in sparse_rows])
00693 |     row["h_rmse_eV_mean"] = mean([number(item.get("h_rmse_eV")) for item in matrix_rows] or [number(item.get("rmse_union_eV")) for item in sparse_rows])
00694 |     row["h_mse_eV2_mean"] = mean([number(item.get("h_mse_eV2")) for item in matrix_rows] or [number(item.get("mse_union_eV2")) for item in sparse_rows])
00695 |     row["r2_mean"] = mean([number(item.get("r2_union")) for item in sparse_rows])
00696 |     row["relative_frobenius_mean"] = mean(
00697 |         [number(item.get("relative_frobenius")) for item in matrix_rows]
00698 |         or [number(item.get("relative_frobenius_union")) for item in sparse_rows]
00699 |     )
00700 |     row["support_precision_mean"] = mean([number(item.get("support_precision")) for item in sparse_rows])
00701 |     row["support_recall_mean"] = mean([number(item.get("support_recall")) for item in sparse_rows])
00702 |     row["support_f1_mean"] = mean([number(item.get("support_f1")) for item in sparse_rows])
00703 |     row["hermiticity_pred_mean"] = mean([number(item.get("hermiticity_pred")) for item in matrix_rows])
00704 |     row["global_rmse_eV_mean"] = mean([number(item.get("global_rmse_eV")) for item in spectral_rows])
00705 |     row["low_energy_rmse_eV_mean"] = mean([number(item.get("low_energy_rmse_eV")) for item in spectral_rows])
00706 |     row["fermi_window_rmse_eV_mean"] = mean([number(item.get("fermi_window_rmse_eV")) for item in spectral_rows])
00707 |     row["frontier_window_rmse_eV_mean"] = mean([number(item.get("frontier_window_rmse_eV")) for item in spectral_rows])
00708 |     row["dos_mae_500_fermi_window_mean"] = mean([number(item.get("dos_mae_500_fermi_window")) for item in dos_rows])
00709 |     row["dos_wasserstein_eV_mean"] = mean([number(item.get("dos_wasserstein_eV")) for item in dos_rows])
00710 |     return row
00711 | 
00712 | 
00713 | def summarize_derivative_method(method: str, derivative_root: Path | None) -> dict[str, Any]:
00714 |     if derivative_root is None:
00715 |         return {"method": method, "derivative_metrics_available": False}
00716 |     derivative_root = Path(derivative_root)
00717 |     if derivative_root.name != "derivative_metrics":
00718 |         derivative_root = derivative_root / "derivative_metrics"
00719 |     manifest = read_json(derivative_root / "manifest.json")
00720 |     rows = read_csv_rows(derivative_root / "derivative_matrix_metrics.csv")
00721 |     hermiticity_rows = read_csv_rows(derivative_root / "derivative_hermiticity.csv")
00722 |     available = bool(manifest or rows)
00723 |     summary: dict[str, Any] = {
00724 |         "method": method,
00725 |         "derivative_metrics_available": available,
00726 |         "derivative_metrics_root": str(derivative_root) if available else "",
00727 |         "derivative_scientific_status": manifest.get("scientific_status") if manifest else "",
00728 |         "derivative_force_constants_used": manifest.get("force_constants_used") if manifest else False,
00729 |         "derivative_paper_level": manifest.get("paper_level") if manifest else False,
00730 |         "derivative_stencils_total": manifest.get("stencils_total") if manifest else 0,
00731 |         "derivative_stencils_ok": manifest.get("stencils_ok") if manifest else 0,
00732 |         "derivative_stencils_failed": manifest.get("stencils_failed") if manifest else 0,
00733 |         "derivative_warning_count": len(manifest.get("warnings") or []) if manifest else 0,
00734 |         "derivative_fatal_error_count": len(manifest.get("fatal_errors") or []) if manifest else 0,
00735 |     }
00736 |     if rows:
00737 |         summary.update(
00738 |             {
00739 |                 "dh_mae_union_eV_per_Ang_mean": mean([number(row.get("dh_mae_union_eV_per_Ang")) for row in rows]),
00740 |                 "dh_rmse_union_eV_per_Ang_mean": mean([number(row.get("dh_rmse_union_eV_per_Ang")) for row in rows]),
00741 |                 "dh_support_f1_mean": mean([number(row.get("dh_support_f1")) for row in rows]),
00742 |                 "dh_relative_frobenius_ref_mean": mean([number(row.get("dh_relative_frobenius_ref")) for row in rows]),
00743 |                 "dh_hermiticity_ref_mean": mean([number(row.get("dH_ref_hermiticity_defect")) for row in hermiticity_rows]),
00744 |                 "dh_hermiticity_pred_mean": mean([number(row.get("dH_pred_hermiticity_defect")) for row in hermiticity_rows]),
00745 |             }
00746 |         )
00747 |     return summary
00748 | 
00749 | 
00750 | def sample_metric_rows(method: str, metrics_root: Path) -> list[dict[str, Any]]:
00751 |     rows = []
00752 |     for row in weighted_sample_rows(metrics_root):
00753 |         rows.append({"method": method, **row})
00754 |     return rows
00755 | 
00756 | 
00757 | def dataset_status(dataset_manifest: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
00758 |     warnings = list(dataset_manifest.get("warnings") or [])
00759 |     if dataset_manifest and not dataset_manifest.get("benchmark_ready", dataset_manifest.get("valid", False)):
00760 |         return "invalid_missing_artifacts", warnings
00761 |     mode = str(dataset_manifest.get("generation_mode") or dataset_manifest.get("mode") or "").strip()
00762 |     if mode == "clean_one_pass":
00763 |         return "valid_joint_one_pass_dataset", warnings
00764 |     if mode == "reused_validated":
00765 |         return "valid_reused_joint_dataset", warnings
00766 |     if mode == "repaired_explicit":
00767 |         warnings.append({"severity": "severe", "kind": "repaired_dataset", "message": "Dataset was explicitly repaired."})
00768 |         return "valid_repaired_dataset_with_warning", warnings
00769 |     return "valid_reused_joint_dataset" if dataset_manifest else "diagnostic_only", warnings
00770 | 
00771 | 
00772 | def build_recommendation(summary_rows: list[dict[str, Any]], status: str, warnings: list[dict[str, Any]]) -> dict[str, Any]:
00773 |     severe = [
00774 |         warning
00775 |         for warning in warnings
00776 |         if isinstance(warning, dict) and str(warning.get("severity") or "").lower() == "severe"
00777 |     ]
00778 |     if status != "valid_joint_one_pass_dataset" and status != "valid_reused_joint_dataset":
00779 |         return {
00780 |             "winner": None,
00781 |             "robust_recommendation": False,
00782 |             "status": status,
00783 |             "reason": "Comparison is not scientifically valid for winner selection.",
00784 |             "severe_warnings": severe,
00785 |         }
00786 |     if severe:
00787 |         return {
00788 |             "winner": None,
00789 |             "robust_recommendation": False,
00790 |             "status": "diagnostic_only",
00791 |             "reason": "Severe warnings prevent winner selection.",
00792 |             "severe_warnings": severe,
00793 |         }
00794 |     values = {row["method"]: number(row.get(PRIMARY_METRIC)) for row in summary_rows}
00795 |     if not all(math.isfinite(value) for value in values.values()) or len(values) < 2:
00796 |         return {
00797 |             "winner": None,
00798 |             "robust_recommendation": False,
00799 |             "status": "diagnostic_only",
00800 |             "reason": f"Primary metric {PRIMARY_METRIC} is unavailable for all methods.",
00801 |             "severe_warnings": severe,
00802 |         }
00803 |     winner = min(values, key=values.get)
00804 |     return {
00805 |         "winner": winner,
00806 |         "robust_recommendation": True,
00807 |         "status": "robust_candidate",
00808 |         "primary_metric": PRIMARY_METRIC,
00809 |         "primary_metric_values": values,
00810 |         "claim_scope": "common_metric_diagnostic_supporting",
00811 |         "supports_final_winner_claim": False,
00812 |         "final_claim_source": "Use final_statistics + gate_status + final_evaluation for paper-ready claims.",
00813 |         "reason": f"{winner} has the lower {PRIMARY_METRIC}; this is a supporting H-MAE recommendation, not a final spectral winner claim.",
00814 |         "severe_warnings": severe,
00815 |     }
00816 | 
00817 | 
00818 | def add_derivative_diagnostic_notes(recommendation: dict[str, Any], derivative_summary_rows: list[dict[str, Any]]) -> dict[str, Any]:
00819 |     notes = list(recommendation.get("diagnostic_notes") or [])
00820 |     for row in derivative_summary_rows:
00821 |         if not row.get("derivative_metrics_available"):
00822 |             continue
00823 |         notes.append(
00824 |             {
00825 |                 "kind": "hamiltonian_derivative_metrics",
00826 |                 "method": row.get("method"),
00827 |                 "scientific_status": row.get("derivative_scientific_status") or "diagnostic_only",
00828 |                 "force_constants_used": bool(row.get("derivative_force_constants_used")),
00829 |                 "paper_level": bool(row.get("derivative_paper_level")),
00830 |                 "winner_metric": False,
00831 |                 "message": "Hamiltonian derivative metrics are diagnostic notes only and are not used for winner selection.",
00832 |             }
00833 |         )
00834 |     if notes:
00835 |         recommendation = dict(recommendation)
00836 |         recommendation["diagnostic_notes"] = notes
00837 |     return recommendation
00838 | 
00839 | 
00840 | def _safe_recommendation_for_display(manifest: dict[str, Any]) -> dict[str, Any]:
00841 |     status = str(manifest.get("status") or "diagnostic_only")
00842 |     recommendation = dict(manifest.get("recommendation") or {})
00843 |     if status == "diagnostic_only" or status.startswith("invalid_"):
00844 |         recommendation["winner"] = None
00845 |         recommendation["robust_recommendation"] = False
00846 |         recommendation.setdefault("status", status)
00847 |         recommendation.setdefault("reason", "Comparison is not scientifically valid for winner selection.")
00848 |     return recommendation
00849 | 
00850 | 
00851 | def _plot_rows(summary_rows: list[dict[str, Any]], metric_group: dict[str, Any]) -> list[dict[str, Any]]:
00852 |     metric_keys = [metric["key"] for metric in metric_group.get("metrics") or []]
00853 |     rows = []
00854 |     for row in summary_rows:
00855 |         next_row: dict[str, Any] = {"method": row.get("method")}
00856 |         for key in metric_keys:
00857 |             next_row[key] = finite_or_none(row.get(key))
00858 |         rows.append(next_row)
00859 |     return rows
00860 | 
00861 | 
00862 | def _missing_metrics(rows: list[dict[str, Any]], metric_group: dict[str, Any]) -> list[dict[str, Any]]:
00863 |     missing: list[dict[str, Any]] = []
00864 |     for metric in metric_group.get("metrics") or []:
00865 |         key = metric["key"]
00866 |         for row in rows:
00867 |             if row.get(key) is None:
00868 |                 missing.append({"method": row.get("method"), "metric": key})
00869 |     return missing
00870 | 
00871 | 
00872 | def build_common_plot_payload(
00873 |     common_metrics_manifest: dict[str, Any] | None,
00874 |     *,
00875 |     artifact_summary: dict[str, Any] | None = None,
00876 |     timing_rows: list[dict[str, Any]] | None = None,
00877 |     timing_scaling_rows: list[dict[str, Any]] | None = None,
00878 |     metric_scaling_rows: list[dict[str, Any]] | None = None,
00879 |     status_payload: dict[str, Any] | None = None,
00880 | ) -> dict[str, Any]:
00881 |     """Build a UI-safe Graph2Mat-vs-DeepH comparison payload.
00882 | 
00883 |     The payload is intentionally conservative: diagnostic/invalid summaries never expose
00884 |     a winner even if the raw metric rows contain lower values for one method.
00885 |     """
00886 |     status_payload = dict(status_payload or {})
00887 |     timing_scaling_rows = timing_scaling_rows or []
00888 |     metric_scaling_rows = metric_scaling_rows or []
00889 |     metric_scaling_group_ids: set[str] = set()
00890 |     timing_scaling_plots = (
00891 |         [
00892 |             {
00893 |                 "id": "timing_scaling",
00894 |                 "kind": "timing_scaling",
00895 |                 "title": "Phase time vs dataset size",
00896 |                 "x_title": "Dataset size (snapshots)",
00897 |                 "y_title": "Seconds",
00898 |                 "rows": timing_scaling_rows,
00899 |             }
00900 |         ]
00901 |         if timing_scaling_rows
00902 |         else []
00903 |     )
00904 |     metric_scaling_plots: list[dict[str, Any]] = []
00905 |     if metric_scaling_rows:
00906 |         for metric_group in COMMON_METRIC_GROUPS:
00907 |             metric_keys = {metric["key"] for metric in metric_group.get("metrics") or []}
00908 |             rows = [row for row in metric_scaling_rows if row.get("metric_key") in metric_keys]
00909 |             if not rows:
00910 |                 continue
00911 |             metric_scaling_group_ids.add(str(metric_group["id"]))
00912 |             metric_scaling_plots.append(
00913 |                 {
00914 |                     "id": f"metric_scaling_{metric_group['id']}",
00915 |                     "kind": "metric_scaling",
00916 |                     "title": f"{metric_group['title']} vs dataset size",
00917 |                     "x_title": "Dataset size (snapshots)",
00918 |                     "y_title": metric_group.get("y_title") or "Metric value",
00919 |                     "metrics": metric_group.get("metrics") or [],
00920 |                     "rows": rows,
00921 |                 }
00922 |             )
00923 |     if not common_metrics_manifest:
00924 |         return {
00925 |             "available": bool(timing_scaling_plots or metric_scaling_plots),
00926 |             "plots": [*metric_scaling_plots, *timing_scaling_plots],
00927 |             "metric_groups": COMMON_METRIC_GROUPS,
00928 |             "derivative_metric_groups": DERIVATIVE_METRIC_GROUPS,
00929 |             "artifact_summary": artifact_summary or {},
00930 |             "timing_rows": timing_rows or [],
00931 |             "timing_scaling_rows": timing_scaling_rows,
00932 |             "metric_scaling_rows": metric_scaling_rows,
00933 |             "message": "No common Graph2Mat/DeepH metrics are available yet.",
00934 |             "status": status_payload,
00935 |         }
00936 | 
00937 |     manifest = dict(common_metrics_manifest)
00938 |     summary_rows = [dict(row) for row in manifest.get("summary_rows") or []]
00939 |     derivative_summary_rows = [dict(row) for row in manifest.get("derivative_summary_rows") or []]
00940 |     rows_for_plots = [dict(row) for row in summary_rows]
00941 |     if derivative_summary_rows:
00942 |         by_method = {str(row.get("method") or ""): row for row in rows_for_plots}
00943 |         for derivative_row in derivative_summary_rows:
00944 |             method = str(derivative_row.get("method") or "")
00945 |             if method not in by_method:
00946 |                 by_method[method] = {"method": method}
00947 |                 rows_for_plots.append(by_method[method])
00948 |             by_method[method].update(derivative_row)
00949 |     scientific_status = str(manifest.get("status") or "diagnostic_only")
00950 |     recommendation = _safe_recommendation_for_display(manifest)
00951 |     plots: list[dict[str, Any]] = []
00952 |     for metric_group in COMMON_METRIC_GROUPS:
00953 |         if str(metric_group["id"]) in metric_scaling_group_ids:
00954 |             continue
00955 |         rows = _plot_rows(summary_rows, metric_group)
00956 |         plots.append(
00957 |             {
00958 |                 "id": metric_group["id"],
00959 |                 "kind": "grouped_bar",
00960 |                 "title": metric_group["title"],
00961 |                 "y_title": metric_group["y_title"],
00962 |                 "metrics": metric_group["metrics"],
00963 |                 "rows": rows,
00964 |                 "missing_metrics": _missing_metrics(rows, metric_group),
00965 |             }
00966 |         )
00967 |     for metric_group in DERIVATIVE_METRIC_GROUPS:
00968 |         rows = _plot_rows(rows_for_plots, metric_group)
00969 |         if not any(row.get(metric["key"]) is not None for row in rows for metric in metric_group.get("metrics") or []):
00970 |             continue
00971 |         plots.append(
00972 |             {
00973 |                 "id": metric_group["id"],
00974 |                 "kind": "grouped_bar",
00975 |                 "title": metric_group["title"],
00976 |                 "y_title": metric_group["y_title"],
00977 |                 "metrics": metric_group["metrics"],
00978 |                 "rows": rows,
00979 |                 "missing_metrics": _missing_metrics(rows, metric_group),
00980 |                 "diagnostic_only": True,
00981 |             }
00982 |         )
00983 |     plots.extend(metric_scaling_plots)
00984 |     plots.extend(timing_scaling_plots)
00985 |     return {
00986 |         "available": True,
00987 |         "schema": "graph2mat_deeph_plot_payload_v1",
00988 |         "scientific_status": scientific_status,
00989 |         "diagnostic_only": scientific_status == "diagnostic_only",
00990 |         "invalid": scientific_status.startswith("invalid_"),
00991 |         "common_metrics": {
00992 |             **manifest,
00993 |             "recommendation": recommendation,
00994 |         },
00995 |         "summary_rows": summary_rows,
00996 |         "metric_groups": COMMON_METRIC_GROUPS,
00997 |         "derivative_summary_rows": derivative_summary_rows,
00998 |         "derivative_metric_groups": DERIVATIVE_METRIC_GROUPS,
00999 |         "artifact_summary": artifact_summary or {},
01000 |         "timing_rows": timing_rows or [],
01001 |         "timing_scaling_rows": timing_scaling_rows,
01002 |         "metric_scaling_rows": metric_scaling_rows,
01003 |         "plots": plots,
01004 |         "warnings": list(manifest.get("warnings") or []),
01005 |         "recommendation": recommendation,
01006 |         "message": recommendation.get("reason") or "",
01007 |         "status": status_payload,
01008 |     }
01009 | 
01010 | 
01011 | def aggregate_common_metrics(
01012 |     *,
01013 |     graph2mat_metrics_root: Path | None,
01014 |     deeph_metrics_root: Path | None,
01015 |     output_dir: Path,
01016 |     frozen_split_manifest_path: Path | None = None,
01017 |     dataset_manifest_path: Path | None = None,
01018 |     graph2mat_derivative_root: Path | None = None,
01019 |     deeph_derivative_root: Path | None = None,
01020 | ) -> dict[str, Any]:
01021 |     output_dir.mkdir(parents=True, exist_ok=True)
01022 |     frozen_split = read_json(frozen_split_manifest_path) if frozen_split_manifest_path else {}
01023 |     dataset_manifest = read_json(dataset_manifest_path) if dataset_manifest_path else {}
01024 |     warnings: list[dict[str, Any]] = []
01025 |     if frozen_split:
01026 |         validate_no_forbidden_references(frozen_split)
01027 |     status, dataset_warnings = dataset_status(dataset_manifest)
01028 |     warnings.extend(dataset_warnings)
01029 | 
01030 |     g2m_ids = sample_ids_from_metrics(graph2mat_metrics_root) if graph2mat_metrics_root else set()
01031 |     deeph_ids = sample_ids_from_metrics(deeph_metrics_root) if deeph_metrics_root else set()
01032 |     if graph2mat_metrics_root and deeph_metrics_root and g2m_ids != deeph_ids:
01033 |         status = "invalid_incompatible_splits"
01034 |         warnings.append(
01035 |             {
01036 |                 "severity": "severe",
01037 |                 "kind": "mismatched_sample_ids",
01038 |                 "graph2mat_only": sorted(g2m_ids - deeph_ids),
01039 |                 "deeph_only": sorted(deeph_ids - g2m_ids),
01040 |             }
01041 |         )
01042 | 
01043 |     summary_rows = [
01044 |         summarize_method(method, root)
01045 |         for method, root in (("graph2mat", graph2mat_metrics_root), ("deeph", deeph_metrics_root))
01046 |         if root is not None
01047 |     ]
01048 |     derivative_summary_rows = [
01049 |         row
01050 |         for row in (
01051 |             summarize_derivative_method("graph2mat", graph2mat_derivative_root),
01052 |             summarize_derivative_method("deeph", deeph_derivative_root),
01053 |         )
01054 |         if row.get("derivative_metrics_available")
01055 |     ]
01056 |     for row in summary_rows:
01057 |         if row["method_status"] != "ok":
01058 |             status = "invalid_prediction_format"
01059 |             warnings.append({"severity": "severe", "kind": f"{row['method']}_metrics_status", "status": row["method_status"]})
01060 |         if row["diagnostic_only"]:
01061 |             if not status.startswith("invalid_"):
01062 |                 status = "diagnostic_only"
01063 |             warnings.append({"severity": "severe", "kind": f"{row['method']}_diagnostic_only"})
01064 |         if row["method"] == "deeph" and not row.get("raw_global_equivalence_proven"):
01065 |             if not status.startswith("invalid_"):
01066 |                 status = "diagnostic_only"
01067 |             warnings.append(
01068 |                 {
01069 |                     "severity": "severe",
01070 |                     "kind": "deeph_adapter_equivalence_not_proven",
01071 |                     "adapter_equivalence_status": row.get("adapter_equivalence_status"),
01072 |                     "message": "DeepH adapter did not prove raw/global HSX equivalence.",
01073 |                 }
01074 |             )
01075 |         if row.get("uses_reference_overlap_k") is not True:
01076 |             if not status.startswith("invalid_"):
01077 |                 status = "diagnostic_only"
01078 |             warnings.append({"severity": "severe", "kind": f"{row['method']}_missing_s_ref"})
01079 |         if row.get("kpoint_metrics_enabled") is False:
01080 |             if not status.startswith("invalid_"):
01081 |                 status = "diagnostic_only"
01082 |             warnings.append({"severity": "severe", "kind": f"{row['method']}_unsupported_kgrid"})
01083 | 
01084 |     sample_rows = [
01085 |         row
01086 |         for method, root in (("graph2mat", graph2mat_metrics_root), ("deeph", deeph_metrics_root))
01087 |         if root is not None
01088 |         for row in sample_metric_rows(method, root)
01089 |     ]
01090 |     recommendation = build_recommendation(summary_rows, status, warnings)
01091 |     recommendation = add_derivative_diagnostic_notes(recommendation, derivative_summary_rows)
01092 |     write_csv_rows(output_dir / "common_method_metrics.csv", summary_rows)
01093 |     if derivative_summary_rows:
01094 |         write_csv_rows(output_dir / "common_derivative_method_metrics.csv", derivative_summary_rows)
01095 |     write_csv_rows(output_dir / "common_sample_metrics.csv", sample_rows)
01096 |     write_json(output_dir / "recommendation.json", recommendation)
01097 |     manifest = {
01098 |         "schema": SCHEMA,
01099 |         "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
01100 |         "status": status,
01101 |         "status_values": sorted(STATUS_VALUES),
01102 |         "graph2mat_metrics_root": str(graph2mat_metrics_root) if graph2mat_metrics_root else "",
01103 |         "deeph_metrics_root": str(deeph_metrics_root) if deeph_metrics_root else "",
01104 |         "output_dir": str(output_dir),
01105 |         "sample_ids": sorted(g2m_ids | deeph_ids),
01106 |         "warnings": warnings,
01107 |         "summary_rows": summary_rows,
01108 |         "cross_structure_metadata": cross_structure_metadata(dataset_manifest_path),
01109 |         "derivative_summary_rows": derivative_summary_rows,
01110 |         "derivative_metrics": {
01111 |             "available": bool(derivative_summary_rows),
01112 |             "winner_metric": False,
01113 |             "paper_level": False,
01114 |             "summary_rows": derivative_summary_rows,
01115 |         },
01116 |         "recommendation": recommendation,
01117 |     }
01118 |     write_json(output_dir / "common_summary.json", manifest)
01119 |     write_json(output_dir / "benchmark_manifest.json", manifest)
01120 |     return manifest
01121 | 
01122 | 
01123 | def parse_args() -> argparse.Namespace:
01124 |     parser = argparse.ArgumentParser(description=__doc__)
01125 |     parser.add_argument("--graph2mat-metrics-root", type=Path, required=True)
01126 |     parser.add_argument("--deeph-metrics-root", type=Path, required=True)
01127 |     parser.add_argument("--output-dir", type=Path, required=True)
01128 |     parser.add_argument("--frozen-split-manifest", type=Path, default=None)
01129 |     parser.add_argument("--dataset-manifest", type=Path, default=None)
01130 |     parser.add_argument("--graph2mat-derivative-root", type=Path, default=None)
01131 |     parser.add_argument("--deeph-derivative-root", type=Path, default=None)
01132 |     return parser.parse_args()
01133 | 
01134 | 
01135 | def main() -> None:
01136 |     args = parse_args()
01137 |     manifest = aggregate_common_metrics(
01138 |         graph2mat_metrics_root=args.graph2mat_metrics_root,
01139 |         deeph_metrics_root=args.deeph_metrics_root,
01140 |         output_dir=args.output_dir,
01141 |         frozen_split_manifest_path=args.frozen_split_manifest,
01142 |         dataset_manifest_path=args.dataset_manifest,
01143 |         graph2mat_derivative_root=args.graph2mat_derivative_root,
01144 |         deeph_derivative_root=args.deeph_derivative_root,
01145 |     )
01146 |     print(json.dumps(json_safe({"status": manifest["status"], "output_dir": manifest["output_dir"]}), ensure_ascii=False))
01147 | 
01148 | 
01149 | if __name__ == "__main__":
01150 |     main()
```

## `tests/test_graphene_band_comparison.py`

SHA-256: `5859d433ddcc29f1e39a065980276321a35c5b11c09786a394c804edb949c698`

```py
00001 | import csv
00002 | import importlib
00003 | import json
00004 | import subprocess
00005 | import sys
00006 | import tempfile
00007 | import unittest
00008 | from pathlib import Path
00009 | 
00010 | 
00011 | REPO_ROOT = Path(__file__).resolve().parents[1]
00012 | SCRIPTS_DIR = REPO_ROOT / "Comparison" / "scripts"
00013 | if str(SCRIPTS_DIR) not in sys.path:
00014 |     sys.path.insert(0, str(SCRIPTS_DIR))
00015 | 
00016 | bands = importlib.import_module("compare_graphene_bands_siesta_g2m_deeph")
00017 | 
00018 | 
00019 | def write_band_csv(path: Path, *, offset: float = 0.0) -> None:
00020 |     path.parent.mkdir(parents=True, exist_ok=True)
00021 |     rows = []
00022 |     for k_index in range(4):
00023 |         rows.append(
00024 |             {
00025 |                 "k_index": k_index,
00026 |                 "k_distance": float(k_index),
00027 |                 "band_index": 0,
00028 |                 "energy_eV": -1.0 + 0.1 * k_index + offset,
00029 |                 "fermi_level_eV": 0.0,
00030 |             }
00031 |         )
00032 |         rows.append(
00033 |             {
00034 |                 "k_index": k_index,
00035 |                 "k_distance": float(k_index),
00036 |                 "band_index": 1,
00037 |                 "energy_eV": 1.0 + 0.1 * k_index + offset,
00038 |                 "fermi_level_eV": 0.0,
00039 |             }
00040 |         )
00041 |     with path.open("w", encoding="utf-8", newline="") as handle:
00042 |         writer = csv.DictWriter(handle, fieldnames=["k_index", "k_distance", "band_index", "energy_eV", "fermi_level_eV"])
00043 |         writer.writeheader()
00044 |         writer.writerows(rows)
00045 | 
00046 | 
00047 | class GrapheneBandComparisonTests(unittest.TestCase):
00048 |     def test_graphene_gkm_path_labels(self) -> None:
00049 |         _name, nodes = bands.load_kpath(type("Args", (), {"kpath": "graphene_gkm", "kpath_json": None})())
00050 | 
00051 |         self.assertEqual([bands.label_for_display(node.label) for node in nodes], ["Γ", "K", "M", "Γ"])
00052 | 
00053 |     def test_graphene_gkm_interpolation_no_duplicate_internal_nodes(self) -> None:
00054 |         _name, nodes = bands.load_kpath(type("Args", (), {"kpath": "graphene_gkm", "kpath_json": None})())
00055 |         records = bands.interpolate_kpath(nodes, points_per_segment=2)
00056 | 
00057 |         labels = [record.k_label for record in records if record.k_label]
00058 |         self.assertEqual(labels, ["Γ", "K", "M", "Γ"])
00059 |         self.assertEqual(labels.count("K"), 1)
00060 |         self.assertEqual(labels.count("M"), 1)
00061 |         self.assertEqual(len(records), 7)
00062 | 
00063 |     def test_bandlines_block_generation(self) -> None:
00064 |         _name, nodes = bands.load_kpath(type("Args", (), {"kpath": "graphene_gkm", "kpath_json": None})())
00065 |         block = bands.bandlines_block(nodes, points_per_segment=80)
00066 | 
00067 |         self.assertIn("BandLinesScale ReciprocalLatticeVectors", block)
00068 |         self.assertIn("%block BandLines", block)
00069 |         self.assertIn("0.3333333333    0.3333333333    0.0000000000    K", block)
00070 |         self.assertIn("0.5000000000    0.0000000000    0.0000000000    M", block)
00071 |         self.assertIn("%endblock BandLines", block)
00072 | 
00073 |     def test_parse_fdf_bandlines_and_temperature(self) -> None:
00074 |         with tempfile.TemporaryDirectory() as tmp:
00075 |             fdf = Path(tmp) / "RUN.fdf"
00076 |             fdf.write_text(
00077 |                 "\n".join(
00078 |                     [
00079 |                         "MD.InitialTemperature 450 K",
00080 |                         "BandLinesScale ReciprocalLatticeVectors",
00081 |                         "%block BandLines",
00082 |                         "1 0.0 0.0 0.0 \\Gamma",
00083 |                         "50 0.33333 0.666667 0.0 K",
00084 |                         "50 0.5 0.5 0.0 M",
00085 |                         "50 0.0 0.0 0.0 \\Gamma",
00086 |                         "%endblock BandLines",
00087 |                     ]
00088 |                 ),
00089 |                 encoding="utf-8",
00090 |             )
00091 | 
00092 |             _name, nodes = bands.parse_fdf_bandlines(fdf)
00093 |             temperature = bands.parse_md_initial_temperature(fdf)
00094 | 
00095 |             self.assertEqual([bands.label_for_display(node.label) for node in nodes], ["Γ", "K", "M", "Γ"])
00096 |             self.assertEqual(nodes[1].k, (0.33333, 0.666667, 0.0))
00097 |             self.assertEqual(temperature["value"], 450.0)
00098 |             self.assertEqual(temperature["unit"], "K")
00099 | 
00100 |     def test_reject_ml_prediction_as_reference(self) -> None:
00101 |         with self.assertRaisesRegex(RuntimeError, "ML_prediction.HSX"):
00102 |             bands.resolve_reference_path(Path("/tmp/ML_prediction.HSX"))
00103 | 
00104 |     def test_energy_alignment_fermi(self) -> None:
00105 |         self.assertAlmostEqual(bands.align_energy(4.25, 1.5, "fermi", None), 2.75)
00106 | 
00107 |     def test_band_error_metrics(self) -> None:
00108 |         rows = [
00109 |             {"error_eV": 1.0},
00110 |             {"error_eV": -2.0},
00111 |             {"error_eV": 2.0},
00112 |         ]
00113 | 
00114 |         metrics = bands.metric_summary(rows)
00115 | 
00116 |         self.assertAlmostEqual(metrics["band_mae_eV"], 5.0 / 3.0)
00117 |         self.assertAlmostEqual(metrics["band_rmse_eV"], 3.0**0.5)
00118 |         self.assertAlmostEqual(metrics["max_abs_error_eV"], 2.0)
00119 | 
00120 |     def test_dirac_diagnostic_flags_shift_and_gap(self) -> None:
00121 |         method = bands.MethodBands(
00122 |             method="SIESTA",
00123 |             sample_id="s0",
00124 |             bands=[[-2.0, -1.0, 1.4, 1.6]],
00125 |             fermi_level_eV=0.0,
00126 |             energy_zero_policy="fermi",
00127 |             raw_source="synthetic",
00128 |             hermiticity_defects=[],
00129 |             overlap_used=True,
00130 |             diagonalization_errors=[],
00131 |         )
00132 | 
00133 |         diagnostic = bands.dirac_diagnostic_for_method(
00134 |             method,
00135 |             k_index=0,
00136 |             occupied_bands=2,
00137 |             fermi_level_eV=0.0,
00138 |             gap_warning_meV=10.0,
00139 |             fermi_warning_meV=50.0,
00140 |         )
00141 | 
00142 |         self.assertAlmostEqual(diagnostic["gap_eV"], 2.4)
00143 |         self.assertAlmostEqual(diagnostic["dirac_minus_fermi_eV"], 0.2)
00144 |         self.assertTrue(diagnostic["warnings"])
00145 | 
00146 |     def test_prediction_dirac_diagnostic_uses_already_aligned_energy(self) -> None:
00147 |         method = bands.MethodBands(
00148 |             method="Graph2Mat",
00149 |             sample_id="s0",
00150 |             bands=[[-0.25, 0.35]],
00151 |             fermi_level_eV=-5.7,
00152 |             energy_zero_policy="fermi",
00153 |             raw_source="synthetic",
00154 |             hermiticity_defects=[],
00155 |             overlap_used=True,
00156 |             diagonalization_errors=[],
00157 |         )
00158 | 
00159 |         diagnostic = bands.dirac_diagnostic_for_method(
00160 |             method,
00161 |             k_index=0,
00162 |             occupied_bands=1,
00163 |             fermi_level_eV=-5.7,
00164 |             gap_warning_meV=1000.0,
00165 |             fermi_warning_meV=1000.0,
00166 |         )
00167 | 
00168 |         self.assertAlmostEqual(diagnostic["dirac_minus_fermi_eV"], 0.05)
00169 |         self.assertEqual(diagnostic["dirac_fermi_convention"], "prediction_already_fermi_aligned")
00170 | 
00171 |     def test_prediction_band_errors_do_not_subtract_reference_fermi_twice(self) -> None:
00172 |         siesta = bands.MethodBands(
00173 |             method="SIESTA",
00174 |             sample_id="s0",
00175 |             bands=[[-6.0, -5.4]],
00176 |             fermi_level_eV=-5.7,
00177 |             energy_zero_policy="fermi",
00178 |             raw_source="synthetic",
00179 |             hermiticity_defects=[],
00180 |             overlap_used=True,
00181 |             diagonalization_errors=[],
00182 |         )
00183 |         g2m = bands.MethodBands(
00184 |             method="Graph2Mat",
00185 |             sample_id="s0",
00186 |             bands=[[-0.25, 0.35]],
00187 |             fermi_level_eV=-5.7,
00188 |             energy_zero_policy="fermi",
00189 |             raw_source="synthetic",
00190 |             hermiticity_defects=[],
00191 |             overlap_used=True,
00192 |             diagonalization_errors=[],
00193 |         )
00194 |         kpoints = [bands.KPointRecord(0, 0.0, 0.0, 0.0, 0.0, "K", "Γ-K")]
00195 | 
00196 |         rows = bands.error_rows(g2m, siesta, kpoints, "fermi", -5.7)
00197 | 
00198 |         self.assertAlmostEqual(rows[0]["siesta_energy_eV"], -0.3)
00199 |         self.assertAlmostEqual(rows[0]["predicted_energy_eV"], -0.25)
00200 |         self.assertAlmostEqual(rows[0]["error_eV"], 0.05)
00201 | 
00202 |     def test_missing_overlap_fail_closed(self) -> None:
00203 |         import numpy as np
00204 | 
00205 |         class FakeHamiltonian:
00206 |             orthogonal = False
00207 | 
00208 |             def Hk(self, k, format="array"):
00209 |                 return np.eye(2)
00210 | 
00211 |         _name, nodes = bands.load_kpath(type("Args", (), {"kpath": "graphene_gkm", "kpath_json": None})())
00212 |         records = bands.interpolate_kpath(nodes, points_per_segment=1)
00213 | 
00214 |         with self.assertRaisesRegex(RuntimeError, "requires S\\(k\\)"):
00215 |             bands.matrix_bands_from_sisl(
00216 |                 method="Graph2Mat",
00217 |                 sample_id="s0",
00218 |                 hamiltonian_obj=FakeHamiltonian(),
00219 |                 reference_obj=FakeHamiltonian(),
00220 |                 kpoints=records,
00221 |                 fermi_level=0.0,
00222 |                 fail_closed=True,
00223 |             )
00224 | 
00225 |     def test_missing_overlap_no_fail_closed_marks_diagnostic(self) -> None:
00226 |         import numpy as np
00227 | 
00228 |         class FakeHamiltonian:
00229 |             orthogonal = False
00230 | 
00231 |             def Hk(self, k, format="array"):
00232 |                 return np.eye(2)
00233 | 
00234 |         _name, nodes = bands.load_kpath(type("Args", (), {"kpath": "graphene_gkm", "kpath_json": None})())
00235 |         records = bands.interpolate_kpath(nodes, points_per_segment=1)
00236 | 
00237 |         result = bands.matrix_bands_from_sisl(
00238 |             method="Graph2Mat",
00239 |             sample_id="s0",
00240 |             hamiltonian_obj=FakeHamiltonian(),
00241 |             reference_obj=FakeHamiltonian(),
00242 |             kpoints=records,
00243 |             fermi_level=0.0,
00244 |             fail_closed=False,
00245 |         )
00246 | 
00247 |         self.assertEqual(result.scientific_status, "diagnostic_only")
00248 |         self.assertFalse(result.overlap_used)
00249 |         self.assertTrue(result.diagonalization_errors)
00250 | 
00251 |     def test_manifest_contains_required_fields_and_outputs(self) -> None:
00252 |         with tempfile.TemporaryDirectory() as tmp:
00253 |             root = Path(tmp)
00254 |             write_band_csv(root / "siesta.csv", offset=0.0)
00255 |             write_band_csv(root / "g2m.csv", offset=0.1)
00256 |             write_band_csv(root / "deeph.csv", offset=-0.1)
00257 |             output = root / "out"
00258 | 
00259 |             completed = subprocess.run(
00260 |                 [
00261 |                     sys.executable,
00262 |                     str(SCRIPTS_DIR / "compare_graphene_bands_siesta_g2m_deeph.py"),
00263 |                     "--sample-id",
00264 |                     "s0",
00265 |                     "--siesta-band-data",
00266 |                     str(root / "siesta.csv"),
00267 |                     "--graph2mat-band-data",
00268 |                     str(root / "g2m.csv"),
00269 |                     "--deeph-band-data",
00270 |                     str(root / "deeph.csv"),
00271 |                     "--output-dir",
00272 |                     str(output),
00273 |                     "--points-per-segment",
00274 |                     "2",
00275 |                     "--fermi-level",
00276 |                     "0.0",
00277 |                     "--skip-plot",
00278 |                 ],
00279 |                 cwd=REPO_ROOT,
00280 |                 text=True,
00281 |                 capture_output=True,
00282 |                 check=False,
00283 |             )
00284 | 
00285 |             self.assertEqual(completed.returncode, 0, completed.stderr)
00286 |             manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
00287 |             self.assertEqual(manifest["overlap_policy"], "siesta_reference_overlap_for_all_methods")
00288 |             self.assertEqual(manifest["energy_zero_policy"], "fermi")
00289 |             self.assertIn("kpath", manifest)
00290 |             self.assertIn("input_hashes", manifest)
00291 |             self.assertEqual(manifest["status"], "completed")
00292 |             self.assertTrue((output / "bands_siesta.csv").exists())
00293 |             self.assertTrue((output / "band_errors_graph2mat.csv").exists())
00294 |             self.assertTrue((output / "band_summary.json").exists())
00295 | 
00296 |     def test_plot_outputs_are_created(self) -> None:
00297 |         try:
00298 |             import matplotlib  # noqa: F401
00299 |         except ModuleNotFoundError as exc:
00300 |             self.skipTest(f"matplotlib unavailable: {exc.name}")
00301 | 
00302 |         with tempfile.TemporaryDirectory() as tmp:
00303 |             root = Path(tmp)
00304 |             write_band_csv(root / "siesta.csv", offset=0.0)
00305 |             write_band_csv(root / "g2m.csv", offset=0.1)
00306 |             write_band_csv(root / "deeph.csv", offset=-0.1)
00307 |             output = root / "out"
00308 | 
00309 |             completed = subprocess.run(
00310 |                 [
00311 |                     sys.executable,
00312 |                     str(SCRIPTS_DIR / "compare_graphene_bands_siesta_g2m_deeph.py"),
00313 |                     "--sample-id",
00314 |                     "s0",
00315 |                     "--siesta-band-data",
00316 |                     str(root / "siesta.csv"),
00317 |                     "--graph2mat-band-data",
00318 |                     str(root / "g2m.csv"),
00319 |                     "--deeph-band-data",
00320 |                     str(root / "deeph.csv"),
00321 |                     "--output-dir",
00322 |                     str(output),
00323 |                     "--points-per-segment",
00324 |                     "2",
00325 |                     "--fermi-level",
00326 |                     "0.0",
00327 |                     "--max-bands",
00328 |                     "2",
00329 |                 ],
00330 |                 cwd=REPO_ROOT,
00331 |                 text=True,
00332 |                 capture_output=True,
00333 |                 check=False,
00334 |             )
00335 | 
00336 |             self.assertEqual(completed.returncode, 0, completed.stderr)
00337 |             self.assertTrue((output / "band_comparison.png").exists())
00338 |             self.assertTrue((output / "band_comparison.pdf").exists())
00339 | 
00340 | 
00341 | if __name__ == "__main__":
00342 |     unittest.main()
```

## `Comparison/scripts/evaluate_hamiltonian_metrics.py` — extractos seleccionados

SHA-256 del archivo completo: `060b6c0fb16f8ebc2a9ff81616c2a29b2b3ce9323769cf3fad568168a3d2d0c0`

### Cabecera, imports y constantes iniciales

```py
00001 | #!/usr/bin/env python3
00002 | """Evaluate sparse, spectral and total-DOS metrics for archived Hamiltonians."""
00003 | 
00004 | from __future__ import annotations
00005 | 
00006 | import argparse
00007 | import concurrent.futures
00008 | import csv
00009 | import json
00010 | import math
00011 | import xml.etree.ElementTree as ET
00012 | from dataclasses import dataclass
00013 | from pathlib import Path
00014 | from typing import Any
00015 | 
00016 | import numpy as np
00017 | import scipy.linalg
00018 | import scipy.stats
00019 | from scipy import sparse
00020 | import sisl
00021 | import yaml
00022 | 
00023 | from reference_selection import REFERENCE_SELECTION_POLICY
00024 | from reference_selection import choose_reference_matrix
00025 | from reference_selection import file_sha256
00026 | 
00027 | 
00028 | SUPPORT_THRESHOLD = 1e-12
00029 | SUPPORT_THRESHOLDS_SWEEP = [1e-12, 1e-10, 1e-8, 1e-6]
00030 | FERMI_WINDOW_EV = 2.0
00031 | DOS_SIGMA_EV = 0.10
00032 | DOS_SIGMA_SWEEP_EV = [0.05, 0.10, 0.20, 0.40]
00033 | DOS_POINTS = 1000
00034 | DOS_FERMI_WINDOW_POINTS = 500
00035 | DOS_FERMI_WINDOW_MIN_EV = -6.0
00036 | DOS_FERMI_WINDOW_MAX_EV = 6.0
00037 | DOS_FERMI_WINDOW_ALIGNMENT = "reference_fermi_level"
00038 | LOW_ENERGY_N_STATES = 10
00039 | LOW_ENERGY_ALIGNMENT = "none"
00040 | COMPLEX_IMAG_TOLERANCE = 1e-12
00041 | OVERLAP_RELATIVE_FROBENIUS_WARNING_THRESHOLD = 1e-6
00042 | METRICS_SCHEMA_VERSION = "h_only_sref_v2"
00043 | METRICS_PROVENANCE_GENERATION = "post_h_only_sref_prediction_safety"
00044 | MATRIX_METRIC_TARGET_SPACE = "raw_global_hamiltonian"
00045 | ORBITAL_PAIR_METRIC_TARGET_SPACE = "raw_global_hamiltonian_orbital_basis"
00046 | ORBITAL_PAIR_BASIS_SOURCE = "ion_xml_pao_degeneracy_generated_labels"
00047 | MATRIX_SEMANTIC_FIELDS = [
00048 |     "metrics_schema_version",
00049 |     "metrics_provenance_generation",
00050 |     "target_component_policy",
00051 |     "reference_component_count",
00052 |     "prediction_component_count",
00053 |     "reference_spin_kind",
00054 |     "prediction_spin_kind",
00055 |     "overlap_source",
00056 |     "prediction_own_overlap_used",
00057 |     "prediction_overlap_relative_frobenius_vs_reference",
00058 |     "prediction_overlap_check_threshold",
00059 |     "graph2mat_auxiliary_component_ignored",
00060 |     "prediction_self_contained_hsx_safe",
00061 |     "prediction_self_contained_hsx_unsafe_reason",
00062 | ]
00063 | DEEPH_COMPARABILITY_STATUS = {
00064 |     "implemented_repo_compatible_metrics": [
00065 |         "hamiltonian_mae_rmse_mse_r2_on_repository_supports",
00066 |         "hamiltonian_mev_aliases",
00067 |         "dos_mae_500_fermi_window_when_reference_fermi_exists",
00068 |         "orbital_pair_metrics_csv_when_basis_mapping_exists",
00069 |     ],
00070 |     "caveats": [
00071 |         "matrix_metrics_use_raw_global_hamiltonian_not_deeph_hprime",
00072 |         "orbital_pair_metrics_use_repository_orbital_basis_not_deeph_local_hprime_blocks",
00073 |         "dos_units_and_system_dimensionality_may_not_match_deeph_2d_examples",
00074 |         "fermi_dependent_metrics_are_unavailable_when_siesta_fermi_is_missing",
00075 |     ],
00076 |     "future_work_not_implemented": {
00077 |         "high_symmetry_kpath_band_structure": (
00078 |             "requires explicit k-path input, k-resolved reference/predicted bands, "
00079 |             "and validation against SIESTA band-structure outputs"
00080 |         ),
00081 |         "soc_complex_hamiltonians": (
00082 |             "current compatibility gates reject complex Hamiltonians and unvalidated "
00083 |             "spin-orbit or multi-component matrix semantics"
00084 |         ),
00085 |         "optical_berry_susceptibility_shift_current": (
00086 |             "requires optical/Berry-response infrastructure, validated wavefunction "
00087 |             "or velocity/dipole data, and material-specific scientific checks"
00088 |         ),
00089 |         "ensemble_uncertainty": (
00090 |             "requires an explicit ensemble protocol and calibrated uncertainty "
00091 |             "validation across independent model instances"
00092 |         ),
00093 |         "deeph_vs_dft_system_size_scaling": (
00094 |             "requires controlled system-size series, DFT/DeepH timing protocol, "
00095 |             "and hardware-normalized scaling analysis"
00096 |         ),
00097 |     },
00098 | }
00099 | PERIODIC_STRUCTURE_TYPES = {"bulk", "crystal", "periodic", "solid", "surface", "slab"}
00100 | UNSUPPORTED_KPOINT_DIRECTIVES = {
00101 |     "kgrid_cutoff",
00102 |     "kgridcutoff",
00103 |     "kgrid_monkhorst_pack",
00104 |     "kgrid.monkhorstpack",
00105 | }
00106 | KGRID_MONKHORST_PACK_DIRECTIVES = {"kgrid_monkhorst_pack", "kgrid.monkhorstpack"}
00107 | RECOMMENDATION_PRIMARY_METRIC_PRIORITY = [
00108 |     "low_energy_rmse_eV",
00109 |     "frontier_window_rmse_eV",
00110 |     "occupied_rmse_eV",
00111 |     "relative_frobenius_union",
00112 |     "dos_wasserstein_eV",
00113 | ]
00114 | DIAGNOSTIC_ONLY_RECOMMENDATION_METRICS = [
00115 |     "global_rmse_eV",
00116 |     "global_mae_eV",
00117 |     "support_precision",
00118 |     "support_recall",
00119 |     "false_zeros",
00120 |     "false_nonzeros",
00121 |     "hermiticity",
00122 | ]
00123 | EIGENSOLVER_DEVICE = "cpu"
00124 | _CUPY: Any | None = None
00125 | 
```

### `MatrixData` — líneas 165–179

```py
00165 | @dataclass
00166 | class MatrixData:
00167 |     path: Path
00168 |     hamiltonian: sparse.csr_matrix
00169 |     overlap: sparse.csr_matrix | None
00170 |     own_eigenvalues: np.ndarray
00171 |     fermi_level: float | None
00172 |     fermi_level_source: str | None
00173 |     orthogonal: bool
00174 |     has_overlap: bool
00175 |     overlap_error: str | None
00176 |     sha256: str | None = None
00177 |     component_count: int = 1
00178 |     spin_kind: str | None = None
00179 |     components: tuple[sparse.csr_matrix, ...] = ()
```

### `MonkhorstPackKGrid` — líneas 182–194

```py
00182 | @dataclass(frozen=True)
00183 | class MonkhorstPackKGrid:
00184 |     mesh: tuple[int, int, int] | None
00185 |     shifts: tuple[float, float, float] | None
00186 |     is_gamma_only: bool
00187 |     source_directive: str | None
00188 |     fractional_kpoints: tuple[tuple[float, float, float], ...] = ()
00189 |     weights: tuple[float, ...] = ()
00190 |     error: str | None = None
00191 | 
00192 |     @property
00193 |     def ok(self) -> bool:
00194 |         return self.error is None and self.mesh is not None and self.shifts is not None
```

### `_monkhorst_pack_axis_points` — líneas 471–475

```py
00471 | def _monkhorst_pack_axis_points(n_points: int, shift: float) -> list[float]:
00472 |     return [
00473 |         _normalize_fractional_kpoint(((index + 0.5) / n_points) - 0.5 + shift)
00474 |         for index in range(n_points)
00475 |     ]
```

### `_monkhorst_pack_points` — líneas 478–486

```py
00478 | def _monkhorst_pack_points(
00479 |     mesh: tuple[int, int, int],
00480 |     shifts: tuple[float, float, float],
00481 | ) -> tuple[tuple[float, float, float], ...]:
00482 |     axes = [
00483 |         _monkhorst_pack_axis_points(mesh[index], shifts[index])
00484 |         for index in range(3)
00485 |     ]
00486 |     return tuple((kx, ky, kz) for kx in axes[0] for ky in axes[1] for kz in axes[2])
```

### `_monkhorst_pack_grid` — líneas 489–507

```py
00489 | def _monkhorst_pack_grid(
00490 |     mesh: tuple[int, int, int],
00491 |     shifts: tuple[float, float, float],
00492 |     source_directive: str,
00493 | ) -> MonkhorstPackKGrid:
00494 |     points = _monkhorst_pack_points(mesh, shifts)
00495 |     weight = 1.0 / len(points)
00496 |     gamma = (
00497 |         mesh == (1, 1, 1)
00498 |         and all(math.isclose(shift, 0.0, rel_tol=0.0, abs_tol=1e-12) for shift in shifts)
00499 |     )
00500 |     return MonkhorstPackKGrid(
00501 |         mesh=mesh,
00502 |         shifts=shifts,
00503 |         is_gamma_only=gamma,
00504 |         source_directive=source_directive,
00505 |         fractional_kpoints=points,
00506 |         weights=tuple(weight for _ in points),
00507 |     )
```

### `parse_monkhorst_pack_kgrid` — líneas 557–591

```py
00557 | def parse_monkhorst_pack_kgrid(structure_path: Path) -> MonkhorstPackKGrid | None:
00558 |     """Parse a SIESTA Monkhorst-Pack k-grid from an FDF file, if present."""
00559 |     if not structure_path.exists():
00560 |         return None
00561 |     kgrid_block_name: str | None = None
00562 |     kgrid_block_rows: list[str] = []
00563 |     try:
00564 |         lines = structure_path.read_text(encoding="utf-8", errors="ignore").splitlines()
00565 |     except OSError as exc:
00566 |         return _kgrid_error(None, str(exc))
00567 |     for raw_line in lines:
00568 |         clean = _strip_fdf_comment(raw_line)
00569 |         if not clean:
00570 |             continue
00571 |         lower = clean.lower()
00572 |         parts = lower.split()
00573 |         key = parts[0] if parts else ""
00574 |         if lower.startswith("%block"):
00575 |             block_name = parts[1] if len(parts) > 1 else ""
00576 |             if block_name in KGRID_MONKHORST_PACK_DIRECTIVES:
00577 |                 kgrid_block_name = block_name
00578 |                 kgrid_block_rows = []
00579 |             continue
00580 |         if lower.startswith("%endblock"):
00581 |             if kgrid_block_name is not None:
00582 |                 return _parse_monkhorst_pack_rows(kgrid_block_rows, kgrid_block_name)
00583 |             continue
00584 |         if kgrid_block_name is not None:
00585 |             kgrid_block_rows.append(clean)
00586 |             continue
00587 |         if key in KGRID_MONKHORST_PACK_DIRECTIVES:
00588 |             return _parse_monkhorst_pack_inline(clean.split()[1:], key)
00589 |     if kgrid_block_name is not None:
00590 |         return _kgrid_error(kgrid_block_name, "unterminated_monkhorst_pack_block")
00591 |     return None
```

### `evaluate_kpoint_sample` — líneas 787–1135

```py
00787 | def evaluate_kpoint_sample(
00788 |     sample: str,
00789 |     predicted_path: Path,
00790 |     reference_path: Path,
00791 |     result_dir: Path,
00792 |     kgrid: MonkhorstPackKGrid,
00793 |     *,
00794 |     target_component_policy: str,
00795 |     low_energy_enabled: bool,
00796 |     low_energy_n_states: int,
00797 |     low_energy_alignment: str,
00798 | ) -> dict[str, list[dict[str, Any]]]:
00799 |     rows = empty_sample_rows()
00800 |     sample_errors: list[dict[str, Any]] = []
00801 |     sample_warnings: list[dict[str, Any]] = []
00802 |     if not kgrid.ok or kgrid.mesh is None or kgrid.shifts is None:
00803 |         issue = append_issue(
00804 |             rows,
00805 |             "fatal_errors",
00806 |             sample=sample,
00807 |             kind="kpoint_grid_parse",
00808 |             message=kgrid.error or "Missing or invalid Monkhorst-Pack k-grid.",
00809 |         )
00810 |         rows["errors"].append(issue)
00811 |         rows["sample_status"].append(
00812 |             sample_status_row(
00813 |                 sample,
00814 |                 prediction_path=predicted_path,
00815 |                 reference_path=reference_path,
00816 |                 errors=[issue],
00817 |                 warnings=[],
00818 |             )
00819 |         )
00820 |         return rows
00821 | 
00822 |     try:
00823 |         reference = read_matrix(reference_path)
00824 |         predicted = read_matrix(predicted_path)
00825 |         reference_obj = sisl.get_sile(str(reference_path)).read_hamiltonian()
00826 |         predicted_obj = sisl.get_sile(str(predicted_path)).read_hamiltonian()
00827 |     except Exception as exc:
00828 |         issue = append_issue(rows, "fatal_errors", sample=sample, kind="read_matrix", message=str(exc))
00829 |         rows["errors"].append(issue)
00830 |         rows["sample_status"].append(
00831 |             sample_status_row(
00832 |                 sample,
00833 |                 prediction_path=predicted_path,
00834 |                 reference_path=reference_path,
00835 |                 errors=[issue],
00836 |                 warnings=[],
00837 |             )
00838 |         )
00839 |         return rows
00840 | 
00841 |     semantics = matrix_semantics_fields(
00842 |         reference,
00843 |         predicted,
00844 |         target_component_policy=target_component_policy,
00845 |     )
00846 |     for kind, data in (("siesta", reference), ("predicted", predicted)):
00847 |         rows["overlap"].append(
00848 |             {
00849 |                 "sample": sample,
00850 |                 "kind": kind,
00851 |                 "matrix_path": str(data.path),
00852 |                 "n_bands": int(data.hamiltonian.shape[0]),
00853 |                 "hamiltonian_components": int(data.component_count),
00854 |                 "spin_kind": data.spin_kind,
00855 |                 "orthogonal": data.orthogonal,
00856 |                 "has_overlap": data.has_overlap,
00857 |                 "overlap_error": data.overlap_error,
00858 |                 "fermi_level_eV": data.fermi_level,
00859 |                 **semantics,
00860 |             }
00861 |         )
00862 | 
00863 |     compatibility_errors = matrix_compatibility_errors(
00864 |         sample,
00865 |         reference,
00866 |         predicted,
00867 |         target_component_policy=target_component_policy,
00868 |     )
00869 |     compatibility_warnings = matrix_compatibility_warnings(sample, reference, predicted)
00870 |     if compatibility_warnings:
00871 |         rows["warnings"].extend(compatibility_warnings)
00872 |         sample_warnings.extend(compatibility_warnings)
00873 |     if compatibility_errors:
00874 |         rows["fatal_errors"].extend(compatibility_errors)
00875 |         rows["errors"].extend(compatibility_errors)
00876 |         sample_errors.extend(compatibility_errors)
00877 |         rows["sample_status"].append(
00878 |             sample_status_row(
00879 |                 sample,
00880 |                 prediction_path=predicted_path,
00881 |                 reference_path=reference_path,
00882 |                 errors=sample_errors,
00883 |                 warnings=sample_warnings,
00884 |             )
00885 |         )
00886 |         return rows
00887 | 
00888 |     mesh = tuple(int(value) for value in kgrid.mesh)
00889 |     shifts = tuple(float(value) for value in kgrid.shifts)
00890 |     fermi_level = reference.fermi_level
00891 |     fermi_source = reference.fermi_level_source or "unavailable"
00892 |     if fermi_level is not None and not math.isfinite(float(fermi_level)):
00893 |         fermi_level = None
00894 |         fermi_source = "unavailable"
00895 |     if fermi_level is None:
00896 |         warning = append_issue(
00897 |             rows,
00898 |             "warnings",
00899 |             sample=sample,
00900 |             kind="missing_fermi_level",
00901 |             message=(
00902 |                 "SIESTA reference does not provide a Fermi level; near-Fermi, "
00903 |                 "occupied-band, frontier, gap, and fixed-window DOS metrics were left unavailable."
00904 |             ),
00905 |         )
00906 |         sample_warnings.append(warning)
00907 | 
00908 |     per_k_spectral: list[dict[str, Any]] = []
00909 |     all_ref_eigenvalues: list[np.ndarray] = []
00910 |     all_pred_eigenvalues: list[np.ndarray] = []
00911 |     all_band_weights: list[np.ndarray] = []
00912 |     for k_index, kpoint in enumerate(kgrid.fractional_kpoints):
00913 |         weight = float(kgrid.weights[k_index])
00914 |         k_label = f"{sample}_k{k_index:04d}"
00915 |         k_metadata = {
00916 |             "sample": sample,
00917 |             "k_index": k_index,
00918 |             "k_label": k_label,
00919 |             "kx": float(kpoint[0]),
00920 |             "ky": float(kpoint[1]),
00921 |             "kz": float(kpoint[2]),
00922 |             "k_weight": weight,
00923 |             "kpoint_mesh": list(mesh),
00924 |             "kpoint_shifts": list(shifts),
00925 |             "kpoint_source": kgrid.source_directive or "",
00926 |         }
00927 |         rows["kpoint_kpoints"].append(k_metadata)
00928 |         try:
00929 |             ref_h_k = kpoint_hamiltonian_matrix(reference_obj, list(kpoint))
00930 |             pred_h_k = kpoint_hamiltonian_matrix(predicted_obj, list(kpoint))
00931 |             ref_s_k = kpoint_overlap_matrix(reference_obj, list(kpoint))
00932 |             matrix_metrics = complex_matrix_error_metrics(ref_h_k, pred_h_k)
00933 |             ref_eig = complex_generalized_eigenvalues(ref_h_k, ref_s_k)
00934 |             pred_eig = complex_generalized_eigenvalues(pred_h_k, ref_s_k)
00935 |             write_csv(
00936 |                 result_dir / "eigenvalues" / "siesta" / f"{k_label}.csv",
00937 |                 ["band", "eigenvalue_eV"],
00938 |                 eigenvalue_rows(ref_eig),
00939 |             )
00940 |             write_csv(
00941 |                 result_dir / "eigenvalues" / "predicted" / f"{k_label}.csv",
00942 |                 ["band", "eigenvalue_eV"],
00943 |                 eigenvalue_rows(pred_eig),
00944 |             )
00945 |             band_rows, spectral_metrics = eigen_error_metrics(
00946 |                 ref_eig,
00947 |                 pred_eig,
00948 |                 fermi_level,
00949 |                 fermi_source,
00950 |             )
00951 |             if low_energy_enabled:
00952 |                 spectral_metrics.update(
00953 |                     low_energy_metrics_from_eigenvalues(
00954 |                         ref_eig,
00955 |                         pred_eig,
00956 |                         n_states=low_energy_n_states,
00957 |                         alignment=low_energy_alignment,
00958 |                     )
00959 |                 )
00960 |             write_csv(
00961 |                 result_dir / "eigenvalues" / "kpoint_band_errors" / f"{k_label}.csv",
00962 |                 ["band", "siesta_eV", "predicted_eV", "error_eV", "abs_error_eV", "siesta_minus_fermi_eV"],
00963 |                 band_rows,
00964 |             )
00965 |             rows["kpoint_matrix"].append(
00966 |                 {
00967 |                     **k_metadata,
00968 |                     "row_type": "per_k",
00969 |                     "n_orbitals": int(ref_h_k.shape[0]),
00970 |                     "n_entries": int(ref_h_k.size),
00971 |                     "h_mae_eV": matrix_metrics["mae_eV"],
00972 |                     "h_rmse_eV": matrix_metrics["rmse_eV"],
00973 |                     "h_mse_eV2": matrix_metrics["mse_eV2"],
00974 |                     "h_max_abs_error_eV": matrix_metrics["max_abs_error_eV"],
00975 |                     "relative_frobenius": matrix_metrics["relative_frobenius"],
00976 |                     "hermiticity_ref": matrix_metrics["reference_hermiticity"],
00977 |                     "hermiticity_pred": matrix_metrics["prediction_hermiticity"],
00978 |                     "uses_reference_overlap_k": True,
00979 |                     **semantics,
00980 |                 }
00981 |             )
00982 |             per_k_spectral.append(
00983 |                 {
00984 |                     **k_metadata,
00985 |                     "n_compared_bands": spectral_metrics.get("n_compared_bands"),
00986 |                     "same_band_count": ref_eig.size == pred_eig.size,
00987 |                     "reference_has_overlap": ref_s_k is not None,
00988 |                     "hamiltonian_symmetrized_for_spectrum": True,
00989 |                     **spectral_metrics,
00990 |                     **semantics,
00991 |                 }
00992 |             )
00993 |             all_ref_eigenvalues.append(np.asarray(ref_eig, dtype=float))
00994 |             all_pred_eigenvalues.append(np.asarray(pred_eig, dtype=float))
00995 |             all_band_weights.append(np.full(ref_eig.size, weight, dtype=float))
00996 |         except Exception as exc:
00997 |             issue = append_issue(
00998 |                 rows,
00999 |                 "fatal_errors",
01000 |                 sample=sample,
01001 |                 kind="kpoint_metrics",
01002 |                 message=f"k-index {k_index} failed: {exc}",
01003 |             )
01004 |             rows["errors"].append(issue)
01005 |             sample_errors.append(issue)
01006 | 
01007 |     if sample_errors:
01008 |         rows["sample_status"].append(
01009 |             sample_status_row(
01010 |                 sample,
01011 |                 prediction_path=predicted_path,
01012 |                 reference_path=reference_path,
01013 |                 errors=sample_errors,
01014 |                 warnings=sample_warnings,
01015 |             )
01016 |         )
01017 |         return rows
01018 | 
01019 |     matrix_per_k = [row for row in rows["kpoint_matrix"] if row.get("row_type") == "per_k"]
01020 |     rows["kpoint_matrix"].append(
01021 |         {
01022 |             "sample": sample,
01023 |             "k_index": "",
01024 |             "k_label": f"{sample}_weighted",
01025 |             "kx": math.nan,
01026 |             "ky": math.nan,
01027 |             "kz": math.nan,
01028 |             "k_weight": 1.0,
01029 |             "kpoint_mesh": list(mesh),
01030 |             "kpoint_shifts": list(shifts),
01031 |             "kpoint_source": kgrid.source_directive or "",
01032 |             "row_type": "weighted_sample",
01033 |             "n_orbitals": int(reference.hamiltonian.shape[0]),
01034 |             "n_entries": int(reference.hamiltonian.shape[0] * reference.hamiltonian.shape[1]),
01035 |             "h_mae_eV": weighted_metric_mean(matrix_per_k, "h_mae_eV"),
01036 |             "h_rmse_eV": weighted_metric_rmse(matrix_per_k, "h_rmse_eV"),
01037 |             "h_mse_eV2": weighted_metric_mean(matrix_per_k, "h_mse_eV2"),
01038 |             "h_max_abs_error_eV": max(
01039 |                 (float(row["h_max_abs_error_eV"]) for row in matrix_per_k if math.isfinite(float(row["h_max_abs_error_eV"]))),
01040 |                 default=math.nan,
01041 |             ),
01042 |             "relative_frobenius": weighted_metric_mean(matrix_per_k, "relative_frobenius"),
01043 |             "hermiticity_ref": weighted_metric_mean(matrix_per_k, "hermiticity_ref"),
01044 |             "hermiticity_pred": weighted_metric_mean(matrix_per_k, "hermiticity_pred"),
01045 |             "uses_reference_overlap_k": True,
01046 |             **semantics,
01047 |         }
01048 |     )
01049 |     rows["kpoint_spectral"].append(
01050 |         {
01051 |             "sample": sample,
01052 |             "kpoint_count": len(kgrid.fractional_kpoints),
01053 |             "kpoint_mesh": list(mesh),
01054 |             "kpoint_shifts": list(shifts),
01055 |             "kpoint_source": kgrid.source_directive or "",
01056 |             "siesta_bands": int(all_ref_eigenvalues[0].size) if all_ref_eigenvalues else 0,
01057 |             "predicted_bands": int(all_pred_eigenvalues[0].size) if all_pred_eigenvalues else 0,
01058 |             "spectral_comparable": bool(per_k_spectral),
01059 |             "same_band_count": all(bool(row.get("same_band_count")) for row in per_k_spectral),
01060 |             "reference_has_overlap": any(bool(row.get("reference_has_overlap")) for row in per_k_spectral),
01061 |             "hamiltonian_symmetrized_for_spectrum": True,
01062 |             "uses_reference_overlap_k": True,
01063 |             "n_compared_bands": weighted_metric_mean(per_k_spectral, "n_compared_bands"),
01064 |             "fermi_ref_eV": fermi_level,
01065 |             "fermi_level_source": fermi_source,
01066 |             "fermi_metric_available": fermi_level is not None,
01067 |             "global_mae_eV": weighted_metric_mean(per_k_spectral, "global_mae_eV"),
01068 |             "global_rmse_eV": weighted_metric_rmse(per_k_spectral, "global_rmse_eV"),
01069 |             "global_max_abs_error_eV": max(
01070 |                 (float(row["global_max_abs_error_eV"]) for row in per_k_spectral if math.isfinite(float(row["global_max_abs_error_eV"]))),
01071 |                 default=math.nan,
01072 |             ),
01073 |             "global_mean_signed_error_eV": weighted_metric_mean(per_k_spectral, "global_mean_signed_error_eV"),
01074 |             "occupied_bands": weighted_metric_mean(per_k_spectral, "occupied_bands"),
01075 |             "occupied_metric_available": any(bool(row.get("occupied_metric_available")) for row in per_k_spectral),
01076 |             "occupied_mae_eV": weighted_metric_mean(per_k_spectral, "occupied_mae_eV"),
01077 |             "occupied_rmse_eV": weighted_metric_rmse(per_k_spectral, "occupied_rmse_eV"),
01078 |             "fermi_window_eV": FERMI_WINDOW_EV,
01079 |             "fermi_window_bands": weighted_metric_mean(per_k_spectral, "fermi_window_bands"),
01080 |             "fermi_window_metric_available": any(bool(row.get("fermi_window_metric_available")) for row in per_k_spectral),
01081 |             "fermi_window_mae_eV": weighted_metric_mean(per_k_spectral, "fermi_window_mae_eV"),
01082 |             "fermi_window_rmse_eV": weighted_metric_rmse(per_k_spectral, "fermi_window_rmse_eV"),
01083 |             "frontier_window_bands": weighted_metric_mean(per_k_spectral, "frontier_window_bands"),
01084 |             "frontier_metric_available": any(bool(row.get("frontier_metric_available")) for row in per_k_spectral),
01085 |             "frontier_window_mae_eV": weighted_metric_mean(per_k_spectral, "frontier_window_mae_eV"),
01086 |             "frontier_window_rmse_eV": weighted_metric_rmse(per_k_spectral, "frontier_window_rmse_eV"),
01087 |             "gap_abs_error_eV": weighted_metric_mean(per_k_spectral, "gap_abs_error_eV"),
01088 |             "low_energy_requested_states": low_energy_n_states,
01089 |             "low_energy_n_states": weighted_metric_mean(per_k_spectral, "low_energy_n_states"),
01090 |             "low_energy_mae_eV": weighted_metric_mean(per_k_spectral, "low_energy_mae_eV"),
01091 |             "low_energy_rmse_eV": weighted_metric_rmse(per_k_spectral, "low_energy_rmse_eV"),
01092 |             "low_energy_max_abs_error_eV": max(
01093 |                 (float(row["low_energy_max_abs_error_eV"]) for row in per_k_spectral if math.isfinite(float(row["low_energy_max_abs_error_eV"]))),
01094 |                 default=math.nan,
01095 |             ),
01096 |             "low_energy_alignment": low_energy_alignment,
01097 |             "low_energy_aligned_rmse_eV": weighted_metric_rmse(per_k_spectral, "low_energy_aligned_rmse_eV"),
01098 |             "low_energy_overlap_used": True,
01099 |             "low_energy_overlap_required": True,
01100 |             "low_energy_solver": eigensolver_name(generalized=True, kpoint=True),
01101 |             "low_energy_warning": "",
01102 |             **semantics,
01103 |         }
01104 |     )
01105 |     if all_ref_eigenvalues and all_pred_eigenvalues and all_band_weights:
01106 |         ref_flat = np.concatenate(all_ref_eigenvalues)
01107 |         pred_flat = np.concatenate(all_pred_eigenvalues)
01108 |         weights_flat = np.concatenate(all_band_weights)
01109 |     else:
01110 |         ref_flat = np.asarray([], dtype=float)
01111 |         pred_flat = np.asarray([], dtype=float)
01112 |         weights_flat = np.asarray([], dtype=float)
01113 |     rows["kpoint_dos"].append(
01114 |         {
01115 |             "sample": sample,
01116 |             "kpoint_count": len(kgrid.fractional_kpoints),
01117 |             "kpoint_mesh": list(mesh),
01118 |             "kpoint_shifts": list(shifts),
01119 |             "kpoint_source": kgrid.source_directive or "",
01120 |             "weighted_eigenvalue_count": int(ref_flat.size),
01121 |             "fermi_level_source": fermi_source,
01122 |             **semantics,
01123 |             **kpoint_weighted_dos_metrics(ref_flat, pred_flat, weights_flat, fermi_level),
01124 |         }
01125 |     )
01126 |     rows["sample_status"].append(
01127 |         sample_status_row(
01128 |             sample,
01129 |             prediction_path=predicted_path,
01130 |             reference_path=reference_path,
01131 |             errors=sample_errors,
01132 |             warnings=sample_warnings,
01133 |         )
01134 |     )
01135 |     return rows
```

### `evaluate_sample` — líneas 1138–1488

```py
01138 | def evaluate_sample(
01139 |     sample: str,
01140 |     prediction_dir: Path | None,
01141 |     reference_dir: Path | None,
01142 |     result_dir: Path,
01143 |     basis_counts: dict[str, int],
01144 |     *,
01145 |     method_id: str = "",
01146 |     target_component_policy: str = "unknown",
01147 |     low_energy_enabled: bool,
01148 |     low_energy_n_states: int,
01149 |     low_energy_alignment: str,
01150 |     enable_kpoint_metrics: bool = False,
01151 | ) -> dict[str, list[dict[str, Any]]]:
01152 |     rows = empty_sample_rows()
01153 |     sample_errors: list[dict[str, Any]] = []
01154 |     sample_warnings: list[dict[str, Any]] = []
01155 |     predicted_path = find_prediction(prediction_dir) if prediction_dir is not None else None
01156 |     prediction_fallbacks = (
01157 |         fallback_prediction_candidates(prediction_dir)
01158 |         if prediction_dir is not None and predicted_path is None
01159 |         else []
01160 |     )
01161 |     reference_selection = choose_reference_matrix(reference_dir) if reference_dir is not None else None
01162 |     reference_path = reference_selection.path if reference_selection and reference_selection.ok else None
01163 |     if predicted_path is None:
01164 |         if prediction_fallbacks:
01165 |             sample_errors.append(
01166 |                 append_issue(
01167 |                     rows,
01168 |                     "fatal_errors",
01169 |                     sample=sample,
01170 |                     kind="noncanonical_prediction",
01171 |                     message=(
01172 |                         "Missing canonical predicted Hamiltonian ML_prediction.HSX; "
01173 |                         "fallback prediction files are not selected automatically."
01174 |                     ),
01175 |                     candidate_count=len(prediction_fallbacks),
01176 |                     candidates=[str(path) for path in prediction_fallbacks],
01177 |                 )
01178 |             )
01179 |         else:
01180 |             sample_errors.append(
01181 |                 append_issue(
01182 |                     rows,
01183 |                     "fatal_errors",
01184 |                     sample=sample,
01185 |                     kind="missing_prediction",
01186 |                     message="Missing predicted Hamiltonian.",
01187 |                 )
01188 |             )
01189 |     if reference_selection is None:
01190 |         sample_errors.append(
01191 |             append_issue(
01192 |                 rows,
01193 |                 "fatal_errors",
01194 |                 sample=sample,
01195 |                 kind="missing_reference_dir",
01196 |                 message="Missing SIESTA reference directory.",
01197 |             )
01198 |         )
01199 |     elif not reference_selection.ok:
01200 |         sample_errors.append(
01201 |             append_issue(
01202 |                 rows,
01203 |                 "fatal_errors",
01204 |                 sample=sample,
01205 |                 kind="reference_selection",
01206 |                 message=reference_selection.reason,
01207 |                 candidate_count=reference_selection.candidate_count,
01208 |                 candidates=list(reference_selection.candidates),
01209 |             )
01210 |         )
01211 |     stale_issue = stale_reference_issue(
01212 |         sample,
01213 |         reference_path,
01214 |         result_dir / "structures" / sample / "RUN.fdf",
01215 |         method_id=method_id,
01216 |     )
01217 |     if stale_issue is not None:
01218 |         if stale_issue.get("severity") == "warning":
01219 |             rows["warnings"].append(stale_issue)
01220 |             sample_warnings.append(stale_issue)
01221 |         else:
01222 |             rows["fatal_errors"].append(stale_issue)
01223 |             sample_errors.append(stale_issue)
01224 |     structure_path = result_dir / "structures" / sample / "RUN.fdf"
01225 |     kgrid = parse_monkhorst_pack_kgrid(structure_path)
01226 |     if enable_kpoint_metrics and kgrid is None:
01227 |         kgrid = _monkhorst_pack_grid((1, 1, 1), (0.0, 0.0, 0.0), "implicit_gamma_only")
01228 |     if (
01229 |         enable_kpoint_metrics
01230 |         and kgrid is not None
01231 |         and kgrid.ok
01232 |         and predicted_path is not None
01233 |         and reference_path is not None
01234 |         and not sample_errors
01235 |     ):
01236 |         kpoint_rows = evaluate_kpoint_sample(
01237 |             sample,
01238 |             predicted_path,
01239 |             reference_path,
01240 |             result_dir,
01241 |             kgrid,
01242 |             target_component_policy=target_component_policy,
01243 |             low_energy_enabled=low_energy_enabled,
01244 |             low_energy_n_states=low_energy_n_states,
01245 |             low_energy_alignment=low_energy_alignment,
01246 |         )
01247 |         if sample_warnings:
01248 |             kpoint_rows["warnings"] = [*sample_warnings, *kpoint_rows["warnings"]]
01249 |             if kpoint_rows["sample_status"]:
01250 |                 current = kpoint_rows["sample_status"][0].get("warnings") or []
01251 |                 kpoint_rows["sample_status"][0]["warnings"] = [*sample_warnings, *current]
01252 |         return kpoint_rows
01253 |     for issue in unsupported_kpoint_issues(sample, structure_path):
01254 |         rows["fatal_errors"].append(issue)
01255 |         sample_errors.append(issue)
01256 |     if sample_errors:
01257 |         rows["errors"].extend(sample_errors)
01258 |         rows["sample_status"].append(
01259 |             sample_status_row(
01260 |                 sample,
01261 |                 prediction_path=predicted_path,
01262 |                 reference_path=reference_path,
01263 |                 errors=sample_errors,
01264 |                 warnings=sample_warnings,
01265 |             )
01266 |         )
01267 |         return rows
01268 | 
01269 |     try:
01270 |         assert reference_path is not None and predicted_path is not None
01271 |         reference = read_matrix(reference_path)
01272 |         predicted = read_matrix(predicted_path)
01273 |     except Exception as exc:
01274 |         sample_errors.append(
01275 |             append_issue(rows, "fatal_errors", sample=sample, kind="read_matrix", message=str(exc))
01276 |         )
01277 |         rows["errors"].extend(sample_errors)
01278 |         rows["sample_status"].append(
01279 |             sample_status_row(
01280 |                 sample,
01281 |                 prediction_path=predicted_path,
01282 |                 reference_path=reference_path,
01283 |                 errors=sample_errors,
01284 |                 warnings=sample_warnings,
01285 |             )
01286 |         )
01287 |         return rows
01288 | 
01289 |     semantics = matrix_semantics_fields(
01290 |         reference,
01291 |         predicted,
01292 |         target_component_policy=target_component_policy,
01293 |     )
01294 | 
01295 |     for kind, data in (("siesta", reference), ("predicted", predicted)):
01296 |         rows["overlap"].append(
01297 |             {
01298 |                 "sample": sample,
01299 |                 "kind": kind,
01300 |                 "matrix_path": str(data.path),
01301 |                 "n_bands": int(data.hamiltonian.shape[0]),
01302 |                 "hamiltonian_components": int(data.component_count),
01303 |                 "spin_kind": data.spin_kind,
01304 |                 "orthogonal": data.orthogonal,
01305 |                 "has_overlap": data.has_overlap,
01306 |                 "overlap_error": data.overlap_error,
01307 |                 "fermi_level_eV": data.fermi_level,
01308 |                 **semantics,
01309 |             }
01310 |         )
01311 | 
01312 |     rows["component"].extend(component_channel_metrics(sample, reference, predicted, semantics))
01313 | 
01314 |     compatibility_errors = matrix_compatibility_errors(
01315 |         sample,
01316 |         reference,
01317 |         predicted,
01318 |         target_component_policy=target_component_policy,
01319 |     )
01320 |     compatibility_warnings = matrix_compatibility_warnings(sample, reference, predicted)
01321 |     if compatibility_warnings:
01322 |         rows["warnings"].extend(compatibility_warnings)
01323 |         sample_warnings.extend(compatibility_warnings)
01324 |     if compatibility_errors:
01325 |         rows["fatal_errors"].extend(compatibility_errors)
01326 |         rows["errors"].extend(compatibility_errors)
01327 |         sample_errors.extend(compatibility_errors)
01328 |         rows["sample_status"].append(
01329 |             sample_status_row(
01330 |                 sample,
01331 |                 prediction_path=predicted_path,
01332 |                 reference_path=reference_path,
01333 |                 errors=sample_errors,
01334 |                 warnings=sample_warnings,
01335 |             )
01336 |         )
01337 |         return rows
01338 | 
01339 |     try:
01340 |         rows["sparse"].append({**sparse_metrics(sample, reference, predicted), **semantics})
01341 |         rows["sparse_sweep"].extend(sparse_threshold_sweep_metrics(sample, reference, predicted))
01342 |     except Exception as exc:
01343 |         issue = append_issue(rows, "fatal_errors", sample=sample, kind="sparse_metrics", message=str(exc))
01344 |         rows["errors"].append(issue)
01345 |         sample_errors.append(issue)
01346 | 
01347 |     try:
01348 |         structural = structural_sparse_metrics(
01349 |             sample,
01350 |             reference,
01351 |             predicted,
01352 |             result_dir / "structures" / sample / "RUN.fdf",
01353 |             basis_counts,
01354 |         )
01355 |         structural_warnings = structural.get("warnings", []) or []
01356 |         if structural_warnings:
01357 |             rows["warnings"].extend(structural_warnings)
01358 |             sample_warnings.extend(structural_warnings)
01359 |         if structural["available"]:
01360 |             rows["block"].extend(structural["block_rows"])
01361 |             rows["species_pair"].extend(structural["species_pair_rows"])
01362 |             rows["distance_bin"].extend(structural["distance_bin_rows"])
01363 |             rows["orbital_pair"].extend(structural["orbital_pair_rows"])
01364 |         else:
01365 |             rows["structural_unavailable"].append({"sample": sample, "reason": structural["reason"]})
01366 |         if structural.get("distance_unavailable_reason"):
01367 |             rows["structural_unavailable"].append(
01368 |                 {"sample": sample, "reason": str(structural["distance_unavailable_reason"])}
01369 |             )
01370 |     except Exception as exc:
01371 |         rows["structural_unavailable"].append({"sample": sample, "reason": str(exc)})
01372 |         sample_warnings.append(
01373 |             append_issue(
01374 |                 rows,
01375 |                 "warnings",
01376 |                 sample=sample,
01377 |                 kind="structural_metrics",
01378 |                 severity="severe",
01379 |                 message=str(exc),
01380 |             )
01381 |         )
01382 | 
01383 |     eigen_root = result_dir / "eigenvalues"
01384 |     dos_root = result_dir / "dos"
01385 |     try:
01386 |         ref_eig = generalized_eigenvalues(reference.hamiltonian, reference.overlap)
01387 |         pred_eig = generalized_eigenvalues(predicted.hamiltonian, reference.overlap)
01388 |         write_csv(eigen_root / "siesta" / f"{sample}.csv", ["band", "eigenvalue_eV"], eigenvalue_rows(ref_eig))
01389 |         write_csv(eigen_root / "predicted" / f"{sample}.csv", ["band", "eigenvalue_eV"], eigenvalue_rows(pred_eig))
01390 |         fermi_level = reference.fermi_level
01391 |         fermi_source = reference.fermi_level_source or "unavailable"
01392 |         same_band_count = ref_eig.size == pred_eig.size
01393 |         spectral_comparable = bool(
01394 |             (reference.orthogonal or reference.has_overlap)
01395 |             and same_band_count
01396 |             and fermi_level is not None
01397 |         )
01398 |         if fermi_level is None or not math.isfinite(fermi_level):
01399 |             sample_warnings.append(
01400 |                 append_issue(
01401 |                     rows,
01402 |                     "warnings",
01403 |                     sample=sample,
01404 |                     kind="missing_fermi_level",
01405 |                     message=(
01406 |                         "SIESTA reference does not provide a Fermi level; "
01407 |                         "near-Fermi, occupied-band, frontier, gap, and fixed-window DOS metrics were left unavailable."
01408 |                     ),
01409 |                 )
01410 |             )
01411 |             fermi_level = None
01412 |             fermi_source = "unavailable"
01413 |             spectral_comparable = False
01414 |         band_rows, spectral_metrics = eigen_error_metrics(
01415 |             ref_eig,
01416 |             pred_eig,
01417 |             fermi_level,
01418 |             fermi_source,
01419 |         )
01420 |         if low_energy_enabled:
01421 |             low_metrics = low_energy_metrics(
01422 |                 reference,
01423 |                 predicted,
01424 |                 n_states=low_energy_n_states,
01425 |                 alignment=low_energy_alignment,
01426 |             )
01427 |             spectral_metrics.update(low_metrics)
01428 |             if low_metrics.get("low_energy_warning"):
01429 |                 sample_warnings.append(
01430 |                     append_issue(
01431 |                         rows,
01432 |                         "warnings",
01433 |                         sample=sample,
01434 |                         kind="low_energy_metrics",
01435 |                         message=str(low_metrics["low_energy_warning"]),
01436 |                     )
01437 |                 )
01438 |         write_csv(
01439 |             eigen_root / "band_errors" / f"{sample}.csv",
01440 |             ["band", "siesta_eV", "predicted_eV", "error_eV", "abs_error_eV", "siesta_minus_fermi_eV"],
01441 |             band_rows,
01442 |         )
01443 |         rows["spectral"].append(
01444 |             {
01445 |                 "sample": sample,
01446 |                 "siesta_bands": int(ref_eig.size),
01447 |                 "predicted_bands": int(pred_eig.size),
01448 |                 "spectral_comparable": spectral_comparable,
01449 |                 "same_band_count": same_band_count,
01450 |                 "reference_has_overlap": reference.has_overlap,
01451 |                 "hamiltonian_symmetrized_for_spectrum": True,
01452 |                 **semantics,
01453 |                 **spectral_metrics,
01454 |             }
01455 |         )
01456 |         dos_grid_rows, dos_metrics = dos_for_sample(ref_eig, pred_eig)
01457 |         _dos_window_grid, dos_window_metrics = dos_fermi_window_metrics(ref_eig, pred_eig, fermi_level)
01458 |         write_csv(
01459 |             dos_root / f"{sample}.csv",
01460 |             ["energy_eV", "siesta_dos", "predicted_dos", "siesta_dos_normalized", "predicted_dos_normalized"],
01461 |             dos_grid_rows,
01462 |         )
01463 |         rows["dos"].append(
01464 |             {
01465 |                 "sample": sample,
01466 |                 "fermi_level_source": fermi_source,
01467 |                 **semantics,
01468 |                 **dos_metrics,
01469 |                 **dos_window_metrics,
01470 |             }
01471 |         )
01472 |         for sigma in DOS_SIGMA_SWEEP_EV:
01473 |             _grid_rows, sweep_metrics = dos_for_sample(ref_eig, pred_eig, sigma_ev=sigma)
01474 |             rows["dos_sweep"].append({"sample": sample, **sweep_metrics})
01475 |     except Exception as exc:
01476 |         issue = append_issue(rows, "fatal_errors", sample=sample, kind="spectral_or_dos_metrics", message=str(exc))
01477 |         rows["errors"].append(issue)
01478 |         sample_errors.append(issue)
01479 |     rows["sample_status"].append(
01480 |         sample_status_row(
01481 |             sample,
01482 |             prediction_path=predicted_path,
01483 |             reference_path=reference_path,
01484 |             errors=sample_errors,
01485 |             warnings=sample_warnings,
01486 |         )
01487 |     )
01488 |     return rows
```

### `read_matrix` — líneas 1511–1552

```py
01511 | def read_matrix(path: Path) -> MatrixData:
01512 |     sile = sisl.get_sile(str(path))
01513 |     hamiltonian_obj = sile.read_hamiltonian()
01514 |     component_count = infer_component_count(hamiltonian_obj)
01515 |     hamiltonian = hamiltonian_obj.tocsr(0)
01516 |     components = [hamiltonian]
01517 |     for component_index in range(1, component_count):
01518 |         components.append(hamiltonian_obj.tocsr(component_index))
01519 |     overlap = None
01520 |     has_overlap = False
01521 |     overlap_error = None
01522 |     try:
01523 |         overlap_obj = sile.read_overlap()
01524 |         overlap = overlap_obj.tocsr()
01525 |         has_overlap = overlap is not None
01526 |     except Exception as exc:  # pragma: no cover - backend dependent.
01527 |         overlap_error = str(exc)
01528 |     try:
01529 |         own_eigenvalues = np.asarray(hamiltonian_obj.eigh(), dtype=float)
01530 |     except Exception:
01531 |         own_eigenvalues = np.asarray([], dtype=float)
01532 |     try:
01533 |         fermi_level = float(sile.read_fermi_level())
01534 |         fermi_level_source = "siesta_file"
01535 |     except Exception:
01536 |         fermi_level = None
01537 |         fermi_level_source = "unavailable"
01538 |     return MatrixData(
01539 |         path=path,
01540 |         hamiltonian=hamiltonian,
01541 |         overlap=overlap,
01542 |         own_eigenvalues=own_eigenvalues,
01543 |         fermi_level=fermi_level,
01544 |         fermi_level_source=fermi_level_source,
01545 |         orthogonal=bool(getattr(hamiltonian_obj, "orthogonal", False)),
01546 |         has_overlap=has_overlap,
01547 |         overlap_error=overlap_error,
01548 |         sha256=file_sha256(path),
01549 |         component_count=component_count,
01550 |         spin_kind=str(getattr(hamiltonian_obj, "spin", "")) or None,
01551 |         components=tuple(components),
01552 |     )
```

### `matrix_compatibility_errors` — líneas 1555–1702

```py
01555 | def matrix_compatibility_errors(
01556 |     sample: str,
01557 |     reference: MatrixData,
01558 |     predicted: MatrixData,
01559 |     *,
01560 |     target_component_policy: str = "unknown",
01561 | ) -> list[dict[str, Any]]:
01562 |     errors: list[dict[str, Any]] = []
01563 |     if reference.hamiltonian.shape != predicted.hamiltonian.shape:
01564 |         errors.append(
01565 |             {
01566 |                 "sample": sample,
01567 |                 "kind": "matrix_shape_mismatch",
01568 |                 "error": (
01569 |                     "Reference and prediction Hamiltonian shapes differ: "
01570 |                     f"{reference.hamiltonian.shape} vs {predicted.hamiltonian.shape}."
01571 |                 ),
01572 |                 "reference_shape": list(reference.hamiltonian.shape),
01573 |                 "predicted_shape": list(predicted.hamiltonian.shape),
01574 |             }
01575 |         )
01576 |     graph2mat_auxiliary_prediction = is_graph2mat_auxiliary_prediction(reference, predicted)
01577 |     for role, data in (("reference", reference), ("prediction", predicted)):
01578 |         if matrix_has_complex_values(data.hamiltonian):
01579 |             errors.append(
01580 |                 {
01581 |                     "sample": sample,
01582 |                     "kind": "unsupported_complex_hamiltonian",
01583 |                     "error": (
01584 |                         f"Unsupported complex-valued {role} Hamiltonian. The current benchmark "
01585 |                         "does not validate spin-orbit/k-point complex matrix semantics."
01586 |                     ),
01587 |                     "matrix_role": role,
01588 |                     "matrix_path": str(data.path),
01589 |                 }
01590 |             )
01591 |         if data.overlap is not None and matrix_has_complex_values(data.overlap):
01592 |             errors.append(
01593 |                 {
01594 |                     "sample": sample,
01595 |                     "kind": "unsupported_complex_overlap",
01596 |                     "error": (
01597 |                         f"Unsupported complex-valued {role} overlap. The current benchmark "
01598 |                         "does not validate complex generalized eigenproblems."
01599 |                     ),
01600 |                     "matrix_role": role,
01601 |                     "matrix_path": str(data.path),
01602 |                 }
01603 |             )
01604 |     if reference.component_count != 1:
01605 |         errors.append(
01606 |             {
01607 |                 "sample": sample,
01608 |                 "kind": "unsupported_matrix_components",
01609 |                 "error": f"Unsupported reference matrix component count: {reference.component_count}.",
01610 |                 "component_count": reference.component_count,
01611 |             }
01612 |         )
01613 |     if (
01614 |         target_component_policy == "h_only"
01615 |         and predicted.component_count != 1
01616 |         and not graph2mat_auxiliary_prediction
01617 |     ):
01618 |         errors.append(
01619 |             {
01620 |                 "sample": sample,
01621 |                 "kind": "target_component_policy_mismatch",
01622 |                 "error": (
01623 |                     "Expected H-only prediction semantics, but the prediction "
01624 |                     f"container reports {predicted.component_count} Hamiltonian components."
01625 |                 ),
01626 |                 "target_component_policy": target_component_policy,
01627 |                 "prediction_component_count": predicted.component_count,
01628 |             }
01629 |         )
01630 |     if predicted.component_count != 1 and not graph2mat_auxiliary_prediction:
01631 |         errors.append(
01632 |             {
01633 |                 "sample": sample,
01634 |                 "kind": "unsupported_matrix_components",
01635 |                 "error": f"Unsupported prediction matrix component count: {predicted.component_count}.",
01636 |                 "component_count": predicted.component_count,
01637 |             }
01638 |         )
01639 |     if (
01640 |         reference.spin_kind
01641 |         and predicted.spin_kind
01642 |         and reference.spin_kind != predicted.spin_kind
01643 |         and not graph2mat_auxiliary_prediction
01644 |     ):
01645 |         errors.append(
01646 |             {
01647 |                 "sample": sample,
01648 |                 "kind": "spin_state_mismatch",
01649 |                 "error": (
01650 |                     "Reference and prediction spin metadata differ: "
01651 |                     f"{reference.spin_kind} vs {predicted.spin_kind}."
01652 |                 ),
01653 |                 "reference_spin": reference.spin_kind,
01654 |                 "predicted_spin": predicted.spin_kind,
01655 |             }
01656 |         )
01657 |     if unsupported_spin_kind(reference.spin_kind):
01658 |         errors.append(
01659 |             {
01660 |                 "sample": sample,
01661 |                 "kind": "unsupported_spin_kind",
01662 |                 "error": f"Unsupported reference spin metadata: {reference.spin_kind}.",
01663 |                 "matrix_role": "reference",
01664 |                 "spin_kind": reference.spin_kind,
01665 |             }
01666 |         )
01667 |     if unsupported_spin_kind(predicted.spin_kind) and not graph2mat_auxiliary_prediction:
01668 |         errors.append(
01669 |             {
01670 |                 "sample": sample,
01671 |                 "kind": "unsupported_spin_kind",
01672 |                 "error": f"Unsupported prediction spin metadata: {predicted.spin_kind}.",
01673 |                 "matrix_role": "prediction",
01674 |                 "spin_kind": predicted.spin_kind,
01675 |             }
01676 |         )
01677 |     if reference.overlap is not None and reference.overlap.shape != reference.hamiltonian.shape:
01678 |         errors.append(
01679 |             {
01680 |                 "sample": sample,
01681 |                 "kind": "invalid_overlap_shape",
01682 |                 "error": (
01683 |                     "Reference overlap shape does not match reference Hamiltonian shape: "
01684 |                     f"{reference.overlap.shape} vs {reference.hamiltonian.shape}."
01685 |                 ),
01686 |                 "overlap_shape": list(reference.overlap.shape),
01687 |                 "reference_shape": list(reference.hamiltonian.shape),
01688 |             }
01689 |         )
01690 |     if not reference.orthogonal and reference.overlap is None:
01691 |         errors.append(
01692 |             {
01693 |                 "sample": sample,
01694 |                 "kind": "missing_required_overlap",
01695 |                 "error": (
01696 |                     "Reference Hamiltonian is non-orthogonal but no overlap matrix was readable; "
01697 |                     "generalized spectral/DOS metrics are invalid."
01698 |                 ),
01699 |                 "overlap_error": reference.overlap_error,
01700 |             }
01701 |         )
01702 |     return errors
```

### `matrix_compatibility_warnings` — líneas 1733–1771

```py
01733 | def matrix_compatibility_warnings(sample: str, reference: MatrixData, predicted: MatrixData) -> list[dict[str, Any]]:
01734 |     warnings: list[dict[str, Any]] = []
01735 |     if is_graph2mat_auxiliary_prediction(reference, predicted):
01736 |         warnings.append(
01737 |             {
01738 |                 "sample": sample,
01739 |                 "kind": "graph2mat_auxiliary_component_ignored",
01740 |                 "severity": "severe",
01741 |                 "error": (
01742 |                     "Graph2Mat wrote a non-orthogonal prediction container with two matrix components. "
01743 |                     "Metrics compare Hamiltonian component 0 only; the auxiliary predicted overlap/spin-like "
01744 |                     "component is ignored, and spectral metrics use the SIESTA reference overlap."
01745 |                 ),
01746 |                 "reference_components": reference.component_count,
01747 |                 "predicted_components": predicted.component_count,
01748 |                 "reference_spin": reference.spin_kind,
01749 |                 "predicted_spin": predicted.spin_kind,
01750 |             }
01751 |         )
01752 |     diagnostics = overlap_diagnostics(reference, predicted)
01753 |     value = diagnostics["prediction_overlap_relative_frobenius_vs_reference"]
01754 |     if isinstance(value, float) and math.isfinite(value) and value > OVERLAP_RELATIVE_FROBENIUS_WARNING_THRESHOLD:
01755 |         warnings.append(
01756 |             {
01757 |                 "sample": sample,
01758 |                 "kind": "prediction_overlap_mismatch",
01759 |                 "severity": "severe",
01760 |                 "error": (
01761 |                     "Prediction-owned overlap differs from the SIESTA reference overlap. "
01762 |                     "Spectral metrics use S_ref; the prediction HSX is not safe as a "
01763 |                     "standalone generalized-eigenproblem input."
01764 |                 ),
01765 |                 "prediction_overlap_relative_frobenius_vs_reference": value,
01766 |                 "overlap_source": diagnostics["overlap_source"],
01767 |                 "prediction_own_overlap_used": diagnostics["prediction_own_overlap_used"],
01768 |                 "prediction_self_contained_hsx_safe": diagnostics["prediction_self_contained_hsx_safe"],
01769 |             }
01770 |         )
01771 |     return warnings
```

### `overlap_diagnostics` — líneas 1785–1819

```py
01785 | def overlap_diagnostics(reference: MatrixData, predicted: MatrixData) -> dict[str, Any]:
01786 |     overlap_source = "siesta_reference" if reference.overlap is not None else "none_standard_eigenproblem"
01787 |     rel_diff = math.nan
01788 |     unavailable_reason = ""
01789 |     if reference.overlap is None:
01790 |         unavailable_reason = "reference_overlap_unavailable"
01791 |     elif predicted.overlap is None:
01792 |         unavailable_reason = "missing_prediction_overlap"
01793 |     elif predicted.overlap.shape != reference.overlap.shape:
01794 |         unavailable_reason = "prediction_overlap_shape_mismatch"
01795 |     else:
01796 |         rel_diff = relative_sparse_frobenius(predicted.overlap - reference.overlap, reference.overlap)
01797 |         if math.isfinite(rel_diff) and rel_diff > OVERLAP_RELATIVE_FROBENIUS_WARNING_THRESHOLD:
01798 |             unavailable_reason = "prediction_overlap_mismatch"
01799 | 
01800 |     auxiliary_ignored = is_graph2mat_auxiliary_prediction(reference, predicted)
01801 |     prediction_safe = True
01802 |     if reference.overlap is not None:
01803 |         prediction_safe = (
01804 |             predicted.overlap is not None
01805 |             and predicted.overlap.shape == reference.overlap.shape
01806 |             and math.isfinite(rel_diff)
01807 |             and rel_diff <= OVERLAP_RELATIVE_FROBENIUS_WARNING_THRESHOLD
01808 |             and not auxiliary_ignored
01809 |         )
01810 |     return {
01811 |         "overlap_source": overlap_source,
01812 |         "prediction_own_overlap_used": False,
01813 |         "prediction_overlap_relative_frobenius_vs_reference": rel_diff,
01814 |         "prediction_overlap_check_threshold": OVERLAP_RELATIVE_FROBENIUS_WARNING_THRESHOLD,
01815 |         "prediction_self_contained_hsx_safe": prediction_safe,
01816 |         "prediction_self_contained_hsx_unsafe_reason": (
01817 |             "graph2mat_auxiliary_component_ignored" if auxiliary_ignored else unavailable_reason
01818 |         ),
01819 |     }
```

### `matrix_semantics_fields` — líneas 1822–1839

```py
01822 | def matrix_semantics_fields(
01823 |     reference: MatrixData,
01824 |     predicted: MatrixData,
01825 |     *,
01826 |     target_component_policy: str,
01827 | ) -> dict[str, Any]:
01828 |     auxiliary_ignored = is_graph2mat_auxiliary_prediction(reference, predicted)
01829 |     return {
01830 |         "metrics_schema_version": METRICS_SCHEMA_VERSION,
01831 |         "metrics_provenance_generation": METRICS_PROVENANCE_GENERATION,
01832 |         "target_component_policy": target_component_policy,
01833 |         "reference_component_count": int(reference.component_count),
01834 |         "prediction_component_count": int(predicted.component_count),
01835 |         "reference_spin_kind": reference.spin_kind,
01836 |         "prediction_spin_kind": predicted.spin_kind,
01837 |         "graph2mat_auxiliary_component_ignored": auxiliary_ignored,
01838 |         **overlap_diagnostics(reference, predicted),
01839 |     }
```

### `component_channel_metrics` — líneas 1846–1932

```py
01846 | def component_channel_metrics(
01847 |     sample: str,
01848 |     reference: MatrixData,
01849 |     predicted: MatrixData,
01850 |     semantics: dict[str, Any],
01851 | ) -> list[dict[str, Any]]:
01852 |     reference_components = matrix_components(reference)
01853 |     prediction_components = matrix_components(predicted)
01854 |     rows: list[dict[str, Any]] = []
01855 |     for index in range(max(len(reference_components), len(prediction_components))):
01856 |         ref_available = index < len(reference_components)
01857 |         pred_available = index < len(prediction_components)
01858 |         channel_role = "hamiltonian" if index == 0 else "auxiliary"
01859 |         policy = str(semantics.get("target_component_policy") or "")
01860 |         official_h_channel = index == 0
01861 |         row: dict[str, Any] = {
01862 |             "sample": sample,
01863 |             "component_index": index,
01864 |             "component_role": channel_role,
01865 |             "component_target_label": "H" if official_h_channel else "auxiliary_non_target",
01866 |             "component_units": "eV" if official_h_channel else "auxiliary_or_dimensionless",
01867 |             "component_is_official_hamiltonian_target": official_h_channel,
01868 |             "component_in_official_h_only_loss": policy == "h_only" and official_h_channel,
01869 |             "component_in_official_sparse_h_metric": official_h_channel,
01870 |             "component_channel_warning": (
01871 |                 ""
01872 |                 if official_h_channel
01873 |                 else "Auxiliary/non-target channel is reported separately and is not mixed into official H metrics."
01874 |             ),
01875 |             "reference_component_available": ref_available,
01876 |             "prediction_component_available": pred_available,
01877 |             **semantics,
01878 |         }
01879 |         if ref_available and pred_available:
01880 |             ref_matrix = reference_components[index]
01881 |             pred_matrix = prediction_components[index]
01882 |             if ref_matrix.shape == pred_matrix.shape:
01883 |                 ref_values = csr_value_dict(ref_matrix, SUPPORT_THRESHOLD)
01884 |                 pred_values = csr_value_dict(pred_matrix, SUPPORT_THRESHOLD)
01885 |                 union = sorted(set(ref_values) | set(pred_values))
01886 |                 deltas = [pred_values.get(key, 0.0) - ref_values.get(key, 0.0) for key in union]
01887 |                 row.update(
01888 |                     {
01889 |                         "component_shape": list(ref_matrix.shape),
01890 |                         "component_mae_eV": mean_abs(deltas),
01891 |                         "component_rmse_eV": rmse(deltas),
01892 |                         "component_mse_eV2": mse(deltas),
01893 |                         "component_max_abs_error_eV": float(
01894 |                             max((abs(value) for value in deltas), default=math.nan)
01895 |                         ),
01896 |                         "component_n_entries": len(union),
01897 |                         "component_metric_available": True,
01898 |                         "component_unavailable_reason": "",
01899 |                     }
01900 |                 )
01901 |             else:
01902 |                 row.update(
01903 |                     {
01904 |                         "component_shape": None,
01905 |                         "component_mae_eV": math.nan,
01906 |                         "component_rmse_eV": math.nan,
01907 |                         "component_mse_eV2": math.nan,
01908 |                         "component_max_abs_error_eV": math.nan,
01909 |                         "component_n_entries": 0,
01910 |                         "component_metric_available": False,
01911 |                         "component_unavailable_reason": "component_shape_mismatch",
01912 |                     }
01913 |                 )
01914 |         else:
01915 |             row.update(
01916 |                 {
01917 |                     "component_shape": None,
01918 |                     "component_mae_eV": math.nan,
01919 |                     "component_rmse_eV": math.nan,
01920 |                     "component_mse_eV2": math.nan,
01921 |                     "component_max_abs_error_eV": math.nan,
01922 |                     "component_n_entries": 0,
01923 |                     "component_metric_available": False,
01924 |                     "component_unavailable_reason": (
01925 |                         "missing_reference_component"
01926 |                         if pred_available
01927 |                         else "missing_prediction_component"
01928 |                     ),
01929 |                 }
01930 |             )
01931 |         rows.append(row)
01932 |     return rows
```

### `hermiticity_defect` — líneas 1935–1939

```py
01935 | def hermiticity_defect(matrix: sparse.csr_matrix) -> float:
01936 |     denominator = sparse_norm(matrix)
01937 |     if denominator == 0:
01938 |         return math.nan
01939 |     return sparse_norm(matrix - matrix.getH()) / denominator
```

### `sparse_metrics` — líneas 2002–2096

```py
02002 | def sparse_metrics(sample: str, reference: MatrixData, predicted: MatrixData) -> dict[str, Any]:
02003 |     ref_values = csr_value_dict(reference.hamiltonian, SUPPORT_THRESHOLD)
02004 |     pred_values = csr_value_dict(predicted.hamiltonian, SUPPORT_THRESHOLD)
02005 |     ref_support = set(ref_values)
02006 |     pred_support = set(pred_values)
02007 |     union_support = ref_support | pred_support
02008 |     intersection = ref_support & pred_support
02009 |     ref_indices = sorted(ref_support)
02010 |     pred_indices = sorted(pred_support)
02011 |     union_indices = sorted(union_support)
02012 | 
02013 |     false_zeros = ref_support - pred_support
02014 |     false_nonzeros = pred_support - ref_support
02015 |     deltas_ref = [pred_values.get(index, 0.0) - ref_values[index] for index in ref_indices]
02016 |     deltas_pred = [pred_values[index] - ref_values.get(index, 0.0) for index in pred_indices]
02017 |     deltas_union = [
02018 |         pred_values.get(index, 0.0) - ref_values.get(index, 0.0)
02019 |         for index in union_indices
02020 |     ]
02021 |     ref_targets_ref = [ref_values[index] for index in ref_indices]
02022 |     pred_targets_ref = [pred_values.get(index, 0.0) for index in ref_indices]
02023 |     ref_targets_pred = [ref_values.get(index, 0.0) for index in pred_indices]
02024 |     pred_targets_pred = [pred_values[index] for index in pred_indices]
02025 |     ref_targets_union = [ref_values.get(index, 0.0) for index in union_indices]
02026 |     pred_targets_union = [pred_values.get(index, 0.0) for index in union_indices]
02027 |     mae_ref = mean_abs(deltas_ref)
02028 |     rmse_ref = rmse(deltas_ref)
02029 |     mae_pred = mean_abs(deltas_pred)
02030 |     rmse_pred = rmse(deltas_pred)
02031 |     mae_union = mean_abs(deltas_union)
02032 |     rmse_union = rmse(deltas_union)
02033 |     ref_fro = float(np.sqrt(sum(abs(value) ** 2 for value in ref_values.values())))
02034 |     ref_pattern_fro = float(np.sqrt(sum(abs(value) ** 2 for value in deltas_ref)))
02035 |     union_fro = float(np.sqrt(sum(abs(value) ** 2 for value in deltas_union)))
02036 |     ref_l1 = float(sum(abs(value) for value in ref_values.values()))
02037 |     union_l1 = float(sum(abs(value) for value in deltas_union))
02038 |     precision = len(intersection) / len(pred_support) if pred_support else math.nan
02039 |     recall = len(intersection) / len(ref_support) if ref_support else math.nan
02040 |     f1 = (
02041 |         2.0 * precision * recall / (precision + recall)
02042 |         if precision == precision and recall == recall and (precision + recall) > 0
02043 |         else math.nan
02044 |     )
02045 |     n_entries = reference.hamiltonian.shape[0] * reference.hamiltonian.shape[1]
02046 |     return {
02047 |         "sample": sample,
02048 |         "n_orbitals": reference.hamiltonian.shape[0],
02049 |         "n_entries": n_entries,
02050 |         "ref_nnz": len(ref_support),
02051 |         "pred_nnz": len(pred_support),
02052 |         "union_nnz": len(union_support),
02053 |         "ref_density": len(ref_support) / n_entries if n_entries else math.nan,
02054 |         "pred_density": len(pred_support) / n_entries if n_entries else math.nan,
02055 |         "matrix_metric_target_space": MATRIX_METRIC_TARGET_SPACE,
02056 |         "h_matrix_metric_independent_of_training_loss": True,
02057 |         "h_matrix_component_index": 0,
02058 |         "h_matrix_target_label": "H",
02059 |         "mae_ref_eV": mae_ref,
02060 |         "rmse_ref_eV": rmse_ref,
02061 |         "mse_ref_eV2": mse(deltas_ref),
02062 |         "r2_ref": r2_score(ref_targets_ref, pred_targets_ref),
02063 |         "mae_ref_meV": ev_to_mev(mae_ref),
02064 |         "rmse_ref_meV": ev_to_mev(rmse_ref),
02065 |         "mae_pred_eV": mae_pred,
02066 |         "rmse_pred_eV": rmse_pred,
02067 |         "mse_pred_eV2": mse(deltas_pred),
02068 |         "r2_pred": r2_score(ref_targets_pred, pred_targets_pred),
02069 |         "mae_pred_meV": ev_to_mev(mae_pred),
02070 |         "rmse_pred_meV": ev_to_mev(rmse_pred),
02071 |         "mae_union_eV": mae_union,
02072 |         "rmse_union_eV": rmse_union,
02073 |         "h_matrix_mae_eV": mae_union,
02074 |         "h_matrix_rmse_eV": rmse_union,
02075 |         "h_matrix_mae_meV": ev_to_mev(mae_union),
02076 |         "h_matrix_rmse_meV": ev_to_mev(rmse_union),
02077 |         "mse_union_eV2": mse(deltas_union),
02078 |         "r2_union": r2_score(ref_targets_union, pred_targets_union),
02079 |         "mae_union_meV": ev_to_mev(mae_union),
02080 |         "rmse_union_meV": ev_to_mev(rmse_union),
02081 |         "max_abs_error_union_eV": float(max((abs(value) for value in deltas_union), default=math.nan)),
02082 |         "relative_frobenius_ref": ref_pattern_fro / ref_fro if ref_fro else math.nan,
02083 |         "relative_frobenius_union": union_fro / ref_fro if ref_fro else math.nan,
02084 |         "relative_l1_union": union_l1 / ref_l1 if ref_l1 else math.nan,
02085 |         "support_precision": precision,
02086 |         "support_recall": recall,
02087 |         "support_f1": f1,
02088 |         "false_zeros": len(false_zeros),
02089 |         "false_nonzeros": len(false_nonzeros),
02090 |         "false_zero_rate": len(false_zeros) / len(ref_support) if ref_support else math.nan,
02091 |         "false_nonzero_rate": len(false_nonzeros) / len(pred_support) if pred_support else math.nan,
02092 |         "weighted_false_zeros_eV": float(sum(abs(ref_values[index]) for index in false_zeros)),
02093 |         "weighted_false_nonzeros_eV": float(sum(abs(pred_values[index]) for index in false_nonzeros)),
02094 |         "hermiticity_ref": hermiticity_defect(reference.hamiltonian),
02095 |         "hermiticity_pred": hermiticity_defect(predicted.hamiltonian),
02096 |     }
```

### `complex_matrix_error_metrics` — líneas 2524–2540

```py
02524 | def complex_matrix_error_metrics(reference: Any, predicted: Any) -> dict[str, Any]:
02525 |     ref = dense_matrix_array(reference)
02526 |     pred = dense_matrix_array(predicted)
02527 |     if ref.shape != pred.shape:
02528 |         raise ValueError(f"Matrix shapes differ: {ref.shape} vs {pred.shape}.")
02529 |     delta = pred - ref
02530 |     abs_delta = np.abs(delta)
02531 |     return {
02532 |         "n_entries": int(delta.size),
02533 |         "mae_eV": float(np.mean(abs_delta)) if delta.size else math.nan,
02534 |         "rmse_eV": float(np.sqrt(np.mean(abs_delta**2))) if delta.size else math.nan,
02535 |         "mse_eV2": float(np.mean(abs_delta**2)) if delta.size else math.nan,
02536 |         "max_abs_error_eV": float(np.max(abs_delta)) if delta.size else math.nan,
02537 |         "relative_frobenius": complex_relative_frobenius(delta, ref),
02538 |         "reference_hermiticity": complex_hermiticity_defect(ref),
02539 |         "prediction_hermiticity": complex_hermiticity_defect(pred),
02540 |     }
```

### `kpoint_hamiltonian_matrix` — líneas 2543–2547

```py
02543 | def kpoint_hamiltonian_matrix(hamiltonian_obj: Any, kpoint: tuple[float, float, float] | list[float]) -> np.ndarray:
02544 |     h_k = getattr(hamiltonian_obj, "Hk", None)
02545 |     if not callable(h_k):
02546 |         raise RuntimeError("Hamiltonian object does not expose Hk(k); cannot construct H(k).")
02547 |     return dense_matrix_array(h_k(kpoint, format="array"))
```

### `kpoint_overlap_matrix` — líneas 2550–2559

```py
02550 | def kpoint_overlap_matrix(hamiltonian_obj: Any, kpoint: tuple[float, float, float] | list[float]) -> np.ndarray | None:
02551 |     if bool(getattr(hamiltonian_obj, "orthogonal", False)):
02552 |         return None
02553 |     s_k = getattr(hamiltonian_obj, "Sk", None)
02554 |     if not callable(s_k):
02555 |         raise RuntimeError("Non-orthogonal reference requires S(k), but the object does not expose Sk(k).")
02556 |     overlap = dense_matrix_array(s_k(kpoint, format="array"))
02557 |     if overlap.size == 0:
02558 |         raise RuntimeError("Non-orthogonal reference returned an empty S(k) matrix.")
02559 |     return overlap
```

### `complex_generalized_eigenvalues` — líneas 2562–2582

```py
02562 | def complex_generalized_eigenvalues(hamiltonian: Any, overlap: Any | None = None) -> np.ndarray:
02563 |     dense_h = symmetrized_hermitian_dense(hamiltonian)
02564 |     if EIGENSOLVER_DEVICE.startswith("cuda"):
02565 |         cp = _CUPY
02566 |         if cp is None:
02567 |             raise RuntimeError("CUDA eigensolver is not configured.")
02568 |         gpu_h = cp.asarray(dense_h)
02569 |         if overlap is None:
02570 |             return cp.asnumpy(cp.linalg.eigvalsh(gpu_h)).astype(float, copy=False)
02571 |         gpu_s = cp.asarray(symmetrized_hermitian_dense(overlap))
02572 |         factor = cp.linalg.cholesky(gpu_s)
02573 |         reduced = cp.linalg.solve(factor.conj(), cp.linalg.solve(factor, gpu_h).T).T
02574 |         reduced = (reduced + reduced.conj().T) * 0.5
02575 |         return cp.asnumpy(cp.linalg.eigvalsh(reduced)).astype(float, copy=False)
02576 |     if overlap is None:
02577 |         return np.asarray(np.linalg.eigvalsh(dense_h), dtype=float)
02578 |     dense_s = symmetrized_hermitian_dense(overlap)
02579 |     return np.asarray(
02580 |         scipy.linalg.eigh(dense_h, dense_s, eigvals_only=True, check_finite=False),
02581 |         dtype=float,
02582 |     )
```

### `kpoint_eigenvalues_with_reference_overlap` — líneas 2585–2592

```py
02585 | def kpoint_eigenvalues_with_reference_overlap(
02586 |     hamiltonian_obj: Any,
02587 |     reference_hamiltonian_obj: Any,
02588 |     kpoint: tuple[float, float, float] | list[float],
02589 | ) -> np.ndarray:
02590 |     h_k = kpoint_hamiltonian_matrix(hamiltonian_obj, kpoint)
02591 |     s_ref_k = kpoint_overlap_matrix(reference_hamiltonian_obj, kpoint)
02592 |     return complex_generalized_eigenvalues(h_k, s_ref_k)
```

### `generalized_eigenvalues` — líneas 2595–2599

```py
02595 | def generalized_eigenvalues(
02596 |     hamiltonian: sparse.csr_matrix,
02597 |     overlap: sparse.csr_matrix | None,
02598 | ) -> np.ndarray:
02599 |     return complex_generalized_eigenvalues(hamiltonian, overlap)
```

### `low_energy_metrics` — líneas 2634–2701

```py
02634 | def low_energy_metrics(
02635 |     reference: MatrixData,
02636 |     predicted: MatrixData,
02637 |     *,
02638 |     n_states: int = LOW_ENERGY_N_STATES,
02639 |     alignment: str = LOW_ENERGY_ALIGNMENT,
02640 | ) -> dict[str, Any]:
02641 |     n_states, alignment = validate_low_energy_config(n_states, alignment)
02642 |     overlap_required = not bool(reference.orthogonal)
02643 |     overlap = reference.overlap
02644 |     overlap_used = overlap is not None
02645 |     metadata = {
02646 |         "low_energy_requested_states": n_states,
02647 |         "low_energy_alignment": alignment,
02648 |         "low_energy_overlap_used": overlap_used,
02649 |         "low_energy_overlap_required": overlap_required,
02650 |         "low_energy_solver": eigensolver_name(generalized=overlap_used),
02651 |         "low_energy_warning": "",
02652 |     }
02653 |     ref_eig, ref_warning = low_energy_eigenvalues(
02654 |         reference.hamiltonian,
02655 |         overlap,
02656 |         overlap_required=overlap_required,
02657 |     )
02658 |     pred_eig, pred_warning = low_energy_eigenvalues(
02659 |         predicted.hamiltonian,
02660 |         overlap,
02661 |         overlap_required=overlap_required,
02662 |     )
02663 |     warning = ref_warning or pred_warning
02664 |     if warning:
02665 |         return {
02666 |             **metadata,
02667 |             "low_energy_n_states": None,
02668 |             "low_energy_mae_eV": math.nan,
02669 |             "low_energy_rmse_eV": math.nan,
02670 |             "low_energy_max_abs_error_eV": math.nan,
02671 |             "low_energy_aligned_rmse_eV": math.nan,
02672 |             "low_energy_warning": warning,
02673 |         }
02674 |     assert ref_eig is not None and pred_eig is not None
02675 |     count = min(n_states, ref_eig.size, pred_eig.size)
02676 |     if count <= 0:
02677 |         return {
02678 |             **metadata,
02679 |             "low_energy_n_states": None,
02680 |             "low_energy_mae_eV": math.nan,
02681 |             "low_energy_rmse_eV": math.nan,
02682 |             "low_energy_max_abs_error_eV": math.nan,
02683 |             "low_energy_aligned_rmse_eV": math.nan,
02684 |             "low_energy_warning": "low-energy eigenvalues unavailable: no common states to compare.",
02685 |         }
02686 |     ref_low = ref_eig[:count]
02687 |     pred_low = pred_eig[:count]
02688 |     delta = pred_low - ref_low
02689 |     aligned_rmse = math.nan
02690 |     if alignment == "global_shift":
02691 |         shift = float(np.mean(ref_low - pred_low))
02692 |         aligned_delta = (pred_low + shift) - ref_low
02693 |         aligned_rmse = float(np.sqrt(np.mean(aligned_delta**2)))
02694 |     return {
02695 |         **metadata,
02696 |         "low_energy_n_states": int(count),
02697 |         "low_energy_mae_eV": float(np.mean(np.abs(delta))),
02698 |         "low_energy_rmse_eV": float(np.sqrt(np.mean(delta**2))),
02699 |         "low_energy_max_abs_error_eV": float(np.max(np.abs(delta))),
02700 |         "low_energy_aligned_rmse_eV": aligned_rmse,
02701 |     }
```

### `eigen_error_metrics` — líneas 2721–2834

```py
02721 | def eigen_error_metrics(
02722 |     reference: np.ndarray,
02723 |     predicted: np.ndarray,
02724 |     fermi_level: float | None,
02725 |     fermi_level_source: str,
02726 | ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
02727 |     n_bands = min(reference.size, predicted.size)
02728 |     reference = reference[:n_bands]
02729 |     predicted = predicted[:n_bands]
02730 |     errors = predicted - reference
02731 |     band_rows = [
02732 |         {
02733 |             "band": index,
02734 |             "siesta_eV": float(reference[index]),
02735 |             "predicted_eV": float(predicted[index]),
02736 |             "error_eV": float(errors[index]),
02737 |             "abs_error_eV": float(abs(errors[index])),
02738 |             "siesta_minus_fermi_eV": None
02739 |             if fermi_level is None
02740 |             else float(reference[index] - fermi_level),
02741 |         }
02742 |         for index in range(n_bands)
02743 |     ]
02744 | 
02745 |     occupied_mask = np.zeros(n_bands, dtype=bool)
02746 |     fermi_mask = np.zeros(n_bands, dtype=bool)
02747 |     frontier_mask = np.zeros(n_bands, dtype=bool)
02748 |     homo_index = None
02749 |     lumo_index = None
02750 |     if fermi_level is not None:
02751 |         occupied_mask = reference <= fermi_level
02752 |         fermi_mask = np.abs(reference - fermi_level) <= FERMI_WINDOW_EV
02753 |         occ_indices = np.where(occupied_mask)[0]
02754 |         virt_indices = np.where(~occupied_mask)[0]
02755 |         if occ_indices.size and virt_indices.size:
02756 |             homo_index = int(occ_indices[-1])
02757 |             lumo_index = int(virt_indices[0])
02758 |     if homo_index is not None:
02759 |         frontier_mask[homo_index] = True
02760 |     if lumo_index is not None:
02761 |         frontier_mask[lumo_index] = True
02762 | 
02763 |     def masked_mae(mask: np.ndarray) -> float:
02764 |         return float(np.mean(np.abs(errors[mask]))) if np.any(mask) else math.nan
02765 | 
02766 |     def masked_rmse(mask: np.ndarray) -> float:
02767 |         return float(np.sqrt(np.mean(errors[mask] ** 2))) if np.any(mask) else math.nan
02768 | 
02769 |     gap_ref = band_gap(reference, fermi_level)
02770 |     gap_pred = band_gap(predicted, fermi_level)
02771 |     if (gap_ref != gap_ref or gap_pred != gap_pred) and homo_index is not None and lumo_index is not None:
02772 |         gap_ref = float(reference[lumo_index] - reference[homo_index])
02773 |         gap_pred = float(predicted[lumo_index] - predicted[homo_index])
02774 | 
02775 |     def aligned_errors(mask: np.ndarray | None = None) -> tuple[float, float, float]:
02776 |         if n_bands == 0:
02777 |             return (math.nan, math.nan, math.nan)
02778 |         use_ref = reference if mask is None else reference[mask]
02779 |         use_pred = predicted if mask is None else predicted[mask]
02780 |         if use_ref.size == 0:
02781 |             return (math.nan, math.nan, math.nan)
02782 |         shift = float(np.mean(use_ref - use_pred))
02783 |         delta = (use_pred + shift) - use_ref
02784 |         return shift, float(np.mean(np.abs(delta))), float(np.sqrt(np.mean(delta**2)))
02785 | 
02786 |     global_shift, global_aligned_mae, global_aligned_rmse = aligned_errors(None)
02787 |     fermi_shift, fermi_aligned_mae, fermi_aligned_rmse = aligned_errors(fermi_mask)
02788 |     homo_shift, homo_aligned_mae, homo_aligned_rmse = (
02789 |         aligned_errors(np.array([i == homo_index for i in range(n_bands)], dtype=bool))
02790 |         if homo_index is not None
02791 |         else (math.nan, math.nan, math.nan)
02792 |     )
02793 |     metrics = {
02794 |         "n_compared_bands": n_bands,
02795 |         "fermi_ref_eV": fermi_level,
02796 |         "fermi_level_source": fermi_level_source,
02797 |         "fermi_metric_available": fermi_level is not None and math.isfinite(float(fermi_level)),
02798 |         "global_mae_eV": float(np.mean(np.abs(errors))) if n_bands else math.nan,
02799 |         "global_rmse_eV": float(np.sqrt(np.mean(errors**2))) if n_bands else math.nan,
02800 |         "global_max_abs_error_eV": float(np.max(np.abs(errors))) if n_bands else math.nan,
02801 |         "global_mean_signed_error_eV": float(np.mean(errors)) if n_bands else math.nan,
02802 |         "occupied_bands": int(np.count_nonzero(occupied_mask)),
02803 |         "occupied_metric_available": bool(np.any(occupied_mask)),
02804 |         "occupied_mae_eV": masked_mae(occupied_mask),
02805 |         "occupied_rmse_eV": masked_rmse(occupied_mask),
02806 |         "fermi_window_eV": FERMI_WINDOW_EV,
02807 |         "fermi_window_bands": int(np.count_nonzero(fermi_mask)),
02808 |         "fermi_window_metric_available": bool(np.any(fermi_mask)),
02809 |         "fermi_window_mae_eV": masked_mae(fermi_mask),
02810 |         "fermi_window_rmse_eV": masked_rmse(fermi_mask),
02811 |         "homo_index": homo_index,
02812 |         "lumo_index": lumo_index,
02813 |         "homo_error_eV": float(errors[homo_index]) if homo_index is not None else math.nan,
02814 |         "lumo_error_eV": float(errors[lumo_index]) if lumo_index is not None else math.nan,
02815 |         "frontier_window_bands": int(np.count_nonzero(frontier_mask)),
02816 |         "frontier_metric_available": bool(np.any(frontier_mask)),
02817 |         "frontier_window_mae_eV": masked_mae(frontier_mask),
02818 |         "frontier_window_rmse_eV": masked_rmse(frontier_mask),
02819 |         "align_global_shift_eV": global_shift,
02820 |         "align_global_mae_eV": global_aligned_mae,
02821 |         "align_global_rmse_eV": global_aligned_rmse,
02822 |         "align_fermi_shift_eV": fermi_shift,
02823 |         "align_fermi_mae_eV": fermi_aligned_mae,
02824 |         "align_fermi_rmse_eV": fermi_aligned_rmse,
02825 |         "align_homo_shift_eV": homo_shift,
02826 |         "align_homo_mae_eV": homo_aligned_mae,
02827 |         "align_homo_rmse_eV": homo_aligned_rmse,
02828 |         "gap_ref_eV": gap_ref,
02829 |         "gap_pred_eV": gap_pred,
02830 |         "gap_abs_error_eV": abs(gap_pred - gap_ref)
02831 |         if gap_ref == gap_ref and gap_pred == gap_pred
02832 |         else math.nan,
02833 |     }
02834 |     return band_rows, metrics
```

### `dos_for_sample` — líneas 2859–2897

```py
02859 | def dos_for_sample(reference: np.ndarray, predicted: np.ndarray, sigma_ev: float = DOS_SIGMA_EV) -> tuple[list[dict[str, Any]], dict[str, Any]]:
02860 |     combined = np.concatenate([reference, predicted])
02861 |     if combined.size == 0:
02862 |         return [], {
02863 |             "dos_wasserstein_eV": math.nan,
02864 |             "dos_l1": math.nan,
02865 |             "dos_l2": math.nan,
02866 |             "energy_min_eV": math.nan,
02867 |             "energy_max_eV": math.nan,
02868 |         }
02869 |     margin = max(5.0 * sigma_ev, 0.05 * float(np.ptp(combined) if combined.size > 1 else 1.0))
02870 |     energy_min = float(np.min(combined) - margin)
02871 |     energy_max = float(np.max(combined) + margin)
02872 |     grid = np.linspace(energy_min, energy_max, DOS_POINTS)
02873 |     dx = float(grid[1] - grid[0]) if grid.size > 1 else 1.0
02874 |     ref_dos = gaussian_dos(reference, grid, sigma_ev)
02875 |     pred_dos = gaussian_dos(predicted, grid, sigma_ev)
02876 |     ref_norm = normalized_density(ref_dos, dx)
02877 |     pred_norm = normalized_density(pred_dos, dx)
02878 |     rows = [
02879 |         {
02880 |             "energy_eV": float(grid[index]),
02881 |             "siesta_dos": float(ref_dos[index]),
02882 |             "predicted_dos": float(pred_dos[index]),
02883 |             "siesta_dos_normalized": float(ref_norm[index]),
02884 |             "predicted_dos_normalized": float(pred_norm[index]),
02885 |         }
02886 |         for index in range(grid.size)
02887 |     ]
02888 |     metrics = {
02889 |         "dos_sigma_eV": sigma_ev,
02890 |         "dos_grid_points": DOS_POINTS,
02891 |         "energy_min_eV": energy_min,
02892 |         "energy_max_eV": energy_max,
02893 |         "dos_wasserstein_eV": wasserstein_from_grid(ref_dos, pred_dos, dx),
02894 |         "dos_l1": float(np.sum(np.abs(ref_norm - pred_norm)) * dx),
02895 |         "dos_l2": float(np.sqrt(np.sum((ref_norm - pred_norm) ** 2) * dx)),
02896 |     }
02897 |     return rows, metrics
```

### `dos_fermi_window_metrics` — líneas 2900–2954

```py
02900 | def dos_fermi_window_metrics(
02901 |     reference: np.ndarray,
02902 |     predicted: np.ndarray,
02903 |     fermi_level: float | None,
02904 |     sigma_ev: float = DOS_SIGMA_EV,
02905 | ) -> tuple[np.ndarray, dict[str, Any]]:
02906 |     metrics: dict[str, Any] = {
02907 |         "dos_window_min_eV": DOS_FERMI_WINDOW_MIN_EV,
02908 |         "dos_window_max_eV": DOS_FERMI_WINDOW_MAX_EV,
02909 |         "dos_window_points": DOS_FERMI_WINDOW_POINTS,
02910 |         "dos_window_sigma_eV": sigma_ev,
02911 |         "dos_window_alignment": DOS_FERMI_WINDOW_ALIGNMENT,
02912 |     }
02913 |     try:
02914 |         fermi_value = float(fermi_level)
02915 |     except (TypeError, ValueError):
02916 |         metrics.update(
02917 |             {
02918 |                 "dos_mae_500_fermi_window": math.nan,
02919 |                 "dos_window_metric_available": False,
02920 |                 "dos_window_unavailable_reason": "missing_fermi_level",
02921 |             }
02922 |         )
02923 |         return np.asarray([], dtype=float), metrics
02924 |     if not math.isfinite(fermi_value):
02925 |         metrics.update(
02926 |             {
02927 |                 "dos_mae_500_fermi_window": math.nan,
02928 |                 "dos_window_metric_available": False,
02929 |                 "dos_window_unavailable_reason": "missing_fermi_level",
02930 |             }
02931 |         )
02932 |         return np.asarray([], dtype=float), metrics
02933 |     if np.concatenate([reference, predicted]).size == 0:
02934 |         metrics.update(
02935 |             {
02936 |                 "dos_mae_500_fermi_window": math.nan,
02937 |                 "dos_window_metric_available": False,
02938 |                 "dos_window_unavailable_reason": "missing_eigenvalues",
02939 |             }
02940 |         )
02941 |         return np.asarray([], dtype=float), metrics
02942 | 
02943 |     relative_grid = np.linspace(DOS_FERMI_WINDOW_MIN_EV, DOS_FERMI_WINDOW_MAX_EV, DOS_FERMI_WINDOW_POINTS)
02944 |     grid = fermi_value + relative_grid
02945 |     reference_dos = gaussian_dos(reference, grid, sigma_ev)
02946 |     predicted_dos = gaussian_dos(predicted, grid, sigma_ev)
02947 |     metrics.update(
02948 |         {
02949 |             "dos_mae_500_fermi_window": mean_abs((predicted_dos - reference_dos).tolist()),
02950 |             "dos_window_metric_available": True,
02951 |             "dos_window_unavailable_reason": "",
02952 |         }
02953 |     )
02954 |     return grid, metrics
```

### `kpoint_weighted_dos_metrics` — líneas 3045–3118

```py
03045 | def kpoint_weighted_dos_metrics(
03046 |     reference: np.ndarray,
03047 |     predicted: np.ndarray,
03048 |     weights: np.ndarray,
03049 |     fermi_level: float | None,
03050 |     sigma_ev: float = DOS_SIGMA_EV,
03051 | ) -> dict[str, Any]:
03052 |     metrics: dict[str, Any] = {
03053 |         "dos_sigma_eV": sigma_ev,
03054 |         "dos_grid_points": DOS_POINTS,
03055 |         "dos_window_min_eV": DOS_FERMI_WINDOW_MIN_EV,
03056 |         "dos_window_max_eV": DOS_FERMI_WINDOW_MAX_EV,
03057 |         "dos_window_points": DOS_FERMI_WINDOW_POINTS,
03058 |         "dos_window_sigma_eV": sigma_ev,
03059 |         "dos_window_alignment": DOS_FERMI_WINDOW_ALIGNMENT,
03060 |     }
03061 |     combined = np.concatenate([reference, predicted])
03062 |     if combined.size == 0:
03063 |         metrics.update(
03064 |             {
03065 |                 "energy_min_eV": math.nan,
03066 |                 "energy_max_eV": math.nan,
03067 |                 "dos_wasserstein_eV": math.nan,
03068 |                 "dos_l1": math.nan,
03069 |                 "dos_l2": math.nan,
03070 |                 "dos_mae_500_fermi_window": math.nan,
03071 |                 "dos_window_metric_available": False,
03072 |                 "dos_window_unavailable_reason": "missing_eigenvalues",
03073 |             }
03074 |         )
03075 |         return metrics
03076 |     margin = max(5.0 * sigma_ev, 0.05 * float(np.ptp(combined) if combined.size > 1 else 1.0))
03077 |     energy_min = float(np.min(combined) - margin)
03078 |     energy_max = float(np.max(combined) + margin)
03079 |     grid = np.linspace(energy_min, energy_max, DOS_POINTS)
03080 |     dx = float(grid[1] - grid[0]) if grid.size > 1 else 1.0
03081 |     ref_dos = gaussian_dos_weighted(reference, weights, grid, sigma_ev)
03082 |     pred_dos = gaussian_dos_weighted(predicted, weights, grid, sigma_ev)
03083 |     ref_norm = normalized_density(ref_dos, dx)
03084 |     pred_norm = normalized_density(pred_dos, dx)
03085 |     metrics.update(
03086 |         {
03087 |             "energy_min_eV": energy_min,
03088 |             "energy_max_eV": energy_max,
03089 |             "dos_wasserstein_eV": wasserstein_from_grid(ref_dos, pred_dos, dx),
03090 |             "dos_l1": float(np.sum(np.abs(ref_norm - pred_norm)) * dx),
03091 |             "dos_l2": float(np.sqrt(np.sum((ref_norm - pred_norm) ** 2) * dx)),
03092 |         }
03093 |     )
03094 |     try:
03095 |         fermi_value = float(fermi_level)
03096 |     except (TypeError, ValueError):
03097 |         fermi_value = math.nan
03098 |     if not math.isfinite(fermi_value):
03099 |         metrics.update(
03100 |             {
03101 |                 "dos_mae_500_fermi_window": math.nan,
03102 |                 "dos_window_metric_available": False,
03103 |                 "dos_window_unavailable_reason": "missing_fermi_level",
03104 |             }
03105 |         )
03106 |         return metrics
03107 |     relative_grid = np.linspace(DOS_FERMI_WINDOW_MIN_EV, DOS_FERMI_WINDOW_MAX_EV, DOS_FERMI_WINDOW_POINTS)
03108 |     window_grid = fermi_value + relative_grid
03109 |     ref_window = gaussian_dos_weighted(reference, weights, window_grid, sigma_ev)
03110 |     pred_window = gaussian_dos_weighted(predicted, weights, window_grid, sigma_ev)
03111 |     metrics.update(
03112 |         {
03113 |             "dos_mae_500_fermi_window": mean_abs((pred_window - ref_window).tolist()),
03114 |             "dos_window_metric_available": True,
03115 |             "dos_window_unavailable_reason": "",
03116 |         }
03117 |     )
03118 |     return metrics
```

### `prediction_artifact_safety_summary` — líneas 3187–3246

```py
03187 | def prediction_artifact_safety_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
03188 |     """Summarize whether predicted HSX artifacts are standalone-safe."""
03189 | 
03190 |     sample_rows: dict[str, dict[str, Any]] = {}
03191 |     for row in rows:
03192 |         sample = str(row.get("sample") or "")
03193 |         if not sample or "prediction_self_contained_hsx_safe" not in row:
03194 |             continue
03195 |         sample_rows.setdefault(sample, row)
03196 | 
03197 |     safe_samples: set[str] = set()
03198 |     unsafe_samples: set[str] = set()
03199 |     auxiliary_samples: set[str] = set()
03200 |     overlap_mismatch_samples: set[str] = set()
03201 |     unsafe_reasons: dict[str, int] = {}
03202 |     for sample, row in sample_rows.items():
03203 |         safe = _semantic_bool(row.get("prediction_self_contained_hsx_safe"))
03204 |         reason = str(row.get("prediction_self_contained_hsx_unsafe_reason") or "").strip()
03205 |         if safe:
03206 |             safe_samples.add(sample)
03207 |         else:
03208 |             unsafe_samples.add(sample)
03209 |             unsafe_reasons[reason or "unspecified"] = unsafe_reasons.get(reason or "unspecified", 0) + 1
03210 |         if _semantic_bool(row.get("graph2mat_auxiliary_component_ignored")):
03211 |             auxiliary_samples.add(sample)
03212 |         try:
03213 |             overlap_rel = float(row.get("prediction_overlap_relative_frobenius_vs_reference"))
03214 |         except (TypeError, ValueError):
03215 |             overlap_rel = math.nan
03216 |         if math.isfinite(overlap_rel) and overlap_rel > OVERLAP_RELATIVE_FROBENIUS_WARNING_THRESHOLD:
03217 |             overlap_mismatch_samples.add(sample)
03218 | 
03219 |     unsafe_reason = ""
03220 |     if unsafe_samples:
03221 |         unsafe_reason = (
03222 |             "ML_prediction.HSX is not a validated standalone generalized-eigenproblem input for "
03223 |             "all compared samples; official spectra use the SIESTA reference overlap."
03224 |         )
03225 |     return {
03226 |         "official_spectral_overlap_policy": "use_siesta_reference_overlap_for_nonorthogonal_predictions",
03227 |         "overlap_source_for_official_spectra": "siesta_reference_when_available",
03228 |         "prediction_own_overlap_used_for_spectra": False,
03229 |         "prediction_overlap_validation_tolerance": OVERLAP_RELATIVE_FROBENIUS_WARNING_THRESHOLD,
03230 |         "samples_with_prediction_semantics": len(sample_rows),
03231 |         "prediction_self_contained_hsx_safe_samples": len(safe_samples),
03232 |         "prediction_self_contained_hsx_unsafe_samples": len(unsafe_samples),
03233 |         "prediction_self_contained_hsx_unsafe_reasons": unsafe_reasons,
03234 |         "graph2mat_auxiliary_component_ignored_samples": len(auxiliary_samples),
03235 |         "prediction_overlap_mismatch_samples": len(overlap_mismatch_samples),
03236 |         "prediction_artifacts_standalone_safe": (
03237 |             None if not sample_rows else len(unsafe_samples) == 0
03238 |         ),
03239 |         "unsafe_sample_ids": sorted(unsafe_samples)[:50],
03240 |         "standalone_hsx_unsafe_reason": unsafe_reason,
03241 |         "standalone_hsx_caveat": (
03242 |             "Do not use ML_prediction.HSX as a standalone Hamiltonian+overlap container unless "
03243 |             "prediction_self_contained_hsx_safe is true for the sample. Official spectral metrics "
03244 |             "use S_ref, not prediction-owned overlap."
03245 |         ),
03246 |     }
```

### `matrix_spectrum_rows` — líneas 3268–3301

```py
03268 | def matrix_spectrum_rows(
03269 |     sparse_rows: list[dict[str, Any]],
03270 |     spectral_rows: list[dict[str, Any]],
03271 | ) -> list[dict[str, Any]]:
03272 |     spectral_by_sample = {str(row["sample"]): row for row in spectral_rows}
03273 |     rows: list[dict[str, Any]] = []
03274 |     for sparse_row in sparse_rows:
03275 |         spectral_row = spectral_by_sample.get(str(sparse_row["sample"]))
03276 |         if spectral_row is None:
03277 |             continue
03278 |         rows.append(
03279 |             {
03280 |                 "sample": sparse_row["sample"],
03281 |                 "matrix_metric_target_space": sparse_row.get("matrix_metric_target_space"),
03282 |                 **{field: sparse_row.get(field) for field in MATRIX_SEMANTIC_FIELDS},
03283 |                 "mae_ref_eV": sparse_row.get("mae_ref_eV"),
03284 |                 "rmse_ref_eV": sparse_row.get("rmse_ref_eV"),
03285 |                 "mse_ref_eV2": sparse_row.get("mse_ref_eV2"),
03286 |                 "r2_ref": sparse_row.get("r2_ref"),
03287 |                 "rmse_union_eV": sparse_row.get("rmse_union_eV"),
03288 |                 "mse_union_eV2": sparse_row.get("mse_union_eV2"),
03289 |                 "r2_union": sparse_row.get("r2_union"),
03290 |                 "relative_frobenius_union": sparse_row.get("relative_frobenius_union"),
03291 |                 "support_f1": sparse_row.get("support_f1"),
03292 |                 "global_rmse_eV": spectral_row.get("global_rmse_eV"),
03293 |                 "low_energy_rmse_eV": spectral_row.get("low_energy_rmse_eV"),
03294 |                 "fermi_window_rmse_eV": spectral_row.get("fermi_window_rmse_eV"),
03295 |                 "frontier_window_rmse_eV": spectral_row.get("frontier_window_rmse_eV"),
03296 |                 "gap_abs_error_eV": spectral_row.get("gap_abs_error_eV"),
03297 |                 "fermi_level_source": spectral_row.get("fermi_level_source"),
03298 |                 "fermi_metric_available": spectral_row.get("fermi_level_source") == "siesta_file",
03299 |             }
03300 |         )
03301 |     return rows
```
