# DATASET-DESIGN-CURVES-V1 — Pre-registro (Fase 0)

Plan fuente: `docs/gola_para_claude.md`. Este documento se congela **antes** de
lanzar ningún entrenamiento nuevo ni crear el test final. Su commit es la marca
temporal. Si algo tiene que cambiar después, se añade una sección
«Desviaciones» con la fecha y el motivo; no se reescribe lo de arriba.

Código: `Comparison/scripts/dataset_design_curves_v1.py`.
Resultados: `Comparison/results/dataset_design_curves_v1/`.

## 1. Preguntas

Por separado para `w90` (grafeno primitivo, 2 átomos, k∈{1,2} átomos
desplazados) y `6x6` (supercelda de 72 átomos, todos desplazados):

- **Q1 (primaria).** ¿Cuál es el menor N∈{4,8,16,32,64} que alcanza una
  precisión próxima a N=64?
- **Q2.** ¿Sobol (`sobol_sparse`) o `random_cartesian` da mejor H-MAE al mismo
  coste SIESTA?
- **Q3 (MD, solo `w90`).** ¿El mejor dataset diseñado es comparable a un dataset
  MD del mismo tamaño evaluado sobre exactamente las mismas estructuras?

No se intenta demostrar: que una familia sea universalmente mejor; que los
errores de `w90` y `6x6` sean comparables entre sí; generalización a otros
materiales; curva anidada para LHS; necesidad de active learning o fonones.

## 2. Métrica primaria

Para cada estructura i se calcula la matriz densa H_i(Γ) (predicha y SIESTA,
`sisl`, `Hk(k=0)`), el MAE sobre todos sus elementos, y después se promedia
entre estructuras:

E = (1/N_test) Σ_i MAE(Ĥ_i(Γ), H_i^SIESTA(Γ)), en meV por elemento.

Es la definición ya usada por `s4.compute_validation_metrics`, así que los
números antiguos y nuevos son comparables dentro de cada sistema. Al ser H(Γ),
en `6x6` hay muchos elementos casi nulos (pares lejanos) y el MAE es mucho
menor que en `w90`; **nunca se comparan valores absolutos entre sistemas**.

Secundarias (mismas funciones, por estructura): H-RMSE, relative Frobenius,
hermiticidad (max |Ĥ − Ĥ†|), error espectral en Γ en ±2 eV de E_F. Coste:
tiempo de entrenamiento, mejor época, época final, memoria GPU máxima, CPU·h de
SIESTA. Bandas y DOS solo para finalistas y baseline MD.

## 3. Conjuntos de desarrollo (selección, early stopping, N*)

- **`w90`:** se mantiene `common_validation` de S5 (24 estructuras LHS:
  k∈{1,2} × 4 dims × 3, amplitud 0.03–0.08 Å), si la Fase 1 verifica que es
  idéntico para los 80 modelos, disjunto por hash de todos los pools y con
  referencia SIESTA completa.
- **`6x6`:** `common_dev_6x6_v1`, 48 estructuras = 4 dims × 4 estratos de
  amplitud real × 3 réplicas. Amplitud real = máximo sobre átomos de |Δr_a|.
  Estratos: (0,0.03], (0.03,0.05], (0.05,0.08], (0.08,0.12] Å. Las 3 réplicas
  de cada estrato son una estructura por familia (sobol, random, LHS),
  elegida entre las estructuras ya etiquetadas del test por receta
  (`seed999`), la de menor índice que caiga en el estrato. Se excluye por
  hash cualquier geometría usada para entrenar cualquiera de las 80 recetas.
  Los estratos sin candidatos (2D_in y 3D a >0.08 Å) se generan con el mismo
  generador de la familia, R=0.12, semilla 999, primera estructura, y se
  etiquetan con SIESTA (6 cálculos). Nunca se usa el error de un modelo para
  elegir estructuras.

Los desarrollos sirven para ordenar recetas, elegirlas, fijar N* y como
validación de early stopping de todos los entrenamientos nuevos. No son el test
final.

## 4. Política de entrenamiento (idéntica en todos los entrenamientos nuevos)

Graph2Mat MACE, `num_interactions=3, correlation=3, max_ell=3,
hidden_irreps=48x0e+48x1o+48x2e+48x3o`, loss `block_type_mae`, lr 0.0018,
batch 32 (Lightning usa min(32, N)), `bf16-mixed`, máximo 600 épocas,
EarlyStopping(val_loss, patience 80, min_delta 0). **Checkpoint:
`ModelCheckpoint(monitor=val_loss, mode=min, save_top_k=1)`; la predicción usa
ese checkpoint (mejor `val_loss`), no el último.** Validación = desarrollo
común del sistema (MD: ver §9). Semilla de entrenamiento 0 en el screening.

Nota: con épocas fijas, N pequeño implica menos pasos de gradiente
(1 paso/época a N≤32 frente a 2 a N=64). Se acepta tal como pide el plan
(«la única variable es N»); se documentará si explica la forma de la curva.

## 5. Selección de recetas para las curvas (Fase 3)

Candidatas: recetas N=64 de cada familia (14 Sobol, 14 random por sistema),
ordenadas por H-MAE en el desarrollo común (`w90`: evaluación existente;
`6x6`: reevaluación de los 80 checkpoints existentes sobre
`common_dev_6x6_v1`).

- **A**: la mejor de la familia.
- **B**: la mejor con dimensionalidad distinta de A.
- **C (control de cobertura)**: se decide solo con A y B, sin mirar su métrica.
  Dimensionalidades como conjuntos de ejes: 1D_in={x}, 1D_z={z},
  2D_in={x,y}, 3D={x,y,z}; distancia = |diferencia simétrica|. dim_C = la
  dimensionalidad no usada por A ni B que maximiza la distancia mínima a A y a
  B; empate → mayor suma; empate → orden [1D_z, 1D_in, 3D, 2D_in]. R_C = el R
  disponible en (familia, dim_C, N=64) que maximiza la distancia mínima en
  log R a R_A y R_B; empate → R mayor.

La selección se guarda en `selection_manifest.json` (recetas, motivo, pool,
sampler seed, ranking previo, hashes del desarrollo) y no se cambia después.

## 6. Curvas de aprendizaje (Fase 4)

Por receta, un pool maestro N=64 (existente, orden de generación, misma
`sampler_seed`) y prefijos exactos pool[0:N], N∈{4,8,16,32,64}, verificados por
hash (D4⊂D8⊂D16⊂D32⊂D64). 6 recetas × 5 N × 2 sistemas = 60 entrenamientos,
cero SIESTA nuevo. Se reentrenan también N=16 y N=64.

Por receta: E(N), ganancia g(N→2N) = (E(N) − E(2N))/E(N), y
**N* = min{N : E(N) ≤ 1.05·E(64)}**.

**N=12** solo se añade a una receta si g(8→16) > 2·max(g(4→8), g(16→32)) y
además E(8) > 1.05·E(64) y E(16) ≤ 1.05·E(64).

## 7. LHS (Fase 5)

Por sistema, la mejor receta LHS N=64 del desarrollo; mismo dataset N=64,
semillas de entrenamiento {0,1,2} (3 entrenamientos nuevos por sistema: los
existentes usan el último checkpoint y no cumplen §4). LHS pasa a finalistas si
las tres ejecuciones son válidas y su media está dentro del 5% del mejor E(64)
Sobol/random (semilla 0). Si no, la rama LHS termina.

## 8. Finalistas (Fase 6) y confirmación (Fase 7)

Coste de un punto = CPU·h SIESTA de sus estructuras de entrenamiento.

1. **Precisión:** menor E(64) entre las 6 recetas (y LHS si pasó §7).
2. **Eficiente:** el par (receta, N) de menor coste con
   E(receta,N) ≤ 1.05·E(precisión, 64); empate → menor E. Si es la misma receta
   que la de precisión, se conserva un solo finalista (su N* pasa a ser ese N)
   y el eficiente es el punto más barato de otra receta que cumpla el mismo
   umbral; si no hay ninguno, el siguiente punto de la frontera de Pareto
   (coste, E) de otra receta.
3. **Alternativo:** la mejor receta restante (por E(64)) de una familia no
   representada; si ambas familias ya lo están: LHS si pasó §7, si no la mejor
   restante con dimensionalidad distinta de las otras dos.

N* de cada finalista es el suyo propio (§6); el eficiente usa el N encontrado.
Confirmación: semillas 1 y 2 en N* y en 64 (la 0 ya existe). N* queda
confirmado si, con las 3 semillas:
(1) media y mediana cumplen Ē(N*) ≤ 1.05·Ē(64);
(2) (1) sigue cumpliéndose al quitar cualquier semilla;
(3) sd_semillas(E(N*)) ≤ max(|Ē(N*) − Ē(64)|, 0.05·Ē(64));
(4) las 6 ejecuciones terminan sin NaN y con hermiticidad < 1e-6 eV.
Si falla, se sube al siguiente N (4→8→16→32→64) y se entrenan sus semillas 1 y 2.
Se reportan media, mediana, sd, min, max e IC bootstrap descriptivo; nunca la
mejor semilla.

## 9. Baseline MD (Fase 8) — solo `w90`

No existe MD `6x6` en el repositorio: `6x6` se presenta como estudio de escalado
y construcción del dataset, **sin afirmar equivalencia con MD**.

Fuente `w90`: `Comparison/datasets/graphene_w90_*` (NVE Verlet, 1 fs, sin
termostato, 150/300/450 K), mismos ajustes SIESTA que las estructuras
sintéticas salvo el bloque MD (verificado por diff de `RUN.fdf`), y geometría
del TSHS igual a la del `RUN.fdf` del frame (≤3e-10 Å). Unidad de separación:
**trayectoria** (deduplicada por (semilla, T)). Solo T∈{150,300,450} K y
trayectorias con ≥7 frames. La autocorrelación del vector interatómico cruza
cero a 6 fs, así que solo son elegibles frames con t ≥ 6 fs (se descarta el
arranque desde la red ideal). **Un frame por trayectoria**: por construcción,
ningún par de frames es consecutivo ni comparte trayectoria entre train, val y
test. Por temperatura se ordenan las trayectorias por sha256(«seed|T»); las
primeras van a test, las siguientes a validación y el resto a entrenamiento.
El frame de cada trayectoria se elige por hash en [6, n−1].

- Test: 16 frames (5 @150 K, 6 @300 K, 5 @450 K).
- Validación: 24 frames (8 por T), el mismo tamaño que el desarrollo `w90`.
- Pool de entrenamiento: 64 frames, orden round-robin 150→300→450 K, prefijos
  N* y 64.

Arquitectura, loss, optimizador, early stopping, checkpoint y test final
idénticos a los diseñados; semillas {0,1,2}; N ∈ {N* del mejor diseñado, 64}.
La diferencia deliberada es que cada estrategia valida con datos de su propio
proceso generador (diseñado → desarrollo sintético; MD → frames MD de
validación).

## 10. Test final (Fase 9)

Se crea después de congelar recetas, N*, semillas y baseline MD. Generador
único para las estructuras sintéticas: por átomo activo, u ~ U(−1,1) en los
ejes de la dimensionalidad; después se escala toda la estructura para que su
amplitud real (máx_a |Δr_a|) sea exactamente A.
Amplitudes A ∈ {0.02, 0.04, 0.06, 0.08, 0.10, 0.12} Å.

- `w90`: k∈{1,2} × 4 dims × 6 A = 48 (átomos activos de
  `select_active_atoms`), semilla 20260919.
- `6x6`: 4 dims × 6 A × 2 réplicas = 48 (72 átomos activos), semilla 20260920.
- `w90` MD: los 16 frames de test de §9 (etiquetas existentes, 0 SIESTA). Se
  reetiquetan 4 con single-point para comprobar que la etiqueta MD equivale a
  la sintética.

Se afirma por hash que no hay solape con ningún pool de entrenamiento, con los
desarrollos ni con la validación MD. SIESTA nuevo: 96 cálculos (48 + 48).
Congelación: IDs, geometrías, hashes, referencia SIESTA, estrato, amplitud,
dimensión y origen en `final_test_manifest.json`.

Regiones respecto a un modelo con dominio R_train: in-domain si A ≤ R_train,
OOD si A > R_train (MD: R_train = amplitud máxima de sus frames de
entrenamiento).

## 11. Evaluación final (Fase 10)

Una sola evaluación de todos los modelos entrenados en esta campaña (curvas,
LHS, confirmación, MD). Comparaciones por sistema: diseñado N* vs 64; diseñado
N* vs MD N*; diseñado 64 vs MD 64; Sobol vs random; in-domain vs OOD;
sintético vs frames MD.

Diferencias emparejadas por estructura: d_i = E_i^A − E_i^B, con E_i el MAE de
la estructura i promediado sobre las 3 semillas. IC bootstrap sobre
estructuras (B = 10 000, semilla 12345, percentiles); límite superior
unilateral 95% = percentil 95.

**Margen de no inferioridad: no fijado.** El usuario no ha definido qué
degradación de bandas/DOS es aceptable para el objetivo físico, y el plan
prohíbe elegir 5 o 10 meV por defecto. Por tanto se reporta Δ con su IC y
**no se usa la palabra «no inferior»**. En desarrollo se reporta la relación
entre H-MAE y error espectral para que el margen pueda fijarse en una campaña
futura antes de abrir otro test.

## 12. Coste

- **Reproducible:** CPU·h SIESTA de las estructuras de entrenamiento (temporizador
  `siesta` de cada `RUN.out`; SIESTA corre en serie, 1 rango) + GPU·h de
  entrenamiento. En MD, el coste SIESTA de un frame es el de los t+1 pasos MD
  necesarios para llegar a él (s/paso medido en el `RUN.out` de su bloque).
- **Incremental:** SIESTA nuevo (0 para curvas) + GPU·h nuevas.
- GPU·h = tiempo de pared del entrenamiento / 3600, registrando cuántos
  entrenamientos compartían la GPU.
- El coste del desarrollo común es fijo y compartido; no se suma a cada punto.

Pareto: A domina a B si E_A ≤ E_B y C_A ≤ C_B, con una desigualdad estricta.

## 13. Criterios de parada

Se detiene la campaña (sin métodos nuevos) si ocurre cualquiera: N=32 dentro del
5% de N=64 con variabilidad solapada; mejora 32→64 menor que la variación entre
semillas; Sobol y random indistinguibles; un único diseño domina la frontera;
el error restante parece arquitectónico. Mezcla diseñado+MD, FPS, committee,
fonones y active learning quedan fuera de esta campaña.
