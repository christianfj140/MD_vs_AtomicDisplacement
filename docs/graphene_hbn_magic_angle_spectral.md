# Campaña espectral grafeno/hBN a ángulo mágico

Esta campaña predice espectros sin Hamiltoniano SIESTA de referencia para el
target. No forma parte del ranking MAE/Frobenius del cross-testing convencional.

## Contrato físico

- Entrenamiento: celdas C4BN de 6 átomos, con stackings AA, AB1, AB2 y las
  traslaciones mínimas del grafeno superior necesarias para cubrir AA/AB/BA.
- Target: aproximante conmensurable `(m,n)=(31,30)`, 16 746 átomos y 117 222
  orbitales.
- hBN y el grafeno inferior permanecen alineados; sólo gira el grafeno superior.
- La geometría es rígida, no relajada, y registra la deformación de hBN.
- `S` procede de integrales PAO exactas de SIESTA con `TS.onlyS T` y cero ciclos
  SCF. No se permite `S=I`.
- `H` del target procede exclusivamente de Graph2Mat o DeepH. No existe
  `H_target` SIESTA ni se publican MAE/Frobenius para el ángulo mágico.

Los smoke históricos 4→16 se conservan como
`legacy_invalid_geometry`, pero están excluidos de la campaña y de sus curvas.

## Ejecución

```bash
.venv/bin/python Comparison/scripts/run_graphene_hbn_moire_spectral_campaign.py plan
.venv/bin/python Comparison/scripts/run_graphene_hbn_moire_spectral_campaign.py resume
.venv/bin/python Comparison/scripts/run_graphene_hbn_moire_spectral_campaign.py status
.venv/bin/python Comparison/scripts/run_graphene_hbn_moire_spectral_campaign.py stop
```

Las acciones individuales son:

```text
generate-training-data  train  build-target  build-overlap
predict                 solve-bands  solve-dos  aggregate
```

`resume` reutiliza datasets, checkpoints, overlap, predicciones y tiers
completados. El lock de campaña impide dos ejecuciones simultáneas.

La configuración reproducible está en
`Comparison/config/graphene_hbn_magic_angle_spectral_campaign.json`.

## Entrenamiento y coste

- Tamaños anidados: 30, 60, 120, 240 y 480 snapshots.
- Seeds científicos previstas: 0, 1 y 2. La ejecución activa usa seed 0 para
  completar el smoke end-to-end sin comprometer el margen de disco; 1–2 quedan
  diferidas para la ampliación estadística. Seed 0 representa Tier B/C.
- Early stopping común: `val_loss`, `patience=150`, `min_delta=1e-5`, máximo
  750 épocas.
- Un solo entrenamiento de cada modelo puede usar la GPU.
- Un lock compartido serializa entrenamientos e inferencias grandes en GPU.
- Antes de arrancar, Graph2Mat exige 18 000 MiB libres y DeepH 24 000 MiB; el
  runner espera hasta una hora en lugar de competir por VRAM.
- Graph2Mat conserva el checkpoint e3nn original y evalúa MACE y sus readouts
  en chunks de 8192 aristas y 512 nodos. En la celda C4BN, la comparación
  chunk/no-chunk sin TF32 dio Frobenius relativo `2.90e-7` y diferencia máxima
  `5.72e-6 eV`; cada manifiesto registra estos parámetros.
- La campaña no inicia un nuevo tamaño o una nueva predicción con menos del
  12 % de disco libre, conservando margen sobre el mínimo operativo del 10 %.

## Predicción DeepH sin H de referencia

DeepH no implementa su rama `create_from_DFT=False`. Por ello la campaña usa
`create_from_DFT=True` en las tareas 2/3, pero el único artefacto DFT del target
que se lee es `overlaps.h5`, empleado para construir vecindarios y coordenadas
locales. No se crea, lee ni requiere `hamiltonians.h5` del target.

## Solver y tiers

El entorno Julia local se crea con:

```bash
.venv/bin/python Comparison/scripts/bootstrap_deeph_sparse_solver.py
.venv/bin/python Comparison/scripts/validate_deeph_sparse_solver.py
```

El backend por defecto sigue siendo `cpu_mkl_pardiso`. El backend opcional
`gpu_cudss` usa el mismo `sparse_calc.jl`, ARPACK y formatos de salida, pero
reemplaza sólo la factorización shift-invert por cuDSS:

```bash
.venv/bin/python Comparison/scripts/bootstrap_deeph_sparse_solver.py --backend gpu_cudss
.venv/bin/python Comparison/scripts/run_deeph_sparse_spectrum.py \
  --backend gpu_cudss --input-dir INPUT --output-dir OUTPUT \
  --job band --fermi-level FERMI
```

Se ejecuta en un proyecto/depot Julia separado, con matrices
`ComplexF64/Int64`, LDLᴴ hermítica indefinida, una factorización reutilizada por
punto k y memoria híbrida host/GPU limitada por defecto a 28 GiB. No hay
fallback silencioso a CPU ni denso. Los manifiestos distinguen
`backend_requested` y `backend_effective`; los guardarraíles paran el proceso
si la GPU alcanza 80 °C, el sistema baja de 20 GiB disponibles o el disco baja
del 12 %.

Cuando Tier B CPU haya terminado, el benchmark autorizado de un único punto Γ
se ejecuta exactamente con:

```bash
.venv/bin/python Comparison/scripts/run_deeph_sparse_spectrum.py \
  --input-dir Comparison/results/graphene_hbn_magic_angle_spectral/predictions/graph2mat/n480/seed0/solver_input \
  --output-dir Comparison/results/graphene_hbn_magic_angle_spectral/solver/gpu_benchmark_magic_angle_gamma \
  --job band --fermi-level -5.392601890862011 --num-bands 16 \
  --gamma-only --backend gpu_cudss
```

No debe ejecutarse mientras siga activo el solver CPU.

No hay fallback denso para 117 222 orbitales.
Cada proceso Julia se ejecuta con un límite de espacio de direcciones que
reserva 12 GiB para el sistema. También exige al menos 12 % de disco libre; un
límite insuficiente produce `resource_blocked` antes de iniciar Pardiso.

- Tier A: Γ, K y M, 40 bandas, todos los tamaños/modelos/seeds.
- Tier B: Γ–K–M–Γ, 8 puntos por segmento (22 k-points) y 16 bandas, sólo N=480 seed 0.
- Tier C: DOS parcial de baja energía en malla 3×3×1, sólo N=480 seed 0.

Los resultados se alinean a una neutralidad estimada desde celdas pequeñas. No
se denomina Fermi exacto del target.

## Artefactos y UI

La raíz persistente es:

```text
Comparison/results/graphene_hbn_magic_angle_spectral/
```

Los endpoints son:

```text
GET  /api/cross-testing/bilayer/spectral/plan
POST /api/cross-testing/bilayer/spectral/launch
GET  /api/cross-testing/bilayer/spectral/status
GET  /api/cross-testing/bilayer/spectral/results
POST /api/cross-testing/bilayer/spectral/stop
```

La UI separa el smoke legacy de la validación física C4BN y muestra bandas, DOS
parcial, consistencia entre modelos y coste. “Consistencia Graph2Mat–DeepH” no
es un error frente a ground truth.

## Gates ya reproducibles

- overlap-only frente a SIESTA completo, C4BN: Frobenius relativo 0 en Γ/K/M;
- solver sintético con shift no nulo: error máximo `3.33e-16 eV`;
- solver físico C4BN, 42 orbitales: error máximo `1.27e-7 eV`;
- cuDSS vs MKL-Pardiso en C4BN: error máximo `9.95e-14 eV`, residual
  relativo máximo `7.42e-15` y diferencia entre dos ejecuciones `0 eV`;
- target exacto: 1 331 054 bloques, 62 927 700 elementos no nulos y hash
  `44381bbcbbbe67e82def583c4ebbcd0dd34be8fff928b837c5d5aa26ac816ff9`.

Si la predicción o factorización grande agota recursos, el estado correcto es
`resource_blocked` con telemetría; nunca se sustituye por un cálculo denso ni
por overlap identidad.
