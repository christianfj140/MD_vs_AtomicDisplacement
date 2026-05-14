# Rendimiento del Pipeline

La pestaña **Rendimiento** de la UI configura únicamente controles cableados al backend del experimento one-click.

## SIESTA

- `max_parallel_siesta_jobs` paraleliza solo single-points independientes de AtomDisplacement y Random Cartesian. Con `1` el flujo queda serial-compatible.
- `OMP_NUM_THREADS`, `MKL_NUM_THREADS` y `OPENBLAS_NUM_THREADS` se exportan a los subprocesses SIESTA y Graph2Mat.
- La trayectoria MD no se paraleliza: es una simulación secuencial y cambiarlo alteraría el significado del dataset.

Evita oversubscription: si ejecutas `N` SIESTA jobs en paralelo y cada uno usa `T` threads, el consumo aproximado es `N*T` threads.

## Graph2Mat / PyTorch

- `compute_accelerator` escribe el accelerator de Lightning/Graph2Mat en la config temporal del workspace.
- `batch_size` y `store_in_memory` son overrides opt-in sobre la config temporal de entrenamiento/testing/predicción.
- `torch_num_threads` llama a `torch.set_num_threads` cuando Torch está disponible.
- `torch_float32_matmul_precision` aplica `torch.set_float32_matmul_precision("high"|"medium")` solo cuando el usuario lo activa.

Si `gpu` se solicita explícitamente y CUDA no está disponible, el experimento falla de forma clara. Con `auto`, se usa GPU si Torch la detecta y CPU si no.

## Orquestacion

- `max_parallel_dataset_jobs` puede ejecutar jobs independientes metodo/dataset
  en paralelo usando workspaces aislados y snapshots de config. Si se usa una
  sola GPU, el runner fuerza serializacion para evitar entrenamientos
  simultaneos en el mismo dispositivo.
- `max_parallel_prediction_jobs`, `max_parallel_evaluation_jobs` y
  `max_parallel_metric_jobs` limitan trabajos cruzados cuando sus directorios de
  salida son independientes. Las agregaciones finales siguen en el proceso padre
  para conservar manifests deterministas.
- `error_policy=fail_fast` aborta al primer fallo. `continue_on_error` deja
  terminar tareas pendientes y marca el experimento como parcial.

## Limitaciones actuales

- No hay cache global segura de SIESTA, metricas, predicciones ni entrenamiento.
- `enable_experiment_cache` esta reservado: si se activa, el backend falla de
  forma explicita hasta que exista una clave de hash completa.
- No se cambian basis, SIESTA SCF settings, splits ni validaciones para ganar
  velocidad.
