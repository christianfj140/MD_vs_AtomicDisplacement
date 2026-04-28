# MD_vs_OnlyAtomDisplacement

Repositorio para comparar las predicciones de `graph2mat` entrenado con dos tipos de datasets sobre la misma molecula de agua:

1. `MD/`: dataset generado a partir de dinamica molecular.
2. `AtomDisplacement/`: dataset generado a partir de pequeñas perturbaciones aleatorias alrededor de una geometria relajada.

El objetivo es estudiar como cambia la calidad de las predicciones del Hamiltoniano cuando el modelo se entrena con trayectorias MD frente a un muestreo local del entorno geometrico de equilibrio.

## Estructura del repositorio

```text
MD_vs_OnlyAtomDisplacement/
├── MD/
│   ├── dataset/
│   ├── scripts/
│   └── training/
├── AtomDisplacement/
│   ├── base/
│   ├── relaxed/
│   ├── dataset/
│   ├── scripts/
│   └── training/
├── README.md
└── readme.txt
```

## Dependencias

Este repo asume que existen al menos estas herramientas:

- `siesta`
- `graph2mat`
- un entorno virtual en `/home/christian/graph2mat-env`

Los scripts de `AtomDisplacement` activan automáticamente ese entorno cuando llaman a `graph2mat` o a `siesta`.

## Flujo `MD`

La carpeta `MD/` contiene el pipeline original basado en dinamica molecular.

### Script principal

- [main_md.py](/home/christian/Escritorio/CINN/repositorios/MD_vs_OnlyAtomDisplacement/MD/scripts/main_md.py)

### Orden del pipeline

1. generar dataset MD
2. entrenar
3. testear
4. predecir

### Ejecucion

Desde la raiz del repo:

```bash
python3 MD/scripts/main_md.py
```

## Flujo `AtomDisplacement`

La carpeta `AtomDisplacement/` contiene un pipeline nuevo para construir un dataset de agua deformada localmente y entrenar/testear/predecir sobre el Hamiltoniano resultante.

### Idea general

1. relajar una molecula de agua de referencia
2. generar muchas copias con pequeños desplazamientos aleatorios
3. correr SIESTA en single-point sobre cada geometria
4. recolectar las salidas
5. entrenar, testear y predecir con `graph2mat`

## Carpetas importantes en `AtomDisplacement`

- `base/`: entrada base de relajacion y pseudopotenciales
- `relaxed/`: salida de la relajacion de referencia
- `dataset/samples/`: una carpeta por muestra generada
- `dataset/collected/`: resumen del dataset en JSON y CSV
- `training/`: artefactos de entrenamiento y test

## Pipeline de dataset en `AtomDisplacement`

### Script principal

- [main_atom_displacement.py](/home/christian/Escritorio/CINN/repositorios/MD_vs_OnlyAtomDisplacement/AtomDisplacement/scripts/main_atom_displacement.py)

### Scripts individuales

- [run_relaxation.py](/home/christian/Escritorio/CINN/repositorios/MD_vs_OnlyAtomDisplacement/AtomDisplacement/scripts/run_relaxation.py)
- [generate_atom_displacement_dataset.py](/home/christian/Escritorio/CINN/repositorios/MD_vs_OnlyAtomDisplacement/AtomDisplacement/scripts/generate_atom_displacement_dataset.py)
- [run_single_points.py](/home/christian/Escritorio/CINN/repositorios/MD_vs_OnlyAtomDisplacement/AtomDisplacement/scripts/run_single_points.py)
- [collect_atom_displacement_dataset.py](/home/christian/Escritorio/CINN/repositorios/MD_vs_OnlyAtomDisplacement/AtomDisplacement/scripts/collect_atom_displacement_dataset.py)

### Ejecucion completa

```bash
python3 AtomDisplacement/scripts/main_atom_displacement.py
```

### Ejecucion paso a paso

```bash
python3 AtomDisplacement/scripts/run_relaxation.py
python3 AtomDisplacement/scripts/generate_atom_displacement_dataset.py --num-samples 100
python3 AtomDisplacement/scripts/run_single_points.py
python3 AtomDisplacement/scripts/collect_atom_displacement_dataset.py
```

## Variables importantes del dataset de `AtomDisplacement`

La mayoria de parametros de generacion se definen en [generate_atom_displacement_dataset.py](/home/christian/Escritorio/CINN/repositorios/MD_vs_OnlyAtomDisplacement/AtomDisplacement/scripts/generate_atom_displacement_dataset.py):

- `--num-samples`: numero de estructuras a generar
- `--sigma`: amplitud del desplazamiento aleatorio en angstrom
- `--seed`: semilla aleatoria
- `--max-displacement-norm`: desplazamiento maximo por atomo
- `--min-oh`, `--max-oh`: filtro de distancia O-H
- `--min-hh`: filtro de distancia H-H
- `--min-angle`, `--max-angle`: filtro del angulo H-O-H

Los parametros del `.fdf` base de relajacion estan en [RUN.fdf](/home/christian/Escritorio/CINN/repositorios/MD_vs_OnlyAtomDisplacement/AtomDisplacement/base/RUN.fdf).

## Pipeline de entrenamiento en `AtomDisplacement`

### Script principal

- [main_atdisp.py](/home/christian/Escritorio/CINN/repositorios/MD_vs_OnlyAtomDisplacement/AtomDisplacement/scripts/main_atdisp.py)

### Scripts individuales

- [run_atdisp_training.py](/home/christian/Escritorio/CINN/repositorios/MD_vs_OnlyAtomDisplacement/AtomDisplacement/scripts/run_atdisp_training.py)
- [run_atdisp_testing.py](/home/christian/Escritorio/CINN/repositorios/MD_vs_OnlyAtomDisplacement/AtomDisplacement/scripts/run_atdisp_testing.py)
- [run_atdisp_prediction.py](/home/christian/Escritorio/CINN/repositorios/MD_vs_OnlyAtomDisplacement/AtomDisplacement/scripts/run_atdisp_prediction.py)

### Ejecucion completa

```bash
python3 AtomDisplacement/scripts/main_atdisp.py
```

### Ejecucion paso a paso

```bash
python3 AtomDisplacement/scripts/run_atdisp_training.py
python3 AtomDisplacement/scripts/run_atdisp_testing.py
python3 AtomDisplacement/scripts/run_atdisp_prediction.py
```

## Artefactos generados en `AtomDisplacement`

### Dataset

- [samples_manifest.json](/home/christian/Escritorio/CINN/repositorios/MD_vs_OnlyAtomDisplacement/AtomDisplacement/dataset/samples_manifest.json)
- [water_atom_displacement_dataset.json](/home/christian/Escritorio/CINN/repositorios/MD_vs_OnlyAtomDisplacement/AtomDisplacement/dataset/collected/water_atom_displacement_dataset.json)
- [water_atom_displacement_summary.csv](/home/christian/Escritorio/CINN/repositorios/MD_vs_OnlyAtomDisplacement/AtomDisplacement/dataset/collected/water_atom_displacement_summary.csv)

### Entrenamiento

- [config.yaml](/home/christian/Escritorio/CINN/repositorios/MD_vs_OnlyAtomDisplacement/AtomDisplacement/training/config.yaml)
- [sample_metrics.csv](/home/christian/Escritorio/CINN/repositorios/MD_vs_OnlyAtomDisplacement/AtomDisplacement/training/sample_metrics.csv)
- `training/lightning_logs/atom_displacement_model/...`

## Estado actual del repositorio

Actualmente el flujo `AtomDisplacement` ya fue ejecutado con una prueba de 20 muestras:

- 20 estructuras generadas
- 20 calculos single-point completados
- entrenamiento, test y prediccion ejecutados
- checkpoint disponible en `AtomDisplacement/training/lightning_logs/atom_displacement_model/version_0/checkpoints/`

## Notas

- En `AtomDisplacement/scripts/` todavia existen algunos scripts `run_md_*` heredados del flujo antiguo. No forman parte del pipeline nuevo de desplazamientos atomicos.
- Si se quiere un modelo mas util fisicamente, conviene aumentar el numero de muestras antes de sacar conclusiones del entrenamiento.

## UI de comparacion

Para lanzar `MD` y `AtomDisplacement` con un solo click:

```bash
python3 Comparison/scripts/pipeline_ui.py
```

Abre `http://127.0.0.1:8770`. La pantalla ejecuta ambos pipelines, muestra logs separados y resume los artefactos que se usaran para comparar resultados.

La pestaña `Experiment` permite barrer tamaños de dataset, por ejemplo:

- `MD`: `50, 100, 200, 500`
- `AtomDisplacement`: `100, 1000, 10000`

Cada tamaño se ejecuta en un workspace aislado dentro de `Comparison/workspaces/` y se archiva en:

- `Comparison/results/results_md/dataset_<N>/run_<timestamp>/`
- `Comparison/results/results_atomdisp/dataset_<N>/run_<timestamp>/`

Cada carpeta de resultados guarda `structures/`, `predicted_hamiltonians/`, `siesta_hamiltonians/`, `sample_metrics.csv` cuando existe, `pipeline_config.yaml`, `run.log` y `manifest.json`.
