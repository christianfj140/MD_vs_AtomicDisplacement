# Punto 1 — Captura y evidencia

Captura realizada el `2026-07-28T10:12:54+02:00`, antes de crear
`audit_bundle_v1/`. Este directorio queda fuera de la propia captura.

## Estado de reproducibilidad

`pinned_dirty`: los tres repositorios relevantes tienen un commit resoluble,
pero los tres contienen cambios locales. Este estado no permite elevar una
ejecución nueva a `paper_ready` según `shared/run_inventory.py`.

| Repositorio | Rama | Commit | Estado |
| --- | --- | --- | --- |
| `MD_vs_AtomicDisplacement` | `main` | `39fc96b508b97fb92c5603a9258d4caf1ce47fa5` | dirty |
| `graph2mat` | `hamiltonian-spin-colineal-support` | `1a131f1c21109fb928b356a1748834c32853c477` | dirty |
| `DeepH-pack` | `agent/deeph-autograd-jvp` | `4fd2f435d09a73194731200a12fa4a37738586fb` | dirty |

La importación efectiva de `graph2mat` procede de
`/home/christian/repositorios/graph2mat/src/graph2mat/__init__.py` y coincide
con el repositorio inspeccionado. `deeph` no se puede importar desde el
intérprete principal `.venv/bin/python`; cualquier ejecución DeepH deberá
declarar su intérprete y entorno por separado.

## Estado local de este repositorio

Cambios versionados:

```text
 M Comparison/scripts/build_graphene_5x5_vacancy_target.py
 M Comparison/ui/app.js
 M Comparison/ui/index.html
```

Cambios no versionados:

```text
?? Comparison/config/cross_vacancy_to_w90_train_payload.json
?? Comparison/config/graphene_5x5_vacancy_pipeline_config.yaml
?? Comparison/config/graphene_5x5_vacancy_snapshot_scaling_payload.json
?? Comparison/scripts/ops/backfill_relative_frobenius.py
?? Comparison/scripts/ops/launch_vacancy_dataset_generation.py
?? Comparison/scripts/ops/merge_5x5_to_w90_into_cross_sweep.py
?? Comparison/scripts/ops/merge_vacancy_to_w90_into_vacancy_tab.py
?? Comparison/scripts/ops/regenerate_derivative_siesta_references.py
?? Comparison/scripts/ops/watch_queue_disk.sh
```

Resumen del diff versionado: 3 archivos, 74 inserciones y 23 eliminaciones.

Firmas de la captura:

| Objeto | SHA-256 |
| --- | --- |
| `git diff --binary` | `80f337d8a09f2e31feb3f5bd4b202fe0d69aeb27b6f5ac598e74aefb144df258` |
| `git status --porcelain=v1` | `b082427700b2f033d9f4bbeb9395b2a5f1b8ef0276d3db4c317c9dc720aa26d5` |
| fuentes versionadas `*.py`, `*.js`, `*.sh` | `d1d656499e485e17018452bf56d17b3e1bc30470dba7765941030ab04dce091c` |
| configuraciones, recetas, materiales y requirements versionados | `1c525ee58c6ce00c025eaef85645db02b77ccc8fd5294f666b5882ac1329ddae` |
| manifiestos de datasets encontrados hasta profundidad 3 | `a39c96555d754a40f4f080af539a541749f3c2636e283e616cfa58d77090530f` |

Las firmas agregadas se obtuvieron ordenando los nombres, calculando
`sha256sum` de cada archivo y calculando un segundo SHA-256 sobre esa lista.

SHA-256 de los archivos no versionados:

```text
466097cce727dec7cbfb1d9ec18910087e8294cd53dc1f1e2dee97a236686fbd  Comparison/config/cross_vacancy_to_w90_train_payload.json
11591f213f7d7f7a9afe417a2331b40b21cb9039eb170205ac9528ef79fc2591  Comparison/config/graphene_5x5_vacancy_pipeline_config.yaml
dfca3c34266eaf4b878218dfe5543a8a2185d2f5a4de44bdb7f1595ff9a8e53a  Comparison/config/graphene_5x5_vacancy_snapshot_scaling_payload.json
6f632a01138197751c57d09de4a1888ee1b843d3a71ab5393fe78a967e831f50  Comparison/scripts/ops/backfill_relative_frobenius.py
69f8a5858e35bee81c3d67123ac5c54627f29b5b3197d211be6b69afab271b5f  Comparison/scripts/ops/launch_vacancy_dataset_generation.py
533117af6c4dba592d5c5750e33921bd701ff36745322ec2ab6585a83eb13f8f  Comparison/scripts/ops/merge_5x5_to_w90_into_cross_sweep.py
151d3f9c200573e21c92b869b11def4cb4e4331106de4b7b0a75b3475e59df4a  Comparison/scripts/ops/merge_vacancy_to_w90_into_vacancy_tab.py
055f2f60731dfac7e26f1d564d67c89ec3c050a3e7aa3e6d0136fd76590fcec2  Comparison/scripts/ops/regenerate_derivative_siesta_references.py
033bd78c44c92b6b9080bc047c94ede58162ae2baa9ab1e46c05962605061504  Comparison/scripts/ops/watch_queue_disk.sh
```

## Entorno científico y hardware

| Componente | Valor |
| --- | --- |
| Sistema | Ubuntu 24.04.4 LTS, kernel `7.0.0-28-generic`, x86-64 |
| CPU | Intel Core Ultra 9 285K, 24 núcleos lógicos |
| RAM | 62 GiB |
| GPU | NVIDIA GeForce RTX 5090, 32607 MiB |
| Driver NVIDIA | 595.71.05 |
| Python | 3.12.3 |
| PyTorch | 2.11.0+cu130 |
| CUDA de PyTorch | 13.0 |
| dtype por defecto | float32 |
| SIESTA | 5.4.2-11-g4e9a46060 |
| SIESTA compiler | GNU 13.3.0, `-O3 -march=native`, MPI |
| NumPy | 2.4.4 |
| SciPy | 1.17.1 |
| ASE | 3.28.0 |
| sisl | 0.16.4 |
| graph2mat | 0.0.13, instalación editable |
| torch-geometric | 2.7.0 |
| e3nn | 0.4.4 |
| pytorch-lightning | 2.6.1 |

Ejecutables:

| Herramienta | Resolución |
| --- | --- |
| `siesta` | `/home/christian/bin/siesta` |
| `graph2mat` | no está en `PATH`; se importa como paquete editable |
| `deeph-preprocess` | no está en `PATH` |
| `deeph-train` | no está en `PATH` |
| `deeph-inference` | no está en `PATH` |

Firmas de entradas principales:

```text
06060045b4d829c79d085c9aa6f088bbcfa0208500da39705365d1904fb7b12e  /home/christian/bin/siesta
5b13ac83c680ffb879c9407af0439e33dff1ee84688f18359351e951ef83f33d  requirements-graph2mat.txt
668b7c5adf47e1e10480c5cd8d1ac68c93e9dd130e20b6a96eefa470c55ec240  MD/pipeline_config.yaml
7904a92df49cf33a117252e6371c051b404b2c8924c1a759c956551f7ae5abed  AtomDisplacement/pipeline_config.yaml
46b76599da2ae2f410cd97b5c802f6610569db61fc2cadeac44264b0604ea1f0  configs/config_md.yaml
0dd627a9f62797da9b9de7ec17ebf4e9cc636830a0a3da62bfa08ec8165b3544  configs/config_fc.yaml
```

## Inventario de código y artefactos

| Área | Cantidad |
| --- | ---: |
| Archivos versionados | 646 |
| Archivos Python versionados | 271 |
| Archivos `tests/test_*.py` | 84 |
| Datasets bajo `Comparison/datasets/` | 23 directorios, 819292 archivos |
| Resultados bajo `Comparison/results/` | 316 directorios de primer nivel, 5370092 archivos |
| Resultados archivados | 4 directorios, 133 archivos |
| Workspaces activos | 0 directorios, 1 archivo |
| `MD/dataset/` | 25 archivos |
| `AtomDisplacement/dataset/` | 2 archivos |

Los cuatro grupos de `Comparison/results_archived/` son:
`g2m_deeph_runs`, `graphene_5x2`, `results` y `workspaces`.

Materiales declarados:

```text
bilayer_graphene_hBN_AA
bilayer_graphene_hBN_AB1
bilayer_graphene_hBN_AB2
bn
graphene
graphene_5x2
graphene_5x5
graphene_5x5_vacancy
graphene_hBN_AA
graphene_hBN_AB1
graphene_hBN_AB2
graphene_hBN_common
h2o
si_amorphous
si_vacancy
```

## Límites de esta captura

- No se ha calculado un hash byte a byte de los más de seis millones de
  archivos generados. Se conservan las firmas de los manifiestos de dataset;
  cada claim seleccionado deberá verificar después sus artefactos concretos.
- El número de archivos no implica que una ejecución sea válida o completa.
- No se clasifican aún resultados como reproducibles, exploratorios o
  publicables: esa clasificación pertenece al dossier 4 y debe leerse desde
  los manifiestos, no inferirse del nombre del directorio.
- No se ha ejecutado SIESTA, Graph2Mat, DeepH ni la suite de tests durante esta
  captura.
