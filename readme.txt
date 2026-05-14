Este repositorio compara predicciones de Hamiltonianos de graph2mat sobre agua
usando tres metodos de generacion de dataset:

- md
- siesta_fc_cartesian
- random_cartesian

La documentacion canonica esta en README.md. El punto de entrada recomendado es:

python3 Comparison/scripts/pipeline_ui.py

La UI se abre en http://127.0.0.1:8770 y permite seleccionar que metodos ejecutar,
generar datasets, lanzar el pipeline completo, revisar plots y borrar artefactos
generados de forma selectiva.
