# C21 - publicacion en la UI

La vista existente consume sin modificar:

- `go4_verdict.json`;
- `graphene_gamma_epc_summary.json`;
- `graphene_gamma_epc_g_blocks.json`;
- el manifest SIESTA E2g original.

`GET /api/epc/graphene-gamma` expone `GO4_finite_PAO`, `full_KS`, los siete
gates, `tau_num`, `tau_backend`, `tau_model`, bloques `G/g`, normas de Frobenius,
valores singulares, contratos e hashes de procedencia. La interfaz muestra
`NO_GO:model_accuracy` como veredicto terminal y los dos checks congelados como
hallazgos de alcance no bloqueantes. La transferencia a validación, la cobertura
del result set, la recomendación y su falta de autorización se muestran como
campos distintos.

La ruta es read-only: si falta un artifact devuelve `available: false`; ni el
endpoint ni el frontend ejecutan fisica o escriben resultados.

```bash
.venv/bin/python -m pytest tests/test_epc_gamma_ui.py
.venv/bin/python Comparison/scripts/pipeline_ui.py
```
