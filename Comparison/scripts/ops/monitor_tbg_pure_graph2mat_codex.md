Supervisa y recupera exclusivamente la campaña TBG puro Graph2Mat:

- Repositorio: `/home/christian/repositorios/MD_vs_AtomicDisplacement`.
- Servicio: `tbg-pure-graph2mat-campaign.service`.
- Controlador reanudable: `Comparison/scripts/run_tbg_pure_graph2mat_campaign.py`.
- Estado: `Comparison/results/tbg_pure_graph2mat/status.json`.
- Gate: `Comparison/results/tbg_pure_graph2mat/precision_gate.json`.
- Resultado final: `Comparison/results/tbg_pure_graph2mat/summary/spectral_results.json`.

Reglas:

1. Usa primero como máximo cinco comandos para comprobar servicio, estado, logs recientes, disco, RAM, CPU y GPU. Si progresa y está sano, termina.
2. DeepH está prohibido. Sólo Graph2Mat, seed 0 y el dataset N=474 ya materializado.
3. Si el gate terminó con `failed` o el estado es `gate_failed`, no reinicies: es una parada científica intencionada.
4. Si terminó con `completed`, verifica que `spectra` no esté vacío y que la UI responda; no reinicies cálculos.
5. Nunca continúes ni reinicies con menos de 12% de disco libre, menos de 20 GiB de RAM, CPU >=80 °C o GPU >=75 °C.
6. No borres artefactos. El controlador reutiliza etapas completas y checkpoints; nunca elimines salidas para empezar de cero. Un solver sin checkpoint sólo puede reiniciarse si falló realmente.
7. Si el servicio falló, identifica la causa en `Comparison/results/tbg_pure_graph2mat/logs/`. Corrige únicamente un fallo reproducible y mínimo, ejecuta la prueba relevante y usa `systemctl --user restart tbg-pure-graph2mat-campaign.service` sólo con recursos seguros.
8. Si faltan recursos o la causa no es recuperable, espera al siguiente cron sin mutar resultados.
9. Preserva el worktree, no hagas commit/push y no lances otras campañas.
10. Sal tras confirmar salud o recuperación; no esperes a que termine una etapa larga.

Informa brevemente de etapa, progreso, acción y recursos.
