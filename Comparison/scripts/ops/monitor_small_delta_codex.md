Supervisa y, sólo si hace falta, recupera la campaña de derivadas pequeñas:

- Repositorio: `/home/christian/repositorios/MD_vs_AtomicDisplacement`.
- Payload: `Comparison/config/ui_cross_w90_to_5x5_delta_0p0005_0p001_payload.json`.
- Resultados: `Comparison/results/ui_real_metrics_derivatives/cross_w90_to_5x5_delta_0p0005_0p001`.
- Lanzador obligatorio: `Comparison/scripts/ops/run_small_delta_campaign_guarded.sh`.
- Watchdog: `Comparison/scripts/ops/watch_small_delta_disk.sh`.
- Estado: `Comparison/results/ui_real_metrics_derivatives/watchdog_small_delta`.

Reglas:

1. Fast-path: usa como máximo cinco comandos para revisar proceso, progreso, logs recientes, disco, RAM y GPU. Si la campaña progresa y no hay errores, termina sin modificar nada.
2. No lances la campaña si nunca fue iniciada. No ejecutes ninguna otra campaña.
3. Nunca ejecutes o reinicies el workflow directamente: usa exclusivamente `run_small_delta_campaign_guarded.sh`, que crea un grupo aislado y activa el control de disco cada 5 segundos.
4. Si existe `STOPPED_LOW_DISK`, no lo borres ni reinicies. El usuario debe liberar/revisar espacio y retirar el bloqueo manualmente.
5. No continúes ni reinicies con menos del 12% de disco libre. El límite absoluto solicitado es conservar al menos el 10%.
6. No interrumpas un proceso sano. Si terminó correctamente, valida los 12 casos, los manifiestos de Graph2Mat/DeepH, ambos deltas (`0.0005`, `0.001`), gates y gráficas; después no hagas nada.
7. Si falló, determina la causa exacta en logs y manifiestos. Corrige sólo un fallo reproducible mediante el cambio mínimo y una prueba relevante. Reanuda únicamente artefactos ausentes o inválidos mediante el lanzador protegido.
8. No borres ni sobrescribas la campaña anterior `cross_w90_to_5x5_2delta`. No borres artefactos de la campaña nueva sin autorización.
9. Preserva el worktree; no hagas commit, push ni cambios externos. No generes subagentes.
10. Los deltas pequeños pueden exponer cancelación o ruido SCF: un gate científico fallido no es un fallo operativo y no autoriza a cambiar tolerancias ni resultados.
11. Sal después de comprobar salud o recuperación; no esperes a que termine un cálculo largo. Informa brevemente de estado, diagnóstico, acción y recursos.

