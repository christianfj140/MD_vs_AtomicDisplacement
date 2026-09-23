Supervisa la campaña cartesiana MD 6x6 que está ejecutando:

  Comparison/scripts/run_6x6_md_label_budget.py all

Objetivo científico inmutable:

- Reproducir con snapshots MD el producto cartesiano sintético ya terminado.
- Ntrain={4,8,16,32,64}, Nval={8,24,48}, Ntest={4,8,16,24,32,48}.
- Seed 0, batch 16, elementwise_mse, 8000 updates, 2000 validaciones,
  cosine y mejor checkpoint por val_loss.
- Sin thinning por autocorrelación ni uso de g.
- Resultados bajo Comparison/results/dataset_design_curves_v1/md_label_budget_6x6/.

En cada revisión:

1. Comprueba el PID de overnight.pid, sus hijos SIESTA/Graph2Mat, run.log,
   overnight.log, temperatura Package id 0, VRAM y artefactos terminados.
2. Resume en watchdog/last_message.md la fase, progreso, temperatura, GPU,
   errores y estimación restante. Actualiza watchdog/status.json.
3. Si todo está sano, no cambies nada.
4. El usuario ha desactivado las pausas térmicas; informa de la temperatura pero
   no pauses procesos ni cambies el paralelismo.
5. Solo si no existe final_summary.json, no queda ningún SIESTA/Graph2Mat activo
   y el lanzador murió por un fallo transitorio inequívoco, reanuda una vez el
   mismo comando `all`; el script ya es reanudable.
6. Ante errores de código, controles científicos fallidos, OOM repetido o duda,
   no edites código ni datos y no relances a ciegas: documenta el bloqueo.
7. Nunca cambies diseño, hiperparámetros, splits, criterios o paralelismo; nunca
   borres artefactos ni lances otra campaña.

Cuando final_summary.json exista, verifica brevemente que finalizó sin procesos
residuales y registra estado `completed`. Sé conciso.
