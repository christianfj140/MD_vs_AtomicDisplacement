#!/usr/bin/env bash
# Watcher: espera a que el sweep de vacancy (predict_metrics seeds 1/2) termine,
# luego mata los dos procesos secuenciales de derivadas y relanza los 12 casos
# como procesos independientes en paralelo (limitado a MAX_PARALLEL a la vez).
#
# Cada caso reutiliza su trabajo previo via --skip-if-exists (los que ya tienen
# SIESTA completa solo repiten autograd; los stencil_retry usan overwrite:true).
#
# Diseñado para sobrevivir sin supervisión: nohup + log propio.
set -uo pipefail

REPO="/home/christian/repositorios/MD_vs_AtomicDisplacement"
PY="$REPO/.venv/bin/python"
LAUNCH="$REPO/Comparison/scripts/ops/launch_ui_real_metrics_derivatives.py"
LOGDIR="/tmp/claude-1003/-home-christian-repositorios-MD-vs-AtomicDisplacement/d70366fa-8fa5-4548-aea0-743a9a447a8a/scratchpad"
WLOG="$LOGDIR/derivatives_parallel_watcher.log"

VACANCY_PID=2200978          # predict_metrics seeds 1/2 v3
STENCIL_PID=1930317          # stencil_retry secuencial (iid50/60/80/90/400/500)
AUTOGRAD_PID=1930895         # autograd_retry secuencial (iid20/30/100/150/200/300)
MAX_PARALLEL=5               # casos de derivadas simultaneos (cada autograd ~8-9 nucleos)

# Los 12 payloads por-caso (ya creados en Comparison/config/).
# Orden: primero los que ya tienen SIESTA (autograd rapido), luego los pesados.
CASES=(
  iid20 iid30 iid100 iid150 iid200 iid300   # SIESTA ya hecha -> solo autograd
  iid50 iid60 iid80 iid90 iid400 iid500      # stencil_retry (iid60/80/90/400/500 rehacen SIESTA)
)

log(){ echo "$(date -Is) $*" >> "$WLOG"; }

log "watcher iniciado. Esperando fin del sweep de vacancy (PID $VACANCY_PID)."

# 1. Esperar a que el proceso de vacancy termine.
while kill -0 "$VACANCY_PID" 2>/dev/null; do
  sleep 30
done
log "sweep de vacancy (PID $VACANCY_PID) finalizado. Procediendo a paralelizar derivadas."

# 2. Matar los procesos secuenciales de derivadas (su progreso de stencils/SIESTA
#    queda en disco y se reutiliza via skip_if_exists).
for pid in "$STENCIL_PID" "$AUTOGRAD_PID"; do
  if kill -0 "$pid" 2>/dev/null; then
    log "matando proceso secuencial de derivadas PID $pid"
    kill "$pid" 2>/dev/null
    sleep 2
    pkill -P "$pid" 2>/dev/null
  fi
done
# Matar tambien cualquier run_graph2mat_autograd_derivative_predictions huerfano
# que hubieran dejado (para que no compitan con los relanzamientos).
sleep 3
pkill -f "run_graph2mat_autograd_derivative_predictions.py" 2>/dev/null
sleep 3
log "procesos secuenciales detenidos. Lanzando casos en paralelo (max $MAX_PARALLEL a la vez)."

# 3. Lanzar cada caso pendiente como proceso independiente, respetando MAX_PARALLEL.
running=0
declare -A PIDS
for tag in "${CASES[@]}"; do
  payload="$REPO/Comparison/config/ui_cross_w90_to_5x5_2delta_${tag}_solo_payload.json"
  if [[ ! -f "$payload" ]]; then
    log "AVISO: no existe payload $payload, salto $tag"
    continue
  fi
  # Si ya hay MAX_PARALLEL corriendo, esperar a que baje.
  while (( running >= MAX_PARALLEL )); do
    sleep 20
    running=0
    for p in "${PIDS[@]}"; do
      kill -0 "$p" 2>/dev/null && running=$((running+1))
    done
  done
  caselog="$LOGDIR/derivatives_solo_${tag}.log"
  nohup "$PY" -u "$LAUNCH" "$payload" > "$caselog" 2>&1 &
  pid=$!
  PIDS[$tag]=$pid
  running=$((running+1))
  log "lanzado $tag -> PID $pid (log: $caselog)"
  sleep 5
done

log "todos los casos encolados. Esperando a que terminen todos."
# 4. Esperar a que todos terminen.
for tag in "${!PIDS[@]}"; do
  wait "${PIDS[$tag]}" 2>/dev/null
  log "caso $tag (PID ${PIDS[$tag]}) terminado."
done
log "watcher: TODOS los casos de derivadas paralelizados han terminado."
