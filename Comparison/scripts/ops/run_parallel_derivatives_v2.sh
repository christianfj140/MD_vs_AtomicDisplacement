#!/usr/bin/env bash
# v2: lanza los casos de derivadas pendientes en paralelo CONTROLADO.
# Cambios vs v1: MAX_PARALLEL=3 (evita la sobre-suscripcion de load~93 en 24
# nucleos observada con 5), y limita los threads BLAS de cada proceso autograd
# (OMP/MKL/OPENBLAS/NUMEXPR + torch) para que 3 casos no saturen la maquina.
# El sweep de vacancy YA termino, asi que este script no espera: arranca directo.
# Salta los casos que ya tengan manifest final (derivative_metrics/*/manifest.json).
set -uo pipefail

REPO="/home/christian/repositorios/MD_vs_AtomicDisplacement"
PY="$REPO/.venv/bin/python"
LAUNCH="$REPO/Comparison/scripts/ops/launch_ui_real_metrics_derivatives.py"
LOGDIR="/tmp/claude-1003/-home-christian-repositorios-MD-vs-AtomicDisplacement/d70366fa-8fa5-4548-aea0-743a9a447a8a/scratchpad"
WLOG="$LOGDIR/derivatives_parallel_v2.log"

MAX_PARALLEL=3
# Cada proceso autograd limitado a 4 threads BLAS: 3 casos x ~4-6 procesos internos
# x 4 threads sigue siendo mucho, pero el cap evita que cada uno acapare 24 threads.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

# Orden: SIESTA-ya-hecha primero (autograd rapido), luego los que rehacen SIESTA.
CASES=(
  iid20 iid30 iid100 iid150 iid200 iid300
  iid50 iid60 iid80 iid90 iid400 iid500
)

log(){ echo "$(date -Is) $*" >> "$WLOG"; }

case_done(){
  local id="$1"
  local base="$REPO/Comparison/results/ui_real_metrics_derivatives/cross_w90_to_5x5_2delta/cross_graphene__graphene_w90_scale_${id}__to__graphene_5x5__graphene_5x5_scale_${id}"
  find "$base/derivative_metrics" -iname "manifest.json" 2>/dev/null | grep -q . && return 0
  return 1
}

log "v2 iniciado. MAX_PARALLEL=$MAX_PARALLEL, threads BLAS=4/proc."

declare -A PIDS
running=0
for id in "${CASES[@]}"; do
  if case_done "$id"; then
    log "$id ya tiene manifest final, salto."
    continue
  fi
  payload="$REPO/Comparison/config/ui_cross_w90_to_5x5_2delta_${id}_solo_payload.json"
  if [[ ! -f "$payload" ]]; then
    log "AVISO: no existe payload $payload, salto $id"
    continue
  fi
  # Esperar hueco si ya hay MAX_PARALLEL corriendo.
  while (( running >= MAX_PARALLEL )); do
    sleep 30
    running=0
    for p in "${PIDS[@]}"; do kill -0 "$p" 2>/dev/null && running=$((running+1)); done
  done
  caselog="$LOGDIR/derivatives_solo_${id}.log"
  nohup "$PY" -u "$LAUNCH" "$payload" > "$caselog" 2>&1 &
  PIDS[$id]=$!
  running=$((running+1))
  log "lanzado $id -> PID ${PIDS[$id]}"
  sleep 5
done

log "todos encolados. Esperando a que terminen."
for id in "${!PIDS[@]}"; do
  wait "${PIDS[$id]}" 2>/dev/null
  log "caso $id (PID ${PIDS[$id]}) terminado."
done
log "v2: TODOS los casos de derivadas terminados."
