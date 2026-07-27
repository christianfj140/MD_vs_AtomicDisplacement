#!/usr/bin/env bash
# Espera a que TERMINE el sweep de derivadas (PID pasado como $1) y ENTONCES
# entrena vacancy seed 1 y, al acabar, seed 2 (1 seed a la vez, para no doblar
# carga de CPU/RAM/GPU). Mismo payload TRAIN e hiperparametros de seed 0.
#
# Uso:  nohup bash Comparison/scripts/ops/enqueue_vacancy_seeds_after_derivatives.sh <PID_DERIVADAS> &
set -uo pipefail

REPO="/home/christian/repositorios/MD_vs_AtomicDisplacement"
PY="$REPO/.venv/bin/python"
RUNNER="$REPO/Comparison/scripts/run_cross_structure_sweep_payload.py"
PAYLOAD="$REPO/Comparison/config/graphene_w90_5x5_to_vacancy_TRAIN_seeds1_2_payload.json"
OUTROOT="$REPO/Comparison/results/ml_vs_siesta_cross_structure_vacancy"
LOGDIR="/tmp/claude-1003/-home-christian-repositorios-MD-vs-AtomicDisplacement/50d6d312-8b39-44ac-b883-ad08233dc06f/scratchpad"
WLOG="$LOGDIR/vacancy_enqueue_watcher.log"
mkdir -p "$LOGDIR"

DERIV_PID="${1:?Falta el PID del sweep de derivadas}"
log(){ echo "$(date -Is) $*" | tee -a "$WLOG"; }

log "watcher iniciado. Esperando fin del sweep de derivadas (PID $DERIV_PID)."
while kill -0 "$DERIV_PID" 2>/dev/null; do sleep 60; done
log "sweep de derivadas termino."

run_seed(){
  local s="$1"
  log "lanzando vacancy seed $s (1 seed a la vez)."
  "$PY" -u "$RUNNER" "$PAYLOAD" \
    --action train \
    --seeds "$s" \
    --output-root "$OUTROOT/seed_$s" \
    --result-json "$OUTROOT/train_seed${s}_result.json" \
    > "$LOGDIR/vacancy_TRAIN_seed${s}.log" 2>&1
  log "vacancy seed $s termino (exit $?)."
}

run_seed 1
run_seed 2
log "vacancy seeds 1 y 2 COMPLETO."
