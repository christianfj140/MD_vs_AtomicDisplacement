#!/usr/bin/env bash
# Espera a que el reentrenamiento de seed 1 (PID $SEED1_PID) termine, y ENTONCES
# lanza seed 2 (1 seed a la vez, para no duplicar carga de CPU/RAM/GPU).
# Ambos usan el mismo payload TRAIN (g2m=6, deeph=4) e hiperparametros de seed 0.
set -uo pipefail

REPO="/home/christian/repositorios/MD_vs_AtomicDisplacement"
PY="$REPO/.venv/bin/python"
PAYLOAD="$REPO/Comparison/config/graphene_w90_5x5_to_vacancy_TRAIN_seeds1_2_payload.json"
OUTROOT="$REPO/Comparison/results/ml_vs_siesta_cross_structure_vacancy"
LOGDIR="/tmp/claude-1003/-home-christian-repositorios-MD-vs-AtomicDisplacement/d70366fa-8fa5-4548-aea0-743a9a447a8a/scratchpad"
WLOG="$LOGDIR/train_seed_watcher.log"

SEED1_PID="$1"

log(){ echo "$(date -Is) $*" >> "$WLOG"; }
log "watcher iniciado. Esperando fin de seed 1 (PID $SEED1_PID)."

while kill -0 "$SEED1_PID" 2>/dev/null; do sleep 60; done
log "seed 1 termino. Lanzando seed 2 (1 seed a la vez)."

nohup "$PY" -u "$REPO/Comparison/scripts/run_cross_structure_sweep_payload.py" \
  "$PAYLOAD" \
  --action train \
  --seeds 2 \
  --output-root "$OUTROOT/seed_2" \
  --result-json "$OUTROOT/train_seed2_result.json" \
  > "$LOGDIR/vacancy_TRAIN_seed2.log" 2>&1 &
seed2_pid=$!
log "seed 2 lanzado -> PID $seed2_pid"
wait "$seed2_pid" 2>/dev/null
log "seed 2 termino. Reentrenamiento de ambas seeds COMPLETO."
