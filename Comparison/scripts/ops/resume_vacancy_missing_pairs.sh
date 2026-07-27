#!/usr/bin/env bash
# RESUME del sweep de vacancy tras el CUDA OOM del run 08:30 (24-jul).
# Entrena SOLO los 20 pares faltantes (iid>=60) por seed, seed 1 y luego seed 2
# (1 seed a la vez). Los iid20/30/50 ya entrenados quedan intactos (excluidos del payload).
# prediction_jobs=1 en el payload evita repetir el OOM.
#
# Uso:  nohup bash Comparison/scripts/ops/resume_vacancy_missing_pairs.sh &
set -uo pipefail

REPO="/home/christian/repositorios/MD_vs_AtomicDisplacement"
PY="$REPO/.venv/bin/python"
RUNNER="$REPO/Comparison/scripts/run_cross_structure_sweep_payload.py"
PAYLOAD="$REPO/Comparison/config/graphene_w90_5x5_to_vacancy_TRAIN_seeds1_2_RESUME_payload.json"
OUTROOT="$REPO/Comparison/results/ml_vs_siesta_cross_structure_vacancy"
LOGDIR="/tmp/claude-1003/-home-christian-repositorios-MD-vs-AtomicDisplacement/50d6d312-8b39-44ac-b883-ad08233dc06f/scratchpad"
WLOG="$LOGDIR/vacancy_resume_watcher.log"
mkdir -p "$LOGDIR"

log(){ echo "$(date -Is) $*" | tee -a "$WLOG"; }

run_seed(){
  local s="$1"
  log "lanzando RESUME vacancy seed $s (20 pares faltantes, prediction_jobs=1)."
  "$PY" -u "$RUNNER" "$PAYLOAD" \
    --action train \
    --seeds "$s" \
    --output-root "$OUTROOT/seed_$s" \
    --result-json "$OUTROOT/resume_seed${s}_result.json" \
    > "$LOGDIR/vacancy_RESUME_seed${s}.log" 2>&1
  log "RESUME vacancy seed $s termino (exit $?)."
}

log "resume iniciado. GPU libre; 1 seed a la vez."
run_seed 1
run_seed 2
log "RESUME vacancy seeds 1 y 2 COMPLETO."
