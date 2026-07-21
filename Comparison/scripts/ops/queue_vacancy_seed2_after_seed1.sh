#!/usr/bin/env bash
# Waits for the seed_1 pending relaunch to finish, then launches all 26 seed_2
# cases with the num_threads=2-fixed payload. Pure bash, no Claude.
set -u

REPO=/home/christian/repositorios/MD_vs_AtomicDisplacement
OUT="$REPO/Comparison/results/ml_vs_siesta_cross_structure_vacancy"
PAYLOAD="$REPO/Comparison/config/graphene_w90_5x5_to_vacancy_predict_metrics_payload.json"
LOG="$OUT/seed2_relaunch.log"
QLOG="$OUT/seed2_queue.log"
SEED1_PID="${1:?pass the seed_1 orchestrator PID as arg 1}"

cd "$REPO" || exit 1
echo "$(date -Is) [queue] waiting for seed_1 orchestrator PID $SEED1_PID" >> "$QLOG"
while kill -0 "$SEED1_PID" 2>/dev/null; do
    sleep 60
done
echo "$(date -Is) [queue] seed_1 finished; launching seed_2 (26 cases, num_threads=2)" >> "$QLOG"

nohup .venv/bin/python -u Comparison/scripts/run_cross_structure_sweep_payload.py \
    "$PAYLOAD" \
    --action train --seeds 2 \
    --output-root "$OUT/seed_2" \
    --result-json "$OUT/seed_2/seed2_result.json" \
    >> "$LOG" 2>&1 &
disown
echo "$(date -Is) [queue] seed_2 launched, pid=$!" >> "$QLOG"
