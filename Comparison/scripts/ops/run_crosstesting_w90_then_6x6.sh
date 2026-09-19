#!/bin/bash
# Sequential chain: full w90 cross-testing-hyperparams campaign (parallel=7, all
# SIESTA already cached, GPU-only cost) -> full 6x6 campaign (parallel=2, real new
# SIESTA at modest worker count for thermal caution).
set -uo pipefail
cd /home/christian/repositorios/MD_vs_AtomicDisplacement || exit 1
LOG_DIR="Comparison/results/dataset_design_w90_001_s9_chain_logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/crosstesting_w90_then_6x6.log"

echo "[$(date -Is)] === w90 campaign starting (80 combos, parallel=7) ===" | tee -a "$LOG"
.venv/bin/python Comparison/scripts/dataset_design_w90_001_s9_crosstesting_campaign_w90.py --parallel 7 \
  > "$LOG_DIR/w90_campaign_run.log" 2>&1
rc=$?
echo "[$(date -Is)] w90 campaign exit code: $rc" | tee -a "$LOG"
if [ "$rc" -ne 0 ]; then
  echo "[$(date -Is)] === chain stopped: w90 failed ===" | tee -a "$LOG"
  exit "$rc"
fi

echo "[$(date -Is)] === 6x6 campaign starting (80 combos, workers=5, parallel=2) ===" | tee -a "$LOG"
.venv/bin/python Comparison/scripts/dataset_design_w90_001_s9_crosstesting_campaign_6x6.py --workers 5 --parallel 2 \
  > "$LOG_DIR/6x6_campaign_run.log" 2>&1
rc2=$?
echo "[$(date -Is)] 6x6 campaign exit code: $rc2" | tee -a "$LOG"
echo "[$(date -Is)] === chain finished ===" | tee -a "$LOG"
