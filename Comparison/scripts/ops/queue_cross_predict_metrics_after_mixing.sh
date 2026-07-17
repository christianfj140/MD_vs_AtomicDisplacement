#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PAYLOAD="$REPO_ROOT/Comparison/config/graphene_w90_5x5_cross_structure_predict_metrics_payload.json"
OUT_ROOT="$REPO_ROOT/Comparison/results/ml_vs_siesta_cross_structure_sweep"
LOG_DIR="$OUT_ROOT/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/cross_predict_metrics_after_mixing_$(date +%Y%m%d_%H%M%S).log"

find_mixing_pids() {
  pgrep -f 'resume_mixing_sweep_oom.py|launch_mixing_sweep_payload.py|run_mixing_e2e_payload_once.py' || true
}

wait_for_pids() {
  local pids="$1"
  [ -z "$pids" ] && return 0
  echo "[queue] waiting for mixing pids: $pids" | tee -a "$LOG"
  while true; do
    local alive=""
    for pid in $pids; do
      if kill -0 "$pid" 2>/dev/null; then
        alive="$alive $pid"
      fi
    done
    [ -z "$alive" ] && break
    sleep 60
  done
}

{
  echo "[queue] repo: $REPO_ROOT"
  echo "[queue] building payload: $PAYLOAD"
  "$REPO_ROOT/.venv/bin/python" "$REPO_ROOT/Comparison/scripts/ops/build_cross_predict_metrics_payload.py" --output "$PAYLOAD"
  wait_for_pids "$(find_mixing_pids)"
  echo "[queue] preparing checkpoint/model artifacts"
  "$REPO_ROOT/.venv/bin/python" "$REPO_ROOT/Comparison/scripts/ops/prepare_cross_predict_metrics_artifacts.py" "$PAYLOAD" --result-json "$OUT_ROOT/predict_metrics_artifacts_prepare.json"
  echo "[queue] launching cross predict_metrics"
  "$REPO_ROOT/.venv/bin/python" "$REPO_ROOT/Comparison/scripts/run_cross_structure_sweep_payload.py" "$PAYLOAD" --output-root "$OUT_ROOT" --result-json "$OUT_ROOT/predict_metrics_result.json"
  echo "[queue] done"
} 2>&1 | tee -a "$LOG"
