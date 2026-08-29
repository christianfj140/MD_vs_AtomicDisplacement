#!/usr/bin/env bash
# The only supported launcher for the delta=[0.0005, 0.001] campaign.
set -euo pipefail

REPO=/home/christian/repositorios/MD_vs_AtomicDisplacement
PYTHON="$REPO/.venv/bin/python"
LAUNCHER="$REPO/Comparison/scripts/ops/launch_ui_real_metrics_derivatives.py"
DEFAULT_PAYLOAD="$REPO/Comparison/config/ui_cross_w90_to_5x5_delta_0p0005_0p001_payload.json"
WATCHDOG="$REPO/Comparison/scripts/ops/watch_small_delta_disk.sh"
STATE="$REPO/Comparison/results/ui_real_metrics_derivatives/watchdog_small_delta"
PID_FILE="$STATE/campaign.pid"
STOP_MARKER="$STATE/STOPPED_LOW_DISK"
LOG="$STATE/campaign.log"
PAYLOAD=${1:-$DEFAULT_PAYLOAD}

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export TORCH_NUM_THREADS=4

[ -f "$PAYLOAD" ] || { echo "Missing campaign payload: $PAYLOAD" >&2; exit 2; }
mkdir -p "$STATE"
[ ! -e "$STOP_MARKER" ] || {
  echo "Campaign blocked by $STOP_MARKER; inspect disk and remove it manually before resuming." >&2
  exit 3
}
"$WATCHDOG" --once

shift "$(( $# > 0 ? 1 : 0 ))"
printf '%s starting guarded campaign with %s\n' "$(date -Is)" "$PAYLOAD" >> "$LOG"
setsid "$PYTHON" -u "$LAUNCHER" "$PAYLOAD" "$@" >> "$LOG" 2>&1 &
campaign_pid=$!
printf '%s\n' "$campaign_pid" > "$PID_FILE"
touch "$STATE/CAMPAIGN_STARTED"
"$WATCHDOG" &
watchdog_pid=$!

cleanup() {
  rm -f "$PID_FILE"
  kill "$watchdog_pid" 2>/dev/null || true
  wait "$watchdog_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

if wait "$campaign_pid"; then
  rc=0
else
  rc=$?
fi
printf '%s campaign exited rc=%s\n' "$(date -Is)" "$rc" >> "$LOG"
exit "$rc"
