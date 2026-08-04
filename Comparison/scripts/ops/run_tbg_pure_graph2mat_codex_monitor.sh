#!/usr/bin/env bash
set -u

REPO=/home/christian/repositorios/MD_vs_AtomicDisplacement
CODEX=/home/christian/.local/bin/codex
PROMPT="$REPO/Comparison/scripts/ops/monitor_tbg_pure_graph2mat_codex.md"
RESULTS="$REPO/Comparison/results/tbg_pure_graph2mat/watchdog"
STATUS="$REPO/Comparison/results/tbg_pure_graph2mat/status.json"
LOCK=/tmp/tbg_pure_graph2mat_codex_monitor.lock

mkdir -p "$RESULTS"
exec 9>"$LOCK"
/usr/bin/flock -n 9 || exit 0

if [ -f "$STATUS" ] && /usr/bin/jq -e '.state == "completed" or .state == "gate_failed"' "$STATUS" >/dev/null 2>&1; then
  exit 0
fi

{
  echo "[$(/usr/bin/date --iso-8601=seconds)] TBG pure monitor started"
  /usr/bin/timeout --signal=TERM 45m "$CODEX" exec \
    --ephemeral \
    --color never \
    --sandbox danger-full-access \
    --config 'approval_policy="never"' \
    --config 'model_reasoning_effort="low"' \
    --cd "$REPO" \
    --output-last-message "$RESULTS/last_message.md" \
    - < "$PROMPT"
  rc=$?
  echo "[$(/usr/bin/date --iso-8601=seconds)] TBG pure monitor exited rc=$rc"
} >> "$RESULTS/codex_monitor.log" 2>&1
