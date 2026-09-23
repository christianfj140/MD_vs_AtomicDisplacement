#!/usr/bin/env bash
set -u

REPO=/home/christian/repositorios/MD_vs_AtomicDisplacement
CODEX=/home/christian/.local/bin/codex
PROMPT="$REPO/Comparison/scripts/ops/monitor_md_label_budget_6x6_codex.md"
RESULTS="$REPO/Comparison/results/dataset_design_curves_v1/md_label_budget_6x6"
WATCHDOG="$RESULTS/watchdog"
FINAL="$RESULTS/final_summary.json"
LOCK=/tmp/md_label_budget_6x6_codex_monitor.lock

mkdir -p "$WATCHDOG"
exec 9>"$LOCK"
/usr/bin/flock -n 9 || exit 0
[ ! -f "$FINAL" ] || exit 0

{
  echo "[$(/usr/bin/date --iso-8601=seconds)] MD label-budget Codex monitor started"
  /usr/bin/timeout --signal=TERM 45m "$CODEX" exec \
    --ephemeral \
    --color never \
    --sandbox danger-full-access \
    --config 'approval_policy="never"' \
    --config 'model_reasoning_effort="low"' \
    --cd "$REPO" \
    --output-last-message "$WATCHDOG/last_message.md" \
    - < "$PROMPT"
  status=$?
  echo "[$(/usr/bin/date --iso-8601=seconds)] MD label-budget Codex monitor exited status=$status"
} >> "$WATCHDOG/codex_monitor.log" 2>&1
