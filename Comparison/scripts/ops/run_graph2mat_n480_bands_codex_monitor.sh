#!/usr/bin/env bash
set -u

REPO=/home/christian/repositorios/MD_vs_AtomicDisplacement
CODEX=/home/christian/.local/bin/codex
PROMPT="$REPO/Comparison/scripts/ops/monitor_graph2mat_n480_bands_codex.md"
RESULTS="$REPO/Comparison/results/graphene_hbn_magic_angle_spectral/watchdog"
FINAL_STATUS="$REPO/Comparison/results/graphene_hbn_magic_angle_spectral/spectra/projected_followup_status.json"
LOCK=/tmp/md_vs_atomic_graph2mat_n480_codex_monitor.lock

mkdir -p "$RESULTS"
exec 9>"$LOCK"
/usr/bin/flock -n 9 || exit 0

if [ -f "$FINAL_STATUS" ] && /usr/bin/jq -e '.status == "completed"' "$FINAL_STATUS" >/dev/null 2>&1; then
  exit 0
fi

{
  echo "[$(/usr/bin/date --iso-8601=seconds)] Codex monitor started"
  /usr/bin/timeout --signal=TERM 45m "$CODEX" exec \
    --ephemeral \
    --color never \
    --sandbox danger-full-access \
    --config 'approval_policy="never"' \
    --config 'model_reasoning_effort="low"' \
    --cd "$REPO" \
    --output-last-message "$RESULTS/last_message.md" \
    - < "$PROMPT"
  status=$?
  echo "[$(/usr/bin/date --iso-8601=seconds)] Codex monitor exited with status $status"
} >> "$RESULTS/codex_monitor.log" 2>&1
