#!/usr/bin/env bash
set -u

REPO=/home/christian/repositorios/MD_vs_AtomicDisplacement
CODEX=/home/christian/.local/bin/codex
PROMPT="$REPO/Comparison/scripts/ops/monitor_small_delta_codex.md"
STATE="$REPO/Comparison/results/ui_real_metrics_derivatives/watchdog_small_delta"
RESULTS="$REPO/Comparison/results/ui_real_metrics_derivatives/cross_w90_to_5x5_delta_0p0005_0p001"
LOCK=/tmp/md_vs_atomic_small_delta_codex_monitor.lock

# Stay completely inert until the guarded launcher has started the campaign.
[ -f "$STATE/CAMPAIGN_STARTED" ] || exit 0
[ ! -f "$STATE/CODEX_HOLD" ] || exit 0

mkdir -p "$STATE"
exec 9>"$LOCK"
/usr/bin/flock -n 9 || exit 0

# Once all 12 cases have both metric manifests, recurring Codex calls add no value.
if [ -d "$RESULTS" ]; then
  graph2mat_done=$(find "$RESULTS" -path '*/derivative_metrics/graph2mat/manifest.json' -type f | wc -l)
  deeph_done=$(find "$RESULTS" -path '*/derivative_metrics/deeph/manifest.json' -type f | wc -l)
  [ "$graph2mat_done" -eq 12 ] && [ "$deeph_done" -eq 12 ] && exit 0
fi

{
  echo "[$(/usr/bin/date --iso-8601=seconds)] small-delta Codex monitor started"
  /usr/bin/timeout --signal=TERM 45m "$CODEX" exec \
    --ephemeral \
    --color never \
    --sandbox danger-full-access \
    --config 'approval_policy="never"' \
    --config 'model_reasoning_effort="low"' \
    --cd "$REPO" \
    --output-last-message "$STATE/last_message.md" \
    - < "$PROMPT"
  rc=$?
  echo "[$(/usr/bin/date --iso-8601=seconds)] small-delta Codex monitor exited rc=$rc"
} >> "$STATE/codex_monitor.log" 2>&1
