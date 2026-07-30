#!/usr/bin/env bash
set -u

REPO=/home/christian/repositorios/MD_vs_AtomicDisplacement
CODEX=/home/christian/.local/bin/codex
PROMPT="$REPO/Comparison/scripts/ops/monitor_graph2mat_n480_bands_codex.md"
RESULTS="$REPO/Comparison/results/graphene_hbn_magic_angle_spectral/watchdog"
MANIFEST="$REPO/Comparison/results/graphene_hbn_magic_angle_spectral/spectra/graph2mat/n480/seed0/tier_b/solver_manifest.json"
LOCK=/tmp/md_vs_atomic_graph2mat_n480_codex_monitor.lock

mkdir -p "$RESULTS"
exec 9>"$LOCK"
/usr/bin/flock -n 9 || exit 0

if [ -f "$MANIFEST" ] && /usr/bin/jq -e '.status == "completed"' "$MANIFEST" >/dev/null 2>&1; then
  exit 0
fi

{
  echo "[$(/usr/bin/date --iso-8601=seconds)] Codex monitor started"
  /usr/bin/timeout --signal=TERM 45m "$CODEX" exec \
    --ephemeral \
    --color never \
    --sandbox workspace-write \
    --config 'approval_policy="never"' \
    --cd "$REPO" \
    --output-last-message "$RESULTS/last_message.md" \
    - < "$PROMPT"
  status=$?
  echo "[$(/usr/bin/date --iso-8601=seconds)] Codex monitor exited with status $status"
} >> "$RESULTS/codex_monitor.log" 2>&1
