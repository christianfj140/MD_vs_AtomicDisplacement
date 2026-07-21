#!/usr/bin/env bash
# Cron: every 2h. Invokes Claude Code non-interactively to diagnose and
# resume the ui_real_metrics_derivatives campaign if it crashed.
# Runs with --dangerously-skip-permissions by explicit user decision
# (2026-07-17): no one is present to approve tool calls on each cron fire.
set -u

REPO=/home/christian/repositorios/MD_vs_AtomicDisplacement
OUT="$REPO/Comparison/results/ui_real_metrics_derivatives"
PROMPT_FILE="$REPO/Comparison/scripts/ops/claude_fix_derivative_campaign_prompt.txt"
LOG="$OUT/claude_fix.log"
CLAUDE_BIN="/home/christian/.nvm/versions/node/v24.17.0/bin/claude"
# Hard stop: after this instant the cron self-removes and never calls Claude again.
DEADLINE="2026-07-21 12:00:00"

mkdir -p "$OUT"
cd "$REPO" || exit 1

if [ "$(date +%s)" -ge "$(date -d "$DEADLINE" +%s)" ]; then
    echo "=== $(date -Is) deadline $DEADLINE reached — self-removing cron, no Claude call ===" >> "$LOG"
    crontab -l 2>/dev/null | grep -v 'claude_fix_derivative_campaign.sh' | crontab -
    exit 0
fi

echo "=== $(date -Is) claude_fix_derivative_campaign start ===" >> "$LOG"
"$CLAUDE_BIN" -p "$(cat "$PROMPT_FILE")" --dangerously-skip-permissions >> "$LOG" 2>&1
echo "=== $(date -Is) claude_fix_derivative_campaign end (exit=$?) ===" >> "$LOG"
