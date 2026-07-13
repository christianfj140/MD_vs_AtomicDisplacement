#!/bin/bash
# Weekend guard for the active sweep: logs status, kills the sweep if disk free% drops below threshold.
# Installed via system crontab (see Comparison/scripts/ops/README_cron.md).

set -u

PROJECT_ROOT="/home/christian/repositorios/MD_vs_AtomicDisplacement"
LOG="$PROJECT_ROOT/Comparison/results/sweep_watchdog.log"
DISK_PATH="$PROJECT_ROOT"
MIN_FREE_PCT=10   # pause sweep if free space drops below this percent
PID_PATTERN="run_g2m_deeph_payload_once.py|mixing_sweep|run_mixing_sweep"

ts() { date '+%Y-%m-%d %H:%M:%S'; }

free_pct=$(df --output=pcent "$DISK_PATH" | tail -1 | tr -dc '0-9')
used_pct=$((100 - free_pct))
free_h=$(df -h --output=avail "$DISK_PATH" | tail -1 | tr -d ' ')

echo "$(ts) disk_free=${free_pct}% (${free_h} avail)" >> "$LOG"

pids=$(pgrep -f -- "$PID_PATTERN" | tr '\n' ' ')
if [ -n "$pids" ]; then
    echo "$(ts) sweep running: pids=$pids" >> "$LOG"
else
    echo "$(ts) no sweep process found" >> "$LOG"
fi

if [ "$free_pct" -lt "$MIN_FREE_PCT" ]; then
    echo "$(ts) LOW DISK ($free_pct% < $MIN_FREE_PCT%) - stopping sweep processes: $pids" >> "$LOG"
    if [ -n "$pids" ]; then
        # ponytail: SIGTERM only, no -9 — let checkpoints/status files flush cleanly
        pkill -f -- "$PID_PATTERN"
    fi
else
    echo "$(ts) disk OK, sweep left running" >> "$LOG"
fi
