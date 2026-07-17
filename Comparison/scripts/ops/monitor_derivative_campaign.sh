#!/usr/bin/env bash
# Disk guard + progress log for the ui_real_metrics_derivatives campaign.
# Cron: */5 * * * *  — logs progress and stops the campaign before the disk fills.
set -u

OUT=/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/results/ui_real_metrics_derivatives
LOG="$OUT/monitor.log"
MIN_FREE_GB=40

mkdir -p "$OUT"
free_gb=$(df -BG --output=avail "$OUT" | tail -1 | tr -dc '0-9')
size=$(du -sh "$OUT" 2>/dev/null | cut -f1)
done_samples=$(find "$OUT" -maxdepth 5 -name '0_NORMAL_EXIT' 2>/dev/null | wc -l)
launcher_pids=$(pgrep -f launch_ui_real_metrics_derivatives | tr '\n' ' ')
siesta_pids=$(pgrep -x siesta | tr '\n' ' ')
echo "$(date -Is) free=${free_gb}G out_size=${size:-0} siesta_done=${done_samples} launcher=[${launcher_pids:-dead}] siesta=[${siesta_pids:-idle}]" >> "$LOG"

if [ "${free_gb:-0}" -lt "$MIN_FREE_GB" ] && [ -n "$launcher_pids" ]; then
    echo "$(date -Is) LOW DISK: free ${free_gb}G < ${MIN_FREE_GB}G — stopping campaign" >> "$LOG"
    # ponytail: SIGTERM launcher then its siesta workers; no retry logic — cron re-fires in 5 min.
    kill $launcher_pids 2>/dev/null
    sleep 5
    pkill -x siesta 2>/dev/null
    touch "$OUT/STOPPED_LOW_DISK"
fi
