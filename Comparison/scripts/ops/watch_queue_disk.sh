#!/bin/bash
# Disk guard for the queued campaigns (derivative regeneration -> vacancy MD generation ->
# vacancy->w90 training). Stops them if free space drops below MIN_FREE_PCT.
#
# NOTE: watch_sweep_disk.sh gets this backwards. `df --output=pcent` reports the percentage
# USED, but that script stores it in free_pct and then tests `free_pct < 10`, which is only
# true when the disk is nearly EMPTY. It has never fired. This computes free = 100 - used.

set -u

PROJECT_ROOT="/home/christian/repositorios/MD_vs_AtomicDisplacement"
LOG="$PROJECT_ROOT/Comparison/results/queue_disk_watchdog.log"
MIN_FREE_PCT=10

# The whole queue, plus the SIESTA workers they spawn (those are what fill the disk).
PID_PATTERN="regenerate_derivative_siesta_references.py|launch_vacancy_dataset_generation.py|queue_vacancy_train.sh|run_cross_structure_sweep_payload.py|run_hamiltonian_derivative_siesta_references.py"

ts() { date '+%Y-%m-%d %H:%M:%S'; }

used_pct=$(df --output=pcent "$PROJECT_ROOT" | tail -1 | tr -dc '0-9')
free_pct=$((100 - used_pct))
avail=$(df -h --output=avail "$PROJECT_ROOT" | tail -1 | tr -d ' ')

pids=$(pgrep -f -- "$PID_PATTERN" | tr '\n' ' ')

if [ "$free_pct" -lt "$MIN_FREE_PCT" ]; then
    echo "$(ts) DISCO BAJO: libre=${free_pct}% (${avail}) < ${MIN_FREE_PCT}% - parando cola: $pids" >> "$LOG"
    if [ -n "$pids" ]; then
        # SIGTERM, nunca -9: que los runners cierren manifests y summaries limpiamente.
        pkill -f -- "$PID_PATTERN"
        sleep 10
        pkill siesta 2>/dev/null
    fi
else
    # Solo se registra el latido cuando hay algo corriendo o poco margen, para no
    # inflar el log con 288 lineas diarias de "todo bien".
    if [ -n "$pids" ] || [ "$free_pct" -lt 20 ]; then
        echo "$(ts) libre=${free_pct}% (${avail}) cola_activa=$(echo "$pids" | wc -w)" >> "$LOG"
    fi
fi
