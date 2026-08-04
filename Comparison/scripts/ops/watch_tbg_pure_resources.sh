#!/usr/bin/env bash
set -u

REPO=/home/christian/repositorios/MD_vs_AtomicDisplacement
UNIT=tbg-pure-graph2mat-campaign.service
LOG="$REPO/Comparison/results/tbg_pure_graph2mat/resource_watchdog.log"
MIN_FREE=12
MAX_CPU=80
MAX_GPU=75

systemctl --user is-active --quiet "$UNIT" || exit 0
used=$(df --output=pcent "$REPO" | tail -1 | tr -dc '0-9')
free=$((100 - used))
cpu=$(sensors 2>/dev/null | awk '/Package id 0:/{gsub(/[+°C]/, "", $4); print int($4); exit}')
gpu=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)

reason=""
[ "$free" -lt "$MIN_FREE" ] && reason="disk_free=${free}%<${MIN_FREE}%"
[ -n "${cpu:-}" ] && [ "$cpu" -ge "$MAX_CPU" ] && reason="cpu=${cpu}C>=${MAX_CPU}C"
[ -n "${gpu:-}" ] && [ "$gpu" -ge "$MAX_GPU" ] && reason="gpu=${gpu}C>=${MAX_GPU}C"
if [ -n "$reason" ]; then
  echo "[$(date --iso-8601=seconds)] stopping $UNIT: $reason" >> "$LOG"
  systemctl --user stop "$UNIT"
fi
