#!/usr/bin/env bash
# Stop the guarded small-delta campaign before the filesystem reaches 10% free.
set -u

REPO=/home/christian/repositorios/MD_vs_AtomicDisplacement
STATE="$REPO/Comparison/results/ui_real_metrics_derivatives/watchdog_small_delta"
PID_FILE="$STATE/campaign.pid"
LOG="$STATE/disk_guard.log"
STOP_MARKER="$STATE/STOPPED_LOW_DISK"
MIN_FREE_BASIS_POINTS=1200  # 12.00%, leaving headroom above the 10% hard floor.
INTERVAL_SECONDS=5

mkdir -p "$STATE"

check_once() {
  read -r total_bytes _ available_bytes < <(df -P -B1 "$REPO" | awk 'NR == 2 {print $2, $3, $4}')
  free_basis_points=$((available_bytes * 10000 / total_bytes))
  printf -v free_pct '%d.%02d' "$((free_basis_points / 100))" "$((free_basis_points % 100))"

  if [ "$free_basis_points" -ge "$MIN_FREE_BASIS_POINTS" ]; then
    return 0
  fi

  printf '%s LOW DISK: free=%s%%; stopping guarded campaign\n' "$(date -Is)" "$free_pct" >> "$LOG"
  touch "$STOP_MARKER"

  [ -s "$PID_FILE" ] || return 1
  campaign_pid=$(tr -dc '0-9' < "$PID_FILE")
  [ -n "$campaign_pid" ] && kill -0 "$campaign_pid" 2>/dev/null || return 1
  campaign_pgid=$(ps -o pgid= -p "$campaign_pid" | tr -dc '0-9')

  # The guarded launcher uses setsid, so PGID must equal PID. Refuse a broad kill
  # if a stale/reused PID does not have that identity.
  if [ "$campaign_pgid" != "$campaign_pid" ]; then
    printf '%s REFUSED: pid=%s pgid=%s is not the isolated campaign group\n' \
      "$(date -Is)" "$campaign_pid" "${campaign_pgid:-missing}" >> "$LOG"
    return 1
  fi

  kill -TERM -- "-$campaign_pgid" 2>/dev/null || true
  printf '%s SIGTERM sent to campaign process group %s\n' "$(date -Is)" "$campaign_pgid" >> "$LOG"
  return 1
}

if [ "${1:-}" = "--once" ]; then
  if check_once; then
    echo "disk guard OK"
    exit 0
  fi
  echo "disk guard STOP"
  exit 1
fi

printf '%s watchdog started: interval=%ss stop_below=12%%\n' "$(date -Is)" "$INTERVAL_SECONDS" >> "$LOG"
while [ -s "$PID_FILE" ]; do
  check_once || exit 1
  sleep "$INTERVAL_SECONDS"
done

