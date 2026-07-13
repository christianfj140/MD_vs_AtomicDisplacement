#!/bin/bash
# Guards the active mixing sweep (launched via /api/mixing/launch on pipeline_ui.py):
#   - disk < MIN_FREE_PCT free -> kill the sweep outright (no restart).
#   - a new OOM-kill is observed -> kill the sweep, lower both training
#     parallelism knobs by STEP (no floor other than 1), then RESUME only the
#     (model, permutation) combos that never completed via
#     resume_mixing_sweep_oom.py (no re-materialization, no re-training of
#     what already succeeded).
#   - if parallelism is already at 1 for both models and another OOM happens,
#     give up: kill everything, do not resume, leave it for a human.

set -u

PROJECT_ROOT="/home/christian/repositorios/MD_vs_AtomicDisplacement"
PAYLOAD="$PROJECT_ROOT/Comparison/config/ml_vs_siesta_mixing_sweep_20_1000_stratified_per_structure_payload.json"
RESUME_SCRIPT="$PROJECT_ROOT/Comparison/scripts/ops/resume_mixing_sweep_oom.py"
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"
LOG="$PROJECT_ROOT/Comparison/results/mixing_sweep_watchdog.log"
STATE_FILE="$PROJECT_ROOT/Comparison/results/mixing_sweep_watchdog_state.json"
DISK_PATH="$PROJECT_ROOT"
MIN_FREE_PCT=10
STEP=2
PID_PATTERN="deeph-train|deeph-preprocess|deeph-inference|graph2mat.*train|run_mixing_sweep|mixing_sweep_runner"
CGROUP_MEMORY_EVENTS="/sys/fs/cgroup/user.slice/user-$(id -u).slice/memory.events"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "$(ts) $1" >> "$LOG"; }

# ── disk guard ──────────────────────────────────────────────────────────────
free_pct=$(df --output=pcent "$DISK_PATH" | tail -1 | tr -dc '0-9')
free_h=$(df -h --output=avail "$DISK_PATH" | tail -1 | tr -d ' ')
log "disk_free=${free_pct}% (${free_h} avail)"

pids=$(pgrep -f -- "$PID_PATTERN" | tr '\n' ' ')
if [ -n "$pids" ]; then
    log "sweep running: pids=$pids"
else
    log "no sweep process found"
fi

if [ "$free_pct" -lt "$MIN_FREE_PCT" ]; then
    log "LOW DISK (${free_pct}% < ${MIN_FREE_PCT}%) - killing sweep outright, no restart. pids=$pids"
    [ -n "$pids" ] && pkill -f -- "$PID_PATTERN"
    exit 0
fi

# ── OOM guard ───────────────────────────────────────────────────────────────
# oom_kill counts every OOM kill for the whole user slice, including processes
# unrelated to this sweep (shared machine). We only care about NEW kills since
# this watchdog started, so the first-ever run just records the baseline and
# never acts on it.
oom_count=0
if [ -r "$CGROUP_MEMORY_EVENTS" ]; then
    oom_count=$(awk '/^oom_kill /{print $2}' "$CGROUP_MEMORY_EVENTS")
fi

g2m_parallel=7
deeph_parallel=5
gave_up=0
if [ ! -f "$STATE_FILE" ]; then
    log "first run: recording oom baseline=${oom_count}, no action taken"
    python3 -c "
import json
json.dump({'last_oom_count': $oom_count, 'g2m_parallel': $g2m_parallel, 'deeph_parallel': $deeph_parallel, 'gave_up': False}, open('$STATE_FILE', 'w'), indent=1)
"
    exit 0
fi

last_oom_count=$(python3 -c "import json; print(json.load(open('$STATE_FILE')).get('last_oom_count', 0))" 2>/dev/null || echo 0)
g2m_parallel=$(python3 -c "import json; print(json.load(open('$STATE_FILE')).get('g2m_parallel', 7))" 2>/dev/null || echo 7)
deeph_parallel=$(python3 -c "import json; print(json.load(open('$STATE_FILE')).get('deeph_parallel', 5))" 2>/dev/null || echo 5)
gave_up=$(python3 -c "import json; print(int(json.load(open('$STATE_FILE')).get('gave_up', False)))" 2>/dev/null || echo 0)

log "oom_kill_count=${oom_count} last_seen=${last_oom_count} current_parallel(g2m=${g2m_parallel},deeph=${deeph_parallel}) gave_up=${gave_up}"

if [ "$gave_up" -eq 1 ]; then
    log "watchdog already gave up on a previous OOM at parallelism=1; taking no further action. Remove $STATE_FILE to reset."
    exit 0
fi

if [ "$oom_count" -gt "$last_oom_count" ]; then
    log "NEW OOM KILL DETECTED (${last_oom_count} -> ${oom_count})"

    if [ "$g2m_parallel" -le 1 ] && [ "$deeph_parallel" -le 1 ]; then
        log "OOM at parallelism=1 for both models - giving up, killing sweep for good."
        [ -n "$pids" ] && pkill -f -- "$PID_PATTERN"
        python3 -c "
import json
json.dump({'last_oom_count': $oom_count, 'g2m_parallel': $g2m_parallel, 'deeph_parallel': $deeph_parallel, 'gave_up': True}, open('$STATE_FILE', 'w'), indent=1)
"
        exit 0
    fi

    new_g2m=$((g2m_parallel - STEP))
    new_deeph=$((deeph_parallel - STEP))
    [ "$new_g2m" -lt 1 ] && new_g2m=1
    [ "$new_deeph" -lt 1 ] && new_deeph=1

    log "lowering parallelism: g2m ${g2m_parallel} -> ${new_g2m}, deeph ${deeph_parallel} -> ${new_deeph}"
    log "killing current sweep processes: $pids"
    [ -n "$pids" ] && pkill -f -- "$PID_PATTERN"
    sleep 10

    log "resuming only incomplete (model, permutation) combos via resume_mixing_sweep_oom.py (background, hours-long)"
    resume_log="$PROJECT_ROOT/Comparison/results/mixing_sweep_resume_$(date +%Y%m%d_%H%M%S).log"
    nohup "$VENV_PYTHON" "$RESUME_SCRIPT" \
        --payload "$PAYLOAD" \
        --g2m-parallel "$new_g2m" \
        --deeph-parallel "$new_deeph" \
        --apply > "$resume_log" 2>&1 &
    disown
    log "resume launched in background, PID=$!, log=${resume_log}"

    python3 -c "
import json
json.dump({'last_oom_count': $oom_count, 'g2m_parallel': $new_g2m, 'deeph_parallel': $new_deeph, 'gave_up': False}, open('$STATE_FILE', 'w'), indent=1)
"
    log "state updated: last_oom_count=${oom_count} g2m_parallel=${new_g2m} deeph_parallel=${new_deeph}"
else
    # keep state file's parallel values in sync even with no new OOM
    python3 -c "
import json
json.dump({'last_oom_count': $oom_count, 'g2m_parallel': $g2m_parallel, 'deeph_parallel': $deeph_parallel, 'gave_up': False}, open('$STATE_FILE', 'w'), indent=1)
"
fi
