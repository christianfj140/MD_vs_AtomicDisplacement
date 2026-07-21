#!/usr/bin/env bash
# Disk guard for the ui_real_metrics_derivatives campaign.
# Cron: every 5 min. Pure bash/coreutils — never calls Claude, costs no credit.
# Failure diagnosis + resume-in-place is handled by a separate claude.ai
# routine (not this script) so a crash gets root-caused before restarting,
# instead of being blindly relaunched.
set -u

REPO=/home/christian/repositorios/MD_vs_AtomicDisplacement
OUT="$REPO/Comparison/results/ui_real_metrics_derivatives"
LOG="$OUT/monitor.log"
CAMPAIGN_PATTERN="launch_ui_real_metrics_derivatives|train_deeph_autograd_models"
MIN_FREE_PCT=10
MIN_GPU_FREE_MIB=6000  # a DeepH/graph2mat autograd prediction needs several GB

mkdir -p "$OUT"
cd "$REPO" || exit 1

avail_pct=$(df --output=pcent "$REPO" | tail -1 | tr -dc '0-9')
free_pct=$((100 - avail_pct))
size=$(du -sh "$OUT" 2>/dev/null | cut -f1)
done_samples=$(find "$OUT" -maxdepth 6 -name '0_NORMAL_EXIT' 2>/dev/null | wc -l)
campaign_pids=$(pgrep -f "$CAMPAIGN_PATTERN" | while read -r pid; do
    case "$(ps -p "$pid" -o comm= | tr -d ' ')" in
        python|python3) echo "$pid" ;;
    esac
done)

echo "$(date -Is) free=${free_pct}% out_size=${size:-0} siesta_done=${done_samples} campaign_pids=${campaign_pids:-none}" >> "$LOG"

# --- Essential guard: stop before free space drops below 10%. ---
if [ "$free_pct" -lt "$MIN_FREE_PCT" ]; then
    if [ -n "$campaign_pids" ]; then
        echo "$(date -Is) LOW DISK: free ${free_pct}% < ${MIN_FREE_PCT}% — stopping campaign" >> "$LOG"
        while read -r campaign_pid; do
            campaign_pgid=$(ps -p "$campaign_pid" -o pgid= | tr -dc '0-9')
            # Each launcher/trainer owns its process group, including its children.
            kill -- "-${campaign_pgid:-$campaign_pid}" 2>/dev/null
        done <<< "$campaign_pids"
        sleep 5
    fi
    touch "$OUT/STOPPED_LOW_DISK"
    echo "$(date -Is) campaign halted, STOPPED_LOW_DISK marker set — will not auto-restart until disk freed and marker removed" >> "$LOG"
    exit 0
fi

# Disk recovered: clear a stale marker so the claude.ai routine knows it can resume.
if [ -f "$OUT/STOPPED_LOW_DISK" ]; then
    echo "$(date -Is) disk recovered (free ${free_pct}%) — clearing STOPPED_LOW_DISK" >> "$LOG"
    rm -f "$OUT/STOPPED_LOW_DISK"
fi

# --- GPU gate: the derivative prediction phase runs autograd on the GPU. If
# --- a parallel job (e.g. the vacancy DeepH training) is hogging it, a restart
# --- would just CUDA-OOM in a loop. This sets GPU_BUSY_HOLD so the claude.ai
# --- routine holds off relaunching until memory frees up. Non-destructive:
# --- it never kills anything (unlike disk), a full GPU just means "wait".
gpu_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | sort -n | head -1)
if [ -n "$gpu_free" ]; then
    if [ "$gpu_free" -lt "$MIN_GPU_FREE_MIB" ]; then
        if [ ! -f "$OUT/GPU_BUSY_HOLD" ]; then
            echo "$(date -Is) GPU busy: ${gpu_free}MiB free < ${MIN_GPU_FREE_MIB}MiB — set GPU_BUSY_HOLD (hold derivative relaunch)" >> "$LOG"
        fi
        touch "$OUT/GPU_BUSY_HOLD"
    elif [ -f "$OUT/GPU_BUSY_HOLD" ]; then
        echo "$(date -Is) GPU free again (${gpu_free}MiB) — clearing GPU_BUSY_HOLD" >> "$LOG"
        rm -f "$OUT/GPU_BUSY_HOLD"
    fi
fi
