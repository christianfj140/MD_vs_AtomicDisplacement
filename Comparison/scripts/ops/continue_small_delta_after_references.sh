#!/usr/bin/env bash
# Validate the reference-only pass, reuse old models, then run prediction/metrics.
set -euo pipefail

REPO=/home/christian/repositorios/MD_vs_AtomicDisplacement
BASE="$REPO/Comparison/results/ui_real_metrics_derivatives"
OLD="$BASE/cross_w90_to_5x5_2delta"
NEW="$BASE/cross_w90_to_5x5_delta_0p0005_0p001"
STATE="$BASE/watchdog_small_delta"
PID_FILE="$STATE/campaign.pid"
HOLD="$STATE/CODEX_HOLD"
LOG="$STATE/continuation.log"
PLAN="$BASE/derivative_campaign_plan_cross_w90_to_5x5_delta_0p0005_0p001.json"
PAYLOAD="$REPO/Comparison/config/ui_cross_w90_to_5x5_delta_0p0005_0p001_payload.json"
LAUNCH="$REPO/Comparison/scripts/ops/run_small_delta_campaign_guarded.sh"

printf '%s waiting for reference-only pass\n' "$(date -Is)" >> "$LOG"
while [ -s "$PID_FILE" ]; do sleep 5; done

valid=0
while IFS= read -r case_id; do
  manifest="$NEW/$case_id/siesta_hamiltonians/derivative_siesta_reference_manifest.json"
  stencil="$NEW/$case_id/derivative_stencil_manifest.json"
  if [ -f "$manifest" ] && [ -f "$stencil" ] &&
     jq -e '.samples_failed == 0' "$manifest" >/dev/null &&
     [ "$(jq -r '.samples_ok' "$manifest")" = "$(jq -r '.sample_count' "$stencil")" ]; then
    valid=$((valid + 1))
  fi
done < <(jq -r '.cases[].id' "$PLAN")

if [ "$valid" -ne 12 ]; then
  printf '%s reference validation failed: valid_cases=%s/12; handing diagnosis to Codex\n' \
    "$(date -Is)" "$valid" >> "$LOG"
  rm -f "$HOLD"
  exit 1
fi

while IFS= read -r case_id; do
  source="$OLD/$case_id/deeph_autograd_model"
  target="$NEW/$case_id/deeph_autograd_model"
  [ -f "$source/train/best_state_dict.pkl" ] || {
    printf '%s missing reusable model: %s\n' "$(date -Is)" "$source" >> "$LOG"
    rm -f "$HOLD"
    exit 1
  }
  [ -e "$target" ] || ln -s "$source" "$target"
done < <(jq -r '.cases[].id' "$PLAN")

printf '%s references valid; 12 DeepH models linked; starting prediction/metrics pass\n' \
  "$(date -Is)" >> "$LOG"
"$LAUNCH" "$PAYLOAD" &
full_wrapper_pid=$!
while [ ! -s "$PID_FILE" ] && kill -0 "$full_wrapper_pid" 2>/dev/null; do sleep 1; done
rm -f "$HOLD"
wait "$full_wrapper_pid"
