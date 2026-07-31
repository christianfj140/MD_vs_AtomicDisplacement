#!/usr/bin/env bash
set -euo pipefail

old_unit=moire-tracked-band-sweep.service
corrected_unit=moire-corrected-band-sweep.service
repo=/home/christian/repositorios/MD_vs_AtomicDisplacement

while systemctl --user is-active --quiet "$old_unit"; do
  for cmdline in /proc/[0-9]*/cmdline; do
    command=$(tr '\0' ' ' 2>/dev/null < "$cmdline" || true)
    if [[ "$command" == *julia*graph2mat/n30/seed0/tier_b_tracked* ]]; then
      systemctl --user stop "$old_unit"
      break 2
    fi
  done
  sleep 15
done

systemd-run --user --unit="$corrected_unit" --collect \
  "$repo/.venv/bin/python" \
  "$repo/Comparison/scripts/run_moire_tracked_band_sweep.py"

while systemctl --user is-active --quiet "$corrected_unit"; do
  sleep 30
done

manifest="$repo/Comparison/results/graphene_hbn_magic_angle_spectral/spectra/graph2mat/n30/seed0/tier_b_sorted_correct_path/solver_manifest.json"
grep -q '"status": "completed"' "$manifest"

exec "$repo/.venv/bin/python" \
  "$repo/Comparison/scripts/run_graphene_unfolded_spectrum.py" \
  --training-size 30 --seed 0 --num-bands 16 --points-per-segment 8
