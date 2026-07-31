#!/usr/bin/env bash
set -euo pipefail

old_unit=moire-tracked-band-sweep.service
corrected_unit=moire-corrected-n480-band.service
repo=/home/christian/repositorios/MD_vs_AtomicDisplacement
legacy_config="$repo/Comparison/results/graphene_hbn_magic_angle_spectral/spectra/graph2mat/n30/seed0/tier_b_tracked/band_config.json"

while systemctl --user is-active --quiet "$old_unit"; do
  if [[ -f "$legacy_config" ]]; then
    systemctl --user stop "$old_unit"
    break
  fi
  sleep 15
done

systemd-run --user --unit="$corrected_unit" --collect \
  "$repo/.venv/bin/python" \
  "$repo/Comparison/scripts/run_moire_tracked_band_sweep.py" \
  --sizes 480 --corrected-path --backend gpu_cudss

while systemctl --user is-active --quiet "$corrected_unit"; do
  sleep 30
done

manifest="$repo/Comparison/results/graphene_hbn_magic_angle_spectral/spectra/graph2mat/n480/seed0/tier_b_sorted_correct_path/solver_manifest.json"
if ! grep -q '"status": "completed"' "$manifest"; then
  corrected_unit=moire-corrected-n480-band-cpu-fallback.service
  systemd-run --user --unit="$corrected_unit" --collect \
    "$repo/.venv/bin/python" \
    "$repo/Comparison/scripts/run_moire_tracked_band_sweep.py" \
    --sizes 480 --corrected-path
  while systemctl --user is-active --quiet "$corrected_unit"; do
    sleep 30
  done
fi
grep -q '"status": "completed"' "$manifest"

if "$repo/.venv/bin/python" \
  "$repo/Comparison/scripts/run_graphene_unfolded_spectrum.py" \
  --training-size 480 --seed 0 --num-bands 16 --points-per-segment 16 \
  --backend gpu_cudss; then
  exit 0
fi

exec "$repo/.venv/bin/python" \
  "$repo/Comparison/scripts/run_graphene_unfolded_spectrum.py" \
  --training-size 480 --seed 0 --num-bands 16 --points-per-segment 16
