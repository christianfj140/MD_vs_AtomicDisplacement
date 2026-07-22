#!/usr/bin/env bash
# End-to-end smoke for the graphene/hBN bilayer -> twisted-moire cross-test.
# Small on purpose (~30 MD snapshots/stacking, few epochs, limit 1 moire sample).
# Proves the flow works end to end; not a paper-ready run.
#
# Usage:  bash Comparison/scripts/ops/smoke_bilayer_moire_pipeline.sh [--skip-md] [--skip-train]
# Requires: siesta on PATH, repo .venv, graph2mat/deeph installed for Phase 3/5.
set -euo pipefail

REPO=/home/christian/repositorios/MD_vs_AtomicDisplacement
cd "$REPO"
# shellcheck disable=SC1091
source .venv/bin/activate
PY=.venv/bin/python
SKIP_MD=0; SKIP_TRAIN=0
for arg in "$@"; do
  case "$arg" in
    --skip-md) SKIP_MD=1 ;;
    --skip-train) SKIP_TRAIN=1 ;;
  esac
done

say() { printf '\n=== %s ===\n' "$1"; }

say "Phase 0: material bundles resolve with 3 species + shared basis"
$PY - <<'PY'
import sys; sys.path.insert(0, "shared")
from material_presets import resolve_material_bundle
hashes = {}
for s in ("graphene_hBN_AA", "graphene_hBN_AB1", "graphene_hBN_AB2"):
    v = resolve_material_bundle({"material": {"preset": s}}).validated
    assert sorted(sp.label for sp in v.species) == ["B", "C", "N"], s
    hashes[s] = v.basis_file_sha256
assert len(set(map(str, hashes.values()))) == 1, "basis hashes differ"
print("Phase 0 OK")
PY

if [ "$SKIP_MD" -eq 0 ]; then
  say "Phase 1: per-stacking MD datasets (SIESTA)"
  for S in AA AB1 AB2; do
    rm -rf "Comparison/datasets/graphene_hBN_${S}_md30" "Comparison/results/graphene_hBN_${S}_md30"
    $PY Comparison/scripts/run_g2m_deeph_payload_once.py \
      "Comparison/config/graphene_hbn_${S}_md30_payload.json" \
      --status-json "/tmp/${S}_md_status.json" \
      --manifest-json "/tmp/${S}_md_manifest.json" --poll-seconds 15
  done
fi

say "Phase 2: fuse AA/AB1/AB2 into one train pool"
# The sweep nests each MD dataset under its recipe slug (graphene_hbn_<s>_md30).
$PY Comparison/scripts/build_graphene_hbn_bilayer_train_dataset.py \
  --source-dataset Comparison/datasets/graphene_hBN_AA_md30/graphene_hbn_aa_md30 \
  --source-dataset Comparison/datasets/graphene_hBN_AB1_md30/graphene_hbn_ab1_md30 \
  --source-dataset Comparison/datasets/graphene_hBN_AB2_md30/graphene_hbn_ab2_md30 \
  --output-root Comparison/datasets/graphene_hBN_bilayer_train --overwrite

if [ "$SKIP_TRAIN" -eq 0 ]; then
  say "Phase 3: train G2M + DeepH on the fused pool (see docs for the payload)"
  echo "NOTE: run your small-epoch snapshot-scaling payload here with"
  echo "      dataset_root=Comparison/datasets/graphene_hBN_bilayer_train"
  echo "      and persist checkpoints to Comparison/results/graphene_hBN_bilayer_train_models/{graph2mat/training,deeph/train}"
fi

say "Phase 4: twisted-moire target (SIESTA)"
$PY Comparison/scripts/build_graphene_hbn_moire_target.py \
  --approximant 2 --commensurate-angle 1,2 --limit 1 --overwrite \
  --output-root Comparison/datasets/graphene_hBN_moire_22deg --siesta-command siesta

say "Phase 5a: regenerate payload + preview compatibility"
$PY Comparison/scripts/ops/build_cross_predict_metrics_payload.py \
  --bilayer-output Comparison/config/graphene_hbn_bilayer_to_moire_predict_metrics_payload.json \
  --bilayer-source Comparison/datasets/graphene_hBN_bilayer_train \
  --bilayer-target Comparison/datasets/graphene_hBN_moire_22deg >/dev/null
$PY Comparison/scripts/run_cross_structure_sweep_payload.py \
  Comparison/config/graphene_hbn_bilayer_to_moire_predict_metrics_payload.json --action preview

say "Phase 5b: predict_metrics (needs Phase 3 checkpoints staged)"
$PY Comparison/scripts/run_cross_structure_sweep_payload.py \
  Comparison/config/graphene_hbn_bilayer_to_moire_predict_metrics_payload.json \
  --action predict_metrics \
  --output-root Comparison/results/ml_vs_siesta_cross_structure_bilayer_moire

say "Smoke complete. Open the UI (Cross testing -> bicapa->moiré) to see the curve."
