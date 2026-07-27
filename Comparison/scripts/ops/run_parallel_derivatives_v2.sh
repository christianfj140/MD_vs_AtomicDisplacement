#!/usr/bin/env bash
# Reanuda de forma segura el sweep de derivadas cross w90 -> 5x5.
set -euo pipefail

REPO="/home/christian/repositorios/MD_vs_AtomicDisplacement"
PY="$REPO/.venv/bin/python"
LAUNCH="$REPO/Comparison/scripts/ops/launch_ui_real_metrics_derivatives.py"
TRAIN="$REPO/Comparison/scripts/ops/train_deeph_autograd_models.py"
PAYLOAD="$REPO/Comparison/config/ui_cross_w90_to_5x5_2delta_payload.json"
LOGDIR="/tmp/claude-1003/-home-christian-repositorios-MD-vs-AtomicDisplacement/d70366fa-8fa5-4548-aea0-743a9a447a8a/scratchpad"
WLOG="$LOGDIR/derivatives_parallel_v2.log"

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

CASES=(
  iid20 iid30 iid100 iid150 iid200 iid300
  iid50 iid60 iid80 iid90 iid400 iid500
)

log(){ echo "$(date -Is) $*" >> "$WLOG"; }

case_done(){
  local id="$1"
  local base="$REPO/Comparison/results/ui_real_metrics_derivatives/cross_w90_to_5x5_2delta/cross_graphene__graphene_w90_scale_${id}__to__graphene_5x5__graphene_5x5_scale_${id}"
  [[ -f "$base/derivative_workflow_manifest.json" &&
     -f "$base/derivative_metrics/graph2mat/manifest.json" &&
     -f "$base/derivative_metrics/deeph/manifest.json" ]]
}

log "v2 iniciado. Reponiendo modelos DeepH autograd ausentes."
"$PY" -u "$TRAIN" "$PAYLOAD" >> "$WLOG" 2>&1
log "Modelos DeepH autograd listos."

# Los reintentos anteriores dejaron referencias SIESTA en curso. Sus resultados
# son reutilizables; esperar evita que dos procesos escriban el mismo caso.
while pgrep -f "[l]aunch_ui_real_metrics_derivatives.py|[r]un_hamiltonian_derivative_siesta_references.py" >/dev/null; do
  log "Esperando calculos de derivadas anteriores."
  sleep 60
done
"$PY" -u "$LAUNCH" "$PAYLOAD" --plots-only >> "$WLOG" 2>&1

for id in "${CASES[@]}"; do
  if case_done "$id"; then
    log "$id completo en Graph2Mat y DeepH, salto."
    continue
  fi
  payload="$REPO/Comparison/config/ui_cross_w90_to_5x5_2delta_${id}_solo_payload.json"
  if [[ ! -f "$payload" ]]; then
    log "AVISO: no existe payload $payload, salto $id"
    continue
  fi
  caselog="$LOGDIR/derivatives_solo_${id}.log"
  completed=false
  for attempt in 1 2 3; do
    log "iniciando $id (intento $attempt/3)."
    if "$PY" -u "$LAUNCH" "$payload" >> "$caselog" 2>&1 && case_done "$id"; then
      completed=true
      log "$id completado."
      break
    fi
    log "AVISO: $id fallo en el intento $attempt/3; se reintentaran solo los artefactos ausentes o fallidos."
    sleep 30
  done
  if [[ "$completed" != true ]]; then
    log "ERROR: $id fallo tres veces; se detiene el sweep para conservar el diagnostico."
    exit 1
  fi
done

log "v2: TODOS los casos de derivadas terminados."
