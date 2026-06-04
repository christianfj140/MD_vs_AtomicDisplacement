#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

mkdir -p Comparison/results/weekend_queue/logs
LOG="Comparison/results/weekend_queue/logs/weekend_queue_$(date +%Y%m%d_%H%M%S).log"

setsid -f .venv/bin/python Comparison/scripts/g2m_deeph_weekend_queue.py \
  --queue iid600,iid1000 \
  --poll-seconds 60 \
  >"${LOG}" 2>&1

echo "Weekend queue started."
echo "Log: ${REPO_ROOT}/${LOG}"
