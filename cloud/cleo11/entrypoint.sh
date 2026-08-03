#!/usr/bin/env bash
set -euo pipefail

PROFILE="${CLEO11_PROFILE:-full}"
CONFIG="${CLEO11_CONFIG:-configs/cleo11_135m.toml}"
NPROC="${CLEO11_NPROC:-1}"
PREPARE="${CLEO11_PREPARE:-1}"
DEVICE="${CLEO11_DEVICE:-cuda}"
MANIFEST="${CLEO11_MANIFEST:-data/cleo11/processed/manifest.json}"

mkdir -p data/cleo11/processed artifacts/cleo11

if [[ "${PREPARE}" == "1" && ! -f "${MANIFEST}" ]]; then
  echo "Prepared shards missing; running cleo11-prepare profile=${PROFILE}"
  cleo-1 cleo11-prepare --cleo11-config "${CONFIG}" --profile "${PROFILE}"
elif [[ "${PREPARE}" == "1" ]]; then
  echo "Using existing prepared manifest at ${MANIFEST}"
fi

EXTRA_ARGS=()
if [[ -n "${CLEO11_MAX_STEPS:-}" ]]; then
  EXTRA_ARGS+=(--max-steps "${CLEO11_MAX_STEPS}")
fi
if [[ -n "${CLEO11_RESUME:-}" ]]; then
  EXTRA_ARGS+=(--resume "${CLEO11_RESUME}")
fi
if [[ -n "${CLEO11_MICROBATCH:-}" ]]; then
  EXTRA_ARGS+=(--microbatch-size "${CLEO11_MICROBATCH}")
fi

echo "Starting Cleo 1.1 pretrain with torchrun nproc=${NPROC} config=${CONFIG}"
exec torchrun --standalone --nproc_per_node="${NPROC}" -m cleo1.cleo11.pretrain_entry \
  --cleo11-config "${CONFIG}" \
  --device "${DEVICE}" \
  "${EXTRA_ARGS[@]}"
