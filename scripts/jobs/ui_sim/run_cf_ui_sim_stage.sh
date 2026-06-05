#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/jobs/ui_sim/run_cf_ui_sim_stage.sh configs/ui_sim_ar_i2v_action_node.yaml <logdir>
#   ELEMENT_LOSS=1 ELEMENT_LOSS_CACHE=/work/$USER/data/graph_elements_fileExp_compact_training \
#     OCR_CACHE=/work/$USER/data/ocr_fileExp_compact_training \
#     bash scripts/jobs/ui_sim/run_cf_ui_sim_stage.sh configs/ui_sim_ar_i2v_action_node.yaml <logdir>
#
# Element loss is intentionally pluggable. Leave ELEMENT_LOSS unset/0 for a
# normal graph-node finetuning run; set ELEMENT_LOSS=1 to mirror the DFoT
# node-conditioned finetuning launcher.

CONFIG_PATH="${1:?usage: run_cf_ui_sim_stage.sh <config.yaml> <logdir> [extra train.py args...]}"
LOGDIR="${2:?usage: run_cf_ui_sim_stage.sh <config.yaml> <logdir> [extra train.py args...]}"
shift 2

: "${WAN_MODEL_DIR:?Set WAN_MODEL_DIR to the HPC directory containing Wan2.1 weights.}"
: "${UI_SIM_CF_LATENT_CACHE:?Set UI_SIM_CF_LATENT_CACHE to the HPC UI latent cache.}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}/external/cf:${PYTHONPATH:-}"

NNODES="${NNODES:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
RDZV_ID="${RDZV_ID:-$RANDOM}"
RDZV_BACKEND="${RDZV_BACKEND:-c10d}"
RDZV_ENDPOINT="${RDZV_ENDPOINT:-${MASTER_ADDR:-127.0.0.1}:${MASTER_PORT:-29500}}"

ELEMENT_LOSS="${ELEMENT_LOSS:-0}"
ELEMENT_LOSS_CACHE="${ELEMENT_LOSS_CACHE:-${UI_SIM_ELEMENT_CACHE:-}}"
OCR_CACHE="${OCR_CACHE:-}"
ELEMENT_BOOST="${ELEMENT_BOOST:-20.0}"
TEXT_BOOST="${TEXT_BOOST:-50.0}"
TEXT_MIN_CONFIDENCE="${TEXT_MIN_CONFIDENCE:-0.5}"
TEXT_PADDING_PX="${TEXT_PADDING_PX:-2}"

OVERRIDES=()
if [[ "${ELEMENT_LOSS}" == "1" || "${ELEMENT_LOSS}" == "true" ]]; then
  if [[ -z "${ELEMENT_LOSS_CACHE}" ]]; then
    echo "ERROR: ELEMENT_LOSS=1 requires ELEMENT_LOSS_CACHE or UI_SIM_ELEMENT_CACHE." >&2
    exit 1
  fi
  OVERRIDES+=(
    element_loss.enabled=true
    "element_loss.cache_dir=${ELEMENT_LOSS_CACHE}"
    "element_loss.text_cache_dir=${OCR_CACHE}"
    "element_loss.element_boost=${ELEMENT_BOOST}"
    "element_loss.text_boost=${TEXT_BOOST}"
    "element_loss.text_min_confidence=${TEXT_MIN_CONFIDENCE}"
    "element_loss.text_padding_px=${TEXT_PADDING_PX}"
  )
else
  OVERRIDES+=(element_loss.enabled=false)
fi

torchrun \
  --nnodes="${NNODES}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --rdzv_id="${RDZV_ID}" \
  --rdzv_backend="${RDZV_BACKEND}" \
  --rdzv_endpoint="${RDZV_ENDPOINT}" \
  train.py \
  --config_path "${CONFIG_PATH}" \
  --logdir "${LOGDIR}" \
  --disable-wandb \
  "${OVERRIDES[@]}" \
  "$@"
