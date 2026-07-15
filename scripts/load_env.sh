#!/bin/bash
# Thin wrapper so marshal conda activate hooks don't error in this repo.
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/workspace/eunbiyoon_umass_edu-paper}"
export SCRATCH_ROOT
export MODELS_ROOT="${MODELS_ROOT:-${SCRATCH_ROOT}/models}"
export PRETRAIN="${PRETRAIN:-${MODELS_ROOT}/Qwen3-0.6B}"
export LORA_RANK="${LORA_RANK:-16}"
export LORA_ALPHA="${LORA_ALPHA:-32}"
export LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
