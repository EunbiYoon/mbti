#!/bin/bash
# Shared conda / CUDA / PYTHONPATH setup for mbti scripts.

: "${ROOT:?ROOT must be set}"

SCRATCH_ROOT="${SCRATCH_ROOT:-$(cd "${ROOT}/.." && pwd)}"
export SCRATCH_ROOT
export MODELS_ROOT="${MODELS_ROOT:-${SCRATCH_ROOT}/models}"
export CONDA_ENV="${CONDA_ENV:-marshal}"
export PRETRAIN="${PRETRAIN:-${MODELS_ROOT}/Qwen3-0.6B}"
export LORA_RANK="${LORA_RANK:-16}"
export LORA_ALPHA="${LORA_ALPHA:-32}"
export LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
export HF_HOME="${HF_HOME:-${SCRATCH_ROOT}/${USER}/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

module load conda/latest cuda/12.6 2>/dev/null || true

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

_scratch_env="${SCRATCH_ROOT}/${USER}/.conda/envs/${CONDA_ENV}"
# USER path may be eunbiyoon_umass_edu under scratch layout
_alt_env="${SCRATCH_ROOT}/eunbiyoon_umass_edu/.conda/envs/${CONDA_ENV}"

if [[ -x "${_scratch_env}/bin/python" ]]; then
  conda activate "${_scratch_env}"
elif [[ -x "${_alt_env}/bin/python" ]]; then
  conda activate "${_alt_env}"
else
  conda activate "${CONDA_ENV}"
fi

export PATH="${CONDA_PREFIX}/bin:${PATH}"
cd "${ROOT}"

echo "=== mbti env: CONDA_PREFIX=${CONDA_PREFIX} PRETRAIN=${PRETRAIN} ===" >&2
if [[ ! -f "${PRETRAIN}/config.json" ]]; then
  echo "WARNING: model missing at ${PRETRAIN}" >&2
fi
