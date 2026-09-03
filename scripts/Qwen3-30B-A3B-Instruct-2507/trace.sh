#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "${repo_root}"

model_id="qwen3_moe_30b_a3b_instruct_2507"
calib_dataset="${CALIB_DATASET:-c4}"
nsamples="${NSAMPLES:-128}"
seqlen="${SEQLEN:-2048}"
seed="${SEED:-0}"

python -m mxmoe.quant.moe_tracer \
  --model "${model_id}" \
  --calib_dataset "${calib_dataset}" \
  --nsamples "${nsamples}" \
  --seqlen "${seqlen}" \
  --seed "${seed}" \
  --trace_gate
