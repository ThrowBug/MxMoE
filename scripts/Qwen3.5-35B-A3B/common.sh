#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "${repo_root}"

model="${MODEL:-Qwen/Qwen3.5-35B-A3B}"
model_id="qwen3_5_moe_35b_a3b"
nsamples="${NSAMPLES:-128}"
seqlen="${SEQLEN:-2048}"
seed="${SEED:-0}"
attn_wbits="${ATTN_WBITS:-4}"
linear_attn_wbits="${LINEAR_ATTN_WBITS:-4}"
shared_expert_wbits="${SHARED_EXPERT_WBITS:-4}"
nominal_bits="${NOMINAL_BITS:-2.0}"
recipe="C4-Seed${seed}-N${nsamples}-L${seqlen}-A${attn_wbits}-LA${linear_attn_wbits}-S${shared_expert_wbits}"
trace_file="${TRACE_FILE:-calib/gate/${model_id}/c4/${seqlen}/${recipe}/moe-gate.json}"
loss_dir="${LOSS_DIR:-calib/${model_id}/${recipe}}"
qconfig="${QCONFIG:-qconfigs/qwen35/${recipe}-E${nominal_bits}.json}"
output="${OUTPUT:-results/fake_quant_models/Qwen/Qwen3.5-35B-A3B/MxMoE/${recipe}-E${nominal_bits}}"

common_args=(
  --model "${model}" --nsamples "${nsamples}" --seqlen "${seqlen}" --seed "${seed}"
  --attn_wbits "${attn_wbits}" --linear_attn_wbits "${linear_attn_wbits}"
  --shared_expert_wbits "${shared_expert_wbits}"
)
