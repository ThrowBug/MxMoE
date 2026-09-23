#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "${repo_root}"

calib_dataset="${CALIB_DATASET:-c4}"
nsamples="${NSAMPLES:-128}"
seqlen="${SEQLEN:-2048}"
seed="${SEED:-0}"
nominal_bits="$(python -c 'import sys; print(float(sys.argv[1]))' "${NOMINAL_BITS:-2.0}")"
if [[ -n "${MXMOE_EFFECTIVE_BITS:-}" ]]; then
  effective_bits="$(python -c 'import sys; print(float(sys.argv[1]))' "${MXMOE_EFFECTIVE_BITS}")"
else
  effective_bits="$(python -c 'import sys; print(float(sys.argv[1]) + 0.25)' "${nominal_bits}")"
fi
model_id="qwen3_moe_30b_a3b_instruct_2507"
loss_dir="calib/${model_id}"
loss_suffix="${calib_dataset}-${nsamples}-${seqlen}-layer_out_norm.json"
w1_loss="${LOSS_W1:-${loss_dir}/${model_id}-MOE-gptq-W1A16_g128_asym-${loss_suffix}}"
w2_loss="${LOSS_W2:-${loss_dir}/${model_id}-MOE-gptq-W2A16_g128_asym-${loss_suffix}}"
w3_loss="${LOSS_W3:-${loss_dir}/${model_id}-MOE-gptq-W3A16_g128_asym-${loss_suffix}}"

# G128 contributes 0.25 metadata bits in MxMoE's original budget accounting.
# NOMINAL_BITS defaults to 2.0; MXMOE_EFFECTIVE_BITS can explicitly override
# the automatically derived nominal + 0.25 allocation budget.
python -m mxmoe.quant.bits_solver \
  --model "${model_id}" \
  --qtype gptq \
  --wbits "${effective_bits}" \
  --solve_mode layer \
  --batch "${PROFILE_BATCH:-8192}" \
  --r "${ACCURACY_WEIGHT:-1.0}" \
  --trace_file "${TRACE_FILE:-calib/gate/${model_id}/${calib_dataset}/${seqlen}/moe-gate.json}" \
  --perf_file "${PERF_FILE:-perf/performance_table.json}" \
  --calib_dataset "${calib_dataset}" \
  --nsamples "${nsamples}" \
  --seqlen "${seqlen}" \
  --seed "${seed}" \
  --loss_file "w1a16_g128_asym=${w1_loss}" \
  --loss_file "w2a16_g128_asym=${w2_loss}" \
  --loss_file "w3a16_g128_asym=${w3_loss}" \
  --filter_list w1a16_g128_asym w2a16_g128_asym w3a16_g128_asym
