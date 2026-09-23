#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "${repo_root}"

model="${MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
calib_dataset="${CALIB_DATASET:-c4}"
nsamples="${NSAMPLES:-128}"
seqlen="${SEQLEN:-2048}"
seed="${SEED:-0}"
profile_batch="${PROFILE_BATCH:-8192}"
nominal_bits="$(python -c 'import sys; print(float(sys.argv[1]))' "${NOMINAL_BITS:-2.0}")"
if [[ -n "${MXMOE_EFFECTIVE_BITS:-}" ]]; then
  effective_bits="$(python -c 'import sys; print(float(sys.argv[1]))' "${MXMOE_EFFECTIVE_BITS}")"
else
  effective_bits="$(python -c 'import sys; print(float(sys.argv[1]) + 0.25)' "${nominal_bits}")"
fi
accuracy_weight="$(python -c 'import sys; print(float(sys.argv[1]))' "${ACCURACY_WEIGHT:-1.0}")"
qconfig="${QCONFIG:-qconfigs/w1a16_g128_asym+w2a16_g128_asym+w3a16_g128_asym/qwen3_moe_30b_a3b_instruct_2507_gptq_Slayer_bs${profile_batch}_wbits${effective_bits}_r${accuracy_weight}.json}"
output="${OUTPUT:-results/fake_quant_models/Qwen/Qwen3-30B-A3B-Instruct-2507/MxMoE/C4-Seed${seed}_E${nominal_bits}-A4-D4}"

python -m mxmoe.quant.qwen3_quantize \
  --model "${model}" \
  --qconfig "${qconfig}" \
  --output "${output}" \
  --calib_dataset "${calib_dataset}" \
  --nsamples "${nsamples}" \
  --seqlen "${seqlen}" \
  --seed "${seed}" \
  --groupsize 128 \
  --blocksize 128 \
  --percdamp 0.01 \
  --attn_wbits 4 \
  --dense_wbits 4 \
  --expected_nominal_bits "${nominal_bits}"
