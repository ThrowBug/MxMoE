#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "${repo_root}"

calib_dataset="${CALIB_DATASET:-c4}"
seqlen="${SEQLEN:-2048}"

# G128 contributes 0.25 metadata bits in MxMoE's original budget accounting.
# Thus 2.25 here corresponds to a 2.0 nominal average over expert weights.
python -m mxmoe.quant.bits_solver \
  --model qwen3_moe_30b_a3b_instruct_2507 \
  --qtype gptq \
  --wbits "${MXMOE_EFFECTIVE_BITS:-2.25}" \
  --solve_mode layer \
  --batch "${PROFILE_BATCH:-8192}" \
  --r "${ACCURACY_WEIGHT:-1.0}" \
  --trace_file "${TRACE_FILE:-calib/gate/qwen3_moe_30b_a3b_instruct_2507/${calib_dataset}/${seqlen}/moe-gate.json}" \
  --perf_file "${PERF_FILE:-perf/performance_table.json}" \
  --filter_list w1a16_g128_asym w2a16_g128_asym w3a16_g128_asym
