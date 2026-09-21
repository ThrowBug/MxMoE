#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
python -m mxmoe.quant.qwen35_allocate \
  --model "${model}" --nsamples "${nsamples}" --seqlen "${seqlen}" --seed "${seed}" \
  --nominal_bits "${nominal_bits}" --trace "${trace_file}" \
  --loss_dir "${loss_dir}" --output "${qconfig}"
