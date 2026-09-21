#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
python -m mxmoe.quant.qwen35_pipeline quantize "${common_args[@]}" \
  --qconfig "${qconfig}" --output "${output}" \
  --expected_nominal_bits "${nominal_bits}"
