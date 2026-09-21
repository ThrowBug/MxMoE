#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
python -m mxmoe.quant.qwen35_pipeline trace "${common_args[@]}" \
  --output "${trace_file}"
