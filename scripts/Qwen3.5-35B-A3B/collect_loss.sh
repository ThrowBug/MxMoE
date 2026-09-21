#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
python -m mxmoe.quant.qwen35_pipeline collect_loss "${common_args[@]}" \
  --output_dir "${loss_dir}"
