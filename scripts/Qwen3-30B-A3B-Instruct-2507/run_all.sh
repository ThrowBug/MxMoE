#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${script_dir}/trace.sh"
bash "${script_dir}/collect_loss.sh"
bash "${script_dir}/allocate.sh"
bash "${script_dir}/quantize.sh"
