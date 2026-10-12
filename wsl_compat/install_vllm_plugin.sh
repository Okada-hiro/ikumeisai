#!/usr/bin/env bash
# Register wsl_compat/ikumeisai_vllm_plugin.py as a vLLM general plugin in ~/venvs/vllm (no pip in that venv).
set -euo pipefail
SP=$(~/venvs/vllm/bin/python -c "import site;print(site.getsitepackages()[0])")
cp "$(dirname "$0")/ikumeisai_vllm_plugin.py" "$SP/"
D="$SP/ikumeisai_vllm_plugin-0.1.dist-info"; mkdir -p "$D"
printf "Metadata-Version: 2.1\nName: ikumeisai-vllm-plugin\nVersion: 0.1\n" > "$D/METADATA"
printf "[vllm.general_plugins]\nikumeisai = ikumeisai_vllm_plugin:register\n" > "$D/entry_points.txt"
printf "ikumeisai_vllm_plugin.py,,\n" > "$D/RECORD"
echo "installed into $SP"
