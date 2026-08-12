#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 CONFIG_PATH" >&2
  exit 2
fi

config_path=$1
python_bin=/home/shuaidong/.conda/envs/openvk/bin/python
server_config=/home/shuaidong/.openviking/ov-openvk-gpt54mini-fallback.conf

api_key=$(
  "$python_bin" -c \
    'import json; print(json.load(open("/home/shuaidong/.openviking/ov-openvk-gpt54mini-fallback.conf"))["vlm"]["api_key"])'
)

export OPENVIKING_CONFIG_FILE=$server_config
export OPENAI_API_KEY=$api_key

"$python_bin" - <<'PY'
from openviking_sdk import SyncHTTPClient

client = SyncHTTPClient(timeout=600)
client.initialize()
client.rm("viking://resources", recursive=True)
print("Cleared viking://resources", flush=True)
PY

exec "$python_bin" benchmark/RAG/run.py --config "$config_path" --step all
