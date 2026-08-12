#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 SERVER_CONFIG BENCHMARK_CONFIG" >&2
  exit 2
fi

server_config=$1
benchmark_config=$2
python_bin=/home/shuaidong/.conda/envs/openvk/bin/python

readarray -t settings < <(
  "$python_bin" - "$server_config" <<'PY'
import json
import sys

config = json.load(open(sys.argv[1]))
server = config["server"]
vlm = config["vlm"]
print(f"http://{server.get('host', '127.0.0.1')}:{server['port']}")
print(vlm["api_key"])
PY
)

server_url=${settings[0]}
api_key=${settings[1]}

export OPENVIKING_CONFIG_FILE=$server_config
export OPENVIKING_URL=$server_url
export OPENAI_API_KEY=$api_key

"$python_bin" - <<'PY'
import os
from openviking_sdk import SyncHTTPClient

client = SyncHTTPClient(url=os.environ["OPENVIKING_URL"], timeout=600)
client.initialize()
client.rm("viking://resources", recursive=True)
print(f"Cleared viking://resources on {os.environ['OPENVIKING_URL']}", flush=True)
PY

exec "$python_bin" benchmark/RAG/run.py --config "$benchmark_config" --step all
