#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 BENCHMARK_CONFIG" >&2
  exit 2
fi

server_config=/home/shuaidong/.openviking/ov-openvk-gpt54mini-fallback.conf

exec "$(dirname "$0")/run_openviking_benchmark.sh" "$server_config" "$1"
