#!/usr/bin/env bash
# Interactive latency under a batch flood, with and without flow control.
#
# Four simulated replicas run at most 8 requests each and slow down under
# load (--time-factor-under-load 2). A batch tenant keeps 64 long requests in
# flight; meanwhile an interactive tenant sends short requests. Without flow
# control both share the engines' FIFO queues. With it, batch may hold only
# 75% of the 32 slots and interactive requests are dispatched first.
#
#   deploy/local/experiments/flow-control.sh   (from the repository root)
set -euo pipefail
cd "$(dirname "$0")/../../.."
F=(-f deploy/local/docker-compose.yaml)
LG=bin/loadgen
OUT=${OUT:-/tmp/vsi-flow}
mkdir -p "$OUT"
go build -o bin/ ./cmd/...

wait_ready() {
  for _ in $(seq 1 60); do curl -sf localhost:8080/readyz >/dev/null && return; sleep 1; done
  echo "gateway not ready" >&2; exit 1
}

for mode in ${MODES:-off on}; do
  export SIM_MAX_NUM_SEQS=8 SIM_LOAD_FACTOR=2.0 VSI_GATEWAY_CONFIG=experiments/flow-$mode.yaml
  docker compose "${F[@]}" up -d --build --force-recreate gateway sim-0 sim-1 sim-2 sim-3 >/dev/null 2>&1
  wait_ready; sleep 3
  $LG -api-key batch-demo-key -label "batch flow=$mode" -requests 100000 -concurrency 64 -groups 4 \
    -prefix-words 300 -max-tokens 64 -timeout 10m >"$OUT/batch-$mode.txt" 2>&1 &
  flood=$!
  sleep 8 # let the flood saturate the pool
  $LG -api-key chat-demo-key -label "interactive flow=$mode" -requests 200 -concurrency 4 -groups 50 \
    -prefix-words 100 -max-tokens 16 -seed 7 -json | tee "$OUT/interactive-$mode.txt"
  kill "$flood" 2>/dev/null || true
  wait "$flood" 2>/dev/null || true
  echo
done
