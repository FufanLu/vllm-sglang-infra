#!/usr/bin/env bash
# Approximate vs precise prefix-aware routing with two gateways.
#
# Two gateway instances share four simulated replicas, and requests alternate
# between them, as behind a load balancer in a highly available deployment.
# The approximate index of each gateway learns only from its own decisions,
# so the two can send the same prefix to different replicas. The precise
# index is fed by the replicas' KV-cache events, which both gateways see.
#
#   deploy/local/experiments/precise-prefix.sh   (from the repository root)
set -euo pipefail
cd "$(dirname "$0")/../../.."
F=(-f deploy/local/docker-compose.yaml -f deploy/local/docker-compose.two-gateways.yaml)
OUT=${OUT:-/tmp/vsi-precise}
mkdir -p "$OUT"
go build -o bin/ ./cmd/...

ready() {
  for _ in $(seq 1 60); do curl -sf "$1/readyz" >/dev/null && return; sleep 1; done
  echo "$1 not ready" >&2; exit 1
}

for policy in ${POLICIES:-prefix-aware precise-prefix-aware}; do
  export VSI_SCHEDULER_POLICY=$policy
  docker compose "${F[@]}" up -d --build --force-recreate kvrelay sim-0 sim-1 sim-2 sim-3 gateway gateway-b >/dev/null 2>&1
  ready http://localhost:8080; ready http://localhost:8081
  sleep 5 # health probes and KV-event subscriptions settle
  bin/loadgen -label "$policy, 2 gateways" -url http://localhost:8080,http://localhost:8081 \
    -requests 600 -concurrency 16 -groups 96 -prefix-words 600 -max-tokens 16 -seed 1 -json | tee "$OUT/$policy.txt"
  echo
done
