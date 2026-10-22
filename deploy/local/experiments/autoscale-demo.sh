#!/usr/bin/env bash
# Autoscaling on a laptop: the gateway launches fake-engine processes as
# demand rises and drains them when it falls.
#
# Load runs in three phases: 4 requests in flight, then 40, then 4 again.
# Each replica should carry 8, so the pool should grow from 1 toward 5 and
# shrink back once demand has stayed low for 10 seconds.
#
#   deploy/local/experiments/autoscale-demo.sh   (from the repository root)
set -euo pipefail
cd "$(dirname "$0")/../../.."
go build -o bin/ ./cmd/...
OUT=${OUT:-/tmp/vsi-autoscale}
mkdir -p "$OUT"
cat >"$OUT/gateway.yaml" <<YAML
listen: "127.0.0.1:8090"
scheduler: { policy: least-load }
health: { interval: 1s, timeout: 500ms, failure_threshold: 1 }
flow: { enabled: true, max_inflight_per_endpoint: 8, batch_share: 1.0 }
autoscale:
  enabled: true
  provider: process
  min: 1
  max: 6
  target_inflight_per_replica: 8
  scale_down_after: 10s
  interval: 2s
  process:
    command: ["bin/fakeengine", "-addr", "127.0.0.1:{port}", "-model", "m", "-ttft", "100ms", "-tpot", "20ms", "-tokens", "32"]
admin: { token: demo }
YAML
bin/gateway -config "$OUT/gateway.yaml" 2>"$OUT/gateway.log" &
gw=$!
trap 'kill $gw $sampler 2>/dev/null; pkill -f "bin/fakeengine -addr 127.0.0.1" 2>/dev/null || true' EXIT
for _ in $(seq 1 50); do curl -sf localhost:8090/readyz >/dev/null && break; sleep 0.2; done

start=$(date +%s)
(while true; do
  n=$(curl -s -H 'Authorization: Bearer demo' localhost:8090/admin/backends | grep -o '"healthy":true' | wc -l | tr -d ' ')
  q=$(curl -s localhost:8090/metrics | awk '/^vsi_flow_queued\{band="interactive"\}/ {print $2}')
  echo "$(( $(date +%s) - start ))s replicas=$n queued=${q:-0}"
  sleep 1
done) >"$OUT/timeline.txt" &
sampler=$!

lg() { bin/loadgen -url http://localhost:8090 -model m -groups 1 -prefix-words 20 -max-tokens 32 -ignore-eos=false "$@" | sed -n '1p;3p'; }
lg -label "phase 1: 4 in flight" -requests 60 -concurrency 4
lg -label "phase 2: 40 in flight" -requests 1500 -concurrency 40
lg -label "phase 3: 4 in flight" -requests 180 -concurrency 4
sleep 15
kill $sampler 2>/dev/null
wait $sampler 2>/dev/null || true
awk 'NR % 3 == 1' "$OUT/timeline.txt"
