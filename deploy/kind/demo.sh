#!/usr/bin/env bash
# The platform on a local Kubernetes cluster (kind), without GPUs.
#
# Installs the Helm chart with four simulated model servers, lets the
# gateway discover them by pod label, routes with the precise prefix index
# (KV events through kvrelay), then turns on autoscaling and drives load so
# the model-server Deployment follows demand.
#
# Needs docker, kind, helm, kubectl.   deploy/kind/demo.sh   (from the repo root)
set -euo pipefail
cd "$(dirname "$0")/../.."
CLUSTER=${CLUSTER:-vsi}
CTX=kind-$CLUSTER
PORT=${PORT:-8085}
go build -o bin/ ./cmd/...

kind get clusters | grep -qx "$CLUSTER" || kind create cluster --name "$CLUSTER" --wait 120s
docker build -q -t vsi-gateway:dev .
kind load docker-image vsi-gateway:dev ghcr.io/llm-d/llm-d-inference-sim:v0.9.0 --name "$CLUSTER"
helm upgrade --install demo deploy/helm/vllm-sglang-infra --kube-context "$CTX" \
  --set gateway.config.scheduler.policy=precise-prefix-aware --wait --timeout 180s
kubectl --context "$CTX" port-forward svc/demo-vsi-gateway "$PORT":80 >/dev/null 2>&1 &
pf=$!
trap 'kill $pf 2>/dev/null' EXIT
sleep 3
TOKEN=$(kubectl --context "$CTX" get secret demo-vsi-admin -o jsonpath='{.data.token}' | base64 -d)

echo "== discovered endpoints"
curl -s -H "Authorization: Bearer $TOKEN" localhost:$PORT/admin/backends |
  python3 -c "import json,sys; [print(' ', b['name'], b['url'], 'healthy' if b['healthy'] else 'down') for b in json.load(sys.stdin)]"
echo "== routing with the precise prefix index"
bin/loadgen -url http://localhost:$PORT -requests 300 -concurrency 16 -groups 24 -prefix-words 300 -max-tokens 8 | sed -n '1p;3p;5p'

echo "== autoscaling"
helm upgrade demo deploy/helm/vllm-sglang-infra --kube-context "$CTX" --reuse-values \
  --set gateway.config.autoscale.enabled=true --set gateway.config.autoscale.max=6 \
  --set gateway.config.autoscale.target_inflight_per_replica=8 --set gateway.config.autoscale.interval=3s \
  --set gateway.config.autoscale.scale_down_after=15s --set gateway.config.flow.max_inflight_per_endpoint=8 \
  --wait --timeout 120s >/dev/null
kill $pf; sleep 2
kubectl --context "$CTX" port-forward svc/demo-vsi-gateway "$PORT":80 >/dev/null 2>&1 &
pf=$!
sleep 3
start=$(date +%s)
show() { echo "  $(( $(date +%s) - start ))s replicas=$(kubectl --context "$CTX" get deploy demo-vsi-sim -o jsonpath='{.spec.replicas}') $1"; }
show "idle"; sleep 25; show "idle 25s: scaled down"
bin/loadgen -url http://localhost:$PORT -requests 2500 -concurrency 40 -groups 8 -prefix-words 50 -max-tokens 32 \
  -ignore-eos=false >/tmp/vsi-kind-load.txt 2>&1 &
load=$!
for _ in 1 2 3 4 5 6; do sleep 5; show "40 in flight"; done
wait $load
sed -n '1p;3p' /tmp/vsi-kind-load.txt
sleep 25; show "idle again: scaled down"
echo "Delete the cluster with: kind delete cluster --name $CLUSTER"
