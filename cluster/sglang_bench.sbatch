#!/bin/bash
#SBATCH --job-name=vsi-sglang
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:h200:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/%u/vsi/logs/%x_%j.out
#
# SGLang behind the gateway: two SGLang replicas on two H200s, the routing
# policies compared on a shared-prefix workload, as the vLLM campaign does.
#   sbatch sglang_bench.sbatch [model]
# Chat requests: SGLang's /tokenize applies no chat template, so the precise
# policy cannot get matching tokens and falls back to approximate prefixes.
# Completion requests: the precise policy uses SGLang's tokens and its KV
# events (radix-cache pages of 16 tokens).
set -euo pipefail
export VSI_ENGINE=sglang
source "${VSI_CODE:-/scratch/$USER/vsi/code}/cluster/common.sh"
MODEL=${1:-Qwen3-8B}
OUT=$ROOT/results/sglang-$(date +%Y%m%d-%H%M%S)
mkdir -p "$OUT"
trap stop_all EXIT
gpu_info 2>"$OUT/gpu.txt"
stage_env
PYTHON=$ENV_DIR/bin/python
LOCAL=$(stage_model "$MODEL")
# KV pool of 64k tokens per replica: the ~96k tokens of prefixes fit in two
# replicas only if each prefix stays on one of them.
KV_TOKENS=${KV_TOKENS:-64000}
WORK=(-model "$MODEL" -requests "${REQUESTS:-800}" -concurrency "${CONCURRENCY:-32}" -groups "${PROMPT_GROUPS:-48}"
  -prefix-words "${PREFIX_WORDS:-1500}" -question-words 24 -max-tokens "${MAX_TOKENS:-64}" -seed 1)

ports=() kvs=()
for gpu in 0 1; do
  port=$(free_port); kv=$(free_port)
  mapfile -t DEFAULTS < <(sglang_default_args "$kv")
  CUDA_VISIBLE_DEVICES=$gpu start_sglang "$OUT/sglang_r$gpu.log" "$port" "$LOCAL" \
    --served-model-name "$MODEL" --context-length 8192 --max-total-tokens "$KV_TOKENS" "${DEFAULTS[@]}"
  ports+=("$port") kvs+=("$kv")
done
for i in 0 1; do
  wait_http "http://127.0.0.1:${ports[$i]}/health" 1800 "${PIDS[$i]}" || { tail -40 "$OUT/sglang_r$i.log" >&2; exit 1; }
done
grep -hE "max_total_num_tokens|KV Cache is allocated|Capture cuda graph end" "$OUT"/sglang_r*.log >"$OUT/startup.txt" || true

GWPORT=$(free_port)
cfg=$OUT/gateway.yaml
{
  echo "listen: \"127.0.0.1:$GWPORT\""
  echo "scheduler: { policy: round-robin }"
  echo "admin: { token: bench }"
  echo "backends:"
  for i in 0 1; do
    echo "  - { name: s$i, url: \"http://127.0.0.1:${ports[$i]}\", engine: sglang, labels: { kv_events: \"tcp://127.0.0.1:${kvs[$i]}\" } }"
  done
} >"$cfg"
local_gateway "$cfg" "$GWPORT"
wait_backends "http://127.0.0.1:$GWPORT" bench 2 120
GW=http://127.0.0.1:$GWPORT
admin() { curl -s --noproxy '*' -X "$1" -H 'Authorization: Bearer bench' "$GW$2" ${3:+-d "$3"}; }
# The engine adapter reads SGLang's metric names.
admin GET /admin/backends >"$OUT/backends.json"

run() { # run LABEL POLICY [loadgen flags]
  local label=$1 policy=$2
  shift 2
  admin PUT /admin/policy "{\"policy\":\"$policy\"}" >/dev/null
  admin POST /admin/reset_prefix_cache >"$OUT/reset-$label.json"
  sleep 3
  log "$label"
  loadgen -url "$GW" -label "$label" "${WORK[@]}" "$@" -json | tee "$OUT/$label.txt"
  curl -s --noproxy '*' "$GW/metrics" | grep -E '^vsi_(kvindex|tokenize_seconds_count|backend_engine)' >"$OUT/$label.metrics" || true
}
for p in round-robin least-load prefix-aware precise-prefix-aware; do
  run "chat-$p" "$p"
done
for p in round-robin prefix-aware precise-prefix-aware; do
  run "completions-$p" "$p" -completions
done
log "results in $OUT"
