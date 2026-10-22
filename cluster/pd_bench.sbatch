#!/bin/bash
#SBATCH --job-name=vsi-pd
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:h200:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/%u/vsi/logs/%x_%j.out
#
# B: prefill/decode disaggregation on two H200s of one node vs the same two
# GPUs serving aggregated. Long prompts and medium outputs, where prefill
# interrupting decode hurts TPOT the most.
#   sbatch pd_bench.sbatch [model]
set -euo pipefail
source "${VSI_CODE:-/scratch/$USER/vsi/code}/cluster/common.sh"
MODEL=${1:-Qwen3-8B}
OUT=$ROOT/results/pd-$(date +%Y%m%d-%H%M%S)
mkdir -p "$OUT"
trap stop_all EXIT
gpu_info 2>"$OUT/gpu.txt"
stage_env
PYTHON=$ENV_DIR/bin/python
LOCAL=$(stage_model "$MODEL")
COMMON=(--served-model-name "$MODEL" --max-model-len 16384 --enable-prompt-tokens-details)
WORK=(-model "$MODEL" -requests "${REQUESTS:-400}" -concurrency "${CONCURRENCY:-16}" -groups "${GROUPS:-400}"
  -prefix-words "${PREFIX_WORDS:-2500}" -max-tokens "${MAX_TOKENS:-256}" -seed 1)
GWPORT=$(free_port)
up() { wait_http "http://127.0.0.1:$1/health" 1800 "$VLLM_PID"; }
gateway() { # gateway NAME PD BACKEND_YAML_LINES...
  local cfg=$OUT/gateway-$1.yaml pd=$2
  shift 2
  { echo "listen: \"127.0.0.1:$GWPORT\""; echo "scheduler: { policy: least-load }"; echo "pd: { enabled: $pd }"
    echo "admin: { token: bench }"; echo "backends:"; printf '  %s\n' "$@"; } >"$cfg"
  local_gateway "$cfg" "$GWPORT"
  wait_backends "http://127.0.0.1:$GWPORT" bench "$#" 120
}

# Aggregated baseline: two ordinary replicas.
a0=$(free_port); a1=$(free_port)
CUDA_VISIBLE_DEVICES=0 start_vllm "$OUT/vllm_agg0.log" "$a0" "$LOCAL" "${COMMON[@]}"
CUDA_VISIBLE_DEVICES=1 start_vllm "$OUT/vllm_agg1.log" "$a1" "$LOCAL" "${COMMON[@]}"
up "$a0"; up "$a1"
gateway agg false "- { name: agg0, url: \"http://127.0.0.1:$a0\" }" "- { name: agg1, url: \"http://127.0.0.1:$a1\" }"
loadgen -url "http://127.0.0.1:$GWPORT" -label aggregated "${WORK[@]}" -json | tee "$OUT/aggregated.txt"
stop_all

# 1 prefill + 1 decode, KV over NIXL (same node: CUDA IPC through UCX).
KVT='{"kv_connector":"NixlConnector","kv_role":"kv_both"}'
pp=$(free_port); dp=$(free_port)
CUDA_VISIBLE_DEVICES=0 VLLM_NIXL_SIDE_CHANNEL_PORT=$(free_port) start_vllm "$OUT/vllm_prefill.log" "$pp" "$LOCAL" "${COMMON[@]}" --kv-transfer-config "$KVT"
CUDA_VISIBLE_DEVICES=1 VLLM_NIXL_SIDE_CHANNEL_PORT=$(free_port) start_vllm "$OUT/vllm_decode.log" "$dp" "$LOCAL" "${COMMON[@]}" --kv-transfer-config "$KVT"
up "$pp"; up "$dp"
gateway pd true "- { name: prefill, url: \"http://127.0.0.1:$pp\", labels: { role: prefill } }" \
  "- { name: decode, url: \"http://127.0.0.1:$dp\", labels: { role: decode } }"
loadgen -url "http://127.0.0.1:$GWPORT" -label 1p1d "${WORK[@]}" -json | tee "$OUT/1p1d.txt"
curl -s --noproxy '*' "http://127.0.0.1:$GWPORT/metrics" | grep -E '^vsi_(pd_prefill_seconds_(sum|count)|retries_total)' | tee "$OUT/pd-metrics.txt"
stop_all
log "results in $OUT"
