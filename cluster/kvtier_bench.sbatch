#!/bin/bash
#SBATCH --job-name=vsi-kvtier
#SBATCH --partition=gpu,gpu-short
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/%u/vsi/logs/%x_%j.out
#
# A: a working set larger than GPU KV memory, with and without vLLM's native
# CPU offloading, and the precise index seeing blocks in the CPU tier.
#   sbatch kvtier_bench.sbatch [model]
set -euo pipefail
source "${VSI_CODE:-/scratch/$USER/vsi/code}/cluster/common.sh"
MODEL=${1:-Qwen3-8B}
OUT=$ROOT/results/kvtier-$(date +%Y%m%d-%H%M%S)
mkdir -p "$OUT"
trap stop_all EXIT
gpu_info 2>"$OUT/gpu.txt"
stage_env
PYTHON=$ENV_DIR/bin/python
LOCAL=$(stage_model "$MODEL")
# ~6 GiB of GPU KV (~43k Qwen3-8B tokens) against ~100k tokens of prefixes.
COMMON=(--served-model-name "$MODEL" --max-model-len 8192 --kv-cache-memory-bytes "${GPU_KV:-6G}")
WORK=(-model "$MODEL" -requests "${REQUESTS:-600}" -concurrency "${CONCURRENCY:-16}" -groups "${PROMPT_GROUPS:-48}"
  -prefix-words "${PREFIX_WORDS:-1500}" -max-tokens 32 -seed 1)
GWPORT=$(free_port)

for mode in gpu-only cpu-offload; do
  port=$(free_port); kv=$(free_port)
  mapfile -t DEFAULTS < <(vllm_default_args "$kv")
  extra=()
  [ "$mode" = cpu-offload ] && extra=(--kv-offloading-size "${CPU_KV_GIB:-48}" --kv-offloading-backend native)
  start_vllm "$OUT/vllm_$mode.log" "$port" "$LOCAL" "${COMMON[@]}" "${DEFAULTS[@]}" "${extra[@]}"
  wait_http "http://127.0.0.1:$port/health" 1800 "$VLLM_PID"
  cfg=$OUT/gateway-$mode.yaml
  printf 'listen: "127.0.0.1:%s"\nscheduler: { policy: precise-prefix-aware }\nadmin: { token: bench }\nbackends:\n  - { name: r0, url: "http://127.0.0.1:%s", labels: { kv_events: "tcp://127.0.0.1:%s" } }\n' \
    "$GWPORT" "$port" "$kv" >"$cfg"
  local_gateway "$cfg" "$GWPORT"
  wait_backends "http://127.0.0.1:$GWPORT" bench 1 120
  # Two passes: the first fills the tiers, the second measures reuse.
  loadgen -url "http://127.0.0.1:$GWPORT" -label "$mode warmup" "${WORK[@]}" >/dev/null
  loadgen -url "http://127.0.0.1:$GWPORT" -label "$mode" "${WORK[@]}" -json | tee "$OUT/$mode.txt"
  curl -s --noproxy '*' "http://127.0.0.1:$GWPORT/metrics" | grep -E '^vsi_kvindex' | tee "$OUT/$mode-index.txt" || true
  stop_all
done
log "results in $OUT"
