#!/bin/bash
#SBATCH --job-name=vsi-moe
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:h200:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/%u/vsi/logs/%x_%j.out
#
# E: MoE serving on two H200s, four ways, on one shared-prefix workload.
#   1. dp-ep-ranks: DP=2 + EP=2, each rank its own endpoint, our routing
#      (least-load, prefix-aware, prefix-aware with a low saturation).
#   2. dp-ep-internal: the same deployment behind vLLM's internal balancer.
#   3. tp2: tensor parallel over both GPUs.
#   4. replicas: two independent copies, our prefix-aware routing.
# MoE ranks step in lockstep (idle ranks run dummy batches), so routing for
# cache affinity has to stay balanced; config 1 tests how much.
#   sbatch moe_bench.sbatch [model]     (default Qwen3-30B-A3B-FP8)
set -euo pipefail
source "${VSI_CODE:-/scratch/$USER/vsi/code}/cluster/common.sh"
MODEL=${1:-Qwen3-30B-A3B-FP8}
OUT=$ROOT/results/moe-$(date +%Y%m%d-%H%M%S)
mkdir -p "$OUT"
trap stop_all EXIT
gpu_info 2>"$OUT/gpu.txt"
stage_env
PYTHON=$ENV_DIR/bin/python
LOCAL=$(stage_model "$MODEL")
COMMON=(--served-model-name "$MODEL" --max-model-len 8192 --enable-prompt-tokens-details)
WORK=(-model "$MODEL" -requests "${REQUESTS:-800}" -concurrency "${CONCURRENCY:-32}" -groups "${PROMPT_GROUPS:-64}"
  -prefix-words "${PREFIX_WORDS:-1000}" -max-tokens "${MAX_TOKENS:-64}" -seed 1)
GWPORT=$(free_port)

gateway_for() { # gateway_for NAME URL... writes a config and starts a gateway
  local name=$1 cfg=$OUT/gateway-$1.yaml
  shift
  { echo "listen: \"127.0.0.1:$GWPORT\""; echo "scheduler: { policy: least-load }"; echo "admin: { token: bench }"; echo "backends:"
    local i=0; for u in "$@"; do echo "  - { name: $name-$i, url: \"$u\" }"; i=$((i + 1)); done; } >"$cfg"
  local_gateway "$cfg" "$GWPORT"
  wait_backends "http://127.0.0.1:$GWPORT" bench "$#" 120
}
policy() { curl -s --noproxy '*' -X PUT -H 'Authorization: Bearer bench' "http://127.0.0.1:$GWPORT/admin/policy" -d "$1" >/dev/null; }
bench() { # bench LABEL
  curl -s --noproxy '*' -X POST -H 'Authorization: Bearer bench' "http://127.0.0.1:$GWPORT/admin/reset_prefix_cache" >/dev/null
  sleep 2
  loadgen -url "http://127.0.0.1:$GWPORT" -label "$1" "${WORK[@]}" -json | tee "$OUT/$1.txt"
}

export NCCL_DEBUG=${NCCL_DEBUG:-INFO} NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-INIT,NET}

# Each configuration runs on its own: one that fails to start is logged
# and the job moves on to the next.
config_ok() { # config_ok NAME PORT... waits for every server to come up
  local name=$1
  shift
  for p in "$@"; do
    if ! wait_http "http://127.0.0.1:$p/health" 1800 "$VLLM_PID"; then
      log "config $name failed to start"
      echo "$name: failed to start" >>"$OUT/failed.txt"
      stop_all
      return 1
    fi
  done
}

export NCCL_DEBUG=${NCCL_DEBUG:-INFO} NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-INIT,NET}

# Each configuration runs on its own: one that fails to start is logged
# and the job moves on to the next.
config_ok() { # config_ok NAME PORT... waits for every server to come up
  local name=$1
  shift
  for p in "$@"; do
    if ! wait_http "http://127.0.0.1:$p/health" 1800 "$VLLM_PID"; then
      log "config $name failed to start"
      echo "$name: failed to start" >>"$OUT/failed.txt"
      stop_all
      return 1
    fi
  done
}

# 1. DP+EP, ranks as endpoints.
p0=$(free_port); p1=$(free_port)
CUDA_VISIBLE_DEVICES=0 start_vllm "$OUT/vllm_rank0.log" "$p0" "$LOCAL" "${COMMON[@]}" --data-parallel-size 2 --data-parallel-rank 0 --enable-expert-parallel
CUDA_VISIBLE_DEVICES=1 start_vllm "$OUT/vllm_rank1.log" "$p1" "$LOCAL" "${COMMON[@]}" --data-parallel-size 2 --data-parallel-rank 1 --enable-expert-parallel
if config_ok dp-ep-ranks "$p0" "$p1"; then
  gateway_for rank "http://127.0.0.1:$p0" "http://127.0.0.1:$p1"
  policy '{"policy":"least-load"}'; bench dp-ep-ranks-least-load
  policy '{"policy":"prefix-aware"}'; bench dp-ep-ranks-prefix-aware
  policy '{"policy":"prefix-aware","queue_saturation":8}'; bench dp-ep-ranks-prefix-aware-sat8
  stop_all
fi

# 2. DP+EP behind vLLM's internal balancer (queue-based, not KV-aware).
p=$(free_port)
start_vllm "$OUT/vllm_internal.log" "$p" "$LOCAL" "${COMMON[@]}" --data-parallel-size 2 --enable-expert-parallel
if config_ok dp-ep-internal "$p"; then
  gateway_for internal "http://127.0.0.1:$p"; bench dp-ep-internal-lb; stop_all
fi

# 3. Tensor parallel.
p=$(free_port)
start_vllm "$OUT/vllm_tp2.log" "$p" "$LOCAL" "${COMMON[@]}" --tensor-parallel-size 2
if config_ok tp2 "$p"; then
  gateway_for tp "http://127.0.0.1:$p"; bench tp2; stop_all
fi

# 4. Two independent replicas (no collective communication at all).
p0=$(free_port); p1=$(free_port)
CUDA_VISIBLE_DEVICES=0 start_vllm "$OUT/vllm_r0.log" "$p0" "$LOCAL" "${COMMON[@]}"
CUDA_VISIBLE_DEVICES=1 start_vllm "$OUT/vllm_r1.log" "$p1" "$LOCAL" "${COMMON[@]}"
if config_ok replicas "$p0" "$p1"; then
  gateway_for rep "http://127.0.0.1:$p0" "http://127.0.0.1:$p1"
  policy '{"policy":"least-load"}'; bench replicas-least-load
  policy '{"policy":"prefix-aware"}'; bench replicas-prefix-aware
  stop_all
fi
log "results in $OUT"
