#!/usr/bin/env bash
# Submits a whole experiment from the login node.
#   campaign.sh routing   gateway (CPU) + 4 H200 replicas + bench; tears down after
# The other experiments are single jobs: coldstart.sbatch, moe_bench.sbatch,
# pd_bench.sbatch, kvtier_bench.sbatch.
set -euo pipefail
C=${VSI_CODE:-/scratch/$USER/vsi/code}/cluster
case "${1:-}" in
routing)
  MODEL=${MODEL:-Qwen3-8B}
  # A KV budget of 12 GiB per replica (~85k tokens of Qwen3-8B) keeps the
  # cache under pressure at this workload, as a larger model would.
  REPLICA_ARGS=${REPLICA_ARGS:---max-model-len 16384 --kv-cache-memory-bytes 12G}
  gw=$(sbatch --parsable "$C/gateway.sbatch")
  reps=()
  for _ in 1 2 3 4; do
    # shellcheck disable=SC2086
    reps+=("$(sbatch --parsable --dependency=after:"$gw" "$C/vllm_replica.sbatch" "$MODEL" $REPLICA_ARGS)")
  done
  bench=$(sbatch --parsable --dependency=after:"$gw" --export=ALL,MODEL="$MODEL",CLEANUP_JOBS="${reps[*]} $gw" "$C/bench.sbatch" 4)
  echo "gateway $gw, replicas ${reps[*]}, bench $bench"
  ;;
*)
  echo "usage: campaign.sh routing" >&2
  exit 2
  ;;
esac
