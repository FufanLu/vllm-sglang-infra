#!/bin/bash
#SBATCH --job-name=vsi-replica
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --signal=B:TERM@600
#SBATCH --output=/scratch/%u/vsi/logs/%x_%j.out
#
# One vLLM replica on one H200, registered with the gateway while it serves.
#   sbatch /scratch/$USER/vsi/code/cluster/vllm_replica.sbatch Qwen3-8B [extra vllm serve args]
# The gpu partition allows one GPU per job and four per user, so four of
# these jobs are the four-replica pool the routing experiments run on.
#
# Ports are picked free at start, since several of these jobs can land on
# one 8-GPU node. The replica publishes KV-cache events (for the precise
# prefix index) and runs in vLLM dev mode (prefix-cache reset, sleep/wake).
#
# Shutdown is a drain, not a kill: Slurm sends SIGTERM ten minutes before the
# walltime (--signal), the job deregisters so the gateway stops sending new
# requests, waits for in-flight ones to finish, then stops vLLM.
set -euo pipefail
source "${VSI_CODE:-/scratch/$USER/vsi/code}/cluster/common.sh"
MODEL=${1:?usage: vllm_replica.sbatch <model-name> [vllm serve args...]}
shift
NAME=${REPLICA_NAME:-r${SLURM_JOB_ID}}
DRAIN_TIMEOUT=${DRAIN_TIMEOUT:-300}

gpu_info
stage_env
PYTHON=$ENV_DIR/bin/python
PORT=${PORT:-$(free_port)}
KV_PORT=${KV_PORT:-$(free_port)}
LOCAL_MODEL=$(stage_model "$MODEL")

VLOG=$LOGS/vllm_${NAME}.log
log "starting vllm serve $MODEL on $(hostname):$PORT, kv events on :$KV_PORT (log: $VLOG)"
t0=$SECONDS
mapfile -t DEFAULTS < <(vllm_default_args "$KV_PORT")
start_vllm "$VLOG" "$PORT" "$LOCAL_MODEL" --served-model-name "$MODEL" "${DEFAULTS[@]}" "$@"

GATEWAY=$(cat "$ROOT/gateway.addr" 2>/dev/null || true)
# The token goes to curl through a private header file, not argv, where any
# user on the node could read it with ps.
AUTH=$(mktemp /tmp/vsi-auth-XXXXXX)
trap 'rm -f "$AUTH"' EXIT
if [ -s "$ROOT/admin.token" ]; then
  printf 'Authorization: Bearer %s\n' "$(cat "$ROOT/admin.token")" >"$AUTH"
else
  GATEWAY=
fi
admin() { # admin METHOD [JSON]
  [ -n "$GATEWAY" ] || return 0
  local args=(-sf --noproxy '*' -o /dev/null -X "$1" -H "@$AUTH" "$GATEWAY/admin/backends/$NAME")
  if [ $# -ge 2 ]; then
    args+=(-H 'Content-Type: application/json' -d "$2")
  fi
  curl "${args[@]}"
}

drain() {
  trap - TERM INT
  log "draining: deregistering $NAME"
  admin DELETE || log "deregister failed (gateway down?)"
  local t=0
  while [ "$t" -lt "$DRAIN_TIMEOUT" ]; do
    running=$(curl -s --noproxy '*' "http://127.0.0.1:$PORT/metrics" |
      awk '/^vllm:num_requests_(running|waiting)/ { n += $NF } END { print int(n) }')
    [ "${running:-0}" -eq 0 ] && break
    sleep 5
    t=$((t + 5))
  done
  log "stopping vllm after ${t}s of drain"
  stop_all
  exit 0
}
trap drain TERM INT

if ! wait_http "http://127.0.0.1:$PORT/health" 1800 "$VLLM_PID"; then
  tail -40 "$VLOG" >&2
  exit 1
fi
log "vllm ready in $((SECONDS - t0))s"
startup_times "$VLOG" >&2

if [ -n "$GATEWAY" ]; then
  gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
  host=$(hostname)
  labels=$(printf '"slurm_job":"%s","node":"%s","gpu":"%s","kv_events":"tcp://%s:%s"' \
    "$SLURM_JOB_ID" "$host" "$gpu" "$host" "$KV_PORT")
  [ -n "${ROLE:-}" ] && labels="$labels,\"role\":\"$ROLE\""
  body=$(printf '{"url":"http://%s:%s","engine":"vllm","labels":{%s}}' "$host" "$PORT" "$labels")
  if admin PUT "$body"; then log "registered $NAME with $GATEWAY"; else log "registration with $GATEWAY failed"; fi
else
  log "no gateway.addr: serving unregistered at http://$(hostname):$PORT"
fi

# wait returns early when a trapped signal arrives; drain then takes over.
while kill -0 "$VLLM_PID" 2>/dev/null; do
  wait "$VLLM_PID" || true
done
log "vllm exited"
admin DELETE || true
