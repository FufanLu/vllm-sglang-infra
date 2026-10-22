# Sourced by every cluster job. Paths, versions, and staging helpers live here.
#
# Layout on the shared filesystem (/scratch is NFS: fine for big sequential
# reads, slow for the thousands of small files a Python env imports):
#   $ROOT/code     scripts and Linux binaries (cluster/push.sh)
#   $ROOT/envs     packed Python environments (setup_env.sbatch)
#   $ROOT/models   model weights (download_model.sbatch)
#   $ROOT/logs     job logs
# Jobs unpack the env and copy the weights to node-local /tmp, once per node.

ROOT=${VSI_ROOT:-/scratch/$USER/vsi}
CODE=${VSI_CODE:-$ROOT/code}
MODELS=$ROOT/models
LOGS=$ROOT/logs

# vLLM 0.26.0 is the newest release pinned to torch 2.11.0. vLLM publishes no
# cu128 wheels; the H200 nodes run driver 570 (CUDA 12.8), so the cu129 build
# relies on CUDA minor-version compatibility. check_gpu.sbatch verifies it.
VLLM_VERSION=${VLLM_VERSION:-0.26.0}
CUDA_TAG=${CUDA_TAG:-cu129}
# Bump ENV_REV when the package set changes: nodes keep unpacked envs, and
# a new name is what makes them unpack the new one. r2 adds NIXL.
ENV_REV=${ENV_REV:-r2}
ENV_NAME=vsi-vllm-${VLLM_VERSION}-${CUDA_TAG}-${ENV_REV}
ENV_DIR=/tmp/$ENV_NAME # identical path on every node, so the packed venv needs no rewriting
ENV_TAR=$ROOT/envs/$ENV_NAME.tar

mkdir -p "$ROOT/envs" "$MODELS" "$LOGS"

# Compute nodes have no internet, and caches must not land in $HOME.
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export HF_HOME=/tmp/vsi-hf-$USER
export XDG_CACHE_HOME=/tmp/vsi-cache-$USER
export VLLM_CACHE_ROOT=/tmp/vsi-vllm-cache-$USER
export PYTHONUNBUFFERED=1

log() { echo "[$(date +%H:%M:%S)] $*" >&2; }

# stage_env unpacks $ENV_TAR to $ENV_DIR unless this node already has it.
# A lock serializes jobs that land on the same node; unpacking into a scratch
# directory and renaming makes a half-finished unpack impossible to pick up.
stage_env() {
  [ -f "$ENV_TAR" ] || { log "missing $ENV_TAR: run setup_env.sbatch first"; return 1; }
  (
    flock 9
    if [ ! -f "$ENV_DIR/.complete" ]; then
      local t0=$SECONDS stage=/tmp/.vsi-stage-$$
      rm -rf "$ENV_DIR" "$stage" && mkdir -p "$stage"
      tar -C "$stage" -xf "$ENV_TAR"
      mv "$stage/$ENV_NAME" "$ENV_DIR" && rmdir "$stage"
      touch "$ENV_DIR/.complete"
      log "unpacked $ENV_NAME in $((SECONDS - t0))s"
    fi
  ) 9>"/tmp/.$ENV_NAME.lock"
  activate_env
}

# activate_env puts the env's tools on PATH (FlashInfer builds kernels at
# runtime with ninja and nvcc) and loads a CUDA 12.8 toolkit, the newest the
# H200 nodes' driver 570 runs. Sampling uses vLLM's native top-k/top-p
# kernels rather than FlashInfer's, which would be compiled at startup.
activate_env() {
  export PATH="$ENV_DIR/bin:$PATH" VIRTUAL_ENV="$ENV_DIR"
  if module load cuda/12.8.0 >/dev/null 2>&1 && command -v nvcc >/dev/null; then
    CUDA_HOME=$(dirname "$(dirname "$(command -v nvcc)")")
    export CUDA_HOME
  fi
  export FLASHINFER_WORKSPACE_BASE=/tmp/vsi-flashinfer-$USER
  export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
}

# stage_model NAME copies $MODELS/NAME to node-local disk unless this node
# already has it, and prints the local path. Loading safetensors straight off
# NFS goes through mmap page faults and is far slower than one sequential copy.
stage_model() {
  local name=$1 src=$MODELS/$1 dst=/tmp/vsi-models/$1
  [ -f "$src/.complete" ] || { log "model $name not downloaded: run download_model.sbatch"; return 1; }
  mkdir -p /tmp/vsi-models
  (
    flock 9
    if [ ! -f "$dst/.complete" ]; then
      local t0=$SECONDS
      rm -rf "$dst" "$dst.partial"
      cp -r "$src" "$dst.partial" && mv "$dst.partial" "$dst"
      log "staged $name ($(du -sh "$dst" | cut -f1)) in $((SECONDS - t0))s"
    fi
  ) 9>"/tmp/vsi-models/.$name.lock"
  echo "$dst"
}

# wait_http URL TIMEOUT_S [PID] waits until URL answers 2xx. It gives up early
# if PID (the server being waited on) exits.
wait_http() {
  local url=$1 timeout=$2 pid=${3:-} t=0
  until curl -sf -o /dev/null --noproxy '*' "$url"; do
    if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
      log "process $pid exited before $url came up"
      return 1
    fi
    if [ "$t" -ge "$timeout" ]; then
      log "timed out after ${timeout}s waiting for $url"
      return 1
    fi
    sleep 5
    t=$((t + 5))
  done
}

# gpu_info logs the GPU, driver, and interconnect topology of this node.
gpu_info() {
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader >&2 || true
  nvidia-smi topo -m >&2 || true
}

# free_port prints a TCP port nobody listens on. Single-GPU jobs often share
# an 8-GPU node, so servers must not assume fixed ports.
free_port() {
  "${PYTHON:-python3}" -P -c 'import socket; s = socket.socket(); s.bind(("", 0)); print(s.getsockname()[1])'
}

# vllm_args_default: flags every replica gets. Dev mode serves
# /reset_prefix_cache and sleep/wake for experiments; prompt-token details
# report cached tokens in usage; KV events feed the precise index.
vllm_default_args() { # vllm_default_args KV_PORT
  printf '%s\n' --enable-prompt-tokens-details \
    --kv-events-config "{\"enable_kv_cache_events\":true,\"publisher\":\"zmq\",\"endpoint\":\"tcp://*:$1\",\"topic\":\"kv\"}"
}
export VLLM_SERVER_DEV_MODE=1

# start_vllm LOG PORT ARGS... starts vllm serve in the background and sets
# VLLM_PID. stop_vllm stops it. Several may run at once (PIDS).
PIDS=()
start_vllm() {
  local log=$1 port=$2
  shift 2
  "$ENV_DIR/bin/vllm" serve "$@" --host 0.0.0.0 --port "$port" >"$log" 2>&1 &
  VLLM_PID=$!
  PIDS+=("$VLLM_PID")
}
stop_all() {
  for p in "${PIDS[@]:-}"; do [ -n "$p" ] && kill "$p" 2>/dev/null || true; done
  for p in "${PIDS[@]:-}"; do [ -n "$p" ] && wait "$p" 2>/dev/null || true; done
  PIDS=()
}

# startup_times LOG prints the phases of a vLLM start from its log.
startup_times() {
  grep -hoE "Model loading took [0-9.]+ GiB( memory)? and [0-9.]+ seconds|torch.compile takes [0-9.]+ s in total|Graph capturing finished in [0-9]+ secs|init engine \(profile, create kv cache, warmup model\) took [0-9.]+ seconds" "$1" || true
}

# wait_backends URL TOKEN N TIMEOUT waits until the gateway has N healthy backends.
wait_backends() {
  local url=$1 token=$2 n=$3 timeout=$4 t=0 have=0
  while [ "$t" -lt "$timeout" ]; do
    # grep finds nothing while no backend is healthy; under pipefail that
    # must not end the job.
    have=$( (curl -s --noproxy '*' -H "Authorization: Bearer $token" "$url/admin/backends" | grep -o '"healthy":true' || true) | wc -l | tr -d ' ')
    [ "$have" -ge "$n" ] && return 0
    sleep 10
    t=$((t + 10))
  done
  log "only $have of $n backends healthy after ${timeout}s"
  return 1
}

# local_gateway CONFIG_FILE PORT starts a gateway inside the job (for
# experiments that run their own pool) and sets GATEWAY_PID.
local_gateway() {
  "$CODE/bin/linux-amd64/gateway" -config "$1" >"$LOGS/gateway_${SLURM_JOB_ID}_$(basename "$1" .yaml).log" 2>&1 &
  GATEWAY_PID=$!
  PIDS+=("$GATEWAY_PID")
  wait_http "http://127.0.0.1:$2/healthz" 60 "$GATEWAY_PID"
}

# loadgen ARGS... runs the load generator.
loadgen() { "$CODE/bin/linux-amd64/loadgen" "$@"; }
