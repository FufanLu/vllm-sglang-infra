#!/bin/bash
#SBATCH --job-name=vsi-coldstart
#SBATCH --partition=gpu,gpu-short
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/%u/vsi/logs/%x_%j.out
#
# D: where a new replica's minutes go, and what shortens them.
#   sbatch coldstart.sbatch [model]
# Measures: env unpack, weight staging (and raw NFS read speed with O_DIRECT,
# which bypasses the page cache), vLLM start with a cold and a warm
# torch.compile cache, start straight from /scratch, and sleep/wake.
set -euo pipefail
source "${VSI_CODE:-/scratch/$USER/vsi/code}/cluster/common.sh"
MODEL=${1:-Qwen3-8B}
OUT=$ROOT/results/coldstart-$(date +%Y%m%d-%H%M%S)
mkdir -p "$OUT"
trap stop_all EXIT
gpu_info 2>"$OUT/gpu.txt"
report() { echo "$1: $2" | tee -a "$OUT/summary.txt"; }

t=$SECONDS; stage_env; report env_unpack_s $((SECONDS - t))
PYTHON=$ENV_DIR/bin/python

big=$(ls -S "$MODELS/$MODEL"/*.safetensors | head -1)
size_mb=$(( $(stat -c %s "$big") / 1048576 ))
t=$SECONDS; dd if="$big" of=/dev/null bs=16M iflag=direct status=none; dt=$((SECONDS - t))
report nfs_direct_read_MBps $(( size_mb / (dt > 0 ? dt : 1) ))
report model_size_GB "$(du -sm "$MODELS/$MODEL" | awk '{printf "%.1f", $1/1024}')"
t=$SECONDS; LOCAL=$(stage_model "$MODEL"); report stage_to_local_nvme_s $((SECONDS - t))

run() { # run NAME CACHE_DIR MODEL_PATH [extra args]
  local name=$1 cache=$2 path=$3
  shift 3
  local port; port=$(free_port)
  local vlog=$OUT/vllm_$name.log
  local t0=$SECONDS
  VLLM_CACHE_ROOT=$cache start_vllm "$vlog" "$port" "$path" --served-model-name "$MODEL" --max-model-len 8192 "$@"
  wait_http "http://127.0.0.1:$port/health" 1800 "$VLLM_PID"
  report "${name}_ready_s" $((SECONDS - t0))
  startup_times "$vlog" | sed "s/^/  $name: /" | tee -a "$OUT/summary.txt"
  PORT=$port
}

COLD=/tmp/vsi-compile-cold-$SLURM_JOB_ID
rm -rf "$COLD"
run cold_cache "$COLD" "$LOCAL"; stop_all
run warm_cache "$COLD" "$LOCAL"; stop_all
run from_nfs "$COLD" "$MODELS/$MODEL"; stop_all

# Sleep mode: offload weights to CPU and free the GPU, then wake.
run sleep_mode "$COLD" "$LOCAL" --enable-sleep-mode
req() { curl -s --noproxy '*' -o /dev/null -w '%{time_total}' "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":8}"; }
report first_request_s "$(req)"
since() { "$PYTHON" -P -c "import sys, time; print(f'{time.time() - float(sys.argv[1]):.2f}')" "$1"; }
t=$(date +%s.%N); curl -s --noproxy '*' -X POST "http://127.0.0.1:$PORT/sleep?level=1" >/dev/null
report sleep_s "$(since "$t")"
nvidia-smi --query-gpu=memory.used --format=csv,noheader | sed 's/^/gpu_memory_while_asleep: /' | tee -a "$OUT/summary.txt"
t=$(date +%s.%N); curl -s --noproxy '*' -X POST "http://127.0.0.1:$PORT/wake_up" >/dev/null
report wake_s "$(since "$t")"
report first_request_after_wake_s "$(req)"
stop_all
log "results in $OUT/summary.txt"
cat "$OUT/summary.txt"
