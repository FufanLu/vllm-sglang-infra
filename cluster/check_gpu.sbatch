#!/bin/bash
#SBATCH --job-name=vsi-check-gpu
#SBATCH --partition=gpu,gpu-short
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/%u/vsi/logs/%x_%j.out
#
# Proves the stack works on an H200 before anything is built on it: the cu129
# CUDA runtime on driver 570, torch on the GPU, Triton/torch.compile kernels
# (vLLM compiles them at startup), and a real chat completion.
#   sbatch /scratch/$USER/vsi/code/cluster/check_gpu.sbatch [model-name]
set -euo pipefail
source "${VSI_CODE:-/scratch/$USER/vsi/code}/cluster/common.sh"
MODEL=${1:-Qwen3-0.6B}
PORT=8000

gpu_info
stage_env
"$ENV_DIR/bin/python" -P -c "
import torch
print('torch', torch.__version__, 'cuda runtime', torch.version.cuda, 'available', torch.cuda.is_available())
x = torch.randn(4096, 4096, device='cuda')
print('matmul ok on', torch.cuda.get_device_name(0), float((x @ x).abs().mean()))
"
LOCAL_MODEL=$(stage_model "$MODEL")

VLOG=$LOGS/vllm_check_${SLURM_JOB_ID}.log
log "starting vllm serve (log: $VLOG)"
t0=$SECONDS
"$ENV_DIR/bin/vllm" serve "$LOCAL_MODEL" --served-model-name "$MODEL" \
  --host 127.0.0.1 --port "$PORT" --max-model-len 4096 --gpu-memory-utilization 0.5 >"$VLOG" 2>&1 &
VLLM_PID=$!
trap 'kill "$VLLM_PID" 2>/dev/null || true' EXIT
if ! wait_http "http://127.0.0.1:$PORT/health" 1200 "$VLLM_PID"; then
  tail -40 "$VLOG" >&2
  exit 1
fi
log "vllm ready in $((SECONDS - t0))s"

curl -s --noproxy '*' "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with one word: hello\"}],\"max_tokens\":32}"
echo
curl -s --noproxy '*' "http://127.0.0.1:$PORT/metrics" | grep -E '^vllm:(num_requests_running|num_requests_waiting|kv_cache_usage_perc)'
log "CHECK OK"
