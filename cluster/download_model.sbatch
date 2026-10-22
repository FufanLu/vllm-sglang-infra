#!/bin/bash
#SBATCH --job-name=vsi-download
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/%u/vsi/logs/%x_%j.out
#
# Downloads a Hugging Face model to $ROOT/models/<name>.
#   sbatch /scratch/$USER/vsi/code/cluster/download_model.sbatch Qwen/Qwen3-0.6B
#   sbatch .../download_model.sbatch Qwen/Qwen3-30B-A3B-FP8 Qwen3-30B-A3B-FP8
# The login node downloads slowly; a short-partition job is much faster.
# Each job gets its own xet cache: a shared one on NFS stalls with stale handles.
set -euo pipefail
source "${VSI_CODE:-/scratch/$USER/vsi/code}/cluster/common.sh"
REPO=${1:?usage: download_model.sbatch <hf-repo-id> [local-name]}
NAME=${2:-${REPO##*/}}
DEST=$MODELS/$NAME

module load anaconda3/2024.06 >/dev/null 2>&1 || true
VENV=/tmp/vsi-hf-venv-$SLURM_JOB_ID
export HF_HUB_OFFLINE=0 HF_XET_CACHE=/tmp/vsi-xet-$SLURM_JOB_ID HF_XET_CHUNK_CACHE_SIZE_BYTES=0
export HF_HUB_DISABLE_PROGRESS_BARS=1
trap 'rm -rf "$VENV" "$HF_XET_CACHE"' EXIT
python3 -m venv "$VENV"
"$VENV/bin/pip" install -q "huggingface_hub[hf_xet]"

t0=$SECONDS
"$VENV/bin/python" -P - "$REPO" "$DEST" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo, dest = sys.argv[1], sys.argv[2]
snapshot_download(repo_id=repo, local_dir=dest,
                  allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken", "*.jinja", "*.py"])
PY
rm -rf "$DEST/.cache"
touch "$DEST/.complete"
log "downloaded $REPO -> $DEST ($(du -sh "$DEST" | cut -f1)) in $((SECONDS - t0))s"
