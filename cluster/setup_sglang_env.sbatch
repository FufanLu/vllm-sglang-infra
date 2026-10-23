#!/bin/bash
#SBATCH --job-name=vsi-setup-sglang
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/%u/vsi/logs/%x_%j.out
#
# Builds the SGLang environment once and packs it for GPU jobs, like
# setup_env.sbatch does for vLLM.
#   sbatch /scratch/$USER/vsi/code/cluster/setup_sglang_env.sbatch
# SGLang 0.5.10 pins torch 2.9.1, whose PyPI build is CUDA 12.8: the newest
# runtime the H200 nodes' driver 570 supports. Later SGLang releases need
# CUDA 13.
set -euo pipefail
export VSI_ENGINE=sglang
source "${VSI_CODE:-/scratch/$USER/vsi/code}/cluster/common.sh"

module load anaconda3/2024.06 >/dev/null 2>&1 || true
PYBASE=$(command -v python3)
"$PYBASE" -c 'import sys; assert sys.version_info >= (3, 10), sys.version'
free_gb=$(df -Pk /tmp | awk 'NR == 2 { print int($4 / 1048576) }')
[ "$free_gb" -ge 30 ] || { log "only ${free_gb}G free in /tmp on $(hostname)"; exit 1; }

export UV_CACHE_DIR=/tmp/vsi-uv-cache-$SLURM_JOB_ID UV_LINK_MODE=copy UV_NO_PROGRESS=1
trap 'rm -rf "$UV_CACHE_DIR"' EXIT
rm -rf "$ENV_DIR"
"$PYBASE" -m venv "$ENV_DIR"
"$ENV_DIR/bin/pip" install -q --upgrade pip uv
log "installing sglang $SGLANG_VERSION into $ENV_DIR"
"$ENV_DIR/bin/uv" pip install --python "$ENV_DIR/bin/python" "sglang==$SGLANG_VERSION"

# No GPU here: this proves the packages import and shows torch's CUDA build.
"$ENV_DIR/bin/python" -I -c "
import importlib.util, torch, sglang, flashinfer
print('torch', torch.__version__, 'cuda runtime', torch.version.cuda)
print('sglang', sglang.__version__, 'flashinfer', flashinfer.__version__)
print('sgl_kernel', 'present' if importlib.util.find_spec('sgl_kernel') else 'MISSING')
assert torch.version.cuda.startswith('12.'), 'driver 570 needs a CUDA 12 build'
"

log "packing $ENV_TAR"
tar -C /tmp -cf "$ENV_TAR.partial" "$ENV_NAME"
mv "$ENV_TAR.partial" "$ENV_TAR"
du -sh "$ENV_TAR" >&2
log "done"
