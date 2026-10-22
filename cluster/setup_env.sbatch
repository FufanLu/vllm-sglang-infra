#!/bin/bash
#SBATCH --job-name=vsi-setup-env
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/%u/vsi/logs/%x_%j.out
#
# Builds the vLLM environment once and packs it for GPU jobs.
#   sbatch /scratch/$USER/vsi/code/cluster/setup_env.sbatch
# Runs on the short partition because its nodes can reach PyPI; GPU nodes
# cannot. The venv is built at the node-local path GPU jobs unpack it to.
set -euo pipefail
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

WHEEL=https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+${CUDA_TAG}-cp38-abi3-manylinux_2_28_x86_64.whl
UV=("$ENV_DIR/bin/uv" pip install --python "$ENV_DIR/bin/python")
log "installing vllm $VLLM_VERSION ($CUDA_TAG) into $ENV_DIR"
if ! "${UV[@]}" --torch-backend "$CUDA_TAG" "vllm @ $WHEEL"; then
  log "retrying with an explicit PyTorch index"
  "${UV[@]}" --extra-index-url "https://download.pytorch.org/whl/$CUDA_TAG" \
    --index-strategy unsafe-best-match "vllm @ $WHEEL"
fi

# NIXL moves KV between prefill and decode instances (PD experiments).
EXTRA_PKGS=${EXTRA_PKGS:-nixl-cu12==1.5.0}
if [ -n "$EXTRA_PKGS" ]; then
  log "installing extras: $EXTRA_PKGS"
  # shellcheck disable=SC2086 # a space-separated list
  "${UV[@]}" --torch-backend "$CUDA_TAG" $EXTRA_PKGS || log "extras failed; PD experiments will not have NIXL"
  # vLLM imports nixl._api, which the nixl meta package provides by loading
  # nixl_cu12. Its own dependencies would also pull nixl-cu13 (CUDA 13,
  # unusable on driver 570), hence --no-deps.
  "${UV[@]}" --no-deps "nixl==1.5.0" || log "nixl meta package failed"
fi

# No GPU here, so this only proves the packages import and shows the CUDA
# runtime torch was built against.
"$ENV_DIR/bin/python" -P -c "
import torch, vllm
print('torch', torch.__version__, 'cuda runtime', torch.version.cuda)
print('vllm', vllm.__version__)
"

log "packing $ENV_TAR"
tar -C /tmp -cf "$ENV_TAR.partial" "$ENV_NAME"
mv "$ENV_TAR.partial" "$ENV_TAR"
du -sh "$ENV_TAR" >&2
log "done"
