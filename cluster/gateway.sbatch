#!/bin/bash
#SBATCH --job-name=vsi-gateway
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/%u/vsi/logs/%x_%j.out
#
# Runs the gateway on a CPU node. Replica jobs find it through
# $ROOT/gateway.addr and register themselves with the admin token in
# $ROOT/admin.token (created here on first use, readable only by you).
#   sbatch /scratch/$USER/vsi/code/cluster/gateway.sbatch
set -euo pipefail
source "${VSI_CODE:-/scratch/$USER/vsi/code}/cluster/common.sh"
PORT=${PORT:-8080}
CONFIG=${GATEWAY_CONFIG:-$CODE/cluster/gateway.yaml}

if [ ! -s "$ROOT/admin.token" ]; then
  (umask 077 && head -c 24 /dev/urandom | od -An -tx1 | tr -d ' \n' >"$ROOT/admin.token")
fi
export VSI_ADMIN_TOKEN
VSI_ADMIN_TOKEN=$(cat "$ROOT/admin.token")

echo "http://$(hostname):$PORT" >"$ROOT/gateway.addr"
trap 'rm -f "$ROOT/gateway.addr"' EXIT
log "gateway at $(cat "$ROOT/gateway.addr")"
"$CODE/bin/linux-amd64/gateway" -config "$CONFIG"
