#!/bin/bash
#SBATCH --job-name=vsi-handover
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/%u/vsi/logs/%x_%j.out
#
# D: walltime handover under load. The gateway runs here with the Slurm
# autoscaler (min = max = 1) and launches the replicas itself, each with a
# short walltime. A replica ending within HANDOVER_LEAD stops counting as
# capacity, so the autoscaler submits its successor while it still serves;
# Slurm's TERM ten minutes before the walltime makes the old one deregister
# and drain. Load runs the whole time; the result is the error count and a
# timeline of which replicas served.
#   sbatch handover_bench.sbatch [model]
# It writes $ROOT/gateway.addr, so do not run it beside a routing campaign.
set -euo pipefail
source "${VSI_CODE:-/scratch/$USER/vsi/code}/cluster/common.sh"
MODEL=${1:-Qwen3-8B}
OUT=$ROOT/results/handover-$(date +%Y%m%d-%H%M%S)
mkdir -p "$OUT"
REPLICA_WALLTIME=${REPLICA_WALLTIME:-00:30:00}
HANDOVER_LEAD=${HANDOVER_LEAD:-25m}
DURATION=${DURATION:-3000} # seconds of load
POOL=vsi-handover-r$SLURM_JOB_ID
# The gateway submits the replicas from this node.
command -v sbatch >/dev/null && command -v squeue >/dev/null || { log "no Slurm client on $(hostname)"; exit 1; }

if [ ! -s "$ROOT/admin.token" ]; then
  (umask 077 && head -c 24 /dev/urandom | od -An -tx1 | tr -d ' \n' >"$ROOT/admin.token")
fi
export VSI_ADMIN_TOKEN
VSI_ADMIN_TOKEN=$(cat "$ROOT/admin.token")
AUTH=$(mktemp /tmp/vsi-auth-XXXXXX)
printf 'Authorization: Bearer %s\n' "$VSI_ADMIN_TOKEN" >"$AUTH"

cleanup() { # gateway first, so it launches nothing more; then the replicas
  stop_all
  rm -f "$ROOT/gateway.addr" "$AUTH"
  for try in 1 2 3 4 5 6; do
    left=$(squeue -h --me -n "$POOL" -t PENDING,RUNNING -o %i 2>/dev/null | tr '\n' ' ') || left=x
    [ -z "${left// /}" ] && break
    [ "$left" = x ] || scancel $left || true
    sleep 20
  done
}
trap cleanup EXIT

PORT=$(free_port)
cfg=$OUT/gateway.yaml
cat >"$cfg" <<EOF
listen: ":$PORT"
scheduler: { policy: least-load }
health: { interval: 2s, timeout: 1s, failure_threshold: 2 }
autoscale:
  enabled: true
  provider: slurm
  min: 1
  max: 1
  target_inflight_per_replica: 64
  scale_down_after: 1h
  handover_lead: $HANDOVER_LEAD
  interval: 15s
  slurm:
    script: $CODE/cluster/vllm_replica.sbatch
    args: [$MODEL, --max-model-len, "8192"]
    job_name: $POOL
EOF
# sbatch run by the gateway takes these; environment beats #SBATCH lines.
export SBATCH_TIMELIMIT=$REPLICA_WALLTIME
echo "http://$(hostname):$PORT" >"$ROOT/gateway.addr"
local_gateway "$cfg" "$PORT"
GW=http://127.0.0.1:$PORT
log "gateway at $(cat "$ROOT/gateway.addr"); replica walltime $REPLICA_WALLTIME, handover lead $HANDOVER_LEAD"

wait_backends "$GW" "$VSI_ADMIN_TOKEN" 1 3600

# Timeline: every 15 s, the healthy replicas and the pool's Slurm jobs.
(
  while true; do
    healthy=$(curl -s --noproxy '*' -H "@$AUTH" "$GW/admin/backends" |
      grep -oE '"name":"[^"]+"|"healthy":(true|false)' | paste -d' ' - - | grep true | cut -d'"' -f4 | tr '\n' ' ' || true)
    jobs=$(squeue -h --me -n "$POOL" -o '%i:%T' 2>/dev/null | tr '\n' ' ' || true)
    echo "$(date +%H:%M:%S) healthy=[${healthy% }] jobs=[${jobs% }]"
    sleep 15
  done
) >"$OUT/timeline.txt" 2>&1 &
PIDS+=("$!")

# Load in rounds of 100 requests until DURATION is up.
WORK=(-model "$MODEL" -requests 100 -concurrency 8 -groups 16 -prefix-words 300 -max-tokens 64)
end=$((SECONDS + DURATION)); round=0
while [ "$SECONDS" -lt "$end" ]; do
  round=$((round + 1))
  loadgen -url "$GW" -label "round-$round" "${WORK[@]}" -seed "$round" -json | grep '^{' >>"$OUT/rounds.jsonl" || true
done
curl -s --noproxy '*' "$GW/metrics" | grep -E '^vsi_(requests_total|retries_total|autoscale)' >"$OUT/metrics.txt" || true
awk -F'"errors":' '{split($2,a,","); e+=a[1]; n++} END {print n" rounds, "e" errors"}' "$OUT/rounds.jsonl" | tee "$OUT/summary.txt"
log "results in $OUT"
