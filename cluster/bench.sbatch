#!/bin/bash
#SBATCH --job-name=vsi-bench
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/%u/vsi/logs/%x_%j.out
#
# Routing experiments (M2, M3) against the live pool behind gateway.sbatch.
# Waits for N healthy replicas, then for each policy: switch the gateway's
# policy, reset every engine's prefix cache, replay the same workload.
#   sbatch bench.sbatch [N]          (cluster/campaign.sh routing submits it)
# Results: $ROOT/results/routing-<time>/<policy>.txt (summary + JSON line).
set -euo pipefail
source "${VSI_CODE:-/scratch/$USER/vsi/code}/cluster/common.sh"
N=${1:-4}
GW=$(cat "$ROOT/gateway.addr")
AUTH=$(mktemp /tmp/vsi-auth-XXXXXX)
trap 'rm -f "$AUTH"' EXIT
printf 'Authorization: Bearer %s\n' "$(cat "$ROOT/admin.token")" >"$AUTH"
admin() { curl -s --noproxy '*' -X "$1" -H "@$AUTH" -H 'Content-Type: application/json' "$GW$2" ${3:+-d "$3"}; }

OUT=$ROOT/results/routing-$(date +%Y%m%d-%H%M%S)
mkdir -p "$OUT"
log "waiting for $N healthy replicas behind $GW"
wait_backends "$GW" "$(cat "$ROOT/admin.token")" "$N" 14400
admin GET /admin/backends >"$OUT/backends.json"

MODEL=${MODEL:-Qwen3-8B}
WORKLOAD=(-model "$MODEL" -requests "${REQUESTS:-1200}" -concurrency "${CONCURRENCY:-32}"
  -groups "${GROUPS:-96}" -prefix-words "${PREFIX_WORDS:-1500}" -question-words 24
  -max-tokens "${MAX_TOKENS:-64}" -seed "${SEED:-1}")
for p in ${POLICIES:-round-robin least-load prefix-aware precise-prefix-aware}; do
  admin PUT /admin/policy "{\"policy\":\"$p\"}" >/dev/null
  admin POST /admin/reset_prefix_cache >"$OUT/reset-$p.json"
  sleep 3
  log "policy $p"
  loadgen -url "$GW" -label "$p" "${WORKLOAD[@]}" -json | tee "$OUT/$p.txt"
  curl -s --noproxy '*' "$GW/metrics" >"$OUT/$p.metrics"
done
log "results in $OUT"
if [ -n "${CLEANUP_JOBS:-}" ]; then
  log "cancelling $CLEANUP_JOBS"
  # shellcheck disable=SC2086
  scancel --signal=TERM --batch $CLEANUP_JOBS || true
fi
