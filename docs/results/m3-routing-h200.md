# M2/M3 — Routing policies on four H200 replicas (measured)

**Real hardware.** Four vLLM 0.26.0 replicas of Qwen3-8B (bf16), one H200
each, as separate Slurm jobs: three on node d4053, one on d4055. The
gateway runs as its own job on a CPU node (c0630) and the replicas register
with it over the cluster network. The bench job switches the gateway's
policy, resets every engine's prefix cache, and replays the same workload.
`cluster/campaign.sh routing`, 2026-10-22 (bench `11310154`).

## Setup

- Each replica gets 12 GiB of KV cache (`--kv-cache-memory-bytes 12G`), about
  85k tokens.
- Workload: 96 prompt groups, each a shared 1,500-word prefix plus a
  24-word question; 1,200 requests, 32 concurrent, 64 output tokens.
- The prefixes total about 190k tokens. That is more than one replica holds
  and less than four together.

## Results

| Policy | Cache hits | req/s | Output tok/s | TTFT p50 | TTFT p99 | TPOT p50 | E2E p50 | E2E p99 | Requests per replica |
|---|---|---|---|---|---|---|---|---|---|
| round-robin | 47.4% | 61.7 | 3952 | 45 ms | 279 ms | 7 ms | 500 ms | 796 ms | 300 / 300 / 300 / 300 |
| least-load | 49.4% | 62.1 | 3971 | 72 ms | 250 ms | 6 ms | 506 ms | 665 ms | 304 / 296 / 304 / 296 |
| prefix-aware | **89.1%** | **76.7** | 4910 | **21 ms** | 229 ms | 5 ms | **393 ms** | 651 ms | 304 / 275 / 296 / 325 |
| precise-prefix-aware | **89.2%** | 76.0 | 4866 | 26 ms | 299 ms | 5 ms | 399 ms | 738 ms | 283 / 319 / 307 / 291 |

After the precise run the gateway's index held 3,905–4,263 GPU blocks per
replica (about 64k tokens each), with no orphaned blocks.

## Reading

- **Prefix affinity nearly doubles the cache hit rate.** The load-only
  policies send every prefix to every replica. Each replica would then need
  all 190k tokens and holds 85k, so about half of each prompt is recomputed.
  The prefix-aware policies give each prefix a home replica. Each replica
  then sees about a quarter of the prefixes, and they fit.
- **That is +24% throughput on the same four GPUs**, and the median TTFT
  falls from 45 ms (round-robin) to 21 ms.
- **Balance holds.** The busiest replica got 325 of 1,200 requests, against
  300 for an even split. The queue term in the score stops a popular prefix
  from piling onto one replica.
- **Approximate and precise agree here.** With one gateway, the approximate
  index sees every routing decision and predicts the caches well. The
  precise index matters when it cannot: several gateways, restarts, or
  evictions it does not see. On simulators, with two gateways in front of
  one pool, precise routing kept 80% hits against 70%
  ([results](m3-precise-prefix-sim.md)).
- **least-load did not beat round-robin.** With equal replicas and uniform
  requests, round-robin is already balanced. least-load's median TTFT was
  higher (72 ms against 45 ms), while its p99 was lower.

## Reproduce

```
cluster/push.sh
ssh explorer 'bash /scratch/$USER/vsi/code/cluster/campaign.sh routing'
```
