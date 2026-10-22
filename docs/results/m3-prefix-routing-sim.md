# M3 — Prefix-aware routing on simulated engines

**These numbers come from simulated engines, not GPUs.** They check that the
mechanism works and that the scheduler balances correctly. The latencies
follow from the simulator's latency model below, so only the ratios mean
anything. The H200 measurement is part of M2/M3 on Explorer.

## Setup

- Four [llm-d-inference-sim](https://github.com/llm-d/llm-d-inference-sim)
  v0.9.0 replicas (`deploy/local/docker-compose.yaml`). Latency model:
  - 20 ms prefill overhead plus 1 ms per *uncached* prompt token, so a cache
    hit skips most of the prefill.
  - 20 ms per output token.
  - KV cache of 2048 blocks × 16 tokens = 32 768 tokens per replica.
- Gateway policies: `round-robin`, `least-load` (queue + KV headroom), and
  `prefix-aware` (approximate prefix index + absolute queue score with
  saturation 32 + KV headroom; weights 1 / 2 / 0.5).
- Workload (`bin/loadgen`):
  - 96 prompt groups, each a ~600-token system prompt; every request adds a
    ~20-token question.
  - Group chosen uniformly; 600 requests; 16 in flight; 16 output tokens.
  - The prefixes total ~59k tokens, more than one replica's cache. Spraying
    every group over every replica therefore thrashes the caches, while
    giving each group a home replica fits ~24 groups (~15k tokens) per
    replica.
- Engines restarted (caches emptied) before every run; identical request
  sequence per seed.

```bash
make up
VSI_SCHEDULER_POLICY=prefix-aware docker compose -f deploy/local/docker-compose.yaml up -d --force-recreate gateway
bin/loadgen -label prefix-aware -requests 600 -concurrency 16 -groups 96 -prefix-words 600 -max-tokens 16 -seed 1
```

## Results

| Policy | Seed | Prompt-cache hit | TTFT p50 / p90 / p99 (ms) | E2E p50 (ms) | Throughput (req/s) | Output tok/s | Requests per replica |
|---|---|---|---|---|---|---|---|
| round-robin | 1 | 38.5% | 660 / 667 / 681 | 977 | 21.3 | 341 | 150 / 150 / 150 / 150 |
| least-load | 1 | 39.2% | 661 / 668 / 689 | 979 | 21.3 | 341 | 153 / 155 / 145 / 147 |
| prefix-aware | 1 | 80.3% | 53 / 662 / 680 | 370 | 32.7 | 524 | 128 / 160 / 176 / 136 |
| round-robin | 2 | 39.0% | 660 / 668 / 684 | 977 | 21.3 | 341 | 150 / 150 / 150 / 150 |
| least-load | 2 | 42.0% | 660 / 667 / 675 | 977 | 21.9 | 351 | 152 / 150 / 151 / 147 |
| prefix-aware | 2 | 80.3% | 53 / 661 / 685 | 370 | 33.7 | 540 | 148 / 163 / 146 / 143 |

## Reading

- **Hit rate doubles (39% → 80%).** The ceiling is about 81%: each group's
  first request is necessarily a miss (96 of 600 requests), and the
  per-request questions are never cached.
- **Median TTFT falls from 660 ms to 53 ms; throughput rises ~55%.** p90 and
  p99 stay near 660 ms because the cold first requests are more than 10% of
  this short run. Longer runs lower the tail.
- **Load stays balanced under affinity.** Each replica serves 128–176
  requests. The absolute queue score lets an endpoint holding a prefix win
  only while it is less than `saturation × w_prefix / w_queue` = 16 requests
  busier than the alternative.
- **Least-load does not beat round-robin here.** The simulator has no
  slowdown under load at this concurrency, so balance is all a load-only
  policy can win. On GPUs, batch size changes step time and the two will
  differ.

## A bug this run found

The first least-load run sent 347 of 600 requests to one replica. Two causes:

1. **Stale load.** Load was max(gateway in-flight, scraped running+waiting).
   Scrapes lag by up to 500 ms, so a replica that had just drained still
   looked busy and was avoided; once a fresh scrape showed it idle, it was
   stampeded.
2. **KV usage decided ties.** Usage differences of a fraction of a percent
   broke ties, so one replica kept winning for a whole scrape interval.

Fixes (in this change):

- Load is now the gateway's exact in-flight count plus only the *external*
  load the engine reports, meaning running + waiting minus what the gateway
  had in flight when the scrape was sent.
- KV usage counts only above 70%, where it actually predicts queueing.

After the fix the spread is 145–155 per replica.
