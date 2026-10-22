# E — MoE on two H200s: DP+EP ranks as endpoints (measured)

**Real hardware.** Two H200s on Explorer node d4053 (NVLink between them),
driver 570.86.15. vLLM 0.26.0 (cu129 build), Qwen3-30B-A3B-FP8 (30B
parameters, 3B active, 128 experts). Job `cluster/moe_bench.sbatch`,
2026-10-22: `11310168` for three configurations, `11310477` for vLLM's
internal balancer (rerun after failure #26).

## Setup

Four ways to serve one model on the same two GPUs:

1. **dp-ep-ranks:** data parallel 2 with expert parallelism. Each rank is
   its own `vllm serve` process on its own port, and the gateway routes to
   ranks as endpoints.
2. **dp-ep-internal:** the same deployment as one `vllm serve`, behind
   vLLM's internal DP balancer (queue-based, not cache-aware).
3. **tp2:** tensor parallel over both GPUs, one endpoint.
4. **replicas:** two independent copies, no collective communication.

Workload: 64 prompt groups with 1,000-word shared prefixes, 800 requests,
32 concurrent, 64 output tokens. Prefix caches are reset before each run.
Kernels: CUTLASS block-FP8 linear layers and Triton fused MoE (DeepGEMM
off, see failure #25).

## Results

| Configuration | Routing | req/s | Output tok/s | Cache hits | TTFT p50 | TTFT p99 | TPOT p50 | E2E p50 | E2E p99 |
|---|---|---|---|---|---|---|---|---|---|
| dp-ep-ranks | least-load | 49.4 | 3159 | 81.2% | 45 ms | 1093 ms | 7 ms | 586 ms | 1591 ms |
| dp-ep-ranks | prefix-aware | 52.5 | 3360 | **88.9%** | 35 ms | **317 ms** | 8 ms | 576 ms | **821 ms** |
| dp-ep-ranks | prefix-aware, saturation 8 | 53.0 | 3390 | 88.9% | 35 ms | 326 ms | 8 ms | 566 ms | 857 ms |
| dp-ep-internal | vLLM's balancer | 45.5 | 2913 | 81.2% | 42 ms | 1064 ms | 8 ms | 639 ms | 1598 ms |
| tp2 | — | 55.3 | 3539 | 88.9% | 34 ms | 1766 ms | 7 ms | 483 ms | 2198 ms |
| replicas | least-load | 55.6 | 3560 | 81.2% | 29 ms | 928 ms | 7 ms | 515 ms | 1420 ms |
| replicas | prefix-aware | 53.9 | 3449 | 88.6% | 28 ms | **227 ms** | 8 ms | 583 ms | **724 ms** |

The prefix-aware split across ranks was 412 / 388 requests (417 / 383 with
saturation 8).

Engine start-up (`init engine` in the vLLM log): 64 s per DP rank, 105 s
for TP=2, 56 s per independent replica.

## Reading

- **Rank-aware routing beats vLLM's own DP balancer.** On the same DP+EP
  deployment, routing to ranks by prefix raises cache hits from 81% to 89%,
  throughput by 15% (45.5 → 52.5 req/s), and cuts p99 TTFT from 1064 to
  317 ms. vLLM's balancer picks the rank by queue length only, so a prefix
  lands on both ranks and is computed twice.
- **Balance is the constraint, and it held.** MoE ranks step in lockstep:
  a rank with no work still runs dummy batches for the collective. Prefix
  routing kept the split at 412 / 388. A lower queue saturation (8, which
  pushes toward balance sooner) changed almost nothing. With 64 prefix
  groups the affinity never had to fight the load term.
- **At this size, DP+EP is not the fastest layout.** The model fits on one
  GPU (31 GB of weights), so two independent replicas avoid the per-step
  all-to-all and are the fastest here: 55.6 req/s. TP=2 matches their
  throughput, but its p99 TTFT is the worst (1766 ms), because long
  prefills hold up the decode steps of every running request. DP+EP pays
  for the expert all-to-all at this scale. It pays off when the experts do
  not fit on one GPU, which is the case it exists for.
- **The same routing helps every multi-endpoint layout.** For replicas,
  prefix-aware routing gives p99 TTFT 928 → 227 ms. For ranks, 1093 → 317 ms.

## Reproduce

```
sbatch cluster/moe_bench.sbatch                          # all four
CONFIGS="dp-ep-internal" sbatch cluster/moe_bench.sbatch  # a subset
```
