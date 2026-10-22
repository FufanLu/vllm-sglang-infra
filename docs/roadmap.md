# Roadmap

Milestones are ordered by dependency and by GPU cost: work that needs no GPU
is done while the H200 queue is long. Every milestone ends with a measured
result, not just code.

## M1 — Gateway core ✅

- OpenAI-compatible `/v1/chat/completions` and `/v1/completions`, streamed
  and not; `/v1/models` aggregated across healthy backends.
- Endpoint registry with static config and an admin API for self-registration.
- Active health probes, passive failure detection (refused connection marks a
  backend down at once), retries on another backend before the first byte.
- Engine metrics scraping with adapters for vLLM (V0 and V1 names) and SGLang.
- Scheduler pipeline (filter → score → pick) with `round-robin` and
  `least-load` (queue + KV usage) policies.
- Client cancellation propagates upstream so the engine stops generating.
- Prometheus series: requests, latency, TTFT, TPOT, tokens, retries, picks,
  per-backend state.
- Local stack: gateway + four llm-d-inference-sim replicas + Prometheus + Grafana.

**Exit:** race-detector test suite green; local stack serves streamed traffic.

## M2 — Bring-up on Explorer H200 🚧

- Packed vLLM environment (vLLM 0.26.0, cu129) unpacked to node-local disk.
- `check_gpu.sbatch`: cu129 runtime on driver 570, Triton compilation, a real
  completion.
- Gateway as a CPU job; replicas as one-GPU jobs that register, and drain
  before their walltime.
- Benchmark harness (vLLM `bench serve` against the gateway) and Grafana
  dashboard.

**Exit:** four H200 replicas behind the gateway; round-robin vs least-load
baseline with P50/P99 TTFT, TPOT, throughput.

## M3 — Prefix-aware routing ✅ (simulated; H200 run in M2)

- ✅ Approximate: the gateway hashes prompt blocks and remembers which
  endpoint served each prefix (LRU, bounded). On simulated engines: hit rate
  39% → 80%, median TTFT 660 → 53 ms ([results](results/m3-prefix-routing-sim.md)).
- ✅ Precise: KV-cache events over ZeroMQ feed an index keyed by chained
  token-block hashes; prompts are tokenized by the engine. With two
  gateways sharing a pool: 70% → 80% hit rate ([results](results/m3-precise-prefix-sim.md)).
- Precise: subscribe to vLLM KV-cache events (ZMQ) and keep a global index of
  which blocks each endpoint holds.
- Scorer combining prefix-match length with load, so affinity yields when an
  endpoint is saturated.

**Exit:** on a shared-system-prompt and multi-turn workload, TTFT and
prefix-cache hit rate vs least-load, approximate vs precise.

## M4 — Tenancy and metering ✅

- API keys mapped to tenants; keys stored hashed.
- Token quotas per tenant (Redis): reserve `prompt + max_tokens` up front,
  settle with actual usage, refund the difference.
- Usage records (PostgreSQL) per request: tenant, model, tokens, cache hits,
  latency.
- Inject `x-llm-d-inference-fairness-id` and strip client-supplied copies.

**Exit:** quota enforcement under concurrent load; usage totals match engine
token counts.

## M5 — Cold start and Slurm autoscaling (D) 🚧

Autoscaler done (Slurm, Kubernetes, and process providers; walltime
handover; tested; [local demo](results/d-autoscale-local.md)). Cold-start
measurements are scripted (`cluster/coldstart.sbatch`) and need H200 time.

- Measure ready time of a new replica under weight-staging strategies: NFS
  direct, sequential pre-read, copy to node-local NVMe, page-cache reuse.
- vLLM sleep/wake to swap models on one GPU in seconds.
- Autoscaler: scale on queue depth; submit replacements before a replica's
  walltime and drain the old one.

**Exit:** ready-time table per strategy; zero failed requests across a
walltime handover.

## M6 — MoE with DP+EP (E)

- Qwen3-30B-A3B (FP8) with data-parallel attention and expert parallelism on
  two H200s (`gpu-short`), each rank exposed on its own port.
- The scheduler routes to ranks. Ranks of a MoE deployment step in lockstep —
  an idle rank runs dummy batches — so concentrating load for cache affinity
  costs more than with independent replicas.
- Compare: our rank-aware routing, vLLM's internal DP balancer (queue-based,
  not KV-aware), TP=2, two independent replicas.

**Exit:** throughput and TTFT for the four setups on a shared-prefix workload.

## M7 — KV-cache tiering and sharing (A)

**Done on one H200:** with 6 GiB of GPU KV and ~100k tokens of prefixes,
a 48 GiB CPU tier raises cache hits from 60% to 99.6% and throughput by
27–53%; the gateway's index tracks both tiers
([results](results/a-kvtier-h200.md)). Cross-replica sharing is not done.

- CPU offload of evicted KV blocks (vLLM native offload or LMCache); H200
  nodes have 1.5 TB of host memory.
- Cross-replica sharing so a replica can fetch another's prefix instead of
  recomputing it.
- Tier-aware scoring: GPU hit > CPU hit > peer hit > miss.

**Exit:** hit rate and TTFT vs GPU-only caching as the working set grows.

## M8 — Flow control (C) ✅

On simulated engines: interactive TTFT p99 4768 → 293 ms under a batch
flood ([results](results/c-flow-control-sim.md)).

- Priority bands (interactive above batch), token-fair queuing across
  tenants within a band, request TTL.
- SLO-aware admission: hold or shed lower-priority traffic when predicted
  TTFT would breach the interactive SLO.

**Exit:** interactive P99 TTFT under a batch flood, with and without flow
control; per-tenant token share.

## M9 — Prefill/decode disaggregation (B) 🚧

Gateway orchestration done and tested (prefill with `do_remote_decode`,
pass-through of `kv_transfer_params`, fallback). Needs the H200 run.

- One prefill and one decode vLLM on two H200s of one node, KV over NIXL.
- The gateway picks a prefill/decode pair per request and orchestrates the
  two phases.

**Exit:** TTFT and TPOT vs the same two GPUs serving aggregated.

## M10 — Kubernetes ✅

Pod discovery by label, Deployment scaling, Helm chart; on kind with
simulated engines: 4 pods discovered, precise routing at 83% cache hits,
autoscaling 4 → 1 → 5 → 1 with demand ([results](results/m10-kubernetes-kind.md)).

- Helm chart for the gateway; kind demo with simulated engines.
- Gateway in front of llm-d's router, injecting the tenant header.

## M11 — Cloud overflow ✅

- External OpenAI-compatible providers as a backend class with per-token cost.
- Overflow when local replicas are saturated or absent; tenants marked
  on-prem-only never leave the cluster.
