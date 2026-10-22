# Design

## 1. Goals

1. Serve open models from a pool of vLLM / SGLang replicas behind one
   OpenAI-compatible endpoint, on a Slurm GPU cluster.
2. Put each request where it finishes fastest: account for load, KV-cache
   pressure, and which replica already holds the request's prefix.
3. Run the pool as a multi-tenant service: authenticate tenants, enforce
   token quotas, meter usage, protect interactive traffic from batch floods.
4. Operate it: replicas come and go with Slurm jobs, drain before their
   walltime, and scale with demand.
5. Measure everything on real hardware (H200), against honest baselines.

**Non-goals.** Modifying inference engines or writing kernels; the platform
uses engine features (prefix caching, KV offload, KV transfer, DP/EP) and
decides when and where to use them. Multi-node wide expert parallelism, which
the cluster cannot schedule.

## 2. Environment constraints

The target cluster is Northeastern's Explorer. Its constraints shape the
design as much as the goals do.

| Constraint | Consequence |
|---|---|
| `gpu` partition: 1 GPU per job, 4 per user, 8 h walltime | A replica is a one-GPU job; the pool is at most 4 replicas; replicas must drain and be replaced before 8 h |
| `gpu-short`: 2 GPUs per job, 2 h | MoE DP+EP and prefill/decode experiments run as 2-GPU jobs and must be scripted end to end |
| H200 driver 570 (CUDA 12.8); vLLM ships cu129/cu130 wheels only | Use the cu129 build under CUDA minor-version compatibility; verified per node type by `check_gpu.sbatch` |
| Compute nodes have no internet | Environments and weights are prepared on the `short` partition |
| `/scratch` is NFS: slow for many small files and for mmap | Pack the Python env as one tarball and copy weights to node-local NVMe, once per node |
| No root, no Kubernetes | The control plane talks to Slurm (`sbatch`, `scancel`); Kubernetes support is a packaging target, demonstrated on kind |

## 3. Components

```
gateway (Go, one process)
├── HTTP front: /v1/chat/completions, /v1/completions, /v1/models, /healthz, /readyz, /metrics
├── admin API: PUT/DELETE/GET /admin/backends — replicas register themselves
├── backend.Registry: endpoints by name; runtime state per endpoint
├── backend.Monitor: health probes, /metrics scraping, model discovery
├── scheduler: filter → score → pick
└── proxy: forward, relay (SSE-aware), retry, measure
```

An **endpoint** is anything routable: a replica, or one data-parallel rank of
a MoE deployment exposed on its own port. Endpoints carry labels (Slurm job,
node, GPU, and later role and rank) that filters can use.

## 4. Request lifecycle

1. Read the body (bounded), parse only `model`, `stream`, `messages`,
   `prompt`. The body is forwarded byte for byte.
2. Flatten the prompt (roles and text in order) for prefix-aware scorers.
3. Schedule: filters, then scorers, then the picker, under one lock that also
   takes an in-flight slot on the chosen endpoint.
4. Forward with the client's context, so a client disconnect aborts the
   upstream request and the engine stops generating. Client `Authorization`
   and `Accept-Encoding` are not forwarded.
5. Retry on another endpoint if the connection fails or the backend answers
   502/503/504 before any byte has been written to the client. After the
   first byte the response is committed to that backend.
6. Relay. Server-sent events are flushed per read; the relay parses events as
   they pass to time the first and last token and to read the usage object.
7. Release the slot; record outcome, latency, TTFT, TPOT, tokens.

## 5. Scheduling

The pipeline mirrors llm-d's Endpoint Picker so policies stay small and
composable, and so results can be compared with llm-d's.

**Filters** (M1): `healthy` (last probe passed), `model` (endpoint serves
the requested model; unknown model lists pass).

**Scorers** (M1), each in [0, 1]:

- `queue`: load_i = max(inflight_i, running_i + waiting_i), score =
  (max − load_i) / (max − min), or 1 when all are equal. The gateway's own
  in-flight count is exact and instant; the engine's counts cover traffic
  that bypassed the gateway.
- `kv-usage`: 1 − kv_usage_i, or 0.5 when the snapshot is older than 5 s, so
  a stale endpoint neither looks idle nor full.

**Pickers**: `max-score` (ties broken uniformly at random) and
`round-robin` (ignores scores; the baseline).

Picks are serialized. Scoring costs microseconds, and serializing means a
burst of requests sees each other's in-flight slots instead of all choosing
the endpoint that looked idle at the last scrape.

**Planned scorers.**

- `prefix` (M3): fraction of the request's prompt blocks already cached on
  the endpoint. Approximate mode keeps a bounded LRU of block-hash →
  endpoint learned from the gateway's own routing; precise mode consumes vLLM
  KV-cache events. Combined with `queue` so that affinity yields under load.
- DP-rank-aware routing (M6): ranks are endpoints. Because MoE ranks step in
  lockstep, imbalance wastes every rank's step, so the load term carries more
  weight than with independent replicas.
- Tier-aware prefix (M7): GPU hit > CPU hit > peer hit > miss.
- Pair selection for prefill/decode (M9): choose a prefill endpoint by
  prefix and queue, a decode endpoint by KV headroom.
- Cost (M11): external providers score lower unless local capacity is
  exhausted.

## 6. Discovery and health

- Static backends come from config. Dynamic ones register through the admin
  API (bearer token, constant-time compare). A Slurm replica registers after
  its `/health` answers and deregisters when it drains.
- A (re)registered endpoint with a new URL starts unhealthy and needs one
  passing probe; re-registering the same URL keeps its state.
- Health: healthy on the first passing probe, unhealthy after
  `failure_threshold` consecutive failures, or at once on a refused
  connection seen by the proxy.
- Metrics adapters map engine series to one snapshot: vLLM
  `vllm:num_requests_running|waiting`, `vllm:kv_cache_usage_perc` (V1) or
  `vllm:gpu_cache_usage_perc` (V0); SGLang `sglang:num_running_reqs`,
  `sglang:num_queue_reqs`, `sglang:token_usage`. Counts are summed across
  series and KV usage takes the maximum, which is right for servers that
  front several DP engine cores.

## 7. Failure handling

| Failure | Behaviour |
|---|---|
| Connection refused / reset before response | Mark endpoint down, retry elsewhere (up to `max_retries`) |
| 502 / 503 / 504 before the first byte | Retry elsewhere |
| 4xx, or 5xx after bytes were sent | Pass through; never retried |
| Client disconnects | Upstream request cancelled; counted as `client_cancel` |
| Replica reaches walltime | SIGTERM 10 min early: deregister, wait for in-flight requests, stop vLLM |
| No eligible endpoint | 503 with an OpenAI-style error |

## 8. Tenancy and quotas (M4)

- API keys (stored as hashes) map to a tenant with a plan: token budget per
  minute, priority class, data-residency flag.
- Quotas are in tokens, not requests: one request can be 10 tokens or
  100 000. The output length is unknown up front, so the gateway reserves
  `prompt_tokens + max_tokens` in a Redis token bucket and settles with the
  usage the engine reports, refunding the difference.
- Every request produces a usage record (PostgreSQL): tenant, model,
  backend, prompt / cached / completion tokens, TTFT, duration, outcome.
- The gateway strips any client-supplied `x-llm-d-inference-fairness-id` and
  injects the authenticated tenant, which is the contract llm-d's flow
  control expects from an upstream gateway.

## 9. Flow control (M8)

Requests are queued in the gateway by (priority band, tenant). Bands are
served in strict order; within a band tenants take turns weighted by tokens
consumed, so a tenant with long prompts cannot starve others. When the pool
is saturated, requests wait in the gateway rather than in engine queues
("no-regret" dispatch), lower bands are bounded, and admission is refused
when the predicted TTFT of the interactive band would exceed its SLO.

## 10. Cold start and autoscaling (M5)

A new replica's ready time is: queue wait + env unpack + weight staging +
engine start (load, compile, CUDA graphs). The first three dominate on this
cluster and are what M5 measures and reduces. The Slurm autoscaler scales on
queue depth with hysteresis, and replaces each replica before its walltime:
submit the successor, wait until it registers, then drain the predecessor.

## 11. Kubernetes and llm-d (M10)

The gateway ships as a container and a Helm chart. On Kubernetes it can stand
in front of llm-d's router: the gateway owns identity, quotas, metering, and
overflow; llm-d owns endpoint picking inside the pool. Demonstrated on kind
with llm-d-inference-sim.

## 12. Cloud overflow (M11)

External OpenAI-compatible providers are a backend class with a per-token
cost and higher latency. They are eligible only when local endpoints are
saturated or absent, and never for tenants marked on-prem-only. Overflowed
usage is metered with the provider's cost.

## 13. Observability

Gateway series (prefix `vsi_`): `requests_total{backend,route,code}`,
`request_duration_seconds`, `ttft_seconds`, `tpot_seconds`,
`tokens_total{kind}`, `retries_total{reason}`,
`scheduler_picks_total{policy,backend}`, and per-backend `backend_healthy`,
`backend_inflight`, `backend_engine_running|waiting|kv_usage`. Engine series
are scraped by Prometheus directly. TTFT is measured at the gateway from
request arrival to the first event carrying generated text, so it includes
queueing and retries — what the client experiences.

## 14. Benchmark methodology

- Workloads: shared system prompt (customer-support style), multi-turn
  conversations replayed turn by turn, long-document QA, and a mix of them
  across tenants.
- Load: Poisson arrivals at increasing rates until saturation; each point
  long enough for steady state.
- Metrics: P50/P99 TTFT and TPOT, output tokens/s, prefix-cache hit rate
  (from engine counters), goodput = requests/s meeting both TTFT and TPOT SLOs.
- Every policy runs on the same replicas, model, and request trace; engine
  versions and flags are recorded with the results.

## 15. Risks

| Risk | Mitigation |
|---|---|
| cu129 build fails on driver 570 (Triton or FlashInfer JIT) | `check_gpu.sbatch` first; fall back to building vLLM against torch cu128, or the CUDA forward-compatibility package |
| H200 queue is long | Develop against simulators; batch GPU experiments; run 2-GPU work on `gpu-short` |
| Cross-job networking between nodes blocked | Verified in M2 bring-up; fall back to co-locating gateway and replicas on one node |
| SGLang's default wheels need CUDA 13 | SGLang support is best effort; the engine adapter keeps the gateway engine-agnostic |
