# vllm-sglang-infra

A multi-tenant LLM inference platform for GPU clusters scheduled by **Slurm**,
with Kubernetes packaging and cloud overflow.

A Go gateway sits in front of vLLM (and, where it runs, SGLang) replicas. It
speaks the OpenAI API, authenticates tenants, enforces their token budgets,
and meters usage. For every request it picks the replica — or the
data-parallel rank of a MoE deployment — where the request will finish
fastest, using live engine telemetry and the KV cache each replica holds.
When the pool is full, interactive traffic goes first, and tenants that
allow it overflow to an external provider.

> **Status.** The gateway and platform features are built and tested end to
> end against simulated engines; results are in [docs/results](docs/results).
> Routing, KV tiering, MoE, and cold start are measured on Explorer H200s;
> prefill/decode disaggregation is in progress. See
> [docs/roadmap.md](docs/roadmap.md); every failure along the way is in
> [docs/failures.md](docs/failures.md).

| Result (simulated engines) | Baseline | This project |
|---|---|---|
| Prompt-cache hit rate, cache-pressured workload | 39% (least-load) | **80%** (prefix-aware) |
| Median TTFT, same workload | 660 ms | **53 ms** |
| Hit rate with two gateways in front of one pool | 70% (approximate index) | **80%** (precise, KV events) |
| Interactive TTFT p99 under a batch flood | 4768 ms (no flow control) | **293 ms** |

| Measured on H200s (Explorer) | Result |
|---|---|
| MoE DP+EP (Qwen3-30B-A3B), vLLM's DP balancer → our rank-aware routing | hits 81% → **89%**, 45.5 → **52.5 req/s**, p99 TTFT 1064 → **317 ms** ([MoE](docs/results/e-moe-h200.md)) |
| Cache hits / throughput, 4 replicas, round-robin → prefix-aware | 47% → **89%** / 61.7 → **76.7 req/s**; median TTFT 45 → 21 ms ([routing](docs/results/m3-routing-h200.md)) |
| vLLM 0.26 (cu129 build) on driver 570 / CUDA 12.8 | works: compile and CUDA graphs included ([cold start](docs/results/d-coldstart-h200.md)) |
| Qwen3-8B replica start: cold / warm compile cache | 80 s / 65 s (weights load in 5 s) |
| Wake a sleeping replica | 0.85 s |
| Prompt-cache hits with 6 GiB of GPU KV, without / with a 48 GiB CPU tier | 60% / **99.6%**; throughput +27–53% ([KV tiering](docs/results/a-kvtier-h200.md)) |

## Why this exists

- **LLM serving stacks assume Kubernetes.** [llm-d](https://github.com/llm-d/llm-d),
  [AIBrix](https://github.com/vllm-project/aibrix), and KServe are
  Kubernetes-native. A large share of GPUs — university HPC, research labs,
  training clusters — are scheduled by Slurm instead. This project brings
  LLM-aware routing, KV-cache-aware scheduling, and autoscaling to that
  setting.
- **llm-d needs an upstream tenant gateway.** llm-d's flow control queues
  requests by a tenant header (`x-llm-d-inference-fairness-id`) and expects a
  trusted gateway in front of it to authenticate API keys and inject that
  header. It has no quotas or billing. This gateway fills that role, so it
  also works in front of llm-d on Kubernetes.

## Architecture

```
 clients (tenants with API keys) ── OpenAI API ──┐
                                                 ▼
 ┌──────────────────────────── gateway (Go) ───────────────────────────────┐
 │ auth → token quota (Redis) → flow control → tokenize → schedule → proxy │
 │        reserve, settle       priority bands   /tokenize  filter → score  │
 │                              fair queuing                → pick         │
 │ usage records (PostgreSQL) · Prometheus · admin API · autoscaler        │
 └──────┬──────────────┬──────────────┬──────────────┬──────────────┬──────┘
        ▼              ▼              ▼              ▼              ▼
   vLLM replica   vLLM replica   prefill  ──KV──▶ decode       external
   (1 H200)       (1 H200)       (PD pair, NIXL)               provider
        └── /metrics, KV-cache events (ZeroMQ) ──→ monitor + precise KV index

 replicas are Slurm jobs (sbatch / scancel, drain before walltime)
 or Kubernetes pods (discovered by label, Deployment scaled)
```

The scheduler follows the filter → score → pick pipeline of llm-d's Endpoint
Picker, so each policy is a composition of small plugins. See
[docs/design.md](docs/design.md).

## Features

| Area | What | Status |
|---|---|---|
| Gateway core | OpenAI-compatible proxy with SSE streaming, health probes, engine metrics, retries before the first byte, upstream cancellation, admin API for self-registering replicas | ✅ |
| Routing | `round-robin`, `least-load`, `prefix-aware` (approximate index), `precise-prefix-aware` (KV-cache events + engine tokenization, tier-aware) | ✅ sim · ✅ H200 |
| Tenancy | API keys (stored hashed), per-tenant token budgets reserved up front and settled with actual usage (memory or Redis), usage records (JSONL or PostgreSQL), tenant header for llm-d | ✅ |
| **C** Flow control | Priority bands, token-fair queuing across tenants (VTC), batch capacity share, queue limits and timeouts | ✅ sim |
| Cloud overflow | External OpenAI-compatible providers; overflow when the pool is full; on-prem-only tenants never leave; cost metered | ✅ sim |
| **B** PD disaggregation | Gateway picks a prefill and a decode endpoint, relays `kv_transfer_params` (vLLM NixlConnector), falls back to aggregated | ✅ sim · 🚧 H200 |
| **D** Autoscaling | Scale on in-flight + queued demand with hysteresis; Slurm walltime handover; providers: Slurm, Kubernetes, local processes | ✅ tests · ✅ H200 cold start · 🚧 H200 handover |
| **E** MoE | DP+EP ranks as endpoints, rank-aware routing | ✅ H200 |
| **A** KV tiering | CPU-tier blocks from KV events weigh in routing; vLLM native offload | ✅ H200 (1 replica) |
| Kubernetes | Pod discovery by label, Deployment scaling, Helm chart, kind demo | ✅ kind |

## Quick start (no GPU)

Requirements: Go 1.26+, Docker.

```bash
make test   # unit and end-to-end tests against fake engines
make build  # bin/gateway, bin/loadgen, bin/fakeengine, bin/kvrelay, bin/kvdump
make up     # gateway + 4 simulated vLLM replicas + kvrelay + Prometheus + Grafana

curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen3-8b-sim", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 20}'
curl -s localhost:8080/metrics | grep ^vsi_
make down
```

The response header `X-VSI-Backend` names the replica that served the
request. Prometheus is on `:9090` and Grafana on `:3000`.

The full platform, with tenants, Redis quotas, PostgreSQL metering, flow
control, and a fake external provider:

```bash
docker compose -f deploy/local/docker-compose.yaml -f deploy/local/docker-compose.platform.yaml up -d --build
bin/loadgen -api-key etl-demo-key -concurrency 100 -requests 300   # batch tenant: overflows when the pool is full
bin/loadgen -api-key bank-demo-key -concurrency 100 -requests 300  # on-prem only: queues instead
docker compose -f deploy/local/docker-compose.yaml -f deploy/local/docker-compose.platform.yaml exec postgres \
  psql -U postgres -d vsi -c 'SELECT tenant, external, count(*), sum(cost_usd) FROM usage_records GROUP BY 1, 2'
```

Reproduce the results (each script prints a summary):

```bash
deploy/local/experiments/flow-control.sh     # interactive latency under a batch flood
deploy/local/experiments/precise-prefix.sh   # approximate vs precise index, two gateways
deploy/local/experiments/autoscale-demo.sh   # replicas follow demand up and down
```

## Running on Slurm

See [cluster/README.md](cluster/README.md): build the vLLM environment once,
download a model, check the GPU stack, then start the gateway and replicas.

## Configuration

```yaml
listen: ":8080"
scheduler:
  policy: precise-prefix-aware   # round-robin | least-load | prefix-aware | precise-prefix-aware
  weights: {precise-prefix: 1.0, queue: 2.0, kv-usage: 0.5}
  queue_saturation: 32           # load at which the queue score reaches 0
  cpu_weight: 0.5                # a CPU-tier block's worth relative to a GPU block
backends:                        # static; replicas can also self-register
  - {name: r1, url: "http://d4052:8000", labels: {kv_events: "tcp://d4052:5557"}}
  - {name: p1, url: "http://d4053:8000", labels: {role: prefill}}
  - {name: cloud, kind: external, url: "https://provider.example/v1", api_key_env: PROVIDER_KEY,
     models: {qwen3-8b: provider-qwen}, price_input: 0.3, price_output: 1.2, max_inflight: 64}
tenancy:
  tenants:
    - {name: support, api_key_hashes: ["sha256:…"], priority: interactive, tokens_per_minute: 200000}
    - {name: etl, api_key_hashes: ["sha256:…"], priority: batch}
    - {name: bank, api_key_hashes: ["sha256:…"], on_prem_only: true}
quota: {store: redis, redis_url: "redis://redis:6379/0", default_max_tokens: 512}
usage: {sink: postgres, postgres_dsn: "postgres://…"}
flow: {enabled: true, max_inflight_per_endpoint: 32, batch_share: 0.75}
pd: {enabled: false}
autoscale:
  enabled: false
  provider: slurm                # slurm | kubernetes | process
  min: 1
  max: 4
  target_inflight_per_replica: 16
  slurm: {script: /scratch/…/vllm_replica.sbatch, args: [Qwen3-8B]}
admin: {token: ""}               # or VSI_ADMIN_TOKEN; empty disables /admin
```

`echo my-key | bin/gateway -hash-key` prints the hash for a tenant's API key.

## Repository layout

```
cmd/gateway         the gateway
cmd/loadgen         shared-prefix workload generator (TTFT percentiles, cache hits, spread)
cmd/fakeengine      fake OpenAI-compatible engine for tests and demos
cmd/kvrelay         fans KV-cache events from dialing publishers out to gateways
cmd/kvdump          prints a replica's KV-cache events
internal/backend    endpoint registry, health probes, metrics scraping, engine adapters
internal/scheduler  filter / score / pick pipeline and plugins
internal/prefix     approximate prefix index (chained block hashes, per-endpoint LRU)
internal/kvevents   ZeroMQ subscriber and decoder for vLLM / simulator KV events
internal/kvindex    precise KV index keyed by chained token-block hashes, per tier
internal/kvsync     subscriptions that follow the registry
internal/tokenize   prompt token IDs from an engine's /tokenize
internal/tenancy    API keys and tenants
internal/quota      token buckets in memory or Redis
internal/usage      usage records to JSONL or PostgreSQL
internal/flow       priority bands and fair queuing
internal/autoscale  autoscaler and its Slurm, Kubernetes, and process providers
internal/k8s        pod discovery and Deployment scaling over the Kubernetes API
internal/gateway    HTTP handlers, request pipeline, SSE relay, PD orchestration, admin API
cluster/            Slurm jobs for Explorer
deploy/local/       docker compose stacks and experiment scripts
docs/               design, roadmap, results
```
