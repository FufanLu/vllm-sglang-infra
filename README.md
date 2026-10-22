# vllm-sglang-infra

A multi-tenant LLM inference platform for GPU clusters scheduled by **Slurm**,
with Kubernetes packaging and cloud overflow.

A Go gateway sits in front of vLLM (and, where it runs, SGLang) replicas. It
speaks the OpenAI API, and for every request it picks the replica — or the
data-parallel rank of a MoE deployment — where the request will finish
fastest, using live engine telemetry and KV-cache locality. The platform is
built and measured on real H200s; everything also runs on a laptop against
simulated engines.

> **Status:** M1 (gateway core) done; M2 (bring-up on H200) in progress.
> See [docs/roadmap.md](docs/roadmap.md).

## Why this exists

- **LLM serving stacks assume Kubernetes.** [llm-d](https://github.com/llm-d/llm-d),
  [AIBrix](https://github.com/vllm-project/aibrix), and KServe are
  Kubernetes-native. A large share of GPUs — university HPC, research labs,
  training clusters — are scheduled by Slurm instead. This project brings
  LLM-aware routing, KV-cache management, and autoscaling to that setting.
- **llm-d needs an upstream tenant gateway.** llm-d's flow control queues
  requests by a tenant header (`x-llm-d-inference-fairness-id`) and expects a
  trusted gateway in front of it to authenticate API keys and inject that
  header. It has no notion of quotas or billing. This gateway fills that role,
  so it also works in front of llm-d on Kubernetes.

## Architecture

```
 clients (tenants with API keys) ── OpenAI API ──┐
                                                 ▼
 ┌──────────────────────── gateway (Go) ─────────────────────────┐
 │ auth → token quotas → flow control → scheduler → proxy → meter │
 │                                filter → score → pick           │
 └──────┬───────────────┬───────────────┬───────────────┬────────┘
        ▼               ▼               ▼               ▼
   vLLM replica    vLLM replica    vLLM replica    vLLM replica    one Slurm job each,
   (H200)          (H200)          (H200)          (H200)          1 GPU per job
        └──── /metrics, KV-cache events ──→ monitor ──→ scheduler

 Slurm controller: sbatch / scancel replicas, drain before walltime
 Observability: Prometheus + Grafana; usage records in PostgreSQL
```

The scheduler follows the filter → score → pick pipeline of llm-d's Endpoint
Picker: filters drop ineligible endpoints, scorers rate the rest in [0, 1],
and a picker chooses from the weighted sum. Policies are compositions of
small plugins. See [docs/design.md](docs/design.md).

## Scope

| Layer | What | Status |
|---|---|---|
| Gateway core | OpenAI-compatible proxy with streaming, health checks, engine metrics scraping, retries and failover, round-robin and least-load scheduling, Prometheus metrics, admin API for self-registering replicas | ✅ M1 |
| Bring-up | vLLM on Explorer H200s, replicas as Slurm jobs, baseline benchmarks | 🚧 M2 |
| Prefix-aware routing | Approximate (gateway-side prefix index) and precise (vLLM KV events) | M3 |
| Tenancy | API keys, token quotas (reserve, then settle), usage metering, llm-d tenant header | M4 |
| **D** Cold start | Weight staging strategies, vLLM sleep/wake, drain before walltime | M5 |
| **E** MoE | DP+EP on 2 GPUs, DP-rank-aware routing vs vLLM's internal balancer | M6 |
| **A** KV tiering | CPU offload and cross-replica KV sharing (LMCache), tier-aware scoring | M7 |
| **C** Flow control | Priority bands, token-fair queuing, SLO-aware admission | M8 |
| **B** PD disaggregation | 1 prefill + 1 decode GPU, KV transfer over NIXL | M9 |
| Kubernetes | Helm chart, kind demo, in front of llm-d | M10 |
| Cloud overflow | External OpenAI-compatible providers, data residency per tenant | M11 |

Out of scope: changing the inference engines, and wide expert parallelism
across many nodes (the cluster allows at most two GPUs per job).

## Quick start (no GPU)

Requirements: Go 1.26+, Docker.

```bash
make test   # unit and end-to-end tests against fake engines
make up     # gateway + 4 simulated vLLM replicas + Prometheus + Grafana

curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen3-8b-sim",
  "messages": [{"role": "user", "content": "hello"}],
  "max_tokens": 20
}'
curl -s localhost:8080/metrics | grep ^vsi_
curl -s -H 'Authorization: Bearer local-dev-token' localhost:8080/admin/backends

make down
```

The response header `X-VSI-Backend` names the replica that served the
request. Prometheus is on `:9090` and Grafana on `:3000`.

## Running on Slurm

See [cluster/README.md](cluster/README.md): build the vLLM environment once,
download a model, check the GPU stack, then start the gateway and replicas.

## Gateway configuration

```yaml
listen: ":8080"
scheduler:
  policy: least-load          # round-robin | least-load
  weights: {queue: 1.0, kv-usage: 0.5}
backends:                     # optional; replicas can also self-register
  - {name: replica-a, url: "http://d4052:8000", engine: vllm}
health: {interval: 2s, timeout: 1s, failure_threshold: 2}
scrape: {interval: 500ms, timeout: 1s}
proxy: {max_retries: 2, max_body_bytes: 16777216}
admin: {token: ""}            # or VSI_ADMIN_TOKEN; empty disables /admin
```

## Repository layout

```
cmd/gateway       gateway binary
cmd/fakeengine    fake OpenAI-compatible engine for tests and demos
internal/backend  endpoint registry, health probes, metrics scraping, engine adapters
internal/scheduler filter / score / pick pipeline and plugins
internal/gateway  HTTP handlers, proxying, SSE relay, admin API
internal/metrics  Prometheus series
cluster/          Slurm jobs for Explorer (env, models, GPU check, gateway, replicas)
deploy/local/     docker compose stack with llm-d-inference-sim
docs/             design and roadmap
```
