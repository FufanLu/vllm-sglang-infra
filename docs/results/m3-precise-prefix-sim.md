# M3 — Precise prefix routing with two gateways (simulated engines)

**These numbers come from simulated engines, not GPUs.**

## Question

The approximate prefix index learns only from its own gateway's routing
decisions. In a highly available deployment several gateways share the
replicas behind a load balancer. Does each gateway's private guess still
work, and does an index fed by the engines' own KV-cache events fix it?

## Setup

- Four llm-d-inference-sim v0.9.0 replicas with KV-cache simulation. Same
  latency model and capacity as [the M3 run](m3-prefix-routing-sim.md): 1 ms
  per uncached prompt token, 32 768 tokens of cache per replica.
- The replicas publish `BlockStored` / `BlockRemoved` events over ZeroMQ.
  The simulator dials out, so `kvrelay` fans the events in and out, and both
  gateways subscribe to it. vLLM binds its own publisher and needs no relay.
- Two gateways run the same policy, and requests alternate between them.
- Workload as in M3: 96 groups of ~600-token system prompts, 600 requests,
  16 in flight, the same seed.
- `precise-prefix-aware`:
  - Tokenizes each request through the engine's `/tokenize`.
  - Hashes full 16-token blocks with a chain hash.
  - Matches the hashes against the event-fed index of each replica.
  - A small recent-picks index covers the gap before events arrive.
- Script: `deploy/local/experiments/precise-prefix.sh`.

## Results

| Policy | Gateways | Prompt-cache hit | TTFT p50 / p90 / p99 (ms) | Throughput (req/s) | Output tok/s |
|---|---|---|---|---|---|
| prefix-aware (approximate), from M3 | 1 | 80.3% | 53 / 662 / 680 | 32.7 | 524 |
| prefix-aware (approximate) | 2 | 70.1% | 53 / 662 / 687 | 29.1 | 465 |
| precise-prefix-aware | 2 | 80.3% | 54 / 662 / 690 | 32.7 | 523 |

## Reading

- **Two approximate indexes disagree.** Each gateway sends a group's first
  request wherever it sees fit. The two often pick different replicas, so a
  prefix ends up cached twice, and each replica's cache holds more distinct
  prefixes than it would otherwise. The hit rate falls from 80% to 70%.
- **The precise index restores the single-gateway result.** Both gateways
  read the same truth from the engines, so they agree on where a prefix
  lives. The hit rate returns to the ~81% ceiling and throughput rises 12%.
- **What it costs.** One `/tokenize` call per request (~1 ms against the
  simulator), and an event stream per replica.

## Bugs this found

1. **The simulator never listened on its event port.** It dials out, so
   subscribers must bind. The fix is `kvrelay`, plus attributing events by
   topic.
2. **v0.9 topics omit the port.** It publishes `kv@<pod-ip>@<model>` rather
   than the documented `kv@<ip>:<port>@<model>`. Events are now matched by
   host when one endpoint lives on that host.

Both bugs were caught by the `vsi_kvindex_blocks` metric reading zero, and
by `cmd/kvdump`.
