# B — Prefill/decode disaggregation on two H200s (measured)

**Real hardware.** Two H200s on Explorer node d4055 (NVLink), driver
570.86.15. vLLM 0.26.0 (cu129 build), Qwen3-8B (bf16), KV transfer with
NIXL 1.5 over UCX. Job `cluster/pd_bench.sbatch` `11310750`, 2026-10-22.

## Setup

- **aggregated:** two ordinary vLLM replicas, one per GPU. The gateway
  routes least-load.
- **1p1d:** one prefill instance and one decode instance, both with
  `NixlConnector`. For each request the gateway:
  1. sends it to the prefill instance with `max_tokens=1` and
     `kv_transfer_params.do_remote_decode`;
  2. passes the returned `kv_transfer_params` to the decode instance;
  3. the decode instance pulls the KV blocks from the prefill GPU and
     streams the answer.
- Workload: long prompts with medium outputs, where a prefill interrupting
  running decodes hurts the most. 400 requests, 16 concurrent, 400 prompt
  groups with 2,500-word prefixes (mostly distinct prompts), 256 output
  tokens.
- UCX transport: TCP on the loopback device plus `cuda_ipc` for the GPU
  copy. The node's own address refuses TCP, so the default set fails
  (failure #23).

## Results

| | aggregated (2 GPUs) | 1p1d (2 GPUs) |
|---|---|---|
| Throughput | 8.3 req/s, 2114 output tok/s | **8.7 req/s, 2237 output tok/s** |
| TTFT p50 / p90 / p99 | 258 / 432 / 509 ms | **112 / 179** / 932 ms |
| TPOT p50 / p90 / p99 | 6 / 7 / 7 ms | **6 / 6 / 6 ms** |
| E2E p50 / p99 | 1937 / 2303 ms | **1791** / 2605 ms |
| Errors | 0 | 0 |

The engines' own numbers during the 1p1d run:

- **Prefill:** mean prefill phase 84 ms per request, measured by the
  gateway (`vsi_pd_prefill_seconds`).
- **Decode:** local prompt throughput 0 tokens/s. Every prompt token came
  from the prefill instance ("external prefix cache hit rate" 100%).
- **NIXL:** a transfer took 3.9–4.9 ms on average (p90 about 7 ms) for
  137–212 MB, which is 35–43 GB/s over NVLink.

The client-side "prompt cache 100%" for 1p1d counts transferred KV as
cached, because vLLM reports it that way. It is not a cache-hit rate
comparable with the aggregated run's 19.8%.

## Reading

- **Decode no longer waits for prefills.** In the aggregated setup each
  GPU interleaves new 3k-token prefills with the decode steps of running
  requests. With a dedicated decode GPU, TPOT is flat (p99 = p50 = 6 ms).
  The median TTFT falls by more than half, because a prefill runs alone on
  its GPU instead of sharing steps with decodes.
- **The KV hand-off is cheap on one node.** About 5 ms against an 84 ms
  prefill. Across nodes it would go over the network, and the gap would
  shrink.
- **Prefill capacity is now the tail.** One prefill GPU serves every
  request, so bursts queue there: TTFT p99 rose from 509 to 932 ms. The
  ratio of prefill to decode instances (xPyD) is the knob. With a
  prefill-heavy workload like this one, 2P1D or a larger prefill pool would
  be the next step.
- Throughput rose slightly (+5%) on the same two GPUs.

## Reproduce

```
sbatch cluster/pd_bench.sbatch      # REQUESTS, CONCURRENCY, PREFIX_WORDS, MAX_TOKENS
```
