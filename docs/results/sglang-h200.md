# SGLang behind the gateway on two H200s (measured)

**Real hardware.** Two SGLang 0.5.10 servers of Qwen3-8B (bf16), one H200
each, on Explorer node d4055, driver 570.86.15. Job
`cluster/sglang_bench.sbatch` `11320726`, 2026-10-23.

## Why 0.5.10

SGLang 0.5.11 and later need CUDA 13 (`cuda-python>=13`, FlashInfer cu13),
which driver 570 cannot run. 0.5.10 pins torch 2.9.1, whose PyPI build is
CUDA 12.8. The env is built by `cluster/setup_sglang_env.sbatch`.

## What the gateway does for SGLang

| | vLLM | SGLang |
|---|---|---|
| Load and KV metrics | `vllm:num_requests_running`, `…_waiting`, `vllm:kv_cache_usage_perc` | `sglang:num_running_reqs`, `num_queue_reqs`, `token_usage` |
| KV events | ZeroMQ, `--kv-events-config` | the same; radix-cache pages, one `BlockStored` per page (16 tokens with `--page-size 16`) |
| `/tokenize` | prompt or chat messages (chat template applied) | prompt only, no chat template |
| Prefix-cache reset | `/reset_prefix_cache` (dev mode) | `/flush_cache` |
| Cached tokens in usage | `--enable-prompt-tokens-details` | `--enable-cache-report` |

Because SGLang's `/tokenize` takes no messages, chat requests cannot get
tokens that match what the engine caches. The gateway therefore skips
SGLang endpoints when it tokenizes a chat request. Without tokens, the
precise policy falls back to approximate (text) prefix matching instead of
load only. Completion requests are tokenized by SGLang and matched exactly
against its KV events.

## Setup

- Each server's KV pool is capped at 64,000 tokens (`--max-total-tokens`).
- Workload: 48 prompt groups with 1,500-word shared prefixes (about 96k
  tokens in all) plus a 24-word question; 800 requests, 32 concurrent, 64
  output tokens. Prefix caches are flushed before each run.
- Chat runs use `/v1/chat/completions`. Completion runs send the same text
  as a prompt to `/v1/completions` (`loadgen -completions`).

## Results

| Workload | Policy | req/s | Cache hits | TTFT p50 | TTFT p99 | E2E p99 | Split |
|---|---|---|---|---|---|---|---|
| chat | round-robin (first run) | 34.5 | 70.4% | 25 ms | 1534 ms | 5264 ms | 400 / 400 |
| chat | least-load | 48.6 | 68.3% | 131 ms | 1237 ms | 1703 ms | 400 / 400 |
| chat | prefix-aware | 48.7 | **88.8%** | 25 ms | 1597 ms | 2050 ms | 383 / 417 |
| chat | precise-prefix-aware (approximate fallback) | 58.2 | **88.1%** | 26 ms | 508 ms | 1041 ms | 396 / 404 |
| completions | round-robin | 48.4 | 71.6% | 22 ms | 508 ms | 1123 ms | 400 / 400 |
| completions | prefix-aware | **58.6** | **90.3%** | 23 ms | 553 ms | 1047 ms | 411 / 389 |
| completions | precise-prefix-aware (SGLang tokens + KV events) | 57.2 | **90.1%** | 36 ms | 571 ms | 1052 ms | 385 / 415 |

- All 5,600 requests completed without an error.
- The gateway's KV index held 3,958–3,986 pages per server after each run,
  that is 63.3k–63.8k tokens of the 64,000-token pool, with no orphaned
  pages. It tracks SGLang's radix cache as closely as vLLM's prefix cache.
- The precise completion run tokenized all 800 requests through SGLang
  (`vsi_tokenize_seconds_count{ok="true"} 800`). The precise chat run made
  no tokenize call: SGLang endpoints are skipped for chat.
- SGLang started in about 43 s on a node that already held the env and
  weights.

## Reading

- **Prefix routing works the same on SGLang.** Cache hits rise from about
  70% to about 90% for both chat and completions. With the prefixes needing
  96k tokens and each server holding 64k, the load-only policies send every
  prefix to both servers, and about a third of each prompt is recomputed.
- **On completions, where the comparison is cleanest, throughput rises 21%**
  (48.4 → 58.6 req/s).
- **Precise and approximate tie again** with a single gateway, as on vLLM.
- **Run-to-run spread is large here.** The two chat runs that use the same
  approximate scorer (prefix-aware, and precise with its fallback) differ
  by 20% in throughput and 3× in p99 TTFT. Chat round-robin ran first after
  the servers came up, and its tail (TPOT p99 59 ms, E2E p99 5.3 s) is out
  of line with every later run. Read the cache-hit column as the result.
  The latency columns would need repeated runs.

## Reproduce

```
sbatch cluster/setup_sglang_env.sbatch     # once, short partition
sbatch cluster/sglang_bench.sbatch
```
