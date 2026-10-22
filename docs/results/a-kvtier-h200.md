# A — KV-cache CPU tier on an H200 (measured)

**Real hardware.** One NVIDIA H200 on Explorer, driver 570.86.15. vLLM
0.26.0 (cu129 build), Qwen3-8B (bf16). Job `cluster/kvtier_bench.sbatch`,
2026-10-22, run twice: `11295986` (node d4052) and `11310160` (d4053, after the
index fix, failure #22).

## Setup

- The GPU KV cache is held to 6 GiB (`--kv-cache-memory-bytes 6G`):
  43,680 tokens.
- The workload's shared prefixes total about 100k tokens: 48 prompt groups
  × 1,500-word prefixes, 600 requests, 16 concurrent, 32 output tokens each.
- Two passes per mode. The first fills the caches. The second is measured.
- **gpu-only:** vLLM prefix caching, nothing else.
- **cpu-offload:** plus vLLM's native offloading connector with 48 GiB of
  host memory (`--kv-offloading-size 48 --kv-offloading-backend native`).
  Evicted blocks are kept in CPU memory and loaded back on a hit.
- The gateway runs `precise-prefix-aware` and subscribes to the engine's KV
  events, so its index shows which blocks sit in which tier.

## Results (second pass)

| | gpu-only | cpu-offload |
|---|---|---|
| Prompt tokens served from cache | 59.6% / 59.7% | **99.6%** / 99.6% |
| Throughput (req/s) | 37.5 / 34.8 | **47.8** / 53.2 |
| Output tokens/s | 1199 / 1114 | 1530 / 1703 |
| TTFT p50 (ms) | 71 / 110 | 110 / 69 |
| TTFT p99 (ms) | 437 / 422 | **187** / 375 |
| TPOT p50 (ms) | 10 / 10 | 7 / 6 |
| E2E p99 (ms) | 624 / 843 | **379** / 603 |

Each cell is run `11310160` / run `11295986`.

Gateway index after the cpu-offload pass of `11310160`:

```
vsi_kvindex_blocks{backend="r0",tier="gpu"} 2723
vsi_kvindex_blocks{backend="r0",tier="cpu"} 5712
vsi_kvindex_orphan_blocks{backend="r0"} 0
```

5,712 blocks of 16 tokens is about 91k tokens, which is the workload's prefix
set. vLLM reported 40.9 GB loaded back from CPU during the run. In the first
run the CPU line read 0 because the CPU events carried no token IDs (see
[failures #22](../failures.md)).

## Reading

- **The CPU tier turns a cache that thrashes into one that holds the whole
  working set.** With 6 GiB the GPU keeps under half the prefixes, so 40% of
  prompt tokens are recomputed. With the CPU tier nearly every prefix is a
  hit, and throughput rises by 27% and 53% in the two runs.
- **The tail gains more than the median.** p99 TTFT and E2E fall in both
  runs. The median TTFT is mixed: a CPU hit still has to copy blocks back to
  the GPU, which costs a little at p50, while a GPU miss costs a full
  prefill.
- **Run-to-run spread is large.** The two runs were on different nodes, and
  the medians moved by up to 40 ms. Read the hit rate and throughput as the
  result. Individual latency medians are not stable enough to quote alone.
- **The gateway sees the tier.** The precise scorer counts a CPU-only hit
  at `cpu_weight` (0.5). One replica cannot show the routing effect. With
  several replicas, a request goes to the replica that holds its prefix on
  GPU, then to one that holds it on CPU, then to any other.

## Reproduce

```
sbatch cluster/kvtier_bench.sbatch            # GPU_KV, CPU_KV_GIB, PROMPT_GROUPS, PREFIX_WORDS
```
