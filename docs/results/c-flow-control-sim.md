# C — Flow control on simulated engines

**These numbers come from simulated engines, not GPUs.** They show the
mechanism. The latencies follow from the simulator's model, so read them as
ratios.

## Question

When a batch tenant floods the pool, what happens to interactive requests,
and what does gateway-side flow control buy?

## Setup

- Four llm-d-inference-sim v0.9.0 replicas. Each runs at most 8 requests at
  once (`--max-num-seqs 8`) and slows down under load
  (`--time-factor-under-load 2`): a full batch takes twice as long per token.
  Extra requests wait in the engine's FIFO queue.
- Two tenants. `batch` (priority batch) keeps 64 long requests in flight:
  ~300-word prompt, 64 output tokens. `chat` (priority interactive) keeps 4
  short requests in flight: ~100-word prompt, 16 output tokens, 200 requests
  measured.
- Gateway policy `least-load` in both runs; only flow control changes.
  - **off:** requests go straight to the engines.
  - **on:** capacity is 8 per replica = 32 slots. Batch may hold at most 75%
    (24 slots). Interactive requests are dispatched first, and requests wait
    in the gateway rather than in engine queues.
- Script: `deploy/local/experiments/flow-control.sh`.

## Results (interactive tenant, under the batch flood)

| Flow control | TTFT p50 | TTFT p90 | TTFT p99 | E2E p50 | Throughput |
|---|---|---|---|---|---|
| off | 3079 ms | 3851 ms | 4768 ms | 3705 ms | 1.0 req/s |
| on | 269 ms | 273 ms | 293 ms | 837 ms | 5.2 req/s |

## Reading

- **Without flow control, interactive requests queue behind batch work.**
  64 batch requests on 32 engine slots leave 8 waiting per replica, and an
  interactive request waits behind them: about three seconds before its
  first token.
- **With flow control, interactive TTFT drops 11× at the median and 16× at
  p99.** Two things combine:
  - Batch is capped at 24 slots, so 8 are always free for interactive work.
  - The gateway never overfills the engines, so nothing waits in an engine
    queue it cannot reorder.
- **What it costs.** Batch work queues in the gateway instead of the engines.
  Its throughput is bounded by its 75% share, which is the intended trade.
  The share is a config knob (`flow.batch_share`).

llm-d 1.0 ships the same idea (priority bands, fair queuing, no-regret
dispatch) inside its router. This is the Slurm-side implementation, with
tenants identified by API key instead of a trusted header.
