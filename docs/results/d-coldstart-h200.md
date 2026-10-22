# D — Cold start of a vLLM replica on an H200 (measured)

**Real hardware.** One NVIDIA H200 (141 GB) on Explorer node d4052,
driver 570.86.15. vLLM 0.26.0 (cu129 build), Qwen3-8B (bf16, 15.3 GB).
Job `cluster/coldstart.sbatch`, 2026-10-22.

## Where a new replica's time goes

| Step | Time | Notes |
|---|---|---|
| Unpack the packed Python env to node-local disk | 33 s | Once per node; later jobs on the node reuse it |
| Copy weights from `/scratch` (NFS) to node-local NVMe | 24 s | 15.3 GB. Raw NFS read with O_DIRECT measured 952 MB/s at the time; it is far slower when the filesystem is busy |
| vLLM start, cold compile cache | **80 s** | Breakdown below |
| vLLM start, warm compile cache | 65 s | torch.compile loads the compiled graph instead of compiling: 18 s → 4 s |
| vLLM start, weights straight from NFS | 70 s | Weights in the page cache after the copy; not a cold-NFS number |
| **Wake from sleep mode** | **0.85 s** | Weights back from CPU memory; first request after wake 51 ms |

Cold start, phase by phase (from the vLLM log):

| Phase | Time |
|---|---|
| Process start, imports, API server, tokenizer (before the engine initializes) | ~25 s |
| Load weights (2.7 s reading) and set up the model | 7 s |
| torch.compile (Dynamo + Inductor) | 18 s |
| Profiling run, KV-cache allocation | 3 s |
| Kernel warmup and autotuning before graph capture | 8 s |
| CUDA graph capture (51 piecewise + 51 full sizes) | 3 s |
| Remaining engine warmup | 13 s |
| API server up | 1 s |

Sleep mode (`--enable-sleep-mode`, level 1):

- Going to sleep took 7.2 s: weights offloaded to CPU, KV cache discarded.
- While asleep the process held 2.3 GB of GPU memory.
- Waking took 0.85 s.

## Reading

- **Loading weights is the small part.** It is 5 s of 80. On this node the
  rest is interpreter start-up, compilation, and warmup. A persistent
  compile cache (`VLLM_CACHE_ROOT` on node-local or shared disk) saves 15 s
  on every restart.
- **A brand-new replica on a fresh node costs ~2.3 min** (33 + 24 + 80 s)
  before the Slurm queue wait, which on a busy shared cluster dominates
  everything here.
- **Waking a sleeping replica is ~90× faster than starting one.** On
  Slurm, sleep does not return the GPU to the scheduler: the job still
  holds its allocation. So sleep is the tool for swapping models on GPUs
  the pool already owns, or for parking a replica during a lull, not for
  giving capacity back.
- For the autoscaler this means:
  - Successors are launched before a walltime because a start costs minutes.
  - Scale-down cancels queued jobs before draining warm ones.
