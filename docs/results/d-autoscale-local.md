# D — Autoscaling on a laptop

**Fake engines as local processes, not GPUs.** The demo shows the control
loop. Real cold starts on H200s (minutes, not milliseconds) are the subject
of the M5 measurements.

Setup (`deploy/local/experiments/autoscale-demo.sh`):

- Gateway with flow control (8 slots per replica), so queued demand is
  visible.
- Autoscaler with the `process` provider: min 1, max 6, target 8 in flight
  per replica, scale down after 10 s of low demand, 2 s passes.
- Load in three phases: 4 in flight, then 40, then 4 again. Each request is
  ~0.7 s (100 ms TTFT, 32 tokens at 20 ms).

Timeline (sampled every 3 s):

```
 0s replicas=1 queued=0     phase 1: 4 in flight
10s replicas=1 queued=0
13s replicas=1 queued=32    phase 2 starts: demand 40 > capacity 8
16s replicas=5 queued=0     scaled to ceil(40/8) = 5; queue drained
...
55s replicas=5 queued=0     phase 3: demand back to 4
58s replicas=1 queued=0     after 10 s of low demand: drained to min
```

| Phase | TTFT p50 | TTFT p99 |
|---|---|---|
| 4 in flight | 102 ms | 108 ms |
| 40 in flight | 102 ms | 1618 ms (requests that queued before scale-up) |
| 4 in flight | 102 ms | 108 ms |

On Slurm the same loop submits `vllm_replica.sbatch` jobs. A replica is
counted as capacity only until 20 minutes before its walltime, so its
successor starts while it still serves.
