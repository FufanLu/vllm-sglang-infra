# D — Walltime handover under load on H200s (measured)

**Real hardware.** Qwen3-8B replicas on Explorer H200s (nodes d4053 and
d4054), vLLM 0.26.0. The gateway runs on a CPU node with the Slurm
autoscaler, which submits and replaces the replica jobs itself. Job
`cluster/handover_bench.sbatch` `11313510`, 2026-10-22.

## Why it matters

A Slurm job has a walltime. A serving pool on Slurm has to replace each
replica before its walltime ends, without dropping requests, while the
successor waits in a shared queue for an unknown time. Kubernetes has no
such deadline. This is the Slurm-specific part of the autoscaler.

## How a handover works

1. The autoscaler (min = max = 1 here) stops counting a replica as
   capacity once its walltime is less than `handover_lead` away, and submits
   a successor while the old one keeps serving.
2. Slurm sends the old replica TERM 15 min before its walltime.
3. The old replica waits until the gateway lists another healthy replica,
   or until only 3 min are left. Then it deregisters, finishes its
   in-flight requests, and stops.

Step 3 was added after the first run (failure #29). Then, each old replica
deregistered on TERM whatever the state of its successor, and the pool was
empty twice: once for about 4 min, once for about 20 s.

## Setup

- Replicas: 40 min walltime, handover lead 35 min.
- Load: rounds of 100 chat requests, 8 concurrent, for 70 min.
- Timeline: the healthy replicas, sampled every 5 s.

## Results

| | First run `11311478` (deregister on TERM) | This run `11313510` (wait for successor) |
|---|---|---|
| Requests | 55,000 completed between two gaps | **90,500, 0 failed** |
| Time with no healthy replica | about 4 min + 20 s | **0 of 545 samples** |
| Handovers | 2, both with a gap | 4, none with a gap |

The four handovers of the second run (Eastern time):

| Old → new | Successor queued | Successor start → registered | Successor healthy | Old deregistered | Old drain |
|---|---|---|---|---|---|
| 11313526 → 11313781 | 8 min | 5.5 min (env unpack 172 s, weights 77 s, vLLM 80 s) | 22:16:59 | 22:22:37 | 5 s |
| 11313781 → 11313947 | 1 min | 48 s | 22:18:39 | 22:36:03 | 0 s |
| 11313947 → 11314063 | 16 min | 1.7 min | 22:43:26 | **22:43:26** | 5 s |
| 11314063 → 11314146 | 23 min | 2 min | 22:52:17 | 23:06:20 | 5 s |

In the third handover the old replica got TERM while its successor was
still starting. It kept serving for another 35 s, and it deregistered in
the same second the successor registered.

## Reading

- **Queue wait dominates a handover on a shared cluster.** Successors waited
  from 1 to 23 minutes in the queue. Starting took 41 s to 2 min when the node
  already held the env, and 5.5 min when it did not (env unpack 172 s from
  a busy `/scratch`). No
  fixed lead time covers that on its own. The old replica has to wait for
  its successor, and it has to have enough walltime left to do so.
- **Drains are short.** With 64-token answers, in-flight requests finished
  within 5 s of deregistration.
- **The lead has to stay well below the walltime.** At 40 min walltime and
  35 min lead, a replica counted as expiring 5 min after it started,
  sometimes before it was serving. So the autoscaler kept submitting
  successors: 5 replicas for 70 min of load. Production walltimes are hours
  (8 h on this partition), where a 30 min lead costs about 6% of a
  replica's time. The bench now defaults to 60 min and 30 min.
- **slurmctld can fail an `sbatch` and still queue the job, 22 s later.**
  Once the autoscaler adopted the late job, a second failed `sbatch` also
  queued a job, so one replica was surplus (11314146 above). After a failed
  `sbatch`, the provider now submits nothing for 3 min and adopts the job if
  it appears (failure #30).

## Reproduce

```
sbatch cluster/handover_bench.sbatch    # REPLICA_WALLTIME, HANDOVER_LEAD, DURATION
```
