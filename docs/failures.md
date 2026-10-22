# Failure log

Every failure met while building and measuring this project: what broke,
why, what changed, and where the fix landed. Times are US Pacific, 2026.
New failures are appended at the end of the table and of the details.

Status: **fixed** (fix verified by a later run), **fix pending** (fix in,
not yet re-run), **open** (cause not established), **worked around**
(environment problem outside the project).

| # | When | Where | Symptom | Root cause | Fix | Commit | Status |
|---|---|---|---|---|---|---|---|
| 1 | 10-21 19:30 | Local, M3 run on simulators | least-load sent 347 of 600 requests to one replica | Load mixed stale scraped counts with the gateway's in-flight count; tiny KV-usage differences broke ties | Load = in-flight + external engine load; KV usage scores only above 70% | `8157369` | fixed |
| 2 | 10-21 19:20 | Local, `push.sh` to Explorer | `._gateway`-style files copied to the cluster | macOS tar adds AppleDouble files and xattrs | `COPYFILE_DISABLE=1 --no-mac-metadata --no-xattrs` | `aa01cc1` | fixed |
| 3 | 10-21 20:05 | Local, platform demo | PostgreSQL exited: `No space left on device` | Docker VM disk full (images and 8.9 GB of build cache) | Pruned the build cache | — | worked around |
| 4 | 10-21 20:05 | Local, platform demo | Gateway exited: cannot resolve `postgres` | It connected once at startup, before the database was up | Retry Redis and PostgreSQL for 30 s at startup | `2911f3d` | fixed |
| 5 | 10-21 20:10 | Local, image build | Build failed: TLS timeout fetching `docker/dockerfile:1` | The `# syntax=` line pulls a frontend from Docker Hub; its cache was pruned in #3 and Docker Hub is unreachable without a proxy here | Removed the `# syntax=` line | `2911f3d` | fixed |
| 6 | 10-21 20:15 | Local, platform demo | Gateway ran `least-load` instead of the configured `prefix-aware` | Compose passed `VSI_SCHEDULER_POLICY` with a default value, which overrides the file | Pass it only when set (`${VSI_SCHEDULER_POLICY:-}`) | `2911f3d` | fixed |
| 7 | 10-21 20:36 | Local, `flow-control.sh` | Script stopped after the first of two modes | Not established; the second mode ran fine on its own | `MODES` variable to run modes separately; long runs in the background | `2911f3d` | open |
| 8 | 10-21 20:40 | Local, precise routing | Precise index stayed empty: no KV events | llm-d-inference-sim v0.9.0 dials out to publish; it never listened on 5557 | `cmd/kvrelay` (publishers connect in, gateways subscribe); `cmd/kvdump` to inspect | `2911f3d` | fixed |
| 9 | 10-21 20:50 | Local, precise routing | Events reached the relay but were dropped | v0.9.0 topics are `kv@<host>@<model>`, without the port the docs show | Attribute by host when one endpoint lives on it | `2911f3d` | fixed |
| 10 | 10-21 20:35 | Local, unit test | Byte-string block hashes decoded as text | msgpack decoder surfaces `bin` as `string` in loose mode | Hex-encode both forms | `2911f3d` | fixed |
| 11 | 10-21 23:00 | Local, SGLang check | Simulator rejected `--engine sglang`; v1.0.0 accepted it but served empty `/metrics` | Not in v0.9.0; incomplete in v1.0.0 | Checked metric names against SGLang 0.5.22 source instead | `4793674` | worked around |
| 12 | 10-21 23:05 | Local, Grafana check | Image pulls failed: `auth.docker.io` EOF | Docker Hub unreachable from this network without a proxy | Image tags overridable (`PROMETHEUS_TAG`, `GRAFANA_TAG`); reuse local images | `b06e00c` | worked around |
| 13 | 10-21 23:10 | Local, Grafana check | Cache-hit panel empty | The simulator reports `prompt_tokens_detail` (singular); the gateway read only vLLM's `prompt_tokens_details` | Read both spellings | `b06e00c` | fixed |
| 14 | 10-21 23:32 | Explorer, ssh | Job watchers stopped | ssh master connection dropped; new logins need password and Duo, so the session does not reconnect itself | User ran `ssh explorer true` | — | worked around |
| 15 | 10-22 00:05 | Explorer, H200 check `11293567` | vLLM exited at the end of startup: `No such file or directory: 'ninja'` | FlashInfer compiles its sampler at startup; the venv's `bin` (ninja) was not on PATH because the venv is used without activation | Activate the env, load `cuda/12.8.0` for runtime builds, use vLLM's native sampler | `0d2764f` | fixed (`11295801` passed) |
| 16 | 10-22 00:23 | Explorer, routing bench `11295951` | Bench exited after 14 s | With no healthy backend yet, `grep` matched nothing and `pipefail` ended the job | `|| true` inside the poll | `81b9788` | fixed (`11295987` kept polling) |
| 17 | 10-22 00:25 | Explorer, review after #16 | Workloads would have used the user's group id as the number of prompt groups | `GROUPS` is a bash builtin, so `${GROUPS:-96}` never took the default | Renamed to `PROMPT_GROUPS`; resubmitted MoE, PD, KV-tier | `81b9788` | fix pending |
| 18 | 10-22 01:20 | Explorer, MoE `11295984` | Both DP ranks segfaulted in `ncclNetPluginInit` | NCCL loaded a network plugin that crashed; source not identified (not in the system lib paths or the CUDA module) | `NCCL_NET_PLUGIN=none` for single-node jobs; log `NCCL_*`/`UCX_*` | `3383e3a` | did not fix it (see #21) |
| 19 | 10-22 01:40 | Explorer, PD `11295985` | Aggregated baseline finished; both 1P1D instances exited: `NIXL is not available` | vLLM imports `nixl._api`, provided by the `nixl` meta package; only `nixl-cu12` (module `nixl_cu12`) was installed | Meta package unpacked into an overlay on `PYTHONPATH` (no 11 GB env rebuild); `setup_env` installs it with `--no-deps` (its deps pull CUDA 13); PD job imports NIXL first and fails fast | `c399aee` | fix pending |
| 20 | 10-22 01:51 | Explorer, `sbatch` of the PD rerun | `Batch job submission failed: Unexpected message received`; a retry succeeded (`11297370`) and a duplicate (`11297371`) appeared moments later | Transient slurmctld error: the "failed" submission had been accepted after all | Cancelled the duplicate; before retrying a failed `sbatch`, check `squeue` again after a short wait | — | worked around |
| 21 | 10-22 02:00 | Explorer, MoE rerun `11297201` | Same segfault in `ncclNetPluginInit` with `NCCL_NET_PLUGIN=none` | NCCL 2.28 starts its built-in transports (IB, socket) through the same plugin path; the crash is likely in IB init on nodes with an IB NIC (not yet confirmed) | `NCCL_IB_DISABLE=1` for single-node jobs; `NCCL_DEBUG=INFO` (INIT,NET) in the MoE job; MoE configurations now run independently so one failing start does not lose the rest | `70fb6e8` | fixed (`11297386`: both ranks past NCCL init) |
| 22 | 10-22 02:05 | Explorer, KV-tier `11295986` | Run completed (CPU offload: cache hits 59.7% → 99.6%), but the gateway's index showed 0 CPU-tier blocks and 0 orphans | vLLM 0.26's offloading connector sends placeholder CPU events (hash only, no token IDs, `block_size` 0) unless `self_describing_kv_events` is set; the index dropped them silently | Job sets `self_describing_kv_events` through `--kv-transfer-config`; the index counts token-less stores as orphans and reference-counts offload tiers; the job flags an empty CPU tier in `failed.txt` | `62daa0a` | fixed (`11310160`: 5712 CPU-tier blocks, 0 orphans) |
| 23 | 10-22 04:14 | Explorer, PD rerun `11297370` | NIXL now imported and the aggregated baseline repeated (8.1 req/s); both 1P1D engines then failed: `NIXL_ERR_BACKEND` from `createBackend('UCX')` | NIXL's UCX backend connects to itself at start-up; UCX picked TCP and `connect()` to the node's own address (`10.99.104.52`) was refused. Why the listener was unreachable is not established | The job probes the UCX backend before the engines start: default transports, then `UCX_TLS=cuda_ipc,cuda_copy,sm,self` (same node only); it logs each attempt to `ucx-probe.txt` and exits early if none works | `93e0d23` | fix pending |
| 24 | 10-22 00:30 | Explorer, routing gateway `11295942` | Gateway died with `Bus error (core dumped)` after 6 min; the four replicas started later, found no gateway, served unregistered and were cancelled by root after about 1 h 25 min (reason not established); the bench waited 4 h for backends that never registered | `push.sh` replaced the gateway binary on `/scratch` from the login node while the job ran it from there; to another NFS client the old file is gone, and its next page fault raises SIGBUS | Jobs copy the binaries to node-local `/tmp` at start and run them from there | `e689ad9` | fix pending |
| 25 | 10-22 02:43 | Explorer, MoE `11297386` | `dp-ep-ranks` did not come up within 30 min, and the job then hung until its 2 h limit | Start-up: the profiling run took 1715 s because DeepGEMM compiled FP8 kernels with nvcc (CUDA 12.8 module) on first use. Hang: `stop_all` sent TERM to engines still starting, they ignored it, and `wait` blocked | MoE job sets `VLLM_USE_DEEP_GEMM=0`, `VLLM_MOE_USE_DEEP_GEMM=0`, `--moe-backend triton`; engines start in their own process group and `stop_all` sends KILL to the group after 60 s | `7d7961d`, `e689ad9` | fixed (`11310168`: ranks ready in 64 s, profiling 1.7 s; the failed config was stopped and the job went on) |
| 26 | 10-22 16:13 | Explorer, MoE `11310168` | `dp-ep-internal` (one `vllm serve`, DP=2 behind vLLM's own balancer) failed: rank 1 hit `CUDA driver error: invalid argument` launching an Inductor Triton kernel in the profiling run; the other three configurations completed | Likely (not confirmed): the kernel came from the torch.compile cache written by the earlier configurations, where every process ran on device 0; here rank 1 runs on device 1 | Internal-balancer config runs with `VLLM_DISABLE_COMPILE_CACHE=1`; `CONFIGS` selects which configurations a job runs, to rerun this one alone | `368eb8c` | fix pending |

Mistakes in my own commands, not in the project. They cost time, so they are listed too:

| When | What happened | Lesson |
|---|---|---|
| 10-21 19:50 | zsh did not split `$F` into compose arguments | Run multi-word variables under `bash -c`, or use arrays |
| 10-21 19:50 | Ran a stale `bin/loadgen` without the new flag | Rebuild (`go build -o bin/ ./cmd/...`) before each run |
| 10-21 20:20 | A nested-quote `curl` returned 400 | Quote JSON with single quotes, outside `bash -c` |
| 10-22 00:35 | `timeout` is not available on macOS | Use the tool's own timeout |
| 10-22 15:50 | The ssh master had dropped; a background `ssh explorer` fell back to password login and failed three times with no one to answer | Every scripted ssh uses `-o BatchMode=yes` (no password fallback) after `ssh -O check`; repeated failed logins lock the login nodes |

## Details

### 1. least-load concentrated load on one replica

Seen in the first simulator comparison: least-load spread 81 / 347 / 89 / 83
across four replicas while round-robin gave 150 each. There were two causes:

- **Stale load.** Load was max(in-flight, scraped running+waiting).
  Scrapes lag up to 500 ms, so a replica that had just drained kept looking
  busy and was avoided, then got stampeded once a scrape showed it idle.
- **Ties.** KV-usage differences of a fraction of a percent broke ties,
  so one replica kept winning for a whole scrape interval.

After the fix: 145–155 per replica.
See [results/m3-prefix-routing-sim.md](results/m3-prefix-routing-sim.md).

### 7. Flow-control script stopped after one mode (open)

The run with flow control off completed and printed its summary. The run
with flow control on never started: its containers were never recreated
and no output file appeared. Run alone with `MODES=on`, it completed
normally. The likeliest explanation is that the long foreground command was
cut off by the tool running it, but this was not confirmed.

### 8, 9. KV events from llm-d-inference-sim v0.9.0

- **No events at all.** `netstat` from a sidecar sharing sim-0's network
  namespace showed only port 8000 listening. The documented "addresses with
  `*` are passive" behaviour is not in v0.9.0: it always dials.
- **Events dropped.** `kvdump` through the relay showed events arriving with
  topic `kv@sim-2@qwen3-8b-sim`, but the gateway matched on `host:port`.
- Caught because the new `vsi_kvindex_blocks` metric read zero.

### 15. First H200 check

The check confirmed the risky part works: torch cu129 on driver 570
(`cuda.is_available()` true, matmul OK), model load, torch.compile, and both
CUDA-graph captures. The engine then died in FlashInfer's JIT build of the
top-k/top-p sampler. The cold-start and KV-tier jobs queued behind it with
`afterok` were cancelled by Slurm and resubmitted. The resubmitted check
`11295801` passed: ready in 80 s, completion served.

### 16, 17. Bench died while waiting for replicas; GROUPS

`wait_backends` ran

```
curl … | grep -o '"healthy":true' | wc -l
```

inside `$(…)` under `set -euo pipefail`. With zero matches `grep` exits 1, so
the assignment failed and the job ended on its first poll.

Reviewing the scripts for the same pattern found `${GROUPS:-96}`. `GROUPS`
is bash's array of the user's group ids, so it expanded to a group id and
never to the default. The MoE, PD, and KV-tier jobs had been submitted with
that bug and sbatch keeps a copy of the script at submission, so they were
cancelled and resubmitted. Two metric snapshots piped through `grep` got
`|| true` as well.

### 18. NCCL segfault in the MoE job

Two vLLM processes, one per GPU, ran as DP ranks with expert parallelism.
Both segfaulted at NCCL communicator init, inside `ncclNetPluginInit` /
`ncclNetInit`, right after "vLLM is using nccl==2.28.9". Weight staging
(31 GB in 72 s) and env unpack had completed. Both GPUs are on one node
and talk over NVLink, so no network plugin is needed. The rerun disables
plugin loading and logs `NCCL_*` / `UCX_*` so the source can be found if it
happens again.

### 19. NIXL not importable in the PD job

The aggregated baseline, two Qwen3-8B replicas on two H200s, finished:
8.1 req/s, 2074 output tok/s, TTFT p50 255 ms, TPOT p50 6 ms. Both
disaggregated instances then failed at engine start with
`NIXL is not available` from vLLM's `nixl_utils`.

vLLM loads `nixl._api`. That module comes from the pure-Python `nixl` meta
package, which detects the CUDA major version (from torch: 12) and loads the
matching backend `nixl_cu12`. The env had only `nixl-cu12`.

The meta package declares both `nixl-cu12` and `nixl-cu13` as plain
dependencies, so it has to be installed with `--no-deps`. For the packed r2
env it sits in `/scratch/$USER/vsi/envs/overlay-r2` on `PYTHONPATH`. That
avoids forcing every node to unpack a new env: an unpack in this job took
651 s, against 33 s in the cold-start run, because `/scratch` was busy.

### 22. KV-tier run: CPU tier invisible to the gateway

Engine side, the run did what it should. With 6 GiB of GPU KV (43,680
tokens) and about 100k tokens of prefixes, the second pass hit the cache for
59.7% of prompt tokens on GPU only and 99.6% with 48 GiB of CPU offload;
throughput went from 34.8 to 53.2 req/s and TTFT p50 from 110 to 69 ms.
vLLM's own metrics showed 44.6 GB loaded back from CPU.

The gateway's index, though, reported `tier="cpu"} 0` and no orphans. In
vLLM 0.26 the CPU manager does emit `BlockStored` / `BlockRemoved` with
`medium: CPU`, but `OffloadingEventsTracker` fills in token IDs, parent hash
and block size only when `kv_connector_extra_config["self_describing_kv_events"]`
is true. Without it each event carries the offload key's hash alone. The
index needs token IDs to compute its own hash, so it returned early, and
the early return was not counted anywhere.

`--kv-offloading-size` builds the connector config itself, but it merges into
an existing `--kv-transfer-config`, so the job passes only the extra key
there. Self-describing chunks that overlap a shared prefix re-announce the
shared blocks, so offload tiers are now reference-counted (with the default
`blocks_per_chunk` of 1 this does not happen, but larger chunks would
otherwise make an eviction drop blocks another chunk still holds).

### 23. PD: NIXL's UCX backend could not reach itself

With the `nixl` meta package in place (#19), both engines got as far as
creating the NIXL agent. UCX then warned that the node's IB device
(`mlx5_0`) had no accelerated support, tried TCP, and got `Connection
refused` connecting to its own worker address. `createBackend` failed, and
so did engine start-up.

Prefill and decode run on two GPUs of the same node, so the KV copy should
not need a network transport: `cuda_ipc` copies GPU to GPU over NVLink, and
shared memory (`sm`) and `self` carry UCX's own messages. The probe tries the
default first, so the log records whether the default works on another node.

### 24. Gateway killed by SIGBUS after a push

The routing campaign runs the gateway as its own job on a CPU node, and the
replicas register with it. Six minutes into the gateway job, `push.sh` (the
push for #16/#17) unpacked new binaries into `/scratch/$USER/vsi/code` from
the login node. tar unlinks a file before writing it, which would be safe on
a local disk. On NFS the unlink happens on the server; only the client that
unlinks keeps the file alive (as a `.nfsXXXX` name), so for the compute node
the mapped executable vanished and the next page fault raised SIGBUS. Every
job now runs its own copy from `/tmp`.

The replicas started 1 h 40 min later, after the gateway was gone. They are
written to serve unregistered in that case, so they held four GPUs for about
1 h 25 min until Slurm cancelled them as root; the cause of that
cancellation does not show in the logs.

### 25. MoE: half-hour start-up, then a hang

#21's change worked: with InfiniBand off, both DP ranks passed NCCL init
and loaded the model in 11 s. Then `Initial profiling/warmup run took
1714.60 s`. The log shows the FP8 linear layers on
`FlashInferFp8DeepGEMMDynamicBlockScaledKernel` and DeepGEMM warning that
nvcc 12.8 is older than it wants: it was compiling kernels at first use.
DeepGEMM's own warmup (503 shapes) and FlashInfer's MoE autotuning then took
about 3 more minutes, and the server was ready at 05:46:31 (Eastern), 3 min
after the job had given up on it.

Giving up did not stop it. `stop_all` sent TERM to the two `vllm serve`
processes and waited; an engine in start-up ignores TERM, so `wait` blocked
until Slurm's time limit, and configurations 2–4 never ran. `stop_all` now
escalates to KILL on the whole process group after 60 s.

