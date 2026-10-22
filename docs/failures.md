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
| 21 | 10-22 02:00 | Explorer, MoE rerun `11297201` | Same segfault in `ncclNetPluginInit` with `NCCL_NET_PLUGIN=none` | NCCL 2.28 starts its built-in transports (IB, socket) through the same plugin path; the crash is likely in IB init on nodes with an IB NIC (not yet confirmed) | `NCCL_IB_DISABLE=1` for single-node jobs; `NCCL_DEBUG=INFO` (INIT,NET) in the MoE job; MoE configurations now run independently so one failing start does not lose the rest | `70fb6e8` | fix pending |

Mistakes in my own commands, not in the project. They cost time, so they are listed too:

| When | What happened | Lesson |
|---|---|---|
| 10-21 19:50 | zsh did not split `$F` into compose arguments | Run multi-word variables under `bash -c`, or use arrays |
| 10-21 19:50 | Ran a stale `bin/loadgen` without the new flag | Rebuild (`go build -o bin/ ./cmd/...`) before each run |
| 10-21 20:20 | A nested-quote `curl` returned 400 | Quote JSON with single quotes, outside `bash -c` |
| 10-22 00:35 | `timeout` is not available on macOS | Use the tool's own timeout |

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
