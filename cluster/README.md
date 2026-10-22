# Running on Slurm (Explorer)

Scripts for Northeastern's Explorer cluster. Paths default to
`/scratch/$USER/vsi`; override with `VSI_ROOT` and `VSI_CODE`.

## Constraints these scripts work around

- `gpu` partition: one GPU per job, four per user, eight hours. Each replica
  is one job. `gpu-short` allows two GPUs for two hours.
- H200 nodes run driver 570 (CUDA 12.8). vLLM publishes no cu128 wheels, so
  the environment uses the cu129 build and relies on CUDA minor-version
  compatibility. `check_gpu.sbatch` verifies this before anything else.
- Compute nodes have no internet. Environments and weights are prepared on
  the `short` partition.
- `/scratch` is NFS. Importing a venv from it takes minutes and mmap-loading
  weights from it crawls, so jobs unpack the env and copy weights to
  node-local `/tmp` once per node; later jobs on the same node reuse them.
- sbatch runs a copy of the script from a spool directory, so jobs locate the
  code through `VSI_CODE`, not `$0`.

## Steps

```bash
# 0. On your laptop, with an open ssh master connection (ssh explorer true):
cluster/push.sh                       # binaries + scripts -> /scratch/$USER/vsi/code

# On the login node:
C=/scratch/$USER/vsi/code/cluster

# 1. Environment (once) and a model (per model):
sbatch $C/setup_env.sbatch
sbatch $C/download_model.sbatch Qwen/Qwen3-0.6B

# 2. Verify the GPU stack on an H200:
sbatch --dependency=afterok:<setup>:<download> $C/check_gpu.sbatch Qwen3-0.6B

# 3. Gateway on a CPU node, then replicas that register with it:
sbatch $C/gateway.sbatch
sbatch $C/download_model.sbatch Qwen/Qwen3-8B
for i in 1 2 3 4; do sbatch $C/vllm_replica.sbatch Qwen3-8B --max-model-len 16384; done

# 4. Use it (from any node):
curl -s $(cat /scratch/$USER/vsi/gateway.addr)/v1/models
```

Logs go to `/scratch/$USER/vsi/logs`. The admin token is created by the
gateway job in `/scratch/$USER/vsi/admin.token` with mode 600.

## Scripts

| Script | Partition | Does |
|---|---|---|
| `setup_env.sbatch` | short | Builds the vLLM venv at its node-local path and packs it to `envs/` |
| `download_model.sbatch` | short | Downloads a Hugging Face model to `models/<name>` |
| `check_gpu.sbatch` | gpu, gpu-short | Driver/runtime, torch on GPU, vLLM startup with compilation, one completion |
| `gateway.sbatch` | short | Runs the gateway; publishes its address and admin token |
| `vllm_replica.sbatch` | gpu | One vLLM replica; registers, and drains on SIGTERM before walltime |
| `push.sh` | laptop | Cross-compiles and copies code over the existing ssh master |
