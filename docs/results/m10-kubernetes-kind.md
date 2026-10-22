# M10 — Kubernetes on kind (simulated engines)

`deploy/kind/demo.sh` installs the Helm chart on a local kind cluster with
four simulated model servers, kvrelay, and the gateway.

**Discovery.** The gateway lists pods labelled `vsi.io/role=model-server`
through the Kubernetes API, using its service account (RBAC: pods
get/list, deployments/scale get/patch). Each ready pod becomes an endpoint
at `http://<pod IP>:8000`. Annotations `vsi.io/*` become endpoint labels,
for example `vsi.io/kv_events`. Static endpoints and self-registered ones
are left alone.

```
demo-vsi-sim-…-jt8h8 http://10.244.0.8:8000  healthy  kv_events=tcp://demo-vsi-kvrelay:5558
demo-vsi-sim-…-qpnvl http://10.244.0.5:8000  healthy  …
demo-vsi-sim-…-r4zmw http://10.244.0.10:8000 healthy  …
demo-vsi-sim-…-w8rgz http://10.244.0.9:8000  healthy  …
```

**Precise routing.** The simulators publish KV events with their pod IP in
the topic. The gateway attributes them through kvrelay, and its index held
213–323 blocks per pod. 300 requests over 24 shared prefixes: 83% of prompt
tokens were served from cache.

**Autoscaling** with the `kubernetes` provider, which scales the
model-server Deployment. Settings: min 1, max 6, target 8 in flight per
replica, scale down after 15 s.

```
  0s  replicas=4  idle
 25s  replicas=1  scaled down after 15 s of low demand
 30s  replicas=5  40 requests in flight -> ceil(40/8) = 5
 41s  replicas=5  all five ready
 86s  replicas=1  load finished, scaled down again
```

2500 requests, 0 errors. TTFT p50 45 ms; p99 1626 ms for requests that
queued in the gateway before the new pods were ready.

For GPU clusters, `vllm.enabled=true` deploys vLLM pods instead. They carry
the annotation `vsi.io/kv_events: "pod:5557"` (the pod's own publisher) and
a 300 s termination grace period, so in-flight requests drain on scale-down.
