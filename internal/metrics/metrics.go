// Package metrics defines the gateway's Prometheus series. Request-path
// series are labelled by backend so per-replica latency can be compared under
// each scheduling policy.
package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/collectors"

	"github.com/FufanLu/vllm-sglang-infra/internal/backend"
)

type Metrics struct {
	Registry *prometheus.Registry

	Requests *prometheus.CounterVec   // backend, route, code
	Duration *prometheus.HistogramVec // backend, route
	TTFT     *prometheus.HistogramVec // backend, route
	TPOT     *prometheus.HistogramVec // backend, route
	Tokens   *prometheus.CounterVec   // backend, kind=prompt|completion
	Retries  *prometheus.CounterVec   // reason
	Picks    *prometheus.CounterVec   // policy, backend

	TenantRequests  *prometheus.CounterVec   // tenant, code
	TenantTokens    *prometheus.CounterVec   // tenant, kind
	QuotaRejections *prometheus.CounterVec   // tenant
	FlowWait        *prometheus.HistogramVec // band
	FlowRejections  *prometheus.CounterVec   // band, reason
	Overflow        *prometheus.CounterVec   // reason
	PDPrefill       *prometheus.HistogramVec // backend
	Tokenize        *prometheus.HistogramVec // ok
}

// FlowStats reports flow control's in-flight and queued counts per band
// (index 0 batch, 1 interactive).
type FlowStats func() (inflight, queued [2]int)

func New(reg *backend.Registry, flow FlowStats) *Metrics {
	m := &Metrics{
		Registry: prometheus.NewRegistry(),
		Requests: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "vsi_requests_total",
			Help: "Inference requests by backend, route, and outcome (HTTP status, or no_backend / upstream_error / client_cancel).",
		}, []string{"backend", "route", "code"}),
		Duration: prometheus.NewHistogramVec(prometheus.HistogramOpts{
			Name:    "vsi_request_duration_seconds",
			Help:    "End-to-end request latency measured at the gateway.",
			Buckets: prometheus.ExponentialBuckets(0.05, 2, 12), // 50ms .. ~100s
		}, []string{"backend", "route"}),
		TTFT: prometheus.NewHistogramVec(prometheus.HistogramOpts{
			Name:    "vsi_ttft_seconds",
			Help:    "Time to first token of streamed responses, from request arrival at the gateway.",
			Buckets: prometheus.ExponentialBuckets(0.01, 2, 12), // 10ms .. ~20s
		}, []string{"backend", "route"}),
		TPOT: prometheus.NewHistogramVec(prometheus.HistogramOpts{
			Name:    "vsi_tpot_seconds",
			Help:    "Mean time per output token after the first, for streamed responses.",
			Buckets: prometheus.ExponentialBuckets(0.002, 2, 10), // 2ms .. ~1s
		}, []string{"backend", "route"}),
		Tokens: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "vsi_tokens_total",
			Help: "Tokens reported in response usage, by kind.",
		}, []string{"backend", "kind"}),
		Retries: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "vsi_retries_total",
			Help: "Requests re-dispatched to another backend, by reason.",
		}, []string{"reason"}),
		Picks: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "vsi_scheduler_picks_total",
			Help: "Scheduling decisions by policy and chosen backend.",
		}, []string{"policy", "backend"}),
		TenantRequests: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "vsi_tenant_requests_total",
			Help: "Requests by tenant and outcome.",
		}, []string{"tenant", "code"}),
		TenantTokens: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "vsi_tenant_tokens_total",
			Help: "Tokens served per tenant, by kind (prompt, completion).",
		}, []string{"tenant", "kind"}),
		QuotaRejections: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "vsi_quota_rejections_total",
			Help: "Requests refused because the tenant's token budget was exhausted.",
		}, []string{"tenant"}),
		FlowWait: prometheus.NewHistogramVec(prometheus.HistogramOpts{
			Name:    "vsi_flow_wait_seconds",
			Help:    "Time requests waited in flow control before dispatch.",
			Buckets: prometheus.ExponentialBuckets(0.001, 4, 10), // 1ms .. ~4min
		}, []string{"band"}),
		FlowRejections: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "vsi_flow_rejections_total",
			Help: "Requests flow control turned away, by band and reason.",
		}, []string{"band", "reason"}),
		Overflow: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "vsi_overflow_total",
			Help: "Requests sent to external providers, by reason.",
		}, []string{"reason"}),
		PDPrefill: prometheus.NewHistogramVec(prometheus.HistogramOpts{
			Name:    "vsi_pd_prefill_seconds",
			Help:    "Duration of the prefill phase of disaggregated requests.",
			Buckets: prometheus.ExponentialBuckets(0.01, 2, 12),
		}, []string{"backend"}),
		Tokenize: prometheus.NewHistogramVec(prometheus.HistogramOpts{
			Name:    "vsi_tokenize_seconds",
			Help:    "Time to fetch prompt token IDs from an engine, by success.",
			Buckets: prometheus.ExponentialBuckets(0.0005, 2, 12),
		}, []string{"ok"}),
	}
	m.Registry.MustRegister(
		collectors.NewGoCollector(),
		collectors.NewProcessCollector(collectors.ProcessCollectorOpts{}),
		m.Requests, m.Duration, m.TTFT, m.TPOT, m.Tokens, m.Retries, m.Picks,
		m.TenantRequests, m.TenantTokens, m.QuotaRejections, m.FlowWait, m.FlowRejections, m.Overflow, m.PDPrefill, m.Tokenize,
		newBackendCollector(reg),
	)
	if flow != nil {
		m.Registry.MustRegister(&flowCollector{
			stats:    flow,
			inflight: prometheus.NewDesc("vsi_flow_inflight", "Requests dispatched and not finished, by band.", []string{"band"}, nil),
			queued:   prometheus.NewDesc("vsi_flow_queued", "Requests waiting in flow control, by band.", []string{"band"}, nil),
		})
	}
	return m
}

type flowCollector struct {
	stats            FlowStats
	inflight, queued *prometheus.Desc
}

func (c *flowCollector) Describe(ch chan<- *prometheus.Desc) { ch <- c.inflight; ch <- c.queued }

func (c *flowCollector) Collect(ch chan<- prometheus.Metric) {
	in, q := c.stats()
	for band, name := range []string{"batch", "interactive"} {
		ch <- prometheus.MustNewConstMetric(c.inflight, prometheus.GaugeValue, float64(in[band]), name)
		ch <- prometheus.MustNewConstMetric(c.queued, prometheus.GaugeValue, float64(q[band]), name)
	}
}

// backendCollector reads endpoint state at scrape time, so endpoints removed
// from the registry stop being reported instead of leaving stale series.
type backendCollector struct {
	reg                                        *backend.Registry
	healthy, inflight, running, waiting, kvUse *prometheus.Desc
}

func newBackendCollector(reg *backend.Registry) *backendCollector {
	d := func(name, help string) *prometheus.Desc {
		return prometheus.NewDesc(name, help, []string{"backend", "engine"}, nil)
	}
	return &backendCollector{
		reg:      reg,
		healthy:  d("vsi_backend_healthy", "1 if the backend passed its last health probe."),
		inflight: d("vsi_backend_inflight", "Requests the gateway has in flight to the backend."),
		running:  d("vsi_backend_engine_running", "Running requests reported by the engine."),
		waiting:  d("vsi_backend_engine_waiting", "Queued requests reported by the engine."),
		kvUse:    d("vsi_backend_engine_kv_usage", "KV-cache usage fraction reported by the engine."),
	}
}

func (c *backendCollector) Describe(ch chan<- *prometheus.Desc) {
	for _, d := range []*prometheus.Desc{c.healthy, c.inflight, c.running, c.waiting, c.kvUse} {
		ch <- d
	}
}

func (c *backendCollector) Collect(ch chan<- prometheus.Metric) {
	for _, ep := range c.reg.List() {
		labels := []string{ep.Name, string(ep.Engine)}
		healthy := 0.0
		if ep.Healthy() {
			healthy = 1
		}
		s := ep.Snapshot()
		ch <- prometheus.MustNewConstMetric(c.healthy, prometheus.GaugeValue, healthy, labels...)
		ch <- prometheus.MustNewConstMetric(c.inflight, prometheus.GaugeValue, float64(ep.Inflight()), labels...)
		if !s.ScrapedAt.IsZero() {
			ch <- prometheus.MustNewConstMetric(c.running, prometheus.GaugeValue, s.Running, labels...)
			ch <- prometheus.MustNewConstMetric(c.waiting, prometheus.GaugeValue, s.Waiting, labels...)
			ch <- prometheus.MustNewConstMetric(c.kvUse, prometheus.GaugeValue, s.KVUsage, labels...)
		}
	}
}

// KVIndexStats reports, for an endpoint, the blocks the precise index holds
// on GPU and CPU and the stored blocks it could not chain.
type KVIndexStats func(ep *backend.Endpoint) (gpu, cpu, orphans int)

// RegisterKVIndex exposes the precise KV index's view of each endpoint.
func (m *Metrics) RegisterKVIndex(reg *backend.Registry, stats KVIndexStats) {
	m.Registry.MustRegister(&kvIndexCollector{reg: reg, stats: stats,
		blocks:  prometheus.NewDesc("vsi_kvindex_blocks", "KV blocks the precise index believes an endpoint holds, by tier.", []string{"backend", "tier"}, nil),
		orphans: prometheus.NewDesc("vsi_kvindex_orphan_blocks", "Stored blocks the index could not key: parent never seen (missed events) or no token IDs in the event.", []string{"backend"}, nil),
	})
}

type kvIndexCollector struct {
	reg             *backend.Registry
	stats           KVIndexStats
	blocks, orphans *prometheus.Desc
}

func (c *kvIndexCollector) Describe(ch chan<- *prometheus.Desc) { ch <- c.blocks; ch <- c.orphans }

func (c *kvIndexCollector) Collect(ch chan<- prometheus.Metric) {
	for _, ep := range c.reg.List() {
		if ep.Label("kv_events") == "" {
			continue
		}
		gpu, cpu, orphans := c.stats(ep)
		ch <- prometheus.MustNewConstMetric(c.blocks, prometheus.GaugeValue, float64(gpu), ep.Name, "gpu")
		ch <- prometheus.MustNewConstMetric(c.blocks, prometheus.GaugeValue, float64(cpu), ep.Name, "cpu")
		ch <- prometheus.MustNewConstMetric(c.orphans, prometheus.GaugeValue, float64(orphans), ep.Name)
	}
}
