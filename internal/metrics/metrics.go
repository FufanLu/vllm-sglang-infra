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
}

func New(reg *backend.Registry) *Metrics {
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
	}
	m.Registry.MustRegister(
		collectors.NewGoCollector(),
		collectors.NewProcessCollector(collectors.ProcessCollectorOpts{}),
		m.Requests, m.Duration, m.TTFT, m.TPOT, m.Tokens, m.Retries, m.Picks,
		newBackendCollector(reg),
	)
	return m
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
