package scheduler

import (
	"math/rand/v2"
	"sync/atomic"
	"time"

	"github.com/FufanLu/vllm-sglang-infra/internal/backend"
)

// defaultMaxAge is how old a metrics snapshot may be before scorers stop
// trusting it. The monitor scrapes every 500ms by default.
const defaultMaxAge = 5 * time.Second

// HealthyFilter keeps endpoints whose last health probe passed.
type HealthyFilter struct{}

func (HealthyFilter) Name() string { return "healthy" }

func (HealthyFilter) Filter(_ *Request, eps []*backend.Endpoint) []*backend.Endpoint {
	return keep(eps, (*backend.Endpoint).Healthy)
}

// ModelFilter keeps endpoints that serve the requested model.
type ModelFilter struct{}

func (ModelFilter) Name() string { return "model" }

func (ModelFilter) Filter(req *Request, eps []*backend.Endpoint) []*backend.Endpoint {
	return keep(eps, func(e *backend.Endpoint) bool { return e.Serves(req.Model) })
}

func keep(eps []*backend.Endpoint, ok func(*backend.Endpoint) bool) []*backend.Endpoint {
	out := eps[:0:0]
	for _, e := range eps {
		if ok(e) {
			out = append(out, e)
		}
	}
	return out
}

// QueueScorer prefers endpoints with fewer outstanding requests. Load is the
// larger of the gateway's own in-flight count (exact, never stale) and the
// engine's running+waiting count (covers traffic that bypasses the gateway).
// Scores are min-max normalized across candidates: least loaded gets 1.
type QueueScorer struct{}

func (QueueScorer) Name() string { return "queue" }

func (QueueScorer) Score(_ *Request, eps []*backend.Endpoint) []float64 {
	loads := make([]float64, len(eps))
	for i, e := range eps {
		s := e.Snapshot()
		loads[i] = max(float64(e.Inflight()), s.Running+s.Waiting)
	}
	return invertMinMax(loads)
}

// KVUsageScorer prefers endpoints with more free KV cache, since a full cache
// forces the engine to queue or preempt. A stale snapshot scores a neutral
// 0.5 rather than looking idle.
type KVUsageScorer struct {
	MaxAge time.Duration
}

func (KVUsageScorer) Name() string { return "kv-usage" }

func (k KVUsageScorer) Score(_ *Request, eps []*backend.Endpoint) []float64 {
	out := make([]float64, len(eps))
	now := time.Now()
	for i, e := range eps {
		s := e.Snapshot()
		if s.ScrapedAt.IsZero() || now.Sub(s.ScrapedAt) > k.MaxAge {
			out[i] = 0.5
			continue
		}
		out[i] = 1 - min(max(s.KVUsage, 0), 1)
	}
	return out
}

// invertMinMax maps the smallest value to 1 and the largest to 0. If all
// values are equal every score is 1.
func invertMinMax(vs []float64) []float64 {
	lo, hi := vs[0], vs[0]
	for _, v := range vs[1:] {
		lo, hi = min(lo, v), max(hi, v)
	}
	out := make([]float64, len(vs))
	for i, v := range vs {
		if hi == lo {
			out[i] = 1
		} else {
			out[i] = (hi - v) / (hi - lo)
		}
	}
	return out
}

// MaxScorePicker picks the highest total score. Ties are broken uniformly at
// random so equally good endpoints share traffic.
type MaxScorePicker struct{}

func (MaxScorePicker) Name() string { return "max-score" }

func (MaxScorePicker) Pick(eps []*backend.Endpoint, scores []float64) *backend.Endpoint {
	const eps64 = 1e-9
	best, n := 0, 0
	for i, s := range scores {
		switch {
		case s > scores[best]+eps64:
			best, n = i, 1
		case s >= scores[best]-eps64:
			// Reservoir sampling over the tied set.
			n++
			if rand.IntN(n) == 0 {
				best = i
			}
		}
	}
	return eps[best]
}

// RoundRobinPicker ignores scores and cycles through the candidates. It is
// the baseline every other policy is measured against.
type RoundRobinPicker struct {
	next atomic.Uint64
}

func (*RoundRobinPicker) Name() string { return "round-robin" }

func (p *RoundRobinPicker) Pick(eps []*backend.Endpoint, _ []float64) *backend.Endpoint {
	return eps[(p.next.Add(1)-1)%uint64(len(eps))]
}
