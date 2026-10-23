package scheduler

import (
	"math/rand/v2"
	"sync/atomic"
	"time"

	"github.com/FufanLu/vllm-sglang-infra/internal/backend"
	"github.com/FufanLu/vllm-sglang-infra/internal/kvindex"
	"github.com/FufanLu/vllm-sglang-infra/internal/prefix"
)

// defaultMaxAge is how old a metrics snapshot may be before scorers stop
// trusting it. The monitor scrapes every 500ms by default.
const defaultMaxAge = 5 * time.Second

// defaultKVThreshold is the KV-cache usage above which the kv-usage scorer
// starts to penalize an endpoint.
const defaultKVThreshold = 0.7

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

// RoleFilter keeps endpoints whose role label matches the request: prefill
// or decode for a disaggregated pool, and for an ordinary request endpoints
// with no role (or role "both").
type RoleFilter struct{}

func (RoleFilter) Name() string { return "role" }

func (RoleFilter) Filter(req *Request, eps []*backend.Endpoint) []*backend.Endpoint {
	return keep(eps, func(e *backend.Endpoint) bool {
		role := e.Label("role")
		if req.Role == "" {
			return role == "" || role == "both"
		}
		return role == req.Role || role == "both"
	})
}

// PlacementFilter decides between the cluster and external providers.
// Local endpoints are preferred while any of them has room (in-flight below
// Saturation); when all are full the request overflows to external
// providers that have room, if the tenant allows it. Without such a
// provider the request stays local and waits in an engine queue.
type PlacementFilter struct {
	Saturation int
}

func (PlacementFilter) Name() string { return "placement" }

func (p PlacementFilter) Filter(req *Request, eps []*backend.Endpoint) []*backend.Endpoint {
	var local, external []*backend.Endpoint
	for _, e := range eps {
		if e.IsExternal() {
			if req.AllowExternal && req.Placement != LocalOnly && e.Inflight() < int64(max(e.External.MaxInflight, 1)) {
				external = append(external, e)
			}
		} else if req.Placement != ExternalOnly {
			local = append(local, e)
		}
	}
	if req.Placement == ExternalOnly || len(external) == 0 {
		if req.Placement == ExternalOnly {
			return external
		}
		return local
	}
	for _, e := range local {
		if e.Inflight() < int64(p.Saturation) {
			return local
		}
	}
	return external
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
// gateway's own in-flight count, which is exact and never stale, plus the
// engine's estimate of load from other clients when the snapshot is fresh.
//
// Scraped running/waiting counts are deliberately not used as the load
// itself: they lag by up to a scrape interval, and an endpoint that just
// finished its requests would keep looking busy, be avoided, then be
// stampeded once the next scrape shows it idle.
//
// With Saturation zero, scores are min-max normalized across candidates: the
// least loaded gets 1, the most loaded 0. That ranks well on its own but
// turns a one-request difference into a full-range gap, so when the score is
// traded against another signal (prefix affinity) Saturation is set and the
// score is absolute: 1 - load/Saturation, floored at 0.
type QueueScorer struct {
	Saturation float64
	MaxAge     time.Duration
}

func (QueueScorer) Name() string { return "queue" }

func (q QueueScorer) Score(_ *Request, eps []*backend.Endpoint) []float64 {
	loads := make([]float64, len(eps))
	now := time.Now()
	for i, e := range eps {
		loads[i] = float64(e.Inflight())
		if s := e.Snapshot(); fresh(s, now, q.MaxAge) {
			loads[i] += s.ExternalLoad()
		}
	}
	if q.Saturation <= 0 {
		return invertMinMax(loads)
	}
	for i, l := range loads {
		loads[i] = max(0, 1-l/q.Saturation)
	}
	return loads
}

func fresh(s backend.Snapshot, now time.Time, maxAge time.Duration) bool {
	return !s.ScrapedAt.IsZero() && now.Sub(s.ScrapedAt) <= maxAge
}

// PrefixScorer prefers endpoints that already hold the request's prompt
// prefix in their KV cache: the score is the fraction of the prompt's blocks
// cached there, per the approximate index. Every pick is recorded, so
// requests that share a prefix keep landing together.
type PrefixScorer struct {
	Index *prefix.Index
}

func (*PrefixScorer) Name() string { return "prefix" }

func (p *PrefixScorer) Score(req *Request, eps []*backend.Endpoint) []float64 {
	out := make([]float64, len(eps))
	hs := req.blockHashes(p.Index)
	if len(hs) == 0 {
		return out
	}
	for i, e := range eps {
		out[i] = float64(p.Index.Match(EndpointKey(e), hs)) / float64(len(hs))
	}
	return out
}

func (p *PrefixScorer) Picked(req *Request, ep *backend.Endpoint) {
	p.Index.Record(EndpointKey(ep), req.blockHashes(p.Index))
}

// EndpointKey identifies an endpoint's cache in the prefix indexes. It
// includes the URL so that a replica re-registered under the same name on
// another node starts empty.
func EndpointKey(e *backend.Endpoint) string {
	return e.Name + "@" + e.URL.String()
}

// PrecisePrefixScorer scores endpoints by the KV blocks they actually hold,
// from the engines' KV-cache events, matched against the request's token
// IDs. Blocks in a slower tier (offloaded to CPU) count CPUWeight each, since
// fetching them is cheaper than recomputing but not free.
//
// Events trail the engine by up to a flush interval, so a burst of requests
// with a new prefix would scatter before any event arrives. Recent, a small
// approximate index of where this scorer just sent each prefix, bridges that
// gap; the score is the better of the two.
type PrecisePrefixScorer struct {
	Index     *kvindex.Index
	Recent    *prefix.Index
	CPUWeight float64
	// Approx scores requests that come without tokens (no engine could
	// tokenize them, e.g. chat requests on an SGLang-only pool).
	Approx *PrefixScorer
}

func (*PrecisePrefixScorer) Name() string      { return "precise-prefix" }
func (*PrecisePrefixScorer) NeedsTokens() bool { return true }

func (p *PrecisePrefixScorer) Score(req *Request, eps []*backend.Endpoint) []float64 {
	out := make([]float64, len(eps))
	if len(req.Tokens) == 0 {
		if p.Approx != nil {
			return p.Approx.Score(req, eps)
		}
		return out
	}
	for i, e := range eps {
		key := EndpointKey(e)
		bs := p.Index.BlockSize(key)
		if bs == 0 {
			bs = defaultBlockTokens
		}
		hs := req.tokenHashes(bs)
		if len(hs) == 0 {
			continue
		}
		gpu, any := p.Index.Match(key, hs)
		exact := (float64(gpu) + p.CPUWeight*float64(any-gpu)) / float64(len(hs))
		recent := float64(p.Recent.Match(key, hs)) / float64(len(hs))
		out[i] = max(exact, recent)
	}
	return out
}

func (p *PrecisePrefixScorer) Picked(req *Request, ep *backend.Endpoint) {
	if p.Approx != nil {
		p.Approx.Picked(req, ep)
	}
	if len(req.Tokens) == 0 {
		return
	}
	key := EndpointKey(ep)
	bs := p.Index.BlockSize(key)
	if bs == 0 {
		bs = defaultBlockTokens
	}
	p.Recent.Record(key, req.tokenHashes(bs))
}

// defaultBlockTokens is vLLM's default block size, used until an endpoint's
// events report its own.
const defaultBlockTokens = 16

// KVUsageScorer steers traffic away from endpoints whose KV cache is nearly
// full, since a full cache forces the engine to queue or preempt. Below
// Threshold every endpoint scores 1: small differences in usage say nothing
// about latency, and letting them decide ties would herd requests onto
// whichever endpoint reported the lowest number at the last scrape. Above
// Threshold the score falls linearly to 0 at full. A stale snapshot scores a
// neutral 0.5.
type KVUsageScorer struct {
	MaxAge    time.Duration
	Threshold float64
}

func (KVUsageScorer) Name() string { return "kv-usage" }

func (k KVUsageScorer) Score(_ *Request, eps []*backend.Endpoint) []float64 {
	out := make([]float64, len(eps))
	now := time.Now()
	for i, e := range eps {
		s := e.Snapshot()
		switch u := min(max(s.KVUsage, 0), 1); {
		case !fresh(s, now, k.MaxAge):
			out[i] = 0.5
		case u <= k.Threshold:
			out[i] = 1
		default:
			out[i] = 1 - (u-k.Threshold)/(1-k.Threshold)
		}
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
