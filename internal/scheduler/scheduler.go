// Package scheduler picks the endpoint for each inference request. It uses
// the filter → score → pick pipeline of llm-d's Endpoint Picker, so a policy
// is a small set of plugins: filters drop ineligible endpoints, scorers rate
// the rest in [0, 1], and a picker chooses from the weighted sum.
package scheduler

import (
	"errors"
	"fmt"
	"slices"
	"sort"
	"sync"

	"github.com/FufanLu/vllm-sglang-infra/internal/backend"
	"github.com/FufanLu/vllm-sglang-infra/internal/kvindex"
	"github.com/FufanLu/vllm-sglang-infra/internal/prefix"
)

// Placement restricts where a request may run.
type Placement int

const (
	// Auto keeps the request on the cluster unless every local endpoint is
	// saturated, then overflows it to an external provider if allowed.
	Auto Placement = iota
	LocalOnly
	ExternalOnly
)

// Request is what the scheduler knows about an inference request.
type Request struct {
	Model string
	// Prompt is the flattened prompt text. Requests that share a prefix
	// (system prompt, earlier turns) share a string prefix here, which is what
	// prefix-aware scorers key on.
	Prompt string
	Tenant string
	// AllowExternal is false for tenants whose data must stay on the cluster.
	AllowExternal bool
	Placement     Placement
	// Role selects prefill or decode endpoints of a disaggregated pool; empty
	// selects aggregated endpoints that do both.
	Role string
	// Tokens are the prompt's token IDs, set when the policy needs them
	// (see NeedsTokens) and the engine could tokenize the request.
	Tokens []uint32

	tokenHashCache map[int][]uint64

	hashes   []uint64 // prompt block hashes, computed once per request
	hashedBy *prefix.Index
}

// tokenHashes returns the chained hashes of the request's full token blocks
// of size bs, computed once per block size.
func (r *Request) tokenHashes(bs int) []uint64 {
	if hs, ok := r.tokenHashCache[bs]; ok {
		return hs
	}
	if r.tokenHashCache == nil {
		r.tokenHashCache = map[int][]uint64{}
	}
	hs := kvindex.Hashes(r.Tokens, bs)
	r.tokenHashCache[bs] = hs
	return hs
}

// blockHashes returns the request's prompt block hashes under ix, computing
// them on first use so the scorer and the recorder share one pass.
func (r *Request) blockHashes(ix *prefix.Index) []uint64 {
	if r.hashedBy != ix {
		r.hashes, r.hashedBy = ix.Hashes(r.Prompt), ix
	}
	return r.hashes
}

// Filter removes endpoints that must not receive the request.
type Filter interface {
	Name() string
	Filter(req *Request, eps []*backend.Endpoint) []*backend.Endpoint
}

// Scorer rates every candidate in [0, 1], higher is better. The returned
// slice is parallel to eps.
type Scorer interface {
	Name() string
	Score(req *Request, eps []*backend.Endpoint) []float64
}

// Picker chooses one endpoint given the weighted total score of each.
type Picker interface {
	Name() string
	Pick(eps []*backend.Endpoint, scores []float64) *backend.Endpoint
}

// Recorder is implemented by plugins that learn from decisions, such as the
// prefix scorer remembering where each prompt was sent.
type Recorder interface {
	Picked(req *Request, ep *backend.Endpoint)
}

type WeightedScorer struct {
	Scorer Scorer
	Weight float64
}

// ErrNoEndpoint means no endpoint survived filtering.
var ErrNoEndpoint = errors.New("no eligible endpoint")

type Scheduler struct {
	Policy  string
	Filters []Filter
	Scorers []WeightedScorer
	Picker  Picker

	mu sync.Mutex
}

// Schedule runs the pipeline over eps, skipping endpoints named in exclude
// (those already tried for this request), and returns the chosen endpoint
// with one in-flight slot acquired. The caller must Release it.
//
// Picks are serialized. Scoring takes microseconds, and serializing means a
// burst of requests sees each other's in-flight counts instead of all landing
// on the endpoint that looked idle at the last metrics scrape.
func (s *Scheduler) Schedule(req *Request, eps []*backend.Endpoint, exclude map[string]bool) (*backend.Endpoint, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	cands := slices.DeleteFunc(slices.Clone(eps), func(e *backend.Endpoint) bool { return exclude[e.Name] })
	if len(cands) == 0 {
		return nil, ErrNoEndpoint
	}
	for _, f := range s.Filters {
		if cands = f.Filter(req, cands); len(cands) == 0 {
			return nil, fmt.Errorf("%w: filter %q removed every candidate", ErrNoEndpoint, f.Name())
		}
	}
	total := make([]float64, len(cands))
	for _, ws := range s.Scorers {
		for i, v := range ws.Scorer.Score(req, cands) {
			total[i] += ws.Weight * v
		}
	}
	ep := s.Picker.Pick(cands, total)
	ep.Acquire()
	for _, ws := range s.Scorers {
		if r, ok := ws.Scorer.(Recorder); ok {
			r.Picked(req, ep)
		}
	}
	return ep, nil
}

// NeedsTokens reports whether a scorer needs the prompt's token IDs.
func (s *Scheduler) NeedsTokens() bool {
	for _, ws := range s.Scorers {
		if n, ok := ws.Scorer.(interface{ NeedsTokens() bool }); ok && n.NeedsTokens() {
			return true
		}
	}
	return false
}

// policies maps a policy name to its scorers and their default weights. A
// nil map means the policy uses no scorers.
var policies = map[string]map[string]float64{
	"round-robin":          nil,
	"least-load":           {"queue": 1.0, "kv-usage": 0.5},
	"prefix-aware":         {"prefix": 1.0, "queue": 2.0, "kv-usage": 0.5},
	"precise-prefix-aware": {"precise-prefix": 1.0, "queue": 2.0, "kv-usage": 0.5},
}

// Policies lists the policy names New accepts.
func Policies() []string {
	names := make([]string, 0, len(policies))
	for n := range policies {
		names = append(names, n)
	}
	sort.Strings(names)
	return names
}

// Options tune a policy. Zero values take the defaults.
type Options struct {
	// Weights overrides the policy's scorer weights by scorer name.
	Weights map[string]float64
	// QueueSaturation is the load (requests) at which the prefix-aware
	// policy's queue score reaches 0. With prefix weight p and queue weight
	// q, an endpoint holding the whole prefix keeps winning until it is
	// QueueSaturation*p/q requests busier than an idle one.
	QueueSaturation float64
	// PrefixBlockChars and PrefixCapacityBlocks size the prefix index.
	PrefixBlockChars     int
	PrefixCapacityBlocks int
	// Saturation is the in-flight count at which a local endpoint counts as
	// full for overflow decisions.
	Saturation int
	// KVIndex is the event-fed index the precise-prefix-aware policy reads.
	KVIndex *kvindex.Index
	// CPUWeight is what a prefix block held only in CPU memory is worth
	// relative to one on GPU (precise-prefix-aware). Default 0.5.
	CPUWeight float64
}

const (
	DefaultSaturation           = 32
	DefaultQueueSaturation      = 32
	DefaultPrefixBlockChars     = 128
	DefaultPrefixCapacityBlocks = 16384
)

// New builds the scheduler for a named policy.
func New(policy string, o Options) (*Scheduler, error) {
	defaults, ok := policies[policy]
	if !ok {
		return nil, fmt.Errorf("unknown scheduler policy %q (want one of %v)", policy, Policies())
	}
	for name := range o.Weights {
		if _, ok := defaults[name]; !ok {
			return nil, fmt.Errorf("policy %q has no scorer %q", policy, name)
		}
	}
	s := &Scheduler{
		Policy: policy,
		Filters: []Filter{HealthyFilter{}, ModelFilter{}, RoleFilter{},
			PlacementFilter{Saturation: cmpOr(o.Saturation, DefaultSaturation)}},
	}
	if policy == "round-robin" {
		s.Picker = &RoundRobinPicker{}
		return s, nil
	}

	var scorers []Scorer
	kvScorer := KVUsageScorer{MaxAge: defaultMaxAge, Threshold: defaultKVThreshold}
	switch policy {
	case "least-load":
		scorers = []Scorer{QueueScorer{MaxAge: defaultMaxAge}, kvScorer}
	case "prefix-aware":
		sat := cmpOr(o.QueueSaturation, DefaultQueueSaturation)
		ix := prefix.New(cmpOr(o.PrefixBlockChars, DefaultPrefixBlockChars),
			cmpOr(o.PrefixCapacityBlocks, DefaultPrefixCapacityBlocks))
		scorers = []Scorer{&PrefixScorer{Index: ix}, QueueScorer{Saturation: sat, MaxAge: defaultMaxAge}, kvScorer}
	case "precise-prefix-aware":
		if o.KVIndex == nil {
			return nil, fmt.Errorf("policy %q needs a KV event index", policy)
		}
		sat := cmpOr(o.QueueSaturation, DefaultQueueSaturation)
		recent := prefix.New(1, 4096) // keyed by token-block hashes; block size unused
		approx := prefix.New(cmpOr(o.PrefixBlockChars, DefaultPrefixBlockChars),
			cmpOr(o.PrefixCapacityBlocks, DefaultPrefixCapacityBlocks))
		scorers = []Scorer{&PrecisePrefixScorer{Index: o.KVIndex, Recent: recent, CPUWeight: cmpOr(o.CPUWeight, 0.5),
			Approx: &PrefixScorer{Index: approx}},
			QueueScorer{Saturation: sat, MaxAge: defaultMaxAge}, kvScorer}
	}
	for _, sc := range scorers {
		w, ok := o.Weights[sc.Name()]
		if !ok {
			w = defaults[sc.Name()]
		}
		s.Scorers = append(s.Scorers, WeightedScorer{Scorer: sc, Weight: w})
	}
	s.Picker = MaxScorePicker{}
	return s, nil
}

// cmpOr returns v, or def when v is zero or negative.
func cmpOr[T int | float64](v, def T) T {
	if v > 0 {
		return v
	}
	return def
}
