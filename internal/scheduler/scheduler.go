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
)

// Request is what the scheduler knows about an inference request.
type Request struct {
	Model string
	// Prompt is the flattened prompt text. Requests that share a prefix
	// (system prompt, earlier turns) share a string prefix here, which is what
	// prefix-aware scorers key on.
	Prompt string
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
	return ep, nil
}

// policies maps a policy name to its default scorer weights. A nil map means
// the policy uses no scorers.
var policies = map[string]map[string]float64{
	"round-robin": nil,
	"least-load":  {"queue": 1.0, "kv-usage": 0.5},
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

// New builds the scheduler for a named policy. weights overrides the
// policy's default scorer weights by scorer name.
func New(policy string, weights map[string]float64) (*Scheduler, error) {
	defaults, ok := policies[policy]
	if !ok {
		return nil, fmt.Errorf("unknown scheduler policy %q (want one of %v)", policy, Policies())
	}
	s := &Scheduler{
		Policy:  policy,
		Filters: []Filter{HealthyFilter{}, ModelFilter{}},
	}
	if policy == "round-robin" {
		s.Picker = &RoundRobinPicker{}
		return s, nil
	}
	for name := range weights {
		if _, ok := defaults[name]; !ok {
			return nil, fmt.Errorf("policy %q has no scorer %q", policy, name)
		}
	}
	for _, sc := range []Scorer{QueueScorer{}, KVUsageScorer{MaxAge: defaultMaxAge}} {
		w, ok := weights[sc.Name()]
		if !ok {
			w = defaults[sc.Name()]
		}
		s.Scorers = append(s.Scorers, WeightedScorer{Scorer: sc, Weight: w})
	}
	s.Picker = MaxScorePicker{}
	return s, nil
}
