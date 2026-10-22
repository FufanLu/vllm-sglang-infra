// Package backend tracks the model-server endpoints the gateway routes to:
// who they are, whether they are alive, and the engine metrics the scheduler
// scores them on.
package backend

import (
	"fmt"
	"maps"
	"net/url"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// Engine identifies the model server behind an endpoint. It decides which
// metric names the monitor reads.
type Engine string

const (
	EngineVLLM   Engine = "vllm"
	EngineSGLang Engine = "sglang"
)

// ParseEngine accepts "vllm" or "sglang" (case-insensitive). An empty string
// means vLLM, the default engine.
func ParseEngine(s string) (Engine, error) {
	switch e := Engine(strings.ToLower(strings.TrimSpace(s))); e {
	case "":
		return EngineVLLM, nil
	case EngineVLLM, EngineSGLang:
		return e, nil
	}
	return "", fmt.Errorf("unknown engine %q (want vllm or sglang)", s)
}

// ParseURL validates an endpoint base URL such as http://d4052:8000.
func ParseURL(raw string) (*url.URL, error) {
	u, err := url.Parse(strings.TrimSpace(raw))
	if err != nil {
		return nil, fmt.Errorf("invalid url %q: %w", raw, err)
	}
	if u.Scheme != "http" && u.Scheme != "https" {
		return nil, fmt.Errorf("invalid url %q: scheme must be http or https", raw)
	}
	if u.Host == "" {
		return nil, fmt.Errorf("invalid url %q: missing host", raw)
	}
	u.Path = strings.TrimRight(u.Path, "/")
	u.RawQuery, u.Fragment = "", ""
	return u, nil
}

// Snapshot is the latest engine-side view of an endpoint, scraped from its
// Prometheus /metrics page.
type Snapshot struct {
	Running   float64   // requests in the engine's running batch
	Waiting   float64   // requests queued inside the engine
	KVUsage   float64   // fraction of KV-cache capacity in use, 0..1
	ScrapedAt time.Time // zero until the first successful scrape
	// GatewayInflight is this gateway's in-flight count when the scrape was
	// sent. Running+Waiting minus it estimates load from other clients.
	GatewayInflight int64
}

// ExternalLoad estimates requests on the engine that did not come through
// this gateway: what the engine reported, minus what the gateway itself had
// in flight at the time. Requests still in transit make the difference
// negative; it is floored at zero.
func (s Snapshot) ExternalLoad() float64 {
	return max(0, s.Running+s.Waiting-float64(s.GatewayInflight))
}

// Kind separates the cluster's own model servers from external providers.
type Kind string

const (
	KindLocal    Kind = "local"
	KindExternal Kind = "external"
)

// External describes an OpenAI-compatible provider outside the cluster,
// used for overflow when local capacity runs out.
type External struct {
	APIKey string
	// Models maps gateway model names to the provider's names; only these
	// models are served by the provider.
	Models      map[string]string
	PriceInput  float64 // USD per million prompt tokens
	PriceOutput float64 // USD per million completion tokens
	MaxInflight int
}

// Cost is the price of a request with the given token counts.
func (x *External) Cost(prompt, completion int) float64 {
	return (float64(prompt)*x.PriceInput + float64(completion)*x.PriceOutput) / 1e6
}

// Spec is everything needed to register an endpoint.
type Spec struct {
	Name     string
	URL      string
	Engine   Engine
	Labels   map[string]string
	Kind     Kind      // empty means local
	External *External // set for KindExternal
}

// Endpoint is one routable model server: a vLLM or SGLang replica, a single
// data-parallel rank of one, or an external provider.
type Endpoint struct {
	Name     string
	URL      *url.URL
	Engine   Engine
	Kind     Kind
	External *External

	inflight atomic.Int64

	mu       sync.RWMutex
	labels   map[string]string
	healthy  bool
	failures int
	downAt   time.Time
	models   []string
	snap     Snapshot
}

// IsExternal reports whether the endpoint is an external provider.
func (e *Endpoint) IsExternal() bool { return e.Kind == KindExternal }

// Label returns one label value.
func (e *Endpoint) Label(key string) string {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return e.labels[key]
}

// Inflight is the number of requests the gateway has sent to this endpoint
// that have not finished yet. Unlike scraped metrics it is never stale.
func (e *Endpoint) Inflight() int64 { return e.inflight.Load() }

// Acquire and Release bracket every request the gateway forwards.
func (e *Endpoint) Acquire() { e.inflight.Add(1) }
func (e *Endpoint) Release() { e.inflight.Add(-1) }

func (e *Endpoint) Healthy() bool {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return e.healthy
}

// Labels carry deployment facts the scheduler may filter on, such as the
// Slurm job id, the node, or the role (prefill/decode) and data-parallel rank.
func (e *Endpoint) Labels() map[string]string {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return maps.Clone(e.labels)
}

func (e *Endpoint) Snapshot() Snapshot {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return e.snap
}

// Models returns the model names the endpoint serves, as reported by its
// /v1/models. It is empty until discovery succeeds.
func (e *Endpoint) Models() []string {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return slices.Clone(e.models)
}

// Serves reports whether the endpoint serves model. A local endpoint whose
// models are not known yet is assumed to serve everything; an external one
// serves exactly the models it maps.
func (e *Endpoint) Serves(model string) bool {
	if e.IsExternal() {
		_, ok := e.External.Models[model]
		return ok
	}
	e.mu.RLock()
	defer e.mu.RUnlock()
	return model == "" || len(e.models) == 0 || slices.Contains(e.models, model)
}

// DownFor is how long the endpoint has been marked down; zero if healthy.
func (e *Endpoint) DownFor() time.Duration {
	e.mu.RLock()
	defer e.mu.RUnlock()
	if e.healthy || e.downAt.IsZero() {
		return 0
	}
	return time.Since(e.downAt)
}

// RecordProbe applies the result of an active health probe. The endpoint
// becomes healthy on the first success and unhealthy after threshold
// consecutive failures. It reports whether the health state changed.
func (e *Endpoint) RecordProbe(ok bool, threshold int) bool {
	e.mu.Lock()
	defer e.mu.Unlock()
	was := e.healthy
	if ok {
		e.failures = 0
		e.healthy = true
	} else {
		e.failures++
		if e.failures >= threshold {
			e.healthy = false
		}
	}
	if was && !e.healthy {
		e.downAt = time.Now()
	}
	return was != e.healthy
}

// MarkDown takes the endpoint out of rotation at once, for failures that are
// unambiguous such as a refused connection. The next successful probe brings
// it back. It reports whether the endpoint was healthy before.
func (e *Endpoint) MarkDown() bool {
	e.mu.Lock()
	defer e.mu.Unlock()
	was := e.healthy
	e.healthy = false
	if was {
		e.downAt = time.Now()
	}
	return was
}

// SetSnapshot records engine metrics. The monitor calls it after each scrape.
func (e *Endpoint) SetSnapshot(s Snapshot) {
	e.mu.Lock()
	e.snap = s
	e.mu.Unlock()
}

// SetModels records the models the endpoint serves.
func (e *Endpoint) SetModels(m []string) {
	e.mu.Lock()
	e.models = m
	e.mu.Unlock()
}

// Registry is the set of endpoints, keyed by name. Endpoints come from the
// static config or register themselves through the admin API (a Slurm job
// does this once its model server is up).
type Registry struct {
	mu  sync.RWMutex
	eps map[string]*Endpoint
}

func NewRegistry() *Registry {
	return &Registry{eps: make(map[string]*Endpoint)}
}

// Upsert adds a local endpoint, or updates the one with the same name.
func (r *Registry) Upsert(name, rawURL string, engine Engine, labels map[string]string) (*Endpoint, error) {
	return r.UpsertSpec(Spec{Name: name, URL: rawURL, Engine: engine, Labels: labels})
}

// UpsertSpec adds an endpoint, or updates the one with the same name. If the
// URL, engine, and kind are unchanged only the labels are updated and runtime
// state is kept. Otherwise the endpoint is replaced; a local one then starts
// unhealthy, so a re-registered replica gets traffic only after it passes a
// probe. External providers have no health endpoint and start healthy.
func (r *Registry) UpsertSpec(spec Spec) (*Endpoint, error) {
	if spec.Name == "" {
		return nil, fmt.Errorf("endpoint name is empty")
	}
	if spec.Kind == "" {
		spec.Kind = KindLocal
	}
	if spec.Kind == KindExternal && spec.External == nil {
		return nil, fmt.Errorf("external endpoint %s has no provider settings", spec.Name)
	}
	u, err := ParseURL(spec.URL)
	if err != nil {
		return nil, err
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if cur, ok := r.eps[spec.Name]; ok && cur.URL.String() == u.String() && cur.Engine == spec.Engine && cur.Kind == spec.Kind {
		cur.mu.Lock()
		cur.labels = maps.Clone(spec.Labels)
		cur.mu.Unlock()
		return cur, nil
	}
	ep := &Endpoint{Name: spec.Name, URL: u, Engine: spec.Engine, Kind: spec.Kind, External: spec.External,
		labels: maps.Clone(spec.Labels), healthy: spec.Kind == KindExternal}
	r.eps[spec.Name] = ep
	return ep, nil
}

func (r *Registry) Remove(name string) bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	_, ok := r.eps[name]
	delete(r.eps, name)
	return ok
}

func (r *Registry) Get(name string) *Endpoint {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.eps[name]
}

// List returns every endpoint sorted by name, so round-robin order is stable.
func (r *Registry) List() []*Endpoint {
	r.mu.RLock()
	out := make([]*Endpoint, 0, len(r.eps))
	for _, ep := range r.eps {
		out = append(out, ep)
	}
	r.mu.RUnlock()
	slices.SortFunc(out, func(a, b *Endpoint) int { return strings.Compare(a.Name, b.Name) })
	return out
}
