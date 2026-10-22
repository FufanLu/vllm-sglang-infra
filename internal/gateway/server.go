// Package gateway is the HTTP front door: OpenAI-compatible inference routes
// that schedule and proxy to model servers, plus health, metrics, and an
// admin API through which replicas register themselves.
package gateway

import (
	"context"
	"crypto/rand"
	"crypto/subtle"
	"encoding/hex"
	"encoding/json"
	"log/slog"
	"net/http"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/prometheus/client_golang/prometheus/promhttp"

	"github.com/FufanLu/vllm-sglang-infra/internal/backend"
	"github.com/FufanLu/vllm-sglang-infra/internal/config"
	"github.com/FufanLu/vllm-sglang-infra/internal/flow"
	"github.com/FufanLu/vllm-sglang-infra/internal/metrics"
	"github.com/FufanLu/vllm-sglang-infra/internal/quota"
	"github.com/FufanLu/vllm-sglang-infra/internal/scheduler"
	"github.com/FufanLu/vllm-sglang-infra/internal/tenancy"
	"github.com/FufanLu/vllm-sglang-infra/internal/tokenize"
	"github.com/FufanLu/vllm-sglang-infra/internal/usage"
)

type Options struct {
	Registry  *backend.Registry
	Scheduler *scheduler.Scheduler
	// SchedulerOptions are kept so the admin API can rebuild the scheduler
	// under another policy.
	SchedulerOptions scheduler.Options
	Metrics          *metrics.Metrics
	Client           *http.Client // upstream; must not have an overall timeout
	Log              *slog.Logger
	MaxRetries       int
	MaxBodyBytes     int64
	AdminToken       string
	// OnRegister runs after an endpoint is added through the admin API, so
	// the caller can probe it right away instead of at the next interval.
	OnRegister func(*backend.Endpoint)

	Tenants *tenancy.Directory // nil: open, every request is tenancy.Anonymous
	Quota   quota.Limiter      // nil: no budgets
	Usage   usage.Sink         // nil: records discarded
	Flow    *flow.Controller   // nil: no gateway-side queueing
	PD      bool               // split requests over prefill and decode endpoints
	// Tokenizer gets prompt token IDs from an engine for policies that
	// match KV blocks exactly (precise-prefix-aware).
	Tokenizer *tokenize.Client
	// DefaultMaxTokens is the output length assumed for quota reservation
	// when a request does not set max_tokens.
	DefaultMaxTokens int
}

type Server struct {
	opts  Options
	sched atomic.Pointer[scheduler.Scheduler]
}

func New(o Options) *Server {
	if o.Tenants == nil {
		o.Tenants = tenancy.NewOpen()
	}
	if o.Quota == nil {
		o.Quota = quota.NewMemory()
	}
	if o.Usage == nil {
		o.Usage = usage.Discard{}
	}
	if o.DefaultMaxTokens <= 0 {
		o.DefaultMaxTokens = 512
	}
	s := &Server{opts: o}
	s.sched.Store(o.Scheduler)
	return s
}

// scheduler returns the current scheduler; the admin API can swap it.
func (s *Server) scheduler() *scheduler.Scheduler { return s.sched.Load() }

func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/chat/completions", s.inference("chat"))
	mux.HandleFunc("POST /v1/completions", s.inference("completions"))
	mux.HandleFunc("GET /v1/models", s.models)
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, _ *http.Request) { w.Write([]byte("ok\n")) })
	mux.HandleFunc("GET /readyz", s.ready)
	mux.Handle("GET /metrics", promhttp.HandlerFor(s.opts.Metrics.Registry, promhttp.HandlerOpts{}))
	mux.HandleFunc("GET /admin/backends", s.admin(s.listBackends))
	mux.HandleFunc("PUT /admin/backends/{name}", s.admin(s.putBackend))
	mux.HandleFunc("DELETE /admin/backends/{name}", s.admin(s.deleteBackend))
	mux.HandleFunc("GET /admin/policy", s.admin(s.getPolicy))
	mux.HandleFunc("PUT /admin/policy", s.admin(s.putPolicy))
	mux.HandleFunc("POST /admin/reset_prefix_cache", s.admin(s.resetPrefixCache))
	return mux
}

type policyBody struct {
	Policy          string             `json:"policy"`
	Weights         map[string]float64 `json:"weights,omitempty"`
	QueueSaturation float64            `json:"queue_saturation,omitempty"`
}

func (s *Server) getPolicy(w http.ResponseWriter, _ *http.Request) {
	sched := s.scheduler()
	body := policyBody{Policy: sched.Policy, Weights: map[string]float64{}}
	for _, ws := range sched.Scorers {
		body.Weights[ws.Scorer.Name()] = ws.Weight
	}
	writeJSON(w, http.StatusOK, body)
}

// putPolicy swaps the scheduling policy at runtime, so experiments can
// compare policies on warm replicas. The new scheduler starts with an empty
// prefix index; reset the engines' prefix caches too for a clean comparison.
func (s *Server) putPolicy(w http.ResponseWriter, r *http.Request) {
	var body policyBody
	if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 64<<10)).Decode(&body); err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request_error", "body must be JSON with policy, weights")
		return
	}
	o := s.opts.SchedulerOptions
	o.Weights = body.Weights
	if body.QueueSaturation > 0 {
		o.QueueSaturation = body.QueueSaturation
	}
	sched, err := scheduler.New(body.Policy, o)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	s.sched.Store(sched)
	s.opts.Log.Info("scheduler policy changed", "policy", sched.Policy)
	s.getPolicy(w, r)
}

// resetPrefixCache asks every local engine to drop its prefix cache (vLLM
// serves /reset_prefix_cache when VLLM_SERVER_DEV_MODE=1).
func (s *Server) resetPrefixCache(w http.ResponseWriter, r *http.Request) {
	results := map[string]string{}
	var mu sync.Mutex
	var wg sync.WaitGroup
	for _, ep := range s.opts.Registry.List() {
		if ep.IsExternal() {
			continue
		}
		wg.Go(func() {
			status := "ok"
			ctx, cancel := context.WithTimeout(r.Context(), 10*time.Second)
			defer cancel()
			req, _ := http.NewRequestWithContext(ctx, http.MethodPost, ep.URL.String()+"/reset_prefix_cache", nil)
			resp, err := s.opts.Client.Do(req)
			switch {
			case err != nil:
				status = err.Error()
			case resp.StatusCode >= 300:
				status = resp.Status
			}
			if resp != nil {
				resp.Body.Close()
			}
			mu.Lock()
			results[ep.Name] = status
			mu.Unlock()
		})
	}
	wg.Wait()
	writeJSON(w, http.StatusOK, results)
}

// ready is 200 once at least one backend is healthy.
func (s *Server) ready(w http.ResponseWriter, _ *http.Request) {
	for _, ep := range s.opts.Registry.List() {
		if ep.Healthy() {
			w.Write([]byte("ready\n"))
			return
		}
	}
	http.Error(w, "no healthy backend", http.StatusServiceUnavailable)
}

// models lists the union of models served by healthy backends.
func (s *Server) models(w http.ResponseWriter, _ *http.Request) {
	set := map[string]bool{}
	for _, ep := range s.opts.Registry.List() {
		if ep.Healthy() {
			for _, m := range ep.Models() {
				set[m] = true
			}
		}
	}
	type model struct {
		ID      string `json:"id"`
		Object  string `json:"object"`
		OwnedBy string `json:"owned_by"`
	}
	resp := struct {
		Object string  `json:"object"`
		Data   []model `json:"data"`
	}{Object: "list", Data: []model{}}
	for m := range set {
		resp.Data = append(resp.Data, model{ID: m, Object: "model", OwnedBy: "vllm-sglang-infra"})
	}
	slices.SortFunc(resp.Data, func(a, b model) int { return strings.Compare(a.ID, b.ID) })
	writeJSON(w, http.StatusOK, resp)
}

// admin wraps an admin handler with bearer-token authentication. With no
// token configured the admin API is disabled.
func (s *Server) admin(h http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if s.opts.AdminToken == "" {
			writeError(w, http.StatusForbidden, "forbidden", "admin API disabled: set admin.token or VSI_ADMIN_TOKEN")
			return
		}
		got, ok := strings.CutPrefix(r.Header.Get("Authorization"), "Bearer ")
		if !ok || subtle.ConstantTimeCompare([]byte(got), []byte(s.opts.AdminToken)) != 1 {
			writeError(w, http.StatusUnauthorized, "unauthorized", "missing or wrong admin token")
			return
		}
		h(w, r)
	}
}

type backendView struct {
	Name      string            `json:"name"`
	URL       string            `json:"url"`
	Engine    string            `json:"engine"`
	Kind      string            `json:"kind"`
	Labels    map[string]string `json:"labels,omitempty"`
	Healthy   bool              `json:"healthy"`
	Inflight  int64             `json:"inflight"`
	Models    []string          `json:"models,omitempty"`
	Running   float64           `json:"engine_running"`
	Waiting   float64           `json:"engine_waiting"`
	KVUsage   float64           `json:"engine_kv_usage"`
	ScrapedAt *time.Time        `json:"scraped_at,omitempty"`
}

func view(ep *backend.Endpoint) backendView {
	snap := ep.Snapshot()
	v := backendView{
		Name: ep.Name, URL: ep.URL.String(), Engine: string(ep.Engine), Kind: string(ep.Kind), Labels: ep.Labels(),
		Healthy: ep.Healthy(), Inflight: ep.Inflight(), Models: ep.Models(),
		Running: snap.Running, Waiting: snap.Waiting, KVUsage: snap.KVUsage,
	}
	if !snap.ScrapedAt.IsZero() {
		v.ScrapedAt = &snap.ScrapedAt
	}
	return v
}

func (s *Server) listBackends(w http.ResponseWriter, _ *http.Request) {
	out := []backendView{}
	for _, ep := range s.opts.Registry.List() {
		out = append(out, view(ep))
	}
	writeJSON(w, http.StatusOK, out)
}

// putBackend registers or updates an endpoint. A Slurm job calls it once its
// model server answers /health, and calls deleteBackend before it exits.
func (s *Server) putBackend(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	if !config.ValidName(name) {
		writeError(w, http.StatusBadRequest, "invalid_request_error", "invalid backend name")
		return
	}
	var body struct {
		URL    string            `json:"url"`
		Engine string            `json:"engine"`
		Labels map[string]string `json:"labels"`
	}
	if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 64<<10)).Decode(&body); err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request_error", "body must be JSON with url, engine, labels")
		return
	}
	engine, err := backend.ParseEngine(body.Engine)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	ep, err := s.opts.Registry.Upsert(name, body.URL, engine, body.Labels)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	s.opts.Log.Info("backend registered", "endpoint", ep.Name, "url", ep.URL.String(), "engine", ep.Engine)
	if s.opts.OnRegister != nil {
		s.opts.OnRegister(ep)
	}
	writeJSON(w, http.StatusOK, view(ep))
}

func (s *Server) deleteBackend(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	if !s.opts.Registry.Remove(name) {
		writeError(w, http.StatusNotFound, "not_found", "no backend named "+name)
		return
	}
	s.opts.Log.Info("backend removed", "endpoint", name)
	w.WriteHeader(http.StatusNoContent)
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(v)
}

func newRequestID() string {
	var b [12]byte
	rand.Read(b[:])
	return hex.EncodeToString(b[:])
}
