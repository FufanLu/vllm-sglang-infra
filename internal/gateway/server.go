// Package gateway is the HTTP front door: OpenAI-compatible inference routes
// that schedule and proxy to model servers, plus health, metrics, and an
// admin API through which replicas register themselves.
package gateway

import (
	"crypto/rand"
	"crypto/subtle"
	"encoding/hex"
	"encoding/json"
	"log/slog"
	"net/http"
	"slices"
	"strings"
	"time"

	"github.com/prometheus/client_golang/prometheus/promhttp"

	"github.com/FufanLu/vllm-sglang-infra/internal/backend"
	"github.com/FufanLu/vllm-sglang-infra/internal/config"
	"github.com/FufanLu/vllm-sglang-infra/internal/metrics"
	"github.com/FufanLu/vllm-sglang-infra/internal/scheduler"
)

type Options struct {
	Registry     *backend.Registry
	Scheduler    *scheduler.Scheduler
	Metrics      *metrics.Metrics
	Client       *http.Client // upstream; must not have an overall timeout
	Log          *slog.Logger
	MaxRetries   int
	MaxBodyBytes int64
	AdminToken   string
	// OnRegister runs after an endpoint is added through the admin API, so
	// the caller can probe it right away instead of at the next interval.
	OnRegister func(*backend.Endpoint)
}

type Server struct {
	opts Options
}

func New(o Options) *Server {
	return &Server{opts: o}
}

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
	return mux
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
		Name: ep.Name, URL: ep.URL.String(), Engine: string(ep.Engine), Labels: ep.Labels(),
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
