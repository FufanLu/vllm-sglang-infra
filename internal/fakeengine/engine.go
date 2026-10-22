// Package fakeengine is a small OpenAI-compatible model server for tests and
// for demos without a GPU. It covers what the gateway depends on: completion
// endpoints (streamed and not), /health, /v1/models, and the vllm:* metrics
// the scheduler reads. For realistic latency, KV-cache simulation, and KV
// events, use llm-d-inference-sim (see deploy/local).
package fakeengine

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

type Config struct {
	Model      string
	TTFT       time.Duration // delay before the first token
	TPOT       time.Duration // delay between later tokens
	Tokens     int           // completion length when the request sets no max_tokens
	KVCapacity int           // concurrent requests that fill the KV cache
	// APIKey, if set, is required as a bearer token, as an external
	// provider would.
	APIKey string
}

type Engine struct {
	cfg        Config
	running    atomic.Int64
	served     atomic.Int64
	failStatus atomic.Int64
	resets     atomic.Int64

	mu         sync.Mutex
	lastHeader http.Header
	lastBody   map[string]any
}

// LastRequest returns the headers and decoded body of the most recent
// completion request.
func (e *Engine) LastRequest() (http.Header, map[string]any) {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.lastHeader, e.lastBody
}

// Resets counts POST /reset_prefix_cache calls.
func (e *Engine) Resets() int64 { return e.resets.Load() }

func New(cfg Config) *Engine {
	if cfg.Tokens <= 0 {
		cfg.Tokens = 16
	}
	if cfg.KVCapacity <= 0 {
		cfg.KVCapacity = 64
	}
	return &Engine{cfg: cfg}
}

// Served counts completion requests that began generating.
func (e *Engine) Served() int64 { return e.served.Load() }

// Running is the number of requests generating right now.
func (e *Engine) Running() int64 { return e.running.Load() }

// SetFailStatus makes every completion request fail with code; 0 restores
// normal behaviour. Tests use it to exercise retries.
func (e *Engine) SetFailStatus(code int) { e.failStatus.Store(int64(code)) }

func (e *Engine) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, _ *http.Request) {})
	mux.HandleFunc("GET /v1/models", e.models)
	mux.HandleFunc("GET /metrics", e.metrics)
	mux.HandleFunc("POST /v1/chat/completions", e.complete(true))
	mux.HandleFunc("POST /v1/completions", e.complete(false))
	mux.HandleFunc("POST /reset_prefix_cache", func(w http.ResponseWriter, _ *http.Request) { e.resets.Add(1) })
	return mux
}

func (e *Engine) models(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	fmt.Fprintf(w, `{"object":"list","data":[{"id":%q,"object":"model","owned_by":"fakeengine"}]}`, e.cfg.Model)
}

func (e *Engine) metrics(w http.ResponseWriter, _ *http.Request) {
	running := e.running.Load()
	kv := min(float64(running)/float64(e.cfg.KVCapacity), 1)
	w.Header().Set("Content-Type", "text/plain; version=0.0.4")
	fmt.Fprintf(w, "# HELP vllm:num_requests_running Number of requests in model execution batches.\n")
	fmt.Fprintf(w, "# TYPE vllm:num_requests_running gauge\n")
	fmt.Fprintf(w, "vllm:num_requests_running{model_name=%q} %d\n", e.cfg.Model, running)
	fmt.Fprintf(w, "# TYPE vllm:num_requests_waiting gauge\n")
	fmt.Fprintf(w, "vllm:num_requests_waiting{model_name=%q} 0\n", e.cfg.Model)
	fmt.Fprintf(w, "# TYPE vllm:kv_cache_usage_perc gauge\n")
	fmt.Fprintf(w, "vllm:kv_cache_usage_perc{model_name=%q} %g\n", e.cfg.Model, kv)
}

type request struct {
	Model         string `json:"model"`
	Stream        bool   `json:"stream"`
	MaxTokens     int    `json:"max_tokens"`
	StreamOptions struct {
		IncludeUsage bool `json:"include_usage"`
	} `json:"stream_options"`
	KVTransferParams *struct {
		DoRemoteDecode  bool `json:"do_remote_decode"`
		DoRemotePrefill bool `json:"do_remote_prefill"`
	} `json:"kv_transfer_params"`
	Messages []struct {
		Content any `json:"content"`
	} `json:"messages"`
	Prompt any `json:"prompt"`
}

func (r *request) promptTokens() int {
	n := len(strings.Fields(fmt.Sprint(r.Prompt)))
	for _, m := range r.Messages {
		n += len(strings.Fields(fmt.Sprint(m.Content)))
	}
	return n
}

func (e *Engine) complete(chat bool) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if code := int(e.failStatus.Load()); code != 0 {
			httpError(w, code, "injected failure")
			return
		}
		if e.cfg.APIKey != "" && r.Header.Get("Authorization") != "Bearer "+e.cfg.APIKey {
			httpError(w, http.StatusUnauthorized, "invalid API key")
			return
		}
		raw, _ := io.ReadAll(r.Body)
		var req request
		if err := json.Unmarshal(raw, &req); err != nil {
			httpError(w, http.StatusBadRequest, "invalid JSON")
			return
		}
		var generic map[string]any
		json.NewDecoder(bytes.NewReader(raw)).Decode(&generic)
		e.mu.Lock()
		e.lastHeader, e.lastBody = r.Header.Clone(), generic
		e.mu.Unlock()
		if req.Model != e.cfg.Model {
			httpError(w, http.StatusNotFound, fmt.Sprintf("The model `%s` does not exist.", req.Model))
			return
		}
		n := req.MaxTokens
		if n <= 0 {
			n = e.cfg.Tokens
		}
		e.served.Add(1)
		e.running.Add(1)
		defer e.running.Add(-1)

		ctx := r.Context()
		if !sleep(ctx, e.cfg.TTFT) {
			return
		}
		if p := req.KVTransferParams; p != nil && p.DoRemoteDecode {
			// Prefill for a remote decoder, as vLLM's NixlConnector does: keep
			// the KV and say where to fetch it.
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]any{"id": "fake-prefill", "object": "chat.completion", "model": e.cfg.Model,
				"choices": []any{map[string]any{"index": 0, "message": map[string]string{"role": "assistant", "content": "tok"}}},
				"usage":   map[string]int{"prompt_tokens": req.promptTokens(), "completion_tokens": 1},
				"kv_transfer_params": map[string]any{"do_remote_prefill": true, "do_remote_decode": false,
					"remote_engine_id": "fake-" + e.cfg.Model, "remote_request_id": "r1",
					"remote_block_ids": []int{1, 2, 3}, "remote_host": "127.0.0.1", "remote_port": 5600}})
			return
		}
		u := map[string]int{"prompt_tokens": req.promptTokens(), "completion_tokens": n, "total_tokens": req.promptTokens() + n}
		if !req.Stream {
			for i := 1; i < n; i++ {
				if !sleep(ctx, e.cfg.TPOT) {
					return
				}
			}
			text := strings.TrimSpace(strings.Repeat("tok ", n))
			var choice map[string]any
			if chat {
				choice = map[string]any{"index": 0, "message": map[string]string{"role": "assistant", "content": text}, "finish_reason": "length"}
			} else {
				choice = map[string]any{"index": 0, "text": text, "finish_reason": "length"}
			}
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]any{"id": "fake", "object": "chat.completion", "model": e.cfg.Model,
				"choices": []any{choice}, "usage": u})
			return
		}

		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		rc := http.NewResponseController(w)
		send := func(v any) bool {
			b, _ := json.Marshal(v)
			if _, err := fmt.Fprintf(w, "data: %s\n\n", b); err != nil {
				return false
			}
			return rc.Flush() == nil
		}
		chunk := func(text string, finish any) map[string]any {
			var choice map[string]any
			if chat {
				choice = map[string]any{"index": 0, "delta": map[string]string{"content": text}, "finish_reason": finish}
			} else {
				choice = map[string]any{"index": 0, "text": text, "finish_reason": finish}
			}
			return map[string]any{"id": "fake", "object": "chat.completion.chunk", "model": e.cfg.Model, "choices": []any{choice}}
		}
		if chat && !send(map[string]any{"id": "fake", "object": "chat.completion.chunk", "model": e.cfg.Model,
			"choices": []any{map[string]any{"index": 0, "delta": map[string]string{"role": "assistant", "content": ""}}}}) {
			return
		}
		for i := 0; i < n; i++ {
			if i > 0 && !sleep(ctx, e.cfg.TPOT) {
				return
			}
			var finish any
			if i == n-1 {
				finish = "length"
			}
			if !send(chunk(fmt.Sprintf("tok%d ", i), finish)) {
				return
			}
		}
		if req.StreamOptions.IncludeUsage && !send(map[string]any{"id": "fake", "object": "chat.completion.chunk",
			"model": e.cfg.Model, "choices": []any{}, "usage": u}) {
			return
		}
		fmt.Fprint(w, "data: [DONE]\n\n")
		rc.Flush()
	}
}

// sleep waits for d unless ctx ends first; it reports whether d elapsed.
func sleep(ctx context.Context, d time.Duration) bool {
	if d <= 0 {
		return ctx.Err() == nil
	}
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-t.C:
		return true
	case <-ctx.Done():
		return false
	}
}

func httpError(w http.ResponseWriter, code int, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	fmt.Fprintf(w, `{"error":{"message":%q,"type":"fake_error","code":%d}}`+"\n", msg, code)
}
