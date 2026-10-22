package gateway

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/FufanLu/vllm-sglang-infra/internal/backend"
	"github.com/FufanLu/vllm-sglang-infra/internal/fakeengine"
	"github.com/FufanLu/vllm-sglang-infra/internal/metrics"
	"github.com/FufanLu/vllm-sglang-infra/internal/scheduler"
)

type harness struct {
	t       *testing.T
	reg     *backend.Registry
	mon     *backend.Monitor
	gw      *httptest.Server
	engines map[string]*fakeengine.Engine
	servers map[string]*httptest.Server
}

// newHarness starts one fake engine per config, registers them, runs one
// monitor round so they are healthy with known models, and starts a gateway.
func newHarness(t *testing.T, policy string, engines map[string]fakeengine.Config) *harness {
	t.Helper()
	h := &harness{t: t, reg: backend.NewRegistry(), engines: map[string]*fakeengine.Engine{}, servers: map[string]*httptest.Server{}}
	for name, cfg := range engines {
		e := fakeengine.New(cfg)
		srv := httptest.NewServer(e.Handler())
		t.Cleanup(srv.Close)
		h.engines[name], h.servers[name] = e, srv
		if _, err := h.reg.Upsert(name, srv.URL, backend.EngineVLLM, nil); err != nil {
			t.Fatal(err)
		}
	}
	log := slog.New(slog.NewTextHandler(io.Discard, nil))
	h.mon = &backend.Monitor{Registry: h.reg, Client: http.DefaultClient, HealthInterval: time.Hour,
		ScrapeInterval: time.Hour, Timeout: 2 * time.Second, FailureThreshold: 1, Log: log}
	h.mon.ProbeAll(context.Background())
	h.mon.ScrapeAll(context.Background())

	sched, err := scheduler.New(policy, scheduler.Options{})
	if err != nil {
		t.Fatal(err)
	}
	s := New(Options{Registry: h.reg, Scheduler: sched, Metrics: metrics.New(h.reg, nil), Client: &http.Client{},
		Log: log, MaxRetries: 2, MaxBodyBytes: 1 << 20, AdminToken: "tok"})
	h.gw = httptest.NewServer(s.Handler())
	t.Cleanup(h.gw.Close)
	return h
}

func (h *harness) post(path, body string) *http.Response {
	h.t.Helper()
	resp, err := http.Post(h.gw.URL+path, "application/json", strings.NewReader(body))
	if err != nil {
		h.t.Fatal(err)
	}
	return resp
}

func fast(model string) fakeengine.Config {
	return fakeengine.Config{Model: model, TTFT: 5 * time.Millisecond, TPOT: time.Millisecond, Tokens: 8}
}

func TestStreamingRelay(t *testing.T) {
	h := newHarness(t, "least-load", map[string]fakeengine.Config{"a": fast("m")})
	resp := h.post("/v1/chat/completions",
		`{"model":"m","stream":true,"stream_options":{"include_usage":true},"messages":[{"role":"user","content":"hi"}]}`)
	defer resp.Body.Close()
	if resp.StatusCode != 200 || !strings.HasPrefix(resp.Header.Get("Content-Type"), "text/event-stream") {
		t.Fatalf("status %d, content-type %q", resp.StatusCode, resp.Header.Get("Content-Type"))
	}
	if resp.Header.Get("X-VSI-Backend") != "a" || resp.Header.Get("X-Request-Id") == "" {
		t.Errorf("missing gateway headers: %v", resp.Header)
	}
	var content, usage int
	var done bool
	sc := bufio.NewScanner(resp.Body)
	for sc.Scan() {
		data, ok := strings.CutPrefix(sc.Text(), "data: ")
		if !ok {
			continue
		}
		if data == "[DONE]" {
			done = true
			continue
		}
		var c streamChunk
		if err := json.Unmarshal([]byte(data), &c); err != nil {
			t.Fatalf("bad chunk %q: %v", data, err)
		}
		if len(c.Choices) > 0 && c.Choices[0].Delta.Content != "" {
			content++
		}
		if c.Usage != nil {
			usage = c.Usage.CompletionTokens
		}
	}
	if content != 8 || usage != 8 || !done {
		t.Errorf("content events %d, usage %d, done %v; want 8, 8, true", content, usage, done)
	}
}

func TestNonStreamingRelay(t *testing.T) {
	h := newHarness(t, "least-load", map[string]fakeengine.Config{"a": fast("m")})
	resp := h.post("/v1/completions", `{"model":"m","prompt":"once upon a time","max_tokens":3}`)
	defer resp.Body.Close()
	var body struct {
		Choices []struct{ Text string } `json:"choices"`
		Usage   usageInfo               `json:"usage"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil || resp.StatusCode != 200 {
		t.Fatalf("status %d, decode err %v", resp.StatusCode, err)
	}
	if body.Choices[0].Text != "tok tok tok" || body.Usage.CompletionTokens != 3 || body.Usage.PromptTokens != 4 {
		t.Errorf("body = %+v", body)
	}
}

func TestRoundRobinSpreadsRequests(t *testing.T) {
	h := newHarness(t, "round-robin", map[string]fakeengine.Config{"a": fast("m"), "b": fast("m")})
	seen := map[string]int{}
	for range 4 {
		resp := h.post("/v1/chat/completions", `{"model":"m","messages":[{"role":"user","content":"hi"}]}`)
		io.Copy(io.Discard, resp.Body)
		resp.Body.Close()
		seen[resp.Header.Get("X-VSI-Backend")]++
	}
	if seen["a"] != 2 || seen["b"] != 2 {
		t.Errorf("distribution = %v, want 2/2", seen)
	}
}

func TestLeastLoadSpreadsConcurrentRequests(t *testing.T) {
	slow := fakeengine.Config{Model: "m", TTFT: 200 * time.Millisecond, Tokens: 1}
	h := newHarness(t, "least-load", map[string]fakeengine.Config{"a": slow, "b": slow})
	errs := make(chan error, 2)
	for range 2 {
		go func() {
			resp, err := http.Post(h.gw.URL+"/v1/chat/completions", "application/json",
				strings.NewReader(`{"model":"m","messages":[{"role":"user","content":"hi"}]}`))
			if err == nil {
				io.Copy(io.Discard, resp.Body)
				resp.Body.Close()
			}
			errs <- err
		}()
	}
	for range 2 {
		if err := <-errs; err != nil {
			t.Fatal(err)
		}
	}
	if a, b := h.engines["a"].Served(), h.engines["b"].Served(); a != 1 || b != 1 {
		t.Errorf("served a=%d b=%d; concurrent requests should go to different endpoints", a, b)
	}
}

func TestFailoverWhenBackendDies(t *testing.T) {
	h := newHarness(t, "round-robin", map[string]fakeengine.Config{"a": fast("m"), "b": fast("m")})
	h.servers["a"].Close() // still marked healthy: the monitor has not noticed yet
	for range 2 {
		resp := h.post("/v1/chat/completions", `{"model":"m","messages":[{"role":"user","content":"hi"}]}`)
		io.Copy(io.Discard, resp.Body)
		resp.Body.Close()
		if resp.StatusCode != 200 || resp.Header.Get("X-VSI-Backend") != "b" {
			t.Fatalf("status %d from %q, want 200 from b", resp.StatusCode, resp.Header.Get("X-VSI-Backend"))
		}
	}
	if h.reg.Get("a").Healthy() {
		t.Error("dead backend still healthy after a refused connection")
	}
}

func TestRetryOnServiceUnavailable(t *testing.T) {
	h := newHarness(t, "round-robin", map[string]fakeengine.Config{"a": fast("m"), "b": fast("m")})
	h.engines["a"].SetFailStatus(http.StatusServiceUnavailable)
	for range 2 {
		resp := h.post("/v1/chat/completions", `{"model":"m","messages":[{"role":"user","content":"hi"}]}`)
		io.Copy(io.Discard, resp.Body)
		resp.Body.Close()
		if resp.StatusCode != 200 || resp.Header.Get("X-VSI-Backend") != "b" {
			t.Fatalf("status %d from %q, want 200 from b", resp.StatusCode, resp.Header.Get("X-VSI-Backend"))
		}
	}
}

func TestClientErrorsAreNotRetried(t *testing.T) {
	h := newHarness(t, "round-robin", map[string]fakeengine.Config{"a": fast("m"), "b": fast("m")})
	h.engines["a"].SetFailStatus(http.StatusBadRequest)
	h.engines["b"].SetFailStatus(http.StatusBadRequest)
	resp := h.post("/v1/chat/completions", `{"model":"m","messages":[]}`)
	resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("status %d, want the backend's 400 passed through", resp.StatusCode)
	}
	if h.engines["a"].Served()+h.engines["b"].Served() != 0 {
		t.Error("failed request counted as served")
	}
}

func TestModelRouting(t *testing.T) {
	h := newHarness(t, "least-load", map[string]fakeengine.Config{"a": fast("qwen"), "b": fast("llama")})
	resp := h.post("/v1/chat/completions", `{"model":"llama","messages":[{"role":"user","content":"hi"}]}`)
	resp.Body.Close()
	if resp.Header.Get("X-VSI-Backend") != "b" {
		t.Errorf("routed to %q, want b (the only backend serving llama)", resp.Header.Get("X-VSI-Backend"))
	}
	resp = h.post("/v1/chat/completions", `{"model":"gpt-9","messages":[]}`)
	resp.Body.Close()
	if resp.StatusCode != http.StatusServiceUnavailable {
		t.Errorf("unknown model: status %d, want 503", resp.StatusCode)
	}

	mresp, err := http.Get(h.gw.URL + "/v1/models")
	if err != nil {
		t.Fatal(err)
	}
	defer mresp.Body.Close()
	var list struct {
		Data []struct{ ID string } `json:"data"`
	}
	json.NewDecoder(mresp.Body).Decode(&list)
	if len(list.Data) != 2 || list.Data[0].ID != "llama" || list.Data[1].ID != "qwen" {
		t.Errorf("/v1/models = %+v", list.Data)
	}
}

func TestClientCancelReleasesSlots(t *testing.T) {
	long := fakeengine.Config{Model: "m", TTFT: time.Millisecond, TPOT: 20 * time.Millisecond, Tokens: 500}
	h := newHarness(t, "least-load", map[string]fakeengine.Config{"a": long})
	ctx, cancel := context.WithCancel(context.Background())
	req, _ := http.NewRequestWithContext(ctx, http.MethodPost, h.gw.URL+"/v1/chat/completions",
		strings.NewReader(`{"model":"m","stream":true,"messages":[{"role":"user","content":"hi"}]}`))
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	bufio.NewReader(resp.Body).ReadString('\n') // first event arrived
	cancel()
	resp.Body.Close()

	deadline := time.Now().Add(3 * time.Second)
	for h.reg.Get("a").Inflight() != 0 || h.engines["a"].Running() != 0 {
		if time.Now().After(deadline) {
			t.Fatalf("after client cancel: gateway inflight %d, engine running %d",
				h.reg.Get("a").Inflight(), h.engines["a"].Running())
		}
		time.Sleep(10 * time.Millisecond)
	}
}

func TestNoBackends(t *testing.T) {
	h := newHarness(t, "least-load", nil)
	resp := h.post("/v1/chat/completions", `{"model":"m","messages":[]}`)
	defer resp.Body.Close()
	var e struct {
		Error struct{ Type string } `json:"error"`
	}
	json.NewDecoder(resp.Body).Decode(&e)
	if resp.StatusCode != http.StatusServiceUnavailable || e.Error.Type != "service_unavailable" {
		t.Errorf("status %d, error type %q", resp.StatusCode, e.Error.Type)
	}
	if r, _ := http.Get(h.gw.URL + "/readyz"); r.StatusCode != http.StatusServiceUnavailable {
		t.Errorf("/readyz = %d with no backends", r.StatusCode)
	}
}

func TestBadRequestBodies(t *testing.T) {
	h := newHarness(t, "least-load", map[string]fakeengine.Config{"a": fast("m")})
	if resp := h.post("/v1/chat/completions", `{not json`); resp.StatusCode != http.StatusBadRequest {
		t.Errorf("invalid JSON: status %d", resp.StatusCode)
	}
	big := fmt.Sprintf(`{"model":"m","prompt":%q}`, strings.Repeat("x", 2<<20))
	if resp := h.post("/v1/completions", big); resp.StatusCode != http.StatusRequestEntityTooLarge {
		t.Errorf("oversized body: status %d", resp.StatusCode)
	}
}

func TestAdminAPI(t *testing.T) {
	h := newHarness(t, "least-load", nil)
	extra := httptest.NewServer(fakeengine.New(fast("m")).Handler())
	defer extra.Close()

	do := func(method, path, token, body string) *http.Response {
		req, _ := http.NewRequest(method, h.gw.URL+path, strings.NewReader(body))
		if token != "" {
			req.Header.Set("Authorization", "Bearer "+token)
		}
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		io.Copy(io.Discard, resp.Body)
		resp.Body.Close()
		return resp
	}
	if r := do("PUT", "/admin/backends/x", "wrong", `{"url":"`+extra.URL+`"}`); r.StatusCode != http.StatusUnauthorized {
		t.Errorf("wrong token: %d", r.StatusCode)
	}
	if r := do("PUT", "/admin/backends/x", "tok", `{"url":"`+extra.URL+`","engine":"vllm","labels":{"slurm_job":"123"}}`); r.StatusCode != 200 {
		t.Fatalf("register: %d", r.StatusCode)
	}
	ep := h.reg.Get("x")
	if ep == nil || ep.Labels()["slurm_job"] != "123" {
		t.Fatal("registered backend missing or without labels")
	}
	h.mon.ProbeAll(context.Background())
	resp := h.post("/v1/chat/completions", `{"model":"m","messages":[{"role":"user","content":"hi"}]}`)
	resp.Body.Close()
	if resp.Header.Get("X-VSI-Backend") != "x" {
		t.Errorf("request not routed to the registered backend")
	}
	if r := do("DELETE", "/admin/backends/x", "tok", ""); r.StatusCode != http.StatusNoContent {
		t.Errorf("delete: %d", r.StatusCode)
	}
	if r := do("DELETE", "/admin/backends/x", "tok", ""); r.StatusCode != http.StatusNotFound {
		t.Errorf("second delete: %d", r.StatusCode)
	}
}

func TestPromptText(t *testing.T) {
	var chat inferenceRequest
	json.Unmarshal([]byte(`{"messages":[
		{"role":"system","content":"You are support."},
		{"role":"user","content":[{"type":"text","text":"hello "},{"type":"image_url","image_url":{"url":"x"}},{"type":"text","text":"there"}]}]}`), &chat)
	if got, want := chat.promptText(), "system\nYou are support.\nuser\nhello there\n"; got != want {
		t.Errorf("chat prompt = %q, want %q", got, want)
	}
	var comp inferenceRequest
	json.Unmarshal([]byte(`{"prompt":["a","b"]}`), &comp)
	if got := comp.promptText(); got != "a\nb" {
		t.Errorf("completion prompt list = %q", got)
	}
	var tokens inferenceRequest
	json.Unmarshal([]byte(`{"prompt":[1,2,3]}`), &tokens)
	if got := tokens.promptText(); got != "" {
		t.Errorf("token-id prompt = %q, want empty", got)
	}
}
