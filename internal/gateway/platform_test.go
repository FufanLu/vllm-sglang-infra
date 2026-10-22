package gateway

import (
	"bufio"
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/FufanLu/vllm-sglang-infra/internal/backend"
	"github.com/FufanLu/vllm-sglang-infra/internal/fakeengine"
	"github.com/FufanLu/vllm-sglang-infra/internal/flow"
	"github.com/FufanLu/vllm-sglang-infra/internal/metrics"
	"github.com/FufanLu/vllm-sglang-infra/internal/quota"
	"github.com/FufanLu/vllm-sglang-infra/internal/scheduler"
	"github.com/FufanLu/vllm-sglang-infra/internal/tenancy"
	"github.com/FufanLu/vllm-sglang-infra/internal/usage"
)

type engineSpec struct {
	name   string
	cfg    fakeengine.Config
	labels map[string]string
	ext    *backend.External // set for an external provider
}

type memSink struct {
	mu   sync.Mutex
	recs []usage.Record
}

func (m *memSink) Write(r usage.Record) { m.mu.Lock(); m.recs = append(m.recs, r); m.mu.Unlock() }
func (m *memSink) Close() error         { return nil }
func (m *memSink) all() []usage.Record {
	m.mu.Lock()
	defer m.mu.Unlock()
	return append([]usage.Record(nil), m.recs...)
}

// platform starts fake engines for specs and a gateway whose Options the
// caller may adjust.
func platform(t *testing.T, policy string, specs []engineSpec, mutate func(*Options)) *harness {
	t.Helper()
	h := &harness{t: t, reg: backend.NewRegistry(), engines: map[string]*fakeengine.Engine{}, servers: map[string]*httptest.Server{}}
	for _, sp := range specs {
		e := fakeengine.New(sp.cfg)
		srv := httptest.NewServer(e.Handler())
		t.Cleanup(srv.Close)
		h.engines[sp.name], h.servers[sp.name] = e, srv
		spec := backend.Spec{Name: sp.name, URL: srv.URL, Engine: backend.EngineVLLM, Labels: sp.labels}
		if sp.ext != nil {
			spec.URL, spec.Kind, spec.External = srv.URL+"/v1", backend.KindExternal, sp.ext
		}
		if _, err := h.reg.UpsertSpec(spec); err != nil {
			t.Fatal(err)
		}
	}
	log := slog.New(slog.NewTextHandler(io.Discard, nil))
	h.mon = &backend.Monitor{Registry: h.reg, Client: http.DefaultClient, HealthInterval: time.Hour,
		ScrapeInterval: time.Hour, Timeout: 2 * time.Second, FailureThreshold: 1, Log: log}
	h.mon.ProbeAll(context.Background())
	sched, err := scheduler.New(policy, scheduler.Options{Saturation: 1})
	if err != nil {
		t.Fatal(err)
	}
	o := Options{Registry: h.reg, Scheduler: sched, SchedulerOptions: scheduler.Options{Saturation: 1},
		Metrics: metrics.New(h.reg, nil), Client: &http.Client{}, Log: log, MaxRetries: 2, MaxBodyBytes: 1 << 20,
		AdminToken: "tok"}
	if mutate != nil {
		mutate(&o)
	}
	h.gw = httptest.NewServer(New(o).Handler())
	t.Cleanup(h.gw.Close)
	return h
}

func (h *harness) do(key, body string, header map[string]string) *http.Response {
	h.t.Helper()
	req, _ := http.NewRequest(http.MethodPost, h.gw.URL+"/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	if key != "" {
		req.Header.Set("Authorization", "Bearer "+key)
	}
	for k, v := range header {
		req.Header.Set(k, v)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		h.t.Fatal(err)
	}
	return resp
}

func drain(resp *http.Response) string {
	b, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	return string(b)
}

const hello = `{"model":"m","messages":[{"role":"user","content":"hello there"}]}`

func tenants(t *testing.T, ts ...tenancy.Tenant) *tenancy.Directory {
	t.Helper()
	hashes := map[string][]string{}
	for _, tn := range ts {
		hashes[tn.Name] = []string{tenancy.HashKey("key-" + tn.Name)}
	}
	d, err := tenancy.New(ts, hashes)
	if err != nil {
		t.Fatal(err)
	}
	return d
}

func TestTenantAuthAndFairnessHeader(t *testing.T) {
	h := platform(t, "least-load", []engineSpec{{name: "a", cfg: fast("m")}}, func(o *Options) {
		o.Tenants = tenants(t, tenancy.Tenant{Name: "support"})
	})
	if r := h.do("", hello, nil); drain(r) == "" || r.StatusCode != http.StatusUnauthorized {
		t.Errorf("no key: status %d", r.StatusCode)
	}
	if r := h.do("wrong", hello, nil); drain(r) == "" || r.StatusCode != http.StatusUnauthorized {
		t.Errorf("wrong key: status %d", r.StatusCode)
	}
	r := h.do("key-support", hello, map[string]string{FairnessHeader: "someone-else"})
	drain(r)
	if r.StatusCode != http.StatusOK {
		t.Fatalf("valid key: status %d", r.StatusCode)
	}
	hdr, _ := h.engines["a"].LastRequest()
	if got := hdr.Get(FairnessHeader); got != "support" {
		t.Errorf("fairness header upstream = %q, want the authenticated tenant, not the client's claim", got)
	}
	if hdr.Get("Authorization") != "" {
		t.Error("the client's API key was forwarded to the engine")
	}
}

func TestQuotaReservesAndRefunds(t *testing.T) {
	sink := &memSink{}
	h := platform(t, "least-load", []engineSpec{{name: "a", cfg: fast("m")}}, func(o *Options) {
		o.Tenants = tenants(t, tenancy.Tenant{Name: "small", TokensPerMinute: 600})
		o.Quota = quota.NewMemory()
		o.Usage = sink
		o.DefaultMaxTokens = 200 // each request reserves ~200; the engine uses ~11
	})
	// Without refunds only three reservations of ~200 fit in 600 tokens.
	for i := range 6 {
		if r := h.do("key-small", hello, nil); drain(r) == "" || r.StatusCode != http.StatusOK {
			t.Fatalf("request %d: status %d; unused reservations should be refunded", i, r.StatusCode)
		}
	}
	r := h.do("key-small", `{"model":"m","max_tokens":5000,"messages":[{"role":"user","content":"hi"}]}`, nil)
	drain(r)
	if r.StatusCode != http.StatusTooManyRequests || r.Header.Get("Retry-After") == "" {
		t.Errorf("over budget: status %d, Retry-After %q", r.StatusCode, r.Header.Get("Retry-After"))
	}
	recs := sink.all()
	if len(recs) != 7 || recs[0].Tenant != "small" || recs[0].CompletionTokens != 8 || recs[6].Status != "quota_exceeded" {
		t.Errorf("usage records = %+v", recs)
	}
}

func TestStreamUsageInjectedAndHidden(t *testing.T) {
	sink := &memSink{}
	h := platform(t, "least-load", []engineSpec{{name: "a", cfg: fast("m")}}, func(o *Options) { o.Usage = sink })
	r := h.do("", `{"model":"m","stream":true,"messages":[{"role":"user","content":"hi"}]}`, nil)
	sc := bufio.NewScanner(r.Body)
	var events int
	for sc.Scan() {
		data, ok := strings.CutPrefix(sc.Text(), "data: ")
		if !ok || data == "[DONE]" {
			continue
		}
		events++
		var c streamChunk
		json.Unmarshal([]byte(data), &c)
		if c.Usage != nil {
			t.Error("client received the usage chunk it did not ask for")
		}
	}
	r.Body.Close()
	_, body := h.engines["a"].LastRequest()
	if so, _ := body["stream_options"].(map[string]any); so["include_usage"] != true {
		t.Errorf("upstream stream_options = %v, want include_usage injected", body["stream_options"])
	}
	recs := sink.all()
	if events != 9 || len(recs) != 1 || recs[0].CompletionTokens != 8 || recs[0].PromptTokens == 0 {
		t.Errorf("events %d, records %+v; want 9 client events and exact usage recorded", events, recs)
	}
}

// overflowPlatform is a pool of one slow local slot plus an external provider.
func overflowPlatform(t *testing.T, tenant tenancy.Tenant, sink *memSink) *harness {
	slow := fakeengine.Config{Model: "m", TTFT: 300 * time.Millisecond, Tokens: 1}
	cloud := fakeengine.Config{Model: "provider-model", Tokens: 4, APIKey: "sk-cloud"}
	ext := &backend.External{APIKey: "sk-cloud", Models: map[string]string{"m": "provider-model"},
		PriceInput: 1, PriceOutput: 2, MaxInflight: 10}
	var h *harness
	h = platform(t, "least-load", []engineSpec{{name: "local", cfg: slow}, {name: "cloud", cfg: cloud, ext: ext}},
		func(o *Options) {
			o.Tenants = tenants(t, tenant)
			o.Usage = sink
			o.Flow = flow.New(flow.Config{
				Capacity:     func() int { return 1 },
				BatchShare:   1,
				MaxQueued:    [2]int{10, 10},
				QueueTimeout: [2]time.Duration{5 * time.Second, 5 * time.Second},
			})
		})
	return h
}

// twoConcurrent sends two requests, the second while the first holds the
// only local slot, and returns the backend that served the second.
func twoConcurrent(t *testing.T, h *harness, key string) string {
	first := make(chan struct{})
	go func() {
		r := h.do(key, hello, nil)
		drain(r)
		close(first)
	}()
	time.Sleep(100 * time.Millisecond) // the first request is now in flight
	r := h.do(key, hello, nil)
	drain(r)
	<-first
	if r.StatusCode != http.StatusOK {
		t.Fatalf("second request: status %d", r.StatusCode)
	}
	return r.Header.Get("X-VSI-Backend")
}

func TestOverflowToExternalProvider(t *testing.T) {
	sink := &memSink{}
	h := overflowPlatform(t, tenancy.Tenant{Name: "web"}, sink)
	if got := twoConcurrent(t, h, "key-web"); got != "cloud" {
		t.Fatalf("second request served by %q, want overflow to cloud", got)
	}
	hdr, body := h.engines["cloud"].LastRequest()
	if body["model"] != "provider-model" || hdr.Get("Authorization") != "Bearer sk-cloud" || hdr.Get(FairnessHeader) != "" {
		t.Errorf("provider got model %v, auth %q, tenant header %q", body["model"], hdr.Get("Authorization"), hdr.Get(FairnessHeader))
	}
	var ext *usage.Record
	for _, r := range sink.all() {
		if r.External {
			ext = &r
		}
	}
	if ext == nil || ext.Backend != "cloud" || ext.CostUSD <= 0 {
		t.Errorf("overflow usage record = %+v, want external with a cost", ext)
	}
}

func TestOnPremTenantNeverOverflows(t *testing.T) {
	h := overflowPlatform(t, tenancy.Tenant{Name: "bank", OnPremOnly: true}, &memSink{})
	if got := twoConcurrent(t, h, "key-bank"); got != "local" {
		t.Errorf("on-prem-only request served by %q; it must wait for the local slot", got)
	}
	if h.engines["cloud"].Served() != 0 {
		t.Error("on-prem-only traffic reached the external provider")
	}
}

func TestFlowControlQueueFull(t *testing.T) {
	slow := fakeengine.Config{Model: "m", TTFT: 400 * time.Millisecond, Tokens: 1}
	h := platform(t, "least-load", []engineSpec{{name: "a", cfg: slow}}, func(o *Options) {
		o.Flow = flow.New(flow.Config{
			Capacity: func() int { return 1 }, BatchShare: 1,
			MaxQueued: [2]int{1, 1}, QueueTimeout: [2]time.Duration{5 * time.Second, 5 * time.Second},
		})
	})
	var wg sync.WaitGroup
	for range 2 { // one in flight, one queued
		wg.Go(func() { drain(h.do("", hello, nil)) })
		time.Sleep(50 * time.Millisecond)
	}
	r := h.do("", hello, nil)
	drain(r)
	wg.Wait()
	if r.StatusCode != http.StatusTooManyRequests {
		t.Errorf("third request: status %d, want 429 with the queue full", r.StatusCode)
	}
}

func TestPrefillDecodeDisaggregation(t *testing.T) {
	h := platform(t, "least-load", []engineSpec{
		{name: "p", cfg: fast("m"), labels: map[string]string{"role": "prefill"}},
		{name: "d", cfg: fast("m"), labels: map[string]string{"role": "decode"}},
	}, func(o *Options) { o.PD = true })
	r := h.do("", `{"model":"m","stream":true,"max_tokens":5,"messages":[{"role":"user","content":"hi"}]}`, nil)
	out := drain(r)
	if r.Header.Get("X-VSI-Backend") != "d" || !strings.Contains(out, "[DONE]") {
		t.Fatalf("served by %q, body %q; want the decode endpoint's stream", r.Header.Get("X-VSI-Backend"), out)
	}
	_, pre := h.engines["p"].LastRequest()
	if pre["max_tokens"] != 1.0 || pre["stream"] != false {
		t.Errorf("prefill request = %v, want max_tokens 1 and not streamed", pre)
	}
	if kv, _ := pre["kv_transfer_params"].(map[string]any); kv["do_remote_decode"] != true {
		t.Errorf("prefill kv_transfer_params = %v", pre["kv_transfer_params"])
	}
	_, dec := h.engines["d"].LastRequest()
	if kv, _ := dec["kv_transfer_params"].(map[string]any); kv["remote_engine_id"] != "fake-m" || kv["do_remote_prefill"] != true {
		t.Errorf("decode kv_transfer_params = %v, want the prefill's response passed through", dec["kv_transfer_params"])
	}
	if dec["max_tokens"] != 5.0 {
		t.Errorf("decode request lost the client's max_tokens: %v", dec["max_tokens"])
	}

	h.servers["p"].Close() // prefill down: the decode endpoint prefills itself
	r = h.do("", `{"model":"m","messages":[{"role":"user","content":"hi"}]}`, nil)
	drain(r)
	_, dec = h.engines["d"].LastRequest()
	if r.StatusCode != http.StatusOK || dec["kv_transfer_params"] != nil {
		t.Errorf("fallback: status %d, decode kv params %v", r.StatusCode, dec["kv_transfer_params"])
	}
}

func TestAdminPolicySwitchAndCacheReset(t *testing.T) {
	h := platform(t, "least-load", []engineSpec{{name: "a", cfg: fast("m")}, {name: "b", cfg: fast("m")}}, nil)
	admin := func(method, path, body string) (*http.Response, string) {
		req, _ := http.NewRequest(method, h.gw.URL+path, strings.NewReader(body))
		req.Header.Set("Authorization", "Bearer tok")
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		return resp, drain(resp)
	}
	if r, body := admin("PUT", "/admin/policy", `{"policy":"prefix-aware","weights":{"prefix":3}}`); r.StatusCode != 200 ||
		!strings.Contains(body, `"policy":"prefix-aware"`) || !strings.Contains(body, `"prefix":3`) {
		t.Errorf("switch: %d %s", r.StatusCode, body)
	}
	if r, _ := admin("PUT", "/admin/policy", `{"policy":"nope"}`); r.StatusCode != http.StatusBadRequest {
		t.Errorf("unknown policy accepted: %d", r.StatusCode)
	}
	if r, body := admin("POST", "/admin/reset_prefix_cache", ""); r.StatusCode != 200 || !strings.Contains(body, `"a":"ok"`) {
		t.Errorf("reset: %d %s", r.StatusCode, body)
	}
	if h.engines["a"].Resets() != 1 || h.engines["b"].Resets() != 1 {
		t.Error("reset did not reach every engine")
	}
}
