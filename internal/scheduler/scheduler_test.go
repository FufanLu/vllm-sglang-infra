package scheduler

import (
	"errors"
	"fmt"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/FufanLu/vllm-sglang-infra/internal/backend"
)

// endpoints builds n healthy endpoints named e0..e{n-1}.
func endpoints(t *testing.T, n int) []*backend.Endpoint {
	t.Helper()
	r := backend.NewRegistry()
	for i := range n {
		ep, err := r.Upsert(fmt.Sprintf("e%d", i), fmt.Sprintf("http://h%d:8000", i), backend.EngineVLLM, nil)
		if err != nil {
			t.Fatal(err)
		}
		ep.RecordProbe(true, 1)
	}
	return r.List()
}

func mustNew(t *testing.T, policy string) *Scheduler {
	t.Helper()
	s, err := New(policy, Options{})
	if err != nil {
		t.Fatal(err)
	}
	return s
}

func TestRoundRobinCycles(t *testing.T) {
	eps := endpoints(t, 3)
	s := mustNew(t, "round-robin")
	var got []string
	for range 6 {
		ep, err := s.Schedule(&Request{}, eps, nil)
		if err != nil {
			t.Fatal(err)
		}
		ep.Release()
		got = append(got, ep.Name)
	}
	if fmt.Sprint(got) != "[e0 e1 e2 e0 e1 e2]" {
		t.Errorf("order = %v", got)
	}
}

func TestLeastLoadUsesInflight(t *testing.T) {
	eps := endpoints(t, 3)
	s := mustNew(t, "least-load")
	// Six concurrent requests without releases must spread two per endpoint:
	// each pick sees the in-flight slots taken by the earlier ones.
	count := map[string]int{}
	for range 6 {
		ep, err := s.Schedule(&Request{}, eps, nil)
		if err != nil {
			t.Fatal(err)
		}
		count[ep.Name]++
	}
	for _, ep := range eps {
		if count[ep.Name] != 2 || ep.Inflight() != 2 {
			t.Errorf("%s got %d picks, inflight %d; want 2", ep.Name, count[ep.Name], ep.Inflight())
		}
	}
}

func TestLeastLoadUsesEngineQueue(t *testing.T) {
	eps := endpoints(t, 2)
	now := time.Now()
	eps[0].SetSnapshot(backend.Snapshot{Running: 8, Waiting: 4, KVUsage: 0.9, ScrapedAt: now})
	eps[1].SetSnapshot(backend.Snapshot{Running: 1, KVUsage: 0.1, ScrapedAt: now})
	s := mustNew(t, "least-load")
	for range 5 {
		ep, _ := s.Schedule(&Request{}, eps, nil)
		ep.Release()
		if ep.Name != "e1" {
			t.Fatalf("picked %s, want the endpoint with the shorter engine queue", ep.Name)
		}
	}
}

func TestKVUsageBreaksTies(t *testing.T) {
	eps := endpoints(t, 2)
	now := time.Now()
	eps[0].SetSnapshot(backend.Snapshot{KVUsage: 0.95, ScrapedAt: now})
	eps[1].SetSnapshot(backend.Snapshot{KVUsage: 0.10, ScrapedAt: now})
	ep, _ := mustNew(t, "least-load").Schedule(&Request{}, eps, nil)
	if ep.Name != "e1" {
		t.Errorf("picked %s, want the endpoint with free KV cache", ep.Name)
	}
}

func TestStaleSnapshotIsNeutral(t *testing.T) {
	eps := endpoints(t, 1)
	eps[0].SetSnapshot(backend.Snapshot{KVUsage: 0.0, ScrapedAt: time.Now().Add(-time.Hour)})
	if got := (KVUsageScorer{MaxAge: time.Second}).Score(&Request{}, eps); got[0] != 0.5 {
		t.Errorf("stale snapshot scored %v, want 0.5", got[0])
	}
}

func TestKVUsageIgnoresSmallDifferences(t *testing.T) {
	eps := endpoints(t, 3)
	now := time.Now()
	for i, u := range []float64{0.01, 0.30, 0.85} {
		eps[i].SetSnapshot(backend.Snapshot{KVUsage: u, ScrapedAt: now})
	}
	got := (KVUsageScorer{MaxAge: time.Second, Threshold: 0.7}).Score(&Request{}, eps)
	if got[0] != 1 || got[1] != 1 || got[2] > 0.51 || got[2] < 0.49 {
		t.Errorf("scores %v, want [1 1 0.5]: only usage above the threshold should count", got)
	}
}

func TestQueueLoadCountsOnlyExternalEngineLoad(t *testing.T) {
	eps := endpoints(t, 2)
	now := time.Now()
	// e0: the engine reports 4 running, all of them the gateway's own: no
	// external load. e1: reports 4 running while the gateway had 1 in
	// flight: 3 requests came from elsewhere.
	for range 4 {
		eps[0].Acquire()
	}
	eps[1].Acquire()
	eps[0].SetSnapshot(backend.Snapshot{Running: 4, GatewayInflight: 4, ScrapedAt: now})
	eps[1].SetSnapshot(backend.Snapshot{Running: 4, GatewayInflight: 1, ScrapedAt: now})
	got := (QueueScorer{Saturation: 8, MaxAge: time.Second}).Score(&Request{}, eps)
	if got[0] != 0.5 || got[1] != 0.5 {
		t.Errorf("scores %v, want [0.5 0.5]: both carry 4 requests", got)
	}
	// A stale snapshot contributes nothing: only the exact in-flight count.
	eps[1].SetSnapshot(backend.Snapshot{Running: 4, GatewayInflight: 1, ScrapedAt: now.Add(-time.Hour)})
	if got := (QueueScorer{Saturation: 8, MaxAge: time.Second}).Score(&Request{}, eps); got[1] != 1-1.0/8 {
		t.Errorf("stale external load counted: %v", got)
	}
}

func TestFiltersAndExclude(t *testing.T) {
	eps := endpoints(t, 3)
	eps[0].MarkDown()
	eps[1].SetModels([]string{"qwen3-8b"})
	eps[2].SetModels([]string{"llama-3.1-8b"})
	s := mustNew(t, "least-load")

	ep, err := s.Schedule(&Request{Model: "qwen3-8b"}, eps, nil)
	if err != nil || ep.Name != "e1" {
		t.Fatalf("got %v, %v; want e1 (only healthy endpoint serving the model)", ep, err)
	}
	ep.Release()

	if _, err := s.Schedule(&Request{Model: "qwen3-8b"}, eps, map[string]bool{"e1": true}); !errors.Is(err, ErrNoEndpoint) {
		t.Errorf("excluding the only candidate: err = %v, want ErrNoEndpoint", err)
	}
	if _, err := s.Schedule(&Request{Model: "missing"}, eps, nil); !errors.Is(err, ErrNoEndpoint) {
		t.Errorf("unknown model: err = %v, want ErrNoEndpoint", err)
	}
	if _, err := s.Schedule(&Request{}, nil, nil); !errors.Is(err, ErrNoEndpoint) {
		t.Errorf("no endpoints: err = %v", err)
	}
}

func TestNewValidates(t *testing.T) {
	if _, err := New("magic", Options{}); err == nil {
		t.Error("unknown policy accepted")
	}
	if _, err := New("least-load", Options{Weights: map[string]float64{"prefix": 1}}); err == nil {
		t.Error("weight for a scorer the policy lacks accepted")
	}
	s, err := New("least-load", Options{Weights: map[string]float64{"kv-usage": 0}})
	if err != nil {
		t.Fatal(err)
	}
	for _, ws := range s.Scorers {
		if ws.Scorer.Name() == "kv-usage" && ws.Weight != 0 {
			t.Error("weight override ignored")
		}
	}
}

func TestPrefixAffinity(t *testing.T) {
	eps := endpoints(t, 4)
	s, err := New("prefix-aware", Options{PrefixBlockChars: 16})
	if err != nil {
		t.Fatal(err)
	}
	sys := strings.Repeat("You are a support bot. ", 20)
	first, _ := s.Schedule(&Request{Prompt: sys + "question 0"}, eps, nil)
	first.Release()
	for i := 1; i < 20; i++ {
		ep, _ := s.Schedule(&Request{Prompt: sys + fmt.Sprintf("question %d", i)}, eps, nil)
		ep.Release()
		if ep != first {
			t.Fatalf("request %d went to %s, want %s which holds the shared prefix", i, ep.Name, first.Name)
		}
	}
}

func TestPrefixAffinityYieldsUnderLoad(t *testing.T) {
	eps := endpoints(t, 2)
	// Weights prefix 1, queue 2, saturation 8: the endpoint holding the
	// prefix wins while it is less than 8*1/2 = 4 requests busier.
	s, _ := New("prefix-aware", Options{PrefixBlockChars: 16, QueueSaturation: 8})
	sys := strings.Repeat("You are a support bot. ", 20)
	home, _ := s.Schedule(&Request{Prompt: sys}, eps, nil) // requests are held, never released
	var picks []*backend.Endpoint
	for range 8 {
		ep, _ := s.Schedule(&Request{Prompt: sys + "x"}, eps, nil)
		picks = append(picks, ep)
	}
	for i, ep := range picks[:3] {
		if ep != home {
			t.Fatalf("pick %d left the cached endpoint at load %d, before the 4-request margin", i, i+1)
		}
	}
	if !slices.ContainsFunc(picks, func(e *backend.Endpoint) bool { return e != home }) {
		t.Error("affinity never yielded although the cached endpoint was saturated")
	}
	// Once the other endpoint served the prefix it holds it too, so load
	// alone balances the rest.
	if d := home.Inflight() - picks[len(picks)-1].Inflight(); d > 4 {
		t.Errorf("load gap %d after the prefix spread to both endpoints", d)
	}
}

func TestShortPromptsFallBackToLoad(t *testing.T) {
	eps := endpoints(t, 2)
	s, _ := New("prefix-aware", Options{PrefixBlockChars: 64})
	a, _ := s.Schedule(&Request{Prompt: "hi"}, eps, nil) // no full block: prefix scores are all 0
	b, _ := s.Schedule(&Request{Prompt: "hi"}, eps, nil)
	if a == b {
		t.Error("without a cacheable prefix, concurrent requests should spread by load")
	}
}
