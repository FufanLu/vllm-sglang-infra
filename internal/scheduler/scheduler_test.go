package scheduler

import (
	"errors"
	"fmt"
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
	s, err := New(policy, nil)
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
	if _, err := New("magic", nil); err == nil {
		t.Error("unknown policy accepted")
	}
	if _, err := New("least-load", map[string]float64{"prefix": 1}); err == nil {
		t.Error("weight for a scorer the policy lacks accepted")
	}
	s, err := New("least-load", map[string]float64{"kv-usage": 0})
	if err != nil {
		t.Fatal(err)
	}
	for _, ws := range s.Scorers {
		if ws.Scorer.Name() == "kv-usage" && ws.Weight != 0 {
			t.Error("weight override ignored")
		}
	}
}
