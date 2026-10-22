package k8s

import (
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/FufanLu/vllm-sglang-infra/internal/backend"
)

// fakeAPI serves the two API calls the client makes.
func fakeAPI(t *testing.T, replicas *int) *Client {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /api/v1/namespaces/llm/pods", func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer tok" || r.URL.Query().Get("labelSelector") != "app=vllm" {
			http.Error(w, "forbidden", http.StatusForbidden)
			return
		}
		io.WriteString(w, `{"items":[
		 {"metadata":{"name":"vllm-a","labels":{"app":"vllm","role":"decode"},"annotations":{"vsi.io/kv_events":"tcp://10.0.0.5:5557"}},
		  "status":{"phase":"Running","podIP":"10.0.0.5","conditions":[{"type":"Ready","status":"True"}]}},
		 {"metadata":{"name":"vllm-b","labels":{"app":"vllm"}},
		  "status":{"phase":"Running","podIP":"10.0.0.6","conditions":[{"type":"Ready","status":"False"}]}},
		 {"metadata":{"name":"vllm-c","labels":{"app":"vllm"},"deletionTimestamp":"2026-10-21T00:00:00Z"},
		  "status":{"phase":"Running","podIP":"10.0.0.7","conditions":[{"type":"Ready","status":"True"}]}}]}`)
	})
	mux.HandleFunc("GET /apis/apps/v1/namespaces/llm/deployments/vllm/scale", func(w http.ResponseWriter, _ *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{"spec": map[string]int{"replicas": *replicas}})
	})
	mux.HandleFunc("PATCH /apis/apps/v1/namespaces/llm/deployments/vllm/scale", func(w http.ResponseWriter, r *http.Request) {
		var p struct {
			Spec struct{ Replicas int } `json:"spec"`
		}
		if r.Header.Get("Content-Type") != "application/merge-patch+json" || json.NewDecoder(r.Body).Decode(&p) != nil {
			http.Error(w, "bad patch", http.StatusBadRequest)
			return
		}
		*replicas = p.Spec.Replicas
	})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	return &Client{Base: srv.URL, Token: "tok", Namespace: "llm", HTTP: srv.Client()}
}

func TestDiscoverySyncsReadyPods(t *testing.T) {
	replicas := 3
	c := fakeAPI(t, &replicas)
	reg := backend.NewRegistry()
	reg.Upsert("static", "http://10.9.9.9:8000", backend.EngineVLLM, nil) // from config: never touched
	reg.Upsert("vllm-old", "http://10.0.0.1:8000", backend.EngineVLLM, map[string]string{SourceLabel: "kubernetes"})
	var added []string
	d := &Discovery{Client: c, Selector: "app=vllm", Port: 8000, Engine: backend.EngineVLLM, Registry: reg,
		OnAdd: func(ep *backend.Endpoint) { added = append(added, ep.Name) }, Log: slog.New(slog.NewTextHandler(io.Discard, nil))}
	if err := d.Sync(context.Background()); err != nil {
		t.Fatal(err)
	}
	var names []string
	for _, ep := range reg.List() {
		names = append(names, ep.Name)
	}
	if strings.Join(names, ",") != "static,vllm-a" || strings.Join(added, ",") != "vllm-a" {
		t.Fatalf("endpoints %v, added %v: want the static one kept, only the ready pod added, the vanished pod removed", names, added)
	}
	a := reg.Get("vllm-a")
	if a.URL.String() != "http://10.0.0.5:8000" || a.Label("role") != "decode" || a.Label("kv_events") != "tcp://10.0.0.5:5557" {
		t.Errorf("vllm-a = %s labels %v", a.URL, a.Labels())
	}
}

func TestScale(t *testing.T) {
	replicas := 2
	c := fakeAPI(t, &replicas)
	ctx := context.Background()
	if n, err := c.Scale(ctx, "vllm"); n != 2 || err != nil {
		t.Fatalf("scale = %d, %v", n, err)
	}
	if err := c.SetScale(ctx, "vllm", 5); err != nil || replicas != 5 {
		t.Errorf("set scale: replicas %d, err %v", replicas, err)
	}
}
