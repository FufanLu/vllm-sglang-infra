package backend

import (
	"math"
	"strings"
	"testing"
)

func TestParsePromText(t *testing.T) {
	in := `# HELP vllm:num_requests_running Number of requests running.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="0",model_name="Qwen/Qwen3-8B"} 3
vllm:num_requests_running{engine="1",model_name="Qwen/Qwen3-8B"} 2
vllm:num_requests_waiting{model_name="weird \"name\" {x}"} 4 1700000000000
vllm:kv_cache_usage_perc 0.25
bad line without value
vllm:nan_metric NaN
`
	got, err := parsePromText(strings.NewReader(in))
	if err != nil {
		t.Fatal(err)
	}
	if v := got["vllm:num_requests_running"]; len(v) != 2 || v[0] != 3 || v[1] != 2 {
		t.Errorf("running = %v, want [3 2]", v)
	}
	if v := got["vllm:num_requests_waiting"]; len(v) != 1 || v[0] != 4 {
		t.Errorf("waiting with escaped quotes and timestamp = %v, want [4]", v)
	}
	if v := got["vllm:kv_cache_usage_perc"]; len(v) != 1 || v[0] != 0.25 {
		t.Errorf("kv usage without labels = %v, want [0.25]", v)
	}
	if v := got["vllm:nan_metric"]; len(v) != 1 || !math.IsNaN(v[0]) {
		t.Errorf("NaN sample = %v", v)
	}
	if _, ok := got["bad"]; ok {
		t.Error("unparsable line produced a sample")
	}
}

func TestExtractSnapshot(t *testing.T) {
	samples := map[string][]float64{
		"vllm:num_requests_running": {3, 2},
		"vllm:num_requests_waiting": {1, 4},
		"vllm:gpu_cache_usage_perc": {0.2, 0.7}, // V0 name, two DP engine cores
	}
	s, ok := extractSnapshot(EngineVLLM, samples)
	if !ok {
		t.Fatal("no snapshot")
	}
	if s.Running != 5 || s.Waiting != 5 || s.KVUsage != 0.7 {
		t.Errorf("got %+v, want running=5 waiting=5 kv=0.7 (sum, sum, max)", s)
	}

	sg, ok := extractSnapshot(EngineSGLang, map[string][]float64{
		"sglang:num_running_reqs": {7}, "sglang:num_queue_reqs": {2}, "sglang:token_usage": {0.5},
	})
	if !ok || sg.Running != 7 || sg.Waiting != 2 || sg.KVUsage != 0.5 {
		t.Errorf("sglang snapshot = %+v ok=%v", sg, ok)
	}

	if _, ok := extractSnapshot(EngineVLLM, map[string][]float64{"other": {1}}); ok {
		t.Error("snapshot from a page with no scheduler metrics")
	}
}

func TestRegistryUpsert(t *testing.T) {
	r := NewRegistry()
	a, err := r.Upsert("a", "http://h1:8000/", EngineVLLM, map[string]string{"node": "d1"})
	if err != nil {
		t.Fatal(err)
	}
	if a.URL.String() != "http://h1:8000" {
		t.Errorf("trailing slash kept: %s", a.URL)
	}
	a.RecordProbe(true, 1)

	same, _ := r.Upsert("a", "http://h1:8000", EngineVLLM, map[string]string{"node": "d2"})
	if same != a || !same.Healthy() || same.Labels()["node"] != "d2" {
		t.Error("same URL should keep the endpoint and its health, updating labels")
	}

	moved, _ := r.Upsert("a", "http://h2:8000", EngineVLLM, nil)
	if moved == a || moved.Healthy() {
		t.Error("a new URL should replace the endpoint and start it unhealthy")
	}

	if _, err := r.Upsert("b", "ftp://x", EngineVLLM, nil); err == nil {
		t.Error("non-http URL accepted")
	}
	r.Upsert("0", "http://h0:8000", EngineSGLang, nil)
	if l := r.List(); len(l) != 2 || l[0].Name != "0" || l[1].Name != "a" {
		t.Errorf("List not sorted by name: %v", l)
	}
	if !r.Remove("a") || r.Remove("a") || r.Get("a") != nil {
		t.Error("Remove")
	}
}

func TestRecordProbeThreshold(t *testing.T) {
	ep := &Endpoint{}
	if !ep.RecordProbe(true, 2) || !ep.Healthy() {
		t.Fatal("first success should mark healthy")
	}
	if ep.RecordProbe(false, 2) || !ep.Healthy() {
		t.Error("one failure below threshold should not mark unhealthy")
	}
	if !ep.RecordProbe(false, 2) || ep.Healthy() {
		t.Error("second consecutive failure should mark unhealthy")
	}
	ep.RecordProbe(true, 2)
	if !ep.MarkDown() || ep.Healthy() {
		t.Error("MarkDown should take a healthy endpoint out at once")
	}
}

func TestParseEngine(t *testing.T) {
	for in, want := range map[string]Engine{"": EngineVLLM, "vLLM": EngineVLLM, " sglang ": EngineSGLang} {
		if got, err := ParseEngine(in); err != nil || got != want {
			t.Errorf("ParseEngine(%q) = %q, %v", in, got, err)
		}
	}
	if _, err := ParseEngine("trtllm"); err == nil {
		t.Error("unknown engine accepted")
	}
}
