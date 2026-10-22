// Command fakeengine runs a fake OpenAI-compatible model server, for trying
// the gateway without a GPU.
package main

import (
	"flag"
	"log"
	"net/http"
	"time"

	"github.com/FufanLu/vllm-sglang-infra/internal/fakeengine"
)

func main() {
	addr := flag.String("addr", ":8000", "listen address")
	model := flag.String("model", "fake-model", "served model name")
	ttft := flag.Duration("ttft", 100*time.Millisecond, "time to first token")
	tpot := flag.Duration("tpot", 20*time.Millisecond, "time per output token after the first")
	tokens := flag.Int("tokens", 32, "completion tokens when the request sets no max_tokens")
	kv := flag.Int("kv-capacity", 64, "concurrent requests that fill the simulated KV cache")
	flag.Parse()

	e := fakeengine.New(fakeengine.Config{Model: *model, TTFT: *ttft, TPOT: *tpot, Tokens: *tokens, KVCapacity: *kv})
	log.Printf("fakeengine serving %q on %s", *model, *addr)
	srv := &http.Server{Addr: *addr, Handler: e.Handler(), ReadHeaderTimeout: 10 * time.Second}
	log.Fatal(srv.ListenAndServe())
}
