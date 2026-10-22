// Command loadgen drives an OpenAI-compatible endpoint with a shared-prefix
// workload and reports latency percentiles, how requests spread over
// backends, and how many prompt tokens the engines served from cache.
//
// The workload has -groups prompt groups. Each group has a long shared
// prefix — one tenant's system prompt, or one document being asked about —
// and each request appends its own question. A request picks its group
// uniformly, or by a Zipf law when -zipf > 1, the shape of multi-tenant
// traffic where a few prompts are hot. Responses are streamed so time to
// first token is measured at the client.
package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"math/rand/v2"
	"net/http"
	"os"
	"slices"
	"strings"
	"sync"
	"time"
)

type config struct {
	url, model, label      string
	requests, warmup       int
	concurrency            int
	rate                   float64
	groups                 int
	prefixWords, qWords    int
	maxTokens              int
	zipf                   float64
	seed                   uint64
	ignoreEOS, jsonSummary bool
	timeout                time.Duration
}

type result struct {
	ttft, e2e              time.Duration
	backend                string
	prompt, cached, output int
	err                    error
}

func main() {
	var c config
	flag.StringVar(&c.url, "url", "http://localhost:8080", "gateway or engine base URL")
	flag.StringVar(&c.model, "model", "qwen3-8b-sim", "model name")
	flag.StringVar(&c.label, "label", "", "label for the summary (e.g. the policy under test)")
	flag.IntVar(&c.requests, "requests", 400, "requests to send")
	flag.IntVar(&c.warmup, "warmup", 0, "leading requests excluded from the statistics")
	flag.IntVar(&c.concurrency, "concurrency", 16, "requests in flight (closed loop)")
	flag.Float64Var(&c.rate, "rate", 0, "Poisson arrival rate in requests/s (open loop); 0 uses -concurrency")
	flag.IntVar(&c.groups, "groups", 32, "prompt groups sharing a prefix")
	flag.IntVar(&c.prefixWords, "prefix-words", 600, "words in each group's shared prefix")
	flag.IntVar(&c.qWords, "question-words", 16, "words in each request's own question")
	flag.IntVar(&c.maxTokens, "max-tokens", 16, "output tokens per request")
	flag.Float64Var(&c.zipf, "zipf", 0, "Zipf exponent (>1) for group popularity; 0 means uniform")
	flag.Uint64Var(&c.seed, "seed", 1, "random seed; equal seeds give equal request sequences")
	flag.BoolVar(&c.ignoreEOS, "ignore-eos", true, "ask the engine to always generate -max-tokens tokens")
	flag.BoolVar(&c.jsonSummary, "json", false, "also print the summary as one JSON line")
	flag.DurationVar(&c.timeout, "timeout", 5*time.Minute, "per-request timeout")
	flag.Parse()
	if c.zipf != 0 && c.zipf <= 1 {
		fmt.Fprintln(os.Stderr, "-zipf must be greater than 1, or 0 for uniform")
		os.Exit(2)
	}

	bodies := buildRequests(c)
	client := &http.Client{Transport: &http.Transport{Proxy: nil, MaxIdleConnsPerHost: 1024, DisableCompression: true}}
	start := time.Now()
	results := run(c, client, bodies)
	report(c, results, time.Since(start))
}

// buildRequests makes the request bodies up front so the request sequence
// depends only on the seed, not on timing.
func buildRequests(c config) [][]byte {
	rng := rand.New(rand.NewPCG(c.seed, 0x5eed))
	text := func(n int) string {
		w := make([]string, n)
		for i := range w {
			w[i] = vocabulary[rng.IntN(len(vocabulary))]
		}
		return strings.Join(w, " ")
	}
	prefixes := make([]string, c.groups)
	for g := range prefixes {
		prefixes[g] = fmt.Sprintf("You are assistant #%d. %s", g, text(c.prefixWords))
	}
	var zipf *rand.Zipf
	if c.zipf > 1 {
		zipf = rand.NewZipf(rng, c.zipf, 1, uint64(c.groups-1))
	}
	bodies := make([][]byte, c.requests)
	for i := range bodies {
		g := rng.IntN(c.groups)
		if zipf != nil {
			g = int(zipf.Uint64())
		}
		req := map[string]any{
			"model":          c.model,
			"stream":         true,
			"stream_options": map[string]bool{"include_usage": true},
			"max_tokens":     c.maxTokens,
			"temperature":    0,
			"messages": []map[string]string{
				{"role": "system", "content": prefixes[g]},
				{"role": "user", "content": fmt.Sprintf("Question %d: %s?", i, text(c.qWords))},
			},
		}
		if c.ignoreEOS {
			req["ignore_eos"] = true
		}
		bodies[i], _ = json.Marshal(req)
	}
	return bodies
}

func run(c config, client *http.Client, bodies [][]byte) []result {
	results := make([]result, len(bodies))
	var wg sync.WaitGroup
	if c.rate > 0 {
		rng := rand.New(rand.NewPCG(c.seed, 0xa441))
		for i := range bodies {
			wg.Go(func() { results[i] = send(c, client, bodies[i]) })
			time.Sleep(time.Duration(rng.ExpFloat64() / c.rate * float64(time.Second)))
		}
	} else {
		next := make(chan int)
		for range c.concurrency {
			wg.Go(func() {
				for i := range next {
					results[i] = send(c, client, bodies[i])
				}
			})
		}
		for i := range bodies {
			next <- i
		}
		close(next)
	}
	wg.Wait()
	return results
}

func send(c config, client *http.Client, body []byte) result {
	ctx, cancel := context.WithTimeout(context.Background(), c.timeout)
	defer cancel()
	t0 := time.Now()
	req, _ := http.NewRequestWithContext(ctx, http.MethodPost, c.url+"/v1/chat/completions", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	resp, err := client.Do(req)
	if err != nil {
		return result{err: err}
	}
	defer resp.Body.Close()
	r := result{backend: resp.Header.Get("X-VSI-Backend")}
	if resp.StatusCode != http.StatusOK {
		var b bytes.Buffer
		b.ReadFrom(resp.Body)
		r.err = fmt.Errorf("status %d: %s", resp.StatusCode, strings.TrimSpace(b.String()))
		return r
	}
	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 64<<10), 1<<20)
	for sc.Scan() {
		data, ok := strings.CutPrefix(sc.Text(), "data: ")
		if !ok || data == "[DONE]" {
			continue
		}
		var ch chunk
		if json.Unmarshal([]byte(data), &ch) != nil {
			continue
		}
		if r.ttft == 0 && slices.ContainsFunc(ch.Choices, func(c choice) bool { return c.Delta.Content != "" }) {
			r.ttft = time.Since(t0)
		}
		if u := ch.Usage; u != nil {
			r.prompt, r.output = u.PromptTokens, u.CompletionTokens
			// vLLM reports prompt_tokens_details; llm-d-inference-sim prompt_tokens_detail.
			for _, d := range []*details{u.Details, u.Detail} {
				if d != nil {
					r.cached = max(r.cached, d.CachedTokens)
				}
			}
		}
	}
	r.e2e = time.Since(t0)
	r.err = sc.Err()
	return r
}

type chunk struct {
	Choices []choice `json:"choices"`
	Usage   *struct {
		PromptTokens     int      `json:"prompt_tokens"`
		CompletionTokens int      `json:"completion_tokens"`
		Details          *details `json:"prompt_tokens_details"`
		Detail           *details `json:"prompt_tokens_detail"`
	} `json:"usage"`
}

type choice struct {
	Delta struct {
		Content string `json:"content"`
	} `json:"delta"`
}

type details struct {
	CachedTokens int `json:"cached_tokens"`
}

type summary struct {
	Label         string         `json:"label"`
	Requests      int            `json:"requests"`
	Errors        int            `json:"errors"`
	WallSeconds   float64        `json:"wall_s"`
	ReqPerSec     float64        `json:"req_per_s"`
	OutTokPerSec  float64        `json:"output_tok_per_s"`
	TTFTms        map[string]int `json:"ttft_ms"`
	E2Ems         map[string]int `json:"e2e_ms"`
	CacheHitRatio float64        `json:"prompt_cache_hit_ratio"`
	Backends      map[string]int `json:"backends"`
}

func report(c config, all []result, wall time.Duration) {
	var ttft, e2e []time.Duration
	s := summary{Label: c.label, Requests: len(all), Backends: map[string]int{}}
	var prompt, cached, output int
	var firstErr error
	for i, r := range all {
		if r.err != nil {
			s.Errors++
			if firstErr == nil {
				firstErr = r.err
			}
			continue
		}
		if r.backend != "" {
			s.Backends[r.backend]++
		}
		if i < c.warmup {
			continue
		}
		ttft, e2e = append(ttft, r.ttft), append(e2e, r.e2e)
		prompt, cached, output = prompt+r.prompt, cached+r.cached, output+r.output
	}
	s.WallSeconds = wall.Seconds()
	s.ReqPerSec = float64(len(all)-s.Errors) / wall.Seconds()
	s.OutTokPerSec = float64(output) / wall.Seconds()
	s.TTFTms, s.E2Ems = percentiles(ttft), percentiles(e2e)
	if prompt > 0 {
		s.CacheHitRatio = float64(cached) / float64(prompt)
	}

	fmt.Printf("%s: %d requests, %d errors, %.1fs wall\n", orDefault(s.Label, "run"), s.Requests, s.Errors, s.WallSeconds)
	fmt.Printf("  throughput   %.1f req/s, %.0f output tok/s\n", s.ReqPerSec, s.OutTokPerSec)
	fmt.Printf("  TTFT ms      p50 %d  p90 %d  p99 %d\n", s.TTFTms["p50"], s.TTFTms["p90"], s.TTFTms["p99"])
	fmt.Printf("  E2E ms       p50 %d  p90 %d  p99 %d\n", s.E2Ems["p50"], s.E2Ems["p90"], s.E2Ems["p99"])
	fmt.Printf("  prompt cache %.1f%% of prompt tokens served from cache\n", 100*s.CacheHitRatio)
	names := make([]string, 0, len(s.Backends))
	for n := range s.Backends {
		names = append(names, n)
	}
	slices.Sort(names)
	var parts []string
	for _, n := range names {
		parts = append(parts, fmt.Sprintf("%s=%d", n, s.Backends[n]))
	}
	fmt.Printf("  backends     %s\n", strings.Join(parts, " "))
	if firstErr != nil {
		fmt.Printf("  first error  %v\n", firstErr)
	}
	if c.jsonSummary {
		b, _ := json.Marshal(s)
		fmt.Println(string(b))
	}
}

// percentiles reports nearest-rank p50/p90/p99 in milliseconds.
func percentiles(ds []time.Duration) map[string]int {
	out := map[string]int{"p50": 0, "p90": 0, "p99": 0}
	if len(ds) == 0 {
		return out
	}
	slices.Sort(ds)
	for name, p := range map[string]float64{"p50": 0.50, "p90": 0.90, "p99": 0.99} {
		i := int(p*float64(len(ds))+0.999999) - 1
		out[name] = int(ds[min(max(i, 0), len(ds)-1)].Milliseconds())
	}
	return out
}

func orDefault(s, def string) string {
	if s == "" {
		return def
	}
	return s
}

var vocabulary = strings.Fields(`account address answer apply balance bill card case change charge check
claim contact cover credit customer date delivery detail device discount email error fee file form help
invoice issue item limit login member message model monthly network notice number offer order owner
package password payment phone plan policy price problem product profile purchase question rate receipt
record refund renewal request reset return review service setting shipping signal speed status store
subscription support system tax ticket transfer update upgrade usage user verify warranty window`)
