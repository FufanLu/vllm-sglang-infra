package backend

import (
	"bufio"
	"io"
	"strconv"
	"strings"
)

// parsePromText reads the Prometheus text exposition format and returns every
// sample value grouped by metric name. Labels are dropped because the monitor
// only aggregates across series (sum or max). Lines it cannot parse are
// skipped, so one odd series does not discard a whole scrape.
func parsePromText(r io.Reader) (map[string][]float64, error) {
	out := make(map[string][]float64)
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 64<<10), 1<<20)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" || line[0] == '#' {
			continue
		}
		name, rest, ok := splitSample(line)
		if !ok {
			continue
		}
		fields := strings.Fields(rest)
		if len(fields) == 0 {
			continue
		}
		v, err := strconv.ParseFloat(fields[0], 64)
		if err != nil {
			continue
		}
		out[name] = append(out[name], v)
	}
	return out, sc.Err()
}

// splitSample splits `name{labels} value [timestamp]` into the metric name and
// the text after the label set. Label values are quoted and may contain
// spaces, braces, and escaped quotes.
func splitSample(line string) (name, rest string, ok bool) {
	i := strings.IndexAny(line, "{ \t")
	if i <= 0 {
		return "", "", false
	}
	name = line[:i]
	if line[i] != '{' {
		return name, line[i:], true
	}
	inQuote, escaped := false, false
	for j := i + 1; j < len(line); j++ {
		c := line[j]
		switch {
		case escaped:
			escaped = false
		case c == '\\':
			escaped = inQuote
		case c == '"':
			inQuote = !inQuote
		case c == '}' && !inQuote:
			return name, line[j+1:], true
		}
	}
	return "", "", false
}

// metricNames lists, per engine, the series that make up a Snapshot. For each
// field the first name present in a scrape wins, which lets one adapter cover
// several engine versions.
type metricNames struct {
	running, waiting, kvUsage []string
}

var engineMetrics = map[Engine]metricNames{
	EngineVLLM: {
		running: []string{"vllm:num_requests_running"},
		waiting: []string{"vllm:num_requests_waiting"},
		// kv_cache_usage_perc is the V1 name (a fraction despite the suffix);
		// gpu_cache_usage_perc is the V0 name.
		kvUsage: []string{"vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"},
	},
	EngineSGLang: {
		running: []string{"sglang:num_running_reqs"},
		waiting: []string{"sglang:num_queue_reqs"},
		kvUsage: []string{"sglang:token_usage"},
	},
}

// extractSnapshot turns scraped samples into a Snapshot. Request counts are
// summed across series (one per engine core when the server fronts several
// data-parallel ranks); KV usage takes the maximum, since the fullest rank is
// the one that will stall.
func extractSnapshot(e Engine, samples map[string][]float64) (Snapshot, bool) {
	names, ok := engineMetrics[e]
	if !ok {
		return Snapshot{}, false
	}
	running, okR := first(samples, names.running, sum)
	waiting, okW := first(samples, names.waiting, sum)
	kv, _ := first(samples, names.kvUsage, maxOf)
	return Snapshot{Running: running, Waiting: waiting, KVUsage: kv}, okR || okW
}

func first(samples map[string][]float64, names []string, agg func([]float64) float64) (float64, bool) {
	for _, n := range names {
		if vs, ok := samples[n]; ok && len(vs) > 0 {
			return agg(vs), true
		}
	}
	return 0, false
}

func sum(vs []float64) float64 {
	t := 0.0
	for _, v := range vs {
		t += v
	}
	return t
}

func maxOf(vs []float64) float64 {
	m := vs[0]
	for _, v := range vs[1:] {
		m = max(m, v)
	}
	return m
}
