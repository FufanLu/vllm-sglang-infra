package gateway

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/FufanLu/vllm-sglang-infra/internal/backend"
	"github.com/FufanLu/vllm-sglang-infra/internal/flow"
	"github.com/FufanLu/vllm-sglang-infra/internal/scheduler"
	"github.com/FufanLu/vllm-sglang-infra/internal/tenancy"
	"github.com/FufanLu/vllm-sglang-infra/internal/usage"
)

// FairnessHeader carries the tenant to model servers and to llm-d's flow
// control, which trusts it only from an upstream gateway like this one.
const FairnessHeader = "X-Llm-D-Inference-Fairness-Id"

// call is the state of one inference request on its way through the
// pipeline: authenticate, reserve quota, wait for a dispatch slot, schedule,
// forward and relay, then settle.
type call struct {
	route     string
	start     time.Time
	reqID     string
	tenant    *tenancy.Tenant
	req       inferenceRequest
	body      []byte // what is forwarded; may differ from the client's body
	dropUsage bool   // usage chunk injected by the gateway, not for the client
	sreq      *scheduler.Request
	estimate  int64 // estimated tokens: prompt + max_tokens
	reserved  int64 // tokens taken from the tenant's quota
	ticket    *flow.Ticket
	queued    time.Duration

	served  *backend.Endpoint // endpoint whose response reached the client
	code    string
	retries int
	stats   relayStats
	relayed bool
	prefill time.Duration // disaggregated serving: time in the prefill phase
}

func (s *Server) inference(route string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		c := &call{route: route, start: time.Now(), reqID: r.Header.Get("X-Request-Id")}
		if c.reqID == "" {
			c.reqID = newRequestID()
		}
		tenant, err := s.opts.Tenants.Authenticate(r.Header.Get("Authorization"))
		if err != nil {
			s.opts.Metrics.Requests.WithLabelValues("none", route, "unauthorized").Inc()
			writeError(w, http.StatusUnauthorized, "authentication_error", err.Error())
			return
		}
		c.tenant = tenant

		raw, err := io.ReadAll(http.MaxBytesReader(w, r.Body, s.opts.MaxBodyBytes))
		if err != nil {
			var tooBig *http.MaxBytesError
			if errors.As(err, &tooBig) {
				writeError(w, http.StatusRequestEntityTooLarge, "invalid_request_error", "request body too large")
				return
			}
			writeError(w, http.StatusBadRequest, "invalid_request_error", "could not read request body")
			return
		}
		if err := json.Unmarshal(raw, &c.req); err != nil {
			writeError(w, http.StatusBadRequest, "invalid_request_error", "request body is not valid JSON")
			return
		}
		c.body, c.dropUsage = prepareBody(raw, &c.req)
		c.sreq = &scheduler.Request{Model: c.req.Model, Prompt: c.req.promptText(), Tenant: tenant.Name,
			AllowExternal: !tenant.OnPremOnly}
		c.estimate = c.req.estimateTokens(s.opts.DefaultMaxTokens)

		if !s.reserve(r.Context(), w, c) {
			return
		}
		defer s.finish(c)
		if !s.admit(r.Context(), w, c) {
			return
		}
		s.tokenize(r.Context(), c)
		if s.opts.PD && c.sreq.Placement != scheduler.ExternalOnly && s.dispatchPD(w, r, c) {
			return
		}
		s.dispatch(w, r, c, c.sreq, func(ep *backend.Endpoint) []byte { return s.bodyFor(c, ep, c.body) })
	}
}

// reserve takes the request's estimated tokens from the tenant's budget.
func (s *Server) reserve(ctx context.Context, w http.ResponseWriter, c *call) bool {
	ok, retryAfter, err := s.opts.Quota.Reserve(ctx, c.tenant.Name, c.tenant.TokensPerMinute, c.estimate)
	switch {
	case err != nil:
		// A broken quota store must not take the pool down with it: admit,
		// and charge actual usage at the end.
		s.opts.Log.Warn("quota check failed; admitting", "tenant", c.tenant.Name, "err", err)
		return true
	case !ok:
		s.opts.Metrics.QuotaRejections.WithLabelValues(c.tenant.Name).Inc()
		w.Header().Set("Retry-After", strconv.Itoa(max(1, int(math.Ceil(retryAfter.Seconds())))))
		writeError(w, http.StatusTooManyRequests, "rate_limit_exceeded",
			fmt.Sprintf("tenant %s is over its token budget", c.tenant.Name))
		c.code = "quota_exceeded"
		s.finish(c)
		return false
	}
	c.reserved = c.estimate
	return true
}

// admit waits for a dispatch slot when flow control is on. If the pool is
// full and the tenant may leave the cluster, the request overflows to an
// external provider instead of queueing.
func (s *Server) admit(ctx context.Context, w http.ResponseWriter, c *call) bool {
	f := s.opts.Flow
	if f == nil {
		return true
	}
	band, weight, cost := c.tenant.Priority, c.tenant.Weight, float64(c.estimate)
	if tk := f.TryAcquire(band, c.tenant.Name, weight, cost); tk != nil {
		c.ticket = tk
		return true
	}
	if c.sreq.AllowExternal && s.externalAvailable(c.req.Model) {
		c.sreq.Placement = scheduler.ExternalOnly
		s.opts.Metrics.Overflow.WithLabelValues("pool_full").Inc()
		return true
	}
	tk, err := f.Acquire(ctx, band, c.tenant.Name, weight, cost)
	switch {
	case err == nil:
		c.ticket, c.queued = tk, tk.Waited
		s.opts.Metrics.FlowWait.WithLabelValues(band.String()).Observe(tk.Waited.Seconds())
		return true
	case errors.Is(err, flow.ErrQueueFull):
		c.code = "queue_full"
		w.Header().Set("Retry-After", "1")
		writeError(w, http.StatusTooManyRequests, "rate_limit_exceeded", "the pool is saturated; retry later")
	case errors.Is(err, flow.ErrQueueTimeout):
		c.code = "queue_timeout"
		writeError(w, http.StatusServiceUnavailable, "service_unavailable", "timed out waiting for capacity")
	default:
		c.code = "client_cancel"
	}
	s.opts.Metrics.FlowRejections.WithLabelValues(band.String(), c.code).Inc()
	return false
}

// tokenize fills in the request's token IDs when the policy matches KV
// blocks exactly. Any healthy engine serving the model can tokenize: they
// share a tokenizer and chat template. On failure the request is still
// scheduled, by load alone.
func (s *Server) tokenize(ctx context.Context, c *call) {
	if s.opts.Tokenizer == nil || !s.scheduler().NeedsTokens() {
		return
	}
	var parts struct {
		Messages json.RawMessage `json:"messages"`
		Prompt   json.RawMessage `json:"prompt"`
	}
	json.Unmarshal(c.body, &parts)
	for _, ep := range s.opts.Registry.List() {
		if ep.IsExternal() || !ep.Healthy() || !ep.Serves(c.req.Model) {
			continue
		}
		t0 := time.Now()
		tctx, cancel := context.WithTimeout(ctx, 2*time.Second)
		tokens, err := s.opts.Tokenizer.Tokens(tctx, ep.URL.String(), c.req.Model, parts.Messages, parts.Prompt)
		cancel()
		s.opts.Metrics.Tokenize.WithLabelValues(strconv.FormatBool(err == nil)).Observe(time.Since(t0).Seconds())
		if err != nil {
			s.opts.Log.Debug("tokenize failed; scheduling by load", "endpoint", ep.Name, "err", err)
			return
		}
		c.sreq.Tokens = tokens
		return
	}
}

func (s *Server) externalAvailable(model string) bool {
	for _, ep := range s.opts.Registry.List() {
		if ep.IsExternal() && ep.Healthy() && ep.Serves(model) && ep.Inflight() < int64(max(ep.External.MaxInflight, 1)) {
			return true
		}
	}
	return false
}

// dispatch schedules sreq and forwards it, retrying on other endpoints while
// nothing has reached the client.
func (s *Server) dispatch(w http.ResponseWriter, r *http.Request, c *call, sreq *scheduler.Request,
	bodyFor func(*backend.Endpoint) []byte) {
	sched := s.scheduler()
	tried := map[string]bool{}
	for attempt := 0; ; attempt++ {
		ep, err := sched.Schedule(sreq, s.opts.Registry.List(), tried)
		if err != nil {
			c.code = "no_backend"
			msg := "no healthy backend serves this model"
			if len(tried) > 0 {
				msg = "all backends that serve this model failed"
			}
			writeError(w, http.StatusServiceUnavailable, "service_unavailable", msg)
			return
		}
		s.opts.Metrics.Picks.WithLabelValues(sched.Policy, ep.Name).Inc()
		if ep.IsExternal() {
			s.opts.Metrics.Overflow.WithLabelValues("dispatched").Inc()
		}
		retry := s.forward(w, r, c, ep, bodyFor(ep), attempt < s.opts.MaxRetries)
		ep.Release()
		if retry == "" {
			return
		}
		tried[ep.Name] = true
		c.retries++
		s.opts.Metrics.Retries.WithLabelValues(retry).Inc()
		s.opts.Log.Warn("retrying on another backend", "request_id", c.reqID, "backend", ep.Name, "reason", retry)
	}
}

// bodyFor adapts the forwarded body to an endpoint: external providers get
// their own model name.
func (s *Server) bodyFor(c *call, ep *backend.Endpoint, body []byte) []byte {
	if !ep.IsExternal() {
		return body
	}
	name := ep.External.Models[c.req.Model]
	b, err := rewriteJSON(body, func(m map[string]any) { m["model"] = name })
	if err != nil {
		return body
	}
	return b
}

// upstreamRequest builds the request to ep for the client's route.
func (s *Server) upstreamRequest(ctx context.Context, r *http.Request, c *call, ep *backend.Endpoint, body []byte) (*http.Request, error) {
	target := *ep.URL
	if ep.IsExternal() {
		// Provider base URLs end in /v1, as in OpenAI SDKs.
		target.Path = ep.URL.Path + strings.TrimPrefix(r.URL.Path, "/v1")
	} else {
		target.Path = ep.URL.Path + r.URL.Path
		target.RawQuery = r.URL.RawQuery
	}
	up, err := http.NewRequestWithContext(ctx, http.MethodPost, target.String(), bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	copyHeaders(up.Header, r.Header, requestDropHeaders)
	up.Header.Set("Content-Type", "application/json")
	up.Header.Set("X-Request-Id", c.reqID)
	if ep.IsExternal() {
		up.Header.Set("Authorization", "Bearer "+ep.External.APIKey)
	} else {
		up.Header.Set(FairnessHeader, c.tenant.Name)
	}
	return up, nil
}

// forward sends one attempt to ep and relays the response. It returns a
// non-empty reason when the attempt failed before anything was written to the
// client and canRetry allows trying another backend.
func (s *Server) forward(w http.ResponseWriter, r *http.Request, c *call, ep *backend.Endpoint,
	body []byte, canRetry bool) (retryReason string) {

	up, err := s.upstreamRequest(r.Context(), r, c, ep, body)
	if err != nil {
		c.code = "internal_error"
		writeError(w, http.StatusInternalServerError, "internal_error", "could not build upstream request")
		return ""
	}
	// The request context ends when the client disconnects, which aborts the
	// upstream request; vLLM and SGLang then stop generating for it.
	resp, err := s.opts.Client.Do(up)
	if err != nil {
		if r.Context().Err() != nil {
			c.code = "client_cancel"
			s.opts.Metrics.Requests.WithLabelValues(ep.Name, c.route, c.code).Inc()
			return ""
		}
		if ep.MarkDown() {
			s.opts.Log.Warn("endpoint down", "endpoint", ep.Name, "err", err)
		}
		if canRetry {
			return "connect_error"
		}
		c.code = "upstream_error"
		s.opts.Metrics.Requests.WithLabelValues(ep.Name, c.route, c.code).Inc()
		writeError(w, http.StatusBadGateway, "upstream_error", "backend "+ep.Name+" unreachable")
		return ""
	}
	defer resp.Body.Close()
	if canRetry && retryableStatus(resp.StatusCode) {
		io.Copy(io.Discard, io.LimitReader(resp.Body, 64<<10))
		return "status_" + strconv.Itoa(resp.StatusCode)
	}

	// From here on the response belongs to this backend: no more retries.
	copyHeaders(w.Header(), resp.Header, hopHeaders)
	w.Header().Set("X-Request-Id", c.reqID)
	w.Header().Set("X-VSI-Backend", ep.Name)
	w.WriteHeader(resp.StatusCode)
	c.served, c.relayed = ep, true

	cw := &clientWriter{w: w, rc: http.NewResponseController(w)}
	if strings.HasPrefix(resp.Header.Get("Content-Type"), "text/event-stream") {
		c.stats, err = relayStream(cw, resp.Body, c.dropUsage)
	} else {
		c.stats, err = relayBody(cw, resp.Body)
	}

	c.code = strconv.Itoa(resp.StatusCode)
	switch {
	case cw.err != nil || r.Context().Err() != nil:
		c.code = "client_cancel"
	case err != nil:
		c.code = "upstream_error"
		s.opts.Log.Warn("relay interrupted", "request_id", c.reqID, "backend", ep.Name, "err", err)
	}
	m := s.opts.Metrics
	m.Requests.WithLabelValues(ep.Name, c.route, c.code).Inc()
	m.Duration.WithLabelValues(ep.Name, c.route).Observe(time.Since(c.start).Seconds())
	st := &c.stats
	if !st.firstToken.IsZero() {
		m.TTFT.WithLabelValues(ep.Name, c.route).Observe(st.firstToken.Sub(c.start).Seconds())
		if n := st.outputTokens(); n > 1 {
			m.TPOT.WithLabelValues(ep.Name, c.route).Observe(st.lastToken.Sub(st.firstToken).Seconds() / float64(n-1))
		}
	}
	if st.promptTokens > 0 {
		m.Tokens.WithLabelValues(ep.Name, "prompt").Add(float64(st.promptTokens))
	}
	if st.cachedTokens > 0 {
		m.Tokens.WithLabelValues(ep.Name, "cached").Add(float64(st.cachedTokens))
	}
	if n := st.outputTokens(); n > 0 {
		m.Tokens.WithLabelValues(ep.Name, "completion").Add(float64(n))
	}
	return ""
}

// finish settles the request: corrects the quota reservation and the flow
// controller's fairness counter with actual usage, and writes the usage record.
func (s *Server) finish(c *call) {
	st := &c.stats
	actual := int64(st.promptTokens + st.outputTokens())
	if actual == 0 && c.relayed && strings.HasPrefix(c.code, "2") {
		actual = c.estimate // served, but the engine reported no usage
	}
	if delta := actual - c.reserved; delta != 0 && c.tenant.TokensPerMinute > 0 {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		if err := s.opts.Quota.Adjust(ctx, c.tenant.Name, c.tenant.TokensPerMinute, delta); err != nil {
			s.opts.Log.Warn("quota settle failed", "tenant", c.tenant.Name, "err", err)
		}
		cancel()
	}
	c.ticket.Release(float64(actual))

	rec := usage.Record{
		Time: c.start, RequestID: c.reqID, Tenant: c.tenant.Name, Model: c.req.Model, Route: c.route,
		Status: c.code, PromptTokens: st.promptTokens, CachedTokens: st.cachedTokens,
		CompletionTokens: st.outputTokens(), DurationMs: ms(time.Since(c.start)), QueueMs: ms(c.queued),
		Retries: c.retries,
	}
	if !st.firstToken.IsZero() {
		rec.TTFTms = ms(st.firstToken.Sub(c.start))
	}
	if ep := c.served; ep != nil {
		rec.Backend = ep.Name
		if ep.IsExternal() {
			rec.External = true
			rec.CostUSD = ep.External.Cost(st.promptTokens, st.outputTokens())
		}
	}
	s.opts.Usage.Write(rec)
	m := s.opts.Metrics
	m.TenantRequests.WithLabelValues(c.tenant.Name, c.code).Inc()
	m.TenantTokens.WithLabelValues(c.tenant.Name, "prompt").Add(float64(st.promptTokens))
	m.TenantTokens.WithLabelValues(c.tenant.Name, "completion").Add(float64(st.outputTokens()))
}

func ms(d time.Duration) float64 { return float64(d.Microseconds()) / 1000 }

func retryableStatus(code int) bool {
	return code == http.StatusBadGateway || code == http.StatusServiceUnavailable || code == http.StatusGatewayTimeout
}

// prepareBody returns the body to forward. A streamed request that did not
// ask for usage gets stream_options.include_usage set, because metering needs
// the engine's token counts; the extra usage chunk is then dropped from the
// stream so the client sees exactly what it asked for.
func prepareBody(raw []byte, req *inferenceRequest) ([]byte, bool) {
	if !req.Stream || (req.StreamOptions != nil && req.StreamOptions.IncludeUsage) {
		return raw, false
	}
	b, err := rewriteJSON(raw, func(m map[string]any) {
		opts, _ := m["stream_options"].(map[string]any)
		if opts == nil {
			opts = map[string]any{}
		}
		opts["include_usage"] = true
		m["stream_options"] = opts
	})
	if err != nil {
		return raw, false
	}
	return b, true
}

// rewriteJSON decodes a JSON object, applies edit, and re-encodes it. Numbers
// are kept as written.
func rewriteJSON(body []byte, edit func(map[string]any)) ([]byte, error) {
	dec := json.NewDecoder(bytes.NewReader(body))
	dec.UseNumber()
	var m map[string]any
	if err := dec.Decode(&m); err != nil {
		return nil, err
	}
	edit(m)
	return json.Marshal(m)
}

// relayStats is what the gateway learns about a response while relaying it.
type relayStats struct {
	firstToken, lastToken time.Time
	contentEvents         int // SSE events that carried generated text
	promptTokens          int // from the usage object, when present
	cachedTokens          int
	completionTokens      int
}

// outputTokens prefers the engine's usage count and falls back to counting
// content events, which vLLM and SGLang emit about once per token.
func (s *relayStats) outputTokens() int {
	if s.completionTokens > 0 {
		return s.completionTokens
	}
	return s.contentEvents
}

func (s *relayStats) setUsage(u *usageInfo) {
	s.promptTokens, s.completionTokens = u.PromptTokens, u.CompletionTokens
	if u.Details != nil {
		s.cachedTokens = u.Details.CachedTokens
	}
}

// usageInfo is the usage object of an OpenAI response.
type usageInfo struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	Details          *struct {
		CachedTokens int `json:"cached_tokens"`
	} `json:"prompt_tokens_details"`
}

// streamChunk is the part of an OpenAI streaming chunk the gateway inspects.
type streamChunk struct {
	Choices []struct {
		Delta struct {
			Content          string `json:"content"`
			ReasoningContent string `json:"reasoning_content"`
		} `json:"delta"`
		Text string `json:"text"` // /v1/completions
	} `json:"choices"`
	Usage *usageInfo `json:"usage"`
}

// clientWriter writes to the client and remembers the first write error, so
// a relay that stops can be blamed on the right side: a client that went away
// is not an upstream failure.
type clientWriter struct {
	w   http.ResponseWriter
	rc  *http.ResponseController
	err error
}

func (c *clientWriter) Write(p []byte) (int, error) {
	if c.err != nil {
		return 0, c.err
	}
	n, err := c.w.Write(p)
	c.err = err
	return n, err
}

func (c *clientWriter) Flush() error {
	if c.err == nil {
		c.err = c.rc.Flush()
	}
	return c.err
}

// relayStream copies a server-sent-event stream to the client one event at a
// time, flushing after each read so tokens are not held in buffers. It times
// the first and last token as events pass and, when dropUsageOnly is set,
// withholds the usage-only chunk the gateway asked for on the client's behalf.
func relayStream(w *clientWriter, body io.Reader, dropUsageOnly bool) (relayStats, error) {
	var st relayStats
	buf := make([]byte, 32<<10)
	var pending []byte
	for {
		n, err := body.Read(buf)
		if n > 0 {
			pending = append(pending, buf[:n]...)
			now := time.Now()
			wrote := false
			for {
				end := eventEnd(pending)
				if end < 0 {
					break
				}
				if !st.observe(pending[:end], now) || !dropUsageOnly {
					if _, werr := w.Write(pending[:end]); werr != nil {
						return st, werr
					}
					wrote = true
				}
				pending = pending[end:]
			}
			if len(pending) > 1<<20 { // not SSE as we know it: pass it through
				if _, werr := w.Write(pending); werr != nil {
					return st, werr
				}
				pending, wrote = nil, true
			}
			pending = bytes.Clone(pending)
			if wrote {
				if ferr := w.Flush(); ferr != nil {
					return st, ferr
				}
			}
		}
		if errors.Is(err, io.EOF) {
			if len(pending) > 0 {
				w.Write(pending)
			}
			return st, w.Flush()
		}
		if err != nil {
			return st, err
		}
	}
}

// eventEnd returns the length of the first complete event in b, including
// its blank-line terminator, or -1.
func eventEnd(b []byte) int {
	if i := bytes.Index(b, []byte("\n\n")); i >= 0 {
		if j := bytes.Index(b, []byte("\r\n\r\n")); j >= 0 && j < i {
			return j + 4
		}
		return i + 2
	}
	if j := bytes.Index(b, []byte("\r\n\r\n")); j >= 0 {
		return j + 4
	}
	return -1
}

// observe records what an event says about timing and usage. It reports
// whether the event is a usage-only chunk (no choices, only usage).
func (st *relayStats) observe(event []byte, now time.Time) (usageOnly bool) {
	for _, line := range bytes.Split(event, []byte("\n")) {
		payload, ok := bytes.CutPrefix(bytes.TrimSpace(line), []byte("data:"))
		if !ok {
			continue
		}
		payload = bytes.TrimSpace(payload)
		if len(payload) == 0 || bytes.Equal(payload, []byte("[DONE]")) {
			continue
		}
		var c streamChunk
		if json.Unmarshal(payload, &c) != nil {
			continue
		}
		for _, ch := range c.Choices {
			if ch.Delta.Content != "" || ch.Delta.ReasoningContent != "" || ch.Text != "" {
				if st.firstToken.IsZero() {
					st.firstToken = now
				}
				st.lastToken = now
				st.contentEvents++
				break
			}
		}
		if c.Usage != nil {
			st.setUsage(c.Usage)
			usageOnly = len(c.Choices) == 0
		}
	}
	return usageOnly
}

// relayBody copies a non-streamed response and reads its usage object. Only
// the first 4 MiB are kept for parsing; LLM responses are far smaller.
func relayBody(w *clientWriter, body io.Reader) (relayStats, error) {
	var st relayStats
	capture := &capBuffer{max: 4 << 20}
	_, err := io.Copy(w, io.TeeReader(body, capture))
	if err == nil && !capture.overflow {
		var resp struct {
			Usage *usageInfo `json:"usage"`
		}
		if json.Unmarshal(capture.buf.Bytes(), &resp) == nil && resp.Usage != nil {
			st.setUsage(resp.Usage)
		}
	}
	return st, err
}

// capBuffer keeps the first max bytes written to it and drops the rest.
type capBuffer struct {
	buf      bytes.Buffer
	max      int
	overflow bool
}

func (c *capBuffer) Write(p []byte) (int, error) {
	if room := c.max - c.buf.Len(); room > 0 {
		c.buf.Write(p[:min(len(p), room)])
	}
	if c.buf.Len() >= c.max {
		c.overflow = true
	}
	return len(p), nil
}

// Hop-by-hop headers (RFC 9110 §7.6.1) never cross a proxy.
var hopHeaders = []string{
	"Connection", "Keep-Alive", "Proxy-Authenticate", "Proxy-Authorization",
	"Proxy-Connection", "Te", "Trailer", "Transfer-Encoding", "Upgrade",
}

// requestDropHeaders are also stripped from requests sent upstream: the
// client's credentials are for the gateway, a forwarded Accept-Encoding would
// let the engine compress the stream we need to read, the body length is set
// by the transport, and the tenant header must come from the gateway, never
// from the client.
var requestDropHeaders = append([]string{"Authorization", "Accept-Encoding", "Content-Length", "Host", FairnessHeader},
	hopHeaders...)

func copyHeaders(dst, src http.Header, drop []string) {
	skip := map[string]bool{}
	for _, h := range drop {
		skip[http.CanonicalHeaderKey(h)] = true
	}
	for _, v := range src.Values("Connection") {
		for _, f := range strings.Split(v, ",") {
			if f = strings.TrimSpace(f); f != "" {
				skip[http.CanonicalHeaderKey(f)] = true
			}
		}
	}
	for k, vs := range src {
		if !skip[http.CanonicalHeaderKey(k)] {
			dst[k] = append(dst[k][:0:0], vs...)
		}
	}
}

// writeError writes an OpenAI-style error body.
func writeError(w http.ResponseWriter, status int, typ, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	fmt.Fprintf(w, `{"error":{"message":%q,"type":%q,"code":%d}}`+"\n", msg, typ, status)
}
