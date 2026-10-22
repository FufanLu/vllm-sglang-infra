package gateway

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/FufanLu/vllm-sglang-infra/internal/backend"
	"github.com/FufanLu/vllm-sglang-infra/internal/scheduler"
)

// inference serves /v1/chat/completions and /v1/completions: schedule,
// forward, relay, and retry on another backend if the chosen one fails before
// any response byte has reached the client.
func (s *Server) inference(route string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, s.opts.MaxBodyBytes))
		if err != nil {
			var tooBig *http.MaxBytesError
			if errors.As(err, &tooBig) {
				writeError(w, http.StatusRequestEntityTooLarge, "invalid_request_error", "request body too large")
				return
			}
			writeError(w, http.StatusBadRequest, "invalid_request_error", "could not read request body")
			return
		}
		var req inferenceRequest
		if err := json.Unmarshal(body, &req); err != nil {
			writeError(w, http.StatusBadRequest, "invalid_request_error", "request body is not valid JSON")
			return
		}
		sreq := &scheduler.Request{Model: req.Model, Prompt: req.promptText()}
		reqID := r.Header.Get("X-Request-Id")
		if reqID == "" {
			reqID = newRequestID()
		}

		tried := map[string]bool{}
		for attempt := 0; ; attempt++ {
			ep, err := s.opts.Scheduler.Schedule(sreq, s.opts.Registry.List(), tried)
			if err != nil {
				s.opts.Metrics.Requests.WithLabelValues("none", route, "no_backend").Inc()
				msg := "no healthy backend serves this model"
				if len(tried) > 0 {
					msg = "all backends that serve this model failed"
				}
				writeError(w, http.StatusServiceUnavailable, "service_unavailable", msg)
				return
			}
			s.opts.Metrics.Picks.WithLabelValues(s.opts.Scheduler.Policy, ep.Name).Inc()
			retry := s.forward(w, r, route, ep, body, reqID, start, attempt < s.opts.MaxRetries)
			ep.Release()
			if retry == "" {
				return
			}
			tried[ep.Name] = true
			s.opts.Metrics.Retries.WithLabelValues(retry).Inc()
			s.opts.Log.Warn("retrying on another backend", "request_id", reqID, "backend", ep.Name, "reason", retry)
		}
	}
}

// forward sends one attempt to ep and relays the response. It returns a
// non-empty reason when the attempt failed before anything was written to the
// client and canRetry allows trying another backend.
func (s *Server) forward(w http.ResponseWriter, r *http.Request, route string, ep *backend.Endpoint,
	body []byte, reqID string, start time.Time, canRetry bool) (retryReason string) {

	target := *ep.URL
	target.Path = ep.URL.Path + r.URL.Path
	target.RawQuery = r.URL.RawQuery
	up, err := http.NewRequestWithContext(r.Context(), http.MethodPost, target.String(), bytes.NewReader(body))
	if err != nil {
		writeError(w, http.StatusInternalServerError, "internal_error", "could not build upstream request")
		return ""
	}
	copyHeaders(up.Header, r.Header, requestDropHeaders)
	up.Header.Set("X-Request-Id", reqID)

	// The request context ends when the client disconnects, which aborts the
	// upstream request; vLLM and SGLang then stop generating for it.
	resp, err := s.opts.Client.Do(up)
	if err != nil {
		if r.Context().Err() != nil {
			s.opts.Metrics.Requests.WithLabelValues(ep.Name, route, "client_cancel").Inc()
			return ""
		}
		if ep.MarkDown() {
			s.opts.Log.Warn("endpoint down", "endpoint", ep.Name, "err", err)
		}
		if canRetry {
			return "connect_error"
		}
		s.opts.Metrics.Requests.WithLabelValues(ep.Name, route, "upstream_error").Inc()
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
	w.Header().Set("X-Request-Id", reqID)
	w.Header().Set("X-VSI-Backend", ep.Name)
	w.WriteHeader(resp.StatusCode)

	cw := &clientWriter{w: w, rc: http.NewResponseController(w)}
	var st relayStats
	if strings.HasPrefix(resp.Header.Get("Content-Type"), "text/event-stream") {
		st, err = relayStream(cw, resp.Body)
	} else {
		st, err = relayBody(cw, resp.Body)
	}

	code := strconv.Itoa(resp.StatusCode)
	switch {
	case cw.err != nil || r.Context().Err() != nil:
		code = "client_cancel"
	case err != nil:
		code = "upstream_error"
		s.opts.Log.Warn("relay interrupted", "request_id", reqID, "backend", ep.Name, "err", err)
	}
	m := s.opts.Metrics
	m.Requests.WithLabelValues(ep.Name, route, code).Inc()
	m.Duration.WithLabelValues(ep.Name, route).Observe(time.Since(start).Seconds())
	if !st.firstToken.IsZero() {
		m.TTFT.WithLabelValues(ep.Name, route).Observe(st.firstToken.Sub(start).Seconds())
		if n := st.outputTokens(); n > 1 {
			m.TPOT.WithLabelValues(ep.Name, route).Observe(st.lastToken.Sub(st.firstToken).Seconds() / float64(n-1))
		}
	}
	if st.promptTokens > 0 {
		m.Tokens.WithLabelValues(ep.Name, "prompt").Add(float64(st.promptTokens))
	}
	if n := st.outputTokens(); n > 0 {
		m.Tokens.WithLabelValues(ep.Name, "completion").Add(float64(n))
	}
	return ""
}

func retryableStatus(code int) bool {
	return code == http.StatusBadGateway || code == http.StatusServiceUnavailable || code == http.StatusGatewayTimeout
}

// relayStats is what the gateway learns about a response while relaying it.
type relayStats struct {
	firstToken, lastToken time.Time
	contentEvents         int // SSE events that carried generated text
	promptTokens          int // from the usage object, when present
	completionTokens      int
}

// outputTokens prefers the engine's usage count; without one (streaming
// without stream_options.include_usage) it falls back to counting content
// events, which vLLM and SGLang emit about once per token.
func (s *relayStats) outputTokens() int {
	if s.completionTokens > 0 {
		return s.completionTokens
	}
	return s.contentEvents
}

type usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
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
	Usage *usage `json:"usage"`
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

// relayStream copies a server-sent-event stream to the client, flushing after
// every read so tokens are not held in buffers, and watches the events go by
// to time the first and last token.
func relayStream(w *clientWriter, body io.Reader) (relayStats, error) {
	var st relayStats
	buf := make([]byte, 32<<10)
	var pending []byte // an incomplete line carried over from the last read
	for {
		n, err := body.Read(buf)
		if n > 0 {
			if _, werr := w.Write(buf[:n]); werr != nil {
				return st, werr
			}
			if ferr := w.Flush(); ferr != nil {
				return st, ferr
			}
			pending = st.observe(append(pending, buf[:n]...), time.Now())
		}
		if errors.Is(err, io.EOF) {
			return st, nil
		}
		if err != nil {
			return st, err
		}
	}
}

// observe consumes the complete lines in data and returns the incomplete
// remainder. All lines of one read share the timestamp of that read.
func (st *relayStats) observe(data []byte, now time.Time) []byte {
	for {
		i := bytes.IndexByte(data, '\n')
		if i < 0 {
			break
		}
		line := bytes.TrimSpace(data[:i])
		data = data[i+1:]
		payload, ok := bytes.CutPrefix(line, []byte("data:"))
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
			st.promptTokens, st.completionTokens = c.Usage.PromptTokens, c.Usage.CompletionTokens
		}
	}
	if len(data) > 1<<20 {
		return nil // not SSE as we know it; stop buffering
	}
	return bytes.Clone(data)
}

// relayBody copies a non-streamed response and reads its usage object. Only
// the first 4 MiB are kept for parsing; LLM responses are far smaller.
func relayBody(w *clientWriter, body io.Reader) (relayStats, error) {
	var st relayStats
	capture := &capBuffer{max: 4 << 20}
	_, err := io.Copy(w, io.TeeReader(body, capture))
	if err == nil && !capture.overflow {
		var resp struct {
			Usage *usage `json:"usage"`
		}
		if json.Unmarshal(capture.buf.Bytes(), &resp) == nil && resp.Usage != nil {
			st.promptTokens, st.completionTokens = resp.Usage.PromptTokens, resp.Usage.CompletionTokens
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
// let the engine compress the stream we need to read, and the body length is
// set by the transport.
var requestDropHeaders = append([]string{"Authorization", "Accept-Encoding", "Content-Length", "Host"}, hopHeaders...)

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
