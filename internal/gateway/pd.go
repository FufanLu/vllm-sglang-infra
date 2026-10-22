package gateway

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/FufanLu/vllm-sglang-infra/internal/backend"
)

// dispatchPD serves a request on a disaggregated pool. A prefill endpoint
// computes the prompt's KV cache, keeps it, and returns kv_transfer_params
// saying where to fetch it; a decode endpoint pulls the KV over NIXL and
// generates. The gateway chooses both endpoints and passes the parameters
// along without interpreting them, so it stays compatible as the connector
// protocol evolves.
//
// It returns false when the pool has no prefill endpoint, leaving the request
// to ordinary dispatch.
func (s *Server) dispatchPD(w http.ResponseWriter, r *http.Request, c *call) bool {
	sched := s.scheduler()
	pre := *c.sreq
	pre.Role = "prefill"
	epP, err := sched.Schedule(&pre, s.opts.Registry.List(), nil)
	if err != nil {
		return false
	}
	s.opts.Metrics.Picks.WithLabelValues(sched.Policy, epP.Name).Inc()
	t0 := time.Now()
	params, err := s.prefill(r.Context(), r, c, epP)
	epP.Release()
	c.prefill = time.Since(t0)
	s.opts.Metrics.PDPrefill.WithLabelValues(epP.Name).Observe(c.prefill.Seconds())

	body := c.body
	switch {
	case r.Context().Err() != nil:
		c.code = "client_cancel"
		return true
	case err != nil:
		// The decode endpoint can still prefill the prompt itself.
		s.opts.Metrics.Retries.WithLabelValues("pd_prefill_failed").Inc()
		s.opts.Log.Warn("prefill failed; decoding without transferred KV", "request_id", c.reqID,
			"prefill", epP.Name, "err", err)
	default:
		if b, err := rewriteJSON(c.body, func(m map[string]any) { m["kv_transfer_params"] = params }); err == nil {
			body = b
		}
	}
	dec := *c.sreq
	dec.Role = "decode"
	s.dispatch(w, r, c, &dec, func(*backend.Endpoint) []byte { return body })
	return true
}

// prefill runs the prompt on ep with max_tokens=1 and returns the
// kv_transfer_params from its response.
func (s *Server) prefill(ctx context.Context, r *http.Request, c *call, ep *backend.Endpoint) (json.RawMessage, error) {
	body, err := rewriteJSON(c.body, func(m map[string]any) {
		m["max_tokens"] = 1
		if _, ok := m["max_completion_tokens"]; ok {
			m["max_completion_tokens"] = 1
		}
		m["stream"] = false
		delete(m, "stream_options")
		m["kv_transfer_params"] = map[string]any{
			"do_remote_decode": true, "do_remote_prefill": false,
			"remote_engine_id": nil, "remote_block_ids": nil, "remote_host": nil, "remote_port": nil,
		}
	})
	if err != nil {
		return nil, err
	}
	up, err := s.upstreamRequest(ctx, r, c, ep, body)
	if err != nil {
		return nil, err
	}
	resp, err := s.opts.Client.Do(up)
	if err != nil {
		if ctx.Err() == nil && ep.MarkDown() {
			s.opts.Log.Warn("endpoint down", "endpoint", ep.Name, "err", err)
		}
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		io.Copy(io.Discard, io.LimitReader(resp.Body, 64<<10))
		return nil, fmt.Errorf("prefill on %s: status %d", ep.Name, resp.StatusCode)
	}
	var out struct {
		KVTransferParams json.RawMessage `json:"kv_transfer_params"`
	}
	if err := json.NewDecoder(io.LimitReader(resp.Body, 1<<20)).Decode(&out); err != nil {
		return nil, err
	}
	if len(out.KVTransferParams) == 0 || string(out.KVTransferParams) == "null" {
		return nil, errors.New("prefill response carries no kv_transfer_params")
	}
	return out.KVTransferParams, nil
}
