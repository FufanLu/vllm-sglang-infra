// Package tokenize asks a model server for a request's prompt token IDs, so
// the gateway sees exactly the tokens the engine will cache: the chat
// template applied, with the engine's own tokenizer. vLLM and
// llm-d-inference-sim both serve POST /tokenize.
package tokenize

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

type Client struct {
	HTTP *http.Client
}

// Tokens returns the prompt tokens for a chat request (messages set) or a
// completion request (prompt set), as the engine at baseURL tokenizes them.
func (c *Client) Tokens(ctx context.Context, baseURL, model string, messages, prompt json.RawMessage) ([]uint32, error) {
	req := map[string]any{"model": model}
	if len(messages) > 0 {
		req["messages"] = messages
		req["add_generation_prompt"] = true
	} else {
		req["prompt"] = prompt
	}
	body, _ := json.Marshal(req)
	hreq, err := http.NewRequestWithContext(ctx, http.MethodPost, baseURL+"/tokenize", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	hreq.Header.Set("Content-Type", "application/json")
	resp, err := c.HTTP.Do(hreq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		io.Copy(io.Discard, io.LimitReader(resp.Body, 64<<10))
		return nil, fmt.Errorf("tokenize: status %d", resp.StatusCode)
	}
	var out struct {
		Tokens []uint32 `json:"tokens"`
	}
	if err := json.NewDecoder(io.LimitReader(resp.Body, 64<<20)).Decode(&out); err != nil {
		return nil, fmt.Errorf("tokenize: %w", err)
	}
	return out.Tokens, nil
}
