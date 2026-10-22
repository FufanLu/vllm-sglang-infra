package gateway

import (
	"encoding/json"
	"strings"
)

// inferenceRequest holds the fields of an OpenAI-style request body that the
// gateway reads. The body itself is forwarded byte for byte.
type inferenceRequest struct {
	Model    string          `json:"model"`
	Stream   bool            `json:"stream"`
	Messages []chatMessage   `json:"messages"` // /v1/chat/completions
	Prompt   json.RawMessage `json:"prompt"`   // /v1/completions
}

type chatMessage struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"`
}

// promptText flattens the prompt into one string, in order. Requests that
// share a prefix of messages (a system prompt, earlier turns of the same
// conversation) produce strings that share a prefix, which is what
// prefix-aware routing keys on. It is not the engine's tokenized prompt.
func (r *inferenceRequest) promptText() string {
	var b strings.Builder
	for _, m := range r.Messages {
		b.WriteString(m.Role)
		b.WriteByte('\n')
		b.WriteString(contentText(m.Content))
		b.WriteByte('\n')
	}
	if len(r.Prompt) > 0 {
		var s string
		var ss []string
		switch {
		case json.Unmarshal(r.Prompt, &s) == nil:
			b.WriteString(s)
		case json.Unmarshal(r.Prompt, &ss) == nil:
			b.WriteString(strings.Join(ss, "\n"))
		}
		// Pre-tokenized prompts (arrays of token ids) carry no text to key on.
	}
	return b.String()
}

// contentText extracts the text of a chat message content, which is either a
// string or an array of parts such as {"type":"text","text":"..."}.
func contentText(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return s
	}
	var parts []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if json.Unmarshal(raw, &parts) != nil {
		return ""
	}
	var b strings.Builder
	for _, p := range parts {
		if p.Type == "text" {
			b.WriteString(p.Text)
		}
	}
	return b.String()
}
