package serveapp

import (
	"encoding/base64"
	"encoding/json"
	"strings"
)

// Reasoning items on /v1/responses (docs/server.md, "Reasoning items on /v1/responses").
//
// A reply's reasoning is returned as a `reasoning` output item ahead of the message and calls: its text in `content`
// (reasoning_text) and, byte for byte, in `encrypted_content`, an opaque value for the client to send back. Clients that
// keep state send reasoning items back in `input`; responseInputToMessages attaches the text to the assistant turn that
// follows the item, as replayed reasoning, and the model's own template decides where it renders (chat/history.go).
// encrypted_content is read first because a client may send back `content` altered or not at all, and the exact bytes are
// what keep the next prompt a prefix extension of the tokens the model generated: dropped or altered reasoning makes the
// prompt leave the KV at the reasoning, and the turn re-prefills from there.

// reasoningBlobPrefix marks an encrypted_content this server wrote. It is not encryption: the value is the reasoning,
// base64, so the server needs no state to read it back.
const reasoningBlobPrefix = "goinfer-reasoning-v1:"

// reasoningItem is a reasoning output item carrying text.
func reasoningItem(id, text string) map[string]any {
	return map[string]any{
		"type": "reasoning", "id": id, "summary": []any{},
		"content":           []any{map[string]any{"type": "reasoning_text", "text": text}},
		"encrypted_content": reasoningBlobPrefix + base64.StdEncoding.EncodeToString([]byte(text)),
	}
}

// reasoningItemText is the reasoning a client's reasoning input item carries: this server's encrypted_content, else the
// reasoning_text content, else the summary text. "" when it carries none (an encrypted_content from another server is
// not readable here and is ignored).
func reasoningItemText(encrypted string, content, summary json.RawMessage) string {
	if b64, ok := strings.CutPrefix(encrypted, reasoningBlobPrefix); ok {
		if b, err := base64.StdEncoding.DecodeString(b64); err == nil {
			return string(b)
		}
	}
	var parts []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	join := func(raw json.RawMessage, typ string) string {
		parts = parts[:0]
		if json.Unmarshal(raw, &parts) != nil {
			return ""
		}
		var b strings.Builder
		for _, p := range parts {
			if p.Type == typ {
				b.WriteString(p.Text)
			}
		}
		return b.String()
	}
	if t := join(content, "reasoning_text"); t != "" {
		return t
	}
	return join(summary, "summary_text")
}

// withReasoning puts the reply's reasoning item ahead of items: the item the stream opened (rs non-nil), else one built
// from the reasoning th kept. items unchanged when the reply did not reason.
func withReasoning(rs *respStream, th *thinkOut, id string, items []any) []any {
	var it map[string]any
	if rs != nil {
		it = rs.closeReasoning()
	} else if th != nil && th.reasoning.Len() > 0 {
		it = reasoningItem(id+"-rs", th.reasoning.String())
	}
	if it == nil {
		return items
	}
	return append([]any{it}, items...)
}
