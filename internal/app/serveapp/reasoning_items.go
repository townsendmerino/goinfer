package serveapp

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"
	"unicode/utf8"
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

// reasoningItem is a reasoning output item carrying text, with encrypted_content enc (a carried-id blob, carried_ids.go), or
// the v1 text blob when enc is "". An item with no text is a carrier-only item: empty content, the blob alone.
func reasoningItem(id, text, enc string) map[string]any {
	if enc == "" {
		enc = reasoningBlobPrefix + base64.StdEncoding.EncodeToString([]byte(text))
	}
	content := []any{}
	if text != "" {
		content = []any{map[string]any{"type": "reasoning_text", "text": text}}
	}
	return map[string]any{"type": "reasoning", "id": id, "summary": []any{}, "content": content, "encrypted_content": enc}
}

// reasoningItemText is the reasoning a client's reasoning input item carries: this server's encrypted_content, else the
// reasoning_text content, else the summary text. encrypted_content wins when it and content disagree: it is the bytes the
// model generated, and content may have been edited. "" when the item carries none (an encrypted_content from another
// server is not readable here and is ignored). An encrypted_content with this server's prefix that does not decode is an
// error, so the request is refused rather than rendered without the reasoning it claims to carry.
func reasoningItemText(encrypted string, content, summary json.RawMessage) (string, error) {
	text, _, err := reasoningItemCarried(encrypted, content, summary)
	return text, err
}

// reasoningItemCarried is reasoningItemText that also returns the carried-id blob the item carries for the assistant turn it
// attaches to: a v2 blob parsed (its text is the item's text), a v1 blob as reason v1, a v2 blob that does not parse as reason
// parse (and the item's text is then read as if it carried no blob). nil when the item carries no blob of this server's.
func reasoningItemCarried(encrypted string, content, summary json.RawMessage) (string, *idsCarrier, error) {
	if strings.HasPrefix(encrypted, reasoningBlobV2Prefix) {
		b, err := parseIDsBlob(encrypted)
		if err != nil {
			text, _ := reasoningItemText("", content, summary)
			return text, &idsCarrier{reason: idsParse}, nil
		}
		return b.text, &idsCarrier{blob: b}, nil
	}
	if b64, ok := strings.CutPrefix(encrypted, reasoningBlobPrefix); ok {
		b, err := base64.StdEncoding.DecodeString(b64)
		if err != nil || !utf8.Valid(b) {
			return "", nil, fmt.Errorf("a reasoning item's encrypted_content is not one this server wrote (%s… does not decode); send it back unchanged", reasoningBlobPrefix)
		}
		return string(b), &idsCarrier{reason: idsV1}, nil
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
		return t, nil, nil
	}
	return join(summary, "summary_text"), nil, nil
}

// withReasoning puts the reply's reasoning item ahead of items: the item the stream opened (rs non-nil), else one built
// from the reasoning th kept, its blob from mint (nil: the v1 text blob). items unchanged when the reply did not reason.
func withReasoning(rs *respStream, th *thinkOut, id string, mint *carriedMint, items []any) []any {
	var it map[string]any
	if rs != nil {
		it = rs.closeReasoning()
	} else if th != nil && th.reasoning.Len() > 0 {
		text := th.reasoning.String()
		it = reasoningItem(id+"-rs", text, mint.blob(text))
	}
	if it == nil {
		return items
	}
	return append([]any{it}, items...)
}
