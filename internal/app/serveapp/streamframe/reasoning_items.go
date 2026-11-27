package streamframe

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"
	"unicode/utf8"

	"github.com/townsendmerino/goinfer/internal/app/serveapp/carriedids"
)

// ReasoningBlobPrefix marks a stateless v1 base64-encoded reasoning encrypted_content value produced by goinfer.
const ReasoningBlobPrefix = "goinfer-reasoning-v1:"

// ReasoningItem builds a /v1/responses reasoning output item with text and encrypted_content enc (falling back to the v1 base64 blob when enc is empty).
func ReasoningItem(id, text, enc string) map[string]any {
	if enc == "" {
		enc = ReasoningBlobPrefix + base64.StdEncoding.EncodeToString([]byte(text))
	}
	content := []any{}
	if text != "" {
		content = []any{map[string]any{"type": "reasoning_text", "text": text}}
	}
	return map[string]any{"type": "reasoning", "id": id, "summary": []any{}, "content": content, "encrypted_content": enc}
}

// OutputMessage builds a completed /v1/responses assistant message output item.
func OutputMessage(itemID, text string) map[string]any {
	return map[string]any{
		"type": "message", "id": itemID, "role": "assistant", "status": "completed",
		"content": []any{map[string]any{"type": "output_text", "text": text, "annotations": []any{}}},
	}
}

// ReasoningItemText extracts replayed reasoning text from a client-supplied /v1/responses reasoning item.
func ReasoningItemText(encrypted string, content, summary json.RawMessage) (string, error) {
	text, _, err := ReasoningItemCarried(encrypted, content, summary)
	return text, err
}

// ReasoningItemCarried extracts both the replayed reasoning text and any carried token-ID carrier from a /v1/responses reasoning input item.
func ReasoningItemCarried(encrypted string, content, summary json.RawMessage) (string, *carriedids.Carrier, error) {
	if strings.HasPrefix(encrypted, carriedids.V2Prefix) {
		b, err := carriedids.ParseBlob(encrypted)
		if err != nil {
			text, _ := ReasoningItemText("", content, summary)
			return text, &carriedids.Carrier{Reason: carriedids.ReasonParse}, nil
		}
		return b.Text, &carriedids.Carrier{Blob: b}, nil
	}
	if b64, ok := strings.CutPrefix(encrypted, ReasoningBlobPrefix); ok {
		b, err := base64.StdEncoding.DecodeString(b64)
		if err != nil || !utf8.Valid(b) {
			return "", nil, fmt.Errorf("a reasoning item's encrypted_content is not one this server wrote (%s… does not decode); send it back unchanged", ReasoningBlobPrefix)
		}
		return string(b), &carriedids.Carrier{Reason: carriedids.ReasonV1}, nil
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

// WithReasoning prepends the reply's reasoning output item (from rs when streaming, else from th) ahead of items.
func WithReasoning(rs *RespStream, th *ThinkOut, id string, mintBlob func(string) string, items []any) []any {
	var it map[string]any
	if rs != nil {
		it = rs.CloseReasoning()
	} else if th != nil && th.ReasoningLen() > 0 {
		text := th.ReasoningText()
		enc := ""
		if mintBlob != nil {
			enc = mintBlob(text)
		}
		it = ReasoningItem(id+"-rs", text, enc)
	}
	if it == nil {
		return items
	}
	return append([]any{it}, items...)
}
