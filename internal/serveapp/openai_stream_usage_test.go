package serveapp

import (
	"encoding/json"
	"strings"
	"testing"
)

// A streaming client has no token count unless the server sends one, and counting SSE chunks is not
// a substitute: streamTokens emits a chunk only when `end > printed`, so a token held back for an
// incomplete UTF-8 rune or a partial stop-string match produces NO chunk, and the token that
// resolves the holdback produces one chunk carrying several tokens' bytes. Chunks <= tokens, always
// in the same direction. bench_peer.py counted chunks and called them tokens.
func TestStreamOptions_includeUsageShape(t *testing.T) {
	got := usageChunk("chatcmpl-x", 1234, "m", usage{PromptTokens: 7, CompletionTokens: 11, TotalTokens: 18})

	if got["object"] != "chat.completion.chunk" {
		t.Errorf("object = %v, want chat.completion.chunk", got["object"])
	}
	// OpenAI's shape: choices is present and EMPTY on the usage chunk, so a client iterating
	// choices sees nothing new and only a usage-aware client reads the counts.
	ch, ok := got["choices"].([]any)
	if !ok || len(ch) != 0 {
		t.Errorf("choices = %#v, want an empty array", got["choices"])
	}
	u, ok := got["usage"].(usage)
	if !ok || u.CompletionTokens != 11 || u.PromptTokens != 7 || u.TotalTokens != 18 {
		t.Errorf("usage = %#v, want 7/11/18", got["usage"])
	}
	// It must survive JSON round-tripping as the client will read it.
	b, err := json.Marshal(got)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if !strings.Contains(string(b), `"completion_tokens":11`) {
		t.Errorf("marshalled chunk lacks completion_tokens: %s", b)
	}
}

// The field must parse from the wire in OpenAI's nested shape, and stay absent (nil) when the
// client does not ask — a non-nil zero value would be indistinguishable from include_usage:false
// only by luck.
func TestStreamOptions_parsing(t *testing.T) {
	var withOpt chatReq
	if err := json.Unmarshal([]byte(`{"model":"m","stream":true,"stream_options":{"include_usage":true}}`), &withOpt); err != nil {
		t.Fatal(err)
	}
	if withOpt.StreamOptions == nil || !withOpt.StreamOptions.IncludeUsage {
		t.Errorf("include_usage did not parse: %#v", withOpt.StreamOptions)
	}
	var without chatReq
	if err := json.Unmarshal([]byte(`{"model":"m","stream":true}`), &without); err != nil {
		t.Fatal(err)
	}
	if without.StreamOptions != nil {
		t.Errorf("absent stream_options should stay nil, got %#v", without.StreamOptions)
	}
}
