package streamframe

import "strings"

// AnthropicThink manages the index-0 `thinking` content block on an Anthropic /v1/messages SSE stream.
// When want is false, reasoning fragments are discarded without opening a block.
type AnthropicThink struct {
	ss       *SSEWriter
	want     bool
	open     bool
	used     bool
	mintBlob func(string) string
	text     strings.Builder
}

// NewAnthropicThink constructs an Anthropic thinking-block stream helper. mintBlob may be nil when carried-id signing is off.
func NewAnthropicThink(ss *SSEWriter, want bool, mintBlob func(string) string) *AnthropicThink {
	return &AnthropicThink{ss: ss, want: want, mintBlob: mintBlob}
}

// Push writes a thinking_delta fragment, lazily opening the thinking block at index 0 on the first fragment.
func (t *AnthropicThink) Push(text string) {
	if t == nil || !t.want || text == "" {
		return
	}
	if !t.open {
		SendEvent(t.ss, "content_block_start", map[string]any{
			"type": "content_block_start", "index": 0,
			"content_block": map[string]any{"type": "thinking", "thinking": "", "signature": ""},
		})
		t.open, t.used = true, true
	}
	t.text.WriteString(text)
	SendEvent(t.ss, "content_block_delta", map[string]any{
		"type": "content_block_delta", "index": 0,
		"delta": map[string]any{"type": "thinking_delta", "thinking": text},
	})
}

// Close emits signature_delta and content_block_stop when the thinking block is open. Idempotent.
func (t *AnthropicThink) Close() {
	if t == nil || !t.open {
		return
	}
	sig := ""
	if t.mintBlob != nil {
		sig = t.mintBlob(t.text.String())
	}
	SendEvent(t.ss, "content_block_delta", map[string]any{
		"type": "content_block_delta", "index": 0,
		"delta": map[string]any{"type": "signature_delta", "signature": sig},
	})
	SendEvent(t.ss, "content_block_stop", map[string]any{"type": "content_block_stop", "index": 0})
	t.open = false
}

// CarrierOnly emits cb (a redacted_thinking carrier block) at index 0 when thinking was requested but no reasoning occurred.
func (t *AnthropicThink) CarrierOnly(cb map[string]any) {
	if t == nil || !t.want || t.used || cb == nil {
		return
	}
	SendEvent(t.ss, "content_block_start", map[string]any{"type": "content_block_start", "index": 0, "content_block": cb})
	SendEvent(t.ss, "content_block_stop", map[string]any{"type": "content_block_stop", "index": 0})
	t.used = true
}

// Offset returns the block index for the first non-thinking block (1 if a thinking block was emitted, else 0).
func (t *AnthropicThink) Offset() int {
	if t != nil && t.used {
		return 1
	}
	return 0
}

// Used reports whether a thinking or carrier block has been emitted.
func (t *AnthropicThink) Used() bool { return t != nil && t.used }

// AnthropicTextStream manages the lazy `text` content block on an Anthropic /v1/messages SSE stream.
type AnthropicTextStream struct {
	ss       *SSEWriter
	think    *AnthropicThink
	open     bool
	everOpen bool
	pending  strings.Builder
}

// NewAnthropicTextStream constructs an Anthropic text-block stream helper ordered after think.
func NewAnthropicTextStream(ss *SSEWriter, think *AnthropicThink) *AnthropicTextStream {
	return &AnthropicTextStream{ss: ss, think: think}
}

func (a *AnthropicTextStream) idx() int { return a.think.Offset() }

// Open reports whether the text block is currently open.
func (a *AnthropicTextStream) Open() bool { return a.open }

// EverOpen reports whether PushPlain opened a text block at any point during the turn.
func (a *AnthropicTextStream) EverOpen() bool { return a.everOpen }

// Push buffers leading whitespace until non-whitespace prose arrives, then closes think, opens the text block, and streams deltas.
func (a *AnthropicTextStream) Push(text string) {
	if !a.open {
		a.pending.WriteString(text)
		if strings.TrimSpace(a.pending.String()) == "" {
			return
		}
		a.think.Close()
		SendEvent(a.ss, "content_block_start", map[string]any{
			"type": "content_block_start", "index": a.idx(),
			"content_block": map[string]any{"type": "text", "text": ""},
		})
		a.open = true
		text = a.pending.String()
		a.pending.Reset()
	}
	SendEvent(a.ss, "content_block_delta", map[string]any{
		"type": "content_block_delta", "index": a.idx(),
		"delta": map[string]any{"type": "text_delta", "text": text},
	})
}

// PushPlain streams text immediately without holding leading whitespace, closing think on the first byte.
func (a *AnthropicTextStream) PushPlain(text string) {
	if !a.open {
		a.think.Close()
		SendEvent(a.ss, "content_block_start", map[string]any{
			"type": "content_block_start", "index": a.idx(),
			"content_block": map[string]any{"type": "text", "text": ""},
		})
		a.open, a.everOpen = true, true
	}
	SendEvent(a.ss, "content_block_delta", map[string]any{
		"type": "content_block_delta", "index": a.idx(),
		"delta": map[string]any{"type": "text_delta", "text": text},
	})
}

// Finish flushes any held prose plus rest, closes the text block, and returns the next available content-block index.
func (a *AnthropicTextStream) Finish(rest string, always bool) int {
	if a.open {
		if rest != "" {
			a.Push(rest)
		}
		a.Close()
		return a.idx() + 1
	}
	text := a.pending.String() + rest
	if always || strings.TrimSpace(text) != "" {
		a.think.Close()
		StreamTextBlock(a.ss, a.idx(), text)
		return a.idx() + 1
	}
	return a.idx()
}

// Close emits content_block_stop if the text block is open.
func (a *AnthropicTextStream) Close() {
	if a.open {
		SendEvent(a.ss, "content_block_stop", map[string]any{"type": "content_block_stop", "index": a.idx()})
		a.open = false
	}
}

// StreamTextBlock emits a complete Anthropic text content block (start, optional delta, stop) at index.
func StreamTextBlock(ss *SSEWriter, index int, text string) {
	SendEvent(ss, "content_block_start", map[string]any{
		"type": "content_block_start", "index": index,
		"content_block": map[string]any{"type": "text", "text": ""},
	})
	if text != "" {
		SendEvent(ss, "content_block_delta", map[string]any{
			"type": "content_block_delta", "index": index,
			"delta": map[string]any{"type": "text_delta", "text": text},
		})
	}
	SendEvent(ss, "content_block_stop", map[string]any{"type": "content_block_stop", "index": index})
}
