package streamframe

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/internal/codec"
)

// ReasoningFormat selects how model reasoning reaches the wire relative to answer content.
type ReasoningFormat uint8

const (
	// FormatSplit ("deepseek") emits clean answer content and puts reasoning in reasoning_content.
	FormatSplit ReasoningFormat = iota
	// FormatLegacy ("deepseek-legacy") fills reasoning_content while keeping raw think tags in content.
	FormatLegacy
	// FormatNone ("none") leaves raw text unsplit in content.
	FormatNone
)

// ParseReasoningFormat parses a -reasoning-format flag or per-request reasoning_format string.
func ParseReasoningFormat(s string) (ReasoningFormat, bool) {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "", "deepseek":
		return FormatSplit, true
	case "deepseek-legacy":
		return FormatLegacy, true
	case "none":
		return FormatNone, true
	}
	return FormatSplit, false
}

// ThinkSettings holds the resolved per-request thinking configuration.
type ThinkSettings struct {
	Budget   int
	Mode     chat.ThinkMode
	Explicit bool
	Format   ReasoningFormat
	Effort   string
}

// ThinkRequest carries the raw request-level thinking fields gathered across OpenAI and Anthropic dialects.
type ThinkRequest struct {
	Kwargs          map[string]json.RawMessage
	ReasoningEffort string
	AnthropicType   string
	Format          string
	Budget          int
}

// ResolveThink combines server default thinking/format settings with a request's overrides.
func ResolveThink(defaultMode chat.ThinkMode, defaultFormat ReasoningFormat, tr ThinkRequest) (ThinkSettings, error) {
	ts := ThinkSettings{Mode: defaultMode, Format: defaultFormat}
	if tr.Format != "" {
		f, ok := ParseReasoningFormat(tr.Format)
		if !ok {
			return ts, fmt.Errorf("reasoning_format %q: want deepseek, deepseek-legacy or none", tr.Format)
		}
		ts.Format = f
	}
	if tr.Budget < 0 {
		return ts, fmt.Errorf("the thinking token budget must be positive, got %d", tr.Budget)
	}
	ts.Budget = tr.Budget
	if e, ok := chat.NormalizeReasoningEffort(tr.ReasoningEffort); ok {
		ts.Effort = e
	}
	set := func(on bool) {
		ts.Explicit = true
		ts.Mode = chat.ThinkOff
		if on {
			ts.Mode = chat.ThinkOn
		}
	}
	if raw, ok := tr.Kwargs["enable_thinking"]; ok {
		var b bool
		if err := json.Unmarshal(raw, &b); err != nil {
			return ts, fmt.Errorf("chat_template_kwargs.enable_thinking must be a boolean")
		}
		set(b)
	} else if strings.EqualFold(strings.TrimSpace(tr.ReasoningEffort), "none") {
		set(false)
	} else if tr.AnthropicType != "" {
		switch tr.AnthropicType {
		case "enabled", "adaptive":
			set(true)
		case "disabled":
			set(false)
		default:
			return ts, fmt.Errorf("thinking.type %q: want enabled, adaptive or disabled", tr.AnthropicType)
		}
	}
	return ts, nil
}

// ThinkOut separates one generation's decoded text into reasoning fragments and answer text watched by stop-sequence logic.
type ThinkOut struct {
	rs               *chat.ReplySplitter
	onReasoning      func(string)
	raw              bool
	endedInReasoning bool
	reasoning        strings.Builder
}

// NewThinkOut constructs a ThinkOut for a generation rendered with tmpl and turns, or returns nil when no splitting applies.
func NewThinkOut(tmpl *chat.Template, turns []chat.Turn, ts ThinkSettings, onReasoning func(string)) *ThinkOut {
	if ts.Format == FormatNone {
		return nil
	}
	rs := codec.NewReplySplitter(tmpl, turns)
	if rs == nil {
		return nil
	}
	if onReasoning == nil {
		onReasoning = func(string) {}
	}
	return &ThinkOut{rs: rs, onReasoning: onReasoning, raw: ts.Format == FormatLegacy}
}

// SetOnReasoning replaces o's reasoning callback.
func (o *ThinkOut) SetOnReasoning(fn func(string)) {
	if o == nil {
		return
	}
	if fn == nil {
		fn = func(string) {}
	}
	o.onReasoning = fn
}

// WrapOnReasoning replaces o's reasoning callback with wrap(prev), allowing generation drivers to count emitted reasoning chunks.
func (o *ThinkOut) WrapOnReasoning(wrap func(prev func(string)) func(string)) {
	if o == nil {
		return
	}
	o.onReasoning = wrap(o.onReasoning)
}

// Feed pushes one decoded text piece through the splitter, emits any reasoning fragment, and returns the text stop-string matching should inspect.
func (o *ThinkOut) Feed(piece string) string {
	r, c := o.rs.Push(piece)
	if r != "" {
		o.reasoning.WriteString(r)
		o.onReasoning(r)
	}
	if o.raw {
		return piece
	}
	return c
}

// Finish flushes any buffered splitter tail at the end of a turn and records whether the turn ended inside an unclosed reasoning block.
func (o *ThinkOut) Finish() string {
	r, c, truncated := o.rs.Finish()
	o.endedInReasoning = truncated
	if r != "" {
		o.reasoning.WriteString(r)
		o.onReasoning(r)
	}
	if o.raw {
		return ""
	}
	return c
}

// ReasoningText returns all reasoning text accumulated so far, or "" for a nil receiver.
func (o *ThinkOut) ReasoningText() string {
	if o == nil {
		return ""
	}
	return o.reasoning.String()
}

// ReasoningLen returns the byte length of accumulated reasoning text, or 0 for a nil receiver.
func (o *ThinkOut) ReasoningLen() int {
	if o == nil {
		return 0
	}
	return o.reasoning.Len()
}

// EndedInReasoning reports whether Finish observed an unclosed reasoning block.
func (o *ThinkOut) EndedInReasoning() bool {
	return o != nil && o.endedInReasoning
}

// ToolCalls returns Harmony-routed tool calls extracted by the underlying ReplySplitter, or nil when none exist.
func (o *ThinkOut) ToolCalls() []chat.ToolCall {
	if o == nil {
		return nil
	}
	return o.rs.ToolCalls()
}
