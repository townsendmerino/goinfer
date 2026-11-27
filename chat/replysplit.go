package chat

import (
	"github.com/townsendmerino/goinfer/internal/chat/harmony"
	"github.com/townsendmerino/goinfer/internal/chat/inkling2"
	"github.com/townsendmerino/goinfer/internal/chat/reasoning"
)

// ReplySplitter separates a streamed reply into UTF-8 rune-safe reasoning, raw answer text, and protocol tool calls.
type ReplySplitter struct {
	*reasoning.ReplySplitter
}

// NewReplySplitter returns a stream splitter for replies to prompts rendered by t, or nil when unmanaged.
func (t *Template) NewReplySplitter(turns []Turn) *ReplySplitter {
	if t != nil && t.name == "harmony" {
		return &ReplySplitter{ReplySplitter: reasoning.NewReplySplitter(harmony.NewSplitter())}
	}
	if t != nil && t.name == "inkling2" {
		return &ReplySplitter{ReplySplitter: reasoning.NewReplySplitter(inkling2.NewSplitter())}
	}
	sp := t.NewReasoningSplitter(turns)
	if sp == nil {
		return nil
	}
	return &ReplySplitter{ReplySplitter: reasoning.NewReplySplitter(sp)}
}

// ToolCalls returns structured tool calls collected by a Harmony or Inkling 2 reply splitter.
func (r *ReplySplitter) ToolCalls() []ToolCall {
	if r == nil {
		return nil
	}
	return fromCoreCalls(r.CoreCalls())
}
