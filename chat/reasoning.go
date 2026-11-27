package chat

import "github.com/townsendmerino/goinfer/internal/chat/reasoning"

// ThinkMode controls per-request thinking scaffold rendering.
type ThinkMode = reasoning.ThinkMode

const (
	ThinkAsIs     = reasoning.ThinkAsIs
	ThinkTemplate = reasoning.ThinkTemplate
	ThinkOn       = reasoning.ThinkOn
	ThinkOff      = reasoning.ThinkOff
)

// ParseThinkMode parses a thinking-mode string ("" and "asis" map to ThinkAsIs).
var ParseThinkMode = reasoning.ParseThinkMode

// Reasoning describes one checkpoint's declared prompt and output thinking protocol.
type Reasoning struct {
	*reasoning.Spec
}

func wrapReasoning(s *reasoning.Spec) *Reasoning {
	if s == nil {
		return nil
	}
	return &Reasoning{Spec: s}
}

func newHistReasoning(h histKind) *Reasoning {
	return &Reasoning{Spec: &reasoning.Spec{Hist: h}}
}

// Open returns the opening delimiter a reply uses around its reasoning.
func (r *Reasoning) Open() string { return r.OpenText }

// Close returns the closing delimiter a reply uses around its reasoning.
func (r *Reasoning) Close() string { return r.CloseText }

// OpenToken returns the single-token text that opens the reasoning block.
func (r *Reasoning) OpenToken() string { return r.OpenTok }

// CloseToken returns the single-token text that closes the reasoning block.
func (r *Reasoning) CloseToken() string { return r.CloseTok }

// DefaultOn reports whether the checkpoint's own template enables thinking when enable_thinking is unset.
func (r *Reasoning) DefaultOn() bool { return r.DefaultIsOn }

func detectChatMLReasoning(tmpl string) *Reasoning {
	return wrapReasoning(reasoning.DetectChatML(tmpl))
}

func detectGemma4Reasoning(tmpl string) *Reasoning {
	return wrapReasoning(reasoning.DetectGemma4(tmpl))
}

func (r *Reasoning) offScaffold() bool {
	if r == nil {
		return true
	}
	return r.OffScaffold()
}

func (r *Reasoning) suffix(m ThinkMode) []Segment {
	return r.Suffix(m)
}

// WithThinking returns a copy of t rendering thinking mode m, or t unchanged when unmanaged.
func (t *Template) WithThinking(m ThinkMode) *Template {
	if t != nil && t.name == "inkling2" && m != t.think {
		return t.inklingWith(t.effort, m)
	}
	if t == nil || t.reason == nil || m == t.think {
		return t
	}
	c := *t
	c.think = m
	return &c
}

// Reasoning returns the family's reasoning spec for this checkpoint, or nil when unmanaged.
func (t *Template) Reasoning() *Reasoning {
	if t == nil {
		return nil
	}
	return t.reason
}

// ThinkMode returns the thinking mode active on t.
func (t *Template) ThinkMode() ThinkMode {
	if t == nil {
		return ThinkAsIs
	}
	return t.think
}

// PromptOpensThink reports whether the generation prompt ends inside an open think block.
func (t *Template) PromptOpensThink() bool {
	if t == nil || t.reason == nil {
		return false
	}
	on, ok := t.reason.EffectiveOn(t.think)
	return ok && on && t.reason.OnOpens
}

// ThinkingPossible reports whether a reply to a prompt from t may contain reasoning.
func (t *Template) ThinkingPossible() bool {
	if t != nil && t.name == "harmony" {
		return true
	}
	if t != nil && t.name == "inkling2" {
		return t.think != ThinkOff
	}
	if t == nil || t.reason == nil {
		return false
	}
	on, ok := t.reason.EffectiveOn(t.think)
	return !ok || on
}

// PromptOpensThinkFor reports whether the generation prompt for turns ends inside an open think block.
func (t *Template) PromptOpensThinkFor(turns []Turn) bool {
	if t.PromptOpensThink() {
		return true
	}
	if t == nil || t.reason == nil || !t.reason.Gemma4 || len(turns) == 0 || turns[len(turns)-1].Role != "tool" {
		return false
	}
	on, ok := t.reason.EffectiveOn(t.think)
	return ok && on
}

// NewReasoningSplitter returns a delimiter splitter for replies to prompts from t, or nil when unmanaged.
func (t *Template) NewReasoningSplitter(turns []Turn) *ThinkSplitter {
	if t == nil || t.reason == nil {
		return nil
	}
	return NewThinkSplitter(t.reason.OpenText, t.reason.CloseText, t.PromptOpensThinkFor(turns))
}

func (t *Template) renderThinking(system string, turns []Turn) []Segment {
	r := t.reason
	if r.Gemma4 {
		on, _ := r.EffectiveOn(t.think)
		return gemma4Segments(system, turns, gemma4Opts{think: on, hist: r.Hist == histGemma4 || r.OldGemma4, old: r.OldGemma4, scaffold: r.offScaffold()})
	}
	if t.name == "flint" {
		return t.render(system, turns)
	}
	segs := chatMLSegments(system, turns, r.Hist)
	return append(segs[:len(segs):len(segs)], r.suffix(t.think)...)
}

// ThinkSplitter incrementally separates a delimited reply into reasoning and content.
type ThinkSplitter struct {
	*reasoning.ThinkSplitter
}

// NewThinkSplitter returns a splitter for open and close delimiters.
func NewThinkSplitter(open, close string, forcedOpen bool) *ThinkSplitter {
	return &ThinkSplitter{ThinkSplitter: reasoning.NewThinkSplitter(open, close, forcedOpen)}
}

// Push feeds the next decoded chunk and returns reasoning and content safe to emit immediately.
func (s *ThinkSplitter) Push(chunk string) (reasoning, content string) {
	return s.ThinkSplitter.Push(chunk)
}

// SplitThink splits a complete reply into reasoning and content.
func (t *Template) SplitThink(reply string) (reasoning, content string) {
	ts := t.NewReasoningSplitter(nil)
	if ts == nil {
		return "", reply
	}
	r, c := ts.Push(reply)
	r2, c2 := ts.Flush()
	return r + r2, c + c2
}

func (t *Template) thinkSuffixSegs() []Segment {
	if t.reason == nil || t.reason.Gemma4 {
		return nil
	}
	return t.reason.suffix(t.think)
}

func (t *Template) gemma4Think() bool {
	if t.reason == nil || !t.reason.Gemma4 {
		return false
	}
	on, ok := t.reason.EffectiveOn(t.think)
	return ok && on
}
