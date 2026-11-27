package reasoning

import (
	"strings"

	"github.com/townsendmerino/goinfer/internal/chat/chatcore"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// ThinkMode controls per-request thinking scaffold rendering.
type ThinkMode uint8

const (
	// ThinkAsIs renders the family's generic generation prompt unchanged.
	ThinkAsIs ThinkMode = iota
	// ThinkTemplate renders what the checkpoint's own template renders with enable_thinking unset.
	ThinkTemplate
	// ThinkOn renders enable_thinking=true.
	ThinkOn
	// ThinkOff renders enable_thinking=false.
	ThinkOff
)

func (m ThinkMode) String() string {
	switch m {
	case ThinkTemplate:
		return "template"
	case ThinkOn:
		return "on"
	case ThinkOff:
		return "off"
	}
	return "asis"
}

// ParseThinkMode parses a thinking-mode string ("" and "asis" map to ThinkAsIs).
func ParseThinkMode(s string) (ThinkMode, bool) {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "", "asis":
		return ThinkAsIs, true
	case "template":
		return ThinkTemplate, true
	case "on":
		return ThinkOn, true
	case "off":
		return ThinkOff, true
	}
	return ThinkAsIs, false
}

// Spec describes one checkpoint's prompt and output thinking protocol.
type Spec struct {
	OpenText, CloseText string
	OpenTok, CloseTok   string
	DefaultIsOn         bool
	Hist                chatcore.HistKind
	OnSuffix, OffSuffix []tokenizer.Segment
	OnOpens             bool
	Gemma4              bool
	OldGemma4           bool
	BareOff             bool
}

// NewFlintSpec returns Flint's default-on <think>...</think> reasoning spec.
func NewFlintSpec() *Spec {
	return &Spec{
		OpenText:    "<think>",
		CloseText:   "</think>",
		OpenTok:     "<think>",
		CloseTok:    "</think>",
		DefaultIsOn: true,
	}
}

func special(s string) tokenizer.Segment { return tokenizer.Segment{Text: s, Special: true} }
func plain(s string) tokenizer.Segment   { return tokenizer.Segment{Text: s} }

var (
	closedSegs = []tokenizer.Segment{special("<think>"), plain("\n\n"), special("</think>"), plain("\n\n")}
	openSegs   = []tokenizer.Segment{special("<think>"), plain("\n")}
)

// DetectChatML inspects a ChatML template's generation-prompt block and returns its thinking spec, or nil if unmanaged.
func DetectChatML(tmpl string) *Spec {
	i := strings.LastIndex(tmpl, "add_generation_prompt")
	if i < 0 {
		return nil
	}
	tail := tmpl[i:]
	if !strings.Contains(tail, "enable_thinking") {
		return nil
	}
	const closedLit = `'<think>\n\n</think>\n\n'`
	const openLit = `'<think>\n'`
	hasClosed := strings.Contains(tail, closedLit)
	hasOpen := strings.Contains(tail, openLit)
	isFalse := strings.Contains(tail, "enable_thinking is defined and enable_thinking is false")
	isTrue := strings.Contains(tail, "enable_thinking is defined and enable_thinking is true")
	r := &Spec{OpenText: "<think>", CloseText: "</think>", OpenTok: "<think>", CloseTok: "</think>"}
	switch {
	case isFalse && hasClosed && !hasOpen && !isTrue:
		r.DefaultIsOn, r.OnSuffix, r.OffSuffix, r.OnOpens = true, nil, closedSegs, false
	case isTrue && hasClosed && hasOpen && !isFalse:
		r.DefaultIsOn, r.OnSuffix, r.OffSuffix, r.OnOpens = false, openSegs, closedSegs, true
	case isFalse && hasClosed && hasOpen && !isTrue:
		r.DefaultIsOn, r.OnSuffix, r.OffSuffix, r.OnOpens = true, openSegs, closedSegs, true
	default:
		return nil
	}
	r.Hist = chatcore.DetectHistoryKind(tmpl)
	return r
}

// DetectGemma4 inspects a Gemma 4 template and returns its thinking spec, or nil if unmanaged.
func DetectGemma4(tmpl string) *Spec {
	if !strings.Contains(tmpl, "<|think|>") {
		return nil
	}
	if r := detectOldGemma4(tmpl); r != nil {
		return r
	}
	if !strings.Contains(tmpl, "enable_thinking | default(false)") {
		return nil
	}
	hist := chatcore.DetectGemma4History(tmpl)
	scaffold := strings.Contains(tmpl, "<|channel>thought\\n<channel|>")
	if !scaffold && hist != chatcore.HistGemma4 {
		return nil
	}
	return &Spec{
		OpenText:    "<|channel>thought\n",
		CloseText:   "<channel|>",
		OpenTok:     "<|channel>",
		CloseTok:    "<channel|>",
		DefaultIsOn: false,
		Gemma4:      true,
		Hist:        hist,
		BareOff:     !scaffold,
	}
}

func detectOldGemma4(tmpl string) *Spec {
	g := strings.LastIndex(tmpl, "add_generation_prompt")
	if g < 0 || !strings.Contains(tmpl, "enable_thinking is defined and enable_thinking") || strings.Contains(tmpl, "default(false)") ||
		strings.Contains(tmpl[g:], "<channel|>") {
		return nil
	}
	return &Spec{
		OpenText:    "<|channel>thought\n",
		CloseText:   "<channel|>",
		OpenTok:     "<|channel>",
		CloseTok:    "<channel|>",
		DefaultIsOn: false,
		Gemma4:      true,
		OldGemma4:   true,
	}
}

// OffScaffold reports whether a Gemma 4 template writes the closed thought scaffold when thinking is off.
func (r *Spec) OffScaffold() bool { return r == nil || !r.OldGemma4 && !r.BareOff }

// EffectiveOn resolves mode m to on/off for r (ok=false for ThinkAsIs).
func (r *Spec) EffectiveOn(m ThinkMode) (on, ok bool) {
	switch m {
	case ThinkOn:
		return true, true
	case ThinkOff:
		return false, true
	case ThinkTemplate:
		return r.DefaultIsOn, true
	}
	return false, false
}

// Suffix returns the generation-prompt segments appended for mode m on ChatML families.
func (r *Spec) Suffix(m ThinkMode) []tokenizer.Segment {
	on, ok := r.EffectiveOn(m)
	if !ok {
		return nil
	}
	if on {
		return r.OnSuffix
	}
	return r.OffSuffix
}
