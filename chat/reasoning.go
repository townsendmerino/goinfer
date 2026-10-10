package chat

// Reasoning: the two halves of "this family thinks before it answers", declared once per family so they cannot disagree.
//
//   - PROMPT half: what the checkpoint's own chat template writes after the assistant tag for each thinking setting
//     (nothing, an open `<think>\n`, or a closed empty block), and, for Gemma 4, a `<|think|>` marker in the system
//     turn. The setting is a ThinkMode; WithThinking returns a Template that renders it.
//   - OUTPUT half: how the reply delimits its reasoning (`<think>…</think>`, Gemma 4's `<|channel>thought\n…<channel|>`).
//     ThinkSplitter separates it into reasoning and content, chunk-boundary safe, so a wire layer can put the answer in
//     `content` and the reasoning somewhere a client can ignore.
//
// Why it is per checkpoint (docs/tasks/task-qwen35-think-prompt-2026-09.md): the generic ChatML renderer writes nothing after
// "<|im_start|>assistant\n", but Qwen3.5's own template writes a think block there, and WHICH block depends on the checkpoint: the
// 0.8B defaults to thinking OFF (closed empty block), the 9B to ON (open `<think>\n`), Qwen3 to ON with nothing written. The same knob
// (`enable_thinking`) means different things per family and size, so nothing here hard-wires "Qwen does X".
//
// FAIL TOWARD TODAY'S BYTES. A template whose thinking control is not recognised gets no Reasoning spec at all: no prefill change,
// no splitting, exactly the bytes and text it produced before thinking was modelled. A default is never guessed: it is read from the
// template's own generation-prompt block, and pinned per checkpoint by TestThinkModes_matchHF against prompts HuggingFace renders from
// each real template.

import (
	"strings"
)

// ThinkMode is the per-request thinking control, resolved by the caller (request field, else server default).
type ThinkMode uint8

const (
	// ThinkAsIs renders exactly what this renderer rendered before thinking was modelled: the family's generic generation
	// prompt and nothing added. It is the zero value, so a caller that never heard of thinking changes nothing.
	ThinkAsIs ThinkMode = iota
	// ThinkTemplate renders what the checkpoint's own chat template renders with enable_thinking UNSET.
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

// ParseThinkMode parses the names String returns ("" is ThinkAsIs).
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

// thinkDefault is what a checkpoint's template does with enable_thinking unset.
type thinkDefault uint8

const (
	defOn  thinkDefault = iota // thinking on (the model opens the block itself, or the prompt does)
	defOff                     // thinking off (the prompt carries a closed block)
)

// Reasoning is one family's declared reasoning behaviour. nil on a Template means the family does not think in its
// template (Phi-3, Gemma 3, Qwen2.5, …): no prefill, no splitting.
type Reasoning struct {
	open, close string // delimiters as they appear in DECODED output

	// openTok / closeTok are the SINGLE TOKENS whose text opens and closes the block (for Qwen the same as open/close; for
	// Gemma 4 the channel markers "<|channel>" / "<channel|>", where open also carries the "thought\n" that follows). A
	// reasoning budget finds the block by these ids and closes it by forcing closeTok.
	openTok, closeTok string

	def  thinkDefault // the template's own default, read from the template text
	hist histKind     // how the template re-renders an assistant turn's reasoning in history (history.go)

	// onSuffix / offSuffix are appended after the family's generic generation prompt (ChatML family), as tagged segments;
	// onOpens reports that onSuffix leaves the prompt inside an open block.
	onSuffix, offSuffix []Segment
	onOpens             bool

	// gemma4 marks the variant whose On form is a system-turn marker rather than a generation-prompt suffix (its generic
	// prompt already carries the closed scaffold, which is its Off form).
	gemma4 bool

	// oldGemma4 marks the earlier Gemma 4 template (the E2B GGUF's): same markers, but its generation prompt NEVER carries the closed
	// scaffold — thinking off is simply `<|turn>model\n`. Writing the scaffold there anyway makes the model answer as if it had
	// reasoned, ending its reply with a bare `<channel|>` and no opener.
	oldGemma4 bool
}

// Open and Close are the delimiters a reply uses around its reasoning, as decoded text.
func (r *Reasoning) Open() string  { return r.open }
func (r *Reasoning) Close() string { return r.close }

// OpenToken and CloseToken are the single-token texts that open and close the block (see the field comment).
func (r *Reasoning) OpenToken() string  { return r.openTok }
func (r *Reasoning) CloseToken() string { return r.closeTok }

// DefaultOn reports whether the checkpoint's own template has thinking ON when enable_thinking is unset.
func (r *Reasoning) DefaultOn() bool { return r.def == defOn }

func special(s string) Segment { return Segment{Text: s, Special: true} }
func plain(s string) Segment   { return Segment{Text: s} }

var (
	closedSegs = []Segment{special("<think>"), plain("\n\n"), special("</think>"), plain("\n\n")}
	openSegs   = []Segment{special("<think>"), plain("\n")}
)

// detectChatMLReasoning reads a ChatML-family template's generation-prompt block (the text after the LAST "add_generation_prompt")
// and classifies it. The three shapes below are the ones read from real checkpoints; anything else returns nil (unmanaged), never a guess.
//
//	Qwen3 (1.7B/4B/30B-A3B): `enable_thinking is false` → closed block; otherwise nothing (model opens it itself).
//	Qwen3.5 0.8B:            `enable_thinking is true`  → open `<think>\n`; otherwise the closed block.
//	Qwen3.5 9B, JEV-9B:      `enable_thinking is false` → closed block; otherwise the open `<think>\n`.
func detectChatMLReasoning(tmpl string) *Reasoning {
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
	r := &Reasoning{open: "<think>", close: "</think>", openTok: "<think>", closeTok: "</think>"}
	switch {
	case isFalse && hasClosed && !hasOpen && !isTrue: // Qwen3
		r.def, r.onSuffix, r.offSuffix, r.onOpens = defOn, nil, closedSegs, false
	case isTrue && hasClosed && hasOpen && !isFalse: // Qwen3.5 small: default off
		r.def, r.onSuffix, r.offSuffix, r.onOpens = defOff, openSegs, closedSegs, true
	case isFalse && hasClosed && hasOpen && !isTrue: // Qwen3.5 9B: default on, prompt opens the block
		r.def, r.onSuffix, r.offSuffix, r.onOpens = defOn, openSegs, closedSegs, true
	default:
		return nil
	}
	r.hist = detectHistoryKind(tmpl)
	return r
}

// detectGemma4Reasoning: Gemma 4's template has `enable_thinking | default(false)`, a `<|think|>` marker in the system
// turn when on, and the closed scaffold `<|channel>thought\n<channel|>` after the generation prompt when off. Recognised
// only when all three markers are present.
func detectGemma4Reasoning(tmpl string) *Reasoning {
	if !strings.Contains(tmpl, "<|think|>") {
		return nil
	}
	if r := detectOldGemma4Reasoning(tmpl); r != nil {
		return r
	}
	if !strings.Contains(tmpl, "<|channel>thought\\n<channel|>") || !strings.Contains(tmpl, "enable_thinking | default(false)") {
		return nil
	}
	return &Reasoning{open: "<|channel>thought\n", close: "<channel|>", openTok: "<|channel>", closeTok: "<channel|>", def: defOff, gemma4: true, hist: detectGemma4History(tmpl)}
}

// detectOldGemma4Reasoning recognises the earlier Gemma 4 template (the E2B GGUF's): thinking is `enable_thinking is defined and
// enable_thinking` → a `<|think|>` line in the system turn, the generation prompt is `<|turn>model\n` in every mode (no closed
// scaffold anywhere), and the only channel it writes is the reasoning of a tool-calling turn in the loop in progress. Its history
// rule is not the canonical one, so hist stays none and history is rendered generically (reasoning of earlier turns dropped, which is
// what the template does for turns without calls).
func detectOldGemma4Reasoning(tmpl string) *Reasoning {
	g := strings.LastIndex(tmpl, "add_generation_prompt")
	if g < 0 || !strings.Contains(tmpl, "enable_thinking is defined and enable_thinking") || strings.Contains(tmpl, "default(false)") ||
		strings.Contains(tmpl[g:], "<channel|>") { // a scaffold after the generation prompt is the canonical form
		return nil
	}
	return &Reasoning{open: "<|channel>thought\n", close: "<channel|>", openTok: "<|channel>", closeTok: "<channel|>", def: defOff, gemma4: true, oldGemma4: true}
}

// effectiveOn resolves a mode to on/off for this checkpoint. ThinkAsIs has no answer of its own (ok=false): it is the
// family's generic rendering, which for the ChatML family is "nothing written" and for Gemma 4 is the closed scaffold.
func (r *Reasoning) effectiveOn(m ThinkMode) (on, ok bool) {
	switch m {
	case ThinkOn:
		return true, true
	case ThinkOff:
		return false, true
	case ThinkTemplate:
		return r.def == defOn, true
	}
	return false, false
}

// suffix is the segments the ChatML family appends after its generic generation prompt for mode m (nil for ThinkAsIs).
func (r *Reasoning) suffix(m ThinkMode) []Segment {
	on, ok := r.effectiveOn(m)
	if !ok {
		return nil
	}
	if on {
		return r.onSuffix
	}
	return r.offSuffix
}

// WithThinking returns t rendering thinking mode m. A family without a Reasoning spec returns t unchanged — the
// fail-toward-today's-bytes rule — so the caller needs no per-family branch.
func (t *Template) WithThinking(m ThinkMode) *Template {
	if t == nil || t.reason == nil || m == t.think {
		return t
	}
	c := *t
	c.think = m
	return &c
}

// Reasoning returns the family's reasoning spec for this checkpoint, or nil when it has none or it was not recognised.
func (t *Template) Reasoning() *Reasoning {
	if t == nil {
		return nil
	}
	return t.reason
}

// ThinkMode is the mode this Template renders (ThinkAsIs unless WithThinking chose another).
func (t *Template) ThinkMode() ThinkMode {
	if t == nil {
		return ThinkAsIs
	}
	return t.think
}

// PromptOpensThink reports whether the generation prompt this Template renders ends INSIDE an open think block, so the
// reply starts with reasoning and only the closing delimiter will be generated. It is what a ThinkSplitter's forcedOpen
// must be for replies to prompts from this Template.
func (t *Template) PromptOpensThink() bool {
	if t == nil || t.reason == nil {
		return false
	}
	on, ok := t.reason.effectiveOn(t.think)
	return ok && on && t.reason.onOpens
}

// ThinkingPossible reports whether a reply to a prompt from this Template can contain a reasoning block: false for a
// family without a spec and under ThinkOff (the prompt carries a closed block), true under ThinkOn, true under ThinkAsIs
// (nothing is written, so the model may open one itself — unknowable), and under ThinkTemplate whatever the checkpoint's
// own default is. A reasoning budget is only installed where this is true.
func (t *Template) ThinkingPossible() bool {
	if t != nil && t.name == "harmony" {
		return true // gpt-oss has no off form in its prompt: every reply opens an analysis message, whatever the mode
	}
	if t == nil || t.reason == nil {
		return false
	}
	on, ok := t.reason.effectiveOn(t.think)
	return !ok || on
}

// PromptOpensThinkFor is PromptOpensThink for one conversation. Gemma 4 with thinking on ends a prompt INSIDE an open thought
// channel (`<|channel>thought\n`) when the conversation's last turn is a tool response — the model is still in the turn that made
// the call and the template reopens its channel — so a reply to such a prompt starts mid-reasoning and only the close is generated,
// exactly like Qwen3.5-9B's open `<think>\n`. Every other prompt is as PromptOpensThink says.
func (t *Template) PromptOpensThinkFor(turns []Turn) bool {
	if t.PromptOpensThink() {
		return true
	}
	if t == nil || t.reason == nil || !t.reason.gemma4 || len(turns) == 0 || turns[len(turns)-1].Role != "tool" {
		return false
	}
	on, ok := t.reason.effectiveOn(t.think)
	return ok && on
}

// NewReasoningSplitter returns a splitter for replies to prompts from t, or nil when t has no reasoning spec (the caller
// then passes text through untouched). turns is the conversation the prompt was rendered from (nil = unknown: only the
// template-level answer is used).
func (t *Template) NewReasoningSplitter(turns []Turn) *ThinkSplitter {
	if t == nil || t.reason == nil {
		return nil
	}
	return NewThinkSplitter(t.reason.open, t.reason.close, t.PromptOpensThinkFor(turns))
}

// renderThinking applies t.think to a base rendering. Only called when t.reason != nil and t.think != ThinkAsIs.
func (t *Template) renderThinking(system string, turns []Turn) []Segment {
	r := t.reason
	if r.gemma4 {
		on, _ := r.effectiveOn(t.think)
		return gemma4Segments(system, turns, on, r.hist == histGemma4 || r.oldGemma4, r.oldGemma4)
	}
	segs := chatMLSegments(system, turns, r.hist)
	return append(segs[:len(segs):len(segs)], r.suffix(t.think)...)
}

// ---------------------------------------------------------------------------------------------------------------------

// ThinkSplitter separates a streamed reply into reasoning and content.
//
//	Undecided  — nothing emitted yet. Optional leading whitespace, then the open delimiter → Reasoning (delimiter and the
//	             newlines after it dropped); anything else → Content. A partial open delimiter at the tail is held.
//	Reasoning  — emitted as reasoning up to the close delimiter; trailing newlines are held until something follows them
//	             (Qwen's template strips them: reasoning_content.strip('\n')), and a partial close delimiter is held.
//	AfterClose — the newlines following the close delimiter are dropped; the first other byte → Content.
//	Content    — emitted as is. An open delimiter seen here is ordinary content: it never re-enters Reasoning.
//
// forcedOpen starts in Reasoning: the prompt ended inside an open block, so only the close delimiter is generated.
// A reply that ends inside Reasoning was TRUNCATED (max_tokens, a stop): everything it produced is reasoning and the
// content is empty — splitting cannot invent an answer (InReasoning reports it).
//
// Correctness is chunk-independence: the concatenated output of any chunking of a reply equals the output of the whole
// reply in one Push (TestThinkSplitter_chunkIndependent, every split point and random partitions).
type ThinkSplitter struct {
	open, close string
	state       splitState
	lead        bool // in Reasoning, before the first byte: newlines and a repeated open delimiter are dropped
	buf         string
}

type splitState uint8

const (
	stUndecided splitState = iota
	stReasoning
	stAfterClose
	stContent
)

// NewThinkSplitter returns a splitter for the given delimiters. Both must be non-empty.
func NewThinkSplitter(open, close string, forcedOpen bool) *ThinkSplitter {
	s := &ThinkSplitter{open: open, close: close}
	if forcedOpen {
		s.state, s.lead = stReasoning, true
	}
	return s
}

// InReasoning reports whether the reply so far is inside an unclosed reasoning block.
func (s *ThinkSplitter) InReasoning() bool { return s.state == stReasoning }

// Push feeds the next decoded chunk and returns the reasoning and content that are safe to emit now (either may be "").
func (s *ThinkSplitter) Push(chunk string) (reasoning, content string) {
	s.buf += chunk
	var r, c strings.Builder
	for {
		switch s.state {
		case stUndecided:
			tr := strings.TrimLeft(s.buf, " \t\r\n")
			switch {
			case tr == "":
				return r.String(), c.String() // whitespace only so far: hold it
			case strings.HasPrefix(tr, s.open):
				s.buf, s.state, s.lead = tr[len(s.open):], stReasoning, true
				continue
			case len(tr) < len(s.open) && strings.HasPrefix(s.open, tr):
				return r.String(), c.String() // a partial open delimiter: hold
			}
			s.state = stContent
			continue
		case stReasoning:
			if s.lead {
				s.buf = strings.TrimLeft(s.buf, "\r\n")
				if s.buf == "" {
					return r.String(), c.String()
				}
				if strings.HasPrefix(s.buf, s.open) { // a model that re-opens a block the prompt already opened
					s.buf = s.buf[len(s.open):]
					continue
				}
				if len(s.buf) < len(s.open) && strings.HasPrefix(s.open, s.buf) {
					return r.String(), c.String()
				}
				s.lead = false
			}
			if i := strings.Index(s.buf, s.close); i >= 0 {
				r.WriteString(strings.TrimRight(s.buf[:i], "\r\n"))
				s.buf, s.state = s.buf[i+len(s.close):], stAfterClose
				continue
			}
			n := StreamableLen(s.buf, s.close)
			safe := strings.TrimRight(s.buf[:n], "\r\n")
			r.WriteString(safe)
			s.buf = s.buf[len(safe):]
			return r.String(), c.String()
		case stAfterClose:
			s.buf = strings.TrimLeft(s.buf, "\r\n")
			if s.buf == "" {
				return r.String(), c.String()
			}
			s.state = stContent
			continue
		default: // stContent
			c.WriteString(s.buf)
			s.buf = ""
			return r.String(), c.String()
		}
	}
}

// Flush ends the reply and returns whatever was held. A reply that ends Undecided or in Content flushes as content (a
// partial open delimiter, or whitespace, was literal text after all); one that ends in Reasoning was truncated and flushes
// as reasoning.
func (s *ThinkSplitter) Flush() (reasoning, content string) {
	b := s.buf
	s.buf = ""
	switch s.state {
	case stReasoning:
		return strings.TrimRight(b, "\r\n"), ""
	case stUndecided, stContent:
		s.state = stContent
		return "", b
	}
	return "", ""
}

// SplitThink splits a whole reply (the non-streaming form of ThinkSplitter).
func (t *Template) SplitThink(reply string) (reasoning, content string) {
	ts := t.NewReasoningSplitter(nil)
	if ts == nil {
		return "", reply
	}
	r, c := ts.Push(reply)
	r2, c2 := ts.Flush()
	return r + r2, c + c2
}

// thinkSuffixText is the ChatML family's think suffix as text, for the tool-prompt path that builds one string.
func (t *Template) thinkSuffixText() string {
	if t.reason == nil || t.reason.gemma4 {
		return ""
	}
	var b strings.Builder
	for _, s := range t.reason.suffix(t.think) {
		b.WriteString(s.Text)
	}
	return b.String()
}

// gemma4Think reports whether this Template renders Gemma 4 with thinking on.
func (t *Template) gemma4Think() bool {
	if t.reason == nil || !t.reason.gemma4 {
		return false
	}
	on, ok := t.reason.effectiveOn(t.think)
	return ok && on
}
