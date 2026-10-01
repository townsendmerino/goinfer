package serveapp

// Thinking on the wire (docs/tasks/task-qwen35-think-prompt-2026-09.md).
//
// The invariant every route keeps: `content` is the answer and never carries think markup; reasoning, when the model
// produced any, travels in a separate field a client is free to ignore. chat.Template says what the checkpoint's template
// does about thinking and splits replies (chat/reasoning.go); this file is the serve half — which mode a request gets, how a
// generation's text is routed through the splitter, and how each wire format surfaces the reasoning.
//
//	request                                   mode
//	chat_template_kwargs.enable_thinking      true → on, false → off        (vLLM / llama.cpp / SGLang's own spelling)
//	reasoning_effort                          "none" → off; any other value changes nothing (see below)
//	anthropic thinking.type                   enabled|adaptive → on, disabled → off
//	(nothing)                                 the server default, -thinking (template | asis | on | off; default template)
//
// -reasoning-format (llama.cpp's names, so configs carry over) decides what a client that understands reasoning gets:
//
//	deepseek         (default) content is clean, reasoning goes in reasoning_content
//	deepseek-legacy  reasoning_content is filled AND content keeps the raw <think> tags, for clients that parse tags
//	none             nothing is separated: the raw text, tags and all, is the content — exactly what serve sent before
//
// A model whose template has no recognised thinking control (chat.Template.Reasoning() == nil) is never split and never
// re-prompted, whatever the request says: unmanaged means today's behaviour.

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/townsendmerino/goinfer/chat"
)

type reasoningFormat uint8

const (
	rfSplit  reasoningFormat = iota // "deepseek"
	rfLegacy                        // "deepseek-legacy"
	rfNone                          // "none"
)

func parseReasoningFormat(s string) (reasoningFormat, bool) {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "", "deepseek":
		return rfSplit, true
	case "deepseek-legacy":
		return rfLegacy, true
	case "none":
		return rfNone, true
	}
	return rfSplit, false
}

// thinkDefault / reasoningFormat resolve the -thinking / -reasoning-format flags (validated at startup). The default is
// `template` — each model's own chat template's default (owner decision 2026-09-30) — and it lives HERE, so a config built
// without the flag, as tests build it, gets exactly what the flag's default gives; `asis` (the pre-thinking prompt bytes)
// has to be asked for by name.
func (c config) thinkDefault() chat.ThinkMode {
	if strings.TrimSpace(c.thinking) == "" {
		return chat.ThinkTemplate
	}
	m, _ := chat.ParseThinkMode(c.thinking)
	return m
}

func (c config) reasoningFormat() reasoningFormat {
	f, _ := parseReasoningFormat(c.reasoningFmt)
	return f
}

// thinkSettings is what one request resolved to.
type thinkSettings struct {
	budget   int             // the request's own reasoning-token budget (0 = none asked for); budget.go clamps it
	mode     chat.ThinkMode  // the mode to render; the server default when the request said nothing
	explicit bool            // the request itself chose on/off (vs inheriting the server default)
	format   reasoningFormat // how reasoning reaches this client
	// effort is the request's reasoning_effort when it is one gpt-oss's template takes (low | medium | high), else "". It only
	// ever reaches a Harmony template (chat.Template.WithReasoningEffort is a no-op for every other family), so a client that
	// sends a bare effort to every endpoint moves no other model's prompt.
	effort string
}

// thinkRequest is the request-side thinking controls, gathered from whichever route they arrived on.
type thinkRequest struct {
	kwargs          map[string]json.RawMessage // chat_template_kwargs
	reasoningEffort string                     // OpenAI reasoning_effort
	anthropicType   string                     // anthropic thinking.type
	format          string                     // reasoning_format
	budget          int                        // thinking_token_budget / Anthropic budget_tokens (0 = none)
}

// resolveThink turns the request's controls plus the server defaults into settings. The error is a client error (400).
func (s *server) resolveThink(tr thinkRequest) (thinkSettings, error) {
	ts := thinkSettings{mode: s.cfg.thinkDefault(), format: s.cfg.reasoningFormat()}
	if tr.format != "" {
		f, ok := parseReasoningFormat(tr.format)
		if !ok {
			return ts, fmt.Errorf("reasoning_format %q: want deepseek, deepseek-legacy or none", tr.format)
		}
		ts.format = f
	}
	if tr.budget < 0 {
		return ts, fmt.Errorf("the thinking token budget must be positive, got %d", tr.budget)
	}
	ts.budget = tr.budget
	if e, ok := chat.NormalizeReasoningEffort(tr.reasoningEffort); ok {
		ts.effort = e
	}
	set := func(on bool) {
		ts.explicit = true
		ts.mode = chat.ThinkOff
		if on {
			ts.mode = chat.ThinkOn
		}
	}
	if raw, ok := tr.kwargs["enable_thinking"]; ok {
		var b bool
		if err := json.Unmarshal(raw, &b); err != nil {
			return ts, fmt.Errorf("chat_template_kwargs.enable_thinking must be a boolean")
		}
		set(b)
	} else if strings.EqualFold(strings.TrimSpace(tr.reasoningEffort), "none") {
		// reasoning_effort can only turn thinking OFF. Clients such as the DeepSeek Harness send a bare
		// reasoning_effort ("high") to every endpoint they do not recognise (docs/server.md), and reading that as "turn
		// thinking on" would change their prompts and, at a small max_tokens, leave them an empty answer. A client that
		// wants thinking on says so in chat_template_kwargs.enable_thinking (or -thinking on serves it everywhere).
		set(false)
	} else if tr.anthropicType != "" {
		switch tr.anthropicType {
		case "enabled", "adaptive":
			set(true)
		case "disabled":
			set(false)
		default:
			return ts, fmt.Errorf("thinking.type %q: want enabled, adaptive or disabled", tr.anthropicType)
		}
	}
	return ts, nil
}

// templateFor is the chat template to render this request with: the model's own (which already carries the server
// default mode), switched to the request's mode when the request chose one. nil when the model has no template.
func (lm *loadedModel) templateFor(ts thinkSettings) *chat.Template {
	t := lm.tmpl.WithReasoningEffort(ts.effort) // gpt-oss only; "" and every other family leave the template as it is
	if t == nil || !ts.explicit {
		return t
	}
	return t.WithThinking(ts.mode)
}

// constrainedTemplate is templateFor for a request whose output a grammar will constrain from its first token
// (response_format json, a forced or lone tool): a prompt that ends inside an open think block would contradict the
// grammar — the mask demands `{` or `<tool_call>` while the prompt says the model is mid-reasoning — so such a request is
// rendered thinking-off whatever the mode.
func (lm *loadedModel) constrainedTemplate(ts thinkSettings) *chat.Template {
	t := lm.templateFor(ts)
	if t.PromptOpensThink() {
		return t.WithThinking(chat.ThinkOff)
	}
	return t
}

// promptForT renders system + turns through tmpl (the per-request template) and encodes the result; with no template it
// falls back to the raw-conversation prompt. EncodeSegments keeps a special-token surface form typed into a user/tool
// message from becoming a real control token that forges a turn boundary (M25); addBOS is false because the template emits
// its own BOS marker. Shared by every chat route, so they all encode prompts identically.
func (lm *loadedModel) promptForT(tmpl *chat.Template, system string, turns []chat.Turn) ([]int, error) {
	if tmpl != nil {
		return lm.tk.EncodeSegments(tmpl.RenderSegments(system, turns), false)
	}
	return lm.encode(rawPrompt(system, turns))
}

// ---------------------------------------------------------------------------------------------------------------------

// thinkOut separates one generation's decoded text into reasoning and the text the STOP LOGIC watches. nil on a genRequest
// means "pass text through untouched" (-reasoning-format none, or a model without a reasoning spec).
//
// Stop strings are matched against the ANSWER only (streamTokens feeds the stop logic what feed returns): a stop string is a
// request about what the model says to the client, and the reasoning is scratch work it never asked to police. Before this, a
// stop string that appeared in the model's thinking ended the reply there with no answer at all. The exception is deepseek-legacy,
// where `content` is the raw text, tags and all: what the client sees is what the stop logic watches, so a stop in the thinking
// still stops it there.
type thinkOut struct {
	rs *chat.ReplySplitter
	// onReasoning receives the reasoning text as it arrives, always on a UTF-8 boundary. Never nil once routed: a route that
	// does not surface reasoning (Responses, an Anthropic request that did not ask for thinking) sets a discarding func, which
	// keeps drive's progress accounting honest while the model reasons.
	onReasoning func(string)
	raw         bool // deepseek-legacy: content is the raw text, tags included; reasoning is ALSO reported
	// endedInReasoning is set when the reply ended inside an unclosed block: truncated while thinking, so there is no
	// answer — content is empty and finish_reason is "length".
	endedInReasoning bool
}

// newThinkOut builds the router for a request rendered with tmpl from turns, or nil when nothing should be split.
func newThinkOut(tmpl *chat.Template, turns []chat.Turn, ts thinkSettings, onReasoning func(string)) *thinkOut {
	if ts.format == rfNone {
		return nil
	}
	rs := tmpl.NewReplySplitter(turns)
	if rs == nil {
		return nil
	}
	if onReasoning == nil {
		onReasoning = func(string) {}
	}
	return &thinkOut{rs: rs, onReasoning: onReasoning, raw: ts.format == rfLegacy}
}

// feed takes one decoded fragment, reports its reasoning, and returns the text the stop logic must watch: the answer (the
// fragment with its reasoning taken out), or — deepseek-legacy — the fragment itself.
func (o *thinkOut) feed(piece string) string {
	r, c := o.rs.Push(piece)
	if r != "" {
		o.onReasoning(r)
	}
	if o.raw {
		return piece
	}
	return c
}

// finish ends the reply: it reports the reasoning still held (and any partial rune, now as it stands) and returns the answer text
// still held for the stop logic. A reply that ends inside an unclosed block has no answer.
func (o *thinkOut) finish() string {
	r, c, truncated := o.rs.Finish()
	o.endedInReasoning = truncated
	if r != "" {
		o.onReasoning(r)
	}
	if o.raw {
		return ""
	}
	return c
}
