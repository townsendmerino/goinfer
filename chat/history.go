package chat

// The history rule: how a family's own chat template re-renders an assistant turn's REASONING when the conversation is
// replayed to it, and the same rule here so a client's replay of the model's reasoning reaches the model the way the model's
// template would have put it there. Read from the three real templates (2026-09-30) and pinned per checkpoint against
// HuggingFace (testdata/chat_think_goldens/think_history.json, scripts/pin_chat_think_history.py):
//
//	Qwen3.5   An assistant turn AFTER the last user query (the tool loop in progress) is rendered
//	          `<think>\n{reasoning}\n</think>\n\n{content}` — always, with an empty block when it has no reasoning. A turn at
//	          or before the last query is rendered as its content alone. Content is trimmed; reasoning is trimmed.
//	Qwen3     The same split, but the block is written only for the LAST message or a turn that has reasoning; reasoning is
//	          stripped of newlines, content of leading newlines.
//	Gemma 4   A model turn after the last user turn, with reasoning, gets `<|channel>thought\n{reasoning}\n<channel|>` before
//	          its content; every model turn's content has its own channel spans removed.
//
// Where the client gave no reasoning but left `<think>…</think>` inside the content (a client that did not split it), the
// Qwen templates extract it from there: reasoning is the text before the first `</think>` (after its last `<think>`), content
// is what follows the last `</think>`. That runs for EVERY assistant turn, including ones before the last query — which is what
// strips a replayed `<think>` block out of old turns instead of re-sending it as prose.
//
// Applies only when the Template is managed (Reasoning spec recognised) and not ThinkAsIs: ThinkAsIs is the pre-thinking bytes.
//
// What is deliberately NOT replicated: the templates trim every message's content (user and system too), which goinfer's
// renderers have never done; and Gemma 4's `preserve_thinking` kwarg. Both are separate fidelity questions from reasoning.

import "slices"

import "strings"

type histKind uint8

const (
	histNone histKind = iota
	histQwen3
	histQwen35
	histGemma4
)

// historyKind is the rule this Template applies to assistant turns (histNone unless managed and not ThinkAsIs).
func (t *Template) historyKind() histKind {
	if t == nil || t.reason == nil || t.think == ThinkAsIs {
		return histNone
	}
	return t.reason.hist
}

// detectHistoryKind reads a ChatML-family template's assistant-turn block. Anything that is not one of the two shapes read from
// real checkpoints is histNone: replay the way goinfer always has.
func detectHistoryKind(tmpl string) histKind {
	const afterQuery = "loop.index0 > ns.last_query_index"
	switch {
	case strings.Contains(tmpl, afterQuery) &&
		strings.Contains(tmpl, `loop.last or (not loop.last and reasoning_content)`) &&
		strings.Contains(tmpl, `'\n<think>\n' + reasoning_content.strip('\n') + '\n</think>\n\n' + content.lstrip('\n')`):
		return histQwen3
	case strings.Contains(tmpl, afterQuery) &&
		strings.Contains(tmpl, `reasoning_content|trim`) &&
		strings.Contains(tmpl, `'\n<think>\n' + reasoning_content + '\n</think>\n\n' + content }}`):
		return histQwen35
	}
	return histNone
}

// detectGemma4History: the thinking gate and the channel the template writes.
func detectGemma4History(tmpl string) histKind {
	if strings.Contains(tmpl, "thinking_gate") && strings.Contains(tmpl, "last_user_idx") &&
		strings.Contains(tmpl, `'<|channel>thought\n' + thinking_text + '\n<channel|>'`) &&
		strings.Contains(tmpl, "strip_thinking") {
		return histGemma4
	}
	return histNone
}

// lastQueryIndex is the index of the last user turn that starts a query — not a turn that only accompanies tool results.
// With no such turn it is the last turn's index (Qwen3's own default), so nothing counts as "after the query" except the last.
func lastQueryIndex(turns []Turn) int {
	for i, turn := range slices.Backward(turns) {
		if turn.Role == "user" && !turn.ToolLoop {
			return i
		}
	}
	return len(turns) - 1
}

// lastUserIndex is Gemma 4's: the last user turn, or -1.
func lastUserIndex(turns []Turn) int {
	for i, turn := range slices.Backward(turns) {
		if turn.Role == "user" && !turn.ToolLoop {
			return i
		}
	}
	return -1
}

// splitTagged is the Qwen templates' extraction of a <think> block left inside assistant content.
func splitTagged(content string) (reasoning, rest string, ok bool) {
	const closeTag, openTag = "</think>", "<think>"
	before, _, ok := strings.Cut(content, closeTag)
	if !ok {
		return "", content, false
	}
	head := strings.TrimRight(before, "\n")
	if i := strings.LastIndex(head, openTag); i >= 0 {
		head = head[i+len(openTag):]
	}
	reasoning = strings.TrimLeft(head, "\n")
	rest = strings.TrimLeft(content[strings.LastIndex(content, closeTag)+len(closeTag):], "\n")
	return reasoning, rest, true
}

// qwenAssistant applies the Qwen history rule to assistant turn idx of n (lq is lastQueryIndex): whether the turn is
// rendered with a think block, that block's text, and the content to render.
func qwenAssistant(kind histKind, t Turn, idx, lq, n int) (block bool, reasoning, content string) {
	content, reasoning = t.Content, t.Reasoning
	if kind == histQwen35 {
		content = strings.TrimSpace(content)
	}
	if reasoning == "" {
		if r, rest, ok := splitTagged(content); ok {
			reasoning, content = r, rest
		}
	}
	if kind == histQwen35 {
		reasoning = strings.TrimSpace(reasoning)
	}
	after := idx > lq
	switch kind {
	case histQwen35:
		if after {
			return true, reasoning, content
		}
	case histQwen3:
		if after && (idx == n-1 || reasoning != "") {
			return true, strings.Trim(reasoning, "\n"), strings.TrimLeft(content, "\n")
		}
	}
	return false, "", content
}

// stripChannels is Gemma 4's strip_thinking: drop every `<|channel>…<channel|>` span, then trim.
func stripChannels(text string) string {
	var out strings.Builder
	for part := range strings.SplitSeq(text, "<channel|>") {
		if before, _, ok := strings.Cut(part, "<|channel>"); ok {
			out.WriteString(before)
		} else {
			out.WriteString(part)
		}
	}
	return strings.TrimSpace(out.String())
}

// detectGroupedToolResults reports whether a ChatML template writes a run of consecutive tool results as ONE user turn — a single
// `<|im_start|>user` before the first and a single `<|im_end|>` after the last, with each result in its own <tool_response> block. Qwen2.5,
// Qwen3 and Qwen3.5 all do (their loops open the user turn only when the previous message was not a tool message). The marker is that
// condition, `role != "tool"`, in a template that also writes <tool_response>.
func detectGroupedToolResults(tmpl string) bool {
	return strings.Contains(tmpl, "<tool_response>") && strings.Contains(tmpl, `role != "tool"`)
}
