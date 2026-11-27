package chatcore

import (
	"regexp"
	"slices"
	"strings"
)

// HistKind names how a template re-renders assistant reasoning across conversation history.
type HistKind uint8

const (
	HistNone HistKind = iota
	HistQwen3
	HistQwen35
	HistGemma4
	HistMellum21
)

// DetectHistoryKind classifies a ChatML-family template's assistant-turn reasoning replay rule.
func DetectHistoryKind(tmpl string) HistKind {
	const afterQuery = "loop.index0 > ns.last_query_index"
	switch {
	case strings.Contains(tmpl, afterQuery) &&
		strings.Contains(tmpl, `loop.last or (not loop.last and reasoning_content)`) &&
		strings.Contains(tmpl, `'\n<think>\n' + reasoning_content.strip('\n') + '\n</think>\n\n' + content.lstrip('\n')`):
		return HistQwen3
	case strings.Contains(tmpl, afterQuery) &&
		strings.Contains(tmpl, `reasoning_content|trim`) &&
		strings.Contains(tmpl, `'\n<think>\n' + reasoning_content + '\n</think>\n\n' + content }}`):
		return HistQwen35
	case strings.Contains(tmpl, afterQuery) && strings.Contains(tmpl, "{%- if reasoning_content %}") &&
		strings.Contains(tmpl, `'\n<think>\n' + reasoning_content.strip('\n') + '\n</think>\n\n' + content.lstrip('\n')`) &&
		!strings.Contains(tmpl, `loop.last or (not loop.last and reasoning_content)`):
		return HistMellum21
	}
	return HistNone
}

// DetectGemma4History reports HistGemma4 when tmpl uses Gemma 4's thinking_gate and thought channel.
func DetectGemma4History(tmpl string) HistKind {
	if strings.Contains(tmpl, "thinking_gate") && strings.Contains(tmpl, "last_user_idx") &&
		strings.Contains(tmpl, `'<|channel>thought\n' + thinking_text + '\n<channel|>'`) &&
		strings.Contains(tmpl, "strip_thinking") {
		return HistGemma4
	}
	return HistNone
}

// LastQueryIndex returns the index of the last user turn that is not a tool-loop accompaniment,
// or len(turns)-1 when none exists.
func LastQueryIndex(turns []Turn) int {
	for i, turn := range slices.Backward(turns) {
		if turn.Role == "user" && !turn.ToolLoop {
			return i
		}
	}
	return len(turns) - 1
}

// LastUserIndex returns the index of the last user turn that is not a tool-loop accompaniment, or -1.
func LastUserIndex(turns []Turn) int {
	for i, turn := range slices.Backward(turns) {
		if turn.Role == "user" && !turn.ToolLoop {
			return i
		}
	}
	return -1
}

// SplitTagged extracts a <think>...</think> block embedded inside assistant content.
func SplitTagged(content string) (reasoning, rest string, ok bool) {
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

// QwenAssistant applies the Qwen/Mellum assistant-turn history rule at turn idx of n (lq is LastQueryIndex).
func QwenAssistant(kind HistKind, t Turn, idx, lq, n int) (block bool, reasoning, content string) {
	content, reasoning = t.Content, t.Reasoning
	if kind == HistQwen35 {
		content = strings.TrimSpace(content)
	}
	if reasoning == "" {
		if r, rest, ok := SplitTagged(content); ok {
			reasoning, content = r, rest
		}
	}
	if kind == HistQwen35 {
		reasoning = strings.TrimSpace(reasoning)
	}
	after := idx > lq
	switch kind {
	case HistQwen35:
		if after {
			return true, reasoning, content
		}
	case HistQwen3:
		if after && (idx == n-1 || reasoning != "") {
			return true, strings.Trim(reasoning, "\n"), strings.TrimLeft(content, "\n")
		}
	case HistMellum21:
		if after && reasoning != "" {
			return true, strings.Trim(reasoning, "\n"), strings.TrimLeft(content, "\n")
		}
	}
	return false, "", content
}

// StripChannels removes every <|channel>...<channel|> span from text and trims surrounding whitespace.
func StripChannels(text string) string {
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

// DetectGroupedToolResults reports whether a ChatML template groups consecutive tool results into one user turn.
func DetectGroupedToolResults(tmpl string) bool {
	return strings.Contains(tmpl, "<tool_response>") && strings.Contains(tmpl, `role != "tool"`)
}

var chatMLDefaultSystemRe = regexp.MustCompile(`<\|im_start\|>system(?:\\n|\n)([^<>{}'"+\\]+?)<\|im_end\|>`)

// ChatMLDefaultSystem returns the literal default system message declared by a ChatML template, or "".
func ChatMLDefaultSystem(tmpl string) string {
	if m := chatMLDefaultSystemRe.FindStringSubmatch(tmpl); m != nil {
		return m[1]
	}
	return ""
}
