package chat

import "unicode/utf8"

// ReplySplitter is what a decode loop holds to separate a streamed reply into reasoning and answer: a ThinkSplitter (a delimited
// <think>…</think> span — Qwen, Gemma 4) or the Harmony parser (gpt-oss's channel messages), plus the one thing a loop that
// prints or forwards the reasoning needs beyond it — the reasoning is reported only on UTF-8 boundaries, so a multi-byte
// character split across two tokens reaches the reader whole. (The answer is returned as it arrives, raw: the loops that
// consume it already hold a trailing partial rune back for their own stop-string and display logic.)
//
// serve, goinfer-chat and the demo agent all use it, so the three cannot disagree about where reasoning ends.
type ReplySplitter struct {
	sp    replyCore
	carry []byte
}

// replyCore is the part of a splitter ReplySplitter needs: ThinkSplitter and the Harmony parser both satisfy it, so the three
// decode loops that hold a ReplySplitter never learn which family they are reading.
type replyCore interface {
	Push(chunk string) (reasoning, content string)
	Flush() (reasoning, content string)
	InReasoning() bool
}

// NewReplySplitter returns a splitter for replies to prompts rendered by t from turns, or nil when t has no reasoning spec (the
// caller then passes every fragment through as answer, exactly as before thinking was modelled).
func (t *Template) NewReplySplitter(turns []Turn) *ReplySplitter {
	if t != nil && t.name == "harmony" {
		return &ReplySplitter{sp: newHarmonySplitter()}
	}
	sp := t.NewReasoningSplitter(turns)
	if sp == nil {
		return nil
	}
	return &ReplySplitter{sp: sp}
}

// Push feeds the next decoded fragment and returns the reasoning (complete runes only) and the answer text that are safe to
// use now — either may be "".
func (r *ReplySplitter) Push(fragment string) (reasoning, answer string) {
	rs, a := r.sp.Push(fragment)
	return r.complete(rs), a
}

// Finish ends the reply and returns what was still held: reasoning (including any partial rune, as it stands) and answer. A reply
// that ended inside an unclosed reasoning block was truncated while thinking — truncated reports it, and answer is empty.
func (r *ReplySplitter) Finish() (reasoning, answer string, truncated bool) {
	truncated = r.sp.InReasoning()
	rs, a := r.sp.Flush()
	out := string(r.carry) + rs
	r.carry = nil
	return out, a, truncated
}

// ToolCalls returns the function calls a Harmony (gpt-oss) reply made, once the reply is finished; nil for every other family. A call
// is a message addressed to a recipient, which the splitter routes out of both the reasoning and the answer — so the text a caller
// buffers never contains it, and this is the only place to read it from. Calls to a recipient that is not `functions.NAME`, and calls
// whose arguments are not a JSON object, are left out.
func (r *ReplySplitter) ToolCalls() []ToolCall {
	if r == nil {
		return nil
	}
	if h, ok := r.sp.(*harmonySplitter); ok {
		return harmonyToolCalls(h.calls)
	}
	return nil
}

// complete returns the prefix of carry+s that ends on a rune boundary and holds the rest back.
func (r *ReplySplitter) complete(s string) string {
	if s == "" && len(r.carry) == 0 {
		return ""
	}
	buf := append(r.carry, s...)
	end := 0
	for i := 0; i < len(buf); {
		if !utf8.FullRune(buf[i:]) {
			break
		}
		_, size := utf8.DecodeRune(buf[i:])
		i += size
		end = i
	}
	out := string(buf[:end])
	r.carry = append(r.carry[:0], buf[end:]...)
	return out
}
