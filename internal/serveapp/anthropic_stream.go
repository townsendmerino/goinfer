package serveapp

import (
	"encoding/json"
	"errors"
	"net/http"
	"strings"

	"github.com/townsendmerino/goinfer/chat"
)

// Anthropic SSE differs from the OpenAI flavor: named events
// (event: <type>\ndata: <json>\n\n), a strict block sequence, and NO [DONE]
// terminator — so these are separate from the sseStart/sseSend/sseDone helpers
// rather than bending them.

func anthropicSSEStart(w http.ResponseWriter) (*sseWriter, bool) {
	f, ok := w.(http.Flusher)
	if !ok {
		writeAnthropicErr(w, http.StatusInternalServerError, "api_error", "streaming unsupported")
		return nil, false
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	return newSSEWriter(w, f), true
}

// anthropicEvent writes one named SSE event.
func anthropicEvent(ss *sseWriter, event string, v any) {
	b, _ := json.Marshal(v)
	ss.frame("event: %s\ndata: %s\n\n", event, b)
}

// anthropicStreamErr emits an Anthropic `error` event mid-stream, when a
// generation fails after message_start has already been sent (200, headers
// flushed — no status code left to set). M1.
func anthropicStreamErr(ss *sseWriter, msg string) {
	anthropicEvent(ss, "error", map[string]any{
		"type": "error", "error": map[string]any{"type": "api_error", "message": msg},
	})
}

// streamMessages runs the Anthropic SSE state machine: message_start, a ping,
// the content block(s), message_delta (stop reason + final usage), message_stop.
// Text streams live (reusing drive's completeUTF8 holdback); a tool call is
// buffered fully (tools.go decides from the whole output) and emitted as one
// input_json_delta — Claude Code accepts the single chunk, as it does for
// llama.cpp.
func (s *server) streamMessages(w http.ResponseWriter, r *http.Request, lm *loadedModel, gr genRequest, toolsActive bool, tools []chat.Tool) {
	ss, ok := anthropicSSEStart(w)
	if !ok {
		return
	}
	id := "msg_" + reqID()
	gr.id = id // K1: registers this generation for cancel-by-id
	anthropicEvent(ss, "message_start", map[string]any{
		"type": "message_start",
		"message": map[string]any{
			"id": id, "type": "message", "role": "assistant", "model": lm.name,
			"content": []any{}, "stop_reason": nil, "stop_sequence": nil,
			"usage": map[string]any{"input_tokens": len(gr.promptIDs), "output_tokens": 0},
		},
	})
	anthropicEvent(ss, "ping", map[string]any{"type": "ping"}) // liveness check; cheap insurance

	if toolsActive {
		s.streamMessagesTools(w, r, ss, lm, gr, tools)
		return
	}

	// Text: a single content block streamed token by token.
	anthropicEvent(ss, "content_block_start", map[string]any{
		"type": "content_block_start", "index": 0,
		"content_block": map[string]any{"type": "text", "text": ""},
	})
	// N-24 (docs/audit-2026-09-10.md): the ping above is a one-shot liveness check, not a
	// keep-alive — nothing else is sent until the first token, which on CPU is after the whole
	// prefill (minutes for an image, ~270s for an 8k agent prompt) against a 300s idle timeout.
	stopBeat := sseHeartbeat(ss)
	finish, nComp, _, stopSeq, _, cancelReason, gerr := lm.drive(r.Context(), gr, s.gens, s.jobs, func(t string) {
		anthropicEvent(ss, "content_block_delta", map[string]any{
			"type": "content_block_delta", "index": 0,
			"delta": map[string]any{"type": "text_delta", "text": t},
		})
	})
	stopBeat()
	if gerr != nil {
		anthropicEvent(ss, "content_block_stop", map[string]any{"type": "content_block_stop", "index": 0})
		anthropicStreamErr(ss, "generation failed: "+gerr.Error())
		return
	}
	anthropicEvent(ss, "content_block_stop", map[string]any{"type": "content_block_stop", "index": 0})
	reason, seq := anthropicStopReason(finish, stopSeq)
	anthropicMessageEnd(ss, reason, seq, nComp, cancelReason)
}

// streamMessagesTools runs a tool-bearing turn on /v1/messages: prose streams as a text content block
// while the model writes it (G21, where the family can stream prose safely — the shared tool turn,
// tool_turn.go), then one tool_use block per parsed call. With no call it is one text block.
//
// Before this, the route Claude Code uses buffered the WHOLE generation and sent only heartbeats
// (audit-2026-09-02 M-19 added those: the single `ping` at message_start was the only byte for the
// entire generation, measured elsewhere at 1682.6s against a 300s idle timeout) — only the OpenAI
// route streamed prose during a tool call.
func (s *server) streamMessagesTools(w http.ResponseWriter, r *http.Request, ss *sseWriter, lm *loadedModel, gr genRequest, tools []chat.Tool) {
	ts := &anthropicTextStream{ss: ss}
	t, gerr := s.runToolTurn(r.Context(), lm, gr, tools, ss, ts.push)
	if gerr != nil && !errors.Is(gerr, errProseDiverged) {
		anthropicStreamErr(ss, "generation failed: "+gerr.Error())
		return
	}
	if gerr != nil {
		anthropicStreamErr(ss, gerr.Error())
		return
	}
	if t.cancelReason != "" {
		// K1: cancelled mid-generation — a partial buffer, no tool call is parsed out of it. Close any
		// open prose block so the stream stays well-formed.
		ts.close()
		reason, seq := anthropicStopReason(t.finish, t.stopSeq)
		anthropicMessageEnd(ss, reason, seq, t.nComp, t.cancelReason)
		return
	}

	if len(t.calls) == 0 { // model declined to call: one text block with the output
		ts.finish(t.rest, true)
		reason, seq := anthropicStopReason(t.finish, t.stopSeq)
		anthropicMessageEnd(ss, reason, seq, t.nComp, "")
		return
	}

	idx := ts.finish(t.rest, false)
	for _, c := range t.calls {
		tu := toolUseBlock(c)
		anthropicEvent(ss, "content_block_start", map[string]any{
			"type": "content_block_start", "index": idx,
			"content_block": map[string]any{"type": "tool_use", "id": tu["id"], "name": tu["name"], "input": map[string]any{}},
		})
		args := strings.TrimSpace(string(c.Arguments))
		if args == "" {
			args = "{}"
		}
		anthropicEvent(ss, "content_block_delta", map[string]any{
			"type": "content_block_delta", "index": idx,
			"delta": map[string]any{"type": "input_json_delta", "partial_json": args},
		})
		anthropicEvent(ss, "content_block_stop", map[string]any{"type": "content_block_stop", "index": idx})
		idx++
	}
	anthropicMessageEnd(ss, "tool_use", nil, t.nComp, "")
}

// anthropicTextStream is the prose half of a streamed tool turn: text content block 0, opened lazily.
// Whitespace-only prose before a call has never produced a text block on this route (the buffered
// version dropped a lead that was only whitespace), so leading whitespace is held until real text
// arrives; if a call comes first, it is dropped exactly as before.
type anthropicTextStream struct {
	ss      *sseWriter
	open    bool
	pending strings.Builder // prose received but not yet sent: whitespace before the block opens
}

func (a *anthropicTextStream) push(text string) {
	if !a.open {
		a.pending.WriteString(text)
		if strings.TrimSpace(a.pending.String()) == "" {
			return
		}
		anthropicEvent(a.ss, "content_block_start", map[string]any{
			"type": "content_block_start", "index": 0,
			"content_block": map[string]any{"type": "text", "text": ""},
		})
		a.open = true
		text = a.pending.String()
		a.pending.Reset()
	}
	anthropicEvent(a.ss, "content_block_delta", map[string]any{
		"type": "content_block_delta", "index": 0,
		"delta": map[string]any{"type": "text_delta", "text": text},
	})
}

// finish sends the prose the turn still held back and closes the text block, returning the index the
// next content block takes. always forces a text block even for empty or whitespace-only prose (the
// no-call turn is always one text block); otherwise whitespace-only prose yields no block.
func (a *anthropicTextStream) finish(rest string, always bool) int {
	if a.open {
		if rest != "" {
			a.push(rest)
		}
		a.close()
		return 1
	}
	text := a.pending.String() + rest
	if always || strings.TrimSpace(text) != "" {
		streamTextBlock(a.ss, 0, text)
		return 1
	}
	return 0
}

func (a *anthropicTextStream) close() {
	if a.open {
		anthropicEvent(a.ss, "content_block_stop", map[string]any{"type": "content_block_stop", "index": 0})
		a.open = false
	}
}

// streamTextBlock emits a complete text content block (start, one delta, stop) —
// used for the buffered fallbacks where the text is already whole.
func streamTextBlock(ss *sseWriter, index int, text string) {
	anthropicEvent(ss, "content_block_start", map[string]any{
		"type": "content_block_start", "index": index,
		"content_block": map[string]any{"type": "text", "text": ""},
	})
	if text != "" {
		anthropicEvent(ss, "content_block_delta", map[string]any{
			"type": "content_block_delta", "index": index,
			"delta": map[string]any{"type": "text_delta", "text": text},
		})
	}
	anthropicEvent(ss, "content_block_stop", map[string]any{"type": "content_block_stop", "index": index})
}

// anthropicMessageEnd writes the closing message_delta (stop reason + final
// output token count) and message_stop. There is no [DONE] terminator.
// cancelReason, when non-empty (K1, docs/tasks/task-halt-2026-09.md), adds one more named event
// naming the admin-cancel reason before message_stop — reason == "cancelled" alone doesn't
// carry WHY, and a client must not be able to mistake this for a natural end_turn.
func anthropicMessageEnd(ss *sseWriter, reason string, seq any, nComp int, cancelReason string) {
	anthropicEvent(ss, "message_delta", map[string]any{
		"type":  "message_delta",
		"delta": map[string]any{"stop_reason": reason, "stop_sequence": seq},
		"usage": map[string]any{"output_tokens": nComp},
	})
	if cancelReason != "" {
		anthropicEvent(ss, "goinfer_cancelled", map[string]any{"type": "goinfer_cancelled", "reason": cancelReason})
	}
	anthropicEvent(ss, "message_stop", map[string]any{"type": "message_stop"})
}
