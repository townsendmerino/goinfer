package serveapp

import (
	"encoding/json"
	"net/http"
	"strings"
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
func (s *server) streamMessages(w http.ResponseWriter, r *http.Request, lm *loadedModel, gr genRequest, toolsActive bool) {
	ss, ok := anthropicSSEStart(w)
	if !ok {
		return
	}
	id := "msg_" + reqID()
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
		s.streamMessagesTools(w, r, ss, lm, gr)
		return
	}

	// Text: a single content block streamed token by token.
	anthropicEvent(ss, "content_block_start", map[string]any{
		"type": "content_block_start", "index": 0,
		"content_block": map[string]any{"type": "text", "text": ""},
	})
	finish, nComp, _, stopSeq, gerr := lm.drive(r.Context(), gr, func(t string) {
		anthropicEvent(ss, "content_block_delta", map[string]any{
			"type": "content_block_delta", "index": 0,
			"delta": map[string]any{"type": "text_delta", "text": t},
		})
	})
	if gerr != nil {
		anthropicEvent(ss, "content_block_stop", map[string]any{"type": "content_block_stop", "index": 0})
		anthropicStreamErr(ss, "generation failed: "+gerr.Error())
		return
	}
	anthropicEvent(ss, "content_block_stop", map[string]any{"type": "content_block_stop", "index": 0})
	reason, seq := anthropicStopReason(finish, stopSeq)
	anthropicMessageEnd(ss, reason, seq, nComp)
}

// streamMessagesTools buffers the generation (a tool decision needs the whole
// output), then emits an optional leading text block and one tool_use block per
// call. When no call is parsed it degrades to a single text block.
func (s *server) streamMessagesTools(w http.ResponseWriter, r *http.Request, ss *sseWriter, lm *loadedModel, gr genRequest) {
	var sb strings.Builder
	// THE THIRD BUFFER-THEN-STREAM SITE. G19 gave the OpenAI tool path and /v1/responses a
	// heartbeat and left this one emitting nothing after the single `ping` at message_start until
	// the entire generation finished — measured elsewhere at 1682.6s against a client whose idle
	// timeout was 300s. /v1/messages is the surface docs/server.md markets for Claude Code, where
	// tool-bearing requests are the NORM, so this was the worst of the three to have missed
	// (audit-2026-09-02 M-19). Safe to add only now: before sseWriter, a ticker writing here would
	// have raced the handler on the same ResponseWriter (C-06).
	stopBeat := sseHeartbeat(ss)
	finish, nComp, _, stopSeq, gerr := lm.drive(r.Context(), gr, func(t string) { sb.WriteString(t) })
	stopBeat()
	if gerr != nil {
		anthropicStreamErr(ss, "generation failed: "+gerr.Error())
		return
	}
	calls, lead := lm.tmpl.ParseToolCalls(sb.String())

	if len(calls) == 0 { // model declined to call: one text block with the output
		streamTextBlock(ss, 0, sb.String())
		reason, seq := anthropicStopReason(finish, stopSeq)
		anthropicMessageEnd(ss, reason, seq, nComp)
		return
	}

	idx := 0
	if strings.TrimSpace(lead) != "" {
		streamTextBlock(ss, idx, lead)
		idx++
	}
	for _, c := range calls {
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
	anthropicMessageEnd(ss, "tool_use", nil, nComp)
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
func anthropicMessageEnd(ss *sseWriter, reason string, seq any, nComp int) {
	anthropicEvent(ss, "message_delta", map[string]any{
		"type":  "message_delta",
		"delta": map[string]any{"stop_reason": reason, "stop_sequence": seq},
		"usage": map[string]any{"output_tokens": nComp},
	})
	anthropicEvent(ss, "message_stop", map[string]any{"type": "message_stop"})
}
