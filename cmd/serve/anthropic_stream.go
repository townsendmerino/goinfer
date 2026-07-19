package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
)

// Anthropic SSE differs from the OpenAI flavor: named events
// (event: <type>\ndata: <json>\n\n), a strict block sequence, and NO [DONE]
// terminator — so these are separate from the sseStart/sseSend/sseDone helpers
// rather than bending them.

func anthropicSSEStart(w http.ResponseWriter) (http.Flusher, bool) {
	f, ok := w.(http.Flusher)
	if !ok {
		writeAnthropicErr(w, http.StatusInternalServerError, "api_error", "streaming unsupported")
		return nil, false
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	return f, true
}

// anthropicEvent writes one named SSE event.
func anthropicEvent(w http.ResponseWriter, f http.Flusher, event string, v any) {
	b, _ := json.Marshal(v)
	fmt.Fprintf(w, "event: %s\ndata: %s\n\n", event, b)
	f.Flush()
}

// anthropicStreamErr emits an Anthropic `error` event mid-stream, when a
// generation fails after message_start has already been sent (200, headers
// flushed — no status code left to set). M1.
func anthropicStreamErr(w http.ResponseWriter, f http.Flusher, msg string) {
	anthropicEvent(w, f, "error", map[string]any{
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
	f, ok := anthropicSSEStart(w)
	if !ok {
		return
	}
	id := "msg_" + reqID()
	anthropicEvent(w, f, "message_start", map[string]any{
		"type": "message_start",
		"message": map[string]any{
			"id": id, "type": "message", "role": "assistant", "model": lm.name,
			"content": []any{}, "stop_reason": nil, "stop_sequence": nil,
			"usage": map[string]any{"input_tokens": len(gr.promptIDs), "output_tokens": 0},
		},
	})
	anthropicEvent(w, f, "ping", map[string]any{"type": "ping"}) // liveness check; cheap insurance

	if toolsActive {
		s.streamMessagesTools(w, r, f, lm, gr)
		return
	}

	// Text: a single content block streamed token by token.
	anthropicEvent(w, f, "content_block_start", map[string]any{
		"type": "content_block_start", "index": 0,
		"content_block": map[string]any{"type": "text", "text": ""},
	})
	finish, nComp, _, stopSeq, gerr := lm.drive(r.Context(), gr, func(t string) {
		anthropicEvent(w, f, "content_block_delta", map[string]any{
			"type": "content_block_delta", "index": 0,
			"delta": map[string]any{"type": "text_delta", "text": t},
		})
	})
	if gerr != nil {
		anthropicEvent(w, f, "content_block_stop", map[string]any{"type": "content_block_stop", "index": 0})
		anthropicStreamErr(w, f, "generation failed: "+gerr.Error())
		return
	}
	anthropicEvent(w, f, "content_block_stop", map[string]any{"type": "content_block_stop", "index": 0})
	reason, seq := anthropicStopReason(finish, stopSeq)
	anthropicMessageEnd(w, f, reason, seq, nComp)
}

// streamMessagesTools buffers the generation (a tool decision needs the whole
// output), then emits an optional leading text block and one tool_use block per
// call. When no call is parsed it degrades to a single text block.
func (s *server) streamMessagesTools(w http.ResponseWriter, r *http.Request, f http.Flusher, lm *loadedModel, gr genRequest) {
	var sb strings.Builder
	finish, nComp, _, stopSeq, gerr := lm.drive(r.Context(), gr, func(t string) { sb.WriteString(t) })
	if gerr != nil {
		anthropicStreamErr(w, f, "generation failed: "+gerr.Error())
		return
	}
	calls, lead := lm.tmpl.ParseToolCalls(sb.String())

	if len(calls) == 0 { // model declined to call: one text block with the output
		streamTextBlock(w, f, 0, sb.String())
		reason, seq := anthropicStopReason(finish, stopSeq)
		anthropicMessageEnd(w, f, reason, seq, nComp)
		return
	}

	idx := 0
	if strings.TrimSpace(lead) != "" {
		streamTextBlock(w, f, idx, lead)
		idx++
	}
	for _, c := range calls {
		tu := toolUseBlock(c)
		anthropicEvent(w, f, "content_block_start", map[string]any{
			"type": "content_block_start", "index": idx,
			"content_block": map[string]any{"type": "tool_use", "id": tu["id"], "name": tu["name"], "input": map[string]any{}},
		})
		args := strings.TrimSpace(string(c.Arguments))
		if args == "" {
			args = "{}"
		}
		anthropicEvent(w, f, "content_block_delta", map[string]any{
			"type": "content_block_delta", "index": idx,
			"delta": map[string]any{"type": "input_json_delta", "partial_json": args},
		})
		anthropicEvent(w, f, "content_block_stop", map[string]any{"type": "content_block_stop", "index": idx})
		idx++
	}
	anthropicMessageEnd(w, f, "tool_use", nil, nComp)
}

// streamTextBlock emits a complete text content block (start, one delta, stop) —
// used for the buffered fallbacks where the text is already whole.
func streamTextBlock(w http.ResponseWriter, f http.Flusher, index int, text string) {
	anthropicEvent(w, f, "content_block_start", map[string]any{
		"type": "content_block_start", "index": index,
		"content_block": map[string]any{"type": "text", "text": ""},
	})
	if text != "" {
		anthropicEvent(w, f, "content_block_delta", map[string]any{
			"type": "content_block_delta", "index": index,
			"delta": map[string]any{"type": "text_delta", "text": text},
		})
	}
	anthropicEvent(w, f, "content_block_stop", map[string]any{"type": "content_block_stop", "index": index})
}

// anthropicMessageEnd writes the closing message_delta (stop reason + final
// output token count) and message_stop. There is no [DONE] terminator.
func anthropicMessageEnd(w http.ResponseWriter, f http.Flusher, reason string, seq any, nComp int) {
	anthropicEvent(w, f, "message_delta", map[string]any{
		"type":  "message_delta",
		"delta": map[string]any{"stop_reason": reason, "stop_sequence": seq},
		"usage": map[string]any{"output_tokens": nComp},
	})
	anthropicEvent(w, f, "message_stop", map[string]any{"type": "message_stop"})
}
