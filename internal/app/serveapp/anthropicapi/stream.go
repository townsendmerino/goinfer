package anthropicapi

import (
	"net/http"
	"strings"

	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/internal/app/serveapp/jobs"
	"github.com/townsendmerino/goinfer/internal/app/serveapp/streamframe"
)

// SSEStart initializes SSE headers for an Anthropic /v1/messages stream.
func SSEStart(w http.ResponseWriter) (*streamframe.SSEWriter, bool) {
	f, ok := w.(http.Flusher)
	if !ok {
		WriteErr(w, http.StatusInternalServerError, "api_error", "streaming unsupported")
		return nil, false
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	return streamframe.NewSSEWriter(w, f), true
}

// SendMessageStart emits the opening message_start and ping SSE events of an Anthropic stream.
func SendMessageStart(ss *streamframe.SSEWriter, id, model string, inputTokens int) {
	streamframe.SendEvent(ss, "message_start", map[string]any{
		"type": "message_start",
		"message": map[string]any{
			"id": id, "type": "message", "role": "assistant", "model": model,
			"content": []any{}, "stop_reason": nil, "stop_sequence": nil,
			"usage": map[string]any{"input_tokens": inputTokens, "output_tokens": 0},
		},
	})
	streamframe.SendEvent(ss, "ping", map[string]any{"type": "ping"})
}

// EmitStreamedToolCalls streams each parsed tool call as content_block_start / input_json_delta / content_block_stop events.
func EmitStreamedToolCalls(ss *streamframe.SSEWriter, idx int, calls []chat.ToolCall) int {
	for _, c := range calls {
		tu := ToolUseBlock(c)
		streamframe.SendEvent(ss, "content_block_start", map[string]any{
			"type": "content_block_start", "index": idx,
			"content_block": map[string]any{"type": "tool_use", "id": tu["id"], "name": tu["name"], "input": map[string]any{}},
		})
		args := strings.TrimSpace(string(c.Arguments))
		if args == "" {
			args = "{}"
		}
		streamframe.SendEvent(ss, "content_block_delta", map[string]any{
			"type": "content_block_delta", "index": idx,
			"delta": map[string]any{"type": "input_json_delta", "partial_json": args},
		})
		streamframe.SendEvent(ss, "content_block_stop", map[string]any{"type": "content_block_stop", "index": idx})
		idx++
	}
	return idx
}

// MessageEnd emits message_delta, optional goinfer_cancelled, and message_stop at the end of an Anthropic stream.
func MessageEnd(ss *streamframe.SSEWriter, clock *jobs.TurnClock, reason string, seq any, usage map[string]any, cancelReason string, empty *streamframe.EmptyAnswer, fromReasoning bool) {
	md := clock.AddTurn(map[string]any{
		"type":  "message_delta",
		"delta": map[string]any{"stop_reason": reason, "stop_sequence": seq},
		"usage": usage,
	})
	if empty != nil {
		md["goinfer_empty_answer"] = empty
	}
	if fromReasoning {
		md[streamframe.FromReasoningField] = true
	}
	streamframe.SendEvent(ss, "message_delta", md)
	if cancelReason != "" {
		streamframe.SendEvent(ss, "goinfer_cancelled", map[string]any{"type": "goinfer_cancelled", "reason": cancelReason})
	}
	streamframe.SendEvent(ss, "message_stop", map[string]any{"type": "message_stop"})
}
