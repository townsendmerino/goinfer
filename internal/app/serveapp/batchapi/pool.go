package batchapi

import (
	"context"
	"errors"
	"net/http"
	"sync/atomic"
	"time"

	"github.com/townsendmerino/goinfer/internal/app/serveapp/anthropicapi"
	"github.com/townsendmerino/goinfer/internal/app/serveapp/stores"
	"github.com/townsendmerino/goinfer/internal/app/serveapp/wirecore"
)

// LineWorkers bounds concurrent pre-queue goroutines per batch.
const LineWorkers = 8

// RunPool dispatches indices 0..n-1 across at most LineWorkers goroutines.
func RunPool(n int, fn func(i int)) {
	if n <= 0 {
		return
	}
	workers := min(n, LineWorkers)
	var next atomic.Int64
	for range workers {
		go runWorker(n, &next, fn)
	}
}

func runWorker(n int, next *atomic.Int64, fn func(i int)) {
	for {
		i := int(next.Add(1) - 1)
		if i >= n {
			return
		}
		fn(i)
	}
}

// ValidateChatLine performs request-shape validation for one OpenAI batch line before model lookup.
func ValidateChatLine(customID string, req wirecore.ChatReq, statusCancelled int, cancelling bool) (stores.BatchLineResult, bool) {
	if cancelling {
		return stores.LineError(customID, statusCancelled, "cancelled", "batch cancelled"), false
	}
	if len(req.Messages) == 0 {
		return stores.LineError(customID, 400, "invalid_request_error",
			"messages is required and must contain at least one message"), false
	}
	if wirecore.HasChatMedia(req.Messages) {
		return stores.LineError(customID, 400, "invalid_request_error",
			"batch requests do not support image or audio inputs this pass; use /v1/chat/completions"), false
	}
	if len(req.Tools) > 0 && wirecore.ToolChoiceMode(req.ToolChoice) != "none" {
		return stores.LineError(customID, 400, "invalid_request_error",
			"batch requests do not support tools this pass; use /v1/chat/completions"), false
	}
	return stores.BatchLineResult{}, true
}

// ValidateMessageLine performs request-shape validation for one Anthropic batch line before model lookup.
func ValidateMessageLine(customID string, req *anthropicapi.Req, statusCancelled int, cancelling bool) (stores.BatchLineResult, bool) {
	if cancelling {
		return stores.LineError(customID, statusCancelled, "cancelled", "batch cancelled"), false
	}
	if req.MaxTokens == nil || *req.MaxTokens <= 0 {
		return stores.LineError(customID, 400, "invalid_request_error", "max_tokens is required and must be > 0"), false
	}
	if len(req.Messages) == 0 {
		return stores.LineError(customID, 400, "invalid_request_error", "messages must not be empty"), false
	}
	if anthropicapi.ContainsImages(req) {
		return stores.LineError(customID, 400, "invalid_request_error",
			"batch requests do not support image inputs this pass; use /v1/messages"), false
	}
	if mode, _ := anthropicapi.ToolMode(req.ToolChoice); len(req.Tools) > 0 && mode != "none" {
		return stores.LineError(customID, 400, "invalid_request_error",
			"batch requests do not support tools this pass; use /v1/messages"), false
	}
	return stores.BatchLineResult{}, true
}

// ChatCompletionBody builds a completed OpenAI batch line's chat.completion body.
func ChatCompletionBody(jobID, model, content, reasoning, finish string, promptTokens, compTokens int, reused wirecore.PrefillReuse) map[string]any {
	bmsg := map[string]any{"role": "assistant", "content": content}
	if reasoning != "" {
		bmsg["reasoning_content"] = reasoning
	}
	return map[string]any{
		"id": jobID, "object": "chat.completion", "created": time.Now().Unix(), "model": model,
		"choices": []any{map[string]any{
			"index":         0,
			"message":       bmsg,
			"finish_reason": finish,
		}},
		"usage": wirecore.NewUsage(promptTokens, compTokens, reused),
	}
}

// MessageBody builds a completed Anthropic batch line's message body.
func MessageBody(jobID, model, answer, reasoning, finish, stopHit string, wantsThinking bool, promptTokens, reusedTokens, compTokens int) map[string]any {
	reason, seq := anthropicapi.StopReason(finish, stopHit)
	content := []map[string]any{anthropicapi.TextBlock(answer)}
	if wantsThinking && reasoning != "" {
		content = append([]map[string]any{anthropicapi.ThinkingBlock(reasoning, "")}, content...)
		if answer == "" {
			content = content[:1]
		}
	}
	return map[string]any{
		"id": jobID, "type": "message", "role": "assistant", "model": model,
		"content":       content,
		"stop_reason":   reason,
		"stop_sequence": seq,
		"usage":         anthropicapi.Usage(promptTokens, reusedTokens, compTokens),
	}
}

// ClassifyDriveError converts a failed or cancelled batch-line drive into a BatchLineResult, prioritizing server shutdown.
func ClassifyDriveError(bgCtx context.Context, shutdownErr error, statusCancelled int, customID string, gerr error, cancelReason string) (r stores.BatchLineResult, failed bool) {
	switch {
	case shutdownErr != nil && errors.Is(context.Cause(bgCtx), shutdownErr):
		return stores.LineError(customID, http.StatusServiceUnavailable, "server_shutdown", "server shutting down"), true
	case gerr != nil:
		return stores.LineError(customID, 500, "api_error", "generation failed: "+gerr.Error()), true
	case cancelReason != "":
		return stores.LineError(customID, statusCancelled, "cancelled", "generation cancelled: "+cancelReason), true
	}
	return stores.BatchLineResult{}, false
}
