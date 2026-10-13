package serveapp

import (
	"errors"
	"fmt"
	"net/http"

	"github.com/townsendmerino/goinfer/chat"
)

// contextOverflowError is a prompt that fills or exceeds the context window it was sent to. It has two readings,
// and which one decides the words a client is given (docs/tasks/task-integration-harness-2026-10.md, "I1"):
//
//   - the conversation is too long, so compacting it would make the request fit: each dialect gets the wording its
//     coding agents recognise as an overflow, and they compact;
//   - the request's fixed part (the system prompt and the tool definitions) leaves no room for a reply, so no
//     compaction can help: the message is for the operator and is worded so that no agent's overflow matcher fires,
//     because an agent that compacts here compacts again on every retry.
type contextOverflowError struct {
	prompt, ctx, modelWindow int // tokens: the prompt, the window that refused it, the model's own window
	fixed                    int // tokens of the system prompt and tool definitions; -1 when the route did not measure them
	reserved                 int // tokens kept for a reply (overflowReserve)
}

// helpless reports whether compaction cannot make the request fit: the fixed part and the room for a reply already
// reach the window. A route that did not measure its fixed part is never helpless.
func (e *contextOverflowError) helpless() bool {
	return e.fixed >= 0 && e.fixed+e.reserved >= e.ctx
}

// Error is the OpenAI-dialect message, which the routes without a dialect of their own (jobs, batches) record.
func (e *contextOverflowError) Error() string { return e.message("openai") }

// message is the error text for a dialect: "anthropic" for /v1/messages, anything else for the OpenAI routes.
func (e *contextOverflowError) message(dialect string) string {
	if e.helpless() {
		msg := fmt.Sprintf("this request cannot fit this server, and shortening the conversation will not help: its system prompt and tool definitions take %d tokens, a reply needs %d more, and the server holds %d positions",
			e.fixed, e.reserved, e.ctx)
		if e.modelWindow > e.ctx {
			return msg + fmt.Sprintf("; restart goinfer-serve with -ctx %d or more (up to %d)", min(e.modelWindow, e.prompt+e.reserved), e.modelWindow)
		}
		return msg + "; that is the model's own size, so only a shorter system prompt or fewer tools can fit"
	}
	var msg string
	if dialect == "anthropic" {
		msg = fmt.Sprintf("prompt is too long: %d tokens > %d maximum", e.prompt, e.ctx)
	} else {
		msg = fmt.Sprintf("This model's maximum context length is %d tokens. However, your messages resulted in %d tokens (context_length_exceeded).", e.ctx, e.prompt)
	}
	// The window that refused the prompt is the server's resident KV capacity, not the model's: say what raises it.
	if e.modelWindow > e.ctx {
		msg += fmt.Sprintf(" This is the server's GPU context (-ctx), not the model's limit (%d tokens): restart goinfer-serve with -ctx %d or more (up to %d) to accept it.",
			e.modelWindow, e.prompt+1, e.modelWindow)
	}
	return msg
}

// overflowReserve is the room a reply is counted as needing when a prompt overflows: the request's own output cap,
// at least 1,024 tokens and at most a quarter of the window. The cap matters because the server clamps the output to
// the room left anyway, and a client that asks for 32,000 tokens on every request would otherwise make every
// overflow look as if compaction could not help.
func overflowReserve(maxTokens, ctx int) int {
	hi := max(ctx/4, 1)
	return min(max(maxTokens, min(1024, hi)), hi)
}

// newContextOverflow returns the overflow error for a prompt of promptLen tokens against a window of ctx, or nil
// when it fits. fixed is the measured fixed part, or nil when the route has none to measure. ctx <= 0 (unknown)
// never rejects.
func newContextOverflow(promptLen, ctx, modelWindow, maxTokens int, fixed func() int) error {
	if ctx <= 0 || promptLen < ctx {
		return nil
	}
	e := &contextOverflowError{prompt: promptLen, ctx: ctx, modelWindow: modelWindow, fixed: -1, reserved: overflowReserve(maxTokens, ctx)}
	if fixed != nil {
		e.fixed = fixed()
	}
	return e
}

// fixedPromptTokens is the request's fixed part in tokens: the prompt this template renders for the system text and
// the tool definitions with one empty user turn, which is what is left of a conversation however far it is
// compacted. -1 when it cannot be rendered or encoded.
func (lm *loadedModel) fixedPromptTokens(tm *chat.Template, system string, tools []chat.Tool) int {
	empty := []chat.Turn{{Role: "user", Content: ""}}
	var ids []int
	var err error
	if len(tools) > 0 && tm != nil {
		ids, err = lm.tk.EncodeSegments(tm.RenderToolsSegments(system, empty, tools), false)
	} else {
		ids, err = lm.promptForT(tm, system, empty)
	}
	if err != nil {
		return -1
	}
	return len(ids)
}

// writePrepareErr answers a failed prepare on an OpenAI-dialect route. A context overflow that compaction can cure
// carries error.code "context_length_exceeded", the field the OpenAI SDKs and the agents built on them read; one
// it cannot cure carries no code.
func writePrepareErr(w http.ResponseWriter, err error) {
	if e, ok := errors.AsType[*contextOverflowError](err); ok {
		body := map[string]any{"message": e.message("openai"), "type": "invalid_request_error", "param": nil, "code": nil}
		if !e.helpless() {
			body["code"] = "context_length_exceeded"
		}
		writeJSON(w, http.StatusBadRequest, map[string]any{"error": body})
		return
	}
	writeErr(w, prepareErrStatus(err), err.Error())
}

// writeAnthropicPrepareErr is writePrepareErr for /v1/messages.
func writeAnthropicPrepareErr(w http.ResponseWriter, err error) {
	if e, ok := errors.AsType[*contextOverflowError](err); ok {
		writeAnthropicErr(w, http.StatusBadRequest, "invalid_request_error", e.message("anthropic"))
		return
	}
	writeAnthropicErr(w, prepareErrStatus(err), "invalid_request_error", err.Error())
}

// writeResponsesPrepareErr is writePrepareErr for /v1/responses. A streaming client that can compact is told through
// the stream, as a response.failed event whose error code is context_length_exceeded, which is where a Responses
// client reads it; every other case is the OpenAI body.
func writeResponsesPrepareErr(w http.ResponseWriter, err error, stream bool, id, model string, created int64) {
	e, ok := errors.AsType[*contextOverflowError](err)
	if !ok || !stream || e.helpless() {
		writePrepareErr(w, err)
		return
	}
	ss, ok := sseStart(w)
	if !ok {
		return
	}
	resp := responseObject(id, model, created, "failed", []any{}, e.prompt, 0)
	resp["error"] = map[string]any{"code": "context_length_exceeded", "message": e.message("openai")}
	sseEvent(ss, "response.failed", map[string]any{"type": "response.failed", "response": resp})
}
