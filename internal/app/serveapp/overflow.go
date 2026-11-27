package serveapp

import (
	"errors"
	"net/http"

	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/internal/app/serveapp/wirecore"
)

type overflowFields = wirecore.Overflow

// contextOverflowError wraps wirecore.Overflow for OpenAI, Responses, and Anthropic prepare-error rendering.
type contextOverflowError struct {
	overflowFields
}

func (e *contextOverflowError) helpless() bool          { return e.Helpless() }
func (e *contextOverflowError) Error() string           { return e.Message("openai") }
func (e *contextOverflowError) message(d string) string { return e.Message(d) }

var overflowReserve = wirecore.OverflowReserve

func newContextOverflow(promptLen, ctx, modelWindow, maxTokens int, fixed func() int) error {
	if ctx <= 0 || promptLen < ctx {
		return nil
	}
	e := &contextOverflowError{wirecore.Overflow{
		Prompt:      promptLen,
		Ctx:         ctx,
		ModelWindow: modelWindow,
		Fixed:       -1,
		Reserved:    overflowReserve(maxTokens, ctx),
	}}
	if fixed != nil {
		e.Fixed = fixed()
	}
	return e
}

func (lm *loadedModel) fixedPromptTokens(tm *chat.Template, system string, tools []chat.Tool) int {
	return wirecore.FixedPromptTokens(lm.tk, tm, system, tools, lm.encode)
}

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

func writeAnthropicPrepareErr(w http.ResponseWriter, err error) {
	if e, ok := errors.AsType[*contextOverflowError](err); ok {
		writeAnthropicErr(w, http.StatusBadRequest, "invalid_request_error", e.message("anthropic"))
		return
	}
	writeAnthropicErr(w, prepareErrStatus(err), "invalid_request_error", err.Error())
}

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
	resp := responseObject(id, model, created, "failed", []any{}, e.Prompt, 0)
	resp["error"] = map[string]any{"code": "context_length_exceeded", "message": e.message("openai")}
	sseEvent(ss, "response.failed", map[string]any{"type": "response.failed", "response": resp})
}
