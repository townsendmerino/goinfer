package serveapp

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/constrain"
)

// handleChatTools serves /v1/chat/completions when tools are present: it renders
// the tool declarations into the prompt (per family), constrains the call when
// unambiguous (one tool or a forced function), generates, then PARSES the output
// against the family's template into OpenAI tool_calls.
func (s *server) handleChatTools(w http.ResponseWriter, r *http.Request, req chatReq) {
	s.withModel(w, req.Model, func(lm *loadedModel) { s.serveChatToolsWith(w, r, req, lm) })
}

// serveChatToolsWith runs the tool-calling generation. Reached ONLY through withModel (liveness RLock held).
func (s *server) serveChatToolsWith(w http.ResponseWriter, r *http.Request, req chatReq, lm *loadedModel) {
	if lm.tmpl == nil || !lm.tmpl.SupportsTools() {
		writeErr(w, http.StatusBadRequest, "this model has no tool-calling template")
		return
	}
	tools := make([]chat.Tool, len(req.Tools))
	for i, t := range req.Tools {
		tools[i] = chat.Tool{Name: t.Function.Name, Description: t.Function.Description, Parameters: t.Function.Parameters}
	}
	system, turns := messagesToTurns(req.Messages)
	ids, err := lm.tk.EncodeSegments(lm.tmpl.RenderToolsSegments(system, turns, tools), false) // M25: harden the no-tools/content spans
	if err != nil {
		writeServerErr(w, "encode: "+err.Error())
		return
	}
	gr, err := lm.prepare(req.sampling, ids, lm.adapter == "")
	if err != nil {
		writeErr(w, http.StatusBadRequest, err.Error())
		return
	}

	// Tight when unambiguous: a forced function or a lone tool ⇒ constrain the
	// call to that tool's schema (when the family has a JSON call form). A NAMED
	// tool_choice that cannot be constrained is a 400, not a silent unconstrained
	// decode (audit M-05).
	forced := forcedTool(req.ToolChoice, tools)
	namedForce := toolChoiceMode(req.ToolChoice) == "function"
	if cerr := constrainForcedTool(lm, &gr, forced, namedForce); cerr != nil {
		writeErr(w, http.StatusBadRequest, cerr.Error())
		return
	}

	if !lm.enter(w) {
		return
	}
	defer lm.exit()
	id := "chatcmpl-" + reqID()
	created := time.Now().Unix()

	// Tool decisions need the whole output, so buffer (even when streaming).
	var sb strings.Builder
	finish, nComp, _, _, gerr := lm.drive(r.Context(), gr, func(t string) { sb.WriteString(t) })
	if gerr != nil { // headers not yet sent (SSE starts below), so a 500 is still valid for both modes
		writeServerErr(w, "generation failed: "+gerr.Error())
		return
	}
	calls, lead := lm.tmpl.ParseToolCalls(sb.String())

	msg := map[string]any{"role": "assistant"}
	if len(calls) > 0 {
		msg["content"] = nil
		if lead != "" {
			msg["content"] = lead
		}
		msg["tool_calls"] = toAPICalls(calls)
		finish = "tool_calls"
	} else {
		msg["content"] = sb.String()
	}
	choice := map[string]any{"index": 0, "message": msg, "finish_reason": finish}
	usagev := usage{len(gr.promptIDs), nComp, len(gr.promptIDs) + nComp}

	if req.Stream {
		f, ok := sseStart(w)
		if !ok {
			return
		}
		// One delta carrying the whole message, then the finish — valid SSE, just
		// not incremental (tool calls are decided from the full output).
		d := map[string]any{"role": "assistant"}
		if _, ok := msg["tool_calls"]; ok {
			d["tool_calls"] = streamToolCalls(calls)
		} else {
			d["content"] = msg["content"]
		}
		sseSend(w, f, map[string]any{"id": id, "object": "chat.completion.chunk", "created": created, "model": lm.name,
			"choices": []any{map[string]any{"index": 0, "delta": d, "finish_reason": nil}}})
		fin := finish
		sseSend(w, f, map[string]any{"id": id, "object": "chat.completion.chunk", "created": created, "model": lm.name,
			"choices": []any{map[string]any{"index": 0, "delta": map[string]any{}, "finish_reason": &fin}}})
		sseDone(w, f)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"id": id, "object": "chat.completion", "created": created, "model": lm.name,
		"choices": []any{choice}, "usage": usagev,
	})
}

// constrainForcedTool wires constrained decoding for a forced/lone tool call. When the
// caller explicitly NAMED the function (namedForce), a family/schema that cannot be
// constrained returns an error the handler renders as a 400: OpenAI and Anthropic both
// guarantee a named tool_choice produces that call, so silently decoding unconstrained
// (the model may emit prose, or a different tool) defeats the guarantee (audit M-05). The
// lone-tool convenience (namedForce=false) never errors — it is only an optimization, and
// the model is still free to answer in prose. Shared by /v1/chat/completions, /v1/responses,
// and the Anthropic Messages surface so all three degrade identically.
func constrainForcedTool(lm *loadedModel, gr *genRequest, forced *chat.Tool, namedForce bool) error {
	if forced == nil {
		return nil
	}
	prefix, suffix, argsKey, array, ok := lm.tmpl.ToolCallWrapper()
	if !ok {
		if namedForce {
			return fmt.Errorf("tool_choice forces function %q but this model's chat template has no constrainable tool-call form", forced.Name)
		}
		return nil
	}
	g, gerr := constrain.ToolCallGrammar(prefix, suffix, argsKey, forced.Name, array, forced.Parameters)
	if gerr != nil {
		if namedForce {
			return fmt.Errorf("tool_choice forces function %q but its schema cannot be constrained: %v", forced.Name, gerr)
		}
		return nil
	}
	eos := append(append([]int(nil), lm.eosIDs...), lm.stopIDs...)
	m := constrain.NewMasker(g, constrain.TokenBytes(lm.vocab, lm.tk.TokenText), eos).StopWhenComplete()
	gr.sp.LogitProcessor = m.Process
	gr.masker = m // enables grammar-fused speculative decode (drive)
	return nil
}

// toAPICalls renders parsed calls in the OpenAI response shape (arguments is a
// JSON string).
func toAPICalls(calls []chat.ToolCall) []map[string]any {
	out := make([]map[string]any, len(calls))
	for i, c := range calls {
		args := string(c.Arguments)
		if args == "" {
			args = "{}"
		}
		id := c.ID
		if id == "" {
			id = "call_" + reqID()
		}
		out[i] = map[string]any{"id": id, "type": "function",
			"function": map[string]any{"name": c.Name, "arguments": args}}
	}
	return out
}

// streamToolCalls adds the index field each delta tool_call carries.
func streamToolCalls(calls []chat.ToolCall) []map[string]any {
	out := toAPICalls(calls)
	for i := range out {
		out[i]["index"] = i
	}
	return out
}

// toolChoiceMode returns "auto" (default), "none", or "required"/"function" from
// the tool_choice field (a string or an object).
func toolChoiceMode(raw json.RawMessage) string {
	if len(raw) == 0 {
		return "auto"
	}
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return s // "auto" | "none" | "required"
	}
	return "function" // an object → a specific function is named
}

// forcedTool returns the single tool the call must be (a forced function, or the
// lone tool) — the "tight when unambiguous" case. nil means don't constrain.
func forcedTool(toolChoice json.RawMessage, tools []chat.Tool) *chat.Tool {
	switch toolChoiceMode(toolChoice) {
	case "none":
		return nil
	case "function":
		var c struct {
			Function struct{ Name string } `json:"function"`
		}
		if json.Unmarshal(toolChoice, &c) == nil {
			for i := range tools {
				if tools[i].Name == c.Function.Name {
					return &tools[i]
				}
			}
		}
		return nil
	}
	if len(tools) == 1 { // lone tool ⇒ unambiguous
		return &tools[0]
	}
	return nil
}
