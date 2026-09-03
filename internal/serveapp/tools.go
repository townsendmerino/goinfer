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
	// G1c, extended (audit-2026-09-02 M-21). The guard reached three routes; this was one of the
	// five that still ran a full O(n) tokenize over an arbitrary body before rejecting it — the
	// G1c comment prices what it removes at "~27 s of BPE + gigabytes of ids" on a multi-MiB body.
	if err := lm.promptTooLargeForContext(chatInputBytes(req.Messages)); err != nil {
		writeErr(w, http.StatusBadRequest, err.Error())
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
	if cerr := constrainForcedTool(lm, &gr, forced, namedForce, tools); cerr != nil {
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
	//
	// G19: when streaming, SSE now starts BEFORE the generation and a keep-alive
	// comment frame ticks while the buffer fills. Without it this path sent zero
	// bytes for the whole generation — measured at 1682.6s to first byte against a
	// harness whose idle timeout was 300s, which no output can survive however
	// correct it is. The buffering itself is unchanged, and comment frames carry no
	// data, so tool-call parsing sees exactly what it saw before.
	//
	// The cost of starting SSE early: a generation error can no longer be a 500 on
	// the streaming path, because the headers are already flushed. That is the M1
	// convention sseErr exists for and what the non-tool streaming paths already do.
	// The non-streaming path below keeps its 500 unchanged.
	var ss *sseWriter
	if req.Stream {
		var ok bool
		if ss, ok = sseStart(w); !ok {
			return
		}
	}
	var sb strings.Builder

	// G21: stream prose INCREMENTALLY where the family allows it, instead of
	// holding the whole generation. Only families whose ParseToolCalls computes
	// lead as a raw untrimmed prefix qualify (Template.ToolCallOpener); the rest
	// keep the G19 behavior exactly — buffered, with heartbeats.
	//
	// The invariant: every byte emitted here must be a prefix of the lead the
	// parser computes below. StreamableLen guarantees it by holding back any
	// suffix that could still grow into the opener, and `toolStarted` stops the
	// stream for good once the opener appears — everything after it belongs to a
	// call, not to prose.
	opener, incremental := "", false
	if ss != nil && lm.tmpl != nil {
		opener, incremental = lm.tmpl.ToolCallOpener()
	}
	var streamed strings.Builder // exactly what left as content deltas
	var prose *chat.ProseStreamer
	if incremental {
		prose = chat.NewProseStreamer(opener)
	}

	var stopBeat func()
	if ss != nil {
		// Heartbeats still run: on an incremental family they cover the silence
		// before the first token and inside a tool call, and on the others they
		// cover the whole generation as before.
		stopBeat = sseHeartbeat(ss)
	}
	finish, nComp, _, _, gerr := lm.drive(r.Context(), gr, func(t string) {
		sb.WriteString(t)
		if prose == nil {
			return
		}
		if out := prose.Push(t); out != "" {
			streamed.WriteString(out)
			sseSend(ss, chatChunk(id, created, lm.name, delta{Content: out}, nil))
		}
	})
	if stopBeat != nil {
		stopBeat() // joins the ticker goroutine before anything else writes to w
	}
	if gerr != nil {
		if ss != nil {
			sseErr(ss, "generation failed: "+gerr.Error())
			sseDone(ss)
			return
		}
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
		// Whatever prose already left as deltas (G21) must not be sent twice, and
		// the parser's view is authoritative: emit only the REMAINDER of it here.
		// On a non-incremental family `already` is empty and this is exactly the
		// G19 behavior — one delta carrying the whole message.
		already := streamed.String()
		full := ""
		if c, ok := msg["content"].(string); ok {
			full = c
		} else if lead != "" {
			full = lead // tool-call case: content is nil, the prose is the lead
		}
		if !strings.HasPrefix(full, already) {
			// Unreachable by construction — StreamableLen only releases bytes that
			// precede the opener, and lead is the raw prefix before it. If it ever
			// fires, bytes were sent that the parser does not agree are prose, and
			// they cannot be recalled: say so loudly rather than emit a stream that
			// silently disagrees with itself.
			sseErr(ss, "internal: streamed prose diverged from the parsed lead; the tool-call "+
				"stream for this family is not prefix-safe (G21)")
			sseDone(ss)
			return
		}
		d := map[string]any{"role": "assistant"}
		if _, ok := msg["tool_calls"]; ok {
			d["tool_calls"] = streamToolCalls(calls)
			if rest := full[len(already):]; rest != "" {
				// Prose that preceded the call and was still held back.
				sseSend(ss, chatChunk(id, created, lm.name, delta{Content: rest}, nil))
			}
		} else {
			d["content"] = full[len(already):]
		}
		sseSend(ss, map[string]any{"id": id, "object": "chat.completion.chunk", "created": created, "model": lm.name,
			"choices": []any{map[string]any{"index": 0, "delta": d, "finish_reason": nil}}})
		fin := finish
		sseSend(ss, map[string]any{"id": id, "object": "chat.completion.chunk", "created": created, "model": lm.name,
			"choices": []any{map[string]any{"index": 0, "delta": map[string]any{}, "finish_reason": &fin}}})
		sendUsage(ss, req.StreamOptions, id, created, lm.name, usagev)
		sseDone(ss)
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
func constrainForcedTool(lm *loadedModel, gr *genRequest, forced *chat.Tool, namedForce bool, tools []chat.Tool) error {
	if forced == nil {
		// N-18: a NAMED tool_choice that matched no tool used to land here and return nil —
		// the request then generated completely unconstrained, having asked for one specific
		// function. The 2026-08-05 audit made "named but unconstrainable" a 400 and left
		// "named but nonexistent" falling through, which is the louder of the two errors: the
		// caller has a typo or a stale tool list, and a prose answer looks like the model
		// simply chose not to call anything.
		if namedForce {
			names := make([]string, 0, len(tools))
			for _, t := range tools {
				names = append(names, t.Name)
			}
			avail := "none were supplied"
			if len(names) > 0 {
				avail = "available: " + strings.Join(names, ", ")
			}
			return fmt.Errorf("tool_choice names a function that is not in tools (%s)", avail)
		}
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
	// N-23: lm.cachedTokenBytes(), not a fresh constrain.TokenBytes. The table is ~152k entries and
	// this site rebuilt it PER REQUEST, before the queue — the 2026-08-05 audit's N-14 disposition
	// says it is built once per model, and openai.go:767 does exactly that. One call site kept the
	// uncached form.
	m := constrain.NewMasker(g, lm.cachedTokenBytes(), eos).StopWhenComplete()
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
