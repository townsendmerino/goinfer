package serveapp

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"slices"
	"strings"
	"time"

	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/internal/app/serveapp/jobs"
	"github.com/townsendmerino/goinfer/internal/app/serveapp/stores"
	"github.com/townsendmerino/goinfer/internal/app/serveapp/streamframe"
)

// OpenAI Responses API (/v1/responses). Stateless input (input, instructions, text.format, tools, streaming) plus
// store/previous_response_id through an in-memory ring, so a continued response is a prompt-prefix extension that
// rides the per-model sessionLRU for warm KV. Out of scope: hosted tools, file inputs.

type respText struct {
	Format *struct {
		Type   string          `json:"type"` // text | json_object | json_schema
		Name   string          `json:"name"`
		Schema json.RawMessage `json:"schema"`
	} `json:"format"`
}

type responseReq struct {
	Model              string          `json:"model"`
	Input              json.RawMessage `json:"input"` // string | message items
	Instructions       string          `json:"instructions"`
	MaxOutputTokens    *int            `json:"max_output_tokens"`
	Text               *respText       `json:"text"`
	Tools              []toolSpec      `json:"tools"`
	ToolChoice         json.RawMessage `json:"tool_choice"`
	Stream             bool            `json:"stream"`
	Store              *bool           `json:"store"` // default true (capped ring)
	PreviousResponseID string          `json:"previous_response_id"`
	// Reasoning.effort is OpenAI's thinking control ("none" turns thinking off). The reasoning text is returned as a
	// reasoning item (reasoning_items.go); output_text is always the clean answer.
	Reasoning *struct {
		Effort string `json:"effort"`
	} `json:"reasoning"`
	// Include lists the extra output a client asked for; "reasoning.encrypted_content" says it round-trips reasoning items,
	// so a reply with no reasoning gets a carrier-only item for its ids (carried_ids_mint.go).
	Include []string `json:"include"`
	sampling
}

func (r *responseReq) thinkRequest() thinkRequest {
	tr := thinkRequest{}
	if r.Reasoning != nil {
		tr.ReasoningEffort = r.Reasoning.Effort
	}
	return tr
}

// --- in-memory response store (id → conversation) ---

func newResponseStore(cap int) *stores.ResponseStore[chatMessage] {
	return stores.NewResponseStore(cap, responseMessagesBytes)
}

// responseMessagesBytes estimates what (model, messages) holds: the variable-length text of every message (content,
// reasoning, tool call ids, names and arguments) and of any carried-id blob (its text, ids at 8 bytes each, and MAC).
// Fixed per-message overhead is left out; the estimate bounds what a request can make the store hold, which this
// variable part dominates.
func responseMessagesBytes(model string, messages []chatMessage) int {
	n := len(model)
	for _, m := range messages {
		n += len(m.Role) + len(m.Content) + len(m.Name) + len(m.ToolCallID) + len(m.ReasoningContent) + len(m.Reasoning)
		for _, tc := range m.ToolCalls {
			n += len(tc.ID) + len(tc.Type) + len(tc.Function.Name) + len(tc.Function.Arguments)
		}
		if c := m.carried; c != nil {
			n += len(c.Reason)
			if b := c.Blob; b != nil {
				n += len(b.Model) + len(b.NS) + len(b.Text) + 8*len(b.IDs) + len(b.MAC)
			}
		}
	}
	return n
}

// --- handler ---

func (s *server) handleResponses(w http.ResponseWriter, r *http.Request) {
	var req responseReq
	if !decodeJSON(w, r, &req) {
		return
	}
	if s.fastRoute(r, &req.Model) {
		req.applyFast()
	}
	req.Tools = dropUnrunnableTools(w, req.Tools)
	s.withModel(w, req.Model, func(lm *loadedModel) { s.serveResponsesWith(w, r, req, lm) })
}

// serveResponsesWith runs a /v1/responses generation. Reached ONLY through withModel (liveness RLock held).
func (s *server) serveResponsesWith(w http.ResponseWriter, r *http.Request, req responseReq, lm *loadedModel) {
	// Assemble the conversation: prior (previous_response_id) + instructions
	// (first turn only) + this request's input.
	var messages []chatMessage
	if req.PreviousResponseID != "" {
		prior := s.responses.Get(req.PreviousResponseID)
		if prior == nil {
			writeErr(w, http.StatusNotFound, fmt.Sprintf("previous_response_id %q not found", req.PreviousResponseID))
			return
		}
		messages = append(messages, prior.Messages...)
	} else if req.Instructions != "" {
		messages = append(messages, chatMessage{Role: "system", Content: rawStr(req.Instructions)})
	}
	inputMsgs, err := responseInputToMessages(req.Input)
	if err != nil {
		writeErr(w, http.StatusBadRequest, err.Error())
		return
	}
	messages = append(messages, inputMsgs...)

	// Reject an over-context body before tokenizing it. Placed before the tools/plain branch so one guard covers
	// both: the assembled messages include anything a stored previous_response_id dragged in, which is the input the
	// BPE would actually run over. req.Tools' schema bytes are added unconditionally too, matching that design: if
	// tools end up active below, RenderToolsSegments renders every one of them into the prompt.
	if err := lm.promptTooLargeForContext(chatInputBytes(messages) + toolSchemaBytes(req.Tools)); err != nil {
		writeErr(w, http.StatusBadRequest, err.Error())
		return
	}

	// input_image parts route to the vision path, under the chat route's history rule (image_history.go) and its
	// refusal of images together with tools.
	noteOmittedImages(w.Header(), omitChatHistoryImages(messages))
	imgs, ierr := chatImages(messages)
	if ierr != nil {
		writeErr(w, http.StatusBadRequest, ierr.Error())
		return
	}

	// Sampling: Responses uses max_output_tokens / text.format (vs the chat API's
	// max_tokens / response_format); map them onto the shared prepare path.
	sm := req.sampling
	sm.outputByWindow = true
	if req.MaxOutputTokens != nil {
		sm.MaxTokens = req.MaxOutputTokens
	}
	sm.ResponseFormat = textFormatToRespFormat(req.Text)

	id := "resp_" + reqID()
	created := time.Now().Unix()
	store := req.Store == nil || *req.Store // OpenAI default: store=true

	// Tool path: render declarations, constrain when unambiguous, buffer the full output, parse into function_call
	// items (buffered, like the chat tools path).
	//
	// toolsActive mirrors serveChatToolsWith and anthropic.go's gate: a template with no tool form must 400, not
	// silently fall through to the tools-less prompt below and drop the caller's tools.
	toolsActive := len(req.Tools) > 0 && toolChoiceMode(req.ToolChoice) != "none"
	if len(imgs) > 0 {
		if toolsActive {
			writeErr(w, http.StatusBadRequest, "tools are not supported together with image inputs; send images or tools, not both")
			return
		}
		s.serveVisionResponses(w, r, lm, req, messages, imgs, sm, id, created, store)
		return
	}
	if toolsActive {
		if lm.tmpl == nil || !lm.tmpl.SupportsTools() {
			writeErr(w, http.StatusBadRequest, "this model has no tool-calling template")
			return
		}
		s.respondTools(w, r, lm, req, messages, sm, id, created, store)
		return
	}

	ts, terr := s.resolveThink(req.thinkRequest())
	if terr != nil {
		writeErr(w, http.StatusBadRequest, terr.Error())
		return
	}
	tm := lm.templateFor(ts)
	if sm.constrainsOutput() {
		tm = lm.constrainedTemplate(ts)
	}
	system, turns, carriers := messagesToTurnsCarried(messages)
	ids, idsOut, err := lm.encodePrompt(tm, system, turns, nil, s.idsCheckFor(lm, r, carriers))
	if err != nil {
		writeServerErr(w, "encode: "+err.Error())
		return
	}
	s.noteIDs(r, idsOut)
	gr, err := lm.prepareTurn(sm, tm, system, turns, nil, ids)
	if err != nil {
		writeResponsesPrepareErr(w, err, req.Stream, id, lm.name, created)
		return
	}
	gr.id = id                                // registers this generation for cancel-by-id
	s.routeThink(lm, &gr, tm, turns, ts, nil) // the reasoning becomes a reasoning item (reasoning_items.go)
	mint := s.mintFor(lm, r, gr)
	if !lm.enter(w, r, admissionRecord{promptIDs: gr.promptIDs, clock: gr.clock}, s.haltState) {
		return
	}
	defer lm.exit()
	inTok := len(gr.promptIDs)

	if req.Stream {
		ss, ok := sseStart(w)
		if !ok {
			return
		}
		streamCtx, cancelStream := ss.CancelOnFail(r.Context())
		defer cancelStream()
		rs := newRespStream(ss, id)
		if mint != nil {
			rs.SetMintBlob(mint.blob)
		}
		rs.Emit("response.created", map[string]any{"response": responseObject(id, lm.name, created, "in_progress", []any{}, inTok, 0)})
		if gr.think != nil {
			gr.think.SetOnReasoning(rs.ReasoningDelta)
		}
		var sb strings.Builder
		// Nothing is sent between response.created and the first token, and on CPU that gap is the whole prefill: a
		// heartbeat covers it, as it does on the tools-active branch.
		stopBeat := sseHeartbeat(ss)
		defer stopBeat()
		res, gerr := lm.drive(streamCtx, gr, s.gens, s.jobs, func(t string) {
			sb.WriteString(t)
			rs.Delta(t)
		})
		stopBeat()
		if gerr != nil {
			sseEvent(ss, "error", map[string]any{"type": "error", "message": "generation failed: " + gerr.Error()})
			sseDone(ss)
			return
		}
		out := withCarrier(rs, mint, req.Include, id, withReasoning(rs, gr.think, id, mint, []any{rs.CloseMessage(sb.String())}))
		rs.Emit("response.completed", map[string]any{"response": withCachedTokens(markEmptyResponse(responseObjectTurn(gr.clock, id, lm.name, created, respStatus(res.finish), out, inTok, res.nComp), emptyAnswerOf(gr, res.finish, res.stopHit, 0, res.nComp)), res.reused.Tokens)})
		if res.cancelReason != "" {
			rs.Emit("response.cancelled", map[string]any{"reason": res.cancelReason})
		}
		sseDone(ss)
		s.maybeStore(store, id, lm.name, messages, sb.String(), gr.think.ReasoningText(), nil)
		return
	}

	var sb strings.Builder
	res, gerr := lm.drive(r.Context(), gr, s.gens, s.jobs, func(t string) { sb.WriteString(t) })
	if gerr != nil {
		writeServerErr(w, "generation failed: "+gerr.Error())
		return
	}
	if res.cancelReason != "" {
		writeErr(w, statusCancelled, "generation cancelled: "+res.cancelReason)
		return
	}
	out := withCarrier(nil, mint, req.Include, id, withReasoning(nil, gr.think, id, mint, []any{outputMessage(id+"-msg", sb.String())}))
	writeJSON(w, http.StatusOK, withCachedTokens(markEmptyResponse(responseObjectTurn(gr.clock, id, lm.name, created, respStatus(res.finish), out, inTok, res.nComp), emptyAnswerOf(gr, res.finish, res.stopHit, 0, res.nComp)), res.reused.Tokens))
	s.maybeStore(store, id, lm.name, messages, sb.String(), gr.think.ReasoningText(), nil)
}

// serveVisionResponses is serveResponsesWith's plain-text branch for input that carries images: the prompt from
// visionPromptN, generation through driveVL, rendered as a Responses object (or its event stream).
func (s *server) serveVisionResponses(w http.ResponseWriter, r *http.Request, lm *loadedModel, req responseReq, messages []chatMessage, imgs []imageRef, sm sampling, id string, created int64, store bool) {
	var hasAudio, hasImage bool
	for _, im := range imgs {
		if im.Audio {
			hasAudio = true
		} else {
			hasImage = true
		}
	}
	if hasAudio && !lm.audioCapable() {
		writeErr(w, http.StatusBadRequest, "this model has no audio tower (audio input needs a Gemma 4 checkpoint with an audio_config, e.g. E2B or E4B)")
		return
	}
	if hasImage && !lm.visionCapable() {
		writeErr(w, http.StatusBadRequest, "this model has no vision tower (start with --vision <dir> to enable image input)")
		return
	}
	if len(imgs) > maxImagesPerTurn {
		writeErr(w, http.StatusBadRequest, tooManyImages(len(imgs)))
		return
	}
	ts, terr := s.resolveThink(req.thinkRequest())
	if terr != nil {
		writeErr(w, http.StatusBadRequest, terr.Error())
		return
	}
	tm := lm.templateFor(ts)
	if sm.constrainsOutput() {
		tm = lm.constrainedTemplate(ts)
	}
	system, turns := messagesToTurns(messages)
	msgText, ordered := chatMediaText(messages)
	vi, err := lm.visionPromptN(tm, system, turns, imgs, ordered, msgText)
	if err != nil {
		writeErr(w, http.StatusBadRequest, err.Error())
		return
	}
	gr, err := lm.prepare(sm, vi.ids, false)
	if err != nil {
		writeResponsesPrepareErr(w, err, req.Stream, id, lm.name, created)
		return
	}
	gr.id = id
	s.routeThink(lm, &gr, tm, turns, ts, nil)
	if !lm.enter(w, r, admissionRecord{promptIDs: gr.promptIDs, clock: gr.clock}, s.haltState) {
		return
	}
	defer lm.exit()
	inTok := len(gr.promptIDs)
	if req.Stream {
		ss, ok := sseStart(w)
		if !ok {
			return
		}
		streamCtx, cancelStream := ss.CancelOnFail(r.Context())
		defer cancelStream()
		rs := newRespStream(ss, id)
		rs.Emit("response.created", map[string]any{"response": responseObject(id, lm.name, created, "in_progress", []any{}, inTok, 0)})
		if gr.think != nil {
			gr.think.SetOnReasoning(rs.ReasoningDelta)
		}
		var sb strings.Builder
		stopBeat := sseHeartbeat(ss) // the image turn's tower and prefill are the silent window
		defer stopBeat()
		res, gerr := lm.driveVL(streamCtx, gr, vi, s.gens, s.jobs, func(t string) {
			sb.WriteString(t)
			rs.Delta(t)
		})
		stopBeat()
		if gerr != nil {
			sseEvent(ss, "error", map[string]any{"type": "error", "message": "generation failed: " + gerr.Error()})
			sseDone(ss)
			return
		}
		out := withReasoning(rs, gr.think, id, nil, []any{rs.CloseMessage(sb.String())})
		rs.Emit("response.completed", map[string]any{"response": withCachedTokens(markEmptyResponse(responseObjectTurn(gr.clock, id, lm.name, created, respStatus(res.finish), out, inTok, res.nComp), emptyAnswerOf(gr, res.finish, res.stopHit, 0, res.nComp)), res.reused.Tokens)})
		if res.cancelReason != "" {
			rs.Emit("response.cancelled", map[string]any{"reason": res.cancelReason})
		}
		sseDone(ss)
		s.maybeStore(store, id, lm.name, messages, sb.String(), gr.think.ReasoningText(), nil)
		return
	}
	var sb strings.Builder
	res, gerr := lm.driveVL(r.Context(), gr, vi, s.gens, s.jobs, func(t string) { sb.WriteString(t) })
	if gerr != nil {
		writeServerErr(w, "generation failed: "+gerr.Error())
		return
	}
	if res.cancelReason != "" {
		writeErr(w, statusCancelled, "generation cancelled: "+res.cancelReason)
		return
	}
	out := withReasoning(nil, gr.think, id, nil, []any{outputMessage(id+"-msg", sb.String())})
	writeJSON(w, http.StatusOK, withCachedTokens(markEmptyResponse(responseObjectTurn(gr.clock, id, lm.name, created, respStatus(res.finish), out, inTok, res.nComp), emptyAnswerOf(gr, res.finish, res.stopHit, 0, res.nComp)), res.reused.Tokens))
	s.maybeStore(store, id, lm.name, messages, sb.String(), gr.think.ReasoningText(), nil)
}

// respondTools generates a (buffered) tool-calling response and emits
// function_call output items (or a message when the model answered normally).
func (s *server) respondTools(w http.ResponseWriter, r *http.Request, lm *loadedModel, req responseReq, messages []chatMessage, sm sampling, id string, created int64, store bool) {
	if err := checkToolCount(len(req.Tools)); err != nil {
		writeErr(w, http.StatusBadRequest, err.Error())
		return
	}
	tools := make([]chat.Tool, len(req.Tools))
	for i, t := range req.Tools {
		tools[i] = chat.Tool{Name: t.Function.Name, Description: t.Function.Description, Parameters: t.Function.Parameters}
	}
	system, turns, carriers := messagesToTurnsCarried(messages)
	ts, terr := s.resolveThink(req.thinkRequest())
	if terr != nil {
		writeErr(w, http.StatusBadRequest, terr.Error())
		return
	}
	tm := lm.templateFor(ts)
	if toolsConstrainedFromStart(forcedTool(req.ToolChoice, tools, endsWithToolResult(turns)), toolChoiceMode(req.ToolChoice) == "function", openAIUnionMode(req.ToolChoice), tools) {
		tm = lm.constrainedTemplate(ts)
	}
	ids, idsOut, err := lm.encodePrompt(tm, system, turns, tools, s.idsCheckFor(lm, r, carriers))
	if err != nil {
		writeServerErr(w, "encode: "+err.Error())
		return
	}
	s.noteIDs(r, idsOut)
	gr, err := lm.prepareTurn(sm, tm, system, turns, tools, ids)
	if err != nil {
		writeResponsesPrepareErr(w, err, req.Stream, id, lm.name, created)
		return
	}
	gr.id = id // registers this generation for cancel-by-id
	forced := forcedTool(req.ToolChoice, tools, endsWithToolResult(turns))
	namedForce := toolChoiceMode(req.ToolChoice) == "function"
	if cerr := constrainForcedTool(lm, &gr, forced, namedForce, openAIUnionMode(req.ToolChoice), tools); cerr != nil {
		writeErr(w, http.StatusBadRequest, cerr.Error()) // named tool_choice unconstrainable → 400
		return
	}
	s.routeThink(lm, &gr, tm, turns, ts, nil) // AFTER the tool constraint, which the budget composes with; the reasoning becomes a reasoning item
	mint := s.mintFor(lm, r, gr)
	if !lm.enter(w, r, admissionRecord{promptIDs: gr.promptIDs, clock: gr.clock}, s.haltState) {
		return
	}
	defer lm.exit()
	inTok := len(gr.promptIDs)
	// Same buffered-and-therefore-silent shape as the chat tools path: start SSE first and keep the stream alive with
	// comment frames while the buffer fills, so a slow generation is not indistinguishable from a dead server. See
	// tools.go for the full rationale and the 500-vs-sseErr consequence.
	var ss *sseWriter
	if req.Stream {
		var ok bool
		if ss, ok = sseStart(w); !ok {
			return
		}
	}
	// The shared tool turn (tool_turn.go). Streaming: response.created goes out first, as on the plain-text path, and
	// prose streams as output_text.delta while the model writes it, where the family allows.
	var onProse func(string)
	var rs *respStream
	if ss != nil {
		rs = newRespStream(ss, id)
		if mint != nil {
			rs.SetMintBlob(mint.blob)
		}
		rs.Emit("response.created", map[string]any{"response": responseObject(id, lm.name, created, "in_progress", []any{}, inTok, 0)})
		onProse = rs.Delta
		if gr.think != nil {
			gr.think.SetOnReasoning(rs.ReasoningDelta)
		}
	}
	t, gerr := s.runToolTurn(r.Context(), lm, gr, tools, ss, onProse)
	if gerr != nil {
		if ss != nil {
			msg := gerr.Error()
			if !errors.Is(gerr, errProseDiverged) {
				msg = "generation failed: " + msg
			}
			sseErr(ss, msg)
			sseDone(ss)
			return
		}
		writeServerErr(w, "generation failed: "+gerr.Error())
		return
	}
	finish, nComp := t.finish, t.nComp
	if t.cancelReason != "" {
		// Cancelled mid-generation: a partial buffer, so no tool call is parsed out of it. Report it as a plain
		// (partial) text message, as the "nothing parseable" fallback below does, so a client still gets a
		// well-formed response object with real usage.
		if ss != nil {
			resp := responseObjectTurn(gr.clock, id, lm.name, created, respStatus(finish), withReasoning(rs, gr.think, id, mint, []any{rs.CloseMessage(t.raw)}), inTok, nComp)
			rs.Emit("response.completed", map[string]any{"response": resp})
			rs.Emit("response.cancelled", map[string]any{"reason": t.cancelReason})
			sseDone(ss)
			return
		}
		writeErr(w, statusCancelled, "generation cancelled: "+t.cancelReason)
		return
	}
	calls, lead := t.calls, ""
	if len(calls) > 0 {
		lead = t.lead
	}
	if rs != nil && wantsCarrier(req.Include) {
		// a reply with no reasoning and nothing streamed yet: its carrier-only item goes first, before the items below open
		if enc := mint.blob(""); enc != "" {
			rs.CarrierOnly(enc)
		}
	}
	if ss != nil && t.rest != "" {
		// The prose still held back when the call opened (or all of it, on a family that cannot
		// stream prose safely): delivered as a delta too, so the deltas always add up to the message.
		onProse(t.rest)
	}

	var out []any
	var toolCalls []apiToolCall // stored for previous_response_id continuity below
	if lead != "" {
		if rs != nil {
			out = append(out, rs.CloseMessage(lead))
		} else {
			out = append(out, outputMessage(id+"-msg", lead))
		}
	} else if rs.MessageOpen() { // prose streamed that the parser then trimmed to nothing: close the item it opened
		out = append(out, rs.CloseMessage(""))
	}
	for i, c := range calls {
		callID := c.ID
		if callID == "" { // model didn't emit an id; synthesize one so function_call_output can correlate (as toAPICalls/toolUseBlock do)
			callID = "call_" + reqID()
		}
		fc := map[string]any{
			"type": "function_call", "id": fmt.Sprintf("%s-fc%d", id, i),
			"call_id": callID, "name": c.Name, "arguments": string(c.Arguments), "status": "completed",
		}
		out = append(out, fc)
		if rs != nil {
			rs.FunctionCall(fc)
		}
		tc := apiToolCall{ID: callID, Type: "function"}
		tc.Function.Name = c.Name
		tc.Function.Arguments = string(c.Arguments)
		toolCalls = append(toolCalls, tc)
	}
	if len(out) == 0 { // model produced nothing parseable → empty message
		if rs != nil {
			out = append(out, rs.CloseMessage(t.raw))
		} else {
			out = append(out, outputMessage(id+"-msg", t.raw))
		}
	}
	out = withReasoning(rs, gr.think, id, mint, out)
	if rs == nil {
		out = withCarrier(nil, mint, req.Include, id, out)
	}
	// Reflect the real finish, like the non-tools paths: a tool turn cut off by max_output_tokens is "incomplete",
	// not "completed".
	resp := withCachedTokens(markEmptyResponse(responseObjectTurn(gr.clock, id, lm.name, created, respStatus(finish), out, inTok, nComp), emptyAnswerOf(gr, finish, t.stopHit, len(calls), nComp)), t.reused.Tokens)
	if t.fromReasoning && len(calls) > 0 {
		resp[fromReasoningField] = true
	}
	if req.Stream {
		rs.Emit("response.completed", map[string]any{"response": resp})
		sseDone(ss)
	} else {
		writeJSON(w, http.StatusOK, resp)
	}
	// Tool-call continuations round-trip via the next request's input; store the
	// assistant text (the lead, if any) AND the tool calls for previous_response_id continuity.
	s.maybeStore(store, id, lm.name, messages, lead, gr.think.ReasoningText(), toolCalls)
}

// maybeStore records this turn's assistant output for a later previous_response_id continuation. serveResponsesWith
// appends prior.messages verbatim onto the next request's input, so whatever is missing here is missing from every
// stateful continuation.
//
// toolCalls are stored along with the lead text (often empty, when the model went straight into a tool call). The SDK
// default is previous_response_id: the client sends only the new function_call_output, and the server is expected to
// have kept the matching function_call from its own prior turn. Without ToolCalls here, that turn reconstructs as
// user, assistant(""), tool(result): a tool result answering a call that, as far as the stored conversation shows,
// was never made. (A stateless caller that resends the whole conversation is handled by responseInputToMessages.)
func (s *server) maybeStore(store bool, id, model string, messages []chatMessage, assistant, reasoning string, toolCalls []apiToolCall) {
	if !store || s.responses == nil {
		return
	}
	full := append(slices.Clone(messages),
		chatMessage{Role: "assistant", Content: rawStr(assistant), ToolCalls: toolCalls, ReasoningContent: reasoning})
	s.responses.Put(id, model, full)
}

// --- shapes + helpers ---

// responseInputToMessages parses the Responses `input` (a string, or an array of
// message items whose content is a string or an array of {type,text} parts).
func responseInputToMessages(raw json.RawMessage) ([]chatMessage, error) {
	if len(raw) == 0 {
		return nil, fmt.Errorf("input is required")
	}
	var str string
	if json.Unmarshal(raw, &str) == nil {
		return []chatMessage{{Role: "user", Content: rawStr(str)}}, nil
	}
	// `type` and the function-call fields are decoded, not just {role, content}. A Responses tool loop feeds the
	// model's own `function_call` back with a `function_call_output`, and neither carries a role or a content field,
	// so both would fall through to the default below as `{Role:"user", Content:""}`: two empty user turns. The model
	// would never see the tool result and would answer without it or call the same tool again, forever, under HTTP
	// 200.
	var items []struct {
		Type      string          `json:"type"`
		Role      string          `json:"role"`
		Content   json.RawMessage `json:"content"`
		CallID    string          `json:"call_id"`
		Name      string          `json:"name"`
		Arguments string          `json:"arguments"`
		Output    json.RawMessage `json:"output"`
		// a reasoning item's text (reasoningItemText)
		EncryptedContent string          `json:"encrypted_content"`
		Summary          json.RawMessage `json:"summary"`
	}
	if err := json.Unmarshal(raw, &items); err != nil {
		return nil, fmt.Errorf("input must be a string or an array of message items")
	}
	msgs := make([]chatMessage, 0, len(items))
	// pending is a reasoning item's text waiting for the assistant turn it belongs to: the next function_call or assistant
	// message. A reasoning item with no assistant turn after it is dropped.
	pending := ""
	// carried is the carried-id blob of the latest reasoning item, for the same assistant turn as pending (carried_ids.go).
	var carried *idsCarrier
	for _, it := range items {
		switch it.Type {
		case "reasoning":
			text, c, err := reasoningItemCarried(it.EncryptedContent, it.Content, it.Summary)
			if err != nil {
				return nil, err
			}
			pending += text
			if c != nil {
				carried = c
			}
			continue
		case "function_call":
			// The model's own call, replayed. Becomes the assistant turn that carries it, which is
			// what the chat template renders as a tool call.
			tc := apiToolCall{ID: it.CallID, Type: "function"}
			tc.Function.Name = it.Name
			tc.Function.Arguments = it.Arguments
			msgs = append(msgs, chatMessage{Role: "assistant", ToolCalls: []apiToolCall{tc}, ReasoningContent: pending, carried: carried})
			pending, carried = "", nil
			continue
		case "function_call_output":
			// The caller's result. `output` is a string in every SDK that sends it, but is typed
			// loosely enough to arrive as an object — contentText handles the string case and
			// falls back to the raw JSON, which is better in the prompt than an empty turn.
			msgs = append(msgs, chatMessage{
				Role:       "tool",
				ToolCallID: it.CallID,
				Name:       it.Name,
				Content:    rawStr(toolOutputText(it.Output)),
			})
			continue
		}
		role := it.Role
		if role == "" {
			role = "user"
		}
		content, err := responsesContent(it.Content)
		if err != nil {
			return nil, err
		}
		m := chatMessage{Role: role, Content: content}
		if role == "assistant" {
			m.ReasoningContent, pending = pending, ""
			m.carried, carried = carried, nil
		}
		msgs = append(msgs, m)
	}
	return msgs, nil
}

// responsesContent is a Responses message content as a chat message's: plain text when it carries no image; otherwise
// a chat content array, each input_image an image_url part in its place among the text parts, so the vision path sees
// the order the caller sent. An image by file_id is refused (this server stores no files).
func responsesContent(raw json.RawMessage) (json.RawMessage, error) {
	var parts []struct {
		Type     string          `json:"type"`
		Text     string          `json:"text"`
		ImageURL json.RawMessage `json:"image_url"`
		FileID   string          `json:"file_id"`
	}
	if json.Unmarshal(raw, &parts) != nil {
		return rawStr(contentText(raw)), nil
	}
	hasImage := false
	for _, p := range parts {
		hasImage = hasImage || p.Type == "input_image"
	}
	if !hasImage {
		return rawStr(contentText(raw)), nil
	}
	type imageURL struct {
		URL string `json:"url"`
	}
	type chatPart struct {
		Type     string    `json:"type"`
		Text     string    `json:"text,omitempty"`
		ImageURL *imageURL `json:"image_url,omitempty"`
	}
	out := make([]chatPart, 0, len(parts))
	for _, p := range parts {
		if p.Type != "input_image" {
			out = append(out, chatPart{Type: "text", Text: p.Text})
			continue
		}
		if p.FileID != "" {
			return nil, fmt.Errorf("input_image by file_id is not supported (this server stores no files); send image_url as a base64 data: URI")
		}
		var url string
		if json.Unmarshal(p.ImageURL, &url) != nil { // the Responses shape is a string; take the chat API's {url} too
			var u imageURL
			_ = json.Unmarshal(p.ImageURL, &u)
			url = u.URL
		}
		if url == "" {
			return nil, fmt.Errorf("input_image needs image_url (a base64 data: URI)")
		}
		out = append(out, chatPart{Type: "image_url", ImageURL: &imageURL{URL: url}})
	}
	b, err := json.Marshal(out)
	return b, err
}

// toolOutputText flattens a `function_call_output`'s `output`: a plain string when it is one, the content-part text
// when it is an array, and the raw JSON otherwise. Never empty for a non-empty input: an empty tool turn means the
// model never sees the result, so the fallback keeps something it can read rather than silently dropping it.
func toolOutputText(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var str string
	if json.Unmarshal(raw, &str) == nil {
		return str
	}
	if t := contentText(raw); t != "" {
		return t
	}
	return string(raw)
}

// contentText flattens a message content (string or []{type,text}) to plain text.
func contentText(raw json.RawMessage) string {
	var str string
	if json.Unmarshal(raw, &str) == nil {
		return str
	}
	var parts []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if json.Unmarshal(raw, &parts) == nil {
		var b strings.Builder
		for _, p := range parts {
			b.WriteString(p.Text)
		}
		return b.String()
	}
	return ""
}

// textFormatToRespFormat maps Responses `text.format` onto the chat API's
// response_format so the shared prepare/grammarFor path constrains identically.
func textFormatToRespFormat(t *respText) *respFormat {
	if t == nil || t.Format == nil {
		return nil
	}
	rf := &respFormat{Type: t.Format.Type}
	if t.Format.Type == "json_schema" {
		rf.JSONSchema = &struct {
			Name   string          `json:"name"`
			Schema json.RawMessage `json:"schema"`
		}{Name: t.Format.Name, Schema: t.Format.Schema}
	}
	return rf
}

func responseObject(id, model string, created int64, status string, output []any, inTok, outTok int) map[string]any {
	o := map[string]any{
		"id": id, "object": "response", "created_at": created, "status": status, "model": model,
		"output": output,
		"usage":  map[string]any{"input_tokens": inTok, "output_tokens": outTok, "total_tokens": inTok + outTok},
	}
	// A generation cut off by max_output_tokens is "incomplete", not "completed".
	if status == "incomplete" {
		o["incomplete_details"] = map[string]any{"reason": "max_output_tokens"}
	}
	return o
}

// responseObjectTurn is a finished turn's responseObject, with the turn's timings and, when asked, its cost beside the
// usage (clock, nil for none).
func responseObjectTurn(clock *jobs.TurnClock, id, model string, created int64, status string, output []any, inTok, outTok int) map[string]any {
	return clock.AddTurn(responseObject(id, model, created, status, output, inTok, outTok))
}

// withCachedTokens reports the prompt tokens whose prefill a prior turn's KV made unnecessary, in OpenAI's own field for
// them (usage.input_tokens_details.cached_tokens; Codex reads it as cached_input_tokens).
func withCachedTokens(resp map[string]any, reused int) map[string]any {
	resp["usage"].(map[string]any)["input_tokens_details"] = map[string]any{"cached_tokens": reused}
	return resp
}

// respStatus maps drive's finish reason to a Responses status: "length" (truncated by max_output_tokens) is
// "incomplete"; everything else is "completed".
func respStatus(finish string) string {
	switch finish {
	case "cancelled": // a real value in OpenAI's own Responses status enum
		return "cancelled"
	case "length":
		return "incomplete"
	default:
		return "completed"
	}
}

var sseEvent = streamframe.SendEvent
