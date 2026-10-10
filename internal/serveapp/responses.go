package serveapp

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/townsendmerino/goinfer/chat"
)

// OpenAI Responses API (/v1/responses). Stateless input (input, instructions, text.format, tools, streaming) plus
// store/previous_response_id through an in-memory ring, so a continued response is a prompt-prefix extension that
// rides the per-model sessionLRU for warm KV. Out of scope: hosted tools, reasoning items, file inputs.

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
	// Reasoning.effort is OpenAI's thinking control ("none" turns thinking off, anything else on); the reasoning text
	// itself is not returned on this route (reasoning items are out of scope) — output_text is always the clean answer.
	Reasoning *struct {
		Effort string `json:"effort"`
	} `json:"reasoning"`
	sampling
}

func (r responseReq) thinkRequest() thinkRequest {
	tr := thinkRequest{}
	if r.Reasoning != nil {
		tr.reasoningEffort = r.Reasoning.Effort
	}
	return tr
}

// --- in-memory response store (id → conversation) ---

type responseEntry struct {
	model    string
	messages []chatMessage // full conversation incl. the assistant output
}

type responseStore struct {
	mu    sync.Mutex
	m     map[string]*responseEntry
	order []string
	cap   int
}

func newResponseStore(cap int) *responseStore {
	return &responseStore{m: map[string]*responseEntry{}, cap: cap}
}

func (s *responseStore) get(id string) *responseEntry {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.m[id]
}

func (s *responseStore) put(id string, e *responseEntry) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.m[id]; !ok {
		s.order = append(s.order, id)
		for len(s.order) > s.cap { // FIFO eviction
			delete(s.m, s.order[0])
			s.order = s.order[1:]
		}
	}
	s.m[id] = e
}

// --- handler ---

func (s *server) handleResponses(w http.ResponseWriter, r *http.Request) {
	var req responseReq
	if !decodeJSON(w, r, &req) {
		return
	}
	s.withModel(w, req.Model, func(lm *loadedModel) { s.serveResponsesWith(w, r, req, lm) })
}

// serveResponsesWith runs a /v1/responses generation. Reached ONLY through withModel (liveness RLock held).
func (s *server) serveResponsesWith(w http.ResponseWriter, r *http.Request, req responseReq, lm *loadedModel) {
	// Assemble the conversation: prior (previous_response_id) + instructions
	// (first turn only) + this request's input.
	var messages []chatMessage
	if req.PreviousResponseID != "" {
		prior := s.responses.get(req.PreviousResponseID)
		if prior == nil {
			writeErr(w, http.StatusNotFound, fmt.Sprintf("previous_response_id %q not found", req.PreviousResponseID))
			return
		}
		messages = append(messages, prior.messages...)
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

	// S11: input_image parts route to the vision path, under the chat route's history rule (image_history.go) and its
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
	system, turns := messagesToTurns(messages)
	ids, err := lm.promptForT(tm, system, turns)
	if err != nil {
		writeServerErr(w, "encode: "+err.Error())
		return
	}
	gr, err := lm.prepare(sm, ids, lm.residentPath())
	if err != nil {
		writeErr(w, prepareErrStatus(err), err.Error())
		return
	}
	gr.id = id                                // registers this generation for cancel-by-id
	s.routeThink(lm, &gr, tm, turns, ts, nil) // reasoning is dropped on this route
	if !lm.enter(w, r, admissionRecord{promptIDs: gr.promptIDs}, s.haltState) {
		return
	}
	defer lm.exit()
	inTok := len(gr.promptIDs)

	if req.Stream {
		ss, ok := sseStart(w)
		if !ok {
			return
		}
		sseEvent(ss, "response.created", map[string]any{
			"type": "response.created", "response": responseObject(id, lm.name, created, "in_progress", []any{}, inTok, 0),
		})
		var sb strings.Builder
		// Nothing is sent between response.created and the first token, and on CPU that gap is the whole prefill: a
		// heartbeat covers it, as it does on the tools-active branch.
		stopBeat := sseHeartbeat(ss)
		finish, nComp, _, _, _, cancelReason, gerr := lm.drive(r.Context(), gr, s.gens, s.jobs, func(t string) {
			sb.WriteString(t)
			sseEvent(ss, "response.output_text.delta", map[string]any{
				"type": "response.output_text.delta", "item_id": id + "-msg", "output_index": 0, "content_index": 0, "delta": t,
			})
		})
		stopBeat()
		if gerr != nil {
			sseEvent(ss, "error", map[string]any{"type": "error", "message": "generation failed: " + gerr.Error()})
			sseDone(ss)
			return
		}
		out := []any{outputMessage(id+"-msg", sb.String())}
		sseEvent(ss, "response.completed", map[string]any{
			"type": "response.completed", "response": responseObject(id, lm.name, created, respStatus(finish), out, inTok, nComp),
		})
		if cancelReason != "" {
			sseEvent(ss, "response.cancelled", map[string]any{"type": "response.cancelled", "reason": cancelReason})
		}
		sseDone(ss)
		s.maybeStore(store, id, lm.name, messages, sb.String(), nil)
		return
	}

	var sb strings.Builder
	finish, nComp, _, _, _, cancelReason, gerr := lm.drive(r.Context(), gr, s.gens, s.jobs, func(t string) { sb.WriteString(t) })
	if gerr != nil {
		writeServerErr(w, "generation failed: "+gerr.Error())
		return
	}
	if cancelReason != "" {
		writeErr(w, statusCancelled, "generation cancelled: "+cancelReason)
		return
	}
	out := []any{outputMessage(id+"-msg", sb.String())}
	writeJSON(w, http.StatusOK, responseObject(id, lm.name, created, respStatus(finish), out, inTok, nComp))
	s.maybeStore(store, id, lm.name, messages, sb.String(), nil)
}

// serveVisionResponses is serveResponsesWith's plain-text branch for input that carries images: the prompt from
// visionPromptN, generation through driveVL, rendered as a Responses object (or its event stream).
func (s *server) serveVisionResponses(w http.ResponseWriter, r *http.Request, lm *loadedModel, req responseReq, messages []chatMessage, imgs []imageRef, sm sampling, id string, created int64, store bool) {
	if imgs[0].audio {
		if !lm.audioCapable() {
			writeErr(w, http.StatusBadRequest, "this model has no audio tower (audio input needs a Gemma 4 checkpoint with an audio_config, e.g. E2B or E4B)")
			return
		}
	} else if !lm.visionCapable() {
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
		writeErr(w, prepareErrStatus(err), err.Error())
		return
	}
	gr.id = id
	if !lm.enter(w, r, admissionRecord{promptIDs: gr.promptIDs}, s.haltState) {
		return
	}
	defer lm.exit()
	inTok := len(gr.promptIDs)
	if req.Stream {
		ss, ok := sseStart(w)
		if !ok {
			return
		}
		sseEvent(ss, "response.created", map[string]any{
			"type": "response.created", "response": responseObject(id, lm.name, created, "in_progress", []any{}, inTok, 0),
		})
		var sb strings.Builder
		stopBeat := sseHeartbeat(ss) // the image turn's tower and prefill are the silent window
		finish, nComp, _, _, _, cancelReason, gerr := lm.driveVL(r.Context(), gr, vi, s.gens, s.jobs, func(t string) {
			sb.WriteString(t)
			sseEvent(ss, "response.output_text.delta", map[string]any{
				"type": "response.output_text.delta", "item_id": id + "-msg", "output_index": 0, "content_index": 0, "delta": t,
			})
		})
		stopBeat()
		if gerr != nil {
			sseEvent(ss, "error", map[string]any{"type": "error", "message": "generation failed: " + gerr.Error()})
			sseDone(ss)
			return
		}
		out := []any{outputMessage(id+"-msg", sb.String())}
		sseEvent(ss, "response.completed", map[string]any{
			"type": "response.completed", "response": responseObject(id, lm.name, created, respStatus(finish), out, inTok, nComp),
		})
		if cancelReason != "" {
			sseEvent(ss, "response.cancelled", map[string]any{"type": "response.cancelled", "reason": cancelReason})
		}
		sseDone(ss)
		s.maybeStore(store, id, lm.name, messages, sb.String(), nil)
		return
	}
	var sb strings.Builder
	finish, nComp, _, _, _, cancelReason, gerr := lm.driveVL(r.Context(), gr, vi, s.gens, s.jobs, func(t string) { sb.WriteString(t) })
	if gerr != nil {
		writeServerErr(w, "generation failed: "+gerr.Error())
		return
	}
	if cancelReason != "" {
		writeErr(w, statusCancelled, "generation cancelled: "+cancelReason)
		return
	}
	out := []any{outputMessage(id+"-msg", sb.String())}
	writeJSON(w, http.StatusOK, responseObject(id, lm.name, created, respStatus(finish), out, inTok, nComp))
	s.maybeStore(store, id, lm.name, messages, sb.String(), nil)
}

// respondTools generates a (buffered) tool-calling response and emits
// function_call output items (or a message when the model answered normally).
func (s *server) respondTools(w http.ResponseWriter, r *http.Request, lm *loadedModel, req responseReq, messages []chatMessage, sm sampling, id string, created int64, store bool) {
	tools := make([]chat.Tool, len(req.Tools))
	for i, t := range req.Tools {
		tools[i] = chat.Tool{Name: t.Function.Name, Description: t.Function.Description, Parameters: t.Function.Parameters}
	}
	system, turns := messagesToTurns(messages)
	ts, terr := s.resolveThink(req.thinkRequest())
	if terr != nil {
		writeErr(w, http.StatusBadRequest, terr.Error())
		return
	}
	tm := lm.templateFor(ts)
	if toolsConstrainedFromStart(forcedTool(req.ToolChoice, tools, endsWithToolResult(turns)), toolChoiceMode(req.ToolChoice) == "function", openAIUnionMode(req.ToolChoice), tools) {
		tm = lm.constrainedTemplate(ts)
	}
	ids, err := lm.tk.EncodeSegments(tm.RenderToolsSegments(system, turns, tools), false)
	if err != nil {
		writeServerErr(w, "encode: "+err.Error())
		return
	}
	gr, err := lm.prepare(sm, ids, lm.residentPath())
	if err != nil {
		writeErr(w, prepareErrStatus(err), err.Error())
		return
	}
	gr.id = id // registers this generation for cancel-by-id
	forced := forcedTool(req.ToolChoice, tools, endsWithToolResult(turns))
	namedForce := toolChoiceMode(req.ToolChoice) == "function"
	if cerr := constrainForcedTool(lm, &gr, forced, namedForce, openAIUnionMode(req.ToolChoice), tools); cerr != nil {
		writeErr(w, http.StatusBadRequest, cerr.Error()) // named tool_choice unconstrainable → 400
		return
	}
	s.routeThink(lm, &gr, tm, turns, ts, nil) // reasoning is dropped on this route; AFTER the tool constraint, which the budget composes with
	if !lm.enter(w, r, admissionRecord{promptIDs: gr.promptIDs}, s.haltState) {
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
	if ss != nil {
		sseEvent(ss, "response.created", map[string]any{"type": "response.created", "response": responseObject(id, lm.name, created, "in_progress", []any{}, inTok, 0)})
		onProse = func(out string) {
			sseEvent(ss, "response.output_text.delta", map[string]any{
				"type": "response.output_text.delta", "item_id": id + "-msg", "output_index": 0, "content_index": 0, "delta": out,
			})
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
		resp := responseObject(id, lm.name, created, respStatus(finish), []any{outputMessage(id+"-msg", t.raw)}, inTok, nComp)
		if ss != nil {
			sseEvent(ss, "response.completed", map[string]any{"type": "response.completed", "response": resp})
			sseEvent(ss, "response.cancelled", map[string]any{"type": "response.cancelled", "reason": t.cancelReason})
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
	if ss != nil && t.rest != "" {
		// The prose still held back when the call opened (or all of it, on a family that cannot
		// stream prose safely): delivered as a delta too, so the deltas always add up to the message.
		onProse(t.rest)
	}

	var out []any
	var toolCalls []apiToolCall // stored for previous_response_id continuity below
	if lead != "" {
		out = append(out, outputMessage(id+"-msg", lead))
	}
	for i, c := range calls {
		callID := c.ID
		if callID == "" { // model didn't emit an id; synthesize one so function_call_output can correlate (as toAPICalls/toolUseBlock do)
			callID = "call_" + reqID()
		}
		out = append(out, map[string]any{
			"type": "function_call", "id": fmt.Sprintf("%s-fc%d", id, i),
			"call_id": callID, "name": c.Name, "arguments": string(c.Arguments), "status": "completed",
		})
		tc := apiToolCall{ID: callID, Type: "function"}
		tc.Function.Name = c.Name
		tc.Function.Arguments = string(c.Arguments)
		toolCalls = append(toolCalls, tc)
	}
	if len(out) == 0 { // model produced nothing parseable → empty message
		out = append(out, outputMessage(id+"-msg", t.raw))
	}
	// Reflect the real finish, like the non-tools paths: a tool turn cut off by max_output_tokens is "incomplete",
	// not "completed".
	resp := responseObject(id, lm.name, created, respStatus(finish), out, inTok, nComp)
	if req.Stream {
		sseEvent(ss, "response.completed", map[string]any{"type": "response.completed", "response": resp})
		sseDone(ss)
	} else {
		writeJSON(w, http.StatusOK, resp)
	}
	// Tool-call continuations round-trip via the next request's input; store the
	// assistant text (the lead, if any) AND the tool calls for previous_response_id continuity.
	s.maybeStore(store, id, lm.name, messages, lead, toolCalls)
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
func (s *server) maybeStore(store bool, id, model string, messages []chatMessage, assistant string, toolCalls []apiToolCall) {
	if !store || s.responses == nil {
		return
	}
	full := append(append([]chatMessage(nil), messages...),
		chatMessage{Role: "assistant", Content: rawStr(assistant), ToolCalls: toolCalls})
	s.responses.put(id, &responseEntry{model: model, messages: full})
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
	}
	if err := json.Unmarshal(raw, &items); err != nil {
		return nil, fmt.Errorf("input must be a string or an array of message items")
	}
	msgs := make([]chatMessage, 0, len(items))
	for _, it := range items {
		switch it.Type {
		case "function_call":
			// The model's own call, replayed. Becomes the assistant turn that carries it, which is
			// what the chat template renders as a tool call.
			tc := apiToolCall{ID: it.CallID, Type: "function"}
			tc.Function.Name = it.Name
			tc.Function.Arguments = it.Arguments
			msgs = append(msgs, chatMessage{Role: "assistant", ToolCalls: []apiToolCall{tc}})
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
		msgs = append(msgs, chatMessage{Role: role, Content: content})
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

// respStatus maps drive's finish reason to a Responses status: "length" (truncated by max_output_tokens) is
// "incomplete"; everything else is "completed".
func respStatus(finish string) string {
	switch finish {
	case "cancelled": // docs/tasks/task-halt-2026-09.md: a real value in OpenAI's own Responses status enum
		return "cancelled"
	case "length":
		return "incomplete"
	default:
		return "completed"
	}
}

func outputMessage(itemID, text string) map[string]any {
	return map[string]any{
		"type": "message", "id": itemID, "role": "assistant", "status": "completed",
		"content": []any{map[string]any{"type": "output_text", "text": text, "annotations": []any{}}},
	}
}

// sseEvent writes a Responses SSE event (both the event: line and the data: JSON,
// which itself carries the "type"). Mirrors OpenAI's wire format.
func sseEvent(ss *sseWriter, event string, payload any) {
	b, _ := json.Marshal(payload)
	ss.frame("event: %s\ndata: %s\n\n", event, b)
}
