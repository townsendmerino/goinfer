package serveapp

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/townsendmerino/goinfer/chat"
)

// OpenAI Responses API (/v1/responses) — Track B Inc4. Phase A is stateless
// (input + instructions + text.format + tools + streaming); Phase B adds
// store/previous_response_id via an in-memory ring, so a continued response is a
// prompt-prefix extension that rides the per-model sessionLRU for warm KV.
// Out of scope: hosted tools, reasoning items, file inputs.

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
	sampling
}

// --- Phase B: in-memory response store (id → conversation) ---

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

	// G1c, extended (audit-2026-09-02 M-21). Placed BEFORE the tools/plain branch so one guard
	// covers both — /v1/responses was two of the five routes that tokenized an arbitrary body
	// before rejecting it, and the assembled `messages` here include anything a stored
	// previous_response_id dragged in, which is the input the BPE would actually run over.
	if err := lm.promptTooLargeForContext(chatInputBytes(messages)); err != nil {
		writeErr(w, http.StatusBadRequest, err.Error())
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

	// Tool path: render declarations, constrain when unambiguous, buffer the full
	// output, parse into function_call items (buffered, like the chat tools path).
	if len(req.Tools) > 0 && toolChoiceMode(req.ToolChoice) != "none" && lm.tmpl != nil && lm.tmpl.SupportsTools() {
		s.respondTools(w, r, lm, req, messages, sm, id, created, store)
		return
	}

	ids, err := lm.chatPrompt(messages)
	if err != nil {
		writeServerErr(w, "encode: "+err.Error())
		return
	}
	gr, err := lm.prepare(sm, ids, lm.adapter == "")
	if err != nil {
		writeErr(w, http.StatusBadRequest, err.Error())
		return
	}
	if !lm.enter(w) {
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
		finish, nComp, _, _, gerr := lm.drive(r.Context(), gr, func(t string) {
			sb.WriteString(t)
			sseEvent(ss, "response.output_text.delta", map[string]any{
				"type": "response.output_text.delta", "item_id": id + "-msg", "output_index": 0, "content_index": 0, "delta": t,
			})
		})
		if gerr != nil {
			sseEvent(ss, "error", map[string]any{"type": "error", "message": "generation failed: " + gerr.Error()})
			sseDone(ss)
			return
		}
		out := []any{outputMessage(id+"-msg", sb.String())}
		sseEvent(ss, "response.completed", map[string]any{
			"type": "response.completed", "response": responseObject(id, lm.name, created, respStatus(finish), out, inTok, nComp),
		})
		sseDone(ss)
		s.maybeStore(store, id, lm.name, messages, sb.String())
		return
	}

	var sb strings.Builder
	finish, nComp, _, _, gerr := lm.drive(r.Context(), gr, func(t string) { sb.WriteString(t) })
	if gerr != nil {
		writeServerErr(w, "generation failed: "+gerr.Error())
		return
	}
	out := []any{outputMessage(id+"-msg", sb.String())}
	writeJSON(w, http.StatusOK, responseObject(id, lm.name, created, respStatus(finish), out, inTok, nComp))
	s.maybeStore(store, id, lm.name, messages, sb.String())
}

// respondTools generates a (buffered) tool-calling response and emits
// function_call output items (or a message when the model answered normally).
func (s *server) respondTools(w http.ResponseWriter, r *http.Request, lm *loadedModel, req responseReq, messages []chatMessage, sm sampling, id string, created int64, store bool) {
	tools := make([]chat.Tool, len(req.Tools))
	for i, t := range req.Tools {
		tools[i] = chat.Tool{Name: t.Function.Name, Description: t.Function.Description, Parameters: t.Function.Parameters}
	}
	system, turns := messagesToTurns(messages)
	ids, err := lm.tk.EncodeSegments(lm.tmpl.RenderToolsSegments(system, turns, tools), false) // M25
	if err != nil {
		writeServerErr(w, "encode: "+err.Error())
		return
	}
	gr, err := lm.prepare(sm, ids, lm.adapter == "")
	if err != nil {
		writeErr(w, http.StatusBadRequest, err.Error())
		return
	}
	forced := forcedTool(req.ToolChoice, tools)
	namedForce := toolChoiceMode(req.ToolChoice) == "function"
	if cerr := constrainForcedTool(lm, &gr, forced, namedForce); cerr != nil {
		writeErr(w, http.StatusBadRequest, cerr.Error()) // named tool_choice unconstrainable → 400 (M-05)
		return
	}
	if !lm.enter(w) {
		return
	}
	defer lm.exit()
	inTok := len(gr.promptIDs)
	// G19: same buffered-and-therefore-silent shape as the chat tools path — start
	// SSE first and keep the stream alive with comment frames while the buffer
	// fills, so a slow generation is not indistinguishable from a dead server. See
	// tools.go for the full rationale and the 500-vs-sseErr consequence.
	var ss *sseWriter
	if req.Stream {
		var ok bool
		if ss, ok = sseStart(w); !ok {
			return
		}
	}
	var sb strings.Builder
	var stopBeat func()
	if ss != nil {
		stopBeat = sseHeartbeat(ss)
	}
	finish, nComp, _, _, gerr := lm.drive(r.Context(), gr, func(t string) { sb.WriteString(t) })
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

	var out []any
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
	}
	if len(out) == 0 { // model produced nothing parseable → empty message
		out = append(out, outputMessage(id+"-msg", sb.String()))
	}
	// Reflect the real finish, like the non-tools paths: a tool turn cut off by max_output_tokens is
	// "incomplete", not "completed" (audit R-16 — the N-15 residual in the tools branch).
	resp := responseObject(id, lm.name, created, respStatus(finish), out, inTok, nComp)
	if req.Stream {
		sseEvent(ss, "response.created", map[string]any{"type": "response.created", "response": responseObject(id, lm.name, created, "in_progress", []any{}, inTok, 0)})
		sseEvent(ss, "response.completed", map[string]any{"type": "response.completed", "response": resp})
		sseDone(ss)
	} else {
		writeJSON(w, http.StatusOK, resp)
	}
	// Tool-call continuations round-trip via the next request's input; store the
	// assistant text (the lead, if any) for previous_response_id continuity.
	s.maybeStore(store, id, lm.name, messages, lead)
}

func (s *server) maybeStore(store bool, id, model string, messages []chatMessage, assistant string) {
	if !store || s.responses == nil {
		return
	}
	full := append(append([]chatMessage(nil), messages...), chatMessage{Role: "assistant", Content: rawStr(assistant)})
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
	// `type` and the function-call fields are decoded, not just {role, content}. A Responses tool
	// loop feeds the model's own `function_call` back with a `function_call_output`, and NEITHER
	// carries a role or a content field — so both used to fall through to the default below and
	// become `{Role:"user", Content:""}`: two empty user turns. The model never saw the tool
	// result, so it either answered without it or called the same tool again, forever, under HTTP
	// 200. docs/server.md and this file's own comment both claim the round-trip works;
	// TestServe_responses step 4 never feeds a result back, which is why nothing caught it
	// (audit-2026-09-02 M-18).
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
		msgs = append(msgs, chatMessage{Role: role, Content: rawStr(contentText(it.Content))})
	}
	return msgs, nil
}

// toolOutputText flattens a `function_call_output`'s `output`: a plain string when it is one, the
// content-part text when it is an array, and the raw JSON otherwise. Never empty for a non-empty
// input — an empty tool turn is the M-18 failure, so the fallback keeps SOMETHING the model can
// read rather than silently dropping the result.
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
	// A generation cut off by max_output_tokens is "incomplete", not "completed" (N-15).
	if status == "incomplete" {
		o["incomplete_details"] = map[string]any{"reason": "max_output_tokens"}
	}
	return o
}

// respStatus maps drive's finish reason to a Responses status: "length" (truncated by
// max_output_tokens) → "incomplete"; everything else → "completed" (N-15).
func respStatus(finish string) string {
	if finish == "length" {
		return "incomplete"
	}
	return "completed"
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
