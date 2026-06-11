package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/constrain"
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
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeErr(w, http.StatusBadRequest, "invalid request body: "+err.Error())
		return
	}
	lm := s.pick(req.Model)
	if lm == nil {
		s.modelNotFound(w, req.Model)
		return
	}

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
		messages = append(messages, chatMessage{Role: "system", Content: req.Instructions})
	}
	inputMsgs, err := responseInputToMessages(req.Input)
	if err != nil {
		writeErr(w, http.StatusBadRequest, err.Error())
		return
	}
	messages = append(messages, inputMsgs...)

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
		s.respondTools(w, lm, req, messages, sm, id, created, store)
		return
	}

	gr, err := lm.prepare(sm, lm.chatPrompt(messages))
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
		f, ok := sseStart(w)
		if !ok {
			return
		}
		sseEvent(w, f, "response.created", map[string]any{
			"type": "response.created", "response": responseObject(id, lm.name, created, "in_progress", []any{}, inTok, 0),
		})
		var sb strings.Builder
		_, nComp, _, _ := lm.drive(r.Context(), gr, func(t string) {
			sb.WriteString(t)
			sseEvent(w, f, "response.output_text.delta", map[string]any{
				"type": "response.output_text.delta", "item_id": id + "-msg", "output_index": 0, "content_index": 0, "delta": t,
			})
		})
		out := []any{outputMessage(id+"-msg", sb.String())}
		sseEvent(w, f, "response.completed", map[string]any{
			"type": "response.completed", "response": responseObject(id, lm.name, created, "completed", out, inTok, nComp),
		})
		sseDone(w, f)
		s.maybeStore(store, id, lm.name, messages, sb.String())
		return
	}

	var sb strings.Builder
	_, nComp, _, _ := lm.drive(r.Context(), gr, func(t string) { sb.WriteString(t) })
	out := []any{outputMessage(id+"-msg", sb.String())}
	writeJSON(w, http.StatusOK, responseObject(id, lm.name, created, "completed", out, inTok, nComp))
	s.maybeStore(store, id, lm.name, messages, sb.String())
}

// respondTools generates a (buffered) tool-calling response and emits
// function_call output items (or a message when the model answered normally).
func (s *server) respondTools(w http.ResponseWriter, lm *loadedModel, req responseReq, messages []chatMessage, sm sampling, id string, created int64, store bool) {
	tools := make([]chat.Tool, len(req.Tools))
	for i, t := range req.Tools {
		tools[i] = chat.Tool{Name: t.Function.Name, Description: t.Function.Description, Parameters: t.Function.Parameters}
	}
	system, turns := messagesToTurns(messages)
	gr, err := lm.prepare(sm, lm.encode(lm.tmpl.RenderTools(system, turns, tools)))
	if err != nil {
		writeErr(w, http.StatusBadRequest, err.Error())
		return
	}
	if forced := forcedTool(req.ToolChoice, tools); forced != nil {
		if prefix, suffix, argsKey, array, ok := lm.tmpl.ToolCallWrapper(); ok {
			if g, gerr := constrain.ToolCallGrammar(prefix, suffix, argsKey, forced.Name, array, forced.Parameters); gerr == nil {
				eos := append(append([]int(nil), lm.eosIDs...), lm.stopIDs...)
				m := constrain.NewMasker(g, constrain.TokenBytes(lm.vocab, lm.tk.TokenText), eos).StopWhenComplete()
				gr.sp.LogitProcessor = m.Process
			}
		}
	}
	if !lm.enter(w) {
		return
	}
	defer lm.exit()
	inTok := len(gr.promptIDs)
	var sb strings.Builder
	_, nComp, _, _ := lm.drive(context.Background(), gr, func(t string) { sb.WriteString(t) })
	calls, lead := lm.tmpl.ParseToolCalls(sb.String())

	var out []any
	if lead != "" {
		out = append(out, outputMessage(id+"-msg", lead))
	}
	for i, c := range calls {
		out = append(out, map[string]any{
			"type": "function_call", "id": fmt.Sprintf("%s-fc%d", id, i),
			"call_id": c.ID, "name": c.Name, "arguments": string(c.Arguments), "status": "completed",
		})
	}
	if len(out) == 0 { // model produced nothing parseable → empty message
		out = append(out, outputMessage(id+"-msg", sb.String()))
	}
	resp := responseObject(id, lm.name, created, "completed", out, inTok, nComp)
	if req.Stream {
		f, ok := sseStart(w)
		if !ok {
			return
		}
		sseEvent(w, f, "response.created", map[string]any{"type": "response.created", "response": responseObject(id, lm.name, created, "in_progress", []any{}, inTok, 0)})
		sseEvent(w, f, "response.completed", map[string]any{"type": "response.completed", "response": resp})
		sseDone(w, f)
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
	full := append(append([]chatMessage(nil), messages...), chatMessage{Role: "assistant", Content: assistant})
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
		return []chatMessage{{Role: "user", Content: str}}, nil
	}
	var items []struct {
		Role    string          `json:"role"`
		Content json.RawMessage `json:"content"`
	}
	if err := json.Unmarshal(raw, &items); err != nil {
		return nil, fmt.Errorf("input must be a string or an array of message items")
	}
	msgs := make([]chatMessage, 0, len(items))
	for _, it := range items {
		role := it.Role
		if role == "" {
			role = "user"
		}
		msgs = append(msgs, chatMessage{Role: role, Content: contentText(it.Content)})
	}
	return msgs, nil
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
	return map[string]any{
		"id": id, "object": "response", "created_at": created, "status": status, "model": model,
		"output": output,
		"usage":  map[string]any{"input_tokens": inTok, "output_tokens": outTok, "total_tokens": inTok + outTok},
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
func sseEvent(w http.ResponseWriter, f http.Flusher, event string, payload any) {
	b, _ := json.Marshal(payload)
	fmt.Fprintf(w, "event: %s\ndata: %s\n\n", event, b)
	f.Flush()
}
