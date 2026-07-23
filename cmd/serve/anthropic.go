package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"

	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/constrain"
)

// Anthropic Messages API (POST /v1/messages, POST /v1/messages/count_tokens) —
// the second de-facto standard chat surface (llama.cpp, Ollama, LM Studio all
// serve it), and the one Anthropic-speaking tools — Claude Code included — speak
// via ANTHROPIC_BASE_URL. We translate at the edge into the same internal path
// the OpenAI handlers use (system + chat.Turn + sampling + chat.Tool) and
// translate the result back out; drive/prepare are unchanged. Compatibility bar
// (same as llama.cpp's): "works for the apps that matter", not full-spec.

// --- request shapes (the subset we honor) ---

type anthropicReq struct {
	Model         string             `json:"model"`
	MaxTokens     *int               `json:"max_tokens"` // REQUIRED for /v1/messages (400 if missing/<=0)
	System        json.RawMessage    `json:"system"`     // string | [{"type":"text","text":…}, …]
	Messages      []anthropicMessage `json:"messages"`
	Tools         []anthropicTool    `json:"tools"`
	ToolChoice    json.RawMessage    `json:"tool_choice"` // {"type":"auto"|"any"|"tool","name":…}
	Temperature   *float64           `json:"temperature"` // Anthropic range 0..1 (no rescale)
	TopP          *float64           `json:"top_p"`
	TopK          *int               `json:"top_k"`
	StopSequences []string           `json:"stop_sequences"`
	Stream        bool               `json:"stream"`
	// metadata and thinking are accepted and ignored (unknown JSON fields are
	// dropped by the decoder); stop_reason is therefore never "thinking" in v1.
}

type anthropicMessage struct {
	Role    string          `json:"role"`    // "user" | "assistant"
	Content json.RawMessage `json:"content"` // string | []anthropicBlock
}

// anthropicTool is a tool declaration. Note: input_schema (a JSON Schema), not
// the OpenAI function.parameters nesting.
type anthropicTool struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	InputSchema json.RawMessage `json:"input_schema"`
}

// anthropicBlock is the union of the content-block types we read across the
// system field and message content. Unknown block types and cache_control
// fields are accepted and ignored (Claude Code sends cache_control on every
// request; rejecting it breaks the headline use case).
type anthropicBlock struct {
	Type string `json:"type"`
	Text string `json:"text"` // type:text
	// type:tool_use (assistant replay):
	ID    string          `json:"id"`
	Name  string          `json:"name"`
	Input json.RawMessage `json:"input"`
	// type:tool_result (user replay):
	ToolUseID string          `json:"tool_use_id"`
	Content   json.RawMessage `json:"content"` // string | []text block
	// type:image (multimodal): source.{type:"base64", media_type, data}
	Source *struct {
		Type      string `json:"type"`
		MediaType string `json:"media_type"`
		Data      string `json:"data"`
	} `json:"source"`
}

// toSampling maps the Anthropic sampling fields onto the shared sampling struct
// so lm.prepare translates them identically to the OpenAI path. stop_sequences
// becomes the JSON "stop" array that parseStop already reads.
func (r *anthropicReq) toSampling() sampling {
	sm := sampling{
		Temperature: r.Temperature,
		TopP:        r.TopP,
		TopK:        r.TopK,
		MaxTokens:   r.MaxTokens,
	}
	if len(r.StopSequences) > 0 {
		b, _ := json.Marshal(r.StopSequences)
		sm.Stop = b
	}
	return sm
}

// --- errors ---

// apiErr carries an Anthropic-shaped error (status + kind + message) so a
// translation step can return it and the handler render it at the edge.
type apiErr struct {
	code int
	kind string
	msg  string
}

func (e *apiErr) Error() string { return e.msg }

func (e *apiErr) write(w http.ResponseWriter) { writeAnthropicErr(w, e.code, e.kind, e.msg) }

func badReq(format string, a ...any) *apiErr {
	return &apiErr{http.StatusBadRequest, "invalid_request_error", fmt.Sprintf(format, a...)}
}

// writeAnthropicErr emits the Anthropic error body shape for any non-2xx. Kinds:
// invalid_request_error (400), not_found_error (404, unknown model),
// overloaded_error (529, model busy), api_error (500).
func writeAnthropicErr(w http.ResponseWriter, code int, kind, msg string) {
	writeJSON(w, code, map[string]any{
		"type":  "error",
		"error": map[string]any{"type": kind, "message": msg},
	})
}

// decodeAnthropicJSON is decodeJSON for the Anthropic dialect: 413 when the
// (size-bounded, see maxBytes) body exceeded the limit, else 400. M3.
func decodeAnthropicJSON(w http.ResponseWriter, r *http.Request, v any) bool {
	if err := json.NewDecoder(r.Body).Decode(v); err != nil {
		var mbe *http.MaxBytesError
		if errors.As(err, &mbe) {
			writeAnthropicErr(w, http.StatusRequestEntityTooLarge, "invalid_request_error", "request body too large")
			return false
		}
		writeAnthropicErr(w, http.StatusBadRequest, "invalid_request_error", "invalid request body: "+err.Error())
		return false
	}
	return true
}

func (s *server) anthropicModelNotFound(w http.ResponseWriter, name string) {
	writeAnthropicErr(w, http.StatusNotFound, "not_found_error",
		fmt.Sprintf("model %q not found (served: %s)", name, strings.Join(s.servedNames(), ", ")))
}

// --- translation: Anthropic request → internal (system, turns) ---

// anthropicText reads a string-or-array-of-text-blocks field (the system prompt,
// a tool_result's content) and returns the concatenated text. Non-text blocks
// are ignored.
func anthropicText(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return s
	}
	var blocks []anthropicBlock
	if json.Unmarshal(raw, &blocks) != nil {
		return ""
	}
	var b strings.Builder
	for _, bl := range blocks {
		if bl.Type == "text" {
			b.WriteString(bl.Text)
		}
	}
	return b.String()
}

// anthropicTurns maps the system prompt + messages into the internal chat shape.
// Content may be a plain string or an array of blocks; tool_use blocks (assistant
// replay) become ToolCalls, tool_result blocks (user replay) become "tool"
// turns, image blocks are a 400, and unknown blocks / cache_control are ignored.
// Adjacent same-role user/assistant turns are merged (the spec says roles
// alternate, but we don't enforce it).
func anthropicTurns(req *anthropicReq) (string, []chat.Turn, *apiErr) {
	system := anthropicText(req.System)
	toolNames := map[string]string{} // tool_use id → name, to label tool-result turns
	var turns []chat.Turn
	for mi, m := range req.Messages {
		// Plain-string content: the common case.
		var s string
		if json.Unmarshal(m.Content, &s) == nil {
			turns = append(turns, chat.Turn{Role: anthropicRole(m.Role), Content: s})
			continue
		}
		var blocks []anthropicBlock
		if err := json.Unmarshal(m.Content, &blocks); err != nil {
			return "", nil, badReq("message %d: content must be a string or an array of blocks", mi)
		}
		var text strings.Builder
		var calls []chat.ToolCall
		for _, bl := range blocks {
			switch bl.Type {
			case "text":
				text.WriteString(bl.Text)
			case "tool_use":
				calls = append(calls, chat.ToolCall{ID: bl.ID, Name: bl.Name, Arguments: defaultObj(bl.Input)})
				if bl.ID != "" {
					toolNames[bl.ID] = bl.Name
				}
			case "tool_result":
				turns = append(turns, chat.Turn{
					Role:       "tool",
					Content:    anthropicText(bl.Content),
					ToolName:   toolNames[bl.ToolUseID],
					ToolCallID: bl.ToolUseID,
				})
			case "image":
				// Collected separately by anthropicImages and routed to the vision
				// path; skip here. The handler 400s an image when no vision tower is
				// loaded, so a text-only model never silently drops it.
			default:
				// unknown block type (and cache_control-bearing blocks) — ignore.
			}
		}
		if m.Role == "assistant" {
			if text.Len() > 0 || len(calls) > 0 {
				turns = append(turns, chat.Turn{Role: "assistant", Content: text.String(), ToolCalls: calls})
			}
		} else if text.Len() > 0 {
			turns = append(turns, chat.Turn{Role: "user", Content: text.String()})
		}
	}
	return system, mergeAdjacent(turns), nil
}

// anthropicImages collects inline images from the message content blocks
// (type:image with a base64 source). A non-base64 source is an error (no URL
// fetch — SSRF guard). No images → nil.
func anthropicImages(req *anthropicReq) ([]imageRef, error) {
	var out []imageRef
	for _, m := range req.Messages {
		var blocks []anthropicBlock
		if json.Unmarshal(m.Content, &blocks) != nil {
			continue // plain-string content carries no images
		}
		for _, bl := range blocks {
			if bl.Type != "image" {
				continue
			}
			if bl.Source == nil || bl.Source.Type != "base64" {
				return nil, fmt.Errorf("image source must be type \"base64\" (url sources are not fetched)")
			}
			ref, err := decodeBase64Image(bl.Source.MediaType, bl.Source.Data)
			if err != nil {
				return nil, err
			}
			out = append(out, ref)
		}
	}
	return out, nil
}

// anthropicRole maps the wire role to an internal turn role (assistant passes
// through; everything else is a user turn).
func anthropicRole(role string) string {
	if role == "assistant" {
		return "assistant"
	}
	return "user"
}

// defaultObj normalizes an empty tool input to "{}" so it renders/parses as an
// object.
func defaultObj(raw json.RawMessage) json.RawMessage {
	if len(raw) == 0 {
		return json.RawMessage("{}")
	}
	return raw
}

// mergeAdjacent concatenates consecutive same-role user/assistant turns (tool
// turns never merge), joining text with a newline and appending tool calls.
func mergeAdjacent(turns []chat.Turn) []chat.Turn {
	var out []chat.Turn
	for _, t := range turns {
		if n := len(out); n > 0 && out[n-1].Role == t.Role && (t.Role == "user" || t.Role == "assistant") {
			prev := &out[n-1]
			switch {
			case prev.Content != "" && t.Content != "":
				prev.Content += "\n" + t.Content
			case t.Content != "":
				prev.Content = t.Content
			}
			prev.ToolCalls = append(prev.ToolCalls, t.ToolCalls...)
			continue
		}
		out = append(out, t)
	}
	return out
}

// anthropicTools maps tool declarations to the internal chat.Tool shape
// (input_schema → Parameters).
func anthropicTools(in []anthropicTool) []chat.Tool {
	out := make([]chat.Tool, len(in))
	for i, t := range in {
		out[i] = chat.Tool{Name: t.Name, Description: t.Description, Parameters: t.InputSchema}
	}
	return out
}

// anthropicToolMode parses tool_choice into its type ("auto" default, "any",
// "tool", "none") and the named tool (for type:"tool").
func anthropicToolMode(raw json.RawMessage) (mode, name string) {
	if len(raw) == 0 {
		return "auto", ""
	}
	var tc struct {
		Type string `json:"type"`
		Name string `json:"name"`
	}
	if json.Unmarshal(raw, &tc) == nil && tc.Type != "" {
		return tc.Type, tc.Name
	}
	return "auto", ""
}

// anthropicForcedTool returns the tool a call MUST be, so constrained decoding
// can make a malformed call impossible: a named tool (type:"tool"), or the lone
// tool under type:"any" (must-call). Unlike the OpenAI path, "auto" never forces
// — Anthropic "auto" means the model may answer with text instead, and forcing a
// lone tool there would trap an agent (Claude Code) in an endless tool loop
// after every tool_result. nil = don't constrain.
func anthropicForcedTool(mode, name string, tools []chat.Tool) *chat.Tool {
	switch mode {
	case "tool":
		for i := range tools {
			if tools[i].Name == name {
				return &tools[i]
			}
		}
	case "any":
		if len(tools) == 1 { // must-call, lone tool ⇒ that one
			return &tools[0]
		}
	}
	return nil // auto / none / "any" with multiple tools: the model decides freely
}

// applyToolConstraint wires constrained decoding for an unambiguous tool call
// (lone or forced tool with a family JSON call form), exactly as handleChatTools.
func applyToolConstraint(lm *loadedModel, gr *genRequest, mode, name string, tools []chat.Tool) {
	forced := anthropicForcedTool(mode, name, tools)
	if forced == nil {
		return
	}
	if prefix, suffix, argsKey, array, ok := lm.tmpl.ToolCallWrapper(); ok {
		if g, gerr := constrain.ToolCallGrammar(prefix, suffix, argsKey, forced.Name, array, forced.Parameters); gerr == nil {
			eos := append(append([]int(nil), lm.eosIDs...), lm.stopIDs...)
			m := constrain.NewMasker(g, constrain.TokenBytes(lm.vocab, lm.tk.TokenText), eos).StopWhenComplete()
			gr.sp.LogitProcessor = m.Process
			gr.masker = m // enables grammar-fused speculative decode (drive)
		}
	}
}

// --- response blocks + stop reason ---

func textBlock(text string) map[string]any { return map[string]any{"type": "text", "text": text} }

// toolUseBlock renders a parsed call as an Anthropic tool_use block (input is the
// decoded arguments object; an empty/missing id gets a toolu_ id).
func toolUseBlock(c chat.ToolCall) map[string]any {
	id := c.ID
	if id == "" {
		id = "toolu_" + reqID()
	}
	var input any = map[string]any{}
	if len(c.Arguments) > 0 {
		var v any
		if json.Unmarshal(c.Arguments, &v) == nil {
			input = v
		}
	}
	return map[string]any{"type": "tool_use", "id": id, "name": c.Name, "input": input}
}

// anthropicStopReason maps drive's finish reason to the Anthropic stop_reason +
// stop_sequence pair: token budget → max_tokens; a stop string hit →
// stop_sequence (with the sequence); otherwise (EOS / turn-stop) → end_turn.
func anthropicStopReason(finish, stopSeq string) (string, any) {
	switch {
	case finish == "length":
		return "max_tokens", nil
	case stopSeq != "":
		return "stop_sequence", stopSeq
	default:
		return "end_turn", nil
	}
}

// --- handlers ---

func (s *server) handleMessages(w http.ResponseWriter, r *http.Request) {
	var req anthropicReq
	if !decodeAnthropicJSON(w, r, &req) {
		return
	}
	if req.MaxTokens == nil || *req.MaxTokens <= 0 {
		writeAnthropicErr(w, http.StatusBadRequest, "invalid_request_error", "max_tokens is required and must be > 0")
		return
	}
	lm := s.pick(req.Model)
	if lm == nil {
		s.anthropicModelNotFound(w, req.Model)
		return
	}
	// Multimodal: image blocks route to the vision path.
	imgs, ierr := anthropicImages(&req)
	if ierr != nil {
		writeAnthropicErr(w, http.StatusBadRequest, "invalid_request_error", ierr.Error())
		return
	}
	if len(imgs) > 0 {
		s.serveVisionMessages(w, r, &req, lm, imgs)
		return
	}
	system, turns, aerr := anthropicTurns(&req)
	if aerr != nil {
		aerr.write(w)
		return
	}

	mode, name := anthropicToolMode(req.ToolChoice)
	toolsActive := len(req.Tools) > 0 && mode != "none"
	var tools []chat.Tool
	var gr genRequest
	var ids []int
	var err error
	if toolsActive {
		if lm.tmpl == nil || !lm.tmpl.SupportsTools() {
			writeAnthropicErr(w, http.StatusBadRequest, "invalid_request_error", "this model has no tool-calling template")
			return
		}
		tools = anthropicTools(req.Tools)
		ids, err = lm.tk.EncodeSegments(lm.tmpl.RenderToolsSegments(system, turns, tools), false) // M25
	} else {
		ids, err = lm.promptFor(system, turns)
	}
	if err != nil {
		writeAnthropicErr(w, http.StatusInternalServerError, "api_error", "encode: "+err.Error())
		return
	}
	if gr, err = lm.prepare(req.toSampling(), ids); err != nil {
		writeAnthropicErr(w, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	if toolsActive {
		applyToolConstraint(lm, &gr, mode, name, tools)
	}

	// A full queue is honest backpressure: 529 overloaded_error (the kind
	// Anthropic clients back off on), not the OpenAI 429.
	if !lm.tryEnter() {
		w.Header().Set("Retry-After", "1")
		writeAnthropicErr(w, 529, "overloaded_error", fmt.Sprintf("model %q is busy; retry", lm.name))
		return
	}
	defer lm.exit()

	if req.Stream {
		s.streamMessages(w, r, lm, gr, toolsActive)
		return
	}

	var sb strings.Builder
	finish, nComp, _, stopSeq, gerr := lm.drive(r.Context(), gr, func(t string) { sb.WriteString(t) })
	if gerr != nil {
		writeAnthropicErr(w, http.StatusInternalServerError, "api_error", "generation failed: "+gerr.Error())
		return
	}

	var content []map[string]any
	reason := ""
	var seq any
	if toolsActive {
		calls, lead := lm.tmpl.ParseToolCalls(sb.String())
		if len(calls) > 0 {
			if strings.TrimSpace(lead) != "" {
				content = append(content, textBlock(lead))
			}
			for _, c := range calls {
				content = append(content, toolUseBlock(c))
			}
			reason = "tool_use"
		}
	}
	if reason == "" { // plain text turn (no tools, or the model declined to call)
		content = []map[string]any{textBlock(sb.String())}
		reason, seq = anthropicStopReason(finish, stopSeq)
	}

	writeJSON(w, http.StatusOK, map[string]any{
		"id":            "msg_" + reqID(),
		"type":          "message",
		"role":          "assistant",
		"model":         lm.name,
		"content":       content,
		"stop_reason":   reason,
		"stop_sequence": seq,
		"usage":         map[string]any{"input_tokens": len(gr.promptIDs), "output_tokens": nComp},
	})
}

// handleCountTokens renders the prompt exactly as /v1/messages would (system +
// messages + tools through the chat template), encodes it, and returns the token
// count — no generation, no decode mutex (encoding is independent of the decoder).
func (s *server) handleCountTokens(w http.ResponseWriter, r *http.Request) {
	var req anthropicReq
	if !decodeAnthropicJSON(w, r, &req) {
		return
	}
	lm := s.pick(req.Model)
	if lm == nil {
		s.anthropicModelNotFound(w, req.Model)
		return
	}
	system, turns, aerr := anthropicTurns(&req)
	if aerr != nil {
		aerr.write(w)
		return
	}
	var ids []int
	var err error
	if mode, _ := anthropicToolMode(req.ToolChoice); len(req.Tools) > 0 && mode != "none" && lm.tmpl != nil && lm.tmpl.SupportsTools() {
		ids, err = lm.tk.EncodeSegments(lm.tmpl.RenderToolsSegments(system, turns, anthropicTools(req.Tools)), false) // M25
	} else {
		ids, err = lm.promptFor(system, turns)
	}
	if err != nil {
		writeAnthropicErr(w, http.StatusInternalServerError, "api_error", "encode: "+err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"input_tokens": len(ids)})
}
