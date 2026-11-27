package responsesapi

import (
	"encoding/json"
	"fmt"
	"slices"
	"strings"

	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/internal/app/serveapp/carriedids"
	"github.com/townsendmerino/goinfer/internal/app/serveapp/jobs"
	"github.com/townsendmerino/goinfer/internal/app/serveapp/streamframe"
	"github.com/townsendmerino/goinfer/internal/app/serveapp/wirecore"
)

// InputToMessages parses the Responses input (a string, or an array of message, reasoning, function_call, and
// function_call_output items) into wirecore.ChatMessage entries.
func InputToMessages(raw json.RawMessage) ([]wirecore.ChatMessage, error) {
	if len(raw) == 0 {
		return nil, fmt.Errorf("input is required")
	}
	var str string
	if json.Unmarshal(raw, &str) == nil {
		return []wirecore.ChatMessage{{Role: "user", Content: wirecore.RawStr(str)}}, nil
	}
	var items []struct {
		Type             string          `json:"type"`
		Role             string          `json:"role"`
		Content          json.RawMessage `json:"content"`
		CallID           string          `json:"call_id"`
		Name             string          `json:"name"`
		Arguments        string          `json:"arguments"`
		Output           json.RawMessage `json:"output"`
		EncryptedContent string          `json:"encrypted_content"`
		Summary          json.RawMessage `json:"summary"`
	}
	if err := json.Unmarshal(raw, &items); err != nil {
		return nil, fmt.Errorf("input must be a string or an array of message items")
	}
	msgs := make([]wirecore.ChatMessage, 0, len(items))
	pending := ""
	var carried *carriedids.Carrier
	for _, it := range items {
		switch it.Type {
		case "reasoning":
			text, c, err := streamframe.ReasoningItemCarried(it.EncryptedContent, it.Content, it.Summary)
			if err != nil {
				return nil, err
			}
			pending += text
			if c != nil {
				carried = c
			}
			continue
		case "function_call":
			tc := wirecore.APIToolCall{ID: it.CallID, Type: "function"}
			tc.Function.Name = it.Name
			tc.Function.Arguments = it.Arguments
			msgs = append(msgs, wirecore.ChatMessage{Role: "assistant", ToolCalls: []wirecore.APIToolCall{tc}, ReasoningContent: pending, Carried: carried})
			pending, carried = "", nil
			continue
		case "function_call_output":
			msgs = append(msgs, wirecore.ChatMessage{
				Role:       "tool",
				ToolCallID: it.CallID,
				Name:       it.Name,
				Content:    wirecore.RawStr(ToolOutputText(it.Output)),
			})
			continue
		}
		role := it.Role
		if role == "" {
			role = "user"
		}
		content, err := Content(it.Content)
		if err != nil {
			return nil, err
		}
		m := wirecore.ChatMessage{Role: role, Content: content}
		if role == "assistant" {
			m.ReasoningContent, pending = pending, ""
			m.Carried, carried = carried, nil
		}
		msgs = append(msgs, m)
	}
	return msgs, nil
}

// Content converts a Responses message content into a chat message content: plain text when no input_image is present,
// or an ordered chat content array of text and image_url parts. Images referenced by file_id are refused.
func Content(raw json.RawMessage) (json.RawMessage, error) {
	var parts []struct {
		Type     string          `json:"type"`
		Text     string          `json:"text"`
		ImageURL json.RawMessage `json:"image_url"`
		FileID   string          `json:"file_id"`
	}
	if json.Unmarshal(raw, &parts) != nil {
		return wirecore.RawStr(ContentText(raw)), nil
	}
	hasImage := false
	for _, p := range parts {
		hasImage = hasImage || p.Type == "input_image"
	}
	if !hasImage {
		return wirecore.RawStr(ContentText(raw)), nil
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
		if json.Unmarshal(p.ImageURL, &url) != nil {
			var u imageURL
			_ = json.Unmarshal(p.ImageURL, &u)
			url = u.URL
		}
		if url == "" {
			return nil, fmt.Errorf("input_image needs image_url (a base64 data: URI)")
		}
		out = append(out, chatPart{Type: "image_url", ImageURL: &imageURL{URL: url}})
	}
	return json.Marshal(out)
}

// ToolOutputText flattens a function_call_output's output field: a plain string when it is one, content-part text
// when it is an array, or the raw JSON otherwise so non-empty tool outputs never become blank turns.
func ToolOutputText(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var str string
	if json.Unmarshal(raw, &str) == nil {
		return str
	}
	if t := ContentText(raw); t != "" {
		return t
	}
	return string(raw)
}

// ContentText flattens a message content (string or []{type,text}) to plain text.
func ContentText(raw json.RawMessage) string {
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

// TextFormatToRespFormat maps Responses text.format onto the chat API's response_format.
func TextFormatToRespFormat(t *RespText) *wirecore.RespFormat {
	if t == nil || t.Format == nil {
		return nil
	}
	rf := &wirecore.RespFormat{Type: t.Format.Type}
	if t.Format.Type == "json_schema" {
		rf.JSONSchema = &struct {
			Name   string          `json:"name"`
			Schema json.RawMessage `json:"schema"`
		}{Name: t.Format.Name, Schema: t.Format.Schema}
	}
	return rf
}

// Object builds a Responses API response object.
func Object(id, model string, created int64, status string, output []any, inTok, outTok int) map[string]any {
	o := map[string]any{
		"id": id, "object": "response", "created_at": created, "status": status, "model": model,
		"output": output,
		"usage":  map[string]any{"input_tokens": inTok, "output_tokens": outTok, "total_tokens": inTok + outTok},
	}
	if status == "incomplete" {
		o["incomplete_details"] = map[string]any{"reason": "max_output_tokens"}
	}
	return o
}

// ObjectTurn builds a finished turn's response object with timings and optional cost from clock.
func ObjectTurn(clock *jobs.TurnClock, id, model string, created int64, status string, output []any, inTok, outTok int) map[string]any {
	return clock.AddTurn(Object(id, model, created, status, output, inTok, outTok))
}

// WithCachedTokens populates usage.input_tokens_details.cached_tokens on resp.
func WithCachedTokens(resp map[string]any, reused int) map[string]any {
	resp["usage"].(map[string]any)["input_tokens_details"] = map[string]any{"cached_tokens": reused}
	return resp
}

// Status maps an engine finish reason to a Responses API status value.
func Status(finish string) string {
	switch finish {
	case "cancelled":
		return "cancelled"
	case "length":
		return "incomplete"
	default:
		return "completed"
	}
}

// WantsCarrier reports whether include requests reasoning.encrypted_content.
func WantsCarrier(include []string) bool {
	return slices.Contains(include, "reasoning.encrypted_content")
}

// WithCarrier prepends a carrier-only reasoning item when include requests reasoning.encrypted_content, no reasoning
// item is already present, and mintBlob produces a non-empty carried-ID token blob.
func WithCarrier(rs *streamframe.RespStream, mintBlob func(string) string, include []string, id string, items []any) []any {
	if !WantsCarrier(include) || mintBlob == nil {
		return items
	}
	if len(items) > 0 {
		if it, ok := items[0].(map[string]any); ok && it["type"] == "reasoning" {
			return items
		}
	}
	enc := mintBlob("")
	if enc == "" {
		return items
	}
	if rs != nil {
		if !rs.CarrierOnly(enc) {
			return items
		}
		return append([]any{rs.ReasoningDoneItem()}, items...)
	}
	return append([]any{streamframe.ReasoningItem(id+"-rs", "", enc)}, items...)
}

// BuildToolOutputs assembles the output items (optional lead message + function_call items, or fallback raw message)
// and stored APIToolCall records for a buffered tool-calling Responses turn.
func BuildToolOutputs(id string, calls []chat.ToolCall, raw, lead string, rs *streamframe.RespStream, newCallID func() string) ([]any, []wirecore.APIToolCall) {
	var out []any
	var toolCalls []wirecore.APIToolCall
	if lead != "" {
		if rs != nil {
			out = append(out, rs.CloseMessage(lead))
		} else {
			out = append(out, streamframe.OutputMessage(id+"-msg", lead))
		}
	} else if rs.MessageOpen() {
		out = append(out, rs.CloseMessage(""))
	}
	for i, c := range calls {
		callID := c.ID
		if callID == "" {
			callID = "call_" + newCallID()
		}
		fc := map[string]any{
			"type": "function_call", "id": fmt.Sprintf("%s-fc%d", id, i),
			"call_id": callID, "name": c.Name, "arguments": string(c.Arguments), "status": "completed",
		}
		out = append(out, fc)
		if rs != nil {
			rs.FunctionCall(fc)
		}
		tc := wirecore.APIToolCall{ID: callID, Type: "function"}
		tc.Function.Name = c.Name
		tc.Function.Arguments = string(c.Arguments)
		toolCalls = append(toolCalls, tc)
	}
	if len(out) == 0 {
		if rs != nil {
			out = append(out, rs.CloseMessage(raw))
		} else {
			out = append(out, streamframe.OutputMessage(id+"-msg", raw))
		}
	}
	return out, toolCalls
}
