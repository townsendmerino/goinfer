package anthropicapi

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/internal/app/serveapp/carriedids"
	"github.com/townsendmerino/goinfer/internal/app/serveapp/media"
	"github.com/townsendmerino/goinfer/internal/app/serveapp/stores"
	"github.com/townsendmerino/goinfer/internal/app/serveapp/wirecore"
)

// Text extracts plain text from a string or array of Anthropic text blocks.
func Text(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return s
	}
	var blocks []Block
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

// InputBytes sums tokenizable text, replayed tool_use calls, and declared tool schemas across req.
func InputBytes(req *Req) int {
	n := len(Text(req.System))
	for i := range req.Messages {
		n += len(Text(req.Messages[i].Content))
		n += ToolCallBytes(req.Messages[i].Content)
	}
	n += ToolSchemaBytes(req.Tools)
	return n
}

// ToolCallBytes sums name and input bytes across tool_use blocks in raw.
func ToolCallBytes(raw json.RawMessage) int {
	var blocks []Block
	if json.Unmarshal(raw, &blocks) != nil {
		return 0
	}
	n := 0
	for _, bl := range blocks {
		if bl.Type == "tool_use" {
			n += len(bl.Name) + len(bl.Input)
		}
	}
	return n
}

// ToolSchemaBytes sums name, description, and input_schema bytes across tools.
func ToolSchemaBytes(tools []Tool) int {
	n := 0
	for _, t := range tools {
		n += len(t.Name) + len(t.Description) + len(t.InputSchema)
	}
	return n
}

// Carrier parses a carried-id blob from a thinking signature or redacted_thinking data field.
func Carrier(enc, visible string) (string, *carriedids.Carrier) {
	if !strings.HasPrefix(enc, carriedids.V2Prefix) {
		return visible, nil
	}
	b, err := carriedids.ParseBlob(enc)
	if err != nil {
		return visible, &carriedids.Carrier{Reason: carriedids.ReasonParse}
	}
	return b.Text, &carriedids.Carrier{Blob: b}
}

// Turns translates req into a system prompt and merged chat.Turn slice.
func Turns(req *Req) (string, []chat.Turn, *APIErr) {
	system, turns, _, aerr := TurnsCarried(req)
	return system, turns, aerr
}

// TurnsCarried translates req into a system prompt, merged chat.Turn slice, and per-turn carried-id map.
func TurnsCarried(req *Req) (string, []chat.Turn, map[int]*carriedids.Carrier, *APIErr) {
	system := Text(req.System)
	toolNames := map[string]string{}
	var turns []chat.Turn
	var carried []*carriedids.Carrier
	for mi, m := range req.Messages {
		if m.Role == "system" && mi > 0 {
			if s := Text(m.Content); s != "" {
				turns, carried = append(turns, chat.Turn{Role: "user", Content: s, ToolLoop: true}), append(carried, nil)
			}
			continue
		}
		if m.Role != "user" && m.Role != "assistant" {
			return "", nil, nil, BadReq("message %d: role must be %q or %q, got %q "+
				"(the system prompt is the top-level \"system\" field on this API; a \"system\" message is accepted only after the first message)",
				mi, "user", "assistant", m.Role)
		}
		var s string
		if json.Unmarshal(m.Content, &s) == nil {
			turns, carried = append(turns, chat.Turn{Role: wireRole(m.Role), Content: s}), append(carried, nil)
			continue
		}
		var blocks []Block
		if err := json.Unmarshal(m.Content, &blocks); err != nil {
			return "", nil, nil, BadReq("message %d: content must be a string or an array of blocks", mi)
		}
		var text, reasoning strings.Builder
		var calls []chat.ToolCall
		var carrier *carriedids.Carrier
		sawToolResult := false
		for _, bl := range blocks {
			switch bl.Type {
			case "text":
				text.WriteString(bl.Text)
			case "thinking", "redacted_thinking":
				enc := bl.Signature
				if bl.Type == "redacted_thinking" {
					enc = bl.Data
				}
				t, c := Carrier(enc, bl.Thinking)
				if c != nil {
					carrier = c
				}
				reasoning.WriteString(t)
			case "tool_use":
				calls = append(calls, chat.ToolCall{ID: bl.ID, Name: bl.Name, Arguments: defaultObj(bl.Input)})
				if bl.ID != "" {
					toolNames[bl.ID] = bl.Name
				}
			case "tool_result":
				sawToolResult = true
				turns = append(turns, chat.Turn{
					Role:       "tool",
					Content:    Text(bl.Content),
					ToolName:   toolNames[bl.ToolUseID],
					ToolCallID: bl.ToolUseID,
				})
				carried = append(carried, nil)
			case "image":
			default:
			}
		}
		if m.Role == "assistant" {
			if text.Len() > 0 || len(calls) > 0 {
				turns = append(turns, chat.Turn{Role: "assistant", Content: text.String(), ToolCalls: calls, Reasoning: reasoning.String()})
				carried = append(carried, carrier)
			}
		} else if text.Len() > 0 {
			turns = append(turns, chat.Turn{Role: "user", Content: text.String(), ToolLoop: sawToolResult})
			carried = append(carried, nil)
		}
	}
	merged, from := MergeAdjacentFrom(turns)
	var carriers map[int]*carriedids.Carrier
	for i, ks := range from {
		var c *carriedids.Carrier
		for _, k := range ks {
			if carried[k] != nil {
				c = carried[k]
				break
			}
		}
		if c == nil {
			continue
		}
		if len(ks) > 1 && c.Reason == "" {
			c = &carriedids.Carrier{Reason: carriedids.ReasonDecode}
		}
		if carriers == nil {
			carriers = map[int]*carriedids.Carrier{}
		}
		carriers[i] = c
	}
	return system, merged, carriers, nil
}

// Images decodes inline base64 image blocks from req.Messages, rejecting URL sources.
func Images(req *Req) ([]media.ImageRef, error) {
	var out []media.ImageRef
	for _, m := range req.Messages {
		var blocks []Block
		if json.Unmarshal(m.Content, &blocks) != nil {
			continue
		}
		at := 0
		for _, bl := range blocks {
			if bl.Type == "text" {
				at += len(bl.Text)
				continue
			}
			if bl.Type != "image" {
				continue
			}
			if bl.Source == nil || bl.Source.Type != "base64" {
				return nil, fmt.Errorf("image source must be type \"base64\" (url sources are not fetched)")
			}
			ref, err := media.DecodeBase64Image(bl.Source.MediaType, bl.Source.Data)
			if err != nil {
				return nil, err
			}
			ref.At = at
			out = append(out, ref)
		}
	}
	return out, nil
}

// MediaText returns the concatenated text of the newest image-bearing message in req and whether it is the final user message.
func MediaText(req *Req) (string, bool) {
	mediaIdx, lastUser := -1, -1
	var text string
	for i, m := range req.Messages {
		if m.Role == "user" {
			lastUser = i
		}
		var blocks []Block
		if len(m.Content) == 0 || json.Unmarshal(m.Content, &blocks) != nil {
			continue
		}
		var b strings.Builder
		saw := false
		for _, bl := range blocks {
			switch bl.Type {
			case "text":
				b.WriteString(bl.Text)
			case "image":
				saw = true
			}
		}
		if saw {
			mediaIdx, text = i, b.String()
		}
	}
	if mediaIdx < 0 {
		return "", false
	}
	return text, mediaIdx == lastUser
}

// OmitHistoryImages replaces image blocks in all but the newest image-bearing message with omission notes.
func OmitHistoryImages(msgs []Message) int {
	cs := make([]*json.RawMessage, len(msgs))
	for i := range msgs {
		cs[i] = &msgs[i].Content
	}
	return media.OmitHistoryImages(cs, "image")
}

// ContainsImages reports whether any message in req carries an image block without decoding base64 payloads.
func ContainsImages(req *Req) bool {
	for _, m := range req.Messages {
		var blocks []struct {
			Type string `json:"type"`
		}
		if json.Unmarshal(m.Content, &blocks) != nil {
			continue
		}
		for _, bl := range blocks {
			if bl.Type == "image" {
				return true
			}
		}
	}
	return false
}

func wireRole(role string) string {
	if role == "assistant" {
		return "assistant"
	}
	return "user"
}

func defaultObj(raw json.RawMessage) json.RawMessage {
	if len(raw) == 0 {
		return json.RawMessage("{}")
	}
	return raw
}

// MergeAdjacent merges consecutive user or assistant turns in turns.
func MergeAdjacent(turns []chat.Turn) []chat.Turn {
	out, _ := MergeAdjacentFrom(turns)
	return out
}

// MergeAdjacentFrom merges consecutive user or assistant turns and returns the input-turn index slices for each merged turn.
func MergeAdjacentFrom(turns []chat.Turn) ([]chat.Turn, [][]int) {
	var out []chat.Turn
	var from [][]int
	for i, t := range turns {
		if n := len(out); n > 0 && out[n-1].Role == t.Role && (t.Role == "user" || t.Role == "assistant") {
			from[n-1] = append(from[n-1], i)
			prev := &out[n-1]
			switch {
			case prev.Content != "" && t.Content != "":
				prev.Content += "\n" + t.Content
			case t.Content != "":
				prev.Content = t.Content
			}
			prev.ToolCalls = append(prev.ToolCalls, t.ToolCalls...)
			switch {
			case prev.Reasoning != "" && t.Reasoning != "":
				prev.Reasoning += "\n" + t.Reasoning
			case t.Reasoning != "":
				prev.Reasoning = t.Reasoning
			}
			prev.ToolLoop = prev.ToolLoop && t.ToolLoop
			continue
		}
		out = append(out, t)
		from = append(from, []int{i})
	}
	return out, from
}

// Tools converts Anthropic tool declarations into chat.Tool values.
func Tools(in []Tool) []chat.Tool {
	out := make([]chat.Tool, len(in))
	for i, t := range in {
		out[i] = chat.Tool{Name: t.Name, Description: t.Description, Parameters: t.InputSchema}
	}
	return out
}

// ToolMode parses Anthropic tool_choice into mode ("auto", "any", "tool", or "none") and optional named tool.
func ToolMode(raw json.RawMessage) (mode, name string) {
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

// ForcedTool returns the single tool forced by an Anthropic tool_choice ("tool" or lone tool under "any").
func ForcedTool(mode, name string, tools []chat.Tool) *chat.Tool {
	switch mode {
	case "tool":
		for i := range tools {
			if tools[i].Name == name {
				return &tools[i]
			}
		}
	case "any":
		if len(tools) == 1 {
			return &tools[0]
		}
	}
	return nil
}

// UnionMode maps an Anthropic tool_choice mode onto wirecore.BuildToolMasker's union mode.
func UnionMode(mode string) string {
	switch mode {
	case "auto":
		return "auto"
	case "any":
		return "required"
	}
	return ""
}

// ToolsConstrainedFromStart reports whether an Anthropic tool_choice constrains output from token 1.
func ToolsConstrainedFromStart(mode, name string, tools []chat.Tool, unionEnabled bool) bool {
	return wirecore.ToolsConstrainedFromStart(ForcedTool(mode, name, tools), mode == "tool", UnionMode(mode), tools, unionEnabled)
}

// ThinkingBlock formats an Anthropic thinking content block.
func ThinkingBlock(text, signature string) map[string]any {
	return map[string]any{"type": "thinking", "thinking": text, "signature": signature}
}

// RedactedCarrierBlock formats an Anthropic redacted_thinking block carrying minted token IDs when present.
func RedactedCarrierBlock(mintBlob func(string) string) map[string]any {
	if mintBlob == nil {
		return nil
	}
	if enc := mintBlob(""); enc != "" {
		return map[string]any{"type": "redacted_thinking", "data": enc}
	}
	return nil
}

// TextBlock formats an Anthropic text content block.
func TextBlock(text string) map[string]any {
	return map[string]any{"type": "text", "text": text}
}

// ToolUseBlock formats a parsed chat.ToolCall as an Anthropic tool_use content block.
func ToolUseBlock(c chat.ToolCall) map[string]any {
	id := c.ID
	if id == "" {
		id = "toolu_" + stores.MintID()
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

// StopReason maps an internal finish reason and matched stop sequence to Anthropic's stop_reason and stop_sequence pair.
func StopReason(finish, stopSeq string) (string, any) {
	switch {
	case finish == "cancelled":
		return "cancelled", nil
	case finish == "length":
		return "max_tokens", nil
	case stopSeq != "":
		return "stop_sequence", stopSeq
	default:
		return "end_turn", nil
	}
}

// Usage formats Anthropic's input_tokens / cache_read_input_tokens / output_tokens usage map.
func Usage(promptLen, reused, nComp int) map[string]any {
	reused = min(max(reused, 0), promptLen)
	return map[string]any{"input_tokens": promptLen - reused, "cache_read_input_tokens": reused, "output_tokens": nComp}
}

// BuildMessageContent assembles the content blocks and stop reason of a non-streaming Anthropic /v1/messages response.
func BuildMessageContent(wantThinking bool, reasoning, text, lead string, calls []chat.ToolCall, finish, stopHit string, mintBlob func(string) string) (content []map[string]any, reason string, seq any) {
	thinking := wantThinking && reasoning != ""
	if thinking {
		sig := ""
		if mintBlob != nil {
			sig = mintBlob(reasoning)
		}
		content = append(content, ThinkingBlock(reasoning, sig))
	} else if wantThinking {
		if cb := RedactedCarrierBlock(mintBlob); cb != nil {
			content = append(content, cb)
		}
	}
	if len(calls) > 0 {
		if strings.TrimSpace(lead) != "" {
			content = append(content, TextBlock(lead))
		}
		for _, c := range calls {
			content = append(content, ToolUseBlock(c))
		}
		return content, "tool_use", nil
	}
	if !(thinking && text == "") {
		content = append(content, TextBlock(text))
	}
	reason, seq = StopReason(finish, stopHit)
	return content, reason, seq
}
