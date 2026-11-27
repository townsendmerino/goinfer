package responsesapi

import (
	"encoding/json"
	"slices"

	"github.com/townsendmerino/goinfer/internal/app/serveapp/stores"
	"github.com/townsendmerino/goinfer/internal/app/serveapp/streamframe"
	"github.com/townsendmerino/goinfer/internal/app/serveapp/wirecore"
)

// RespText describes the Responses API text.format field.
type RespText struct {
	Format *struct {
		Type   string          `json:"type"` // text | json_object | json_schema
		Name   string          `json:"name"`
		Schema json.RawMessage `json:"schema"`
	} `json:"format"`
}

// Req is the decoded JSON body of a POST /v1/responses request.
type Req struct {
	Model              string              `json:"model"`
	Input              json.RawMessage     `json:"input"` // string | message items
	Instructions       string              `json:"instructions"`
	MaxOutputTokens    *int                `json:"max_output_tokens"`
	Text               *RespText           `json:"text"`
	Tools              []wirecore.ToolSpec `json:"tools"`
	ToolChoice         json.RawMessage     `json:"tool_choice"`
	Stream             bool                `json:"stream"`
	Store              *bool               `json:"store"` // default true (capped ring)
	PreviousResponseID string              `json:"previous_response_id"`
	Reasoning          *struct {
		Effort string `json:"effort"`
	} `json:"reasoning"`
	Include []string `json:"include"`
	wirecore.Sampling
}

// ThinkRequest projects Req's reasoning effort onto streamframe.ThinkRequest.
func (r *Req) ThinkRequest() streamframe.ThinkRequest {
	tr := streamframe.ThinkRequest{}
	if r.Reasoning != nil {
		tr.ReasoningEffort = r.Reasoning.Effort
	}
	return tr
}

// ApplyFast rewrites Req for a "<model>-fast" alias request by turning reasoning off while preserving output limits.
func (r *Req) ApplyFast() {
	r.Reasoning = &struct {
		Effort string `json:"effort"`
	}{Effort: "none"}
}

// SamplingSettings maps Responses max_output_tokens and text.format onto the shared wirecore.Sampling contract.
func (r *Req) SamplingSettings() wirecore.Sampling {
	sm := r.Sampling
	sm.OutputByWindow = true
	if r.MaxOutputTokens != nil {
		sm.MaxTokens = r.MaxOutputTokens
	}
	sm.ResponseFormat = TextFormatToRespFormat(r.Text)
	return sm
}

// ShouldStore reports whether the response should be stored in the previous_response_id ring (default true).
func (r *Req) ShouldStore() bool {
	return r.Store == nil || *r.Store
}

// NewStore creates a capped in-memory response store priced by MessagesBytes.
func NewStore(cap int) *stores.ResponseStore[wirecore.ChatMessage] {
	return stores.NewResponseStore(cap, MessagesBytes)
}

// MessagesBytes estimates the variable-length byte footprint held by (model, messages), including carried-ID blobs.
func MessagesBytes(model string, messages []wirecore.ChatMessage) int {
	n := len(model)
	for _, m := range messages {
		n += len(m.Role) + len(m.Content) + len(m.Name) + len(m.ToolCallID) + len(m.ReasoningContent) + len(m.Reasoning)
		for _, tc := range m.ToolCalls {
			n += len(tc.ID) + len(tc.Type) + len(tc.Function.Name) + len(tc.Function.Arguments)
		}
		if c := m.Carried; c != nil {
			n += len(c.Reason)
			if b := c.Blob; b != nil {
				n += len(b.Model) + len(b.NS) + len(b.Text) + 8*len(b.IDs) + len(b.MAC)
			}
		}
	}
	return n
}

// AppendTurn returns a cloned message slice with the assistant's text, reasoning, and tool calls appended.
func AppendTurn(messages []wirecore.ChatMessage, assistant, reasoning string, toolCalls []wirecore.APIToolCall) []wirecore.ChatMessage {
	return append(slices.Clone(messages),
		wirecore.ChatMessage{Role: "assistant", Content: wirecore.RawStr(assistant), ToolCalls: toolCalls, ReasoningContent: reasoning})
}
