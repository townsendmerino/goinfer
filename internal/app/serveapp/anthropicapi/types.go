package anthropicapi

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"github.com/townsendmerino/goinfer/internal/app/serveapp/jobs"
	"github.com/townsendmerino/goinfer/internal/app/serveapp/netguard"
	"github.com/townsendmerino/goinfer/internal/app/serveapp/streamframe"
	"github.com/townsendmerino/goinfer/internal/app/serveapp/wirecore"
)

// Req is an Anthropic /v1/messages or /v1/messages/count_tokens request body.
type Req struct {
	Model         string              `json:"model"`
	MaxTokens     *int                `json:"max_tokens"`
	System        json.RawMessage     `json:"system"`
	Messages      []Message           `json:"messages"`
	Tools         []Tool              `json:"tools"`
	ToolChoice    json.RawMessage     `json:"tool_choice"`
	Temperature   *float64            `json:"temperature"`
	TopP          *float64            `json:"top_p"`
	TopK          *int                `json:"top_k"`
	StopSequences []string            `json:"stop_sequences"`
	Stream        bool                `json:"stream"`
	LogitBias     map[string]float64  `json:"logit_bias"`
	Goinfer       *jobs.VendorOptions `json:"goinfer"`
	Thinking      *struct {
		Type         string `json:"type"`
		BudgetTokens int    `json:"budget_tokens"`
	} `json:"thinking"`
	OutputConfig *struct {
		Format *struct {
			Type   string          `json:"type"`
			Schema json.RawMessage `json:"schema"`
		} `json:"format"`
	} `json:"output_config"`
}

// OutputFormat validates output_config.format and maps it onto wirecore.RespFormat.
func (r *Req) OutputFormat() (*wirecore.RespFormat, *APIErr) {
	if r.OutputConfig == nil || r.OutputConfig.Format == nil {
		return nil, nil
	}
	f := r.OutputConfig.Format
	if f.Type != "json_schema" {
		return nil, BadReq("output_config.format.type must be %q (got %q)", "json_schema", f.Type)
	}
	if s := strings.TrimSpace(string(f.Schema)); s == "" || s == "null" {
		return nil, BadReq("output_config.format.schema is required when output_config.format.type is %q", "json_schema")
	}
	if mode, _ := ToolMode(r.ToolChoice); len(r.Tools) > 0 && mode != "none" {
		return nil, BadReq("output_config.format is not supported together with tools on this server; send one or the other, or tool_choice {\"type\": \"none\"}")
	}
	rf := &wirecore.RespFormat{Type: "json_schema", JSONSchema: &struct {
		Name   string          `json:"name"`
		Schema json.RawMessage `json:"schema"`
	}{Schema: f.Schema}}
	if _, err := wirecore.GrammarFor(rf); err != nil {
		return nil, BadReq("output_config.format.schema cannot be constrained: %v", err)
	}
	return rf, nil
}

// ThinkRequest converts r.Thinking into streamframe.ThinkRequest.
func (r *Req) ThinkRequest() streamframe.ThinkRequest {
	tr := streamframe.ThinkRequest{}
	if r.Thinking != nil {
		tr.AnthropicType = r.Thinking.Type
		if r.Thinking.Type == "enabled" {
			tr.Budget = r.Thinking.BudgetTokens
		}
	}
	return tr
}

// WantsThinking reports whether the caller requested thinking blocks back ("enabled" or "adaptive").
func (r *Req) WantsThinking() bool {
	return r.Thinking != nil && (r.Thinking.Type == "enabled" || r.Thinking.Type == "adaptive")
}

// ToSampling maps Anthropic sampling and output_config fields onto wirecore.Sampling.
func (r *Req) ToSampling() (wirecore.Sampling, *APIErr) {
	rf, aerr := r.OutputFormat()
	if aerr != nil {
		return wirecore.Sampling{}, aerr
	}
	sm := wirecore.Sampling{
		Temperature:    r.Temperature,
		TopP:           r.TopP,
		TopK:           r.TopK,
		MaxTokens:      r.MaxTokens,
		LogitBias:      r.LogitBias,
		Goinfer:        r.Goinfer,
		ResponseFormat: rf,
	}
	if len(r.StopSequences) > 0 {
		b, _ := json.Marshal(r.StopSequences)
		sm.Stop = b
	}
	return sm, nil
}

// ApplyFast disables thinking on r for a -fast alias request.
func (r *Req) ApplyFast() {
	r.Thinking = &struct {
		Type         string `json:"type"`
		BudgetTokens int    `json:"budget_tokens"`
	}{Type: "disabled"}
}

// Message is one message in an Anthropic conversation.
type Message struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"`
}

// Tool is an Anthropic tool declaration with input_schema.
type Tool struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	InputSchema json.RawMessage `json:"input_schema"`
}

// Block is the union of Anthropic content-block shapes read across system and message content.
type Block struct {
	Type      string          `json:"type"`
	Text      string          `json:"text"`
	Thinking  string          `json:"thinking"`
	Signature string          `json:"signature"`
	Data      string          `json:"data"`
	ID        string          `json:"id"`
	Name      string          `json:"name"`
	Input     json.RawMessage `json:"input"`
	ToolUseID string          `json:"tool_use_id"`
	Content   json.RawMessage `json:"content"`
	Source    *struct {
		Type      string `json:"type"`
		MediaType string `json:"media_type"`
		Data      string `json:"data"`
	} `json:"source"`
}

// APIErr carries an Anthropic HTTP status code, error kind, and message.
type APIErr struct {
	Code int
	Kind string
	Msg  string
}

func (e *APIErr) Error() string               { return e.Msg }
func (e *APIErr) Write(w http.ResponseWriter) { WriteErr(w, e.Code, e.Kind, e.Msg) }

// BadReq constructs a 400 invalid_request_error APIErr.
func BadReq(format string, a ...any) *APIErr {
	return &APIErr{Code: http.StatusBadRequest, Kind: "invalid_request_error", Msg: fmt.Sprintf(format, a...)}
}

// WriteErr writes an Anthropic-shaped JSON error response.
func WriteErr(w http.ResponseWriter, code int, kind, msg string) {
	netguard.WriteJSON(w, code, map[string]any{
		"type":  "error",
		"error": map[string]any{"type": kind, "message": msg},
	})
}

// DecodeJSON decodes r.Body into v and writes an Anthropic 400/413 error without leaking Go type names on failure.
func DecodeJSON(w http.ResponseWriter, r *http.Request, v any) bool {
	if err := json.NewDecoder(r.Body).Decode(v); err != nil {
		msg, tooLarge := netguard.JSONDecodeMessage(err)
		if tooLarge {
			WriteErr(w, http.StatusRequestEntityTooLarge, "invalid_request_error", msg)
			return false
		}
		WriteErr(w, http.StatusBadRequest, "invalid_request_error", msg)
		return false
	}
	return true
}
