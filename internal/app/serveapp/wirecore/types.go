package wirecore

import (
	"encoding/json"
	"fmt"
	"maps"
	"math"
	"net/http"
	"strconv"
	"strings"

	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/constrain"
	"github.com/townsendmerino/goinfer/decoder/config"
	"github.com/townsendmerino/goinfer/engine"
	"github.com/townsendmerino/goinfer/internal/app/serveapp/carriedids"
	"github.com/townsendmerino/goinfer/internal/app/serveapp/jobs"
	"github.com/townsendmerino/goinfer/internal/app/serveapp/media"
	"github.com/townsendmerino/goinfer/internal/app/serveapp/streamframe"
)

// Per-request count bounds against per-step amplifiers.
const (
	MaxStopSequences    = 16
	MaxToolsPerRequest  = 128
	MaxLogitBiasEntries = 1024
)

// CheckToolCount rejects requests declaring more than MaxToolsPerRequest tools.
func CheckToolCount(n int) error {
	if n > MaxToolsPerRequest {
		return fmt.Errorf("a request may declare at most %d tools (got %d)", MaxToolsPerRequest, n)
	}
	return nil
}

// Sampling holds the OpenAI-shaped sampling and output-format controls shared across HTTP routes.
type Sampling struct {
	Temperature         *float64            `json:"temperature"`
	TopP                *float64            `json:"top_p"`
	TopK                *int                `json:"top_k"`
	MaxTokens           *int                `json:"max_tokens"`
	MaxCompletionTokens *int                `json:"max_completion_tokens"`
	Seed                *int64              `json:"seed"`
	FrequencyPenalty    *float64            `json:"frequency_penalty"`
	PresencePenalty     *float64            `json:"presence_penalty"`
	Stop                json.RawMessage     `json:"stop"`
	Logprobs            bool                `json:"logprobs"`
	TopLogprobs         *int                `json:"top_logprobs"`
	LogitBias           map[string]float64  `json:"logit_bias"`
	ResponseFormat      *RespFormat         `json:"response_format"`
	Confidence          bool                `json:"goinfer_confidence"`
	ConfidenceOK        bool                `json:"-"`
	Goinfer             *jobs.VendorOptions `json:"goinfer"`
	MinP                *float64            `json:"-"`
	FixedTokens         func() int          `json:"-"`
	OutputByWindow      bool                `json:"-"`
}

// ConstrainsOutput reports whether sm's response_format constrains output from the first token.
func (sm *Sampling) ConstrainsOutput() bool {
	return sm.ResponseFormat != nil && (sm.ResponseFormat.Type == "json_object" || sm.ResponseFormat.Type == "json_schema")
}

// WithRecommended fills unset sampling fields from a model's recommended sampling defaults, leaving temperature 0 untouched.
func WithRecommended(sm Sampling, rs *config.RecommendedSampling) (Sampling, []string) {
	if rs == nil || (sm.Temperature != nil && *sm.Temperature == 0) {
		return sm, nil
	}
	var filled []string
	fill := func(name string, dst **float64, src *float64) {
		if *dst == nil && src != nil {
			v := *src
			*dst = &v
			filled = append(filled, name)
		}
	}
	fill("temperature", &sm.Temperature, rs.Temperature)
	fill("top_p", &sm.TopP, rs.TopP)
	if sm.TopK == nil && rs.TopK != nil {
		k := *rs.TopK
		sm.TopK = &k
		filled = append(filled, "top_k")
	}
	fill("min_p", &sm.MinP, rs.MinP)
	fill("presence_penalty", &sm.PresencePenalty, rs.PresencePenalty)
	return sm, filled
}

// RespFormat is the OpenAI response_format specification.
type RespFormat struct {
	Type       string `json:"type"`
	JSONSchema *struct {
		Name   string          `json:"name"`
		Schema json.RawMessage `json:"schema"`
	} `json:"json_schema"`
}

// GrammarFor compiles rf into a constraint grammar, returning nil when unconstrained.
func GrammarFor(rf *RespFormat) (constrain.Grammar, error) {
	if rf == nil {
		return nil, nil
	}
	switch rf.Type {
	case "", "text":
		return nil, nil
	case "json_object":
		return constrain.JSON(), nil
	case "json_schema":
		if rf.JSONSchema == nil || len(rf.JSONSchema.Schema) == 0 {
			return nil, fmt.Errorf("response_format json_schema requires a schema")
		}
		return constrain.JSONSchema(rf.JSONSchema.Schema)
	default:
		return nil, fmt.Errorf("unsupported response_format type %q", rf.Type)
	}
}

// StreamOptions holds OpenAI's stream_options object.
type StreamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

// Delta is one OpenAI chat.completion.chunk choice delta.
type Delta struct {
	Role             string `json:"role,omitempty"`
	Content          string `json:"content,omitempty"`
	ReasoningContent string `json:"reasoning_content,omitempty"`
}

// ChatReq is an OpenAI /v1/chat/completions request body.
type ChatReq struct {
	Model               string                     `json:"model"`
	Messages            []ChatMessage              `json:"messages"`
	Stream              bool                       `json:"stream"`
	StreamOptions       *StreamOptions             `json:"stream_options"`
	Tools               []ToolSpec                 `json:"tools"`
	ToolChoice          json.RawMessage            `json:"tool_choice"`
	N                   *int                       `json:"n"`
	ChatTemplateKwargs  map[string]json.RawMessage `json:"chat_template_kwargs"`
	ReasoningEffort     string                     `json:"reasoning_effort"`
	ReasoningFormat     string                     `json:"reasoning_format"`
	ThinkingTokenBudget *int                       `json:"thinking_token_budget"`
	Sampling
}

// Think returns the request's thinking controls for streamframe.ResolveThink.
func (r *ChatReq) Think() streamframe.ThinkRequest {
	tr := streamframe.ThinkRequest{Kwargs: r.ChatTemplateKwargs, ReasoningEffort: r.ReasoningEffort, Format: r.ReasoningFormat}
	if r.ThinkingTokenBudget != nil {
		tr.Budget = *r.ThinkingTokenBudget
	}
	return tr
}

// ApplyFast disables thinking on r for a -fast alias request without mutating any caller-held map in place.
func (r *ChatReq) ApplyFast() {
	if r.ChatTemplateKwargs == nil {
		r.ChatTemplateKwargs = map[string]json.RawMessage{}
	} else {
		m := make(map[string]json.RawMessage, len(r.ChatTemplateKwargs)+1)
		maps.Copy(m, r.ChatTemplateKwargs)
		r.ChatTemplateKwargs = m
	}
	r.ChatTemplateKwargs["enable_thinking"] = json.RawMessage("false")
	r.ThinkingTokenBudget = nil
}

// ChatMessage is an OpenAI wire message.
type ChatMessage struct {
	Role             string              `json:"role"`
	Content          json.RawMessage     `json:"content"`
	Name             string              `json:"name,omitempty"`
	ToolCallID       string              `json:"tool_call_id,omitempty"`
	ToolCalls        []APIToolCall       `json:"tool_calls,omitempty"`
	ReasoningContent string              `json:"reasoning_content,omitempty"`
	Reasoning        string              `json:"reasoning,omitempty"`
	Carried          *carriedids.Carrier `json:"-"`
}

// Text returns m's plain string content or the concatenated text parts of a content array.
func (m ChatMessage) Text() string { return media.ContentPartsText(m.Content) }

// ImageData decodes any inline base64 data-URI media parts in m.Content.
func (m ChatMessage) ImageData() ([]media.ImageRef, error) {
	return media.ContentPartsImages(m.Content)
}

// RawStr marshals s as a JSON string value.
func RawStr(s string) json.RawMessage { b, _ := json.Marshal(s); return b }

// ToolSpec decodes both Chat Completions nested function tools and Responses flat function tools.
type ToolSpec struct {
	Type     string `json:"type"`
	Function struct {
		Name        string          `json:"name"`
		Description string          `json:"description"`
		Parameters  json.RawMessage `json:"parameters"`
	} `json:"function"`
}

// UnmarshalJSON accepts both nested {"type":"function","function":{...}} and flat {"type":"function","name":...} tool shapes.
func (t *ToolSpec) UnmarshalJSON(b []byte) error {
	type nested ToolSpec
	var w struct {
		nested
		Name        string          `json:"name"`
		Description string          `json:"description"`
		Parameters  json.RawMessage `json:"parameters"`
	}
	if err := json.Unmarshal(b, &w); err != nil {
		return err
	}
	*t = ToolSpec(w.nested)
	if t.Function.Name == "" && w.Name != "" {
		t.Function.Name, t.Function.Description, t.Function.Parameters = w.Name, w.Description, w.Parameters
	}
	return nil
}

// FunctionTools partitions tools into runnable function tools and dropped tool descriptors.
func FunctionTools(tools []ToolSpec) (kept []ToolSpec, dropped []string) {
	for _, t := range tools {
		if (t.Type == "" || t.Type == "function") && t.Function.Name != "" {
			kept = append(kept, t)
			continue
		}
		d := t.Type
		if t.Function.Name != "" {
			d += ":" + t.Function.Name
		}
		dropped = append(dropped, d)
	}
	return kept, dropped
}

// DropUnrunnableTools filters tools to runnable function tools and records any dropped entries on X-Goinfer-Tools-Dropped.
func DropUnrunnableTools(w http.ResponseWriter, tools []ToolSpec) []ToolSpec {
	kept, dropped := FunctionTools(tools)
	if len(dropped) > 0 {
		w.Header().Set("X-Goinfer-Tools-Dropped", strings.Join(dropped, ","))
	}
	return kept
}

// APIToolCall is the OpenAI wire representation of a tool call.
type APIToolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

// CompletionReq is a legacy OpenAI /v1/completions request body.
type CompletionReq struct {
	Model         string          `json:"model"`
	Prompt        json.RawMessage `json:"prompt"`
	Stream        bool            `json:"stream"`
	Logprobs      *int            `json:"logprobs"`
	StreamOptions *StreamOptions  `json:"stream_options"`
	N             *int            `json:"n"`
	Sampling
}

// PrefillReuse summarizes how many leading prompt tokens skipped prefill and from which cache tier.
type PrefillReuse struct {
	Tokens, Shared, Host, Promoted int
}

// ReuseOf extracts PrefillReuse counters from rep.
func ReuseOf(rep *engine.Report) PrefillReuse {
	if rep == nil {
		return PrefillReuse{}
	}
	return PrefillReuse{Tokens: rep.PrefillReused, Shared: rep.PrefillShared, Host: rep.KVSnapRestored + rep.KVPromotedRows, Promoted: rep.KVPromotedBlocks}
}

// Sources formats r as the vendor prefill_reuse object, or nil when no tokens were reused.
func (r PrefillReuse) Sources() *jobs.PrefillReuseSources {
	if r.Tokens <= 0 {
		return nil
	}
	return &jobs.PrefillReuseSources{OwnTokens: max(r.Tokens-r.Shared-r.Host, 0), SharedTokens: r.Shared, HostTokens: r.Host, PromotedBlocks: r.Promoted}
}

// NewUsage builds an OpenAI usage block with prefill-reuse accounting.
func NewUsage(prompt, nComp int, r PrefillReuse) jobs.Usage {
	return jobs.Usage{PromptTokens: prompt, CompletionTokens: nComp, TotalTokens: prompt + nComp, PrefillReusedTokens: r.Tokens, PrefillReuse: r.Sources()}
}

// ValidateN rejects an explicit n != 1.
func ValidateN(n *int) error {
	if n != nil && *n != 1 {
		return fmt.Errorf("n=%d is not supported; this server always returns exactly one choice", *n)
	}
	return nil
}

// ConfResult holds per-field confidences or the error explaining why none were captured.
type ConfResult struct {
	Fields []constrain.FieldConfidence
	Err    error
}

// Payload formats c for the goinfer_confidence response field.
func (c *ConfResult) Payload() any {
	if c.Err != nil {
		return map[string]any{"error": c.Err.Error()}
	}
	if c.Fields == nil {
		return []constrain.FieldConfidence{}
	}
	return c.Fields
}

// MessagesToTurns converts OpenAI wire messages into a system prompt and chat.Turn slice.
func MessagesToTurns(msgs []ChatMessage) (string, []chat.Turn) {
	system, turns, _ := MessagesToTurnsCarried(msgs)
	return system, turns
}

// MessagesToTurnsCarried converts OpenAI wire messages into a system prompt, chat.Turn slice, and per-turn carried-id map.
func MessagesToTurnsCarried(msgs []ChatMessage) (string, []chat.Turn, map[int]*carriedids.Carrier) {
	var system string
	var carriers map[int]*carriedids.Carrier
	var turns []chat.Turn
	for _, m := range msgs {
		switch m.Role {
		case "system", "developer":
			text := m.Text()
			switch {
			case text == "":
			case len(turns) > 0:
				turns = append(turns, chat.Turn{Role: "user", Content: text, ToolLoop: true})
			case system == "":
				system = text
			default:
				system += "\n\n" + text
			}
		case "tool":
			turns = append(turns, chat.Turn{Role: "tool", Content: m.Text(), ToolName: m.Name, ToolCallID: m.ToolCallID})
		case "assistant":
			var tc []chat.ToolCall
			for _, c := range m.ToolCalls {
				tc = append(tc, chat.ToolCall{ID: c.ID, Name: c.Function.Name, Arguments: json.RawMessage(c.Function.Arguments)})
			}
			reasoning := m.ReasoningContent
			if reasoning == "" {
				reasoning = m.Reasoning
			}
			if m.Carried != nil {
				if carriers == nil {
					carriers = map[int]*carriedids.Carrier{}
				}
				carriers[len(turns)] = m.Carried
			}
			turns = append(turns, chat.Turn{Role: "assistant", Content: m.Text(), ToolCalls: tc, Reasoning: reasoning})
		default:
			turns = append(turns, chat.Turn{Role: "user", Content: m.Text()})
		}
	}
	return system, turns, carriers
}

// RawPrompt formats system and turns as plain newline-delimited text for models without a chat template.
func RawPrompt(system string, turns []chat.Turn) string {
	var b strings.Builder
	if system != "" {
		b.WriteString(system + "\n\n")
	}
	for _, t := range turns {
		b.WriteString(t.Content + "\n")
	}
	return b.String()
}

// ParseLogitBias validates and converts an OpenAI logit_bias map into token-id float32 biases.
func ParseLogitBias(in map[string]float64, vocab int) (map[int]float32, error) {
	if len(in) > MaxLogitBiasEntries {
		return nil, fmt.Errorf("logit_bias accepts at most %d entries (got %d)", MaxLogitBiasEntries, len(in))
	}
	out := make(map[int]float32, len(in))
	for k, v := range in {
		id, err := strconv.Atoi(k)
		if err != nil || id < 0 || (vocab > 0 && id >= vocab) {
			return nil, fmt.Errorf("logit_bias key %q is not a token id in [0,%d)", k, vocab)
		}
		if math.IsNaN(v) || v < -100 || v > 100 {
			return nil, fmt.Errorf("logit_bias value for token %d must be in [-100,100] (got %v)", id, v)
		}
		out[id] = float32(v)
	}
	return out, nil
}

// ChatInputBytes sums tokenizable text and replayed tool-call bytes across msgs.
func ChatInputBytes(msgs []ChatMessage) int {
	n := 0
	for _, m := range msgs {
		n += len(m.Text())
		for _, tc := range m.ToolCalls {
			n += len(tc.Function.Name) + len(tc.Function.Arguments)
		}
	}
	return n
}

// ToolSchemaBytes sums rendered name, description, and schema bytes across tools.
func ToolSchemaBytes(tools []ToolSpec) int {
	n := 0
	for _, t := range tools {
		n += len(t.Function.Name) + len(t.Function.Description) + len(t.Function.Parameters)
	}
	return n
}

// ChatImages collects inline images and audio clips across msgs in order.
func ChatImages(msgs []ChatMessage) ([]media.ImageRef, error) {
	var out []media.ImageRef
	for _, m := range msgs {
		imgs, err := m.ImageData()
		if err != nil {
			return nil, err
		}
		out = append(out, imgs...)
	}
	return out, nil
}

// HasChatMedia reports whether any message in msgs carries an image_url or input_audio content part without decoding base64 data.
func HasChatMedia(msgs []ChatMessage) bool {
	for _, m := range msgs {
		var parts []struct {
			Type string `json:"type"`
		}
		if json.Unmarshal(m.Content, &parts) != nil {
			continue
		}
		for _, p := range parts {
			if p.Type == "image_url" || p.Type == "input_audio" {
				return true
			}
		}
	}
	return false
}
