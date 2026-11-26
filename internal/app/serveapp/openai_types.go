package serveapp

import (
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"strconv"
	"strings"

	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/constrain"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/engine"
	"github.com/townsendmerino/goinfer/internal/app/serveapp/jobs"
	samplingp "github.com/townsendmerino/goinfer/internal/runtime/sampling"
)

// --- OpenAI request shapes (the subset we honor) ---

type sampling struct {
	Temperature         *float64        `json:"temperature"`
	TopP                *float64        `json:"top_p"`
	TopK                *int            `json:"top_k"` // extension (not in the OpenAI API)
	MaxTokens           *int            `json:"max_tokens"`
	MaxCompletionTokens *int            `json:"max_completion_tokens"` // current OpenAI SDKs send this instead of max_tokens; preferred when set
	Seed                *int64          `json:"seed"`
	FrequencyPenalty    *float64        `json:"frequency_penalty"`
	PresencePenalty     *float64        `json:"presence_penalty"`
	Stop                json.RawMessage `json:"stop"` // string | []string
	Logprobs            bool            `json:"logprobs"`
	TopLogprobs         *int            `json:"top_logprobs"`
	// LogitBias is OpenAI's logit_bias: token id (as a decimal string) to a value in [-100,100] added to that token's logit
	// before sampling; -100 bans the token in practice. /v1/messages takes it as a goinfer extension.
	LogitBias      map[string]float64 `json:"logit_bias"`
	ResponseFormat *respFormat        `json:"response_format"`
	// Confidence is the goinfer_confidence vendor extension: with
	// response_format json_schema, the response carries each enum, boolean and integer field's confidence. Only the
	// routes that write it back set confidenceOK; prepare refuses the flag everywhere else, so no route can accept it
	// and silently drop the result.
	Confidence   bool `json:"goinfer_confidence"`
	confidenceOK bool
	// Goinfer is the request's "goinfer" vendor object (jobs.VendorOptions): {"cost": true} asks for the turn's cost report.
	Goinfer *jobs.VendorOptions `json:"goinfer"`
	// minP has no request field; withRecommended sets it from the model's recommended sampling.
	minP *float64
	// fixedTokens measures the request's fixed part (system prompt and tool definitions) in tokens, for a route that
	// has one. prepare calls it only when the prompt overflows, to choose the refusal's wording (overflow.go).
	fixedTokens func() int
	// outputByWindow gives a request that names no output cap connect.OutputLimit of the window in place of
	// defaultMaxTokens. /v1/responses sets it: that API has no default cap, and Codex sends none.
	outputByWindow bool
}

type respFormat struct {
	Type       string `json:"type"` // "text" | "json_object" | "json_schema"
	JSONSchema *struct {
		Name   string          `json:"name"`
		Schema json.RawMessage `json:"schema"`
	} `json:"json_schema"`
}

// constrainsOutput reports whether the request's response_format makes a grammar constrain the output from the first
// token (json_object / json_schema) — the case think.go renders thinking-off, since an open think block in the prompt
// would contradict the mask.
func (sm *sampling) constrainsOutput() bool {
	return sm.ResponseFormat != nil && (sm.ResponseFormat.Type == "json_object" || sm.ResponseFormat.Type == "json_schema")
}

type chatReq struct {
	Model    string        `json:"model"`
	Messages []chatMessage `json:"messages"`
	Stream   bool          `json:"stream"`
	// StreamOptions follows OpenAI's shape: with include_usage, a final chunk carries an empty choices array and a
	// usage object. Counting SSE chunks is no substitute for a token count: a token held back for an incomplete UTF-8
	// rune or a partial stop-string match emits no chunk, and the token that resolves the holdback emits one chunk
	// covering several tokens' bytes.
	StreamOptions *streamOptions  `json:"stream_options"`
	Tools         []toolSpec      `json:"tools"`
	ToolChoice    json.RawMessage `json:"tool_choice"` // "auto"|"none"|{"type":"function","function":{"name":…}}
	// N is OpenAI's "how many choices" field; this server generates exactly one. It is parsed (see validateN) because
	// an unrecognized JSON key is silently dropped, so n:3 would otherwise get one choice back under a 200. *int so
	// that n:0 and omitted are distinguishable from n:1.
	N *int `json:"n"`
	// Thinking controls (think.go): vLLM/llama.cpp's chat_template_kwargs.enable_thinking, OpenAI's reasoning_effort, and
	// llama.cpp's reasoning_format. Absent = the server defaults (-thinking, -reasoning-format).
	ChatTemplateKwargs map[string]json.RawMessage `json:"chat_template_kwargs"`
	ReasoningEffort    string                     `json:"reasoning_effort"`
	ReasoningFormat    string                     `json:"reasoning_format"`
	// ThinkingTokenBudget is vLLM's name for a cap on reasoning tokens; serve force-closes the block at it (budget.go).
	ThinkingTokenBudget *int `json:"thinking_token_budget"`
	sampling
}

// think is the request's thinking controls as think.go's input.
func (r *chatReq) think() thinkRequest {
	tr := thinkRequest{kwargs: r.ChatTemplateKwargs, reasoningEffort: r.ReasoningEffort, format: r.ReasoningFormat}
	if r.ThinkingTokenBudget != nil {
		tr.budget = *r.ThinkingTokenBudget
	}
	return tr
}

type chatMessage struct {
	Role       string          `json:"role"`
	Content    json.RawMessage `json:"content"`                // string | [{"type":"text"|"image_url",…}, …]
	Name       string          `json:"name,omitempty"`         // tool messages: function name
	ToolCallID string          `json:"tool_call_id,omitempty"` // tool messages: id answered
	ToolCalls  []apiToolCall   `json:"tool_calls,omitempty"`   // assistant messages
	// ReasoningContent / Reasoning are an assistant message's own reasoning, replayed by a client that kept it (the two spellings
	// llama.cpp / the DeepSeek API and vLLM use). A model with a history rule renders it the way its own template would for the
	// turns of the tool loop in progress and drops it for earlier turns (chat/history.go); any other model ignores it, as before.
	ReasoningContent string `json:"reasoning_content,omitempty"`
	Reasoning        string `json:"reasoning,omitempty"`
	// carried is the carried-id blob a Responses reasoning item gave this assistant message (carried_ids.go); nil when none.
	carried *idsCarrier
}

// text returns the message's text: the plain-string content, or the concatenated
// text parts of an OpenAI content array (image_url and other parts ignored here —
// images are pulled separately by imageData).
func (m chatMessage) text() string { return contentPartsText(m.Content) }

// imageData returns the data-URI images carried in an OpenAI content array
// (image_url parts), as decoded bytes. URL (non-data:) images return an error so
// the handler can reject them (no server-side fetch — SSRF guard).
func (m chatMessage) imageData() ([]imageRef, error) { return contentPartsImages(m.Content) }

// rawStr wraps a plain string as JSON content (for messages we construct rather
// than parse, e.g. /v1/responses building an internal message list).
func rawStr(s string) json.RawMessage { b, _ := json.Marshal(s); return b }

// toolSpec is an OpenAI tool definition (we honor type:"function").
//
// It decodes both wire shapes (H-a, docs/measurements/harness-dialects-2026-10-09.md): Chat Completions nests the
// function, `{"type":"function","function":{"name",…}}`, while the Responses API flattens it,
// `{"type":"function","name",…,"parameters"}`. /v1/responses decoded only the nested shape, so every
// Responses-shaped tool (Codex is Responses-only) arrived with an empty name and no schema.
type toolSpec struct {
	Type     string `json:"type"`
	Function struct {
		Name        string          `json:"name"`
		Description string          `json:"description"`
		Parameters  json.RawMessage `json:"parameters"`
	} `json:"function"`
}

func (t *toolSpec) UnmarshalJSON(b []byte) error {
	type nested toolSpec // no methods, so no recursion
	var w struct {
		nested
		Name        string          `json:"name"`
		Description string          `json:"description"`
		Parameters  json.RawMessage `json:"parameters"`
	}
	if err := json.Unmarshal(b, &w); err != nil {
		return err
	}
	*t = toolSpec(w.nested)
	if t.Function.Name == "" && w.Name != "" { // the Responses (flat) shape
		t.Function.Name, t.Function.Description, t.Function.Parameters = w.Name, w.Description, w.Parameters
	}
	return nil
}

// functionTools keeps the function tools and names the rest. A client may declare tool types this server cannot
// run (Responses `custom` freeform-grammar tools, `web_search`, `tool_search`, `namespace`, `local_shell`): before
// H-a each became a function tool with an empty name in the prompt. They are now left out of the prompt, and the
// handler reports them in the X-Goinfer-Tools-Dropped header so the omission is visible.
func functionTools(tools []toolSpec) (kept []toolSpec, dropped []string) {
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

// dropUnrunnableTools filters a request's tool list with functionTools, setting the header when any is dropped.
func dropUnrunnableTools(w http.ResponseWriter, tools []toolSpec) []toolSpec {
	kept, dropped := functionTools(tools)
	if len(dropped) > 0 {
		w.Header().Set("X-Goinfer-Tools-Dropped", strings.Join(dropped, ","))
	}
	return kept
}

// apiToolCall is the OpenAI wire form of a tool call (arguments is a JSON string).
type apiToolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

type completionReq struct {
	Model  string          `json:"model"`
	Prompt json.RawMessage `json:"prompt"` // string | []string
	Stream bool            `json:"stream"`
	// Logprobs shadows sampling.Logprobs (embedded below) for the /v1/completions surface. The legacy Completions API
	// types logprobs as an integer (the number of top alternatives), not the chat API's bool: the standard SDK sends
	// `logprobs: 5`, which would fail to decode into a bool. The outer (shallower) field wins during JSON decode, so
	// req.sampling.Logprobs stays false and the expensive per-token logprobs path never engages; the handler answers
	// 400 when it is set (unimplemented here).
	Logprobs *int `json:"logprobs"`
	// StreamOptions is parsed so include_usage is honored on this surface too.
	StreamOptions *streamOptions `json:"stream_options"`
	// N is rejected unless 1, as in chatReq (see validateN).
	N *int `json:"n"`
	sampling
}

// --- response shapes ---

// usage and prefillReuseSources are the generation accounting a job records (internal/app/serveapp/jobs).
type (
	usage               = jobs.Usage
	prefillReuseSources = jobs.PrefillReuseSources
)

// prefillReuse is how many leading prompt tokens a generation's prefill skipped, and from which source (engine.Report's PrefillReused,
// PrefillShared, KVSnapRestored and KVPromotedRows).
type prefillReuse struct {
	Tokens, Shared, Host, Promoted int
}

func reuseOf(rep *engine.Report) prefillReuse {
	if rep == nil {
		return prefillReuse{}
	}
	return prefillReuse{Tokens: rep.PrefillReused, Shared: rep.PrefillShared, Host: rep.KVSnapRestored + rep.KVPromotedRows, Promoted: rep.KVPromotedBlocks}
}

// sources is the vendor extension for r, or nil when nothing was reused.
func (r prefillReuse) sources() *prefillReuseSources {
	if r.Tokens <= 0 {
		return nil
	}
	return &prefillReuseSources{OwnTokens: max(r.Tokens-r.Shared-r.Host, 0), SharedTokens: r.Shared, HostTokens: r.Host, PromotedBlocks: r.Promoted}
}

// turnResult is what drive and driveVL report about one generation, which each route's wire encoder reads: the finish
// reason ("stop", "length" or "cancelled"), the completion token count, the per-token logprobs (nil with no engine
// report), the stop string that ended the turn ("" for none), the prefill reuse (zero when the path does not track it),
// and the admin-cancel reason ("" unless an admin cancel, not a client disconnect or shutdown, ended the turn).
type turnResult struct {
	finish       string
	nComp        int
	logprobs     []decoder.SampleInfo
	stopHit      string
	reused       prefillReuse
	cancelReason string
}

// newUsage is the OpenAI usage block for a generation of nComp tokens over prompt tokens, with r's reuse.
func newUsage(prompt, nComp int, r prefillReuse) usage {
	return usage{PromptTokens: prompt, CompletionTokens: nComp, TotalTokens: prompt + nComp, PrefillReusedTokens: r.Tokens, PrefillReuse: r.sources()}
}

// validateN rejects an explicit "n" other than 1: this server always generates exactly one choice, and an n:3 request
// would otherwise get one choice back under a 200 with nothing in the response naming what was ignored. n omitted
// (nil) is the default and passes, as does n:1.
func validateN(n *int) error {
	if n != nil && *n != 1 {
		return fmt.Errorf("n=%d is not supported; this server always returns exactly one choice", *n)
	}
	return nil
}

// genRequest is a prepared generation: prompt ids, sampling params (with the
// constraint masker already wired into LogitProcessor), limits, and stop strings.
type genRequest struct {
	promptIDs   []int
	sp          samplingp.SamplingParams
	maxTokens   int
	stopStrings []string
	// id, when set, registers this generation in the server's cancel-by-id registry for the duration of
	// drive/driveVL. "" (the zero value) skips registration, so a caller not wired for cancellation (or a test
	// calling drive/driveVL directly) is unaffected.
	id     string
	masker *constrain.Masker     // set on constrained requests (response_format / tool grammar); enables grammar-spec
	lazy   *constrain.LazyMasker // an auto turn's lazy tool union on a speculating server (armLazy); engine.Request.Lazy
	// conf, when the request asked for goinfer_confidence, receives the per-field confidences once streamTokens
	// has drained the generation. Its presence also keeps drive on plain constrained decode: the grammar-fused
	// speculative path drives the masker without Process, which is where the capture records.
	conf *confResult
	// think, when set, routes the decoded text through a reasoning splitter (think.go): content goes to drive's onText,
	// reasoning to think.onReasoning. nil = pass text through untouched.
	think *thinkOut
	// outcome receives what streamTokens saw of the answer (emptyAnswerOf reads it). Set by prepare.
	outcome *turnOutcome
	// boundary is how many leading ids of promptIDs are the stable system-and-tools section (loadedModel.prefixBoundary), 0 when unknown. It reaches the
	// decoder as KVScope.Boundary, where a prefix snapshot may be taken at it during the cold prefill.
	boundary int
	// keep is how many leading ids of promptIDs the next turn's prompt repeats (loadedModel.recurrentKeep), 0 when unknown or not a recurrent model
	// whose snapshots carry its state. It reaches the decoder as KVScope.Keep, where the state is kept for the next turn.
	keep int
	// msgBoundary is how many leading ids of promptIDs come before the last turn's content (loadedModel.recurrentMsgBoundary), 0 when unknown or not
	// a recurrent model whose snapshots carry its state. It reaches the decoder as KVScope.MsgBoundary.
	msgBoundary int
	// fromModel names the sampling fields prepare took from the model's recommended sampling (withRecommended).
	fromModel []string
	// callPostCheck is a must-call tool_choice on a family with no JSON call form (forcedCallEnforcement "post-check"):
	// runToolTurn streams no prose and returns errNoForcedCall when the reply holds no call.
	callPostCheck bool
	// clock is the turn's stage boundaries and, when asked, its cost counter (jobs/turn_timings.go). Set by prepare; shared by the
	// copies a genRequest value makes as it is passed along.
	clock *jobs.TurnClock
}

// kvScope is the request's KV sharing scope, of which the shell sets Boundary, the end of the system-and-tools section a
// prefix snapshot is taken at, and Keep, the prefix a recurrent model's snapshot keeps.
type kvScope = decoder.KVScope

// kvScope is the KV sharing scope the request hands the engine.
func (gr *genRequest) kvScope() kvScope {
	return kvScope{Boundary: gr.boundary, Keep: gr.keep, MsgBoundary: gr.msgBoundary}
}

// confResult is a generation's per-field confidence (or why there is none).
type confResult struct {
	fields []constrain.FieldConfidence
	err    error
}

// payload is the goinfer_confidence value a response carries: the field records (an empty list when no field
// qualified), or an object naming the error.
func (c *confResult) payload() any {
	if c.err != nil {
		return map[string]any{"error": c.err.Error()}
	}
	if c.fields == nil {
		return []constrain.FieldConfidence{}
	}
	return c.fields
}

// grammarFor maps response_format to a constraint grammar (nil = unconstrained).
func grammarFor(rf *respFormat) (constrain.Grammar, error) {
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

// messagesToTurns maps OpenAI messages to chat turns, carrying tool history (assistant tool_calls and tool-result
// turns). The system message is returned separately because families place it differently.
//
// "developer" is an alias for "system": OpenAI's newer APIs send the system prompt under that role for
// reasoning-class models, and agent harnesses follow. It is an alias and nothing more: same position, same handling,
// no new concept downstream. It is not a general unknown-role tolerance; every other unrecognized role keeps the
// default arm below, which demotes it to a user turn (for "developer" that delivered a harness's entire agent
// scaffold as the user's first message).
//
// Several system messages are all kept. Those before the conversation starts are joined, in order, into the system
// prompt: a harness may send its instructions in more than one (Codex sends two), and keeping only the last dropped
// the rest without a word. One sent after the conversation has started stays where it is, as user-side text that is
// not a new query (chat.Turn.ToolLoop), the rule /v1/messages follows (anthropicTurns): moved into the system prompt
// it would replace or reorder what the model was told first and change the prompt's prefix.
func messagesToTurns(msgs []chatMessage) (string, []chat.Turn) {
	system, turns, _ := messagesToTurnsCarried(msgs)
	return system, turns
}

// messagesToTurnsCarried is messagesToTurns that also returns, by turn index, the carried-id blob of each assistant turn that
// has one.
func messagesToTurnsCarried(msgs []chatMessage) (string, []chat.Turn, map[int]*idsCarrier) {
	var system string
	var carriers map[int]*idsCarrier
	var turns []chat.Turn
	for _, m := range msgs {
		switch m.Role {
		case "system", "developer":
			text := m.text()
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
			turns = append(turns, chat.Turn{Role: "tool", Content: m.text(), ToolName: m.Name, ToolCallID: m.ToolCallID})
		case "assistant":
			var tc []chat.ToolCall
			for _, c := range m.ToolCalls {
				tc = append(tc, chat.ToolCall{ID: c.ID, Name: c.Function.Name, Arguments: json.RawMessage(c.Function.Arguments)})
			}
			reasoning := m.ReasoningContent
			if reasoning == "" {
				reasoning = m.Reasoning
			}
			if m.carried != nil {
				if carriers == nil {
					carriers = map[int]*idsCarrier{}
				}
				carriers[len(turns)] = m.carried
			}
			turns = append(turns, chat.Turn{Role: "assistant", Content: m.text(), ToolCalls: tc, Reasoning: reasoning})
		default:
			turns = append(turns, chat.Turn{Role: "user", Content: m.text()})
		}
	}
	return system, turns, carriers
}

// rawPrompt is the unrecognized-template fallback (plain conversation text).
func rawPrompt(system string, turns []chat.Turn) string {
	var b strings.Builder
	if system != "" {
		b.WriteString(system + "\n\n")
	}
	for _, t := range turns {
		b.WriteString(t.Content + "\n")
	}
	return b.String()
}

// parseLogitBias validates a request's logit_bias and converts it to the sampler's form. A key must be a token id inside
// the vocabulary (vocab 0: unknown, so only the sign is checked) and a value must lie in [-100,100], OpenAI's range; anything
// else is a 400, so a typo cannot be silently ignored.
func parseLogitBias(in map[string]float64, vocab int) (map[int]float32, error) {
	if len(in) > maxLogitBiasEntries {
		return nil, fmt.Errorf("logit_bias accepts at most %d entries (got %d)", maxLogitBiasEntries, len(in))
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
