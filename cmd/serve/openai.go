package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/townsendmerino/aikit/embed"
	"github.com/townsendmerino/aikit/encoder"
	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/constrain"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

const defaultMaxTokens = 512

// loadedModel is one resident generative model and the per-model state a request
// needs: its tokenizer, chat template, stop ids, vocab, warm-KV sessions, and the
// mutex that serializes its generations (one model = one shared compute stream).
// The server holds exactly one today; multi-model serving (Track B Inc2) makes it
// a registry keyed by served name, each with its own mutex so distinct models run
// in parallel.
type loadedModel struct {
	tk       *tokenizer.Tokenizer
	model    *decoder.Model
	tmpl     *chat.Template // nil → raw completion
	stopIDs  []int          // turn-stop token ids from the template
	eosIDs   []int
	vocab    int
	name     string      // served id (reported by /v1/models, matched on the request model field)
	fp       string      // model fingerprint (binds --session-dir snapshots)
	sessions *sessionLRU // prefix-keyed KV reuse across requests
	mu       sync.Mutex  // serialize this model's generations
}

type server struct {
	// Generative (decoder) half — nil when only an embedding model is served.
	gen *loadedModel

	// Embedding (encoder) half — nil when only a generative model is served.
	// The encoder is goroutine-safe for concurrent Encode, so /v1/embeddings is
	// served without a mutex (the per-model mutex guards only the shared decoder).
	embed    encoder.Encoder
	embedTok *embed.Tokenizer // counts tokens for usage.prompt_tokens
	embedID  string
	embedDim int
}

// --- OpenAI request shapes (the subset we honor) ---

type sampling struct {
	Temperature      *float64        `json:"temperature"`
	TopP             *float64        `json:"top_p"`
	TopK             *int            `json:"top_k"` // extension (not in the OpenAI API)
	MaxTokens        *int            `json:"max_tokens"`
	Seed             *int64          `json:"seed"`
	FrequencyPenalty *float64        `json:"frequency_penalty"`
	PresencePenalty  *float64        `json:"presence_penalty"`
	Stop             json.RawMessage `json:"stop"` // string | []string
	Logprobs         bool            `json:"logprobs"`
	TopLogprobs      *int            `json:"top_logprobs"`
	ResponseFormat   *respFormat     `json:"response_format"`
}

type respFormat struct {
	Type       string `json:"type"` // "text" | "json_object" | "json_schema"
	JSONSchema *struct {
		Name   string          `json:"name"`
		Schema json.RawMessage `json:"schema"`
	} `json:"json_schema"`
}

type chatReq struct {
	Model      string          `json:"model"`
	Messages   []chatMessage   `json:"messages"`
	Stream     bool            `json:"stream"`
	Tools      []toolSpec      `json:"tools"`
	ToolChoice json.RawMessage `json:"tool_choice"` // "auto"|"none"|{"type":"function","function":{"name":…}}
	sampling
}

type chatMessage struct {
	Role       string        `json:"role"`
	Content    string        `json:"content"`
	Name       string        `json:"name,omitempty"`         // tool messages: function name
	ToolCallID string        `json:"tool_call_id,omitempty"` // tool messages: id answered
	ToolCalls  []apiToolCall `json:"tool_calls,omitempty"`   // assistant messages
}

// toolSpec is an OpenAI tool definition (we honor type:"function").
type toolSpec struct {
	Type     string `json:"type"`
	Function struct {
		Name        string          `json:"name"`
		Description string          `json:"description"`
		Parameters  json.RawMessage `json:"parameters"`
	} `json:"function"`
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
	sampling
}

// --- response shapes ---

type usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// handleModels reports the served model(s) — the decoder and/or the embedding
// model (Open WebUI calls this to populate its model picker).
func (s *server) handleModels(w http.ResponseWriter, _ *http.Request) {
	created := time.Now().Unix()
	data := []map[string]any{}
	if s.gen != nil {
		data = append(data, map[string]any{"id": s.gen.name, "object": "model", "created": created, "owned_by": "goinfer"})
	}
	if s.embed != nil {
		data = append(data, map[string]any{"id": s.embedID, "object": "model", "created": created, "owned_by": "goinfer"})
	}
	writeJSON(w, http.StatusOK, map[string]any{"object": "list", "data": data})
}

func (s *server) handleChat(w http.ResponseWriter, r *http.Request) {
	var req chatReq
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeErr(w, http.StatusBadRequest, "invalid request body: "+err.Error())
		return
	}
	if len(req.Tools) > 0 && toolChoiceMode(req.ToolChoice) != "none" {
		s.handleChatTools(w, r, req)
		return
	}
	lm := s.gen
	gr, err := lm.prepare(req.sampling, lm.chatPrompt(req.Messages))
	if err != nil {
		writeErr(w, http.StatusBadRequest, err.Error())
		return
	}
	lm.mu.Lock()
	defer lm.mu.Unlock()
	id := "chatcmpl-" + reqID()
	created := time.Now().Unix()

	if req.Stream {
		f, ok := sseStart(w)
		if !ok {
			return
		}
		role := chatChunk(id, created, lm.name, delta{Role: "assistant"}, nil)
		sseSend(w, f, role)
		finish, _, _ := lm.drive(r.Context(), gr, func(t string) {
			sseSend(w, f, chatChunk(id, created, lm.name, delta{Content: t}, nil))
		})
		sseSend(w, f, chatChunk(id, created, lm.name, delta{}, &finish))
		sseDone(w, f)
		return
	}

	var sb strings.Builder
	finish, nComp, lps := lm.drive(r.Context(), gr, func(t string) { sb.WriteString(t) })
	choice := map[string]any{
		"index":         0,
		"message":       chatMessage{Role: "assistant", Content: sb.String()},
		"finish_reason": finish,
	}
	if req.Logprobs {
		choice["logprobs"] = lm.logprobs(lps)
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"id": id, "object": "chat.completion", "created": created, "model": lm.name,
		"choices": []any{choice},
		"usage":   usage{len(gr.promptIDs), nComp, len(gr.promptIDs) + nComp},
	})
}

func (s *server) handleCompletions(w http.ResponseWriter, r *http.Request) {
	var req completionReq
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeErr(w, http.StatusBadRequest, "invalid request body: "+err.Error())
		return
	}
	lm := s.gen
	prompt := firstString(req.Prompt)
	ids, err := lm.tk.Encode(prompt, true) // raw completion: tokenizer adds BOS
	if err != nil {
		writeErr(w, http.StatusInternalServerError, "encode: "+err.Error())
		return
	}
	gr, err := lm.prepare(req.sampling, ids)
	if err != nil {
		writeErr(w, http.StatusBadRequest, err.Error())
		return
	}
	lm.mu.Lock()
	defer lm.mu.Unlock()
	id := "cmpl-" + reqID()
	created := time.Now().Unix()

	if req.Stream {
		f, ok := sseStart(w)
		if !ok {
			return
		}
		finish, _, _ := lm.drive(r.Context(), gr, func(t string) {
			sseSend(w, f, completionChunk(id, created, lm.name, t, nil))
		})
		sseSend(w, f, completionChunk(id, created, lm.name, "", &finish))
		sseDone(w, f)
		return
	}
	var sb strings.Builder
	finish, nComp, _ := lm.drive(r.Context(), gr, func(t string) { sb.WriteString(t) })
	writeJSON(w, http.StatusOK, map[string]any{
		"id": id, "object": "text_completion", "created": created, "model": lm.name,
		"choices": []any{map[string]any{"index": 0, "text": sb.String(), "finish_reason": finish}},
		"usage":   usage{len(gr.promptIDs), nComp, len(gr.promptIDs) + nComp},
	})
}

// genRequest is a prepared generation: prompt ids, sampling params (with the
// constraint masker already wired into LogitProcessor), limits, and stop strings.
type genRequest struct {
	promptIDs   []int
	sp          decoder.SamplingParams
	maxTokens   int
	stopStrings []string
}

// prepare translates the OpenAI sampling fields into goinfer's SamplingParams,
// wires response_format into a constraint masker, and resolves stop strings.
func (lm *loadedModel) prepare(sm sampling, promptIDs []int) (genRequest, error) {
	sp := decoder.SamplingParams{
		Temperature: deref(sm.Temperature, 1.0),
		Seed:        deref(sm.Seed, 0),
		StopIDs:     lm.stopIDs,
		Logprobs:    sm.Logprobs,
		TopLogprobs: deref(sm.TopLogprobs, 0),
	}
	if sm.TopP != nil && *sm.TopP < 1 {
		sp.TopP = *sm.TopP
	}
	if sm.TopK != nil {
		sp.TopK = *sm.TopK
	}
	if sm.FrequencyPenalty != nil {
		sp.FrequencyPenalty = *sm.FrequencyPenalty
	}
	if sm.PresencePenalty != nil {
		sp.PresencePenalty = *sm.PresencePenalty
	}
	gr := genRequest{
		promptIDs:   promptIDs,
		sp:          sp,
		maxTokens:   deref(sm.MaxTokens, defaultMaxTokens),
		stopStrings: parseStop(sm.Stop),
	}
	g, err := grammarFor(sm.ResponseFormat)
	if err != nil {
		return genRequest{}, err
	}
	if g != nil {
		eos := append(append([]int(nil), lm.eosIDs...), lm.stopIDs...)
		m := constrain.NewMasker(g, constrain.TokenBytes(lm.vocab, lm.tk.TokenText), eos).StopWhenComplete()
		gr.sp.LogitProcessor = m.Process
	}
	return gr, nil
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

// messagesToTurns maps OpenAI messages to chat turns, carrying tool history
// (assistant tool_calls and tool-result turns). The system message is returned
// separately (families place it differently).
func messagesToTurns(msgs []chatMessage) (string, []chat.Turn) {
	var system string
	var turns []chat.Turn
	for _, m := range msgs {
		switch m.Role {
		case "system":
			system = m.Content
		case "tool":
			turns = append(turns, chat.Turn{Role: "tool", Content: m.Content, ToolName: m.Name, ToolCallID: m.ToolCallID})
		case "assistant":
			var tc []chat.ToolCall
			for _, c := range m.ToolCalls {
				tc = append(tc, chat.ToolCall{ID: c.ID, Name: c.Function.Name, Arguments: json.RawMessage(c.Function.Arguments)})
			}
			turns = append(turns, chat.Turn{Role: "assistant", Content: m.Content, ToolCalls: tc})
		default:
			turns = append(turns, chat.Turn{Role: "user", Content: m.Content})
		}
	}
	return system, turns
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

// encode tokenizes a rendered prompt; rendered templates already include the
// family BOS marker, so only the raw fallback asks the tokenizer to add one.
func (lm *loadedModel) encode(prompt string) []int {
	ids, _ := lm.tk.Encode(prompt, lm.tmpl == nil)
	return ids
}

// chatPrompt renders system + messages into the model's chat template (no tools).
func (lm *loadedModel) chatPrompt(msgs []chatMessage) []int {
	system, turns := messagesToTurns(msgs)
	if lm.tmpl != nil {
		return lm.encode(lm.tmpl.Render(system, turns))
	}
	return lm.encode(rawPrompt(system, turns))
}

// drive runs the generation, applying stop strings and UTF-8 holdback, calling
// onText with each newly-completed text fragment. Returns the finish reason
// ("stop" | "length"), the completion token count, and (non-stream) per-token
// logprobs. The context is cancelled on a stop-string hit to end generation.
func (lm *loadedModel) drive(parent context.Context, gr genRequest, onText func(string)) (string, int, []decoder.SampleInfo) {
	ctx, cancel := context.WithCancel(parent)
	defer cancel()
	// Reuse the KV of whichever cached session already holds this prompt as a
	// prefix (continuing chat / agent loop): only the new suffix is prefilled.
	sess := lm.sessions.acquire(gr.promptIDs)
	stream, gen := sess.Generate(ctx, gr.promptIDs, gr.maxTokens, gr.sp)

	var ids []int
	printed := 0
	finish := ""
	stopping := false
	for id := range stream {
		if stopping {
			continue // drain so the generation goroutine exits cleanly
		}
		ids = append(ids, id)
		text, _ := lm.tk.Decode(ids)
		if cut, hit := firstStop(text, gr.stopStrings); hit {
			if cut > printed {
				onText(text[printed:cut])
			}
			finish, stopping = "stop", true
			cancel()
			continue
		}
		if end := completeUTF8(text); end > printed {
			onText(text[printed:end])
			printed = end
		}
	}
	if !stopping { // flush any held-back trailing bytes
		if text, _ := lm.tk.Decode(ids); len(text) > printed {
			onText(text[printed:])
		}
		if len(ids) >= gr.maxTokens {
			finish = "length"
		} else {
			finish = "stop" // EOS / turn-stop
		}
	}
	return finish, len(ids), gen.Logprobs
}

// logprobs maps goinfer's per-token SampleInfo to the OpenAI chat logprobs shape.
func (lm *loadedModel) logprobs(lps []decoder.SampleInfo) map[string]any {
	content := make([]any, 0, len(lps))
	for _, lp := range lps {
		top := make([]any, 0, len(lp.Top))
		for _, t := range lp.Top {
			top = append(top, map[string]any{"token": lm.tokenText(t.ID), "logprob": t.Logprob})
		}
		content = append(content, map[string]any{
			"token": lm.tokenText(lp.ID), "logprob": lp.Logprob, "top_logprobs": top,
		})
	}
	return map[string]any{"content": content}
}

func (lm *loadedModel) tokenText(id int) string { return string(lm.tk.TokenText(id)) }
