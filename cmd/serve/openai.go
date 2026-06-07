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
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

const defaultMaxTokens = 512

type server struct {
	tk      *tokenizer.Tokenizer
	model   *decoder.Model
	tmpl    *chat.Template // nil → raw completion
	stopIDs []int          // turn-stop token ids from the template
	eosIDs  []int
	vocab   int
	modelID string
	mu      sync.Mutex // serialize generations (one model, shared compute)
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
	Model    string        `json:"model"`
	Messages []chatMessage `json:"messages"`
	Stream   bool          `json:"stream"`
	sampling
}

type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
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

// handleModels reports the single served model (Open WebUI calls this to populate
// its model picker).
func (s *server) handleModels(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{
		"object": "list",
		"data": []map[string]any{
			{"id": s.modelID, "object": "model", "created": time.Now().Unix(), "owned_by": "goinfer"},
		},
	})
}

func (s *server) handleChat(w http.ResponseWriter, r *http.Request) {
	var req chatReq
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeErr(w, http.StatusBadRequest, "invalid request body: "+err.Error())
		return
	}
	gr, err := s.prepare(req.sampling, s.chatPrompt(req.Messages))
	if err != nil {
		writeErr(w, http.StatusBadRequest, err.Error())
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	id := "chatcmpl-" + reqID()
	created := time.Now().Unix()

	if req.Stream {
		f, ok := sseStart(w)
		if !ok {
			return
		}
		role := chatChunk(id, created, s.modelID, delta{Role: "assistant"}, nil)
		sseSend(w, f, role)
		finish, _, _ := s.drive(r.Context(), gr, func(t string) {
			sseSend(w, f, chatChunk(id, created, s.modelID, delta{Content: t}, nil))
		})
		sseSend(w, f, chatChunk(id, created, s.modelID, delta{}, &finish))
		sseDone(w, f)
		return
	}

	var sb strings.Builder
	finish, nComp, lps := s.drive(r.Context(), gr, func(t string) { sb.WriteString(t) })
	choice := map[string]any{
		"index":         0,
		"message":       chatMessage{Role: "assistant", Content: sb.String()},
		"finish_reason": finish,
	}
	if req.Logprobs {
		choice["logprobs"] = s.logprobs(lps)
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"id": id, "object": "chat.completion", "created": created, "model": s.modelID,
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
	prompt := firstString(req.Prompt)
	ids, err := s.tk.Encode(prompt, true) // raw completion: tokenizer adds BOS
	if err != nil {
		writeErr(w, http.StatusInternalServerError, "encode: "+err.Error())
		return
	}
	gr, err := s.prepare(req.sampling, ids)
	if err != nil {
		writeErr(w, http.StatusBadRequest, err.Error())
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	id := "cmpl-" + reqID()
	created := time.Now().Unix()

	if req.Stream {
		f, ok := sseStart(w)
		if !ok {
			return
		}
		finish, _, _ := s.drive(r.Context(), gr, func(t string) {
			sseSend(w, f, completionChunk(id, created, s.modelID, t, nil))
		})
		sseSend(w, f, completionChunk(id, created, s.modelID, "", &finish))
		sseDone(w, f)
		return
	}
	var sb strings.Builder
	finish, nComp, _ := s.drive(r.Context(), gr, func(t string) { sb.WriteString(t) })
	writeJSON(w, http.StatusOK, map[string]any{
		"id": id, "object": "text_completion", "created": created, "model": s.modelID,
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
func (s *server) prepare(sm sampling, promptIDs []int) (genRequest, error) {
	sp := decoder.SamplingParams{
		Temperature: deref(sm.Temperature, 1.0),
		Seed:        deref(sm.Seed, 0),
		StopIDs:     s.stopIDs,
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
		eos := append(append([]int(nil), s.eosIDs...), s.stopIDs...)
		m := constrain.NewMasker(g, constrain.TokenBytes(s.vocab, s.tk.TokenText), eos).StopWhenComplete()
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

// chatPrompt renders the messages into the model's chat template (or a raw
// fallback) and encodes them. The system message is passed separately.
func (s *server) chatPrompt(msgs []chatMessage) []int {
	var system string
	var turns []chat.Turn
	for _, m := range msgs {
		switch m.Role {
		case "system":
			system = m.Content
		case "assistant":
			turns = append(turns, chat.Turn{Role: "assistant", Content: m.Content})
		default:
			turns = append(turns, chat.Turn{Role: "user", Content: m.Content})
		}
	}
	var prompt string
	addBOS := true
	if s.tmpl != nil {
		prompt = s.tmpl.Render(system, turns) // includes the family BOS marker
		addBOS = false
	} else {
		if system != "" {
			prompt = system + "\n\n"
		}
		for _, t := range turns {
			prompt += t.Content + "\n"
		}
	}
	ids, _ := s.tk.Encode(prompt, addBOS)
	return ids
}

// drive runs the generation, applying stop strings and UTF-8 holdback, calling
// onText with each newly-completed text fragment. Returns the finish reason
// ("stop" | "length"), the completion token count, and (non-stream) per-token
// logprobs. The context is cancelled on a stop-string hit to end generation.
func (s *server) drive(parent context.Context, gr genRequest, onText func(string)) (string, int, []decoder.SampleInfo) {
	ctx, cancel := context.WithCancel(parent)
	defer cancel()
	stream, gen := s.model.Generate(ctx, gr.promptIDs, gr.maxTokens, gr.sp)

	var ids []int
	printed := 0
	finish := ""
	stopping := false
	for id := range stream {
		if stopping {
			continue // drain so the generation goroutine exits cleanly
		}
		ids = append(ids, id)
		text, _ := s.tk.Decode(ids)
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
		if text, _ := s.tk.Decode(ids); len(text) > printed {
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
func (s *server) logprobs(lps []decoder.SampleInfo) map[string]any {
	content := make([]any, 0, len(lps))
	for _, lp := range lps {
		top := make([]any, 0, len(lp.Top))
		for _, t := range lp.Top {
			top = append(top, map[string]any{"token": s.tokenText(t.ID), "logprob": t.Logprob})
		}
		content = append(content, map[string]any{
			"token": s.tokenText(lp.ID), "logprob": lp.Logprob, "top_logprobs": top,
		})
	}
	return map[string]any{"content": content}
}

func (s *server) tokenText(id int) string { return string(s.tk.TokenText(id)) }
