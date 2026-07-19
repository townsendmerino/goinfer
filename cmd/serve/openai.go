package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/townsendmerino/aikit/embed"
	"github.com/townsendmerino/aikit/encoder"
	"github.com/townsendmerino/aikit/vision"
	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/constrain"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/multimodal"
	"github.com/townsendmerino/goinfer/tokenizer"
)

const defaultMaxTokens = 512

// loadedModel is one resident generative model and the per-model state a request
// needs: its tokenizer, chat template, stop ids, vocab, warm-KV sessions, and the
// mutex that serializes its generations (one model = one shared compute stream).
// The server registry holds N of them keyed by served name; each has its own
// mutex, so requests to distinct models run in parallel.
type loadedModel struct {
	tk       *tokenizer.Tokenizer
	model    *decoder.Model
	tmpl     *chat.Template // nil → raw completion
	stopIDs  []int          // turn-stop token ids from the template
	eosIDs   []int
	vocab    int
	name     string      // served id (reported by /v1/models, matched on the request model field)
	fp       string      // model fingerprint (binds --session-dir snapshots)
	adapter  string      // compute-time LoRA adapter name (#7); "" = base model. Shares model with its base.
	spec     bool        // --spec ngram: lossless n-gram (prompt-lookup) speculative decode with adaptive depth
	sessions *sessionLRU // prefix-keyed KV reuse across requests
	mu       sync.Mutex  // serialize this model's generations (the single decode worker)
	// queue bounds in-flight+waiting requests (cap = 1 running + --max-queue
	// waiting); a request claims a slot before mu. nil = unbounded. Honest
	// backpressure, not continuous batching — queue-full returns 429 Retry-After.
	queue chan struct{}

	// Vision tower (--vision): nil unless a multimodal model is loaded. When
	// present the model is "vision-capable" — serve accepts image content parts,
	// runs them through preprocess → encoder → projector, and routes the turn to
	// GenerateVL. The decode mutex above serializes vision turns too.
	venc    *vision.Encoder
	vproj   *multimodal.Projector
	vcfg    vision.Config
	vimgTok int // image-soft-token id (the placeholder embed-by-vector overrides); -1 if unresolved

	// Qwen2.5-VL vision tower (P5; nil ⇒ Gemma3/text). The merger is in the encoder
	// (no separate projector); preprocessing + m-RoPE are Qwen-specific, so the image
	// path branches on qwenEnc != nil.
	qwenEnc    *vision.QwenVisionEncoder
	qwenPP     multimodal.QwenPreprocessConfig
	qwenMerge  int // spatial_merge_size
	qwenImgTok int // <|image_pad|> id
}

// visionCapable reports whether this model has a loaded vision tower.
func (lm *loadedModel) visionCapable() bool {
	return (lm.venc != nil && lm.vproj != nil) || lm.qwenEnc != nil
}

// tryEnter claims a queue slot then locks the model's mutex (the decode worker).
// It returns false (writing nothing) when the queue is full, so each API surface
// can render the backpressure failure in its own error shape.
func (lm *loadedModel) tryEnter() bool {
	if lm.queue != nil {
		select {
		case lm.queue <- struct{}{}:
		default:
			return false
		}
	}
	lm.mu.Lock()
	return true
}

// enter is the OpenAI-flavored wrapper: a full queue writes a 429 + Retry-After.
func (lm *loadedModel) enter(w http.ResponseWriter) bool {
	if !lm.tryEnter() {
		w.Header().Set("Retry-After", "1")
		writeErr(w, http.StatusTooManyRequests, fmt.Sprintf("model %q queue full; retry", lm.name))
		return false
	}
	return true
}

// exit releases the mutex then the queue slot (paired with enter).
func (lm *loadedModel) exit() {
	lm.mu.Unlock()
	if lm.queue != nil {
		<-lm.queue
	}
}

type server struct {
	// Generative (decoder) registry — served name → model. Empty when only an
	// embedding model is served. Requests route on the OpenAI `model` field via
	// pick; each model has its own mutex, so distinct models run in parallel.
	// regMu guards the map structure (dynamic load/unload mutate it concurrently
	// with request routing); a request holds the picked *loadedModel beyond the
	// RLock, so unload uses the model's own mutex to refuse a busy model.
	regMu  sync.RWMutex
	models map[string]*loadedModel
	cfg    config // backend/quant/lora/kv/session-dir/allow-admin for admin loads

	// Embedding (encoder) half — nil when only a generative model is served.
	// The encoder is goroutine-safe for concurrent Encode, so /v1/embeddings is
	// served without a mutex (the per-model mutex guards only the shared decoder).
	embed    encoder.Encoder
	embedTok *embed.Tokenizer // counts tokens for usage.prompt_tokens
	embedID  string
	embedDim int

	// Responses API (/v1/responses) state store for store/previous_response_id.
	responses *responseStore
}

// pick resolves the OpenAI `model` field to a loaded generative model: an exact
// served-name match, else (for single-model OpenAI compatibility, where clients
// send an arbitrary name) the sole model when only one is loaded. nil otherwise —
// the handler returns an OpenAI-shaped 404.
func (s *server) pick(name string) *loadedModel {
	s.regMu.RLock()
	defer s.regMu.RUnlock()
	if lm, ok := s.models[name]; ok {
		return lm
	}
	if len(s.models) == 1 {
		for _, lm := range s.models {
			return lm
		}
	}
	return nil
}

// modelNotFound writes the OpenAI-shaped 404 for an unknown model field.
func (s *server) modelNotFound(w http.ResponseWriter, name string) {
	writeErr(w, http.StatusNotFound, fmt.Sprintf("model %q not found (served: %s)", name, strings.Join(s.servedNames(), ", ")))
}

// servedNames lists the loaded generative + embedding model ids (sorted).
func (s *server) servedNames() []string {
	s.regMu.RLock()
	names := make([]string, 0, len(s.models)+1)
	for n := range s.models {
		names = append(names, n)
	}
	s.regMu.RUnlock()
	if s.embed != nil {
		names = append(names, s.embedID)
	}
	sort.Strings(names)
	return names
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
	Role       string          `json:"role"`
	Content    json.RawMessage `json:"content"`                // string | [{"type":"text"|"image_url",…}, …]
	Name       string          `json:"name,omitempty"`         // tool messages: function name
	ToolCallID string          `json:"tool_call_id,omitempty"` // tool messages: id answered
	ToolCalls  []apiToolCall   `json:"tool_calls,omitempty"`   // assistant messages
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
	for _, name := range s.servedNames() {
		data = append(data, map[string]any{"id": name, "object": "model", "created": created, "owned_by": "goinfer"})
	}
	writeJSON(w, http.StatusOK, map[string]any{"object": "list", "data": data})
}

func (s *server) handleChat(w http.ResponseWriter, r *http.Request) {
	var req chatReq
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeErr(w, http.StatusBadRequest, "invalid request body: "+err.Error())
		return
	}
	// Multimodal: a message carrying an image_url part routes to the vision path.
	imgs, ierr := chatImages(req.Messages)
	if ierr != nil {
		writeErr(w, http.StatusBadRequest, ierr.Error())
		return
	}
	if len(imgs) > 0 {
		s.serveVisionChat(w, r, req, imgs)
		return
	}
	if len(req.Tools) > 0 && toolChoiceMode(req.ToolChoice) != "none" {
		s.handleChatTools(w, r, req)
		return
	}
	lm := s.pick(req.Model)
	if lm == nil {
		s.modelNotFound(w, req.Model)
		return
	}
	ids, err := lm.chatPrompt(req.Messages)
	if err != nil {
		writeServerErr(w, "encode: "+err.Error())
		return
	}
	gr, err := lm.prepare(req.sampling, ids)
	if err != nil {
		writeErr(w, http.StatusBadRequest, err.Error())
		return
	}
	if !lm.enter(w) {
		return
	}
	defer lm.exit()
	id := "chatcmpl-" + reqID()
	created := time.Now().Unix()

	if req.Stream {
		f, ok := sseStart(w)
		if !ok {
			return
		}
		role := chatChunk(id, created, lm.name, delta{Role: "assistant"}, nil)
		sseSend(w, f, role)
		finish, _, _, _, gerr := lm.drive(r.Context(), gr, func(t string) {
			sseSend(w, f, chatChunk(id, created, lm.name, delta{Content: t}, nil))
		})
		if gerr != nil {
			sseErr(w, f, "generation failed: "+gerr.Error())
			sseDone(w, f)
			return
		}
		sseSend(w, f, chatChunk(id, created, lm.name, delta{}, &finish))
		sseDone(w, f)
		return
	}

	var sb strings.Builder
	finish, nComp, lps, _, gerr := lm.drive(r.Context(), gr, func(t string) { sb.WriteString(t) })
	if gerr != nil {
		writeServerErr(w, "generation failed: "+gerr.Error())
		return
	}
	choice := map[string]any{
		"index":         0,
		"message":       map[string]any{"role": "assistant", "content": sb.String()},
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
	lm := s.pick(req.Model)
	if lm == nil {
		s.modelNotFound(w, req.Model)
		return
	}
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
	if !lm.enter(w) {
		return
	}
	defer lm.exit()
	id := "cmpl-" + reqID()
	created := time.Now().Unix()

	if req.Stream {
		f, ok := sseStart(w)
		if !ok {
			return
		}
		finish, _, _, _, gerr := lm.drive(r.Context(), gr, func(t string) {
			sseSend(w, f, completionChunk(id, created, lm.name, t, nil))
		})
		if gerr != nil {
			sseErr(w, f, "generation failed: "+gerr.Error())
			sseDone(w, f)
			return
		}
		sseSend(w, f, completionChunk(id, created, lm.name, "", &finish))
		sseDone(w, f)
		return
	}
	var sb strings.Builder
	finish, nComp, _, _, gerr := lm.drive(r.Context(), gr, func(t string) { sb.WriteString(t) })
	if gerr != nil {
		writeServerErr(w, "generation failed: "+gerr.Error())
		return
	}
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
	masker      *constrain.Masker // set on constrained requests (response_format / tool grammar); enables grammar-spec
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
		gr.masker = m // enables grammar-fused speculative decode (drive)
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
			system = m.text()
		case "tool":
			turns = append(turns, chat.Turn{Role: "tool", Content: m.text(), ToolName: m.Name, ToolCallID: m.ToolCallID})
		case "assistant":
			var tc []chat.ToolCall
			for _, c := range m.ToolCalls {
				tc = append(tc, chat.ToolCall{ID: c.ID, Name: c.Function.Name, Arguments: json.RawMessage(c.Function.Arguments)})
			}
			turns = append(turns, chat.Turn{Role: "assistant", Content: m.text(), ToolCalls: tc})
		default:
			turns = append(turns, chat.Turn{Role: "user", Content: m.text()})
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
// encode tokenizes prompt. The error is a server-side condition (a decode-only
// vocab, or a tokenizer that failed to load) — it was silently dropped before
// (M1), yielding an empty prompt and a generation from BOS alone.
func (lm *loadedModel) encode(prompt string) ([]int, error) {
	return lm.tk.Encode(prompt, lm.tmpl == nil)
}

// chatPrompt renders system + messages into the model's chat template (no tools).
func (lm *loadedModel) chatPrompt(msgs []chatMessage) ([]int, error) {
	system, turns := messagesToTurns(msgs)
	return lm.promptFor(system, turns)
}

// promptFor renders system + turns into token ids via the model's chat template
// (raw-conversation fallback when the family is unrecognized). Shared by the
// OpenAI and Anthropic chat paths so both encode prompts identically.
func (lm *loadedModel) promptFor(system string, turns []chat.Turn) ([]int, error) {
	if lm.tmpl != nil {
		return lm.encode(lm.tmpl.Render(system, turns))
	}
	return lm.encode(rawPrompt(system, turns))
}

// genErr filters a generation's terminal error (gen.Err()) down to what's worth
// surfacing to the client: context.Canceled — our own stop-string cancel, or a
// client disconnect — is a clean end, not a failure. A non-nil result becomes a
// 500 (or an error SSE event mid-stream). M1.
func genErr(err error) error {
	if err != nil && !errors.Is(err, context.Canceled) {
		return err
	}
	return nil
}

// drive runs the generation, applying stop strings and UTF-8 holdback, calling
// onText with each newly-completed text fragment. Returns the finish reason
// ("stop" | "length"), the completion token count, (non-stream) per-token
// logprobs, the stop string that was hit (empty unless a stop sequence ended the
// turn), and any terminal generation error (nil on a clean end — see genErr). The
// context is cancelled on a stop-string hit to end generation.
func (lm *loadedModel) drive(parent context.Context, gr genRequest, onText func(string)) (string, int, []decoder.SampleInfo, string, error) {
	ctx, cancel := context.WithCancel(parent)
	defer cancel()
	var stream <-chan int
	var gen *decoder.Generation

	// GPU-resident models take the STATELESS path. decoder.Generate only engages the resident
	// DecodeRunner when there is no session commit and no prefix reuse (model.go:
	// useGPU = resident != nil && prefillFrom == 0 && commit == nil) — the resident's KV lives
	// on the GPU, while a session's prefix-reuse cache is CPU-side, so the two cannot both be
	// the source of truth. Going through a session therefore silently dropped every request to
	// the staged/CPU path: measured 13 tok/s vs ~460 resident on a 0.5B (RTX 2070 SUPER).
	//
	// Trading prefix reuse for resident decode is a large net win and semantically safe: the
	// OpenAI API is stateless (the client resends the whole conversation), so sessions here are
	// purely a TTFT optimisation, not correctness. Spec-decode is session-only and does not mix
	// with residency in any case (a CPU drafter against a GPU target measured 0.11x).
	if lm.model.ResidentActive() {
		stream, gen = lm.model.Generate(ctx, gr.promptIDs, gr.maxTokens, gr.sp)
		finish, n, stopHit := lm.streamTokens(cancel, stream, gr, onText)
		return finish, n, gen.Logprobs, stopHit, genErr(gen.Err())
	}

	// Reuse the KV of whichever cached session already holds this prompt as a
	// prefix (continuing chat / agent loop): only the new suffix is prefilled.
	sess := lm.sessions.acquire(gr.promptIDs)
	switch {
	case lm.spec && gr.masker != nil && gr.sp.Temperature == 0:
		// Constrained request (response_format / tool grammar), greedy: grammar-fused
		// speculative decode (01/03). A RouterDrafter fuses the grammar's forced byte-run
		// (structural tokens) with an n-gram copy of free values that echo the context;
		// the masked verify keeps output identical to constrained Generate. A miss costs
		// ~nothing, so it's safe to always run here. Falls back to plain constrained
		// decode on any validation error (the error precedes touching the session cache).
		spSpec := gr.sp
		spSpec.LogitProcessor = nil // the verify applies the grammar mask itself
		drafter := &decoder.RouterDrafter{Sources: []decoder.Drafter{
			&decoder.GrammarDrafter{Mask: gr.masker, Encode: func(s string) []int { ids, _ := lm.tk.Encode(s, false); return ids }},
			&decoder.NgramDrafter{},
		}}
		var err error
		stream, gen, err = sess.GenerateGrammarSpeculative(ctx, gr.promptIDs, gr.maxTokens, gr.masker, drafter, 8, spSpec)
		if err != nil {
			stream, gen = sess.Generate(ctx, gr.promptIDs, gr.maxTokens, gr.sp)
		}
	case lm.spec:
		// Lossless n-gram speculative decode with adaptive depth. Falls back to plain
		// Generate when the request's sampler isn't yet supported on the spec path
		// (repetition/presence/frequency penalties, logit bias, or a constrained/tool
		// LogitProcessor with temperature>0) — the validation error is returned before
		// the session cache is touched, so the fallback is exact.
		var err error
		stream, gen, err = sess.GenerateNgramSpeculativeAdaptive(ctx, gr.promptIDs, gr.maxTokens, &decoder.NgramDrafter{}, &decoder.AdaptiveDepth{MaxDraft: 8}, gr.sp)
		if err != nil {
			stream, gen = sess.Generate(ctx, gr.promptIDs, gr.maxTokens, gr.sp)
		}
	default:
		stream, gen = sess.Generate(ctx, gr.promptIDs, gr.maxTokens, gr.sp)
	}
	finish, n, stopHit := lm.streamTokens(cancel, stream, gr, onText)
	return finish, n, gen.Logprobs, stopHit, genErr(gen.Err())
}

// driveVL is drive for a multimodal turn: it prefills gr.promptIDs with the
// projected vision `feats` spliced in at the [imgPos, imgPos+imgLen) placeholder
// run (GenerateVL), then streams the continuation through the same stop/UTF-8
// machinery as drive. Stateless — no warm-KV session (multimodal opts out of
// prefix reuse). Returns finish reason, completion token count, stop string, and
// any terminal generation error (nil on a clean end — see genErr).
func (lm *loadedModel) driveVL(parent context.Context, gr genRequest, vi visionInput, onText func(string)) (string, int, string, error) {
	ctx, cancel := context.WithCancel(parent)
	defer cancel()
	var stream <-chan int
	var gen *decoder.Generation
	if vi.qwen {
		stream, gen = lm.model.GenerateQwenVL(ctx, gr.promptIDs, vi.feats, vi.imgPos, vi.imgLen, [][3]int{vi.grid}, lm.qwenMerge, lm.qwenImgTok, gr.maxTokens, gr.sp)
	} else {
		stream, gen = lm.model.GenerateVL(ctx, gr.promptIDs, vi.feats, vi.imgPos, vi.imgLen, gr.maxTokens, gr.sp)
	}
	finish, n, stopHit := lm.streamTokens(cancel, stream, gr, onText)
	return finish, n, stopHit, genErr(gen.Err())
}

// streamTokens consumes a token-id channel, applying stop strings and UTF-8
// holdback and calling onText with each newly-completed fragment. It is the
// shared tail of every generation path (text via Session.Generate, multimodal
// via driveVL) — the token *source* is orthogonal to this stop/stream logic.
// cancel ends the producing generation on a stop-string hit. Returns the finish
// reason ("stop" | "length"), the completion token count, and the stop string
// that was hit (empty unless a stop sequence ended the turn).
func (lm *loadedModel) streamTokens(cancel context.CancelFunc, stream <-chan int, gr genRequest, onText func(string)) (string, int, string) {
	var ids []int
	printed := 0
	finish := ""
	stopHit := ""
	stopping := false
	for id := range stream {
		if stopping {
			continue // drain so the generation goroutine exits cleanly
		}
		ids = append(ids, id)
		text, _ := lm.tk.Decode(ids)
		if cut, which, hit := firstStop(text, gr.stopStrings); hit {
			if cut > printed {
				onText(text[printed:cut])
			}
			finish, stopHit, stopping = "stop", which, true
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
	return finish, len(ids), stopHit
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
