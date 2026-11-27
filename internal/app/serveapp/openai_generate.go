package serveapp

import (
	"context"
	"errors"
	"fmt"
	"io"
	"math"
	"math/rand/v2"
	"net/http"
	"os"
	"slices"
	"strings"
	"sync"

	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/constrain"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/engine"
	"github.com/townsendmerino/goinfer/internal/app/connect"
	"github.com/townsendmerino/goinfer/internal/app/serveapp/jobs"
	"github.com/townsendmerino/goinfer/internal/app/serveapp/stores"
	"github.com/townsendmerino/goinfer/internal/clef"
	"github.com/townsendmerino/goinfer/internal/cost"
	"github.com/townsendmerino/goinfer/internal/decide"
	samplingp "github.com/townsendmerino/goinfer/internal/runtime/sampling"
	"github.com/townsendmerino/goinfer/internal/runtime/scheduler"
	"github.com/townsendmerino/goinfer/tokenizer"
)

const defaultMaxTokens = 512

// maxTopLogprobs is the largest `top_logprobs` a request may ask for: OpenAI's own documented ceiling, so no
// compatible client is refused by it. Unbounded, one request retains max_tokens x top_logprobs entries and builds a
// map per entry before writing a byte.
const maxTopLogprobs = 20

// maxOutputTokensCeiling is the hard upper bound on a request's max_tokens. KV is preallocated as
// len(prompt)+max_tokens per layer, so an unbounded value would trigger a fatal Go out-of-memory throw that kills the
// server for every client. A request above it is rejected 400, not clamped, so the caller learns its request was too
// big.
const maxOutputTokensCeiling = 131072

// modelCaches groups the lazily computed, sync.Once-guarded derived tables cached per loadedModel.
type modelCaches struct {
	featOnce        sync.Once // vision_serve.go: the tower-output cache, made on first image
	feat            *stores.FeatureCache
	tokenBytesOnce  sync.Once // constraint masker's token-to-bytes table (one entry per vocab id)
	tokenBytes      [][]byte
	tokenBytesFrom  *loadedModel // base model whose table this adapter shares
	idsIdentOnce    sync.Once    // model-and-tokenizer identity carried-id blobs name (idsIdentity)
	idsIdent        string
	maxTokBytesOnce sync.Once // byte length of the longest token in the vocab (pre-tokenize guard)
	maxTokBytes     int
}

// decisionState groups the /v1/systemone decision-routing state on a loadedModel (Route B head, Route C Clef, or lazy label decider).
type decisionState struct {
	head        *decide.Head // trained decision head (Route B), nil for label scoring
	clef        *clef.Model  // Clef joint-head model (Route C), set when model dir has joint_head.safetensors
	deciderOnce sync.Once
	decider     *decide.Decider
	deciderErr  error
}

// loadedModel is one resident generative model and the per-model state a request needs: its tokenizer, chat template,
// stop ids, vocab, warm-KV sessions, and the admission (turns) that serializes its generations. The server registry
// holds them keyed by served name; requests to distinct models run in parallel.
type loadedModel struct {
	tk           *tokenizer.Tokenizer
	model        *decoder.Model
	closer       io.Closer      // test-only liveness/close stand-in when model is nil; production uses model directly (livenessKey)
	tmpl         *chat.Template // nil → raw completion
	stopIDs      []int          // turn-stop token ids from the template
	eosIDs       []int
	vocab        int
	name         string // served id (reported by /v1/models, matched on the request model field)
	fp           string // model fingerprint (binds --session-dir snapshots)
	source       string // the model path as loadDecoder resolved it (an hf:/demo: reference → what it fetched); "" for an adapter
	adapter      string // compute-time LoRA adapter name (#7); "" = base model. Shares model with its base.
	spec         bool   // --spec ngram: lossless n-gram (prompt-lookup) speculative decode with adaptive depth
	specAdaptive bool   // --spec-adaptive: speculate only while alone, join the batched decode otherwise
	// blockSpec is an attached pretrained block drafter (--drafter), nil when unused. It is attached once at load:
	// the weight upload is a per-process cost, and paying it per request is a several-fold slowdown (docs/spec/08-dspark-dflash.md).
	blockSpec blockDrafter
	// sampling is the checkpoint's recommended sampling (decoder.Config.Sampling), which prepare applies to the fields a
	// request leaves out; nil when it carries none.
	sampling  *decoder.RecommendedSampling
	sessions  *sessionLRU // prefix-keyed KV reuse across requests
	decisions decisionState
	caches    modelCaches
	turns     admission // FIFO, context-aware granter of decode turns; a waiter can give up on disconnect or halt
	// sessMu guards sessions, the sessionLRU, which is not goroutine-safe (sessions.go): drive's acquire and checkin
	// and the background save, restore and demote goroutines take it. It is held around those operations only, not
	// across a generation: the session a generation is using is checked out instead (sessionLRU.busy), and the LRU
	// never hands out, evicts, demotes, saves or reads a busy session. That is what lets several generations of one
	// CPU model run at once (-max-concurrent) while nothing frees or moves memory under a running generation.
	sessMu sync.Mutex
	// queue bounds in-flight plus waiting requests (cap = concurrent + --max-queue); a request claims a slot before
	// its turn. nil = unbounded. Backpressure, not batching: a full queue returns 429 Retry-After.
	queue chan struct{}
	// concurrent is how many generations of this model may run at once (see setConcurrency); 1 = serialized.
	concurrent int

	// vision is the entry's image input (--vision; nil: the model takes no image). A model with one is vision-capable:
	// serve accepts image content parts, runs them through the family's preprocessing and tower, and routes the turn to
	// the family's generator.
	vision visionTower

	// visionShare counts the entries holding this entry's startup vision tower (nil: none shares it); the last to unload
	// closes it.
	visionShare *visionShare

	// audio is the entry's audio input (nil: the model takes no audio clip): a Gemma 4 checkpoint with an audio_config,
	// Qwen3-ASR or Voxtral Mini. Its encoder loads on the first clip.
	audio audioTower

	// scripted is a test seam (a test sets fakeEngine to scriptedEngine): when non-nil, drive streams these token ids (at most the request's max tokens; the end of the
	// slice is the model's end of turn) instead of running model. Production never sets it.
	scripted []int
	// onScripted, when set with scripted, receives each scripted generation's prompt ids.
	onScripted func(promptIDs []int)
	// scriptedRecurrent makes a scripted model one whose snapshots carry its recurrent state (recurrentKeep computes a Keep for it), and
	// onScriptedScope receives the KV scope each scripted generation would hand the decoder. Test seams, as scripted.
	scriptedRecurrent bool
	onScriptedScope   func(kvScope)
	// fakeEngine, when set, is the engine every text generation runs on. A test seam.
	fakeEngine engine.Engine
}

// cachedTokenBytes returns the constraint masker's token-to-bytes table, built once per model and reused across
// constrained requests.
func (lm *loadedModel) cachedTokenBytes() [][]byte {
	if lm.caches.tokenBytesFrom != nil {
		return lm.caches.tokenBytesFrom.cachedTokenBytes()
	}
	lm.caches.tokenBytesOnce.Do(func() {
		lm.caches.tokenBytes = constrain.TokenBytes(lm.vocab, lm.tk.TokenText)
	})
	return lm.caches.tokenBytes
}

// maxTokenBytes returns the byte length of the longest token in the vocab (at least 1), computed once and cached. It
// is the per-token byte ceiling of the tokenization guard.
func (lm *loadedModel) maxTokenBytes() int {
	lm.caches.maxTokBytesOnce.Do(func() {
		m := 1
		for id := 0; id < lm.vocab; id++ {
			if n := len(lm.tk.TokenText(id)); n > m {
				m = n
			}
		}
		lm.caches.maxTokBytes = m
	})
	return lm.caches.maxTokBytes
}

// promptTooLargeForContext cheaply rejects an input whose tokenizable text cannot fit the model's context window,
// before the expensive tokenize. It is a conservative upper bound that never rejects a servable prompt (ctx tokens
// have text of at most ctx*maxTok bytes), so the exact post-tokenize check (contextLengthError) still runs for what
// passes. It uses MaxPositions, not the smaller resident cap, to stay an upper bound; prepare applies the resident
// cap.
func (lm *loadedModel) promptTooLargeForContext(inputBytes int) error {
	if lm.model == nil {
		return nil
	}
	return promptByteBudgetError(inputBytes, lm.model.Config().MaxPositions, lm.maxTokenBytes())
}

// promptByteBudgetError is the pure guard: input text longer than ctx·maxTokenBytes needs
// more than ctx tokens and cannot fit. ctx ≤ 0 (unknown) or maxTokenBytes ≤ 0 never rejects.
func promptByteBudgetError(inputBytes, ctx, maxTokenBytes int) error {
	if ctx > 0 && maxTokenBytes > 0 && inputBytes > ctx*maxTokenBytes {
		return fmt.Errorf("prompt is too large for the model's context window of %d tokens (context_length_exceeded)", ctx)
	}
	return nil
}

// chatInputBytes sums the tokenizable text across chat messages: the input the BPE runs over (JSON structure and
// image data are not tokenized), so it is what the promptTooLargeForContext guard bounds. It includes each message's
// replayed tool_calls[].function.arguments: messagesToTurns converts an assistant message's ToolCalls
// unconditionally, and a chat template renders them into the prompt whenever they appear in history, tools active
// this turn or not, so an unpriced arguments blob would run the full BPE past the guard.
func chatInputBytes(msgs []chatMessage) int {
	n := 0
	for _, m := range msgs {
		n += len(m.text())
		for _, tc := range m.ToolCalls {
			n += len(tc.Function.Name) + len(tc.Function.Arguments)
		}
	}
	return n
}

// toolSchemaBytes sums an OpenAI-shaped tool declaration list's rendered bytes. RenderToolsSegments renders every
// tool's name, description and parameters into the prompt whenever tools are active, so a large schema list is priced
// at the same guard as the messages, not left to run the full BPE unpriced.
func toolSchemaBytes(tools []toolSpec) int {
	n := 0
	for _, t := range tools {
		n += len(t.Function.Name) + len(t.Function.Description) + len(t.Function.Parameters)
	}
	return n
}

// setConcurrency decides how many generations of this model run at once and sizes the admission and queue to match:
// -max-concurrent, but 1 unless the model is safe to run concurrently on the CPU (decoder.Model.CPUConcurrentSafe:
// not GPU-resident, not weight-streaming) and has no vision tower, and never more than -kv-sessions keeps, since each
// running generation holds a session and the LRU must always have an idle one or room. The queue holds the running
// generations plus -max-queue waiting ones.
func (lm *loadedModel) setConcurrency(cfg config) (line string) {
	n := max(1, cfg.maxConcurrent)
	switch {
	case n <= 1 || lm.model == nil || lm.visionCapable():
		n = 1
	case lm.model.ResidentActive():
		// A GPU-resident model runs n generations at once only when its resident can batch their decode tokens
		// (decoder.Model.EnableResidentConcurrency: a dense family on at least two resident KV slots, capped by the
		// slot count that --kv-sessions sets). Speculative decode, block drafters and adapters take the resident
		// exclusively for a whole generation, so a model serving them keeps one generation at a time, unless
		// --spec-adaptive is also set: then a spec generation gives the resident up at a round boundary whenever
		// another is waiting. Block drafters and adapters are unaffected by --spec-adaptive.
		if lm.blockSpec != nil || lm.adapter != "" || (lm.spec && !lm.specAdaptive) {
			n = 1
		} else {
			n = lm.model.EnableResidentConcurrency(n)
			if lm.spec && lm.specAdaptive {
				lm.model.SetSpecAdaptive(n > 1)
			}
		}
	case !lm.model.CPUConcurrentSafe():
		n = 1
	}
	if cfg.kvSessions > 0 {
		n = min(n, cfg.kvSessions)
	}
	// A CPU model's concurrent generations may also join their decode tokens into batched steps
	// (decoder.Model.EnableCPUBatch decides from -cpu-batch, the family and the model's size).
	cpuBatched := false
	if lm.model != nil && !lm.model.ResidentActive() {
		cpuBatched = lm.model.EnableCPUBatch(n)
	}
	lm.concurrent = n
	lm.turns.setCap(n)
	if cfg.maxQueue > 0 {
		lm.queue = scheduler.NewQueue(n, cfg.maxQueue)
	}
	if lm.model == nil {
		return ""
	}
	return concurrencyLine(bannerFacts{resident: lm.model.ResidentActive(), kvSlots: lm.model.ResidentKVSlots(), concurrent: n,
		cpuBatched: cpuBatched}, cfg)
}

// audioCapable reports whether this model can take an audio clip (a Gemma 4 checkpoint with an audio tower, a Qwen3-ASR checkpoint, or Voxtral Mini).
func (lm *loadedModel) audioCapable() bool {
	return lm.audio != nil
}

// visionCapable reports whether this model has a loaded vision tower.
func (lm *loadedModel) visionCapable() bool {
	return lm.vision != nil
}

// tryEnter claims a queue slot, then waits for this model's turn. It returns (false, "") when the queue is full, so
// each API surface can render the backpressure failure in its own error shape, and also when ctx ends while queued
// (client disconnect, or a halt racing the wait): admission.enter returns without first being granted the turn.
//
// haltState, when non-nil, is also checked once the turn is granted; a non-nil result is returned as (false, reason)
// with the turn and queue slot released again. That catches a halt landing between the grant and this check: haltGate
// (main.go) runs once, at the front of the chain, before admission, and nothing has registered this generation in the
// generations registry yet (that happens inside drive/driveVL).
func (lm *loadedModel) tryEnter(ctx context.Context, rec admissionRecord, haltState func() *haltInfo) (ok bool, haltReason string) {
	var halted func() (string, bool)
	if haltState != nil {
		halted = func() (string, bool) {
			if hi := haltState(); hi != nil {
				return hi.reason, true
			}
			return "", false
		}
	}
	rec.clock.Queue()
	ok, haltReason = scheduler.Admit(ctx, &lm.turns.Admission, lm.queue, scheduler.Record{PromptIDs: rec.promptIDs, ID: rec.id}, halted)
	if ok {
		rec.clock.Admit()
	}
	return ok, haltReason
}

// enter is the OpenAI-flavored wrapper: a full queue writes a 429 + Retry-After; a halt found
// after the queue wait writes the same 503 shape haltGate does at the front of the chain. A
// context that ended while queued (client gone) writes nothing at all — the client is no longer
// listening.
func (lm *loadedModel) enter(w http.ResponseWriter, r *http.Request, rec admissionRecord, haltState func() *haltInfo) bool {
	ok, haltReason := lm.tryEnter(r.Context(), rec, haltState)
	if ok {
		return true
	}
	if haltReason != "" {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"error": "halted", "reason": haltReason})
		return false
	}
	if r.Context().Err() != nil {
		return false // client disconnected while queued; nothing to write to
	}
	w.Header().Set("Retry-After", "1")
	writeErr(w, http.StatusTooManyRequests, lm.queueFullMsg())
	return false
}

// queueFullMsg is the 429's text when a model's queue is full. cap(lm.queue) minus the concurrency is the configured
// -max-queue depth, the one number always true of a full queue (a live waiting count would be stale by the time a
// client reads it). A nil queue (unbounded) cannot reach this, but a message with no number is the safe fallback.
func (lm *loadedModel) queueFullMsg() string {
	if lm.queue != nil {
		return fmt.Sprintf("model %q queue full (max %d queued); retry", lm.name, scheduler.MaxQueued(lm.queue, lm.concurrent))
	}
	return fmt.Sprintf("model %q queue full; retry", lm.name)
}

// exit releases the turn, then the queue slot (the reverse of enter's acquisition). Turns are interchangeable, so
// release needs no per-caller identity. drive itself checks the session back in.
func (lm *loadedModel) exit() { scheduler.Exit(&lm.turns.Admission, lm.queue) }

// contextWindow is the context cap prepare enforces for one request: the model's MaxPositions, lowered to the
// resident KV cap when the request runs the stateless resident path (residentPath). One function serves both the
// enforcement and what /v1/models publishes as context_window, so the number a client plans against is the number
// that rejects it. 0 = unknown.
//
// Why the resident cap: the stateless resident path prefills the fixed-size resident KV, which is often smaller than
// MaxPositions. A prompt between the two would pass the MaxPositions check and die mid-prefill with a 500 whose body
// leaks an internal hint, so it is rejected as a clean context_length_exceeded 400 instead. Vision requests (CPU VL)
// pass residentPath false and are bounded by MaxPositions.
func (lm *loadedModel) contextWindow(residentPath bool) int {
	if lm.model == nil {
		return 0
	}
	ctx := lm.model.Config().MaxPositions
	if residentPath {
		if rc := lm.model.ResidentContextCap(); rc > 0 && (ctx <= 0 || rc < ctx) {
			ctx = rc
		}
	}
	return ctx
}

// modelContextWindow is the model's own window, with no resident cap applied — what -ctx can raise the GPU context up to. Next to
// contextWindow so the limits are derived in one place (prepare reads neither MaxPositions nor ResidentContextCap itself).
func (lm *loadedModel) modelContextWindow() int {
	if lm.model == nil {
		return 0
	}
	return lm.model.Config().MaxPositions
}

// residentPath reports whether a text-completion request against lm will run the stateless GPU-resident decode path,
// for contextWindow and prepare's residentPath argument. It is not `lm.adapter == ""`: an adapter model's first turn
// (prefillFrom==0) still runs resident GPU decode (decoder/model.go's generateInto useGPU admits it), so keying on
// "no adapter" enforced the uncapped MaxPositions against a request that generateInto binds to the smaller
// ResidentContextCap(), and it died mid-prefill with a leaking 500 instead of a clean 400. ResidentActive() is safe
// for every other combination: where nothing is resident ResidentContextCap() reports 0, so contextWindow's `rc > 0`
// guard makes the value irrelevant. Guarded against lm.model == nil (an embedding-only entry) as contextWindow is.
func (lm *loadedModel) residentPath() bool {
	return lm.model != nil && lm.model.ResidentActive()
}

// prepare translates the OpenAI sampling fields into goinfer's SamplingParams, wires response_format into a
// constraint masker, and resolves stop strings. residentPath says whether this request will run the stateless
// GPU-resident decode path, so the resident context cap binds (see residentPath): text-completion callers pass
// lm.residentPath(), and vision requests pass false (GenerateVL is CPU-prefilled; the resident image-reuse path
// checks its own cap in decoder/generate_vl.go).
func (lm *loadedModel) prepare(sm sampling, promptIDs []int, residentPath bool) (genRequest, error) {
	sm, fromModel := withRecommended(sm, lm.sampling)
	sp := samplingp.SamplingParams{
		Temperature: deref(sm.Temperature, 1.0),
		MinP:        deref(sm.minP, 0),
		Seed:        seedOrRandom(sm.Seed), // omitted seed: fresh random, not seed 0
		StopIDs:     lm.stopIDs,
		Logprobs:    sm.Logprobs,
		TopLogprobs: deref(sm.TopLogprobs, 0),
	}
	// A negative temperature is rejected, as a negative top_p is. 0 is the documented greedy setting and stays valid.
	if sm.Temperature != nil && *sm.Temperature < 0 {
		return genRequest{}, fmt.Errorf("temperature must be >= 0 (got %v); 0 selects greedy/deterministic decoding", *sm.Temperature)
	}
	// top_p == 0 is the tightest nucleus, which is greedy, but the sampler treats TopP == 0 as disabled and would
	// draw from the full vocabulary: reject outside [0,1] and map an explicit 0 to greedy.
	if sm.TopP != nil {
		switch p := *sm.TopP; {
		case p < 0 || p > 1:
			return genRequest{}, fmt.Errorf("top_p must be in [0,1] (got %v)", p)
		case p == 0:
			sp.Temperature = 0
		case p < 1:
			sp.TopP = p
		}
	}
	// top_logprobs is validated because it is the most expensive sampling field to get wrong: each retained entry is
	// a TokenLogprob, and the response builder materializes a map per entry before writing a byte, so a huge value
	// with a large max_tokens exhausts memory in one request (a fatal allocation failure, not a 500). [0,20] is
	// OpenAI's own documented range, so this rejects nothing a compatible client sends.
	if n := deref(sm.TopLogprobs, 0); n < 0 || n > maxTopLogprobs {
		return genRequest{}, fmt.Errorf("top_logprobs must be in [0,%d] (got %d)", maxTopLogprobs, n)
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
	if len(sm.LogitBias) > 0 {
		b, err := parseLogitBias(sm.LogitBias, lm.vocab)
		if err != nil {
			return genRequest{}, err
		}
		sp.LogitBias = b
	}
	maxTok := sm.MaxTokens
	if sm.MaxCompletionTokens != nil { // OpenAI's newer field wins over the legacy max_tokens
		maxTok = sm.MaxCompletionTokens
	}
	if maxTok != nil {
		// A zero or negative value reaches NewCache(len(prompt)+maxTokens) as a negative makeslice cap, an
		// unrecovered panic (fatal on the VL path's bare goroutine), and a huge value exhausts memory. Reject both
		// here so every endpoint that calls prepare answers a clean 400.
		if *maxTok < 1 {
			return genRequest{}, fmt.Errorf("max_tokens must be >= 1 (got %d)", *maxTok)
		}
		if *maxTok > maxOutputTokensCeiling {
			return genRequest{}, fmt.Errorf("max_tokens %d exceeds the server ceiling of %d", *maxTok, maxOutputTokensCeiling)
		}
	}
	stops := parseStop(sm.Stop)
	if len(stops) > maxStopSequences {
		return genRequest{}, fmt.Errorf("stop accepts at most %d sequences (got %d)", maxStopSequences, len(stops))
	}
	gr := genRequest{
		promptIDs:   promptIDs,
		sp:          sp,
		maxTokens:   deref(maxTok, defaultMaxTokens),
		stopStrings: stops,
		outcome:     &turnOutcome{},
		fromModel:   fromModel,
		clock:       jobs.NewTurnClock(sm.Goinfer.WantsCost()),
	}
	// Reject a prompt that does not fit the context window, then clamp max_tokens to the room left (clampMaxTokens).
	if lm.model != nil {
		ctx := lm.contextWindow(residentPath)
		if maxTok == nil && sm.outputByWindow && ctx > 0 {
			gr.maxTokens = connect.OutputLimit(ctx)
		}
		if err := newContextOverflow(len(promptIDs), ctx, lm.modelContextWindow(), gr.maxTokens, sm.fixedTokens); err != nil {
			return genRequest{}, err
		}
		gr.maxTokens = clampMaxTokens(gr.maxTokens, len(promptIDs), ctx)
	}
	g, err := grammarFor(sm.ResponseFormat)
	if err != nil {
		return genRequest{}, err
	}
	if sm.Confidence {
		switch {
		case !sm.confidenceOK:
			return genRequest{}, fmt.Errorf("goinfer_confidence is supported on /v1/chat/completions and /v1/completions with response_format json_schema, not with tools, images, jobs or batches")
		case sm.ResponseFormat == nil || sm.ResponseFormat.Type != "json_schema":
			return genRequest{}, fmt.Errorf("goinfer_confidence needs response_format {\"type\": \"json_schema\"}: the schema is what says each field's kind")
		}
	}
	if g != nil {
		eos := slices.Concat(lm.eosIDs, lm.stopIDs)
		m := constrain.NewMasker(g, lm.cachedTokenBytes(), eos).StopWhenComplete()
		if sm.Confidence {
			m.CaptureConfidence(constrain.ConfidenceOptions{})
			gr.conf = &confResult{}
		}
		gr.sp.LogitProcessor = m.Process
		gr.masker = m // enables grammar-fused speculative decode (drive)
	}
	// The load-time fit guard prices the worst case a request could reach, once, at load; it cannot see the request
	// that arrives. This check runs last, after promptIDs and maxTokens are fully resolved (including the clamp
	// above), so the numbers in a refusal are the real ones, and every prepare caller gets it without its own copy.
	if lm.model != nil {
		// With several generations allowed at once, this request's prefill shares the memory margin with the ones
		// running or queued ahead of it: counted now, capped at the model's concurrency. A lone request keeps the
		// whole margin, and so does one that prefills on a GPU resident: its prefill passes run one at a time, and
		// the margin is live memory, which already excludes what the other generations hold. Splitting it there
		// refuses ordinary prompts (docs/measurements/spec-vs-batching-metal-2026-09-27.md section 5).
		share := 1
		if lm.concurrent > 1 && !(residentPath && lm.model.ResidentActive()) {
			share = min(lm.concurrent, lm.turns.load()+1)
		}
		if aerr := lm.model.AdmitPrefillMemoryShare(len(gr.promptIDs), gr.maxTokens, residentPath, share); aerr != nil {
			return genRequest{}, &prefillMemoryError{aerr}
		}
	}
	return gr, nil
}

// prefillMemoryError marks the prefill-memory admission refusal, as distinct from prepare's ordinary validation
// errors (bad field, out-of-range value), so every call site can answer 413 instead of 400: a request that would page
// is not a malformed request, and a client's retry logic should treat the two differently. See prepareErrStatus.
type prefillMemoryError struct{ err error }

func (e *prefillMemoryError) Error() string { return e.err.Error() }
func (e *prefillMemoryError) Unwrap() error { return e.err }

// prepareErrStatus is the one place every prepare call site decides its status code, in either response format
// (writeErr's OpenAI shape or writeAnthropicErr's), so the 413 split cannot regress to a flat 400 at a site someone
// forgets to update.
func prepareErrStatus(err error) int {
	if _, ok := errors.AsType[*prefillMemoryError](err); ok {
		return http.StatusRequestEntityTooLarge
	}
	return http.StatusBadRequest
}

// seedOrRandom returns the request's seed, or a fresh random seed when absent. OpenAI's contract is that an omitted
// seed varies output run to run (best-of-N, "regenerate" and agent retry-for-diversity depend on it). A supplied
// seed, including 0, is honored verbatim for reproducibility.
func seedOrRandom(seed *int64) int64 {
	if seed != nil {
		return *seed
	}
	return rand.Int64()
}

// clampMaxTokens bounds max_tokens by the model's context window. The KV cache is preallocated as
// NewCache(len(prompt)+max_tokens), and the server ceiling is far larger than most models' context, so a request at
// the ceiling against a small-context model would preallocate tens of GiB per layer and OOM-kill the server. The
// model cannot attend past ctx (MaxPositions) anyway, so tokens beyond it are wasted: clamp to what fits, as the
// resident path's ContextCap clamp does (decoder/model.go). It only shrinks: ctx <= 0 (unknown) or a request already
// within context is returned unchanged, and a prompt that itself meets or exceeds ctx is left to contextLengthError,
// not clamped to 0.
func clampMaxTokens(maxTokens, promptLen, ctx int) int {
	if ctx > 0 && promptLen < ctx {
		if room := ctx - promptLen; room < maxTokens {
			return room
		}
	}
	return maxTokens
}

// encode tokenizes a rendered prompt; rendered templates already include the family BOS marker, so only the raw
// fallback asks the tokenizer to add one. The error is a server-side condition (a decode-only vocab, or a tokenizer
// that failed to load) and must be surfaced: dropped, it yields an empty prompt and a generation from BOS alone.
func (lm *loadedModel) encode(prompt string) ([]int, error) {
	return lm.tk.Encode(prompt, lm.tmpl == nil)
}

// genErr filters a generation's terminal error (gen.Err()) down to what is worth surfacing to the client:
// context.Canceled (our own stop-string cancel, or a client disconnect) is a clean end, not a failure. A non-nil
// result becomes a 500, or an error SSE event mid-stream.
func genErr(err error) error {
	if err != nil && !errors.Is(err, context.Canceled) {
		return err
	}
	return nil
}

// drive runs the generation, applying stop strings and UTF-8 holdback, and calls onText with each newly completed
// text fragment. It returns the turn's turnResult and the terminal generation error (nil on a clean end; see genErr).
// It cancels its derived context on a stop-string hit to end generation.
//
// gens is the server's cancel-by-id registry; nil, or gr.id == "", skips registration. Every generation funnels
// through here or driveVL, so that bookkeeping lives here and not at each call site (see generations.go). jobs is the
// job store, nil only in a unit test calling drive directly; it is gated on the same gr.id != "" and shares the id,
// so the job and registry records join by id.
func (lm *loadedModel) drive(parent context.Context, gr genRequest, gens *stores.GenerationRegistry, js *jobs.Store, onText func(string)) (res turnResult, err error) {
	defer lm.logExpertCache(lm.expertCacheNow())
	var g *stores.Generation
	if gens != nil && gr.id != "" {
		// The admin cancel must land on parent itself, not on ctx below (drive's stop-string-derived child):
		// streamTokens tells an admin cancel apart from a stop-string hit by checking parent.Err(), because the
		// stop-string cancel only ever touches ctx. Registering ctx's cancel would silence the loop but leave
		// parent.Err() nil, making an admin cancel indistinguishable from a natural stop.
		var adminCancel context.CancelFunc
		parent, adminCancel = context.WithCancel(parent)
		defer adminCancel()
		g = gens.Register(gr.id, lm.name, adminCancel)
		defer gens.Remove(gr.id)
		wrapped := onText
		onText = func(t string) { g.AddTokens(1); wrapped(t) }
		if gr.think != nil { // a model that is still reasoning is making progress, not stuck
			wr := gr.think.onReasoning
			gr.think.onReasoning = func(t string) { g.AddTokens(1); wr(t) }
		}
	}
	if js != nil && gr.id != "" {
		j := js.GetOrCreate(gr.id, lm.name, gr.promptIDs)
		// The job's result text accumulates independently of whatever the caller's own onText does, so every job gets
		// one uniformly.
		var resultText strings.Builder
		wrapped := onText
		onText = func(t string) { resultText.WriteString(t); wrapped(t) }
		// Reads the named return values, so it sees the final res and err whichever return statement ran.
		defer func() {
			state := jobs.Done
			var errMsg string
			switch {
			case errors.Is(context.Cause(parent), errServerShutdown):
				state, errMsg = jobs.Interrupted, errServerShutdown.Error()
			case res.cancelReason != "":
				state = jobs.Cancelled
			case err != nil:
				state, errMsg = jobs.Failed, err.Error()
			}
			u := newUsage(len(gr.promptIDs), res.nComp, res.reused)
			js.FinishTurn(j, state, &u, gr.clock, errMsg, &jobs.Result{Content: resultText.String(), FinishReason: res.finish, StopSeq: res.stopHit})
		}()
	}
	ctx, cancel := context.WithCancel(parent)
	defer cancel()
	gr.clock.Start()
	st := lm.genEngine().Start(cost.WithCounter(ctx, gr.clock.CostCounter()), engine.Request{
		PromptIDs: gr.promptIDs, MaxTokens: gr.maxTokens, Sampling: decoder.SamplingParams(gr.sp), Masker: gr.masker, Lazy: gr.lazy,
		NoSpeculation: gr.conf != nil, Scope: gr.kvScope(),
	})
	defer st.Close()
	finish, n, stopHit := lm.streamTokens(parent, cancel, st.Tokens(), gr, st.Report, onText)
	gr.clock.End()
	rep := st.Report()
	if rep == nil { // an engine that keeps no report (the scripted seam): no cancelled finish, logprobs, reuse or error
		return turnResult{finish: finish, nComp: n, stopHit: stopHit, cancelReason: cancelledReason(g, parent, stopHit)}, nil
	}
	cr := cancelledReason(g, parent, stopHit)
	if cr != "" {
		finish = "cancelled"
	}
	logKVCopy(rep)
	return turnResult{finish: finish, nComp: n, logprobs: rep.Logprobs, stopHit: stopHit, reused: reuseOf(rep), cancelReason: cr}, genErr(rep.Err)
}

// cancelledReason reports the admin-cancel reason for this generation's finish, or "" if it was not an admin cancel.
// g is nil for an unregistered generation (gr.id == "" or gens == nil). Like streamTokens' "length" fallback, it
// reads parent (not the stop-string-derived ctx) for the external-cancel signal, and a stop-string hit always wins:
// the generation completed naturally at about the same moment, so the natural "stop" is reported rather than a race
// with an admin cancel that arrived too late to matter.
func cancelledReason(g *stores.Generation, parent context.Context, stopHit string) string {
	if g == nil || parent.Err() == nil || stopHit != "" {
		return ""
	}
	return g.Reason()
}

// driveVL is drive for a multimodal turn: it prefills gr.promptIDs with the projected vision features (from
// vi.features, invoked lazily) spliced in at the image placeholder spans, then streams the continuation through the
// same stop and UTF-8 machinery as drive. There is no warm-KV decoder.Session (multimodal opts out of that CPU-side
// prefix reuse), but on a resident backend GenerateVL and GenerateQwenVL reuse the resident GPU KV when the same
// image is resent, and vi.features is then never invoked. Its results and its gens and jobs parameters are drive's;
// see drive.
func (lm *loadedModel) driveVL(parent context.Context, gr genRequest, vi visionInput, gens *stores.GenerationRegistry, js *jobs.Store, onText func(string)) (res turnResult, err error) {
	defer lm.logExpertCache(lm.expertCacheNow())
	var g *stores.Generation
	if gens != nil && gr.id != "" {
		// See drive's identical block for why this must wrap parent, not the ctx derived below.
		var adminCancel context.CancelFunc
		parent, adminCancel = context.WithCancel(parent)
		defer adminCancel()
		g = gens.Register(gr.id, lm.name, adminCancel)
		defer gens.Remove(gr.id)
		wrapped := onText
		onText = func(t string) { g.AddTokens(1); wrapped(t) }
	}
	if js != nil && gr.id != "" {
		j := js.GetOrCreate(gr.id, lm.name, gr.promptIDs)
		var resultText strings.Builder
		wrapped := onText
		onText = func(t string) { resultText.WriteString(t); wrapped(t) }
		defer func() {
			state := jobs.Done
			var errMsg string
			switch {
			case errors.Is(context.Cause(parent), errServerShutdown):
				state, errMsg = jobs.Interrupted, errServerShutdown.Error()
			case res.cancelReason != "":
				state = jobs.Cancelled
			case err != nil:
				state, errMsg = jobs.Failed, err.Error()
			}
			u := newUsage(len(gr.promptIDs), res.nComp, res.reused)
			js.FinishTurn(j, state, &u, gr.clock, errMsg, &jobs.Result{Content: resultText.String(), FinishReason: res.finish, StopSeq: res.stopHit})
		}()
	}
	ctx, cancel := context.WithCancel(parent)
	defer cancel()
	// The image paths take every image's span; a one-media builder (GLM-OCR, audio) sets only the first.
	spans, grids := decoderSpans(vi.spans), vi.grids
	if len(spans) == 0 {
		spans, grids = []decoder.ImageSpan{{Pos: vi.imgPos, Len: vi.imgLen, Hash: vi.imgHash}}, [][3]int{vi.grid}
	}
	gr.clock.Start()
	st := lm.genEngine().Start(cost.WithCounter(ctx, gr.clock.CostCounter()), engine.Request{
		PromptIDs: gr.promptIDs, MaxTokens: gr.maxTokens, Sampling: decoder.SamplingParams(gr.sp),
		Media: vi.media(spans, grids, lm),
	})
	defer st.Close()
	if vi.replyPrefix != "" {
		onText(vi.replyPrefix)
	}
	finish, n, stopHit := lm.streamTokens(parent, cancel, st.Tokens(), gr, st.Report, onText)
	gr.clock.End()
	rep := st.Report()
	if rep == nil { // an engine that keeps no report (the scripted seam): no cancelled finish, logprobs, reuse or error
		return turnResult{finish: finish, nComp: n, stopHit: stopHit, cancelReason: cancelledReason(g, parent, stopHit)}, nil
	}
	if len(spans) > 1 && rep.ImgPrefillDecline != "" { // say why a multi-span turn prefilled on the CPU
		fmt.Fprintf(os.Stderr, "vision: %d image(s), %d spans; the resident image prefill declined (%s), so the CPU prefill ran and was uploaded\n", max(vi.images, 1), len(spans), rep.ImgPrefillDecline)
	}
	if vi.gemma4 { // log where this image turn decoded
		where := "cpu"
		if rep.DecodeResident {
			where = "resident"
		}
		prefill := "cpu"
		if rep.ImgPrefillResident { // the image turn's prefill ran on the resident too
			prefill = "resident"
		}
		fmt.Fprintf(os.Stderr, "vision: decoded %d tokens on the %s path (prefill %s)\n", n, where, prefill)
	}
	cr := cancelledReason(g, parent, stopHit)
	if cr != "" {
		finish = "cancelled"
	}
	logKVCopy(rep)
	return turnResult{finish: finish, nComp: n, logprobs: rep.Logprobs, stopHit: stopHit, reused: reuseOf(rep), cancelReason: cr}, genErr(rep.Err)
}

// media is vi as the engine's Media over spans and grids (driveVL's, with the one-media fallback), the kind chosen by
// vi's flags in this order: audio, Qwen, Gemma 4, causal, else bidirectional images.
func (vi visionInput) media(spans []decoder.ImageSpan, grids [][3]int, lm *loadedModel) *engine.Media {
	md := &engine.Media{Kind: engine.MediaImage, Spans: spans, Features: vi.features}
	switch {
	case vi.asr: // Qwen3-ASR, Voxtral: the encoder's embeddings replace the audio run, causal prefill, CPU decode
		md.Kind = engine.MediaAudio
	case vi.qwen: // Qwen2.5-VL, Qwen3-VL (with DeepStack sets), GLM-OCR
		g := lm.qwenGrid()
		md.Kind, md.Grids, md.Merge, md.ImageToken, md.DeepSets = engine.MediaQwen, grids, g.merge, g.imgTok, vi.deepSets
	case vi.gemma4:
		md.Kind = engine.MediaGemma4
	case vi.causal: // causal image tokens (Ministral 3, LFM2-VL), one span per run
		md.Kind = engine.MediaImageCausal
	}
	return md
}

// streamTokens consumes a token-id channel, applying stop strings (holding back a trailing partial stop match) and
// UTF-8 holdback, and calls onText with each newly completed fragment. It is the shared tail of every generation path
// (text via Session.Generate, multimodal via driveVL): the token source is orthogonal to this stop/stream logic.
// cancel ends the producing generation on a stop-string hit. Returns the finish reason ("stop" or "length"), the
// completion token count, and the stop string that was hit (empty unless a stop sequence ended the turn).
func (lm *loadedModel) streamTokens(parent context.Context, cancel context.CancelFunc, stream <-chan int, gr genRequest, report func() *genReport, onText func(string)) (string, int, string) {
	// Reasoning is taken out before the stop logic below, so a stop string is matched against the answer only (see
	// thinkOut): the reasoning goes to its own callback, and what the stop logic watches, holds back and emits is the
	// answer text. With no splitter (th == nil) the stop logic watches everything the model wrote.
	th := gr.think
	if o := gr.outcome; o != nil {
		emit := onText
		onText = func(s string) {
			if !o.answered && strings.TrimSpace(s) != "" {
				o.answered = true
			}
			emit(s)
		}
	}
	var ids []int
	// sb accumulates the decoded text incrementally instead of re-decoding the whole ids sequence every token, which
	// is O(n^2) in output length. Decode's per-token loop (tokenizer/sentencepiece.go) has no state that depends on
	// chunk boundaries, so DecodePiece(id) appended one token at a time is byte-identical to DecodeContinuation(ids)
	// computed fresh (TestDecodeContinuation_isIncrementallyAssociative in tokenizer/ proves it across a
	// multi-byte-emoji byte-fallback run). A strings.Builder, not `text += piece`: Go strings are immutable, so naive
	// concatenation is itself O(n) per append.
	var sb strings.Builder
	printed := 0
	finish := ""
	stopHit := ""
	stopping := false
	// step runs the stop logic over what sb holds now: look for a stop string in the not-yet-emitted tail, else emit up to
	// the safe boundary. Once per token, and once more when the splitter hands back what it held at the end.
	step := func() {
		text := sb.String() // O(1): a view over the Builder's buffer, not a copy
		// tail is text[printed:], the not-yet-emitted suffix, bounded however long the total output grows; scanning
		// it instead of the whole text keeps the loop linear. text[:printed] provably never contains a stop match:
		// stopTailHold holds back every suffix of the current text that could be a stop's prefix, so printed never
		// passes the start of a still-possible match, and a match that completes lies within tail. firstStop returns
		// an offset into whatever string it searches, so cut and end are translated back to absolute offsets (+=
		// printed) before use.
		tail := text[printed:]
		if cut, which, hit := firstStop(tail, gr.stopStrings); hit {
			cut += printed
			if cut > printed {
				onText(text[printed:cut])
			}
			finish, stopHit, stopping = "stop", which, true
			cancel()
			return
		}
		// Emit up to the UTF-8 boundary, but never past a trailing partial stop match: those bytes wait until the
		// next token proves them stop or not.
		end := completeUTF8(tail) + printed
		if safe := printed + len(tail) - stopTailHold(tail, gr.stopStrings); safe < end {
			end = safe
		}
		if end > printed {
			onText(text[printed:end])
			printed = end
		}
	}
	tr := traceFrom(parent) // -log-requests (reqlog.go); nil when it is off
	for id := range stream {
		gr.clock.FirstToken() // the turn's first token received; later calls keep it
		if stopping {
			continue // drain so the generation goroutine exits cleanly
		}
		if tr != nil && len(ids) == 0 {
			tr.firstToken()
		}
		ids = append(ids, id)
		if gr.outcome != nil {
			gr.outcome.ids = ids // read by a reasoning item that closes mid-stream (carriedMint)
		}
		// DecodePiece, not Decode or DecodeContinuation: these ids continue the prompt (no sequence-level
		// dummy-prefix strip), and appending each token's own piece is exactly what the whole-sequence decode does
		// internally.
		piece, _ := lm.tk.DecodePiece(id)
		if th != nil {
			piece = th.feed(piece)
		}
		sb.WriteString(piece)
		step()
	}
	if th != nil {
		// The splitter may still hold text (a partial tag, or whitespace that turned out to be answer): it goes through the
		// stop logic like any other answer text. After a stop hit only the reasoning side is flushed.
		if tail := th.finish(); tail != "" && !stopping {
			sb.WriteString(tail)
			step()
		}
	}
	if !stopping { // flush any held-back trailing bytes
		if text := sb.String(); len(text) > printed {
			onText(text[printed:])
		}
		// Compare against the effective budget, not the requested max_tokens: a resident context-cap clamp ends the
		// turn short of the request, and reporting that as a clean "stop" tells the client the model finished when it
		// was truncated. The report is safe to read here: the stream has closed.
		if len(ids) >= reportNow(report).BudgetFor(gr.maxTokens) {
			finish = "length"
		} else {
			finish = "stop" // EOS / turn-stop
		}
	}
	// A generation cut short from outside with no stop string is truncated, and must not be reported as a clean
	// finish. This reads parent, not ctx: the stop-string cancel fires on the derived context, so parent sees only
	// external cancellation: graceful shutdown (main.go's srvCancel() runs before Shutdown, so the client is still
	// connected and gets a 200 with a partial answer) or a client disconnect.
	//
	// "length" rather than an error, deliberately. It is truthful in both cases, where a 500 would be wrong for a
	// disconnect (nobody is reading), and the two are indistinguishable here: both arrive as a cancelled request
	// context. What matters is that a client never reads a truncated answer as complete, and "length" is the signal
	// it already acts on.
	if parent.Err() != nil && stopHit == "" {
		finish = "length"
	}
	if gr.conf != nil && gr.masker != nil {
		// goinfer_confidence: the stream has drained, so ids is everything this generation emitted (up to a stop
		// string, if one ended it early — the confidences then cover the part the client received).
		gr.conf.fields, gr.conf.err = gr.masker.FieldConfidence(ids)
	}
	if gr.outcome != nil {
		gr.outcome.ids, gr.outcome.done, gr.outcome.stopHit = ids, true, stopHit != ""
	}
	if tr != nil {
		tr.generated(lm, gr, reportNow(report), ids)
	}
	return finish, len(ids), stopHit
}

// logprobs maps goinfer's per-token SampleInfo to the OpenAI chat logprobs shape.
func (lm *loadedModel) logprobs(lps []decoder.SampleInfo) map[string]any {
	content := make([]any, 0, len(lps))
	for _, lp := range lps {
		top := make([]any, 0, len(lp.Top))
		for _, t := range lp.Top {
			top = append(top, map[string]any{"token": lm.tokenText(t.ID), "logprob": jsonLogprob(t.Logprob)})
		}
		content = append(content, map[string]any{
			"token": lm.tokenText(lp.ID), "logprob": jsonLogprob(lp.Logprob), "top_logprobs": top,
		})
	}
	return map[string]any{"content": content}
}

// logprobImpossible is the logprob reported for a token whose probability is exactly zero: OpenAI's own convention
// (-9999.0), because -Inf, the true value, cannot be written in JSON. A zero-probability candidate is normal, not an
// error: when the thinking budget forces the end-of-thinking token, that token has probability 1 and every other
// candidate, including the filler entries that pad the top-k list, has none. NaN is not mapped: it would be a bug,
// and writeJSON reports it.
const logprobImpossible = -9999.0

func jsonLogprob(x float64) float64 {
	if math.IsInf(x, -1) {
		return logprobImpossible
	}
	return x
}

func (lm *loadedModel) tokenText(id int) string { return string(lm.tk.TokenText(id)) }

// logKVCopy reports a generation that took KV rows from another conversation's slot, or whose copy was declined: the
// rows and the copy's wall time are the sharing gate's copy cost (docs/measurements/kv2-2026-10-14.md, "The copy's cost, and admission's").
// expertCacheNow reads the resident's MoE expert-cache counters, for logExpertCache to difference after the turn.
func (lm *loadedModel) expertCacheNow() (decoder.ExpertCacheStats, bool) {
	if lm.model == nil {
		return decoder.ExpertCacheStats{}, false
	}
	return lm.model.ExpertCacheStats()
}

// logExpertCache logs one turn's expert-cache lookups: the slots the cache was granted and the turn's hit rate, the figure
// that sets a paged MoE's decode rate (each miss is one expert copied over PCIe).
func (lm *loadedModel) logExpertCache(before decoder.ExpertCacheStats, ok bool) {
	if !ok {
		return
	}
	now, _ := lm.model.ExpertCacheStats()
	h, m := now.Hits-before.Hits, now.Misses-before.Misses
	if h+m == 0 {
		return
	}
	fmt.Fprintf(os.Stderr, "expert cache: %d slots/layer; this turn %d hits, %d misses (%.1f%% hit)\n", now.Slots, h, m, 100*float64(h)/float64(h+m))
}

func logKVCopy(rep *genReport) {
	if rep.PrefillShared > 0 {
		fmt.Fprintf(os.Stderr, "kv share: copied %d rows from another conversation in %.2f ms\n", rep.PrefillShared, float64(rep.KVCopy.Microseconds())/1000)
	}
	if rep.KVCopyDeclined != "" {
		fmt.Fprintf(os.Stderr, "kv share: a copy was declined (%s); the rows were prefilled\n", rep.KVCopyDeclined)
	}
}
