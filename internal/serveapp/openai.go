package serveapp

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"math"
	"math/rand"
	"net/http"
	"os"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/townsendmerino/aikit/audio"
	"github.com/townsendmerino/aikit/embed"
	"github.com/townsendmerino/aikit/encoder"
	"github.com/townsendmerino/aikit/vision"
	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/constrain"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/embeddinggemma2"
	"github.com/townsendmerino/goinfer/internal/clef"
	"github.com/townsendmerino/goinfer/internal/decide"
	"github.com/townsendmerino/goinfer/multimodal"
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

// loadedModel is one resident generative model and the per-model state a request needs: its tokenizer, chat template,
// stop ids, vocab, warm-KV sessions, and the admission (turns) that serializes its generations. The server registry
// holds them keyed by served name; requests to distinct models run in parallel.
type loadedModel struct {
	featCacheOnce sync.Once // vision_serve.go: the tower-output cache, made on first image
	featCache     *featureCache
	tk            *tokenizer.Tokenizer
	model         *decoder.Model
	tmpl          *chat.Template // nil → raw completion
	stopIDs       []int          // turn-stop token ids from the template
	eosIDs        []int
	vocab         int
	name          string // served id (reported by /v1/models, matched on the request model field)
	fp            string // model fingerprint (binds --session-dir snapshots)
	source        string // the model path as loadDecoder resolved it (an hf:/demo: reference → what it fetched); "" for an adapter
	adapter       string // compute-time LoRA adapter name (#7); "" = base model. Shares model with its base.
	spec          bool   // --spec ngram: lossless n-gram (prompt-lookup) speculative decode with adaptive depth
	specAdaptive  bool   // --spec-adaptive: speculate only while alone, join the batched decode otherwise
	// blockSpec is an attached pretrained block drafter (--drafter), nil when unused. It is attached once at load:
	// the weight upload is a per-process cost, and paying it per request is a several-fold slowdown (docs/spec/08-dspark-dflash.md).
	blockSpec *decoder.BlockSpec
	sessions  *sessionLRU // prefix-keyed KV reuse across requests
	// head is the entry's trained decision head, nil for label scoring. The decider POST /v1/systemone answers with
	// is built on first use, since its label tokens are resolved once per tokenizer.
	head *decide.Head
	// clef is the entry's Clef decision model, set when the model directory carries joint_head.safetensors. It
	// answers /v1/systemone instead of the label decider.
	clef        *clef.Model
	deciderOnce sync.Once
	decider     *decide.Decider
	deciderErr  error
	turns       admission // FIFO, context-aware granter of decode turns; a waiter can give up on disconnect or halt
	// sessMu guards sessions, the sessionLRU, which is not goroutine-safe (sessions.go): drive's acquire and checkin
	// and the background save, restore and demote goroutines take it. It is held around those operations only, not
	// across a generation: the session a generation is using is checked out instead (sessionLRU.busy), and the LRU
	// never hands out, evicts, demotes, saves or reads a busy session. That is what lets several generations of one
	// CPU model run at once (-max-concurrent) while nothing frees or moves memory under a running generation.
	sessMu sync.Mutex

	// tokenBytes is the constraint masker's token-to-bytes table (one entry per vocab id). It is a pure function of
	// (vocab, tokenizer), so it is built once per model, not per constrained request.
	tokenBytesOnce sync.Once
	tokenBytes     [][]byte
	// maxTokBytes is the byte length of the longest token in the vocab, computed once. It bounds tokenization cost: a
	// servable prompt is at most ctx tokens, so its text is at most ctx*maxTokBytes bytes, and a longer input cannot
	// fit and is rejected before the O(n) BPE runs.
	maxTokBytesOnce sync.Once
	maxTokBytes     int
	// queue bounds in-flight plus waiting requests (cap = concurrent + --max-queue); a request claims a slot before
	// its turn. nil = unbounded. Backpressure, not batching: a full queue returns 429 Retry-After.
	queue chan struct{}
	// concurrent is how many generations of this model may run at once (see setConcurrency); 1 = serialized.
	concurrent int

	// Vision tower (--vision): nil unless a multimodal model is loaded. A model with one is vision-capable: serve
	// accepts image content parts, runs them through preprocess, encoder and projector, and routes the turn to
	// GenerateVL.
	venc    *vision.Encoder
	vproj   *multimodal.Projector
	vcfg    vision.Config
	vimgTok int // image-soft-token id (the placeholder embed-by-vector overrides); -1 if unresolved

	// Qwen2.5-VL vision tower (nil for Gemma 3 or text). The merger is in the encoder (no separate projector);
	// preprocessing and m-RoPE are Qwen-specific, so the image path branches on qwenEnc != nil.
	qwenEnc     *vision.QwenVisionEncoder
	qwenDevMu   sync.Mutex // serializes the Qwen2.5-VL tower and its CPU fallback (device_fallback.go)
	qwenRequire bool       // -require-backend: a device-memory failure of the Qwen2.5-VL tower fails the request instead of falling back to the CPU
	qwenPP      multimodal.QwenPreprocessConfig
	qwenMerge   int // spatial_merge_size
	qwenImgTok  int // <|image_pad|> id

	// Qwen3.5+ vision tower (nil: not Qwen3.5, or no tower). It shares qwenPP/qwenMerge/qwenImgTok and the
	// GenerateQwenVL route with the Qwen2.5-VL path above, but loads lazily on the first image request (qwen3Tower):
	// every Qwen3.5 checkpoint carries one, so eager loading would tax every text-only user's startup and memory.
	qwen3 *qwen3Tower

	// GLM-OCR vision tower (nil: not GLM-OCR, or no tower). It shares qwenPP/qwenMerge/qwenImgTok and the
	// GenerateQwenVL route with the two Qwen paths above, but its tower, prompt block and template are GLM's
	// (glm_ocr_vision.go). Loaded lazily.
	glm *glmOcrTower

	// Ministral 3's Pixtral image path (nil: not Ministral 3, or no tower); see pixtral_vision.go.
	pixtral *pixtralTower

	// Gemma 4 vision tower (nil: not gemma4, or no tower). There is no separate projector: Gemma4Encoder.Forward
	// includes the embed_vision projection. gemma4MaxSoft is the checkpoint's vision_soft_tokens_per_image budget
	// (vision.Gemma4Preprocess's maxSoftTokens); the actual per-image token count is data-dependent and computed per
	// request via multimodal.Gemma4PooledTokens.
	gemma4Enc     *vision.Gemma4Encoder
	gemma4Tower   multimodal.Gemma4TowerAccelerator // nil: the CPU tower (docs/multimodal.md)
	gemma4MaxSoft int
	gemma4ImgTok  int // <|image|> id
	// Gemma 4 audio: set when the checkpoint has an audio_config. The tower loads on the first audio request
	// (gemma4AudioEncoder), so a server that never hears audio never pays its memory. gemma4AudioTok is <|audio|>.
	gemma4AudioDir string
	// Qwen3-ASR: the checkpoint directory (its audio encoder is in the same safetensors as the decoder), the
	// <|audio_pad|> id, and the encoder, loaded on the first clip.
	qwenASRDir      string
	qwenASRTok      int
	qwenASR         *audio.QwenASREncoder
	qwenASRErr      error
	qwenASROnce     sync.Once
	gemma4AudioTok  int
	gemma4AudioOnce sync.Once
	gemma4Audio     *audio.Gemma4AudioEncoder
	gemma4AudioErr  error
	// gemma4AudioDevice is "metal" when the tower's conformer blocks run on Metal (EmbeddingGemma 2's accelerator),
	// "" for the CPU; gemma4AudioAcc is that accelerator once built, nil if it declined (the CPU then, said once).
	gemma4AudioDevice string
	gemma4AudioAcc    embeddinggemma2.AudioAccelerator
}

// cachedTokenBytes returns the constraint masker's token-to-bytes table, built once per model and reused across
// constrained requests.
func (lm *loadedModel) cachedTokenBytes() [][]byte {
	lm.tokenBytesOnce.Do(func() {
		lm.tokenBytes = constrain.TokenBytes(lm.vocab, lm.tk.TokenText)
	})
	return lm.tokenBytes
}

// maxTokenBytes returns the byte length of the longest token in the vocab (at least 1), computed once and cached. It
// is the per-token byte ceiling of the tokenization guard.
func (lm *loadedModel) maxTokenBytes() int {
	lm.maxTokBytesOnce.Do(func() {
		m := 1
		for id := 0; id < lm.vocab; id++ {
			if n := len(lm.tk.TokenText(id)); n > m {
				m = n
			}
		}
		lm.maxTokBytes = m
	})
	return lm.maxTokBytes
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
		lm.queue = make(chan struct{}, n+cfg.maxQueue)
	}
	if lm.model == nil {
		return ""
	}
	return concurrencyLine(bannerFacts{resident: lm.model.ResidentActive(), kvSlots: lm.model.ResidentKVSlots(), concurrent: n,
		cpuBatched: cpuBatched}, cfg)
}

// audioCapable reports whether this model can take an audio clip (a Gemma 4 checkpoint with an audio tower, or a Qwen3-ASR checkpoint).
func (lm *loadedModel) audioCapable() bool { return lm.gemma4AudioDir != "" || lm.qwenASRDir != "" }

// visionCapable reports whether this model has a loaded vision tower.
func (lm *loadedModel) visionCapable() bool {
	return (lm.venc != nil && lm.vproj != nil) || lm.qwenEnc != nil || lm.qwen3 != nil || lm.glm != nil || lm.gemma4Enc != nil || lm.pixtral != nil
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
	if lm.queue != nil {
		select {
		case lm.queue <- struct{}{}:
		default:
			return false, ""
		}
	}
	if !lm.turns.enter(ctx, rec) {
		if lm.queue != nil {
			<-lm.queue
		}
		return false, ""
	}
	if haltState != nil {
		if hi := haltState(); hi != nil {
			lm.exit()
			return false, hi.reason
		}
	}
	return true, ""
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
		return fmt.Sprintf("model %q queue full (max %d queued); retry", lm.name, cap(lm.queue)-max(1, lm.concurrent))
	}
	return fmt.Sprintf("model %q queue full; retry", lm.name)
}

// exit releases the turn, then the queue slot (the reverse of enter's acquisition). Turns are interchangeable, so
// release needs no per-caller identity. drive itself checks the session back in.
func (lm *loadedModel) exit() {
	lm.turns.release()
	if lm.queue != nil {
		<-lm.queue
	}
}

type server struct {
	// Generative (decoder) registry: served name to model, empty when only an embedding model is served. Requests
	// route on the OpenAI `model` field through withModel. regMu guards the map structure (dynamic load and unload
	// mutate it concurrently with routing); a request holds the picked *loadedModel beyond the RLock, and unload
	// drains in-flight requests through liveness before freeing the model.
	regMu  sync.RWMutex
	models map[string]*loadedModel
	cfg    config // backend/quant/lora/kv/session-dir/allow-admin for admin loads

	// pulls serialises -web model downloads to one at a time (webui.go).
	pulls pullState
	// loads serialises -web model loads the same way (webui.go handleWebLoad).
	loads pullState

	// liveness tracks, per underlying *decoder.Model, the request holders (rw) and the number of registry entries
	// backed by it (refs), so unload can drain in-flight work before freeing native memory instead of racing it into
	// a use-after-free. A base and its compute-time adapters share one *decoder.Model and thus one entry. Guarded by
	// regMu; see liveness.go and docs/completed/task-admin-unload-drain.md.
	liveness map[*decoder.Model]*modelLiveness
	// draining is the set of served names whose entry has been unpublished but whose native memory
	// is not yet freed (the detached drain is still running). Surfaced by /health so an operator can
	// tell when a 202'd unload has actually reclaimed memory before reloading. Guarded by regMu.
	draining map[string]struct{}

	// Embedding (encoder) half — nil when only a generative model is served.
	// The encoder is goroutine-safe for concurrent Encode, so /v1/embeddings is
	// served without a mutex (the per-model mutex guards only the shared decoder).
	embed    encoder.Encoder
	embedTok *embed.Tokenizer // counts tokens for usage.prompt_tokens
	embedID  string
	embedDim int
	// embedMRLMin is the smallest width this embedder may be truncated to (0 = NOT truncatable,
	// the safe default). Only Matryoshka-trained models may be sliced; see resolveDimensions.
	embedMRLMin int
	// embedWidths, when set, is the exact set of widths the embedder was trained to be truncated to
	// (EmbeddingGemma 2: 768, 512, 256, 128); `dimensions` must be one of them. It takes precedence
	// over embedMRLMin's floor.
	embedWidths []int

	// Responses API (/v1/responses) state store for store/previous_response_id.
	responses *responseStore

	// gens is the cancel-by-id registry (docs/tasks/task-halt-2026-09.md), shared by every generation surface. Never
	// nil after newServer.
	gens *generationRegistry

	// jobs is the job registry (docs/tasks/task-work-queue-2026-09.md), shared by every generation surface the way
	// gens is. Never nil after newServer: every generation gets a job whether or not -job-dir is set; jobs.journal is
	// what is nil without it.
	jobs *jobStore

	// files and batches hold OpenAI's uploaded and assembled files and both dialects' batch records. Never nil after
	// newServer.
	files   *fileStore
	batches *batchStore

	// halted is the global halt state (docs/tasks/task-halt-2026-09.md); nil = running normally. See halt.go. The
	// zero value is nil, so newServer needs no init.
	halted atomic.Pointer[haltInfo]

	// swapGuardTripped and swapWatch are the serving-side swap tripwire (docs/tasks/task-never-swap-2026-09.md); see
	// swapguard.go. The zero values mean not armed yet: startSwapGuard sets swapWatch during newServer, and a nil
	// swapWatch (GOINFER_SWAP_GUARD=off) is a valid, permanent state.
	swapGuardTripped atomic.Bool
	swapWatch        *decoder.SwapWatch
}

// lookupLocked resolves a request's model name to a loaded entry: an exact served-name match, else the sole model
// when the name is empty on a single-model server; nil otherwise, and the handler answers the OpenAI-shaped 404. The
// CALLER MUST HOLD regMu (read or write). It is unexported and lock-requiring so it cannot be the route a handler
// uses: withModel (liveness.go) is the only way a request reaches a *loadedModel, because it also takes the liveness
// read lock that keeps the model alive for the request. A handler calling this directly would skip that lock.
func (s *server) lookupLocked(name string) *loadedModel {
	if lm, ok := s.models[name]; ok {
		return lm
	}
	// An omitted model on a single-model server routes to that model. A non-empty unknown name is rejected by the
	// caller (modelNotFound) rather than served, so a client that sent the wrong id gets an error naming what is
	// served instead of confident wrong output.
	if name == "" && len(s.models) == 1 {
		for _, lm := range s.models {
			return lm
		}
	}
	return nil
}

// resolveBodyCaps returns the (text, vision, embed, file) request-body caps in bytes. override > 0 sets all four
// verbatim; otherwise the text cap is derived from the largest served decoder's context window (ctx tokens x the
// longest token's byte length x 4 for JSON structure and escaping), floored at maxBodyBytes so a small-context model
// keeps a usable budget. The vision cap adds base64-image headroom on top (at least maxVisionBodyBytes).
//
// The embed cap is independent of both: /v1/embeddings is served by the encoder, which is not in s.models, so a
// decoder-derived cap would measure the wrong thing (see maxEmbedBodyBytes). fileCap is independent for the same
// reason (see maxBatchFileBytes). All four are reported on startup.
func (s *server) resolveBodyCaps(override int64) (textCap, visionCap, embedCap, fileCap int64) {
	textCap = maxBodyBytes // 4 MiB floor
	if override > 0 {
		textCap = override
	} else {
		var derived int64
		for _, lm := range s.modelList() {
			if lm.model == nil {
				continue
			}
			ctx := lm.model.Config().MaxPositions
			if ctx <= 0 {
				continue
			}
			if b := int64(ctx) * int64(lm.maxTokenBytes()) * 4; b > derived {
				derived = b
			}
		}
		if derived > textCap {
			textCap = derived
		}
	}
	visionCap = textCap + maxVisionBodyBytes // image data on top of the text budget
	embedCap = maxEmbedBodyBytes
	fileCap = maxBatchFileBytes
	if override > 0 {
		embedCap = override // an explicit -max-body-bytes governs every route
		fileCap = override
	}
	return textCap, visionCap, embedCap, fileCap
}

// modelNotFound writes the OpenAI-shaped 404 for an unknown model field.
func (s *server) modelNotFound(w http.ResponseWriter, name string) {
	writeErr(w, http.StatusNotFound, fmt.Sprintf("model %q not found (served: %s)", name, strings.Join(s.servedNames(), ", ")))
}

// modelByName is an exact registry lookup under the read lock. Unlike lookupLocked it has no single-model fallback,
// which is right for listing: the fallback would attach a decoder's paths to the embedding-model entry.
func (s *server) modelByName(name string) *loadedModel {
	s.regMu.RLock()
	defer s.regMu.RUnlock()
	return s.models[name]
}

// pathFields returns the resolved decode/prefill path fields for one served model, or nil when the
// name has no decoder (an embedding-only entry). ONE source for both /v1/models and /health, so the
// vendor extension and the operator surface can never disagree about what the server resolved to.
func (s *server) pathFields(name string) map[string]any {
	lm := s.modelByName(name)
	if lm == nil || lm.model == nil {
		return nil
	}
	batched, why := lm.model.PrefillPath()
	f := map[string]any{
		"decode_path":     lm.model.DecodePath(),
		"prefill_batched": batched,
		"prefill_path":    why,
		// Quant and resident size, so the web UI's Models tab can list what is resident (and size an Unload button)
		// from the one request it already makes.
		"quant":          lm.model.Quant(),
		"resident_bytes": lm.model.ResidentWeightBytes() + lm.model.ExtraResidentBytes(),
	}
	// context_window is the cap a text chat request on a base model is held to (contextWindow). Vision requests are
	// bounded by MaxPositions, which is never smaller, so this is the conservative number for a client to plan
	// against. Omitted when unknown, not published as 0.
	if ctx := lm.contextWindow(lm.adapter == ""); ctx > 0 {
		f["context_window"] = ctx
	}
	// vision reports whether image content parts are accepted, by the same visionCapable the vision path checks, so a
	// client can hide image input on a model that would refuse it.
	f["vision"] = lm.visionCapable()
	return f
}

// servedNames lists the loaded generative and embedding model ids, sorted.
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
	ResponseFormat      *respFormat     `json:"response_format"`
	// Confidence is the goinfer_confidence vendor extension (docs/tasks/task-constrained-confidence.md): with
	// response_format json_schema, the response carries each enum, boolean and integer field's confidence. Only the
	// routes that write it back set confidenceOK; prepare refuses the flag everywhere else, so no route can accept it
	// and silently drop the result.
	Confidence   bool `json:"goinfer_confidence"`
	confidenceOK bool
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
func (sm sampling) constrainsOutput() bool {
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
func (r chatReq) think() thinkRequest {
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

type usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
	// PrefillReusedTokens is a goinfer vendor extension (not part of the OpenAI schema; the same convention as
	// pathFields's decode_path and prefill_path on /v1/models): how many leading prompt tokens this generation's
	// prefill skipped because a prior turn left them resident (decoder.Generation.PrefillReused). 0 on a cold
	// prefill, on a path that does not track reuse, or when nothing was reused; the number alone cannot tell those
	// apart. An unrecognised JSON key, so any standard OpenAI client ignores it.
	PrefillReusedTokens int `json:"prefill_reused_tokens"`
}

// handleModels reports the served model(s) — the decoder and/or the embedding
// model (Open WebUI calls this to populate its model picker).
func (s *server) handleModels(w http.ResponseWriter, _ *http.Request) {
	created := time.Now().Unix()
	data := []map[string]any{}
	for _, name := range s.servedNames() {
		e := map[string]any{"id": name, "object": "model", "created": created, "owned_by": "goinfer"}
		// Vendor extension (goinfer-only, not in the OpenAI schema): the resolved compute paths. The resident decode
		// path and the batched prefill each fall back silently per model, so a client that cares about TTFT (batch
		// jobs, benchmarks) can read which one it got instead of inferring it from latency. Absent for encoder-only
		// entries. Unknown keys are ignored by the Go, Python and JS OpenAI clients, but a strict typed decoder in
		// another language may reject them: GET /health carries the same fields on a payload with no compatibility
		// contract, for operators who need a surface that cannot break a client.
		maps.Copy(e, s.pathFields(name))
		if d := s.decisionsField(name); d != nil {
			e["decisions"] = d // vendor extension, same convention: POST /v1/systemone's support for this entry
		}
		data = append(data, e)
	}
	writeJSON(w, http.StatusOK, map[string]any{"object": "list", "data": data})
}

// decisionsField is a /v1/models entry's decisions support: the route, kinds and template POST /v1/systemone answers
// with, and whether a calibration was loaded. nil for an entry that cannot answer (an encoder, or a compute-time
// adapter, which label scoring would answer with the base model).
func (s *server) decisionsField(name string) map[string]any {
	s.regMu.RLock()
	lm := s.models[name]
	s.regMu.RUnlock()
	if lm == nil || lm.model == nil || lm.adapter != "" {
		return nil
	}
	if lm.clef != nil {
		return map[string]any{"endpoint": "/v1/systemone", "route": routeClef, "kinds": []string{decide.KindNoul, decide.KindChoice, decide.KindScore},
			"calibrated": false, "confidence": clefConfidence}
	}
	if h := lm.head; h != nil {
		return map[string]any{"endpoint": "/v1/systemone", "route": decide.RouteHead, "kinds": h.Kinds(), "template": h.Template,
			"calibrated": h.Calibration != nil, "head": h.Name + " " + h.Version}
	}
	tmpl := s.cfg.decisionsTemplate
	if tmpl == "" {
		tmpl = decide.TemplateChat
	}
	return map[string]any{"endpoint": "/v1/systemone", "route": decide.RouteLabel, "kinds": []string{decide.KindNoul, decide.KindChoice, decide.KindScore},
		"template": tmpl, "calibrated": s.cfg.decisionsCal != ""}
}

func (s *server) handleChat(w http.ResponseWriter, r *http.Request) {
	var req chatReq
	if !decodeJSON(w, r, &req) {
		return
	}
	if err := validateN(req.N); err != nil {
		writeErr(w, http.StatusBadRequest, err.Error())
		return
	}
	// A request with no messages is a 400 naming the field, not a confident generation from a BOS-only prompt.
	if len(req.Messages) == 0 {
		writeErr(w, http.StatusBadRequest, "messages is required and must contain at least one message")
		return
	}
	// Multimodal: a message carrying an image_url part routes to the vision path. Images in earlier messages are replaced by a note first
	// (image_history.go): the vision path takes one image, and a chat client resends its whole history.
	noteOmittedImages(w.Header(), omitChatHistoryImages(req.Messages))
	imgs, ierr := chatImages(req.Messages)
	if ierr != nil {
		writeErr(w, http.StatusBadRequest, ierr.Error())
		return
	}
	if len(imgs) > 0 {
		// serveVisionChat neither renders nor parses tools, so a tools+image request would silently drop the tools:
		// fail loudly instead.
		if len(req.Tools) > 0 && toolChoiceMode(req.ToolChoice) != "none" {
			writeErr(w, http.StatusBadRequest, "tools are not supported together with image inputs; send images or tools, not both")
			return
		}
		// The vision stream has no logprobs field, so a stream:true request would silently drop them: the text route's rule (400).
		if req.Stream && req.sampling.Logprobs {
			writeErr(w, http.StatusBadRequest, "logprobs is not supported together with stream:true")
			return
		}
		s.serveVisionChat(w, r, req, imgs)
		return
	}
	if len(req.Tools) > 0 && toolChoiceMode(req.ToolChoice) != "none" {
		s.handleChatTools(w, r, req)
		return
	}
	s.withModel(w, req.Model, func(lm *loadedModel) { s.serveChatText(w, r, req, lm) })
}

// serveChatText runs the text (non-vision, non-tool) chat generation. Reached ONLY through withModel,
// so the model's liveness RLock is held for this whole call — unload cannot free it out from under us.
func (s *server) serveChatText(w http.ResponseWriter, r *http.Request, req chatReq, lm *loadedModel) {
	// Reject an over-context prompt before tokenizing it, so a multi-MiB body costs a byte-length comparison, not a
	// full BPE pass.
	if err := lm.promptTooLargeForContext(chatInputBytes(req.Messages)); err != nil {
		writeErr(w, http.StatusBadRequest, err.Error())
		return
	}
	ts, terr := s.resolveThink(req.think())
	if terr != nil {
		writeErr(w, http.StatusBadRequest, terr.Error())
		return
	}
	tm := lm.templateFor(ts)
	if req.sampling.constrainsOutput() {
		tm = lm.constrainedTemplate(ts)
	}
	system, turns := messagesToTurns(req.Messages)
	ids, err := lm.promptForT(tm, system, turns)
	if err != nil {
		writeServerErr(w, "encode: "+err.Error())
		return
	}
	req.sampling.confidenceOK = true // written back below
	gr, err := lm.prepare(req.sampling, ids, lm.residentPath())
	if err != nil {
		writeErr(w, prepareErrStatus(err), err.Error())
		return
	}
	if req.Stream && req.Logprobs { // the stream path has no logprobs field — reject rather than silently drop
		writeErr(w, http.StatusBadRequest, "logprobs is not supported together with stream:true")
		return
	}
	if !lm.enter(w, r, admissionRecord{promptIDs: gr.promptIDs}, s.haltState) {
		return
	}
	defer lm.exit()
	id := "chatcmpl-" + reqID()
	gr.id = id // drive/driveVL register the generation for cancel-by-id
	created := time.Now().Unix()

	if req.Stream {
		ss, ok := sseStart(w)
		if !ok {
			return
		}
		role := chatChunk(id, created, lm.name, delta{Role: "assistant"}, nil)
		sseSend(ss, role)
		// Nothing else is sent between here and the first token, and on CPU that gap is the whole prefill: a
		// heartbeat keeps a client's idle timeout from firing during it.
		stopBeat := sseHeartbeat(ss)
		s.routeThink(lm, &gr, tm, turns, ts, func(t string) {
			sseSend(ss, chatChunk(id, created, lm.name, delta{ReasoningContent: t}, nil))
		})
		finish, nComp, _, _, reused, cancelReason, gerr := lm.drive(r.Context(), gr, s.gens, s.jobs, func(t string) {
			sseSend(ss, chatChunk(id, created, lm.name, delta{Content: t}, nil))
		})
		stopBeat()
		if gerr != nil {
			sseErr(ss, "generation failed: "+gerr.Error())
			sseDone(ss)
			return
		}
		sseSend(ss, chatChunk(id, created, lm.name, delta{}, &finish))
		if gr.conf != nil {
			sseSend(ss, map[string]any{"id": id, "goinfer_confidence": gr.conf.payload()})
		}
		if cancelReason != "" {
			// One final SSE event naming the reason, so a client cannot mistake this for a natural stop even though
			// finish_reason is already "cancelled" above.
			sseSend(ss, map[string]any{"goinfer_cancelled": map[string]any{"id": id, "reason": cancelReason}})
		}
		sendUsage(ss, req.StreamOptions, id, created, lm.name,
			usage{PromptTokens: len(gr.promptIDs), CompletionTokens: nComp, TotalTokens: len(gr.promptIDs) + nComp, PrefillReusedTokens: reused})
		sseDone(ss)
		return
	}

	var sb, rb strings.Builder
	s.routeThink(lm, &gr, tm, turns, ts, func(t string) { rb.WriteString(t) })
	finish, nComp, lps, _, reused, cancelReason, gerr := lm.drive(r.Context(), gr, s.gens, s.jobs, func(t string) { sb.WriteString(t) })
	if gerr != nil {
		writeServerErr(w, "generation failed: "+gerr.Error())
		return
	}
	if cancelReason != "" {
		writeErr(w, statusCancelled, "generation cancelled: "+cancelReason)
		return
	}
	msg := map[string]any{"role": "assistant", "content": sb.String()}
	if rb.Len() > 0 {
		msg["reasoning_content"] = rb.String()
	}
	choice := map[string]any{
		"index":         0,
		"message":       msg,
		"finish_reason": finish,
	}
	if req.Logprobs {
		choice["logprobs"] = lm.logprobs(lps)
	}
	resp := map[string]any{
		"id": id, "object": "chat.completion", "created": created, "model": lm.name,
		"choices": []any{choice},
		"usage":   usage{PromptTokens: len(gr.promptIDs), CompletionTokens: nComp, TotalTokens: len(gr.promptIDs) + nComp, PrefillReusedTokens: reused},
	}
	if gr.conf != nil {
		resp["goinfer_confidence"] = gr.conf.payload()
	}
	writeJSON(w, http.StatusOK, resp)
}

func (s *server) handleCompletions(w http.ResponseWriter, r *http.Request) {
	var req completionReq
	if !decodeJSON(w, r, &req) {
		return
	}
	if err := validateN(req.N); err != nil {
		writeErr(w, http.StatusBadRequest, err.Error())
		return
	}
	s.withModel(w, req.Model, func(lm *loadedModel) { s.serveCompletion(w, r, req, lm) })
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

// serveCompletion runs a /v1/completions generation. Reached ONLY through withModel (liveness RLock held).
func (s *server) serveCompletion(w http.ResponseWriter, r *http.Request, req completionReq, lm *loadedModel) {
	if req.Logprobs != nil { // legacy Completions logprobs (integer) is unimplemented: reject cleanly
		writeErr(w, http.StatusBadRequest, "logprobs is not supported on /v1/completions; use /v1/chat/completions with logprobs:true")
		return
	}
	prompt, perr := singlePromptString(req.Prompt)
	if perr != nil { // a []int token-id prompt or a batch array used to decode to "" → BOS-only 200
		writeErr(w, http.StatusBadRequest, perr.Error())
		return
	}
	if err := lm.promptTooLargeForContext(len(prompt)); err != nil { // reject before tokenizing
		writeErr(w, http.StatusBadRequest, err.Error())
		return
	}
	ids, err := lm.tk.Encode(prompt, true) // raw completion: tokenizer adds BOS
	if err != nil {
		writeErr(w, http.StatusInternalServerError, "encode: "+err.Error())
		return
	}
	req.sampling.confidenceOK = true // written back below
	gr, err := lm.prepare(req.sampling, ids, lm.residentPath())
	if err != nil {
		writeErr(w, prepareErrStatus(err), err.Error())
		return
	}
	if !lm.enter(w, r, admissionRecord{promptIDs: gr.promptIDs}, s.haltState) {
		return
	}
	defer lm.exit()
	id := "cmpl-" + reqID()
	gr.id = id
	created := time.Now().Unix()

	if req.Stream {
		ss, ok := sseStart(w)
		if !ok {
			return
		}
		// Same prefill silence as the chat stream: heartbeat until the first token.
		stopBeat := sseHeartbeat(ss)
		finish, nComp, _, _, reused, cancelReason, gerr := lm.drive(r.Context(), gr, s.gens, s.jobs, func(t string) {
			sseSend(ss, completionChunk(id, created, lm.name, t, nil))
		})
		stopBeat()
		if gerr != nil {
			sseErr(ss, "generation failed: "+gerr.Error())
			sseDone(ss)
			return
		}
		sseSend(ss, completionChunk(id, created, lm.name, "", &finish))
		if gr.conf != nil {
			sseSend(ss, map[string]any{"id": id, "goinfer_confidence": gr.conf.payload()})
		}
		if cancelReason != "" {
			sseSend(ss, map[string]any{"goinfer_cancelled": map[string]any{"id": id, "reason": cancelReason}})
		}
		sendUsage(ss, req.StreamOptions, id, created, lm.name,
			usage{PromptTokens: len(gr.promptIDs), CompletionTokens: nComp, TotalTokens: len(gr.promptIDs) + nComp, PrefillReusedTokens: reused})
		sseDone(ss)
		return
	}
	var sb strings.Builder
	finish, nComp, _, _, reused, cancelReason, gerr := lm.drive(r.Context(), gr, s.gens, s.jobs, func(t string) { sb.WriteString(t) })
	if gerr != nil {
		writeServerErr(w, "generation failed: "+gerr.Error())
		return
	}
	if cancelReason != "" {
		writeErr(w, statusCancelled, "generation cancelled: "+cancelReason)
		return
	}
	resp := map[string]any{
		"id": id, "object": "text_completion", "created": created, "model": lm.name,
		"choices": []any{map[string]any{"index": 0, "text": sb.String(), "finish_reason": finish}},
		"usage":   usage{PromptTokens: len(gr.promptIDs), CompletionTokens: nComp, TotalTokens: len(gr.promptIDs) + nComp, PrefillReusedTokens: reused},
	}
	if gr.conf != nil {
		resp["goinfer_confidence"] = gr.conf.payload()
	}
	writeJSON(w, http.StatusOK, resp)
}

// genRequest is a prepared generation: prompt ids, sampling params (with the
// constraint masker already wired into LogitProcessor), limits, and stop strings.
type genRequest struct {
	promptIDs   []int
	sp          decoder.SamplingParams
	maxTokens   int
	stopStrings []string
	// id, when set, registers this generation in the server's cancel-by-id registry for the duration of
	// drive/driveVL. "" (the zero value) skips registration, so a caller not wired for cancellation (or a test
	// calling drive/driveVL directly) is unaffected.
	id     string
	masker *constrain.Masker // set on constrained requests (response_format / tool grammar); enables grammar-spec
	// conf, when the request asked for goinfer_confidence, receives the per-field confidences once streamTokens
	// has drained the generation. Its presence also keeps drive on plain constrained decode: the grammar-fused
	// speculative path drives the masker without Process, which is where the capture records.
	conf *confResult
	// think, when set, routes the decoded text through a reasoning splitter (think.go): content goes to drive's onText,
	// reasoning to think.onReasoning. nil = pass text through untouched.
	think *thinkOut
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
	sp := decoder.SamplingParams{
		Temperature: deref(sm.Temperature, 1.0),
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
	gr := genRequest{
		promptIDs:   promptIDs,
		sp:          sp,
		maxTokens:   deref(maxTok, defaultMaxTokens),
		stopStrings: parseStop(sm.Stop),
	}
	// Reject a prompt that does not fit the context window, then clamp max_tokens to the room left (clampMaxTokens).
	if lm.model != nil {
		ctx := lm.contextWindow(residentPath)
		if err := contextLengthErrorFor(len(promptIDs), ctx, lm.modelContextWindow()); err != nil {
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
		eos := append(append([]int(nil), lm.eosIDs...), lm.stopIDs...)
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

// contextLengthError rejects a prompt that alone fills or exceeds the model's context window. Unchecked, a multi-MiB
// body tokenizes to ~1M ids and preallocates tens of GiB of KV, and within memory a position past the trained context
// drives out-of-range RoPE and returns plausible garbage under HTTP 200. ctx <= 0 (unknown) never rejects.
func contextLengthError(promptLen, ctx int) error {
	if ctx > 0 && promptLen >= ctx {
		return fmt.Errorf("prompt is %d tokens but the model's context window is %d (context_length_exceeded)", promptLen, ctx)
	}
	return nil
}

// contextLengthErrorFor is contextLengthError plus the remedy, for when the window that rejected the prompt is the
// server's resident KV capacity, not the model's own (ctx < modelWindow): "context window is 8192" reads as the
// model's limit, so the client compacts instead of the operator raising -ctx. When the model's own window is the
// limit, -ctx cannot help and the message is unchanged.
func contextLengthErrorFor(promptLen, ctx, modelWindow int) error {
	err := contextLengthError(promptLen, ctx)
	if err == nil || modelWindow <= ctx {
		return err
	}
	return fmt.Errorf("%w; this is the server's GPU context (-ctx), not the model's limit (%d tokens): restart goinfer-serve with -ctx %d or more (up to %d) to accept it",
		err, modelWindow, promptLen+1, modelWindow)
}

// seedOrRandom returns the request's seed, or a fresh random seed when absent. OpenAI's contract is that an omitted
// seed varies output run to run (best-of-N, "regenerate" and agent retry-for-diversity depend on it). A supplied
// seed, including 0, is honored verbatim for reproducibility.
func seedOrRandom(seed *int64) int64 {
	if seed != nil {
		return *seed
	}
	return rand.Int63()
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
// reasoning-class models, and agent harnesses follow. It is an alias and nothing more: same position, same
// last-one-wins precedence two "system" messages already have, no new concept downstream. It is not a general
// unknown-role tolerance; every other unrecognized role keeps the default arm below, which demotes it to a user turn
// (for "developer" that delivered a harness's entire agent scaffold as the user's first message).
func messagesToTurns(msgs []chatMessage) (string, []chat.Turn) {
	var system string
	var turns []chat.Turn
	for _, m := range msgs {
		switch m.Role {
		case "system", "developer":
			system = m.text()
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
			turns = append(turns, chat.Turn{Role: "assistant", Content: m.text(), ToolCalls: tc, Reasoning: reasoning})
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
// text fragment. It returns the finish reason ("stop", "length" or "cancelled"), the completion token count, the
// per-token logprobs (non-stream), the stop string that was hit (empty unless a stop sequence ended the turn), how
// many leading prompt tokens the prefill skipped through resident or session reuse (decoder.Generation.PrefillReused;
// 0 when nothing was reused or the path does not track it), the admin-cancel reason (empty unless an admin cancel,
// not a client disconnect or shutdown, ended the turn), and the terminal generation error (nil on a clean end; see
// genErr). It cancels its derived context on a stop-string hit to end generation.
//
// gens is the server's cancel-by-id registry; nil, or gr.id == "", skips registration. Every generation funnels
// through here or driveVL, so that bookkeeping lives here and not at each call site (see generations.go). jobs is the
// job store, nil only in a unit test calling drive directly; it is gated on the same gr.id != "" and shares the id,
// so the job and registry records join by id.
func (lm *loadedModel) drive(parent context.Context, gr genRequest, gens *generationRegistry, jobs *jobStore, onText func(string)) (finish string, nComp int, logprobs []decoder.SampleInfo, stopHitOut string, prefillReused int, cancelReason string, err error) {
	var g *generation
	if gens != nil && gr.id != "" {
		// The admin cancel must land on parent itself, not on ctx below (drive's stop-string-derived child):
		// streamTokens tells an admin cancel apart from a stop-string hit by checking parent.Err(), because the
		// stop-string cancel only ever touches ctx. Registering ctx's cancel would silence the loop but leave
		// parent.Err() nil, making an admin cancel indistinguishable from a natural stop.
		var adminCancel context.CancelFunc
		parent, adminCancel = context.WithCancel(parent)
		defer adminCancel()
		g = gens.register(gr.id, lm.name, adminCancel)
		defer gens.remove(gr.id)
		wrapped := onText
		onText = func(t string) { g.tokens.Add(1); wrapped(t) }
		if gr.think != nil { // a model that is still reasoning is making progress, not stuck
			wr := gr.think.onReasoning
			gr.think.onReasoning = func(t string) { g.tokens.Add(1); wr(t) }
		}
	}
	if jobs != nil && gr.id != "" {
		j := jobs.getOrCreate(gr.id, lm.name, gr.promptIDs)
		// The job's result text accumulates independently of whatever the caller's own onText does, so every job gets
		// one uniformly.
		var resultText strings.Builder
		wrapped := onText
		onText = func(t string) { resultText.WriteString(t); wrapped(t) }
		// Reads the named return values, so it sees the final finish, nComp, prefillReused, cancelReason and err
		// whichever return statement ran.
		defer func() {
			state := jobDone
			var errMsg string
			switch {
			case cancelReason != "":
				state = jobCancelled
			case err != nil:
				state, errMsg = jobFailed, err.Error()
			}
			jobs.finish(j, state, &usage{
				PromptTokens: len(gr.promptIDs), CompletionTokens: nComp,
				TotalTokens: len(gr.promptIDs) + nComp, PrefillReusedTokens: prefillReused,
			}, errMsg, &jobResult{Content: resultText.String(), FinishReason: finish, StopSeq: stopHitOut})
		}()
	}
	ctx, cancel := context.WithCancel(parent)
	defer cancel()
	var stream <-chan int
	var gen *decoder.Generation

	// GPU-resident models take the stateless path. Generate engages the resident DecodeRunner only with no session
	// commit and no prefix reuse (decoder/model.go's useGPU): the resident's KV lives on the GPU and a session's
	// prefix-reuse cache is CPU-side, so the two cannot both be the source of truth, and going through a session
	// would silently drop the request to the staged CPU path. The OpenAI API is stateless (the client resends the
	// whole conversation), so sessions here are only a TTFT optimisation and trading prefix reuse for resident decode
	// is safe.
	//
	// N-gram speculative decode does mix with residency: the drafter is pure Go and the verify is the resident
	// batched ForwardN. genNgramInto claims the shared resident KV (resBusy) like Generate. Constrained and tool
	// requests (grammar masker) keep plain resident Generate; a validation error (sampler not on the spec path) falls
	// back to plain Generate before the KV is touched, so the fallback is exact.
	//
	// Adapter (compute-time LoRA) models must not take this path: the LoRA is applied only through the session
	// binding (sessionLRU.bindAdapter, Session.UseAdapter), and the stateless Generate runs on a fresh cache with no
	// LoRA, so it would silently return base-model output. Adapter requests take the session path below. There a
	// session's first turn can still run on the resident GPU path (generateInto's useGPU, decoder/model.go); a later
	// turn continuing off the reused warm prefix drops to CPU, since compute-time LoRA is not wired into the resident
	// prefix-reuse path.
	if lm.model.ResidentActive() && lm.adapter == "" {
		if lm.blockSpec != nil && gr.masker == nil {
			// Pretrained block drafter (--drafter): a whole block per round, verified in one batched pass.
			// GenerateStream validates the sampler and returns an error before touching any state, so a request with
			// temperature or penalties falls back exactly.
			if s, gn, err := lm.blockSpec.GenerateStream(ctx, gr.promptIDs, gr.maxTokens, gr.sp); err == nil {
				stream, gen = s, gn
			} else {
				stream, gen = lm.model.Generate(ctx, gr.promptIDs, gr.maxTokens, gr.sp)
			}
		} else if lm.spec && gr.masker == nil {
			if s, gn, err := lm.model.GenerateNgramSpeculativeAdaptive(ctx, gr.promptIDs, gr.maxTokens, &decoder.NgramDrafter{}, &decoder.AdaptiveDepth{MaxDraft: 8}, gr.sp); err == nil {
				stream, gen = s, gn
			} else {
				stream, gen = lm.model.Generate(ctx, gr.promptIDs, gr.maxTokens, gr.sp)
			}
		} else {
			stream, gen = lm.model.Generate(ctx, gr.promptIDs, gr.maxTokens, gr.sp)
		}
		finish, n, stopHit := lm.streamTokens(parent, cancel, stream, gr, gen, onText)
		cr := cancelledReason(g, parent, stopHit)
		if cr != "" {
			finish = "cancelled"
		}
		return finish, n, gen.Logprobs, stopHit, gen.PrefillReused, cr, genErr(gen.Err())
	}

	// Reuse the KV of whichever cached session already holds this prompt as a prefix (continuing chat, agent loop):
	// only the new suffix is prefilled. The session is checked out for this generation: sessMu covers the LRU
	// operation only, and checkin runs once streamTokens has drained the stream (the generation goroutine reconciles
	// the session before closing it), so the next acquire, or a background demote or save, sees a consistent session.
	lm.sessMu.Lock()
	sess := lm.sessions.acquire(gr.promptIDs)
	lm.sessMu.Unlock()
	defer func() {
		lm.sessMu.Lock()
		lm.sessions.checkin(sess)
		lm.sessMu.Unlock()
	}()
	switch {
	case gr.conf != nil:
		// goinfer_confidence needs plain constrained decode: the speculative paths below drive the masker without
		// Process (grammar-fused) or batch verify positions past what is accepted (n-gram), and the capture records
		// in Process, one call per emitted position.
		stream, gen = sess.Generate(ctx, gr.promptIDs, gr.maxTokens, gr.sp)
	case lm.spec && gr.masker != nil && gr.sp.Temperature == 0:
		// Constrained request (response_format or tool grammar), greedy: grammar-fused speculative decode. A
		// RouterDrafter fuses the grammar's forced byte-run (structural tokens) with an n-gram copy of free values
		// that echo the context; the masked verify keeps output identical to constrained Generate, and a miss costs
		// almost nothing. Any validation error, which precedes touching the session cache, falls back to plain
		// constrained decode.
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
		// Lossless n-gram speculative decode with adaptive depth. Falls back to plain Generate when the request's
		// sampler is not supported on the spec path (repetition, presence or frequency penalties, logit bias, or a
		// constrained or tool LogitProcessor with temperature>0); the validation error precedes touching the session
		// cache, so the fallback is exact.
		var err error
		stream, gen, err = sess.GenerateNgramSpeculativeAdaptive(ctx, gr.promptIDs, gr.maxTokens, &decoder.NgramDrafter{}, &decoder.AdaptiveDepth{MaxDraft: 8}, gr.sp)
		if err != nil {
			stream, gen = sess.Generate(ctx, gr.promptIDs, gr.maxTokens, gr.sp)
		}
	default:
		stream, gen = sess.Generate(ctx, gr.promptIDs, gr.maxTokens, gr.sp)
	}
	finish, n, stopHit := lm.streamTokens(parent, cancel, stream, gr, gen, onText)
	cr := cancelledReason(g, parent, stopHit)
	if cr != "" {
		finish = "cancelled"
	}
	return finish, n, gen.Logprobs, stopHit, gen.PrefillReused, cr, genErr(gen.Err())
}

// cancelledReason reports the admin-cancel reason for this generation's finish, or "" if it was not an admin cancel.
// g is nil for an unregistered generation (gr.id == "" or gens == nil). Like streamTokens' "length" fallback, it
// reads parent (not the stop-string-derived ctx) for the external-cancel signal, and a stop-string hit always wins:
// the generation completed naturally at about the same moment, so the natural "stop" is reported rather than a race
// with an admin cancel that arrived too late to matter.
func cancelledReason(g *generation, parent context.Context, stopHit string) string {
	if g == nil || parent.Err() == nil || stopHit != "" {
		return ""
	}
	return g.reason()
}

// driveVL is drive for a multimodal turn: it prefills gr.promptIDs with the projected vision features (from
// vi.features, invoked lazily) spliced in at the image placeholder spans, then streams the continuation through the
// same stop and UTF-8 machinery as drive. There is no warm-KV decoder.Session (multimodal opts out of that CPU-side
// prefix reuse), but on a resident backend GenerateVL and GenerateQwenVL reuse the resident GPU KV when the same
// image is resent, and vi.features is then never invoked. Its results and its gens and jobs parameters are drive's;
// see drive.
func (lm *loadedModel) driveVL(parent context.Context, gr genRequest, vi visionInput, gens *generationRegistry, jobs *jobStore, onText func(string)) (finish string, nComp int, logprobs []decoder.SampleInfo, stopHitOut string, prefillReused int, cancelReason string, err error) {
	var g *generation
	if gens != nil && gr.id != "" {
		// See drive's identical block for why this must wrap parent, not the ctx derived below.
		var adminCancel context.CancelFunc
		parent, adminCancel = context.WithCancel(parent)
		defer adminCancel()
		g = gens.register(gr.id, lm.name, adminCancel)
		defer gens.remove(gr.id)
		wrapped := onText
		onText = func(t string) { g.tokens.Add(1); wrapped(t) }
	}
	if jobs != nil && gr.id != "" {
		j := jobs.getOrCreate(gr.id, lm.name, gr.promptIDs)
		var resultText strings.Builder
		wrapped := onText
		onText = func(t string) { resultText.WriteString(t); wrapped(t) }
		defer func() {
			state := jobDone
			var errMsg string
			switch {
			case cancelReason != "":
				state = jobCancelled
			case err != nil:
				state, errMsg = jobFailed, err.Error()
			}
			jobs.finish(j, state, &usage{
				PromptTokens: len(gr.promptIDs), CompletionTokens: nComp,
				TotalTokens: len(gr.promptIDs) + nComp, PrefillReusedTokens: prefillReused,
			}, errMsg, &jobResult{Content: resultText.String(), FinishReason: finish, StopSeq: stopHitOut})
		}()
	}
	ctx, cancel := context.WithCancel(parent)
	defer cancel()
	var stream <-chan int
	var gen *decoder.Generation
	// The image paths take every image's span; a one-media builder (GLM-OCR, audio) sets only the first.
	spans, grids := vi.spans, vi.grids
	if len(spans) == 0 {
		spans, grids = []decoder.ImageSpan{{Pos: vi.imgPos, Len: vi.imgLen, Hash: vi.imgHash}}, [][3]int{vi.grid}
	}
	if vi.asr { // Qwen3-ASR: the encoder's embeddings replace the <|audio_pad|> run, causal prefill, CPU decode
		stream, gen = lm.model.GenerateAudio(ctx, gr.promptIDs, vi.imgPos, vi.imgLen, vi.features, gr.maxTokens, gr.sp)
	} else if vi.qwen && vi.deepSets > 0 { // Qwen3-VL: split the flat features into the merged rows and the DeepStack sets
		total := 0
		for _, sp := range spans {
			total += sp.Len
		}
		n := total * lm.model.Config().HiddenDim
		stream, gen = lm.model.GenerateQwenVLDeepstackSpans(ctx, gr.promptIDs, spans, func() ([]float32, [][]float32, error) {
			flat, err := vi.features()
			if err != nil {
				return nil, nil, err
			}
			deep := make([][]float32, vi.deepSets)
			for l := range deep {
				deep[l] = flat[(l+1)*n : (l+2)*n]
			}
			return flat[:n], deep, nil
		}, grids, lm.qwenMerge, lm.qwenImgTok, gr.maxTokens, gr.sp)
	} else if vi.qwen {
		stream, gen = lm.model.GenerateQwenVLDeepstackSpans(ctx, gr.promptIDs, spans, func() ([]float32, [][]float32, error) {
			f, err := vi.features()
			return f, nil, err
		}, grids, lm.qwenMerge, lm.qwenImgTok, gr.maxTokens, gr.sp)
	} else if vi.gemma4 {
		stream, gen = lm.model.GenerateGemma4VLSpans(ctx, gr.promptIDs, spans, vi.features, gr.maxTokens, gr.sp)
	} else if vi.pixtral { // Ministral 3's image tokens are causal, one span per merged row
		stream, gen = lm.model.GenerateVLCausalSpans(ctx, gr.promptIDs, spans, vi.features, gr.maxTokens, gr.sp)
	} else {
		stream, gen = lm.model.GenerateVLSpans(ctx, gr.promptIDs, spans, vi.features, gr.maxTokens, gr.sp)
	}
	finish, n, stopHit := lm.streamTokens(parent, cancel, stream, gr, gen, onText)
	if len(spans) > 1 && gen.ImgPrefillDecline != "" { // say why a multi-span turn prefilled on the CPU
		fmt.Fprintf(os.Stderr, "vision: %d image(s), %d spans; the resident image prefill declined (%s), so the CPU prefill ran and was uploaded\n", max(vi.images, 1), len(spans), gen.ImgPrefillDecline)
	}
	if vi.gemma4 { // log where this image turn decoded
		where := "cpu"
		if gen.DecodeResident {
			where = "resident"
		}
		prefill := "cpu"
		if gen.ImgPrefillResident { // the image turn's prefill ran on the resident too
			prefill = "resident"
		}
		fmt.Fprintf(os.Stderr, "vision: decoded %d tokens on the %s path (prefill %s)\n", n, where, prefill)
	}
	cr := cancelledReason(g, parent, stopHit)
	if cr != "" {
		finish = "cancelled"
	}
	return finish, n, gen.Logprobs, stopHit, gen.PrefillReused, cr, genErr(gen.Err())
}

// streamTokens consumes a token-id channel, applying stop strings (holding back a trailing partial stop match) and
// UTF-8 holdback, and calls onText with each newly completed fragment. It is the shared tail of every generation path
// (text via Session.Generate, multimodal via driveVL): the token source is orthogonal to this stop/stream logic.
// cancel ends the producing generation on a stop-string hit. Returns the finish reason ("stop" or "length"), the
// completion token count, and the stop string that was hit (empty unless a stop sequence ended the turn).
func (lm *loadedModel) streamTokens(parent context.Context, cancel context.CancelFunc, stream <-chan int, gr genRequest, gen *decoder.Generation, onText func(string)) (string, int, string) {
	// Reasoning is taken out before the stop logic below, so a stop string is matched against the answer only (see
	// thinkOut): the reasoning goes to its own callback, and what the stop logic watches, holds back and emits is the
	// answer text. With no splitter (th == nil) the stop logic watches everything the model wrote.
	th := gr.think
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
		if stopping {
			continue // drain so the generation goroutine exits cleanly
		}
		if tr != nil && len(ids) == 0 {
			tr.firstToken()
		}
		ids = append(ids, id)
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
		// was truncated. gen is safe to read here: the stream has closed.
		if len(ids) >= effectiveBudget(gen, gr.maxTokens) {
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
	if tr != nil {
		tr.generated(lm.name, len(gr.promptIDs), len(ids))
	}
	return finish, len(ids), stopHit
}

// effectiveBudget returns the token budget a finish_reason is judged against: the resident context-cap-clamped budget
// when the decoder published one (gen.BudgetClamped), else the requested max_tokens. A generation the resident
// truncated at the KV cap emits fewer tokens than requested, so judging against the request would report "stop"
// (clean) instead of "length" (truncated), and the client would never continue. gen is nil on paths that do not clamp
// (VL, speculative), which fall back to the requested value.
func effectiveBudget(gen *decoder.Generation, requested int) int {
	// Trust Budget only when the resident cap actually clamped this turn: Budget can be a genuine 0 (the prompt fills
	// the whole context, so 0 tokens are emitted and the turn is "length"), which a `> 0` test mis-reads as
	// unclamped.
	if gen != nil && gen.BudgetClamped {
		return gen.Budget
	}
	return requested
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
