package serveapp

import (
	"context"
	"errors"
	"fmt"
	"io"
	"maps"
	"net/http"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/townsendmerino/aikit/embed"
	"github.com/townsendmerino/aikit/encoder"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/app/serveapp/jobs"
	"github.com/townsendmerino/goinfer/internal/app/serveapp/stores"
	"github.com/townsendmerino/goinfer/internal/decide"
	"github.com/townsendmerino/goinfer/internal/guard/swapguard"
)

// maxAutoTextBodyBytes caps the context-derived default text body limit (64 MiB) so ultra-long-context models
// (e.g. 10M positions) cannot silently raise the default cap into gigabytes; -max-body-bytes still overrides it.
const maxAutoTextBodyBytes = 64 << 20

type server struct {
	// Generative (decoder) registry: served name to model, empty when only an embedding model is served. Requests
	// route on the OpenAI `model` field through withModel. regMu guards the map structure (dynamic load and unload
	// mutate it concurrently with routing); a request holds the picked *loadedModel beyond the RLock, and unload
	// drains in-flight requests through liveness before freeing the model.
	regMu  sync.RWMutex
	models map[string]*loadedModel
	speech map[string]*speechModel // Whisper models (speech.go), by served name; guarded by regMu
	cfg    config                  // backend/quant/lora/kv/session-dir/allow-admin for admin loads

	// pulls serialises -web model downloads to one at a time (webui.go).
	pulls pullState
	// loads serialises -web model loads the same way (webui.go handleWebLoad).
	loads pullState

	// liveness tracks, per underlying *decoder.Model, the request holders (rw) and the number of registry entries
	// backed by it (refs), so unload can drain in-flight work before freeing native memory instead of racing it into
	// a use-after-free. A base and its compute-time adapters share one *decoder.Model and thus one entry. Guarded by
	// regMu; see liveness.go.
	liveness map[io.Closer]*modelLiveness
	// draining is the set of served names whose entry has been unpublished but whose native memory
	// is not yet freed (the detached drain is still running). Surfaced by /health so an operator can
	// tell when a 202'd unload has actually reclaimed memory before reloading. Guarded by regMu.
	draining map[string]struct{}

	// Embedding (encoder) half — nil when only a generative model is served.
	// The encoder is goroutine-safe for concurrent Encode, so /v1/embeddings takes no
	// per-request mutex; embedGate only keeps Close from tearing it down under a request.
	embed     encoder.Encoder
	embedGate embedGate        // /v1/embeddings requests hold it shared; Close takes it to close embed
	embedTok  *embed.Tokenizer // counts tokens for usage.prompt_tokens
	embedID   string
	embedDim  int
	// embedMRLMin is the smallest width this embedder may be truncated to (0 = NOT truncatable,
	// the safe default). Only Matryoshka-trained models may be sliced; see resolveDimensions.
	embedMRLMin int
	// embedWidths, when set, is the exact set of widths the embedder was trained to be truncated to
	// (EmbeddingGemma 2: 768, 512, 256, 128); `dimensions` must be one of them. It takes precedence
	// over embedMRLMin's floor.
	embedWidths []int

	// Responses API (/v1/responses) state store for store/previous_response_id.
	responses *stores.ResponseStore[chatMessage]

	// idsKey signs the token ids a reply carries back (carried_ids.go): random per start, or --ids-key-file. nil turns the
	// feature off (a server built without newServer). idsStats are the carried-id totals /admin/status reports.
	idsKey   *idsKey
	idsStats idsCounters
	// tokenLog is --log-token-ids' file, nil when the flag is off.
	tokenLog *tokenIDLog

	// gens is the cancel-by-id registry, shared by every generation surface. Never
	// nil after newServer.
	gens *stores.GenerationRegistry

	// jobs is the job registry, shared by every generation surface the way
	// gens is. Never nil after newServer: every generation gets a job whether or not -job-dir is set; jobs.journal is
	// what is nil without it.
	jobs *jobs.Store

	// files and batches hold OpenAI's uploaded and assembled files and both dialects' batch records. Never nil after
	// newServer.
	files   *stores.FileStore
	batches *stores.BatchStore

	// halted is the global halt state; nil = running normally. See halt.go. The
	// zero value is nil, so newServer needs no init.
	halted atomic.Pointer[haltInfo]

	// swapGuardTripped and swapWatch are the serving-side swap tripwire; see
	// swapguard.go. The zero values mean not armed yet: startSwapGuard sets swapWatch during newServer, and a nil
	// swapWatch (GOINFER_SWAP_GUARD=off) is a valid, permanent state.
	swapGuardTripped atomic.Bool
	swapWatch        *swapguard.Watch

	// lifeCtx is the server-lifetime context cancelled on shutdown/Close; background jobs and batch lines derive from it.
	lifeCtx context.Context
}

// errServerShutdown is the context cancellation cause used by server shutdown and handlerCloser.Close
// so jobs and batch lines interrupted by shutdown record jobs.Interrupted instead of user jobs.Cancelled.
var errServerShutdown = errors.New("server shutting down")

func (s *server) initLifeContext() (context.Context, context.CancelCauseFunc) {
	ctx, cancel := context.WithCancelCause(context.Background())
	s.lifeCtx = ctx
	return ctx, cancel
}

func (s *server) lifeContext() context.Context {
	if s.lifeCtx != nil {
		return s.lifeCtx
	}
	return context.Background()
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
// autoTextBodyCap clamps an auto-derived text body cap between the 4 MiB floor and the 64 MiB ceiling.
func autoTextBodyCap(derived int64) int64 {
	if derived <= maxBodyBytes {
		return maxBodyBytes
	}
	return min(derived, maxAutoTextBodyBytes)
}

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
		textCap = autoTextBodyCap(derived)
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
// kvShareField is /v1/models' kv_share vendor field: decoder.Model.KVShareState as "on", or "off: why" and "none: why".
func kvShareField(m *decoder.Model) string {
	state, why := m.KVShareState()
	if why == "" {
		return state
	}
	return state + ": " + why
}

func (s *server) pathFields(name string) map[string]any {
	lm := s.modelByName(name)
	if lm == nil || lm.model == nil {
		return nil
	}
	batched, why := lm.model.PrefillPath()
	f := map[string]any{
		"decode_path":     lm.model.DecodePath(),
		"kv_share":        kvShareField(lm.model),
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
		// H-b (docs/measurements/harness-dialects-2026-10-09.md): the same number under the names harnesses read.
		// Hermes takes max_model_len or context_length (else assumes 128K, and refuses under 64K), Oh My Pi's
		// discovery reads the same two (else 128K); a client that over-assumes the window overruns ours.
		f["context_length"] = ctx
		f["max_model_len"] = ctx
	}
	// max_positions is the model's own maximum context, which -ctx cannot exceed, so a client told the window is too
	// small can tell a larger -ctx from a model that cannot hold the request at all.
	if mp := lm.model.Config().MaxPositions; mp > 0 {
		f["max_positions"] = mp
	}
	// vision reports whether image content parts are accepted, by the same visionCapable the vision path checks, so a
	// client can hide image input on a model that would refuse it.
	f["vision"] = lm.visionCapable()
	if e := forcedCallEnforcement(lm.tmpl); e != "" {
		f["forced_tool_call"] = e // how a tool_choice required / any is enforced on this model (forcedCallEnforcement)
	}
	return f
}

// servedNames lists the loaded generative and embedding model ids, sorted.
func (s *server) servedNames() []string {
	s.regMu.RLock()
	names := make([]string, 0, len(s.models)+len(s.speech)+1)
	for n := range s.models {
		names = append(names, n)
	}
	for n := range s.speech {
		names = append(names, n)
	}
	s.regMu.RUnlock()
	if s.embed != nil {
		names = append(names, s.embedID)
	}
	slices.Sort(names)
	return names
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
		if alias := s.fastAliasOf(name); alias != "" {
			// Vendor extension, same convention: the alias connect points an agent's small-model setting at (fast_alias.go).
			e["fast_model"] = alias
		}
		if s.speechByName(name) != nil {
			e["capabilities"] = []string{"audio.transcriptions"} // a speech model: no decoder paths, one route
		}
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
	if lm.decisions.clef != nil {
		return map[string]any{"endpoint": "/v1/systemone", "route": routeClef, "kinds": []string{decide.KindNoul, decide.KindChoice, decide.KindScore},
			"calibrated": false, "confidence": clefConfidence}
	}
	if h := lm.decisions.head; h != nil {
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
	if s.fastRoute(r, &req.Model) {
		req.applyFast()
	}
	req.Tools = dropUnrunnableTools(w, req.Tools)
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
	gr, err := lm.prepareTurn(req.sampling, tm, system, turns, nil, ids)
	if err != nil {
		writePrepareErr(w, err)
		return
	}
	if req.Stream && req.Logprobs { // the stream path has no logprobs field — reject rather than silently drop
		writeErr(w, http.StatusBadRequest, "logprobs is not supported together with stream:true")
		return
	}
	if !lm.enter(w, r, admissionRecord{promptIDs: gr.promptIDs, clock: gr.clock}, s.haltState) {
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
		streamCtx, cancelStream := ss.CancelOnFail(r.Context())
		defer cancelStream()
		role := chatChunk(id, created, lm.name, delta{Role: "assistant"}, nil)
		sseSend(ss, role)
		// Nothing else is sent between here and the first token, and on CPU that gap is the whole prefill: a
		// heartbeat keeps a client's idle timeout from firing during it.
		stopBeat := sseHeartbeat(ss)
		defer stopBeat()
		s.routeThink(lm, &gr, tm, turns, ts, func(t string) {
			sseSend(ss, chatChunk(id, created, lm.name, delta{ReasoningContent: t}, nil))
		})
		res, gerr := lm.drive(streamCtx, gr, s.gens, s.jobs, func(t string) {
			sseSend(ss, chatChunk(id, created, lm.name, delta{Content: t}, nil))
		})
		stopBeat()
		if gerr != nil {
			sseErr(ss, "generation failed: "+gerr.Error())
			sseDone(ss)
			return
		}
		sseSend(ss, finishChunk(id, created, lm.name, res.finish, emptyAnswerOf(gr, res.finish, res.stopHit, 0, res.nComp), false))
		if gr.conf != nil {
			sseSend(ss, map[string]any{"id": id, "goinfer_confidence": gr.conf.payload()})
		}
		if res.cancelReason != "" {
			// One final SSE event naming the reason, so a client cannot mistake this for a natural stop even though
			// finish_reason is already "cancelled" above.
			sseSend(ss, map[string]any{"goinfer_cancelled": map[string]any{"id": id, "reason": res.cancelReason}})
		}
		sendUsage(ss, gr.clock, req.StreamOptions, id, created, lm.name,
			newUsage(len(gr.promptIDs), res.nComp, res.reused))
		sseDone(ss)
		return
	}

	var sb, rb strings.Builder
	s.routeThink(lm, &gr, tm, turns, ts, func(t string) { rb.WriteString(t) })
	res, gerr := lm.drive(r.Context(), gr, s.gens, s.jobs, func(t string) { sb.WriteString(t) })
	if gerr != nil {
		writeServerErr(w, "generation failed: "+gerr.Error())
		return
	}
	if res.cancelReason != "" {
		writeErr(w, statusCancelled, "generation cancelled: "+res.cancelReason)
		return
	}
	msg := map[string]any{"role": "assistant", "content": sb.String()}
	if rb.Len() > 0 {
		msg["reasoning_content"] = rb.String()
	}
	choice := map[string]any{
		"index":         0,
		"message":       msg,
		"finish_reason": res.finish,
	}
	if e := emptyAnswerOf(gr, res.finish, res.stopHit, 0, res.nComp); e != nil {
		choice["goinfer_empty_answer"] = e
	}
	if req.Logprobs {
		choice["logprobs"] = lm.logprobs(res.logprobs)
	}
	resp := map[string]any{
		"id": id, "object": "chat.completion", "created": created, "model": lm.name,
		"choices": []any{choice},
		"usage":   newUsage(len(gr.promptIDs), res.nComp, res.reused),
	}
	if gr.conf != nil {
		resp["goinfer_confidence"] = gr.conf.payload()
	}
	writeJSON(w, http.StatusOK, gr.clock.AddTurn(resp))
}

func (s *server) handleCompletions(w http.ResponseWriter, r *http.Request) {
	var req completionReq
	if !decodeJSON(w, r, &req) {
		return
	}
	s.fastRoute(r, &req.Model) // /v1/completions renders no template, so the alias has nothing to turn off there
	if err := validateN(req.N); err != nil {
		writeErr(w, http.StatusBadRequest, err.Error())
		return
	}
	s.withModel(w, req.Model, func(lm *loadedModel) { s.serveCompletion(w, r, req, lm) })
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
		writePrepareErr(w, err)
		return
	}
	if !lm.enter(w, r, admissionRecord{promptIDs: gr.promptIDs, clock: gr.clock}, s.haltState) {
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
		streamCtx, cancelStream := ss.CancelOnFail(r.Context())
		defer cancelStream()
		// Same prefill silence as the chat stream: heartbeat until the first token.
		stopBeat := sseHeartbeat(ss)
		defer stopBeat()
		res, gerr := lm.drive(streamCtx, gr, s.gens, s.jobs, func(t string) {
			sseSend(ss, completionChunk(id, created, lm.name, t, nil))
		})
		stopBeat()
		if gerr != nil {
			sseErr(ss, "generation failed: "+gerr.Error())
			sseDone(ss)
			return
		}
		sseSend(ss, completionChunk(id, created, lm.name, "", &res.finish))
		if gr.conf != nil {
			sseSend(ss, map[string]any{"id": id, "goinfer_confidence": gr.conf.payload()})
		}
		if res.cancelReason != "" {
			sseSend(ss, map[string]any{"goinfer_cancelled": map[string]any{"id": id, "reason": res.cancelReason}})
		}
		sendUsage(ss, gr.clock, req.StreamOptions, id, created, lm.name,
			newUsage(len(gr.promptIDs), res.nComp, res.reused))
		sseDone(ss)
		return
	}
	var sb strings.Builder
	res, gerr := lm.drive(r.Context(), gr, s.gens, s.jobs, func(t string) { sb.WriteString(t) })
	if gerr != nil {
		writeServerErr(w, "generation failed: "+gerr.Error())
		return
	}
	if res.cancelReason != "" {
		writeErr(w, statusCancelled, "generation cancelled: "+res.cancelReason)
		return
	}
	resp := map[string]any{
		"id": id, "object": "text_completion", "created": created, "model": lm.name,
		"choices": []any{map[string]any{"index": 0, "text": sb.String(), "finish_reason": res.finish}},
		"usage":   newUsage(len(gr.promptIDs), res.nComp, res.reused),
	}
	if gr.conf != nil {
		resp["goinfer_confidence"] = gr.conf.payload()
	}
	writeJSON(w, http.StatusOK, gr.clock.AddTurn(resp))
}
