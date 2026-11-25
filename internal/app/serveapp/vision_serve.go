package serveapp

import (
	"fmt"
	"net/http"
	"os"
	"slices"
	"strings"
	"time"

	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/internal/codec"
	"github.com/townsendmerino/goinfer/internal/runtime/mem"
	"github.com/townsendmerino/goinfer/multimodal"
)

// imageSoftToken is the Gemma 3 image placeholder token; the per-image block and
// run-finder live in the vision package (shared with demo/agent).
const (
	imageSoftToken   = multimodal.ImageSoftToken
	maxImagesPerTurn = 8 // images in the newest message, each its own block; a guard on tower work per request
)

// lastUserTurn returns the index of the last user turn (where v1 attaches the
// image, per the Gemma 3 convention of the image leading the user content), or -1.
func lastUserTurn(turns []chat.Turn) int {
	for i, turn := range slices.Backward(turns) {
		if turn.Role == "user" {
			return i
		}
	}
	return -1
}

// visionInput is the assembled multimodal prompt: encoded ids carrying the image
// placeholder run [imgPos,imgPos+imgLen), the vision features replacing it, and (for
// Qwen2.5-VL) the image grid driving m-RoPE. qwen selects the decode path.
type visionInput struct {
	ids            []int
	features       func() ([]float32, error) // runs the tower+projector; invoked at most once, lazily
	imgHash        uint64                    // content hash of the raw image bytes, for resident image reuse
	imgPos, imgLen int
	grid           [3]int // Qwen m-RoPE grid (t,h,w in patch units); zero ⇒ Gemma 3
	qwen           bool
	deepSets       int    // Qwen3-VL: features() returns the merged rows, then this many DeepStack sets of the same size
	gemma4         bool   // selects GenerateGemma4VL in driveVL
	asr            bool   // selects GenerateAudio in driveVL (Qwen3-ASR, Voxtral)
	replyPrefix    string // text the prompt already ends with that belongs at the head of the reply; driveVL emits it before the generated text
	causal         bool   // Ministral 3 and LFM2-VL, causal image tokens (GenerateVLCausalSpans)
	images         int    // the images in the turn (a Pixtral image is one span per merged row, so not len(spans))
	// Every image's span and Qwen grid, in prompt order (imgPos/imgLen/imgHash/grid are the first's). features then returns
	// every image's rows concatenated (merged rows, then each DeepStack set across images). Empty for the one-media builders
	// (GLM-OCR, audio), whose single span driveVL reads from imgPos/imgLen.
	spans []imageSpan
	grids [][3]int
}

// encodeVisionSegments encodes a vision prompt the way the TEXT path does: the template's structural markers and the image
// block as Special segments, the user's own words as ordinary content the added-token trie never sees. Without that
// (Tokenizer.Encode is not for untrusted content) a user message in an IMAGE request containing
// "<end_of_turn>\n<start_of_turn>model\n" or "<|im_end|>...<|im_start|>system" becomes real control tokens and forges a turn
// boundary.
//
// The image block has to stay SPECIAL: its sentinels and soft-token run are what FindImageRun locates and what the
// embed-by-vector seam replaces, so it cannot simply be prepended to the content segment, which is untrusted by construction.
// It is spliced back in as its own Special segment, and a block that fails to splice is an error rather than a prompt that
// silently tokenizes the sentinels as text (the imgLen check downstream would catch it, but late and with a misleading message
// about a template mismatch).
func encodeVisionSegments(lm *loadedModel, tm *chat.Template, system string, turns []chat.Turn, block string) ([]int, error) {
	segs, err := codec.ImageSegments(tm, system, turns, block)
	if err != nil {
		return nil, err
	}
	return codec.Encode(lm.tk, segs)
}

// visionPrompt runs img through the tower and assembles the multimodal prompt: the text turns with the family's image block
// prepended to the last user turn, so the rendered+encoded ids carry a placeholder run the embed seam overrides. tm is the
// per-request template (think.go): the model's own, switched to the request's thinking mode.
func (lm *loadedModel) visionPrompt(tm *chat.Template, system string, turns []chat.Turn, img imageRef) (visionInput, error) {
	vi, err := lm.visionPromptUncached(tm, system, turns, img)
	if err != nil {
		return vi, err
	}
	// One place for every family: the cache sits between the prompt builders and the generate call.
	return lm.withFeatureCache(vi, img.data, lm.towerFamily(img.audio)), nil
}

// withFeatureCache answers vi's features from this model's per-image cache when it holds these bytes' encode. A miss runs the tower through
// recoverDeviceTower, so a tower that panics (on any device, for any family) fails the request rather than the server.
func (lm *loadedModel) withFeatureCache(vi visionInput, raw []byte, family string) visionInput {
	tower := vi.features
	vi.features = lm.visionFeatureCache().wrap(raw, func() ([]float32, error) { return recoverDeviceTower(family, tower) })
	return vi
}

// visionFeatureCache is this model's per-image tower-output cache, made on first use and sized then by
// featureCacheSize.
func (lm *loadedModel) visionFeatureCache() *featureCache {
	lm.featCacheOnce.Do(func() { lm.featCache = newFeatureCache(lm.featureCacheSize()) })
	return lm.featCache
}

// featureCacheSize is featureCacheBudgetFor over this model's fit plan, read against the memory available when it is
// called, so a model loaded since this one is counted. With the fit guard off (GOINFER_NO_FIT_GUARD) or --fit=off it is
// featureCacheBudget.
func (lm *loadedModel) featureCacheSize() int {
	if lm.model == nil || lm.model.FitDisabled() {
		return featureCacheBudget
	}
	if v, _ := lm.model.Knob("GOINFER_NO_FIT_GUARD"); v != "" {
		return featureCacheBudget
	}
	_, kvBytes, _, budgetBytes, known := lm.model.FitBudgetSummary()
	n := featureCacheBudgetFor(known, budgetBytes, kvBytes)
	if n < featureCacheBudget {
		fmt.Fprintf(os.Stderr, "vision: the image feature cache holds %d MiB, not %d: that is what the fit budget leaves after this model's KV\n",
			n>>20, featureCacheBudget>>20)
	}
	return n
}

func (lm *loadedModel) visionPromptUncached(tm *chat.Template, system string, turns []chat.Turn, img imageRef) (visionInput, error) {
	if img.audio && lm.audio != nil {
		return lm.audio.prompt(lm, tm, system, turns, img)
	}
	if lm.tmpl == nil {
		return visionInput{}, fmt.Errorf("this model has no chat template for vision")
	}
	idx := lastUserTurn(turns)
	if idx < 0 {
		return visionInput{}, fmt.Errorf("no user turn to attach the image to")
	}
	if g := lm.glmOcr(); g != nil {
		return lm.glmOcrVisionPrompt(g, tm, system, turns, idx, img)
	}
	if _, ok := lm.vision.(qwenGridTower); ok {
		return lm.imagesPrompt(tm, system, turns, idx, []imageRef{img}, false, "", false)
	}
	if img.audio {
		return visionInput{}, fmt.Errorf("this model has no audio tower")
	}
	return lm.imagesPrompt(tm, system, turns, idx, []imageRef{img}, false, "", false)
}

// glmOcr is this entry's GLM-OCR tower, nil for any other.
func (lm *loadedModel) glmOcr() *glmVision {
	g, _ := lm.vision.(*glmVision)
	return g
}

// residentTower is the part of aikit's Qwen2.5-VL encoder the fallback uses: whether its device tower is attached, and detaching it.
type residentTower interface {
	ResidentEnabled() bool
	Close()
}

// qwenDeviceForward runs forward on the Qwen2.5-VL encoder t, a panic becoming an error. On a device allocation failure with the device tower attached it detaches the
// tower (the encoder then runs aikit's CPU path) and runs the image again there; under require (-require-backend) it fails the request instead and keeps the tower.
func qwenDeviceForward(t residentTower, require bool, forward func() ([]float32, error)) ([]float32, error) {
	out, err := recoverDeviceTower("Qwen2.5-VL", forward)
	if !t.ResidentEnabled() || !mem.DeviceOutOfMemory(err) {
		return out, err
	}
	if require {
		return nil, refuseCPUFallback("Qwen2.5-VL", err)
	}
	fmt.Fprintf(os.Stderr, "vision: the Qwen2.5-VL tower ran out of device memory (%v); it runs on the CPU from now on\n", err)
	t.Close()
	return recoverDeviceTower("Qwen2.5-VL", forward)
}

// recoverDeviceTower runs a tower's forward and turns a panic into an error that names the likely cause. A device tower grows
// its scratch on the first image, after the resident decoder has taken the device's memory, and a tower may panic on an
// allocation failure (`CUDA_ERROR_OUT_OF_MEMORY`) instead of returning it (aikit's Qwen2.5-VL tower did): unrecovered, one big
// image kills the whole server. The request fails with the reason; the server and the other requests live. Every tower passes
// through it: the grid towers inside deviceFallback.run, Qwen2.5-VL inside qwenDeviceForward, and every family's features
// (Gemma 3's SigLIP, Gemma 4's image and audio towers included) at withFeatureCache.
func recoverDeviceTower(family string, fn func() ([]float32, error)) (out []float32, err error) {
	defer func() {
		if r := recover(); r != nil {
			hint := ""
			if mem.DeviceOutOfMemory(fmt.Errorf("%v", r)) {
				hint = ": the input needs more device memory than the decoder left it; lower -ctx or -kv-sessions, resize the image, or use -vision-device cpu"
			}
			out, err = nil, fmt.Errorf("the %s tower failed (%v)%s", family, r, hint)
		}
	}()
	return fn()
}

// towerFamily names this model's tower for an error message and the feature cache's key: the audio tower's family for
// a clip, else the vision tower's.
func (lm *loadedModel) towerFamily(audio bool) string {
	switch {
	case audio && lm.audio != nil:
		return lm.audio.family()
	case audio:
		return "Gemma 4 audio"
	case lm.vision != nil:
		return lm.vision.family()
	}
	return "Gemma 3 SigLIP"
}

// glmOcrVisionPrompt is the GLM-OCR image path: smart-resize preprocess (halved pixel bounds) -> the tower (the merged rows
// replace the <|image|> run) -> the prompt with the image block prepended to the last user turn. The grid drives m-RoPE in
// GenerateQwenVL, exactly as for Qwen.
//
// The image LEADS the user turn and the task prompt follows it with no separator (the checkpoint's own template:
// `<|begin_of_image|><|image|>x N<|end_of_image|>Text Recognition:`); a request with no text part gets plain text
// recognition rather than an image with nothing to do. The task is chosen by the user's own text ("Text Recognition:",
// "Formula Recognition:", "Table Recognition:", multimodal.GlmOcrPrompt*).
func (lm *loadedModel) glmOcrVisionPrompt(g *glmVision, tm *chat.Template, system string, turns []chat.Turn, idx int, img imageRef) (visionInput, error) {
	pv, grid, err := multimodal.QwenPreprocess(img.data, g.pp)
	if err != nil {
		return visionInput{}, err
	}
	imgHash := multimodal.HashImageBytes(img.data)
	n := multimodal.QwenMergedTokens(grid, g.merge)
	if err := lm.imageFitsContext(n, grid); err != nil { // before the tower: minutes of CPU for an answer that is known now
		return visionInput{}, err
	}
	hiddenDim := lm.model.Config().HiddenDim
	features := func() ([]float32, error) {
		feats, err := g.features(pv, grid)
		if err != nil {
			return nil, fmt.Errorf("glm-ocr vision encoder: %w", err)
		}
		if len(feats) != n*hiddenDim {
			return nil, fmt.Errorf("glm-ocr encoder emitted %d features, want %d", len(feats), n*hiddenDim)
		}
		return feats, nil
	}
	if strings.TrimSpace(turns[idx].Content) == "" {
		turns[idx].Content = multimodal.GlmOcrDefaultPrompt
	}
	block := multimodal.GlmOcrImageBlock(n)
	turns[idx].Content = block + turns[idx].Content
	ids, err := encodeVisionSegments(lm, tm, system, turns, block)
	if err != nil {
		return visionInput{}, fmt.Errorf("encode: %w", err)
	}
	imgPos, imgLen := multimodal.FindImageRun(ids, g.imgTok)
	if imgLen != n {
		return visionInput{}, fmt.Errorf("image placeholder run = %d image tokens, want %d (template mismatch)", imgLen, n)
	}
	return visionInput{ids: ids, features: features, imgHash: imgHash, imgPos: imgPos, imgLen: imgLen, grid: grid, qwen: true}, nil
}

// serveVisionChat handles an OpenAI /v1/chat/completions request that carries an
// image. The image runs through the tower, the prompt is assembled with the
// image block, and generation goes through the multimodal path (driveVL, which
// bypasses the warm-KV session). usage.prompt_tokens includes the image tokens
// (they occupy real KV positions). Streaming + non-streaming.
func (s *server) serveVisionChat(w http.ResponseWriter, r *http.Request, req chatReq, imgs []imageRef) {
	s.withModel(w, req.Model, func(lm *loadedModel) { s.serveVisionChatWith(w, r, req, imgs, lm) })
}

// serveVisionChatWith runs the multimodal generation. Reached ONLY through withModel (liveness RLock held).
func (s *server) serveVisionChatWith(w http.ResponseWriter, r *http.Request, req chatReq, imgs []imageRef, lm *loadedModel) {
	if imgs[0].audio {
		if !lm.audioCapable() {
			writeErr(w, http.StatusBadRequest, "this model has no audio tower (audio input needs a Gemma 4 checkpoint with an audio_config, e.g. E2B or E4B)")
			return
		}
	} else if !lm.visionCapable() {
		writeErr(w, http.StatusBadRequest, "this model has no vision tower (start with --vision <dir> to enable image input)")
		return
	}
	if len(imgs) > maxImagesPerTurn {
		writeErr(w, http.StatusBadRequest, tooManyImages(len(imgs)))
		return
	}
	// Image bytes are excluded (chatInputBytes counts tokenizable TEXT), so this cannot reject a valid image request on a
	// small-context model.
	if err := lm.promptTooLargeForContext(chatInputBytes(req.Messages)); err != nil {
		writeErr(w, http.StatusBadRequest, err.Error())
		return
	}
	system, turns := messagesToTurns(req.Messages)
	// An image plus response_format json_schema on GLM-OCR is an extraction request; its prompt is the schema's JSON template.
	if err := lm.glmOcrExtractionTurn(req.sampling.ResponseFormat, turns); err != nil {
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
	msgText, ordered := chatMediaText(req.Messages)
	vi, err := lm.visionPromptN(tm, system, turns, imgs, ordered, msgText)
	if err != nil {
		writeErr(w, http.StatusBadRequest, err.Error())
		return
	}
	req.sampling.confidenceOK = true // written back below (the same two sites as the text route)
	gr, err := lm.prepare(req.sampling, vi.ids, false)
	if err != nil {
		writePrepareErr(w, err)
		return
	}
	if !lm.enter(w, r, admissionRecord{promptIDs: gr.promptIDs, clock: gr.clock}, s.haltState) {
		return
	}
	defer lm.exit()
	id := "chatcmpl-" + reqID()
	gr.id = id // registers this generation for cancel-by-id
	created := time.Now().Unix()

	if req.Stream {
		ss, ok := sseStart(w)
		if !ok {
			return
		}
		sseSend(ss, chatChunk(id, created, lm.name, delta{Role: "assistant"}, nil))
		// include_usage needs the real generated-token count (nComp), which no count of emitted chunks can report. Nothing
		// else is sent before the first token and a CPU image prefill can take minutes, so a heartbeat runs while driveVL
		// works, as on the text-only lm.drive sites.
		stopBeat := sseHeartbeat(ss)
		s.routeThink(lm, &gr, tm, turns, ts, func(t string) {
			sseSend(ss, chatChunk(id, created, lm.name, delta{ReasoningContent: t}, nil))
		})
		res, gerr := lm.driveVL(r.Context(), gr, vi, s.gens, s.jobs, func(t string) {
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
			sseSend(ss, map[string]any{"goinfer_cancelled": map[string]any{"id": id, "reason": res.cancelReason}})
		}
		sendUsage(ss, gr.clock, req.StreamOptions, id, created, lm.name,
			newUsage(len(gr.promptIDs), res.nComp, res.reused))
		sseDone(ss)
		return
	}
	var sb, rb strings.Builder
	s.routeThink(lm, &gr, tm, turns, ts, func(t string) { rb.WriteString(t) })
	res, gerr := lm.driveVL(r.Context(), gr, vi, s.gens, s.jobs, func(t string) { sb.WriteString(t) })
	if gerr != nil {
		writeServerErr(w, "generation failed: "+gerr.Error())
		return
	}
	if res.cancelReason != "" {
		writeErr(w, statusCancelled, "generation cancelled: "+res.cancelReason)
		return
	}
	vmsg := map[string]any{"role": "assistant", "content": sb.String()}
	if rb.Len() > 0 {
		vmsg["reasoning_content"] = rb.String()
	}
	vresp := map[string]any{
		"id": id, "object": "chat.completion", "created": created, "model": lm.name,
		"choices": []any{map[string]any{
			"index":         0,
			"message":       vmsg,
			"finish_reason": res.finish,
		}},
		"usage": newUsage(len(gr.promptIDs), res.nComp, res.reused),
	}
	if e := emptyAnswerOf(gr, res.finish, res.stopHit, 0, res.nComp); e != nil {
		vresp["choices"].([]any)[0].(map[string]any)["goinfer_empty_answer"] = e
	}
	if req.Logprobs {
		vresp["choices"].([]any)[0].(map[string]any)["logprobs"] = lm.logprobs(res.logprobs)
	}
	if gr.conf != nil {
		vresp["goinfer_confidence"] = gr.conf.payload()
	}
	writeJSON(w, http.StatusOK, gr.clock.AddTurn(vresp))
}

// serveVisionMessages handles an Anthropic /v1/messages request carrying an image
// block: same tower → prompt → driveVL path as serveVisionChat, rendered in the
// Anthropic message shape (named-event SSE when streaming). input_tokens include
// the image tokens.
func (s *server) serveVisionMessages(w http.ResponseWriter, r *http.Request, req *anthropicReq, lm *loadedModel, imgs []imageRef) {
	if !lm.visionCapable() {
		writeAnthropicErr(w, http.StatusBadRequest, "invalid_request_error", "this model has no vision tower (start with --vision <dir> to enable image input)")
		return
	}
	if len(imgs) > maxImagesPerTurn {
		writeAnthropicErr(w, http.StatusBadRequest, "invalid_request_error", tooManyImages(len(imgs)))
		return
	}
	// This function has no tokenizer call of its own (visionPrompt tokenizes), so the guard is spelled out here. Image bytes
	// are excluded from the count, so a valid image request is never rejected by it.
	if err := lm.promptTooLargeForContext(anthropicInputBytes(req)); err != nil {
		writeAnthropicErr(w, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	system, turns, aerr := anthropicTurns(req) // image blocks skipped; text turns only
	if aerr != nil {
		aerr.write(w)
		return
	}
	ts, terr := s.resolveThink(req.thinkRequest())
	if terr != nil {
		writeAnthropicErr(w, http.StatusBadRequest, "invalid_request_error", terr.Error())
		return
	}
	sm, serr := req.toSampling()
	if serr != nil {
		serr.write(w)
		return
	}
	tm := lm.templateFor(ts)
	if sm.constrainsOutput() {
		tm = lm.constrainedTemplate(ts)
	}
	msgText, ordered := anthropicMediaText(req)
	vi, err := lm.visionPromptN(tm, system, turns, imgs, ordered, msgText)
	if err != nil {
		writeAnthropicErr(w, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	gr, err := lm.prepare(sm, vi.ids, false)
	if err != nil {
		writeAnthropicPrepareErr(w, err)
		return
	}
	if ok, haltReason := lm.tryEnter(r.Context(), admissionRecord{promptIDs: gr.promptIDs, clock: gr.clock}, s.haltState); !ok {
		if haltReason != "" {
			writeAnthropicErr(w, http.StatusServiceUnavailable, "overloaded_error", "halted: "+haltReason)
			return
		}
		if r.Context().Err() != nil {
			return // client disconnected while queued; nothing to write to
		}
		w.Header().Set("Retry-After", "1")
		writeAnthropicErr(w, 529, "overloaded_error", fmt.Sprintf("model %q is busy; retry", lm.name))
		return
	}
	defer lm.exit()
	id := "msg_" + reqID()
	gr.id = id // registers this generation for cancel-by-id

	if req.Stream {
		ss, ok := anthropicSSEStart(w)
		if !ok {
			return
		}
		anthropicEvent(ss, "message_start", map[string]any{
			"type": "message_start",
			"message": map[string]any{
				"id": id, "type": "message", "role": "assistant", "model": lm.name,
				"content": []any{}, "stop_reason": nil, "stop_sequence": nil,
				"usage": map[string]any{"input_tokens": len(gr.promptIDs), "output_tokens": 0},
			},
		})
		anthropicEvent(ss, "ping", map[string]any{"type": "ping"})
		th := &anthropicThink{ss: ss, want: req.wantsThinking()}
		s.routeThink(lm, &gr, tm, turns, ts, th.push)
		if th.want {
			streamMessagesThinking(ss, th, gr, func(onText func(string)) (turnResult, error) {
				return lm.driveVL(r.Context(), gr, vi, s.gens, s.jobs, onText)
			})
			return
		}
		anthropicEvent(ss, "content_block_start", map[string]any{
			"type": "content_block_start", "index": 0,
			"content_block": map[string]any{"type": "text", "text": ""},
		})
		// The ping above is one-shot, not a keep-alive: an image prefill on CPU can take minutes with nothing sent until the
		// first token.
		stopBeat := sseHeartbeat(ss)
		res, gerr := lm.driveVL(r.Context(), gr, vi, s.gens, s.jobs, func(t string) {
			anthropicEvent(ss, "content_block_delta", map[string]any{
				"type": "content_block_delta", "index": 0,
				"delta": map[string]any{"type": "text_delta", "text": t},
			})
		})
		stopBeat()
		if gerr != nil {
			anthropicEvent(ss, "content_block_stop", map[string]any{"type": "content_block_stop", "index": 0})
			anthropicStreamErr(ss, "generation failed: "+gerr.Error())
			return
		}
		anthropicEvent(ss, "content_block_stop", map[string]any{"type": "content_block_stop", "index": 0})
		reason, seq := anthropicStopReason(res.finish, res.stopHit)
		anthropicMessageEnd(ss, gr.clock, reason, seq, anthropicUsage(len(gr.promptIDs), res.reused.Tokens, res.nComp), res.cancelReason, emptyAnswerOf(gr, res.finish, res.stopHit, 0, res.nComp), false)
		return
	}
	var sb, rb strings.Builder
	s.routeThink(lm, &gr, tm, turns, ts, func(t string) { rb.WriteString(t) })
	res, gerr := lm.driveVL(r.Context(), gr, vi, s.gens, s.jobs, func(t string) { sb.WriteString(t) })
	if gerr != nil {
		writeAnthropicErr(w, http.StatusInternalServerError, "api_error", "generation failed: "+gerr.Error())
		return
	}
	if res.cancelReason != "" {
		writeAnthropicErr(w, statusCancelled, "cancelled", "generation cancelled: "+res.cancelReason)
		return
	}
	reason, seq := anthropicStopReason(res.finish, res.stopHit)
	vcontent := []map[string]any{textBlock(sb.String())}
	if req.wantsThinking() && rb.Len() > 0 {
		vcontent = append([]map[string]any{thinkingBlock(rb.String(), "")}, vcontent...)
		if sb.Len() == 0 { // cut off while thinking: no answer, so no text block
			vcontent = vcontent[:1]
		}
	}
	vmsg := map[string]any{
		"id": id, "type": "message", "role": "assistant", "model": lm.name,
		"content":       vcontent,
		"stop_reason":   reason,
		"stop_sequence": seq,
		"usage":         anthropicUsage(len(gr.promptIDs), res.reused.Tokens, res.nComp),
	}
	if e := emptyAnswerOf(gr, res.finish, res.stopHit, 0, res.nComp); e != nil {
		vmsg["goinfer_empty_answer"] = e
	}
	writeJSON(w, http.StatusOK, gr.clock.AddTurn(vmsg))
}
