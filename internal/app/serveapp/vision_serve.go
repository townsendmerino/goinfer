package serveapp

import (
	"fmt"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/internal/app/serveapp/stores"
	"github.com/townsendmerino/goinfer/internal/app/serveapp/streamframe"
	"github.com/townsendmerino/goinfer/internal/app/serveapp/towerfit"
	"github.com/townsendmerino/goinfer/internal/app/serveapp/towers"
	"github.com/townsendmerino/goinfer/multimodal"
)

const (
	imageSoftToken   = multimodal.ImageSoftToken
	maxImagesPerTurn = 8
)

var (
	lastUserTurn       = towers.LastUserTurn
	recoverDeviceTower = towerfit.RecoverDeviceTower
)

func encodeVisionSegments(lm *loadedModel, tm *chat.Template, system string, turns []chat.Turn, block string) ([]int, error) {
	return towers.EncodeVisionSegments(lm.tk, tm, system, turns, block)
}

func (lm *loadedModel) visionPrompt(tm *chat.Template, system string, turns []chat.Turn, img imageRef) (visionInput, error) {
	vi, err := lm.visionPromptUncached(tm, system, turns, img)
	if err != nil {
		return vi, err
	}
	return lm.withFeatureCache(vi, img.Data, lm.towerFamily(img.Audio)), nil
}

func (lm *loadedModel) withFeatureCache(vi visionInput, raw []byte, family string) visionInput {
	tower := vi.Features
	vi.Features = lm.visionFeatureCache().Wrap(raw, func() ([]float32, error) { return recoverDeviceTower(family, tower) })
	return vi
}

func (lm *loadedModel) visionFeatureCache() *stores.FeatureCache {
	lm.caches.featOnce.Do(func() { lm.caches.feat = stores.NewFeatureCache(lm.featureCacheSize()) })
	return lm.caches.feat
}

func (lm *loadedModel) featureCacheSize() int {
	if lm.model == nil || lm.model.FitDisabled() {
		return stores.FeatureCacheBudget
	}
	if v, _ := lm.model.Knob("GOINFER_NO_FIT_GUARD"); v != "" {
		return stores.FeatureCacheBudget
	}
	_, kvBytes, _, budgetBytes, known := lm.model.FitBudgetSummary()
	n := stores.FeatureCacheBudgetFor(known, budgetBytes, kvBytes)
	if n < stores.FeatureCacheBudget {
		fmt.Fprintf(os.Stderr, "vision: the image feature cache holds %d MiB, not %d: that is what the fit budget leaves after this model's KV\n",
			n>>20, stores.FeatureCacheBudget>>20)
	}
	return n
}

func (lm *loadedModel) visionPromptUncached(tm *chat.Template, system string, turns []chat.Turn, img imageRef) (visionInput, error) {
	if img.Audio && lm.audio != nil {
		return lm.audio.Prompt(lm.audioModel(), tm, system, turns, img)
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
	if img.Audio {
		return visionInput{}, fmt.Errorf("this model has no audio tower")
	}
	return lm.imagesPrompt(tm, system, turns, idx, []imageRef{img}, false, "", false)
}

func (lm *loadedModel) glmOcr() *glmVision {
	g, _ := lm.vision.(*glmVision)
	return g
}

func (lm *loadedModel) towerFamily(audio bool) string {
	switch {
	case audio && lm.audio != nil:
		return lm.audio.Family()
	case audio:
		return "Gemma 4 audio"
	case lm.vision != nil:
		return lm.vision.Family()
	}
	return "Gemma 3 SigLIP"
}

func (lm *loadedModel) glmOcrVisionPrompt(g *glmVision, tm *chat.Template, system string, turns []chat.Turn, idx int, img imageRef) (visionInput, error) {
	pv, grid, err := multimodal.QwenPreprocess(img.Data, g.PP)
	if err != nil {
		return visionInput{}, err
	}
	imgHash := multimodal.HashImageBytes(img.Data)
	n := multimodal.QwenMergedTokens(grid, g.Merge)
	if err := lm.imageFitsContext(n, grid); err != nil {
		return visionInput{}, err
	}
	hiddenDim := lm.hiddenDim()
	features := func() ([]float32, error) {
		feats, err := g.Features(pv, grid)
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
	imgPos, imgLen := multimodal.FindImageRun(ids, g.ImgTok)
	if imgLen != n {
		return visionInput{}, fmt.Errorf("image placeholder run = %d image tokens, want %d (template mismatch)", imgLen, n)
	}
	return visionInput{IDs: ids, Features: features, ImgHash: imgHash, ImgPos: imgPos, ImgLen: imgLen, Grid: grid, Qwen: true}, nil
}

func (s *server) serveVisionChat(w http.ResponseWriter, r *http.Request, req chatReq, imgs []imageRef) {
	s.withModel(w, req.Model, func(lm *loadedModel) { s.serveVisionChatWith(w, r, req, imgs, lm) })
}

func (s *server) serveVisionChatWith(w http.ResponseWriter, r *http.Request, req chatReq, imgs []imageRef, lm *loadedModel) {
	if imgs[0].Audio {
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
	if err := lm.promptTooLargeForContext(chatInputBytes(req.Messages)); err != nil {
		writeErr(w, http.StatusBadRequest, err.Error())
		return
	}
	system, turns := messagesToTurns(req.Messages)
	if err := lm.glmOcrExtractionTurn(req.Sampling.ResponseFormat, turns); err != nil {
		writeErr(w, http.StatusBadRequest, err.Error())
		return
	}
	ts, terr := s.resolveThink(req.Think())
	if terr != nil {
		writeErr(w, http.StatusBadRequest, terr.Error())
		return
	}
	tm := lm.templateFor(ts)
	if req.Sampling.ConstrainsOutput() {
		tm = lm.constrainedTemplate(ts)
	}
	msgText, ordered := chatMediaText(req.Messages)
	vi, err := lm.visionPromptN(tm, system, turns, imgs, ordered, msgText)
	if err != nil {
		writeErr(w, http.StatusBadRequest, err.Error())
		return
	}
	req.Sampling.ConfidenceOK = true
	gr, err := lm.prepare(req.Sampling, vi.IDs, false)
	if err != nil {
		writePrepareErr(w, err)
		return
	}
	if !lm.enter(w, r, admissionRecord{promptIDs: gr.promptIDs, clock: gr.clock}, s.haltState) {
		return
	}
	defer lm.exit()
	id := "chatcmpl-" + reqID()
	gr.id = id
	created := time.Now().Unix()

	if req.Stream {
		ss, ok := sseStart(w)
		if !ok {
			return
		}
		streamCtx, cancelStream := ss.CancelOnFail(r.Context())
		defer cancelStream()
		sseSend(ss, chatChunk(id, created, lm.name, delta{Role: "assistant"}, nil))
		stopBeat := sseHeartbeat(ss)
		defer stopBeat()
		s.routeThink(lm, &gr, tm, turns, ts, func(t string) {
			sseSend(ss, chatChunk(id, created, lm.name, delta{ReasoningContent: t}, nil))
		})
		res, gerr := lm.driveVL(streamCtx, gr, vi, s.gens, s.jobs, func(t string) {
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
			sseSend(ss, map[string]any{"id": id, "goinfer_confidence": gr.conf.Payload()})
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
		vresp["goinfer_confidence"] = gr.conf.Payload()
	}
	writeJSON(w, http.StatusOK, gr.clock.AddTurn(vresp))
}

func (s *server) serveVisionMessages(w http.ResponseWriter, r *http.Request, req *anthropicReq, lm *loadedModel, imgs []imageRef) {
	if !lm.visionCapable() {
		writeAnthropicErr(w, http.StatusBadRequest, "invalid_request_error", "this model has no vision tower (start with --vision <dir> to enable image input)")
		return
	}
	if len(imgs) > maxImagesPerTurn {
		writeAnthropicErr(w, http.StatusBadRequest, "invalid_request_error", tooManyImages(len(imgs)))
		return
	}
	if err := lm.promptTooLargeForContext(anthropicInputBytes(req)); err != nil {
		writeAnthropicErr(w, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	system, turns, aerr := anthropicTurns(req)
	if aerr != nil {
		aerr.Write(w)
		return
	}
	ts, terr := s.resolveThink(req.ThinkRequest())
	if terr != nil {
		writeAnthropicErr(w, http.StatusBadRequest, "invalid_request_error", terr.Error())
		return
	}
	sm, serr := req.ToSampling()
	if serr != nil {
		serr.Write(w)
		return
	}
	tm := lm.templateFor(ts)
	if sm.ConstrainsOutput() {
		tm = lm.constrainedTemplate(ts)
	}
	msgText, ordered := anthropicMediaText(req)
	vi, err := lm.visionPromptN(tm, system, turns, imgs, ordered, msgText)
	if err != nil {
		writeAnthropicErr(w, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	gr, err := lm.prepare(sm, vi.IDs, false)
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
			return
		}
		w.Header().Set("Retry-After", "1")
		writeAnthropicErr(w, 529, "overloaded_error", fmt.Sprintf("model %q is busy; retry", lm.name))
		return
	}
	defer lm.exit()
	id := "msg_" + reqID()
	gr.id = id

	if req.Stream {
		ss, ok := anthropicSSEStart(w)
		if !ok {
			return
		}
		streamCtx, cancelStream := ss.CancelOnFail(r.Context())
		defer cancelStream()
		anthropicEvent(ss, "message_start", map[string]any{
			"type": "message_start",
			"message": map[string]any{
				"id": id, "type": "message", "role": "assistant", "model": lm.name,
				"content": []any{}, "stop_reason": nil, "stop_sequence": nil,
				"usage": map[string]any{"input_tokens": len(gr.promptIDs), "output_tokens": 0},
			},
		})
		anthropicEvent(ss, "ping", map[string]any{"type": "ping"})
		wantThinking := req.WantsThinking()
		th := streamframe.NewAnthropicThink(ss, wantThinking, nil)
		s.routeThink(lm, &gr, tm, turns, ts, th.Push)
		if wantThinking {
			streamMessagesThinking(ss, th, gr, func(onText func(string)) (turnResult, error) {
				return lm.driveVL(streamCtx, gr, vi, s.gens, s.jobs, onText)
			})
			return
		}
		anthropicEvent(ss, "content_block_start", map[string]any{
			"type": "content_block_start", "index": 0,
			"content_block": map[string]any{"type": "text", "text": ""},
		})
		stopBeat := sseHeartbeat(ss)
		defer stopBeat()
		res, gerr := lm.driveVL(streamCtx, gr, vi, s.gens, s.jobs, func(t string) {
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
	if req.WantsThinking() && rb.Len() > 0 {
		vcontent = append([]map[string]any{thinkingBlock(rb.String(), "")}, vcontent...)
		if sb.Len() == 0 {
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
