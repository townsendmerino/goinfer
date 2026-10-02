package serveapp

import (
	"fmt"
	"net/http"
	"slices"
	"strings"
	"time"

	"github.com/townsendmerino/aikit/vision"
	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/multimodal"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// imageSoftToken is the Gemma 3 image placeholder token; the per-image block and
// run-finder live in the vision package (shared with demo/agent).
const (
	imageSoftToken   = multimodal.ImageSoftToken
	maxImagesPerTurn = 1 // v1: a single image per request (the interleave API is shaped for N)
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
	imgHash        uint64                    // content hash of the raw image bytes, for P9(a) resident reuse
	imgPos, imgLen int
	grid           [3]int // Qwen m-RoPE grid (t,h,w in patch units); zero ⇒ Gemma 3
	qwen           bool
	gemma4         bool // selects GenerateGemma4VL in driveVL
}

// visionPrompt runs img through the tower and assembles the multimodal prompt: the
// text turns with the family's image block prepended to the last user turn, so the
// rendered+encoded ids carry a placeholder run the embed seam overrides.

// encodeVisionSegments encodes a vision prompt the way the TEXT path already does: the template's
// structural markers and the image block as Special segments, the user's own words as ordinary
// content the added-token trie never sees.
//
// M-22. The vision path called lm.encode(lm.tmpl.Render(...)) — Tokenizer.Encode, whose own doc
// says "do NOT use this on untrusted content" — while the text path had used EncodeSegments since
// M25. So a user message in an IMAGE request containing "<end_of_turn>\n<start_of_turn>model\n"
// (or "<|im_end|>…<|im_start|>system") became real control tokens and forged a turn boundary: the
// hardening reached one route and not the other, which is audit §0 theme 2 exactly.
//
// The image block has to stay SPECIAL — its sentinels and soft-token run are what FindImageRun
// locates and what the embed-by-vector seam replaces — so it cannot simply be prepended to the
// content segment, which is untrusted by construction. It is spliced back in as its own Special
// segment instead, and a block that fails to splice is an error rather than a prompt that silently
// tokenizes the sentinels as text (the imgLen check downstream would catch it, but late and with a
// misleading message about a template mismatch).
func encodeVisionSegments(lm *loadedModel, tm *chat.Template, system string, turns []chat.Turn, block string) ([]int, error) {
	segs, err := spliceImageBlock(tm.RenderSegments(system, turns), block)
	if err != nil {
		return nil, err
	}
	return lm.tk.EncodeSegments(segs, false)
}

// spliceImageBlock re-tags the image block as its own Special segment, splitting the content
// segment that contains it.
//
// The block does NOT start its segment: the template renders the role prefix and the message body
// into one non-special span, so a Gemma-4 user turn arrives as "user\n<start_of_image>…<end_of_image>\nhello".
// The first cut of this looked for the block as a PREFIX, found it nowhere, and would have refused
// every vision request — caught by the test, which is why the test drives this function rather than
// re-implementing its logic beside it.
func spliceImageBlock(segs []tokenizer.Segment, block string) ([]tokenizer.Segment, error) {
	// V-19 (docs/review-2026-09-04.md): search from the END and splice the LAST occurrence, not
	// the first. The block is always appended to the CURRENT (last) user turn, which renders
	// last; an EARLIER turn that happens to contain the same literal text — a user asking what
	// the sentinel means, say, as ordinary words — must not be mistaken for it. Splicing the
	// wrong occurrence reopens the exact special-token-forging class M-22 closed, just for this
	// sentinel instead of a role marker: the real image tokens stay unspliced plain text (and
	// fail downstream with a misleading "template mismatch"), while unrelated earlier text gets
	// tagged Special and parsed as sentinels it was never meant to be.
	segIdx := -1
	for i, seg := range slices.Backward(segs) {
		if !seg.Special && strings.Contains(seg.Text, block) {
			segIdx = i
			break
		}
	}
	if segIdx < 0 {
		return nil, fmt.Errorf("vision: the image block was not found in the rendered prompt " +
			"(template changed?); refusing to encode its sentinels as ordinary text")
	}
	sg := segs[segIdx]
	i := strings.LastIndex(sg.Text, block) // last occurrence within the segment too, same reason
	out := make([]tokenizer.Segment, 0, len(segs)+2)
	out = append(out, segs[:segIdx]...)
	if before := sg.Text[:i]; before != "" {
		out = append(out, tokenizer.Segment{Text: before}) // the template's role prefix
	}
	out = append(out, tokenizer.Segment{Text: block, Special: true})
	if after := sg.Text[i+len(block):]; after != "" {
		out = append(out, tokenizer.Segment{Text: after}) // the user's own words
	}
	out = append(out, segs[segIdx+1:]...)
	return out, nil
}

// tm is the per-request template (think.go): the model's own, switched to the request's thinking mode.
func (lm *loadedModel) visionPrompt(tm *chat.Template, system string, turns []chat.Turn, img imageRef) (visionInput, error) {
	if lm.tmpl == nil {
		return visionInput{}, fmt.Errorf("this model has no chat template for vision")
	}
	idx := lastUserTurn(turns)
	if idx < 0 {
		return visionInput{}, fmt.Errorf("no user turn to attach the image to")
	}
	if lm.glm != nil {
		return lm.glmOcrVisionPrompt(tm, system, turns, idx, img)
	}
	if lm.qwenEnc != nil || lm.qwen3 != nil {
		return lm.qwenVisionPrompt(tm, system, turns, idx, img)
	}
	if lm.gemma4Enc != nil {
		return lm.gemma4VisionPrompt(tm, system, turns, idx, img)
	}
	pv, err := vision.Preprocess(img.data, lm.vcfg)
	if err != nil {
		return visionInput{}, err
	}
	imgHash := multimodal.HashImageBytes(img.data)
	n := lm.vproj.MMTokens()
	hiddenDim := lm.model.Config().HiddenDim
	features := func() ([]float32, error) {
		hidden, err := lm.venc.Forward(pv.Data)
		if err != nil {
			return nil, fmt.Errorf("vision encoder: %w", err)
		}
		feats, err := lm.vproj.Forward(hidden)
		if err != nil {
			return nil, fmt.Errorf("vision projector: %w", err)
		}
		if len(feats) != n*hiddenDim {
			return nil, fmt.Errorf("projector emitted %d features, want %d", len(feats), n*hiddenDim)
		}
		return feats, nil
	}
	block := multimodal.Gemma3PromptBlock(n)
	turns[idx].Content = block + turns[idx].Content
	ids, err := encodeVisionSegments(lm, tm, system, turns, block)
	if err != nil {
		return visionInput{}, fmt.Errorf("encode: %w", err)
	}
	imgPos, imgLen := multimodal.FindImageRun(ids, lm.vimgTok)
	if imgLen != n {
		return visionInput{}, fmt.Errorf("image placeholder run = %d soft tokens, want %d (tokenizer/template mismatch)", imgLen, n)
	}
	return visionInput{ids: ids, features: features, imgHash: imgHash, imgPos: imgPos, imgLen: imgLen}, nil
}

// qwenForward runs whichever tower this model carries on the shared Qwen-shaped route: the Qwen2.5-VL ViT (eager), the
// Qwen3.5+ tower or the GLM-OCR tower (both loaded on first use). glmOcrVisionPrompt (glm_ocr_vision.go) calls it too.
func (lm *loadedModel) qwenForward(pv []float32, grid [3]int) ([]float32, error) {
	if lm.glm != nil {
		enc, err := lm.glm.encoder()
		if err != nil {
			return nil, err
		}
		return enc.Forward(pv, [][3]int{grid})
	}
	if lm.qwen3 != nil {
		enc, err := lm.qwen3.encoder()
		if err != nil {
			return nil, err
		}
		return enc.Forward(pv, [][3]int{grid})
	}
	return lm.qwenEnc.Forward(pv, [][3]int{grid})
}

// qwenVisionPrompt is the Qwen2.5-VL and Qwen3.5+ image path: smart-resize preprocess → ViT +
// merger (the merged features replace the <|image_pad|> run) → the prompt with the
// vision block prepended. The grid drives m-RoPE in GenerateQwenVL.
func (lm *loadedModel) qwenVisionPrompt(tm *chat.Template, system string, turns []chat.Turn, idx int, img imageRef) (visionInput, error) {
	pv, grid, err := multimodal.QwenPreprocess(img.data, lm.qwenPP)
	if err != nil {
		return visionInput{}, err
	}
	imgHash := multimodal.HashImageBytes(img.data)
	n := multimodal.QwenMergedTokens(grid, lm.qwenMerge)
	hiddenDim := lm.model.Config().HiddenDim
	features := func() ([]float32, error) {
		feats, err := lm.qwenForward(pv, grid)
		if err != nil {
			return nil, fmt.Errorf("qwen vision encoder: %w", err)
		}
		if len(feats) != n*hiddenDim {
			return nil, fmt.Errorf("qwen encoder emitted %d features, want %d", len(feats), n*hiddenDim)
		}
		return feats, nil
	}
	// M-38 (audit-2026-09-10): Qwen2.5-VL's real chat_template.json (verified live against
	// Qwen/Qwen2.5-VL-7B-Instruct) splices <|vision_start|><|image_pad|><|vision_end|> inline
	// with NO adjacent newline on either side — the trailing "\n" this used to append doesn't
	// exist in the real template.
	block := multimodal.QwenImageBlock(n)
	turns[idx].Content = block + turns[idx].Content
	ids, err := encodeVisionSegments(lm, tm, system, turns, block)
	if err != nil {
		return visionInput{}, fmt.Errorf("encode: %w", err)
	}
	imgPos, imgLen := multimodal.FindImageRun(ids, lm.qwenImgTok)
	if imgLen != n {
		return visionInput{}, fmt.Errorf("image placeholder run = %d pads, want %d (template mismatch)", imgLen, n)
	}
	return visionInput{ids: ids, features: features, imgHash: imgHash, imgPos: imgPos, imgLen: imgLen, grid: grid, qwen: true}, nil
}

// glmOcrVisionPrompt is the GLM-OCR image path: smart-resize preprocess (halved pixel bounds) -> the tower (the merged rows
// replace the <|image|> run) -> the prompt with the image block prepended to the last user turn. The grid drives m-RoPE in
// GenerateQwenVL, exactly as for Qwen.
//
// The image LEADS the user turn and the task prompt follows it with no separator (the checkpoint's own template:
// `<|begin_of_image|><|image|>x N<|end_of_image|>Text Recognition:`); a request with no text part gets plain text
// recognition rather than an image with nothing to do. The task is chosen by the user's own text ("Text Recognition:",
// "Formula Recognition:", "Table Recognition:", multimodal.GlmOcrPrompt*).
func (lm *loadedModel) glmOcrVisionPrompt(tm *chat.Template, system string, turns []chat.Turn, idx int, img imageRef) (visionInput, error) {
	pv, grid, err := multimodal.QwenPreprocess(img.data, lm.qwenPP)
	if err != nil {
		return visionInput{}, err
	}
	imgHash := multimodal.HashImageBytes(img.data)
	n := multimodal.QwenMergedTokens(grid, lm.qwenMerge)
	if err := lm.imageFitsContext(n, grid); err != nil { // before the tower: minutes of CPU for an answer that is known now
		return visionInput{}, err
	}
	hiddenDim := lm.model.Config().HiddenDim
	features := func() ([]float32, error) {
		feats, err := lm.qwenForward(pv, grid)
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
	imgPos, imgLen := multimodal.FindImageRun(ids, lm.qwenImgTok)
	if imgLen != n {
		return visionInput{}, fmt.Errorf("image placeholder run = %d image tokens, want %d (template mismatch)", imgLen, n)
	}
	return visionInput{ids: ids, features: features, imgHash: imgHash, imgPos: imgPos, imgLen: imgLen, grid: grid, qwen: true}, nil
}

// gemma4VisionPrompt is the Gemma 4 image path: aspect-ratio-preserving preprocess
// → the tower (projection baked in, no separate projector) → the prompt with the
// image block prepended. n (soft-token count) is computed from THIS image's own
// patch grid (multimodal.Gemma4PooledTokens), not a fixed per-checkpoint constant
// — see that function's doc comment. CPU-only v1 (decoder.GenerateGemma4VL): see
// its own doc comment for why no resident GPU bridge is attempted here.
func (lm *loadedModel) gemma4VisionPrompt(tm *chat.Template, system string, turns []chat.Turn, idx int, img imageRef) (visionInput, error) {
	patches, positionIDs, err := vision.Gemma4Preprocess(img.data, lm.gemma4MaxSoft)
	if err != nil {
		return visionInput{}, err
	}
	imgHash := multimodal.HashImageBytes(img.data)
	n := multimodal.Gemma4PooledTokens(positionIDs, lm.gemma4Enc.Cfg.PoolingKernelSize)
	hiddenDim := lm.model.Config().HiddenDim
	features := func() ([]float32, error) {
		feats, err := lm.gemma4Enc.Forward(patches, positionIDs)
		if err != nil {
			return nil, fmt.Errorf("gemma4 vision encoder: %w", err)
		}
		if len(feats) != n*hiddenDim {
			return nil, fmt.Errorf("gemma4 encoder emitted %d features, want %d", len(feats), n*hiddenDim)
		}
		return feats, nil
	}
	// M-38 (audit-2026-09-10): Gemma 4's own processor (processing_gemma4.py, verified against the
	// real transformers source) does f"{boi_token}{image_tokens}{eoi_token}" — no adjacent
	// newline at all, unlike the trailing "\n" this used to append.
	block := multimodal.Gemma4ImageBlock(n)
	turns[idx].Content = block + turns[idx].Content
	ids, err := encodeVisionSegments(lm, tm, system, turns, block)
	if err != nil {
		return visionInput{}, fmt.Errorf("encode: %w", err)
	}
	imgPos, imgLen := multimodal.FindImageRun(ids, lm.gemma4ImgTok)
	if imgLen != n {
		return visionInput{}, fmt.Errorf("image placeholder run = %d soft tokens, want %d (tokenizer/template mismatch)", imgLen, n)
	}
	return visionInput{ids: ids, features: features, imgHash: imgHash, imgPos: imgPos, imgLen: imgLen, gemma4: true}, nil
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
	if !lm.visionCapable() {
		writeErr(w, http.StatusBadRequest, "this model has no vision tower (start with --vision <dir> to enable image input)")
		return
	}
	if len(imgs) > maxImagesPerTurn {
		writeErr(w, http.StatusBadRequest, fmt.Sprintf("v1 supports %d image per request, got %d", maxImagesPerTurn, len(imgs)))
		return
	}
	// G1c, extended (audit-2026-09-02 M-21). Image bytes are excluded (chatInputBytes counts tokenizable TEXT), so this cannot reject a
	// valid image request on a small-context model.
	if err := lm.promptTooLargeForContext(chatInputBytes(req.Messages)); err != nil {
		writeErr(w, http.StatusBadRequest, err.Error())
		return
	}
	system, turns := messagesToTurns(req.Messages)
	ts, terr := s.resolveThink(req.think())
	if terr != nil {
		writeErr(w, http.StatusBadRequest, terr.Error())
		return
	}
	tm := lm.templateFor(ts)
	if req.sampling.constrainsOutput() {
		tm = lm.constrainedTemplate(ts)
	}
	vi, err := lm.visionPrompt(tm, system, turns, imgs[0])
	if err != nil {
		writeErr(w, http.StatusBadRequest, err.Error())
		return
	}
	gr, err := lm.prepare(req.sampling, vi.ids, false)
	if err != nil {
		writeErr(w, prepareErrStatus(err), err.Error())
		return
	}
	if !lm.enter(w, r, admissionRecord{promptIDs: gr.promptIDs}, s.haltState) {
		return
	}
	defer lm.exit()
	id := "chatcmpl-" + reqID()
	gr.id = id // K1: registers this generation for cancel-by-id
	created := time.Now().Unix()

	if req.Stream {
		ss, ok := sseStart(w)
		if !ok {
			return
		}
		sseSend(ss, chatChunk(id, created, lm.name, delta{Role: "assistant"}, nil))
		// nComp was discarded here; include_usage needs the real generated-token count,
		// which no count of emitted chunks can report (M-26).
		//
		// N-24 (docs/audit-2026-09-10.md): nothing else is sent before the first token, and on
		// CPU an image prefill can take minutes — the M-19 gate that catches a missing heartbeat
		// on the text-only lm.drive( sites never enumerated the driveVL sites at all.
		stopBeat := sseHeartbeat(ss)
		s.routeThink(lm, &gr, tm, turns, ts, func(t string) {
			sseSend(ss, chatChunk(id, created, lm.name, delta{ReasoningContent: t}, nil))
		})
		finish, nComp, _, reused, cancelReason, gerr := lm.driveVL(r.Context(), gr, vi, s.gens, s.jobs, func(t string) {
			sseSend(ss, chatChunk(id, created, lm.name, delta{Content: t}, nil))
		})
		stopBeat()
		if gerr != nil {
			sseErr(ss, "generation failed: "+gerr.Error())
			sseDone(ss)
			return
		}
		sseSend(ss, chatChunk(id, created, lm.name, delta{}, &finish))
		if cancelReason != "" {
			sseSend(ss, map[string]any{"goinfer_cancelled": map[string]any{"id": id, "reason": cancelReason}})
		}
		sendUsage(ss, req.StreamOptions, id, created, lm.name,
			usage{PromptTokens: len(gr.promptIDs), CompletionTokens: nComp, TotalTokens: len(gr.promptIDs) + nComp, PrefillReusedTokens: reused})
		sseDone(ss)
		return
	}
	var sb, rb strings.Builder
	s.routeThink(lm, &gr, tm, turns, ts, func(t string) { rb.WriteString(t) })
	finish, nComp, _, reused, cancelReason, gerr := lm.driveVL(r.Context(), gr, vi, s.gens, s.jobs, func(t string) { sb.WriteString(t) })
	if gerr != nil {
		writeServerErr(w, "generation failed: "+gerr.Error())
		return
	}
	if cancelReason != "" {
		writeErr(w, statusCancelled, "generation cancelled: "+cancelReason)
		return
	}
	vmsg := map[string]any{"role": "assistant", "content": sb.String()}
	if rb.Len() > 0 {
		vmsg["reasoning_content"] = rb.String()
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"id": id, "object": "chat.completion", "created": created, "model": lm.name,
		"choices": []any{map[string]any{
			"index":         0,
			"message":       vmsg,
			"finish_reason": finish,
		}},
		"usage": usage{PromptTokens: len(gr.promptIDs), CompletionTokens: nComp, TotalTokens: len(gr.promptIDs) + nComp, PrefillReusedTokens: reused},
	})
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
		writeAnthropicErr(w, http.StatusBadRequest, "invalid_request_error", fmt.Sprintf("v1 supports %d image per request, got %d", maxImagesPerTurn, len(imgs)))
		return
	}
	// G1c, extended (audit-2026-09-02 M-21). NOT in the audit's list of five — found by widening
	// the anti-drift gate to prompt builders that tokenize transitively, which is how the vision
	// routes hide: this function contains no tokenizer call of its own, visionPrompt does. Image
	// bytes are excluded from the count, so a valid image request is never rejected by it.
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
	tm := lm.templateFor(ts)
	vi, err := lm.visionPrompt(tm, system, turns, imgs[0])
	if err != nil {
		writeAnthropicErr(w, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	gr, err := lm.prepare(req.toSampling(), vi.ids, false)
	if err != nil {
		writeAnthropicErr(w, prepareErrStatus(err), "invalid_request_error", err.Error())
		return
	}
	if ok, haltReason := lm.tryEnter(r.Context(), admissionRecord{promptIDs: gr.promptIDs}, s.haltState); !ok {
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
	gr.id = id // K1: registers this generation for cancel-by-id

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
			streamMessagesThinking(ss, th, func(onText func(string)) (string, int, string, string, error) {
				finish, nComp, stopSeq, _, cancelReason, gerr := lm.driveVL(r.Context(), gr, vi, s.gens, s.jobs, onText)
				return finish, nComp, stopSeq, cancelReason, gerr
			})
			return
		}
		anthropicEvent(ss, "content_block_start", map[string]any{
			"type": "content_block_start", "index": 0,
			"content_block": map[string]any{"type": "text", "text": ""},
		})
		// N-24 (docs/audit-2026-09-10.md): the ping above is one-shot, not a keep-alive — an
		// image prefill on CPU can take minutes with nothing sent until the first token.
		stopBeat := sseHeartbeat(ss)
		finish, nComp, stopSeq, _, cancelReason, gerr := lm.driveVL(r.Context(), gr, vi, s.gens, s.jobs, func(t string) {
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
		reason, seq := anthropicStopReason(finish, stopSeq)
		anthropicMessageEnd(ss, reason, seq, nComp, cancelReason)
		return
	}
	var sb, rb strings.Builder
	s.routeThink(lm, &gr, tm, turns, ts, func(t string) { rb.WriteString(t) })
	finish, nComp, stopSeq, _, cancelReason, gerr := lm.driveVL(r.Context(), gr, vi, s.gens, s.jobs, func(t string) { sb.WriteString(t) })
	if gerr != nil {
		writeAnthropicErr(w, http.StatusInternalServerError, "api_error", "generation failed: "+gerr.Error())
		return
	}
	if cancelReason != "" {
		writeAnthropicErr(w, statusCancelled, "cancelled", "generation cancelled: "+cancelReason)
		return
	}
	reason, seq := anthropicStopReason(finish, stopSeq)
	vcontent := []map[string]any{textBlock(sb.String())}
	if req.wantsThinking() && rb.Len() > 0 {
		vcontent = append([]map[string]any{thinkingBlock(rb.String())}, vcontent...)
		if sb.Len() == 0 { // cut off while thinking: no answer, so no text block
			vcontent = vcontent[:1]
		}
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"id": id, "type": "message", "role": "assistant", "model": lm.name,
		"content":       vcontent,
		"stop_reason":   reason,
		"stop_sequence": seq,
		"usage":         map[string]any{"input_tokens": len(gr.promptIDs), "output_tokens": nComp},
	})
}
