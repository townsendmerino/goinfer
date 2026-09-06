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
	feats          []float32
	imgPos, imgLen int
	grid           [3]int // Qwen m-RoPE grid (t,h,w in patch units); zero ⇒ Gemma 3
	qwen           bool
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
func encodeVisionSegments(lm *loadedModel, system string, turns []chat.Turn, block string) ([]int, error) {
	segs, err := spliceImageBlock(lm.tmpl.RenderSegments(system, turns), block)
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

func (lm *loadedModel) visionPrompt(system string, turns []chat.Turn, img imageRef) (visionInput, error) {
	if lm.tmpl == nil {
		return visionInput{}, fmt.Errorf("this model has no chat template for vision")
	}
	idx := lastUserTurn(turns)
	if idx < 0 {
		return visionInput{}, fmt.Errorf("no user turn to attach the image to")
	}
	if lm.qwenEnc != nil {
		return lm.qwenVisionPrompt(system, turns, idx, img)
	}
	pv, err := vision.Preprocess(img.data, lm.vcfg)
	if err != nil {
		return visionInput{}, err
	}
	hidden, err := lm.venc.Forward(pv.Data)
	if err != nil {
		return visionInput{}, fmt.Errorf("vision encoder: %w", err)
	}
	feats, err := lm.vproj.Forward(hidden)
	if err != nil {
		return visionInput{}, fmt.Errorf("vision projector: %w", err)
	}
	n := lm.vproj.MMTokens()
	block := multimodal.Gemma3ImageBlock(n) + "\n"
	turns[idx].Content = block + turns[idx].Content
	ids, err := encodeVisionSegments(lm, system, turns, block)
	if err != nil {
		return visionInput{}, fmt.Errorf("encode: %w", err)
	}
	imgPos, imgLen := multimodal.FindImageRun(ids, lm.vimgTok)
	if imgLen != n {
		return visionInput{}, fmt.Errorf("image placeholder run = %d soft tokens, want %d (tokenizer/template mismatch)", imgLen, n)
	}
	if len(feats) != n*lm.model.Config().HiddenDim {
		return visionInput{}, fmt.Errorf("projector emitted %d features, want %d", len(feats), n*lm.model.Config().HiddenDim)
	}
	return visionInput{ids: ids, feats: feats, imgPos: imgPos, imgLen: imgLen}, nil
}

// qwenVisionPrompt is the Qwen2.5-VL image path: smart-resize preprocess → ViT +
// merger (the merged features replace the <|image_pad|> run) → the prompt with the
// vision block prepended. The grid drives m-RoPE in GenerateQwenVL.
func (lm *loadedModel) qwenVisionPrompt(system string, turns []chat.Turn, idx int, img imageRef) (visionInput, error) {
	pv, grid, err := multimodal.QwenPreprocess(img.data, lm.qwenPP)
	if err != nil {
		return visionInput{}, err
	}
	feats, err := lm.qwenEnc.Forward(pv, [][3]int{grid})
	if err != nil {
		return visionInput{}, fmt.Errorf("qwen vision encoder: %w", err)
	}
	n := multimodal.QwenMergedTokens(grid, lm.qwenMerge)
	block := multimodal.QwenImageBlock(n) + "\n"
	turns[idx].Content = block + turns[idx].Content
	ids, err := encodeVisionSegments(lm, system, turns, block)
	if err != nil {
		return visionInput{}, fmt.Errorf("encode: %w", err)
	}
	imgPos, imgLen := multimodal.FindImageRun(ids, lm.qwenImgTok)
	if imgLen != n {
		return visionInput{}, fmt.Errorf("image placeholder run = %d pads, want %d (template mismatch)", imgLen, n)
	}
	if len(feats) != n*lm.model.Config().HiddenDim {
		return visionInput{}, fmt.Errorf("qwen encoder emitted %d features, want %d", len(feats), n*lm.model.Config().HiddenDim)
	}
	return visionInput{ids: ids, feats: feats, imgPos: imgPos, imgLen: imgLen, grid: grid, qwen: true}, nil
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
	vi, err := lm.visionPrompt(system, turns, imgs[0])
	if err != nil {
		writeErr(w, http.StatusBadRequest, err.Error())
		return
	}
	gr, err := lm.prepare(req.sampling, vi.ids, false)
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
		ss, ok := sseStart(w)
		if !ok {
			return
		}
		sseSend(ss, chatChunk(id, created, lm.name, delta{Role: "assistant"}, nil))
		// nComp was discarded here; include_usage needs the real generated-token count,
		// which no count of emitted chunks can report (M-26).
		finish, nComp, _, gerr := lm.driveVL(r.Context(), gr, vi, func(t string) {
			sseSend(ss, chatChunk(id, created, lm.name, delta{Content: t}, nil))
		})
		if gerr != nil {
			sseErr(ss, "generation failed: "+gerr.Error())
			sseDone(ss)
			return
		}
		sseSend(ss, chatChunk(id, created, lm.name, delta{}, &finish))
		sendUsage(ss, req.StreamOptions, id, created, lm.name,
			usage{len(gr.promptIDs), nComp, len(gr.promptIDs) + nComp})
		sseDone(ss)
		return
	}
	var sb strings.Builder
	finish, nComp, _, gerr := lm.driveVL(r.Context(), gr, vi, func(t string) { sb.WriteString(t) })
	if gerr != nil {
		writeServerErr(w, "generation failed: "+gerr.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"id": id, "object": "chat.completion", "created": created, "model": lm.name,
		"choices": []any{map[string]any{
			"index":         0,
			"message":       map[string]any{"role": "assistant", "content": sb.String()},
			"finish_reason": finish,
		}},
		"usage": usage{len(gr.promptIDs), nComp, len(gr.promptIDs) + nComp},
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
	vi, err := lm.visionPrompt(system, turns, imgs[0])
	if err != nil {
		writeAnthropicErr(w, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	gr, err := lm.prepare(req.toSampling(), vi.ids, false)
	if err != nil {
		writeAnthropicErr(w, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	if !lm.tryEnter() {
		w.Header().Set("Retry-After", "1")
		writeAnthropicErr(w, 529, "overloaded_error", fmt.Sprintf("model %q is busy; retry", lm.name))
		return
	}
	defer lm.exit()
	id := "msg_" + reqID()

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
		anthropicEvent(ss, "content_block_start", map[string]any{
			"type": "content_block_start", "index": 0,
			"content_block": map[string]any{"type": "text", "text": ""},
		})
		finish, nComp, stopSeq, gerr := lm.driveVL(r.Context(), gr, vi, func(t string) {
			anthropicEvent(ss, "content_block_delta", map[string]any{
				"type": "content_block_delta", "index": 0,
				"delta": map[string]any{"type": "text_delta", "text": t},
			})
		})
		if gerr != nil {
			anthropicEvent(ss, "content_block_stop", map[string]any{"type": "content_block_stop", "index": 0})
			anthropicStreamErr(ss, "generation failed: "+gerr.Error())
			return
		}
		anthropicEvent(ss, "content_block_stop", map[string]any{"type": "content_block_stop", "index": 0})
		reason, seq := anthropicStopReason(finish, stopSeq)
		anthropicMessageEnd(ss, reason, seq, nComp)
		return
	}
	var sb strings.Builder
	finish, nComp, stopSeq, gerr := lm.driveVL(r.Context(), gr, vi, func(t string) { sb.WriteString(t) })
	if gerr != nil {
		writeAnthropicErr(w, http.StatusInternalServerError, "api_error", "generation failed: "+gerr.Error())
		return
	}
	reason, seq := anthropicStopReason(finish, stopSeq)
	writeJSON(w, http.StatusOK, map[string]any{
		"id": id, "type": "message", "role": "assistant", "model": lm.name,
		"content":       []map[string]any{textBlock(sb.String())},
		"stop_reason":   reason,
		"stop_sequence": seq,
		"usage":         map[string]any{"input_tokens": len(gr.promptIDs), "output_tokens": nComp},
	})
}
