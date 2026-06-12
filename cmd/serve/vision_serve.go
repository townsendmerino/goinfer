package main

import (
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/townsendmerino/aikit/vision"
	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/multimodal"
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
	for i := len(turns) - 1; i >= 0; i-- {
		if turns[i].Role == "user" {
			return i
		}
	}
	return -1
}

// visionPrompt runs img through the tower (preprocess → SigLIP encoder →
// projector) and assembles the multimodal prompt: the text turns with the
// Gemma 3 image block prepended to the last user turn, so the rendered+encoded
// ids carry a run of image-soft-token ids the embed seam overrides. Returns the
// encoded ids, the projected features ([MMTokens*textHidden]), and the
// placeholder run [imgPos, imgPos+imgLen).
func (lm *loadedModel) visionPrompt(system string, turns []chat.Turn, img imageRef) (ids []int, feats []float32, imgPos, imgLen int, err error) {
	if lm.tmpl == nil {
		return nil, nil, 0, 0, fmt.Errorf("this model has no chat template for vision")
	}
	idx := lastUserTurn(turns)
	if idx < 0 {
		return nil, nil, 0, 0, fmt.Errorf("no user turn to attach the image to")
	}
	pv, err := vision.Preprocess(img.data, lm.vcfg)
	if err != nil {
		return nil, nil, 0, 0, err
	}
	hidden, err := lm.venc.Forward(pv.Data)
	if err != nil {
		return nil, nil, 0, 0, fmt.Errorf("vision encoder: %w", err)
	}
	feats, err = lm.vproj.Forward(hidden)
	if err != nil {
		return nil, nil, 0, 0, fmt.Errorf("vision projector: %w", err)
	}
	n := lm.vproj.MMTokens()
	turns[idx].Content = multimodal.Gemma3ImageBlock(n) + "\n" + turns[idx].Content
	ids = lm.encode(lm.tmpl.Render(system, turns))
	imgPos, imgLen = multimodal.FindImageRun(ids, lm.vimgTok)
	if imgLen != n {
		return nil, nil, 0, 0, fmt.Errorf("image placeholder run = %d soft tokens, want %d (tokenizer/template mismatch)", imgLen, n)
	}
	if len(feats) != n*lm.model.Config().HiddenDim {
		return nil, nil, 0, 0, fmt.Errorf("projector emitted %d features, want %d", len(feats), n*lm.model.Config().HiddenDim)
	}
	return ids, feats, imgPos, imgLen, nil
}

// serveVisionChat handles an OpenAI /v1/chat/completions request that carries an
// image. The image runs through the tower, the prompt is assembled with the
// image block, and generation goes through the multimodal path (driveVL, which
// bypasses the warm-KV session). usage.prompt_tokens includes the image tokens
// (they occupy real KV positions). Streaming + non-streaming.
func (s *server) serveVisionChat(w http.ResponseWriter, r *http.Request, req chatReq, imgs []imageRef) {
	lm := s.pick(req.Model)
	if lm == nil {
		s.modelNotFound(w, req.Model)
		return
	}
	if !lm.visionCapable() {
		writeErr(w, http.StatusBadRequest, "this model has no vision tower (start with --vision <dir> to enable image input)")
		return
	}
	if len(imgs) > maxImagesPerTurn {
		writeErr(w, http.StatusBadRequest, fmt.Sprintf("v1 supports %d image per request, got %d", maxImagesPerTurn, len(imgs)))
		return
	}
	system, turns := messagesToTurns(req.Messages)
	ids, feats, imgPos, imgLen, err := lm.visionPrompt(system, turns, imgs[0])
	if err != nil {
		writeErr(w, http.StatusBadRequest, err.Error())
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
		sseSend(w, f, chatChunk(id, created, lm.name, delta{Role: "assistant"}, nil))
		finish, _, _ := lm.driveVL(r.Context(), gr, feats, imgPos, imgLen, func(t string) {
			sseSend(w, f, chatChunk(id, created, lm.name, delta{Content: t}, nil))
		})
		sseSend(w, f, chatChunk(id, created, lm.name, delta{}, &finish))
		sseDone(w, f)
		return
	}
	var sb strings.Builder
	finish, nComp, _ := lm.driveVL(r.Context(), gr, feats, imgPos, imgLen, func(t string) { sb.WriteString(t) })
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
	system, turns, aerr := anthropicTurns(req) // image blocks skipped; text turns only
	if aerr != nil {
		aerr.write(w)
		return
	}
	ids, feats, imgPos, imgLen, err := lm.visionPrompt(system, turns, imgs[0])
	if err != nil {
		writeAnthropicErr(w, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	gr, err := lm.prepare(req.toSampling(), ids)
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
		f, ok := anthropicSSEStart(w)
		if !ok {
			return
		}
		anthropicEvent(w, f, "message_start", map[string]any{
			"type": "message_start",
			"message": map[string]any{
				"id": id, "type": "message", "role": "assistant", "model": lm.name,
				"content": []any{}, "stop_reason": nil, "stop_sequence": nil,
				"usage": map[string]any{"input_tokens": len(gr.promptIDs), "output_tokens": 0},
			},
		})
		anthropicEvent(w, f, "ping", map[string]any{"type": "ping"})
		anthropicEvent(w, f, "content_block_start", map[string]any{
			"type": "content_block_start", "index": 0,
			"content_block": map[string]any{"type": "text", "text": ""},
		})
		finish, nComp, stopSeq := lm.driveVL(r.Context(), gr, feats, imgPos, imgLen, func(t string) {
			anthropicEvent(w, f, "content_block_delta", map[string]any{
				"type": "content_block_delta", "index": 0,
				"delta": map[string]any{"type": "text_delta", "text": t},
			})
		})
		anthropicEvent(w, f, "content_block_stop", map[string]any{"type": "content_block_stop", "index": 0})
		reason, seq := anthropicStopReason(finish, stopSeq)
		anthropicMessageEnd(w, f, reason, seq, nComp)
		return
	}
	var sb strings.Builder
	finish, nComp, stopSeq := lm.driveVL(r.Context(), gr, feats, imgPos, imgLen, func(t string) { sb.WriteString(t) })
	reason, seq := anthropicStopReason(finish, stopSeq)
	writeJSON(w, http.StatusOK, map[string]any{
		"id": id, "type": "message", "role": "assistant", "model": lm.name,
		"content":       []map[string]any{textBlock(sb.String())},
		"stop_reason":   reason,
		"stop_sequence": seq,
		"usage":         map[string]any{"input_tokens": len(gr.promptIDs), "output_tokens": nComp},
	})
}
