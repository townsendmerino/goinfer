package decoder

import (
	"context"
	"fmt"

	"github.com/townsendmerino/goinfer/internal/runtime/rope"
)

// ExactPrefill reports whether this model runs its prefill with bit-exact f64-accumulated attention.
func (m *Model) ExactPrefill() bool { return m.opts.ExactPrefill }

func (m *Model) canBatchN(K int) bool {
	e := m.cpuEngine()
	return e.CanBatchN(K)
}

func (m *Model) hasRecurrentState() bool {
	e := m.cpuEngine()
	return e.HasRecurrentState()
}

// specRollbackSafe reports whether speculative decode's rollback (KVCache.TruncateTo after a partial accept) correctly restores
// this model's state. It is true for softmax/GQA and MLA, whose state lives in the cache and is resliced. It is false for every
// family hasRecurrentState marks Recurrent: TruncateTo does not roll back their in-place state, so verifying a K-token block
// over-advances it and the next round decodes from it, a silent distribution bug rather than a crash. It is also false for
// windowed models (see below). The n-gram speculative entry points refuse unsafe models and the caller falls back to plain
// decode; no checkpoint-at-block-start/restore path exists.
func (m *Model) specRollbackSafe() bool { return m.specRollbackSafeVia(nil, 0) }

// specRollbackSafeVia is specRollbackSafe for a loop that verifies through fold, a resident's recurrent fold path, in passes of up
// to width rows (BlockSpec; nil for a loop that rolls back with TruncateTo or ForwardN). A recurrent model is admitted only there:
// the fold path never writes the committed recurrent state during a verify and folds only the committed rows into it
// (RecurrentSpecAvailable). The resident reports rows only when it serves every recurrent layer, and the model's own layers must
// all be Gated DeltaNet (RecurrentSpecArenaBytes), the kind the path is built for.
func (m *Model) specRollbackSafeVia(fold ResidentRuntimeSubsystem, width int) bool {
	a := m.w.Arch
	if m.hasRecurrentState() && (fold == nil || width < 1 || width > fold.RecurrentSpecRows() || RecurrentSpecArenaBytes(a, 1) == 0) {
		return false
	}
	// A staged sliding-window cache stores local layers in physical rings. Once a ring wraps (context > window), a rollback of more
	// than one position cannot restore the evicted positions, so verify would read stale history and diverge. The resident
	// positional path is itself safe, but the speculative loops can use the staged ring even when m.resident != nil (and a resident
	// CAS loss falls back to staged mid-flight), so windowed models are refused for speculation unconditionally; the cost is plain
	// decode. That keeps the staged rollback sites (KVCache.TruncateTo) to ring-free caches, where it is always exact. Re-enabling
	// windowed speculation means consuming an exact-rollback bool at each rollback site.
	if a.SlidingWindow > 0 {
		return false
	}
	return true
}

func (m *Model) forwardLayersN(reqCtx context.Context, ids []int, cache *KVCache, fastAttn bool) ([]float32, error) {
	e := m.cpuEngine()
	return e.ForwardLayersN(reqCtx, ids, unwrapCache(cache), fastAttn)
}

func (m *Model) embedN(ids []int) []float32 {
	e := m.cpuEngine()
	return e.EmbedN(ids)
}

func (m *Model) embedNInto(h []float32, ids []int) []float32 {
	e := m.cpuEngine()
	return e.EmbedNInto(h, ids)
}

func (m *Model) runLayersFromEmbedN(reqCtx context.Context, h []float32, cache *KVCache, fastAttn bool) ([]float32, error) {
	e := m.cpuEngine()
	return e.RunLayersFromEmbedN(reqCtx, h, unwrapCache(cache), fastAttn)
}

func (m *Model) lmHeadN(h []float32, M int) []float32 {
	e := m.cpuEngine()
	return e.LMHeadN(h, M)
}

func (m *Model) forwardN(reqCtx context.Context, ids []int, cache *KVCache) ([][]float32, error) {
	e := m.cpuEngine()
	return e.ForwardN(reqCtx, ids, unwrapCache(cache))
}

func (m *Model) prefillLogits(ctx context.Context, prompt []int, cache *KVCache) ([]float32, error) {
	e := m.cpuEngine()
	return e.PrefillLogits(ctx, prompt, unwrapCache(cache))
}

func (m *Model) prefillPassesN(ctx context.Context, n int, cache *KVCache, embed func(dst []float32, lo, hi int), keep func(lo int, rows []float32)) ([]float32, error) {
	e := m.cpuEngine()
	return e.PrefillPassesN(ctx, n, unwrapCache(cache), embed, keep)
}

func (m *Model) lmHeadNInto(h []float32, M int, dst []float32) []float32 {
	e := m.cpuEngine()
	return e.LMHeadNInto(h, M, dst)
}

func (m *Model) qwen35BatchN(K int, cache *KVCache) bool {
	e := m.cpuEngine()
	return e.Qwen35BatchN(K, unwrapCache(cache))
}

func (m *Model) qwen35BatchNAnyPos(K int, cache *KVCache) bool {
	e := m.cpuEngine()
	return e.Qwen35BatchNAnyPos(K, unwrapCache(cache))
}

func (m *Model) runLayersQwen35N(reqCtx context.Context, h []float32, cache *KVCache) ([]float32, error) {
	e := m.cpuEngine()
	return e.RunLayersQwen35N(reqCtx, h, unwrapCache(cache))
}

func (m *Model) runLayersGemma4FromEmbedN(reqCtx context.Context, h []float32, pleIDs []int, spans []ImageSpan, cache *KVCache) ([]float32, error) {
	e := m.cpuEngine()
	return e.RunLayersGemma4FromEmbedN(reqCtx, h, pleIDs, spans, unwrapCache(cache))
}

// prefillLogitsVL prefills a multimodal prompt and returns the last-position
// logits: the text token embeddings with the projected vision embeddings
// (imageEmbeds, [imgLen*HiddenDim]) substituted at the <image> placeholder run
// [imgPos, imgPos+imgLen), under a bidirectional attention mask over that block
// (so the image tokens see each other). Requires the batched path — the
// bidirectional image-block attention lives in attendBatchedHeads. The injected
// embeddings are RAW (the projector output), matching HF's masked_scatter, which
// overwrites the scaled placeholder embed. See docs/multimodal.md §4–5.
func (m *Model) prefillLogitsVL(ctx context.Context, ids []int, imageEmbeds []float32, imgPos, imgLen int, cache *KVCache) ([]float32, error) {
	return m.prefillLogitsVLSpans(ctx, ids, []ImageSpan{{Pos: imgPos, Len: imgLen}}, imageEmbeds, cache, false)
}

// prefillLogitsVLSpans is prefillLogitsVL for several images: each span is its own bidirectional block, and
// imageEmbeds holds every span's rows concatenated in span order.
// causal (Pixtral) splices the spans without a bidirectional block: plain causal attention over the prompt.
func (m *Model) prefillLogitsVLSpans(ctx context.Context, ids []int, spans []ImageSpan, imageEmbeds []float32, cache *KVCache, causal bool) ([]float32, error) {
	var hN []float32
	var err error
	if causal && m.w.Arch.LFM2 != nil {
		hN, err = m.prefillHiddenLFM2VL(ctx, ids, spans, imageEmbeds, cache)
	} else {
		hN, err = m.prefillVLRows(ctx, ids, spans, imageEmbeds, cache, causal, false)
	}
	if err != nil {
		return nil, err
	}
	hidden := m.w.Arch.HiddenDim
	return m.lmHeadNInto(hN[len(hN)-hidden:], 1, cache.c.HeadLogits(m.w.Arch.VocabSize)), nil
}

// prefillHiddenVLSpans runs prefillLogitsVLSpans's prefill and returns the final hidden rows of every position (before
// the LM head), so a test can grade every position, not only the last, through the production path.
func (m *Model) prefillHiddenVLSpans(ctx context.Context, ids []int, spans []ImageSpan, imageEmbeds []float32, cache *KVCache, causal bool) ([]float32, error) {
	if causal && m.w.Arch.LFM2 != nil { // LFM2-VL: lfm2 has no batched pass; its own forward, one row at a time
		return m.prefillHiddenLFM2VL(ctx, ids, spans, imageEmbeds, cache)
	}
	return m.prefillVLRows(ctx, ids, spans, imageEmbeds, cache, causal, true)
}

// prefillVLRows runs the image prefill of prefillHiddenVLSpans in passes (prefillPassesN), each embedding its own rows with the
// raw projected features spliced in. It returns every position's rows when all, else the last pass's.
func (m *Model) prefillVLRows(ctx context.Context, ids []int, spans []ImageSpan, imageEmbeds []float32, cache *KVCache, causal, all bool) ([]float32, error) {
	if !m.canBatchN(len(ids)) {
		return nil, fmt.Errorf("decoder: multimodal prefill needs the batched path (canBatchN false)")
	}
	hidden := m.w.Arch.HiddenDim
	if err := checkImageSpans(spans, len(ids), imageEmbeds, hidden); err != nil {
		return nil, err
	}
	if !causal {
		cache.SetImageBlocks(imageSpanBlocks(spans))
	}
	return m.prefillSpliced(ctx, ids, spans, imageEmbeds, cache, all)
}

// prefillSpliced runs ids through prefillPassesN with each pass's image rows replaced by their features (raw, no embed scale),
// returning every position's rows when all, else the last pass's.
func (m *Model) prefillSpliced(ctx context.Context, ids []int, spans []ImageSpan, feats []float32, cache *KVCache, all bool) ([]float32, error) {
	hidden := m.w.Arch.HiddenDim
	var out []float32
	var keep func(lo int, rows []float32)
	if all {
		out = make([]float32, len(ids)*hidden)
		keep = func(lo int, rows []float32) { copy(out[lo*hidden:], rows) }
	}
	h, err := m.prefillPassesN(ctx, len(ids), cache, func(dst []float32, lo, hi int) {
		m.embedNInto(dst, ids[lo:hi])
		spliceImageSpansRange(dst, spans, feats, hidden, lo, hi)
	}, keep)
	if all {
		return out, err
	}
	return h, err
}

// prefillHiddenLFM2VL is prefillHiddenVLSpans for lfm2 (LFM2-VL): the text embeddings with the image rows spliced
// in, each run through the per-token walker in order (the conv windows see every row, image rows included), and each
// position's hidden state final-normed, as the batched entry returns them.
func (m *Model) prefillHiddenLFM2VL(ctx context.Context, ids []int, spans []ImageSpan, imageEmbeds []float32, cache *KVCache) ([]float32, error) {
	arch := m.w.Arch
	hidden := arch.HiddenDim
	if err := checkImageSpans(spans, len(ids), imageEmbeds, hidden); err != nil {
		return nil, err
	}
	h := m.embedN(ids)
	spliceImageSpans(h, spans, imageEmbeds, hidden)
	for i := range ids {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		row := h[i*hidden : (i+1)*hidden]
		out, err := m.runLayersFromEmbed(row, cache)
		if err != nil {
			return nil, err
		}
		m.finalNorm(out)
	}
	return h, nil
}

// prefillLogitsQwenVL prefills a Qwen2.5-VL multimodal prompt and returns the
// last-position logits: text embeddings with the merged vision features
// (imageFeats, [imgLen*HiddenDim] from the ViT+merger) substituted at the <image>
// run [imgPos,imgPos+imgLen), under m-RoPE 3D positions (mropePos, one (t,h,w) per
// absolute sequence position). Two deltas vs prefillLogitsVL (Gemma 3): the image
// tokens attend CAUSALLY — no bidirectional image block (Qwen's bidirectionality is
// inside the ViT, not the decoder) — and the rotary uses m-RoPE (rope.At reads
// cache.c.MRopePos). The merged features are RAW (no embed scale), matching HF's
// scatter into inputs_embeds.
func (m *Model) prefillLogitsQwenVL(ctx context.Context, ids []int, imageFeats []float32, imgPos, imgLen int, mropePos [][3]int, cache *KVCache) ([]float32, error) {
	return m.prefillLogitsQwenVLSpans(ctx, ids, []ImageSpan{{Pos: imgPos, Len: imgLen}}, imageFeats, mropePos, cache)
}

// prefillLogitsQwenVLSpans is prefillLogitsQwenVL for several images: every span's merged rows, concatenated in
// span order in imageFeats, replace that span's placeholder run, and mropePos covers them all (rope.MRoPEPositions with one
// grid per image).
func (m *Model) prefillLogitsQwenVLSpans(ctx context.Context, ids []int, spans []ImageSpan, imageFeats []float32, mropePos [][3]int, cache *KVCache) ([]float32, error) {
	if m.w.Arch.Qwen35 != nil { // the Gated-DeltaNet hybrids take their own prefill (batched when qwen35BatchNAnyPos applies)
		return m.prefillLogitsQwen35VL(ctx, ids, imageFeats, spans, mropePos, cache)
	}
	if !m.canBatchN(len(ids)) {
		return nil, fmt.Errorf("decoder: Qwen2.5-VL prefill needs the batched path (canBatchN false)")
	}
	hidden := m.w.Arch.HiddenDim
	if err := checkImageSpans(spans, len(ids), imageFeats, hidden); err != nil {
		return nil, err
	}
	if len(mropePos) != len(ids) {
		return nil, fmt.Errorf("decoder: mropePos len %d, want %d (one per token)", len(mropePos), len(ids))
	}
	cache.c.MRopePos = mropePos                              // rope.At switches to m-RoPE for this prefill
	cache.c.MRopeDelta = rope.MRoPEDelta(mropePos, len(ids)) // decode past the prefill rotates at seqPos+delta
	hN, err := m.prefillSpliced(ctx, ids, spans, imageFeats, cache, false)
	if err != nil {
		return nil, err
	}
	return m.lmHeadNInto(hN[len(hN)-hidden:], 1, cache.c.HeadLogits(m.w.Arch.VocabSize)), nil
}
