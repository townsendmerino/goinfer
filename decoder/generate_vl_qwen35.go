package decoder

import (
	"context"
	"fmt"
)

// prefillLogitsQwen35VL is prefillLogitsQwenVL for the Gated-DeltaNet hybrid (qwen3_5 /
// qwen3_5_moe, P8a): the same splice — the merged tower rows replace the placeholder run
// [imgPos, imgPos+imgLen) raw, no embed scale, exactly HF's inputs_embeds.masked_scatter — and the
// same m-RoPE positions on the full-attention layers, but through the per-token
// runLayersQwen35FromEmbed loop, because the recurrent state must see every token in order and
// there is no batched hybrid prefill (canBatchN is false for this family).
//
// It leaves cache.mropePos / cache.mropeDelta set so decode past the prompt rotates at
// seqPos+delta, exactly as prefillLogitsQwenVL does.
func (m *Model) prefillLogitsQwen35VL(ctx context.Context, ids []int, imageFeats []float32, spans []ImageSpan, mropePos [][3]int, cache *KVCache) ([]float32, error) {
	arch := m.w.arch
	hidden := arch.HiddenDim
	if err := checkImageSpans(spans, len(ids), imageFeats, hidden); err != nil {
		return nil, err
	}
	if len(mropePos) != len(ids) {
		return nil, fmt.Errorf("decoder: mropePos len %d, want %d (one per token)", len(mropePos), len(ids))
	}
	if err := m.checkQwen35VLReady(); err != nil {
		return nil, err
	}
	cache.mropePos = mropePos
	cache.mropeDelta = mropeDelta(mropePos, len(ids))
	if !qwen35VLPerToken && m.qwen35BatchNAnyPos(len(ids), cache) {
		return m.prefillQwen35VLBatched(ctx, ids, imageFeats, spans, cache)
	}
	return m.prefillQwen35VLPerToken(ctx, ids, imageFeats, spans, cache)
}

// prefillQwen35VLBatched is P26: every projection one M=len(ids) matmul, so each weight is read once for the prompt instead of
// once per token (a 662-token image turn on Qwen3.5-0.8B took ~34 s one token at a time, which is decode speed). The DeltaNet
// recurrence stays sequential inside runLayersQwen35N; its full-attention layers rotate q and k by cache.mropePos exactly as the
// per-token loop does. NOT bit-identical to that loop (a batched matmul can reduce in a different order):
// TestQwen35VL_batchedPrefillMatchesPerToken bounds the difference, and the HF-golden tests hold either path to the same bars.
// The caller has checked qwen35BatchNAnyPos and set cache.mropePos / mropeDelta.
func (m *Model) prefillQwen35VLBatched(ctx context.Context, ids []int, imageFeats []float32, spans []ImageSpan, cache *KVCache) ([]float32, error) {
	hidden := m.w.arch.HiddenDim
	h := make([]float32, len(ids)*hidden)
	for i := range ids {
		m.w.Embed.Row(ids[i], h[i*hidden:(i+1)*hidden])
	}
	spliceImageSpans(h, spans, imageFeats, hidden)
	if cache.scr == nil { // a cache built via NewKVCache directly (tests) skips runLayers' setup
		cache.scr = newDecodeScratch(m.w.arch)
	}
	hn, err := m.runLayersQwen35N(ctx, h, cache)
	if err != nil {
		return nil, err
	}
	return m.logitsFromNormed(hn[(len(ids)-1)*hidden:], cache), nil
}

// prefillQwen35VLPerToken is the original loop, kept for the families and caches qwen35BatchNAnyPos excludes (a layer pager, a
// capture request, Olmo Hybrid) and as the reference the batched path is tested against.
func (m *Model) prefillQwen35VLPerToken(ctx context.Context, ids []int, imageFeats []float32, spans []ImageSpan, cache *KVCache) ([]float32, error) {
	hidden := m.w.arch.HiddenDim
	all := make([]float32, len(ids)*hidden)
	for i := range ids {
		m.w.Embed.Row(ids[i], all[i*hidden:(i+1)*hidden])
	}
	spliceImageSpans(all, spans, imageFeats, hidden)
	var last []float32
	for i := range ids {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		h := all[i*hidden : (i+1)*hidden]
		var err error
		if last, err = m.runLayersQwen35FromEmbed(h, cache); err != nil {
			return nil, err
		}
	}
	return m.logitsFromHidden(last, cache), nil
}

// qwen35VLPerToken is a test seam: true forces prefillLogitsQwen35VL's one-token-at-a-time loop, so a test can run both paths over
// the same prompt. Production never sets it.
var qwen35VLPerToken bool

// checkQwen35VLReady refuses an image turn on an architecture whose m-RoPE the image path cannot
// honour: not the qwen3_5 family, or a config without a usable mrope_section. Text-only loading
// never depends on it (parseMRopeFlat returns nil rather than failing); the image path does, and a
// silent fall-back to scalar RoPE would put every image token at a wrong angle without an error.
func (m *Model) checkQwen35VLReady() error {
	arch := m.w.arch
	if arch.qwen35 == nil {
		return fmt.Errorf("decoder: %q is not a Gated-DeltaNet hybrid", arch.Name)
	}
	if len(arch.MRopeSection) != 3 {
		return fmt.Errorf("decoder: %s has no mrope_section in rope_parameters; refusing an image turn rather than rotating image tokens by scalar position", arch.Name)
	}
	if want := len(arch.ropeInvFreq(0)); arch.MRopeSection[0]+arch.MRopeSection[1]+arch.MRopeSection[2] != want {
		return fmt.Errorf("decoder: %s mrope_section %v sums to %d, want %d (rotary_dim/2)", arch.Name, arch.MRopeSection, arch.MRopeSection[0]+arch.MRopeSection[1]+arch.MRopeSection[2], want)
	}
	return nil
}
