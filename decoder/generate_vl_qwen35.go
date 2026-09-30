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
func (m *Model) prefillLogitsQwen35VL(ctx context.Context, ids []int, imageFeats []float32, imgPos, imgLen int, mropePos [][3]int, cache *KVCache) ([]float32, error) {
	arch := m.w.arch
	hidden := arch.HiddenDim
	if imgPos < 0 || imgLen <= 0 || imgPos+imgLen > len(ids) {
		return nil, fmt.Errorf("decoder: image run [%d,%d) out of range for %d tokens", imgPos, imgPos+imgLen, len(ids))
	}
	if len(imageFeats) != imgLen*hidden {
		return nil, fmt.Errorf("decoder: imageFeats len %d, want %d (%d tokens × %d)", len(imageFeats), imgLen*hidden, imgLen, hidden)
	}
	if len(mropePos) != len(ids) {
		return nil, fmt.Errorf("decoder: mropePos len %d, want %d (one per token)", len(mropePos), len(ids))
	}
	if err := m.checkQwen35VLReady(); err != nil {
		return nil, err
	}
	cache.mropePos = mropePos
	cache.mropeDelta = mropeDelta(mropePos, len(ids))
	var last []float32
	for i := range ids {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		h := make([]float32, hidden)
		if i >= imgPos && i < imgPos+imgLen {
			copy(h, imageFeats[(i-imgPos)*hidden:(i-imgPos+1)*hidden])
		} else {
			m.w.Embed.Row(ids[i], h)
		}
		var err error
		if last, err = m.runLayersQwen35FromEmbed(h, cache); err != nil {
			return nil, err
		}
	}
	return m.logitsFromHidden(last, cache), nil
}

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
