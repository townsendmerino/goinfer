package decoder

import (
	"path/filepath"
	"testing"
)

// TestEstimateSafetensors_gptqPackFactor gates M-29's GPTQ/AWQ sub-fix on a synthetic checkpoint
// (no real GPTQ download needed — the estimator only reads tensor NAMES and SHAPES, never
// dtype-specific values, so a fake .qweight tensor with the real packed SHAPE exercises the same
// code path a real one would). qweight's on-disk shape is [in/8, out] (decoder/gptq.go: "8 4-bit
// codes pack into each int32... qweight packs the input dim"); the true logical matrix it
// represents once reconstructed and re-quantized is [in, out] — 8x more elements than its raw
// shape reports. Also confirms qzeros/g_idx/scales are excluded entirely (M-29's second half),
// not double-counted as if they persisted resident alongside the corrected qweight price.
func TestEstimateSafetensors_gptqPackFactor(t *testing.T) {
	const in, out = 4096, 4096 // must be a multiple of 8 for the packed dims below
	dir := t.TempDir()
	fill := func(n int) []float32 {
		f := make([]float32, n)
		for i := range f {
			f[i] = float32(i%7) - 3
		}
		return f
	}
	writeSafetensors(t, filepath.Join(dir, "model.safetensors"), map[string]stTensor{
		"model.layers.0.self_attn.q_proj.qweight": {[]int{in / 8, out}, fill((in / 8) * out)},
		"model.layers.0.self_attn.q_proj.qzeros":  {[]int{32, out / 8}, fill(32 * (out / 8))},
		"model.layers.0.self_attn.q_proj.scales":  {[]int{32, out}, fill(32 * out)},
		"model.layers.0.self_attn.q_proj.g_idx":   {[]int{in}, fill(in)},
	})

	const q = quantInt4
	got := estimateSafetensorsWeightBytes(dir, q)

	// Expected: ONLY the (pack-corrected) qweight contributes, priced at quantInt4's bytes/elem —
	// qzeros/g_idx/scales must contribute ZERO, not their own (wrongly-priced) element counts.
	want := int64(quantBytesPerElem(q) * float64(in) * float64(out))
	if got != want {
		t.Errorf("estimateSafetensorsWeightBytes = %d, want %d (in*out=%d elements at quantInt4's "+
			"rate — pack-factor-corrected qweight only, qzeros/g_idx/scales excluded)", got, want, in*out)
	}
}

// TestEstimateSafetensors_visionTowerPricedAtF32 gates M-29's vision-tower sub-fix: a bundled
// multimodal checkpoint's vision_tower.*/multi_modal_projector.* tensors load at their OWN
// precision (aikit/vision.LoadEncoder's independent -vision-quant knob, default f32) rather than
// the text model's requested quant — decoder/weights.go's text loader already never requests
// these names, but this estimator walks every tensor in the checkpoint's metadata and must
// recognize them explicitly or they get swept into the text quant's rate.
func TestEstimateSafetensors_visionTowerPricedAtF32(t *testing.T) {
	const dim = 1024
	dir := t.TempDir()
	fill := func(n int) []float32 {
		f := make([]float32, n)
		for i := range f {
			f[i] = float32(i%7) - 3
		}
		return f
	}
	writeSafetensors(t, filepath.Join(dir, "model.safetensors"), map[string]stTensor{
		"model.layers.0.self_attn.q_proj.weight": {[]int{dim, dim}, fill(dim * dim)},
		"vision_tower.blocks.0.attn.qkv.weight":  {[]int{dim, dim}, fill(dim * dim)},
		"multi_modal_projector.linear.weight":    {[]int{dim, dim}, fill(dim * dim)},
	})

	const q = quantInt4
	got := estimateSafetensorsWeightBytes(dir, q)
	want := int64(quantBytesPerElem(q)*float64(dim*dim)) + 2*int64(4*float64(dim*dim)) // text at quant, both vision tensors at f32
	if got != want {
		t.Errorf("estimateSafetensorsWeightBytes = %d, want %d (text tensor at quantInt4's rate, "+
			"vision_tower/multi_modal_projector tensors at f32 regardless of the text quant)", got, want)
	}
}
