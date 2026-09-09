//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"encoding/json"
	"os"
	"testing"

	gc "github.com/eitamring/gocudrv/cuda"
	"github.com/townsendmerino/aikit/vision"
	"github.com/townsendmerino/goinfer/decoder"
)

// TestQwen25VLResidentReal_gate is gap 0's real-checkpoint continuation of
// decoder/qwen25vl_real_test.go's TestQwen25VLReal_gate, which is prefill-only by its own doc
// comment ("decoding PAST an image block... is a genuinely different code path that no existing
// Go test exercises yet"). This is that continuation: it drives the SAME real image through the
// real vision encoder and the real CPU prefill (prefillLogitsQwenVL, via
// PrefillLogitsQwenVLForTest), then compares ONE decode step computed two ways — plain CPU
// (ForwardForTest) and the gap-0 hybrid (UploadKV then resident ForwardMRoPE) — by COSINE on the
// raw logits, both arms fed the identical next token so neither can wander off the other's
// trajectory.
//
// WHY COSINE ON A FORCED TRAJECTORY, NOT TOKEN-STREAM IDENTITY THROUGH GenerateQwenVL. An
// earlier version of this gate compared GenerateQwenVL's greedy (Temperature=0) SAMPLED token
// streams, f32 CPU vs int4 CUDA resident, over 12 tokens — and found a real divergence at step 7.
// Investigated before concluding anything: a throwaway probe ran plain decoder.Model.Generate
// (NO vision, NO gap-0 code at all, just this repo's existing, already-shipped int4 CUDA resident
// decode) on the SAME checkpoint and found divergence starting EVEN EARLIER (step 5) — proving the
// effect is pre-existing f32-vs-int4 quantization noise on Qwen2.5-VL-3B's resident decode,
// unrelated to anything built for gap 0. Once one token's argmax flips under quantization, every
// later token in a GREEDY rollout is computed from a different context than the reference, so the
// two streams necessarily diverge completely — an expected property of comparing different
// precisions through autoregressive sampling, not a defect. This gate instead does what
// gpu/nemotron_resident_parity_test.go's own doc comment calls "matched precision... isolates
// WIRING from quant quality": one controlled step, same precision, same forced trajectory, cosine
// not exact-match — the comparison that actually answers "is the hybrid decode's OWN math
// correct," independent of the orthogonal question of how quantization noise compounds through
// free-running greedy sampling.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags 'cuda goinfer_testhooks' ./cuda/ -run TestQwen25VLResidentReal_gate -v -timeout 30m
func TestQwen25VLResidentReal_gate(t *testing.T) {
	requireHeavyModel(t)
	ckpt := decoder.AssetPathForTest(t, "GOINFER_QWEN25VL_3B")
	const golden = "../testdata/qwen25vl_real_golden.json"
	raw, err := os.ReadFile(golden)
	if err != nil {
		t.Skipf("no golden (%v) — run scripts/pin_qwen25vl_real.py", err)
	}
	var g struct {
		InputIDs     []int     `json:"input_ids"`
		ImageToken   int       `json:"image_token_id"`
		ImageStart   int       `json:"image_token_start"`
		NImageTokens int       `json:"n_image_tokens"`
		GridTHW      [][3]int  `json:"grid_thw"`
		PixelValues  []float32 `json:"pixel_values"`
		Argmax       int       `json:"argmax"`
		LastLogits   []float32 `json:"last_logits"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}
	const merge = 2

	if err := gc.Init(); err != nil {
		t.Skipf("cuInit: %v", err)
	}
	if _, err := gc.GetDevice(0); err != nil {
		t.Skipf("no device: %v", err)
	}

	enc, err := vision.LoadQwenVisionEncoder(ckpt, false)
	if err != nil {
		t.Fatalf("LoadQwenVisionEncoder: %v", err)
	}
	feats, err := enc.Forward(g.PixelValues, g.GridTHW)
	if err != nil {
		t.Fatalf("encoder Forward: %v", err)
	}
	mropePos, err := decoder.MRopePositionsForTest(g.InputIDs, g.ImageToken, g.GridTHW, merge)
	if err != nil {
		t.Fatalf("MRopePositionsForTest: %v", err)
	}

	mc, err := decoder.Load(ckpt, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer mc.Close()
	rf := mc.ResidentForwardForTest()
	if rf == nil {
		t.Skip("qwen2_5_vl not resident-eligible on this build")
	}
	mrope, ok := rf.(decoder.ResidentMRoPE)
	if !ok {
		t.Fatal("cudaResident does not satisfy decoder.ResidentMRoPE")
	}
	_, nLayers, _, _, _, _, _ := mc.Dims()

	// CPU prefill: the real image through the real bidirectional-image-block prefill path.
	cache := mc.NewCache(len(g.InputIDs) + 8)
	cpuPrefillLogits, err := mc.PrefillLogitsQwenVLForTest(context.Background(), g.InputIDs, feats, g.ImageStart, g.NImageTokens, mropePos, cache)
	if err != nil {
		t.Fatalf("PrefillLogitsQwenVLForTest: %v", err)
	}
	// Cross-check against the existing golden — this also re-confirms TestQwen25VLReal_gate's own
	// claim on this same checkpoint before anything gap-0-specific is exercised.
	if cos := cosine(cpuPrefillLogits, g.LastLogits); cos < 0.99 {
		t.Fatalf("CPU prefill logits vs golden: cosine %.6f < 0.99 — the reference itself doesn't hold, stop here", cos)
	}
	mropeDelta := cache.MRopeDeltaForTest()
	t.Logf("mropeDelta = %d (nonzero confirms this prompt actually exercises the m-RoPE decode split)", mropeDelta)
	if mropeDelta == 0 {
		t.Fatal("mropeDelta == 0 — this golden's image grid doesn't compress positions, so this gate " +
			"would not actually exercise ForwardMRoPE's ropePos != pos path at all")
	}

	// Upload the CPU prefill's KV into the resident cache — the gap-0 bridge under test.
	for l := range nLayers {
		k, v, base := cache.LayerKVForTest(l)
		if len(k) == 0 {
			t.Fatalf("layer %d: LayerKVForTest returned empty K", l)
		}
		if err := rf.UploadKV(l, base, k, v); err != nil {
			t.Fatalf("UploadKV layer %d: %v", l, err)
		}
	}

	// One decode step, same forced next token on both arms (argmax of the CPU prefill logits —
	// deterministic, and it's the real greedy choice a Temperature=0 turn would make here).
	nextTok := argmaxF(cpuPrefillLogits)
	gpuPos := len(g.InputIDs)

	cpuDecodeLogits, err := mc.ForwardForTest(nextTok, cache)
	if err != nil {
		t.Fatalf("CPU decode ForwardForTest: %v", err)
	}
	hybridLogits, err := mrope.ForwardMRoPE(mc.EmbedResidentForTest(nextTok), gpuPos, gpuPos+mropeDelta)
	if err != nil {
		t.Fatalf("resident ForwardMRoPE: %v", err)
	}

	cos := cosine(cpuDecodeLogits, hybridLogits)
	gotArg, wantArg := argmaxF(hybridLogits), argmaxF(cpuDecodeLogits)
	t.Logf("real image, one decode step past the image block: CPU argmax=%d hybrid argmax=%d | cosine=%.6f", wantArg, gotArg, cos)
	if cos < 0.99 { // same floor TestQwen25VLReal_gate uses for this checkpoint's vision path
		t.Errorf("decode-step cosine %.6f < 0.99 — the hybrid CPU-prefill/resident-decode step diverges from CPU on the real checkpoint", cos)
	}
	if gotArg != wantArg {
		t.Errorf("argmax = %d, want %d (CPU) — the hybrid decode picked a different greedy token on the real checkpoint", gotArg, wantArg)
	}
}
