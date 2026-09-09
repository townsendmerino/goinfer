//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"os"
	"testing"

	gc "github.com/eitamring/gocudrv/cuda"
	"github.com/townsendmerino/aikit/vision"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/multimodal"
)

// TestGemma3ResidentReal_gate is gap 0's real-checkpoint gate for GenerateVL (Gemma 3) — the
// Gemma-3 twin of qwen25vl_resident_real_test.go's TestQwen25VLResidentReal_gate, closing the
// gap that file's own doc comment and docs/multimodal.md's gap-0 entry both name: "no real-image
// end-to-end gate yet" for GenerateVL specifically. Simpler than the Qwen twin — Gemma 3 has no
// m-RoPE, so plain Forward (not ForwardMRoPE) is the whole story once the CPU prefill's KV is
// uploaded.
//
// Same methodology as the Qwen gate, for the same reason (see that file's doc comment for the
// full rationale): one forced-trajectory decode step, matched precision (int4 both arms), cosine
// on raw logits rather than sampled greedy-token-stream identity — comparing SAMPLED tokens
// across a multi-step free-running rollout would conflate this design's own correctness with
// ordinary f32-vs-int4 quantization noise compounding through greedy decode, which is an
// orthogonal, pre-existing property this repo already understands (measured directly on this
// same box's Qwen2.5-VL checkpoint via a throwaway probe before the Qwen gate was rebuilt this
// way).
//
//	GOINFER_HEAVY_TESTS=1 go test -tags 'cuda goinfer_testhooks' ./cuda/ -run TestGemma3ResidentReal_gate -v -timeout 30m
func TestGemma3ResidentReal_gate(t *testing.T) {
	requireHeavyModel(t)
	home, _ := os.UserHomeDir()
	ckpt := os.Getenv("GEMMA3_4B")
	if ckpt == "" {
		ckpt = home + "/models/gemma-3-4b-it"
	}
	if _, err := os.Stat(ckpt); err != nil {
		t.Skipf("no gemma-3-4b-it at %s: %v", ckpt, err)
	}
	const golden = "../testdata/gemma3_real_golden.json.gz"
	var g struct {
		InputIDs        []int     `json:"input_ids"`
		ImageTokenStart int       `json:"image_token_start"`
		MMTokens        int       `json:"mm_tokens_per_image"`
		PixelValues     []float32 `json:"pixel_values"`
		Argmax          int       `json:"argmax"`
		LastLogits      []float32 `json:"last_logits"`
	}
	if err := decoder.ReadGoldenJSONForTest(golden, &g); err != nil {
		t.Skipf("no golden (%v) — run scripts/pin_gemma3_real.py", err)
	}

	if err := gc.Init(); err != nil {
		t.Skipf("cuInit: %v", err)
	}
	if _, err := gc.GetDevice(0); err != nil {
		t.Skipf("no device: %v", err)
	}

	enc, err := vision.LoadEncoder(ckpt, false) // f32 — matches the pin script's f32 reference
	if err != nil {
		t.Fatalf("LoadEncoder: %v", err)
	}
	visionHidden, err := enc.Forward(g.PixelValues)
	if err != nil {
		t.Fatalf("encoder Forward: %v", err)
	}
	proj, err := multimodal.LoadProjector(ckpt)
	if err != nil {
		t.Fatalf("LoadProjector: %v", err)
	}
	if got := proj.MMTokens(); got != g.MMTokens {
		t.Fatalf("projector MMTokens = %d, want %d (golden's mm_tokens_per_image)", got, g.MMTokens)
	}
	feats, err := proj.Forward(visionHidden)
	if err != nil {
		t.Fatalf("projector Forward: %v", err)
	}

	mc, err := decoder.Load(ckpt, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer mc.Close()
	rf := mc.ResidentForwardForTest()
	if rf == nil {
		t.Skip("gemma3 not resident-eligible on this build")
	}
	_, nLayers, _, _, _, _, _ := mc.Dims()

	// CPU prefill: the real image through the real bidirectional-image-block prefill path.
	cache := mc.NewCache(len(g.InputIDs) + 8)
	cpuPrefillLogits, err := mc.PrefillLogitsVLForTest(context.Background(), g.InputIDs, feats, g.ImageTokenStart, g.MMTokens, cache)
	if err != nil {
		t.Fatalf("PrefillLogitsVLForTest: %v", err)
	}
	if cos := cosine(cpuPrefillLogits, g.LastLogits); cos < 0.99 {
		t.Fatalf("CPU prefill logits vs golden: cosine %.6f < 0.99 — the reference itself doesn't hold, stop here", cos)
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

	// One decode step, same forced next token on both arms.
	nextTok := argmaxF(cpuPrefillLogits)
	gpuPos := len(g.InputIDs)

	cpuDecodeLogits, err := mc.ForwardForTest(nextTok, cache)
	if err != nil {
		t.Fatalf("CPU decode ForwardForTest: %v", err)
	}
	hybridLogits, err := rf.Forward(mc.EmbedResidentForTest(nextTok), gpuPos)
	if err != nil {
		t.Fatalf("resident Forward: %v", err)
	}

	cos := cosine(cpuDecodeLogits, hybridLogits)
	gotArg, wantArg := argmaxF(hybridLogits), argmaxF(cpuDecodeLogits)
	t.Logf("real image, one decode step past the image block: CPU argmax=%d hybrid argmax=%d | cosine=%.6f", wantArg, gotArg, cos)
	if cos < 0.99 {
		t.Errorf("decode-step cosine %.6f < 0.99 — the hybrid CPU-prefill/resident-decode step diverges from CPU on the real checkpoint", cos)
	}
	if gotArg != wantArg {
		t.Errorf("argmax = %d, want %d (CPU) — the hybrid decode picked a different greedy token on the real checkpoint", gotArg, wantArg)
	}
}
