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

// TestQwen25VLMRoPEPrefillResidentReal_gate is the real-checkpoint gate for the resident m-RoPE
// prefill fast path (decoder.ResidentMRoPEPrefill) — the direct Qwen2.5-VL sibling of
// gemma3_img_prefill_resident_real_test.go's TestGemma3ImgPrefillResidentReal_gate, same
// methodology and same two-half structure, but exercising a COLD (first-time) image turn's
// PREFILL through the m-RoPE batched-rotation kernel instead of Gemma-3's bidirectional-attention
// one.
//
// Reuses the EXISTING real golden (testdata/qwen25vl_real_golden.json) — no new pin script. That
// golden's own image grid already compresses post-image text positions (confirmed by
// TestQwen25VLResidentReal_gate's own mropeDelta!=0 assertion on this same fixture), so this gate
// exercises the genuinely-divergent-rotation path, not just the degenerate scalar-equivalent one.
//
// Matched precision (int4 both arms) and forced-trajectory cosine on raw logits, same reasoning as
// every other resident-vs-CPU gate in this package.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags 'cuda goinfer_testhooks' ./cuda/ -run TestQwen25VLMRoPEPrefillResidentReal_gate -v -timeout 30m
func TestQwen25VLMRoPEPrefillResidentReal_gate(t *testing.T) {
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
	rmp, ok := rf.(decoder.ResidentMRoPEPrefill)
	if !ok {
		t.Skip("rope_kv_mrope_batched did not load on this build — ResidentMRoPEPrefill unavailable")
	}

	// --- Half 1: primitive-level logits comparison ---

	cpuCache := mc.NewCache(len(g.InputIDs) + 8)
	cpuLogits, err := mc.PrefillLogitsQwenVLForTest(context.Background(), g.InputIDs, feats, g.ImageStart, g.NImageTokens, mropePos, cpuCache)
	if err != nil {
		t.Fatalf("PrefillLogitsQwenVLForTest: %v", err)
	}
	if cos := cosine(cpuLogits, g.LastLogits); cos < 0.99 {
		t.Fatalf("CPU prefill logits vs golden: cosine %.6f < 0.99 — the reference itself doesn't hold, stop here", cos)
	}

	residentLogits, gpuPos, err := mc.ResidentMRoPEPrefillForTest(context.Background(), rmp, g.InputIDs, feats, g.ImageStart, g.NImageTokens, mropePos)
	if err != nil {
		t.Fatalf("ResidentMRoPEPrefillForTest: %v", err)
	}
	if gpuPos != len(g.InputIDs) {
		t.Errorf("gpuPos = %d, want %d (len(InputIDs))", gpuPos, len(g.InputIDs))
	}
	cos := cosine(cpuLogits, residentLogits)
	gotArg, wantArg := argmaxF(residentLogits), argmaxF(cpuLogits)
	t.Logf("real image, resident m-RoPE prefill vs CPU prefill: CPU argmax=%d resident argmax=%d | cosine=%.6f", wantArg, gotArg, cos)
	if cos < 0.99 {
		t.Errorf("prefill-logits cosine %.6f < 0.99 — the resident m-RoPE prefill diverges from CPU on the real checkpoint", cos)
	}
	if gotArg != wantArg {
		t.Errorf("argmax = %d, want %d (CPU) — the resident m-RoPE prefill picked a different greedy token on the real checkpoint", gotArg, wantArg)
	}

	// --- Half 2: integration-level, through the real GenerateQwenVL entrypoint ---
	// Reuses mc rather than loading a second resident instance — same reasoning as
	// TestGemma3ImgPrefillResidentReal_gate: two resident int4 instances don't fit together on this
	// 8GB card, and GenerateQwenVL's ordinary path is always a full, unconditional prefill overwrite
	// regardless of whatever the resident cache held before.
	features := func() ([]float32, error) { return feats, nil }
	const maxNew = 4
	stream, gen := mc.GenerateQwenVL(context.Background(), g.InputIDs, g.ImageStart, g.NImageTokens, 42, features, g.GridTHW, merge, g.ImageToken, maxNew, decoder.SamplingParams{Temperature: 0})
	var got []int
	for id := range stream {
		got = append(got, id)
	}
	if err := gen.Err(); err != nil {
		t.Fatalf("GenerateQwenVL: %v", err)
	}
	if len(got) == 0 {
		t.Fatal("GenerateQwenVL streamed no tokens")
	}
	if !gen.ImgPrefillResident {
		t.Error("Generation.ImgPrefillResident = false — GenerateQwenVL fell through to the CPU-prefill+UploadKV " +
			"bridge instead of taking the resident m-RoPE prefill fast path this gate exists to prove")
	}
	if got[0] != wantArg {
		t.Errorf("GenerateQwenVL's first greedy token = %d, want %d (the CPU prefill's own argmax, matching "+
			"the primitive-level comparison above)", got[0], wantArg)
	}
	t.Logf("GenerateQwenVL streamed %d tokens via the resident m-RoPE prefill fast path (ImgPrefillResident=true), first=%d", len(got), got[0])
}
