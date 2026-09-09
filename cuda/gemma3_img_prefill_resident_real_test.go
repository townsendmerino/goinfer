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

// TestGemma3ImgPrefillResidentReal_gate is the real-checkpoint gate for the resident image-prefill
// fast path (finishing gap 0, decoder.ResidentImagePrefill) — the direct sibling of
// gemma3_resident_real_test.go's TestGemma3ResidentReal_gate, same methodology, but exercising a
// COLD (first-time) image turn's PREFILL on the resident GPU instead of the CPU-prefill-then-
// UploadKV bridge that gate covers.
//
// Same reasoning as that gate for matched precision (int4 both arms — a CPU-f32-vs-GPU-int4
// comparison would conflate this design's own correctness with ordinary quantization noise) and
// forced-trajectory cosine on raw logits rather than sampled greedy-token-stream identity.
//
// Two halves:
//
//   - PRIMITIVE-LEVEL (via ResidentImagePrefillForTest, the exact call GenerateVL's fast path
//     makes internally): compares the resident image-prefill's last-token logits directly against
//     PrefillLogitsVLForTest's CPU reference on the SAME real model/real image — this is the part
//     that proves the real weights/embeddings/rope tables flow through the new kernel correctly,
//     which the synthetic kernel-level tests (attn_img_batched_test.go) cannot see (they use
//     synthetic Q/K/V, not this checkpoint's real values).
//
//   - INTEGRATION-LEVEL (via the real GenerateVL entrypoint): confirms the wiring — claim, type
//     assertion, decode continuation, Generation.ImgPrefillResident — actually engages end to end,
//     not just that the underlying primitive is correct in isolation.
//
//     GOINFER_HEAVY_TESTS=1 go test -tags 'cuda goinfer_testhooks' ./cuda/ -run TestGemma3ImgPrefillResidentReal_gate -v -timeout 30m
func TestGemma3ImgPrefillResidentReal_gate(t *testing.T) {
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

	enc, err := vision.LoadEncoder(ckpt, false) // f32 — matches the pin script's own reference
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
	rip, ok := rf.(decoder.ResidentImagePrefill)
	if !ok {
		t.Skip("attn_img_batched did not load on this build — ResidentImagePrefill unavailable")
	}

	// --- Half 1: primitive-level logits comparison ---

	cpuCache := mc.NewCache(len(g.InputIDs) + 8)
	cpuLogits, err := mc.PrefillLogitsVLForTest(context.Background(), g.InputIDs, feats, g.ImageTokenStart, g.MMTokens, cpuCache)
	if err != nil {
		t.Fatalf("PrefillLogitsVLForTest: %v", err)
	}
	if cos := cosine(cpuLogits, g.LastLogits); cos < 0.99 {
		t.Fatalf("CPU prefill logits vs golden: cosine %.6f < 0.99 — the reference itself doesn't hold, stop here", cos)
	}

	residentLogits, gpuPos, err := mc.ResidentImagePrefillForTest(context.Background(), rip, g.InputIDs, feats, g.ImageTokenStart, g.MMTokens)
	if err != nil {
		t.Fatalf("ResidentImagePrefillForTest: %v", err)
	}
	if gpuPos != len(g.InputIDs) {
		t.Errorf("gpuPos = %d, want %d (len(InputIDs))", gpuPos, len(g.InputIDs))
	}
	cos := cosine(cpuLogits, residentLogits)
	gotArg, wantArg := argmaxF(residentLogits), argmaxF(cpuLogits)
	t.Logf("real image, resident image-prefill vs CPU prefill: CPU argmax=%d resident argmax=%d | cosine=%.6f", wantArg, gotArg, cos)
	if cos < 0.99 {
		t.Errorf("prefill-logits cosine %.6f < 0.99 — the resident image-prefill diverges from CPU on the real checkpoint", cos)
	}
	if gotArg != wantArg {
		t.Errorf("argmax = %d, want %d (CPU) — the resident image-prefill picked a different greedy token on the real checkpoint", gotArg, wantArg)
	}

	// --- Half 2: integration-level, through the real GenerateVL entrypoint ---
	// Reuses mc rather than loading a second resident instance: two resident int4
	// gemma-3-4b-it instances (weights + a 4096-position KV cache each) do not fit together on
	// this 8GB card (measured directly building this session's P9(a) timing driver). Safe to
	// reuse — ResidentImagePrefillForTest above left mc.resIDs untouched (nil; that bookkeeping
	// is P9(a)'s, not this primitive's), and GenerateVL's ordinary path is ALWAYS a full,
	// unconditional prefill overwrite regardless of whatever the resident cache held before.
	features := func() ([]float32, error) { return feats, nil }
	const maxNew = 4
	stream, gen := mc.GenerateVL(context.Background(), g.InputIDs, g.ImageTokenStart, g.MMTokens, 42, features, maxNew, decoder.SamplingParams{Temperature: 0})
	var got []int
	for id := range stream {
		got = append(got, id)
	}
	if err := gen.Err(); err != nil {
		t.Fatalf("GenerateVL: %v", err)
	}
	if len(got) == 0 {
		t.Fatal("GenerateVL streamed no tokens")
	}
	if !gen.ImgPrefillResident {
		t.Error("Generation.ImgPrefillResident = false — GenerateVL fell through to the CPU-prefill+UploadKV " +
			"bridge instead of taking the resident image-prefill fast path this gate exists to prove")
	}
	if got[0] != wantArg {
		t.Errorf("GenerateVL's first greedy token = %d, want %d (the CPU prefill's own argmax, matching "+
			"the primitive-level comparison above)", got[0], wantArg)
	}
	t.Logf("GenerateVL streamed %d tokens via the resident image-prefill fast path (ImgPrefillResident=true), first=%d", len(got), got[0])
}
