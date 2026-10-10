//go:build realckpt

// Real-model gate for Qwen2.5-VL (Qwen/Qwen2.5-VL-3B-Instruct, model_type "qwen2_5_vl"): the T3 promotion of the qwen2_5_vl
// family from tiny-golden to a released checkpoint.
//
// The oracle runs a REAL image: TestQwen25VL_e2eChain is already encoder→decoder, but on SYNTHETIC pixel_values with no real
// processor in the loop. This gate runs the real AutoImageProcessor on a real image (testdata/qwen25vl_preprocess_image.png,
// pre-sized so smart_resize is a no-op, isolating decoder-on-real-weights from resize/bicubic parity, which
// pin_qwen25vl_preprocess.py pins separately) through the real vision encoder and the real text decoder. Fixture:
// scripts/pin_qwen25vl_real.py.
//
// SCOPED TO THE PREFILL FORWARD ONLY, NOT GREEDY CONTINUATION. Decoding PAST an image block (m-RoPE position continuation
// from the image grid's max position) is a different code path from plain-text m.forward(id, cache); building it here would
// risk conflating a new harness's correctness with the checkpoint's. It is gated on the tiny fixture by
// TestQwen25VL_imageParity and TestQwen25VL_generate, not on real weights.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags realckpt ./decoder/ -run TestQwen25VLReal -v -timeout 30m
package decoder

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/townsendmerino/aikit/vision"
)

func TestQwen25VLReal_gate(t *testing.T) {
	requireHeavyModel(t)
	ckpt := assetPath(t, "GOINFER_QWEN25VL_3B")
	const golden = "../testdata/qwen25vl_real_golden.json.gz"
	raw, err := readGolden(golden)
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
	const merge = 2 // vision_config.spatial_merge_size, Qwen2.5-VL-3B-Instruct

	enc, err := vision.LoadQwenVisionEncoder(ckpt, false) // f32 → tight cosine
	if err != nil {
		t.Fatalf("LoadQwenVisionEncoder: %v", err)
	}
	feats, err := enc.Forward(g.PixelValues, g.GridTHW)
	if err != nil {
		t.Fatalf("encoder Forward: %v", err)
	}

	m, err := Load(ckpt, Options{}) // f32 → tight cosine
	if err != nil {
		t.Fatalf("Load(%s): %v", ckpt, err)
	}
	defer m.Close()
	if m.w.arch.Name != "qwen2_5_vl" {
		t.Fatalf("arch = %q, want qwen2_5_vl", m.w.arch.Name)
	}

	mropePos, err := mropePositions(g.InputIDs, g.ImageToken, g.GridTHW, merge)
	if err != nil {
		t.Fatalf("mropePositions: %v", err)
	}
	cache := m.NewCache(len(g.InputIDs))
	logits, err := m.prefillLogitsQwenVL(context.Background(), g.InputIDs, feats, g.ImageStart, g.NImageTokens, mropePos, cache)
	if err != nil {
		t.Fatalf("prefillLogitsQwenVL: %v", err)
	}
	gotArg := argmax(logits)
	cos := logitCosine(logits, g.LastLogits)
	t.Logf("qwen2.5-VL-3B real image: argmax got=%d want=%d | logit cosine=%.6f", gotArg, g.Argmax, cos)
	if gotArg != g.Argmax {
		t.Errorf("argmax = %d, want %d", gotArg, g.Argmax)
	}
	if cos < 0.99 { // same floor TestQwen25VL_e2eChain uses — the vision path is looser than text-only
		t.Errorf("last-logit cosine %.6f < 0.99", cos)
	}
	// Real-weight vision encoder + real-weight decoder, on a real (if pre-sized) image: no
	// greedy continuation checked here (see file doc comment), so this is a single-forward
	// numeric oracle, not a full-generation one — argmax_pct is 100 because that single
	// forward's argmax matched when the gate passed, not because a continuation was run.
	emitParityRow(t, "qwen2_5_vl", "full-forward-oracle", "HF f32 (Qwen/Qwen2.5-VL-3B-Instruct, real image, prefill forward only)", 100.0, float64(cos), float64(cos))
}
