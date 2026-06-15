package decoder

import (
	"encoding/json"
	"errors"
	"io/fs"
	"math"
	"os"
	"testing"

	"github.com/townsendmerino/aikit/vision"
)

// TestQwen25VL_e2eChain runs the REAL goinfer pipeline end-to-end: the aikit ViT
// encoder (Forward on the golden pixel_values) → the decoder image prefill (inject
// + m-RoPE) → logits, matched against HF. Unlike TestQwen25VL_imageParity (which
// feeds the GOLDEN image_features to isolate the decoder), this chains the two
// stages goinfer actually wires, so it catches an encoder↔decoder seam mismatch.
func TestQwen25VL_e2eChain(t *testing.T) {
	const golden = "../testdata/qwen25vl_tiny_image_golden.json"
	const ckpt = "../testdata/qwen25vl-tiny"
	raw, err := os.ReadFile(golden)
	if errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no golden — run scripts/pin_qwen25vl_image.py")
	}
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	if _, err := os.Stat(ckpt); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no checkpoint — run scripts/pin_qwen25vl_tiny.py")
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

	enc, err := vision.LoadQwenVisionEncoder(ckpt, false)
	if err != nil {
		t.Fatalf("LoadQwenVisionEncoder: %v", err)
	}
	feats, err := enc.Forward(g.PixelValues, g.GridTHW) // ViT + merger → merged features
	if err != nil {
		t.Fatalf("encoder Forward: %v", err)
	}

	m, err := Load(ckpt, Options{})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer m.Close()
	mropePos, err := mropePositions(g.InputIDs, g.ImageToken, g.GridTHW, merge)
	if err != nil {
		t.Fatalf("mropePositions: %v", err)
	}
	cache := m.NewCache(len(g.InputIDs))
	logits, err := m.prefillLogitsQwenVL(g.InputIDs, feats, g.ImageStart, g.NImageTokens, mropePos, cache)
	if err != nil {
		t.Fatalf("prefillLogitsQwenVL: %v", err)
	}
	gotArg := argmax(logits)
	cos := logitCosine(logits, g.LastLogits)
	t.Logf("qwen2.5-VL e2e (encoder→decoder): argmax got=%d want=%d | cosine=%.6f", gotArg, g.Argmax, cos)
	if gotArg != g.Argmax {
		t.Errorf("e2e argmax = %d, want %d", gotArg, g.Argmax)
	}
	if cos < 0.99 {
		t.Errorf("e2e last-logit cosine %.6f < 0.99", cos)
	}
}

// TestMRoPEPositions gates the get_rope_index port against the pinned HF golden
// (scripts/pin_qwen25vl_image.py): the 3D (t,h,w) positions for a text+image
// sequence must match HF token-for-token — text scalar, image block on the merged
// grid, scalar resuming from the block max + 1.
func TestMRoPEPositions(t *testing.T) {
	const golden = "../testdata/qwen25vl_tiny_image_golden.json"
	raw, err := os.ReadFile(golden)
	if errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no golden — run scripts/pin_qwen25vl_image.py")
	}
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	var g struct {
		InputIDs    []int    `json:"input_ids"`
		ImageToken  int      `json:"image_token_id"`
		GridTHW     [][3]int `json:"grid_thw"`
		PositionIDs [][]int  `json:"position_ids"` // [3][seq] = t,h,w rows
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}
	const merge = 2 // testdata/qwen25vl-tiny spatial_merge_size

	got, err := mropePositions(g.InputIDs, g.ImageToken, g.GridTHW, merge)
	if err != nil {
		t.Fatalf("mropePositions: %v", err)
	}
	if len(got) != len(g.InputIDs) {
		t.Fatalf("got %d positions, want %d", len(got), len(g.InputIDs))
	}
	for i := range got {
		for c := range 3 {
			if got[i][c] != g.PositionIDs[c][i] {
				t.Errorf("pos[%d][%d] = %d, want %d (got row-major %v)", i, c, got[i][c], g.PositionIDs[c][i], got[i])
			}
		}
	}
}

// TestApplyMRoPE_scalarEquivalence is the text-path-stays-inert gate at the kernel
// level: for a TEXT token (all three position components equal), applyMRoPE must
// produce EXACTLY what applyRoPE does for that scalar position — so text decode is
// bit-identical whether or not the m-RoPE path is taken.
func TestApplyMRoPE_scalarEquivalence(t *testing.T) {
	const heads, headDim = 2, 16
	half := headDim / 2
	section := []int{4, 2, 2} // sums to half=8
	invFreq := make([]float64, half)
	for d := range invFreq {
		invFreq[d] = math.Pow(10000, -2*float64(d)/float64(headDim))
	}
	base := make([]float32, heads*headDim)
	for i := range base {
		base[i] = float32((i*7)%13)*0.1 - 0.6
	}
	for _, pos := range []int{0, 1, 5, 42, 1000} {
		a := append([]float32(nil), base...)
		b := append([]float32(nil), base...)
		applyRoPE(a, heads, headDim, pos, invFreq, 1.0)
		applyMRoPE(b, heads, headDim, [3]int{pos, pos, pos}, section, invFreq, 1.0)
		for i := range a {
			if a[i] != b[i] {
				t.Fatalf("pos %d: applyMRoPE[%d]=%v != applyRoPE %v", pos, i, b[i], a[i])
			}
		}
	}
	// A divergent (image) position must NOT equal the scalar result (non-vacuous).
	a := append([]float32(nil), base...)
	b := append([]float32(nil), base...)
	applyRoPE(a, heads, headDim, 4, invFreq, 1.0)
	applyMRoPE(b, heads, headDim, [3]int{4, 5, 6}, section, invFreq, 1.0)
	same := true
	for i := range a {
		if a[i] != b[i] {
			same = false
			break
		}
	}
	if same {
		t.Error("divergent t/h/w positions produced the scalar result — m-RoPE has no effect")
	}
}

// TestQwen25VL_imageParity is the P5 end-to-end decoder gate: inject the (HF
// golden) merged image features at the <image> run + thread m-RoPE 3D positions
// through the prefill, and match HF's last-position logits. Decoupled from the ViT
// (uses the pinned image_features, like the Gemma 3 image gate) so this isolates
// the decoder-side image path — feature injection, causal (not bidirectional)
// attention over the image tokens, and m-RoPE.
func TestQwen25VL_imageParity(t *testing.T) {
	const golden = "../testdata/qwen25vl_tiny_image_golden.json"
	const ckpt = "../testdata/qwen25vl-tiny"
	raw, err := os.ReadFile(golden)
	if errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no golden — run scripts/pin_qwen25vl_image.py")
	}
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	if _, err := os.Stat(ckpt); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no checkpoint — run scripts/pin_qwen25vl_tiny.py")
	}
	var g struct {
		InputIDs      []int     `json:"input_ids"`
		ImageToken    int       `json:"image_token_id"`
		ImageStart    int       `json:"image_token_start"`
		NImageTokens  int       `json:"n_image_tokens"`
		GridTHW       [][3]int  `json:"grid_thw"`
		ImageFeatures []float32 `json:"image_features"`
		Argmax        int       `json:"argmax"`
		LastLogits    []float32 `json:"last_logits"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}
	const merge = 2 // testdata/qwen25vl-tiny spatial_merge_size

	m, err := Load(ckpt, Options{})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer m.Close()

	mropePos, err := mropePositions(g.InputIDs, g.ImageToken, g.GridTHW, merge)
	if err != nil {
		t.Fatalf("mropePositions: %v", err)
	}
	cache := m.NewCache(len(g.InputIDs))
	logits, err := m.prefillLogitsQwenVL(g.InputIDs, g.ImageFeatures, g.ImageStart, g.NImageTokens, mropePos, cache)
	if err != nil {
		t.Fatalf("prefillLogitsQwenVL: %v", err)
	}
	gotArg := argmax(logits)
	cos := logitCosine(logits, g.LastLogits)
	t.Logf("qwen2.5-VL image parity: argmax got=%d want=%d | logit cosine=%.6f", gotArg, g.Argmax, cos)
	if gotArg != g.Argmax {
		t.Errorf("image-path argmax = %d, want %d", gotArg, g.Argmax)
	}
	if cos < 0.99 {
		t.Errorf("image-path last-logit cosine %.6f < 0.99", cos)
	}
}

// TestQwen25VL_textParity loads the tiny Qwen2.5-VL checkpoint (scripts/
// pin_qwen25vl_tiny.py) through goinfer's loader + forward and asserts the
// TEXT-ONLY path matches the HF golden — the P5 invariant: a Qwen2.5-VL
// checkpoint's text decoder (vision tower / merger ignored) loads and runs exactly
// like a Qwen2, since m-RoPE degenerates to scalar RoPE when the 3 position
// components are equal (all text tokens). This is the prerequisite before the
// image seam (m-RoPE 3D positions + the ViT) in later increments.
func TestQwen25VL_textParity(t *testing.T) {
	const golden = "../testdata/qwen25vl_tiny_text_golden.json"
	const ckpt = "../testdata/qwen25vl-tiny"
	raw, err := os.ReadFile(golden)
	if errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no golden — run scripts/pin_qwen25vl_tiny.py")
	}
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	if _, err := os.Stat(ckpt); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no checkpoint at %s — run scripts/pin_qwen25vl_tiny.py", ckpt)
	}
	var g struct {
		PromptIDs       []int     `json:"prompt_ids"`
		Argmax          int       `json:"argmax"`
		LastLogits      []float32 `json:"last_logits"`
		NNew            int       `json:"n_new"`
		ContinuationIDs []int     `json:"continuation_ids"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}

	m, err := Load(ckpt, Options{})
	if err != nil {
		t.Fatalf("Load(%s): %v", ckpt, err)
	}
	defer m.Close()

	cache := m.NewCache(len(g.PromptIDs) + g.NNew)
	var logits []float32
	for _, id := range g.PromptIDs {
		if logits, err = m.forward(id, cache); err != nil {
			t.Fatalf("forward: %v", err)
		}
	}
	gotArg := argmax(logits)
	cos := logitCosine(logits, g.LastLogits)
	t.Logf("qwen2.5-VL text parity: argmax got=%d want=%d | logit cosine=%.6f", gotArg, g.Argmax, cos)
	if gotArg != g.Argmax {
		t.Errorf("last argmax = %d, want %d", gotArg, g.Argmax)
	}
	if cos < 0.9999 {
		t.Errorf("last-logit cosine %.6f < 0.9999", cos)
	}

	// Greedy continuation must match the HF golden token-for-token.
	got := make([]int, 0, g.NNew)
	for range g.NNew {
		id := argmax(logits)
		got = append(got, id)
		if logits, err = m.forward(id, cache); err != nil {
			t.Fatalf("continuation forward: %v", err)
		}
	}
	for i := range g.ContinuationIDs {
		if got[i] != g.ContinuationIDs[i] {
			t.Errorf("continuation[%d] = %d, want %d (got %v want %v)", i, got[i], g.ContinuationIDs[i], got, g.ContinuationIDs)
			break
		}
	}
}
