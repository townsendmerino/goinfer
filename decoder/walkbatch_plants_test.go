package decoder

import (
	"context"
	"encoding/json"
	"errors"
	"io/fs"
	"math"
	"os"
	"testing"
)

// withBatchPlant runs f with plant p set, and clears it after.
func withBatchPlant(p walkBatchPlant, f func()) {
	walkBatchPlantForTest = p
	defer func() { walkBatchPlantForTest = plantBatchNone }()
	f()
}

// TestWalkBatchPlant_qwen35GateSkipped: skipping the gated attention's sigmoid gate on one layer must push PromptHidden (the
// batched walker) past the bound TestPromptHidden_batchedMatchesSequential holds it to against the per-token forward.
func TestWalkBatchPlant_qwen35GateSkipped(t *testing.T) {
	m, err := Load("testdata/qwen3_5-tiny-normw", Options{})
	if err != nil {
		t.Skipf("no qwen3_5-tiny-normw fixture: %v", err)
	}
	defer m.Close()
	const n, bar = 32, 1e-6
	ids := make([]int, n)
	for i := range ids {
		ids[i] = (i*37 + 11) % m.w.arch.VocabSize
	}
	want, err := m.hiddenLastSequential(ids)
	if err != nil {
		t.Fatal(err)
	}
	rel := func() float64 {
		got, err := m.PromptHidden(context.Background(), ids)
		if err != nil {
			t.Fatal(err)
		}
		return relL2(got, want)
	}
	clean := rel()
	var planted float64
	withBatchPlant(plantBatchNoAttnGate, func() { planted = rel() })
	t.Logf("relative L2 against the per-token forward: clean %.3g, gate skipped %.3g (bar %.0e)", clean, planted, bar)
	if clean > bar {
		t.Fatalf("clean walker already past the bar: %.3g", clean)
	}
	if planted <= bar {
		t.Errorf("BLIND: the skipped gate stays within the bar (%.3g)", planted)
	}
}

// TestWalkBatchPlant_qwen35ConvWindowDropped: dropping the conv window at a pass's start must break the split-prefill identity
// (TestQwen35SplitPrefill_bitIdentical): two passes [0,b) then [b,n) leave what one pass over [0,n) leaves.
func TestWalkBatchPlant_qwen35ConvWindowDropped(t *testing.T) {
	m, err := Load("../testdata/qwen35-tiny", Options{})
	if err != nil {
		t.Skipf("no qwen35-tiny fixture: %v", err)
	}
	defer m.Close()
	const n, b = 40, 17
	ids := make([]int, n)
	for i := range ids {
		ids[i] = (i*37 + 5) % m.w.arch.VocabSize
	}
	hidden := m.w.arch.HiddenDim
	last := func(cuts ...int) []float32 {
		cache := m.NewCache(n)
		var out []float32
		from := 0
		for _, to := range append(cuts, n) {
			hN, err := m.runLayersQwen35N(context.Background(), m.embedN(ids[from:to]), cache)
			if err != nil {
				t.Fatal(err)
			}
			out = append([]float32(nil), hN[(to-from-1)*hidden:]...)
			from = to
		}
		return out
	}
	one := last()
	same := func(a, c []float32) bool {
		for i := range a {
			if math.Float32bits(a[i]) != math.Float32bits(c[i]) {
				return false
			}
		}
		return true
	}
	if !same(one, last(b)) {
		t.Fatal("clean walker: the split prefill already differs from one pass")
	}
	var split []float32
	withBatchPlant(plantBatchDropConvWindow, func() { split = last(b) })
	if same(one, split) {
		t.Error("BLIND: the dropped conv window leaves the split prefill identical to one pass")
	}
}

// TestWalkBatchPlant_gemma4BidirectionalDropped: ending an image row's key range at its own position (a causal block) must fail
// the bidirectional image golden that TestGemma4VLBidir_imageParity holds the prefill to (argmax and cosine 0.999).
func TestWalkBatchPlant_gemma4BidirectionalDropped(t *testing.T) {
	const golden = "../testdata/gemma4_vl_bidir_tiny_image_golden.json"
	const ckpt = "../testdata/gemma4-vl-bidir-tiny"
	raw, err := os.ReadFile(golden)
	if errors.Is(err, fs.ErrNotExist) {
		t.Skip("no image golden")
	}
	if err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(ckpt + "/model.safetensors"); errors.Is(err, fs.ErrNotExist) {
		t.Skip("no checkpoint")
	}
	var g struct {
		InputIDs        []int     `json:"input_ids"`
		ImageTokenStart int       `json:"image_token_start"`
		NImageTokens    int       `json:"n_image_tokens"`
		ImageFeatures   []float32 `json:"image_features"`
		Argmax          int       `json:"argmax"`
		LastLogits      []float32 `json:"last_logits"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	m, err := Load(ckpt, Options{})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	run := func() (int, float64) {
		lg, err := m.prefillLogitsGemma4VLBidirectional(context.Background(), g.InputIDs, g.ImageFeatures, g.ImageTokenStart, g.NImageTokens, m.NewCache(len(g.InputIDs)))
		if err != nil {
			t.Fatal(err)
		}
		return argmax(lg), logitCosine(lg, g.LastLogits)
	}
	ca, cc := run()
	var pa int
	var pc float64
	withBatchPlant(plantBatchNoBidirectional, func() { pa, pc = run() })
	t.Logf("against the golden: clean argmax %d cosine %.6f; causal block argmax %d cosine %.6f (want %d, bar 0.999)", ca, cc, pa, pc, g.Argmax)
	if ca != g.Argmax || cc < 0.999 {
		t.Fatalf("clean walker already fails the golden")
	}
	if pa == g.Argmax && pc >= 0.999 {
		t.Error("BLIND: the causal block still passes the bidirectional golden")
	}
}

// TestWalkBatchPlant_gemma4ImagePLE: with a one-row image block the bidirectional mask is the causal one, so the batched prefill
// must equal the per-token causal prefill (prefillLogitsGemma4VL), which gives the image row the pad id's PLE. Giving it its
// token id's PLE instead must break that agreement. It needs a PLE fixture (gemma4-emodel-tiny).
func TestWalkBatchPlant_gemma4ImagePLE(t *testing.T) {
	const ckpt = "../testdata/gemma4-emodel-tiny"
	if _, err := os.Stat(ckpt + "/model.safetensors"); errors.Is(err, fs.ErrNotExist) {
		t.Skip("no gemma4-emodel-tiny checkpoint")
	}
	m, err := Load(ckpt, Options{})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if m.w.arch.gemma4.HiddenSizePerLayerInput == 0 {
		t.Fatal("test bug: gemma4-emodel-tiny carries no PLE")
	}
	const n, imgPos, bar = 20, 7, 1e-5
	hidden := m.w.arch.HiddenDim
	ids := make([]int, n)
	for i := range ids {
		ids[i] = (i*37 + 11) % m.w.arch.VocabSize
	}
	feats := make([]float32, hidden)
	for j := range feats {
		feats[j] = float32(math.Sin(float64(j*131))) * 0.5
	}
	want, err := m.prefillLogitsGemma4VL(context.Background(), ids, feats, imgPos, 1, m.NewCache(n))
	if err != nil {
		t.Fatal(err)
	}
	rel := func() float64 {
		got, err := m.prefillLogitsGemma4VLBidirectional(context.Background(), ids, feats, imgPos, 1, m.NewCache(n))
		if err != nil {
			t.Fatal(err)
		}
		return relL2(got, want)
	}
	clean := rel()
	var planted float64
	withBatchPlant(plantBatchImagePLEToken, func() { planted = rel() })
	t.Logf("relative L2 of the last logits, batched against per-token: clean %.3g, image PLE from its token id %.3g (bar %.0e)", clean, planted, bar)
	if clean > bar {
		t.Fatalf("clean walker already past the bar: %.3g", clean)
	}
	if planted <= bar {
		t.Errorf("BLIND: the image row's token-id PLE stays within the bar (%.3g)", planted)
	}
}
