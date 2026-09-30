package decoder

import (
	"context"
	"encoding/json"
	"errors"
	"io/fs"
	"os"
	"slices"
	"testing"

	"github.com/townsendmerino/aikit/vision"
)

// P8a gate G1 (docs/measurements/p8a-qwen35-vl-2026-09/preregistration.md): the tiny Qwen3.5 image
// fixture (scripts/pin_qwen35_vl_tiny.py) — real tower structure, the 3:1 Gated-DeltaNet hybrid,
// interleaved m-RoPE with partial rotary. Bars: position_ids and rope_delta EXACT; image_features
// per-row cosine >= 0.9999; last_logits cosine >= 0.9999; argmax exact and all 8 continuation
// tokens exact.
const (
	qwen35VLCkpt   = "../testdata/qwen35vl-tiny"
	qwen35VLGolden = "../testdata/qwen35vl_tiny_image_golden.json"
	qwen35VLMerge  = 2 // vision_config.spatial_merge_size
)

type qwen35VLGold struct {
	InputIDs      []int     `json:"input_ids"`
	ImageToken    int       `json:"image_token_id"`
	ImageStart    int       `json:"image_token_start"`
	NImageTokens  int       `json:"n_image_tokens"`
	GridTHW       [][3]int  `json:"grid_thw"`
	PixelValues   []float32 `json:"pixel_values"`
	ImageFeatures []float32 `json:"image_features"`
	PositionIDs   [][]int   `json:"position_ids"`
	RopeDelta     int       `json:"rope_delta"`
	Argmax        int       `json:"argmax"`
	LastLogits    []float32 `json:"last_logits"`
	Continuation  []int     `json:"continuation_ids"`
}

func loadQwen35VLTiny(t *testing.T) (*Model, qwen35VLGold) {
	t.Helper()
	raw, err := os.ReadFile(qwen35VLGolden)
	if errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no golden — run scripts/pin_qwen35_vl_tiny.py")
	}
	if err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(qwen35VLCkpt); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no checkpoint — run scripts/pin_qwen35_vl_tiny.py")
	}
	var g qwen35VLGold
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	m, err := Load(qwen35VLCkpt, Options{})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	t.Cleanup(func() { _ = m.Close() })
	if m.w.arch.Name != "qwen3_5" || m.w.arch.qwen35 == nil {
		t.Fatalf("arch %q: the wrong adapter ran for a qwen3_5 image checkpoint", m.w.arch.Name)
	}
	return m, g
}

// rowCosMin is the worst per-row cosine of two [n, dim] matrices.
func rowCosMin(a, b []float32, dim int) float64 {
	worst := 2.0
	for r := 0; r+dim <= len(a); r += dim {
		if c := logitCosine(a[r:r+dim], b[r:r+dim]); c < worst {
			worst = c
		}
	}
	return worst
}

func TestQwen35VL_loaderSetsMRope(t *testing.T) {
	m, _ := loadQwen35VLTiny(t)
	a := m.w.arch
	if !slices.Equal(a.MRopeSection, []int{3, 3, 2}) || !a.MRopeInterleaved {
		t.Fatalf("MRopeSection %v interleaved %v, want [3 3 2] true (rope_parameters)", a.MRopeSection, a.MRopeInterleaved)
	}
	if err := m.checkQwen35VLReady(); err != nil {
		t.Fatalf("checkQwen35VLReady on the image checkpoint: %v", err)
	}
	// The refusal must be real: a config without a usable section is refused, not scalar-rotated.
	saved := a.MRopeSection
	defer func() { a.MRopeSection = saved }()
	for _, bad := range [][]int{nil, {3, 3}, {3, 3, 3}} {
		a.MRopeSection = bad
		if err := m.checkQwen35VLReady(); err == nil {
			t.Errorf("MRopeSection %v accepted", bad)
		}
	}
}

func TestQwen35VL_positions(t *testing.T) {
	_, g := loadQwen35VLTiny(t)
	got, err := mropePositions(g.InputIDs, g.ImageToken, g.GridTHW, qwen35VLMerge)
	if err != nil {
		t.Fatal(err)
	}
	for i := range got {
		for c := range 3 {
			if got[i][c] != g.PositionIDs[c][i] {
				t.Fatalf("pos[%d][%d] = %d, HF get_rope_index %d", i, c, got[i][c], g.PositionIDs[c][i])
			}
		}
	}
	if d := mropeDelta(got, len(g.InputIDs)); d != g.RopeDelta {
		t.Errorf("mropeDelta %d, HF rope_deltas %d", d, g.RopeDelta)
	}
}

// qwen35VLDecode prefills through the image seam and decodes greedily, returning the last-prompt
// logits and the continuation.
func qwen35VLDecode(t *testing.T, m *Model, g qwen35VLGold, feats []float32) ([]float32, []int) {
	t.Helper()
	pos, err := mropePositions(g.InputIDs, g.ImageToken, g.GridTHW, qwen35VLMerge)
	if err != nil {
		t.Fatal(err)
	}
	cache := m.NewCache(len(g.InputIDs) + len(g.Continuation))
	logits, err := m.prefillLogitsQwenVL(context.Background(), g.InputIDs, feats, g.ImageStart, g.NImageTokens, pos, cache)
	if err != nil {
		t.Fatalf("prefillLogitsQwenVL: %v", err)
	}
	last := slices.Clone(logits)
	var got []int
	for range g.Continuation {
		id := argmax(logits)
		got = append(got, id)
		if logits, err = m.forward(id, cache); err != nil {
			t.Fatalf("decode forward: %v", err)
		}
	}
	return last, got
}

func checkQwen35VLLogits(t *testing.T, last []float32, got []int, g qwen35VLGold) {
	t.Helper()
	cos := logitCosine(last, g.LastLogits)
	t.Logf("last-logit cosine %.9f, argmax %d (HF %d), continuation %v (HF %v)", cos, argmax(last), g.Argmax, got, g.Continuation)
	if argmax(last) != g.Argmax {
		t.Errorf("argmax %d, want %d", argmax(last), g.Argmax)
	}
	if cos < 0.9999 {
		t.Errorf("last-logit cosine %.9f < 0.9999", cos)
	}
	if !slices.Equal(got, g.Continuation) {
		t.Errorf("continuation %v, want %v (decode past an image: m-RoPE resumes at seqPos+delta)", got, g.Continuation)
	}
}

// TestQwen35VL_imageParity feeds HF's own image_features, isolating the decoder side (splice,
// m-RoPE on the full-attention layers, recurrent layers ignoring positions, decode delta).
func TestQwen35VL_imageParity(t *testing.T) {
	m, g := loadQwen35VLTiny(t)
	last, got := qwen35VLDecode(t, m, g, g.ImageFeatures)
	checkQwen35VLLogits(t, last, got, g)
}

// TestQwen35VL_e2eChain chains the two stages goinfer wires: the aikit tower on the golden
// pixel_values, then the decoder, so an encoder<->decoder seam mismatch cannot hide.
func TestQwen35VL_e2eChain(t *testing.T) {
	m, g := loadQwen35VLTiny(t)
	enc, err := vision.LoadQwen3VisionEncoder(qwen35VLCkpt, false)
	if err != nil {
		t.Fatalf("LoadQwen3VisionEncoder: %v", err)
	}
	feats, err := enc.Forward(g.PixelValues, g.GridTHW)
	if err != nil {
		t.Fatal(err)
	}
	hidden := m.w.arch.HiddenDim
	if len(feats) != g.NImageTokens*hidden {
		t.Fatalf("tower emitted %d floats, want %d tokens x %d", len(feats), g.NImageTokens, hidden)
	}
	if c := rowCosMin(feats, g.ImageFeatures, hidden); c < 0.9999 {
		t.Errorf("image_features worst-row cosine %.9f < 0.9999", c)
	}
	last, got := qwen35VLDecode(t, m, g, feats)
	checkQwen35VLLogits(t, last, got, g)
}

// TestGenerateQwenVL_recurrentTakesNoResidentBranch: with a resident that implements
// ResidentMRoPE attached (every real executor does), an image turn on a Gated-DeltaNet hybrid must
// not touch it — not UploadKV (which would skip the KV-less DeltaNet layers and leave their state
// zeroed), not ForwardMRoPE — and must produce the CPU continuation. The fake fails the test if
// any resident method runs.
func TestGenerateQwenVL_recurrentTakesNoResidentBranch(t *testing.T) {
	m, g := loadQwen35VLTiny(t)
	rf := &fakeResident{vocab: m.w.arch.VocabSize}
	m.resident = rf
	m.resIDs = []int{9, 9, 9}
	if !m.hasRecurrentState() {
		t.Fatal("qwen3_5 is not classified recurrent: the guard under test would never fire")
	}
	features := func() ([]float32, error) { return g.ImageFeatures, nil }
	stream, gen := m.GenerateQwenVL(context.Background(), g.InputIDs, g.ImageStart, g.NImageTokens, 0xfeed, features,
		g.GridTHW, qwen35VLMerge, g.ImageToken, len(g.Continuation), SamplingParams{Temperature: 0})
	var got []int
	for id := range stream {
		got = append(got, id)
	}
	if err := gen.Err(); err != nil {
		t.Fatalf("GenerateQwenVL: %v", err)
	}
	if rf.forwards != 0 || len(rf.uploadKVs) != 0 || rf.lastRopePos != 0 {
		t.Errorf("resident touched: %d forwards, %d UploadKV, ropePos %d — the zero-state bridge ran", rf.forwards, len(rf.uploadKVs), rf.lastRopePos)
	}
	if gen.ImgPrefillResident || gen.PrefillReused != 0 {
		t.Errorf("Generation claims resident work: ImgPrefillResident=%v PrefillReused=%d", gen.ImgPrefillResident, gen.PrefillReused)
	}
	if !slices.Equal(got, g.Continuation) {
		t.Errorf("GenerateQwenVL streamed %v, want %v", got, g.Continuation)
	}
	if !slices.Equal(m.resIDs, []int{9, 9, 9}) {
		t.Errorf("resIDs changed to %v by a CPU-only image turn", m.resIDs)
	}
}
