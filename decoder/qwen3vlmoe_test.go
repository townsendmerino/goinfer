package decoder

import (
	"context"
	"encoding/json"
	"errors"
	"io/fs"
	"math"
	"os"
	"slices"
	"testing"
)

// G-S10q-a (docs/tasks/task-multimodal-support-2026-10.md, "S10, Qwen3-VL MoE"): the qwen3_vl_moe decoder against HF's
// Qwen3VLMoeForConditionalGeneration on a tiny random checkpoint whose experts are on disk in the released transformers
// 4.57 layout (scripts/pin_qwen3vlmoe_tiny.py; committed, so this runs in CI). Two inputs, every position at cosine
// >= 0.99999: a text prompt, and an image prompt with random merged rows and three DeepStack sets through the m-RoPE
// image entry (the positions themselves checked against HF's get_rope_index). Planted defects, each must miss the bar:
// the experts read without the transpose, the router not renormalised, the m-RoPE not interleaved, each DeepStack set
// added one layer late.

type qwen3vlmoeGolden struct {
	Vocab      int `json:"vocab"`
	Hidden     int `json:"hidden"`
	ImageToken int `json:"image_token"`
	Merge      int `json:"merge"`
	Text       struct {
		IDs    []int     `json:"ids"`
		Logits []float32 `json:"logits"`
	} `json:"text"`
	Image struct {
		IDs       []int       `json:"ids"`
		Grid      [3]int      `json:"grid"`
		Positions [][3]int    `json:"positions"`
		Merged    []float32   `json:"merged"`
		Deepstack [][]float32 `json:"deepstack"`
		Logits    []float32   `json:"logits"`
	} `json:"image"`
}

const qwen3vlmoeTiny = "../testdata/qwen3vlmoe-tiny"

func loadQwen3VLMoeTiny(t *testing.T) (*Model, qwen3vlmoeGolden) {
	t.Helper()
	raw, err := os.ReadFile("../testdata/qwen3vlmoe_tiny_golden.json")
	if err != nil {
		t.Fatal(err)
	}
	var g qwen3vlmoeGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(qwen3vlmoeTiny + "/model.safetensors"); errors.Is(err, fs.ErrNotExist) {
		t.Fatalf("no %s (it is committed: scripts/pin_qwen3vlmoe_tiny.py)", qwen3vlmoeTiny)
	}
	m, err := Load(qwen3vlmoeTiny, Options{})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	t.Cleanup(func() { m.Close() })
	if m.w.arch.Name != "qwen3_vl_moe" || m.w.arch.MoE == nil || !m.w.arch.MRopeInterleaved {
		t.Fatalf("resolved %q (MoE %v, interleaved m-RoPE %v), want qwen3_vl_moe with both", m.w.arch.Name, m.w.arch.MoE != nil, m.w.arch.MRopeInterleaved)
	}
	return m, g
}

// worstPositionCosine is the lowest per-position cosine between two [n, vocab] logit sets.
func worstPositionCosine(a, b []float32, n int) float64 {
	v := len(a) / n
	worst := 1.0
	for i := range n {
		worst = math.Min(worst, logitCosine(a[i*v:(i+1)*v], b[i*v:(i+1)*v]))
	}
	return worst
}

// qwen3vlmoeText is every position's logits for the text prompt.
func qwen3vlmoeText(t *testing.T, m *Model, ids []int) []float32 {
	hN, err := m.runLayersFromEmbedN(context.Background(), m.embedN(ids), m.NewCache(len(ids)), m.cpuFastAttention())
	if err != nil {
		t.Fatal(err)
	}
	return m.lmHeadN(hN, len(ids))
}

// qwen3vlmoeImage is every position's logits for the image prompt: prefillLogitsQwenVLSpans' path kept open, the
// DeepStack rows the caller's.
func qwen3vlmoeImage(t *testing.T, m *Model, g qwen3vlmoeGolden, mp [][3]int, deep [][]float32) []float32 {
	ids := g.Image.IDs
	hidden := m.w.arch.HiddenDim
	pos, n := -1, 0
	for i, id := range ids {
		if id == g.ImageToken {
			if pos < 0 {
				pos = i
			}
			n++
		}
	}
	spans := []ImageSpan{{Pos: pos, Len: n}}
	cache := m.NewCache(len(ids))
	cache.mropePos, cache.mropeDelta = mp, mropeDelta(mp, len(ids))
	cache.deepstack = &deepstackRows{spans: spans, rows: deep}
	h := m.embedN(ids)
	spliceImageSpans(h, spans, g.Image.Merged, hidden)
	hN, err := m.runLayersFromEmbedN(context.Background(), h, cache, m.cpuFastAttention())
	if err != nil {
		t.Fatal(err)
	}
	return m.lmHeadN(hN, len(ids))
}

func TestQwen3VLMoe_tinyMatchesHF(t *testing.T) {
	m, g := loadQwen3VLMoeTiny(t)
	const bar = 0.99999
	txt := qwen3vlmoeText(t, m, g.Text.IDs)
	ct := worstPositionCosine(txt, g.Text.Logits, len(g.Text.IDs))
	t.Logf("text: worst position cosine %.9f over %d positions", ct, len(g.Text.IDs))
	if ct < bar {
		t.Errorf("text: worst position cosine %.9f under %.5f", ct, bar)
	}

	mp, err := mropePositions(g.Image.IDs, g.ImageToken, [][3]int{g.Image.Grid}, g.Merge)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(mp, g.Image.Positions) {
		t.Fatalf("m-RoPE positions differ from HF's get_rope_index:\n got %v\n HF  %v", mp, g.Image.Positions)
	}
	img := qwen3vlmoeImage(t, m, g, mp, g.Image.Deepstack)
	ci := worstPositionCosine(img, g.Image.Logits, len(g.Image.IDs))
	t.Logf("image: worst position cosine %.9f over %d positions", ci, len(g.Image.IDs))
	if ci < bar {
		t.Errorf("image: worst position cosine %.9f under %.5f", ci, bar)
	}
	// The production entry's last logits are the open loop's.
	pc := m.NewCache(len(g.Image.IDs))
	pos := slices.Index(g.Image.IDs, g.ImageToken)
	spans := []ImageSpan{{Pos: pos, Len: len(g.Image.Merged) / g.Hidden}}
	pc.deepstack = &deepstackRows{spans: spans, rows: g.Image.Deepstack}
	prod, err := m.prefillLogitsQwenVLSpans(context.Background(), g.Image.IDs, spans, g.Image.Merged, mp, pc)
	if err != nil {
		t.Fatal(err)
	}
	v := len(img) / len(g.Image.IDs)
	if c := logitCosine(prod, img[(len(g.Image.IDs)-1)*v:]); c < 0.9999999 {
		t.Errorf("prefillLogitsQwenVLSpans' last logits are not the open loop's (cosine %.9f)", c)
	}
}

func TestQwen3VLMoe_tinyPlantedDefects(t *testing.T) {
	m, g := loadQwen3VLMoeTiny(t)
	const bar = 0.99999
	mp, err := mropePositions(g.Image.IDs, g.ImageToken, [][3]int{g.Image.Grid}, g.Merge)
	if err != nil {
		t.Fatal(err)
	}
	red := func(name string, worst float64) {
		t.Logf("planted %s: worst position cosine %.6f", name, worst)
		if worst >= bar {
			t.Errorf("BLIND: %s reads %.9f", name, worst)
		}
	}

	// 1. The experts read in the row layout whatever their shape (the element-count load).
	fusedExpertsTransposedForTest = true
	bad, err := Load(qwen3vlmoeTiny, Options{})
	fusedExpertsTransposedForTest = false
	if err != nil {
		t.Fatalf("Load with the row layout forced: %v", err)
	}
	red("experts not transposed", worstPositionCosine(qwen3vlmoeText(t, bad, g.Text.IDs), g.Text.Logits, len(g.Text.IDs)))
	bad.Close()

	// 2. The router not renormalised over the top k.
	m.w.arch.MoE.NormTopKProb = false
	red("router not renormalised", worstPositionCosine(qwen3vlmoeText(t, m, g.Text.IDs), g.Text.Logits, len(g.Text.IDs)))
	m.w.arch.MoE.NormTopKProb = true

	// 3. The m-RoPE not interleaved (Qwen2.5-VL's chunked layout): visible only where t, h and w differ, so on the image.
	m.w.arch.MRopeInterleaved = false
	red("m-RoPE chunked, not interleaved", worstPositionCosine(qwen3vlmoeImage(t, m, g, mp, g.Image.Deepstack), g.Image.Logits, len(g.Image.IDs)))
	m.w.arch.MRopeInterleaved = true

	// 4. Each DeepStack set one layer late: an empty set first, so set l lands after layer l+1.
	late := append([][]float32{make([]float32, len(g.Image.Deepstack[0]))}, g.Image.Deepstack...)
	red("DeepStack one layer late", worstPositionCosine(qwen3vlmoeImage(t, m, g, mp, late), g.Image.Logits, len(g.Image.IDs)))

	// Control: the model is back as it was.
	if c := worstPositionCosine(qwen3vlmoeImage(t, m, g, mp, g.Image.Deepstack), g.Image.Logits, len(g.Image.IDs)); c < bar {
		t.Errorf("after the planted defects the model reads %.9f: a defect leaked", c)
	}
}

// TestFusedLayout pins the layout rule loadFusedExperts reads by, HF's own (Transpose(1, 2, check_dims=True)): the row
// shape reads as rows, the square case included (HF does not transpose it either); only the transposed shape reads
// transposed; anything else is refused.
func TestFusedLayout(t *testing.T) {
	for _, c := range []struct {
		shape          []int
		a, b           int
		wantTransposed bool
		wantErr        bool
	}{
		{[]int{128, 1536, 2048}, 1536, 2048, false, false}, // Qwen3.5-MoE's [E, 2I, H]
		{[]int{128, 2048, 1536}, 1536, 2048, true, false},  // the 4.57 layout [E, H, 2I]
		{[]int{8, 64, 64}, 64, 64, false, false},           // square: HF reads it as is
		{[]int{128, 1536 * 2048}, 1536, 2048, false, true}, // equal element count, wrong shape: refused
	} {
		got, err := fusedLayout("t", c.shape, c.shape[0], c.a, c.b)
		if (err != nil) != c.wantErr || (err == nil && got != c.wantTransposed) {
			t.Errorf("shape %v: transposed %v err %v; want %v, error %v", c.shape, got, err, c.wantTransposed, c.wantErr)
		}
	}
}
