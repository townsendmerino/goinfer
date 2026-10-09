//go:build darwin && goinfer_testhooks

package metal

import (
	"context"
	"encoding/json"
	"math"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// G-IP2 (S17's Metal image prefill, docs/tasks/task-multimodal-support-2026-10.md): gemma3-vl-tiny's image turn through the
// resident image prefill (PrefillImageLast, via the decoder's own splice), its last-token logits and 8 decode steps after it
// teacher-forced on the CPU's continuation, against the CPU's image prefill (prefillLogitsVL) at int4 with per-32 activation
// scales. Bars: cosine >= 0.995, relL2 <= 0.15. Planted defect: the same spliced rows through the pass with no image block
// (causal everywhere) must read red, or the fixture is blind to the mask and the gate rests on G-IP3 (logged, not failed).
func TestImagePrefillResident_gemma3Tiny(t *testing.T) {
	raw, err := os.ReadFile("../testdata/gemma3_vl_tiny_image_golden.json")
	if err != nil {
		t.Skipf("no golden: %v", err)
	}
	var g struct {
		InputIDs   []int     `json:"input_ids"`
		ImgTok     int       `json:"image_token_index"`
		ImgStart   int       `json:"image_token_start"`
		NImg       int       `json:"mm_tokens_per_image"`
		ImageFeats []float32 `json:"image_features"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	const ckpt, steps = "../testdata/gemma3-vl-tiny", 8
	ctx := context.Background()
	ref, err := decoder.Load(ckpt, decoder.Options{Backend: "cpu", Quant: "int4", ActQuantGroup: 32})
	if err != nil {
		t.Fatal(err)
	}
	defer ref.Close()
	rc := ref.NewCache(len(g.InputIDs) + steps + 1)
	first, err := ref.PrefillLogitsVLForTest(ctx, g.InputIDs, g.ImageFeats, g.ImgStart, g.NImg, rc)
	if err != nil {
		t.Fatal(err)
	}
	want := [][]float32{append([]float32(nil), first...)}
	teach := []int{argmaxF(first)}
	for k := range steps {
		l, err := ref.ForwardForTest(teach[k], rc)
		if err != nil {
			t.Fatal(err)
		}
		want = append(want, append([]float32(nil), l...))
		teach = append(teach, argmaxF(l))
	}
	m, err := decoder.Load(ckpt, decoder.Options{Backend: "metal", Quant: "int4"})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	a, ok := m.ResidentForwardForTest().(*metalResident)
	if !ok {
		t.Fatalf("not Metal-resident: %s", m.ResidentDecline())
	}
	n := len(g.InputIDs)
	decodeAfter := func(first []float32) (worstCos, worstRel float64) {
		worstCos = 1
		got := [][]float32{first}
		for k := range steps {
			l, err := a.Forward(m.EmbedResidentForTest(teach[k]), n+k)
			if err != nil {
				t.Fatal(err)
			}
			got = append(got, append([]float32(nil), l...))
		}
		for i := range got {
			worstCos = math.Min(worstCos, cosF(got[i], want[i]))
			worstRel = math.Max(worstRel, pwRelL2(got[i], want[i]))
		}
		return
	}
	a.Reset()
	l, gpuPos, err := m.ResidentImagePrefillForTest(ctx, a, g.InputIDs, g.ImageFeats, g.ImgStart, g.NImg)
	if err != nil {
		t.Fatalf("resident image prefill declined: %v", err)
	}
	if gpuPos != n {
		t.Fatalf("gpuPos %d, want %d", gpuPos, n)
	}
	c, r := decodeAfter(append([]float32(nil), l...))
	t.Logf("resident image prefill: worst cos %.6f, worst relL2 %.4f (last token + %d steps)", c, r, steps)
	if c < 0.995 || r > 0.15 {
		t.Errorf("resident image prefill diverges from the CPU: worst cosine %.6f (>= 0.995), relL2 %.4f (<= 0.15)", c, r)
	}
	// Planted defect: the same spliced rows, no image block.
	hidden, _, _, _, _, _, _ := m.Dims()
	rows := make([][]float32, n)
	for i, id := range g.InputIDs {
		if i >= g.ImgStart && i < g.ImgStart+g.NImg {
			rows[i] = g.ImageFeats[(i-g.ImgStart)*hidden : (i-g.ImgStart+1)*hidden]
		} else {
			rows[i] = m.EmbedResidentForTest(id)
		}
	}
	a.Reset()
	bad, err := a.batchedPrefill(rows, 0, 0, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	bc, br := decodeAfter(append([]float32(nil), bad...))
	t.Logf("planted defect (no image block, causal everywhere): worst cos %.6f, worst relL2 %.4f", bc, br)
	if bc >= 0.995 && br <= 0.15 {
		t.Logf("BLIND: gemma3-vl-tiny's 4-token block does not move the logits past the bar without the mask; G-IP3 carries the gate (as registered)")
	}
}
