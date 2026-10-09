//go:build darwin && goinfer_testhooks

package metal

import (
	"encoding/json"
	"math"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// G-RS1 (S17, docs/tasks/task-multimodal-support-2026-10.md): the batched pass's residual stored scaled by 1/s against
// unscaled, on gemma3-vl-tiny (a sandwich-norm family whose residual stays inside f16). s is a power of two and every read
// is an RMSNorm, so the two differ only through eps and subnormals: last-token cosine >= 0.99999. A family the scale does
// not apply to reports s = 1 and runs the unchanged kernels.
func TestPrefillResidScale_identityTiny(t *testing.T) {
	raw, err := os.ReadFile("../testdata/gemma3_vl_tiny_image_golden.json")
	if err != nil {
		t.Skipf("no golden: %v", err)
	}
	var g struct {
		InputIDs   []int     `json:"input_ids"`
		ImgStart   int       `json:"image_token_start"`
		NImg       int       `json:"mm_tokens_per_image"`
		ImageFeats []float32 `json:"image_features"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	m, err := decoder.Load("../testdata/gemma3-vl-tiny", decoder.Options{Backend: "metal", Quant: "int4"})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	a, ok := m.ResidentForwardForTest().(*metalResident)
	if !ok {
		t.Fatalf("not Metal-resident: %s", m.ResidentDecline())
	}
	if s := a.r.prefillResidScale(nil); s != gemmaPrefillResidScale {
		t.Fatalf("gemma3-vl-tiny's residual scale is %v, want %v: the comparison below would prove nothing", s, gemmaPrefillResidScale)
	}
	hidden, _, _, _, _, _, _ := m.Dims()
	rows := make([][]float32, len(g.InputIDs))
	for i, id := range g.InputIDs {
		if i >= g.ImgStart && i < g.ImgStart+g.NImg {
			rows[i] = g.ImageFeats[(i-g.ImgStart)*hidden : (i-g.ImgStart+1)*hidden]
		} else {
			rows[i] = m.EmbedResidentForTest(id)
		}
	}
	run := func(s float32) []float32 {
		prefillResidScaleForTest = s
		defer func() { prefillResidScaleForTest = 0 }()
		a.Reset()
		l, err := a.batchedPrefill(rows, 0, 0, nil, nil)
		if err != nil {
			t.Fatal(err)
		}
		return append([]float32(nil), l...)
	}
	plain, scaled := run(1), run(0)
	md := 0.0
	for i := range plain {
		md = math.Max(md, math.Abs(float64(plain[i]-scaled[i])))
	}
	c := cosF(plain, scaled)
	t.Logf("s=%d against s=1: last-token cosine %.8f, max |diff| %.3g", gemmaPrefillResidScale, c, md)
	if c < 0.99999 {
		t.Errorf("the scaled residual moves the last token: cosine %.8f (>= 0.99999)", c)
	}
	// A family the scale does not apply to: no sandwich norms.
	b := tinyPrefillResident(t, decoder.Options{Quant: "int4"}, 256)
	if s := b.r.prefillResidScale(nil); s != 1 {
		t.Errorf("a non-sandwich resident's residual scale is %v, want 1", s)
	}
}
