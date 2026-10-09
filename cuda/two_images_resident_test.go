//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"encoding/json"
	"math"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestImagePrefillResident_twoImagesTinyCUDA (G-S11c, CUDA; S11 of docs/tasks/task-multimodal-support-2026-10.md):
// gemma3-vl-tiny with two images (the two-image golden's layouts and features), the resident multi-block image prefill
// (PrefillImageBlocksLast) against the CPU's two-image prefill at int4 per-32, the last token and 8 teacher-forced decode
// steps. Bars as Metal's: cosine >= 0.995, relL2 <= 0.15.
func TestImagePrefillResident_twoImagesTinyCUDA(t *testing.T) {
	raw, err := os.ReadFile("../testdata/gemma3_vl_tiny_two_images_golden.json")
	if err != nil {
		t.Skipf("no golden: %v", err)
	}
	var g struct {
		ImageFeatures [][]float32 `json:"image_features"`
		Layouts       map[string]struct {
			InputIDs []int    `json:"input_ids"`
			Spans    [][2]int `json:"spans"`
		} `json:"layouts"`
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
	m, err := decoder.Load(ckpt, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Skipf("no CUDA load: %v", err)
	}
	defer m.Close()
	rf := m.ResidentForwardForTest()
	r, ok := rf.(*cudaResident)
	if !ok {
		t.Skipf("not CUDA-resident: %s", m.ResidentDecline())
	}
	if !r.imgPrefillReady {
		t.Skip("attn_img_batched did not load on this build")
	}
	feats := append(append([]float32(nil), g.ImageFeatures[0]...), g.ImageFeatures[1]...)
	for _, name := range []string{"interleaved", "adjacent"} {
		l := g.Layouts[name]
		spans := []decoder.ImageSpan{{Pos: l.Spans[0][0], Len: l.Spans[0][1]}, {Pos: l.Spans[1][0], Len: l.Spans[1][1]}}
		n := len(l.InputIDs)
		rc := ref.NewCache(n + steps + 1)
		first, err := ref.PrefillLogitsVLSpansForTest(ctx, l.InputIDs, spans, feats, rc)
		if err != nil {
			t.Fatal(err)
		}
		want := [][]float32{append([]float32(nil), first...)}
		teach := []int{argmaxF(first)}
		for k := range steps {
			lg, err := ref.ForwardForTest(teach[k], rc)
			if err != nil {
				t.Fatal(err)
			}
			want = append(want, append([]float32(nil), lg...))
			teach = append(teach, argmaxF(lg))
		}
		rf.Reset()
		lg, gpuPos, err := m.ResidentImagePrefillSpansForTest(ctx, r, l.InputIDs, feats, spans)
		if err != nil {
			t.Fatalf("%s: the resident two-image prefill declined: %v", name, err)
		}
		if gpuPos != n {
			t.Fatalf("%s: gpuPos %d, want %d", name, gpuPos, n)
		}
		got := [][]float32{append([]float32(nil), lg...)}
		for k := range steps {
			lg, err := rf.Forward(m.EmbedResidentForTest(teach[k]), n+k)
			if err != nil {
				t.Fatal(err)
			}
			got = append(got, append([]float32(nil), lg...))
		}
		worstCos, worstRel := 1.0, 0.0
		for i := range got {
			worstCos = math.Min(worstCos, cosineF32(got[i], want[i]))
			var num, den float64
			for j := range got[i] {
				d := float64(got[i][j] - want[i][j])
				num += d * d
				den += float64(want[i][j]) * float64(want[i][j])
			}
			worstRel = math.Max(worstRel, math.Sqrt(num/den))
		}
		t.Logf("%s: resident two-image prefill: worst cos %.6f, worst relL2 %.4f (last token + %d steps)", name, worstCos, worstRel, steps)
		if worstCos < 0.995 || worstRel > 0.15 {
			t.Errorf("%s: the resident two-image prefill diverges from the CPU: worst cosine %.6f (>= 0.995), relL2 %.4f (<= 0.15)", name, worstCos, worstRel)
		}
	}
}
