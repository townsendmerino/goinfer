//go:build darwin && goinfer_testhooks

package metal

import (
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime/debug"
	"strings"
	"testing"

	"github.com/townsendmerino/aikit/vision"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/multimodal"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// G-IP3 (S17's Metal image prefill, docs/tasks/task-multimodal-support-2026-10.md): Gemma 3 4B's image turn on S3's four
// images, the last row and 8 teacher-forced decode steps, against the CPU's image prefill at int4 with per-32 activation
// scales (the reference; Step 0 of docs/completed/task-metal-prefill-precision-2026-10.md: the batched f16 pass sits closer
// to it than to per-row). Arms on the Metal resident:
//   - production today: the CPU W4A8 image prefill uploaded, then Metal decode;
//   - the resident image prefill (PrefillImageLast), then Metal decode;
//   - the planted defect: the same spliced rows through the pass with no image block (causal everywhere).
//
// Bar: the resident path's worst cosine >= production's worst - 0.005, and every argmax difference against the reference an
// R10 near-tie (the reference's own p(other) >= half p(top)). The planted defect must miss that bar. GOINFER_HEAVY_TESTS=1.
func TestImagePrefillResident_gemma3Real(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") != "1" {
		t.Skip("heavy-checkpoint test: set GOINFER_HEAVY_TESTS=1")
	}
	home, _ := os.UserHomeDir()
	dir := filepath.Join(home, "models", "gemma-3-4b-it")
	src := dir
	if _, err := os.Stat(dir + ".int4.metal.giw"); err == nil {
		src = dir + ".int4.metal.giw"
	}
	tk, err := tokenizer.Load(filepath.Join(dir, "tokenizer.json"))
	if err != nil {
		t.Fatal(err)
	}
	enc, err := vision.LoadEncoder(dir, false)
	if err != nil {
		t.Fatal(err)
	}
	proj, err := multimodal.LoadProjector(dir)
	if err != nil {
		t.Fatal(err)
	}
	const steps = 8
	ctx := context.Background()
	type turn struct {
		name          string
		ids           []int
		feats         []float32
		imgPos, imgLn int
	}
	var turns []turn
	for _, name := range s3Images {
		data, err := os.ReadFile(filepath.Join("../testdata", name))
		if err != nil {
			t.Fatal(err)
		}
		pv, err := vision.Preprocess(data, vision.Gemma3())
		if err != nil {
			t.Fatal(err)
		}
		h, err := enc.Forward(pv.Data)
		if err != nil {
			t.Fatal(err)
		}
		f, err := proj.Forward(h)
		if err != nil {
			t.Fatal(err)
		}
		// The shape serve builds (vision_serve.go: Gemma3PromptBlock, <bos> first).
		text := "<start_of_turn>user\n" + multimodal.Gemma3PromptBlock(256) + "Describe this image.<end_of_turn>\n<start_of_turn>model\n"
		ids, err := tk.Encode(text, true)
		if err != nil {
			t.Fatal(err)
		}
		soft, _ := tk.TokenID("<image_soft_token>")
		pos, n := -1, 0
		for i, id := range ids {
			if id == soft {
				if pos < 0 {
					pos = i
				}
				n++
			}
		}
		if n != 256 || !strings.Contains(text, "<start_of_image>") {
			t.Fatalf("%s: %d image tokens at %d, want 256", name, n, pos)
		}
		mx := 0.0
		for _, v := range f {
			mx = math.Max(mx, math.Abs(float64(v)))
		}
		turns = append(turns, turn{name, ids, f, pos, n})
		fmt.Fprintf(os.Stderr, "[G-IP3] %s: %d positions, image block [%d,%d), max|feature| %.2f\n", name, len(ids), pos, pos+n, mx)
	}
	// The reference: CPU int4 per-32, prefill + 8 greedy steps (the teacher), then closed before the Metal load.
	ref := make([][][]float32, len(turns))
	teach := make([][]int, len(turns))
	{
		mc, err := decoder.Load(src, decoder.Options{Backend: "cpu", Quant: "int4", ActQuantGroup: 32})
		if err != nil {
			t.Fatal(err)
		}
		for i, tr := range turns {
			c := mc.NewCache(len(tr.ids) + steps + 1)
			l, err := mc.PrefillLogitsVLForTest(ctx, tr.ids, tr.feats, tr.imgPos, tr.imgLn, c)
			if err != nil {
				t.Fatal(err)
			}
			ref[i] = [][]float32{append([]float32(nil), l...)}
			teach[i] = []int{argmaxF(l)}
			for k := range steps {
				l, err := mc.ForwardForTest(teach[i][k], c)
				if err != nil {
					t.Fatal(err)
				}
				ref[i] = append(ref[i], append([]float32(nil), l...))
				teach[i] = append(teach[i], argmaxF(l))
			}
		}
		mc.Close()
		debug.FreeOSMemory()
	}
	m, err := decoder.Load(src, decoder.Options{Backend: "metal", Quant: "int4", ResidentContext: 1024})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	a, ok := m.ResidentForwardForTest().(*metalResident)
	if !ok {
		t.Fatalf("not Metal-resident: %s", m.ResidentDecline())
	}
	hidden, _, _, _, _, _, _ := m.Dims()
	// grade returns an arm's worst cosine against the reference and its non-tie argmax differences.
	grade := func(i int, first []float32) (worst float64, real int) {
		tr := turns[i]
		got := [][]float32{first}
		for k := range steps {
			l, err := a.Forward(m.EmbedResidentForTest(teach[i][k]), len(tr.ids)+k)
			if err != nil {
				t.Fatal(err)
			}
			got = append(got, append([]float32(nil), l...))
		}
		worst = 1
		for k := range got {
			worst = math.Min(worst, cosF(got[k], ref[i][k]))
			if ra, ga := argmaxF(ref[i][k]), argmaxF(got[k]); ra != ga {
				if lp := logSoftmaxF(ref[i][k]); math.Exp(lp[ga]) < math.Exp(lp[ra])/2 {
					real++
				}
			}
		}
		return
	}
	prodWorst, newWorst, badWorst := 1.0, 1.0, 1.0
	newReal := 0
	for i, tr := range turns {
		// Production today: the CPU W4A8 image prefill, uploaded.
		c := m.NewCache(len(tr.ids) + steps + 1)
		l, err := m.PrefillLogitsVLForTest(ctx, tr.ids, tr.feats, tr.imgPos, tr.imgLn, c)
		if err != nil {
			t.Fatal(err)
		}
		l = append([]float32(nil), l...)
		a.Reset()
		if err := m.ResidentUploadPrefillForTest(c); err != nil {
			t.Fatal(err)
		}
		pw, _ := grade(i, l)
		// The resident image prefill.
		a.Reset()
		ln, _, err := m.ResidentImagePrefillForTest(ctx, a, tr.ids, tr.feats, tr.imgPos, tr.imgLn)
		if err != nil {
			t.Fatalf("%s: resident image prefill declined: %v", tr.name, err)
		}
		nw, nr := grade(i, append([]float32(nil), ln...))
		// The planted defect: no image block.
		rows := make([][]float32, len(tr.ids))
		for j, id := range tr.ids {
			if j >= tr.imgPos && j < tr.imgPos+tr.imgLn {
				rows[j] = tr.feats[(j-tr.imgPos)*hidden : (j-tr.imgPos+1)*hidden]
			} else {
				rows[j] = m.EmbedResidentForTest(id)
			}
		}
		a.Reset()
		lb, err := a.batchedPrefill(rows, 0, 0, nil, nil)
		if err != nil {
			t.Fatal(err)
		}
		bw, _ := grade(i, append([]float32(nil), lb...))
		fmt.Fprintf(os.Stderr, "[G-IP3] %-32s worst cos: production %.6f | resident image prefill %.6f (%d non-tie argmax) | planted no-block %.6f\n", tr.name, pw, nw, nr, bw)
		prodWorst, newWorst, badWorst = math.Min(prodWorst, pw), math.Min(newWorst, nw), math.Min(badWorst, bw)
		newReal += nr
	}
	fmt.Fprintf(os.Stderr, "[G-IP3] over the four images: production %.6f | resident %.6f (%d non-tie) | planted %.6f | bar %.6f\n", prodWorst, newWorst, newReal, badWorst, prodWorst-0.005)
	if newWorst < prodWorst-0.005 || newReal > 0 {
		t.Errorf("resident image prefill: worst %.6f against the bar %.6f, %d non-tie argmax differences", newWorst, prodWorst-0.005, newReal)
	}
	if badWorst >= prodWorst-0.005 {
		t.Errorf("the gate is BLIND: with no image block the pass reads %.6f >= the bar %.6f", badWorst, prodWorst-0.005)
	}
}

// TestImagePrefillResident_gemma3TwoImagesReal (S11 step 4; an exploratory real check at G-IP3's bar, not a registered gate):
// Gemma 3 4B with table.png and formula.png in one prompt, the shape serve builds. Against the CPU's two-image prefill at int4
// per-32 (the reference), the last row and 8 teacher-forced steps: today's CPU W4A8 bridge uploaded, and the resident
// two-block prefill. Asserted: the resident path's worst cosine >= the bridge's worst - 0.005. Logged, not asserted: argmax
// differences against the reference and whether each is an R10 near-tie. The first read (2026-10-09) had one that is not
// (step 1, " are" 0.819 against "'" 0.143 in the reference), the G-IP4 pattern; whether it is accepted for several images is
// the owner's open decision (docs/tasks/task-multimodal-support-2026-10.md, S11 step 4). GOINFER_HEAVY_TESTS=1.
func TestImagePrefillResident_gemma3TwoImagesReal(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") != "1" {
		t.Skip("heavy-checkpoint test: set GOINFER_HEAVY_TESTS=1")
	}
	home, _ := os.UserHomeDir()
	dir := filepath.Join(home, "models", "gemma-3-4b-it")
	src := dir + ".int4.metal.giw"
	tk, err := tokenizer.Load(filepath.Join(dir, "tokenizer.json"))
	if err != nil {
		t.Fatal(err)
	}
	enc, err := vision.LoadEncoder(dir, false)
	if err != nil {
		t.Fatal(err)
	}
	proj, err := multimodal.LoadProjector(dir)
	if err != nil {
		t.Fatal(err)
	}
	var feats []float32
	for _, name := range []string{"glm_ocr/table.png", "glm_ocr/formula.png"} {
		data, err := os.ReadFile(filepath.Join("../testdata", name))
		if err != nil {
			t.Fatal(err)
		}
		pv, err := vision.Preprocess(data, vision.Gemma3())
		if err != nil {
			t.Fatal(err)
		}
		h, err := enc.Forward(pv.Data)
		if err != nil {
			t.Fatal(err)
		}
		f, err := proj.Forward(h)
		if err != nil {
			t.Fatal(err)
		}
		feats = append(feats, f...)
	}
	block := multimodal.Gemma3PromptBlock(256)
	ids, err := tk.Encode("<start_of_turn>user\nHere are two images. "+block+" and "+block+" What does each image show? Answer briefly, one line per image.<end_of_turn>\n<start_of_turn>model\n", true)
	if err != nil {
		t.Fatal(err)
	}
	soft, _ := tk.TokenID("<image_soft_token>")
	var spans []decoder.ImageSpan
	for _, r := range multimodal.FindImageRuns(ids, soft) {
		spans = append(spans, decoder.ImageSpan{Pos: r[0], Len: r[1]})
	}
	if len(spans) != 2 || spans[0].Len != 256 || spans[1].Len != 256 {
		t.Fatalf("image runs %v, want two of 256", spans)
	}
	const steps = 8
	ctx := context.Background()
	var ref [][]float32
	var teach []int
	{
		mc, err := decoder.Load(src, decoder.Options{Backend: "cpu", Quant: "int4", ActQuantGroup: 32})
		if err != nil {
			t.Fatal(err)
		}
		c := mc.NewCache(len(ids) + steps + 1)
		l, err := mc.PrefillLogitsVLSpansForTest(ctx, ids, spans, feats, c)
		if err != nil {
			t.Fatal(err)
		}
		ref, teach = [][]float32{append([]float32(nil), l...)}, []int{argmaxF(l)}
		for k := range steps {
			l, err := mc.ForwardForTest(teach[k], c)
			if err != nil {
				t.Fatal(err)
			}
			ref = append(ref, append([]float32(nil), l...))
			teach = append(teach, argmaxF(l))
		}
		mc.Close()
	}
	m, err := decoder.Load(src, decoder.Options{Backend: "metal", Quant: "int4", ResidentContext: 1024})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	a, ok := m.ResidentForwardForTest().(*metalResident)
	if !ok {
		t.Fatalf("not Metal-resident: %s", m.ResidentDecline())
	}
	grade := func(first []float32) (worst float64, real int) {
		got := [][]float32{first}
		for k := range steps {
			l, err := a.Forward(m.EmbedResidentForTest(teach[k]), len(ids)+k)
			if err != nil {
				t.Fatal(err)
			}
			got = append(got, append([]float32(nil), l...))
		}
		worst = 1
		for k := range got {
			worst = math.Min(worst, cosF(got[k], ref[k]))
			if ra, ga := argmaxF(ref[k]), argmaxF(got[k]); ra != ga {
				lp, gp := logSoftmaxF(ref[k]), logSoftmaxF(got[k])
				rs, _ := tk.Decode([]int{ra})
				gs, _ := tk.Decode([]int{ga})
				fmt.Fprintf(os.Stderr, "[S11 real]   step %d: reference %q p %.3f, this arm %q (reference p %.3f; this arm's own p %.3f vs %.3f)\n", k, rs, math.Exp(lp[ra]), gs, math.Exp(lp[ga]), math.Exp(gp[ga]), math.Exp(gp[ra]))
				if math.Exp(lp[ga]) < math.Exp(lp[ra])/2 {
					real++
				}
			}
		}
		return
	}
	c := m.NewCache(len(ids) + steps + 1)
	l, err := m.PrefillLogitsVLSpansForTest(ctx, ids, spans, feats, c)
	if err != nil {
		t.Fatal(err)
	}
	l = append([]float32(nil), l...)
	a.Reset()
	if err := m.ResidentUploadPrefillForTest(c); err != nil {
		t.Fatal(err)
	}
	fmt.Fprintln(os.Stderr, "[S11 real] the bridge:")
	bw, br := grade(l)
	a.Reset()
	ln, _, err := m.ResidentImagePrefillSpansForTest(ctx, a, ids, feats, spans)
	if err != nil {
		t.Fatalf("the resident two-image prefill declined: %v", err)
	}
	fmt.Fprintln(os.Stderr, "[S11 real] the resident two-block prefill:")
	rw, rr := grade(append([]float32(nil), ln...))
	fmt.Fprintf(os.Stderr, "[S11 real] %d positions, blocks %v: worst cos: bridge %.6f (%d non-tie) | resident two-block %.6f (%d non-tie) | bar %.6f\n", len(ids), spans, bw, br, rw, rr, bw-0.005)
	if rw < bw-0.005 {
		t.Errorf("the resident two-block prefill: worst %.6f against the bar %.6f", rw, bw-0.005)
	}
	if rr > 0 {
		t.Logf("the resident two-block prefill has %d argmax differences that are not R10 near-ties (logged above); open for the owner, see the doc comment", rr)
	}
}
