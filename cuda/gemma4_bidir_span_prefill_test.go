//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"errors"
	"io/fs"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// S19: the bidirectional Gemma 4 image prefill on the resident (PrefillSpansLast, attn_span_batched). The plan, the gates
// and the bars: docs/tasks/task-multimodal-support-2026-10.md ("S19").

// spanFixtures are the two bidirectional fixtures: tiny (4 layers, window 3) and scaled (12 layers, the real 256/512 head
// dims, window 8). Both windows are shorter than every span below, so a span row's reach-back to its span start is
// exercised on every sliding layer.
var spanFixtures = []string{"../testdata/gemma4-vl-bidir-tiny", "../testdata/gemma4-vl-bidir-scaled"}

// spanLayouts place the spans in a 26-position prompt: one 9-row span at the start, in the middle and at the end, and
// two spans.
var spanLayouts = []struct {
	name  string
	spans []decoder.ImageSpan
}{
	{"span-first", []decoder.ImageSpan{{Pos: 0, Len: 9}}},
	{"span-middle", []decoder.ImageSpan{{Pos: 8, Len: 9}}},
	{"span-last", []decoder.ImageSpan{{Pos: 17, Len: 9}}},
	{"two-spans", []decoder.ImageSpan{{Pos: 2, Len: 7}, {Pos: 13, Len: 10}}},
}

const spanPromptLen = 26

// spanMinCos is G-S19a's cosine bar (S9's G1q row bar, as amended 2026-10-23): both sides run the same int4 weights.
const spanMinCos = 0.999

// spanPrompt is a fixed prompt of text ids (the span rows are overwritten by features).
func spanPrompt() []int {
	ids := make([]int, spanPromptLen)
	for i := range ids {
		ids[i] = 2 + (i*37)%200
	}
	return ids
}

type spanGrade struct {
	layout       string
	cos          float64
	argGPU, argC int
}

func (g spanGrade) pass() bool { return g.argGPU == g.argC && g.cos >= spanMinCos }

// spanRig is one fixture loaded on CUDA and on the CPU, both int4.
type spanRig struct {
	name    string
	mg, mc4 *decoder.Model
	r       *cudaResident
}

func loadSpanRig(t *testing.T, dir string) *spanRig {
	t.Helper()
	if _, err := os.Stat(dir + "/model.safetensors"); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no fixture (%s): run scripts/pin_gemma4_vl_bidir_tiny.py / pin_gemma4_vl_bidir_scaled.py", dir)
	}
	mg, err := decoder.Load(dir, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load (cuda int4): %v", err)
	}
	t.Cleanup(func() { mg.Close() })
	mc4, err := decoder.Load(dir, decoder.Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load (cpu int4): %v", err)
	}
	t.Cleanup(func() { mc4.Close() })
	if got := mg.Config().UseBidirectionalAttention; got != "vision" {
		t.Fatalf("%s: use_bidirectional_attention %q, want \"vision\"", dir, got)
	}
	r := mustResident(t, mg)
	if !r.spanPrefillReady {
		t.Fatal("attn_span_batched did not load")
	}
	return &spanRig{name: dir, mg: mg, mc4: mc4, r: r}
}

// grade runs every layout through the resident's span prefill and the CPU forward.
func (s *spanRig) grade(t *testing.T) []spanGrade {
	t.Helper()
	ctx := context.Background()
	ids := spanPrompt()
	hidden := s.r.hidden
	var out []spanGrade
	for _, lay := range spanLayouts {
		n := 0
		for _, sp := range lay.spans {
			n += sp.Len
		}
		feats := eModelFeatures(n, hidden)
		rows, err := s.mg.Gemma4BidirImageRowsForTest(ids, feats, lay.spans)
		if err != nil {
			t.Fatalf("%s: rows: %v", lay.name, err)
		}
		gpu, err := s.r.PrefillSpansLast(ctx, rows, 0, spanBlocks(lay.spans))
		if err != nil {
			t.Fatalf("%s: PrefillSpansLast: %v", lay.name, err)
		}
		gpu = append([]float32(nil), gpu...)
		cpu, err := s.mc4.PrefillLogitsGemma4VLBidirectionalSpansForTest(ctx, ids, feats, lay.spans, s.mc4.NewCache(len(ids)+1))
		if err != nil {
			t.Fatalf("%s: cpu reference: %v", lay.name, err)
		}
		g := spanGrade{layout: lay.name, argGPU: argmaxF(gpu), argC: argmaxF(cpu)}
		g.cos, _ = cosMaxAbs(gpu, cpu)
		out = append(out, g)
	}
	return out
}

func spanBlocks(spans []decoder.ImageSpan) [][2]int {
	b := make([][2]int, len(spans))
	for i, s := range spans {
		b[i] = [2]int{s.Pos, s.Pos + s.Len}
	}
	return b
}

// TestGemma4BidirSpanPrefill_matchesCPU is G-S19a: on both fixtures and every layout, the resident's last-position logits
// against the CPU bidirectional forward (argmax identical, cosine at least spanMinCos); one pass bit-identical to passes of
// 4 rows cut outside the spans; and the spans dropped (a causal prefill) must fail the same bar, or the fixture cannot
// tell the mask from no mask.
func TestGemma4BidirSpanPrefill_matchesCPU(t *testing.T) {
	ctx := context.Background()
	for _, dir := range spanFixtures {
		s := loadSpanRig(t, dir)
		for _, g := range s.grade(t) {
			t.Logf("%s %-11s cos %.6f argmax gpu=%d cpu=%d", dir, g.layout, g.cos, g.argGPU, g.argC)
			if !g.pass() {
				t.Errorf("G-S19a FAIL %s %s: cos %.6f (bar %.3f), argmax gpu=%d cpu=%d", dir, g.layout, g.cos, spanMinCos, g.argGPU, g.argC)
			}
		}
		// The control G-S19a's bar is read beside: the shipped text prefill (PrefillLast on embedResident rows) against the CPU's
		// causal forward of the same ids, no span anywhere. On the scaled fixture it sits under spanMinCos too.
		ids0 := spanPrompt()
		er := make([][]float32, len(ids0))
		for i, id := range ids0 {
			er[i] = s.mg.EmbedResidentForTest(id)
		}
		if txt, err := s.r.PrefillLast(ctx, er, 0); err != nil {
			t.Fatal(err)
		} else {
			txt = append([]float32(nil), txt...)
			cpu, err := s.mc4.PrefillLogitsForTest(ctx, ids0, s.mc4.NewCache(len(ids0)+1))
			if err != nil {
				t.Fatal(err)
			}
			c, _ := cosMaxAbs(txt, cpu)
			t.Logf("%s control, text only: shipped PrefillLast vs CPU int4 cos %.6f argmax gpu=%d cpu=%d", dir, c, argmaxF(txt), argmaxF(cpu))
		}
		// Bit identity across pass cuts: one pass against passes of 4 rows (a span longer than 4 rows takes a pass of its own).
		ids := spanPrompt()
		for _, lay := range spanLayouts {
			n := 0
			for _, sp := range lay.spans {
				n += sp.Len
			}
			rows, err := s.mg.Gemma4BidirImageRowsForTest(ids, eModelFeatures(n, s.r.hidden), lay.spans)
			if err != nil {
				t.Fatal(err)
			}
			one, err := s.r.PrefillSpansLast(ctx, rows, 0, spanBlocks(lay.spans))
			if err != nil {
				t.Fatal(err)
			}
			one = append([]float32(nil), one...)
			restore := decoder.SetKnobForTest(s.mg, "GOINFER_PREFILL_CHUNK", "4")
			cut, err := s.r.PrefillSpansLast(ctx, rows, 0, spanBlocks(lay.spans))
			restore()
			if err != nil {
				t.Fatal(err)
			}
			if d, first := countDiff(one, cut); d != 0 {
				t.Errorf("%s %s: one pass and 4-row passes differ in %d logits (first %d)", dir, lay.name, d, first)
			} else {
				t.Logf("%s %-11s one pass == 4-row passes: bit-identical", dir, lay.name)
			}
		}
		// Non-vacuity: with the spans dropped the same rows prefill causally, and that must not pass.
		spanDefectForTest = spanDefectCausal
		causal := s.grade(t)
		spanDefectForTest = 0
		red := false
		for _, g := range causal {
			t.Logf("%s %-11s causal (spans dropped): cos %.6f argmax gpu=%d cpu=%d", dir, g.layout, g.cos, g.argGPU, g.argC)
			red = red || !g.pass()
		}
		if !red {
			t.Errorf("%s: a causal prefill passes the bar on every layout: the gate cannot see the mask (vacuous)", dir)
		}
	}
}

// TestGemma4BidirSpanPrefill_plantedDefects is G-S19b: each planted defect alone must turn G-S19a red on at least one
// layout of at least one fixture.
func TestGemma4BidirSpanPrefill_plantedDefects(t *testing.T) {
	rigs := make([]*spanRig, 0, len(spanFixtures))
	for _, dir := range spanFixtures {
		rigs = append(rigs, loadSpanRig(t, dir))
	}
	defects := []struct {
		name string
		set  func()
	}{
		{"mask left causal", func() { spanDefectForTest = spanDefectCausal }},
		{"span end one row short", func() { spanDefectForTest = spanDefectShortEnd }},
		{"no reach-back to the span start (Gemma 3's window rule)", func() { spanDefectForTest = spanDefectNoReachBack }},
		{"image rows scaled by the embed scale", func() { decoder.SetGemma4VLRowsDefectForTest(2) }},
	}
	for _, d := range defects {
		d.set()
		red := 0
		for _, s := range rigs {
			for _, g := range s.grade(t) {
				t.Logf("  [%s] %s %-11s cos %.6f argmax gpu=%d cpu=%d pass=%v", d.name, s.name, g.layout, g.cos, g.argGPU, g.argC, g.pass())
				if !g.pass() {
					red++
				}
			}
		}
		spanDefectForTest = 0
		decoder.SetGemma4VLRowsDefectForTest(0)
		if red == 0 {
			t.Errorf("G-S19b FAIL: planted defect %q stayed green on every layout", d.name)
		} else {
			t.Logf("G-S19b: %q red on %d layouts", d.name, red)
		}
	}
}
