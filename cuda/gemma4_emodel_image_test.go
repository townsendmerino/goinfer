//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"math"
	"strconv"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// S9 on CUDA, part B: the Gemma 4 E-model image turn prefilled on the resident in one batched pass, from rows the decoder builds. The plan, the gates and the
// bars: docs/tasks/task-multimodal-support-2026-10.md ("S9 on CUDA, part B").

// eModelImageLayouts are the three places the image block can sit in the 18-position prompt (the golden's prompt plus its continuation): the start, the
// middle and the end, 6 soft tokens each, so the PLE tails of image rows meet text rows on both sides and the shared layers attend across the boundary.
var eModelImageLayouts = []struct {
	name        string
	pos, length int
}{{"image-first", 0, 6}, {"image-middle", 5, 6}, {"image-last", 12, 6}}

// eModelFeatures returns n synthetic projected features of width hidden: deterministic, zero-mean, with a magnitude comparable to a scaled token embedding so the
// defect that scales a feature by the embed scale moves the logits.
func eModelFeatures(n, hidden int) []float32 {
	f := make([]float32, n*hidden)
	var s uint32 = 97531
	for i := range f {
		s = s*1664525 + 1013904223
		f[i] = float32(int32(s>>16)%2001-1000) / 1000 * 3
	}
	return f
}

// eModelImageGrade is one layout's reading: the resident's batched last-position logits on the built rows against the CPU int4 forward (the image turn's
// reference, prefillLogitsGemma4VL) and against the CPU f32 forward (the envelope), under G1c's rule.
type eModelImageGrade struct {
	name                     string
	cosCPU4, cosEnv          float64
	argCuda, argCPU4         int
	gap                      float64
	seqIdentical, nonVacuous bool
}

func (g eModelImageGrade) pass() (bool, string) {
	if g.argCuda != g.argCPU4 && g.gap > 0.03 {
		return false, "argmax diverges by more than 3% (not a near-tie)"
	}
	if g.cosCPU4 < g.cosEnv {
		return false, "CUDA-vs-CPUint4 cosine below the CPU's own int4-vs-f32 cosine"
	}
	return true, ""
}

func runEModelImageG1q(t *testing.T) []eModelImageGrade {
	t.Helper()
	mg, r := loadEModelResident(t)
	mc4, err := decoder.Load(eModelDir, decoder.Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load (cpu int4): %v", err)
	}
	defer mc4.Close()
	mcF, err := decoder.Load(eModelDir, decoder.Options{Quant: "f32"})
	if err != nil {
		t.Fatalf("load (cpu f32): %v", err)
	}
	defer mcF.Close()
	ids := eModelPrompt(t)
	hidden := r.hidden
	ctx := context.Background()
	var out []eModelImageGrade
	for _, lay := range eModelImageLayouts {
		feats := eModelFeatures(lay.length, hidden)
		rows, err := mg.Gemma4EModelImageRowsForTest(ids, feats, lay.pos, lay.length)
		if err != nil {
			t.Fatalf("%s: rows: %v", lay.name, err)
		}
		// The resident, batched. Non-vacuity as in G1p: passPromptLen is written only by prefillCore.
		r.Reset()
		r.passPromptLen = 0
		batched, err := r.PrefillLast(ctx, rows, 0)
		if err != nil {
			t.Fatalf("%s: PrefillLast: %v", lay.name, err)
		}
		batched = append([]float32(nil), batched...)
		g := eModelImageGrade{name: lay.name, nonVacuous: r.passPromptLen == len(ids)}
		// The same rows, one token at a time on the same resident: the batched pass must be bit-identical (the rows are the variable under test, not the pass).
		seq := seqLogits(t, r, rows)
		d, _ := countDiff(batched, seq[len(ids)-1])
		g.seqIdentical = d == 0
		// The CPU references.
		cpu4, err := mc4.PrefillLogitsGemma4VLForTest(ctx, ids, feats, lay.pos, lay.length, mc4.NewCache(len(ids)+1))
		if err != nil {
			t.Fatalf("%s: cpu int4 reference: %v", lay.name, err)
		}
		cpuF, err := mcF.PrefillLogitsGemma4VLForTest(ctx, ids, feats, lay.pos, lay.length, mcF.NewCache(len(ids)+1))
		if err != nil {
			t.Fatalf("%s: cpu f32 reference: %v", lay.name, err)
		}
		g.cosCPU4, _ = cosMaxAbs(cpu4, batched)
		g.cosEnv, _ = cosMaxAbs(cpuF, cpu4)
		g.argCuda, g.argCPU4 = argmaxF(batched), argmaxF(cpu4)
		if g.argCuda != g.argCPU4 {
			g.gap = (float64(cpu4[g.argCPU4]) - float64(cpu4[g.argCuda])) / (math.Abs(float64(cpu4[g.argCPU4])) + 1e-30)
		}
		t.Logf("  %-13s CUDA-batched-vs-CPUint4 %.6f | CPUint4-vs-f32 %.6f | argmax cpu=%d cuda=%d gap=%.2f%% | batched==sequential: %v", lay.name, g.cosCPU4, g.cosEnv, g.argCPU4, g.argCuda, g.gap*100, g.seqIdentical)
		out = append(out, g)
	}
	return out
}

// TestGemma4EModelImage_batchedPrefillMatchesCPU is G1q: for each layout the image turn's batched resident logits against the CPU forward under G1c's rule,
// and bit-identical to the same rows run sequentially on the resident.
func TestGemma4EModelImage_batchedPrefillMatchesCPU(t *testing.T) {
	for _, g := range runEModelImageG1q(t) {
		if !g.nonVacuous {
			t.Errorf("%s: passPromptLen was not set: the batched pass did not run (vacuous)", g.name)
		}
		if !g.seqIdentical {
			t.Errorf("%s: batched last-row logits differ from the sequential per-row forward on the same rows", g.name)
		}
		if ok, why := g.pass(); !ok {
			t.Errorf("G1q FAIL %s: %s (cos %.6f, envelope %.6f)", g.name, why, g.cosCPU4, g.cosEnv)
		}
	}
}

// TestGemma4EModelImage_plantedDefects is G2q: each planted defect in the decoder's image-row builder, alone, must turn G1q red on at least one layout.
func TestGemma4EModelImage_plantedDefects(t *testing.T) {
	defects := []struct {
		name string
		id   int
	}{
		{"(1) image tail built from the placeholder id, not the pad id", 1},
		{"(2) image feature scaled by the embed scale", 2},
		{"(3) image tail zero", 3},
		{"(4) image positions built as text rows", 4},
	}
	for _, d := range defects {
		t.Run(d.name, func(t *testing.T) {
			decoder.SetGemma4VLRowsDefectForTest(d.id)
			defer decoder.SetGemma4VLRowsDefectForTest(0)
			red := 0
			for _, g := range runEModelImageG1q(t) {
				if ok, why := g.pass(); !ok {
					red++
					t.Logf("G2q %s: %s RED (%s)", d.name, g.name, why)
				}
			}
			if red == 0 {
				t.Errorf("planted defect %s left G1q green on every layout — the fixture cannot see it", d.name)
			}
		})
	}
}

// TestGemma4EModelImage_generateTakesTheResidentPath drives GenerateGemma4VL end to end on the tiny E-model, three turns on one CUDA model (image, the same
// image again, then plain text): the image turn must report that its prefill ran resident, decode resident, and generate the CPU model's greedy tokens; with
// batched prefill switched off (GOINFER_BATCHED_PREFILL=0) the very same turn must take today's CPU prefill + upload bridge (ImgPrefillResident false,
// DecodeResident true) and still match, which is the fall-through every decline relies on.
func TestGemma4EModelImage_generateTakesTheResidentPath(t *testing.T) {
	mg, r := loadEModelResident(t)
	mc, err := decoder.Load(eModelDir, decoder.Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load (cpu): %v", err)
	}
	defer mc.Close()
	ids := eModelPrompt(t)
	lay := eModelImageLayouts[1]
	feats := eModelFeatures(lay.length, r.hidden)
	turn := func(m *decoder.Model, withImage bool) ([]int, *decoder.Generation) {
		t.Helper()
		var ch <-chan int
		var g *decoder.Generation
		if withImage {
			ch, g = m.GenerateGemma4VL(context.Background(), ids, lay.pos, lay.length, 42, func() ([]float32, error) { return feats, nil }, 6, decoder.SamplingParams{})
		} else {
			ch, g = m.Generate(context.Background(), ids, 6, decoder.SamplingParams{})
		}
		var out []int
		for id := range ch {
			out = append(out, id)
		}
		if err := g.Err(); err != nil {
			t.Fatalf("generate: %v", err)
		}
		return out, g
	}
	same := func(name string, got, want []int) {
		t.Helper()
		if len(got) != len(want) {
			t.Fatalf("%s: %d tokens, want %d", name, len(got), len(want))
		}
		for i := range got {
			if got[i] != want[i] {
				t.Errorf("%s: token %d = %d, the CPU model generated %d", name, i, got[i], want[i])
			}
		}
	}
	wantImg, gc := turn(mc, true)
	if gc.ImgPrefillResident || gc.DecodeResident {
		t.Fatalf("the CPU model reports a resident path (%v, %v)", gc.ImgPrefillResident, gc.DecodeResident)
	}
	wantTxt, _ := turn(mc, false)

	for i := 1; i <= 2; i++ {
		got, g := turn(mg, true)
		if !g.ImgPrefillResident || !g.DecodeResident {
			t.Errorf("image turn %d: ImgPrefillResident=%v DecodeResident=%v, want both true", i, g.ImgPrefillResident, g.DecodeResident)
		}
		same("image turn "+strconv.Itoa(i), got, wantImg)
	}
	got, _ := turn(mg, false)
	same("text turn after the image turns", got, wantTxt)

	decoder.SetKnobEnvForTest(t, mg, "GOINFER_BATCHED_PREFILL", "0")
	got, g := turn(mg, true)
	if g.ImgPrefillResident || !g.DecodeResident {
		t.Errorf("batched prefill off: ImgPrefillResident=%v DecodeResident=%v, want false and true (the CPU prefill + upload bridge)", g.ImgPrefillResident, g.DecodeResident)
	}
	same("image turn, batched prefill off", got, wantImg)
}
