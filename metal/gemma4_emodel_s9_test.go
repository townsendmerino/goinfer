//go:build darwin && goinfer_testhooks

package metal

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"os"
	"slices"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// G-S9a and G-S9b of docs/tasks/task-multimodal-support-2026-10.md (S9, registered and amended before this ran), on
// the tiny E-model: the layer-major prefill (prefill_emodel.go, through PrefillLast) against the sequential resident loop
// (Forward, one row at a time) on the same resident and rows. G-S9a: the last row's logits and the next 8 decode steps'
// logits from the prefilled cache are bit-identical, for a text-only prompt and one carrying a run of image rows (built
// by the decoder's own gemma4ResidentMediaRow: the feature, PAD in PLE), at the prompt's length and past one chunk
// (emodelChunk rows). G-S9b: each planted defect, alone, changes them.

const s9Steps = 8

// s9Run prefills rows (layer-major when lm, else sequentially) and decodes s9Steps tokens forced to `forced` (nil: its
// own argmax), returning the last prefill row's logits followed by each step's, and the tokens it fed.
func s9Run(t *testing.T, a *metalResident, mg *decoder.Model, rows [][]float32, lm bool, forced []int) ([][]float32, []int) {
	t.Helper()
	var out [][]float32
	var last []float32
	if lm {
		before := a.r.emodelLayerMajorRuns
		lg, err := a.PrefillLast(context.Background(), rows, 0)
		if err != nil {
			t.Fatalf("PrefillLast: %v", err)
		}
		if a.r.emodelLayerMajorRuns != before+1 {
			t.Fatal("PrefillLast did not take the E-model layer-major pass")
		}
		last = lg
	} else {
		for i, row := range rows {
			lg, err := a.Forward(row, i)
			if err != nil {
				t.Fatal(err)
			}
			last = lg
		}
	}
	out = append(out, slices.Clone(last))
	var fed []int
	for k := range s9Steps {
		next := s9Argmax(last)
		if forced != nil {
			next = forced[k]
		}
		fed = append(fed, next)
		lg, err := a.Forward(mg.EmbedResidentForTest(next), len(rows)+k)
		if err != nil {
			t.Fatal(err)
		}
		last = lg
		out = append(out, slices.Clone(lg))
	}
	return out, fed
}

func s9Argmax(l []float32) int {
	b := 0
	for i, v := range l {
		if v > l[b] {
			b = i
		}
	}
	return b
}

// s9Same reports whether two runs' logits are bit-identical, and the first differing step and the largest |diff|.
func s9Same(a, b [][]float32) (bool, int, float64) {
	first, md := -1, 0.0
	for s := range a {
		for i := range a[s] {
			if math.Float32bits(a[s][i]) != math.Float32bits(b[s][i]) {
				if first < 0 {
					first = s
				}
				md = math.Max(md, math.Abs(float64(a[s][i]-b[s][i])))
			}
		}
	}
	return first < 0, first, md
}

func TestGemma4EModel_layerMajorPrefill(t *testing.T) {
	if _, err := os.Stat(eModelDir); err != nil {
		t.Skipf("no fixture (%s)", eModelDir)
	}
	prompt := eModelPrompt(t)
	mg, err := decoder.Load(eModelDir, decoder.Options{Quant: "int4"})
	if err != nil {
		t.Fatal(err)
	}
	defer mg.Close()
	r, err := buildResident(mg)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	a := &metalResident{r: r, hidden: r.H}
	if !a.emodelLayerMajor() {
		t.Fatal("the tiny E-model does not take the layer-major pass")
	}
	H := mg.Config().HiddenDim
	rng := rand.New(rand.NewSource(91))
	build := func(n int, image bool) [][]float32 {
		rows := make([][]float32, n)
		for i := range rows {
			rows[i] = mg.EmbedResidentForTest(prompt[i%len(prompt)])
		}
		if image { // a run of image rows after the first two text rows
			for i := 2; i < min(n-2, 2+12); i++ {
				f := make([]float32, H)
				for j := range f {
					f[j] = float32(rng.NormFloat64())
				}
				rows[i] = mg.Gemma4ResidentMediaRowForTest(f)
			}
		}
		return rows
	}
	type tc struct {
		name string
		rows [][]float32
		ref  [][]float32
		toks []int
	}
	var cases []tc
	for _, n := range []int{len(prompt), emodelChunk + 37} {
		for _, image := range []bool{false, true} {
			rows := build(n, image)
			ref, toks := s9Run(t, a, mg, rows, false, nil)
			cases = append(cases, tc{fmt.Sprintf("%d rows, image %v", n, image), rows, ref, toks})
		}
	}
	// G-S9a
	for _, c := range cases {
		got, _ := s9Run(t, a, mg, c.rows, true, c.toks)
		same, step, md := s9Same(c.ref, got)
		fmt.Fprintf(os.Stderr, "[G-S9a] %-22s layer-major against sequential: bit-identical %v (first differing step %d, max |diff| %.3g)\n", c.name, same, step, md)
		if !same {
			t.Errorf("G-S9a %s: the layer-major pass differs from the sequential loop at step %d (max |diff| %g)", c.name, step, md)
		}
	}
	// G-S9b: each defect alone must change the result, on the longer image case (rows from both chunks).
	c := cases[len(cases)-1]
	for _, d := range []struct {
		name  string
		plant func()
	}{
		{"(1) each row's PLE inputs bound to the next row's", func() { emodelPlant.pleShift = true }},
		{"(2) the PLE term dropped", func() { emodelPlant.pleZero = true }},
		{"(3) every row one position late", func() { emodelPlant.posShift = true }},
		{"(4) rows encoded in reverse order within a layer", func() {
			emodelRowOrderHook = func(n int) []int {
				o := make([]int, n)
				for i := range o {
					o[i] = n - 1 - i
				}
				return o
			}
		}},
	} {
		d.plant()
		got, _ := s9Run(t, a, mg, c.rows, true, c.toks)
		emodelPlant.pleShift, emodelPlant.pleZero, emodelPlant.posShift = false, false, false
		emodelRowOrderHook = nil
		same, step, md := s9Same(c.ref, got)
		fmt.Fprintf(os.Stderr, "[G-S9b] %-50s first differing step %d, max |diff| %.3g\n", d.name, step, md)
		if same {
			t.Errorf("G-S9b: planted defect %s left the result bit-identical: the check cannot see it", d.name)
		}
	}
}
