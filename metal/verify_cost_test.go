//go:build darwin && goinfer_testhooks

package metal

import (
	"context"
	"fmt"
	"os"
	"slices"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestVerifyCurveFrom pins E-P06's curve arithmetic: the measured 2/4/8-row points in token units, 3 and 5-7
// interpolated, 9 an 8-row step plus one token (stepVerifyCost's shape), and the refusals.
func TestVerifyCurveFrom(t *testing.T) {
	c, why := verifyCurveFrom(10, 15.9, 17.1, 17.6) // the 1.5B at depth 128 (stepVerifyCost's comment)
	if why != "" {
		t.Fatalf("refused a rising curve: %s", why)
	}
	want := []float64{0, 1, 1.59, 1.65, 1.71, 1.7225, 1.735, 1.7475, 1.76, 2.76}
	if len(c) != len(stepVerifyCost) {
		t.Fatalf("curve has %d entries, stepVerifyCost %d", len(c), len(stepVerifyCost))
	}
	for i := range want {
		if d := c[i] - want[i]; d > 1e-9 || d < -1e-9 {
			t.Fatalf("curve[%d] = %v, want %v (curve %v)", i, c[i], want[i], c)
		}
	}
	for _, bad := range [][4]float64{{10, 9, 12, 14}, {10, 12, 11, 14}, {10, 12, 13, 12.5}, {0, 12, 13, 14}, {10, -1, 13, 14}} {
		if _, why := verifyCurveFrom(bad[0], bad[1], bad[2], bad[3]); why == "" {
			t.Fatalf("accepted times %v", bad)
		}
	}
}

// TestVerifyCost_measuredAtLoadLossless is E-P06 on a resident: a load with Options.SpecNgram measures the curve
// (calibrateVerifyCost) or keeps stepVerifyCost when the measurement is refused, and either way n-gram speculation's
// tokens equal plain greedy's, as do plain greedy's own with and without the calibration (the K/V it writes at the
// calibration depth must never be read). A load without SpecNgram reports stepVerifyCost.
func TestVerifyCost_measuredAtLoadLossless(t *testing.T) {
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	path := writeMC3Fixture(t, 4096)
	prompt := make([]int, 48)
	for i := range prompt {
		prompt[i] = 100 + i*37
	}
	load := func(spec bool) (*decoder.Model, *metalResident) {
		m, err := decoder.Load(path, decoder.Options{Backend: "metal", Quant: "int4", ResidentContext: 1024, ResidentKVSlots: 2, SpecNgram: spec})
		if err != nil {
			t.Fatalf("load: %v", err)
		}
		a, ok := m.ResidentForwardForTest().(*metalResident)
		if !ok {
			m.Close()
			t.Fatalf("no metal resident: %s", m.ResidentDecline())
		}
		return m, a
	}
	var drafted, accepted int
	var ref []int
	gen := func(m *decoder.Model, prompt []int, spec bool) []int {
		var ch <-chan int
		var g *decoder.Generation
		var err error
		if spec {
			// The drafter proposes greedy's own continuation with every 5th token wrong, so rounds both accept and
			// reject: random weights repeat no n-gram, and an n-gram drafter here proposes nothing, which would compare
			// plain decode with itself (as a first version of this test did).
			if ch, g, err = m.GenerateNgramSpeculativeAdaptive(context.Background(), prompt, 64, &oracleDrafter{ref: ref, base: len(prompt), v: mfVocab},
				&decoder.AdaptiveDepth{MaxDraft: 8}, decoder.SamplingParams{}); err != nil {
				t.Fatalf("spec: %v", err)
			}
		} else {
			ch, g = m.Generate(context.Background(), prompt, 64, decoder.SamplingParams{})
		}
		var ids []int
		for id := range ch {
			ids = append(ids, id)
		}
		if err := g.Err(); err != nil {
			t.Fatalf("generate: %v", err)
		}
		if spec && g.Spec != nil {
			drafted, accepted = g.Spec.Drafted, g.Spec.Accepted
		}
		return ids
	}
	plain, a0 := load(false)
	if a0.r.verifyCost != nil || !slices.Equal(a0.VerifyCost(), stepVerifyCost) {
		t.Fatalf("a load without SpecNgram measured a curve (%v) or reports %v, not stepVerifyCost", a0.r.verifyCost, a0.VerifyCost())
	}
	ref = gen(plain, prompt, false)
	plain.Close()

	m, a := load(true)
	defer m.Close()
	cost := a.VerifyCost()
	fmt.Fprintf(os.Stderr, "[verify-cost] fixture with SpecNgram: measured %v, reports %v\n", a.r.verifyCost != nil, cost)
	if a.r.verifyCost != nil && !slices.Equal(cost, a.r.verifyCost) {
		t.Fatalf("VerifyCost reports %v, not the measured %v", cost, a.r.verifyCost)
	}
	if a.r.verifyCost == nil && !slices.Equal(cost, stepVerifyCost) {
		t.Fatalf("an unmeasured curve reports %v, not stepVerifyCost", cost)
	}
	if got := gen(m, prompt, false); !slices.Equal(got, ref) {
		t.Fatalf("greedy after the calibration differs from greedy without it:\n%v\n%v", got, ref)
	}
	if got := gen(m, prompt, true); !slices.Equal(got, ref) {
		t.Fatalf("speculation with the model's own curve differs from plain greedy:\n%v\n%v", got, ref)
	}
	fmt.Fprintf(os.Stderr, "[verify-cost] speculation drafted %d tokens, accepted %d\n", drafted, accepted)
	if drafted == 0 || accepted == 0 || accepted == drafted {
		t.Fatalf("speculation drafted %d and accepted %d: the identity above needs rounds that accept and rounds that reject", drafted, accepted)
	}
}

// oracleDrafter proposes the reference greedy continuation ref of a base-token prompt, with every 5th proposed token
// replaced by its neighbour in the vocabulary, so a verify both accepts and rejects.
type oracleDrafter struct {
	ref     []int
	base, v int
	buf     []int
}

func (d *oracleDrafter) Draft(ctx []int, k int) []int {
	j := len(ctx) - d.base // ctx is the prompt and the tokens emitted so far, ending at the current one
	if j < 0 || j >= len(d.ref) {
		return nil
	}
	d.buf = d.buf[:0]
	for i := j; i < min(j+k, len(d.ref)); i++ {
		id := d.ref[i]
		if i%5 == 4 {
			id = (id + 1) % d.v
		}
		d.buf = append(d.buf, id)
	}
	return d.buf
}
