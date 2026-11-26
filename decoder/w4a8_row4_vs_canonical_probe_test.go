//go:build arm64

package decoder

import (
	"math/rand"
	"os"
	"testing"
	"time"

	"github.com/townsendmerino/aikit/linalg"

	"github.com/townsendmerino/goinfer/internal/testgate"
)

// TestRow4_vsCanonical_gemma4Shapes is the follow-up the mmap-vs-heap result suggests: memory source is ruled out
// (TestRow4_mmapVsHeapResident), so if gemma4's paged decode is still slower with kind-4 than kind-3,
// the remaining candidate is that the row4 kernel is
// not faster than canonical on THESE SPECIFIC expert shapes (the original speedup may have been measured on different tensor dimensions). Runs the
// canonical kernel (MatmulBTW4A8F16Into) and the row4 one (the WeightMat's MatmulBTW4A8Into, which takes row4 at M=1) on the
// same real gemma4 expert bytes, same activation, same dst: isolates kernel choice as the only variable.
//
// It runs on no machine we have, so it is not coverage: it builds only on arm64, and it needs the 26B kind-4 .giw on a Mac
// that can hold it, which the 16 GB Mac cannot; nobara is amd64. A skip here vouches for nothing.
func TestRow4_vsCanonical_gemma4Shapes(t *testing.T) {
	testgate.RequireHeavy(t)
	kind4 := expandHome(t, envOr("GOINFER_GEMMA4_26B_GIW_ROW4", "~/models/gemma4-26b-int4-row4.giw"))
	if _, err := os.Stat(kind4); err != nil {
		t.Skipf("no kind-4 gemma4 .giw at %s: %v", kind4, err)
	}

	m, err := Load(kind4, Options{})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()

	type sample struct {
		wm                *linalg.WeightMat // canonical and row4: MatmulBTW4A8Into takes row4 at M=1
		q4                []byte
		q4s16             []uint16
		group, rows, cols int
		act, dstA, dstB   []float32
	}
	var gateUp, down []sample
	rng := rand.New(rand.NewSource(1))
	build := func(wm *linalg.WeightMat) (sample, bool) {
		q4, q4s16, group, ok := wm.Int4F16()
		if !ok {
			return sample{}, false
		}
		if _, _, ok := wm.Int4Row4F16(); !ok {
			return sample{}, false
		}
		act := make([]float32, wm.Cols())
		for j := range act {
			act[j] = rng.Float32()*2 - 1
		}
		return sample{
			wm: wm, q4: q4, q4s16: q4s16,
			group: group, rows: wm.Rows(), cols: wm.Cols(),
			act: act, dstA: make([]float32, wm.Rows()), dstB: make([]float32, wm.Rows()),
		}, true
	}
	for i := range m.w.Layers {
		gm := m.w.Layers[i].Gemma4MoE
		if gm == nil || len(gateUp) >= 24 {
			continue
		}
		for e := 0; e < len(gm.ExpertsGateUp) && len(gateUp) < 24; e += 200 {
			if s, ok := build(&gm.ExpertsGateUp[e]); ok {
				gateUp = append(gateUp, s)
			}
			if s, ok := build(&gm.ExpertsDown[e]); ok {
				down = append(down, s)
			}
		}
	}
	if len(gateUp) == 0 || len(down) == 0 {
		t.Fatal("no samples built")
	}
	t.Logf("gateUp shape: rows=%d cols=%d group=%d (%d samples)", gateUp[0].rows, gateUp[0].cols, gateUp[0].group, len(gateUp))
	t.Logf("down shape: rows=%d cols=%d group=%d (%d samples)", down[0].rows, down[0].cols, down[0].group, len(down))

	const reps = 4000
	var ws linalg.Workspace

	runCanon := func(samples []sample) time.Duration {
		start := time.Now()
		for r := range reps {
			s := samples[r%len(samples)]
			linalg.MatmulBTW4A8F16Into(&ws, s.act, s.q4, s.q4s16, s.dstA, 1, s.cols, s.rows, s.group)
		}
		return time.Since(start)
	}
	runRow4 := func(samples []sample) time.Duration {
		start := time.Now()
		for r := range reps {
			s := samples[r%len(samples)]
			s.wm.MatmulBTW4A8Into(&ws, s.act, s.dstB, 1)
		}
		return time.Since(start)
	}

	report := func(name string, samples []sample) {
		// Order-alternate to cancel warm-up bias: canon, row4, row4, canon -- average each pair.
		c1 := runCanon(samples)
		r1 := runRow4(samples)
		r2 := runRow4(samples)
		c2 := runCanon(samples)
		canonAvg := (c1 + c2) / 2
		row4Avg := (r1 + r2) / 2
		canonRate := float64(reps) / canonAvg.Seconds()
		row4Rate := float64(reps) / row4Avg.Seconds()
		delta := (row4Rate/canonRate - 1) * 100
		t.Logf("%s: canonical %v (%.0f calls/s, runs %v/%v) vs row4 %v (%.0f calls/s, runs %v/%v) -- row4 is %+.1f%% vs canonical",
			name, canonAvg, canonRate, c1, c2, row4Avg, row4Rate, r1, r2, delta)
	}

	report("gateUp (fused gate+up, gemma4-specific)", gateUp)
	report("down", down)
}
