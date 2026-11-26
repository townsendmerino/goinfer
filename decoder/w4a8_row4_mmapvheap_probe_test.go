package decoder

import (
	"math/rand"
	"os"
	"slices"
	"testing"
	"time"

	"github.com/townsendmerino/aikit/linalg"

	"github.com/townsendmerino/goinfer/internal/testgate"
)

// heapifyRow4 replaces wm with a row4-only WeightMat over fresh heap copies of its row4
// bytes/scales: the SAME values, backed by memory the runtime allocated instead of a page
// of the mmap'd file. Correctness is unaffected -- MatmulBTW4A8Into reads only
// q4Row4/q4Row4Scales when present, so dropping the canonical bytes changes nothing the
// timed call reads.
func heapifyRow4(wm *linalg.WeightMat) bool {
	_, _, group, ok := wm.Int4F16()
	if !ok {
		return false
	}
	row4, row4Scales, ok := wm.Int4Row4F16()
	if !ok {
		return false
	}
	heap, ok := linalg.WrapInt4(slices.Clone(row4), slices.Clone(row4Scales), wm.Rows(), wm.Cols(), group, linalg.LayoutRow4)
	if !ok {
		return false
	}
	*wm = heap
	return true
}

// TestRow4_mmapVsHeapResident is the discriminating experiment: same kernel, same bytes, one variable (memory source), on
// a HANDFUL of experts, not the whole model. Heapifying every expert tensor (~14 GB of fresh heap) thrashes swap on a
// 16 GB box, so this times raw MatmulBTW4A8Into calls directly on a small, fixed set of experts, keeping the extra heap
// footprint under 100 MB.
//
// The resident-path row4 speedup was measured against heap-resident
// repacked bytes (RepackInt4Row4, the GGUF/safetensors streaming loaders). The kind-4 .giw path's row4 bytes are
// mmap-aliased even when NOT paged (no StreamWeights): Load() always mmaps a .giw file read-only. If mmap-resident calls
// are slower than heap-resident calls on the identical bytes, the quiet-machine gemma4 gap
// is memory-source mechanics (TLB/page-fault residue
// on mapped pages), not paging-machinery overhead.
func TestRow4_mmapVsHeapResident(t *testing.T) {
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

	// Collect a handful of expert WeightMats spread across layers -- enough to average
	// out per-shape variance, few enough that heapifying them costs single-digit MB.
	const wantSamples = 24
	type sample struct {
		wm  *linalg.WeightMat
		act []float32
		dst []float32
	}
	var samples []sample
	rng := rand.New(rand.NewSource(1))
	for i := range m.w.Layers {
		gm := m.w.Layers[i].gemma4moe
		if gm == nil || len(samples) >= wantSamples {
			continue
		}
		for e := 0; e < len(gm.expertsGateUp) && len(samples) < wantSamples; e += 200 { // sparse spread, not every expert
			wm := &gm.expertsGateUp[e]
			if _, _, ok := int4Row4F32(wm); !ok {
				continue
			}
			act := make([]float32, wm.Cols())
			for j := range act {
				act[j] = rng.Float32()*2 - 1
			}
			samples = append(samples, sample{wm: wm, act: act, dst: make([]float32, wm.Rows())})
		}
	}
	if len(samples) == 0 {
		t.Fatal("no row4-eligible expert samples found")
	}
	t.Logf("sampled %d experts across layers", len(samples))

	const reps = 4000
	var ws linalg.Workspace

	timeRuns := func() time.Duration {
		start := time.Now()
		for r := range reps {
			s := samples[r%len(samples)]
			s.wm.MatmulBTW4A8Into(&ws, s.act, s.dst, 1)
		}
		return time.Since(start)
	}

	mmapElapsed := timeRuns()
	mmapWarmElapsed := timeRuns() // warm-up control: same mmap-backed calls, repeated

	heapified := 0
	for i := range samples {
		if heapifyRow4(samples[i].wm) {
			heapified++
		}
	}
	if heapified == 0 {
		t.Fatal("heapifyRow4 succeeded on zero samples")
	}
	heapElapsed := timeRuns()

	mmapRate := float64(reps) / mmapElapsed.Seconds()
	mmapWarmRate := float64(reps) / mmapWarmElapsed.Seconds()
	heapRate := float64(reps) / heapElapsed.Seconds()
	warmupEffect := (mmapWarmRate/mmapRate - 1) * 100
	adjustedDelta := (heapRate/mmapWarmRate - 1) * 100

	t.Logf("heapified %d/%d samples", heapified, len(samples))
	t.Logf("mmap-resident (1st, cold): %v (%.0f calls/s)", mmapElapsed, mmapRate)
	t.Logf("mmap-resident (2nd, warm-up control): %v (%.0f calls/s) -- %+.1f%% vs 1st from warm-up alone", mmapWarmElapsed, mmapWarmRate, warmupEffect)
	t.Logf("heap-resident (3rd): %v (%.0f calls/s)", heapElapsed, heapRate)
	t.Logf("heap vs 2nd mmap (warm-up adjusted, the real signal): %+.1f%% -- %s", adjustedDelta,
		map[bool]string{true: "memory-source mechanics (TLB/page-fault residue) -- confirmed", false: "NOT memory-source -- look at paging machinery instead"}[adjustedDelta > 15])
}
