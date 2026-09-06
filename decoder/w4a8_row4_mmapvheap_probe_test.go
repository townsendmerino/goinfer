package decoder

import (
	"math/rand"
	"os"
	"testing"
	"time"

	"github.com/townsendmerino/aikit/linalg"
)

// heapifyRow4 replaces wm's row4 bytes/scales with fresh heap allocations holding the
// SAME values, leaving canonical q4/q4s untouched. Correctness is unaffected --
// MatmulBTW4A8Into reads only q4Row4/q4Row4Scales when present, and those values are
// byte-identical to what mmap held, just backed by a new []byte the runtime allocated
// on the heap instead of a page from the mmap'd file.
func heapifyRow4(wm *linalg.WeightMat) bool {
	q4, q4s, group, ok := wm.Int4()
	if !ok {
		return false
	}
	row4, row4Scales, ok := wm.Int4Row4()
	if !ok {
		return false
	}
	heapRow4 := append([]byte(nil), row4...)
	heapScales := append([]float32(nil), row4Scales...)
	*wm = linalg.WrapInt4Row4(q4, q4s, wm.Rows(), wm.Cols(), group, heapRow4, heapScales)
	return true
}

// TestRow4_mmapVsHeapResident is the discriminating experiment: same kernel, same
// bytes, one variable (memory source), on a HANDFUL of experts -- not the whole
// model. The first version of this test heapified all ~7680 expert tensors at once
// (~14 GB of fresh heap on this 16 GB box) and drove the machine into severe swap
// thrashing (15 of 16 GB swap used) before being killed; this version times raw
// MatmulBTW4A8Into calls directly on a small, fixed set of experts instead of running
// full model decode, keeping the extra heap footprint under 100 MB.
//
// The resident-path 1.6-1.75x figure (docs/task-w4a8-neon-bandwidth.md) was measured
// against heap-resident repacked bytes (RepackInt4Row4, the GGUF/safetensors streaming
// loaders). The kind-4 .giw path's row4 bytes are mmap-aliased even when NOT paged (no
// StreamWeights) -- Load() always mmaps a .giw file read-only regardless of streaming
// mode. If mmap-resident calls are slower than heap-resident calls on the identical
// bytes, the quiet-machine gemma4 gap (docs/task-zeno-compare.md's "Quiet-machine
// re-measure") is memory-source mechanics (TLB/page-fault residue on mapped pages),
// not paging-machinery overhead.
func TestRow4_mmapVsHeapResident(t *testing.T) {
	requireHeavyModel(t)
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
			if _, _, ok := wm.Int4Row4(); !ok {
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
