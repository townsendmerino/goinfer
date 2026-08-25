package decoder

import (
	"math/rand"
	"os"
	"testing"
	"time"
	"unsafe"

	"github.com/townsendmerino/aikit/linalg"
)

// TestRow4_coldTouchLatency is the experiment the first two ruled-out hypotheses
// (mmap-vs-heap: +1.7% noise; row4-vs-canonical on gemma4 shapes: row4 is +57-67%
// FASTER, matching the original 1.6-1.75x claim) point to directly: both of those
// tests reused the SAME small set of already-resident experts thousands of times,
// so they measured warm, steady-state kernel speed -- never the real production path
// (expertPager.touch() -> WILLNEED -> matmul) on a genuinely cold, first-ever touch.
// Real decode touches ~240 DISTINCT experts per token under a real cache budget; this
// test replicates that shape directly: many distinct experts, each touched exactly
// once, through the real pager, comparing kind-3 (canonical dispatch) against kind-4
// (row4 dispatch) per-touch latency.
func TestRow4_coldTouchLatency(t *testing.T) {
	requireHeavyModel(t)
	kind3 := expandHome(t, envOr("GOINFER_GEMMA4_26B_GIW", "~/models/gemma4-26b-int4.giw"))
	kind4 := expandHome(t, envOr("GOINFER_GEMMA4_26B_GIW_ROW4", "~/models/gemma4-26b-int4-row4.giw"))
	if _, err := os.Stat(kind3); err != nil {
		t.Skipf("no kind-3 gemma4 .giw at %s: %v", kind3, err)
	}
	if _, err := os.Stat(kind4); err != nil {
		t.Skipf("no kind-4 gemma4 .giw at %s: %v", kind4, err)
	}

	const budget = 512 << 20 // small: keep almost nothing resident, force real touches
	const nExperts = 300     // distinct experts, each touched exactly once

	measure := func(path, label string) (totalTouch, totalMatmul time.Duration, n int) {
		m, err := Load(path, Options{StreamWeights: true, WeightCacheBytes: budget})
		if err != nil {
			t.Fatalf("%s: load: %v", label, err)
		}
		defer m.Close()
		if m.pager == nil {
			t.Fatalf("%s: no pager built", label)
		}

		rng := rand.New(rand.NewSource(2))
		var ws linalg.Workspace
		stride := len(m.w.Layers[0].gemma4moe.expertsGateUp) * len(m.w.Layers) / nExperts
		if stride < 1 {
			stride = 1
		}
		count := 0
		for li := 0; li < len(m.w.Layers) && count < nExperts; li++ {
			gm := m.w.Layers[li].gemma4moe
			if gm == nil {
				continue
			}
			for e := (li * 37) % len(gm.expertsGateUp); e < len(gm.expertsGateUp) && count < nExperts; e += stride {
				gu := &gm.expertsGateUp[e]
				dn := &gm.expertsDown[e]
				key := unsafe.Pointer(gu)

				actGU := make([]float32, gu.Cols())
				actDn := make([]float32, dn.Cols())
				for j := range actGU {
					actGU[j] = rng.Float32()*2 - 1
				}
				for j := range actDn {
					actDn[j] = rng.Float32()*2 - 1
				}
				dstGU := make([]float32, gu.Rows())
				dstDn := make([]float32, dn.Rows())

				t0 := time.Now()
				m.pager.touch(key)
				touchElapsed := time.Since(t0)

				t1 := time.Now()
				gu.MatmulBTW4A8Into(&ws, actGU, dstGU, 1)
				dn.MatmulBTW4A8Into(&ws, actDn, dstDn, 1)
				matmulElapsed := time.Since(t1)

				totalTouch += touchElapsed
				totalMatmul += matmulElapsed
				count++
			}
		}
		hits, misses, evictions := m.pager.stats()
		t.Logf("%s: %d touches, hits=%d misses=%d evictions=%d", label, count, hits, misses, evictions)
		return totalTouch, totalMatmul, count
	}

	k3Touch, k3Matmul, k3N := measure(kind3, "kind-3")
	k4Touch, k4Matmul, k4N := measure(kind4, "kind-4")

	k3Total := k3Touch + k3Matmul
	k4Total := k4Touch + k4Matmul
	k3PerTouch := k3Total / time.Duration(k3N)
	k4PerTouch := k4Total / time.Duration(k4N)
	delta := (float64(k4PerTouch)/float64(k3PerTouch) - 1) * 100

	t.Logf("kind-3: touch=%v matmul=%v total=%v (%v/touch avg over %d)", k3Touch, k3Matmul, k3Total, k3PerTouch, k3N)
	t.Logf("kind-4: touch=%v matmul=%v total=%v (%v/touch avg over %d)", k4Touch, k4Matmul, k4Total, k4PerTouch, k4N)
	t.Logf("kind-4 vs kind-3, per-touch (real pager path, first-ever touch): %+.1f%%", delta)
}
