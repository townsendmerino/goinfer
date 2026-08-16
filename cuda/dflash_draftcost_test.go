//go:build cuda && goinfer_testhooks

package cuda

import (
	"os"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestDFlashDraftCostProbe measures the one term P10's gate-3 projection could not settle:
// what a RESIDENT DFlash trunk would cost per block (docs/spec/08).
//
// It does it WITHOUT building the trunk, on a structural fact worth stating plainly: the
// DFlash drafter is exactly FIVE LAYERS OF THE TARGET'S OWN LAYER SHAPE plus `fc`. Same
// hidden 2560, same 32/8 GQA at head_dim 128, same 9728 SwiGLU — 5 × 100.9 M + 32.8 M =
// 537.4 M, which matches the checkpoint's tensor count to the digit. So the resident runner
// ALREADY executes the drafter's per-layer work 36 times per token; the drafter is 5/36ths
// of it, and the cost can be read off the target instead of modelled.
//
// The probe isolates the LM head (launchToken's `head` flag), because the head is 389 M of
// the target's 4.02 B and the drafter has none — it borrows the target's, and that cost is
// already inside the verify. Attributing head time to the drafter would overstate it.
//
// WHAT THIS IS NOT: a claim that the trunk will hit this number. It excludes the drafter's
// non-causal attention over [ctx‖block], which is the one part with no counterpart in the
// target's per-token path, and it assumes the same kernels. It is a floor with a named
// omission, not a prediction.
//
//	GOINFER_HEAVY_TESTS=1 GOINFER_CUDA_MODEL=$HOME/models/qwen3-4b \
//	  go test -tags 'cuda goinfer_testhooks' -run TestDFlashDraftCostProbe -v
func TestDFlashDraftCostProbe(t *testing.T) {
	requireHeavyModel(t)
	path := os.Getenv("GOINFER_CUDA_MODEL")
	if path == "" {
		path = os.ExpandEnv("$HOME/models/qwen3-4b")
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no model at %s", path)
	}
	mc, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer mc.Close()
	rf := mc.ResidentForwardForTest()
	if rf == nil {
		t.Fatal("resident did not engage")
	}
	r, ok := rf.(*cudaResident)
	if !ok {
		t.Fatalf("resident is %T", rf)
	}
	_, _, _, _, _, _, vocab := mc.Dims()
	emb := mc.EmbedResidentForTest(1234 % (vocab - 1))

	const depth = 1024
	warm := make([][]float32, depth)
	for i := range warm {
		warm[i] = mc.EmbedResidentForTest((i*2654435761 + 1) % (vocab - 1))
	}
	if _, e := r.PrefillLast(warm, 0); e != nil {
		t.Fatalf("warm: %v", e)
	}

	// Time on the executor thread; best-of-N to shed scheduler noise, as the ceiling probe does.
	timeIt := func(head bool) time.Duration {
		best := time.Hour
		for range 7 {
			t0 := time.Now()
			err := r.do(func() error {
				if e := r.launchToken(emb, depth, head); e != nil {
					return e
				}
				return r.stream.Sync()
			})
			if err != nil {
				t.Fatalf("launchToken(head=%v): %v", head, err)
			}
			if d := time.Since(t0); d < best {
				best = d
			}
		}
		return best
	}
	withHead := timeIt(true)
	noHead := timeIt(false)
	msOf := func(d time.Duration) float64 { return float64(d.Microseconds()) / 1000 }

	nL := r.nLayers
	headMs := msOf(withHead) - msOf(noHead)
	perLayer := msOf(noHead) / float64(nL)
	t.Logf("M=1 @depth %d: with head %.3f ms | layers only %.3f ms | LM head %.3f ms (%.0f%%)",
		depth, msOf(withHead), msOf(noHead), headMs, 100*headMs/msOf(withHead))
	t.Logf("per-layer (M=1): %.4f ms over %d layers", perLayer, nL)

	// The drafter is 5 of those layers plus fc (a 2560x12800 projection, ~0.33 of one layer's
	// weight bytes). At M=1 that is the weight-read floor; the block runs at M=16, where the
	// weights are read ONCE for all 16 rows — the same amortization the verify enjoys.
	const draftLayers = 5
	draftM1 := float64(draftLayers)*perLayer + 0.33*perLayer
	t.Logf("=> drafter weight-read floor (M=1-equivalent): %.2f ms  [5 layers + fc, no attention]", draftM1)
	t.Logf("   for scale: this box's M=16 batched verify is 46.4 ms and M=1 decode 11.1 ms")
	t.Logf("   NOTE excludes the drafter's non-causal attention over [ctx||block] — a floor, not a prediction")
}
