//go:build cuda && goinfer_testhooks

package cuda

import (
	"math"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestResidentDrafter_fuseContextReleasesScratch is M-21's own gate (docs/audit-2026-09-10.md):
// FuseContext's four per-call scratch buffers (ctxIn/ctxQ/ctxSc/ctxFuse) grew on every call whose
// row count exceeded the previous high-water mark, abandoning the old buffers to the device
// ledger instead of releasing them — the C-12 (audit-2026-09-02) shape, which C-12's own fix
// reached only logitsB, not this drafter.
//
// A DOUBLING ramp (4, 8, 16, 32, 64 rows), not a fixed size repeated: each call must exceed the
// previous ctxCap to force a grow-and-release cycle every time (a repeated size only reuses the
// existing buffers — the bug this test is for never fires at all on that shape). If leaking, free
// VRAM after the ramp reflects the SUM of all five allocations (≈2x the largest alone, for a
// doubling series); fixed, it reflects only the largest (current) allocation.
//
//	GOINFER_HEAVY_TESTS=1 GOINFER_CUDA_MODEL=$HOME/models/qwen3-4b \
//	  go test -tags 'cuda goinfer_testhooks' -run TestResidentDrafter_fuseContextReleasesScratch -v
func TestResidentDrafter_fuseContextReleasesScratch(t *testing.T) {
	requireHeavyModel(t)
	tgt := os.Getenv("GOINFER_CUDA_MODEL")
	if tgt == "" {
		tgt = os.ExpandEnv("$HOME/models/qwen3-4b")
	}
	ddir := decoder.AssetPathForTest(t, "GOINFER_DFLASH_F32")
	if _, err := os.Stat(tgt); err != nil {
		t.Skipf("no target at %s", tgt)
	}
	mc, err := decoder.Load(tgt, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load target: %v", err)
	}
	defer mc.Close()
	r, ok := mc.ResidentForwardForTest().(*cudaResident)
	if !ok {
		t.Fatal("resident did not engage")
	}
	dr, err := decoder.LoadDFlashDrafter(ddir)
	if err != nil {
		t.Fatalf("load drafter: %v", err)
	}
	defer dr.Close()

	var w decoder.BlockDrafterWeights = dr
	rd, err := r.AttachDrafter(w)
	if err != nil {
		t.Fatalf("AttachDrafter: %v", err)
	}
	K := w.DrafterFC().Cols()
	hidden := w.DrafterGeometry().Hidden

	fillCtx := func(n int) [][]float32 {
		ctx := make([][]float32, n)
		for i := range ctx {
			ctx[i] = make([]float32, K)
			for j := range ctx[i] {
				ctx[i][j] = float32(math.Sin(float64(i*K+j)*0.001)) * 0.4
			}
		}
		return ctx
	}

	// Warm up at the smallest size so the FIRST measured allocation is a real grow, same as every
	// later one in the ramp — avoids conflating a one-time driver/context cost with the pattern
	// under test.
	if _, err := rd.FuseContext(fillCtx(4)); err != nil {
		t.Fatalf("warmup FuseContext: %v", err)
	}
	free0, _, err := r.dev.Context().MemInfo()
	if err != nil {
		t.Skipf("MemInfo: %v", err)
	}

	sizes := []int{8, 16, 32, 64}
	for _, n := range sizes {
		if _, err := rd.FuseContext(fillCtx(n)); err != nil {
			t.Fatalf("FuseContext(n=%d): %v", n, err)
		}
	}
	free1, _, err := r.dev.Context().MemInfo()
	if err != nil {
		t.Skipf("MemInfo: %v", err)
	}
	drop := int64(free0) - int64(free1)

	// Expected steady-state cost: the largest (last) call's own four buffers, computed from the
	// same sizing formula FuseContext itself uses — ctxIn f32[n*K], ctxQ int32[n*K/4] (packed
	// int8), ctxSc f32[n], ctxFuse f32[n*hidden].
	n := sizes[len(sizes)-1]
	singleCallBytes := int64(n*K)*4 + int64(n*(K/4))*4 + int64(n)*4 + int64(n*hidden)*4
	t.Logf("free VRAM before ramp: %d, after: %d (drop %d bytes; one call's own buffers ≈ %d bytes)",
		free0, free1, drop, singleCallBytes)
	// Generous multiple: a doubling ramp that leaked every prior size would sum to roughly 2x the
	// largest single allocation; 1.5x cleanly separates "released" (≈1x) from "leaked" (≈2x)
	// without being tight enough to flake on allocator rounding/fragmentation.
	if bound := singleCallBytes + singleCallBytes/2; drop > bound {
		t.Errorf("free VRAM dropped by %d bytes over the ramp, more than 1.5x one call's own buffers "+
			"(%d bytes, bound %d) — FuseContext's grown buffers are not being released", drop, singleCallBytes, bound)
	}
}
