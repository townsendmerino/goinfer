//go:build realckpt

package decoder

import (
	"fmt"
	"os"
	"testing"
	"time"
)

// How the DFlash draft cost scales with CONTEXT LENGTH — the term gate 3's projection never
// priced, and the one increment 4 is most exposed on.
//
// WHY THIS EXISTS. The 6.6 ms draft in docs/spec/08 is derived by scaling the TARGET's measured
// per-layer GPU cost across the drafter's 5 layers. `cuda/dflash_draftcost_test.go` says plainly
// what that omits: "the drafter's non-causal attention over [ctx‖block], which is the one part
// with no counterpart in the target's per-token path... a floor with a named omission, not a
// prediction." This measures the omission's SHAPE.
//
// There are two draft paths and they differ asymptotically per round:
//
//	DraftBlock(fused, block)       NewContext + ExtendContext over the WHOLE fused context,
//	                               then draft.                                     O(ctx)
//	DraftBlockCtx(ctx, block)      caller keeps the context; only the newly accepted rows are
//	                               projected in.                                   O(new)
//
// The acceptance harness calls the first — correct for numerics (identical output, and
// acceptance is all it measures) but it rebuilds every round. A serving path that did the same
// would pay a full drafter-prefill of the context on every block, which at realistic context
// lengths dwarfs the block work the 6.6 ms floor accounts for.
//
// What this reports, per context length: the rebuild path, the incremental path, and the ratio.
// The ratio is the architectural finding and transfers to any backend; the absolute CPU
// milliseconds do not.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags realckpt ./decoder/ -run TestDFlashDraftScaling -v
func TestDFlashDraftScaling(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("heavy: set GOINFER_HEAVY_TESTS=1")
	}
	ddir := envOr(os.Getenv("GOINFER_DFLASH_DRAFTER"), func() string { return assetPath(t, "GOINFER_DFLASH_F32") })
	d, err := LoadDFlashDrafter(ddir)
	if err != nil {
		t.Fatalf("LoadDFlashDrafter: %v", err)
	}
	defer d.Close()
	be := &cpuBackend{}

	B := d.BlockSize()
	block := make([][]float32, B)
	for i := range block {
		block[i] = make([]float32, d.hidden)
		for j := range block[i] {
			block[i][j] = float32((i*3+j)%11) / 11
		}
	}
	mkctx := func(n int) [][]float32 {
		rows := make([][]float32, n)
		for i := range rows {
			rows[i] = make([]float32, d.fc.Cols())
			for j := range rows[i] {
				rows[i][j] = float32((i*7+j)%13) / 13
			}
		}
		return rows
	}
	// One round commits the anchor plus the accepted drafts. 5 is close to the measured mean
	// accepted at the widths that matter (3.97 at k=7, 4.24 at k=8), so it is the realistic
	// number of new context rows per round.
	const newPerRound = 5

	t.Logf("drafter: %d layers, hidden %d, block %d, fc in %d", len(d.layers), d.hidden, B, d.fc.Cols())
	// Warm-up, discarded: the first draft pays allocator and cache costs that would otherwise
	// land entirely on the first row and make it read slower than a LARGER context — which is
	// exactly what the first version of this table showed.
	if wf, err := d.FuseContext(be, mkctx(64)); err == nil {
		_, _ = d.DraftBlock(be, wf, block)
	}
	t.Logf("%8s %14s %14s %10s", "ctx", "rebuild ms", "incremental ms", "ratio")
	for _, ctxLen := range []int{64, 128, 256, 512, 1024} {
		ctxCat := mkctx(ctxLen)

		// --- rebuild path: what the acceptance harness does every round ---
		fused, err := d.FuseContext(be, ctxCat)
		if err != nil {
			t.Fatalf("FuseContext: %v", err)
		}
		start := time.Now()
		if _, err := d.DraftBlock(be, fused, block); err != nil {
			t.Fatalf("DraftBlock: %v", err)
		}
		rebuild := time.Since(start)

		// --- incremental path: context already built, only the round's new rows go in ---
		warm := d.NewContext()
		d.ExtendContext(be, warm, fused)
		newRows, err := d.FuseContext(be, mkctx(newPerRound))
		if err != nil {
			t.Fatalf("FuseContext(new): %v", err)
		}
		start = time.Now()
		d.ExtendContext(be, warm, newRows)
		if _, err := d.DraftBlockCtx(be, warm, block); err != nil {
			t.Fatalf("DraftBlockCtx: %v", err)
		}
		incr := time.Since(start)

		t.Logf("%8d %14.1f %14.1f %9.1fx", ctxLen,
			float64(rebuild.Microseconds())/1000, float64(incr.Microseconds())/1000,
			float64(rebuild)/float64(incr))
	}
	fmt.Fprintln(os.Stderr)
}
