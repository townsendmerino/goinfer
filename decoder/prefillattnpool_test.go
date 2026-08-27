package decoder

import (
	"context"
	"fmt"
	"os"
	"testing"
)

// G16 gate — prefill attention's head-parallel fan-out must be BIT-IDENTICAL to
// the serial path it replaces.
//
// A1's constraint is the whole reason this is allowed at all: "Parallelism may
// only split independent outputs across workers/registers — heads, ...". Splitting
// heads across workers therefore cannot change any value, only who computes it.
// That is a claim about the code, and this is the test that makes it a fact.
//
// A1 asserted its own bit-identity through the parity goldens and left no
// pool-invariance test, so this is new: same prompt, same cache, pool len 1 vs
// the budgeted count, compared float-for-float. Exact equality, not a tolerance —
// a tolerance here would silently accept the reassociation A1 exists to prevent.
func TestPrefillAttnPoolInvariance(t *testing.T) {
	m, err := loadBenchModel()
	if err != nil {
		t.Skipf("no model (%v); set GOINFER_PREQUANT_GGUF", err)
	}
	for _, K := range []int{64, 512, 1200} {
		if !m.canBatchN(K) {
			t.Skipf("model has no batched prefill at K=%d", K)
		}
		ids := make([]int, K)
		for i := range ids {
			ids[i] = 700 + i%64 // varied ids: a degenerate prompt can hide a reassociation
		}

		run := func(workers string) []float32 {
			t.Helper()
			t.Setenv("GOINFER_PREFILL_ATTN_WORKERS", workers)
			out, err := m.forwardLayersN(context.Background(), ids, m.NewCache(K+8), false)
			if err != nil {
				t.Fatalf("K=%d workers=%s: %v", K, workers, err)
			}
			return out
		}

		serial := run("1")
		parallel := run("6")

		if len(serial) != len(parallel) {
			t.Fatalf("K=%d: length %d vs %d", K, len(serial), len(parallel))
		}
		diffs := 0
		firstAt := -1
		for i := range serial {
			if serial[i] != parallel[i] {
				diffs++
				if firstAt < 0 {
					firstAt = i
				}
			}
		}
		if diffs != 0 {
			t.Errorf("K=%d: head-parallel prefill is NOT bit-identical — %d/%d floats differ, first at %d (%v vs %v)",
				K, diffs, len(serial), firstAt, serial[firstAt], parallel[firstAt])
		} else {
			t.Logf("K=%d: %d floats bit-identical, serial vs 6 workers", K, len(serial))
		}
	}
}

// The worker count must stay inside its caps and honor its override.
//
// NOTE — this test was rewritten when G20 landed, and the reason matters more
// than the assertions. As written for G16 it asserted that K=32768 "must fall
// back to serial, a 4 GB slot", because an untiled slot's scores buffer was
// K*nKeys floats. G20 tiles the query rows, so no such slot exists any more: a
// slot is one row tile wide (attnScoreTileBytes), and six workers at 32k cost
// ~150 MB, not ~25 GB. The old assertion was not wrong when written; its premise
// was removed. That is why it is replaced rather than relaxed — the property it
// protected (unbounded per-slot growth must not happen) is now asserted directly,
// against the real tiled size, by TestAttnRowTileBoundsScratch.
func TestPrefillAttnWorkerBudget(t *testing.T) {
	os.Unsetenv("GOINFER_PREFILL_ATTN_WORKERS")
	os.Unsetenv("GOINFER_ATTN_ROW_TILE")
	const hd, nH = 128, 12

	for _, K := range []int{64, 512, 1520, 3020, 8192, 32768} {
		got := prefillAttnWorkers(K, K, hd, nH)
		if got < 1 {
			t.Errorf("K=%d: workers = %d, must never be below 1", K, got)
		}
		if got > maxAttnWorkers || got > nH {
			t.Errorf("K=%d: workers = %d exceeds the P-core/head cap (%d/%d)", K, got, maxAttnWorkers, nH)
		}
		tile := attnRowTile(K, K)
		perSlot := 4 * (tile*K + 2*K*hd + 2*tile*hd)
		t.Logf("K=%-6d tile=%-5d workers=%d  per-slot %.1f MB, total %.1f MB",
			K, tile, got, float64(perSlot)/(1<<20), float64(perSlot*got)/(1<<20))
	}

	// A head count below the pool size binds before the P-core cap does.
	if got := prefillAttnWorkers(512, 512, hd, 2); got != 2 {
		t.Errorf("nH=2: workers = %d, want 2 — the head count must cap the pool", got)
	}
	// The escape hatch restores the exact pre-G16 serial path.
	t.Setenv("GOINFER_PREFILL_ATTN_WORKERS", "1")
	if got := prefillAttnWorkers(64, 64, hd, nH); got != 1 {
		t.Errorf("override to 1: got %d, want 1", got)
	}
}

// G20 gate — query-row tiling must be BIT-IDENTICAL to the untiled path.
//
// Tiling splits independent outputs (each query row's attention is its own
// reduction over keys), which is what A1 permits. The hazard is not the matmuls
// but the POSITION MAPPING: the softmax uses startPos+row, treeRowPos[row] and
// treeMask[row], all indexed by the GLOBAL row, while the buffers are indexed
// within the tile. An off-by-tile there is a silent attention-mask bug that
// produces plausible output — exactly the kind of defect a tolerance-based
// comparison would wave through, so this compares float-for-float.
func TestPrefillAttnRowTileInvariance(t *testing.T) {
	m, err := loadBenchModel()
	if err != nil {
		t.Skipf("no model (%v); set GOINFER_PREQUANT_GGUF", err)
	}
	prog := newProgress(t, t.Name(), 3*5).Uneven() // 3 context lengths x (untiled + 4 tiles); cost grows with K
	for _, K := range []int{64, 512, 1200} {
		prog.Phase(fmt.Sprintf("K=%d", K))
		if !m.canBatchN(K) {
			t.Skipf("model has no batched prefill at K=%d", K)
		}
		ids := make([]int, K)
		for i := range ids {
			ids[i] = 700 + i%64
		}
		run := func(tile string) []float32 {
			t.Helper()
			t.Setenv("GOINFER_ATTN_ROW_TILE", tile)
			t.Setenv("GOINFER_PREFILL_ATTN_WORKERS", "1") // isolate tiling from fan-out
			out, err := m.forwardLayersN(context.Background(), ids, m.NewCache(K+8), false)
			if err != nil {
				t.Fatalf("K=%d tile=%s: %v", K, tile, err)
			}
			return out
		}
		// A tile equal to K is the untiled shape: one pass, exactly as before G20.
		untiled := run(fmt.Sprint(K))
		prog.Step(1)
		for _, tile := range []string{"1", "7", "64", "333"} {
			got := run(tile)
			prog.Step(1)
			if len(got) != len(untiled) {
				t.Fatalf("K=%d tile=%s: length %d vs %d", K, tile, len(got), len(untiled))
			}
			diffs, firstAt := 0, -1
			for i := range untiled {
				if untiled[i] != got[i] {
					diffs++
					if firstAt < 0 {
						firstAt = i
					}
				}
			}
			if diffs != 0 {
				t.Errorf("K=%d tile=%s: NOT bit-identical to untiled — %d/%d floats differ, first at %d (%v vs %v)",
					K, tile, diffs, len(untiled), firstAt, untiled[firstAt], got[firstAt])
			}
		}
		t.Logf("K=%d: tiles 1/7/64/333 all bit-identical to untiled (%d floats)", K, len(untiled))
	}
}

// The tile must bound scratch rather than track prompt length: that is the whole
// point, and it is what lets the G16 pool keep its workers on a long prompt.
func TestAttnRowTileBoundsScratch(t *testing.T) {
	os.Unsetenv("GOINFER_ATTN_ROW_TILE")
	os.Unsetenv("GOINFER_PREFILL_ATTN_WORKERS")
	const hd, nH = 128, 12
	prevBytes := 0
	for _, K := range []int{512, 1520, 3020, 8192, 32768} {
		tile := attnRowTile(K, K)
		scoreBytes := 4 * tile * K
		if scoreBytes > attnScoreTileBytes {
			t.Errorf("K=%d: score tile is %d bytes, over the %d budget", K, scoreBytes, attnScoreTileBytes)
		}
		w := prefillAttnWorkers(K, K, hd, nH)
		t.Logf("K=%-6d tile=%-5d scores=%4.1f MB  workers=%d", K, tile, float64(scoreBytes)/(1<<20), w)
		prevBytes = scoreBytes
	}
	_ = prevBytes
	// The case G20 exists for: at 8k the untiled slot was 272 MB and forced the
	// pool to a single worker. Tiled, it must afford the full fan-out.
	if w := prefillAttnWorkers(8192, 8192, hd, nH); w < maxAttnWorkers {
		t.Errorf("K=8192: workers = %d, want %d — tiling was supposed to make long prompts affordable", w, maxAttnWorkers)
	}
}
