package decoder

import (
	"context"
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
			out, err := m.forwardLayersN(context.Background(), ids, m.NewCache(K+8))
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

// The budget must actually bind, and must degrade toward serial rather than
// letting per-slot scratch grow without bound. These are pure arithmetic over
// the sizing rule, so they run everywhere with no model.
func TestPrefillAttnWorkerBudget(t *testing.T) {
	os.Unsetenv("GOINFER_PREFILL_ATTN_WORKERS")
	const hd, nH = 128, 12

	// Short prompt: the P-core / head cap binds, not the budget.
	if got := prefillAttnWorkers(64, 64, hd, nH); got != maxAttnWorkers {
		t.Errorf("K=64: workers = %d, want %d (budget must not bind on a short prompt)", got, maxAttnWorkers)
	}
	// The count must be monotonically non-increasing in prompt length, and must
	// reach 1 rather than allocating unboundedly on a very long one.
	prev := maxAttnWorkers
	for _, K := range []int{512, 1520, 3020, 8192, 32768} {
		got := prefillAttnWorkers(K, K, hd, nH)
		if got > prev {
			t.Errorf("K=%d: workers rose to %d from %d — the budget must never grow with prompt length", K, got, prev)
		}
		if got < 1 {
			t.Errorf("K=%d: workers = %d, must never be below 1", K, got)
		}
		perSlotMB := float64(4*(K*K+2*K*hd+2*K*hd)) / (1 << 20)
		t.Logf("K=%-6d workers=%d  (per-slot scratch %.1f MB, total %.1f MB)", K, got, perSlotMB, perSlotMB*float64(got))
		prev = got
	}
	if got := prefillAttnWorkers(32768, 32768, hd, nH); got != 1 {
		t.Errorf("K=32768: workers = %d, want 1 — a 4 GB slot must fall back to serial", got)
	}
	// The escape hatch restores the exact pre-G16 path.
	t.Setenv("GOINFER_PREFILL_ATTN_WORKERS", "1")
	if got := prefillAttnWorkers(64, 64, hd, nH); got != 1 {
		t.Errorf("override to 1: got %d, want 1", got)
	}
}
