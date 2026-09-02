package decoder

import (
	"os"
	"runtime"
	"testing"
)

// P-01 / M-03: A DECODE ALLOCATION GATE. P-01's own note says one would have caught M-03, so here
// it is.
//
// M-03: headWorkerPool allocated a fresh fusedScratch per pool slot, per layer, per decoded token,
// on a path that never reads it — decode runs acc64=true and attendBatchedHeads computes
// fusedOK = !useAcc64 && treeMask == nil. Since 84e0f13 made GOINFER_FUSED_ATTENTION default-on,
// that was the shipped default. Measured before the fix, TotalAlloc per decoded token on
// qwen2.5-coder-0.5b int8int8, interleaved:
//
//	context 256    11.9 MB/token (fused on)  vs  74 KB/token (=0)   160x
//	context 1024   42.6 MB/token (fused on)  vs  74 KB/token (=0)   573x
//
// linear in context, zero-filled, GC-churned, and unread. After: identical with the schedule on and
// off.
//
// The gate asserts the INVARIANT rather than a byte count, which would be a benchmark pinned as a
// test and would drift with every unrelated allocation: turning the default-on fused schedule on or
// off must not change what a decode step allocates, because decode never uses it.
func TestDecode_fusedScheduleCostsNothingPerToken(t *testing.T) {
	m, err := loadBenchModel()
	if err != nil {
		t.Skipf("no model (%v); set GOINFER_PREQUANT_GGUF", err)
	}
	const ctxLen = 256
	cache := m.NewCache(ctxLen + 8)
	for i := range ctxLen {
		if _, err := m.forward(785+(i%7), cache); err != nil {
			t.Fatalf("prefill: %v", err)
		}
	}

	perToken := func(n int) uint64 {
		var a, b runtime.MemStats
		runtime.GC()
		runtime.ReadMemStats(&a)
		for range n {
			if _, err := m.forward(785, cache); err != nil {
				t.Fatalf("forward: %v", err)
			}
			cache.pos-- // hold the context length fixed
		}
		runtime.ReadMemStats(&b)
		return (b.TotalAlloc - a.TotalAlloc) / uint64(n)
	}

	prev, had := os.LookupEnv("GOINFER_FUSED_ATTENTION")
	defer func() {
		if had {
			os.Setenv("GOINFER_FUSED_ATTENTION", prev)
		} else {
			os.Unsetenv("GOINFER_FUSED_ATTENTION")
		}
	}()

	// Warm twice: the first measured window after a prefill pays one-time pool growth (the kh/vt/
	// scores buffers reaching nKeys), which is not per-token cost and would swamp the comparison.
	os.Unsetenv("GOINFER_FUSED_ATTENTION")
	perToken(6)
	perToken(6)

	on := perToken(12)
	os.Setenv("GOINFER_FUSED_ATTENTION", "0")
	off := perToken(12)

	t.Logf("decode allocation: fused schedule on %d B/token, off %d B/token", on, off)
	// A little slack for scheduler noise; the defect this guards was 160x at this context.
	if off > 0 && on > off+off/4 {
		t.Errorf("the default-on fused schedule costs %d B/token against %d with it off — decode "+
			"does not use it (acc64 ⇒ fusedOK false), so it must cost nothing. That is M-03, which "+
			"was 11.9 MB/token here and 42.6 MB at 1k context", on, off)
	}
}
