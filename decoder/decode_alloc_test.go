package decoder

import (
	"runtime"
	"testing"
)

// A decode allocation gate (P-01 / M-03). headWorkerPool must not allocate a fusedScratch per pool slot, per layer,
// per decoded token on a path that never reads it: decode runs acc64=true, and attendBatchedHeads computes
// fusedOK = !useAcc64 && treeMask == nil. With GOINFER_FUSED_ATTENTION default-on such an allocation is linear in
// context, zero-filled, GC-churned and unread (two orders of magnitude of TotalAlloc per token at context 256-1024).
//
// The gate asserts the INVARIANT rather than a byte count, which would be a benchmark pinned as a test and would drift
// with every unrelated allocation: turning the default-on fused schedule on or off must not change what a decode step
// allocates, because decode never uses it.
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

	// Warm twice: the first measured window after a prefill pays one-time pool growth (the kh/vt/
	// scores buffers reaching nKeys), which is not per-token cost and would swamp the comparison.
	unsetKnob(t, m, knobFusedAttention)
	perToken(6)
	perToken(6)

	on := perToken(12)
	setKnob(t, m, knobFusedAttention, "0")
	off := perToken(12)

	t.Logf("decode allocation: fused schedule on %d B/token, off %d B/token", on, off)
	// A little slack for scheduler noise; the defect this guards was 160x at this context.
	if off > 0 && on > off+off/4 {
		t.Errorf("the default-on fused schedule costs %d B/token against %d with it off — decode "+
			"does not use it (acc64 ⇒ fusedOK false), so it must cost nothing. That is M-03, which "+
			"was 11.9 MB/token here and 42.6 MB at 1k context", on, off)
	}
}
