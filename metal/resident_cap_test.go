//go:build darwin

package metal

import "testing"

// TestMetalResidentCheckCap gates C3 (Metal half): writes at/past the resident KV cap are refused
// — a real device write there is out-of-bounds and, on unified memory, silently corrupts adjacent
// MTLBuffers. Pure logic (checkCap only reads metalCtxCap), so no Metal device is needed.
func TestMetalResidentCheckCap(t *testing.T) {
	r := &metalResident{}
	if r.ContextCap() != metalCtxCap {
		t.Fatalf("ContextCap = %d, want %d", r.ContextCap(), metalCtxCap)
	}
	for _, c := range []struct {
		pos, n int
		ok     bool
	}{
		{0, 1, true}, {metalCtxCap - 1, 1, true}, {metalCtxCap, 1, false},
		{0, metalCtxCap, true}, {0, metalCtxCap + 1, false}, {-1, 1, false},
	} {
		err := r.checkCap(c.pos, c.n)
		if (err == nil) != c.ok {
			t.Errorf("checkCap(%d,%d) err=%v, want ok=%v", c.pos, c.n, err, c.ok)
		}
	}
}

// TestMetalCtxCapWithinKernelBound pins the invariant that keeps the resident context ceiling a
// FACT rather than an assertion: checkCap only bounds nKeys to metalCtxCap, so the attention
// kernel's static `threadgroup float sc[4096]` (attnScoreKeyBound) is what actually caps a correct
// run — the guard is only safe because metalCtxCap ≤ that array. Gemma 4 advertises 256K context
// and its five global layers grow with position, so nKeys past the ceiling IS reachable; this test
// fails the moment someone bumps metalCtxCap past the kernel's score buffer without resizing sc[],
// turning a silent OOB threadgroup write (unified-memory corruption) into a compile-then-test stop.
// The matching correctness measurement at the exact boundary (nKeys=4096) lives in
// TestAttention_ShippedKernelShapes. Pure logic — no Metal device needed.
func TestMetalCtxCapWithinKernelBound(t *testing.T) {
	if metalCtxCap > attnScoreKeyBound {
		t.Fatalf("metalCtxCap=%d exceeds the attention kernel's sc[%d] score buffer — a run at nKeys in (%d,%d] is an out-of-bounds threadgroup write; resize `threadgroup float sc[...]` in kernels.go before raising the cap",
			metalCtxCap, attnScoreKeyBound, attnScoreKeyBound, metalCtxCap)
	}
}
