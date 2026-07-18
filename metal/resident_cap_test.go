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
