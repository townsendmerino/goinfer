//go:build cuda

package cuda

import "testing"

// TestCudaResidentCheckCap gates C3: writes at/past the resident KV cap are refused (a real
// device write there is out-of-bounds memory corruption). Pure logic — no device needed.
func TestCudaResidentCheckCap(t *testing.T) {
	r := &cudaResident{}
	if r.ContextCap() != cudaCtxCap {
		t.Fatalf("ContextCap = %d, want %d", r.ContextCap(), cudaCtxCap)
	}
	for _, c := range []struct {
		pos, n int
		ok     bool
	}{
		{0, 1, true}, {cudaCtxCap - 1, 1, true}, {cudaCtxCap, 1, false},
		{0, cudaCtxCap, true}, {0, cudaCtxCap + 1, false}, {-1, 1, false},
	} {
		err := r.checkCap(c.pos, c.n)
		if (err == nil) != c.ok {
			t.Errorf("checkCap(%d,%d) err=%v, want ok=%v", c.pos, c.n, err, c.ok)
		}
	}
}
