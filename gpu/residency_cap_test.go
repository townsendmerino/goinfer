//go:build gpu

package gpu

import (
	"strings"
	"testing"
)

// TestResidentDecoderCheckCap gates M20/C3: writes at/past the resident KV cap are refused
// (a real Forward there is a silent out-of-bounds device write, WGSL-clamped → garbage reads).
// Pure logic — no device.
func TestResidentDecoderCheckCap(t *testing.T) {
	rd := &residentDecoder{ctxCap: 8}
	if rd.ContextCap() != 8 {
		t.Fatalf("ContextCap = %d, want 8", rd.ContextCap())
	}
	for _, c := range []struct {
		pos, n int
		ok     bool
	}{
		{0, 1, true}, {7, 1, true}, {8, 1, false}, // pos 8 == cap → OOB
		{0, 8, true}, {0, 9, false}, // ForwardN: fills exactly / overruns
		{6, 2, true}, {7, 2, false}, {-1, 1, false},
	} {
		err := rd.checkCap(c.pos, c.n)
		if (err == nil) != c.ok {
			t.Errorf("checkCap(%d,%d) err=%v, want ok=%v", c.pos, c.n, err, c.ok)
		}
		if err != nil && !strings.Contains(err.Error(), "context cap") {
			t.Errorf("checkCap(%d,%d) message %q missing 'context cap'", c.pos, c.n, err)
		}
	}
}
