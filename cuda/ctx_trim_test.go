//go:build cuda

package cuda

import "testing"

// TestTrimUnpinnedCtx pins the arithmetic behind the unpinned-context trim (the 2026-10-07 night gate's five default-context failures): the figures are the
// two real misses, a 7B at 16000 positions (1.84 GB of KV, 1.80 GB free) and Gemma-3 at 11800 (3.29 GB, 3.35 GB free).
func TestTrimUnpinnedCtx(t *testing.T) {
	const margin = 384 << 20
	const gb = 1e9
	cases := []struct {
		name              string
		cap               int
		need, free, extra int64
		wantC             int
		wantOK            bool
	}{
		{"fits already", 16384, int64(1.0 * gb), int64(4 * gb), 0, 16384, false},
		{"7B misses by the margin", 16000, int64(1.84 * gb), int64(1.80 * gb), 0, 0, true},
		{"gemma-3 misses by the margin", 11800, int64(3.29 * gb), int64(3.35 * gb), 0, 0, true},
		{"a companion attach's bytes count", 16384, int64(2.0 * gb), int64(2.5 * gb), int64(0.5 * gb), 0, true},
		{"below the floor declines as before", 16384, int64(8 * gb), int64(1 * gb), 0, 16384, false},
	}
	for _, c := range cases {
		got, ok := trimUnpinnedCtx(c.cap, c.need, c.free, c.extra, margin)
		if ok != c.wantOK {
			t.Errorf("%s: ok=%v, want %v (got %d)", c.name, ok, c.wantOK, got)
			continue
		}
		if !ok {
			if got != c.cap {
				t.Errorf("%s: no trim must return the cap %d, got %d", c.name, c.cap, got)
			}
			continue
		}
		if got < cudaCtxCapDefault || got >= c.cap {
			t.Errorf("%s: trimmed to %d, want in [%d, %d)", c.name, got, cudaCtxCapDefault, c.cap)
		}
		if used := int64(got) * (c.need / int64(c.cap)); used+c.extra+margin > c.free {
			t.Errorf("%s: the trimmed %d positions still need %d B beside %d B reserved, over the %d B free", c.name, got, used, c.extra+margin, c.free)
		}
		if next := int64(got+1) * (c.need / int64(c.cap)); next+c.extra+margin <= c.free {
			t.Errorf("%s: trimmed to %d but %d positions would also fit — not the largest", c.name, got, got+1)
		}
	}
}
