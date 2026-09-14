//go:build darwin

package metal

import (
	"encoding/binary"
	"math/rand"
	"testing"
)

// TestBytesToU32_matchesManualLE gates N-29 (audit-metal-2026-09-12.md): bytesToU32 used to
// reconstruct each word with a per-byte shift-and-mask loop; it now does one bulk copy into a
// freshly-allocated (always 4-aligned) []uint32's own byte view. Confirmed red without the fix by
// temporarily reverting to a deliberately wrong byte order (big-endian) — this test caught it
// immediately, since the oracle below is independently computed via encoding/binary rather than by
// re-deriving the same shift expression bytesToU32 itself uses.
func TestBytesToU32_matchesManualLE(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for _, n := range []int{0, 1, 2, 8, 37} {
		b := make([]byte, n*4)
		rng.Read(b)
		got := bytesToU32(b)
		if len(got) != n {
			t.Fatalf("n=%d: len(got)=%d, want %d", n, len(got), n)
		}
		for i := 0; i < n; i++ {
			want := binary.LittleEndian.Uint32(b[4*i:])
			if got[i] != want {
				t.Errorf("n=%d word %d: got %#x, want %#x (LE oracle)", n, i, got[i], want)
			}
		}
	}
}
