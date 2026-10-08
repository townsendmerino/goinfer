//go:build cuda

package cuda

import "testing"

// TestPackedAllocSlack pins the rounding rule and the walk that applies it: a buffer under a quantum costs nothing extra, one of a quantum or more rounds up
// to the next multiple, and every hostW under a nested struct/slice/pointer is counted exactly as upW allocates it (wpk, then ws or ws16), while a plain
// vector of numbers beside them is not walked.
func TestPackedAllocSlack(t *testing.T) {
	const q = allocQuantumBytes
	for _, c := range []struct{ n, want int64 }{{0, 0}, {1, 0}, {q - 1, 0}, {q, 0}, {q + 1, q - 1}, {3 * q / 2 * 2, 0}, {5 << 20, 1 << 20}, {9 << 20, 1 << 20}, {6 << 20, 0}} {
		if got := allocRoundSlack(c.n); got != c.want {
			t.Errorf("allocRoundSlack(%d) = %d, want %d", c.n, got, c.want)
		}
	}
	type layer struct {
		q, k   hostW
		norm   []float32
		extras []hostW
		nested *struct{ inner hostW }
	}
	// 5 MiB of wpk (1,310,720 words) slack 1 MiB; 3 MiB of ws16 (1,572,864 halfwords) slack 1 MiB; 9 MiB of ws (2,359,296 floats) slack 1 MiB.
	q4 := hostW{kind: "int4", wpk: make([]uint32, 5<<18), ws16: make([]uint16, 3<<19)}
	q8 := hostW{kind: "int8", wpk: make([]uint32, 3<<18), ws: make([]float32, 9<<18)}
	small := hostW{kind: "int4", wpk: make([]uint32, 1000), ws16: make([]uint16, 100)}
	l := layer{q: q4, k: q8, norm: make([]float32, 8<<20), extras: []hostW{small, q4}, nested: &struct{ inner hostW }{q4}}
	got := packedAllocSlack([]any{[]layer{l}, q8})
	// q4 is counted three times (q, extras[1], nested): 3 x (1 MiB + 1 MiB); q8 twice: 2 x (wpk 3 MiB -> 4: 1 MiB, ws 9 MiB -> 10: 1 MiB); small: 0.
	if want := int64(3*2+2*2) << 20; got != want {
		t.Errorf("packedAllocSlack = %d MiB, want %d MiB", got>>20, want>>20)
	}
}
