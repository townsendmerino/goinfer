//go:build cuda

package cuda

import "testing"

// TestPackedAllocSlack pins the rounding rule and the walk that applies it: a buffer of a quantum or more rounds up to the next multiple, the
// buffers under a quantum cost the pages they leave part empty (TestPagePacker), every hostW under a nested struct/slice/pointer is counted
// exactly as upW allocates it (wpk, then ws or ws16), and a plain vector of numbers beside them is one buffer.
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
	// q4 is counted three times (q, extras[1], nested): 3 x (1 MiB + 1 MiB); q8 twice: 2 x (wpk 3 MiB -> 4: 1 MiB, ws 9 MiB -> 10: 1 MiB); the
	// 32 MiB vector rounds by nothing. small is two buffers under a quantum (4,000 and 200 bytes): no rounding, but they share one page the
	// rest of which nothing fills, 2 MiB less their 4,200 bytes.
	if want := int64(3*2+2*2)<<20 + allocQuantumBytes - 4200; got != want {
		t.Errorf("packedAllocSlack = %d, want %d", got, want)
	}
}

// TestPagePacker pins the placement rule: a buffer under a quantum goes into the partly used page with the least room that holds
// it (not the newest page, not the oldest), a buffer of a quantum or more takes no part, and the waste is the pages' bytes less
// the requested ones. The case is the one that separated the models on Qwen2.5-Coder-1.5B: a 6.56 MiB buffer is followed by a
// 0.82 MiB one, and the 0.82 MiB one fills an older page's room, not a new page.
func TestPagePacker(t *testing.T) {
	const kib = 1 << 10
	var p pagePacker
	for _, n := range []int64{1500 * kib, 1900 * kib, 900 * kib, 100 * kib} {
		p.add(n)
	}
	// 1500 -> page A (548 left); 1900 -> page B (148 left); 900 -> fits neither: page C (1148 left); 100 -> the least room that holds it: B.
	if p.pages != 3 {
		t.Errorf("pages = %d, want 3", p.pages)
	}
	p.add(140 * kib) // B's 48 KiB left cannot hold it; A's 548 and C's 1148 can: the least, A
	p.add(500 * kib) // A has 408 left: C
	if p.pages != 3 {
		t.Errorf("pages = %d after two buffers that fit older pages, want 3", p.pages)
	}
	p.add(1100 * kib) // fits no page: a fourth
	if p.pages != 4 {
		t.Errorf("pages = %d, want 4", p.pages)
	}
	p.add(allocQuantumBytes) // a quantum or more is allocRoundSlack's, not a page's
	p.add(5 << 20)
	p.add(0)
	var sum int64
	for _, n := range []int64{1500, 1900, 900, 100, 140, 500, 1100} {
		sum += n * kib
	}
	if want := 4*allocQuantumBytes - sum; p.waste() != want {
		t.Errorf("waste = %d, want %d", p.waste(), want)
	}
}

// TestPackedAllocSlack_packsTheSmallOnes: the walk's figure is the rounding of the large buffers plus the pages the small ones
// leave part empty, so a model of many 1.5 MiB buffers (no rounding each, a page each) is priced by the pages.
func TestPackedAllocSlack_packsTheSmallOnes(t *testing.T) {
	type layer struct{ a, b hostW }
	one := hostW{kind: "int4", wpk: make([]uint32, 3<<17)} // 1.5 MiB
	got := packedAllocSlack([]any{[]layer{{one, one}, {one, one}}})
	// Four 1.5 MiB buffers, one page each: 4 x 0.5 MiB left over, nothing else to fill it.
	if want := int64(4) * (allocQuantumBytes - 3<<19); got != want {
		t.Errorf("packedAllocSlack = %d, want %d", got, want)
	}
	// A vector of numbers is a buffer too: 0.25 MiB of norms fits the room beside the first.
	type withNorm struct {
		w    hostW
		norm []float32
	}
	got = packedAllocSlack([]any{withNorm{one, make([]float32, 1<<16)}})
	if want := int64(allocQuantumBytes - 3<<19 - 1<<18); got != want {
		t.Errorf("packedAllocSlack with a norm = %d, want %d", got, want)
	}
}
