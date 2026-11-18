//go:build cuda

package cuda

import "testing"

// TestPackedAllocSlack pins the rounding rule and the walk that applies it: every hostW under a nested struct/slice/pointer is counted
// exactly as upW allocates it (wpk, then ws or ws16), a plain vector of numbers beside them is one buffer, and all of them share one
// arena whose only slack is each buffer's padding to weightAlign and the arena's one rounding to a quantum.
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
	// q4 is 5 MiB of wpk and 3 MiB of ws16, q8 3 MiB of wpk and 9 MiB of ws: each a multiple of weightAlign.
	q4 := hostW{kind: "int4", wpk: make([]uint32, 5<<18), ws16: make([]uint16, 3<<19)}
	q8 := hostW{kind: "int8", wpk: make([]uint32, 3<<18), ws: make([]float32, 9<<18)}
	small := hostW{kind: "int4", wpk: make([]uint32, 1000), ws16: make([]uint16, 100)}
	l := layer{q: q4, k: q8, norm: make([]float32, 8<<20), extras: []hostW{small, q4}, nested: &struct{ inner hostW }{q4}}
	got := packedAllocSlack([]any{[]layer{l}, q8})
	// q4 three times (24 MiB), q8 twice (24 MiB), the 32 MiB norm, and small's 4,000 and 200 bytes padded to 4,096 and 256: an arena
	// of 80 MiB + 4,352 bytes, which rounds to 82 MiB. The slack is that less the 80 MiB + 4,200 bytes requested.
	if want := int64(allocQuantumBytes - 4200); got != want {
		t.Errorf("packedAllocSlack = %d, want %d", got, want)
	}
	if got, in := weightArenaBytes([]any{[]layer{l}, q8}); got != 80<<20+4352 || in != 80<<20+4200 {
		t.Errorf("weightArenaBytes = %d, %d requested, want %d, %d", got, in, 80<<20+4352, 80<<20+4200)
	}
}

// TestPackedAllocSlack_expertsOwnBuffers: an hlayer's expert stacks are not in the arena (they upload on the experts line): each of
// their buffers is rounded and packed on its own, as every weight buffer was before the arena.
func TestPackedAllocSlack_expertsOwnBuffers(t *testing.T) {
	q4 := hostW{kind: "int4", wpk: make([]uint32, 5<<18), ws16: make([]uint16, 3<<19)} // 8 MiB in the arena: no slack
	big := hostW{kind: "int4", wpk: make([]uint32, 5<<18)}                             // 5 MiB on its own: 1 MiB of rounding
	small := hostW{kind: "int4", wpk: make([]uint32, 1000), ws16: make([]uint16, 100)} // 4,200 bytes in one page of their own
	got := packedAllocSlack([]any{[]hlayer{{q: q4, expGU: big, expDown: small}}})
	if want := int64(1<<20 + allocQuantumBytes - 4200); got != want {
		t.Errorf("packedAllocSlack = %d, want %d", got, want)
	}
	if got, _ := weightArenaBytes([]any{[]hlayer{{q: q4, expGU: big, expDown: small}}}); got != 8<<20 {
		t.Errorf("weightArenaBytes = %d, want %d (the experts are not in it)", got, 8<<20)
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

// TestPackedAllocSlack_oneArena: the weights share one allocation, so many buffers of 1.5 MiB cost no page each, and an arena under a
// quantum is one small buffer, packed into a page like any other.
func TestPackedAllocSlack_oneArena(t *testing.T) {
	type layer struct{ a, b hostW }
	one := hostW{kind: "int4", wpk: make([]uint32, 3<<17)} // 1.5 MiB
	if got := packedAllocSlack([]any{[]layer{{one, one}, {one, one}}}); got != 0 {
		t.Errorf("packedAllocSlack of four 1.5 MiB weights = %d, want 0: they fill a 6 MiB arena", got)
	}
	type withNorm struct {
		w    hostW
		norm []float32
	}
	// 1.5 MiB and a quarter of a MiB of norms: a 1.75 MiB arena in a page of its own.
	if got, want := packedAllocSlack([]any{withNorm{one, make([]float32, 1<<16)}}), int64(allocQuantumBytes-3<<19-1<<18); got != want {
		t.Errorf("packedAllocSlack with a norm = %d, want %d", got, want)
	}
}
