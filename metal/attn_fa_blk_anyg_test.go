//go:build darwin

package metal

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"testing"
)

// TestAttnFABlk_anyGMatchesG7 is B-P02's identity gate (docs/audit-metal-2026-09-30.md): attention_fa_blk at every
// newly instantiated group size (2, 3, 4, 5, 8) against the graded g7, on one KV head with the same K/V and the same
// leading query heads. Every per-head statement in the kernel touches only head g's state, so each head's partial
// (running max, sum and accumulator, per split) must equal g7's for that head bit for bit. 777 keys: whole 32-key
// blocks and a short tail, over the production split count (16). A kernel whose per-head arithmetic depended on G
// would fail here; that is what lets the new sizes inherit the graded kernel's fidelity.
func TestAttnFABlk_anyGMatchesG7(t *testing.T) {
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	r := chainFixtureResident(t, 1024, true)
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	pool := NewARPool()
	defer pool.Drain()
	lib, err := r.d.CompileLibrary(allKernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	const hd, nKeys, nSplit = 128, 777, 16
	rng := rand.New(rand.NewSource(7))
	rnd := func(n int, sc float64) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(rng.NormFloat64() * sc)
		}
		return v
	}
	q8 := rnd(8*hd, 1)                         // 8 query heads; a size-G run takes the first G
	k, v := rnd(nKeys*hd, 1), rnd(nKeys*hd, 1) // one KV head, kvDim = hd
	half := func(x []float32) Buffer {
		h := make([]uint16, len(x))
		for i, f := range x {
			h[i] = f32ToF16(f)
		}
		return NewBufferU16s(r.d, h)
	}
	kc, vc := half(k), half(v)
	uNKV, uNKeys := NewBufferU32(r.d, 1), NewBufferU32(r.d, nKeys)
	uScale, uWin, uSplit := NewBufferFloats(r.d, []float32{float32(1 / math.Sqrt(hd))}), NewBufferU32(r.d, 0), NewBufferU32(r.d, nSplit)
	run := func(G int) []float32 {
		p, err := r.d.NewComputePipeline(lib, fmt.Sprintf("attention_fa_blk_g%d", G))
		if err != nil {
			t.Fatalf("pipeline g%d: %v", G, err)
		}
		qb := NewBufferFloats(r.d, q8[:G*hd])
		part := r.d.NewBufferLen(nSplit * G * (hd + 2))
		e := r.q.Begin()
		e.DispatchTG(p, nSplit*128, 128, 128*6*G*4, qb, kc, vc, part, uNKV, NewBufferU32(r.d, uint32(G)), uNKeys, uScale, uWin, uSplit)
		e.End()
		if err := e.Err(); err != nil {
			t.Fatalf("g%d: %v", G, err)
		}
		return append([]float32(nil), part.Floats()[:nSplit*G*(hd+2)]...)
	}
	ref := run(7)
	for _, G := range []int{2, 3, 4, 5, 8} {
		got := run(G)
		diff, n := 0, 0
		for s := range nSplit {
			for g := range min(G, 7) {
				a := got[(s*G+g)*(hd+2) : (s*G+g+1)*(hd+2)]
				b := ref[(s*7+g)*(hd+2) : (s*7+g+1)*(hd+2)]
				for i := range a {
					n++
					if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
						diff++
					}
				}
			}
		}
		t.Logf("g%d against g7: %d of %d partial values differ, over %d heads x %d splits", G, diff, n, min(G, 7), nSplit)
		if diff != 0 || n == 0 {
			t.Errorf("g%d: %d of %d partial values differ from g7's for the same heads", G, diff, n)
		}
	}
}
