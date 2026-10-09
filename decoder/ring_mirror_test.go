package decoder

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"testing"
)

// Audit R-12 (docs/tasks/task-recompute-audit.md §5). K=1 decode on an f32 sliding-window ring used to copy every resident row into scratch each token. It now
// writes the new row first and reads the window in place, from a ring that keeps a MIRROR of its slots ([w, 2w) equal [0, w)) so any window is one contiguous
// slice. Two gates: (1) the ring invariant under every kind of mutation, model-free; (2) the whole decode, bit for bit against the copy path, on real tiny
// sliding-window checkpoints across several wraps.

// mirrorEqual reports whether the mirror half equals the canonical half (true when there is no mirror).
func mirrorEqual(r *ring) (bool, string) {
	if !r.mirrored() {
		return true, ""
	}
	ws := r.w * r.stride
	for i := range ws {
		if math.Float32bits(r.k[i]) != math.Float32bits(r.k[ws+i]) {
			return false, fmt.Sprintf("k slot %d differs from its mirror", i/r.stride)
		}
		if math.Float32bits(r.v[i]) != math.Float32bits(r.v[ws+i]) {
			return false, fmt.Sprintf("v slot %d differs from its mirror", i/r.stride)
		}
	}
	return true, ""
}

// TestRingMirror_invariantAndWindow drives one f32 ring through random writes, batched commits (K from 1 to 2w, the wrap-crossing and the longer-than-the-window
// cases), truncations, a snapshot-style rebuild of k/v at w slots, and decode window reads. After every operation the mirror must equal the canonical half, and
// every window read must equal the rows the copy path (batchReadLocal) assembles for the same positions.
func TestRingMirror_invariantAndWindow(t *testing.T) {
	for _, W := range []int{1, 3, 4, 7} {
		const nKV, hd = 1, 3
		const stride = nKV * hd
		rng := rand.New(rand.NewSource(int64(W) * 17))
		c := NewKVCache(1, nKV, hd, W, 64, nil)
		c.enableRings(W, func(int) bool { return false })
		r := c.rings[0]
		pos := 0
		sawMirror := false
		check := func(op string) {
			t.Helper()
			if ok, why := mirrorEqual(r); !ok {
				t.Fatalf("W=%d after %s (pos %d, count %d): %s", W, op, pos, r.count, why)
			}
		}
		readWindow := func() {
			t.Helper()
			if r.count == 0 {
				return
			}
			p := r.count - 1 // the newest resident position: the one a decode just wrote
			base := max(p-W+1, 0)
			n := p - base + 1
			wk, wv := r.window(base, n)
			// reference: the copy path over the same live rows, assembled the way batchReadLocal does for startPos=p with the newest row as "new"
			dk, dv := make([]float32, n*stride), make([]float32, n*stride)
			newK := append([]float32(nil), r.k[(p%W)*stride:(p%W+1)*stride]...)
			newV := append([]float32(nil), r.v[(p%W)*stride:(p%W+1)*stride]...)
			// history = rows [base, p) read from the ring; batchReadLocal reads them from slots, which the newest row's slot may alias only when n == W+1, never here
			c.rings[0].count-- // batchReadLocal's contract: the new row is NOT yet in the ring
			c.batchReadLocal(0, p, 1, newK, newV, dk, dv)
			c.rings[0].count++
			for i := range wk {
				if math.Float32bits(wk[i]) != math.Float32bits(dk[i]) || math.Float32bits(wv[i]) != math.Float32bits(dv[i]) {
					t.Fatalf("W=%d window [%d,%d) differs from the copy path at float %d", W, base, base+n, i)
				}
			}
			if r.mirrored() {
				sawMirror = true
			}
		}
		vec := func() []float32 { return randVec(rng, stride) }
		for range 400 {
			switch op := rng.Intn(10); {
			case op < 5: // a decode token: write the next position
				r.write(pos, vec(), vec())
				pos++
				check("write")
				readWindow()
			case op < 7: // a batched commit of K rows at the current position
				K := 1 + rng.Intn(2*W)
				nk, nv := make([]float32, K*stride), make([]float32, K*stride)
				for i := range nk {
					nk[i], nv[i] = float32(rng.NormFloat64()), float32(rng.NormFloat64())
				}
				c.commitBatch(0, pos, K, nk, nv)
				pos += K
				check("commitBatch")
				readWindow()
			case op < 8 && pos > 0: // truncate to a random earlier position, then keep writing from there
				p := rng.Intn(pos + 1)
				r.truncate(p)
				pos = p
				check("truncate")
			case op < 9 && r.mirrored(): // a snapshot restore rebuilds k/v at w slots: the mirror is gone and must come back coherent
				ws := r.w * r.stride
				r.k, r.v = append([]float32(nil), r.k[:ws]...), append([]float32(nil), r.v[:ws]...)
				if r.mirrored() {
					t.Fatalf("W=%d: a w-slot ring reports mirrored", W)
				}
				readWindow() // ensureMirror inside window() when the run wraps
				check("rebuild + read")
			default:
				readWindow()
			}
		}
		if W > 1 && !sawMirror {
			t.Fatalf("W=%d: the run never wrapped a window, so the mirror path was never exercised", W)
		}
	}
}

// TestRingMirror_decodeBitIdenticalToTheCopyPath runs real tiny sliding-window checkpoints token by token across several wraps, once with the old copy path
// (ringDirectDecode=false) and once with the in-place window, and requires every logit of every step to match bit for bit. Non-vacuity: after the run some
// local layer must hold a mirror (the in-place read took a wrapped window), and the copy-path run must hold none.
func TestRingMirror_decodeBitIdenticalToTheCopyPath(t *testing.T) {
	for _, fx := range []string{"gemma2-tiny", "gemma2-hd-tiny", "cohere2-tiny"} {
		t.Run(fx, func(t *testing.T) {
			dir := "../testdata/" + fx
			if _, err := os.Stat(dir); err != nil {
				t.Skipf("no fixture %s", dir)
			}
			m, err := Load(dir, Options{Backend: "cpu"})
			if err != nil {
				t.Fatalf("load: %v", err)
			}
			defer m.Close()
			prev := ringDirectDecode
			defer func() { ringDirectDecode = prev }()
			vocab := m.w.arch.VocabSize
			if vocab <= 8 {
				t.Fatalf("vocab %d", vocab)
			}
			const N = 52 // 6 to 13 windows deep at W=8..4, past every wrap
			toks := make([]int, N)
			rng := rand.New(rand.NewSource(5))
			for i := range toks {
				toks[i] = 1 + rng.Intn(vocab-1)
			}
			run := func(direct bool) ([][]float32, *KVCache) {
				ringDirectDecode = direct
				cache := m.NewCache(N + 2)
				var all [][]float32
				for i, tok := range toks {
					lg, err := m.forward(tok, cache)
					if err != nil {
						t.Fatalf("direct=%v step %d: %v", direct, i, err)
					}
					all = append(all, append([]float32(nil), lg...))
				}
				return all, cache
			}
			ref, refC := run(false)
			got, gotC := run(true)
			for i := range ref {
				for j := range ref[i] {
					if math.Float32bits(ref[i][j]) != math.Float32bits(got[i][j]) {
						t.Fatalf("step %d logit %d: in-place %v != copy path %v (NOT bit-identical)", i, j, got[i][j], ref[i][j])
					}
				}
			}
			mirrors := func(c *KVCache) int {
				n := 0
				for _, r := range c.rings {
					if r != nil && r.mirrored() {
						n++
					}
				}
				return n
			}
			if mirrors(gotC) == 0 {
				t.Fatal("no local layer holds a mirror after the in-place run: the wrapped-window read never happened, so this proved nothing")
			}
			if mirrors(refC) != 0 {
				t.Fatal("the copy-path run allocated a mirror: ringDirectDecode=false must leave rings exactly as before")
			}
			for l, r := range gotC.rings {
				if r != nil {
					if ok, why := mirrorEqual(r); !ok {
						t.Fatalf("layer %d: %s", l, why)
					}
				}
			}
		})
	}
}

// TestKvBytesForCtx_countsTheRingMirror pins the fit guard to what the ring really holds: a local f32 layer past its window is 2*W positions (the window and its
// mirror), a layer short of its window or an int8 ring is W (or ctx) positions, and a global layer is ctx.
func TestKvBytesForCtx_countsTheRingMirror(t *testing.T) {
	const nLayers, nH, nKV, hd, W = 2, 4, 2, 8, 16
	arch := ringTestArch(nH, nKV, hd, W, func(l int) bool { return l == 1 }) // layer 0 local, layer 1 global
	arch.NumLayers = nLayers
	dim := arch.kvDimAt(0)
	if dim != nKV*hd {
		t.Fatalf("kvDimAt = %d, want %d: the test arch does not price layers the way the model does", dim, nKV*hd)
	}
	perPos := func(positions int, perElem float64) float64 { return 2 * perElem * float64(dim) * float64(positions) }
	cases := []struct {
		name        string
		ctx         int
		kvI8        bool
		local, glob int // positions priced for the local and the global layer
		perElem     float64
	}{
		{"short of the window", W - 3, false, W - 3, W - 3, 4},
		{"past the window, f32: window plus mirror", 3 * W, false, 2 * W, 3 * W, 4},
		{"past the window, int8: no mirror", 3 * W, true, W, 3 * W, 1.125},
	}
	for _, c := range cases {
		want := int64(perPos(c.local, c.perElem) + perPos(c.glob, c.perElem))
		if got := kvBytesForCtx(arch, c.ctx, false, c.kvI8); got != want {
			t.Errorf("%s: kvBytesForCtx = %d, want %d (local %d positions + global %d)", c.name, got, want, c.local, c.glob)
		}
	}
}
