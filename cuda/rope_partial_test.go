//go:build cuda

package cuda

import (
	"context"
	"math"
	"testing"

	gc "github.com/eitamring/gocudrv/cuda"
)

// TestRopePartial is the kernel gate for partial rotary (rotary_dim < head_dim: GLM, Phi).
//
// It exists because no admitted model reaches that path yet — glm-tiny also needs the shared
// expert, which is not built. Landing a kernel with no gate on the reasoning that "the model
// test will catch it later" is how dead code rots into a silent bug the day it wakes up, so the
// kernel is gated in isolation now, the same way the MoE kernels were gated before their
// dispatch existed.
//
// The specific bug this is written to catch: the pre-partial kernel gave each k thread the pair
// (d, d+hd/2) and had it store BOTH elements, which covered all hd elements only because
// 2*(hd/2) == hd. With rhalf < hd/2 the pair threads touch just [0, 2*rhalf), so the tail
// [2*rhalf, hd) is never written into the KV cache. Nothing errors — attention simply reads
// whatever was in the cache, which for position 0 of a fresh buffer is zeros. A "half the head
// dims are silently zero" bug produces plausible-looking logits.
//
// The reference is written from the SEMANTICS (rotate the first rotaryDim dims as (d, d+rhalf)
// pairs; pass the tail through; cache the whole head) rather than transliterated from the
// kernel, so a shared misreading cannot cancel out.
func TestRopePartial(t *testing.T) {
	if err := gc.Init(); err != nil {
		t.Skipf("cuInit: %v", err)
	}
	dev, err := gc.GetDevice(0)
	if err != nil {
		t.Skipf("no device: %v", err)
	}
	cx, err := dev.Primary()
	if err != nil {
		t.Skipf("primary ctx: %v", err)
	}
	defer cx.Close()
	bg := context.Background()

	mod, err := cx.LoadModule(gemvFwdPTX)
	if err != nil {
		t.Fatalf("LoadModule: %v", err)
	}
	fn, err := mod.Function("rope_kv")
	if err != nil {
		t.Fatalf("Function(rope_kv): %v", err)
	}
	stream := mustStream(t, cx)

	cases := []struct {
		name          string
		nH, nKV, hd   int
		rotaryDim     int
		pos           int
		wantTailUnrot bool
	}{
		{"full-rotary/unchanged", 4, 2, 16, 16, 3, false},
		{"glm/half-rotary", 4, 2, 16, 8, 5, true},
		{"phi/quarter-rotary", 8, 2, 32, 8, 7, true},
		{"partial/pos0", 2, 1, 8, 4, 0, true},
		{"odd-ish/6-of-16", 4, 4, 16, 6, 2, true},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			rhalf := c.rotaryDim / 2
			qDim, kvDim := c.nH*c.hd, c.nKV*c.hd
			const cap = 8

			var seed uint32 = 424242
			rnd := func() float32 {
				seed = seed*1664525 + 1013904223
				return float32(int32(seed>>8)%2000-1000) / 500
			}
			q := make([]float32, qDim)
			k := make([]float32, kvDim)
			v := make([]float32, kvDim)
			for i := range q {
				q[i] = rnd()
			}
			for i := range k {
				k[i] = rnd()
				v[i] = rnd()
			}
			invF := make([]float32, c.hd/2)
			for d := range invF {
				invF[d] = float32(1.0 / math.Pow(10000, float64(2*d)/float64(c.rotaryDim)))
			}

			// --- CPU reference ---
			wq := append([]float32(nil), q...)
			wk := append([]float32(nil), k...)
			wkc := make([]float32, cap*kvDim)
			wvc := make([]float32, cap*kvDim)
			rot := func(buf []float32, nHeads int) {
				for h := 0; h < nHeads; h++ {
					base := h * c.hd
					for d := 0; d < rhalf; d++ {
						ang := float64(c.pos) * float64(invF[d])
						cs, sn := math.Cos(ang), math.Sin(ang)
						a, b := float64(buf[base+d]), float64(buf[base+d+rhalf])
						buf[base+d] = float32(a*cs - b*sn)
						buf[base+d+rhalf] = float32(a*sn + b*cs)
					}
					// [rotaryDim, hd) deliberately untouched.
				}
			}
			rot(wq, c.nH)
			rot(wk, c.nKV)
			for h := 0; h < c.nKV; h++ {
				for d := 0; d < c.hd; d++ { // the WHOLE head is cached, rotated or not
					wkc[c.pos*kvDim+h*c.hd+d] = wk[h*c.hd+d]
					wvc[c.pos*kvDim+h*c.hd+d] = v[h*c.hd+d]
				}
			}

			// --- GPU ---
			dq := mustAlloc[float32](t, cx, qDim)
			dk := mustAlloc[float32](t, cx, kvDim)
			dv := mustAlloc[float32](t, cx, kvDim)
			dinv := mustAlloc[float32](t, cx, len(invF))
			dkc := mustAlloc[float32](t, cx, cap*kvDim)
			dvc := mustAlloc[float32](t, cx, cap*kvDim)
			defer dq.Close()
			defer dk.Close()
			defer dv.Close()
			defer dinv.Close()
			defer dkc.Close()
			defer dvc.Close()
			// Poison the caches: a tail the kernel forgets to store must be DETECTABLE. Zero-fill
			// would hide it, since the reference tail could coincidentally be near zero.
			poison := make([]float32, cap*kvDim)
			for i := range poison {
				poison[i] = -777.0
			}
			for _, p := range []struct {
				b *gc.Buffer[float32]
				h []float32
			}{{dq, q}, {dk, k}, {dv, v}, {dinv, invF}, {dkc, poison}, {dvc, poison}} {
				if e := gc.CopyHtoD(bg, p.b, p.h); e != nil {
					t.Fatalf("H2D: %v", e)
				}
			}

			tail := c.hd - 2*rhalf
			n := c.nH*rhalf + c.nKV*rhalf + c.nKV*tail
			if e := fn.LaunchOn(bg, stream, g1cfg(n, 256),
				gc.Arg(dq), gc.Arg(dk), gc.Arg(dv), gc.Arg(dinv), gc.Arg(dkc), gc.Arg(dvc),
				gc.ArgValue(int32(c.nH)), gc.ArgValue(int32(c.nKV)), gc.ArgValue(int32(c.hd)),
				gc.ArgValue(int32(c.pos)), gc.ArgValue(int32(rhalf))); e != nil {
				t.Fatalf("launch: %v", e)
			}
			if e := stream.Synchronize(bg); e != nil {
				t.Fatalf("sync: %v", e)
			}
			gq := make([]float32, qDim)
			gkc := make([]float32, cap*kvDim)
			gvc := make([]float32, cap*kvDim)
			for _, p := range []struct {
				h []float32
				b *gc.Buffer[float32]
			}{{gq, dq}, {gkc, dkc}, {gvc, dvc}} {
				if e := gc.CopyDtoH(bg, p.h, p.b); e != nil {
					t.Fatalf("D2H: %v", e)
				}
			}

			const tol = 1e-5
			for i := range wq {
				if math.Abs(float64(gq[i]-wq[i])) > tol {
					t.Fatalf("q[%d] = %v, want %v (head %d, dim %d)", i, gq[i], wq[i], i/c.hd, i%c.hd)
				}
			}
			// The whole cached head at this position — tail included. This is the assertion the
			// pair-only kernel fails: it leaves the tail at the -777 poison.
			for h := 0; h < c.nKV; h++ {
				for d := 0; d < c.hd; d++ {
					o := c.pos*kvDim + h*c.hd + d
					if math.Abs(float64(gkc[o]-wkc[o])) > tol {
						t.Fatalf("kc[pos=%d head=%d dim=%d] = %v, want %v — dim %d of %d is %s",
							c.pos, h, d, gkc[o], wkc[o], d, c.hd,
							map[bool]string{true: "in the ROTATED range", false: "in the un-rotated TAIL, which the kernel must still cache"}[d < 2*rhalf])
					}
					if math.Abs(float64(gvc[o]-wvc[o])) > tol {
						t.Fatalf("vc[pos=%d head=%d dim=%d] = %v, want %v — v is never roped, so every "+
							"dim must be cached verbatim", c.pos, h, d, gvc[o], wvc[o])
					}
				}
			}
			// Vacuity guard: a partial case must actually LEAVE the tail unrotated, or the case is
			// silently testing full rotary and proves nothing about the partial path.
			if c.wantTailUnrot {
				same := true
				for h := 0; h < c.nH; h++ {
					for d := 2 * rhalf; d < c.hd; d++ {
						if gq[h*c.hd+d] != q[h*c.hd+d] {
							same = false
						}
					}
				}
				if !same {
					t.Fatal("tail dims CHANGED — partial rotary must pass [rotaryDim, hd) through untouched")
				}
				if 2*rhalf >= c.hd {
					t.Fatalf("case claims partial but rotaryDim=%d covers head_dim=%d", c.rotaryDim, c.hd)
				}
			}
			t.Logf("nH=%d nKV=%d hd=%d rotaryDim=%d (rhalf=%d, tail=%d/head): q rotated, "+
				"full head cached", c.nH, c.nKV, c.hd, c.rotaryDim, rhalf, tail)
		})
	}
}
