//go:build cuda

package cuda

import (
	"context"
	"math"
	"testing"

	gc "github.com/eitamring/gocudrv/cuda"
)

// Kernel-level gates for cuda/rope_pairwise.cu: the GPT-J PAIRWISE rotary twins of rope_kv,
// rope_kv_batched and rope_kv_mrope_batched (Cohere/Cohere2/Aya, GLM-OCR).
//
// The reference is written from the SEMANTICS in float64 — pair (2d, 2d+1) rotated by
// theta_d = pos*invFreq[d] (times mscale), the first 2*rhalf dims only, the tail passed through and
// still cached, V cached verbatim, Q scaled by qTempScale after the rotation — not transliterated
// from the kernel, and it is the same rule decoder.applyRoPEInterleaved / applyMRoPEPairwise
// implement on the CPU. Each case also runs the NeoX kernel on the same input and requires it to
// DISAGREE with the reference, so a vacuous case (pos 0, identical pairs) cannot pass by accident.

type pwRef struct {
	q, k, kc, vc []float32
}

// pwReference rotates M rows. comp(m, d) gives the position used for row m's frequency d.
func pwReference(q, k, v, invF []float32, nH, nKV, hd, rhalf, M, startPos int, mscale float64,
	comp func(m, d int) int, qTemp func(m int) float64) pwRef {
	qDim, kvDim := nH*hd, nKV*hd
	out := pwRef{
		q:  append([]float32(nil), q...),
		k:  append([]float32(nil), k...),
		kc: make([]float32, (startPos+M)*kvDim),
		vc: make([]float32, (startPos+M)*kvDim),
	}
	rot := func(buf []float32, off, heads, m int, scaleQ float64) {
		for h := range heads {
			base := off + h*hd
			for d := range rhalf {
				ang := float64(comp(m, d)) * float64(invF[d])
				c, s := math.Cos(ang)*mscale, math.Sin(ang)*mscale
				a, b := float64(buf[base+2*d]), float64(buf[base+2*d+1])
				buf[base+2*d] = float32((a*c - b*s) * scaleQ)
				buf[base+2*d+1] = float32((a*s + b*c) * scaleQ)
			}
			// [2*rhalf, hd) deliberately untouched.
		}
	}
	for m := range M {
		rot(out.q, m*qDim, nH, m, qTemp(m))
		rot(out.k, m*kvDim, nKV, m, 1)
		for h := range nKV {
			for d := range hd { // the WHOLE head is cached, rotated or not
				o := (startPos+m)*kvDim + h*hd + d
				out.kc[o] = out.k[m*kvDim+h*hd+d]
				out.vc[o] = v[m*kvDim+h*hd+d]
			}
		}
	}
	_ = qDim
	return out
}

type pwCase struct {
	name        string
	nH, nKV, hd int
	rotaryDim   int
	startPos    int
	ropePos     int // decode only: rotation position (pos != ropePos exercises the m-RoPE decode split)
	M           int
	mscale      float32
	qTempScale  float32 // decode: scalar; batched: row m scale is qTempScale^(m+1) via a table
}

func pwInputs(c pwCase) (q, k, v, invF []float32) {
	var seed uint32 = 91821
	rnd := func() float32 {
		seed = seed*1664525 + 1013904223
		return float32(int32(seed>>8)%2000-1000) / 500
	}
	q = make([]float32, c.M*c.nH*c.hd)
	k = make([]float32, c.M*c.nKV*c.hd)
	v = make([]float32, c.M*c.nKV*c.hd)
	for i := range q {
		q[i] = rnd()
	}
	for i := range k {
		k[i] = rnd()
		v[i] = rnd()
	}
	rhalf := c.rotaryDim / 2
	invF = make([]float32, c.hd/2)
	for d := range invF {
		invF[d] = float32(1.0 / math.Pow(10000, float64(2*d)/float64(c.rotaryDim)))
	}
	_ = rhalf
	return
}

func pwTable() []pwCase {
	return []pwCase{
		{"full-rotary", 4, 2, 16, 16, 5, 11, 1, 1, 1},
		{"partial-half", 4, 2, 16, 8, 7, 7, 1, 1, 1},
		{"mscale+qtemp", 4, 2, 16, 16, 9, 9, 1, 0.85, 1.3},
		{"ropePos-ne-pos", 2, 1, 8, 8, 5, 12, 1, 1, 1},
		{"partial-6of16/odd-rhalf", 4, 4, 16, 6, 3, 3, 1, 1, 1},
	}
}

func pwMaxDiff(a, b []float32) (float64, int) {
	var mx float64
	at := -1
	for i := range a {
		if d := math.Abs(float64(a[i] - b[i])); d > mx || math.IsNaN(d) {
			mx, at = d, i
		}
	}
	return mx, at
}

func pwModules(t *testing.T) (*gc.Context, *gc.Stream, *gc.Module, *gc.Module, *gc.Module, *gc.Module) {
	t.Helper()
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
	t.Cleanup(func() { cx.Close() })
	load := func(ptx []byte, what string) *gc.Module {
		m, err := cx.LoadModule(ptx)
		if err != nil {
			t.Fatalf("LoadModule(%s): %v", what, err)
		}
		return m
	}
	return cx, mustStream(t, cx), load(ropePairwisePTX, "rope_pairwise"), load(gemvFwdPTX, "gemv_fwd"),
		load(prefillBatchedPTX, "prefill_batched"), load(ropeMRopePrefillPTX, "rope_mrope_prefill")
}

type pwBufs struct {
	q, k, v, invF, kc, vc *gc.Buffer[float32]
}

func pwUpload(t *testing.T, cx *gc.Context, q, k, v, invF []float32, cacheRows, kvDim int) pwBufs {
	t.Helper()
	bg := context.Background()
	b := pwBufs{
		q: mustAlloc[float32](t, cx, len(q)), k: mustAlloc[float32](t, cx, len(k)), v: mustAlloc[float32](t, cx, len(v)),
		invF: mustAlloc[float32](t, cx, len(invF)),
		kc:   mustAlloc[float32](t, cx, cacheRows*kvDim), vc: mustAlloc[float32](t, cx, cacheRows*kvDim),
	}
	poison := make([]float32, cacheRows*kvDim)
	for i := range poison {
		poison[i] = -777 // a tail or V the kernel forgets to store stays detectable
	}
	for _, p := range []struct {
		b *gc.Buffer[float32]
		h []float32
	}{{b.q, q}, {b.k, k}, {b.v, v}, {b.invF, invF}, {b.kc, poison}, {b.vc, poison}} {
		if e := gc.CopyHtoD(bg, p.b, p.h); e != nil {
			t.Fatalf("H2D: %v", e)
		}
	}
	t.Cleanup(func() {
		for _, x := range []*gc.Buffer[float32]{b.q, b.k, b.v, b.invF, b.kc, b.vc} {
			x.Close()
		}
	})
	return b
}

func pwDownload(t *testing.T, b pwBufs, nq, nk, ncache int) pwRef {
	t.Helper()
	bg := context.Background()
	o := pwRef{q: make([]float32, nq), k: make([]float32, nk), kc: make([]float32, ncache), vc: make([]float32, ncache)}
	for _, p := range []struct {
		h []float32
		b *gc.Buffer[float32]
	}{{o.q, b.q}, {o.k, b.k}, {o.kc, b.kc}, {o.vc, b.vc}} {
		if e := gc.CopyDtoH(bg, p.h, p.b); e != nil {
			t.Fatalf("D2H: %v", e)
		}
	}
	return o
}

func pwAssert(t *testing.T, label string, got, want pwRef) {
	t.Helper()
	const tol = 2e-5
	for _, p := range []struct {
		n    string
		g, w []float32
	}{{"q", got.q, want.q}, {"k", got.k, want.k}, {"kcache", got.kc, want.kc}, {"vcache", got.vc, want.vc}} {
		if d, at := pwMaxDiff(p.g, p.w); d > tol || at >= 0 && math.IsNaN(d) {
			t.Fatalf("%s: %s[%d] = %v, want %v (max|diff| %.3g > %.0e)", label, p.n, at, p.g[at], p.w[at], d, tol)
		}
	}
}

// TestRopePairwiseKernels_decode: rope_kv_pw against the semantic reference, over full and partial
// rotary, a YaRN mscale, a Q-temperature scale and ropePos != pos; and the NeoX rope_kv on the same
// input must DISAGREE with it (the discrimination control).
func TestRopePairwiseKernels_decode(t *testing.T) {
	cx, stream, pw, gemv, _, _ := pwModules(t)
	bg := context.Background()
	fnPW, err := pw.Function("rope_kv_pw")
	if err != nil {
		t.Fatal(err)
	}
	fnNeoX, err := gemv.Function("rope_kv")
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range pwTable() {
		t.Run(c.name, func(t *testing.T) {
			rhalf := c.rotaryDim / 2
			kvDim := c.nKV * c.hd
			q, k, v, invF := pwInputs(c)
			want := pwReference(q, k, v, invF, c.nH, c.nKV, c.hd, rhalf, 1, c.startPos, float64(c.mscale),
				func(_, _ int) int { return c.ropePos }, func(int) float64 { return float64(c.qTempScale) })
			tail := c.hd - 2*rhalf
			n := c.nH*rhalf + c.nKV*rhalf + c.nKV*tail
			run := func(fn *gc.Function) pwRef {
				b := pwUpload(t, cx, q, k, v, invF, c.startPos+1, kvDim)
				if e := fn.LaunchOn(bg, stream, gc.LaunchConfig1D(n, 256),
					gc.Arg(b.q), gc.Arg(b.k), gc.Arg(b.v), gc.Arg(b.invF), gc.Arg(b.kc), gc.Arg(b.vc),
					gc.ArgValue(int32(c.nH)), gc.ArgValue(int32(c.nKV)), gc.ArgValue(int32(c.hd)),
					gc.ArgValue(int32(c.startPos)), gc.ArgValue(int32(c.ropePos)), gc.ArgValue(int32(rhalf)),
					gc.ArgValue(c.mscale), gc.ArgValue(c.qTempScale)); e != nil {
					t.Fatalf("launch: %v", e)
				}
				if e := stream.Synchronize(bg); e != nil {
					t.Fatalf("sync: %v", e)
				}
				return pwDownload(t, b, len(q), len(k), (c.startPos+1)*kvDim)
			}
			got := run(fnPW)
			// The reference only fills row startPos of the cache; mask the rest of the (poisoned) cache.
			maskRows(&got, &want, c.startPos, kvDim)
			pwAssert(t, "rope_kv_pw", got, want)
			// Discrimination control: NeoX must be visibly wrong whenever the rotation is not the identity.
			if c.ropePos != 0 {
				neox := run(fnNeoX)
				maskRows(&neox, &want, c.startPos, kvDim)
				if d, _ := pwMaxDiff(neox.q, want.q); d < 1e-2 {
					t.Fatalf("control: the NeoX rope_kv agrees with the pairwise reference (max|diff| %.3g) — the case does not discriminate", d)
				} else {
					t.Logf("pairwise kernel == reference; NeoX rope_kv differs from it by %.3g (control)", d)
				}
			}
		})
	}
}

// maskRows zeroes every cache row but `row` in both sides, so the poison in rows the kernel never
// writes does not count (decode writes exactly one).
func maskRows(got, want *pwRef, row, kvDim int) {
	for _, r := range []*pwRef{got, want} {
		for i := range r.kc {
			if i < row*kvDim || i >= (row+1)*kvDim {
				r.kc[i], r.vc[i] = 0, 0
			}
		}
	}
}

// TestRopePairwiseKernels_batched: rope_kv_batched_pw over M rows at startPos+m, with and without the
// per-row Q-temperature table, against the reference; NeoX rope_kv_batched as the control.
func TestRopePairwiseKernels_batched(t *testing.T) {
	cx, stream, pw, _, pb, _ := pwModules(t)
	bg := context.Background()
	fnPW, err := pw.Function("rope_kv_batched_pw")
	if err != nil {
		t.Fatal(err)
	}
	fnNeoX, err := pb.Function("rope_kv_batched")
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range pwTable() {
		for _, withTemp := range []bool{false, true} {
			name := c.name
			if withTemp {
				name += "/qTempRows"
			}
			t.Run(name, func(t *testing.T) {
				c.M = 37 // >= 32 positions, past any single block's worth of rows
				rhalf := c.rotaryDim / 2
				kvDim := c.nKV * c.hd
				q, k, v, invF := pwInputs(c)
				temps := make([]float32, c.M)
				for m := range temps {
					temps[m] = 1
					if withTemp {
						temps[m] = 1 + 0.01*float32(m)
					}
				}
				want := pwReference(q, k, v, invF, c.nH, c.nKV, c.hd, rhalf, c.M, c.startPos, float64(c.mscale),
					func(m, _ int) int { return c.startPos + m }, func(m int) float64 { return float64(temps[m]) })
				tail := c.hd - 2*rhalf
				n := c.nH*rhalf + c.nKV*rhalf + c.nKV*tail
				tb := mustAlloc[float32](t, cx, c.M)
				defer tb.Close()
				if e := gc.CopyHtoD(bg, tb, temps); e != nil {
					t.Fatal(e)
				}
				run := func(fn *gc.Function) pwRef {
					b := pwUpload(t, cx, q, k, v, invF, c.startPos+c.M, kvDim)
					tbArg := gc.ArgValue(uint64(0)) // NULL: no attention temperature
					if withTemp {
						tbArg = gc.Arg(tb)
					}
					cfg := gc.LaunchConfig{GridX: uint32((n + 255) / 256), GridY: uint32(c.M), GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1}
					if e := fn.LaunchOn(bg, stream, cfg,
						gc.Arg(b.q), gc.Arg(b.k), gc.Arg(b.v), gc.Arg(b.invF), gc.Arg(b.kc), gc.Arg(b.vc),
						gc.ArgValue(int32(c.nH)), gc.ArgValue(int32(c.nKV)), gc.ArgValue(int32(c.hd)),
						gc.ArgValue(int32(c.startPos)), gc.ArgValue(int32(rhalf)), gc.ArgValue(int32(c.M)),
						gc.ArgValue(c.mscale), tbArg); e != nil {
						t.Fatalf("launch: %v", e)
					}
					if e := stream.Synchronize(bg); e != nil {
						t.Fatalf("sync: %v", e)
					}
					return pwDownload(t, b, len(q), len(k), (c.startPos+c.M)*kvDim)
				}
				got := run(fnPW)
				// Rows [0, startPos) of the cache are never written by a batch; mask them.
				for _, r := range []*pwRef{&got, &want} {
					for i := 0; i < c.startPos*kvDim; i++ {
						r.kc[i], r.vc[i] = 0, 0
					}
				}
				pwAssert(t, "rope_kv_batched_pw", got, want)
				neox := run(fnNeoX)
				if d, _ := pwMaxDiff(neox.q, want.q); d < 1e-2 {
					t.Fatalf("control: NeoX rope_kv_batched agrees with the pairwise reference (max|diff| %.3g)", d)
				}
			})
		}
	}
}

// TestRopePairwiseKernels_mrope: rope_kv_mrope_batched_pw against the contiguous-section pairwise
// reference (decoder.applyMRoPEPairwise's rule) with distinct per-row (t,h,w) triples, then the
// degenerate case: with t==h==w it must be BIT-IDENTICAL to rope_kv_batched_pw — the kernel-level
// statement of "a text token reduces exactly to the scalar rotation".
func TestRopePairwiseKernels_mrope(t *testing.T) {
	cx, stream, pw, _, _, mrope := pwModules(t)
	bg := context.Background()
	fnMR, err := pw.Function("rope_kv_mrope_batched_pw")
	if err != nil {
		t.Fatal(err)
	}
	fnSc, err := pw.Function("rope_kv_batched_pw")
	if err != nil {
		t.Fatal(err)
	}
	fnNeoX, err := mrope.Function("rope_kv_mrope_batched")
	if err != nil {
		t.Fatal(err)
	}
	c := pwCase{"glm-ocr-shape", 4, 2, 16, 16, 3, 0, 33, 1, 1}
	rhalf := c.rotaryDim / 2
	sec := [3]int{2, 3, 3} // sums to rhalf = 8, GLM-OCR's [16 24 24] shape in miniature
	sec0, sec1 := int32(sec[0]), int32(sec[0]+sec[1])
	kvDim := c.nKV * c.hd
	q, k, v, invF := pwInputs(c)
	// Image-style positions: a t=const block with an (h,w) grid, then text resuming compressed.
	pT, pH, pW := make([]int32, c.M), make([]int32, c.M), make([]int32, c.M)
	for m := range c.M {
		switch {
		case m < 9:
			pT[m], pH[m], pW[m] = 4, int32(4+m/3), int32(4+m%3)
		default:
			p := int32(4 + 3 + (m - 9))
			pT[m], pH[m], pW[m] = p, p, p
		}
	}
	posOf := func(m, d int) int {
		switch {
		case d < sec[0]:
			return int(pT[m])
		case d < sec[0]+sec[1]:
			return int(pH[m])
		}
		return int(pW[m])
	}
	want := pwReference(q, k, v, invF, c.nH, c.nKV, c.hd, rhalf, c.M, c.startPos, 1, posOf, func(int) float64 { return 1 })
	tail := c.hd - 2*rhalf
	n := c.nH*rhalf + c.nKV*rhalf + c.nKV*tail
	up := func(h []int32) *gc.Buffer[int32] {
		b := mustAlloc[int32](t, cx, len(h))
		t.Cleanup(func() { b.Close() })
		if e := gc.CopyHtoD(bg, b, h); e != nil {
			t.Fatal(e)
		}
		return b
	}
	dT, dH, dW := up(pT), up(pH), up(pW)
	runMR := func(fn *gc.Function, tt, hh, ww *gc.Buffer[int32]) pwRef {
		b := pwUpload(t, cx, q, k, v, invF, c.startPos+c.M, kvDim)
		cfg := gc.LaunchConfig{GridX: uint32((n + 255) / 256), GridY: uint32(c.M), GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1}
		if e := fn.LaunchOn(bg, stream, cfg,
			gc.Arg(b.q), gc.Arg(b.k), gc.Arg(b.v), gc.Arg(b.invF), gc.Arg(b.kc), gc.Arg(b.vc),
			gc.ArgValue(int32(c.nH)), gc.ArgValue(int32(c.nKV)), gc.ArgValue(int32(c.hd)),
			gc.ArgValue(int32(c.startPos)), gc.ArgValue(int32(rhalf)), gc.ArgValue(int32(c.M)), gc.ArgValue(c.mscale),
			gc.Arg(tt), gc.Arg(hh), gc.Arg(ww), gc.ArgValue(sec0), gc.ArgValue(sec1)); e != nil {
			t.Fatalf("launch: %v", e)
		}
		if e := stream.Synchronize(bg); e != nil {
			t.Fatalf("sync: %v", e)
		}
		return pwDownload(t, b, len(q), len(k), (c.startPos+c.M)*kvDim)
	}
	got := runMR(fnMR, dT, dH, dW)
	for _, r := range []*pwRef{&got, &want} {
		for i := 0; i < c.startPos*kvDim; i++ {
			r.kc[i], r.vc[i] = 0, 0
		}
	}
	pwAssert(t, "rope_kv_mrope_batched_pw", got, want)
	neox := runMR(fnNeoX, dT, dH, dW)
	if d, _ := pwMaxDiff(neox.q, want.q); d < 1e-2 {
		t.Fatalf("control: the NeoX m-RoPE kernel agrees with the pairwise reference (max|diff| %.3g)", d)
	} else {
		t.Logf("pairwise m-RoPE == reference; NeoX rope_kv_mrope_batched differs by %.3g (control)", d)
	}

	// Degenerate: t == h == w == startPos+m  =>  bit-identical to the scalar pairwise batched kernel.
	seq := make([]int32, c.M)
	for m := range seq {
		seq[m] = int32(c.startPos + m)
	}
	dS := up(seq)
	a := runMR(fnMR, dS, dS, dS)
	bScalar := func() pwRef {
		b := pwUpload(t, cx, q, k, v, invF, c.startPos+c.M, kvDim)
		cfg := gc.LaunchConfig{GridX: uint32((n + 255) / 256), GridY: uint32(c.M), GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1}
		if e := fnSc.LaunchOn(bg, stream, cfg,
			gc.Arg(b.q), gc.Arg(b.k), gc.Arg(b.v), gc.Arg(b.invF), gc.Arg(b.kc), gc.Arg(b.vc),
			gc.ArgValue(int32(c.nH)), gc.ArgValue(int32(c.nKV)), gc.ArgValue(int32(c.hd)),
			gc.ArgValue(int32(c.startPos)), gc.ArgValue(int32(rhalf)), gc.ArgValue(int32(c.M)), gc.ArgValue(c.mscale),
			gc.ArgValue(uint64(0))); e != nil {
			t.Fatalf("launch: %v", e)
		}
		if e := stream.Synchronize(bg); e != nil {
			t.Fatalf("sync: %v", e)
		}
		return pwDownload(t, b, len(q), len(k), (c.startPos+c.M)*kvDim)
	}()
	for _, p := range []struct {
		n    string
		g, w []float32
	}{{"q", a.q, bScalar.q}, {"k", a.k, bScalar.k}, {"kcache", a.kc[c.startPos*kvDim:], bScalar.kc[c.startPos*kvDim:]}} {
		for i := range p.g {
			if p.g[i] != p.w[i] {
				t.Fatalf("degenerate m-RoPE %s[%d]: mrope_pw=%v scalar_pw=%v — must be bit-identical", p.n, i, p.g[i], p.w[i])
			}
		}
	}
}
