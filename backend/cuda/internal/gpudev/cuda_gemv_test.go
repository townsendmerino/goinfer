//go:build linux

package gpudev

import (
	"math"
	"testing"
)

// cuda_gemv_test.go gates the tuned quantized GEMVs lifted from goinfer's decode
// path (the Phase-1b blob-split). The reference is an exact integer reimplementation
// of each kernel's arithmetic: both accumulate int8 products in int32 via __dp4a, so
// the dot itself is EXACT and only the final f32 rescale can round. That makes these
// gates sharp — a packing, permutation or reduction bug cannot hide behind "close
// enough".
//
// Two properties beyond plain correctness are gated here because goinfer's decode
// depends on them and a silent regression would be a numeric bug in a shipped model:
// the `accum` epilogue (dst += val, which absorbs the residual add), and the
// optional-bias null bind (ArgNull, since an absent bias is a real null pointer).

// dp4 is one __dp4a: four int8 products of two packed words, summed into acc.
func dp4(x, y int32) int32 {
	var s int32
	for b := range 4 {
		s += int32(int8(x>>(8*b))) * int32(int8(y>>(8*b)))
	}
	return s
}

// lcg is a deterministic byte source, so a failure reproduces exactly.
type lcg uint32

func (r *lcg) next() int32 { *r = (*r)*1664525 + 1013904223; return int32(int8(*r >> 24)) }
func (r *lcg) word() int32 {
	return (r.next()&0xff)<<0 | (r.next()&0xff)<<8 | (r.next()&0xff)<<16 | (r.next()&0xff)<<24
}

// TestCUDA_gemvW8A8_exact gates the W8A8 GEMV against an exact int32 reference over
// shapes that straddle the warps-per-block boundary (so the grid overhangs N and the
// row bounds-check is exercised) and K values that are not multiples of the unroll.
func TestCUDA_gemvW8A8_exact(t *testing.T) {
	d, q, _ := setup(t, "vadd") // device + queue; pipelines come from NewQuantGEMV
	g, err := d.NewQuantGEMV()
	if err != nil {
		t.Fatalf("NewQuantGEMV: %v", err)
	}

	for _, shape := range []struct{ N, K int }{
		{1, 4},      // single row, single word
		{7, 128},    // fewer rows than one block's warps
		{8, 64},     // exactly one block
		{9, 32},     // one row past a block
		{129, 1536}, // realistic projection shape, N ∤ 8
	} {
		if shape.K%4 != 0 {
			t.Fatalf("bad test shape: K=%d must be a multiple of 4", shape.K)
		}
		Kdiv4 := shape.K / 4
		rng := lcg(12345)
		W := make([]int32, shape.N*Kdiv4)
		a := make([]int32, Kdiv4)
		wScale := make([]float32, shape.N)
		bias := make([]float32, shape.N)
		for i := range W {
			W[i] = rng.word()
		}
		for i := range a {
			a[i] = rng.word()
		}
		for i := range wScale {
			wScale[i] = 0.01 + float32(i%7)*0.001
			bias[i] = float32(i%5) * 0.5
		}
		const aScale = float32(0.02)

		dW := NewBufferOf(d, W)
		dA := NewBufferOf(d, a)
		dWS := NewBufferOf(d, wScale)
		dAS := NewBufferOf(d, []float32{aScale})
		dBias := NewBufferOf(d, bias)
		dst := NewBufferLenOf[float32](d, shape.N)

		// exact int reference
		ref := make([]float32, shape.N)
		for n := range shape.N {
			var acc int32
			for k := range Kdiv4 {
				acc += dp4(W[n*Kdiv4+k], a[k])
			}
			ref[n] = float32(acc)*wScale[n]*aScale + bias[n]
		}

		cfg := GEMVGrid(shape.N, GEMVWarpsPerBlock)
		if err := q.Launch(g.W8A8, cfg,
			Arg(dW), Arg(dA), Arg(dWS), Arg(dAS), Arg(dBias),
			ArgValue(int32(shape.N)), ArgValue(int32(Kdiv4)), Arg(dst), ArgValue(int32(0))); err != nil {
			t.Fatalf("N=%d K=%d Launch: %v", shape.N, shape.K, err)
		}
		if err := q.Sync(); err != nil {
			t.Fatalf("Sync: %v", err)
		}
		got := make([]float32, shape.N)
		if err := Download(dst, got); err != nil {
			t.Fatalf("Download: %v", err)
		}
		for n := range shape.N {
			if d := math.Abs(float64(got[n] - ref[n])); d > 1e-3 {
				t.Fatalf("N=%d K=%d row %d: got %v want %v (Δ %.3g)", shape.N, shape.K, n, got[n], ref[n], d)
			}
		}

		// accum epilogue: a second launch with accum=1 must DOUBLE every row (dst
		// already holds ref). This is the residual-add fusion goinfer relies on.
		if err := q.Launch(g.W8A8, cfg,
			Arg(dW), Arg(dA), Arg(dWS), Arg(dAS), Arg(dBias),
			ArgValue(int32(shape.N)), ArgValue(int32(Kdiv4)), Arg(dst), ArgValue(int32(1))); err != nil {
			t.Fatalf("accum Launch: %v", err)
		}
		if err := q.Sync(); err != nil {
			t.Fatalf("Sync: %v", err)
		}
		acc := make([]float32, shape.N)
		if err := Download(dst, acc); err != nil {
			t.Fatalf("Download: %v", err)
		}
		for n := range shape.N {
			if d := math.Abs(float64(acc[n] - 2*ref[n])); d > 2e-3 {
				t.Fatalf("N=%d row %d accum: got %v want %v", shape.N, n, acc[n], 2*ref[n])
			}
		}

		for _, b := range []Buffer{dW, dA, dWS, dAS, dBias, dst} {
			d.ReleaseBuf(b)
		}
	}
	t.Log("W8A8 GEMV ≡ exact int32 reference across 5 shapes; accum epilogue doubles")
}

// TestCUDA_gemvW8A8_nullBias gates the optional-bias bind: with ArgNull the kernel's
// `bias ? bias[n] : 0.f` guard must take the null branch and add nothing. Binding a
// zero Buffer instead cannot express this (gocudrv rejects it), which is exactly why
// ArgNull exists — so this is the test that the two paths agree.
func TestCUDA_gemvW8A8_nullBias(t *testing.T) {
	d, q, _ := setup(t, "vadd")
	g, err := d.NewQuantGEMV()
	if err != nil {
		t.Fatalf("NewQuantGEMV: %v", err)
	}
	const N, Kdiv4 = 16, 8
	rng := lcg(999)
	W := make([]int32, N*Kdiv4)
	a := make([]int32, Kdiv4)
	for i := range W {
		W[i] = rng.word()
	}
	for i := range a {
		a[i] = rng.word()
	}
	wScale := make([]float32, N)
	for i := range wScale {
		wScale[i] = 0.05
	}
	zeros := make([]float32, N)

	dW, dA, dWS := NewBufferOf(d, W), NewBufferOf(d, a), NewBufferOf(d, wScale)
	dAS := NewBufferOf(d, []float32{0.1})
	dZero := NewBufferOf(d, zeros)
	dstNull := NewBufferLenOf[float32](d, N)
	dstZero := NewBufferLenOf[float32](d, N)
	cfg := GEMVGrid(N, GEMVWarpsPerBlock)

	run := func(bias KernelArg, dst Buffer) {
		t.Helper()
		if err := q.Launch(g.W8A8, cfg, Arg(dW), Arg(dA), Arg(dWS), Arg(dAS), bias,
			ArgValue(int32(N)), ArgValue(int32(Kdiv4)), Arg(dst), ArgValue(int32(0))); err != nil {
			t.Fatalf("Launch: %v", err)
		}
		if err := q.Sync(); err != nil {
			t.Fatalf("Sync: %v", err)
		}
	}
	run(ArgNull(), dstNull)  // absent bias
	run(Arg(dZero), dstZero) // explicit all-zero bias

	gotNull := make([]float32, N)
	gotZero := make([]float32, N)
	if err := Download(dstNull, gotNull); err != nil {
		t.Fatalf("Download: %v", err)
	}
	if err := Download(dstZero, gotZero); err != nil {
		t.Fatalf("Download: %v", err)
	}
	nonzero := false
	for n := range N {
		if gotNull[n] != gotZero[n] {
			t.Fatalf("row %d: null-bias %v != zero-bias %v — the null guard is not taking the right branch", n, gotNull[n], gotZero[n])
		}
		if gotNull[n] != 0 {
			nonzero = true
		}
	}
	if !nonzero {
		t.Error("vacuous: every output was zero, so null-vs-zero bias agreement proves nothing")
	}
	t.Log("ArgNull bias ≡ explicit zero bias, on non-trivial output")
}

// f32tof16 encodes an IEEE float32 into a float16 bit pattern (round-to-nearest-even,
// simple) — the same encoder goinfer packs group scales with.
func f32tof16(f float32) uint16 {
	b := math.Float32bits(f)
	s := uint16((b >> 16) & 0x8000)
	e := int32((b>>23)&0xff) - 127 + 15
	m := b & 0x7fffff
	if e <= 0 {
		return s
	}
	if e >= 0x1f {
		return s | 0x7c00
	}
	return s | uint16(e<<10) | uint16(m>>13)
}

// vsub4 is __vsub4: independent per-BYTE subtraction, no borrow across lanes.
func vsub4(x, y uint32) int32 {
	var o uint32
	for b := range 4 {
		o |= uint32(byte(x>>(8*b))-byte(y>>(8*b))) << (8 * b)
	}
	return int32(o)
}

// TestCUDA_gemvW4A8_exact gates the int4-weight GEMV. The weight words are random
// bit patterns interpreted exactly as the kernel interprets them, so the nibble
// permutation is out of scope here (it is a PACK-time concern) and what is gated is
// the kernel's arithmetic: even/odd nibble unpack via __vsub4, __dp4a accumulation,
// per-group f16 rescale, warp reduction, epilogue.
//
// Unlike W8A8, this one cannot be gated bit-exactly: the kernel float-accumulates
// per-word scaled partials warp-strided, then reduces, so its summation ORDER differs
// from a sequential reference. The int partials are exact; only the f32 accumulation
// reassociates. Hence a relative tolerance, and group scales chosen as exact powers of
// two so the f16 round-trip contributes no error of its own.
func TestCUDA_gemvW4A8_exact(t *testing.T) {
	d, q, _ := setup(t, "vadd")
	g, err := d.NewQuantGEMV()
	if err != nil {
		t.Fatalf("NewQuantGEMV: %v", err)
	}
	// Kwords values straddling the 64-word main loop so the 32-stride tail runs.
	for _, shape := range []struct{ N, Kwords int }{
		{1, 4},    // pure tail
		{8, 64},   // exactly one main-loop pass, no tail
		{9, 70},   // main loop + tail, N past a block
		{129, 96}, // realistic, N ∤ 8
	} {
		Kgroups := (shape.Kwords + 3) / 4
		rng := lcg(4242)
		W := make([]uint32, shape.N*shape.Kwords)
		a := make([]int32, 2*shape.Kwords)
		for i := range W {
			W[i] = uint32(rng.word())
		}
		for i := range a {
			a[i] = rng.word()
		}
		// exact-in-f16 powers of two ⇒ no encoding error
		pow2 := []float32{0.03125, 0.0625, 0.125, 0.25}
		gsF := make([]float32, shape.N*Kgroups)
		gs16 := make([]uint16, shape.N*Kgroups)
		for i := range gsF {
			gsF[i] = pow2[i%len(pow2)]
			gs16[i] = f32tof16(gsF[i])
		}
		bias := make([]float32, shape.N)
		for i := range bias {
			bias[i] = float32(i%3) * 0.25
		}
		const aScale = float32(0.02)

		dW := NewBufferOf(d, W)
		dA := NewBufferOf(d, a)
		dGS := NewBufferOf(d, gs16)
		dAS := NewBufferOf(d, []float32{aScale})
		dBias := NewBufferOf(d, bias)
		dst := NewBufferLenOf[float32](d, shape.N)

		ref := make([]float64, shape.N)
		for n := range shape.N {
			var facc float64
			for wi := range shape.Kwords {
				w := W[n*shape.Kwords+wi]
				p := dp4(vsub4(w&0x0F0F0F0F, 0x08080808), a[2*wi])
				p += dp4(vsub4((w>>4)&0x0F0F0F0F, 0x08080808), a[2*wi+1])
				facc += float64(p) * float64(gsF[n*Kgroups+(wi>>2)])
			}
			ref[n] = facc*float64(aScale) + float64(bias[n])
		}

		cfg := GEMVGrid(shape.N, GEMVWarpsPerBlock)
		if err := q.Launch(g.W4A8, cfg,
			Arg(dW), Arg(dA), Arg(dGS), Arg(dAS), Arg(dBias),
			ArgValue(int32(shape.N)), ArgValue(int32(shape.Kwords)), ArgValue(int32(Kgroups)),
			Arg(dst), ArgValue(int32(0))); err != nil {
			t.Fatalf("N=%d Kwords=%d Launch: %v", shape.N, shape.Kwords, err)
		}
		if err := q.Sync(); err != nil {
			t.Fatalf("Sync: %v", err)
		}
		got := make([]float32, shape.N)
		if err := Download(dst, got); err != nil {
			t.Fatalf("Download: %v", err)
		}
		worst := 0.0
		for n := range shape.N {
			den := math.Abs(ref[n])
			if den < 1 {
				den = 1
			}
			if rel := math.Abs(float64(got[n])-ref[n]) / den; rel > worst {
				worst = rel
			}
		}
		if worst > 1e-5 {
			t.Fatalf("N=%d Kwords=%d: worst relative Δ %.3g exceeds 1e-5 (float reassociation cannot explain this)", shape.N, shape.Kwords, worst)
		}
		for _, b := range []Buffer{dW, dA, dGS, dAS, dBias, dst} {
			d.ReleaseBuf(b)
		}
	}
	t.Log("W4A8 GEMV ≡ reference across 4 shapes (main loop + 32-stride tail)")
}
