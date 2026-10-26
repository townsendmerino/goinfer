//go:build linux

package gpudev

import (
	"math/rand"
	"testing"
)

// TestCUDA_gemmW8A8Reg gates audit M-12's int8 register-blocked kernel.
//
// Two things, in this order. First that GEMMW8A8Plan ROUTES correctly — the
// kernel has no bounds checks, so a misaligned shape reaching it is
// out-of-bounds device access, and the fixtures the ViT parity tests use
// (hidden 32) are far too small to align, meaning those tests exercise the
// tiled fallback and prove nothing about this kernel. Second that where it does
// run it is BIT-IDENTICAL to gemm_w8a8_tiled: both accumulate an exact int32
// and apply aScale[m]*bScale[n] in the same order, so equality is the right
// assertion, not a tolerance.
func TestCUDA_gemmW8A8Reg(t *testing.T) {
	d, q, v := vitSetup(t)

	// Routing: aligned shapes take the register kernel, everything else must
	// fall back to the bounds-checked tiled one.
	for _, c := range []struct {
		M, N, K int
		wantReg bool
	}{
		{64, 64, 16, true},
		{128, 256, 1152, true},
		{64, 4304, 1152, true}, // N%64!=0 is fine now: edge tiles stage zeros (so400m MLP)
		{64, 64, 1152, true},
		{63, 64, 16, true}, // M edge tile
		{64, 63, 16, true}, // N edge tile
		{64, 64, 15, false},
		{0, 64, 16, false},
	} {
		p, _ := v.GEMMW8A8Plan(c.M, c.N, c.K)
		if gotReg := p == v.GEMMW8A8Reg; gotReg != c.wantReg {
			t.Fatalf("GEMMW8A8Plan(%d,%d,%d) picked reg=%v, want %v — a misaligned shape on the unchecked kernel is out-of-bounds access",
				c.M, c.N, c.K, gotReg, c.wantReg)
		}
	}

	// Equality against the tiled kernel, on shapes the register kernel takes.
	rng := rand.New(rand.NewSource(99))
	for _, sh := range []struct{ M, N, K int }{
		{64, 64, 16},
		{64, 64, 1152},
		{128, 192, 512},
		{256, 64, 4304},   // K = the so400m intermediate
		{63, 65, 512},     // both edges ragged
		{100, 4304, 1152}, // the so400m MLP shape, M and N both ragged
	} {
		A := make([]int8, sh.M*sh.K)
		B := make([]int8, sh.N*sh.K)
		for i := range A {
			A[i] = int8(rng.Intn(255) - 127)
		}
		for i := range B {
			B[i] = int8(rng.Intn(255) - 127)
		}
		as, bs := randF32(rng, sh.M, 0.01), randF32(rng, sh.N, 0.01)
		dA, dB := NewBufferOf(d, A), NewBufferOf(d, B)
		dAs, dBs := NewBufferOf(d, as), NewBufferOf(d, bs)
		cReg := NewBufferLenOf[float32](d, sh.M*sh.N)
		cTiled := NewBufferLenOf[float32](d, sh.M*sh.N)

		p, cfg := v.GEMMW8A8Plan(sh.M, sh.N, sh.K)
		if p != v.GEMMW8A8Reg {
			t.Fatalf("shape %v did not route to the register kernel", sh)
		}
		if err := q.Launch(p, cfg,
			Arg(dA), Arg(dAs), Arg(dB), Arg(dBs), Arg(cReg),
			ArgValue(int32(sh.M)), ArgValue(int32(sh.N)), ArgValue(int32(sh.K))); err != nil {
			t.Fatalf("reg: %v", err)
		}
		if err := q.Launch(v.GEMMW8A8Tiled, TileGrid(sh.M, sh.N),
			Arg(dA), Arg(dAs), Arg(dB), Arg(dBs), Arg(cTiled),
			ArgValue(int32(sh.M)), ArgValue(int32(sh.N)), ArgValue(int32(sh.K))); err != nil {
			t.Fatalf("tiled: %v", err)
		}
		if err := q.Sync(); err != nil {
			t.Fatalf("Sync: %v", err)
		}
		gr, gt := make([]float32, sh.M*sh.N), make([]float32, sh.M*sh.N)
		if err := Download(cReg, gr); err != nil {
			t.Fatalf("Download: %v", err)
		}
		if err := Download(cTiled, gt); err != nil {
			t.Fatalf("Download: %v", err)
		}
		for i := range gr {
			if gr[i] != gt[i] {
				t.Fatalf("shape %v elem %d: reg %v != tiled %v", sh, i, gr[i], gt[i])
			}
		}
		// A kernel that wrote nothing would also "match" an all-zero tiled
		// result on a degenerate input; check it actually computed something.
		nonzero := false
		for _, x := range gr {
			if x != 0 {
				nonzero = true
				break
			}
		}
		if !nonzero {
			t.Fatalf("shape %v: register kernel produced all zeros", sh)
		}
	}
}
