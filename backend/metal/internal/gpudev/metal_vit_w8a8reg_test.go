//go:build darwin

package gpudev

import (
	"math/rand"
	"testing"
)

// TestMetal_gemmW8A8Reg gates the Metal port of CUDA's audit-M-12 register-blocked int8
// kernel (M-15, audit-metal-2026-09-12.md). Mirrors cuda_vit_w8a8reg_test.go's structure
// exactly, adapted to Metal's Run2D geometry.
//
// Two things, in this order. First that GEMMW8A8Plan ROUTES correctly — the kernel has no
// bounds checks, so a misaligned shape reaching it is out-of-bounds device access, and the
// fixtures the ViT parity tests use are far too small to align, meaning those tests
// exercise the tiled fallback and prove nothing about this kernel. Second that where it
// does run it is BIT-IDENTICAL to gemm_w8a8_tiled: both accumulate an exact int32 and
// apply aScale[m]*bScale[n] in the same order, so equality is the right assertion, not a
// tolerance.
func TestMetal_gemmW8A8Reg(t *testing.T) {
	d, q, v := vitSetupM(t)

	// Routing: aligned shapes take the register kernel, everything else must fall back to
	// the bounds-checked tiled one.
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
		p, _, _, _, _ := v.GEMMW8A8Plan(c.M, c.N, c.K)
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
		as, bs := randF32M(rng, sh.M, 0.01), randF32M(rng, sh.N, 0.01)
		dA, dB := NewBufferOf(d, A), NewBufferOf(d, B)
		dAs, dBs := NewBufferOf(d, as), NewBufferOf(d, bs)
		cReg := d.NewBufferLen(sh.M * sh.N)
		cTiled := d.NewBufferLen(sh.M * sh.N)

		p, gx, gy, tgx, tgy := v.GEMMW8A8Plan(sh.M, sh.N, sh.K)
		if p != v.GEMMW8A8Reg {
			t.Fatalf("shape %v did not route to the register kernel", sh)
		}
		run2d(q, p, gx, gy, tgx, tgy, dA, dAs, dB, dBs, cReg, i32b(d, sh.M), i32b(d, sh.N), i32b(d, sh.K))
		tgx2, tgy2, tgtx, tgty := TileDims(sh.M, sh.N)
		run2d(q, v.GEMMW8A8Tiled, tgx2, tgy2, tgtx, tgty, dA, dAs, dB, dBs, cTiled, i32b(d, sh.M), i32b(d, sh.N), i32b(d, sh.K))

		gr, gt := cReg.Floats(), cTiled.Floats()
		for i := range gr {
			if gr[i] != gt[i] {
				t.Fatalf("shape %v elem %d: reg %v != tiled %v", sh, i, gr[i], gt[i])
			}
		}
		// A kernel that wrote nothing would also "match" an all-zero tiled result on a
		// degenerate input; check it actually computed something.
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
