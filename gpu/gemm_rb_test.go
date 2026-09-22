//go:build gpu

package gpu

import (
	"math"
	"testing"
	"time"

	"github.com/townsendmerino/aikit/linalg"
)

// rbShapes covers aligned tiles, ragged edges in every dimension, and the real 1.5B/0.5B
// projection shapes: (M, N, K).
var rbShapes = [][3]int{
	{37, 100, 192},    // ragged M, N; K a multiple of 64 but not of 256
	{64, 64, 64},      // one tile exactly
	{65, 65, 68},      // one past a tile in M and N; K not a multiple of 64 (17 words)
	{256, 1536, 1536}, // 1.5B hidden→hidden (o_proj, q)
	{128, 8960, 1536}, // 1.5B gate/up
	{128, 1536, 8960}, // 1.5B down
	{300, 256, 896},   // 0.5B kv (small N) with ragged M
	{512, 2048, 1536}, // 1.5B qkv-ish
}

// TestTiledRB64_bitIdentical pins the R10 kernel's promise: on every shape, every output
// float is bit-identical (Float32bits) to the 16×16 kernel it replaces, in both the plain
// and the bias forms, at this context's dot4 flavour. The exact i32 K-sum makes this a
// construction property; the test is what stops a future edit from quietly breaking it.
func TestTiledRB64_bitIdentical(t *testing.T) {
	ctx := newOrSkipHW(t)
	defer ctx.Close()
	old, new := matmulTiledW8A8ShaderWGSL, matmulRB64W8A8ShaderWGSL
	if ctx.hasDP4A {
		old, new = matmulTiledW8A8DP4AShaderWGSL, matmulRB64W8A8DP4AShaderWGSL
	}
	for _, s := range rbShapes {
		M, N, K := s[0], s[1], s[2]
		weight := randMat(N*K, uint64(N*K)+1)
		bq, bScales := linalg.QuantizeRowsInt8(weight, N, K)
		rm, err := ctx.UploadW8A8(bq, bScales, N, K)
		if err != nil {
			t.Fatalf("%v UploadW8A8: %v", s, err)
		}
		act := randMat(M*K, uint64(M*K)+7)
		aq, aScales := linalg.QuantizeRowsInt8(act, M, K)
		want, err := runTiledKernelTile(ctx, old, 16, aq, aScales, rm, M)
		if err != nil {
			t.Fatalf("%v tiled16: %v", s, err)
		}
		got, err := runTiledKernelTile(ctx, new, 64, aq, aScales, rm, M)
		if err != nil {
			t.Fatalf("%v rb64: %v", s, err)
		}
		if d := firstBitDiff(want, got); d >= 0 {
			t.Fatalf("%v: NOT bit-identical at element %d (row %d col %d): tiled16 %08x vs rb64 %08x",
				s, d, d/N, d%N, math.Float32bits(want[d]), math.Float32bits(got[d]))
		}
		// The production entry point, at the active tile (64 by default), against the same reference.
		if ctx.GEMMTile() == 64 {
			prod, err := ctx.MatmulW8A8Tiled(aq, aScales, rm, M)
			if err != nil {
				t.Fatalf("%v MatmulW8A8Tiled: %v", s, err)
			}
			if d := firstBitDiff(want, prod); d >= 0 {
				t.Fatalf("%v: production path differs at element %d", s, d)
			}
		}
		rm.Release()
	}
	t.Logf("bit-identical on %d shapes (dp4a=%v)", len(rbShapes), ctx.hasDP4A)
}

func firstBitDiff(a, b []float32) int {
	for i := range a {
		if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
			return i
		}
	}
	return -1
}

// TestTiledRB64_microbench is the kernel-level signal, at the 1.5B shapes the profile put
// the class at ~975 GFLOPS: the two kernels back to back, same buffers, alternating.
func TestTiledRB64_microbench(t *testing.T) {
	if testing.Short() {
		t.Skip("microbench")
	}
	ctx := newOrSkipHW(t)
	defer ctx.Close()
	old, new := matmulTiledW8A8ShaderWGSL, matmulRB64W8A8ShaderWGSL
	if ctx.hasDP4A {
		old, new = matmulTiledW8A8DP4AShaderWGSL, matmulRB64W8A8DP4AShaderWGSL
	}
	shapes := [][3]int{{512, 1536, 1536}, {512, 8960, 1536}, {512, 1536, 8960}, {1024, 8960, 896}}
	for _, s := range shapes {
		M, N, K := s[0], s[1], s[2]
		weight := randMat(N*K, 3)
		bq, bScales := linalg.QuantizeRowsInt8(weight, N, K)
		rm, err := ctx.UploadW8A8(bq, bScales, N, K)
		if err != nil {
			t.Fatalf("UploadW8A8: %v", err)
		}
		act := randMat(M*K, 9)
		aq, aScales := linalg.QuantizeRowsInt8(act, M, K)
		for range 2 { // warm-up both
			runTiledKernelTile(ctx, old, 16, aq, aScales, rm, M)
			runTiledKernelTile(ctx, new, 64, aq, aScales, rm, M)
		}
		const iters = 5
		var dOld, dNew time.Duration
		for i := 0; i < iters; i++ {
			t0 := time.Now()
			if _, err := runTiledKernelTile(ctx, old, 16, aq, aScales, rm, M); err != nil {
				t.Fatal(err)
			}
			dOld += time.Since(t0)
			t1 := time.Now()
			if _, err := runTiledKernelTile(ctx, new, 64, aq, aScales, rm, M); err != nil {
				t.Fatal(err)
			}
			dNew += time.Since(t1)
		}
		gf := func(d time.Duration) float64 { return float64(2*M*N*K) * iters / d.Seconds() / 1e9 }
		t.Logf("M=%d N=%d K=%d: tiled16 %.2f ms (%.0f GFLOPS) | rb64 %.2f ms (%.0f GFLOPS) | %.2fx  (wall incl. upload+readback)",
			M, N, K, float64(dOld.Milliseconds())/iters, gf(dOld), float64(dNew.Milliseconds())/iters, gf(dNew), dOld.Seconds()/dNew.Seconds())
		rm.Release()
	}
}
