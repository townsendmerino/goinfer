//go:build gpu && goinfer_testhooks

package gpu

import (
	"math"
	"math/rand"
	"runtime"
	"strings"
	"sync"
	"testing"
)

// mmF32OnDevice runs matmulF32Device on host operands and reads the product back.
func mmF32OnDevice(t *testing.T, c *Context, a, b []float32, M, K, N int) []float32 {
	t.Helper()
	ad, err := c.UploadF32(a)
	if err != nil {
		t.Fatal(err)
	}
	defer ad.Release()
	bd, err := c.UploadF32(b)
	if err != nil {
		t.Fatal(err)
	}
	defer bd.Release()
	out, err := c.matmulF32Device(ad.buf, bd.buf, M, K, N)
	if err != nil {
		t.Fatal(err)
	}
	defer out.Release()
	got, err := c.Readback(out)
	if err != nil {
		t.Fatal(err)
	}
	return got
}

// mmF64Host is a·bᵀ in float64 on the host, rows in parallel.
func mmF64Host(a, b []float32, M, K, N int) []float32 {
	dst := make([]float32, M*N)
	var wg sync.WaitGroup
	workers := runtime.GOMAXPROCS(0)
	for w := range workers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for m := w; m < M; m += workers {
				ar := a[m*K : m*K+K]
				for n := range N {
					br := b[n*K : n*K+K]
					var acc float64
					for k, x := range ar {
						acc += float64(x) * float64(br[k])
					}
					dst[m*N+n] = float32(acc)
				}
			}
		}()
	}
	wg.Wait()
	return dst
}

// worstRel is the largest |got-want| over (|want| + the rms of want): relative, with a floor so a near-zero output of
// a sum of cancelling terms does not read as a large relative error.
func worstRel(got, want []float32) float64 {
	var ss float64
	for _, x := range want {
		ss += float64(x) * float64(x)
	}
	rms := math.Sqrt(ss / float64(len(want)))
	worst := 0.0
	for i := range want {
		d := math.Abs(float64(got[i])-float64(want[i])) / (math.Abs(float64(want[i])) + rms)
		if d > worst || d != d {
			worst = d
			if d != d {
				return math.NaN()
			}
		}
	}
	return worst
}

// TestMatmulTiledF32_matchesNaive is G-S12.3b-1 (docs/tasks/task-multimodal-support-2026-10.md, S12.3b): the tiled
// float32 matmul against the naive kernel, at shapes that are and are not multiples of 16 in each dimension and at the
// SigLIP tower's own, every output within a relative 1e-5; the naive kernel against a float64 host product as the
// control; and, planted, a strip tail read past K, which must be red at a K that is not a multiple of 16.
func TestMatmulTiledF32_matchesNaive(t *testing.T) {
	c, err := New()
	if err != nil {
		t.Skipf("no gpu: %v", err)
	}
	defer c.Close()
	if err := c.ensureVision(); err != nil {
		t.Fatal(err)
	}
	shapes := [][3]int{ // M, K, N
		{16, 16, 16}, {17, 31, 19}, {5, 100, 3}, {64, 72, 64}, {33, 1152, 47}, {130, 72, 130}, {1, 5, 1},
		{4096, 1152, 4304}, // the tower's fc1 over every patch
		{4096, 72, 4096},   // one head's scores
		{4096, 4096, 72},   // scores times V
	}
	if testing.Short() {
		shapes = shapes[:7]
	}
	rng := rand.New(rand.NewSource(3))
	const tol = 1e-5
	for _, sh := range shapes {
		M, K, N := sh[0], sh[1], sh[2]
		a, b := make([]float32, M*K), make([]float32, N*K)
		for i := range a {
			a[i] = rng.Float32()*2 - 1
		}
		for i := range b {
			b[i] = rng.Float32()*2 - 1
		}
		visionNaiveMatmulForTest = true
		naive := mmF32OnDevice(t, c, a, b, M, K, N)
		visionNaiveMatmulForTest = false
		tiled := mmF32OnDevice(t, c, a, b, M, K, N)
		host := mmF64Host(a, b, M, K, N)
		tn, nh := worstRel(tiled, naive), worstRel(naive, host)
		t.Logf("M=%d K=%d N=%d: tiled against naive %.2e, naive against the float64 host product %.2e", M, K, N, tn, nh)
		if !(tn <= tol) {
			t.Errorf("M=%d K=%d N=%d: the tiled kernel differs from the naive one by a relative %.2e, want at most %v", M, K, N, tn, tol)
		}
		if !(nh <= 1e-4) {
			t.Errorf("M=%d K=%d N=%d: the control failed, the naive kernel is a relative %.2e from the float64 host product", M, K, N, nh)
		}
	}
	// Planted: the strip's tail is not zero-filled, so a K that is not a multiple of 16 reads the next row's values.
	code := strings.ReplaceAll(matmulTiledF32WGSL, " && kc < dims.k", "")
	if code == matmulTiledF32WGSL {
		t.Fatal("the planted variant did not change the shader")
	}
	_, pl, ly, err := c.mkPipeline("matmulTiledF32-planted", code)
	if err != nil {
		t.Fatal(err)
	}
	savedPl, savedLy := c.mmTiledF32Pipeline, c.mmTiledF32Layout
	c.mmTiledF32Pipeline, c.mmTiledF32Layout = pl, ly
	defer func() { c.mmTiledF32Pipeline, c.mmTiledF32Layout = savedPl, savedLy }()
	M, K, N := 17, 31, 19
	a, b := make([]float32, M*K), make([]float32, N*K)
	for i := range a {
		a[i] = rng.Float32()*2 - 1
	}
	for i := range b {
		b[i] = rng.Float32()*2 - 1
	}
	planted := worstRel(mmF32OnDevice(t, c, a, b, M, K, N), mmF64Host(a, b, M, K, N))
	t.Logf("planted (no zero fill past K) at K=31: relative %.2e", planted)
	if planted <= tol {
		t.Errorf("the planted defect left the gate green (%.2e)", planted)
	}
}
