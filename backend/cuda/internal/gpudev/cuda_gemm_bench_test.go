//go:build linux

package gpudev

import (
	"fmt"
	"math/rand"
	"testing"
)

// cuda_gemm_bench_test.go is a KERNEL microbenchmark, in the role docs/BENCH-gpu.md
// assigns them: tuning, not headline. Per that methodology it reports STEADY COMPUTE
// only — weights resident, warm, one Sync per iteration — and deliberately excludes
// the one-time costs (context create, PTX JIT, upload) and the H2D/D2H transfer, which
// on CUDA are explicit copies and would otherwise be folded into a single misleading
// "GPU time".
//
// Shapes are the real ViT projections: SigLIP-so400m is hidden=1152, inter=4304 at
// np=4096 patches, which is the fat-GEMM regime the whole native-GPU bet is aimed at.
func benchGEMM(b *testing.B, kind string, M, N, K int) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		b.Skipf("no CUDA device: %v", err)
	}
	defer d.ReleaseObjects()
	v, err := d.NewViT()
	if err != nil {
		b.Fatalf("NewViT: %v", err)
	}
	q := d.NewCommandQueue()
	rng := rand.New(rand.NewSource(5))
	A := make([]int8, M*K)
	Bq := make([]int8, N*K)
	for i := range A {
		A[i] = int8(rng.Intn(255) - 127)
	}
	for i := range Bq {
		Bq[i] = int8(rng.Intn(255) - 127)
	}
	as := make([]float32, M)
	bs := make([]float32, N)
	for i := range as {
		as[i] = 0.01
	}
	for i := range bs {
		bs[i] = 0.01
	}
	dA, dB := NewBufferOf(d, A), NewBufferOf(d, Bq)
	dAs, dBs := NewBufferOf(d, as), NewBufferOf(d, bs)
	dC := NewBufferLenOf[float32](d, M*N)

	// THREE arms, not two (audit G-08). This used to compare the tiled kernel
	// only against the naive one nobody runs, which is why it could not see
	// M-12: tiled beats naive comfortably and still sat at ~7% of roof. "reg"
	// is the register-blocked int8 kernel the ViT path now takes.
	var p Pipeline
	var cfg LaunchConfig
	switch kind {
	case "untiled":
		p, cfg = v.GEMMW8A8, Grid1D(M*N, 256)
	case "tiled":
		p, cfg = v.GEMMW8A8Tiled, TileGrid(M, N)
	case "reg":
		p, cfg = v.GEMMW8A8Plan(M, N, K)
		if p != v.GEMMW8A8Reg {
			b.Skipf("shape %dx%dx%d does not align for the register kernel", M, N, K)
		}
	default:
		b.Fatalf("unknown kernel %q", kind)
	}
	run := func() {
		if err := q.Launch(p, cfg, Arg(dA), Arg(dAs), Arg(dB), Arg(dBs), Arg(dC),
			ArgValue(int32(M)), ArgValue(int32(N)), ArgValue(int32(K))); err != nil {
			b.Fatalf("Launch: %v", err)
		}
		if err := q.Sync(); err != nil {
			b.Fatalf("Sync: %v", err)
		}
	}
	for range 5 { // warm: JIT, caches, clocks
		run()
	}
	b.ResetTimer()
	for range b.N {
		run()
	}
	b.StopTimer()
	// 2*M*N*K flops (one multiply + one add per k)
	gflop := 2.0 * float64(M) * float64(N) * float64(K) / 1e9
	b.ReportMetric(gflop/(b.Elapsed().Seconds()/float64(b.N)), "GFLOP/s")
}

var gemmShapes = []struct {
	name    string
	M, N, K int
}{
	{"siglip_qkv/np4096", 4096, 1152, 1152},
	{"siglip_fc1/np4096", 4096, 4304, 1152},
	{"siglip_qkv/np1024", 1024, 1152, 1152},
	{"small/np256", 256, 768, 768},
}

func BenchmarkGEMMW8A8(b *testing.B) {
	for _, s := range gemmShapes {
		b.Run(fmt.Sprintf("%s/untiled", s.name), func(b *testing.B) { benchGEMM(b, "untiled", s.M, s.N, s.K) })
		b.Run(fmt.Sprintf("%s/tiled", s.name), func(b *testing.B) { benchGEMM(b, "tiled", s.M, s.N, s.K) })
		b.Run(fmt.Sprintf("%s/reg", s.name), func(b *testing.B) { benchGEMM(b, "reg", s.M, s.N, s.K) })
	}
}
