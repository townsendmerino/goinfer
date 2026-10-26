//go:build darwin

package gpudev

import (
	"math/rand"
	"testing"
)

// BenchmarkMetalAttention measures the ViT attention kernel on Metal at real tower shapes,
// mirroring BenchmarkCUDAAttention in cuda_attention_bench_test.go.
func BenchmarkMetalAttention(b *testing.B) {
	d, q, v := vitSetupM(b)
	rng := rand.New(rand.NewSource(21))

	for _, sh := range []struct {
		name       string
		np, nH, hd int
	}{
		{"so400m/np729", 729, 16, 72},
		{"so400m/np1024", 1024, 16, 72},
		{"qwen/np1024", 1024, 16, 80},
		{"hires/np2048", 2048, 16, 72},
		{"hires/np3072", 3072, 16, 72},
		{"hires/np4096", 4096, 16, 72},
	} {
		hidden := sh.nH * sh.hd
		n := sh.np * hidden
		qh := randF32M(rng, n, 0.5)
		kh := randF32M(rng, n, 0.5)
		vh := randF32M(rng, n, 0.5)
		dq, dk, dv := NewBufferOf(d, qh), NewBufferOf(d, kh), NewBufferOf(d, vh)
		out := d.NewBufferLen(n)
		scale := float32(1.0 / 8.485)

		b32np := i32b(d, sh.np)
		b32nH := i32b(d, sh.nH)
		b32hd := i32b(d, sh.hd)
		b32scale := f32b(d, scale)

		b.Run(sh.name+"/untiled", func(b *testing.B) {
			run := func() {
				run1dTG(q, v.Attention, sh.np*sh.nH*ViTBlock, ViTBlock, sh.np*4,
					dq, dk, dv, out, b32np, b32nH, b32hd, b32scale)
			}
			for range 3 {
				run()
			}
			b.ResetTimer()
			for range b.N {
				run()
			}
			b.StopTimer()
			macs := 2.0 * float64(sh.nH) * float64(sh.np) * float64(sh.np) * float64(sh.hd)
			b.ReportMetric(macs/(b.Elapsed().Seconds()/float64(b.N))/1e9, "GMAC/s")
		})

		if AttentionTiledEligible(sh.hd) {
			b.Run(sh.name+"/tiled", func(b *testing.B) {
				nGrid, tg := AttentionTiledDispatch(sh.np, sh.nH)
				run := func() {
					run1d(q, v.AttentionTiled, nGrid, tg,
						dq, dk, dv, out, b32np, b32nH, b32hd, b32scale)
				}
				for range 3 {
					run()
				}
				b.ResetTimer()
				for range b.N {
					run()
				}
				b.StopTimer()
				macs := 2.0 * float64(sh.nH) * float64(sh.np) * float64(sh.np) * float64(sh.hd)
				b.ReportMetric(macs/(b.Elapsed().Seconds()/float64(b.N))/1e9, "GMAC/s")
			})
		}
	}
}
