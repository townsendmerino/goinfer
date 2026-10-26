//go:build linux

package gpudev

import (
	"math/rand"
	"testing"
)

// BenchmarkCUDAAttention measures the ViT attention kernel at real tower
// shapes. There was no benchmark for it, which is why audit M-14 — one block
// per (head, query), re-streaming K for every query through an uncoalesced
// score loop — sat unmeasured.
//
// Shapes: SigLIP-so400m is nH=16 hd=72 at np=729; the np=4096 row is the
// high-resolution case the audit's 4.9 GB/layer figure is counted from.
func BenchmarkCUDAAttention(b *testing.B) {
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
		qh := randF32(rng, n, 0.5)
		kh := randF32(rng, n, 0.5)
		vh := randF32(rng, n, 0.5)
		dq, dk, dv := NewBufferOf(d, qh), NewBufferOf(d, kh), NewBufferOf(d, vh)
		out := NewBufferLenOf[float32](d, n)
		scale := float32(1.0 / 8.485)

		b.Run(sh.name+"/untiled", func(b *testing.B) {
			run := func() {
				if err := q.Launch(v.Attention, AttentionGrid(sh.np, sh.nH),
					Arg(dq), Arg(dk), Arg(dv), Arg(out),
					ArgValue(int32(sh.np)), ArgValue(int32(sh.nH)), ArgValue(int32(sh.hd)),
					ArgValue(scale)); err != nil {
					b.Fatalf("Launch: %v", err)
				}
				if err := q.Sync(); err != nil {
					b.Fatalf("Sync: %v", err)
				}
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
				cfg := AttentionTiledLaunchConfig(sh.np, sh.nH)
				run := func() {
					if err := q.Launch(v.AttentionTiled, cfg,
						Arg(dq), Arg(dk), Arg(dv), Arg(out),
						ArgValue(int32(sh.np)), ArgValue(int32(sh.nH)), ArgValue(int32(sh.hd)),
						ArgValue(scale)); err != nil {
						b.Fatalf("Launch: %v", err)
					}
					if err := q.Sync(); err != nil {
						b.Fatalf("Sync: %v", err)
					}
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
