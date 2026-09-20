//go:build gpu && goinfer_testhooks

package gpu

import (
	"math"
	"math/rand"
	"testing"

	"github.com/oliverbestmann/webgpu/wgpu"
	"github.com/townsendmerino/goinfer/decoder"
)

// TestGumbelDeviceAgreesWithHost is the WebGPU kernel gate for the Gumbel-max sampler (R7b). Same PRE-REGISTERED
// rule as cuda/gumbel_test.go, written before the first run:
//
//  1. Overall agreement with the host reference (decoder.gumbelDraw) >= 99.99% of draws.
//  2. EVERY mismatch is a genuine near-tie: the HOST's own keys for the device's token and the host's token differ
//     by <= 5e-5. A larger gap would mean the noise itself differs (a wrong mulhi, a bad log1p) — a bug, not rounding.
//
// WGSL specifics (gpu/gumbel.go): mulhi is hand-built, log1p is a polynomial, and WGSL leaves inf undefined, so
// masked entries are a large FINITE negative here (the device path is never used on a masked row).
func TestGumbelDeviceAgreesWithHost(t *testing.T) {
	c := newOrSkipHW(t)
	defer c.Close()

	rng := rand.New(rand.NewSource(17))
	mk := func(kind string, v int) []float32 {
		l := make([]float32, v)
		switch kind {
		case "normal":
			for i := range l {
				l[i] = float32(rng.NormFloat64() * 3)
			}
		case "peaked":
			for r, id := range rng.Perm(v) {
				l[id] = float32(9 - 2.0*math.Log(float64(r+1)))
			}
		case "masked":
			for i := range l {
				l[i] = -1e30
			}
			for k := 0; k < 12 && k < v; k++ {
				l[rng.Intn(v)] = float32(rng.NormFloat64())
			}
		case "flat":
			for i := range l {
				l[i] = 0.75
			}
		}
		return l
	}
	seeds := []uint64{0, 1, 0xFFFFFFFF, 0x123456789ABCDEF0}
	total, mismatch := 0, 0
	var worst float32
	for _, v := range []int{151936, 50001, 4099, 1001, 67} {
		buf, err := c.device.TryCreateBuffer(&wgpu.BufferDescriptor{Size: uint64(v * 4), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst})
		if err != nil {
			t.Fatalf("logits buffer: %v", err)
		}
		var rel []func()
		s, err := newGumbelState(c, buf, v, func(f func()) { rel = append(rel, f) })
		if err != nil {
			t.Fatalf("newGumbelState: %v", err)
		}
		for _, kind := range []string{"normal", "peaked", "masked", "flat"} {
			l := mk(kind, v)
			if err := c.queue.TryWriteBuffer(buf, 0, wgpu.ToBytes(l)); err != nil {
				t.Fatal(err)
			}
			for _, T := range []float64{0.3, 1.0, 2.0} {
				draws := 300
				if v > 10000 {
					draws = 50 // the sequential host reference costs ~10 ms/draw at a 152k vocab
				}
				for d := 0; d < draws; d++ {
					seed := seeds[d%len(seeds)]
					draw := uint64(d)
					if d%7 == 0 {
						draw += 1 << 33 // the high counter word
					}
					dev, err := s.sample(c, nil, v, float32(1/T), seed, draw)
					if err != nil {
						t.Fatalf("v=%d %s T=%.1f: %v", v, kind, T, err)
					}
					host := decoder.GumbelDrawForTest(l, T, seed, draw)
					total++
					if dev == host {
						continue
					}
					mismatch++
					gap := decoder.GumbelKeyForTest(l, T, seed, draw, host) - decoder.GumbelKeyForTest(l, T, seed, draw, dev)
					if gap < 0 {
						gap = -gap
					}
					if gap > worst {
						worst = gap
					}
					if gap > 5e-5 {
						t.Errorf("v=%d %s T=%.1f seed=%x draw=%d: device %d != host %d and the host's own keys differ by %.3g — not a near-tie",
							v, kind, T, seed, draw, dev, host, gap)
					}
				}
			}
		}
		for _, f := range rel {
			f()
		}
		buf.Release()
	}
	rate := float64(mismatch) / float64(total)
	t.Logf("%d draws: %d mismatches (%.2e); worst host-key gap among them %.2e", total, mismatch, rate, worst)
	if rate > 1e-4 {
		t.Errorf("mismatch rate %.2e exceeds the pre-registered 1e-4", rate)
	}
}
