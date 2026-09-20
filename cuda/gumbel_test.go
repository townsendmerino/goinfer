//go:build cuda && goinfer_testhooks

package cuda

import (
	"math"
	"math/rand"
	"os"
	"testing"

	gc "github.com/eitamring/gocudrv/cuda"
	"github.com/townsendmerino/goinfer/decoder"
)

// TestGumbelDeviceAgreesWithHost is the kernel gate for gumbel_stage1/2 (R7b).
//
// PRE-REGISTERED RULE (written before the first run). The host draw (decoder.gumbelDraw, f64 noise transform) is
// the reference; the device computes the same argmax with an f32 transform. Philox is integer-exact, so the two
// agree except where the best two keys are within a few f32 ulps. Therefore:
//
//  1. Overall agreement >= 99.99% of draws (mismatch rate <= 1e-4).
//  2. EVERY mismatch is a genuine near-tie: the HOST's own keys for the device's token and the host's token differ
//     by <= 5e-5. A mismatch with a larger gap would mean the noise itself differs — a bug, not rounding.
//
// Rows: the real vocab and sizes that are not multiples of 4 or of the block, normal / peaked / -inf-masked /
// flat; temperatures 0.3, 1.0, 2.0; seeds and draw indices that exercise the high words of both.
func TestGumbelDeviceAgreesWithHost(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 (loads a 0.5B model for the compiled gumbel kernels)")
	}
	if err := gc.Init(); err != nil {
		t.Skipf("cuInit: %v", err)
	}
	if _, err := gc.GetDevice(0); err != nil {
		t.Skipf("no device: %v", err)
	}
	path := modelPath("qwen2.5-coder-0.5b-instruct-q4_k_m.gguf")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no fixture at %s", path)
	}
	mc, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer mc.Close()
	rf, ok := mc.ResidentForwardForTest().(*cudaResident)
	if !ok || rf == nil {
		t.Skip("model did not go resident")
	}

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
				l[i] = float32(math.Inf(-1))
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
	worstGap := float32(0)
	for _, v := range []int{rf.vocab, 50001, 4099, 1001, 67} {
		for _, kind := range []string{"normal", "peaked", "masked", "flat"} {
			l := mk(kind, v)
			for _, T := range []float64{0.3, 1.0, 2.0} {
				draws := 400
				if v > 10000 {
					draws = 60 // the sequential host reference costs ~10 ms/draw at a 152k vocab
				}
				for d := 0; d < draws; d++ {
					seed := seeds[d%len(seeds)]
					draw := uint64(d)
					if d%7 == 0 {
						draw += 1 << 33 // the high counter word
					}
					dev, err := rf.GumbelForTest(l, T, seed, draw)
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
					if gap > worstGap {
						worstGap = gap
					}
					if gap > 5e-5 {
						t.Errorf("v=%d %s T=%.1f seed=%x draw=%d: device %d != host %d and the host's own keys differ by %.3g — not a near-tie",
							v, kind, T, seed, draw, dev, host, gap)
					}
				}
			}
		}
	}
	rate := float64(mismatch) / float64(total)
	t.Logf("%d draws: %d mismatches (%.2e); worst host-key gap among them %.2e", total, mismatch, rate, worstGap)
	if rate > 1e-4 {
		t.Errorf("mismatch rate %.2e exceeds the pre-registered 1e-4", rate)
	}
}
