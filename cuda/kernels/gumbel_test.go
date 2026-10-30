//go:build cuda

package kernels

import (
	"math"
	"math/rand"
	"testing"

	gc "github.com/eitamring/gocudrv/cuda"
	"github.com/townsendmerino/goinfer/cuda/internal/launchgeom"
	"github.com/townsendmerino/goinfer/internal/sampling"
)

var adGumbel = adMod{"gumbel", readPTX("gumbel")}

// gumbelVocab is the Qwen2.5-Coder 0.5B's vocabulary, the largest size the test draws over.
const gumbelVocab = 151936

// TestGumbelDeviceAgreesWithHost is the kernel gate for the device Gumbel-max sampler: gumbel_stage1 and gumbel_stage2,
// launched as cudaResident.gumbelPick launches them, against the host reference (sampling.GumbelDraw).
//
// PRE-REGISTERED RULE. The host's noise transform is f64 and the device's f32, so the two pick different tokens only when
// the two best keys are within a few ulps:
//  1. overall agreement with the host is at least 99.99% of draws (a mismatch rate of at most 1e-4);
//  2. every mismatch is a near-tie: the host's own keys for the device's token and the host's token differ by at most
//     5e-5. A larger gap means the noise itself differs (the Philox words or the log transform), a bug, not rounding.
//
// The resident's own launch, upload and all -inf fallback are tested through gumbelPick by cuda's
// TestGumbelPick_residentWiring.
func TestGumbelDeviceAgreesWithHost(t *testing.T) {
	r, _ := newAdRig(t)
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
	for _, v := range []int{gumbelVocab, 50001, 4099, 1001, 67} {
		nb := launchgeom.GumbelBlocks(v)
		key, idx, out := r.zeros(nb), r.i32(make([]int32, nb)), r.i32([]int32{0})
		for _, kind := range []string{"normal", "peaked", "masked", "flat"} {
			l := mk(kind, v)
			dl := r.f32(l)
			// The rig's stream is non-blocking, and a copy from pageable memory can return before its DMA lands, so the
			// first launch would read a partly written row; the resident's upload is ordered with its own stream.
			if err := r.cx.Synchronize(r.bg); err != nil {
				t.Fatalf("sync after upload: %v", err)
			}
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
					dev := gumbelDraw(t, r, dl, v, float32(1/T), seed, draw, key, idx, out)
					host := sampling.GumbelDraw(l, T, seed, draw)
					total++
					if dev == host {
						continue
					}
					mismatch++
					gap := sampling.GumbelKeyAt(l, T, seed, draw, host) - sampling.GumbelKeyAt(l, T, seed, draw, dev)
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
		r.release()
	}
	rate := float64(mismatch) / float64(total)
	t.Logf("%d draws: %d mismatches (%.2e); worst host-key gap among them %.2e", total, mismatch, rate, worstGap)
	if rate > 1e-4 {
		t.Errorf("mismatch rate %.2e exceeds the pre-registered 1e-4", rate)
	}
}

// gumbelDraw runs the two Gumbel kernels over logits[:v] with gumbelPick's grid and arguments and returns the index
// stage 2 wrote. Every case here has a finite logit, so a -1 (nothing comparable) fails the test.
func gumbelDraw(t *testing.T, r *adRig, logits *gc.Buffer[float32], v int, invT float32, seed, draw uint64, key *gc.Buffer[float32], idx, out *gc.Buffer[int32]) int {
	t.Helper()
	nb := launchgeom.GumbelBlocks(v)
	r.launch(adGumbel, "gumbel_stage1", adCfg(nb, 1, 1, launchgeom.GumbelThreads, 0), gc.Arg(logits), adI(v), gc.ArgValue(invT),
		gc.ArgValue(uint32(seed)), gc.ArgValue(uint32(seed>>32)), gc.ArgValue(uint32(draw)), gc.ArgValue(uint32(draw>>32)), gc.Arg(key), gc.Arg(idx))
	r.launch(adGumbel, "gumbel_stage2", adCfg(1, 1, 1, launchgeom.GumbelThreads, 0), gc.Arg(key), gc.Arg(idx), adI(nb), gc.Arg(out))
	got := make([]int32, 1)
	if err := gc.CopyDtoH(r.bg, got, out); err != nil {
		t.Fatalf("D2H: %v", err)
	}
	if got[0] < 0 {
		t.Fatalf("v=%d seed=%x draw=%d: stage 2 found nothing comparable on a row with finite logits", v, seed, draw)
	}
	return int(got[0])
}
