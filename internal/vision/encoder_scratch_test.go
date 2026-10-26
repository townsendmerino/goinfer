package vision

import (
	"math"
	"math/rand"
	"testing"
)

// synthEncoder builds a small deterministic SigLIP encoder (random weights) so the
// CPU forward can be exercised without a checkpoint.
func synthEncoder(seed int64, quant bool) ([]float32, *Encoder) {
	rng := rand.New(rand.NewSource(seed))
	rnd := func(n int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = float32(rng.NormFloat64()) * 0.1
		}
		return s
	}
	const hidden, nH, inter, nLayers = 16, 2, 32, 3
	const patch, imgSize, chans = 4, 8, 3
	cpp := chans * patch * patch
	grid := imgSize / patch
	np := grid * grid
	e := &Encoder{
		Cfg: EncoderConfig{
			HiddenSize: hidden, NumAttentionHeads: nH, IntermediateSize: inter,
			NumHiddenLayers: nLayers, NumChannels: chans, ImageSize: imgSize,
			PatchSize: patch, LayerNormEps: 1e-6,
		},
		grid: grid, numPatches: np,
		patchW: rnd(hidden * cpp), patchB: rnd(hidden), posEmb: rnd(np * hidden),
		postLNw: rnd(hidden), postLNb: rnd(hidden),
	}
	for range nLayers {
		e.layers = append(e.layers, encLayer{
			ln1w: rnd(hidden), ln1b: rnd(hidden),
			qw: newQMat(rnd(hidden*hidden), hidden, hidden, quant), qb: rnd(hidden),
			kw: newQMat(rnd(hidden*hidden), hidden, hidden, quant), kb: rnd(hidden),
			vw: newQMat(rnd(hidden*hidden), hidden, hidden, quant), vb: rnd(hidden),
			ow: newQMat(rnd(hidden*hidden), hidden, hidden, quant), ob: rnd(hidden),
			ln2w: rnd(hidden), ln2b: rnd(hidden),
			fc1w: newQMat(rnd(inter*hidden), inter, hidden, quant), fc1b: rnd(inter),
			fc2w: newQMat(rnd(hidden*inter), hidden, inter, quant), fc2b: rnd(hidden),
		})
	}
	return rnd(chans * imgSize * imgSize), e
}

// TestEncoder_scratchNumericGuard is the numeric-regression gate for AUDIT #5/#12:
// allocating the per-layer scratch once per Forward (instead of a fresh set every
// layer) and threading one Workspace through the projections must not change the
// computation. The expected numbers were captured from the pre-#5 code; the SigLIP
// parity test (against a real checkpoint) skips in CI, so this synthetic forward is
// the guard.
//
// The assertion is a tight tolerance, not float32 bit-equality: a pure-Go matmul is
// not bit-portable across architectures (the compiler contracts a*b+c into a
// single-rounding FMA on arm64 but not identically on amd64), so exact pins captured
// on the arm64 dev box drift in the last 1–2 ULPs on amd64/windows CI. The tolerance
// is ~250× the observed cross-arch drift and ~10⁶× smaller than any real
// scratch-aliasing bug (which moves results by order 1e-2+), so it stays a real gate.
func TestEncoder_scratchNumericGuard(t *testing.T) {
	// tol combines an absolute floor with a relative term so it holds across the
	// three orders of magnitude the outputs span (~8e-4 … ~2e-1).
	const tol = 1e-6
	cases := []struct {
		name      string
		quant     bool
		wantFirst []float32
		wantSumSq float64
	}{
		{"f32", false, []float32{-0.12652746, -0.0008432368, -0.1994251, 0.014424909}, 1.136807045},
		{"int8(W8A8)", true, []float32{-0.12667489, -0.00064121914, -0.19966985, 0.014695232}, 1.136773173},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			pixels, e := synthEncoder(1234, tc.quant)
			out, err := e.Forward(pixels)
			if err != nil {
				t.Fatal(err)
			}
			var sum float64
			for _, v := range out {
				sum += float64(v) * float64(v)
			}
			for i, w := range tc.wantFirst {
				if d := math.Abs(float64(out[i] - w)); d > tol*(1+math.Abs(float64(w))) {
					t.Errorf("out[%d] = %v, want %v (Δ=%.2g — scratch/Workspace reuse changed the numerics)", i, out[i], w, d)
				}
			}
			if got := math.Sqrt(sum); math.Abs(got-tc.wantSumSq) > 1e-6 {
				t.Errorf("checksum(sumsq) = %.10g, want %.10g", got, tc.wantSumSq)
			}
		})
	}
}

// benchEncoder builds a mid-size encoder (np=256, hidden=128, 12 layers) to show
// the per-layer scratch reuse (#5).
func benchEncoder(seed int64, quant bool) ([]float32, *Encoder) {
	rng := rand.New(rand.NewSource(seed))
	rnd := func(n int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = float32(rng.NormFloat64()) * 0.1
		}
		return s
	}
	const hidden, nH, inter, nLayers = 128, 8, 512, 12
	const patch, imgSize, chans = 4, 64, 3
	cpp := chans * patch * patch
	grid := imgSize / patch
	np := grid * grid // 256
	e := &Encoder{
		Cfg: EncoderConfig{HiddenSize: hidden, NumAttentionHeads: nH, IntermediateSize: inter,
			NumHiddenLayers: nLayers, NumChannels: chans, ImageSize: imgSize, PatchSize: patch, LayerNormEps: 1e-6},
		grid: grid, numPatches: np,
		patchW: rnd(hidden * cpp), patchB: rnd(hidden), posEmb: rnd(np * hidden),
		postLNw: rnd(hidden), postLNb: rnd(hidden),
	}
	for range nLayers {
		e.layers = append(e.layers, encLayer{
			ln1w: rnd(hidden), ln1b: rnd(hidden),
			qw: newQMat(rnd(hidden*hidden), hidden, hidden, quant), qb: rnd(hidden),
			kw: newQMat(rnd(hidden*hidden), hidden, hidden, quant), kb: rnd(hidden),
			vw: newQMat(rnd(hidden*hidden), hidden, hidden, quant), vb: rnd(hidden),
			ow: newQMat(rnd(hidden*hidden), hidden, hidden, quant), ob: rnd(hidden),
			ln2w: rnd(hidden), ln2b: rnd(hidden),
			fc1w: newQMat(rnd(inter*hidden), inter, hidden, quant), fc1b: rnd(inter),
			fc2w: newQMat(rnd(hidden*inter), hidden, inter, quant), fc2b: rnd(hidden),
		})
	}
	return rnd(chans * imgSize * imgSize), e
}

func BenchmarkForward(b *testing.B) {
	pixels, e := benchEncoder(9, false)
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		_, _ = e.Forward(pixels)
	}
}
