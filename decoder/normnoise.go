package decoder

import (
	"math"
	"math/rand/v2"
	"os"
	"strconv"
)

// applyNormULPNoiseDiag is a diagnostic: when GOINFER_NORM_ULP_NOISE=<seed> is set (read once per load; unset changes
// nothing; an unparsable seed means 1), every f32 norm vector the model carries (pre/post-attention, pre/post-MLP, per-head
// QK norms, the final norm) gets an independent -1/0/+1 ULP nudge per element. That is noise of f32-rounding size (~1.2e-7
// relative), dense across every activation: the magnitude and shape of the difference between two correct forwards that sum
// in different orders (a GPU's tree reductions against the CPU's sequential ones).
//
// It measures the model's own sensitivity to that noise. Run the CPU forward with and without it: the logit cosine between
// the two runs is the floor below which a resident-vs-CPU cosine on that checkpoint says nothing about the kernels, because
// the quantized (W4A8/W8A8) forward re-rounds activations to int8 at every projection and a perturbation far below one int8
// step still flips rounding decisions that compound per layer. Measurement and the case it explained:
// docs/code-notes/decoder.md#applyNormULPNoiseDiag.
func applyNormULPNoiseDiag(w *Weights) {
	v := os.Getenv("GOINFER_NORM_ULP_NOISE")
	if v == "" || w == nil {
		return
	}
	seed, err := strconv.ParseUint(v, 10, 64)
	if err != nil {
		seed = 1
	}
	rng := rand.New(rand.NewPCG(seed, seed^0x9e3779b97f4a7c15))
	// Copies, never in place: a loader may hand back a norm that aliases a read-only mmap.
	nudge := func(vec []float32) []float32 {
		if len(vec) == 0 {
			return vec
		}
		out := make([]float32, len(vec))
		for i, x := range vec {
			if x == 0 || math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
				out[i] = x
				continue
			}
			bits := math.Float32bits(x)
			switch rng.IntN(3) {
			case 0:
				bits--
			case 2:
				bits++
			}
			out[i] = math.Float32frombits(bits)
		}
		return out
	}
	w.FinalNorm = nudge(w.FinalNorm)
	for i := range w.Layers {
		l := &w.Layers[i]
		l.PreAttnNorm = nudge(l.PreAttnNorm)
		l.PreMLPNorm = nudge(l.PreMLPNorm)
		l.PostAttnNorm = nudge(l.PostAttnNorm)
		l.PostMLPNorm = nudge(l.PostMLPNorm)
		l.QNorm = nudge(l.QNorm)
		l.KNorm = nudge(l.KNorm)
	}
}
