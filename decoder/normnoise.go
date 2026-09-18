package decoder

import (
	"math"
	"math/rand/v2"
	"os"
	"strconv"
)

// GOINFER_NORM_ULP_NOISE=<seed> is a DIAGNOSTIC (default-off, one env read per load): every f32
// norm vector the loaded model carries (pre/post-attention, pre/post-MLP, per-head QK norms, the
// final norm) gets an independent −1/0/+1 ULP nudge per element, seeded by <seed>. That is noise
// of exactly f32-rounding size (~1.2e-7 relative), dense across every activation — the same
// magnitude and shape as the difference between two CORRECT implementations of one forward that
// sum in different orders (a GPU's tree reductions vs the CPU's sequential ones).
//
// What it measures: the model's OWN sensitivity to that noise. Run the CPU forward twice, once
// with this set and once without, and the logit cosine between the two runs is the floor below
// which a resident-vs-CPU cosine on that checkpoint carries no information about the kernels —
// the quantized (W4A8/W8A8) forward re-rounds activations to int8 at every projection, and a
// perturbation far below one int8 step still flips a fraction of rounding decisions, each by a
// whole step, compounding per layer. Measured on a 4-layer phi3-mini-shaped checkpoint (docs/
// tasks/task-webgpu-nogqa-decode-bug.md): CPU-vs-CPU under this noise 0.9994–0.9996, argmax flip
// at the same prompt position the WebGPU resident path flipped at — indistinguishable from the
// resident-vs-CPU gap that had been read as a kernel bug. Observe-only: unset, nothing changes.
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
