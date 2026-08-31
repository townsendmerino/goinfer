//go:build cuda && goinfer_testhooks

package cuda

import (
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestMellumResidentParityCUDA is the CUDA counterpart of metal/mellum_real_test.go, and it
// exists for a reason that is worth stating plainly: declaring FeatRopeMscale so gpt-oss's YaRN
// can work ALSO admits Mellum, because mellumArchitecture requires exactly {FeatMoE,
// FeatPerLayerRoPE, FeatQKNorm, FeatRopeMscale, FeatSlidingWindow} and CUDA already declared the
// other four. One flag is the entire admission.
//
// Metal hit the same coupling (G10) and resolved it by an explicit owner call, because no Mellum
// checkpoint was reachable on that machine — waiting was not an available option. On this box a
// real 4-layer weight slice IS present, so the choice there is a measurement here.
//
// WHAT THIS PROVES AND WHAT IT DOES NOT. It proves the CUDA resident forward agrees with the CPU
// forward on real Mellum weights, which is what the admission is a claim about. It does not prove
// Mellum is fast, or that a FULL 24-layer Mellum behaves like the 4-layer slice — the slice is
// what decoder/mellum_slice_test.go validated against the HF reference (argmax 417, cosine
// 1.00000000), so it is the piece with a known-good answer to compare against.
//
// The slice carries the thing FeatRopeMscale is actually about: YaRN mscale 1.2772588722239782 on
// its full-attention layer and 1.0 on the sliding ones. A backend that ignored the factor would
// still produce finite, plausible logits — which is why this is a parity gate and not a smoke test.
func TestMellumResidentParityCUDA(t *testing.T) {
	requireHeavyModel(t)
	path := "../decoder/testdata/mellum-mellum2-slice"
	if _, err := os.Stat(path + "/model.safetensors"); err != nil {
		t.Skipf("no Mellum2 slice at %s (4 GB, gitignored) — regenerate per docs/completed/queue-correctness.md G11", path)
	}
	if !decoder.ResidentBackendFeatures("cuda")[decoder.FeatRopeMscale] {
		t.Skip("cuda does not declare FeatRopeMscale — see docs/queue-correctness.md G7")
	}
	// The prompt is the one residentCosineParity's round-trip precondition is written against
	// ("capitalofFrance"), not a code prompt, and that is deliberate rather than lazy: this gate
	// compares the CUDA resident forward against the CPU forward on IDENTICAL tokens, so the
	// prompt's semantic fit to a code model changes nothing it measures. Using a different string
	// only trips the helper's hardcoded round-trip assertion — which exists because a Gemma parity
	// number was once reported on ids that decoded to gibberish, and is worth keeping.
	residentCosineParity(t, path, "The capital of France is Paris. The city is")
}
