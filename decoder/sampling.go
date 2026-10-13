package decoder

import "github.com/townsendmerino/goinfer/internal/sampling"

// The sampler lives in internal/sampling, a package that imports nothing from decoder, so its statistical tests are
// cached apart from decoder's and a decoder edit does not re-run them (docs/measurements/test-wait-2026-10.md §2).
// These aliases keep decoder's exported API and its call sites unchanged.
type (
	SamplingParams = sampling.SamplingParams
	Sampler        = sampling.Sampler
	SampleInfo     = sampling.SampleInfo
	TokenLogprob   = sampling.TokenLogprob
)

// NewSampler builds a sampler from params.
func NewSampler(p SamplingParams) *Sampler { return sampling.NewSampler(p) }

func argmax(logits []float32) int { return sampling.Argmax(logits) }

func softmaxStable(logits []float32, temperature float64) []float64 {
	return sampling.SoftmaxStable(logits, temperature)
}
