package kernels

import (
	"fmt"
	"strings"
)

// AttnScoreMSL is the function every attention kernel in AllKernels and PrefillKernels passes a scaled score through
// before the softmax, so a transform of the score has one place to live. It is spliced into both sources after their
// includes.
//
// attn_score takes the score as the kernel has it (the dot product times the scale, in whichever order that kernel
// multiplies). attn_score_log2 is the same for the steel prefill kernel, whose score carries log2(e) because its
// softmax runs in base 2.
//
// attention_fa_blk does not call it: that kernel's text is pinned to the source that was graded
// (TestAttnFABlkIsTheGradedKernel), and an edit to it needs that grading re-run.
const AttnScoreMSL = `
inline float attn_score(float x) { return x; }
inline float attn_score_log2(float x) { return x; }
`

// attnScoreSoftcapMSL is AttnScoreMSL for a model with an attention-score softcap: cap·tanh(score/cap), the cap written
// in as a literal. The tanh argument is clamped because MSL's tanh overflows to NaN on a large one (glu_act's comment
// has the case); tanh is ±1 to f32 precision well inside ±15, so the clamp changes no value. In attn_score_log2 the
// score and the cap are both in base-2 units.
const attnScoreSoftcapMSL = `
inline float attn_score(float x) { const float c = %[1]s; return c * tanh(clamp(x / c, -15.0f, 15.0f)); }
inline float attn_score_log2(float x) { const float c = %[1]s * 1.4426950408889634f; return c * tanh(clamp(x / c, -15.0f, 15.0f)); }
`

// WithAttnSoftcap returns src, a kernel source that contains AttnScoreMSL, built for a model whose attention-score
// softcap is cap. A cap of 0 returns src itself, so a model without a cap compiles exactly the source it always has; a
// cap replaces the one occurrence of AttnScoreMSL with the capped functions.
//
// The softcap is a variant of the source and not a kernel argument so that a model without one runs no branch per
// score, and so that its kernels are the same text as before the softcap existed
// (docs/tasks/task-forward-architecture-2026-10.md §4.6).
func WithAttnSoftcap(src string, cap float32) (string, error) {
	if cap == 0 {
		return src, nil
	}
	if !(cap > 0) {
		return "", fmt.Errorf("attention softcap %v is not a positive number", cap)
	}
	if n := strings.Count(src, AttnScoreMSL); n != 1 {
		return "", fmt.Errorf("the kernel source has %d copies of the attention score function, want 1", n)
	}
	return strings.Replace(src, AttnScoreMSL, fmt.Sprintf(attnScoreSoftcapMSL, fmt.Sprintf("%.9ef", cap)), 1), nil
}
