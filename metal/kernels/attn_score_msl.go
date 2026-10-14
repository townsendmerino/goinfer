package kernels

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
