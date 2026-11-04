// attnscore.cuh — the one place a text-path attention kernel turns a raw q·k dot product into the score the softmax reads.
//
// A kernel calls attn_score (a plain multiply) or attn_score_rn (an explicit round-to-nearest multiply, which the compiler cannot fuse
// into a following add) according to the multiply it has always used: the two are not interchangeable, and each returns exactly the
// value its call site computed before the call existed.
//
// -DATTN_SOFTCAP builds the capped variant (Gemma 2's attn_logit_softcapping): the three kernels whose signatures end in
// ATTN_CAP_PARAM take a trailing `float attnCap`, and attn_score(dot, scale ATTN_CAP_ARG) returns cap·tanh(dot·scale / cap), as
// decoder/attention.go:softcapScore does, before the mask and the softmax. Without the define both macros are empty and a module's
// source, and so its PTX, is what it was without this section. A kernel that does not carry ATTN_CAP_PARAM must not be built with
// the define: a lane selector keeps a capped layer off it.
#ifndef GOINFER_ATTNSCORE_CUH
#define GOINFER_ATTNSCORE_CUH
#ifdef ATTN_SOFTCAP
#define ATTN_CAP_PARAM , float attnCap
#define ATTN_CAP_ARG , attnCap
__device__ __forceinline__ float attn_score(float dot, float scale, float cap) { return cap * tanhf((dot * scale) / cap); }
#else
#define ATTN_CAP_PARAM
#define ATTN_CAP_ARG
__device__ __forceinline__ float attn_score(float dot, float scale) { return dot * scale; }
#endif
__device__ __forceinline__ float attn_score_rn(float dot, float scale) { return __fmul_rn(dot, scale); }
#endif
