// attnscore.cuh — the one place a text-path attention kernel turns a raw q·k dot product into the score the softmax reads.
//
// A kernel calls attn_score (a plain multiply) or attn_score_rn (an explicit round-to-nearest multiply, which the compiler cannot fuse
// into a following add) according to the multiply it has always used: the two are not interchangeable, and each returns exactly the
// value its call site computed before the call existed. The modules built without -DATTN_SOFTCAP are byte-for-byte what they were.
#ifndef GOINFER_ATTNSCORE_CUH
#define GOINFER_ATTNSCORE_CUH
__device__ __forceinline__ float attn_score(float dot, float scale) { return dot * scale; }
__device__ __forceinline__ float attn_score_rn(float dot, float scale) { return __fmul_rn(dot, scale); }
#endif
